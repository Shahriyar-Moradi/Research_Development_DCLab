"""Local development with one command (package 12.0): ``make dev``, ``make db-reset``, ``make test-db``.

    make dev                   # PostgreSQL checked, dclab_dev created and migrated, the API and the job worker with reload, the frontend rebuilt on change
    make db-reset CONFIRM=yes  # a clean dclab_dev: dropped, created, migrated (asks for CONFIRM: it deletes the database)
    make test-db               # the database the tests use (dclab_test), created and migrated

No containers: the developer's own PostgreSQL (``brew services start postgresql@15`` on a Mac). The database is
``DCLAB_DATABASE_URL`` (from the shell or ``.env``, as the server reads it), else ``postgresql+psycopg://$USER@/dclab_dev``.
The database must be named ``dclab_<letters, digits, _>`` and live on this machine: a URL naming another database, or
another host (unless ``--remote``), is refused before anything is created, migrated or dropped.
The server reached must also answer from a socket or loopback. A tunnel that forwards a local port to another
server passes both checks (the far server sees loopback): there only the ``dclab_*`` name and ``CONFIRM=yes`` stand.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WEB_SRC = ROOT / "dclab_rnd" / "agentic" / "web" / "src"
START_HINT = ("Start one: brew services start postgresql@15 (macOS), or sudo systemctl start postgresql (Linux). "
              "Then run this again. Without a database, `make notebook` runs the app on files.")


class DevError(SystemExit):
    pass


def default_url(name: str) -> str:
    return f"postgresql+psycopg://{getpass.getuser()}@/{name}"


def database_name(url: str) -> str:
    from sqlalchemy.engine import make_url

    return make_url(url).database or ""


LOCAL = ("", "localhost", "127.0.0.1", "::1")


def _values(value) -> list[str]:
    """A query value as a list of strings (libpq takes a comma-separated host list; SQLAlchemy may give a tuple)."""
    items = value if isinstance(value, (tuple, list)) else [value] if value else []
    return [part.strip() for item in items for part in str(item).split(",")]


def check_name(url: str, remote: bool = False) -> str:
    """The database's name, when this command may create, migrate or drop it: named dclab_*, and on this machine.
    Every place libpq takes a host from is read: the URL, its ``host``/``hostaddr`` query (which win over the URL), and,
    when the URL names none, ``PGHOST``/``PGHOSTADDR``; a service file (``service``, ``PGSERVICE``) is refused."""
    from sqlalchemy.engine import make_url

    parsed = make_url(url)
    name = parsed.database or ""
    if not re.fullmatch(r"dclab_[A-Za-z0-9_]+", name):
        raise DevError(f"refusing to touch the database {name!r}: this command only creates, migrates or drops databases named dclab_*")
    if remote:
        return name
    hosts = _values(parsed.query.get("host")) + _values(parsed.query.get("hostaddr")) + _values(parsed.host)
    if not hosts:
        hosts = _values(os.environ.get("PGHOST")) + _values(os.environ.get("PGHOSTADDR"))
    if parsed.query.get("service") or (not parsed.host and os.environ.get("PGSERVICE")):
        raise DevError(f"refusing to touch {name!r} through a service file: name the host in the URL (pass --remote to mean another host)")
    for host in hosts:
        if host not in LOCAL and not host.startswith("/"):
            raise DevError(f"refusing to touch {name!r} on {host}: this command works on this machine's PostgreSQL (pass --remote to mean it)")
    return name


def check_server(connection, name: str) -> None:
    """The server this connection reached is this machine (a unix socket, or loopback), whatever the URL said."""
    import sqlalchemy as sa

    address = connection.execute(sa.text("select host(inet_server_addr())")).scalar()
    if address not in (None, "127.0.0.1", "::1"):
        raise DevError(f"refusing to touch {name!r}: the server answered from {address}, not this machine (pass --remote to mean it)")


def server_reachable(url: str) -> str | None:
    """None when the PostgreSQL server answers (on its maintenance database), else a plain reason."""
    import sqlalchemy as sa
    from sqlalchemy.engine import make_url

    try:
        engine = sa.create_engine(make_url(url).set(database="postgres"))
        with engine.connect():
            pass
        engine.dispose()
        return None
    except Exception as exc:  # noqa: BLE001 — no server, no driver, no role: the reason, never the URL
        return f"{type(exc).__name__}: the PostgreSQL server did not answer"


def ensure_database(url: str, reset: bool = False, remote: bool = False) -> str:
    """Create the database when it is missing (drop it first with ``reset``), then run the migrations. Returns what was done."""
    import sqlalchemy as sa
    from sqlalchemy.engine import make_url

    from dclab_rnd.storage import db

    name = check_name(url, remote)
    problem = server_reachable(url)
    if problem:
        raise DevError(f"{problem}. {START_HINT}")
    db.engine(url).dispose()  # this process's own connections first: a drop waits for none
    admin = sa.create_engine(make_url(url).set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        if not remote:
            check_server(c, name)  # the host the driver really reached: a query string, hostaddr or PG* variable cannot bypass it
        exists = c.execute(sa.text("select 1 from pg_database where datname = :n"), {"n": name}).scalar() is not None
        if reset and exists:
            c.execute(sa.text(f'drop database "{name}" with (force)'))
            exists = False
        if not exists:
            c.execute(sa.text(f'create database "{name}"'))
    admin.dispose()
    before = db.current(url)
    db.upgrade(url)
    after = db.current(url)
    return ("reset and " if reset else "created and " if before is None else "") + (f"migrated to {after}" if before != after else f"already at {after}")


def changed_since(folder: Path, last: float) -> float | None:
    """The newest modification time under ``folder`` when it is newer than ``last``, else None. A file that vanishes
    while it is read (an editor's temporary file) is skipped."""
    newest = 0.0
    for path in folder.rglob("*"):
        try:
            if path.is_file():
                newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest if newest > last else None


def watch_frontend(build, stop: threading.Event, every: float = 1.0, folder: Path = WEB_SRC) -> None:
    """Rebuild the frontend whenever a file under the web sources changes (polling: no extra dependency)."""
    try:
        last = changed_since(folder, 0.0) or 0.0
    except Exception:  # noqa: BLE001 — the first scan failed: start from now
        last = time.time()
    while not stop.wait(every):
        try:  # the watcher never dies: a failed scan or build is reported and the next change is built again
            newer = changed_since(folder, last)
            if newer is None:
                continue
            last = newer
            build()
        except Exception as exc:  # noqa: BLE001 — a broken source file is reported; the server keeps running
            print(f"frontend build failed: {type(exc).__name__}: {exc}", flush=True)


def build_frontend() -> None:
    done = subprocess.run([sys.executable, "-m", "dclab_rnd.agentic.web.build"], cwd=ROOT, capture_output=True, text=True)
    print(("frontend rebuilt: " + done.stdout.strip().splitlines()[-1]) if done.returncode == 0 and done.stdout.strip()
          else f"frontend build failed:\n{done.stdout}{done.stderr}", flush=True)


def worker_command() -> list[str]:
    """The job worker, restarted when its code changes when ``watchfiles`` is installed; without it, run once."""
    import importlib.util

    if importlib.util.find_spec("watchfiles") is None:
        print("watchfiles is not installed: the job worker runs without reload (restart make dev after changing its code)", flush=True)
        return [sys.executable, "-m", "dclab_rnd.worker"]
    return [sys.executable, "-m", "watchfiles", "--filter", "python", f"{sys.executable} -m dclab_rnd.worker", "dclab_rnd"]


def watch_worker(worker, stop: threading.Event) -> None:
    """Say so loudly when the worker ends while the API still runs: queued jobs would wait for ever."""
    code = worker.wait()
    if not stop.is_set():
        print(f"\nTHE JOB WORKER STOPPED (exit {code}): stage runs, data pipelines and intern turns stay queued. "
              "Restart make dev, or run python -m dclab_rnd.worker in another terminal.", file=sys.stderr, flush=True)


def serve(url: str, port: int, remote: bool = False) -> int:
    # The worker runs on its own (package 10.3), as in production: the API only queues jobs.
    env = {**os.environ, "DCLAB_DATABASE_URL": url, "DCLAB_WORKER": "external"}
    print(f"database {database_name(url)}: {ensure_database(url, remote=remote)}", flush=True)
    build_frontend()
    stop = threading.Event()
    watcher = threading.Thread(target=watch_frontend, args=(build_frontend, stop), daemon=True)
    watcher.start()
    command = [sys.executable, "-m", "uvicorn", "dclab_rnd.agentic.server:app", "--host", "127.0.0.1", "--port", str(port),
               "--reload", "--reload-dir", "dclab_rnd", "--workers", "1"]
    print(f"API on http://127.0.0.1:{port} and a job worker (both reload on code changes; the frontend rebuilds when {WEB_SRC.relative_to(ROOT)} changes). Ctrl-C stops.", flush=True)
    worker = subprocess.Popen(worker_command(), cwd=ROOT, env=env)
    threading.Thread(target=watch_worker, args=(worker, stop), daemon=True).start()
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        return process.wait()
    finally:
        stop.set()
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/dev.py", description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["serve", "db-reset", "test-db"])
    parser.add_argument("--yes", action="store_true", help="db-reset: yes, delete the development database")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--remote", action="store_true", help="allow a database on another host (it is never the default)")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)  # the server reads .env the same way: both use the same database
    if args.port is None:
        try:
            args.port = int(os.environ.get("DCLAB_PORT") or 8765)
        except ValueError:
            args.port = 8765
    try:
        if args.command == "test-db":
            url = os.environ.get("DCLAB_TEST_DATABASE_URL") or default_url("dclab_test")
            print(f"database {database_name(url)}: {ensure_database(url, remote=args.remote)} (the tests empty it; each parallel worker adds {database_name(url)}_p<N>)")
            return 0
        url = os.environ.get("DCLAB_DATABASE_URL") or default_url("dclab_dev")
        if args.command == "db-reset":
            if not args.yes:
                raise DevError(f"this deletes every project, draft and session in {database_name(url)}: run make db-reset CONFIRM=yes")
            print(f"database {database_name(url)}: {ensure_database(url, reset=True, remote=args.remote)}")
            return 0
        return serve(url, args.port, args.remote)
    except DevError as exc:
        print(exc.code if isinstance(exc.code, str) else exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
