"""Backups and restores (package 12.5): the database and the file storage, restorable on a clean machine.

    python -m dclab_rnd.storage backup --to backups              # make backup
    python -m dclab_rnd.storage restore backups/<stamp> --home DIR   # make restore FROM=backups/<stamp> HOME_DIR=DIR

A backup is a folder: ``database.dump`` (``pg_dump``, custom format, without owners or grants, so it restores under
another role), ``files.tar.gz`` (the workspace folder: every project's, draft's and workspace's files), and
``manifest.json`` with the schema revision, the rows in every table, the files, and the SHA-256 of both archives.
A restore goes only into an empty database and an empty folder (it never writes over data), checks both archives
against the manifest first, and compares the rows and files with it after. The rows and the dump come from one
snapshot; the files are archived just after it (stop the app and the workers for an exact pair). Linked folders are
followed and linked files archived as their content. A backup that fails half way is removed. The password travels in ``PGPASSWORD``,
never on a command line. With files in a bucket (``DCLAB_FILES_URL``), the bucket's own versioning keeps them:
the archive then holds the workspace folder's local copies and records.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SKIP = (".ready-", ".jobs.", ".dclab_files.", ".retiring-")  # probes and temporary files, never part of a workspace


class BackupError(RuntimeError):
    pass


def tool(name: str) -> str | None:
    """pg_dump or pg_restore: from DCLAB_PG_BIN (a folder) when set, else from PATH. Its major version must be at least the server's."""
    folder = os.environ.get("DCLAB_PG_BIN", "").strip()
    if folder:
        found = Path(folder) / name
        return str(found) if found.is_file() else None
    import glob

    # PATH first, then where Homebrew and Debian put versioned client tools (not on PATH by default): the newest wins
    places = sorted(glob.glob(f"/opt/homebrew/opt/postgresql@*/bin/{name}") + glob.glob(f"/usr/local/opt/postgresql@*/bin/{name}")
                    + glob.glob(f"/usr/lib/postgresql/*/bin/{name}"), key=lambda p: int("".join(c for c in p.split("postgresql")[1].split("/")[0] if c.isdigit()) or 0))
    return shutil.which(name) or (places[-1] if places else None)


def _libpq(url: str) -> dict[str, str]:
    """The connection as libpq's environment (PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE): never on a command line."""
    from sqlalchemy.engine import make_url

    u = make_url(url)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PG")}  # only the URL decides where it goes
    host = u.host or u.query.get("host")  # a socket folder and a port can also come in the query (?host=/tmp&port=5433)
    pairs = [("PGHOST", host), ("PGPORT", u.port or u.query.get("port")), ("PGUSER", u.username), ("PGPASSWORD", u.password), ("PGDATABASE", u.database)]
    # the URL's TLS and timeout options too: a dump carries the whole database, never over a weaker connection than the app's
    pairs += [(variable, u.query.get(option)) for option, variable in (("sslmode", "PGSSLMODE"), ("sslrootcert", "PGSSLROOTCERT"),
              ("sslcert", "PGSSLCERT"), ("sslkey", "PGSSLKEY"), ("sslcrl", "PGSSLCRL"), ("connect_timeout", "PGCONNECT_TIMEOUT"),
              ("target_session_attrs", "PGTARGETSESSIONATTRS"))]
    for key, value in pairs:
        if value:
            env[key] = str(value if not isinstance(value, tuple) else value[0])
    return env


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _counts(url: str) -> dict[str, int]:
    import sqlalchemy as sa

    from . import db
    from .models import TABLES

    with db.engine(url).connect() as c:
        return {t: int(c.execute(sa.text(f'select count(*) from "{t}"')).scalar()) for t in TABLES}


def _files(home: Path) -> list[str]:
    """Every file of the workspace, through linked folders too (a projects folder linked to another disk is still the workspace's)."""
    out = []
    for root, _, names in os.walk(home, followlinks=True):
        for name in names:
            path = Path(root) / name
            if not name.startswith(SKIP) and path.is_file():
                out.append(path.relative_to(home).as_posix())
    return sorted(out)


def _run(command: list[str], env: dict[str, str]) -> None:
    done = subprocess.run(command, env=env, capture_output=True, text=True)
    if done.returncode != 0:  # the tool's own words name a table or an option, never the password (it is in the environment)
        raise BackupError(f"{Path(command[0]).name} failed: {(done.stderr or done.stdout).strip().splitlines()[-1][:300] if (done.stderr or done.stdout).strip() else 'no output'}")


def backup(url: str, home: Path, out: Path) -> dict[str, Any]:
    """Write a backup of the database at ``url`` and the workspace folder ``home`` into a new folder under ``out``."""
    from . import db

    dump = tool("pg_dump")
    if dump is None:
        raise BackupError("pg_dump was not found: install PostgreSQL's client tools, or set DCLAB_PG_BIN to their folder")
    home = Path(home)
    if not home.is_dir():
        raise BackupError(f"the workspace folder {home} does not exist (DCLAB_AGENT_HOME, or --home)")
    if Path(out).resolve().is_relative_to(home.resolve()):
        raise BackupError("the backup folder is inside the workspace folder: it would archive itself; choose another (TO=…)")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(out) / stamp
    folder.mkdir(parents=True, exist_ok=False)
    try:
        return _backup(url, home, folder, dump)
    except BaseException as exc:  # a half-made backup is not left looking like one
        shutil.rmtree(folder, ignore_errors=True)
        if isinstance(exc, OSError):
            raise BackupError(f"the workspace files could not be archived ({type(exc).__name__}); nothing was kept") from None
        raise


def _backup(url: str, home: Path, folder: Path, dump: str) -> dict[str, Any]:
    from . import db

    env = _libpq(url)
    import sqlalchemy as sa

    from .models import TABLES

    # one snapshot for the dump and the counts: a row written meanwhile (an agent's event) is in neither
    with db.engine(url).connect().execution_options(isolation_level="REPEATABLE READ") as c, c.begin():
        snapshot = c.execute(sa.text("select pg_export_snapshot()")).scalar()
        rows = {t: int(c.execute(sa.text(f'select count(*) from "{t}"')).scalar()) for t in TABLES}
        schema = c.execute(sa.text("select version_num from alembic_version")).scalar()
        _run([dump, "--format=custom", "--no-owner", "--no-privileges", f"--snapshot={snapshot}", "--file", str(folder / "database.dump")], env)
    # The files after the dump: a file written meanwhile can be archived without its row, and a project deleted meanwhile
    # keeps its row without its files (``vanished`` names them). Stop the app and the workers for an exact pair.
    archived, vanished, size = [], [], 0
    with tarfile.open(folder / "files.tar.gz", "w:gz", dereference=True) as tar:  # a linked file is archived as its content
        for name in _files(home):
            try:
                length = (home / name).stat().st_size
                tar.add(home / name, arcname=name, recursive=False)
            except FileNotFoundError:
                vanished.append(name)
                continue
            archived.append(name)
            size += length
    manifest = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "schema": schema, "rows": rows,
                "files": len(archived), "bytes": size, "vanished": vanished,
                "sha256": {"database.dump": _sha256(folder / "database.dump"), "files.tar.gz": _sha256(folder / "files.tar.gz")}}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return {**manifest, "folder": str(folder)}


def restore(folder: Path, url: str, home: Path) -> dict[str, Any]:
    """Restore a backup into an empty database and an empty folder; refuses anything else. Returns what was compared."""
    pg_restore = tool("pg_restore")
    if pg_restore is None:
        raise BackupError("pg_restore was not found: install PostgreSQL's client tools, or set DCLAB_PG_BIN to their folder")
    folder, home = Path(folder), Path(home)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        if _sha256(folder / name) != expected:
            raise BackupError(f"{name} does not match its manifest: the backup is damaged; nothing was restored")
    import sqlalchemy as sa

    from . import db

    with db.engine(url).connect() as c:
        tables = c.execute(sa.text("select count(*) from information_schema.tables where table_schema = 'public'")).scalar()
    if tables:
        raise BackupError("the target database is not empty: a restore never writes over data (create a new database)")
    if home.exists() and any(home.iterdir()):
        raise BackupError("the target folder is not empty: a restore never writes over files (name a new folder)")
    env = _libpq(url)
    _run([pg_restore, "--no-owner", "--no-privileges", "--exit-on-error", "--dbname", env.get("PGDATABASE", ""), str(folder / "database.dump")], env)
    home.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(folder / "files.tar.gz", "r:gz") as tar:
            tar.extractall(home, filter="data")  # the "data" filter refuses absolute paths, links out of the folder and devices
    except (OSError, tarfile.TarError) as exc:
        raise BackupError(f"the database was restored but the files were not ({type(exc).__name__}): drop that database and empty "
                          "the folder before trying again") from None
    db.engine(url).dispose()
    rows, files = _counts(url), _files(home)
    differences = [f"{t}: {manifest['rows'][t]} in the backup, {rows.get(t)} restored" for t in manifest["rows"] if rows.get(t) != manifest["rows"][t]]
    if len(files) != manifest["files"]:
        differences.append(f"files: {manifest['files']} in the backup, {len(files)} restored")
    return {"rows": rows, "files": len(files), "differences": differences, "schema": db.current(url)}
