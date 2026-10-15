"""Run the test suite in parallel: one process per test file, several files at a time.

    python tests/run_parallel.py                 # every tests/test_*.py, as many at a time as the cores and memory allow
    python tests/run_parallel.py --jobs 4 --pg   # the suite on PostgreSQL (what ``make test-pg`` runs)
    python tests/run_parallel.py test_studio test_graph

Each file runs as ``python -m unittest discover -s tests -p <file>`` (what ``make test`` ran for the whole folder), so a
test sees what it saw before. Files never share a process, a temporary folder or a database:

- every run gets its own TMPDIR, removed when it ends (the tests make folders they do not remove);
- the tests that need PostgreSQL empty their database, so each worker slot gets its own: ``<test database>_p<slot>``,
  created and migrated on first use. ``--pg`` also points the product's stores at it (``DCLAB_DATABASE_URL``), as
  ``make test-pg`` did with the one shared database. When a slot database cannot be created, the files run one at a time.

By default the files run in processes forked from one that has already imported the heavy libraries (numpy, pandas,
scikit-learn, LightGBM, XGBoost, SQLAlchemy, FastAPI): each file skips their import and shares their memory with the
others, so more files fit in the same memory and each starts at once (Python's forkserver; ``--no-fork`` runs each
file in a fresh interpreter as before). The project's own modules are imported in each file's process, after its
environment is set. A file starts only when there is memory for it (about 1 GB; other programs share the machine), so on a busy machine
fewer run at a time instead of swapping. The slowest files start first, from the durations of the last run (``tests/.timings.json`` and ``.timings-pg.json``, not committed). A failing
file's output is printed whole; the exit code is 1 when any file failed. ``--jobs 1`` runs the files one at a time.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TIMINGS = HERE / ".timings.json"  # .timings-pg.json for the PostgreSQL run
# Several files run at once, so each gets one thread: LightGBM, XGBoost and BLAS otherwise each start a thread per core
# in every worker, and the machine spends its time switching between them (the suite took 7x its serial CPU time).
ONE_THREAD = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                                 "LOKY_MAX_CPU_COUNT")}
RAN = re.compile(r"^Ran (\d+) tests? in", re.M)
TALLY = re.compile(r"(failures|errors|skipped|expected failures|unexpected successes)=(\d+)")


NEED_GB = 0.8  # free memory a new file needs before it starts (a worker peaks near 1 GB; the import is most of it)
FORKED_NEED_GB = 0.45  # forked from the preloaded server, it shares the libraries' memory: about half of it is its own
# (measured on the 8 GB M1, 2026-10-14: the whole suite 479 s in fresh interpreters, 328 s forked; starting files until
# only 10% of the memory was left made it 377 s: the system compresses memory before it swaps, and that costs more)
PRELOAD = ("numpy", "pandas", "scipy", "scipy.stats", "sklearn", "sklearn.ensemble", "sklearn.linear_model", "sklearn.model_selection",
           "sklearn.metrics", "sklearn.preprocessing", "lightgbm", "xgboost", "pydantic", "sqlalchemy", "fastapi", "starlette",
           "fastapi.testclient", "httpx", "psutil", "pyarrow", "pyarrow.parquet", "threadpoolctl")
FORK = {"on": False, "context": None}
SPLIT_SECONDS = 150  # a file that took longer last time runs as several units at once (forked mode): its classes, or its tests
# (only the slowest: a split repeats each unit's setup, and the suite is bound by the cores: 40 s split 87 files, 274 s against 248)
RAMP_SECONDS = 6  # a file started this recently has not taken its memory yet: count it as taken
_START = threading.Lock()
_STARTED: list[float] = []
_RUNNING = [0]
_WAITED = [0]  # files that waited for memory before they started
_KILLED = [0]  # units the system ended (a signal) and that ran again


def available_gb() -> float | None:
    """Memory the system can give a new process now, or None when this platform does not say."""
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except ImportError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def pressure() -> int | None:
    """macOS's memory pressure: 1 normal (green), 2 warning, 4 critical; None elsewhere. Forked files are small (140 to 200
    MB each, measured 2026-10-14), so they start while the system says green or yellow, even when little memory is free:
    it compresses and swaps (the owner allows up to 6 GB of swap); only red waits."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True, timeout=2).stdout
        return int(out.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def wait_for_memory() -> None:
    """Start a file only when the memory holds it: other programs share the machine, so the number of files that fit
    changes during a run (a run that swaps took 538 s instead of 156 s). One file always runs, whatever is free."""
    with _START:
        waited = False
        while True:
            now = time.monotonic()
            recent = sum(1 for t in _STARTED if now - t < RAMP_SECONDS)
            free = available_gb()
            need = FORKED_NEED_GB if FORK["on"] else NEED_GB
            level = pressure() if FORK["on"] else None
            if level is not None and (_RUNNING[0] == 0 or level < 4):  # green or yellow: start; only red waits (the owner's rule)
                break
            if level is None and (_RUNNING[0] == 0 or free is None or free - recent * need >= need):
                break
            waited = True
            time.sleep(0.5)
        _WAITED[0] += waited
        _STARTED.append(time.monotonic())
        _RUNNING[0] += 1


def default_jobs() -> int:
    """As many files at a time as the cores allow and the memory holds: a worker that loads pandas, scikit-learn and
    LightGBM peaks near 1 GB, and a machine that swaps is slower than one that runs fewer files (on an 8 GB M1: 6 jobs
    98 s, 8 jobs 268 s)."""
    cores = os.cpu_count() or 4
    try:
        memory_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return max(1, cores // 2)
    return max(1, min(cores, int(memory_gb / 1.3)))


def files(names: list[str]) -> list[str]:
    found = sorted(p.name for p in HERE.glob("test_*.py"))
    if not names:
        return found
    wanted = [n if n.endswith(".py") else f"{n}.py" for n in names]
    unknown = [n for n in wanted if n not in found]
    if unknown:
        raise SystemExit(f"no such test file: {', '.join(unknown)}")
    return wanted


def base_url() -> str:
    return (os.environ.get("DCLAB_TEST_DATABASE_URL") or os.environ.get("DCLAB_DATABASE_URL")
            or f"postgresql+psycopg://{getpass.getuser()}@/dclab_test")


def slot_databases(jobs: int, required: bool) -> list[str] | None:
    """One test database per worker slot, next to the base one; None when they cannot be made (no server, no right to
    create a database). ``required`` (the --pg run) makes that an error instead of a quiet fallback."""
    try:
        import sqlalchemy as sa
        from sqlalchemy.engine import make_url

        sys.path.insert(0, str(ROOT))
        from dclab_rnd.storage import db

        base = make_url(base_url())
        if db.reachable(base.render_as_string(hide_password=False)) is not None:
            raise RuntimeError(f"the test database {base.database} does not answer")
        urls = []
        admin = sa.create_engine(base, isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            have = {r[0] for r in c.execute(sa.text("select datname from pg_database"))}
            for slot in range(1, jobs + 1):
                name = f"{base.database}_p{slot}"
                if name not in have:
                    c.execute(sa.text(f'create database "{name}"'))
                # a commit need not wait for the disk in a database the tests empty anyway: much faster, and safe here
                c.execute(sa.text(f'alter database "{name}" set synchronous_commit = off'))
                urls.append(base.set(database=name).render_as_string(hide_password=False))
        admin.dispose()
        for url in urls:
            db.upgrade(url)
        return urls
    except Exception as exc:  # noqa: BLE001 — the files can still run, one database at a time
        if required:
            raise SystemExit(f"cannot prepare a PostgreSQL database per worker: {exc}") from exc
        return None


OWNER_MODEL_CHOICE = ("DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL")  # set empty, so .env (read without override) cannot set them


def run_one(name: str, slots: Queue, pg: bool, verbose: bool, threads_one: bool = True) -> dict:
    slot = slots.get()
    tmp = tempfile.mkdtemp(prefix=f"dclab-{name.split('::')[0][5:-3]}-")
    env = {**ONE_THREAD, **os.environ, "TMPDIR": tmp} if threads_one else {**os.environ, "TMPDIR": tmp}
    env.setdefault("DCLAB_DECISION_SHADOW", "0")  # 22.4: the fitted classifiers live on some machines only; a test turns them on itself
    env.update({k: "" for k in OWNER_MODEL_CHOICE})  # the owner's .env picks a provider; a test sets the model settings it reads itself
    if slot is not None:
        env["DCLAB_TEST_DATABASE_URL"] = slot
        if pg:
            env["DCLAB_DATABASE_URL"] = slot
    wait_for_memory()
    started = time.perf_counter()
    try:
        if FORK["on"]:
            code, output = _forked(name, env, tmp, verbose)
        else:
            done = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", name] + (["-v"] if verbose else []),
                                  cwd=ROOT, env=env, capture_output=True, text=True)
            code, output = done.returncode, done.stdout + done.stderr
    finally:
        with _START:
            _RUNNING[0] -= 1
        shutil.rmtree(tmp, ignore_errors=True)
        slots.put(slot)
    ran = RAN.search(output)
    tally = {k: int(v) for k, v in TALLY.findall(output.rsplit("\nRan ", 1)[-1])}
    return {"file": name, "ok": code == 0, "tests": int(ran.group(1)) if ran else 0, "seconds": time.perf_counter() - started,
            "failed": tally.get("failures", 0) + tally.get("errors", 0), "skipped": tally.get("skipped", 0), "output": output}


def units_of(name: str) -> list[str]:
    """A slow file's units: each test of a class without class- or module-level setup, each class with one (read from
    the file's source: nothing is imported here)."""
    import ast

    tree = ast.parse((HERE / name).read_text(encoding="utf-8"))
    if any(isinstance(n, ast.FunctionDef) and n.name in ("setUpModule", "tearDownModule") for n in tree.body):
        return [name]
    out = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not node.bases:  # a class with no base is a mixin: its tests run in the classes that inherit it
            continue
        methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        tests = [m for m in methods if m.startswith("test")]
        if not tests:
            continue
        if {"setUpClass", "tearDownClass"} & set(methods) or len(tests) == 1:
            out.append(f"{name}::{node.name}")
        else:
            out.extend(f"{name}::{node.name}.{t}" for t in tests)
    return out or [name]


def _child(name: str, env: dict, out_path: str, verbose: bool) -> None:
    """One test file in a process forked from the preloaded server: its own environment, temporary folder and output,
    the project's modules imported fresh; exits 0 when every test passed."""
    code = 1
    try:
        fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        sys.stdout = open(1, "w", buffering=1, closefd=False, encoding="utf-8", errors="replace")
        sys.stderr = open(2, "w", buffering=1, closefd=False, encoding="utf-8", errors="replace")
        os.environ.clear()
        os.environ.update(env)
        tempfile.tempdir = None  # gettempdir() is cached: the file's own TMPDIR from here on
        os.chdir(ROOT)
        sys.path[:0] = [str(HERE), str(ROOT)]  # as `python -m unittest discover -s tests` from the root
        import unittest

        file, _, selector = name.partition("::")
        if selector:  # one class or one test of a split file
            suite = unittest.defaultTestLoader.loadTestsFromName(f"{file[:-3]}.{selector}")
        else:
            suite = unittest.defaultTestLoader.discover(str(HERE), pattern=file, top_level_dir=str(HERE))
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2 if verbose else 1).run(suite)
        code = 0 if result.wasSuccessful() else 1
        Path(out_path + ".result").write_text(str(code), encoding="utf-8")  # the verdict, whatever the exit after it
    except BaseException:  # noqa: BLE001 — an import that fails is the file's failure, printed like any other
        import traceback

        traceback.print_exc()
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(code)


def _forked(name: str, env: dict, tmp: str, verbose: bool, again: bool = True) -> tuple[int, str]:
    """One unit in a forked process. A process the system ended (a signal: memory pressure lets macOS kill one) did not
    fail a test: it runs once more, and the output says so."""
    out = os.path.join(tmp, ".output")
    process = FORK["context"].Process(target=_child, args=(name, env, out, verbose))
    process.start()
    process.join()
    try:
        output = Path(out).read_text(encoding="utf-8", errors="replace")
    except OSError:
        output = ""
    code = process.exitcode
    try:  # the tests' own verdict, written before the process ended: an exit after it (a library's) is not a failure
        verdict = int(Path(out + ".result").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        verdict = None
    if verdict is not None:
        if code != 0:
            output += f"\n(the tests {'passed' if verdict == 0 else 'failed'}; the process then ended with code {code})"
        return verdict, output
    if code is not None and code < 0 and again:
        _KILLED[0] += 1
        retried, more = _forked(name, env, tmp, verbose, again=False)
        return retried, f"(ended by signal {-code}; ran again)\n" + more
    if code != 0:
        output += f"\n(the process ended with code {code})"
    return (0 if code == 0 else 1), output


def start_forkserver() -> bool:
    """Start the server every file is forked from, with the libraries imported once (single-threaded, as the files run)."""
    import multiprocessing

    if sys.platform == "win32" or "forkserver" not in multiprocessing.get_all_start_methods():
        return False
    os.environ.update(ONE_THREAD)  # the server's libraries start with one thread, as each file's must
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")  # macOS: a forked child may use what its parent loaded
    context = multiprocessing.get_context("forkserver")
    importable = []
    for module in PRELOAD:
        try:
            __import__(module)  # only what this environment has; a missing extra is imported (or skipped) by its tests
            importable.append(module)
        except Exception:  # noqa: BLE001
            continue
    context.set_forkserver_preload(importable)
    FORK.update(on=True, context=context)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tests/run_parallel.py", description=__doc__.split("\n")[0])
    parser.add_argument("names", nargs="*", help="test files to run (default: every tests/test_*.py)")
    parser.add_argument("--jobs", "-j", type=int, default=default_jobs(), help="files at a time (default: what the cores and the memory allow)")
    parser.add_argument("--pg", action="store_true", help="run the product's stores on PostgreSQL, one database per worker")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every file's output, not only the failing ones")
    parser.add_argument("--no-fork", action="store_true", help="each file in a fresh interpreter (slower, uses more memory)")
    args = parser.parse_args(argv)
    forking = not args.no_fork and start_forkserver()
    if forking and args.jobs == default_jobs():
        args.jobs = (os.cpu_count() or args.jobs) + 2  # shared libraries: the cores are the limit; two more fill a database's waits

    names = files(args.names)
    jobs = max(1, min(args.jobs, len(names)))
    urls = slot_databases(jobs, required=args.pg)
    if urls is None:
        if jobs > 1:
            print("note: no PostgreSQL database per worker; the files that empty the test database could collide, so they run one at a time")
        jobs, urls = 1, [None]
    slots: Queue = Queue()
    for url in urls[:jobs]:
        slots.put(url)

    timings = TIMINGS.with_name(".timings-pg.json") if args.pg else TIMINGS
    try:
        last = json.loads(timings.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        last = {}
    if forking:  # a slow file runs as units at once, so it no longer sets the run's length alone
        names = [u for n in names for u in (units_of(n) if float(last.get(n, 0)) > SPLIT_SECONDS else [n])]
        share = {u: float(last.get(u.split("::")[0], 60)) / max(1, sum(1 for v in names if v.split("::")[0] == u.split("::")[0])) for u in names}
        names.sort(key=lambda n: -share[n])
    else:
        names.sort(key=lambda n: -float(last.get(n, 60)))  # the slowest first; a new file counts as slow

    count = len({n.split("::")[0] for n in names})
    print(f"{len(names)} units of {count} test files, {jobs} at a time{' on PostgreSQL' if args.pg else ''}{', forked from preloaded libraries' if forking else ''}", flush=True)
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_one, n, slots, args.pg, args.verbose, jobs > 1) for n in names]
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            mark = "ok  " if r["ok"] else "FAIL"
            print(f"  {mark} {r['file']:34s} {r['tests']:4d} tests {r['seconds']:6.1f}s" + (f"  ({r['skipped']} skipped)" if r["skipped"] else ""), flush=True)
            if args.verbose or not r["ok"]:
                print(r["output"], flush=True)
    wall = time.perf_counter() - started

    by_file: dict[str, float] = {}
    for r in results:  # a split file's time is its units' sum: what it would take alone
        by_file[r["file"].split("::")[0]] = by_file.get(r["file"].split("::")[0], 0.0) + r["seconds"]
    if len(by_file) == len(files([])):  # only a whole run updates the timings
        try:
            timings.write_text(json.dumps({k: round(v, 1) for k, v in by_file.items()}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        except OSError:
            pass
    failed = [r for r in results if not r["ok"]]
    total = sum(r["tests"] for r in results)
    skipped = sum(r["skipped"] for r in results)
    print(f"\nRan {total} tests in {len(by_file)} files ({len(results)} units) in {wall:.1f}s (one at a time: {sum(r['seconds'] for r in results):.1f}s)"
          + (f"; {_WAITED[0]} files waited for free memory" if _WAITED[0] else "")
          + (f"; {_KILLED[0]} ended by the system and ran again" if _KILLED[0] else ""))
    if failed:
        print(f"FAILED ({sum(r['failed'] for r in failed)} failing tests in {len(failed)} files: {', '.join(r['file'] for r in failed)})")
        return 1
    print(f"OK (skipped={skipped})" if skipped else "OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
