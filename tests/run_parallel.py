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

A file starts only when there is memory for it (about 1 GB; other programs share the machine), so on a busy machine
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
RAMP_SECONDS = 6  # a file started this recently has not taken its memory yet: count it as taken
_START = threading.Lock()
_STARTED: list[float] = []
_RUNNING = [0]
_WAITED = [0]  # files that waited for memory before they started


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


def wait_for_memory() -> None:
    """Start a file only when the memory holds it: other programs share the machine, so the number of files that fit
    changes during a run (a run that swaps took 538 s instead of 156 s). One file always runs, whatever is free."""
    with _START:
        waited = False
        while True:
            now = time.monotonic()
            recent = sum(1 for t in _STARTED if now - t < RAMP_SECONDS)
            free = available_gb()
            if _RUNNING[0] == 0 or free is None or free - recent * NEED_GB >= NEED_GB:
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


def run_one(name: str, slots: Queue, pg: bool, verbose: bool, threads_one: bool = True) -> dict:
    slot = slots.get()
    tmp = tempfile.mkdtemp(prefix=f"dclab-{name[5:-3]}-")
    env = {**ONE_THREAD, **os.environ, "TMPDIR": tmp} if threads_one else {**os.environ, "TMPDIR": tmp}
    if slot is not None:
        env["DCLAB_TEST_DATABASE_URL"] = slot
        if pg:
            env["DCLAB_DATABASE_URL"] = slot
    wait_for_memory()
    started = time.perf_counter()
    try:
        done = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", name] + (["-v"] if verbose else []),
                              cwd=ROOT, env=env, capture_output=True, text=True)
    finally:
        with _START:
            _RUNNING[0] -= 1
        shutil.rmtree(tmp, ignore_errors=True)
        slots.put(slot)
    output = done.stdout + done.stderr
    ran = RAN.search(output)
    tally = {k: int(v) for k, v in TALLY.findall(output.rsplit("\nRan ", 1)[-1])}
    return {"file": name, "ok": done.returncode == 0, "tests": int(ran.group(1)) if ran else 0, "seconds": time.perf_counter() - started,
            "failed": tally.get("failures", 0) + tally.get("errors", 0), "skipped": tally.get("skipped", 0), "output": output}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tests/run_parallel.py", description=__doc__.split("\n")[0])
    parser.add_argument("names", nargs="*", help="test files to run (default: every tests/test_*.py)")
    parser.add_argument("--jobs", "-j", type=int, default=default_jobs(), help="files at a time (default: what the cores and the memory allow)")
    parser.add_argument("--pg", action="store_true", help="run the product's stores on PostgreSQL, one database per worker")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every file's output, not only the failing ones")
    args = parser.parse_args(argv)

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
    names.sort(key=lambda n: -float(last.get(n, 60)))  # the slowest first; a new file counts as slow

    print(f"{len(names)} test files, {jobs} at a time{' on PostgreSQL' if args.pg else ''}", flush=True)
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

    if len(results) == len(files([])):  # only a whole run updates the timings
        try:
            timings.write_text(json.dumps({r["file"]: round(r["seconds"], 1) for r in results}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        except OSError:
            pass
    failed = [r for r in results if not r["ok"]]
    total = sum(r["tests"] for r in results)
    skipped = sum(r["skipped"] for r in results)
    print(f"\nRan {total} tests in {len(results)} files in {wall:.1f}s (one at a time: {sum(r['seconds'] for r in results):.1f}s)"
          + (f"; {_WAITED[0]} files waited for free memory" if _WAITED[0] else ""))
    if failed:
        print(f"FAILED ({sum(r['failed'] for r in failed)} failing tests in {len(failed)} files: {', '.join(r['file'] for r in failed)})")
        return 1
    print(f"OK (skipped={skipped})" if skipped else "OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
