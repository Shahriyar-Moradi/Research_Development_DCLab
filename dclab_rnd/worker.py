"""A worker process for the job table (package 10.3): ``python -m dclab_rnd.worker``.

It opens the same workspace the server does (``DCLAB_AGENT_HOME``, and PostgreSQL when ``DCLAB_DATABASE_URL`` is set),
recovers jobs a dead worker left running, then claims queued jobs and runs them until it is stopped (Ctrl-C or SIGTERM:
the jobs it runs stop at their next checkpoint as interrupted, and any worker can retry them). Run as many as you like
against PostgreSQL: each claim skips rows another worker holds. With the workspace on files, one machine only.

Not to be confused with ``dclab_rnd.agentic.worker``, the subprocess the legacy research campaign runs experiments in.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.worker", description=__doc__.split("\n")[0])
    parser.add_argument("--home", type=Path, default=None, help="the workspace folder (default: DCLAB_AGENT_HOME, as the server)")
    parser.add_argument("--threads", type=int, default=None, help="jobs run at a time (default: DCLAB_WORKER_THREADS or 4)")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    from .settings import ROOT, Settings

    load_dotenv(ROOT / ".env", override=False)  # the server reads .env the same way: both open the same workspace
    settings = Settings.load(args.home)
    problems = settings.problems()
    if problems:
        print("The worker cannot start: " + "; ".join(problems), file=sys.stderr)
        return 2
    from .agentic.pool import Pool

    pool = Pool(settings.model_copy(update={"worker": "inline", **({"worker_threads": max(1, args.threads)} if args.threads else {})}))
    from . import observe

    observe.configure_logging()
    observe.serve_metrics()  # the worker's own counters: jobs, durations, failures, model use (12.4)
    pool.start_all()  # with accounts on, a worker for every workspace (package 10.2)
    worker = pool.default.worker
    print(f"worker {worker.id}: {worker.threads} at a time on {settings.agent_home}"
          f"{' (PostgreSQL)' if settings.database_url else ' (files)'}; Ctrl-C stops", flush=True)
    done = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())
    while not done.wait(60):
        pool.start_all()  # a workspace made since: its jobs are run too (a database briefly away is reported, and tried again)
    print("stopping: running jobs stop at their next checkpoint", flush=True)
    for each in pool.all():
        each.worker.stop(wait=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
