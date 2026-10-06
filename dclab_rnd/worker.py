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
    from .agentic.services import Services

    services = Services(settings.model_copy(update={"worker": "inline", **({"worker_threads": max(1, args.threads)} if args.threads else {})}))
    worker = services.worker.start()
    print(f"worker {worker.id}: {worker.threads} at a time on {settings.agent_home}"
          f"{' (PostgreSQL)' if settings.database_url else ' (files)'}; Ctrl-C stops", flush=True)
    done = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())
    done.wait()
    print("stopping: running jobs stop at their next checkpoint", flush=True)
    worker.stop(wait=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
