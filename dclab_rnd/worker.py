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


def watch_step(pool, protection, state: dict, check=None) -> bool:
    """Every five seconds inside an ECS task (12.6): hold scale-in protection while a job runs; once a minute ask
    whether a deployment replaced this task, and if so drain (claim nothing new, finish what runs). Returns True when
    drained and idle: the process then ends, so ECS starts whatever the service's task definition now is (the new
    version, or the old one again after a rollback), and no worker stays drained."""
    from . import observe
    from .jobs import queue_metric

    check = check or queue_metric.superseded
    state["ticks"] += 1
    if not state["drained"] and state["ticks"] % 12 == 0:
        try:
            if check():
                state["drained"] = True
                observe.event("worker draining: a newer version replaces it")
        except Exception as exc:  # noqa: BLE001 — a minute later it is asked again
            observe.event("deployment not checked", error=type(exc).__name__)
    busy = False
    for s in pool.all():
        if state["drained"]:
            s.worker.drain()  # every tick: a workspace opened since gets a draining worker too
        busy = busy or s.worker.has_work()  # a job just claimed counts, before its thread starts
    try:
        protection.update(busy)
    except Exception as exc:  # noqa: BLE001 — five seconds later it is tried again
        observe.event("task protection not set", error=type(exc).__name__)
    # idle on two ticks in a row after draining: a claim that was under way when it drained has landed by then
    state["idle"] = state.get("idle", 0) + 1 if state["drained"] and not busy else 0
    if state["idle"] >= 2:
        observe.event("worker drained: ending so the service starts its current version")
        return True
    return False


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
    from .jobs import queue_metric

    scaler = queue_metric.PoolScaler.from_environment()  # Google Cloud: the worker pool follows the queue (12.6)
    protection = queue_metric.TaskProtection()  # AWS: a busy task is never stopped by scaling in (12.6)
    if protection.uri:
        def protect() -> None:
            state = {"ticks": 0, "drained": False}
            while not done.wait(5):
                if watch_step(pool, protection, state):
                    done.set()  # drained and idle: the process ends, and ECS starts the service's current version
                    return
        threading.Thread(target=protect, name="dclab-task-protection", daemon=True).start()
    while not done.wait(60):
        pool.start_all()  # a workspace made since: its jobs are run too (a database briefly away is reported, and tried again)
        try:  # the queue for the cloud's autoscaler (package 12.6; DCLAB_QUEUE_METRIC)
            queued, active = queue_metric.counts(pool.all())
        except Exception as exc:  # noqa: BLE001 — a minute later it is tried again; the jobs run either way
            observe.event("queue not counted", error=type(exc).__name__)
            continue
        try:
            queue_metric.publish(queued, active)
        except Exception as exc:  # noqa: BLE001 — the scaling below does not need the metric
            observe.event("queue metric not published", error=type(exc).__name__)
        try:
            scaled = scaler.check(queued, active) if scaler is not None else None
            if scaled is not None:
                observe.event("workers scaled", workers=scaled, queued=queued, active=active)
        except Exception as exc:  # noqa: BLE001
            observe.event("workers not scaled", error=type(exc).__name__)
    print("stopping: running jobs stop at their next checkpoint", flush=True)
    for each in pool.all():
        each.worker.stop(wait=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
