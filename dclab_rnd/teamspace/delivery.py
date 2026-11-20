"""Delivering what travels between teams (W4, 2026-11-20): one ``team_delivery`` job per task, its rows in order.

The store saves a task, a clarification, its answer or a reply together with its delivery row (``teamspace/store.py``).
A task's pending rows are delivered by one job of the job table (``dclab_rnd/jobs``), keyed by the task, oldest first:
a newer row of a task is never delivered before an older one (a corrected reply never lands before the reply it
corrects). For each row the job:

1. rechecks the member who started it, now, with write permission (``memory.scope.revalidate_job_scope``): a member who
   lost access, or was made a viewer, gets the row refused and the target is never called;
2. hands the row to the delivery target (:func:`set_target`; W5 supplies the agents; until then the default target only
   acknowledges it: the row is the recipient's inbox), within a time limit (``DCLAB_TEAM_DELIVERY_TIMEOUT_SECONDS``, 30);
3. marks the row delivered, and a reply posted to its thread, once.

A failure ends the job: the attempt is counted on the row with its next time, doubling from
``DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS`` (2 s, at most 60 s); after ``DCLAB_TEAM_DELIVERY_ATTEMPTS`` (5) the row is
blocked. Nothing waits inside a worker. The relay, :func:`requeue_undelivered`, gives every task whose next row is due a
job again; a worker runs it when it starts and on every recovery tick, so a row saved before a crash, one whose job
failed or stopped, and one waiting for its next attempt are all picked up. A job that failed for any other reason counts
an attempt too (``settle``), so the limit holds.

What the requester sees (``reason``) is fixed words and the error's class; the error's text stays in ``last_error``,
scrubbed. A target is called with the row, whose ``id`` is its idempotency key: a target cut off by a crash or the time
limit after it acted may be called again for the same id, and must treat that as the same delivery.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

from .store import Row, Written, is_due, open_team_work, team_label

KIND = "team_delivery"
Target = Callable[[Any, Row], Any]
REFUSED = "Not delivered: the member who asked can no longer change work in this project."


class TemporaryFailure(Exception):
    """The recipient could not take it now: tried again later, within the bound."""


class PermanentFailure(Exception):
    """The recipient will never take it as it is: blocked at once."""


class TargetTimeout(TemporaryFailure):
    """The recipient did not answer within the time limit."""


class JobFailed(Exception):
    """The delivery job itself failed before it could record an attempt."""


_HOOKS: dict[str, Any] = {"target": None, "membership": None}


def set_target(fn: Optional[Target]) -> Optional[Target]:
    """The function that hands a row to its recipient (W5: the team agents). Returns the one it replaces."""
    previous, _HOOKS["target"] = _HOOKS["target"], fn
    return previous


def set_membership_checker(fn: Optional[Callable[[str, str], Optional[str]]]) -> Any:
    """How a member's live role is read at delivery (``uid, workspace -> role or None``). Unset, the accounts database
    is asked when sign-in is on (``revalidate_job_scope``), and a local workspace without sign-in checks the project only."""
    previous, _HOOKS["membership"] = _HOOKS["membership"], fn
    return previous


def acknowledge(env: Any, row: Row) -> Row:
    """The default target until W5: nothing to hand over, the row is the recipient's inbox."""
    return {"inbox": row["to_team"]}


def max_attempts() -> int:
    return max(1, int(os.environ.get("DCLAB_TEAM_DELIVERY_ATTEMPTS") or 5))


def backoff_seconds() -> float:
    return max(0.0, float(os.environ.get("DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS") or 2.0))


def timeout_seconds() -> float:
    return max(0.01, float(os.environ.get("DCLAB_TEAM_DELIVERY_TIMEOUT_SECONDS") or 30.0))


def _home(env: Any) -> Optional[Path]:
    for home in (getattr(getattr(env, "store", None), "home", None), getattr(env, "home", None)):
        if isinstance(home, (str, os.PathLike)) and str(home):
            return Path(home)
    return None


def store_for(env: Any) -> Any:
    """The env's team work (``env.teamwork`` when it has one), else the one of its workspace folder."""
    held = getattr(env, "teamwork", None)
    if held is not None and hasattr(held, "pending_heads"):
        return held
    home = _home(env)
    if home is None:
        raise LookupError("no workspace to deliver in")
    return open_team_work(home, projects=getattr(env, "projects", None))


def _jobs_for(env: Any, jobs: Any = None) -> Any:
    if jobs is not None:
        return jobs
    held = getattr(env, "job_store", None)
    if held is not None:
        return held
    from ..jobs import open_jobs

    home = _home(env)
    if home is None:
        raise LookupError("no workspace to queue in")
    return open_jobs(home)


def job_key(project_id: str, task_id: str) -> str:
    """One delivery job per task (ids are unique within a project)."""
    return f"{project_id}/{task_id}"


@contextlib.contextmanager
def _as_initiator(row: Row) -> Iterator[None]:
    """Queue the job as the member who started the row, so the worker rechecks that member (and no one else)."""
    who = row.get("initiator") or {}
    if not who.get("user_id") or not who.get("role"):
        yield
        return
    from ..accounts.principal import Principal, acting

    with acting(Principal(role=str(who["role"]), workspace_id=who.get("workspace_id"), user_id=str(who["user_id"]), via="job")):
        yield


def enqueue(env: Any, row: Optional[Row], jobs: Any = None) -> Optional[Row]:
    """The job that delivers this row's task (the one already queued or running for it, if any); None when the row is not
    pending. The job runs as the member of the task's oldest pending row."""
    if not row or row.get("status") != "pending":
        return None
    from ..jobs import ActiveJob

    heads = [r for r in store_for(env).deliveries(status="pending", project_id=row["project_id"], task_id=row["task_id"])]
    head = heads[0] if heads else row
    try:
        with _as_initiator(head):
            job = _jobs_for(env, jobs).enqueue(KIND, job_key(row["project_id"], row["task_id"]),
                                               {"project_id": row["project_id"], "task_id": row["task_id"]}, by="agent")
    except ActiveJob as held:
        job = held.job
    worker = getattr(env, "worker", None)
    if worker is not None and hasattr(worker, "notify"):
        worker.notify()
    return job


def send(env: Any, written: Written, jobs: Any = None) -> Optional[Row]:
    """Queue the delivery a store change saved (if any)."""
    return enqueue(env, written.delivery, jobs)


def requeue_undelivered(env: Any, jobs: Any = None) -> list[str]:
    """The relay: every task whose next pending row is due gets a job (a task already queued or running keeps its job).
    Run by a worker when it starts and on each recovery tick; quiet when the env has no workspace or nothing was saved."""
    if _home(env) is None and getattr(env, "teamwork", None) is None:
        return []
    from ..storage import db

    home = _home(env)
    if getattr(env, "teamwork", None) is None and not db.database_url(required=False) \
            and not (home is not None and (home / "teamspace").is_dir()):
        return []  # a folder workspace that never saved team work: nothing to read, nothing written
    queued = []
    for row in store_for(env).pending_heads():
        if is_due(row) and enqueue(env, row, jobs) is not None:
            queued.append(job_key(row["project_id"], row["task_id"]))
    return queued


def authorize(env: Any, row: Row) -> None:
    """The member who started it may still change work in this project, now; AccessDenied otherwise."""
    from ..memory.scope import revalidate_job_scope

    revalidate_job_scope(env, {"user": row.get("initiator"), "payload": {"project_id": row["project_id"]}}, require_write=True,
                         membership_checker=_HOOKS["membership"])


def _call(target: Target, env: Any, row: Row, timeout: float) -> Any:
    """The target, on its own thread with this context, given ``timeout`` seconds; TargetTimeout when it does not answer
    (its thread is left to end on its own: a hung recipient never holds the worker)."""
    box: dict[str, Any] = {}
    context = contextvars.copy_context()

    def go() -> None:
        try:
            box["value"] = context.run(target, env, dict(row))
        except BaseException as exc:  # noqa: BLE001 — handed back to the job below
            box["error"] = exc

    thread = threading.Thread(target=go, name="dclab-team-delivery-target", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TargetTimeout(f"no answer within {timeout:g} s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _failed(store: Any, row: Row, error: BaseException) -> Row:
    return store.delivery_failed(row["id"], error, max_attempts=max_attempts(), backoff_seconds=backoff_seconds(),
                                 project_id=row["project_id"])


def run(env: Any, payload: Mapping[str, Any]) -> list[Row]:
    """Deliver one task's pending rows, oldest first (the job's work); stop at a row whose next attempt is not due yet, or
    after a failed attempt. Returns the rows as they ended."""
    from ..jobs.worker import checkpoint
    from ..memory.scope import AccessDenied

    store = store_for(env)
    project_id, task_id = str(payload["project_id"]), str(payload["task_id"])
    done: list[Row] = []
    while True:
        checkpoint()
        pending = store.deliveries(status="pending", project_id=project_id, task_id=task_id)
        if not pending or not is_due(pending[0]):
            return done  # nothing left, or the oldest waits for its next attempt: the relay queues the task again when due
        row = pending[0]
        try:
            authorize(env, row)
        except AccessDenied as exc:
            done.append(store.delivery_stop(row["id"], "refused", REFUSED, error=exc, project_id=project_id))
            continue
        try:
            receipt = _call(_HOOKS["target"] or acknowledge, env, row, timeout_seconds())
        except PermanentFailure as exc:
            done.append(store.delivery_stop(row["id"], "blocked", f"Not delivered: {team_label(row['to_team'])} cannot take it "
                                            f"({type(exc).__name__}).", error=exc, project_id=project_id))
            continue
        except Exception as exc:  # noqa: BLE001 — a temporary failure, or one the target did not name: counted, retried later
            ended = _failed(store, row, exc)
            done.append(ended)
            if ended["status"] == "pending":
                return done
            continue  # blocked: it will never be delivered, so the next row is not delivered before it
        done.append(store.delivery_done(row["id"], receipt, project_id=project_id)[0])


def settle(env: Any, payload: Mapping[str, Any], reason: str, job: Mapping[str, Any]) -> None:
    """The job ended without finishing. A member who lost access gets the task's next row refused (the worker refused the
    job before it ran); any other failure counts an attempt on that row, so the attempt limit holds. A stopped or
    interrupted job counts nothing: the relay queues the task again."""
    from ..memory.scope import AccessDenied

    if reason != "failed":
        return
    try:
        store = store_for(env)
        pending = store.deliveries(status="pending", project_id=str(payload.get("project_id") or ""),
                                   task_id=str(payload.get("task_id") or ""))
    except Exception:  # noqa: BLE001 — the store is away: the relay tries again
        return
    if not pending:
        return
    row = pending[0]
    try:
        authorize(env, row)
    except AccessDenied as exc:
        store.delivery_stop(row["id"], "refused", REFUSED, error=exc, project_id=row["project_id"])
        return
    except Exception:  # noqa: BLE001 — the check itself failed: counted below like any failure
        pass
    try:
        _failed(store, row, JobFailed("the delivery job failed before it recorded an attempt"))
    except Exception:  # noqa: BLE001
        pass


__all__ = ["KIND", "JobFailed", "PermanentFailure", "REFUSED", "TargetTimeout", "TemporaryFailure", "acknowledge", "authorize", "enqueue",
           "job_key", "requeue_undelivered", "run", "send", "set_membership_checker", "set_target", "settle", "store_for"]
