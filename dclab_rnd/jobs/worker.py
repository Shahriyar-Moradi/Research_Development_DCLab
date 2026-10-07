"""Running jobs (package 10.3): the worker that claims them, the checkpoint a job calls between its steps, recovery.

A handler runs one kind of job. Its ``run(env, payload)`` does the work; ``settle(env, payload, reason, job)`` puts the
domain back in order when the job ended without finishing (a stage left "queued", an asset left "cleaning", an intern
session left "running"), and does nothing when there is nothing to fix, so calling it twice is safe.

Between its steps a job calls ``checkpoint()``. It raises ``Stopped`` when a person asked to stop the job, or when its
worker is shutting down; outside a job it does nothing, so the same code runs from a tool call or a test. A step
already running is never cut: a stage finishes, then the job stops (a stage stays deterministic: same seed, same record).

The worker keeps a heartbeat on the jobs it runs. A job whose heartbeat is older than ``stale`` seconds belonged to
a worker that died (a killed server); any worker that sees it marks it interrupted and settles it, and a person can
retry it from the Compute page.
"""

from __future__ import annotations

import os
import secrets
import socket
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from .store import ActiveJob, Jobs

TEXT = {"stopped": "Stopped by a person before it finished",
        "interrupted": "Interrupted: the worker stopped before it finished; retry it from the Compute page"}


class Stopped(BaseException):
    """Raised by ``checkpoint()``. A BaseException, like asyncio's CancelledError: the ``except Exception`` that turns a
    tool's failure into an error result must not swallow a stop."""

    def __init__(self, reason: str = "stopped"):
        super().__init__(TEXT[reason])
        self.reason = reason


@dataclass
class Handler:
    run: Callable[[Any, dict[str, Any]], Any]
    settle: Callable[[Any, dict[str, Any], str, dict[str, Any]], None]


HANDLERS: dict[str, Handler] = {}


def handler(kind: str, run, settle) -> None:
    HANDLERS[kind] = Handler(run, settle)


_LOCAL = threading.local()


class Control:
    def __init__(self, store: Jobs, job: dict[str, Any], worker: "Worker"):
        self.store, self.job, self.worker = store, job, worker
        self.observed: str | None = None  # the reason a checkpoint stopped the job, even when the handler recorded it and returned
        self.lost = False  # another worker holds the job now: it settled it, so this worker writes nothing more about it

    def check(self) -> None:
        reason = "interrupted" if self.worker.stopping else None
        if reason is None:
            try:
                row = self.store.get(self.job["id"])
            except KeyError:
                row = None
            if row is not None and (row["status"] != "running" or row.get("worker") != self.worker.id):
                reason, self.lost = "interrupted", True  # found stale and marked by another worker (or retried there): not ours any more
            elif row is not None and row.get("cancel_requested"):
                reason = "stopped"
        if reason:
            self.observed = reason
            raise Stopped(reason)

    def owned(self) -> bool:
        """Whether this worker still holds the job (read without raising); a job taken over is marked lost."""
        try:
            row = self.store.get(self.job["id"])
        except KeyError:
            return False
        if row["status"] != "running" or row.get("worker") != self.worker.id:
            self.lost = True
        return not self.lost

    def progress(self, values: dict[str, Any]) -> None:
        try:
            self.store.progress(self.job["id"], values, worker=self.worker.id)
        except KeyError:
            pass


def current() -> Control | None:
    return getattr(_LOCAL, "control", None)


def checkpoint() -> None:
    """Stop here if this job was asked to stop; a no-op outside a job."""
    control = current()
    if control is not None:
        control.check()


def progress(**values: Any) -> None:
    """Record where the job is (shown on the Compute page); a no-op outside a job."""
    control = current()
    if control is not None:
        control.progress(values)


def _acting(env: Any, job: dict[str, Any]):
    import contextlib

    from .. import context
    from ..accounts.principal import Principal, acting

    stack = contextlib.ExitStack()
    stack.enter_context(context.using(env))
    user = job.get("user")
    if isinstance(user, dict) and user.get("role"):
        stack.enter_context(acting(Principal(role=user["role"], workspace_id=user.get("workspace_id"), user_id=user.get("user_id"),
                                             email=user.get("email") or "", name=user.get("name") or "", via="job")))
    return stack


def _settle(env: Any, job: dict[str, Any], reason: str) -> None:
    h = HANDLERS.get(job["kind"])
    if h is None:
        return
    try:
        h.settle(env, job.get("payload") or {}, reason, job)
    except Exception:  # noqa: BLE001 — a domain object already gone (a deleted project) has nothing to settle
        pass


class Worker:
    """Claims queued jobs and runs them, ``threads`` at a time. ``threads=0`` claims nothing: it only keeps the heartbeat
    of jobs this process runs itself (``execute`` from a request with ``?wait=true``) and recovers stale ones."""

    def __init__(self, store: Jobs, env: Any, threads: int = 4, poll: float = 2.0, beat: float = 5.0, stale: float | None = None):
        self.store, self.env, self.threads, self.poll, self.beat = store, env, threads, poll, beat
        self.stale = float(stale if stale is not None else os.environ.get("DCLAB_JOB_STALE_SECONDS") or 120)
        self.id = f"{socket.gethostname()}:{os.getpid()}:{secrets.token_hex(3)}"
        self.stopping = False
        self.draining = False  # claims nothing new; finishes what it runs (a cloud deployment replacing it: 12.6)
        self.running: dict[str, Control] = {}
        self._wake, self._halt = threading.Event(), threading.Event()
        self._started = False
        self._guard = threading.Lock()
        self._mine: set[str] = set()  # jobs the dispatcher claimed and runs on their own threads

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "Worker":
        with self._guard:
            if self._started:
                return self
            self._started, self.stopping = True, False
            self._halt, self._wake = threading.Event(), threading.Event()
        self.recover()
        loops = [(self._keep, "dclab-job-heartbeat"), (self._retain, "dclab-retention")] + ([(self._loop, "dclab-job-dispatch")] if self.threads > 0 else [])
        for target, name in loops:
            threading.Thread(target=target, args=(self._halt,), name=name, daemon=True).start()
        return self

    def notify(self) -> None:
        self._wake.set()

    def stop(self, wait: float = 5.0) -> None:
        """Claim nothing more; the jobs running here stop at their next checkpoint as interrupted (retryable). A job still
        inside a long step when ``wait`` ends keeps its thread; if the process then exits, a later recovery marks it."""
        with self._guard:
            if not self._started:
                return
            self._started, self.stopping = False, True
            self._halt.set()
            self._wake.set()
        deadline = time.monotonic() + wait
        while self.running and time.monotonic() < deadline:
            time.sleep(0.05)

    # ------------------------------------------------------------------ running
    def _loop(self, halt: threading.Event) -> None:
        """One thread claims; each claimed job runs on its own thread, at most ``threads`` at a time."""
        while not halt.is_set():
            job = None
            self._wake.clear()  # before claiming: a job queued from now on wakes the wait below
            if self._busy() < self.threads and not self.draining:
                try:
                    job = self.store.claim(self.id)
                except Exception:  # noqa: BLE001 — a database that is briefly away: try again on the next poll
                    traceback.print_exc()
            if job is None:
                self._wake.wait(self.poll)
                continue
            self._mine.add(job["id"])
            threading.Thread(target=self._run_claimed, args=(job,), name=f"dclab-job-{job['kind']}", daemon=True).start()

    def drain(self) -> None:
        """Claim nothing more; the jobs already running finish (another worker, a newer version, takes the queue)."""
        self.draining = True

    def has_work(self) -> bool:
        """A job claimed (even one whose thread has not started yet) or running here."""
        return bool(self._mine or self.running)

    def _busy(self) -> int:
        return len(self._mine)

    def _run_claimed(self, job: dict[str, Any]) -> None:
        try:
            self.execute(job)
        finally:
            self._mine.discard(job["id"])
            self._wake.set()  # a slot is free: claim the next one

    def execute(self, job: dict[str, Any]) -> dict[str, Any]:
        """Run a job this worker has claimed, in the calling thread, and record how it ended."""
        control = Control(self.store, job, self)
        began = time.monotonic()
        self.running[job["id"]] = control
        status, error = "done", None
        try:
            h = HANDLERS.get(job["kind"])
            if h is None:
                raise RuntimeError(f"no handler for jobs of kind {job['kind']!r}")
            outer, _LOCAL.control = current(), control  # a job run inside another job's thread (the agent's synthetic request)
            try:
                with _acting(self.env, job):  # the job's workspace and the person who started it (package 10.2)
                    h.run(self.env, job.get("payload") or {})
            finally:
                _LOCAL.control = outer
        except Stopped as stop:
            control.observed = stop.reason
        except Exception as exc:  # noqa: BLE001 — the job's failure, recorded on the job
            status, error = "failed", f"{type(exc).__name__}: {str(exc)[:400]}"
        if control.observed:
            status, error = ("cancelled" if control.observed == "stopped" else "interrupted"), TEXT[control.observed]
        try:
            if status != "done" and not control.lost:  # a lost job was settled by the worker that marked it
                _settle(self.env, job, control.observed or "failed")
            from .. import observe

            ended = self.store.finish(job["id"], status, error, worker=self.id)
            observe.observe_job(job["kind"], status, time.monotonic() - began)  # 12.4: durations and failures, by kind
            if status != "done":
                observe.event("job ended", job=job["id"], kind=job["kind"], status=status, attempts=job.get("attempts"))
            return ended
        finally:
            self.running.pop(job["id"], None)

    def run_here(self, job: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        """Run a job now in this thread: it was queued already claimed by this worker, or is claimed here if still queued;
        when another worker has it, wait for it to end."""
        if job["status"] == "running" and job.get("worker") == self.id and job["id"] not in self.running:
            return self.execute(job)
        claimed = self.store.claim(self.id, job_id=job["id"])
        if claimed is not None:
            return self.execute(claimed)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            current_job = self.store.get(job["id"])
            if current_job["status"] not in ("queued", "running") or (deadline is not None and time.monotonic() > deadline):
                return current_job
            time.sleep(0.1)

    # ------------------------------------------------------------------ heartbeat and recovery
    def _keep(self, halt: threading.Event) -> None:
        last_recovery = time.monotonic()
        while not halt.wait(self.beat):
            try:
                self.store.beat(list(self.running), self.id)
                if time.monotonic() - last_recovery >= min(self.stale / 2, 60):
                    last_recovery = time.monotonic()
                    self.recover()
            except Exception:  # noqa: BLE001 — the next beat tries again
                traceback.print_exc()

    def _retain(self, halt: threading.Event) -> None:
        """Raw uploads past DCLAB_RETENTION_RAW_DAYS (package 12.5): on a thread of its own, so a long sweep never delays a heartbeat."""
        from .. import retention

        state: dict[str, float] = {}
        while not halt.wait(60):
            try:
                retention.sweep_due(self.env, state, time.monotonic())
            except Exception:  # noqa: BLE001 — the next hour tries again
                traceback.print_exc()

    def recover(self) -> list[dict[str, Any]]:
        """Mark interrupted every running job whose worker stopped beating, or whose worker was a process of this machine
        that no longer exists (a server killed and started again within the stale time), and put its domain back in order."""
        done = []
        try:
            stale = [(j, self.stale) for j in self.store.stale(self.stale)]
            quick = max(3 * self.beat, 15.0)  # a dead process here: no need to wait the full stale time, but a few missed beats
            # still (two containers may share a host name without sharing process ids)
            stale += [(j, quick) for j in self.store.list(active=True, limit=1000)
                      if j["status"] == "running" and _dead_here(j.get("worker")) and j["id"] not in {s["id"] for s, _ in stale}]
        except Exception:  # noqa: BLE001
            return done
        for job, older_than in stale:
            if job["id"] in self.running or job.get("worker") == self.id:
                continue
            marked = self.store.interrupt(job["id"], job.get("worker"), TEXT["interrupted"], older_than=older_than)
            if marked is not None:
                from .. import observe

                _settle(self.env, marked, "interrupted")
                done.append(marked)
                observe.observe_job(marked["kind"], "interrupted", _since(marked.get("started")))  # a crashed worker's job counts too (12.4)
                observe.event("job ended", job=marked["id"], kind=marked["kind"], status="interrupted", attempts=marked.get("attempts"), recovered=True)
        return done


def _since(started: Any) -> float:
    """Seconds since an ISO time (0 when unknown): how long a recovered job had been running."""
    from datetime import datetime, timezone

    try:
        at = datetime.fromisoformat(str(started))
        return max(0.0, (datetime.now(timezone.utc) - (at if at.tzinfo else at.replace(tzinfo=timezone.utc))).total_seconds())
    except (TypeError, ValueError):
        return 0.0


def _dead_here(worker: str | None) -> bool:
    """True when ``worker`` (``host:pid:token``) names a process of this machine that has ended."""
    try:
        host, pid, _ = str(worker or "").rsplit(":", 2)
        pid = int(pid)
    except ValueError:
        return False
    if host != socket.gethostname() or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:  # alive, owned by someone else
        return False
    return False


__all__ = ["ActiveJob", "Control", "HANDLERS", "Handler", "Stopped", "TEXT", "Worker", "checkpoint", "current", "handler", "progress"]
