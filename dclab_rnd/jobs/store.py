"""The job table (package 10.3): one row per piece of background work, so it outlives the process that started it.

A job is ``{id, kind, key, payload, status, progress, created, started, finished, heartbeat, worker, error, attempts,
cancel_requested, by}``. ``kind`` names its handler (``jobs/handlers.py``); ``key`` is what it works on (a project, an
intern session, a draft's asset), and only one queued or running job may hold a key, so two requests cannot run the
same project's stages at once, whichever app instance they reach.

Two backends with one interface: ``jobs.json`` in the workspace folder (a file lock lets the server and a worker
process share it), or the ``job`` table in PostgreSQL, where a worker claims with ``SELECT … FOR UPDATE SKIP LOCKED``
so any number of workers can run. Statuses: queued → running → done, failed, cancelled (a person stopped it) or
interrupted (its worker died or stopped; ``retry`` queues it again).
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ACTIVE = ("queued", "running")
ENDED = ("done", "failed", "cancelled", "interrupted")
RETRYABLE = ("failed", "cancelled", "interrupted")
KEEP_ENDED = 500  # the file store keeps the newest ended jobs; the table keeps every row


class ActiveJob(Exception):
    """Another queued or running job already holds this key."""

    def __init__(self, job: dict[str, Any] | None):
        super().__init__("a job is already queued or running for this")
        self.job = job


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_job(kind: str, key: str, payload: dict[str, Any], by: str = "human", worker: str | None = None) -> dict[str, Any]:
    running = worker is not None  # claimed at once: the request that queued it runs it now (``?wait=true``)
    return {"id": "j" + secrets.token_hex(6), "kind": kind, "key": key, "payload": payload, "by": by,
            "status": "running" if running else "queued", "progress": {}, "created": now(),
            "started": now() if running else None, "finished": None, "heartbeat": time.time() if running else None,
            "worker": worker, "error": None, "attempts": 1 if running else 0, "cancel_requested": False}


def _claimed(job: dict[str, Any], worker: str) -> dict[str, Any]:
    job.update(status="running", worker=worker, started=now(), heartbeat=time.time(), attempts=int(job.get("attempts") or 0) + 1)
    return job


class Jobs:
    """What both backends share; each implements ``_insert``, ``_change``, ``get``, ``list`` and ``claim``."""

    def _insert(self, job: dict[str, Any]) -> dict[str, Any]: ...
    def _change(self, job_id: str, fn: Callable[[dict[str, Any]], bool]) -> dict[str, Any]: ...  # fn edits in place; False keeps it
    def get(self, job_id: str) -> dict[str, Any]: ...  # KeyError when unknown
    def list(self, kind: str | None = None, key: str | None = None, active: bool | None = None, limit: int = 200) -> list[dict[str, Any]]: ...
    def claim(self, worker: str, job_id: str | None = None) -> dict[str, Any] | None: ...
    def stale(self, seconds: float) -> list[dict[str, Any]]: ...

    def enqueue(self, kind: str, key: str, payload: dict[str, Any], by: str = "human", worker: str | None = None) -> dict[str, Any]:
        return self._insert(new_job(kind, key, payload, by, worker))

    def active(self, kind: str, key: str) -> dict[str, Any] | None:
        found = self.list(kind=kind, key=key, active=True, limit=1)
        return found[0] if found else None

    def latest(self, kind: str, key: str) -> dict[str, Any] | None:
        found = self.list(kind=kind, key=key, limit=1)
        return found[0] if found else None

    def progress(self, job_id: str, values: dict[str, Any], worker: str | None = None) -> None:
        def put(j):
            if worker is not None and j.get("worker") != worker:
                return False  # another worker holds it now (this one was found stale): its progress is not ours to write
            j["progress"] = values
            j["heartbeat"] = time.time() if j["status"] == "running" else j.get("heartbeat")
            return True
        self._change(job_id, put)

    def finish(self, job_id: str, status: str, error: str | None = None, worker: str | None = None) -> dict[str, Any]:
        assert status in ENDED, status

        def put(j):
            if j["status"] != "running" or (worker is not None and j.get("worker") != worker):
                return False  # already ended (a recovery marked it interrupted), or retried by another worker: the first ending stands
            j.update(status=status, finished=now(), error=error)
            return True
        return self._change(job_id, put)

    def cancel(self, job_id: str) -> dict[str, Any]:
        """A queued job is cancelled at once; a running one is asked to stop at its next checkpoint."""
        def put(j):
            if j["status"] == "queued":
                j.update(status="cancelled", finished=now(), cancel_requested=True, error="Stopped by a person before it started")
                return True
            if j["status"] == "running" and not j.get("cancel_requested"):
                j["cancel_requested"] = True
                return True
            return False
        return self._change(job_id, put)

    def retry(self, job_id: str) -> dict[str, Any]:
        """Queue an ended job again (same payload; its attempts are kept). ValueError unless it failed, was stopped or interrupted."""
        def put(j):
            if j["status"] not in RETRYABLE:
                raise ValueError(f"only a failed, stopped or interrupted job can be retried; this one is {j['status']}")
            j.update(status="queued", error=None, cancel_requested=False, worker=None, started=None, finished=None, heartbeat=None, progress={})
            return True
        return self._change(job_id, put)

    def beat(self, job_ids: list[str], worker: str) -> None:
        for job_id in job_ids:
            def put(j):
                if j["status"] != "running" or j.get("worker") != worker:
                    return False
                j["heartbeat"] = time.time()
                return True
            try:
                self._change(job_id, put)
            except KeyError:
                continue

    def interrupt(self, job_id: str, worker: str | None, error: str, older_than: float | None = None) -> dict[str, Any] | None:
        """Mark a running job interrupted, only when the same worker still holds it (two recoveries never both act) and,
        with ``older_than``, only when its heartbeat is still that old (a beat that landed since the job was found stale wins)."""
        changed: list[bool] = []

        def put(j):
            if j["status"] != "running" or j.get("worker") != worker:
                return False
            if older_than is not None and float(j.get("heartbeat") or 0) >= time.time() - older_than:
                return False
            j.update(status="interrupted", finished=now(), error=error)
            changed.append(True)
            return True
        job = self._change(job_id, put)
        return job if changed else None


# ---------------------------------------------------------------------- files

_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


class FileJobs(Jobs):
    """``jobs.json`` in the workspace folder, replaced whole on each write, under this process's lock and a file lock."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _GUARD:
            self._lock = _LOCKS.setdefault(str(self.path.resolve()), threading.Lock())

    @contextlib.contextmanager
    def _locked(self):
        with self._lock, open(self.path.with_name(self.path.name + ".lock"), "a+") as lock:
            try:
                import fcntl

                fcntl.flock(lock, fcntl.LOCK_EX)
            except ImportError:  # Windows: this process's lock only
                pass
            yield

    def _all(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, rows: dict[str, dict[str, Any]]) -> None:
        ended = [k for k, j in rows.items() if j["status"] in ENDED]
        for k in ended[: max(0, len(ended) - KEEP_ENDED)]:  # the oldest ended jobs go first
            rows.pop(k)
        handle, temporary = tempfile.mkstemp(prefix=".jobs.", suffix=".tmp", dir=self.path.parent)
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(json.dumps(rows, ensure_ascii=False, default=str))
        os.replace(temporary, self.path)

    def _insert(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._locked():
            rows = self._all()
            held = next((j for j in rows.values() if j["kind"] == job["kind"] and j["key"] == job["key"] and j["status"] in ACTIVE), None)
            if held is not None:
                raise ActiveJob(held)
            rows[job["id"]] = job
            self._write(rows)
        return dict(job)

    def _change(self, job_id: str, fn) -> dict[str, Any]:
        with self._locked():
            rows = self._all()
            job = rows[job_id]  # KeyError when unknown
            if fn(job):
                if job["status"] in ACTIVE:  # a retry may not take a key another job holds now
                    held = next((j for k, j in rows.items() if k != job_id and j["kind"] == job["kind"] and j["key"] == job["key"] and j["status"] in ACTIVE), None)
                    if held is not None:
                        raise ActiveJob(held)
                self._write(rows)
            return dict(job)

    def get(self, job_id: str) -> dict[str, Any]:
        return dict(self._all()[job_id])

    def list(self, kind=None, key=None, active=None, limit=200):
        rows = [j for j in reversed(list(self._all().values()))  # newest first
                if (kind is None or j["kind"] == kind) and (key is None or j["key"] == key)
                and (active is None or (j["status"] in ACTIVE) == active)]
        return rows[:limit]

    def claim(self, worker, job_id=None):
        with self._locked():
            rows = self._all()
            for job in rows.values():  # oldest first
                if job["status"] == "queued" and not job.get("cancel_requested") and (job_id is None or job["id"] == job_id):
                    _claimed(job, worker)
                    self._write(rows)
                    return dict(job)
        return None

    def stale(self, seconds):
        cutoff = time.time() - seconds
        return [j for j in self._all().values() if j["status"] == "running" and float(j.get("heartbeat") or 0) < cutoff]


# ---------------------------------------------------------------------- PostgreSQL

class PgJobs(Jobs):
    def __init__(self, workspace_id: str, url: str | None = None):
        from ..storage import db

        self.workspace_id, self.engine = workspace_id, db.engine(url)

    @staticmethod
    def _columns(job: dict[str, Any]) -> dict[str, Any]:
        return {"kind": job["kind"], "key": job["key"], "status": job["status"], "cancel_requested": bool(job.get("cancel_requested")),
                "worker": job.get("worker"), "heartbeat": job.get("heartbeat"), "doc": job}

    def _insert(self, job):
        import sqlalchemy as sa

        from ..storage.models import job as t

        try:
            with self.engine.begin() as c:
                c.execute(sa.insert(t).values(id=job["id"], workspace_id=self.workspace_id, **self._columns(job)))
        except sa.exc.IntegrityError:
            raise ActiveJob(self.active(job["kind"], job["key"])) from None
        return dict(job)

    def _change(self, job_id, fn):
        import sqlalchemy as sa

        from ..storage.models import job as t

        try:
            with self.engine.begin() as c:
                doc = c.execute(sa.select(t.c.doc).where(t.c.workspace_id == self.workspace_id, t.c.id == job_id).with_for_update()).scalar()
                if doc is None:
                    raise KeyError(job_id)
                if fn(doc):
                    c.execute(sa.update(t).where(t.c.workspace_id == self.workspace_id, t.c.id == job_id).values(**self._columns(doc)))
        except sa.exc.IntegrityError:  # a retry onto a key another job holds now
            raise ActiveJob(self.active(doc["kind"], doc["key"])) from None
        return doc

    def get(self, job_id):
        import sqlalchemy as sa

        from ..storage.models import job as t

        with self.engine.connect() as c:
            doc = c.execute(sa.select(t.c.doc).where(t.c.workspace_id == self.workspace_id, t.c.id == job_id)).scalar()
        if doc is None:
            raise KeyError(job_id)
        return doc

    def list(self, kind=None, key=None, active=None, limit=200):
        import sqlalchemy as sa

        from ..storage.models import job as t

        q = sa.select(t.c.doc).where(t.c.workspace_id == self.workspace_id)
        if kind is not None:
            q = q.where(t.c.kind == kind)
        if key is not None:
            q = q.where(t.c.key == key)
        if active is not None:
            q = q.where(t.c.status.in_(ACTIVE) if active else t.c.status.notin_(ACTIVE))
        with self.engine.connect() as c:
            return [r[0] for r in c.execute(q.order_by(t.c.created.desc(), t.c.id.desc()).limit(limit))]

    def claim(self, worker, job_id=None):
        import sqlalchemy as sa

        from ..storage.models import job as t

        q = sa.select(t.c.doc).where(t.c.workspace_id == self.workspace_id, t.c.status == "queued", t.c.cancel_requested.is_(False))
        if job_id is not None:
            q = q.where(t.c.id == job_id)
        with self.engine.begin() as c:  # SKIP LOCKED: two workers never claim the same row, and neither waits for the other
            doc = c.execute(q.order_by(t.c.created, t.c.id).limit(1).with_for_update(skip_locked=True)).scalar()
            if doc is None:
                return None
            _claimed(doc, worker)
            c.execute(sa.update(t).where(t.c.workspace_id == self.workspace_id, t.c.id == doc["id"]).values(**self._columns(doc)))
        return doc

    def stale(self, seconds):
        import sqlalchemy as sa

        from ..storage.models import job as t

        with self.engine.connect() as c:
            return [r[0] for r in c.execute(sa.select(t.c.doc).where(
                t.c.workspace_id == self.workspace_id, t.c.status == "running",
                sa.func.coalesce(t.c.heartbeat, 0) < time.time() - seconds))]


def open_jobs(home: Path) -> Jobs:
    from ..storage import db

    if db.database_url(required=False):
        return PgJobs(db.workspace_for(Path(home)))
    return FileJobs(Path(home) / "jobs.json")
