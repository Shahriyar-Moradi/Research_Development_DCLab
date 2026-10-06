"""The projects, drafts and intern sessions in PostgreSQL (package 9.2): the same interface as the file stores.

Every store belongs to one workspace and never returns another workspace's row (``get`` answers KeyError, as for an
unknown id). Each ``update`` locks the row it changes (``SELECT … FOR UPDATE``), so two app instances cannot lose each
other's write. The draft's event numbers are handed out by an UPDATE on the draft row, so they never repeat; an agent
turn is a PostgreSQL advisory lock, which holds across processes.

Tables and datasets are still files (``data_dir``, ``export_dir``): package 9.3 moves them behind file storage.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..draft.store import new_draft, now as draft_now
from ..intern.sessions import new_session
from ..studio.store import STAGE_KEYS, migrate, new_project
from . import db
from .models import activity, draft, draft_event, intern_session, project, stage_record, transition

ID = re.compile(r"[0-9a-f]{12}")


def _key(value: str) -> str:
    if not ID.fullmatch(value or ""):
        raise KeyError(value)
    return value


def _doc(row: Any) -> dict[str, Any]:
    return dict(row)  # a fresh copy: callers change what they get


class _Base:
    def __init__(self, workspace_id: str, files_home: Path, url: str | None = None):
        self.workspace_id, self.files_home = workspace_id, Path(files_home)
        self.engine = db.engine(url)

    def location(self) -> str:
        return f"PostgreSQL workspace {self.workspace_id}; files in {self.files_home}"


class PgProjects(_Base):
    def _folder(self, project_id: str) -> Path:
        return self.files_home / "projects" / _key(project_id)

    def data_dir(self, project_id: str) -> Path:
        path = self._folder(project_id) / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def export_dir(self, project_id: str) -> Path:
        path = self._folder(project_id) / "exports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------ documents
    def create(self, name: str, industry: str = "general", goal: str = "") -> dict[str, Any]:
        for _ in range(5):
            doc = new_project(uuid.uuid4().hex[:12], name, industry, goal)
            try:
                with self.engine.begin() as c:
                    c.execute(sa.insert(project).values(id=doc["id"], workspace_id=self.workspace_id, name=doc["name"], industry=doc["industry"], doc=doc))
                break
            except sa.exc.IntegrityError:  # the (rare) id collision: try another
                continue
        else:
            raise RuntimeError("could not allocate a project id")
        self.data_dir(doc["id"])
        self.log(doc["id"], "created", {"name": doc["name"]})
        return doc

    def _select(self, project_id: str, lock: bool = False):
        query = sa.select(project.c.doc).where(project.c.id == _key(project_id), project.c.workspace_id == self.workspace_id)
        return query.with_for_update() if lock else query

    def get(self, project_id: str) -> dict[str, Any]:
        with self.engine.begin() as c:
            doc = c.execute(self._select(project_id, lock=True)).scalar()
            if doc is None:
                raise KeyError(project_id)
            if migrate(doc):  # an old document is brought up to date once, when it is first read
                self._write(c, doc)
            return doc

    def _write(self, c, doc: dict[str, Any]) -> None:
        doc["updated"] = draft_now()
        result = c.execute(sa.update(project).where(project.c.id == _key(doc["id"]), project.c.workspace_id == self.workspace_id)
                           .values(name=str(doc.get("name") or ""), industry=str(doc.get("industry") or "general"), doc=doc, updated=sa.func.now()))
        if result.rowcount == 0:
            raise KeyError(doc["id"])

    def save(self, doc: dict[str, Any]) -> dict[str, Any]:
        with self.engine.begin() as c:
            self._write(c, doc)
        return doc

    def update(self, project_id: str, **values: Any) -> dict[str, Any]:
        with self.engine.begin() as c:
            doc = c.execute(self._select(project_id, lock=True)).scalar()
            if doc is None:
                raise KeyError(project_id)
            doc.update(values)
            self._write(c, doc)
            return doc

    def list(self, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
        query = sa.select(project.c.doc).where(project.c.workspace_id == self.workspace_id).order_by(project.c.updated.desc(), project.c.id)
        if limit is not None:
            query = query.limit(limit).offset(offset)
        with self.engine.connect() as c:
            docs = [row[0] for row in c.execute(query)]
        for doc in docs:
            migrate(doc)
        return docs

    def delete(self, project_id: str) -> None:
        import shutil

        with self.engine.begin() as c:
            c.execute(sa.delete(project).where(project.c.id == _key(project_id), project.c.workspace_id == self.workspace_id))
        shutil.rmtree(self._folder(project_id), ignore_errors=True)

    # ------------------------------------------------------------------ stage records
    def _owned(self, c, project_id: str) -> None:
        if c.execute(self._select(project_id)).scalar() is None:
            raise KeyError(project_id)

    def write_stage(self, project_id: str, stage: str, record: dict[str, Any]) -> str:
        if stage not in STAGE_KEYS:
            raise KeyError(stage)
        with self.engine.begin() as c:
            self._owned(c, project_id)
            # strict, as the file store is: a stage record with a score that is not a number is a defect to surface, not to store as null
            insert = pg_insert(stage_record).values(project_id=project_id, stage=stage, record=json.loads(json.dumps(record, ensure_ascii=False, default=str, allow_nan=False)))
            c.execute(insert.on_conflict_do_update(index_elements=["project_id", "stage"], set_={"record": insert.excluded.record, "written": sa.func.now()}))
        return f"{project_id}/{stage}"

    def read_stage(self, project_id: str, stage: str) -> dict[str, Any] | None:
        if stage not in STAGE_KEYS:
            raise KeyError(stage)
        with self.engine.connect() as c:
            self._owned(c, project_id)
            return c.execute(sa.select(stage_record.c.record).where(stage_record.c.project_id == project_id, stage_record.c.stage == stage)).scalar()

    def has_stage(self, project_id: str, stage: str) -> bool:
        return self.read_stage(project_id, stage) is not None

    def records(self, project_id: str) -> dict[str, dict[str, Any]]:
        with self.engine.connect() as c:
            self._owned(c, project_id)
            rows = {r.stage: r.record for r in c.execute(sa.select(stage_record.c.stage, stage_record.c.record).where(stage_record.c.project_id == project_id))}
        return {stage: rows[stage] for stage in STAGE_KEYS if stage in rows}

    def clear_stages(self, project_id: str, from_stage: str = "data") -> None:
        """Downstream results are invalid once the data or the solution changes."""
        start = STAGE_KEYS.index(from_stage)
        with self.engine.begin() as c:
            doc = c.execute(self._select(project_id, lock=True)).scalar()
            if doc is None:
                raise KeyError(project_id)
            c.execute(sa.delete(stage_record).where(stage_record.c.project_id == project_id, stage_record.c.stage.in_(STAGE_KEYS[start:])))
            for stage in STAGE_KEYS[start:]:
                doc["stages"][stage] = {"status": "pending"}
            doc["decisions"] = {k: v for k, v in doc["decisions"].items() if STAGE_KEYS.index(k) < start} if doc["decisions"] else {}
            self._write(c, doc)

    # ------------------------------------------------------------------ append-only logs
    def log(self, project_id: str, kind: str, payload: Any, at: str | None = None) -> None:
        with self.engine.begin() as c:
            self._owned(c, project_id)
            c.execute(sa.insert(activity).values(project_id=project_id, at=at or draft_now(), kind=kind, payload=json.loads(db.dumps(payload))))

    def activity(self, project_id: str, limit: int = 60) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            self._owned(c, project_id)
            rows = c.execute(sa.select(activity.c.at, activity.c.kind, activity.c.payload).where(activity.c.project_id == project_id)
                             .order_by(activity.c.id.desc()).limit(limit)).all()
        return [{"at": r.at, "kind": r.kind, "payload": r.payload} for r in reversed(rows)]

    def transition(self, project_id: str, entry: dict[str, Any]) -> None:
        """Append one validated move (allowed or not). Written by ``graph.log``; never edited."""
        with self.engine.begin() as c:
            self._owned(c, project_id)
            c.execute(sa.insert(transition).values(project_id=project_id, at=entry.get("at"), move=entry.get("move"), status=entry.get("status"),
                                                   actor=entry.get("actor"), entry=json.loads(db.dumps(entry))))

    def transitions(self, project_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            self._owned(c, project_id)
            rows = c.execute(sa.select(transition.c.entry).where(transition.c.project_id == project_id).order_by(transition.c.id.desc()).limit(limit)).all()
        return [r[0] for r in reversed(rows)]


class _Turn:
    """One agent turn at a time per draft, across threads and processes: a session advisory lock, re-entrant per thread."""

    def __init__(self, engine: sa.Engine, draft_id: str, depth: threading.local):
        digest = hashlib.sha1(f"dclab-draft-turn:{draft_id}".encode()).digest()
        self.engine, self.key, self.depth, self.id = engine, int.from_bytes(digest[:8], "big", signed=True), depth, draft_id

    def __enter__(self):
        held = self.depth.__dict__.setdefault("held", {})
        if self.id not in held:
            connection = self.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
            connection.execute(sa.text("select pg_advisory_lock(:k)"), {"k": self.key})
            held[self.id] = [connection, 0]
        held[self.id][1] += 1
        return self

    def __exit__(self, *exc):
        held = self.depth.held
        held[self.id][1] -= 1
        if held[self.id][1] == 0:
            connection = held.pop(self.id)[0]
            try:
                connection.execute(sa.text("select pg_advisory_unlock(:k)"), {"k": self.key})
            finally:
                connection.close()
        return False

    def acquire(self, timeout: float | None = None) -> bool:  # the interface test's non-blocking probe
        import time

        deadline = time.time() + (timeout or 0)
        connection = self.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        try:
            while True:
                if connection.execute(sa.text("select pg_try_advisory_lock(:k)"), {"k": self.key}).scalar():
                    connection.execute(sa.text("select pg_advisory_unlock(:k)"), {"k": self.key})
                    return True
                if time.time() >= deadline:
                    return False
                time.sleep(0.02)
        finally:
            connection.close()


class PgDrafts(_Base):
    def __init__(self, workspace_id: str, files_home: Path, url: str | None = None):
        super().__init__(workspace_id, files_home, url)
        self._threads = threading.local()

    def data_dir(self, draft_id: str) -> Path:
        path = self.files_home / "drafts" / _key(draft_id) / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def turn(self, draft_id: str) -> _Turn:
        return _Turn(self.engine, _key(draft_id), self._threads)

    def _select(self, draft_id: str, lock: bool = False):
        query = sa.select(draft.c.doc).where(draft.c.id == _key(draft_id), draft.c.workspace_id == self.workspace_id)
        return query.with_for_update() if lock else query

    def _write(self, c, doc: dict[str, Any]) -> None:
        doc["updated"] = draft_now()
        result = c.execute(sa.update(draft).where(draft.c.id == _key(doc["id"]), draft.c.workspace_id == self.workspace_id)
                           .values(status=str(doc.get("status") or "open"), doc=doc, updated=sa.func.now()))
        if result.rowcount == 0:
            raise KeyError(doc["id"])

    def create(self, problem: str = "", pack: str | None = None) -> dict[str, Any]:
        doc = new_draft(secrets.token_hex(6), problem, pack)
        with self.engine.begin() as c:
            c.execute(sa.insert(draft).values(id=doc["id"], workspace_id=self.workspace_id, status=doc["status"], doc=doc))
        return doc

    def get(self, draft_id: str) -> dict[str, Any]:
        with self.engine.connect() as c:
            doc = c.execute(self._select(draft_id)).scalar()
        if doc is None:
            raise KeyError(draft_id)
        return doc

    def save(self, doc: dict[str, Any]) -> dict[str, Any]:
        with self.engine.begin() as c:
            self._write(c, doc)
        return doc

    def update(self, draft_id: str, fn: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        """Read, change with fn(draft) and save as one step, under a row lock (the pipeline and the chat both write)."""
        with self.engine.begin() as c:
            doc = c.execute(self._select(draft_id, lock=True)).scalar()
            if doc is None:
                raise KeyError(draft_id)
            fn(doc)
            self._write(c, doc)
            return doc

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            return [r[0] for r in c.execute(sa.select(draft.c.doc).where(draft.c.workspace_id == self.workspace_id)
                                            .order_by(draft.c.updated.desc(), draft.c.id).limit(limit))]

    def delete(self, draft_id: str) -> None:
        import shutil

        with self.engine.begin() as c:
            c.execute(sa.delete(draft).where(draft.c.id == _key(draft_id), draft.c.workspace_id == self.workspace_id))
        shutil.rmtree(self.files_home / "drafts" / draft_id, ignore_errors=True)

    # ------------------------------------------------------------------ the event log
    def emit(self, draft_id: str, kind: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Append one event; its number comes from an UPDATE of the draft row, so concurrent emits never share one."""
        with self.engine.begin() as c:
            seq = c.execute(sa.update(draft).where(draft.c.id == _key(draft_id), draft.c.workspace_id == self.workspace_id)
                            .values(event_seq=draft.c.event_seq + 1).returning(draft.c.event_seq)).scalar()
            if seq is None:
                raise KeyError(draft_id)
            event = {"seq": int(seq), "at": draft_now(), "kind": kind, "data": json.loads(db.dumps(data or {}))}
            c.execute(sa.insert(draft_event).values(draft_id=draft_id, **event))
        return event

    def events(self, draft_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            rows = c.execute(sa.select(draft_event.c.seq, draft_event.c.at, draft_event.c.kind, draft_event.c.data)
                             .join(draft, draft.c.id == draft_event.c.draft_id)
                             .where(draft_event.c.draft_id == _key(draft_id), draft.c.workspace_id == self.workspace_id, draft_event.c.seq > after)
                             .order_by(draft_event.c.seq)).all()
        return [{"seq": r.seq, "at": r.at, "kind": r.kind, "data": r.data} for r in rows]

    def events_version(self, draft_id: str) -> tuple[int, int]:
        with self.engine.connect() as c:
            seq = c.execute(sa.select(draft.c.event_seq).where(draft.c.id == _key(draft_id), draft.c.workspace_id == self.workspace_id)).scalar()
        return (int(seq or 0), 0)


class PgSessions(_Base):
    def _select(self, session_id: str):
        return sa.select(intern_session.c.doc).where(intern_session.c.id == _key(session_id), intern_session.c.workspace_id == self.workspace_id)

    def create(self, task: str, *, mode: str, model: str | None, budget: dict[str, Any] | None = None, project_id: str | None = None) -> dict[str, Any]:
        return self.save(new_session(uuid.uuid4().hex[:12], task, mode=mode, model=model, budget=budget, project_id=project_id))

    def get(self, session_id: str) -> dict[str, Any]:
        with self.engine.connect() as c:
            doc = c.execute(self._select(session_id)).scalar()
        if doc is None:
            raise KeyError(session_id)
        return doc

    def save(self, session: dict[str, Any]) -> dict[str, Any]:
        session["updated"] = draft_now()
        insert = pg_insert(intern_session).values(id=_key(session["id"]), workspace_id=self.workspace_id,
                                                                      status=str(session.get("status") or "queued"), doc=session)
        with self.engine.begin() as c:
            # an id owned by another workspace is never overwritten
            saved = c.execute(insert.on_conflict_do_update(index_elements=["id"], set_={"status": insert.excluded.status, "doc": insert.excluded.doc, "updated": sa.func.now()},
                                                           where=intern_session.c.workspace_id == self.workspace_id).returning(intern_session.c.id)).scalar()
            if saved is None:
                raise KeyError(session["id"])
        return session

    def list(self, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
        """Newest first, without the conversation and the steps (only how many steps there are)."""
        query = (sa.select((intern_session.c.doc.op("-")(sa.literal("messages")).op("-")(sa.literal("steps"))).label("doc"),
                           sa.func.coalesce(sa.func.jsonb_array_length(intern_session.c.doc["steps"]), 0).label("steps"))
                 .where(intern_session.c.workspace_id == self.workspace_id).order_by(intern_session.c.updated.desc(), intern_session.c.id))
        if limit is not None:
            query = query.limit(limit).offset(offset)
        with self.engine.connect() as c:
            return [{**r.doc, "steps": int(r.steps)} for r in c.execute(query)]

    def delete(self, session_id: str) -> None:
        with self.engine.begin() as c:
            c.execute(sa.delete(intern_session).where(intern_session.c.id == _key(session_id), intern_session.c.workspace_id == self.workspace_id))


def open_stores(home: Path, url: str | None = None) -> tuple[PgProjects, PgDrafts, PgSessions]:
    """The three stores for the workspace that lives in ``home`` (a folder keeps its files; a new folder is a new workspace)."""
    home = Path(home)
    wid = db.workspace_for(home, url)  # the folder's marker: a moved folder is the same workspace (package 9.3)
    return PgProjects(wid, home, url), PgDrafts(wid, home, url), PgSessions(wid, home, url)
