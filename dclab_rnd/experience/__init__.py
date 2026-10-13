"""This studio's record of every session (packages 19.2 and 19.3): one append-only log per workspace, in the shape of
``dclab_core.experience`` (the contract DCLab's main product reads its own tables into).

Writing. ``record(kind, actor, payload)`` from anywhere: the ids come from the context (the workspace and its log from
``dclab_rnd.context``; the visit, the request, the person, and whatever a caller bound with :func:`bind`), so a writer
names only what it knows (a draft, a project, a run). Writing is best effort, as for the audit: an event that cannot be
written, or that the contract refuses (a credential, a value under a lower class), is reported on stderr and dropped;
the person's work goes on. Outside a workspace (a script, a test without one) nothing is written.

Where. PostgreSQL ``experience_event`` (append-only: a trigger refuses update and delete) when
``DCLAB_DATABASE_URL`` is set; otherwise ``experience/YYYY/MM/DD/<visit>.jsonl`` in the workspace folder, one file per
visit and day, so a session is one file to read.

Ids. The visit is minted by the server's middleware (a cookie, renewed while the person keeps working, a new visit
after 30 idle minutes) and carried into jobs with the request id; ``bind`` adds the conversation, the project, the run.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import itertools
import json
import os
import secrets
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Protocol

try:
    from dclab_core import experience as contract
except ImportError:  # an environment without the solving core's dependencies (the campaign's agent env): nothing is recorded
    contract = None  # type: ignore[assignment]

_IDS: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("dclab_experience_ids", default={})
# visits this web process minted that have recorded nothing yet: the first event of one is preceded by its
# visit.started (a poller that never acts leaves nothing; a worker process, which mints none, never repeats one)
_PENDING: dict[str, str] = {}
_PENDING_LOCK = threading.Lock()
PENDING_MAX = 10_000
_SEQ = itertools.count(1)
ID_FIELDS = ("user", "visit", "conversation", "project", "run", "job", "request", "model_request", "parent")
VISIT_IDLE_SECONDS = 30 * 60
DROPPED = {"count": 0}  # events the contract refused or the log could not take, since the process started


def new_visit() -> str:
    return "v" + secrets.token_hex(8)


@contextlib.contextmanager
def bind(**ids: Any) -> Iterator[dict[str, Any]]:
    """Ids every event recorded inside the block carries (``None`` values leave an outer one in place)."""
    merged = {**_IDS.get(), **{k: str(v) for k, v in ids.items() if v is not None and k in ID_FIELDS}}
    token = _IDS.set(merged)
    try:
        yield merged
    finally:
        _IDS.reset(token)


def mark_new(visit: str, surface: str) -> None:
    """A visit this process just minted: its visit.started waits for its first event."""
    with _PENDING_LOCK:
        if len(_PENDING) >= PENDING_MAX:
            _PENDING.pop(next(iter(_PENDING)))
        _PENDING[visit] = surface


def _started(visit: str | None) -> Optional[str]:
    if not visit:
        return None
    with _PENDING_LOCK:
        return _PENDING.pop(visit, None)


def bound() -> dict[str, Any]:
    return dict(_IDS.get())


def origin() -> dict[str, Any]:
    """What a job carries from the request that queued it: the visit and the request."""
    ids = _ids({})
    return {k: ids[k] for k in ("visit", "request") if ids.get(k)}


def _ids(extra: dict[str, Any]) -> dict[str, Any]:
    from .. import observe
    from ..accounts.principal import current

    ids = dict(_IDS.get())
    request = observe.REQUEST_ID.get()
    if request and "request" not in ids:
        ids["request"] = request
    person = current()
    if person is not None and person.user_id and "user" not in ids:
        ids["user"] = person.user_id
    ids.update({k: str(v) for k, v in extra.items() if v is not None and k in ID_FIELDS})
    return ids


# ---------------------------------------------------------------------- the log


class ExperienceLog(Protocol):
    workspace: str

    def append(self, events: list[contract.ExperienceEvent]) -> None: ...
    def read(self, *, visit: str | None = None, project: str | None = None, conversation: str | None = None,
             since: str | None = None, limit: int = 5000) -> list[contract.ExperienceEvent]: ...
    def visits(self, limit: int = 50) -> list[str]: ...  # newest first


_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


def file_workspace_id(home: Path) -> str:
    """The id a folder workspace goes by: the same formula PostgreSQL uses for it (storage.db.workspace), read without
    writing anything into the folder (its marker when there is one, else its path, as ``storage.db.workspace_key``)."""
    from ..storage.db import MARKER

    try:
        key = (Path(home) / MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        key = ""
    return "w" + hashlib.sha1((key or str(Path(home).resolve())).encode()).hexdigest()[:15]


class FileLog:
    """``experience/YYYY/MM/DD/<visit>.jsonl``: one line per event, only ever appended."""

    def __init__(self, root: Path, workspace: str):
        self.root, self.workspace = Path(root), workspace
        with _GUARD:
            self._lock = _LOCKS.setdefault(str(self.root.resolve()), threading.Lock())

    def _path(self, event: contract.ExperienceEvent) -> Path:
        day = datetime.fromisoformat(event.at)
        name = event.ids.visit or "no-visit"
        return self.root / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}" / f"{name}.jsonl"

    def append(self, events: list[contract.ExperienceEvent]) -> None:
        with self._lock:
            for event in events:
                path = self._path(event)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as out:
                    out.write(event.model_dump_json() + "\n")

    def _files(self) -> list[Path]:
        return sorted(self.root.glob("*/*/*/*.jsonl")) if self.root.exists() else []

    def read(self, *, visit: str | None = None, project: str | None = None, conversation: str | None = None,
             since: str | None = None, limit: int = 5000) -> list[contract.ExperienceEvent]:
        files = [f for f in self._files() if visit is None or f.stem == visit]
        out = []
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    event = contract.ExperienceEvent.read(line)
                except Exception:  # noqa: BLE001 - a line cut by a crash mid-write: the rest stands
                    continue
                if _matches(event, visit, project, conversation, since):
                    out.append(event)
        return contract.timeline(out)[-limit:]

    def visits(self, limit: int = 50) -> list[str]:
        latest: dict[str, float] = {}
        for path in self._files():
            if path.stem != "no-visit":
                latest[path.stem] = max(latest.get(path.stem, 0.0), path.stat().st_mtime)
        return sorted(latest, key=lambda v: -latest[v])[:limit]


def _matches(event: contract.ExperienceEvent, visit, project, conversation, since) -> bool:
    return ((visit is None or event.ids.visit == visit) and (project is None or event.ids.project == project)
            and (conversation is None or event.ids.conversation == conversation)
            and (since is None or datetime.fromisoformat(event.at) >= datetime.fromisoformat(since)))


class PgLog:
    """``experience_event``: the event whole in ``doc``, its ids in columns for finding it."""

    def __init__(self, workspace: str, engine: Any = None, url: str | None = None):
        from ..storage import db

        self.workspace, self.engine = workspace, engine if engine is not None else db.engine(url)

    def append(self, events: list[contract.ExperienceEvent]) -> None:
        import sqlalchemy as sa

        from ..storage.models import experience_event as t

        rows = [{"id": e.id, "workspace_id": self.workspace, "at": e.at, "seq": e.seq, "kind": e.kind, "actor": e.actor,
                 "visit": e.ids.visit, "user_id": e.ids.user, "conversation": e.ids.conversation, "project_id": e.ids.project,
                 "run_id": e.ids.run, "job_id": e.ids.job, "request_id": e.ids.request, "model_request": e.ids.model_request,
                 "doc": json.loads(e.model_dump_json())} for e in events]
        with self.engine.begin() as c:
            c.execute(sa.insert(t), rows)

    def read(self, *, visit: str | None = None, project: str | None = None, conversation: str | None = None,
             since: str | None = None, limit: int = 5000) -> list[contract.ExperienceEvent]:
        import sqlalchemy as sa

        from ..storage.models import experience_event as t

        q = sa.select(t.c.doc).where(t.c.workspace_id == self.workspace)
        for column, value in ((t.c.visit, visit), (t.c.project_id, project), (t.c.conversation, conversation)):
            if value is not None:
                q = q.where(column == value)
        if since is not None:  # times are stored as UTC ISO text of one width, so text order is time order
            q = q.where(t.c.at >= datetime.fromisoformat(since).astimezone(timezone.utc).isoformat(timespec="microseconds"))
        q = q.order_by(t.c.at.desc(), t.c.seq.desc()).limit(limit)
        with self.engine.connect() as c:
            docs = [r[0] for r in c.execute(q)]
        return contract.timeline(contract.ExperienceEvent.model_validate(d) for d in docs)

    def visits(self, limit: int = 50) -> list[str]:
        import sqlalchemy as sa

        from ..storage.models import experience_event as t

        q = (sa.select(t.c.visit, sa.func.max(t.c.at).label("last")).where(t.c.workspace_id == self.workspace, t.c.visit.is_not(None))
             .group_by(t.c.visit).order_by(sa.text("last desc")).limit(limit))
        with self.engine.connect() as c:
            return [r[0] for r in c.execute(q)]


def open_log(home: Path, workspace_id: str | None = None) -> ExperienceLog:
    """The log of the workspace in ``home``: PostgreSQL when it is configured, its folder otherwise."""
    from ..storage import db

    if workspace_id and db.database_url(required=False):
        return PgLog(workspace_id)
    return FileLog(Path(home) / "experience", workspace_id or file_workspace_id(home))


def current_log() -> Optional[ExperienceLog]:
    """The log of the workspace this request or job acts in, or None outside one."""
    from .. import context

    services = context.services()
    return getattr(services, "experience", None) if services is not None else _INSTALLED.get("log")


_INSTALLED: dict[str, Any] = {}


def install(log: Optional[ExperienceLog]) -> None:
    """The process's log when no workspace is set (the server's own, as for the model gateway)."""
    _INSTALLED["log"] = log


# ---------------------------------------------------------------------- writing


def record(kind: str, actor: str = "system", payload: dict[str, Any] | None = None, *, log: ExperienceLog | None = None,
           at: str | None = None, **ids: Any) -> Optional[contract.ExperienceEvent]:
    """Record one event in the current workspace's log; None when there is no log, or when it was refused or lost."""
    target = log if log is not None else current_log()
    if target is None or contract is None:
        return None
    surface = _started(_IDS.get().get("visit")) if kind != "visit.started" else None
    if surface is not None:
        record("visit.started", "person", {"surface": surface}, log=target)
    try:
        event = contract.make(kind, actor, {"workspace": target.workspace, **_ids(ids)}, payload or {},
                              id="ev" + secrets.token_hex(10), at=at, seq=next(_SEQ))
    except Exception as exc:  # noqa: BLE001 - the contract refused it (a credential, a value under a lower class)
        DROPPED["count"] += 1
        found = [line.strip() for line in str(exc).splitlines()]
        reason = next((line.removeprefix("Value error, ").split(" [type=")[0] for line in found if line.startswith("Value error")),
                      next((line for line in found if line and "pydantic" not in line), type(exc).__name__))
        print(f"experience: {kind} refused: {reason[:200]}", file=sys.stderr, flush=True)
        return None
    try:
        target.append([event])
    except Exception as exc:  # noqa: BLE001 - never stop a person's work because the log is briefly away
        DROPPED["count"] += 1
        print(f"experience: could not record {kind}: {type(exc).__name__}", file=sys.stderr, flush=True)
        return None
    return event


def model_text_allowed() -> bool:
    """Prompts and replies are kept only when the workspace keeps them: DCLAB_LOG_PROMPTS=1, the usage log's own switch."""
    return os.environ.get("DCLAB_LOG_PROMPTS") == "1"


_LAST_MODEL_REQUEST: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("dclab_last_model_request", default=None)


def new_model_request() -> str:
    """The id of a model request about to be recorded; the agent step it answers carries it (``last_model_request``)."""
    rid = "mr" + secrets.token_hex(8)
    _LAST_MODEL_REQUEST.set(rid)
    return rid


def last_model_request() -> Optional[str]:
    return _LAST_MODEL_REQUEST.get()


def sessions(log: ExperienceLog, limit: int = 50) -> list[contract.ExperienceSession]:
    """The cards of the latest visits, newest first."""
    out = []
    for visit in log.visits(limit):
        events = log.read(visit=visit)
        if events:
            out.append(contract.session_card(events))
    return out


def lines(events: Iterable[contract.ExperienceEvent]) -> list[str]:
    """A timeline as text a person can read: time, who, what, the ids that matter."""
    out = []
    for e in events:
        what = ", ".join(f"{k}={_short(v)}" for k, v in e.payload.items())
        where = " ".join(f"{k}:{v}" for k, v in (("draft/session", e.ids.conversation), ("project", e.ids.project), ("job", e.ids.job)) if v)
        out.append(f"{e.at[11:23]}  {e.actor:<6} {e.kind:<20} {what}  {where}".rstrip())
    return out


def _short(value: Any, n: int = 80) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")
