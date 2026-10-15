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


_QUIET: contextvars.ContextVar[bool] = contextvars.ContextVar("dclab_experience_quiet", default=False)


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Nothing recorded inside the block: work done on a scratch copy (a replay, 19.5) is not something that happened
    to the project; the caller records what it means (one ``run.recorded``) after the block."""
    token = _QUIET.set(True)
    try:
        yield
    finally:
        _QUIET.reset(token)


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
             since: str | None = None, until: str | None = None, user: str | None = None,
             kinds: Iterable[str] | None = None, visits: Iterable[str] | None = None, refused_only: bool = False,
             limit: int = 5000) -> list[contract.ExperienceEvent]: ...  # the newest ``limit``
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
        from ..filelock import held

        with self._lock, held(self.root):  # a deletion's rewrite (experience.retention) waits, and is waited for
            for event in events:
                path = self._path(event)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as out:
                    out.write(event.model_dump_json() + "\n")

    def _files(self) -> list[Path]:
        return sorted(self.root.glob("*/*/*/*.jsonl")) if self.root.exists() else []

    def read(self, *, visit: str | None = None, project: str | None = None, conversation: str | None = None,
             since: str | None = None, until: str | None = None, user: str | None = None,
             kinds: Iterable[str] | None = None, visits: Iterable[str] | None = None, refused_only: bool = False,
             limit: int = 5000) -> list[contract.ExperienceEvent]:
        q = _Query(visit, project, conversation, since, until, user, kinds, refused_only)
        among = frozenset(visits) if visits is not None else None
        out = []
        for path in self._files():
            if (visit is not None and path.stem != visit) or (among is not None and path.stem not in among) or not q.day_may_match(path):
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    event = contract.ExperienceEvent.read(line)
                except Exception:  # noqa: BLE001 - a line cut by a crash mid-write: the rest stands
                    continue
                if q.matches(event):
                    out.append(event)
        return contract.timeline(out)[-limit:]

    def visits(self, limit: int = 50) -> list[str]:
        latest: dict[str, float] = {}
        for path in self._files():
            if path.stem != "no-visit":
                latest[path.stem] = max(latest.get(path.stem, 0.0), path.stat().st_mtime)
        return sorted(latest, key=lambda v: -latest[v])[:limit]


def _utc(text: str) -> str:
    """A time as the log stores it: UTC ISO text of one width (a time without a zone is taken as UTC; a trailing Z is
    UTC too). A text that is not a time, or a time UTC cannot hold, raises ValueError."""
    try:
        moment = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"not a time: {text!r}") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    try:
        return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except OverflowError as exc:  # year 1 or 9999 with an offset
        raise ValueError(f"a time out of range: {text!r}") from exc


# A check said no: these kinds always, a move only when refused, a tool call only when denied (the session card's count).
REFUSED_KINDS = ("draft.refused", "solution.refused", "decision.refused")


def refused(event: Any) -> bool:
    if event.kind == "graph.move":
        return event.payload.get("allowed") is False
    if event.kind == "tool.call":
        return event.payload.get("verdict") == "denied"
    return event.kind in REFUSED_KINDS


class _Query:
    """What a reading asks for: ids that must be equal, a time window (``until`` excluded), kinds."""

    def __init__(self, visit, project, conversation, since, until, user, kinds, refused_only=False):
        self.ids = {"visit": visit, "project": project, "conversation": conversation, "user": user}
        self.refused_only = refused_only
        self.since = _utc(since) if since else None
        self.until = _utc(until) if until else None
        self.kinds = frozenset(kinds) if kinds is not None else None

    def day_may_match(self, path: Path) -> bool:
        """A file holds one UTC day (``YYYY/MM/DD/<visit>.jsonl``): the days outside the window are not opened."""
        day = f"{path.parent.parent.parent.name}-{path.parent.parent.name}-{path.parent.name}"
        return (self.since is None or day >= self.since[:10]) and (self.until is None or day <= self.until[:10])

    def matches(self, event: contract.ExperienceEvent) -> bool:
        at = _utc(event.at)
        return (all(want is None or getattr(event.ids, name) == want for name, want in self.ids.items())
                and (self.kinds is None or event.kind in self.kinds) and (not self.refused_only or refused(event))
                and (self.since is None or at >= self.since) and (self.until is None or at < self.until))


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
             since: str | None = None, until: str | None = None, user: str | None = None,
             kinds: Iterable[str] | None = None, visits: Iterable[str] | None = None, refused_only: bool = False,
             limit: int = 5000) -> list[contract.ExperienceEvent]:
        import sqlalchemy as sa

        from ..storage.models import experience_event as t

        q = sa.select(t.c.doc).where(t.c.workspace_id == self.workspace)
        for column, value in ((t.c.visit, visit), (t.c.project_id, project), (t.c.conversation, conversation), (t.c.user_id, user)):
            if value is not None:
                q = q.where(column == value)
        if kinds is not None:
            q = q.where(t.c.kind.in_(sorted(set(kinds))))
        if visits is not None:
            q = q.where(t.c.visit.in_(sorted(set(visits))))
        if refused_only:  # in the query, so allowed moves never crowd refusals out of a limited read
            payload = t.c.doc["payload"]
            q = q.where(sa.or_(t.c.kind.in_(REFUSED_KINDS),
                               sa.and_(t.c.kind == "graph.move", payload["allowed"].as_boolean().is_(False)),
                               sa.and_(t.c.kind == "tool.call", payload["verdict"].as_string() == "denied")))
        if since is not None:  # times are stored as UTC ISO text of one width, so text order is time order
            q = q.where(t.c.at >= _utc(since))
        if until is not None:
            q = q.where(t.c.at < _utc(until))
        q = q.order_by(t.c.at.desc(), t.c.seq.desc()).limit(limit)
        with self.engine.connect() as c:
            docs = [r[0] for r in c.execute(q)]
        return contract.timeline(contract.ExperienceEvent.read(d) for d in docs)  # a v1 row is read as v2

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

    log: Any = PgLog(workspace_id) if workspace_id and db.database_url(required=False) else FileLog(Path(home) / "experience", workspace_id or file_workspace_id(home))
    log.home = Path(home)  # 23.1: the folder whose consent.json stamps every record written here
    return log


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
    if target is None or contract is None or _QUIET.get():
        return None
    surface = _started(_IDS.get().get("visit")) if kind != "visit.started" else None
    if surface is not None:
        record("visit.started", "person", {"surface": surface}, log=target)
    resolved = _ids(ids)
    try:
        event = contract.make(kind, actor, {"workspace": target.workspace, **resolved}, payload or {},
                              id="ev" + secrets.token_hex(10), at=at, seq=next(_SEQ), consent=_consent_now(target, resolved.get("project")))
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


def _consent_now(log: Any, project: str | None) -> str:
    """The level this event is written under (23.1): from the workspace folder the log belongs to, and the project's."""
    from .. import consent as consents

    root = getattr(log, "root", None)
    home = getattr(log, "home", None) or (Path(root).parent if root is not None else None)
    try:
        if home is None:  # a log opened bare: only the current workspace's switches may stamp it
            from .. import context

            services = context.services()
            if services is None or getattr(log, "workspace", None) != getattr(services, "workspace_id", None):
                return "operate"
        return consents.current_level(home=home, project=project)
    except Exception:  # noqa: BLE001 - a stamp that cannot be read is the lowest: the event never leaves
        return "operate"


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


VISIT_EVENTS = 5000  # the events a card reads per visit; a longer visit's card says so (``cut``)


def visit_events(log: ExperienceLog, visits: list[str]) -> dict[str, list]:
    """Each visit's events, from one read of the log."""
    by_visit: dict[str, list] = {v: [] for v in visits}
    if visits:
        for event in log.read(visits=visits, limit=VISIT_EVENTS * len(visits)):
            if event.ids.visit in by_visit:
                by_visit[event.ids.visit].append(event)
    return by_visit


def cards(log: ExperienceLog, visits: list[str]) -> list[contract.ExperienceSession]:
    """The cards of these visits in this order, from one read of the log."""
    by_visit = visit_events(log, visits)
    return [contract.session_card(by_visit[v]) for v in visits if by_visit[v]]


def flags(events: list) -> list:
    """What the failure detectors find in one visit (19.9); none when the solving core cannot load."""
    try:
        from dclab_core import signals
    except ImportError:
        return []
    return signals.detect(events)


def sessions(log: ExperienceLog, limit: int = 50) -> list[contract.ExperienceSession]:
    """The cards of the latest visits, newest first."""
    return cards(log, log.visits(limit))


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
