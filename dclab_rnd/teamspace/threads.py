"""Each team's own conversation on a problem (W3 of the collaborative workspace, 2026-11-20).

A **thread** is one team's private conversation on one problem of a project: one thread a (workspace, project, problem,
team). The team's people talk to their agent there, and what the other teams' agents send them arrives there. Its
**messages** are appended and never rewritten. Each keeps its words whole (the owner's rule: never cut before saving),
its author (a person, or a team's agent), when it was written, its visibility (``team``: the thread's readers only), and,
when the delivery outbox posted it, the handoff it carries (the W4 delivery, its task and what it carried).

Two places that behave the same, as the work between teams (``teamspace/store.py``):

- Files (``FileThreads``): ``teamspace/<project>/threads.json`` in the workspace folder (W4's project folder, beside
  ``work.json``), read and replaced whole under that folder's lock. One host only.
- PostgreSQL (``PgThreads``, migration 0028): ``team_thread`` and ``team_message`` rows of one workspace. A message's
  number comes from an UPDATE of its thread row, so two posts never share one; a trigger refuses rewriting a message.
  A post wakes the thread's event streams on every instance (``pg_notify`` on the drafts' channel, ``storage/notify.py``,
  keyed ``team_thread:<id>``, so the one LISTEN connection a server holds serves both).

**Who reads what** (:func:`access`): a thread's team members read it, and those whose role may write post to it; the
owner reads every thread of the workspace. Everyone else (another team, a viewer of the workspace outside the team) is
refused. Without sign-in (``DCLAB_AUTH`` unset) the local owner is the only person and reads and posts everywhere, as on
every other route. The project's readers get the shared handoff history instead (``store.handoff_history``): one line per
thing that travelled between teams, never a thread's own messages.

**The team of a thread** is a team's id (``storage/teams.py``). What travels between agents names a team agent
(``business``, ``data_science``; ``dclab_core.team_work``): :func:`team_key` maps that name to the workspace's one team of
that kind, and keeps the name itself when the workspace has no such team, or several (then only the owner reads that
thread until a team is chosen: a guess could show one team's words to another).

**A deleted project** keeps its threads and messages (the owner's rule: the store keeps deleted data; only the owner's
audited ``store forget`` removes anything). Nobody reads them through the routes any more: every route resolves the project
first (``memory.scope.resolve_scope``), which refuses a project that is gone, so its threads answer 403 like any other
project's (``tests/test_teamspace_routes.py``).

**The delivery hook** (:func:`post_delivery`, and :func:`deliver`, the outbox's default target): a delivered row is posted
into its recipient's thread, a reply into the thread its task came from, once per delivery id whatever the retries.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

Row = dict[str, Any]
FILE = "threads.json"
VISIBILITY = "team"
AUTHOR_KINDS = ("person", "agent")
NOTIFY_PREFIX = "team_thread:"
TEXT_MAX = 100_000  # a person's message; what an agent posts is kept whole whatever its length


class ThreadError(ValueError):
    """A message or a thread the store refuses: no words, an unknown author kind, an unknown thread."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _author(value: Mapping[str, Any]) -> Row:
    kind = str(value.get("kind") or "")
    if kind not in AUTHOR_KINDS:
        raise ThreadError(f"an author is a person or a team's agent, not {kind!r}")
    if not str(value.get("id") or "").strip():
        raise ThreadError("an author names who they are")
    out = {"kind": kind, "id": str(value["id"])}
    for k in ("team", "name"):
        if value.get(k):
            out[k] = str(value[k])
    return out


def _words(text: Any, author: Mapping[str, Any]) -> str:
    """The words whole, as written. A person's message says something; what an agent carries is posted as it travelled
    (a report's answer may be anything ``team_work`` accepted), only never nothing at all."""
    words = str(text if text is not None else "")
    if not (words.strip() if author.get("kind") == "person" else words):
        raise ThreadError("a message says something")
    return words


def _thread_doc(*, id: str, workspace_id: str, project_id: str, problem_id: str, team: str, label: str, by: Optional[Row]) -> Row:
    return {"id": id, "workspace_id": workspace_id, "project_id": project_id, "problem_id": problem_id, "team": team,
            "team_label": label, "created_at": _now(), "created_by": by, "messages": 0}


def _message(thread: Row, seq: int, author: Row, text: str, handoff: Optional[Row], key: Optional[str]) -> Row:
    return {"id": "tm_" + secrets.token_hex(8), "thread_id": thread["id"], "seq": seq, "author": author, "text": text,
            "handoff": handoff, "key": key, "visibility": VISIBILITY, "created_at": _now(),
            "project_id": thread["project_id"], "problem_id": thread["problem_id"], "team": thread["team"]}


def record_event(what: str, thread: Row, author: Optional[Row], state: str) -> None:
    """After the change is saved: one ``team.handoff`` event (W4's kind, ``what`` says which), names only, never the words."""
    from .. import experience

    who = author or {}
    experience.record("team.handoff", "person" if who.get("kind") == "person" else "agent",
                      {"what": what, "thread": thread["id"], "team": thread["team"],
                       "from_node": who.get("team") or who.get("kind") or "system", "to_node": thread["team"], "state": state},
                      project=thread["project_id"])


# ---------------------------------------------------------------------------------------------------------- files
class FileThreads:
    """``teamspace/<project>/threads.json``: the project's threads and their messages, behind the project folder's lock."""

    def __init__(self, home: Path, workspace: str = "local"):
        from .store import FileTeamWork

        self.home, self.workspace = Path(home), workspace
        self._folders = FileTeamWork(self.home)  # the same project folder W4 keeps its work in
        self.folder = self._folders.folder

    def location(self) -> str:
        return f"files in {self.folder}"

    def wait_for_events(self) -> None:
        return None  # one process writes the files: a stream polls

    @staticmethod
    def _load(path: Path) -> Row:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"threads": {}, "messages": {}}
        if not isinstance(data, dict) or not isinstance(data.get("threads"), dict) or not isinstance(data.get("messages"), dict):
            raise ThreadError("a project's threads file cannot be read: nothing was changed")
        return data

    @staticmethod
    def _write(folder: Path, data: Row) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".threads.", suffix=".part", dir=folder)
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
        os.replace(temporary, folder / FILE)

    @contextlib.contextmanager
    def _change(self, project_id: str) -> Iterator[Row]:
        from ..filelock import held

        folder = self._folders.project_dir(project_id)
        with held(folder):
            data = self._load(folder / FILE)
            data["project_id"] = project_id
            before = json.dumps(data, sort_keys=True)
            yield data  # an exception leaves the file as it was
            if json.dumps(data, sort_keys=True) != before:
                self._write(folder, data)

    def _documents(self, project_id: Optional[str] = None) -> list[Row]:
        from ..filelock import held

        folders = [self._folders.project_dir(project_id)] if project_id is not None else \
            (sorted(p.parent for p in self.folder.glob(f"*/{FILE}")) if self.folder.is_dir() else [])
        out = []
        for folder in folders:
            if (folder / FILE).is_file():
                with held(folder):
                    out.append(self._load(folder / FILE))
        return out

    def _locate(self, thread_id: str) -> Optional[Row]:
        for data in self._documents():
            if thread_id in data["threads"]:
                return data
        return None

    # threads
    def open_thread(self, project_id: str, problem_id: str, team: str, *, label: str = "", by: Optional[Mapping[str, Any]] = None) -> tuple[Row, bool]:
        """The team's thread on this problem, made the first time (whether it was made now)."""
        with self._change(project_id) as data:
            found = next((t for t in data["threads"].values() if (t["problem_id"], t["team"]) == (problem_id, team)), None)
            if found is not None:
                return copy.deepcopy(found), False
            thread = _thread_doc(id="th_" + secrets.token_hex(8), workspace_id=self.workspace, project_id=project_id,
                                 problem_id=problem_id, team=team, label=label or team, by=_author(by) if by else None)
            data["threads"][thread["id"]] = thread
            data["messages"][thread["id"]] = []
        return copy.deepcopy(thread), True

    def thread(self, thread_id: str) -> Optional[Row]:
        data = self._locate(thread_id)
        return copy.deepcopy(data["threads"][thread_id]) if data is not None else None

    def find_thread(self, project_id: str, problem_id: str, team: str) -> Optional[Row]:
        for data in self._documents(project_id):
            for t in data["threads"].values():
                if (t["problem_id"], t["team"]) == (problem_id, team):
                    return copy.deepcopy(t)
        return None

    def threads(self, *, project_id: Optional[str] = None, problem_id: Optional[str] = None, teams: Optional[Sequence[str]] = None) -> list[Row]:
        wanted = set(teams) if teams is not None else None
        out = [copy.deepcopy(t) for data in self._documents(project_id) for t in data["threads"].values()
               if (problem_id is None or t["problem_id"] == problem_id) and (wanted is None or t["team"] in wanted)]
        return sorted(out, key=lambda t: (t["created_at"], t["id"]))

    # messages
    def post(self, thread_id: str, author: Mapping[str, Any], text: Any, *, handoff: Optional[Mapping[str, Any]] = None,
             key: Optional[str] = None) -> tuple[Row, bool]:
        """Append a message (whether it was added now). A ``key`` already posted in this project returns that message."""
        who = _author(author)
        words = _words(text, who)
        located = self._locate(thread_id)
        if located is None:
            raise KeyError(thread_id)
        with self._change(located["project_id"]) as data:
            thread = data["threads"].get(thread_id)
            if thread is None:
                raise KeyError(thread_id)
            if key:
                for messages in data["messages"].values():
                    held = next((m for m in messages if m.get("key") == key), None)
                    if held is not None:
                        return copy.deepcopy(held), False
            thread["messages"] = int(thread.get("messages") or 0) + 1
            message = _message(thread, thread["messages"], who, words, dict(handoff) if handoff else None, key)
            data["messages"].setdefault(thread_id, []).append(message)
            shown = copy.deepcopy(thread)
        record_event("thread.message", shown, who, "posted")
        return copy.deepcopy(message), True

    def messages(self, thread_id: str, after: int = 0) -> list[Row]:
        data = self._locate(thread_id)
        if data is None:
            return []
        return [copy.deepcopy(m) for m in data["messages"].get(thread_id, []) if int(m["seq"]) > int(after)]

    def counts(self) -> dict[str, int]:
        docs = self._documents()
        return {"threads": sum(len(d["threads"]) for d in docs), "messages": sum(len(m) for d in docs for m in d["messages"].values())}


# ---------------------------------------------------------------------------------------------------------- PostgreSQL
class PgThreads:
    """The ``team_thread`` and ``team_message`` rows of one workspace."""

    def __init__(self, workspace_id: str, url: Optional[str] = None):
        from ..storage import db

        self.workspace, self.engine = workspace_id, db.engine(url)

    @property
    def workspace_id(self) -> str:
        return self.workspace

    def location(self) -> str:
        return f"PostgreSQL workspace {self.workspace}"

    def wait_for_events(self):
        """The server's listener (the drafts' one): it wakes a thread's streams when any instance posts to it."""
        from ..storage.notify import listener

        return listener(self.engine.url.render_as_string(hide_password=False))

    @staticmethod
    def _iso(value: Any) -> str:
        return value.astimezone(timezone.utc).isoformat(timespec="milliseconds") if hasattr(value, "astimezone") else str(value)

    def _thread_row(self, r: Any) -> Row:
        return {**r.doc, "id": r.id, "workspace_id": self.workspace, "project_id": r.project_id, "problem_id": r.problem_id,
                "team": r.team, "messages": int(r.message_seq)}

    def _select_threads(self, c: Any, *where: Any) -> list[Row]:
        import sqlalchemy as sa

        from ..storage.models import team_thread as t

        q = sa.select(t.c.id, t.c.project_id, t.c.problem_id, t.c.team, t.c.message_seq, t.c.doc).where(t.c.workspace_id == self.workspace, *where)
        return [self._thread_row(r) for r in c.execute(q.order_by(t.c.created, t.c.id))]

    def open_thread(self, project_id: str, problem_id: str, team: str, *, label: str = "", by: Optional[Mapping[str, Any]] = None) -> tuple[Row, bool]:
        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert

        from ..storage.models import team_thread as t

        doc = _thread_doc(id="th_" + secrets.token_hex(8), workspace_id=self.workspace, project_id=project_id, problem_id=problem_id,
                          team=team, label=label or team, by=_author(by) if by else None)
        with self.engine.begin() as c:
            made = c.execute(insert(t).values(id=doc["id"], workspace_id=self.workspace, project_id=project_id, problem_id=problem_id, team=team,
                                              doc={k: doc[k] for k in ("team_label", "created_at", "created_by")})
                             .on_conflict_do_nothing(index_elements=[t.c.workspace_id, t.c.project_id, t.c.problem_id, t.c.team])
                             .returning(t.c.id)).first()
            found = self._select_threads(c, t.c.project_id == project_id, t.c.problem_id == problem_id, t.c.team == team)
        return found[0], made is not None

    def thread(self, thread_id: str) -> Optional[Row]:
        from ..storage.models import team_thread as t

        with self.engine.connect() as c:
            found = self._select_threads(c, t.c.id == thread_id)
        return found[0] if found else None

    def find_thread(self, project_id: str, problem_id: str, team: str) -> Optional[Row]:
        from ..storage.models import team_thread as t

        with self.engine.connect() as c:
            found = self._select_threads(c, t.c.project_id == project_id, t.c.problem_id == problem_id, t.c.team == team)
        return found[0] if found else None

    def threads(self, *, project_id: Optional[str] = None, problem_id: Optional[str] = None, teams: Optional[Sequence[str]] = None) -> list[Row]:
        from ..storage.models import team_thread as t

        where = []
        if project_id is not None:
            where.append(t.c.project_id == project_id)
        if problem_id is not None:
            where.append(t.c.problem_id == problem_id)
        if teams is not None:
            if not teams:
                return []
            where.append(t.c.team.in_(list(teams)))
        with self.engine.connect() as c:
            return self._select_threads(c, *where)

    def _message_row(self, r: Any) -> Row:
        return {**r.doc, "id": r.id, "thread_id": r.thread_id, "seq": int(r.seq), "key": r.key, "project_id": r.project_id}

    def _by_key(self, c: Any, project_id: str, key: str) -> Optional[Row]:
        import sqlalchemy as sa

        from ..storage.models import team_message as m

        r = c.execute(sa.select(m.c.thread_id, m.c.seq, m.c.id, m.c.key, m.c.project_id, m.c.doc)
                      .where(m.c.workspace_id == self.workspace, m.c.project_id == project_id, m.c.key == key)).first()
        return self._message_row(r) if r is not None else None

    def post(self, thread_id: str, author: Mapping[str, Any], text: Any, *, handoff: Optional[Mapping[str, Any]] = None,
             key: Optional[str] = None) -> tuple[Row, bool]:
        import sqlalchemy as sa
        from sqlalchemy.exc import IntegrityError

        from ..storage.models import team_message as m, team_thread as t
        from ..storage.notify import CHANNEL

        who = _author(author)
        words = _words(text, who)
        thread = self.thread(thread_id)
        if thread is None:
            raise KeyError(thread_id)
        try:
            with self.engine.begin() as c:
                if key:
                    held = self._by_key(c, thread["project_id"], key)
                    if held is not None:
                        return held, False
                seq = c.execute(sa.update(t).where(t.c.id == thread_id, t.c.workspace_id == self.workspace)
                                .values(message_seq=t.c.message_seq + 1).returning(t.c.message_seq)).scalar()
                if seq is None:
                    raise KeyError(thread_id)
                message = _message(thread, int(seq), who, words, dict(handoff) if handoff else None, key)
                doc = {k: v for k, v in message.items() if k not in ("id", "thread_id", "seq", "key", "project_id")}
                c.execute(sa.insert(m).values(thread_id=thread_id, seq=int(seq), id=message["id"], workspace_id=self.workspace,
                                              project_id=thread["project_id"], key=key, doc=doc))
                # delivered when this transaction commits, to every instance's streams on this thread
                c.execute(sa.text("select pg_notify(:channel, :key)"), {"channel": CHANNEL, "key": NOTIFY_PREFIX + thread_id})
        except IntegrityError:
            if not key:
                raise
            with self.engine.connect() as c:  # the same delivery posted at the same moment by another worker: theirs stands
                held = self._by_key(c, thread["project_id"], key)
            if held is None:
                raise
            return held, False
        thread["messages"] = int(seq)
        record_event("thread.message", thread, who, "posted")
        return message, True

    def messages(self, thread_id: str, after: int = 0) -> list[Row]:
        import sqlalchemy as sa

        from ..storage.models import team_message as m

        with self.engine.connect() as c:
            rows = c.execute(sa.select(m.c.thread_id, m.c.seq, m.c.id, m.c.key, m.c.project_id, m.c.doc)
                             .where(m.c.workspace_id == self.workspace, m.c.thread_id == thread_id, m.c.seq > int(after)).order_by(m.c.seq))
            return [self._message_row(r) for r in rows]

    def counts(self) -> dict[str, int]:
        import sqlalchemy as sa

        from ..storage.models import team_message as m, team_thread as t

        with self.engine.connect() as c:
            return {name: int(c.execute(sa.select(sa.func.count()).select_from(x).where(x.c.workspace_id == self.workspace)).scalar())
                    for name, x in (("threads", t), ("messages", m))}


def import_project(store: "PgThreads", data: Row) -> dict[str, int]:
    """One project's ``threads.json`` copied into the database workspace (``storage migrate``): each thread with its id,
    each message with its number, author, words, handoff and key. A thread or message the database already holds is left
    as it is (a second run copies nothing); a thread's last number is its highest message's."""
    import sqlalchemy as sa
    from sqlalchemy.dialects.postgresql import insert

    from ..storage.models import team_message as m, team_thread as t

    copied = {"threads": 0, "messages": 0}
    with store.engine.begin() as c:
        for thread in (data.get("threads") or {}).values():
            messages = sorted((data.get("messages") or {}).get(thread["id"], []), key=lambda x: int(x["seq"]))
            last = max([int(x["seq"]) for x in messages] + [int(thread.get("messages") or 0)])
            made = c.execute(insert(t).values(id=thread["id"], workspace_id=store.workspace, project_id=thread["project_id"],
                                              problem_id=thread["problem_id"], team=thread["team"], message_seq=last,
                                              doc={k: thread.get(k) for k in ("team_label", "created_at", "created_by")})
                             .on_conflict_do_nothing().returning(t.c.id)).first()
            copied["threads"] += made is not None
            for message in messages:
                doc = {k: v for k, v in message.items() if k not in ("id", "thread_id", "seq", "key", "project_id")}
                done = c.execute(insert(m).values(thread_id=thread["id"], seq=int(message["seq"]), id=message["id"],
                                                  workspace_id=store.workspace, project_id=thread["project_id"], key=message.get("key"),
                                                  doc=doc).on_conflict_do_nothing().returning(sa.literal(1))).first()
                copied["messages"] += done is not None
    return copied


def open_threads(home: Path, url: Optional[str] = None) -> "FileThreads | PgThreads":
    """The team threads of the workspace in ``home``: PostgreSQL when DCLAB_DATABASE_URL is set, as the other stores
    choose, else the folder's files (workspace ``local``, as ``memory.scope.workspace_of`` names it)."""
    from ..storage import db

    if url or db.database_url(required=False):
        return PgThreads(db.workspace_for(Path(home), url), url)
    return FileThreads(Path(home))


# ---------------------------------------------------------------------------------------------------------- whose thread
def team_key(teams: Any, name: str) -> tuple[str, str]:
    """The thread team for a team agent's name, and how people call it: a team's id as it is; the workspace's one team of
    that kind; else the name itself (no such team, or several: only the owner reads that thread)."""
    from .store import team_label

    try:
        found = teams.teams()
    except Exception:  # noqa: BLE001 — no teams to ask (a broken file): the agent's name, which only the owner reads
        found = []
    same_id = [t for t in found if t["id"] == name]
    if same_id:
        return same_id[0]["id"], same_id[0]["name"]
    of_kind = [t for t in found if t.get("kind") == name]
    if len(of_kind) == 1:
        return of_kind[0]["id"], of_kind[0]["name"]
    return name, team_label(name)


def access(thread: Mapping[str, Any], principal: Any, teams: Any, *, auth_on: bool) -> Optional[str]:
    """What this person may do on this thread: ``"write"`` (read and post), ``"read"``, or None (refused).

    Without sign-in, the local owner: write. With it: a member of the thread's team reads it, and posts when their role
    may write; the owner reads every thread (and posts in their own teams'). Anyone else: None."""
    from ..accounts.roles import allows

    if not auth_on:
        return None if principal is not None and getattr(principal, "via", None) == "anonymous" else "write"
    uid, role = getattr(principal, "user_id", None), getattr(principal, "role", None)
    if not uid or not role:
        return None
    try:
        member = bool(teams.is_member(str(thread["team"]), str(uid)))
    except Exception:  # noqa: BLE001 — the teams cannot be read: no one is a member
        member = False
    if member:
        return "write" if allows(role, "write") else "read"
    return "read" if role == "owner" else None


# ---------------------------------------------------------------------------------------------------------- the delivery hook
def _home(env: Any) -> Optional[Path]:
    from .delivery import _home as home_of

    return home_of(env)


def threads_for(env: Any) -> Any:
    """The env's threads (``env.team_threads`` when it has them), else those of its workspace folder."""
    held = getattr(env, "team_threads", None)
    if held is not None and hasattr(held, "post"):
        return held
    home = _home(env)
    if home is None:
        raise LookupError("no workspace to post in")
    return open_threads(home)


def _teams_for(env: Any) -> Any:
    held = getattr(env, "teams", None)
    if held is not None and hasattr(held, "is_member"):
        return held
    from ..storage import open_teams

    return open_teams(_home(env))


def _recipients(teams: Any, team: str, key: str, row: Mapping[str, Any]) -> bool:
    """Whether the thread a row names belongs to its recipient team: the team the row's name maps to, the name itself, or
    another team of that kind (a workspace with two Business teams) that the member who started the task is in. Without
    sign-in and with no member named, a team of that kind is enough (the local owner is the only person)."""
    to_team = str(row["to_team"])
    if team == key:
        return True
    if team == to_team:  # a thread named after the agent (only the owner reads it): only while no real team of that kind is
        return key == to_team  # there; once one is, the reply goes to that team's thread, where its members read it
    try:
        found = teams.team(team)
    except Exception:  # noqa: BLE001 — the teams cannot be read: never a guess
        return False
    if found is None or found.get("kind") != to_team:
        return False
    who = (row.get("initiator") or {}).get("user_id")
    if who:
        try:
            return bool(teams.is_member(team, str(who)))
        except Exception:  # noqa: BLE001
            return False
    from ..memory.scope import auth_enabled

    return not auth_enabled()


def post_delivery(env: Any, row: Mapping[str, Any]) -> Row:
    """Post a delivered row where it lands, once per delivery id: a reply (or a clarification) into the thread its task came
    from when that thread is the recipient team's, else into the recipient team's thread on the problem (opened the first
    time). The words are the row's, whole; the author is the sending team's agent. Returns where it went: ``thread_id``,
    ``message_id``, ``seq`` and whether it was ``posted`` now (False: it was already there)."""
    store, teams = threads_for(env), _teams_for(env)
    key, label = team_key(teams, str(row["to_team"]))
    target = None
    if row.get("thread_id"):
        named = store.thread(str(row["thread_id"]))
        if named is not None and (named["project_id"], named["problem_id"]) == (row["project_id"], row["problem_id"]) \
                and _recipients(teams, named["team"], key, row):
            target = named
    if target is None:
        target, made = store.open_thread(str(row["project_id"]), str(row["problem_id"]), key, label=label,
                                         by={"kind": "agent", "id": str(row["from_team"]), "team": str(row["from_team"])})
        if made:
            record_event("thread.opened", target, {"kind": "agent", "team": str(row["from_team"])}, "open")
    handoff = {"delivery_id": row["id"], "kind": row.get("kind"), "task_id": row.get("task_id"), "item_id": row.get("item_id"),
               "from_team": row.get("from_team"), "to_team": row.get("to_team")}
    message, posted = store.post(target["id"], {"kind": "agent", "id": str(row["from_team"]), "team": str(row["from_team"])},
                                 row["text"], handoff=handoff, key=f"delivery:{row['id']}")
    return {"thread_id": target["id"], "message_id": message["id"], "seq": message["seq"], "posted": posted}


def deliver(env: Any, row: Mapping[str, Any]) -> Row:
    """The outbox's default target (``delivery.set_target`` replaces it; W5's agents call :func:`post_delivery` themselves):
    the row is the recipient's inbox, and it is posted into its thread. An env with no workspace only acknowledges it."""
    from .delivery import PermanentFailure

    receipt: Row = {"inbox": row["to_team"]}
    if getattr(env, "team_threads", None) is None and _home(env) is None:
        return receipt
    try:
        return {**receipt, **post_delivery(env, row)}
    except ThreadError as exc:  # the row as it is can never be posted: blocked at once, never retried
        raise PermanentFailure(str(exc)) from None


__all__ = ["AUTHOR_KINDS", "FileThreads", "import_project", "NOTIFY_PREFIX", "PgThreads", "ThreadError", "VISIBILITY", "access", "deliver", "record_event",
           "open_threads", "post_delivery", "team_key", "threads_for"]
