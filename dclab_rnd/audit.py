"""One audit trail per workspace (package 10.4): every governed action, append-only, with who, when and what.

    move            a validated move, allowed, refused or waiting for a person (studio/graph.py ``log``; the Home agent's guard)
    approval        a person approved a gate (``graph.approve_gate``)
    approval_used   the final stage used a holdout approval
    policy          a project's gate switch or training opt-in changed
    routing         a purpose moved to another model, or a shadow set (models/routing.py)
    solution        a project's or a draft's solution saved
    data_import     a table attached to a project, or a file brought into a draft, with its source and SHA-256
    lesson_review   a workspace lesson accepted or rejected
    memory_removed  a project memory note removed by a person
    sign_in, role_change   reserved for accounts (package 10.2)

A row is ``{id, at, kind, actor, who, user_id, project_id, project_name, draft_id, move, status, detail}``. The
application can only append and read: neither backend has an update or a delete, and in PostgreSQL a trigger refuses
both (``audit_event``). Files: ``audit.jsonl`` in the workspace folder. Writing is best effort for the action (an
audit that cannot be written is reported on stderr, never stops a person's work); reading is one query.

The per-project transition and activity logs stay as they are: trajectories, the ops page and the agents read them.
On the first open of a workspace whose audit is empty, the history those logs hold is imported once (``backfill``).
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

KINDS = ("move", "approval", "approval_used", "policy", "routing", "solution", "data_import", "lesson_review", "memory_removed",
         "sign_in", "role_change", "deletion", "retention")
FILTERS = ("kind", "status", "actor", "project_id", "draft_id")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def who_of(actor: str | None) -> str:
    return "Intern" if actor == "agent" else "Person"


class AuditLog(Protocol):
    def append(self, rows: list[dict[str, Any]]) -> None: ...
    def query(self, limit: int, offset: int, **filters: Any) -> tuple[int, list[dict[str, Any]]]: ...  # (total, newest first)
    def counts(self) -> dict[str, int]: ...  # by kind, and by status for moves ("status:blocked")
    def empty(self) -> bool: ...


def _row(kind: str, actor: str | None, who: str | None = None, *, project: dict[str, Any] | None = None, project_id: str | None = None,
         draft_id: str | None = None, move: str | None = None, status: str | None = None, user_id: str | None = None,
         at: str | None = None, **detail: Any) -> dict[str, Any]:
    assert kind in KINDS, kind
    if project is not None:
        project_id = project.get("id")
    from .accounts.principal import current

    person = current()  # the signed-in person (package 10.2); an agent's move is still the agent's
    if person is not None and person.user_id and (actor or "human") == "human":
        user_id = user_id or person.user_id
        who = who or person.who
    return {"at": at or now(), "kind": kind, "actor": actor or "human", "who": who or who_of(actor), "user_id": user_id,
            "project_id": project_id, "project_name": (project or {}).get("name"), "draft_id": draft_id, "move": move,
            "status": status, "detail": json.loads(json.dumps(detail, default=str))}


# ---------------------------------------------------------------------- files

_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


class FileAudit:
    """``audit.jsonl``: one line per row, only ever appended (a lock beside it serialises the server and a worker)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with _GUARD:
            self._lock = _LOCKS.setdefault(str(self.path.resolve()), threading.Lock())

    def _locked(self):
        import contextlib

        @contextlib.contextmanager
        def held():
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path.with_name(self.path.name + ".lock"), "a+") as lock:
                    try:
                        import fcntl

                        fcntl.flock(lock, fcntl.LOCK_EX)
                    except ImportError:
                        pass
                    yield
        return held()

    def _all(self) -> list[dict[str, Any]]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        rows = []
        for n, line in enumerate(lines, 1):  # a row's id is its line: the file is only ever appended to
            try:
                rows.append({"id": n, **json.loads(line)})
            except json.JSONDecodeError:  # a line cut by a crash mid-write: the rest stands
                continue
        return rows

    def append(self, rows: list[dict[str, Any]]) -> None:
        with self._locked():
            self._write(rows)

    def _write(self, rows: list[dict[str, Any]]) -> None:
        """Append rows (the caller holds the lock); after a line cut by a crash, the next row starts on its own line."""
        with self.path.open("a+b") as out:
            if out.tell():
                out.seek(-1, os.SEEK_END)
                if out.read(1) != b"\n":
                    out.write(b"\n")
            out.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))

    def query(self, limit: int, offset: int, **filters: Any) -> tuple[int, list[dict[str, Any]]]:
        rows = [r for r in reversed(self._all()) if all(v is None or r.get(k) == v for k, v in filters.items())]
        return len(rows), rows[offset:offset + limit]

    def counts(self) -> dict[str, int]:
        return _count(self._all())

    def empty(self) -> bool:
        try:
            return self.path.stat().st_size == 0
        except OSError:
            return True


def _count(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r["kind"]] = out.get(r["kind"], 0) + 1
        if r["kind"] == "move" and r.get("status"):
            out["status:" + r["status"]] = out.get("status:" + r["status"], 0) + 1
    return out


# ---------------------------------------------------------------------- PostgreSQL

class PgAudit:
    def __init__(self, workspace_id: str, engine: Any = None, url: str | None = None):
        from .storage import db

        self.workspace_id, self.engine = workspace_id, engine if engine is not None else db.engine(url)

    def append(self, rows: list[dict[str, Any]]) -> None:
        import sqlalchemy as sa

        from .storage.models import audit_event as t

        if rows:
            with self.engine.begin() as c:
                c.execute(sa.insert(t), [{"workspace_id": self.workspace_id, **r} for r in rows])

    def _where(self, q, filters: dict[str, Any]):
        from .storage.models import audit_event as t

        q = q.where(t.c.workspace_id == self.workspace_id)
        for key, value in filters.items():
            if value is not None:
                q = q.where(getattr(t.c, key) == value)
        return q

    def query(self, limit: int, offset: int, **filters: Any) -> tuple[int, list[dict[str, Any]]]:
        import sqlalchemy as sa

        from .storage.models import audit_event as t

        cols = [t.c.id, t.c.at, t.c.kind, t.c.actor, t.c.who, t.c.user_id, t.c.project_id, t.c.project_name, t.c.draft_id, t.c.move, t.c.status, t.c.detail]
        with self.engine.connect() as c:
            total = c.execute(self._where(sa.select(sa.func.count()).select_from(t), filters)).scalar() or 0
            rows = [dict(r._mapping) for r in c.execute(self._where(sa.select(*cols), filters).order_by(t.c.id.desc()).limit(limit).offset(offset))]
        return int(total), rows

    def counts(self) -> dict[str, int]:
        import sqlalchemy as sa

        from .storage.models import audit_event as t

        out: dict[str, int] = {}
        with self.engine.connect() as c:
            for kind, n in c.execute(self._where(sa.select(t.c.kind, sa.func.count()), {}).group_by(t.c.kind)):
                out[kind] = int(n)
            for status, n in c.execute(self._where(sa.select(t.c.status, sa.func.count()), {"kind": "move"}).group_by(t.c.status)):
                if status:
                    out["status:" + status] = int(n)
        return out

    def empty(self) -> bool:
        import sqlalchemy as sa

        from .storage.models import audit_event as t

        with self.engine.connect() as c:
            return c.execute(self._where(sa.select(t.c.id), {}).limit(1)).first() is None


# ---------------------------------------------------------------------- where a workspace's audit lives

def open_audit(home: Path) -> AuditLog:
    from .storage import db

    if db.database_url(required=False):
        return PgAudit(db.workspace_for(Path(home)))
    return FileAudit(Path(home) / "audit.jsonl")


def for_store(store: Any) -> AuditLog | None:
    """The audit of the workspace a project or draft store belongs to. A file store knows it only when it was opened for
    a workspace (``storage.open_stores`` sets ``audit_path``); a store made for anything else (a test, a script) writes none."""
    if getattr(store, "workspace_id", None):
        return PgAudit(store.workspace_id, getattr(store, "engine", None))
    path = getattr(store, "audit_path", None)
    return FileAudit(Path(path)) if path else None


def record(log: AuditLog | None, kind: str, actor: str | None = "human", who: str | None = None, **values: Any) -> None:
    """Append one row; an audit that cannot be written is reported, and the action it records still stands."""
    if log is None:
        return
    try:
        log.append([_row(kind, actor, who, **values)])
    except Exception as exc:  # noqa: BLE001 — never stop a person's work because the audit is briefly away
        print(f"audit: could not record {kind}: {type(exc).__name__}", file=sys.stderr, flush=True)


def record_for(store: Any, kind: str, actor: str | None = "human", who: str | None = None, **values: Any) -> None:
    try:
        log = for_store(store)
    except Exception:  # noqa: BLE001
        log = None
    record(log, kind, actor, who, **values)


def solution(store: Any, project_or_draft: dict[str, Any], value: dict[str, Any], actor: str, draft: bool = False) -> None:
    """A saved solution: its target, task, forbidden columns and hash (what a later review checks a run against)."""
    from .studio.graph import solution_hash

    where = {"draft_id": project_or_draft.get("id")} if draft else {"project": project_or_draft}
    record_for(store, "solution", actor, move="set_solution", status="allowed", **where, target=value.get("target"), task=value.get("task"),
               forbidden=[f.get("column") for f in value.get("forbidden") or [] if isinstance(f, dict)], solution_hash=solution_hash(value))


def sha256_of(path: Path) -> str | None:
    import hashlib

    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------- reading: the Admin audit tab

POLICY_NAMES = {"require_solution_signoff": "solution sign-off", "require_holdout_approval": "holdout approval",
                "share_for_training": "training opt-in"}


def item(row: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    """A row as the Admin page shows it (the shape the page read before 10.4, with the detail kept)."""
    d = dict(row.get("detail") or {})
    pid = row.get("project_id")
    out = {"id": row.get("id"), "kind": row["kind"], "at": row["at"], "actor": row["actor"], "who": row["who"], "user_id": row.get("user_id"),
           "move": row.get("move"), "args": d.pop("args", {}) or {}, "status": row.get("status"), "message": d.pop("message", "") or "",
           "outcome": d.pop("outcome", None), "rules": d.pop("rules", []) or [],
           "project": {"id": pid, "name": names.get(pid) or row.get("project_name") or pid} if pid else None,
           "draft_id": row.get("draft_id"), "detail": d}
    if row["kind"] == "move":
        out["from"], out["to"] = d.pop("from", None), d.pop("to", None)
    return out


def page(log: AuditLog, projects: Any, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
    total, rows = log.query(limit, offset, **{k: v for k, v in filters.items() if v not in (None, "")})
    plist = projects.list()
    names = {p["id"]: p.get("name") or p["id"] for p in plist}
    c = log.counts()
    counts = {"moves": c.get("move", 0), "approvals": c.get("approval", 0), "blocked": c.get("status:blocked", 0),
              "waiting": c.get("status:needs_approval", 0), "policy_changes": c.get("policy", 0), "routing_changes": c.get("routing", 0),
              "solutions": c.get("solution", 0), "data_imports": c.get("data_import", 0), "all": sum(v for k, v in c.items() if ":" not in k)}
    return {"total": total, "limit": limit, "offset": offset, "items": [item(r, names) for r in rows], "projects": len(plist), "counts": counts,
            "kinds": [k for k in KINDS if c.get(k)]}


# ---------------------------------------------------------------------- the history before 10.4, imported once

def history(projects: Any, routing: Any = None) -> list[dict[str, Any]]:
    """What the per-project logs and the routing history recorded before the audit table existed, oldest first."""
    from .studio import graph as studio_graph

    keyed: list[tuple[tuple[str, int], dict[str, Any]]] = []
    for p in projects.list():
        entries: list[dict[str, Any]] = []
        for m in projects.transitions(p["id"], 1_000_000):
            if m.get("move") == "approve_gate" and m.get("status") == "allowed":
                continue  # the approval below carries who, when and the reason
            entries.append(_row("move", m.get("actor"), project=p, move=m.get("move"), status=m.get("status"), at=m.get("at"),
                                args=m.get("args") or {}, message=m.get("message"), outcome=m.get("outcome"), rules=m.get("rules") or [],
                                **{"from": m.get("from"), "to": m.get("to")}, imported=True))
        for a in p.get("approvals") or []:
            gate = studio_graph.LEGACY_GATES.get(a.get("gate"), a.get("gate"))
            entries.append(_row("approval", "human", (a.get("by") or "owner").capitalize(), project=p, move="approve_gate", status="allowed",
                                at=a.get("at"), args={"gate": gate}, message=a.get("reason") or "",
                                rules=["DCLAB-R01"] if gate == "solution" else ["DCLAB-R17"], imported=True))
            if a.get("used"):
                entries.append(_row("approval_used", "agent", project=p, move="use_approval", status="allowed", at=a.get("used"),
                                    args={"gate": gate}, message="The final stage used this approval.", rules=["DCLAB-R17"], imported=True))
        for a in projects.activity(p["id"], 1_000_000):
            if a.get("kind") == "policy_changed":
                for key, change in ((a.get("payload") or {}).get("changes") or {}).items():
                    entries.append(_row("policy", "human", project=p, move="set_policy", status="allowed", at=a.get("at"),
                                        args={"policy": key, "label": POLICY_NAMES.get(key, key), "on": bool((change or {}).get("to"))}, imported=True))
        keyed += [((e["at"] or "", i), e) for i, e in enumerate(entries)]
    if routing is not None:
        try:
            changes = routing.get().get("history") or []
        except Exception:  # noqa: BLE001
            changes = []
        keyed += [((h.get("at") or "", i), _row("routing", "human", (h.get("by") or "person").capitalize(), move="set_routing", status="allowed",
                                                at=h.get("at"), args={k: h.get(k) for k in ("purpose", "setting", "from", "to")},
                                                message=h.get("reason") or "", imported=True))
                  for i, h in enumerate(changes)]
    keyed.sort(key=lambda pair: pair[0])
    return [e for _, e in keyed]


def backfill(log: AuditLog, projects: Any, routing: Any = None) -> int:
    """Import the history once, into an empty audit. Returns the rows imported (0 when the audit had any row)."""
    if isinstance(log, PgAudit):
        import hashlib

        import sqlalchemy as sa

        from .storage.models import audit_event as t

        key = int.from_bytes(hashlib.sha1(f"dclab-audit-backfill:{log.workspace_id}".encode()).digest()[:8], "big", signed=True)
        with log.engine.begin() as c:  # one instance imports; the others find rows and skip
            c.execute(sa.text("select pg_advisory_xact_lock(:k)"), {"k": key})
            if c.execute(sa.select(t.c.id).where(t.c.workspace_id == log.workspace_id).limit(1)).first() is not None:
                return 0
            rows = history(projects, routing)
            if rows:
                c.execute(sa.insert(t), [{"workspace_id": log.workspace_id, **r} for r in rows])
            return len(rows)
    with log._locked():
        if not log.empty():
            return 0
        rows = history(projects, routing)
        log._write(rows)
        return len(rows)


__all__ = ["FILTERS", "KINDS", "AuditLog", "FileAudit", "PgAudit", "backfill", "for_store", "history", "open_audit", "page", "record",
           "record_for", "sha256_of"]
