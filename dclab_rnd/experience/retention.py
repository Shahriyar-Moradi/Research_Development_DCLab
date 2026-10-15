"""What is kept, for how long, and what a deletion removes from the record (package 19.10).

- **Deleting a project or a draft** removes every event it left in the session record, every decision (with its
  labels) in the ledger and the prompts the gateway's usage log kept for it; **deleting a session** removes its
  events. What stays is the audit's tombstone (what was deleted, by whom, when, and what could not be removed; never
  what it held).
- **Retention per class**, the owner's choice (confirmed 2026-10-15): events are deleted after 730 days
  (``DCLAB_RETENTION_EVENT_DAYS``), prompts and replies are removed from their events after 365 days
  (``DCLAB_RETENTION_MODEL_TEXT_DAYS``, the rest of the event kept). Either setting changes its number; ``0`` turns it
  off.

On PostgreSQL the append-only triggers let only these functions through (``SET LOCAL dclab.privileged_delete``,
migration 0017); on files the day files are rewritten without what is removed.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

MODEL_TEXT = ("prompt", "reply")


DEFAULT_DAYS = {"DCLAB_RETENTION_EVENT_DAYS": 730, "DCLAB_RETENTION_MODEL_TEXT_DAYS": 365}  # the owner's choice, 2026-10-15


def days(name: str) -> Optional[int]:
    """The days a class is kept: its setting, or the owner's default when unset; None when off (0) or unreadable."""
    raw = (os.environ.get(name) or "").strip()
    try:
        value = int(raw) if raw else DEFAULT_DAYS.get(name, 0)
    except ValueError:
        return None
    return value if value > 0 else None


def _privileged(connection: Any) -> None:
    import sqlalchemy as sa

    connection.execute(sa.text("set local dclab.privileged_delete = 'on'"))


# ---------------------------------------------------------------------- the session record

def _rewrite(root: Path, keep: Callable[[dict[str, Any]], Optional[dict[str, Any]]], lock: Optional[Path] = None) -> int:
    """Rewrite every jsonl file under ``root``: each line through ``keep`` (None drops it). Returns lines changed. It
    holds the folder's lock (``filelock.held`` on ``lock``, by default ``root``), the one every appender holds, so no
    line appended meanwhile is lost and two rewrites never cross."""
    from ..filelock import held

    if not root.exists():
        return 0
    with held(lock or root):
        return _rewrite_held(root, keep)


def _rewrite_held(root: Path, keep: Callable[[dict[str, Any]], Optional[dict[str, Any]]]) -> int:
    changed = 0
    for path in sorted(root.glob("**/*.jsonl")):
        lines, out, touched = path.read_text(encoding="utf-8").splitlines(), [], False
        for line in lines:
            try:
                doc = json.loads(line)
            except ValueError:
                out.append(line)
                continue
            new = keep(doc)
            if new is None:
                touched, changed = True, changed + 1
            elif new is not doc:
                out.append(json.dumps(new, ensure_ascii=False, separators=(",", ":")))
                touched, changed = True, changed + 1
            else:
                out.append(line)
        if touched:
            tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
            tmp.write_text("".join(x + "\n" for x in out), encoding="utf-8")
            tmp.replace(path)
            if not out:
                path.unlink()
    return changed


def delete_events(log: Any, *, project: str | None = None, visit: str | None = None, conversation: str | None = None) -> int:
    """Every event of a project, a visit or a conversation, removed from the record. Returns how many."""
    if not (project or visit or conversation):
        raise ValueError("name what to delete: a project, a visit or a conversation")
    want = {"project": project, "visit": visit, "conversation": conversation}
    if hasattr(log, "engine"):
        import sqlalchemy as sa

        from ..storage.models import experience_event as t

        q = sa.delete(t).where(t.c.workspace_id == log.workspace)
        for column, value in ((t.c.project_id, project), (t.c.visit, visit), (t.c.conversation, conversation)):
            if value is not None:
                q = q.where(column == value)
        with log.engine.begin() as c:
            _privileged(c)
            return int(c.execute(q).rowcount or 0)
    return _rewrite(Path(log.root), lambda d: None if all(v is None or (d.get("ids") or {}).get(k) == v for k, v in want.items()) else d)


def sweep_events(log: Any, now: Optional[datetime] = None, event_days: Optional[int] = None, text_days: Optional[int] = None) -> dict[str, int]:
    """Retention by class: events past ``event_days`` deleted, prompts and replies past ``text_days`` removed."""
    now = now or datetime.now(timezone.utc)
    event_days = event_days if event_days is not None else days("DCLAB_RETENTION_EVENT_DAYS")
    text_days = text_days if text_days is not None else days("DCLAB_RETENTION_MODEL_TEXT_DAYS")
    out = {"events_deleted": 0, "texts_removed": 0}
    if not event_days and not text_days:
        return out
    old = (now - timedelta(days=event_days)).isoformat(timespec="microseconds") if event_days else None
    stale = (now - timedelta(days=text_days)).isoformat(timespec="microseconds") if text_days else None
    if hasattr(log, "engine"):
        import sqlalchemy as sa

        from ..storage.models import experience_event as t

        with log.engine.begin() as c:
            _privileged(c)
            if old:
                out["events_deleted"] = int(c.execute(sa.delete(t).where(t.c.workspace_id == log.workspace, t.c.at < old)).rowcount or 0)
            if stale:
                rows = c.execute(sa.select(t.c.id, t.c.doc).where(t.c.workspace_id == log.workspace, t.c.kind == "model.request", t.c.at < stale)).all()
                for rid, doc in rows:
                    payload = doc.get("payload") or {}
                    if any(k in payload for k in MODEL_TEXT):
                        c.execute(sa.update(t).where(t.c.id == rid).values(doc=_without_text(doc)))
                        out["texts_removed"] += 1
        return out

    def keep(doc: dict[str, Any]) -> Optional[dict[str, Any]]:
        at = str(doc.get("at") or "")
        if old and at < old:
            out["events_deleted"] += 1
            return None
        if stale and at < stale and doc.get("kind") == "model.request" and any(k in (doc.get("payload") or {}) for k in MODEL_TEXT):
            out["texts_removed"] += 1
            return _without_text(doc)
        return doc

    _rewrite(Path(log.root), keep)
    return out


def _without_text(doc: dict[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in (doc.get("payload") or {}).items() if k not in MODEL_TEXT}
    classes = {k: v for k, v in (doc.get("classes") or {}).items() if k not in MODEL_TEXT}
    removed = sorted(set(doc.get("redacted") or []) | {f"{k}:model_text" for k in MODEL_TEXT if k in (doc.get("payload") or {})})
    return {**doc, "payload": payload, "classes": classes, "redacted": removed}


# ---------------------------------------------------------------------- the decision ledger

def delete_decisions(ledger: Any, *, project: str | None = None, draft: str | None = None) -> int:
    """Every decision of a project or a draft, with its labels, removed from the ledger. Returns how many decisions."""
    if not (project or draft):
        raise ValueError("name what to delete: a project or a draft")
    if hasattr(ledger, "engine"):
        import sqlalchemy as sa

        from ..storage.models import decision_label as labels, decision_outcome as outcomes, decision_record as t

        q = sa.select(t.c.id).where(t.c.workspace_id == ledger.workspace)
        q = q.where(t.c.project_id == project) if project else q.where(t.c.draft_id == draft)
        with ledger.engine.begin() as c:
            _privileged(c)
            ids = [r[0] for r in c.execute(q)]
            for start in range(0, len(ids), 5000):
                chunk = ids[start:start + 5000]
                c.execute(sa.delete(labels).where(labels.c.workspace_id == ledger.workspace, labels.c.record_id.in_(chunk)))
                c.execute(sa.delete(outcomes).where(outcomes.c.workspace_id == ledger.workspace, outcomes.c.record_id.in_(chunk)))  # 23.5
                c.execute(sa.delete(t).where(t.c.id.in_(chunk)))
            if project:  # the rows of the project's drafts' decisions carry the project too
                c.execute(sa.delete(outcomes).where(outcomes.c.workspace_id == ledger.workspace, outcomes.c.project_id == project))
            return len(ids)
    gone: set[str] = set()

    def records(doc: dict[str, Any]) -> Optional[dict[str, Any]]:
        if (project and doc.get("project") == project) or (draft and doc.get("draft") == draft):
            gone.add(doc.get("id"))
            return None
        return doc

    from ..filelock import held

    with held(Path(ledger.root)):  # the ledger's appenders hold its root: both folders under one lock
        _rewrite_held(Path(ledger.root) / "decisions", records) if (Path(ledger.root) / "decisions").exists() else None
        _rewrite_held(Path(ledger.root) / "labels", lambda d: None if d.get("record") in gone else d) if (Path(ledger.root) / "labels").exists() else None
        _rewrite_held(Path(ledger.root) / "outcomes", lambda d: None if d.get("record") in gone or (project and d.get("project") == project) else d) \
            if (Path(ledger.root) / "outcomes").exists() else None
    return len(gone)


# ---------------------------------------------------------------------- what the delete routes call

def forget(*, project: str | None = None, draft: str | None = None, visit: str | None = None, log: Any = None, ledger: Any = None,
           usage: Any = None) -> dict[str, Any]:
    """Everything a deleted project, draft or session left in this workspace's record and ledger, removed, and the
    prompts the gateway's usage log kept for a project or a draft; how many of each. Never raises: the deletion it
    follows stands, and ``failed`` says what could not be removed (the tombstone carries it)."""
    import sys

    from .. import experience
    from ..ledger import current_ledger

    log = log if log is not None else experience.current_log()
    ledger = ledger if ledger is not None else current_ledger()
    out: dict[str, Any] = {"events": 0, "decisions": 0, "prompts": 0}
    for name, step in (("events", lambda: delete_events(log, project=project, conversation=draft, visit=visit) if log is not None else 0),
                       ("decisions", lambda: delete_decisions(ledger, project=project, draft=draft) if ledger is not None and (project or draft) else 0),
                       ("prompts", lambda: usage.forget_prompts(project_id=project, draft_id=draft) if usage is not None and (project or draft) else 0)):
        try:
            out[name] = step()
        except Exception as exc:  # noqa: BLE001
            out.setdefault("failed", []).append(name)
            print(f"retention: could not remove the {name} of what was deleted: {type(exc).__name__}", file=sys.stderr, flush=True)
    return out
