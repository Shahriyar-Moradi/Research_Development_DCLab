"""Editable Project and Problem bases, idempotent material event capture, and provisional drafts (Package A4).

Every confirmed fact is written both to the base ledger (tracking subject, field, source, optimistic revision, and
dependencies) and to the append-only ``MemoryStore`` (``dclab_core.memory.MemoryItem`` v2) so existing brief, recall,
and upkeep paths read the exact same truth.

Rules enforced here:
- Explicit user edits require write authorization and support optimistic concurrency (``expected_revision`` -> conflict).
- Material decisions and consequential outcomes are idempotent by ``(source_type, source_id, source_revision, scope)``.
- AI interpretations and Advisor suggestions stay provisional (``status="draft"``); retries never activate them.
- Corrections and retirements preserve history, increment revision, mark dependent drafts/advice stale, and invalidate
  derived summaries.
- Raw rows, secrets, credentials, protected benchmark columns, and trivial greetings are never stored.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from dclab_core.memory import MemoryItem, check, new, retire

from . import Refused, event, latest, open_memory, remember, retire_scope, secret
from .readers import line as clean_line
from .scope import AccessDenied, AuthorizedScope

BASES_DIR = Path("memory") / "bases"
OUTCOME_STATUSES = ("proposed", "succeeded", "blocked", "failed", "stopped")
GREETING_RE = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|got it|sure|bye|goodbye|سلام|ممنون|مرسی)[!.\s]*$", re.I)


class RevisionConflictError(ValueError):
    """Raised when an optimistic base edit's expected_revision does not match the current revision."""

    def __init__(self, *, scope: str, scope_id: str, expected_revision: int, current_revision: int,
                 current_entries: list[dict[str, Any]], attempted: dict[str, Any]):
        self.scope = scope
        self.scope_id = scope_id
        self.expected_revision = expected_revision
        self.current_revision = current_revision
        self.current_entries = list(current_entries)
        self.attempted = dict(attempted)
        super().__init__(
            f"Revision conflict on {scope} base ({scope_id}): expected revision {expected_revision}, "
            f"current revision is {current_revision}. Reload and reconcile both edits."
        )

    def to_detail(self) -> dict[str, Any]:
        return {
            "error": "revision_conflict",
            "scope": self.scope,
            "scope_id": self.scope_id,
            "expected_revision": self.expected_revision,
            "current_revision": self.current_revision,
            "current_entries": self.current_entries,
            "attempted": self.attempted,
        }


class StaleDependencyError(ValueError):
    """Raised when confirming or executing against a draft or decision whose depended base revision changed."""

    def __init__(self, message: str, *, changed_sources: list[dict[str, Any]]):
        self.changed_sources = list(changed_sources)
        super().__init__(message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _base_file(home: Path, scope_kind: str, scope_id: str) -> Path:
    digest = hashlib.sha256(f"{scope_kind}:{scope_id}".encode("utf-8")).hexdigest()[:24]
    return Path(home) / BASES_DIR / f"{scope_kind}_{digest}.json"


def _lock_dir(home: Path) -> Path:
    return Path(home) / BASES_DIR


def _scope_target(scope: AuthorizedScope, target: str) -> tuple[str, str]:
    if target == "project":
        if not scope.project_id or not scope.project_key:
            raise AccessDenied("authorized project_id is required for project base")
        return "project", scope.project_key
    if target == "problem":
        if not (scope.project_id and scope.problem_id and scope.problem_key):
            raise AccessDenied("authorized project_id and problem_id are required for problem base")
        return "problem", scope.problem_key
    raise ValueError(f"unsupported base target {target!r}; expected 'project' or 'problem'")


def _empty_doc(scope_kind: str, scope_id: str, scope: AuthorizedScope) -> dict[str, Any]:
    return {
        "schema_version": "dclab.memory_base.v1",
        "scope": scope_kind,
        "scope_id": scope_id,
        "workspace_id": scope.workspace_id,
        "project_id": scope.project_id,
        "problem_id": scope.problem_id if scope_kind == "problem" else None,
        "revision": 0,
        "updated_at": _now(),
        "entries": [],
        "events_index": {},
    }


def _read_doc_unlocked(path: Path, scope_kind: str, scope_id: str, scope: AuthorizedScope) -> dict[str, Any]:
    if not path.is_file():
        return _empty_doc(scope_kind, scope_id, scope)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and doc.get("scope") == scope_kind and doc.get("scope_id") == scope_id:
            doc.setdefault("entries", [])
            doc.setdefault("events_index", {})
            return doc
    except (OSError, ValueError):
        pass
    return _empty_doc(scope_kind, scope_id, scope)


def _write_doc_unlocked(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.part")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _validate_text(text: Any, protected: Iterable[str] = ()) -> str:
    cleaned = clean_line(text)
    if not cleaned:
        raise Refused("memory text must contain at least one visible word")
    if secret(cleaned):
        raise Refused("looks like a key, a token or a password: never in memory")
    # Also enforce protected column name check via dclab_core.memory.named.
    from dclab_core.memory import named

    found = named(cleaned, tuple(protected))
    if found:
        raise Refused(f"names protected column(s): {', '.join(found)}")
    return cleaned


def _mark_dependents_stale(doc: dict[str, Any], *, changed_field: str, old_entry_id: Optional[str],
                           new_entry_id: Optional[str], new_revision: int, reason: str) -> list[str]:
    """Mark provisional drafts or pending decisions whose base revision is older than new_revision as stale."""
    affected: list[str] = []
    change_info = {
        "field": changed_field,
        "old_entry_id": old_entry_id,
        "new_entry_id": new_entry_id,
        "new_revision": new_revision,
        "reason": reason,
    }
    for ent in doc.get("entries") or []:
        if ent.get("status") == "retired":
            continue
        dep = ent.get("depends_on") or {}
        dep_rev = dep.get("base_revision")
        dep_items = set(dep.get("entry_ids") or []) | set(dep.get("item_ids") or [])
        if ent.get("status") == "draft" or ent.get("kind") == "episode":
            if (dep_rev is not None and int(dep_rev) < new_revision) or (old_entry_id and old_entry_id in dep_items):
                ent["stale"] = True
                ent["stale_reason"] = reason or f"{changed_field} changed at revision {new_revision}"
                changes = list(ent.get("changed_sources") or [])
                changes.append(change_info)
                ent["changed_sources"] = changes[-8:]
                affected.append(str(ent["entry_id"]))
    return affected


def _invalidate_summary_best_effort(home: Path, scope: AuthorizedScope) -> None:
    try:
        from . import summary as summary_mod

        summary_mod.invalidate_summary(home, scope.workspace_id, scope.project_id, scope.problem_id)
    except Exception:  # noqa: BLE001
        pass


def _write_memory_item(home: Path, item: MemoryItem, *, protected: Iterable[str] = (), log: Any = None,
                       retire_old_id: Optional[str] = None, by: str = "person:owner", reason: str = "") -> MemoryItem:
    store = open_memory(home)
    written = remember(store, item, protected)
    event(log, "memory.written" if item.status != "draft" else "memory.proposed", written, reason=reason)
    if retire_old_id and retire_old_id != written.id:
        held = next((v for v in latest(store.log()) if v.id == retire_old_id), None)
        if held is not None and held.status != "retired":
            store.write(retire(held, by=written.id))
            event(log, "memory.retired", held, reason=reason or "superseded", by=by, by_item=written.id)
    return written


# ------------------------------------------------------------------ Public Read & Write Operations

def read_base(home: Path, scope: AuthorizedScope, target: str = "problem") -> dict[str, Any]:
    """Read an authorized Project or Problem base (active entries, pending drafts, retired history, current revision)."""
    from ..filelock import held

    scope_kind, scope_id = _scope_target(scope, target)
    path = _base_file(home, scope_kind, scope_id)
    lock_parent = _lock_dir(home)
    if path.is_file():
        with held(lock_parent):
            doc = _read_doc_unlocked(path, scope_kind, scope_id, scope)
    else:
        doc = _empty_doc(scope_kind, scope_id, scope)
    entries = list(doc.get("entries") or [])
    return {
        "scope": scope_kind,
        "scope_id": scope_id,
        "workspace_id": scope.workspace_id,
        "project_id": scope.project_id,
        "problem_id": scope.problem_id if scope_kind == "problem" else None,
        "revision": int(doc.get("revision") or 0),
        "updated_at": doc.get("updated_at"),
        "active": [dict(e) for e in entries if e.get("status") == "active"],
        "drafts": [dict(e) for e in entries if e.get("status") == "draft"],
        "retired": [dict(e) for e in entries if e.get("status") == "retired"],
    }


def base_revision(home: Path, scope: AuthorizedScope, target: str = "problem") -> int:
    """Current integer revision of an authorized base (0 when never edited)."""
    return int(read_base(home, scope, target)["revision"])


def upsert_base_fact(
    home: Path,
    scope: AuthorizedScope,
    *,
    target: str = "problem",
    subject: str,
    field: str,
    text: str,
    source: str,
    expected_revision: Optional[int] = None,
    reason: str = "",
    protected: Iterable[str] = (),
    log: Any = None,
) -> dict[str, Any]:
    """Explicit authorized user edit to Project or Problem base with optimistic concurrency control."""
    from ..filelock import held

    if not scope.can_write:
        raise AccessDenied(f"role {scope.role!r} cannot edit {target} base")
    scope_kind, scope_id = _scope_target(scope, target)
    cleaned = _validate_text(text, protected)
    subj = clean_line(subject or target)[:80] or target
    fld = clean_line(field or "fact")[:80] or "fact"
    src = clean_line(source or f"ui:{target}_base")[:180]
    author = f"person:{scope.principal_id}"
    path = _base_file(home, scope_kind, scope_id)

    with held(_lock_dir(home)):
        doc = _read_doc_unlocked(path, scope_kind, scope_id, scope)
        cur_rev = int(doc.get("revision") or 0)
        active_entries = [dict(e) for e in doc["entries"] if e.get("status") == "active"]
        if expected_revision is not None and int(expected_revision) != cur_rev:
            raise RevisionConflictError(
                scope=scope_kind, scope_id=scope_id,
                expected_revision=int(expected_revision), current_revision=cur_rev,
                current_entries=active_entries,
                attempted={"subject": subj, "field": fld, "text": cleaned, "source": src, "author": author},
            )

        # Find an existing active fact for the same (subject, field), if any.
        existing = next(
            (e for e in doc["entries"] if e.get("status") == "active" and e.get("kind") == "fact"
             and e.get("subject") == subj and e.get("field") == fld),
            None,
        )
        if existing is not None and existing.get("text") == cleaned:
            return {"entry": dict(existing), "revision": cur_rev, "unchanged": True, "stale_dependents": []}

        new_rev = cur_rev + 1
        now_ts = _now()
        is_correction = existing is not None
        mem_item = new(
            kind="fact",
            scope=scope_kind,
            scope_id=scope_id,
            text=cleaned,
            source=src,
            who=author,
            at=now_ts,
            status="active",
            correction=is_correction,
            supersedes=existing["item_id"] if existing is not None else None,
        )
        written_item = _write_memory_item(
            home, mem_item, protected=protected, log=log,
            retire_old_id=existing["item_id"] if existing is not None else None,
            by=author, reason=reason or (f"updated {fld}" if is_correction else f"added {fld}"),
        )

        entry_id = "ent_" + secrets.token_hex(5)
        old_entry_id = None
        if existing is not None:
            old_entry_id = existing["entry_id"]
            existing["status"] = "retired"
            existing["retired_by"] = entry_id
            existing["updated_at"] = now_ts

        entry = {
            "entry_id": entry_id,
            "item_id": written_item.id,
            "kind": "fact",
            "subject": subj,
            "field": fld,
            "text": written_item.text,
            "source": src,
            "source_type": "user_edit",
            "source_id": entry_id,
            "source_revision": str(new_rev),
            "author": author,
            "status": "active",
            "review_state": "confirmed",
            "correction": is_correction,
            "prior_revision": cur_rev,
            "effective_revision": new_rev,
            "supersedes": old_entry_id,
            "retired_by": None,
            "depends_on": {},
            "uncertainty": "",
            "stale": False,
            "stale_reason": "",
            "changed_sources": [],
            "outcome_status": None,
            "created_at": now_ts,
            "updated_at": now_ts,
        }
        doc["entries"].append(entry)
        doc["revision"] = new_rev
        doc["updated_at"] = now_ts
        stale_ids = _mark_dependents_stale(
            doc, changed_field=fld, old_entry_id=old_entry_id, new_entry_id=entry_id,
            new_revision=new_rev, reason=reason or f"{fld} updated to revision {new_rev}",
        )
        _write_doc_unlocked(path, doc)

    _invalidate_summary_best_effort(home, scope)
    return {"entry": dict(entry), "revision": new_rev, "unchanged": False, "stale_dependents": stale_ids}


def correct_base_fact(
    home: Path,
    scope: AuthorizedScope,
    entry_id: str,
    new_text: str,
    *,
    target: str = "problem",
    expected_revision: Optional[int] = None,
    reason: str = "",
    protected: Iterable[str] = (),
    log: Any = None,
) -> dict[str, Any]:
    """Correct a specific active base fact by entry_id; preserves history, bumps revision, invalidates dependents."""
    from ..filelock import held

    if not scope.can_write:
        raise AccessDenied(f"role {scope.role!r} cannot correct {target} base")
    scope_kind, scope_id = _scope_target(scope, target)
    cleaned = _validate_text(new_text, protected)
    author = f"person:{scope.principal_id}"
    path = _base_file(home, scope_kind, scope_id)

    with held(_lock_dir(home)):
        doc = _read_doc_unlocked(path, scope_kind, scope_id, scope)
        cur_rev = int(doc.get("revision") or 0)
        active_entries = [dict(e) for e in doc["entries"] if e.get("status") == "active"]
        if expected_revision is not None and int(expected_revision) != cur_rev:
            raise RevisionConflictError(
                scope=scope_kind, scope_id=scope_id,
                expected_revision=int(expected_revision), current_revision=cur_rev,
                current_entries=active_entries,
                attempted={"entry_id": entry_id, "text": cleaned, "author": author, "reason": reason},
            )
        old = next((e for e in doc["entries"] if e.get("entry_id") == entry_id), None)
        if old is None:
            raise KeyError(entry_id)
        if old.get("status") != "active":
            raise Refused(f"only an active base entry can be corrected; status is {old.get('status')!r}")
        if old.get("source_type") == "run_record" or old.get("kind") == "episode":
            raise Refused("a measured run result or historical episode cannot be rewritten by a correction")
        if old.get("text") == cleaned:
            raise Refused("a correction must change the words")

        new_rev = cur_rev + 1
        now_ts = _now()
        mem_item = new(
            kind="fact",
            scope=scope_kind,
            scope_id=scope_id,
            text=cleaned,
            source=f"correction:{old['item_id']}",
            who=author,
            at=now_ts,
            status="active",
            correction=True,
            supersedes=old["item_id"],
        )
        written_item = _write_memory_item(
            home, mem_item, protected=protected, log=log,
            retire_old_id=old["item_id"], by=author, reason=reason or "corrected",
        )
        new_id = "ent_" + secrets.token_hex(5)
        old["status"] = "retired"
        old["retired_by"] = new_id
        old["updated_at"] = now_ts

        entry = {
            "entry_id": new_id,
            "item_id": written_item.id,
            "kind": "fact",
            "subject": old.get("subject", target),
            "field": old.get("field", "fact"),
            "text": written_item.text,
            "source": f"correction:{old['entry_id']}",
            "source_type": "user_correction",
            "source_id": new_id,
            "source_revision": str(new_rev),
            "author": author,
            "status": "active",
            "review_state": "confirmed",
            "correction": True,
            "prior_revision": cur_rev,
            "effective_revision": new_rev,
            "supersedes": old["entry_id"],
            "retired_by": None,
            "depends_on": {},
            "uncertainty": "",
            "stale": False,
            "stale_reason": "",
            "changed_sources": [],
            "outcome_status": None,
            "created_at": now_ts,
            "updated_at": now_ts,
        }
        doc["entries"].append(entry)
        doc["revision"] = new_rev
        doc["updated_at"] = now_ts
        stale_ids = _mark_dependents_stale(
            doc, changed_field=str(old.get("field") or "fact"), old_entry_id=old["entry_id"], new_entry_id=new_id,
            new_revision=new_rev, reason=reason or f"{old.get('field', 'fact')} corrected at revision {new_rev}",
        )
        _write_doc_unlocked(path, doc)

    _invalidate_summary_best_effort(home, scope)
    return {"entry": dict(entry), "superseded": dict(old), "revision": new_rev, "stale_dependents": stale_ids}


def retire_base_fact(
    home: Path,
    scope: AuthorizedScope,
    entry_id: str,
    *,
    target: str = "problem",
    expected_revision: Optional[int] = None,
    reason: str = "",
    log: Any = None,
) -> dict[str, Any]:
    """Retire an active base fact; keeps history, bumps revision, invalidates dependents and summaries."""
    from ..filelock import held

    if not scope.can_write:
        raise AccessDenied(f"role {scope.role!r} cannot retire from {target} base")
    scope_kind, scope_id = _scope_target(scope, target)
    author = f"person:{scope.principal_id}"
    path = _base_file(home, scope_kind, scope_id)

    with held(_lock_dir(home)):
        doc = _read_doc_unlocked(path, scope_kind, scope_id, scope)
        cur_rev = int(doc.get("revision") or 0)
        active_entries = [dict(e) for e in doc["entries"] if e.get("status") == "active"]
        if expected_revision is not None and int(expected_revision) != cur_rev:
            raise RevisionConflictError(
                scope=scope_kind, scope_id=scope_id,
                expected_revision=int(expected_revision), current_revision=cur_rev,
                current_entries=active_entries,
                attempted={"entry_id": entry_id, "action": "retire", "author": author, "reason": reason},
            )
        old = next((e for e in doc["entries"] if e.get("entry_id") == entry_id), None)
        if old is None:
            raise KeyError(entry_id)
        if old.get("status") == "retired":
            return {"entry": dict(old), "revision": cur_rev, "stale_dependents": []}

        new_rev = cur_rev + 1
        now_ts = _now()
        store = open_memory(home)
        held_item = next((v for v in latest(store.log()) if v.id == old["item_id"]), None)
        if held_item is not None and held_item.status != "retired":
            store.write(retire(held_item))
            event(log, "memory.retired", held_item, reason=reason or "retired by user", by=author)

        old["status"] = "retired"
        old["updated_at"] = now_ts
        doc["revision"] = new_rev
        doc["updated_at"] = now_ts
        stale_ids = _mark_dependents_stale(
            doc, changed_field=str(old.get("field") or "fact"), old_entry_id=old["entry_id"], new_entry_id=None,
            new_revision=new_rev, reason=reason or f"{old.get('field', 'fact')} retired at revision {new_rev}",
        )
        _write_doc_unlocked(path, doc)

    _invalidate_summary_best_effort(home, scope)
    return {"entry": dict(old), "revision": new_rev, "stale_dependents": stale_ids}


def record_material_event(
    home: Path,
    scope: AuthorizedScope,
    *,
    target: str = "problem",
    source_type: str,
    source_id: str,
    source_revision: str,
    text: str,
    outcome_status: str = "succeeded",
    who: Optional[str] = None,
    depends_on_revision: Optional[int] = None,
    protected: Iterable[str] = (),
    log: Any = None,
) -> dict[str, Any]:
    """Capture a consequential decision or measured outcome idempotently; skips greetings and replays cleanly."""
    from ..filelock import held

    if outcome_status not in OUTCOME_STATUSES:
        raise ValueError(f"outcome_status must be one of {OUTCOME_STATUSES}")
    raw = str(text or "").strip()
    if not raw or GREETING_RE.match(raw):
        return {"skipped": True, "reason": "non_material_text", "replayed": False}

    scope_kind, scope_id = _scope_target(scope, target)
    cleaned = _validate_text(raw, protected)
    idem_key = f"{source_type}:{source_id}:{source_revision}:{scope_kind}:{scope_id}"
    author = who or f"agent:{scope.role}"
    if not author.startswith(("person:", "agent:", "rule:")):
        author = f"agent:{author}"
    path = _base_file(home, scope_kind, scope_id)

    with held(_lock_dir(home)):
        doc = _read_doc_unlocked(path, scope_kind, scope_id, scope)
        cur_rev = int(doc.get("revision") or 0)
        existing_id = doc["events_index"].get(idem_key)
        if existing_id:
            existing = next((e for e in doc["entries"] if e.get("entry_id") == existing_id), None)
            if existing is not None:
                return {"entry": dict(existing), "revision": cur_rev, "replayed": True, "skipped": False}

        now_ts = _now()
        src_ref = clean_line(f"{source_type}:{source_id}@rev{source_revision}")[:180]
        mem_item = new(
            kind="episode",
            scope=scope_kind,
            scope_id=scope_id,
            text=cleaned,
            source=src_ref,
            who=author,
            at=now_ts,
            status="active",
        )
        written_item = _write_memory_item(home, mem_item, protected=protected, log=log, by=author, reason=f"material:{source_type}")
        entry_id = "ent_" + secrets.token_hex(5)
        dep_rev = cur_rev if depends_on_revision is None else int(depends_on_revision)
        entry = {
            "entry_id": entry_id,
            "item_id": written_item.id,
            "kind": "episode",
            "subject": source_type,
            "field": source_type,
            "text": written_item.text,
            "source": src_ref,
            "source_type": source_type,
            "source_id": source_id,
            "source_revision": str(source_revision),
            "author": author,
            "status": "active",
            "review_state": "recorded",
            "correction": False,
            "prior_revision": cur_rev,
            "effective_revision": cur_rev,
            "supersedes": None,
            "retired_by": None,
            "depends_on": {"base_revision": dep_rev},
            "uncertainty": "",
            "stale": dep_rev < cur_rev,
            "stale_reason": ("recorded against older revision" if dep_rev < cur_rev else ""),
            "changed_sources": [],
            "outcome_status": outcome_status,
            "created_at": now_ts,
            "updated_at": now_ts,
        }
        doc["entries"].append(entry)
        doc["events_index"][idem_key] = entry_id
        doc["updated_at"] = now_ts
        _write_doc_unlocked(path, doc)

    _invalidate_summary_best_effort(home, scope)
    return {"entry": dict(entry), "revision": cur_rev, "replayed": False, "skipped": False}


def propose_draft(
    home: Path,
    scope: AuthorizedScope,
    *,
    target: str = "problem",
    subject: str,
    field: str,
    text: str,
    source_type: str,
    source_id: str,
    source_revision: str = "1",
    author: str = "agent:advisor",
    uncertainty: str = "",
    references: Iterable[str] = (),
    depends_on_revision: Optional[int] = None,
    protected: Iterable[str] = (),
    log: Any = None,
) -> dict[str, Any]:
    """Store an AI interpretation or Advisor suggestion as a provisional draft; retries never activate it."""
    from ..filelock import held

    scope_kind, scope_id = _scope_target(scope, target)
    cleaned = _validate_text(text, protected)
    idem_key = f"draft:{source_type}:{source_id}:{source_revision}:{scope_kind}:{scope_id}"
    who = author if author.startswith(("agent:", "rule:")) else f"agent:{author}"
    path = _base_file(home, scope_kind, scope_id)

    with held(_lock_dir(home)):
        doc = _read_doc_unlocked(path, scope_kind, scope_id, scope)
        cur_rev = int(doc.get("revision") or 0)
        existing_id = doc["events_index"].get(idem_key)
        if existing_id:
            existing = next((e for e in doc["entries"] if e.get("entry_id") == existing_id), None)
            if existing is not None:
                # A retry NEVER activates a draft.
                return {"entry": dict(existing), "revision": cur_rev, "replayed": True}

        now_ts = _now()
        src_ref = clean_line(f"{source_type}:{source_id}@rev{source_revision}")[:180]
        mem_item = new(
            kind="fact",
            scope=scope_kind,
            scope_id=scope_id,
            text=cleaned,
            source=src_ref,
            who=who,
            at=now_ts,
            status="draft",
        )
        store = open_memory(home)
        why = check(mem_item, tuple(protected))
        if why:
            raise Refused(why)
        held_item = next((v for v in latest(store.log()) if v.id == mem_item.id), None)
        if held_item is None:
            store.write(mem_item)
            event(log, "memory.proposed", mem_item, origin=source_type)

        dep_rev = cur_rev if depends_on_revision is None else int(depends_on_revision)
        active_ids = [e["entry_id"] for e in doc["entries"] if e.get("status") == "active" and e.get("kind") == "fact"]
        entry_id = "ent_" + secrets.token_hex(5)
        is_stale = dep_rev < cur_rev
        entry = {
            "entry_id": entry_id,
            "item_id": mem_item.id,
            "kind": "fact",
            "subject": clean_line(subject or target)[:80] or target,
            "field": clean_line(field or "proposal")[:80] or "proposal",
            "text": mem_item.text,
            "source": src_ref,
            "source_type": source_type,
            "source_id": source_id,
            "source_revision": str(source_revision),
            "author": who,
            "status": "draft",
            "review_state": "pending",
            "provisional": True,
            "carries_approval": False,
            "correction": False,
            "prior_revision": dep_rev,
            "effective_revision": None,
            "supersedes": None,
            "retired_by": None,
            "depends_on": {"base_revision": dep_rev, "entry_ids": active_ids, "references": list(references)[:12]},
            "uncertainty": clean_line(uncertainty)[:240] if uncertainty else "",
            "stale": is_stale,
            "stale_reason": ("underlying base changed before advice arrived" if is_stale else ""),
            "changed_sources": ([{"field": "base", "new_revision": cur_rev, "reason": "delayed_advice_after_revision_change"}] if is_stale else []),
            "outcome_status": "proposed",
            "created_at": now_ts,
            "updated_at": now_ts,
        }
        doc["entries"].append(entry)
        doc["events_index"][idem_key] = entry_id
        doc["updated_at"] = now_ts
        _write_doc_unlocked(path, doc)

    _invalidate_summary_best_effort(home, scope)
    return {"entry": dict(entry), "revision": cur_rev, "replayed": False}


def confirm_draft(
    home: Path,
    scope: AuthorizedScope,
    entry_id: str,
    *,
    action: str = "accept",
    target: str = "problem",
    expected_revision: Optional[int] = None,
    allow_stale: bool = False,
    log: Any = None,
) -> dict[str, Any]:
    """Explicit user review promoting or rejecting a provisional draft; blocks stale drafts unless re-checked."""
    from ..filelock import held

    if not scope.can_write:
        raise AccessDenied(f"role {scope.role!r} cannot review drafts in {target} base")
    if action not in ("accept", "reject"):
        raise ValueError("action must be 'accept' or 'reject'")
    scope_kind, scope_id = _scope_target(scope, target)
    author = f"person:{scope.principal_id}"
    path = _base_file(home, scope_kind, scope_id)

    with held(_lock_dir(home)):
        doc = _read_doc_unlocked(path, scope_kind, scope_id, scope)
        cur_rev = int(doc.get("revision") or 0)
        if expected_revision is not None and int(expected_revision) != cur_rev:
            raise RevisionConflictError(
                scope=scope_kind, scope_id=scope_id,
                expected_revision=int(expected_revision), current_revision=cur_rev,
                current_entries=[dict(e) for e in doc["entries"] if e.get("status") == "active"],
                attempted={"entry_id": entry_id, "action": action, "author": author},
            )
        draft = next((e for e in doc["entries"] if e.get("entry_id") == entry_id), None)
        if draft is None:
            raise KeyError(entry_id)
        if draft.get("status") != "draft":
            raise Refused(f"only a pending draft can be confirmed; status is {draft.get('status')!r}")
        if action == "accept" and draft.get("stale") and not allow_stale:
            raise StaleDependencyError(
                f"draft {entry_id} is stale ({draft.get('stale_reason') or 'source changed'}) and must be rechecked before confirmation",
                changed_sources=list(draft.get("changed_sources") or []),
            )

        now_ts = _now()
        store = open_memory(home)
        held_item = next((v for v in latest(store.log()) if v.id == draft["item_id"]), None)
        if action == "reject":
            draft["status"] = "retired"
            draft["review_state"] = "rejected"
            draft["updated_at"] = now_ts
            if held_item is not None and held_item.status != "retired":
                store.write(retire(held_item))
                event(log, "memory.retired", held_item, reason="rejected", by=author)
            doc["updated_at"] = now_ts
            _write_doc_unlocked(path, doc)
            _invalidate_summary_best_effort(home, scope)
            return {"entry": dict(draft), "revision": cur_rev}

        new_rev = cur_rev + 1
        draft["status"] = "active"
        draft["review_state"] = "confirmed"
        draft["provisional"] = False
        draft["effective_revision"] = new_rev
        draft["confirmed_by"] = author
        draft["updated_at"] = now_ts
        if held_item is not None and held_item.status == "draft":
            active_item = MemoryItem.model_validate({**held_item.model_dump(), "status": "active"})
            store.write(active_item)
            event(log, "memory.written", active_item, by=author, reason="accepted")
        doc["revision"] = new_rev
        doc["updated_at"] = now_ts
        _write_doc_unlocked(path, doc)

    _invalidate_summary_best_effort(home, scope)
    return {"entry": dict(draft), "revision": new_rev}


def purge_scope(home: Path, scope_kind: str, scope_id: str, *, workspace_id: str = "local",
                project_id: Optional[str] = None, problem_id: Optional[str] = None, log: Any = None) -> dict[str, int]:
    """Purge a deleted project or problem scope: retire MemoryStore items, remove base index, summary, and vectors."""
    from ..filelock import held

    retired_items = 0
    retired_texts: list[str] = []
    if (Path(home) / "memory" / "items.jsonl").is_file() or os.environ.get("DCLAB_DATABASE_URL"):
        try:
            store = open_memory(home)
            all_latest = latest(store.log())
            for item in all_latest:
                if item.status == "retired":
                    continue
                if (item.scope == scope_kind and item.scope_id == scope_id) or (
                    scope_kind == "project" and project_id and (
                        (item.scope == "project" and (item.scope_id == project_id or item.scope_id.endswith(f":{project_id}")))
                        or (item.scope == "problem" and f":{project_id}:" in item.scope_id)
                    )
                ):
                    retired_texts.append(item.text)
            retired_items = retire_scope(store, scope_kind, scope_id, log=log)
            if scope_kind == "project" and project_id:
                extra_scopes = {
                    (i.scope, i.scope_id) for i in all_latest
                    if i.status != "retired" and (
                        (i.scope == "project" and (i.scope_id == project_id or i.scope_id.endswith(f":{project_id}")))
                        or (i.scope == "problem" and f":{project_id}:" in i.scope_id)
                    ) and (i.scope, i.scope_id) != (scope_kind, scope_id)
                }
                for sk, sid in extra_scopes:
                    retired_items += retire_scope(store, sk, sid, log=log)
        except Exception:  # noqa: BLE001
            retired_items = 0

    removed_base = 0
    bdir = _lock_dir(home)
    if bdir.is_dir():
        with held(bdir):
            path = _base_file(home, scope_kind, scope_id)
            if path.is_file():
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                    retired_texts.extend(str(e.get("text") or "") for e in (doc.get("entries") or []) if e.get("text"))
                except (OSError, ValueError):
                    pass
                path.unlink()
                removed_base += 1
            if scope_kind == "project" and project_id:
                for p in list(bdir.glob("*.json")):
                    try:
                        doc = json.loads(p.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if doc.get("project_id") == project_id and (not workspace_id or doc.get("workspace_id") == workspace_id):
                        retired_texts.extend(str(e.get("text") or "") for e in (doc.get("entries") or []) if e.get("text"))
                        p.unlink()
                        removed_base += 1

    removed_summaries = 0
    try:
        from . import summary as summary_mod

        removed_summaries = summary_mod.purge_summary(home, workspace_id, project_id, problem_id, purged_texts=retired_texts)
    except Exception:  # noqa: BLE001
        pass

    return {"retired_items": retired_items, "removed_bases": removed_base, "removed_summaries": removed_summaries}


__all__ = [
    "BASES_DIR", "OUTCOME_STATUSES", "RevisionConflictError", "StaleDependencyError",
    "base_revision", "confirm_draft", "correct_base_fact", "propose_draft", "purge_scope",
    "read_base", "record_material_event", "retire_base_fact", "upsert_base_fact",
]
