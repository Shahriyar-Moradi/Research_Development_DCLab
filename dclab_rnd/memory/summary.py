"""Active, versioned resumable summary per project/problem and visibility class (Package A5).

Stores a compact deterministic projection under ``<home>/memory/summaries/`` keyed by
``(workspace_id, project_id, problem_id, visibility_class)``. Shared summaries contain ONLY shareable
project and problem facts, decisions, open questions, pending approvals, provisional advice, and omitted
counts. Personal preferences are never persisted in a shared summary; they are applied as a viewer-specific
overlay only after authorization.

Publishing uses a compare-and-swap check against the monotonic source cursor so a late worker building from
an older state can never overwrite a correction that arrived in the meantime.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from dclab_core.memory import BRIEF_TOKENS, tokens

from . import open_memory, view
from .bases import read_base
from .scope import AccessDenied, AuthorizedScope

SUMMARY_DIR = Path("memory") / "summaries"
SCHEMA_VERSION = "dclab.problem_summary.v1"
REQUIRED_FIELDS = frozenset({"prediction_moment", "target", "exclusions", "forbidden", "split", "metric", "acceptance_criteria"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _summary_key(workspace_id: str, project_id: str, problem_id: str, visibility_class: str = "shared") -> str:
    return f"{workspace_id}:{project_id}:{problem_id}:{visibility_class}"


def _summary_file(home: Path, workspace_id: str, project_id: str, problem_id: str, visibility_class: str = "shared") -> Path:
    digest = hashlib.sha256(_summary_key(workspace_id, project_id, problem_id, visibility_class).encode("utf-8")).hexdigest()[:24]
    return Path(home) / SUMMARY_DIR / f"sum_{digest}.json"


def _lock_dir(home: Path) -> Path:
    return Path(home) / SUMMARY_DIR


def _cursor_from_bases(proj_base: Mapping[str, Any], prob_base: Mapping[str, Any]) -> list[int]:
    """Monotonic source cursor across project and problem bases; strictly increases on every edit, correction,
    retirement, material event, or draft review."""
    proj_rev = int(proj_base.get("revision") or 0)
    prob_rev = int(prob_base.get("revision") or 0)
    proj_total = len(proj_base.get("active") or []) + len(proj_base.get("drafts") or []) + len(proj_base.get("retired") or [])
    prob_total = len(prob_base.get("active") or []) + len(prob_base.get("drafts") or []) + len(prob_base.get("retired") or [])
    retired_total = len(proj_base.get("retired") or []) + len(prob_base.get("retired") or [])
    return [proj_rev + prob_rev, proj_rev, prob_rev, proj_total + prob_total, retired_total]


def capture_snapshot(
    home: Path,
    scope: AuthorizedScope,
    *,
    open_questions: Iterable[Mapping[str, Any]] = (),
    pending_approvals: Iterable[Mapping[str, Any]] = (),
    next_proposed_action: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Capture a consistent authorized snapshot of project + problem bases and their monotonic cursor."""
    if not (scope.project_id and scope.problem_id):
        raise AccessDenied("authorized project_id and problem_id are required for a problem summary")
    proj_base = read_base(home, scope, "project")
    prob_base = read_base(home, scope, "problem")
    cursor = _cursor_from_bases(proj_base, prob_base)
    return {
        "workspace_id": scope.workspace_id,
        "project_id": scope.project_id,
        "problem_id": scope.problem_id,
        "visibility_class": "shared",
        "cursor": cursor,
        "project_revision": int(proj_base.get("revision") or 0),
        "problem_revision": int(prob_base.get("revision") or 0),
        "project_base": proj_base,
        "problem_base": prob_base,
        "open_questions": [dict(q) for q in open_questions if isinstance(q, Mapping)],
        "pending_approvals": [dict(a) for a in pending_approvals if isinstance(a, Mapping)],
        "next_proposed_action": dict(next_proposed_action) if isinstance(next_proposed_action, Mapping) else None,
    }


def _fact_priority(entry: Mapping[str, Any]) -> tuple[int, int, int, str]:
    is_corr = 0 if entry.get("correction") else 1
    is_req = 0 if str(entry.get("field") or "").lower() in REQUIRED_FIELDS else 1
    is_prob = 0 if entry.get("_scope") == "problem" else 1
    return (is_corr, is_req, is_prob, str(entry.get("updated_at") or ""))


def build_from_snapshot(snapshot: Mapping[str, Any], *, budget: int = BRIEF_TOKENS) -> dict[str, Any]:
    """Deterministically build the compact shared resumable summary from a captured snapshot."""
    proj_base = snapshot["project_base"]
    prob_base = snapshot["problem_base"]
    prob_rev = int(snapshot["problem_revision"])
    proj_rev = int(snapshot["project_revision"])

    raw_facts: list[dict[str, Any]] = []
    raw_episodes: list[dict[str, Any]] = []
    for scope_name, base in (("problem", prob_base), ("project", proj_base)):
        for ent in base.get("active") or []:
            item = {
                "entry_id": ent["entry_id"],
                "item_id": ent["item_id"],
                "scope": scope_name,
                "_scope": scope_name,
                "subject": ent.get("subject", scope_name),
                "field": ent.get("field", "fact"),
                "text": ent.get("text", ""),
                "source": ent.get("source", ""),
                "author": ent.get("author", ""),
                "correction": bool(ent.get("correction")),
                "revision": int(ent.get("effective_revision") or 0),
                "updated_at": ent.get("updated_at", ""),
                "outcome_status": ent.get("outcome_status"),
                "depends_on": dict(ent.get("depends_on") or {}),
                "stale": bool(ent.get("stale")),
                "stale_reason": ent.get("stale_reason", ""),
                "changed_sources": list(ent.get("changed_sources") or []),
            }
            if ent.get("kind") == "episode":
                raw_episodes.append(item)
            else:
                raw_facts.append(item)

    raw_facts.sort(key=_fact_priority)
    raw_episodes.sort(key=lambda e: str(e.get("updated_at") or ""), reverse=True)

    # Mark pending approvals and next proposed action stale if their depended revision is behind current problem_revision.
    pending_approvals: list[dict[str, Any]] = []
    for ap in snapshot.get("pending_approvals") or []:
        dep_rev = ap.get("depends_on_revision")
        stale = bool(ap.get("stale")) or (dep_rev is not None and int(dep_rev) < prob_rev)
        pending_approvals.append({**ap, "stale": stale, **({"stale_reason": "problem facts changed since proposal"} if stale and not ap.get("stale_reason") else {})})

    next_action = dict(snapshot["next_proposed_action"]) if isinstance(snapshot.get("next_proposed_action"), Mapping) else None
    if next_action is not None:
        dep_rev = next_action.get("depends_on_revision")
        if dep_rev is not None and int(dep_rev) < prob_rev:
            next_action["stale"] = True
            next_action.setdefault("stale_reason", "problem facts changed since proposal")

    pending_drafts: list[dict[str, Any]] = []
    for scope_name, base in (("problem", prob_base), ("project", proj_base)):
        for d in base.get("drafts") or []:
            dep_rev = (d.get("depends_on") or {}).get("base_revision")
            cur_scope_rev = prob_rev if scope_name == "problem" else proj_rev
            stale = bool(d.get("stale")) or (dep_rev is not None and int(dep_rev) < cur_scope_rev)
            pending_drafts.append({
                "entry_id": d["entry_id"],
                "item_id": d["item_id"],
                "scope": scope_name,
                "field": d.get("field", "proposal"),
                "text": d.get("text", ""),
                "author": d.get("author", ""),
                "uncertainty": d.get("uncertainty", ""),
                "provisional": True,
                "carries_approval": False,
                "stale": stale,
                "stale_reason": d.get("stale_reason") or ("underlying base changed" if stale else ""),
                "changed_sources": list(d.get("changed_sources") or []),
                "depends_on": dict(d.get("depends_on") or {}),
            })

    # Apply token budget: preserve required constraints & corrections first, then other facts, then recent episodes.
    kept_facts: list[dict[str, Any]] = []
    kept_episodes: list[dict[str, Any]] = []
    lines: list[str] = []
    omitted = 0

    for f in raw_facts:
        tag = f"[{f['scope']}:{f['field']}@r{f['revision']}{'*' if f['correction'] else ''}]"
        line = f"- {tag} {f['text']}"
        is_required = f["correction"] or str(f["field"]).lower() in REQUIRED_FIELDS
        if tokens("\n".join([*lines, line])) <= budget or (is_required and not kept_facts):
            lines.append(line)
            clean_f = {k: v for k, v in f.items() if k != "_scope"}
            kept_facts.append(clean_f)
        else:
            omitted += 1

    for ep in raw_episodes[:12]:
        status_note = f" ({ep['outcome_status']})" if ep.get("outcome_status") else ""
        stale_note = " [STALE]" if ep.get("stale") else ""
        line = f"- [decision:{ep['entry_id']}{status_note}{stale_note}] {ep['text']}"
        if tokens("\n".join([*lines, line])) <= budget:
            lines.append(line)
            clean_ep = {k: v for k, v in ep.items() if k != "_scope"}
            kept_episodes.append(clean_ep)
        else:
            omitted += 1
    omitted += max(0, len(raw_episodes) - 12)

    open_qs = list(snapshot.get("open_questions") or [])
    for q in open_qs:
        q_line = f"- [open question] {q.get('question') or q.get('text') or ''}"
        if q_line.strip() and tokens("\n".join([*lines, q_line])) <= budget:
            lines.append(q_line)

    text = "\n".join(lines)
    return {
        "schema_version": SCHEMA_VERSION,
        "workspace_id": snapshot["workspace_id"],
        "project_id": snapshot["project_id"],
        "problem_id": snapshot["problem_id"],
        "visibility_class": "shared",
        "cursor": list(snapshot["cursor"]),
        "project_revision": proj_rev,
        "problem_revision": prob_rev,
        "active_facts": kept_facts,
        "active_fact_ids": [f["item_id"] for f in kept_facts],
        "recent_decisions": kept_episodes,
        "open_questions": open_qs,
        "pending_approvals": pending_approvals,
        "pending_drafts": pending_drafts,
        "next_proposed_action": next_action,
        "omitted_count": omitted,
        "tokens": tokens(text),
        "budget": budget,
        "text": text,
        "rebuilt_at": _now(),
    }


def publish_summary(home: Path, scope: AuthorizedScope, summary_doc: Mapping[str, Any]) -> dict[str, Any]:
    """Compare-and-swap publish: rejects late builds from an older source cursor so stale workers never reverse edits."""
    from ..filelock import held

    if not (scope.project_id and scope.problem_id):
        raise AccessDenied("authorized project_id and problem_id are required to publish a summary")
    if (str(summary_doc.get("workspace_id")) != scope.workspace_id
            or str(summary_doc.get("project_id")) != scope.project_id
            or str(summary_doc.get("problem_id")) != scope.problem_id):
        raise AccessDenied("summary scope mismatch")

    path = _summary_file(home, scope.workspace_id, scope.project_id, scope.problem_id, "shared")
    candidate_cursor = tuple(int(x) for x in (summary_doc.get("cursor") or [0, 0, 0, 0, 0]))

    with held(_lock_dir(home)):
        # Re-read live cursor from authoritative bases to reject any build that started before a concurrent correction.
        live_snap = capture_snapshot(home, scope)
        live_cursor = tuple(int(x) for x in live_snap["cursor"])
        if candidate_cursor < live_cursor:
            return {"published": False, "reason": "stale_source_cursor", "candidate_cursor": list(candidate_cursor),
                    "live_cursor": list(live_cursor)}

        if path.is_file():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                stored_cursor = tuple(int(x) for x in (stored.get("cursor") or [0, 0, 0, 0, 0]))
                if candidate_cursor < stored_cursor:
                    return {"published": False, "reason": "older_than_stored_summary", "candidate_cursor": list(candidate_cursor),
                            "stored_cursor": list(stored_cursor)}
            except (OSError, ValueError):
                pass

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.part")
        tmp.write_text(json.dumps(dict(summary_doc), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    return {"published": True, "cursor": list(candidate_cursor)}


def refresh_summary(
    home: Path,
    scope: AuthorizedScope,
    *,
    budget: int = BRIEF_TOKENS,
    open_questions: Iterable[Mapping[str, Any]] = (),
    pending_approvals: Iterable[Mapping[str, Any]] = (),
    next_proposed_action: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Capture current authorized sources, build deterministic summary, and publish via CAS."""
    snap = capture_snapshot(
        home, scope, open_questions=open_questions,
        pending_approvals=pending_approvals, next_proposed_action=next_proposed_action,
    )
    built = build_from_snapshot(snap, budget=budget)
    pub = publish_summary(home, scope, built)
    return {**built, "published": bool(pub.get("published"))}


def _personal_overlay(home: Path, scope: AuthorizedScope) -> list[dict[str, Any]]:
    """Viewer-specific private personal preferences, read only after authorization and never persisted in shared summary."""
    try:
        store = open_memory(home)
        items = view(store, scope.personal_chain())
        return [{"item_id": i.id, "text": i.text, "source": i.source, "scope": i.scope} for i in items]
    except Exception:  # noqa: BLE001
        return []


def read_summary(
    home: Path,
    scope: AuthorizedScope,
    *,
    budget: int = BRIEF_TOKENS,
    auto_rebuild: bool = True,
    include_personal_overlay: bool = True,
    open_questions: Iterable[Mapping[str, Any]] = (),
    pending_approvals: Iterable[Mapping[str, Any]] = (),
    next_proposed_action: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Read the active versioned summary, validating scope, cursor, and active references; rebuilds if missing or stale."""
    if not (scope.project_id and scope.problem_id):
        raise AccessDenied("authorized project_id and problem_id are required to read a problem summary")

    live_snap = capture_snapshot(
        home, scope, open_questions=open_questions,
        pending_approvals=pending_approvals, next_proposed_action=next_proposed_action,
    )
    live_cursor = list(live_snap["cursor"])
    path = _summary_file(home, scope.workspace_id, scope.project_id, scope.problem_id, "shared")
    cached: Optional[dict[str, Any]] = None
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if (isinstance(doc, dict)
                    and doc.get("workspace_id") == scope.workspace_id
                    and doc.get("project_id") == scope.project_id
                    and doc.get("problem_id") == scope.problem_id
                    and doc.get("cursor") == live_cursor
                    and int(doc.get("budget") or BRIEF_TOKENS) == budget
                    and not open_questions and not pending_approvals and next_proposed_action is None):
                # Also verify all referenced fact item IDs remain active.
                live_ids = {
                    e["item_id"]
                    for base in (live_snap["project_base"], live_snap["problem_base"])
                    for e in (base.get("active") or [])
                    if e.get("kind") != "episode"
                }
                if all(fid in live_ids for fid in (doc.get("active_fact_ids") or [])):
                    cached = doc
        except (OSError, ValueError):
            cached = None

    if cached is not None:
        out = dict(cached)
        out["fresh"] = True
        out["from_cache"] = True
    elif auto_rebuild:
        built = build_from_snapshot(live_snap, budget=budget)
        publish_summary(home, scope, built)
        out = dict(built)
        out["fresh"] = True
        out["from_cache"] = False
    else:
        out = {
            "schema_version": SCHEMA_VERSION,
            "workspace_id": scope.workspace_id,
            "project_id": scope.project_id,
            "problem_id": scope.problem_id,
            "visibility_class": "shared",
            "cursor": live_cursor,
            "project_revision": live_snap["project_revision"],
            "problem_revision": live_snap["problem_revision"],
            "active_facts": [],
            "recent_decisions": [],
            "open_questions": list(open_questions),
            "pending_approvals": list(pending_approvals),
            "pending_drafts": [],
            "next_proposed_action": next_proposed_action,
            "omitted_count": 0,
            "text": "",
            "fresh": False,
            "from_cache": False,
        }

    if include_personal_overlay:
        out["personal_preferences"] = _personal_overlay(home, scope)
    return out


def invalidate_summary(home: Path, workspace_id: str, project_id: Optional[str], problem_id: Optional[str] = None) -> int:
    """Delete cached summaries affected by a base change so the next read rebuilds from current sources."""
    from ..filelock import held

    sdir = _lock_dir(home)
    if not sdir.is_dir() or not project_id:
        return 0
    removed = 0
    with held(sdir):
        if problem_id:
            p = _summary_file(home, workspace_id, project_id, problem_id, "shared")
            if p.is_file():
                p.unlink()
                removed += 1
        else:
            for p in list(sdir.glob("sum_*.json")):
                try:
                    doc = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if doc.get("workspace_id") == workspace_id and doc.get("project_id") == project_id:
                    p.unlink()
                    removed += 1
    return removed


def purge_vectors_for_texts(home: Path, texts: Iterable[str]) -> int:
    """Remove cached search vectors matching deleted or purged item texts from memory/vectors-*.json."""
    from ..filelock import held
    from .vectors import PREFIXES

    clean_texts = [str(t) for t in texts if t]
    mem_dir = Path(home) / "memory"
    if not clean_texts or not mem_dir.is_dir():
        return 0
    prefixes = {doc_p for _q, doc_p in PREFIXES.values()} | {""}
    target_keys = {
        hashlib.sha256(f"{pref}\0{text}".encode()).hexdigest()[:24]
        for pref in prefixes
        for text in clean_texts
    }
    removed = 0
    with held(mem_dir):
        for vpath in list(mem_dir.glob("vectors-*.json")):
            try:
                data = json.loads(vpath.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            kept = {k: v for k, v in data.items() if k not in target_keys}
            if len(kept) != len(data):
                removed += len(data) - len(kept)
                tmp = vpath.with_name(f".{vpath.name}.{secrets.token_hex(4)}.part")
                tmp.write_text(json.dumps(kept), encoding="utf-8")
                os.replace(tmp, vpath)
    return removed


def purge_summary(home: Path, workspace_id: str, project_id: Optional[str], problem_id: Optional[str] = None,
                  purged_texts: Iterable[str] = ()) -> int:
    """Delete derived summaries and any cached vectors when a source project or problem is purged."""
    removed = invalidate_summary(home, workspace_id, project_id, problem_id)
    purge_vectors_for_texts(home, purged_texts)
    return removed


__all__ = [
    "REQUIRED_FIELDS", "SCHEMA_VERSION", "SUMMARY_DIR",
    "build_from_snapshot", "capture_snapshot", "invalidate_summary",
    "publish_summary", "purge_summary", "purge_vectors_for_texts",
    "read_summary", "refresh_summary",
]
