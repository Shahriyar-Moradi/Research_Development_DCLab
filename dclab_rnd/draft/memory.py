"""A memory of the whole Home conversation (package H5 of the Home plan): what the model keeps in view however long the
conversation grows.

The model's state used to hold the last twelve messages: a fact said in the second message was gone by the thirtieth.
``summary`` is built by code from the draft's own records, so it needs no model and says nothing the records do not: the
kind of help and how it was read, every field known (the prediction's understanding and the track's brief) with where it
came from, the questions answered and how, the questions still open, the data's shape. The model sees the state, this
summary and the last ``RECENT`` messages.
"""

from __future__ import annotations

from typing import Any

RECENT = 8  # the last messages the model sees beside the summary (the plan: state + summary + the last 8)
FIELD_TEXT = 160


def _clip(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= FIELD_TEXT else text[:FIELD_TEXT - 1] + "…"


def summary(draft: dict[str, Any]) -> dict[str, Any]:
    """What the whole conversation has settled, from the draft's records alone."""
    intent = draft.get("intent") or {}
    track = intent.get("track") or "prediction"
    plan = draft.get("plan") or {}
    known: dict[str, dict[str, Any]] = {}
    for field, value in (draft.get("understanding") or {}).items():
        if field in ("notes", "assumed") or value in (None, "", [], {}):
            continue
        entry = plan.get(field) if isinstance(plan.get(field), dict) else {}
        assumed = ((draft.get("understanding") or {}).get("assumed") or {}).get(field) == value
        known[field] = {"value": _clip(value), "from": "assumed" if assumed else (entry or {}).get("source") or "the person's words"}
    for t, brief in (draft.get("track_brief") or {}).items():
        for field, entry in (brief or {}).items():
            if isinstance(entry, dict) and entry.get("value") not in (None, ""):
                known.setdefault(field, {"value": _clip(entry["value"]), "from": entry.get("source") or "rules", "track": t})
    answered = [{"field": q.get("field"), "answer": _clip(q.get("answered")), **({"declined": True} if q.get("declined") else {})}
                for q in draft.get("questions") or [] if q.get("answered") and not q.get("superseded")]
    still_open = [{"field": q.get("field"), "question": _clip(q.get("text"))}
                  for q in draft.get("questions") or [] if not q.get("answered")]
    data = ((draft.get("analysis") or {}).get("summary") or {})
    return {"track": track, "track_read": {k: intent.get(k) for k in ("status", "source") if intent.get(k)},
            "known": known, "answered": answered, "open": still_open,
            "data": {k: data.get(k) for k in ("rows", "columns") if data.get(k) is not None} or None,
            "messages": len(draft.get("messages") or [])}


def recent(draft: dict[str, Any], n: int = RECENT) -> list[dict[str, str]]:
    """The last ``n`` messages as the model reads them."""
    return [{"role": "user" if m.get("role") == "user" else "assistant", "content": m["text"]}
            for m in (draft.get("messages") or [])[-n:] if m.get("text")]


LABELS = {"target": "Outcome predicted", "prediction_moment": "Prediction made", "action": "What each prediction is for",
          "entity": "Who it is about", "measure": "Measure", "breakdown": "Split by", "period": "Period", "filters": "Filters",
          "rules": "Steps", "destination": "Result goes to", "kpis": "Numbers shown", "audience": "For", "refresh": "Updates",
          "decision": "Decision", "success_measure": "Measure of success", "period_compared": "Periods compared", "topic": "Topic"}


def facts(draft: dict[str, Any]) -> list[str]:
    return [line for line, _ in confirmed_facts(draft)]


def confirmed_facts(draft: dict[str, Any]) -> list[tuple[str, bool]]:
    """The lines worth remembering for the project: only fields stated in the person's own words (a quote, or their answer),
    never an assumption, a default or a guess; one line each, no cell value (column names and the person's words only). Each with
    whether the person confirmed it (an answer or a correction on the sheet) or Home read it from their words."""
    out: list[tuple[str, bool]] = []
    plan = draft.get("plan") or {}
    assumed = (draft.get("understanding") or {}).get("assumed") or {}
    for field, value in (draft.get("understanding") or {}).items():
        entry = plan.get(field) if isinstance(plan.get(field), dict) else {}
        if field in LABELS and value and assumed.get(field) != value and (entry.get("status") == "stated" or entry.get("source") in ("answer", "person")):
            out.append((f"{LABELS[field]}: {_clip(value)}", entry.get("source") in ("answer", "person")))
    for brief in (draft.get("track_brief") or {}).values():
        for field, entry in (brief or {}).items():
            if field in LABELS and isinstance(entry, dict) and entry.get("value") and entry.get("source") in ("user", "rules") and entry.get("quote"):
                out.append((f"{LABELS[field]}: {_clip(entry['value'])}", entry.get("source") == "user"))
    seen: set[str] = set()
    return [(line, ok) for line, ok in out if not (line in seen or seen.add(line))]


def hand_off(store: Any, draft: dict[str, Any], project_id: str, log: Any = None) -> int:
    """H5: when a conversation becomes a project, what the person stated becomes the project's memory (dclab_rnd.memory, one
    fact per line, source home brief:<draft>). Best effort: a refused line (a key, a password) or no memory is skipped."""
    from pathlib import Path

    try:
        from .. import memory as project_memory
        from ..memory.writing import record_fact
    except ImportError:
        return 0
    home = getattr(store, "files_home", None) or Path(store.home).parent
    try:
        book = project_memory.open_memory(Path(home))
    except Exception:  # noqa: BLE001 - no memory here: nothing to hand off
        return 0
    import sys

    try:
        from ..accounts.principal import current
        person = getattr(current(), "user_id", None) or "owner"
    except Exception:  # noqa: BLE001 - no accounts: the workspace's owner
        person = "owner"
    written = 0
    for line, confirmed in confirmed_facts(draft):
        try:  # a memory write never fails a build (the review of H5): any error is said and the next line goes on
            record_fact(book, scope="project", scope_id=project_id, text=line, source=f"home brief:{draft['id']}",
                        who=f"person:{person}" if confirmed else "agent:home", log=log)
            written += 1
        except Exception as error:  # noqa: BLE001
            print(f"home memory: a line was not kept ({type(error).__name__})", file=sys.stderr)
    return written


__all__ = ["RECENT", "summary", "recent", "facts", "confirmed_facts", "hand_off"]
