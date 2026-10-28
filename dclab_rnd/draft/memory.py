"""A memory of the whole conversation (package H5).

The Home agent's model saw its state and the last 12 messages, so a long conversation forgot what was said early: an
outcome stated in message 2 was gone by message 30 unless the plan happened to hold it. This module keeps a summary of
the whole draft, built by code from what the draft records, so it needs no model and cannot be wrong about what was
said:

- the kind of help, and how it was decided (the person's choice, the rules, the model);
- what Home understood for a prediction (the plan's fields) and for the other tracks (their briefs), each with how it
  is known, and the defaults it took;
- every question asked and its answer (a "not sure" says so), and the question still open;
- the data brought, by name and status (never a value);
- the person's own remarks kept as notes.

The model's state becomes the state, this summary and the last ``RECENT`` messages. When the draft becomes a project,
the confirmed fields go into the project's memory (A5.2) as notes, so the project's agents start from what the
conversation settled. Lines are single lines of at most ``studio.memory.TEXT`` characters: a person's text inside a
prompt must not pass for an instruction of its own.

An optional model summary of free remarks (the plan's second step) is not built: the notes are kept whole instead,
which is honest but longer.
"""

from __future__ import annotations

from typing import Any

from dclab_rnd import prompts

from ..studio import memory as studio_memory
from . import plan as plans

RECENT = 8  # messages the model sees in full, after the summary
QUESTIONS = 24  # questions and answers kept in the summary (the latest)
_line = studio_memory._line


def summary(draft: dict[str, Any]) -> list[str]:
    """The whole conversation as short lines, from what the draft records (no model)."""
    lines: list[str] = []
    intent = draft.get("intent") or {}
    if intent:
        how = {"user": "the person chose it", "rules": "read from their words", "model": "read by the model",
               "default": "the usual start"}.get(intent.get("source"), "read")
        status = "not decided yet" if intent.get("status") == "asked" else intent.get("track")
        lines.append(_line(f"Kind of help: {status} ({how})."))
    u = draft.get("understanding") or {}
    plan = draft.get("plan") or {}
    texts = plans.user_text(draft)
    for field in plans.ALL_FIELDS:
        if u.get(field):
            entry = plan.get(field) or {}
            if _persons(draft, field, texts):
                how = "the person's words" + (", answered" if entry.get("source") == "answer" else ", in the request")
            elif entry.get("source") == "model" or entry.get("source") == "answer":
                how = "the model's reading of what they said"
            elif entry.get("source") == "data":
                how = "from the data"
            else:
                how = "how it is known was not recorded"
            lines.append(_line(f"Understood {field.replace('_', ' ')}: “{u[field]}” ({how})."))
    for track, entries in (draft.get("track_brief") or {}).items():
        for field, entry in (entries or {}).items():
            if not isinstance(entry, dict) or not entry.get("value"):
                continue
            how = {"problem": "from the request", "answer": "answered", "default": "default, not stated"}.get(entry.get("source"), "")
            lines.append(_line(f"{track.capitalize()} {field.replace('_', ' ')}: “{entry['value']}”{' (' + how + ')' if how else ''}."))
    asked = [q for q in draft.get("questions") or [] if not q.get("superseded")]
    for q in asked[-QUESTIONS:]:
        if q.get("answered"):
            lines.append(_line(f"Asked about {q.get('field')}: answered “{q['answered']}”."))
    pending = next((q for q in asked if not q.get("answered")), None)
    if pending:
        lines.append(_line(f"Open question about {pending.get('field')}: {pending.get('text')}"))
    for asset in draft.get("assets") or []:
        lines.append(_line(f"Data: {asset.get('name') or asset.get('kind')} ({asset.get('status')}{', synthetic' if asset.get('synthetic') else ''})."))
    if u.get("notes"):
        lines.append(_line(f"Notes recorded (by the person or the model): {u['notes']}"))
    return lines


def _persons(draft: dict[str, Any], field: str, texts: list[str]) -> bool:
    """A plan field is the person's own when it came from their request or their answer and its value is their words."""
    entry = (draft.get("plan") or {}).get(field) or {}
    value = (draft.get("understanding") or {}).get(field)
    return entry.get("source") in ("problem", "answer") and bool(value) and plans.quoted(str(value), texts)


def history(draft: dict[str, Any]) -> list[dict[str, str]]:
    """The summary as one message, then the last RECENT messages in full."""
    recent = [{"role": "user" if m["role"] == "user" else "assistant", "content": m["text"]}
              for m in (draft.get("messages") or []) if m.get("text")][-RECENT:]
    lines = summary(draft)
    # the framing is a prompt file (prompts/home_memory.md): records of what was said, never instructions
    head = [{"role": "system", "content": prompts.text("home_memory") + "\n- " + "\n- ".join(lines)}] if lines else []
    return head + recent


def hand_off(project: dict[str, Any], draft: dict[str, Any]) -> list[dict[str, Any]]:
    """The confirmed fields of the conversation as project-memory notes (the caller saves the project). Only what the
    person stated or answered is handed on; a default is not a decision."""
    notes = []
    u = draft.get("understanding") or {}
    texts = plans.user_text(draft)
    for field in plans.ALL_FIELDS:
        if _persons(draft, field, texts):  # the person's own words only: never a model's reading, a default or a not-sure
            notes.append(studio_memory.add(project, "context", f"{field.replace('_', ' ')}: {u[field]}",
                                           "the person's words in the Home conversation; a later decision overrides it", "human", "home_handoff"))
    for track, entries in (draft.get("track_brief") or {}).items():
        for field, entry in (entries or {}).items():
            if isinstance(entry, dict) and entry.get("value") and entry.get("source") in ("problem", "answer"):
                notes.append(studio_memory.add(project, "context", f"{track} {field.replace('_', ' ')}: {entry['value']}",
                                               "the person's words in the Home conversation; a later decision overrides it", "human", "home_handoff"))
    return notes
