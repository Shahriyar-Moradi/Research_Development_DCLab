"""A project's memory (package A5.2): short notes on what was decided, why, by whom and by which move.

A note is written when a solution is saved, a stage is approved and a gate is approved; when a person changes what an
agent decided (a solution the agent saved, the rule's choice of a stage), the note is a correction. Agents read the
latest notes as part of their state, so a second session on a project starts from what the first one decided instead
of asking again.

A note never holds a cell value: it is built from column names, the task, the metric, the prediction moment the
person wrote, stage and option ids and a reason, never from the table's values (the positive label, a value of the
target, is left out). A person can remove a note; it stays in the project, marked removed, and the removal is
written to the activity log.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any

SHOWN = 20  # notes an agent reads (the latest)
TEXT = 300  # characters of a decision or a reason


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _line(text: Any) -> str:
    """One line, cut to TEXT characters: a note is read inside an agent's prompt, where a line break would let a person's
    text pass for an instruction of its own."""
    text = " ".join(str(text or "").split())
    return text if len(text) <= TEXT else text[:TEXT - 1] + "…"


def add(project: dict[str, Any], kind: str, decision: str, reason: str, who: str, move: str, stage: str | None = None) -> dict[str, Any]:
    """Append a note to the project (the caller saves the project)."""
    note = {"id": "n" + secrets.token_hex(4), "at": now(), "kind": kind, "decision": _line(decision), "reason": _line(reason),
            "who": who, "move": move, "stage": stage, "removed": None}
    project.setdefault("memory", []).append(note)
    return note


def _solution_line(solution: dict[str, Any]) -> str:
    forbidden = [f["column"] for f in solution.get("forbidden") or []]
    parts = [f"target {solution.get('target')} ({solution.get('task')})",
             f"forbidden {', '.join(forbidden[:8])}{' …' if len(forbidden) > 8 else ''}" if forbidden else "nothing forbidden"]
    if solution.get("identifiers"):
        parts.append(f"identifiers {', '.join(solution['identifiers'][:6])}")
    if solution.get("time_column"):
        parts.append(f"time column {solution['time_column']}")
    if solution.get("group_column"):
        parts.append(f"group column {solution['group_column']}")
    if solution.get("metric"):
        parts.append(f"metric {solution['metric']}")
    return "; ".join(parts)


def _changes(before: dict[str, Any], after: dict[str, Any]) -> str:
    out = []
    old_f, new_f = {f["column"] for f in before.get("forbidden") or []}, {f["column"] for f in after.get("forbidden") or []}
    if new_f - old_f:
        out.append("forbade " + ", ".join(sorted(new_f - old_f)[:6]))
    if old_f - new_f:
        out.append("allowed " + ", ".join(sorted(old_f - new_f)[:6]))
    for key in ("target", "task", "time_column", "group_column", "metric"):
        if before.get(key) != after.get(key):
            out.append(f"{key.replace('_', ' ')} {before.get(key) or 'none'} → {after.get(key) or 'none'}")
    if set(before.get("identifiers") or []) != set(after.get("identifiers") or []):
        out.append("changed the identifiers")
    return "; ".join(out) or "edited the solution"


def solution_saved(project: dict[str, Any], before: dict[str, Any] | None, after: dict[str, Any], who: str) -> dict[str, Any]:
    """The note for a saved solution: a decision, or a correction when a person changes what the agent saved."""
    by_agent = (project.get("solution_saved_by") or None) == "agent"
    project["solution_saved_by"] = who
    reason = f"Prediction moment: {after.get('prediction_moment') or 'not written'}"
    if before and who == "human" and by_agent:
        return add(project, "correction", "The person changed the agent's solution: " + _changes(before, after), reason, who, "set_solution")
    return add(project, "decision", "Solution saved: " + _solution_line(after), reason, who, "set_solution")


def stage_approved(project: dict[str, Any], stage: str, decision: dict[str, Any] | None, choice: str | None, who: str) -> dict[str, Any]:
    decision = decision or {}
    chosen = choice or decision.get("chosen") or decision.get("selected")
    if who == "human" and choice and choice != decision.get("selected"):  # a correction is a person changing the rule's choice
        return add(project, "correction", f"Stage {stage} approved with {choice} instead of the rule's {decision.get('selected')}",
                   f"Rule: {decision.get('rule') or 'the stage rule'}", who, "approve_stage", stage)
    return add(project, "decision", f"Stage {stage} approved" + (f" ({chosen})" if chosen else ""), f"Rule: {decision.get('rule') or 'the stage rule'}",
               who, "approve_stage", stage)


def gate_approved(project: dict[str, Any], gate: str, who: str, reason: str) -> dict[str, Any]:
    note = add(project, "decision", f"Gate {gate} approved", reason or "no reason given", who, "approve_gate")
    note["gate"] = gate
    return note


def _gate_holds(project: dict[str, Any], gate: str | None) -> bool:
    """A gate approval holds while the graph still counts it: the solution's while the signed solution is unchanged,
    the holdout's until the final stage uses it."""
    from . import graph

    if gate == "solution":
        return graph.solution_signed(project)
    if gate == "holdout":
        return any(a.get("gate") == "holdout" and not a.get("used") for a in project.get("approvals") or [])
    return True


def data_replaced(project: dict[str, Any], filename: str, who: str = "human") -> None:
    """New data clears the solution and every stage, so what the notes settled no longer holds: each active note is
    marked removed by the system (kept, with the reason), and one note says what happened."""
    for note in project.get("memory") or []:
        if not note.get("removed"):
            note["removed"] = {"at": now(), "by": "system", "reason": "the data was replaced: the solution and every stage were cleared"}
    project.pop("solution_saved_by", None)
    add(project, "decision", f"Data replaced with {filename}; the solution and every stage were cleared", "Earlier decisions were about other data", who, "attach_data")


def active(project: dict[str, Any], limit: int = SHOWN) -> list[dict[str, Any]]:
    """The notes an agent reads, latest last. An approval counts only while it holds: the latest approval of a stage
    that is still approved, and the latest approval of a gate the graph still counts (an earlier one, one whose stage
    was cleared by a later change, or a signed solution that was changed since would read as settled when it is not)."""
    stages = project.get("stages") or {}
    notes = [n for n in project.get("memory") or [] if not n.get("removed")]
    latest: dict[tuple[str, str], str] = {}
    for n in notes:
        if n.get("move") in ("approve_stage", "approve_gate"):
            latest[(n["move"], n.get("stage") or n["decision"])] = n["id"]
    held = [n for n in notes if n.get("move") not in ("approve_stage", "approve_gate")
            or (latest.get((n["move"], n.get("stage") or n["decision"])) == n["id"]
                and (_gate_holds(project, n.get("gate") or n["decision"].split()[1]) if n["move"] == "approve_gate"
                     else (stages.get(n.get("stage")) or {}).get("status") == "approved"))]
    return held[-limit:]


def summary(project: dict[str, Any], limit: int = SHOWN) -> list[str]:
    """What an agent reads: one line per note, latest last."""
    return [f"{n['at'][:10]} · {n['kind']} by {n['who']} ({n['move']}{', ' + n['stage'] if n.get('stage') else ''}): {n['decision']}. {n['reason']}"
            for n in active(project, limit)]


def remove(store: Any, project_id: str, note_id: str, by: str, reason: str) -> dict[str, Any]:
    """A person removes a note: it stays in the project, marked removed, and the activity log keeps the removal."""
    project = store.get(project_id)
    note = next((n for n in project.get("memory") or [] if n["id"] == note_id), None)
    if note is None:
        raise KeyError(note_id)
    if note.get("removed"):
        return note
    note["removed"] = {"at": now(), "by": by, "reason": str(reason or "")[:TEXT]}
    store.save(project)
    store.log(project_id, "memory_note_removed", {"note": note_id, "decision": note["decision"], "by": by, "reason": note["removed"]["reason"]})
    return note
