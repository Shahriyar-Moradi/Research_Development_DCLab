"""The context builder (package 37.2, docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): what the lead agent sees on each turn.

One function assembles it, within a budget: the case file whole (every fact with its source, the assumptions, the
unknowns, readiness and the question ceiling), the table's view (column names, kinds and what DCLab noticed, never a
value), memory lines and evidence when the caller has them, and the recent turns of the conversation, newest first until
the budget is spent. Older turns are dropped, never summarised by a model: what they established is already in the case
file, which is always shown whole. A turn the budget cannot hold is counted, so a reader knows the context was compacted.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Optional

from dclab_core import case as C

BUDGET_CHARS = 48_000  # about 12,000 tokens: the case and the recent turns, below the model's window with room to think
MIN_TURNS = 2  # the person's latest words and the turn before always fit, whatever the budget


def case_text(case: C.CaseFile) -> str:
    """The case file as the agent reads it: facts by field with their source, then assumptions, unknowns and readiness."""
    r = C.readiness(case)
    lines = ["The case file (version %d):" % case.version, "- the person's words so far (quote from these): "
             + " | ".join(f"\"{w}\"" for w in case.words)]
    for field in C.SINGLE:
        fact = getattr(case, field)
        lines.append(f"- {field}: " + (f"{fact.value} [{fact.status}" + (f", their words: \"{fact.quote}\"" if fact.quote else "")
                                         + (f", computed from {fact.computed}" if fact.computed else "") + "]" if fact else "unknown"))
    for label, facts in (("what the data cannot answer", case.cannot_answer), ("limits", case.constraints)):
        for fact in facts:
            lines.append(f"- {label}: {fact.value} [{fact.status}]")
    for i, fact in enumerate(case.assumptions):
        lines.append(f"- assumption {i}: {fact.value} [{fact.status}]")
    for u in case.unknowns:
        lines.append(f"- unknown ({u.field}, blocks {u.blocks}): {u.question}")
    if case.options:
        lines.append("- options shown: " + "; ".join(f"{o.title} ({o.track}/{o.tool})" for o in case.options))
    if case.chosen:
        lines.append(f"- the person chose: {case.chosen}")
    if case.route:
        lines.append(f"- work started: {case.route.track}/{case.route.tool}")
    lines.append(f"- questions asked {case.asked}, passed over {case.declined} (ceiling: {C.CEILING['asked']} asked or {C.CEILING['declined']} passed over)")
    lines.append("- readiness: " + ("ready to propose" if r.ready else "not ready: " + "; ".join(r.reasons))
                 + ("" if r.ready or not r.may_propose else " (you may propose now, stating what is open as assumptions)"))
    return "\n".join(lines)


def table_text(table: Optional[Mapping[str, Any]]) -> str:
    if not table:
        return "No table shared yet."
    cols = "; ".join(f"{c['name']} ({c.get('kind')}{', ' + ', '.join(c['noticed']) if c.get('noticed') else ''})" for c in table.get("columns") or [])
    return f"The shared table {table.get('table')}: {table.get('rows')} rows. Columns: {cols}"


def build(case: C.CaseFile, turns: list[Mapping[str, str]], table: Optional[Mapping[str, Any]] = None,
          memory: Iterable[str] = (), evidence: Iterable[str] = (), budget: int = BUDGET_CHARS) -> dict[str, Any]:
    """The turn's context: ``state`` (always whole) and the recent ``turns`` that fit, oldest first, with how many were
    dropped. ``turns`` are {"role": "user"|"assistant", "content"} in order, ending on the person's latest words."""
    parts = [case_text(case), table_text(table)]
    memory, evidence = [m for m in memory if m], [e for e in evidence if e]
    if memory:
        parts.append("What DCLab remembers about this person and project:\n" + "\n".join(f"- {m}" for m in memory))
    if evidence:
        parts.append("Evidence and lessons that may apply:\n" + "\n".join(f"- {e}" for e in evidence))
    state = "\n\n".join(parts)
    left = budget - len(state)
    kept: list[Mapping[str, str]] = []
    for i, turn in enumerate(reversed(turns)):
        size = len(str(turn.get("content") or ""))
        if i >= MIN_TURNS and size > left:
            break
        kept.append(turn)
        left -= size
    kept.reverse()
    return {"state": state, "turns": [dict(t) for t in kept], "dropped": len(turns) - len(kept), "chars": budget - left}


def messages(system: str, built: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The request's messages: the agent's instructions, the turns, and the state under the person's latest words (so the
    instructions stay the same from turn to turn and are cached)."""
    turns = [dict(t) for t in built["turns"]]
    note = f"\n\n[{built['dropped']} earlier turns are in the case file]" if built["dropped"] else ""
    tail = f"\n\n---\n{built['state']}{note}"
    if turns and turns[-1]["role"] == "user":
        turns[-1] = {**turns[-1], "content": f"{turns[-1]['content']}{tail}"}
    else:
        turns.append({"role": "user", "content": tail.strip()})
    return [{"role": "system", "content": system}, *turns]


def dumps(built: Mapping[str, Any]) -> str:
    return json.dumps(built, ensure_ascii=False)
