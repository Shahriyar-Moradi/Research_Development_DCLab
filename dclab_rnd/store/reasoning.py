"""The reasoning of every LLM example (32.10), always labelled by where it came from:

- ``recorded``: the model's own thinking, kept whole where it was made (the gateway's ``model_exchanges``: Claude's
  extended thinking; the AI-first expert's; the development sessions' thinking in their transcripts);
- ``from_record``: a rule's or a person's choice explained by code from its own record (what it read, the options, what
  was answered and why, what was used): it holds no claim beyond the record;
- ``teacher``: written afterwards by a teacher model (Claude through the gateway's ``reasoning_teacher`` purpose, capped,
  planned and reported) only where neither exists, and checked: every number in it must be a number of its record, and no
  sentence may claim more than a record can (``cited.overclaims``). A reasoning that fails is dropped: the example keeps
  no reasoning (it says why), never an unchecked one. Teacher reasoning weighs less, as a silver label does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from dclab_core.store import StoreRecord

PURPOSE = "reasoning_teacher"
PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "reasoning_teacher.md"
TEACHER_WEIGHT = 0.5


def recorded(exchange: dict[str, Any]) -> Optional[str]:
    """A kept exchange's thinking, whole (None when the model did not think, or its text was not kept)."""
    parts = [str(b.get("text") or b.get("thinking") or "") for b in exchange.get("thinking") or [] if isinstance(b, dict)]
    text = "\n\n".join(p for p in parts if p.strip())
    return text or None


def from_record(record: StoreRecord) -> Optional[str]:
    """A decision or a step explained from its own record, by code."""
    b = record.body
    if record.kind == "decisions" and record.type == "decision":
        state = ", ".join(f"{k} {v}" for k, v in sorted((b.get("state") or {}).items()) if isinstance(v, (str, int, float, bool)))
        answers = "; ".join(f"{a.get('by')} answered {a.get('value')}" + (f" because {a.get('reason')}" if a.get("reason") else "")
                            for a in b.get("answers") or [] if isinstance(a, dict))
        return (f"The decision is {b.get('point')}. What was known then: {state or 'nothing recorded'}. "
                f"The options: {', '.join(map(str, b.get('options') or [])) or 'yes or no'}. {answers or 'No answer was recorded'}. "
                f"The studio used the {b.get('used')}'s answer, {b.get('value_used')}.")
    if record.kind == "workflows" and record.type == "graph_move":
        checks = ", ".join(map(str, b.get("failed_checks") or []))
        return (f"The move {b.get('move')} from {b.get('from')} to {b.get('to')} was {b.get('status')}"
                + (f": it failed {checks}" if checks else "") + (f", by the rules {', '.join(map(str, b.get('rules') or []))}" if b.get("rules") else "") + ".")
    if record.kind == "workflows" and record.type == "stage_run":
        choices = "; ".join(f"{c.get('field')} {c.get('value')} ({c.get('reason')})" for c in (b.get("choices") or []) if isinstance(c, dict))
        return f"The stage {b.get('stage')} chose: {choices}." if choices else None
    return None


def check(reasoning: str, record_text: str) -> Optional[str]:
    """Why a teacher's reasoning cannot be kept (None when it can): a number its record does not hold, or a sentence that claims
    more than a record can."""
    from .. import cited

    extra = cited.numbers(reasoning) - cited.numbers(record_text, source=True)
    if extra:
        return f"numbers its record does not hold: {sorted(extra)[:5]}"
    if any(cited.overclaims(s) for s in cited.sentences(reasoning)):
        return "a sentence claims more than a record can (a guarantee, a proof, a cause, readiness)"
    return None


def teacher(record: StoreRecord, answer: str, gateway: Any = None, cap_eur: Optional[float] = None) -> tuple[Optional[str], str]:
    """(the checked reasoning, why) for one example's answer from its record; (None, why) when no model serves, or it failed its check."""
    from ..models import installed

    gateway = gateway or installed()
    client = gateway.client(PURPOSE) if gateway is not None else None
    if client is None:
        return None, "no teacher model serves here"
    if cap_eur is not None:
        client.run_limit_eur = cap_eur
    text = json.dumps(record.body, ensure_ascii=False, default=str, sort_keys=True)
    prompt = PROMPT.read_text(encoding="utf-8")
    reply = client.complete([{"role": "system", "content": prompt},
                             {"role": "user", "content": f"The record:\n{text}\n\nThe answer to explain:\n{answer}"}])
    reasoning = str((reply or {}).get("content") or "").strip()
    why = check(reasoning, text + "\n" + answer) if reasoning else "the teacher wrote nothing"
    return (None, why) if why else (reasoning, "checked: every number from its record")


__all__ = ["recorded", "from_record", "check", "teacher", "PURPOSE", "TEACHER_WEIGHT"]
