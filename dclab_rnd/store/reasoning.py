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


class Teacher:
    """One teacher for one run: one gateway client, so ``cap_eur`` caps the whole run (never each call); records whose consent
    does not allow training are never sent; the record's words are redacted again before they leave."""

    def __init__(self, gateway: Any = None, cap_eur: Optional[float] = None):
        from ..models import for_tools

        gateway = gateway or for_tools()
        self.client = gateway.client(PURPOSE) if gateway is not None else None
        if self.client is not None and cap_eur is not None:
            self.client.run_limit_eur = cap_eur
        self.prompt = PROMPT.read_text(encoding="utf-8")

    @property
    def spent_eur(self) -> float:
        return float(getattr(self.client, "spent_eur", 0.0) or 0.0)

    def explain(self, record: StoreRecord, answer: str) -> tuple[Optional[str], str]:
        from dclab_core import redact

        if self.client is None:
            return None, "no teacher model serves here"
        if record.provenance.consent != "train":
            return None, f"the record's consent ({record.provenance.consent}) does not allow training"
        whole = json.dumps(record.body, ensure_ascii=False, default=str, sort_keys=True)
        text = redact.redact_text(whole)[0]  # what leaves: contacts replaced
        try:
            reply = self.client.complete([{"role": "system", "content": self.prompt},
                                          {"role": "user", "content": f"The record:\n{text}\n\nThe answer to explain:\n{answer}"}])
        except Exception as exc:  # noqa: BLE001 - a refused or failed request (the cap reached): no reasoning, and why
            return None, f"{type(exc).__name__}: {str(exc)[:200]}"
        reasoning = str((reply or {}).get("content") or "").strip()
        why = check(reasoning, whole + "\n" + answer) if reasoning else "the teacher wrote nothing"  # grounded in the record itself
        return (None, why) if why else (reasoning, "checked: every number from its record")


def teacher(record: StoreRecord, answer: str, gateway: Any = None, cap_eur: Optional[float] = None) -> tuple[Optional[str], str]:
    """One record's reasoning (a run of one): see :class:`Teacher`."""
    return Teacher(gateway, cap_eur).explain(record, answer)


def run(layout: Any, *, cap_eur: float, limit: int = 100, yes: bool = False, gateway: Any = None, log: Any = print) -> dict[str, Any]:
    """Teacher reasoning for the kept exchanges that have no thinking of their own, within one cap for the whole run, each kept
    under ``raw/teachers/`` as ``reasoning/<record>.json`` (source teacher, weight :data:`TEACHER_WEIGHT`). Without ``yes`` it
    says what it would do and spends nothing."""
    from . import shapers
    from .store import Store, now

    todo = [r for r in shapers.records(layout, "model_requests") if r.body.get("whole") and not r.body.get("thinking") and r.body.get("reply")
            and r.provenance.consent == "train"][:limit]
    if not yes:
        log(f"would ask the teacher for {len(todo)} reasonings, capped at EUR {cap_eur} for the run; nothing sent (add --yes)")
        return {"planned": len(todo), "cap_eur": cap_eur, "sent": 0}
    t, store, kept, refused = Teacher(gateway, cap_eur), Store(layout), 0, {}
    for r in todo:
        reasoning, why = t.explain(r, str(r.body.get("reply")))
        if reasoning is None:
            refused[why.split(":")[0]] = refused.get(why.split(":")[0], 0) + 1
            if why.startswith(("RuntimeError: BudgetExceeded", "no teacher")):
                break
            continue
        store.keep("teachers", r.provenance.workspace, f"reasoning/{r.id.replace(':', '_').replace('#', '_')}.json",
                   json.dumps({"record": r.id, "reasoning": reasoning, "source": "teacher", "weight": TEACHER_WEIGHT, "at": now(),
                               "model": getattr(t.client, "model", None)}).encode(), consent="train")
        kept += 1
    log(f"kept {kept} teacher reasonings, spent EUR {t.spent_eur:.4f} of {cap_eur}; not kept: {refused}")
    return {"kept": kept, "spent_eur": round(t.spent_eur, 6), "refused": refused}


__all__ = ["recorded", "from_record", "check", "teacher", "Teacher", "run", "PURPOSE", "TEACHER_WEIGHT"]
