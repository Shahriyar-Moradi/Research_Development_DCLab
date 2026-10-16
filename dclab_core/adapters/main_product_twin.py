"""The main product's rule-and-Jev pairs as :mod:`dclab_core.twin` steps (package 25.1).

The main product already answers each governed semantic question twice: ``semantic_decision_answers`` keeps the rule's
answer and Jev's (its model) side by side on one question digest, and a later label row says which was right. That is a
twin step at ``decision:<point>``: the rule is the ``system`` arm, Jev the ``ai_first`` arm (its model id and release
are the answerer), the question digest the input's fingerprint, and a label row's ground truth the truth (``label``).
A row of a purpose with no ledger point, or with no rule answer, is not a step; an answer outside Jev's acting band (the
main product's abstain) is an abstained arm, not a wrong one; of a chain of labels, the latest version is the truth,
and a label whose chain is not in the rows is left out (the step stays unjudged). What the rows do not hold (the state
Jev saw, its reasoning, its cost) is left empty, never guessed. Read field by field from the column lists recorded at
one commit (``main_product_fields.json``), through :mod:`dclab_core.adapters.main_product_ledger`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from dclab_core.adapters.main_product_ledger import from_semantic_answer
from dclab_core.twin import DECISION, ArmAnswer, Truth, TwinStep, step, with_truth

NOT_MAPPED = {
    "reasons": "semantic_decision_answers keeps no reasoning: the steps' reasons are empty",
    "cost_eur": "a row's cost is in llm_invocations, joined by llm_invocation_id: the host adds it",
    "tools": "Jev answers one question with no tools: the AI arm's tools are empty",
}


def twin_steps(rows: Iterable[Mapping[str, Any]], session: str) -> list[TwinStep]:
    """Answer rows paired with their labels (a label row supersedes its answer row, possibly through other labels)."""
    rows = list(rows)
    chain = {str(r["id"]): r for r in rows}
    out: dict[str, TwinStep] = {}
    labels: dict[str, tuple[int, Any]] = {}
    for row in rows:
        try:
            record, label = from_semantic_answer(row, chain)
        except ValueError:  # a label whose earlier labels are not in these rows: its decision stays unjudged
            continue
        if label is not None:
            version = int(row.get("labels_version") or 0)
            if label.record not in labels or version > labels[label.record][0]:
                labels[label.record] = (version, label)
            continue
        if record is None:
            continue
        rule = next((a for a in record.answers if a.by == "rule"), None)
        jev = next((a for a in record.answers if a.by == "model_role"), None)
        if rule is None:
            continue
        digest = str(row.get("question_digest") or record.id)
        system = ArmAnswer(arm="system", answerer=rule.answerer, value={"value": rule.value})
        if jev is None:
            ai = ArmAnswer(arm="ai_first", answerer="jev", status="skipped", note="Jev gave no answer (unavailable)")
        elif jev.value is None or row.get("agreement") == "abstain":
            ai = ArmAnswer(arm="ai_first", answerer=jev.answerer, status="abstained", note="outside Jev's acting band: the main product's abstain")
        else:
            ai = ArmAnswer(arm="ai_first", answerer=jev.answerer, value={"value": jev.value},
                           note=f"confidence {jev.confidence}" if jev.confidence is not None else None)
        out[record.id] = step(record.id, session, DECISION + record.point, record.at, {"question_digest": digest}, system, ai,
                              subject=record.subject.key, input_summary={"point": record.point, "question_digest": digest[:64]})
    for record_id, (_version, label) in labels.items():
        if record_id in out and label.value is not None:
            out[record_id] = with_truth(out[record_id], Truth(value={"value": label.value}, source="label", note=label.source))
    return list(out.values())
