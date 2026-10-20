"""The one guarded write of the curators (32.9): every placement a curator proposes, a rule's or a model's, is checked here
before it counts, and a refused one is kept with why (never silently dropped).

``check(layout, decision, record)`` refuses a placement:

- whose model type or folder the layout does not allow (a path past ``layout.sets``);
- that is not a valid :class:`dclab_core.store.CurationDecision`;
- of a record whose consent does not allow training (anything but ``train``) into a set;
- that keeps a decision-model example as ``keep`` with no confirmed label (a rule's or a teacher's answer is ``silver`` at
  most, kept apart with its weight);
- whose folder is not one its model type has (a stage, a process, what is judged, a corpus, a system: :data:`FOLDERS`).

A record's split is never the curator's: the builders take it from the record, and a sealed record stays sealed.
"""

from __future__ import annotations

from typing import Any, Optional

from dclab_core.store import CONFIRMED, CurationDecision, LEDGER_SOURCES, StoreRecord

DECISION_MODELS = ("m1_students", "m2_workflow_decisions", "m3_ml_decisions", "m6_stepwise_decisions")
FOLDERS = {  # the first part of a folder, per model type (None: any point or family name)
    "m2_workflow_decisions": ("dclab", "rnd", "internal"), "m4_workflow_llm": ("understand_problem", "understand_data", "design_solution", "plan", "code", "report"),
    "m5_solver_llm": ("trajectories", "cases"), "m6_stepwise_decisions": ("walk", "graph", "job", "stage", "intern", "team", "conductor"),
    "m7_judges": ("code", "correction", "result", "step", "preferences", "plan", "solution", "report"), "m8_retrieval": ("evidence", "examples", "pitfalls", "cases"),
}


def check(layout: Any, decision: dict[str, Any], record: StoreRecord) -> tuple[Optional[CurationDecision], Optional[str]]:
    """(the decision, None) when it may count; (the decision marked refused, why) when it may not; (None, why) when it is not a
    decision at all."""
    try:
        made = CurationDecision.model_validate(decision)
    except Exception as exc:  # noqa: BLE001 - a model's malformed answer is refused and kept
        return None, f"not a curation decision: {type(exc).__name__}: {str(exc)[:300]}"
    why = None
    try:
        if made.folder:
            layout.sets(made.model_type, *made.folder.split("/"))
            known = FOLDERS.get(made.model_type)
            if known and made.folder.split("/")[0] not in known:
                why = f"the folder {made.folder.split('/')[0]!r} is not one of {made.model_type}'s ({', '.join(known)})"
    except ValueError as exc:
        why = f"the folder is not one the layout allows: {exc}"
    if why is None and made.quality != "out" and record.provenance.consent != "train":
        why = f"the record's consent ({record.provenance.consent}) does not allow training"
    if why is None and made.quality == "keep" and made.model_type in DECISION_MODELS and record.kind == "decisions" and record.type == "label":
        source = LEDGER_SOURCES.get(str(record.body.get("source")), record.body.get("source"))
        if source not in CONFIRMED:
            why = f"a label from {source!r} is silver at most, never keep"
    if why is None:
        return made, None
    return made.model_copy(update={"used": False, "refused": why}), why


__all__ = ["check", "DECISION_MODELS"]
