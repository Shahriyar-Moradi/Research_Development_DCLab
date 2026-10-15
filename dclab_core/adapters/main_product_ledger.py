"""DCLab's main product's decisions as :mod:`dclab_core.ledger` records and labels (package 19.6), and back.

Two of its tables hold decisions. ``semantic_decision_answers`` is Jev's ledger: per question, the rule's answer, Jev's
raw answer and confidence, the agreement, the level, the value used and, on a label row, the ground truth with its
source. ``project_decision_records`` is the project's governed history (a split plan created, a winner locked, an
operating point chosen, a problem spec locked). Both are read here field by field from the column lists recorded at one
commit (``main_product_fields.json``); a record goes back as a ``semantic_decision_answers`` row
(:func:`to_semantic_answer`), so a decision model trained here can answer there through the same columns.

What the rows do not hold is never guessed. The state Jev saw is not stored there (only its digest): a record read from
a row has an empty state, and the main product's side of 19.7 builds it with the same bands (``STATE_MISSING``). Jev's
raw answer is ``{"value": …}``: a probability for a yes/no purpose, a label for a choice, a level for a score; it is
banded here with each purpose's own thresholds (``JEV_BANDS``, read from its release), as the main product bands it.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Optional

from dclab_core.adapters.main_product import recorded
from dclab_core.ledger import POINTS, Answer, DecisionLabel, DecisionRecord, Subject

# each Jev purpose's acting band (apps/api/app/agents/semantic/releases.py at the recorded commit)
JEV_BANDS = {"column.is_identifier": {"yes_at": 0.90, "no_at": 0.10}, "feature.leakage_suspect": {"yes_at": 0.90, "no_at": None},
             "column.semantic_role": {"confidence_at": 0.80}, "command.intent_route": {"confidence_at": 0.80},
             "proposal.completeness": {}}
# the main product compares a leakage rule's answer as a flag (semantic/policy.py _comparable): these values say yes
LEAKAGE_RULE_YES = ("exclude", "review_flag")
# each purpose's evidence partition (governance/decision_points.py): what its state is read from
PARTITIONS = {"feature.leakage_suspect": "train", "column.missing_value_action": "train"}
USED = {"rule": "rule", "ai": "model_role", "human": "person"}  # policy_outcome → who the value came from ("review": none)
LABEL_SOURCES = {"blind": "blind", "benchmark": "benchmark", "user_decision_exposed": "user_decision_exposed"}
BLIND_WEIGHT_NOTE = ("The main product keeps a blind case's sampling probability with its draw, not on this row "
                     "(ADR 0008 4); weight it from there before any estimate.")
STATE_MISSING = "semantic_decision_answers keeps the state's digest (question_digest), not the bands: the main product builds them"
# project_decision_records types that are decisions at a point of the ledger, and the ones that are not (and why)
DECISION_TYPE_POINTS = {"split_plan_created": "split.strategy", "winner_locked": "model.family",
                        "operating_point_chosen": "threshold.operating_point", "problem_spec_locked": "problem.ready",
                        "champion_promoted": "model.promote", "experiment_accepted": "experiment.accept",
                        "experiment_rejected": "experiment.accept"}
# the value each type decided, from the facts its service writes (split_plan_service, operating_point_service,
# decision_record_service.winner_locked_record); the others are a yes or no carried by their type and state
FACT_VALUES = {"split_plan_created": "holdout_strategy", "operating_point_chosen": "threshold"}
FLAG_TYPES = {"problem_spec_locked": True, "champion_promoted": True, "experiment_accepted": True, "experiment_rejected": False}
DECISION_TYPES_NOT_LEDGERED = {
    "ref_initialized": "a project's ref created: bookkeeping, not a decision with options",
    "ref_moved": "a ref moved with its champion or experiment decision, which is read",
    "proposal_accepted": "a person's verdict on an agent proposal: a label once agent_proposals is read",
    "proposal_rejected": "a person's verdict on an agent proposal: a label once agent_proposals is read",
    "proposal_reverted": "an undo of an accepted proposal",
    "decision_point_resolved": "its point and value are in the proposal (agent_proposals), not read yet",
}
ACTORS = {"human": "person", "rule": "rule", "agent": "model_role"}


def check_columns(table: str, row: Mapping[str, Any]) -> None:
    unknown = set(row) - set(recorded()["experience_tables"][table])
    if unknown:
        raise ValueError(f"{table}: columns not recorded at the read commit: {sorted(unknown)}; read the main product again")


def _at(value: Any) -> str:
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")
    return str(value)


def _id(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _value(blob: Any) -> Any:
    return blob.get("value") if isinstance(blob, Mapping) else None


def point_of(purpose: str) -> Optional[str]:
    return next((k for k, p in POINTS.items() if purpose in (p.jev, p.main_product)), None)


def banded(purpose: str, raw: Any, confidence: Optional[float]) -> Any:
    """Jev's raw answer as a value in the rule's vocabulary, as ``semantic/policy.band`` reads it: a yes/no purpose's
    probability against its yes and no bands, a choice only at its confidence, a score never (display only)."""
    band, value = JEV_BANDS.get(purpose, {}), _value(raw)
    if "yes_at" in band:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        if value >= band["yes_at"]:
            return True
        return False if band.get("no_at") is not None and value <= band["no_at"] else None
    if "confidence_at" in band:
        return value if confidence is not None and confidence >= band["confidence_at"] else None
    return None  # a score is never in its acting band


def _rule_value(purpose: str, value: Any) -> Any:
    if purpose == "feature.leakage_suspect":
        return value in LEAKAGE_RULE_YES or value is True
    return value


def from_semantic_answer(row: Mapping[str, Any], chain: Optional[Mapping[str, Mapping[str, Any]]] = None
                         ) -> tuple[Optional[DecisionRecord], Optional[DecisionLabel]]:
    """One ``semantic_decision_answers`` row: an answer row (``labels_version`` 0) is a decision record; a label row
    (1 and later, superseding the row before it) is a label of the decision at the root of its chain, and no second
    record. ``chain`` maps the rows' ids to rows, for a label that supersedes another label. (None, None) for a purpose
    with no point."""
    check_columns("semantic_decision_answers", row)
    purpose = str(row.get("purpose") or row.get("decision_point_key") or "")
    point = point_of(purpose)
    if point is None:
        return None, None
    version = int(row.get("labels_version") or 0)
    if version >= 1:
        return None, _label(row, point, chain or {})
    confidence = float(row["confidence"]) if row.get("confidence") is not None else None
    rule_raw = _value(row.get("rule_answer"))
    answers = [Answer(by="rule", answerer=f"rule:{purpose}", value=_rule_value(purpose, rule_raw), raw=rule_raw)] if row.get("rule_answer") is not None else []
    if row.get("answer") and row.get("agreement") != "unavailable":  # Jev answered (an unavailable row has none)
        answers.append(Answer(by="model_role", answerer=f"{row.get('model_id')}@{row.get('release_version')}",
                              value=banded(purpose, row.get("answer"), confidence), raw=_value(row.get("answer")), confidence=confidence,
                              source=_id(row.get("llm_invocation_id"))))
    outcome = str(row.get("policy_outcome") or "rule")
    if outcome not in USED and outcome != "review":
        raise ValueError(f"policy_outcome {outcome!r} is not one the recorded commit writes")
    subjects = POINTS[point].subjects
    record = DecisionRecord(id=f"mp-sda-{row['id']}", point=point, subject=Subject(kind="project" if "project" in subjects else subjects[0],
                                                                                    key=str(row.get("question_key") or "?")[:200]),
                            answers=answers, value_used=_rule_value(purpose, _value(row.get("value_used"))) if outcome != "ai" else _value(row.get("value_used")),
                            used=USED.get(outcome, "none"),
                            review=outcome == "review", level=int(row.get("level") or 0),
                            evidence_partition=row.get("evidence_partition") or PARTITIONS.get(purpose, "metadata"),
                            workspace=_id(row.get("workspace_id")), project=_id(row.get("project_id")), run=_id(row.get("agent_run_id")),
                            experiment=_id(row.get("experiment_id")), dataset=_id(row.get("dataset_id")), at=_at(row.get("created_at")))
    return record, None


def _label(row: Mapping[str, Any], point: str, chain: Mapping[str, Mapping[str, Any]]) -> Optional[DecisionLabel]:
    source = LABEL_SOURCES.get(str(row.get("label_source") or ""))
    if not source or row.get("ground_truth") is None:
        return None
    previous = _id(row.get("supersedes_label_id"))
    root, seen = previous, set()
    while root in chain and int(chain[root].get("labels_version") or 0) >= 1 and root not in seen:  # walk back to the answer row
        seen.add(root)
        root = _id(chain[root].get("supersedes_label_id"))
    if int(row.get("labels_version") or 0) > 1 and previous not in chain:
        raise ValueError("a label that supersedes another label needs the chain (its rows) to find the decision it labels")
    return DecisionLabel(id=f"mp-sdl-{row['id']}", record=f"mp-sda-{root}", point=point, value=_value(row["ground_truth"]), source=source,
                         by=_id(row.get("labeled_by_user_id")) or source, at=_at(row.get("labeled_at") or row.get("created_at")),
                         supersedes=f"mp-sdl-{previous}" if int(row.get("labels_version") or 0) > 1 else None,
                         weight_note=BLIND_WEIGHT_NOTE if source == "blind" else None)


def to_semantic_answer(record: DecisionRecord, model_id: str, release_version: str, question_digest: str) -> dict[str, Any]:
    """A record as a ``semantic_decision_answers`` answer row's content (the ids the main product assigns left out):
    how a decision model trained here answers there through the same columns. Only a Jev purpose can go back, and
    only with its workspace."""
    point = POINTS[record.point]
    if point.jev is None:
        raise ValueError(f"{record.point} is not one of Jev's purposes: the main product has no answer row for it")
    if not record.workspace:
        raise ValueError("a row needs its workspace (workspace_id is NOT NULL there)")
    rule, model = record.answer("rule"), record.answer("classifier") or record.answer("model_role")
    rule_value = None if rule is None else (rule.raw if rule.raw is not None else rule.value)
    raw, confidence = (as_raw(record.point, model) if model else (None, None))
    if raw is None:
        model = None  # an answer with no value is no answer
    in_band = banded(point.jev, {"value": raw}, confidence) if model else None  # the main product's own bands decide, never the answerer's
    if model is None:
        agreement = "unavailable"
    elif in_band is None:
        agreement = "abstain"
    else:
        agreement = "agree" if rule is not None and _comparable(point.jev, rule.value) == _comparable(point.jev, in_band) else "disagree"
    return {"decision_point_key": point.main_product or point.jev, "purpose": point.jev, "release_version": release_version, "model_id": model_id,
            "question_key": record.subject.key, "question_digest": question_digest, "data_class": "metadata",
            "evidence_partition": record.evidence_partition or PARTITIONS.get(point.jev, "metadata"),
            "primitive": point.answer if point.answer in ("noul", "choice", "score") else "choice",
            "answer": {"value": raw} if model else {},
            "confidence": confidence, "in_acting_band": in_band is not None,
            "rule_answer": {"value": rule_value} if rule else None, "agreement": agreement, "level": record.level,
            "policy_outcome": record.policy_outcome,
            # the rule's own words when its value was used (a leakage rule says "exclude", read as yes)
            "value_used": {"value": rule.raw if rule is not None and rule.raw is not None and record.used in ("rule", "none")
                           and record.value_used == rule.value else record.value_used},
            "project_id": record.project, "workspace_id": record.workspace, "agent_run_id": record.run,
            "experiment_id": record.experiment, "dataset_id": record.dataset}


def as_raw(point: str, answer: Answer) -> tuple[Any, Optional[float]]:
    """An answer as the main product reads one (22.7): Jev's own raw answer as it came; a classifier's yes or no as the
    probability of yes with no confidence (a confident no is a probability near 0, never its confidence); a choice as
    its label with its confidence."""
    from dclab_core.answerers import _option

    if answer.value is None and answer.raw is None:
        return None, None  # no answer (a score's raw answer stands though it is never in its band)
    noul = POINTS[point].answer == "noul"
    if answer.by != "classifier":
        raw = answer.raw if answer.raw is not None else answer.value
        if noul and isinstance(raw, bool):  # a model's plain yes or no is read there as a probability of 1 or 0
            raw = 1.0 if raw else 0.0
        return raw, answer.confidence
    if noul:
        value = _option(answer.value)
        sure = 1.0 if answer.confidence is None else float(answer.confidence)
        return round(sure if value is True else 1.0 - sure, 6), None
    return answer.value, answer.confidence


def _comparable(purpose: str, value: Any) -> Any:
    return _rule_value(purpose, value) if purpose == "feature.leakage_suspect" else value


def from_decision_record(row: Mapping[str, Any]) -> Optional[DecisionRecord]:
    """One ``project_decision_records`` row at its point, by its actor; None for a type the ledger does not hold. The
    value is what the row's facts say was decided (a split's holdout strategy, the chosen threshold, the winning
    candidate) or, for a yes/no type, its type and state: a rejected row says no, a proposed one was not used. A value
    the row does not hold is None, never guessed."""
    check_columns("project_decision_records", row)
    kind = str(row.get("decision_type"))
    point = DECISION_TYPE_POINTS.get(kind)
    if point is None:
        return None
    by = ACTORS.get(str(row.get("actor_kind")), "rule")
    facts = row.get("facts") if isinstance(row.get("facts"), Mapping) else {}
    state = str(row.get("state") or "accepted")
    if kind in FLAG_TYPES:
        value: Any = FLAG_TYPES[kind] if state != "rejected" else not FLAG_TYPES[kind]
    elif kind == "winner_locked":
        value = f"candidate:{row['candidate_id']}" if row.get("candidate_id") else None
    else:
        fact = facts.get(FACT_VALUES.get(kind, ""))
        value = fact if isinstance(fact, (str, int, float)) and not isinstance(fact, bool) else None
    answerer = row.get("actor_rule") or (f"user:{row['actor_user_id']}" if row.get("actor_user_id") else None) or by
    reason = None if row.get("rationale_untrusted") else (str(row.get("rationale"))[:600] if row.get("rationale") else None)
    used = by if state != "proposed" else "none"
    return DecisionRecord(id=f"mp-pdr-{row['id']}", point=point, subject=Subject(kind="project", key=str(row.get("project_id") or "?")),
                          answers=[Answer(by=by, answerer=str(answerer)[:120], value=value, reason=reason)], value_used=value if used != "none" else None,
                          used=used, level=2 if by == "model_role" and used != "none" else 0,
                          workspace=_id(row.get("workspace_id")), project=_id(row.get("project_id")), run=_id(row.get("actor_agent_run_id")),
                          experiment=_id(row.get("experiment_id")), dataset=_id(row.get("dataset_id")), at=_at(row.get("event_at")))
