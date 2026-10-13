"""Label rows from what a person does to a decision.

* a stage approved: the rule's choice stands (``label == rule_answer``) or the person chose another option;
* a solution saved by a person: every column judgment it states (forbidden, identifier) and the problem-level
  choices (target, task, split, metric) on the first save; afterwards only what changed;
* a gate approved: the solution gate confirms the solution's judgments; the holdout gate, the final choice.

Only a person's action is a label (``actor == "human"`` or a signed-in person): an agent's approval is the rule
approving itself. Every label is ``human_exposed``: the person saw the rule's answer first (ADR 0008's rule: such
labels train and monitor a model, never promote it). A label points at the latest sample with the same key in the
project (or, for a contract field, in the draft the project was built from), when one exists, and always carries
the key, so a label written before its sample still joins by key at export time. A label's reason is one of
``schema.REASONS``: never the words a person typed.
"""

from __future__ import annotations

from typing import Any

from .schema import key_of, label, split_kind
from .store import SampleStore


def _person(user_id: str | None = None) -> tuple[str | None, str | None]:
    """The signed-in person, when there is one (package 10.2)."""
    try:
        from ..accounts.principal import current

        person = current()
        if person is not None and person.user_id:
            return user_id or person.user_id, person.who
    except Exception:  # noqa: BLE001 — accounts are optional
        pass
    return user_id, None


def _draft_of(project: dict[str, Any]) -> str | None:
    source = (project.get("data") or {}).get("source") or {}
    return source.get("draft_id") if isinstance(source, dict) and source.get("kind") == "draft" else None


class _Index:
    """The latest sample per key for one project, read once (an approval of the data stage labels every column)."""

    def __init__(self, store: SampleStore | None, project: dict[str, Any]):
        self.project_id = project.get("id")
        self.latest: dict[str, dict[str, Any]] = {}
        if store is None:
            return
        try:
            rows = store.samples(self.project_id)
            draft = _draft_of(project)
            if draft:
                rows = store.samples(draft_id=draft) + rows  # the brief's samples, before the project's own
        except Exception:  # noqa: BLE001 — the key still joins at export
            rows = []
        for r in sorted(rows, key=lambda r: str(r.get("at") or "")):
            self.latest[r["key"]] = r

    def one(self, point: str, subject: str | None, value: Any, who: str, *, reason: str = "", move: str | None = None,
            user_id: str | None = None) -> dict[str, Any]:
        key = key_of(point, subject)
        return label(self.latest.get(key), key, self.project_id, value, "human_exposed", who, user_id=user_id, reason=reason, move=move)


STAGE_POINTS = {"features": "feature.recipe", "models": "model.family", "final": "optimization.config"}


def stage_approved(store: SampleStore | None, project: dict[str, Any], stage: str, record: dict[str, Any] | None,
                   choice: str | None, actor: str = "human", user_id: str | None = None) -> list[dict[str, Any]]:
    """The approval of a stage: a label for the stage's own decision (the chosen option, the rule's when none was given)
    and, for the data and leakage stages, a confirmation of every column judgment the stage made."""
    user_id, who = _person(user_id)
    if actor != "human" and who is None:
        return []
    who = who or actor
    index = _Index(store, project)
    decision = (record or {}).get("decision") or {}
    rows = []
    point = STAGE_POINTS.get(stage)
    if point and (choice or decision.get("selected") is not None):
        value = choice or decision.get("chosen") or decision.get("selected")
        rows.append(index.one(point, None, value, who, move="approve_stage", user_id=user_id,
                              reason="the rule's choice, approved" if value == decision.get("selected") else "chosen by the person"))
    evidence = (record or {}).get("evidence") or {}
    if stage == "data":
        for column, role in (evidence.get("column_roles") or {}).items():
            rows.append(index.one("column.role", str(column), role, who, move="approve_stage", user_id=user_id, reason="stage approved as computed"))
    if stage == "leakage":
        declared = {str(c) for c in evidence.get("declared_leakage_features") or []}
        flagged = {str(f.get("feature")) for f in evidence.get("heuristic_review_candidates") or [] if isinstance(f, dict)}
        for column in sorted(declared | flagged):
            rows.append(index.one("column.leakage", column, "exclude" if column in declared else "review_flag", who,
                                  move="approve_stage", user_id=user_id, reason="stage approved as computed"))
    return rows


def _forbidden(solution: dict[str, Any] | None) -> set[str]:
    return {str(f.get("column")) for f in (solution or {}).get("forbidden") or [] if isinstance(f, dict)}


def _split(solution: dict[str, Any]) -> str:
    if solution.get("spatial"):
        return "spatial"
    if solution.get("time_column") and solution.get("group_column"):
        return "time_group"
    if solution.get("time_column"):
        return "time"
    if solution.get("group_column"):
        return "group"
    return "stratified" if solution.get("task") in ("binary", "multiclass") else "random"  # what the engine does with no key


def solution_saved(store: SampleStore | None, project: dict[str, Any], before: dict[str, Any] | None, after: dict[str, Any],
                   who: str = "human", user_id: str | None = None, move: str = "set_solution") -> list[dict[str, Any]]:
    """A person saved the solution: labels for every column judgment it states (forbidden, identifier) and for the
    problem-level choices (target, task, split, metric, time column). Only what changed from the solution before is
    labelled a second time; the first save labels everything it states."""
    user_id, person = _person(user_id)
    who = person or who
    index = _Index(store, project)
    rows = []
    old_f, new_f = _forbidden(before), _forbidden(after)
    old_ids = {str(c) for c in (before or {}).get("identifiers") or []}
    new_ids = {str(c) for c in after.get("identifiers") or []}
    columns = [str(c) for c in ((project.get("data") or {}).get("columns") or [])]
    first = before is None
    reason = "first save of the solution" if first else ""
    for column in sorted(set(columns) | new_f | new_ids | old_f | old_ids):
        if column == after.get("target"):
            continue
        was, now = column in old_f, column in new_f
        if first or was != now:
            rows.append(index.one("column.forbidden", column, "exclude" if now else "clear", who, move=move, user_id=user_id,
                                  reason=reason or ("excluded by the person" if now else "cleared by the person")))
        was, now = column in old_ids, column in new_ids
        if first or was != now:
            rows.append(index.one("column.identifier", column, "identifier" if now else "feature", who, move=move, user_id=user_id, reason=reason))
    for field, point in (("target", "contract.target_column"), ("task", "contract.task"), ("metric", "metric.primary"),
                         ("time_column", "contract.time_column"), ("labels", "contract.label_coverage")):
        value = after.get(field)
        if value is None or value == "":
            continue
        if first or (before or {}).get(field) != value:
            rows.append(index.one(point, None, value, who, move=move, user_id=user_id, reason=reason))
    if after.get("prediction_moment") and (first or (before or {}).get("prediction_moment") != after["prediction_moment"]):
        rows.append(index.one("contract.prediction_moment", None, "stated", who, move=move, user_id=user_id, reason=reason))  # the words stay out
    split = _split(after)
    if first or _split(before or {}) != split:
        rows.append(index.one("split.strategy", None, split_kind(split), who, move=move, user_id=user_id, reason=reason))
    return rows


def gate_approved(store: SampleStore | None, project: dict[str, Any], gate: str, by: str, reason: str = "",
                  user_id: str | None = None) -> list[dict[str, Any]]:
    """A gate approval confirms what it covers: the solution gate, the solution's judgments (labelled as on a first
    save); the holdout gate, the latest final choice. The person's reason stays in the audit, not here."""
    user_id, person = _person(user_id)
    who = person or by
    if gate == "solution" and project.get("solution"):
        rows = solution_saved(store, project, None, project["solution"], who, user_id, move="approve_gate")
        for r in rows:
            r["reason"] = "gate approved"
        return rows
    if gate == "holdout":
        index = _Index(store, project)
        latest = index.latest.get(key_of("optimization.config", None))
        value = (project.get("decisions") or {}).get("final") or (latest or {}).get("rule_answer")
        if value is not None:
            return [index.one("optimization.config", None, value, who, move="approve_gate", reason="gate approved", user_id=user_id)]
    return []
