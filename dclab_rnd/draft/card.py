"""The draft's problem card (20.1): ``dclab_core.card`` read from what Home knows of a draft.

The brief is read from the person's words (the problem and their messages) and the files' column names, then each
answer the person gave on Home, and the solution they saved in the wizard, overrides what the words alone suggested.
What the files say comes from the pipeline: a label file joined to the table, or a list of known cases alone.
"""

from __future__ import annotations

import re
from typing import Any

from dclab_core.card import LABEL_WORDS, ProblemCard, problem_card
from dclab_core.contracts import CoverageFact, LabelSourceFact, TaskFact, TextFact
from dclab_core.understand import understand

from . import plan as plans

# the problem kind a person chose (Solution.kind, 16.1) as the card's family; places keep the task's family
KIND_FAMILY = {"binary": "classification", "multiclass": "classification", "text_tabular": "classification", "regression": "regression",
               "rare_event": "rare_event", "forecasting": "forecasting", "time_to_event": "time_to_event", "ranking": "ranking",
               "segmentation": "segmentation", "uplift": "uplift", "entity_tables": "entity_tables", "streams": "streams",
               "embedded": "text_or_images"}
UNIT_ANSWER = re.compile(r"^\s*(one|each|a single)\b", re.I)  # "One delivery", "Each order": a unit, not an entity


def _stated(value: str) -> TextFact:
    return TextFact(value=value[:4000], status="stated", quote=value[:4000], source="person")


def _task_of(column: dict[str, Any] | None) -> str | None:
    """The task an outcome column's profile implies: two values a yes or no, a few a class, many numbers a regression."""
    if not column:
        return None
    unique = int(column.get("unique") or 0)
    if unique == 2:
        return "binary"
    if column.get("kind") == "numeric" and unique > 20:
        return "regression"
    return "multiclass" if 2 < unique <= 20 else None


def card_of(draft: dict[str, Any]) -> ProblemCard:
    u = draft.get("understanding") or {}
    solution = draft.get("solution") or {}
    words = " ".join([str(draft.get("problem") or "")] + [str(m.get("text") or "") for m in draft.get("messages") or [] if m.get("role") == "user"])
    profile = (draft.get("analysis") or {}).get("profile") or {}
    columns = [str(c.get("name")) for c in profile.get("columns") or []]
    active = next((a for a in draft.get("assets") or [] if a.get("id") == draft.get("active_asset")), {}) or {}
    brief = understand(str(draft.get("problem") or ""), {str(active.get("name") or "table"): columns} if columns else None)
    update: dict[str, Any] = {}
    own = active.get("suggestion") or {}  # a sample or a synthetic table knows its outcome and its moment (as the wizard uses)
    target = solution.get("target") or u.get("target") or own.get("target")
    if target and target in columns:
        update["target_column"] = _stated(str(target))
    elif target and not columns:  # before any data, the outcome in words; with a table, only a column or a file of cases is one
        update["target_definition"] = _stated(str(target))  # ("it is anomaly detection", typed when asked the outcome, is not)
    task = solution.get("task") or own.get("task") or _task_of(next((c for c in profile.get("columns") or [] if c.get("name") == target), None))
    if task in ("binary", "multiclass", "regression"):
        source = "person" if solution.get("task") else "data"
        update["task"] = (TaskFact(value=task, status="stated", quote=task, source="person") if source == "person"
                          else TaskFact(value=task, status="inferred", source="data", reason=f"the outcome column {target}'s values"))
    moment = solution.get("prediction_moment") or u.get("prediction_moment") or own.get("prediction_moment")
    if moment:
        update["prediction_moment"] = _stated(str(moment))
    if u.get("action"):
        update["action"] = _stated(str(u["action"]))
    if u.get("unit"):
        update["unit"] = _stated(str(u["unit"]))
    elif u.get("entity") and UNIT_ANSWER.match(str(u["entity"])):
        update["unit"] = _stated(str(u["entity"]))
    labels = str(u.get("labels") or "")
    if labels:
        said = next((value for value, pattern in LABEL_WORDS if pattern.search(labels)), None)  # "no" is read first
        if said == "none":
            update["label_source"] = LabelSourceFact(value="none", status="stated", quote=labels[:4000], source="person")
        elif said == "column":
            update["label_source"] = LabelSourceFact(value="column", status="stated", quote=labels[:4000], source="person")
        else:
            from .api import _coverage

            update["label_coverage"] = CoverageFact(value=_coverage(labels), status="stated", quote=labels[:4000], source="person")
    if update:
        brief = brief.model_copy(update=update)
    cases = (draft.get("analysis") or {}).get("cases_table")
    has_labels = plans.label_asset(draft) is not None
    named = bool(target and target in columns)
    data = {"label_file": has_labels, "cases_only": bool(cases) and not has_labels and not named,
            "tables": sum(1 for a in draft.get("assets") or [] if a.get("status") == "ready")}
    return problem_card(brief, words=words, data=data, family=KIND_FAMILY.get(str(solution.get("kind") or "")))
