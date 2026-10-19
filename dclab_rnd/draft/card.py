"""The draft's problem card (20.1): ``dclab_core.card`` read from what Home knows of a draft.

The brief is read from the person's words (the problem and their messages) and the files' column names, then each
answer the person gave on Home, and the solution they saved in the wizard, overrides what the words alone suggested.
What the files say comes from the pipeline: a label file joined to the table, or a list of known cases alone.
"""

from __future__ import annotations

import re
from typing import Any

from dclab_core.card import FAMILIES, SHAPED_AS, ProblemCard, label_said, problem_card, refine_family
from dclab_core.contracts import CoverageFact, LabelSourceFact, TaskFact, TextFact
from dclab_core.understand import understand

from . import plan as plans

# the problem kind a person chose (Solution.kind, 16.1) as the card's family; places keep the task's family
KIND_FAMILY = {"binary": "classification", "multiclass": "classification", "text_tabular": "classification", "regression": "regression",
               "rare_event": "rare_event", "forecasting": "forecasting", "time_to_event": "time_to_event", "ranking": "ranking",
               "segmentation": "segmentation", "uplift": "uplift", "entity_tables": "entity_tables", "streams": "streams",
               "embedded": "text_or_images", "anomaly": "anomaly"}
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


INPUTS_NOT = {"identifier", "entity", "time", "constant"}  # never an input; with them, a text column may be alone


def _text_only(data_card: dict[str, Any], target: Any) -> bool:
    """The first table's only inputs are long texts (beside its identifiers, sources, times and the outcome): what is
    learned is read from the words, through an encoder (20.10 v2). A table with other inputs beside a text is not."""
    tables = data_card.get("tables") or []
    if len(tables) != 1:
        return False
    inputs = [c for c in tables[0].get("columns") or [] if c.get("name") != target and c.get("purpose") not in INPUTS_NOT]
    return bool(inputs) and all(c.get("purpose") == "text" for c in inputs)


def _answers(u: dict[str, Any]) -> dict[str, Any]:
    """30.13: what the person said of the card's v2 fields on Home (the budget answer is the team's capacity)."""
    return {"decision": u.get("decision"), "decision_owner": u.get("decision_owner"), "costs": u.get("costs"),
            "capacity": u.get("capacity") or u.get("budget"), "baseline": u.get("baseline"), "success": u.get("success")}


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
    elif target:  # the outcome in words: before any data an outcome; with a table only for entity tables and streams (the
        update["target_definition"] = _stated(str(target))  # card decides: "it is anomaly detection", typed when asked the outcome, is not)
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
    elif u.get("entity") and UNIT_ANSWER.match(str(u["entity"])) and len(str(u["entity"]).split()) <= 4:
        update["unit"] = _stated(str(u["entity"]))  # "One customer"; "One biker can fake many rides" names a group, not the unit
    if solution.get("time_column") and solution["time_column"] in columns:
        update["time_column"] = _stated(str(solution["time_column"]))  # 20.4: the time the split follows, checked against the table
    elif u.get("time_column") and u["time_column"] in columns:
        update["time_column"] = _stated(str(u["time_column"]))  # 30.13: a fit edit the person accepted
    if u.get("judged_on"):  # 30.13: the fit's edit for noisy labels, accepted: judged on the confirmed labels only
        update["target_definition"] = _stated(str(u["judged_on"]))
    labels = str(u.get("labels") or "")
    if labels:
        said = label_said(labels)  # "no" is read first, a file said apart from it wins
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
            "tables": sum(1 for a in draft.get("assets") or [] if a.get("status") == "ready" and a.get("kind") != "labels"), "columns": bool(columns),
            "text_only": _text_only(draft.get("data_card") or {}, target)}
    said = (refine_family(u.get("family"), words, u.get("family_shape_only") == "yes")  # 20.2: said in an answer; 20.10 v2: a shape only
            if u.get("family") in FAMILIES else None)
    stated = KIND_FAMILY.get(str(solution.get("kind") or "")) or said
    if stated == said and said in SHAPED_AS and u.get("family_shape_only") == "yes":
        # 20.10 v4: an answer that gave only an output's shape is evidence, not a statement: the card is inferred without it
        # (the words, the data: a table of long text alone), and kept when it is a family of that shape; else the shape stands
        inferred = problem_card(brief, words=words, data=data, answers=_answers(u))
        if inferred.family.value in (said, *SHAPED_AS[said]) and not inferred.in_play:
            return inferred
    if stated == said and u.get("family_by") == "reader":
        # 20.10 v4: the reader's family is the model's reading of the person, not their statement: what the card infers from
        # what the person stated and the data (a label file: rare events) comes first; the reader's family names what the
        # inference left open or gave only as a shape
        inferred = problem_card(brief, words=words, data=data, answers=_answers(u))
        if inferred.family.value is not None and not inferred.in_play and inferred.family.value not in SHAPED_AS:
            return inferred
    return problem_card(brief, words=words, data=data, family=stated, answers=_answers(u))


def meeting(draft: dict[str, Any], store: Any, with_fit: bool = False) -> Any:
    """The problem card against the data card of the draft's tables (20.4), the card built again now with the outcome
    the draft names today (it may have been named after the data was analysed); the label file counts as joined when
    the draft's own label review found its key. None with no table. When the check itself fails, a blocking
    ``not_checked``: a build never goes past a check that did not run."""
    import sys

    from dclab_core.meet import Mismatch, ProblemDataCheck

    active = next((a for a in draft.get("assets") or [] if a.get("id") == draft.get("active_asset") and a.get("status") == "ready"), None)
    if active is None or not active.get("clean_file"):
        return (None, None) if with_fit else None
    try:
        import pandas as pd

        from dclab_core.datacard import card_of as data_card_of
        from dclab_core.meet import check

        from .plan import label_asset

        tables = {str(active.get("name") or "table"): pd.read_parquet(store.data_dir(draft["id"]) / active["clean_file"])}
        labels = store.data_dir(draft["id"]) / "labels_clean.parquet"
        if labels.exists():
            tables["labels"] = pd.read_parquet(labels)
        problem = card_of(draft)
        target = problem.brief.target_column.value
        main = next(iter(tables.values()))
        file = label_asset(draft)
        joined = bool(((file or {}).get("label_review") or {}).get("key")) if file is not None else None
        data = data_card_of(tables, target=target if target in main.columns else None)
        checked = check(problem, data, joined=joined)
        if with_fit:  # 30.13: G2's three-way verdict, from the same two cards
            from dclab_core.meet import fit

            return checked, fit(problem, data, checked)
        return checked
    except Exception as exc:  # noqa: BLE001
        print(f"meeting: not checked: {type(exc).__name__}", file=sys.stderr, flush=True)
        failed = ProblemDataCheck(mismatches=[Mismatch(code="not_checked", blocking=True,
                                                       question="DCLab could not check the problem against the data. Try the build again; "
                                                                "if it fails again, the draft's table needs a look.",
                                                       evidence={"error": type(exc).__name__})], ready=False)
        return (failed, None) if with_fit else failed


EDIT_FIELDS = {"target_column": "target", "unit": "unit", "family": "family", "time_column": "time_column", "labels": "labels",
               "target_definition": "judged_on"}  # a fit edit's card field, as Home keeps the person's answer


def accept_fit(draft: dict[str, Any], edits: list[dict[str, Any]]) -> list[str]:
    """30.13: the fit's edits a person accepted, written as their answers (the card is built again from them): the
    fields changed."""
    u = draft.setdefault("understanding", {})
    changed = []
    for e in edits:
        key = EDIT_FIELDS.get(str(e.get("field")))
        if key and e.get("value"):
            u[key] = str(e["value"])[:600]
            if key == "family":
                u["family_by"] = "person"
            changed.append(str(e["field"]))
    draft.setdefault("fit_accepted", []).extend({"field": e.get("field"), "code": e.get("code")} for e in edits if EDIT_FIELDS.get(str(e.get("field"))))
    return changed
