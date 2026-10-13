"""The R&D studio as one consumer of dclab_core: a project to a brief and a plan, and back (package 17.1).

The studio keeps its own records (the project's goal and its ``Solution``). This adapter shows them as dclab_core
contracts and turns them back, losing nothing: the brief carries what the problem *is* (goal, task, target, moment,
entities, time); the plan carries what was *decided* (the metric, the columns kept out and why, identifiers, text
columns, the positive label, notes). The main product is the other consumer, with its own adapter (package 17.8).

A solution field is the person's statement only when a person signed the solution (the graph's solution gate) and the
signed solution is unchanged; otherwise it is an inference, with that reason, because the intern's tool can write a
solution before anyone signs it.
"""

from __future__ import annotations

from typing import Any, Optional

from dclab_core.contracts import Decision, LabelSourceFact, NamesFact, ProblemBrief, SolutionPlan, TaskFact, TextFact

PLAN_KEYS = {
    "metric": "metric.primary",
    "forbidden": "features.forbidden",
    "identifiers": "features.identifiers",
    "text_columns": "features.text_columns",
    "positive_label": "target.positive_label",
    "notes": "solution.notes",
}


def _signed(project: dict[str, Any]) -> Optional[dict[str, Any]]:
    from .studio import graph

    try:
        return project.get("signoff") if graph.solution_signed(project) else None
    except Exception:  # noqa: BLE001 - a project shape the graph does not know is simply unsigned
        return None


def _fact(cls: type, value: Any, field: str, signoff: Optional[dict[str, Any]]):
    if value in (None, "", []):
        return cls()
    if signoff:
        return cls(value=value, status="stated", quote=f"{field}: {value}", source="person")
    return cls(value=value, status="inferred", reason=f"from the project's solution ({field}), not signed by a person yet")


def _why(signoff: Optional[dict[str, Any]]) -> str:
    if signoff:
        return f"Set in the project's solution and signed by {signoff.get('by', 'a person')} on {str(signoff.get('at', ''))[:10]}."
    return "Set in the project's solution; not signed by a person yet."


def brief_from_project(project: dict[str, Any]) -> ProblemBrief:
    """What the studio knows about the problem."""
    goal = (project.get("goal") or "").strip()
    solution = project.get("solution") or {}
    signoff = _signed(project)
    group = solution.get("group_column")
    return ProblemBrief(
        goal=TextFact(value=goal, status="stated", quote=goal, source="person") if goal else TextFact(),
        task=_fact(TaskFact, solution.get("task"), "task", signoff),
        target_column=_fact(TextFact, solution.get("target"), "target", signoff),
        label_source=_fact(LabelSourceFact, "column" if solution.get("target") else None, "label source", signoff),
        prediction_moment=_fact(TextFact, solution.get("prediction_moment"), "prediction moment", signoff),
        entities=_fact(NamesFact, [group] if group else None, "group column", signoff),
        time_column=_fact(TextFact, solution.get("time_column"), "time column", signoff),
    )


def plan_from_project(project: dict[str, Any]) -> SolutionPlan:
    """What the studio's solution decided, as decisions (only those that are set)."""
    solution = project.get("solution") or {}
    signoff = _signed(project)
    decisions = []
    for field, key in PLAN_KEYS.items():
        value = solution.get(field)
        if value in (None, "", []):
            continue
        decisions.append(Decision(key=key, choice=value, reason=_why(signoff), source="person" if signoff else "rule",
                                  status="accepted" if signoff else "proposed"))
    return SolutionPlan(task=solution.get("task") or "unknown", decisions=decisions)


def project_fields(brief: ProblemBrief, plan: SolutionPlan) -> dict[str, Any]:
    """The project's goal and solution fields, back from a brief and a plan (unknown fields are None or empty)."""
    entities = brief.entities.value if brief.entities.known else None
    solution: dict[str, Any] = {
        "target": brief.target_column.value,
        "task": brief.task.value,
        "prediction_moment": brief.prediction_moment.value,
        "group_column": entities[0] if entities else None,
        "time_column": brief.time_column.value,
    }
    defaults: dict[str, Any] = {"metric": None, "forbidden": [], "identifiers": [], "text_columns": [], "positive_label": None, "notes": ""}
    for field, key in PLAN_KEYS.items():
        decision = plan.decision(key)
        solution[field] = decision.choice if decision is not None else defaults[field]
    return {"goal": brief.goal.value or "", "solution": solution}


STUDIO_METRICS = {"binary": ("roc_auc", "average_precision"), "multiclass": ("macro_f1",), "regression": ("mae",)}


def solution_fields_from_plan(plan: SolutionPlan, target: str, prediction_moment: str) -> dict[str, Any]:
    """The studio solution a dclab_core plan proposes (package 17.4): the split's columns, the metric (the nearest the
    studio's engine scores, with the plan's own named in the notes), the columns kept out and the identifiers."""
    def choice(key: str, default: Any = None) -> Any:
        decision = plan.decision(key)
        return decision.choice if decision is not None else default

    split = choice("split.strategy", {}) or {}
    metric = choice("metric.primary")
    allowed = STUDIO_METRICS.get(plan.task, ())
    notes = []
    if metric and metric not in allowed:
        notes.append(f"The plan's primary metric is {metric}; the studio scores {allowed[0] if allowed else 'its default'} for now.")
        metric = allowed[0] if allowed else None
    if isinstance(split, str):
        split = {"kind": split}
    time_column = split.get("time_column")
    forbidden = ([{"column": f["column"] if isinstance(f, dict) else str(f), "reason": (f.get("reason", "") if isinstance(f, dict) else
                   "Kept out by the plan.")[:600] or "Kept out by the plan."} for f in choice("features.forbidden", []) or []]
                 + [{"column": f["column"], "reason": ("Kept out until a person confirms when it is written: " + f["question"])[:600]}
                    for f in choice("features.to_confirm", []) or []])
    forbidden = [f for f in forbidden if f["column"] != time_column]  # the column that orders the split is not a feature anyway
    seen: set[str] = set()  # a column named twice (by the rules and a role, or forbidden and to confirm) is kept out once, first reason kept
    forbidden = [f for f in forbidden if not (f["column"] in seen or seen.add(f["column"]))]
    return {
        "target": target, "task": plan.task, "prediction_moment": prediction_moment,
        "group_column": split.get("group_column"), "time_column": time_column,
        **({"spatial": {"lat_column": split["lat_column"], "lon_column": split["lon_column"], "block_km": float(split["block_km"])}}
           if split.get("kind") == "spatial_block" and split.get("lat_column") and split.get("lon_column") and split.get("block_km") else {}),
        "forbidden": forbidden,
        "identifiers": list(choice("features.identifiers", []) or []), "metric": metric, "notes": " ".join(notes),
        **({"labels": "partial", "label_provenance": "the plan reads the labels as partial: an unmarked row is unknown"}
           if choice("labels.coverage") == "partial" else {}),
    }
