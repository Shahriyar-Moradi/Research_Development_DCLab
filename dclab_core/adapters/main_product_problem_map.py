"""A problem map (24.1) as DCLab's main product holds a problem.

The main product states one problem per ``ProblemSpec`` (an objective, a task type, a target, a unit, a moment, a metric) and has
no relation between two specs: nothing says that one reads the other's score, or that their scores are fused and then cut to a
budget. :func:`problem_specs` therefore gives one ``ProblemSpec`` proposal per sub-problem, each carrying its place in the map in
``constraints`` under the free-form key ``dclab_problem_map`` (the fingerprint, its part and rank, what it reads, what composes
it), and the composition once, at the map's level, for the host to keep beside the specs or to restate by a person. The task type
is the main product's where the family has one (the same table as :func:`main_product.problem_card`); the others say why not.
Nothing here is locked: a person locks a spec, and a map with a read that fails ``problem_map.validate`` gives no specs."""

from __future__ import annotations

from typing import Any

from dclab_core import problem_map as core
from dclab_core.adapters import main_product as mp

FIELDS = ("task_type", "business_objective", "target_column", "prediction_unit", "prediction_time_column", "prediction_horizon",
          "primary_metric", "constraints", "success_criteria")
KEY = "dclab_problem_map"


def _spec(m: core.ProblemMap, s: core.SubProblem) -> dict[str, Any]:
    limits = mp.recorded()["contracts"]["ProblemSpec"]["max_length"]
    notes: dict[str, str] = {}
    task = "multiclass" if s.family == "classification" and s.kind == "multiclass" else mp.FAMILY_TO_TASK.get(s.family)
    if task is None:
        notes["task_type"] = mp.FAMILY_NOTES.get(s.family, f"no task type for the family {s.family}")
    elif s.family == "rare_event":
        notes["task_type"] = "a rare event is binary; its imbalance is the main product's problem profile's to find"
    notes["target_column"] = ("no column is chosen yet: " + (s.outcome or "the outcome is not named")) if s.labels in ("file", "column", "unknown") \
        else "no labels: the main product's runs need a target (confirmed cases make it binary)"
    notes["prediction_time_column"] = "a moment in words, not a column: " + (s.moment or "not said")
    notes["primary_metric"] = "chosen later, by the plan"
    here = {"fingerprint": m.fingerprint, "sub_problem": s.id, "part": s.part, "family": s.family, "labels": s.labels,
            "predict_rank": s.predict_rank, "known_rank": s.known_rank,
            "reads": [{"sub_problem": r.sub_problem, "what": r.what, **({"via": mp._clip(r.via, 200)} if r.via else {})} for r in s.reads],
            "composed_by": [c.output for c in m.composition.steps if s.id in c.inputs]}
    objective = mp._clip(s.question or s.title or s.id, limits["business_objective"])
    unit = mp._clip(s.unit, limits["prediction_unit"]) if s.unit else None  # the contract's own limit (120) is under the main product's (128): nothing is cut
    if not s.unit:
        notes["prediction_unit"] = "the map does not say what one row is"
    fields = {"task_type": task, "business_objective": objective, "target_column": None, "prediction_unit": unit, "prediction_time_column": None,
              "prediction_horizon": None, "primary_metric": None, "constraints": {KEY: here}, "success_criteria": {}}
    notes["constraints"] = (f"{KEY} is a free-form key (parse_objective keeps unknown keys as they are): the sub-problem's place in the map; "
                            "the main product does not relate specs, so the host keeps the map or a person restates the links")
    return mp._out("ProblemSpec", fields, notes, decision_point="spec.objective", about=f"{s.id} of the problem map {m.fingerprint}")


def problem_specs(problem_map: Any) -> dict[str, Any]:
    """One ``ProblemSpec`` proposal per sub-problem, the composition once, and what the main product cannot hold."""
    m = mp._as(core.ProblemMap, problem_map)
    issues = core.validate(m)
    if issues:
        raise ValueError("the map is not sound, so it maps onto nothing: " + "; ".join(issues))
    specs = {s.id: _spec(m, s) for s in m.sub_problems}  # a spec's constraints hold one sub-problem's place (at most 10 reads, each key clipped): far below json_max_bytes, and a test holds the largest
    composition = {"method": m.composition.method, "budget": m.composition.budget,
                   "steps": [{"kind": c.kind, "inputs": list(c.inputs), "output": c.output, **({"joins": mp._clip(c.joins, 200)} if c.joins else {})} for c in m.composition.steps]}
    return {"contract": "ProblemMap", "fingerprint": m.fingerprint, "size": m.size, "template": m.template,
            "problem_specs": specs, "composition": composition,
            "not_mapped": {"composition": "the main product has no object that relates specs or combines their outputs: the host stores the composition "
                                          "beside the specs (this dictionary) or a person restates it",
                           "parts": "a part groups sub-problems by unit and moment; it is carried in each spec's place in the map, not as a record"}}


__all__ = ["problem_specs", "FIELDS", "KEY"]
