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


def _spec(m: core.ProblemMap, s: core.SubProblem, route: Any = None) -> dict[str, Any]:
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
    if route is not None:  # 24.3: the approach the router chose and the others with what blocks them: a proposal, never a setting of the main product
        here["route"] = {"chosen": route.chosen, "source": route.source, "candidates": [{"approach": c.approach, "blocked_by": list(c.blocking)} for c in route.candidates][:12]}
        notes["route"] = "the approach is a proposal in the free-form key: the main product's runs choose their own model families, and a person decides"
    objective = mp._clip(s.question or s.title or s.id, limits["business_objective"])
    unit = mp._clip(s.unit, limits["prediction_unit"]) if s.unit else None  # the contract's own limit (120) is under the main product's (128): nothing is cut
    if not s.unit:
        notes["prediction_unit"] = "the map does not say what one row is"
    fields = {"task_type": task, "business_objective": objective, "target_column": None, "prediction_unit": unit, "prediction_time_column": None,
              "prediction_horizon": None, "primary_metric": None, "constraints": {KEY: here}, "success_criteria": {}}
    notes["constraints"] = (f"{KEY} is a free-form key (parse_objective keeps unknown keys as they are): the sub-problem's place in the map; "
                            "the main product does not relate specs, so the host keeps the map or a person restates the links")
    return mp._out("ProblemSpec", fields, notes, decision_point="spec.objective", about=f"{s.id} of the problem map {m.fingerprint}")


def problem_specs(problem_map: Any, routes: Any = None) -> dict[str, Any]:
    """One ``ProblemSpec`` proposal per sub-problem, the composition once, and what the main product cannot hold. ``routes``: the router's
    route per sub-problem id (``dclab_core.solutions.Route`` or its JSON), kept beside each sub-problem's place in the map."""
    from dclab_core.solutions import Route

    m = mp._as(core.ProblemMap, problem_map)
    routes = {k: r for k, r in ((k, mp._as(Route, v)) for k, v in (routes or {}).items()) if r.map in (None, m.fingerprint)}  # a route made for another map is not this map's
    issues = core.validate(m)
    if issues:
        raise ValueError("the map is not sound, so it maps onto nothing: " + "; ".join(issues))
    specs = {s.id: _spec(m, s, routes.get(s.id)) for s in m.sub_problems}  # a spec's constraints hold one sub-problem's place (at most 10 reads, each key clipped): far below json_max_bytes, and a test holds the largest
    composition = {"method": m.composition.method, "budget": m.composition.budget,
                   "steps": [{"kind": c.kind, "inputs": list(c.inputs), "output": c.output, **({"joins": mp._clip(c.joins, 200)} if c.joins else {})} for c in m.composition.steps]}
    return {"contract": "ProblemMap", "fingerprint": m.fingerprint, "size": m.size, "template": m.template,
            "problem_specs": specs, "composition": composition,
            "not_mapped": {"composition": "the main product has no object that relates specs or combines their outputs: the host stores the composition "
                                          "beside the specs (this dictionary) or a person restates it",
                           "parts": "a part groups sub-problems by unit and moment; it is carried in each spec's place in the map, not as a record"}}


def holdout_plan(solution_map: Any) -> dict[str, Any]:
    """The solution map's one holdout design as the main product's ``HoldoutPlan`` proposal (24.4): every project of the programme takes it, and the
    composition reads the holdout once. A split by time and by entity together has no verified holdout there (its own planner answers 'unsupported'): the
    field says so and a person decides."""
    from dclab_core.solution_map import SolutionMap

    sm = mp._as(SolutionMap, solution_map)
    split = sm.holdout.split
    kind = "time_and_group" if split.strategy == "time_group" else split.strategy
    strategy = mp.SPLIT_TO_HOLDOUT.get(kind)
    notes = {"test_size": "the map pins the share and the seed so every leaf draws the same holdout; the main product's floor (0.2) holds", "random_state": "the map's seed",
             "reason": "one holdout for the whole programme: every leaf splits as the whole does, and the composition reads it once, after the person's gate"}
    if strategy == "unsupported":
        notes["strategy"] = ("rows repeat for an entity and the outcome moves over time: the main product has no verified combined holdout, and its own "
                             "planner answers 'unsupported' here too; a person decides")
    group = split.keys[0] if split.keys and strategy in ("group_disjoint", "unsupported") else None
    time = split.moment if strategy in ("temporal_future", "unsupported") else None
    if len(split.keys) > 1:
        notes["group_column"] = f"the map names {len(split.keys)} entity columns; the main product's plan takes one (the first)"
    fields = {"strategy": strategy, "test_size": split.share, "random_state": split.seed,
              "stratified": (strategy == "stratified_random") if strategy not in (None, "unsupported") else None,
              "group_column": group, "time_column": time,
              "reason": notes["reason"], "evidence": {"problem_map": sm.map, "read_by": sm.holdout.read_by, "blocks": len(sm.blocks)},
              "plan_version": mp.recorded()["contracts"]["HoldoutPlan"]["version"]}
    return mp._out("HoldoutPlan", fields, notes, decision_point="split.strategy", about=f"the one holdout of the solution map {sm.fingerprint}")


def programme_experiments(programme: Any) -> dict[str, Any]:
    """A programme (24.5) as the main product's experiments: one ``ExperimentPlan`` proposal per leaf that runs a model or a rule, each with the programme's one split.
    The main product runs each as an experiment of its own; it has nothing that relates experiments, so the composition (which leaves are combined, how, the
    budget) and the one read of the holdout stay in the programme document the host keeps. A split by time and entity together has no ``PlanSplit`` there: the
    field is empty and says so."""
    from dclab_core.programme import Programme

    p = mp._as(Programme, programme)
    strategy = mp.SPLIT_TO_HOLDOUT.get("time_and_group" if p.split.strategy == "time_group" else p.split.strategy)
    split = None
    split_note = ("test_size is left to the host (its floor is 0.2); every leaf of the programme takes the same split, so the holdout is one, and no leaf reads it on its own: "
                  "the composition does, once")
    if strategy in mp.recorded()["vocabularies"]["plan_split_strategies"]:
        split = {"strategy": strategy, "group_column": p.split.keys[0] if strategy == "group_disjoint" and p.split.keys else None,
                 "time_column": p.split.moment if strategy == "temporal_future" else None, "test_size": None}
    else:
        split_note = (f"the programme splits by {p.split.strategy.replace('_', ' and ')}: the main product has no PlanSplit for that (its planner answers 'unsupported'), "
                      "so a person decides; every leaf must still take the same one")
    plans: dict[str, Any] = {}
    for leaf in p.leaves:
        if leaf.status in ("composition", "dropped"):
            continue
        notes = {"split": split_note,
                 "families": f"the approach is {leaf.approach or 'not chosen yet'}: the host maps it onto its own registry families, and a person decides",
                 "primary_metric": "the programme judges a leaf at the review budget (recall of the confirmed cases among the budget's rows), a measure the main product's plan has no name for",
                 "target_column": "the leaf's outcome is in the problem map; a column is chosen when the host builds it",
                 "max_training_seconds": "dclab_core sets no time budget: the host's rule holds",
                 "missing_values": "dclab_core proposes no per-column imputation"}
        if leaf.status == "blocked":
            notes["status"] = f"blocked: {leaf.why}"
        fields = {"target_column": None, "primary_metric": None, "split": split, "families": None, "max_training_seconds": None, "missing_values": None}
        plans[leaf.block] = mp._out("ExperimentPlan", fields, notes, decision_point="split.strategy", about=f"{leaf.block} ({leaf.title}) of the programme {p.fingerprint}")
    return {"contract": "Programme", "fingerprint": p.fingerprint, "state": p.state, "experiment_plans": plans,
            "not_mapped": {"composition": "the main product has no object that relates experiments or combines their outputs: the host keeps the programme document",
                           "holdout_read": "the one read of the holdout (by the composition, after every leaf is chosen) is the programme's rule; the main product's runs each read a holdout "
                                           "of their own, so the host must not let a leaf score the holdout it shares",
                           "result": "the composed score and its difference from the best leaf have no row there: they belong to the programme's report"}}


__all__ = ["problem_specs", "holdout_plan", "programme_experiments", "FIELDS", "KEY"]
