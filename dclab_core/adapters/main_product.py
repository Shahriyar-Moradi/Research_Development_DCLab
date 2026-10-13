"""dclab_core's outputs shaped like DCLab main product's contracts, field by field (package 17.8).

DCLab's main product (github.com/Shahriyar-Moradi/DCLab) keeps its own contracts: a versioned ``ProblemSpec``, the
engine's planners (``ProblemProfile``, ``MetricPlan``, ``HoldoutPlan``, ``ValidationPlan``, ``Objective`` and the
decision threshold), the ``ExperimentPlan`` that travels into a run with one field group per governed decision point,
and the experiment critic's ``ExperimentReview`` and the ``ExperimentChangeSet`` of a branch. Their field lists were
read from its code at one commit and recorded in ``main_product_fields.json``; this module maps onto them without
importing anything of the main product.

Every function returns a plain JSON dict::

    {"contract": ..., "main_product": {repository, commit, path, version}, "status": "proposal" | "evidence",
     "decision_point": ..., "fields": {exactly the recorded fields}, "notes": {field: why}, "excluded": {...}}

``fields`` holds exactly the recorded field names. A field dclab_core cannot fill is ``None`` with a note saying why
(never a guessed value); a value that needs a person's eye (inferred, translated, dropped parts) has a note too. What
is mapped is a *proposal*: the host adopts it through its own governed decision point, where its validator re-checks
it against its own data and a person accepts it at the point's level. Nothing here applies anything, and nothing that
came from holdout rows crosses into a field (``excluded`` says what was left out on purpose).
"""

from __future__ import annotations

import functools
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dclab_core.assess import stated_costs
from dclab_core.contracts import DataFindings, Decision, Finding, ProblemBrief, Review, SolutionPlan, Versioned
from dclab_core.plan import METRIC_ALIASES as CORE_METRIC_ALIASES

FIELDS_FILE = Path(__file__).resolve().with_name("main_product_fields.json")

# dclab_core's metric names -> the main product's, where they differ in name only
CORE_TO_MAIN_METRIC = {"average_precision": "pr_auc", "brier": "brier_score"}
# dclab_core metrics the main product cannot score as a metric (yet)
NO_MAIN_METRIC = {
    "recall_at_budget": "recall at a review budget has no metric in the main product; its operating points page shows "
                        "recall against the flagged share, and a recall constraint (metric_constraints) is the nearest field",
    "precision_at_budget": "precision at a review budget has no metric in the main product; a precision constraint "
                           "(metric_constraints) is the nearest field",
    "top_k_accuracy": "the main product has no top-k accuracy",
    "quantile_loss": "the main product has no quantile loss",
    "share_within_tolerance": "the main product has no share-within-tolerance metric; the success criterion stays text "
                              "for a person (mape or smape constraints are not the same thing)",
}

# dclab_core's model families -> the main product's open-ingest portfolio (engine/search/generator.py)
FAMILY_TO_PORTFOLIO = {
    "classification": {"logistic_regression": ["logistic_regression"], "random_forest": ["random_forest"],
                       "gradient_boosting": ["xgboost", "lightgbm", "catboost"]},
    "regression": {"ridge": ["linear_regression"], "random_forest": ["random_forest_regressor"],
                   "gradient_boosting": ["xgboost_regressor", "lightgbm_regressor", "catboost_regressor"]},
}
FAMILY_NOTES = {
    "gradient_boosting": "gradient_boosting is the portfolio's boosted families (xgboost, lightgbm, catboost); the "
                         "registry's sklearn gradient_boosting(_regressor) is not in the open-ingest portfolio",
    "ridge": "ridge maps to the portfolio's linear family, linear_regression (ridge is in the registry but not in the "
             "open-ingest portfolio)",
}

SPLIT_TO_HOLDOUT = {"stratified": "stratified_random", "random": "random", "group": "group_disjoint",
                    "time": "temporal_future", "time_and_group": "unsupported"}

# each plan decision: (the main product's decision point, the fields it reaches, how it travels)
DECISIONS: dict[str, tuple[Optional[str], list[str], str]] = {
    "task.framing": ("spec.objective", ["ProblemSpec.task_type"],
                     "The task type travels in a ProblemSpec proposal. Tiny classes, a capped target and partial labels "
                     "have no field there: they reach a person as notes and as limits in the report."),
    "labels.coverage": (None, [],
                        "The main product has no notion of partial labels: an unlabelled row is a negative there. A "
                        "person must know this before a run; it is a limit of every precision the run reports."),
    "labels.join": (None, [],
                    "No decision point joins a label file: the host builds the label column on this key before the "
                    "upload, and target.column then names it."),
    "split.strategy": ("split.strategy", ["ExperimentPlan.split", "HoldoutPlan", "ValidationPlan"],
                       "Cap L1: a person accepts the plan; a structure tighter than the rule's stops the run at "
                       "needs_input before run_holdout_lock, and a looser one is refused."),
    "metric.primary": ("spec.objective", ["ExperimentPlan.primary_metric", "ProblemSpec.primary_metric",
                                          "Objective.primary_metric", "MetricPlan.primary_metric"],
                       "Cap L1: the metric is the person's intent; the host re-validates it for the task."),
    "metric.secondary": (None, ["MetricPlan.secondary_metrics"],
                         "No decision point: the host's metric planner reports its own secondary metrics; these are a "
                         "proposal for what the report shows beside the primary metric."),
    "baseline": (None, [],
                 "Nothing to propose: the main product always evaluates its dummy baselines (majority, mean, median) "
                 "and they cannot be excluded."),
    "features.forbidden": ("feature.leakage_suspect", [],
                           "A flag for a person (cap L1) beside the host's leakage auditor; keeping a column out is an "
                           "exclusion, which only a person decides."),
    "features.to_confirm": ("feature.leakage_suspect", [],
                            "A question for a person: when is the column written relative to the prediction moment?"),
    "features.identifiers": ("column.is_identifier", [],
                             "An exclusion (cap L1): the host's identifier rule and a person decide."),
    "features.direction": (None, [],
                           "No decision point builds entity-history features: feature_transform_add takes only the "
                           "transforms of the host's feature registry (engine/features/contract.py)."),
    "data.cleaning": (None, [],
                      "Structural cleaning is the host's stage (run_structural_cleaning); these steps are notes for a "
                      "person, not answers to a point."),
    "model.families": ("training.families_budget", ["ExperimentPlan.families"],
                       "Cap L2 for a subset of the portfolio; the dummy baseline is always added by the host."),
    "threshold.policy": ("spec.objective", ["Objective.cost_false_positive", "Objective.cost_false_negative",
                                            "ProblemSpec.constraints.cost_matrix"],
                         "Only stated costs have a field (a cost matrix, binary only). A review budget has none: a "
                         "person picks an operating point on the host's operating points page."),
}

EXPERIMENT_TO_CHANGE = {
    "weights.balanced": {"kind": "class_weighting", "mode": "balanced", "weights": None},
}
NO_CHANGE_KIND = {
    "loss.median": "no change kind sets a loss; a hyperparameter_override would have to name a family and its own "
                   "parameter, which is the host's registry to say",
    "target.log": "no change kind transforms the target",
    "predictions.clip": "no change kind holds predictions to the training range",
    "features.entity_history": "no registered feature transform computes an entity's history",
    "features.by_segment": "needs a person to say what sets the segment apart",
    "classes.merge_tiny": "merging classes changes the target, which a change set may never do",
}


# ------------------------------------------------------------------------------------------------ the record


@functools.lru_cache(maxsize=1)
def _recorded_text() -> str:
    return FIELDS_FILE.read_text(encoding="utf-8")


def recorded() -> dict[str, Any]:
    """The field lists read from the main product, with the commit and the files read (a fresh copy)."""
    return json.loads(_recorded_text())


def fields_of(contract: str) -> list[str]:
    return list(recorded()["contracts"][contract]["fields"])


def _vocab() -> dict[str, Any]:
    return recorded()["vocabularies"]


def _out(contract: str, fields: dict[str, Any], notes: dict[str, str], *, status: str = "proposal",
         decision_point: Any = None, excluded: Optional[dict[str, str]] = None, about: str = "") -> dict[str, Any]:
    record = recorded()
    spec = record["contracts"][contract]
    out = {"contract": contract,
           "main_product": {"repository": record["repository"], "commit": record["commit"], "path": spec["path"],
                            "version": spec["version"]},
           "status": status, "decision_point": decision_point, "fields": fields,
           "notes": {k: v for k, v in notes.items() if v}, "excluded": excluded or {}, "about": about}
    return json.loads(json.dumps(out, allow_nan=False))  # plain JSON, and a NaN is a bug, not a value


def _as(cls: type[Versioned], value: Any) -> Any:
    if value is None or isinstance(value, cls):
        return value
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        return cls.read(value)
    raise TypeError(f"expected a {cls.__name__} or its JSON, not {type(value).__name__}")


def _size(value: Any) -> int:
    """Bytes of a value as the main product measures its JSON fields (json.dumps, UTF-8)."""
    return len(json.dumps(value, default=str).encode("utf-8"))


def _clip(text: Any, n: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _task_family(task: Optional[str]) -> Optional[str]:
    return "classification" if task in ("binary", "multiclass") else "regression" if task == "regression" else None


def _finding(findings: DataFindings, fid: str) -> Optional[Finding]:
    return next((f for f in findings.findings if f.id == fid), None)


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ------------------------------------------------------------------------------------------------ names


def metric_name(name: Any, task: Optional[str], kind: str = "primary") -> tuple[Optional[str], Optional[str]]:
    """(the main product's name for a metric, None) or (None, why it has none) for ``task``. ``kind`` is ``primary``
    (the metrics the main product may select by) or ``secondary`` (what its metric plan reports)."""
    vocab = _vocab()
    if task not in vocab["primary_metrics"]:
        return None, f"the task is {task!r}: the main product's metrics are per binary, multiclass or regression task"
    if name is None or not str(name).strip():
        return None, "no metric was named"
    allowed = set(vocab["primary_metrics"][task])
    if kind == "secondary":
        allowed |= set(vocab["secondary_metrics"][task])
    text = str(name).strip().lower()
    snake = re.sub(r"[\s\-]+", "_", text)
    core = CORE_METRIC_ALIASES.get(snake, snake)
    if task == "multiclass" and core == "f1":
        core = "macro_f1"
    for candidate in (vocab["metric_aliases"].get(text, text), vocab["metric_aliases"].get(snake, snake),
                      CORE_TO_MAIN_METRIC.get(core, core)):
        if candidate in allowed:
            return candidate, None
    if core in NO_MAIN_METRIC:
        return None, f"{core}: {NO_MAIN_METRIC[core]}"
    return None, f"{_clip(name, 80)!r} is not a {task} {kind} metric of the main product (engine/modeling/objective.py)"


def families(names: Sequence[str], task: Optional[str]) -> tuple[Optional[list[str]], list[str]]:
    """(the portfolio families, in the portfolio's order, or None when none maps) and notes for what changed.

    The list is the *recorded* portfolio: the host intersects it with its own ``open_ingest_families(task)`` (what is
    installed) before submitting, because a family outside that list gets the whole answer refused
    (``family_not_in_portfolio``)."""
    group = _task_family(task)
    if group is None:
        return None, [f"the task is {task!r}: no portfolio to map onto"]
    table, portfolio = FAMILY_TO_PORTFOLIO[group], _vocab()["portfolio_families"][group]
    wanted, notes, unmapped = set(), [], []
    for name in names:
        if name == "dummy":
            notes.append("dummy left out: the host's generator always adds its dummy baseline")
        elif name in table:
            wanted.update(table[name])
            if name in FAMILY_NOTES:
                notes.append(FAMILY_NOTES[name])
        else:
            unmapped.append(str(name))
    if unmapped:
        notes.append(f"not in the main product's registry, so not proposed: {', '.join(unmapped)} (they stay a "
                     "recommendation for a person; adding a family is the main product's own change)")
    chosen = [f for f in portfolio if f in wanted]
    if chosen:
        notes.append("the host keeps only the families it has installed (open_ingest_families) before submitting: one "
                     "outside its portfolio gets the whole answer refused")
    return (chosen or None), notes


# ------------------------------------------------------------------------------------------------ ProblemSpec


def _text_field(fact: Any, field: str, notes: dict[str, str], unknown: str) -> Optional[str]:
    limit = recorded()["contracts"]["ProblemSpec"]["max_length"].get(field)
    if fact is None or not fact.known or fact.value in (None, ""):
        notes[field] = unknown
        return None
    value = str(fact.value).strip()
    if limit is not None and len(value) > limit:
        notes[field] = f"longer than the main product's limit of {limit} characters: a person shortens it"
        return None
    if fact.status == "inferred":
        notes[field] = f"inferred, not stated ({fact.source}): {_clip(fact.reason, 200)}; a person confirms it"
    return value


def _task_of(brief: Optional[ProblemBrief], plan: Optional[SolutionPlan]) -> tuple[Optional[str], Optional[str]]:
    if brief is not None and brief.task.known and brief.task.value in ("binary", "multiclass", "regression"):
        return str(brief.task.value), ("inferred, not stated: " + _clip(brief.task.reason, 200)
                                       if brief.task.status == "inferred" else None)
    if plan is not None and plan.task in ("binary", "multiclass", "regression"):
        return plan.task, "from the data (the solution plan), not from the description: a person confirms it"
    return None, ("unknown: the description does not say whether the task is binary, multiclass or regression, and "
                  "there is no plan from the data; the wizard asks (it also accepts 'classification')")


def _primary(plan: Optional[SolutionPlan], brief: Optional[ProblemBrief], task: Optional[str]
             ) -> tuple[Optional[str], Optional[str], Optional[Decision]]:
    """(metric, note, the decision it came from)."""
    decision = plan.decision("metric.primary") if plan is not None else None
    if decision is not None:
        name, why = metric_name(decision.choice, task)
        return name, why, decision
    if brief is not None and brief.primary_metric.known:
        name, why = metric_name(brief.primary_metric.value, task)
        return name, why or ("named in the description: " + _clip(brief.primary_metric.quote or brief.primary_metric.reason, 200)), None
    return None, "no metric yet: the description names none and there is no plan; the host's metric planner chooses", None


def _costs(plan: Optional[SolutionPlan], brief: Optional[ProblemBrief], task: Optional[str]
           ) -> tuple[Optional[dict[str, float]], Optional[str]]:
    texts = []
    policy = plan.decision("threshold.policy") if plan is not None else None
    if policy is not None and isinstance(policy.choice, Mapping) and policy.choice.get("kind") == "expected_cost":
        texts.append(str(policy.choice.get("costs") or ""))
    texts += [c.text for c in (brief.constraints if brief is not None else []) if c.kind == "budget"]
    if not texts:
        return None, "no cost of a false alarm or of a miss is known"
    costs = next((c for c in (stated_costs(t) for t in texts) if c), None)
    if task != "binary":
        return None, f"the brief speaks of costs ({_clip(texts[0], 120)}), but a cost matrix is for binary tasks only"
    if costs is None:
        return None, (f"the brief speaks of costs ({_clip(texts[0], 120)}) without a number for both a false alarm and a "
                      "miss: a person gives them")
    if costs["false_positive"] == 0 and costs["false_negative"] == 0:
        return None, "both stated costs are zero, which the main product refuses"
    return costs, None


def problem_spec(brief: Any, plan: Any = None) -> dict[str, Any]:
    """The content of a ProblemSpec from a problem brief (and the plan's metric, when there is a plan).

    The spec is the person's intent: the host proposes it (``proposal:problem_spec``, cap L1) and a person locks it.
    ``status`` (draft or locked) is not content and is left to the host."""
    brief, plan = _as(ProblemBrief, brief), _as(SolutionPlan, plan)
    notes: dict[str, str] = {}
    task, note = _task_of(brief, plan)
    if note:
        notes["task_type"] = note
    target = _text_field(brief.target_column, "target_column", notes, "no column holds the label, or the description "
                         "does not name it")
    if target is None and brief.target_definition.known:
        notes["target_column"] = (f"no column holds the label: {_clip(brief.target_definition.value, 200)}; the host "
                                  "builds the label column (labels.join chooses the key) and target.column names it")
    elif target is None and brief.label_source.value == "none":
        notes["target_column"] = "the description says there are no labels: the main product's runs need a target"
    metric, why, decision = _primary(plan, brief, task)
    if why:
        notes["primary_metric"] = why
    constraints: dict[str, Any] = {}
    if metric is not None and decision is not None:
        constraints["primary_metric_reason"] = _clip(decision.reason, 500)
    costs, cost_note = _costs(plan, brief, task)
    if costs is not None:
        constraints["cost_matrix"] = costs
    stated = [{"kind": c.kind, "text": c.text, "status": c.status, **({"quote": c.quote} if c.quote else {}),
               **({"reason": c.reason} if c.reason else {})} for c in brief.constraints]
    if stated:
        constraints["brief_constraints"] = stated
    constraint_notes = []
    if costs is None and cost_note and any(c.kind == "budget" for c in brief.constraints):
        constraint_notes.append(cost_note)
    if any(c.kind == "review_capacity" for c in brief.constraints):
        constraint_notes.append("a review capacity has no structured key: it is kept in brief_constraints, and a person "
                                "turns it into an operating point")
    constraint_notes.append("brief_constraints is a free-form key (parse_objective keeps unknown keys as they are); "
                            "metric_constraints is left to a person: dclab_core does not turn text into metric bounds")
    notes["constraints"] = " ".join(constraint_notes)
    criteria = [{"text": s.value, "status": s.status, **({"quote": s.quote} if s.quote else {}),
                 **({"reason": s.reason} if s.reason else {})} for s in brief.success_criteria if s.known]
    if not criteria:
        notes["success_criteria"] = "none stated in the description"
    limit = recorded()["contracts"]["ProblemSpec"]["json_max_bytes"]
    if _size(constraints) > limit and "brief_constraints" in constraints:
        del constraints["brief_constraints"]
        notes["constraints"] += (f" brief_constraints left out: with them the constraints exceed the main product's "
                                 f"{limit} bytes; a person carries what matters")
    if _size(constraints) > limit:
        constraints = {}
        notes["constraints"] = f"left empty: they exceed the main product's {limit} bytes; a person writes them"
    if criteria and _size({"criteria": criteria}) > limit:
        criteria = []
        notes["success_criteria"] = f"left out: they exceed the main product's {limit} bytes; a person shortens them"
    fields = {
        "task_type": task,
        "business_objective": _text_field(brief.goal, "business_objective", notes, "the description states no goal: "
                                          "the wizard asks for one (the field is required)"),
        "target_column": target,
        "prediction_unit": _text_field(brief.unit, "prediction_unit", notes, "the description does not say what one "
                                       "scored row is"),
        "prediction_time_column": _text_field(brief.time_column, "prediction_time_column", notes,
                                              "no column says when each row is predicted"),
        "prediction_horizon": _text_field(brief.horizon, "prediction_horizon", notes, "the description gives no horizon"),
        "primary_metric": metric,
        "constraints": constraints,
        "success_criteria": {"criteria": criteria} if criteria else {},
    }
    return _out("ProblemSpec", fields, notes, decision_point="spec.objective")


# ------------------------------------------------------------------------------------------------ ProblemProfile


def _span_seconds(evidence: Mapping[str, Any]) -> Optional[float]:
    try:
        start, end = datetime.fromisoformat(str(evidence["start"])), datetime.fromisoformat(str(evidence["end"]))
    except (KeyError, ValueError):
        return None
    return float((end - start).total_seconds())


def problem_profile(findings: Any) -> dict[str, Any]:
    """Data findings in the shape of the main product's ProblemProfile: evidence beside the profile the host builds on
    its own locked training partition (``build_problem_profile``), never a replacement for it."""
    findings = _as(DataFindings, findings)
    task = findings.task if findings.task != "unknown" else None
    notes: dict[str, str] = {}
    if task is None:
        notes["task_type"] = "the findings do not say the task"
    if findings.target is None:
        notes["target"] = "the findings were computed without a target"
    if findings.training_rows is None:
        notes["row_count"] = "the findings do not say how many training rows they used"
    else:
        notes["row_count"] = "the training rows the findings' target statistics used (the host's locked training partition)"
    minority = imbalance = regression_target = None
    prevalence, classes, dist = (_finding(findings, i) for i in ("F-prevalence", "F-classes", "F-distribution"))
    excluded: dict[str, str] = {}
    all_rows = findings.training_rows is None or _finding(findings, "F-no-split") is not None
    if all_rows and findings.target is not None:
        why = ("computed on every row, not on a training partition: target statistics that may include holdout rows never "
               "cross (pass the host's locked training rows to analyse)")
        notes["minority_class_fraction"] = notes["imbalance_ratio"] = notes["regression_target"] = why
        excluded["all_rows_target_statistics"] = why
    elif task == "binary" and prevalence is not None and _number(prevalence.evidence.get("positive_rate")):
        rate = float(prevalence.evidence["positive_rate"])
        minority = min(rate, 1 - rate)
        imbalance = round((1 - minority) / minority, 4) if minority > 0 else None
        notes["minority_class_fraction"] = "from the share of positives on the training rows (F-prevalence)"
        notes["imbalance_ratio"] = "majority over minority share, from F-prevalence" if imbalance else "no minority rows"
    elif task == "multiclass" and classes is not None:
        minority = _number(classes.evidence.get("smallest_share"))
        imbalance = _number(classes.evidence.get("imbalance_ratio"))
        notes["minority_class_fraction"] = "the smallest class's share on the training rows (F-classes)"
        notes["imbalance_ratio"] = "largest over smallest class (F-classes)"
    else:
        reason = ("a regression target has no classes" if task == "regression" else
                  "the findings hold no class balance for this task")
        notes["minority_class_fraction"] = notes["imbalance_ratio"] = reason
    if all_rows and findings.target is not None:
        pass  # said above: nothing computed on every row crosses
    elif task == "regression" and dist is not None:
        regression_target = {key: (_number(dist.evidence.get(key)) if key in ("min", "max", "median") else None)
                             for key in recorded()["contracts"]["ProblemProfile"]["nested"]["regression_target"]}
        notes["regression_target"] = ("min, max and median from F-distribution (training rows); count, missing_count, "
                                      "mean and std are not in the findings")
    else:
        notes["regression_target"] = ("not a regression task" if task != "regression" else
                                      "the findings hold no target distribution")
    keys = _finding(findings, "F-keys")
    constant = _finding(findings, "F-constant")
    entities = [f for f in findings.findings if f.topic == "entities"]
    time = _finding(findings, "F-time")
    profile_nested = recorded()["contracts"]["ProblemProfile"]["nested"]
    repeated = []
    for f in entities:
        item = {key: None for key in profile_nested["repeated_entity_candidates[]"]}
        item.update(column=str(f.evidence.get("column") or ""), unique_count=f.evidence.get("entities"),
                    max_rows_per_entity=f.evidence.get("rows_per_entity_max"))
        repeated.append(item)
    notes["repeated_entity_candidates"] = (
        "column, unique_count and max_rows_per_entity from dclab_core's entity findings; the others are not in the findings "
        "(dclab_core keeps the median rows per entity, not the mean, and how well the entity tells the target out of "
        "fold, which has no field here)" if repeated else "no column repeats for an entity in the findings")
    times = []
    if time is not None and time.evidence.get("column"):
        item = {key: None for key in profile_nested["time_candidates[]"]}
        item.update(column=str(time.evidence["column"]), span_seconds=_span_seconds(time.evidence))
        times.append(item)
        notes["time_candidates"] = "column and span from F-time; the parse and order statistics are the host's"
    else:
        notes["time_candidates"] = "the findings name no time column"
    for field in ("numeric_columns", "categorical_columns", "boolean_columns", "free_text_columns",
                  "high_cardinality_columns"):
        notes[field] = "column roles are the host's (infer_column_roles); dclab_core's findings do not list them"
    notes["feature_count"] = "the modelled feature count is the host's (from its column roles)"
    notes["class_distribution"] = "the findings keep class shares and the tiny classes, not every class's count"
    notes["datetime_columns"] = "the host's column roles; dclab_core names its time column in time_candidates"
    notes["high_missing_columns"] = ("dclab_core reports missingness only where it speaks about the target, not every "
                                     "column above the host's 20% line")
    notes["geo_coordinate_candidates"] = "the findings do not look for coordinates"
    notes["identifier_columns"] = "columns unique per row, on all rows (F-keys)" if keys else "the findings hold no key check"
    notes["constant_columns"] = "dclab_core's constant-column check (F-constant)" if constant else (
        "dclab_core found no constant column")
    fields = {
        "task_type": task,
        "target": findings.target,
        "row_count": findings.training_rows,
        "feature_count": None,
        "class_distribution": None,
        "minority_class_fraction": minority,
        "imbalance_ratio": imbalance,
        "regression_target": regression_target,
        "numeric_columns": None,
        "categorical_columns": None,
        "boolean_columns": None,
        "datetime_columns": None,
        "identifier_columns": [str(c) for c in keys.evidence.get("unique_columns", [])] if keys else None,
        "free_text_columns": None,
        "high_cardinality_columns": None,
        "high_missing_columns": None,
        "constant_columns": [str(c) for c in constant.evidence.get("columns", [])] if constant else [],
        "repeated_entity_candidates": repeated,
        "time_candidates": times,
        "geo_coordinate_candidates": None,
        "version": recorded()["contracts"]["ProblemProfile"]["version"],
    }
    return _out("ProblemProfile", fields, notes, status="evidence", excluded=excluded)


# ------------------------------------------------------------------------------------------------ planners


def _split(plan: SolutionPlan) -> tuple[Optional[Decision], dict[str, Any]]:
    decision = plan.decision("split.strategy")
    if decision is None:
        return None, {}
    choice = decision.choice if isinstance(decision.choice, Mapping) else {"kind": decision.choice}
    return decision, dict(choice)


def _decision_evidence(decision: Decision) -> dict[str, Any]:
    return {"dclab_core_decision": decision.key, "cites": list(decision.cites), "source": decision.source,
            "status": decision.status,
            "alternatives": [{"choice": a.choice, "why_not": _clip(a.why_not, 300)} for a in decision.alternatives]}


def metric_plan(plan: Any) -> dict[str, Any]:
    """The plan's metric decisions as a MetricPlan (what the host's metric planner would hold)."""
    plan = _as(SolutionPlan, plan)
    notes: dict[str, str] = {}
    primary = plan.decision("metric.primary")
    secondary = plan.decision("metric.secondary")
    name, why = metric_name(primary.choice, plan.task) if primary is not None else (None, "the plan has no metric decision")
    if why:
        notes["primary_metric"] = why
    names, dropped = [], []
    for item in (secondary.choice if secondary is not None and isinstance(secondary.choice, list) else []):
        mapped, reason = metric_name(item, plan.task, "secondary")
        if mapped is not None and mapped != name and mapped not in names:
            names.append(mapped)
        elif mapped is None:
            dropped.append(f"{item} ({reason})")
    if dropped:
        notes["secondary_metrics"] = "left out, no main-product metric: " + "; ".join(dropped)
    if secondary is None:
        notes["secondary_metrics"] = "the plan has no secondary metrics"
    if primary is None:
        notes["reason"] = "the plan has no metric decision"
    fields = {"primary_metric": name, "secondary_metrics": names if secondary is not None else None,
              "reason": primary.reason if primary is not None else None,
              "version": recorded()["contracts"]["MetricPlan"]["version"]}
    return _out("MetricPlan", fields, notes, decision_point="spec.objective")


def holdout_plan(plan: Any) -> dict[str, Any]:
    """The plan's split decision as a HoldoutPlan: the holdout is locked first and used once."""
    plan = _as(SolutionPlan, plan)
    decision, choice = _split(plan)
    notes: dict[str, str] = {"test_size": "dclab_core does not choose the fraction: the host's default (0.2) and floor (0.2) hold",
                             "random_state": "the host's seed"}
    kind = choice.get("kind")
    strategy = SPLIT_TO_HOLDOUT.get(str(kind)) if kind is not None else None
    if decision is None:
        notes.update(strategy="the plan has no split decision", stratified="no split decision", reason="no split decision")
    elif strategy is None:
        notes["strategy"] = notes["stratified"] = f"{kind!r} is not a split the main product knows"
    elif strategy == "unsupported":
        notes["strategy"] = ("rows repeat for an entity and the target moves over time: the main product has no verified "
                             "combined holdout, and its own planner answers 'unsupported' here too; a person decides")
    group = choice.get("group_column") if strategy in ("group_disjoint", "unsupported") else None
    time = choice.get("time_column") if strategy in ("temporal_future", "unsupported") else None
    if group is None:
        notes["group_column"] = "the split does not group rows"
    if time is None:
        notes["time_column"] = "the split does not order rows in time"
    fields = {
        "strategy": strategy,
        "test_size": None,
        "random_state": None,
        "stratified": (plan.task in ("binary", "multiclass") if strategy == "unsupported" else strategy == "stratified_random")
                      if strategy is not None else None,
        "group_column": group,
        "time_column": time,
        "reason": decision.reason if decision is not None else None,
        "evidence": _decision_evidence(decision) if decision is not None else {},
        "plan_version": recorded()["contracts"]["HoldoutPlan"]["version"],
    }
    return _out("HoldoutPlan", fields, notes, decision_point="split.strategy")


def validation_plan(plan: Any) -> dict[str, Any]:
    """The plan's split decision as the folds of a ValidationPlan, on the locked training rows."""
    plan = _as(SolutionPlan, plan)
    decision, choice = _split(plan)
    classification = plan.task in ("binary", "multiclass")
    kind = choice.get("kind")
    table = {
        "stratified": ("StratifiedKFold", True, True),
        "random": ("KFold", True, False),
        "group": ("StratifiedGroupKFold", True, True) if classification else ("GroupKFold", False, False),
        "time": ("TimeSeriesSplit", False, False),
        "time_and_group": ("unsupported", False, classification),
    }
    strategy, shuffle, stratified = table.get(str(kind), (None, None, None))
    notes: dict[str, str] = {
        "requested_folds": "dclab_core does not choose the number of folds: the host's default (5) holds",
        "actual_folds": "the host's planner counts the folds the training rows can support",
        "random_state": "the host's seed",
        "fallback_reason": "the host's planner says why when it reduces folds or falls back",
    }
    if decision is None:
        notes.update(strategy="the plan has no split decision", shuffle="no split decision",
                     stratified="no split decision", reason="no split decision")
    elif strategy is None:
        notes.update(strategy=f"{kind!r} is not a split the main product knows", shuffle="unknown split",
                     stratified="unknown split")
    elif strategy == "StratifiedGroupKFold":
        notes["strategy"] = "the host falls back to GroupKFold when grouped class occupancy cannot fill the folds"
    elif strategy == "unsupported":
        notes["strategy"] = ("rows repeat for an entity and the target moves over time: the main product has no verified "
                             "joint splitter, and its own planner answers 'unsupported' here too")
    group = choice.get("group_column") if kind in ("group", "time_and_group") else None
    time = choice.get("time_column") if kind in ("time", "time_and_group") else None
    if group is None:
        notes["group_column"] = "the folds do not group rows"
    if time is None:
        notes["time_column"] = "the folds do not order rows in time"
    fields = {
        "strategy": strategy,
        "requested_folds": None,
        "actual_folds": None,
        "shuffle": shuffle,
        "random_state": None,
        "group_column": group,
        "time_column": time,
        "stratified": stratified,
        "reason": decision.reason if decision is not None else None,
        "fallback_reason": None,
        "evidence": _decision_evidence(decision) if decision is not None else {},
        "version": recorded()["contracts"]["ValidationPlan"]["version"],
    }
    return _out("ValidationPlan", fields, notes, decision_point="split.strategy")


def objective(plan: Any, brief: Any = None) -> dict[str, Any]:
    """The run objective: the primary metric with its reason and, for a binary task, the stated costs of errors."""
    plan, brief = _as(SolutionPlan, plan), _as(ProblemBrief, brief)
    notes: dict[str, str] = {}
    metric, why, decision = _primary(plan, brief, plan.task if plan.task != "unknown" else None)
    if why:
        notes["primary_metric"] = why
    if metric is None or decision is None:
        notes["primary_metric_reason"] = "no metric decision to give a reason for"
    costs, cost_note = _costs(plan, brief, plan.task)
    if costs is None:
        notes["cost_false_positive"] = notes["cost_false_negative"] = cost_note or "no costs"
    notes["constraints"] = ("metric constraints are the person's: dclab_core does not turn success criteria into metric "
                            "bounds" + (" (the brief states some: they travel as text in the ProblemSpec)"
                                        if brief is not None and brief.success_criteria else ""))
    fields = {"primary_metric": metric,
              "primary_metric_reason": _clip(decision.reason, 500) if metric is not None and decision is not None else None,
              "constraints": None,
              "cost_false_positive": costs["false_positive"] if costs else None,
              "cost_false_negative": costs["false_negative"] if costs else None,
              "version": recorded()["contracts"]["Objective"]["version"]}
    return _out("Objective", fields, notes, decision_point="spec.objective")


# ------------------------------------------------------------------------------------------------ ExperimentPlan


def experiment_plan(plan: Any, brief: Any = None, findings: Any = None) -> dict[str, Any]:
    """The plan as the payload of an ExperimentPlanProposal: one field group per decision point (``PLAN_FIELDS``).

    Target, metric and split are L1 answers (only a plan a person accepted may carry them); families and the time
    budget are L2 kinds. The host's run re-validates every answer against its own data at the stage that owns it."""
    plan, brief, findings = _as(SolutionPlan, plan), _as(ProblemBrief, brief), _as(DataFindings, findings)
    task = plan.task if plan.task != "unknown" else None
    notes: dict[str, str] = {}
    target = brief.target_column.value if brief is not None and brief.target_column.known else None
    if target is not None and brief.target_column.status == "inferred":
        notes["target_column"] = (f"inferred, not stated ({brief.target_column.source}): "
                                  f"{_clip(brief.target_column.reason, 200)}; a person confirms it (an L1 answer)")
    if target is None and findings is not None and findings.target:
        target = findings.target
        notes["target_column"] = ("the column dclab_core analysed as the target; when it was built from a label file, the "
                                  "host builds it the same way before the upload")
    elif target is None:
        notes["target_column"] = "no target column is known (target.column: the host's rule and a person choose)"
    if target is not None and len(str(target)) > 256:
        target, notes["target_column"] = None, "a column name longer than 256 characters"
    metric, why, _ = _primary(plan, brief, task)
    if why:
        notes["primary_metric"] = why
    decision, choice = _split(plan)
    strategy = SPLIT_TO_HOLDOUT.get(str(choice.get("kind"))) if decision is not None else None
    split = None
    if strategy in recorded()["vocabularies"]["plan_split_strategies"]:
        split = {"strategy": strategy,
                 "group_column": choice.get("group_column") if strategy == "group_disjoint" else None,
                 "time_column": choice.get("time_column") if strategy == "temporal_future" else None,
                 "test_size": None}
        notes["split"] = ("test_size is left to the host (its floor is 0.2); a split tighter than the host's rule stops "
                          "the run for a person before the holdout is locked")
    elif decision is None:
        notes["split"] = "the plan has no split decision"
    else:
        notes["split"] = (f"{choice.get('kind')!r} has no PlanSplit strategy (a time-and-entity split is not supported "
                          "by the main product): a person decides")
    fam = plan.decision("model.families")
    chosen, fam_notes = families(fam.choice if fam is not None and isinstance(fam.choice, list) else [], task)
    if fam is None:
        fam_notes = ["the plan has no families decision"]
    if fam_notes:
        notes["families"] = "; ".join(fam_notes)
    notes["max_training_seconds"] = "dclab_core sets no time budget: the host's rule (600 s) holds"
    notes["missing_values"] = ("dclab_core proposes no per-column imputation; columns kept out for leakage go to "
                               "feature.leakage_suspect, never through missing_values")
    fields = {"target_column": target, "primary_metric": metric, "split": split, "families": chosen,
              "max_training_seconds": None, "missing_values": None}
    points = {name: list(cols) for name, cols in recorded()["plan_fields"]["fields"].items()}
    return _out("ExperimentPlan", fields, notes, decision_point=points)


def decision_points(plan: Any) -> list[dict[str, Any]]:
    """Each decision of the plan with the main product's decision point it is proposed through (or why none), the
    fields it reaches, the point's cap, and the value translated to the main product's names where one exists."""
    plan = _as(SolutionPlan, plan)
    registry = recorded()["decision_points"]["points"]
    task = plan.task if plan.task != "unknown" else None
    out = []
    for d in plan.decisions:
        point, reaches, how = DECISIONS.get(d.key, (None, [], "not mapped yet: a new decision key needs its mapping"))
        proposal: Any = None
        note = how
        if d.key == "task.framing":
            proposal = task
        elif d.key == "split.strategy":
            proposal = SPLIT_TO_HOLDOUT.get(str((d.choice if isinstance(d.choice, Mapping) else {"kind": d.choice}).get("kind")))
        elif d.key == "metric.primary":
            proposal, why = metric_name(d.choice, task)
            note = how + (f" Not translatable: {why}." if why else "")
        elif d.key == "model.families":
            proposal, extra = families(d.choice if isinstance(d.choice, list) else [], task)
            note = how + (" " + "; ".join(extra) + "." if extra else "")
        elif d.key in ("features.forbidden", "features.to_confirm"):
            proposal = [str(x.get("column")) if isinstance(x, Mapping) else str(x) for x in (d.choice or [])]
        elif d.key == "features.identifiers":
            proposal = [str(x) for x in (d.choice or [])]
        elif d.key == "threshold.policy":
            costs = stated_costs(str(d.choice.get("costs") or "")) if isinstance(d.choice, Mapping) and \
                d.choice.get("kind") == "expected_cost" else None
            proposal = {"cost_matrix": costs} if costs else None
        out.append({"decision": d.key, "choice": d.choice, "reason": d.reason, "cites": list(d.cites),
                    "decision_point": point, "cap": registry[point]["cap"] if point in registry else None,
                    "reaches": reaches, "proposal": proposal, "status": "proposal", "note": note})
    return json.loads(json.dumps(out, allow_nan=False))


# ------------------------------------------------------------------------------------------------ after a run


def _threshold_kind(chosen_by: str) -> str:
    text = chosen_by.lower()
    if text.startswith(("the lowest expected cost", "no alerts")):
        return "cost"
    if text.startswith("the riskiest"):
        return "budget"
    if text.startswith("the best f1"):
        return "f1"
    return "other"


def decision_threshold(assessment: Mapping[str, Any]) -> dict[str, Any]:
    """An ``assess()`` result's threshold in the shape of the main product's decision threshold (binary only).

    Only what was chosen and measured on out-of-fold training predictions crosses; the holdout figures stay with the
    host's one final evaluation (``excluded``). The other checks (calibration, per-class figures, intervals) have no
    contract there: they belong in the technical report, listed in ``excluded`` by name."""
    names = recorded()["contracts"]["DecisionThreshold"]
    report_only = sorted(k for k in assessment if k not in ("task", "threshold"))
    excluded = {"holdout": "the holdout figures (threshold.holdout and every holdout metric) never cross: the host scores "
                           "its holdout once, at its own locked threshold"}
    if report_only:
        excluded["report_only"] = ("no main-product contract holds these; they belong in the technical report: "
                                   + ", ".join(report_only))
    if assessment.get("task") != "binary" or not isinstance(assessment.get("threshold"), Mapping):
        fields = {name: None for name in names["fields"]}
        fields["version"] = names["version"]
        why = "operating points and the decision threshold apply to binary classification only"
        return _out("DecisionThreshold", fields, {name: why for name in names["fields"] if name != "version"},
                    status="evidence", excluded=excluded)
    threshold = assessment["threshold"]
    chosen_by = str(threshold.get("chosen_by") or "")
    kind = _threshold_kind(chosen_by)
    training = threshold.get("training") or {}
    tp, fp, fn = (training.get(k) for k in ("true_positives", "false_positives", "false_negatives"))
    f1 = (2 * tp / (2 * tp + fp + fn)) if all(isinstance(v, int) for v in (tp, fp, fn)) and (2 * tp + fp + fn) else None
    at = {key: None for key in names["nested"]["oof_at_threshold"]}
    at.update(tp=tp, fp=fp, fn=fn, precision=training.get("precision"), recall=training.get("recall"),
              f1=round(f1, 4) if f1 is not None else None)
    notes = {
        "oof_row_count": "assess() does not record how many out-of-fold rows it saw",
        "threshold_status": "metric constraints are the host's; assess() checks none",
        "status": "metric constraints are the host's; assess() checks none",
        "expected_cost": "assess() does not record the expected cost",
        "constraints": "assess() evaluates no metric constraints",
        "oof_at_threshold": ("tp, fp, fn, precision and recall from the out-of-fold training rows; f1 from those counts; "
                             "tn, specificity, accuracy and balanced accuracy are not recorded by assess()"),
    }
    source = goal = None
    if kind == "cost":
        source, goal = "cost_matrix", "expected_cost"
    elif kind == "budget":
        notes["source"] = notes["goal_metric"] = (
            "a review budget has no threshold source in the main product: a person records it as a chosen operating "
            "point (its operating points page)")
    elif kind == "f1":
        goal = "f1"
        notes["source"] = ("the F1-best threshold without a stated goal: the main product keeps 0.5 when nothing is "
                           "requested, so a person asks for F1 (an f1 primary metric or constraint) or gives a goal")
    else:
        notes["source"] = notes["goal_metric"] = "the assessment does not say how the threshold was chosen"
    if threshold.get("chosen_on") and "out-of-fold" not in str(threshold.get("chosen_on")):
        notes["selected_on"] = "not chosen on out-of-fold predictions: it cannot be a decision threshold"
    if assessment.get("precision_note"):
        notes["value"] = _clip(assessment["precision_note"], 300)
    fields = {
        "version": names["version"],
        "value": _number(threshold.get("value")),
        "source": source,
        "selected_on": "out_of_fold_cv" if "selected_on" not in notes else None,
        "oof_row_count": None,
        "goal_metric": goal,
        "threshold_status": None,
        "status": None,
        "reason": chosen_by or None,
        "expected_cost": None,
        "constraints": None,
        "oof_at_threshold": at,
    }
    if fields["reason"] is None:
        notes["reason"] = "the assessment gives no reason"
    return _out("DecisionThreshold", fields, notes, status="proposal", excluded=excluded)


def _check_code(finding_id: str) -> str:
    code = re.sub(r"[^a-z0-9_]+", "_", finding_id.lower().removeprefix("f-")).strip("_")
    code = code if code[:1].isalpha() else f"w_{code}"
    return code[:64]


def experiment_review(review: Any) -> dict[str, Any]:
    """A review's measured weaknesses in the shape of the experiment critic's ExperimentReview (advice, never a state
    change). dclab_core judges no promotion and cites no stored CV metric, so those fields are the host's."""
    review = _as(Review, review)
    status = {"critical": "fail", "warning": "warning", "info": "warning"}
    findings = [{"check": _check_code(f.id), "status": status[f.severity], "note": _clip(f.statement, 500)}
                for f in review.weaknesses[:10]]
    summary = (f"{len(review.weaknesses)} weakness(es) measured on out-of-fold training predictions"
               + (": " + "; ".join(_clip(f.statement, 160) for f in review.weaknesses[:5]) if review.weaknesses else "")
               + (f". Next: {', '.join(e.name for e in review.next_experiments[:5])}." if review.next_experiments else "."))
    notes = {
        "verdict": "dclab_core proposes improvements; it does not judge promotion (that is the critic's and a person's)",
        "cv_metrics": "the critic cites the host's stored CV metrics of the reviewed run; dclab_core holds none",
        "confidence": "dclab_core's review is measured, not a model's judgement: no confidence to give",
        "findings": ("each weakness as a check (its id) with its statement; the critic's validator accepts only checks "
                     "recorded on the reviewed run, so the host records these first" if findings else "no weakness measured"),
        "summary": "numbers in it come from the weaknesses' evidence; the critic's validator requires them to be cited values",
    }
    if len(review.weaknesses) > 10:
        notes["findings"] += f"; {len(review.weaknesses) - 10} more left out (at most 10)"
    fields = {"verdict": None, "summary": _clip(summary, 2000), "cv_metrics": None, "findings": findings, "confidence": None}
    return _out("ExperimentReview", fields, notes, decision_point="experiment.review")


def experiment_changes(review: Any) -> dict[str, Any]:
    """The review's next experiments as an ExperimentChangeSet, at ``improve.next_action`` (cap L1).

    Only experiments with a change kind of the main product cross; the others are listed with why. A change set
    never touches the dataset, split, target or task (the main product's rule)."""
    review = _as(Review, review)
    changes, unmapped = [], []
    for experiment in review.next_experiments:
        change = EXPERIMENT_TO_CHANGE.get(experiment.name)
        if change is not None and change not in changes:
            changes.append(dict(change))
        elif change is None:
            unmapped.append(f"{experiment.name} ({NO_CHANGE_KIND.get(experiment.name, 'no change kind of the main product')})")
    notes = {"schema_version": "", "changes": "; ".join(
        (["no next experiment has a change kind of the main product"] if not changes else [])
        + (["not mapped: " + "; ".join(unmapped)] if unmapped else []))}
    fields = {"schema_version": 1, "changes": changes or None}
    return _out("ExperimentChangeSet", fields, notes, decision_point="improve.next_action",
                about=("improve.next_action is in the main product's registry, but its improve loop (ADR 0007) was "
                       "proposed, not built, at the commit read: until it is, a person starts a branch with these changes"))


# ------------------------------------------------------------------------------------------------ all at once


# 20.1: a problem card's family as the main product's task type, where it has one (its task types: binary, multiclass,
# regression, classification); the others are None with the reason
FAMILY_TO_TASK = {"classification": "classification", "regression": "regression", "rare_event": "binary"}
FAMILY_NOTES = {
    "anomaly": "no task type for an unlabelled search for unusual rows: its runs need a target; confirmed cases make it binary",
    "forecasting": "its runs have no forecasting task type: a horizon and rolling origins are not in its spec",
    "time_to_event": "its runs have no censored-duration task type",
    "ranking": "its runs have no ranking task type",
    "segmentation": "its runs need a target; groups have none",
    "uplift": "its runs have no treatment-effect task type",
    "entity_tables": "the outcome per entity maps to binary, multiclass or regression once its tables are joined into one",
    "streams": "its runs have no stream task type",
    "text_or_images": "its runs read tables: an encoder's vectors are columns of one, then binary, multiclass or regression",
}


def problem_card(card: Any) -> dict[str, Any]:
    """A problem card (20.1) as the main product's proposals: the ProblemSpec content (from the card's brief, the task
    type from the family when the brief does not say), the family at ``spec.objective``, the outcome at
    ``target.column``, and whether a spec may be locked from it (``lock``: only a ready card), with what blocks it."""
    from dclab_core.card import ProblemCard

    card = _as(ProblemCard, card)
    spec = problem_spec(card.brief)
    family = card.family.value
    if spec["fields"].get("task_type") is None and family:
        spec["fields"]["task_type"] = FAMILY_TO_TASK.get(family)
        spec["notes"]["task_type"] = (f"from the problem's family ({family})" if family in FAMILY_TO_TASK
                                      else FAMILY_NOTES.get(family, f"no task type for the family {family}"))
    points = {"spec.objective": {"proposal": {"family": family, "in_play": list(card.in_play), "labels": card.labels.value},
                                 "reason": card.family.reason or ("the words leave " + " or ".join(card.in_play) if card.in_play else "not settled")},
              "target.column": {"proposal": spec["fields"].get("target_column"),
                                "reason": card.outcome or "no outcome named yet"}}
    known = set(recorded()["decision_points"]["points"])
    return {"contract": "ProblemCard", "ProblemSpec": spec, "decision_points": {k: v for k, v in points.items() if k in known},
            "lock": card.ready, "blocking": list(card.blocking)}


def proposals(brief: Any = None, findings: Any = None, plan: Any = None) -> dict[str, Any]:
    """Every mapping the inputs allow, by contract name, and the decision points of the plan."""
    out: dict[str, Any] = {}
    if brief is not None:
        out["ProblemSpec"] = problem_spec(brief, plan)
    if findings is not None:
        out["ProblemProfile"] = problem_profile(findings)
    if plan is not None:
        out.update(MetricPlan=metric_plan(plan), HoldoutPlan=holdout_plan(plan), ValidationPlan=validation_plan(plan),
                   Objective=objective(plan, brief), ExperimentPlan=experiment_plan(plan, brief, findings),
                   decision_points=decision_points(plan))
    return out
