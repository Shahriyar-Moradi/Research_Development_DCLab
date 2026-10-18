"""The solution plan: decisions with reasons and alternatives (package 17.4).

``plan(brief, findings)`` turns what is known about the problem (17.2) and what the data shows (17.3) into a
:class:`~dclab_core.contracts.SolutionPlan`: one :class:`~dclab_core.contracts.Decision` per decision point, each with
its choice, its reason citing the findings it rests on (``F-...`` ids) and the alternatives set aside with why. The
host adopts the decisions through its own governed moves; a person can override any one with a reason
(:func:`override`), and :func:`validate` checks any plan (the rules', a model's or a person's) against the findings.

Decision points: ``task.framing``, ``labels.coverage``, ``labels.join``, ``split.strategy``, ``metric.primary``,
``metric.secondary``, ``baseline``, ``features.forbidden``, ``features.to_confirm``, ``features.identifiers``, ``features.direction``,
``data.cleaning``, ``model.families``, ``threshold.policy``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Optional, Sequence

from dclab_core.contracts import DataFindings, Decision, Finding, ProblemBrief, Risk, SolutionPlan

from dclab_core.modifiers import RARE_BELOW  # noqa: E402  (30.11: one cut-off for rare positives, the owner's 5%)

RARE = RARE_BELOW         # the rare-event families and recall at a budget join the plan below it (was 1%)
IMBALANCED = RARE_BELOW   # average precision becomes the primary metric below it (was 10%)

METRICS = {
    "binary": ("average_precision", "roc_auc", "recall_at_budget", "precision_at_budget", "f1", "log_loss", "brier"),
    "multiclass": ("macro_f1", "weighted_f1", "balanced_accuracy", "accuracy", "top_k_accuracy", "log_loss"),
    "regression": ("mae", "rmse", "median_absolute_error", "quantile_loss", "share_within_tolerance", "r2"),
}
SPLITS = ("stratified", "random", "group", "time", "time_and_group", "spatial_block")
FAMILIES = {
    "binary": ["logistic_regression", "random_forest", "gradient_boosting"],
    "multiclass": ["logistic_regression", "random_forest", "gradient_boosting"],
    "regression": ["ridge", "random_forest", "gradient_boosting"],
}


def _d(key: str, choice: Any, reason: str, cites: list[str] | None = None, alternatives: list[tuple[Any, str]] | None = None) -> Decision:
    return Decision(key=key, choice=choice, reason=reason, cites=list(dict.fromkeys(cites or []))[:20],
                    alternatives=[{"choice": c, "why_not": w} for c, w in (alternatives or [])][:8])


METRIC_ALIASES = {"auc": "roc_auc", "roc_auc": "roc_auc", "pr_auc": "average_precision", "ap": "average_precision",
                  "average_precision": "average_precision", "f1_score": "f1", "mean_absolute_error": "mae",
                  "root_mean_squared_error": "rmse"}
MIN_GROUPS = 10  # an entity with fewer values than this cannot make grouped folds and a grouped holdout


def _target_finding(findings: DataFindings) -> list[str]:
    return [f.id for f in findings.findings if f.id in ("F-prevalence", "F-classes", "F-distribution")][:1]


def _by_topic(findings: DataFindings, topic: str) -> list[Finding]:
    return [f for f in findings.findings if f.topic == topic]


def _get(findings: DataFindings, fid: str) -> Optional[Finding]:
    return next((f for f in findings.findings if f.id == fid), None)


def _task(brief: Optional[ProblemBrief], findings: DataFindings) -> str:
    if findings.task and findings.task != "unknown":
        return findings.task
    if brief is not None and brief.task.known and brief.task.value != "unknown":
        return str(brief.task.value)
    return "unknown"


# ------------------------------------------------------------------------------------------------ decisions


def _framing(task: str, brief: Optional[ProblemBrief], findings: DataFindings) -> Decision:
    cites, notes, alternatives = [], [], []
    tiny = _get(findings, "F-tiny-classes")
    cap = _get(findings, "F-cap")
    partial = _get(findings, "F-partial-labels")
    choice: dict[str, Any] = {"task": task}
    if tiny:
        cites.append(tiny.id)
        choice["tiny_classes"] = "keep, stratify and report separately; consider merging into 'other'"
        notes.append(f"{len(tiny.evidence.get('tiny_classes', []))} classes are too small to learn or measure well")
        alternatives.append(("merge the tiny classes into 'other' now", "a person should decide: merging changes what the model answers"))
    if cap:
        cites.append(cap.id)
        choice["cap"] = "predict below the cap; report rows at the cap separately (censored)"
        notes.append(f"{cap.evidence.get('share_at_cap', 0):.1%} of targets sit at a cap, so their true value is unknown")
        alternatives.append(("a classifier for 'at the cap' plus a regressor below it", "worth a test, not the default: two models to keep"))
    if partial:
        cites.append(partial.id)
        choice["labels"] = "partial"
        notes.append("the labels are a sample, so unlabelled rows are not known negatives")
        alternatives.append(("treat unlabelled rows as negatives", "they include frauds nobody found: precision would be biased and the model taught that they are clean"))
    reason = f"The problem is {task}" + (": " + "; ".join(notes) + "." if notes else ".")
    if brief is not None and brief.task.known:
        reason += f" The brief says so ({brief.task.status})."
    return _d("task.framing", choice, reason, cites, alternatives)


def _labels(findings: DataFindings) -> list[Decision]:
    out = []
    partial = _get(findings, "F-partial-labels")
    out.append(_d("labels.coverage", "partial" if partial else "complete",
                  "The brief says the labels are a sample: unlabelled is not negative." if partial else
                  "Nothing says the labels are a sample; every row with a label is treated as labelled.",
                  [partial.id] if partial else _target_finding(findings)))
    join = _get(findings, "F-labels-join")
    if join:
        key = join.evidence.get("key") or join.statement.split(" best on ", 1)[1].split(":", 1)[0]
        refused = [f for f in findings.findings if f.id.startswith("F-labels-inflation-")]
        out.append(_d("labels.join", {"key": key},
                      f"Joining on {key} finds the label rows one to one; the others would mark many rows per label.",
                      [join.id] + [f.id for f in refused],
                      [(f.evidence.get("key"), f"{f.evidence.get('rows_per_label')} rows per label: it would label every row "
                        "that shares the value") for f in refused[:6]]))
    return out


RECURRING = 0.9  # when this share of the latest rows belongs to entities seen earlier, a time-and-group split cannot test


def _split(task: str, brief: Optional[ProblemBrief], findings: DataFindings) -> Decision:
    named = {str(c) for c in (brief.entities.value or [])} if brief is not None and brief.entities.known else set()
    usable = [f for f in _by_topic(findings, "entities") if f.severity in ("warning", "critical")
              and int(f.evidence.get("entities", 0)) >= MIN_GROUPS]
    # the entity the person names first, then the one that tells the target most (out of fold)
    entities = sorted(usable, key=lambda f: (_entity_column(f) not in named, -float(f.evidence.get("target_from_entity_out_of_fold", 0))))
    group = entities[0] if entities else None
    others = [f for f in _by_topic(findings, "entities") if f is not group and f.severity in ("warning", "critical")]
    time = _get(findings, "F-time")
    drifting = time is not None and time.severity == "warning"
    cites = [f.id for f in ([group] if group else []) + ([time] if time else [])]
    column = _entity_column(group) if group else None
    kept_out = {c for f in _by_topic(findings, "leakage") for c in _leak_columns(f)} | {
        str(f.evidence.get("column")) for f in _by_topic(findings, "leakage") if f.evidence.get("column")}
    tcol = time.evidence.get("column") if time and time.evidence.get("column") not in kept_out else None
    drifting = drifting and tcol is not None
    no_split = _get(findings, "F-no-split")
    if no_split:
        cites.append(no_split.id)
    overlap = float(group.evidence.get("later_rows_seen_before", 0.0)) if group else 0.0
    places = _get(findings, "F-places")
    clustered = places is not None and places.severity == "warning"
    if clustered:
        cites.append(places.id)
    if clustered and not group and not tcol:  # 16.10: places that cluster are split by area rather than at random
        e = places.evidence
        choice = {"kind": "spatial_block", "lat_column": e.get("lat_column"), "lon_column": e.get("lon_column"), "block_km": e.get("block_km")}
        reason = (f"Nearby rows have alike targets (Moran's I {e.get('moran_i')}): test on whole blocks of about {e.get('block_km')} km the "
                  "model has not seen, not on rows whose neighbours it trained on.")
        if e.get("optimism") is not None:
            reason += f" On the training rows random folds score {e.get('random_folds')} against {e.get('block_folds')} for blocks."
        alternatives = [("stratified" if task in ("binary", "multiclass") else "random", "a row's neighbours on the other side: an optimistic score")]
        if tcol:
            alternatives.append(("time", f"rows also arrive over time ({tcol}), but neighbours on both sides are the larger risk here"))
    elif group and drifting and overlap >= RECURRING:
        choice = {"kind": "group", "group_column": column}
        reason = (f"Rows repeat for {column} and the target moves over time, but {overlap:.0%} of the latest rows belong to a {column} "
                  f"seen earlier: keeping both apart would leave almost nothing to test. Each {column} stays on one side; the "
                  "drift is a stated risk.")
        alternatives = [("time_and_group", f"{overlap:.0%} of later rows would be removed"), ("time", f"lets a {column} sit on both sides")]
    elif group and drifting:
        choice = {"kind": "time_and_group", "time_column": tcol, "group_column": column}
        reason = (f"Rows repeat for {column} and the target moves over time: train on earlier rows, test on later ones, "
                  f"with no {column} on both sides.")
        alternatives = [("group", "ignores that later rows differ"), ("time", f"lets a {column} sit on both sides"),
                        ("stratified", "both problems at once")]
    elif group:
        choice = {"kind": "group", "group_column": column}
        reason = f"Rows repeat for {column}: keep each {column} on one side, so the score is for new ones, not recognised ones."
        alternatives = [("stratified", f"a {column} on both sides lets the model recognise it: an optimistic score")]
        if tcol:
            alternatives.append(("time", f"no drift was seen, and grouping by {column} is the larger risk"))
    elif drifting:
        choice = {"kind": "time", "time_column": tcol}
        reason = "The target moves over time: validate on rows later than the training rows."
        alternatives = [("stratified", "a random split would hide what the model cannot know yet")]
    elif tcol:  # DCLAB-R02: rows that arrive over time are tested on later rows, drift measured or not
        choice = {"kind": "time", "time_column": tcol}
        reason = (f"Rows arrive over time ({tcol}): train on earlier rows and test on later ones, so the score is for the period the "
                  "model will meet. No drift was measured, but a test this size can miss it, and a random split would hide it.")
        alternatives = [("stratified" if task in ("binary", "multiclass") else "random", "mixes later rows into training")]
    else:
        choice = {"kind": "stratified" if task in ("binary", "multiclass") else "random"}
        reason = "No entity repeats strongly and no time orders the rows: a " + choice["kind"] + " split is fair."
        alternatives = []
    if clustered and choice.get("kind") != "spatial_block":  # time or an entity group came first: blocks are the stated risk
        reason += (f" Nearby rows also have alike targets ({places.id}): the score may flatter areas seen in training; blocks of about "
                   f"{places.evidence.get('block_km')} km were set aside for this split.")
        alternatives.append(("spatial_block", f"{choice['kind']} comes first here; nearby rows on both sides remain a risk"))
    choice["holdout"] = "locked first, used once, after the choice"
    for f in others[:3]:
        n = int(f.evidence.get("entities", 0))
        alternatives.append((f"group by {_entity_column(f)}",
                             f"only {n} values: too few for grouped folds" if n < MIN_GROUPS else "tells the target less, out of fold"))
    return _d("split.strategy", choice, reason, cites, alternatives)


def _entity_column(finding: Finding) -> str:
    column = finding.evidence.get("column")
    if column:
        return str(column)
    return finding.statement.split("Rows repeat for ", 1)[1].split(":", 1)[0] if "Rows repeat for " in finding.statement else ""


def _metrics(task: str, brief: Optional[ProblemBrief], findings: DataFindings) -> list[Decision]:
    capacity = next((c for c in (brief.constraints if brief else []) if c.kind == "review_capacity"), None)
    raw = str(brief.primary_metric.value).lower().replace(" ", "_").replace("-", "_") if brief is not None and brief.primary_metric.known else None
    named = METRIC_ALIASES.get(raw, raw) if raw else None
    if named == "f1" and task == "multiclass":
        named = "macro_f1"
    if task == "binary":
        prevalence = _get(findings, "F-prevalence")
        rate = float(prevalence.evidence.get("positive_rate", 0.5)) if prevalence else 0.5
        primary = "average_precision" if rate < IMBALANCED else "roc_auc"
        secondary = ["roc_auc" if primary == "average_precision" else "average_precision", "brier"]
        if capacity or rate < RARE:
            secondary.insert(0, "recall_at_budget")
        reason = (f"Positives are {rate:.2%} of rows: " + ("average precision ranks the rare cases; accuracy would reward "
                  "always saying no." if primary == "average_precision" else "ROC AUC ranks well at this balance."))
        alternatives = [("accuracy", "depends on the threshold and on the class balance"), ("f1", "measured at one threshold, not the ranking")]
        cites = [prevalence.id] if prevalence else []
    elif task == "multiclass":
        tiny = _get(findings, "F-tiny-classes")
        primary = "macro_f1"
        secondary = ["accuracy", "balanced_accuracy"] + (["top_k_accuracy"] if capacity else [])
        reason = "Every class counts in macro F1" + (", and some classes are tiny: accuracy would hide them." if tiny else ".")
        alternatives = [("accuracy", "the largest classes dominate it")]
        cites = [f.id for f in (tiny, _get(findings, "F-classes")) if f]
    elif task == "regression":
        dist = _get(findings, "F-distribution")
        cap = _get(findings, "F-cap")
        skew = float(dist.evidence.get("skew", 0)) if dist else 0.0
        primary = "mae"
        secondary = ["rmse", "median_absolute_error"] + (["quantile_loss"] if skew > 1 or cap else [])
        if brief is not None and brief.success_criteria:
            secondary.insert(0, "share_within_tolerance")
        reason = ("MAE reads in the target's unit and is not dominated by " + ("the cap and " if cap else "")
                  + ("the long tail" if skew > 1 else "a few large errors") + ".")
        alternatives = [("rmse", "squared errors chase the largest values" + (", here the capped ones" if cap else "")), ("r2", "not in the target's unit")]
        cites = [f.id for f in (dist, cap) if f]
    else:
        return []
    if named and named in METRICS.get(task, ()):
        if named != primary:
            alternatives.insert(0, (primary, f"the brief names {named}, so it leads; {primary} is reported beside it"))
            secondary = [primary] + [m for m in secondary if m != named]
        primary = named
        reason = f"The brief names {named}. " + reason
    elif raw:
        alternatives.insert(0, (raw, f"the brief names {raw}, which is not a {task} metric DCLab scores; ask what it means here"))
    return [_d("metric.primary", primary, reason, cites, alternatives),
            _d("metric.secondary", secondary, "Reported beside the primary metric, so one number does not hide a weakness.", cites)]


def _baseline(task: str, findings: DataFindings) -> Decision:
    choice = {"binary": ["majority class", "random ranking (average precision = prevalence)"],
              "multiclass": ["majority class", "class priors"],
              "regression": ["median of the training target", "mean per entity when entities repeat"]}.get(task, [])
    return _d("baseline", choice, "Every model is compared with what needs no model: a score only means something above these.",
              _target_finding(findings))


def _leak_columns(f: Finding) -> list[str]:
    """The columns a leakage finding says to keep out: the later of two times that recompute the target, both parts of
    a sum that adds up to it, or the one column."""
    if f.evidence.get("later"):
        return [str(f.evidence["later"])]
    columns = f.evidence.get("columns")
    if isinstance(columns, list) and columns:
        return [str(c) for c in columns]
    return [str(f.evidence["column"])] if f.evidence.get("column") else []


def _features(findings: DataFindings, brief: Optional[ProblemBrief], task: str) -> list[Decision]:
    out = []
    leaks = [f for f in _by_topic(findings, "leakage") if f.severity in ("warning", "critical")]
    forbidden = []
    for f in leaks:
        for column in _leak_columns(f):  # a recomputed target names two columns; the later one is written after the outcome
            if column and column not in [x["column"] for x in forbidden]:
                forbidden.append({"column": str(column), "reason": f.statement})
    out.append(_d("features.forbidden", forbidden,
                  ("These columns look written after the prediction moment (name, single-column power or recomputing the target): "
                   "keep them out unless a person confirms they are known at the moment.") if forbidden else
                  ("No column is accused of being written after the moment, but strong columns are held back until a person "
                   "confirms them (features.to_confirm)." if any(f.severity == "info" and "features.to_confirm" in f.affects
                                                                  for f in _by_topic(findings, "leakage") if f.id.startswith("F-confirm-"))
                   else "No column looks written after the moment by name, power or recomputation; the moment still decides."),
                  [f.id for f in leaks]))
    confirm = [{"column": str(f.evidence.get("column")), "question": f.implication} for f in _by_topic(findings, "leakage")
               if f.severity == "info" and f.evidence.get("column") not in [x["column"] for x in forbidden]]
    if confirm:
        out.append(_d("features.to_confirm", confirm,
                      "Keep these out until a person says when they are written: names that sound written after the outcome but "
                      "predict little, or strong columns on a task with several unrelated strong signals (it may simply be easy); "
                      "a wrong guess either way costs more than the question.",
                      [f.id for f in _by_topic(findings, "leakage") if f.severity == "info"]))
    keys = _get(findings, "F-keys")
    identifiers = list(keys.evidence.get("unique_columns", [])) if keys else []
    out.append(_d("features.identifiers", identifiers, "A column unique per row can only memorise: it identifies, it does not describe.",
                  [keys.id] if keys else []))
    directions, cites = [], []
    for f in _by_topic(findings, "entities"):
        if f.severity in ("warning", "critical"):
            column = _entity_column(f)
            directions.append(f"behaviour of each {column}: counts, repeats and shrunk rates over its other rows, fitted inside the folds")
            cites.append(f.id)
    time = _get(findings, "F-time")
    if time:
        directions.append("calendar parts and time since the entity's previous row, from the past only")
        cites.append(time.id)
    for f in _by_topic(findings, "missingness"):
        if f.severity == "warning":
            directions.append(f"a missing indicator for {f.evidence.get('column')}, after checking when it goes missing")
            cites.append(f.id)
    dist = _get(findings, "F-distribution")
    if task == "regression" and dist and float(dist.evidence.get("skew", 0)) > 1:
        directions.append("a log target, compared with the raw one")
        cites.append(dist.id)
    out.append(_d("features.direction", directions or ["the raw columns, then the feature ladder's recipes"],
                  "Where the findings point: behaviour over repeated entities, time, and missingness that speaks.", cites))
    return out


def _cleaning(findings: DataFindings) -> Optional[Decision]:
    steps, cites = [], []
    dup = _get(findings, "F-duplicates")
    if dup:
        steps.append(f"drop the {dup.evidence.get('duplicate_rows')} exact copies before the split")
        cites.append(dup.id)
    for f in _by_topic(findings, "quality"):
        steps.append(f"treat impossible values of {f.evidence.get('column')} as missing, inside the pipeline")
        cites.append(f.id)
    constant = _get(findings, "F-constant")
    if constant:
        steps.append("drop constant columns")
        cites.append(constant.id)
    return _d("data.cleaning", steps, "What the data needs before any model sees it.", cites) if steps else None


def _families(task: str, findings: DataFindings) -> Decision:
    families = list(FAMILIES.get(task, []))
    cites, why = [], "linear and tree families on identical folds"
    partial = _get(findings, "F-partial-labels")
    prevalence = _get(findings, "F-prevalence")
    rare = prevalence is not None and float(prevalence.evidence.get("positive_rate", 1)) < RARE
    if task == "binary" and (partial or rare):
        families += ["pu_bagging", "isolation_forest", "ecod", "percentile_rules"]
        cites += [f.id for f in (partial, prevalence) if f]
        why += "; with rare or partial labels, positive-unlabelled learning, anomaly detectors and rules join the screen"
    cap = _get(findings, "F-cap")
    if task == "regression" and cap:
        families.append("quantile_gradient_boosting")
        cites.append(cap.id)
        why += "; with a capped target, a quantile model is worth comparing"
    families.insert(0, "dummy")
    cites = cites or _target_finding(findings)
    return _d("model.families", families, f"Screen {why}, with a dummy baseline always in.", cites)


def _threshold(task: str, brief: Optional[ProblemBrief], findings: DataFindings) -> Optional[Decision]:
    if task != "binary":
        return None
    cites = _target_finding(findings)
    capacity = next((c for c in (brief.constraints if brief else []) if c.kind == "review_capacity"), None)
    budget = next((c for c in (brief.constraints if brief else []) if c.kind == "budget"), None)
    if capacity:
        return _d("threshold.policy", {"kind": "review_budget", "capacity": capacity.text},
                  f"The team can review {capacity.text}: alert on that many of the riskiest, chosen on out-of-fold scores.", cites,
                  alternatives=[("0.5", "a probability threshold ignores how many cases the team can review")])
    if budget:
        return _d("threshold.policy", {"kind": "expected_cost", "costs": budget.text},
                  "The brief gives the cost of errors: choose the threshold that minimises expected cost on out-of-fold scores.", cites,
                  alternatives=[("0.5", "ignores the costs the brief states")])
    return _d("threshold.policy", {"kind": "ask", "default": "report the ranking; choose the threshold with the person"},
              "No capacity or cost is known yet: the ranking is reported and the threshold is chosen with the person.", cites)


def _risks(findings: DataFindings, brief: Optional[ProblemBrief]) -> list[Risk]:
    risks = []
    for f in findings.findings:
        if f.severity == "critical":
            risks.append(Risk(statement=f.statement, check=f.implication, cites=[f.id]))
    no_split = _get(findings, "F-no-split")
    if no_split:
        risks.append(Risk(statement=no_split.statement, check=no_split.implication, cites=[no_split.id]))
    disagree = _get(findings, "F-labels-disagree")
    if disagree:
        risks.append(Risk(statement=disagree.statement, check="Ask what the column means in the label file; state what the "
                          "table cannot show as a limit.", cites=[disagree.id]))
    time = _get(findings, "F-time")
    if time is not None and time.severity == "warning":
        risks.append(Risk(statement="The data changes over time.", check="Report a forward-in-time check beside the main split.",
                          cites=[time.id]))
    return risks[:30]


def _would_change(brief: Optional[ProblemBrief]) -> list[str]:
    out = []
    if brief is None or not brief.prediction_moment.known:
        out.append("The prediction moment: an earlier moment turns more columns into leaks; it is assumed, not stated.")
    if brief is None or brief.label_coverage.value != "partial":
        out.append("If the labels turn out to be a sample, unlabelled rows are no longer negatives.")
    out.append("A cost of errors or a review capacity would set the threshold and may change the primary metric.")
    return out


def plan(brief: Optional[ProblemBrief], findings: DataFindings) -> SolutionPlan:
    """Decide what the findings imply. Every decision cites the findings it rests on."""
    task = _task(brief, findings)
    decisions = [_framing(task, brief, findings), *_labels(findings), _split(task, brief, findings),
                 *_metrics(task, brief, findings), _baseline(task, findings), *_features(findings, brief, task)]
    for extra in (_cleaning(findings), _families(task, findings), _threshold(task, brief, findings)):
        if extra is not None:
            decisions.append(extra)
    return SolutionPlan(task=task if task in ("binary", "multiclass", "regression") else "unknown",
                        decisions=decisions, risks=_risks(findings, brief), would_change=_would_change(brief))


# ------------------------------------------------------------------------------------------------ check and override


def _persons(decision: Optional[Decision]) -> bool:
    """A person's override is the person's call; a model's override is checked like any proposal."""
    return decision is not None and decision.status == "overridden" and decision.source == "person"


def _split_kind(choice: Any) -> tuple[Optional[str], Optional[str]]:
    """(kind, problem): a model may send "stratified" for {"kind": "stratified"}; anything else must be a mapping."""
    if isinstance(choice, str):
        return choice, None
    if isinstance(choice, dict):
        return choice.get("kind"), None
    return None, f"split.strategy must name a kind, not {type(choice).__name__}"


def _columns_of(choice: Any) -> tuple[set[str], Optional[str]]:
    """The columns of a features.forbidden choice: entries may be {"column": ...} or plain names."""
    if not isinstance(choice, list):
        return set(), "features.forbidden must be a list of columns"
    out = set()
    for item in choice:
        if isinstance(item, str):
            out.add(item)
        elif isinstance(item, dict) and item.get("column"):
            out.add(str(item["column"]))
        else:
            return out, "features.forbidden holds an entry that names no column"
    return out, None


def validate(plan_: SolutionPlan, findings: DataFindings) -> list[str]:
    """Problems with a plan (the rules', a model's or a person's), against the findings. Empty means valid. Only a
    person's override may set a finding aside; a model's override is checked like any proposal."""
    problems = []
    ids = {f.id for f in findings.findings}
    for d in plan_.decisions:
        unknown = [c for c in d.cites if c.startswith("F-") and c not in ids]
        if unknown:
            problems.append(f"{d.key} cites findings that do not exist: {', '.join(unknown)}")
    metric = plan_.decision("metric.primary")
    if metric is not None and plan_.task in METRICS and metric.choice not in METRICS[plan_.task]:
        problems.append(f"metric.primary {metric.choice!r} is not a {plan_.task} metric")
    split = plan_.decision("split.strategy")
    if split is not None:
        kind, shape = _split_kind(split.choice)
        if shape:
            problems.append(shape)
        elif kind not in SPLITS:
            problems.append(f"split.strategy {kind!r} is not a known split")
        places = next((f for f in findings.findings if f.id == "F-places" and f.severity == "warning"), None)
        if places is not None and kind in ("stratified", "random") and not _persons(split):
            problems.append(f"split.strategy is {kind} although nearby rows have alike targets ({places.id}): split by blocks of places, "
                            "or a person must say why not")
        strong = [f for f in findings.findings if f.topic == "entities" and f.severity in ("warning", "critical")
                  and int(f.evidence.get("entities", 0)) >= MIN_GROUPS]
        if strong and kind in ("stratified", "random") and not _persons(split):
            problems.append(f"split.strategy is {kind} although rows repeat for an entity ({strong[0].id}); group the folds or "
                            "let a person override with a reason")
        if kind == "time_and_group" and isinstance(split.choice, Mapping) and not _persons(split):
            # 28.6 (the debate's live run): a grouped forward split on an entity that recurs over time leaves almost nothing to
            # test, and the engine refuses it; the rules group without the time order there, and only a person may decide otherwise
            column = split.choice.get("group_column")
            recurs = [f for f in findings.findings if f.topic == "entities" and f.evidence.get("column") == column
                      and float(f.evidence.get("later_rows_seen_before", 0.0) or 0.0) >= RECURRING]
            if recurs:
                problems.append(f"split.strategy groups by {column} and orders by time, but "
                                f"{float(recurs[0].evidence['later_rows_seen_before']):.0%} of the latest rows belong to a {column} seen earlier "
                                f"({recurs[0].id}): almost nothing would be left to test")
    forbidden = plan_.decision("features.forbidden")
    kept_out, shape = _columns_of(forbidden.choice) if forbidden is not None else (set(), None)
    if shape:
        problems.append(shape)
    for f in findings.findings:
        if f.topic == "leakage" and f.severity == "critical" and not _persons(forbidden):
            missing = [c for c in _leak_columns(f) if c not in kept_out]  # every part of a sum, the later of two times
            if missing:
                problems.append(f"{f.id} is critical but {', '.join(missing)} is not in features.forbidden")
    confirm = plan_.decision("features.to_confirm")
    held, _ = _columns_of(confirm.choice) if confirm is not None else (set(), None)
    if not (_persons(forbidden) or (confirm is not None and _persons(confirm))):
        for f in findings.findings:  # a column held for a person's confirmation stays held until a person decides
            if f.topic == "leakage" and "features.to_confirm" in f.affects and f.evidence.get("column"):
                column = str(f.evidence["column"])
                if column not in kept_out and column not in held:
                    problems.append(f"{f.id}: {column} waits for a person's confirmation but is in neither features.forbidden "
                                    "nor features.to_confirm")
    join = plan_.decision("labels.join")
    if join is not None and not _persons(join):
        key = join.choice.get("key") if isinstance(join.choice, dict) else join.choice
        if "F-labels-no-key" in ids:
            problems.append("labels.join names a key, but no shared column identifies a case (F-labels-no-key)")
        inflating = {str(f.evidence.get("key")) for f in findings.findings if f.id.startswith("F-labels-inflation-")}
        if str(key) in inflating:
            problems.append(f"labels.join on {key} inflates the labels (it marks many rows per label)")
    return problems


def override(plan_: SolutionPlan, key: str, choice: Any, reason: str, by: str = "person", cites: Optional[list[str]] = None) -> SolutionPlan:
    """Replace one decision; the old choice is kept as an alternative with the reason. ``by="advisor"`` marks a model's
    change, which :func:`validate` still checks; anyone else is a person, whose override stands."""
    old = plan_.decision(key)
    alternatives = ([{"choice": old.choice, "why_not": f"overridden by {by}: {reason}"}] if old is not None else [])
    if old is not None:
        alternatives += [a.model_dump() for a in old.alternatives]
    new = Decision(key=key, choice=choice, reason=reason, cites=(cites or (old.cites if old is not None else []))[:20],
                   alternatives=alternatives[:8], source="advisor" if by == "advisor" else "person", status="overridden")
    decisions = [new if d.key == key else d for d in plan_.decisions]
    if old is None:
        decisions.append(new)
    return SolutionPlan.model_validate({**plan_.model_dump(), "decisions": [d.model_dump() for d in decisions]})


# ------------------------------------------------------------------------------------------------ the solution planner (a role)

PLANNABLE = ("task.framing", "labels.coverage", "split.strategy", "metric.primary", "metric.secondary", "baseline",
             "features.forbidden", "features.to_confirm", "features.identifiers", "features.direction", "data.cleaning",
             "model.families", "threshold.policy")


def _compact(value: Any, n: int = 300) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:n]


def _plain_choice(choice: Any) -> Any:
    """A rule choice for the planner's evidence: a column list as its names (never cut mid-entry), else compact JSON."""
    if isinstance(choice, list) and choice and all(isinstance(c, Mapping) and "column" in c for c in choice):
        return [str(c["column"])[:100] for c in choice][:40]
    return _compact(choice, 300)


def _columns(choice: Any) -> set[str]:
    return {str(c.get("column")) if isinstance(c, Mapping) else str(c) for c in (choice or []) if c} if isinstance(choice, list) else set()


ADDITIONS_ONLY = ("features.forbidden", "features.to_confirm", "features.identifiers")
THRESHOLD_KINDS = ("review_budget", "expected_cost", "ask")


def _known_families() -> set[str]:
    return {"dummy", "pu_bagging", "isolation_forest", "ecod", "percentile_rules", "quantile_gradient_boosting",
            "extra_trees", "lightgbm", "xgboost", "hist_gradient_boosting"} | {f for fams in FAMILIES.values() for f in fams}


def _guard(key: str, choice: Any, current: SolutionPlan, findings: DataFindings, columns: Optional[Sequence[str]] = None) -> Optional[str]:
    """What an advisor may not do to a decision, whatever the validator checks: undo a safety decision (a column kept
    out, an identifier, the grouping a strong entity calls for), change the task, write a shape the host cannot run, or
    (``columns``: the table's, when the host gives them) name a column the table does not have."""
    old = current.decision(key)
    if columns is not None:
        named = set(_columns(choice)) if key in ADDITIONS_ONLY and isinstance(choice, list) else \
            {str(choice.get(f)) for f in ("group_column", "time_column") if isinstance(choice, Mapping) and choice.get(f)} if key == "split.strategy" else set()
        unknown = sorted(named - {str(c) for c in columns})
        if unknown:  # 28.6 (the debate's live run): a derived feature named as a column refused the solution built from the plan
            return f"names a column the table does not have: {', '.join(unknown[:5])}"
    if key in ADDITIONS_ONLY:
        if not isinstance(choice, list):
            return "a column list is a list"
        # each entry shaped as the rules write it (the review of 28.6: a model's names or partial objects passed the checks
        # and broke the solution built from the plan): identifiers are names; a column kept out is {column, reason}; a column
        # to confirm is {column, question}
        shape = {"features.identifiers": (str, ()), "features.forbidden": (Mapping, ("column", "reason")),
                 "features.to_confirm": (Mapping, ("column", "question"))}[key]
        bad = [c for c in choice if not isinstance(c, shape[0]) or (shape[1] and not all(isinstance(c.get(f), str) and c.get(f).strip() for f in shape[1]))]
        if bad:
            return (f"each entry of {key} is " + ("a column's name" if shape[0] is str else "{" + ", ".join(shape[1]) + "}")
                    + f" (not {str(bad[0])[:80]})")
        lost = _columns(old.choice if old else []) - _columns(choice)
        if lost:
            return f"an advisor may add columns here, never remove them (would drop {', '.join(sorted(lost)[:5])})"
    elif key == "split.strategy":
        kind, problem = _split_kind(choice)
        if problem or kind not in SPLITS:
            return problem or f"{kind!r} is not a split"
        old_choice = old.choice if old is not None and isinstance(old.choice, Mapping) else {}
        if old_choice.get("kind") == "spatial_block" and kind in ("stratified", "random"):
            return "the findings call for folds of whole blocks of places (F-places): an advisor may not replace them by a random split"
        if old_choice.get("group_column"):
            if kind not in ("group", "time_and_group") or not isinstance(choice, Mapping) or choice.get("group_column") != old_choice["group_column"]:
                return f"the findings call for folds grouped by {old_choice['group_column']}: an advisor may not drop it"
        time = _get(findings, "F-time")
        if isinstance(choice, Mapping) and choice.get("time_column") and (time is None or choice["time_column"] != time.evidence.get("column")):
            return "the time column must be the one the data has (F-time)"
    elif key == "task.framing":
        if not isinstance(choice, Mapping) or choice.get("task") != current.task:
            return f"the task is {current.task}: an advisor may refine its framing, not change it"
    elif key == "model.families":
        if not isinstance(choice, list) or "dummy" not in choice or not set(map(str, choice)) <= _known_families():
            return "families must be known names, with the dummy baseline kept"
    elif key == "labels.coverage":
        if choice not in ("partial", "complete"):
            return "coverage is partial or complete"
        if choice == "complete" and _get(findings, "F-partial-labels") is not None:
            return "the findings say the labels are partial (F-partial-labels)"
    elif key == "threshold.policy":
        if not isinstance(choice, Mapping) or choice.get("kind") not in THRESHOLD_KINDS:
            return f"a threshold policy is an object whose kind is one of {', '.join(THRESHOLD_KINDS)}"
    elif key == "metric.secondary":
        if not isinstance(choice, list) or not all(isinstance(m, str) and len(m) <= 60 for m in choice):
            return "secondary metrics are a list of names"
    return None


def _planner_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    brief, findings, base = inputs["brief"], inputs["findings"], inputs["plan"]
    known = {}
    if brief is not None:
        for name in ("goal", "action", "unit", "task", "target_column", "label_source", "label_coverage", "prediction_moment",
                     "horizon", "entities", "time_column", "primary_metric"):
            fact = getattr(brief, name)
            if fact.known:
                known[name] = _compact(fact.value, 200)
        known["constraints"] = [_compact(f"{c.kind}: {c.text}", 200) for c in brief.constraints][:8]
    return {"brief": known, "task": base.task,
            "findings": [_compact(f"{f.id} [{f.severity}, {f.topic}] {f.statement}", 300) for f in findings.findings][:50],
            "rule_plan": {d.key: _plain_choice(d.choice) for d in base.decisions},
            "decision_keys": list(PLANNABLE)}


def _planner_parse(answer: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Each proposal is an advisor's override of one decision, kept only if it cites measured findings and the plan
    validator finds no new problem with it; every refusal is recorded with its reason."""
    proposals = answer.get("decisions")
    if not isinstance(proposals, list):
        raise ValueError("decisions must be a list")
    findings: DataFindings = inputs["findings"]
    ids = {f.id for f in findings.findings}
    current: SolutionPlan = inputs["plan"]
    before = set(validate(current, findings))
    kept, refused = [], []
    for index, item in enumerate(proposals):
        if index >= len(PLANNABLE) or not isinstance(item, Mapping):
            refused.append({"key": str(item.get("key") if isinstance(item, Mapping) else item)[:80], "choice": None,
                            "why": "more proposals than decisions" if index >= len(PLANNABLE) else "not a proposal object"})
            continue
        key, choice, reason, cites = item.get("key"), item.get("choice"), str(item.get("reason") or ""), item.get("cites") or []
        why = None
        if key not in PLANNABLE:
            why = f"{key!r} is not a decision the planner may propose"
        elif not isinstance(cites, list) or not cites or not set(map(str, cites)) <= ids:
            why = "it must cite findings that were measured"
        elif len(reason.strip()) < 10:
            why = "it needs a reason"
        elif current.decision(key) is not None and current.decision(key).choice == choice:
            continue  # the rules already chose this: nothing to change
        if why is None:
            why = _guard(key, choice, current, findings, inputs.get("columns"))
        if why is None:
            try:
                candidate = override(current, key, choice, reason[:MAX_REASON], by="advisor", cites=list(map(str, cites)))
            except Exception as exc:  # noqa: BLE001 - a choice the contract cannot hold
                why = f"not a valid decision: {type(exc).__name__}"
            else:
                new = sorted(set(validate(candidate, findings)) - before)
                if new:
                    why = "the plan validator refused it: " + "; ".join(new)[:400]
                else:
                    current, before = candidate, set(validate(candidate, findings))
                    kept.append({"key": key, "choice": choice, "reason": reason[:400], "cites": list(map(str, cites))[:10]})
                    continue
        refused.append({"key": str(key)[:80], "choice": _compact(choice, 200), "why": why})
    return {"plan": current, "kept": kept, "refused": refused}


MAX_REASON = 3900


def _planner_role():
    from dclab_core.agents import Role

    return Role(name="solution_planner",
                description="Proposes changes to the rule plan's decisions from the brief and the findings, each citing them; the plan "
                            "validator refuses an invalid one, and every refusal is recorded.",
                evidence=_planner_evidence, parse=_planner_parse,
                fallback=lambda inputs: {"plan": inputs["plan"], "kept": [], "refused": []})


SOLUTION_PLANNER = _planner_role()


def plan_with(advisor: Any, brief: Optional[ProblemBrief], findings: DataFindings, columns: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """The rule plan, then the solution planner's checked changes (none without an advisor). ``columns``: the table's,
    when the host has them: a change naming another column is refused."""
    from dclab_core.agents import run_role

    base = plan(brief, findings)
    result = run_role(SOLUTION_PLANNER, advisor, {"brief": brief, "findings": findings, "plan": base, "columns": columns})
    return {**result.output, "rule_plan": base, "source": result.source, "reason": result.reason}
