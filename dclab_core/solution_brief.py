"""The Solution Brief and G3 (package 30.14, the owner's framework, phase 3): what will be built, how it will be judged,
and the bar it must meet, written and signed before anything is trained.

The framework asks eight questions before a model is built: is machine learning needed at all (the approach level, the
first rung of the ladder that can meet the bar); which recipe (the kind and its modifiers); which baselines to beat
(today's process, and the recipe's mandatory simple model); how it is validated (the split, the holdout read once); what
the success bar is (the primary metric's target, against a named baseline when relative, with guard metrics that must not
get worse); the decision rule (a threshold, the top K a period, a ranking); the risks, each with the check that catches
it; and at least one alternative with why it lost.

``fill`` writes a brief by rules from what the earlier cards hold; a person may change its words and its bar. G3
(``problems``) refuses a brief without a success bar, without a baseline or without an alternative. A person signs the
brief's ``fingerprint``: the bar is frozen at that signature, and per the owner's decision D3 the signature covers the one
holdout read. ``judge`` reads the holdout once against the frozen bar: met, not met, or cannot be judged, with why. A bar
that is not met is never retuned on the holdout: the work goes back to the brief, with a new holdout.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal, Mapping, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, MAX_TEXT, Alternative, Contract, Risk, Versioned, _version

LADDER = ("rule", "template", "classical_ml", "small_model", "frontier_llm")  # the first rung that can meet the bar
Op = Literal[">=", ">", "<=", "<"]


class Baseline(Contract):
    """A baseline to beat: today's process (the person's words) or the recipe's simple model (computed at the holdout)."""

    name: str = Field(min_length=1, max_length=80)
    kind: Literal["today", "simple_model"]
    description: str = Field(default="", max_length=MAX_TEXT)
    value: Optional[float] = Field(default=None, description="Today's number on the bar's metric, when a person knows it: a relative bar against today needs it.")


class Guard(Contract):
    """A metric that must not get worse than a baseline's (or than a value)."""

    metric: str = Field(min_length=1, max_length=60)
    op: Op = ">="
    value: Optional[float] = None
    relative_to: Optional[str] = Field(default=None, max_length=80, description="A baseline's name: no worse than it.")


class SuccessBar(Contract):
    """The primary metric's target: an absolute value, or relative to a named baseline (a factor times it, or a margin
    beyond it); guards that must hold beside it; the business number it stands for."""

    metric: str = Field(min_length=1, max_length=60)
    op: Op = ">"
    value: Optional[float] = Field(default=None, description="An absolute target.")
    relative_to: Optional[str] = Field(default=None, max_length=80, description="The baseline it is measured against.")
    factor: Optional[float] = Field(default=None, gt=0, description="Relative: the target is the baseline's value times this.")
    margin: Optional[float] = Field(default=None, description="Relative: the target is the baseline's value plus this (minus for a lower-is-better metric).")
    guards: list[Guard] = Field(default_factory=list, max_length=8)
    business: Optional[str] = Field(default=None, max_length=MAX_TEXT)


# which way a metric is better, for the metrics the engines report (a metric not here is not checked): a bar or a guard
# whose comparison goes the other way is refused by G3 (an mae "above" the baseline was judged met)
HIGHER = {"accuracy": True, "balanced_accuracy": True, "roc_auc": True, "average_precision": True, "macro_f1": True, "f1": True,
          "precision": True, "recall": True, "recall_at_budget": True, "precision_at_k": True, "r2": True, "ndcg_at_k": True,
          "qini_coefficient": True, "silhouette": True, "incident_recall": True, "planted_recall_at_budget": True,
          "mae": False, "rmse": False, "mape": False, "mase": False, "integrated_brier": False, "log_loss": False, "brier": False}


def _direction_problem(what: str, metric: str, op: str) -> Optional[str]:
    higher = HIGHER.get(metric)
    if higher is None or (op in (">", ">=")) == higher:
        return None
    return f"{what} on {metric} says {op}, but {'higher' if higher else 'lower'} is better for {metric}"


class Approval(Contract):
    by: str = Field(min_length=1, max_length=120)
    at: str = Field(min_length=10, max_length=40)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{16}$")


class SolutionBrief(Versioned):
    """What will be built, how it is judged and the bar it must meet; signed before training (G3)."""

    schema_version: Literal["dclab_core.solution_brief.v1"] = _version("solution_brief")
    approach_level: Literal["rule", "template", "classical_ml", "small_model", "frontier_llm"]
    approach_why: str = Field(min_length=3, max_length=MAX_TEXT, description="Why the simpler rungs cannot meet the bar.")
    recipe: list[str] = Field(default_factory=list, max_length=12, description="The kind's and the modifiers' parts, name@version.")
    baselines: list[Baseline] = Field(default_factory=list, max_length=8)
    validation: str = Field(min_length=3, max_length=MAX_TEXT, description="The split, the folds, the holdout read once.")
    success: Optional[SuccessBar] = None
    decision_rule: str = Field(default="", max_length=MAX_TEXT)
    features: str = Field(default="", max_length=MAX_TEXT, description="The feature groups, and the columns kept out with why.")
    risks: list[Risk] = Field(default_factory=list, max_length=12)
    alternatives: list[Alternative] = Field(default_factory=list, max_length=8)
    approval: Optional[Approval] = None


CONTRACTS["solution_brief"] = SolutionBrief


def fingerprint(brief: SolutionBrief) -> str:
    """What a signature covers: everything but the approval itself."""
    doc = json.loads(brief.model_dump_json(exclude={"approval"}))
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


def problems(brief: SolutionBrief) -> list[str]:
    """G3: why the brief cannot be signed yet (empty when it can)."""
    out = []
    named = {b.name: b for b in brief.baselines}
    bar = brief.success
    if bar is None:
        out.append("no success bar: say which metric must reach what, before anything is trained")
    elif bar.value is None and not (bar.relative_to and (bar.factor is not None or bar.margin is not None)):
        out.append("the success bar has no target: an absolute value, or a factor or margin against a named baseline")
    elif bar.relative_to and bar.relative_to not in named:
        out.append(f"the success bar is measured against {bar.relative_to}, which is not one of the brief's baselines")
    elif bar.relative_to and bar.value is None and named[bar.relative_to].kind == "today" and named[bar.relative_to].value is None:
        out.append(f"the success bar is measured against {bar.relative_to}, today's process, but no number is given for it: give "
                   "today's number on the metric, or measure the bar against the simple model")
    if bar is not None:
        found = _direction_problem("the success bar", bar.metric, bar.op)
        out += [found] if found else []
        reported = REPORTS.get(bar.relative_to or "")
        if reported and bar.value is None and named.get(bar.relative_to) and named[bar.relative_to].kind == "simple_model" and bar.metric not in reported:
            out.append(f"the simple model {bar.relative_to} reports {', '.join(sorted(reported))} on the holdout, not {bar.metric}: measure the "
                       "bar on one of them, or give an absolute target")
        for g in bar.guards:
            if g.value is None and not g.relative_to:
                out.append(f"the guard on {g.metric} has no target: a value, or a baseline it must be no worse than")
            elif g.relative_to and g.relative_to not in named:
                out.append(f"the guard on {g.metric} is measured against {g.relative_to}, which is not one of the brief's baselines")
            found = _direction_problem(f"the guard", g.metric, g.op)
            out += [found] if found else []
    if len(named) != len(brief.baselines):
        out.append("two baselines share a name: give each its own, so the bar names one")
    if not brief.baselines:
        out.append("no baseline: name today's process or the recipe's simple model, the number to beat")
    if not brief.alternatives:
        out.append("no alternative: name at least one other way, and why it lost")
    return out


def signed(brief: SolutionBrief) -> bool:
    """Signed, and not changed since."""
    return brief.approval is not None and brief.approval.fingerprint == fingerprint(brief)


def sign(brief: SolutionBrief, by: str, at: str) -> SolutionBrief:
    """The brief signed by a person, refused while G3 has problems."""
    found = problems(brief)
    if found:
        raise ValueError("the brief cannot be signed yet: " + "; ".join(found))
    return brief.model_copy(update={"approval": Approval(by=by, at=at, fingerprint=fingerprint(brief))})


SIMPLE = {"binary": ("majority_class", "average_precision_random_ranking"), "multiclass": ("majority_class", "macro_f1"),
          "regression": ("training_mean", "mae")}  # the engine's simple models and the metric each is read on


def fill(task: str, metric: str, higher_is_better: bool, recipe: list[str], split: str, *, capacity: Optional[Mapping[str, Any]] = None,
         baseline_today: Optional[str] = None, business: Optional[str] = None, excluded: Optional[Mapping[str, str]] = None,
         risks: Optional[list[Risk]] = None, simple: Optional[str] = None, budget: Optional[float] = None) -> SolutionBrief:
    """A brief by rules from what the cards and the recipe hold. The default bar is the honest least: better than the
    recipe's simple model on the primary metric (a person raises it before signing); a team of K a period adds precision
    at K as the decision rule. ``simple``: a kind's own first baseline (its machine reports it), "" for a kind with none
    (the bar is then left to a person, and G3 asks for it); ``budget``: the review budget's share, the bar of recall at
    the budget (reviewing that share at random finds that share of the cases)."""
    simple = SIMPLE.get(task, ("majority_class", metric))[0] if simple is None else simple
    baselines = [Baseline(name=simple, kind="simple_model", description=f"the recipe's mandatory simple model ({simple.replace('_', ' ')})")] if simple else []
    if baseline_today:
        baselines.insert(0, Baseline(name="today", kind="today", description=str(baseline_today)[:MAX_TEXT]))
    op = ">" if higher_is_better else "<"
    if metric == "recall_at_budget" and budget:
        bar = SuccessBar(metric=metric, op=op, value=float(budget), business=business, guards=[])  # better than reviewing at random
    elif simple:
        bar = SuccessBar(metric=metric, op=op, relative_to=simple, margin=0.0, business=business, guards=[])
    else:
        bar = None
    rule = (f"the top {capacity['k']} cases a {capacity.get('period') or 'period'}, the threshold the K-th score of an average period"
            if capacity and capacity.get("k") else "the operating point from the review budget or the costs, set on the training rows")
    kept_out = "; ".join(f"`{c}`: {why}" for c, why in list((excluded or {}).items())[:12])
    return SolutionBrief(
        approach_level="classical_ml",
        approach_why=("a rule alone is today's process or the simple model, the bars to beat; a classical model on the table's "
                      "columns is the first rung that can learn from the outcome; a larger model is not needed for a table"),
        recipe=list(recipe)[:12], baselines=baselines,
        validation=f"{split} split with folds on the training rows; the holdout read once, after this brief is signed",
        success=bar, decision_rule=rule, features=("every column known at the prediction moment" + (f"; kept out: {kept_out}" if kept_out else "")),
        risks=list(risks or [])[:12],
        alternatives=[*([Alternative(choice=f"the {simple.replace('_', ' ')} alone", why_not="it is the bar to beat, not a solution")] if simple else []),
                      Alternative(choice="a frontier language model", why_not="the ladder stops at the first rung that can meet the bar: a classical "
                                                                             "model reads a table's columns, at a fraction of the cost and with a fixed answer")])


class BarVerdict(Contract):
    verdict: Literal["met", "not_met", "cannot_judge"]
    metric: str = Field(max_length=60)
    value: Optional[float] = None
    target: Optional[float] = None
    baseline: Optional[str] = Field(default=None, max_length=80)
    baseline_value: Optional[float] = None
    guards: list[dict[str, Any]] = Field(default_factory=list, max_length=8)
    reason: str = Field(default="", max_length=MAX_TEXT)


def _holds(value: float, op: str, target: float) -> bool:
    return {">": value > target, ">=": value >= target, "<": value < target, "<=": value <= target}[op]


def judge(brief: SolutionBrief, holdout: Mapping[str, Any], baselines: Mapping[str, Mapping[str, Any]]) -> BarVerdict:
    """The holdout's one read against the signed bar. ``holdout``: the holdout's metrics; ``baselines``: each simple
    model's metrics on the same holdout. An unsigned or changed brief, a missing metric or a missing baseline cannot be
    judged, and says why."""
    bar = brief.success
    if bar is None:
        return BarVerdict(verdict="cannot_judge", metric="", reason="the brief has no success bar")
    if not signed(brief):
        return BarVerdict(verdict="cannot_judge", metric=bar.metric, reason="the brief was not signed, or changed after it was")
    value = holdout.get(bar.metric)
    if not _number(value):
        return BarVerdict(verdict="cannot_judge", metric=bar.metric, reason=f"the holdout has no {bar.metric}, or it is not a finite number")
    target, base_value = bar.value, None
    if bar.relative_to:
        base_value = _baseline_value(brief, baselines, bar.relative_to, bar.metric)
        if base_value is None:
            return BarVerdict(verdict="cannot_judge", metric=bar.metric, value=float(value), baseline=bar.relative_to,
                              reason=f"the baseline {bar.relative_to} has no {bar.metric} on the holdout")
        target = base_value * bar.factor if bar.factor is not None else base_value + ((bar.margin or 0.0) if bar.op in (">", ">=") else -(bar.margin or 0.0))
    if target is None:
        return BarVerdict(verdict="cannot_judge", metric=bar.metric, value=float(value), reason="the bar has no target")
    guards = []
    for g in bar.guards:
        got = holdout.get(g.metric)
        ref = g.value if g.value is not None else (_baseline_value(brief, baselines, g.relative_to, g.metric) if g.relative_to else None)
        if not (_number(got) and _number(ref)):  # a guard that cannot be read leaves the bar unjudged, never failed
            return BarVerdict(verdict="cannot_judge", metric=bar.metric, value=float(value), baseline=bar.relative_to,
                              reason=f"the guard on {g.metric} cannot be read: the holdout's {g.metric} or its reference is missing")
        guards.append({"metric": g.metric, "value": float(got), "target": float(ref), "held": _holds(float(got), g.op, float(ref))})
    met = _holds(float(value), bar.op, float(target)) and all(g["held"] for g in guards)
    said = f"{bar.metric} {float(value):.4g} against the bar {bar.op} {float(target):.4g}" + (f" ({bar.relative_to} {base_value:.4g})" if base_value is not None else "")
    broken = [f"{g['metric']} {g['value']:.4g} against {g['target']:.4g}" for g in guards if not g["held"]]
    said += ("; a guard did not hold: " + ", ".join(broken)) if broken else ""
    return BarVerdict(verdict="met" if met else "not_met", metric=bar.metric, value=float(value), target=float(target), baseline=bar.relative_to,
                      baseline_value=base_value, guards=guards, reason=said + ("" if met else "; not met: the work goes back to the brief, with a new holdout"))


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _baseline_value(brief: SolutionBrief, baselines: Mapping[str, Mapping[str, Any]], name: str, metric: str) -> Optional[float]:
    """A named baseline's number on a metric: today's when a person gave it, else the holdout's simple model's (under the
    engine's own name for it when it has one)."""
    own = next((b for b in brief.baselines if b.name == name), None)
    if own is not None and own.kind == "today":
        return float(own.value) if _number(own.value) else None
    base = baselines.get(name) or {}
    for key in (metric, *SIMPLE_METRIC_ALIASES.get(name, {}).get(metric, ())):
        if _number(base.get(key)):
            return float(base[key])
    return None


# what each of the engine's simple models reports on the holdout (its aliases included): a relative bar on another metric
# could never be judged
REPORTS = {"majority_class": {"accuracy", "roc_auc", "average_precision", "macro_f1"},
           "training_mean": {"mae", "rmse", "mape", "r2", "bias"}, "training_median": {"mae", "rmse", "mape", "r2", "bias"}}

# a simple model's metric under another name in the engine's baselines (a random ranking's average precision is its rate)
SIMPLE_METRIC_ALIASES = {"majority_class": {"average_precision": ("average_precision_random_ranking",), "roc_auc": ("roc_auc",)},
                         "training_mean": {"mae": ("mae",), "rmse": ("rmse",)}, "training_median": {"mae": ("mae",)}}

__all__ = ["LADDER", "HIGHER", "Baseline", "Guard", "SuccessBar", "Approval", "SolutionBrief", "BarVerdict", "fingerprint", "problems", "signed", "sign",
           "fill", "judge", "SIMPLE"]
