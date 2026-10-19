"""Review and improve: where the first model fails, what follows from it, and what to try next (package 17.6).

``review(task, y, pred, ...)`` reads the out-of-fold predictions of the *training* rows (never the holdout) and returns
a :class:`~dclab_core.contracts.Review`:

* **weaknesses**, measured: lopsided errors under an absolute-error metric, errors that grow with the prediction, a
  capped target, predictions outside the range the training rows span, a segment or an entity that holds far more of
  the error than of the rows, classes the model rarely finds, a pair of classes it confuses;
* **hypotheses**, each citing the weaknesses it follows from (a model may propose more: :func:`check` refuses one that
  cites nothing measured);
* **next experiments**, from :data:`CATALOGUE`, ranked: the ones a host can run on training folds first (a median
  loss, a log target, predictions held to the training range, class weights), then the ones that need a person or a
  capability not built yet (entity history features, merging tiny classes).

:func:`adopt` is the rule a host applies to a round: the change is kept only if its paired gain over the same folds
beats both the margin and one standard error of the fold differences. Nothing here trains a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from dclab_core.contracts import Experiment, Finding, Hypothesis, Review

ABSOLUTE_METRICS = ("mae", "median_absolute_error", "mean_absolute_error")
SHARE_RATIO = 1.5        # entities holding 1.5 times their share of rows in error
MIN_SHARE = 0.05         # a segment with at least 5% of the rows
MAX_SEGMENT_LEVELS = 20


@dataclass(frozen=True)
class Kind:
    """One experiment a review can propose: the change a host makes, and whether a host can run it unattended."""

    tasks: tuple[str, ...]
    change: dict[str, Any]
    cost: str
    gain: str
    runnable: bool = True
    needs: str = ""
    addresses: tuple[str, ...] = field(default_factory=tuple)


CATALOGUE: dict[str, Kind] = {
    "loss.median": Kind(("regression",), {"loss": "median"}, "low",
                        "an absolute-error metric is minimised by the median: a median loss stops chasing the long tail",
                        addresses=("residual_skew", "cap")),
    "target.log": Kind(("regression",), {"target": "log1p"}, "low",
                       "errors that grow with the prediction shrink on a log scale",
                       addresses=("errors_grow",)),
    "predictions.clip": Kind(("regression",), {"clip": "fit_range"}, "low",
                             "predictions beyond what the target can be (a cap, a floor) only add error",
                             addresses=("out_of_range", "cap")),
    "weights.balanced": Kind(("binary", "multiclass"), {"class_weight": "balanced"}, "low",
                             "classes the model rarely finds get more weight in training",
                             addresses=("class_recall",)),
    "features.entity_history": Kind(("binary", "multiclass", "regression"), {"features": "entity_history"}, "medium",
                                    "an entity's own past behaviour, computed inside the training folds, explains its errors",
                                    runnable=False, needs="fold-safe entity behaviour features (package 15.4)",
                                    addresses=("entity",)),
    "features.by_segment": Kind(("binary", "multiclass", "regression"), {"features": "segment_interactions"}, "medium",
                                "a segment that behaves differently needs features or a model of its own",
                                runnable=False, needs="a person to say what sets the segment apart",
                                addresses=("segment",)),
    "classes.merge_tiny": Kind(("multiclass",), {"classes": "merge_tiny"}, "low",
                               "a class with too few examples to learn may be merged or routed by a rule",
                               runnable=False, needs="the person's decision: merging changes what is predicted",
                               addresses=("class_recall",)),
}


def _r(x: Any, digits: int = 4) -> Optional[float]:
    value = float(x)
    return round(value, digits) if np.isfinite(value) else None


def _w(key: str, topic: str, statement: str, evidence: dict[str, Any], implication: str, severity: str = "warning") -> Finding:
    slug = "".join(c if c.isalnum() or c in "_.-" else "-" for c in key.lower())[:100]
    return Finding(id=f"F-weak-{slug}", topic=topic, statement=statement[:3900], evidence=evidence, implication=implication[:3900],
                   affects=["model.families"], severity=severity, scope="training_rows")


# ------------------------------------------------------------------------------------------------ weaknesses


def _row_error(task: str, y: np.ndarray, pred: np.ndarray, classes: Optional[Sequence[Any]]) -> np.ndarray:
    if task == "regression":
        return np.abs(y.astype(float) - pred.astype(float))
    if task == "binary":
        return np.abs(y.astype(float) - pred.astype(float))
    if pred.ndim == 2:  # multiclass probabilities: one minus the probability of the true class
        index = {c: i for i, c in enumerate(classes)} if classes is not None else None
        cols = np.array([index.get(v, -1) if index is not None else int(v) for v in y])
        ok = (cols >= 0) & (cols < pred.shape[1])
        p_true = np.zeros(len(y))
        p_true[ok] = pred[np.flatnonzero(ok), cols[ok]]
        return 1.0 - p_true
    return (pd.Series(y).astype(str).to_numpy() != pd.Series(pred).astype(str).to_numpy()).astype(float)


SHUFFLES = 50            # entity labels shuffled this many times for the null of "one entity holds the error"


def _expected_error(task: str, pred: np.ndarray) -> Optional[np.ndarray]:
    """The error a calibrated model would make on each row: 2p(1-p) for binary, 1 - sum p^2 for multiclass."""
    if task == "binary":
        p = pred.astype(float)
        return 2 * p * (1 - p)
    if task == "multiclass" and pred.ndim == 2:
        return 1 - np.sum(pred.astype(float) ** 2, axis=1)
    return None


def _segments(values: pd.Series, error: np.ndarray, residual: Optional[np.ndarray], expected: Optional[np.ndarray],
              key: str) -> list[Finding]:
    """A segment where the model is off by more than chance allows: for a number, a bias (it is too high or too low
    there); for classes, more error than a calibrated model would make there. A segment that is merely noisier, or has
    a higher base rate the model already predicts, is not a weakness."""
    levels = values.astype(str).to_numpy()
    if residual is not None:
        signal, scale = residual, float(np.mean(np.abs(residual)))
    elif expected is not None:
        signal, scale = error - expected, float(np.mean(error))
    else:
        return []
    if scale <= 0:
        return []
    out = []
    for level in pd.unique(levels):
        mask = levels == level
        share = float(mask.mean())
        if share < MIN_SHARE or mask.sum() < 30:
            continue
        part = signal[mask]
        mean, se = float(part.mean()), float(part.std(ddof=1) / np.sqrt(len(part)))
        excess = abs(mean) if residual is not None else mean
        if excess >= 0.25 * scale and se > 0 and excess / se >= 3:
            what = (f"its predictions are {'too low' if mean > 0 else 'too high'} by {abs(mean):.3g} on average"
                    if residual is not None else f"its error is {mean:.3g} above what a calibrated model would make there")
            out.append(_w(f"segment-{key}-{level}", "segment",
                          f"Where {key} is {str(level)[:60]} ({share:.0%} of the training rows), {what} (out of fold).",
                          {"column": key, "level": str(level)[:100], "row_share": _r(share), "excess": _r(mean), "standard_error": _r(se)},
                          "This segment behaves differently from the rest: it needs features that say why, or a model of its own."))
    return sorted(out, key=lambda f: -abs(f.evidence["excess"]))[:3]


def _top_share(codes: np.ndarray, error: np.ndarray, groups: int) -> tuple[float, float]:
    sums = np.bincount(codes, weights=error, minlength=groups)
    sizes = np.bincount(codes, minlength=groups)
    top = np.argsort(-sums)[:max(1, groups // 10)]
    return float(sums[top].sum() / error.sum()), float(sizes[top].sum() / len(codes))


def _entity(values: pd.Series, error: np.ndarray, key: str) -> list[Finding]:
    """The 10% of entities with the most error hold far more of it than of the rows, and more than the same entities'
    sizes would give by chance (the entity labels shuffled ``SHUFFLES`` times)."""
    total = float(error.sum())
    codes, uniques = pd.factorize(values.astype(str))
    groups = len(uniques)
    if total <= 0 or groups < 10:
        return []
    error_share, row_share = _top_share(codes, error, groups)
    rng = np.random.default_rng(0)
    null = [_top_share(rng.permutation(codes), error, groups)[0] for _ in range(SHUFFLES)]
    beyond = float(np.quantile(null, 0.95))
    if error_share < SHARE_RATIO * row_share or error_share < 0.2 or error_share <= beyond:
        return []
    return [_w(f"entity-{key}", "entity",
               f"The 10% of {key} values with the most error ({max(1, groups // 10)} of {groups}) hold {error_share:.0%} of the out-of-fold "
               f"error on {row_share:.0%} of the training rows; with the {key} labels shuffled, at most {beyond:.0%} (95th percentile).",
               {"column": key, "entities": int(groups), "top_entities": int(max(1, groups // 10)), "row_share": _r(row_share),
                "error_share": _r(error_share), "shuffled_95th_percentile": _r(beyond)},
               "Some entities behave unlike the rest: their own history (computed inside the training folds) is the next "
               "thing to learn from.")]


def _regression(y: np.ndarray, pred: np.ndarray, metric: str, cap: Optional[float]) -> list[Finding]:
    out: list[Finding] = []
    residual = y - pred
    mae = float(np.mean(np.abs(residual)))
    if mae <= 0:
        return out
    skew = float(pd.Series(residual).skew())
    shift = (float(np.mean(residual)) - float(np.median(residual))) / mae
    if metric in ABSOLUTE_METRICS and (abs(skew) > 1 or abs(shift) > 0.1):
        out.append(_w("residual_skew", "residuals",
                      f"Out-of-fold errors are lopsided (skew {skew:.2f}; the mean error sits {shift:+.0%} of the MAE from the median "
                      f"error) while the work is judged on {metric}.",
                      {"residual_skew": _r(skew), "mean_minus_median_over_mae": _r(shift), "mae": _r(mae), "metric": metric},
                      "A squared loss predicts the mean and chases the tail; an absolute-error metric is minimised by the median."))
    if float(np.min(y)) > -1:
        grow = pd.Series(np.abs(residual)).corr(pd.Series(pred), method="spearman")
        if pd.notna(grow) and grow > 0.3:
            out.append(_w("errors_grow", "residuals",
                          f"Errors grow with the prediction (Spearman {grow:.2f} between the absolute error and the prediction).",
                          {"spearman": _r(grow)}, "On a log scale the errors would be closer to even: a log target is worth a fold test."))
    lo, hi = float(np.min(y)), float(np.max(y))
    outside = float(np.mean((pred < lo) | (pred > hi)))
    if outside > 0.01:
        out.append(_w("out_of_range", "predictions",
                      f"{outside:.1%} of out-of-fold predictions fall outside the range the target takes ({lo:.4g} to {hi:.4g}).",
                      {"share_outside": _r(outside), "low": _r(lo), "high": _r(hi)},
                      "A prediction the target cannot take only adds error: hold predictions to the training range."))
    if cap is not None:
        at = y >= cap
        if at.any() and (~at).any():
            mae_at, mae_below = float(np.mean(np.abs(residual[at]))), float(np.mean(np.abs(residual[~at])))
            above = float(np.mean(pred > cap))
            if mae_at >= SHARE_RATIO * mae_below or above > 0.01:
                out.append(_w("cap", "target",
                              f"The target is capped at {cap:.4g}: rows at the cap have an out-of-fold error of {mae_at:.3g} against "
                              f"{mae_below:.3g} below it, and {above:.1%} of predictions exceed the cap.",
                              {"cap": _r(cap), "share_at_cap": _r(float(np.mean(at))), "mae_at_cap": _r(mae_at), "mae_below_cap": _r(mae_below),
                               "share_predicted_above_cap": _r(above)},
                              "A capped value is censored: its true size is unknown, so a loss that chases it learns noise; a median "
                              "loss and predictions held to the cap are the experiments."))
    return out


def _classes(y: np.ndarray, pred: np.ndarray, classes: Optional[Sequence[Any]]) -> list[Finding]:
    truth = pd.Series(y).astype(str)
    if pred.ndim == 2:
        labels = [str(c) for c in classes] if classes is not None else [str(i) for i in range(pred.shape[1])]
        guess = pd.Series([labels[i] for i in pred.argmax(axis=1)])
    else:
        guess = pd.Series(pred).astype(str)
    recall = {c: float(((truth == c) & (guess == c)).sum()) / n for c, n in truth.value_counts().items()}
    support = truth.value_counts()
    macro = float(np.mean(list(recall.values())))
    weak = sorted((c for c in recall if recall[c] < 0.5 * macro), key=lambda c: recall[c])
    out = []
    if weak:
        out.append(_w("class_recall", "classes",
                      f"{len(weak)} class(es) are rarely found out of fold ({', '.join(f'{c} {recall[c]:.0%}' for c in weak[:5])}), against a "
                      f"macro recall of {macro:.0%}.",
                      {"classes": [str(c)[:100] for c in weak[:10]], "recall": [_r(recall[c]) for c in weak[:10]],
                       "support": [int(support[c]) for c in weak[:10]], "macro_recall": _r(macro)},
                      "Weighting classes by their rarity in training, or merging the tiny ones with the person, are the next steps."))
    wrong = pd.DataFrame({"t": truth, "p": guess})
    wrong = wrong[wrong["t"] != wrong["p"]]
    if len(wrong):
        (t, p), n = next(iter(wrong.value_counts().head(1).items()))
        share = n / max(int(support[t]), 1)
        if share >= 0.25:
            out.append(_w(f"confusion-{t}-{p}", "classes",
                          f"{share:.0%} of {t} rows are predicted as {p} out of fold.",
                          {"true": str(t)[:100], "predicted": str(p)[:100], "rows": int(n), "share_of_true": _r(share)},
                          "Features that tell these two apart are what the model lacks.", severity="info"))
    return out


def weaknesses(task: str, y: Sequence[Any], pred: Any, *, metric: str = "", frame: Optional[pd.DataFrame] = None,
               cap: Optional[float] = None, entities: Sequence[str] = (), classes: Optional[Sequence[Any]] = None) -> list[Finding]:
    """Where the model fails on the training rows' out-of-fold predictions. ``frame``: the training rows' inputs, in the
    same order as ``y`` (for segments and entities); ``entities``: its columns that name an entity (a customer, a biker)."""
    y_arr, p_arr = np.asarray(y), np.asarray(pred)
    if p_arr.ndim == 1 and task != "multiclass":
        p_arr = p_arr.astype(float)
    out: list[Finding] = []
    if task == "regression":
        out += _regression(y_arr.astype(float), p_arr.astype(float), metric, cap)
    elif task == "multiclass":
        out += _classes(y_arr, p_arr, classes)
    error = _row_error(task, y_arr, p_arr, classes)
    residual = y_arr.astype(float) - p_arr.astype(float) if task == "regression" else None
    expected = _expected_error(task, p_arr)
    if frame is not None and len(frame) == len(y_arr):
        for column in entities:
            if column in frame:
                out += _entity(frame[column], error, column)
        cats = [c for c in frame.columns if c not in entities and not pd.api.types.is_numeric_dtype(frame[c])
                and 2 <= frame[c].nunique() <= MAX_SEGMENT_LEVELS]
        for column in cats[:10]:
            out += _segments(frame[column], error, residual, expected, column)
    return out


# ------------------------------------------------------------------------------------------------ hypotheses and experiments


def _topic_of(finding: Finding) -> str:
    key = finding.id.removeprefix("F-weak-")
    return next((t for t in ("residual_skew", "errors_grow", "out_of_range", "cap", "class_recall", "entity", "segment")
                 if key.startswith(t)), "other")


def review(task: str, y: Sequence[Any], pred: Any, *, metric: str = "", frame: Optional[pd.DataFrame] = None,
           cap: Optional[float] = None, entities: Sequence[str] = (), classes: Optional[Sequence[Any]] = None,
           tried: Sequence[str] = ()) -> Review:
    """Weaknesses, the hypotheses that follow from them, and the next experiments (runnable first; ``tried`` left out)."""
    found = weaknesses(task, y, pred, metric=metric, frame=frame, cap=cap, entities=entities, classes=classes)
    by_topic: dict[str, list[Finding]] = {}
    for finding in found:
        by_topic.setdefault(_topic_of(finding), []).append(finding)
    hypotheses: list[Hypothesis] = []
    experiments: list[tuple[tuple, Experiment]] = []
    for name, kind in CATALOGUE.items():
        if task not in kind.tasks or name in tried:
            continue
        cited = [f for topic in kind.addresses for f in by_topic.get(topic, [])]
        if not _allowed(name, cited, metric):
            continue
        hypotheses.append(Hypothesis(statement=f"{kind.gain[:1].upper()}{kind.gain[1:]}.", cites=[f.id for f in cited][:20],
                                     test=("Cross-validate the same recipe and family with this change on the same training folds; "
                                           "keep it only if the paired gain beats the margin and one standard error.")
                                     if kind.runnable else f"Not run unattended: needs {kind.needs}."))
        severity = max((2 if f.severity == "critical" else 1 if f.severity == "warning" else 0) for f in cited)
        rank = (0 if kind.runnable else 1, -severity, {"low": 0, "medium": 1, "high": 2}[kind.cost], -len(cited))
        experiments.append((rank, Experiment(name=name, hypothesis=len(hypotheses) - 1, expected_gain=kind.gain[:500], cost=kind.cost)))
    ranked = [e for _, e in sorted(experiments, key=lambda pair: pair[0])]
    return Review(weaknesses=found[:100], hypotheses=hypotheses[:20], next_experiments=ranked[:20])


def _allowed(name: str, cited: Sequence[Finding], metric: str) -> bool:
    """Whether an experiment follows from the cited weaknesses: one of them must be what it addresses, and its own
    condition must hold (a median loss only for an absolute-error metric; clipping only when predictions leave the range)."""
    kind = CATALOGUE.get(name)
    relevant = [f for f in cited if kind is not None and _topic_of(f) in kind.addresses]
    if not relevant:
        return False
    if name == "predictions.clip" and not any(_topic_of(f) == "out_of_range" or
                                              (f.evidence.get("share_predicted_above_cap") or 0) > 0.01 for f in relevant):
        return False  # a cap alone does not make clipping useful: only predictions beyond it do
    if name == "loss.median" and metric not in ABSOLUTE_METRICS:
        return False
    return True


def runnable(experiments: Sequence[Experiment]) -> list[Experiment]:
    return [e for e in experiments if e.name in CATALOGUE and CATALOGUE[e.name].runnable]


def check(hypothesis: Hypothesis, measured: Sequence[Finding]) -> Optional[str]:
    """Why a proposed hypothesis is refused (it cites no measured weakness), or None when it may stand."""
    ids = {f.id for f in measured}
    missing = [c for c in hypothesis.cites if c not in ids]
    if missing:
        return f"cites what was not measured: {', '.join(missing[:5])}"
    return None


# ------------------------------------------------------------------------------------------------ a round's decision


# the one-sided 90% quantile of Student's t by degrees of freedom (no scipy: dclab_core keeps its four dependencies)
T90 = {1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476, 6: 1.440, 7: 1.415, 8: 1.397, 9: 1.383, 10: 1.372,
       15: 1.341, 20: 1.325, 30: 1.310}


def _t90(df: int) -> float:
    return T90[max((k for k in T90 if k <= df), default=1)]  # between rows, the smaller df: the stricter bound


def adopt(base_folds: Sequence[float], variant_folds: Sequence[float], higher: bool, margin: float = 0.0) -> dict[str, Any]:
    """Keep a change only if its paired gain on the same folds beats ``margin`` and the one-sided 90% bound of the fold
    differences (t quantile times the standard error: 1.89 with 3 folds); with fewer than two folds, never. A change that
    does nothing is kept about one time in ten, per round."""
    base, variant = np.asarray(base_folds, dtype=float), np.asarray(variant_folds, dtype=float)
    if len(base) != len(variant) or len(base) < 2:
        return {"adopted": False, "why": "the folds do not pair"}
    gains = (variant - base) if higher else (base - variant)
    mean, se = float(gains.mean()), float(gains.std(ddof=1) / np.sqrt(len(gains)))
    bound = _t90(len(gains) - 1) * se
    adopted = mean > margin and mean > bound
    why = (f"paired gain {mean:+.4g} beats the margin {margin:.3g} and {bound:.3g} (t 0.90 x standard error)" if adopted else
           f"paired gain {mean:+.4g} does not beat both the margin {margin:.3g} and {bound:.3g} (t 0.90 x standard error)")
    return {"adopted": bool(adopted), "mean_gain": _r(mean, 6), "standard_error": _r(se, 6), "bound": _r(bound, 6),
            "fold_gains": [_r(g, 6) for g in gains],
            "folds_better": int((gains > 0).sum()), "why": why}


# ------------------------------------------------------------------------------------------------ the critic (a role)

def _critic_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    review: Review = inputs["review"]
    task = inputs["task"]
    return {"task": task, "metric": inputs.get("metric", ""), "model": inputs.get("model", ""),
            "training_cv": inputs.get("score"), "tried": list(inputs.get("tried", []))[:10],
            "weaknesses": [f"{w.id}: {w.statement[:300]}" for w in review.weaknesses][:30],
            "experiments": {name: f"{kind.gain} ({'runs on folds' if kind.runnable else 'needs ' + kind.needs})"
                            for name, kind in CATALOGUE.items() if task in kind.tasks},
            **({"pitfalls_of_similar_problems": [str(x)[:480] for x in inputs["examples"][:3]]} if inputs.get("examples") else {})}


def _critic_parse(answer: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Hypotheses kept only when they cite measured weaknesses and name an experiment the catalogue has for this task;
    the order of the kept ones is the critic's ranking."""
    items = answer.get("hypotheses")
    if not isinstance(items, list):
        raise ValueError("hypotheses must be a list")
    review, task, tried = inputs["review"], inputs["task"], set(inputs.get("tried", []))
    kept, refused, order = [], [], []
    for index, item in enumerate(items):
        if index >= 10 or not isinstance(item, Mapping):
            refused.append({"experiment": str(item.get("experiment") if isinstance(item, Mapping) else item)[:80],
                            "why": "more than ten hypotheses" if index >= 10 else "not a hypothesis object"})
            continue
        name = item.get("experiment")
        try:
            hypothesis = Hypothesis(statement=str(item.get("statement") or "")[:3900] or "(no statement)",
                                    cites=[str(c) for c in item.get("cites") or []][:20] or ["(none)"],
                                    test=f"run {name} on the same folds")
        except Exception:  # noqa: BLE001 - a malformed proposal
            refused.append({"experiment": str(name)[:80], "why": "malformed"})
            continue
        why = check(hypothesis, review.weaknesses)
        if why is None and (name not in CATALOGUE or task not in CATALOGUE[name].tasks):
            why = f"{name!r} is not an experiment of the catalogue for a {task} task"
        if why is None:
            cited = [w for w in review.weaknesses if w.id in set(hypothesis.cites)]
            if not _allowed(name, cited, inputs.get("metric", "")):
                why = f"{name} does not follow from the weaknesses cited (none is what it addresses, or its condition fails)"
        if why is None and name in tried:
            why = f"{name} was already tried"
        if why:
            refused.append({"experiment": str(name)[:80], "why": why})
            continue
        kept.append({"experiment": name, "statement": hypothesis.statement, "cites": hypothesis.cites})
        if name not in order:
            order.append(name)
    return {"order": order, "kept": kept, "refused": refused}


def _critic_role():
    from dclab_core.agents import Role

    return Role(name="critic",
                description="Reads the measured weaknesses and ranks the catalogue's experiments, each hypothesis citing what it fixes; "
                            "code refuses one that cites nothing measured or names an experiment the catalogue lacks.",
                evidence=_critic_evidence, parse=_critic_parse,
                fallback=lambda inputs: {"order": [e.name for e in inputs["review"].next_experiments], "kept": [], "refused": []})


CRITIC = _critic_role()


def critique(advisor: Any, review_: Review, task: str, *, metric: str = "", model: str = "", score: Optional[float] = None,
             tried: Sequence[str] = (), examples: Sequence[str] = ()) -> dict[str, Any]:
    """The experiments in the critic's order (the rule order with no advisor, or when it abstains or is refused). ``examples`` (30.18):
    what goes wrong in the closest business examples, read as advice; a hypothesis still cites a measured weakness."""
    from dclab_core.advisor import EvidenceError
    from dclab_core.agents import run_role

    score = score if score is not None and np.isfinite(score) else None  # a missing score is said as missing, never NaN
    try:
        result = run_role(CRITIC, advisor, {"review": review_, "task": task, "metric": metric, "model": model, "score": score,
                                             "tried": list(tried), "examples": list(examples)})
    except EvidenceError:  # the critic is advice: a bad summary falls back to the rules' order, never drops the review
        return {"order": [e.name for e in review_.next_experiments], "kept": [], "refused": [], "source": "fallback",
                "reason": "the evidence could not be summarised"}
    order = result.output["order"] + [e.name for e in review_.next_experiments if e.name not in result.output["order"]]
    return {**result.output, "order": order, "source": result.source, "reason": result.reason}
