"""A better workflow per task: what a careful data scientist checks after the first model (package 17.5).

``assess(task, ...)`` takes the out-of-fold predictions of the training rows and the predictions on the holdout, and
returns a JSON-ready summary. Everything that is *chosen* (a threshold, an interval's width) is chosen on the
out-of-fold predictions and only then read on the holdout, which is used once; everything is numbers, never rows.

* binary: the alert threshold from a review budget or from the cost of errors (else the F1-best one, said so), with
  precision and recall at it; calibration (Brier score and the expected calibration error); precision marked as a lower
  bound when the labels are partial;
* multiclass: per-class precision, recall and support, the most confused pairs of classes, how the tiny classes fare,
  top-k accuracy when probabilities exist;
* regression: prediction intervals from out-of-fold residuals and their coverage on the holdout, error by range of the
  prediction and by segment, behaviour at a cap, and whether errors grow with the prediction (a log target to try).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import re

import numpy as np
import pandas as pd

MAX_CLASSES = 30
MAX_SEGMENTS = 10


def _r(x: Any, digits: int = 4) -> Optional[float]:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return round(value, digits) if np.isfinite(value) else None


# ------------------------------------------------------------------------------------------------ binary


def _confusion_at(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    flags = score >= threshold
    tp = int((flags & (y == 1)).sum())
    fp = int((flags & (y == 0)).sum())
    fn = int((~flags & (y == 1)).sum())
    return {"alerts": int(flags.sum()), "alert_share": _r(flags.mean()), "true_positives": tp, "false_positives": fp,
            "false_negatives": fn, "precision": _r(tp / flags.sum()) if flags.sum() else None,
            "recall": _r(tp / max(int((y == 1).sum()), 1))}


def _sweep(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every distinct score as a threshold (alert when score >= it), with the true and false positives at each."""
    order = np.argsort(-score, kind="mergesort")
    s, t = score[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]  # the last row of each run of equal scores
    return s[last], np.cumsum(t)[last], np.cumsum(1 - t)[last]


def _threshold(y: np.ndarray, score: np.ndarray, budget: Optional[float], costs: Optional[Mapping[str, float]]) -> tuple[float, str]:
    if budget is not None and 0 < budget < 1:
        return float(np.quantile(score, 1 - budget)), f"the riskiest {budget:.2%} of rows, the review budget"
    thresholds, tp, fp = _sweep(y, score)
    positives = int((y == 1).sum())
    if costs and costs.get("false_positive") is not None and costs.get("false_negative") is not None:
        fp_cost, fn_cost = float(costs["false_positive"]), float(costs["false_negative"])
        cost = fp_cost * fp + fn_cost * (positives - tp)
        alert_none = fn_cost * positives  # a threshold above every score: no alerts at all
        if alert_none <= cost.min():
            return float(np.nextafter(score.max(), np.inf)), f"no alerts: every threshold costs more (a false alarm {fp_cost:g}, a miss {fn_cost:g})"
        return float(thresholds[int(np.argmin(cost))]), f"the lowest expected cost (a false alarm {fp_cost:g}, a miss {fn_cost:g})"
    f1 = np.where(tp > 0, 2 * tp / (2 * tp + fp + (positives - tp)), 0.0)
    return float(thresholds[int(np.argmax(f1))]), "the best F1 (no budget or cost was given: ask for one)"


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, Any]:
    if p.min() < 0 or p.max() > 1:
        return {"note": "scores are not probabilities: calibration not assessed"}
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    rows = []
    for b in range(bins):
        mask = which == b
        if mask.any():
            gap = abs(float(p[mask].mean()) - float(y[mask].mean()))
            ece += mask.mean() * gap
            rows.append({"lower": _r(edges[b], 2), "count": int(mask.sum()), "predicted": _r(p[mask].mean()), "observed": _r(y[mask].mean())})
    return {"brier": _r(np.mean((p - y) ** 2)), "expected_calibration_error": _r(ece), "bins": rows,
            "verdict": "close to calibrated" if ece < 0.05 else "not calibrated: read scores as a ranking, or calibrate on the training folds"}


def _binary(y_oof, p_oof, y_test, p_test, budget, costs, partial) -> dict[str, Any]:
    y_oof, y_test = np.asarray(y_oof).astype(int), np.asarray(y_test).astype(int)
    p_oof, p_test = np.asarray(p_oof, dtype=float), np.asarray(p_test, dtype=float)
    threshold, why = _threshold(y_oof, p_oof, budget, costs)
    out = {"threshold": {"value": _r(threshold, 6), "chosen_by": why, "chosen_on": "out-of-fold training predictions",
                         "training": _confusion_at(y_oof, p_oof, threshold), "holdout": _confusion_at(y_test, p_test, threshold)},
           "calibration": _calibration(y_test, p_test)}
    if partial:
        out["precision_note"] = ("The labels are partial: alerts without a label may be cases nobody found, so every precision here "
                                 "is a lower bound.")
    return out


# ------------------------------------------------------------------------------------------------ multiclass


def _multiclass(y_oof, p_oof, y_test, p_test, classes, tiny) -> dict[str, Any]:
    y_test = np.asarray(y_test)
    proba = np.asarray(p_test)
    proba = proba.astype(float) if proba.ndim == 2 else proba  # a probability matrix, or the predicted labels
    labels = list(classes) if classes is not None else None
    if proba.ndim == 2:
        pred_idx = proba.argmax(axis=1)
        pred = np.array([labels[i] for i in pred_idx]) if labels else pred_idx
    else:
        pred = proba
    truth = pd.Series(y_test).astype(str)
    guess = pd.Series(pred).astype(str)
    per_class = []
    for cls, support in truth.value_counts().items():
        tp = int(((truth == cls) & (guess == cls)).sum())
        predicted = int((guess == cls).sum())
        per_class.append({"class": cls, "support": int(support), "precision": _r(tp / predicted) if predicted else None,
                          "recall": _r(tp / support)})
    wrong = pd.DataFrame({"true": truth, "predicted": guess})
    wrong = wrong[wrong["true"] != wrong["predicted"]]
    pairs = wrong.value_counts().head(5)
    confused = [{"true": t, "predicted": p, "rows": int(n), "share_of_true": _r(n / max(int((truth == t).sum()), 1))}
                for (t, p), n in pairs.items()]
    out: dict[str, Any] = {"per_class": per_class[:MAX_CLASSES], "most_confused": confused,  # the record keeps the largest classes
                           "macro_recall": _r(np.mean([c["recall"] for c in per_class])) if per_class else None}  # over every class
    if tiny:
        out["tiny_classes"] = [c for c in per_class if c["class"] in {str(t) for t in tiny}][:MAX_CLASSES]
    if proba.ndim == 2 and labels:
        order = np.argsort(-proba, axis=1)
        index = {str(c): i for i, c in enumerate(labels)}
        true_idx = np.array([index.get(str(v), -1) for v in y_test])
        out["top_k_accuracy"] = {f"top_{k}": _r(np.mean([t in row[:k] for t, row in zip(true_idx, order)]))
                                 for k in (2, 3) if k < proba.shape[1]}
    return out


# ------------------------------------------------------------------------------------------------ regression


def _regression(y_oof, p_oof, y_test, p_test, segments: Optional[Mapping[str, Sequence[Any]]], cap: Optional[float]) -> dict[str, Any]:
    y_oof, p_oof = np.asarray(y_oof, dtype=float), np.asarray(p_oof, dtype=float)
    y_test, p_test = np.asarray(y_test, dtype=float), np.asarray(p_test, dtype=float)
    res_oof, res_test = y_oof - p_oof, y_test - p_test
    intervals = {}
    for level, (lo, hi) in {"80%": (0.1, 0.9), "90%": (0.05, 0.95)}.items():
        a, b = np.quantile(res_oof, [lo, hi])
        covered = float(np.mean((res_test >= a) & (res_test <= b)))
        intervals[level] = {"from_prediction": [_r(a), _r(b)], "holdout_coverage": _r(covered)}
    bins = pd.qcut(pd.Series(p_test).rank(method="first"), min(5, len(p_test)), labels=False)
    by_range = [{"prediction_from": _r(p_test[bins == b].min()), "prediction_to": _r(p_test[bins == b].max()),
                 "mae": _r(np.mean(np.abs(res_test[bins == b]))), "rows": int((bins == b).sum())} for b in sorted(bins.unique())]
    out: dict[str, Any] = {"mae": _r(np.mean(np.abs(res_test))), "intervals": intervals, "error_by_prediction_range": by_range}
    spread = pd.Series(np.abs(res_oof)).corr(pd.Series(p_oof), method="spearman")
    if pd.notna(spread):
        out["errors_grow_with_prediction"] = {"spearman": _r(spread), "suggest": "compare a log target" if spread > 0.3 else None}
    if cap is not None:  # found on the training rows (analyse's F-cap), never on the holdout
        at = y_test == cap
        out["at_cap"] = {"cap": _r(cap), "holdout_rows_at_cap": int(at.sum()),
                         "mae_at_cap": _r(np.mean(np.abs(res_test[at]))) if at.any() else None,
                         "mae_below_cap": _r(np.mean(np.abs(res_test[~at]))) if (~at).any() else None,
                         "note": "values at the cap are censored: their error is a lower bound"}
    if segments:
        seg_out = {}
        for name, values in list(segments.items())[:3]:
            s = pd.Series(list(values), dtype="object").astype(str)
            if len(s) != len(res_test):
                continue
            table = pd.DataFrame({"segment": s, "abs": np.abs(res_test)}).groupby("segment")["abs"].agg(["mean", "size"])
            table = table.sort_values("size", ascending=False).head(MAX_SEGMENTS)
            seg_out[str(name)[:100]] = {str(k)[:100]: {"mae": _r(v["mean"]), "rows": int(v["size"])} for k, v in table.iterrows()}
        out["error_by_segment"] = seg_out
    return out


def assess(task: str, y_oof: Sequence[Any], p_oof: Any, y_test: Sequence[Any], p_test: Any, *,
           budget: Optional[float] = None, costs: Optional[Mapping[str, float]] = None, partial_labels: bool = False,
           classes: Optional[Sequence[Any]] = None, tiny_classes: Optional[Sequence[Any]] = None,
           segments: Optional[Mapping[str, Sequence[Any]]] = None, cap: Optional[float] = None) -> dict[str, Any]:
    """The task's checks after the first model. ``p_oof``/``p_test``: scores for binary, a probability matrix (or
    labels) for multiclass, predictions for regression; ``segments``: holdout values of a few categorical columns;
    ``cap``: the target's cap found on the training rows (:func:`dclab_core.analyse.target_cap`)."""
    if task in ("binary", "binary_imbalanced", "text_tabular_binary"):
        return {"task": "binary", **_binary(y_oof, p_oof, y_test, p_test, budget, costs, partial_labels)}
    if task == "multiclass":
        return {"task": "multiclass", **_multiclass(y_oof, p_oof, y_test, p_test, classes, tiny_classes)}
    return {"task": "regression", **_regression(y_oof, p_oof, y_test, p_test, segments, cap)}


# ------------------------------------------------------------------------------------------------ from the plan

PERIOD_DAYS = {"day": 1.0, "week": 7.0, "month": 30.44}
PERCENT = re.compile(r"(\d*\.?\d+)\s*%")
PER_PERIOD = re.compile(r"(\d[\d,]*)\b[^%]*?\b(?:a|per|each|every)\s+(day|week|month)\b", re.I)
COST_OF = re.compile(r"(false (?:alarm|positive)s?|miss(?:es|ed)?|false negatives?)\D{0,40}?(\d[\d,]*(?:\.\d+)?)", re.I)


def capacity_share(text: str, rows: int, span_days: Optional[float]) -> Optional[float]:
    """A stated review capacity as a share of the rows: "0.5% of payments", or "200 rides a day" over the days the
    training rows span. None when it cannot be turned into a share (no period, or no time to count periods by)."""
    m = PERCENT.search(text or "")
    if m:
        share = float(m.group(1)) / 100
        return share if 0 < share < 1 else None
    m = PER_PERIOD.search(text or "")
    if not m or not span_days or rows <= 0:
        return None
    n = float(m.group(1).replace(",", ""))
    share = n * (span_days / PERIOD_DAYS[m.group(2).lower()]) / rows
    return share if 0 < share < 1 else None


def stated_costs(text: str) -> Optional[dict[str, float]]:
    """The cost of a false alarm and of a miss, when a sentence names both with a number."""
    found: dict[str, float] = {}
    for kind, number in COST_OF.findall(text or ""):
        key = "false_negative" if kind.lower().startswith(("miss", "false neg")) else "false_positive"
        found.setdefault(key, float(number.replace(",", "")))
    return found if len(found) == 2 else None


def threshold_inputs(plan: Optional[Mapping[str, Any]], rows: int, span_days: Optional[float]) -> dict[str, Any]:
    """What a plan (a :class:`~dclab_core.contracts.SolutionPlan` as a dict) says about the threshold and the labels:
    ``budget``, ``costs``, ``partial_labels`` for :func:`assess`, and a ``note`` when a stated policy could not be used."""
    decisions = {d.get("key"): d.get("choice") for d in (plan or {}).get("decisions", []) if isinstance(d, Mapping)}
    out: dict[str, Any] = {"budget": None, "costs": None, "partial_labels": decisions.get("labels.coverage") == "partial", "note": None}
    policy = decisions.get("threshold.policy")
    if isinstance(policy, Mapping) and policy.get("kind") == "review_budget":
        out["budget"] = capacity_share(str(policy.get("capacity") or ""), rows, span_days)
        if out["budget"] is None:
            out["note"] = f"the review capacity \"{str(policy.get('capacity'))[:80]}\" could not be turned into a share of rows: ask"
    elif isinstance(policy, Mapping) and policy.get("kind") == "expected_cost":
        out["costs"] = stated_costs(str(policy.get("costs") or ""))
        if out["costs"] is None:
            out["note"] = "the brief speaks of costs without a number for both a false alarm and a miss: ask"
    return out
