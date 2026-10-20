"""A model's evaluation as numbers and curves (the owner, 2026-10-20: "there is no evaluation part: precision, recall,
F1 score and the other important things, and a confusion matrix").

``evaluate(task, y_true, scores, ...)`` turns one set of true answers and a model's scores into an :class:`Evaluation`:

- binary: the measures at a cutoff (precision, recall, F1, accuracy, balanced accuracy, specificity, MCC) and those that
  need no cutoff (ROC-AUC, average precision, Brier score, log loss); the confusion counts at every cutoff on a grid of
  51, so a page can move the cutoff without asking again; the ROC and precision-recall curves, cumulative gains,
  calibration in ten bins and the scores' histogram by class;
- multiclass: accuracy, macro and weighted precision, recall and F1, log loss, each class's precision, recall, F1 and
  support, and the confusion matrix;
- regression: MAE, RMSE, R², median absolute error, MAPE when no answer is zero, the residuals' histogram and the
  answers against the predictions in ten bins of the prediction.

Nothing here chooses anything: it describes scores already made. The caller decides which rows they are (the
validation folds of the training rows, or the holdout read once) and says so in ``rows``. Every list is bounded
(at most ``MAX_POINTS`` points a curve), and no row of a table is kept: only counts, rates and bin edges.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

VERSION = "1"
MAX_POINTS = 101
CUTOFFS = tuple(round(float(t), 2) for t in np.linspace(0, 1, 51))
BINARY = ("binary", "binary_imbalanced", "text_tabular_binary")


class Evaluation(BaseModel):
    """One evaluation, on one set of rows: measures, confusion counts and curves."""

    model_config = ConfigDict(extra="forbid")

    version: str = VERSION
    kind: Literal["binary", "multiclass", "regression"]
    rows: Literal["holdout", "validation"]
    source: Literal["scores", "record"] = "scores"  # from the model's scores, or read back from the numbers a stage record kept
    n: int
    metrics: dict[str, Optional[float]] = Field(default_factory=dict)
    cutoff: Optional[float] = None
    confusion: Optional[dict[str, Any]] = None
    cutoffs: list[dict[str, Any]] = Field(default_factory=list)
    curves: dict[str, Any] = Field(default_factory=dict)
    per_class: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _f(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, 6) if np.isfinite(value) else None


def _thin(points: Sequence[Sequence[float]], limit: int = MAX_POINTS) -> list[list[float]]:
    """At most ``limit`` points, the first and the last kept."""
    pts = [[_f(a), _f(b)] for a, b in points]
    pts = [p for p in pts if p[0] is not None and p[1] is not None]
    if len(pts) <= limit:
        return pts
    keep = np.unique(np.linspace(0, len(pts) - 1, limit).round().astype(int))
    return [pts[i] for i in keep]


def _at(y: np.ndarray, s: np.ndarray, t: float) -> dict[str, int]:
    flag = s >= t
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": int((y == 1).sum()) - tp, "tn": int((y == 0).sum()) - fp}


def at_cutoff(c: dict[str, int]) -> dict[str, Optional[float]]:
    """The measures a cutoff gives, from its confusion counts."""
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if precision is not None and recall is not None else None)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"precision": _f(precision), "recall": _f(recall), "f1": _f(f1), "specificity": _f(specificity),
            "accuracy": _f((tp + tn) / n) if n else None,
            "balanced_accuracy": _f((recall + specificity) / 2) if recall is not None and specificity is not None else None,
            "mcc": _f((tp * tn - fp * fn) / denom) if denom else None,
            "flagged_share": _f((tp + fp) / n) if n else None}


def _binary(y: np.ndarray, s: np.ndarray, cutoff: float, rows: str) -> Evaluation:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve, roc_auc_score, roc_curve

    y, s = y.astype(int), np.clip(s.astype(float), 0.0, 1.0)
    both = len(np.unique(y)) == 2
    notes = [] if both else ["The rows hold one class only: the ranking measures and curves cannot be measured."]
    grid = [{"t": t, **_at(y, s, t)} for t in CUTOFFS]
    if cutoff not in CUTOFFS:
        grid = sorted(grid + [{"t": round(float(cutoff), 4), **_at(y, s, cutoff)}], key=lambda r: r["t"])
    chosen = _at(y, s, cutoff)
    metrics = {**at_cutoff(chosen), "positive_rate": _f(y.mean()) if len(y) else None,
               "roc_auc": _f(roc_auc_score(y, s)) if both else None,
               "average_precision": _f(average_precision_score(y, s)) if both else None,
               "brier": _f(brier_score_loss(y, s)) if len(y) else None,
               "log_loss": _f(log_loss(y, np.clip(s, 1e-6, 1 - 1e-6), labels=[0, 1])) if len(y) else None}
    curves: dict[str, Any] = {}
    if both:
        fpr, tpr, _ = roc_curve(y, s)
        precision, recall, _ = precision_recall_curve(y, s)
        order = np.argsort(-s, kind="stable")
        found = np.cumsum(y[order]) / max(int(y.sum()), 1)
        share = np.arange(1, len(y) + 1) / len(y)
        curves["roc"] = _thin(list(zip(fpr, tpr)))
        curves["pr"] = _thin(list(zip(recall[::-1], precision[::-1])))
        curves["gains"] = _thin([[0.0, 0.0]] + list(zip(share, found)))
    edges = np.linspace(0, 1, 11)
    bins = []
    for i, lower in enumerate(edges[:-1]):
        mask = (s >= lower) & ((s < edges[i + 1]) if i < 9 else (s <= 1))
        if mask.any():
            bins.append({"lower": _f(lower), "upper": _f(edges[i + 1]), "count": int(mask.sum()), "predicted": _f(s[mask].mean()), "observed": _f(y[mask].mean())})
    curves["calibration"] = bins
    hist_edges = np.linspace(0, 1, 21)
    curves["histogram"] = {"edges": [_f(e) for e in hist_edges],
                           "positive": np.histogram(s[y == 1], hist_edges)[0].astype(int).tolist(),
                           "negative": np.histogram(s[y == 0], hist_edges)[0].astype(int).tolist()}
    return Evaluation(kind="binary", rows=rows, n=int(len(y)), metrics=metrics, cutoff=_f(cutoff),
                      confusion={"labels": ["negative", "positive"], **chosen}, cutoffs=grid, curves=curves, notes=notes)


def _multiclass(y: np.ndarray, proba: np.ndarray, labels: Sequence[Any], rows: str) -> Evaluation:
    from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support

    k = proba.shape[1]
    y = y.astype(int)
    pred = proba.argmax(axis=1)
    idx = list(range(k))
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=idx, zero_division=0)
    names = [str(labels[i]) if i < len(labels) else str(i) for i in idx]
    weights = support / support.sum() if support.sum() else np.zeros(k)
    metrics = {"accuracy": _f((pred == y).mean()) if len(y) else None,
               "macro_precision": _f(p.mean()), "macro_recall": _f(r.mean()), "macro_f1": _f(f.mean()),
               "weighted_precision": _f((p * weights).sum()), "weighted_recall": _f((r * weights).sum()), "weighted_f1": _f((f * weights).sum()),
               "log_loss": _f(log_loss(y, np.clip(proba, 1e-9, 1), labels=idx)) if len(y) else None}
    per_class = [{"label": names[i], "precision": _f(p[i]), "recall": _f(r[i]), "f1": _f(f[i]), "support": int(support[i])} for i in idx]
    matrix = confusion_matrix(y, pred, labels=idx).astype(int).tolist()
    return Evaluation(kind="multiclass", rows=rows, n=int(len(y)), metrics=metrics, confusion={"labels": names, "matrix": matrix},
                      per_class=per_class)


def _regression(y: np.ndarray, pred: np.ndarray, rows: str) -> Evaluation:
    y, pred = y.astype(float), pred.astype(float)
    err = pred - y
    total = ((y - y.mean()) ** 2).sum()
    metrics = {"mae": _f(np.abs(err).mean()), "rmse": _f(np.sqrt((err ** 2).mean())), "median_absolute_error": _f(np.median(np.abs(err))),
               "r2": _f(1 - (err ** 2).sum() / total) if total else None,
               "mape": _f(np.abs(err / y).mean()) if len(y) and not (y == 0).any() else None,
               "mean_error": _f(err.mean())}
    edges = np.histogram_bin_edges(err, bins=20) if len(err) else np.linspace(-1, 1, 21)
    order = np.argsort(pred, kind="stable")
    binned = []
    for part in np.array_split(order, 10):
        if len(part):
            binned.append({"count": int(len(part)), "predicted": _f(pred[part].mean()), "actual": _f(y[part].mean()),
                           "mae": _f(np.abs(err[part]).mean())})
    curves = {"residuals": {"edges": [_f(e) for e in edges], "counts": np.histogram(err, edges)[0].astype(int).tolist()},
              "by_prediction": binned}
    notes = [] if metrics["mape"] is not None else ["MAPE is left out: an answer of zero makes it undefined."]
    return Evaluation(kind="regression", rows=rows, n=int(len(y)), metrics=metrics, curves=curves, notes=notes)


def evaluate(task: str, y_true: Any, scores: Any, *, rows: Literal["holdout", "validation"], cutoff: float = 0.5,
             labels: Optional[Sequence[Any]] = None) -> Evaluation:
    """The evaluation of ``scores`` against ``y_true``: a positive class's probability for a binary task, each class's
    probabilities (rows of ``scores``) for a multiclass one, the predicted value for a regression."""
    y, s = np.asarray(y_true), np.asarray(scores)
    if task in BINARY:
        if s.ndim == 2:
            s = s[:, -1]
        return _binary(y, s, float(cutoff), rows)
    if task == "multiclass":
        if s.ndim == 1:
            raise ValueError("a multiclass evaluation needs each class's probabilities")
        return _multiclass(y, s, list(labels or range(s.shape[1])), rows)
    if task in ("regression", "timeseries_regression"):
        return _regression(y, s, rows)
    raise ValueError(f"no evaluation for the task {task!r}")


def _confusion(n: int, pos: int, flagged: int, tp: int) -> Optional[dict[str, int]]:
    """The four counts from the rows, the cases, the rows flagged and the cases among them; None when they do not add up."""
    fp, fn = flagged - tp, pos - tp
    tn = n - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn} if min(tp, fp, fn, tn) >= 0 else None


def from_record(evidence: dict[str, Any]) -> Optional[Evaluation]:
    """The holdout evaluation of a final-stage record: the full one it kept (``evidence.evaluation``) when the stage ran
    with it, else what the numbers it did record give back: the measures, the confusion counts at its cutoffs (exact: the
    cases, the rows flagged and the cases among them are in the record), calibration, and each class's results. The curves
    that need the scores (ROC, precision-recall, gains, the score histogram) are not there for such a record, and it says so.
    None when the record holds no holdout metrics."""
    stored = (evidence.get("evaluation") or {}).get("holdout")
    if isinstance(stored, dict):
        try:
            return Evaluation.model_validate(stored)
        except ValueError:
            pass
    hm = evidence.get("holdout_metrics") or {}
    n = int(evidence.get("holdout_rows") or 0)
    assessment = evidence.get("task_assessment") or {}
    nums = {k: _f(v) for k, v in hm.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "fit_seconds"}
    if "per_class_f1" in hm:
        per_class = []
        for row in assessment.get("per_class") or []:
            p, r = row.get("precision"), row.get("recall")
            f1 = 2 * p * r / (p + r) if p and r else 0.0
            per_class.append({"label": str(row.get("class")), "precision": _f(p), "recall": _f(r), "f1": _f(f1), "support": int(row.get("support") or 0)})
        metrics = {**nums, "macro_recall": _f(assessment.get("macro_recall")), **{f"{k}_accuracy": _f(v) for k, v in (assessment.get("top_k_accuracy") or {}).items()}}
        return Evaluation(kind="multiclass", rows="holdout", source="record", n=n, metrics=metrics, per_class=per_class,
                          curves={"most_confused": assessment.get("most_confused") or []},
                          notes=["The full confusion matrix is not in this project's record; its most confused pairs are."])
    if "mae" in hm or "rmse" in hm:
        return Evaluation(kind="regression", rows="holdout", source="record", n=n, metrics=nums,
                          notes=["The error curves need the predictions, which this project's record did not keep."])
    if "roc_auc" not in hm and "precision" not in hm:
        return None
    rate = hm.get("positive_rate")
    pos = int(round(float(rate) * n)) if isinstance(rate, (int, float)) and n else None
    grid: dict[float, dict[str, int]] = {}
    if pos is not None:
        for row in evidence.get("threshold_analysis") or []:
            tp, alerts, t = row.get("true_positives"), row.get("alerts"), row.get("threshold")
            c = _confusion(n, pos, int(alerts), int(tp)) if None not in (tp, alerts, t) else None
            if c:
                grid[round(float(t), 4)] = c
    held = ((assessment.get("threshold") or {}).get("holdout") or {})
    chosen_at = (assessment.get("threshold") or {}).get("value")
    if held and chosen_at is not None and n:
        c = {"tp": int(held["true_positives"]), "fp": int(held["false_positives"]), "fn": int(held["false_negatives"])}
        c["tn"] = n - sum(c.values())
        if c["tn"] >= 0:
            grid[round(float(chosen_at), 4)] = c
    cutoff = hm.get("threshold")
    cutoff = float(cutoff) if isinstance(cutoff, (int, float)) else 0.5
    precision, recall = hm.get("precision"), hm.get("recall")
    if pos is not None and isinstance(recall, (int, float)):
        tp = int(round(recall * pos))
        flagged = int(round(tp / precision)) if precision else None
        c = _confusion(n, pos, flagged, tp) if flagged is not None else None
        if c:
            grid.setdefault(round(cutoff, 4), c)
    cutoffs = [{"t": t, **c} for t, c in sorted(grid.items())]
    confusion = grid.get(round(cutoff, 4))
    metrics = {**nums}
    if confusion:
        extra = at_cutoff(confusion)
        for k in ("specificity", "balanced_accuracy", "mcc", "flagged_share"):
            metrics[k] = extra[k]
    cal = evidence.get("calibration_bins") or ((assessment.get("calibration") or {}).get("bins")) or []
    curves = {"calibration": [{k: b.get(k) for k in ("lower", "upper", "count", "predicted", "observed")} for b in cal]}
    return Evaluation(kind="binary", rows="holdout", source="record", n=n, metrics=metrics, cutoff=_f(cutoff),
                      confusion={"labels": ["negative", "positive"], **confusion} if confusion else None, cutoffs=cutoffs, curves=curves,
                      notes=["ROC, precision-recall, gains and the score histogram need the model's scores, which this project's record did not keep: "
                             "a project whose final stage runs from now on keeps them. The confusion counts here are exact, from the cases and the alerts recorded."])


__all__ = ["Evaluation", "evaluate", "from_record", "at_cutoff", "CUTOFFS", "MAX_POINTS", "VERSION"]
