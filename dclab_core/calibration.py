"""A calibrator fitted on out-of-fold predictions (package 30.15, the framework's step 8): a model's scores turned into
probabilities that mean what they say, chosen and fitted on the training rows only, never on the holdout.

The training rows' out-of-fold scores are split into folds; each method (none, Platt's sigmoid on the score's logit,
isotonic) is fitted on the other folds and scored by its Brier score on the fold left out, and the method with the
lowest mean wins (none on a tie: a calibrator must earn its place). The winner is then fitted on every out-of-fold score.
``apply`` maps new scores through it; the holdout is read once, after, to say what the calibrator changed.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Sequence

import numpy as np
from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

METHODS = ("none", "sigmoid", "isotonic")
MIN_ROWS, MIN_EACH = 200, 20  # fewer out-of-fold rows, or fewer of either class, and nothing is fitted: the scores are kept as they are
EPS = 1e-6


class Calibrator(Versioned):
    """How scores become probabilities: the method, its fitted numbers, and how it was chosen (no row of the table)."""

    schema_version: Literal["dclab_core.calibrator.v1"] = _version("calibrator")
    method: Literal["none", "sigmoid", "isotonic"]
    a: Optional[float] = Field(default=None, description="sigmoid: the slope on the score's logit.")
    b: Optional[float] = Field(default=None, description="sigmoid: the intercept.")
    x: list[float] = Field(default_factory=list, max_length=200, description="isotonic: the scores at the steps' edges, increasing.")
    y: list[float] = Field(default_factory=list, max_length=200, description="isotonic: the probability at each edge.")
    chosen_on: str = Field(default="", max_length=300, description="What it was chosen on: the out-of-fold rows and folds.")
    brier: dict[str, float] = Field(default_factory=dict, description="Each method's mean Brier score on the folds left out.")
    reason: str = Field(default="", max_length=600)


CONTRACTS["calibrator"] = Calibrator


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _fit(method: str, scores: np.ndarray, y: np.ndarray) -> Calibrator:
    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(scores).reshape(-1, 1), y)
        return Calibrator(method="sigmoid", a=float(model.coef_[0][0]), b=float(model.intercept_[0]))
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, y)
        xs, ys = np.asarray(iso.X_thresholds_, dtype=float), np.asarray(iso.y_thresholds_, dtype=float)
        if len(xs) > 200:  # the steps kept as at most 200 edges, read at evenly spaced positions
            keep = np.unique(np.linspace(0, len(xs) - 1, 200).round().astype(int))
            xs, ys = xs[keep], ys[keep]
        return Calibrator(method="isotonic", x=[float(v) for v in xs], y=[float(v) for v in ys])
    return Calibrator(method="none")


def apply(calibrator: Calibrator, scores: Sequence[float]) -> np.ndarray:
    """New scores as probabilities through the calibrator (the scores themselves when it is none)."""
    s = np.asarray(scores, dtype=float)
    if calibrator.method == "sigmoid":
        return 1.0 / (1.0 + np.exp(-(float(calibrator.a) * _logit(s) + float(calibrator.b))))
    if calibrator.method == "isotonic" and calibrator.x:
        return np.interp(s, np.asarray(calibrator.x), np.asarray(calibrator.y))
    return s


def brier(y: Sequence[float], p: Sequence[float]) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def fit(oof_scores: Sequence[float], y: Sequence[int], folds: int = 5, seed: int = 0) -> Calibrator:
    """The calibrator chosen on the out-of-fold scores' folds (the Brier score of each method on the fold it never saw),
    then fitted on all of them. Too few rows, a class too small, or scores outside [0, 1]: none, with why."""
    s, t = np.asarray(oof_scores, dtype=float), np.asarray(y, dtype=int)
    n, pos = len(s), int(t.sum()) if len(t) else 0
    if n < MIN_ROWS or min(pos, n - pos) < MIN_EACH:
        return Calibrator(method="none", chosen_on=f"{n} out-of-fold rows, {pos} positive",
                          reason=f"too few out-of-fold rows or cases to fit a calibrator (at least {MIN_ROWS} rows and {MIN_EACH} of each class)")
    if not np.all(np.isfinite(s)) or s.min() < 0 or s.max() > 1:
        return Calibrator(method="none", chosen_on=f"{n} out-of-fold rows", reason="the scores are not probabilities between 0 and 1: kept as they are")
    rng = np.random.default_rng(seed)
    order = np.empty(n, dtype=int)
    for cls in (0, 1):  # stratified folds: each fold has its share of the cases
        idx = np.flatnonzero(t == cls)
        rng.shuffle(idx)
        order[idx] = np.arange(len(idx)) % folds
    scores: dict[str, list[float]] = {m: [] for m in METHODS}
    for k in range(folds):
        fit_rows, held = order != k, order == k
        if t[fit_rows].min() == t[fit_rows].max():
            continue
        for m in METHODS:
            scores[m].append(brier(t[held], apply(_fit(m, s[fit_rows], t[fit_rows]), s[held])))
    mean = {m: float(np.mean(v)) for m, v in scores.items() if v}
    best = min(METHODS, key=lambda m: (round(mean.get(m, math.inf), 6), METHODS.index(m)))  # a tie goes to the simpler
    chosen = _fit(best, s, t)
    said = (f"{best}: the lowest Brier score on the folds left out ({mean.get(best, float('nan')):.4f} against none's "
            f"{mean.get('none', float('nan')):.4f})" if best != "none" else "none: no method lowered the Brier score on the folds left out")
    return chosen.model_copy(update={"chosen_on": f"{n} out-of-fold rows of the training set, {folds} folds, {pos} positive",
                                     "brier": {m: round(v, 6) for m, v in mean.items()}, "reason": said})


__all__ = ["METHODS", "Calibrator", "apply", "brier", "fit"]
