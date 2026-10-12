"""Rules, anomaly detectors and positive-unlabelled learning for rare events (package 15.5).

Each is a scikit-learn style estimator with ``fit(X, y)`` and ``predict_proba(X)``, so the screen ranks it on the same
folds and metrics as the classifiers. The three detectors (:class:`PercentileRules`, :class:`ECOD`,
:class:`IsolationForestDetector`) never look at ``y``: their score says how unusual a row is against the fit rows, and
the runner always fits them on every fit row, never a resampled set. Their "probability" is the share of fit rows
scoring lower (an empirical CDF): a ranking, not a calibrated probability.

:class:`PUBagging` learns from labelled positives against the unlabelled rows when labels are partial (an unlabelled
row may be a case nobody found): each of its trees is fitted on the positives and an equal random draw of unlabelled
rows, and the scores are averaged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

DETECTORS = ("percentile_rules", "ecod", "isolation_forest")  # never see labels, never resampled rows


def _as_matrix(X: Any) -> np.ndarray:
    X = X.toarray() if hasattr(X, "toarray") else X
    return np.asarray(X, dtype=float)


class _Detector(BaseEstimator, ClassifierMixin):
    """Fits on the rows only; scores as the share of fit rows that score lower."""

    def _raw(self, X: np.ndarray) -> np.ndarray:  # higher = more unusual
        raise NotImplementedError

    def _learn(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def fit(self, X: Any, y: Any = None, **_: Any) -> "_Detector":
        X = _as_matrix(X)
        self.fill_ = np.nan_to_num(np.nanmedian(X, axis=0)) if len(X) else np.zeros(X.shape[1])
        X = np.where(np.isnan(X), self.fill_, X)
        self._learn(X)
        self.reference_ = np.sort(self._raw(X))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X = _as_matrix(X)
        X = np.where(np.isnan(X), self.fill_, X)
        share = np.searchsorted(self.reference_, self._raw(X), side="right") / max(len(self.reference_), 1)
        return np.column_stack([1 - share, share])


class PercentileRules(_Detector):
    """A rule score: how far each value lies beyond its column's 1st or 99th percentile on the fit rows (in units of
    the column's interquartile range), summed over columns. The thresholds a team would write by hand, from data."""

    def __init__(self, low: float = 1.0, high: float = 99.0):
        self.low, self.high = low, high

    def _learn(self, X: np.ndarray) -> None:
        self.lo_, self.hi_ = np.percentile(X, self.low, axis=0), np.percentile(X, self.high, axis=0)
        q1, q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
        self.scale_ = np.where(q3 - q1 > 0, q3 - q1, np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0))

    def _raw(self, X: np.ndarray) -> np.ndarray:
        beyond = np.maximum(0, X - self.hi_) + np.maximum(0, self.lo_ - X)
        return (beyond / self.scale_).sum(axis=1) + (beyond > 0).sum(axis=1)


class ECOD(_Detector):
    """Empirical-cumulative-distribution outlier detection (Li et al., 2022): for each column, the tail probability of
    the value on the fit rows (the left, right or skew-chosen tail), summed as -log over columns."""

    GRID = 1001  # the fit rows' distribution as quantiles: a detector's state never holds a row's value

    def _learn(self, X: np.ndarray) -> None:
        self.sorted_ = np.quantile(X, np.linspace(0, 1, min(self.GRID, max(len(X), 2))), axis=0)
        mean, std = X.mean(axis=0), X.std(axis=0)
        self.skew_ = np.where(std > 0, ((X - mean) ** 3).mean(axis=0) / np.where(std > 0, std, 1) ** 3, 0.0)

    def _raw(self, X: np.ndarray) -> np.ndarray:
        n = self.sorted_.shape[0]
        left = np.column_stack([np.searchsorted(self.sorted_[:, j], X[:, j], side="right") for j in range(X.shape[1])]) / n
        right = np.column_stack([n - np.searchsorted(self.sorted_[:, j], X[:, j], side="left") for j in range(X.shape[1])]) / n
        left, right = np.clip(left, 1 / n, 1), np.clip(right, 1 / n, 1)
        tail_left, tail_right = -np.log(left), -np.log(right)
        skewed = np.where(self.skew_ < 0, tail_left, tail_right)
        return np.maximum.reduce([tail_left.sum(axis=1), tail_right.sum(axis=1), skewed.sum(axis=1)])


class IsolationForestDetector(_Detector):
    """scikit-learn's Isolation Forest: rows that few random splits isolate are unusual."""

    def __init__(self, n_estimators: int = 200, random_state: int = 42, n_jobs: int = 4):
        self.n_estimators, self.random_state, self.n_jobs = n_estimators, random_state, n_jobs

    def _learn(self, X: np.ndarray) -> None:
        from sklearn.ensemble import IsolationForest

        self.forest_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=self.n_jobs).fit(X)

    def _raw(self, X: np.ndarray) -> np.ndarray:
        return -self.forest_.score_samples(X)


class PUBagging(BaseEstimator, ClassifierMixin):
    """Positive-unlabelled bagging (Mordelet and Vert, 2014): each round fits a small tree ensemble on every labelled
    positive against an equal-sized random draw of unlabelled rows; the scores are averaged."""

    def __init__(self, n_estimators: int = 15, trees: int = 50, random_state: int = 42, n_jobs: int = 4):
        self.n_estimators, self.trees, self.random_state, self.n_jobs = n_estimators, trees, random_state, n_jobs

    def fit(self, X: Any, y: Any, **_: Any) -> "PUBagging":
        from sklearn.ensemble import ExtraTreesClassifier

        X = _as_matrix(X)
        self.fill_ = np.nan_to_num(np.nanmedian(X, axis=0))
        X = np.where(np.isnan(X), self.fill_, X)
        y = np.asarray(y).astype(int)
        positives, unlabelled = np.flatnonzero(y == 1), np.flatnonzero(y != 1)
        rng = np.random.default_rng(self.random_state)
        self.models_ = []
        draw = max(1, min(len(unlabelled), max(len(positives), 1)))
        for round_ in range(self.n_estimators):
            chosen = rng.choice(unlabelled, size=draw, replace=False) if len(unlabelled) else unlabelled
            rows = np.concatenate([positives, chosen])
            model = ExtraTreesClassifier(n_estimators=self.trees, min_samples_leaf=2, random_state=self.random_state + round_,
                                         n_jobs=self.n_jobs)
            self.models_.append(model.fit(X[rows], y[rows]))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X = _as_matrix(X)
        X = np.where(np.isnan(X), self.fill_, X)
        score = np.mean([m.predict_proba(X)[:, list(m.classes_).index(1)] if 1 in m.classes_ else np.zeros(len(X))
                         for m in self.models_], axis=0)
        return np.column_stack([1 - score, score])

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.mean([m.feature_importances_ for m in self.models_], axis=0)
