"""Stage implementations, result writing, and reports for ``expansion_v1``.

Five stages run per dataset, in order: data_understanding, leakage_audit,
feature_engineering, model_selection, optimization_reliability.  The first four
see training rows only (holdout *labels* are never read before the last stage;
data_understanding compares holdout *feature* distributions only, as the
original campaign did).  The last stage consumes the locked holdout exactly
once.  Every claim cites ``evidence.<key>`` paths that exist in its result.

This module intentionally does not import :mod:`dclab_rnd.science` (which pulls
in the full general_pipeline stack, including catboost); the conventions are
mirrored instead: ``_claim`` shape, schema_version 2 results, ``mean - 0.25*std``
ranking, smallest-within-tolerance feature selection, and an explicit tuning
acceptance margin.
"""

from __future__ import annotations

import json
import math
import re
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mutual_info_score,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from dclab_rnd.expansion import CAMPAIGN_ID
from dclab_rnd.expansion.datasets import (
    DATASET_ORDER,
    HIGHER_IS_BETTER,
    RANDOM_STATE,
    SPECS,
    DatasetSpec,
    TaskBundle,
)
from dclab_rnd.provenance import capture_provenance

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

SCHEMA_VERSION = 2
STAGES = (
    "data_understanding",
    "leakage_audit",
    "feature_engineering",
    "model_selection",
    "optimization_reliability",
)
FIRST_EXPERIMENT_NUMBER = 51
N_JOBS = 4
ADJUST_STD = 0.25
BOOTSTRAP_DRAWS = 400
TEXT_TREE_MAX_FEATURES = 2000
REQUIRED_KEYS = (
    "schema_version",
    "campaign_id",
    "experiment_id",
    "dataset",
    "kind",
    "task_type",
    "primary_metric",
    "question",
    "hypothesis",
    "decision_time_rule",
    "holdout_policy",
    "source",
    "status",
    "started_at",
    "completed_at",
    "setup_summary",
    "evidence",
    "claims",
    "provenance",
)
CLAIM_KINDS = {"fact", "risk", "decision", "recommendation"}
METRIC_LABEL = {
    "average_precision": "average precision (PR-AUC)",
    "roc_auc": "ROC-AUC",
    "macro_f1": "macro-F1",
    "mae": "MAE",
}
_SUSPICIOUS_NAME = re.compile(
    r"(^|_)(target|label|outcome|result|response|accepted|approved|converted|defaulted|churned|final)(_|$)",
    re.IGNORECASE,
)

_PROFILE = {"fast": False}


@contextmanager
def fast_profile():
    """Shrink ensembles, candidates, and bootstrap draws (used by unit tests)."""
    from threadpoolctl import threadpool_limits

    previous = _PROFILE["fast"]
    _PROFILE["fast"] = True
    try:
        with threadpool_limits(limits=1):  # single-threaded BLAS/OpenMP: predictable on shared CI
            yield
    finally:
        _PROFILE["fast"] = previous


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _claim(
    claim_id: str,
    kind: str,
    statement: str,
    evidence: list[str],
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "kind": kind,
        "statement": statement,
        "evidence": evidence,
        "limitations": limitations or [],
    }


def _f(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if abs(number) >= 100:
        return f"{number:.1f}"
    return f"{number:.{digits}f}"


def experiment_id(dataset: str, kind: str) -> str:
    number = FIRST_EXPERIMENT_NUMBER + DATASET_ORDER.index(dataset) * len(STAGES) + STAGES.index(kind)
    return f"EXP-{number:03d}"


def result_filename(dataset: str, kind: str) -> str:
    return f"{experiment_id(dataset, kind)}_{dataset}_{kind}.json"


def higher_is_better(metric: str) -> bool:
    return HIGHER_IS_BETTER[metric]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def recall_at_precision(y_true: Any, scores: Any, target: float) -> dict[str, Any]:
    """Highest recall whose precision is >= ``target`` (threshold included)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0:
        return {"recall": 0.0, "precision": None, "threshold": None, "reached": False}
    precision, recall, thresholds = precision_recall_curve(y, s)
    precision, recall = precision[:-1], recall[:-1]
    mask = precision >= target
    if not mask.any():
        return {"recall": 0.0, "precision": None, "threshold": None, "reached": False}
    index = int(np.argmax(np.where(mask, recall, -1.0)))
    return {
        "recall": float(recall[index]),
        "precision": float(precision[index]),
        "threshold": float(thresholds[index]),
        "reached": True,
    }


def binary_metrics(y_true: Any, scores: Any, *, precision_target: float = 0.9) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    labels = (p >= 0.5).astype(int)
    both = len(np.unique(y)) == 2
    k = int(y.sum())
    top = np.argsort(-p, kind="stable")[:k] if k else np.array([], dtype=int)
    positive_rate = float(y.mean()) if len(y) else 0.0
    return {
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "recall_at_precision_target": recall_at_precision(y, p, precision_target)["recall"] if both else None,
        "precision_target": precision_target,
        "precision_at_k_positives": float(y[top].mean()) if k else None,
        "accuracy": float(accuracy_score(y, labels)),
        "majority_baseline_accuracy": float(max(positive_rate, 1 - positive_rate)),
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels, zero_division=0)),
        "f1": float(f1_score(y, labels, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "positive_rate": positive_rate,
        "threshold": 0.5,
    }


def multiclass_metrics(y_true: Any, proba: Any, n_classes: int, *, detail: bool = False) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(proba, dtype=float), 1e-9, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    predicted = p.argmax(axis=1)
    labels = list(range(n_classes))
    metrics: dict[str, Any] = {
        "macro_f1": float(f1_score(y, predicted, average="macro", labels=labels, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "accuracy": float(accuracy_score(y, predicted)),
        "log_loss": float(log_loss(y, p, labels=labels)),
    }
    if detail:
        per_class = f1_score(y, predicted, average=None, labels=labels, zero_division=0)
        metrics["per_class_f1"] = [float(value) for value in per_class]
        worst = np.argsort(per_class, kind="stable")[:3]
        metrics["worst_3_classes"] = [
            {"class_index": int(index), "f1": float(per_class[index])} for index in worst
        ]
    return metrics


def regression_metrics(y_true: Any, predictions: Any) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(predictions, dtype=float)
    errors = p - y
    nonzero = np.abs(y) > 1e-9
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float(np.mean(np.abs(errors[nonzero]) / np.abs(y[nonzero]))) if nonzero.any() else None,
        "r2": float(r2_score(y, p)) if len(y) > 1 else None,
        "bias": float(np.mean(errors)),
    }


def compute_metrics(
    task_type: str,
    y_true: Any,
    predictions: Any,
    *,
    n_classes: int | None = None,
    detail: bool = False,
    precision_target: float = 0.9,
) -> dict[str, Any]:
    if task_type in {"binary_imbalanced", "text_tabular_binary"}:
        return binary_metrics(y_true, predictions, precision_target=precision_target)
    if task_type == "multiclass":
        return multiclass_metrics(y_true, predictions, int(n_classes or 0), detail=detail)
    if task_type == "timeseries_regression":
        return regression_metrics(y_true, predictions)
    raise ValueError(f"unknown task type: {task_type}")


def primary_metric_fn(task_type: str, metric: str, n_classes: int | None = None) -> Callable[[Any, Any], float]:
    """Fast scorer for bootstrap resampling."""
    if metric == "average_precision":
        return lambda y, p: float(average_precision_score(y, p))
    if metric == "roc_auc":
        return lambda y, p: float(roc_auc_score(y, p))
    if metric == "macro_f1":
        labels = list(range(int(n_classes or 0)))
        return lambda y, p: float(
            f1_score(y, np.asarray(p).argmax(axis=1), average="macro", labels=labels, zero_division=0)
        )
    if metric == "mae":
        return lambda y, p: float(mean_absolute_error(y, p))
    raise ValueError(metric)


def _aggregate(values: Iterable[Any]) -> dict[str, float] | None:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(arr):
        return None
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def bootstrap_ci(
    y_true: Any,
    predictions: Any,
    metric_fn: Callable[[Any, Any], float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    groups: Any = None,
    block_length: int | None = None,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Percentile bootstrap: iid rows, whole groups (cluster), or moving blocks."""
    if _PROFILE["fast"]:
        draws = min(draws, 40)
    y = np.asarray(y_true)
    p = np.asarray(predictions)
    n = len(y)
    rng = np.random.default_rng(seed)
    if groups is not None:
        groups = np.asarray(groups)
        unique = np.unique(groups)
        positions = {g: np.flatnonzero(groups == g) for g in unique}
        method = f"cluster bootstrap over {len(unique)} groups"
    elif block_length:
        block_length = max(1, min(int(block_length), n))
        method = f"moving-block bootstrap (block length {block_length})"
    else:
        method = "iid row bootstrap"
    values: list[float] = []
    for _ in range(draws):
        if groups is not None:
            chosen = rng.choice(unique, size=len(unique), replace=True)
            index = np.concatenate([positions[g] for g in chosen])
        elif block_length:
            starts = rng.integers(0, n - block_length + 1, size=int(math.ceil(n / block_length)))
            index = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]
        else:
            index = rng.integers(0, n, size=n)
        try:
            value = metric_fn(y[index], p[index])
        except ValueError:
            continue
        if value is not None and np.isfinite(value):
            values.append(float(value))
    point = metric_fn(y, p)
    if not values:
        return {"point": point, "low": point, "median": point, "high": point, "draws": 0, "method": method}
    return {
        "point": float(point),
        "low": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "high": float(np.quantile(values, 0.975)),
        "draws": len(values),
        "method": method,
    }


# --------------------------------------------------------------------------
# selection helpers
# --------------------------------------------------------------------------


def adjusted_score(mean: float, std: float, higher: bool) -> float:
    """Mirror of the original rule: mean - 0.25*std (higher is better) or mean + 0.25*std."""
    return mean - ADJUST_STD * std if higher else mean + ADJUST_STD * std


def oriented_gain(new: float, old: float, higher: bool) -> float:
    """Positive when ``new`` is better than ``old``."""
    return new - old if higher else old - new


def _metric_stats(row: dict[str, Any], metric: str) -> tuple[float, float]:
    stats = row["metrics"][metric]
    return float(stats["mean"]), float(stats["std"])


def rank_rows(rows: list[dict[str, Any]], metric: str, higher: bool) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, float]:
        mean, std = _metric_stats(row, metric)
        score = adjusted_score(mean, std, higher)
        return (-score if higher else score, row.get("elapsed_seconds", 0.0))

    return sorted(rows, key=key)


def select_within_tolerance(
    rows: list[dict[str, Any]],
    metric: str,
    higher: bool,
    tolerance: float,
    *,
    relative: bool = False,
) -> dict[str, Any]:
    """Smallest (fewest features, then fastest) row within tolerance of the best mean."""
    means = [_metric_stats(row, metric)[0] for row in rows]
    best = max(means) if higher else min(means)
    slack = abs(best) * tolerance if relative else tolerance
    eligible = [
        row
        for row, mean in zip(rows, means)
        if (mean >= best - slack if higher else mean <= best + slack)
    ]
    return min(eligible, key=lambda row: (row["feature_count_mean"], row.get("elapsed_seconds", 0.0)))


def tuning_decision(
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
    metric: str,
    higher: bool,
    margin: float,
    *,
    relative: bool = False,
) -> dict[str, Any]:
    """Accept the best adjusted candidate only if its mean beats baseline by ``margin``."""
    base_mean = _metric_stats(baseline, metric)[0]
    if not candidates:
        return {"accepted": False, "best_candidate": None, "gain": 0.0, "required_gain": 0.0, "selected": baseline}
    best = rank_rows(candidates, metric, higher)[0]
    gain = oriented_gain(_metric_stats(best, metric)[0], base_mean, higher)
    required = abs(base_mean) * margin if relative else margin
    accepted = gain >= required
    return {
        "accepted": bool(accepted),
        "best_candidate": best.get("config_id"),
        "gain": float(gain),
        "required_gain": float(required),
        "selected": best if accepted else baseline,
    }


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def _have(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


MODEL_FAMILIES = {
    "binary_imbalanced": ("logistic_regression", "extra_trees", "hist_gradient_boosting", "lightgbm", "xgboost"),
    "multiclass": ("logistic_regression", "extra_trees", "hist_gradient_boosting", "lightgbm", "xgboost"),
    "timeseries_regression": ("ridge", "extra_trees", "hist_gradient_boosting", "lightgbm", "xgboost"),
    "text_tabular_binary": ("logistic_regression", "linear_svm", "complement_nb", "lightgbm", "xgboost"),
}


def model_families(task_type: str) -> list[str]:
    available = []
    for family in MODEL_FAMILIES[task_type]:
        if family == "lightgbm" and not _have("lightgbm"):
            continue
        if family == "xgboost" and not _have("xgboost"):
            continue
        available.append(family)
    return available


def _fallback_family(family: str) -> str:
    if family == "lightgbm" and not _have("lightgbm"):
        return "hist_gradient_boosting"
    return family


from sklearn.feature_selection import SelectKBest  # noqa: E402


class _QuietSelectKBest(SelectKBest):
    """SelectKBest that silently keeps every column when k exceeds the column count."""

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "_QuietSelectKBest":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return super().fit(X, y, **kwargs)


def make_model(
    family: str,
    task_type: str,
    params: dict[str, Any] | None = None,
    *,
    sparse_input: bool = False,
) -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler

    regression = task_type == "timeseries_regression"
    steps: list[tuple[str, Any]] = []
    if family in {"logistic_regression", "ridge", "linear_svm"}:
        if sparse_input:
            steps.append(("scale", MaxAbsScaler()))
        else:
            steps += [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]
    if sparse_input and family in {"lightgbm", "xgboost", "extra_trees"}:
        from sklearn.feature_selection import chi2

        # Tree ensembles on 30k-60k sparse TF-IDF columns are slow and mostly split on
        # the strongest terms; keep the top chi2 columns, fitted on the fit fold only.
        steps.append(("select", _QuietSelectKBest(chi2, k=TEXT_TREE_MAX_FEATURES)))
    if family == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        if sparse_input:
            model = LogisticRegression(C=4.0, solver="liblinear", max_iter=3000)
        else:
            model = LogisticRegression(C=1.0, max_iter=3000)
    elif family == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
    elif family == "linear_svm":
        from sklearn.svm import LinearSVC

        model = LinearSVC(C=0.1, max_iter=5000, random_state=RANDOM_STATE)
    elif family == "complement_nb":
        from sklearn.naive_bayes import ComplementNB

        model = ComplementNB(alpha=1.0)
    elif family == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        if not sparse_input:
            steps.append(("impute", SimpleImputer(strategy="median", keep_empty_features=True)))
        cls = ExtraTreesRegressor if regression else ExtraTreesClassifier
        model = cls(n_estimators=300, min_samples_leaf=2, n_jobs=N_JOBS, random_state=RANDOM_STATE)
    elif family == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        cls = HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier
        model = cls(random_state=RANDOM_STATE)
    elif family == "lightgbm":
        import lightgbm as lgb

        cls = lgb.LGBMRegressor if regression else lgb.LGBMClassifier
        model = cls(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=-1,
        )
    elif family == "xgboost":
        import xgboost as xgb

        common = dict(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            tree_method="hist",
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        model = xgb.XGBRegressor(**common) if regression else xgb.XGBClassifier(**common)
    else:
        raise ValueError(f"unknown model family: {family}")
    steps.append(("model", model))
    pipeline = Pipeline(steps)
    if params:
        pipeline.set_params(**params)
    if _PROFILE["fast"]:
        inner = pipeline.named_steps["model"]
        params_now = inner.get_params()
        if "n_jobs" in params_now:
            inner.set_params(n_jobs=1)  # avoid OpenMP oversubscription on shared CI runners
        for name, value in (("n_estimators", 10), ("max_iter", 10)):
            if name in inner.get_params() and family not in {"logistic_regression", "linear_svm", "ridge", "complement_nb"}:
                inner.set_params(**{name: value})
    return pipeline


def predict_scores(model: Any, matrix: Any, task_type: str, n_classes: int | None = None) -> np.ndarray:
    if task_type == "timeseries_regression":
        return np.asarray(model.predict(matrix), dtype=float)
    if task_type == "multiclass":
        proba = np.asarray(model.predict_proba(matrix), dtype=float)
        classes = np.asarray(model.classes_, dtype=int)
        full = np.zeros((proba.shape[0], int(n_classes or proba.shape[1])))
        full[:, classes] = proba
        return full
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix), dtype=float)[:, 1]
    values = np.asarray(model.decision_function(matrix), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def prior_correct(probabilities: np.ndarray, negative_fraction: float) -> np.ndarray:
    """Undo negative subsampling: true odds = fitted odds x kept negative fraction."""
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    return p * negative_fraction / (p * negative_fraction + (1.0 - p))


def _fit_rows(y: np.ndarray, negative_fraction: float | None, seed: int) -> np.ndarray:
    index = np.arange(len(y))
    if not negative_fraction or negative_fraction >= 1:
        return index
    rng = np.random.default_rng(seed)
    positives = index[y == 1]
    negatives = index[y != 1]
    kept = rng.choice(negatives, size=max(1, int(round(len(negatives) * negative_fraction))), replace=False)
    return np.sort(np.concatenate([positives, kept]))


def _importance(model: Any, names: list[str]) -> dict[str, float]:
    inner = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    selector = model.named_steps.get("select") if hasattr(model, "named_steps") else None
    if selector is not None and hasattr(selector, "get_support"):
        mask = selector.get_support()
        if len(mask) == len(names):
            names = [n for n, keep in zip(names, mask) if keep]
    values = None
    if hasattr(inner, "feature_importances_"):
        values = np.asarray(inner.feature_importances_, dtype=float)
    elif hasattr(inner, "coef_"):
        coef = np.asarray(inner.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0) if coef.ndim == 2 else np.abs(coef)
    if values is None or len(values) != len(names):
        return {}
    total = float(np.abs(values).sum())
    if total > 0:
        values = np.abs(values) / total
    return dict(zip(names, values.tolist()))


def optimization_candidates(family: str, task_type: str) -> list[dict[str, Any]]:
    """Small explicit parameter candidates per family (no random search)."""
    regression = task_type == "timeseries_regression"
    sparse_text = task_type == "text_tabular_binary"
    spaces: dict[str, list[dict[str, Any]]] = {
        "logistic_regression": (
            [{"model__C": 1.0}, {"model__C": 12.0}, {"model__C": 4.0, "model__class_weight": "balanced"}]
            if sparse_text
            else [{"model__C": 0.1}, {"model__C": 10.0}, {"model__C": 1.0, "model__class_weight": "balanced"}]
        ),
        "ridge": [{"model__alpha": 0.1}, {"model__alpha": 10.0}, {"model__alpha": 100.0}],
        "linear_svm": [{"model__C": 0.03}, {"model__C": 0.3}, {"model__C": 0.1, "model__class_weight": "balanced"}],
        "complement_nb": [{"model__alpha": 0.1}, {"model__alpha": 0.3}, {"model__alpha": 3.0}],
        "extra_trees": [
            {"model__n_estimators": 500, "model__min_samples_leaf": 1, "model__max_features": "sqrt"},
            {"model__n_estimators": 500, "model__min_samples_leaf": 3, "model__max_features": 0.7},
            {"model__n_estimators": 400, "model__min_samples_leaf": 1, "model__max_features": 1.0 if regression else 0.5},
        ],
        "hist_gradient_boosting": [
            {"model__learning_rate": 0.05, "model__max_iter": 300, "model__max_leaf_nodes": 31},
            {"model__learning_rate": 0.1, "model__max_iter": 200, "model__max_leaf_nodes": 63, "model__l2_regularization": 1.0},
            {"model__learning_rate": 0.03, "model__max_iter": 500, "model__max_leaf_nodes": 15, "model__min_samples_leaf": 10},
        ],
        "lightgbm": (
            [
                {"model__n_estimators": 500, "model__learning_rate": 0.03, "model__num_leaves": 15, "model__min_child_samples": 10},
                {"model__n_estimators": 800, "model__learning_rate": 0.02, "model__num_leaves": 7, "model__min_child_samples": 5, "model__objective": "l1"},
                {"model__n_estimators": 300, "model__learning_rate": 0.05, "model__num_leaves": 31, "model__colsample_bytree": 0.7, "model__min_child_samples": 20},
            ]
            if regression
            else [
                {"model__n_estimators": 600, "model__learning_rate": 0.03, "model__num_leaves": 31, "model__colsample_bytree": 0.8},
                {"model__n_estimators": 400, "model__learning_rate": 0.05, "model__num_leaves": 63, "model__min_child_samples": 10},
                {"model__n_estimators": 500, "model__learning_rate": 0.03, "model__num_leaves": 15, "model__min_child_samples": 50, "model__reg_lambda": 1.0},
            ]
        ),
        "xgboost": [
            {"model__n_estimators": 500, "model__learning_rate": 0.05, "model__max_depth": 4, "model__subsample": 0.8, "model__colsample_bytree": 0.8},
            {"model__n_estimators": 300, "model__learning_rate": 0.1, "model__max_depth": 8, "model__min_child_weight": 1},
            {"model__n_estimators": 600, "model__learning_rate": 0.03, "model__max_depth": 6, "model__reg_lambda": 2.0},
        ],
    }
    candidates = spaces.get(family, [])
    if _PROFILE["fast"]:
        candidates = candidates[:2]
    return [
        {"candidate_id": f"{family}_C{index:02d}", "params": params}
        for index, params in enumerate(candidates, start=1)
    ]


# --------------------------------------------------------------------------
# feature recipes (all fitted on the fit fold only)
# --------------------------------------------------------------------------


def _as_text(series: pd.Series) -> pd.Series:
    """String view with an explicit token for missing values (pandas 3 keeps NaN under astype(str))."""
    return series.astype(object).where(series.notna(), "__missing__").astype(str)


def _identifier_like(series: pd.Series, unique_ratio: float) -> bool:
    """Near-unique keys: non-numeric or integer-valued columns only (continuous floats are naturally unique)."""
    if unique_ratio < 0.98:
        return False
    if not pd.api.types.is_numeric_dtype(series):
        return True
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return bool(len(values) and np.all(np.mod(values, 1.0) == 0))


class _Encoder:
    """Numeric pass-through plus one-hot (fit-fold categories) for non-numeric columns and category codes.

    A numeric column named in ``categorical`` holds category codes (DCLAB-R11): it is one-hot
    encoded like a text column and never passed to the model as a number.
    """

    def fit(self, frame: pd.DataFrame, max_categories: int = 40, categorical: Iterable[str] = ()) -> "_Encoder":
        codes = set(categorical)
        self.numeric = [c for c in frame.columns if c not in codes and pd.api.types.is_numeric_dtype(frame[c])]
        self.categories = {
            c: _as_text(frame[c]).value_counts().index[:max_categories].tolist()
            for c in frame.columns
            if c not in self.numeric
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        parts = {c: pd.to_numeric(frame[c], errors="coerce").astype(float) for c in self.numeric}
        for column, values in self.categories.items():
            text = _as_text(frame[column])
            for value in values:
                parts[f"{column}={value}"] = (text == value).astype(float)
        return pd.DataFrame(parts, index=frame.index)


def past_only_lag_table(dates: Iterable[Any], y: Iterable[float], min_lag: int = 2) -> pd.DataFrame:
    """Lag/rolling features for day d built only from targets at day d-``min_lag`` or earlier.

    The series is re-indexed to a full daily calendar first, so a missing source
    day yields NaN instead of silently shifting by rows.
    """
    stamp = pd.to_datetime(pd.Series(list(dates)).astype(str))
    series = pd.Series(np.asarray(list(y), dtype=float), index=stamp.to_numpy()).sort_index()
    series = series[~series.index.duplicated()]
    full = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="D"))
    base = full.shift(min_lag)
    table = pd.DataFrame(index=full.index)
    table["past_lag3"] = full.shift(max(3, min_lag))
    table["past_lag7"] = full.shift(max(7, min_lag))
    table["past_lag14"] = full.shift(max(14, min_lag))
    table["past_roll7_mean"] = base.rolling(7, min_periods=3).mean()
    table["past_roll7_std"] = base.rolling(7, min_periods=3).std()
    table["past_roll28_mean"] = base.rolling(28, min_periods=7).mean()
    table["past_same_weekday_mean4"] = pd.concat(
        [full.shift(k) for k in (7, 14, 21, 28)], axis=1
    ).mean(axis=1, skipna=True)
    table["past_lag2_over_roll28"] = base / table["past_roll28_mean"]
    table.index = table.index.strftime("%Y-%m-%d")
    return table


def _lag_table(bundle: TaskBundle) -> pd.DataFrame:
    if "lag_table" not in bundle.extras:
        bundle.extras["lag_table"] = past_only_lag_table(
            bundle.X[bundle.spec.time_column],
            bundle.y,
            int(bundle.spec.settings.get("min_lag_days", 2)),
        )
    return bundle.extras["lag_table"]


LETTER_RATIOS = (
    ("aspect", "width", "high", "ratio"),
    ("box_aspect", "x-box", "y-box", "ratio"),
    ("variance_ratio", "x2bar", "y2bar", "ratio"),
    ("edge_ratio", "x-ege", "y-ege", "ratio"),
    ("edge_corr_ratio", "xegvy", "yegvx", "ratio"),
    ("moment_ratio", "x2ybr", "xy2br", "ratio"),
    ("centroid_diff", "x-bar", "y-bar", "diff"),
)


def encoded_codes(spec: DatasetSpec) -> tuple[str, ...]:
    """Category-code columns the recipes one-hot encode. ``settings["one_hot_codes"] = False`` passes
    them as numbers instead, the behaviour before DCLAB-R11 reached the encoder (kept for paired measurement)."""
    return tuple(spec.categorical_columns) if spec.settings.get("one_hot_codes", True) else ()


def _quantity_columns(spec: DatasetSpec, frame: pd.DataFrame) -> list[str]:
    """Numeric inputs that measure an amount: not blocked, keys, text, group/time, or category codes (DCLAB-R11)."""
    excluded = (set(spec.blocked_features) | set(spec.identifier_columns) | set(spec.text_columns)
                | set(spec.categorical_columns) | {spec.group_column, spec.time_column})
    return [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]


def _derive(bundle: TaskBundle, name: str, frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    if name == "log_amount":
        out["log_amount"] = np.log1p(pd.to_numeric(frame["Amount"], errors="coerce").clip(lower=0))
    elif name == "hour_of_day":
        hour = (pd.to_numeric(frame["Time"], errors="coerce") / 3600.0) % 24.0
        out["hour_of_day"] = hour
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    elif name == "letter_ratios":
        for label, a, b, op in LETTER_RATIOS:
            if a in frame and b in frame:
                left = frame[a].astype(float)
                right = frame[b].astype(float)
                out[label] = left / (right + 1.0) if op == "ratio" else left - right
        if {"onpix", "width", "high"} <= set(frame.columns):
            out["ink_density"] = frame["onpix"] / (frame["width"] * frame["high"] + 1.0)
    elif name == "poly2":
        columns = _quantity_columns(bundle.spec, frame)
        values = {}
        for i, a in enumerate(columns):
            for b in columns[i + 1 :]:
                values[f"{a}*{b}"] = frame[a].astype(float) * frame[b].astype(float)
        out = pd.DataFrame(values, index=frame.index)
    elif name == "calendar":
        stamp = pd.to_datetime(frame[bundle.spec.time_column].astype(str))
        day_of_year = stamp.dt.dayofyear.astype(float)
        out["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
        out["day_of_month"] = stamp.dt.day.astype(float)
        out["week_of_year"] = stamp.dt.isocalendar().week.astype(float).to_numpy()
        out["is_weekend"] = (stamp.dt.dayofweek >= 5).astype(float)
    elif name == "log_numeric":
        # Generic single transform for user data: signed log1p of every numeric input that is a quantity.
        for column in _quantity_columns(bundle.spec, frame):
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.nunique() > 2:
                out[f"log_{column}"] = np.sign(values) * np.log1p(values.abs())
    elif name == "past_lags":
        table = _lag_table(bundle)
        keys = frame[bundle.spec.time_column].astype(str)
        joined = table.reindex(keys.to_numpy())
        joined.index = frame.index
        out = joined.astype(float)
    else:
        raise ValueError(f"unknown derivation: {name}")
    return out


class FrameRecipe:
    """Dense tabular recipe: safe base columns (+ derivations) -> one-hot -> optional MI selection."""

    sparse = False

    def __init__(
        self,
        bundle: TaskBundle,
        name: str,
        *,
        include_blocked: Iterable[str] = (),
        derive: tuple[str, ...] = (),
        keep: tuple[str, ...] | None = None,
        select_k: int | None = None,
    ) -> None:
        self.bundle = bundle
        self.name = name
        self.include_blocked = set(include_blocked)
        self.derive = derive
        self.keep = keep
        self.select_k = select_k

    def base_columns(self, frame: pd.DataFrame) -> list[str]:
        spec = self.bundle.spec
        excluded = set(spec.identifier_columns) | set(spec.text_columns)
        if spec.group_column:
            excluded.add(spec.group_column)
        excluded |= set(spec.blocked_features) - self.include_blocked
        columns = [c for c in frame.columns if c not in excluded]
        if self.keep is not None:
            columns = [c for c in columns if c in self.keep or c in self.include_blocked]
        return columns

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        base = self.encoder.transform(frame[self.columns])
        extra = [_derive(self.bundle, name, frame) for name in self.derive]
        return pd.concat([base, *extra], axis=1) if extra else base

    def fit(self, frame: pd.DataFrame, y: pd.Series) -> "FrameRecipe":
        self.columns = self.base_columns(frame)
        self.encoder = _Encoder().fit(frame[self.columns], categorical=encoded_codes(self.bundle.spec))
        matrix = self._matrix(frame)
        self.names = list(matrix.columns)
        if self.select_k and self.select_k < len(self.names):
            from sklearn.feature_selection import mutual_info_classif

            scores = mutual_info_classif(
                matrix.fillna(matrix.median()).to_numpy(), np.asarray(y), random_state=RANDOM_STATE
            )
            order = np.argsort(-scores, kind="stable")[: self.select_k]
            self.names = [self.names[i] for i in sorted(order)]
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        matrix = self._matrix(frame).reindex(columns=self.names)
        return matrix.to_numpy(dtype=np.float32), list(self.names)


class TextRecipe:
    """Sparse text(+tabular) recipe: TF-IDF fitted on the fit fold only."""

    sparse = True

    def __init__(
        self,
        bundle: TaskBundle,
        name: str,
        *,
        include_blocked: Iterable[str] = (),
        word: bool = False,
        char: bool = False,
        tabular: bool = False,
        meta: bool = False,
    ) -> None:
        self.bundle = bundle
        self.name = name
        self.include_blocked = set(include_blocked)
        self.word, self.char, self.tabular, self.meta = word, char, tabular, meta

    def _docs(self, frame: pd.DataFrame) -> list[str]:
        parts = [frame[c].fillna("").astype(str) for c in self.bundle.spec.text_columns if c in frame]
        if not parts:
            return [""] * len(frame)
        joined = parts[0]
        for part in parts[1:]:
            joined = joined + " . " + part
        return joined.tolist()

    def _meta(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for column in self.bundle.spec.text_columns:
            if column not in frame:
                continue
            text = frame[column]
            out[f"{column}:missing"] = text.isna().astype(float)
            out[f"{column}:log_words"] = np.log1p(text.fillna("").astype(str).str.split().str.len())
        return out

    def _tabular_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        parts = []
        if self.tabular:
            parts.append(self.encoder.transform(frame[self.columns]))
        if self.meta:
            parts.append(self._meta(frame))
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)

    def fit(self, frame: pd.DataFrame, y: pd.Series) -> "TextRecipe":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import MinMaxScaler

        docs = self._docs(frame)
        self.vectorizers = []
        if self.word:
            self.vectorizers.append(
                ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True, strip_accents="unicode").fit(docs))
            )
        if self.char:
            self.vectorizers.append(
                ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=30000, sublinear_tf=True).fit(docs))
            )
        spec = self.bundle.spec
        excluded = set(spec.identifier_columns) | set(spec.text_columns)
        if spec.group_column:
            excluded.add(spec.group_column)
        excluded |= set(spec.blocked_features) - self.include_blocked
        self.columns = [c for c in frame.columns if c not in excluded]
        self.encoder = _Encoder().fit(frame[self.columns], categorical=encoded_codes(spec))
        dense = self._tabular_frame(frame)
        self.dense_names = list(dense.columns)
        self.medians = dense.median()
        self.scaler = MinMaxScaler(clip=True).fit(dense.fillna(self.medians).to_numpy()) if self.dense_names else None
        self.names = [f"{prefix}:{token}" for prefix, vec in self.vectorizers for token in vec.get_feature_names_out()]
        self.names += self.dense_names
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[Any, list[str]]:
        docs = self._docs(frame)
        blocks = [vec.transform(docs) for _, vec in self.vectorizers]
        if self.dense_names:
            dense = self._tabular_frame(frame).reindex(columns=self.dense_names).fillna(self.medians)
            blocks.append(sparse.csr_matrix(self.scaler.transform(dense.to_numpy())))
        matrix = sparse.hstack(blocks, format="csr").astype(np.float32)
        return matrix, list(self.names)


RECIPES: dict[str, dict[str, dict[str, Any]]] = {
    "credit_card_fraud": {
        "raw": {"description": "V1-V28 + Amount (absolute Time blocked)", "derive": ()},
        "log_amount": {"description": "raw + log1p(Amount)", "derive": ("log_amount",)},
        "hour_of_day": {"description": "raw + hour-of-day (and sin/cos) derived from Time", "derive": ("hour_of_day",)},
        "log_amount_hour": {"description": "raw + log1p(Amount) + hour-of-day", "derive": ("log_amount", "hour_of_day")},
    },
    "letter_recognition": {
        "raw": {"description": "the 16 shape statistics", "derive": ()},
        "ratios": {"description": "raw + 8 geometric ratios/differences (aspect, ink density, moments)", "derive": ("letter_ratios",)},
        "poly2": {"description": "raw + all 120 pairwise products", "derive": ("poly2",)},
        "selected_mi": {"description": "top-10 raw features by fit-fold mutual information", "derive": (), "select_k": 10},
    },
    "bike_sharing_daily": {
        "raw": {"description": "provided calendar/weather columns + days_since_2011 + cnt_2d_bfr (one-hot categoricals)", "derive": ()},
        "calendar": {"description": "raw + day-of-year sin/cos, day-of-month, ISO week, weekend flag", "derive": ("calendar",)},
        "calendar_lags": {"description": "calendar + past-only lags (3/7/14 days) and rolling means/std ending at d-2", "derive": ("calendar", "past_lags")},
        "lags_compact": {
            "description": "weather + workday/holiday + trend + cnt_2d_bfr + past-only lags (no season/month one-hots)",
            "derive": ("past_lags",),
            "keep": ("temp", "atemp", "hum", "windspeed", "weather", "workday", "holiday", "days_since_2011", "cnt_2d_bfr"),
        },
    },
    "ecommerce_clothing_reviews": {
        "tabular": {"description": "Age + one-hot Division/Department/Class (no text)", "tabular": True},
        "tfidf_word": {"description": "word 1-2gram TF-IDF of Title + Review Text", "word": True},
        "tfidf_word_char": {"description": "word TF-IDF + char_wb 3-5gram TF-IDF", "word": True, "char": True},
        "combined": {"description": "word TF-IDF + tabular + text-missing/length meta", "word": True, "tabular": True, "meta": True},
        "combined_char": {"description": "word + char TF-IDF + tabular + meta", "word": True, "char": True, "tabular": True, "meta": True},
    },
}
LEAKAGE_RECIPE = {
    "credit_card_fraud": "raw",
    "letter_recognition": "raw",
    "bike_sharing_daily": "raw",
    "ecommerce_clothing_reviews": "combined",
}


def make_recipe(bundle: TaskBundle, name: str, include_blocked: Iterable[str] = ()) -> Any:
    config = RECIPES[bundle.spec.key][name]
    if bundle.task_type == "text_tabular_binary":
        return TextRecipe(
            bundle,
            name,
            include_blocked=include_blocked,
            word=config.get("word", False),
            char=config.get("char", False),
            tabular=config.get("tabular", False),
            meta=config.get("meta", False),
        )
    return FrameRecipe(
        bundle,
        name,
        include_blocked=include_blocked,
        derive=tuple(config.get("derive", ())),
        keep=config.get("keep"),
        select_k=config.get("select_k"),
    )


# --------------------------------------------------------------------------
# cross-validation engine
# --------------------------------------------------------------------------


def _n_classes(bundle: TaskBundle) -> int | None:
    if bundle.task_type != "multiclass":
        return None
    return int(len(bundle.class_labels) if bundle.class_labels else bundle.y.nunique())


def _scalar_metric_names(rows: list[dict[str, Any]]) -> list[str]:
    names = []
    for key, value in rows[0].items():
        if key in {"fold", "threshold", "precision_target"}:
            continue
        if isinstance(value, (int, float)) or value is None:
            names.append(key)
    return names


def fit_and_predict(
    bundle: TaskBundle,
    recipe_name: str,
    family: str,
    params: dict[str, Any] | None,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
    *,
    include_blocked: Iterable[str] = (),
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    task = bundle.task_type
    family = _fallback_family(family)
    recipe = make_recipe(bundle, recipe_name, include_blocked)
    recipe.fit(X_fit, y_fit)
    fit_matrix, names = recipe.transform(X_fit)
    eval_matrix, _ = recipe.transform(X_eval)
    fraction = bundle.spec.settings.get("fit_negative_fraction")
    y_values = y_fit.to_numpy()
    rows = _fit_rows(y_values, fraction, seed)
    model = make_model(family, task, params, sparse_input=recipe.sparse)
    started = time.perf_counter()
    model.fit(fit_matrix[rows], y_values[rows])
    fit_seconds = time.perf_counter() - started
    predictions = predict_scores(model, eval_matrix, task, _n_classes(bundle))
    if fraction:
        predictions = prior_correct(predictions, fraction)
    return {
        "predictions": predictions,
        "model": model,
        "names": names,
        "fit_rows": int(len(rows)),
        "fit_seconds": float(fit_seconds),
        "feature_count": int(fit_matrix.shape[1]),
    }


def cross_validate(
    bundle: TaskBundle,
    recipe_name: str,
    family: str,
    params: dict[str, Any] | None = None,
    *,
    include_blocked: Iterable[str] = (),
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    config_id: str | None = None,
    split_label: str | None = None,
) -> dict[str, Any]:
    """Train-fitted recipe + model on training folds only; returns summary and private OOF."""
    X_train, y_train = bundle.X_train, bundle.y_train
    task = bundle.task_type
    n_classes = _n_classes(bundle)
    precision_target = float(bundle.spec.settings.get("precision_target", 0.9))
    splits = splits if splits is not None else bundle.cv_splits
    fold_rows: list[dict[str, Any]] = []
    importances: dict[str, list[float]] = defaultdict(list)
    feature_counts: list[int] = []
    oof_index: list[np.ndarray] = []
    oof_pred: list[np.ndarray] = []
    started = time.perf_counter()
    for fold, (fit_idx, valid_idx) in enumerate(splits, start=1):
        X_fit = X_train.iloc[fit_idx].reset_index(drop=True)
        y_fit = y_train.iloc[fit_idx].reset_index(drop=True)
        X_valid = X_train.iloc[valid_idx].reset_index(drop=True)
        y_valid = y_train.iloc[valid_idx].reset_index(drop=True)
        out = fit_and_predict(
            bundle, recipe_name, family, params, X_fit, y_fit, X_valid,
            include_blocked=include_blocked, seed=RANDOM_STATE + fold,
        )
        metrics = compute_metrics(task, y_valid, out["predictions"], n_classes=n_classes, precision_target=precision_target)
        metrics.update(
            fold=fold,
            fit_rows=out["fit_rows"],
            validation_rows=int(len(valid_idx)),
            fit_seconds=out["fit_seconds"],
        )
        fold_rows.append(metrics)
        feature_counts.append(out["feature_count"])
        for name, value in _importance(out["model"], out["names"]).items():
            importances[name].append(value)
        oof_index.append(np.asarray(valid_idx))
        oof_pred.append(out["predictions"])
    summary = {name: _aggregate(row.get(name) for row in fold_rows) for name in _scalar_metric_names(fold_rows)}
    summary = {k: v for k, v in summary.items() if v is not None}
    folds = len(fold_rows)
    top = sorted(importances, key=lambda n: -sum(importances[n]) / folds)[:10]
    return {
        "config_id": config_id or f"{recipe_name}/{family}",
        "optimization": config_id or f"{recipe_name}/{family}",  # original-schema alias
        "stage": recipe_name,  # original-schema alias for the feature recipe
        "recipe": recipe_name,
        "model": family,
        "model_params": params or {},
        "include_blocked": sorted(include_blocked),
        "cv": split_label or bundle.cv_description,
        "folds": folds,
        "feature_count_mean": float(np.mean(feature_counts)),
        "metrics": summary,
        "fold_metrics": fold_rows,
        "top_features": [
            {"feature": n, "mean_normalized_importance": float(sum(importances[n]) / folds)} for n in top
        ],
        "elapsed_seconds": float(time.perf_counter() - started),
        "_oof_index": np.concatenate(oof_index),
        "_oof_pred": np.concatenate(oof_pred),
    }


def _mean(row: dict[str, Any], metric: str) -> float:
    return float(row["metrics"][metric]["mean"])


def _std(row: dict[str, Any], metric: str) -> float:
    return float(row["metrics"][metric]["std"])


def _signed(value: float, metric: str) -> str:
    return f"{value:+.1f}" if metric == "mae" else f"{value:+.4f}"


def _ms(row: dict[str, Any], metric: str, digits: int = 4) -> str:
    return f"{_f(_mean(row, metric), digits)}±{_f(_std(row, metric), digits)}"


# --------------------------------------------------------------------------
# heuristics for profiling and leakage
# --------------------------------------------------------------------------


def _psi(train: pd.Series, test: pd.Series, bins: int = 10) -> float | None:
    a = pd.to_numeric(train, errors="coerce").dropna()
    b = pd.to_numeric(test, errors="coerce").dropna()
    if len(a) < 20 or len(b) < 20 or a.nunique() < 2:
        return None
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    a_hist = np.clip(np.histogram(a, bins=edges)[0] / len(a), 1e-6, None)
    b_hist = np.clip(np.histogram(b, bins=edges)[0] / len(b), 1e-6, None)
    return float(np.sum((a_hist - b_hist) * np.log(a_hist / b_hist)))


def _role(spec: DatasetSpec, column: str) -> str:
    if column in spec.blocked_features:
        return "blocked"
    if column in spec.identifier_columns:
        return "identifier"
    if column == spec.group_column:
        return "group_key"
    if column in spec.text_columns:
        return "text"
    if column == spec.time_column:
        return "time"
    return "feature"


def _binned(series: pd.Series, bins: int = 20) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.5 and numeric.nunique() > bins:
        ranks = numeric.rank(method="first")
        return pd.qcut(ranks, bins, labels=False, duplicates="drop").fillna(-1).to_numpy()
    return pd.factorize(_as_text(series))[0]


def univariate_signal(series: pd.Series, y: pd.Series, task_type: str) -> dict[str, Any]:
    """Task-appropriate single-feature predictive power (training rows only)."""
    numeric = pd.to_numeric(series, errors="coerce")
    is_numeric = pd.api.types.is_numeric_dtype(series) and numeric.notna().sum() >= 20
    out: dict[str, Any] = {}
    if not is_numeric and series.nunique(dropna=False) > 0.5 * max(len(series), 1):
        # In-sample category statistics would simply memorize a high-cardinality key.
        return {"signal": None, "note": "high-cardinality non-numeric column; in-sample signal not meaningful"}
    if task_type in {"binary_imbalanced", "text_tabular_binary"}:
        if is_numeric and numeric.nunique() > 1:
            raw = roc_auc_score(y, numeric.fillna(numeric.median()))
            out["univariate_auc"] = float(max(raw, 1 - raw))
        elif not is_numeric:
            rates = y.groupby(_as_text(series)).transform("mean")
            raw = roc_auc_score(y, rates)
            out["univariate_auc"] = float(max(raw, 1 - raw))
            out["auc_basis"] = "in-sample category positive rate"
        out["signal"] = out.get("univariate_auc")
    elif task_type == "multiclass":
        codes = _binned(series)
        entropy = mutual_info_score(y, y)
        out["normalized_mutual_information"] = float(mutual_info_score(y, codes) / entropy) if entropy > 0 else 0.0
        out["signal"] = out["normalized_mutual_information"]
    else:
        if is_numeric and numeric.nunique() > 1:
            out["abs_spearman"] = float(abs(numeric.corr(y, method="spearman")))
        codes = _binned(series)
        bin_means = y.groupby(codes).transform("mean")
        out["binned_r2"] = float(r2_score(y, bin_means))
        out["signal"] = out["binned_r2"]
    return out


STRONG_SIGNAL = {"binary_imbalanced": 0.90, "text_tabular_binary": 0.90, "multiclass": 0.50, "timeseries_regression": 0.80}
EXTREME_SIGNAL = {"binary_imbalanced": 0.98, "text_tabular_binary": 0.98, "multiclass": 0.90, "timeseries_regression": 0.98}


def suspicious_features(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    *,
    declared: Iterable[str] = (),
    skip: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Review candidates (never automatic declarations), adapted per task type."""
    declared = set(declared)
    skip = set(skip)
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    findings = []
    numeric_columns = [c for c in X.columns if c not in skip and pd.api.types.is_numeric_dtype(X[c])]
    identity_pairs: dict[str, list[str]] = defaultdict(list)
    if task_type == "timeseries_regression":
        target = y.to_numpy(dtype=float)
        for i, a in enumerate(numeric_columns):
            for b in numeric_columns[i + 1 :]:
                total = X[a].to_numpy(dtype=float) + X[b].to_numpy(dtype=float)
                if np.allclose(total, target, atol=1e-6, equal_nan=False):
                    identity_pairs[a].append(b)
                    identity_pairs[b].append(a)
    for column in X.columns:
        if column in skip:
            continue
        series = X[column]
        unique_ratio = float(series.nunique(dropna=False) / max(len(series), 1))
        signal = univariate_signal(series, y, task_type)
        reasons = []
        if column in declared:
            reasons.append("declared_post_outcome_or_contested")
        exact = False
        numeric = pd.to_numeric(series, errors="coerce")
        if task_type in {"binary_imbalanced", "text_tabular_binary"}:
            if numeric.notna().all() and numeric.nunique() <= 2:
                values = numeric.astype(int).to_numpy()
                exact = bool(np.array_equal(values, y.to_numpy()) or np.array_equal(values, 1 - y.to_numpy()))
        elif task_type == "multiclass":
            if series.nunique() <= 0.5 * len(series):
                purity = y.groupby(_as_text(series)).agg(lambda s: s.value_counts(normalize=True).iloc[0])
                exact = bool((purity >= 1.0).all() and series.nunique() >= y.nunique())
        else:
            if numeric.notna().all():
                exact = bool(np.allclose(numeric.to_numpy(dtype=float), y.to_numpy(dtype=float)))
        if exact:
            reasons.append("exact_target_proxy")
        if column in identity_pairs:
            reasons.append("arithmetic_target_identity")
        if _SUSPICIOUS_NAME.search(str(column)):
            reasons.append("suspicious_name")
        value = signal.get("signal")
        if value is not None and value >= EXTREME_SIGNAL[task_type]:
            reasons.append("extreme_univariate_signal")
        elif value is not None and value >= STRONG_SIGNAL[task_type]:
            reasons.append("strong_univariate_signal")
        if _identifier_like(series, unique_ratio):
            reasons.append("identifier_like")
        if reasons:
            finding = {
                "feature": str(column),
                "reasons": reasons,
                "unique_ratio": unique_ratio,
                "requires_human_decision_time_review": True,
                **{k: v for k, v in signal.items() if k != "signal"},
            }
            if column in identity_pairs:
                finding["sums_to_target_with"] = identity_pairs[column]
            findings.append(finding)
    return sorted(findings, key=lambda f: ("declared_post_outcome_or_contested" not in f["reasons"], f["feature"]))


def univariate_ranking(X: pd.DataFrame, y: pd.Series, task_type: str, skip: Iterable[str] = (), top: int = 8) -> list[dict[str, Any]]:
    skip = set(skip)
    rows = []
    for column in X.columns:
        if column in skip:
            continue
        signal = univariate_signal(X[column].reset_index(drop=True), y.reset_index(drop=True), task_type)
        if signal.get("signal") is not None:
            rows.append({"feature": str(column), **signal})
    return sorted(rows, key=lambda r: -r["signal"])[:top]


def _canary_check(X: pd.DataFrame, y: pd.Series, task_type: str, skip: Iterable[str]) -> dict[str, Any]:
    canary = X.reset_index(drop=True).copy()
    target = y.reset_index(drop=True)
    canary["__canary_target_copy"] = target.to_numpy()
    checks = {}
    if task_type == "timeseries_regression":
        part = np.floor(target.to_numpy(dtype=float) * 0.37)
        canary["__canary_part_a"] = part
        canary["__canary_part_b"] = target.to_numpy(dtype=float) - part
    sample = canary
    if len(sample) > 50000:
        sample = canary.sample(50000, random_state=RANDOM_STATE)
        target = target.loc[sample.index]
    findings = {
        f["feature"]: f["reasons"]
        for f in suspicious_features(
            sample[[c for c in sample.columns if c.startswith("__canary")]], target, task_type
        )
    }
    checks["exact_copy_detected"] = "exact_target_proxy" in findings.get("__canary_target_copy", [])
    if task_type == "timeseries_regression":
        checks["sum_identity_detected"] = "arithmetic_target_identity" in findings.get("__canary_part_a", [])
    checks["passed"] = all(checks.values())
    return checks


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


def _stage_data_understanding(bundle: TaskBundle, eid: str) -> tuple[dict, list, str]:
    spec, task = bundle.spec, bundle.task_type
    X_tr, y_tr, X_te = bundle.X_train, bundle.y_train, bundle.X_test
    profiles = []
    for column in X_tr.columns:
        series = X_tr[column]
        missing = float(series.isna().mean())
        unique_ratio = float(series.nunique(dropna=False) / max(len(series), 1))
        top_frequency = float(series.value_counts(dropna=False, normalize=True).iloc[0]) if len(series) else 0.0
        psi = _psi(series, X_te[column]) if pd.api.types.is_numeric_dtype(series) else None
        risks = []
        if missing > 0.05:
            risks.append("missingness")
        if _identifier_like(series, unique_ratio) and column not in spec.text_columns:
            risks.append("identifier_like")
        if top_frequency > 0.995:
            risks.append("near_constant")
        if psi is not None and psi > 0.20:
            risks.append("train_holdout_feature_shift")
        profiles.append(
            {
                "feature": str(column),
                "role": _role(spec, column),
                "dtype": str(series.dtype),
                "missing_rate": missing,
                "unique_count": int(series.nunique(dropna=False)),
                "unique_ratio": unique_ratio,
                "top_value_frequency": top_frequency,
                "train_vs_holdout_psi": psi,
                "production_risks": risks,
            }
        )
    feature_cols = [c for c in X_tr.columns if c not in spec.identifier_columns and c != spec.group_column]
    evidence: dict[str, Any] = {
        "source_rows": bundle.source_rows,
        "raw_feature_count": int(X_tr.shape[1]),
        "train_rows": int(len(X_tr)),
        "holdout_rows": int(len(X_te)),
        "holdout_split": bundle.holdout_description,
        "cv_protocol": bundle.cv_description,
        "sampling": bundle.sampling,
        "missing_cell_rate_train": float(X_tr.isna().sum().sum() / max(X_tr.size, 1)),
        "duplicate_feature_row_rate_train": float(X_tr[feature_cols].duplicated().mean()),
        "duplicate_feature_rows_train": int(X_tr[feature_cols].duplicated().sum()),
        "feature_profiles": profiles,
        "risk_features": {p["feature"]: p["production_risks"] for p in profiles if p["production_risks"]},
        "column_roles": {p["feature"]: p["role"] for p in profiles},
        "note": "Holdout feature distributions (never labels) are compared via PSI, as in model_building_50_v1.",
    }
    claims: list[dict[str, Any]] = []
    if task in {"binary_imbalanced", "text_tabular_binary"}:
        positives = int(y_tr.sum())
        rate = float(y_tr.mean())
        evidence["target_summary"] = {
            "positive_rate_train": rate,
            "positives_train": positives,
            "negatives_train": int(len(y_tr) - positives),
            "imbalance_ratio_neg_per_pos": float((len(y_tr) - positives) / max(positives, 1)),
            "majority_baseline_accuracy_train": float(max(rate, 1 - rate)),
        }
        target_text = (
            f"a training positive rate of {rate:.4f} ({positives} positives vs {len(y_tr) - positives} negatives)"
        )
    elif task == "multiclass":
        counts = y_tr.value_counts().sort_index()
        p = counts / counts.sum()
        evidence["target_summary"] = {
            "classes": int(len(counts)),
            "class_counts_train": {str(bundle.class_labels[i] if bundle.class_labels else i): int(v) for i, v in counts.items()},
            "smallest_class_count": int(counts.min()),
            "largest_class_count": int(counts.max()),
            "imbalance_ratio_max_min": float(counts.max() / counts.min()),
            "normalized_entropy": float(-(p * np.log(p)).sum() / np.log(len(p))),
            "majority_baseline_accuracy_train": float(p.max()),
            "label_encoding": "PMLB integer targets mapped in sorted order to class_index 0..K-1",
        }
        target_text = (
            f"{len(counts)} classes with {counts.min()}–{counts.max()} training rows each "
            f"(max/min ratio {counts.max() / counts.min():.2f})"
        )
    else:
        evidence["target_summary"] = {
            "mean_train": float(y_tr.mean()),
            "std_train": float(y_tr.std()),
            "min_train": float(y_tr.min()),
            "median_train": float(y_tr.median()),
            "max_train": float(y_tr.max()),
            "coefficient_of_variation_train": float(y_tr.std() / y_tr.mean()),
        }
        target_text = (
            f"a training target mean of {y_tr.mean():.0f} (std {y_tr.std():.0f}, range {y_tr.min():.0f}–{y_tr.max():.0f})"
        )
    claims.append(
        _claim(
            f"{eid}-C1",
            "fact",
            f"{spec.key} has {bundle.source_rows} source rows and {X_tr.shape[1]} raw columns; the locked holdout is "
            f"{bundle.holdout_description} ({len(X_te)} rows), leaving {len(X_tr)} training rows with {target_text}.",
            ["evidence.source_rows", "evidence.raw_feature_count", "evidence.train_rows", "evidence.holdout_rows", "evidence.target_summary"],
        )
    )
    if spec.time_column:
        if spec.time_column == "Time":
            evidence["time_range"] = {
                "train_seconds": [float(X_tr["Time"].min()), float(X_tr["Time"].max())],
                "holdout_seconds": [float(X_te["Time"].min()), float(X_te["Time"].max())],
                "train_hours": float((X_tr["Time"].max() - X_tr["Time"].min()) / 3600),
                "holdout_hours": float((X_te["Time"].max() - X_te["Time"].min()) / 3600),
            }
        else:
            dates = pd.to_datetime(bundle.X[spec.time_column].astype(str))
            full = pd.date_range(dates.min(), dates.max(), freq="D")
            missing_days = sorted(set(full.strftime("%Y-%m-%d")) - set(dates.dt.strftime("%Y-%m-%d")))
            evidence["time_range"] = {
                "train": [str(X_tr[spec.time_column].iloc[0]), str(X_tr[spec.time_column].iloc[-1])],
                "holdout": [str(X_te[spec.time_column].iloc[0]), str(X_te[spec.time_column].iloc[-1])],
                "missing_calendar_days": missing_days,
            }
    if spec.text_columns:
        stats = {}
        for column in spec.text_columns:
            text = X_tr[column]
            words = text.dropna().astype(str).str.split().str.len()
            stats[column] = {
                "missing_rate": float(text.isna().mean()),
                "median_words": float(words.median()),
                "mean_words": float(words.mean()),
                "p95_words": float(words.quantile(0.95)),
                "max_words": int(words.max()),
                "duplicate_nonmissing_rate": float(text.dropna().duplicated().mean()),
            }
        evidence["text_stats"] = stats
    if spec.group_column:
        sizes = X_tr[spec.group_column].value_counts()
        evidence["group_stats"] = {
            "train_groups": int(len(sizes)),
            "holdout_groups": int(bundle.X_test[spec.group_column].nunique()),
            "largest_group_rows": int(sizes.iloc[0]),
            "largest_group_share": float(sizes.iloc[0] / sizes.sum()),
            "top10_group_share": float(sizes.iloc[:10].sum() / sizes.sum()),
            "median_rows_per_group": float(sizes.median()),
            "singleton_groups": int((sizes == 1).sum()),
        }
    risk_items = evidence["risk_features"]
    risk_text = "; ".join(f"{k}: {', '.join(v)}" for k, v in risk_items.items()) or "none"
    claims.append(
        _claim(
            f"{eid}-C2",
            "risk",
            f"{len(risk_items)} of {X_tr.shape[1]} columns triggered missingness, identifier, near-constant, or train/holdout PSI>0.20 review rules ({risk_text}).",
            ["evidence.feature_profiles", "evidence.risk_features"],
            ["PSI on feature distributions is a smoke check; it uses no holdout labels and does not prove production stability."],
        )
    )
    if task == "binary_imbalanced":
        ts = evidence["target_summary"]
        claims.append(
            _claim(
                f"{eid}-C3",
                "risk",
                f"Accuracy is misleading here: predicting 'not fraud' for every training row already scores {ts['majority_baseline_accuracy_train']:.4f} accuracy, "
                f"so ranking metrics (average precision primary) are required. Training covers {evidence['time_range']['train_hours']:.1f} h and the time-ordered holdout the final {evidence['time_range']['holdout_hours']:.1f} h; "
                f"{evidence['duplicate_feature_rows_train']} training rows are exact feature duplicates.",
                ["evidence.target_summary", "evidence.time_range", "evidence.duplicate_feature_rows_train"],
                ["V1-V28 are PCA components released already fitted on the full two-day extract (including the holdout period); this cannot be undone downstream."],
            )
        )
    elif task == "multiclass":
        claims.append(
            _claim(
                f"{eid}-C3",
                "risk",
                f"{evidence['duplicate_feature_rows_train']} training rows ({evidence['duplicate_feature_row_rate_train']:.2%}) repeat an earlier feature vector exactly; "
                "with 16 small-integer features, identical vectors can fall on both sides of any random split and inflate scores.",
                ["evidence.duplicate_feature_rows_train", "evidence.duplicate_feature_row_rate_train"],
            )
        )
    elif task == "timeseries_regression":
        tr = evidence["time_range"]
        claims.append(
            _claim(
                f"{eid}-C3",
                "fact",
                f"Training spans {tr['train'][0]}..{tr['train'][1]} and the time-ordered holdout {tr['holdout'][0]}..{tr['holdout'][1]}; "
                f"missing calendar days: {tr['missing_calendar_days'] or 'none'}. The target's coefficient of variation is {evidence['target_summary']['coefficient_of_variation_train']:.2f} with a strong upward trend, so random splits would mix future demand levels into training.",
                ["evidence.time_range", "evidence.target_summary"],
                ["The source `atemp` column is on an unusual scale (mean ~32) relative to `temp`; its derivation in this redistribution is undocumented."],
            )
        )
    else:
        gs, txt = evidence["group_stats"], evidence["text_stats"]
        claims.append(
            _claim(
                f"{eid}-C3",
                "risk",
                f"Reviews are concentrated by product: the largest of {gs['train_groups']} training Clothing IDs holds {gs['largest_group_share']:.1%} of training rows and the top 10 hold {gs['top10_group_share']:.1%}, "
                f"so random splits would leak product identity; Review Text is missing in {txt['Review Text']['missing_rate']:.1%} and Title in {txt['Title']['missing_rate']:.1%} of training rows (median review {txt['Review Text']['median_words']:.0f} words).",
                ["evidence.group_stats", "evidence.text_stats"],
            )
        )
    summary = (
        f"Profiled {len(X_tr)} training rows × {X_tr.shape[1]} raw columns of {spec.key} (holdout: {bundle.holdout_description}); "
        f"computed target summary, per-column missingness/uniqueness/PSI, duplicates"
        + (", time range" if spec.time_column else "")
        + (", text length stats" if spec.text_columns else "")
        + (", group concentration" if spec.group_column else "")
        + f"; {len(risk_items)} columns flagged for review."
    )
    return evidence, claims, summary


def _stage_leakage_audit(bundle: TaskBundle, eid: str) -> tuple[dict, list, str]:
    spec, task = bundle.spec, bundle.task_type
    metric = spec.primary_metric
    higher = spec.higher_is_better
    X_tr, y_tr = bundle.X_train, bundle.y_train
    declared = [c for c in spec.blocked_features if c in X_tr.columns]
    skip = list(spec.text_columns)
    sample_X, sample_y = X_tr, y_tr
    if len(X_tr) > 60000:
        sample_X = X_tr.sample(60000, random_state=RANDOM_STATE)
        sample_y = y_tr.loc[sample_X.index]
    findings = suspicious_features(sample_X, sample_y, task, declared=declared, skip=skip)
    canary = _canary_check(sample_X[[c for c in sample_X.columns if c not in skip]].iloc[:, :3], sample_y, task, skip)
    ranking = univariate_ranking(sample_X, sample_y, task, skip=skip)
    audit_model = _fallback_family("lightgbm") if task != "text_tabular_binary" else "logistic_regression"
    recipe = LEAKAGE_RECIPE[spec.key]
    safe = cross_validate(bundle, recipe, audit_model, config_id="safe")
    comparison: dict[str, Any] = {"model": audit_model, "recipe": recipe, "metric": metric, "safe": safe}
    lift = 0.0
    if declared:
        unsafe = cross_validate(bundle, recipe, audit_model, include_blocked=declared, config_id="unsafe_all_blocked")
        lift = oriented_gain(_mean(unsafe, metric), _mean(safe, metric), higher)
        comparison["unsafe"] = unsafe
        comparison["apparent_lift"] = lift
        if len(declared) > 1:
            comparison["per_column"] = {}
            for column in declared:
                single = cross_validate(bundle, recipe, audit_model, include_blocked=[column], config_id=f"unsafe_{column}")
                comparison["per_column"][column] = {
                    "metric_mean": _mean(single, metric),
                    "apparent_lift": oriented_gain(_mean(single, metric), _mean(safe, metric), higher),
                }
    non_declared = [f for f in findings if "declared_post_outcome_or_contested" not in f["reasons"]]
    evidence: dict[str, Any] = {
        "declared_leakage_features": declared,
        "policy_rationale": " ".join(f"`{c}`: {r}" for c, r in spec.blocked_features.items()) or "No column is blocked: " + spec.decision_time_contract,
        "identifier_exclusions": dict(spec.identifier_columns),
        "decision_time_contract": spec.decision_time_contract,
        "heuristic_review_candidates": findings,
        "heuristic_rules": {
            "strong_univariate_signal": STRONG_SIGNAL[task],
            "extreme_univariate_signal": EXTREME_SIGNAL[task],
            "signal_measure": {
                "binary_imbalanced": "max(AUC, 1-AUC) of the raw column",
                "text_tabular_binary": "max(AUC, 1-AUC); categoricals via in-sample category positive rate",
                "multiclass": "mutual information of the (20-quantile-binned) column / H(y)",
                "timeseries_regression": "R² of 20-quantile-bin target means (|Spearman| also reported); plus exact pairwise column-sum == target identity",
            }[task],
            "identifier_like": "unique_ratio >= 0.98 and non-numeric or integer-valued",
            "rows_scanned": int(len(sample_X)),
        },
        "top_univariate_signals": ranking,
        "detector_canary": canary,
        "detector_canary_passed": bool(canary["passed"]),
        "safe_vs_unsafe_training_cv": comparison,
        "apparent_lift": float(lift),
        "apparent_lift_metric": metric,
        "cv_protocol": bundle.cv_description,
        "warning": "Name, univariate-signal, identity, and uniqueness rules propose review; only decision-time semantics confirm leakage.",
    }
    claims = [
        _claim(
            f"{eid}-C1",
            "fact",
            f"The {task} leakage detector caught its synthetic canaries: {canary}.",
            ["evidence.detector_canary", "evidence.detector_canary_passed"],
        )
    ]
    if declared:
        unsafe = comparison["unsafe"]
        per_col = comparison.get("per_column", {})
        identities = sorted({f["feature"] for f in findings if "arithmetic_target_identity" in f["reasons"]})
        identity_text = (
            f" The identity rule independently found that {' + '.join(identities)} == {spec.target} exactly on every training row."
            if identities
            else ""
        )
        per_text = (
            " Per column: " + ", ".join(f"{c} {_signed(v['apparent_lift'], metric)}" for c, v in per_col.items()) + "."
            if per_col
            else ""
        )
        claims.append(
            _claim(
                f"{eid}-C2",
                "decision",
                f"Declared decision-time exclusions for {spec.key}: {declared}. Including them moves training-CV {METRIC_LABEL[metric]} "
                f"({audit_model}, {recipe} recipe) from {_ms(safe, metric)} to {_ms(unsafe, metric)}, "
                + (
                    f"an apparent lift of {_signed(lift, metric)} that would not exist in production."
                    if lift > 0
                    else f"i.e. no apparent lift ({_signed(lift, metric)}); the exclusion rests on decision-time semantics, not on CV inflation."
                )
                + f"{per_text}{identity_text}",
                ["evidence.declared_leakage_features", "evidence.policy_rationale", "evidence.safe_vs_unsafe_training_cv", "evidence.apparent_lift"],
                [evidence["warning"]],
            )
        )
    else:
        claims.append(
            _claim(
                f"{eid}-C2",
                "decision",
                f"No column of {spec.key} is blocked under the decision-time contract; the safe {audit_model} {recipe} baseline scores {_ms(safe, metric)} training-CV {METRIC_LABEL[metric]}, apparent lift 0 by construction.",
                ["evidence.declared_leakage_features", "evidence.policy_rationale", "evidence.safe_vs_unsafe_training_cv", "evidence.apparent_lift"],
                [evidence["warning"]],
            )
        )
    flagged = ", ".join(f"{f['feature']} ({'/'.join(f['reasons'])})" for f in non_declared) or "none"
    claims.append(
        _claim(
            f"{eid}-C3",
            "risk",
            f"Heuristics flagged {len(non_declared)} non-declared column(s) for human review: {flagged}. Strongest single-feature signal: "
            + (f"{ranking[0]['feature']} ({ranking[0]['signal']:.3f})." if ranking else "n/a."),
            ["evidence.heuristic_review_candidates", "evidence.top_univariate_signals", "evidence.heuristic_rules"],
            ["Strong signal alone is not leakage; e.g. legitimate lags or text sentiment are expected to be predictive."],
        )
    )
    summary_parts = [
        f"Scanned {len(sample_X)} training rows with {task} heuristics (univariate signal, exact proxy/identity, name, uniqueness) and synthetic canaries;"
    ]
    if declared:
        summary_parts.append(
            f"compared {audit_model} on the {recipe} recipe with vs without blocked {declared} on {bundle.cv_description}: "
            f"{METRIC_LABEL[metric]} {_ms(safe, metric)} safe vs {_ms(comparison['unsafe'], metric)} unsafe (apparent lift {_signed(lift, metric)})."
        )
    else:
        summary_parts.append(f"no blocked columns; safe {audit_model} {recipe} baseline {METRIC_LABEL[metric]} {_ms(safe, metric)}.")

    if task == "timeseries_regression":
        from sklearn.model_selection import KFold

        random_splits = list(KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE).split(np.arange(len(X_tr))))
        random_cv = cross_validate(bundle, recipe, audit_model, splits=random_splits, config_id="safe_random_kfold", split_label="KFold(3, shuffle) — ignores time order")
        gap = oriented_gain(_mean(random_cv, metric), _mean(safe, metric), higher)
        evidence["random_vs_time_cv"] = {
            "time_ordered": {"metric_mean": _mean(safe, metric), "metric_std": _std(safe, metric), "protocol": bundle.cv_description},
            "random_kfold": random_cv,
            "optimism_gap": gap,
            "optimism_ratio": _mean(safe, metric) / _mean(random_cv, metric),
        }
        claims.append(
            _claim(
                f"{eid}-C4",
                "risk",
                f"Random KFold(3) reports MAE {_ms(random_cv, metric, 1)} for the same safe {audit_model} recipe versus {_ms(safe, metric, 1)} with expanding-window time-ordered CV: "
                f"random splits look {gap:.1f} MAE better ({evidence['random_vs_time_cv']['optimism_ratio']:.2f}× ratio) because they interpolate between neighbouring days and see future demand levels.",
                ["evidence.random_vs_time_cv"],
            )
        )
        summary_parts.append(f"Also compared random KFold(3) vs time-ordered CV for the safe recipe (MAE {_f(_mean(random_cv, metric), 1)} vs {_f(_mean(safe, metric), 1)}).")
    if task == "multiclass":
        features = X_tr.copy()
        keys = pd.util.hash_pandas_object(features, index=False).to_numpy()
        hits = []
        oof_order = safe["_oof_index"]
        oof_pred = safe["_oof_pred"].argmax(axis=1)
        position = {idx: i for i, idx in enumerate(oof_order)}
        for fit_idx, valid_idx in bundle.cv_splits:
            seen = set(keys[fit_idx].tolist())
            for idx in valid_idx:
                hits.append((idx, keys[idx] in seen))
        dup_mask = np.array([h for _, h in hits])
        idxs = np.array([i for i, _ in hits])
        correct = np.array([oof_pred[position[i]] == y_tr.iloc[i] for i in idxs])
        holdout_keys = pd.util.hash_pandas_object(bundle.X_test, index=False).to_numpy()
        train_keys = set(keys.tolist())
        evidence["duplicate_contamination"] = {
            "validation_rows_with_exact_duplicate_in_fit_fold": int(dup_mask.sum()),
            "validation_rows_total": int(len(dup_mask)),
            "accuracy_on_duplicated_rows": float(correct[dup_mask].mean()) if dup_mask.any() else None,
            "accuracy_on_unique_rows": float(correct[~dup_mask].mean()) if (~dup_mask).any() else None,
            "holdout_rows_whose_features_appear_in_training": int(sum(k in train_keys for k in holdout_keys)),
            "holdout_rows": int(len(holdout_keys)),
            "note": "Feature-vector overlap only; holdout labels are not read.",
        }
        dc = evidence["duplicate_contamination"]
        claims.append(
            _claim(
                f"{eid}-C4",
                "risk",
                f"{dc['validation_rows_with_exact_duplicate_in_fit_fold']} of {dc['validation_rows_total']} CV validation rows have an identical feature vector in their fit fold; "
                f"OOF accuracy is {_f(dc['accuracy_on_duplicated_rows'])} on those vs {_f(dc['accuracy_on_unique_rows'])} on unique rows, and {dc['holdout_rows_whose_features_appear_in_training']} of {dc['holdout_rows']} holdout rows repeat a training feature vector, so scores are mildly optimistic for genuinely new glyphs.",
                ["evidence.duplicate_contamination"],
                ["Identical integer summaries may come from different source images; this is contamination risk, not proven label leakage."],
            )
        )
        summary_parts.append("Measured exact duplicate feature-vector contamination across CV folds and train/holdout.")
    if task == "binary_imbalanced":
        claims.append(
            _claim(
                f"{eid}-C4",
                "risk",
                "V1-V28 were produced by a PCA the data publisher fitted on all 284,807 transactions, including the holdout period; this global fit is an irreducible look-ahead that no downstream split can remove.",
                ["evidence.policy_rationale", "evidence.decision_time_contract"],
                ["The size of this effect cannot be measured from the released data."],
            )
        )
    if task == "text_tabular_binary":
        tab_safe = cross_validate(bundle, "tabular", "logistic_regression", config_id="tabular_safe")
        tab_unsafe = cross_validate(bundle, "tabular", "logistic_regression", include_blocked=declared, config_id="tabular_unsafe")
        tab_lift = oriented_gain(_mean(tab_unsafe, metric), _mean(tab_safe, metric), higher)
        evidence["tabular_only_safe_vs_unsafe"] = {"safe": tab_safe, "unsafe": tab_unsafe, "apparent_lift": tab_lift}
        claims.append(
            _claim(
                f"{eid}-C5",
                "risk",
                f"Without text the leak dominates: a tabular-only logistic model goes from ROC-AUC {_ms(tab_safe, metric)} (safe) to {_ms(tab_unsafe, metric)} with the blocked columns "
                f"(apparent lift {_signed(tab_lift, metric)}); with text the lift shrinks to {_signed(lift, metric)} because the review text already expresses the same sentiment.",
                ["evidence.tabular_only_safe_vs_unsafe", "evidence.safe_vs_unsafe_training_cv"],
            )
        )
        summary_parts.append(
            f"Also tabular-only logistic safe vs unsafe: ROC-AUC {_f(_mean(tab_safe, metric))} vs {_f(_mean(tab_unsafe, metric))}."
        )
        rating = X_tr["Rating"]
        crosstab = pd.crosstab(rating, y_tr)
        rate_by_rating = (y_tr.groupby(rating).mean()).to_dict()
        evidence["rating_vs_target_train"] = {
            "recommend_rate_by_rating": {str(k): float(v) for k, v in rate_by_rating.items()},
            "rows_by_rating": {str(k): int(v) for k, v in crosstab.sum(axis=1).items()},
        }
        claims.append(
            _claim(
                f"{eid}-C4",
                "fact",
                "In training rows the recommendation rate is "
                + ", ".join(f"{float(v):.1%} at Rating {k}" for k, v in rate_by_rating.items())
                + ": the star rating almost restates the label, so any model given Rating mostly learns that rule.",
                ["evidence.rating_vs_target_train"],
            )
        )
    return evidence, claims, " ".join(summary_parts)


def _settings(bundle: TaskBundle, key: str, default: Any) -> Any:
    return bundle.spec.settings.get(key, default)


def _stage_feature_engineering(bundle: TaskBundle, eid: str) -> tuple[dict, list, str]:
    spec = bundle.spec
    metric, higher = spec.primary_metric, spec.higher_is_better
    model = _fallback_family(_settings(bundle, "fe_stage_model", "lightgbm"))
    tolerance = float(_settings(bundle, "fe_tolerance", 0.002))
    relative = bool(_settings(bundle, "fe_tolerance_relative", False))
    rows = []
    for name, config in RECIPES[spec.key].items():
        row = cross_validate(bundle, name, model, config_id=name)
        row["description"] = config["description"]
        rows.append(row)
    selected = select_within_tolerance(rows, metric, higher, tolerance, relative=relative)
    best = sorted(rows, key=lambda r: -_mean(r, metric) if higher else _mean(r, metric))[0]
    first = rows[0]
    rule = (
        f"Choose the recipe with the fewest features (then fastest) whose mean training-CV {METRIC_LABEL[metric]} is within "
        + (f"{tolerance:.0%} (relative)" if relative else f"{tolerance}")
        + f" of the best recipe; all recipes use {model} with default parameters on identical folds ({bundle.cv_description})."
    )
    evidence: dict[str, Any] = {
        "stage_results": rows,
        "stage_model": model,
        "selected_recipe": selected["recipe"],
        "selected_stage": selected["recipe"],
        "selected_recipe_description": selected["description"],
        "selection_rule": rule,
        "best_recipe": best["recipe"],
        "selected_metric_mean": _mean(selected, metric),
        "best_metric_mean": _mean(best, metric),
        "reference_recipe": first["recipe"],
        "reference_metric_mean": _mean(first, metric),
        "selected_vs_reference_gain": oriented_gain(_mean(selected, metric), _mean(first, metric), higher),
        "cv_protocol": bundle.cv_description,
    }
    if spec.key == "bike_sharing_daily":
        evidence["lag_construction"] = (
            f"Lag/rolling features come from a daily-reindexed target series shifted by >= {_settings(bundle, 'min_lag_days', 2)} days "
            "(value at d-2 or earlier); the missing day 2011-03-10 yields NaN rather than a row shift. Holdout rows may use counts "
            "from earlier holdout days, which are observed by forecast time under the 2-day-horizon contract."
        )
    table = "; ".join(f"{r['recipe']} ({r['feature_count_mean']:.0f} feats, {_ms(r, metric)})" for r in rows)
    claims = [
        _claim(
            f"{eid}-C1",
            "recommendation",
            f"Use the `{selected['recipe']}` recipe ({selected['description']}) for {spec.key}: {_ms(selected, metric)} training-CV {METRIC_LABEL[metric]} with {selected['feature_count_mean']:.0f} features; "
            f"the best recipe was `{best['recipe']}` at {_f(_mean(best, metric))}.",
            ["evidence.stage_results", "evidence.selection_rule", "evidence.selected_recipe"],
            ["Recipes were compared with a single default-parameter model; a different family could rank them differently."],
        ),
        _claim(
            f"{eid}-C2",
            "fact",
            f"Recipe comparison ({model}, {bundle.cv_description}): {table}.",
            ["evidence.stage_results"],
        ),
    ]
    summary = f"{len(rows)} recipes compared with {model} on {bundle.cv_description}: {table}. Selected: {selected['recipe']}."
    return evidence, claims, summary


def _read_dependency(results_dir: Path, dataset: str, kind: str) -> dict[str, Any]:
    path = results_dir / result_filename(dataset, kind)
    if not path.is_file():
        raise RuntimeError(f"{kind} result for {dataset} is missing ({path}); run that stage first")
    payload = json.loads(path.read_text())
    if payload.get("status") != "completed":
        raise RuntimeError(f"{path} is not completed")
    return payload


def _stage_model_selection(bundle: TaskBundle, eid: str, results_dir: Path) -> tuple[dict, list, str]:
    spec = bundle.spec
    metric, higher = spec.primary_metric, spec.higher_is_better
    recipe = _read_dependency(results_dir, spec.key, "feature_engineering")["evidence"]["selected_recipe"]
    rows = [cross_validate(bundle, recipe, family, config_id=family) for family in model_families(bundle.task_type)]
    for row in rows:
        row["adjusted_score"] = adjusted_score(_mean(row, metric), _std(row, metric), higher)
    ranked = rank_rows(rows, metric, higher)
    selected = ranked[0]
    sign = "-" if higher else "+"
    evidence = {
        "feature_recipe": recipe,
        "model_results": rows,
        "ranking": [
            {"model": r["model"], "mean": _mean(r, metric), "std": _std(r, metric), "adjusted_score": r["adjusted_score"], "elapsed_seconds": r["elapsed_seconds"]}
            for r in ranked
        ],
        "selected_model": selected["model"],
        "selection_rule": f"Rank by mean training-CV {METRIC_LABEL[metric]} {sign} 0.25×fold std ({'higher' if higher else 'lower'} is better), then runtime; identical folds ({bundle.cv_description}); default parameters.",
        "selected_metric_mean": _mean(selected, metric),
        "selected_metric_std": _std(selected, metric),
        "cv_protocol": bundle.cv_description,
    }
    if bundle.task_type == "text_tabular_binary":
        evidence["sparse_tree_preselection"] = (
            f"lightgbm/xgboost receive the top {TEXT_TREE_MAX_FEATURES} chi2-ranked columns (SelectKBest fitted on each fit fold); "
            "linear models and naive Bayes receive the full sparse matrix."
        )
    if bundle.spec.settings.get("fit_negative_fraction"):
        evidence["fit_negative_fraction"] = bundle.spec.settings["fit_negative_fraction"]
    table = "; ".join(f"{r['model']} {_ms(r, metric)} ({r['elapsed_seconds']:.1f}s)" for r in ranked)
    runner_up = ranked[1] if len(ranked) > 1 else selected
    claims = [
        _claim(
            f"{eid}-C1",
            "recommendation",
            f"{selected['model']} is the training-CV model candidate for {spec.key} on the `{recipe}` recipe ({_ms(selected, metric)} {METRIC_LABEL[metric]}, adjusted {selected['adjusted_score']:.4f}); "
            f"runner-up {runner_up['model']} at {_ms(runner_up, metric)}.",
            ["evidence.model_results", "evidence.ranking", "evidence.selection_rule"],
            ["A cross-validation leader is not a universal algorithm winner and has not yet seen the locked holdout."],
        ),
        _claim(
            f"{eid}-C2",
            "fact",
            f"Model screen ranking (mean±std, runtime for 3 folds): {table}.",
            ["evidence.ranking"],
        ),
    ]
    summary = f"{len(rows)} model families screened on the `{recipe}` recipe with identical {bundle.cv_description}, ranked by mean {sign} 0.25×std {METRIC_LABEL[metric]}: {table}. Selected: {selected['model']}."
    return evidence, claims, summary


def _baselines(bundle: TaskBundle, selected_recipe: str, predictions: np.ndarray, metric_fn: Callable) -> dict[str, Any]:
    spec, task = bundle.spec, bundle.task_type
    y_test = bundle.y_test
    out: dict[str, Any] = {}
    if task == "binary_imbalanced":
        rate = float(y_test.mean())
        out["majority_class"] = {"accuracy": float(max(rate, 1 - rate)), "average_precision_random_ranking": rate}
    elif task == "multiclass":
        majority = int(bundle.y_train.value_counts().idxmax())
        n_classes = _n_classes(bundle)
        constant = np.zeros((len(y_test), n_classes))
        constant[:, majority] = 1.0
        prior = bundle.y_train.value_counts(normalize=True).reindex(range(n_classes), fill_value=0).to_numpy()
        out["majority_class"] = {
            "macro_f1": float(f1_score(y_test, np.full(len(y_test), majority), average="macro", labels=list(range(n_classes)), zero_division=0)),
            "accuracy": float((y_test == majority).mean()),
        }
        out["training_prior_log_loss"] = float(log_loss(y_test, np.tile(prior, (len(y_test), 1)), labels=list(range(n_classes))))
    elif task == "timeseries_regression":
        lag2 = bundle.X_test["cnt_2d_bfr"].to_numpy(dtype=float)
        table = _lag_table(bundle).reindex(bundle.X_test[spec.time_column].astype(str).to_numpy())
        lag7 = table["past_lag7"].to_numpy()
        lag7 = np.where(np.isfinite(lag7), lag7, lag2)
        mean = np.full(len(y_test), float(bundle.y_train.mean()))
        block = int(_settings(bundle, "block_length", 7))
        for name, values in (("naive_lag2_cnt_2d_bfr", lag2), ("seasonal_naive_lag7", lag7), ("training_mean", mean)):
            metrics = regression_metrics(y_test, values)
            delta = bootstrap_ci(
                y_test.to_numpy(),
                np.column_stack([predictions, values]),
                lambda y, p: float(np.mean(np.abs(p[:, 0] - y)) - np.mean(np.abs(p[:, 1] - y))),
                block_length=block,
                seed=RANDOM_STATE + 1,
            )
            out[name] = {**metrics, "model_minus_baseline_mae_ci": delta}
    else:
        rate = float(y_test.mean())
        out["majority_class"] = {"roc_auc": 0.5, "average_precision_random_ranking": rate}
        reference = fit_and_predict(bundle, "tabular", "logistic_regression", None, bundle.X_train, bundle.y_train, bundle.X_test)
        ref_pred = reference["predictions"]
        delta = bootstrap_ci(
            y_test.to_numpy(),
            np.column_stack([predictions, ref_pred]),
            lambda y, p: float(roc_auc_score(y, p[:, 0]) - roc_auc_score(y, p[:, 1])),
            groups=bundle.groups("test"),
            seed=RANDOM_STATE + 1,
        )
        out["tabular_only_logistic_reference"] = {
            **binary_metrics(y_test, ref_pred),
            "model_minus_reference_auc_ci": delta,
            "note": "Pre-declared reference (tabular recipe, default logistic regression); not used for any selection.",
        }
    return out


def _stage_optimization_reliability(bundle: TaskBundle, eid: str, results_dir: Path) -> tuple[dict, list, str, dict]:
    spec, task = bundle.spec, bundle.task_type
    metric, higher = spec.primary_metric, spec.higher_is_better
    n_classes = _n_classes(bundle)
    recipe = _read_dependency(results_dir, spec.key, "feature_engineering")["evidence"]["selected_recipe"]
    family = _read_dependency(results_dir, spec.key, "model_selection")["evidence"]["selected_model"]
    baseline = cross_validate(bundle, recipe, family, config_id=f"{family}_baseline")
    candidates = optimization_candidates(family, task)
    tuned = [cross_validate(bundle, recipe, family, c["params"], config_id=c["candidate_id"]) for c in candidates]
    margin = float(_settings(bundle, "tuning_margin", 0.001))
    relative = bool(_settings(bundle, "tuning_margin_relative", False))
    decision = tuning_decision(baseline, tuned, metric, higher, margin, relative=relative)
    selected = decision.pop("selected")
    decision["selected_config"] = selected["config_id"]

    started = time.perf_counter()
    final = fit_and_predict(bundle, recipe, family, selected["model_params"] or None, bundle.X_train, bundle.y_train, bundle.X_test)
    final_seconds = time.perf_counter() - started
    predictions = final["predictions"]
    y_test = bundle.y_test
    holdout_metrics = compute_metrics(
        task, y_test, predictions, n_classes=n_classes, detail=True,
        precision_target=float(_settings(bundle, "precision_target", 0.9)),
    )
    holdout_metrics["fit_seconds"] = float(final_seconds)
    if task == "multiclass" and bundle.class_labels:
        for item in holdout_metrics["worst_3_classes"]:
            item["source_label"] = bundle.class_labels[item["class_index"]]
    metric_fn = primary_metric_fn(task, metric, n_classes)
    ci = bootstrap_ci(
        y_test.to_numpy(),
        predictions,
        metric_fn,
        groups=bundle.groups("test") if spec.split_strategy == "group" else None,
        block_length=int(_settings(bundle, "block_length", 0)) or None,
    )
    baselines = _baselines(bundle, recipe, predictions, metric_fn)
    importance = _importance(final["model"], final["names"])
    top_features = [
        {"feature": k, "normalized_importance": v}
        for k, v in sorted(importance.items(), key=lambda kv: -kv[1])[:20]
    ]
    cv_mean = _mean(selected, metric)
    holdout_value = holdout_metrics[metric]
    evidence: dict[str, Any] = {
        "feature_recipe": recipe,
        "model": family,
        "configuration_results": [baseline, *tuned],
        "optimization_search_space": candidates,
        "tuning_decision": decision,
        "selected_optimization": selected["config_id"],
        "selected_model_params": selected["model_params"],
        "selection_rule": (
            f"Evaluate {len(candidates)} explicit parameter candidates on training CV ({bundle.cv_description}); take the best by mean "
            f"{'-' if higher else '+'} 0.25×std and accept it only if its mean {METRIC_LABEL[metric]} beats the default by at least "
            + (f"{margin:.0%} (relative)" if relative else f"{margin}")
            + ", otherwise keep the default configuration."
        ),
        "holdout_consumed": True,
        "holdout_split": bundle.holdout_description,
        "holdout_rows": int(len(y_test)),
        "train_rows": int(len(bundle.y_train)),
        "holdout_metrics": holdout_metrics,
        "holdout_primary_metric_ci": ci,
        "cv_selected_metric_mean": cv_mean,
        "cv_to_holdout_gap": oriented_gain(holdout_value, cv_mean, higher),
        "baselines": baselines,
        "top_final_features": top_features,
        "sampling": bundle.sampling,
        "production_feature_rule": "Promote a feature only if it is decision-time available, stable across CV folds, obtainable in production, and monitored for missingness/drift — never from importance alone.",
    }
    label = METRIC_LABEL[metric]
    claims = [
        _claim(
            f"{eid}-C1",
            "fact",
            f"The preselected {family}/{selected['config_id']} on the `{recipe}` recipe scored {label} {_f(holdout_value)} on the once-consumed holdout "
            f"({bundle.holdout_description}; {ci['method']} 95% interval {_f(ci['low'])}–{_f(ci['high'])}); training-CV mean was {_f(cv_mean)}.",
            ["evidence.holdout_metrics", "evidence.holdout_primary_metric_ci", "evidence.holdout_consumed", "evidence.cv_selected_metric_mean"],
            [
                "CV models are fitted on part of the training rows while the final model uses all of them, so the holdout can legitimately beat the CV mean; a single holdout is one draw."
            ],
        ),
        _claim(
            f"{eid}-C2",
            "decision",
            f"Tuning {'accepted' if decision['accepted'] else 'rejected'}: best candidate {decision['best_candidate']} changed mean training-CV {label} by {decision['gain']:+.4f} "
            f"(required ≥ {decision['required_gain']:.4f}), so {selected['config_id']} was used.",
            ["evidence.tuning_decision", "evidence.configuration_results", "evidence.selection_rule"],
            ["Only a handful of explicit candidates were tried; this is a sanity check, not an exhaustive search."],
        ),
    ]
    hm = holdout_metrics
    if task == "binary_imbalanced":
        target = float(_settings(bundle, "precision_target", 0.9))
        oof_y = bundle.y_train.to_numpy()[selected["_oof_index"]]
        oof_p = selected["_oof_pred"]
        cv_threshold = recall_at_precision(oof_y, oof_p, target)
        threshold_rows = []
        for t in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
            flags = predictions >= t
            tp = int((flags & (y_test.to_numpy() == 1)).sum())
            threshold_rows.append(
                {
                    "threshold": t,
                    "alerts": int(flags.sum()),
                    "true_positives": tp,
                    "precision": float(tp / flags.sum()) if flags.sum() else None,
                    "recall": float(tp / max(int(y_test.sum()), 1)),
                }
            )
        applied = None
        if cv_threshold["reached"]:
            flags = predictions >= cv_threshold["threshold"]
            tp = int((flags & (y_test.to_numpy() == 1)).sum())
            applied = {
                "threshold": cv_threshold["threshold"],
                "alerts": int(flags.sum()),
                "true_positives": tp,
                "precision": float(tp / flags.sum()) if flags.sum() else None,
                "recall": float(tp / max(int(y_test.sum()), 1)),
            }
        order = np.argsort(-predictions, kind="stable")
        top_k = {
            str(k): float(y_test.to_numpy()[order[:k]].mean()) for k in (25, 50, 100, 200) if k <= len(order)
        }
        oracle = recall_at_precision(y_test.to_numpy(), predictions, target)
        evidence["threshold_analysis"] = {
            "probability_scale": "prior-corrected for fit-time negative subsampling (true base rate)",
            "precision_target": target,
            "threshold_from_training_oof": cv_threshold,
            "holdout_at_training_threshold": applied,
            "holdout_threshold_grid": threshold_rows,
            "holdout_precision_at_top_k": top_k,
            "holdout_oracle_threshold_for_target": {**oracle, "note": "Diagnostic only: chosen on holdout labels, not usable for deployment."},
            "holdout_positives": int(y_test.sum()),
        }
        claims.append(
            _claim(
                f"{eid}-C3",
                "fact",
                f"Accuracy is uninformative on the holdout: the model's {hm['accuracy']:.4f} at threshold 0.5 compares with {baselines['majority_class']['accuracy']:.4f} for always predicting 'not fraud', "
                f"while average precision is {hm['average_precision']:.4f} vs {baselines['majority_class']['average_precision_random_ranking']:.4f} for a random ranking (ROC-AUC {hm['roc_auc']:.4f}).",
                ["evidence.holdout_metrics", "evidence.baselines"],
            )
        )
        if applied:
            text = (
                f"A threshold of {cv_threshold['threshold']:.4f} chosen on training out-of-fold scores to reach precision ≥ {target:.2f} (OOF recall {cv_threshold['recall']:.3f}) "
                f"gives holdout precision {_f(applied['precision'], 3)} and recall {applied['recall']:.3f} ({applied['true_positives']} of {int(y_test.sum())} frauds, {applied['alerts']} alerts)"
            )
        else:
            text = f"No threshold reached precision ≥ {target:.2f} on training out-of-fold scores"
        text += (
            f"; a holdout-chosen (oracle) threshold would reach recall {oracle['recall']:.3f} at precision ≥ {target:.2f}. Precision at top-100 scores is {_f(top_k.get('100'), 3)}."
        )
        claims.append(
            _claim(
                f"{eid}-C4",
                "decision",
                text,
                ["evidence.threshold_analysis"],
                ["With only ~"+str(int(y_test.sum()))+" holdout frauds, precision/recall at a single threshold moves by several points per misclassified case."],
            )
        )
    elif task == "multiclass":
        worst = ", ".join(
            f"class {w['class_index']} (label {w.get('source_label', w['class_index'])}) F1 {w['f1']:.3f}" for w in hm["worst_3_classes"]
        )
        claims.append(
            _claim(
                f"{eid}-C3",
                "fact",
                f"Holdout balanced accuracy {hm['balanced_accuracy']:.4f}, accuracy {hm['accuracy']:.4f}, log loss {hm['log_loss']:.4f} (training-prior log loss {baselines['training_prior_log_loss']:.4f}); "
                f"majority-class macro-F1 would be {baselines['majority_class']['macro_f1']:.4f}. Worst classes: {worst}.",
                ["evidence.holdout_metrics", "evidence.baselines"],
                ["Source labels are PMLB's integer codes 1-26; mapping them to letters assumes PMLB kept alphabetical order (A=1), which was not verified."],
            )
        )
    elif task == "timeseries_regression":
        naive = baselines["naive_lag2_cnt_2d_bfr"]
        seasonal = baselines["seasonal_naive_lag7"]
        mean_b = baselines["training_mean"]
        d = naive["model_minus_baseline_mae_ci"]
        claims.append(
            _claim(
                f"{eid}-C3",
                "fact",
                f"Holdout MAE {hm['mae']:.1f} (RMSE {hm['rmse']:.1f}, MAPE {hm['mape']:.1%}, bias {hm['bias']:+.1f}) versus naive baselines: lag-2 (cnt_2d_bfr) MAE {naive['mae']:.1f}, "
                f"seasonal lag-7 {seasonal['mae']:.1f}, training mean {mean_b['mae']:.1f}. Model minus lag-2 MAE = {d['point']:+.1f} (block-bootstrap 95% {d['low']:+.1f}..{d['high']:+.1f}).",
                ["evidence.holdout_metrics", "evidence.baselines"],
                ["The holdout covers only ~5 months (late 2012, incl. the Hurricane Sandy period); one season is not a full-year test."],
            )
        )
    else:
        ref = baselines["tabular_only_logistic_reference"]
        d = ref["model_minus_reference_auc_ci"]
        claims.append(
            _claim(
                f"{eid}-C3",
                "fact",
                f"On unseen products the text-aware model's holdout ROC-AUC {hm['roc_auc']:.4f} / average precision {hm['average_precision']:.4f} compares with {ref['roc_auc']:.4f} / {ref['average_precision']:.4f} "
                f"for the pre-declared tabular-only reference; AUC difference {d['point']:+.4f} (cluster-bootstrap 95% {d['low']:+.4f}..{d['high']:+.4f}).",
                ["evidence.holdout_metrics", "evidence.baselines"],
            )
        )
    claims.append(
        _claim(
            f"{eid}-C{len(claims) + 1}",
            "recommendation",
            "Do not promote features from importance alone; require decision-time availability, cross-fold stability, production availability, and monitoring.",
            ["evidence.top_final_features", "evidence.production_feature_rule"],
        )
    )
    top_level = {
        "metrics": holdout_metrics,
        "model_name": family,
        "optimization": selected["config_id"],
        "mode": "safe",
        "feature_count": int(final["feature_count"]),
        "train_rows": int(len(bundle.y_train)),
        "test_rows": int(len(y_test)),
    }
    cand_text = "; ".join(f"{r['config_id']} {_ms(r, metric)}" for r in [baseline, *tuned])
    summary = (
        f"Compared default {family} vs {len(candidates)} explicit candidates on the `{recipe}` recipe ({bundle.cv_description}): {cand_text}; "
        f"tuning {'accepted' if decision['accepted'] else 'rejected'} (gain {decision['gain']:+.4f} vs required {decision['required_gain']:.4f}). "
        f"Final fit on {len(bundle.y_train)} training rows, scored once on the holdout ({bundle.holdout_description}): {label} {_f(holdout_value)} [{_f(ci['low'])}, {_f(ci['high'])}]."
    )
    return evidence, claims, summary, top_level


# --------------------------------------------------------------------------
# manifest, execution, validation, and reports
# --------------------------------------------------------------------------

QUESTIONS = {
    "data_understanding": "What does the dataset contain, how is the target distributed, and which quality, identifier, time, or text risks must be resolved before modeling?",
    "leakage_audit": "Which columns violate the decision-time contract, and how much would they inflate the primary metric if left in?",
    "feature_engineering": "Which task-appropriate, train-fitted feature recipe gives the best primary metric for the fewest features?",
    "model_selection": "Which model family ranks best on identical training folds once fold variability is penalized?",
    "optimization_reliability": "Does a small explicit parameter search beat the default by a declared margin, and how does the frozen recipe perform on the locked holdout against naive baselines?",
}
HYPOTHESES = {
    "binary_imbalanced": {
        "data_understanding": "Extreme imbalance makes accuracy meaningless and requires ranking metrics and a time-aware split.",
        "leakage_audit": "The absolute Time counter is not a production feature; removing it costs little or nothing on time-ordered CV.",
        "feature_engineering": "Hour-of-day adds signal beyond the PCA components while log(Amount) is neutral for tree models.",
        "model_selection": "Gradient-boosted trees beat linear and bagged models on average precision.",
        "optimization_reliability": "A frozen boosted model keeps average precision well above the prevalence baseline on the later time window, and a training-chosen threshold transfers approximately.",
    },
    "multiclass": {
        "data_understanding": "Classes are near-balanced, so macro-F1 and accuracy agree; duplicates are the main data risk.",
        "leakage_audit": "No decision-time violation exists, but exact duplicate feature vectors can make random-split scores optimistic.",
        "feature_engineering": "Geometric ratios or pairwise products help shallow-signal classes more than feature selection does.",
        "model_selection": "Boosted trees or extra trees clearly beat a linear multinomial model on 26 classes.",
        "optimization_reliability": "The frozen model's holdout macro-F1 lies within the CV fold range, with errors concentrated in visually similar letters.",
    },
    "timeseries_regression": {
        "data_understanding": "Daily demand trends upward and is seasonal, so time-ordered evaluation is mandatory.",
        "leakage_audit": "casual + registered reconstruct cnt exactly; random KFold looks substantially better than time-ordered CV.",
        "feature_engineering": "Past-only lags and rolling means beat calendar-only features under time-ordered CV.",
        "model_selection": "A model that can extrapolate the trend (linear) or uses level-relative lags beats pure trees on later periods.",
        "optimization_reliability": "The frozen model beats naive lag-2, seasonal lag-7, and training-mean baselines on the later holdout window.",
    },
    "text_tabular_binary": {
        "data_understanding": "Reviews concentrate on a few products and text is mostly present, so grouped evaluation and text features are both necessary.",
        "leakage_audit": "Rating nearly restates the recommendation label and would inflate ROC-AUC dramatically if left in.",
        "feature_engineering": "Review text carries far more signal than tabular attributes; char n-grams add little over word n-grams.",
        "model_selection": "Linear models on sparse TF-IDF match or beat boosted trees on text.",
        "optimization_reliability": "The text model generalizes to unseen products and clearly beats the tabular-only reference on the group holdout.",
    },
}


def build_manifest(results_dir: Path | None = None) -> dict[str, Any]:
    experiments = []
    for key in DATASET_ORDER:
        spec = SPECS[key]
        for kind in STAGES:
            status = "planned"
            if results_dir is not None and (results_dir / result_filename(key, kind)).is_file():
                status = "completed"
            experiments.append(
                {
                    "campaign_id": CAMPAIGN_ID,
                    "experiment_id": experiment_id(key, kind),
                    "dataset": key,
                    "dataset_name": spec.name,
                    "task_type": spec.task_type,
                    "primary_metric": spec.primary_metric,
                    "kind": kind,
                    "question": QUESTIONS[kind],
                    "hypothesis": HYPOTHESES[spec.task_type][kind],
                    "source_url": spec.url,
                    "status": status,
                }
            )
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "objective": "Extend the evidence-backed model-building workflow to imbalanced, multiclass, time-series regression, and text+tabular tasks with task-appropriate splits, leakage checks, and metrics.",
        "extends": "model_building_50_v1",
        "datasets": len(DATASET_ORDER),
        "experiment_count": len(experiments),
        "scientific_contract": {
            "selection_data": "training-only CV with task-appropriate folds (time-ordered, grouped, or stratified)",
            "final_holdout": "consumed once in optimization_reliability after recipe/model/parameter selection",
            "leakage": "declared decision-time exclusions with written rationale; deterministic heuristics only propose review candidates",
            "random_state": RANDOM_STATE,
        },
        "experiments": experiments,
    }


def run_experiment(
    root: Path,
    bundle: TaskBundle,
    kind: str,
    results_dir: Path,
) -> dict[str, Any]:
    """Run one stage for one dataset bundle, write its JSON, and return it."""
    spec = bundle.spec
    eid = experiment_id(spec.key, kind)
    started_at = _utc_now()
    clock = time.perf_counter()
    top_level: dict[str, Any] = {}
    if kind == "data_understanding":
        evidence, claims, summary = _stage_data_understanding(bundle, eid)
    elif kind == "leakage_audit":
        evidence, claims, summary = _stage_leakage_audit(bundle, eid)
    elif kind == "feature_engineering":
        evidence, claims, summary = _stage_feature_engineering(bundle, eid)
    elif kind == "model_selection":
        evidence, claims, summary = _stage_model_selection(bundle, eid, results_dir)
    elif kind == "optimization_reliability":
        evidence, claims, summary, top_level = _stage_optimization_reliability(bundle, eid, results_dir)
    else:
        raise ValueError(f"unknown stage: {kind}")
    evidence["elapsed_seconds"] = float(time.perf_counter() - clock)
    result = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "experiment_id": eid,
        "dataset": spec.key,
        "kind": kind,
        "task_type": spec.task_type,
        "primary_metric": spec.primary_metric,
        "question": QUESTIONS[kind],
        "hypothesis": HYPOTHESES[spec.task_type][kind],
        "decision_time_rule": spec.decision_time_contract,
        "holdout_policy": spec.holdout_policy(),
        "source": spec.url,
        "status": "completed",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "setup_summary": summary,
        "evidence": evidence,
        "claims": claims,
        "provenance": capture_provenance(root, data_paths=bundle.data_paths, random_state=RANDOM_STATE),
        **top_level,
    }
    result = _jsonable(result)
    problems = validate_result(result)
    if problems:
        raise RuntimeError(f"{eid} failed validation: {problems}")
    path = results_dir / result_filename(spec.key, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temp.replace(path)
    return result


def _resolve(payload: dict[str, Any], dotted: str) -> bool:
    node: Any = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def validate_result(result: dict[str, Any]) -> list[str]:
    problems = [f"missing key {k}" for k in REQUIRED_KEYS if k not in result]
    if "llm_review" in result:
        problems.append("llm_review must not be present")
    if result.get("task_type") not in {"binary_imbalanced", "multiclass", "timeseries_regression", "text_tabular_binary"}:
        problems.append("bad task_type")
    if result.get("kind") not in STAGES:
        problems.append("bad kind")
    if not isinstance(result.get("setup_summary"), str) or not result.get("setup_summary"):
        problems.append("setup_summary must be a non-empty string")
    for claim in result.get("claims", []):
        if claim.get("kind") not in CLAIM_KINDS:
            problems.append(f"{claim.get('claim_id')}: bad claim kind")
        if set(claim) != {"claim_id", "kind", "statement", "evidence", "limitations"}:
            problems.append(f"{claim.get('claim_id')}: bad claim keys")
        for path in claim.get("evidence", []):
            if not path.startswith("evidence.") or not _resolve(result, path):
                problems.append(f"{claim.get('claim_id')}: unresolved evidence path {path}")
    if result.get("kind") == "leakage_audit":
        evidence = result.get("evidence", {})
        for key in ("declared_leakage_features", "policy_rationale", "apparent_lift"):
            if key not in evidence:
                problems.append(f"leakage evidence missing {key}")
    if result.get("kind") == "optimization_reliability" and not result.get("evidence", {}).get("holdout_consumed"):
        problems.append("holdout_consumed must be true")
    return problems


def campaign_dir(root: Path) -> Path:
    return Path(root) / "evidence/campaigns" / CAMPAIGN_ID


def load_results(results_dir: Path) -> dict[str, dict[str, Any]]:
    results = {}
    for path in sorted(results_dir.glob("EXP-*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        payload["_path"] = path
        results[payload["experiment_id"]] = payload
    return results


def _rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def render_memory(results: dict[str, dict[str, Any]], root: Path) -> str:
    lines = []
    for eid in sorted(results):
        result = results[eid]
        for claim in result.get("claims", []):
            lines.append(
                json.dumps(
                    {
                        "schema_version": 1,
                        "campaign_id": CAMPAIGN_ID,
                        "experiment_id": eid,
                        "dataset": result.get("dataset"),
                        "kind": result.get("kind"),
                        "task_type": result.get("task_type"),
                        "claim": claim,
                        "source_path": _rel(result["_path"], root),
                    },
                    sort_keys=True,
                )
            )
    return "\n".join(lines) + ("\n" if lines else "")


def render_report(results: dict[str, dict[str, Any]]) -> str:
    by = {(r["dataset"], r["kind"]): r for r in results.values()}
    total = sum(r.get("evidence", {}).get("elapsed_seconds", 0.0) for r in results.values())
    lines = [
        "# Campaign Report — expansion_v1",
        "",
        "Task-type-aware extension of `model_building_50_v1` to four task types it lacked. "
        f"Completed experiments: **{len(results)} / {len(DATASET_ORDER) * len(STAGES)}**; summed stage runtime **{total / 60:.1f} min**.",
        "",
        "Each dataset ran five stages in order. Stages 1–4 select on training rows only (time-ordered, grouped, or stratified CV). "
        "Stage 5 consumes the locked holdout once. Every number below comes from a result JSON in `results/`; claims and evidence paths are in `agent_memory.jsonl`.",
        "",
        "## Overview",
        "",
        "| Dataset | Task | Primary metric | Leakage lift (CV) | Recipe | Model / config | Holdout (95% CI) | Naive baseline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in DATASET_ORDER:
        spec = SPECS[key]
        leak = by.get((key, "leakage_audit"), {}).get("evidence", {})
        fe = by.get((key, "feature_engineering"), {}).get("evidence", {})
        opt = by.get((key, "optimization_reliability"), {})
        ev = opt.get("evidence", {})
        ci = ev.get("holdout_primary_metric_ci", {})
        hold = ev.get("holdout_metrics", {}).get(spec.primary_metric)
        baseline_text = "n/a"
        bl = ev.get("baselines", {})
        if spec.task_type == "binary_imbalanced" and bl:
            baseline_text = f"random-ranking AP {_f(bl['majority_class']['average_precision_random_ranking'])}; majority accuracy {_f(bl['majority_class']['accuracy'])}"
        elif spec.task_type == "multiclass" and bl:
            baseline_text = f"majority macro-F1 {_f(bl['majority_class']['macro_f1'])}"
        elif spec.task_type == "timeseries_regression" and bl:
            baseline_text = f"lag-2 MAE {_f(bl['naive_lag2_cnt_2d_bfr']['mae'], 1)}; lag-7 {_f(bl['seasonal_naive_lag7']['mae'], 1)}; mean {_f(bl['training_mean']['mae'], 1)}"
        elif bl:
            baseline_text = f"tabular-only AUC {_f(bl['tabular_only_logistic_reference']['roc_auc'])}"
        lift = leak.get("apparent_lift")
        lines.append(
            f"| {key} | {spec.task_type} | {METRIC_LABEL[spec.primary_metric]} | "
            + (f"{_signed(lift, spec.primary_metric)} ({', '.join(leak.get('declared_leakage_features', [])) or 'none blocked'})" if lift is not None else "pending")
            + f" | {fe.get('selected_recipe', 'pending')} | {ev.get('model', 'pending')} / {ev.get('selected_optimization', '-')} | "
            + (f"{_f(hold)} ({_f(ci.get('low'))}–{_f(ci.get('high'))})" if hold is not None else "pending")
            + f" | {baseline_text} |"
        )
    for key in DATASET_ORDER:
        spec = SPECS[key]
        lines += ["", f"## {spec.name} — `{key}` ({spec.task_type})", "", spec.description, "", f"*Decision-time contract:* {spec.decision_time_contract}", ""]
        lines += [f"*Holdout policy:* {spec.holdout_policy()}", ""]
        lines += ["| Exp | Stage | What was compared | Key claim |", "|---|---|---|---|"]
        for kind in STAGES:
            result = by.get((key, kind))
            if not result:
                lines.append(f"| {experiment_id(key, kind)} | {kind} | pending | |")
                continue
            claim = result["claims"][0]["statement"] if kind != "leakage_audit" else result["claims"][1]["statement"]
            lines.append(
                f"| {result['experiment_id']} | {kind} | {result['setup_summary'].replace('|', '/')} | {claim.replace('|', '/')} |"
            )
        opt = by.get((key, "optimization_reliability"))
        if opt:
            lines += ["", "**Holdout claims:**", ""]
            for claim in opt["claims"]:
                lines.append(f"- `{claim['claim_id']}` ({claim['kind']}): {claim['statement']}")
        limitations = sorted({lim for r in results.values() if r["dataset"] == key for c in r["claims"] for lim in c["limitations"]})
        if limitations:
            lines += ["", "**Limitations:**", ""]
            lines += [f"- {lim}" for lim in limitations]
    lines += [
        "",
        "## Campaign-wide limitations",
        "",
        "- One dataset per task type and one holdout per dataset; results are evidence about these datasets, not universal rankings.",
        "- 3-fold CV with a single repeat and small explicit parameter grids keep runtime low; fold standard deviations are coarse.",
        "- Fraud model fits keep all positives but only 25% of negatives (seeded); validation and holdout scoring always use every row, and probabilities are prior-corrected.",
        "- Heuristic leakage flags are review prompts. Only the written decision-time contracts declare exclusions.",
        "- No LLM review has been run on these results yet.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(root: Path, out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = out_dir or campaign_dir(root)
    results_dir = out_dir / "results"
    results = load_results(results_dir)
    paths = {
        "manifest": out_dir / "manifest.json",
        "report": out_dir / "CAMPAIGN_REPORT.md",
        "memory": out_dir / "agent_memory.jsonl",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(build_manifest(results_dir), indent=2, sort_keys=True) + "\n")
    paths["report"].write_text(render_report(results))
    paths["memory"].write_text(render_memory(results, root))
    return paths


def status(root: Path, out_dir: Path | None = None) -> list[dict[str, Any]]:
    out_dir = out_dir or campaign_dir(root)
    rows = []
    for key in DATASET_ORDER:
        for kind in STAGES:
            path = out_dir / "results" / result_filename(key, kind)
            state = "planned"
            elapsed = None
            if path.is_file():
                try:
                    payload = json.loads(path.read_text())
                    state = payload.get("status", "unknown")
                    elapsed = payload.get("evidence", {}).get("elapsed_seconds")
                except json.JSONDecodeError:
                    state = "corrupt"
            rows.append({"experiment_id": experiment_id(key, kind), "dataset": key, "kind": kind, "status": state, "elapsed_seconds": elapsed})
    return rows


def run_campaign(
    root: Path,
    *,
    datasets: Iterable[str] | None = None,
    stages: Iterable[str] | None = None,
    force: bool = False,
    out_dir: Path | None = None,
    loader: Callable[[Path, str], TaskBundle] | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    from dclab_rnd.expansion.datasets import load_bundle

    loader = loader or load_bundle
    out_dir = out_dir or campaign_dir(root)
    results_dir = out_dir / "results"
    chosen_datasets = list(datasets) if datasets else list(DATASET_ORDER)
    chosen_stages = list(stages) if stages else list(STAGES)
    executed = []
    for key in chosen_datasets:
        bundle = None
        for kind in STAGES:
            if kind not in chosen_stages:
                continue
            path = results_dir / result_filename(key, kind)
            if path.is_file() and not force:
                try:
                    if json.loads(path.read_text()).get("status") == "completed":
                        log(f"skip {path.name} (completed; use --force to rerun)")
                        continue
                except json.JSONDecodeError:
                    pass
            if bundle is None:
                log(f"loading {key} ...")
                bundle = loader(root, key)
            log(f"run  {experiment_id(key, kind)} {key} {kind} ...")
            started = time.perf_counter()
            result = run_experiment(root, bundle, kind, results_dir)
            log(f"done {experiment_id(key, kind)} in {time.perf_counter() - started:.1f}s — {result['setup_summary'][:160]}")
            executed.append(result)
    write_outputs(root, out_dir)
    return executed
