"""Scientific building blocks for the 50-experiment model-building campaign.

This module deliberately keeps model selection on training/CV data.  The locked
holdout is consumed only by the final reliability experiment for each dataset.
LLM-facing claims are derived from the evidence produced here; an LLM is never
allowed to invent metrics or silently choose a production feature.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "dclab-rd-cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from dclab_rnd.provenance import capture_provenance
from general_pipeline.playbook.features import FeatureEngineer
from general_pipeline.playbook.policy import get_policy

RANDOM_STATE = 42
FEATURE_STAGES = ("raw", "logs", "ratios", "interactions", "full_fe", "selected")
MODEL_FAMILIES = (
    "logistic_regression",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
)
_SUSPICIOUS_NAME = re.compile(
    r"(^|_)(target|label|outcome|result|response|accepted|approved|converted|defaulted|churned|final)(_|$)",
    re.IGNORECASE,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_dataset(
    root: Path,
    key: str,
    *,
    max_rows: int | None,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Load one cached real dataset and create one deterministic locked holdout."""
    data_dir = root / "external_data" / key
    x_path = data_dir / "X.parquet"
    y_path = data_dir / "y.parquet"
    meta_path = data_dir / "meta.json"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"cached dataset is missing for {key}: {data_dir}")

    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)["target"].astype(int)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"key": key}
    source_rows = len(X)

    if max_rows is not None and len(X) > max_rows:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=max_rows,
            stratify=y,
            random_state=random_state,
        )
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=random_state,
    )
    policy = get_policy(key)
    return {
        "X": X,
        "y": y,
        "X_train": X.iloc[train_idx].reset_index(drop=True),
        "y_train": y.iloc[train_idx].reset_index(drop=True),
        "X_test": X.iloc[test_idx].reset_index(drop=True),
        "y_test": y.iloc[test_idx].reset_index(drop=True),
        "meta": meta,
        "source_rows": source_rows,
        "analyzed_rows": len(X),
        "policy": policy,
        "data_paths": [x_path, y_path, meta_path],
    }


def safe_features(frame: pd.DataFrame, policy: Any) -> pd.DataFrame:
    blocked = [
        column for column in policy.unsafe_only_features if column in frame.columns
    ]
    return frame.drop(columns=blocked, errors="ignore")


def expected_calibration_error(
    y_true: Iterable[int], probabilities: Iterable[float], bins: int = 10
) -> float:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (p >= lower) & (p < upper if index < bins - 1 else p <= upper)
        if not mask.any():
            continue
        score += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(score)


def probability_metrics(
    y_true: pd.Series | np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    labels = (p >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "avg_precision": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, labels)),
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels, zero_division=0)),
        "f1": float(f1_score(y, labels, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10": expected_calibration_error(y, p, bins=10),
        "threshold": 0.5,
    }


def _predict_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        values = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))
    return np.asarray(model.predict(X), dtype=float)


def _bounded_model(
    model_name: str,
    optimization: str = "baseline",
    model_params: dict[str, Any] | None = None,
) -> Any:
    # Importing this module loads optional GBDT libraries, so keep it lazy.
    from general_pipeline.models import get_model

    # The repository's old "optimized" profile was tuned for HyperAck. New
    # dataset-specific candidates start from the transparent baseline instead
    # of presenting transferred parameters as fresh optimization evidence.
    factory_level = optimization if model_params is None else "baseline"
    model = get_model(
        model_name,
        optimization=factory_level,
        mode="safe",
        cat_cols=[],
        num_cols=[],
    )
    if model_params:
        model.set_params(**model_params)
    try:
        params = model.get_params(deep=True)
        updates = {key: 1 for key in params if key.endswith(("n_jobs", "thread_count"))}
        if updates:
            model.set_params(**updates)
    except (AttributeError, TypeError, ValueError):
        pass
    return model


def optimization_candidates(model_name: str, *, quick: bool) -> list[dict[str, Any]]:
    """Return a small explicit search space for the selected model family."""
    spaces: dict[str, list[dict[str, Any]]] = {
        "logistic_regression": [
            {"model__C": 0.2},
            {"model__C": 3.0},
            {"model__C": 1.0, "model__class_weight": "balanced"},
            {"model__C": 0.2, "model__class_weight": "balanced"},
        ],
        "extra_trees": [
            {
                "model__n_estimators": 400,
                "model__max_features": "sqrt",
                "model__min_samples_leaf": 2,
            },
            {
                "model__n_estimators": 400,
                "model__max_depth": 14,
                "model__min_samples_leaf": 3,
            },
            {
                "model__n_estimators": 500,
                "model__max_features": 0.7,
                "model__min_samples_leaf": 2,
            },
            {
                "model__n_estimators": 500,
                "model__class_weight": "balanced_subsample",
                "model__min_samples_leaf": 3,
            },
        ],
        "hist_gradient_boosting": [
            {
                "model__learning_rate": 0.05,
                "model__max_iter": 250,
                "model__max_leaf_nodes": 31,
            },
            {
                "model__learning_rate": 0.03,
                "model__max_iter": 350,
                "model__max_leaf_nodes": 20,
                "model__l2_regularization": 1.0,
            },
            {
                "model__learning_rate": 0.08,
                "model__max_iter": 180,
                "model__max_leaf_nodes": 15,
                "model__min_samples_leaf": 30,
            },
            {
                "model__learning_rate": 0.04,
                "model__max_iter": 300,
                "model__max_leaf_nodes": 40,
                "model__l2_regularization": 2.0,
            },
        ],
        "lightgbm": [
            {
                "model__n_estimators": 300,
                "model__learning_rate": 0.03,
                "model__num_leaves": 24,
                "model__min_child_samples": 30,
            },
            {
                "model__n_estimators": 450,
                "model__learning_rate": 0.02,
                "model__num_leaves": 16,
                "model__min_child_samples": 50,
                "model__reg_lambda": 1.0,
            },
            {
                "model__n_estimators": 250,
                "model__learning_rate": 0.05,
                "model__num_leaves": 31,
                "model__subsample": 0.8,
                "model__colsample_bytree": 0.8,
            },
            {
                "model__n_estimators": 400,
                "model__learning_rate": 0.025,
                "model__num_leaves": 40,
                "model__min_child_samples": 60,
                "model__reg_lambda": 2.0,
            },
        ],
        "xgboost": [
            {
                "model__n_estimators": 300,
                "model__learning_rate": 0.04,
                "model__max_depth": 4,
                "model__min_child_weight": 3,
            },
            {
                "model__n_estimators": 450,
                "model__learning_rate": 0.025,
                "model__max_depth": 5,
                "model__subsample": 0.8,
                "model__colsample_bytree": 0.8,
            },
            {
                "model__n_estimators": 250,
                "model__learning_rate": 0.06,
                "model__max_depth": 3,
                "model__reg_lambda": 2.0,
            },
            {
                "model__n_estimators": 400,
                "model__learning_rate": 0.03,
                "model__max_depth": 6,
                "model__min_child_weight": 5,
                "model__reg_lambda": 3.0,
            },
        ],
    }
    candidates = spaces.get(model_name, [])
    if quick:
        candidates = candidates[:3]
    return [
        {"candidate_id": f"{model_name}_C{index:02d}", "params": params}
        for index, params in enumerate(candidates, start=1)
    ]


def _extract_importance(model: Any, columns: list[str]) -> dict[str, float]:
    candidate = model
    if hasattr(candidate, "named_steps"):
        candidate = candidate.named_steps.get("model", candidate)
    values = None
    if hasattr(candidate, "feature_importances_"):
        values = np.asarray(candidate.feature_importances_, dtype=float)
    elif hasattr(candidate, "coef_"):
        values = np.abs(np.asarray(candidate.coef_, dtype=float).reshape(-1))
    if values is None or len(values) != len(columns):
        return {}
    total = float(np.abs(values).sum())
    if total > 0:
        values = np.abs(values) / total
    return {column: float(value) for column, value in zip(columns, values)}


def _aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    mean_value = float(arr.mean())
    std_value = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    half_width = 1.96 * std_value / math.sqrt(max(len(arr), 1))
    return {
        "mean": mean_value,
        "std": std_value,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci95_low": max(0.0, mean_value - half_width),
        "ci95_high": min(1.0, mean_value + half_width),
    }


def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    stage: str,
    model_name: str,
    optimization: str = "baseline",
    model_params: dict[str, Any] | None = None,
    folds: int = 3,
    repeats: int = 1,
) -> dict[str, Any]:
    """Evaluate a train-fitted feature/model recipe without touching the holdout."""
    cv = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=repeats,
        random_state=RANDOM_STATE,
    )
    rows: list[dict[str, float]] = []
    importances: dict[str, list[float]] = defaultdict(list)
    top_counts: Counter[str] = Counter()
    feature_counts: list[int] = []
    started = time.perf_counter()

    for fold_index, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_fit = X.iloc[train_idx].reset_index(drop=True)
        y_fit = y.iloc[train_idx].reset_index(drop=True)
        X_valid = X.iloc[valid_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)

        engineer = FeatureEngineer(stage=stage, random_state=RANDOM_STATE + fold_index)
        engineer.fit(X_fit, y_fit)
        fit_matrix = engineer.transform(X_fit)
        valid_matrix = engineer.transform(X_valid)
        model = _bounded_model(
            model_name,
            optimization,
            model_params=model_params,
        )
        fit_started = time.perf_counter()
        model.fit(fit_matrix, y_fit)
        probabilities = _predict_probabilities(model, valid_matrix)
        metrics = probability_metrics(y_valid, probabilities)
        metrics["fit_seconds"] = float(time.perf_counter() - fit_started)
        rows.append(metrics)
        feature_counts.append(fit_matrix.shape[1])

        importance = _extract_importance(model, list(fit_matrix.columns))
        for feature, value in importance.items():
            importances[feature].append(value)
        for feature, _ in sorted(
            importance.items(), key=lambda item: item[1], reverse=True
        )[:10]:
            top_counts[feature] += 1

    metric_names = (
        "roc_auc",
        "avg_precision",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "brier",
        "log_loss",
        "ece_10",
    )
    summary = {name: _aggregate([row[name] for row in rows]) for name in metric_names}
    stability = []
    total_folds = len(rows)
    for feature in sorted(
        importances,
        key=lambda name: (-top_counts[name], -np.mean(importances[name]), name),
    )[:20]:
        stability.append(
            {
                "feature": feature,
                "observed_folds": len(importances[feature]),
                "top10_frequency": top_counts[feature] / max(total_folds, 1),
                "mean_normalized_importance": float(np.mean(importances[feature])),
            }
        )
    return {
        "stage": stage,
        "model": model_name,
        "optimization": optimization,
        "model_params": model_params or {},
        "folds": folds,
        "repeats": repeats,
        "evaluations": len(rows),
        "feature_count_mean": float(np.mean(feature_counts)),
        "metrics": summary,
        "fold_metrics": rows,
        "feature_stability": stability,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _population_stability_index(
    train: pd.Series, test: pd.Series, bins: int = 10
) -> float | None:
    a = pd.to_numeric(train, errors="coerce").dropna()
    b = pd.to_numeric(test, errors="coerce").dropna()
    if len(a) < 20 or len(b) < 20 or a.nunique() < 2:
        return None
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    a_hist = np.histogram(a, bins=edges)[0] / len(a)
    b_hist = np.histogram(b, bins=edges)[0] / len(b)
    a_hist = np.clip(a_hist, 1e-6, None)
    b_hist = np.clip(b_hist, 1e-6, None)
    return float(np.sum((a_hist - b_hist) * np.log(a_hist / b_hist)))


def suspicious_features(
    X: pd.DataFrame, y: pd.Series, declared: Iterable[str] = ()
) -> list[dict[str, Any]]:
    """Generate review candidates, never automatic declarations of leakage."""
    declared_set = set(declared)
    findings = []
    for column in X.columns:
        series = X[column]
        numeric = pd.to_numeric(series, errors="coerce")
        unique_ratio = float(series.nunique(dropna=False) / max(len(series), 1))
        exact = False
        inverse = False
        if numeric.notna().all() and numeric.nunique() <= 2:
            values = numeric.astype(int).to_numpy()
            target = y.astype(int).to_numpy()
            exact = bool(np.array_equal(values, target))
            inverse = bool(np.array_equal(values, 1 - target))
        auc = None
        if numeric.notna().sum() >= 20 and numeric.nunique() > 1:
            filled = numeric.fillna(numeric.median())
            try:
                raw_auc = float(roc_auc_score(y, filled))
                auc = max(raw_auc, 1.0 - raw_auc)
            except ValueError:
                auc = None
        reasons = []
        if column in declared_set:
            reasons.append("declared_post_outcome_or_contested")
        if exact or inverse:
            reasons.append("exact_target_proxy")
        if _SUSPICIOUS_NAME.search(str(column)):
            reasons.append("suspicious_name")
        if auc is not None and auc >= 0.98:
            reasons.append("extreme_univariate_auc")
        if unique_ratio >= 0.98:
            reasons.append("identifier_like")
        if reasons:
            findings.append(
                {
                    "feature": str(column),
                    "reasons": reasons,
                    "univariate_auc": auc,
                    "unique_ratio": unique_ratio,
                    "requires_human_decision_time_review": True,
                }
            )
    return sorted(
        findings,
        key=lambda item: (
            "declared_post_outcome_or_contested" not in item["reasons"],
            item["feature"],
        ),
    )


def profile_dataset(bundle: dict[str, Any]) -> dict[str, Any]:
    X_train, X_test, y_train = bundle["X_train"], bundle["X_test"], bundle["y_train"]
    profiles = []
    for column in X_train.columns:
        series = X_train[column]
        top_frequency = (
            float(series.value_counts(dropna=False, normalize=True).iloc[0])
            if len(series)
            else 0.0
        )
        psi = _population_stability_index(series, X_test[column])
        risks = []
        missing = float(series.isna().mean())
        unique_ratio = float(series.nunique(dropna=False) / max(len(series), 1))
        if missing > 0.05:
            risks.append("missingness")
        if unique_ratio > 0.98:
            risks.append("identifier_like")
        if top_frequency > 0.995:
            risks.append("near_constant")
        if psi is not None and psi > 0.20:
            risks.append("split_distribution_shift")
        profiles.append(
            {
                "feature": str(column),
                "dtype": str(series.dtype),
                "missing_rate": missing,
                "unique_count": int(series.nunique(dropna=False)),
                "unique_ratio": unique_ratio,
                "top_value_frequency": top_frequency,
                "random_split_psi": psi,
                "production_risks": risks,
            }
        )
    return {
        "source_rows": bundle["source_rows"],
        "analyzed_rows": bundle["analyzed_rows"],
        "train_rows": len(X_train),
        "holdout_rows": len(X_test),
        "feature_count": X_train.shape[1],
        "positive_rate_train": float(y_train.mean()),
        "minority_rate_train": float(min(y_train.mean(), 1 - y_train.mean())),
        "missing_cell_rate_train": float(
            X_train.isna().sum().sum() / max(X_train.size, 1)
        ),
        "duplicate_row_rate_train": float(X_train.duplicated().mean()),
        "feature_profiles": profiles,
        "risk_feature_count": sum(bool(item["production_risks"]) for item in profiles),
        "limitations": [
            "The cached UCI matrices factorized categoricals before this campaign; raw category semantics must be restored for production encoders.",
            "Random train/holdout PSI is a smoke check, not evidence of temporal production stability.",
        ],
    }


def _bootstrap_auc(
    y_true: pd.Series, probabilities: np.ndarray, draws: int = 400
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)
    values = []
    for _ in range(draws):
        indices = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        values.append(roc_auc_score(y[indices], p[indices]))
    if not values:
        score = float(roc_auc_score(y, p))
        return {"low": score, "median": score, "high": score, "draws": 0}
    return {
        "low": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "high": float(np.quantile(values, 0.975)),
        "draws": len(values),
    }


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


def _result_base(
    root: Path, task: dict[str, Any], bundle: dict[str, Any], started: str
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "campaign_id": task["campaign_id"],
        "experiment_id": task["experiment_id"],
        "dataset": task["dataset"],
        "kind": task["kind"],
        "question": task["question"],
        "hypothesis": task["hypothesis"],
        "status": "completed",
        "started_at": started,
        "completed_at": _utc_now(),
        "source": bundle["meta"].get("url"),
        "decision_time_rule": "Features must exist at the declared prediction moment; suspicious features require human semantic review.",
        "holdout_policy": "Model and feature choices use training-only CV. The locked holdout is consumed only by optimization_reliability.",
        "provenance": capture_provenance(
            root, data_paths=bundle["data_paths"], random_state=RANDOM_STATE
        ),
    }


def _read_dependency(
    results_dir: Path, dataset: str, kind: str
) -> dict[str, Any] | None:
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("dataset") == dataset
            and payload.get("kind") == kind
            and payload.get("status") == "completed"
        ):
            return payload
    return None


def run_task(
    root: Path,
    task: dict[str, Any],
    *,
    results_dir: Path,
    max_rows: int | None,
    quick: bool,
) -> dict[str, Any]:
    """Execute one manifest task and return a fully attributable evidence record."""
    started = _utc_now()
    bundle = load_dataset(root, task["dataset"], max_rows=max_rows)
    policy = bundle["policy"]
    X_train_all = bundle["X_train"]
    y_train = bundle["y_train"]
    X_train = safe_features(X_train_all, policy)
    folds = 3
    repeats = 1 if quick else 2
    base = _result_base(root, task, bundle, started)
    evidence: dict[str, Any]
    claims: list[dict[str, Any]]

    if task["kind"] == "data_understanding":
        evidence = profile_dataset(bundle)
        claims = [
            _claim(
                f"{task['experiment_id']}-C1",
                "fact",
                f"{task['dataset']} has {evidence['source_rows']} source rows, {evidence['feature_count']} cached features, and a training positive rate of {evidence['positive_rate_train']:.3f}.",
                [
                    "evidence.source_rows",
                    "evidence.feature_count",
                    "evidence.positive_rate_train",
                ],
            ),
            _claim(
                f"{task['experiment_id']}-C2",
                "risk",
                f"{evidence['risk_feature_count']} features triggered missingness, identifier, near-constant, or random-split shift review rules.",
                ["evidence.feature_profiles"],
                evidence["limitations"],
            ),
        ]

    elif task["kind"] == "leakage_audit":
        declared = [
            column
            for column in policy.unsafe_only_features
            if column in X_train_all.columns
        ]
        findings = suspicious_features(X_train_all, y_train, declared)
        canary = X_train_all.copy()
        canary["__canary_target_copy"] = y_train.to_numpy()
        canary_detected = any(
            item["feature"] == "__canary_target_copy"
            and "exact_target_proxy" in item["reasons"]
            for item in suspicious_features(canary, y_train, declared)
        )
        comparison = None
        if declared:
            safe_cv = evaluate_cv(
                X_train,
                y_train,
                stage="raw",
                model_name="lightgbm",
                folds=folds,
                repeats=1,
            )
            unsafe_cv = evaluate_cv(
                X_train_all,
                y_train,
                stage="raw",
                model_name="lightgbm",
                folds=folds,
                repeats=1,
            )
            comparison = {
                "safe": safe_cv,
                "unsafe": unsafe_cv,
                "apparent_auc_lift": unsafe_cv["metrics"]["roc_auc"]["mean"]
                - safe_cv["metrics"]["roc_auc"]["mean"],
            }
        evidence = {
            "declared_leakage_features": declared,
            "policy_rationale": policy.rationale,
            "heuristic_review_candidates": findings,
            "detector_canary_passed": canary_detected,
            "safe_vs_unsafe_training_cv": comparison,
            "warning": "Name, univariate AUC, and uniqueness rules propose review; only decision-time semantics can confirm leakage.",
        }
        lift = comparison["apparent_auc_lift"] if comparison else None
        claims = [
            _claim(
                f"{task['experiment_id']}-C1",
                "fact",
                f"The leakage detector caught its synthetic exact-target canary: {canary_detected}.",
                ["evidence.detector_canary_passed"],
            ),
            _claim(
                f"{task['experiment_id']}-C2",
                "decision",
                f"Declared decision-time exclusions for {task['dataset']}: {declared or 'none'}"
                + (
                    f"; their apparent training-CV AUC lift was {lift:+.4f}."
                    if lift is not None
                    else "."
                ),
                [
                    "evidence.declared_leakage_features",
                    "evidence.safe_vs_unsafe_training_cv",
                ],
                [evidence["warning"]],
            ),
        ]

    elif task["kind"] == "feature_engineering":
        rows = [
            evaluate_cv(
                X_train,
                y_train,
                stage=stage,
                model_name="lightgbm",
                folds=folds,
                repeats=1,
            )
            for stage in FEATURE_STAGES
        ]
        best_auc = max(row["metrics"]["roc_auc"]["mean"] for row in rows)
        eligible = [
            row for row in rows if row["metrics"]["roc_auc"]["mean"] >= best_auc - 0.002
        ]
        selected = min(
            eligible,
            key=lambda row: (row["feature_count_mean"], row["elapsed_seconds"]),
        )
        raw = next(row for row in rows if row["stage"] == "raw")
        evidence = {
            "stage_results": rows,
            "selected_stage": selected["stage"],
            "selection_rule": "Choose the smallest feature matrix within 0.002 mean training-CV ROC-AUC of the best stage.",
            "selected_auc_mean": selected["metrics"]["roc_auc"]["mean"],
            "raw_auc_mean": raw["metrics"]["roc_auc"]["mean"],
            "selected_vs_raw_auc": selected["metrics"]["roc_auc"]["mean"]
            - raw["metrics"]["roc_auc"]["mean"],
        }
        claims = [
            _claim(
                f"{task['experiment_id']}-C1",
                "recommendation",
                f"Use the {selected['stage']} feature recipe for the next model screen on {task['dataset']}; it achieved {selected['metrics']['roc_auc']['mean']:.4f} mean training-CV ROC-AUC with about {selected['feature_count_mean']:.0f} features.",
                ["evidence.stage_results", "evidence.selection_rule"],
                [
                    "Generic mathematical feature generation may lack domain meaning; a domain expert must review production semantics."
                ],
            )
        ]

    elif task["kind"] == "model_selection":
        feature_result = _read_dependency(
            results_dir, task["dataset"], "feature_engineering"
        )
        stage = (feature_result or {}).get("evidence", {}).get("selected_stage", "raw")
        rows = [
            evaluate_cv(
                X_train, y_train, stage=stage, model_name=model, folds=folds, repeats=1
            )
            for model in MODEL_FAMILIES
        ]
        ranked = sorted(
            rows,
            key=lambda row: (
                -(
                    row["metrics"]["roc_auc"]["mean"]
                    - 0.25 * row["metrics"]["roc_auc"]["std"]
                ),
                row["elapsed_seconds"],
            ),
        )
        selected = ranked[0]
        evidence = {
            "feature_stage": stage,
            "model_results": rows,
            "selected_model": selected["model"],
            "selection_rule": "Rank by mean ROC-AUC minus 0.25×fold standard deviation, then runtime; training CV only.",
            "selected_auc_mean": selected["metrics"]["roc_auc"]["mean"],
            "selected_auc_std": selected["metrics"]["roc_auc"]["std"],
        }
        claims = [
            _claim(
                f"{task['experiment_id']}-C1",
                "recommendation",
                f"{selected['model']} is the training-CV model candidate for {task['dataset']} using {stage} features ({selected['metrics']['roc_auc']['mean']:.4f}±{selected['metrics']['roc_auc']['std']:.4f} ROC-AUC across folds).",
                ["evidence.model_results", "evidence.selection_rule"],
                [
                    "A cross-validation leader is not a universal algorithm winner and has not yet seen the locked holdout."
                ],
            )
        ]

    elif task["kind"] == "optimization_reliability":
        feature_result = _read_dependency(
            results_dir, task["dataset"], "feature_engineering"
        )
        model_result = _read_dependency(results_dir, task["dataset"], "model_selection")
        stage = (feature_result or {}).get("evidence", {}).get("selected_stage", "raw")
        model_name = (
            (model_result or {}).get("evidence", {}).get("selected_model", "lightgbm")
        )
        baseline = evaluate_cv(
            X_train,
            y_train,
            stage=stage,
            model_name=model_name,
            optimization="baseline",
            folds=folds,
            repeats=repeats,
        )
        candidates = optimization_candidates(model_name, quick=quick)
        configurations = [baseline]
        for candidate in candidates:
            evaluated = evaluate_cv(
                X_train,
                y_train,
                stage=stage,
                model_name=model_name,
                optimization=candidate["candidate_id"],
                model_params=candidate["params"],
                folds=folds,
                repeats=repeats,
            )
            configurations.append(evaluated)
        best_tuned = max(
            configurations[1:] or [baseline],
            key=lambda row: (
                row["metrics"]["roc_auc"]["mean"]
                - 0.25 * row["metrics"]["roc_auc"]["std"]
            ),
        )
        auc_delta = (
            best_tuned["metrics"]["roc_auc"]["mean"]
            - baseline["metrics"]["roc_auc"]["mean"]
        )
        selected_config = best_tuned if auc_delta >= 0.001 else baseline

        engineer = FeatureEngineer(stage=stage, random_state=RANDOM_STATE)
        engineer.fit(X_train, y_train)
        train_matrix = engineer.transform(X_train)
        test_matrix = engineer.transform(safe_features(bundle["X_test"], policy))
        final_model = _bounded_model(
            model_name,
            selected_config["optimization"],
            model_params=selected_config.get("model_params"),
        )
        final_started = time.perf_counter()
        final_model.fit(train_matrix, y_train)
        probabilities = _predict_probabilities(final_model, test_matrix)
        final_metrics = probability_metrics(bundle["y_test"], probabilities)
        final_metrics["fit_seconds"] = float(time.perf_counter() - final_started)
        final_importance = _extract_importance(final_model, list(train_matrix.columns))
        top_features = [
            {"feature": feature, "normalized_importance": value}
            for feature, value in sorted(
                final_importance.items(), key=lambda item: item[1], reverse=True
            )[:20]
        ]
        evidence = {
            "feature_stage": stage,
            "model": model_name,
            "configuration_results": configurations,
            "optimization_search_space": candidates,
            "optimized_vs_baseline_cv_auc": auc_delta,
            "selected_optimization": selected_config["optimization"],
            "selected_model_params": selected_config.get("model_params", {}),
            "selection_rule": "Evaluate explicit dataset-specific parameter candidates on training CV; use the best tuned candidate only when mean ROC-AUC improves by at least 0.001, otherwise prefer baseline.",
            "holdout_consumed": True,
            "holdout_metrics": final_metrics,
            "holdout_roc_auc_bootstrap_ci": _bootstrap_auc(
                bundle["y_test"], probabilities
            ),
            "top_final_features": top_features,
            "cv_feature_stability": selected_config["feature_stability"],
            "production_feature_rule": "A feature is a stronger candidate when it is decision-time available, stable across CV folds, contractually obtainable, monitored for missingness/drift, and not merely important in one fit.",
        }
        interval = evidence["holdout_roc_auc_bootstrap_ci"]
        claims = [
            _claim(
                f"{task['experiment_id']}-C1",
                "fact",
                f"The preselected {model_name}/{selected_config['optimization']} recipe scored {final_metrics['roc_auc']:.4f} ROC-AUC on the once-consumed holdout (bootstrap 95% interval {interval['low']:.4f}–{interval['high']:.4f}).",
                [
                    "evidence.holdout_metrics",
                    "evidence.holdout_roc_auc_bootstrap_ci",
                    "evidence.holdout_consumed",
                ],
            ),
            _claim(
                f"{task['experiment_id']}-C2",
                "recommendation",
                "Do not promote features from importance alone; require decision-time availability, cross-fold stability, production availability, and monitoring.",
                [
                    "evidence.top_final_features",
                    "evidence.cv_feature_stability",
                    "evidence.production_feature_rule",
                ],
            ),
        ]
        base["metrics"] = final_metrics
        base["model_name"] = model_name
        base["optimization"] = selected_config["optimization"]
        base["mode"] = "safe"
        base["feature_count"] = int(train_matrix.shape[1])
        base["train_rows"] = len(train_matrix)
        base["test_rows"] = len(test_matrix)
    else:
        raise ValueError(f"unsupported experiment kind: {task['kind']}")

    base["evidence"] = _jsonable(evidence)
    base["claims"] = _jsonable(claims)
    base["llm_review"] = {
        "status": "pending",
        "role": "critic_and_hypothesis_generator_only",
        "required_outputs": [
            "challenge unsupported claims",
            "identify missed leakage or validation risks",
            "propose the smallest next experiment with a falsifiable success gate",
            "cite experiment_id and evidence paths for every conclusion",
        ],
    }
    base["completed_at"] = _utc_now()
    return _jsonable(base)
