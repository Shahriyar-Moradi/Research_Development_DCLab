"""Shared, reproducible evaluation utilities for HyperAck experiments."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dclab_rnd.provenance import capture_provenance  # noqa: E402

DATA_PATH = PROJECT_ROOT / "hyper_ackt-dataset.csv"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
SPLIT_PATH = RESULTS_DIR / "split_indices.npz"
TARGET = "hyper_ack"
RANDOM_STATE = 42


def load_clean_df() -> pd.DataFrame:
    """Load the delivery data and apply the one common row-cleaning rule."""
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["total_distance"]).copy()
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["first_created_at"] = pd.to_datetime(
        df["first_created_at"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    return df


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a persisted 80/20 stratified split, shared by all experiments."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row_ids = df.index.to_numpy()
    if SPLIT_PATH.exists():
        saved = np.load(SPLIT_PATH)
        train_ids, test_ids = saved["train_ids"], saved["test_ids"]
        if set(train_ids).issubset(row_ids) and set(test_ids).issubset(row_ids):
            return df.loc[train_ids].copy(), df.loc[test_ids].copy()

    train_ids, test_ids = train_test_split(
        row_ids,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )
    np.savez(SPLIT_PATH, train_ids=train_ids, test_ids=test_ids)
    return df.loc[train_ids].copy(), df.loc[test_ids].copy()


def base_features(
    df: pd.DataFrame,
    *,
    include_final_fares: bool = True,
    include_time: bool = False,
) -> pd.DataFrame:
    """Return numeric baseline features; no target-derived transformations."""
    columns = [
        "deliverey_category_id",
        "weekday",
        "time_bucket",
        "total_distance",
        "sum_product",
        "source_latitude",
        "source_longitude",
        "destination_latitude",
        "destination_longitude",
        "first_customer_fare",
    ]
    if include_final_fares:
        columns += ["final_customer_fare", "final_biker_fare"]
    X = df[columns].copy()
    if include_time:
        X = add_time_features(df, X)
    return X.replace([np.inf, -np.inf], np.nan)


def add_time_features(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Add operational time features derived from order creation time."""
    timestamp = df["first_created_at"]
    hour = timestamp.dt.hour.fillna(df["time_bucket"] // 6).astype(float)
    weekday = df["weekday"].astype(float)
    X = X.copy()
    X["hour"] = hour
    X["is_rush_hour"] = hour.isin([11, 12, 13, 18, 19, 20, 21]).astype(int)
    X["is_weekend"] = weekday.isin([5, 6, 7]).astype(int)
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    X["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    X["day_of_month"] = df["created_date"].dt.day.astype(float)
    return X


def add_pricing_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add price/distance transforms, preserving missing values for imputation."""
    X = X.copy()
    distance = X["total_distance"].clip(lower=0.05)
    X["log_distance"] = np.log1p(X["total_distance"])
    X["first_fare_per_km"] = X["first_customer_fare"] / distance
    if "final_customer_fare" in X:
        X["final_customer_fare_per_km"] = X["final_customer_fare"] / distance
        X["customer_fare_delta"] = X["final_customer_fare"] - X["first_customer_fare"]
        X["customer_fare_change_pct"] = X["customer_fare_delta"] / X[
            "first_customer_fare"
        ].clip(lower=1)
        X["log_final_customer_fare"] = np.log1p(X["final_customer_fare"])
    if "final_biker_fare" in X:
        X["biker_customer_gap"] = X["final_biker_fare"] - X.get(
            "final_customer_fare", X["first_customer_fare"]
        )
        X["biker_fare_per_km"] = X["final_biker_fare"] / distance
        X["log_final_biker_fare"] = np.log1p(X["final_biker_fare"])
    return X.replace([np.inf, -np.inf], np.nan)


def add_geo_features(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Add delivery direction and great-circle distance features."""
    X = X.copy()
    lat1, lon1 = np.radians(df["source_latitude"]), np.radians(df["source_longitude"])
    lat2, lon2 = np.radians(df["destination_latitude"]), np.radians(
        df["destination_longitude"]
    )
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    X["haversine_km"] = 6371 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
    X["latitude_delta"] = df["destination_latitude"] - df["source_latitude"]
    X["longitude_delta"] = df["destination_longitude"] - df["source_longitude"]
    X["geo_bearing_sin"] = np.sin(
        np.arctan2(
            np.sin(dlon) * np.cos(lat2),
            np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
        )
    )
    X["geo_bearing_cos"] = np.cos(
        np.arctan2(
            np.sin(dlon) * np.cos(lat2),
            np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
        )
    )
    return X.replace([np.inf, -np.inf], np.nan)


def make_xy(
    df: pd.DataFrame,
    *,
    include_final_fares: bool = True,
    pricing: bool = False,
    time_features: bool = False,
    geo: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a configured feature matrix and target vector."""
    X = base_features(
        df, include_final_fares=include_final_fares, include_time=time_features
    )
    if pricing:
        X = add_pricing_features(X)
    if geo:
        X = add_geo_features(df, X)
    return X, df[TARGET].astype(int).copy()


def evaluate(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Fit a classifier and return a standardized held-out evaluation."""
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    probabilities = model.predict_proba(X_test)[:, 1]
    predict_seconds = time.perf_counter() - predict_start
    labels = (probabilities >= threshold).astype(int)
    y_true = y_test.to_numpy().astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, labels)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "avg_precision": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, labels)),
        "recall": float(recall_score(y_true, labels)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "threshold": float(threshold),
        "fit_seconds": round(fit_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
        "confusion_matrix": confusion_matrix(y_true, labels).tolist(),
        # Kept for notebook inspection; stripped before JSON save.
        "y_true": y_true,
        "y_pred": labels,
        "y_prob": probabilities,
    }


def actual_vs_predicted_report(
    y_true: Any,
    y_pred: Any,
    y_prob: Any | None = None,
    *,
    sample_size: int = 25,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Build simple actual-vs-predicted tables for notebook review."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_prob is None:
        y_prob = np.full(len(y_true), np.nan)
    else:
        y_prob = np.asarray(y_prob, dtype=float)

    actual_counts = pd.Series(y_true).value_counts().sort_index()
    predicted_counts = pd.Series(y_pred).value_counts().sort_index()
    comparison = pd.DataFrame(
        {
            "label": [0, 1],
            "actual_count": [int(actual_counts.get(0, 0)), int(actual_counts.get(1, 0))],
            "predicted_count": [
                int(predicted_counts.get(0, 0)),
                int(predicted_counts.get(1, 0)),
            ],
        }
    )
    comparison["actual_pct"] = (
        comparison["actual_count"] / max(len(y_true), 1) * 100
    ).round(2)
    comparison["predicted_pct"] = (
        comparison["predicted_count"] / max(len(y_true), 1) * 100
    ).round(2)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_table = pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
    tn, fp, fn, tp = cm.ravel()
    outcome = pd.DataFrame(
        [
            {"outcome": "true_negative (actual 0, pred 0)", "count": int(tn)},
            {"outcome": "false_positive (actual 0, pred 1)", "count": int(fp)},
            {"outcome": "false_negative (actual 1, pred 0)", "count": int(fn)},
            {"outcome": "true_positive (actual 1, pred 1)", "count": int(tp)},
        ]
    )

    per_class = (
        pd.DataFrame(
            classification_report(
                y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0
            )
        )
        .T.reset_index()
        .rename(columns={"index": "class"})
    )

    detail = pd.DataFrame(
        {
            "row_id": np.arange(len(y_true)),
            "actual": y_true,
            "predicted": y_pred,
            "probability_class_1": np.round(y_prob, 4),
            "correct": y_true == y_pred,
        }
    )
    # Balanced sample of correct and incorrect rows for inspection.
    rng = np.random.default_rng(random_state)
    correct_idx = detail.index[detail["correct"]].to_numpy()
    wrong_idx = detail.index[~detail["correct"]].to_numpy()
    take_correct = min(sample_size // 2, len(correct_idx))
    take_wrong = min(sample_size - take_correct, len(wrong_idx))
    chosen = np.concatenate(
        [
            rng.choice(correct_idx, size=take_correct, replace=False)
            if take_correct
            else np.array([], dtype=int),
            rng.choice(wrong_idx, size=take_wrong, replace=False)
            if take_wrong
            else np.array([], dtype=int),
        ]
    )
    sample = detail.loc[chosen].sort_values(["correct", "row_id"]).reset_index(drop=True)

    return {
        "class_counts": comparison,
        "confusion_matrix": cm_table,
        "outcomes": outcome,
        "per_class_metrics": per_class,
        "prediction_sample": sample,
        "full_predictions": detail,
    }


def save_result(
    exp_id: str,
    name: str,
    strategy: str,
    metrics: dict[str, Any],
    *,
    best_model: str,
    notes: str,
    feature_count: int,
    results_dir: Path | None = None,
) -> Path:
    """Persist one complete benchmark record for the final leaderboard."""
    out_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{exp_id}_{name}.json"
    # Drop large array payloads before writing JSON.
    clean_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"y_true", "y_pred", "y_prob"}
    }
    payload = {
        "schema_version": 1,
        "exp_id": str(exp_id),
        "name": name,
        "strategy": strategy,
        "primary_metric": "roc_auc",
        "metrics": clean_metrics,
        "best_model": best_model,
        "feature_count": int(feature_count),
        "notes": notes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": capture_provenance(
            PROJECT_ROOT,
            data_paths=[DATA_PATH, SPLIT_PATH],
            random_state=RANDOM_STATE,
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def results_frame(paths: Iterable[Path] | None = None) -> pd.DataFrame:
    """Read saved experiment records into a rankable DataFrame."""
    paths = paths or RESULTS_DIR.glob("[0-9][0-9]_*.json")
    rows = []
    for path in paths:
        record = json.loads(Path(path).read_text())
        metrics = record.get("metrics", {})
        # skip incomplete / license-blocked runs
        if metrics.get("roc_auc") is None:
            continue
        rows.append(
            {
                "experiment": f"{record['exp_id']} - {record['name']}",
                "strategy": record["strategy"],
                "model": record["best_model"],
                "features": record["feature_count"],
                **{
                    key: value
                    for key, value in metrics.items()
                    if key not in {"y_true", "y_pred", "y_prob"}
                },
                "notes": record["notes"],
            }
        )
    return pd.DataFrame(rows)
