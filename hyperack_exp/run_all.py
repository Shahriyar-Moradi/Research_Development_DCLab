"""Run all HyperAck experiments 01–15 and write standardized result JSON files."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from shared.protocol import (  # noqa: E402
    evaluate,
    load_clean_df,
    make_xy,
    add_pricing_features,
    save_result,
    split_frame,
)

RANDOM_STATE = 42


def log(msg: str) -> None:
    print(msg, flush=True)


def run_01(train_df, test_df):
    X_train, y_train = make_xy(train_df)
    X_test, y_test = make_xy(test_df)
    geo_cols = [
        "source_latitude",
        "source_longitude",
        "destination_latitude",
        "destination_longitude",
    ]
    geo_imputer = SimpleImputer(strategy="median")
    geo_train = geo_imputer.fit_transform(train_df[geo_cols])
    geo_test = geo_imputer.transform(test_df[geo_cols])
    geo_scaler = StandardScaler()
    kmeans = KMeans(n_clusters=3, n_init=20, random_state=RANDOM_STATE)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["geo_cluster"] = kmeans.fit_predict(geo_scaler.fit_transform(geo_train))
    X_test["geo_cluster"] = kmeans.predict(geo_scaler.transform(geo_test))

    models = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", SVC(C=3.0, probability=True, random_state=RANDOM_STATE)),
            ]
        ),
    }
    all_metrics = {
        name: evaluate(model, X_train, y_train, X_test, y_test)
        for name, model in models.items()
    }
    best_name = max(all_metrics, key=lambda n: all_metrics[n]["roc_auc"])
    save_result(
        "01",
        "baseline_current",
        "Current feature set + train-only KMeans(3); select best LR/XGB/SVM",
        all_metrics[best_name],
        best_model=best_name,
        notes="The model comparison is performed on the common held-out split.",
        feature_count=X_train.shape[1],
    )
    return all_metrics[best_name]["roc_auc"]


def lgbm(n_estimators=700, learning_rate=0.04, num_leaves=31, **kwargs):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    subsample=kwargs.get("subsample", 0.85),
                    colsample_bytree=kwargs.get("colsample_bytree", 0.9),
                    reg_lambda=kwargs.get("reg_lambda", 1.0),
                    min_child_samples=kwargs.get("min_child_samples", 20),
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def run_02(train_df, test_df):
    X_train, y_train = make_xy(train_df)
    X_test, y_test = make_xy(test_df)
    metrics = evaluate(lgbm(), X_train, y_train, X_test, y_test)
    save_result(
        "02",
        "lightgbm_strong_baseline",
        "LightGBM on raw baseline features",
        metrics,
        best_model="LGBMClassifier",
        notes="Strong tree baseline before feature engineering.",
        feature_count=X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_03(train_df, test_df):
    X_train, y_train = make_xy(train_df)
    X_test, y_test = make_xy(test_df)
    X_train = add_pricing_features(X_train)
    X_test = add_pricing_features(X_test)
    metrics = evaluate(lgbm(), X_train, y_train, X_test, y_test)
    save_result(
        "03",
        "fare_pricing_features",
        "LightGBM with fare, fare-delta, and fare-per-km features",
        metrics,
        best_model="LGBMClassifier",
        notes="Pricing features may be highly predictive but can be post-decision leakage.",
        feature_count=X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_04(train_df, test_df):
    X_train, y_train = make_xy(train_df, time_features=True)
    X_test, y_test = make_xy(test_df, time_features=True)
    metrics = evaluate(lgbm(), X_train, y_train, X_test, y_test)
    save_result(
        "04",
        "time_cyclical_features",
        "LightGBM with hour, rush-hour, weekend, and cyclical time features",
        metrics,
        best_model="LGBMClassifier",
        notes="Time features are available early and are generally deployment-safe.",
        feature_count=X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_05(train_df, test_df):
    X_train, y_train = make_xy(train_df, geo=True)
    X_test, y_test = make_xy(test_df, geo=True)
    metrics = evaluate(lgbm(), X_train, y_train, X_test, y_test)
    save_result(
        "05",
        "geo_distance_bearing",
        "LightGBM with Haversine route distance and direction features",
        metrics,
        best_model="LGBMClassifier",
        notes="Geographic transforms retain route structure more faithfully than raw deltas alone.",
        feature_count=X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_06(train_df, test_df):
    X_train, y_train = make_xy(train_df, pricing=True, time_features=True, geo=True)
    X_test, y_test = make_xy(test_df, pricing=True, time_features=True, geo=True)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["distance_x_first_fare"] = X_train["total_distance"] * X_train["first_customer_fare"]
    X_test["distance_x_first_fare"] = X_test["total_distance"] * X_test["first_customer_fare"]
    X_train["category_x_hour"] = X_train["deliverey_category_id"] * X_train["hour"]
    X_test["category_x_hour"] = X_test["deliverey_category_id"] * X_test["hour"]
    bin_cols = ["total_distance", "first_customer_fare"]
    binning = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("bin", KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="quantile")),
        ]
    )
    bins_train = binning.fit_transform(X_train[bin_cols])
    bins_test = binning.transform(X_test[bin_cols])
    for idx, name in enumerate(bin_cols):
        X_train[f"{name}_qbin"] = bins_train[:, idx]
        X_test[f"{name}_qbin"] = bins_test[:, idx]
    metrics = evaluate(
        lgbm(n_estimators=800, learning_rate=0.035), X_train, y_train, X_test, y_test
    )
    save_result(
        "06",
        "interactions_bins",
        "LightGBM with engineered interactions and train-fitted quantile bins",
        metrics,
        best_model="LGBMClassifier",
        notes="Bins are fitted on training rows only.",
        feature_count=X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_07(train_df, test_df):
    X_train, y_train = make_xy(train_df, pricing=True, time_features=True, geo=True)
    X_test, y_test = make_xy(test_df, pricing=True, time_features=True, geo=True)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("select", SelectKBest(mutual_info_classif, k=min(18, X_train.shape[1]))),
            (
                "model",
                LGBMClassifier(
                    n_estimators=800,
                    learning_rate=0.035,
                    num_leaves=31,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )
    metrics = evaluate(model, X_train, y_train, X_test, y_test)
    save_result(
        "07",
        "feature_selection",
        "Mutual-information selection followed by LightGBM",
        metrics,
        best_model="SelectKBest + LGBMClassifier",
        notes="Selection is fit exclusively on training data inside the pipeline.",
        feature_count=X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_08(train_df, test_df):
    X_train, y_train = make_xy(train_df, pricing=True, time_features=True, geo=True)
    X_test, y_test = make_xy(test_df, pricing=True, time_features=True, geo=True)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
                    ),
                ),
            ]
        ),
        param_distributions={
            "model__n_estimators": randint(300, 1200),
            "model__max_depth": randint(3, 10),
            "model__learning_rate": uniform(0.015, 0.135),
            "model__subsample": uniform(0.65, 0.35),
            "model__colsample_bytree": uniform(0.65, 0.35),
            "model__min_child_weight": randint(1, 12),
            "model__reg_lambda": uniform(0.1, 8.0),
        },
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    metrics = evaluate(search, X_train, y_train, X_test, y_test)
    save_result(
        "08",
        "xgboost_tuned",
        "20-trial randomized-search XGBoost on full engineered features",
        metrics,
        best_model="RandomizedSearchCV(XGBClassifier)",
        notes=f"Best params: {search.best_params_}",
        feature_count=X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_09(train_df, test_df):
    X_train, y_train = make_xy(train_df, pricing=True, time_features=True, geo=True)
    X_test, y_test = make_xy(test_df, pricing=True, time_features=True, geo=True)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
                    ),
                ),
            ]
        ),
        param_distributions={
            "model__n_estimators": randint(300, 1400),
            "model__learning_rate": uniform(0.015, 0.12),
            "model__num_leaves": randint(12, 96),
            "model__max_depth": randint(3, 12),
            "model__min_child_samples": randint(10, 80),
            "model__subsample": uniform(0.65, 0.35),
            "model__colsample_bytree": uniform(0.65, 0.35),
            "model__reg_lambda": uniform(0.1, 8.0),
        },
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    metrics = evaluate(search, X_train, y_train, X_test, y_test)
    save_result(
        "09",
        "lightgbm_tuned",
        "20-trial randomized-search LightGBM on full engineered features",
        metrics,
        best_model="RandomizedSearchCV(LGBMClassifier)",
        notes=f"Best params: {search.best_params_}",
        feature_count=X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_10(train_df, test_df):
    X_train, y_train = make_xy(train_df, pricing=True, time_features=True, geo=True)
    X_test, y_test = make_xy(test_df, pricing=True, time_features=True, geo=True)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=500,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    metrics = evaluate(model, X_train, y_train, X_test, y_test)
    save_result(
        "10",
        "histgb_sklearn",
        "sklearn HistGradientBoosting on full engineered features",
        metrics,
        best_model="HistGradientBoostingClassifier",
        notes="A dependency-light boosted-tree competitor.",
        feature_count=X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_11(train_df, test_df):
    from tabpfn import TabPFNClassifier

    X_train, y_train = make_xy(train_df, pricing=True, time_features=True, geo=True)
    X_test, y_test = make_xy(test_df, pricing=True, time_features=True, geo=True)
    if len(X_train) > 1024:
        _, context_idx = train_test_split(
            np.arange(len(X_train)),
            test_size=1024,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
        X_context, y_context = X_train.iloc[context_idx], y_train.iloc[context_idx]
    else:
        X_context, y_context = X_train, y_train

    device = "cpu"
    try:
        import torch

        if torch.backends.mps.is_available():
            device = "mps"
    except ImportError:
        pass

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                TabPFNClassifier(
                    n_estimators=8, device=device, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    metrics = evaluate(model, X_context, y_context, X_test, y_test)
    save_result(
        "11",
        "tabpfn",
        "TabPFN on a deterministic 1,024-row stratified context",
        metrics,
        best_model="TabPFNClassifier",
        notes="Context is capped to 1,024 rows for reliable local inference.",
        feature_count=X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_12(train_df, test_df):
    X_train, y_train = make_xy(train_df, pricing=True, time_features=True, geo=True)
    X_test, y_test = make_xy(test_df, pricing=True, time_features=True, geo=True)
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    base = lgbm(n_estimators=900, learning_rate=0.035, reg_lambda=1.5)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_fit, y_fit)
    calibration_prob = model.predict_proba(X_cal)[:, 1]
    threshold = max(
        np.linspace(0.20, 0.80, 121),
        key=lambda t: f1_score(y_cal, calibration_prob >= t),
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    metrics = evaluate(model, X_train, y_train, X_test, y_test, threshold=threshold)
    save_result(
        "12",
        "threshold_calibration",
        "Isotonic-calibrated LightGBM with threshold selected on an inner validation split",
        metrics,
        best_model="CalibratedClassifierCV(LGBMClassifier)",
        notes="Primary ranking remains ROC-AUC; threshold targets F1.",
        feature_count=X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_13(train_df, test_df):
    X_train, y_train = make_xy(train_df, pricing=True, time_features=True, geo=True)
    X_test, y_test = make_xy(test_df, pricing=True, time_features=True, geo=True)
    linear = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(C=1.0, max_iter=3000, random_state=RANDOM_STATE),
            ),
        ]
    )
    lgbm_m = lgbm(n_estimators=800, learning_rate=0.04)
    xgb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=800,
                    max_depth=6,
                    learning_rate=0.04,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model = StackingClassifier(
        estimators=[("linear", linear), ("lgbm", lgbm_m), ("xgb", xgb)],
        final_estimator=LogisticRegression(
            C=1.0, max_iter=3000, random_state=RANDOM_STATE
        ),
        cv=4,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    metrics = evaluate(model, X_train, y_train, X_test, y_test)
    save_result(
        "13",
        "stacking_ensemble",
        "Out-of-fold stacking of logistic regression, LightGBM, and XGBoost",
        metrics,
        best_model="StackingClassifier",
        notes="Meta-learner receives only cross-validated base probabilities.",
        feature_count=X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_14(train_df, test_df):
    X_train, y_train = make_xy(
        train_df, include_final_fares=False, pricing=True, time_features=True, geo=True
    )
    X_test, y_test = make_xy(
        test_df, include_final_fares=False, pricing=True, time_features=True, geo=True
    )
    metrics = evaluate(
        lgbm(n_estimators=900, learning_rate=0.035, reg_lambda=1.5),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    save_result(
        "14",
        "leakage_safe_features",
        "LightGBM with full safe FE but without final customer/biker fares",
        metrics,
        best_model="LGBMClassifier",
        notes="Use this result for realistic pre-decision performance.",
        feature_count=X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_15(train_df, test_df):
    X_train, y_train = make_xy(train_df, pricing=True, time_features=True, geo=True)
    X_test, y_test = make_xy(test_df, pricing=True, time_features=True, geo=True)

    class ProbabilityBlend:
        def __init__(self):
            self.models = [
                lgbm(
                    n_estimators=1000,
                    learning_rate=0.03,
                    num_leaves=47,
                    min_child_samples=25,
                    reg_lambda=1.5,
                ),
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            XGBClassifier(
                                n_estimators=1000,
                                max_depth=6,
                                learning_rate=0.03,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                min_child_weight=3,
                                reg_lambda=2.0,
                                eval_metric="logloss",
                                random_state=RANDOM_STATE,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
            ]
            self.weights = np.array([0.60, 0.40])

        def fit(self, X, y):
            for candidate in self.models:
                candidate.fit(X, y)
            return self

        def predict_proba(self, X):
            positive = sum(
                weight * candidate.predict_proba(X)[:, 1]
                for weight, candidate in zip(self.weights, self.models)
            )
            return np.column_stack([1 - positive, positive])

    metrics = evaluate(ProbabilityBlend(), X_train, y_train, X_test, y_test)
    save_result(
        "15",
        "full_fe_best_blend",
        "0.60 LightGBM + 0.40 XGBoost blend on pricing/time/geographic features",
        metrics,
        best_model="ProbabilityBlend(LGBMClassifier, XGBClassifier)",
        notes="Fixed blend is a robust final candidate; compare against stack and standalone tuned models.",
        feature_count=X_train.shape[1],
    )
    return metrics["roc_auc"]


EXPERIMENTS = [
    ("01", run_01),
    ("02", run_02),
    ("03", run_03),
    ("04", run_04),
    ("05", run_05),
    ("06", run_06),
    ("07", run_07),
    ("08", run_08),
    ("09", run_09),
    ("10", run_10),
    ("11", run_11),
    ("12", run_12),
    ("13", run_13),
    ("14", run_14),
    ("15", run_15),
]


def main():
    train_df, test_df = split_frame(load_clean_df())
    log(f"train={len(train_df)} test={len(test_df)}")
    summary = []
    for exp_id, fn in EXPERIMENTS:
        started = time.perf_counter()
        log(f"\n=== Running experiment {exp_id} ===")
        try:
            roc = fn(train_df, test_df)
            elapsed = time.perf_counter() - started
            log(f"OK {exp_id}: roc_auc={roc:.4f} in {elapsed:.1f}s")
            summary.append((exp_id, roc, elapsed, "ok"))
        except Exception as exc:
            elapsed = time.perf_counter() - started
            log(f"FAIL {exp_id}: {exc}")
            traceback.print_exc()
            summary.append((exp_id, None, elapsed, str(exc)))

    log("\n=== Summary ===")
    for exp_id, roc, elapsed, status in summary:
        roc_s = f"{roc:.4f}" if roc is not None else "FAIL"
        log(f"{exp_id}: {roc_s} ({elapsed:.1f}s) {status}")


if __name__ == "__main__":
    main()
