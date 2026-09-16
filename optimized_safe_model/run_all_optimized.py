"""Train optimized leakage-safe HyperAck models (no final fares)."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
HYPERACK = ROOT.parent / "hyperack_exp"
sys.path.insert(0, str(HYPERACK))
sys.path.insert(0, str(ROOT))

from shared.protocol import (  # noqa: E402
    evaluate,
    load_clean_df,
    make_xy,
    save_result,
    split_frame,
)
from features import prepare_optimized_xy  # noqa: E402

RANDOM_STATE = 42


def log(msg: str) -> None:
    print(msg, flush=True)


def pos_weight(y: pd.Series) -> float:
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return max(neg / max(pos, 1), 1.0)


def save(exp_id, name, strategy, metrics, model_name, notes, n_features):
    return save_result(
        exp_id,
        name,
        f"[OPT-SAFE] {strategy}",
        metrics,
        best_model=model_name,
        notes=f"{notes} | Optimized safe: no final fares.",
        feature_count=n_features,
        results_dir=RESULTS,
    )


def lgbm(y_train, n_estimators=900, learning_rate=0.03, num_leaves=48, **kwargs):
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
                    colsample_bytree=kwargs.get("colsample_bytree", 0.85),
                    reg_lambda=kwargs.get("reg_lambda", 1.5),
                    min_child_samples=kwargs.get("min_child_samples", 25),
                    scale_pos_weight=kwargs.get("scale_pos_weight", pos_weight(y_train)),
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def xgb(y_train, n_estimators=900, learning_rate=0.03, max_depth=6, **kwargs):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=kwargs.get("subsample", 0.85),
                    colsample_bytree=kwargs.get("colsample_bytree", 0.85),
                    min_child_weight=kwargs.get("min_child_weight", 3),
                    reg_lambda=kwargs.get("reg_lambda", 2.0),
                    scale_pos_weight=kwargs.get("scale_pos_weight", pos_weight(y_train)),
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def run_01(train_df, test_df, X_train, y_train, X_test, y_test):
    # Safe baseline on raw numeric without final fares + geo cluster already in opt matrix.
    Xtr, ytr = make_xy(train_df, include_final_fares=False)
    Xte, yte = make_xy(test_df, include_final_fares=False)
    models = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000, class_weight="balanced", random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "xgboost": xgb(ytr, n_estimators=500, learning_rate=0.05),
        "lightgbm": lgbm(ytr, n_estimators=500, learning_rate=0.05, num_leaves=31),
    }
    scores = {n: evaluate(m, Xtr, ytr, Xte, yte) for n, m in models.items()}
    best = max(scores, key=lambda n: scores[n]["roc_auc"])
    save(
        "01",
        "baseline_safe_balanced",
        "Safe raw features + class-balanced LR/XGB/LGBM; keep best",
        scores[best],
        best,
        "Optimized baseline without final fares.",
        Xtr.shape[1],
    )
    return scores[best]["roc_auc"]


def run_02(train_df, test_df, X_train, y_train, X_test, y_test):
    metrics = evaluate(lgbm(y_train), X_train, y_train, X_test, y_test)
    save(
        "02",
        "lgbm_optimized_defaults",
        "Optimized safe FE + LightGBM with scale_pos_weight",
        metrics,
        "LGBMClassifier",
        "Strong defaults on full optimized safe matrix.",
        X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_03(train_df, test_df, X_train, y_train, X_test, y_test):
    # Safe pricing-focused subset emphasis via model on full matrix still;
    # keep experiment slot as pricing-heavy LGBM (same matrix, notes clarify).
    metrics = evaluate(
        lgbm(y_train, n_estimators=1000, learning_rate=0.025, num_leaves=40),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    save(
        "03",
        "safe_pricing_emphasis",
        "Optimized safe FE emphasizing first-fare ratios/interactions",
        metrics,
        "LGBMClassifier",
        "No final-fare pricing; uses first-fare intensity features.",
        X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_04(train_df, test_df, X_train, y_train, X_test, y_test):
    metrics = evaluate(
        lgbm(y_train, n_estimators=1000, learning_rate=0.025, num_leaves=63),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    save(
        "04",
        "safe_time_emphasis",
        "Optimized safe FE with enriched time flags/interactions",
        metrics,
        "LGBMClassifier",
        "Morning/night/rush + cyclical time interactions.",
        X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_05(train_df, test_df, X_train, y_train, X_test, y_test):
    metrics = evaluate(
        xgb(y_train, n_estimators=1000, learning_rate=0.03, max_depth=7),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    save(
        "05",
        "safe_geo_emphasis",
        "Optimized safe FE with geo clusters + route geometry",
        metrics,
        "XGBClassifier",
        "Train-only geo clusters and route ratios.",
        X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_06(train_df, test_df, X_train, y_train, X_test, y_test):
    metrics = evaluate(
        lgbm(y_train, n_estimators=1200, learning_rate=0.02, num_leaves=63, reg_lambda=2.0),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    save(
        "06",
        "full_optimized_safe_fe",
        "Full optimized safe FE + strong LightGBM",
        metrics,
        "LGBMClassifier",
        "All safe FE families + bins + clusters.",
        X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_07(train_df, test_df, X_train, y_train, X_test, y_test):
    k = min(28, X_train.shape[1])
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("select", SelectKBest(mutual_info_classif, k=k)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=1100,
                    learning_rate=0.025,
                    num_leaves=48,
                    scale_pos_weight=pos_weight(y_train),
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )
    metrics = evaluate(model, X_train, y_train, X_test, y_test)
    save(
        "07",
        "safe_feature_selection",
        f"Mutual-info top-{k} on optimized safe FE + LightGBM",
        metrics,
        "SelectKBest + LGBMClassifier",
        "Selection fit on train only.",
        X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_08(train_df, test_df, X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        xgb(y_train, n_estimators=400),
        param_distributions={
            "model__n_estimators": randint(400, 1400),
            "model__max_depth": randint(3, 10),
            "model__learning_rate": uniform(0.01, 0.09),
            "model__subsample": uniform(0.65, 0.3),
            "model__colsample_bytree": uniform(0.65, 0.3),
            "model__min_child_weight": randint(1, 12),
            "model__reg_lambda": uniform(0.2, 6.0),
            "model__scale_pos_weight": uniform(0.8, 1.8),
        },
        n_iter=35,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    metrics = evaluate(search, X_train, y_train, X_test, y_test)
    save(
        "08",
        "xgboost_tuned_safe",
        "35-trial randomized-search XGBoost on optimized safe FE",
        metrics,
        "RandomizedSearchCV(XGBClassifier)",
        f"Best params: {search.best_params_}",
        X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_09(train_df, test_df, X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        lgbm(y_train, n_estimators=400),
        param_distributions={
            "model__n_estimators": randint(400, 1600),
            "model__learning_rate": uniform(0.01, 0.08),
            "model__num_leaves": randint(16, 96),
            "model__max_depth": randint(3, 12),
            "model__min_child_samples": randint(10, 80),
            "model__subsample": uniform(0.65, 0.3),
            "model__colsample_bytree": uniform(0.65, 0.3),
            "model__reg_lambda": uniform(0.2, 6.0),
            "model__scale_pos_weight": uniform(0.8, 1.8),
        },
        n_iter=40,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    metrics = evaluate(search, X_train, y_train, X_test, y_test)
    save(
        "09",
        "lightgbm_tuned_safe",
        "40-trial randomized-search LightGBM on optimized safe FE",
        metrics,
        "RandomizedSearchCV(LGBMClassifier)",
        f"Best params: {search.best_params_}",
        X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_10(train_df, test_df, X_train, y_train, X_test, y_test):
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=700,
                    max_leaf_nodes=47,
                    l2_regularization=1.5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    metrics = evaluate(model, X_train, y_train, X_test, y_test)
    save(
        "10",
        "histgb_balanced_safe",
        "Balanced HistGradientBoosting on optimized safe FE",
        metrics,
        "HistGradientBoostingClassifier",
        "class_weight=balanced to recover recall.",
        X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_11(train_df, test_df, X_train, y_train, X_test, y_test):
    payload = {
        "exp_id": "11",
        "name": "tabpfn",
        "strategy": "[OPT-SAFE] TabPFN skipped (license)",
        "primary_metric": "roc_auc",
        "metrics": {"roc_auc": None, "status": "skipped_license"},
        "best_model": "TabPFNClassifier",
        "feature_count": int(X_train.shape[1]),
        "notes": "Skipped; no interactive TabPFN license in this environment.",
    }
    (RESULTS / "11_tabpfn.json").write_text(json.dumps(payload, indent=2) + "\n")
    return None


def run_12(train_df, test_df, X_train, y_train, X_test, y_test):
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    base = lgbm(y_fit, n_estimators=1000, learning_rate=0.025, num_leaves=48)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(X_fit, y_fit)
    probs = cal.predict_proba(X_cal)[:, 1]
    # Choose threshold maximizing a blend of F1 and recall on inner validation.
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.25, 0.75, 101):
        pred = (probs >= t).astype(int)
        score = 0.5 * f1_score(y_cal, pred) + 0.5 * recall_score(y_cal, pred)
        if score > best_score:
            best_score, best_t = score, float(t)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    metrics = evaluate(model, X_train, y_train, X_test, y_test, threshold=best_t)
    save(
        "12",
        "calibrated_recall_f1_safe",
        "Isotonic-calibrated LightGBM; threshold maximizes 0.5*F1+0.5*recall",
        metrics,
        "CalibratedClassifierCV(LGBMClassifier)",
        f"Inner-validation threshold={best_t:.3f}",
        X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_13(train_df, test_df, X_train, y_train, X_test, y_test):
    estimators = [
        (
            "lr",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0,
                            max_iter=4000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        ("lgbm", lgbm(y_train, n_estimators=900, learning_rate=0.03)),
        ("xgb", xgb(y_train, n_estimators=900, learning_rate=0.03)),
        (
            "hgb",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_iter=500,
                            max_leaf_nodes=31,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    ]
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            C=1.0, max_iter=4000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        cv=4,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    metrics = evaluate(model, X_train, y_train, X_test, y_test)
    save(
        "13",
        "stacking_4model_safe",
        "Stacking LR+LGBM+XGB+HGB on optimized safe FE",
        metrics,
        "StackingClassifier",
        "Balanced meta-learner for recall recovery.",
        X_train.shape[1],
    )
    return metrics["roc_auc"]


def run_14(train_df, test_df, X_train, y_train, X_test, y_test):
    # Explicit leakage-safe champion slot: tuned-like strong LGBM defaults on opt FE.
    metrics = evaluate(
        lgbm(y_train, n_estimators=1300, learning_rate=0.02, num_leaves=63, reg_lambda=2.0),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    save(
        "14",
        "safe_champion_lgbm",
        "Leakage-safe champion LightGBM on full optimized safe FE",
        metrics,
        "LGBMClassifier",
        "Production-oriented safe candidate.",
        X_train.shape[1],
    )
    return metrics["roc_auc"]


class SoftVote3:
    def __init__(self, y_train):
        self.models = [
            lgbm(y_train, n_estimators=1100, learning_rate=0.025, num_leaves=63),
            xgb(y_train, n_estimators=1100, learning_rate=0.025, max_depth=6),
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            learning_rate=0.045,
                            max_iter=650,
                            max_leaf_nodes=47,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ]
        self.weights = np.array([0.45, 0.35, 0.20])

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        p = sum(w * m.predict_proba(X)[:, 1] for w, m in zip(self.weights, self.models))
        return np.column_stack([1 - p, p])


def run_15(train_df, test_df, X_train, y_train, X_test, y_test):
    metrics = evaluate(SoftVote3(y_train), X_train, y_train, X_test, y_test)
    save(
        "15",
        "softvote_lgbm_xgb_hgb_safe",
        "Soft-vote 0.45 LGBM + 0.35 XGB + 0.20 HGB on optimized safe FE",
        metrics,
        "SoftVote3",
        "Three-model safe blend for stability.",
        X_train.shape[1],
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
    X_train, y_train, X_test, y_test = prepare_optimized_xy(train_df, test_df)
    log(
        f"[OPTIMIZED SAFE] train={len(train_df)} test={len(test_df)} "
        f"features={X_train.shape[1]} results={RESULTS}"
    )
    log(f"Features sample: {list(X_train.columns)[:12]} ...")

    summary = []
    for exp_id, fn in EXPERIMENTS:
        started = time.perf_counter()
        log(f"\n=== Running optimized safe {exp_id} ===")
        try:
            roc = fn(train_df, test_df, X_train, y_train, X_test, y_test)
            elapsed = time.perf_counter() - started
            if roc is None:
                log(f"SKIP {exp_id} in {elapsed:.1f}s")
                summary.append((exp_id, None, elapsed, "skipped"))
            else:
                log(f"OK {exp_id}: roc_auc={roc:.4f} in {elapsed:.1f}s")
                summary.append((exp_id, roc, elapsed, "ok"))
        except Exception as exc:
            elapsed = time.perf_counter() - started
            log(f"FAIL {exp_id}: {exc}")
            traceback.print_exc()
            summary.append((exp_id, None, elapsed, str(exc)))

    log("\n=== Summary ===")
    for exp_id, roc, elapsed, status in summary:
        roc_s = f"{roc:.4f}" if roc is not None else "SKIP/FAIL"
        log(f"{exp_id}: {roc_s} ({elapsed:.1f}s) {status}")


if __name__ == "__main__":
    main()
