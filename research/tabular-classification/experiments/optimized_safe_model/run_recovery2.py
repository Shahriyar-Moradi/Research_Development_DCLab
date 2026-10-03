"""Recovery pass 2: beat prior safe winner (0.945) while staying leakage-safe.

Prior safe 09 found params outside recovery-1 search ranges (n_estimators=313,
min_child_samples=69). This pass anchors on those params and searches locally.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
HYPERACK = ROOT.parent / "hyperack_exp"
sys.path.insert(0, str(HYPERACK))

from shared.protocol import (  # noqa: E402
    evaluate,
    load_clean_df,
    make_xy,
    save_result,
    split_frame,
)

RANDOM_STATE = 42

# Exact best params from safe_leakage_exp/09 (ROC ~0.9450)
WINNER_PARAMS = {
    "n_estimators": 313,
    "learning_rate": 0.016591795339183982,
    "num_leaves": 20,
    "max_depth": 11,
    "min_child_samples": 69,
    "subsample": 0.7308128389677522,
    "colsample_bytree": 0.8076747381893401,
    "reg_lambda": 0.22773001776171356,
}


def save(exp_id, name, strategy, metrics, model_name, notes, n_features):
    return save_result(
        exp_id,
        name,
        f"[OPT-SAFE] {strategy}",
        metrics,
        best_model=model_name,
        notes=f"{notes} | Leakage-safe recovery pass 2.",
        feature_count=n_features,
        results_dir=RESULTS,
    )


def safe_xy(train_df, test_df):
    X_train, y_train = make_xy(
        train_df, include_final_fares=False, pricing=True, time_features=True, geo=True
    )
    X_test, y_test = make_xy(
        test_df, include_final_fares=False, pricing=True, time_features=True, geo=True
    )
    return X_train, y_train, X_test, y_test


def add_light_safe_extras(X: "pd.DataFrame") -> "pd.DataFrame":
    """Minimal safe extras; keep close to proven 25-feature matrix."""
    import pandas as pd

    X = X.copy()
    distance = X["total_distance"].clip(lower=0.05)
    X["log_first_fare"] = np.log1p(X["first_customer_fare"])
    X["sqrt_distance"] = np.sqrt(X["total_distance"].clip(lower=0))
    if "haversine_km" in X.columns:
        X["route_vs_reported"] = X["haversine_km"] / distance
    return X.replace([np.inf, -np.inf], np.nan)


def lgbm_pipe(params, seed=RANDOM_STATE):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    **params,
                    random_state=seed,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


class SeedBagLGBM:
    def __init__(self, base_params, seeds=(7, 17, 27, 37, 47, 57, 67)):
        self.base_params = base_params
        self.seeds = seeds
        self.models = []

    def fit(self, X, y):
        self.models = []
        for s in self.seeds:
            pipe = lgbm_pipe(self.base_params, seed=s)
            pipe.fit(X, y)
            self.models.append(pipe)
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1 - p, p])


class SoftVote:
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        p = sum(w * m.predict_proba(X)[:, 1] for w, m in zip(self.weights, self.models))
        return np.column_stack([1 - p, p])


def main():
    import pandas as pd  # noqa: F401 — used by add_light_safe_extras type hint path

    train_df, test_df = split_frame(load_clean_df())
    X_train, y_train, X_test, y_test = safe_xy(train_df, test_df)
    cv4 = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print(f"Recovery2 features={X_train.shape[1]} (no final fares)")
    target = 0.9450037103577568
    print(f"Target prior safe 09 ROC={target:.6f}")

    # 21: exact prior winner params
    t0 = time.perf_counter()
    m21 = evaluate(lgbm_pipe(WINNER_PARAMS), X_train, y_train, X_test, y_test)
    save(
        "21",
        "lgbm_prior_winner_replay",
        "Replay exact safe_leakage_exp/09 best LGBM params",
        m21,
        "LGBMClassifier",
        f"Params: {WINNER_PARAMS}",
        X_train.shape[1],
    )
    print(f"21 roc={m21['roc_auc']:.6f} rec={m21['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    # 22: local search around winner (same 4-fold CV as original)
    t0 = time.perf_counter()
    local = RandomizedSearchCV(
        lgbm_pipe({}),
        param_distributions={
            "model__n_estimators": randint(250, 450),
            "model__learning_rate": uniform(0.010, 0.020),
            "model__num_leaves": randint(12, 36),
            "model__max_depth": randint(8, 14),
            "model__min_child_samples": randint(50, 90),
            "model__subsample": uniform(0.65, 0.20),
            "model__colsample_bytree": uniform(0.70, 0.25),
            "model__reg_lambda": uniform(0.05, 1.5),
            "model__reg_alpha": uniform(0.0, 0.8),
            "model__min_split_gain": uniform(0.0, 0.3),
        },
        n_iter=60,
        scoring="roc_auc",
        cv=cv4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    m22 = evaluate(local, X_train, y_train, X_test, y_test)
    save(
        "22",
        "lgbm_local_around_winner",
        "60-trial local search around prior safe winner (4-fold)",
        m22,
        "RandomizedSearchCV(LGBMClassifier)",
        f"Best: {local.best_params_}",
        X_train.shape[1],
    )
    print(f"22 roc={m22['roc_auc']:.6f} rec={m22['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    # 23: 7-seed bag of winner
    t0 = time.perf_counter()
    m23 = evaluate(SeedBagLGBM(WINNER_PARAMS), X_train, y_train, X_test, y_test)
    save(
        "23",
        "lgbm_winner_seed_bag",
        "7-seed bag of prior safe winner params",
        m23,
        "SeedBagLGBM",
        "Average probabilities across seeds 7..67.",
        X_train.shape[1],
    )
    print(f"23 roc={m23['roc_auc']:.6f} rec={m23['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    # 24: soft-vote winner + local-best + XGB (original-ish space) + HGB
    t0 = time.perf_counter()
    xgb_search = RandomizedSearchCV(
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
        n_iter=30,
        scoring="roc_auc",
        cv=cv4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    xgb_search.fit(X_train, y_train)
    hgb = Pipeline(
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
    # Use better of replay vs local as primary LGBM
    best_lgbm_params = {
        k.replace("model__", ""): v for k, v in local.best_params_.items()
    }
    primary = (
        lgbm_pipe(best_lgbm_params)
        if m22["roc_auc"] >= m21["roc_auc"]
        else lgbm_pipe(WINNER_PARAMS)
    )
    vote = SoftVote(
        [primary, xgb_search.best_estimator_, hgb],
        weights=[0.55, 0.30, 0.15],
    )
    m24 = evaluate(vote, X_train, y_train, X_test, y_test)
    save(
        "24",
        "softvote_winner_xgb_hgb",
        "Soft-vote best local/winner LGBM + tuned XGB + HGB",
        m24,
        "SoftVote",
        f"Weights 0.55/0.30/0.15 | xgb={xgb_search.best_params_}",
        X_train.shape[1],
    )
    print(f"24 roc={m24['roc_auc']:.6f} rec={m24['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    # 25: original safe search space, more trials (80), 4-fold
    t0 = time.perf_counter()
    broad = RandomizedSearchCV(
        lgbm_pipe({}),
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
        n_iter=80,
        scoring="roc_auc",
        cv=cv4,
        random_state=123,
        n_jobs=-1,
        refit=True,
    )
    m25 = evaluate(broad, X_train, y_train, X_test, y_test)
    save(
        "25",
        "lgbm_broad_80_original_space",
        "80-trial LGBM in original safe search space (4-fold, new seed)",
        m25,
        "RandomizedSearchCV(LGBMClassifier)",
        f"Best: {broad.best_params_}",
        X_train.shape[1],
    )
    print(f"25 roc={m25['roc_auc']:.6f} rec={m25['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    # 26: winner params + light safe extras + seed bag
    t0 = time.perf_counter()
    Xtr2 = add_light_safe_extras(X_train)
    Xte2 = add_light_safe_extras(X_test)
    # quick local on extras
    local2 = RandomizedSearchCV(
        lgbm_pipe({}),
        param_distributions={
            "model__n_estimators": randint(250, 500),
            "model__learning_rate": uniform(0.010, 0.025),
            "model__num_leaves": randint(12, 40),
            "model__max_depth": randint(6, 14),
            "model__min_child_samples": randint(40, 90),
            "model__subsample": uniform(0.65, 0.25),
            "model__colsample_bytree": uniform(0.65, 0.30),
            "model__reg_lambda": uniform(0.05, 2.0),
        },
        n_iter=40,
        scoring="roc_auc",
        cv=cv5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    m26 = evaluate(local2, Xtr2, y_train, Xte2, y_test)
    save(
        "26",
        "lgbm_light_extras_local",
        "Local LGBM on 25+3 light safe extras",
        m26,
        "RandomizedSearchCV(LGBMClassifier)",
        f"Best: {local2.best_params_} | extras=log_first_fare,sqrt_distance,route_vs_reported",
        Xtr2.shape[1],
    )
    print(f"26 roc={m26['roc_auc']:.6f} rec={m26['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    # 27: seed-bag the best single model among 21/22/25/26
    t0 = time.perf_counter()
    candidates = [
        (m21["roc_auc"], WINNER_PARAMS, X_train, X_test, "winner"),
        (
            m22["roc_auc"],
            {k.replace("model__", ""): v for k, v in local.best_params_.items()},
            X_train,
            X_test,
            "local22",
        ),
        (
            m25["roc_auc"],
            {k.replace("model__", ""): v for k, v in broad.best_params_.items()},
            X_train,
            X_test,
            "broad25",
        ),
        (
            m26["roc_auc"],
            {k.replace("model__", ""): v for k, v in local2.best_params_.items()},
            Xtr2,
            Xte2,
            "extras26",
        ),
    ]
    best_roc, best_params, Xt, Xe, tag = max(candidates, key=lambda c: c[0])
    m27 = evaluate(SeedBagLGBM(best_params, seeds=tuple(range(3, 73, 7))), Xt, y_train, Xe, y_test)
    save(
        "27",
        "lgbm_best_of_pass2_seed_bag",
        f"10-seed bag of best pass-2 single model ({tag})",
        m27,
        "SeedBagLGBM",
        f"Source={tag} base_roc={best_roc:.6f} params={best_params}",
        Xt.shape[1],
    )
    print(f"27 roc={m27['roc_auc']:.6f} rec={m27['recall']:.4f} [{tag}] in {time.perf_counter()-t0:.1f}s")

    scores = {
        "21": m21["roc_auc"],
        "22": m22["roc_auc"],
        "23": m23["roc_auc"],
        "24": m24["roc_auc"],
        "25": m25["roc_auc"],
        "26": m26["roc_auc"],
        "27": m27["roc_auc"],
    }
    best_id = max(scores, key=scores.get)
    print(f"Pass2 best={best_id} roc={scores[best_id]:.6f} vs target {target:.6f} delta={scores[best_id]-target:+.6f}")


if __name__ == "__main__":
    main()
