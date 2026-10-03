"""Recovery pass: beat previous safe winner while staying leakage-safe."""

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


def save(exp_id, name, strategy, metrics, model_name, notes, n_features):
    return save_result(
        exp_id,
        name,
        f"[OPT-SAFE] {strategy}",
        metrics,
        best_model=model_name,
        notes=f"{notes} | Leakage-safe recovery pass.",
        feature_count=n_features,
        results_dir=RESULTS,
    )


def safe_xy(train_df, test_df):
    # Proven safe matrix from safe_leakage_exp winner path.
    X_train, y_train = make_xy(
        train_df, include_final_fares=False, pricing=True, time_features=True, geo=True
    )
    X_test, y_test = make_xy(
        test_df, include_final_fares=False, pricing=True, time_features=True, geo=True
    )
    return X_train, y_train, X_test, y_test


def main():
    train_df, test_df = split_frame(load_clean_df())
    X_train, y_train, X_test, y_test = safe_xy(train_df, test_df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print(f"Recovery features={X_train.shape[1]} (no final fares)")

    # 16: deeper LGBM search WITHOUT scale_pos_weight (protect ROC-AUC)
    t0 = time.perf_counter()
    lgbm_search = RandomizedSearchCV(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
                ),
            ]
        ),
        param_distributions={
            "model__n_estimators": randint(500, 2000),
            "model__learning_rate": uniform(0.01, 0.07),
            "model__num_leaves": randint(15, 80),
            "model__max_depth": randint(3, 12),
            "model__min_child_samples": randint(8, 60),
            "model__subsample": uniform(0.7, 0.25),
            "model__colsample_bytree": uniform(0.7, 0.25),
            "model__reg_lambda": uniform(0.1, 5.0),
            "model__reg_alpha": uniform(0.0, 2.0),
        },
        n_iter=50,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    m16 = evaluate(lgbm_search, X_train, y_train, X_test, y_test)
    save(
        "16",
        "lgbm_deep_tune_safe",
        "50-trial 5-fold LGBM on proven safe FE (no class weight)",
        m16,
        "RandomizedSearchCV(LGBMClassifier)",
        f"Best: {lgbm_search.best_params_}",
        X_train.shape[1],
    )
    print(f"16 roc={m16['roc_auc']:.4f} rec={m16['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    # 17: deeper XGB search
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
            "model__n_estimators": randint(500, 1800),
            "model__max_depth": randint(3, 9),
            "model__learning_rate": uniform(0.01, 0.08),
            "model__subsample": uniform(0.7, 0.25),
            "model__colsample_bytree": uniform(0.7, 0.25),
            "model__min_child_weight": randint(1, 10),
            "model__reg_lambda": uniform(0.2, 6.0),
            "model__gamma": uniform(0.0, 2.0),
        },
        n_iter=45,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    m17 = evaluate(xgb_search, X_train, y_train, X_test, y_test)
    save(
        "17",
        "xgb_deep_tune_safe",
        "45-trial 5-fold XGBoost on proven safe FE",
        m17,
        "RandomizedSearchCV(XGBClassifier)",
        f"Best: {xgb_search.best_params_}",
        X_train.shape[1],
    )
    print(f"17 roc={m17['roc_auc']:.4f} rec={m17['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    # 18: seed bagging of best LGBM params (stability + small lift)
    t0 = time.perf_counter()
    best_params = {
        k.replace("model__", ""): v for k, v in lgbm_search.best_params_.items()
    }

    class SeedBagLGBM:
        def __init__(self, base_params, seeds=(7, 17, 27, 37, 47)):
            self.base_params = base_params
            self.seeds = seeds
            self.models = []

        def fit(self, X, y):
            self.models = []
            for s in self.seeds:
                pipe = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            LGBMClassifier(
                                **self.base_params,
                                random_state=s,
                                n_jobs=-1,
                                verbosity=-1,
                            ),
                        ),
                    ]
                )
                pipe.fit(X, y)
                self.models.append(pipe)
            return self

        def predict_proba(self, X):
            p = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
            return np.column_stack([1 - p, p])

    m18 = evaluate(SeedBagLGBM(best_params), X_train, y_train, X_test, y_test)
    save(
        "18",
        "lgbm_seed_bag_safe",
        "5-seed bag of best safe LGBM params",
        m18,
        "SeedBagLGBM",
        "Average probabilities across seeds.",
        X_train.shape[1],
    )
    print(f"18 roc={m18['roc_auc']:.4f} rec={m18['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    # 19: soft vote best LGBM search + best XGB search + HGB
    t0 = time.perf_counter()

    class SoftVoteBest:
        def __init__(self):
            self.models = [
                lgbm_search.best_estimator_,
                xgb_search.best_estimator_,
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            HistGradientBoostingClassifier(
                                learning_rate=0.05,
                                max_iter=800,
                                max_leaf_nodes=31,
                                l2_regularization=1.0,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
            ]
            self.weights = np.array([0.5, 0.35, 0.15])

        def fit(self, X, y):
            # first two already refit by search; refit all for cleanliness
            for m in self.models:
                m.fit(X, y)
            return self

        def predict_proba(self, X):
            p = sum(w * m.predict_proba(X)[:, 1] for w, m in zip(self.weights, self.models))
            return np.column_stack([1 - p, p])

    m19 = evaluate(SoftVoteBest(), X_train, y_train, X_test, y_test)
    save(
        "19",
        "softvote_deep_tuned_safe",
        "Soft-vote best LGBM + best XGB + HGB on proven safe FE",
        m19,
        "SoftVoteBest",
        "Weights 0.50/0.35/0.15",
        X_train.shape[1],
    )
    print(f"19 roc={m19['roc_auc']:.4f} rec={m19['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    # 20: more trees around best LGBM with slightly lower lr
    t0 = time.perf_counter()
    params = dict(best_params)
    params["n_estimators"] = int(params.get("n_estimators", 800) * 1.4)
    params["learning_rate"] = float(params.get("learning_rate", 0.03)) * 0.75
    model20 = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
            ),
        ]
    )
    m20 = evaluate(model20, X_train, y_train, X_test, y_test)
    save(
        "20",
        "lgbm_best_params_more_trees_safe",
        "Best safe LGBM params with more trees / lower LR",
        m20,
        "LGBMClassifier",
        f"Params: {params}",
        X_train.shape[1],
    )
    print(f"20 roc={m20['roc_auc']:.4f} rec={m20['recall']:.4f} in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
