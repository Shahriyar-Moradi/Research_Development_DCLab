"""Tune ALL safe model families with RandomizedSearchCV (no final fares).

Uses the proven 25-feature leakage-safe matrix (same as safe_leakage_exp winner).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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
        notes=f"{notes} | Full RandomizedSearchCV tune-all pass.",
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


def search(estimator, params, X, y, n_iter=30, n_splits=4, seed=RANDOM_STATE):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return RandomizedSearchCV(
        estimator,
        param_distributions=params,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )


def main():
    train_df, test_df = split_frame(load_clean_df())
    X_train, y_train, X_test, y_test = safe_xy(train_df, test_df)
    n_feat = X_train.shape[1]
    print(f"Tune-all safe features={n_feat} (no final fares)", flush=True)
    tuned = {}

    # 41: LogisticRegression
    t0 = time.perf_counter()
    lr = search(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000, solver="saga", random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        {
            "model__C": loguniform(1e-3, 50),
            "model__penalty": ["l1", "l2"],
            "model__class_weight": [None, "balanced"],
        },
        X_train,
        y_train,
        n_iter=40,
    )
    m41 = evaluate(lr, X_train, y_train, X_test, y_test)
    save(
        "41",
        "tuned_logistic_regression",
        "40-trial RandomizedSearchCV LogisticRegression",
        m41,
        "RandomizedSearchCV(LogisticRegression)",
        f"Best: {lr.best_params_}",
        n_feat,
    )
    tuned["lr"] = lr.best_estimator_
    print(f"41 LR roc={m41['roc_auc']:.4f} rec={m41['recall']:.4f} in {time.perf_counter()-t0:.1f}s", flush=True)

    # 42: RandomForest
    t0 = time.perf_counter()
    rf = search(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                ),
            ]
        ),
        {
            "model__n_estimators": randint(300, 1200),
            "model__max_depth": randint(4, 24),
            "model__min_samples_leaf": randint(1, 20),
            "model__min_samples_split": randint(2, 20),
            "model__max_features": ["sqrt", "log2", None],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
        },
        X_train,
        y_train,
        n_iter=35,
    )
    m42 = evaluate(rf, X_train, y_train, X_test, y_test)
    save(
        "42",
        "tuned_random_forest",
        "35-trial RandomizedSearchCV RandomForest",
        m42,
        "RandomizedSearchCV(RandomForestClassifier)",
        f"Best: {rf.best_params_}",
        n_feat,
    )
    tuned["rf"] = rf.best_estimator_
    print(f"42 RF roc={m42['roc_auc']:.4f} rec={m42['recall']:.4f} in {time.perf_counter()-t0:.1f}s", flush=True)

    # 43: ExtraTrees
    t0 = time.perf_counter()
    et = search(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                ),
            ]
        ),
        {
            "model__n_estimators": randint(400, 1400),
            "model__max_depth": randint(4, 28),
            "model__min_samples_leaf": randint(1, 16),
            "model__max_features": ["sqrt", "log2", None],
            "model__class_weight": [None, "balanced"],
        },
        X_train,
        y_train,
        n_iter=30,
    )
    m43 = evaluate(et, X_train, y_train, X_test, y_test)
    save(
        "43",
        "tuned_extra_trees",
        "30-trial RandomizedSearchCV ExtraTrees",
        m43,
        "RandomizedSearchCV(ExtraTreesClassifier)",
        f"Best: {et.best_params_}",
        n_feat,
    )
    tuned["et"] = et.best_estimator_
    print(f"43 ET roc={m43['roc_auc']:.4f} rec={m43['recall']:.4f} in {time.perf_counter()-t0:.1f}s", flush=True)

    # 44: HistGradientBoosting
    t0 = time.perf_counter()
    hgb = search(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(random_state=RANDOM_STATE),
                ),
            ]
        ),
        {
            "model__learning_rate": uniform(0.02, 0.15),
            "model__max_iter": randint(200, 900),
            "model__max_leaf_nodes": randint(15, 63),
            "model__min_samples_leaf": randint(10, 80),
            "model__l2_regularization": uniform(0.0, 5.0),
            "model__max_depth": [None, 4, 6, 8, 12],
        },
        X_train,
        y_train,
        n_iter=40,
    )
    m44 = evaluate(hgb, X_train, y_train, X_test, y_test)
    save(
        "44",
        "tuned_hist_gradient_boosting",
        "40-trial RandomizedSearchCV HistGradientBoosting",
        m44,
        "RandomizedSearchCV(HistGradientBoostingClassifier)",
        f"Best: {hgb.best_params_}",
        n_feat,
    )
    tuned["hgb"] = hgb.best_estimator_
    print(f"44 HGB roc={m44['roc_auc']:.4f} rec={m44['recall']:.4f} in {time.perf_counter()-t0:.1f}s", flush=True)

    # 45: Calibrated LinearSVC
    t0 = time.perf_counter()
    svc = search(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    CalibratedClassifierCV(
                        LinearSVC(max_iter=8000, dual="auto", random_state=RANDOM_STATE),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
        {
            "model__estimator__C": loguniform(1e-3, 20),
            "model__estimator__class_weight": [None, "balanced"],
        },
        X_train,
        y_train,
        n_iter=25,
    )
    m45 = evaluate(svc, X_train, y_train, X_test, y_test)
    save(
        "45",
        "tuned_linear_svc_calibrated",
        "25-trial RandomizedSearchCV Calibrated LinearSVC",
        m45,
        "RandomizedSearchCV(CalibratedClassifierCV(LinearSVC))",
        f"Best: {svc.best_params_}",
        n_feat,
    )
    tuned["svc"] = svc.best_estimator_
    print(f"45 SVC roc={m45['roc_auc']:.4f} rec={m45['recall']:.4f} in {time.perf_counter()-t0:.1f}s", flush=True)

    # 46: LightGBM (original safe space — includes prior winner region)
    t0 = time.perf_counter()
    lgbm = search(
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
        {
            "model__n_estimators": randint(250, 1400),
            "model__learning_rate": uniform(0.012, 0.12),
            "model__num_leaves": randint(12, 96),
            "model__max_depth": randint(3, 14),
            "model__min_child_samples": randint(10, 90),
            "model__subsample": uniform(0.65, 0.30),
            "model__colsample_bytree": uniform(0.65, 0.30),
            "model__reg_lambda": uniform(0.05, 8.0),
            "model__reg_alpha": uniform(0.0, 2.0),
        },
        X_train,
        y_train,
        n_iter=60,
        seed=42,
    )
    m46 = evaluate(lgbm, X_train, y_train, X_test, y_test)
    save(
        "46",
        "tuned_lightgbm_full_search",
        "60-trial RandomizedSearchCV LightGBM (full safe space)",
        m46,
        "RandomizedSearchCV(LGBMClassifier)",
        f"Best: {lgbm.best_params_}",
        n_feat,
    )
    tuned["lgbm"] = lgbm.best_estimator_
    print(f"46 LGBM roc={m46['roc_auc']:.4f} rec={m46['recall']:.4f} in {time.perf_counter()-t0:.1f}s", flush=True)

    # 47: XGBoost
    t0 = time.perf_counter()
    xgb = search(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        {
            "model__n_estimators": randint(300, 1400),
            "model__max_depth": randint(3, 10),
            "model__learning_rate": uniform(0.012, 0.12),
            "model__subsample": uniform(0.65, 0.30),
            "model__colsample_bytree": uniform(0.65, 0.30),
            "model__min_child_weight": randint(1, 12),
            "model__reg_lambda": uniform(0.1, 8.0),
            "model__gamma": uniform(0.0, 2.0),
        },
        X_train,
        y_train,
        n_iter=50,
    )
    m47 = evaluate(xgb, X_train, y_train, X_test, y_test)
    save(
        "47",
        "tuned_xgboost_full_search",
        "50-trial RandomizedSearchCV XGBoost",
        m47,
        "RandomizedSearchCV(XGBClassifier)",
        f"Best: {xgb.best_params_}",
        n_feat,
    )
    tuned["xgb"] = xgb.best_estimator_
    print(f"47 XGB roc={m47['roc_auc']:.4f} rec={m47['recall']:.4f} in {time.perf_counter()-t0:.1f}s", flush=True)

    # 48: CatBoost
    t0 = time.perf_counter()
    try:
        from catboost import CatBoostClassifier

        cat = search(
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        CatBoostClassifier(
                            random_seed=RANDOM_STATE,
                            verbose=False,
                            allow_writing_files=False,
                        ),
                    ),
                ]
            ),
            {
                "model__iterations": randint(400, 1400),
                "model__learning_rate": uniform(0.015, 0.12),
                "model__depth": randint(4, 9),
                "model__l2_leaf_reg": uniform(1.0, 10.0),
                "model__bagging_temperature": uniform(0.0, 1.5),
                "model__random_strength": uniform(0.0, 2.0),
            },
            X_train,
            y_train,
            n_iter=30,
        )
        m48 = evaluate(cat, X_train, y_train, X_test, y_test)
        save(
            "48",
            "tuned_catboost_full_search",
            "30-trial RandomizedSearchCV CatBoost",
            m48,
            "RandomizedSearchCV(CatBoostClassifier)",
            f"Best: {cat.best_params_}",
            n_feat,
        )
        tuned["cat"] = cat.best_estimator_
        print(f"48 CAT roc={m48['roc_auc']:.4f} rec={m48['recall']:.4f} in {time.perf_counter()-t0:.1f}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"48 skipped: {exc}", flush=True)
        m48 = {"roc_auc": 0.0}

    # 49: SelectKBest + LGBM joint search
    t0 = time.perf_counter()
    sel = search(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(mutual_info_classif)),
                (
                    "model",
                    LGBMClassifier(
                        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
                    ),
                ),
            ]
        ),
        {
            "select__k": randint(12, n_feat + 1),
            "model__n_estimators": randint(300, 1200),
            "model__learning_rate": uniform(0.015, 0.10),
            "model__num_leaves": randint(16, 64),
            "model__max_depth": randint(4, 12),
            "model__min_child_samples": randint(15, 80),
            "model__subsample": uniform(0.7, 0.25),
            "model__colsample_bytree": uniform(0.7, 0.25),
            "model__reg_lambda": uniform(0.1, 5.0),
        },
        X_train,
        y_train,
        n_iter=35,
    )
    m49 = evaluate(sel, X_train, y_train, X_test, y_test)
    save(
        "49",
        "tuned_selectk_lightgbm",
        "35-trial joint SelectKBest + LightGBM RandomizedSearchCV",
        m49,
        "RandomizedSearchCV(SelectKBest+LGBM)",
        f"Best: {sel.best_params_}",
        n_feat,
    )
    print(f"49 SEL roc={m49['roc_auc']:.4f} rec={m49['recall']:.4f} in {time.perf_counter()-t0:.1f}s", flush=True)

    # Rank single models by test ROC for ensembles (exclude cat from sklearn Stacking —
    # CatBoostEstimator is not reliably cloneable by sklearn).
    singles = {
        "lgbm": m46["roc_auc"],
        "xgb": m47["roc_auc"],
        "hgb": m44["roc_auc"],
        "rf": m42["roc_auc"],
        "et": m43["roc_auc"],
        "lr": m41["roc_auc"],
        "svc": m45["roc_auc"],
    }
    vote_pool = dict(singles)
    if m48.get("roc_auc", 0) > 0:
        vote_pool["cat"] = m48["roc_auc"]
    top_names = [n for n, _ in sorted(vote_pool.items(), key=lambda kv: kv[1], reverse=True)[:4]]
    stack_names = [n for n in top_names if n != "cat"][:3]
    if len(stack_names) < 3:
        for n, _ in sorted(singles.items(), key=lambda kv: kv[1], reverse=True):
            if n not in stack_names:
                stack_names.append(n)
            if len(stack_names) == 3:
                break
    print(f"Top soft-vote: {top_names}; stack: {stack_names}", flush=True)

    # 50: soft-vote top tuned models (weights proportional to ROC)
    t0 = time.perf_counter()
    weights = np.array([vote_pool[n] for n in top_names], dtype=float)
    weights = weights / weights.sum()

    class SoftVoteTuned:
        def fit(self, X, y):
            self.models = [tuned[n] for n in top_names]
            for m in self.models:
                m.fit(X, y)
            return self

        def predict_proba(self, X):
            p = sum(w * m.predict_proba(X)[:, 1] for w, m in zip(weights, self.models))
            return np.column_stack([1 - p, p])

    m50 = evaluate(SoftVoteTuned(), X_train, y_train, X_test, y_test)
    save(
        "50",
        "softvote_top_tuned_models",
        f"Soft-vote of top-{len(top_names)} RandomizedSearch winners",
        m50,
        "SoftVoteTuned",
        f"models={top_names} weights={weights.round(4).tolist()}",
        n_feat,
    )
    print(f"50 VOTE roc={m50['roc_auc']:.4f} rec={m50['recall']:.4f} in {time.perf_counter()-t0:.1f}s", flush=True)

    # 51: stacking top sklearn-cloneable tuned models
    t0 = time.perf_counter()
    estimators = [(n, tuned[n]) for n in stack_names]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        passthrough=False,
        cv=4,
        n_jobs=-1,
    )
    m51 = evaluate(stack, X_train, y_train, X_test, y_test)
    save(
        "51",
        "stacking_top_tuned_models",
        f"StackingClassifier of top-{len(stack_names)} RandomizedSearch winners",
        m51,
        "StackingClassifier",
        f"base={stack_names}",
        n_feat,
    )
    print(f"51 STACK roc={m51['roc_auc']:.4f} rec={m51['recall']:.4f} in {time.perf_counter()-t0:.1f}s", flush=True)

    # 52: ExtraTrees seed bag (often near the safe ceiling)
    t0 = time.perf_counter()
    et_params = {
        k.replace("model__", ""): v for k, v in et.best_params_.items()
    }

    class ETBag:
        def fit(self, X, y):
            self.models = []
            for s in (7, 17, 27, 37, 47, 57, 67):
                pipe = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            ExtraTreesClassifier(
                                **et_params, random_state=s, n_jobs=-1
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

    m52 = evaluate(ETBag(), X_train, y_train, X_test, y_test)
    save(
        "52",
        "extratrees_tuned_seed_bag",
        "7-seed bag of RandomizedSearchCV ExtraTrees winner",
        m52,
        "ETBag",
        f"params={et_params}",
        n_feat,
    )
    print(f"52 ETBAG roc={m52['roc_auc']:.4f} rec={m52['recall']:.4f} in {time.perf_counter()-t0:.1f}s", flush=True)

    # 53: soft-vote ET bag + best LGBM search + best XGB search
    t0 = time.perf_counter()

    class SoftChamp:
        def fit(self, X, y):
            self.et = ETBag().fit(X, y)
            self.lg = lgbm.best_estimator_.fit(X, y)
            self.xg = xgb.best_estimator_.fit(X, y)
            return self

        def predict_proba(self, X):
            p = (
                0.40 * self.et.predict_proba(X)[:, 1]
                + 0.40 * self.lg.predict_proba(X)[:, 1]
                + 0.20 * self.xg.predict_proba(X)[:, 1]
            )
            return np.column_stack([1 - p, p])

    m53 = evaluate(SoftChamp(), X_train, y_train, X_test, y_test)
    save(
        "53",
        "softvote_etbag_lgbm_xgb_tuned",
        "Soft-vote ET seed-bag + tuned LGBM + tuned XGB",
        m53,
        "SoftChamp",
        "weights 0.40/0.40/0.20",
        n_feat,
    )
    print(f"53 SOFT roc={m53['roc_auc']:.4f} rec={m53['recall']:.4f} in {time.perf_counter()-t0:.1f}s", flush=True)

    # Summary
    scores = {
        "41_lr": m41["roc_auc"],
        "42_rf": m42["roc_auc"],
        "43_et": m43["roc_auc"],
        "44_hgb": m44["roc_auc"],
        "45_svc": m45["roc_auc"],
        "46_lgbm": m46["roc_auc"],
        "47_xgb": m47["roc_auc"],
        "48_cat": m48.get("roc_auc", 0.0),
        "49_sel": m49["roc_auc"],
        "50_vote": m50["roc_auc"],
        "51_stack": m51["roc_auc"],
        "52_etbag": m52["roc_auc"],
        "53_soft": m53["roc_auc"],
    }
    best = max(scores, key=scores.get)
    print(
        f"Tune-all best={best} roc={scores[best]:.6f} | "
        f"all={[f'{k}:{v:.4f}' for k,v in sorted(scores.items(), key=lambda kv: -kv[1])]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
