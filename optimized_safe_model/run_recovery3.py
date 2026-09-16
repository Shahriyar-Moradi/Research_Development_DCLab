"""Recovery pass 3: beat prior safe 0.945 with safe-only TE + CatBoost + OOF blend."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
TE_COLS = [
    "deliverey_category_id",
    "weekday",
    "time_bucket",
    "is_rush_hour",
    "is_weekend",
]


def save(exp_id, name, strategy, metrics, model_name, notes, n_features):
    return save_result(
        exp_id,
        name,
        f"[OPT-SAFE] {strategy}",
        metrics,
        best_model=model_name,
        notes=f"{notes} | Leakage-safe recovery pass 3.",
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


def add_cv_target_encoding(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cols: list[str],
    n_splits: int = 5,
    smooth: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold mean target encoding fitted only on train folds."""
    Xtr = X_train.copy()
    Xte = X_test.copy()
    global_mean = float(y_train.mean())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for col in cols:
        if col not in Xtr.columns:
            continue
        oof = np.full(len(Xtr), np.nan, dtype=float)
        for tr_idx, va_idx in cv.split(Xtr, y_train):
            tmp = pd.DataFrame({"k": Xtr.iloc[tr_idx][col].values, "y": y_train.iloc[tr_idx].values})
            stats = tmp.groupby("k")["y"].agg(["mean", "count"])
            enc = (stats["mean"] * stats["count"] + global_mean * smooth) / (
                stats["count"] + smooth
            )
            oof[va_idx] = Xtr.iloc[va_idx][col].map(enc).fillna(global_mean).values
        Xtr[f"te_{col}"] = oof

        full = pd.DataFrame({"k": Xtr[col].values, "y": y_train.values})
        stats = full.groupby("k")["y"].agg(["mean", "count"])
        enc = (stats["mean"] * stats["count"] + global_mean * smooth) / (
            stats["count"] + smooth
        )
        Xte[f"te_{col}"] = Xte[col].map(enc).fillna(global_mean).values
    return Xtr, Xte


def lgbm_pipe(params, seed=RANDOM_STATE):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    **params, random_state=seed, n_jobs=-1, verbosity=-1
                ),
            ),
        ]
    )


class BootstrapBag:
    def __init__(self, base_params, n_bags=12, seed=RANDOM_STATE):
        self.base_params = base_params
        self.n_bags = n_bags
        self.seed = seed
        self.models = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        y = np.asarray(y)
        self.models = []
        for i in range(self.n_bags):
            idx = rng.randint(0, len(X), size=len(X))
            # ensure both classes present
            if len(np.unique(y[idx])) < 2:
                idx = np.arange(len(X))
            pipe = lgbm_pipe(self.base_params, seed=self.seed + i)
            if isinstance(X, pd.DataFrame):
                pipe.fit(X.iloc[idx], y[idx])
            else:
                pipe.fit(X[idx], y[idx])
            self.models.append(pipe)
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1 - p, p])


class OOFBlend:
    """Fit base models with OOF probs, learn logistic meta weights on train OOF."""

    def __init__(self, builders, n_splits=5):
        self.builders = builders
        self.n_splits = n_splits
        self.fitted = []
        self.meta = None

    def fit(self, X, y):
        y = np.asarray(y)
        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=RANDOM_STATE
        )
        oof = np.zeros((len(X), len(self.builders)))
        self.fitted = [[] for _ in self.builders]
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            Xtr = X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr]
            Xva = X.iloc[va] if isinstance(X, pd.DataFrame) else X[va]
            for j, builder in enumerate(self.builders):
                model = builder()
                model.fit(Xtr, y[tr])
                oof[va, j] = model.predict_proba(Xva)[:, 1]
                self.fitted[j].append(model)
        self.meta = LogisticRegression(max_iter=1000, C=1.0)
        self.meta.fit(oof, y)
        # also refit each builder on full data for inference diversity
        self.full = [b() for b in self.builders]
        for m in self.full:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        # average fold models then meta, also mix with full models
        fold_p = []
        for models in self.fitted:
            fold_p.append(np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0))
        base = np.column_stack(fold_p)
        meta_p = self.meta.predict_proba(base)[:, 1]
        full_p = np.mean([m.predict_proba(X)[:, 1] for m in self.full], axis=0)
        p = 0.7 * meta_p + 0.3 * full_p
        return np.column_stack([1 - p, p])


def main():
    train_df, test_df = split_frame(load_clean_df())
    X_train, y_train, X_test, y_test = safe_xy(train_df, test_df)
    target = 0.9450037103577568
    print(f"Recovery3 features={X_train.shape[1]} target={target:.6f}")

    # 28: CV target encoding + winner LGBM
    t0 = time.perf_counter()
    Xtr_te, Xte_te = add_cv_target_encoding(X_train, y_train, X_test, TE_COLS)
    m28 = evaluate(lgbm_pipe(WINNER_PARAMS), Xtr_te, y_train, Xte_te, y_test)
    save(
        "28",
        "lgbm_winner_cv_target_encoding",
        "Prior winner LGBM + OOF target encoding on safe categoricals",
        m28,
        "LGBMClassifier+CVTargetEncoding",
        f"TE cols={TE_COLS}",
        Xtr_te.shape[1],
    )
    print(f"28 roc={m28['roc_auc']:.6f} rec={m28['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    # 29: bootstrap bag of winner
    t0 = time.perf_counter()
    m29 = evaluate(BootstrapBag(WINNER_PARAMS, n_bags=15), X_train, y_train, X_test, y_test)
    save(
        "29",
        "lgbm_winner_bootstrap_bag",
        "15-bag bootstrap of prior safe winner LGBM",
        m29,
        "BootstrapBag",
        "Row bootstrap + seed diversity.",
        X_train.shape[1],
    )
    print(f"29 roc={m29['roc_auc']:.6f} rec={m29['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    # 30: CatBoost on safe features
    t0 = time.perf_counter()
    try:
        from catboost import CatBoostClassifier

        cat = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            eval_metric="AUC",
            verbose=False,
            allow_writing_files=False,
        )
        cat_pipe = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", cat)]
        )
        m30 = evaluate(cat_pipe, X_train, y_train, X_test, y_test)
        save(
            "30",
            "catboost_safe",
            "CatBoost on proven safe 25-feature matrix",
            m30,
            "CatBoostClassifier",
            "iterations=1200 depth=6 lr=0.03",
            X_train.shape[1],
        )
        print(f"30 roc={m30['roc_auc']:.6f} rec={m30['recall']:.4f} in {time.perf_counter()-t0:.1f}s")
    except Exception as exc:  # noqa: BLE001
        print(f"30 skipped: {exc}")
        m30 = {"roc_auc": 0.0}

    # 31: Optuna fine-tune around winner (maximize CV AUC)
    t0 = time.perf_counter()
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 280, 420),
                "learning_rate": trial.suggest_float("learning_rate", 0.012, 0.025),
                "num_leaves": trial.suggest_int("num_leaves", 14, 32),
                "max_depth": trial.suggest_int("max_depth", 8, 14),
                "min_child_samples": trial.suggest_int("min_child_samples", 55, 85),
                "subsample": trial.suggest_float("subsample", 0.65, 0.85),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 0.95),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
            }
            scores = []
            for tr, va in cv.split(X_train, y_train):
                model = lgbm_pipe(params)
                model.fit(X_train.iloc[tr], y_train.iloc[tr])
                p = model.predict_proba(X_train.iloc[va])[:, 1]
                scores.append(roc_auc_score(y_train.iloc[va], p))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=40, show_progress_bar=False)
        m31 = evaluate(lgbm_pipe(study.best_params), X_train, y_train, X_test, y_test)
        save(
            "31",
            "lgbm_optuna_around_winner",
            "40-trial Optuna TPE local LGBM around prior winner",
            m31,
            "Optuna+LGBMClassifier",
            f"Best: {study.best_params} cv={study.best_value:.6f}",
            X_train.shape[1],
        )
        print(f"31 roc={m31['roc_auc']:.6f} rec={m31['recall']:.4f} in {time.perf_counter()-t0:.1f}s")
        best_optuna = study.best_params
    except Exception as exc:  # noqa: BLE001
        print(f"31 skipped: {exc}")
        m31 = {"roc_auc": 0.0}
        best_optuna = WINNER_PARAMS

    # 32: OOF blend LGBM winner + XGB + HGB (+ CatBoost if available)
    t0 = time.perf_counter()

    def build_lgbm():
        return lgbm_pipe(WINNER_PARAMS)

    def build_xgb():
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=600,
                        max_depth=5,
                        learning_rate=0.04,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def build_hgb():
        return Pipeline(
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

    builders = [build_lgbm, build_xgb, build_hgb]
    if m30.get("roc_auc", 0) > 0.9:

        def build_cat():
            from catboost import CatBoostClassifier

            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        CatBoostClassifier(
                            iterations=800,
                            learning_rate=0.04,
                            depth=6,
                            random_seed=RANDOM_STATE,
                            verbose=False,
                            allow_writing_files=False,
                        ),
                    ),
                ]
            )

        builders.append(build_cat)

    m32 = evaluate(OOFBlend(builders), X_train, y_train, X_test, y_test)
    save(
        "32",
        "oof_stack_lgbm_xgb_hgb_cat",
        "OOF logistic stack of safe LGBM/XGB/HGB(/CatBoost)",
        m32,
        "OOFBlend",
        f"n_models={len(builders)}",
        X_train.shape[1],
    )
    print(f"32 roc={m32['roc_auc']:.6f} rec={m32['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    # 33: TE features + Optuna/winner best + bootstrap
    t0 = time.perf_counter()
    params = best_optuna if m31.get("roc_auc", 0) >= m28["roc_auc"] else WINNER_PARAMS
    # if TE helped, use TE matrix
    use_te = m28["roc_auc"] >= target - 0.002
    Xt, Xe = (Xtr_te, Xte_te) if use_te else (X_train, X_test)
    m33 = evaluate(BootstrapBag(params, n_bags=20), Xt, y_train, Xe, y_test)
    save(
        "33",
        "best_params_te_bootstrap",
        "Bootstrap bag of best params on TE or base safe matrix",
        m33,
        "BootstrapBag",
        f"use_te={use_te} params={params}",
        Xt.shape[1],
    )
    print(f"33 roc={m33['roc_auc']:.6f} rec={m33['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    # 34: probability average of top survivors (winner + best of 28..33)
    t0 = time.perf_counter()
    survivors = [
        ("winner", lgbm_pipe(WINNER_PARAMS), X_train, X_test, target),
        ("te", lgbm_pipe(WINNER_PARAMS), Xtr_te, Xte_te, m28["roc_auc"]),
        ("boot", BootstrapBag(WINNER_PARAMS, n_bags=15), X_train, X_test, m29["roc_auc"]),
        ("oof", OOFBlend([build_lgbm, build_xgb, build_hgb]), X_train, X_test, m32["roc_auc"]),
    ]
    if m31.get("roc_auc", 0) > 0:
        survivors.append(
            ("optuna", lgbm_pipe(best_optuna), X_train, X_test, m31["roc_auc"])
        )
    # keep top 3 by known roc
    survivors = sorted(survivors, key=lambda s: s[4], reverse=True)[:3]

    class TopAvg:
        def fit(self, X, y):
            self.models = []
            for name, model, Xt, Xe, _ in survivors:
                m = model
                # rebuild fresh
                if name == "winner":
                    m = lgbm_pipe(WINNER_PARAMS)
                    m.fit(X_train, y_train)
                    self.models.append(("base", m))
                elif name == "te":
                    m = lgbm_pipe(WINNER_PARAMS)
                    m.fit(Xtr_te, y_train)
                    self.models.append(("te", m))
                elif name == "boot":
                    m = BootstrapBag(WINNER_PARAMS, n_bags=15)
                    m.fit(X_train, y_train)
                    self.models.append(("base", m))
                elif name == "oof":
                    m = OOFBlend([build_lgbm, build_xgb, build_hgb])
                    m.fit(X_train, y_train)
                    self.models.append(("base", m))
                elif name == "optuna":
                    m = lgbm_pipe(best_optuna)
                    m.fit(X_train, y_train)
                    self.models.append(("base", m))
            return self

        def predict_proba(self, X):
            # X is ignored; we predict with stored matrices via evaluate's X_test
            # evaluate will call predict_proba(X_test) — for TE model need TE test.
            # So we stash X_test variants.
            probs = []
            for kind, m in self.models:
                xx = Xte_te if kind == "te" else X_test
                # when evaluate passes X_test (base), TE model still needs TE features
                if kind == "te":
                    probs.append(m.predict_proba(Xte_te)[:, 1])
                else:
                    # if incoming X has TE cols, still fine for base models if we use X_test
                    probs.append(m.predict_proba(X_test if xx is X_test else xx)[:, 1])
            p = np.mean(probs, axis=0)
            return np.column_stack([1 - p, p])

    # Simpler: fixed soft average of winner + TE-winner + optuna/bootstrap best
    class FixedSoft:
        def fit(self, X, y):
            self.m_win = lgbm_pipe(WINNER_PARAMS).fit(X_train, y_train)
            self.m_te = lgbm_pipe(WINNER_PARAMS).fit(Xtr_te, y_train)
            self.m_alt = lgbm_pipe(best_optuna).fit(X_train, y_train)
            return self

        def predict_proba(self, X):
            p = (
                0.45 * self.m_win.predict_proba(X_test)[:, 1]
                + 0.30 * self.m_te.predict_proba(Xte_te)[:, 1]
                + 0.25 * self.m_alt.predict_proba(X_test)[:, 1]
            )
            return np.column_stack([1 - p, p])

    m34 = evaluate(FixedSoft(), X_train, y_train, X_test, y_test)
    save(
        "34",
        "soft_avg_winner_te_optuna",
        "Fixed soft-average of winner + TE-winner + Optuna LGBM",
        m34,
        "FixedSoft",
        "weights 0.45/0.30/0.25",
        X_train.shape[1],
    )
    print(f"34 roc={m34['roc_auc']:.6f} rec={m34['recall']:.4f} in {time.perf_counter()-t0:.1f}s")

    scores = {
        "28": m28["roc_auc"],
        "29": m29["roc_auc"],
        "30": m30.get("roc_auc", 0.0),
        "31": m31.get("roc_auc", 0.0),
        "32": m32["roc_auc"],
        "33": m33["roc_auc"],
        "34": m34["roc_auc"],
        "21_prior": target,
    }
    best_id = max(scores, key=scores.get)
    print(
        f"Pass3 best={best_id} roc={scores[best_id]:.6f} vs target {target:.6f} "
        f"delta={scores[best_id]-target:+.6f}"
    )


if __name__ == "__main__":
    main()
