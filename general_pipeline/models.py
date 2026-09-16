"""Model factory for the General Tabular Pipeline.

Provides baseline and optimized configurations for:
- Logistic Regression
- Calibrated LinearSVC
- Random Forest
- Extra Trees
- HistGradientBoosting
- LightGBM
- XGBoost
- CatBoost
- Tabular Transformer (FT-Transformer)
- Stacking Ensemble
- Soft-Voting Blend
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .transformer import FTTransformerClassifier

RANDOM_STATE = 42

MODEL_REGISTRY = [
    "logistic_regression",
    "linear_svc",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
    "catboost",
    "tabular_transformer",
    "stacking_ensemble",
    "soft_vote_blend",
]


class SoftVotingClassifier:
    """Soft-voting probability blender."""

    def __init__(self, estimators: List[Tuple[str, Any]], weights: Optional[List[float]] = None):
        self.estimators = estimators
        if weights is None:
            self.weights = np.ones(len(estimators)) / len(estimators)
        else:
            w = np.asarray(weights, dtype=float)
            self.weights = w / w.sum()
        self.fitted_models = []

    def fit(self, X, y):
        self.fitted_models = []
        for name, model in self.estimators:
            m = model.fit(X, y)
            self.fitted_models.append(m)
        return self

    def predict_proba(self, X):
        probs = [m.predict_proba(X)[:, 1] for m in self.fitted_models]
        p = sum(w * prob for w, prob in zip(self.weights, probs))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)


def make_pipeline(model: Any, with_scaler: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if with_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def get_model(
    model_name: str,
    optimization: str = "baseline",
    mode: str = "safe",
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
) -> Any:
    """Instantiate a model given its name, optimization level ('baseline' or 'optimized'), and mode ('safe' or 'unsafe')."""
    is_opt = (optimization == "optimized")

    if model_name == "logistic_regression":
        if is_opt:
            clf = LogisticRegression(C=1.2, penalty="l2", solver="saga", max_iter=4000, random_state=RANDOM_STATE)
        else:
            clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        return make_pipeline(clf, with_scaler=True)

    elif model_name == "linear_svc":
        if is_opt:
            base_svc = LinearSVC(C=0.5, max_iter=6000, dual="auto", random_state=RANDOM_STATE)
        else:
            base_svc = LinearSVC(max_iter=3000, dual="auto", random_state=RANDOM_STATE)
        calibrated = CalibratedClassifierCV(base_svc, method="sigmoid", cv=3)
        return make_pipeline(calibrated, with_scaler=True)

    elif model_name == "random_forest":
        if is_opt:
            clf = RandomForestClassifier(
                n_estimators=800,
                max_depth=16,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        return make_pipeline(clf)

    elif model_name == "extra_trees":
        if is_opt:
            # Proven optimal ExtraTrees parameters from 43/52
            clf = ExtraTreesClassifier(
                n_estimators=800,
                max_depth=14,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            clf = ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        return make_pipeline(clf)

    elif model_name == "hist_gradient_boosting":
        if is_opt:
            clf = HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=600,
                max_leaf_nodes=35,
                min_samples_leaf=30,
                l2_regularization=1.5,
                random_state=RANDOM_STATE,
            )
        else:
            clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        return make_pipeline(clf)

    elif model_name == "lightgbm":
        if is_opt:
            if mode == "safe":
                # Proven champion params from 09/21
                clf = LGBMClassifier(
                    n_estimators=313,
                    learning_rate=0.0166,
                    num_leaves=20,
                    max_depth=11,
                    min_child_samples=69,
                    subsample=0.7308,
                    colsample_bytree=0.8077,
                    reg_lambda=0.2277,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                )
            else:
                # Unsafe optimized LightGBM
                clf = LGBMClassifier(
                    n_estimators=450,
                    learning_rate=0.02,
                    num_leaves=31,
                    max_depth=9,
                    min_child_samples=40,
                    subsample=0.80,
                    colsample_bytree=0.80,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                )
        else:
            clf = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
        return make_pipeline(clf)

    elif model_name == "xgboost":
        if is_opt:
            clf = XGBClassifier(
                n_estimators=750,
                learning_rate=0.025,
                max_depth=6,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.80,
                reg_lambda=2.5,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            clf = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
        return make_pipeline(clf)

    elif model_name == "catboost":
        if is_opt:
            clf = CatBoostClassifier(
                iterations=900,
                learning_rate=0.035,
                depth=6,
                l2_leaf_reg=4.0,
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            )
        else:
            clf = CatBoostClassifier(
                iterations=400,
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            )
        return make_pipeline(clf)

    elif model_name == "tabular_transformer":
        if is_opt:
            # Tuned FT-Transformer: deeper attention blocks, larger embedding dim, careful learning rate
            return FTTransformerClassifier(
                learning_rate=7e-4,
                num_heads=4,
                num_attn_blocks=3,
                input_embed_dim=48,
                attn_dropout=0.1,
                ff_dropout=0.1,
                batch_size=256,
                max_epochs=12,
                accelerator="cpu",
                random_state=RANDOM_STATE,
                categorical_cols=cat_cols,
                continuous_cols=num_cols,
            )
        else:
            # Baseline FT-Transformer
            return FTTransformerClassifier(
                learning_rate=1e-3,
                num_heads=4,
                num_attn_blocks=2,
                input_embed_dim=32,
                batch_size=256,
                max_epochs=6,
                accelerator="cpu",
                random_state=RANDOM_STATE,
                categorical_cols=cat_cols,
                continuous_cols=num_cols,
            )

    elif model_name == "stacking_ensemble":
        # Tuned ET + LGBM + XGB stacked with Logistic Regression
        et = get_model("extra_trees", optimization="optimized", mode=mode)
        lgbm = get_model("lightgbm", optimization="optimized", mode=mode)
        xgb = get_model("xgboost", optimization="optimized", mode=mode)
        stack = StackingClassifier(
            estimators=[("et", et), ("lgbm", lgbm), ("xgb", xgb)],
            final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            stack_method="predict_proba",
            cv=4,
            n_jobs=-1,
        )
        return stack

    elif model_name == "soft_vote_blend":
        # Champion blend: ExtraTrees + LightGBM + XGBoost
        et = get_model("extra_trees", optimization="optimized", mode=mode)
        lgbm = get_model("lightgbm", optimization="optimized", mode=mode)
        xgb = get_model("xgboost", optimization="optimized", mode=mode)
        blend = SoftVotingClassifier(
            estimators=[("et", et), ("lgbm", lgbm), ("xgb", xgb)],
            weights=[0.40, 0.40, 0.20],
        )
        return blend

    else:
        raise ValueError(f"Unknown model_name: {model_name}. Supported: {MODEL_REGISTRY}")
