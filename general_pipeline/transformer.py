"""Scikit-Learn compatible Tabular Transformer Classifier based on FT-Transformer."""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from typing import Any, List, Optional

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_tabular").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

TARGET_COL = "__target__"


class FTTransformerClassifier(BaseEstimator, ClassifierMixin):
    """FT-Transformer (Feature Tokenizer Transformer) for tabular classification."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        num_heads: int = 4,
        num_attn_blocks: int = 2,
        input_embed_dim: int = 32,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        batch_size: int = 256,
        max_epochs: int = 10,
        accelerator: str = "cpu",
        random_state: int = 42,
        categorical_cols: Optional[List[str]] = None,
        continuous_cols: Optional[List[str]] = None,
    ):
        self.learning_rate = learning_rate
        self.num_heads = num_heads
        self.num_attn_blocks = num_attn_blocks
        self.input_embed_dim = input_embed_dim
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.random_state = random_state
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.model_ = None
        self.classes_ = None

    def fit(self, X: pd.DataFrame, y: Any):
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
        from pytorch_tabular.models import FTTransformerConfig

        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        if not isinstance(X, pd.DataFrame):
            cols = [f"feat_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=cols)
        else:
            X_df = X.copy()

        # Identify categorical and continuous columns
        if self.categorical_cols is not None:
            cat_cols = [c for c in self.categorical_cols if c in X_df.columns]
        else:
            cat_cols = [
                c
                for c in X_df.columns
                if X_df[c].dtype == "object"
                or X_df[c].nunique() < 10
                or c in ["deliverey_category_id", "weekday", "time_bucket", "is_rush_hour", "is_weekend"]
            ]

        if self.continuous_cols is not None:
            num_cols = [c for c in self.continuous_cols if c in X_df.columns]
        else:
            num_cols = [c for c in X_df.columns if c not in cat_cols]

        train_data = X_df.copy()
        train_data[TARGET_COL] = y_arr

        # Handle missing continuous values with median before feeding
        for col in num_cols:
            if train_data[col].isna().any():
                train_data[col] = train_data[col].fillna(train_data[col].median())

        for col in cat_cols:
            if train_data[col].isna().any():
                train_data[col] = train_data[col].fillna(-1)

        data_config = DataConfig(
            target=[TARGET_COL],
            continuous_cols=num_cols,
            categorical_cols=cat_cols,
        )

        trainer_config = TrainerConfig(
            auto_lr_find=False,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=1,
            checkpoints=None,
            early_stopping=None,
            progress_bar="none",
            seed=self.random_state,
        )

        optimizer_config = OptimizerConfig()

        model_config = FTTransformerConfig(
            task="classification",
            learning_rate=self.learning_rate,
            num_heads=self.num_heads,
            num_attn_blocks=self.num_attn_blocks,
            input_embed_dim=self.input_embed_dim,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
        )

        self.model_ = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            verbose=False,
        )

        self.model_.fit(train=train_data)
        self.cat_cols_ = cat_cols
        self.num_cols_ = num_cols
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")

        if not isinstance(X, pd.DataFrame):
            cols = [f"feat_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=cols)
        else:
            X_df = X.copy()

        # Handle NaNs identically
        for col in self.num_cols_:
            if col in X_df.columns and X_df[col].isna().any():
                X_df[col] = X_df[col].fillna(X_df[col].median())
        for col in self.cat_cols_:
            if col in X_df.columns and X_df[col].isna().any():
                X_df[col] = X_df[col].fillna(-1)

        preds = self.model_.predict(X_df)

        prob_1_col = f"{TARGET_COL}_1_probability"
        prob_0_col = f"{TARGET_COL}_0_probability"

        if prob_1_col in preds.columns:
            p1 = preds[prob_1_col].to_numpy()
            p0 = preds[prob_0_col].to_numpy() if prob_0_col in preds.columns else 1.0 - p1
        elif "prediction_probability" in preds.columns:
            p1 = preds["prediction_probability"].to_numpy()
            p0 = 1.0 - p1
        else:
            prob_cols = [c for c in preds.columns if "prob" in c.lower()]
            if len(prob_cols) >= 2:
                p0 = preds[prob_cols[0]].to_numpy()
                p1 = preds[prob_cols[1]].to_numpy()
            elif len(prob_cols) == 1:
                p1 = preds[prob_cols[0]].to_numpy()
                p0 = 1.0 - p1
            else:
                p1 = preds["prediction"].to_numpy().astype(float)
                p0 = 1.0 - p1

        return np.column_stack([p0, p1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
