"""Dataset utilities for the General Tabular Pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HYPERACK_DIR = ROOT / "hyperack_exp"
sys.path.insert(0, str(HYPERACK_DIR))

from shared.protocol import (  # noqa: E402
    TARGET,
    load_clean_df,
    make_xy,
    split_frame,
)

CATEGORICAL_COLS = [
    "deliverey_category_id",
    "weekday",
    "time_bucket",
    "is_rush_hour",
    "is_weekend",
]


def load_dataset(mode: str = "safe") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str]]:
    """Load train and test splits for either 'safe' or 'unsafe' feature mode.

    Args:
        mode: 'safe' (pre-decision only, 25 features) or 'unsafe' (with final fares, 34 features).

    Returns:
        (X_train, y_train, X_test, y_test, cat_cols, num_cols)
    """
    train_df, test_df = split_frame(load_clean_df())
    include_final = (mode == "unsafe")

    X_train, y_train = make_xy(
        train_df,
        include_final_fares=include_final,
        pricing=True,
        time_features=True,
        geo=True,
    )
    X_test, y_test = make_xy(
        test_df,
        include_final_fares=include_final,
        pricing=True,
        time_features=True,
        geo=True,
    )

    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    return X_train, y_train, X_test, y_test, cat_cols, num_cols
