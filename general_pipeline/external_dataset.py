"""Load cached external datasets into locked train/test splits for the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "external_data"
RANDOM_STATE = 42


def list_available_datasets() -> list[str]:
    if not DATA_DIR.exists():
        return []
    keys = []
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_dir() and (p / "X.parquet").exists() and (p / "y.parquet").exists():
            keys.append(p.name)
    return keys


def load_external_dataset(
    key: str,
    *,
    test_size: float = 0.2,
    max_rows: int | None = 20000,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str], dict]:
    """Return (X_train, y_train, X_test, y_test, cat_cols, num_cols, meta).

    All features are numeric (categoricals factorized at download time), so cat_cols is empty.
    """
    ds_dir = DATA_DIR / key
    if not (ds_dir / "X.parquet").exists():
        raise FileNotFoundError(f"Dataset '{key}' not found under {DATA_DIR}. Run download_external_datasets.py first.")

    X = pd.read_parquet(ds_dir / "X.parquet")
    y = pd.read_parquet(ds_dir / "y.parquet")["target"].astype(int)
    meta_path = ds_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"key": key}

    if max_rows is not None and len(X) > max_rows:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=max_rows,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        meta["subsampled_to"] = max_rows

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    num_cols = list(X_train.columns)
    cat_cols: list[str] = []
    meta.update(
        {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "pos_rate_train": float(y_train.mean()),
            "pos_rate_test": float(y_test.mean()),
        }
    )
    return X_train, y_train, X_test, y_test, cat_cols, num_cols, meta
