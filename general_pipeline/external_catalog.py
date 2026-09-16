"""Catalog of public tabular binary-classification datasets similar to HyperAck.

Selection criteria (aligned with HyperAck ~11k rows, mixed features, real-world ops/risk):
- Tabular classification (binary or easily binarized)
- ~500–50k instances
- Mixed numeric/categorical preferred
- Well-known UCI benchmarks used in tabular ML literature
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    uci_id: Optional[int]
    source: str
    description: str
    url: str


DATASET_CATALOG: list[DatasetSpec] = [
    DatasetSpec(
        key="adult",
        name="Adult (Census Income)",
        uci_id=2,
        source="UCI",
        description="Predict income >$50K from census demographics (mixed categorical/numeric).",
        url="https://archive.ics.uci.edu/dataset/2/adult",
    ),
    DatasetSpec(
        key="bank_marketing",
        name="Bank Marketing",
        uci_id=222,
        source="UCI",
        description="Predict term-deposit subscription from phone marketing campaign data.",
        url="https://archive.ics.uci.edu/dataset/222/bank+marketing",
    ),
    DatasetSpec(
        key="breast_cancer",
        name="Breast Cancer Wisconsin (Diagnostic)",
        uci_id=17,
        source="UCI",
        description="Binary diagnosis (malignant/benign) from cell nuclei measurements.",
        url="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
    ),
    DatasetSpec(
        key="heart_disease",
        name="Heart Disease",
        uci_id=45,
        source="UCI",
        description="Predict presence of heart disease from clinical attributes.",
        url="https://archive.ics.uci.edu/dataset/45/heart+disease",
    ),
    DatasetSpec(
        key="credit_default",
        name="Default of Credit Card Clients",
        uci_id=350,
        source="UCI",
        description="Predict next-month credit-card default from payment history.",
        url="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
    ),
    DatasetSpec(
        key="german_credit",
        name="Statlog German Credit",
        uci_id=144,
        source="UCI",
        description="Classify credit applicants as good/bad risk.",
        url="https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
    ),
    DatasetSpec(
        key="mushroom",
        name="Mushroom",
        uci_id=73,
        source="UCI",
        description="Classify mushrooms as edible vs poisonous from categorical traits.",
        url="https://archive.ics.uci.edu/dataset/73/mushroom",
    ),
    DatasetSpec(
        key="spambase",
        name="Spambase",
        uci_id=94,
        source="UCI",
        description="Detect spam email from word/character frequency features.",
        url="https://archive.ics.uci.edu/dataset/94/spambase",
    ),
    DatasetSpec(
        key="online_shoppers",
        name="Online Shoppers Purchasing Intention",
        uci_id=468,
        source="UCI",
        description="Predict whether a session ends in a purchase (e-commerce intent).",
        url="https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset",
    ),
    DatasetSpec(
        key="wine_quality",
        name="Wine Quality (Binary)",
        uci_id=186,
        source="UCI",
        description="Physicochemical wine tests; quality binarized as high (>=6) vs low.",
        url="https://archive.ics.uci.edu/dataset/186/wine+quality",
    ),
]


def _to_binary_series(y: pd.Series) -> pd.Series:
    """Map arbitrary labels to {0,1}; for multiclass keep the majority vs rest of top-2 if needed."""
    if y.dtype == bool:
        return y.astype(int)
    # Numeric already binary
    vals = pd.Series(y).copy()
    if pd.api.types.is_numeric_dtype(vals):
        uniq = sorted(vals.dropna().unique().tolist())
        if set(uniq).issubset({0, 1}):
            return vals.astype(int)
        if len(uniq) == 2:
            mapping = {uniq[0]: 0, uniq[1]: 1}
            return vals.map(mapping).astype(int)
        # Heart disease style: disease if >0
        if all(u >= 0 for u in uniq) and max(uniq) > 1:
            return (vals > 0).astype(int)
        # Wine quality style: threshold at median / 6
        return (vals >= 6).astype(int)

    # String / object labels
    cleaned = vals.astype(str).str.strip().str.lower()
    positive_tokens = {
        "yes",
        "y",
        "true",
        "1",
        ">50k",
        ">50k.",
        "malignant",
        "m",
        "spam",
        "poisonous",
        "p",
        "bad",
        "default",
        "dropout",
        "high",
    }
    uniq = cleaned.dropna().unique().tolist()
    if len(uniq) == 2:
        # Prefer known positive token; else lexicographic second as positive
        if any(u in positive_tokens for u in uniq):
            return cleaned.isin(positive_tokens).astype(int)
        ordered = sorted(uniq)
        return (cleaned == ordered[1]).astype(int)
    # Multiclass string: majority class as 0, rest as 1 is unstable; use top-2 only
    top2 = cleaned.value_counts().index[:2].tolist()
    mask = cleaned.isin(top2)
    return cleaned.where(mask).map({top2[0]: 0, top2[1]: 1}).astype("float").astype("Int64")


def binarize_target(y: pd.Series, key: str) -> pd.Series:
    """Dataset-specific target binarization with safe fallbacks."""
    s = y.copy()
    if key == "adult":
        return (s.astype(str).str.replace(".", "", regex=False).str.strip() == ">50K").astype(int)
    if key == "bank_marketing":
        return (s.astype(str).str.strip().str.lower() == "yes").astype(int)
    if key == "breast_cancer":
        return (s.astype(str).str.strip().str.upper().isin(["M", "MALIGNANT", "1"])).astype(int)
    if key == "heart_disease":
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int)
    if key == "credit_default":
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    if key == "german_credit":
        # UCI often encodes 1=good, 2=bad
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() > 0.9:
            return (num == 2).astype(int)
        return (s.astype(str).str.lower().isin(["bad", "2"])).astype(int)
    if key == "mushroom":
        return (s.astype(str).str.strip().str.lower().isin(["p", "poisonous"])).astype(int)
    if key == "spambase":
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    if key == "online_shoppers":
        if s.dtype == bool:
            return s.astype(int)
        return (s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])).astype(int)
    if key == "wine_quality":
        return (pd.to_numeric(s, errors="coerce") >= 6).astype(int)
    return _to_binary_series(s)


def prepare_xy(X: pd.DataFrame, y: pd.Series, key: str) -> tuple[pd.DataFrame, pd.Series]:
    """Clean features/target into numeric matrix suitable for sklearn pipelines."""
    y_bin = binarize_target(y, key)
    df = X.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    # Drop ID-like columns
    drop_cols = [c for c in df.columns if c.lower() in {"id", "unnamed:_0", "row_id"}]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Convert object columns via factorize (train-only encoding applied later at split time —
    # here we only coerce numerics; categoricals stay as codes after global factorize for cache simplicity)
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # Stable codes including NaN as -1
            codes, _ = pd.factorize(df[col].astype(str).str.strip(), use_na_sentinel=True)
            df[col] = codes.astype(float)

    df = df.replace([np.inf, -np.inf], np.nan)
    mask = y_bin.notna()
    df = df.loc[mask].reset_index(drop=True)
    y_bin = y_bin.loc[mask].astype(int).reset_index(drop=True)
    return df, y_bin
