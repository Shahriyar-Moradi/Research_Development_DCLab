"""The real score of an arm's choices: one fixed model fitted on the visible table with the arm's features, scored on the
future set (leaks blank). The same model for every arm, so the score compares choices, not modelling skill."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

SEED = 0
MAX_LEVELS = 50  # a text column with more distinct values than this (free text, an id) is not one-hot encoded: it is left out


def _matrix(frame: pd.DataFrame, features: list[str], columns: Optional[list[str]] = None) -> pd.DataFrame:
    parts = {}
    for name in features:
        col = frame[name]
        if pd.api.types.is_datetime64_any_dtype(col):
            parts[name] = (col - pd.Timestamp("1970-01-01")).dt.total_seconds()  # a time as seconds; a blank stays blank
        elif pd.api.types.is_bool_dtype(col) or pd.api.types.is_numeric_dtype(col):
            parts[name] = pd.to_numeric(col, errors="coerce").astype(float)
        else:
            parts[name] = col.astype("string")
    times = [n for n in features if pd.api.types.is_datetime64_any_dtype(frame[n])]
    for i, a in enumerate(times):  # how a person uses two times (delivered minus ordered): a tree cannot subtract by itself
        for b in times[i + 1:]:
            parts[f"{a}-{b}"] = (frame[a] - frame[b]).dt.total_seconds() / 60.0
    X = pd.DataFrame(parts, index=frame.index)
    text = [c for c in X.columns if X[c].dtype == "string"]
    wide = [c for c in text if X[c].nunique(dropna=True) > MAX_LEVELS]
    X = pd.get_dummies(X.drop(columns=wide), columns=[c for c in text if c not in wide], dummy_na=True)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X.astype(float)


def real_score(case: Any, features: list[str]) -> Optional[float]:
    """None when the arm named no usable feature (nothing to fit)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import f1_score, mean_absolute_error

    used = [f for f in features if f in case.table.columns and f != case.target]
    if not used:
        return None
    train = case.table.dropna(subset=[case.target])
    future = case.future.dropna(subset=[case.target])
    X = _matrix(train, used)
    Xf = _matrix(future, used, list(X.columns))
    if case.task == "regression":
        model = HistGradientBoostingRegressor(random_state=SEED).fit(X, train[case.target].astype(float))
        return float(mean_absolute_error(future[case.target].astype(float), model.predict(Xf)))
    model = HistGradientBoostingClassifier(random_state=SEED).fit(X, train[case.target].astype(str))
    return float(f1_score(future[case.target].astype(str), model.predict(Xf), average="macro"))
