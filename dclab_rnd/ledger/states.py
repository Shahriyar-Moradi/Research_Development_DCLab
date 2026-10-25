"""The states of table-level decisions, built one way (29.2, the review: a student trained on the homework must read
the same fields production records): the task's, the split's, a cases table's and a column's cleaning step, each from
a table's profile (``studio.data.profile_table``: names, counts and bands; never a value). Production capture and the
homework generator (``decisions_gen.decide_v2``) both call these.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from dclab_core.bands import DTYPE_KINDS, VALUE_PATTERNS, cardinality_band, name_tokens, ratio_band, rows_band

CASE_NAME = re.compile(r"case|fraud|confirm|known|label", re.I)
REASON_NAME = re.compile(r"reason|type|kind|category|cause", re.I)


def _columns(profile: Optional[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return list((profile or {}).get("columns") or [])


def column_bands(col: Optional[Mapping[str, Any]], name: Optional[str] = None) -> dict[str, Any]:
    """One column's bands from its profile entry, its name's tokens."""
    col = col or {}
    name = str(col.get("name") or name or "")
    state = {"column": name[:200] or None, "name_tokens": name_tokens(name) or None,
             "dtype": col.get("dtype_band") if col.get("dtype_band") in DTYPE_KINDS else None,
             "nulls": ratio_band(col.get("missing_rate")) if col.get("missing_rate") is not None else None,
             "uniqueness": ratio_band(col.get("unique_ratio")) if col.get("unique_ratio") is not None else None,
             "cardinality": cardinality_band(col.get("unique")) if col.get("unique") is not None else None,
             "value_pattern": col.get("value_pattern") if col.get("value_pattern") in VALUE_PATTERNS else None}
    return {k: v for k, v in state.items() if v is not None}


def _base(profile: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    cols = _columns(profile)
    return {k: v for k, v in {"rows": rows_band((profile or {}).get("rows")), "columns": len(cols) or None}.items() if v is not None}


def task_state(profile: Optional[Mapping[str, Any]], target: str) -> dict[str, Any]:
    """The outcome column's bands, how many values it takes, whether they are whole numbers."""
    col = next((c for c in _columns(profile) if c.get("name") == target), None)
    out = {**_base(profile), **column_bands(col, target)}
    if col is not None and col.get("unique") is not None:
        out["classes"] = int(col["unique"])
        out["integer_valued"] = col.get("dtype_band") == "integer"
    return out


def split_state(profile: Optional[Mapping[str, Any]], time_column: Optional[str], group_column: Optional[str]) -> dict[str, Any]:
    """The columns the split would read (declared or proposed), and what the table holds that could change it: how many
    columns look like times or identifiers, and how much the most repeated identifier repeats."""
    cols = _columns(profile)
    ids = [c for c in cols if c.get("id_like") or c.get("name_id_like")]
    repeats = [1.0 - float(c.get("unique_ratio") or 0.0) for c in ids if c.get("unique_ratio") is not None]
    return {**_base(profile), "has_time": bool(time_column), "has_groups": bool(group_column),
            "date_columns": sum(bool(c.get("time_like") or c.get("dtype_band") == "datetime") for c in cols),
            "id_columns": len(ids), "repeat_share": round(max(repeats), 4) if repeats else 0.0}


def cases_state(profile: Optional[Mapping[str, Any]], filename: str = "") -> dict[str, Any]:
    """What a cases table looks like: a key different on every row, a column naming the kind of case, a column in one
    state, a file name that says so."""
    cols = _columns(profile)
    return {**_base(profile), "key_unique": any(c.get("id_like") and c.get("name_id_like") for c in cols),
            "reason_like": any(REASON_NAME.search(str(c.get("name"))) for c in cols),
            "status_constant": any(c.get("unique") == 1 for c in cols), "name_says_cases": bool(CASE_NAME.search(filename or ""))}


def cleaning_state(profile: Optional[Mapping[str, Any]], column: str) -> dict[str, Any]:
    """A raw column's bands before cleaning, with the table's size."""
    col = next((c for c in _columns(profile) if c.get("name") == column), None)
    return {**column_bands(col, column), **({"rows": rows_band((profile or {}).get("rows"))} if (profile or {}).get("rows") else {})}


def tuning_state(tuning: Mapping[str, Any], evidence: Mapping[str, Any], metric: Optional[str]) -> dict[str, Any]:
    """``tuning.accept``'s state (2026-10-25): what judges the best candidate's gain on the folds. The best of several
    candidates on the same folds is partly luck: fresh draws of self-play's replicas mostly said "do not keep it" where
    the rule's 0.001 margin kept it. So beside the gain (``tuning_gain``): the baseline's fold mean and spread (``cv_mean``,
    ``cv_std``), the gain in units of that spread (``margin``), how many candidates were tried (``rounds``), the model
    family (``kind``) and the metric. Scores and counts, in existing state fields; never a value of the table."""
    def number(v: Any) -> Optional[float]:
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else None

    gain = number(tuning.get("gain"))
    rows = [r for r in evidence.get("configuration_results") or [] if isinstance(r, Mapping)]
    stats = ((rows[0].get("metrics") or {}).get(metric) or {}) if rows and metric else {}  # the baseline: the first configuration
    mean, std = (number(stats.get("mean")), number(stats.get("std"))) if isinstance(stats, Mapping) else (None, None)
    code = lambda v: re.sub(r"[^a-z0-9_]+", "_", str(v or "").lower()).strip("_")[:60] or None  # noqa: E731
    state = {"tuning_gain": gain, "cv_mean": round(mean, 6) if mean is not None else None, "cv_std": round(std, 6) if std is not None else None,
             "margin": round(gain / std, 4) if gain is not None and std is not None and std > 1e-9 else None,  # a spread of 0 says nothing
             "rounds": max(len(rows) - 1, 0) if rows else None,
             "kind": code(evidence.get("model")), "metric": code(metric)}
    return {k: v for k, v in state.items() if v is not None}


HIGH_CARD = 50  # a category with more distinct values than this is high-cardinality (one-hot gets wide; trees and encoders differ)


def coarse_task(task: Any) -> Optional[str]:
    """The engine's task name (binary_imbalanced, timeseries_regression, text_tabular_binary...) as the ledger's task band."""
    name = str(task or "").lower()
    if "multiclass" in name:
        return "multiclass_classification"
    if "binary" in name or name in ("classification", "binary_classification"):
        return "binary_classification"
    if "regression" in name:
        return "regression"
    return None


def table_state(profile: Optional[Mapping[str, Any]], target: Optional[str] = None, task: Optional[str] = None,
                positive_rate: Optional[float] = None, usable_features: Optional[int] = None) -> dict[str, Any]:
    """A table at a glance (S7, "more ability"): what an expert reads to guess which model family will win before trying
    them all (``model.first``): its rows and columns, the features by kind (numbers, categories, text, dates; an identifier
    apart), how many categories have many values, how many columns have missing values, and the outcome's classes.
    Counts and bands of the profile, never a value; the outcome is not counted among the features."""
    cols = [c for c in _columns(profile) if c.get("name") != target]
    plain = [c for c in cols if not c.get("id_like")]
    kind = lambda c: c.get("kind")  # noqa: E731
    outcome = next((c for c in _columns(profile) if target is not None and c.get("name") == target), None)
    task_band = coarse_task(task)
    classified = kind(outcome) in ("categorical", "boolean") if outcome else False
    classified = classified or (task_band or "").endswith("classification")  # classes written as digits
    classes = outcome.get("unique") if outcome and classified else None
    state = {**_base(profile), "columns": len(cols) or None,
             "numeric_columns": sum(kind(c) == "numeric" for c in plain), "category_columns": sum(kind(c) in ("categorical", "boolean") for c in plain),
             "text_columns": sum(kind(c) == "text" for c in plain), "date_columns": sum(kind(c) == "datetime" for c in plain),
             "high_card_columns": sum(kind(c) == "categorical" and (c.get("unique") or 0) > HIGH_CARD for c in plain),
             "missing_columns": sum((c.get("missing_rate") or 0) > 0 for c in cols), "id_columns": len(cols) - len(plain),
             "classes": int(classes) if isinstance(classes, (int, float)) and classes <= HIGH_CARD else None, "task": task_band,
             "positive_rate": ratio_band(positive_rate) if isinstance(positive_rate, (int, float)) and 0 <= positive_rate <= 1 else None,
             "usable_features": int(usable_features) if isinstance(usable_features, int) and usable_features >= 0 else None}
    return {k: v for k, v in state.items() if v is not None}


__all__ = ["column_bands", "task_state", "split_state", "cases_state", "cleaning_state", "tuning_state", "table_state", "coarse_task"]
