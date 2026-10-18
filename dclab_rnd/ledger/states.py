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


__all__ = ["column_bands", "task_state", "split_state", "cases_state", "cleaning_state"]
