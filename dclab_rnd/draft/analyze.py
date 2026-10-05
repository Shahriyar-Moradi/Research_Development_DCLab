"""Descriptive analysis of a fresh table, for the Home chat and its charts.

Descriptive only: shape, distributions, top values, missingness, time coverage, and rank
correlations between columns that could be *features*. Relations with the target, or with a
column whose name reads like an outcome (the given ``target`` and every column ``profile_table``
flags ``target_name_like``), are left out on purpose: reading them on the full table, before the
train/test split, is how leakage starts (CLAUDE.md rules 1-2). That analysis waits for the
train-only EDA after the split.

The output is plain JSON (str, int, float, bool, None, list, dict). Floats keep 4 decimals, or
4 significant digits below 1, so small rates do not round to zero. Tables over 200,000 rows are
described from a seeded 200,000-row sample (``summary.sampled_rows``); shape, missingness and
duplicates always use every row.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from dclab_rnd.studio.data import profile_table
from dclab_rnd.tools import POST_OUTCOME_NAME

from .clean import category_code_columns
from .structure import parse_times, unique_names

SAMPLE_ROWS = 200_000
SEED = 0
MAX_BINS = 20
TOP_VALUES = 10
MAX_TIME_BUCKETS = 40
MAX_CORRELATED = 25
MIN_RHO = 0.3
TOP_PAIRS = 10
MISSING_WARNING = 0.30
NEAR_CONSTANT = 0.99
GAP_MIN_PER_PERIOD = 5
# (name, pandas period, label format); the finest period with at most 40 buckets is used
PERIODS = (("minute", "min", "%Y-%m-%d %H:%M"), ("hour", "h", "%Y-%m-%d %H:00"), ("day", "D", "%Y-%m-%d"),
           ("week", "W", "%Y-%m-%d"), ("month", "M", "%Y-%m"), ("year", "Y", "%Y"))
PERIOD_SECONDS = {"minute": 60, "hour": 3600, "day": 86400, "week": 7 * 86400}


def analyze(frame: pd.DataFrame, *, target: str | None = None, max_columns: int = 60) -> dict[str, Any]:
    """Summary, profile, per-column chart data, feature-feature correlations and highlights.

    Correlations never include ``target`` or any target candidate (see the module docstring).
    Only the first ``max_columns`` columns get per-column chart data.
    """
    names = [str(c) for c in frame.columns]
    if len(set(names)) < len(names):
        names = unique_names(names)
    view = frame.copy(deep=False)
    view.columns = names  # string, unique names: the profile and the charts address columns by name
    profile = profile_table(view)
    facts = {c["name"]: c for c in profile["columns"]}
    rows = int(len(view))
    sample = view.sample(n=SAMPLE_ROWS, random_state=SEED) if rows > SAMPLE_ROWS else view

    kinds: dict[str, int] = {}
    for column in profile["columns"]:
        kinds[column["kind"]] = kinds.get(column["kind"], 0) + 1
    cells = rows * len(names)
    summary: dict[str, Any] = {
        "rows": rows, "columns": len(names), "kinds": kinds,
        "missing_cell_rate": float(view.isna().sum().sum() / cells) if cells else 0.0,
        "duplicate_rows": int(round(profile["duplicate_row_rate"] * rows)),
        "memory_mb": profile["memory_mb"],
    }
    if sample is not view:
        summary["sampled_rows"] = int(len(sample))
    described = names[:max_columns]
    if len(described) < len(names):
        summary["columns_described"] = len(described)

    columns = [_describe(name, view[name], sample[name], facts[name]) for name in described]
    # The stated target, and columns whose names read like an outcome. Not every target candidate: profile_table
    # ranks up to eight eligible columns, which on a narrow table would leave nothing to describe.
    likely = [c["name"] for c in profile["columns"] if c.get("target_name_like")]
    excluded = sorted({*([target] if target else []), *likely})
    summary["correlation_excluded"] = excluded
    correlations = _correlations(sample, [c["name"] for c in profile["columns"] if c["kind"] == "numeric" and c["name"] not in excluded])
    highlights = _highlights(view, profile, columns, target, summary)
    return _json_safe({"summary": summary, "profile": profile, "columns": columns,
                       "correlations": correlations, "highlights": highlights})


# ---------------------------------------------------------------------- per column


def _describe(name: str, full: pd.Series, sample: pd.Series, fact: dict[str, Any]) -> dict[str, Any]:
    kind = fact["kind"]
    out: dict[str, Any] = {"name": name, "kind": kind, "dtype": str(full.dtype),
                           "missing_rate": float(full.isna().mean()) if len(full) else 0.0, "unique": fact["unique"]}
    present = sample.dropna()
    if kind == "numeric":
        out.update(_numeric(present))
    elif kind in ("categorical", "boolean"):
        out.update(_top(present))
    elif kind == "datetime":
        out["time"] = _time(present)
    elif kind == "text":
        text = present.astype(str)
        out["text"] = {"median_chars": float(text.str.len().median()) if len(text) else None,
                       "median_words": float(text.str.split().str.len().median()) if len(text) else None}
    if kind != "text" and len(present):  # share of all rows, so a mostly empty column is not "constant"
        out["top_share"] = float(present.astype(str).value_counts().iloc[0] / len(sample))
    return out


def _numeric(present: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(present, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"hist": None, "quantiles": None, "min": None, "max": None, "mean": None, "std": None}
    bins = int(min(MAX_BINS, max(1, len(np.unique(values)))))
    counts, edges = np.histogram(values, bins=bins)
    q = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])
    return {"hist": {"edges": edges.tolist(), "counts": counts.tolist()},
            "quantiles": dict(zip(("p01", "p25", "p50", "p75", "p99"), q.tolist())),
            "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None}


def _top(present: pd.Series) -> dict[str, Any]:
    counts = present.astype(str).value_counts()
    top = counts.head(TOP_VALUES)
    return {"top": [[str(value)[:80], int(count)] for value, count in top.items()],
            "other_count": int(counts.sum() - top.sum())}


def _time(present: pd.Series) -> dict[str, Any] | None:
    stamped = present if pd.api.types.is_datetime64_any_dtype(present) else parse_times(present.astype(str))
    stamped = stamped.dropna()
    if stamped.empty:
        return None
    times = stamped.dt.tz_localize(None) if stamped.dt.tz is not None else stamped  # calendar buckets use wall-clock time
    start, end = times.min(), times.max()
    for name, freq, label in PERIODS:
        if name in PERIOD_SECONDS:
            buckets = (end - start).total_seconds() / PERIOD_SECONDS[name] + 1
        elif name == "month":
            buckets = (end.year - start.year) * 12 + end.month - start.month + 1
        else:
            buckets = end.year - start.year + 1
        if buckets <= MAX_TIME_BUCKETS + 1 or name == "year":
            periods = times.dt.to_period(freq)
            span = pd.period_range(periods.min(), periods.max(), freq=freq)
            if len(span) > MAX_TIME_BUCKETS and name != "year":
                continue
            counts = periods.value_counts().reindex(span, fill_value=0)
            by = [[(p.start_time if name == "week" else p).strftime(label), int(n)] for p, n in counts.items()]
            return {"min": stamped.min().isoformat(), "max": stamped.max().isoformat(), "period": name, "by": by}
    return None


# ---------------------------------------------------------------------- relations between features


def _correlations(sample: pd.DataFrame, numeric: list[str]) -> list[dict[str, Any]]:
    """Spearman rho between numeric non-target columns (at most 25); the 10 strongest with |rho| >= 0.3."""
    usable = [c for c in numeric if sample[c].nunique(dropna=True) >= 2][:MAX_CORRELATED]
    if len(usable) < 2:
        return []
    matrix = sample[usable].apply(lambda s: pd.to_numeric(s, errors="coerce").astype("float64"))
    rho = matrix.corr(method="spearman", min_periods=10)
    pairs = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            value = rho.at[a, b]
            if pd.notna(value) and abs(value) >= MIN_RHO:
                pairs.append({"a": a, "b": b, "rho": float(value)})
    pairs.sort(key=lambda p: -abs(p["rho"]))
    return pairs[:TOP_PAIRS]


# ---------------------------------------------------------------------- highlights


def _highlights(view: pd.DataFrame, profile: dict[str, Any], columns: list[dict[str, Any]], target: str | None,
                summary: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rows = summary["rows"]

    def add(severity: str, title: str, text: str, names: list[str]) -> None:
        out.append({"severity": severity, "title": title, "text": text, "columns": names})

    after = [c for c in view.columns if c != target and POST_OUTCOME_NAME.search(c)]
    if after:
        add("warning", "Check when these values are known",
            f"The name{'s' if len(after) > 1 else ''} of {_list(after)} suggest{'' if len(after) > 1 else 's'} a value recorded "
            "after the outcome. Check whether this is known when you predict; if it is not, it must stay out of the model.", after)

    sparse = [c for c in profile["columns"] if c["missing_rate"] > MISSING_WARNING]
    if sparse:
        detail = ", ".join(f"`{c['name']}` ({c['missing_rate']:.0%})" for c in sparse[:6])
        add("warning", "Columns with many missing values",
            f"More than {MISSING_WARNING:.0%} of the values are missing in {detail}{' and more' if len(sparse) > 6 else ''}. "
            "Decide whether to drop them or keep missingness as a signal; any filling happens inside the training folds.",
            [c["name"] for c in sparse])

    duplicates = summary["duplicate_rows"]
    if duplicates:
        share = duplicates / max(summary["rows"], 1)
        add("warning" if share > 0.01 else "info", "Duplicate rows",
            f"{duplicates:,} rows ({share:.1%}) repeat another row exactly. Copies can land on both sides of a split "
            "and make scores look better than they are.", [])

    ids = [c for c in profile["id_candidates"] if c != target]
    if ids:
        add("info", "Likely identifiers",
            f"{_list(ids)} look{'' if len(ids) > 1 else 's'} like {'identifiers' if len(ids) > 1 else 'an identifier'} "
            "(unique per row or named like one). Identifiers rarely carry signal that holds on new rows; "
            "keep them for joins and de-duplication.", ids)

    constant = [c["name"] for c in columns if c.get("top_share") is not None and c["top_share"] >= NEAR_CONSTANT and rows >= 10]
    if constant:
        add("info", "Nearly constant columns",
            f"{_list(constant)} {'have' if len(constant) > 1 else 'has'} the same value in at least {NEAR_CONSTANT:.0%} "
            f"of the rows, so {'they carry' if len(constant) > 1 else 'it carries'} almost no information.", constant)

    codes = [c for c in category_code_columns(view) if c != target]
    if codes:
        add("info", "Numbers that may be category codes",
            f"{_list(codes)} hold{'' if len(codes) > 1 else 's'} whole numbers with few distinct values. If they stand for "
            "categories, their order and size mean nothing, so they should not feed arithmetic features.", codes)

    for column in columns:
        time = column.get("time")
        # An empty period is only surprising when the others are full: at least 5 rows per period on average.
        if not time or len(time["by"]) < 3 or sum(n for _, n in time["by"]) < GAP_MIN_PER_PERIOD * len(time["by"]):
            continue
        inner = time["by"][1:-1]
        empty = sum(1 for _, n in inner if n == 0)
        if empty:
            add("info", "Gaps in time",
                f"`{column['name']}` has no rows in {empty} of {len(time['by'])} {time['period']}s between "
                f"{time['by'][0][0]} and {time['by'][-1][0]}. Check whether data is missing for those periods.", [column["name"]])
    return out


def _list(names: list[str], limit: int = 5) -> str:
    shown = [f"`{n}`" for n in names[:limit]]
    if len(names) > limit:
        return ", ".join(shown) + f" and {len(names) - limit} more"
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


# ---------------------------------------------------------------------- JSON


def _round(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    if value == 0 or abs(value) >= 1:
        return round(value, 4)
    return float(f"{value:.4g}")


def _json_safe(value: Any) -> Any:
    """Plain JSON types only: numpy scalars unwrapped, NaN and NaT to None, floats rounded."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _round(float(value))
    if isinstance(value, str):
        return value
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    return str(value)
