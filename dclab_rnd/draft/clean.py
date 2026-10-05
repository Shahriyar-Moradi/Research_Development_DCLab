"""Structural, lossless cleaning of a fresh table, with a step-by-step log for the user.

Each step fixes how values are *written*, never what they say: column names, missing-value
markers, stray spaces, numbers and dates stored as text, yes/no columns, empty rows and
columns, exact duplicate rows. No step learns a statistic from the data (no imputation,
scaling, encoding or outlier rule) and no step changes a column because of another column, so
cleaning the whole table before the train/test split leaks nothing. Everything that learns
from the data waits for the training folds (CLAUDE.md rules 1-2).

Two conversions can turn a few values into missing ones, and both say so in the log: a column
becomes numeric when at least 95% of its values read as numbers, and a date column when at
least 90% read as dates. Text that looks numeric but has leading zeros ("007", "02134") stays
text: those are codes, and a number would lose the zeros.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from dclab_rnd.categoricals import looks_like_codes
from dclab_rnd.studio.data import _datetime_like

from .structure import parse_times, unique_names

MISSING_MARKERS = frozenset({"", "NA", "N/A", "n/a", "null", "NULL", "None", "none", "-", "?", "nan", "NaN"})
BOOLEAN_WORDS = {"yes": True, "no": False, "true": True, "false": False, "y": True, "n": False,
                 "t": True, "f": False, "1": True, "0": False}
NUMBER_SHARE = 0.95
DATE_SHARE = 0.90
NUMBER = re.compile(
    r"^(?P<sign>[+-])?(?P<currency>[$€£])?\s*(?P<sign2>[+-])?"
    r"(?P<digits>[1-9]\d{0,2}(?P<sep>[,_   ])\d{3}(?:(?P=sep)\d{3})*(?:\.\d+)?"
    r"|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"
    r"\s*(?P<percent>%)?$"
)
LEADING_ZERO = re.compile(r"^0\d")

Entry = dict[str, Any]


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[Entry]]:
    """A cleaned copy of ``frame`` and the log of the steps that changed something.

    Each log entry: ``step`` (machine key), ``title``, ``detail``, ``columns``,
    ``rows_before``, ``rows_after``, ``changed`` (names, cells, or dropped rows and columns).
    """
    out = frame.copy(deep=True)
    log: list[Entry] = []
    for step in STEPS:
        rows_before = int(len(out))
        out, entry = step(out)
        if entry and entry["changed"]:
            log.append({"step": entry["step"], "title": entry["title"], "detail": entry["detail"],
                        "columns": [str(c) for c in entry["columns"]], "rows_before": rows_before,
                        "rows_after": int(len(out)), "changed": int(entry["changed"])})
    return out, log


def category_code_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric columns that look like category codes (DCLAB-R11 heuristic); for a note, never a change."""
    return [str(c) for c in frame.columns
            if not pd.api.types.is_bool_dtype(frame[c]) and looks_like_codes(frame[c])]


# ---------------------------------------------------------------------- steps


def _texty(series: pd.Series) -> bool:
    return series.dtype == object or isinstance(series.dtype, pd.StringDtype)


def _text_columns(frame: pd.DataFrame) -> list[Any]:
    return [c for c in frame.columns if _texty(frame[c])]


def _quote(values: list[Any], limit: int = 3) -> str:
    return ", ".join(repr(str(v)[:30]) for v in values[:limit])


def _names(columns: list[Any], limit: int = 5) -> str:
    shown = ", ".join(f"`{c}`" for c in columns[:limit])
    return shown + (f" and {len(columns) - limit} more" if len(columns) > limit else "")


def _step_names(frame: pd.DataFrame) -> tuple[pd.DataFrame, Entry | None]:
    old = list(frame.columns)
    new = unique_names(" ".join(str(c).split()) for c in old)
    changed = [(o, n) for o, n in zip(old, new) if not isinstance(o, str) or o != n]
    frame.columns = new
    detail = "; ".join(f"{str(o)!r} → {n!r}" for o, n in changed[:5]) + (f"; and {len(changed) - 5} more" if len(changed) > 5 else "")
    return frame, {"step": "names", "title": "Tidied column names", "columns": [n for _, n in changed], "changed": len(changed),
                   "detail": f"Removed extra spaces and made names unique: {detail}."}


def _step_missing_markers(frame: pd.DataFrame) -> tuple[pd.DataFrame, Entry | None]:
    total, touched, seen = 0, [], set()
    for column in _text_columns(frame):
        series = frame[column]
        hits = series.map(lambda v: isinstance(v, str) and v.strip() in MISSING_MARKERS).astype(bool)
        count = int(hits.sum())
        if count:
            seen.update(str(v).strip() for v in series[hits].unique())
            frame[column] = series.mask(hits, np.nan)
            total += count
            touched.append(column)
    return frame, {"step": "missing_markers", "title": "Marked missing values", "columns": touched, "changed": total,
                   "detail": f"{total:,} cells written as {', '.join(repr(m) for m in sorted(seen))} are now missing, "
                             f"in {_names(touched)}."}


def _step_trim(frame: pd.DataFrame) -> tuple[pd.DataFrame, Entry | None]:
    total, touched = 0, []
    for column in _text_columns(frame):
        series = frame[column]
        tidy = series.map(lambda v: re.sub(r" {2,}", " ", v.strip()) if isinstance(v, str) else v)
        diff = series.map(lambda v: isinstance(v, str)).astype(bool) & (tidy != series)
        count = int(diff.sum())
        if count:
            frame[column] = tidy
            total += count
            touched.append(column)
    return frame, {"step": "trim", "title": "Trimmed spaces", "columns": touched, "changed": total,
                   "detail": f"Removed leading, trailing and repeated spaces in {total:,} cells of {_names(touched)}."}


def _read_number(value: Any) -> tuple[float | None, bool, bool]:
    """(number or None, was a percentage, has a leading zero) for one cell."""
    match = NUMBER.match(str(value).strip())
    if not match or (match["sign"] and match["sign2"]):
        return None, False, False
    digits = match["digits"]
    if match["sep"]:
        digits = digits.replace(match["sep"], "")
    number = float(digits)
    if (match["sign"] or match["sign2"]) == "-":
        number = -number
    return number, bool(match["percent"]), bool(LEADING_ZERO.match(match["digits"]))


def _step_numbers(frame: pd.DataFrame) -> tuple[pd.DataFrame, Entry | None]:
    total, touched, parts = 0, [], []
    for column in _text_columns(frame):
        series = frame[column]
        present = series.notna()
        if not present.any():
            continue
        reads = series[present].map(_read_number)
        numbers = reads.map(lambda r: r[0])
        ok = numbers.notna()
        if ok.mean() < NUMBER_SHARE or reads.map(lambda r: r[2]).any():
            continue
        converted = pd.Series(np.nan, index=series.index, dtype="float64")
        converted[present] = numbers.astype("float64")
        if converted.notna().all() and (converted % 1 == 0).all():
            converted = converted.astype("int64")
        failed = series[present][~ok]
        note = f"`{column}`"
        if reads.map(lambda r: r[1]).any():
            note += " (values were percentages, kept as written: 45% → 45)"
        if len(failed):
            note += f" ({len(failed)} value{'s' if len(failed) > 1 else ''} that did not read as numbers became missing: {_quote(failed.tolist())})"
        frame[column] = converted
        total += int(present.sum())
        touched.append(column)
        parts.append(note)
    return frame, {"step": "numbers", "title": "Read numbers written as text", "columns": touched, "changed": total,
                   "detail": "Removed currency signs and thousands separators and stored numbers: " + "; ".join(parts) + "."}


def _has_date_shape(value: str) -> bool:
    return any(ch.isdigit() for ch in value) and not value.isdigit()


def _step_dates(frame: pd.DataFrame) -> tuple[pd.DataFrame, Entry | None]:
    total, touched, parts = 0, [], []
    for column in _text_columns(frame):
        series = frame[column]
        if not _datetime_like(series):
            continue
        present = series.notna()
        text = series[present].astype(str).str.strip()
        shaped = text.map(_has_date_shape).astype(bool)
        parsed = parse_times(text.where(shaped))
        if parsed.notna().mean() < DATE_SHARE:
            continue
        converted = pd.Series(pd.NaT, index=series.index, dtype=parsed.dtype)
        converted[present] = parsed
        failed = text[parsed.isna()]
        note = f"`{column}`"
        if str(parsed.dtype).endswith(", UTC]"):
            note += " (stored in UTC)"
        if len(failed):
            note += f" ({len(failed)} value{'s' if len(failed) > 1 else ''} that did not read as dates became missing: {_quote(failed.tolist())})"
        frame[column] = converted
        total += int(present.sum())
        touched.append(column)
        parts.append(note)
    return frame, {"step": "dates", "title": "Read dates", "columns": touched, "changed": total,
                   "detail": "Stored dates and times as dates: " + "; ".join(parts) + "."}


def _step_booleans(frame: pd.DataFrame) -> tuple[pd.DataFrame, Entry | None]:
    total, touched = 0, []
    for column in _text_columns(frame):
        series = frame[column]
        present = series.notna()
        if not present.any():
            continue
        words = series[present].map(lambda v: str(v).strip().lower())
        if not words.isin(list(BOOLEAN_WORDS)).all():
            continue
        converted = pd.Series(pd.NA, index=series.index, dtype="boolean")
        converted[present] = words.map(BOOLEAN_WORDS).astype(bool)
        frame[column] = converted
        total += int(present.sum())
        touched.append(column)
    return frame, {"step": "booleans", "title": "Read yes/no columns", "columns": touched, "changed": total,
                   "detail": f"Stored yes/no, true/false, y/n and 1/0 answers as true/false in {_names(touched)}."}


def _step_empty(frame: pd.DataFrame) -> tuple[pd.DataFrame, Entry | None]:
    empty_columns = [c for c in frame.columns if frame[c].isna().all()]
    frame = frame.drop(columns=empty_columns)
    empty_rows = frame.isna().all(axis=1) if frame.shape[1] else pd.Series(False, index=frame.index)
    rows = int(empty_rows.sum())
    frame = frame[~empty_rows]
    parts = []
    if empty_columns:
        parts.append(f"{len(empty_columns)} column{'s' if len(empty_columns) > 1 else ''} with no values ({_names(empty_columns)})")
    if rows:
        parts.append(f"{rows:,} row{'s' if rows > 1 else ''} with no values")
    return frame, {"step": "empty", "title": "Dropped empty rows and columns", "columns": empty_columns,
                   "changed": len(empty_columns) + rows, "detail": "Dropped " + " and ".join(parts) + "."}


def _step_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, Entry | None]:
    try:
        repeated = frame.duplicated(keep="first")
    except TypeError:  # unhashable cells (lists, dicts): compare their text
        repeated = frame.astype(str).duplicated(keep="first")
    count = int(repeated.sum())
    frame = frame[~repeated].reset_index(drop=True)
    return frame, {"step": "duplicates", "title": "Dropped duplicate rows", "columns": [], "changed": count,
                   "detail": f"Dropped {count:,} row{'s' if count != 1 else ''} that repeated an earlier row exactly; the first copy is kept."}


STEPS: list[Callable[[pd.DataFrame], tuple[pd.DataFrame, Entry | None]]] = [
    _step_names, _step_missing_markers, _step_trim, _step_numbers, _step_dates, _step_booleans, _step_empty, _step_duplicates,
]
