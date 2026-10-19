"""Ask the workspace (package 30.21, the owner's pattern 7): a chat over a workspace's own tables where every number comes from code run on
them, never from the model's head.

* **Typed computations, not free code** (:class:`Computation`, :func:`compute`): a count, a sum, a mean, a median, a minimum, a maximum, a
  share or a count of distinct values of one column, grouped by up to two columns and filtered by equality, run by pandas on the table.
  A model only chooses one and fills it (the same rule as ``metric_change``'s templates); nothing it writes runs.
* **Personal data masked before any model** (:func:`masked`): a value that reads as personal (an e-mail, a phone, a card, an identifier
  of a person) is replaced in a tool's result before the model sees it (``dclab_core.redact``); a column a table names as personal is
  never computed on, filtered on or grouped by; and no value is shown over fewer than ``MIN_ROWS`` rows (a smaller group is left out, a
  smaller filtered set refused), so no answer is one person's.
* **Every number grounded** (:func:`ungrounded`): the numbers of an answer must be numbers a computation returned (or counts it gave);
  an answer with any other is refused and the person told so.

Pure numpy and pandas; the host owns the tables, the loop, the model and the audit.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal, Mapping, Optional, Sequence

import pandas as pd
from pydantic import Field

from dclab_core.contracts import Contract

Op = Literal["count", "sum", "mean", "median", "min", "max", "share", "distinct"]
MAX_GROUPS = 50
MIN_ROWS = 5  # a value over fewer rows is close to one person's: never shown (a group under it is left out, a filter under it refused)
NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")


class Computation(Contract):
    table: str = Field(max_length=120)
    op: Op
    column: Optional[str] = Field(default=None, max_length=120)  # every op but count reads a column
    group_by: list[str] = Field(default_factory=list, max_length=2)
    where: dict[str, Any] = Field(default_factory=dict)  # column -> value, equality only


def masked(value: Any) -> Any:
    """A value as a model may see it: personal-looking text replaced (``dclab_core.redact``), numbers and plain words kept."""
    from dclab_core import redact

    if isinstance(value, str):
        return redact.redact_text(value)[0]
    if isinstance(value, Mapping):
        return {masked(k) if isinstance(k, str) else k: masked(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [masked(v) for v in value]
    return value


def _clean(x: Any) -> Any:
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else round(x, 6)
    if hasattr(x, "item"):
        return _clean(x.item())
    return x


def compute(frame: pd.DataFrame, c: Computation, personal: Sequence[str] = ()) -> dict[str, Any]:
    """The computation on the table: its result (a number, or one per group, at most ``MAX_GROUPS`` largest), the rows it read, and what
    it was, so the answer can say it. Raises ValueError for a column the table does not hold or a personal column as a group."""
    for col in [*([c.column] if c.column else []), *c.group_by, *c.where]:
        if col not in frame.columns:
            raise ValueError(f"the table {c.table} has no column {col!r}")
    touched = set(c.group_by) | set(c.where) | ({c.column} if c.column else set())
    if touched & set(personal):
        raise ValueError(f"{', '.join(sorted(touched & set(personal)))} holds a person's data: it is never computed on, filtered on or a group")
    if c.op != "count" and not c.column:
        raise ValueError(f"{c.op} reads a column")
    rows = frame
    for col, value in c.where.items():
        rows = rows[_equal(rows[col], value)]
    if len(rows) < MIN_ROWS:
        raise ValueError(f"the filter leaves {len(rows)} rows, fewer than {MIN_ROWS}: a value over so few rows is too close to one person's")

    def one(part: pd.DataFrame) -> Any:
        if c.op == "count":
            return int(len(part))
        series = part[c.column]
        if c.op == "distinct":
            return int(series.nunique(dropna=True))
        if c.op == "share":  # the share of rows where the column is true (or equals 1, "yes")
            truthy = series.astype(str).str.lower().isin(("1", "true", "yes", "y"))
            return _clean(float(truthy.mean())) if len(part) else None
        numbers = pd.to_numeric(series, errors="coerce")
        return _clean(float(getattr(numbers, c.op)())) if numbers.notna().any() else None

    if c.group_by:
        groups = [(k if isinstance(k, tuple) else (k,), g) for k, g in rows.groupby(c.group_by, dropna=True)]
        small = sum(len(g) < MIN_ROWS for _, g in groups)
        groups = sorted((kg for kg in groups if len(kg[1]) >= MIN_ROWS), key=lambda kg: -len(kg[1]))
        result = [{**{col: _clean(v) for col, v in zip(c.group_by, key)}, "value": one(g), "rows": int(len(g))} for key, g in groups[:MAX_GROUPS]]
        more = max(0, len(groups) - MAX_GROUPS)
    else:
        result, more, small = one(rows), 0, 0
    return {"computation": c.model_dump(mode="json"), "result": result, "rows_read": int(len(rows)), "groups_left_out": more,
            "groups_too_small": small}


def _equal(series: pd.Series, value: Any) -> pd.Series:
    """Rows where the column equals the value: as numbers when the column is numeric (1 equals 1.0), else as text."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return series == number if pd.notna(number) else pd.Series(False, index=series.index)
    return series.astype(str) == str(value)


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUMBER.findall(text or ""):
        raw = m.rstrip("%").replace(",", "")
        try:
            out.append(float(raw))  # a share said as 42.1% is read as 42.1; the grounding reads a result's share times 100 too
        except ValueError:
            continue
    return out


def _results_numbers(results: Sequence[Mapping[str, Any]]) -> list[float]:
    found: list[float] = []

    def walk(x: Any) -> None:
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            found.append(float(x))
        elif isinstance(x, Mapping):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    for r in results:
        walk(r.get("result"))
        walk(r.get("rows_read"))
        for v in ((r.get("computation") or {}).get("where") or {}).values():  # what the question filtered on is the person's own number
            try:
                found.append(float(v))
            except (TypeError, ValueError):
                continue
    return found


def ungrounded(answer: str, results: Sequence[Mapping[str, Any]], tolerance: float = 0.005) -> list[float]:
    """The numbers of an answer that no computation returned (a rounding of one counts: 0.4213 said as 42.1% or 0.42; a year or a
    count of the groups is a number it may say only if a result holds it too)."""
    known = _results_numbers(results)
    bad = []
    for n in numbers_in(answer):
        if any(abs(n - k) <= tolerance * max(1.0, abs(k)) or (0 <= k <= 1 and abs(n - k * 100) <= tolerance * 100) or _rounds_to(k, n) for k in known):
            continue
        bad.append(n)
    return bad


def _rounds_to(known: float, said: float) -> bool:
    for digits in (0, 1, 2, 3):
        if round(known, digits) == said or (0 <= known <= 1 and round(known * 100, digits) == said):  # a share said as a percent
            return True
    return False


__all__ = ["Computation", "Op", "compute", "masked", "numbers_in", "ungrounded", "MAX_GROUPS", "MIN_ROWS"]
