"""Questions answered from a table, never by listing its rows (package H6 of the Home plan).

A person who wants numbers from their data (an analysis, a report, a business question) gets them through a typed
``QueryPlan``: filters, up to three groups, an optional period with its grain, the measures, a sort and a limit. Code
validates the plan against the table's columns before anything runs, executes it with pandas, removes every group
smaller than the row floor (counted, never shown), and records which columns were related (the exposure log), so a later
prediction on the same table knows what the person already looked at.

The rules:
- every plan aggregates: no plan lists rows, ever;
- no unknown column; no free-text column in a group or a filter's value; no grouping by an identifier-like column;
- a sum, mean, median or percentile only on a numeric column; a minimum or maximum on a numeric column or a time (the
  first and last date); any measure may carry its own condition (``where``: the rows it is computed over), so two periods
  or two groups are compared in one plan; a condition never names one identifier or a free text, and a conditioned
  measure over fewer than ``floor`` rows is withheld; at most 3 groups; a limit of at most 500;
- a group of fewer than ``floor`` rows (5 by default) is removed and counted in ``suppressed_groups``.

The contract is ``QueryPlan`` v1 (a host stores it with the answer) and ``QueryResult`` v1. Only numpy, pandas and
pydantic are used.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FilterOp = Literal["eq", "ne", "in", "not_in", "gt", "ge", "lt", "le", "between", "is_null", "not_null"]
MeasureOp = Literal["count", "count_distinct", "sum", "mean", "median", "min", "max", "rate", "percentile"]
Grain = Literal["day", "week", "month", "quarter", "year"]
NUMERIC_OPS = frozenset({"sum", "mean", "median", "min", "max", "percentile"})
MAX_GROUPS, MAX_LIMIT, FLOOR = 3, 500, 5
ID_NAME = re.compile(r"(?:^|_)(?:id|key|code|uuid|number|no|email|phone|name)$|^id_|(?:^|_)id(?:_|$)", re.I)
EXTREME = frozenset({"min", "max"})  # one row's own value: withheld on a group under twice the floor
Scalar = Union[str, int, float, bool, None]


class Filter(Contract):
    column: str = Field(max_length=200)
    op: FilterOp
    value: Union[Scalar, list[Scalar]] = None


class Period(Contract):
    column: str = Field(max_length=200)
    grain: Grain = "month"
    start: Optional[str] = Field(default=None, max_length=40)  # ISO date, inclusive
    end: Optional[str] = Field(default=None, max_length=40)  # ISO date, inclusive


class Measure(Contract):
    name: str = Field(max_length=80)
    op: MeasureOp
    column: Optional[str] = Field(default=None, max_length=200)  # none for a count of rows
    where: Optional[Filter] = None  # rate: the share of rows where this holds; any other measure: the rows it is computed over (0.159.0)
    q: Optional[float] = Field(default=None, ge=0, le=100)  # percentile: which one


class Sort(Contract):
    by: str = Field(max_length=200)
    desc: bool = True


class QueryPlan(Versioned):
    schema_version: Literal["dclab_core.query_plan.v1"] = _version("query_plan")
    filters: list[Filter] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    period: Optional[Period] = None
    measures: list[Measure] = Field(default_factory=list)
    sort: list[Sort] = Field(default_factory=list)
    limit: int = Field(default=50, ge=1)


class QueryResult(Versioned):
    schema_version: Literal["dclab_core.query_result.v1"] = _version("query_result")
    columns: list[str]
    rows: list[list[Any]]  # aggregates only: one row per kept group (or one row for the whole table)
    suppressed_groups: int = 0  # groups under the floor, removed and never shown
    rows_scanned: int
    plan_hash: str


CONTRACTS["query_plan"] = QueryPlan
CONTRACTS["query_result"] = QueryResult


def plan_hash(plan: QueryPlan) -> str:
    return hashlib.sha256(json.dumps(plan.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16]


def _kinds(frame: pd.DataFrame) -> dict[str, str]:
    """numeric, datetime, text (free text), identifier (nearly unique), or categorical, by column."""
    out = {}
    n = max(len(frame), 1)
    for c in frame.columns:
        s = frame[c]
        unique = s.nunique(dropna=True)
        if pd.api.types.is_bool_dtype(s):
            out[str(c)] = "categorical"
        elif pd.api.types.is_numeric_dtype(s):  # a number is an identifier only when its name says so (an amount can be unique)
            out[str(c)] = "identifier" if ID_NAME.search(str(c)) and unique / n > 0.95 else "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            out[str(c)] = "datetime"
        elif is_time(s):  # dates kept as text (2025-11-28): a time, never an identifier however many distinct days it holds (0.159.0)
            out[str(c)] = "datetime"
        else:
            text = s.dropna().astype(str)
            words = float(text.str.split().str.len().median()) if len(text) else 0.0
            # a person's id is an identifier however often it repeats (the review of H6: 1,200 customer ids over 2,020 orders
            # read as categorical, and grouping by them listed named customers' totals)
            out[str(c)] = "text" if words >= 4 else "identifier" if ID_NAME.search(str(c)) or (unique / n > 0.5 and n > 20) else "categorical"
    return out


WHERE_FIX = ('Fix: a share is {"name": "left_share", "op": "rate", "where": {"column": "<column>", "op": "eq", "value": "<value>"}} '
             '(the condition is the where, with no "column" on the measure itself); the number of rows meeting a condition is '
             '{"name": "left", "op": "count", "where": {"column": "<column>", "op": "eq", "value": "<value>"}}')
YES_WORDS, NO_WORDS = frozenset({"yes", "y", "true", "t", "1"}), frozenset({"no", "n", "false", "f", "0"})


def _flag(series: pd.Series) -> dict[str, Any]:
    """The values of a column that holds at most two, as lower-case text to the value the table has (empty when it holds more)."""
    found = series.dropna().unique()
    if not 1 <= len(found) <= 2:
        return {}
    return {str(v).strip().lower(): (v.item() if hasattr(v, "item") else v) for v in found}


def _align(f: Filter, frame: pd.DataFrame, notes: list[str]) -> Filter:
    """A yes/no column asked with true, "yes", 1 and the like: the value the table really holds, said in ``notes``."""
    if f.column not in frame.columns or f.op not in ("eq", "ne") or isinstance(f.value, list) or f.value is None:
        return f
    held = _flag(frame[f.column])
    want = str(f.value).strip().lower()
    if not held or want in held:
        actual = held.get(want, f.value)
    elif want in YES_WORDS | NO_WORDS and set(held) <= YES_WORDS | NO_WORDS:
        actual = next((v for k, v in held.items() if (k in YES_WORDS) == (want in YES_WORDS)), f.value)
    else:
        return f
    if actual == f.value and type(actual) is type(f.value):
        return f
    notes.append(f"{f.column}: {f.value!r} read as {actual!r}, the value the table holds")
    return f.model_copy(update={"value": actual})


def repair(plan: QueryPlan, frame: pd.DataFrame) -> tuple[QueryPlan, list[str]]:
    """What a plain misreading of the plan's shape can be made right, said in the notes and never silent: a yes/no column
    asked with true or "yes" gets the value the table holds, and a share asked of a yes/no column with no condition counts
    that column's yes value. Anything else is left for ``validate`` to refuse."""
    notes: list[str] = []
    filters = [_align(f, frame, notes) for f in plan.filters]
    measures = []
    for m in plan.measures:
        if m.where is not None:
            m = m.model_copy(update={"where": _align(m.where, frame, notes)})
        elif m.op == "rate" and m.column in frame.columns:
            held = _flag(frame[m.column])
            yes = [(k, v) for k, v in held.items() if k in YES_WORDS]
            says_no = bool(re.search(r"(?:^|_)(?:no|not|stay\w*|retain\w*|kept)(?:_|$)", m.name.lower()))  # a share named for the other value is not guessed
            if len(held) == 2 and len(yes) == 1 and set(held) <= YES_WORDS | NO_WORDS and not says_no:  # both values say yes or no: never "x"/"y"
                m = m.model_copy(update={"where": Filter(column=m.column, op="eq", value=yes[0][1]), "column": None})
                notes.append(f"the share {m.name} counts the rows where {m.where.column} = {yes[0][1]!r}")
        measures.append(m)
    return plan.model_copy(update={"filters": filters, "measures": measures}), notes


def validate(plan: QueryPlan, frame: pd.DataFrame) -> list[str]:
    """The reasons the plan is refused on this table (empty: it may run)."""
    kinds = _kinds(frame)
    why: list[str] = []

    def known(column: Optional[str], role: str) -> bool:
        if column is None or column not in kinds:
            why.append(f"{role}: the table has no column {column!r}")
            return False
        return True

    if not plan.measures:
        why.append("a plan must aggregate: no measure means listing rows, which is never done")
    if len(plan.group_by) > MAX_GROUPS:
        why.append(f"at most {MAX_GROUPS} groups")
    if plan.limit > MAX_LIMIT:
        why.append(f"a limit of at most {MAX_LIMIT}")
    for g in plan.group_by:
        if known(g, "group"):
            if kinds[g] in ("text", "identifier"):
                why.append(f"group: {g} is {'free text' if kinds[g] == 'text' else 'identifier-like'}; grouping by it would list rows")
    for f in plan.filters:
        if not known(f.column, "filter"):
            continue
        if kinds[f.column] in ("text", "identifier") and f.op not in ("is_null", "not_null"):
            why.append(f"filter: {f.column} is {'free text' if kinds[f.column] == 'text' else 'identifier-like'}; it is not filtered by "
                       "value (filtering one row out of a total would reveal it)")
        if _names_a_moment(frame, kinds, f):
            why.append(f"filter: {f.column} is a time with nearly one row a value; compare it with a range (between, gt, lt), "
                       "never one value (filtering one row out of a total would reveal it)")
        if f.op == "between" and not (isinstance(f.value, list) and len(f.value) == 2):
            why.append(f"filter: between on {f.column} needs two values")
        values = f.value if isinstance(f.value, list) else [f.value]
        if f.op in ("gt", "ge", "lt", "le", "between") and kinds[f.column] == "numeric" and not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            why.append(f"filter: {f.column} is numeric; compare it with numbers")
    if plan.period is not None and known(plan.period.column, "period"):
        parsed = pd.to_datetime(frame[plan.period.column], errors="coerce", utc=True) if kinds[plan.period.column] != "numeric" else None
        if parsed is None or parsed.notna().mean() < 0.5:
            why.append(f"period: {plan.period.column} is not a time")
        for edge in (plan.period.start, plan.period.end):
            if edge is not None and not DATE.match(str(edge)):
                why.append(f"period: {edge!r} is not a date (YYYY-MM-DD)")
        if plan.period.column in plan.group_by:
            why.append(f"period: {plan.period.column} is also a group")
    for m in plan.measures:
        # a measure over the rows that meet a condition: a count (the owner's Telco session, 2026-11-02) and, since 0.159.0, a
        # sum, a mean or any other measure (HAB-002: "the sum of the amounts in one quarter" was refused eleven times)
        if m.op != "rate" and m.where is not None:
            if known(m.where.column, f"measure {m.name} condition") and kinds[m.where.column] in ("text", "identifier") \
                    and m.where.op not in ("is_null", "not_null"):
                why.append(f"measure {m.name}: a measure's condition never names one identifier or a free text")
            if m.where.column in kinds and _names_a_moment(frame, kinds, m.where):
                why.append(f"measure {m.name}: {m.where.column} is a time with nearly one row a value; its condition is a range, never one value")
            if m.where.op == "between" and not (isinstance(m.where.value, list) and len(m.where.value) == 2):
                why.append(f"measure {m.name}: between needs two values")
        if m.op == "count" and m.column is None:
            continue
        if m.op == "rate":
            if m.where is None or not known(m.where.column, "rate"):
                why.append(f"measure {m.name}: a rate needs the condition it counts (where). {WHERE_FIX}")
            elif kinds[m.where.column] in ("text", "identifier") and m.where.op not in ("is_null", "not_null"):
                why.append(f"measure {m.name}: a rate's condition never names one identifier or a free text")
            elif _names_a_moment(frame, kinds, m.where):
                why.append(f"measure {m.name}: {m.where.column} is a time with nearly one row a value; a rate's condition is a range, never one value")
            continue
        if known(m.column, f"measure {m.name}"):
            if m.op in EXTREME and kinds[m.column] != "numeric" and is_time(frame[m.column]):
                continue  # the first and the last date of a time column
            if m.op in NUMERIC_OPS and kinds[m.column] != "numeric":
                why.append(f"measure {m.name}: {m.op} needs a numeric column; {m.column} is {kinds[m.column]}")
            if m.op not in ("count", "count_distinct") and kinds[m.column] in ("text", "identifier"):
                why.append(f"measure {m.name}: {m.column} is {kinds[m.column]}; only counted")
        if m.op == "percentile" and m.q is None:
            why.append(f"measure {m.name}: a percentile needs q")
    names = [m.name for m in plan.measures]
    if len(set(names)) < len(names):
        why.append("each measure needs its own name")
    if set(names) & set(plan.group_by):
        why.append("a measure's name must differ from every group")
    for s in plan.sort:
        if s.by not in names and s.by not in plan.group_by and not (plan.period and s.by == plan.period.column):
            why.append(f"sort: {s.by} is neither a measure nor a group")
    return why


def _names_a_moment(frame: pd.DataFrame, kinds: dict[str, str], f: Filter) -> bool:
    """A condition that names single values of a time column holding nearly one row a value (a timestamp): the same row
    differencing an identifier's value would allow (the review of 0.159.0: text dates are times, no longer identifiers)."""
    if kinds.get(f.column) != "datetime" or f.op not in ("eq", "ne", "in", "not_in"):
        return False
    s = frame[f.column]
    n = len(s)
    return n > 20 and s.nunique(dropna=True) / n > 0.5


_DAY = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*$")


def _time_mask(s: pd.Series, f: Filter) -> Optional[pd.Series]:
    """A condition on a time column compared as times, never as text (28/11/2025 against 2025-10-01), a bound given as a
    day covering the whole of that day (2025-12-31 14:00 is inside "to 2025-12-31"), as a period's end does. None when a
    bound is no time: the caller then compares the values as they are."""
    import warnings

    def stamp(v: Any) -> tuple[pd.Timestamp, bool]:
        t = pd.Timestamp(v)
        if t is pd.NaT:
            raise ValueError(v)
        return (t.tz_convert(None) if t.tzinfo is not None else t), bool(isinstance(v, str) and _DAY.match(v))

    try:
        values = [stamp(v) for v in (f.value if isinstance(f.value, list) else [f.value])]
    except (ValueError, TypeError):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = pd.to_datetime(s if pd.api.types.is_datetime64_any_dtype(s) else s.astype(str).where(s.notna()), errors="coerce", utc=True).dt.tz_localize(None)
    day = pd.Timedelta(days=1)
    upto = lambda v, whole: (t < v + day) if whole else (t <= v)  # noqa: E731
    if f.op == "between":
        if len(values) != 2:
            return None
        (lo, _), (hi, whole) = values
        return (t >= lo) & upto(hi, whole)
    if f.op in ("in", "not_in"):
        hit = pd.Series(False, index=s.index)
        for v, whole in values:
            hit |= (t.dt.normalize() == v) if whole else (t == v)
        return hit if f.op == "in" else ~hit
    v, whole = values[0]
    if f.op == "eq":
        return (t.dt.normalize() == v) if whole else (t == v)
    if f.op == "ne":
        return ~((t.dt.normalize() == v) if whole else (t == v))
    if f.op == "ge":
        return t >= v
    if f.op == "gt":
        return (t >= v + day) if whole else (t > v)
    if f.op == "le":
        return upto(v, whole)
    if f.op == "lt":
        return t < v
    return None


def _mask(frame: pd.DataFrame, f: Filter) -> pd.Series:
    s, v = frame[f.column], f.value
    if f.op == "is_null":
        return s.isna()
    if f.op == "not_null":
        return s.notna()
    if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and is_time(s):
        timed = _time_mask(s, f)
        if timed is not None:
            return timed
    if f.op == "in":
        return s.isin(v if isinstance(v, list) else [v])
    if f.op == "not_in":
        return ~s.isin(v if isinstance(v, list) else [v])
    if f.op == "between":
        lo, hi = (v if isinstance(v, list) and len(v) == 2 else (None, None))
        return s.between(lo, hi)
    ops = {"eq": "__eq__", "ne": "__ne__", "gt": "__gt__", "ge": "__ge__", "lt": "__lt__", "le": "__le__"}
    return getattr(s, ops[f.op])(v)


_DATE_LIKE = re.compile(r"^\s*(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(?:[ T]\d{1,2}:\d{2}.*)?\s*$")


def is_time(series: pd.Series) -> bool:
    """A column of dates or times: so typed, or text of which at least half reads as a date (never a number)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    text = series.dropna().astype(str)
    if not len(text) or text.str.match(_DATE_LIKE).mean() < 0.5:  # written as a date (a year alone, or a code of digits, is none)
        return False
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(pd.to_datetime(text, errors="coerce", utc=True).notna().mean() >= 0.5)


def _measure(rows: pd.DataFrame, m: Measure, floor: int = FLOOR) -> Any:
    if m.op not in ("count", "rate") and m.where is not None:  # computed over the rows that meet the condition
        rows = rows[_mask(rows, m.where)]
        if 0 < len(rows) < floor:
            return None  # a sum or a mean over one or two rows is those rows' own values: withheld
    if m.op in EXTREME and m.column is not None and not pd.api.types.is_numeric_dtype(rows[m.column]) and is_time(rows[m.column]):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = pd.to_datetime(rows[m.column].astype(str), errors="coerce", utc=True).dropna()
        if len(t) < 2 * floor:
            return None  # one row's own date in a small group: withheld, like any extreme
        found = t.min() if m.op == "min" else t.max()
        return found.strftime("%Y-%m-%d") if found == found.normalize() else found.strftime("%Y-%m-%d %H:%M")
    if m.op == "count":
        if m.where is not None:  # only the rows that meet the condition are counted
            rows = rows[_mask(rows, m.where)]
            n = int(len(rows)) if m.column is None else int(rows[m.column].notna().sum())
            return n if n == 0 or n >= floor else None  # one or two rows meeting a condition are never counted out (the review of 37.11j)
        return int(len(rows)) if m.column is None else int(rows[m.column].notna().sum())
    if m.op == "count_distinct":
        return int(rows[m.column].nunique(dropna=True))
    if m.op == "rate":
        return float(_mask(rows, m.where).mean()) if len(rows) else None
    s = pd.to_numeric(rows[m.column], errors="coerce").dropna()
    if not len(s):
        return None
    extreme = m.op in EXTREME or (m.op == "percentile" and (m.q or 0) <= 5 or (m.op == "percentile" and (m.q or 0) >= 95))
    if extreme and len(s) < 2 * floor:
        return None  # one person's own value in a small group: withheld (the review of H6)
    if m.op == "percentile":
        return float(np.percentile(s, m.q))
    return float(getattr(s, m.op)())


def execute(plan: QueryPlan, frame: pd.DataFrame, floor: int = FLOOR) -> QueryResult:
    """Run a validated plan; every group under ``floor`` rows is removed and counted. Raises ValueError when refused."""
    why = validate(plan, frame)
    if why:
        raise ValueError("; ".join(why))
    try:
        return _execute(plan, frame, floor)
    except (TypeError, ValueError, KeyError) as error:  # a plan the validator let through but the table refuses: said, never a crash
        raise ValueError(f"the plan could not run on this table ({type(error).__name__}: {str(error)[:160]})") from None


def _execute(plan: QueryPlan, frame: pd.DataFrame, floor: int) -> QueryResult:
    rows = frame
    for f in plan.filters:
        rows = rows[_mask(rows, f)]
    keys = list(plan.group_by)
    if plan.period is not None:
        t = pd.to_datetime(rows[plan.period.column], errors="coerce", utc=True).dt.tz_localize(None)
        keep = t.notna()
        if plan.period.start:
            keep &= t >= pd.Timestamp(plan.period.start)
        if plan.period.end:  # inclusive: the whole of the end day (the review of H6)
            keep &= t < pd.Timestamp(plan.period.end) + pd.Timedelta(days=1)
        rows, t = rows[keep], t[keep]
        freq = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}[plan.period.grain]
        rows = rows.assign(**{plan.period.column: t.dt.to_period(freq).astype(str)})
        keys.append(plan.period.column)
    columns = [*keys, *(m.name for m in plan.measures)]
    out: list[list[Any]] = []
    suppressed = 0
    groups = rows.groupby(keys, dropna=False, sort=True) if keys else [((), rows)]
    for key, part in groups:
        if len(part) < floor:
            suppressed += 1
            continue
        key = key if isinstance(key, tuple) else (key,)
        out.append([*(None if pd.isna(k) else (k.item() if hasattr(k, "item") else k) for k in key),
                    *(_measure(part, m, floor) for m in plan.measures)])
    for s in reversed(plan.sort):  # a group with no value comes last either way
        i = columns.index(s.by)
        valued = sorted((r for r in out if r[i] is not None), key=lambda r: r[i], reverse=s.desc)
        out = valued + [r for r in out if r[i] is None]
    return QueryResult(columns=columns, rows=out[:plan.limit], suppressed_groups=suppressed, rows_scanned=int(len(frame)),
                       plan_hash=plan_hash(plan))


def exposure(plan: QueryPlan) -> dict[str, Any]:
    """What one executed plan related: every grouped or filtered column with every measured one (H8 reads it when an
    analysis becomes a prediction on the same table)."""
    sides = sorted({*plan.group_by, *(f.column for f in plan.filters), *([plan.period.column] if plan.period else [])})
    measured = sorted({c for m in plan.measures for c in ([m.column] if m.column else []) + ([m.where.column] if m.where else [])})
    related = [[a, b] for a in sides for b in measured if a != b]
    for m in plan.measures:  # mean(tenure where churned = Yes) compares tenure with churned as a group by churned does
        if m.where is not None and m.column and m.where.column != m.column and [m.where.column, m.column] not in related:
            related.append([m.where.column, m.column])
    return {"plan_hash": plan_hash(plan), "columns_related": related,
            "rows": "filter" if plan.filters else "all"}


__all__ = ["QueryPlan", "QueryResult", "Filter", "Period", "Measure", "Sort", "validate", "execute", "exposure", "plan_hash", "is_time", "FLOOR"]

# The contracts that read this module's names, registered once it is complete: imported here rather than by
# contracts.py, so `from dclab_core import query` as a program's first import works (contracts.py imports this module
# while it is still loading, and recipe's `from dclab_core.query import Filter` would find no Filter yet).
import dclab_core.recipe  # noqa: E402,F401  (a table prepared step by step, the original never changed, H9)
import dclab_core.change  # noqa: E402,F401  (why a number changed between two periods, H11)
import dclab_core.dashboard  # noqa: E402,F401  (KPIs and dashboards on a person's table, H10)
