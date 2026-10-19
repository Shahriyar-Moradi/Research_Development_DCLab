"""Why did this number change (package 30.20, the owner's pattern 6): a question about a business number answered from approved
definitions and tested queries, with the reason and not only the number.

A brand head asks "why are our sales down in this mall against last year?". The answer must not be a confident wrong number: free text-to-SQL
joins the wrong tables or reads the wrong dates. So:

1. **The semantic layer** (:class:`SemanticLayer`) holds the approved definitions: base metrics (a sum of one table's column), ratios of
   metrics, the dimensions and the column each table names them by (the approved joins: tables meet on their date and the dimensions both
   hold), the factorizations (sales = footfall x conversion x basket, checked to telescope), and the concepts the data does not hold with
   what they would need.
2. **Tested query templates** (``TEMPLATES``): a question becomes one template and its parameters, chosen by rules or by a model (an
   :class:`~dclab_core.advisor.Advisor`) that only *chooses and fills*: it never writes SQL, and its choice is validated (the metric
   approved, every filter's value present in the data, the period inside it) or the rules' choice stands. A question that names its
   template (``why_changed: metric=sales; brand=Arlo; period=2026-09; compare=last_year``) is parsed exactly and no model is asked.
3. **The answer** runs each template's query, one plan rendered both as SQL (parameters bound, never written in) and as its pandas
   computation: a host that runs SQL (SQLite in the studio, a warehouse in the main product) passes its ``runner`` and the SQL shown
   under the answer is the statement that made its numbers; without one the same plan is computed on the tables given, and a test runs
   both and compares them. The core itself never reaches a database. Before a why is answered, coverage is checked:
   a day where one table has rows and a factor's table has none, or a denominator of zero, is a gap, and the why is refused with what is
   missing (the metric's own change is still given when its table is whole).
4. **Why** decomposes the change in logs: ``d ln(sales) = d ln(footfall) + d ln(conversion) + d ln(basket)``, exactly; each factor's share
   with a 90% interval from a bootstrap over the days of each period; a factor *moved* when its interval excludes zero. The first factor is
   then drilled down by every dimension its tables hold (a sum by segment; a ratio by segment into mix, the segments' weights moving, against
   rate, each segment's own ratio moving), and the dimension where one segment carries most of the change names *where*.
5. **An action** follows from what moved (advisory: words, never a number), with the follow-up to measure whether it helped.

Pure numpy and pandas; deterministic per seed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from dclab_core.advisor import Advisor, consult
from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

TemplateName = Literal["metric_value", "metric_change", "why_changed", "top_segments"]
Compare = Literal["last_year", "previous_period"]
TEMPLATES: dict[str, dict[str, Any]] = {
    "metric_value": {"needs": ("metric", "period"), "says": "the metric's value over a period, optionally filtered"},
    "metric_change": {"needs": ("metric", "period", "compare"), "says": "the metric in a period against a comparison period, with the change's interval"},
    "why_changed": {"needs": ("metric", "period", "compare"), "says": "why the metric changed: its factors, which moved, where, and an action"},
    "top_segments": {"needs": ("metric", "dimension", "period"), "says": "the segments of one dimension ranked by the metric over a period"},
}
PURPOSE = "metric_question"   # the advisor's purpose: choose a template and fill its parameters
BOOTSTRAP, LEVEL, SEED = 1000, 0.90, 0
MONTHS = {m.lower(): i for i, m in enumerate(("January", "February", "March", "April", "May", "June", "July", "August", "September",
                                              "October", "November", "December"), start=1)}


# ---------------------------------------------------------------------------------------------------- the semantic layer


class Table(Contract):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,40}$")
    date: str = Field(min_length=1, max_length=80)  # the column that holds each row's day


class Metric(Contract):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,40}$")
    label: str = Field(max_length=120)
    kind: Literal["sum", "ratio"]
    table: Optional[str] = None       # a sum: the table and column it adds up
    column: Optional[str] = None
    numerator: Optional[str] = None   # a ratio: two metrics' names
    denominator: Optional[str] = None
    unit: str = Field(default="", max_length=40)
    synonyms: list[str] = Field(default_factory=list, max_length=12)
    approved_by: str = Field(default="", max_length=120)

    @model_validator(mode="after")
    def _shape(self) -> "Metric":
        if self.kind == "sum" and not (self.table and self.column):
            raise ValueError(f"the sum {self.name} names its table and column")
        if self.kind == "ratio" and not (self.numerator and self.denominator):
            raise ValueError(f"the ratio {self.name} names its numerator and denominator")
        return self


class Dimension(Contract):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,40}$")
    label: str = Field(max_length=120)
    columns: dict[str, str] = Field(min_length=1)  # table -> the column naming this dimension there
    synonyms: list[str] = Field(default_factory=list, max_length=12)


class Factorization(Contract):
    metric: str
    factors: list[str] = Field(min_length=2, max_length=6)


class NotHeld(Contract):
    name: str = Field(max_length=60)
    synonyms: list[str] = Field(default_factory=list, max_length=12)
    needs: str = Field(max_length=300)  # what the data would need to answer it


class SemanticLayer(Versioned):
    """The approved definitions a question is answered from (v1)."""

    schema_version: Literal["dclab_core.semantic_layer.v1"] = _version("semantic_layer")
    tables: list[Table] = Field(min_length=1, max_length=20)
    metrics: list[Metric] = Field(min_length=1, max_length=40)
    dimensions: list[Dimension] = Field(default_factory=list, max_length=20)
    factorizations: list[Factorization] = Field(default_factory=list, max_length=20)
    not_held: list[NotHeld] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def _consistent(self) -> "SemanticLayer":
        tables, metrics = {t.name for t in self.tables}, {m.name: m for m in self.metrics}
        if len(metrics) != len(self.metrics) or len(tables) != len(self.tables) or len({d.name for d in self.dimensions}) != len(self.dimensions):
            raise ValueError("every table, metric and dimension has its own name")
        for m in self.metrics:
            if m.kind == "sum" and m.table not in tables:
                raise ValueError(f"{m.name} reads the table {m.table}, which the layer does not hold")
            if m.kind == "ratio" and not {m.numerator, m.denominator} <= set(metrics):
                raise ValueError(f"{m.name} divides metrics the layer does not define")
            if m.kind == "ratio" and not all(metrics[x].kind == "sum" for x in (m.numerator, m.denominator) if x in metrics):
                raise ValueError(f"{m.name} divides two sums: a ratio of ratios is written as its own factorization")
        for d in self.dimensions:
            if not set(d.columns) <= tables:
                raise ValueError(f"the dimension {d.name} names a table the layer does not hold")
        for f in self.factorizations:
            if f.metric not in metrics or not set(f.factors) <= set(metrics):
                raise ValueError(f"the factorization of {f.metric} names a metric the layer does not define")
            if _fraction(f.metric, metrics) != _product(f.factors, metrics):
                raise ValueError(f"{' x '.join(f.factors)} is not {f.metric}: a factorization must telescope")
        return self

    def metric(self, name: str) -> Metric:
        return {m.name: m for m in self.metrics}[name]

    def factors_of(self, name: str) -> list[str]:
        return next((list(f.factors) for f in self.factorizations if f.metric == name), [name])

    def base(self, name: str) -> list[str]:
        """The sums a metric is made of."""
        m = self.metric(name)
        return [name] if m.kind == "sum" else list(dict.fromkeys(self.base(m.numerator) + self.base(m.denominator)))  # type: ignore[arg-type]

    def column(self, dimension: str, table: str) -> Optional[str]:
        d = next((d for d in self.dimensions if d.name == dimension), None)
        return None if d is None else d.columns.get(table)


def _fraction(name: str, metrics: dict[str, Metric]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """A metric as a fraction of sums, cancelled: (numerator sums, denominator sums), each sorted."""
    up, down = _parts(name, metrics)
    for s in list(up):
        if s in down:
            up.remove(s)
            down.remove(s)
    return tuple(sorted(up)), tuple(sorted(down))


def _parts(name: str, metrics: dict[str, Metric]) -> tuple[list[str], list[str]]:
    m = metrics[name]
    if m.kind == "sum":
        return [name], []
    nu, nd = _parts(m.numerator, metrics)  # type: ignore[arg-type]
    du, dd = _parts(m.denominator, metrics)  # type: ignore[arg-type]
    return nu + dd, nd + du


def _product(names: list[str], metrics: dict[str, Metric]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    up, down = [], []
    for n in names:
        u, d = _parts(n, metrics)
        up += u
        down += d
    for s in list(up):
        if s in down:
            up.remove(s)
            down.remove(s)
    return tuple(sorted(up)), tuple(sorted(down))


CONTRACTS["semantic_layer"] = SemanticLayer


# ---------------------------------------------------------------------------------------------------- the choice


class Params(Contract):
    metric: Optional[str] = None
    filters: dict[str, str] = Field(default_factory=dict)
    period: Optional[tuple[str, str]] = None
    compare: Optional[Compare] = None
    dimension: Optional[str] = None
    direction: Optional[Literal["down", "up"]] = None  # the way the person says it moved: where to look when it moved within its variation


class Choice(Contract):
    """A question as one template and its parameters, or a refusal with what is missing."""

    template: Optional[TemplateName] = None
    params: Params = Field(default_factory=Params)
    refused: bool = False
    missing: list[str] = Field(default_factory=list, max_length=10)
    reason: str = Field(default="", max_length=4000)
    source: Literal["named", "rules", "model"] = "rules"
    assumed: list[str] = Field(default_factory=list, max_length=10)  # what the question did not say and the rules chose
    notes: list[str] = Field(default_factory=list, max_length=10)    # what the question mentions that the data does not hold, said beside the answer


class Coverage(Contract):
    """One ratio's two tables over one period, checked at the grain they share (the day and every dimension both hold): a key where the
    numerator is positive and the denominator has no row or reads zero, or where the denominator is positive and the numerator's table has
    no row, is a gap. A key with no row in either table is a closed day, not a gap."""

    numerator: str
    denominator: str
    period: tuple[str, str]
    days: int                                               # calendar days in the period
    grain: list[str] = Field(default_factory=list)          # the dimensions checked beside the day
    missing: list[str] = Field(default_factory=list)        # keys the denominator's table has no row for while the numerator is positive
    zero: list[str] = Field(default_factory=list)           # keys the denominator reads zero while the numerator is positive
    numerator_missing: list[str] = Field(default_factory=list)  # keys the numerator's table has no row for while the denominator is positive

    @property
    def gaps(self) -> int:
        return len(self.missing) + len(self.zero) + len(self.numerator_missing)


class Factor(Contract):
    name: str
    label: str
    log_change: float
    change_pct: float
    share: Optional[float] = None  # of the metric's log change (they add up to one)
    ci: Optional[tuple[float, float]] = None  # of the log change, at ``level``: one interval for every factor searched (Bonferroni)
    level: float = LEVEL
    moved: bool


class Segment(Contract):
    value: str
    contribution_pct: float        # of the factor's previous value (per day): the segments add up to its change
    mix_pct: float = 0.0
    rate_pct: float = 0.0
    ci: Optional[tuple[float, float]] = None  # of the contribution, at the drill's level: every segment of every dimension searched (Bonferroni)
    change_pct: Optional[float] = None  # the segment's own change


class Drill(Contract):
    factor: str
    dimension: str
    lead: str                      # the segment that moved the factor most the way it moved (or either way, when neither moved nor was said)
    concentration: float           # the lead's share of the absolute contributions
    effect: Literal["level", "rate", "mix"]
    level: float = LEVEL
    segments: list[Segment] = Field(default_factory=list)


class Query(Contract):
    table: str
    sql: str
    params: list[Any] = Field(default_factory=list)
    purpose: Literal["numbers", "drill", "coverage"] = "numbers"


class MetricAnswer(Versioned):
    """The answer to one question about a number (v1): the choice, the query under it, and the reason."""

    schema_version: Literal["dclab_core.metric_answer.v1"] = _version("metric_answer")
    question: str = Field(max_length=100_000)  # kept whole
    choice: Choice
    status: Literal["answered", "refused"]
    metric: Optional[str] = None
    period: Optional[tuple[str, str]] = None
    compare: Optional[tuple[str, str]] = None
    days: Optional[tuple[int, int]] = None      # calendar days of the period and of the comparison: a sum is compared per day
    current: Optional[float] = None             # the metric over the period (a sum: its total)
    previous: Optional[float] = None
    change_pct: Optional[float] = None          # per day for a sum, so periods of unequal length compare
    ci90: Optional[tuple[float, float]] = None  # of the change, in per cent
    moved: Optional[bool] = None
    factors: list[Factor] = Field(default_factory=list)
    first_factor: Optional[str] = None
    drills: list[Drill] = Field(default_factory=list)
    first_segment: Optional[dict[str, str]] = None  # dimension, value, effect
    segments: list[dict[str, Any]] = Field(default_factory=list)  # top_segments: value and metric
    queries: list[Query] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    coverage: list[Coverage] = Field(default_factory=list)
    refused_because: str = Field(default="", max_length=20_000)
    missing: list[str] = Field(default_factory=list)
    action: str = Field(default="", max_length=600)
    follow_up: str = Field(default="", max_length=300)
    notes: list[str] = Field(default_factory=list)
    words: str = Field(default="", max_length=20_000)  # the answer in a sentence or three, from the numbers above


CONTRACTS["metric_answer"] = MetricAnswer


class Unanswerable(ValueError):
    """A question the data cannot answer as asked: what is missing, and why."""

    def __init__(self, reason: str, missing: list[str]):
        super().__init__(reason)
        self.reason, self.missing = reason, missing


# ---------------------------------------------------------------------------------------------------- the data


Runner = Callable[[str, list[Any]], pd.DataFrame]  # the host's: runs one statement with its parameters bound, returns its rows


class Plan:
    """One tested query: a table's sums by day (and by key columns), filtered and dated. The same plan is the SQL shown under the answer
    (``sql``, ``params``: bound, never written in) and, where no host runs SQL, its pandas computation (``on``); a test runs both and
    compares them. A key column's empty value is never a segment."""

    def __init__(self, table: str, date: str, sums: list[tuple[str, str]], filters: list[tuple[str, str]], period: tuple[str, str],
                 keys: Sequence[tuple[str, str]] = ()):
        self.table, self.date, self.sums, self.filters, self.period, self.keys = table, date, sums, filters, period, list(keys)

    @property
    def sql(self) -> str:
        cols = [f"{_ident(self.date)} AS day"] + [f"{_ident(c)} AS {_ident(a)}" for c, a in self.keys]
        sel = ", ".join(cols + [f"SUM({_ident(col)}) AS {_ident(alias)}" for alias, col in self.sums])
        where = ([f"{_ident(self.date)} BETWEEN ? AND ?"] + [f"{_ident(col)} = ?" for col, _ in self.filters]
                 + [f"{_ident(c)} IS NOT NULL" for c, _ in self.keys])
        group = ", ".join(["day", *(_ident(a) for _, a in self.keys)])
        return f"SELECT {sel} FROM {_ident(self.table)} WHERE {' AND '.join(where)} GROUP BY {group} ORDER BY {group}"

    @property
    def params(self) -> list[Any]:
        return [self.period[0], self.period[1], *(v for _, v in self.filters)]

    def on(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The statement's rows, computed on a table whose dates are ISO days."""
        day = frame[self.date]
        keep = (day >= self.period[0]) & (day <= self.period[1])
        for col, value in self.filters:
            keep &= frame[col].astype(str) == value
        for col, _ in self.keys:
            keep &= frame[col].notna()
        rows = frame.loc[keep]
        grouped = rows.groupby([self.date, *(c for c, _ in self.keys)], dropna=True)
        out = pd.DataFrame({alias: grouped[col].sum(min_count=1) for alias, col in self.sums}).reset_index()  # SUM: NULL where every value is
        out = out.rename(columns={self.date: "day", **{c: a for c, a in self.keys}})
        order = ["day", *(a for _, a in self.keys)]
        return out[[*order, *(a for a, _ in self.sums)]].sort_values(order).reset_index(drop=True)


class Data:
    """The tables a layer reads (dates as ISO days) and each dimension's values. With a ``runner`` (the host's: SQLite in the
    studio, its warehouse in the main product) every plan's SQL is run there; without one, its pandas computation is."""

    def __init__(self, layer: SemanticLayer, tables: Optional[Mapping[str, pd.DataFrame]] = None, runner: Optional[Runner] = None):
        self.layer, self.runner = layer, runner
        self.tables: dict[str, pd.DataFrame] = {}
        self.values: dict[str, list[str]] = {}
        self.days: dict[str, tuple[str, str]] = {}
        for t in layer.tables:
            if tables is not None and t.name in tables:
                frame = tables[t.name].copy()
                frame[t.date] = pd.to_datetime(frame[t.date], errors="coerce").dt.strftime("%Y-%m-%d")
                self.tables[t.name] = frame
                day = frame[t.date].dropna()
                self.days[t.name] = (str(day.min()), str(day.max()))
            elif runner is not None:
                got = runner(f"SELECT MIN({_ident(t.date)}) AS first, MAX({_ident(t.date)}) AS last FROM {_ident(t.name)}", [])
                self.days[t.name] = (str(got.iloc[0]["first"])[:10], str(got.iloc[0]["last"])[:10])
            else:
                raise ValueError(f"the layer reads the table {t.name}, which was not given")
        for d in layer.dimensions:
            seen: set[str] = set()
            for table, column in d.columns.items():
                if table in self.tables:
                    seen |= {str(v) for v in pd.unique(self.tables[table][column].dropna())}
                else:
                    seen |= {str(v) for v in runner(f"SELECT DISTINCT {_ident(column)} AS v FROM {_ident(table)}", [])["v"].dropna()}  # type: ignore[misc]
            self.values[d.name] = sorted(seen)

    @property
    def first(self) -> str:
        """The first day every table holds."""
        return max(a for a, _ in self.days.values())

    @property
    def last(self) -> str:
        """The last day every table holds."""
        return min(b for _, b in self.days.values())

    def run(self, plan: Plan) -> pd.DataFrame:
        if self.runner is not None:
            return self.runner(plan.sql, plan.params)
        return plan.on(self.tables[plan.table])


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table(layer: SemanticLayer, metric: str) -> str:
    return str(layer.metric(metric).table)


def daily(data: Data, sums: list[str], params: Params, period: tuple[str, str], by: Optional[str] = None,
          purpose: str = "numbers") -> tuple[pd.DataFrame, list[Query]]:
    """Each sum by day (and by one dimension's segment, where its table holds it): one tested query per table."""
    layer, out, queries = data.layer, None, []
    for table in dict.fromkeys(_table(layer, s) for s in sums):
        t = next(x for x in layer.tables if x.name == table)
        filters = [(col, value) for dim, value in params.filters.items() if (col := layer.column(dim, table)) is not None]
        # a table that does not hold a dimension is not split by it (every visitor counts for every category)
        seg = layer.column(by, table) if by else None
        plan = Plan(table, t.date, [(s, str(layer.metric(s).column)) for s in sums if _table(layer, s) == table], filters, period,
                    [(seg, "segment")] if seg else ())
        frame = data.run(plan)
        if seg:
            frame["segment"] = frame["segment"].astype(str)
        queries.append(Query(table=table, sql=plan.sql, params=plan.params, purpose=purpose))  # type: ignore[arg-type]
        if out is None:
            out = frame
        else:
            on = ["day"] + (["segment"] if "segment" in frame.columns and "segment" in out.columns else [])
            out = out.merge(frame, on=on, how="outer")
    return out if out is not None else pd.DataFrame(columns=["day", *sums]), queries


def _value(layer: SemanticLayer, name: str, totals: Mapping[str, Any]) -> Any:
    m = layer.metric(name)
    if m.kind == "sum":
        return totals[name]
    num, den = _value(layer, m.numerator, totals), _value(layer, m.denominator, totals)  # type: ignore[arg-type]
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def pairs(layer: SemanticLayer, names: Sequence[str]) -> list[tuple[str, str]]:
    """The (numerator, denominator) sums of every ratio among ``names``."""
    return list(dict.fromkeys((m.numerator, m.denominator) for m in map(layer.metric, names) if m.kind == "ratio"))  # type: ignore[misc]


def touching(layer: SemanticLayer, sums: Sequence[str]) -> list[tuple[str, str]]:
    """Every ratio of the layer with a side among ``sums``: its other side says when one of them is missing."""
    return [(n, d) for n, d in pairs(layer, [m.name for m in layer.metrics]) if n in sums or d in sums]


def coverage(data: Data, params: Params, period: tuple[str, str], ratios: Sequence[tuple[str, str]]) -> tuple[list[Coverage], list[Query]]:
    """Each ratio's two tables checked at the grain they share (the day and every dimension both hold), under the question's filters."""
    layer, out, queries = data.layer, [], []
    days = len(pd.date_range(period[0], period[1], freq="D"))
    for num, den in ratios:
        tn, td = _table(layer, num), _table(layer, den)
        grain = [d.name for d in layer.dimensions if d.columns.get(tn) and d.columns.get(td)]
        frames = []
        for side, table in ((num, tn), (den, td)):
            t = next(x for x in layer.tables if x.name == table)
            plan = Plan(table, t.date, [(side, str(layer.metric(side).column))],
                        [(col, v) for dim, v in params.filters.items() if (col := layer.column(dim, table)) is not None], period,
                        [(str(layer.column(g, table)), f"k_{g}") for g in grain])
            frames.append(data.run(plan))
            queries.append(Query(table=table, sql=plan.sql, params=plan.params, purpose="coverage"))
        keys = ["day", *(f"k_{g}" for g in grain)]
        a, b = (f.assign(**{k: f[k].astype(str) for k in keys}) for f in frames)
        both = a.merge(b, on=keys, how="outer", suffixes=("_n", "_d")) if num != den else a
        n_col = num if num in both.columns else f"{num}_n"
        d_col = den if den in both.columns else f"{den}_d"

        def label(row: Any) -> str:
            return " ".join([str(row["day"]), *(f"{g}={row[f'k_{g}']}" for g in grain)])

        n_pos, d_pos = both[n_col].fillna(0) > 0, both[d_col].fillna(0) > 0
        missing = [label(r) for _, r in both[n_pos & both[d_col].isna()].iterrows()]
        zero = [label(r) for _, r in both[n_pos & both[d_col].notna() & ~d_pos].iterrows()]
        num_missing = [label(r) for _, r in both[d_pos & both[n_col].isna()].iterrows()] if tn != td else []
        out.append(Coverage(numerator=num, denominator=den, period=period, days=days, grain=grain, missing=missing, zero=zero,
                            numerator_missing=num_missing))
    return out, queries


# ---------------------------------------------------------------------------------------------------- the numbers


def _counts(n: int, rng: np.random.Generator, b: int) -> np.ndarray:
    """``b`` bootstrap resamples of ``n`` days, as how many times each day is drawn."""
    draws = rng.integers(0, n, size=(b, n))
    return np.stack([np.bincount(row, minlength=n) for row in draws]) if n else np.zeros((b, 0))


def _ci(x: np.ndarray, level: float = LEVEL) -> Optional[tuple[float, float]]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    a = (1 - level) / 2
    return (round(float(np.quantile(x, a)), 6), round(float(np.quantile(x, 1 - a)), 6))


def _excludes_zero(ci: Optional[tuple[float, float]]) -> bool:
    return bool(ci is not None and (ci[0] > 0 or ci[1] < 0))


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.4g}"


def _pct(log_change: float) -> float:
    return round(100 * (math.exp(log_change) - 1), 3) if np.isfinite(log_change) else float("nan")


def _days(period: tuple[str, str]) -> int:
    return len(pd.date_range(period[0], period[1], freq="D"))


class _Periods:
    """Daily sums of the two periods, per calendar day of each period (so periods of unequal length compare), and the bootstrap's day
    counts drawn once for every number of the answer."""

    def __init__(self, now: pd.DataFrame, then: pd.DataFrame, sums: list[str], days: tuple[int, int], rng: np.random.Generator, b: int):
        self.sums, self.days = sums, days
        self.now = now.groupby("day")[sums].sum() if len(now) else pd.DataFrame(columns=sums)
        self.then = then.groupby("day")[sums].sum() if len(then) else pd.DataFrame(columns=sums)
        self.cn, self.ct = _counts(len(self.now), rng, b), _counts(len(self.then), rng, b)

    def totals(self) -> tuple[dict[str, float], dict[str, float]]:
        return ({s: float(self.now[s].sum()) for s in self.sums}, {s: float(self.then[s].sum()) for s in self.sums})

    def per_day(self) -> tuple[dict[str, float], dict[str, float]]:
        n, t = self.totals()
        return ({s: v / self.days[0] for s, v in n.items()}, {s: v / self.days[1] for s, v in t.items()})

    def boot(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        return ({s: self.cn @ self.now[s].to_numpy(dtype=float) / self.days[0] for s in self.sums},
                {s: self.ct @ self.then[s].to_numpy(dtype=float) / self.days[1] for s in self.sums})


def _log(a: Any, b: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.asarray(a, dtype=float)) - np.log(np.asarray(b, dtype=float))


def drill(data: Data, factor: str, dimension: str, now: pd.DataFrame, then: pd.DataFrame, rng: np.random.Generator, b: int,
          level: float = LEVEL, toward: int = 0, days: tuple[int, int] = (1, 1)) -> Optional[Drill]:
    """One factor's change by one dimension's segments, per calendar day of each period. A sum: each segment's change over the whole's
    previous value. A ratio N/D: where both tables hold the dimension, mix (the segment's weight in D moving, against the previous
    average) and rate (its own ratio moving), adding up to the change; where only N's table holds it, every segment's N over the shared D
    (rate only); where only D's table holds it, no drill (N cannot be split). A segment with a positive numerator over a zero denominator
    in either period, or any number that is not finite, gives no drill. The lead is the segment that moved the factor most ``toward`` the
    way it moved (either way when 0)."""
    layer = data.layer
    m = layer.metric(factor)
    sums = layer.base(factor)
    has = {s: layer.column(dimension, _table(layer, s)) is not None for s in sums}
    if not any(has.values()):
        return None
    if m.kind == "ratio" and not has[str(m.numerator)]:
        return None  # only the denominator's table holds the dimension: the numerator cannot be split by it
    segs = sorted(set(now.get("segment", pd.Series(dtype=str)).dropna()) | set(then.get("segment", pd.Series(dtype=str)).dropna()))
    if len(segs) < 2:
        return None
    dn, dt = sorted(now["day"].unique()), sorted(then["day"].unique())
    cn, ct = _counts(len(dn), rng, b), _counts(len(dt), rng, b)

    def cube(frame: pd.DataFrame, ds: list[str], s: str) -> np.ndarray:
        """days x segments of a sum (a sum its table does not split: the same day total in every column)."""
        if has[s]:
            p = frame.pivot_table(index="day", columns="segment", values=s, aggfunc="sum").reindex(index=ds, columns=segs)
            return p.fillna(0).to_numpy(dtype=float)
        t = frame.drop_duplicates("day").set_index("day")[s].reindex(ds) if "segment" in frame.columns else frame.groupby("day")[s].sum().reindex(ds)
        return np.repeat(t.fillna(0).to_numpy(dtype=float)[:, None], len(segs), axis=1)

    cubes_n = {s: cube(now, dn, s) / days[0] for s in sums}
    cubes_t = {s: cube(then, dt, s) / days[1] for s in sums}

    def contributions(cnt_n: np.ndarray, cnt_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(mix, rate, the segments' own change) per resample row: in per cent of the factor's previous value."""
        tot_n = {s: cnt_n @ cubes_n[s] for s in sums}  # resamples x segments
        tot_t = {s: cnt_t @ cubes_t[s] for s in sums}
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if m.kind == "sum":
                prev = tot_t[factor].sum(axis=1, keepdims=True)
                rate = (tot_n[factor] - tot_t[factor]) / prev
                own = tot_n[factor] / tot_t[factor] - 1
                return np.zeros_like(rate), 100 * rate, 100 * own
            num, den = str(m.numerator), str(m.denominator)  # two sums (the layer refuses a ratio of ratios)
            if has[den]:
                wn, wt = tot_n[den] / tot_n[den].sum(axis=1, keepdims=True), tot_t[den] / tot_t[den].sum(axis=1, keepdims=True)
                rn, rt = tot_n[num] / tot_n[den], tot_t[num] / tot_t[den]
                rt_all = tot_t[num].sum(axis=1, keepdims=True) / tot_t[den].sum(axis=1, keepdims=True)
                rn, rt = np.where(tot_n[den] > 0, rn, 0.0), np.where(tot_t[den] > 0, rt, 0.0)
                mix = (wn - wt) * (rt - rt_all) / rt_all
                rate = wn * (rn - rt) / rt_all
                return 100 * mix, 100 * rate, 100 * (np.where(rt > 0, rn / rt, np.nan) - 1)
            shared_n, shared_t = tot_n[den][:, :1], tot_t[den][:, :1]  # the denominator is not split: one total in every column
            rt_all = tot_t[num].sum(axis=1, keepdims=True) / shared_t
            rate = (tot_n[num] / shared_n - tot_t[num] / shared_t) / rt_all
            return np.zeros_like(rate), 100 * rate, 100 * (tot_n[num] / tot_t[num] * shared_t / shared_n - 1)

    if m.kind == "ratio" and has[str(m.denominator)]:
        num, den = str(m.numerator), str(m.denominator)
        for cubes in (cubes_n, cubes_t):  # a positive numerator over a zero denominator: a gap, never a number
            if bool(((cubes[num].sum(axis=0) > 0) & (cubes[den].sum(axis=0) <= 0)).any()):
                return None
    point = contributions(np.ones((1, len(dn))), np.ones((1, len(dt))))
    mix, rate, own = (x[0] for x in point)
    total = mix + rate
    if not (np.isfinite(mix).all() and np.isfinite(rate).all()):
        return None
    boots = contributions(cn, ct)
    weight = np.abs(total)
    if not weight.sum() > 0:
        return None
    lead = int(np.argmax(toward * total)) if toward else int(np.argmax(weight))
    effect: Literal["level", "rate", "mix"] = "level" if m.kind == "sum" else ("mix" if abs(mix[lead]) > abs(rate[lead]) else "rate")
    order = np.argsort(-weight, kind="stable")
    segments = [Segment(value=segs[i], contribution_pct=round(float(total[i]), 4), mix_pct=round(float(mix[i]), 4), rate_pct=round(float(rate[i]), 4),
                        ci=_ci(boots[0][:, i] + boots[1][:, i], level), change_pct=None if not np.isfinite(own[i]) else round(float(own[i]), 3))
                for i in order]
    return Drill(factor=factor, dimension=dimension, lead=segs[lead], concentration=round(float(weight[lead] / weight.sum()), 4), effect=effect,
                 segments=segments, level=round(level, 6))


# ---------------------------------------------------------------------------------------------------- choosing a template


CAUSAL = re.compile(r"\b(cause|caused|causing|because of|due to|the reason|blame|blamed|effect of|impact of|behind|hurt|hurts|hit|affect|affected|"
                    r"drove|driven)\b")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip()


def _find(words: str, phrase: str) -> Optional[int]:
    m = re.search(r"(?<![a-z0-9])" + re.escape(_norm(phrase)) + r"(?![a-z0-9])", words)
    return m.start() if m else None


def _month_range(year: int, month: int) -> tuple[str, str]:
    first = pd.Timestamp(year=year, month=month, day=1)
    return (str(first.date()), str((first + pd.offsets.MonthEnd(0)).date()))


def _compared(period: tuple[str, str], compare: str) -> tuple[str, str]:
    a, b = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    if compare == "last_year":
        return (str((a - pd.DateOffset(years=1)).date()), str((b - pd.DateOffset(years=1)).date()))
    if a.day == 1 and (b + pd.Timedelta(days=1)).day == 1:  # a whole month: the month before
        before = (a - pd.Timedelta(days=1)).replace(day=1)
        return (str(before.date()), str((before + pd.offsets.MonthEnd(0)).date()))
    length = (b - a).days + 1
    return (str((a - pd.Timedelta(days=length)).date()), str((a - pd.Timedelta(days=1)).date()))


def _last_whole_month(data: "Data") -> pd.Timestamp:
    last = pd.Timestamp(data.last)
    return last if (last + pd.Timedelta(days=1)).day == 1 else last.replace(day=1) - pd.Timedelta(days=1)


def _names(items: Sequence[str], most: int = 12) -> str:
    items = list(items)
    return ", ".join(items[:most]) + (f" and {len(items) - most} more" if len(items) > most else "")


NAMED = re.compile(r"^\s*(metric_value|metric_change|why_changed|top_segments)\s*:\s*(.*)$", re.S)
NAMED_KEYS = {"metric", "period", "compare", "dimension", "direction"}


def named(question: str, data: Data) -> Optional[Choice]:
    """A question that names its template (``why_changed: metric=sales; brand=Arlo; period=2026-09; compare=last_year``), parsed exactly:
    a key that is neither a parameter nor a dimension is refused, never dropped."""
    m = NAMED.match(question)
    if not m:
        return None
    template, rest = m.group(1), m.group(2)
    fields: dict[str, str] = {}
    for part in re.split(r"[;\n]", rest):
        if not part.strip():
            continue
        if "=" not in part:
            return _refused(f"'{part.strip()[:80]}' is not a key=value parameter.", ["parameter"], "named")
        k, v = part.split("=", 1)
        fields[k.strip().lower()] = v.strip()
    unknown = sorted(set(fields) - NAMED_KEYS - set(data.values))
    if unknown:
        return _refused(f"{', '.join(unknown)} is neither a template's parameter ({', '.join(sorted(NAMED_KEYS))}) nor a dimension ({_names(data.values)}).",
                        ["parameter"], "named")
    period = None
    if fields.get("period"):
        p = fields["period"]
        if re.fullmatch(r"\d{4}-\d{2}", p):
            period = _month_range(int(p[:4]), int(p[5:7]))
        elif ".." in p:
            a, b = p.split("..", 1)
            period = (a.strip(), b.strip())
        else:
            return _refused(f"The period '{p[:40]}' is neither a month (YYYY-MM) nor a range (first..last).", ["period"], "named")
    raw = {"template": template, "params": {"metric": fields.get("metric"), "filters": {k: v for k, v in fields.items() if k in data.values}, "period": period,
                                            "compare": fields.get("compare"), "dimension": fields.get("dimension"), "direction": fields.get("direction")}}
    try:
        return _validated(raw, data, "named")
    except Unanswerable as why:
        return _refused(why.reason, why.missing, "named")
    except ValueError as why:
        return _refused(str(why), ["parameter"], "named")


def _refused(reason: str, missing: list[str], source: str) -> Choice:
    return Choice(refused=True, missing=missing[:10] or ["something the data does not hold"], reason=reason[:4000], source=source)  # type: ignore[arg-type]


def _validated(raw: Mapping[str, Any], data: Data, source: str) -> Choice:
    """A choice checked against the layer and the data: the template known, its parameters present and approved, every filter's value in the
    data, the period and its comparison inside the data. Raises ValueError (Unanswerable for what the data cannot answer) with what is wrong."""
    layer = data.layer
    if raw.get("refused"):  # a concept the layer does not hold is named as the layer names it, whether the reply gave its name or its needs
        missing = [str(x) for x in (raw.get("missing") or [])]
        named_as = [next((nh.name for nh in layer.not_held if any(_find(_norm(x), w) is not None for w in [nh.name, *nh.synonyms])
                          or _norm(x) == _norm(nh.needs)), x) for x in missing]
        return _refused(str(raw.get("reason") or ""), list(dict.fromkeys(named_as)), source)
    template = raw.get("template")
    if template not in TEMPLATES:
        raise ValueError(f"no tested template is named {template!r}")
    p = dict(raw.get("params") or {})
    unknown = sorted(set(p) - {"metric", "filters", "period", "compare", "dimension", "direction"})
    if unknown:
        raise ValueError(f"{', '.join(unknown)} is not a template's parameter")
    metric = p.get("metric")
    if metric not in {m.name for m in layer.metrics}:
        raise Unanswerable(f"{metric!r} is not an approved metric ({_names([m.name for m in layer.metrics])}).", ["metric"])
    filters = {str(k): str(v) for k, v in (p.get("filters") or {}).items()}
    for k, v in filters.items():
        if k not in data.values:
            raise Unanswerable(f"{k!r} is not a dimension of the layer ({_names(data.values)}).", ["dimension"])
        if v not in data.values[k]:
            raise Unanswerable(f"{v!r} is not a {k} in the data.", [k])
    compare = p.get("compare")
    if compare is not None and compare not in ("last_year", "previous_period"):
        raise ValueError(f"{compare!r} is not a comparison")
    period = p.get("period")
    if period is not None:
        try:
            period = (str(pd.Timestamp(period[0]).date()), str(pd.Timestamp(period[1]).date()))
        except (ValueError, TypeError, IndexError) as exc:
            raise ValueError(f"the period {period!r} is not two days") from exc
        spans = [("the period", period)] + ([("its comparison", _compared(period, compare))] if compare and TEMPLATES[template]["needs"].count("compare") else [])
        for what, (a, b) in spans:
            if a > b or a < data.first or b > data.last:
                raise Unanswerable(f"{what} {a} to {b} is not inside the data ({data.first} to {data.last}).", ["period"])
    dimension = p.get("dimension")
    if dimension is not None and dimension not in data.values:
        raise Unanswerable(f"{dimension!r} is not a dimension of the layer ({_names(data.values)}).", ["dimension"])
    direction = p.get("direction") or None
    if direction is not None and direction not in ("down", "up"):
        raise ValueError(f"{direction!r} is not a direction (down or up)")
    for need in TEMPLATES[template]["needs"]:
        if {"metric": metric, "period": period, "compare": compare, "dimension": dimension}[need] is None:
            raise Unanswerable(f"{template} needs its {need}.", [need])
    return Choice(template=template, params=Params(metric=metric, filters=filters, period=period, compare=compare, dimension=dimension, direction=direction),
                  source=source, assumed=[str(x) for x in (raw.get("assumed") or [])][:10], notes=[str(x) for x in (raw.get("notes") or [])][:10])  # type: ignore[arg-type]


def rules(question: str, data: Data) -> Choice:
    """The deterministic reading of a question: what the data does not hold (refused when the question asks about it as a cause or names no
    approved metric, else said beside the answer), the metric, the filters (every dimension's values found by name), the period (a month and
    year, a month alone at its latest whole occurrence in the data, an ISO month, "last month"; else the last whole month, said), the
    comparison (else last year, said) and the template (why, which ... most, change, else the value)."""
    layer, words = data.layer, _norm(question)
    found = sorted(((pos, len(s), m.name) for m in layer.metrics for s in [m.name, m.label, *m.synonyms] if (pos := _find(words, s)) is not None),
                   key=lambda x: (x[0], -x[1]))
    notes = []
    for nh in layer.not_held:
        if any(_find(words, s) is not None for s in [nh.name, *nh.synonyms]):
            if not found or CAUSAL.search(words):
                return _refused(f"The data does not hold {nh.name}: it would need {nh.needs}.", [nh.name], "rules")
            notes.append(f"The data does not hold {nh.name} (it would need {nh.needs}): the answer says what moved, not whether {nh.name} did.")
    if not found:
        return _refused("The question names no approved metric (" + _names([m.name for m in layer.metrics]) + ").", ["metric"], "rules")
    metric = found[0][2]
    filters: dict[str, str] = {}
    for dim, values in data.values.items():
        hits = sorted((pos, v) for v in values if (pos := _find(words, v)) is not None)
        if hits:
            filters[dim] = hits[0][1]
    assumed: list[str] = []
    whole = _last_whole_month(data)
    with_year = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(\d{4})\b", words)
    alone = re.search(r"\b(" + "|".join(MONTHS) + r")\b", words)
    iso = re.search(r"\b(\d{4})-(\d{2})\b", words)
    if with_year:
        period = _month_range(int(with_year.group(2)), MONTHS[with_year.group(1)])
    elif iso:
        period = _month_range(int(iso.group(1)), int(iso.group(2)))
    elif alone:  # a month without its year: its latest whole occurrence in the data, said
        month = MONTHS[alone.group(1)]
        year = whole.year if month <= whole.month else whole.year - 1
        period = _month_range(year, month)
        assumed.append(f"the year: {alone.group(1).capitalize()} {year}, the latest whole one in the data")
    else:
        period = _month_range(whole.year, whole.month)
        if not re.search(r"\blast month\b|\bthis month\b", words):
            assumed.append(f"the period: the last whole month in the data ({period[0][:7]})")
    if re.search(r"\b(previous|prior) (month|period)\b|\bmonth before\b|\bthe month earlier\b", words):
        compare = "previous_period"
    elif re.search(r"\blast year\b|\ba year (ago|before|earlier)\b|\byear (over|on) year\b|\byoy\b", words):
        compare = "last_year"
    else:
        compare = "last_year"
        assumed.append("the comparison: the same period last year")
    dimension = None
    which = re.search(r"\bwhich (\w+)", words)
    if re.search(r"\bwhy\b|\bwhat (caused|explains|drove)\b|\bexplain\b|\breason\b", words):
        template = "why_changed"
    elif which and re.search(r"\b(most|top|best|worst|highest|lowest)\b", words):
        template = "top_segments"
        dimension = next((d.name for d in layer.dimensions if any(_find(which.group(0), s) is not None for s in [d.name, d.label, *d.synonyms])), None)
        if dimension is None:
            return _refused("Which segments? The layer's dimensions are " + _names(data.values) + ".", ["dimension"], "rules")
    elif re.search(r"\b(change|changed|compare|against|versus|vs|up|down|rise|rose|fall|fell|drop|dropped)\b", words):
        template = "metric_change"
    else:
        template = "metric_value"
    if template in ("metric_value", "top_segments"):
        compare, assumed = None, [a for a in assumed if not a.startswith("the comparison")]
    direction = ("down" if re.search(r"\b(down|drop|dropped|fall|fell|falling|lower|decline|declined|less)\b", words) else
                 "up" if re.search(r"\b(up|rise|rose|rising|higher|grew|increase|increased|more)\b", words) else None)
    try:
        return _validated({"template": template, "params": {"metric": metric, "filters": filters, "period": period, "compare": compare, "dimension": dimension,
                                                            "direction": direction}, "assumed": assumed, "notes": notes}, data, "rules")
    except Unanswerable as why:
        return _refused(why.reason, why.missing, "rules")


MENU_VALUES = 150  # at most this many dimension values in one request, in all (the purpose's data limit)


def menu(data: Data) -> dict[str, Any]:
    """What the advisor sees: the templates and their parameters, the approved metrics, the dimensions and their values (at most
    ``MENU_VALUES`` in all, shared among the dimensions), the concepts the data does not hold, and the data's dates. Names and
    definitions only, never a row."""
    layer = data.layer
    each = max(1, min(50, MENU_VALUES // max(len(data.values), 1)))
    return {"templates": {k: v["says"] + " (needs " + ", ".join(v["needs"]) + ")" for k, v in TEMPLATES.items()},
            "metrics": {m.name: f"{m.label}" + (f" (also: {', '.join(m.synonyms)})" if m.synonyms else "") for m in layer.metrics},
            "dimensions": {d: v[:each] for d, v in list(data.values.items())[:MENU_VALUES]},
            "not_held": {n.name: n.needs for n in layer.not_held},
            "dates": {"first": data.first, "last": data.last},
            "answer_as": "JSON: {\"template\": name, \"params\": {\"metric\", \"filters\": {dimension: value}, \"period\": [first day, last day], "
                         "\"compare\": \"last_year\"|\"previous_period\", \"dimension\", \"direction\": \"down\"|\"up\" (as the person says it moved)}} "
                         "or {\"refused\": true, \"missing\": [what], \"reason\": why}"}


def choose(question: str, data: Data, advisor: Optional[Advisor] = None,
           on_error: Optional[Callable[[str, Optional[BaseException]], None]] = None) -> tuple[Choice, dict[str, Any]]:
    """The question's template and parameters: named exactly when the question names its template (no model asked), else the advisor's
    validated choice, else the rules'. Returns the choice and what each reader said, the model's reply whole (kept for the record)."""
    from dclab_core.advisor import EvidenceError

    said: dict[str, Any] = {}
    exact = named(question, data)
    if exact is not None:
        return exact, {"named": exact.model_dump(mode="json")}
    by_rules = rules(question, data)
    said["rules"] = by_rules.model_dump(mode="json")
    if advisor is None:
        return by_rules, said
    errors: list[str] = []

    def failed(kind: str, exc: Optional[BaseException]) -> None:
        errors.append(f"{kind}: {exc}" if exc else kind)
        if on_error is not None:
            on_error(kind, exc)

    def parse(reply: Mapping[str, Any]) -> Choice:
        said["model_raw"] = dict(reply)  # the reply as it came, before any check: kept whole
        return _validated(reply, data, "model")

    try:
        chosen = consult(advisor, PURPOSE, {"question": question, "menu": menu(data)}, parse, on_error=failed)
    except EvidenceError as exc:  # the menu would carry more than a summary: the rules' reading stands
        failed("evidence", exc)
        chosen = None
    if errors:
        said["model_error"] = errors
    if chosen is None:
        return by_rules, said
    said["model"] = chosen.model_dump(mode="json")
    return chosen, said


# ---------------------------------------------------------------------------------------------------- answering


ACTIONS = {
    ("footfall", "down"): "Fewer people came in: check what kept visitors away (access, works, events, the mall's own traffic) with the landlord, and "
                          "bring them back with local offers.",
    ("footfall", "up"): "More people came in: find what brought them (an event, an opening nearby) and repeat it.",
    ("conversion", "down"): "Visitors came but fewer bought: check stock and sizes{where}, staffing at the busy hours, and target offers at visitors who "
                            "leave without buying.",
    ("conversion", "up"): "More visitors bought{where}: keep what changed (stock, staff, an offer) and test it in other stores.",
    ("basket", "down"): "Each paying visit spent less{where}: check discounts and prices, and cross-sell with the larger categories.",
    ("basket", "up"): "Each paying visit spent more{where}: keep the bundles and prices that did it.",
    ("basket", "down", "mix"): "Baskets fell because more of the sales came from cheaper items{where}: review the promotion that moved the mix, and "
                               "cross-sell from it to the larger categories.",
}


def _action(factor: Optional[str], direction: str, segment: Optional[dict[str, str]]) -> str:
    if factor is None or direction == "none":
        return ""
    where = f" in {segment['value']}" if segment else ""
    key = (factor, direction, segment.get("effect")) if segment and (factor, direction, segment.get("effect")) in ACTIONS else (factor, direction)
    text = ACTIONS.get(key)  # type: ignore[call-overload]
    return text.format(where=where) if text else f"{factor} moved {direction}{where}: look into what changed there first."


def ask(question: str, layer: SemanticLayer, tables: Optional[Mapping[str, pd.DataFrame]], advisor: Optional[Advisor] = None, *,
        data: Optional[Data] = None, seed: int = SEED, bootstrap: int = BOOTSTRAP,
        on_error: Optional[Callable[[str, Optional[BaseException]], None]] = None) -> tuple[MetricAnswer, dict[str, Any]]:
    """Answer one question: the choice, its tested queries (run by the host's runner, or computed on ``tables``), coverage, the numbers
    and, for a why, the factors, the drill-down and an action. Returns the answer and what each reader said about the choice (the record
    keeps both whole)."""
    data = data or Data(layer, tables)
    choice, said = choose(question, data, advisor, on_error=on_error)
    if choice.refused:
        return MetricAnswer(question=question, choice=choice, status="refused", refused_because=choice.reason, missing=list(choice.missing),
                            words=choice.reason, notes=list(choice.notes)), said
    return answer(question, choice, data, seed=seed, bootstrap=bootstrap), said


def answer(question: str, choice: Choice, data: Data, *, seed: int = SEED, bootstrap: int = BOOTSTRAP) -> MetricAnswer:
    """The numbers for a validated choice."""
    layer, p, rng = data.layer, choice.params, np.random.default_rng(seed)
    assert p.metric is not None and p.period is not None
    label = layer.metric(p.metric).label
    factors_of = layer.factors_of(p.metric) if choice.template == "why_changed" else [p.metric]
    sums = list(dict.fromkeys(layer.base(p.metric) + [s for f in factors_of for s in layer.base(f)]))  # every sum the answer reads
    checks = touching(layer, sums)  # every ratio with a side the answer reads: its two tables say when a day is missing
    own = touching(layer, layer.base(p.metric))  # those of the metric's own sums
    notes = [f"Assumed {a}." for a in choice.assumed] + list(choice.notes)
    for dim in p.filters:
        unsplit = sorted({_table(layer, s) for s in sums if layer.column(dim, _table(layer, s)) is None})
        if unsplit:
            notes.append(f"{', '.join(unsplit)} does not hold {dim}: its numbers are every {dim} together, within the other filters.")
    base: dict[str, Any] = {"question": question, "choice": choice, "metric": p.metric, "period": p.period, "notes": notes}
    periods = [p.period] + ([_compared(p.period, p.compare or "last_year")] if choice.template in ("metric_change", "why_changed") else [])
    cov, cov_queries = [], []
    for span in periods:
        c, q = coverage(data, p, span, checks)
        cov += c
        cov_queries += q
    gaps = [c for c in cov if c.gaps]
    own_gap = any(c.gaps for c in cov if (c.numerator, c.denominator) in own)
    if choice.template == "top_segments":
        frame, queries = daily(data, sums, p, p.period, by=p.dimension)
        queries += cov_queries
        if own_gap:
            return MetricAnswer(**base, status="refused", queries=queries, sources=sorted({q.table for q in queries}), coverage=cov,
                                missing=_gap_names(layer, gaps), refused_because=_gap_words(gaps) + " The segments cannot be ranked until those days are back.",
                                words=f"Not answered: {', '.join(_gap_names(layer, gaps))} is missing on some days, so {label} cannot be ranked.")
        if "segment" not in frame.columns:
            return MetricAnswer(**base, status="refused", queries=queries, refused_because=f"No table of {label} holds {p.dimension}.",
                                missing=[str(p.dimension)], words=f"Not answered: no table of {label} holds {p.dimension}.")
        totals = frame.groupby("segment")[sums].sum()
        values = pd.Series(_value(layer, p.metric, {s: totals[s] for s in sums}), index=totals.index)
        rows = [{"value": str(k), "metric": round(float(v), 4)} for k, v in values.sort_values(ascending=False).items() if np.isfinite(v)]
        return MetricAnswer(**base, status="answered", segments=rows, queries=queries, sources=sorted({q.table for q in queries}), coverage=cov,
                            words=f"{rows[0]['value']} leads on {label} ({_fmt(rows[0]['metric'])})." if rows else "No segment has a value.")
    now, q_now = daily(data, sums, p, p.period)
    if choice.template == "metric_value":
        queries = q_now + cov_queries
        totals = {s: float(now[s].sum()) if s in now.columns and len(now) else 0.0 for s in sums}
        if not len(now):  # the review: a store that does not exist has no rows, and no number
            return MetricAnswer(**base, status="refused", queries=queries, sources=sorted({q.table for q in queries}), coverage=cov, missing=sorted(sums),
                                refused_because=f"No row holds {label} for these filters over the period: there is no such number in the data.",
                                words=f"Not answered: the data has no row of {label} for these filters over the period.")
        defined = layer.metric(p.metric).kind == "sum" or all(totals[s] > 0 for s in layer.base(p.metric))
        value = float(_value(layer, p.metric, totals)) if defined else float("nan")
        if own_gap or not np.isfinite(value):
            return MetricAnswer(**base, status="refused", queries=queries, sources=sorted({q.table for q in queries}), coverage=cov,
                                refused_because=_gap_words(gaps) or f"{label} is not defined over the period (a zero denominator).",
                                missing=_gap_names(layer, gaps), words=f"Not answered: {label} cannot be read over the period.")
        return MetricAnswer(**base, status="answered", current=round(value, 6), queries=queries, sources=sorted({q.table for q in queries}), coverage=cov,
                            words=f"{label}: {_fmt(value)} from {p.period[0]} to {p.period[1]}.")
    compare = periods[1]
    then, q_then = daily(data, sums, p, compare)
    queries, days = q_now + q_then + cov_queries, (_days(p.period), _days(compare))
    base.update(compare=compare, days=days)
    if days[0] != days[1] and any(layer.metric(s).kind == "sum" for s in [p.metric, *factors_of]):
        notes.append(f"Compared per day: the period has {days[0]} days and its comparison {days[1]}.")
    sources = sorted({q.table for q in queries})
    periods_ = _Periods(now, then, sums, days, rng, bootstrap)
    tn_total, tt_total = periods_.totals()
    if any(not tn_total[s] > 0 or not tt_total[s] > 0 for s in sums):
        empty = sorted(s for s in sums if not (tn_total[s] > 0 and tt_total[s] > 0))
        return MetricAnswer(**base, status="refused", queries=queries, sources=sources, coverage=cov, missing=empty,
                            refused_because=f"{', '.join(empty)} is zero or absent in one of the periods: there is no change to explain.",
                            words=f"Not answered: {', '.join(empty)} is zero or absent in one of the periods.")
    tn, tt = periods_.per_day()
    bn, bt = periods_.boot()
    cur, prev = float(_value(layer, p.metric, tn_total)), float(_value(layer, p.metric, tt_total))
    log_change = float(_log(_value(layer, p.metric, tn), _value(layer, p.metric, tt)))
    ci_log = _ci(_log(_value(layer, p.metric, bn), _value(layer, p.metric, bt)))
    ci = (_pct(ci_log[0]), _pct(ci_log[1])) if ci_log else None
    moved = _excludes_zero(ci_log)
    said = f"{label} changed {_pct(log_change):+.1f}%" + (f" (90% interval {ci[0]:+.1f}% to {ci[1]:+.1f}%)" if ci else "")
    numbers = {"current": round(cur, 6), "previous": round(prev, 6), "change_pct": _pct(log_change), "ci90": ci, "moved": moved}
    if gaps and (own_gap or choice.template == "why_changed"):
        partial = "" if own_gap else f" {said}, from tables without gaps."
        missing = _gap_names(layer, gaps)
        return MetricAnswer(**base, **({} if own_gap else numbers), status="refused", queries=queries, sources=sources, coverage=cov, missing=missing,
                            refused_because=_gap_words(gaps) + f" The change {'of ' + label + ' cannot be read' if own_gap else 'cannot be split'} until "
                                            "those days are back." + partial,
                            words=f"Not answered: {', '.join(missing)} is missing on some days, so "
                                  f"{label + ' cannot be read' if own_gap else 'the change cannot be split'}.{partial}")
    base.update(numbers)
    if choice.template == "metric_change":
        return MetricAnswer(**base, status="answered", queries=queries, sources=sources, coverage=cov,
                            words=said + ("." if moved else ", within its day-to-day variation."))
    factors, flevel = [], 1 - (1 - LEVEL) / 2 / len(factors_of)  # one chance in ten, in all, that a factor or a segment is said to move by chance: half to the factors
    for f in factors_of:
        fl = float(_log(_value(layer, f, tn), _value(layer, f, tt)))
        fci = _ci(_log(_value(layer, f, bn), _value(layer, f, bt)), flevel)
        factors.append(Factor(name=f, label=layer.metric(f).label, log_change=round(fl, 6), change_pct=_pct(fl),
                              share=round(fl / log_change, 4) if abs(log_change) > 1e-12 else None, ci=fci, level=round(flevel, 6), moved=_excludes_zero(fci)))
    said_way = {"down": -1, "up": 1}.get(p.direction or "")
    toward = (-1 if log_change < 0 else 1) if moved else (said_way or 0)  # 0: neither moved nor said: the largest move either way
    if moved and said_way and said_way != toward:
        notes.append(f"The question says {p.direction}; {label} moved {'down' if toward < 0 else 'up'}.")
    ranked = sorted(factors, key=lambda f: -toward * f.log_change if toward else -abs(f.log_change))
    first = next((f for f in ranked if f.moved and (toward * f.log_change > 0 if toward else True)), None)  # the review: any factor that moved, not only the first ranked
    lead = first or ranked[0]
    lead_way = (-1 if lead.log_change < 0 else 1) if first is not None else toward
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for dim in data.values:
        if dim in p.filters:
            continue
        fn, qn = daily(data, layer.base(lead.name), p, p.period, by=dim, purpose="drill")
        ft, qt = daily(data, layer.base(lead.name), p, compare, by=dim, purpose="drill")
        if "segment" in fn.columns:
            frames[dim] = (fn, ft)
            queries += qn + qt
    searched = sum(len(set(fn["segment"].dropna()) | set(ft["segment"].dropna())) for fn, ft in frames.values())
    slevel = 1 - (1 - LEVEL) / 2 / max(searched, 1)  # the other half to every segment of every dimension searched
    drills = [d for dim, (fn, ft) in frames.items() if (d := drill(data, lead.name, dim, fn, ft, rng, bootstrap, slevel, lead_way, days)) is not None]
    drills.sort(key=lambda d: -d.concentration)
    seg = None
    for d in drills:  # where: the most concentrated dimension whose lead moved, beyond its variation, the way the factor moved (or was said to)
        top = next(s for s in d.segments if s.value == d.lead)
        way = lead_way or (1 if top.contribution_pct > 0 else -1)
        if top.ci is not None and way * top.contribution_pct > 0 and (top.ci[0] > 0 if way > 0 else top.ci[1] < 0):
            seg = {"dimension": d.dimension, "value": top.value, "effect": d.effect}
            break
    fdir = "none" if first is None else ("down" if first.log_change < 0 else "up")
    action = _action(first.name if first else None, fdir, seg)
    if first is None:
        words = said + ": no factor moved beyond its day-to-day variation" + (f"; {seg['dimension']} {seg['value']} moved {lead.label} most." if seg else ".")
    else:
        rest = ", ".join(f"{f.label} {f.change_pct:+.1f}%" for f in ranked if f is not first)
        words = (said + f". {first.label} moved most ({first.change_pct:+.1f}%)" + (f", led by {seg['dimension']} {seg['value']} ({seg['effect']})" if seg else "")
                 + (f"; {rest}." if rest else "."))
    return MetricAnswer(**base, status="answered", factors=ranked, first_factor=first.name if first else None, drills=drills, first_segment=seg,
                        queries=queries, sources=sources, coverage=cov, action=action,
                        follow_up="Measure the same factor next period against this one, to see whether the action helped." if action else "", words=words)


def _gap_names(layer: SemanticLayer, gaps: list[Coverage]) -> list[str]:
    """The sums missing somewhere: a denominator with no row or zero, a numerator with no row."""
    out = []
    for c in gaps:
        if c.missing or c.zero:
            out.append(c.denominator)
        if c.numerator_missing:
            out.append(c.numerator)
    return sorted(set(out))


def _gap_words(gaps: list[Coverage]) -> str:
    if not gaps:
        return ""
    return "The data has gaps: " + "; ".join(
        f"from {c.period[0]} to {c.period[1]}, {c.denominator} has {len(c.missing)} days with no row and {len(c.zero)} reading zero while "
        f"{c.numerator} is positive, and {c.numerator} has {len(c.numerator_missing)} days with no row while {c.denominator} is positive"
        + (f" (by day and {', '.join(c.grain)})" if c.grain else "") for c in gaps if c.gaps) + "."


__all__ = ["SemanticLayer", "Metric", "Dimension", "Factorization", "NotHeld", "Table", "Params", "Choice", "MetricAnswer", "Factor", "Drill",
           "Segment", "Query", "Coverage", "Data", "Plan", "Runner", "Unanswerable", "TEMPLATES", "PURPOSE", "ask", "answer", "choose", "rules",
           "named", "menu", "daily", "drill", "coverage", "pairs", "touching"]
