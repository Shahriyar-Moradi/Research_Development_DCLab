"""The monitoring plan (package 30.15, the framework's step 12): what a model's inputs and scores looked like on the
training rows, written into its artifact with the thresholds that raise an alert, and ``monitor(plan, new_rows)`` that
says whether new rows still look like them.

Each column is kept as the training rows' bins (numeric: deciles' edges; a category: its most common values and the rest
together) with the share of rows in each, and the score as its own bins; the flagged share is the share of training
rows the operating rule flags. ``monitor`` reads a batch: the population stability index (PSI) of every column and of the
score against its bins, a warning at ``psi_warn`` and an alert at ``psi_alert``, and the flagged share against its band
(a binomial interval at the batch's size around the training share, widened by ``share_slack``). A batch below
``min_rows`` is read and said, never alerted on.

The plan holds the bins' edges and the categories' names, which a model's artifact needs; it never holds a row. A column
the host names as personal is left out.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

PSI_WARN, PSI_ALERT = 0.10, 0.25  # the conventional bands: under 0.1 stable, 0.1 to 0.25 a shift to watch, above 0.25 a shift to act on
MIN_ROWS = 200
TOP_CATEGORIES = 20
MAX_COLUMNS = 100
OTHER, MISSING = "__other__", "__missing__"
SHARE_DIGITS = 4  # a bin's share to four decimals: the index's own floor is 1e-4
FLOOR = 1e-4  # a bin with no rows on one side still counts, as this share (the index is undefined at zero)


class ColumnBaseline(Contract):
    column: str = Field(min_length=1, max_length=200)
    kind: Literal["numeric", "category"]
    edges: list[float] = Field(default_factory=list, max_length=12, description="numeric: the inner edges of the bins, increasing.")
    categories: list[str] = Field(default_factory=list, max_length=TOP_CATEGORIES + 2, description="category: the bins' values, the rest as __other__.")
    shares: list[float] = Field(default_factory=list, max_length=TOP_CATEGORIES + 2, description="Each bin's share of the training rows (the missing values last, as their own bin).")
    rows: int = 0


class MonitoringPlan(Versioned):
    """The training rows as a model's monitor reads them, and what raises an alert."""

    schema_version: Literal["dclab_core.monitoring_plan.v1"] = _version("monitoring_plan")
    columns: list[ColumnBaseline] = Field(default_factory=list, max_length=MAX_COLUMNS)
    score: Optional[ColumnBaseline] = None
    flagged_share: Optional[float] = Field(default=None, ge=0, le=1, description="The share of training rows the operating rule flags.")
    rule: str = Field(default="", max_length=300, description="The operating rule the flagged share counts (a threshold, the top K).")
    threshold: Optional[float] = None
    psi_warn: float = PSI_WARN
    psi_alert: float = PSI_ALERT
    share_slack: float = Field(default=0.5, ge=0, description="The flagged share's band, widened by this share of the training share.")
    min_rows: int = MIN_ROWS
    left_out: list[str] = Field(default_factory=list, max_length=200, description="Columns not monitored, with why (personal, an identifier).")
    built_on: str = Field(default="", max_length=300)


CONTRACTS["monitoring_plan"] = MonitoringPlan


class ColumnReading(Contract):
    column: str
    psi: Optional[float] = None
    status: Literal["stable", "warn", "alert", "unread"]
    why: str = ""


class MonitorReport(Versioned):
    schema_version: Literal["dclab_core.monitor_report.v1"] = _version("monitor_report")
    rows: int
    columns: list[ColumnReading] = Field(default_factory=list)
    score: Optional[ColumnReading] = None
    flagged_share: Optional[float] = None
    flagged_band: Optional[tuple[float, float]] = None
    flagged_status: Literal["within", "above", "below", "unread"] = "unread"
    alerts: list[str] = Field(default_factory=list, description="What crossed an alert line, in words.")
    warnings: list[str] = Field(default_factory=list)
    status: Literal["stable", "warn", "alert", "too_few_rows"]


CONTRACTS["monitor_report"] = MonitorReport


def _numeric(values: Any) -> np.ndarray:
    import pandas as pd

    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def baseline(column: str, values: Sequence[Any], numeric: Optional[bool] = None) -> ColumnBaseline:
    """One column's bins and shares on the training rows."""
    import pandas as pd

    series = pd.Series(list(values) if not isinstance(values, pd.Series) else values)
    is_num = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) if numeric is None else numeric
    if is_num:
        x = _numeric(series)
        ok = x[np.isfinite(x)]
        edges = _midpoints(ok, np.quantile(ok, np.linspace(0.1, 0.9, 9))) if len(ok) else []
        return ColumnBaseline(column=column, kind="numeric", edges=edges, shares=_numeric_shares(x, edges), rows=len(x))
    text = series.astype("string").fillna(MISSING)
    top = [str(v) for v in text[text != MISSING].value_counts().index[:TOP_CATEGORIES]]
    return ColumnBaseline(column=column, kind="category", categories=top + [OTHER], shares=_category_shares(text, top), rows=len(text))


def _midpoints(values: np.ndarray, cuts: np.ndarray) -> list[float]:
    """Each cut moved to the midpoint between the two training values around it, so no edge is a row's own value (a
    decile of n rows is one of them when (n - 1)q is whole: the review of 30.15)."""
    u = np.unique(values)
    out = set()
    for cut in cuts:
        i = int(np.searchsorted(u, cut, side="right"))
        if 0 < i < len(u):
            low, high = float(u[i - 1]), float(u[i])
            mid = (low + high) / 2.0
            # the shortest number still between the two values: no long run of digits a reader could mistake for a value
            out.add(next((r for r in (round(mid, d) for d in range(0, 9)) if low < r < high), mid))
    return sorted(out)


def _numeric_shares(x: np.ndarray, edges: list[float]) -> list[float]:
    n = len(x)
    if n == 0:
        return []
    ok = np.isfinite(x)
    counts = np.bincount(np.searchsorted(np.asarray(edges), x[ok], side="right"), minlength=len(edges) + 1).astype(float)
    return [round(float(c / n), SHARE_DIGITS) for c in counts] + [round(float((~ok).sum() / n), SHARE_DIGITS)]


def _category_shares(text: Any, top: list[str]) -> list[float]:
    n = len(text)
    if n == 0:
        return []
    values = [str(v) for v in text]
    known = {c: i for i, c in enumerate(top)}
    counts = np.zeros(len(top) + 2)
    for v in values:
        counts[len(top) + 1 if v == MISSING else known.get(v, len(top))] += 1
    return [round(float(c / n), SHARE_DIGITS) for c in counts]


def psi(expected: Sequence[float], actual: Sequence[float]) -> float:
    """The population stability index of one distribution of bin shares against another."""
    e = np.maximum(np.asarray(expected, dtype=float), FLOOR)
    a = np.maximum(np.asarray(actual, dtype=float), FLOOR)
    return float(np.sum((a - e) * np.log(a / e)))


def shares_of(base: ColumnBaseline, values: Sequence[Any]) -> list[float]:
    import pandas as pd

    if base.kind == "numeric":
        return _numeric_shares(_numeric(values), base.edges)
    text = pd.Series(list(values)).astype("string").fillna(MISSING)
    return _category_shares(text, base.categories[:-1])


def plan(train: Any, *, scores: Optional[Sequence[float]] = None, threshold: Optional[float] = None, flagged_share: Optional[float] = None,
         rule: str = "", personal: Sequence[str] = (), skip: Mapping[str, str] | None = None, built_on: str = "") -> MonitoringPlan:
    """The plan from the training rows (a frame of the model's input columns), their out-of-fold scores, and the rule."""
    left = [f"{c}: personal, never monitored by value" for c in personal if c in getattr(train, "columns", [])]
    left += [f"{c}: {why}" for c, why in (skip or {}).items()]
    out = set(personal) | set(skip or {})
    cols = [baseline(str(c), train[c]) for c in list(train.columns) if c not in out][:MAX_COLUMNS]
    score = baseline("score", scores, numeric=True) if scores is not None and len(scores) else None
    if flagged_share is None and scores is not None and threshold is not None and len(scores):
        flagged_share = float(np.mean(np.asarray(scores, dtype=float) >= float(threshold)))
    return MonitoringPlan(columns=cols, score=score, flagged_share=flagged_share, threshold=threshold, rule=rule[:300], left_out=left[:200],
                          built_on=built_on[:300])


def _reading(base: ColumnBaseline, values: Sequence[Any], plan_: MonitoringPlan) -> ColumnReading:
    got = shares_of(base, values)
    if not got or not base.shares:
        return ColumnReading(column=base.column, status="unread", why="no values")
    index = psi(base.shares, got)
    status = "alert" if index >= plan_.psi_alert else "warn" if index >= plan_.psi_warn else "stable"
    return ColumnReading(column=base.column, psi=round(index, 6), status=status)


def _band(share: float, n: int, slack: float) -> tuple[float, float]:
    half = 1.96 * math.sqrt(max(share * (1 - share), 1e-12) / max(n, 1)) + slack * share
    return max(0.0, share - half), min(1.0, share + half)


def monitor(plan_: MonitoringPlan, new_rows: Any, new_scores: Optional[Sequence[float]] = None) -> MonitorReport:
    """New rows (a frame with the plan's columns) and, when given, their scores, read against the plan."""
    n = len(new_rows)
    readings, alerts, warnings = [], [], []
    for base in plan_.columns:
        if base.column not in getattr(new_rows, "columns", []):
            readings.append(ColumnReading(column=base.column, status="unread", why="the column is not in the new rows"))
            warnings.append(f"{base.column} is missing from the new rows")
            continue
        r = _reading(base, new_rows[base.column], plan_)
        readings.append(r)
        if n >= plan_.min_rows and r.status == "alert":
            alerts.append(f"{base.column} shifted (PSI {r.psi:.3f}, the alert line {plan_.psi_alert})")
        elif n >= plan_.min_rows and r.status == "warn":
            warnings.append(f"{base.column} moved (PSI {r.psi:.3f})")
    score = flagged = band = None
    flagged_status = "unread"
    if new_scores is not None and len(new_scores):
        if plan_.score is not None:
            score = _reading(plan_.score, new_scores, plan_)
            if n >= plan_.min_rows and score.status == "alert":
                alerts.append(f"the score's distribution shifted (PSI {score.psi:.3f})")
            elif n >= plan_.min_rows and score.status == "warn":
                warnings.append(f"the score's distribution moved (PSI {score.psi:.3f})")
        if plan_.threshold is not None and plan_.flagged_share is not None:
            flagged = float(np.mean(np.asarray(new_scores, dtype=float) >= float(plan_.threshold)))
            band = _band(plan_.flagged_share, len(new_scores), plan_.share_slack)
            flagged_status = "above" if flagged > band[1] else "below" if flagged < band[0] else "within"
            if n >= plan_.min_rows and flagged_status != "within":
                alerts.append(f"the rule flags {flagged:.1%} of the new rows, outside {band[0]:.1%} to {band[1]:.1%} (training: {plan_.flagged_share:.1%})")
    status = "too_few_rows" if n < plan_.min_rows else "alert" if alerts else "warn" if warnings else "stable"
    return MonitorReport(rows=n, columns=readings, score=score, flagged_share=flagged, flagged_band=band, flagged_status=flagged_status,
                         alerts=alerts, warnings=warnings, status=status)


__all__ = ["PSI_WARN", "PSI_ALERT", "ColumnBaseline", "MonitoringPlan", "ColumnReading", "MonitorReport", "baseline", "plan", "psi", "monitor"]
