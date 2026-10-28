"""Why a number changed between two periods (package H11 of the Home plan).

``compare`` splits the change of a total (a sum of a column, or a count of rows) between period A and period B, segment by
segment of one grouping column, into three parts that add up exactly to that segment's change:

- volume: the number of rows changed (N_B - N_A), at A's mix and A's rate;
- mix: the segment's share of the rows changed (w_B - w_A), at B's volume and A's rate;
- rate: the value per row within the segment changed (r_B - r_A), at B's volume and B's mix.

For a segment s with N rows in all, a share w of them and a mean r per row, its total is N·w·r and
N_B·w_B·r_B − N_A·w_A·r_A = (N_B−N_A)·w_A·r_A + N_B·(w_B−w_A)·r_A + N_B·w_B·(r_B−r_A) exactly; the segments sum to the
total change. The order (volume, then mix, then rate) is a convention, said in the result. A segment under the row floor in
either period is pooled into one "other (small groups)" segment, so no group of a few rows is shown and the sum still holds.
``best_split`` picks, among the grouping columns offered, the one whose largest segment moving the same way as the total
carries the largest share of the change.
It shows association, never cause: the result says so.
"""

from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version
from dclab_core.query import FLOOR

OTHER = "other (small groups)"
ORDER = "volume, then mix, then rate"
CAVEAT = "This shows where the change happened, not why: the data shows association, not cause."


class Window(Contract):
    start: str = Field(max_length=40, description="ISO date, inclusive")
    end: str = Field(max_length=40, description="ISO date, inclusive (the whole day)")
    label: str = Field(default="", max_length=80)


class Segment(Contract):
    segment: str = Field(max_length=200)
    rows_a: int
    rows_b: int
    value_a: float
    value_b: float
    change: float
    volume: float
    mix: float
    rate: float


class ChangeResult(Versioned):
    schema_version: Literal["dclab_core.change_result.v1"] = _version("change_result")
    measure: Optional[str] = Field(default=None, max_length=200, description="the summed column; None for a count of rows")
    op: Literal["sum", "count"]
    date_column: str = Field(max_length=200)
    by: str = Field(max_length=200)
    a: Window
    b: Window
    total_a: float
    total_b: float
    change: float
    segments: list[Segment] = Field(default_factory=list, description="largest change first; the small groups pooled last")
    missing_measure: int = Field(default=0, description="rows of the two periods with no value for the measure, left out of the sums")
    floor: int = FLOOR
    order: str = ORDER
    caveat: str = CAVEAT


CONTRACTS["change_result"] = ChangeResult


def _rows(frame: pd.DataFrame, date_column: str, w: Window) -> pd.DataFrame:
    when = pd.to_datetime(frame[date_column], errors="coerce")
    if getattr(when.dt, "tz", None) is not None:
        when = when.dt.tz_localize(None)
    end = pd.Timestamp(w.end) + pd.Timedelta(days=1)
    return frame[(when >= pd.Timestamp(w.start)) & (when < end)]


def compare(frame: pd.DataFrame, *, date_column: str, a: Window, b: Window, by: str, measure: Optional[str] = None,
            floor: int = FLOOR) -> ChangeResult:
    """The change of the total from period A to period B, split per segment of ``by`` into volume, mix and rate."""
    for c in [date_column, by] + ([measure] if measure else []):
        if c not in frame.columns:
            raise ValueError(f"no column {c!r}")
    if measure and not pd.api.types.is_numeric_dtype(frame[measure]):
        raise ValueError(f"{measure!r} is not a number column")
    ra, rb = _rows(frame, date_column, a), _rows(frame, date_column, b)
    missing = 0
    if measure:  # a row with no value is left out of the sum and counted, never a row worth 0 (it would read as a rate change)
        missing = int(ra[measure].isna().sum() + rb[measure].isna().sum())
        ra, rb = ra[ra[measure].notna()], rb[rb[measure].notna()]
    if ra.empty or rb.empty:
        raise ValueError("a period has no rows")

    def stats(rows: pd.DataFrame) -> pd.DataFrame:
        key = rows[by].astype(str).where(rows[by].notna(), "(no value)")
        value = rows[measure].fillna(0.0) if measure else pd.Series(1.0, index=rows.index)
        return pd.DataFrame({"n": key.groupby(key).size(), "total": value.groupby(key).sum()})

    sa, sb = stats(ra), stats(rb)
    keys = sorted(set(sa.index) | set(sb.index))
    sa, sb = sa.reindex(keys, fill_value=0), sb.reindex(keys, fill_value=0)
    small = [k for k in keys if 0 < sa.loc[k, "n"] < floor or 0 < sb.loc[k, "n"] < floor]
    if small:  # pooled: no group of a few rows is ever shown, and the segments still sum to the total
        def thin(group: list[str]) -> bool:  # the pool itself under the floor in a period that has it would show its rows
            n_a, n_b = sa.loc[group, "n"].sum(), sb.loc[group, "n"].sum()
            return 0 < n_a < floor or 0 < n_b < floor
        rest = sorted((k for k in keys if k not in small), key=lambda k: sa.loc[k, "n"] + sb.loc[k, "n"])
        while thin(small) and rest:  # the smallest named groups join the pool until it holds the floor in both periods
            small.append(rest.pop(0))
        if thin(small):
            raise ValueError("too few rows in the two periods to show any group")
        sa.loc[OTHER] = sa.loc[small].sum()
        sb.loc[OTHER] = sb.loc[small].sum()
        sa, sb = sa.drop(index=small), sb.drop(index=small)
    na, nb = float(sa["n"].sum()), float(sb["n"].sum())
    segments = []
    for k in sa.index:
        n_a, n_b = float(sa.loc[k, "n"]), float(sb.loc[k, "n"])
        t_a, t_b = float(sa.loc[k, "total"]), float(sb.loc[k, "total"])
        w_a, w_b = n_a / na, n_b / nb
        r_a = t_a / n_a if n_a else (t_b / n_b if n_b else 0.0)  # a new segment: its own rate, all of its change is volume and mix
        r_b = t_b / n_b if n_b else r_a
        volume = (nb - na) * w_a * r_a
        mix = nb * (w_b - w_a) * r_a
        rate = nb * w_b * (r_b - r_a)
        segments.append(Segment(segment=str(k), rows_a=int(n_a), rows_b=int(n_b), value_a=t_a, value_b=t_b, change=t_b - t_a,
                                volume=volume, mix=mix, rate=rate))
    shown = sorted([s for s in segments if s.segment != OTHER], key=lambda s: -abs(s.change)) + [s for s in segments if s.segment == OTHER]
    total_a, total_b = float(sa["total"].sum()), float(sb["total"].sum())
    return ChangeResult(measure=measure, op="sum" if measure else "count", date_column=date_column, by=by, a=a, b=b,
                        total_a=total_a, total_b=total_b, change=total_b - total_a, segments=shown, floor=floor, missing_measure=missing)


def split_scores(frame: pd.DataFrame, columns: list[str], **kw) -> dict[str, float]:
    """Each grouping column offered and the share of the change its largest same-direction segment carries (None-free: a
    column that cannot be compared is left out), so the choice is kept and can be checked."""
    out = {}
    for c in columns:
        try:
            r = compare(frame, by=c, **kw)
        except ValueError:
            continue
        out[c] = max((s.change / r.change for s in r.segments if s.segment != OTHER), default=0.0) if r.change else 0.0
    return out


def best_split(frame: pd.DataFrame, columns: list[str], **kw) -> Optional[ChangeResult]:
    """Among the grouping columns offered, the comparison whose largest segment moving the same way as the total carries the
    largest share of the change (ties to the column offered first); None when no column gives one."""
    best, score = None, -1.0
    for c in columns:
        try:
            r = compare(frame, by=c, **kw)
        except ValueError:
            continue
        # the share of the change its largest segment moving the same way carries (a segment moving against it explains none)
        share = max((s.change / r.change for s in r.segments if s.segment != OTHER), default=0.0) if r.change else 0.0
        if share > score + 1e-12:
            best, score = r, share
    return best


__all__ = ["Window", "Segment", "ChangeResult", "compare", "best_split", "split_scores", "OTHER", "ORDER", "CAVEAT"]
