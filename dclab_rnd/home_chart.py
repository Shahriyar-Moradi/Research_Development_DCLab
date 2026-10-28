"""The chart for an answer from a person's table (package H7, part b): chosen by code from the result's shape, never by a
model, and drawn from the result's own numbers only.

- one number (no group): a KPI tile;
- a period (a time axis) of at least two points, with or without one group of at most ``MAX_LINES`` values: a line per
  group over the periods, in time order;
- one group of at most ``MAX_BARS`` values: bars, largest first;
- anything else (two or three groups, many values): a table.

The spec is plain data (kind, x, series, values, and each number as the answer writes it), so the page draws it and an
export can redraw it; nothing in it is not in the result. Its title is the plan's own description (operation, column,
filters, period), never a measure's name, which a model may have written.
"""

from __future__ import annotations

from typing import Any, Optional

MAX_BARS = 12
MAX_LINES = 4  # the page's four series styles


def spec(plan: Any, result: Any) -> dict[str, Any]:
    """The chart for one answer: {"kind": kpi|line|bar|table, "title", ...} from the plan's shape and the result's rows."""
    from .home_query import OP_WORDS, describe, key_text, shown

    measures = [m.name for m in plan.measures]
    keys = result.columns[:len(result.columns) - len(measures)]
    rows = result.rows
    heads = [{"count": "rows", "count_distinct": f"distinct {m.column}", "rate": "share"}.get(m.op, f"{OP_WORDS.get(m.op, m.op)} of {m.column}")
             for m in plan.measures]  # headers from the plan, never a measure's name
    numbers = {"columns": [*keys, *heads],  # every chart carries the numbers' table, each as the answer writes it
               "rows": [[key_text(v) for v in r[:len(keys)]] + [shown(v) for v in r[len(keys):]] for r in rows]}
    base = {"title": describe(plan), "suppressed_groups": result.suppressed_groups, "table": numbers}
    table = {**base, "kind": "table"}
    if not rows:
        return table
    if not keys:
        value = rows[0][-len(measures)]
        return {**base, "kind": "kpi", "value": value, "text": shown(value)}
    at = len(keys)
    period: Optional[str] = plan.period.column if getattr(plan, "period", None) else None
    if period and period in keys and len(keys) <= 2:
        x_index = keys.index(period)
        group_index = next((i for i, k in enumerate(keys) if k != period), None)
        xs = _every_period(sorted({str(r[x_index]) for r in rows}), plan.period.grain)
        names = {key_text(r[group_index]) for r in rows} if group_index is not None else {"all"}
        if len(xs) >= 2 and len(names) <= MAX_LINES:
            series: dict[str, list[Any]] = {}
            for r in rows:
                name = key_text(r[group_index]) if group_index is not None else "all"
                series.setdefault(name, [None] * len(xs))[xs.index(str(r[x_index]))] = r[at]
            return {**base, "kind": "line", "x": xs, "series": [{"name": n, "values": v} for n, v in sorted(series.items())]}
    if len(keys) == 1 and len(rows) <= MAX_BARS:
        ordered = sorted(rows, key=lambda r: (r[at] is None, -(r[at] or 0)))
        return {**base, "kind": "bar", "x": [key_text(r[0]) for r in ordered], "values": [r[at] for r in ordered],
                "texts": [shown(r[at]) for r in ordered]}
    return table


FREQ = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}


def _every_period(present: list[str], grain: str) -> list[str]:
    """Every period from the first to the last, so a period the floor removed is a gap on the axis, never two periods
    joined as if they were next to each other (the review of H7b). Periods that do not parse stay as they are."""
    import pandas as pd

    try:
        freq = FREQ[grain]
        first, last = pd.Period(present[0], freq=freq), pd.Period(present[-1], freq=freq)
        full = [str(p) for p in pd.period_range(first, last, freq=freq)]
    except Exception:  # noqa: BLE001 - an unknown grain or label: the present periods only
        return present
    return full if set(present) <= set(full) and len(full) <= 400 else present


__all__ = ["spec", "MAX_BARS", "MAX_LINES"]
