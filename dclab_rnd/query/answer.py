"""An answer from a result: the chart chosen from its shape, the sentence, and the notebook that reproduces it (H7).

- ``chart(result, plan)``: a KPI tile for one number, a line for a period, bars for one group, a table otherwise.
- ``template(plan, result, notes)``: the answer written by code, every number taken from the result as it is stored.
- ``sentence(...)``: a model's one to three sentences (purpose ``home_answer``, only while
  ``DCLAB_MODEL_READS_AGGREGATES`` is on), kept only when every number in it is in the result (``cited.numbers``);
  otherwise the template stands.
- ``display(rows)``: the rows as the answer shows them (numbers rounded as written), kept on the answer so a check
  of its numbers (``cited.numbers``) finds each one where it came from.
- ``edited(plan, change)``: the answer's controls (split by, per period, the aggregate, the dates) applied in code.
- ``notebook(answers)``: an .ipynb with one cell per saved answer that reproduces it with pandas, the floor included.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from dclab_rnd import cited, prompts

TOP = 6  # groups named in a sentence
MODEL_ROWS = 20  # rows of a result a model may read for its sentence (home_answer)
MODEL_COLUMNS = 6  # and columns of each (group keys first)


def chart(result: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    cols = result["columns"]
    keys = [c for c in cols if c == "period" or c in (plan.get("group_by") or [])]
    values = [c for c in cols if c not in keys]
    if not keys:
        return {"kind": "kpi", "values": values}
    if keys == ["period"]:
        return {"kind": "line", "x": "period", "y": values[0]}
    if len(keys) == 1:
        return {"kind": "bar", "x": keys[0], "y": values[0]}
    return {"kind": "table"}


def shown(value: Any) -> Any:
    """A number rounded as the answer shows it: two decimals, three below one (a share)."""
    if isinstance(value, float) and not isinstance(value, bool):
        out = round(value, 2 if abs(value) >= 1 else 3)
        return int(out) if out == int(out) and abs(out) < 1e15 else out
    return value


def display(rows: list[list[Any]]) -> list[list[Any]]:
    return [[shown(v) for v in row] for row in rows]


def chart_kinds(answer: dict[str, Any]) -> list[str]:
    """The charts an answer's shape can take: a table always; a KPI tile for a single row; a line over periods; bars for groups."""
    natural = chart(answer["result"], answer["plan"])["kind"]
    return {"kpi": ["kpi", "table"], "line": ["line", "bar", "table"], "bar": ["bar", "table"], "table": ["table"]}[natural]


def chart_as(answer: dict[str, Any], kind: str) -> dict[str, Any]:
    base = chart(answer["result"], answer["plan"])
    return {**base, "kind": kind} if kind in chart_kinds(answer) else base


def fmt(value: Any) -> str:
    """A value as the answer writes it (after ``shown``)."""
    if value is None:
        return "not shown (a small group)"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def label(plan: dict[str, Any]) -> str:
    m = plan["measures"][0]
    what = {"count": "Number of rows", "count_distinct": f"Distinct {m.get('column')}", "sum": f"Total {m.get('column')}",
            "mean": f"Average {m.get('column')}", "median": f"Median {m.get('column')}", "min": f"Lowest {m.get('column')}",
            "max": f"Highest {m.get('column')}", "rate": "Share of rows"}.get(m["op"], f"{m['op']} of {m.get('column')}")
    parts = [what]
    if plan.get("group_by"):
        parts.append("by " + ", ".join(plan["group_by"]))
    period = plan.get("period") or {}
    if period.get("grain"):
        parts.append(f"per {period['grain']}")
    if period.get("from") or period.get("to"):
        parts.append(f"from {period.get('from') or 'the start'} to {period.get('to') or 'the end'}")
    for f in plan.get("filters") or []:
        v = f.get("value")
        parts.append(f"{f['column']} {'in' if isinstance(v, list) else 'is'} {', '.join(map(str, v)) if isinstance(v, list) else v}")
    return " · ".join(parts)


def template(plan: dict[str, Any], result: dict[str, Any], notes: Optional[list[str]] = None) -> str:
    """The answer written by code: what was computed, then the values as ``display`` shows them."""
    rows, cols = display(result["rows"]), result["columns"]
    head = label(plan)
    measures = [m["name"] for m in plan["measures"]]
    if not rows:
        text = f"{head}: no group has at least {result['floor']} rows, so there is nothing to show."
    elif len(cols) == len(measures):  # one number per measure
        text = f"{head}: " + "; ".join(f"{name} {fmt(v)}" for name, v in zip(cols, rows[0])) + "."
    else:
        keys = len(cols) - len(measures)

        def values(r):
            return fmt(r[keys]) if len(measures) == 1 else ", ".join(f"{n} {fmt(v)}" for n, v in zip(measures, r[keys:]))
        listed = [f"{' / '.join(str(x) for x in r[:keys])}: {values(r)}" for r in rows[:TOP]]
        more = "; the rest are in the table" if len(rows) > TOP else ""
        text = f"{head}. " + "; ".join(listed) + more + "."
    if result.get("suppressed_groups"):
        n = result["suppressed_groups"]
        text += f" {n} small group{'s were' if n != 1 else ' was'} hidden (fewer than {result['floor']} rows)."
    for note in notes or []:
        text += " " + note
    return text


def sentence(client: Any, question: str, plan: dict[str, Any], result: dict[str, Any]) -> Optional[str]:
    """A model's short answer, or None: off, failed, or a number that is not in the result."""
    from .execute import model_reads_aggregates

    if client is None or not model_reads_aggregates():
        return None
    state = {"question": str(question)[:600], "what_was_computed": label(plan), "columns": result["columns"][:MODEL_COLUMNS],
             "rows": [r[:MODEL_COLUMNS] for r in display(result["rows"])[:MODEL_ROWS]], "more_rows": len(result["rows"]) > MODEL_ROWS,
             "hidden_small_groups": result.get("suppressed_groups", 0), "floor": result.get("floor")}
    try:
        reply = client.complete([{"role": "system", "content": prompts.text("home_answer")},
                                 {"role": "user", "content": json.dumps(state, ensure_ascii=False, default=str)}], None)
    except Exception:  # noqa: BLE001 — the template answers
        return None
    text = str((reply or {}).get("content") if isinstance(reply, dict) else reply or "").strip()[:800]
    allowed = cited.numbers(json.dumps(state, default=str), source=True)
    ok = bool(text) and not (cited.numbers(text) - allowed)
    if getattr(client, "output", None):
        client.output(ok, "" if ok else "a number in the answer is not in the result")
    return text if ok else None


# ------------------------------------------------------------------------------------------- the controls
EDIT_OPS = ("count", "sum", "mean", "median")


def edited(plan: dict[str, Any], change: dict[str, Any]) -> dict[str, Any]:
    """A copy of the plan with the controls' change: ``split_by`` (a column, or "" for none), ``grain`` (a period, or
    "" for none, on the plan's date column), ``op`` (the first measure's aggregate), ``from``/``to`` (the dates). The
    validator checks the result as it checks any plan."""
    import copy

    out = copy.deepcopy(plan)
    if "split_by" in change:
        col = str(change.get("split_by") or "")
        out["group_by"] = [col] if col else []
    if "grain" in change:
        grain = str(change.get("grain") or "")
        period = out.get("period") or {}
        if grain:
            column = period.get("column") or change.get("date_column")
            if not column:
                raise ValueError("the plan has no date column to split by period")
            out["period"] = {**period, "column": column, "grain": grain}
        elif period:
            period.pop("grain", None)
            out["period"] = period if period.get("from") or period.get("to") else None
    if change.get("op"):
        if change["op"] not in EDIT_OPS:
            raise ValueError(f"the aggregate must be one of {', '.join(EDIT_OPS)}")
        m = dict(out["measures"][0])
        if change["op"] != "count" and not m.get("column"):
            raise ValueError("a count has no column to sum or average; pick count")
        m["op"] = change["op"]
        if change["op"] == "count":
            m.pop("column", None)
        out["measures"] = [m] + out["measures"][1:]
    for edge in ("from", "to"):
        if edge in change:
            period = dict(out.get("period") or {})
            if not period.get("column") and not change.get("date_column"):
                raise ValueError("the plan has no date column for a period")
            period.setdefault("column", change.get("date_column"))
            if change[edge]:
                period[edge] = str(change[edge])[:10]
            else:
                period.pop(edge, None)
            out["period"] = period
    timed = bool((out.get("period") or {}).get("grain"))
    keys = set(out.get("group_by") or []) | ({"period"} if timed else set())
    names = {m["name"] for m in out["measures"]}
    if {"split_by", "grain"} & set(change):  # a new shape: periods in time order, groups largest first
        out["sort"] = [{"by": "period", "desc": False}] if timed else [{"by": out["measures"][0]["name"], "desc": True}] if keys else []
    else:
        out["sort"] = [s for s in out.get("sort") or [] if s.get("by") in keys | names]
    if not out.get("period"):
        out.pop("period", None)
    return out


# ------------------------------------------------------------------------------------------- the notebook
_RUN = '''import json
import math

import pandas as pd

FLOOR = {floor}  # a group needs at least this many rows, and as many different values of each column below
PEOPLE = {people}  # the columns that name a person or a record (identifiers), as DCLab read the table
EXTREME_FACTOR = {extreme}  # a min, a max or a percentile outside p5..p95 is shown only for groups this many floors big
FREQ = {{"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}}


def naive(s):
    if pd.api.types.is_datetime64_any_dtype(s) and getattr(s.dt, "tz", None) is not None:
        return s.dt.tz_convert(None)
    return s


def mask(frame, f):
    s, op, v = naive(frame[f["column"]]), f["op"], f.get("value")
    if pd.api.types.is_datetime64_any_dtype(s) and op not in ("is_null", "not_null"):
        v = [pd.to_datetime(x) for x in v] if isinstance(v, list) else pd.to_datetime(v)
    return {{"eq": lambda: s == v, "ne": lambda: s != v, "in": lambda: s.isin(v), "not_in": lambda: ~s.isin(v), "gt": lambda: s > v,
             "ge": lambda: s >= v, "lt": lambda: s < v, "le": lambda: s <= v, "between": lambda: s.between(v[0], v[1]),
             "is_null": lambda: s.isna(), "not_null": lambda: s.notna()}}[op]()


def percentile(op):
    return int(op[1:]) if op.startswith("p") and op[1:].isdigit() else None


def run(plan, frame):
    \"\"\"The plan's answer: groups under the floor hidden, extremes of small groups withheld (None), as in DCLab.\"\"\"
    data = frame
    for f in plan.get("filters") or []:
        data = data[mask(data, f)]
    keys = list(plan.get("group_by") or [])
    p = plan.get("period")
    if p:
        t = naive(pd.to_datetime(data[p["column"]], errors="coerce"))
        keep = t.notna()
        if p.get("from") is not None:
            keep &= t >= pd.to_datetime(p["from"])
        if p.get("to") is not None:
            keep &= t <= pd.to_datetime(p["to"]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        data = data[keep]
        if p.get("grain"):
            data = data.assign(period=t[keep].dt.to_period(FREQ[p["grain"]]).astype(str))
            keys = ["period", *keys]
    extreme = {{m["name"] for m in plan["measures"] if m["op"] in ("min", "max")
               or (percentile(m["op"]) is not None and not 5 <= percentile(m["op"]) <= 95)}}

    def enough(part):
        return len(part) >= FLOOR and all(part[c].nunique(dropna=True) >= FLOOR for c in PEOPLE)

    def measures(part):
        out = {{}}
        for m in plan["measures"]:
            op = m["op"]
            if m["name"] in extreme and len(part) < EXTREME_FACTOR * FLOOR:
                out[m["name"]] = None
            elif op == "count":
                out[m["name"]] = len(part)
            elif op == "rate":
                out[m["name"]] = float(mask(part, m["where"]).mean()) if len(part) else None
            elif op == "count_distinct":
                out[m["name"]] = part[m["column"]].nunique(dropna=True)
            elif percentile(op) is not None:
                out[m["name"]] = part[m["column"]].quantile(percentile(op) / 100) if len(part[m["column"]].dropna()) else None
            else:
                out[m["name"]] = getattr(part[m["column"]], op)() if len(part[m["column"]].dropna()) else None
        return out

    if keys:
        rows = [{{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **measures(g)}}
                for k, g in data.groupby(keys, dropna=False, sort=True, observed=True) if enough(g)]
    else:
        rows = [measures(data)] if enough(data) else []

    def missing(v):
        return v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT

    for s in reversed(plan.get("sort") or []):
        present = [r for r in rows if not missing(r.get(s["by"]))]
        try:
            present.sort(key=lambda r: r.get(s["by"]), reverse=bool(s.get("desc")))
        except TypeError:
            present.sort(key=lambda r: str(r.get(s["by"])), reverse=bool(s.get("desc")))
        rows = present + [r for r in rows if missing(r.get(s["by"]))]
    columns = [*keys, *[m["name"] for m in plan["measures"]]]
    return pd.DataFrame([[r.get(c) for c in columns] for r in rows[: plan.get("limit", 100)]], columns=columns)


frame = pd.read_parquet("clean.parquet")  # the cleaned table DCLab answered from (download it from the draft's data)
'''


def notebook(answers: list[dict[str, Any]], floor: int, people: Optional[list[str]] = None, skipped: int = 0) -> dict[str, Any]:
    """One notebook that reproduces every saved answer from the cleaned table: the same plan, the same floor on rows
    and on the identifier columns (``people``), the same withheld extremes."""
    from .execute import EXTREME_FACTOR

    def cell(kind: str, text: str) -> dict[str, Any]:
        base = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        return base | ({"outputs": [], "execution_count": None} if kind == "code" else {})
    cells = [cell("markdown", "# Answers from DCLab Home\n\nEach cell runs one saved question's plan on the cleaned table, with the same "
                              "row floor: groups with fewer rows, or fewer different people, are hidden." + (f"\n\n{skipped} saved answer(s) were computed on another table and are not here." if skipped else "")), cell("code", _RUN.format(floor=int(floor), people=json.dumps(sorted(people or [])), extreme=int(EXTREME_FACTOR)))]
    for a in answers:
        cells.append(cell("markdown", f"## {a.get('title') or label(a['plan'])}"))
        cells.append(cell("code", f"plan = json.loads({json.dumps(json.dumps(a['plan'], default=str))})\nrun(plan, frame)"))
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}}
