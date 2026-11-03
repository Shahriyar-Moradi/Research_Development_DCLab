"""The data tools behind the guard layer (package 37.6, docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): what an agent may ask of
a person's table, each run and checked by code, never by the model.

* :func:`query`: an aggregate question (``dclab_core.query``): counts, rates, means by group, groups under the row floor
  hidden; never a row. A column the table lacks is refused with its name, so the agent can say it is missing.
* :func:`leakage`: which columns look written after the outcome (``dclab_core.analyse``'s leakage findings), with the
  evidence in words; a suspicion is tested, never trusted.
* :func:`preview_step`: what a cleaning step would do (``dclab_core.recipe.preview``): rows before and after, and whether
  it removes enough to need the person's yes. Nothing is applied here.

Each returns a JSON-ready result or ``{"error": ...}`` with the reason, so a refusal reaches the agent as a tool result.
The lead agent (``case_agent``) and the exam's Claude-alone arm call the same functions: the same data tools for both.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

import pandas as pd

MAX_GROUPS = 20  # groups one answer may hold: group names are the only cell values an agent sees
MAX_GROUP_BY = 2  # columns one answer may group by (the review of 37.6: three gave up to 60 names)
MAX_NAMES = 40  # distinct group names one case may see across its answers: the case_agent purpose's cell_values
NO_SPLIT = "computed on every row of the table: no holdout exists yet, so nothing here has looked at one"


def _floor() -> int:
    from dclab_rnd.home_query import floor

    return floor()


def plan_of(frame: pd.DataFrame, group_by: Optional[list[str]] = None, measures: Optional[list[Mapping[str, Any]]] = None,
            filters: Optional[list[Mapping[str, Any]]] = None, sort: Optional[list[Mapping[str, Any]]] = None,
            personal: Optional[set[str]] = None) -> tuple[Any, list[str], dict[str, Any]]:
    """The checked plan an agent's arguments mean, and what was made right in them (``Q.repair``); or ``(None, [], {"error": ...})``
    with the reasons, and how to write it. One reading for a query, an answer card and a dashboard tile."""
    from dclab_core import query as Q

    if len(group_by or []) > MAX_GROUP_BY:
        return None, [], {"error": f"group by at most {MAX_GROUP_BY} columns at a time"}
    try:
        plan = Q.QueryPlan(group_by=list(group_by or []), measures=[Q.Measure(**m) for m in measures or [{"name": "rows", "op": "count"}]],
                           filters=[Q.Filter(**f) for f in filters or []], sort=[Q.Sort(**s) for s in sort or []], limit=500)  # dclab_core.query's most
    except (TypeError, ValueError) as error:
        return None, [], {"error": f"the question could not be read: {error}"}
    plan, counted = Q.repair(plan, frame)  # a plain misreading is made right and said, never silently
    problems = Q.validate(plan, frame)
    if problems:
        return None, [], {"error": "; ".join(problems), "columns": list(map(str, frame.columns))}
    used = {*plan.group_by, *(f.column for f in plan.filters), *(m.column for m in plan.measures if m.column),
            *(m.where.column for m in plan.measures if m.where), *([plan.period.column] if plan.period else [])}
    if personal and used & personal:  # the studio marks these columns personal: no answer, chart or tile uses them (the review of 37.11j)
        return None, [], {"error": f"{', '.join(sorted(used & personal))} holds personal data: DCLab does not answer with it. Use another column."}
    return plan, counted, {}


def query(frame: pd.DataFrame, group_by: Optional[list[str]] = None, measures: Optional[list[Mapping[str, Any]]] = None,
          filters: Optional[list[Mapping[str, Any]]] = None, sort: Optional[list[Mapping[str, Any]]] = None,
          seen: Optional[set[str]] = None, keep: Optional[list[Any]] = None, personal: Optional[set[str]] = None) -> dict[str, Any]:
    """``seen``: the group names the case has already been shown; an answer that would take it past MAX_NAMES is refused.
    ``keep``: a list the checked plan is appended to when the query ran (the lead agent's answer card is drawn from it)."""
    from dclab_core import query as Q

    plan, counted, refused = plan_of(frame, group_by, measures, filters, sort, personal)
    if plan is None:
        return refused
    try:
        result = Q.execute(plan, frame, floor=_floor())
    except (ValueError, TypeError) as error:  # the review: a plan valid on paper that cannot run is a refusal, never a crash
        return {"error": f"the question could not run: {error}"[:300]}
    rows = result.rows[:MAX_GROUPS]
    names = {str(v) for row in rows for v in row[:len(plan.group_by)]}
    if seen is not None and len(seen | names) > MAX_NAMES:
        return {"error": f"this case has seen {len(seen)} group names; this answer would pass {MAX_NAMES}: narrow it with a filter"}
    if seen is not None:
        seen |= names
    if keep is not None:
        keep.append(plan)
    out = {"columns": result.columns, "rows": rows, "as_said": _as_said(plan, rows), "groups_not_shown": max(0, len(result.rows) - MAX_GROUPS),
           "groups_hidden_under_the_floor": result.suppressed_groups, "rows_scanned": result.rows_scanned, "note": NO_SPLIT,
           "exposure": Q.exposure(plan)}
    if counted:
        out["counted"] = counted
    return out


def _as_said(plan: Any, rows: list[list[Any]]) -> list[list[str]]:
    """Each row as it may be said to the person: a share as a percentage with one decimal (26.5%), a count with its thousands
    separated. These strings are what the guard compares a sentence with, so a correct sentence is never dropped for rounding."""
    keys = len(plan.group_by)
    ops = [m.op for m in plan.measures]
    out = []
    for row in rows:
        said = [str(v) for v in row[:keys]]
        for op, v in zip(ops, row[keys:]):
            said.append("withheld" if v is None else f"{v * 100:.1f}%" if op == "rate" else f"{v:,}" if isinstance(v, int)
                        else f"{v:,.2f}".rstrip("0").rstrip("."))
        out.append(said)
    return out


def leakage(frame: pd.DataFrame, outcome: str, moment: Optional[str] = None) -> dict[str, Any]:
    from dclab_core.analyse import analyse

    if outcome not in frame.columns:
        return {"error": f"the table has no column {outcome!r}", "columns": list(map(str, frame.columns))}
    try:
        found = analyse({"table": frame}, target=outcome)
    except Exception as error:  # noqa: BLE001 - an analysis that cannot run is said, never guessed
        return {"error": f"the leakage check could not run: {type(error).__name__}"}
    leaks = [{"statement": f.statement, "implication": f.implication, "severity": f.severity}
             for f in found.findings if f.topic == "leakage"]
    return {"outcome": outcome, "suspects": leaks, "checked_columns": len(frame.columns) - 1,
            "exposure": {"columns_related": [[str(c), outcome] for c in frame.columns if c != outcome], "rows": "all"}, "note": NO_SPLIT
            + ("; the moment you named was not used: these are tests of the columns themselves" if moment else "")}


def preview_step(frame: pd.DataFrame, kind: str, columns: Optional[list[str]] = None, filters: Optional[list[Mapping[str, Any]]] = None,
                 keys: Optional[list[str]] = None) -> dict[str, Any]:
    from dclab_core import recipe as R
    from dclab_core.query import Filter

    try:
        step = R.Step(kind=kind, columns=list(columns or []), filters=[Filter(**f) for f in filters or []], keys=list(keys or []))
    except (TypeError, ValueError) as error:
        return {"error": f"the step could not be read: {error}"}
    missing = [c for c in [*step.columns, *step.keys] if c not in frame.columns]
    if missing:
        return {"error": f"the table has no column {', '.join(missing)}", "columns": list(map(str, frame.columns))}
    if step.filters:  # the review: a filter's counts are a query: the same rules on identifiers apply
        from dclab_core import query as Q

        problems = Q.validate(Q.QueryPlan(filters=step.filters, measures=[Q.Measure(name="rows", op="count")]), frame)
        if problems:
            return {"error": "; ".join(problems)}
    try:
        preview, _ = R.preview(step, frame)
    except Exception as error:  # noqa: BLE001
        return {"error": f"the step could not be previewed: {type(error).__name__}: {error}"[:300]}
    out = json.loads(preview.model_dump_json())
    if step.filters and out.get("rows_after", 0) < _floor():  # a count this small could name a person: said as "fewer than"
        out["rows_after"] = f"fewer than {_floor()}"
    out["applied"] = False
    if preview.needs_confirmation:
        out["next"] = "ask the person before anything runs, with these counts in plain words"
    return out


TOOLS = {"query": query, "leakage": leakage, "preview_step": preview_step}
SCHEMAS = {
    "query": {"type": "object", "properties": {
        "group_by": {"type": "array", "items": {"type": "string"}},
        "measures": {"type": "array", "items": {"type": "object", "required": ["name", "op"], "properties": {
            "name": {"type": "string"}, "op": {"type": "string", "enum": ["count", "count_distinct", "sum", "mean", "median", "min", "max", "rate", "percentile"]},
            "column": {"type": "string"}, "q": {"type": "number"},
            "where": {"type": "object", "properties": {"column": {"type": "string"}, "op": {"type": "string"}, "value": {}}}}}},
        "filters": {"type": "array", "items": {"type": "object", "required": ["column", "op"], "properties": {
            "column": {"type": "string"}, "op": {"type": "string", "enum": ["eq", "ne", "in", "not_in", "gt", "ge", "lt", "le", "between", "is_null", "not_null"]},
            "value": {}}}},
        "sort": {"type": "array", "items": {"type": "object", "required": ["by"], "properties": {"by": {"type": "string"}, "desc": {"type": "boolean"}}}}}},
    "leakage": {"type": "object", "required": ["outcome"], "properties": {"outcome": {"type": "string"}}},
    "preview_step": {"type": "object", "required": ["kind"], "properties": {
        "kind": {"type": "string", "enum": ["drop_columns", "filter", "dedupe", "to_date", "drop_missing", "sort"]},
        "columns": {"type": "array", "items": {"type": "string"}}, "keys": {"type": "array", "items": {"type": "string"}},
        "filters": {"type": "array", "items": {"type": "object", "properties": {"column": {"type": "string"}, "op": {"type": "string"}, "value": {}}}}}},
}
DESCRIPTIONS = {
    "query": "Ask the shared table an aggregate question: counts, rates, means, sums by group. DCLab's code computes it; only "
             "aggregates come back, never a row, and small groups are hidden. A share is {name, op:'rate', where:{column, op:'eq', "
             "value}} (the condition is the where; the measure has no column); the rows meeting a condition are {name, op:'count', "
             "where:{...}}. Say a result's numbers exactly as 'as_said' gives them.",
    "leakage": "Check which columns look written after the outcome (they would leak the answer). DCLab's code tests them.",
    "preview_step": "Preview what a cleaning step would do to the table: rows before and after, and whether it needs the person's "
                    "yes. Nothing is applied.",
}

__all__ = ["DESCRIPTIONS", "MAX_GROUPS", "SCHEMAS", "TOOLS", "leakage", "preview_step", "query"]
