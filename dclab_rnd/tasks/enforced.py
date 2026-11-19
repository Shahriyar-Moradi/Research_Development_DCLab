"""Did the engine obey the plan? (Hierarchical Planning, stage H0, 2026-11-16.)

A plan that catches a trap and a model that falls into it are the same failure with a nicer report. This module reads
what a plan caught and what the built project actually did, and says trap by trap whether the catch was enforced. Pure:
it runs nothing and reads no table.

A trap is enforceable when code can tell what would enforce it:

- a column known only after the outcome (``post_outcome_<column>``, a leak named with its column): the column is in the
  solution's ``forbidden`` list and is no feature of the final model;
- rows of one entity on both sides (``grouped_split_<column>``, ``repeated_entity_leak_<column>``): the split keeps
  that column's groups on one side;
- ``time_split_required`` (or a drift in time): the holdout is split by time.

Every other trap (a cost asymmetry, a missing horizon, an uplift design) is a thing to say to the person: it is counted
as "not enforceable by code" and never as enforced.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

FORBID = re.compile(r"^(?:post_outcome|leak(?:age)?|target_leak|label_leak|future_information)(?:_|$)")
GROUP = re.compile(r"^(?:grouped_split|repeated_entity_leak|entity_leak|group_leak)(?:_|$)")
TIME = re.compile(r"^(?:time_split_required|time_drift_random_split_leak|temporal_split_required|time_order_required)$")
KINDS = ("forbid", "group", "time_split")


def column_of(trap: str, columns: Sequence[str]) -> Optional[str]:
    """The table's column a trap names: the longest column name the trap's slug ends with."""
    slug = str(trap or "").lower()
    named = [c for c in columns if c and (slug == str(c).lower() or slug.endswith("_" + str(c).lower()))]
    return max(named, key=len) if named else None


def kind_of(trap: str) -> Optional[str]:
    slug = str(trap or "").lower()
    return "forbid" if FORBID.match(slug) else "group" if GROUP.match(slug) else "time_split" if TIME.match(slug) else None


def final_features(stages: Mapping[str, Any]) -> list[str]:
    """The raw columns behind the final model's features (``closure=1`` is the column ``closure``)."""
    results = ((stages.get("final") or {}).get("evidence") or {}).get("configuration_results") or []
    names = [str(f.get("feature") or "") for r in results for f in r.get("top_features") or []]
    return sorted({n.split("=", 1)[0] for n in names if n})


def forbidden_columns(solution: Mapping[str, Any]) -> set[str]:
    """The columns a solution keeps out (each entry is a column with its reason, or a bare name in an older record)."""
    return {str(f.get("column")) if isinstance(f, Mapping) else str(f) for f in solution.get("forbidden") or []}


def split_of(stages: Mapping[str, Any]) -> str:
    return str(((stages.get("data") or {}).get("evidence") or {}).get("holdout_split") or "").lower()


def obeyed(traps: Iterable[str], solution: Mapping[str, Any], stages: Mapping[str, Any], columns: Sequence[str]) -> list[dict[str, Any]]:
    """One record per trap the plan caught: its kind, the column it names, whether code can enforce it, and whether the
    built project did. ``enforced`` is None for a trap code cannot enforce, and for one whose stage has not run."""
    forbidden = forbidden_columns(solution)
    used, split = set(final_features(stages)), split_of(stages)
    out: list[dict[str, Any]] = []
    for trap in dict.fromkeys(str(t) for t in traps or []):
        kind, column = kind_of(trap), column_of(trap, columns)
        record: dict[str, Any] = {"trap": trap, "kind": kind, "column": column, "enforceable": False, "enforced": None, "how": ""}
        if column and column == solution.get("target"):
            pass  # the outcome is never a catch about itself: a target named `returned` is not a column to keep out
        elif kind == "forbid" and column:
            record.update(enforceable=True, enforced=(column in forbidden and column not in used) if "final" in stages else (column in forbidden),
                          how=("in forbidden" if column in forbidden else "not in forbidden") + ("; a feature of the final model" if column in used else ""))
        elif kind == "group" and column:
            grouped = solution.get("group_column") == column or ("group" in split and column.lower() in split)
            record.update(enforceable=True, enforced=bool(grouped) if split or solution.get("group_column") else False,
                          how=f"split: {split or 'not run'}; group column: {solution.get('group_column')}")
        elif kind == "time_split":
            # by time: the split says so, or it is ordered by the solution's own time column ("last 120 of 600 rows by
            # `signup_date`": a real Home build's split, read as not by time until 2026-11-16)
            by_time = any(w in split for w in ("time", "latest", "chronolog")) or bool(
                solution.get("time_column") and str(solution["time_column"]).lower() in split.lower())
            record.update(enforceable=True, enforced=("random" not in split and by_time) if split else None,
                          how=f"split: {split or 'not run'}")
        out.append(record)
    return out


MAX_SHARE = 3  # at most a third of the table's columns are kept out on the plan's word alone; past that the person is asked


def changes(traps: Iterable[str], calls: Iterable[Mapping[str, Any]], solution: Mapping[str, Any], columns: Sequence[str],
            time_candidates: Sequence[str] = (), can_group: Optional[Callable[[str, Optional[str]], bool]] = None) -> list[dict[str, Any]]:
    """What a plan's catches ask of the solution, before the first stage runs (stage H2): for each trap code can
    enforce, the one field to change, or the question to ask when code cannot tell what would enforce it.

    Every change makes the evaluation stricter, never looser: a column kept out, rows of one entity kept on one side, a
    split by time. So code applies it on the plan's word, says so, and the person can put it back. A change code cannot
    make by itself is a question (no column named, two candidates, a group already set to another column, too many
    columns at once, or a group the engine cannot split on beside the time column: ``can_group(group, time)``, given by
    the caller that reads the table). The outcome column and a column the table does not have are never touched.

    ``{"trap", "kind", "action": "apply" | "ask" | "none", "field", "value", "why"}``."""
    columns = [str(c) for c in columns]
    target = solution.get("target")
    kept_out = forbidden_columns(solution)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, Optional[str]]] = set()
    wanted: list[tuple[str, str, Optional[str]]] = [(str(t), kind_of(str(t)) or "", column_of(str(t), columns)) for t in dict.fromkeys(str(t) for t in traps or [])]
    for call in calls or []:  # a check the plan chose is its intention too: a split by time on a column, groups by an entity
        params = call.get("params") or {}
        if call.get("task_type") == "time_split_check" and params.get("column") in columns:
            wanted.append((f"time_split_check on {params['column']}", "time_split", str(params["column"])))
        if call.get("task_type") == "grouped_split" and params.get("entity") in columns:
            wanted.append((f"grouped_split on {params['entity']}", "group", str(params["entity"])))
    named_time = next((c for _, k, c in wanted if k == "time_split" and c), None)
    budget = max(1, len(columns) // MAX_SHARE)
    # a column the solution uses to split (its time or group column, or the time column this plan is about to set) is
    # never also kept out: the solution's own rules refuse it, and one refused change must not undo the others
    will_time = solution.get("time_column") or (named_time if any(k == "time_split" for _, k, _c in wanted) else None) or (
        time_candidates[0] if len(time_candidates) == 1 and any(k == "time_split" for _, k, _c in wanted) else None)
    splits_on = {c for c in (solution.get("time_column"), solution.get("group_column"), will_time) if c}
    for trap, kind, column in wanted:
        if kind not in KINDS:
            continue
        key = (kind, column if kind != "time_split" else None)
        if key in seen:
            continue
        seen.add(key)
        record: dict[str, Any] = {"trap": trap, "kind": kind, "action": "none", "field": None, "value": None, "why": ""}
        if kind == "forbid":
            if not column or column == target or column in splits_on:
                continue  # no column named, the outcome itself, or a column the split is made on: nothing code can keep out
            if column in kept_out:
                record.update(field="forbidden", value=column, why="already kept out")
            elif budget <= 0:
                record.update(action="ask", field="forbidden", value=column,
                              why=f"The plan suspects `{column}` is known only after the outcome, and it has already kept out as many columns as it may on its own word. Keep `{column}` out too?")
            else:
                budget -= 1
                record.update(action="apply", field="forbidden", value=column, why=f"the plan caught `{column}` as known only after the outcome ({trap})")
        elif kind == "group":
            current = solution.get("group_column")
            if not column:
                continue
            if current == column:
                record.update(field="group_column", value=column, why="already the group column")
            elif current:
                record.update(action="ask", field="group_column", value=column,
                              why=f"The plan asks to keep the rows of one `{column}` on one side of the split; the solution groups by `{current}`. Which one keeps related rows together?")
            elif can_group is not None and not can_group(column, will_time):
                # every entity in every period: the engine cannot split by time and keep entities apart, and the build would
                # fail with nothing built (the practice set, 2026-11-19). The split stays by time; the person chooses
                record.update(action="ask", field="group_column", value=column,
                              why=f"The plan asks to keep the rows of one `{column}` on one side of the split (the model must serve new ones), and "
                                  f"the split is by time on `{will_time}`. Every `{column}` appears in both periods, so both cannot hold: the split "
                                  f"stays by time. To test on unseen `{column}` values instead, set `{column}` as the group column and remove the time column.")
            else:
                record.update(action="apply", field="group_column", value=column, why=f"the plan caught rows of one `{column}` on both sides of the split ({trap})")
        else:
            current = solution.get("time_column")
            candidate = named_time or (time_candidates[0] if len(time_candidates) == 1 else None)
            if current:
                record.update(field="time_column", value=current, why="the split is already by time")
            elif candidate and candidate in columns:
                record.update(action="apply", field="time_column", value=str(candidate), why=f"the plan asks for a split by time ({trap})")
            else:
                record.update(action="ask", field="time_column",
                              why="The plan asks for a split by time, and the solution names no time column. Which column holds the time of each row?"
                                  + (f" Candidates: {', '.join(f'`{c}`' for c in time_candidates)}." if time_candidates else ""))
        out.append(record)
    return out


def summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts for one project: caught, enforceable by code, enforced, and the ones caught and not enforced."""
    able = [r for r in records if r.get("enforceable")]
    read = [r for r in able if r.get("enforced") is not None]
    return {"caught": len(records), "enforceable": len(able), "read": len(read), "enforced": sum(bool(r["enforced"]) for r in read),
            "broken": [r["trap"] for r in read if not r["enforced"]],
            "rate": round(sum(bool(r["enforced"]) for r in read) / len(read), 4) if read else None}
