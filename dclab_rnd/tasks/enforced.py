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
from typing import Any, Iterable, Mapping, Optional, Sequence

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


def split_of(stages: Mapping[str, Any]) -> str:
    return str(((stages.get("data") or {}).get("evidence") or {}).get("holdout_split") or "").lower()


def obeyed(traps: Iterable[str], solution: Mapping[str, Any], stages: Mapping[str, Any], columns: Sequence[str]) -> list[dict[str, Any]]:
    """One record per trap the plan caught: its kind, the column it names, whether code can enforce it, and whether the
    built project did. ``enforced`` is None for a trap code cannot enforce, and for one whose stage has not run."""
    forbidden = {str(c) for c in solution.get("forbidden") or []}
    used, split = set(final_features(stages)), split_of(stages)
    out: list[dict[str, Any]] = []
    for trap in dict.fromkeys(str(t) for t in traps or []):
        kind, column = kind_of(trap), column_of(trap, columns)
        record: dict[str, Any] = {"trap": trap, "kind": kind, "column": column, "enforceable": False, "enforced": None, "how": ""}
        if kind == "forbid" and column:
            record.update(enforceable=True, enforced=(column in forbidden and column not in used) if "final" in stages else (column in forbidden),
                          how=("in forbidden" if column in forbidden else "not in forbidden") + ("; a feature of the final model" if column in used else ""))
        elif kind == "group" and column:
            grouped = solution.get("group_column") == column or ("group" in split and column.lower() in split)
            record.update(enforceable=True, enforced=bool(grouped) if split or solution.get("group_column") else False,
                          how=f"split: {split or 'not run'}; group column: {solution.get('group_column')}")
        elif kind == "time_split":
            record.update(enforceable=True, enforced=("random" not in split and any(w in split for w in ("time", "latest", "chronolog"))) if split else None,
                          how=f"split: {split or 'not run'}")
        out.append(record)
    return out


def summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts for one project: caught, enforceable by code, enforced, and the ones caught and not enforced."""
    able = [r for r in records if r.get("enforceable")]
    read = [r for r in able if r.get("enforced") is not None]
    return {"caught": len(records), "enforceable": len(able), "read": len(read), "enforced": sum(bool(r["enforced"]) for r in read),
            "broken": [r["trap"] for r in read if not r["enforced"]],
            "rate": round(sum(bool(r["enforced"]) for r in read) / len(read), 4) if read else None}
