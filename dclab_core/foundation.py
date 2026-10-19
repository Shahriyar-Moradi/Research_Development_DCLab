"""One data foundation (package 30.23, the owner's pattern 10): the same customer in four systems, one view of it, tables with owners and
checks, and history read as it was.

* **A catalog entry per table** (:func:`entry`): its owner, its keys, its date column, and quality checks run on it (the key unique and
  never empty, the dates readable and the newest one, the rows): a failed check is said on the entry.
* **One entity across systems** (:func:`resolve`): the systems' records in one table, matched by the matching kind (``dclab_core.matching``:
  the split by entity, blocking, the model chosen on validation pairs, the holdout read once) and linked over every record into entities
  (union-find of the links); the view says how many records and entities, and the pair F1 with its interval.
* **History as it was** (:func:`as_of`): a table read as of a date returns no row written after it, and for a history table (valid from,
  valid to) the row valid on that date, so a model built on the foundation never trains on the future.

Pure numpy and pandas; the host owns the systems and their files.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional, Sequence

import pandas as pd
from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version


class Check(Contract):
    name: str
    passed: bool
    detail: str = ""


class TableEntry(Versioned):
    """One table of the foundation (v1)."""

    schema_version: Literal["dclab_core.table_entry.v1"] = _version("table_entry")
    name: str = Field(max_length=120)
    system: str = Field(max_length=120)
    owner: str = Field(max_length=120)
    keys: list[str] = Field(default_factory=list)
    date_column: Optional[str] = None
    rows: int = 0
    checks: list[Check] = Field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return all(c.passed for c in self.checks)


CONTRACTS["table_entry"] = TableEntry


def entry(frame: pd.DataFrame, *, name: str, system: str, owner: str, keys: Sequence[str], date_column: Optional[str] = None) -> TableEntry:
    """A table's catalog entry with its checks run (names and counts, never a value)."""
    checks = []
    for k in keys:
        if k not in frame.columns:
            checks.append(Check(name=f"{k} exists", passed=False, detail="the key column is not in the table"))
            continue
        empty = int(frame[k].isna().sum())
        repeated = int(frame[k].dropna().duplicated().sum())
        checks.append(Check(name=f"{k} never empty", passed=empty == 0, detail=f"{empty} empty"))
        checks.append(Check(name=f"{k} unique", passed=repeated == 0, detail=f"{repeated} repeated"))
    if date_column:
        dates = pd.to_datetime(frame[date_column], errors="coerce") if date_column in frame.columns else pd.Series(dtype="datetime64[ns]")
        bad = int(dates.isna().sum()) if date_column in frame.columns else len(frame)
        newest = str(dates.max().date()) if dates.notna().any() else "none"
        checks.append(Check(name=f"{date_column} readable", passed=bad == 0, detail=f"{bad} unreadable; the newest {newest}"))
    checks.append(Check(name="rows", passed=len(frame) > 0, detail=f"{len(frame)} rows"))
    return TableEntry(name=name, system=system, owner=owner, keys=list(keys), date_column=date_column, rows=len(frame), checks=checks)


def as_of(frame: pd.DataFrame, when: str, *, date_column: Optional[str] = None, valid_from: Optional[str] = None, valid_to: Optional[str] = None) -> pd.DataFrame:
    """The table as it was on ``when``: rows written on or before it (``date_column``); for a history table, the rows valid on it (from
    <= when < to, an empty ``to`` still valid), and of those, with ``date_column`` too, only the ones already written then (a backdated
    correction written later is not what was known)."""
    at = pd.Timestamp(when)
    if not (valid_from or date_column):
        raise ValueError("say the date column, or the valid-from (and valid-to) columns")
    keep = pd.Series(True, index=frame.index)
    if valid_from:
        start = pd.to_datetime(frame[valid_from], errors="coerce")
        end = pd.to_datetime(frame[valid_to], errors="coerce") if valid_to else pd.Series(pd.NaT, index=frame.index)
        keep &= (start <= at) & (end.isna() | (end > at))
    if date_column:
        keep &= pd.to_datetime(frame[date_column], errors="coerce") <= at
    return frame[keep].copy()


def resolve(systems: Mapping[str, pd.DataFrame], *, id_column: str, fields: Sequence[Mapping[str, Any]], matches: pd.DataFrame,
            block_text: Optional[str] = None, block_equal: Sequence[str] = (), seed: int = 0) -> dict[str, Any]:
    """The systems' records as one view: ``systems`` name -> its table (each with ``id_column`` and the ``fields`` compared); ``matches``:
    the pairs known to be the same entity (two columns of ``system:id``). Returns each record's entity, the counts, and the matching's
    measurement (pair F1 on the holdout with its interval, the model chosen on validation)."""
    from dclab_core import matching

    frames = []
    for name, table in systems.items():
        t = table.copy()
        t["__record"] = name + ":" + t[id_column].astype(str)
        t["__system"] = name
        frames.append(t)
    merged = pd.concat(frames, ignore_index=True)
    ctx = matching.setup(merged, id_left="__record", fields=list(fields), matches=matches, block_text=block_text, block_equal=list(block_equal), seed=seed)
    picked = matching.validate(ctx)
    threshold = picked["scores"][picked["chosen"]].get("threshold")
    held = matching.score(ctx, picked["chosen"], threshold, picked["scores"]["single_field"])
    links = matching.link(ctx, picked["chosen"], threshold, picked["scores"]["single_field"])
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keys = list(ctx["left"]["__key"])
    for k in keys:
        find(k)
    for a, b, _ in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    record_of = dict(zip(ctx["left"]["__key"], merged["__record"]))
    entity = {record_of[k]: record_of[find(k)] for k in keys}
    return {"entity": entity, "records": len(keys), "entities": len(set(entity.values())), "links": len(links), "chosen": picked["chosen"],
            "pair_f1": held["pair_f1"], "pair_f1_ci95": held.get("pair_f1_ci95"), "beats": held.get("beats")}


__all__ = ["Check", "TableEntry", "entry", "as_of", "resolve"]
