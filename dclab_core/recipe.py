"""Preparing a table on request, step by step (package H9 of the Home plan).

A ``Recipe`` is a list of typed ``Step``s, each applied to a copy: the original table is never changed. Every step has a
``preview`` (rows and columns before and after, the share of rows it removes, what it added or dropped) and the guard's
verdict: a step that removes more than ``GUARD_SHARE`` of the rows, or a join that changes the row count, needs the person's
yes before it is applied. Undo is the recipe without its last step, run again from the original.

The steps: ``rename``, ``drop_columns``, ``filter`` (the query engine's filters, all of them must hold), ``dedupe`` (on keys,
or on every column), ``to_date``, ``drop_missing``, ``sort``, ``months_since`` (whole months from a date column to a stated
day, the column's latest by default), ``join`` (another named table, on keys; left by default), ``pivot`` (one row per
index value, one column per value of another, aggregated). Nothing here reads a model; a step a model proposes is checked
here like any other.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import Field, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version
from dclab_core.query import Filter, _mask

GUARD_SHARE = 0.05  # a step removing more than this share of the rows asks first (H1's trap 3: "keep only ... give me that table")
StepKind = Literal["rename", "drop_columns", "filter", "dedupe", "to_date", "drop_missing", "sort", "months_since", "join", "pivot"]
Agg = Literal["sum", "mean", "count", "min", "max", "median"]


class Step(Contract):
    kind: StepKind
    columns: list[str] = Field(default_factory=list, max_length=200)
    to: Optional[str] = Field(default=None, max_length=200, description="rename: the new name; months_since: the new column")
    filters: list[Filter] = Field(default_factory=list, max_length=20)
    keys: list[str] = Field(default_factory=list, max_length=10, description="dedupe: the columns that make a row the same; join: the keys")
    table: Optional[str] = Field(default=None, max_length=200, description="join: the other table's name")
    how: Literal["left", "inner"] = "left"
    index: Optional[str] = Field(default=None, max_length=200)
    spread: Optional[str] = Field(default=None, max_length=200, description="pivot: the column whose values become columns")
    values: Optional[str] = Field(default=None, max_length=200)
    agg: Agg = "sum"
    at: Optional[str] = Field(default=None, max_length=40, description="months_since: the day counted to (ISO); the column's latest by default")
    desc: bool = False
    why: str = Field(default="", max_length=400, description="the person's words this step comes from, or why it was proposed")

    @model_validator(mode="after")
    def _complete(self) -> "Step":
        need = {"rename": self.columns and self.to, "drop_columns": self.columns, "filter": self.filters, "to_date": self.columns,
                "drop_missing": self.columns, "sort": self.columns, "months_since": self.columns and self.to,
                "join": self.table and self.keys, "pivot": self.index and self.spread and self.values, "dedupe": True}
        if not need[self.kind]:
            raise ValueError(f"a {self.kind} step is missing what it acts on")
        if self.kind in ("rename", "months_since") and len(self.columns) != 1:
            raise ValueError(f"a {self.kind} step acts on one column")
        return self


class StepPreview(Contract):
    kind: StepKind
    rows_before: int
    rows_after: int
    removed_share: float = Field(ge=0, le=1)
    columns_before: int
    columns_after: int
    added: list[str] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)
    became_missing: int = Field(default=0, description="to_date: values that could not be read as dates")
    needs_confirmation: bool = False
    reason: Optional[str] = None


class Recipe(Versioned):
    schema_version: Literal["dclab_core.recipe.v1"] = _version("recipe")
    source_sha256: Optional[str] = Field(default=None, max_length=64, description="the original table the recipe starts from")
    steps: list[Step] = Field(default_factory=list, max_length=100)


CONTRACTS["recipe"] = Recipe


def _known(frame: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what}: no column {', '.join(repr(c) for c in missing)}")


def run(step: Step, frame: pd.DataFrame, tables: Optional[dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """One step on a copy of ``frame`` (ValueError when it cannot apply: an unknown column or table, a bad date)."""
    t = frame.copy()
    k = step.kind
    if k == "rename":
        _known(t, step.columns, "rename")
        if step.to in t.columns:
            raise ValueError(f"rename: a column {step.to!r} already exists")
        return t.rename(columns={step.columns[0]: step.to})
    if k == "drop_columns":
        _known(t, step.columns, "drop_columns")
        return t.drop(columns=step.columns)
    if k == "filter":
        _known(t, [f.column for f in step.filters], "filter")
        keep = pd.Series(True, index=t.index)
        for f in step.filters:
            try:
                keep &= _mask(t, f).fillna(False).astype(bool)
            except (TypeError, ValueError) as error:
                raise ValueError(f"filter on {f.column!r}: {error}") from None
        return t[keep]
    if k == "dedupe":
        _known(t, step.keys, "dedupe")
        return t.drop_duplicates(subset=step.keys or None, keep="first")
    if k == "to_date":
        _known(t, step.columns, "to_date")
        for c in step.columns:
            t[c] = pd.to_datetime(t[c], errors="coerce")
        return t
    if k == "drop_missing":
        _known(t, step.columns, "drop_missing")
        return t.dropna(subset=step.columns)
    if k == "sort":
        _known(t, step.columns, "sort")
        return t.sort_values(step.columns, ascending=not step.desc, kind="stable")
    if k == "months_since":
        _known(t, step.columns, "months_since")
        if step.to in t.columns:
            raise ValueError(f"months_since: a column {step.to!r} already exists")
        when = pd.to_datetime(t[step.columns[0]], errors="coerce")
        at = pd.Timestamp(step.at) if step.at else when.max()
        if pd.isna(at):
            raise ValueError("months_since: no date to count to")
        months = (at.year - when.dt.year) * 12 + (at.month - when.dt.month) - (at.day < when.dt.day).astype("Int64")
        t[step.to] = months.astype("Int64")
        return t
    if k == "join":
        other = (tables or {}).get(str(step.table))
        if other is None:
            raise ValueError(f"join: no table {step.table!r}")
        _known(t, step.keys, "join (this table)")
        _known(other, step.keys, f"join ({step.table})")
        bring = [c for c in (step.columns or list(other.columns)) if c not in step.keys]
        _known(other, bring, f"join ({step.table})")
        clash = [c for c in bring if c in t.columns]
        return t.merge(other[[*step.keys, *bring]], on=step.keys, how=step.how, suffixes=("", f"_{step.table}") if clash else ("", ""))
    if k == "pivot":
        _known(t, [str(step.index), str(step.spread), str(step.values)], "pivot")
        out = t.pivot_table(index=step.index, columns=step.spread, values=step.values, aggfunc=step.agg, observed=True)
        out.columns = [str(c) for c in out.columns]
        return out.reset_index()
    raise ValueError(f"unknown step {k}")


def preview(step: Step, frame: pd.DataFrame, tables: Optional[dict[str, pd.DataFrame]] = None) -> tuple[StepPreview, pd.DataFrame]:
    """What a step would do, with the guard's verdict, and the table it makes (on a copy)."""
    after = run(step, frame, tables)
    removed = max(0, len(frame) - len(after)) if step.kind in ("filter", "dedupe", "drop_missing") else 0
    share = removed / len(frame) if len(frame) else 0.0
    missing = 0
    if step.kind == "to_date":
        missing = int(sum(int(after[c].isna().sum() - frame[c].isna().sum()) for c in step.columns))
    reason = None
    if share > GUARD_SHARE:
        reason = f"it removes {removed:,} of {len(frame):,} rows ({share:.0%})"
    elif step.kind == "join" and len(after) != len(frame):
        reason = f"the join changes the row count from {len(frame):,} to {len(after):,}"
    elif missing and missing / max(len(frame), 1) > GUARD_SHARE:
        reason = f"{missing:,} values could not be read as dates and become empty"
    p = StepPreview(kind=step.kind, rows_before=len(frame), rows_after=len(after), removed_share=round(share, 6),
                    columns_before=frame.shape[1], columns_after=after.shape[1],
                    added=[str(c) for c in after.columns if c not in frame.columns], dropped=[str(c) for c in frame.columns if c not in after.columns],
                    became_missing=missing, needs_confirmation=reason is not None, reason=reason)
    return p, after


def apply(recipe: Recipe, frame: pd.DataFrame, tables: Optional[dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """Every step in order, from the original (which is never changed)."""
    out = frame
    for step in recipe.steps:
        out = run(step, out, tables)
    return out


def undo(recipe: Recipe) -> Recipe:
    """The recipe without its last step."""
    return recipe.model_copy(update={"steps": recipe.steps[:-1]})


def recipe_hash(recipe: Recipe) -> str:
    return hashlib.sha256(json.dumps([s.model_dump(mode="json") for s in recipe.steps], sort_keys=True).encode()).hexdigest()[:16]


__all__ = ["GUARD_SHARE", "Step", "StepPreview", "Recipe", "run", "preview", "apply", "undo", "recipe_hash"]
