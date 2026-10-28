"""KPIs and dashboards on a person's table (package H10 of the Home plan).

A ``Kpi`` is a named question asked of a table, as a ``dclab_core.query.QueryPlan`` (so every number on a dashboard comes
from the same engine as an answer, with its row floor), with its version: a changed definition is a new version, never an
edit in place. A ``Dashboard`` holds its tiles (one per KPI, its chart chosen by code), who it is for, how often it is
refreshed, the data version (the table's hash) every tile was computed on, and each ``Refresh`` run: when, on which data
version, and per tile its plan's hash and the groups the floor hid. ``refresh`` computes every tile on a table and records
the run; a tile that fails says why, never stops the others.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import pandas as pd
from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version
from dclab_core.query import FLOOR, QueryPlan, QueryResult, execute, plan_hash

Refresh = Literal["daily", "weekly", "monthly", "on_new_data"]


class Kpi(Contract):
    name: str = Field(max_length=120, description="what the tile shows, written by code from the plan")
    plan: QueryPlan
    version: int = Field(default=1, ge=1)
    said: str = Field(default="", max_length=200, description="the person's words this KPI was read from")


class TileRun(Contract):
    kpi: str = Field(max_length=120)
    plan_hash: str = Field(max_length=32)
    suppressed_groups: int = 0
    error: Optional[str] = Field(default=None, max_length=300)


class RefreshRun(Contract):
    at: str = Field(max_length=40)
    data_version: Optional[str] = Field(default=None, max_length=64, description="the table's hash the tiles were computed on")
    tiles: list[TileRun] = Field(default_factory=list)


class Dashboard(Versioned):
    schema_version: Literal["dclab_core.dashboard.v1"] = _version("dashboard")
    title: str = Field(max_length=200)
    audience: Optional[str] = Field(default=None, max_length=120)
    refresh: Refresh = "on_new_data"
    kpis: list[Kpi] = Field(default_factory=list, max_length=12)
    data_version: Optional[str] = Field(default=None, max_length=64)
    refreshed_at: Optional[str] = Field(default=None, max_length=40)
    runs: list[RefreshRun] = Field(default_factory=list)
    floor: int = FLOOR


CONTRACTS["dashboard"] = Dashboard


def refresh(board: Dashboard, frame: pd.DataFrame, *, data_version: Optional[str], at: str, floor: int = FLOOR,
            chart: Optional[Callable[[QueryPlan, QueryResult], Any]] = None) -> tuple[Dashboard, list[dict[str, Any]]]:
    """Every tile computed on ``frame``; the run recorded on the board. Returns the board and, per tile, its result (as a
    dict), its chart (when ``chart`` is given) or its error."""
    tiles, runs = [], []
    for k in board.kpis:
        try:
            result = execute(k.plan, frame, floor=floor)
            tiles.append({"kpi": k.name, "version": k.version, "result": result.model_dump(mode="json"),
                          **({"chart": chart(k.plan, result)} if chart else {})})
            runs.append(TileRun(kpi=k.name, plan_hash=plan_hash(k.plan), suppressed_groups=result.suppressed_groups))
        except ValueError as error:  # one tile that cannot run is said; the others still refresh
            tiles.append({"kpi": k.name, "version": k.version, "error": str(error)[:300]})
            runs.append(TileRun(kpi=k.name, plan_hash=plan_hash(k.plan), error=str(error)[:300]))
    run = RefreshRun(at=at, data_version=data_version, tiles=runs)
    return board.model_copy(update={"data_version": data_version, "refreshed_at": at, "runs": [*board.runs, run][-50:]}), tiles


__all__ = ["Kpi", "TileRun", "RefreshRun", "Dashboard", "refresh"]
