"""One held-out case of ``hp_gate_v3`` built by one arm: v2's builder, unchanged (its hash is pinned in both gates), and
what the built project then holds that v3 also reads: the solution's split columns, whether an entity split was set on
a column the case says needs none, and the planted traps by kind."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..planning_gate_v2.run import TRAP_NAME, Watched, build as build_v2, load_held as _load, planted, request, seed  # noqa: F401

HELD = Path(__file__).resolve().parent / "held"


def build(case: Mapping[str, Any], frames: Mapping[str, Any], arm: str, folder: Path, gateway: Any = None, spent: Any = None) -> dict[str, Any]:
    from dclab_rnd.storage import open_stores

    row = build_v2(case, frames, arm, folder, gateway=gateway, spent=spent)
    needless = case.get("no_group_needed")
    row["no_group_case"] = bool(needless)
    try:  # the review: a reader error must never lose a paid row nor stop the one run
        solution = (open_stores(folder)[0].get(row["project_id"]) or {}).get("solution") or {}
        row.update(time_column=solution.get("time_column"), group_column=solution.get("group_column"),
                   needless_group=int(bool(needless) and solution.get("group_column") == needless),
                   planted_by_kind={k: [sum(1 for t in row["planted_traps"] if t["kind"] == k),
                                        sum(1 for t in row["planted_traps"] if t["kind"] == k and t["enforced"])]
                                    for k in sorted({t["kind"] for t in row["planted_traps"]})})
    except Exception as error:  # noqa: BLE001
        row.update(reader_error=type(error).__name__, needless_group=None)
    return row


def load_held(folder: Optional[Path] = None) -> Any:
    return _load(folder or HELD)
