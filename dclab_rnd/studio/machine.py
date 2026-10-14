"""The engine's side of kinds with their own machine (package 16.15): what it prepares for them. The machine itself,
its stages and the shared record fields are in ``dclab_rnd.kinds.machine``."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..kinds.machine import COMMON, Machine, bootstrap_mean, claim, shape  # noqa: F401  (re-exported for the engine)
from .solution import Solution


class MachinePrepared:
    """What the engine prepares for a kind with its own machine (the tabular ``Prepared``'s counterpart)."""

    def __init__(self, store: Any, project: dict[str, Any], solution: Solution, kind: Any, frame: pd.DataFrame,
                 tables: dict[str, pd.DataFrame] | None = None):
        self.store, self.project, self.solution, self.kind, self.frame = store, project, solution, kind, frame
        cls = kind.own_machine()
        # 16.15 C: a machine that reads related tables gets them; the others keep their four arguments
        self.machine: Machine = (cls(kind, solution, frame, project, tables=tables or {}) if getattr(cls, "uses_tables", False)
                                 else cls(kind, solution, frame, project))
        self.sampling = {"source_rows": int(len(frame)), "rows_used": int(len(frame)), "rule": "every row (the kind's machine reads the whole table)"}
        self.plan, self.label_info, self.critic, self.class_labels = project.get("plan"), project.get("labels"), None, None

    @property
    def task(self) -> str:
        return self.solution.task

    @property
    def metric(self) -> str:
        return self.machine.primary_metric

    @property
    def task_type(self) -> str:
        return self.machine.task_type

    def holdout_policy(self) -> str:
        return self.machine.holdout_policy()

    @property
    def cv_description(self) -> str:
        return self.machine.cv_description()

    @property
    def holdout_description(self) -> str:
        return self.machine.holdout_description()


def prepare(store: Any, project: dict[str, Any], solution: Solution, kind: Any) -> MachinePrepared:
    from .data import data_path, load_table

    frame = load_table(data_path(store, project))  # checked against the recorded SHA-256
    solution.check_columns(list(frame.columns))
    missing = [c for c in kind.columns_of(solution) if c not in frame.columns]
    if missing:
        raise ValueError(f"The {kind.title.lower()} kind names columns the table does not have: {', '.join(missing)}")
    from .data import load_tables

    tables = load_tables(store, project, kind.tables_of(solution))  # each checked against its recorded SHA-256
    for name, columns in kind.table_columns_of(solution).items():
        lost = [c for c in columns if c not in tables[name].columns]
        if lost:
            raise ValueError(f"The table {name} has no column {', '.join(lost)}")
    return MachinePrepared(store, project, solution, kind, frame, tables)
