"""A case: a replica made by code (never a model's words), the request built from it, and the future set that scores it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import FUTURE_SEED_OFFSET

ROWS, FUTURE_ROWS = 3000, 1500


@dataclass
class Case:
    id: str
    kind: str
    seed: int
    table: pd.DataFrame  # what every arm sees
    future: pd.DataFrame  # what no arm sees: new cases, the leak columns blank as at the prediction moment
    target: str
    task: str  # "multiclass" or "regression"
    metric: str  # "macro_f1" or "mae"
    moment: str
    leaks: list[str]
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return f"{self.kind}.csv"

    def request(self) -> str:
        """The person's request, built by code from the case."""
        return (f"I attached {self.filename}. Predict `{self.target}` for new cases. The prediction is made at this moment: "
                f"{self.moment} Build the most accurate model that is honest at that moment, and tell me how well it will do "
                "on new cases, with an interval.")


def _replica(kind: str, seed: int, n: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if kind == "ticket_routing":
        from dclab_core.replicas import ticket_routing

        r = ticket_routing.generate(seed=seed, n=n)
        return r.tickets, r.facts
    if kind == "delivery_time":
        from dclab_core.replicas import delivery_time

        r = delivery_time.generate(seed=seed, n=n)
        return r.orders, r.facts
    raise ValueError(f"no case kind {kind!r}")


SPEC = {
    "ticket_routing": {"target": "assigned_team", "task": "multiclass", "metric": "macro_f1",
                       "moment": "When the ticket is created, before anyone reads it."},
    "delivery_time": {"target": "actual_minutes", "task": "regression", "metric": "mae",
                      "moment": "When the order is placed, before a courier is assigned."},
}


def make(kind: str, seed: int) -> Case:
    spec = SPEC[kind]
    table, facts = _replica(kind, seed, ROWS)
    future, _ = _replica(kind, seed + FUTURE_SEED_OFFSET, FUTURE_ROWS)
    leaks = [c for c in facts.get("leaky_columns") or [] if c in table.columns]
    future = future.copy()
    for column in leaks:  # at the prediction moment these do not exist yet
        future[column] = np.nan if pd.api.types.is_numeric_dtype(future[column]) else None
    return Case(id=f"{kind}-s{seed}", kind=kind, seed=seed, table=table.reset_index(drop=True), future=future.reset_index(drop=True),
                target=spec["target"], task=spec["task"], metric=spec["metric"], moment=spec["moment"], leaks=leaks, facts=facts)


def dev_cases() -> list[Case]:
    from . import DEV_SEEDS, KINDS

    return [make(kind, seed) for seed in DEV_SEEDS for kind in KINDS]


def held_out_cases() -> list[Case]:
    """The gate's eight cases (e2e_gate_v1), drawn by the gate command alone and run once."""
    from . import HELD_OUT_SEEDS, KINDS

    return [make(kind, seed) for seed in HELD_OUT_SEEDS for kind in KINDS]
