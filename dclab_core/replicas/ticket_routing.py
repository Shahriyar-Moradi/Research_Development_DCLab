"""Ticket routing, invented: route each support ticket to one of 12 teams when it is created (a multiclass replica).

The traps a careful data scientist must see:

* **Tiny classes.** Two teams receive about 0.3 % of tickets each: accuracy hides them, macro metrics and stratified
  folds do not; merging or a hierarchy is worth a look.
* **Columns written after the moment.** ``resolved_by`` (the agent who closed the ticket, always from the handling
  team) is close to a copy of the label; ``first_response_minutes`` depends on the team's queue. Neither exists when the
  ticket is created.
* **Repeated tickets.** About 2 % of rows are exact duplicates (a form submitted twice): a split can put copies on both
  sides.
* **Customers repeat.** Customers open several tickets and keep to a few teams: grouping by customer is worth testing.
* **A team appears late.** ``integrations`` starts in the last month, so a random split overstates how well it is learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TEAMS = ("billing", "login", "technical", "shipping", "returns", "account", "mobile_app", "web_app", "data_export",
         "integrations", "security", "legal")
AREAS = {"billing": "payments", "login": "access", "technical": "platform", "shipping": "orders", "returns": "orders",
         "account": "access", "mobile_app": "apps", "web_app": "apps", "data_export": "platform", "integrations": "platform",
         "security": "access", "legal": "policy"}
WEIGHTS = np.array([18, 14, 16, 12, 10, 9, 7, 6, 4, 3, 0.35, 0.3])


@dataclass(frozen=True)
class Replica:
    tickets: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def generate(seed: int = 0, n: int = 6000, days: int = 120) -> Replica:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-06")
    customers = np.arange(10_000, 10_000 + n // 4)
    affinity = {c: rng.choice(len(TEAMS), 2, replace=False, p=WEIGHTS / WEIGHTS.sum()) for c in customers}
    plan_of = dict(zip(customers, rng.choice(["free", "pro", "enterprise"], len(customers), p=[0.6, 0.3, 0.1])))

    customer = rng.choice(customers, n)
    day = rng.integers(0, days, n)
    team = np.empty(n, dtype=object)
    for i in range(n):
        if rng.random() < 0.55:  # customers keep to their usual teams
            team[i] = TEAMS[rng.choice(affinity[customer[i]])]
        else:
            team[i] = TEAMS[rng.choice(len(TEAMS), p=WEIGHTS / WEIGHTS.sum())]
    late = day < days - 30
    team = np.where((team == "integrations") & late, "technical", team)  # integrations only exists in the last month

    area = np.array([AREAS[t] if rng.random() < 0.7 else rng.choice(sorted(set(AREAS.values()))) for t in team])
    words = {"mentions_invoice": ("billing", "returns"), "mentions_password": ("login", "security", "account"),
             "mentions_error": ("technical", "mobile_app", "web_app", "integrations"), "mentions_parcel": ("shipping", "returns"),
             "mentions_export": ("data_export", "integrations"), "mentions_contract": ("legal", "billing")}
    frame = pd.DataFrame({
        "ticket_id": np.arange(500_000, 500_000 + n),
        "created_at": (start + pd.to_timedelta(day, unit="D") + pd.to_timedelta(rng.integers(0, 86_400, n), unit="s")),
        "customer_id": customer,
        "plan": [plan_of[c] for c in customer],
        "channel": rng.choice(["email", "chat", "phone", "form"], n, p=[0.4, 0.3, 0.1, 0.2]),
        "product_area": area,
        "subject_words": rng.poisson(7, n) + 1,
        "body_words": rng.poisson(60, n) + 5,
        "has_attachment": rng.random(n) < 0.2,
    })
    for column, owners in words.items():
        frame[column] = np.where(np.isin(team, owners), rng.random(n) < 0.6, rng.random(n) < 0.06)
    agents = {t: [f"agent_{t[:3]}_{k}" for k in range(3 if t not in ("security", "legal") else 1)] for t in TEAMS}
    frame["resolved_by"] = [rng.choice(agents[t]) for t in team]                      # written when the ticket is closed
    queue = {t: rng.uniform(10, 240) for t in TEAMS}
    frame["first_response_minutes"] = np.round([rng.gamma(2, queue[t] / 2) for t in team], 1)  # after the moment
    frame["assigned_team"] = team
    frame = frame.sort_values("created_at", kind="stable").reset_index(drop=True)

    dup = rng.choice(len(frame), int(0.02 * n), replace=False)
    frame = pd.concat([frame, frame.iloc[dup]], ignore_index=True).sort_values("created_at", kind="stable").reset_index(drop=True)
    counts = frame["assigned_team"].value_counts()
    facts = {
        "rows": int(len(frame)), "classes": int(counts.size), "duplicate_rows": int(frame.duplicated().sum()),
        "tiny_classes": sorted(counts[counts / len(frame) < 0.01].index.tolist()),
        "late_class": "integrations", "leaky_columns": ["resolved_by", "first_response_minutes"],
        "entity": "customer_id", "seed": seed,
    }
    return Replica(tickets=frame, facts=facts)
