"""A shop's customers, orders and support tickets, invented: which customers stop ordering in the 30 days after a
cutoff (a several-tables replica).

The traps a careful data scientist must see:

* **Events after the cutoff.** The question is asked on a date; a customer's orders and tickets after it are the
  answer, not a feature. Counting all of a customer's orders reads the future.
* **A snapshot taken at the end.** The customer table was exported on the last day: ``last_order_date`` and
  ``lifetime_orders`` describe the customer at the export, not at any cutoff.
* **A join that repeats rows.** ``addresses`` holds several rows per customer: joined as it is, it repeats customers.
* **Slowing down before leaving.** Customers who stop ordering first order less often and open more tickets: window
  counts and trends see it; one total does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Replica:
    customers: pd.DataFrame
    orders: pd.DataFrame
    tickets: pd.DataFrame
    addresses: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def generate(seed: int = 0, n: int = 1500, days: int = 365) -> Replica:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    signup = rng.integers(0, days // 2, n)
    leave = np.where(rng.uniform(size=n) < 0.6, signup + rng.integers(60, days, n), days + 100)  # the day a customer stops ordering
    rate = rng.gamma(2.0, 0.05, n)  # orders a day while active
    orders, tickets = [], []
    for c in range(n):
        for d in range(signup[c], min(leave[c], days)):
            slowing = 0.35 if leave[c] - d < 45 else 1.0  # the weeks before leaving: fewer orders, more tickets
            if rng.uniform() < rate[c] * slowing:
                orders.append((c, d, round(float(rng.lognormal(3.3, 0.4)), 2)))
            if rng.uniform() < (0.012 if leave[c] - d < 45 else 0.003):
                tickets.append((c, d, rng.choice(["low", "high"], p=[0.7, 0.3])))
    cid = lambda c: f"K{seed:02d}{c:05d}"  # noqa: E731
    stamp = lambda d: (start + pd.Timedelta(days=int(d)) + pd.Timedelta(seconds=int(rng.integers(0, 86400)))).strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    o = pd.DataFrame([(cid(c), stamp(d), a) for c, d, a in orders], columns=["customer_id", "ordered_at", "amount"])
    t = pd.DataFrame([(cid(c), stamp(d), s) for c, d, s in tickets], columns=["customer_id", "opened_at", "severity"])
    last = o.groupby("customer_id")["ordered_at"].max()
    customers = pd.DataFrame({"customer_id": [cid(c) for c in range(n)],
                              "signup_date": [(start + pd.Timedelta(days=int(s))).strftime("%Y-%m-%d") for s in signup],
                              "plan": rng.choice(["basic", "plus"], n, p=[0.7, 0.3]), "region": rng.choice(["north", "south"], n)})
    customers["last_order_date"] = customers["customer_id"].map(last).str[:10]
    customers["lifetime_orders"] = customers["customer_id"].map(o.groupby("customer_id").size()).fillna(0).astype(int)
    addresses = pd.DataFrame({"customer_id": np.repeat(customers["customer_id"], rng.integers(1, 4, n)).to_numpy()})
    addresses["city"] = rng.choice(["Tehran", "Shiraz", "Tabriz"], len(addresses))
    cutoffs = [(start + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in range(150, days - 29, 30)]
    cutoffs.insert(-1, (start + pd.Timedelta(days=range(150, days - 29, 30)[-1] - 15)).strftime("%Y-%m-%d"))  # its horizon runs into the holdout's
    return Replica(customers, o, t, addresses, {"key": "customer_id", "cutoffs": cutoffs, "horizon_days": 30, "target_table": "orders",
                                                 "written_later": ["last_order_date", "lifetime_orders"], "repeating_table": "addresses",
                                                 "late_cutoff": (start + pd.Timedelta(days=days - 10)).strftime("%Y-%m-%d"),
                                                 "synthetic": True})
