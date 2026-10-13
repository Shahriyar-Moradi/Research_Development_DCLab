"""A shop's views and purchases, invented: which items to show each customer next (a ranking replica).

The traps a careful data scientist must see:

* **The future in the item table.** ``lifetime_sales`` is each item's purchases over the whole log, the weeks after
  the recommendation moment included. Ranking by it looks excellent and cannot be done on the day.
* **Items that did not exist yet.** New items keep arriving; an item added after the moment cannot be recommended at
  it, and a popularity count taken over the whole log would put the new hits on top.
* **Popularity is strong.** A few items sell to everyone: every model must be compared with the most popular items,
  and with the items popular lately.
* **Taste.** Each customer prefers a few categories: a model that learns it beats popularity on the customers who
  have a history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Replica:
    interactions: pd.DataFrame
    items: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def generate(seed: int = 0, users: int = 800, items: int = 300, days: int = 120, categories: int = 6) -> Replica:
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, categories, items)
    quality = rng.gamma(1.2, 1.0, items)  # a few items everyone likes
    created = np.where(rng.uniform(size=items) < 0.8, 0, rng.integers(20, days, items))  # a fifth arrive later
    price = np.round(rng.lognormal(3, 0.5, items), 2)
    taste = rng.dirichlet(np.full(categories, 0.4), users)  # each customer leans to a few categories
    activity = rng.gamma(1.5, 1.0, users)
    rows = []
    for day in range(days):
        live = np.flatnonzero(created <= day)
        fresh = np.exp(-(day - created[live]) / 30.0)  # a new item gets attention for a few weeks
        for u in np.flatnonzero(rng.uniform(size=users) < 0.06 * activity):
            w = taste[u, cat[live]] * quality[live] * (1 + 2 * fresh)
            chosen = rng.choice(live, size=min(len(live), 1 + rng.poisson(1.5)), replace=False, p=w / w.sum())
            for i in chosen:
                rows.append((u, i, day, "view"))
                if rng.uniform() < 0.15 + 0.25 * taste[u, cat[i]]:
                    rows.append((u, i, day, "purchase"))
    log = pd.DataFrame(rows, columns=["user", "item", "day", "event"])
    log["user_id"] = "U" + log["user"].astype(str).str.zfill(4)
    log["item_id"] = "I" + log["item"].astype(str).str.zfill(4)
    start = pd.Timestamp("2025-01-01")
    log["time"] = (start + pd.to_timedelta(log["day"], unit="D") + pd.to_timedelta(rng.integers(0, 86400, len(log)), unit="s")).dt.strftime("%Y-%m-%d %H:%M:%S")
    sales = log[log.event == "purchase"].groupby("item").size().reindex(range(items), fill_value=0)
    catalogue = pd.DataFrame({"item_id": ["I" + str(i).zfill(4) for i in range(items)], "category": [f"c{c}" for c in cat],
                              "price": price, "created_at": (start + pd.to_timedelta(created, unit="D")).strftime("%Y-%m-%d"),
                              "lifetime_sales": sales.to_numpy()})
    moment = (start + pd.Timedelta(days=days - 21)).strftime("%Y-%m-%d")
    return Replica(log[["user_id", "item_id", "time", "event"]].sort_values("time").reset_index(drop=True), catalogue,
                   {"user": "user_id", "item": "item_id", "time": "time", "event": "event", "relevant": "purchase", "moment": moment,
                    "window_days": 14, "created": "created_at", "leak": "lifetime_sales", "k": 10, "synthetic": True})
