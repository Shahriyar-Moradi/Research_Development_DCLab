"""Customers, invented, with three planted behaviours and noise: groups with no outcome to predict (a segmentation
replica).

The behaviours: bargain hunters (frequent small baskets, mostly on discount, through the app), loyal premium buyers
(regular large baskets, little discount, in store) and occasional visitors (rare, medium baskets, recently inactive).
A tenth of the customers follow none of them. ``region`` and ``signup_week`` carry no behaviour: a segmentation that
splits on them is splitting noise. The planted behaviour of each row is in ``facts`` (for judging only), and so is a
later outcome (``renewed``) that differs by behaviour.

The traps: no ground truth in the table; skewed amounts that dominate an unscaled distance; a number of segments that
looks better with more groups but does not come back when the data is resampled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Replica:
    customers: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def generate(seed: int = 0, n: int = 1500) -> Replica:
    rng = np.random.default_rng(seed)
    kind = rng.choice([0, 1, 2, 3], n, p=[0.35, 0.3, 0.25, 0.1])  # 3: no behaviour
    orders = np.select([kind == 0, kind == 1, kind == 2], [rng.gamma(9, 0.8, n), rng.gamma(6, 0.7, n), rng.gamma(2, 0.4, n)], rng.gamma(2, 2, n))
    basket = np.select([kind == 0, kind == 1, kind == 2], [rng.lognormal(3.0, 0.3, n), rng.lognormal(4.6, 0.3, n), rng.lognormal(3.8, 0.3, n)],
                       rng.lognormal(3.8, 0.8, n))
    discount = np.clip(np.select([kind == 0, kind == 1, kind == 2], [rng.normal(0.65, 0.1, n), rng.normal(0.08, 0.05, n), rng.normal(0.3, 0.1, n)],
                                 rng.uniform(0, 1, n)), 0, 1)
    idle = np.select([kind == 0, kind == 1, kind == 2], [rng.exponential(6, n), rng.exponential(10, n), rng.exponential(70, n) + 30], rng.exponential(40, n))
    channel = np.where(kind == 0, rng.choice(["app", "web", "store"], n, p=[0.8, 0.15, 0.05]),
                       np.where(kind == 1, rng.choice(["app", "web", "store"], n, p=[0.1, 0.2, 0.7]),
                                np.where(kind == 2, rng.choice(["app", "web", "store"], n, p=[0.2, 0.6, 0.2]), rng.choice(["app", "web", "store"], n))))
    frame = pd.DataFrame({"customer_id": [f"C{seed:02d}{i:05d}" for i in range(n)], "orders_per_month": orders.round(2),
                          "avg_basket": basket.round(2), "discount_share": discount.round(3), "days_since_last_order": idle.round(),
                          "channel": channel, "region": rng.choice(["north", "south", "east", "west"], n),
                          "signup_week": rng.integers(1, 53, n)})
    renewed = (rng.uniform(size=n) < np.select([kind == 0, kind == 1, kind == 2], [0.55, 0.85, 0.3], 0.5)).astype(int)
    return Replica(frame, {"behaviour": kind.tolist(), "behaviours": 3, "noise_label": 3, "noise_columns": ["region", "signup_week"],
                           "identifier": "customer_id", "renewed": renewed.tolist(), "synthetic": True})
