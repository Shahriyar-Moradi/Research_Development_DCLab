"""Delivery time, invented: estimate how many minutes an order will take, when it is placed (a regression replica).

The traps a careful data scientist must see:

* **A heavy tail and a cap.** Times are right-skewed, and the source system stores anything above 60 minutes as 60:
  the top value is a cap, not a measurement (censoring), so squared error chases it and the tail is unknown.
* **The target can be recomputed.** ``delivered_at`` minus ``order_time`` is the target, to the minute; and
  ``courier_rating`` is given after the delivery. Neither exists when the order is placed.
* **A few impossible values.** Some distances are negative (a sign error in one data feed).
* **Restaurants repeat.** Each restaurant has its own preparation speed: grouping by restaurant shows how a model does on
  a new one.
* **Weather goes missing more in storms.** Missing ``weather`` happens more often on slow days, so missingness speaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

CAP = 60.0


@dataclass(frozen=True)
class Replica:
    orders: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def generate(seed: int = 0, n: int = 8000, days: int = 60, n_restaurants: int = 300) -> Replica:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-05-05")
    restaurants = np.arange(7_000, 7_000 + n_restaurants)
    prep = dict(zip(restaurants, rng.gamma(4, 3.0, n_restaurants)))  # minutes of preparation, per restaurant

    weights = rng.lognormal(0, 0.9, n_restaurants)  # a few busy restaurants, none with half of all orders
    restaurant = rng.choice(restaurants, n, p=weights / weights.sum())
    order_time = start + pd.to_timedelta(rng.integers(0, days * 86_400, n), unit="s")
    hour = order_time.hour.to_numpy()
    distance = np.round(rng.gamma(2.2, 1.4, n), 2)
    couriers = rng.integers(5, 60, n)
    storm = rng.random(n) < 0.08
    weather = np.where(storm, "storm", rng.choice(["clear", "rain", "cloudy"], n, p=[0.6, 0.2, 0.2]))
    rush = np.isin(hour, [12, 13, 19, 20, 21])
    minutes = (np.array([prep[r] for r in restaurant]) + 3.0 * distance + 60 / couriers + 5 * rush + 8 * storm
               + rng.normal(0, 2, n))
    delayed = rng.random(n) < 0.25  # a quarter of orders hit a delay (a queue, a lost courier): the long right tail
    minutes = minutes + np.where(delayed, rng.exponential(22, n), 0.0)
    minutes = np.round(minutes, 1)
    capped = np.minimum(minutes, CAP)

    weather_obs = np.where((storm & (rng.random(n) < 0.45)) | (rng.random(n) < 0.03), None, weather)
    bad = rng.random(n) < 0.004
    frame = pd.DataFrame({
        "order_id": np.arange(900_000, 900_000 + n),
        "order_time": order_time,
        "restaurant_id": restaurant,
        "distance_km": np.where(bad, -distance, distance),
        "couriers_online": couriers,
        "weather": weather_obs,
        "basket_items": rng.poisson(3, n) + 1,
        "order_value": np.round(rng.gamma(3, 9, n), 2),
        "delivered_at": order_time + pd.to_timedelta(capped, unit="m"),     # recorded after delivery
        "courier_rating": np.clip(np.round(5.4 - capped / 25 + rng.normal(0, 0.5, n)), 1, 5),  # given after delivery
        "actual_minutes": capped,
    }).sort_values("order_time", kind="stable").reset_index(drop=True)
    facts = {
        "rows": n, "cap": CAP, "capped_share": float((minutes >= CAP).mean()), "negative_distances": int(bad.sum()),
        "leaky_columns": ["delivered_at", "courier_rating"], "entity": "restaurant_id",
        "missing_weather_in_storms": float(pd.isna(weather_obs[storm]).mean()), "seed": seed,
    }
    return Replica(orders=frame, facts=facts)
