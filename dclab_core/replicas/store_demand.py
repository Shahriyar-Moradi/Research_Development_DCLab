"""Store demand, invented: daily units sold per store and product, forecast a week ahead (a forecasting replica).

The traps a careful data scientist must see:

* **Known ahead, and not.** ``on_promotion`` is planned weeks ahead: known at the forecast moment for the days
  forecast. ``rain_mm`` is the day's measured rain: it moves demand, but on the forecast moment it is not known for
  the days ahead (only a weather forecast would be). Using it is a leak that backtests at a random split hide.
* **A week's season.** Weekends sell more: a model must beat the seasonal naive (the same weekday last week), not
  only the naive last value.
* **The future in a random split.** Rows of the same series are ordered in time; a shuffled split lets lags of the
  future into training. Backtests must cut at origins and use only the past.
* **Short series.** A few products start late: too short for a model of their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Replica:
    sales: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def generate(seed: int = 0, stores: int = 4, products: int = 5, days: int = 240) -> Replica:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-03-01", periods=days, freq="D")
    rows = []
    for s in range(stores):
        for p in range(products):
            start = 0 if p < products - 1 else days // 2  # the last product starts late: a short series
            base = rng.uniform(20, 80) * (1 + 0.3 * s)
            weekly = np.array([0.9, 0.85, 0.9, 1.0, 1.15, 1.4, 1.3])
            promo = rng.uniform(size=days) < 0.08
            rain = np.clip(rng.gamma(0.6, 4.0, days) - 1.0, 0, None)
            trend = 1 + 0.0008 * np.arange(days)
            mean = base * weekly[dates.dayofweek] * trend * np.where(promo, 1.5, 1.0) * np.exp(-0.06 * rain)
            units = rng.poisson(mean)
            for i in range(start, days):
                rows.append({"store": f"S{s + 1}", "product": f"P{p + 1}", "date": dates[i].strftime("%Y-%m-%d"),
                             "on_promotion": int(promo[i]), "rain_mm": round(float(rain[i]), 1), "units": int(units[i])})
    frame = pd.DataFrame(rows)
    frame["series"] = frame["store"] + "-" + frame["product"]
    return Replica(frame, {"series": "series", "time": "date", "target": "units", "known_ahead": ["on_promotion"], "leak": "rain_mm",
                           "season": 7, "horizon": 7, "synthetic": True})
