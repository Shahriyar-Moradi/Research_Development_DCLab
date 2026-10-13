"""Flat prices in a city, invented: a regression where places cluster (a spatial replica).

The traps a careful data scientist must see:

* **Neighbours are alike.** Each district has its own price level, a smooth field over the city that no column but
  the coordinates carries. A random split puts a flat's neighbours in training and the flat in the test: the score
  measures how well the model remembers the district, not how it prices a flat in an area it has not seen.
* **Places cluster.** Flats come in neighbourhoods of different density; a block of the city is the honest unit to hold
  out.

The price field's centres and the correlation range are in ``facts`` (for judging only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Replica:
    flats: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def generate(seed: int = 0, n: int = 3000, districts: int = 14) -> Replica:
    rng = np.random.default_rng(seed)
    centre_lat, centre_lon = 35.70, 51.40
    km = 111.32
    cx, cy = rng.uniform(-14, 14, districts), rng.uniform(-14, 14, districts)  # district centres, km
    pick = rng.integers(0, districts, n)
    x, y = cx[pick] + rng.normal(0, 2.5, n), cy[pick] + rng.normal(0, 2.5, n)
    bumps = rng.normal(0, 0.35, districts)
    level = sum(b * np.exp(-((x - a) ** 2 + (y - c) ** 2) / (2 * 4.0 ** 2)) for a, c, b in zip(cx, cy, bumps))  # a smooth field, ~4 km wide
    size = rng.gamma(6, 13, n).round()
    rooms = np.clip(np.round(size / 30 + rng.normal(0, 0.6, n)), 1, 6)
    age = rng.integers(0, 50, n)
    log_price = 9.5 + 0.9 * np.log(size) - 0.006 * age + 1.3 * level + rng.normal(0, 0.12, n)
    frame = pd.DataFrame({"lat": (centre_lat + y / km).round(6), "lon": (centre_lon + x / (km * np.cos(np.radians(centre_lat)))).round(6),
                          "size_m2": size, "rooms": rooms, "age_years": age, "price": np.exp(log_price).round(-3)})
    return Replica(frame, {"target": "price", "lat": "lat", "lon": "lon", "field_width_km": 4.0, "districts": districts, "synthetic": True})
