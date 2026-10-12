"""Ride pricing, invented: a courier is offered a fee for a ride and accepts it or not (a pricing replica, shaped like the
first part of a delivery challenge: choose the fee).

Two versions of the same market:

* ``varied=True``: the fee offered is the usual fee for the trip times a multiplier drawn at random (a pricing test), so
  the data can show how acceptance responds to the fee.
* ``varied=False``: the fee is a fixed formula of distance and the hour, never varied on its own: the data cannot tell
  the fee's effect from the trip's, and a careful data scientist says so instead of drawing a curve.

The traps a careful data scientist must see:

* **Acceptance after the offer.** ``accepted_at`` exists only when the courier accepted: it is a copy of the label.
* **The fee is a decision, not only a feature.** Choosing it needs a model of acceptance by fee and a margin, not the
  best accuracy.
* **Edges.** The best fee must be read within the fees tried.

The truth (``facts``): acceptance is logistic in the log of the multiplier with elasticity ``ELASTICITY``, plus the
trip's context; the margin per offer at multiple m is ``(m * usual_fee - cost) * P(accept | m)`` with the platform
earning ``CUSTOMER_PRICE_FACTOR`` times the usual fee, so the best multiple can be computed exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

ELASTICITY = 4.0          # log-odds of acceptance per unit of log(multiplier)
CUSTOMER_PRICE_FACTOR = 1.6
MULTIPLIERS = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3)


@dataclass(frozen=True)
class Replica:
    offers: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def _accept_probability(multiplier: np.ndarray, base_logit: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(base_logit + ELASTICITY * np.log(multiplier))))


def generate(seed: int = 0, n: int = 6000, varied: bool = True) -> Replica:
    rng = np.random.default_rng(seed)
    distance = rng.gamma(2.5, 1.6, n).round(2) + 0.3
    hour = rng.integers(0, 24, n)
    rain = rng.binomial(1, 0.15, n)
    couriers_nearby = rng.poisson(6, n)
    usual_fee = (12000 + 4000 * distance + 3000 * ((hour >= 18) & (hour <= 21))).round(-2)
    multiplier = rng.choice(MULTIPLIERS, n) if varied else np.ones(n)
    fee = (usual_fee * multiplier).round(-2)
    base_logit = 0.3 - 0.12 * distance - 0.6 * rain + 0.08 * couriers_nearby
    accepted = rng.uniform(size=n) < _accept_probability(fee / usual_fee, base_logit)
    created = pd.Timestamp("2026-02-01") + pd.to_timedelta(np.sort(rng.integers(0, 30 * 86400, n)), unit="s")
    accepted_at = np.where(accepted, (created + pd.to_timedelta(rng.integers(5, 90, n), unit="s")).strftime("%Y-%m-%d %H:%M:%S"), None)
    frame = pd.DataFrame({"offer_id": [f"O{seed:02d}{i:06d}" for i in range(n)], "created": created.strftime("%Y-%m-%d %H:%M:%S"),
                          "distance_km": distance, "hour": hour, "rain": rain, "couriers_nearby": couriers_nearby,
                          "usual_fee": usual_fee, "fee": fee, "accepted_at": accepted_at, "accepted": accepted.astype(int)})
    grid = np.linspace(0.8, 1.3, 51)
    margin = [float(np.mean((CUSTOMER_PRICE_FACTOR * usual_fee - m * usual_fee) * _accept_probability(np.full(n, m), base_logit)))
              for m in grid]
    return Replica(frame, {"target": "accepted", "price": "fee", "reference": "usual_fee", "leak": "accepted_at", "varied": varied,
                           "elasticity": ELASTICITY, "customer_price_factor": CUSTOMER_PRICE_FACTOR,
                           "best_multiple": float(grid[int(np.argmax(margin))]), "synthetic": True})
