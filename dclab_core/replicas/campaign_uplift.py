"""A discount campaign, invented: who buys because of the offer (an uplift replica), randomised and observational.

The traps a careful data scientist must see:

* **Who buys is not who responds.** Loyal customers with many past purchases buy often with or without the offer;
  the offer changes the minds of young app users only. A model of who buys when offered targets the loyal ones and
  wastes the budget.
* **Not every campaign is an experiment.** In the observational version the marketing team sent the offer mostly to
  customers with many past purchases: the arms barely overlap, and a difference between them is not the offer's
  effect.

The planted responsive segment and every row's true uplift are in ``facts`` (for judging only).
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


def generate(seed: int = 0, n: int = 20000, randomised: bool = True) -> Replica:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 75, n)
    app = rng.binomial(1, 0.45, n)
    past = rng.poisson(2.0, n)
    tenure = rng.integers(1, 60, n)
    region = rng.choice(["north", "south", "east", "west"], n)
    base = 1 / (1 + np.exp(-(-2.4 + 0.35 * past + 0.01 * tenure)))  # loyal customers buy anyway
    responsive = (app == 1) & (age < 35)
    uplift = np.where(responsive, 0.2, 0.005)
    if randomised:
        treated = rng.binomial(1, 0.5, n)
    else:  # the team targeted its best customers: almost all with many past purchases, almost none of the others
        treated = rng.binomial(1, np.where(past >= 3, 0.97, 0.02))
    bought = rng.uniform(size=n) < np.clip(base + treated * uplift, 0, 1)
    frame = pd.DataFrame({"customer_id": [f"P{seed:02d}{i:06d}" for i in range(n)], "age": age, "app_user": app, "past_purchases": past,
                          "tenure_months": tenure, "region": region, "got_offer": treated, "bought": bought.astype(int)})
    return Replica(frame, {"treatment": "got_offer", "outcome": "bought", "randomised": randomised, "identifier": "customer_id",
                           "responsive": responsive.tolist(), "true_uplift": uplift.tolist(), "segment_rule": "app users under 35", "synthetic": True})
