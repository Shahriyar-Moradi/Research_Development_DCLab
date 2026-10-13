"""Subscription lifetimes, invented: how long until a subscriber cancels, when many have not cancelled yet (a
time-to-event replica).

The traps a careful data scientist must see:

* **Right-censoring.** Subscribers sign up over two years and the data ends on one day: a subscriber who signed up
  last month and is still active has a duration of one month and no event. Dropping such rows keeps the short
  lives and forgets the long ones; calling them cancellations ends lives that go on. Both make subscriptions look
  shorter than they are.
* **Counted during the follow-up.** ``months_billed`` and ``tickets_so_far`` grow with every month a subscriber stays:
  they are not known at signup, the moment the forecast is made, and they predict the duration only because they
  measure it.
* **Rows arrive over time.** The holdout is the latest signups, whose follow-up is short: a horizon is scored only
  where the follow-up reaches it.

The true survival at each horizon and every row's true lifetime are in ``facts`` (before censoring), so a test can say
which estimate, and which score, is honest; a solver never reads them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Replica:
    subscribers: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def generate(seed: int = 0, n: int = 3000, days: int = 730) -> Replica:
    rng = np.random.default_rng(seed)
    start = np.sort(rng.integers(0, days, n))
    plan = rng.choice(["basic", "plus", "pro"], n, p=[0.5, 0.3, 0.2])
    discount = rng.binomial(1, 0.3, n)  # signed up on a discount: deal seekers leave sooner
    calls = rng.poisson(0.8, n)  # support calls in the signup week, known at signup
    age = rng.integers(18, 75, n)
    eta = 0.6 * discount + 0.35 * calls - 0.5 * (plan == "pro") - 0.2 * (plan == "plus") - 0.01 * (age - 40)
    scale, shape = 420.0, 1.3  # Weibull lifetimes, proportional hazards in eta
    life = scale * (-np.log(rng.uniform(size=n)) / np.exp(eta)) ** (1 / shape)
    follow = days - start  # the data ends on day `days`
    observed = np.minimum(life, follow).round().clip(1, None)
    cancelled = (life <= follow).astype(int)
    months = np.ceil(observed / 30).astype(int)
    frame = pd.DataFrame({
        "subscriber_id": [f"L{seed:02d}{i:06d}" for i in range(n)],
        "signup_date": (pd.Timestamp("2024-01-01") + pd.to_timedelta(start, unit="D")).strftime("%Y-%m-%d"),
        "plan": plan, "signup_discount": discount, "signup_week_calls": calls, "age": age,
        "months_billed": months, "tickets_so_far": rng.poisson(0.25 * months + 0.1),
        "days_subscribed": observed.astype(int), "cancelled": cancelled,
    })
    horizons = (90, 180, 365)
    truth = {str(h): round(float(np.mean(life > h)), 3) for h in horizons}
    return Replica(frame, {"duration": "days_subscribed", "event": "cancelled", "start": "signup_date", "group": "plan",
                           "identifier": "subscriber_id", "leaks": ["months_billed", "tickets_so_far"], "horizons": list(horizons),
                           "true_survival": truth, "true_lifetime": [round(float(x), 2) for x in life],  # for judging only
                           "censored_share": round(float(1 - cancelled.mean()), 3), "synthetic": True})
