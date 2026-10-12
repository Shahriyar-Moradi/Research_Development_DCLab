"""Subscription churn, invented: which subscribers cancel before their next renewal (a binary replica).

The traps a careful data scientist must see:

* **A column written after the outcome.** ``final_refund_amount`` is paid only when a subscriber cancels: it is close to
  a copy of the label and does not exist at renewal time.
* **An identifier.** ``subscriber_id`` is unique per row; a model can memorise it and learn nothing.
* **Rows arrive over time.** Subscribers sign up over two years and the cancel rate drifts upward: a random split
  mixes the future into training, the holdout should be the latest signups.
* **Accuracy flatters.** About a quarter cancel: predicting "stays" for everyone is right three times out of four.
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


def generate(seed: int = 0, n: int = 3000) -> Replica:
    rng = np.random.default_rng(seed)
    signup = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 730 * 24, n)), unit="h")
    plan = rng.choice(["basic", "plus", "pro"], n, p=[0.5, 0.3, 0.2])
    tenure = rng.integers(1, 36, n)
    calls = rng.poisson(1.1, n)
    fee = np.where(plan == "basic", 9.0, np.where(plan == "plus", 19.0, 39.0)) * rng.uniform(0.9, 1.1, n)
    late = rng.binomial(1, 0.12, n)
    drift = np.linspace(-0.3, 0.3, n)  # later signups cancel more
    logit = -1.4 + 0.55 * calls - 0.06 * tenure + 0.9 * late + 0.5 * (plan == "basic") + drift
    churned = rng.uniform(size=n) < 1 / (1 + np.exp(-logit))
    frame = pd.DataFrame({
        "subscriber_id": [f"S{seed:02d}{i:06d}" for i in range(n)],
        "signup_time": signup.strftime("%Y-%m-%d %H:%M:%S"),
        "plan": plan, "monthly_fee": fee.round(2), "tenure_months": tenure, "support_calls": calls, "late_payments": late,
        "final_refund_amount": np.where(churned, rng.uniform(4, 40, n), 0.0).round(2),
        "churned": np.where(churned, "yes", "no"),
    })
    return Replica(frame, {"target": "churned", "positive": "yes", "leak": "final_refund_amount", "identifier": "subscriber_id",
                           "time_column": "signup_time", "positive_rate": float(churned.mean()), "synthetic": True})
