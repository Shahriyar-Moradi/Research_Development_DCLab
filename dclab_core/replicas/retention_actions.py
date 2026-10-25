"""Keeping customers with a call or an email, invented: which of two actions each customer gets (a two-action replica).

A subscription business can call a customer (a person's time, so few calls a week) or send an email (almost free, as
many as it likes). A past test gave each customer no contact, an email or a call at random, and recorded who stayed. A
customer kept is worth a year of their plan (``monthly_fee`` times 12, known when the action is chosen); a call costs
15, an email 0.5.

The traps a careful data scientist must see:

* **The customers a call helps are not the ones an email helps.** An email moves app users a little; a call moves older
  customers who have called support before a lot. One ranking for "contact" mixes the two.
* **Who leaves is not who is saved.** Customers in their first months leave most often, with or without a contact: a
  model of who leaves spends the calls on them.
* **A call must beat an email, not only no contact.** A customer an email keeps nearly as well should get the email;
  the scarce calls go where the call's extra lift, times the customer's value, pays its extra cost.
* **Compare the arms, never the contacted with the rest.** Only the random test's arms tell an action's effect; the
  value of an action for a row is its lift over no contact, estimated from the arms on the training rows only.

Every row's true lift of each action is in ``facts`` (for judging only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

ACTIONS = ("email", "call")
COSTS = {"email": 0.5, "call": 15.0}


@dataclass(frozen=True)
class Replica:
    customers: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def generate(seed: int = 0, n: int = 24000) -> Replica:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n)
    app = rng.binomial(1, 0.5, n)
    tenure = rng.integers(1, 72, n)
    support_calls = rng.poisson(1.0, n)
    fee = np.round(rng.uniform(8, 60, n), 2)
    region = rng.choice(["north", "south", "east", "west"], n)
    leave = 1 / (1 + np.exp(-(0.4 - 0.05 * tenure + 0.15 * support_calls)))  # new customers leave most, contact or not
    email_lift = np.where(app == 1, 0.06, 0.005)
    call_lift = np.where((age >= 50) & (support_calls >= 2), 0.20, 0.03)
    arm = rng.choice(["none", "email", "call"], n, p=[0.4, 0.3, 0.3])
    lift = np.where(arm == "email", email_lift, np.where(arm == "call", call_lift, 0.0))
    stayed = (rng.uniform(size=n) < np.clip(1 - leave + lift, 0, 1)).astype(int)
    frame = pd.DataFrame({"customer_id": [f"R{seed:02d}{i:06d}" for i in range(n)], "age": age, "app_user": app, "tenure_months": tenure,
                          "support_calls": support_calls, "monthly_fee": fee, "region": region, "contact": arm, "stayed": stayed})
    return Replica(frame, {"treatment": "contact", "arms": ["none", *ACTIONS], "control": "none", "outcome": "stayed",
                           "value_column": "monthly_fee", "value_per_month_kept": 12, "costs": COSTS, "identifier": "customer_id",
                           "randomised": True, "true_lift": {"email": email_lift.tolist(), "call": call_lift.tolist()},
                           "segments": {"email": "app users", "call": "aged 50 or more with 2 or more support calls"}, "synthetic": True})
