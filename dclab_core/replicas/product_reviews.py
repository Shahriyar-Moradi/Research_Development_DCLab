"""Product reviews, invented: whether a review is negative, from its text and the fields that come with it (a text +
tabular replica).

The traps a careful data scientist must see:

* **The signal is in the words.** The structured fields (category, price, the reviewer's past reviews) carry little;
  a model on them alone is weak. Words decide.
* **A reply written after the review.** ``shop_reply`` is the shop's answer, written after the review and its rating:
  an apology gives the outcome away, and it does not exist when the review arrives.
* **An identifier.** ``review_id`` is unique per row.
* **Negative reviews are the minority.** About one in five; accuracy flatters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

GOOD = ("great", "love", "perfect", "comfortable", "fits", "soft", "beautiful", "recommend", "happy", "quality")
BAD = ("broke", "returned", "cheap", "tight", "disappointed", "refund", "torn", "faded", "waste", "itchy")
NEUTRAL = ("dress", "shirt", "colour", "size", "fabric", "ordered", "arrived", "wore", "wash", "price", "the", "and", "it", "was")


@dataclass(frozen=True)
class Replica:
    reviews: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def _text(rng: np.random.Generator, negative: bool) -> str:
    cue = BAD if negative else GOOD
    other = GOOD if negative else BAD
    words = list(rng.choice(NEUTRAL, rng.integers(6, 14))) + list(rng.choice(cue, rng.integers(1, 3)))
    if rng.uniform() < 0.45:  # mixed feelings
        words += list(rng.choice(other, rng.integers(1, 3)))
    rng.shuffle(words)
    return " ".join(words)


def generate(seed: int = 0, n: int = 2500) -> Replica:
    rng = np.random.default_rng(seed)
    category = rng.choice(["dresses", "tops", "trousers", "knitwear"], n)
    price = rng.gamma(3, 15, n).round(2)
    past = rng.poisson(2, n)
    logit = -1.6 + 0.004 * price + 0.15 * (category == "knitwear")
    negative = rng.uniform(size=n) < 1 / (1 + np.exp(-logit))
    reply = np.where(negative, rng.choice(["We are sorry to hear this, please contact us.", "Apologies, a refund is on its way."], n),
                     rng.choice(["Thank you!", "Glad you like it.", ""], n))
    frame = pd.DataFrame({"review_id": [f"R{seed:02d}{i:06d}" for i in range(n)], "category": category, "price": price,
                          "past_reviews": past, "review_text": [_text(rng, bool(v)) for v in negative], "shop_reply": reply,
                          "negative": negative.astype(int)})
    return Replica(frame, {"target": "negative", "text": "review_text", "leak": "shop_reply", "identifier": "review_id",
                           "positive_rate": float(negative.mean()), "synthetic": True})
