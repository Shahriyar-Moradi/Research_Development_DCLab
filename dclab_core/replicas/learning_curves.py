"""Two invented tables with a known learning curve (package 34.13): one whose score still rises with more rows, one
already flat.

* ``rising``: the outcome depends a little on each of many columns (100 numeric signals, every weight small, on 1200 rows). A model
  needs many rows to tell each small weight from noise, so its score is still climbing when it has every training row:
  more rows of the same kind help.
* ``flat``: the outcome depends on one strong column, and a third of the labels are noise. A few hundred rows teach
  everything there is to learn; the rest add almost nothing: more rows do not help, better columns would.

Each table also has a ``region`` column. In ``rising``, rows of region ``east`` follow the same weights but are rarer
and three in ten of their labels are flipped, so the errors concentrate there (the per-segment curve's case).

The traps a careful data scientist must see:

* **The holdout is not a curve's yardstick.** A curve read on the holdout chooses how much to collect on the rows that
  judge the final model; read it on the training folds only.
* **Subsets must keep the split's shape.** A random quarter of a grouped or timed table puts one customer's rows on both
  sides, so the small subsets look better than they are and the curve looks flat.
* **Two points are not a slope.** A small subset's score is noisy: read the slope with its interval, over repeats.
* **A flat curve is not a finished model.** Noisy labels or a missing kind of column flatten it too: more rows of the
  same kind will not help, but cleaner labels or better columns may.
* **An average hides a segment.** A rare region with noisy labels can still be learning while the whole is flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

Shape = Literal["rising", "flat"]


@dataclass(frozen=True)
class Replica:
    rows: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def generate(shape: Shape = "rising", seed: int = 0, n: int | None = None) -> Replica:
    rng = np.random.default_rng(seed)
    if shape == "rising":
        n = n or 1200
        k = 100
        X = rng.normal(size=(n, k))
        weights = rng.choice([-1, 1], k) * rng.uniform(0.15, 0.3, k)
        region = rng.choice(["north", "south", "east"], n, p=[0.45, 0.4, 0.15])
        logit = X @ weights + rng.normal(scale=0.6, size=n)
        columns = {f"signal_{i:02d}": X[:, i].round(4) for i in range(k)}
    elif shape == "flat":
        n = n or 3000
        strong = rng.normal(size=n)
        region = rng.choice(["north", "south", "east"], n, p=[0.45, 0.4, 0.15])
        logit = 1.6 * strong
        columns = {"strong": strong.round(4), "weak_noise_a": rng.normal(size=n).round(4), "weak_noise_b": rng.normal(size=n).round(4)}
    else:
        raise ValueError(f"unknown shape {shape!r}: rising or flat")
    outcome = rng.uniform(size=n) < 1 / (1 + np.exp(-logit))
    if shape == "flat":
        flip = rng.uniform(size=n) < 1 / 3  # a third of the labels are noise: the curve's ceiling comes early
        outcome = np.where(flip, rng.uniform(size=n) < 0.5, outcome)
    else:
        flip = (region == "east") & (rng.uniform(size=n) < 0.3)  # east's labels are noisier: the errors concentrate there
        outcome = np.where(flip, ~outcome, outcome)
    frame = pd.DataFrame({"row_id": [f"R{seed:02d}{i:06d}" for i in range(n)], **columns, "region": region,
                          "outcome": np.where(outcome, "yes", "no")})
    return Replica(frame, {"target": "outcome", "positive": "yes", "identifier": "row_id", "shape": shape,
                           "expected_answer": "likely_help" if shape == "rising" else "unlikely_help",
                           "concentrated_segment": ("region", "east") if shape == "rising" else None, "synthetic": True})
