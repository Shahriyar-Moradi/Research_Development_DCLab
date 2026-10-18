"""Truth by measured outcome (package 29.3): a label for a choice among measured options, from the numbers alone.

A rule chooses among options it measured on validation (a feature recipe, a model family): the simplest within a
tolerance of the best, the best mean less a share of its spread. Whether it chose right is a fact only where validation
can tell the options apart: one option ahead of every other by more than their combined fold standard deviation (both
options' spreads together, ``sqrt(sd_a² + sd_b²)``: deliberately wider than a paired bound, since only the means and
spreads are kept). Then that option is the label: right for the rule when it chose it, wrong when it chose another. Where
validation cannot tell them apart nothing is labelled: the highest mean inside the folds' noise is no truth, and a label
that copies it would teach a student to copy noise (the review of 29.3).

An option the rule may not choose (``eligible: false``: a baseline the others are judged against, a model scored on
fewer, easier rows) is never the label. The metric's direction is the host's (an error such as MAE, MASE or a Brier score
is better when lower); unknown, nothing is labelled.

Nothing here reads a holdout: a tuning's worth cannot be judged by its holdout without scoring the default on that same
read, and that is the owner's call under the holdout-once rule, not this module's.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

KIND = "validation_clear_best"


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def clear_best(options: Sequence[Mapping[str, Any]], higher_is_better: Optional[bool]) -> Optional[tuple[str, dict[str, Any]]]:
    """(the option clearly best on validation, evidence) or None. ``options``: each with ``id``, ``mean``, ``std`` and
    optionally ``eligible``. None when the direction is unknown, fewer than two eligible options carry a mean and a spread,
    a spread is missing or both are zero, or the leader is not ahead of every other by their combined spread."""
    if not isinstance(higher_is_better, bool):
        return None
    scored = []
    for o in options or ():
        if not isinstance(o, Mapping) or o.get("id") is None or o.get("eligible") is False:
            continue
        mean, std = _number(o.get("mean")), _number(o.get("std"))
        if mean is None:
            continue
        if std is None or std < 0:
            return None  # a spread unknown: the noise cannot be measured, so neither can the best
        scored.append((str(o["id"]), mean, std))
    if len(scored) < 2:
        return None
    sign = 1.0 if higher_is_better else -1.0
    scored.sort(key=lambda x: sign * x[1], reverse=True)
    best, mean, std = scored[0]
    margins = []
    for _, other_mean, other_std in scored[1:]:
        noise = math.sqrt(std ** 2 + other_std ** 2)
        gap = sign * (mean - other_mean)
        if noise <= 0 or gap <= noise:
            return None
        margins.append(gap)
    return best, {"kind": KIND, "value": round(min(margins), 6), "ref": f"decision.options.{best}"[:200]}


__all__ = ["KIND", "clear_best"]
