"""The state's vocabulary and the banding of numbers (package 19.11): what a decision's state (``dclab_core.ledger``)
and a data card (``dclab_core.datacard``) say about a column, in the same bands. Jev's bands, as the main product's
release defines them, and this studio's own. No dependency but ``re``, so every module can import it first."""

from __future__ import annotations

import re

# Jev's bands, as its release defines them (main product apps/api/app/agents/semantic/releases.py at the recorded commit)
DTYPE_KINDS = ("integer", "float", "boolean", "datetime", "string", "categorical", "other")
RATIO_BANDS = ("none", "under_1pct", "from_1_to_10pct", "from_10_to_50pct", "from_50_to_90pct", "from_90_to_99pct", "over_99pct", "all")
CARDINALITY_BANDS = ("constant", "binary", "low", "medium", "high", "very_high")
VALUE_PATTERNS = ("digits", "mixed", "date_like", "long_text", "short_text", "not_text")
TASKS = ("binary_classification", "multiclass_classification", "regression")
ROLES = ("numeric", "categorical_code", "identifier", "datetime", "free_text", "other")
AVAILABILITY = ("before_prediction", "after_outcome", "unknown")
COUNT_BANDS = ("none", "one", "several")
# this studio's own bands
SIZE_BANDS = ("tiny", "small", "medium", "large", "very_large")            # rows: <200, <2k, <50k, <1M, more
FAMILIES = ("classification", "regression", "rare_event", "anomaly", "forecasting", "time_to_event", "ranking",
            "segmentation", "uplift", "entity_tables", "streams", "text_or_images")
LABEL_SITUATIONS = ("column", "file", "none", "unknown")


def ratio_band(value: float | None) -> str:
    """As the main product's ``releases.ratio_band``."""
    if value is None or value != value or value <= 0:
        return "none"
    if value >= 1:
        return "all"
    for bound, band in ((0.01, "under_1pct"), (0.10, "from_1_to_10pct"), (0.50, "from_10_to_50pct"), (0.90, "from_50_to_90pct"),
                        (0.99, "from_90_to_99pct")):
        if value < bound:
            return band
    return "over_99pct"


def cardinality_band(distinct: int | None) -> str | None:
    if distinct is None:
        return None
    if distinct <= 1:
        return "constant"
    if distinct == 2:
        return "binary"
    return "low" if distinct <= 10 else "medium" if distinct <= 100 else "high" if distinct <= 1000 else "very_high"


def name_tokens(name: str) -> list[str]:
    """As the main product's ``releases.name_tokens``: lower-case tokens of a column's name."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [t.lower() for t in re.split(r"[^0-9A-Za-z]+", spaced) if t and len(t) <= 40][:16]


def rows_band(rows: int | None) -> str | None:
    if rows is None:
        return None
    return "tiny" if rows < 200 else "small" if rows < 2000 else "medium" if rows < 50_000 else "large" if rows < 1_000_000 else "very_large"
