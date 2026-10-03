"""Task-type-aware expansion campaign (``expansion_v1``).

The original ``model_building_50_v1`` campaign covers balanced-ish binary
classification only.  This package extends the same evidence discipline
(training-only selection, one locked holdout, claims that cite evidence) to
imbalanced fraud detection, multiclass classification, time-series regression,
and text+tabular classification.

Importing this package never touches the network; downloads happen only when a
dataset loader is called and the pinned raw file is missing.
"""

CAMPAIGN_ID = "expansion_v1"

__all__ = ["CAMPAIGN_ID"]
