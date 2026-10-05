"""Which numeric columns are category codes rather than quantities (rule DCLAB-R11).

A factorized category is a number without order or scale: in the cached UCI tables
``month`` is May=0, Jun=1, ... in order of first appearance. Its log, its ratio to
another column or its product with one means nothing, so feature recipes keep such
columns out of every arithmetic transform. The columns themselves stay in the matrix: the
DCLab notebook one-hot encodes them inside each training fold, while the playbook
``FeatureEngineer`` passes them as numbers unless ``one_hot_codes=True``.

Declared metadata wins. The R&D dataset catalog (``dclab_rnd/agentic/catalog.py``)
lists the categorical columns of every repository dataset, and the agentic worker
already refuses arithmetic on them. Only a table without a declaration falls back to
a conservative heuristic: whole-number columns with few distinct values are treated
as codes. On the ten cached UCI tables it finds 74 of the 77 declared columns. It
misses ``native-country`` and ``day_of_week``, which have more levels, and mushroom's
constant ``veil-type``. It also flags six small integer counts, which then only lose
derived features.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

MAX_CODE_LEVELS = 20  # same cut as the notebook profile's ``low_cardinality``
HEURISTIC_RULE = f"no declaration for this table, so whole numbers with at most {MAX_CODE_LEVELS} distinct values count as codes"


def declared_categorical(key: str, columns: Iterable[str]) -> list[str] | None:
    """The catalog's categorical columns of dataset ``key`` among ``columns``; None when the catalog has no entry."""
    from dclab_rnd.agentic.catalog import DATASETS  # stdlib-only module

    policy = DATASETS.get(key)
    if policy is None:
        return None
    declared = policy["categorical"]
    return [str(c) for c in columns if declared == "ALL" or c in declared]


def looks_like_codes(series: pd.Series, max_levels: int = MAX_CODE_LEVELS) -> bool:
    """A numeric column whose values are whole numbers with 2..``max_levels`` distinct values."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    levels = np.asarray(series.dropna().unique(), dtype=float)
    return 2 <= len(levels) <= max_levels and bool(np.all(np.mod(levels, 1.0) == 0))


def category_code_columns(frame: pd.DataFrame, declared: Sequence[str] | None = None) -> list[str]:
    """Numeric columns of ``frame`` that hold category codes.

    With a declaration (possibly empty) only the declared columns count; without one the
    heuristic decides. Non-numeric columns are categorical by dtype already and are not listed.
    """
    if declared is not None:
        names = set(declared)
        return [c for c in frame.columns if c in names and pd.api.types.is_numeric_dtype(frame[c])]
    return [c for c in frame.columns if looks_like_codes(frame[c])]
