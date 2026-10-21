"""Text and times read the same way on pandas 2 and pandas 3.

pandas 3 stores text in its string type, not ``object``, so ``dtype == object`` no longer finds text columns; and its
per-value ("mixed") time parsing reads a time without a year, such as syslog's "Oct  1 14:32:00", as year 1 where
pandas 2 left it missing. A value read before the year 1000 is taken as one whose year was not written.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

FIRST_YEAR = 1000  # before it, the year was filled in by the parser, not read from the text


def is_text(values: Any) -> bool:
    """A column (or a dtype) of text: ``object`` on pandas 2, the string type on pandas 3 (or when asked for)."""
    dtype = getattr(values, "dtype", values)
    return dtype == object or isinstance(dtype, pd.StringDtype)


def without_yearless(parsed: Any) -> Any:
    """Parsed times with any value before ``FIRST_YEAR`` made missing (a Series or a DatetimeIndex; others unchanged)."""
    if isinstance(parsed, pd.Series) and pd.api.types.is_datetime64_any_dtype(parsed):
        return parsed.mask(parsed.dt.year < FIRST_YEAR)
    if isinstance(parsed, pd.DatetimeIndex):
        return parsed.where(parsed.year >= FIRST_YEAR)
    return parsed


def read_times(values: Any, **kwargs: Any) -> Any:
    """``pd.to_datetime(values, errors="coerce", format="mixed")``, with values whose year was not written left missing."""
    return without_yearless(pd.to_datetime(values, errors="coerce", format="mixed", **kwargs))


__all__ = ["FIRST_YEAR", "is_text", "read_times", "without_yearless"]
