"""Columnar files for the sets (package 23.10): a set's rows as Parquet beside the JSONL, when pyarrow is installed.

A row is a dictionary with scalars and nested values (lists, dictionaries); a column file wants one type per column. Scalars
stay columns; every nested value becomes a JSON string in its column, so the file reads back with ``pandas.read_parquet`` and
``json.loads`` on the few nested columns. Nothing here changes a row: the JSONL stays the source of truth.

    pip install "dclab-core[parquet]"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

SCALARS = (str, int, float, bool, type(None))


def available() -> bool:
    try:
        import pyarrow  # noqa: F401  (the optional extra; imported here only)
    except ImportError:
        return False
    return True


def frame_of(rows: Iterable[dict[str, Any]]) -> Any:
    """A pandas frame of the rows: scalars as they are, anything nested as JSON text."""
    import pandas as pd

    flat = [{k: (v if isinstance(v, SCALARS) else json.dumps(v, ensure_ascii=False, default=str, sort_keys=True)) for k, v in row.items()} for row in rows]
    return pd.DataFrame(flat)


def write_parquet(path: Path, rows: Iterable[dict[str, Any]]) -> bool:
    """Write the rows as a Parquet file; False (and no file) when pyarrow is not installed or there are no rows."""
    rows = list(rows)
    if not rows or not available():
        Path(path).unlink(missing_ok=True)
        return False
    frame_of(rows).to_parquet(path, index=False)
    return True


__all__ = ["available", "frame_of", "write_parquet"]
