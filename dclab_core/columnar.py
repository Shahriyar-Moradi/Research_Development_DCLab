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


BIG = 2**62


def _kind(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and abs(value) >= BIG:
        return "json"  # beyond what a 64-bit column holds
    if isinstance(value, (int, float)):
        return "num"
    return "str" if isinstance(value, str) else "json"


def frame_of(rows: Iterable[dict[str, Any]]) -> Any:
    """A pandas frame of the rows: a column of one scalar kind as it is; a column that is nested, or mixes kinds (a string in one
    row, a boolean or a number in another), as JSON text in every row, so the file always writes and reads back."""
    import pandas as pd

    rows = [dict(r) for r in rows]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    as_json = {k for k in keys if len({kind for kind in (_kind(r.get(k)) for r in rows) if kind != "none"}) > 1
               or any(_kind(r.get(k)) == "json" for r in rows)}
    return pd.DataFrame([{k: (json.dumps(r[k], ensure_ascii=False, default=str, sort_keys=True) if k in as_json and r.get(k) is not None else r.get(k)) for k in keys}
                         for r in rows])


def write_parquet(path: Path, rows: Iterable[dict[str, Any]]) -> bool:
    """Write the rows as a Parquet file; False (and no file) when pyarrow is not installed, there are no rows, or the file cannot be
    written. An older file is removed first: a file that holds rows the JSONL no longer holds must not outlive them."""
    Path(path).unlink(missing_ok=True)
    rows = list(rows)
    if not rows or not available():
        return False
    try:
        frame_of(rows).to_parquet(path, index=False)
    except Exception:  # noqa: BLE001 - the columns are a convenience: the JSONL stands, and the build goes on
        Path(path).unlink(missing_ok=True)
        return False
    return True


__all__ = ["available", "frame_of", "write_parquet"]
