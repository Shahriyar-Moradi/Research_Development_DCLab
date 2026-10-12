"""A label file next to the table (package 15.2): which key joins them, checked by code, and the target it makes.

Labels often arrive as a second file: a sample of known cases, keyed by an order, a customer or a pair of ids. The
wrong key is the classic way to inflate them (a (customer, courier) pair marks every ride the two ever shared, not the
fraudulent one). :func:`review` measures every candidate key with :func:`dclab_core.analyse.label_join` and chooses the
one that finds the most label rows while marking about one table row per label; a key that marks more is refused, with
its numbers, unless the person says the label is per entity. Label rows not found in the table are counted and kept
apart, and columns with the same name that disagree between the files are named: they are never taken from the label
file, and they are stated as a limit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

INFLATION = 1.5  # a key that marks more than this many table rows per label row is refused (multi-drop orders reach ~1.1)


def review(table: pd.DataFrame, labels: pd.DataFrame, key: Optional[list[str]] = None, per_entity: bool = False) -> dict[str, Any]:
    """Every candidate key with its numbers and a verdict, the chosen key, and the columns that disagree."""
    from dclab_core.analyse import _disagreeing, label_join

    joins = label_join(table, labels)
    candidates = []
    for j in joins:
        cols = j["key"].split("+")
        inflating = j["rows_per_label"] > INFLATION and not per_entity
        verdict = "refused" if inflating or j["label_rows_found"] == 0 else "possible"
        reason = (f"marks {j['rows_marked']} rows from {j['label_rows_found']} labels ({j['rows_per_label']} per label): every row that "
                  "shares the value would be called a case" if inflating else
                  ("finds no label row in the table" if j["label_rows_found"] == 0 else "about one row per label"))
        candidates.append({"key": cols, "rows_marked": j["rows_marked"], "label_rows_found": j["label_rows_found"],
                           "rows_per_label": j["rows_per_label"], "verdict": verdict, "reason": reason})
    chosen = None
    if key is not None:
        chosen = next((c for c in candidates if c["key"] == list(key)), None)
        if chosen is None:
            raise ValueError(f"{'+'.join(key)} is not a column both files share")
        if chosen["verdict"] == "refused":
            raise ValueError(f"Refused: joining on {'+'.join(key)} {chosen['reason']}")
    else:
        possible = [c for c in candidates if c["verdict"] == "possible"]
        chosen = max(possible, key=lambda c: (c["label_rows_found"], -len(c["key"]), -c["rows_per_label"]), default=None)
    if chosen is not None:
        chosen["verdict"] = "chosen"
    label_rows = int(len(labels))
    found = chosen["label_rows_found"] if chosen else 0
    return {"key": chosen["key"] if chosen else None, "rows_marked": chosen["rows_marked"] if chosen else 0,
            "label_rows": label_rows, "label_rows_found": found, "not_found": label_rows - found if chosen else label_rows,
            "candidates": candidates, "per_entity": per_entity,
            "disagreeing_columns": _disagreeing(table, labels, chosen["key"][0]) if chosen and len(chosen["key"]) == 1 else {}}


def mark(table: pd.DataFrame, labels: pd.DataFrame, key: list[str]) -> pd.Series:
    """1 for the table rows whose key is in the label file, else 0 (unlabelled: a negative only when labels are complete)."""
    from dclab_core.analyse import _key_text

    wanted = labels[key].dropna().apply(_key_text)
    have = table[key].apply(_key_text)
    if len(key) == 1:
        return have[key[0]].isin(set(wanted[key[0]])).astype(int)
    index = pd.MultiIndex.from_frame(have)
    return pd.Series(index.isin(pd.MultiIndex.from_frame(wanted)).astype(int), index=table.index)


def path(store, project: dict[str, Any]) -> Path:
    """The project's label file, checked against the SHA-256 recorded when it was attached."""
    from ..storage.files import FileError, files_for
    from .data import DataError

    info = project["labels"]
    target = store.data_dir(project["id"]) / info["filename"]
    try:
        files = files_for(store)
        return files.path(files.key_of(target), expected=info.get("sha256"))
    except FileError as exc:
        raise DataError(str(exc)) from None


def apply(store, project: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """The table with the label file's target column (``project["labels"]["target"]``), joined on the recorded key."""
    from .data import load_table

    info = project["labels"]
    labels = load_table(path(store, project))
    out = frame.copy()
    out[info["target"]] = mark(frame, labels, list(info["key"]))
    return out
