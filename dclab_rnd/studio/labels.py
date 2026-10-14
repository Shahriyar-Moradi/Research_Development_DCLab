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

import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

CASE_WORDS = re.compile(r"(reason|type|kind|fraud|case|abuse|anomal|scam|fake)", re.I)
FILE_WORDS = re.compile(r"(fraud|label|case|known|positive|confirm|abuse|anomal|scam|fake|flagged)", re.I)
BOOLEAN_VALUES = {"0", "1", "0.0", "1.0", "true", "false", "yes", "no", "y", "n", "t", "f"}
STATUS_WORDS = re.compile(r"(status|state|result|outcome|flag|stage)", re.I)
INFLATION = 1.5  # a key that marks more than this many table rows per label row is refused (multi-drop orders reach ~1.1)


def _repeats(table: pd.DataFrame, labels: pd.DataFrame, key: list[str], candidates: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Keys that repeat in the table, found before the join (21.3), on the coarsest key that finds as many cases as the
    chosen one: how many of its values sit on several rows, how many of those carry a label, and how they are
    resolved: by the chosen key itself when it is finer (only the labelled rows are marked), or every row of a labelled
    key marked a case."""
    chosen = next((c for c in candidates if c["key"] == key), None)
    if chosen is None:
        return None

    def repeats(cols: list[str]) -> pd.Series:
        if not all(c in table.columns for c in cols):
            return pd.Series(dtype=int)
        counts = table[cols].dropna().astype(str).value_counts()
        return counts[counts > 1]

    coarse = [c for c in candidates if c["verdict"] != "refused" and c["label_rows_found"] >= chosen["label_rows_found"] and len(repeats(c["key"]))]
    coarsest = max(coarse, key=lambda c: c["rows_per_label"], default=None)
    if coarsest is None:
        return None
    cols = coarsest["key"]
    repeated = repeats(cols)
    labelled = labels[cols].dropna().astype(str).drop_duplicates()
    marked = int(labelled.merge(repeated.rename("n").reset_index(), on=cols).shape[0]) if len(labelled) else 0
    finer = key if cols != key else next((c["key"] for c in candidates if c["key"] != key and c["verdict"] != "refused"
                                          and c["label_rows_found"] >= chosen["label_rows_found"] and not len(repeats(c["key"]))), None)
    resolution = (f"joined on {'+'.join(key)}, which tells the rows of a repeated {'+'.join(cols)} apart: only the labelled rows are marked"
                  if cols != key else "every row of a labelled key is marked a case"
                  + (f"; {'+'.join(finer)} would tell those rows apart" if finer else ""))
    return {"key": cols, "keys_repeated": int(len(repeated)), "rows_in_repeats": int(repeated.sum()), "labelled_keys_repeated": marked,
            "resolution": resolution, "finer_key": finer}


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
    repeats = _repeats(table, labels, chosen["key"], candidates) if chosen else None
    return {"key": chosen["key"] if chosen else None, "rows_marked": chosen["rows_marked"] if chosen else 0, "repeats": repeats,
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


def cases_table(profile: dict[str, Any], filename: str = "") -> Optional[dict[str, Any]]:
    """Whether a table is a list of known cases rather than the table to learn from: a key different on every row and
    named as a key, a column naming the kind of case (a reason, a type), and every row in the same state, or a file name
    that says so. Read from the profile only (names and counts). Such a file is the labels of another table; alone, it
    holds no row that is not a case, so there is nothing to tell the cases from."""
    columns = profile.get("columns") or []
    keys = [c["name"] for c in columns if c.get("id_like") and c.get("name_id_like")]
    kinds = {c["name"]: c.get("preview", [])[:5] for c in columns
             if c.get("kind") == "categorical" and 2 <= (c.get("unique") or 0) <= 20 and CASE_WORDS.search(str(c["name"]))}
    same = {c["name"]: (c.get("preview") or [None])[0] for c in columns if (c.get("unique") or 0) == 1 and STATUS_WORDS.search(str(c["name"]))}
    named = bool(FILE_WORDS.search(Path(str(filename)).stem))
    # a two-valued column besides the kinds of case that reads as an outcome (chargeback 0/1, is_fraud yes/no) is the
    # table's own outcome: a table to learn from; a two-valued attribute (box or no box) is not
    outcome = [c["name"] for c in columns if (c.get("unique") or 0) == 2 and c["name"] not in kinds
               and (c.get("target_name_like") or {str(v).strip().lower() for v in c.get("preview") or []} <= BOOLEAN_VALUES)]
    if outcome or not keys or not kinds or not same:  # every row in one state: the file's name alone is not enough
        return None
    return {"keys": keys, "case_kinds": kinds, "same_on_every_row": same, "file_name_says_so": named, "rows": int(profile.get("rows") or 0)}


MIN_COVERAGE = 0.5  # a table holding fewer of the cases' keys shares a column, not the cases (21.3)
LABEL_SHARE = 0.5   # as the draft pipeline: a file of cases is at most half the rows of the table it labels


def joinable(cases: pd.DataFrame, table_path: Any) -> bool:
    """Whether the draft pipeline would read ``cases`` as the label file of the table stored at ``table_path``: at most
    half its rows, and a column of its own (a sample of the same table is not a file of cases)."""
    import pyarrow.parquet as pq

    meta = pq.read_metadata(table_path)
    columns = set(pq.read_schema(table_path).names)
    return len(cases) <= LABEL_SHARE * meta.num_rows and bool(set(map(str, cases.columns)) - columns)


def tables_it_labels(drafts: Any, draft_id: str, cases: pd.DataFrame, keys: list[str], limit: int = 3) -> list[dict[str, Any]]:
    """Tables in the workspace's other drafts that a file of cases could label: each ready table holding one of its key
    columns and at least half of the cases' keys, with how many and what share (its coverage). Only the key column is
    read."""
    try:
        return _tables_it_labels(drafts, draft_id, cases, keys, limit)
    except Exception:  # noqa: BLE001 - a hint only: the upload of the cases never fails for it
        return []


def _tables_it_labels(drafts: Any, draft_id: str, cases: pd.DataFrame, keys: list[str], limit: int) -> list[dict[str, Any]]:
    found = []
    for other in drafts.list(limit=50):
        if other.get("id") == draft_id:
            continue
        try:
            full = drafts.get(other["id"])
        except KeyError:
            continue
        # the draft's active table only: every table of a draft is cleaned into the same file, the newest one
        for asset in [a for a in full.get("assets") or [] if a.get("id") == full.get("active_asset")]:
            if asset.get("status") != "ready" or asset.get("kind") == "labels" or not asset.get("clean_file"):
                continue
            path = drafts.data_dir(other["id"]) / asset["clean_file"]
            if not joinable(cases, path):
                continue
            for key in keys:
                try:
                    values = pd.read_parquet(path, columns=[key])[key]
                except Exception:  # noqa: BLE001 - a table without this column is not a match
                    continue
                wanted = cases[key].dropna().astype(str)
                matched = int(wanted.isin(set(values.dropna().astype(str))).sum())
                coverage = matched / max(len(wanted), 1)
                if matched and coverage >= MIN_COVERAGE:
                    found.append({"draft_id": other["id"], "asset": asset.get("name"), "rows": int(len(values)), "key": key,
                                  "matched": matched, "of": int(len(wanted)), "coverage": round(coverage, 3),
                                  "key_repeats": int(values.dropna().duplicated().sum())})
    return sorted(found, key=lambda f: -f["matched"] / max(f["of"], 1))[:limit]
