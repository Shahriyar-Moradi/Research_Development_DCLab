"""Public tables that are copies of each other (the review of S8 A1, 2026-10-25): ``python -m dclab_rnd.table_twins``
writes ``dclab_core/lake_twins.json``, which ``dclab_core.lake.split_of`` reads.

OpenML holds one table under several ids (``glass`` as 41 and ``prnn_fglass`` as 952; ``iris`` four times), or one
table's columns with another outcome (``thyroid-allbp`` and ``thyroid-allhyper``), and the lake's hash splits the
copies apart: a copy of a sealed table sat in training. Two tables are twins when they have the same number of rows,
the same column names (lower case, any order) and the same OpenML qualities (class sizes, missing values, feature
counts). That also groups some synthetic variants that share a schema (``fri_c*``): holding them is cautious, and for
the column students (roles, identifiers, leaks) they are the same homework.

- ``held``: a train table with a twin in dev or sealed. ``lake.split_of`` gives it dev: never trained on, never
  sealed (a dev table was read). No dev or sealed table moves.
- ``sealed_with_twins``: a sealed table with a twin outside sealed. A model trained before this file existed may have
  trained on its twin, so an exam must skip it to stay blind.

Built from the lake's and the collector's derived records (``real_shadow.DERIVED_DIRS``) and the OpenML list's
qualities (``data/downloads/openml/list.json``); counts and ids only, never a value. ``--check`` says whether the
committed file is what the records give.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dclab_core" / "lake_twins.json"
LIST = ROOT / "data" / "downloads" / "openml" / "list.json"
VERSION = "lake_twins_v1"


def _qualities(path: Path = LIST) -> dict[int, tuple]:
    if not path.is_file():
        return {}
    out = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        if isinstance(item, dict) and item.get("did"):
            out[int(item["did"])] = tuple(sorted((str(q.get("name")), str(q.get("value"))) for q in item.get("quality") or []))
    return out


def _derived(dirs: Optional[list[Path]] = None) -> list[dict[str, Any]]:
    from . import real_shadow

    out, seen = [], set()
    for folder in dirs or list(real_shadow.DERIVED_DIRS):
        for path in sorted(Path(folder).glob("*.json")) if Path(folder).is_dir() else []:
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if d.get("failed") or not d.get("columns") or str(d.get("did")) in seen:
                continue
            seen.add(str(d.get("did")))
            out.append(d)
    return out


def build(dirs: Optional[list[Path]] = None, qualities: Optional[dict[int, tuple]] = None) -> dict[str, Any]:
    """The twin groups that span splits, the train tables held as dev, the sealed tables with a twin outside sealed."""
    from dclab_core.lake import base_split_of

    q = _qualities() if qualities is None else qualities
    records = _derived(dirs)
    groups: dict[tuple, set[int]] = defaultdict(set)
    for d in records:
        did = int(d["did"])
        groups[(d.get("rows"), tuple(sorted(str(c).strip().lower() for c in d["columns"])), q.get(did, ()))].add(did)
    held: dict[str, list[str]] = {}
    sealed: dict[str, list[str]] = {}
    spans = []
    for members in groups.values():
        if len(members) < 2:
            continue
        split = {d: base_split_of(f"openml:{d}") for d in members}
        if len(set(split.values())) < 2:
            continue
        spans.append(sorted(members))
        for d in sorted(members):
            others = [f"openml:{x}" for x in sorted(members) if x != d and split[x] != split[d]]
            if split[d] == "train" and any(split[x] in ("dev", "sealed") for x in members):
                held[f"openml:{d}"] = [f"openml:{x}" for x in sorted(members) if split[x] in ("dev", "sealed")]
            elif split[d] == "sealed" and others:
                sealed[f"openml:{d}"] = others
    return {"version": VERSION, "signature": "the same number of rows, column names (lower case, any order) and OpenML qualities",
            "rule": "a held train table is split dev (never trained on); a sealed table with a twin outside sealed is skipped by exams",
            "tables_read": len(records), "groups_across_splits": len(spans),
            "held": dict(sorted(held.items(), key=lambda kv: int(kv[0].split(":")[1]))),
            "sealed_with_twins": dict(sorted(sealed.items(), key=lambda kv: int(kv[0].split(":")[1])))}


def write(doc: dict[str, Any], out: Path = OUT) -> bool:
    text = json.dumps(doc, indent=1) + "\n"
    changed = not out.is_file() or out.read_text(encoding="utf-8") != text
    if changed:
        out.write_text(text, encoding="utf-8")
    return changed


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dclab_rnd.table_twins", description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if not doc["tables_read"]:
        print("no derived records here (data/downloads/openml): nothing written")
        return 0
    if args.check:
        fresh = OUT.is_file() and OUT.read_text(encoding="utf-8") == json.dumps(doc, indent=1) + "\n"
        print("lake twins are fresh" if fresh else f"lake twins are stale: run python -m dclab_rnd.table_twins")
        return 0 if fresh else 1
    write(doc)
    print(f"{len(doc['held'])} train tables held as dev, {len(doc['sealed_with_twins'])} sealed tables with twins, "
          f"from {doc['tables_read']} tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
