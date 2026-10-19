"""Tier 3, the training sets (32.11 to 32.13): one folder per model type, each set with its train, dev and sealed splits,
its ``MANIFEST.json`` (:class:`dclab_core.store.SetManifest`) and its ``CARD.md``.

``write_set`` is the one writer: the split is each example's own (from its record's sealed key, one rule for every set,
``dclab_core.datasets.split_of``); a set's sealed partition is written once and never rewritten (rows that arrive sealed
later wait in ``sealed.pending.jsonl`` for the next version); every example parses as its contract before it is written.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable, Optional

from dclab_core.store import EXAMPLE_SHAPES, SetManifest

from ..layout import Layout
from ..store import now, write_json

SPLIT_RULE = "dclab_core.datasets.split_of(the record's sealed key), salt dclab-datasets-v1: sealed below 0.2, dev below 0.4, else train"


def write_set(layout: Layout, model_type: str, folder: str, examples: Iterable[Any], *, version: str = "v1", sources: Iterable[dict[str, Any]] = (),
              about: str = "", decontaminated: Iterable[str] = ("the store's own sealed partition",), commit: Optional[str] = None) -> SetManifest:
    """Write one set; returns its manifest."""
    examples = list(examples)
    shape = examples[0].schema_version if examples else f"dclab_core.{EXAMPLE_SHAPES[model_type][0]}.v1"
    base = layout.sets(model_type, *folder.split("/"), version)
    base.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[str]] = {"train": [], "dev": [], "sealed": []}
    for ex in sorted(examples, key=lambda e: e.id):
        if ex.model_type != model_type:
            raise ValueError(f"{ex.id} is an example of {ex.model_type}, not {model_type}")
        by_split[ex.split].append(ex.model_dump_json() + "\n")
    for split in ("train", "dev"):
        (base / f"{split}.jsonl").write_text("".join(by_split[split]), encoding="utf-8")
    sealed, pending = base / "sealed.jsonl", base / "sealed.pending.jsonl"
    if not sealed.exists():
        sealed.write_text("".join(by_split["sealed"]), encoding="utf-8")  # once
        waiting = 0
    else:
        have = set(sealed.read_text(encoding="utf-8").splitlines())
        later = [line for line in by_split["sealed"] if line.rstrip("\n") not in have]
        pending.write_text("".join(later), encoding="utf-8")
        waiting = len(later)
    labels = Counter(getattr(ex, "label_source", None) or getattr(ex, "decided_by", None) or getattr(ex, "evidence", None) or "none" for ex in examples)
    weights = {}
    for ex in examples:
        key = getattr(ex, "label_source", None) or "none"
        weights.setdefault(key, ex.weight)
    manifest = SetManifest(model_type=model_type, name=folder, version=version, shape=shape, sources=sorted(sources, key=lambda s: str(s.get("entry"))),
                           split_rule=SPLIT_RULE, counts={k: len(v) for k, v in by_split.items()} | {"sealed_pending": waiting},
                           label_sources=dict(labels), weights=weights, decontaminated=list(decontaminated), commit=commit, built_at=now())
    write_json(base / "MANIFEST.json", json.loads(manifest.model_dump_json()))
    (base / "CARD.md").write_text(card(manifest, about), encoding="utf-8")
    return manifest


def card(m: SetManifest, about: str) -> str:
    counts = ", ".join(f"{k} {v}" for k, v in m.counts.items())
    labels = ", ".join(f"{k} {v}" for k, v in sorted(m.label_sources.items())) or "none"
    return (f"# {m.model_type} / {m.name} ({m.version})\n\n{about}\n\n- Shape: `{m.shape}`\n- Rows: {counts}\n- Labels by source: {labels}\n"
            f"- Split: {m.split_rule}\n- Decontaminated against: {', '.join(m.decontaminated)}\n- Built: {m.built_at}\n\n"
            "The sealed split is written once and is never read to tune anything. A teacher's or a rule's label is silver: its weight is "
            "below a confirmed label's, and it is kept apart by its source.\n")


__all__ = ["write_set", "card", "SPLIT_RULE"]
