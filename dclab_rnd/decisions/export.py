"""The export: every sample with its labels, as the training-lake sample record, split by dataset.

    python -m dclab_rnd.decisions export --home agent_runs --out out/decisions

writes ``samples.jsonl`` (one line per sample, with ``labels``: the label rows that point at it, by id or by key, in
time order, and ``label``: the latest human label or null) and ``manifest.json`` (counts per point, per label value,
per split; the split rule; the builder's commit). A dataset (by its content digest) is in exactly one split: ``sealed``
for about one in five by hash, ``val`` for another fifth, ``train`` for the rest, so no column of a table is seen in
two splits. Project ids are masked (a hash) and user ids dropped: what a model needs is the evidence and the answer.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .store import SampleStore, open_samples

MASK_CHARS = 12


def _mask(value: str | None) -> str | None:
    return hashlib.sha256(str(value).encode()).hexdigest()[:MASK_CHARS] if value else None


def split_of(dataset_ref: str | None, project_id: str | None, draft_id: str | None = None) -> str:
    """By the dataset's digest (the project or draft id when there is no table yet): 0–1 sealed, 2–3 val, 4–9 train."""
    key = dataset_ref or project_id or draft_id or "none"
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10
    return "sealed" if bucket < 2 else "val" if bucket < 4 else "train"


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, check=False).stdout.strip() or None
    except OSError:
        return None


def join(samples: list[dict[str, Any]], labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each sample with the labels that point at it: by ``sample_id`` when the label has one, otherwise the latest
    sample with the label's key in the same project that was written before the label (or the earliest after it,
    for a label written before its sample)."""
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_key: dict[tuple[str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_key[(s.get("project_id"), s.get("key"))].append(s)
    for s_list in by_key.values():
        s_list.sort(key=lambda s: str(s.get("at") or ""))
    for lab in sorted(labels, key=lambda r: str(r.get("at") or "")):
        if lab.get("sample_id"):
            by_id[lab["sample_id"]].append(lab)
            continue
        candidates = by_key.get((lab.get("project_id"), lab.get("key"))) or []
        before = [s for s in candidates if str(s.get("at") or "") <= str(lab.get("at") or "")]
        target = before[-1] if before else (candidates[0] if candidates else None)
        if target is not None:
            by_id[target["sample_id"]].append(lab)
    out = []
    for s in samples:
        attached = by_id.get(s["sample_id"], [])
        human = [r for r in attached if str(r.get("source", "")).startswith("human")]
        out.append({**s, "labels": attached, "label": human[-1]["label"] if human else None,
                    "label_source": human[-1]["source"] if human else None})
    return out


def export(home: Path, out: Path, store: SampleStore | None = None) -> dict[str, Any]:
    store = store or open_samples(Path(home))
    rows = join(store.samples(), store.labels())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Any] = {"samples": len(rows), "labelled": 0, "by_point": Counter(), "by_split": Counter(), "by_label": defaultdict(Counter),
                              "agreement": defaultdict(lambda: {"agree": 0, "differ": 0})}
    with (out / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for r in rows:
            split = split_of(r.get("dataset_ref"), r.get("project_id"), r.get("draft_id"))
            masked = {**r, "project_id": _mask(r.get("project_id")), "draft_id": _mask(r.get("draft_id")), "user_id": None, "split": split,
                      "labels": [{**lab, "project_id": _mask(lab.get("project_id")), "user_id": None, "who": None} for lab in r["labels"]]}
            handle.write(json.dumps(masked, ensure_ascii=False, default=str) + "\n")
            counts["by_point"][r["point"]] += 1
            counts["by_split"][split] += 1
            if r["label"] is not None:
                counts["labelled"] += 1
                counts["by_label"][r["point"]][json.dumps(r["label"], sort_keys=True, default=str)] += 1
                counts["agreement"][r["point"]]["agree" if r["label"] == r.get("rule_answer") else "differ"] += 1
    manifest = {"samples": counts["samples"], "labelled": counts["labelled"], "by_point": dict(counts["by_point"]), "by_split": dict(counts["by_split"]),
                "by_label": {k: dict(v) for k, v in counts["by_label"].items()}, "rule_agreement": {k: dict(v) for k, v in counts["agreement"].items()},
                "split_rule": "by dataset digest: 20% sealed, 20% val, 60% train; the sealed split is scored once per evaluation run",
                "masking": "project and draft ids hashed; user ids and names dropped", "builder_commit": _commit(), "schema_version": 1}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
