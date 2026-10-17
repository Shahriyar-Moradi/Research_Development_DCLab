"""The pretraining corpus of an in-context tabular model (26.6b), written to the lake: invented tables, never trained here.

``dclab_core.priors`` makes each table from a seed; this writes many of them in shards a training run streams, each
table's split from its seed's hash (``split_of``: train, dev, sealed; the sealed tables measure the model once), and
beside each shard one line per table: its prior, sizes, balance and how far a quick model gets above the majority class
on its own held-out rows (``learnable``; tables without signal are kept and marked, as the prior holds them).

    python -m dclab_rnd.lake priors [--tables 2000] [--start 0] [--shard 250]
    → lake/bronze/priors/priors_v1/<split>/shard-<first seed>.npz (+ .jsonl), lake/bronze/priors/priors_v1/MANIFEST.json

A shard's ``.npz`` holds ``X`` (all its tables' rows, float32, columns padded with NaN to the widest), ``y``, ``offsets``
(where each table's rows start), ``widths`` (each table's real column count) and ``categorical`` (padded the same way).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from dclab_core.lake import split_of

from . import LAKE

RULE = ("Invented tables from random generating processes (dclab_core.priors), for pretraining an in-context tabular model; "
        "no real table, no person's data. A model trained on them is evidence about these priors until it is measured on "
        "real tables (26.6's sealed set of at least 30 real datasets).")


def folder(lake: Path = LAKE) -> Path:
    from dclab_core.priors import VERSION

    return Path(lake) / "bronze" / "priors" / VERSION


def _write(out: Path, split: str, tables: list[Any]) -> str:
    """A shard named by its first table's seed: writing the same seeds again rewrites the same files."""
    width = max(t.X.shape[1] for t in tables)
    X = np.full((sum(len(t.y) for t in tables), width), np.nan, dtype=np.float32)
    cat = np.zeros((len(tables), width), dtype=bool)
    offsets, at = [], 0
    for i, t in enumerate(tables):
        X[at:at + len(t.y), :t.X.shape[1]] = t.X
        cat[i, :t.X.shape[1]] = t.categorical
        offsets.append(at)
        at += len(t.y)
    name = f"shard-{tables[0].meta['seed']:07d}"
    (out / split).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / split / f"{name}.npz", X=X, y=np.concatenate([t.y for t in tables]), offsets=np.array(offsets + [at]),
                        widths=np.array([t.X.shape[1] for t in tables]), categorical=cat)
    (out / split / f"{name}.jsonl").write_text("".join(json.dumps(t.meta) + "\n" for t in tables), encoding="utf-8")
    return f"{split}/{name}"


def build(tables: int = 2000, start: int = 0, shard: int = 250, lake: Path = LAKE, check: bool = True) -> dict[str, Any]:
    from dclab_core import priors

    out = folder(lake)
    kept = _seeds(out)  # a seed already in the corpus is never written twice, whatever the start or the shard size
    pending: dict[str, list[Any]] = {"train": [], "dev": [], "sealed": []}
    written: list[str] = []
    counts: Counter = Counter()
    for seed in range(start, start + tables):
        if seed in kept:
            counts["already kept"] += 1
            continue
        t = priors.sample(seed)
        if check:
            t.meta["learnable"] = priors.learnable(t)
        split = split_of(f"prior:{seed}")
        t.meta["split"] = split
        pending[split].append(t)
        if len(pending[split]) >= shard:
            written.append(_write(out, split, pending[split]))
            pending[split] = []
    for split, rest in pending.items():
        if rest:
            written.append(_write(out, split, rest))
    return manifest(lake, {"added": {"start": start, "tables": tables, "skipped_as_kept": counts["already kept"], "shards": written}})


def _seeds(out: Path) -> dict[int, str]:
    """Every seed in the corpus and the shard holding it; a seed in two shards is an error, never counted twice."""
    seen: dict[int, str] = {}
    for p in sorted(out.rglob("shard-*.jsonl")) if out.is_dir() else []:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seed = int(json.loads(line)["seed"])
                if seed in seen:
                    raise ValueError(f"seed {seed} is in {seen[seed]} and {p.name}: the corpus holds a table twice")
                seen[seed] = p.name
    return seen


def read(path: Path) -> list[dict[str, Any]]:
    """A shard back as its tables: ``X`` (its own columns only), ``y``, ``categorical`` and ``meta``."""
    data = np.load(path)
    metas = [json.loads(x) for x in path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    off, widths = data["offsets"], data["widths"]
    return [{"X": data["X"][off[i]:off[i + 1], :widths[i]], "y": data["y"][off[i]:off[i + 1]], "categorical": data["categorical"][i, :widths[i]],
             "meta": metas[i]} for i in range(len(metas))]


def manifest(lake: Path = LAKE, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    out = folder(lake)
    _seeds(out)  # refuses a corpus holding a table twice
    metas = [json.loads(x) for p in sorted(out.rglob("shard-*.jsonl")) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    learn = [m["learnable"] for m in metas if (m.get("learnable") or {}).get("signal") is not None]  # checked tables only
    doc = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "version": out.name, "tables": len(metas),
           "rows": int(sum(m["rows"] for m in metas)), "by_split": dict(Counter(m["split"] for m in metas)),
           "by_prior": dict(Counter(m["prior"] for m in metas)), "by_classes": dict(sorted(Counter(m["classes"] for m in metas).items())),
           "with_signal": sum(1 for x in learn if x["signal"]), "checked": len(learn),
           "not_checked": sum(1 for m in metas if m.get("learnable") and m["learnable"].get("signal") is None),
           "with_outcome_effects_shown": sum(1 for m in metas if (m.get("params") or {}).get("effects_shown")),
           "mean_lift": round(float(np.mean([x["lift"] for x in learn])), 4) if learn else None,
           "seeds": [min((m["seed"] for m in metas), default=None), max((m["seed"] for m in metas), default=None)],
           "shards": len(list(out.rglob("shard-*.npz"))), "rule": RULE, **(extra or {})}
    out.mkdir(parents=True, exist_ok=True)
    (out / "MANIFEST.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


__all__ = ["build", "folder", "manifest", "read"]
