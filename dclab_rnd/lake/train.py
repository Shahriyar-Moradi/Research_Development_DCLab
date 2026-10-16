"""Training files from the lake (26.4, 27.7): what the owner's GPU run reads, and nothing it may not.

``python -m dclab_rnd.lake export --target decisions|solver [--out DIR] [--weights family=2,...]`` writes to
``lake/export/<target>/`` (not committed), from the gold sets only:

- ``train.chat.jsonl`` and ``val.chat.jsonl`` (and ``train.jsonl`` / ``valid.jsonl`` for MLX): chat rows the existing
  LoRA script (``sft/train_lora.py --data-dir``) and ``mlx_lm.lora`` read;
- ``MANIFEST.json``: counts by family or point, by origin and split, the gold manifests it came from, the mix weights,
  and what it refused to include.

Never in a training file: a row of the ``sealed`` split (scored once, by a frozen model), an ``eval_only`` row (Claude's
answers, DCLab's build history), a row whose origin is not trainable. A decision row is asked exactly as the served
model will be asked (``decision_export.request`` beside the ``decision_answer`` prompt), so a tuned model answers through
the gateway's tuned tier unchanged. The run itself is the owner's (``TRAINING.md``).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from dclab_core import lake as L

from . import LAKE, ROOT, VERSION

OUT = LAKE / "export"  # rows: not committed, like the rest of lake/
SPENT_SEALED = {"decisions/generated_v1": "read once by the classifiers' gate (22.3): no longer unseen by any model"}


def _rows(folder: Path, split: str) -> list[dict[str, Any]]:
    path = folder / VERSION / f"{split}.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []


def _decision_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    from dclab_core import prompts
    from dclab_core.decision_export import PURPOSE

    label = row["label"]
    answer = "yes" if label is True else "no" if label is False else str(label)
    return [{"role": "system", "content": prompts.text(PURPOSE)}, {"role": "user", "content": row["state_text"]},
            {"role": "assistant", "content": json.dumps({"answer": answer}, ensure_ascii=False)}]


def export(target: str, lake: Path = LAKE, out: Path | None = None, weights: dict[str, float] | None = None, seed: int = 0) -> dict[str, Any]:
    """The training and validation files of one target, from gold only; returns the manifest written beside them."""
    if target not in ("decisions", "solver"):
        raise ValueError("target is decisions or solver")
    out = Path(out) if out is not None else OUT / target
    gold = lake / "gold" / target
    weights = dict(weights or {})
    refused: Counter = Counter()
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    by_set: Counter = Counter()
    by_origin: Counter = Counter()
    for folder in sorted(p for p in gold.glob("*") if p.is_dir()):
        name = folder.name
        for split, into in (("train", "train"), ("dev", "val")):
            for row in _rows(folder, split):
                prov = row["provenance"]
                if prov["split"] in ("sealed", "eval_only") or prov["origin"] not in L.TRAINABLE or prov.get("from_sealed"):
                    refused[f"{prov['split']}/{prov['origin']}"] += 1
                    continue
                messages = _decision_messages(row) if target == "decisions" else row["messages"]
                item = {"messages": messages, "metadata": {"set": name, "origin": prov["origin"], "group": prov["group"], "split": into}}
                copies = weights.get(name, 1.0) if into == "train" else 1.0  # the mix: a family repeated or thinned in train only
                whole, part = int(copies), copies - int(copies)
                rng = random.Random(f"{seed}:{name}:{L.fingerprint(messages)[:12]}")
                n = whole + (1 if rng.random() < part else 0)
                rows[into] += [item] * n
                by_set[f"{into}:{name}"] += n
                by_origin[prov["origin"]] += n
        for split in ("sealed", "eval_only"):
            refused[f"{split} (never exported)"] += len(_rows(folder, split))
    random.Random(seed).shuffle(rows["train"])
    out.mkdir(parents=True, exist_ok=True)
    for name, items in (("train.chat.jsonl", rows["train"]), ("val.chat.jsonl", rows["val"])):
        (out / name).write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    for name, items in (("train.jsonl", rows["train"]), ("valid.jsonl", rows["val"])):  # mlx_lm reads these names
        (out / name).write_text("".join(json.dumps({"messages": i["messages"]}, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    manifests = {}
    for folder in sorted(p for p in gold.glob("*") if p.is_dir()):
        try:
            m = json.loads((folder / VERSION / "manifest.json").read_text(encoding="utf-8"))
            manifests[folder.name] = {k: m.get(k) for k in ("rows", "by_split", "by_origin", "built_at", "builder_commit")}
        except (OSError, ValueError):
            continue
    manifest = {"target": target, "train": len(rows["train"]), "val": len(rows["val"]), "by_set": dict(sorted(by_set.items())),
                "by_origin": dict(by_origin), "weights": weights, "seed": seed, "refused": dict(refused), "gold": manifests,
                "sealed_spent": SPENT_SEALED if target == "decisions" else {},
                "rule": "Built from the lake's gold sets only; sealed and evaluation-only rows are never in it. Scores of a model "
                        "trained on it say how it does on these sources, never that it is ready for production."}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
