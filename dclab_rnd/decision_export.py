"""The decision datasets exported for open-source models (package 22.5): ``make decisions-export``.

Reads the generated decision sets of 22.1 (``decisions/generated_v1/decisions/<point>/rows.jsonl``: train and dev; the
sealed rows live in ``sealed.jsonl``, which is never opened here) and writes ``decisions/export_v1/``: ``train.chat.jsonl``
and ``val.chat.jsonl`` (the dev partition) (the chat shape fine-tuning tools take), ``README.md`` (its Hugging Face dataset card),
``tabular.parquet`` (one column per state field), ``typed.jsonl`` (typed questions) and ``MANIFEST.json`` (counts, and
that all three formats hold the same records). The rows are not committed (they are rebuilt from the generated sets);
the card and the manifest are.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "research/llm-fine-tuning/experiments/decisions/generated_v1/decisions"
OUT = ROOT / "research/llm-fine-tuning/experiments/decisions/export_v1"
NAME = "DCLab decisions v1 (generated tables)"
LIMITS = ("Invented tables with planted columns (22.1): evidence about these kinds of columns, not about real tables.",
          "Labels are benchmark labels, set when each table was invented.",
          "The sealed partition is not included: it is read once, to compare a tuned model with the rules and the classifiers.")


def rows(sets: Path = SETS) -> list[dict[str, Any]]:
    out = []
    for folder in sorted(p for p in Path(sets).iterdir() if p.is_dir()):
        with open(folder / "rows.jsonl", encoding="utf-8") as handle:  # train and dev only: sealed.jsonl is never opened
            out += [json.loads(line) for line in handle if line.strip()]
    return out


def build(sets: Path = SETS, out: Path = OUT) -> dict[str, Any]:
    import pandas as pd

    from dclab_core import decision_export as de

    data = rows(sets)
    if any(r.get("split") == "sealed" for r in data):
        raise ValueError("a sealed row is in rows.jsonl: the generated sets are not what 22.1 wrote")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chats = de.chat(data)
    for split, name in (("train", "train"), ("dev", "val")):
        with open(out / f"{name}.chat.jsonl", "w", encoding="utf-8") as handle:
            for c in chats:
                if c["metadata"]["split"] == split:
                    handle.write(json.dumps(c, ensure_ascii=False) + "\n")
    table = de.tabular(data)
    pd.DataFrame(table).to_parquet(out / "tabular.parquet", index=False)
    typed = de.typed(data)
    with open(out / "typed.jsonl", "w", encoding="utf-8") as handle:
        for t in typed:
            handle.write(json.dumps(t, ensure_ascii=False) + "\n")
    same = [c["metadata"]["record"] for c in chats] == [t["record"] for t in table] == [t["record"] for t in typed]
    if not same:
        raise ValueError("the three formats do not hold the same records")
    (out / "README.md").write_text(de.card(NAME, data, "benchmark labels of the generated tables of 22.1", LIMITS), encoding="utf-8")
    manifest = {"name": NAME, "source": "decisions/generated_v1 (rows.jsonl; sealed.jsonl not read)", **de.summary(data),
                "formats": {"chat": ["train.chat.jsonl", "val.chat.jsonl"], "tabular": "tabular.parquet", "typed": "typed.jsonl"},
                "same_records_in_every_format": same, "limits": list(LIMITS)}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    if not SETS.exists():
        print("no generated decision sets: run make decisions-gen first")
        return 2
    print(json.dumps(build(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
