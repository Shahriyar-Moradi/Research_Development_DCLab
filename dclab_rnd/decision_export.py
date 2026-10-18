"""The decision datasets exported for open-source models (package 22.5; every point with a set since 29.9): ``make decisions-export``.

Reads every decision set (``SETS``: 22.1's columns, 29.2's table-level choices, 26.8's images and 26.3's workflow points:
``decisions/<set>/decisions/<point>/rows.jsonl``, train and dev; the sealed rows live in ``sealed.jsonl``, which is never
opened here; a workflow row is read through ``dclab_core.decision_export.normalize``) and writes ``decisions/export_v2/``
(``export_v1`` is 22.5's, the columns only, kept as it was): ``train.chat.jsonl``
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
DECISIONS = ROOT / "research/llm-fine-tuning/experiments/decisions"
SETS = tuple(DECISIONS / name / "decisions" for name in ("generated_v1", "generated_v2", "generated_images_v1", "workflow_v1"))  # students.SETS
OUT = DECISIONS / "export_v2"
NAME = "DCLab decisions v2 (every point with a set)"
LIMITS = ("Invented tables (22.1, 29.2), invented image sets (26.8) and written journeys (26.3): evidence about these kinds of "
          "decisions, not about real tables.",
          "Labels are benchmark labels, set when each table was invented or each journey written; a workflow decision's journey may "
          "accept several answers (the comparison counts each as right).",
          "The sealed partitions are not included: each is read once, to compare a tuned model with the rules and the classifiers.",
          "The workflow points have no sealed set left: their own (journeys_workflow_sealed) was read by DWM-001 to 004, and the sealed "
          "journey sets their rows come from were each read by their gate before they were played here (a set not read yet is not).")


def rows(sets: Any = SETS) -> list[dict[str, Any]]:
    """Every set's train and dev rows (a folder, or several), normalised; sealed.jsonl is never opened."""
    from dclab_core.decision_export import normalize

    out = []
    for root in ([sets] if isinstance(sets, (str, Path)) else list(sets)):
        for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()) if Path(root).is_dir() else []:
            with open(folder / "rows.jsonl", encoding="utf-8") as handle:  # train and dev only: sealed.jsonl is never opened
                out += [normalize(json.loads(line)) for line in handle if line.strip()]
    return out


def build(sets: Any = SETS, out: Path = OUT) -> dict[str, Any]:
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
    (out / "README.md").write_text(de.card(NAME, data, "benchmark labels of the invented tables, image sets and written journeys", LIMITS),
                                   encoding="utf-8")
    sources = [str(Path(x).parent.name) for x in ([sets] if isinstance(sets, (str, Path)) else sets)]
    manifest = {"name": NAME, "source": f"decisions/{{{','.join(sources)}}} (rows.jsonl; sealed.jsonl not read)", **de.summary(data),
                "formats": {"chat": ["train.chat.jsonl", "val.chat.jsonl"], "tabular": "tabular.parquet", "typed": "typed.jsonl"},
                "same_records_in_every_format": same, "limits": list(LIMITS)}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    if not any(Path(s).exists() for s in SETS):
        print("no decision sets: run make decisions-gen first")
        return 2
    print(json.dumps(build(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
