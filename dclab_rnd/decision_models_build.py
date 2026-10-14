"""Classical decision classifiers, trained (package 22.2): ``make decision-models``.

For each point of the generated decision sets (``make decisions-gen``), a classifier chosen by table-grouped
out-of-fold accuracy (``dclab_core.decision_models.train``), measured on the dev rows beside the rule. Each card is
written to ``research/llm-fine-tuning/experiments/decisions/models_v1/<point>.json`` (committed: what was trained on
what, and how it compares with the rule); the fitted models to ``models_v1/<point>.joblib`` (rebuilt, not committed).
The sealed rows are never read here: 22.3 reads them once, under a gate fixed before.

    python -m dclab_rnd.decision_models_build [--points p1,p2] [--quick]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "generated_v1" / "decisions"
OUT = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "models_v1"


def rows(point: str, sets: Path = SETS) -> list[dict[str, Any]]:
    """The training and dev rows of a point; the sealed file is not opened."""
    path = Path(sets) / point / "rows.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def build(points: list[str] | None = None, quick: bool = False, sets: Path = SETS, out: Path = OUT) -> dict[str, Any]:
    import joblib

    from dclab_core.decision_models import train

    points = points or sorted(p.name for p in Path(sets).iterdir() if p.is_dir())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cards = {}
    for point in points:
        data = rows(point, sets)
        sources = {(r.get("label") or {}).get("source") for r in data if r.get("label")}
        if sources - {"benchmark"}:  # a measured label comes from a state field, which must then be left out
            raise ValueError(f"{point}: labels from {sorted(sources - {'benchmark'})} need the fields they are measured "
                             "from excluded (22.4); this build trains on benchmark labels only")
        model, card = train(point, [r for r in data if r["split"] == "train"], [r for r in data if r["split"] == "dev"],
                            **({"families": ("logistic",), "strengths": (1.0,)} if quick else {}))
        (out / f"{point}.json").write_text(card.model_dump_json(indent=1) + "\n", encoding="utf-8")
        joblib.dump(model, out / f"{point}.joblib")
        cards[point] = card
    record_fitted(out)
    return cards


def record_fitted(out: Path) -> dict[str, str]:
    """``fitted.json``: each fitted model's SHA-256, committed beside the cards. A model file is unpickled only when its
    bytes match (22.4): unpickling runs code, so a file swapped in the folder is refused before it is read."""
    import hashlib
    import json

    out = Path(out)
    hashes = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.joblib"))}
    (out / "fitted.json").write_text(json.dumps(hashes, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return hashes


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.decision_models_build", description=__doc__.split("\n")[0])
    parser.add_argument("--points", default="")
    parser.add_argument("--quick", action="store_true", help="one candidate per point (for a quick look)")
    args = parser.parse_args(argv)
    if not SETS.exists():
        print("No decision sets: run make decisions-gen first.", file=sys.stderr)
        return 2
    cards = build([p for p in args.points.split(",") if p] or None, args.quick)
    for point, c in cards.items():
        d, r = c.dev, c.rule_dev
        print(f"{point:<26} {c.family}:{c.strength:<5} dev accuracy {d.accuracy} [{d.accuracy_ci[0]}, {d.accuracy_ci[1]}]  "
              f"precision {d.precision} recall {d.recall} | rule {r.accuracy} / {r.precision} / {r.recall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
