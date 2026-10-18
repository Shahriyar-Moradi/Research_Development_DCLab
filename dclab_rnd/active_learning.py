"""Does a student learn more from the labels it asks for? (package 29.11): ``python -m dclab_rnd.active_learning [--write]``.

On the generated sets (29.2's table-level choices: invented tables, labels true by construction), for each point and
seed, two students are trained on 200 labelled decisions of the training rows and scored on every dev decision:

- **random**: the first 200 of the training rows in the seed's order;
- **asked for**: the same first 20, then rounds of 20 chosen from the student fitted on what it has so far, up to 200:
  by ``dclab_core.active.select`` (``--gate v1``: a disagreement with the rule first, the least sure first; ALQ-001,
  failed) or by ``select_mixed`` (``--gate v2``, the default: half the least sure, half at random; ALQ-002, written after
  ALQ-001 failed and judged on the same dev rows: not passed either).

Both use the same quick student (``decision_models.train``, logistic, C=1) so only the choice of labels differs. The gate
(``GATE``) was fixed before the first run; the sealed rows are never read (``rows.jsonl`` holds train and dev only).
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "research/llm-fine-tuning/experiments/decisions/generated_v2/decisions"
CAMPAIGN = ROOT / "evidence/campaigns/active_learning_v1"
QUICK = {"families": ("logistic",), "strengths": (1.0,)}
GATE = {"version": "active_learning_gate_v1", "fixed": "2026-10-19, before the first run",
        "points": ["split.strategy", "data.cases_table", "task.framing"], "seeds": [0, 1, 2, 3, 4], "budget": 200, "start": 20, "batch": 20,
        "pass": "the asked-for student's mean dev accuracy over the seeds exceeds the random one's by at least 0.01 on at least two of the "
                "three points, and is below it by more than 0.01 on none", "rule": "disagree_first"}  # "rule" named after ALQ-001 ran (its stored gate lacks it)
# ALQ-001 failed under GATE (a disagreement-first choice skewed the labels). The second rule, fixed before its run
# (2026-10-19, after ALQ-001 and before ALQ-002): half the batch the least sure answers, half drawn at random; the same
# points, seeds, budget and pass. A second attempt, said as one: ALQ-001 stays as it was.
GATE_V2 = {**GATE, "version": "active_learning_gate_v2", "fixed": "2026-10-19, after ALQ-001 failed, before ALQ-002", "rule": "least_sure_and_random"}


def load(point: str, sets: Path = SETS) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [json.loads(x) for x in (Path(sets) / point / "rows.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if (r.get("label") or {}).get("value") is not None]
    return [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "dev"]


def _rule(row: dict[str, Any]) -> Any:
    return next((a.get("value") for a in row.get("answers") or [] if a.get("by") == "rule"), row.get("value_used"))


def _fit(point: str, rows: list[dict[str, Any]]) -> Any:
    from dclab_core.decision_models import train

    model, _ = train(point, rows, [], **QUICK)
    return model


def accuracy(model: Any, point: str, dev: list[dict[str, Any]]) -> float:
    from dclab_core.answerers import _key

    right = 0
    for r in dev:
        got = model.answer(point, r["state"], list(r.get("options") or []))
        right += got is not None and _key(got.value) == _key(r["label"]["value"])
    return round(right / len(dev), 4) if dev else 0.0


def _order(train: list[dict[str, Any]], point: str, seed: int) -> list[dict[str, Any]]:
    """The seed's order of the training rows, starting with ``start`` rows of two classes and two groups at least."""
    from dclab_core.answerers import _key

    rows = list(train)
    random.Random(f"{seed}:{point}").shuffle(rows)
    start = GATE["start"]
    for i in range(len(rows) - start):  # rotate until the start set can be fitted (two classes, two groups)
        head = rows[i:i + start]
        if len({_key(r["label"]["value"]) for r in head}) > 1 and len({r["group"] for r in head}) > 1:
            return rows[i:] + rows[:i]
    return rows


def asked_for(point: str, ordered: list[dict[str, Any]], gate: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    from dclab_core.active import select, select_mixed

    gate = gate or GATE

    labelled, pool = list(ordered[:gate["start"]]), list(ordered[gate["start"]:])
    rounds = 0
    while len(labelled) < gate["budget"] and pool:
        model = _fit(point, labelled)
        items = []
        for r in pool:
            got = model.answer(point, r["state"], list(r.get("options") or []))
            items.append({"point": point, "student": got.value if got else None, "rule": _rule(r), "confidence": got.confidence if got else None})
        k = min(gate["batch"], gate["budget"] - len(labelled))
        rounds += 1
        chosen = set(select_mixed(items, k, f"{point}:{len(labelled)}:{rounds}") if gate.get("rule") == "least_sure_and_random" else select(items, k))
        labelled += [r for i, r in enumerate(pool) if i in chosen]
        pool = [r for i, r in enumerate(pool) if i not in chosen]
    return labelled


def run(points: list[str] | None = None, seeds: list[int] | None = None, sets: Path = SETS, log=print, gate: dict[str, Any] | None = None) -> dict[str, Any]:
    gate = gate or GATE
    out: dict[str, Any] = {}
    for point in points or gate["points"]:
        train, dev = load(point, sets)
        per_seed = []
        for seed in seeds if seeds is not None else gate["seeds"]:
            started = time.perf_counter()
            ordered = _order(train, point, seed)
            random_acc = accuracy(_fit(point, ordered[:gate["budget"]]), point, dev)
            asked = asked_for(point, ordered, gate)
            asked_acc = accuracy(_fit(point, asked), point, dev)
            per_seed.append({"seed": seed, "random": random_acc, "asked_for": asked_acc, "seconds": round(time.perf_counter() - started, 1)})
            log(f"{point} seed {seed}: random {random_acc}, asked for {asked_acc}")
        mean = lambda k: round(sum(s[k] for s in per_seed) / len(per_seed), 4)  # noqa: E731
        out[point] = {"train_pool": len(train), "dev": len(dev), "seeds": per_seed, "random_mean": mean("random"), "asked_for_mean": mean("asked_for"),
                      "difference": round(mean("asked_for") - mean("random"), 4),
                      "asked_for_better_on_seeds": sum(s["asked_for"] > s["random"] for s in per_seed)}
    better = sum(v["difference"] >= 0.01 for v in out.values())
    worse = sum(v["difference"] < -0.01 for v in out.values())
    return {"gate": gate, "points": out, "verdict": {"better_by_0_01": better, "worse_by_0_01": worse, "pass": better >= 2 and worse == 0}}


def write(doc: dict[str, Any]) -> Path:
    from datetime import datetime, timezone

    results = CAMPAIGN / "results"
    results.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in results.glob("ALQ-*.json")] or [0])
    ident = f"ALQ-{n:03d}"
    doc = {"id": ident, "experiment_id": ident, "campaign_id": "active_learning_v1", "kind": "active_learning", "status": "completed",
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **doc,
           "rule": "Two quick students per point and seed, each on 200 labelled training decisions of the generated sets: the first 200 in the "
                   "seed's order, or the same first 20 then rounds of 20 chosen by what they would teach; both scored on every dev decision. "
                   "Invented tables: evidence about choosing labels, not about real projects."}
    path = results / f"{ident}_generated_v2.json"
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.active_learning", description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="store the result in evidence/campaigns/active_learning_v1")
    parser.add_argument("--gate", choices=("v1", "v2"), default="v2", help="v1: disagreement first (ALQ-001, failed); v2: least sure and random")
    args = parser.parse_args(argv)
    if not SETS.is_dir():
        print("no generated sets: run make decisions-gen first")
        return 2
    doc = run(gate=GATE if args.gate == "v1" else GATE_V2)
    print(json.dumps(doc["verdict"], indent=1))
    if args.write:
        print(write(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
