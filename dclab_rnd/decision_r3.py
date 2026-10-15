"""An R3-shaped run of DCLab's decision classifiers against the rule (package 22.7): ``make decision-r3``.

The generated decision sets' dev rows (``make decisions-gen``; the sealed file is never opened: 22.3 read it once) of
the main product's three Jev purposes become R3 cases (the table group is the cluster), and each fitted classifier
(``ledger.shadow.answerers``) answers them through ``dclab_core.adapters.main_product_r3`` twice: on the state as it was
trained (every band DCLab reads) and on the state the main product sends for the purpose (its release's fields). Each
run is an ``r3_runs``-shaped row, stored in ``evidence/campaigns/decision_r3_v1/results``.

    python -m dclab_rnd.decision_r3 [--per-point N] [--seed S] [--no-store]
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evidence" / "campaigns" / "decision_r3_v1" / "results"


def cases(per_point: int = 600, seed: int = 0) -> list[dict[str, Any]]:
    """Dev rows of the Jev purposes as R3 cases, drawn with a seed (the same draw for every arm)."""
    from dclab_core.adapters.main_product_r3 import releases

    from .decision_models_build import rows

    out = []
    for point in sorted(releases()):
        dev = [r for r in rows(point) if r["split"] == "dev" and (r.get("label") or {}).get("value") is not None]
        random.Random(seed).shuffle(dev)
        for r in dev[:per_point]:
            state = dict(r["state"])
            context = {k: state.pop(k) for k in ("target", "task") if k in state}
            rule = next((a["value"] for a in r["answers"] if a["by"] == "rule"), None)
            out.append({"point": point, "dataset": r["group"], "partition": r["split"], "fields": state, "context": context,
                        "column": state.get("column"), "options": r["options"], "rule": rule, "label": r["label"]["value"],
                        "label_source": r["label"]["source"]})
    return out


def measure(per_point: int = 600, seed: int = 0) -> list[dict[str, Any]]:
    from dclab_core.adapters.main_product_r3 import R3Answerer, run

    from .ledger import shadow

    drawn = cases(per_point, seed)
    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for model in shadow.answerers():
        mine = [c for c in drawn if c["point"] in model.points]
        if not mine:
            continue
        for projected in (False, True):
            row = run(mine, R3Answerer(model, projected=projected), candidate=f"dclab:{model.name}@{model.version}" + (":as-sent" if projected else ":as-trained"),
                      model_id=str(model.name)[:64], release_version=str(model.version)[:32], at=at)
            out.append({"answerer": f"{model.name}@{model.version}", "state": "as the main product sends it" if projected else "as trained", "row": row})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-point", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-store", action="store_true")
    args = parser.parse_args()
    runs = measure(args.per_point, args.seed)
    for r in runs:
        rep = r["row"]["report"]
        for point, p in rep["points"].items():
            a = p["sources"]["all"]
            print(f"{r['answerer']:52} {r['state']:30} n={a['total']} rule {a['rule_accuracy_all']} policy {a['ai_policy_accuracy_all']} "
                  f"in band {a['in_band']['n']} right {a['ai_in_band_accuracy']} gap {a['gap']['point']} [{a['gap']['lb']}, {a['gap']['ub']}] {a['gap']['method']} "
                  f"ablation {rep['ablation']['ai_off_equals_rule']}")
    if not args.no_store:
        RESULTS.mkdir(parents=True, exist_ok=True)
        n = 1 + max([int(p.stem.split("-")[1]) for p in RESULTS.glob("R3D-*.json")] or [0])
        path = RESULTS / f"R3D-{n:03d}.json"
        doc = {"id": f"R3D-{n:03d}", "campaign_id": "decision_r3_v1", "kind": "r3_shaped_run", "split": "dev", "per_point": args.per_point,
               "seed": args.seed, "runs": runs, "status": "completed", "experiment_id": f"R3D-{n:03d}", "spent_eur": 0.0,
               "rule": "An R3-shaped reading of DCLab's classifiers on generated dev cases, as the main product would read them; not "
                       "its R3 run (that runs there, on its cases), and not a claim about any real table."}
        path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
