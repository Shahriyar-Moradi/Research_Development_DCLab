"""One data foundation on the four-system replica (package 30.23): ``python -m dclab_rnd.foundation [--write]``.

Each system's table gets its catalog entry (owner, keys, checks), the customers are resolved into one view by the matching kind, and the
view is checked against the replica's truth (how many records land with their own person, and none with another). ``--write`` stores the
record in ``evidence/campaigns/foundation_v1/results/FND-<n>_four_systems.json`` with every record's entity beside it.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "evidence" / "campaigns" / "foundation_v1"
OWNERS = {"crm": "sales operations", "billing": "finance", "support": "customer care", "web": "e-commerce"}
FIELDS = [{"column": "name", "kind": "text"}, {"column": "email", "kind": "text"}, {"column": "phone", "kind": "text"}, {"column": "city", "kind": "category"}]


def run(seed: int = 0) -> dict[str, Any]:
    from dclab_core import foundation
    from dclab_core.replicas import four_systems

    r = four_systems.generate(seed)
    entries = [foundation.entry(t, name=f"{s}_customers", system=s, owner=OWNERS[s], keys=["id"], date_column="since").model_dump(mode="json")
               for s, t in r.systems.items()]
    out = foundation.resolve(r.systems, id_column="id", fields=FIELDS, matches=r.matches, block_text="email", seed=seed)
    groups: dict[str, set[int]] = {}
    for record, ent in out["entity"].items():
        groups.setdefault(ent, set()).add(r.truth[record])
    mixed = sum(1 for people in groups.values() if len(people) > 1)  # an entity holding two people: a wrong merge
    people = len(set(r.truth.values()))
    return {"seed": seed, "entries": entries, "records": out["records"], "entities": out["entities"], "people": people,
            "entities_mixing_people": mixed, "split_people": out["entities"] - people + mixed, "pair_f1": out["pair_f1"], "pair_f1_ci95": out["pair_f1_ci95"],
            "chosen": out["chosen"], "beats": out["beats"], "entity": out["entity"],
            "limits": ["Invented customers: the e-mail is close to unique, so one field nearly decides; real systems are harder.",
                       "The known matches are every pair of the replica's truth; a company knows only the pairs it confirmed."]}


def write(doc: dict[str, Any], campaign: Path = CAMPAIGN) -> Path:
    results, runs = campaign / "results", campaign / "runs"
    results.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in results.glob("FND-*.json")] or [0])
    ident = f"FND-{n:03d}"
    path, kept = results / f"{ident}_four_systems.json", runs / f"{ident}_entities.jsonl.gz"
    if path.exists() or kept.exists():
        raise SystemExit(f"{ident} exists: results are never overwritten")
    with gzip.open(kept, "wt", encoding="utf-8") as fh:
        for record, entity in sorted(doc["entity"].items()):
            fh.write(json.dumps({"record": record, "entity": entity}) + "\n")
    path.write_text(json.dumps({"id": ident, "experiment_id": ident, "campaign_id": "foundation_v1", "kind": "data_foundation", "status": "completed",
                                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **{k: v for k, v in doc.items() if k != "entity"},
                                "detail": f"runs/{kept.name}"}, indent=1, default=str) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.foundation", description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    doc = run(args.seed)
    print(json.dumps({k: v for k, v in doc.items() if k not in ("entity", "entries")}, indent=1, default=str))
    if args.write:
        print(write(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
