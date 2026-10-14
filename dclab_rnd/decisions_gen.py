"""Generated decision benchmarks, built (package 22.1): ``make decisions-gen``.

Each base (a replica with a named outcome) is sampled under several seeds, its planted columns added
(``dclab_core.benchgen``), and each table taken twice: with its names and with every name hidden (``f000``…). DCLab's
rules decide on every planted column, and each decision is kept with its benchmark label. The decisions go through
the same builder as the sets from real use (``dclab_core.datasets.decisions``): one set per point, grouped by table,
a sealed partition in a file of its own, never used for tuning.

Written to ``research/llm-fine-tuning/experiments/decisions/generated_v1/`` (the rows are rebuilt by the command and
not committed; ``MANIFEST.json`` is, with every set's counts and the fingerprint of its sealed rows, so the sealed
partition cannot move unseen).

    python -m dclab_rnd.decisions_gen [--seeds N] [--rows N] [--out DIR]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "generated_v1"
AT = "2026-10-14T00:00:00+00:00"  # one time for every record: the same seeds build the same files


def bases() -> list[tuple[str, Any, str]]:
    """(name, table, outcome) for every replica whose outcome is a column."""
    from dclab_core.replicas import (campaign_uplift, city_prices, delivery_time, product_reviews, ride_pricing, shapes_and_notes,
                                     store_demand, subscription_churn, subscription_lifetimes, ticket_routing)

    return [("campaign_uplift", campaign_uplift.generate(seed=0).customers, "bought"),
            ("city_prices", city_prices.generate(seed=0).flats, "price"),
            ("delivery_time", delivery_time.generate(seed=0).orders, "actual_minutes"),
            ("product_reviews", product_reviews.generate(seed=0).reviews, "negative"),
            ("ride_pricing", ride_pricing.generate(seed=0).offers, "accepted"),
            ("notes", shapes_and_notes.notes(seed=0).table, "urgent"),
            ("store_demand", store_demand.generate(seed=0).sales, "units"),
            ("subscription_churn", subscription_churn.generate(seed=0).subscribers, "churned"),
            ("subscription_lifetimes", subscription_lifetimes.generate(seed=0).subscribers, "days_subscribed"),
            ("ticket_routing", ticket_routing.generate(seed=0).tickets, "assigned_team")]


def build(seeds: int = 24, rows: int = 1500, out: Path = OUT) -> dict[str, Any]:
    from dclab_core import benchgen as bg
    from dclab_core import datasets as builders

    records, labels, tables = [], [], []
    for name, base, target in bases():
        for seed in range(seeds):
            frame, planted = bg.plant(base, target, seed=seed * 7919 + len(name), rows=rows)
            for hidden in (False, True):
                table = bg.table_id(name, seed, hidden)
                f, p, t = bg.obfuscate(frame, planted, target, seed=seed * 104729 + len(name)) if hidden else (frame, planted, target)
                r, l = bg.decide(f, p, t, table, AT, group=bg.table_id(name, seed, False))  # the twins share a group
                records += r
                labels += l
                tables.append(table)
    by_record: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        by_record.setdefault(label.record, []).append(json.loads(label.model_dump_json()))
    docs = [{**json.loads(r.model_dump_json()), "labels": by_record.get(r.id, [])} for r in records]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    sets: dict[str, Any] = {}
    for point in sorted({d["point"] for d in docs}):
        rows_, manifest = builders.decisions(docs, point, consent="invented tables: no consent needed")
        folder = out / "decisions" / point
        folder.mkdir(parents=True, exist_ok=True)
        sealed = [x for x in rows_ if x["split"] == "sealed"]
        for file, items in (("rows.jsonl", [x for x in rows_ if x["split"] != "sealed"]), ("sealed.jsonl", sealed)):
            (folder / file).write_text("".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in items), encoding="utf-8")
        (folder / "manifest.json").write_text(manifest.model_dump_json(indent=1) + "\n", encoding="utf-8")
        right = sum(1 for x in rows_ if x["label"] and x["value_used"] == x["label"]["value"])
        content = sorted(json.dumps({k: x[k] for k in ("record", "state", "label")}, sort_keys=True, default=str) for x in sealed)
        sets[point] = {"decisions": len(rows_), "tables": len({x["group"] for x in rows_}), "splits": manifest.splits,
                       "rule_right": right, "sealed_fingerprint": hashlib.sha256("\n".join(content).encode()).hexdigest()}  # what, not only which
    summary = {"version": "benchgen_v1", "at": AT, "grouping": "base and seed; a table and its hidden-name twin share a group. The "
               "sealed partition holds new seeds of the same ten bases, not new tables: it measures reading planted kinds "
               "in tables like those trained on.", "seeds": seeds, "rows": rows, "bases": [b[0] for b in bases()],
               "tables": len(tables), "decisions": len(docs), "sets": sets,
               "rule": "Invented tables with planted columns: labels are true by construction; scores on them say how a "
                       "model reads these planted kinds, not real tables. The sealed partition is never used for tuning."}
    (out / "MANIFEST.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.decisions_gen", description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    done = build(args.seeds, args.rows, args.out)
    print(f"{done['tables']} tables, {done['decisions']} decisions")
    for point, s in done["sets"].items():
        print(f"  {point:<26} {s['decisions']:>6} decisions on {s['tables']} tables; splits {s['splits']}; rules right {s['rule_right']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
