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
    python -m dclab_rnd.decisions_gen --images [--seeds N]   # 26.8: the image points, to generated_images_v1/
    python -m dclab_rnd.decisions_gen --seeds 12 --first 24 --out research/llm-fine-tuning/experiments/decisions/generated_v1_committee   # 29.16
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

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


def build(seeds: int = 24, rows: int = 1500, out: Path = OUT, first: int = 0) -> dict[str, Any]:
    """``first``: the first seed (29.16: a fresh set of new seeds, whose tables no earlier set holds, has a sealed
    partition of its own)."""
    from dclab_core import benchgen as bg
    from dclab_core import datasets as builders

    records, labels, tables = [], [], []
    for name, base, target in bases():
        for seed in range(first, first + seeds):
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
               **({"first_seed": first} if first else {}), "tables": len(tables), "decisions": len(docs), "sets": sets,
               "rule": "Invented tables with planted columns: labels are true by construction; scores on them say how a "
                       "model reads these planted kinds, not real tables. The sealed partition is never used for tuning."}
    (out / "MANIFEST.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


IMAGES = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "generated_images_v1"


def _write_sets(docs: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    """One set per point, through the builder of the sets from real use; the sealed partition in a file of its own."""
    from dclab_core import datasets as builders

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
        positive = sum(1 for x in rows_ if x["label"] and x["label"]["value"] is True)
        content = sorted(json.dumps({k: x[k] for k in ("record", "state", "label")}, sort_keys=True, default=str) for x in sealed)
        sets[point] = {"decisions": len(rows_), "tables": len({x["group"] for x in rows_}), "splits": manifest.splits, "rule_right": right,
                       "label_yes": positive, "sealed_fingerprint": hashlib.sha256("\n".join(content).encode()).hexdigest()}
    return sets


def build_images(seeds: int = 40, out: Path = IMAGES) -> dict[str, Any]:
    """26.8: the image decision sets. Each seed draws two tables of shapes (circles and squares; left and right arrows),
    each with scenes filed again under a new name and a few shots given the other label; DCLab's image rules decide on
    every table (``dclab_core.image_decisions``) and each decision keeps its benchmark label."""
    from dclab_core import image_decisions as idc
    from dclab_core.replicas import shapes_and_notes as replica

    docs, tables = [], []
    for seed in range(seeds):
        for arrows in (False, True):
            table = f"{'arrows' if arrows else 'shapes'}-s{seed}"
            r = replica.shapes(seed * 7919 + (1 if arrows else 0), reshot=4 + seed % 5, mislabel=0.02 + 0.01 * (seed % 4), arrows=arrows)
            records, labels = idc.decide(r, table=table, at=AT)
            by_record: dict[str, list[dict[str, Any]]] = {}
            for label in labels:
                by_record.setdefault(label.record, []).append(json.loads(label.model_dump_json()))
            docs += [{**json.loads(x.model_dump_json()), "labels": by_record.get(x.id, [])} for x in records]
            tables.append(table)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    summary = {"version": "image_decisions_v1", "at": AT, "seeds": seeds, "tables": len(tables), "decisions": len(docs),
               "grouping": "one table per group: its images never sit in two splits",
               "sets": _write_sets(docs, out), "threshold": idc.THRESHOLD, "flip_drop": idc.FLIP_DROP,
               "rule": "Invented images with planted traps (scenes filed again, shots given the other label, arrows a flip changes): "
                       "labels are true by construction; scores on them say how a model reads these traps, not real photographs. "
                       "The sealed partition is never used for tuning."}
    (out / "MANIFEST.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


V2 = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "generated_v2"


def _record(point: str, subject: tuple[str, str], state: dict[str, Any], options: list[Any], rule_id: str, rule: Any, truth: Any,
            table: str, key: str, group: Optional[str] = None) -> tuple[Any, Any]:
    from dclab_core.ledger import Answer, DecisionLabel, DecisionRecord

    digest = hashlib.sha256(f"{table}|{key}|{point}".encode()).hexdigest()[:20]
    record = DecisionRecord(id=f"g2{digest}", point=point, subject={"kind": subject[0], "key": subject[1][:200]}, state=state,
                            options=[str(o) for o in options], answers=[Answer(by="rule", answerer=rule_id, value=rule)], value_used=rule,
                            used="rule", dataset=(group or table)[:80], at=AT)  # a table and its hidden-name twin share a group
    label = DecisionLabel(id=f"l2{digest}", record=record.id, point=point, value=truth, source="benchmark", by="generated:benchgen_v2", at=AT)
    return record, label


def _hide(frame: Any, seed: int) -> tuple[Any, dict[str, str]]:
    """Every column renamed ``f000``… in an order shuffled by ``seed`` (v1's hidden-name twin): a name never says the truth."""
    import numpy as np

    order = [str(c) for c in np.random.default_rng(seed).permutation(np.array(frame.columns, dtype=object))]
    names = {c: f"f{i:03d}" for i, c in enumerate(order)}
    return frame.rename(columns=names), names


def decide_v2(seed: int) -> tuple[list[Any], list[Any]]:
    """The rules' decisions on one seed's four tables (benchgen v2), each table with its names and with them hidden (one
    group: the twins never sit in two splits), each decision's state built as production builds it (``ledger.states``)
    and its planted truth beside it."""
    from dclab_core import benchgen2 as g

    from .draft.clean import clean
    from .kinds.base import default_split
    from .ledger import states
    from .studio.data import detect_task, profile_table
    from .studio.labels import cases_table as rule_cases
    from .studio.solution import propose

    records, labels = [], []

    def add(pair: tuple[Any, Any]) -> None:
        records.append(pair[0])
        labels.append(pair[1])

    def twins(frame: Any, salt: int) -> list[tuple[str, Any, dict[str, str]]]:
        hidden, names = _hide(frame, seed * 104729 + salt)
        return [("", frame, {str(c): str(c) for c in frame.columns}), ("-ob", hidden, names)]

    # the outcome's kind
    frame, target, facts = g.task_table(seed)
    group = f"task-s{seed}"
    for suffix, f, names in twins(frame, 1):
        t = names[target]
        add(_record("task.framing", ("draft", group + suffix), states.task_state(profile_table(f), t), ["binary", "multiclass", "regression"],
                    "rule:detect_task", detect_task(f, t)["task"], facts["task"], group + suffix, t, group))
    # the split: the product's rule on the product's proposal (its first time and group candidates, as the wizard declares them)
    frame, facts = g.split_table(seed)
    group = f"split-s{seed}"
    for suffix, f, names in twins(frame, 2):
        profile = profile_table(f)
        proposal = propose(f, profile, names["outcome"])
        time_c = (proposal.get("time_candidates") or [None])[0]
        group_c = (proposal.get("group_candidates") or [None])[0]
        add(_record("split.strategy", ("draft", group + suffix), states.split_state(profile, time_c, group_c), list(g.SPLITS), "rule:kind_split",
                    default_split({"time_column": time_c, "group_column": group_c, "spatial": False}), facts["split"], group + suffix, "split", group))
    # a list of known cases
    frame, filename, facts = g.cases_table(seed)
    group = f"cases-s{seed}"
    for suffix, f, _ in twins(frame, 3):
        profile = profile_table(f)
        add(_record("data.cases_table", ("draft", group + suffix), states.cases_state(profile, filename), ["yes", "no"], "rule:cases_table",
                    rule_cases(profile, filename) is not None, facts["cases"], group + suffix, "cases", group))
    # each column's cleaning step
    frame, truth = g.dirty_table(seed)
    group = f"clean-s{seed}"
    for suffix, f, names in twins(frame, 4):
        profile = profile_table(f)
        _, log = clean(f)
        applied: dict[str, str] = {}
        for entry in log:
            if entry["step"] in g.CLEAN_STEPS:
                for c in entry["columns"]:
                    applied.setdefault(str(c), entry["step"])
        for column, step in truth.items():
            c = names[column]
            add(_record("data.cleaning_step", ("column", c), states.cleaning_state(profile, c), list(g.CLEAN_STEPS), "rule:clean",
                        applied.get(c, "none"), step, group + suffix, c, group))
    return records, labels


def build_v2(seeds: int = 2400, out: Path = V2) -> dict[str, Any]:
    """29.2: the table-level homework (benchgen v2): every seed's four tables, the rules' decisions and the planted truth,
    one set per point with a sealed partition by table, through the same builder as every other set."""
    records, labels = [], []
    for seed in range(seeds):
        r, l = decide_v2(seed)
        records += r
        labels += l
    by_record: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        by_record.setdefault(label.record, []).append(json.loads(label.model_dump_json()))
    docs = [{**json.loads(r.model_dump_json()), "labels": by_record.get(r.id, [])} for r in records]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    summary = {"version": "benchgen_v2", "at": AT, "seeds": seeds, "tables": seeds * 8, "decisions": len(docs),
               "grouping": "a table and its hidden-name twin are one group: their decisions never sit in two splits", "sets": _write_sets(docs, out),
               "states": "built as production builds them (dclab_rnd/ledger/states.py), from each table's profile",
               "rules": {"task.framing": "studio.data.detect_task", "split.strategy": "kinds.base.default_split on the proposal's first time and group candidates (the wizard's declaration; Home's build declares no group)",
                         "data.cases_table": "studio.labels.cases_table", "data.cleaning_step": "draft.clean.clean, the first column step or none"},
               "rule": "Invented tables with planted truths (the outcome's kind, the split, a list of known cases, each column's cleaning "
                       "step) and traps; names drawn from pools shared across truths and every table also with its names hidden, so no "
                       "name says the truth. Labels are true by construction; scores on them say how a model reads these traps, not "
                       "real tables. The sealed partition is never used for tuning."}
    (out / "MANIFEST.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.decisions_gen", description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=None, help="24 for the tables, 40 for the images")
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--images", action="store_true", help="26.8: the image decision sets (generated_images_v1)")
    parser.add_argument("--v2", action="store_true", help="29.2: the table-level homework (generated_v2)")
    parser.add_argument("--first", type=int, default=0, help="29.16: the first seed of the tables (the committee's fresh set: "
                                                             "--seeds 12 --first 24 --out .../generated_v1_committee)")
    args = parser.parse_args(argv)
    done = (build_v2(args.seeds or 2400, args.out or V2) if args.v2 else
            build_images(args.seeds or 40, args.out or IMAGES) if args.images else build(args.seeds or 24, args.rows, args.out or OUT, first=args.first))
    print(f"{done['tables']} tables, {done['decisions']} decisions")
    for point, s in done["sets"].items():
        print(f"  {point:<26} {s['decisions']:>6} decisions on {s['tables']} tables; splits {s['splits']}; rules right {s['rule_right']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
