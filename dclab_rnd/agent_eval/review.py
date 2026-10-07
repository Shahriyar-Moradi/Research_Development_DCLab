"""Expert review of the benchmark's labels (package 14.4): the kit for two reviewers.

``export`` writes one blank form per reviewer for a stratified sample of cases (dev and test), without the expected
decisions and under anonymous references, and a key that stays with the owner. Each row is one column of one case: its
name, kind, missing share, distinct count and three example values (the tables are seeded or public, never user data; the columns of
a case come in a shuffled order, because a generator appends its planted column last; several cases share a base table, so a reviewer
who compares cases of one base can see what was added: ask them not to, or give each reviewer different bases),
and the one thing the reviewer fills in: ``keep_out`` (not known at the prediction moment, or derived from the outcome,
or an identifier), ``usable``, ``time`` (the column that orders rows in time) or ``group`` (the entity several rows
belong to). ``agreement`` reads the two returned forms and reports the agreement between the reviewers and between each
of them and the benchmark (the share that agrees and Cohen's kappa), every disagreement for the resolution, and which
cases both reviewers confirm. Nothing here edits a case: a changed label on the sealed test set is a new benchmark
version, never an edit (the fingerprints test stands).

    python -m dclab_rnd.agent_eval.review export --per-trap 1 --reviewers A,B --out evaluation/review
    python -m dclab_rnd.agent_eval.review agreement --a evaluation/review/form_A.csv --b evaluation/review/form_B.csv \\
        --key evaluation/review/key.json --output evidence/campaigns/benchmark_v2/review/REV-001.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import benchmark

DECISIONS = ("keep_out", "usable", "time", "group")
FORM_COLUMNS = ("case_ref", "prediction_moment", "column", "kind", "missing", "unique", "examples", "decision")


def expected(case: Any) -> dict[str, str]:
    """The benchmark's label for each column it has an opinion on (the others stay unlabelled)."""
    labels = {c: "usable" for c in case.clean_inputs}
    labels.update({c: "keep_out" for c in case.leaks})
    if case.time_column:
        labels[case.time_column] = "time"
    if case.group_column:
        labels[case.group_column] = "group"
    return labels


def sample(per_trap: int, seed: int = 14) -> list[Any]:
    """Up to ``per_trap`` cases of every trap family from each split, drawn with a fixed seed."""
    rng = random.Random(seed)
    chosen: list[Any] = []
    for split in ("dev", "test"):
        by_trap: dict[str, list[Any]] = {}
        for case in benchmark.split(split):
            by_trap.setdefault(case.trap, []).append(case)
        for trap in sorted(by_trap):
            pool = sorted(by_trap[trap], key=lambda c: c.id)
            chosen.extend(rng.sample(pool, min(per_trap, len(pool))))
    return chosen


def _column_rows(case: Any, ref: str, seed: int = 14) -> list[dict[str, Any]]:
    """One row per column, in an order of its own: the generators append the planted column last, so the table's order
    would show it. The order is seeded by the reference, and is the same on every reviewer's form."""
    frame = case.table()
    rows = []
    names = [c for c in frame.columns if c != "target"]
    random.Random(f"{seed}-{ref}").shuffle(names)
    for name in names:
        column = frame[name]
        kind = "numeric" if str(column.dtype) in ("int64", "float64", "int32", "float32") else ("datetime" if "datetime" in str(column.dtype) else "category")
        examples = [str(v)[:24] for v in column.dropna().iloc[:3].tolist()]
        rows.append({"case_ref": ref, "prediction_moment": case.moment, "column": name, "kind": kind, "missing": round(float(column.isna().mean()), 3),
                     "unique": int(column.nunique()), "examples": " | ".join(examples), "decision": ""})
    return rows


def export(out: Path, reviewers: list[str], per_trap: int, seed: int = 14) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    cases = sample(per_trap, seed)
    order = list(range(len(cases)))
    random.Random(seed + 1).shuffle(order)  # the reference carries no order of the benchmark
    key = {f"R-{rank + 1:03d}": {"case": cases[i].id, "split": "dev" if cases[i] in benchmark.split("dev") else "test"} for rank, i in enumerate(order)}
    by_case = {k["case"]: ref for ref, k in key.items()}
    for reviewer in reviewers:
        path = out / f"form_{reviewer}.csv"
        if path.exists():
            raise FileExistsError(f"{path} exists; a form is never overwritten")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FORM_COLUMNS)
            writer.writeheader()
            for case in sorted(cases, key=lambda c: by_case[c.id]):
                writer.writerows(_column_rows(case, by_case[case.id], seed))
    key_path = out / "key.json"
    if key_path.exists():
        raise FileExistsError(f"{key_path} exists; a key is never overwritten")
    key_path.write_text(json.dumps({"benchmark": benchmark.VERSION, "seed": seed, "per_trap": per_trap, "key": key,
                                    "note": "Owner only: the references map to cases. Do not send this file to a reviewer."}, indent=2) + "\n", encoding="utf-8")
    return {"cases": len(cases), "reviewers": reviewers, "forms": [str(out / f"form_{r}.csv") for r in reviewers], "key": str(key_path)}


def read_form(path: Path) -> dict[tuple[str, str], str]:
    decisions: dict[tuple[str, str], str] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = (row.get("decision") or "").strip().lower().replace(" ", "_")
            if value:
                if value not in DECISIONS:
                    raise ValueError(f"{path}: {row['case_ref']} / {row['column']}: '{row['decision']}' is not one of {', '.join(DECISIONS)}")
                decisions[(row["case_ref"], row["column"])] = value
    return decisions


def kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two lists of decisions (None when it is undefined: no rows, or both always say the same one thing)."""
    n = len(a)
    if not n:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    chance = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if chance == 1 else round((observed - chance) / (1 - chance), 3)


def _compare(x: list[str], y: list[str]) -> dict[str, Any]:
    return {"rows": len(x), "agreement": round(sum(i == j for i, j in zip(x, y)) / len(x), 3) if x else None, "kappa": kappa(x, y)}


def agreement(a: Path, b: Path, key_path: Path) -> dict[str, Any]:
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))["key"]
    cases = {c.id: c for c in [*benchmark.split("dev"), *benchmark.split("test")]}
    da, db = read_form(a), read_form(b)
    both = sorted(set(da) & set(db))
    labels = {ref: expected(cases[k["case"]]) for ref, k in key.items()}
    bench = {(ref, col): lab for ref, cols in labels.items() for col, lab in cols.items()}
    disagreements, per_case = [], {}
    for ref, column in both:
        x, y, z = da[(ref, column)], db[(ref, column)], bench.get((ref, column))
        if x != y or (z is not None and x != z):
            disagreements.append({"case": key[ref]["case"], "ref": ref, "column": column, "reviewer_a": x, "reviewer_b": y, "benchmark": z})
        per_case.setdefault(ref, []).append(x == y and (z is None or x == z))
    # a labelled column that either reviewer left blank is not reviewed: the case is not confirmed, and the gap is listed
    undecided = [{"case": key[ref]["case"], "ref": ref, "column": col, "benchmark": lab, "reviewer_a": da.get((ref, col)), "reviewer_b": db.get((ref, col))}
                 for ref, cols in labels.items() for col, lab in cols.items() if (ref, col) not in da or (ref, col) not in db]
    gaps = {u["ref"] for u in undecided}
    confirmed = sorted(key[ref]["case"] for ref, ok in per_case.items() if all(ok) and ref not in gaps)
    labelled = [rc for rc in both if rc in bench]
    return {
        "campaign_id": benchmark.VERSION, "kind": "benchmark_label_review", "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows_both_reviewers": len(both), "reviewers": _compare([da[rc] for rc in both], [db[rc] for rc in both]),
        "benchmark_labelled_rows": len(labelled),
        "reviewer_a_vs_benchmark": _compare([da[rc] for rc in labelled], [bench[rc] for rc in labelled]),
        "reviewer_b_vs_benchmark": _compare([db[rc] for rc in labelled], [bench[rc] for rc in labelled]),
        "cases_reviewed": len(per_case), "cases_confirmed_by_both": confirmed, "disagreements": disagreements,
        "labelled_columns_not_decided_by_both": undecided,
        "next": "Resolve every disagreement together and record the decision; a label that changes on the sealed test set makes a new benchmark version "
                "(benchmark_v3), never an edit; a label that changes on dev is corrected in benchmark.py with the reason.",
        "limitations": ["Two reviewers on a sample of cases: the agreement says how clear the labels are, not that every unreviewed label is right.",
                        "A row counts only where both reviewers decided; a blank is not an agreement, and a case with a labelled column left blank by either "
                        "reviewer is listed under labelled_columns_not_decided_by_both and is not confirmed."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.review", description="Expert review of the benchmark's labels (14.4).")
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export", help="blank forms for the reviewers and the owner's key")
    e.add_argument("--per-trap", type=int, default=1)
    e.add_argument("--reviewers", default="A,B")
    e.add_argument("--seed", type=int, default=14)
    e.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("agreement", help="read two returned forms")
    g.add_argument("--a", type=Path, required=True)
    g.add_argument("--b", type=Path, required=True)
    g.add_argument("--key", type=Path, required=True)
    g.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        print(json.dumps(export(args.out, [r.strip() for r in args.reviewers.split(",") if r.strip()], args.per_trap, args.seed), indent=2))
        return 0
    if args.output.exists():
        parser.error(f"{args.output} exists; results are never overwritten")
    report = agreement(args.a, args.b, args.key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"{report['cases_reviewed']} cases · reviewers agree {report['reviewers']['agreement']} (kappa {report['reviewers']['kappa']}) · "
          f"{len(report['disagreements'])} disagreements · stored {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
