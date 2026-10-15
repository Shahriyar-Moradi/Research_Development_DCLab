"""Benchmark v4: every case filed by its problem kind, results reported per kind with intervals (package 16.14).

v3 stays as it is with its fingerprints and its results. v4 keeps v3's bases and traps on new seeds (every table
differs) and files each case by the kind it is a case of (a binary base is ``binary``, a regression base
``regression``, a rare base ``rare_event``), and adds the planted traps of the kinds' own replicas
(``dclab_rnd.kinds.<kind>.case(seed)``) that the scripted policies can judge: a table with one outcome column, yes or
no or a number, whose leak columns the replica declares (``binary``: the refund paid after a cancellation;
``regression``: the delivery's end time and the courier's rating; ``text_tabular``: the shop's reply). The other kinds
are judged by their own acceptance on their replicas (``kinds.campaign``, campaign ``kinds_v1``: each decision on three
seeds with an interval); ``by_kind.report`` puts both in one table per kind.

Split by source, like v3: **dev** = v3's dev bases and replica seeds 0 and 1; **test** (sealed) = v3's test bases on
new seeds and replica seeds 3, 4 and 5 (seeds 0 to 2 were read by the kinds' acceptance and pitfalls). The test fingerprints are committed in ``benchmark_v4_sealed.json``; a run on
the test set is always recorded; never change a prompt, a rule or a policy because of a test-set result.

    python -m dclab_rnd.agent_eval.benchmark_v4 --split dev
    python -m dclab_rnd.agent_eval.benchmark_v4 --split test --output evidence/campaigns/benchmark_v4/results/BEN4-002_test_scripted.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import benchmark_v3 as v3
from .cases import Case

VERSION = "benchmark_v4"
SALT = "frozen-2026-10-15"  # set before any v4 run
SEALED = Path(__file__).with_name("benchmark_v4_sealed.json")
REPLICA_KINDS = ("binary", "regression", "text_tabular")  # the kinds whose replica the scripted policies can judge
REPLICA_SEEDS = {"dev": (0, 1), "test": (3, 4, 5)}  # seed 2 was run by kinds_v1 and the pitfalls (0 to 2): never sealed
ROWS = 1500  # a replica is read on its first rows, so a run stays the size of a v3 case


def _seed(case_id: str) -> int:
    return int(hashlib.sha256(f"{VERSION}:{SALT}:{case_id}".encode()).hexdigest()[:8], 16)


def kind_of(case: Case) -> str:
    """The kind a v3 base is a case of."""
    if len(case.tags) > 3 and case.tags[3] == "rare":
        return "rare_event"
    return "binary" if case.tags[2] == "binary" else "regression"


def _v3_cases() -> list[Case]:
    out = []
    for c in v3.CASES:
        cid = c.id.replace("BV3-", "BV4-", 1)
        out.append(Case(cid, c.suite, c.trap, c.title, c.build, _seed(cid), moment=c.moment, leaks=c.leaks, innocent=c.innocent,
                        time_column=c.time_column, group_column=c.group_column, flag_duplicates=c.flag_duplicates, labels=c.labels,
                        strict_group=c.strict_group, note=c.note, tags=(*c.tags, kind_of(c))))
    return out


def _replica_builder(kind: str, seed: int) -> Callable[[np.random.Generator], pd.DataFrame]:
    def build(rng: np.random.Generator) -> pd.DataFrame:  # the replica is seeded by its own seed: the case's rng is not needed
        from dclab_rnd import kinds

        case = kinds.get(kind).case(seed)
        frame = case.table.head(ROWS).rename(columns={case.target: "target"})
        return frame.reset_index(drop=True)
    return build


def _replica_cases() -> list[Case]:
    from dclab_rnd import kinds

    out = []
    for split, seeds in REPLICA_SEEDS.items():
        for kind in REPLICA_KINDS:
            for seed in seeds:
                case = kinds.get(kind).case(seed)
                facts = case.facts or {}
                leaks = tuple(facts.get("leaky_columns") or ([facts["leak"]] if facts.get("leak") else []))
                time_column, entity = facts.get("time_column"), facts.get("entity")
                skip = {case.target, *leaks, facts.get("identifier"), time_column, entity, facts.get("text")}
                table = case.table
                innocent = tuple(c for c in table.columns if c not in skip  # the replica's honest columns: keeping one out is a false alarm
                                 and not str(c).lower().endswith("id") and not pd.api.types.is_datetime64_any_dtype(table[c]))  # keys, times
                cid = f"BV4-KIND-{kind}-{seed}"
                out.append(Case(cid, "leakage", f"{kind}_replica", f"{kind}: the replica's planted traps (seed {seed})",
                                _replica_builder(kind, seed), _seed(cid), moment=case.moment, leaks=leaks, innocent=innocent,
                                time_column=time_column, group_column=entity,
                                note=f"the {kind} kind's replica: its leaks, its time and its entity as the replica declares them",
                                tags=(split, f"replica:{kind}", case.task, "common", kind)))
    return out


CASES: tuple[Case, ...] = tuple(_v3_cases() + _replica_cases())


def split(name: str) -> tuple[Case, ...]:
    return tuple(c for c in CASES if c.tags[0] == name)


def sealed_fingerprints() -> dict[str, str]:
    return {c.id: c.fingerprint() for c in split("test")}


def write_sealed() -> None:
    """Freeze the test set: once, when v4 is made; never to make a failing check pass."""
    SEALED.write_text(json.dumps({"version": VERSION, "cases": sealed_fingerprints()}, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def by_kind(results: list[dict[str, Any]], cases: tuple[Case, ...]) -> dict[str, dict[str, Any]]:
    """Per kind and policy: the leaks caught and the false alarms, each [k, n] with its 95% Wilson interval."""
    from .run import wilson

    kind = {c.id: c.tags[-1] for c in cases}
    out: dict[str, dict[str, Any]] = {}
    for r in results:
        cell = out.setdefault(kind.get(r["case"], "?"), {}).setdefault(r["policy"], {"leaks_caught": [0, 0], "false_alarms": [0, 0]})
        if r.get("leak_caught") is not None:
            cell["leaks_caught"][0] += int(bool(r["leak_caught"]))
            cell["leaks_caught"][1] += 1
        if r.get("false_alarm") is not None:
            cell["false_alarms"][0] += int(bool(r["false_alarm"]))
            cell["false_alarms"][1] += 1
    for policies in out.values():
        for cell in policies.values():
            cell["leaks_caught_ci95"] = list(wilson(*cell["leaks_caught"])) if cell["leaks_caught"][1] else None
            cell["false_alarms_ci95"] = list(wilson(*cell["false_alarms"])) if cell["false_alarms"][1] else None
    return out


def run(split_name: str, policies: tuple[str, ...] = ("standard", "audit", "bad")) -> dict:
    """Score the policies on one split; only the command line records a run on the sealed set."""
    import importlib

    suite = importlib.import_module("dclab_rnd.agent_eval.run")
    cases = split(split_name)
    report = suite.run(policies, cases)
    report.update({"campaign_id": VERSION, "kind": "benchmark_scripted", "suite": f"{VERSION}:{split_name}", "prompt_hashes": None,
                   "split": split_name, "sealed": split_name == "test",
                   "question": "Per problem kind, on seeded tables and the kinds' replicas, which policies keep the leak out, follow the time and the "
                               "entity, and keep clean columns usable?",
                   "by_family": suite.by_trap_families(report["results"], {c.id: c.trap for c in cases}),
                   "by_kind": by_kind(report["results"], cases), "code": _provenance(),
                   "limitations": [
                       "Scripted policies on seeded, planted traps: evidence about these policies on these cases, not about any model's readiness for production.",
                       "Four kinds only (binary, regression, rare_event, text_tabular): the scripted policies judge a table with one yes or no or numeric "
                       "outcome; every other kind is judged by its acceptance on its replica (kinds_v1).",
                       "Cases built from one base or one replica share its columns and its generator: they are not independent of each other.",
                       "v3's test generators were scored before (BEN3-005 to BEN3-008); v4 reseeds them, so only the replica cases' sealed seeds are "
                       "new to every rule.",
                       "A leak counts as caught when the column is kept out of the model; the reason given is not checked.",
                   ]})
    return report


GUARDED = ("dclab_rnd/agent_eval/benchmark_v4_sealed.json", "dclab_rnd/agent_eval/benchmark_v4.py", "dclab_rnd/agent_eval/benchmark_v3.py",
           "dclab_rnd/agent_eval/benchmark.py", "dclab_rnd/agent_eval/cases.py", "dclab_rnd/agent_eval/run.py")


def _provenance() -> dict:
    return v3._provenance()


def frozen_problem() -> str | None:
    """Why the sealed set may not run now, or None (as v3: frozen as committed)."""
    frozen = json.loads(SEALED.read_text(encoding="utf-8")).get("cases") if SEALED.exists() else None
    if frozen != sealed_fingerprints():
        return "the sealed set does not match benchmark_v4_sealed.json (missing, or a case changed)"
    code, _ = v3._git("ls-files", "--error-unmatch", GUARDED[0])
    if code != 0:
        return "benchmark_v4_sealed.json is not committed"
    code, _ = v3._git("diff", "--quiet", "HEAD", "--", *GUARDED)
    if code != 0:
        return "the generators, the cases, the scorer or the fingerprints have uncommitted changes"
    return None


def main(argv: list[str] | None = None) -> int:
    import argparse
    import importlib
    import re

    suite = importlib.import_module("dclab_rnd.agent_eval.run")
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.benchmark_v4", description="Run benchmark v4 (16.14) with scripted policies.")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--policy", action="append", choices=sorted(suite.POLICIES))
    parser.add_argument("--output", type=Path, help="write the report to this new file (never overwritten); required for the sealed test set")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error(f"{args.output} exists; results are never overwritten")
    if args.split == "test" and not args.output:
        parser.error("a run on the sealed test set is always recorded: give --output evidence/campaigns/benchmark_v4/results/BEN4-<n>_test_….json")
    if args.split == "test":
        problem = frozen_problem()
        if problem:
            parser.error(f"refusing to run the sealed set: {problem}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump({"status": "started", "split": args.split, "campaign_id": VERSION, "code": _provenance()}, handle, indent=2)
            handle.write("\n")
    report = run(args.split, tuple(args.policy or ("standard", "audit", "bad")))
    for kind, policies in sorted(report["by_kind"].items()):
        for policy, s in policies.items():
            print(f"{kind:13s} {policy:9s} leaks caught {s['leaks_caught'][0]}/{s['leaks_caught'][1]} (95% {s['leaks_caught_ci95']})  "
                  f"false alarms {s['false_alarms'][0]}/{s['false_alarms'][1]} (95% {s['false_alarms_ci95']})")
    if args.output:
        found = re.match(r"([A-Z0-9]+-\d+)_", args.output.name)
        report = {"experiment_id": found.group(1) if found else args.output.stem, "status": "completed", **report}
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
