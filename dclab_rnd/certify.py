"""The first certification of the recipes' modifiers (package 30.12): ``python -m dclab_rnd.certify [--write]``.

Each modifier (rare positives, groups, time order, noisy labels, review capacity) is judged on the owner's recipes
document's five pass criteria, on a trap replica and two real datasets, every run through the engine with no model asked
(``acceptance.engine_run``: the LLM is off by construction):

- the retail returns replica, seeds 0 to 2 (``rare_event.retail_case``): labels from a late-evening rule, a cluster of
  look-alikes and a team checking five returns a day with a random part; its traps (``rare_event.accept_retail``);
- Telco churn and HyperAck (``data/project``): real tables whose known traps are their identifier and their final fares.

Per run and part: **detection** (what the proposal found on the rows, before anything the case states: the part
detected where it should apply, not where it should not; review capacity, which only a person can state, asked),
**traps** (every trap of the run caught: the run's own checks, the same for every part), **baseline** (the holdout's
average precision above a random ranking's), **artifact** (save, reload, predict: not built until 30.15) and **llm_off**
(the same result with the model on and off: no run with a model is compared yet); the last two are ``None``, pending.
``dclab_core.recipes.certify`` counts only runs where the part applies toward its level (a run where it rightly does not
apply checks its absence): certified needs a trap replica and two real datasets, so with two criteria pending every part
stays experimental and says what is missing. Telco and HyperAck only show the new modifiers' right absence; the owner's
fraud data (D6) is the real dataset named for labels and capacity.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "evidence" / "campaigns" / "certification_v1"
PARTS = ("rare_positives", "groups", "time_order", "noisy_labels", "review_capacity")
TELCO = ROOT / "data" / "project" / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
HYPERACK = ROOT / "data" / "project" / "hyperack" / "hyper_ackt-dataset.csv"
REAL_FOR = {"binary": ("telco", "hyperack")}  # 30.19: the recipes' two real datasets (D6), for a kind certified on them
FAMILY_OF = {"binary": "classification", "multiclass": "classification", "regression": "regression"}  # a kind's problem family, for detection


def cases(seeds: tuple[int, ...] = (0, 1, 2)) -> list[dict[str, Any]]:
    """Each run: its case, where it comes from, the parts it must apply (True), must not (False) or says nothing of
    (None), and its traps."""
    import pandas as pd

    from .kinds import rare_event
    from .kinds.acceptance import Case

    out = []
    for seed in seeds:
        case, _ = rare_event.KIND.retail_case(seed)
        out.append({"name": f"retail_returns s{seed}", "source": "replica", "case": case,
                    "expect": {"rare_positives": True, "groups": None, "time_order": True, "noisy_labels": True, "review_capacity": "asks"},
                    "traps": "retail"})
    telco = pd.read_csv(TELCO)
    out.append({"name": "telco", "source": "real", "expect": {**{p: False for p in PARTS}, "review_capacity": "asks"}, "traps": {"identifiers": ["customerID"]},
                "case": Case(telco, "Churn", "binary", "At the start of each month, from the account as it stands then.",
                             {"identifiers": ["customerID"], "positive_label": "Yes"}, quick=True)})
    hyper = pd.read_csv(HYPERACK)
    fares = [{"column": c, "reason": "the fare paid after the order is accepted"} for c in ("final_customer_fare", "final_biker_fare")]
    out.append({"name": "hyperack", "source": "real", "expect": {**{p: False for p in PARTS}, "time_order": True, "review_capacity": "asks"},
                "traps": {"blocked": ["final_customer_fare", "final_biker_fare"]},
                "case": Case(hyper, "hyper_ack", "binary", "When the order is created, before dispatch; the final fares are not known then.",
                             {"time_column": "first_created_at", "forbidden": fares}, quick=True)})
    return out


def _trap_checks(entry: dict[str, Any], run: Any) -> list[dict[str, Any]]:
    from .kinds import rare_event
    from .kinds.acceptance import check

    if entry["traps"] == "retail":
        return rare_event.KIND.accept_retail(run)
    roles = run.records["data"].get("evidence", {}).get("column_roles") or {}
    out = [check(f"`{c}` is never a feature", roles.get(c) == "identifier", str(roles.get(c))) for c in entry["traps"].get("identifiers", [])]
    out += [check(f"`{c}` is kept out", roles.get(c) == "blocked", str(roles.get(c))) for c in entry["traps"].get("blocked", [])]
    return out


def judge(entry: dict[str, Any], run: Any) -> dict[str, Any]:
    """One run's criteria for every part it speaks of."""
    recipe = run.records["data"].get("evidence", {}).get("recipe") or {}
    applied = {m.split("@")[0] for m in recipe.get("modifiers") or []}
    found = {d.get("name"): d for d in recipe.get("detected") or []}  # what the proposal read on the rows, before the case's statements
    final = run.evidence("final")
    ap = (final.get("holdout_metrics") or {}).get("average_precision")
    chance = (final.get("baselines") or {}).get("majority_class", {}).get("average_precision_random_ranking")
    checks = _trap_checks(entry, run)
    traps = all(c["ok"] for c in checks)
    baseline = ap is not None and chance is not None and ap > chance
    by_part, applies = {}, {}
    for part in PARTS:
        want = entry["expect"].get(part)
        if want is None:
            continue
        d = found.get(part) or {}
        right = bool(d.get("asks_user")) and not d.get("applies") if want == "asks" else bool(d.get("applies")) == want
        applies[part] = want is True or (want == "asks" and part in applied)  # capacity applies where the person states it
        by_part[part] = {"detection": right, "traps": traps, "baseline": baseline, "artifact": None, "llm_off": None}
    return {"name": entry["name"], "source": entry["source"], "applied": sorted(applied), "expected": entry["expect"],
            "detected": [{k: d.get(k) for k in ("name", "applies", "asks_user", "reason")} for d in recipe.get("detected") or []],
            "checks": checks, "holdout_average_precision": ap, "random_ranking_average_precision": chance,
            "capacity": final.get("capacity"), "label_sources": final.get("label_sources"), "criteria": by_part, "applies": applies}


def run(seeds: tuple[int, ...] = (0, 1, 2), log=print) -> dict[str, Any]:
    from dclab_core import modifiers as M
    from dclab_core.recipes import certify

    from .kinds.acceptance import engine_run

    judged = []
    for entry in cases(seeds):
        started = time.perf_counter()
        done = judge(entry, engine_run(entry["case"]))
        done["seconds"] = round(time.perf_counter() - started, 1)
        judged.append(done)
        log(f"{done['name']}: applied {done['applied']}; traps {all(c['ok'] for c in done['checks'])}; AP {done['holdout_average_precision']} "
            f"against {done['random_ranking_average_precision']}")
    parts = {part: certify([{"name": j["name"], "source": j["source"], "applies": j["applies"][part], "criteria": j["criteria"][part]}
                            for j in judged if part in j["criteria"]]) for part in PARTS}
    return {"runs": judged, "parts": parts, "versions": {name: M.PARTS[name].version for name in parts if name in M.PARTS},  # the version each verdict judged
            "limits": ["Every run is the engine's quick mode with no model asked: the same result with a model on is not compared (pending).",
                       "The artifact criterion (save, reload, predict) is not built until 30.15: pending on every run.",
                       "The trap criterion is each run's own checks, the same for every part.",
                       "On Telco and HyperAck the noisy-labels, capacity and rare modifiers can only show their right absence; the owner's "
                       "fraud data (D6) is the real dataset for them."]}


def entries_for(key: str, seeds: tuple[int, ...] = (0, 1, 2)) -> list[tuple[str, str, Any, Any]]:
    """What a kind is certified on: its replica's seeds, the real cases its module names, and (30.19) for a kind the engine runs without
    a machine of its own (binary) the real datasets of ``REAL_FOR``, each with its own traps."""
    from . import kinds

    kind = kinds.get(key)
    entries = [(f"{key} replica s{seed}", "replica", kind.case(seed), None) for seed in seeds]
    entries += [(f"{key} {case.facts.get('source')}", "real", case, None) for case in getattr(kind, "real_cases", lambda: [])()]
    entries += [(f"{key} {e['name']}", "real", e["case"], e) for e in cases(seeds=()) if e["name"] in REAL_FOR.get(key, ())]
    return entries


def run_kind(key: str, seeds: tuple[int, ...] = (0, 1, 2), log=print) -> dict[str, Any]:
    """30.16: a kind certified as a part, on its trap replica (``case(seed)``) and the real datasets it names
    (``real_cases()``), every run through the engine with no model asked: detection (the declared kind is the one that
    runs), traps (its acceptance checks on the library and on the engine's records), baseline (the kind's own), artifact
    (the code export rerun in an isolated Python reproduces the holdout number) and the LLM off (no model is in its path)."""
    import tempfile
    from unittest import mock

    from dclab_core.recipes import certify

    from . import kinds
    from .expansion import runner
    from .kinds import acceptance
    from .models.gateway import private_scope
    from .studio import code, engine

    kind = kinds.get(key)
    entries = entries_for(key, seeds)
    judged = []
    for name, source, case, entry in entries:
        started = time.perf_counter()
        own = kind.solve(case) if source == "replica" else None
        checks = kind.accept(own) if own is not None else []
        store, pid = acceptance.build(case, Path(tempfile.mkdtemp()))
        with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(store, pid)
        records = store.records(pid)
        run_ = acceptance.Run(case, store.get(pid)["solution"], records)
        if own is not None or kind.own_machine() is not None:
            checks += kind.accept_engine(run_)
        elif source == "replica":
            checks += kind.accept(run_)  # an engine kind: its acceptance reads the engine's run
        if entry is not None:
            checks += _trap_checks(entry, run_)  # the real dataset's own traps (an identifier, the fares paid after the order)
        final = records["final"]["evidence"]
        exported = code.export(store, pid, Path(tempfile.mkdtemp()) / "code", check=True)["checked"]
        beats = final.get("beats")
        metric = kind.metrics[0][0]
        baseline = all(beats.values()) and bool(beats) if isinstance(beats, dict) else all(
            ((final.get("holdout_metrics") or {}).get(metric) or 0) > ((final.get("baselines") or {}).get(b, {}).get(metric) or 0) for b in kind.baselines)
        criteria = {"detection": _routed(kind, case), "traps": all(c["ok"] for c in checks), "baseline": baseline,
                    "artifact": bool(exported.get("same")), "llm_off": True}
        judged.append({"name": name, "source": source, "applies": True, "criteria": criteria, "checks": checks,
                       "holdout_metrics": final.get("holdout_metrics"), "baselines": final.get("baselines"), "g4": (final.get("gate_g4") or {}).get("passed"),
                       "seconds": round(time.perf_counter() - started, 1)})
        log(f"{name}: {criteria}; {metric} {(final.get('holdout_metrics') or {}).get(metric)} against {final.get('baselines')}")
    replicas = [j for j in judged if j["source"] == "replica"]
    whole = all(all(v is not False for v in j["criteria"].values()) for j in replicas)
    counted = [j for j in judged if j["source"] != "replica"]
    if replicas:  # the review: one seed passing is not the trap passed; the replica counts once, and only when every seed passes
        counted.append({"name": f"{key} replica, {len(replicas)} seeds", "source": "replica", "applies": True,
                        "criteria": {c: all(j["criteria"][c] for j in replicas) for c in replicas[0]["criteria"]}})
    limits = ["Every run is the engine's quick mode with no model asked, so the LLM-off criterion holds by construction, said rather than measured.",
              "Detection is the router's choice on each case's own data card for the problem's family; the replica counts once, "
              "only when every seed passes every criterion."]
    if key == "matching":
        limits += ["The replica's known matches are every copy (the truth); a business knows only the copies it confirmed.",
                   "The real datasets are public benchmarks the owner chose to download (2026-10-19): two catalogues each, labelled in full."]
    if REAL_FOR.get(key):
        limits += ["Telco and HyperAck are the recipes' two real datasets (D6): one public churn table and the owner's delivery table; a pass says "
                   "the kind holds on them under these protocols, not on any other table."]
    from .recipes import part_of

    return {"runs": judged, "parts": {key: certify(counted)}, "versions": {key: part_of(kinds.get(key)).version},  # the version the verdict judged
            "replica_every_seed": whole, "limits": limits}


def _routed(kind: Any, case: Any) -> bool:
    """Detection: the router chooses the kind on the case's own data card, for the problem's family (no declaration read)."""
    import warnings

    from dclab_core import datacard as dc, problem_map as pm, solutions as sol

    warnings.filterwarnings("ignore")
    tables, roles = {"table": case.table}, {}
    for name, frame in (case.tables or {}).items():
        tables[name], roles[name] = frame, "related"
    sub = pm.SubProblem(id="sp1", part="part1", title="The problem", family=case.facts.get("family", FAMILY_OF.get(kind.key, "rare_event")),
                        labels=case.facts.get("labels", "column"))
    m = pm.ProblemMap(size="single", parts=[pm.Part(id="part1", title="The problem")], sub_problems=[sub])
    return sol.route(sub, dc.card_of(tables, name=kind.key, roles=roles), m).chosen == kind.key


def write(doc: dict[str, Any], campaign: Path = CAMPAIGN, what: str = "modifiers") -> Path:
    results = Path(campaign) / "results"
    results.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in results.glob("CRT-*.json")] or [0])
    ident = f"CRT-{n:03d}"
    out = {"id": ident, "experiment_id": ident, "campaign_id": "certification_v1", "kind": "certification", "status": "completed",
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **doc,
           "rule": ("Each modifier judged on the recipes document's five criteria on the retail returns replica (three seeds), Telco and "
                    "HyperAck, every run through the engine with no model asked; a part is certified only on a trap replica and two real "
                    "datasets passing every criterion." if what == "modifiers" else
                    f"The {what} kind judged as a part on the recipes document's five criteria on its trap replica (three seeds) and the real "
                    "datasets it names, every run through the engine with no model asked; certified only on a trap replica and two real datasets "
                    "passing every criterion.")}
    path = results / f"{ident}_{what}.json"
    path.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.certify", description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="store the result in evidence/campaigns/certification_v1")
    parser.add_argument("--kind", default=None, help="certify a kind as a part (30.16: matching) instead of the modifiers")
    args = parser.parse_args(argv)
    doc = run_kind(args.kind) if args.kind else run()
    for part, verdict in doc["parts"].items():
        print(f"{part:<16} {verdict['level']:<13} missing: {'; '.join(verdict['missing']) or '-'}; pending: {', '.join(verdict['pending']) or '-'}; "
              f"failed: {', '.join(verdict['failed']) or '-'}")
    if args.write:
        print(write(doc, what=args.kind or "modifiers"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
