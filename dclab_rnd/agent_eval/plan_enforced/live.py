"""The plan with room, on several models (stages H3 and H6 of Hierarchical Planning): on invented tables with planted traps,
the rules' plan against a model's plan, the model named on the command line. Scored by code against the cases' written
keys (``task_planning_v4.gate.score_proposal``): the traps caught, the checks a case needs, the kinds of sub-problem,
whether it asked the person only when it had to. No judge. Every plan is kept whole, with its path, in the run's folder.

    python -m dclab_rnd.agent_eval.plan_enforced.live --model qwen/qwen3.8-flash --cases 6 --cap-eur 0.4 [--store PE-00N]

Practice cases only (``task_planning_v4`` dev): its sealed cases are never read here. No Claude model, at any provider.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import CAMPAIGN, RESULTS, ROOT

MEASURES = ("trap_catch_rate", "required_coverage", "composite_kind_match", "question_precision")


def mean(rows: list[dict[str, Any]], key: str) -> Optional[float]:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(sum(values) / len(values), 4) if values else None


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {**{m: mean(rows, m) for m in MEASURES}, "forbidden": sum(len(r.get("forbidden_triggered") or ()) for r in rows),
            "seconds": mean(rows, "seconds"), "cases": len(rows)}


def path_stats(plan: dict[str, Any]) -> dict[str, Any]:
    legs = (plan.get("path") or {}).get("legs") or []
    steps = [s for leg in legs for s in leg.get("steps") or []]
    return {"sub_problems": len(legs), "ordered": sum(bool(leg.get("after")) for leg in legs), "steps": len(steps),
            "for_person": sum(bool(s.get("for_person")) for s in steps), "refused_parts": len(plan.get("refusals") or ())}


def rules_plan(case: Any, frames: Any) -> dict[str, Any]:
    from dclab_core.task_catalog import CatalogTaskCall, solution_path
    from dclab_rnd.tasks.catalog import _collect_traps, deterministic_catalog_plan, execute_catalog_task, kind_of, rules_decomposition

    began = time.monotonic()
    det = deterministic_catalog_plan(case.brief, case.tables, case.known_facts)
    calls = [CatalogTaskCall.model_validate(c) for c in det["calls"]]
    rows = [execute_catalog_task(c, tables=case.tables, known_facts=case.known_facts, frames=frames) for c in calls]
    decomp = rules_decomposition(case.brief, case.known_facts)
    return {**det, "caught_traps": _collect_traps([], decomp, rows), "subproblems": [kind_of(case.known_facts)], "decomposition": decomp.model_dump(mode="json"),
            "path": solution_path(decomp, calls, rows), "routed_to": "rules", "seconds": round(time.monotonic() - began, 3)}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.plan_enforced.live", description=__doc__)
    parser.add_argument("--model", required=True, help="the model that plans (the rules' plan is always made beside it)")
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--cap-eur", type=float, default=0.4)
    parser.add_argument("--store", default=None, help="a result id (PE-00N): write the result into the campaign")
    args = parser.parse_args(argv)
    if "claude" in args.model.lower():
        print("refused: the owner's rule of 2026-11-15: no Claude model in any test", file=sys.stderr)
        return 2
    if args.store and (RESULTS / f"{args.store}.json").exists():
        print(f"refused: {args.store}.json exists (results are never rewritten)", file=sys.stderr)
        return 2
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    from dclab_rnd.tasks.catalog import plan_with_model

    from ..conversation_exam import Budget, practice_effort
    from ..harness_ablation import KEEP_ROOT, _ModelView, owner_allows, price_snapshot, run_gateway
    from ..task_planning_v4.gate import load_split, score_proposal
    from ..task_planning_v4.replicas import replica_frames

    why = owner_allows(args.cap_eur)
    if why:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    cases, seeds, sha = load_split("dev")
    chosen = [c for c in cases if c.needs_model_planning][:args.cases]
    stamp = datetime.now(timezone.utc)
    run_dir = KEEP_ROOT / f"{stamp:%Y-%m-%d}-plan-path-{args.store or stamp.strftime('%H%M%S')}"
    price_snapshot(run_dir, args.model)
    gateway = run_gateway(run_dir)
    effort = practice_effort()
    budget = Budget(gateway, args.cap_eur)
    seed, rows, stopped = seeds[0], {"rules": [], "model": []}, None
    plans_dir = run_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    for case in chosen:
        frames = replica_frames(case, seed)
        rules = rules_plan(case, frames)
        rows["rules"].append({**score_proposal(case, seed, rules), "path": path_stats(rules)})
        began = time.monotonic()
        try:
            plan = plan_with_model(case.brief, case.tables, case.known_facts, project_id=f"{case.id}_s{seed}",
                                   gateway=_ModelView(budget, f"plan-path:{case.id}", args.model), cap_eur=args.cap_eur, frames=frames)
        except Exception as error:  # noqa: BLE001 - the run's cap or the provider: the run stops, marked incomplete
            stopped = f"{type(error).__name__}: {str(error)[:200]}"
            break
        plan["seconds"] = round(time.monotonic() - began, 2)
        (plans_dir / f"{case.id}.json").write_text(json.dumps({"rules": rules, "model": plan}, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        rows["model"].append({**score_proposal(case, seed, plan), "routed_to": plan.get("routed_to"), "path": path_stats(plan)})
        print(f"  {case.id} ({case.stratum}): rules traps {rows['rules'][-1]['trap_catch_rate']:.2f}, model traps {rows['model'][-1]['trap_catch_rate']:.2f} "
              f"({plan.get('routed_to')}, {plan['seconds']:.0f} s, {rows['model'][-1]['path']['sub_problems']} sub-problems, {rows['model'][-1]['path']['steps']} steps)", flush=True)
    done = len(rows["model"])
    doc = {"schema": "dclab_rnd.plan_path_live.v1", "campaign_id": CAMPAIGN, "id": args.store or "unstored", "at": stamp.isoformat(timespec="seconds"),
           "model": args.model, "thinking_effort": effort, "cases_sha256": sha, "seed": seed, "complete": stopped is None and done == len(chosen),
           "stopped": stopped, "spent_eur": round(budget.spent_eur(), 4), "run": run_dir.name,
           "summary": {"rules": summary(rows["rules"][:done]), "model": summary(rows["model"])},
           "answered_by_the_model": sum(r.get("routed_to") == "model" for r in rows["model"]),
           "path": {key: round(sum(r["path"][key] for r in rows["model"]) / done, 2) if done else None for key in ("sub_problems", "ordered", "steps", "for_person", "refused_parts")},
           "rows": rows,
           "limits": ["practice cases only, one seed, a handful of cases: a direction, not a proof", "scored by code against written keys; no judge",
                      "the rules' plan and the keys were written by the same project: the rules may be favoured"]}
    (run_dir / "result.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ("model", "complete", "stopped", "spent_eur", "summary", "answered_by_the_model", "path")}, indent=1))
    if args.store and doc["complete"]:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{args.store}.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"stored {(RESULTS / (args.store + '.json')).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
