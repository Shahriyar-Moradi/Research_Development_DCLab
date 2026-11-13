"""TP-08 / P1+P7: frozen planning-quality benchmark runner, replica eval, live capture & offline tests."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from dclab_rnd.tasks.catalog import (
    CatalogTaskCall,
    compile_catalog_to_task_graph,
    deterministic_catalog_plan,
    execute_catalog_task,
    plan_with_model,
)
from .gate import Case, GATE_V4, judge_v4, load_split, score_proposal, verify_headroom, verify_sealed
from .replicas import evaluate_holdout, replica_frames

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "evidence/campaigns/task_planning_v4"
FIXED_SEVEN_STAGES = ("data", "leakage", "features", "models", "holdout", "final", "report")


def _dummy_workflow(family: str = "classification"):
    from dclab_core.solution_workflow import SolutionWorkflow, Step

    steps = [
        Step(id=s, block="prepare", name=s.title(), purpose="baseline stage", check="done", rule="ok")
        for s in FIXED_SEVEN_STAGES
    ]
    return SolutionWorkflow(
        family=family if family in {"classification", "regression", "rare_event", "uplift", "segmentation", "entity_tables", "streams"} else "classification",
        steps=steps,
    )


def run_legacy_baseline_case(case: Case, seed: int, *, with_replica_holdout: bool = False) -> dict[str, Any]:
    """Fixed 7-stage baseline (Arm B0): runs the generic stage order without catalog tasks or decomposition."""
    t0 = time.monotonic()
    target_str = str(case.known_facts.get("target") or "")
    inferred_subproblem = "regression" if ("price" in target_str or "duration" in target_str or "minutes" in target_str) else "classification"
    selected = [*FIXED_SEVEN_STAGES, "baseline"] if case.stratum == "standard" else list(FIXED_SEVEN_STAGES)
    proposal: dict[str, Any] = {
        "selected_tasks": selected,
        "caught_traps": [],
        "subproblems": [inferred_subproblem],
        "asked_owner": False,
        "forbidden_triggered": [],
        "safety": {},
        "holdout_score": None,
        "extra_prompts": 0,
        "cost_usd": 0.0,
        "seconds": round(time.monotonic() - t0, 4),
        "mode": "legacy_seven_stage",
    }
    if with_replica_holdout:
        proposal["holdout_score"] = evaluate_holdout(case, seed, proposal)
        proposal["seconds"] = round(time.monotonic() - t0, 4)
    return score_proposal(case, seed, proposal)


def run_deterministic_catalog_case(case: Case, seed: int, *, with_replica_holdout: bool = False) -> dict[str, Any]:
    """Rules-only P2 catalog baseline (Arm B1): explicit structural regex/fact rules with zero model calls."""
    t0 = time.monotonic()
    frames = replica_frames(case, seed) if with_replica_holdout else None
    det = deterministic_catalog_plan(case.brief, case.tables, case.known_facts)
    calls = [CatalogTaskCall.model_validate(c) for c in det["calls"]]
    evidence_rows = [execute_catalog_task(c, tables=case.tables, known_facts=case.known_facts, frames=frames) for c in calls]
    traps = sorted({t for ev in evidence_rows for t in ev["traps"]})
    target_str = str(case.known_facts.get("target") or "")
    sub = "regression" if ("price" in target_str or "duration" in target_str or "minutes" in target_str) else "classification"
    proposal: dict[str, Any] = {
        **det,
        "caught_traps": traps,
        "subproblems": [sub],
        "forbidden_triggered": [],
        "safety": {},
        "holdout_score": None,
        "extra_prompts": 0,
        "cost_usd": 0.0,
        "mode": "deterministic_catalog",
    }
    if with_replica_holdout:
        proposal["holdout_score"] = evaluate_holdout(case, seed, proposal)
    proposal["seconds"] = round(time.monotonic() - t0, 4)
    return score_proposal(case, seed, proposal)


def run_adaptive_case(
    case: Case,
    seed: int,
    *,
    gateway: Any,
    with_replica_holdout: bool = False,
    cap_eur: float = 0.05,
) -> dict[str, Any]:
    """Candidate Arm C: P6 router -> P5 evidence -> P3 decomposition -> P2 catalog -> P4 bounded re-planning."""
    t0 = time.monotonic()
    frames = replica_frames(case, seed) if with_replica_holdout else None
    plan = plan_with_model(
        case.brief,
        case.tables,
        case.known_facts,
        project_id=f"{case.id}_s{seed}",
        gateway=gateway,
        max_revisions=2,
        cap_eur=cap_eur,
        frames=frames,
    )
    if plan.get("decomposition") and plan.get("calls"):
        try:
            wf = _dummy_workflow((plan.get("subproblems") or ["classification"])[0])
            tg = compile_catalog_to_task_graph(
                f"{case.id}_s{seed}",
                wf,
                plan["decomposition"],
                plan["calls"],
                source="model" if plan.get("routed_to") == "model" else "deterministic",
            )
            plan["task_graph_fingerprint"] = tg.fingerprint
        except Exception as exc:
            plan["task_graph_error"] = str(exc)

    if with_replica_holdout:
        plan["holdout_score"] = evaluate_holdout(case, seed, plan)
    plan["seconds"] = round(time.monotonic() - t0, 4)
    row = score_proposal(case, seed, plan)
    row["routed_to"] = plan.get("routed_to")
    row["revisions"] = plan.get("revisions")
    row["model_request"] = plan.get("model_request")
    row["task_graph_fingerprint"] = plan.get("task_graph_fingerprint")
    return row


def run_headroom_check(split: str = "dev") -> dict[str, Any]:
    """Run the legacy fixed-pipeline baseline on the benchmark and prove real planning headroom."""
    cases, seeds, sha = load_split(split)
    rows = [run_legacy_baseline_case(case, seed) for case in cases for seed in seeds]
    report = verify_headroom(rows)
    return {
        "schema": "dclab_rnd.task_planning_v4_headroom.v1",
        "split": split,
        "cases_sha256": sha,
        "cases": len(cases),
        "seeds": list(seeds),
        "gate": GATE_V4["version"],
        **report,
    }


def _scripted_offline_server():
    class ScriptedPlannerServer:
        thinking = None

        def complete(self, messages, tools=None, max_tokens=None, **options):
            user_raw = str((messages[-1] if messages else {}).get("content") or "{}")
            req = json.loads(user_raw)
            if "stage_evidence" in req:
                # Re-plan turn keeps original calls plus any suggested followup.
                prev = json.loads(str(messages[-2]["content"]))
                existing = list(prev.get("calls") or [])
                seen = {c["task_type"] for c in existing}
                for f in req.get("suggested_followups") or []:
                    if f["task_type"] not in seen:
                        existing.append({"task_type": f["task_type"], "params": f.get("params") or {}, "reason": f.get("reason") or "Evidence followup", "citations": []})
                        seen.add(f["task_type"])
                prev["calls"] = existing
                return {
                    "content": json.dumps(prev),
                    "finish_reason": "stop",
                    "tool_calls": [],
                    "thinking": [{"text": "Offline scripted re-plan added evidence followup."}],
                    "usage": {"input_tokens": 20, "output_tokens": 30},
                }

            tables: dict[str, list[str]] = req.get("tables") or {}
            facts: dict[str, Any] = req.get("known_facts") or {}
            brief: str = str(req.get("brief") or "").lower()
            sources = list((req.get("evidence_sources") or {}).keys())
            cite = [sources[0]] if sources else []
            calls: list[dict[str, Any]] = []
            subproblems: list[dict[str, Any]] = []
            traps: list[str] = []

            if "target" in facts and facts.get("target") is None:
                calls.append({"task_type": "define_target", "params": {}, "reason": "Resolve ambiguous target", "citations": cite})
                calls.append({"task_type": "ask_owner", "params": {"field": "target", "question": "Confirm exact target definition.", "blocking": True}, "reason": "Target ambiguous", "citations": cite})
                traps.append("ambiguous_target")
            if ("unit" in facts and not facts.get("unit")) or ("horizon_days" in facts and facts.get("horizon_days") is None):
                if not any(c["task_type"] == "confirm_unit_and_horizon" for c in calls):
                    calls.append({"task_type": "confirm_unit_and_horizon", "params": {}, "reason": "Confirm unit & horizon", "citations": cite})
                if not any(c["task_type"] == "ask_owner" for c in calls):
                    calls.append({"task_type": "ask_owner", "params": {"field": "horizon", "question": "Confirm unit & horizon.", "blocking": True}, "reason": "Missing unit/horizon", "citations": cite})
                traps.append("missing_horizon_or_unit")

            for tname, cols in tables.items():
                if len(tables) > 1 and tname != list(tables)[0]:
                    shared = next((c for c in cols if c in tables[list(tables)[0]]), None)
                    if shared:
                        calls.append({"task_type": "check_join_key", "params": {"table": tname, "key": shared, "parent_table": list(tables)[0]}, "reason": "Verify join key", "citations": cite})
                for col in cols:
                    if any(w in col for w in ("refund", "last_order_date", "lifetime_orders", "investigation_outcome", "courier_rating", "delivered_at")):
                        calls.append({"task_type": "leakage_probe", "params": {"table": tname, "column": col}, "reason": "Check post-outcome/snapshot leak", "citations": cite})
                        traps.append(f"post_outcome_{col}")

            if facts.get("time_column") or "drift" in brief or "cutoff" in brief:
                tcol = facts.get("time_column") or next((c for cols in tables.values() for c in cols if c.endswith(("_time", "_date", "_at"))), "cutoff_date")
                calls.append({"task_type": "time_split_check", "params": {"column": tcol}, "reason": "Forward time split", "citations": cite})
                traps.append("time_drift_random_split_leak")
            if facts.get("entity_key"):
                calls.append({"task_type": "grouped_split", "params": {"entity": facts["entity_key"]}, "reason": "Group split", "citations": cite})
                traps.append(f"grouped_split_{facts['entity_key']}")
            if "uplift" in brief or "offer" in brief or "call vs email" in brief:
                subproblems.append({"id": "sub_uplift", "kind": "uplift", "objective": "Incremental causal uplift"})
                subproblems.append({"id": "sub_seg", "kind": "segmentation", "objective": "Responder segments"})
                calls.append({"task_type": "segment_subproblem", "params": {"kind": "uplift", "objective": "Uplift"}, "reason": "Uplift decomposition", "citations": cite})
                calls.append({"task_type": "segment_subproblem", "params": {"kind": "segmentation", "objective": "Segmentation"}, "reason": "Segment heterogeneity", "citations": cite})
                calls.append({"task_type": "threshold_for_cost", "params": {}, "reason": "Net value threshold", "citations": cite})
                traps.extend(["high_churn_is_not_high_uplift", "observational_treatment_bias"])
            if "persona" in brief or "segment" in brief:
                if not any(s["kind"] == "segmentation" for s in subproblems):
                    subproblems.append({"id": "sub_seg", "kind": "segmentation", "objective": "Segments"})
                subproblems.append({"id": "sub_cls", "kind": "classification", "objective": "Propensity"})
                calls.append({"task_type": "segment_subproblem", "params": {"kind": "segmentation", "objective": "Segments"}, "reason": "Segment first", "citations": cite})
                traps.append("single_classifier_misses_persona_segments")
            if facts.get("positive_rate") is not None and float(facts["positive_rate"]) < 0.05:
                subproblems.append({"id": "sub_rare", "kind": "rare_event", "objective": "Rare events"})
                calls.append({"task_type": "calibration", "params": {"positive_rate": float(facts["positive_rate"])}, "reason": "Rare event calibration", "citations": cite})
                calls.append({"task_type": "threshold_for_cost", "params": {"capacity_limit": facts.get("daily_review_cap")}, "reason": "Capacity threshold", "citations": cite})
                traps.extend(["extreme_class_imbalance", "capacity_constrained_threshold"])
            if "failed_within" in str(facts.get("target") or "") or "pump" in brief or "sensor" in brief:
                subproblems.append({"id": "sub_streams", "kind": "streams", "objective": "Streaming sensor telemetry"})
                subproblems.append({"id": "sub_rare2", "kind": "rare_event", "objective": "Rare fault detection"})
                calls.append({"task_type": "calibration", "params": {}, "reason": "Fault calibration", "citations": cite})
                calls.append({"task_type": "threshold_for_cost", "params": {"cost_ratio": facts.get("cost_ratio")}, "reason": "Cost ratio threshold", "citations": cite})
                traps.extend(["repeated_pump_readings_leak", "asymmetric_failure_cost"])
            if "fraud_flag" in str(facts.get("target") or ""):
                subproblems.append({"id": "sub_rare3", "kind": "rare_event", "objective": "Rare fraud"})
            if not subproblems:
                tstr = str(facts.get("target") or "")
                base_kind = "regression" if ("price" in tstr or "duration" in tstr or "minutes" in tstr) else ("entity_tables" if len(tables) > 1 and "inactive" in tstr or "escalated" in tstr else "classification")
                subproblems.append({"id": "main", "kind": base_kind, "objective": req.get("brief", "Solve task")[:120]})
            if not calls:
                calls.append({"task_type": "baseline", "params": {}, "reason": "Baseline", "citations": cite})

            reply_doc = {
                "version": "tp08.v1",
                "decomposition": {
                    "schema_version": "dclab_core.problem_decomposition.v1",
                    "decision": req.get("brief", "Decision")[:200],
                    "target": facts.get("target"),
                    "unit": facts.get("unit"),
                    "horizon_days": facts.get("horizon_days"),
                    "success_metric": "roc_auc",
                    "subproblems": subproblems,
                    "risks": [],
                    "unknowns": [],
                    "source": "model",
                },
                "calls": calls,
                "caught_traps": sorted(set(traps)),
            }
            return {
                "content": json.dumps(reply_doc),
                "finish_reason": "stop",
                "tool_calls": [],
                "thinking": [{"text": "Offline scripted planner JSON."}],
                "usage": {"input_tokens": 30, "output_tokens": 40},
            }

    return ScriptedPlannerServer()


def run_offline(*, split: str = "dev", limit: int | None = 6) -> dict[str, Any]:
    """Run $0 offline validation of Arms B0, B1, and C with real replica evaluation and capture check."""
    from dclab_rnd.agent_eval.task_planning.probe import offline_environment
    from dclab_rnd.models import FileUsage, Gateway
    from dclab_rnd.models.gateway import mark_invented_tables

    cases, seeds, sha = load_split(split)
    selected = cases[:limit] if limit else cases
    server = _scripted_offline_server()
    scores: dict[str, list[dict[str, Any]]] = {"B0": [], "B1": [], "C": []}
    with tempfile.TemporaryDirectory(prefix="tpv4-offline-") as tmp, offline_environment():
        home = Path(tmp)
        gw = mark_invented_tables(
            Gateway(FileUsage(home / "usage.jsonl"), transport=lambda tier, purpose: server),
            "tp08_offline_invented",
        )
        gw.home = home
        for case in selected:
            for seed in seeds:
                scores["B0"].append(run_legacy_baseline_case(case, seed, with_replica_holdout=True))
                scores["B1"].append(run_deterministic_catalog_case(case, seed, with_replica_holdout=True))
                scores["C"].append(run_adaptive_case(case, seed, gateway=gw, with_replica_holdout=True))
    judgment = judge_v4(scores, baseline_arm="B0", candidate_arm="C") if limit is None else None
    return {
        "schema": "dclab_rnd.task_planning_v4_offline.v1",
        "split": split,
        "cases_sha256": sha,
        "cases_run": len(selected),
        "seeds": list(seeds),
        "scores": scores,
        "judgment": judgment,
    }


def _summarize_exchange_capture(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.glob("**/model_exchanges/**/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    ok_rows = [r for r in rows if r.get("outcome") == "ok"]
    return {
        "total": len(rows),
        "ok": len(ok_rows),
        "whole_true": sum(1 for r in ok_rows if r.get("whole") is True),
        "raw_reasoning_present": sum(1 for r in ok_rows if r.get("reasoning") or r.get("reasoning_details") or r.get("thinking")),
        "purposes": sorted({str(r.get("purpose")) for r in rows}),
    }


def _export_dev_training_records(root: Path, c_rows: list[dict[str, Any]], out_path: Path) -> int:
    """Save clean dev-only model exchanges + thinking + outcomes into training_records/datasets/ (NEVER sealed)."""
    exchanges: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("**/model_exchanges/**/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                doc = json.loads(line)
                rid = doc.get("model_request")
                if rid:
                    exchanges[rid] = doc
    rows_out: list[str] = []
    for r in c_rows:
        ex = exchanges.get(r.get("model_request") or "")
        if not ex or not ex.get("whole"):
            continue
        rows_out.append(
            json.dumps(
                {
                    "schema": "dclab_rnd.tp08_sft_example.v1",
                    "case_id": r["case_id"],
                    "seed": r["seed"],
                    "stratum": r["stratum"],
                    "required_coverage": r["required_coverage"],
                    "trap_catch_rate": r["trap_catch_rate"],
                    "composite_kind_match": r["composite_kind_match"],
                    "holdout_score": r["holdout_score"],
                    "messages": ex.get("messages"),
                    "reply": ex.get("reply"),
                    "reasoning": ex.get("reasoning") or ex.get("thinking"),
                },
                ensure_ascii=False,
            )
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows_out) + ("\n" if rows_out else ""), encoding="utf-8")
    return len(rows_out)


def _live_worker(args: tuple[list[dict[str, Any]], list[int], str, str, str, str]) -> list[dict[str, dict[str, Any]]]:
    case_dicts, seeds, model, shard_cap, work_dir, price_snap_path = args
    from dclab_rnd import control
    from dclab_rnd.agent_eval.task_planning_v2 import DollarCeiling, _load_price_table, load_dotenv
    from dclab_rnd.agent_eval.task_planning_v3.gate import BudgetGateway
    from dclab_rnd.models import FileUsage, prices, settings

    load_dotenv()
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    os.environ.setdefault("OPENROUTER_REASONING", "1")
    os.environ["DCLAB_THINKING_EFFORT"] = os.environ.get("TP4_REASONING_EFFORT", "low")
    os.environ.setdefault("DCLAB_REASONING_TOKENS", "2048")
    if not control.live_allowed():
        raise RuntimeError("owner_has_stopped_live_calls")
    if settings.served_model("task_decomposer") != model:
        raise RuntimeError(f"model_mismatch:task_decomposer:{settings.served_model('task_decomposer')}!={model}")
    home = Path(work_dir)
    home.mkdir(parents=True, exist_ok=True)
    price_table = _load_price_table(Path(price_snap_path))
    ceiling = DollarCeiling(Decimal(shard_cap), max_calls=200)
    per_case_cap = min(0.05, float(shard_cap))
    out: list[dict[str, dict[str, Any]]] = []
    with patch.dict(prices.PRICES, price_table) if price_table else ExitStack():
        gw = BudgetGateway(FileUsage(home / "usage.jsonl"), ceiling, model, home=home, invented_tables="tp08_live")
        for idx, body in enumerate(case_dicts, 1):
            case = Case(
                id=body["id"],
                stratum=body["stratum"],
                replica=body["replica"],
                needs_model_planning=bool(body["needs_model_planning"]),
                brief=body["brief"],
                tables=dict(body["tables"]),
                known_facts=dict(body["known_facts"]),
                expected_tasks=tuple(body["expected_tasks"]),
                expected_traps=tuple(body["expected_traps"]),
                expected_subproblems=tuple(body["expected_subproblems"]),
                forbidden_actions=tuple(body["forbidden_actions"]),
            )
            for seed in seeds:
                b0 = run_legacy_baseline_case(case, seed, with_replica_holdout=True)
                b1 = run_deterministic_catalog_case(case, seed, with_replica_holdout=True)
                c_row = run_adaptive_case(case, seed, gateway=gw, with_replica_holdout=True, cap_eur=per_case_cap)
                print(
                    f"[{home.name} {idx}/{len(case_dicts)}] {case.id}:s{seed} "
                    f"cov={c_row['required_coverage']:.2f}(B0={b0['required_coverage']:.2f},B1={b1['required_coverage']:.2f}) "
                    f"trap={c_row['trap_catch_rate']:.2f} sub={c_row['composite_kind_match']:.2f} q={c_row['question_precision']:.1f} "
                    f"cost=${c_row['cost_usd']:.5f} {c_row['seconds']:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                out.append({"B0": b0, "B1": b1, "C": c_row})
    return out


def run_live(
    *,
    split: str = "dev",
    workers: int = 4,
    cap_usd: str = "1.20",
    tag: str = "run1",
    limit: int | None = None,
    result_name: str | None = None,
) -> dict[str, Any]:
    """Run matched B0 (legacy), B1 (deterministic catalog), and live Arm C across dev or sealed (once!)."""
    from dclab_rnd.agent_eval.task_planning_v2 import load_dotenv, snapshot_openrouter_prices
    from dclab_rnd.models import settings

    load_dotenv()
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    os.environ.setdefault("OPENROUTER_REASONING", "1")
    os.environ["DCLAB_THINKING_EFFORT"] = os.environ.get("TP4_REASONING_EFFORT", "low")
    os.environ.setdefault("DCLAB_REASONING_TOKENS", "2048")
    model = settings.served_model("task_decomposer")
    if not model:
        raise RuntimeError("missing_live_model_for_task_decomposer")

    cases, seeds, sha = load_split(split)
    selected = cases[:limit] if limit else cases
    cap = Decimal(str(cap_usd))
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    (CAMPAIGN / "results").mkdir(parents=True, exist_ok=True)
    snap_path = CAMPAIGN / "price_snapshot_usd.json"
    snapshot_openrouter_prices(model, snap_path)

    live_dir = ROOT / "agent_runs" / "live_tests" / f"task_planning_v4_{split}_{tag}"
    live_dir.mkdir(parents=True, exist_ok=True)
    if result_name:
        result_path = CAMPAIGN / "results" / result_name
        if result_path.exists():
            raise FileExistsError(f"immutable result exists: {result_path}")
    else:
        result_path = None

    case_dicts = [
        {
            "id": c.id,
            "stratum": c.stratum,
            "replica": c.replica,
            "needs_model_planning": c.needs_model_planning,
            "brief": c.brief,
            "tables": c.tables,
            "known_facts": c.known_facts,
            "expected_tasks": list(c.expected_tasks),
            "expected_traps": list(c.expected_traps),
            "expected_subproblems": list(c.expected_subproblems),
            "forbidden_actions": list(c.forbidden_actions),
        }
        for c in selected
    ]
    n_workers = max(1, min(workers, len(case_dicts)))
    shards: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
    for i, item in enumerate(case_dicts):
        shards[i % n_workers].append(item)
    shard_cap = (cap / Decimal(n_workers)).quantize(Decimal("0.01"))
    if shard_cap <= 0:
        shard_cap = Decimal("0.05")

    jobs = [(shard, list(seeds), str(model), str(shard_cap), str(live_dir / f"shard_{idx}"), str(snap_path)) for idx, shard in enumerate(shards) if shard]
    scores: dict[str, list[dict[str, Any]]] = {"B0": [], "B1": [], "C": []}
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(_live_worker, job) for job in jobs]
        for fut in as_completed(futures):
            for entry in fut.result():
                for arm in ("B0", "B1", "C"):
                    scores[arm].append(entry[arm])

    order = {f"{c['id']}:{s}": i for i, c in enumerate(case_dicts) for s in seeds}
    for arm in ("B0", "B1", "C"):
        scores[arm].sort(key=lambda r: order[r["id"]])

    j_b0 = judge_v4(scores, baseline_arm="B0", candidate_arm="C")
    j_b1 = judge_v4(scores, baseline_arm="B1", candidate_arm="C")
    capture = _summarize_exchange_capture(live_dir)
    exported = 0
    if split == "dev" and limit is None:
        exported = _export_dev_training_records(live_dir, scores["C"], ROOT / "training_records" / "datasets" / "task_planning_v4_dev.jsonl")

    total_cost = round(sum(float(r.get("cost_usd") or 0.0) for r in scores["C"]), 6)
    doc: dict[str, Any] = {
        "schema": "dclab_rnd.task_planning_v4_result.v1",
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "campaign_id": "task_planning_v4",
        "split": split,
        "cases_sha256": sha,
        "model": model,
        "live_model": model,
        "cap_usd": str(cap),
        "spent_eur": total_cost,
        "verdict": j_b0["verdict"],
        "gate": GATE_V4["version"],
        "judgment": j_b0,
        "vs_deterministic_catalog": j_b1,
        "exchange_capture": capture,
        "dev_training_exported": exported,
        "live_artifacts_dir": str(live_dir.relative_to(ROOT)),
        "scores": scores,
    }
    if result_path is not None:
        result_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


__all__ = [
    "run_legacy_baseline_case",
    "run_deterministic_catalog_case",
    "run_adaptive_case",
    "run_headroom_check",
    "run_offline",
    "run_live",
    "load_split",
    "verify_headroom",
    "verify_sealed",
    "judge_v4",
    "GATE_V4",
]
