"""End-to-end Live Studio + SolutionTeam (P2–P7) verification across realistic multi-table, drift, rare-event, ambiguous, and clean scenarios."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from dclab_rnd import experience, ledger
from dclab_rnd.agent_eval.task_planning_v2 import DollarCeiling, load_dotenv, snapshot_openrouter_prices
from dclab_rnd.agent_eval.task_planning_v3.gate import BudgetGateway
from dclab_rnd.expansion import runner
from dclab_rnd.models import FileUsage, settings as model_settings
from dclab_rnd.orchestra.checkpoints import open_checkpointer
from dclab_rnd.orchestra.solution import SolutionTeam
from dclab_rnd.studio import ProjectStore, data, graph
from dclab_rnd.tasks.dispatch import TaskDispatcher
from dclab_rnd.tasks.host import planning_host
from dclab_rnd.tasks.view import task_plan_view

ROOT = Path(__file__).resolve().parents[3]


def _scenario_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "studio_multitable_post_outcome",
            "stratum": "multi_table_leakage",
            "seed": 401,
            "expect_routed_to": "model",
            "goal": "Predict customer churn in the next 30 days across customer and order tables; outreach happens before closure review.",
            "tables": {"customers": ["customer_id", "tenure", "monthly_charge", "churn", "closure_recorded_after_review"], "orders": ["customer_id", "order_count", "refund_amount"]},
            "target": "churn",
            "task": "binary",
            "positive_label": "yes",
            "time_column": None,
            "forbidden": [],
            "unresolved": False,
        },
        {
            "id": "studio_timedrift_fraud_capacity",
            "stratum": "time_capacity",
            "seed": 402,
            "expect_routed_to": "model",
            "goal": "Predict disputed claims ordered by time under a 35 claims/day review capacity.",
            "tables": {"claims": ["claim_id", "submitted_at", "amount", "prior_claims", "fraud_flag"]},
            "target": "fraud_flag",
            "task": "binary",
            "positive_label": "yes",
            "time_column": "submitted_at",
            "daily_review_cap": 35,
            "positive_rate": 0.03,
            "forbidden": [],
            "unresolved": False,
        },
        {
            "id": "studio_composite_uplift",
            "stratum": "composite",
            "seed": 403,
            "expect_routed_to": "model",
            "goal": "Predict 30-day renewal probability and voucher uplift sensitivity for retention outreach.",
            "tables": {"subscribers": ["subscriber_id", "tenure", "spend", "voucher_offer", "renewed"]},
            "target": "renewed",
            "task": "binary",
            "positive_label": "yes",
            "time_column": None,
            "treatment": "voucher_offer",
            "forbidden": [],
            "unresolved": False,
        },
        {
            "id": "studio_ambiguous_ask_owner",
            "stratum": "ambiguous_target",
            "seed": 404,
            "expect_routed_to": "model",
            "goal": "Improve account health by prioritizing high-risk accounts before monthly review.",
            "tables": {"accounts": ["account_id", "usage_score", "tickets_open", "at_risk_flag"]},
            "target": None,
            "resolved_target": "at_risk_flag",
            "task": "binary",
            "positive_label": "yes",
            "time_column": None,
            "forbidden": [],
            "unresolved": True,
        },
        {
            "id": "studio_clean_deterministic_fastpath",
            "stratum": "standard",
            "seed": 405,
            "expect_routed_to": "deterministic",
            "goal": "Predict binary subscriber churn from clean account tenure and monthly billing features.",
            "tables": None,
            "target": "churn",
            "task": "binary",
            "positive_label": "yes",
            "time_column": None,
            "forbidden": [],
            "unresolved": False,
        },
    ]


def _build_project(store: ProjectStore, spec: dict[str, Any]) -> str:
    rng = np.random.default_rng(spec["seed"])
    n = 180
    sid = spec["id"]
    proj = store.create(f"Live {sid}", "retail", spec["goal"])
    pid = proj["id"]

    if sid == "studio_clean_deterministic_fastpath":
        tenure = rng.integers(1, 72, n)
        charge = rng.uniform(15, 110, n)
        p = 1 / (1 + np.exp(-(-0.6 - 0.025 * tenure + 0.014 * charge)))
        y = (rng.random(n) < p).astype(int)
        df = pd.DataFrame({"member_id": [f"m_{i}" for i in range(n)], "tenure": tenure, "monthly_charge": charge, "churn": np.where(y == 1, "yes", "no")})
        fname = "members.csv"
    elif sid == "studio_timedrift_fraud_capacity":
        amount = rng.uniform(20, 2000, n)
        prior = rng.integers(0, 5, n)
        p = 1 / (1 + np.exp(-(-1.6 + 0.0008 * amount + 0.4 * prior)))
        y = (rng.random(n) < p).astype(int)
        if y.sum() < 16:
            y[:16] = 1
        df = pd.DataFrame({
            "claim_id": [f"cl_{i}" for i in range(n)],
            "submitted_at": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "amount": amount,
            "prior_claims": prior,
            "fraud_flag": np.where(y == 1, "yes", "no"),
        })
        fname = "claims.csv"
    elif sid == "studio_composite_uplift":
        tenure = rng.integers(1, 48, n)
        spend = rng.uniform(20, 300, n)
        voucher = rng.integers(0, 2, n)
        p = 1 / (1 + np.exp(-(-0.3 + 0.02 * tenure + 0.004 * spend + 0.35 * voucher)))
        y = (rng.random(n) < p).astype(int)
        df = pd.DataFrame({
            "subscriber_id": [f"sub_{i}" for i in range(n)],
            "tenure": tenure,
            "spend": spend,
            "voucher_offer": voucher,
            "renewed": np.where(y == 1, "yes", "no"),
        })
        fname = "subscribers.csv"
    elif sid == "studio_ambiguous_ask_owner":
        usage = rng.uniform(5, 95, n)
        tix = rng.integers(0, 12, n)
        p = 1 / (1 + np.exp(-(0.5 - 0.03 * usage + 0.25 * tix)))
        y = (rng.random(n) < p).astype(int)
        df = pd.DataFrame({
            "account_id": [f"acc_{i}" for i in range(n)],
            "usage_score": usage,
            "tickets_open": tix,
            "at_risk_flag": np.where(y == 1, "yes", "no"),
        })
        fname = "accounts.csv"
    else:
        tenure = rng.integers(1, 60, n)
        charges = rng.uniform(15, 110, n)
        p = 1 / (1 + np.exp(-(-0.7 - 0.02 * tenure + 0.012 * charges)))
        y = (rng.random(n) < p).astype(int)
        df = pd.DataFrame({
            "customer_id": [f"c_{i}" for i in range(n)],
            "tenure": tenure,
            "monthly_charge": charges,
            "churn": np.where(y == 1, "yes", "no"),
            "closure_recorded_after_review": np.where(rng.random(n) < np.where(y == 1, 0.78, 0.22), 1, 0),
        })
        fname = "customers.csv"

    df.to_csv(store.data_dir(pid) / fname, index=False)
    data.attach_data(store, pid, fname)
    proj = store.get(pid)
    hints: dict[str, Any] = {}
    if spec.get("tables"):
        hints["tables"] = dict(spec["tables"])
    if spec.get("treatment"):
        hints["treatment"] = spec["treatment"]
    if spec.get("time_column"):
        hints["time_column"] = spec["time_column"]
    if spec.get("daily_review_cap"):
        hints["daily_review_cap"] = spec["daily_review_cap"]
    if spec.get("positive_rate") is not None:
        hints["positive_rate"] = spec["positive_rate"]
    if spec.get("unresolved"):
        hints["prediction_moment_unresolved"] = True
    if hints:
        proj["catalog_hints"] = hints
    proj["solution"] = {
        "target": spec.get("resolved_target") or spec["target"],
        "task": spec["task"],
        **({"positive_label": spec["positive_label"]} if spec["positive_label"] else {}),
        "prediction_moment": "At prediction cutoff before outreach or review.",
        "forbidden": list(spec.get("forbidden") or ()),
        "identifiers": [str(df.columns[0])],
        "metric": "roc_auc",
        "labels": "complete",
    }
    proj["settings"].update(quick=True, max_rows=500, private=False)
    proj["policy"] = {"require_solution_signoff": True, "require_holdout_approval": True}
    store.save(proj)
    graph.approve_gate(store, pid, "solution", by="live-studio-owner", reason="Predeclared live verification")
    return pid


def _spent_usd(usage_path: Path) -> float:
    if not usage_path.is_file():
        return 0.0
    total = 0.0
    for line in usage_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                doc = json.loads(line)
                total += float(doc.get("cost_eur") or doc.get("cost_usd") or 0.0)
            except ValueError:
                pass
    return total


def run_live_studio(
    *,
    cap_usd: str = "1.20",
    tag: str = "adaptive_studio_live_20261113",
) -> dict[str, Any]:
    """Run all 5 Studio + SolutionTeam scenarios against live OpenRouter under BudgetGateway."""
    from dclab_rnd.agent_eval.task_planning_v4 import _summarize_exchange_capture

    load_dotenv()
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    os.environ.setdefault("OPENROUTER_REASONING", "1")
    os.environ["DCLAB_THINKING_EFFORT"] = os.environ.get("TP4_REASONING_EFFORT", "low")
    os.environ.setdefault("DCLAB_REASONING_TOKENS", "2048")
    model = model_settings.served_model("task_decomposer")
    if not model:
        raise RuntimeError("missing_live_model_for_task_decomposer")

    snap_path = ROOT / "evidence" / "campaigns" / "task_planning_v4" / "price_snapshot_usd.json"
    snapshot_openrouter_prices(model, snap_path)
    out_dir = ROOT / "agent_runs" / "live_tests" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = Decimal(str(cap_usd))

    results: list[dict[str, Any]] = []
    with ExitStack() as stack:
        prev_log, prev_ledger = experience.current_log(), ledger.current_ledger()
        experience.install(experience.FileLog(out_dir / "experience", tag))
        file_ledger = ledger.FileLedger(out_dir / "ledger", tag)
        ledger.install(file_ledger)
        stack.callback(experience.install, prev_log)
        stack.callback(ledger.install, prev_ledger)
        stack.enter_context(runner.fast_profile())
        stack.enter_context(patch("dclab_rnd.experience.model_text_allowed", return_value=True))
        saver = stack.enter_context(open_checkpointer(out_dir, database_url=""))
        store = ProjectStore(out_dir / "projects")
        ceiling = DollarCeiling(cap, max_calls=200)
        usage_path = out_dir / "usage.jsonl"
        gw = BudgetGateway(
            FileUsage(usage_path),
            ceiling,
            model,
            home=out_dir,
            invented_tables="adaptive_studio_live",
        )

        for spec in _scenario_specs():
            t0 = time.monotonic()
            pid = _build_project(store, spec)
            before_spent = _spent_usd(usage_path)
            with planning_host(name=f"live-{spec['id']}"):
                team = SolutionTeam(store, saver, home=out_dir, task_execution=True, admit_proposals=True, gateway=gw)
                res = team.run(pid)
                saw_clarify = False
                for _ in range(6):
                    if res.get("status") != "paused":
                        break
                    asks = res.get("asks") or {}
                    if asks.get("field") == "catalog_clarify" or asks.get("key") == "catalog_clarify":
                        saw_clarify = True
                        res = team.run(pid, resume="Confirmed binary target at monthly cutoff.")
                    elif asks.get("gate") == "audit":
                        res = team.run(pid, resume=False)
                    elif asks.get("gate") == "holdout":
                        res = team.run(pid, resume=True)
                    else:
                        res = team.run(pid, resume=True)

            st = TaskDispatcher(store, saver).state(pid) or {}
            cp = st.get("catalog_plan") or {}
            view = task_plan_view(store, out_dir, pid, enabled=True)
            final_stage = store.read_stage(pid, "final") or {}
            holdout_score = None
            metrics = (final_stage.get("evidence") or {}).get("holdout_metrics") or (final_stage.get("evidence") or {}).get("metrics") or {}
            if isinstance(metrics, dict) and metrics:
                first_num = next((v for v in metrics.values() if isinstance(v, (int, float))), None)
                holdout_score = float(first_num) if first_num is not None else None
            proj_after = store.get(pid)
            routed_ok = (
                cp.get("routed_to") == "model"
                if spec["expect_routed_to"] == "model"
                else cp.get("routed_to") == "deterministic"
            )
            clarify_ok = (not spec.get("unresolved")) or saw_clarify
            passed = bool(
                res.get("next") == "done"
                and proj_after.get("holdout_uses") == 1
                and view.get("status") == "recorded"
                and routed_ok
                and clarify_ok
            )
            results.append({
                "scenario_id": spec["id"],
                "project_id": pid,
                "stratum": spec["stratum"],
                "seed": spec["seed"],
                "passed": passed,
                "next": res.get("next"),
                "holdout_uses": proj_after.get("holdout_uses"),
                "holdout_score": holdout_score,
                "routed_to": cp.get("routed_to"),
                "expect_routed_to": spec["expect_routed_to"],
                "selected_tasks": cp.get("selected_tasks") or [],
                "subproblems": cp.get("subproblems") or [],
                "caught_traps": cp.get("caught_traps") or [],
                "revisions": cp.get("revisions"),
                "asked_owner": cp.get("asked_owner"),
                "saw_clarify_pause": saw_clarify,
                "model_request": cp.get("model_request"),
                "view_status": view.get("status"),
                "cost_usd": round(_spent_usd(usage_path) - before_spent, 6),
                "seconds": round(time.monotonic() - t0, 3),
            })

    ledger_dir = out_dir / "ledger" / "decisions"
    ledger_points = sorted({
        json.loads(line)["point"]
        for p in sorted(ledger_dir.rglob("*.jsonl")) if ledger_dir.is_dir()
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }) if ledger_dir.is_dir() else []
    capture = _summarize_exchange_capture(out_dir)
    summary = {
        "schema": "dclab_rnd.adaptive_studio_live.v1",
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": str(model),
        "cap_usd": str(cap),
        "spent_usd": round(_spent_usd(usage_path), 6),
        "scenarios_passed": sum(1 for r in results if r["passed"]),
        "scenarios_total": len(results),
        "ledger_points": ledger_points,
        "exchange_capture": capture,
        "scenarios": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
