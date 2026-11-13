"""Offline integration tests for P2–P7 runtime wiring: TaskDispatcher, SolutionTeam, view, and catalog_lake."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dclab_rnd import experience, ledger
from dclab_rnd.agent_eval.task_planning.probe import offline_environment
from dclab_rnd.agent_eval.task_planning.runner import _owner_complete
from dclab_rnd.expansion import runner
from dclab_rnd.models import FileUsage, Gateway
from dclab_rnd.orchestra.checkpoints import open_checkpointer
from dclab_rnd.orchestra.solution import SolutionTeam
from dclab_rnd.studio import ProjectStore, data, graph
from dclab_rnd.tasks.catalog import replan_from_evidence
from dclab_rnd.tasks.catalog_lake import build_planning_gold_set
from dclab_rnd.tasks.dispatch import TaskDispatcher
from dclab_rnd.tasks.host import planning_host
from dclab_rnd.tasks.view import task_plan_view


class _HybridCatalogServer:
    """Handles TP-02/TP-03 decompose/plan_task AND P2-P6 problem_planner + P4 stage_evidence replanning."""

    thinking = None

    def __init__(self, *, ask_owner_on_first: bool = False):
        self.calls: list[str] = []
        self.ask_owner_on_first = ask_owner_on_first

    def complete(self, messages, tools=None, max_tokens=None, **options):
        body = json.loads(messages[-1]["content"])
        if "deterministic_graph" in body:
            self.calls.append("decompose")
            baseline = body["deterministic_graph"]
            leaves = [n for n in baseline["nodes"] if n["kind"] != "group"]
            answer = {
                "version": "tp02.v1",
                "baseline_fingerprint": baseline["fingerprint"],
                "groups": [{"id": "group_0", "parent_id": None, "members": [n["id"] for n in leaves]}],
                "dependencies": [{"task_id": n["id"], "requires": n["dependencies"]} for n in leaves],
            }
        elif "baseline" in body:
            baseline = body["baseline"]
            self.calls.append(f"plan:{baseline.get('task_id')}")
            answer = {
                "version": "tp03.v1",
                "baseline_fingerprint": baseline["fingerprint"],
                **{k: baseline[k] for k in ("actions", "budget", "policy")},
            }
        elif "stage_evidence" in body and "current_plan" in body:
            self.calls.append("catalog_replan")
            cites = list((body.get("evidence_sources") or {}).keys())[:1]
            answer = {
                "decomposition": body["current_plan"].get("decomposition") or {
                    "decision": body.get("brief", "churn"),
                    "target": "churn",
                    "unit": "customer",
                    "horizon_days": 30,
                    "success_metric": "pr_auc",
                    "subproblems": [{"id": "main", "kind": "classification", "objective": "churn", "target": "churn"}],
                    "risks": [{"trap_kind": "post_outcome_leakage", "column": "closure_recorded_after_review", "severity": "high"}],
                    "unknowns": [],
                },
                "calls": [
                    {"task_type": "leakage_probe", "params": {"column": "closure_recorded_after_review"}, "purpose": "Probe post-outcome flag", "citations": cites},
                    {"task_type": "baseline", "params": {"metric": "pr_auc"}, "purpose": "Leakage-free baseline", "citations": cites},
                ],
                "caught_traps": ["post_outcome_closure_recorded_after_review"],
            }
        elif "suggested_followups" in body:
            self.calls.append("catalog_inline_followup")
            cites = list((body.get("evidence_sources") or {}).keys())[:1]
            answer = {
                "calls": [
                    {"task_type": "leakage_probe", "params": {"column": "closure_recorded_after_review"}, "purpose": "Probe post-outcome flag", "citations": cites},
                    {"task_type": "baseline", "params": {"metric": "roc_auc"}, "purpose": "Train clean baseline", "citations": cites},
                ],
                "caught_traps": ["post_outcome_closure_recorded_after_review"],
            }
        else:
            self.calls.append("catalog_plan")
            cites = list((body.get("evidence_sources") or {}).keys())[:1]
            tables = body.get("tables") or {}
            first_tbl = next(iter(tables), "main")
            cols = tables.get(first_tbl) or []
            calls = [
                {"task_type": "time_split_check", "params": {"column": "signup_date" if "signup_date" in cols else ""}, "purpose": "Respect time ordering", "citations": cites},
                {"task_type": "profile_column", "params": {"table": first_tbl, "column": "closure_recorded_after_review" if "closure_recorded_after_review" in cols else cols[0]}, "purpose": "Profile suspect field", "citations": cites},
                {"task_type": "baseline", "params": {"metric": "roc_auc"}, "purpose": "Initial baseline", "citations": cites},
            ]
            if self.ask_owner_on_first:
                calls.append({"task_type": "ask_owner", "params": {"field": "horizon", "question": "Confirm 30-day churn horizon?", "blocking": True}, "purpose": "Verify horizon", "citations": cites})
            answer = {
                "decomposition": {
                    "decision": str(body.get("brief") or "Decide retention actions"),
                    "target": "churn",
                    "unit": "customer",
                    "horizon_days": 30,
                    "success_metric": "roc_auc",
                    "subproblems": [{"id": "main", "kind": "classification", "objective": "Predict 30d churn", "target": "churn"}],
                    "risks": [],
                    "unknowns": [],
                },
                "calls": calls,
                "caught_traps": ["temporal_drift"],
            }
        return {
            "content": json.dumps(answer),
            "finish_reason": "stop",
            "tool_calls": [],
            "thinking": [{"text": "SCRIPTED: catalog runtime verification."}],
            "usage": {"input_tokens": 120, "output_tokens": 90},
        }


def _make_runtime_project(store: ProjectStore, seed: int = 17, *, include_unresolved_leak: bool = True, unresolved_target: bool = False) -> str:
    proj = store.create("Runtime Catalog Project", "telecom", "Predict 30-day customer churn with temporal cutoff before outreach.")
    pid = proj["id"]
    rng = np.random.default_rng(seed)
    n = 160
    tenure = rng.integers(1, 72, n)
    charges = rng.uniform(15, 110, n)
    p = 1 / (1 + np.exp(-(-0.8 - 0.02 * tenure + 0.01 * charges)))
    churn = (rng.random(n) < p).astype(int)
    df = pd.DataFrame(
        {
            "customer_id": [f"c_{i}" for i in range(n)],
            "tenure": tenure,
            "monthly_charge": charges,
            "churn": np.where(churn == 1, "yes", "no"),
            **({"signup_date": pd.date_range("2025-01-01", periods=n, freq="D").astype(str), "closure_recorded_after_review": np.where(rng.random(n) < np.where(churn == 1, 0.78, 0.22), 1, 0)} if include_unresolved_leak else {}),
        }
    )
    df.to_csv(store.data_dir(pid) / "customers.csv", index=False)
    data.attach_data(store, pid, "customers.csv")
    proj = store.get(pid)
    proj["solution"] = {
        "target": "churn",
        "task": "binary",
        "positive_label": "yes",
        "prediction_moment": "At monthly review cutoff prior to outreach.",
        "forbidden": [],
        "identifiers": ["customer_id"],
        "metric": "roc_auc",
        "labels": "complete",
    }
    if unresolved_target:
        proj["catalog_hints"] = {"prediction_moment_unresolved": True}
    proj["settings"].update(quick=True, max_rows=500, private=False)
    proj["policy"] = {"require_solution_signoff": True, "require_holdout_approval": True}
    store.save(proj)
    graph.approve_gate(store, pid, "solution", by="tester", reason="Runtime test")
    return pid


class TaskCatalogRuntimeTests(unittest.TestCase):
    def test_solution_team_records_catalog_plan_ledger_and_view(self):
        server = _HybridCatalogServer()
        with tempfile.TemporaryDirectory(prefix="cat-rt-") as tmp, offline_environment(), ExitStack() as stack:
            home = Path(tmp)
            prev_log, prev_ledger = experience.current_log(), ledger.current_ledger()
            experience.install(experience.FileLog(home / "experience", "cat-rt"))
            file_ledger = ledger.FileLedger(home / "ledger", "cat-rt")
            ledger.install(file_ledger)
            stack.callback(experience.install, prev_log)
            stack.callback(ledger.install, prev_ledger)
            stack.enter_context(runner.fast_profile())
            saver = stack.enter_context(open_checkpointer(home, database_url=""))
            store = ProjectStore(home / "projects")
            pid = _make_runtime_project(store, include_unresolved_leak=True)
            gw = Gateway(FileUsage(home / "usage.jsonl"), transport=lambda tier, purpose: server, sleep=lambda _: None)

            with planning_host(name="cat-rt"):
                team = SolutionTeam(store, saver, home=home, task_execution=True, admit_proposals=True, gateway=gw)
                res = team.run(pid)
                self.assertEqual(res.get("status"), "paused")
                seen_gates = []
                while res.get("status") == "paused" and len(seen_gates) < 4:
                    gate = (res.get("asks") or {}).get("gate")
                    seen_gates.append(gate)
                    # On audit gate keep column (False) to continue through holdout without invalidating source fingerprint
                    res = team.run(pid, resume=(False if gate == "audit" else True))
                self.assertIn("holdout", seen_gates)

            self.assertEqual(res.get("next"), "done")
            self.assertTrue(_owner_complete(store, pid))
            self.assertIn("catalog_plan", server.calls)

            disp = TaskDispatcher(store, saver)
            st = disp.state(pid)
            self.assertIn("catalog_plan", st)
            self.assertEqual(st["catalog_plan"]["routed_to"], "model")
            self.assertIn("post_outcome_closure_recorded_after_review", st["catalog_plan"]["caught_traps"])

            # Verify ledger recorded map.size, problem.family, workflow.template
            records = [
                json.loads(line)
                for p in sorted((home / "ledger" / "decisions").rglob("*.jsonl"))
                for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            points = {r["point"] for r in records}
            self.assertTrue({"map.size", "problem.family", "workflow.template"} <= points)

            # Verify read-only view exposes allowlisted catalog_plan metadata
            view_doc = task_plan_view(store, home, pid, enabled=True)
            self.assertEqual(view_doc["status"], "recorded")
            self.assertIn("catalog_plan", view_doc)
            self.assertEqual(view_doc["catalog_plan"]["routed_to"], "model")

    def test_clean_single_table_uses_deterministic_p6_router(self):
        server = _HybridCatalogServer()
        with tempfile.TemporaryDirectory(prefix="cat-det-") as tmp, offline_environment(), ExitStack() as stack:
            home = Path(tmp)
            prev_log, prev_ledger = experience.current_log(), ledger.current_ledger()
            experience.install(experience.FileLog(home / "experience", "cat-det"))
            ledger.install(None)
            stack.callback(experience.install, prev_log)
            stack.callback(ledger.install, prev_ledger)
            stack.enter_context(runner.fast_profile())
            saver = stack.enter_context(open_checkpointer(home, database_url=""))
            store = ProjectStore(home / "projects")
            pid = _make_runtime_project(store, include_unresolved_leak=False)
            gw = Gateway(FileUsage(home / "usage.jsonl"), transport=lambda tier, purpose: server, sleep=lambda _: None)

            with planning_host(name="cat-det"):
                team = SolutionTeam(store, saver, home=home, task_execution=True, admit_proposals=True, gateway=gw)
                res = team.run(pid)
                self.assertEqual(res.get("status"), "paused")
                res = team.run(pid, resume=True)

            self.assertEqual(res.get("next"), "done")
            self.assertNotIn("catalog_plan", server.calls)
            st = TaskDispatcher(store, saver).state(pid)
            self.assertEqual(st["catalog_plan"]["routed_to"], "deterministic")
            self.assertEqual(st["catalog_plan"]["cost_usd"], 0.0)

    def test_blocking_ask_owner_pauses_and_resumes_cleanly(self):
        server = _HybridCatalogServer(ask_owner_on_first=True)
        with tempfile.TemporaryDirectory(prefix="cat-ask-") as tmp, offline_environment(), ExitStack() as stack:
            home = Path(tmp)
            prev_log, prev_ledger = experience.current_log(), ledger.current_ledger()
            experience.install(experience.FileLog(home / "experience", "cat-ask"))
            ledger.install(None)
            stack.callback(experience.install, prev_log)
            stack.callback(ledger.install, prev_ledger)
            saver = stack.enter_context(open_checkpointer(home, database_url=""))
            store = ProjectStore(home / "projects")
            pid = _make_runtime_project(store, include_unresolved_leak=True, unresolved_target=True)
            gw = Gateway(FileUsage(home / "usage.jsonl"), transport=lambda tier, purpose: server, sleep=lambda _: None)

            disp = TaskDispatcher(store, saver, gateway=gw, enabled=True, admit_proposals=True)
            with planning_host(name="cat-ask"):
                team = SolutionTeam(store, saver, home=home, task_execution=True, admit_proposals=True, gateway=gw)
                # Only test pre-stage pause on catalog_clarify when unresolved_target is set
                res = team.run(pid)
                self.assertEqual(res.get("status"), "paused")
                self.assertEqual(res.get("asks", {}).get("field"), "catalog_clarify")
                res2 = team.run(pid, resume="30-day churn confirmed")
                # Next pause should be audit or holdout approval after stages progress!
                self.assertIn(res2.get("status"), ("paused", "done"))
            st = disp.state(pid)
            self.assertTrue(st["catalog_plan"]["owner_confirmed"])

    def test_replan_from_evidence_merges_new_traps_within_limit(self):
        server = _HybridCatalogServer()
        with tempfile.TemporaryDirectory(prefix="cat-rep-") as tmp, offline_environment():
            home = Path(tmp)
            gw = Gateway(FileUsage(home / "usage.jsonl"), transport=lambda tier, purpose: server, sleep=lambda _: None)
            current = {
                "selected_tasks": ["baseline"],
                "calls": [{"task_type": "baseline", "params": {"metric": "roc_auc"}, "reason": "initial"}],
                "caught_traps": [],
                "revisions": 1,
                "decomposition": {
                    "decision": "Predict churn",
                    "target": "churn",
                    "unit": "customer",
                    "horizon_days": 30,
                    "success_metric": "roc_auc",
                    "subproblems": [{"id": "main", "kind": "classification", "objective": "churn", "target": "churn"}],
                    "risks": [],
                    "open_unknowns": [],
                },
            }
            out = replan_from_evidence(
                "Predict churn",
                {"main": ["customer_id", "closure_recorded_after_review", "churn"]},
                {"target": "churn"},
                current,
                [{"task_type": "profile_column", "findings": ["suspect"], "traps": ["post_outcome_closure_recorded_after_review"]}],
                gateway=gw,
            )
            self.assertTrue(out["revised"])
            self.assertEqual(out["plan"]["revisions"], 2)
            self.assertIn("post_outcome_closure_recorded_after_review", out["plan"]["caught_traps"])
            # Second attempt respects max_revisions=2
            out2 = replan_from_evidence("Predict churn", {"main": ["churn"]}, {"target": "churn"}, out["plan"], [], gateway=gw)
            self.assertFalse(out2["revised"])
            self.assertEqual(out2["reason"], "revision_limit")

    def test_catalog_lake_excludes_sealed_and_builds_gold_set(self):
        with tempfile.TemporaryDirectory(prefix="cat-lake-") as tmp:
            root = Path(tmp)
            ds_dir = root / "training_records" / "datasets"
            ds_dir.mkdir(parents=True)
            res_dir = root / "evidence" / "campaigns" / "task_planning_v4" / "results"
            res_dir.mkdir(parents=True)
            # Load real sealed case ID if present to assert exclusion
            from dclab_rnd.tasks.catalog_lake import _sealed_case_ids

            sealed_ids = _sealed_case_ids()
            sealed_id = next(iter(sealed_ids), "sealed_dummy_case")
            dev_rows = [
                {
                    "schema": "dclab_rnd.tp08_sft_example.v1",
                    "case_id": "dev_sample_case",
                    "seed": 0,
                    "stratum": "leakage_trap",
                    "required_coverage": 1.0,
                    "trap_catch_rate": 1.0,
                    "composite_kind_match": 1.0,
                    "holdout_score": 0.88,
                    "messages": [{"role": "user", "content": "{}"}],
                    "reply": {"content": "{}"},
                    "reasoning": [{"text": "drop post-outcome col"}],
                },
                {
                    "schema": "dclab_rnd.tp08_sft_example.v1",
                    "case_id": sealed_id,
                    "seed": 0,
                    "stratum": "leakage_trap",
                    "required_coverage": 1.0,
                    "trap_catch_rate": 1.0,
                    "composite_kind_match": 1.0,
                    "holdout_score": 0.90,
                    "messages": [{"role": "user", "content": "{}"}],
                    "reply": {"content": "{}"},
                    "reasoning": [{"text": "MUST BE EXCLUDED"}],
                },
            ]
            (ds_dir / "task_planning_v4_dev.jsonl").write_text("\n".join(json.dumps(r) for r in dev_rows) + "\n", encoding="utf-8")
            (res_dir / "TP4-001_dev_live.json").write_text(
                json.dumps({"scores": {"B0": [{"id": "dev_sample_case:0", "holdout_score": 0.65}], "B1": [{"id": "dev_sample_case:0", "holdout_score": 0.82}], "C": [{"id": "dev_sample_case:0", "holdout_score": 0.88}]}}),
                encoding="utf-8",
            )
            manifest = build_planning_gold_set(root=root)
            self.assertEqual(manifest["total"], 1)
            self.assertEqual(manifest["positive"], 1)
            self.assertTrue(manifest["sealed_excluded"])
            written = [json.loads(l) for l in Path(manifest["dataset_path"]).read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0]["case_id"], "dev_sample_case")
            self.assertAlmostEqual(written[0]["holdout_delta_vs_b0"], 0.23, places=3)


if __name__ == "__main__":
    unittest.main()
