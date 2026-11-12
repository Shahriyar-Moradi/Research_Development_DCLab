"""Offline tests for task_planning_v3 protocol, dual-gate logic, Gateway capture, and harness."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dclab_rnd.agent_eval.task_planning.probe import offline_environment, run_probe
from dclab_rnd.agent_eval.task_planning_v3 import freeze, run_offline, verify_freeze
from dclab_rnd.agent_eval.task_planning_v3.gate import BudgetGateway, CASES, DollarCeiling, SEEDS, judge_v2
from dclab_rnd.models import FileUsage


def _sample_row(case, *, arm: str, seconds: float, correct: int = 1, completed: int = 0, recovered: int = 0,
                extra_prompts: int = 0, cost_eur: float = 0.0, **safety_overrides):
    safety = {
        "unauthorized_tool_calls": 0,
        "false_completion": 0,
        "holdout_violations": 0,
        "leakage_bypasses": 0,
        "repeated_non_idempotent_actions": 0,
        **safety_overrides,
    }
    return {
        "id": f"{case.id}:{SEEDS[0]}",
        "case_id": case.id,
        "seed": SEEDS[0],
        "stratum": case.stratum,
        "phase": case.phase,
        "scenario": case.scenario,
        "fault": case.fault,
        "expected": case.expected,
        "judged": True,
        "status": "complete" if completed else "blocked",
        "correct": correct,
        "completed": completed,
        "recovered": recovered,
        "proper_abstention": 1 if case.expected in ("blocked", "clarify") else None,
        "instrumented": True,
        "owner_evidence_checked": True,
        "safety": safety,
        "source_fingerprint": "a" * 64,
        "seconds": seconds,
        "compute_seconds": min(seconds, 1.0),
        "cost_eur": cost_eur,
        "extra_prompts": extra_prompts,
        "provider_requests": extra_prompts,
        "tokens": extra_prompts * 100,
        "confirmations": 0,
        "tool_calls": 6 if arm != "A" else 4,
        "valid_graph_rate": 1,
        "valid_plan_rate": 1,
        "task_coverage_rate": 1.0,
        "reexecution_avoided": 0,
    }


class TaskPlanningV3EvalTests(unittest.TestCase):
    def test_freeze_locks_v3_protocol_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = Path(tmp) / "task_planning_v3"
            doc = freeze(camp, live_model=None, cap_usd="3.50")
            self.assertEqual(doc["campaign_id"], "task_planning_v3")
            self.assertEqual(doc["protocol_file"], "docs/guides/TASK_PLANNING_V3_PROTOCOL.md")
            verify_freeze(camp)
            frozen_path = camp / "freeze_v1.json"
            body = json.loads(frozen_path.read_text())
            body["code_sha256"]["dclab_rnd/tasks/dispatch.py"] = "0" * 64
            frozen_path.write_text(json.dumps(body))
            with self.assertRaises(RuntimeError):
                verify_freeze(camp)

    def test_budget_gateway_sets_invented_tables_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            gw = BudgetGateway(FileUsage(Path(tmp) / "usage.jsonl"), DollarCeiling("0.50"), "qwen/qwen3.8-flash", home=Path(tmp))
            self.assertEqual(gw.invented_tables, "invented_accounts_v2")

    def test_judge_v2_applies_floor_and_keeps_strict_safety_and_cost_thresholds(self):
        probe = run_probe()
        scores = {"A": [], "B": [], "C": []}
        for case in CASES:
            is_rec = case.scenario in ("transient", "resume")
            is_comp = case.expected == "complete"
            a_ok = 0 if is_rec else 1
            a_sec = 0.005 if case.phase == "admission" else (0.02 if is_rec else 150.0)
            c_sec = 0.012 if case.phase == "admission" else (120.0 if is_comp else a_sec)
            scores["A"].append(_sample_row(case, arm="A", seconds=a_sec, correct=a_ok, completed=int(is_comp and not is_rec)))
            scores["B"].append(_sample_row(case, arm="B", seconds=1.5 if is_comp else a_sec, correct=1, completed=int(is_comp), recovered=int(is_rec)))
            scores["C"].append(
                _sample_row(
                    case,
                    arm="C",
                    seconds=c_sec,
                    correct=1,
                    completed=int(is_comp),
                    recovered=int(is_rec),
                    extra_prompts=7 if is_comp else 0,
                    cost_eur=0.005 if is_comp else 0.0,
                )
            )
        verdict = judge_v2(scores, configured_model=True, c_integrated=True, proposal_probe=probe)
        self.assertEqual(verdict["verdict"], "FAIL")  # strict v1 flags crashed-A 0.02s & 5ms->12ms jitter
        self.assertEqual(verdict["verdict_v2"], "PASS")
        self.assertTrue(verdict["promoted_v2"])

        # Prompt > 8 must STILL fail gate_v2 (never loosened).
        bad_prompts = {arm: [dict(r) for r in rows] for arm, rows in scores.items()}
        bad_prompts["C"][0]["extra_prompts"] = 9
        verdict_bad_prompts = judge_v2(bad_prompts, configured_model=True, c_integrated=True, proposal_probe=probe)
        self.assertEqual(verdict_bad_prompts["verdict_v2"], "FAIL")

        # Comparable baseline regression (> 2x on completed A) must STILL fail gate_v2.
        slow_comp = {arm: [dict(r) for r in rows] for arm, rows in scores.items()}
        slow_comp["A"][0]["seconds"] = 50.0
        slow_comp["C"][0]["seconds"] = 120.0
        verdict_slow = judge_v2(slow_comp, configured_model=True, c_integrated=True, proposal_probe=probe)
        self.assertEqual(verdict_slow["verdict_v2"], "FAIL")

    def test_offline_v3_harness_runs_with_zero_critical_violations(self):
        with offline_environment():
            res = run_offline(limit=4)
        self.assertEqual(res["cases_run"], 4)
        for arm in ("A", "B", "C"):
            self.assertEqual(len(res["scores"][arm]), 4)
            for row in res["scores"][arm]:
                self.assertEqual(row["correct"], 1)
                self.assertTrue(all(v == 0 for v in row["safety"].values()))


if __name__ == "__main__":
    unittest.main()
