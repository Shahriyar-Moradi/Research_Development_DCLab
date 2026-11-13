"""Tests for P1 / TP-08 task_planning_v4 benchmark, sealed SHA verification, headroom check, and gate."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval.task_planning_v4 import (  # noqa: E402
    GATE_V4,
    judge_v4,
    load_split,
    run_headroom_check,
    run_legacy_baseline_case,
    run_offline,
    verify_headroom,
    verify_sealed,
)
from dclab_rnd.agent_eval.task_planning_v4.gate import score_proposal  # noqa: E402


class TaskPlanningV4BenchmarkTests(unittest.TestCase):
    def test_sealed_set_is_locked_and_verified(self):
        sha = verify_sealed()
        self.assertEqual(len(sha), 64)
        dev_cases, dev_seeds, _ = load_split("dev")
        sealed_cases, sealed_seeds, sealed_sha = load_split("sealed")
        self.assertEqual(sha, sealed_sha)
        self.assertGreaterEqual(len(dev_cases), 18)
        self.assertGreaterEqual(len(sealed_cases), 12)
        self.assertEqual(len(dev_seeds), 2)
        self.assertEqual(len(sealed_seeds), 2)

    def test_headroom_check_proves_legacy_baseline_leaves_room_to_win(self):
        rep = run_headroom_check("dev")
        self.assertTrue(rep["has_headroom"])
        self.assertLessEqual(rep["mean_required_coverage"], GATE_V4["headroom_max_baseline_coverage"])
        self.assertLessEqual(rep["mean_trap_catch_rate"], GATE_V4["headroom_max_baseline_trap_catch"])

    def test_headroom_check_rejects_saturated_baseline(self):
        cases, seeds, _ = load_split("dev")
        perfect_rows = [
            score_proposal(
                case,
                seed,
                {
                    "selected_tasks": case.expected_tasks,
                    "caught_traps": case.expected_traps,
                    "subproblems": case.expected_subproblems,
                    "asked_owner": "ask_owner" in case.expected_tasks,
                },
            )
            for case in cases
            for seed in seeds
        ]
        with self.assertRaises(ValueError):
            verify_headroom(perfect_rows)

    def test_gate_passes_valid_gain_and_blocks_spurious_questions_or_violations(self):
        cases, seeds, _ = load_split("dev")
        baseline = [run_legacy_baseline_case(c, s) for c in cases for s in seeds]
        candidate = [
            score_proposal(
                c,
                s,
                {
                    "selected_tasks": c.expected_tasks,
                    "caught_traps": c.expected_traps,
                    "subproblems": c.expected_subproblems,
                    "asked_owner": "ask_owner" in c.expected_tasks,
                    "extra_prompts": 3 if c.needs_model_planning else 0,
                    "cost_usd": 0.004 if c.needs_model_planning else 0.0,
                },
            )
            for c in cases
            for s in seeds
        ]
        ok = judge_v4({"B0": baseline, "C": candidate})
        self.assertEqual(ok["verdict"], "PASS", ok)
        self.assertTrue(ok["promoted"])

        # Asking owner unnecessarily on a standard clear case violates safety/forbidden_actions.
        spurious = [dict(r) for r in candidate]
        spurious_case = next(c for c in cases if not c.needs_model_planning)
        spurious[-1] = score_proposal(
            spurious_case,
            seeds[-1],
            {
                "selected_tasks": [*spurious_case.expected_tasks, "ask_owner"],
                "caught_traps": [],
                "subproblems": spurious_case.expected_subproblems,
                "asked_owner": True,
            },
        )
        bad = judge_v4({"B0": baseline, "C": spurious})
        self.assertEqual(bad["verdict"], "FAIL")
        self.assertGreater(bad["critical"]["C"]["forbidden_actions"], 0)

    def test_offline_full_dev_suite_passes_v4_gate_with_real_replicas(self):
        doc = run_offline(split="dev", limit=None)
        self.assertEqual(doc["cases_run"], 20)
        j = doc["judgment"]
        self.assertIsNotNone(j)
        self.assertEqual(j["verdict"], "PASS", j)
        self.assertTrue(j["promoted"])
        self.assertGreater(j["comparisons"]["required_coverage_hard"]["difference"], 0.25)
        self.assertGreater(j["comparisons"]["trap_catch_hard"]["difference"], 0.25)
        self.assertGreaterEqual(j["comparisons"]["holdout_score"]["difference"], 0.0)
        self.assertTrue(all(r.get("task_graph_fingerprint") for r in doc["scores"]["C"]))


if __name__ == "__main__":
    unittest.main()
