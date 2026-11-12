"""Offline tests for task_planning_v2 protocol, budget ceilings, and Arm A/B/C harness."""
from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dclab_rnd.agent_eval.task_planning.probe import offline_environment
from dclab_rnd.agent_eval.task_planning_v2 import _load_price_table, freeze, run_offline, verify_freeze
from dclab_rnd.agent_eval.task_planning_v2.gate import BudgetGateway, CASES, DollarCeiling, GATE
from dclab_rnd.models import FileUsage, prices


class TaskPlanningV2EvalTests(unittest.TestCase):
    def test_case_inventory_meets_tp06_release_floors(self):
        completion = {c.id for c in CASES if c.expected == "complete"}
        recovery = {c.id for c in CASES if c.scenario in ("transient", "resume")}
        self.assertGreaterEqual(len(CASES), GATE["min_matched_cases"])
        self.assertEqual(len({c.id for c in CASES}), len(CASES))
        self.assertGreaterEqual(len(completion), GATE["min_completion_groups"])
        self.assertGreaterEqual(len(recovery), GATE["min_recovery_groups"])

    def test_dollar_ceiling_blocks_excess_and_over_4_usd(self):
        with self.assertRaises(ValueError):
            DollarCeiling("4.01")
        ceiling = DollarCeiling("0.10")
        ceiling.reserve(0.06)
        for bad in (0.05, 0, -0.01, float("nan"), None):
            with self.assertRaises(ValueError):
                ceiling.reserve(bad)
        self.assertEqual(ceiling.committed, Decimal("0.06"))

    def test_freeze_locks_hashes_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = Path(tmp) / "task_planning_v2"
            freeze(camp, live_model=None, cap_usd="3.80")
            verify_freeze(camp)
            frozen_path = camp / "freeze_v1.json"
            body = json.loads(frozen_path.read_text())
            body["case_sha256"] = "0" * 64
            frozen_path.write_text(json.dumps(body))
            with self.assertRaises(RuntimeError):
                verify_freeze(camp)

    def test_offline_scripted_sample_pairs_a_b_c_with_zero_critical_violations(self):
        with offline_environment():
            res = run_offline(limit=4)
        self.assertEqual(res["cases_run"], 4)
        for arm in ("A", "B", "C"):
            rows = res["scores"][arm]
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row["correct"], 1)
                self.assertTrue(all(v == 0 for v in row["safety"].values()))
        self.assertTrue(all(r["model_graph_admitted"] for r in res["scores"]["C"]))

    def test_load_price_table_returns_price_objects_for_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / "price_snapshot_usd.json"
            snap.write_text(json.dumps({"prices": {"qwen/qwen3.8-flash": {"input": 0.15, "output": 0.47, "updated": "openrouter_live_snapshot"}}}))
            table = _load_price_table(snap)
            self.assertIsInstance(table["qwen/qwen3.8-flash"], prices.Price)
            with patch.dict(prices.PRICES, table):
                est = prices.estimate("qwen/qwen3.8-flash", False, 1500, 800)
            self.assertGreater(est, 0.0)


if __name__ == "__main__":
    unittest.main()
