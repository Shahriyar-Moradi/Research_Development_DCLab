"""The planning practice set (2026-11-19): every planted trap read into where it was caught or lost, on the exam's own
builder, with no model."""
import hashlib
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TABLES = textwrap.dedent('''
    import numpy as np
    import pandas as pd

    def make_inv1(seed):
        rng = np.random.default_rng(seed)
        n = 400
        at = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 200, n)), unit="D")
        spend = rng.uniform(5, 100, n)
        churned = (rng.random(n) < 1 / (1 + np.exp(-(spend - 50) / 15))).astype(int)
        return {"orders.csv": pd.DataFrame({"order_id": np.arange(n), "created_at": at, "shop_id": rng.integers(0, 40, n), "spend": spend,
                                            "closed_at": at + pd.to_timedelta(rng.uniform(1, 30, n), unit="D"), "churned": churned})}
''')
CASES = {"seeds": [7], "cases": [{"id": "inv1", "brief": "Which orders lead to churn?", "known_facts": {"target": "churned"},
                                  "enforceable_traps": [{"kind": "late_column", "column": "closed_at"}, {"kind": "time_split", "column": "created_at"}]}]}


def folder():
    d = Path(tempfile.mkdtemp())
    (d / "tables.py").write_text(TABLES, encoding="utf-8")
    (d / "cases.json").write_text(json.dumps(CASES), encoding="utf-8")
    (d / "cases.sha256").write_text("".join(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()} {n}\n" for n in ("cases.json", "tables.py")), encoding="utf-8")
    return d


class PracticeTests(unittest.TestCase):
    def test_each_trap_says_where_it_was_caught_or_lost(self):
        from dclab_rnd.agent_eval.planning_gate_v2.practice import explain

        row = {"case": "c", "seed": 1, "forbidden_before": ["closed_at"]}
        stages = {"data": {"evidence": {"holdout_split": "random 20%"}}}
        plan = {"caught_traps": ["grouped_split_shop_id"], "enforcement": {"asked": [{"kind": "time_split", "value": "created_at"}]}}
        cols = ["closed_at", "created_at", "shop_id", "acct"]
        read = lambda kind, column, done: explain({"kind": kind, "column": column, "enforced": done}, row, {"forbidden": ["closed_at"]}, plan, stages, cols)
        self.assertEqual(read("late_column", "closed_at", True)["where"], "enforced before the plan")
        self.assertEqual(read("time_split", "created_at", False)["where"], "asked, not applied")
        self.assertEqual(read("group_split", "shop_id", False)["where"], "caught, not enforced")
        lost = read("group_split", "acct", False)
        self.assertEqual(lost["where"], "missed")
        self.assertIn("random 20%", lost["did"])

    def test_a_case_is_built_with_no_model_and_its_traps_are_read(self):
        from dclab_rnd.agent_eval.planning_gate_v2 import practice
        from dclab_rnd.expansion.runner import fast_profile

        with mock.patch.object(practice, "FOLDER", folder()), fast_profile():
            row = practice._one((CASES["cases"][0], 7, tempfile.mkdtemp()))
        self.assertNotIn("error", row)
        self.assertEqual([t["kind"] for t in row["traps"]], ["late_column", "time_split"])
        late = row["traps"][0]
        self.assertTrue(late["enforced"])  # the standard plan keeps closed_at out
        self.assertEqual(row["solution"]["target"], "churned")
        s = practice.summary([row])
        self.assertEqual((s["projects"], s["planted"], s["crashed"]), (1, 2, 0))


if __name__ == "__main__":
    unittest.main()
