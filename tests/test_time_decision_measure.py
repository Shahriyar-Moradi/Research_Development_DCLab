"""34.1 on a time split: the churn replica through the engine with a share capacity (one pool) and a daily one (day by
day), and the campaign's gates on made-up rows."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy  # noqa: F401

    from dclab_rnd import time_decision_measure as tm

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class TimeDecisionTest(unittest.TestCase):
    def test_each_arm_runs_on_a_time_split_and_keeps_to_its_capacity(self):
        scratch = Path(tempfile.mkdtemp())
        with mock.patch.object(tm, "CAMPAIGN", scratch / "c"), mock.patch.object(tm, "WORK", scratch / "w"):
            pooled, daily = tm.run_one(0, "pooled"), tm.run_one(0, "daily")
        for r in (pooled, daily):
            self.assertEqual(r["decision_status"], "operating_point", r.get("warning"))
            self.assertTrue(r["within_capacity"])
            self.assertEqual(r["holdout_uses"], 1)
            self.assertEqual(len(r["prediction_interval"]), 2)
        self.assertEqual((pooled["split"], daily["split"]), ("time", "time"))
        for r in (pooled, daily):
            self.assertGreater(r["rates"]["holdout_positive_rate"], 0)
            self.assertEqual(len(r["rates"]["fold_values"]), 3)  # each fold's value, beside the rates (the review)
        self.assertIsNone(pooled["days"])  # a share: one pool
        self.assertEqual(daily["days"]["most_in_a_day"], 1)  # a time split's holdout, day by day
        self.assertIn("days of the holdout's sizes", daily["prediction_interval_method"])
        self.assertIn("nearest the holdout", pooled["prediction_interval_method"])  # 0.118.0: the latest fold's rows
        for r in (pooled, daily):  # the engine flags the latest fold: TimeSeriesSplit(3) of 2,400 rows, its last 600 out-of-fold rows
            self.assertEqual(r["latest_fold_rows"], 600)
            self.assertTrue(r["latest_fold_is_the_last_block"])
            self.assertEqual(len(r["all_folds_interval"]), 2)  # the paired reading from every fold, on the same seed
        self.assertTrue((scratch / "c" / "results").exists())

    def test_the_gates_and_the_next_step_fixed_with_them(self):
        good = {"decision_status": "operating_point", "chosen_share": 0.1, "warning": None, "within_capacity": True,
                "inside_prediction_interval": True, "z": 0.0, "miss_side": "inside", "split": "time", "holdout_uses": 1}
        n = len(tm.SEEDS)
        both = [dict(good, z=0.9 if i % 2 else -0.9) for i in range(n)]
        self.assertEqual(tm.judge(both)["verdict"], "pass")
        self.assertEqual(tm.judge(both)["next"], "keep the latest fold's interval on time splits, consistent with v2's daily miss being the drift")
        above = [dict(good, z=1.0) for _ in range(n)]  # every seed realises above the centre
        verdict = tm.judge(above)
        self.assertFalse(verdict["gates"]["centre"])
        self.assertEqual(verdict["next"], "read the seeds")  # no fold values recorded: nothing says the folds' models differ
        climbing = [dict(r, rates={"fold_values": [4.0, 4.5, 5.0]}) for r in above]
        self.assertEqual(tm.judge(climbing)["next"], "score the out-of-fold rows with models of the final size")
        below = [dict(good, z=-1.0) for _ in range(n)]
        self.assertEqual(tm.judge(below)["next"], "correct the optimism of choosing the model")
        few = [dict(good, chosen_share=0.0, inside_prediction_interval=None, z=None)] * (n - 10) + [dict(good)] * 10
        self.assertTrue(tm.judge(few)["verdict"].startswith("inconclusive"))
        self.assertTrue(tm.judge(both[:5])["verdict"].startswith("not yet"))
        other = both[:-1] + [dict(good, split="stratified")]  # a seed whose proposal split otherwise never counts as time
        self.assertFalse(tm.judge(other)["gates"]["time_split"])
        over = both[:-1] + [dict(good, within_capacity=False)]
        self.assertFalse(tm.judge(over)["gates"]["capacity"])
        twice = both[:-1] + [dict(good, holdout_uses=2)]
        self.assertFalse(tm.judge(twice)["gates"]["capacity"])
        narrow = [dict(good, z=2.0 if i % 2 else -2.0, inside_prediction_interval=i % 5 != 0,
                       miss_side="inside" if i % 5 else ("above" if i % 2 else "below")) for i in range(n)]
        self.assertEqual(tm.judge(narrow)["next"], "add blocks of days to the samples")
        wide = [dict(good, z=0.3 if i % 2 else -0.3) for i in range(n)]
        self.assertEqual(tm.judge(wide)["next"], "sample the latest two folds")

    def test_a_failed_run_is_recorded_and_counted(self):
        import json

        scratch = Path(tempfile.mkdtemp())
        with mock.patch.object(tm, "CAMPAIGN", scratch / "c"):
            tm.failed(tm.SEEDS[0], "daily", ValueError("no rows"))
            (scratch / "c" / "results" / "TDC-900_decision.json").write_text(json.dumps({"id": "TDC-900"}))  # never read as a run
            out = tm.report()
        self.assertEqual(out["arms"]["daily"]["seeds"], 1)
        self.assertFalse(out["arms"]["daily"]["gates"]["chosen"])

    def test_the_case_is_the_binary_kinds_apart_from_the_drift(self):
        """The review of v3: the campaign rebuilds the churn case with the replica's drift set; nothing else may differ."""
        from dclab_core.replicas import subscription_churn

        from dclab_rnd import kinds

        self.assertTrue(subscription_churn.generate(4).subscribers.equals(subscription_churn.generate(4, drift=0.3).subscribers))
        template = kinds.get("binary").case(4)
        replica = subscription_churn.generate(4)
        self.assertTrue(template.table.equals(replica.subscribers))  # the kind's case is the replica's table, as rebuilt
        self.assertEqual(template.solution, {"positive_label": "yes"})
        self.assertEqual(template.facts, replica.facts)

    def test_numbers_never_collide_between_arms(self):
        numbers = {tm._number(s, a) for s in tm.SEEDS for a in tm.ARMS}
        self.assertEqual(len(numbers), len(tm.SEEDS) * len(tm.ARMS))
        self.assertNotIn("TDC-900", numbers)  # the verdict's own number


if __name__ == "__main__":
    unittest.main()
