"""34.1 and 34.4 on the delivery-fraud replica: a review decision stated as a person would (a fraud caught saves the
ride's fare, a review costs 2, at most 0.5% of the rides) runs through the engine; the final stage chooses it on the
out-of-fold scores, ranks rides by what each is worth, keeps to the capacity on the holdout and reads it once. The
campaign's gates judge made-up rows as they say."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy  # noqa: F401

    from dclab_rnd import decision_measure as dm
    from dclab_rnd import kinds

    MISSING = "" if kinds.get("rare_event").case(0) is not None else "the delivery-fraud replica is not part of this edition"
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class FraudDecisionTest(unittest.TestCase):
    def test_a_review_decision_on_the_fraud_replica_is_chosen_ranked_by_fare_and_kept_to_its_capacity(self):
        scratch = Path(tempfile.mkdtemp())
        with mock.patch.object(dm, "CAMPAIGN", scratch / "fraud_decision_test"), mock.patch.object(dm, "WORK", scratch / "work"):
            r = dm.run_one(0)
        self.assertEqual(r["decision_status"], "operating_point", r.get("warning"))
        self.assertIsNone(r["warning"])
        self.assertEqual(r["ranked_by"], "expected gain")  # a value per ride: ranked by what each is worth
        self.assertLessEqual(r["chosen_share"], dm.DECISION["capacity_share"])
        self.assertTrue(r["within_capacity"])
        self.assertIsNotNone(r["holdout_value_per_row"])
        self.assertEqual(len(r["prediction_interval"]), 2)
        self.assertEqual(r["holdout_uses"], 1)  # read once, by the final stage
        self.assertEqual(r["experiment_id"], "FDC-001")

    def test_the_gates(self):
        good = {"decision_status": "operating_point", "chosen_share": 0.004, "warning": None, "within_capacity": True,
                "inside_prediction_interval": True}
        rows = [dict(good) for _ in dm.SEEDS]
        self.assertEqual(dm.judge(rows)["verdict"], "pass")
        two_out = [dict(good, inside_prediction_interval=i >= 2) for i in range(len(dm.SEEDS))]
        self.assertEqual(dm.judge(two_out)["verdict"], "pass")  # 6 of 8 inside
        three_out = [dict(good, inside_prediction_interval=i >= 3) for i in range(len(dm.SEEDS))]
        self.assertFalse(dm.judge(three_out)["gates"]["coverage"])
        refused = rows[:-1] + [dict(good, warning="a value column with missing values", chosen_share=None)]
        self.assertFalse(dm.judge(refused)["gates"]["chosen"])  # a refusal is no choice
        over = rows[:-1] + [dict(good, within_capacity=False)]
        self.assertFalse(dm.judge(over)["gates"]["capacity"])
        self.assertTrue(dm.judge(rows[:3])["verdict"].startswith("not yet"))


if __name__ == "__main__":
    unittest.main()
