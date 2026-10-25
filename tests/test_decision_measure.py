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
        self.assertEqual(r["experiment_id"], "FDC-000")

    def test_the_gates(self):
        good = {"decision_status": "operating_point", "chosen_share": 0.004, "warning": None, "within_capacity": True,
                "inside_prediction_interval": True, "holdout_actions": 16, "z": 0.0, "miss_side": "inside"}
        n = len(dm.SEEDS)
        rows = [dict(good) for _ in dm.SEEDS]
        verdict = dm.judge(rows)
        self.assertEqual((verdict["verdict"], verdict["next"]), ("pass", "keep the interval"))
        miss = lambda i: dict(good, inside_prediction_interval=False, z=-2.0 if i % 2 else 2.0, miss_side="below" if i % 2 else "above")
        six_out = [miss(i) if i < 6 else dict(good) for i in range(n)]  # 24 of 30 inside: 80%
        self.assertEqual(dm.judge(six_out)["verdict"], "pass")
        seven_out = [miss(i) if i < 7 else dict(good) for i in range(n)]
        self.assertFalse(dm.judge(seven_out)["gates"]["coverage"])
        nothing = rows[:-1] + [dict(good, chosen_share=0.0, holdout_actions=None, within_capacity=None, inside_prediction_interval=None, z=None)]
        self.assertEqual(dm.judge(nothing)["verdict"], "pass")  # no review paid on a seed: a choice, not a capacity failure (v1's flaw)

    def test_a_seed_that_chose_to_act_but_acted_on_no_holdout_row_still_counts(self):
        """The review of v2: such a seed realises 0 below its interval; dropping it would bias the count toward a pass."""
        good = {"decision_status": "operating_point", "chosen_share": 0.004, "warning": None, "within_capacity": True,
                "inside_prediction_interval": True, "holdout_actions": 16, "z": 0.0, "miss_side": "inside"}
        rows = ([dict(good)] * 23 + [dict(good, inside_prediction_interval=False, z=2.0, miss_side="above")] * 5
                + [dict(good, holdout_actions=0, inside_prediction_interval=False, z=-2.5, miss_side="below")] * 2)
        verdict = dm.judge(rows)
        self.assertEqual(verdict["acted"], 30)
        self.assertFalse(verdict["gates"]["coverage"])  # 23 of 30, not 23 of 28

    def test_a_shifted_centre_or_a_narrow_width_fails_and_names_its_next_step(self):
        good = {"decision_status": "operating_point", "chosen_share": 0.004, "warning": None, "within_capacity": True,
                "inside_prediction_interval": True, "holdout_actions": 16, "z": 0.0, "miss_side": "inside"}
        n = len(dm.SEEDS)
        shifted = [dict(good, z=-1.0) for _ in range(n)]  # every seed inside, all below the centre
        verdict = dm.judge(shifted)
        self.assertTrue(verdict["gates"]["coverage"])
        self.assertFalse(verdict["gates"]["centre"])
        self.assertEqual(verdict["next"], "correct the optimism of choosing the model")
        narrow = ([dict(good, z=1.5 if i % 2 else -1.5) for i in range(n - 6)]
                  + [dict(good, inside_prediction_interval=False, z=2.0 if i % 2 else -2.0, miss_side="above" if i % 2 else "below") for i in range(6)])
        verdict = dm.judge(narrow)
        self.assertTrue(verdict["gates"]["coverage"])  # 80% inside, yet the misses are too large on average
        self.assertFalse(verdict["gates"]["width"])
        self.assertEqual(verdict["next"], "a holdout-sized entity bootstrap")

    def test_too_few_seeds_acting_is_inconclusive(self):
        good = {"decision_status": "operating_point", "chosen_share": 0.004, "warning": None, "within_capacity": True,
                "inside_prediction_interval": True, "holdout_actions": 16, "z": 0.0, "miss_side": "inside"}
        n = len(dm.SEEDS)
        few = [dict(good, chosen_share=0.0, holdout_actions=None, inside_prediction_interval=None, z=None)] * (n - 10) + [dict(good)] * 10
        self.assertTrue(dm.judge(few)["verdict"].startswith("inconclusive"))

if __name__ == "__main__":
    unittest.main()
