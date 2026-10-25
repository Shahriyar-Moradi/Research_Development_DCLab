"""The measurement of 34.1 on the delivery-fraud replica: under a binding capacity the decision is a quota. Carried to
the holdout as a threshold, it acted on 6 rides where the team could review 16 (the holdout's scores sat lower than the
training rows' out-of-fold scores) and realised far less than its interval. Now, when the capacity ends the list, both
the interval's draws and the holdout act on the highest-ranked rows the capacity allows, each still worth acting on;
when the value ends it, the threshold stands as before."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np

    from dclab_core.decide import DecisionSpec, choose_operating_point

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


def scored(n, seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    s = np.clip(0.03 + 0.5 * y + rng.normal(0, 0.15, n), 0.001, 0.999)
    return y, s


@unittest.skipIf(MISSING, MISSING)
class QuotaTest(unittest.TestCase):
    def test_a_binding_capacity_fills_its_quota_on_a_holdout_scored_lower(self):
        y, s = scored(6000, 0)
        yt, st = scored(3000, 1)
        st = st * 0.6  # the holdout's scores on a lower scale, as unseen entities often get
        spec = DecisionSpec(action="review", value_per_case=100.0, cost_per_action=1.0, capacity_share=0.01)
        out = choose_operating_point(y, s, spec, y_test=yt, score_test=st, bootstrap=50)
        self.assertTrue(out["capacity_binds"])
        self.assertIn("quota", out["applied_as"])
        quota = int(np.floor(0.01 * len(yt)))
        worth = int((st * 100 - (1 - st) * 0 - 1 > 0).sum())
        self.assertEqual(out["holdout"]["actions"], min(quota, worth))  # the capacity filled, never with a row not worth it
        carried = int((st >= out["threshold"]).sum())
        self.assertLess(carried, quota)  # the old reading: a threshold carried over would have acted on fewer

    def test_without_a_binding_capacity_the_threshold_stands(self):
        y, s = scored(6000, 0)
        yt, st = scored(3000, 1)
        spec = DecisionSpec(action="review", value_per_case=100.0, cost_per_action=20.0)
        out = choose_operating_point(y, s, spec, y_test=yt, score_test=st, bootstrap=0)
        self.assertFalse(out["capacity_binds"])
        self.assertIn("threshold", out["applied_as"])
        self.assertEqual(out["holdout"]["actions"], int((st >= out["threshold"]).sum()))

    def test_low_probabilities_keep_the_value_inside_its_interval_and_the_quota_filled(self):
        """The review of 0.113.0: a 'worth it' filter stricter than the training's own choice put the point outside its
        interval and left the holdout's quota under-filled."""
        y, s = scored(6000, 0)
        yt, st = scored(3000, 1)
        spec = DecisionSpec(action="review", value_per_case=100.0, cost_per_action=20.0, capacity_share=0.02)
        out = choose_operating_point(y, s * 0.3, spec, y_test=yt, score_test=st * 0.3, bootstrap=200)
        lo, hi = out["expected_value_per_row_interval"]
        self.assertTrue(lo <= out["expected_value_per_row"] <= hi, (lo, out["expected_value_per_row"], hi))
        self.assertGreaterEqual(out["holdout"]["actions"], int((st * 0.3 >= out["threshold"]).sum()) if out["threshold"] else 0)

    def test_a_tie_across_the_capacity_still_acts(self):
        """The review of 0.113.0: a list may end only where the rank changes, so a tie across the capacity left only 'act
        on none'; the capacity itself may now end it."""
        rng = np.random.default_rng(3)
        s = np.r_[np.full(50, 0.9), rng.uniform(0, 0.2, 950)]
        y = np.r_[(rng.random(50) < 0.9).astype(int), (rng.random(950) < 0.05).astype(int)]
        spec = DecisionSpec(action="review", value_per_case=100.0, cost_per_action=1.0, capacity_share=0.01)
        out = choose_operating_point(y, s, spec, y_test=y, score_test=s, bootstrap=0)
        self.assertEqual(out["actions_training"], 10)
        self.assertNotIn("note", out)  # never "no action pays"
        self.assertEqual(out["holdout"]["actions"], 10)

    def test_no_row_worth_acting_on_is_ever_taken_to_fill_a_quota(self):
        y, s = scored(6000, 0)
        yt, _ = scored(3000, 1)
        st = np.full(len(yt), 0.001)  # nothing on the holdout is worth a review at these numbers
        spec = DecisionSpec(action="review", value_per_case=100.0, cost_per_action=1.0, capacity_share=0.01)
        out = choose_operating_point(y, s, spec, y_test=yt, score_test=st, bootstrap=0)
        self.assertEqual(out["holdout"]["actions"], 0)


if __name__ == "__main__":
    unittest.main()
