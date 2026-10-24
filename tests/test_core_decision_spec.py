"""34.1, one decision core: a decision states what acting on a row is worth and costs, and the operating point with the
highest expected value is chosen on out-of-fold scores, under a capacity, read once on the holdout.

The review budget (16.9) and the stated-cost threshold (17.5) are special cases and give the same answers; a value per
row ranks rows by expected gain and changes which rows are chosen as an explicit search says; a capacity is never
exceeded; the main product's objective receives the costs it has fields for and is told what it has none for.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

try:
    from dclab_core import decide  # noqa: F401

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


def scores(n=2000, seed=0, rate=0.12):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    p = np.clip(0.08 + 0.55 * y + rng.normal(0, 0.18, n), 0.001, 0.999)
    return y, p


@unittest.skipIf(MISSING, MISSING)
class DecisionSpecTests(unittest.TestCase):
    def test_a_decision_must_say_something_worth_deciding(self):
        from dclab_core.decide import DecisionSpec

        with self.assertRaises(ValueError):
            DecisionSpec(action="review")  # nothing is worth anything: no point is better than another
        with self.assertRaises(ValueError):
            DecisionSpec(action="review", value_per_case=-1)
        with self.assertRaises(ValueError):
            DecisionSpec(action="review", value_per_case=10, capacity_share=1.5)
        spec = DecisionSpec(action="review", value_per_case=10, cost_per_action=1)
        self.assertEqual(spec.schema_version, "dclab_core.decision_spec.v1")

    def test_a_review_budget_is_a_special_case_with_the_same_answer(self):
        from dclab_core.decide import DecisionSpec, choose_operating_point, review_budget

        y, p = scores()
        old = review_budget(y, p, value_per_case=40, cost_per_review=3, bootstrap=50)
        new = choose_operating_point(y, p, DecisionSpec(action="review", value_per_case=40, cost_per_action=3), bootstrap=50)
        self.assertEqual((new["share"], new["threshold"], new["expected_value_per_row"]),
                         (old["share"], old["threshold"], old["expected_value_per_row"]))

    def test_stated_costs_give_the_stated_cost_threshold(self):
        from dclab_core.assess import _threshold
        from dclab_core.decide import DecisionSpec, choose_operating_point

        y, p = scores(seed=1)
        threshold, _ = _threshold(y, p, None, {"false_positive": 5, "false_negative": 60})
        new = choose_operating_point(y, p, DecisionSpec(action="alert", cost_false_positive=5, cost_false_negative=60), bootstrap=20)
        self.assertAlmostEqual(new["threshold"], round(threshold, 6), places=6)

    def test_a_value_per_row_ranks_by_expected_gain_and_the_best_count_is_the_searched_one(self):
        from dclab_core.decide import DecisionSpec, choose_operating_point

        y, p = scores(seed=2)
        rng = np.random.default_rng(3)
        value = rng.choice([5.0, 500.0], size=len(y), p=[0.8, 0.2])  # most cases are worth little, some a lot
        spec = DecisionSpec(action="review", cost_per_action=8, value_column="order_value")
        out = choose_operating_point(y, p, spec, value_per_case=value, bootstrap=20)
        self.assertEqual(out["ranked_by"], "expected gain")
        gain = p * value - 8
        order = np.argsort(-gain, kind="mergesort")
        realised = np.r_[0, np.cumsum((y * value - 8)[order])]
        self.assertEqual(out["actions_training"], int(np.argmax(realised)))
        by_score = choose_operating_point(y, p, spec, value_per_case=np.full(len(y), value.mean()), bootstrap=20)
        self.assertNotEqual(out["actions_training"], by_score["actions_training"])

    def test_a_capacity_is_never_exceeded(self):
        from dclab_core.decide import DecisionSpec, choose_operating_point

        y, p = scores(seed=4)
        free = choose_operating_point(y, p, DecisionSpec(action="review", value_per_case=50, cost_per_action=1), bootstrap=20)
        capped = choose_operating_point(y, p, DecisionSpec(action="review", value_per_case=50, cost_per_action=1, capacity_share=0.03),
                                        bootstrap=20)
        self.assertGreater(free["share"], 0.03)
        self.assertLessEqual(capped["actions_training"], int(0.03 * len(y)))
        self.assertTrue(capped["capacity_binds"])

    def test_the_intervals_hold_what_they_say_the_expected_value_and_a_holdouts_value(self):
        """Measured 2026-10-24: a 90% interval for the expected value held a same-sized holdout's value in only 45 of 60
        seeds, as it should (the holdout has its own noise); the expected value's interval is checked against the value on
        a very large sample, the holdout against the prediction interval."""
        from dclab_core.decide import DecisionSpec, choose_operating_point

        spec = DecisionSpec(action="review", value_per_case=40, cost_per_action=3)
        big_y, big_p = scores(n=200000, seed=999)
        held_mean = held_holdout = 0
        seeds = 40
        for seed in range(seeds):
            y, p = scores(n=3000, seed=100 + seed)
            yt, pt = scores(n=3000, seed=500 + seed)
            out = choose_operating_point(y, p, spec, y_test=yt, score_test=pt, bootstrap=200, seed=seed)
            self.assertEqual(out["chosen_on"], "out-of-fold training predictions")
            k = int(round(out["share"] * len(big_y)))
            top = np.argsort(-big_p, kind="mergesort")[:k]
            truth = (40 * big_y[top].sum() - 3 * k) / len(big_y)  # the chosen share's value on a very large sample
            low, high = out["expected_value_per_row_interval"]
            held_mean += low <= truth <= high
            plow, phigh = out["holdout"]["prediction_interval"]
            held_holdout += plow <= out["holdout"]["expected_value_per_row"] <= phigh
        self.assertGreaterEqual(held_mean, int(0.78 * seeds), held_mean)  # nominal 90%: a margin for 40 seeds
        self.assertGreaterEqual(held_holdout, int(0.78 * seeds), held_holdout)

    def test_no_action_pays_says_so(self):
        from dclab_core.decide import DecisionSpec, choose_operating_point

        y, p = scores(seed=7)
        out = choose_operating_point(y, p, DecisionSpec(action="review", value_per_case=1, cost_per_action=50), bootstrap=10)
        self.assertEqual(out["actions_training"], 0)
        self.assertIn("No action pays", out["note"])

    def test_the_assessment_reads_the_core_for_stated_costs(self):
        from dclab_core import assess, decide

        y, p = scores(seed=8)
        with mock.patch.object(decide, "choose_operating_point", wraps=decide.choose_operating_point) as spy:
            threshold, why = assess._threshold(y, p, None, {"false_positive": 2, "false_negative": 30})
        self.assertTrue(spy.called)
        self.assertIn("expected cost", why)

    def test_tied_scores_are_taken_together_as_a_threshold_takes_them(self):
        """The review of 34.1: a count cut through a run of equal scores, then flagged the whole run at the threshold."""
        from dclab_core import assess
        from dclab_core.decide import DecisionSpec, choose_operating_point

        y, s = np.array([1, 1, 0]), np.array([0.9, 0.5, 0.5])
        threshold, _ = assess._threshold(y, s, None, {"false_positive": 10, "false_negative": 1})
        self.assertEqual(threshold, 0.9)  # the old answer: flagging the tie costs a false alarm of 10 to save a miss of 1

        def old(y, score, fp_cost, fn_cost):  # 17.5's search, kept here as the reference: cuts only between distinct scores
            thresholds, tp, fp = assess._sweep(y, score)
            cost = fp_cost * fp + fn_cost * (int((y == 1).sum()) - tp)
            if fn_cost * int((y == 1).sum()) <= cost.min():
                return float(np.nextafter(score.max(), np.inf))
            return float(thresholds[int(np.argmin(cost))])

        rng = np.random.default_rng(0)
        for seed in range(100):
            y, p = scores(n=400, seed=seed)
            p = np.round(p, 1)  # trees and neighbours give ties
            fp_cost, fn_cost = float(rng.integers(1, 20)), float(rng.integers(1, 60))
            self.assertEqual(assess._threshold(y, p, None, {"false_positive": fp_cost, "false_negative": fn_cost})[0], old(y, p, fp_cost, fn_cost), seed)
            out = choose_operating_point(y, p, DecisionSpec(action="alert", value_per_case=30, cost_per_action=2), bootstrap=0)
            if out["actions_training"]:
                self.assertEqual(out["actions_training"], int((p >= out["threshold"]).sum()), seed)  # a count a threshold can take

    def test_values_per_row_and_daily_capacities_are_never_silently_dropped(self):
        from dclab_core.decide import DecisionSpec, choose_operating_point

        y, p = scores(seed=9)
        yt, pt = scores(seed=10)
        valued = DecisionSpec(action="review", cost_per_action=8, value_column="order_value")
        with self.assertRaises(ValueError):
            choose_operating_point(y, p, valued)  # the column's values were not given
        with self.assertRaises(ValueError):
            choose_operating_point(y, p, valued, value_per_case=np.full(len(y), 50.0), y_test=yt, score_test=pt)  # nor the holdout's
        daily = DecisionSpec(action="review", value_per_case=50, cost_per_action=1, capacity_per_day=10)
        with self.assertRaises(ValueError):
            choose_operating_point(y, p, daily)  # a daily cap with no rows a day cannot hold
        both = DecisionSpec(action="review", value_per_case=50, cost_per_action=1, capacity_per_day=10, capacity_share=0.5)
        out = choose_operating_point(y, p, both, rows_per_day=1000, bootstrap=10)
        self.assertLessEqual(out["actions_training"], int(len(y) * 10 / 1000))  # the tighter cap holds
        with self.assertRaises(ValueError):
            choose_operating_point(np.array([], dtype=int), np.array([]), DecisionSpec(action="review", value_per_case=1))

    def test_the_holdout_never_goes_over_the_capacity(self):
        from dclab_core.decide import DecisionSpec, choose_operating_point

        y, p = scores(n=10, seed=11)
        yt, pt = scores(n=15, seed=12)
        out = choose_operating_point(y, p, DecisionSpec(action="review", value_per_case=100, cost_per_action=1, capacity_share=0.1),
                                     y_test=yt, score_test=pt, bootstrap=10)
        if "holdout" in out:
            self.assertLessEqual(out["holdout"]["actions"], int(np.floor(0.1 * 15)))

    def test_the_main_products_objective_gets_the_costs_and_hears_what_has_no_field(self):
        from dclab_core.adapters.main_product import objective_of
        from dclab_core.decide import DecisionSpec

        out = objective_of(DecisionSpec(action="review", cost_false_positive=4, cost_false_negative=90, value_per_case=10, capacity_share=0.02))
        self.assertEqual(out["fields"], {"Objective.cost_false_positive": 4.0, "Objective.cost_false_negative": 90.0})
        self.assertTrue(any("value_per_case" in n for n in out["notes"]))
        self.assertTrue(any("capacity" in n for n in out["notes"]))


if __name__ == "__main__":
    unittest.main()
