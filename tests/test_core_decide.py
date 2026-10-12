"""Package 16.9: decisions on top of a model. A review budget from the value and cost of a review; a price from a model
of acceptance by price, with an interval, and a refusal when prices were never varied. No model is called."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import decide  # noqa: E402
from dclab_core.replicas import ride_pricing as rp  # noqa: E402

CONTEXT = ["distance_km", "hour", "rain", "couriers_nearby"]


def scored(n=4000, rate=0.02, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < rate).astype(int)
    return y, np.clip(rng.normal(0.2 + 0.5 * y, 0.15), 0, 1)


class ReviewBudgetTest(unittest.TestCase):
    def test_the_budget_follows_the_value_and_the_cost(self):
        y, s = scored()
        cheap = decide.review_budget(y, s, value_per_case=100, cost_per_review=2, bootstrap=50)
        dear = decide.review_budget(y, s, value_per_case=100, cost_per_review=40, bootstrap=50)
        self.assertGreater(cheap["share"], dear["share"])  # cheaper reviews: review more
        self.assertGreater(cheap["expected_value_per_row"], 0)
        lo, hi = cheap["expected_value_per_row_interval"]
        self.assertLessEqual(lo, hi)
        self.assertEqual(cheap["break_even_precision"], 0.02)
        self.assertIn("does not change the outcome", cheap["assumption"])

    def test_no_review_pays_when_a_review_costs_more_than_a_case_is_worth(self):
        y, s = scored()
        out = decide.review_budget(y, s, value_per_case=1, cost_per_review=5, bootstrap=20)
        self.assertEqual(out["share"], 0.0)
        self.assertIn("No review pays", out["note"])

    def test_partial_labels_make_the_value_a_lower_bound_and_entities_widen_the_interval(self):
        y, s = scored()
        groups = np.repeat(np.arange(len(y) // 40), 40).astype(float)
        groups[::97] = np.nan  # rows without an entity are their own units
        out = decide.review_budget(y, s, value_per_case=100, cost_per_review=2, partial=True, groups=groups, bootstrap=50,
                                   y_test=y[:1000], score_test=s[:1000])
        self.assertIn("lower bound", out["partial_note"])
        self.assertIn("entities", out["interval"])
        self.assertIn("cases_found", out["holdout"])

    def test_a_value_or_cost_must_be_positive(self):
        y, s = scored()
        with self.assertRaises(ValueError):
            decide.review_budget(y, s, value_per_case=0, cost_per_review=1)


class PriceTest(unittest.TestCase):
    def curve(self, varied, seed=0):
        offers = rp.generate(seed, varied=varied).offers
        return decide.price_curve(offers["fee"], offers["accepted"], reference=offers["usual_fee"], context=offers[CONTEXT],
                                  unit_value=rp.CUSTOMER_PRICE_FACTOR * offers["usual_fee"], bootstrap=60)

    def test_a_varied_market_gets_a_curve_whose_interval_holds_the_true_best(self):
        truth = rp.generate(0).facts["best_multiple"]
        out = self.curve(True)
        self.assertTrue(out["learnable"])
        lo, hi = out["best_multiple_interval"]
        self.assertLessEqual(lo - 0.03, truth)
        self.assertLessEqual(truth, hi + 0.03)
        self.assertGreater(out["expected_margin_per_offer"], out["margin_per_offer_as_priced"])  # better than pricing at random
        self.assertEqual(out["side"], "pay")
        self.assertIn("confounder", out["assumption"])
        self.assertEqual(len(out["curve"]), decide.GRID)

    def test_a_fixed_formula_price_gets_the_cannot_learn_warning(self):
        out = self.curve(False)
        self.assertFalse(out["learnable"])
        self.assertIn("Cannot learn how acceptance responds to price", out["warning"])
        self.assertIn("distinct price levels", out["warning"])  # the fee never moved against its usual level
        self.assertNotIn("best_multiple", out)

    def test_variation_checks_count_levels_spread_and_formula(self):
        price = np.full(500, 100.0)
        self.assertFalse(decide.price_variation(price)["learnable"])
        rng = np.random.default_rng(0)
        x = pd.DataFrame({"distance": rng.uniform(1, 10, 500)})
        varied = 50 + 10 * x["distance"] * rng.choice([0.8, 0.9, 1.0, 1.1, 1.2], 500)
        self.assertTrue(decide.price_variation(varied, x, 50 + 10 * x["distance"])["learnable"])
        with_reference = x.assign(usual=50 + 10 * x["distance"])  # the usual price in the context does not hide a real test
        self.assertTrue(decide.price_variation(varied, with_reference, with_reference["usual"])["learnable"])
        formula = 50 + 10 * x["distance"]
        check = decide.price_variation(formula, x)
        self.assertFalse(check["learnable"])
        self.assertGreater(check["explained_by_context_r2"], 0.97)


class PriceEdgeTest(unittest.TestCase):
    def market(self, n=3000, seed=0):
        rng = np.random.default_rng(seed)
        zone = rng.choice(["north", "south", "centre", "airport"], n)
        usual = 100 + 20 * rng.uniform(size=n)
        arm = rng.choice([0.8, 0.9, 1.0, 1.1, 1.2], n)
        accept = rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 - 3.0 * np.log(arm) + 0.4 * (zone == "airport"))))
        return pd.DataFrame({"zone": zone, "usual": usual, "arm": arm, "price": usual * arm, "accepted": accept.astype(int)})

    def test_a_category_that_sets_the_price_is_a_formula(self):
        f = self.market()
        tariff = f["zone"].map({"north": 1.0, "south": 1.1, "centre": 1.25, "airport": 1.5})
        out = decide.price_variation(100 * tariff, f[["zone"]])
        self.assertFalse(out["learnable"])
        self.assertEqual(out["explains_the_price"][0], "zone")

    def test_a_logged_test_arm_explains_the_price_and_is_named(self):
        f = self.market()
        logged = decide.price_variation(f["price"], f[["zone", "arm"]], f["usual"])
        self.assertFalse(logged["learnable"])
        self.assertIn("arm", logged["explains_the_price"])  # named, so the person can say it is the test's arm
        self.assertTrue(decide.price_variation(f["price"], f[["zone"]], f["usual"])["learnable"])  # without it: a real test

    def test_selling_with_categories_entities_and_missing_rows(self):
        f = self.market()
        f.loc[:9, "price"] = np.nan  # ten rows without a price are left out, and counted
        groups = np.arange(len(f)) // 5
        out = decide.price_curve(f["price"], f["accepted"], reference=f["usual"], context=f[["zone"]], unit_cost=60.0, groups=groups,
                                 bootstrap=30)
        self.assertTrue(out["learnable"], out.get("warning"))
        self.assertEqual((out["side"], out["rows_left_out"]), ("sell", 10))
        self.assertIn("entities", out["interval"])
        self.assertIsNotNone(out["expected_margin_per_offer"])

    def test_a_value_known_only_for_accepted_offers_is_refused(self):
        f = self.market()
        value = np.where(f["accepted"] == 1, 150.0, np.nan)
        out = decide.price_curve(f["price"], f["accepted"], reference=f["usual"], unit_value=value, bootstrap=10)
        self.assertFalse(out["learnable"])
        self.assertIn("missing", out["warning"])


if __name__ == "__main__":
    unittest.main()
