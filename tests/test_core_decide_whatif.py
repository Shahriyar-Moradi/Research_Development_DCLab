"""34.3, what-if: a chosen decision re-scored under changed values, costs, capacity or base rate, without refitting the
model; and how far each assumption can move before the decision changes.

A changed cost gives exactly the point a fresh choice under that cost gives; a changed base rate (cases more or less
common) matches choosing again on a resampled table with that rate, within sampling noise; every answer says what was
not refitted; the sensitivity table names, for each stated number, the change that moves the decision.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

try:
    from dclab_core import decide  # noqa: F401

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


def scores(n=6000, seed=0, rate=0.15):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    p = np.clip(0.1 + 0.5 * y + rng.normal(0, 0.2, n), 0.001, 0.999)
    return y, p


@unittest.skipIf(MISSING, MISSING)
class WhatIfTests(unittest.TestCase):
    def test_a_changed_cost_gives_the_point_a_fresh_choice_gives(self):
        from dclab_core.decide import DecisionSpec, choose_operating_point, what_if

        y, p = scores()
        spec = DecisionSpec(action="review", value_per_case=40, cost_per_action=3)
        out = what_if(y, p, spec, {"cost_per_action": 6})
        fresh = choose_operating_point(y, p, spec.model_copy(update={"cost_per_action": 6}), bootstrap=0)
        self.assertEqual((out["after"]["share"], out["after"]["threshold"]), (fresh["share"], fresh["threshold"]))
        self.assertLess(out["after"]["share"], out["before"]["share"])  # a dearer review: fewer reviews
        self.assertGreaterEqual(out["after"]["value_per_row"], out["keep_the_old_threshold"]["value_per_row"])
        self.assertIn("not refitted", out["note"])

    def test_a_changed_base_rate_matches_choosing_again_on_a_table_with_that_rate(self):
        from dclab_core.decide import DecisionSpec, choose_operating_point, what_if

        y, p = scores(n=40000, seed=1, rate=0.15)
        spec = DecisionSpec(action="review", value_per_case=40, cost_per_action=3)
        out = what_if(y, p, spec, {"base_rate": 0.05})
        rng = np.random.default_rng(2)  # the same model's scores on a world where cases are a third as common
        keep = (y == 0) | (rng.random(len(y)) < (0.05 / 0.95) / (0.15 / 0.85))
        fresh = choose_operating_point(y[keep], p[keep], spec, bootstrap=0)
        self.assertAlmostEqual(out["after"]["share"], fresh["share"], delta=0.02)
        self.assertLess(out["after"]["share"], out["before"]["share"])

    def test_a_capacity_change_holds_and_bad_changes_are_refused(self):
        from dclab_core.decide import DecisionSpec, what_if

        y, p = scores(seed=3)
        spec = DecisionSpec(action="review", value_per_case=40, cost_per_action=1)
        out = what_if(y, p, spec, {"capacity_share": 0.02})
        self.assertLessEqual(out["after"]["share"], 0.02)
        with self.assertRaises(ValueError):
            what_if(y, p, spec, {"price_of_tea": 3})  # not a number the decision states
        with self.assertRaises(ValueError):
            what_if(y, p, spec, {"base_rate": 1.5})

    def test_the_sensitivity_table_names_what_moves_the_decision(self):
        from dclab_core.decide import DecisionSpec, sensitivity

        y, p = scores(seed=4)
        table = sensitivity(y, p, DecisionSpec(action="review", value_per_case=40, cost_per_action=3))
        terms = {row["term"] for row in table["rows"]}
        self.assertEqual(terms, {"value_per_case", "cost_per_action", "base_rate"})
        for row in table["rows"]:
            self.assertEqual(len(row["at"]), len(table["multipliers"]))
            self.assertIn("share", row["at"][0])
        cost = next(r for r in table["rows"] if r["term"] == "cost_per_action")
        shares = [a["share"] for a in cost["at"]]
        self.assertEqual(shares, sorted(shares, reverse=True))  # dearer reviews never mean more reviews
        self.assertIn("not refitted", table["note"])

    def test_keeping_the_old_decision_keeps_its_very_rows(self):
        """The review of 34.3: on tied scores the old threshold, rounded, dropped the tied rows; with values per row the old
        rows were re-ranked with the new costs; a smaller capacity was ignored."""
        from dclab_core.decide import DecisionSpec, what_if

        rng = np.random.default_rng(7)
        y = (rng.random(3000) < 0.2).astype(int)
        p = np.round(np.clip(0.15 + 0.5 * y + rng.normal(0, 0.25, 3000), 0, 1) * 7) / 7  # seven neighbours: a 1/7 grid
        spec = DecisionSpec(action="review", value_per_case=40, cost_per_action=12)
        same = what_if(y, p, spec, {"cost_per_action": 12})
        self.assertEqual(same["keep_the_old_threshold"]["share"], same["before"]["share"])
        self.assertAlmostEqual(same["regret_if_unchanged"], 0.0, places=6)
        v = rng.choice([5.0, 300.0], size=3000, p=[0.7, 0.3])
        valued = DecisionSpec(action="review", value_column="order_value", cost_per_action=1)
        dearer = what_if(y, p, valued, {"cost_per_action": 8}, value_per_case=v)
        self.assertEqual(dearer["keep_the_old_threshold"]["share"], dearer["before"]["share"])
        wide = DecisionSpec(action="review", value_per_case=40, cost_per_action=1, capacity_share=0.5)
        yc, pc = scores(seed=11)  # distinct scores: a threshold can take any count (on ties a cap may only cut between them)
        narrow = what_if(yc, pc, wide, {"capacity_share": 0.02})
        self.assertLessEqual(narrow["keep_the_old_threshold"]["share"], 0.02)
        self.assertTrue(narrow["keep_the_old_threshold"]["cut_to_capacity"])
        self.assertGreaterEqual(narrow["regret_if_unchanged"], 0)  # the new choice is the best the new cap allows

    def test_values_per_row_scale_together_and_a_single_value_is_refused(self):
        from dclab_core.decide import DecisionSpec, sensitivity, what_if

        y, p = scores(seed=8)
        v = np.random.default_rng(9).gamma(2.0, 40.0, len(y))
        spec = DecisionSpec(action="review", value_column="order_value", value_per_case=50, cost_per_action=3)
        with self.assertRaises(ValueError):
            what_if(y, p, spec, {"value_per_case": 5000}, value_per_case=v)  # each row has its own value: this one is never read
        cheaper = what_if(y, p, spec, {"value_scale": 0.1}, value_per_case=v)
        self.assertLess(cheaper["after"]["share"], cheaper["before"]["share"])
        terms = {r["term"] for r in sensitivity(y, p, spec, value_per_case=v)["rows"]}
        self.assertIn("value_scale", terms)
        self.assertNotIn("value_per_case", terms)

    def test_weighted_rows_choose_a_point_only_and_weigh_as_shares(self):
        from dclab_core.decide import DecisionSpec, choose_operating_point

        y, p = scores(seed=10)
        spec = DecisionSpec(action="review", value_per_case=40, cost_per_action=3)
        with self.assertRaises(ValueError):
            choose_operating_point(y, p, spec, weights=np.ones(len(y)))  # with the default bootstrap: intervals read rows
        one = choose_operating_point(y, p, spec, weights=np.ones(len(y)), bootstrap=0)
        two = choose_operating_point(y, p, spec, weights=np.full(len(y), 2.0), bootstrap=0)
        self.assertEqual((one["share"], one["expected_value_per_row"]), (two["share"], two["expected_value_per_row"]))
        plain = choose_operating_point(y, p, spec, bootstrap=0)
        self.assertEqual((plain["share"], plain["expected_value_per_row"]), (one["share"], one["expected_value_per_row"]))
        every = np.ones(len(y), dtype=int)  # every row a case: no base rate to move
        self.assertNotIn("base_rate", {r["term"] for r in __import__("dclab_core.decide", fromlist=["x"]).sensitivity(every, p, spec)["rows"]})


if __name__ == "__main__":
    unittest.main()
