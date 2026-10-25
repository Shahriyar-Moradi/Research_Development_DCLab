"""34.3's open item: what-if for targeting (34.2's target_by_uplift) and for several actions (choose_actions), with no
refit. A changed number gives exactly the choice a fresh call with that number gives; the old decision kept (the same
rows, cut to a smaller capacity) is valued under the new numbers; not changing never beats the new choice; a change the
decision does not read is refused."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np

    from dclab_core import decide
    from dclab_core.decide import DecisionSpec

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


def campaign(n=4000, seed=0):
    """A randomised campaign: the offer lifts conversion for high scores and does nothing for low ones."""
    rng = np.random.default_rng(seed)
    score = rng.random(n)
    treated = rng.integers(0, 2, n)
    base = 0.05 + 0.05 * rng.random(n)
    lift = np.clip(score - 0.5, 0, None) * 0.4
    outcome = (rng.random(n) < base + treated * lift).astype(int)
    return score, treated, outcome


@unittest.skipIf(MISSING, MISSING)
class TargetingWhatIfTest(unittest.TestCase):
    def setUp(self):
        self.s, self.t, self.y = campaign()
        self.spec = DecisionSpec(action="send the offer", value_per_case=40.0, cost_per_action=2.0)

    def test_a_changed_cost_chooses_as_a_fresh_call_and_not_changing_never_wins(self):
        r = decide.targeting_what_if(self.s, self.t, self.y, self.spec, {"cost_per_action": 5.0})
        fresh = decide.target_by_uplift(self.s, self.t, self.y, self.spec.model_copy(update={"cost_per_action": 5.0}), bootstrap=0)
        before = decide.target_by_uplift(self.s, self.t, self.y, self.spec, bootstrap=0)
        self.assertEqual((r["after"]["share"], r["after"]["value_per_row"]), (fresh["share"], fresh["value_per_row"]))
        self.assertEqual(r["before"]["share"], before["share"])
        top = np.argsort(-self.s, kind="mergesort")[: before["treated_training"]]  # the same rows, valued anew by hand
        lift = self.y[top][self.t[top] == 1].mean() - self.y[top][self.t[top] == 0].mean()
        self.assertAlmostEqual(r["keep_the_old_rows"]["value_per_row"], round((lift * 40 - 5.0) * len(top) / len(self.s), 4), places=4)
        self.assertGreaterEqual(r["regret_if_unchanged"], 0)
        self.assertIn("not refitted", r["note"])

    def test_a_smaller_capacity_cuts_the_old_rows_and_a_base_rate_is_refused(self):
        before = decide.target_by_uplift(self.s, self.t, self.y, self.spec, bootstrap=0)
        cap = round(before["share"] / 2, 4)
        r = decide.targeting_what_if(self.s, self.t, self.y, self.spec, {"capacity_share": cap})
        self.assertTrue(r["keep_the_old_rows"]["cut_to_capacity"])
        self.assertLessEqual(r["keep_the_old_rows"]["share"], cap)
        self.assertLessEqual(r["after"]["share"], cap)
        tight = decide.targeting_what_if(self.s, self.t, self.y, self.spec, {"capacity_share": 0.012})  # too few rows per arm to read
        self.assertTrue(tight["keep_the_old_rows"]["value_per_row"] is None or tight["regret_if_unchanged"] >= 0)
        rounded = np.round(self.s, 1)  # tied scores, as tree uplift models give: a cut only where the score changes
        tied = decide.target_by_uplift(rounded, self.t, self.y, self.spec, bootstrap=0)
        half = decide.targeting_what_if(rounded, self.t, self.y, self.spec, {"capacity_share": round(tied["share"] * 0.5, 4)})
        self.assertTrue(half["keep_the_old_rows"]["value_per_row"] is None or half["regret_if_unchanged"] >= -1e-9)
        self.assertNotIn("pays", r["before"])  # chosen without a bootstrap: it could never say yes
        with self.assertRaises(ValueError):
            decide.targeting_what_if(self.s, self.t, self.y, self.spec, {"base_rate": 0.2})
        with self.assertRaises(ValueError):
            decide.targeting_what_if(self.s, self.t, self.y, self.spec, {"cost_false_negative": 3})


@unittest.skipIf(MISSING, MISSING)
class ActionsWhatIfTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.v = rng.normal(1.0, 2.0, (80, 3))
        self.v[rng.random((80, 3)) < 0.1] = -np.inf  # some rows may not take some actions
        self.names = ["call", "email", "visit"]
        self.caps = [12, None, 5]

    def test_a_changed_capacity_or_value_chooses_as_a_fresh_call(self):
        r = decide.actions_what_if(self.v, self.caps, {"capacities": {"visit": 2}}, actions=self.names)
        fresh = decide.choose_actions(self.v, [12, None, 2], self.names)
        self.assertEqual(r["after"]["counts"], fresh["counts"])
        self.assertEqual(r["after"]["total_value"], fresh["total_value"])
        scaled = self.v.copy()
        scaled[:, 0] *= 0.5
        scaled[:, 1] -= 1.5
        r2 = decide.actions_what_if(self.v, self.caps, {"value_scale": {"call": 0.5}, "added_cost": {"email": 1.5}}, actions=self.names)
        self.assertEqual(r2["after"]["total_value"], decide.choose_actions(scaled, self.caps, self.names)["total_value"])

    def test_the_old_assignment_kept_is_cut_to_the_new_capacity_and_valued_anew(self):
        old = decide.choose_actions(self.v, self.caps, self.names)
        r = decide.actions_what_if(self.v, self.caps, {"capacities": {"call": 4}}, actions=self.names)
        keep = r["keep_the_old_assignment"]
        self.assertEqual(keep["counts"]["call"], min(4, old["counts"]["call"]))
        self.assertTrue(keep["cut_to_capacity"])
        calls = [i for i, a in enumerate(old["assignment"]) if a == 0]
        kept_calls = sorted(calls, key=lambda i: -self.v[i, 0])[:4]  # the old call rows worth most stay
        others = sum(self.v[i, a] for i, a in enumerate(old["assignment"]) if a > 0)
        self.assertAlmostEqual(keep["total_value"], round(others + sum(self.v[i, 0] for i in kept_calls), 6), places=5)
        self.assertGreaterEqual(r["regret_if_unchanged"], -1e-9)

    def test_an_unknown_action_or_a_bad_change_is_refused(self):
        for changes in ({"capacities": {"fax": 3}}, {"capacities": {"call": -1}}, {"value_scale": {"call": 0}}, {"discount": {"call": 1}}):
            with self.assertRaises(ValueError, msg=str(changes)):
                decide.actions_what_if(self.v, self.caps, changes, actions=self.names)
        with self.assertRaises(ValueError):  # one capacity short: said, never an index error
            decide.actions_what_if(self.v, self.caps[:2], {"capacities": {"call": 3}}, actions=self.names)


if __name__ == "__main__":
    unittest.main()
