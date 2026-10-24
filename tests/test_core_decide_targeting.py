"""34.2, targeting and several actions: whom to treat from an uplift model under a budget, which of several actions each
row gets under each action's capacity, and the class with the lowest expected cost.

On the campaign replica (the true uplift of every row is known), treating the share the validation rows choose leaves
less value on the table than treating everyone or a random share of the same size, measured with its interval on the
holdout's true uplift; the assignment of several actions equals an exhaustive search on small cases; a cost matrix's
choice is each row's lowest expected cost.
"""

import itertools
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

try:
    from dclab_core import decide  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


def campaign(seed=0):
    """The replica split into fit, validation and holdout rows, an uplift model fitted on the fit rows only."""
    from dclab_core import uplift as U
    from dclab_core.replicas import campaign_uplift

    rep = campaign_uplift.generate(seed=seed, n=24000)
    f = rep.customers
    features = ["age", "app_user", "past_purchases", "tenure_months"]
    rng = np.random.default_rng(seed)
    part = rng.choice(3, size=len(f), p=[0.4, 0.3, 0.3])
    X, t, y = f[features].to_numpy(dtype=float), f["got_offer"].to_numpy(), f["bought"].to_numpy()
    model = U.TLearner().fit(X[part == 0], t[part == 0], y[part == 0])
    truth = np.asarray(rep.facts["true_uplift"])
    val, test = part == 1, part == 2
    return {"score_val": model.uplift(X[val]), "t_val": t[val], "y_val": y[val],
            "score_test": model.uplift(X[test]), "t_test": t[test], "y_test": y[test], "truth_test": truth[test]}


@unittest.skipIf(MISSING, MISSING)
class TargetingTests(unittest.TestCase):
    def test_targeting_leaves_less_on_the_table_than_treating_everyone_or_a_random_share(self):
        from dclab_core.decide import DecisionSpec, target_by_uplift

        c = campaign()
        spec = DecisionSpec(action="send the offer", value_per_case=40, cost_per_action=2)
        out = target_by_uplift(c["score_val"], c["t_val"], c["y_val"], spec, score_test=c["score_test"], t_test=c["t_test"],
                               y_test=c["y_test"], bootstrap=100)
        self.assertEqual(out["decision"], "targeting")
        self.assertGreater(out["share"], 0)
        self.assertLess(out["share"], 1)
        value = c["truth_test"] * 40 - 2  # what treating each holdout row is truly worth
        chosen = c["score_test"] >= out["threshold"]
        oracle = np.clip(value, 0, None).sum()
        regret = {"targeted": oracle - value[chosen].sum(), "everyone": oracle - value.sum()}
        rng = np.random.default_rng(1)
        regret["random"] = np.mean([oracle - value[rng.random(len(value)) < chosen.mean()].sum() for _ in range(200)])
        self.assertLess(regret["targeted"], regret["everyone"])
        self.assertLess(regret["targeted"], regret["random"])
        rows = np.arange(len(value))
        draws = []
        for _ in range(200):  # the difference to treating everyone, with its interval over holdout rows
            r = rng.choice(rows, len(rows))
            draws.append(value[r][chosen[r]].sum() - value[r].sum())
        self.assertGreater(np.quantile(draws, 0.05), 0)

    def test_the_measured_value_comes_from_the_arms_and_the_holdout_is_read_once(self):
        from dclab_core.decide import DecisionSpec, target_by_uplift

        c = campaign(seed=2)
        out = target_by_uplift(c["score_val"], c["t_val"], c["y_val"], DecisionSpec(action="offer", value_per_case=40, cost_per_action=2),
                               score_test=c["score_test"], t_test=c["t_test"], y_test=c["y_test"], bootstrap=50)
        self.assertEqual(out["chosen_on"], "the validation rows' arms")
        self.assertIn("treat everyone", out["compared"])
        self.assertIn("holdout", out)
        truth = (c["truth_test"][c["score_test"] >= out["threshold"]] * 40 - 2).sum() / len(c["truth_test"])
        low, high = out["holdout"]["prediction_interval"]
        self.assertLessEqual(low, truth)
        self.assertLessEqual(truth, high)

    def test_a_budget_caps_the_share_treated(self):
        from dclab_core.decide import DecisionSpec, target_by_uplift

        c = campaign(seed=3)
        out = target_by_uplift(c["score_val"], c["t_val"], c["y_val"],
                               DecisionSpec(action="offer", value_per_case=40, cost_per_action=0.5, capacity_share=0.05), bootstrap=20)
        self.assertLessEqual(out["share"], 0.05)

    def test_weighted_arms_are_read_weighted_on_the_holdout_too(self):
        """The review of 34.2: offers given mostly to loyal customers; the holdout read without its weights was 1.40 per
        row against a true 0.81."""
        from dclab_core.decide import DecisionSpec, target_by_uplift

        rng = np.random.default_rng(5)
        n = 30000
        past = rng.poisson(2.0, n)
        e = np.where(past >= 3, 0.8, 0.2)  # who got the offer depended on past purchases
        t = (rng.random(n) < e).astype(int)
        base = 1 / (1 + np.exp(-(-2.4 + 0.35 * past)))
        lift = np.where(past < 2, 0.15, 0.0)
        y = (rng.random(n) < np.clip(base + t * lift, 0, 1)).astype(int)
        score = lift + rng.normal(0, 0.02, n)
        w = t / e + (1 - t) / (1 - e)
        val, test = np.arange(n) % 2 == 0, np.arange(n) % 2 == 1
        spec = DecisionSpec(action="offer", value_per_case=40, cost_per_action=2)
        with self.assertRaises(ValueError):  # weighted arms without the holdout's weights cannot be read
            target_by_uplift(score[val], t[val], y[val], spec, weights=w[val], score_test=score[test], t_test=t[test], y_test=y[test])
        out = target_by_uplift(score[val], t[val], y[val], spec, weights=w[val], score_test=score[test], t_test=t[test], y_test=y[test],
                               w_test=w[test], bootstrap=50)
        chosen = score[test] >= out["threshold"]
        truth = (lift[test][chosen] * 40 - 2).sum() / test.sum()
        self.assertAlmostEqual(out["holdout"]["value_per_row"], truth, delta=0.3)

    def test_noise_says_it_cannot_tell_and_unread_fields_and_bad_scores_are_refused(self):
        from dclab_core.decide import DecisionSpec, target_by_uplift

        rng = np.random.default_rng(6)
        n = 7200
        t, y, s = rng.integers(0, 2, n), (rng.random(n) < 0.1).astype(int), rng.random(n)  # no uplift, no signal
        out = target_by_uplift(s, t, y, DecisionSpec(action="offer", value_per_case=40, cost_per_action=2), bootstrap=100)
        self.assertIn(out["pays"], ("no", "cannot tell"))
        with self.assertRaises(ValueError):
            target_by_uplift(s, t, y, DecisionSpec(action="offer", value_per_case=40, cost_per_action=2, cost_false_positive=30))
        bad = s.copy()
        bad[:5] = np.nan
        with self.assertRaises(ValueError):
            target_by_uplift(bad, t, y, DecisionSpec(action="offer", value_per_case=40))

    def test_arms_too_small_to_compare_are_refused(self):
        from dclab_core.decide import DecisionSpec, target_by_uplift

        with self.assertRaises(ValueError):
            target_by_uplift(np.random.rand(40), np.ones(40), np.random.randint(0, 2, 40), DecisionSpec(action="offer", value_per_case=1))


@unittest.skipIf(MISSING, MISSING)
class SeveralActionsTests(unittest.TestCase):
    def brute(self, values, capacities):
        n, a = values.shape
        best, pick = -np.inf, None
        for choice in itertools.product(range(-1, a), repeat=n):
            counts = [sum(1 for c in choice if c == j) for j in range(a)]
            if any(cap is not None and counts[j] > cap for j, cap in enumerate(capacities)):
                continue
            total = sum(values[i, c] for i, c in enumerate(choice) if c >= 0)
            if total > best + 1e-12:
                best, pick = total, choice
        return best, pick

    def test_the_assignment_equals_an_exhaustive_search_on_small_cases(self):
        from dclab_core.decide import choose_actions

        rng = np.random.default_rng(0)
        for case in range(150):
            n, a = int(rng.integers(2, 7)), int(rng.integers(1, 4))
            values = np.round(rng.normal(0, 10, size=(n, a)), 2)
            capacities = [int(rng.integers(0, n + 1)) if rng.random() < 0.7 else None for _ in range(a)]
            best, _ = self.brute(values, capacities)
            out = choose_actions(values, capacities)
            self.assertAlmostEqual(out["total_value"], round(best, 6), places=5, msg=(case, values.tolist(), capacities))
            counts = np.bincount(np.asarray(out["assignment"])[np.asarray(out["assignment"]) >= 0], minlength=a)
            for j, cap in enumerate(capacities):
                if cap is not None:
                    self.assertLessEqual(counts[j], cap)

    def test_a_row_takes_no_action_when_none_is_worth_it(self):
        from dclab_core.decide import choose_actions

        out = choose_actions(np.array([[-1.0, -2.0], [3.0, 1.0]]), [None, None], actions=["call", "email"])
        self.assertEqual(out["assignment"], [-1, 0])
        self.assertEqual(out["counts"], {"call": 1, "email": 0, "none": 1})

    def test_shapes_and_capacities_are_checked(self):
        from dclab_core.decide import choose_actions

        with self.assertRaises(ValueError):
            choose_actions(np.array([[np.nan, 5.0]]), [None, None])  # a missing value would drop the row
        self.assertEqual(choose_actions(np.array([[-np.inf, 5.0]]), [None, None])["assignment"], [1])  # minus infinity: not allowed

        with self.assertRaises(ValueError):
            choose_actions(np.array([1.0, 2.0]), [1])
        with self.assertRaises(ValueError):
            choose_actions(np.ones((3, 2)), [1])
        with self.assertRaises(ValueError):
            choose_actions(np.ones((3, 2)), [1, -1])


@unittest.skipIf(MISSING, MISSING)
class CostMatrixTests(unittest.TestCase):
    def test_equal_costs_choose_the_likeliest_class_and_a_costly_miss_shifts_the_choice(self):
        from dclab_core.decide import choose_by_cost_matrix

        proba = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.45, 0.1, 0.45]])
        even = 1 - np.eye(3)
        self.assertEqual(choose_by_cost_matrix(proba, even)["choice"], [0, 1, 0])
        costly = even.copy()
        costly[2, :2] = 20  # missing class 2 costs twenty times more
        out = choose_by_cost_matrix(proba, costly, classes=["low", "mid", "high"])
        self.assertEqual(out["labels"], ["high", "high", "high"])
        expected = proba @ costly
        self.assertTrue(np.allclose(out["expected_cost"], expected[np.arange(3), out["choice"]]))

    def test_a_cost_matrix_must_match_the_classes(self):
        from dclab_core.decide import choose_by_cost_matrix

        with self.assertRaises(ValueError):
            choose_by_cost_matrix(np.ones((2, 3)) / 3, np.ones((2, 2)))
        with self.assertRaises(ValueError):
            choose_by_cost_matrix(np.ones((2, 2)) / 2, -np.ones((2, 2)))
        with self.assertRaises(ValueError):
            choose_by_cost_matrix(np.array([[0.5, 0.2], [0.5, 0.5]]), np.ones((2, 2)))  # a row that does not add up to 1
        with self.assertRaises(ValueError):
            choose_by_cost_matrix(np.array([[np.nan, 1.0], [0.5, 0.5]]), np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
