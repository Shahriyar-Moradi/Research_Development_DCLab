"""Package 24.5: the composition. Candidates, the choice on validation (a tie to the simpler one), the one read of the holdout, the paired interval."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dclab_core import compose


def _world(n: int = 4000, seed: int = 0):
    """Two kinds of fraud, each found by one leaf only: leaf a ranks the first kind high, leaf b the second; both are noisy elsewhere."""
    rng = np.random.default_rng(seed)
    kind = rng.choice([0, 1, 2], n, p=[0.94, 0.03, 0.03])
    y = pd.Series((kind > 0).astype(int), name="confirmed")
    noise = lambda: rng.normal(0, 1, n)
    a = pd.Series(noise() + 4.0 * (kind == 1))
    b = pd.Series(noise() + 4.0 * (kind == 2))
    c = pd.Series(noise() + 0.3 * (kind > 0))          # a weak leaf
    group = pd.Series(rng.integers(0, 200, n), name="biker")
    return {"a": a, "b": b, "c": c}, y, group


class BuildingBlocksTest(unittest.TestCase):
    def test_a_budget_is_a_share_a_count_or_one_percent(self):
        self.assertEqual(compose.budget_rows(1000, 0.02), 20)
        self.assertEqual(compose.budget_rows(1000, 50), 50)
        self.assertEqual(compose.budget_rows(1000, None), 10)
        self.assertEqual(compose.budget_rows(10, 0.001), 1)  # never an empty list

    def test_percentile_puts_scores_of_any_scale_on_one_scale(self):
        s = pd.Series([10.0, 200.0, 3000.0, 4.0])
        p = compose.percentile(s)
        self.assertEqual(list(p.rank()), list(s.rank()))
        self.assertAlmostEqual(float(p.max()), 1.0)

    def test_fusions_take_the_top_of_any_leaf_or_the_mean_of_all(self):
        x = pd.Series([0.1, 0.9, 0.2, 0.3])
        y = pd.Series([0.8, 0.1, 0.2, 0.9])
        top = compose.fuse({"x": x, "y": y}, "rank_max")
        mean = compose.fuse({"x": x, "y": y}, "rank_mean")
        expected = pd.concat([compose.percentile(x), compose.percentile(y)], axis=1).max(axis=1)
        self.assertTrue(top.equals(expected))
        self.assertTrue((top >= mean - 1e-12).all(), "the max of ranks is never below their mean")

    def test_a_weighted_fusion_follows_its_weights(self):
        x = pd.Series([0.1, 0.9, 0.5])
        y = pd.Series([0.9, 0.1, 0.5])
        only_x = compose.fuse({"x": x, "y": y}, "rank_weighted", {"x": 1.0, "y": 0.0})
        self.assertEqual(list(only_x.rank()), list(x.rank()))
        equal = compose.fuse({"x": x, "y": y}, "rank_weighted", {})
        mean = compose.fuse({"x": x, "y": y}, "rank_mean")
        self.assertTrue(np.allclose(equal, mean), "no weights: the leaves count alike")

    def test_an_unknown_fusion_and_an_empty_fusion_are_refused(self):
        with self.assertRaises(ValueError):
            compose.fuse({"x": pd.Series([1.0])}, "magic")
        with self.assertRaises(ValueError):
            compose.fuse({}, "rank_max")

    def test_a_gate_keeps_the_rows_the_gate_leaf_puts_on_top_above_the_rest(self):
        score = pd.Series([0.9, 0.8, 0.7, 0.6])
        by = pd.Series([0.1, 0.2, 0.3, 0.9])
        out = compose.gated(score, by, 0.25)  # only row 3 is inside the gate
        self.assertEqual(int(out.idxmax()), 3)
        self.assertTrue(out[3] > out[[0, 1, 2]].max())

    def test_rows_that_tie_at_the_cut_share_the_places_left_whatever_order_they_are_in(self):
        flat = np.zeros(10)
        first = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        last = first[::-1].copy()
        self.assertAlmostEqual(compose.recall_at(flat, first, 3), 0.3)
        self.assertAlmostEqual(compose.recall_at(flat, last, 3), 0.3, msg="a leaf with few non-zero scores does not win or lose by where its zeros sit")
        mixed = np.array([5.0, 4.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(compose.recall_at(mixed, y, 4), (1 + 2 * 0.25) / 2, msg="two sure places, then two of the four tied rows, one of which is a case")
        self.assertEqual(compose.recall_at(mixed, y, 10), 1.0)
        self.assertEqual(compose.recall_at(mixed, y, 0), 0.0)

    def test_recall_at_a_budget_counts_confirmed_cases_in_the_top_rows(self):
        score = np.array([0.9, 0.8, 0.1, 0.7, 0.2])
        y = np.array([1, 0, 1, 1, 0])
        self.assertAlmostEqual(compose.recall_at(score, y, 2), 1 / 3)
        self.assertAlmostEqual(compose.recall_at(score, y, 4), 2 / 3)
        self.assertEqual(compose.recall_at(score, np.zeros(5), 2), 0.0)


class ChoiceTest(unittest.TestCase):
    def test_a_fusion_that_beats_every_leaf_on_validation_is_chosen(self):
        scores, y, _ = _world()
        choice = compose.choose(scores, y, share=0.06)
        self.assertEqual(choice.chosen, "rank_max")
        self.assertGreater(choice.candidates["rank_max"], max(choice.candidates[k] for k in scores))
        self.assertIn(choice.best_leaf, scores)
        self.assertEqual(choice.best_leaf_score, choice.candidates[choice.best_leaf])

    def test_the_best_leaf_is_what_the_composition_must_beat_and_it_is_judged_on_validation(self):
        scores, y, _ = _world()
        choice = compose.choose(scores, y, share=0.06)
        alone = {k: choice.candidates[k] for k in scores}
        self.assertEqual(choice.best_leaf, max(alone, key=alone.get))

    def test_when_no_combination_helps_the_best_leaf_alone_is_the_composition(self):
        rng = np.random.default_rng(3)
        n = 3000
        y = pd.Series((rng.random(n) < 0.03).astype(int))
        strong = pd.Series(rng.normal(0, 1, n) + 5.0 * y.to_numpy())
        noise = pd.Series(rng.normal(0, 1, n))
        choice = compose.choose({"strong": strong, "noise": noise}, y, share=0.05)
        self.assertEqual(choice.chosen, "strong")
        self.assertTrue(any("beat" in w for w in choice.warnings), "the result says the combination did not help")

    def test_a_tie_goes_to_the_simpler_candidate(self):
        y = pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        same = pd.Series([0.9, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
        choice = compose.choose({"one": same, "two": same.copy()}, y, share=0.1)
        self.assertEqual(choice.chosen, "one", "every candidate found the case: the first leaf alone is the simplest")

    def test_few_confirmed_cases_are_said(self):
        scores, y, _ = _world(500)
        y = pd.Series(np.zeros(len(y), dtype=int), index=y.index)
        y.iloc[:2] = 1
        choice = compose.choose(scores, y, share=0.05)
        self.assertTrue(any("coin toss" in w for w in choice.warnings))

    def test_gates_are_candidates_only_when_asked_and_only_between_known_leaves(self):
        scores, y, _ = _world()
        plain = compose.choose(scores, y, share=0.06)
        gated = compose.choose(scores, y, share=0.06, gates=[("a", "b"), ("a", "zzz")])
        self.assertFalse(any(k.startswith("gate:") for k in plain.candidates))
        self.assertEqual(sorted(k for k in gated.candidates if k.startswith("gate:")), sorted(f"gate:a>b@{int(round(s * 100))}" for s in compose.GATE_SHARES))

    def test_average_precision_is_the_metric_without_a_budget(self):
        scores, y, _ = _world()
        choice = compose.choose(scores, y, metric="average_precision")
        self.assertEqual(choice.metric, "average_precision")
        self.assertEqual(choice.budget_rows, 0)
        self.assertTrue(all(0 <= v <= 1 for v in choice.candidates.values()))

    def test_a_candidate_is_built_the_same_way_on_any_rows(self):
        scores, y, _ = _world()
        a = compose.build("rank_max", scores)
        b = compose.build("rank_max", {k: v.copy() for k, v in scores.items()})
        self.assertTrue(a.equals(b))
        with self.assertRaises(ValueError):
            compose.build("gate:a>missing@50", scores)
        with self.assertRaises(ValueError):
            compose.build("nonsense", scores)


class HoldoutReadTest(unittest.TestCase):
    def _read(self, seed: int = 1, **kwargs):
        val, yv, _ = _world(seed=seed)
        hol, yh, group = _world(seed=seed + 100)
        choice = compose.choose(val, yv, share=0.06)
        return choice, compose.read_holdout(hol, yh, choice, clusters=group, bootstraps=200, seed=0, **kwargs), yh

    def test_the_holdout_is_read_once_and_the_composition_beats_the_best_leaf_with_its_interval_above_zero(self):
        choice, result, _ = self._read()
        self.assertEqual(result.reads, 1)
        self.assertEqual(result.chosen, choice.chosen)
        self.assertEqual(result.best_leaf, choice.best_leaf)
        self.assertGreater(result.composed.value, result.best_leaf_estimate.value)
        self.assertGreater(result.difference.low, 0)
        self.assertTrue(result.beats_best_leaf)
        self.assertEqual(result.cluster_key, "biker")
        self.assertLessEqual(result.composed.low, result.composed.value)
        self.assertGreaterEqual(result.composed.high, result.composed.value)

    def test_the_best_leaf_it_is_compared_with_was_chosen_on_validation_not_on_the_holdout(self):
        val, yv, _ = _world(seed=1)
        hol, yh, group = _world(seed=101)
        choice = compose.choose(val, yv, share=0.06)
        flipped = dict(hol)
        flipped[choice.best_leaf] = pd.Series(np.random.default_rng(9).normal(0, 1, len(yh)))  # the leaf that did best on validation does badly on the holdout
        result = compose.read_holdout(flipped, yh, choice, clusters=group, bootstraps=100, seed=0)
        k = compose.budget_rows(len(yh), 0.06)
        on_holdout = {l: compose.recall_at(flipped[l].to_numpy(float), yh.to_numpy(), k) for l in flipped}
        self.assertNotEqual(max(on_holdout, key=on_holdout.get), choice.best_leaf, "on the holdout another leaf is the best one")
        self.assertEqual(result.best_leaf, choice.best_leaf, "the bar does not move to whichever leaf did best on the holdout")
        self.assertAlmostEqual(result.best_leaf_estimate.value, round(on_holdout[choice.best_leaf], 4), places=4,
                               msg="the bar is the validation-best leaf's own score on the holdout, not the holdout's best score")
        self.assertLess(result.best_leaf_estimate.value, max(on_holdout.values()) - 0.05)

    def test_the_best_leaf_is_taken_among_every_leaf_and_the_fusions_combine_only_the_fused_ones(self):
        scores, y, _ = _world()
        strong = scores["a"] + 6.0 * y.to_numpy()                       # a leaf outside the fusion that is the best alone
        allv = {**scores, "z": strong}
        choice = compose.choose(allv, y, share=0.06, fused=["a", "b"])
        self.assertEqual(choice.best_leaf, "z", "a built leaf that the problem does not fuse is still the bar")
        self.assertEqual(choice.fused, ["a", "b"])
        by_hand = compose.fuse({"a": scores["a"], "b": scores["b"]}, "rank_max")
        self.assertTrue(compose.build("rank_max", allv, choice.weights, choice.fused).equals(by_hand), "z does not enter the fusion")
        self.assertFalse(compose.build("rank_max", allv, choice.weights).equals(by_hand), "without the fused list it would")

    def test_nothing_about_the_holdout_reaches_the_choice(self):
        val, yv, _ = _world(seed=1)
        choice = compose.choose(val, yv, share=0.06)
        again = compose.choose({k: v.copy() for k, v in val.items()}, yv.copy(), share=0.06)
        self.assertEqual(choice.model_dump(), again.model_dump())

    def test_when_the_chosen_composition_is_the_best_leaf_the_difference_is_zero_and_said(self):
        rng = np.random.default_rng(5)
        n = 3000
        yv = pd.Series((rng.random(n) < 0.03).astype(int))
        val = {"strong": pd.Series(rng.normal(0, 1, n) + 5.0 * yv.to_numpy()), "noise": pd.Series(rng.normal(0, 1, n))}
        yh = pd.Series((rng.random(n) < 0.03).astype(int))
        hol = {"strong": pd.Series(rng.normal(0, 1, n) + 5.0 * yh.to_numpy()), "noise": pd.Series(rng.normal(0, 1, n))}
        choice = compose.choose(val, yv, share=0.05)
        result = compose.read_holdout(hol, yh, choice, bootstraps=100, seed=0)
        self.assertEqual(result.difference.value, 0.0)
        self.assertFalse(result.beats_best_leaf)
        self.assertTrue(any("best leaf alone" in l for l in result.limits))

    def test_the_same_seed_gives_the_same_interval(self):
        a = self._read()[1]
        b = self._read()[1]
        self.assertEqual(a.model_dump(), b.model_dump())

    def test_few_confirmed_cases_make_the_interval_say_so(self):
        val, yv, _ = _world(seed=1)
        choice = compose.choose(val, yv, share=0.06)
        hol, yh, _ = _world(500, seed=7)
        yh[:] = 0
        yh.iloc[:2] = 1
        result = compose.read_holdout({k: v.iloc[:500].reset_index(drop=True) for k, v in hol.items()}, yh, choice, bootstraps=50, seed=0)
        self.assertTrue(any("wide" in l for l in result.limits))

    def test_without_clusters_rows_are_resampled_one_by_one_and_it_is_said(self):
        val, yv, _ = _world(seed=1)
        hol, yh, _ = _world(seed=101)
        choice = compose.choose(val, yv, share=0.06)
        result = compose.read_holdout(hol, yh, choice, bootstraps=50, seed=0)
        self.assertIsNone(result.cluster_key)
        self.assertTrue(any("rows" in l and "bootstrap" in l for l in result.limits))

    def test_with_no_resamples_the_interval_is_the_point_and_no_gain_is_claimed(self):
        val, yv, _ = _world(seed=1)
        hol, yh, group = _world(seed=101)
        choice = compose.choose(val, yv, share=0.06)
        result = compose.read_holdout(hol, yh, choice, clusters=group, bootstraps=0, seed=0)
        self.assertGreater(result.difference.value, 0)
        self.assertEqual((result.difference.low, result.difference.high), (result.difference.value, result.difference.value))
        self.assertFalse(result.beats_best_leaf, "a gain is claimed only from an interval, and there is none")
        self.assertTrue(any("resamples were drawn" in l for l in result.limits))

    def test_resamples_with_no_confirmed_case_are_counted_not_hidden(self):
        val, yv, _ = _world(seed=1)
        hol, yh, group = _world(seed=101)
        choice = compose.choose(val, yv, share=0.06)
        sparse = pd.Series(0, index=yh.index)
        sparse.iloc[0] = 1                                                 # one case in one cluster: about a third of the resamples miss it
        result = compose.read_holdout(hol, sparse, choice, clusters=group, bootstraps=300, seed=0)
        self.assertLess(result.bootstraps, 300)
        self.assertTrue(any("held no confirmed case" in l and str(300 - result.bootstraps) in l for l in result.limits))

    def test_the_limits_always_say_the_cases_are_partial(self):
        self.assertTrue(any("confirmed" in l and "biased" in l for l in self._read()[1].limits))


if __name__ == "__main__":
    unittest.main()
