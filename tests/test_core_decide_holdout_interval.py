"""dclab-core 0.114.0: the holdout's prediction interval from holdout-sized samples of each bootstrap draw (the
holdout's number of rows, whole entities; one fold's rows each when the folds are given), the decision applied to each
as it runs. fraud_decision_v2 measured the old normal
approximation too narrow by about a quarter on an entity split; on fraud-like data (a few entities hold the cases) the
new interval holds about nine holdouts in ten and its misses are of the size it says, and it names its method."""

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


def world(seed, n_entities=400):
    rng = np.random.default_rng(seed)
    sizes = np.minimum(rng.pareto(1.2, n_entities) * 10 + 3, 600).astype(int)  # skewed: a few entities hold many rows
    bad = rng.random(n_entities) < 0.04  # a few entities hold the cases
    entity = np.repeat(np.arange(n_entities), sizes)
    rate = np.where(bad[entity], 0.35, 0.002)
    y = (rng.random(len(entity)) < rate).astype(int)
    s = np.clip(rate * 2 * rng.lognormal(0, 0.5, len(entity)) + rng.normal(0, 0.02, len(entity)), 0.001, 0.999)
    value = rng.lognormal(12.4, 0.35, len(entity))
    test = (rng.random(n_entities) < 0.2)[entity]
    return (y[~test], s[~test], value[~test], entity[~test]), (y[test], s[test], value[test], entity[test])


@unittest.skipIf(MISSING, MISSING)
class HoldoutIntervalTest(unittest.TestCase):
    def test_on_fraud_like_data_it_holds_about_nine_in_ten_with_misses_of_its_own_size(self):
        spec = DecisionSpec(action="review", value_column="fare", cost_per_action=25000.0, capacity_share=0.005)
        inside, z = 0, []
        for seed in range(30):
            (y, s, v, g), (yt, st, vt, gt) = world(seed)
            out = choose_operating_point(y, s, spec, value_per_case=v, groups=g, y_test=yt, score_test=st, value_test=vt,
                                         bootstrap=200, seed=seed)
            h = out["holdout"]
            lo, hi = h["prediction_interval"]
            self.assertIn(f"({len(yt)} rows", h["prediction_interval_method"])  # sized by the holdout's rows
            inside += lo <= h["expected_value_per_row"] <= hi
            z.append((h["expected_value_per_row"] - (lo + hi) / 2) / ((hi - lo) / 2 / 1.645))
        self.assertGreaterEqual(inside, 24)  # about nine in ten (24 of 30 or more)
        rms = float(np.sqrt(np.mean(np.square(z))))
        self.assertLess(rms, 1.35)  # misses of about the size it says (the old one's were 1.29 on the replica)
        self.assertGreater(rms, 0.6)  # and not wastefully wide

    def test_rows_with_no_entity_never_shrink_the_samples(self):
        """The review of 0.114.0: about 150 rides without a biker each count as their own entity in training; a sample of
        the holdout's count of entities came out a third smaller than the holdout. Samples are sized by rows."""
        import re

        (y, s, v, g), (yt, st, vt, gt) = world(1)
        g = g.astype(object)
        g[:150] = None  # rows with no entity, in training only
        spec = DecisionSpec(action="review", value_column="fare", cost_per_action=25000.0, capacity_share=0.005)
        out = choose_operating_point(y, s, spec, value_per_case=v, groups=g, y_test=yt, score_test=st, value_test=vt, bootstrap=100)
        drawn = int(re.search(r"about (\d+) drawn", out["holdout"]["prediction_interval_method"]).group(1))
        self.assertLess(abs(drawn - len(yt)) / len(yt), 0.15)  # about the holdout's rows, never a third fewer

    def test_a_short_history_holdout_like_a_time_and_entity_split_holds(self):
        """The review of 0.114.0: on a time split with entities the holdout's entities are new and have few rows; a sample
        of their count of training entities was several times too large."""
        rng = np.random.default_rng(5)
        spec = DecisionSpec(action="review", value_column="fare", cost_per_action=25000.0, capacity_share=0.005)
        inside = 0
        for seed in range(30):
            (y, s, v, g), _ = world(seed, n_entities=600)
            r = np.random.default_rng(seed + 100)
            k = 400  # the holdout: new entities with about 9 rows each
            gt = np.repeat(np.arange(10_000, 10_000 + k), r.integers(5, 14, k))
            bad = r.random(k) < 0.04
            rate = np.where(np.isin(gt, np.arange(10_000, 10_000 + k)[bad]), 0.35, 0.002)
            yt = (r.random(len(gt)) < rate).astype(int)
            st = np.clip(rate * 2 * r.lognormal(0, 0.5, len(gt)) + r.normal(0, 0.02, len(gt)), 0.001, 0.999)
            vt = r.lognormal(12.4, 0.35, len(gt))
            out = choose_operating_point(y, s, spec, value_per_case=v, groups=g, y_test=yt, score_test=st, value_test=vt, bootstrap=200, seed=seed)
            lo, hi = out["holdout"]["prediction_interval"]
            inside += lo <= out["holdout"]["expected_value_per_row"] <= hi
        self.assertGreaterEqual(inside, 24)

    def test_one_folds_rows_a_sample_and_the_training_interval_never_moves(self):
        (y, s, v, g), (yt, st, vt, _) = world(2)
        folds = np.random.default_rng(0).integers(0, 5, len(y))
        spec = DecisionSpec(action="review", value_column="fare", cost_per_action=25000.0, capacity_share=0.005)
        with_holdout = choose_operating_point(y, s, spec, value_per_case=v, groups=g, y_test=yt, score_test=st, value_test=vt,
                                              folds=folds, bootstrap=100)
        self.assertIn("one fold's rows each", with_holdout["holdout"]["prediction_interval_method"])
        alone = choose_operating_point(y, s, spec, value_per_case=v, groups=g, bootstrap=100)
        self.assertEqual(with_holdout["expected_value_per_row_interval"], alone["expected_value_per_row_interval"])  # its own stream
        with self.assertRaises(ValueError):
            choose_operating_point(y, s, spec, value_per_case=v, groups=g, y_test=yt, score_test=st, value_test=vt, folds=folds[:-1])

    def test_without_a_holdout_or_a_draw_nothing_changes(self):
        (y, s, v, g), _ = world(0)
        spec = DecisionSpec(action="review", value_column="fare", cost_per_action=25000.0, capacity_share=0.005)
        out = choose_operating_point(y, s, spec, value_per_case=v, groups=g, bootstrap=50)
        self.assertNotIn("holdout", out)


if __name__ == "__main__":
    unittest.main()
