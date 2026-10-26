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



@unittest.skipIf(MISSING, MISSING)
class NearestRowsTest(unittest.TestCase):
    """0.118.0, time_decision_v1's next step: on a drifting table, the samples come from the rows nearest the holdout."""

    def drifting(self, seed=0, n=3000, n_test=600):
        rng = np.random.default_rng(seed)
        level = np.linspace(-1.6, -0.9, n + n_test)  # later rows are cases more often
        x = rng.normal(size=n + n_test)
        p = 1 / (1 + np.exp(-(level + x)))
        y = (rng.random(n + n_test) < p).astype(int)
        return y[:n], p[:n], y[n:], p[n:]

    def test_the_interval_moves_toward_the_latest_rows(self):
        y, s, yt, st = self.drifting()
        spec = DecisionSpec(action="call", value_per_case=120, cost_per_action=10, capacity_share=0.1)
        latest = np.arange(len(y)) >= int(0.75 * len(y))
        every = choose_operating_point(y, s, spec, y_test=yt, score_test=st, bootstrap=200)["holdout"]
        near = choose_operating_point(y, s, spec, y_test=yt, score_test=st, bootstrap=200, sample_rows=latest)["holdout"]
        self.assertGreater(sum(near["prediction_interval"]), sum(every["prediction_interval"]))  # the later, case-richer rows
        self.assertIn("nearest the holdout", near["prediction_interval_method"])
        self.assertEqual(near["expected_value_per_row"], every["expected_value_per_row"])  # the holdout's reading is the same
        self.assertIn("assumes the world stays as it was", near["drift"])  # time_decision_v2: said beside the reading
        self.assertNotIn("drift", every)

    def test_entities_are_drawn_from_the_nearest_ones_and_a_draw_without_them_never_mixes(self):
        y, s, yt, st = self.drifting(n=3000)
        groups = np.repeat(np.arange(1000), 3)  # three rows an entity, in time order
        spec = DecisionSpec(action="call", value_per_case=120, cost_per_action=10, capacity_share=0.1)
        latest = np.arange(len(y)) >= int(0.75 * len(y))
        every = choose_operating_point(y, s, spec, groups=groups, y_test=yt, score_test=st, bootstrap=200)["holdout"]
        near = choose_operating_point(y, s, spec, groups=groups, y_test=yt, score_test=st, bootstrap=200, sample_rows=latest)["holdout"]
        self.assertGreater(sum(near["prediction_interval"]), sum(every["prediction_interval"]))
        self.assertIn("whole entities", near["prediction_interval_method"])
        one = np.zeros(len(y), bool)
        one[-3:] = True  # a single near entity: most draws hold none of it, and they draw it, never every row
        tiny = choose_operating_point(y, s, spec, groups=groups, y_test=yt, score_test=st, bootstrap=60, sample_rows=one)["holdout"]
        self.assertIsNotNone(tiny["prediction_interval"][0])

    def test_refits_widen_the_interval_by_the_model_s_own_spread(self):
        """0.120.0, time_decision_v4's next step: the samples rank their rows by one of several refits of the model, so
        the interval holds the fit-to-fit spread the final model's ranking has (one call a day hangs on it). Widening is
        an average, not a promise for one table (measured on five: four wider by 10 to 25%, one narrower), so three."""
        spec = DecisionSpec(action="call", value_per_case=120, cost_per_action=10, capacity_share=0.1)
        width = lambda h: h["prediction_interval"][1] - h["prediction_interval"][0]  # noqa: E731
        widths = {"near": [], "refit": [], "same": []}
        for seed in (0, 1, 2):
            y, p, yt, st = self.drifting(seed)
            rng = np.random.default_rng(seed + 50)
            latest = np.arange(len(y)) >= int(0.75 * len(y))
            s = np.clip(p + rng.normal(0, 0.15, len(p)), 0, 1)  # the model: the truth seen through one fit's noise
            refits = np.full((8, len(y)), np.nan)
            refits[:, latest] = np.clip(p[latest] + rng.normal(0, 0.15, (8, int(latest.sum()))), 0, 1)  # as good, ranking otherwise
            run = lambda **kw: choose_operating_point(y, s, spec, y_test=yt, score_test=st, bootstrap=200, sample_rows=latest, **kw)["holdout"]  # noqa: E731
            near, refit = run(), run(sample_scores=refits)
            same = run(sample_scores=np.where(latest, s, np.nan)[None, :].repeat(4, axis=0))  # refits equal to the model
            widths["near"].append(width(near)); widths["refit"].append(width(refit)); widths["same"].append(width(same))
            self.assertIn("refits of the model", refit["prediction_interval_method"])
            self.assertEqual(refit["expected_value_per_row"], near["expected_value_per_row"])  # the holdout's reading is the same
        self.assertGreater(np.mean(widths["refit"]), np.mean(widths["near"]))  # the refits' spread is in it
        self.assertAlmostEqual(np.mean(widths["same"]), np.mean(widths["near"]), delta=0.1 * np.mean(widths["near"]))

    def test_refit_scores_are_checked(self):
        y, s, yt, st = self.drifting(n=400, n_test=100)
        spec = DecisionSpec(action="call", value_per_case=120, cost_per_action=10, capacity_share=0.1)
        latest = np.arange(len(y)) >= 300
        good = np.where(latest, s, np.nan)[None, :].repeat(3, axis=0)
        holed = good.copy()
        holed[0, -1] = np.nan  # a near row without a refit's score
        for bad in ({"sample_scores": good},  # no rows to sample from: refits score the latest fold's rows only
                    {"sample_rows": latest, "sample_scores": good[:1]},  # one refit is no spread
                    {"sample_rows": latest, "sample_scores": good[:, :-1]},  # one score per out-of-fold row
                    {"sample_rows": latest, "sample_scores": holed}):
            with self.assertRaises(ValueError, msg=str(list(bad))):
                choose_operating_point(y, s, spec, y_test=yt, score_test=st, bootstrap=20, **bad)

    def test_the_rows_given_are_checked(self):
        y, s, yt, st = self.drifting(n=400, n_test=100)
        spec = DecisionSpec(action="call", value_per_case=120, cost_per_action=10, capacity_share=0.1)
        for bad in ({"sample_rows": np.zeros(len(y), bool)}, {"sample_rows": np.ones(len(y) - 1, bool)},
                    {"sample_rows": np.ones(len(y), bool), "folds": np.zeros(len(y), int)}):
            with self.assertRaises(ValueError, msg=str(list(bad))):
                choose_operating_point(y, s, spec, y_test=yt, score_test=st, bootstrap=20, **bad)


if __name__ == "__main__":
    unittest.main()
