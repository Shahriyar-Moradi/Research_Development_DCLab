"""Packages 15.4 and 15.5: entity behaviour features fitted inside folds (dclab_core.behaviour) and the rule, anomaly
and PU families of the rare-event screen (dclab_rnd/expansion/detectors.py)."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rides(n=600, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-03-01")
    lat, lon = 35.7 + rng.normal(0, 0.02, n), 51.4 + rng.normal(0, 0.02, n)
    return pd.DataFrame({"biker": rng.integers(0, 40, n), "at": start + pd.to_timedelta(np.sort(rng.integers(0, 30 * 86400, n)), unit="s"),
                         "source_lat": lat, "source_lon": lon, "destination_lat": lat + rng.normal(0, 0.01, n),
                         "destination_lon": lon + rng.normal(0, 0.01, n), "fare": rng.gamma(3, 30, n)})


class BehaviourTest(unittest.TestCase):
    def model(self, mode):
        from dclab_core.behaviour import EntityBehaviour

        return EntityBehaviour(["biker"], time_column="at", mode=mode, numeric=["fare"])

    def test_no_test_row_changes_a_training_feature(self):
        frame = rides()
        y = pd.Series((np.random.default_rng(1).random(len(frame)) < 0.1).astype(int))
        fit, other = frame.iloc[:400].reset_index(drop=True), frame.iloc[400:].reset_index(drop=True)
        for mode in ("audit", "event"):
            with self.subTest(mode=mode):
                model = self.model(mode).fit(fit, y.iloc[:400])
                alone = model.transform(fit)
                changed = other.assign(fare=other["fare"] * 50, biker=0)  # the test rows change completely
                model.transform(changed)
                pd.testing.assert_frame_equal(model.transform(fit), alone)  # the training rows' features do not move
                again = self.model(mode).fit(fit, pd.Series(np.zeros(400)))  # labels change only the shrunk rates
                same = [c for c in alone.columns if not c.endswith("_rate")]
                pd.testing.assert_frame_equal(again.transform(fit)[same], alone[same])

    def test_the_event_mode_uses_no_later_row(self):
        frame = rides()
        model = self.model("event").fit(frame, pd.Series(np.zeros(len(frame))))
        full = model.transform(frame)
        cut = 300
        early = model.transform(frame.iloc[:cut])
        later = frame.iloc[cut:].assign(fare=1e6, destination_lat=frame["destination_lat"].iloc[0])
        mixed = model.transform(pd.concat([frame.iloc[:cut], later], ignore_index=True))
        pd.testing.assert_frame_equal(mixed.iloc[:cut].reset_index(drop=True), early)  # later rows change nothing earlier
        self.assertTrue((full["biker_count"].iloc[:5] <= 4).all())  # counts of earlier rows only

    def test_a_fit_row_never_sees_its_own_label_and_a_grouped_entity_has_no_rate(self):
        from dclab_core.behaviour import SHRINK, EntityBehaviour

        frame = rides()
        y = pd.Series((np.random.default_rng(3).random(len(frame)) < 0.2).astype(int))
        model = self.model("audit").fit(frame, y)
        rate = model.transform(frame)["biker_rate"].to_numpy()
        prior = y.mean()
        for i in (0, 7, 42):
            same = frame["biker"] == frame["biker"].iloc[i]
            others = y[same].sum() - y.iloc[i]
            self.assertAlmostEqual(rate[i], (others + SHRINK * prior) / (same.sum() - 1 + SHRINK))  # leave-one-out
        flipped = y.copy()
        flipped.iloc[0] = 1 - y.iloc[0]
        self.assertAlmostEqual(self.model("audit").fit(frame, flipped).transform(frame)["biker_rate"].iloc[0], rate[0], places=2)  # only the prior moves
        grouped = EntityBehaviour(["biker"], time_column="at", mode="audit", grouped=["biker"]).fit(frame, y).transform(frame)
        self.assertNotIn("biker_rate", grouped)  # a test biker is never seen: its rate would always be the mean

    def test_the_audit_window_ends_with_the_rows_day(self):
        frame = rides()
        model = self.model("audit").fit(frame, pd.Series(np.zeros(len(frame))))
        days = pd.to_datetime(frame["at"]).dt.floor("D")
        cut = days.iloc[300]
        early = frame[days < cut]
        alone = model.transform(early)
        later = frame[days >= cut].assign(biker=frame["biker"].iloc[0], fare=1e6)
        mixed = model.transform(pd.concat([early, later], ignore_index=True))
        pd.testing.assert_frame_equal(mixed.iloc[:len(early)].reset_index(drop=True), alone)  # no later day reaches back

    def test_the_mode_follows_the_moment_and_places_come_from_coordinates(self):
        from dclab_core.behaviour import coordinate_pairs, mode_for

        self.assertEqual(mode_for("After delivery, in a daily audit of the day's rides"), "audit")
        self.assertEqual(mode_for("When the payment is authorised"), "event")
        self.assertEqual(mode_for("After the ride, from its own details"), "event")  # one ride, not a window
        self.assertEqual([p[0] for p in coordinate_pairs(list(rides().columns))], ["source", "destination"])
        out = self.model("audit").fit(rides(), pd.Series(np.zeros(600))).transform(rides())
        self.assertIn("trip_metres", out)
        self.assertIn("biker_destination_repeat", out)
        self.assertIn("fare_per_metre", out)


class RareFamiliesTest(unittest.TestCase):
    def test_detectors_never_see_labels_or_resampled_rows(self):
        from dclab_rnd.expansion import runner

        rng = np.random.default_rng(2)
        X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
        y = pd.Series((rng.random(500) < 0.05).astype(int))
        X.loc[y == 1, "a"] += 6  # the cases are unusual

        class Bundle:
            task_type = "binary_imbalanced"
            spec = type("Spec", (), {"settings": {"fit_negative_fraction": 0.2}, "key": "t", "categorical_columns": ()})()

        recipe = mock.Mock(sparse=False)
        recipe.transform = lambda frame: (frame.to_numpy(), list(frame.columns))
        with mock.patch.object(runner, "make_recipe", return_value=recipe):
            for family in ("percentile_rules", "ecod", "isolation_forest"):
                with self.subTest(family=family):
                    out = runner.fit_and_predict(Bundle, "raw", family, None, X, y, X)
                    self.assertEqual(out["fit_rows"], len(X))  # every fit row: no resampling precedes a detector
                    shuffled = runner.fit_and_predict(Bundle, "raw", family, None, X, y.sample(frac=1, random_state=0).reset_index(drop=True), X)
                    np.testing.assert_allclose(out["predictions"], shuffled["predictions"])  # labels play no part
                    self.assertGreater(out["predictions"][y == 1].mean(), out["predictions"][y == 0].mean())
            pu = runner.fit_and_predict(Bundle, "raw", "pu_bagging", None, X, y, X)
            self.assertLess(pu["fit_rows"], len(X))  # a classifier may be fitted on resampled rows
            self.assertGreater(pu["predictions"][y == 1].mean(), pu["predictions"][y == 0].mean())

    def test_the_rare_event_rule_prefers_the_simpler_family_within_noise(self):
        from dclab_rnd.studio.engine import _rare_choice

        row = lambda model, folds: {"model": model, "fold_metrics": [{"average_precision": v} for v in folds],  # noqa: E731
                                     "metrics": {"average_precision": {"mean": float(np.mean(folds)), "std": float(np.std(folds))}}}
        near = [row("xgboost", [0.62, 0.40, 0.70]), row("isolation_forest", [0.60, 0.45, 0.62])]
        chosen, why = _rare_choice(near, "average_precision", True)
        self.assertEqual(chosen["model"], "isolation_forest")
        self.assertIn("simpler", why["why"])
        far = [row("xgboost", [0.62, 0.64, 0.66]), row("isolation_forest", [0.20, 0.22, 0.21])]
        self.assertEqual(_rare_choice(far, "average_precision", True)[0]["model"], "xgboost")


if __name__ == "__main__":
    unittest.main()
