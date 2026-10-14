"""Package 20.7, the library: anomaly detection without labels. The rows to review are the latest whole days; a detector
is chosen on anomalies planted in a validation period before them, never on the rows to review; the chosen one ranks
them into a review list of the budget's size, measured on anomalies planted in a copy of those rows, with its limits."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd

    from dclab_core import anomaly
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""


def table(n=4000, days=10, seed=0):
    rng = np.random.default_rng(seed)
    who = rng.integers(0, 80, n)
    frame = pd.DataFrame({"day": pd.to_datetime("2025-01-01") + pd.to_timedelta(np.sort(rng.integers(0, days, n)), unit="D"),
                          "who": who, "order_id": np.arange(n), "amount": rng.lognormal(3, 0.6, n), "minutes": rng.normal(30, 6, n)})
    frame["distance"] = frame["minutes"] * 0.4 + rng.normal(0, 1, n)
    return frame


@unittest.skipIf(MISSING, MISSING)
class AnomalyTest(unittest.TestCase):
    def test_the_rows_to_review_are_whole_latest_days(self):
        frame = table()
        parts = anomaly.split(frame, "day")
        held, train = parts["holdout"]["day"], parts["train"]["day"]
        self.assertGreater(held.min(), train.max())  # never cut inside a day
        self.assertGreater(parts["validation"]["day"].min(), parts["fit"]["day"].max())
        self.assertEqual(parts["split"]["by"], "day")

    def test_the_choice_never_reads_the_rows_to_review(self):
        frame = table()
        parts = anomaly.split(frame, "day")
        cols = ["amount", "minutes", "distance"]
        clean = anomaly.choose(parts, cols, "who", "day")
        poisoned = dict(parts)
        poisoned["holdout"] = parts["holdout"].assign(amount=-1e6, minutes=np.nan)
        dirty = anomaly.choose(poisoned, cols, "who", "day")
        self.assertEqual((clean["chosen"], clean["validation"]), (dirty["chosen"], dirty["validation"]))

    def test_the_review_list_is_the_budgets_size_and_finds_a_planted_far_value(self):
        frame = table()
        parts = anomaly.split(frame, "day")
        out = anomaly.review(parts, "rules", ["amount", "minutes", "distance"], "who", "day", budget=0.02, key="order_id")
        self.assertEqual(len(out["review_list"]), out["budget_rows"])
        self.assertEqual(out["budget_rows"], round(0.02 * len(parts["holdout"])))
        self.assertGreater(out["planted"]["far_value"]["chosen"], 0.8)
        self.assertTrue(out["limits"] and all(x["key"] is not None for x in out["review_list"]))

    def test_a_planted_pair_is_broken_against_its_direction_and_shapes_are_reported(self):
        frame = table()
        Z = anomaly.Features(["amount", "minutes", "distance"]).fit(frame).transform(frame)
        planted, truth = anomaly.plant(Z, "broken_pair", 40, np.random.default_rng(0))
        self.assertEqual(int(truth.sum()), 40)
        moved = planted.loc[truth == 1, ["minutes", "distance"]]
        self.assertTrue((moved["minutes"] < Z["minutes"].median()).all() and (moved["distance"] > Z["distance"].median()).all())

    def test_a_scored_rows_features_do_not_depend_on_the_days_after_it(self):
        frame = table()
        parts = anomaly.split(frame, "day")
        feats = anomaly.Features(["amount", "minutes"], "who", "day").fit(parts["train"])
        first_day = parts["holdout"]["day"].min()
        one = parts["holdout"][parts["holdout"]["day"] == first_day]
        alone = anomaly.with_history(feats, parts["train"], one)
        with_later = anomaly.with_history(feats, parts["train"], parts["holdout"]).iloc[: len(one)]
        self.assertTrue(np.allclose(alone.to_numpy(), with_later.to_numpy()))  # in audit mode a row sees nothing after its day
        bare = feats.transform(one)  # without history its entity counts would start from nothing
        self.assertFalse(np.allclose(alone.to_numpy(), bare.to_numpy()))

    def test_the_choice_and_the_review_score_their_rows_with_history(self):
        from unittest import mock

        frame = table()
        parts = anomaly.split(frame, "day")
        calls = []
        real = anomaly.with_history

        def spy(feats, history, rows):
            calls.append((len(history), len(rows)))
            return real(feats, history, rows)

        with mock.patch.object(anomaly, "with_history", side_effect=spy):
            anomaly.choose(parts, ["amount", "minutes"], "who", "day")
            anomaly.review(parts, "rules", ["amount", "minutes"], "who", "day")
        self.assertIn((len(parts["fit"]), len(parts["validation"])), calls)
        self.assertIn((len(parts["train"]), len(parts["holdout"])), calls)

    def test_a_table_with_nothing_to_score_is_refused(self):
        frame = table()[["day", "order_id"]]
        with self.assertRaises(ValueError):
            anomaly.choose(anomaly.split(frame, "day"), [], None, "day")

    def test_without_a_time_column_the_split_is_a_seeded_share(self):
        parts = anomaly.split(table().drop(columns=["day"]), None)
        self.assertEqual(parts["split"]["by"], "row")
        self.assertEqual(anomaly.budget_rows(1000, None), 10)
        self.assertEqual(anomaly.budget_rows(1000, 25), 25)


if __name__ == "__main__":
    unittest.main()
