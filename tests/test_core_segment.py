"""Package 16.6: segmentation. With no outcome to predict, segments are judged by stability under resampling and by
separation: the planted behaviours are recovered with a stability interval, more segments than planted are rejected as
unstable, columns no group uses are left out with the reason, the preprocessing is recorded, each segment is described
in the data's own columns, and the record says there is no ground truth."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import segment  # noqa: E402
from dclab_core.replicas import customer_behaviours  # noqa: E402


class PrepareTest(unittest.TestCase):
    def test_every_preprocessing_choice_is_recorded(self):
        frame = pd.DataFrame({"amount": [1.0, 2, 3, 100, 2, 1, 4, 3], "flat": 1, "id": [f"x{i}" for i in range(8)],
                              "colour": ["red", "blue", "red", "blue", "red", "red", "blue", "red"]})
        X, record = segment.prepare(frame)
        self.assertIn("log", record["numbers"]["amount"])
        self.assertIn("flat", record["left_out"])
        self.assertIn("id", record["left_out"])
        self.assertEqual(X.shape, (8, 1 + 2))
        np.testing.assert_allclose(np.abs(X[:, 1:]).sum(axis=1), np.full(8, 1 / np.sqrt(2)))  # a category weighs 1/sqrt(levels)

    def test_stability_is_high_for_real_groups_and_low_for_none(self):
        rng = np.random.default_rng(0)
        groups = np.vstack([rng.normal(c, 0.3, (150, 2)) for c in ((0, 0), (4, 0), (0, 4))])
        self.assertGreater(segment.stability("kmeans", 3, groups, resamples=8)["ci90"][0], 0.9)
        noise = rng.uniform(0, 1, (450, 2))  # no groups: four of them do not come back the same
        self.assertLess(segment.stability("kmeans", 7, noise, resamples=8)["ci90"][0], 0.75)


class ReplicaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replica = customer_behaviours.generate(0)
        cls.frame = cls.replica.customers.assign(renewed=cls.replica.facts["renewed"])
        cls.result = segment.segment(cls.frame, ks=range(2, 8), outcome="renewed", resamples=10)

    def test_the_planted_behaviours_are_recovered_and_too_many_rejected(self):
        from sklearn.metrics import adjusted_rand_score

        r, truth = self.result, np.array(self.replica.facts["behaviour"])
        self.assertEqual((r["chosen"]["method"], r["chosen"]["k"]), ("kmeans", 3))
        self.assertGreaterEqual(r["chosen"]["stability"]["ci90"][0], r["min_stability"])
        planted = truth != 3
        self.assertGreater(adjusted_rand_score(truth[planted], np.array(r["labels"])[planted]), 0.9)
        above = [x for x in r["rejected_as_unstable"] if x["method"] == "kmeans" and x["k"] > 3]  # the chosen method's own
        self.assertTrue(above)
        self.assertIn("lower end of its 90% stability interval", above[0]["why"])
        mixture = r["candidates"]["mixture:3"]  # the mixture holds on a table with categories too (reg_covar)
        self.assertGreaterEqual(mixture["stability"]["ci90"][0], r["min_stability"])

    def test_columns_no_group_uses_are_left_out_and_the_outcome_never_forms_the_groups(self):
        r = self.result
        self.assertEqual(set(r["passes"]["unused_columns"]), {"region", "signup_week"})
        self.assertIn("region", r["preprocessing"]["left_out"])
        self.assertNotIn("renewed", r["passes"]["first"]["columns"])  # the outcome describes, it never forms the groups
        self.assertIn("does not make them right", r["outcome"]["note"])
        for g in r["outcome"]["by_segment"].values():
            self.assertLessEqual(g["ci95"][0], g["rate"])

    def test_segments_are_described_in_the_datas_columns_with_the_limit(self):
        r = self.result
        self.assertEqual(len(r["profiles"]), 3)
        columns = set(self.frame.columns)
        for p in r["profiles"]:
            self.assertTrue(p["sets_it_apart"])
            self.assertTrue(all(d["column"] in columns for d in p["sets_it_apart"]))
        self.assertIn("no ground truth", r["limit"])
        self.assertIn("density_view", r)

    def test_data_without_groups_gets_no_segments_and_one_column_with_two_humps_does(self):
        rng = np.random.default_rng(1)
        flat = pd.DataFrame({"a": rng.uniform(size=400), "b": rng.uniform(size=400)})
        r = segment.segment(flat, ks=range(2, 7), resamples=8)
        self.assertIsNone(r["chosen"])
        self.assertIn("no number of segments", r["why"])
        self.assertTrue(r["rejected_as_not_separated"])  # k-means splits a square stably: stable is not the same as grouped
        blob = pd.DataFrame(rng.normal(size=(500, 3)), columns=["a", "b", "c"])
        self.assertIsNone(segment.segment(blob, ks=range(2, 5), resamples=8)["chosen"])
        humps = pd.DataFrame({"x": np.r_[rng.normal(-3, 0.5, 450), rng.normal(3, 0.5, 450)], "n1": rng.normal(size=900), "n2": rng.normal(size=900)})
        two = segment.segment(humps, ks=range(2, 6), resamples=8)
        self.assertEqual(two["chosen"]["k"], 2)  # the groups live in one column's shape: still groups

    def test_a_column_that_only_a_finer_grouping_uses_is_kept(self):
        rng = np.random.default_rng(2)
        g = rng.choice([0, 1, 2], 1200, p=[0.5, 0.25, 0.25])  # A against B and C on four columns; z alone tells B from C
        cols = {f"a{i}": np.where(g == 0, 2.0, -2.0) + rng.normal(0, 0.4, 1200) for i in range(4)}
        cols["z"] = np.select([g == 1, g == 2], [-3.0, 3.0], 0.0) + rng.normal(0, 0.4, 1200)
        cols.update({f"n{i}": rng.normal(size=1200) for i in range(3)})
        r = segment.segment(pd.DataFrame(cols), ks=range(2, 6), resamples=8)
        self.assertNotIn("z", r["passes"]["unused_columns"])
        self.assertEqual(set(r["passes"]["unused_columns"]), {"n0", "n1", "n2"})
        self.assertEqual(r["chosen"]["k"], 3)

    def test_small_tables_text_outcomes_and_missing_values_are_handled_and_said(self):
        tiny = pd.DataFrame({"a": [1.0, 1.1, 0.9, 5.0, 5.1, 4.9, 1.0, 5.0], "b": [0, 0, 0, 1, 1, 1, 0, 1]})
        self.assertIn("density_view", segment.segment(tiny, ks=range(2, 4), resamples=4))
        r = self.result
        words = self.frame.assign(renewed=np.where(self.frame["renewed"] == 1, "yes", "no"))
        said = segment.segment(words, ks=range(3, 4), outcome="renewed", resamples=4)
        self.assertIn("'yes' counted as 1", said["outcome"]["levels"])
        self.assertTrue(all(v["rows"] > 0 for v in said["outcome"]["by_segment"].values()))
        missing = segment.segment(self.frame, ks=range(3, 4), outcome="renewal", resamples=4)
        self.assertIn("not a column", missing["outcome"]["note"])
        holes = self.frame.copy()
        holes.loc[:20, "avg_basket"] = np.nan
        holes.loc[:20, "channel"] = None
        holes["number_id"] = np.arange(len(holes))
        _, record = segment.prepare(holes.drop(columns=["renewed"]))
        self.assertIn("missing filled with the median", record["numbers"]["avg_basket"])
        self.assertIn("(missing)", record["categories"]["channel"])
        self.assertIn("identifier", record["left_out"]["number_id"])
        self.assertTrue(r["chosen"])


class KindTest(unittest.TestCase):
    def setUp(self):
        try:
            from dclab_rnd import kinds
        except ImportError as error:
            self.skipTest(str(error))
        self.kinds = kinds

    def test_the_acceptance_holds_and_reads_no_answer_while_solving(self):
        kind = self.kinds.get("segmentation")
        case = kind.case(0)
        run = kind.solve(type(case)(**{**case.__dict__, "facts": {}}))  # solving never reads the planted behaviours
        decisions = kind.accept({**run, "facts": case.facts})
        self.assertTrue(all(c["ok"] for c in decisions), decisions)

    def test_detected_by_its_words_and_refused_by_the_engine(self):
        from dclab_rnd.studio.solution import Solution

        self.assertEqual(self.kinds.detect("Segment our customers into groups for campaigns")["key"], "segmentation")
        for other in ("Predict which customers will churn next month, reported per customer segment",
                      "Predict delivery time; the data has a city cluster column"):
            self.assertNotEqual(self.kinds.detect(other)["key"], "segmentation", other)
        with self.assertRaises(ValueError):
            Solution(target="renewed", task="multiclass", prediction_moment="Now", kind="segmentation", kind_fields={})


if __name__ == "__main__":
    unittest.main()
