"""Package 16.5: ranking. Nothing after the recommendation moment is read (changing every interaction after it changes
no list), the ranker's features end before its own training window, no item created after the moment is recommended
even when it has earlier interactions, item columns that know the future are found from the data (for any event),
the model is chosen before the moment's window, the metrics agree with hand-worked values, and the kind is refused by
the engine until the engine runs it."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import ranking  # noqa: E402
from dclab_core.replicas import shop_interactions  # noqa: E402

COLS = dict(user="user_id", item="item_id", time="time", event="event", relevant="purchase")


class MetricTest(unittest.TestCase):
    def test_ndcg_by_hand(self):
        # relevant at ranks 1 and 3 of 3; three relevant items, so the ideal fills all three ranks
        self.assertAlmostEqual(ranking._ndcg(["a", "x", "b"], {"a", "b", "c"}, 3), (1 + 1 / 2) / (1 + 1 / np.log2(3) + 1 / 2))
        self.assertAlmostEqual(ranking._ndcg(["x", "a"], {"a"}, 2), (1 / np.log2(3)) / 1.0)  # one relevant: the ideal is rank 1 only

    def test_the_baseline_is_never_the_chosen_model(self):
        scores = {"popularity": {"ndcg@10": 0.3}, "mf": {"ndcg@10": 0.1}, "ranker": {"ndcg@10": 0.2}}
        self.assertEqual(ranking.choose(scores, 10), "ranker")
        self.assertIsNone(ranking.choose({"popularity": {"ndcg@10": 0.3}}, 10))

    def test_recall_coverage_novelty_and_the_bootstrap_by_hand(self):
        # popularity at the origin: i1 bought 3 times, i2 once, i3 never (seen once); two users buy after it
        rows = [("u1", "i1", "2025-01-01", "purchase"), ("u2", "i1", "2025-01-01", "purchase"), ("u3", "i1", "2025-01-02", "purchase"),
                ("u1", "i2", "2025-01-02", "purchase"), ("u2", "i3", "2025-01-02", "view"),
                ("u1", "i3", "2025-01-05", "purchase"), ("u2", "i1", "2025-01-05", "purchase"), ("u2", "i2", "2025-01-05", "purchase")]
        log = pd.DataFrame(rows, columns=["user_id", "item_id", "time", "event"])
        items = pd.DataFrame({"item_id": ["i1", "i2", "i3"], "created_at": "2024-12-01", "category": "c"})
        found = ranking._origin(log.assign(_t=pd.to_datetime(log["time"])), items, pd.Timestamp("2025-01-03"),
                                {"user": "user_id", "item": "item_id", "event": "event", "relevant": {"purchase"}, "created": "created_at",
                                 "category": "category", "attributes": []}, 3, 1, 0)
        s = ranking._scores(found, 1, 0)["popularity"]
        self.assertEqual(s["recall@1"], 0.25)  # top 1 is i1: u1 bought i3 (0 of 1), u2 bought i1 and i2 (1 of 2)
        self.assertAlmostEqual(s["coverage"], 1 / 3, places=3)  # one item of three recommended
        pop = np.array([3 + 0, 1, 0.3])  # purchases weigh 1, other events 0.3
        self.assertAlmostEqual(s["novelty"], -np.log2(pop[0] / pop.sum()), places=3)  # the rarer, the higher
        frame = pd.DataFrame({"user": [f"u{i}" for i in range(50)] * 2, "model": ["popularity"] * 50 + ["mf"] * 50,
                              "ndcg": [0.0] * 50 + list(np.linspace(0, 0.4, 50)), "history": True, "recall": 0.0,
                              "items": [["i1"]] * 100})
        boot = ranking._scores({"state": found["state"], "truth": {f"u{i}": set() for i in range(50)}, "frame": frame}, 1, 0)["mf"]["against_popularity"]
        lo, hi = boot["ci90"]
        self.assertTrue(lo < boot["ndcg_gain"] < hi and hi - lo > 0.02)


class RankingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replica = shop_interactions.generate(0, users=400, items=150, days=90)
        cls.moment = cls.replica.facts["moment"]
        cls.items = cls.replica.items.drop(columns=["lifetime_sales"])
        cls.args = dict(**COLS, moment=cls.moment, created="created_at", category="category", item_features=["price"])
        cls.result = ranking.evaluate(cls.replica.interactions, cls.items, **cls.args)

    def test_nothing_after_the_moment_is_read(self):
        log = self.replica.interactions.copy()
        later = pd.to_datetime(log["time"]) > pd.Timestamp(self.moment)
        future = log[later].copy()
        future["item_id"] = np.random.default_rng(1).permutation(future["item_id"].to_numpy())  # a different future
        again = ranking.evaluate(pd.concat([log[~later], future]), self.items, **self.args)
        shared = set(self.result["lists"]["popularity"]) & set(again["lists"]["popularity"])
        self.assertTrue(shared)
        for model in ("popularity", "recent_popularity", "mf", "ranker"):
            for u in shared:
                self.assertEqual(self.result["lists"][model][u], again["lists"][model][u], (model, u))
        self.assertEqual(again["chosen"], self.result["chosen"])  # the choice was made before the moment's window
        self.assertEqual(again["validation"]["scores"], self.result["validation"]["scores"])

    def test_the_rankers_features_end_before_its_training_window(self):
        t = self.result["ranker_trained_on"]
        window_start = pd.Timestamp(self.moment) - pd.Timedelta(days=self.result["window_days"])
        self.assertLessEqual(pd.Timestamp(t["features_until"]), window_start)
        self.assertLessEqual(pd.Timestamp(t["labels_until"]), pd.Timestamp(self.moment))
        self.assertGreater(pd.Timestamp(t["labels_until"]), window_start)

    def test_an_item_created_after_the_moment_is_never_recommended(self):
        log = self.replica.interactions
        best = log[(log.event == "purchase") & (pd.to_datetime(log.time) <= pd.Timestamp(self.moment))]["item_id"].value_counts().index[0]
        items = self.items.copy()
        items.loc[items.item_id == best, "created_at"] = (pd.Timestamp(self.moment) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        r = ranking.evaluate(log, items, **self.args)  # the top seller, its interactions kept, now "created" after the moment
        for lists in r["lists"].values():
            self.assertTrue(all(best not in ranked for ranked in lists.values()))
        self.assertEqual(set(r["created_after_recommended"].values()), {0})
        self.assertIn(best, self.result["lists"]["popularity"][next(iter(self.result["lists"]["popularity"]))])  # it was on top before

    def test_every_model_is_compared_with_popularity_and_chosen_before_the_window(self):
        s = self.result["scores"]
        self.assertEqual(set(s), {"popularity", "recent_popularity", "mf", "ranker"})
        self.assertEqual(s["popularity"]["against_popularity"]["ndcg_gain"], 0.0)
        v = self.result["validation"]["scores"]
        self.assertEqual(self.result["chosen"], max((m for m in v if m != "popularity"), key=lambda m: v[m]["ndcg@10"]))
        self.assertTrue(self.result["beats_popularity"][self.result["chosen"]])

    def test_item_columns_that_know_the_future_are_found_for_any_event(self):
        log = self.replica.interactions
        t = pd.to_datetime(log.time)
        at = pd.Timestamp(self.moment)
        items = self.replica.items.copy()

        def count(mask):
            return log[mask].groupby("item_id").size().reindex(items.item_id, fill_value=0).to_numpy()
        purchase, view = log.event == "purchase", log.event == "view"
        items["lifetime_views"] = count(view)
        items["sales_7d_after"] = count(purchase & (t > at) & (t <= at + pd.Timedelta(days=7)))
        items["sales_so_far"] = count(purchase & (t <= at))  # honest: up to the moment
        items["views_30d"] = count(view & (t <= at) & (t > at - pd.Timedelta(days=30)))
        found = ranking.item_check(items, log, **COLS, moment=self.moment, created="created_at")
        self.assertEqual(set(found["with_the_future"]), {"lifetime_sales", "lifetime_views", "sales_7d_after"})
        self.assertGreater(found["created_after"], 0)

    def test_a_count_column_is_kept_out_of_the_ranker(self):
        log = self.replica.interactions
        items = self.items.assign(sales_so_far=log[log.event == "purchase"].groupby("item_id").size().reindex(self.items.item_id, fill_value=0).to_numpy())
        r = ranking.evaluate(log, items, **{**self.args, "item_features": ["price", "sales_so_far"]})
        self.assertEqual(r["item_features"], ["price"])
        self.assertIn("one moment only", r["kept_out_of_ranker"]["sales_so_far"])

    def test_edges_are_refused_plainly(self):
        log = self.replica.interactions
        with self.assertRaises(ValueError):
            ranking.evaluate(log, self.items, **{**self.args, "moment": "2030-01-01"})  # nothing in the window after it
        with self.assertRaises(ValueError):
            ranking.evaluate(log, self.items, **{**self.args, "moment": "2024-01-01"})  # nothing known before it
        with self.assertRaises(ValueError):
            ranking._matrix(log.head(5), list(range(50_000)), list(range(1_000)), "user_id", "item_id", "event", "purchase")


class KindTest(unittest.TestCase):
    def setUp(self):
        try:
            from dclab_rnd import kinds
        except ImportError as error:
            self.skipTest(str(error))
        self.kinds = kinds

    def test_the_acceptance_holds_and_fails_when_the_leak_is_kept(self):
        kind = self.kinds.get("ranking")
        case = kind.case(0)
        run = kind.solve(type(case)(**{**case.__dict__, "facts": {}}))  # solving never reads the answer
        decisions = kind.accept({**run, "facts": case.facts})
        self.assertTrue(all(c["ok"] for c in decisions), decisions)
        leaked = {**run, "facts": case.facts, "evaluation": {**run["evaluation"], "item_features": run["evaluation"]["item_features"] + ["lifetime_sales"]}}
        self.assertFalse(kind.accept(leaked)[1]["ok"])
        no_ranker = {**run, "facts": case.facts, "evaluation": {**run["evaluation"], "scores": {m: v for m, v in run["evaluation"]["scores"].items() if m != "ranker"}}}
        self.assertFalse(kind.accept(no_ranker)[3]["ok"])

    def test_detected_only_with_an_item_to_recommend_and_refused_by_the_engine(self):
        from dclab_rnd.studio.solution import Solution

        self.assertEqual(self.kinds.detect("Recommend products each customer will buy next")["key"], "ranking")
        for other in ("Predict whether a review is negative so we can recommend a reply", "Recommend whether to approve each loan",
                      "Predict churn so we can recommend a retention offer"):
            self.assertNotEqual(self.kinds.detect(other)["key"], "ranking", other)
        with self.assertRaises(ValueError):
            Solution(target="purchase", task="binary", prediction_moment="On the day the list is shown", kind="ranking",
                     kind_fields={"user_column": "u", "item_column": "i", "time_column": "t", "event_column": "e",
                                  "relevant_event": "purchase", "moment": "2025-04-10"})


if __name__ == "__main__":
    unittest.main()
