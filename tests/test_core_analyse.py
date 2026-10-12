"""Package 17.3: initial data analysis for decisions (dclab_core.analyse), for binary, multiclass and regression.

Each replica plants traps (dclab_core/replicas); the analysis must name each one with the implication a careful data
scientist draws, with numbers that match a hand computation, and must look only at the training rows it is given for
anything that steers a choice.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pydantic  # noqa: F401
except ImportError as error:
    PYDANTIC_MISSING = str(error)
else:
    PYDANTIC_MISSING = ""


def by_id(findings):
    return {f.id: f for f in findings.findings}


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class MulticlassTicketsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_core.analyse import analyse
        from dclab_core.replicas import ticket_routing

        cls.replica = ticket_routing.generate(seed=0)
        cls.frame = cls.replica.tickets
        cls.train = cls.frame.index[: int(len(cls.frame) * 0.8)]  # the earlier 80 %: the host's training partition
        cls.result = analyse({"tickets.csv": cls.frame}, target="assigned_team", train_index=cls.train)
        cls.f = by_id(cls.result)

    def test_the_task_and_the_tiny_classes(self):
        self.assertEqual(self.result.task, "multiclass")
        tiny = self.f["F-tiny-classes"]
        y = self.frame.loc[self.train, "assigned_team"]
        share = y.value_counts(normalize=True)
        expected = sorted(share[(share < 0.01) | (y.value_counts() < 30)].index)
        self.assertEqual(tiny.evidence["tiny_classes"], expected)
        self.assertIn("merging", tiny.implication)
        self.assertIn("metric.primary", tiny.affects)

    def test_the_label_copy_and_the_post_moment_column(self):
        self.assertEqual(self.f["F-leak-resolved_by"].severity, "critical")
        self.assertIn("features.forbidden", self.f["F-leak-resolved_by"].affects)
        name_only = self.f["F-name-first_response_minutes"]  # sounds post-outcome, predicts little: a question, not a verdict
        self.assertEqual(name_only.severity, "info")
        self.assertNotIn("features.forbidden", name_only.affects)
        for honest in ("plan", "channel", "subject_words"):
            self.assertNotIn(f"F-leak-{honest}", self.f)

    def test_duplicates_counted_on_every_row(self):
        dup = self.f["F-duplicates"]
        self.assertEqual(dup.evidence["duplicate_rows"], int(self.frame.duplicated().sum()))
        self.assertIn("Drop the copies", dup.implication)
        self.assertEqual(self.f["F-keys"].evidence["unique_columns"], ["ticket_id"])  # an id, despite its copied rows
        self.assertEqual(dup.scope, "all_rows")  # a copy across the split is the problem, so all rows count

    def test_customers_repeat_and_a_class_appears_late(self):
        self.assertEqual(self.f["F-entity-customer_id"].severity, "info")  # customers repeat but do not tell the team
        self.assertNotIn("F-entity-ticket_id", self.f)  # an id repeated only by duplicated rows is not an entity
        self.assertIn("integrations", self.f["F-time"].evidence["classes_only_in_last_third"])


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class RegressionDeliveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_core.analyse import analyse
        from dclab_core.replicas import delivery_time

        cls.replica = delivery_time.generate(seed=0)
        cls.frame = cls.replica.orders
        cls.train = cls.frame.index[: int(len(cls.frame) * 0.8)]
        cls.result = analyse({"orders.csv": cls.frame}, target="actual_minutes", train_index=cls.train)
        cls.f = by_id(cls.result)

    def test_the_cap(self):
        self.assertEqual(self.result.task, "regression")
        cap = self.f["F-cap"]
        y = self.frame.loc[self.train, "actual_minutes"]
        self.assertEqual(cap.evidence["cap"], 60.0)
        self.assertAlmostEqual(cap.evidence["share_at_cap"], round(float((y == 60).mean()), 4))
        self.assertIn("censored", cap.implication)

    def test_the_recomputable_target_and_the_rating(self):
        recompute = self.f["F-leak-recompute-order_time-delivered_at"]
        self.assertEqual((recompute.severity, recompute.evidence["later"]), ("critical", "delivered_at"))
        self.assertEqual(self.f["F-leak-courier_rating"].severity, "warning")

    def test_impossible_values_and_speaking_missingness(self):
        self.assertEqual(self.f["F-impossible-distance_km"].evidence["rows"], self.replica.facts["negative_distances"])
        self.assertIn("F-missing-weather", self.f)
        self.assertEqual(self.f["F-missing-weather"].severity, "warning")

    def test_restaurants_drive_the_target(self):
        entity = self.f["F-entity-restaurant_id"]
        self.assertGreaterEqual(entity.evidence["target_from_entity_out_of_fold"], 0.3)
        self.assertEqual(entity.severity, "warning")
        self.assertIn("split.strategy", entity.affects)

    def test_the_long_tail_below_the_cap(self):
        dist = self.f["F-distribution"]
        self.assertGreater(dist.evidence["upper_to_lower_tail"], 1.5)
        self.assertIn("long right tail", dist.implication)


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class BinaryFraudTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_core.analyse import analyse
        from dclab_core.replicas import delivery_fraud
        from dclab_core.understand import understand

        cls.replica = delivery_fraud.generate(seed=0)
        rides = cls.replica.rides
        cls.rides = rides[(rides["status"] == "DELIVERED") & rides["biker_id"].notna()].copy()
        cls.rides["is_fraud"] = cls.rides["order_id"].isin(set(cls.replica.sample["order_id"])).astype(int)
        cls.train = cls.rides.index[: int(len(cls.rides) * 0.8)]
        brief = understand("Detect fraudulent activity by couriers. A sample of known forged rides is in known_frauds.csv.",
                           {"rides.csv": list(rides.columns), "known_frauds.csv": list(cls.replica.sample.columns)})
        cls.result = analyse({"rides.csv": cls.rides, "known_frauds.csv": cls.replica.sample}, target="is_fraud", brief=brief,
                             labels="known_frauds.csv", train_index=cls.train)
        cls.f = by_id(cls.result)

    def test_rare_and_partial(self):
        prevalence = self.f["F-prevalence"]
        self.assertAlmostEqual(prevalence.evidence["positive_rate"], round(float(self.rides.loc[self.train, "is_fraud"].mean()), 5))
        self.assertIn("review budget", prevalence.implication)
        self.assertIn("lower bound", self.f["F-partial-labels"].implication)

    def test_the_join_the_pair_inflation_and_the_column_that_means_something_else(self):
        join = self.f["F-labels-join"]
        self.assertIn("best on order_id", join.statement)
        self.assertEqual(join.evidence["joins"]["order_id"]["label_rows_found"],
                         len(self.replica.sample) - self.replica.facts["sample_rows_not_in_table"])
        pair = self.f["F-labels-inflation-account_id-biker_id"]
        marked = self.rides.set_index(["account_id", "biker_id"]).index.isin(
            list(zip(self.replica.sample["account_id"], self.replica.sample["biker_id"].astype(float)))).sum()
        self.assertEqual(pair.evidence["rows_marked"], int(marked))
        self.assertEqual(pair.severity, "critical")
        self.assertIn("sender_phone", self.f["F-labels-disagree"].evidence["share_equal"])
        self.assertNotIn("F-labels-inflation-status", self.f)  # not every shared column is a key

    def test_frauds_sit_on_a_few_bikers_and_shuffled_labels_say_nothing(self):
        import numpy as np

        from dclab_core.analyse import analyse

        entity = self.f["F-entity-biker_id"]
        self.assertEqual(entity.severity, "warning")
        self.assertIn("Group the folds by biker_id", entity.implication)
        shuffled = self.rides.copy()
        shuffled["is_fraud"] = np.random.default_rng(1).permutation(shuffled["is_fraud"].to_numpy())
        control = by_id(analyse({"rides.csv": shuffled}, target="is_fraud", train_index=self.train))
        self.assertEqual(control["F-entity-biker_id"].severity, "info")
        self.assertFalse([k for k in control if k.startswith("F-leak")])

    def test_coordinates_are_not_identifiers(self):
        self.assertEqual(sorted(self.f["F-keys"].evidence["unique_columns"]), ["order_id", "terminal_ref"])


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class ScopeTest(unittest.TestCase):
    def test_target_statistics_never_look_past_the_training_rows(self):
        from dclab_core.analyse import analyse
        from dclab_core.replicas import ticket_routing

        frame = ticket_routing.generate(seed=0).tickets
        train = frame.index[frame["assigned_team"] != "legal"]  # a class that only exists outside the training rows
        result = by_id(analyse({"t.csv": frame}, target="assigned_team", train_index=train))
        self.assertNotIn("legal", result["F-tiny-classes"].evidence["tiny_classes"])
        self.assertEqual(result["F-classes"].evidence["classes"], 11)

    def test_without_a_split_the_findings_say_so(self):
        from dclab_core.analyse import analyse
        from dclab_core.replicas import delivery_time

        result = analyse({"o.csv": delivery_time.generate(seed=0, n=600).orders}, target="actual_minutes")
        self.assertIsNone(result.training_rows)
        self.assertEqual(by_id(result)["F-no-split"].severity, "warning")

    def test_findings_are_a_valid_contract(self):
        from dclab_core.analyse import analyse
        from dclab_core.contracts import DataFindings
        from dclab_core.replicas import delivery_time

        frame = delivery_time.generate(seed=1, n=800).orders
        result = analyse({"o.csv": frame}, target="actual_minutes", train_index=frame.index[:600])
        self.assertEqual(DataFindings.read(result.model_dump_json()), result)



@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class FailurePathTest(unittest.TestCase):
    """What the review found: each case once, small and synthetic."""

    def frame(self, n=1200, seed=0):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(seed)
        return pd.DataFrame({"x": rng.normal(size=n), "plan": rng.choice(["a", "b", "c"], n), "y": rng.choice([0, 1], n, p=[0.8, 0.2])}), rng

    def test_honest_history_columns_are_not_leaks(self):
        from dclab_core.analyse import analyse

        df, rng = self.frame()
        for name in ("previous_rating", "response_time_sla_target", "delivered_count_last_30d", "avg_rating", "prior_outcome",
                     "last_order_completed"):
            df[name] = rng.normal(size=len(df))
        found = by_id(analyse({"t": df}, target="y", train_index=df.index[:900]))
        self.assertFalse([k for k in found if k.startswith(("F-leak", "F-name"))])

    def test_a_copy_of_a_multiclass_label_is_critical_even_when_one_class_dominates(self):
        from dclab_core.analyse import analyse

        df, rng = self.frame()
        df["y"] = rng.choice(["a", "b", "c"], len(df), p=[0.7, 0.2, 0.1])
        df["status_code"] = df["y"]
        self.assertEqual(by_id(analyse({"t": df}, target="y", train_index=df.index[:900]))["F-leak-status_code"].severity, "critical")

    def test_odd_names_many_classes_and_missing_labels_never_raise(self):
        import numpy as np
        import pandas as pd

        from dclab_core.analyse import analyse

        rng = np.random.default_rng(0)
        n = 2400
        df = pd.DataFrame({"Order Time": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 1e6, n), unit="s"),
                           "Customer ID": rng.integers(0, 300, n), "y": rng.integers(0, 600, n).astype(str)})
        df["Delivered At"] = df["Order Time"] + pd.to_timedelta(rng.integers(1, 90, n), unit="m")
        analyse({"t": df}, target="y", train_index=df.index[:2000])  # 600 classes, spaces in names
        reg = df.assign(y=np.where(rng.random(n) < 0.5, np.nan, ((df["Delivered At"] - df["Order Time"]).dt.total_seconds() / 60)))
        found = by_id(analyse({"t": reg}, target="y", task="regression", train_index=reg.index[:2000]))
        self.assertTrue(any(k.startswith("F-leak-recompute") for k in found))
        empty = reg.assign(y=np.nan)
        analyse({"t": empty}, target="y", task="regression", train_index=empty.index[:2000])

    def test_many_disagreeing_columns_are_capped(self):
        import numpy as np
        import pandas as pd

        from dclab_core.analyse import analyse

        rng = np.random.default_rng(0)
        main = pd.DataFrame({"order_id": np.arange(500), **{f"c{i}": rng.normal(size=500) for i in range(40)}})
        labels = main.sample(50, random_state=0).copy()
        for i in range(40):
            labels[f"c{i}"] = rng.normal(size=50)
        found = by_id(analyse({"main": main.assign(y=main["order_id"].isin(labels["order_id"]).astype(int)), "labels": labels},
                              target="y", labels="labels", train_index=main.index[:400]))
        self.assertEqual(found["F-labels-disagree"].evidence["count"], 40)
        self.assertLessEqual(len(found["F-labels-disagree"].evidence["share_equal"]), 20)

    def test_text_time_columns_are_found_with_the_string_dtype(self):
        import numpy as np
        import pandas as pd

        from dclab_core.analyse import _time_columns

        n = 300
        df = pd.DataFrame({"created_at": pd.Series(pd.date_range("2025-01-01", periods=n, freq="h").astype(str), dtype="string"),
                           "x": np.arange(n)})
        self.assertEqual(_time_columns(df), ["created_at"])

    def test_labels_coded_one_and_two_take_the_rarer_as_positive(self):
        import numpy as np
        import pandas as pd

        from dclab_core.analyse import _positive

        y = pd.Series(np.r_[np.ones(995), np.full(5, 2)])
        self.assertEqual(_positive(y), 2)
        self.assertEqual(_positive(pd.Series(["no"] * 9 + ["yes"])), "yes")

    def test_long_ids_keep_every_digit(self):
        import pandas as pd

        from dclab_core.analyse import _key_text

        keys = _key_text(pd.Series(["123456789012345678", "123456789012345679", "unknown", "42.0"]))
        self.assertEqual(keys.tolist(), ["123456789012345678", "123456789012345679", "unknown", "42"])

    def test_a_discrete_scale_is_not_a_cap(self):
        import numpy as np
        import pandas as pd

        from dclab_core.analyse import analyse

        df = pd.DataFrame({"x": np.arange(4000) % 7, "y": np.tile(np.arange(40), 100).astype(float)})
        self.assertNotIn("F-cap", by_id(analyse({"t": df}, target="y", task="regression", train_index=df.index[:3000])))

    def test_multiclass_missingness_and_late_positives(self):
        import numpy as np
        import pandas as pd

        from dclab_core.analyse import analyse

        rng = np.random.default_rng(0)
        n = 3000
        y = rng.choice(["a", "b", "c"], n)
        df = pd.DataFrame({"y": y, "note": np.where((y == "c") & (rng.random(n) < 0.8), None, "x")})
        self.assertEqual(by_id(analyse({"t": df}, target="y", train_index=df.index[:2400]))["F-missing-note"].severity, "warning")
        late = pd.DataFrame({"when": pd.date_range("2025-01-01", periods=n, freq="h"),
                             "y": np.r_[np.zeros(2000), (rng.random(1000) < 0.1).astype(int)]})
        self.assertEqual(by_id(analyse({"t": late}, target="y", train_index=late.index))["F-time"].severity, "warning")

    def test_conflicting_labels_are_counted_on_training_rows_only(self):
        import pandas as pd

        from dclab_core.analyse import analyse

        df = pd.DataFrame({"x": [1, 2, 3, 4] * 50, "y": [0, 1, 0, 1] * 50})
        df.loc[150:, "y"] = 1 - df.loc[150:, "y"]  # the conflicts exist only after row 150
        found = by_id(analyse({"t": df}, target="y", train_index=df.index[:150]))
        self.assertNotIn("F-conflicting-labels", found)


if __name__ == "__main__":
    unittest.main()
