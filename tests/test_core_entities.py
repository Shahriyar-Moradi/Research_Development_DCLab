"""Package 16.8: several tables. Features per (entity, cutoff) read only events before the cutoff (changing every event
after a row's cutoff changes none of its features), the target comes from the horizon after it and is unknown when the
horizon passes the data's end, entity columns written after the cutoff are kept out, a join that repeats rows is
refused with the numbers, training targets end by the holdout's cutoff, and a label table joins by key."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import entities  # noqa: E402
from dclab_core.replicas import shop_customers  # noqa: E402


def tables_of(r):
    return {"orders": {"frame": r.orders, "time": "ordered_at", "sums": ["amount"]}, "tickets": {"frame": r.tickets, "time": "opened_at"}}


class WindowTest(unittest.TestCase):
    def test_window_features_and_target_by_hand(self):
        rows = pd.DataFrame({"k": ["a", "a"], "cutoff": pd.to_datetime(["2025-01-10", "2025-01-20"])})
        ev = pd.DataFrame({"k": ["a"] * 4, "t": ["2025-01-01", "2025-01-09", "2025-01-10", "2025-01-15"], "x": [1.0, 2.0, 4.0, 8.0]})
        f = entities.window_features(rows, ev, key="k", time="t", prefix="e", sums=["x"], windows=(7,))
        # at the 10th: the 9th only (the 1st is 9 days back, the 10th is the cutoff itself, not known yet);
        # at the 20th: the 15th only (the 10th is 10 days back)
        self.assertEqual(list(f["e_count_7d"]), [1, 1])
        self.assertEqual(list(f["e_x_sum_7d"]), [2.0, 8.0])
        self.assertEqual(list(f["e_count_all"]), [2, 4])
        self.assertEqual(list(f["e_days_since_last"]), [1.0, 5.0])
        self.assertTrue((pd.to_datetime(f["_e_last_read"]) < rows["cutoff"]).all())

    def test_an_entity_starts_only_after_its_first_event(self):
        ev = pd.DataFrame({"k": ["a", "b"], "t": ["2025-01-05", "2025-01-10"]})  # b's first event is the cutoff itself
        rows = entities.snapshot_rows(ev, key="k", time="t", cutoffs=["2025-01-10", "2025-01-10", "2025-01-20"])
        self.assertEqual(sorted(map(tuple, rows.assign(cutoff=rows["cutoff"].astype(str)).to_numpy().tolist())),
                         [("a", "2025-01-10"), ("a", "2025-01-20"), ("b", "2025-01-20")])  # each cutoff once

    def test_the_trend_by_hand(self):
        rows = pd.DataFrame({"k": ["a"], "cutoff": pd.to_datetime(["2025-03-01"])})
        days = ["2025-02-25", "2025-02-20", "2025-01-20", "2025-01-15", "2025-01-10"]  # 2 in the last 30 days, 3 in the 30 before
        f = entities.window_features(rows, pd.DataFrame({"k": "a", "t": days}), key="k", time="t", prefix="e", windows=(7,))
        self.assertEqual(list(f["e_trend_30d"]), [2 - 3])

    def test_the_target_window_starts_at_the_cutoff_and_ends_before_the_horizon(self):
        rows = pd.DataFrame({"k": ["a", "b", "c"], "cutoff": pd.to_datetime(["2025-03-01"] * 2 + ["2025-03-20"])})
        ev = pd.DataFrame({"k": ["a", "b"], "t": ["2025-03-01", "2025-03-31"]})  # a: on the cutoff day; b: exactly at cutoff + 30 days
        y = entities.window_target(rows, ev, key="k", time="t", horizon_days=30, data_end="2025-04-05")
        self.assertEqual(y["target"].iloc[0], 0.0)  # an order on the cutoff day counts: the features stopped just before it
        self.assertEqual(y["target"].iloc[1], 1.0)  # [cutoff, cutoff + 30 days): the 31st is the next window's
        self.assertTrue(np.isnan(y["target"].iloc[2]))  # its horizon passes the data's end: unknown, never negative
        self.assertEqual(pd.Timestamp(y["target_last_read"].iloc[0]), pd.Timestamp("2025-03-01"))

    def test_a_join_that_repeats_rows_is_refused_with_the_numbers(self):
        left = pd.DataFrame({"k": ["a", "b", "c"]})
        with self.assertRaises(entities.DuplicatingJoin) as refused:
            entities.check_join(left, pd.DataFrame({"k": ["a", "a", "b"]}), "k", name="addresses")
        self.assertIn("3 rows into 4", str(refused.exception))
        self.assertEqual(entities.check_join(left, pd.DataFrame({"k": ["a", "b"]}), "k")["rows_after"], 3)
        with self.assertRaises(entities.DuplicatingJoin):  # missing keys match missing keys in a join
            entities.check_join(pd.DataFrame({"k": ["a", None, "b"]}), pd.DataFrame({"k": ["a", None, None]}), "k")

    def test_a_label_table_joins_only_before_its_label(self):
        rows = pd.DataFrame({"k": ["a", "a", "b"], "cutoff": pd.to_datetime(["2025-01-01", "2025-06-01", "2025-01-01"])})
        labels = pd.DataFrame({"k": ["a", "b"], "y": [1, 0], "decided": ["2025-03-01", "2025-03-01"]})
        joined, numbers = entities.label_join(rows, labels, key="k", label="y", label_time="decided")
        self.assertEqual(len(joined), 2)  # a's June row would read behaviour after its outcome
        self.assertEqual(numbers["after_the_label"], 1)
        with self.assertRaises(ValueError):
            entities.label_join(rows, labels.drop(columns=["decided"]), key="k", label="y")


class ReplicaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = shop_customers.generate(0, n=600)
        cls.cutoffs = cls.r.facts["cutoffs"]
        cls.built = entities.build(cls.r.customers, tables_of(cls.r), key="customer_id", cutoffs=cls.cutoffs, target_table="orders", horizon_days=30)

    def test_no_event_after_a_rows_cutoff_is_read(self):
        b = self.built
        self.assertEqual(b["events_after_cutoff_read"], 0)
        first = pd.Timestamp(self.cutoffs[0])
        later = self.r.orders.copy()
        after = pd.to_datetime(later["ordered_at"]) >= first
        later.loc[after, "amount"] = later.loc[after, "amount"] * 100 + 5  # a different future for every row of the first cutoff
        extra = later[after].sample(frac=0.5, random_state=0)
        changed = shop_customers.Replica(self.r.customers, pd.concat([later, extra]), self.r.tickets, self.r.addresses, self.r.facts)
        again = entities.build(changed.customers, tables_of(changed), key="customer_id", cutoffs=self.cutoffs, target_table="orders", horizon_days=30)
        features = [c for c in b["table"].columns if c.startswith(("orders_", "tickets_"))]
        one, two = (x["table"][x["table"]["cutoff"] == first].set_index("customer_id")[features].sort_index() for x in (b, again))
        pd.testing.assert_frame_equal(one, two)

    def test_export_numbers_are_kept_out_with_gaps_and_as_epochs(self):
        customers = self.r.customers.copy()
        customers["lifetime_orders"] = customers["lifetime_orders"].replace(0, np.nan)  # a left-join export leaves gaps
        customers["last_order_epoch"] = (pd.to_datetime(customers["last_order_date"]) - pd.Timestamp("1970-01-01")).dt.total_seconds()
        customers["days_since_last_at_export"] = (pd.Timestamp("2026-01-01") - pd.to_datetime(customers["last_order_date"])).dt.days
        customers["household_size"] = np.random.default_rng(0).integers(1, 6, len(customers))  # known at signup
        rows = entities.snapshot_rows(self.r.orders, key="customer_id", time="ordered_at", cutoffs=self.cutoffs)
        out = entities.entity_check(rows, customers, tables_of(self.r), key="customer_id")
        self.assertTrue({"lifetime_orders", "last_order_epoch", "days_since_last_at_export", "last_order_date"} <= set(out["after_the_cutoff"]))
        self.assertNotIn("household_size", out["after_the_cutoff"])

    def test_entity_columns_written_after_the_cutoff_are_kept_out(self):
        out = self.built["entity_columns_kept_out"]
        self.assertEqual(set(out["after_the_cutoff"]), {"last_order_date", "lifetime_orders"})
        self.assertNotIn("signup_date", out["after_the_cutoff"])  # a later signup is never after its own rows' cutoffs
        self.assertNotIn("lifetime_orders", self.built["table"].columns)

    def test_rows_past_the_data_end_are_left_out_and_training_ends_before_the_holdout(self):
        late = self.r.facts["late_cutoff"]
        b = entities.build(self.r.customers, tables_of(self.r), key="customer_id", cutoffs=[*self.cutoffs, late], target_table="orders", horizon_days=30)
        self.assertGreater(b["rows_left_out_unknown"], 0)
        self.assertNotIn(pd.Timestamp(late), set(b["table"]["cutoff"]))
        e = entities.evaluate(b, key="customer_id")
        self.assertLess(pd.Timestamp(e["train_targets_read_until"]), pd.Timestamp(e["holdout_cutoff"]))
        self.assertEqual(e["holdout_cutoff"], self.cutoffs[-1])
        mid = self.cutoffs[-2]  # 15 days before the last: its 30-day horizon runs into the holdout's window
        self.assertNotIn(mid, e["train_cutoffs"])
        self.assertIn(self.cutoffs[-3], e["train_cutoffs"])
        self.assertGreater(e["model"]["roc_auc"], e["recency_only"]["roc_auc"])


class KindTest(unittest.TestCase):
    def setUp(self):
        try:
            from dclab_rnd import kinds
        except ImportError as error:
            self.skipTest(str(error))
        self.kinds = kinds

    def test_the_acceptance_holds_without_reading_the_answer(self):
        kind = self.kinds.get("entity_tables")
        case = kind.case(0)
        run = kind.solve(type(case)(**{**case.__dict__, "facts": {}}))
        decisions = kind.accept({**run, "facts": case.facts})
        self.assertTrue(all(c["ok"] for c in decisions), decisions)
        no_refusal = {**run, "refused_joins": {}, "facts": case.facts}
        self.assertFalse(kind.accept(no_refusal)[2]["ok"])

    def test_detected_only_when_several_tables_or_a_cutoff_are_said(self):
        self.assertEqual(self.kinds.detect("Customers, orders and tickets in separate tables: who stops ordering as of each cutoff?")["key"], "entity_tables")
        for other in ("Forecast daily sales from the orders table for each store.", "Recommend products per customer from their orders",
                      "Predict whether an order is delivered late, from the orders table."):
            self.assertNotEqual(self.kinds.detect(other)["key"], "entity_tables", other)

    def test_the_engine_runs_it_once_its_event_tables_are_named(self):  # 16.15 C: refused before, a machine of its own now
        from dclab_rnd.studio.solution import Solution

        fields = {"key_column": "k", "target_table": "orders", "horizon_days": 30, "cutoffs": ["2025-01-01", "2025-02-01"]}
        with self.assertRaises(ValueError):
            Solution(task="binary", prediction_moment="On each cutoff", kind="entity_tables", kind_fields=fields)
        Solution(task="binary", prediction_moment="On each cutoff", kind="entity_tables",
                 kind_fields={**fields, "event_tables": [{"name": "orders", "time_column": "at"}]})
        self.assertTrue(self.kinds.get("entity_tables").runs_in_engine)

if __name__ == "__main__":
    unittest.main()
