"""The query engine (package H6): plans, every refusal, the row floor, the checked table and the exposure log.

The H1 analysis requests' answers are reproduced by hand-written plans on the seeded tables (dev requests only).
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dclab_rnd.agent_eval.home import tables  # noqa: E402
from dclab_rnd.query import PlanError, execute, plan_hash, run, validate  # noqa: E402
from dclab_rnd.query.execute import TableError, floor  # noqa: E402
from dclab_rnd.query.plan import related  # noqa: E402


def sales():
    f = tables.build("sales")
    f["order_date"] = pd.to_datetime(f["order_date"])
    return f


def customers():
    f = tables.build("customers")
    f["signup_date"] = pd.to_datetime(f["signup_date"])
    return f


def answer(plan, frame):
    return execute(validate(plan, frame), frame)


class PlanTests(unittest.TestCase):
    def test_revenue_per_region_last_quarter_paid_only(self):  # d-a1
        f = sales()
        plan = {"filters": [{"column": "status", "op": "eq", "value": "paid"}], "group_by": ["region"],
                "period": {"column": "order_date", "from": "2026-01-01", "to": "2026-03-31"},
                "measures": [{"name": "revenue", "op": "sum", "column": "amount"}], "sort": [{"by": "revenue", "desc": True}]}
        out = answer(plan, f)
        paid = f[(f.status == "paid") & (f.order_date >= "2026-01-01") & (f.order_date <= "2026-03-31")]
        expected = paid.groupby("region")["amount"].sum().sort_values(ascending=False)
        self.assertEqual([r[0] for r in out["rows"]], list(expected.index))
        for row, value in zip(out["rows"], expected.values):
            self.assertAlmostEqual(row[1], float(value), places=4)
        self.assertEqual(out["columns"], ["region", "revenue"])

    def test_average_order_value_by_channel_over_a_year(self):  # d-a2
        f = sales()
        out = answer({"filters": [{"column": "channel", "op": "in", "value": ["web", "app"]}], "group_by": ["channel"],
                      "period": {"column": "order_date", "from": "2025-01-01", "to": "2025-12-31"},
                      "measures": [{"name": "average_order_value", "op": "mean", "column": "amount"}]}, f)
        part = f[f.channel.isin(["web", "app"]) & (f.order_date.dt.year == 2025)]
        self.assertEqual({r[0]: round(r[1], 4) for r in out["rows"]}, {k: round(float(v), 4) for k, v in part.groupby("channel")["amount"].mean().items()})

    def test_orders_per_month_and_a_share(self):  # d-a3, d-a5
        f, c = sales(), customers()
        months = answer({"period": {"column": "order_date", "grain": "month", "from": "2025-01-01", "to": "2025-12-31"},
                         "measures": [{"name": "orders", "op": "count"}]}, f)
        self.assertEqual(months["columns"], ["period", "orders"])
        self.assertEqual(sum(r[1] for r in months["rows"]), int((f.order_date.dt.year == 2025).sum()))
        share = answer({"group_by": ["city"], "measures": [{"name": "share", "op": "rate", "where": {"column": "support_tickets_90d", "op": "gt", "value": 3}}]}, c)
        expected = (c.support_tickets_90d > 3).groupby(c.city).mean()
        for city, value in share["rows"]:
            self.assertAlmostEqual(value, float(expected[city]), places=5)

    def test_average_spend_by_plan_and_a_column_that_is_not_there(self):  # d-a4, d-a6
        c = customers()
        out = answer({"group_by": ["plan"], "measures": [{"name": "avg_spend", "op": "mean", "column": "monthly_spend"}]}, c)
        expected = c.groupby("plan")["monthly_spend"].mean()
        self.assertEqual({r[0]: round(r[1], 4) for r in out["rows"]}, {k: round(float(v), 4) for k, v in expected.items()})
        with self.assertRaises(PlanError) as err:  # the trap: lifetime_spend is not a column
            validate({"group_by": ["segment"], "measures": [{"name": "avg", "op": "mean", "column": "lifetime_spend"}]}, c)
        self.assertIn("lifetime_spend", " ".join(err.exception.problems))

    def test_refunds_per_category_this_year(self):  # d-s6
        f = sales()
        out = answer({"filters": [{"column": "status", "op": "eq", "value": "refunded"}], "group_by": ["product_category"],
                      "period": {"column": "order_date", "from": "2026-01-01"},
                      "measures": [{"name": "refunds", "op": "count"}, {"name": "refunded", "op": "sum", "column": "amount"}]}, f)
        part = f[(f.status == "refunded") & (f.order_date >= "2026-01-01")]
        expected = part.groupby("product_category").size()
        self.assertEqual({r[0]: r[1] for r in out["rows"]}, {k: int(v) for k, v in expected.items() if v >= floor()})

    def test_percentiles_and_distinct_counts(self):
        f = sales()
        out = answer({"measures": [{"name": "p90", "op": "p90", "column": "amount"}, {"name": "buyers", "op": "count_distinct", "column": "customer_id"}]}, f)
        self.assertAlmostEqual(out["rows"][0][0], float(f.amount.quantile(0.9)), places=4)
        self.assertEqual(out["rows"][0][1], int(f.customer_id.nunique()))

    def test_the_same_plan_has_the_same_hash(self):
        plan = {"measures": [{"name": "orders", "op": "count"}]}
        self.assertEqual(plan_hash(validate(plan, sales())), plan_hash(validate(dict(plan), sales())))


class RefusalTests(unittest.TestCase):
    def refused(self, plan, frame=None):
        with self.assertRaises(PlanError) as caught:
            validate(plan, frame if frame is not None else sales())
        return " ".join(caught.exception.problems)

    def test_no_plan_lists_rows(self):
        self.assertIn("at least one measure", self.refused({"group_by": ["region"]}))
        self.assertIn("at least one measure", self.refused({}))
        self.assertIn("identifier", self.refused({"group_by": ["order_id"], "measures": [{"name": "n", "op": "count"}]}))
        self.assertIn("identifier", self.refused({"group_by": ["customer_id"], "measures": [{"name": "n", "op": "count"}]}))  # repeats, still an id
        self.assertIn("identifier", self.refused({"filters": [{"column": "order_id", "op": "eq", "value": "ORD-10001"}],
                                                  "measures": [{"name": "n", "op": "count"}]}))
        self.assertIn("one row's value", self.refused({"measures": [{"name": "x", "op": "max", "column": "region"}]}))

    def test_malformed_plans_are_refused_never_crash(self):
        f = sales()
        for bad in ({"filters": ["x"], "measures": [{"name": "n", "op": "count"}]}, {"period": "2025", "measures": [{"name": "n", "op": "count"}]},
                    {"measures": [{"name": "n", "op": "count"}], "sort": ["n"]}, {"measures": [{"name": ["n"], "op": "count"}]},
                    {"filters": [{"column": "amount", "op": "gt", "value": "abc"}], "measures": [{"name": "n", "op": "count"}]},
                    {"filters": [{"column": "order_date", "op": "gt", "value": "not a date"}], "measures": [{"name": "n", "op": "count"}]},
                    {"measures": [{"name": "r", "op": "rate", "where": {"column": "amount", "op": "gt", "value": "x"}}]},
                    {"group_by": ["period"], "period": {"column": "order_date", "grain": "month"}, "measures": [{"name": "n", "op": "count"}]},
                    {"period": {"column": "order_date", "grain": "month"}, "measures": [{"name": "period", "op": "count"}]}, "not a plan"):
            with self.assertRaises(PlanError, msg=str(bad)):
                validate(bad, f)

    def test_odd_tables_still_answer_as_json(self):
        from decimal import Decimal

        f = pd.DataFrame({"g": ["a"] * 6 + [None] * 6 + [Decimal("1")] * 0 + ["b"] * 6, "v": np.arange(18.0),
                          "t": pd.date_range("2025-01-01", periods=18, tz="Asia/Dubai")})
        out = answer({"group_by": ["g"], "period": {"column": "t", "from": "2025-01-01", "to": "2025-02-01"},
                      "filters": [{"column": "t", "op": "ge", "value": "2025-01-02"}], "measures": [{"name": "n", "op": "count"}],
                      "sort": [{"by": "g", "desc": True}]}, f)
        json.dumps(out)
        d = pd.DataFrame({"g": [Decimal("1.5")] * 6, "v": np.arange(6.0)})
        json.dumps(answer({"group_by": ["g"], "measures": [{"name": "n", "op": "count"}]}, d))

    def test_every_other_refusal(self):
        self.assertIn("no column named", self.refused({"measures": [{"name": "x", "op": "sum", "column": "profit"}]}))
        self.assertIn("needs a numeric column", self.refused({"measures": [{"name": "x", "op": "sum", "column": "region"}]}))
        self.assertIn("limit", self.refused({"measures": [{"name": "n", "op": "count"}], "limit": 501}))
        self.assertIn("at most 3", self.refused({"group_by": ["region", "channel", "status", "product_category"], "measures": [{"name": "n", "op": "count"}]}))
        self.assertIn("unknown op", self.refused({"measures": [{"name": "n", "op": "list"}]}))
        self.assertIn("not a date", self.refused({"period": {"column": "region", "grain": "month"}, "measures": [{"name": "n", "op": "count"}]}))
        self.assertIn("unknown keys", self.refused({"rows": True, "measures": [{"name": "n", "op": "count"}]}))
        self.assertIn("repeated", self.refused({"measures": [{"name": "n", "op": "count"}, {"name": "n", "op": "count"}]}))
        notes = pd.DataFrame({"note": [f"the customer said something long about delivery number {i} and more" for i in range(30)], "x": range(30)})
        self.assertIn("text", self.refused({"group_by": ["note"], "measures": [{"name": "n", "op": "count"}]}, notes))
        self.assertIn("numeric column with many values", self.refused({"group_by": ["amount"], "measures": [{"name": "n", "op": "count"}]}))


class PeopleTests(unittest.TestCase):
    """Columns about a person are identifiers whatever their uniqueness; a group needs as many different people."""

    def orders(self):
        rng = np.random.default_rng(0)
        people = [f"person{i}@mail.com" for i in range(40)]
        return pd.DataFrame({"email": rng.choice(people, 400), "phone": [f"+971 50 {1000000 + i % 40}" for i in range(400)],
                             "customer_name": [f"Name {i % 40}" for i in range(400)], "CustomerID": [f"C{i % 40}" for i in range(400)],
                             "contact": rng.choice(people, 400), "amount": rng.random(400) * 100, "city": rng.choice(["a", "b"], 400)})

    def test_personal_columns_are_never_groups_or_filters(self):
        f = self.orders()
        for col in ("email", "phone", "customer_name", "CustomerID", "contact"):
            with self.assertRaises(PlanError, msg=col):
                validate({"group_by": [col], "measures": [{"name": "total", "op": "sum", "column": "amount"}]}, f)
        with self.assertRaises(PlanError):
            validate({"filters": [{"column": "email", "op": "eq", "value": "person3@mail.com"}], "measures": [{"name": "n", "op": "count"}]}, f)
        self.assertEqual(answer({"group_by": ["city"], "measures": [{"name": "total", "op": "sum", "column": "amount"}]}, f)["suppressed_groups"], 0)

    def test_a_group_of_one_persons_rows_is_hidden(self):
        f = pd.DataFrame({"customer_id": [f"C{i % 200}" for i in range(1000)] + ["C999"] * 12,
                          "city": ["big"] * 1000 + ["tiny"] * 12, "amount": [10.0] * 1000 + [50.0] * 12})
        out = answer({"group_by": ["city"], "measures": [{"name": "total", "op": "sum", "column": "amount"}]}, f)
        self.assertEqual([r[0] for r in out["rows"]], ["big"])
        self.assertEqual(out["suppressed_groups"], 1)


class FloorTests(unittest.TestCase):
    def test_a_hidden_group_never_comes_with_an_exact_count(self):
        f = pd.DataFrame({"region": ["vip"] + ["std"] * 40, "salary": [987654.0] + [1000.0] * 40})
        out = answer({"filters": [{"column": "region", "op": "eq", "value": "vip"}, {"column": "salary", "op": "gt", "value": 500000}],
                      "measures": [{"name": "n", "op": "count"}]}, f)
        self.assertEqual(out["rows"], [])
        self.assertNotIn("rows_matched", out)

    def test_an_extreme_is_shown_only_for_large_groups(self):
        f = pd.DataFrame({"g": ["a"] * 8 + ["b"] * 40, "v": np.arange(48.0)})
        out = answer({"group_by": ["g"], "measures": [{"name": "top", "op": "max", "column": "v"}, {"name": "p99", "op": "p99", "column": "v"},
                                                      {"name": "mid", "op": "median", "column": "v"}]}, f)
        rows = {r[0]: r[1:] for r in out["rows"]}
        self.assertEqual(rows["a"][:2], [None, None])  # one row's value in a group of 8
        self.assertIsNotNone(rows["a"][2])
        self.assertIsNotNone(rows["b"][0])
        self.assertEqual(out["withheld_extremes"], 2)


    def test_a_small_group_is_removed_and_counted(self):
        f = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 3, "v": np.arange(13.0)})
        out = answer({"group_by": ["g"], "measures": [{"name": "total", "op": "sum", "column": "v"}]}, f)
        self.assertEqual([r[0] for r in out["rows"]], ["a"])
        self.assertEqual(out["suppressed_groups"], 1)

    def test_a_whole_answer_under_the_floor_is_withheld(self):
        f = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 3, "v": np.arange(13.0)})
        out = answer({"filters": [{"column": "g", "op": "eq", "value": "b"}], "measures": [{"name": "total", "op": "sum", "column": "v"}]}, f)
        self.assertEqual(out["rows"], [])
        self.assertEqual(out["suppressed_groups"], 1)

    def test_the_floor_can_be_raised_never_lowered_below_three(self):
        with mock.patch.dict(os.environ, {"DCLAB_ROW_FLOOR": "1"}):
            self.assertEqual(floor(), 3)
        with mock.patch.dict(os.environ, {"DCLAB_ROW_FLOOR": "20"}):
            self.assertEqual(floor(), 20)


class AdminTextTests(unittest.TestCase):
    def test_the_admin_page_states_the_floor(self):
        from dclab_rnd.agentic.pages.platform import _answers
        with mock.patch.dict(os.environ, {"DCLAB_ROW_FLOOR": "7"}):
            text = _answers()["text"]
        self.assertIn("fewer than 7 rows", text)
        self.assertIn("not anonymisation", text)


class RunTests(unittest.TestCase):
    def draft_with(self, frame):
        from dclab_rnd import audit
        from dclab_rnd.draft.store import DraftStore

        store = DraftStore(Path(tempfile.mkdtemp()))
        d = store.create("How much revenue per region?")
        path = store.data_dir(d["id"]) / "clean.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        store.update(d["id"], lambda x: x.update(assets=[{"id": "a1", "status": "ready", "clean_file": "clean.parquet", "clean_sha256": audit.sha256_of(path)}],
                                                 active_asset="a1"))
        return store, d["id"], path

    def test_run_logs_what_it_related(self):
        store, did, _ = self.draft_with(sales())
        run(store, did, {"filters": [{"column": "status", "op": "eq", "value": "paid"}], "group_by": ["region"],
                         "measures": [{"name": "revenue", "op": "sum", "column": "amount"}]})
        log = store.get(did)["exposure"]
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["columns_related"], [["amount", "region"], ["amount", "status"]])
        self.assertEqual(log[0]["rows"], "filter")
        self.assertEqual(related({"group_by": ["a"], "measures": [{"name": "r", "op": "rate", "where": {"column": "y", "op": "eq", "value": 1}}]}), [("a", "y")])

    def test_the_log_says_which_table(self):
        store, did, _ = self.draft_with(sales())
        run(store, did, {"measures": [{"name": "n", "op": "count"}]})
        entry = store.get(did)["exposure"][0]
        self.assertEqual(entry["asset"], "a1")
        self.assertTrue(entry["table_sha256"])

    def test_a_table_that_changed_is_not_read(self):
        store, did, path = self.draft_with(sales())
        sales().head(10).to_parquet(path)
        with self.assertRaises(TableError):
            run(store, did, {"measures": [{"name": "n", "op": "count"}]})

    def test_results_are_json(self):
        store, did, _ = self.draft_with(sales())
        out = run(store, did, {"period": {"column": "order_date", "grain": "quarter"}, "measures": [{"name": "n", "op": "count"}, {"name": "m", "op": "median", "column": "amount"}]})
        json.dumps(out)


if __name__ == "__main__":
    unittest.main()
