"""Questions answered from a table, never by listing its rows (dclab_core.query, package H6 of the Home plan)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import query as Q  # noqa: E402


def sales(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "order_id": [f"ORD-{10000 + i}" for i in range(n)],
        "order_date": pd.date_range("2026-01-01", periods=n, freq="6h"),
        "region": rng.choice(["Dubai", "Abu Dhabi", "Sharjah"], n),
        "status": rng.choice(["paid", "refunded"], n, p=[0.8, 0.2]),
        "amount": rng.gamma(2.0, 50.0, n).round(2),
        "note": ["customer asked for a faster delivery next time please"] * n,
    })


def plan(**kw):
    return Q.QueryPlan(**{"measures": [{"name": "revenue", "op": "sum", "column": "amount"}], **kw})


class ValidateTests(unittest.TestCase):
    def test_every_refusal(self):
        t = sales()
        cases = {
            "a plan must aggregate": Q.QueryPlan(group_by=["region"]),
            "no column 'margin'": plan(measures=[{"name": "m", "op": "sum", "column": "margin"}]),
            "identifier-like": plan(group_by=["order_id"]),
            "free text": plan(group_by=["note"]),
            "needs a numeric column": plan(measures=[{"name": "m", "op": "mean", "column": "region"}]),
            "at most 3 groups": plan(group_by=["region", "status", "order_date", "amount"]),
            "a limit of at most 500": plan(limit=900),
            "needs the condition": plan(measures=[{"name": "r", "op": "rate"}]),
        }
        for words, p in cases.items():
            self.assertTrue(any(words in w for w in Q.validate(p, t)), words)
        self.assertEqual(Q.validate(plan(group_by=["region"]), t), [])

    def test_no_plan_lists_rows(self):
        t = sales()
        with self.assertRaises(ValueError):
            Q.execute(Q.QueryPlan(group_by=["order_id"], measures=[{"name": "n", "op": "count"}]), t)  # one group per row: refused
        r = Q.execute(plan(group_by=["region"]), t)
        self.assertLessEqual(len(r.rows), 3)
        self.assertNotIn("ORD-10007", str(r.rows))


class ExecuteTests(unittest.TestCase):
    def test_revenue_per_region_last_quarter_only_paid(self):  # the H1 request d-a1, by a hand-written plan
        t = sales()
        p = plan(group_by=["region"], filters=[{"column": "status", "op": "eq", "value": "paid"}],
                 period={"column": "order_date", "grain": "quarter", "start": "2026-01-01", "end": "2026-03-31"},
                 sort=[{"by": "revenue", "desc": True}])
        r = Q.execute(p, t)
        paid = t[(t.status == "paid") & (t.order_date < "2026-04-01")]  # the whole of the end day (the review of H6)
        want = paid.groupby("region")["amount"].sum()
        got = {row[0]: row[2] for row in r.rows}
        for region, value in want.items():
            self.assertAlmostEqual(got[region], value, places=6)
        self.assertEqual([row[2] for row in r.rows], sorted((row[2] for row in r.rows), reverse=True))
        self.assertEqual(r.columns, ["region", "order_date", "revenue"])

    def test_the_floor_removes_a_small_group_and_counts_it(self):
        t = sales()
        t.loc[:2, "region"] = "Fujairah"  # a 3-row group
        r = Q.execute(plan(group_by=["region"]), t)
        self.assertNotIn("Fujairah", [row[0] for row in r.rows])
        self.assertEqual(r.suppressed_groups, 1)
        r = Q.execute(plan(group_by=["region"]), t, floor=2)
        self.assertIn("Fujairah", [row[0] for row in r.rows])

    def test_counts_rates_and_percentiles(self):
        t = sales()
        r = Q.execute(Q.QueryPlan(measures=[{"name": "orders", "op": "count"},
                                            {"name": "refund_rate", "op": "rate", "where": {"column": "status", "op": "eq", "value": "refunded"}},
                                            {"name": "p90", "op": "percentile", "column": "amount", "q": 90}]), t)
        orders, rate, p90 = r.rows[0]
        self.assertEqual(orders, len(t))
        self.assertAlmostEqual(rate, (t.status == "refunded").mean())
        self.assertAlmostEqual(p90, float(np.percentile(t.amount, 90)))

    def test_the_exposure_log_records_the_related_pair(self):
        e = Q.exposure(plan(group_by=["region"], filters=[{"column": "status", "op": "eq", "value": "paid"}]))
        self.assertIn(["region", "amount"], e["columns_related"])
        self.assertIn(["status", "amount"], e["columns_related"])
        self.assertEqual(e["rows"], "filter")

    def test_the_contract_reads_back(self):
        p = plan(group_by=["region"])
        self.assertEqual(Q.QueryPlan.read(p.model_dump_json()), p)
        r = Q.execute(p, sales())
        self.assertEqual(Q.QueryResult.read(r.model_dump_json()).plan_hash, Q.plan_hash(p))



class H1RequestsOnTheReplicaTables(unittest.TestCase):
    """The plan's done-when: the H1 analysis requests' answers, by hand-written plans on the replica tables, each checked
    against pandas computed here."""

    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agent_eval.home import tables
        cls.sales, cls.customers = tables.build("sales"), tables.build("customers")

    def test_d_a1_revenue_per_region_last_quarter_only_paid(self):
        t = self.sales
        p = plan(group_by=["region"], filters=[{"column": "status", "op": "eq", "value": "paid"}],
                 period={"column": "order_date", "grain": "quarter", "start": "2026-01-01", "end": "2026-03-31"})
        got = {(r[0], r[1]): r[2] for r in Q.execute(p, t).rows}
        d = pd.to_datetime(t.order_date)
        want = t[(t.status == "paid") & (d >= "2026-01-01") & (d < "2026-04-01")].groupby("region")["amount"].sum()
        self.assertEqual(set(got), {(region, "2026Q1") for region in want.index})
        for region, value in want.items():
            self.assertAlmostEqual(got[(region, "2026Q1")], value, places=6)

    def test_d_a2_average_order_value_web_and_app_over_2025(self):
        t = self.sales
        p = Q.QueryPlan(group_by=["channel"], filters=[{"column": "channel", "op": "in", "value": ["web", "app"]}],
                        period={"column": "order_date", "grain": "year", "start": "2025-01-01", "end": "2025-12-31"},
                        measures=[{"name": "aov", "op": "mean", "column": "amount"}])
        got = {r[0]: r[2] for r in Q.execute(p, t).rows}
        d = pd.to_datetime(t.order_date)
        want = t[t.channel.isin(["web", "app"]) & (d.dt.year == 2025)].groupby("channel")["amount"].mean()
        for channel, value in want.items():
            self.assertAlmostEqual(got[channel], value, places=6)

    def test_d_a3_orders_per_month_of_2025(self):
        t = self.sales
        p = Q.QueryPlan(period={"column": "order_date", "grain": "month", "start": "2025-01-01", "end": "2025-12-31"},
                        measures=[{"name": "orders", "op": "count"}])
        got = {r[0]: r[1] for r in Q.execute(p, t).rows}
        d = pd.to_datetime(t.order_date)
        want = d[d.dt.year == 2025].dt.to_period("M").astype(str).value_counts()
        self.assertEqual(got, {k: int(v) for k, v in want.items() if v >= Q.FLOOR})

    def test_d_a4_average_monthly_spend_by_plan(self):
        t = self.customers
        p = Q.QueryPlan(group_by=["plan"], measures=[{"name": "avg_spend", "op": "mean", "column": "monthly_spend"}])
        got = {r[0]: r[1] for r in Q.execute(p, t).rows}
        for name, value in t.groupby("plan")["monthly_spend"].mean().items():
            self.assertAlmostEqual(got[name], value, places=6)

    def test_d_a5_share_with_more_than_three_tickets_per_city(self):
        t = self.customers
        p = Q.QueryPlan(group_by=["city"], measures=[{"name": "share", "op": "rate",
                                                      "where": {"column": "support_tickets_90d", "op": "gt", "value": 3}}])
        got = {r[0]: r[1] for r in Q.execute(p, t).rows}
        for city, value in (t.support_tickets_90d > 3).groupby(t.city).mean().items():
            self.assertAlmostEqual(got[city], value, places=9)

    def test_d_a6_a_column_that_is_not_there_is_refused(self):
        p = Q.QueryPlan(group_by=["segment"], measures=[{"name": "avg", "op": "mean", "column": "lifetime_spend"}])
        self.assertTrue(any("no column 'lifetime_spend'" in w for w in Q.validate(p, self.customers)))


class ReviewOfH6Tests(unittest.TestCase):
    """The review's findings (H6), each with the plan that showed it."""

    def test_a_persons_id_is_never_a_group_a_filter_or_a_measured_value(self):
        from dclab_rnd.agent_eval.home import tables
        t = tables.build("sales")  # 1,200 customer ids over 2,020 orders
        refused = [plan(group_by=["customer_id"]),
                   plan(filters=[{"column": "customer_id", "op": "ne", "value": "C-00104"}]),  # a total minus one person
                   Q.QueryPlan(measures=[{"name": "r", "op": "rate", "where": {"column": "order_id", "op": "eq", "value": "ORD-10007"}}])]
        for p in refused:
            self.assertTrue(Q.validate(p, t), p)
        self.assertEqual(Q.validate(Q.QueryPlan(measures=[{"name": "people", "op": "count_distinct", "column": "customer_id"}]), t), [])

    def test_the_extremes_of_a_small_group_are_withheld(self):
        t = sales(n=21)  # about 7 rows per region: above the floor, under twice it
        r = Q.execute(Q.QueryPlan(group_by=["region"], measures=[{"name": "top", "op": "max", "column": "amount"},
                                                                  {"name": "orders", "op": "count"}]), t, floor=5)
        small = [row for row in r.rows if row[2] < 10]
        self.assertTrue(small)
        self.assertTrue(all(row[1] is None for row in small))  # one person's own value is never shown

    def test_malformed_plans_are_refused_never_crash(self):
        t = sales()
        for p in (plan(period={"column": "region", "grain": "month"}),
                  plan(filters=[{"column": "amount", "op": "gt", "value": "abc"}]),
                  plan(filters=[{"column": "amount", "op": "between", "value": 5}]),
                  plan(period={"column": "order_date", "grain": "month", "start": "last quarter"})):
            self.assertTrue(Q.validate(p, t), p)
            with self.assertRaises(ValueError):
                Q.execute(p, t)

    def test_a_timezone_aware_column_and_none_sorted_last(self):
        t = sales()
        t["order_date"] = t["order_date"].dt.tz_localize("Asia/Dubai")
        r = Q.execute(plan(period={"column": "order_date", "grain": "month", "start": "2026-01-01", "end": "2026-01-31"}), t)
        self.assertEqual([row[0] for row in r.rows], ["2026-01"])


if __name__ == "__main__":
    unittest.main()


class ImportOrderTests(unittest.TestCase):
    """The review of 2026-10-28: `from dclab_core import query` first raised ImportError (contracts imported query while
    it was loading, then recipe, which read query's Filter). Every module a host may import first, in a fresh interpreter."""

    def test_every_table_module_imports_first(self):
        import subprocess
        for name in ("query", "recipe", "change", "dashboard", "contracts"):
            done = subprocess.run([sys.executable, "-c", f"from dclab_core import {name}; from dclab_core.contracts import CONTRACTS; "
                                   "assert {'query_plan', 'recipe', 'change_result', 'dashboard'} <= set(CONTRACTS)"],
                                  cwd=str(ROOT), capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 0, f"{name}: {done.stderr[-400:]}")

