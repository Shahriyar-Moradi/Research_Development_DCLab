"""The chart for an answer (package H7, part b): chosen by code from the result's shape, drawn from its numbers only."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import query as Q  # noqa: E402
from dclab_rnd import home_chart  # noqa: E402


def table(n=600, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"order_date": pd.date_range("2025-01-01", periods=n, freq="15h"),
                         "region": rng.choice(["Dubai", "Sharjah", "Ajman"], n), "channel": rng.choice(["web", "app"], n),
                         "amount": rng.gamma(2.0, 40.0, n).round(2)})


def chart(**kw):
    plan = Q.QueryPlan(measures=[{"name": "revenue", "op": "sum", "column": "amount"}], **kw)
    result = Q.execute(plan, table())
    return home_chart.spec(plan, result), result


class ChartTests(unittest.TestCase):
    def test_one_number_is_a_tile(self):
        s, r = chart()
        self.assertEqual((s["kind"], s["value"]), ("kpi", r.rows[0][0]))

    def test_a_group_is_bars_largest_first(self):
        s, r = chart(group_by=["region"])
        self.assertEqual(s["kind"], "bar")
        self.assertEqual(s["values"], sorted(s["values"], reverse=True))
        self.assertEqual(set(s["x"]), {row[0] for row in r.rows})

    def test_a_period_is_a_line_per_group(self):
        s, r = chart(group_by=["region"], period={"column": "order_date", "grain": "month"})
        self.assertEqual(s["kind"], "line")
        self.assertEqual({x["name"] for x in s["series"]}, {"Dubai", "Sharjah", "Ajman"})
        drawn = sorted(v for x in s["series"] for v in x["values"] if v is not None)
        self.assertEqual(drawn, sorted(row[2] for row in r.rows))  # nothing drawn that is not in the result

    def test_two_groups_are_a_table(self):
        s, _ = chart(group_by=["region", "channel"])
        self.assertEqual(s["kind"], "table")

    def test_the_title_is_the_plan_never_a_measure_name_a_model_wrote(self):
        plan = Q.QueryPlan(group_by=["region"], measures=[{"name": "Revenue over 1000", "op": "mean", "column": "amount"}])
        s = home_chart.spec(plan, Q.execute(plan, table()))
        self.assertEqual(s["title"], "Mean of amount by region")
        self.assertNotIn("1000", str(s))
        self.assertEqual(len(s["texts"]), len(s["values"]))  # each number as the answer writes it

    def test_one_period_point_or_many_lines_are_not_a_line(self):
        plan = Q.QueryPlan(measures=[{"name": "n", "op": "count"}], period={"column": "order_date", "grain": "year"})
        self.assertNotEqual(home_chart.spec(plan, Q.execute(plan, table(n=200)))["kind"], "line")  # 2025 only: one point
        t = table()
        t["region"] = [f"r{i % 9}" for i in range(len(t))]  # nine groups: too many lines
        plan = Q.QueryPlan(measures=[{"name": "n", "op": "count"}], group_by=["region"], period={"column": "order_date", "grain": "quarter"})
        self.assertEqual(home_chart.spec(plan, Q.execute(plan, t))["kind"], "table")

    def test_a_withheld_value_is_said_never_drawn_as_zero(self):
        t = table(n=40)
        plan = Q.QueryPlan(group_by=["region"], measures=[{"name": "top", "op": "max", "column": "amount"}])
        s = home_chart.spec(plan, Q.execute(plan, t, floor=5))
        withheld = [x for x, v in zip(s["texts"], s["values"]) if v is None]
        self.assertTrue(all(x == "withheld (too few rows)" for x in withheld))

    def test_an_empty_group_and_a_whole_number_key_are_written_as_the_answer_writes_them(self):
        t = table()
        t["store"] = [None if i % 4 == 0 else float(10001 + i % 3) for i in range(len(t))]
        plan = Q.QueryPlan(group_by=["store"], measures=[{"name": "total", "op": "sum", "column": "amount"}])
        s = home_chart.spec(plan, Q.execute(plan, t))
        self.assertIn("(no value)", s["x"])
        self.assertIn("10001", s["x"])
        keys = [r[0] for r in s["table"]["rows"]]
        self.assertIn("(no value)", keys)
        self.assertNotIn("withheld (too few rows)", keys)  # a key is never taken for a withheld number

    def test_a_period_the_floor_removed_is_a_gap_never_joined(self):
        t = table(n=600)
        t = t[~((t.order_date >= "2025-02-01") & (t.order_date < "2025-03-01"))].copy()
        t = pd.concat([t, t.iloc[:2].assign(order_date=pd.Timestamp("2025-02-10"))])  # two February rows: under the floor
        plan = Q.QueryPlan(measures=[{"name": "n", "op": "count"}], period={"column": "order_date", "grain": "month"})
        s = home_chart.spec(plan, Q.execute(plan, t))
        self.assertEqual(s["x"][:3], ["2025-01", "2025-02", "2025-03"])
        self.assertIsNone(s["series"][0]["values"][1])


if __name__ == "__main__":
    unittest.main()
