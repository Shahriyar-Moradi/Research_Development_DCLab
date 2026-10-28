"""Preparing a table step by step (dclab_core.recipe, package H9 of the Home plan): each H1 dev preparation request, by a
hand-written recipe on the replica tables, checked against pandas computed here; the guard; the original untouched."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import recipe as R  # noqa: E402


def S(**kw):
    return R.Step(**kw)


class H1PreparationRequests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agent_eval.home import tables
        cls.sales, cls.customers = tables.build("sales"), tables.build("customers")

    def test_d_r1_remove_duplicate_orders(self):
        t = self.sales.copy()
        before = t.copy()
        out = R.apply(R.Recipe(steps=[S(kind="dedupe", keys=["order_id"])]), t)
        self.assertEqual(len(out), t["order_id"].nunique())
        self.assertFalse(out["order_id"].duplicated().any())
        pd.testing.assert_frame_equal(t, before)  # the original is never changed

    def test_d_r2_join_the_customers_city_and_segment_onto_each_order(self):
        step = S(kind="join", table="customers", keys=["customer_id"], columns=["city", "segment"])
        p, out = R.preview(step, self.sales, {"customers": self.customers})
        self.assertEqual(len(out), len(self.sales))  # a left join on a unique key keeps every order
        self.assertFalse(p.needs_confirmation)
        self.assertEqual(p.added, ["city", "segment"])
        want = self.sales.merge(self.customers[["customer_id", "city", "segment"]], on="customer_id", how="left")
        pd.testing.assert_frame_equal(out.reset_index(drop=True), want.reset_index(drop=True))

    def test_d_r3_dates_read_and_rows_without_an_amount_dropped(self):
        recipe = R.Recipe(steps=[S(kind="to_date", columns=["order_date"]), S(kind="drop_missing", columns=["amount"])])
        out = R.apply(recipe, self.sales)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["order_date"]))
        self.assertEqual(len(out), self.sales["amount"].notna().sum())

    def test_d_r4_pivot_region_by_category_summing_amount(self):
        out = R.run(S(kind="pivot", index="region", spread="product_category", values="amount", agg="sum"), self.sales)
        want = self.sales.pivot_table(index="region", columns="product_category", values="amount", aggfunc="sum")
        self.assertEqual(len(out), self.sales["region"].nunique())
        for region, row in want.iterrows():
            got = out[out["region"] == region].iloc[0]
            for cat, value in row.items():
                self.assertAlmostEqual(got[str(cat)], value, places=6)

    def test_d_r5_keep_only_paid_orders_from_2026_asks_first(self):  # H1's trap 3
        step = S(kind="filter", filters=[{"column": "status", "op": "eq", "value": "paid"},
                                         {"column": "order_date", "op": "ge", "value": "2026-01-01"}])
        p, out = R.preview(step, self.sales)
        self.assertTrue(p.needs_confirmation)
        self.assertIn("removes", p.reason)
        self.assertTrue((out["status"] == "paid").all())
        self.assertTrue((pd.to_datetime(out["order_date"]) >= "2026-01-01").all())

    def test_d_r6_months_since_signup_as_a_new_column(self):
        step = S(kind="months_since", columns=["signup_date"], to="months_since_signup", at="2026-06-30")
        p, out = R.preview(step, self.customers)
        self.assertEqual(p.added, ["months_since_signup"])
        d = pd.to_datetime(self.customers["signup_date"])
        want = (2026 - d.dt.year) * 12 + (6 - d.dt.month) - (30 < d.dt.day).astype(int)
        self.assertEqual(out["months_since_signup"].astype(int).tolist(), want.tolist())


class GuardTests(unittest.TestCase):
    def test_a_join_that_changes_the_row_count_asks(self):
        a = pd.DataFrame({"k": [1, 2], "x": [1, 2]})
        b = pd.DataFrame({"k": [1, 1, 2], "y": [5, 6, 7]})
        p, _ = R.preview(S(kind="join", table="b", keys=["k"]), a, {"b": b})
        self.assertTrue(p.needs_confirmation)
        self.assertIn("row count", p.reason)

    def test_a_small_removal_does_not_ask_and_bad_steps_are_refused(self):
        t = pd.DataFrame({"a": range(100), "b": [None] + list(range(99))})
        p, _ = R.preview(S(kind="drop_missing", columns=["b"]), t)
        self.assertFalse(p.needs_confirmation)  # 1% of the rows
        for bad in (S(kind="drop_columns", columns=["zzz"]), S(kind="join", table="nope", keys=["a"])):
            with self.assertRaises(ValueError):
                R.run(bad, t, {})
        with self.assertRaises(ValueError):
            R.Step(kind="rename", columns=["a"])  # no new name

    def test_undo_and_the_contract_reads_back(self):
        r = R.Recipe(steps=[S(kind="sort", columns=["a"]), S(kind="dedupe")])
        self.assertEqual(len(R.undo(r).steps), 1)
        self.assertEqual(R.Recipe.read(r.model_dump_json()), r)
        self.assertEqual(len(R.recipe_hash(r)), 16)


if __name__ == "__main__":
    unittest.main()
