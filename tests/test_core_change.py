"""Why a number changed between two periods (dclab_core.change, package H11): the parts add up exactly, small groups are
pooled, the H1 business request d-s1 gets its top contributor, and the result says association, not cause."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import change as C  # noqa: E402

FEB = C.Window(start="2026-02-01", end="2026-02-28", label="February 2026")
MAR = C.Window(start="2026-03-01", end="2026-03-31", label="March 2026")


def orders(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for month, n_dubai, price_sharjah in ((2, 300, 100.0), (3, 180, 70.0)):  # Dubai loses volume, Sharjah its price
        for region, n, price in (("Dubai", n_dubai, 120.0), ("Sharjah", 200, price_sharjah), ("Ajman", 100, 80.0), ("Fujairah", 3, 50.0)):
            for i in range(n):
                rows.append({"order_date": f"2026-0{month}-{1 + i % 27:02d}", "region": region,
                             "channel": rng.choice(["web", "app"]), "amount": price * rng.uniform(0.9, 1.1)})
    return pd.DataFrame(rows)


class ChangeTests(unittest.TestCase):
    def test_the_parts_add_up_to_each_segment_and_the_segments_to_the_total(self):
        r = C.compare(orders(), date_column="order_date", a=FEB, b=MAR, by="region", measure="amount")
        for s in r.segments:
            self.assertAlmostEqual(s.volume + s.mix + s.rate, s.change, places=6)
        self.assertAlmostEqual(sum(s.change for s in r.segments), r.change, places=6)
        t = orders()
        d = pd.to_datetime(t.order_date)
        self.assertAlmostEqual(r.total_a, t[d.dt.month == 2].amount.sum(), places=6)
        self.assertAlmostEqual(r.total_b, t[d.dt.month == 3].amount.sum(), places=6)

    def test_small_groups_are_pooled_never_shown(self):
        r = C.compare(orders(), date_column="order_date", a=FEB, b=MAR, by="region", measure="amount")
        names = [s.segment for s in r.segments]
        self.assertNotIn("Fujairah", names)  # three rows a month
        self.assertEqual(names[-1], C.OTHER)

    def test_the_top_contributor_and_its_kind_of_change(self):
        r = C.compare(orders(), date_column="order_date", a=FEB, b=MAR, by="region", measure="amount")
        top = r.segments[0]
        self.assertEqual(top.segment, "Dubai")  # 120 fewer orders at about 120 each
        sharjah = next(s for s in r.segments if s.segment == "Sharjah")
        self.assertLess(sharjah.rate, 0)  # the same orders at a lower price: a rate effect
        self.assertIn("association, not cause", r.caveat)

    def test_best_split_and_counts_and_refusals(self):
        r = C.best_split(orders(), ["channel", "region"], date_column="order_date", a=FEB, b=MAR, measure="amount")
        self.assertEqual(r.by, "region")  # region explains the drop; channel is random
        n = C.compare(orders(), date_column="order_date", a=FEB, b=MAR, by="region")
        self.assertEqual(n.op, "count")
        self.assertAlmostEqual(n.change, -120.0)
        with self.assertRaises(ValueError):
            C.compare(orders(), date_column="order_date", a=FEB, b=MAR, by="region", measure="channel")
        with self.assertRaises(ValueError):
            C.compare(orders(), date_column="order_date", a=C.Window(start="2030-01-01", end="2030-01-31"), b=MAR, by="region")
        self.assertEqual(C.ChangeResult.read(n.model_dump_json()), n)

    def test_d_s1_on_the_replica_sales_table(self):  # "Why did revenue drop in March 2026 compared with February?"
        from dclab_rnd.agent_eval.home import tables
        t = tables.build("sales")
        r = C.best_split(t, ["region", "channel", "product_category", "status"], date_column="order_date", a=FEB, b=MAR, measure="amount")
        self.assertIsNotNone(r)
        for s in r.segments:
            self.assertAlmostEqual(s.volume + s.mix + s.rate, s.change, places=6)
        self.assertAlmostEqual(sum(s.change for s in r.segments), r.change, places=6)
        d = pd.to_datetime(t.order_date)
        want = t[(d.dt.year == 2026) & (d.dt.month == 3)].groupby(r.by)["amount"].sum().sub(
            t[(d.dt.year == 2026) & (d.dt.month == 2)].groupby(r.by)["amount"].sum(), fill_value=0)
        if r.segments[0].segment != C.OTHER:
            self.assertAlmostEqual(r.segments[0].change, want[r.segments[0].segment], places=6)


if __name__ == "__main__":
    unittest.main()
