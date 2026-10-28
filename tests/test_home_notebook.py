"""Saved answers as a notebook (dclab_rnd/home_notebook.py, package H7 part c): every saved answer reproduces from the
cleaned table, and the notebook holds no row."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import query as Q  # noqa: E402
from dclab_rnd import home_notebook, home_query  # noqa: E402
from dclab_rnd.agent_eval.home import tables  # noqa: E402


def answer(plan, table, floor=Q.FLOOR, saved=True):
    result = Q.execute(plan, table, floor=floor)
    return {"plan": plan.model_dump(mode="json"), "shown": home_query.shown_numbers(result), "floor": floor, "saved": saved}


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.table = tables.build("sales")
        self.table.to_parquet(self.folder / "clean.parquet")

    def test_every_saved_answer_reproduces_and_an_unsaved_one_is_left_out(self):
        by_region = Q.QueryPlan(group_by=["region"], measures=[{"name": "total", "op": "sum", "column": "amount"}])
        per_month = Q.QueryPlan(measures=[{"name": "rows", "op": "count"}],
                                period={"column": "order_date", "grain": "month", "start": "2025-01-01", "end": "2025-12-31"})
        loose = Q.QueryPlan(measures=[{"name": "avg", "op": "mean", "column": "amount"}])
        draft = {"answers": [answer(by_region, self.table), answer(loose, self.table, saved=False), answer(per_month, self.table, floor=8)]}
        nb = home_notebook.build(draft)
        self.assertEqual(home_notebook.check(nb, self.folder), 2)
        text = json.dumps(nb)
        self.assertIn("Total of amount by region", text)
        self.assertNotIn("Mean of amount", text)  # not saved: not exported
        self.assertNotIn("C-00", text)  # no customer id, no row
        self.assertNotIn("ORD-", text)

    def test_a_changed_table_fails_the_check(self):
        plan = Q.QueryPlan(group_by=["region"], measures=[{"name": "total", "op": "sum", "column": "amount"}])
        nb = home_notebook.build({"answers": [answer(plan, self.table)]})
        self.table.assign(amount=self.table["amount"] * 2).to_parquet(self.folder / "clean.parquet")
        with self.assertRaises(AssertionError):
            home_notebook.check(nb, self.folder)


if __name__ == "__main__":
    unittest.main()
