"""studio.derive: the Data engineer's new columns and lookup joins, parsed and checked, never run as code (2026-10-21)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_rnd.studio import derive  # noqa: E402

FRAME = pd.DataFrame({"spend": [10.0, 20.0, 30.0, 0.0], "tenure_months": [2, 4, 0, 5], "plan": ["basic", "pro", "pro", "basic"],
                      "signup": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"]),
                      "last_seen": pd.to_datetime(["2025-01-11", "2025-02-03", "2025-03-01", "2025-04-30"]),
                      "churned": [0, 1, 0, 1], "refund_amount": [0, 5, 0, 9], "monthly fee": [5.0, 9.0, 9.0, 5.0]})


class FormulaTest(unittest.TestCase):
    def test_arithmetic_dates_functions_and_quoted_names(self):
        self.assertEqual(derive.compute(FRAME, "spend / tenure_months").round(2).tolist()[:2], [5.0, 5.0])
        self.assertTrue(np.isnan(derive.compute(FRAME, "spend / tenure_months")[2]))  # divided by zero: missing, not infinite
        self.assertEqual(derive.compute(FRAME, "days_between(signup, last_seen)").tolist(), [10.0, 2.0, 0.0, 29.0])
        self.assertEqual(derive.compute(FRAME, "month(signup)").tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(derive.compute(FRAME, "`monthly fee` * 12").tolist(), [60.0, 108.0, 108.0, 60.0])
        self.assertEqual(derive.compute(FRAME, "spend > 15 and tenure_months > 1").tolist(), [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(derive.compute(FRAME, "round(log1p(spend), 2)").tolist()[0], 2.4)

    def test_nothing_but_the_allowed_parts_is_ever_run(self):
        for bad in ("__import__('os').system('echo hi')", "spend.__class__", "open('x')", "[c for c in spend]", "spend[0]", "lambda: 1",
                    "eval('1')", "spend if 1 else 2", "'text' + plan", "exec", "nosuch * 2", "", "1 + 2", "f'{spend}'", "spend; tenure_months"):
            with self.assertRaises(derive.ChangeRefused, msg=bad):
                derive.compute(FRAME, bad)

    def test_a_new_column_never_reads_the_outcome_a_forbidden_column_or_a_leak_suspect(self):
        for formula, word in (("churned * 2", "outcome"), ("refund_amount + spend", "forbidden"), ("tenure_months + 1", "leak suspect")):
            with self.assertRaises(derive.ChangeRefused) as caught:
                derive.check_add(FRAME, "x", formula, "churned", ["refund_amount"], ["tenure_months"])
            self.assertIn(word, str(caught.exception))
        op = derive.check_add(FRAME, "spend_per_month", "spend / tenure_months", "churned", ["refund_amount"])
        self.assertEqual((op["reads"], op["missing_share"]), (["spend", "tenure_months"], 0.25))
        for name in ("spend", "1x", "a b", ""):
            with self.assertRaises(derive.ChangeRefused):
                derive.check_add(FRAME, name, "spend * 2", "churned", [])


class JoinTest(unittest.TestCase):
    def test_a_lookup_joins_and_an_event_table_is_refused(self):
        plans = pd.DataFrame({"plan": ["basic", "pro"], "price": [5.0, 9.0], "spend": [1, 2]})
        op = derive.check_join(FRAME, plans, "plans", "plan", "churned", [])
        self.assertEqual(op["columns"], ["price", "plans_spend"])  # a name the table has is kept apart
        out = derive.apply(FRAME, [op], lambda name: plans)
        self.assertEqual(out["price"].tolist(), [5.0, 9.0, 9.0, 5.0])
        self.assertEqual(len(out), len(FRAME))
        events = pd.DataFrame({"plan": ["basic", "basic", "pro"], "amount": [1, 2, 3]})
        with self.assertRaises(derive.ChangeRefused) as caught:
            derive.check_join(FRAME, events, "orders", "plan", "churned", [])
        self.assertIn("table of events", str(caught.exception))
        with self.assertRaises(derive.ChangeRefused):
            derive.check_join(FRAME, pd.DataFrame({"plan": ["basic"], "churned": [1]}), "labels", "plan", "churned", [])

    def test_changes_apply_in_order_and_the_table_keeps_its_rows(self):
        plans = pd.DataFrame({"plan": ["basic", "pro"], "price": [5.0, 9.0]})
        ops = [derive.check_join(FRAME, plans, "plans", "plan", "churned", []), {"op": "add_column", "name": "price_share", "formula": "price / spend"}]
        out = derive.apply(FRAME, ops, lambda name: plans)
        self.assertEqual(out["price_share"].round(2).tolist()[:2], [0.5, 0.45])
        self.assertEqual(list(FRAME.columns), [c for c in out.columns if c not in ("price", "price_share")])


if __name__ == "__main__":
    unittest.main()
