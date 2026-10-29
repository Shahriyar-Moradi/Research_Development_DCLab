"""The three traps Home's sixth sealed set caught it on (HOME-025, 2026-10-29; the set is spent, so its cases are dev now):
a blanket "drop every order with any missing value" applied without asking, a list of customers with their IDs in the chat
read as an analysis, and an average of a column the table lacks kept as the measure."""
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval import home  # noqa: E402
from dclab_rnd.draft import brief as briefs  # noqa: E402
from dclab_rnd.draft import exits, intent, prepare  # noqa: E402

SALES = ["order_id", "order_date", "region", "channel", "product_category", "amount", "status", "customer_id"]


class RuleTests(unittest.TestCase):
    def test_a_measure_on_a_column_the_table_lacks_is_missing_in_plain_words_too(self):
        self.assertEqual(briefs.absent_columns("average discount per order", SALES, "measure"), ["discount"])
        for fine in ("average order value", "total amount", "count of orders", "the share of orders per region"):
            self.assertEqual(briefs.absent_columns(fine, SALES, "measure"), [], fine)
        self.assertEqual(briefs.absent_columns("average discount per order", SALES, "breakdown"), [])  # a measure's words only
        self.assertEqual(briefs.absent_columns("average MonthlyCharges", ["MonthlyCharges", "tenure"], "measure"), [])
        self.assertEqual(briefs.absent_columns("anything", None, "measure"), [])  # no table: nothing is said missing

    def test_any_missing_value_is_a_step_on_every_column_and_duplicate_ids_use_the_id(self):
        frame = pd.DataFrame({"order_id": [1, 1, 2, 3], "amount": [5.0, 5.0, None, 7.0], "status": ["paid", "paid", "paid", None]})
        steps, unread = prepare.read_steps("Clean it: drop every order that has any missing value and remove duplicate order_ids.", frame)
        kinds = {s.kind: s for s in steps}
        self.assertEqual(sorted(kinds["drop_missing"].columns), ["amount", "order_id", "status"])
        self.assertEqual(kinds["dedupe"].keys, ["order_id"])
        self.assertEqual(unread, [])

    def test_rows_with_their_ids_in_the_chat_are_a_prepared_table(self):
        words = "Show me here in the chat all the customers who churned last month, with their customer_id, city and plan."
        self.assertEqual(intent.read(words)["track"], "preparation")
        self.assertEqual(exits.suggest(words, "analysis")["track"], "preparation")
        self.assertNotEqual(intent.read("Show me the average spend of all customers per plan")["track"], "preparation")


class ReviewTests(unittest.TestCase):  # the review of C1 and C5 (2026-10-29)
    def test_function_words_after_an_aggregate_are_never_a_missing_column(self):
        for words in ("total by region", "total for each region", "average across all orders", "count of distinct customers",
                      "average and median amount", "total last month", "total in euros", "on average how much"):
            self.assertEqual(briefs.absent_columns(words, SALES, "measure"), [], words)

    def test_missing_values_in_a_named_column_stay_that_column(self):
        frame = pd.DataFrame({"order_id": [1, 2], "amount": [1.0, None], "status": ["a", None]})
        steps, _ = prepare.read_steps("drop rows with missing values in amount", frame)
        self.assertEqual([(s.kind, s.columns) for s in steps], [("drop_missing", ["amount"])])
        steps, unread = prepare.read_steps("remove rows with missing values in the discount column", frame)
        self.assertEqual((steps, len(unread)), ([], 1))  # a column the table lacks is said unread, never every column

    def test_numbers_per_group_are_never_a_rows_request(self):
        for words in ("display every region with its manager name and its total sales",
                      "Show me the total revenue for each region with their store names", "show me all orders per region with their totals"):
            self.assertNotEqual(intent.read(words)["track"], "preparation", words)
            self.assertIsNone(exits.suggest(words, "analysis"), words)
        self.assertEqual(intent.read("List all customers with their emails")["track"], "preparation")


class SpentSealedCasesTests(unittest.TestCase):
    def test_the_three_missed_traps_pass_with_the_rules(self):
        cases = [c for c in json.loads((ROOT / "dclab_rnd/agent_eval/home/requests_sealed_v6.json").read_text())
                 if c["id"] in ("s6-16", "s6-22", "s6-30")]
        self.assertEqual(len(cases), 3)
        result = home.measure(cases)
        for row in result["rows"]:
            self.assertTrue(row["scores"]["rules"]["trap"], row["case"])
        s16 = next(r for r in result["rows"] if r["case"] == "s6-16")
        self.assertEqual(s16["scores"]["rules"]["fields"]["rules"], "right")  # the steps, not "Clean the sales table for me"


if __name__ == "__main__":
    unittest.main()
