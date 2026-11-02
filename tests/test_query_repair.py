"""The owner's Telco session (2026-11-02): a share asked of a yes/no column, a count with a condition, and a refusal that says
how to write it. Pure code on a small table; no model."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import query as Q  # noqa: E402
from dclab_rnd.agents import data_tools  # noqa: E402


def telco() -> pd.DataFrame:
    rows = []
    for contract, total, left in (("Month-to-month", 40, 20), ("One year", 30, 6), ("Two year", 30, 1)):
        rows += [{"Contract": contract, "Churn": "Yes" if i < left else "No", "tenure": i} for i in range(total)]
    return pd.DataFrame(rows)


class CountWhereAndShare(unittest.TestCase):
    def test_a_count_with_a_condition_counts_only_the_rows_that_meet_it(self):
        out = data_tools.query(telco(), group_by=["Contract"], measures=[
            {"name": "left", "op": "count", "where": {"column": "Churn", "op": "eq", "value": "Yes"}}])
        self.assertNotIn("error", out)
        by = {r[0]: r[1] for r in out["rows"]}
        self.assertEqual(by, {"Month-to-month": 20, "One year": 6, "Two year": None})  # one row meeting the condition is withheld, never counted out
        self.assertEqual(by["Month-to-month"], 20)  # never the whole group (40): the silent wrong count of the session

    def test_a_share_asked_of_a_yes_no_column_without_a_condition_counts_its_yes_value_and_says_so(self):
        out = data_tools.query(telco(), group_by=["Contract"], measures=[{"name": "churn_rate", "op": "rate", "column": "Churn"}])
        self.assertNotIn("error", out)
        self.assertEqual({r[0]: round(r[1], 3) for r in out["rows"]}["Month-to-month"], 0.5)
        self.assertTrue(any("Churn" in n and "Yes" in n for n in out["counted"]))

    def test_true_asked_of_a_column_holding_yes_and_no_is_read_as_yes_and_said(self):
        out = data_tools.query(telco(), group_by=["Contract"], measures=[
            {"name": "share", "op": "rate", "where": {"column": "Churn", "op": "eq", "value": True}}])
        self.assertNotIn("error", out)
        self.assertEqual({r[0]: round(r[1], 3) for r in out["rows"]}["Month-to-month"], 0.5)
        self.assertTrue(any("True" in n and "Yes" in n for n in out["counted"]))

    def test_a_share_of_a_column_that_is_not_yes_no_is_still_refused_with_the_way_to_write_it(self):
        out = data_tools.query(telco(), group_by=["Churn"], measures=[{"name": "share", "op": "rate", "column": "Contract"}])
        self.assertIn("a rate needs the condition it counts (where)", out["error"])
        self.assertIn('"op": "rate", "where"', out["error"])  # the fix is in the refusal

    def test_a_condition_on_a_mean_is_refused_and_says_what_a_condition_belongs_to(self):
        out = data_tools.query(telco(), measures=[{"name": "m", "op": "mean", "column": "tenure",
                                                  "where": {"column": "Churn", "op": "eq", "value": "Yes"}}])
        self.assertIn("belongs to a count or a share", out["error"])

    def test_the_result_carries_each_number_as_it_may_be_said(self):
        out = data_tools.query(telco(), group_by=["Contract"], measures=[
            {"name": "customers", "op": "count"}, {"name": "churn_rate", "op": "rate", "column": "Churn"}], sort=[{"by": "churn_rate"}])
        said = {r[0]: r[1:] for r in out["as_said"]}
        self.assertEqual(said["Month-to-month"], ["40", "50.0%"])

    def test_repair_changes_nothing_when_the_plan_is_right(self):
        plan = Q.QueryPlan(measures=[Q.Measure(name="s", op="rate", where=Q.Filter(column="Churn", op="eq", value="Yes"))])
        fixed, notes = Q.repair(plan, telco())
        self.assertEqual((fixed, notes), (plan, []))

    def test_a_count_with_a_condition_never_shows_one_or_two_rows(self):
        frame = telco()
        frame.loc[frame.index[-1], "tenure"] = 999  # one row holds a value
        out = data_tools.query(frame, group_by=["Contract"], measures=[{"name": "n", "op": "count", "where": {"column": "tenure", "op": "eq", "value": 999}}])
        self.assertNotIn(1, [r[1] for r in out["rows"]])  # withheld, not counted out
        self.assertIn(None, [r[1] for r in out["rows"]])

    def test_a_share_named_for_the_other_value_is_not_guessed(self):
        out = data_tools.query(telco(), group_by=["Contract"], measures=[{"name": "no_share", "op": "rate", "column": "Churn"}])
        self.assertIn("a rate needs the condition it counts (where)", out["error"])


if __name__ == "__main__":
    unittest.main()
