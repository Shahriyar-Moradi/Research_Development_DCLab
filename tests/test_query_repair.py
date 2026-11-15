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

    def test_a_condition_on_a_mean_is_the_mean_of_the_rows_that_meet_it(self):
        """Refused until dclab-core 0.159.0 ("belongs to a count or a share"); HAB-002 showed the lead agent refused eleven times
        for asking a sum or a mean of some rows. The condition never names one identifier."""
        frame = telco()
        out = data_tools.query(frame, measures=[{"name": "m", "op": "mean", "column": "tenure",
                                                "where": {"column": "Churn", "op": "eq", "value": "Yes"}}])
        self.assertNotIn("error", out)
        self.assertAlmostEqual(out["rows"][0][0], float(frame.loc[frame["Churn"] == "Yes", "tenure"].mean()), places=6)

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


class TimesAfterTheReview(unittest.TestCase):
    """The cloud review of 0.159.0 (2026-11-15): a time kept as text is compared as a time, one value of a timestamp is one
    row, a conditioned measure is a comparison H8 must see, and a day's label or a first date is a cell value."""

    def frame(self):
        stamps = pd.date_range("2025-10-01 09:30", periods=92, freq="D").strftime("%Y-%m-%d %H:%M")
        return pd.DataFrame({"at": list(stamps), "amount": [10.0] * 92, "region": ["north", "south"] * 46,
                             "left": ["Yes", "No", "No", "No"] * 23})

    def test_a_day_bound_covers_the_whole_of_that_day_on_a_text_timestamp(self):
        plan = Q.QueryPlan(measures=[{"name": "q4", "op": "sum", "column": "amount",
                                      "where": {"column": "at", "op": "between", "value": ["2025-10-01", "2025-12-31"]}}])
        frame = self.frame()
        self.assertEqual(Q.validate(plan, frame), [])
        self.assertEqual(Q.execute(plan, frame).rows[0][0], 920.0)  # 2025-12-31 09:30 is inside "to 2025-12-31"

    def test_a_date_written_day_first_is_compared_as_a_date_not_as_text(self):
        frame = pd.DataFrame({"at": pd.date_range("2025-11-13", periods=40, freq="D").strftime("%d/%m/%Y"), "amount": [1.0] * 40})
        plan = Q.QueryPlan(measures=[{"name": "n", "op": "count"}], filters=[{"column": "at", "op": "ge", "value": "2025-12-01"}])
        self.assertEqual(Q.execute(plan, frame).rows[0][0], 22)  # as text, "13/11/2025" >= "2025-12-01" matched nothing

    def test_one_value_of_a_timestamp_is_one_row_and_is_refused(self):
        frame = self.frame()
        one = frame["at"][5]
        refused = Q.validate(Q.QueryPlan(measures=[{"name": "s", "op": "sum", "column": "amount"}], filters=[{"column": "at", "op": "ne", "value": one}]), frame)
        self.assertTrue(any("range" in w for w in refused), refused)
        refused = Q.validate(Q.QueryPlan(measures=[{"name": "s", "op": "sum", "column": "amount", "where": {"column": "at", "op": "ne", "value": one}}]), frame)
        self.assertTrue(any("range" in w for w in refused), refused)
        refused = Q.validate(Q.QueryPlan(measures=[{"name": "s", "op": "rate", "where": {"column": "at", "op": "eq", "value": one}}]), frame)
        self.assertTrue(any("range" in w for w in refused), refused)
        ranged = Q.QueryPlan(measures=[{"name": "s", "op": "sum", "column": "amount"}], filters=[{"column": "at", "op": "lt", "value": "2025-11-01"}])
        self.assertEqual(Q.validate(ranged, frame), [])

    def test_a_conditioned_measure_is_recorded_as_a_comparison_with_its_condition(self):
        plan = Q.QueryPlan(measures=[{"name": "stay", "op": "mean", "column": "amount", "where": {"column": "left", "op": "eq", "value": "No"}},
                                     {"name": "go", "op": "mean", "column": "amount", "where": {"column": "left", "op": "eq", "value": "Yes"}}])
        self.assertEqual(Q.exposure(plan)["columns_related"], [["left", "amount"]])

    def test_a_days_label_and_a_first_date_count_as_cell_values(self):
        frame, seen = self.frame(), set()
        out = data_tools.query(frame, measures=[{"name": "n", "op": "count"}], seen=seen,
                               period={"column": "at", "grain": "month", "start": "2025-10-01", "end": "2025-12-31"})
        self.assertNotIn("error", out)
        self.assertEqual(seen, set())  # a month's label is derived: no cell value
        frame = pd.concat([frame] * 12, ignore_index=True)
        out = data_tools.query(frame, measures=[{"name": "n", "op": "count"}], seen=seen,
                               period={"column": "at", "grain": "day", "start": "2025-10-01", "end": "2025-10-05"})
        self.assertNotIn("error", out)
        self.assertEqual(len(seen), 5)  # five dates of the column were shown
        out = data_tools.query(frame, measures=[{"name": "first", "op": "min", "column": "at"}], seen=seen)
        self.assertNotIn("error", out)
        self.assertEqual(len(seen), 6)

    def test_a_first_date_per_group_is_a_table_and_a_conditioned_total_says_its_condition(self):
        from dclab_rnd import home_chart, home_query

        frame = pd.concat([self.frame()] * 3, ignore_index=True)
        plan = Q.QueryPlan(group_by=["region"], measures=[{"name": "last", "op": "max", "column": "at"}])
        self.assertEqual(home_chart.spec(plan, Q.execute(plan, frame))["kind"], "table")  # no bar has a date's height: it crashed on the sort
        plan = Q.QueryPlan(measures=[{"name": "q4", "op": "sum", "column": "amount", "where": {"column": "at", "op": "between", "value": ["2025-10-01", "2025-12-31"]}}])
        self.assertIn("where at between", home_query.describe(plan))
        self.assertIn("where at between", home_chart.spec(plan, Q.execute(plan, frame))["table"]["columns"][0])
        self.assertIn("condition on at", home_query.model_text(plan))
        self.assertNotIn("2025", home_query.model_text(plan))  # the Home model is shown no value
