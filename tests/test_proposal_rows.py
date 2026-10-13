"""The solution proposal reads the target on the provisional training rows only (16.15's follow-up): the earliest 75 %
when a column orders the rows in time, so the holdout every engine split later takes (the latest 20 % of a tabular
split, the latest 25 % of starts, a forecast's last dates) never shapes what is proposed; a seeded 75 % otherwise, said
plainly. Changing only the targets of those later rows must leave the proposal as it was."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd

    from dclab_rnd import kinds
    from dclab_rnd.kinds import acceptance
    from dclab_rnd.studio.data import profile_table
    from dclab_rnd.studio.solution import PROVISIONAL_SHARE, propose, provisional_reading, provisional_training_rows
except ImportError as error:  # the agent environment has no pandas
    MISSING = str(error)
else:
    MISSING = ""

TIMED = ("binary", "multiclass", "regression", "forecasting", "time_to_event")  # replicas whose table has a time column


def changed(y, rows):
    """The targets of ``rows`` changed beyond recognition: numbers scaled, classes moved to the next class."""
    y = y.copy()
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        y[rows] = y[rows] * 3 + 1
    else:
        values = sorted(y.dropna().unique(), key=str)
        following = {v: values[(i + 1) % len(values)] for i, v in enumerate(values)}
        y[rows] = y[rows].map(following)
    return y


def summary(proposal):
    return {"task": proposal["task"], "metric": proposal["metric"], "positive_label": proposal.get("positive_label"),
            "forbidden": sorted(f["column"] for f in proposal["forbidden"]),
            "plan": json.dumps(proposal.get("plan"), sort_keys=True, default=str), "rows_read": proposal["rows_read"]}


def first_time_column(table, target):
    return next(c for c in profile_table(table)["time_candidates"] if c != target)


def stamps(column):
    return column if pd.api.types.is_numeric_dtype(column) else pd.to_datetime(column, errors="coerce")


@unittest.skipIf(MISSING, MISSING)
class LaterTargetsDoNotShapeTheProposalTest(unittest.TestCase):
    def test_changing_the_latest_quarter_of_targets_changes_nothing_proposed(self):
        for key in TIMED:
            with self.subTest(kind=key):
                case = kinds.get(key).case(0)
                table, profile = case.table, profile_table(case.table)
                time = first_time_column(table, case.target)
                later = stamps(table[time]).rank(method="first") > PROVISIONAL_SHARE * len(table)
                before = summary(propose(table, profile, case.target, case.task, case.pack))
                after = summary(propose(table.assign(**{case.target: changed(table[case.target], later)}), profile, case.target, case.task, case.pack))
                self.assertEqual(before, after)
                self.assertIn(f"by `{time}`", before["rows_read"])

    def test_the_rows_read_end_before_each_engine_holdout_begins(self):
        """The engine's own holdout, read from the project it builds, starts after the last row the proposal read."""
        from dclab_core import forecast, survival
        from dclab_rnd.studio import engine

        for key in ("binary", "regression", "forecasting", "time_to_event"):
            with self.subTest(kind=key):
                case = kinds.get(key).case(0)
                table = case.table
                time = first_time_column(table, case.target)
                read = provisional_training_rows(table.index[table[case.target].notna()], table.loc[table[case.target].notna(), time])
                last_read = stamps(table.loc[read, time]).max()
                if key == "forecasting":
                    f = case.solution["kind_fields"]
                    first_held = table.loc[pd.to_datetime(table[time]) > forecast.holdout_origin(table, time=time, horizon=f["horizon"],
                                                                                                gap=f.get("gap", 0)), time]
                    first_held = pd.to_datetime(first_held).min()
                elif key == "time_to_event":
                    f = case.solution["kind_fields"]
                    first_held = pd.Timestamp(survival.split(table, duration=f["duration_column"], event=f["event_column"],
                                                             start=f["start_column"])["split"]["holdout_from"])
                else:
                    store, pid = acceptance.build(case, Path(tempfile.mkdtemp()))
                    project = store.get(pid)
                    self.assertEqual(project["solution"]["time_column"], time)  # the split follows the column the proposal read by
                    bundle = engine.prepare(store, project).bundle
                    first_held = stamps(bundle.X.iloc[bundle.test_idx][time]).min()
                self.assertLess(last_read, first_held)

    def test_without_a_time_column_a_seeded_share_is_read_and_said(self):
        rng = np.random.default_rng(0)
        table = pd.DataFrame({"x": rng.normal(size=400), "y": rng.integers(0, 2, 400).astype(float)})
        table.loc[:39, "y"] = np.nan  # rows without a target are never read
        proposal = propose(table, profile_table(table), "y", "binary")
        self.assertIn("seeded 270 of 360 rows", proposal["rows_read"])
        self.assertIn("random holdout", proposal["rows_read"])
        self.assertFalse(set(provisional_training_rows(table.index[table["y"].notna()])) & set(range(40)))

    def test_a_timed_table_with_missing_targets_and_invalid_times_is_read_before_the_engines_holdout(self):
        """Partial labels and broken stamps: the proposal neither fails nor reads a row the engine holds out."""
        rng = np.random.default_rng(1)
        n = 1000
        stamps = pd.date_range("2025-01-01", periods=n, freq="h")
        table = pd.DataFrame({"created_at": stamps.strftime("%Y-%m-%d %H:%M:%S"), "x": rng.normal(size=n),
                              "y": rng.integers(0, 2, n).astype(float)})
        table.loc[rng.choice(n, 99, replace=False), "created_at"] = "not a date"  # 90.1 % valid
        table.loc[n - 60:, "y"] = np.nan  # the latest rows have no outcome yet
        proposal = propose(table, profile_table(table), "y", "binary")
        self.assertIn("by `created_at`", proposal["rows_read"])
        present = table.index[table["y"].notna()]
        read, by_time = provisional_reading(present, table.loc[present, "created_at"])
        self.assertTrue(by_time)
        self.assertFalse(set(read) & set(table.index[table["created_at"] == "not a date"]))
        case = acceptance.Case(table=table, target="y", task="binary", moment="When the order is created, before anything else",
                               solution={"time_column": "created_at"})
        store, pid = acceptance.build(case, Path(tempfile.mkdtemp()))
        from dclab_rnd.studio import engine

        bundle = engine.prepare(store, store.get(pid)).bundle
        held = set(pd.to_datetime(bundle.X.iloc[bundle.test_idx]["created_at"]))
        self.assertFalse(held & set(pd.to_datetime(table.loc[read, "created_at"])))  # not one held-out row was read

    def test_the_metric_follows_the_class_balance_of_the_rows_read(self):
        """The latest quarter's balance crosses the line between two metrics; the proposal must not see it."""
        n = 1000
        table = pd.DataFrame({"t": pd.date_range("2025-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
                              "x": np.arange(n) % 7, "y": (np.arange(n) % 11 == 0).astype(int)})
        later = table.index >= 750
        quieter = table.assign(y=np.where(later, (np.arange(n) % 40 == 0).astype(int), table["y"]))
        louder = table.assign(y=np.where(later, (np.arange(n) % 7 == 0).astype(int), table["y"]))
        metrics = {propose(t, profile_table(t), "y", "binary")["metric"] for t in (quieter, louder)}
        self.assertEqual(len(metrics), 1, metrics)

    def test_a_growing_panel_keeps_the_forecast_holdout_unread(self):
        """Few series early, many late: the forecast's last dates are more than a quarter of the rows."""
        from dclab_core import forecast

        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        rows = [(f"s{i}", d) for i in range(10) for d in dates] + [(f"s{i}", d) for i in range(10, 110) for d in dates[60:]]
        table = pd.DataFrame(rows, columns=["series", "date"]).assign(units=1.0)
        origin = forecast.holdout_origin(table, time="date", horizon=14)
        self.assertGreater((table["date"] > origin).mean(), 1 - PROVISIONAL_SHARE)  # the scenario: the holdout is over 25 % of rows
        read = provisional_training_rows(table.index, table["date"])
        self.assertLessEqual(table.loc[read, "date"].max(), origin)

    def test_too_few_valid_times_read_a_seeded_share_and_say_so(self):
        table = pd.DataFrame({"event_time": [np.nan] * 380 + list(range(20)), "x": np.arange(400.0), "y": np.arange(400) % 2})
        read, by_time = provisional_reading(table.index, table["event_time"])
        self.assertFalse(by_time)
        proposal = propose(table, profile_table(table), "y", "binary", time_column="event_time")
        self.assertIn("too few valid times", proposal["rows_read"])
        self.assertNotIn("earliest", proposal["rows_read"])

    def test_the_share_leaves_room_for_every_engine_holdout(self):
        from dclab_rnd.expansion import datasets
        from dclab_core import survival
        import inspect

        survival_holdout = inspect.signature(survival.split).parameters["holdout"].default
        self.assertLessEqual(PROVISIONAL_SHARE, 1 - datasets.HOLDOUT_FRACTION)
        self.assertLessEqual(PROVISIONAL_SHARE, 1 - survival_holdout)


if __name__ == "__main__":
    unittest.main()
