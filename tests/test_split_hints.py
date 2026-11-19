"""The planning practice set (2026-11-19) found the rules plan never kept new entities apart (0 of 16), split at random
when a table had two dates, and lost an outcome named in backticks to an everyday word. Each fix, and the guard that an
entity split the engine cannot make is asked about instead of failing the build."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dclab_core import split_hints as H  # noqa: E402


class WordsTests(unittest.TestCase):
    def test_new_entities_are_read_from_the_brief(self):
        self.assertEqual(H.new_entities("Three new presses arrive next quarter"), "presses")
        self.assertEqual(H.new_entities("including the six stores we open next year"), "stores")
        self.assertEqual(H.new_entities("for the buildings we are taking over next year"), "buildings")
        self.assertEqual(H.new_entities("It will be used on next year's intake."), "")

    def test_a_new_thing_that_is_no_entity_is_not_read_as_one(self):
        for brief in ("When a new support ticket arrives, route it", "each new booking", "Our new CRM shows accounts",
                      "Which customers will leave next month?", "Predict returns for our customers",
                      # the review's briefs: a place, a negation, a column, a kind of row, a category
                      "Forecast weekly sales for our New York stores.", "Forecast sales for existing stores only; new stores are out of scope.",
                      "We do not plan to open new branches.", "The export now has additional fields such as rainfall.",
                      "Flag fraud, especially on transactions from new accounts.", "The model must handle unseen categories"):
            self.assertIsNone(H.new_entities(brief), brief)


class ColumnsTests(unittest.TestCase):
    def frame(self):
        rng = np.random.default_rng(3)
        n = 600
        return pd.DataFrame({"unit_id": [f"u{i}" for i in range(n)], "store_no": rng.integers(0, 30, n), "course_code": rng.integers(0, 12, n),
                             "student_no": [f"s{i}" for i in rng.integers(0, 150, n)], "pickup_hour": rng.integers(0, 24, n), "y": rng.random(n)})

    def test_the_entity_is_a_key_named_column_with_repeats_named_by_the_brief_or_the_finest(self):
        f = self.frame()
        self.assertEqual(H.entity_candidates(f, list(f.columns), exclude=["y"]), ["store_no", "course_code", "student_no"])  # not unit_id (unique), not pickup_hour (no key name)
        self.assertEqual(H.entity_column(f, list(f.columns), "stores", exclude=["y"]), "store_no")
        self.assertEqual(H.entity_column(f, list(f.columns), "", exclude=["y"]), "student_no")  # an intake: the finest entity
        self.assertIsNone(H.entity_column(f[["pickup_hour", "y"]], ["pickup_hour", "y"], "stores"))
        self.assertIsNone(H.entity_column(f, list(f.columns), "presses", exclude=["y"]))  # no column the noun names: never a guess

    def test_a_count_or_a_flag_is_no_entity(self):
        rng = np.random.default_rng(4)
        f = pd.DataFrame({"num_items": rng.integers(0, 40, 500), "no_show": rng.integers(0, 2, 500), "n_number": rng.integers(0, 40, 500)})
        self.assertEqual(H.entity_candidates(f, list(f.columns)), [])

    def test_of_two_dates_the_one_known_first_in_the_same_period_orders_the_split(self):
        rng = np.random.default_rng(5)
        booked = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.uniform(0, 300, 500), unit="D")
        f = pd.DataFrame({"booking_date": booked.astype(str), "appt_date": booked + pd.to_timedelta(rng.uniform(0, 30, 500), unit="D"),
                          "birth_date": pd.Timestamp("1960-01-01") + pd.to_timedelta(rng.uniform(0, 12000, 500), unit="D"),
                          "other": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.uniform(0, 300, 500), unit="D")})
        self.assertEqual(H.first_known_time(f, ["appt_date", "booking_date"]), "booking_date")
        self.assertIsNone(H.first_known_time(f, ["birth_date", "appt_date"]))  # decades earlier: an attribute, not the row's time
        self.assertIsNone(H.first_known_time(f, ["booking_date", "other"]))  # neither comes first: a person's choice
        self.assertIsNone(H.first_known_time(f, ["booking_date"]))

    def test_a_sign_up_date_beside_the_rows_snapshot_date_is_not_the_rows_time(self):
        rng = np.random.default_rng(6)
        snapshot = pd.to_datetime(rng.choice(pd.date_range("2025-01-01", "2025-06-01", freq="MS"), 600))
        signup = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.uniform(0, 2600, 600), unit="D")
        f = pd.DataFrame({"signup_date": signup.where(signup < snapshot, snapshot - pd.Timedelta(days=1)), "snapshot_date": snapshot})
        self.assertIsNone(H.first_known_time(f, ["signup_date", "snapshot_date"]))

    def test_dates_with_and_without_a_zone_are_read_one_way_and_never_raise(self):
        days = pd.date_range("2025-01-01", periods=300, freq="D")
        f = pd.DataFrame({"created_at": days.strftime("%Y-%m-%dT%H:%M:%SZ"), "due_date": (days + pd.Timedelta(days=3)).strftime("%Y-%m-%d")})
        self.assertEqual(H.first_known_time(f, ["due_date", "created_at"]), "created_at")
        f["due_date"] = "not a date"
        self.assertIsNone(H.first_known_time(f, ["due_date", "created_at"]))


class ProductTests(unittest.TestCase):
    def test_a_name_in_backticks_wins_over_an_everyday_word(self):
        from dclab_rnd.studio.data import named_outcome

        profile = {"columns": [{"name": n, "kind": "categorical", "unique": 5} for n in ("line", "shift", "defect")]}
        self.assertEqual(named_outcome(profile, "Catch units on the line that fail. Predict `defect`."), "defect")
        self.assertIsNone(named_outcome(profile, "Catch units on the line that turn out defect."))  # two names, none pointed at: asked
        churn = {"columns": [{"name": n, "kind": "categorical", "unique": 5} for n in ("churn", "plan_type", "tenure")]}
        self.assertEqual(named_outcome(churn, "Predict churn next month. Note: `plan_type` was renamed last year."), "churn")

    def test_the_standard_plan_orders_two_dates_by_the_first_known(self):
        from dclab_rnd.intern.loop import found_time_column

        booked = pd.Series(pd.date_range("2025-01-01", periods=300, freq="D"))
        frame = pd.DataFrame({"booking_date": booked, "appt_date": booked + pd.Timedelta(days=3)})
        profile = {"columns": [{"name": "booking_date", "kind": "datetime"}, {"name": "appt_date", "kind": "datetime"}]}
        why = {}
        self.assertEqual(found_time_column({}, {"time_candidates": ["appt_date", "booking_date"]}, profile, frame, why), "booking_date")
        self.assertEqual(why, {"among": ["appt_date"]})  # the note to the person names the date it was chosen over
        why = {}
        self.assertEqual(found_time_column({}, {"time_candidates": ["booking_date"]}, profile, frame, why), "booking_date")
        self.assertEqual(why, {})  # one date: no choice was made, and no note claims one

    def test_a_group_the_engine_cannot_split_beside_time_is_a_question_on_every_path(self):
        """The review: the task-execution path applied the entity split with no guard. The guard is in the change rule."""
        from dclab_rnd.tasks import enforced

        traps, solution, columns = ["grouped_split_store_no"], {"target": "y", "time_column": "week"}, ["store_no", "week", "y"]
        out = enforced.changes(traps, [], solution, columns, can_group=lambda g, t: False)
        self.assertEqual([(c["action"], c["field"]) for c in out], [("ask", "group_column")])
        self.assertIn("both cannot hold", out[0]["why"])
        self.assertEqual(enforced.changes(traps, [], solution, columns, can_group=lambda g, t: True)[0]["action"], "apply")

    def test_the_engines_fold_rule_is_read_too(self):
        """The review: 30 stores every week and 10 new ones in the last weeks passed the holdout rule; the folds then failed."""
        from dclab_rnd.expansion.datasets import time_group_feasible

        weeks = pd.date_range("2025-01-06", periods=40, freq="W-MON")
        rows = [(w, s) for i, w in enumerate(weeks) for s in range(30)] + [(w, 100 + s) for w in weeks[-4:] for s in range(10)]
        frame = pd.DataFrame(rows, columns=["week", "store"])
        self.assertFalse(time_group_feasible(frame["week"], frame["store"])[0])


class PlanTests(unittest.TestCase):
    """The rules plan reads new entities from the brief and applies the entity split only when the engine can make it."""

    def project(self, brief, stores_open_over_time):
        from dclab_rnd.studio import ProjectStore, data

        store = ProjectStore(Path(tempfile.mkdtemp()) / "projects")
        pid = store.create("weekly sales", "retail", brief)["id"]
        rng = np.random.default_rng(9)
        weeks = pd.date_range("2025-01-06", periods=40, freq="W-MON")
        if stores_open_over_time:  # a few stores open each week: the latest weeks hold stores never seen before
            rows = [(w, s) for i, w in enumerate(weeks) for s in range(i * 2, i * 2 + 10)]
        else:  # every store every week
            rows = [(w, s) for w in weeks for s in range(30)]
        frame = pd.DataFrame(rows, columns=["week_start", "store_no"])
        frame["footfall"] = rng.integers(100, 900, len(frame))
        frame["units_sold"] = frame["footfall"] * 0.1 + frame["store_no"] % 7 + rng.normal(0, 3, len(frame))
        frame["week_start"] = frame["week_start"].dt.strftime("%Y-%m-%d")
        frame.to_csv(store.data_dir(pid) / "weeks.csv", index=False)
        data.attach_data(store, pid, "weeks.csv")
        project = store.get(pid)
        project["solution"] = {"target": "units_sold", "task": "regression", "prediction_moment": "A week ahead.", "forbidden": [],
                               "identifiers": [], "time_column": "week_start", "labels": "complete"}
        store.save(project)
        return store, pid

    def plan(self, brief, open_over_time):
        from dclab_rnd.tasks import path as P

        store, pid = self.project(brief, open_over_time)
        return P.plan_project(store, pid), store.get(pid)["solution"]

    def test_new_stores_get_an_entity_split_when_the_engine_can_make_it(self):
        out, solution = self.plan("Units per store next week, including the stores we open next year.", True)
        self.assertEqual(solution.get("group_column"), "store_no")
        self.assertEqual(solution.get("time_column"), "week_start")  # both hold: the latest weeks' new stores are the test
        self.assertTrue(any(a["field"] == "group_column" for a in out["applied"]))

    def test_when_every_store_is_in_every_week_it_is_asked_and_the_build_is_not_broken(self):
        out, solution = self.plan("Units per store next week, including the stores we open next year.", False)
        self.assertIsNone(solution.get("group_column"))
        self.assertEqual(solution.get("time_column"), "week_start")
        asked = [a for a in out["asks"] if a["field"] == "group_column"]
        self.assertEqual(len(asked), 1)
        self.assertIn("both cannot hold", asked[0]["question"])

    def test_without_new_entities_in_the_brief_the_same_stores_stay_on_both_sides(self):
        out, solution = self.plan("Units per store next week for our stores.", True)
        self.assertIsNone(solution.get("group_column"))
        self.assertFalse([a for a in out["asks"] if a["field"] == "group_column"])


if __name__ == "__main__":
    unittest.main()
