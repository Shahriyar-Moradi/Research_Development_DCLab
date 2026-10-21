"""Task and time rules of the standard plan, on invented columns only."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class CodedClassesTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_even_codes_from_one_are_classes_everywhere(self):
        from dclab_core.analyse import _infer_task, coded_classes
        from dclab_rnd.studio.data import detect_task

        codes = pd.Series(self.rng.integers(1, 27, 4000))
        self.assertTrue(coded_classes(codes))
        self.assertEqual(_infer_task(codes), "multiclass")
        self.assertEqual(detect_task(pd.DataFrame({"y": codes}), "y")["task"], "multiclass")

    def test_measures_and_long_tails_stay_numbers(self):
        from dclab_core.analyse import _infer_task, coded_classes

        ages = pd.Series(self.rng.integers(18, 44, 4000))
        tail = pd.Series(np.minimum(self.rng.geometric(0.08, 4000), 40))
        decimals = pd.Series(self.rng.normal(size=500))
        for series in (ages, tail, decimals):
            self.assertFalse(coded_classes(series))
        self.assertEqual(_infer_task(ages), "regression")
        self.assertEqual(_infer_task(decimals), "regression")

    def test_the_task_note_says_when_it_is_a_coded_reading(self):
        from dclab_rnd.studio.data import detect_task

        note = detect_task(pd.DataFrame({"y": pd.Series(self.rng.integers(1, 27, 4000))}), "y")["note"]
        self.assertIn("coded classes", note)
        self.assertIn("set the task to regression", note)


class FoundTimeColumnTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        n = 400
        self.frame = pd.DataFrame({"elapsed": np.sort(rng.integers(0, 10_000, n)), "wait_time_min": rng.integers(1, 90, n),
                                   "created": pd.date_range("2025-01-01", periods=n, freq="h")})
        self.profile = {"columns": [{"name": "created", "kind": "datetime"}, {"name": "elapsed", "kind": "numeric"},
                                    {"name": "wait_time_min", "kind": "numeric"}, {"name": "chargeback_at", "kind": "datetime"}]}

    def found(self, candidates, suggestion=None, forbidden=()):
        from dclab_rnd.intern.loop import found_time_column

        proposal = {"time_candidates": list(candidates), "forbidden": [{"column": c} for c in forbidden], "target": "y"}
        return found_time_column(suggestion or {}, proposal, self.profile, self.frame)

    def test_a_date_column_or_a_counter_that_only_goes_up_orders_the_split(self):
        self.assertEqual(self.found(["created"]), "created")
        self.assertEqual(self.found(["elapsed"]), "elapsed")
        self.assertEqual(self.found(["created"], {"time_column": "known"}), "known")

    def test_a_duration_is_a_feature_not_an_order(self):
        self.assertIsNone(self.found(["wait_time_min"]))

    def test_a_time_flagged_as_a_leak_never_orders_the_split(self):
        self.assertIsNone(self.found(["chargeback_at"], forbidden=["chargeback_at"]))
        self.frame["resolved_at"] = self.frame["created"]
        self.profile["columns"].append({"name": "resolved_at", "kind": "datetime"})
        self.assertIsNone(self.found(["resolved_at"], forbidden=["resolved_at"]))  # named as written after the outcome
        self.assertEqual(self.found(["created"], forbidden=["created"]), "created")  # an event date flagged for its drift
        self.assertEqual(self.found(["elapsed"], forbidden=["elapsed"]), "elapsed")  # a counter the rows follow: still the order

    def test_a_date_with_gaps_never_orders_the_split(self):
        self.frame["shipped"] = self.frame["created"].where(self.frame.index % 2 == 0)
        self.profile["columns"].append({"name": "shipped", "kind": "datetime"})
        self.assertIsNone(self.found(["shipped"]))

    def test_two_columns_are_left_to_a_person(self):
        self.assertIsNone(self.found(["created", "elapsed"]))
        self.assertIsNone(self.found([]))


class NewestPerTableTest(unittest.TestCase):
    def test_one_solve_per_table_and_solver_the_newest(self):
        from types import SimpleNamespace

        from dclab_rnd.store.sets.llm import newest_per_table

        def t(i, answerer, when):
            return SimpleNamespace(id=i, problem={"data_sha256": "sha", "answerer": answerer}, submission={"completed_at": when})

        old, new, claude = t("a", "rules", 1.0), t("b", "rules", 2.0), t("c", "claude:x", 1.5)
        kept = {x.id: newest for _f, x, newest in newest_per_table([("studio", old), ("studio", new), ("silver", claude)])}
        self.assertEqual(kept, {"a": False, "b": True, "c": True})  # the rules' older solve goes; Claude's own solve stays


class PlantedTaskTest(unittest.TestCase):
    def test_the_truth_names_the_task_when_the_solution_states_it(self):
        from unittest import mock

        from dclab_rnd.twin import compare

        known = {"target": "kind", "task": "multiclass", "forbidden": [{"column": "later"}], "time_column": None}
        with mock.patch("dclab_rnd.studio.data.load_sample", return_value=(pd.DataFrame(), known)):
            truth = compare.planted("sample", "invented")["solution.columns"]
        self.assertEqual(truth, {"target": "kind", "leakage": ["later"], "task": "multiclass"})
        with mock.patch("dclab_rnd.studio.data.load_sample", return_value=(pd.DataFrame(), {**known, "task": None})):
            self.assertNotIn("task", compare.planted("sample", "invented")["solution.columns"])


if __name__ == "__main__":
    unittest.main()
