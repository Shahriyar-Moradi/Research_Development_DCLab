"""DCLab's engine on what real tables hold (2026-10-25: the first real-table screens crashed on two of three tables):
a column with no value at all (OpenML 2), and classes of a single row (OpenML 7)."""

import os
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(frame, target, task):
    from dclab_rnd.kinds import acceptance as A

    return A.engine_run(A.Case(table=frame, target=target, task=task, moment="When a new row arrives, before its outcome is known."))


class RealTablesTest(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "DCLAB_DECISION_SHADOW": "0"})
        env.start()
        self.addCleanup(env.stop)
        self.rng = np.random.default_rng(0)

    def test_a_table_with_nothing_left_to_learn_from_is_refused_clearly(self):  # S8 B3, live: a ranking log run as binary crashed in LightGBM
        n = 600
        frame = pd.DataFrame({"user_id": [f"u{i % 90}" for i in range(n)], "item_id": [f"i{i % 37}" for i in range(n)],
                              "time": pd.date_range("2025-01-01", periods=n, freq="h").astype(str), "event": self.rng.integers(0, 2, n)})
        with self.assertRaisesRegex(ValueError, "No column is left to learn from"):
            from dclab_rnd.kinds import acceptance as A
            from dclab_rnd.studio import engine

            with mock.patch.object(engine, "_register_recipes", side_effect=AssertionError("prepared past the check")):
                A.engine_run(A.Case(table=frame, target="event", task="binary", moment="When a new row arrives, before its outcome is known.",
                                    solution={"identifiers": ["user_id", "item_id"], "time_column": "time"}))

    def test_a_column_with_no_value_at_all_never_stops_a_run(self):
        n = 300
        x = self.rng.normal(size=n)
        wide = {f"x{i}": self.rng.normal(size=n).round(3) for i in range(14)}  # wider than the selected_mi recipe's ten
        frame = pd.DataFrame({"a": x.round(3), **wide, "empty_1": np.nan, "empty_2": np.nan,
                              "kind": self.rng.choice(["p", "q", "r"], n), "y": np.where(x + self.rng.normal(0, 0.5, n) > 0, "yes", "no")})
        done = run(frame, "y", "binary")
        self.assertTrue(done.evidence("models").get("selected_model"))

    def test_classes_of_a_single_row_never_stop_the_split(self):
        from dclab_rnd.expansion.datasets import strata

        n = 240
        labels = np.array(["a"] * 100 + ["b"] * 80 + ["c"] * 56 + ["d", "e", "f", "g"])  # four classes of one row each
        self.rng.shuffle(labels)
        frame = pd.DataFrame({"u": self.rng.normal(size=n).round(3), "v": self.rng.normal(size=n).round(3), "y": labels})
        done = run(frame, "y", "multiclass")
        self.assertTrue(done.evidence("models").get("selected_model"))
        y = pd.Series(["a", "a", "b", "b", "c"])
        self.assertEqual(list(strata(y)), ["a", "a", "b", "b", "a"])  # the single row drawn with the commonest class
        whole = pd.Series([1, 1, 2, 2])
        self.assertIs(strata(whole), whole)  # every class with two rows or more: the split exactly as before

    def test_a_lone_positive_is_refused_clearly_and_many_small_classes_split_at_random(self):  # the review's 2 and the left-over case
        from dclab_rnd.expansion.datasets import strata, strata_note

        lone = pd.Series(["no"] * 199 + ["yes"])
        self.assertIs(strata(lone, 40), lone)  # one class of two rows or more: the split's own error says why, at once
        frame = pd.DataFrame({"u": self.rng.normal(size=200).round(3), "v": self.rng.normal(size=200).round(3), "y": lone.to_numpy()})
        with self.assertRaises(ValueError) as caught:
            run(frame, "y", "binary")
        self.assertIn("least populated class", str(caught.exception))
        many = pd.Series([f"c{i // 2}" for i in range(50)])  # 25 classes of two rows, a holdout of ten
        self.assertIsNone(strata(many, 10))
        self.assertIn("not stratified", strata_note(many, None))
        pooled = pd.Series(["a"] * 6 + ["b"] * 6 + ["c", "d"])
        self.assertIn("2 classes of one row drawn with `a`", strata_note(pooled, strata(pooled, 3)))

    def test_a_text_table_with_an_empty_number_column_runs(self):  # the review's 3: the text recipe's fill
        n = 240
        words = np.array(["late delivery refund", "great service", "broken item refund", "fast and kind", "never again refund"])
        text = self.rng.choice(words, n)
        frame = pd.DataFrame({"note": text, "amount": self.rng.gamma(2, 20, n).round(2), "empty": np.nan,
                              "y": np.where(np.char.find(text.astype(str), "refund") >= 0, "yes", "no")})
        done = run(frame, "y", "binary")
        self.assertTrue(done.evidence("models").get("selected_model"))

    def test_the_notebooks_split_cell_draws_the_engines_strata(self):  # the review's 1
        from sklearn.model_selection import train_test_split

        from dclab_rnd.expansion.datasets import strata
        from dclab_rnd.studio import export

        code = export._split_code(None, None, False, {}, 3)
        y = pd.Series(list("aaaaabbbbbcccccd"))  # one class of a single row
        X = pd.DataFrame({"x": range(len(y))})
        space = {"np": np, "pd": pd, "X": X, "y": y, "RANDOM_STATE": 42}
        exec(code, space)
        engine = train_test_split(np.arange(len(y)), test_size=0.2, stratify=strata(y, int(np.ceil(0.2 * len(y)))), random_state=42)
        self.assertEqual(sorted(space["X_test"]["x"]), sorted(engine[1]))


if __name__ == "__main__":
    unittest.main()
