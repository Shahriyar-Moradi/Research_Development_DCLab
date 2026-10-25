"""model.first's student (S7, 2026-10-25): picks judged by hit, fairness (within the folds' noise) and regret, beside the
fixed order and the commonest winner; trained on training screens, measured on real dev screens, sealed never read."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import first_family as F  # noqa: E402


def screen(did, metric, scores, first="logistic_regression", state=None, task="binary", profile="full"):
    return {"did": did, "metric": metric, "first": first, "task": task, "profile": profile,
            "state": state or {"rows": "small", "task": "binary_classification"},
            "screen": [{"model": m, "mean": mean, "std": std} for m, mean, std in scores]}


class JudgeTest(unittest.TestCase):
    def test_a_pick_is_judged_by_its_gap_in_units_of_noise(self):
        s = screen(1, "roc_auc", [("lightgbm", 0.90, 0.01), ("logistic_regression", 0.89, 0.01), ("extra_trees", 0.80, 0.01)])
        self.assertEqual(F.winner(s), "lightgbm")
        self.assertEqual(F.regret(s, "lightgbm"), 0.0)
        self.assertLess(F.regret(s, "logistic_regression"), 1.0)  # within noise: a fair first try
        self.assertGreater(F.regret(s, "extra_trees"), 5.0)
        self.assertIsNone(F.regret(s, "xgboost"))  # not screened here
        low = screen(2, "mae", [("ridge", 3.0, 0.1), ("lightgbm", 2.0, 0.1)])
        self.assertEqual(F.winner(low), "lightgbm")  # lower is better for an error
        got = F.judge([s, s, low], ["lightgbm", "logistic_regression", "ridge"])
        self.assertEqual((got["hit"], got["fair"]), (round(1 / 3, 4), round(2 / 3, 4)))
        flat = screen(3, "roc_auc", [("a", 0.9, 0.0), ("b", 0.8, 0.0), ("c", 0.9, float("nan"))])
        self.assertEqual(F.regret(flat, "b"), F.REGRET_CAP)  # no spread: a gap is the cap, never infinite
        self.assertEqual(F.regret(flat, "c"), 0.0)  # a tie on the best mean
        self.assertEqual(F.judge([flat], ["xgboost"])["mean_regret"], F.REGRET_CAP)  # an unscored pick counts as the cap
        lead = screen(5, "roc_auc", [("lightgbm", 0.95, 0.01), ("logistic_regression", 0.89, 0.01), ("extra_trees", 0.80, 0.01)])
        self.assertIsNone(F.clear(s))  # ahead by 0.01 with a combined noise of 0.014: not clearly best
        self.assertEqual((F.clear(lead), F.clear(screen(4, "roc_auc", [("a", 0.9, 0.05), ("b", 0.89, 0.05)]))), ("lightgbm", None))

    def test_only_full_profile_screens_are_read_each_table_once(self):
        home = Path(tempfile.mkdtemp())
        (home / "dev").mkdir()
        lines = [screen(1, "roc_auc", [("a", 0.8, 0.01), ("b", 0.7, 0.01)]), screen(1, "roc_auc", [("a", 0.6, 0.01), ("b", 0.7, 0.01)]),
                 screen(2, "roc_auc", [("a", 0.8, 0.01), ("b", 0.7, 0.01)], profile="fast"), screen(3, "roc_auc", [("a", 0.8, 0.01)])]
        (home / "dev" / "screens.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        got = F.screens(home, "dev")
        self.assertEqual([(g["did"], F.winner(g)) for g in got], [(1, "b")])  # the latest line of table 1; no fast one; no single family


class TrainTest(unittest.TestCase):
    def setUp(self):
        self.real, self.craft, self.out = (Path(tempfile.mkdtemp()) for _ in range(3))
        rng = np.random.default_rng(0)

        def write(home, split, n, start):
            (home / split).mkdir(parents=True, exist_ok=True)
            with (home / split / "screens.jsonl").open("w") as fh:
                for i in range(n):
                    wide = int(rng.integers(1, 40))
                    best = "lightgbm" if wide > 10 else "logistic_regression"  # wide tables favour boosting here
                    other = "logistic_regression" if best == "lightgbm" else "lightgbm"
                    fh.write(json.dumps(screen(start + i, "roc_auc", [(best, 0.9, 0.01), (other, 0.8, 0.01)],
                                               state={"rows": "small", "task": "binary_classification", "numeric_columns": wide, "columns": wide})) + "\n")
        write(self.real, "train", 120, 0)
        write(self.real, "dev", 40, 1000)
        write(self.craft, "train", 40, 2000)
        (self.real / "sealed").mkdir()
        (self.real / "sealed" / "screens.jsonl").write_text("this file must never be opened\n")

    def test_the_student_learns_what_the_fixed_order_misses_and_never_opens_the_sealed_screens(self):
        real_open = Path.read_text

        def guarded(path, *a, **k):
            if "sealed" in str(path):
                raise AssertionError("a sealed screen was read")
            return real_open(path, *a, **k)
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}), mock.patch.object(Path, "read_text", guarded):
            got = F.train(self.real, self.craft, self.out)
        self.assertGreater(got["real_only"]["student"]["hit"], got["rule"]["hit"] + 0.2)  # the fixed first family is wrong on wide tables
        self.assertLess(got["real_only"]["student"]["mean_regret"], got["rule"]["mean_regret"])
        self.assertEqual(got["trained_on"], {"real": 120, "craft": 40})
        self.assertIsNotNone(got["with_craft"])  # both arms reported; the real-only one kept
        self.assertTrue((self.out / "model.first.json").is_file() and (self.out / "MEASURED.json").is_file())

    def test_too_little_homework_is_refused(self):
        with self.assertRaises(SystemExit):
            F.train(Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()), self.out)


if __name__ == "__main__":
    unittest.main()
