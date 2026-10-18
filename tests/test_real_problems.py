"""Phase 8 (30.2 to 30.4): the real problems' runner: a private problem never runs with a model, a run needs its files, a
live run needs a cap, and the result is stored with the earlier solution beside it."""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import real_problems as rp  # noqa: E402


class RunnerTests(unittest.TestCase):
    def test_the_owners_fraud_data_is_private_and_never_runs_with_a_model(self):
        self.assertTrue(rp.PROBLEMS["fraud"]["private"])
        with mock.patch.object(Path, "is_file", return_value=True), self.assertRaises(SystemExit) as caught:
            rp.run("fraud", live=True)
        self.assertIn("private", str(caught.exception))

    def test_a_run_needs_its_files_and_a_live_run_its_cap(self):
        with mock.patch.dict(rp.PROBLEMS, {"x": {"private": False, "files": [ROOT / "nope.csv"], "words": "w", "answers": {}, "earlier": {}}}):
            with self.assertRaises(SystemExit) as caught:
                rp.run("x")
        self.assertIn("nope.csv", str(caught.exception))
        self.assertEqual(rp.main(["telco", "--live"]), 2)  # no cap: refused before anything runs

    def test_every_problem_names_its_earlier_solution_and_its_protocol(self):
        for name, problem in rp.PROBLEMS.items():
            self.assertTrue(problem["earlier"]["numbers"] and problem["earlier"]["protocol"], name)


if __name__ == "__main__":
    unittest.main()
