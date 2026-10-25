"""tuning.accept's state (2026-10-25): what a tuning student needs to judge a gain on the folds (its size next to the
folds' spread, how many candidates were tried), in fields the ledger accepts, never a table's value."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.ledger.states import tuning_state  # noqa: E402

EVIDENCE = {"model": "Hist Gradient Boosting", "configuration_results": [{"metrics": {"roc_auc": {"mean": 0.8, "std": 0.02}}}]
            + [{"metrics": {"roc_auc": {"mean": 0.81, "std": 0.03}}}] * 6}


class TuningStateTest(unittest.TestCase):
    def test_the_gain_is_read_beside_the_folds_spread_and_the_candidates_tried(self):
        state = tuning_state({"gain": 0.005, "accepted": True}, EVIDENCE, "roc_auc")
        self.assertEqual(state, {"tuning_gain": 0.005, "cv_mean": 0.8, "cv_std": 0.02, "margin": 0.25, "rounds": 6,
                                 "kind": "hist_gradient_boosting", "metric": "roc_auc"})

    def test_what_is_missing_is_left_out_never_guessed(self):
        self.assertEqual(tuning_state({}, {}, None), {})
        state = tuning_state({"gain": 0.01}, {"configuration_results": [{"metrics": {"rmse": {"mean": 3.0, "std": 0.0}}}]}, "rmse")
        self.assertNotIn("margin", state)  # no spread: no ratio
        self.assertEqual(state["rounds"], 0)
        self.assertNotIn("tuning_gain", tuning_state({"gain": True}, {}, None))  # a flag is not a gain
        tiny = tuning_state({"gain": 0.01}, {"configuration_results": [{"metrics": {"rmse": {"mean": 3.0, "std": 1e-16}}}]}, "rmse")
        self.assertNotIn("margin", tiny)  # the review: a spread of 1e-16 made a ratio the ledger refuses, and the record was lost
        self.assertEqual(tuning_state({}, {"model": "x"}, "precision@50")["metric"], "precision_50")  # a code the ledger accepts

    def test_the_ledger_keeps_it(self):
        import os

        from dclab_rnd.ledger import capture, install, open_ledger

        with mock.patch.dict(os.environ, {"DCLAB_DECISION_SHADOW": "0"}):
            book = open_ledger(Path(tempfile.mkdtemp()))
            install(book)
            self.addCleanup(install, None)
            state = {"rows": "large", "columns": 8, **tuning_state({"gain": 0.005}, EVIDENCE, "roc_auc")}
            record = capture._put("tuning.accept", ("project", "p1"), [capture.answer("rule", "rule:tuning_margin", True)], True, "rule",
                                  state=state, options=["yes", "no"], project="p1")
        self.assertIsNotNone(record)  # every field is a state field of its declared type
        self.assertEqual(record.state["margin"], 0.25)


if __name__ == "__main__":
    unittest.main()
