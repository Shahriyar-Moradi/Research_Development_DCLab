"""A built project's chat answers about the project (the owner, 2026-10-20): asked how to improve the model, it said the plan
again. Now the answer is the project's own records read into advice, each suggestion from a number in them."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.studio import agent as studio_agent  # noqa: E402

RECORDS = {
    "models": {"evidence": {"ranking": [{"model": "logistic_regression", "mean": 0.7624, "std": 0.0356}, {"model": "extra_trees", "mean": 0.7274, "std": 0.0131},
                                         {"model": "hist_gradient_boosting", "mean": 0.7147, "std": 0.0356}]}},
    "features": {"evidence": {"selected_recipe": "log_numeric", "selected_vs_reference_gain": 0.004}},
    "final": {"primary_metric": "roc_auc", "evidence": {"holdout_metrics": {"roc_auc": 0.7235}, "holdout_primary_metric_ci": {"low": 0.6731, "high": 0.7716},
                                                          "cv_selected_metric_mean": 0.764, "cv_to_holdout_gap": -0.0405, "holdout_rows": 1600, "train_rows": 6400,
                                                          "baselines": {"majority_class": {"roc_auc": 0.5}},
                                                          "tuning_decision": {"gain": 0.0017, "accepted": True, "required_gain": 0.001},
                                                          "threshold_analysis": [{"threshold": 0.5}]}},
}


class BuiltChatTests(unittest.TestCase):
    def test_how_to_improve_is_read_from_the_projects_own_numbers(self):
        out = studio_agent.answer("you have built the model how we can improve it", {"id": "p1"}, RECORDS, "binary_imbalanced")
        steps = out["steps"]
        text = " ".join(steps)
        self.assertIn("0.7235", steps[0])  # where it stands, with its interval and the baseline
        self.assertIn("0.6731", steps[0])
        self.assertIn("More cases would help first", text)  # the interval is wide
        self.assertIn("gap", text)  # the folds said more than the holdout
        self.assertIn("extra_trees", text)  # within one fold spread of the winner
        self.assertIn("log_numeric", text)
        self.assertIn("cutoff", text)
        self.assertIn("DCLAB-R22", out["proof"])
        self.assertNotIn("Here is the solution I would build", out["answer"])

    def test_nothing_is_suggested_without_its_evidence(self):
        steps, _ = studio_agent.improvements({"final": {"primary_metric": "roc_auc", "evidence": {"holdout_metrics": {"roc_auc": 0.9}}}})
        self.assertEqual(len(steps), 1)  # only where it stands

    # a message on a built chat (the orchestrator, its follow-ups and its jobs): tests/test_home_orchestrator.py


if __name__ == "__main__":
    unittest.main()
