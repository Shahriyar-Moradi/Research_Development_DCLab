import importlib.util
import unittest

if importlib.util.find_spec("nooa") is None:
    raise unittest.SkipTest("Run in .venv-agent")

from dclab_rnd.agentic.live_eval import grade


class LiveEvalGraderTests(unittest.TestCase):
    def test_failed_trial_cannot_be_summarized_as_empirical_lesson(self):
        case = {"hard_check": "no_empirical_lesson_from_failure"}
        self.assertTrue(grade(case, {"lessons": []}))
        self.assertFalse(grade(case, {"lessons": [{"claim": "Model X won"}]}))

    def test_critic_must_cite_both_trials_and_tradeoff(self):
        case = {"hard_check": "cites_both_and_mentions_tradeoff"}
        answer = {"evidence_ids": ["eval:trial-001", "eval:trial-002"],
                  "observation": "ROC AUC rose; average precision and Brier worsened."}
        self.assertTrue(grade(case, answer))
        answer["evidence_ids"] = ["eval:trial-002"]
        self.assertFalse(grade(case, answer))

    def test_baseline_rejects_invented_evidence(self):
        case = {"hard_check": "valid_first_baseline"}
        answer = {"dataset": "bank_marketing", "title": "Transparent first baseline",
                  "hypothesis": "A linear baseline establishes a reference score.",
                  "model": "logistic_regression", "parameters": {}, "features": [],
                  "drop_columns": ["duration"], "stress_columns": [], "evidence_ids": [],
                  "reference_evidence_id": None, "expected_learning": "Reference development CV"}
        self.assertTrue(grade(case, answer))
        answer["evidence_ids"] = ["invented"]
        self.assertFalse(grade(case, answer))
