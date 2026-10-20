"""A built project's chat answers about the project (the owner, 2026-10-20): asked how to improve the model, it said the plan
again. Now the answer is the project's own records read into advice, each suggestion from a number in them."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402
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

    def test_a_message_on_a_built_chat_is_answered_about_the_project_not_by_the_plan(self):
        store = DraftStore(Path(tempfile.mkdtemp()))
        draft = store.create("Which leads convert this month?")["id"]
        store.update(draft, lambda d: d.update(status="built", project_id="p1"))
        agent = HomeAgent(store, None)
        agent.reply_built(draft, "how can we improve it", lambda q: studio_agent.answer(q, {"id": "p1"}, RECORDS, "binary_imbalanced"))
        agent.reply_built(draft, "build it", lambda q: self.fail("a word to build asks nothing of the project"))
        messages = store.get(draft)["messages"]
        kinds = [(m["role"], m["kind"]) for m in messages[-4:]]
        self.assertEqual(kinds, [("user", "text"), ("agent", "project_answer"), ("user", "text"), ("agent", "text")])
        answer = messages[-3]
        self.assertTrue(answer["lines"] and "0.7235" in answer["lines"][0])
        self.assertIn("already built", messages[-1]["text"])

    def test_asked_to_improve_the_model_for_me_it_does_so_and_asked_how_it_only_advises(self):
        store = DraftStore(Path(tempfile.mkdtemp()))
        draft = store.create("Which leads convert this month?")["id"]
        store.update(draft, lambda d: d.update(status="built", project_id="p1"))
        agent = HomeAgent(store, None)
        started = []
        answer = lambda q: studio_agent.answer(q, {"id": "p1"}, RECORDS, "binary_imbalanced")  # noqa: E731
        for text in ("ok improve the model for me", "please improve it", "improve it now"):
            agent.reply_built(draft, text, answer, lambda: started.append(1) or {"job": {"id": "j1"}})
        self.assertEqual(len(started), 3)  # it does the work
        for text in ("how can we improve it", "what should we improve?", "which aspects can improve"):
            agent.reply_built(draft, text, answer, lambda: started.append(1) or {"job": {"id": "j1"}})
        self.assertEqual(len(started), 3)  # a question gets advice, not a run
        last = store.get(draft)["messages"][-1]
        self.assertEqual(last["kind"], "project_answer")
        self.assertIn("improve the model for me", last["lines"][-1])  # and says how to ask for the doing
        busy = lambda: (_ for _ in ()).throw(ValueError("Something is running in this project: wait for it, or stop it."))  # noqa: E731
        agent.reply_built(draft, "improve the model for me", answer, busy)
        self.assertIn("Something is running", store.get(draft)["messages"][-1]["text"])


if __name__ == "__main__":
    unittest.main()
