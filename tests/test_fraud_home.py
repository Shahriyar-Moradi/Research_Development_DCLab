"""Package 15.6: the fraud pack from Home to the engine. A draft of the delivery-fraud replica, started with "find
fraudulent rides of bikers" and given both files, recognises the label file, asks the pack's questions (the entity, the
label coverage, the review budget) within MAX_QUESTIONS, and builds a project that runs with the pack's defaults."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core.replicas import delivery_fraud as dfr
    from dclab_rnd.draft import pipeline
    from dclab_rnd.draft.api import build_project
    from dclab_rnd.draft.chat import MAX_QUESTIONS, HomeAgent
    from dclab_rnd.draft.store import DraftStore
    from dclab_rnd.expansion import runner
    from dclab_rnd.studio import ProjectStore, engine
    from dclab_rnd.studio.solution import Solution
except ImportError as error:  # pydantic and pandas are not in every environment
    MISSING = str(error)
else:
    MISSING = ""

ANSWERS = {"prediction_moment": "After delivery, in a daily audit of the day's rides; no label is known when a ride is scored.",
           "entity": "The biker", "labels": "A sample of the cases we found", "budget": "About 80 a day",
           "target": "Fraudulent rides"}


@unittest.skipIf(MISSING, MISSING)
class FraudHomeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replica = dfr.generate(seed=0)

    def draft_with_both_files(self, agent_client=None):
        store = DraftStore(Path(tempfile.mkdtemp()))
        agent = HomeAgent(store, None)
        draft = store.create("Find fraudulent rides of bikers")
        agent.start(draft["id"])
        paths = dfr.save(self.replica, Path(tempfile.mkdtemp()))
        for key, name in (("rides", dfr.RIDE_FILE), ("sample", dfr.SAMPLE_FILE)):
            (store.data_dir(draft["id"]) / name).write_bytes(paths[key].read_bytes())
            asset = pipeline.new_asset(store, draft["id"], "upload", name, name)
            pipeline.run(store, draft["id"], asset["id"], agent=agent)
        return store, agent, draft["id"]

    def answer_all(self, store, agent, draft_id):
        for _ in range(MAX_QUESTIONS + 2):
            pending = agent.pending(store.get(draft_id))
            if pending is None:
                break
            agent.reply(draft_id, ANSWERS.get(pending["field"], "Upload a sample"))

    def test_home_asks_the_packs_questions_and_the_project_runs_with_its_defaults(self):
        store, agent, draft_id = self.draft_with_both_files()
        draft = store.get(draft_id)
        self.assertEqual(draft["pack"]["key"], "imbalanced")
        labels = next(a for a in draft["assets"] if a.get("kind") == "labels")
        self.assertEqual(labels["label_review"]["key"], ["order_id"])
        self.assertTrue(any("label file" in m.get("text", "") for m in draft["messages"]))
        self.answer_all(store, agent, draft_id)
        draft = store.get(draft_id)
        asked = [q["field"] for q in draft["questions"] if not q.get("superseded")]
        self.assertLessEqual(len(asked), MAX_QUESTIONS)
        self.assertTrue({"entity", "labels", "budget"} <= set(asked), asked)
        standing = [q["field"] for q in draft["questions"] if not q.get("superseded")]
        self.assertNotIn("target", standing)  # asked before the label file came, then withdrawn: the label file is the target

        projects = ProjectStore(Path(tempfile.mkdtemp()))
        project = build_project(store, projects, draft_id)
        self.assertEqual(project["pack"], "imbalanced")
        self.assertEqual(project["labels"]["key"], ["order_id"])
        suggestion = project["suggestion"]
        self.assertEqual((suggestion["group_column"], suggestion["labels"]), ("biker_id", "partial"))
        self.assertGreater(suggestion["alert_budget"], 0)
        self.assertLess(suggestion["alert_budget"], 0.05)  # about 80 a day over the table's days, as a share of its rows

        project["solution"] = Solution(**{**suggestion, "prediction_moment": ANSWERS["prediction_moment"]}).model_dump()
        project["settings"]["quick"] = True
        projects.save(project)
        with runner.fast_profile():
            engine.run_all(projects, project["id"])
        models = projects.read_stage(project["id"], "models")["evidence"]
        self.assertIn("Rare event", models["selection_rule"])  # the pack's defaults reached the engine
        self.assertTrue({"isolation_forest", "pu_bagging"} <= {r["model"] for r in models["model_results"]})
        final = projects.read_stage(project["id"], "final")["evidence"]
        self.assertIn("recall_at_budget", final["holdout_metrics"])
        self.assertIn("lower bound", final["precision_note"])

    def test_a_model_may_ask_and_record_the_packs_fields(self):
        from tests.test_draft_flow import ScriptedClient

        store, agent, draft_id = self.draft_with_both_files()
        pending = agent.pending(store.get(draft_id))
        client = ScriptedClient([("record", {"field": pending["field"], "value": ANSWERS[pending["field"]]}),
                                 ("ask_user", {"field": "budget", "question": "How many alerts can your team review each day?",
                                               "options": ["About 50", "About 200"]})])
        HomeAgent(store, client).reply(draft_id, ANSWERS[pending["field"]])
        draft = store.get(draft_id)
        self.assertEqual(draft["understanding"][pending["field"]], ANSWERS[pending["field"]])
        self.assertEqual(draft["questions"][-1]["field"], "budget")


if __name__ == "__main__":
    unittest.main()
