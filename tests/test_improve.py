"""Improving a finished model, practically (the owner, 2026-10-20): ways tried on the training folds only, each paired with the
current model on the same folds; the held-back cases untouched; a way that cleared can be adopted for the next final run."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd

    from dclab_rnd.studio import ProjectStore, data, engine, improve
except ImportError as error:
    raise unittest.SkipTest(str(error))


def build(store):
    """A small invented project taken through its five stages."""
    rng = np.random.default_rng(11)
    n = 700
    frame = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n), "segment": rng.choice(["x", "y", "z"], n)})
    frame["outcome"] = (frame["a"] + 0.5 * frame["b"] + rng.normal(scale=1.0, size=n) > 0.9).astype(int)
    pid = store.create("Improve", "general", "Predict the outcome")["id"]
    frame.to_parquet(store.data_dir(pid) / "t.parquet", index=False)
    data.attach_data(store, pid, "t.parquet")
    project = store.get(pid)
    project["solution"] = {"target": "outcome", "task": "binary", "positive_label": "1", "prediction_moment": "At the time of the visit.", "forbidden": [],
                           "identifiers": [], "time_column": None, "group_column": None, "text_columns": [], "metric": "roc_auc", "notes": ""}
    project["settings"]["quick"] = True
    store.save(project)
    for stage in ("data", "leakage", "features", "models", "final"):
        engine.execute(store, pid, stage, "human")
    return pid


class ImproveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = ProjectStore(Path(tempfile.mkdtemp()) / "projects")
        cls.pid = build(cls.store)

    def test_the_ways_are_tried_on_the_training_folds_paired_and_the_holdout_is_untouched(self):
        before = self.store.get(self.pid)
        record = improve.run(self.store, self.pid, budget_seconds=240)
        self.assertGreaterEqual(len(record["candidates"]), 2)
        folds = record["folds"]
        self.assertEqual(len(record["current"]["fold_scores"]), folds)
        for c in record["candidates"]:
            self.assertEqual(len(c["fold_scores"]), folds)  # every way scored on the same folds as the current model
            self.assertIn("why", c)
        kinds = {c["aspect"] for c in record["candidates"]}
        self.assertTrue({"Other model families"} <= kinds or {"Class balance"} <= kinds, kinds)
        after = self.store.get(self.pid)
        self.assertEqual(after["holdout_uses"], before["holdout_uses"])  # the held-back cases were not read again
        self.assertEqual(after["improvements"]["ran_at"], record["ran_at"])
        self.assertIn("optimistic", record["note"])
        self.assertIsNone(after.get("running"))
        winners = [c for c in record["candidates"] if c["clears"] and c["adoptable"]]
        if winners:
            adopted = improve.adopt_candidate(self.store, self.pid, winners[0]["id"])
            self.assertEqual(self.store.get(self.pid)["adopted_improvement"]["id"], winners[0]["id"])
            self.assertEqual(adopted["family"], record["current"]["family"])
        losers = [c for c in record["candidates"] if not c["clears"]]
        if losers:
            with self.assertRaises(ValueError):
                improve.adopt_candidate(self.store, self.pid, losers[0]["id"])  # a way inside the noise is never offered
        with self.assertRaises(KeyError):
            improve.adopt_candidate(self.store, self.pid, "not-a-way")

    def test_the_job_reports_in_the_projects_chat_with_the_ways_and_what_cleared(self):
        import os
        from unittest import mock

        from dclab_rnd.agentic.services import Services
        from dclab_rnd.jobs import handlers
        from dclab_rnd.settings import Settings

        home = Path(tempfile.mkdtemp())
        with mock.patch.dict(os.environ, {"DCLAB_AGENT_HOME": str(home), "DCLAB_DATABASE_URL": "", "DCLAB_NO_LIVE_MODELS": "1"}):
            s = Services(Settings.load())
            pid = build(s.projects)
            draft = s.drafts.create("Predict the outcome")["id"]
            project = s.projects.get(pid)
            project["draft"] = {"id": draft}
            s.projects.save(project)
            handlers._improve_run(s, {"project_id": pid})
            last = s.drafts.get(draft)["messages"][-1]
        self.assertEqual(last["kind"], "improvement_report")
        report = last["report"]
        self.assertEqual(report["project"], pid)
        self.assertTrue(report["candidates"] and report["aspects"] and "optimistic" in report["note"])
        self.assertIn("training folds only", last["text"])
        self.assertIsNone(s.projects.get(pid).get("running"))  # the project is free again

    def test_a_project_without_a_final_stage_cannot_be_improved(self):
        fresh = self.store.create("No models", "general", "x")["id"]
        with self.assertRaises(ValueError):
            improve.run(self.store, fresh)


if __name__ == "__main__":
    unittest.main()
