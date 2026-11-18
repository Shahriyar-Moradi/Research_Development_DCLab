"""The owner, 2026-11-18: everything Claude wrote is held until the rights review, also the written people and requests
the intent classifier, the understanding students and the workflow models learn from. Each refuses before it fits or
writes anything; the decision export skips the journey rows and counts them."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HOLD = "dclab_rnd.store.provider_outputs.may_train"


def journey_rows(folder, point, source, n=12):
    path = Path(folder) / "decisions" / point / "rows.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"record": f"{source}-{i}", "point": point, "split": "train" if i % 3 else "dev", "group": f"{source}:{i}", "source": source,
             "state": {"turns": "few"}, "options": ["yes", "no"], "value_used": "yes", "label": {"value": "yes", "accepted": ["yes"], "source": "benchmark"}}
            for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class TrainersHoldTests(unittest.TestCase):
    def test_the_intent_classifier_refuses_requests_claude_wrote(self):
        from dclab_rnd import intent_classifier
        from dclab_rnd.store import writers

        folder = Path(tempfile.mkdtemp())
        requests = folder / "intent_requests_v1.json"
        requests.write_text(json.dumps([{"id": "r1", "text": "average sales per month", "track": "analysis", "writer": "a", "batch": "analysis-a-1",
                                         "model": "claude-sonnet-5-5"}]), encoding="utf-8")
        with mock.patch(HOLD, return_value=False):
            with self.assertRaises(writers.HeldForRights):
                intent_classifier.train(requests=requests, out=folder / "model")
        self.assertFalse((folder / "model").exists())  # nothing fitted, nothing written

    def test_the_workflow_models_refuse_rows_from_written_journeys(self):
        from dclab_rnd import workflow_models
        from dclab_rnd.store import writers

        sets, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()) / "models"
        journey_rows(sets, "problem.ready", "sealed_v2")
        with mock.patch(HOLD, return_value=False):
            with self.assertRaises(writers.HeldForRights):
                workflow_models.train(points=["problem.ready"], sets=sets, out=out)
        self.assertFalse(out.exists())

    def test_the_understanding_students_refuse_rows_from_written_journeys(self):
        from dclab_rnd import understand_students
        from dclab_rnd.store import writers

        sets, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()) / "models"
        journey_rows(sets, "problem.family", "homework_v1")
        with mock.patch(HOLD, return_value=False):
            with self.assertRaises(writers.HeldForRights):
                understand_students.train(sets=sets, out=out)
        self.assertFalse(out.exists())

    def test_the_decision_export_skips_journey_rows_and_counts_them(self):
        from dclab_rnd import decision_export

        root = Path(tempfile.mkdtemp())
        workflow = root / "workflow_v1"
        journey_rows(workflow, "problem.ready", "dev", n=3)
        held: dict = {}
        with mock.patch(HOLD, return_value=False), mock.patch("dclab_core.decision_export.normalize", side_effect=lambda r: r):
            got = decision_export.rows(workflow / "decisions", held)
        self.assertEqual(got, [])
        self.assertEqual(held, {"workflow_v1: written journeys": 3})


if __name__ == "__main__":
    unittest.main()
