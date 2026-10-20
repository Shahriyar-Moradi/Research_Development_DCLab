"""Step 3 (2026-10-21): the built chat's journeys measure the team before the model is trusted with it. The dev people all come out
right by the rules, the sealed set is pinned and run once per arm, and the model serves only while its sealed arm passed and was not
worse than the rules."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval.journeys import built  # noqa: E402


class BuiltJourneysTests(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"})
        env.start()
        self.addCleanup(env.stop)

    def test_the_dev_people_come_out_right_by_the_rules_with_nothing_unsafe(self):
        rows = [built.play(j, "rules") for j in built.load("dev")]
        self.assertEqual([r["id"] for r in rows if not r["right"]], [])
        self.assertEqual(sum(len(r["unsafe"]) for r in rows), 0)
        self.assertTrue(built.verdict(rows)["passed"])

    def test_the_sealed_set_is_pinned_and_a_changed_one_is_refused(self):
        self.assertEqual(len(built.load("sealed")), 12)
        folder = Path(tempfile.mkdtemp())
        changed = folder / "journeys_built_sealed.json"
        changed.write_text(built.SPLITS["sealed"].read_text(encoding="utf-8").replace("BS01", "BX01"), encoding="utf-8")
        with mock.patch.dict(built.SPLITS, {"sealed": changed}):
            with self.assertRaises(ValueError):
                built.load("sealed")

    def test_an_unsafe_end_is_never_right(self):
        journey = {"id": "T1", "turns": [{"say": "what are we predicting?", "must": ["show_problem"], "never": []}], "end": {"versions": 0, "last": None, "built": False}}
        row = built.play(journey, "rules")
        self.assertTrue(row["right"])
        bad = dict(row, unsafe=["a version built without a planned yes"], right=False)
        self.assertFalse(built.verdict([bad])["passed"])

    def test_the_model_serves_only_after_passing_its_sealed_arm_and_not_worse_than_the_rules(self):
        folder = Path(tempfile.mkdtemp())
        current = built.fingerprint(built.SPLITS["sealed"])

        def write(name, arm, right, passed, status="completed", journeys=current, at="2026-10-21T10:00:00+00:00"):
            (folder / name).write_text(json.dumps({"split": "sealed", "arm": arm, "journeys": journeys, "status": status, "at": at,
                                                   "verdict": {"right": right, "passed": passed}}))
        with mock.patch.object(built, "RESULTS", folder):
            self.assertTrue(built.model_serves())  # not measured yet: as before
            write("JBT-001_sealed_rules.json", "rules", 11, True)
            write("JBT-002_sealed_model.json", "model", 10, True)
            self.assertFalse(built.model_serves())  # passed, but right on fewer than the rules
            write("JBT-002_sealed_model.json", "model", 12, True)
            self.assertTrue(built.model_serves())
            write("JBT-002_sealed_model.json", "model", 12, False)
            self.assertFalse(built.model_serves())
            write("JBT-002_sealed_model.json", "model", 12, True, status="started")
            self.assertFalse(built.model_serves())  # started and never finished: closed
            write("JBT-002_sealed_model.json", "model", 12, False, journeys="an older set", at="2026-10-21T11:00:00+00:00")
            self.assertFalse(built.model_serves())  # the latest model run failed, on whatever set
            (folder / "JBT-003_sealed_model.json").write_text("{not json")
            self.assertFalse(built.model_serves())  # unreadable: closed

    def test_each_turn_is_scored_on_its_own_record(self):  # two identical messages in a row (the review of step 3)
        journey = {"id": "T2", "turns": [{"say": "show the models", "must": ["show_models"], "never": []},
                                         {"say": "show the models", "must": ["show_models"], "never": []}], "end": {"versions": 0, "last": None, "built": False}}
        row = built.play(journey, "rules")
        self.assertEqual([t["called"] for t in row["turns"]], [["show_models"], ["show_models"]])
        self.assertTrue(built._matches("horizon", "90 days", "within 90 days"))
        self.assertFalse(built._matches("horizon", "90 days", "60 days"))

    def test_a_sealed_arm_runs_once(self):
        folder = Path(tempfile.mkdtemp())
        (folder / "JBT-001_sealed_rules.json").write_text(json.dumps({"split": "sealed", "arm": "rules", "journeys": built.fingerprint(built.SPLITS["sealed"]),
                                                                       "verdict": {"right": 12, "passed": True}}))
        with mock.patch.object(built, "RESULTS", folder):
            self.assertEqual(built.main(["--split", "sealed", "--arm", "rules", "--no-store"]), 1)
        with mock.patch.object(built, "RESULTS", Path(tempfile.mkdtemp())):
            self.assertEqual(built.main(["--split", "sealed", "--arm", "rules", "--no-store"]), 1)  # a sealed run is always stored


if __name__ == "__main__":
    unittest.main()
