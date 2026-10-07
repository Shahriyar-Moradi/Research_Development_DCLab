"""Package 14.6: the study harness records only enumerated fields, in order, and analyzes them per condition."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_rnd.agent_eval import study  # noqa: E402


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "events.jsonl"

    def rec(self, kind, p, t, **kw):
        return study.record(kind, p, t, self.path, kw.pop("condition", None), **kw)

    def test_a_session_is_started_measured_and_finished(self):
        self.rec("start", "P1", "T1", condition="with")
        self.rec("correction", "P1", "T1", seconds=40)
        self.rec("citation", "P1", "T1", correct=True)
        self.rec("severe_false_alarm", "P1", "T1")
        done = self.rec("finish", "P1", "T1", completed=True)
        self.assertEqual(done["condition"], "with")  # a later event takes the condition of the start: it cannot be changed midway
        self.assertGreaterEqual(done["seconds"], 0)
        kinds = [e["kind"] for e in study.read(self.path)]
        self.assertEqual(kinds, ["start", "correction", "citation", "severe_false_alarm", "finish"])

    def test_nothing_free_is_ever_stored(self):
        self.rec("start", "P1", "T1", condition="with")
        self.rec("correction", "P1", "T1", seconds=12)
        allowed = {"kind", "participant", "task", "at", "condition", "seconds", "correct", "completed"}
        for line in self.path.read_text().splitlines():
            self.assertLessEqual(set(json.loads(line)), allowed)
        # an unknown payload is dropped by the code, and a name or a note has no place to go
        with self.assertRaises(ValueError):
            self.rec("start", "Maria", "T2", condition="with")
        with self.assertRaises(ValueError):
            self.rec("correction", "P1", "T1", seconds="a customer called 4411")
        with self.assertRaises(ValueError):  # free text has no field: it is refused, not silently dropped
            study.record("correction", "P1", "T1", self.path, None, seconds=3, note="income column looked wrong")

    def test_order_and_repeats_are_enforced(self):
        with self.assertRaises(ValueError):
            self.rec("correction", "P1", "T1", seconds=5)  # not started
        self.rec("start", "P1", "T1", condition="with")
        with self.assertRaises(ValueError):
            self.rec("start", "P1", "T1", condition="without")
        self.rec("finish", "P1", "T1", completed=False)
        with self.assertRaises(ValueError):
            self.rec("citation", "P1", "T1", correct=True)  # finished: closed
        with self.assertRaises(ValueError):
            self.rec("finish", "P1", "T2", completed=True)

    def test_the_assistants_events_exist_only_with_the_assistant_and_a_participant_can_be_deleted(self):
        self.rec("start", "P1", "T1", condition="without")
        for kind, kw in (("correction", {"seconds": 5}), ("citation", {"correct": True}), ("severe_false_alarm", {})):
            with self.assertRaises(ValueError):
                self.rec(kind, "P1", "T1", **kw)
        self.rec("start", "P2", "T1", condition="with")
        self.rec("citation", "P2", "T1", correct=True)
        self.rec("finish", "P2", "T1", completed=True)
        self.assertEqual(study.delete_participant("P2", self.path), 3)
        self.assertEqual({e["participant"] for e in study.read(self.path)}, {"P1"})
        self.assertFalse(list(self.path.parent.glob("*.part")))
        self.assertEqual(study.delete_participant("P9", self.path), 0)
        with self.assertRaises(ValueError):
            study.delete_participant("Maria", self.path)

    def test_the_analysis_is_per_condition_and_descriptive(self):
        for p, cond, done in (("P1", "with", True), ("P2", "with", True), ("P3", "without", False), ("P4", "without", True)):
            self.rec("start", p, "T1", condition=cond)
            if cond == "with":
                self.rec("correction", p, "T1", seconds=30)
                self.rec("citation", p, "T1", correct=True)
            self.rec("finish", p, "T1", completed=done)
        report = study.analyze(self.path)
        w, wo = report["by_condition"]["with"], report["by_condition"]["without"]
        self.assertEqual((w["tasks_completed"], wo["tasks_completed"]), ([2, 2], [1, 2]))
        self.assertEqual((w["correction_seconds_total"], w["citations_correct"]), (60, [2, 2]))
        self.assertEqual(wo["citations_correct"], [0, 0])
        self.assertIsNone(wo["citations_correct_ci95"])
        self.assertIn("descriptive", " ".join(report["limitations"]))
        self.assertNotIn("p_value", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
