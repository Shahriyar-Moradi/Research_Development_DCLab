"""Package 32.14: the tutor.

A point with enough labelled rows gets a student, fitted on the store's set and measured on dev beside the rule; its version
is kept whole (the fitted model, its card, its training snapshot, its measurement) and is never promoted; a second run keeps
nothing new; a point with too few rows says what it lacks; the run is a session of the harness agent "tutor".
"""

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import sklearn  # noqa: F401

    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)


def example(i, point, label, split):
    return {"schema_version": "dclab_core.decision_example.v1", "id": f"m1_students:r{i}", "model_type": "m1_students", "split": split,
            "provenance": {"entry": "e", "sha256": "0" * 64, "source": "s", "workspace": "w", "at": "2026-10-20T10:00:00+00:00", "project": f"p{i % 8}",
                           "consent": "train"},
            "weight": 1.0, "point": point, "group": point, "state": {"uniqueness": "all_unique" if label else "low", "nulls": "none"},
            "options": ["yes", "no"], "answers": [{"by": "rule", "value": i % 3 != 0}], "label": label, "label_source": "person"}


@unittest.skipIf(MISSING, MISSING)
class TutorTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout

        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        for point, n in (("column.is_identifier", 60), ("column.time", 6)):
            base = self.layout.sets("m1_students", point, "v1")
            base.mkdir(parents=True)
            rows = {"train": [], "dev": [], "sealed": []}
            for i in range(n):
                split = "dev" if i % 4 == 0 else "train"
                rows[split].append(json.dumps(example(i, point, i % 3 != 1, split)))
            for split, lines in rows.items():
                (base / f"{split}.jsonl").write_text("".join(x + "\n" for x in lines), encoding="utf-8")
        self.log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        experience.install(self.log)
        self.addCleanup(experience.install, None)

    def test_a_point_with_enough_rows_gets_a_student_kept_whole_and_never_promoted(self):
        from dclab_rnd.store import Store, tutor

        report = tutor.teach(self.layout, run="tutor-1")
        student = report["students"]["column.is_identifier"]
        self.assertFalse(student["promoted"])
        self.assertIn("column.time", report["waiting"])  # six rows: it says what it lacks
        store = Store(self.layout)
        sources = {e.source for e in store.entries("students")}
        prefix = f"students/column.is_identifier/"
        kept = [s for s in sources if s.startswith(prefix)]
        self.assertEqual({s.rsplit("/", 1)[1] for s in kept}, {"model.pkl", "card.json", "training_snapshot.json", "measurement.json", "shadow.json"})
        shadow = json.loads(store.whole("students", "rnd", f"{prefix}shadow.json"))
        self.assertEqual(shadow["level"], "shadow")  # the point's student, in shadow, never served
        model = pickle.loads(store.whole("students", "rnd", next(s for s in kept if s.endswith("model.pkl"))))
        self.assertIsNotNone(model.answer("column.is_identifier", {"uniqueness": "all_unique", "nulls": "none"}, ["yes", "no"]))
        self.assertEqual(student["dev_accuracy"], 1.0)  # the state decides: above the rule
        tutor.teach(self.layout, run="tutor-2")
        self.assertEqual(len(store.entries("students")), len(kept) + 0)  # the same version: nothing new kept
        steps = [e for e in self.log.read() if e.kind == "agent.step" and e.ids.run == "tutor-1"]
        self.assertTrue(any(str(e.payload.get("tool")).startswith("teach:") for e in steps))


if __name__ == "__main__":
    unittest.main()
