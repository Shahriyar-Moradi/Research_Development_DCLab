"""36.4 in the store: an exam's rows never reach a training set, and a changed or wrongly pinned exam is refused."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import exam as core  # noqa: E402
from dclab_rnd.store import exams  # noqa: E402
from dclab_rnd.store.layout import Layout  # noqa: E402

AT = "2026-10-28T09:00:00+00:00"


def example(n):
    return {"id": f"x{n}", "model_type": "m1_students", "split": "sealed", "point": "column.is_identifier",
            "state": {"unique_share": "high", "n": n}, "label": "yes", "provenance": {"entry": f"entry{n}", "line": n}}


def exam_of(rows):
    items = [core.ExamItem(id=exams.row_key(r), example=r["id"], set="m1_students/column.is_identifier/v1",
                           sha256=__import__("hashlib").sha256(json.dumps(r, sort_keys=True).encode()).hexdigest(),
                           content=exams.content_hash(r), group=f"g{i % 3}", check="label", expect="yes", expect_source="label:person")
             for i, r in enumerate(rows)]
    return core.build("m1_students", "exam_v1", "drawn once", [core.ExamSection(name="column.is_identifier", items=items)], AT)


class GuardTest(unittest.TestCase):
    def test_an_exam_row_is_held_out_by_its_key_or_its_content(self):
        rows = [example(n) for n in range(3)]
        guard = exams.ExamGuard([exam_of(rows)])
        self.assertTrue(guard.why(rows[0]))
        copy = {**rows[1], "id": "another-id", "split": "train", "provenance": {"entry": "elsewhere", "line": 9}}
        self.assertTrue(guard.why(copy))  # the same row kept again under another id: its content gives it away
        self.assertIsNone(guard.why(example(99)))
        self.assertFalse(exams.ExamGuard([]))


class LoadTest(unittest.TestCase):
    def setUp(self):
        self.layout = Layout(Path(tempfile.mkdtemp()))
        self.exam = exam_of([example(n) for n in range(3)])
        path = exams.exam_file(self.layout, "m1_students")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.exam.model_dump_json(), encoding="utf-8")
        self.pins = Path(tempfile.mkdtemp()) / "pins.json"

    def test_the_pinned_exam_loads(self):
        self.pins.write_text(json.dumps({"m1_students": {"exam_v1": self.exam.hash}}))
        self.assertEqual(exams.load(self.layout, "m1_students", pins=self.pins), self.exam)

    def test_an_exam_that_is_not_its_pin_is_refused(self):
        self.pins.write_text(json.dumps({"m1_students": {"exam_v1": "0" * 64}}))
        with self.assertRaises(exams.ExamError):
            exams.load(self.layout, "m1_students", pins=self.pins)

    def test_an_exam_changed_after_it_was_drawn_is_refused(self):
        path = exams.exam_file(self.layout, "m1_students")
        doc = json.loads(path.read_text())
        doc["sections"][0]["items"][0]["expect"] = "no"
        path.write_text(json.dumps(doc))
        with self.assertRaises(exams.ExamError):
            exams.load(self.layout, "m1_students", pins=self.pins)

    def test_a_pinned_exam_whose_file_is_gone_is_refused(self):
        self.pins.write_text(json.dumps({"m1_students": {"exam_v1": self.exam.hash}}))
        exams.exam_file(self.layout, "m1_students").unlink()
        with self.assertRaises(exams.ExamError):
            exams.load(self.layout, "m1_students", pins=self.pins)


def decision(n, table, label="yes", split="sealed"):
    return {"schema_version": "dclab_core.decision_example.v2", "id": f"m1_students:entry{n}#{n}", "model_type": "m1_students",
            "split": split, "point": "column.is_identifier", "group": table, "split_group": table,
            "state": {"unique_share": "high", "n": n}, "label": label, "label_source": "person",
            "provenance": {"entry": f"entry{n}", "workspace": "w"}}


class FlowTest(unittest.TestCase):
    """The review's scenarios on a tiny store: draw, sit, accept, refuse a version that forgot."""

    def setUp(self):
        self.layout = Layout(Path(tempfile.mkdtemp()))
        self.pins = Path(tempfile.mkdtemp()) / "pins.json"
        sealed = [decision(n, f"t{n % 12}", "yes" if n % 2 else "no") for n in range(40)]
        folder = self.layout.sets("m1_students", "column.is_identifier", "v1")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "sealed.jsonl").write_text("\n".join(json.dumps(r) for r in sealed) + "\n")
        copy = {**sealed[0], "id": "m1_students:other#9", "split": "train", "provenance": {"entry": "other", "workspace": "w"}}
        (folder / "train.jsonl").write_text(json.dumps(copy) + "\n")  # one sealed row's content also in train
        self.sealed = sealed

    def test_a_row_also_in_train_is_never_drawn_and_the_exam_is_clean(self):
        exam = exams.draw(self.layout, "m1_students", pins=self.pins)
        self.assertEqual(len(exam.items()), 39)  # the row whose content is in train is left out
        self.assertTrue(exams.verify(self.layout, exam)["clean"])
        self.assertEqual(json.loads(self.pins.read_text())["m1_students"]["exam_v1"], exam.hash)

    def test_a_version_with_scrambled_answers_is_refused_and_the_skill_named(self):
        exam = exams.draw(self.layout, "m1_students", pins=self.pins)
        right = {i.id: i.expect for i in exam.items()}
        exams.sit(self.layout, exam, right, "v1")
        from unittest import mock

        with mock.patch.object(exams, "load", lambda layout, mt, version=exams.EXAM_VERSION, pins=None: exam):
            self.assertTrue(exams.accept(self.layout, "m1_students", "v1", "first")["pass"])
            scrambled = {i.id: ("no" if i.expect == "yes" else "yes") for i in exam.items()}
            exams.sit(self.layout, exam, scrambled, "v2")
            with self.assertRaises(exams.ExamError):
                exams.accept(self.layout, "m1_students", "v2", "scrambled")
            self.assertEqual(exams.current(self.layout, "m1_students"), "v1")
        with self.assertRaises(exams.ExamError):
            exams.sit(self.layout, exam, right, "v1")  # a version sits an exam once

    def test_a_model_version_with_any_name_can_sit(self):
        exam = exams.draw(self.layout, "m1_students", pins=self.pins)
        res = exams.sit(self.layout, exam, {}, "dclab-tuned:v2 Qwen3.5-4B")
        self.assertEqual(res.model_version, "dclab-tuned:v2 Qwen3.5-4B")

    def test_a_trajectory_row_key_is_its_whole_id(self):
        a = {"id": "m5_solver_llm:dev:abc", "model_type": "m5_solver_llm", "provenance": {"entry": "dev"}}
        b = {"id": "m5_solver_llm:dev:def", "model_type": "m5_solver_llm", "provenance": {"entry": "dev"}}
        self.assertNotEqual(exams.row_key(a), exams.row_key(b))


if __name__ == "__main__":
    unittest.main()
