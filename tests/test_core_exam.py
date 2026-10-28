"""36.4: a frozen exam per model type and the gate that refuses a version which lost a skill (dclab_core.exam)."""

import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from dclab_core import exam as ex  # noqa: E402
from dclab_core.contracts import CONTRACTS  # noqa: E402

AT = "2026-10-28T09:00:00+00:00"


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def item(n, section):
    return ex.ExamItem(id=f"{section}:{n}", example=f"e{section}{n}", set=f"m1_students/{section}/v1", sha256=h(f"row{section}{n}"),
                       content=h(f"content{section}{n}"), group=f"table{n % 8}", check="label", expect="yes", expect_source="label:person")


def exam(sizes=(40, 40)):
    sections = [ex.ExamSection(name=f"skill_{k}", items=[item(n, k) for n in range(size)]) for k, size in enumerate(sizes)]
    return ex.build("m1_students", "exam_v1", "drawn once from the sealed rows", sections, AT)


def scores(e, right=1.0, wrong_section=None):
    return {i.id: (0.0 if wrong_section is not None and i.id.startswith(f"{wrong_section}:") else right) for i in e.items()}


class ExamTest(unittest.TestCase):
    def test_the_hash_pins_the_exam(self):
        e = exam()
        self.assertEqual(ex.Exam.read(e.model_dump_json()), e)
        changed = e.model_dump()
        changed["sections"][0]["items"][0]["expect"] = "no"
        with self.assertRaises(ValidationError):
            ex.Exam.model_validate(changed)  # changed after it was drawn
        self.assertIs(CONTRACTS["exam"], ex.Exam)
        self.assertIs(CONTRACTS["exam_result"], ex.ExamResult)

    def test_a_row_is_in_one_section_only(self):
        a = ex.ExamSection(name="a", items=[item(1, 0)])
        b = ex.ExamSection(name="b", items=[item(1, 0)])
        with self.assertRaises(ValidationError):
            ex.build("m1_students", "exam_v1", "rule", [a, b], AT)

    def test_a_result_answers_this_exam_and_no_other(self):
        e = exam()
        with self.assertRaises(ValueError):
            ex.score(e, {"not:an:item": 1.0}, "v2", AT)


class GateTest(unittest.TestCase):
    def test_a_copy_of_the_previous_version_passes(self):
        e = exam()
        before = ex.score(e, scores(e), "v1", AT)
        verdict = ex.no_forgetting(e, before, ex.score(e, scores(e), "v2", AT))
        self.assertTrue(verdict["pass"])
        self.assertEqual(verdict["forgotten"], [])

    def test_a_version_that_lost_one_skill_is_refused_and_the_skill_named(self):
        e = exam()
        before = ex.score(e, scores(e), "v1", AT)
        verdict = ex.no_forgetting(e, before, ex.score(e, scores(e, wrong_section=1), "v2", AT))
        self.assertFalse(verdict["pass"])
        self.assertEqual(verdict["forgotten"], ["skill_1"])

    def test_the_first_version_has_nothing_to_lose(self):
        e = exam()
        verdict = ex.no_forgetting(e, None, ex.score(e, scores(e), "v1", AT))
        self.assertTrue(verdict["pass"])
        self.assertIn("first", verdict["note"])

    def test_a_small_section_is_named_weakly_protected(self):
        e = exam(sizes=(40, 5))
        before = ex.score(e, scores(e), "v1", AT)
        verdict = ex.no_forgetting(e, before, ex.score(e, scores(e), "v2", AT))
        self.assertIn("skill_1", verdict["weakly_protected"])

    def test_a_result_of_another_exam_is_refused(self):
        e, other = exam(), exam(sizes=(40, 41))
        verdict = ex.no_forgetting(e, ex.score(e, scores(e), "v1", AT), ex.score(other, scores(other), "v2", AT))
        self.assertFalse(verdict["pass"])
        self.assertIn("not a result of this exam", verdict["refused"])


    def test_a_section_of_two_groups_cannot_hide_a_lost_half(self):
        """The review's case: 100 items in 2 groups, one group lost; a bootstrap over 2 groups could not bound it."""
        items = [ex.ExamItem(id=f"s:{n}", example=f"e{n}", set="m1_students/s/v1", sha256=h(f"r{n}"), content=h(f"c{n}"),
                             group="g1" if n < 50 else "g2", check="label", expect="yes", expect_source="label:person") for n in range(100)]
        e = ex.build("m1_students", "exam_v1", "rule", [ex.ExamSection(name="two_groups", items=items)], AT)
        before = ex.score(e, {i.id: 1.0 for i in items}, "v1", AT)
        after = ex.score(e, {i.id: (0.0 if i.group == "g1" else 1.0) for i in items}, "v2", AT)
        verdict = ex.no_forgetting(e, before, after)
        self.assertFalse(verdict["pass"])
        self.assertEqual(verdict["forgotten"], ["two_groups"])


if __name__ == "__main__":
    unittest.main()
