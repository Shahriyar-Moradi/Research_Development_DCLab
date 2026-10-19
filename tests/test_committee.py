"""Package 29.16: a committee. Each member votes with the log-odds of its measured accuracy over chance (a member at
chance never votes); the heaviest option wins; a tie goes to the heaviest member's choice, then to the committee's order;
an abstention gives no vote; another weighting is another version."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class VoteTests(unittest.TestCase):
    def test_weights_follow_measured_accuracy_and_chance_never_votes(self):
        from dclab_core.committee import weight

        self.assertEqual(weight(0.5), 0.0)  # a yes/no member at chance
        self.assertEqual(weight(None), 0.0)
        self.assertGreater(weight(0.9), weight(0.8))
        self.assertGreater(weight(0.4, options=6), 0.0)  # above chance among six options
        self.assertLess(weight(1.0), 10)  # always right on finitely many rows: heavy, never infinite

    def test_the_heaviest_option_wins_and_a_tie_goes_to_the_heaviest_member(self):
        from dclab_core.committee import vote, weight

        w = {"rule": weight(0.8), "student": weight(0.7), "gliner": weight(0.7)}
        value, share, _ = vote({"rule": True, "student": False, "gliner": False}, w, ["rule", "student", "gliner"])
        self.assertIs(value, False)  # two members at 0.7 outweigh one at 0.8
        self.assertGreater(share, 0.5)
        value, _, _ = vote({"rule": True, "student": False, "gliner": None}, {"rule": 1.0, "student": 1.0, "gliner": 1.0}, ["rule", "student", "gliner"])
        self.assertIs(value, True)  # a tie: the committee's order, the rule first
        value, _, _ = vote({"rule": True, "gliner": True, "student": False}, {"rule": 1.0, "gliner": 1.0, "student": 2.0}, ["rule", "student", "gliner"])
        self.assertIs(value, False)  # a real tie (2.0 each): the heaviest member's choice, not the rule's
        self.assertEqual(vote({"rule": None}, {"rule": 2.0}, ["rule"])[0], None)  # no member answered

    def test_a_committee_answers_with_its_given_and_live_members_and_another_weighting_is_another_version(self):
        from dclab_core.answerers import ScriptedAnswerer
        from dclab_core.committee import Committee

        student = ScriptedAnswerer("student", "numeric", ["column.semantic_role"])
        broken = type("Broken", (), {"answer": lambda self, *a: 1 / 0})()
        c = Committee("column.semantic_role", ["rule", "student", "broken"], {"rule": 1.0, "student": 2.0, "broken": 5.0},
                      members={"student": student, "broken": broken}, versions={"student": "1"})
        got = c.answer("column.semantic_role", {}, ["numeric", "identifier"], given={"rule": "identifier"})
        self.assertEqual((got.value, got.by), ("numeric", "classifier"))  # the failing member abstained
        self.assertIsNone(c.answer("column.time", {}, ["yes", "no"]))
        other = Committee("column.semantic_role", ["rule", "student", "broken"], {"rule": 1.0, "student": 2.5, "broken": 5.0})
        self.assertNotEqual(c.version, other.version)


try:
    import pandas  # noqa: F401

    HAVE_ML = True
except ImportError:  # the agent environment
    HAVE_ML = False


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class ExamTests(unittest.TestCase):
    """The weights come from dev; the fresh sealed rows are read once, only when they match their committed fingerprint
    and the student is the one the weights were measured for; every answerer is judged on the same decisions."""

    def test_the_exam_reads_its_sealed_rows_once_and_judges_every_answerer_on_them(self):
        import json
        import tempfile

        import joblib

        from dclab_core.decision_models import train
        from dclab_rnd import committee as K
        from dclab_rnd import decisions_gen as G
        from dclab_rnd.decision_models_build import record_fitted

        tmp = Path(tempfile.mkdtemp())
        point = "column.is_identifier"
        G.build(seeds=2, rows=300, out=tmp / "dev", first=60)
        G.build(seeds=2, rows=300, out=tmp / "fresh", first=70)
        rows = [json.loads(x) for x in (tmp / "dev" / "decisions" / point / "rows.jsonl").read_text().splitlines()]
        model, card = train(point, [r for r in rows if r["split"] == "train"], [], families=("logistic",), strengths=(1.0,))
        (tmp / "models").mkdir()
        (tmp / "models" / f"{point}.json").write_text(card.model_dump_json())
        joblib.dump(model, tmp / "models" / f"{point}.joblib")
        record_fitted(tmp / "models")
        sealed = [json.loads(x) for x in (tmp / "fresh" / "decisions" / point / "sealed.jsonl").read_text().splitlines()]
        folder = tmp / "lake" / "bronze" / "answers" / K.GLINER / "committee" / point
        folder.mkdir(parents=True)
        (folder / "sealed.jsonl").write_text("".join(json.dumps({"sample_id": r["record"], "value": r["label"]["value"]}) + "\n" for r in sealed[::2]))
        plan = K.plan((point,), sets=tmp / "dev", students=tmp / "models", lake=tmp / "lake", gliner={point: {}})
        self.assertEqual(plan["points"][point]["weights"]["gliner"], 0.0)  # never measured on dev: no vote
        kw = {"fresh": tmp / "fresh", "students": tmp / "models", "campaign": tmp / "campaign", "lake": tmp / "lake"}
        doc = json.loads(K.exam(point, plan, **kw).read_text())
        self.assertEqual(doc["plan"], K.plan_hash(plan))  # each result names the plan its weights came from
        self.assertEqual(set(doc["verdicts"]), {"rule", "student", "gliner", "committee"})
        self.assertEqual(doc["sealed"]["decisions"], len(sealed))
        self.assertEqual(doc["gliner_answered"], len(sealed[::2]))
        with self.assertRaises(ValueError):  # spent: a sealed set is read once
            K.exam(point, plan, **kw)
        other = json.loads(json.dumps(plan))
        other["points"][point]["versions"]["student"] = "someone@else"
        with self.assertRaises(ValueError):  # the weights belong to the student they were measured for
            K.exam(point, other, **{**kw, "campaign": tmp / "campaign2"})
        with self.assertRaises(ValueError):  # the plan is fixed once anything was examined
            K.write_plan(plan, campaign=tmp / "campaign")
        weighted = json.loads(json.dumps(plan))
        weighted["points"][point]["weights"]["gliner"] = 1.5  # a member with a weight that did not answer every decision
        with self.assertRaises(ValueError):
            K.exam(point, weighted, **{**kw, "campaign": tmp / "campaign4"})
        self.assertFalse((tmp / "campaign4" / "results").exists())  # refused before anything is written: the set is not spent
        # the Claude arm: evaluation only, after its point's exam, once per point and seed
        advisor = type("Scripted", (), {"cap_eur": 0.1, "spent_eur": lambda self: 0.0,
                                        "ask": lambda self, purpose, evidence: {"answer": "yes", "reason": "a key"}})()
        with self.assertRaises(ValueError):
            K.exam_claude((point,), 4, advisor, plan, fresh=tmp / "fresh", students=tmp / "models", campaign=tmp / "campaign5", lake=tmp / "lake")
        path = K.exam_claude((point,), 4, advisor, plan, fresh=tmp / "fresh", students=tmp / "models", campaign=tmp / "campaign", lake=tmp / "lake")
        sample = json.loads(path.read_text())
        self.assertEqual((sample["points"][point]["decisions"], sample["cap_eur"], sample["spent_eur"]), (4, 0.1, 0.0))
        kept = [json.loads(x) for x in (tmp / "lake" / "bronze" / "answers" / "claude" / "committee" / point / "sealed.jsonl").read_text().splitlines()]
        self.assertTrue(len(kept) == 4 and all(x["use"] == "eval_only" and x["silver"] is False for x in kept))
        with self.assertRaises(ValueError):  # the same sample twice
            K.exam_claude((point,), 4, advisor, plan, fresh=tmp / "fresh", students=tmp / "models", campaign=tmp / "campaign", lake=tmp / "lake")
        import shutil

        (tmp / "campaign6" / "results").mkdir(parents=True)  # the exam's result without the Claude one: its kept answers still refuse
        shutil.copy(next((tmp / "campaign" / "results").glob(f"CMT-001_{point}.json")), tmp / "campaign6" / "results")
        with self.assertRaises(ValueError):
            K.exam_claude((point,), 4, advisor, plan, fresh=tmp / "fresh", students=tmp / "models", campaign=tmp / "campaign6", lake=tmp / "lake")
        (tmp / "fresh" / "decisions" / point / "sealed.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sealed[1:]))
        with self.assertRaises(ValueError):  # a sealed file that moved is never read
            K.exam(point, plan, **{**kw, "campaign": tmp / "campaign3"})


class RegisterTests(unittest.TestCase):
    def test_a_committee_is_shown_only_beside_the_student_it_held(self):
        import json
        import tempfile
        from unittest import mock

        from dclab_rnd import students

        tmp = Path(tempfile.mkdtemp())
        results = tmp / "committee_v1" / "results"
        results.mkdir(parents=True)
        (results / "CMT-001_column.time.json").write_text(json.dumps({
            "id": "CMT-001", "kind": "committee_exam", "point": "column.time", "committee": "committee@abc", "result": "pass",
            "versions": {"student": "column.time.boosting@aaa"}, "verdicts": {"committee": {"criteria": []}}, "accuracy": {"committee": 0.97}}))
        with mock.patch.object(students, "CAMPAIGNS", tmp):
            for version, shown in (("aaa", True), ("bbb", False)):
                card = {"column.time": {"answerer": f"column.time.boosting@{version}", "family": "boosting"}}
                with mock.patch.object(students, "students", return_value=card):
                    row = next(r for r in students.register()["items"] if r["point"] == "column.time")
                self.assertEqual(row["committee"] is not None, shown)  # a refit student is another committee, never examined


if __name__ == "__main__":
    unittest.main()
