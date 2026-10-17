"""Package 29.17: Claude and GLiNER beside every student: the same dev decisions answered by the rule, the student, a
model through the gateway and GLiNER's captured answers, scored against the planted truth; the student's disagreements
first; every model answer kept whole, never as a label on homework; the plan printed before anything is sent."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core.answerers import Answer
    from dclab_rnd import students_exam as X
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""


def row(i, point="column.is_identifier", rule=True, truth=True, group="dataset:t1"):
    return {"record": f"r{i}", "point": point, "state": {"column": f"c{i}", "dtype": "integer"}, "options": ["yes", "no"],
            "value_used": rule, "label": {"value": truth, "source": "benchmark"}, "group": group, "split": "dev"}


class Student:
    point = "column.is_identifier"

    def __init__(self, values):
        self.values = values

    def answer(self, point, state, options):
        v = self.values.get(state["column"])
        return None if v is None else Answer(by="classifier", answerer="student@1", value=v)


class Advisor:
    """A scripted gateway advisor: answers yes for even columns, no for odd, with a reason; counts what it was asked."""

    def __init__(self):
        self.asked = []

    def ask(self, purpose, evidence):
        self.asked.append(purpose)
        import re

        n = int(re.search(r"\bc(\d)\b", str(evidence)).group(1))  # the column's name, c0 … c3
        return {"answer": "yes" if n % 2 == 0 else "no", "reason": "the name says so"}


@unittest.skipIf(MISSING, MISSING)
class ExamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sets = self.tmp / "sets"
        (self.sets / "column.is_identifier").mkdir(parents=True)
        rows = [row(0, rule=True, truth=True), row(1, rule=True, truth=False), row(2, rule=False, truth=True), row(3, rule=False, truth=False),
                {**row(4), "split": "train"}]
        (self.sets / "column.is_identifier" / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (self.sets / "column.is_identifier" / "sealed.jsonl").write_text("not json: the sealed file is never opened\n")
        self.lake = self.tmp / "lake"
        g = self.lake / "bronze" / "answers" / X.GLINER / "lake" / "column.is_identifier"
        g.mkdir(parents=True)
        (g / "dev.jsonl").write_text("".join(json.dumps({"sample_id": f"r{i}", "value": v}) + "\n" for i, v in ((0, True), (1, True), (2, True))))
        self.student = {"column.is_identifier": Student({"c0": True, "c1": False, "c2": False, "c3": False})}

    def test_the_sample_puts_the_students_disagreements_first_and_reads_dev_only(self):
        got = X.sample(2, seed=0, sets=self.sets, students=self.student, lake=self.lake)
        self.assertEqual(len(got), 2)
        self.assertIn("r1", [r["record"] for r in got])  # the only row where the student and the rule disagree
        self.assertTrue(all(r["split"] == "dev" for r in got))
        every = X.sample(10, seed=0, sets=self.sets, students=self.student, lake=self.lake)
        self.assertEqual(sorted(r["record"] for r in every), ["r0", "r1", "r2", "r3"])  # never the train row, never the sealed file

    def test_four_examinees_are_scored_on_the_same_decisions_and_claudes_answers_are_kept_never_as_labels(self):
        rows = X.sample(10, seed=0, sets=self.sets, students=self.student, lake=self.lake)
        advisor = Advisor()
        out = X.run(rows, advisor, students=self.student, lake=self.lake)
        s = out["scores"]["column.is_identifier"]
        self.assertEqual(s["decisions"], 4)
        self.assertEqual(s["answered_by"], {"rule": 4, "student": 4, X.CLAUDE: 4, X.GLINER: 3})
        self.assertEqual(s["common"], 3)  # GLiNER did not answer r3: compared where all four answered
        self.assertEqual(s["on_common"]["rule"]["accuracy"], round(1 / 3, 4))  # r0 right, r1 wrong, r2 wrong
        self.assertEqual(s["student_vs_rule"]["disagreements"], 1)  # r1 only: on r2 the student and the rule both said no
        self.assertEqual((s["student_vs_rule"]["right"]["student"], s["student_vs_rule"]["right"]["rule"]), (1, 0))  # r1's truth is no
        self.assertEqual(len(advisor.asked), 4)
        self.assertEqual(set(advisor.asked), {"decision_answer"})
        kept = [json.loads(x) for x in (self.lake / "bronze" / "answers" / X.CLAUDE / "lake" / "column.is_identifier" / "dev.jsonl").read_text().splitlines()]
        self.assertEqual(len(kept), 4)
        self.assertTrue(all(k["use"] == "eval_only" and k["silver"] is False and k["reason"] == "the name says so" for k in kept))
        path = X.write({"decisions": 4, **out}, results=self.tmp / "results")
        self.assertEqual(json.loads(path.read_text())["id"], "SEX-001")
        self.assertEqual(json.loads(X.write({"decisions": 4}, results=self.tmp / "results").read_text())["id"], "SEX-002")
        again = X.rescore(rows, lake=self.lake)  # from the kept answers: the same scores, no request
        self.assertEqual((again["scores"], again["requests"], len(advisor.asked)), (out["scores"], 0, 4))

    def test_an_examinee_with_no_answer_at_a_point_is_left_out_not_a_hole(self):
        rows = [row(0), row(1, rule=True, truth=False)]
        answers = {"rule": {"r0": True, "r1": True}, "student": {"r0": True, "r1": False}, X.CLAUDE: {"r0": True, "r1": False}, X.GLINER: {}}
        s = X.score(rows, answers)["column.is_identifier"]
        self.assertEqual((s["compared"], s["common"]), (["rule", "student", X.CLAUDE], 2))
        self.assertEqual(s["on_common"]["student"]["accuracy"], 1.0)

    def test_the_batched_student_answers_are_the_one_by_one_answers(self):
        from dclab_rnd.ledger import shadow

        models = {m.point: m for m in shadow.answerers()}
        if "feature.leakage_suspect" not in models:
            self.skipTest("no fitted student on this machine (make decision-models)")
        m = models["feature.leakage_suspect"]
        rows = X._rows("feature.leakage_suspect")[:300]
        one = [None if (a := m.answer(m.point, r["state"], list(r.get("options") or []))) is None else a.value for r in rows]
        self.assertEqual(X.student_values(m, m.point, rows), one)

    def test_the_plan_is_printed_and_nothing_is_sent_without_yes_and_a_cap(self):
        from unittest import mock

        with mock.patch.object(X, "SETS", self.sets), mock.patch.object(X, "students_by_point", return_value=self.student), \
                mock.patch.object(X, "gliner", return_value={}), mock.patch.object(X, "run", side_effect=AssertionError("sent")):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(X.main(["--per-point", "10"]), 0)
            self.assertIn("Plan: 4 dev decisions", out.getvalue())
            self.assertIn("about EUR", out.getvalue())
            with redirect_stdout(io.StringIO()):
                self.assertEqual(X.main(["--per-point", "10", "--yes"]), 2)  # no cap: refused before any request
            from tests.test_models_gateway import CLEAN

            with mock.patch.dict("os.environ", {**CLEAN, "DCLAB_AGENT_HOME": str(self.tmp / "home")}), \
                    mock.patch("dotenv.load_dotenv", return_value=False), redirect_stdout(io.StringIO()) as said:
                self.assertEqual(X.main(["--per-point", "10", "--yes", "--cap-eur", "1"]), 2)  # no model here: said, nothing written
            self.assertIn("no model serves decision_answer", said.getvalue())

    def test_a_run_no_model_answered_keeps_nothing(self):
        class Silent:
            def ask(self, purpose, evidence):
                return None  # what the gateway's advisor returns when no model serves the purpose

        rows = X.sample(10, seed=0, sets=self.sets, students=self.student, lake=self.lake)
        with self.assertRaises(RuntimeError):
            X.run(rows, Silent(), students=self.student, lake=self.lake)
        self.assertFalse((self.lake / "bronze" / "answers" / X.CLAUDE).exists())


@unittest.skipIf(MISSING, MISSING)
class JudgeTests(unittest.TestCase):
    """29.18: Claude as the blind judge of past decisions, measured first where the truth is known."""

    def test_the_judge_sees_no_answer_judges_every_decision_and_is_scored_against_the_truth(self):
        from dclab_rnd.ledger import model_labels

        tmp = Path(tempfile.mkdtemp())
        rows = [{**row(0, rule=True, truth=True), "_student": False}, {**row(1, rule=True, truth=False), "_student": False},
                {**row(2, rule=False, truth=True), "_student": None}, {**row(3, rule=False, truth=False), "_student": None}]
        rows = [{**r, "record": f"judge-{r['record']}-0000"} for r in rows]  # ledger ids: six characters at least
        store = X.as_ledger(rows, tmp / "ledger")
        seen = []

        def ask(view):
            seen.append(view)
            n = int(view["state"]["column"][1:])
            if n == 3:
                return {"answer": None, "reason": "cannot tell"}
            return {"answer": "yes" if n in (0, 2) else "no", "confidence": "high" if n < 2 else "low", "reason": f"because {n}"}

        counts = model_labels.to_label(store, ask, "claude-test", every=True, workers=3)
        self.assertEqual((counts["asked"], counts["labelled"], counts["unsure"]), (4, 3, 1))
        for view in seen:  # blind: never the rule's answer, the value used, or who decided
            self.assertNotIn("value_used", view)
            self.assertNotIn("answers", view)
        j = X.judged(rows, store, "claude-test")["column.is_identifier"]
        self.assertEqual((j["judged"], j["not_judged"], j["accuracy"]), (3, 1, 1.0))
        self.assertEqual(j["on_disagreements"], {"judged": 2, "sided_with_student": 1, "sided_with_rule": 1, "right": 2})
        self.assertEqual(j["by_confidence"]["high"], {"judged": 2, "right": 2})
        mine = [x for x in store.labels() if x.source == "model"]
        self.assertTrue(all(x.weight == 1.0 or x.weight > 1 for x in mine))  # a census weighs 1; a drawn one keeps its draw's weight
        lake = tmp / "lake"
        self.assertEqual(X.keep_verdicts(rows, store, "claude-test", lake=lake), 3)
        kept = [json.loads(x) for x in (lake / "bronze" / "answers" / X.CLAUDE / "lake" / "column.is_identifier" / "dev.jsonl").read_text().splitlines()]
        self.assertEqual({(k["role"], k["use"], k["package"]) for k in kept}, {("judge", "eval_only", "29.18")})
        self.assertIn("because", kept[0]["reason"])
        again = model_labels.to_label(store, ask, "claude-test", every=True, workers=3)  # a decision judged once is not asked again
        self.assertEqual(again["asked"], 1)  # only the one it was unsure of

    def test_every_reply_is_kept_whole_before_anything_is_cut(self):  # the owner's rule, 2026-10-18
        from dclab_rnd.ledger import model_labels

        tmp = Path(tempfile.mkdtemp())
        long_reason = "because " + "the bands say so " * 40  # far past the ledger's 200 characters

        class Client:
            model = "claude-sonnet-5-5"
            spent_eur = 0.0
            run_limit_eur = None

            def complete(self, messages, tools=None, max_tokens=1200):
                return {"content": json.dumps({"answer": "yes", "confidence": "high", "reason": long_reason}), "usage": {"input_tokens": 500, "output_tokens": 90}}

        class Gateway:
            def client(self, purpose):
                return Client()

        ask, _ = model_labels.asker(Gateway(), 1.0, keep=tmp / "judgments")
        rows = [{**row(0), "record": "judge-r0-0000"}]
        store = X.as_ledger(rows, tmp / "ledger")
        model_labels.to_label(store, ask, "claude-sonnet-5-5", every=True)
        kept = [json.loads(x) for x in (tmp / "judgments" / "judgments.jsonl").read_text().splitlines()]
        self.assertEqual((kept[0]["record"], kept[0]["parsed"]["reason"]), ("judge-r0-0000", long_reason))  # whole, keyed by the decision
        self.assertNotIn("record", kept[0]["sent"])  # the id is kept, never sent
        self.assertEqual((kept[0]["input_tokens"], kept[0]["output_tokens"]), (500, 90))
        self.assertGreater(kept[0]["cost_eur"], 0)
        label = next(x for x in store.labels() if x.source == "model")
        self.assertLessEqual(len(label.evidence.ref), 200)  # the ledger's copy is cut, after the whole one was kept
        X.keep_verdicts(rows, store, "claude-sonnet-5-5", lake=tmp / "lake", full=tmp / "judgments" / "judgments.jsonl")
        lake_row = json.loads((tmp / "lake" / "bronze" / "answers" / X.CLAUDE / "lake" / "column.is_identifier" / "dev.jsonl").read_text().splitlines()[0])
        self.assertEqual(lake_row["reason_full"], long_reason)

    def test_the_exams_replies_are_kept_whole_too(self):
        tmp = Path(tempfile.mkdtemp())

        class Inner:
            def ask(self, purpose, evidence):
                return {"answer": "yes", "reason": "x" * 900}

        w = X.Whole(Inner(), tmp / "answers.jsonl")
        self.assertEqual(len(w.ask("decision_answer", {"q": 1})["reason"]), 900)
        kept = json.loads((tmp / "answers.jsonl").read_text().splitlines()[0])
        self.assertEqual((kept["purpose"], len(kept["reply"]["reason"])), ("decision_answer", 900))

    def test_a_cap_stops_the_judge_and_keeps_what_was_written(self):
        from dclab_rnd.ledger import model_labels

        rows = [{**row(i), "record": f"judge-r{i}-0000"} for i in range(6)]
        store = X.as_ledger(rows, Path(tempfile.mkdtemp()) / "ledger")
        calls = {"n": 0}

        def ask(view):
            calls["n"] += 1
            if calls["n"] > 2:
                raise RuntimeError("the run's cap is reached")
            return {"answer": "yes", "reason": "r"}

        counts = model_labels.to_label(store, ask, "claude-test", every=True, workers=1)
        self.assertEqual((counts["labelled"], counts["stopped"]), (2, 1))
        self.assertIn("cap", counts["why_stopped"])


if __name__ == "__main__":
    unittest.main()
