"""Package 26.9: every decision DCLab's rules, Claude, classifiers and people made, gathered whole from every workspace."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def record(i, by_model=True, shadow=False):
    answers = [{"by": "rule", "answerer": "rule:x", "value": "a"}]
    if by_model:
        answers.append({"by": "model_role", "answerer": "home", "value": "b", "reason": "the person said so"})
    if shadow:
        answers.append({"by": "classifier", "answerer": "clf@1", "value": "a", "confidence": 0.9, "source": "shadow"})
    return {"id": f"dr{i:020d}", "point": "problem.pack", "subject": {"kind": "draft", "key": "d1"}, "state": {"pack": "x"},
            "options": ["a", "b"], "answers": answers, "value_used": "a", "used": "rule", "at": "2026-10-17T00:00:00+00:00"}


def workspace(root, name, records, model="claude-sonnet-5-5"):
    ws = root / name
    (ws / "ledger" / "decisions" / "2026" / "10").mkdir(parents=True)
    (ws / "ledger" / "decisions" / "2026" / "10" / "17.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (ws / "model_requests.jsonl").write_text(json.dumps({"at": "t", "purpose": "home_agent", "model": model, "outcome": "ok"}) + "\n")
    return ws


class HarvestTests(unittest.TestCase):
    def setUp(self):
        self.root, self.lake = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())

    def harvest(self):
        from dclab_rnd.lake.harvest import harvest

        return harvest(self.root, self.lake, database=False)

    def test_every_decision_is_kept_whole_once_with_where_it_was_found(self):
        workspace(self.root, "dev", [record(1), record(2, shadow=True)])
        workspace(self.root, "copy/twin", [record(1)])  # the same record kept in two places
        m = self.harvest()
        self.assertEqual((m["ledgers"], m["decisions"], m["model_requests"]), (2, 2, 1))
        rows = [json.loads(x) for x in (self.lake / "bronze" / "sessions" / "ledgers" / "decisions.jsonl").read_text().splitlines()]
        one = next(r for r in rows if r["id"] == record(1)["id"])
        self.assertEqual(sorted(one["found_in"]), ["copy/twin", "dev"])
        self.assertEqual(len(one["answers"]), 2)
        self.assertEqual(one["served_models"], {"home_agent": "claude-sonnet-5-5"})

    def test_each_answer_goes_under_its_answerer_with_what_it_may_be_used_for(self):
        workspace(self.root, "dev", [record(1, shadow=True)])
        workspace(self.root, "local", [record(2)], model="qwen2.5-coder:1.5b")  # a local model's role: not Claude's
        m = self.harvest()
        self.assertEqual(m["answers_by_answerer"], {"dclab-rules": 2, "claude": 1, "dclab-classifiers": 1})
        base = self.lake / "bronze" / "answers"
        claude = [json.loads(x) for x in (base / "claude" / "sessions" / "problem.pack" / "all.jsonl").read_text().splitlines()]
        self.assertEqual((claude[0]["use"], claude[0]["value"], claude[0]["reason"]), ("eval_only", "b", "the person said so"))
        rules = [json.loads(x) for x in (base / "dclab-rules" / "sessions" / "problem.pack" / "all.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["use"] == "train" for r in rules))

    def test_a_test_scripts_clicks_are_never_filed_as_peoples_answers(self):
        """The review of 2026-10-25: the end-to-end tests' kept workspaces hold scripted "person" answers; filed as people's
        they would have outnumbered every real person's answer in the lake."""
        person = dict(record(1, by_model=False), answers=[{"by": "rule", "answerer": "rule:x", "value": "a"},
                                                         {"by": "person", "answerer": "owner", "value": "b"}])
        real = dict(person, id="dr" + "9" * 20)
        ws = workspace(self.root, "e2e/2026-10-25/164537_product_e2e", [person])
        (ws.parent.parent / "scripted.json").write_text('{"by": "script"}')  # the marker sits above the kept workspace
        workspace(self.root, "dev", [real])
        m = self.harvest()
        self.assertEqual(m["answers_by_answerer"].get("people"), 1)  # the real person's only
        self.assertEqual(m["answers_by_answerer"].get("dclab-rules"), 2)  # the rules' answers in both stay

    def test_claudes_verdicts_on_real_decisions_are_silver_weighted_by_its_measured_accuracy(self):  # 29.18
        from dclab_rnd import ledger as L
        from dclab_rnd.lake import harvest as H

        book = L.FileLedger(self.root / "w1" / "ledger", "w1")
        rec = L.decide("column.is_identifier", ("column", "order_id"), answers=[L.answer("rule", "id_like", True)], value_used=True,
                       state={"column": "order_id"}, ledger=book, project="p1")
        L.label(rec, True, "model", "model:claude-sonnet-5-5", weight=1.0, ledger=book, evidence={"kind": "review", "ref": "high: the bands say so"})
        L.label(rec, False, "model", "model:qwen-local", weight=1.0, ledger=book, evidence={"kind": "review", "ref": "low: x"})  # not Claude: not here
        with mock.patch.object(H, "judge_accuracy", return_value={rec.point: 0.86}):
            H.harvest(self.root, self.lake, database=False)
        kept = [json.loads(x) for x in (self.lake / "bronze" / "answers" / "claude" / "sessions" / rec.point / "judge.jsonl").read_text().splitlines()]
        self.assertEqual(len(kept), 1)
        k = kept[0]
        self.assertEqual((k["use"], k["role"], k["silver_weight"], k["usable"], k["confidence"], k["reason"]),
                         ("silver", "judge", 0.86, True, "high", "the bands say so"))
        claude_led = L.decide("column.is_identifier", ("column", "rider_id"), answers=[L.answer("rule", "id_like", False), L.answer("model_role", "home", True)],
                              value_used=True, used="model_role", level=2, state={"column": "rider_id"}, ledger=book, project="p1")  # Claude decided it
        (self.root / "w1" / "model_requests.jsonl").write_text(json.dumps({"at": "t", "purpose": "home_agent", "model": "claude-sonnet-5-5"}) + "\n")
        L.label(claude_led, True, "model", "model:claude-sonnet-5-5", weight=1.0, ledger=book, evidence={"kind": "review", "ref": "high: same"})
        with mock.patch.object(H, "judge_accuracy", return_value={rec.point: 0.86}):
            H.harvest(self.root, self.lake, database=False)
        rows = {json.loads(x)["sample_id"]: json.loads(x) for x in (self.lake / "bronze" / "answers" / "claude" / "sessions" / rec.point / "judge.jsonl").read_text().splitlines()}
        self.assertEqual((rows[claude_led.id]["self_judged"], rows[claude_led.id]["usable"]), (True, False))  # never grading its own answer
        with mock.patch.object(H, "judge_accuracy", return_value={rec.point: 0.70, f"{rec.point}|high": 1.0}):  # its high verdicts measured apart
            H.harvest(self.root, self.lake, database=False)
        k = {json.loads(x)["sample_id"]: json.loads(x) for x in (self.lake / "bronze" / "answers" / "claude" / "sessions" / rec.point / "judge.jsonl").read_text().splitlines()}[rec.id]
        self.assertEqual((k["silver_weight"], k["usable"]), (1.0, True))  # a weak point overall, but its high-confidence verdicts proved right
        with mock.patch.object(H, "judge_accuracy", return_value={}):  # never measured on this point: kept, not usable
            H.harvest(self.root, self.lake, database=False)
        k = {json.loads(x)["sample_id"]: json.loads(x) for x in (self.lake / "bronze" / "answers" / "claude" / "sessions" / rec.point / "judge.jsonl").read_text().splitlines()}[rec.id]
        self.assertEqual((k["silver_weight"], k["usable"]), (None, False))

    def test_a_twin_runs_steps_are_kept_and_claudes_side_is_evaluation_only(self):
        run = self.root / "twin" / "runs" / "20261017-000000-abc"
        run.mkdir(parents=True)
        step = {"id": "st1", "step": "solution.columns", "system": {"answerer": "rules", "status": "abstained"},
                "ai_first": {"answerer": "claude-sonnet-5-5", "status": "answered", "answer": {"target": "Churn"}}, "truth": {"value": "x"}}
        (run / "steps.jsonl").write_text(json.dumps(step) + "\n")
        (run / "ai_first" / "design").mkdir(parents=True)
        (run / "ai_first" / "design" / "DESIGN.md").write_text("# design")
        m = self.harvest()
        self.assertEqual(m["twin_runs"], 1)
        self.assertTrue((self.lake / "bronze" / "sessions" / "twin" / run.name / "ai_first" / "design" / "DESIGN.md").is_file())
        claude = json.loads((self.lake / "bronze" / "answers" / "claude" / "twin" / "solution.columns" / "all.jsonl").read_text())
        self.assertEqual((claude["use"], claude["value"]), ("eval_only", {"target": "Churn"}))

    def test_the_rules_answer_every_test_in_the_same_layout(self):
        from dclab_core import lake as L
        from dclab_rnd.lake import answers

        sample = L.DecisionSample(sample_id="s1", decision_point="column.time", subject_kind="column", subject_ref="c",
                                  features={"column": "at", "dtype": "datetime"}, state_text="{}", options=["yes", "no"], label=True,
                                  rule_answer=True, provenance=L.Provenance(source="t", origin="construction", consent="ours", group="g", split="dev"))
        folder = self.lake / "gold" / "decisions" / "column.time" / "v1"
        folder.mkdir(parents=True)
        (folder / "dev.jsonl").write_text(sample.model_dump_json() + "\n")
        got = answers.record_rules(("lake",), self.lake)
        self.assertEqual(got, {"lake/column.time/dev.jsonl": 1})
        rec = answers.stored("dclab-rules", "lake", "column.time", "dev", self.lake)[0]
        self.assertEqual((rec["value"], rec["use"], rec["answerer"]), (True, "train", "dclab-rules"))


if __name__ == "__main__":
    unittest.main()
