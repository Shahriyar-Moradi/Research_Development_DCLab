"""Package 26.9: every decision DCLab's rules, Claude, classifiers and people made, gathered whole from every workspace."""

import json
import tempfile
import unittest
from pathlib import Path


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
