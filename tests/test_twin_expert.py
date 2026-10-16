"""Claude the expert (package 25.3): the expert record of how the problem and the data are understood, explored,
prepared, typed, solved and drawn as workflows, kept version by version beside what the agent then did; its decisions
on the ledger's points; its versions compared with the current system as twin steps; Claude's thinking kept."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from dclab_core import expert as E  # noqa: E402

try:
    import pandas  # noqa: F401

    from dclab_rnd import twin  # noqa: E402
    from test_twin_runs import MOMENT, ScriptedClaude, reply  # noqa: E402
    MISSING = None
except ImportError as error:  # pragma: no cover
    MISSING = f"the studio needs pandas: {error}"

PLAN = [("S1", "WF-02", "describe_data", None), ("S2", "WF-01", "understand_problem", None), ("S3", "WF-01", "propose_solution", None),
        ("S4", "WF-01", "set_solution", None), ("S5", "WF-04", "run_stage", "data"), ("S6", "WF-04", "approve_stage", "data"),
        ("S7", "WF-05", "run_stage", "leakage"), ("S8", "WF-05", "review_leakage", None), ("S9", "WF-05", "approve_stage", "leakage"),
        ("S10", "WF-06", "run_stage", "features"), ("S11", "WF-06", "approve_stage", "features"), ("S12", "WF-07", "run_stage", "models"),
        ("S13", "WF-07", "approve_stage", "models"), ("S14", "WF-09", "run_stage", "final"), ("S15", "WF-09", "approve_stage", "final"),
        ("S16", "WF-10", "finish", None)]


def record(checkpoint):
    families = [{"family": "logistic_regression", "rank": 1, "why": "small table, a linear baseline first"},
                {"family": "extra_trees", "rank": 2, "why": "interactions"}]
    if checkpoint not in ("initial", "data"):
        families.append({"family": "hist_gradient_boosting", "rank": 3, "why": "no leak found: room for boosting"})
    out = {"problem": {"restatement": "Predict heart disease at the end of the examination.", "decision": "refer the patient", "unit": "one patient",
                       "moment": MOMENT, "outcome": "heart disease is present", "outcome_known": "after the diagnosis", "target": "target",
                       "success": "beat the majority baseline on ROC-AUC"},
           "problem_type": {"family": "classification", "kind": "binary", "task": "binary", "labels": "column", "why": "a yes/no outcome with its label in a column",
                            "alternatives": [{"family": "rare_event", "why_not": "the positives are not rare"}]},
           "data": [{"name": "heart", "grain": "one patient's examination", "columns": [{"name": "thal", "role": "feature", "why": "a test result"}],
                     "identifiers": [], "times": [], "groups": [],
                     "leak_suspects": [{"column": "thal", "why": "may be recorded after the diagnosis", "test": "apparent lift on identical folds", "action": "review"}],
                     "quality": ["a small table"]}],
           "eda": [{"question": "How balanced is the outcome?", "tool": "describe_data", "answer": None, "conclusion": "stratify the split"}],
           "preprocessing": [{"step": "one-hot encode the coded categories", "columns": ["cp", "thal"], "why": "codes are not amounts", "inside_folds": True}],
           "problem_workflow": {"size": "single", "parts": [{"title": "heart disease", "family": "classification", "unit": "one patient"}], "composition": None},
           "algorithms": {"ranked": families, "baseline": "majority class", "would_change": "a leak or a time column"},
           "evaluation": {"split": "stratified", "split_column": None, "split_why": "no time column", "metric": "roc_auc", "baseline": "majority", "holdout": "once, at the end"},
           "solution_workflow": [{"id": i, "wf": wf, "tool": t, "stage": st, "does": f"{t} {st or ''}".strip(), "why": "in order", "evidence": "the stage record",
                                  "decision_rule": "the best mean within one std", "after": [PLAN[k - 1][0]] if k else []} for k, (i, wf, t, st) in enumerate(PLAN)],
           "routes": [{"from": "S9", "to": "S4", "when": "a leak is confirmed"}], "risks": ["a small holdout"], "questions": ["Is thal measured before the diagnosis?"],
           "confidence": "medium", "changes": []}
    if checkpoint == "leakage":
        out["changes"] = [{"field": "algorithms.ranked", "from": "two families", "to": "three families", "because": "no leak found", "evidence": "apparent lift"}]
        out["eda"][0]["answer"] = "about half positive"
    if checkpoint == "final":
        out["retrospective"] = {"what_worked": ["the stratified split"], "what_changed": ["one more family"], "would_do_differently": ["ask about thal first"]}
    return "Here is my record:\n```json\n" + json.dumps(out) + "\n```"


class ScriptedExpert(ScriptedClaude):
    """The scripted agent of the run tests, which also writes the expert record when asked, with its thinking."""

    def complete(self, messages, tools=None, max_tokens=None, **options):
        last = str(messages[-1].get("content") or "")
        if not tools and last.startswith("DESIGN REQUEST:"):
            head = last.splitlines()[0]
            checkpoint = "initial" if "initial" in head else "final" if "final" in head else head.split("after the ")[1].split(" stage")[0]
            out = reply(record(checkpoint), tokens=(900, 400))
            out["thinking"] = [{"text": f"Thinking about the {checkpoint} record: the target is a diagnosis.", "signature": "s"}]
            return out
        return super().complete(messages, tools, max_tokens, **options)


class ExpertCoreTest(unittest.TestCase):
    def test_review_findings_on_reading(self):  # 25.3 review: F8, F9, F10, F11, F12
        trace = E.read({"algorithms": {"ranked": [{"family": "x", "rank": float("inf")}, {"family": "y", "rank": "inf"}]}})
        self.assertEqual([a.rank for a in trace.algorithms.ranked], [1, 2])
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            E.ExpertTrace.model_validate({"risks": ["x" * 100000]})
        with self.assertRaises(ValidationError):
            E.ExpertTrace.model_validate({"retrospective": {"k" * 50: ["y"]}})
        wide = E.read({"data": [{"name": f"t{t}", "identifiers": [f"id{t}_{i}" for i in range(40)]} for t in range(2)]})
        self.assertEqual(len(E.step_values(wide)["solution.columns"]["identifiers"]), 60)
        roles = E.read({"data": [{"name": "t", "columns": [{"name": "paid_amount", "role": "leak"}, {"name": "order_id", "role": "identifier"},
                                                          {"name": "created", "role": "time"}],
                                  "leak_suspects": [{"column": "refund", "action": "drop"}, {"column": "notes", "action": "maybe"}]}]})
        self.assertEqual(E.intents(roles)["feature.leakage_suspect"], {"refund": True, "notes": True, "paid_amount": True})
        self.assertEqual(E.step_values(roles)["solution.columns"]["leakage"], ["paid_amount", "refund"])
        self.assertEqual((roles.data[0].identifiers, roles.data[0].times), (["order_id"], ["created"]))
        self.assertTrue(any("'maybe'" in p for p in roles.problems))
        late = E.read({"solution_workflow": [{"id": "S1", "tool": "get_results", "stage": "leakage"}, {"id": "S2", "tool": "approve_stage", "stage": "features"},
                                             {"id": "S3", "tool": "Approve_Stage", "stage": "models"}, {"id": "S4", "tool": "run_stage", "stage": "final"},
                                             {"id": "S5", "tool": "run_stage", "stage": "leakage"}]})
        for problem in ("the holdout is read before the leakage audit", "a feature recipe is chosen before the leakage audit",
                        "a model is chosen before the leakage audit"):
            self.assertIn(problem, late.problems)

    def test_a_partial_answer_is_kept_with_its_problems_never_refused(self):
        trace = E.read({"problem": {"target": "y"}, "problem_type": {"family": "astrology"}}, model="m")
        self.assertEqual(trace.problem.target, "y")
        self.assertIn("missing: data", trace.problems)
        self.assertTrue(any("astrology" in p for p in trace.problems))
        self.assertEqual(E.ExpertTrace.read(trace.model_dump(mode="json")), trace)

    def test_a_workflow_that_reads_the_holdout_before_the_leakage_audit_is_named(self):
        bad = E.read({"solution_workflow": [{"id": "S1", "tool": "run_stage", "stage": "final"}, {"id": "S2", "tool": "run_stage", "stage": "leakage"},
                                            {"id": "S3", "tool": "run_stage", "stage": "final", "after": ["S9"]}],
                      "routes": [{"from": "S1", "to": "S7"}]})
        problems = set(bad.problems)
        self.assertIn("the holdout is read before the leakage audit", problems)
        self.assertIn("the holdout is read more than once (the final stage runs twice)", problems)
        self.assertIn("workflow step S3: needs S9, which is not a step", problems)
        self.assertTrue(any(p.startswith("route S1→S7") for p in problems))
        self.assertIn("workflow step S1: tool 'run_stage' does not exist", E.workflow_problems(bad, tools=["describe_data"]))

    def test_the_records_decisions_are_on_the_ledgers_points_and_compare_as_twin_steps(self):
        raw, _ = E.extract(record("initial"))
        trace = E.read(raw, model="claude")
        self.assertEqual(trace.problems, [])
        intents = E.intents(trace)
        self.assertEqual((intents["target.column"]["-"], intents["problem.family"]["-"], intents["split.strategy"]["-"], intents["model.family"]["-"]),
                         ("target", "classification", "stratified", "logistic_regression"))
        self.assertEqual(intents["feature.leakage_suspect"], {"thal": True})
        from dclab_core.ledger import POINTS
        self.assertTrue(set(intents) <= set(POINTS))
        steps = E.twin_steps(trace, {"stage.data": {"split": "stratified", "metric": "roc_auc"}, "stage.models": {"family": "extra_trees"}},
                             session="s", at="2026-10-16", input={"t": 1})
        by = {s.step: s for s in steps}
        self.assertTrue(all(d.kind in ("same", "neither") for d in by["stage.data"].differences))
        self.assertEqual(next(d for d in by["stage.models"].differences if d.field == "family").kind, "differs")
        self.assertEqual(by["home.map"].system.status, "skipped")  # the system was given no map step here: nothing to contest
        self.assertEqual(by["home.map"].verdict.winner, "unjudged")


class UnsafeExpert(ScriptedExpert if not MISSING else object):
    """An expert whose first workflow reads the holdout before the leakage audit."""

    def complete(self, messages, tools=None, max_tokens=None, **options):
        out = super().complete(messages, tools, max_tokens, **options)
        last = str(messages[-1].get("content") or "")
        if not tools and last.startswith("DESIGN REQUEST: initial"):
            raw = json.loads(out["content"].split("```json\n")[1].split("\n```")[0])
            raw["solution_workflow"] = [{"id": "S1", "wf": "WF-09", "tool": "run_stage", "stage": "final"},
                                        {"id": "S2", "wf": "WF-05", "tool": "run_stage", "stage": "leakage"}]
            out = reply("```json\n" + json.dumps(raw) + "\n```", tokens=(900, 400))
        return out


@unittest.skipIf(MISSING, MISSING)
class UnsafeRecordTest(unittest.TestCase):
    def test_a_record_that_reads_the_holdout_first_is_kept_and_never_handed_to_the_agent(self):
        from dclab_rnd.intern.sessions import SessionStore

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            m = twin.run("sample:heart_disease", home=Path(tmp), transport=UnsafeExpert(), max_requests=60, max_steps=24, max_minutes=10,
                         run_id="unsafe", design_checkpoints=(), log=lambda *a: None)
            root = Path(tmp) / "twin" / "runs" / "unsafe" / "ai_first"
            timeline = [json.loads(line) for line in (root / "design" / "timeline.jsonl").read_text().splitlines()]
            self.assertTrue(any(r["kind"] == "design_not_used" and "before the leakage audit" in r["outcome"] for r in timeline))
            session = SessionStore(root / "intern").get(m["lanes"]["ai_first"]["session_id"])
            first = next(msg["content"] for msg in session["messages"] if msg.get("role") == "user")
            self.assertNotIn("You wrote this record", first)  # the task carries no record to follow


@unittest.skipIf(MISSING, MISSING)
class ExpertRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.home = Path(cls.tmp.name) / "workspace"
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            cls.manifest = twin.run("sample:heart_disease", home=cls.home, transport=ScriptedExpert(), max_requests=90, max_steps=24, max_minutes=10,
                                    run_id="expert-run", blind=True, log=lambda *a: None)
        cls.root = cls.home / "twin" / "runs" / "expert-run"
        cls.design = cls.root / "ai_first" / "design"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, name):
        return [json.loads(line) for line in (self.design / name).read_text().splitlines()]

    def test_every_version_is_kept_as_an_expert_trace(self):
        self.assertEqual(self.manifest["design"], {"on": True, "checkpoints": ["data", "leakage", "models"], "steer": True})
        for name in ("design_v01.json", "design_v01.md", "designs.jsonl", "expert.jsonl", "timeline.jsonl", "adherence.json", "intents.jsonl",
                     "sft_candidates.jsonl", "summary.json", "DESIGN.md"):
            self.assertTrue((self.design / name).is_file(), name)
        traces = [E.ExpertTrace.read(r) for r in self.rows("expert.jsonl")]
        self.assertEqual([t.checkpoint for t in traces], ["initial", "data", "leakage", "models", "final"])
        self.assertTrue(all(not t.problems for t in traces), [t.problems for t in traces])
        self.assertEqual(traces[2].changes[0].field, "algorithms.ranked")
        self.assertEqual(traces[2].eda[0].answer, "about half positive")
        self.assertEqual(traces[-1].retrospective.would_do_differently, ["ask about thal first"])
        page = (self.design / "design_v01.md").read_text()
        for section in ("## What kind of problem", "## The data: heart", "## Exploring the data", "## Preparing the data", "## The problem's workflow",
                        "## Algorithms", "## The solution's workflow"):
            self.assertIn(section, page)
        self.assertFalse((self.root / "workflow" / "design").exists())

    def test_the_agent_followed_the_record_and_the_timeline_watched_it(self):
        calls = [r for r in self.rows("timeline.jsonl") if r["kind"] == "tool_call"]
        self.assertEqual(calls[0]["tool"], "describe_data")
        self.assertEqual(calls[0]["planned_step"], "S1")
        self.assertEqual([r["kind"] for r in self.rows("timeline.jsonl")].count("design"), 5)
        a = json.loads((self.design / "adherence.json").read_text())["to_first"]
        self.assertEqual(a["planned_steps"], 16)
        self.assertIn("S8", a["skipped"])  # the review step the agent never took

    def test_the_records_intents_and_its_twin_steps_against_the_system(self):
        intents = self.rows("intents.jsonl")
        self.assertTrue(any(r["key"] == "feature.leakage_suspect:thal" and r["value"] is True and r["source"] == "ai_first@expert" for r in intents))
        from dclab_core.twin import TwinStep

        steps = [TwinStep.read(json.loads(line)) for line in (self.root / "expert_steps.jsonl").read_text().splitlines()]
        self.assertEqual({s.subject for s in steps}, {"v1:initial"})  # only the first version answered the system's input
        columns = next(s for s in steps if s.step == "solution.columns" and s.subject == "v1:initial")
        self.assertEqual((columns.truth.source, columns.verdict.winner), ("planted", "both"))
        c = json.loads((self.root / "compare.json").read_text())
        self.assertTrue(c["expert_judged"])
        rows = {r["key"]: r for r in c["design"]["decisions"]}
        self.assertEqual(rows["target.column:-"]["did_ai_first"], "target")
        self.assertIn("## How Claude, as the expert, designed it", (self.root / "COMPARE.md").read_text())

    def test_claudes_thinking_is_kept_with_each_version(self):
        designs = self.rows("designs.jsonl")
        self.assertTrue(all(d["thinking"] for d in designs))
        thoughts = [json.loads(line) for line in (self.root / "ai_first" / "transcripts" / "thinking.jsonl").read_text().splitlines()]
        self.assertEqual(len(thoughts), 5)
        self.assertEqual(thoughts[0]["purpose"], "ai_first")
        self.assertEqual(self.manifest["model_usage"]["thinking_replies"], 5)

    def test_the_record_never_sees_a_cell_value(self):
        for sample in self.rows("sft_candidates.jsonl"):
            self.assertNotIn("\"examples\"", sample["messages"][1]["content"])
            self.assertEqual(sample["metadata"]["source"], "twin:claude")


if __name__ == "__main__":
    unittest.main()
