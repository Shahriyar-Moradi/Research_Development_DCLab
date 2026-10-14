"""Twin runs (R&D): the current system and a scripted "Claude" on the same sample, at the same time, in two separate
workspace folders; everything each did is on disk; the comparison reads only those files; nothing is sent without a
key or under DCLAB_NO_LIVE_MODELS; the request cap stops the AI-first lane; a run never touches the main workspace."""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import twin  # noqa: E402
from dclab_rnd.twin import claude  # noqa: E402

MOMENT = "At the end of the examination, after every listed test, before the diagnosis is confirmed."


def reply(content=None, calls=(), tokens=(120, 20)):
    calls = [{"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)]
    return {"content": content, "finish_reason": "tool_calls" if calls else "stop", "tool_calls": calls,
            "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
            "assistant_message": {"role": "assistant", "content": content,
                                  "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]}}


PLAN = [("S1", "WF-02", "describe_data", None), ("S2", "WF-01", "understand_problem", None), ("S3", "WF-01", "propose_solution", None),
        ("S4", "WF-01", "set_solution", None), ("S5", "WF-04", "run_stage", "data"), ("S6", "WF-04", "approve_stage", "data"),
        ("S7", "WF-05", "run_stage", "leakage"), ("S8", "WF-05", "review_leakage", None), ("S9", "WF-05", "approve_stage", "leakage"),
        ("S10", "WF-06", "run_stage", "features"), ("S11", "WF-06", "approve_stage", "features"), ("S12", "WF-07", "run_stage", "models"),
        ("S13", "WF-07", "approve_stage", "models"), ("S14", "WF-09", "run_stage", "final"), ("S15", "WF-09", "approve_stage", "final"),
        ("S16", "WF-10", "finish", None)]
SAID = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S10", "S11", "S12", None, "S13", "S14", "S15", "S16"]  # what the agent names before each call


def design(checkpoint):
    """The scripted designer: one leakage hypothesis the rule does not share, a review step the agent never takes, a
    change of families after the leakage stage, a retrospective at the end."""
    families = ["logistic_regression", "extra_trees"] if checkpoint in ("initial", "data") else ["logistic_regression", "extra_trees", "hist_gradient_boosting"]
    out = {"understanding": {"restatement": "Predict heart disease at the end of the examination.", "task": "binary", "target": "target",
                             "prediction_moment": MOMENT, "unit": "one patient", "success": "beat the majority baseline on ROC-AUC"},
           "data_plan": {"checks": ["missing values", "class balance"], "expected_issues": ["small table"]},
           "leakage_hypotheses": [{"column": "thal", "why": "may be recorded after the diagnosis", "test": "apparent lift on identical folds", "action": "review"}],
           "identifiers": [], "split": {"strategy": "stratified", "column": None, "why": "no time column"},
           "features": {"direction": "raw columns first", "why": "small table"}, "models": {"families": families, "why": "linear and trees"},
           "evaluation": {"metric": "roc_auc", "baseline": "majority class", "holdout": "once, at the end"},
           "workflow": [{"id": i, "wf": wf, "tool": t, "stage": st, "does": f"{t} {st or ''}".strip(), "why": "in order", "evidence": "the stage record",
                         "decision_rule": "the best mean within one std", "after": [PLAN[k - 1][0]] if k else []} for k, (i, wf, t, st) in enumerate(PLAN)],
           "routes": [{"from": "S9", "to": "S4", "when": "a leak is confirmed"}], "risks": ["a small holdout"], "open_questions": [],
           "confidence": "medium", "changes": []}
    if checkpoint == "leakage":
        out["changes"] = [{"field": "models.families", "from": families[:2], "to": families, "because": "no leak found, room for a boosted family",
                           "evidence": "apparent lift"}]
    if checkpoint == "final":
        out["retrospective"] = {"what_worked": ["the stratified split"], "what_changed": ["one more family"], "would_do_differently": ["review thal earlier"]}
    return "Here is the design:\n```json\n" + json.dumps(out) + "\n```"


class ScriptedClaude:
    """A transport in the ChatClient's shape that plays an AI-first agent: it designs the solution when asked, reads
    the project id from the task and drives the real tools in its own order (not the standard plan's), naming the
    design step before each call and overriding one stage choice on the way."""

    model, base_url = "claude-scripted", claude.ANTHROPIC_OPENAI_URL

    def complete(self, messages, tools=None, max_tokens=None, **options):
        last = str(messages[-1].get("content") or "")
        if not tools and last.startswith("DESIGN REQUEST:"):
            head = last.splitlines()[0]
            checkpoint = "initial" if "initial" in head else "final" if "final" in head else head.split("after the ")[1].split(" stage")[0]
            return reply(design(checkpoint), tokens=(900, 400))
        if not tools:  # a role, a stage note or the leakage reviewer asked a plain question: a short JSON-free answer
            return reply("No change: the rules' reading stands.", tokens=(80, 12))
        text = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
        pid = re.search(r"existing project (\w{6,})", text).group(1)
        n = sum(1 for m in messages if m.get("role") == "assistant")
        said = (SAID[n] + ": the next step of my design.") if n < len(SAID) and SAID[n] else ("Reading the ranking before I choose." if n < len(SAID) else None)
        script = [
            [("describe_data", {"project_id": pid})],
            [("understand_problem", {"project_id": pid})],
            [("propose_solution", {"project_id": pid, "target": "target"})],
            [("set_solution", {"project_id": pid, "target": "target", "task": "binary", "prediction_moment": MOMENT, "forbidden": [], "identifiers": [],
                               "metric": "roc_auc"})],
            [("run_stage", {"project_id": pid, "stage": "data"})],
            [("approve_stage", {"project_id": pid, "stage": "data"})],
            [("run_stage", {"project_id": pid, "stage": "leakage"})],
            [("approve_stage", {"project_id": pid, "stage": "leakage"})],
            [("run_stage", {"project_id": pid, "stage": "features"})],
            [("approve_stage", {"project_id": pid, "stage": "features"})],
            [("run_stage", {"project_id": pid, "stage": "models"})],
            [("get_results", {"project_id": pid, "stage": "models"})],
        ]
        if n < len(script):
            return reply(said, script[n])
        if n == len(script):  # the AI-first agent prefers the runner-up family: the difference the comparison must show
            results = next((m for m in reversed(messages) if m.get("role") == "tool"), {})
            content = str(results.get("content") or "")
            options = re.findall(r'"model": "([a-z_]+)"', content)
            selected = re.search(r'"selected": "([a-z_]+)"', content)
            other = next((o for o in options if o != (selected.group(1) if selected else None)), None)
            return reply(said, [("approve_stage", {"project_id": pid, "stage": "models", **({"choice": other} if other else {})})])
        if n == len(script) + 1:
            return reply(said, [("run_stage", {"project_id": pid, "stage": "final"})])
        if n == len(script) + 2:
            return reply(said, [("approve_stage", {"project_id": pid, "stage": "final"})])
        return reply(said, [("finish", {"report": "AI-first run: solution set from the brief, five stages run, the runner-up family chosen on the models stage."})])


class TwinRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.home = Path(cls.tmp.name) / "workspace"
        cls.transport = ScriptedClaude()
        cls.manifest = twin.run("sample:heart_disease", home=cls.home, transport=cls.transport, max_requests=60, max_steps=24, max_minutes=10,
                                run_id="test-run", log=lambda *a: None)
        cls.root = cls.home / "twin" / "test-run"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_both_lanes_ran_in_their_own_folders_and_the_main_workspace_is_untouched(self):
        m = self.manifest
        self.assertEqual((m["lanes"]["workflow"]["status"], m["lanes"]["ai_first"]["status"]), ("completed", "completed"), m)
        for lane in ("workflow", "ai_first"):
            self.assertTrue((self.root / lane / "projects").is_dir())
            self.assertTrue((self.root / lane / "intern").is_dir())
            self.assertTrue((self.root / lane / "audit.jsonl").is_file(), lane)
            self.assertTrue((self.root / lane / "agent_steps.jsonl").is_file(), lane)
            self.assertTrue((self.root / lane / "decision_samples.jsonl").is_file(), lane)
            self.assertTrue((self.root / lane / "summary.json").is_file(), lane)
        self.assertNotEqual(m["lanes"]["workflow"]["project_id"], m["lanes"]["ai_first"]["project_id"])
        self.assertFalse((self.home / "projects").exists())  # nothing in the main workspace
        self.assertFalse((self.home / "audit.jsonl").exists())
        for name in ("run.json", "README.md", "compare.json", "COMPARE.md"):
            self.assertTrue((self.root / name).is_file(), name)

    def test_the_ai_first_lane_captured_every_exchange_and_the_workflow_lane_sent_nothing(self):
        calls = [json.loads(l) for l in (self.root / "ai_first" / "transcripts" / "model_calls.jsonl").read_text().splitlines()]
        self.assertGreaterEqual(len(calls), 12)
        first = next(c for c in calls if c.get("purpose") == "intern")
        self.assertEqual(first["messages"][0]["role"], "system")
        self.assertIn("tool_calls", first["answer"])
        self.assertTrue(all(c.get("answer") is not None or c.get("check") for c in calls if c.get("outcome", "ok") == "ok"))
        self.assertEqual(calls[-1]["outcome"], "ok")
        usage = [json.loads(l) for l in (self.root / "ai_first" / "model_requests.jsonl").read_text().splitlines()]
        self.assertEqual(len(usage), sum(1 for c in calls if "check" not in c and c.get("purpose") != "monitor"))
        self.assertFalse([c for c in calls if c.get("purpose") == "monitor"])  # the monitor never failed
        self.assertEqual(usage[0]["endpoint"], "api.anthropic.com")
        self.assertFalse((self.root / "workflow" / "transcripts").exists())
        tool_messages = [m for c in calls if "messages" in c for m in c["messages"] if m.get("role") == "tool"]
        self.assertTrue(tool_messages)
        self.assertTrue(all("\"examples\"" not in str(m.get("content")) for m in tool_messages))  # cell examples scrubbed from the transcript
        roles = [c for c in calls if c.get("purpose") in ("problem_analyst", "data_investigator", "solution_planner", "feature_engineer")]
        self.assertTrue(roles, "the AI-first lane's roles should have asked Claude")
        self.assertEqual(self.manifest["roles_on"][0], "problem_analyst")
        self.assertFalse((self.root / "workflow" / "model_requests.jsonl").exists())
        self.assertEqual(self.manifest["model_usage"]["requests"], len(usage))

    def test_the_comparison_shows_the_difference_the_agent_made(self):
        c = json.loads((self.root / "compare.json").read_text())
        self.assertEqual(c["status"], {"workflow": "completed", "ai_first": "completed"})
        self.assertIn("models", c["stage_differences"], c["stages"])
        models = next(r for r in c["stages"] if r["stage"] == "models")
        self.assertTrue(models["ai_first"]["overridden"])
        self.assertFalse(models["workflow"]["overridden"])
        self.assertIsNotNone(c["holdout"])
        self.assertEqual(c["holdout"]["metric"], "roc_auc")
        self.assertEqual(c["tool_calls"]["ai_first"][0], "describe_data")
        self.assertIn("understand_problem", c["tool_calls"]["ai_first"])
        self.assertNotIn("understand_problem", c["tool_calls"]["workflow"])  # the standard plan never asks the brief
        self.assertEqual(c["moves"]["workflow"]["refused"], 0)
        self.assertEqual(c["moves"]["workflow"]["allowed"], c["moves"]["workflow"]["total"])
        self.assertIn("approve_stage:models:ok", c["moves"]["sequence"]["ai_first"])
        md = (self.root / "COMPARE.md").read_text()
        self.assertIn("| models |", md)
        self.assertIn("**no**", md)
        self.assertIn("Holdout roc_auc", md)

    def test_compare_can_be_rebuilt_from_the_files_alone(self):
        before = json.loads((self.root / "compare.json").read_text())
        again = twin.compare_run(self.root)
        self.assertEqual(again["stage_differences"], before["stage_differences"])
        self.assertEqual(again["solution"], before["solution"])
        runs = twin.list_runs(self.home)
        self.assertEqual([r["run_id"] for r in runs], ["test-run"])
        self.assertEqual(runs[0]["status"], {"workflow": "completed", "ai_first": "completed"})

    def test_decision_samples_were_written_in_both_lanes_and_the_agents_approvals_are_labels(self):
        for lane in ("workflow", "ai_first"):
            samples = [json.loads(l) for l in (self.root / lane / "decision_samples.jsonl").read_text().splitlines()]
            self.assertTrue(any(s["point"] == "model.family" for s in samples), lane)
            self.assertTrue(any(s["point"] == "column.leakage" for s in samples), lane)
        self.assertFalse((self.root / "workflow" / "decision_labels.jsonl").exists())  # the standard plan approves nothing
        ai = [json.loads(l) for l in (self.root / "ai_first" / "decision_labels.jsonl").read_text().splitlines()]
        chosen = next(l for l in ai if l["key"] == "model.family:-")
        self.assertEqual((chosen["reason"], chosen["who"], chosen["source"]), ("chosen by the agent", "agent", "agent"))  # the agent's override, kept apart from a person's


class DesignTests(unittest.TestCase):
    """A6.7: how Claude designed the solution and its workflow, watched against what the agent then did."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.home = Path(cls.tmp.name) / "workspace"
        cls.manifest = twin.run("sample:heart_disease", home=cls.home, transport=ScriptedClaude(), max_requests=80, max_steps=24, max_minutes=10,
                                run_id="design-run", log=lambda *a: None)
        cls.root = cls.home / "twin" / "design-run"
        cls.design = cls.root / "ai_first" / "design"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, name):
        return [json.loads(l) for l in (self.design / name).read_text().splitlines()]

    def test_every_version_is_kept_with_its_checkpoint_and_the_workflow_lane_has_no_design(self):
        self.assertEqual(self.manifest["design"], {"on": True, "checkpoints": ["data", "leakage", "models"], "steer": True})
        for name in ("design_v01.json", "design_v01.md", "designs.jsonl", "timeline.jsonl", "adherence.json", "intents.jsonl", "sft_candidates.jsonl",
                     "summary.json", "DESIGN.md"):
            self.assertTrue((self.design / name).is_file(), name)
        versions = self.rows("designs.jsonl")
        self.assertEqual([v["checkpoint"] for v in versions], ["initial", "data", "leakage", "models", "final"])
        self.assertTrue(all(v["parsed"] and not v["problems"] for v in versions), versions)
        self.assertEqual(json.loads((self.design / "design_v03.json").read_text())["changes"][0]["field"], "models.families")
        self.assertIn("retrospective", json.loads((self.design / "design_v05.json").read_text()))
        self.assertFalse((self.root / "workflow" / "design").exists())
        purposes = [json.loads(l).get("purpose") for l in (self.root / "ai_first" / "transcripts" / "model_calls.jsonl").read_text().splitlines()]
        self.assertEqual(purposes.count("twin_designer"), 5)  # every design request is in the transcript and in the budget

    def test_the_agent_was_given_the_design_and_each_revision(self):
        from dclab_rnd.intern.sessions import SessionStore

        session = SessionStore(self.root / "ai_first" / "intern").get(self.manifest["lanes"]["ai_first"]["session_id"])
        users = [m["content"] for m in session["messages"] if m.get("role") == "user"]
        self.assertIn("You designed this solution before starting (version v01)", users[0])
        self.assertIn("| S8 | WF-05 | review_leakage |", users[0])
        revised = [u for u in users if u.startswith("Design revised to")]
        self.assertEqual([u.split(" ")[3] for u in revised], ["v02", "v03", "v04"])
        self.assertIn("models.families", revised[1])

    def test_the_timeline_follows_every_call_against_the_design(self):
        calls = [r for r in self.rows("timeline.jsonl") if r["kind"] == "tool_call"]
        tools = json.loads((self.root / "ai_first" / "summary.json").read_text())["session"]["tools"]
        self.assertEqual([c["tool"] for c in calls], tools)
        first = calls[0]
        self.assertEqual((first["tool"], first["planned_step"], first["plan"], first["named_steps"], first["design_version"]), ("describe_data", "S1", "in_order", ["1"], 1))
        ranking = next(c for c in calls if c["tool"] == "get_results")
        self.assertEqual((ranking["plan"], ranking["planned_step"]), ("unplanned", None))
        self.assertEqual(calls[-1]["tool"], "finish")  # the last step, read from the session after the run
        self.assertEqual({c["plan"] for c in calls if c["planned_step"]}, {"in_order"})  # the skipped S8 makes nothing after it a reorder
        self.assertEqual(next(c for c in calls if c["tool"] == "approve_stage" and c["stage"] == "models")["design_version"], 3)
        kinds = [r["kind"] for r in self.rows("timeline.jsonl")]
        self.assertEqual(kinds.count("design"), 5)
        self.assertEqual(kinds.count("design_handed_to_agent"), 3)
        self.assertTrue(all("\"examples\"" not in json.dumps(c) for c in calls))

    def test_plan_against_act_and_the_comparison(self):
        a = json.loads((self.design / "adherence.json").read_text())["to_first"]
        self.assertEqual((a["planned_steps"], a["done"], a["skipped"], a["unplanned"], a["unplanned_writes"], a["failed_or_refused"]),
                         (16, 15, ["S8"], ["get_results:models"], [], []))
        self.assertEqual(json.loads((self.design / "summary.json").read_text())["last_revision"], 4)  # the final version is not the yardstick
        c = json.loads((self.root / "compare.json").read_text())
        rows = {r["key"]: r for r in c["design"]["decisions"]}
        self.assertEqual(rows["contract.target_column:-"]["claude_first"], "target")
        self.assertEqual(rows["contract.target_column:-"]["did_ai_first"], "target")
        thal = rows["column.leakage:thal"]
        self.assertEqual((thal["claude_first"], thal["rule"], thal["first_agrees_with_rule"]), ("review_flag", "clear", False))
        self.assertIn("column.leakage:thal", c["design"]["disagreements_with_rule"])
        self.assertTrue(rows["model.families:-"]["changed"])
        self.assertEqual(rows["contract.prediction_moment:-"]["claude_first"], "stated")
        self.assertTrue(rows["contract.prediction_moment:-"]["did_as_last_design"])
        self.assertEqual(rows["split.strategy:-"]["did_ai_first"], "stratified")
        self.assertIsNotNone(c["design_seconds"])
        self.assertTrue(c["design"]["workflow_lane_plan"])  # the current system's plan, beside Claude's
        md = (self.root / "COMPARE.md").read_text()
        self.assertIn("## How Claude designed it", md)
        self.assertIn("**≠ rule**", md)
        story = (self.design / "DESIGN.md").read_text()
        self.assertIn("## What changed, and why", story)
        self.assertIn("## Plan against act", story)

    def test_design_requests_hold_no_cells_and_no_holdout_before_the_end(self):
        samples = self.rows("sft_candidates.jsonl")
        self.assertEqual(len(samples), 5)
        self.assertEqual(samples[0]["metadata"]["stage"], "solution_design")
        self.assertFalse(samples[0]["metadata"]["verified"])
        pid = self.manifest["lanes"]["ai_first"]["project_id"]
        for sample in samples:
            user = sample["messages"][1]["content"]
            self.assertNotIn("\"examples\"", user)
            self.assertNotIn(pid, user)
            self.assertEqual(json.loads(sample["messages"][2]["content"])["understanding"]["target"], "target")
            if sample["metadata"]["checkpoint"] != "final":
                self.assertNotIn("interval95", user)  # the final stage's record reaches only the last revision
        self.assertIn("interval95", samples[-1]["messages"][1]["content"])
        self.assertIsNotNone(samples[0]["metadata"]["outcome"]["holdout"])


class MalformedDesign(ScriptedClaude):
    """Claude answers the designer with fields of the wrong shape: the run must not notice."""

    def complete(self, messages, tools=None, max_tokens=None, **options):
        if not tools and str(messages[-1].get("content") or "").startswith("DESIGN REQUEST:"):
            return reply('{"understanding": "a churn problem", "workflow": 5, "risks": 3, "leakage_hypotheses": 2, "identifiers": "id", '
                         '"models": {"families": "lightgbm"}, "changes": {"field": "x"}}')
        return super().complete(messages, tools, max_tokens, **options)


class MalformedAndUnsteeredTests(unittest.TestCase):
    def test_a_malformed_design_is_kept_with_its_problems_and_the_run_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            m = twin.run("sample:heart_disease", home=home, transport=MalformedDesign(), max_requests=60, max_steps=24, max_minutes=10, run_id="bad",
                         design_checkpoints=("leakage",), design_steer=False, log=lambda *a: None)
            self.assertEqual(m["lanes"]["ai_first"]["status"], "completed", m["lanes"]["ai_first"])
            root = home / "twin" / "bad" / "ai_first"
            versions = [json.loads(l) for l in (root / "design" / "designs.jsonl").read_text().splitlines()]
            self.assertEqual([v["checkpoint"] for v in versions], ["initial", "leakage", "final"])
            self.assertTrue(any("workflow is not a list" in p for p in versions[0]["problems"]), versions[0]["problems"])
            self.assertTrue(any("understanding is not an object" in p for p in versions[0]["problems"]))
            intents = [json.loads(l) for l in (root / "design" / "intents.jsonl").read_text().splitlines() if json.loads(l)["version"] == 1]
            self.assertEqual({i["key"]: i["value"] for i in intents}, {"model.families:-": ["lightgbm"], "column.identifier:id": "identifier"})  # a string is one item
            from dclab_rnd.intern.sessions import SessionStore

            session = SessionStore(root / "intern").get(m["lanes"]["ai_first"]["session_id"])
            self.assertFalse([x for x in session["messages"] if x.get("role") == "user" and str(x.get("content")).startswith("Design revised")])  # --no-steer
            timeline = [json.loads(l) for l in (root / "design" / "timeline.jsonl").read_text().splitlines()]
            self.assertFalse([r for r in timeline if r["kind"] in ("monitor_error", "design_failed")], timeline)
            self.assertEqual({r["plan"] for r in timeline if r["kind"] == "tool_call"}, {"unplanned"})  # no workflow, so nothing was planned
            self.assertTrue((root / "design" / "DESIGN.md").is_file())


class DesignUnitTests(unittest.TestCase):
    def test_a_refused_call_carries_out_no_step_and_its_retry_does(self):
        from dclab_rnd.twin import design as d

        plan = {"workflow": [{"id": "S1", "tool": "approve_stage", "stage": "data"}, {"id": "S2", "tool": "run_stage", "stage": "leakage"}]}
        a = d.adherence(plan, [{"tool": "approve_stage", "stage": "data", "ok": False}, {"tool": "approve_stage", "stage": "data", "ok": True},
                               {"tool": "run_stage", "stage": "leakage", "ok": True}])
        self.assertEqual((a["in_order"], a["unplanned"], a["failed_or_refused"]), (["S1", "S2"], [], ["approve_stage:data"]))
        self.assertEqual(d.replay({"workflow": [{"id": "A", "tool": "run_stage", "stage": "leakage"}, {"id": "B", "tool": "describe_data"}]},
                                  [{"tool": "run_stage", "stage": "leakage", "ok": True}]), ({"A"}, 0))

    def test_the_designer_never_sees_the_targets_values(self):
        from dclab_rnd.twin import design as d

        cleaned = d.clean({"numbers": {"target_summary": {"positive_label": "yes", "positive_rate_train": 0.4}}, "columns": [{"name": "a", "examples": ["1"]}],
                           "solution": {"positive_label": "yes", "target": "y"}})
        self.assertEqual(cleaned, {"numbers": {}, "columns": [{"name": "a"}], "solution": {"target": "y"}})
        from types import SimpleNamespace

        monitor = d.Monitor(SimpleNamespace(current=lambda: None))  # a tool result in the timeline, as the designer later reads it
        row = monitor._row("S3", {"id": "c", "type": "function", "function": {"name": "propose_solution", "arguments": "{}"}},
                           {"role": "tool", "tool_call_id": "c", "content": json.dumps({"detected": {"classes": ["no", "yes"], "positive_label": "yes", "task": "binary"}})})
        self.assertNotIn("yes", row["result"])
        self.assertIn("binary", row["result"])

    def test_parse_reads_fenced_or_bare_json_and_names_what_is_wrong(self):
        from dclab_rnd.twin import design as d

        got, problems = d.parse("text\n```json\n{\"understanding\": {\"task\": \"binary\"}}\n```")
        self.assertEqual(got["understanding"]["task"], "binary")
        self.assertIn("missing: workflow", problems)
        self.assertEqual(d.parse("no design here"), (None, ["no JSON object in the answer"]))
        self.assertIsNone(d.parse("{not json}")[0])
        bad = {"workflow": [{"id": "S1", "wf": "WF-99", "tool": "nope"}, {"id": "S1"}], "routes": [{"from": "S1", "to": "S7"}], "split": {"strategy": "magic"}}
        found = d.problems(bad, ["describe_data"])
        self.assertTrue(any("WF-99" in p for p in found) and any("does not exist" in p for p in found) and any("twice" in p for p in found)
                        and any("S7" in p for p in found) and any("magic" in p for p in found), found)

    def test_adherence_counts_order_skips_and_unplanned_writes(self):
        from dclab_rnd.twin import design as d

        plan = {"workflow": [{"id": "S1", "tool": "describe_data"}, {"id": "S2", "tool": "run_stage", "stage": "data"}, {"id": "S3", "tool": "run_stage", "stage": "leakage"}]}
        a = d.adherence(plan, [{"tool": "run_stage", "stage": "leakage"}, {"tool": "describe_data"}, {"tool": "set_solution"}])
        self.assertEqual((a["in_order"], a["out_of_order"], a["skipped"], a["unplanned_writes"]), (["S3"], ["S1"], ["S2"], ["set_solution"]))
        self.assertEqual(d.intents({"leakage_hypotheses": [{"column": "x", "action": "exclude"}], "identifiers": ["id"]}),
                         {"column.leakage:x": "exclude", "column.forbidden:x": "exclude", "column.identifier:id": "identifier"})


class DraftReplayTests(unittest.TestCase):
    def test_a_drafts_chat_is_replayed_in_both_lanes_and_compared(self):
        from dclab_rnd.storage import open_stores

        class Chatty:
            model, base_url = "claude-scripted", claude.ANTHROPIC_OPENAI_URL

            def complete(self, messages, tools=None, max_tokens=None, **options):
                return reply("Which column holds the customer id, and when exactly is the prediction made?")

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            _p, drafts, _s = open_stores(home)
            d = drafts.create("We want to predict which customers will churn next month so the retention team can call them first.")
            drafts.update(d["id"], lambda x: x["messages"].extend([{"id": "u1", "role": "user", "kind": "text", "text": "The label is the churned column."},
                                                                   {"id": "u2", "role": "user", "kind": "text", "text": "We decide at the start of each month."}]))
            m = twin.run(f"draft:{d['id']}", home=home, transport=Chatty(), run_id="draftrun", log=lambda *a: None)
            self.assertEqual({k: v["status"] for k, v in m["lanes"].items()}, {"workflow": "completed", "ai_first": "completed"})
            self.assertEqual(m["draft_turns"], 2)
            c = json.loads((home / "twin" / "draftrun" / "compare.json").read_text())
            self.assertIn("draft", c)
            self.assertGreaterEqual(c["draft"]["agent_turns"]["ai_first"], 2)
            self.assertGreaterEqual(c["draft"]["agent_turns"]["workflow"], 2)
            self.assertNotEqual(m["lanes"]["workflow"]["draft_id"], d["id"])  # copies, never the source draft
            self.assertEqual(len(drafts.get(d["id"])["messages"]), 2)  # the source is untouched
            calls = (home / "twin" / "draftrun" / "ai_first" / "transcripts" / "model_calls.jsonl").read_text().splitlines()
            self.assertGreaterEqual(len(calls), 2)
            self.assertIn("Draft chat", (home / "twin" / "draftrun" / "COMPARE.md").read_text())
            self.assertFalse((home / "twin" / "draftrun" / "ai_first" / "design").exists())  # the design is for project runs
            self.assertFalse(m["design"]["on"])


class CapsAndGuardsTests(unittest.TestCase):
    def test_the_request_cap_stops_the_ai_first_lane_and_the_run_still_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            m = twin.run("sample:heart_disease", home=home, transport=ScriptedClaude(), max_requests=3, max_steps=24, max_minutes=10, run_id="capped", log=lambda *a: None)
            self.assertEqual(m["lanes"]["workflow"]["status"], "completed")
            self.assertIn(m["lanes"]["ai_first"]["status"], ("completed", "budget_exhausted", "failed"))
            calls = [json.loads(l) for l in (home / "twin" / "capped" / "ai_first" / "transcripts" / "model_calls.jsonl").read_text().splitlines()]
            self.assertTrue(any(str(c.get("outcome", "")).startswith("refused: the twin run's request cap") for c in calls))
            self.assertEqual(m["model_usage"]["requests"], 3)
            self.assertTrue((home / "twin" / "capped" / "COMPARE.md").is_file())
            timeline = [json.loads(l) for l in (home / "twin" / "capped" / "ai_first" / "design" / "timeline.jsonl").read_text().splitlines()]
            self.assertTrue(any(r["kind"] == "design_failed" and "request cap" in r["outcome"] for r in timeline))  # a refused revision is an event, not a failure

    def test_client_refuses_and_captures_a_provider_failure(self):
        class Broken:
            model, base_url = "x", claude.ANTHROPIC_OPENAI_URL

            def complete(self, *a, **k):
                raise RuntimeError("RateLimitError: slow down")

        with tempfile.TemporaryDirectory() as tmp:
            t = claude.Transcript(Path(tmp) / "t.jsonl")
            c = claude.ClaudeClient(Broken(), t, purpose="intern", max_requests=5)
            with self.assertRaises(RuntimeError):
                c.complete([{"role": "user", "content": "hi"}])
            rows = [json.loads(l) for l in (Path(tmp) / "t.jsonl").read_text().splitlines()]
            self.assertTrue(rows[0]["outcome"].startswith("RuntimeError"))
            self.assertEqual(rows[0]["messages"][0]["content"], "hi")

    def test_the_cli_sends_nothing_without_yes_or_a_key(self):
        from unittest import mock

        from dclab_rnd.twin.__main__ import main

        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "", "DCLAB_NO_LIVE_MODELS": "1"}):
            self.assertEqual(main(["run", "--sample", "heart_disease"]), 0)  # the plan only
            self.assertEqual(main(["run", "--sample", "heart_disease", "--no-design", "--design-checkpoints", "leakage"]), 0)
            self.assertEqual(main(["run", "--sample", "heart_disease", "--yes"]), 2)  # refused: no live model here, no key
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                twin.run("sample:heart_disease", home=Path(tmp), transport=ScriptedClaude(), design_checkpoints=("final",), log=lambda *a: None)
            with self.assertRaises(ValueError):
                twin.run("benchmark:x", home=Path(tmp), transport=ScriptedClaude(), log=lambda *a: None)
            with mock.patch.dict("os.environ", {"DCLAB_NO_LIVE_MODELS": "1"}), self.assertRaises(RuntimeError):
                twin.run("sample:heart_disease", home=Path(tmp), transport=None, log=lambda *a: None)  # never a live model here


if __name__ == "__main__":
    unittest.main()
