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


class ScriptedClaude:
    """A transport in the ChatClient's shape that plays an AI-first agent: it reads the project id from the task and
    drives the real tools in its own order (not the standard plan's), overriding one stage choice on the way."""

    model, base_url = "claude-scripted", claude.ANTHROPIC_OPENAI_URL

    def complete(self, messages, tools=None, max_tokens=None, **options):
        if not tools:  # a role, a stage note or the leakage reviewer asked a plain question: a short JSON-free answer
            return reply("No change: the rules' reading stands.", tokens=(80, 12))
        text = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
        pid = re.search(r"project (\w{6,})", text).group(1)
        n = sum(1 for m in messages if m.get("role") == "assistant")
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
            return reply(None, script[n])
        if n == len(script):  # the AI-first agent prefers the runner-up family: the difference the comparison must show
            results = next((m for m in reversed(messages) if m.get("role") == "tool"), {})
            content = str(results.get("content") or "")
            options = re.findall(r'"model": "([a-z_]+)"', content)
            selected = re.search(r'"selected": "([a-z_]+)"', content)
            other = next((o for o in options if o != (selected.group(1) if selected else None)), None)
            return reply(None, [("approve_stage", {"project_id": pid, "stage": "models", **({"choice": other} if other else {})})])
        if n == len(script) + 1:
            return reply(None, [("run_stage", {"project_id": pid, "stage": "final"})])
        if n == len(script) + 2:
            return reply(None, [("approve_stage", {"project_id": pid, "stage": "final"})])
        return reply(None, [("finish", {"report": "AI-first run: solution set from the brief, five stages run, the runner-up family chosen on the models stage."})])


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
        self.assertEqual(len(usage), sum(1 for c in calls if "check" not in c))
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
            self.assertEqual(main(["run", "--sample", "heart_disease", "--yes"]), 2)  # refused: no live model here, no key
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                twin.run("benchmark:x", home=Path(tmp), transport=ScriptedClaude(), log=lambda *a: None)
            with mock.patch.dict("os.environ", {"DCLAB_NO_LIVE_MODELS": "1"}), self.assertRaises(RuntimeError):
                twin.run("sample:heart_disease", home=Path(tmp), transport=None, log=lambda *a: None)  # never a live model here


if __name__ == "__main__":
    unittest.main()
