"""D2 (2026-10-23), the thinking slot on every recorded step, the tools offered, and per-tool counts with the same columns.

A step's thought is the model's reasoning when the transport returned it ("recorded"), else the words it wrote before
its tool call ("visible"), else nothing said plainly; the standard plan's steps carry the plan's own sentence
("from_record"). Every run's record counts calls, errors and refusals per tool it could call, zeros for the unused, so
runs of one policy always have the same columns. A v1 example still reads under the v2 contracts.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)


class PureTests(unittest.TestCase):
    def test_the_thinking_slot_prefers_recorded_reasoning_then_visible_words(self):
        from dclab_rnd.agents.runtime import thought_of

        self.assertEqual(thought_of({"content": "I will look.", "reasoning": "Plan: look first."}), ("Plan: look first.", "recorded"))
        self.assertEqual(thought_of({"content": "I will look.", "thinking": [{"text": "A thought.", "signature": "s"}]}), ("A thought.", "recorded"))
        self.assertEqual(thought_of({"content": "  I will look.  ", "thinking": []}), ("I will look.", "visible"))
        self.assertEqual(thought_of({"content": "", "tool_calls": [{"name": "x"}]}), (None, None))

    def test_tool_counts_have_the_same_columns_every_run(self):
        from dclab_rnd.experience.hooks import tool_counts

        counts = tool_counts([("a", "ok"), ("a", "error"), ("b", "blocked"), ("nope", "error"), (None, "answered")], ["a", "b", "c"])
        self.assertEqual(counts["calls:a"], 2.0)
        self.assertEqual(counts["errors:a"], 1.0)
        self.assertEqual((counts["calls:b"], counts["refused:b"]), (1.0, 1.0))
        self.assertEqual((counts["calls:c"], counts["errors:c"], counts["refused:c"]), (0.0, 0.0, 0.0))  # offered, never called: still a column
        self.assertEqual((counts["calls:_not_offered"], counts["refused:_not_offered"]), (1.0, 1.0))
        self.assertEqual(set(counts), set(tool_counts([], ["a", "b", "c"])))  # the same columns whatever the run did

    def test_a_v1_example_reads_under_the_v2_contract(self):
        from dclab_core.store import ChatExample, Provenance, SolverTrajectory, TrajectoryStep

        p = Provenance(entry="e", sha256="0" * 64, source="s", workspace="w", at="2026-10-20T10:00:00+00:00", consent="train")
        v1 = {"schema_version": "dclab_core.chat_example.v1", "id": "a", "model_type": "m4_workflow_llm", "split": "train", "provenance": p.model_dump(),
              "task": "t", "stage": "plan", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok", "no_reasoning": "none kept"}]}
        read = ChatExample.model_validate(v1)
        self.assertEqual((read.schema_version, read.tools), ("dclab_core.chat_example.v2", []))
        t = SolverTrajectory(id="b", model_type="m5_solver_llm", split="train", provenance=p, problem_kind="k",
                             steps=[TrajectoryStep(n=0, thought="I will look.", thought_source="visible")])
        self.assertEqual(json.loads(t.model_dump_json())["steps"][0]["thought_source"], "visible")


@unittest.skipIf(MISSING, MISSING)
class RecordedRunTests(unittest.TestCase):
    def setUp(self):
        from tests.test_models_gateway import CLEAN

        env = mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-" + "test-" + "x" * 8, "OPENAI_MODEL": "std-model"})
        env.start()
        self.addCleanup(env.stop)
        self.home = Path(tempfile.mkdtemp())
        self.log = experience.FileLog(self.home / "experience", "wtest")
        experience.install(self.log)
        self.addCleanup(experience.install, None)

    def run_intern(self, replies):
        from dclab_rnd.intern.loop import Intern
        from dclab_rnd.intern.tools import Toolbox
        from tests.test_intern import make

        class Model:
            model = "scripted"

            def __init__(self):
                self.turn = 0

            def complete(self, messages, tools):
                self.turn += 1
                if self.turn > len(replies):
                    return {"content": "All done.", "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1},
                            "assistant_message": {"role": "assistant", "content": "All done."}}
                content, tool, extra = replies[self.turn - 1]
                call = {"id": f"c{self.turn}", "name": tool, "arguments": {}}
                return {"content": content, "tool_calls": [call], "usage": {"input_tokens": 1, "output_tokens": 1}, **extra,
                        "assistant_message": {"role": "assistant", "content": content,
                                              "tool_calls": [{"id": call["id"], "type": "function", "function": {"name": tool, "arguments": "{}"}}]}}

        projects, sessions = make(str(self.home / "w"))
        intern = Intern(sessions, Toolbox(projects), Model())
        return intern.run(intern.start("Look around first", {"max_steps": 6, "max_minutes": 5})["id"])

    def run_record(self, run_id):
        runs = [e for e in self.log.read() if e.kind == "run.recorded" and e.ids.run == run_id]
        self.assertEqual(len(runs), 1)
        return runs[0].payload

    def test_every_step_of_a_model_run_has_its_thinking_slot_filled_or_marked(self):
        session = self.run_intern([("", "list_samples", {"reasoning": "Plan: see what data exists."}),
                                   ("Now I will read the evidence.", "search_evidence", {})])
        steps = session["steps"]
        self.assertEqual([(s.get("thought"), s.get("thought_source")) for s in steps[:2]],
                         [("Plan: see what data exists.", "recorded"), ("Now I will read the evidence.", "visible")])
        events = [e for e in self.log.read() if e.kind == "agent.step" and e.ids.run == session["id"]]
        self.assertEqual([e.payload.get("thought_source") for e in events][:2], ["recorded", "visible"])
        record = self.run_record(session["id"])
        numbers = record["params"]
        self.assertEqual((numbers["calls:list_samples"], numbers["calls:search_evidence"]), (1.0, 1.0))
        self.assertEqual((numbers["calls:export_notebook"], numbers["errors:export_notebook"], numbers["refused:export_notebook"]), (0.0, 0.0, 0.0))
        self.assertEqual(numbers["calls:_not_offered"], 0.0)
        self.assertEqual(len(record["params"]["tools_sha256"]), 16)
        self.assertNotIn("tool_names", record["params"])  # the event cuts a string to 300 characters: the columns name the tools
        self.assertEqual(events[-1].payload.get("tool"), None)  # the answer in words is a step with its slot too
        self.assertEqual(events[-1].payload.get("thought_source"), "visible")

    def test_a_thought_is_redacted_in_the_store_like_every_other_text(self):
        from dclab_rnd.store import shapers

        body = shapers._clean({"steps": [{"thought": "Write to john.smith@acme.com first", "thought_source": "visible", "summary": "x"}]})
        self.assertEqual(body["steps"][0]["thought"], "Write to [EMAIL] first")
        self.assertTrue(shapers._version().startswith("shapers-7:"))  # every kept session is shaped again

    def test_a_call_to_a_tool_not_offered_is_counted_and_refused(self):
        session = self.run_intern([("", "delete_everything", {})])
        numbers = self.run_record(session["id"])["params"]
        self.assertEqual((numbers["calls:_not_offered"], numbers["refused:_not_offered"], numbers["errors:_not_offered"]), (1.0, 1.0, 1.0))
        self.assertEqual(session["steps"][0].get("thought_source"), None)  # no words, no reasoning: said plainly, never invented

    def test_the_standard_plans_steps_carry_the_plans_own_sentence(self):
        from dclab_rnd.intern.loop import Intern
        from dclab_rnd.intern.tools import Toolbox
        from tests.test_intern import make

        projects, sessions = make(str(self.home / "w"))
        intern = Intern(sessions, Toolbox(projects))
        session = intern.run(intern.start("Model the quarterly widget pipeline of my company.", {"max_steps": 5, "max_minutes": 1})["id"])
        first = session["steps"][0]
        self.assertEqual((first["tool"], first.get("thought_source")), ("list_samples", "from_record"))
        self.assertTrue(first.get("thought"))
        numbers = self.run_record(session["id"])["params"]
        self.assertEqual(numbers["calls:list_samples"], 1.0)

    def test_the_tools_command_prints_the_counts(self):
        from dclab_rnd.experience import __main__ as cli

        session = self.run_intern([("", "list_samples", {"reasoning": "look"})])
        import io
        from contextlib import redirect_stdout

        out = io.StringIO()
        with mock.patch.object(cli, "_log", return_value=self.log), redirect_stdout(out):
            code = cli.main(["tools", session["id"], "--json"])
        self.assertEqual(code, 0)
        table = json.loads(out.getvalue())
        self.assertEqual(table["list_samples"], {"calls": 1.0, "errors": 0.0, "refused": 0.0})


if __name__ == "__main__":
    unittest.main()
