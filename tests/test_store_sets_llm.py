"""Package 32.12: the LLM sets.

A kept exchange is an M4 example in its stage, its reply carrying the model's recorded thinking as its reasoning; a Home
conversation is one too, its agent's turns saying why they carry no reasoning; a Claude Code session is one M5 trajectory
whose steps carry each thought, tool call and result in order; an example holding a sealed person's words or a benchmark
task's name never reaches train or dev; every example parses as its contract.
"""

import json
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

AT = "2026-10-20T10:00:00+00:00"


@unittest.skipIf(MISSING, MISSING)
class LlmSetsTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout, Store, curators, shapers
        from dclab_rnd.store.sets import llm

        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        store = Store(self.layout)
        exchange = {"at": AT, "model_request": "mr1", "purpose": "home_agent", "whole": True, "outcome": "ok",
                    "messages": [{"role": "system", "content": "You help."}, {"role": "user", "content": "Predict late orders."}],
                    "reply": "What happens at the prediction moment?", "thinking": [{"text": "The moment decides which columns leak.", "signature": "s"}]}
        sealed = sorted(llm.sealed_texts())[0]
        leaked = {**exchange, "model_request": "mr2", "messages": [{"role": "user", "content": sealed}], "reply": "ok", "thinking": []}
        bench = {**exchange, "model_request": "mr3", "messages": [{"role": "user", "content": "solve spaceship-titanic now"}], "reply": "ok", "thinking": []}
        tools = [{"type": "function", "function": {"name": "describe_data", "parameters": {"type": "object", "properties": {}}}}]
        # D2 / P4: a local model's exchange whose earlier turn carries the reasoning and the tool call the agent sent back
        multi = {**exchange, "model_request": "mr4", "purpose": "intern", "model": "qwen3.6:35b-a3b", "tools": tools,
                 "messages": [{"role": "system", "content": "You are the intern."}, {"role": "user", "content": "Look at the data."},
                              {"role": "assistant", "content": "", "reasoning_content": "First read the profile.",
                               "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "describe_data", "arguments": "{\"project_id\": \"p1\"}"}}]},
                              {"role": "tool", "tool_call_id": "c1", "content": "{\"rows\": 2000}"}],
                 "reply": "The table has 2000 rows.", "tool_calls": [], "thinking": [{"text": "Now I can answer."}]}
        claude_multi = {**multi, "model_request": "mr5", "model": "claude-sonnet-5-5"}
        # the same local model with no reasoning on its earlier turn: in its stage set, never in the reasoning-only set
        partial = {**multi, "model_request": "mr6", "messages": [{**m, "reasoning_content": None} if m.get("role") == "assistant" else m
                                                                 for m in multi["messages"]]}
        # an earlier call whose arguments quote a sealed person's words and an address: kept out, and redacted where kept
        called = lambda text: {**multi, "messages": [{**m, "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "ask_user", "arguments": json.dumps({"question": text})}}]} if m.get("role") == "assistant" else m for m in multi["messages"]]}
        sealed_call = {**called(sealed + " Also write to maria.lopez@acme.com"), "model_request": "mr7"}
        mailed_call = {**called("Write to maria.lopez@acme.com about the late orders"), "model_request": "mr8"}
        store.keep_log("model_requests", "dev", "model_exchanges/2026/10/20.jsonl",
                       "".join(json.dumps(x) + "\n" for x in (exchange, leaked, bench, multi, claude_multi, partial, sealed_call, mailed_call)).encode(),
                       consent="train")
        store.keep("conversations", "dev", "drafts/d1/draft.json", json.dumps({"id": "d1", "project_id": "p1", "messages": [
            {"role": "user", "text": "Which orders arrive late?"}, {"role": "agent", "text": "When is the prediction made?"}]}).encode(), consent="train")
        lines = [{"type": "user", "sessionId": "s1", "timestamp": AT, "uuid": "u1", "message": {"role": "user", "content": "Fix the failing test"}},
                 {"type": "assistant", "sessionId": "s1", "timestamp": "2026-10-20T10:00:01+00:00", "uuid": "u2", "message": {"role": "assistant", "content": [
                     {"type": "thinking", "thinking": "Run the test first."}, {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pytest"}}]}},
                 {"type": "user", "sessionId": "s1", "timestamp": "2026-10-20T10:00:02+00:00", "uuid": "u3", "message": {"role": "user", "content": [
                     {"type": "tool_result", "tool_use_id": "t1", "content": "1 failed"}]}},
                 {"type": "assistant", "sessionId": "s1", "timestamp": "2026-10-20T10:00:03+00:00", "uuid": "u4", "message": {"role": "assistant", "content": [
                     {"type": "text", "text": "The test fails on the holdout read."}]}}]
        store.keep_log("agents", "rnd", "claude_code/main/s1.jsonl", "".join(json.dumps(x) + "\n" for x in lines).encode(), consent="train")
        shapers.shape(self.layout)
        curators.curate(self.layout, run="r1")
        self.benchmarks = Path(tempfile.mkdtemp()) / "benchmarks.txt"
        self.benchmarks.write_text("# tasks\nspaceship-titanic\n", encoding="utf-8")

    def examples(self, model_type, folder, version="v1"):
        base = self.layout.sets(model_type, *folder.split("/"), version)
        return [json.loads(line) for s in ("train", "dev", "sealed") if (base / f"{s}.jsonl").exists()
                for line in (base / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_m4_and_m5_examples_carry_their_reasoning_and_contaminated_ones_stay_out(self):
        from dclab_core.store import ChatExample, SolverTrajectory
        from dclab_rnd.store.sets import llm

        with mock.patch.object(llm, "BENCHMARKS", self.benchmarks):
            report = llm.build(self.layout)
        self.assertIn("m4_workflow_llm/understand_problem", report["sets"])
        m4 = [ChatExample.model_validate(e) for e in self.examples("m4_workflow_llm", "understand_problem", "v2")]  # v2: earlier turns whole
        home = next(e for e in m4 if e.task == "home_agent" and e.messages[-1].content == "What happens at the prediction moment?")
        self.assertEqual((home.messages[-1].reasoning, home.messages[-1].reasoning_source), ("The moment decides which columns leak.", "recorded"))
        convo = next(e for e in m4 if e.task == "home_conversation")
        self.assertTrue(convo.messages[1].no_reasoning)
        text = json.dumps([e.model_dump(mode="json") for e in m4 if e.split != "sealed"])
        self.assertNotIn("spaceship-titanic", text)
        kept_out = report["kept_out"]
        for found, planted in (("a sealed person's words", "mr2"), ("a benchmark task (spaceship-titanic)", "mr3")):
            in_sets = [e for e in m4 if planted in e.provenance.entry or planted in e.model_dump_json()]
            self.assertTrue(any(found in k for k in kept_out) or all(e.split == "sealed" for e in in_sets), found)  # never in train or dev
        m5 = [SolverTrajectory.model_validate(e) for e in self.examples("m5_solver_llm", "trajectories/development", "v2")]  # v2: each session by its own id
        self.assertEqual(len(m5), 1)
        steps = m5[0].steps
        self.assertEqual([(s.tool, s.thought, s.result) for s in steps], [("Bash", "Run the test first.", "1 failed"),
                                                                          (None, None, "The test fails on the holdout read.")])
        self.assertTrue(steps[1].no_thought)
        self.assertEqual(m5[0].problem["request"], "Fix the failing test")

    def test_every_home_turn_has_its_reasoning_and_its_source(self):
        # the owner (2026-10-28): every training example carries thinking and reasoning, each labelled by where it came from
        from dclab_core.store import ChatExample
        from dclab_rnd.store import Store, curators, shapers
        from dclab_rnd.store.sets import llm

        store = Store(self.layout)
        exchange = {"at": "2026-10-28T23:00:00+00:00", "model_request": "mr9", "purpose": "home_agent", "whole": True, "outcome": "ok",
                    "model": "claude-sonnet-5-5",
                    "messages": [{"role": "user", "content": "A weekly report of refunds."}], "reply": "Which numbers should it show?",
                    "thinking": [{"text": "A report needs its numbers first.", "signature": "s"}]}
        empty = {**exchange, "model_request": "mr10", "thinking": [{"text": "", "signature": "s"}]}  # the default display: no text
        store.keep_log("model_requests", "dev", "model_exchanges/2026/10/28.jsonl", (json.dumps(exchange) + "\n" + json.dumps(empty) + "\n").encode(),
                       consent="train")
        event = lambda mid, mr: {"schema_version": "dclab_core.experience_event.v13", "id": "ev" + mid, "at": exchange["at"], "seq": 1,
                                 "ids": {"conversation": "d2", "model_request": mr}, "actor": "agent", "kind": "chat.agent",
                                 "payload": {"text": "x", "kind": "text", "message": mid}}
        store.keep_log("sessions", "dev", "experience/2026/10/28.jsonl",
                       (json.dumps(event("m2", "mr9")) + "\n" + json.dumps(event("m3", "mr10")) + "\n").encode(), consent="train")
        draft = {"id": "d2", "created": "2026-10-20T09:00:00+00:00", "intent": {"track": "reporting"}, "questions": [{"id": "q1", "field": "refresh", "why": "How often it updates sets how it is built."}],
                 "messages": [{"role": "user", "text": "A weekly report of refunds."},
                              {"role": "agent", "id": "m2", "by": "model", "text": "Which numbers should it show?"},
                              {"role": "agent", "id": "m3", "by": "model", "text": "Noted."},
                              {"role": "agent", "id": "m4", "kind": "question", "text": "How often?", "question": {"id": "q1"}},
                              {"role": "agent", "id": "m5", "kind": "analysis", "text": "Your data is ready."},
                              {"role": "agent", "id": "m6", "kind": "text", "text": "An old turn."},
                              {"role": "agent", "id": "m7", "kind": "text", "text": "A turn after the fix.", "at": "2026-10-29T08:00:00+00:00"}]}
        store.keep("conversations", "dev", "drafts/d2/draft.json", json.dumps(draft).encode(), consent="train")
        shapers.shape(self.layout)
        curators.curate(self.layout, run="r2")
        with mock.patch.object(llm, "BENCHMARKS", self.benchmarks):
            llm.build(self.layout)
        convo = next(ChatExample.model_validate(e) for e in self.examples("m4_workflow_llm", "home_reporting", "v1"))
        turns = {m.content: m for m in convo.messages if m.role == "assistant"}
        self.assertEqual((turns["Which numbers should it show?"].reasoning, turns["Which numbers should it show?"].reasoning_source),
                         ("A report needs its numbers first.", "recorded"))  # the model's own, joined through its exchange
        self.assertIn("thinking was not kept", turns["Noted."].no_reasoning)  # an empty block is no thinking
        self.assertEqual(turns["How often?"].reasoning_source, "from_record")
        self.assertIn("refresh", turns["How often?"].reasoning)
        self.assertIn("How often it updates", turns["How often?"].reasoning)  # the rules' own stated reason
        self.assertEqual(turns["Your data is ready."].reasoning_source, "from_record")
        self.assertTrue(turns["An old turn."].no_reasoning.startswith("lost before 2026-10-28"))
        self.assertTrue(turns["A turn after the fix."].no_reasoning.startswith("not kept"))  # dated by the turn's own time
        self.assertEqual(convo.check["track"], "reporting")
        counts = json.loads((self.layout.sets("m4_workflow_llm", "home_reporting", "v1") / "REASONING.json").read_text(encoding="utf-8"))
        self.assertEqual(counts, {"recorded": 1, "from_record": 2, "lost": 2, "none": 1})  # the owner sees the gap, per source
        from dclab_rnd.store import catalog

        row = next(r for r in catalog._sets(self.layout)["m4_workflow_llm"] if r["name"] == "home_reporting")
        self.assertEqual(row["reasoning"], counts)
        silver = [ChatExample.model_validate(e) for e in self.examples("m4_workflow_llm", "reasoning/silver_frontier", "v1")]
        self.assertIn("A report needs its numbers first.", [e.messages[-1].reasoning for e in silver])  # Claude's thinking, silver
        self.assertTrue(all(e.weight == 0.5 for e in silver))

    def test_a_question_in_a_model_run_conversation_is_never_credited_to_the_rules(self):
        from dclab_rnd.store.sets import llm

        q = {"id": "q1", "field": "target", "why": "The model's own words."}
        turn = {"role": "agent", "kind": "question", "text": "What is predicted?", "question": {"id": "q1"}}
        self.assertEqual(llm.home_turn(turn, {"q1": q}, {}, "2026-10-20", model_mode=False)["reasoning_source"], "from_record")
        self.assertIn("no_reasoning", llm.home_turn(turn, {"q1": q}, {}, "2026-10-20", model_mode=True))
        critic = {**q, "source": "critic"}
        self.assertIn("no_reasoning", llm.home_turn(turn, {"q1": critic}, {}, "2026-10-20"))

    def test_every_purpose_but_embedding_has_a_stage(self):
        from dclab_rnd.models import settings
        from dclab_rnd.store.sets import llm

        self.assertEqual(sorted(set(settings.PURPOSES) - set(llm.STAGES) - set(settings.NO_THINKING) - llm.NOT_WORKFLOW), [])  # 35.5: every embedding purpose
        self.assertFalse(llm.NOT_WORKFLOW & set(llm.STAGES))  # 2026-10-30: the routing labeller's exchanges are no m4 examples

    def test_earlier_turns_keep_their_reasoning_and_calls_and_the_reasoning_only_set_takes_clean_whole_ones(self):
        """D2 / P4: the local model's earlier turn keeps the reasoning and the call it sent back, the tool turn its call id, the
        example its tools; the reasoning-only set holds exchanges whose every assistant turn is recorded, by a clean model only."""
        from dclab_core.store import ChatExample
        from dclab_rnd.store.sets import llm

        with mock.patch.object(llm, "BENCHMARKS", self.benchmarks):
            llm.build(self.layout)
        stage = {e["id"]: ChatExample.model_validate(e) for e in self.examples("m4_workflow_llm", "plan", "v2")}
        local = next(e for e in stage.values() if e.check.get("model") == "qwen3.6:35b-a3b")
        earlier, tool, reply = local.messages[2], local.messages[3], local.messages[4]
        self.assertEqual((earlier.reasoning, earlier.reasoning_source), ("First read the profile.", "recorded"))
        self.assertEqual(earlier.tool_calls, [{"id": "c1", "name": "describe_data", "arguments": {"project_id": "p1"}}])  # an object, not text
        self.assertEqual((tool.role, tool.tool_call_id), ("tool", "c1"))
        self.assertEqual((reply.reasoning, reply.reasoning_source), ("Now I can answer.", "recorded"))
        self.assertEqual(local.tools[0]["function"]["name"], "describe_data")
        reasoning = {e["id"]: ChatExample.model_validate(e) for e in self.examples("m4_workflow_llm", "reasoning")}
        self.assertIn(local.id, reasoning)
        self.assertFalse(any(e.check.get("model") == "claude-sonnet-5-5" for e in reasoning.values()))  # a provider's model: silver until D11
        self.assertFalse(any(e.task == "home_conversation" for e in reasoning.values()))  # its turns carry no reasoning
        self.assertFalse(any(e.task == "home_agent" for e in reasoning.values()))  # recorded, but its exchange names no model: not clean
        partial = next(e for e in stage.values() if e.check.get("model") == "qwen3.6:35b-a3b" and e.messages[2].no_reasoning)
        self.assertNotIn(partial.id, reasoning)  # its earlier turn carried no reasoning: the stage set only
        # an earlier call's arguments: redacted where kept, and a sealed person's words keep the exchange out of train and dev
        mailed = next(e for e in stage.values() if any("about the late orders" in json.dumps(m.tool_calls) for m in e.messages))
        self.assertIn("[EMAIL]", json.dumps(mailed.messages[2].tool_calls))
        self.assertNotIn("maria.lopez", json.dumps(mailed.model_dump(mode="json")))
        self.assertFalse(any("Also write to" in json.dumps(e.model_dump(mode="json")) for e in stage.values() if e.split != "sealed"))

    def test_an_exchange_whose_last_turn_was_thought_on_trains_that_turn(self):
        """The owner (2026-10-28): every turn a model thought on is kept for training. An agent sends its history back as text,
        so earlier turns carry no thinking in a later exchange; the final-turn set keeps the exchange for its last turn, the
        earlier turns as context (weight 0 at export), a provider's model apart as silver."""
        from dclab_core.store import ChatExample
        from dclab_rnd.store import export_chat
        from dclab_rnd.store.sets import llm

        with mock.patch.object(llm, "BENCHMARKS", self.benchmarks):
            llm.build(self.layout)
        stage = {e["id"]: ChatExample.model_validate(e) for e in self.examples("m4_workflow_llm", "plan", "v2")}
        partial = next(e for e in stage.values() if e.check.get("model") == "qwen3.6:35b-a3b" and e.messages[2].no_reasoning)
        final = {e["id"] for e in self.examples("m4_workflow_llm", "reasoning/final_turn")}
        self.assertIn(partial.id, final)  # left out of the all-turns set, kept here for its last turn
        self.assertTrue(all(llm.final_recorded(stage[i]) for i in final if i in stage))
        line = export_chat.chat_line(partial, final_only=True)
        weights = [m["weight"] for m in line["messages"] if m["role"] == "assistant"]
        self.assertEqual(weights, [0] * (len(weights) - 1) + [1])
        self.assertNotIn("weight", json.dumps([m for m in export_chat.chat_line(partial)["messages"]]))  # other sets: unchanged
        silver = [ChatExample.model_validate(e) for e in self.examples("m4_workflow_llm", "reasoning/final_turn_silver_frontier")]
        self.assertTrue(all(e.weight == 0.5 and e.check.get("model") for e in silver))


if __name__ == "__main__":
    unittest.main()
