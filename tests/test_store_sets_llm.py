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
        store.keep_log("model_requests", "dev", "model_exchanges/2026/10/20.jsonl", "".join(json.dumps(x) + "\n" for x in (exchange, leaked, bench)).encode(),
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

    def examples(self, model_type, folder):
        base = self.layout.sets(model_type, *folder.split("/"), "v1")
        return [json.loads(line) for s in ("train", "dev", "sealed") if (base / f"{s}.jsonl").exists()
                for line in (base / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_m4_and_m5_examples_carry_their_reasoning_and_contaminated_ones_stay_out(self):
        from dclab_core.store import ChatExample, SolverTrajectory
        from dclab_rnd.store.sets import llm

        with mock.patch.object(llm, "BENCHMARKS", self.benchmarks):
            report = llm.build(self.layout)
        self.assertIn("m4_workflow_llm/understand_problem", report["sets"])
        m4 = [ChatExample.model_validate(e) for e in self.examples("m4_workflow_llm", "understand_problem")]
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
        m5 = [SolverTrajectory.model_validate(e) for e in self.examples("m5_solver_llm", "trajectories/development")]
        self.assertEqual(len(m5), 1)
        steps = m5[0].steps
        self.assertEqual([(s.tool, s.thought, s.result) for s in steps], [("Bash", "Run the test first.", "1 failed"),
                                                                          (None, None, "The test fails on the holdout read.")])
        self.assertTrue(steps[1].no_thought)
        self.assertEqual(m5[0].problem["request"], "Fix the failing test")


if __name__ == "__main__":
    unittest.main()
