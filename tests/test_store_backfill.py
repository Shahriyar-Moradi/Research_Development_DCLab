"""Thinking for the past (store/backfill.py; the owner, 2026-10-28): which kept calls could be asked again, never exam
material, never a sealed one, never one whose inputs the shaper changed."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core.store import Provenance, StoreRecord  # noqa: E402
from dclab_rnd.store import backfill  # noqa: E402


def record(rid, body, workspace="dev", consent="train", split="train"):
    return StoreRecord(id=rid, kind="model_requests", type="exchange", at="2026-10-20T10:00:00+00:00", body=body, split=split,
                       provenance=Provenance(workspace=workspace, source="model_exchanges/2026/10/20.jsonl", entry=f"e:{rid}",
                                             at="2026-10-20T10:00:00+00:00", sha256="0" * 64, consent=consent))


BASE = {"whole": True, "purpose": "home_agent", "messages": [{"role": "user", "content": "Predict late orders."}], "reply": "When?",
        "usage": {"input_tokens": 1000}}


class ReplayableTests(unittest.TestCase):
    def test_only_whole_calls_as_sent_without_thinking_text_and_never_exam_material(self):
        rows = [record("a", {**BASE}),
                record("b", {**BASE, "thinking": [{"text": "kept"}]}),
                record("c", {**BASE, "thinking": [{"text": "", "signature": "s"}]}),  # an empty block is no thinking
                record("d", {**BASE, "purpose": "journey_writer"}),  # writes evaluation people
                record("e", {**BASE}, split="sealed"),  # a sealed original is never re-asked
                record("f", {**BASE, "whole": False}),
                record("g", {**BASE}, consent="improve"),
                record("h", {**BASE, "messages": [{"role": "user", "content": "[a table of 40 lines removed]"}]})]  # not as sent
        self.assertEqual([r.id for r in backfill.replayable(rows)], ["a", "c"])
        self.assertGreater(backfill.estimate(backfill.replayable(rows)), 0)


class SameDecisionTests(unittest.TestCase):
    def test_the_same_decision_in_other_words_agrees_another_decision_never(self):
        rec = lambda field, value: {"id": "c", "function": {"name": "record", "arguments": f'{{"field": "{field}", "value": "{value}"}}'}}  # noqa: E731
        orig = {"reply": "", "tool_calls": [rec("costs", "A retention call costs about 5 euros; losing a customer costs about 300 euros.")]}
        same = {"reply": "", "tool_calls": [rec("costs", "Each retention call costs about 5 euros; losing a customer costs about 300 euros.")]}
        other_field = {"reply": "", "tool_calls": [rec("target", "A retention call costs about 5 euros; losing a customer costs about 300 euros.")]}
        more = {"reply": "", "tool_calls": [*same["tool_calls"], {"function": {"name": "ask_user", "arguments": "{}"}}]}
        self.assertTrue(backfill.same_decision(orig, same))
        self.assertFalse(backfill.same_decision(orig, other_field))
        self.assertFalse(backfill.same_decision(orig, more))  # another action taken
        self.assertTrue(backfill.same_decision({"reply": '{"target": {"status": "stated", "value": "Whether a customer will leave within the next month"}}'},
                                               {"reply": '{"target": {"status": "stated", "value": "Whether a customer leaves within the next month"}}'}))
        self.assertFalse(backfill.same_decision({"reply": '{"target": {"status": "stated", "value": "x"}}'},
                                                {"reply": '{"target": {"status": "inferred", "value": "x"}}'}))


class ReplayTests(unittest.TestCase):
    """The paid replay, with a scripted model: only an agreeing replay's thinking explains the original, labelled replayed."""

    def setUp(self):
        import json
        import os
        import tempfile
        from unittest import mock

        from dclab_rnd.store.layout import Layout
        self.home = Path(tempfile.mkdtemp())
        self.layout = Layout(self.home / "training_records" / "store")
        runs = self.home / "agent_runs" / "model_exchanges" / "2026" / "10"
        runs.mkdir(parents=True)
        lines = [{**BASE, "model_request": "mr-a", "at": "2026-10-20T10:00:00+00:00"},
                 {**BASE, "model_request": "mr-c", "at": "2026-10-20T10:01:00+00:00", "reply": "Which column?"}]
        (runs / "20.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
        rows = [record("a", {**BASE, "model_request": "mr-a"}), record("c", {**BASE, "model_request": "mr-c", "reply": "Which column?"})]
        self.patch = mock.patch.object(backfill.shapers, "records", lambda layout, kind: rows if kind == "model_requests" else [])
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.env = mock.patch.dict(os.environ, {"DCLAB_LOG_PROMPTS": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def gateway(self, replies):
        import os

        class Client:
            spent_eur, run_limit_eur = 0.0, None

            def complete(self, messages, tools=None, **kw):
                seen.append(os.environ.get("DCLAB_LOG_PROMPTS"))  # the replay's own call keeps its shape only
                self.spent_eur = 0.01
                return replies.pop(0)
        seen = self.seen = []
        g = type("G", (), {"client": lambda self, purpose: Client()})()
        return g

    def test_an_agreeing_replay_is_kept_and_read_as_replayed_a_disagreeing_one_never(self):
        import os
        replies = [{"content": "When?", "tool_calls": [], "thinking": [{"text": "The moment is unknown, so ask it."}]},  # the same answer
                   {"content": "Something else entirely about the regions and their totals this year", "tool_calls": [],
                    "thinking": [{"text": "A different line of thought."}]}]
        out = backfill.replay(self.layout, cap_usd=1.0, yes=True, base=self.home / "agent_runs", gateway=self.gateway(replies), log=lambda *a: None)
        self.assertEqual((out["sent"], out["agreed"]), (2, 1))
        self.assertEqual(os.environ.get("DCLAB_LOG_PROMPTS"), "1")  # put back after the run
        self.assertEqual(set(self.seen), {"0"})  # never a second conversation of the same inputs
        kept = backfill.replays(self.layout)
        self.assertEqual(list(kept), ["mr-a"])  # only the agreeing one explains its original
        self.assertIn("moment is unknown", kept["mr-a"]["text"])
        from dclab_rnd.store.sets import llm
        ex = llm.from_exchange(record("a", {**BASE, "model_request": "mr-a"}), {}, kept)
        reply = ex.messages[-1]
        self.assertEqual((reply.reasoning_source, ex.weight), ("replayed", 0.5))  # never "recorded"
        again = backfill.replay(self.layout, cap_usd=1.0, yes=False, base=self.home / "agent_runs")
        self.assertEqual(again["to_replay"], 0)  # never asked twice

    def test_the_cap_stops_the_run_and_the_plan_sends_nothing(self):
        plan = backfill.replay(self.layout, cap_usd=1.0, yes=False, base=self.home / "agent_runs")
        self.assertEqual((plan["to_replay"], plan["sent"]), (2, 0))
        replies = [{"content": "When?", "tool_calls": [], "thinking": [{"text": "x"}]}, {"content": "y", "tool_calls": []}]
        out = backfill.replay(self.layout, cap_usd=0.005, yes=True, base=self.home / "agent_runs", gateway=self.gateway(replies), log=lambda *a: None)
        self.assertEqual(out["sent"], 1)  # the first call spent past the cap: the run stopped


if __name__ == "__main__":
    unittest.main()
