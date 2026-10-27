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


if __name__ == "__main__":
    unittest.main()
