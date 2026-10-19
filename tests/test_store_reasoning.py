"""Package 32.10: the reasoning of every LLM example.

The gateway keeps each exchange whole (every message, never cut, the reply, its tool calls and the model's thinking) where
the workspace keeps model text, and only its shape for a purpose shown cell values; a decision's reasoning is written by
code from its own record; a teacher's reasoning is kept only when every number in it is its record's, and a planted
number refuses it; with no teacher model nothing is invented.
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


def reply(content, thinking=None):
    return {"content": content, "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1}, "thinking": thinking or [],
            "assistant_message": {"role": "assistant", "content": content}}


def gateway(home, replies):
    from dclab_rnd.models.gateway import Gateway
    from dclab_rnd.models.usage import FileUsage

    queue = list(replies)

    class Scripted:
        def complete(self, messages, tools=None, max_tokens=1800, **options):
            return queue.pop(0)

    return Gateway(FileUsage(Path(home) / "model_requests.jsonl"), transport=lambda tier, purpose: Scripted())


def record(body):
    from dclab_core.store import Provenance, StoreRecord

    prov = Provenance(entry="e", sha256="0" * 64, source="s", workspace="w", at="2026-10-20T10:00:00+00:00", consent="train")
    return StoreRecord(id="r1", kind="decisions", type="decision", at="2026-10-20T10:00:00+00:00", body=body, provenance=prov)


@unittest.skipIf(MISSING, MISSING)
class ReasoningTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.models import settings

        clean = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", "DCLAB_NO_LIVE_MODELS")} \
            | {f"DCLAB_TIER_{t.upper()}_{n}": "" for t in settings.TIERS for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")} | {"OPENAI_API_KEY": "k"}
        env = mock.patch.dict(os.environ, clean)
        env.start()
        self.addCleanup(env.stop)
        self.home = Path(tempfile.mkdtemp())

    def exchanges(self):
        return [json.loads(line) for p in sorted((self.home / "model_exchanges").rglob("*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]

    def test_the_gateway_keeps_each_exchange_whole_with_its_thinking(self):
        long = "x" * 30_000
        with mock.patch.dict(os.environ, {"DCLAB_LOG_PROMPTS": "1"}):
            gw = gateway(self.home, [reply("The answer.", [{"text": "First I read the record.", "signature": "s"}])])
            gw.client("stage_notes").complete([{"role": "user", "content": long}])
        doc = self.exchanges()[0]
        self.assertTrue(doc["whole"])
        self.assertEqual(len(doc["messages"][0]["content"]), 30_000)  # never cut
        self.assertEqual(doc["thinking"][0]["text"], "First I read the record.")
        from dclab_rnd.store import reasoning

        self.assertEqual(reasoning.recorded(doc), "First I read the record.")

    def test_a_purpose_shown_cells_or_a_workspace_keeping_no_text_keeps_its_shape_only(self):
        gw = gateway(self.home, [reply("a"), reply("b")])
        with mock.patch.dict(os.environ, {"DCLAB_LOG_PROMPTS": "0"}):
            gw.client("stage_notes").complete([{"role": "user", "content": "secret words"}])
        with mock.patch.dict(os.environ, {"DCLAB_LOG_PROMPTS": "1"}):
            gw.client("intern").complete([{"role": "user", "content": "cell 4111"}])  # the intern is shown cell values
        for doc in self.exchanges():
            self.assertFalse(doc["whole"])
            self.assertNotIn("secret words", json.dumps(doc))
            self.assertNotIn("cell 4111", json.dumps(doc))

    def test_reasoning_from_the_record_and_a_teachers_checked_against_it(self):
        from dclab_rnd.store import reasoning

        r = record({"point": "column.is_identifier", "state": {"rows": "1k-10k", "uniqueness": "all_unique"}, "options": ["yes", "no"],
                    "answers": [{"by": "rule", "value": True, "reason": "every value is unique"}], "used": "rule", "value_used": True})
        text = reasoning.from_record(r)
        self.assertIn("all_unique", text)
        self.assertIn("rule answered True because every value is unique", text)
        good = gateway(self.home, [reply("Every value is unique across the 1k-10k rows, so the column names rows rather than describing them.")])
        kept, why = reasoning.teacher(r, "yes, an identifier", gateway=good)
        self.assertIsNotNone(kept, why)
        planted = gateway(self.home, [reply("About 9,412 values repeat, so it is an identifier.")])
        kept, why = reasoning.teacher(r, "yes, an identifier", gateway=planted)
        self.assertIsNone(kept)
        self.assertIn("numbers its record does not hold", why)
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            from dclab_rnd.models.gateway import Gateway
            from dclab_rnd.models.usage import FileUsage

            none = reasoning.teacher(r, "yes", gateway=Gateway(FileUsage(self.home / "u.jsonl")))
        self.assertEqual(none, (None, "no teacher model serves here"))


if __name__ == "__main__":
    unittest.main()
