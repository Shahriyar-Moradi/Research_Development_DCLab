"""Prompt caching on Anthropic's Messages API (platform.claude.com/docs/en/build-with-claude/prompt-caching).

The stable prefix of every request is marked for the cache: the last tool (so every tool definition) and the system
prompt. A conversation already under way (an assistant turn in it, so the request will be sent again with more) also
gets the API's automatic breakpoint, which moves to the end of the conversation; a single request with a new question
does not, so it never pays the write premium on text that will not be read again. DCLAB_PROMPT_CACHE_TTL=1h asks for
the hour-long cache, DCLAB_PROMPT_CACHE=off sends no breakpoint. The reply's cache reads and writes are read from its
usage, priced at their own rates (a Sonnet 5.5 read is 0.05 of the input price, a write 1.25 or 2), kept in the usage
log, and the estimate made before sending stays an upper bound.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.models import prices  # noqa: E402
from dclab_rnd.models.anthropic_client import AnthropicClient, from_message  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": f"tool_{i}", "description": "d", "parameters": {"type": "object", "properties": {}}}} for i in range(3)]
SYSTEM = {"role": "system", "content": "You are DCLab's planner. " * 80}
ASK = {"role": "user", "content": "Plan the next stage."}


def request(messages, tools=TOOLS, **env):
    with mock.patch.dict(os.environ, env):
        return AnthropicClient(api_key="test")._request(messages, tools, 1000, {})


class BreakpointTest(unittest.TestCase):
    def test_the_tools_and_the_system_prompt_are_cached_and_a_single_request_gets_no_moving_breakpoint(self):
        kw = request([SYSTEM, ASK])
        self.assertEqual(kw["system"][-1]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(kw["system"][-1]["text"], SYSTEM["content"])
        self.assertEqual(kw["tools"][-1]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", kw["tools"][0])
        self.assertNotIn("cache_control", (kw.get("extra_body") or {}))

    def test_a_conversation_under_way_also_gets_the_automatic_breakpoint(self):
        turns = [SYSTEM, ASK, {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "tool_0", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": "c1", "content": "a result"}]
        kw = request(turns)
        self.assertEqual(kw["extra_body"]["cache_control"], {"type": "ephemeral"})
        marked = sum(1 for b in kw["system"] if "cache_control" in b) + sum(1 for t in kw["tools"] if "cache_control" in t) + 1
        self.assertLessEqual(marked, 4)  # the API takes at most four breakpoints

    def test_the_hour_long_cache_and_the_off_switch(self):
        kw = request([SYSTEM, ASK], DCLAB_PROMPT_CACHE_TTL="1h")
        self.assertEqual(kw["system"][-1]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(kw["tools"][-1]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        off = request([SYSTEM, ASK], DCLAB_PROMPT_CACHE="off")
        self.assertEqual(off["system"], SYSTEM["content"])
        self.assertTrue(all("cache_control" not in t for t in off["tools"]))
        with self.assertRaises(ValueError):
            request([SYSTEM, ASK], DCLAB_PROMPT_CACHE_TTL="2h")

    def test_a_structured_output_keeps_its_format_beside_the_breakpoint(self):
        turns = [SYSTEM, ASK, {"role": "assistant", "content": "first"}, {"role": "user", "content": "again"}]
        with mock.patch.dict(os.environ, {}):
            kw = AnthropicClient(api_key="test")._request(turns, None, 1000, {"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}})
        self.assertEqual(kw["extra_body"]["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(kw["extra_body"]["cache_control"], {"type": "ephemeral"})


class UsageAndPriceTest(unittest.TestCase):
    def reply(self, read=0, write_5m=0, write_1h=0, fresh=100, out=50):
        usage = SimpleNamespace(input_tokens=fresh, output_tokens=out, cache_read_input_tokens=read, cache_creation_input_tokens=write_5m + write_1h,
                                cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=write_5m, ephemeral_1h_input_tokens=write_1h))
        return from_message(SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], usage=usage, stop_reason="end_turn"))

    def test_the_replys_cache_reads_and_writes_are_read_from_its_usage(self):
        usage = self.reply(read=800, write_5m=60, write_1h=40)["usage"]
        self.assertEqual(usage["input_tokens"], 1000)  # every input token, cached or not
        self.assertEqual((usage["cache_read_tokens"], usage["cache_write_tokens"], usage["cache_write_1h_tokens"]), (800, 100, 40))

    def test_cached_tokens_are_priced_at_their_own_rates(self):
        # Sonnet 5.5: 2 a million in, 10 out; a read 0.05 of the input price, a 5-minute write 1.25, an hour's write 2
        eur, basis = prices.cost("claude-sonnet-5-5", False, 1000, 50, cache_read=800, cache_write=100, cache_write_1h=40)
        expected = (100 * 2 + 800 * 2 * 0.05 + 60 * 2 * 1.25 + 40 * 2 * 2.0 + 50 * 10) / 1_000_000
        self.assertEqual((eur, basis), (round(expected, 6), "price"))
        self.assertEqual(prices.cost("claude-sonnet-5-5", False, 1000, 50)[0], round((1000 * 2 + 50 * 10) / 1_000_000, 6))  # no cache: as before
        self.assertLess(eur, prices.cost("claude-sonnet-5-5", False, 1000, 50)[0])

    def test_the_estimate_before_sending_stays_an_upper_bound(self):
        chars, out = 3000, 100
        with mock.patch.dict(os.environ, {"DCLAB_PROMPT_CACHE_TTL": "1h"}):
            bound = prices.estimate("claude-sonnet-5-5", False, chars, out)
        worst = prices.cost("claude-sonnet-5-5", False, chars // 3, out, cache_write=chars // 3, cache_write_1h=chars // 3)[0]
        self.assertGreaterEqual(bound, worst)

    def test_the_usage_log_keeps_the_cached_tokens(self):
        from dclab_rnd.models import Gateway
        from dclab_rnd.models.usage import FileUsage

        log = FileUsage(Path(tempfile.mkdtemp()) / "model_requests.jsonl")
        with mock.patch.dict(os.environ, {"DCLAB_LLM_PROVIDER": "anthropic"}):
            eur = Gateway(log).record("home_agent", "claude-sonnet-5-5", self.reply(read=800, write_5m=100)["usage"], 1.0, 1, "ok")
        row = log.recent(1)[0]
        self.assertEqual((row["input_tokens"], row["cache_read_tokens"], row["cache_write_tokens"]), (1000, 800, 100))
        self.assertEqual(row["cost_eur"], eur)
        self.assertLess(eur, prices.cost("claude-sonnet-5-5", False, 1000, 50)[0])


if __name__ == "__main__":
    unittest.main()
