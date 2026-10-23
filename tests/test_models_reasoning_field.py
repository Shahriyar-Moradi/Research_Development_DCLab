"""D2 (2026-10-23): a local open model's reasoning is read from every reply, kept, and sent back on every earlier turn.

The OpenAI-compatible client reads the reply's reasoning under the field the tier names (never guessed from the model's
name), puts it in ``thinking`` (the shape the exchange log keeps) and in ``assistant_message`` under that field, sends
the tier's chat-template setting with every request (Qwen's preserve_thinking), and sends tool arguments back as
objects where the tier says so. With no field set, nothing is read or sent: the client behaves as before.
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


def message(content="", tool_calls=None, **fields):
    return type("M", (), {"content": content, "tool_calls": tool_calls or [], **fields})()


def call(name, arguments):
    return type("TC", (), {"id": "c1", "function": type("F", (), {"name": name, "arguments": arguments})()})()


def response(msg):
    usage = type("U", (), {"prompt_tokens": 7, "completion_tokens": 3})()
    return type("R", (), {"choices": [type("C", (), {"message": msg, "finish_reason": "stop"})()], "usage": usage})()


class ClientTests(unittest.TestCase):
    def fake(self, create, **options):
        from dclab_rnd.models import client as mc

        c = mc.ChatClient(model="qwen-x", base_url="http://127.0.0.1:8080/v1", api_key="local", **options)
        c._client = type("O", (), {"chat": type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(create)})()})()})()
        self.addCleanup(mc.TOOLS_WITHOUT_REASONING.discard, "qwen-x")
        from dclab_rnd.models import pause

        self.addCleanup(pause.clear)
        return c

    def test_the_reasoning_is_read_kept_and_sent_back_under_the_configured_field(self):
        sent = []

        def create(**kwargs):
            sent.append(kwargs)
            return response(message("Done.", [call("note", '{"text": "ready"}')], reasoning_content="I pick QWERTY."))

        c = self.fake(create, reasoning_field="reasoning_content", chat_template_kwargs={"preserve_thinking": True})
        out = c.complete([{"role": "user", "content": "go"}], tools=[{"type": "function", "function": {"name": "note"}}])
        self.assertEqual(out["reasoning"], "I pick QWERTY.")
        self.assertEqual(out["thinking"], [{"text": "I pick QWERTY."}])  # the shape the exchange log and the sets read
        self.assertEqual(out["assistant_message"]["reasoning_content"], "I pick QWERTY.")  # goes back with the next request
        self.assertEqual(out["tool_calls"][0]["arguments"], {"text": "ready"})
        self.assertEqual(out["assistant_message"]["tool_calls"][0]["function"]["arguments"], '{"text": "ready"}')  # the wire format by default
        self.assertEqual(sent[0]["extra_body"], {"chat_template_kwargs": {"preserve_thinking": True}})

    def test_without_a_field_nothing_is_read_or_sent(self):
        sent = []

        def create(**kwargs):
            sent.append(kwargs)
            return response(message("Done.", reasoning_content="hidden"))

        c = self.fake(create)
        out = c.complete([{"role": "user", "content": "go"}])
        self.assertNotIn("reasoning", out)
        self.assertNotIn("thinking", out)
        self.assertNotIn("reasoning_content", out["assistant_message"])
        self.assertNotIn("extra_body", sent[0])
        self.assertEqual(out["reasoning_fields_seen"], ["reasoning_content"])  # named for the tier's setting, never kept or sent

    def test_a_call_without_arguments_runs_with_none(self):
        def create(**kwargs):
            return response(message("", [call("note", None)]))

        out = self.fake(create).complete([{"role": "user", "content": "go"}], tools=[{"type": "function", "function": {"name": "note"}}])
        self.assertEqual(out["tool_calls"][0]["arguments"], {})

    def test_the_other_field_name_and_object_arguments(self):
        def create(**kwargs):
            return response(message("", [call("note", '{"text": "ready"}')], reasoning="plan: call note"))

        c = self.fake(create, reasoning_field="reasoning", tool_arguments="object")
        out = c.complete([{"role": "user", "content": "go"}], tools=[{"type": "function", "function": {"name": "note"}}])
        self.assertEqual(out["assistant_message"]["reasoning"], "plan: call note")
        self.assertEqual(out["assistant_message"]["tool_calls"][0]["function"]["arguments"], {"text": "ready"})  # a dict for Qwen's template

    def test_a_stream_joins_the_reasoning_pieces_and_sends_the_template_setting(self):
        sent = []

        def create(**kwargs):
            sent.append(kwargs)
            usage = type("U", (), {"prompt_tokens": 7, "completion_tokens": 2})()

            def chunk(delta):
                return type("K", (), {"usage": None, "choices": [type("Ch", (), {"delta": delta, "finish_reason": None})()]})()
            deltas = [type("D", (), {"content": None, "tool_calls": None, "reasoning_content": "First, "})(),
                      type("D", (), {"content": None, "tool_calls": None, "reasoning_content": "think."})(),
                      type("D", (), {"content": "hi", "tool_calls": None, "reasoning_content": None})()]
            return iter([chunk(d) for d in deltas] + [type("K", (), {"usage": usage, "choices": []})()])

        c = self.fake(create, reasoning_field="reasoning_content", chat_template_kwargs={"preserve_thinking": True})
        out = c.stream([{"role": "user", "content": "go"}])
        self.assertEqual((out["content"], out["reasoning"]), ("hi", "First, think."))
        self.assertEqual(out["assistant_message"]["reasoning_content"], "First, think.")
        self.assertEqual(sent[0]["extra_body"], {"chat_template_kwargs": {"preserve_thinking": True}})
        self.assertTrue(sent[0]["stream"])


class SettingsTests(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.models import settings

        clean = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY")} \
            | {f"DCLAB_TIER_{t.upper()}_{n}": "" for t in settings.ROUTABLE
               for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL", "REASONING_FIELD", "CHAT_TEMPLATE_KWARGS", "TOOL_ARGUMENTS")} | {"OPENAI_API_KEY": "k"}
        env = mock.patch.dict(os.environ, clean)
        env.start()
        self.addCleanup(env.stop)

    def test_the_options_are_per_tier_and_default_to_none(self):
        from dclab_rnd.models import settings

        with mock.patch.dict(os.environ, {"DCLAB_TIER_CHEAP_BASE_URL": "http://127.0.0.1:8080/v1", "DCLAB_TIER_CHEAP_MODEL": "qwen3.6",
                                          "DCLAB_TIER_CHEAP_REASONING_FIELD": "reasoning_content",
                                          "DCLAB_TIER_CHEAP_CHAT_TEMPLATE_KWARGS": '{"preserve_thinking": true}', "DCLAB_TIER_CHEAP_TOOL_ARGUMENTS": "object"}):
            cheap, standard = settings.tier("cheap"), settings.tier("standard")
        self.assertEqual((cheap["reasoning_field"], cheap["chat_template_kwargs"], cheap["tool_arguments"]),
                         ("reasoning_content", {"preserve_thinking": True}, "object"))
        self.assertEqual((standard["reasoning_field"], standard["chat_template_kwargs"], standard["tool_arguments"]), (None, {}, "json"))
        shown = settings.public("cheap")
        self.assertNotIn("api_key", shown)
        self.assertEqual(shown["reasoning_field"], None)  # the patch is over: the default again

    def test_a_wrong_value_is_a_settings_problem(self):
        from dclab_rnd.models import settings
        from dclab_rnd.settings import Settings

        with mock.patch.dict(os.environ, {"DCLAB_TIER_CHEAP_REASONING_FIELD": "thoughts"}):
            with self.assertRaises(ValueError):
                settings.options("cheap")
            self.assertTrue(any("REASONING_FIELD" in p for p in Settings.load(tempfile.mkdtemp()).problems()))
        with mock.patch.dict(os.environ, {"DCLAB_TIER_CHEAP_CHAT_TEMPLATE_KWARGS": "not json"}):
            with self.assertRaises(ValueError):
                settings.options("cheap")

    def test_the_gateway_passes_the_options_to_the_transport(self):
        from dclab_rnd.models import gateway as gw

        with mock.patch.dict(os.environ, {"DCLAB_TIER_CHEAP_BASE_URL": "http://127.0.0.1:8080/v1", "DCLAB_TIER_CHEAP_MODEL": "qwen3.6",
                                          "DCLAB_TIER_CHEAP_REASONING_FIELD": "reasoning", "DCLAB_TIER_CHEAP_TOOL_ARGUMENTS": "object"}):
            from dclab_rnd.models import settings

            transport = gw._transport(settings.tier("cheap"), settings.purpose("intern"))
        self.assertEqual((transport.reasoning_field, transport.tool_arguments, transport.chat_template_kwargs), ("reasoning", "object", {}))


def scripted_gateway(home, replies):
    from dclab_rnd.models.gateway import Gateway
    from dclab_rnd.models.usage import FileUsage

    queue = list(replies)

    class Scripted:
        def complete(self, messages, tools=None, max_tokens=1800, **options):
            return queue.pop(0)

    return Gateway(FileUsage(Path(home) / "model_requests.jsonl"), transport=lambda tier, purpose: Scripted())


class ExchangeLogTests(unittest.TestCase):
    def setUp(self):
        try:
            from dclab_rnd import experience
        except ImportError as error:  # pragma: no cover
            self.skipTest(str(error))
        if experience.contract is None:
            self.skipTest("dclab_core is not installed here")
        clean = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", "DCLAB_NO_LIVE_MODELS")} \
            | {"OPENAI_API_KEY": "k", "DCLAB_LOG_PROMPTS": "1"}
        env = mock.patch.dict(os.environ, clean)
        env.start()
        self.addCleanup(env.stop)
        self.home = Path(tempfile.mkdtemp())

    def test_the_exchange_keeps_the_reasoning_and_the_tools_offered(self):
        tools = [{"type": "function", "function": {"name": "note", "parameters": {"type": "object", "properties": {}}}}]
        reply = {"content": "", "tool_calls": [{"id": "c1", "name": "note", "arguments": {"text": "ready"}}], "usage": {"input_tokens": 1, "output_tokens": 1},
                 "reasoning": "I will call note.", "thinking": [{"text": "I will call note."}],
                 "assistant_message": {"role": "assistant", "content": "", "reasoning_content": "I will call note."}}
        gw = scripted_gateway(self.home, [reply])
        gw.client("home_agent").complete([{"role": "user", "content": "go"}], tools)  # a purpose that keeps text
        docs = [json.loads(line) for p in sorted((self.home / "model_exchanges").rglob("*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]
        doc = docs[0]
        self.assertTrue(doc["whole"])
        self.assertEqual(doc["thinking"], [{"text": "I will call note."}])
        self.assertEqual(doc["tools"], tools)
        self.assertEqual(doc["tool_names"], ["note"])
        self.assertEqual(len(doc["tools_sha256"]), 16)

    def test_a_shape_only_exchange_keeps_the_tool_names_but_no_schemas_or_text(self):
        tools = [{"type": "function", "function": {"name": "describe_data", "parameters": {"type": "object", "properties": {}}}}]
        reply = {"content": "the cells", "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1}, "thinking": [{"text": "hm"}],
                 "assistant_message": {"role": "assistant", "content": "the cells"}}
        gw = scripted_gateway(self.home, [reply])
        gw.client("intern").complete([{"role": "user", "content": "go"}], tools)  # a purpose shown cell values: its shape only
        doc = [json.loads(line) for p in sorted((self.home / "model_exchanges").rglob("*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()][0]
        self.assertFalse(doc["whole"])
        self.assertEqual(doc["tool_names"], ["describe_data"])
        self.assertNotIn("tools", doc)
        self.assertEqual(doc["messages"], 1)  # a count, never the text


if __name__ == "__main__":
    unittest.main()
