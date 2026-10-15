"""Anthropic's Messages API behind the gateway: the provider switch, the translation both ways, JSON requests, the
pause after a refusal that stays, and the embedding purpose left off. No request leaves this machine: the SDK client
is replaced by a recording fake."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.models import anthropic_client, gateway, prices, settings  # noqa: E402
from dclab_rnd.models.anthropic_client import AnthropicClient  # noqa: E402

PLACEHOLDER = "placeholder-not-a-real-credential"
ANTHROPIC_ENV = {"DCLAB_LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": PLACEHOLDER, "OPENAI_API_KEY": "openai-placeholder",
                 "OPENAI_MODEL": "gpt-example", "OPENAI_BASE_URL": ""}


def clean_env(**extra):
    keep = {k: v for k, v in os.environ.items() if not k.startswith(("DCLAB_TIER_", "ANTHROPIC_", "OPENAI_", "DCLAB_LLM_", "DCLAB_INTERN_MODEL"))}
    return mock.patch.dict(os.environ, {**keep, **extra}, clear=True)


class FakeMessages:
    def __init__(self, reply=None, error=None, chunks=("Hel", "lo")):
        self.reply, self.error, self.chunks, self.sent = reply, error, chunks, []

    def create(self, **kwargs):
        self.sent.append(kwargs)
        if self.error:
            raise self.error
        return self.reply

    def stream(self, **kwargs):
        self.sent.append(kwargs)
        fake = self

        class Stream:
            text_stream = iter(fake.chunks)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return fake.reply
        return Stream()


def reply(*blocks, stop="end_turn"):
    return NS(content=list(blocks), stop_reason=stop, usage=NS(input_tokens=11, output_tokens=7, cache_read_input_tokens=0, cache_creation_input_tokens=0))


def client_with(messages: FakeMessages) -> AnthropicClient:
    c = AnthropicClient(model="claude-sonnet-5-5", api_key=PLACEHOLDER)
    c._client = NS(messages=messages)
    return c


class ProviderSwitchTest(unittest.TestCase):
    def test_the_switch_makes_every_tier_claude_with_the_anthropic_key_only(self):
        with clean_env(**ANTHROPIC_ENV):
            for name in settings.TIERS:
                t = settings.tier(name)
                self.assertEqual((t["provider"], t["model"], t["api_key"]), ("anthropic", "claude-sonnet-5-5", PLACEHOLDER), name)
                self.assertEqual(t["base_url"], "https://api.anthropic.com")
            self.assertEqual(settings.public("standard")["provider"], "anthropic")
            self.assertNotIn("api_key", settings.public("standard"))

    def test_a_named_model_and_a_tier_of_its_own_are_kept(self):
        with clean_env(**ANTHROPIC_ENV, DCLAB_TIER_CHEAP_MODEL="claude-haiku-5-5",
                       DCLAB_TIER_STRONG_BASE_URL="http://127.0.0.1:11434/v1", DCLAB_TIER_STRONG_MODEL="qwen2.5-coder:1.5b"):
            self.assertEqual(settings.tier("cheap")["model"], "claude-haiku-5-5")
            strong = settings.tier("strong")
            self.assertEqual((strong["provider"], strong["api_key"]), ("openai", "local"))  # the Anthropic key never goes to another host

    def test_anthropics_openai_compatible_url_stays_openai_and_a_claude_tier_gets_a_claude_model(self):
        with clean_env(OPENAI_API_KEY="openai-placeholder", OPENAI_BASE_URL="https://api.anthropic.com/v1", OPENAI_MODEL="claude-sonnet-5-5"):
            t = settings.tier("standard")
            self.assertEqual((t["provider"], t["api_key"]), ("openai", "openai-placeholder"))
        with clean_env(OPENAI_API_KEY="openai-placeholder", ANTHROPIC_API_KEY=PLACEHOLDER, DCLAB_TIER_STRONG_BASE_URL="https://api.anthropic.com"):
            strong = settings.tier("strong")
            self.assertEqual((strong["provider"], strong["model"], strong["api_key"]), ("anthropic", "claude-sonnet-5-5", PLACEHOLDER))
            self.assertEqual(settings.tier("standard")["provider"], "openai")

    def test_a_dated_snapshot_takes_its_alias_price_and_nothing_else_does(self):
        self.assertEqual(prices.price_of("claude-sonnet-5-5-20261001").output, 10.0)
        self.assertIsNone(prices.price_of("claude-sonnet-5-5-preview"))

    def test_without_the_switch_nothing_changes(self):
        with clean_env(OPENAI_API_KEY="openai-placeholder", OPENAI_MODEL="gpt-example"):
            t = settings.tier("standard")
            self.assertEqual((t["provider"], t["model"], t["api_key"]), ("openai", "gpt-example", "openai-placeholder"))

    def test_the_gateway_sends_through_the_messages_api_and_keeps_embeddings_off(self):
        with clean_env(**ANTHROPIC_ENV):
            self.assertIsInstance(gateway._transport(settings.tier("standard"), settings.purpose("home_agent")), AnthropicClient)
            g = gateway.Gateway(transport=lambda t, p: None)
            self.assertTrue(g.available("home_agent"))
            self.assertFalse(g.available("embedding"))

    def test_sonnet_has_a_dated_price_so_euro_caps_stop_it(self):
        price = prices.load()["claude-sonnet-5-5"]
        self.assertEqual((price.input, price.output), (2.0, 10.0))
        self.assertIsNotNone(prices.estimate("claude-sonnet-5-5", False, 3000, 1000))


class TranslationTest(unittest.TestCase):
    def test_messages_and_tools_are_translated_and_the_reply_comes_back_in_the_gateways_shape(self):
        fake = FakeMessages(reply(NS(type="text", text="Looking."), NS(type="tool_use", id="tu_1", name="profile", input={"column": "x"}), stop="tool_use"))
        out = client_with(fake).complete(
            [{"role": "system", "content": "Be careful."},
             {"role": "user", "content": "Profile x"},
             {"role": "assistant", "content": "", "tool_calls": [{"id": "tu_0", "type": "function", "function": {"name": "list", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "tu_0", "content": "x, y"},
             {"role": "user", "content": "Go on"}],
            tools=[{"type": "function", "function": {"name": "profile", "description": "d", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}}}}],
            max_tokens=None)
        sent = fake.sent[0]
        self.assertEqual(sent["system"], "Be careful.")
        self.assertEqual(sent["max_tokens"], anthropic_client.DEFAULT_MAX_TOKENS)
        self.assertEqual([t["role"] for t in sent["messages"]], ["user", "assistant", "user"])  # the tool result and the next words: one turn
        self.assertEqual(sent["messages"][1]["content"][0], {"type": "tool_use", "id": "tu_0", "name": "list", "input": {}})
        self.assertEqual(sent["messages"][2]["content"][0]["type"], "tool_result")
        self.assertEqual(sent["tools"][0]["input_schema"]["properties"]["column"]["type"], "string")
        self.assertEqual(out["tool_calls"], [{"id": "tu_1", "name": "profile", "arguments": {"column": "x"}}])
        self.assertEqual((out["finish_reason"], out["usage"]), ("tool_calls", {"input_tokens": 11, "output_tokens": 7}))
        self.assertEqual(out["assistant_message"]["tool_calls"][0]["function"]["name"], "profile")

    def test_json_requests_use_structured_outputs_or_ask_in_the_system_prompt(self):
        fake = FakeMessages(reply(NS(type="text", text="{}")))
        c = client_with(fake)
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False}
        c.complete([{"role": "user", "content": "q"}], response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}, temperature=0)
        self.assertEqual(fake.sent[0]["extra_body"]["output_config"]["format"]["schema"], schema)
        self.assertEqual(fake.sent[0]["temperature"], 0)
        c.complete([{"role": "user", "content": "q"}], response_format={"type": "json_object"})
        self.assertIn(anthropic_client.JSON_ONLY, fake.sent[1]["system"])
        self.assertNotIn("response_format", fake.sent[1])

    def test_a_stream_is_heard_as_it_is_written(self):
        fake = FakeMessages(reply(NS(type="text", text="Hello")))
        heard = []
        out = client_with(fake).stream([{"role": "user", "content": "hi"}], on_text=heard.append)
        self.assertEqual(("".join(heard), out["content"]), ("Hello", "Hello"))

    def test_a_refused_key_pauses_and_its_message_is_never_passed_on(self):
        from dclab_rnd.models import pause

        pause.clear()
        error = type("AuthenticationError", (Exception,), {})("leaked header text")
        error.status_code, error.body = 401, {"type": "error", "error": {"type": "authentication_error", "message": "leaked header text"}}
        c = client_with(FakeMessages(error=error))
        with self.assertRaises(RuntimeError) as caught:
            c.complete([{"role": "user", "content": "q"}])
        self.assertIn("the provider refused the key", str(caught.exception))
        self.assertNotIn("leaked", str(caught.exception))
        with self.assertRaises(RuntimeError) as again:
            c.complete([{"role": "user", "content": "q"}])
        self.assertIn("ModelPaused", str(again.exception))
        pause.clear()

    def test_no_credit_and_an_overload_are_told_apart(self):
        from dclab_rnd.models import client as base

        credit = type("BadRequestError", (Exception,), {})("x")
        credit.status_code, credit.body = 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "Your credit balance is too low"}}
        self.assertEqual(base.failure(credit), ("the model account has no credit left", True))
        busy = type("APIStatusError", (Exception,), {})("x")
        busy.status_code, busy.body = 200, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}  # inside a stream
        self.assertEqual(base.failure(busy), ("the provider had an error", False))

    def test_a_stream_brings_its_tool_calls_and_several_results_join_one_turn(self):
        fake = FakeMessages(reply(NS(type="tool_use", id="tu_9", name="profile", input={}), stop="tool_use"), chunks=())
        out = client_with(fake).stream(
            [{"role": "user", "content": "go"},
             {"role": "assistant", "content": "Two.", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                                                                      {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{\"x\": 1}"}}]},
             {"role": "tool", "tool_call_id": "a", "content": "1"}, {"role": "tool", "tool_call_id": "b", "content": "2"}],
            tools=[{"type": "function", "function": {"name": "profile", "parameters": {"type": "object", "properties": {}}}}])
        sent = fake.sent[0]["messages"]
        self.assertEqual([b["type"] for b in sent[1]["content"]], ["text", "tool_use", "tool_use"])
        self.assertEqual([(b["type"], b["tool_use_id"]) for b in sent[2]["content"]], [("tool_result", "a"), ("tool_result", "b")])
        self.assertEqual(sent[1]["content"][2]["input"], {"x": 1})
        self.assertEqual((out["finish_reason"], out["tool_calls"][0]["name"]), ("tool_calls", "profile"))

    def test_no_limit_is_the_limit_the_cost_estimate_assumes(self):
        fake = FakeMessages(reply(NS(type="text", text="{}")))
        client_with(fake).complete([{"role": "user", "content": "q"}], max_tokens=None)
        self.assertEqual(fake.sent[0]["max_tokens"], prices.DEFAULT_OUTPUT_ESTIMATE)

    def test_embeddings_are_refused(self):
        with self.assertRaises(RuntimeError):
            client_with(FakeMessages()).embed(["a"])


if __name__ == "__main__":
    unittest.main()
