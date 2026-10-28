"""Every agent's model calls ask for the model's thinking and keep it for training (the owner, 2026-10-28: "what I want is
the model's thinking kept for training"; plan ~/.claude/plans/deep-meandering-hippo.md, Part A).

Measured that night: 2,095 saved model calls, of which 723 thought and kept only "thought: true"; only 15 of 42 purposes
asked for thinking; a process that did not load .env kept no text; a gateway with no workspace kept nothing; a shadow's
thinking was thrown away; a local model's reasoning was dropped unless its tier named the field.
"""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import experience  # noqa: E402
from dclab_rnd.agents import harness  # noqa: E402
from dclab_rnd.models import client as mc  # noqa: E402
from dclab_rnd.models import gateway as gw  # noqa: E402
from dclab_rnd.models import settings  # noqa: E402

CLEAN = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY",
                         "DCLAB_NO_LIVE_MODELS", "DCLAB_THINKING")} | {"OPENAI_API_KEY": "k"}


def reply(i=0):
    return {"content": f"answer {i}", "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1},
            "thinking": [{"text": f"why {i}"}], "assistant_message": {"role": "assistant", "content": f"answer {i}"}}


class Scripted:
    def __init__(self, replies):
        self.replies, self.thinking = list(replies), None

    def complete(self, messages, tools=None, max_tokens=1800, **options):
        return self.replies.pop(0)


def exchanges(home):
    return [json.loads(line) for p in sorted(Path(home).rglob("model_exchanges/**/*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]


class EveryPurposeThinks(unittest.TestCase):
    def test_every_purpose_asks_for_thinking_but_the_listed_exceptions(self):
        with mock.patch.dict(os.environ, {"DCLAB_THINKING": ""}):
            for name in settings.PURPOSES:
                want = None if name in settings.NO_THINKING else "adaptive"
                self.assertEqual(settings.thinking_for(name), want, name)
        for name, why in settings.NO_THINKING.items():  # a short list, each with its written reason
            self.assertIn(name, settings.PURPOSES)
            self.assertGreater(len(why), 15, name)
        self.assertLessEqual(len(settings.NO_THINKING), 2)

    def test_every_agent_and_model_use_thinks(self):
        purposes = set(harness.MODEL_USES) | {p for a in harness.AGENTS.values() for p in a.purposes}
        with mock.patch.dict(os.environ, {"DCLAB_THINKING": ""}):
            for name in sorted(purposes):
                if name in settings.PURPOSES and name not in settings.NO_THINKING:
                    self.assertEqual(settings.thinking_for(name), "adaptive", name)

    def test_the_override_still_works(self):
        with mock.patch.dict(os.environ, {"DCLAB_THINKING": "off"}):
            self.assertIsNone(settings.thinking_for("home_agent"))
        with mock.patch.dict(os.environ, {"DCLAB_THINKING": "home_agent"}):
            self.assertEqual(settings.thinking_for("home_agent"), "adaptive")
            self.assertIsNone(settings.thinking_for("stage_notes"))


class KeptByDefault(unittest.TestCase):
    def test_text_is_kept_unless_switched_off(self):
        env = dict(os.environ)
        env.pop("DCLAB_LOG_PROMPTS", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(experience.model_text_allowed())  # a process that never loaded .env keeps it too
        for off in ("", "0", "off", "false"):
            with mock.patch.dict(os.environ, {"DCLAB_LOG_PROMPTS": off}):
                self.assertFalse(experience.model_text_allowed(), off)
        with mock.patch.dict(os.environ, {"DCLAB_LOG_PROMPTS": "1"}):
            self.assertTrue(experience.model_text_allowed())


class GatewayKeeps(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {**CLEAN, "DCLAB_LOG_PROMPTS": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.home = Path(tempfile.mkdtemp())

    def test_a_live_reply_with_no_workspace_is_kept_in_the_agent_home(self):
        transport = Scripted([reply()])
        with mock.patch.object(gw, "_transport", lambda tier, purpose: transport), \
                mock.patch.dict(os.environ, {"DCLAB_AGENT_HOME": str(self.home)}), \
                mock.patch.object(experience, "current_log", lambda: None):
            g = gw.Gateway(transport=gw._transport)  # the real transport: a live reply
            g.client("stage_notes").complete([{"role": "user", "content": "go"}], None)
        docs = exchanges(self.home)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["thinking"], [{"text": "why 0"}])

    def test_home_business_keeps_its_thinking(self):  # H11b: a new purpose's thinking is stored (rule 8), redacted: it sees group names
        transport = Scripted([reply(7)])
        with mock.patch.object(gw, "_transport", lambda tier, purpose: transport), \
                mock.patch.dict(os.environ, {"DCLAB_AGENT_HOME": str(self.home)}), \
                mock.patch.object(experience, "current_log", lambda: None):
            gw.Gateway(transport=gw._transport).client("home_business").complete([{"role": "user", "content": "the brief"}], None)
        docs = exchanges(self.home)
        self.assertEqual(len(docs), 1)
        kept = json.dumps({k: docs[0].get(k) for k in ("thinking", "thinking_redacted")})
        self.assertIn("why 7", kept)  # the text, never only "thought: true"
        self.assertEqual(docs[0]["purpose"], "home_business")

    def test_each_exchange_says_what_thinking_its_request_asked_for(self):
        # live 2026-10-28 (HyperAck): home_agent and leakage_review kept `thinking: []`; the line could not tell "asked, the
        # model chose not to think" from "never asked". The setting the request carried is now on every line.
        def run(purpose, env, answer):
            transport = Scripted([answer])
            home = Path(tempfile.mkdtemp())
            with mock.patch.object(gw, "_transport", lambda tier, p: transport), \
                    mock.patch.dict(os.environ, {"DCLAB_AGENT_HOME": str(home), **env}), \
                    mock.patch.object(experience, "current_log", lambda: None):
                gw.Gateway(transport=gw._transport).client(purpose).complete([{"role": "user", "content": "go"}], None)
            docs = exchanges(home)
            self.assertEqual(len(docs), 1)
            return docs[0]

        silent = {**reply(), "thinking": []}
        asked = run("leakage_review", {}, silent)
        self.assertEqual((asked["thinking_asked"], asked["thinking"]), ("adaptive", []))  # asked; the model did not think
        off = run("leakage_review", {"DCLAB_THINKING": "off"}, silent)
        self.assertIsNone(off["thinking_asked"])  # never asked
        cells = run("home_business", {}, reply(3))  # shown cells: only its shape is kept, the setting with it
        self.assertFalse(cells["whole"])
        self.assertEqual(cells["thinking_asked"], "adaptive")
        from dclab_rnd.models import anthropic_client

        with mock.patch.object(anthropic_client, "NO_THINKING_MODELS", {"m-old"}), \
                mock.patch.object(gw.Bound, "_think", lambda self: setattr(self, "model", "m-old") or setattr(self._transport, "thinking", "adaptive")):
            dropped = run("leakage_review", {}, silent)
        self.assertEqual(dropped["thinking_asked"], "adaptive")
        self.assertIn("refused", dropped["thinking_dropped"])  # the client sent it without thinking: said, never hidden
        # review: another request marked the model while this one, sent with thinking, was out: its thinking is no drop
        with mock.patch.object(anthropic_client, "NO_THINKING_MODELS", {"m-old"}), \
                mock.patch.object(gw.Bound, "_think", lambda self: setattr(self, "model", "m-old") or setattr(self._transport, "thinking", "adaptive")):
            thought = run("leakage_review", {}, reply(5))
        self.assertNotIn("thinking_dropped", thought)
        self.assertEqual(thought["thinking"], [{"text": "why 5"}])

    def test_a_transport_with_no_thinking_setting_is_not_read_as_never_asked(self):
        # review: an OpenAI-style local client (ChatClient) has no `thinking` attribute yet may return reasoning
        class Local:
            def __init__(self):
                self.replies = [reply(9)]

            def complete(self, messages, tools=None, max_tokens=1800, **options):
                return self.replies.pop(0)
        local = Local()
        with mock.patch.object(gw, "_transport", lambda tier, p: local), \
                mock.patch.dict(os.environ, {"DCLAB_AGENT_HOME": str(self.home)}), \
                mock.patch.object(experience, "current_log", lambda: None):
            gw.Gateway(transport=gw._transport).client("stage_notes").complete([{"role": "user", "content": "go"}], None)
        doc = exchanges(self.home)[0]
        self.assertEqual((doc["thinking_asked"], doc["thinking"]), (gw.TRANSPORT_DECIDES, [{"text": "why 9"}]))
        self.assertIsNone(gw._asked_of(None))  # no transport was made: the call never ran

    def test_a_scripted_gateway_with_no_workspace_writes_nowhere(self):
        with mock.patch.dict(os.environ, {"DCLAB_AGENT_HOME": str(self.home)}), mock.patch.object(experience, "current_log", lambda: None):
            gw.Gateway(transport=lambda tier, purpose: Scripted([reply()])).client("stage_notes").complete([{"role": "user", "content": "go"}], None)
        self.assertEqual(exchanges(self.home), [])  # tests never write into a real workspace

    def test_a_shadow_keeps_its_reply_and_thinking_as_its_own_exchange(self):
        from dclab_rnd.models.shadow import FileShadows
        from dclab_rnd.models.usage import FileUsage

        shadow_tier = {"name": "local", "model": "qwen"}
        seen = []

        class ThinkingShadow(Scripted):
            pass
        served, local = Scripted([reply(0)]), ThinkingShadow([reply(1)])
        g = gw.Gateway(FileUsage(self.home / "model_requests.jsonl"), transport=lambda tier, purpose: local if tier.get("name") == "local" else served)
        g.shadows = FileShadows(self.home / "model_shadow.jsonl")
        g.shadow_runner = lambda fn: fn()
        bound = g.client("stage_notes")
        bound._shadow_tier = shadow_tier
        bound._shadow([{"role": "user", "content": "go"}], None, 100, reply(0))
        rows = [json.loads(line) for line in (self.home / "model_shadow.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertNotIn("thinking", rows[0])  # the comparison log stays numbers only
        docs = [d for d in exchanges(self.home) if d.get("shadow")]
        self.assertEqual(len(docs), 1)
        self.assertEqual((docs[0]["model"], docs[0]["reply"], docs[0]["thinking"]), ("qwen", "answer 1", [{"text": "why 1"}]))
        self.assertEqual(local.thinking, "adaptive")  # the shadow was asked to think
        self.assertEqual(docs[0]["thinking_asked"], "adaptive")  # and its line says so (the shadow's own setting)

    def test_a_prompt_shown_cells_never_reaches_the_usage_log(self):
        from dclab_rnd.models.usage import FileUsage

        env = dict(os.environ)
        env.pop("DCLAB_LOG_PROMPTS", None)
        with mock.patch.dict(os.environ, env, clear=True):
            g = gw.Gateway(FileUsage(self.home / "model_requests.jsonl"), transport=lambda tier, purpose: Scripted([reply()]))
            g.client("intern").complete([{"role": "user", "content": "row: alice@example.com 4111"}], None)
        text = (self.home / "model_requests.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("alice@example.com", text)
        self.assertNotIn("4111", text)


class LocalModelsKeepTheirReasoning(unittest.TestCase):
    def client(self, field=None):
        return mc.ChatClient.__new__(mc.ChatClient) if False else types.SimpleNamespace(reasoning_field=field, tool_arguments="string")

    def test_reasoning_is_read_with_no_field_configured(self):
        c = self.client()
        message = types.SimpleNamespace(reasoning_content="I weigh the two columns.")
        self.assertEqual(mc.ChatClient._reasoning(c, message), "I weigh the two columns.")

    def test_inline_think_is_split_out_of_the_reply(self):
        c = self.client()
        out = mc.ChatClient._reply(c, "<think>The moment is checkout.</think>\nUse a time split.", [], "stop", None, "", {})
        self.assertEqual(out["content"], "Use a time split.")
        self.assertEqual(out["thinking"], [{"text": "The moment is checkout."}])
        self.assertNotIn("reasoning_content", out["assistant_message"])  # sent back only under a configured field

    def test_think_only_at_the_start_and_a_closing_tag_alone(self):
        c = self.client()
        code = 'Use this:\n```python\nre.sub(r"<think>.*?</think>", "", text)\n```'
        out = mc.ChatClient._reply(c, code, [], "stop", None, "", {})
        self.assertEqual(out["content"], code)  # code that mentions the tags is the answer, kept whole
        self.assertNotIn("thinking", out)
        out = mc.ChatClient._reply(c, "The holdout is the latest month.</think>\nSplit by time.", [], "stop", None, "", {})
        self.assertEqual((out["content"], out["thinking"]), ("Split by time.", [{"text": "The holdout is the latest month."}]))



class ClaudeThinkingParameters(unittest.TestCase):
    """2026-10-28, live on claude-sonnet-5-5: a thinking budget is refused with a 400, and with no display the thinking
    text comes back empty (the 317 earlier 'whole with thinking' exchanges held only signatures)."""

    def test_a_budget_becomes_adaptive_on_a_newer_model_and_stays_on_an_older_one(self):
        from dclab_rnd.models import anthropic_client as ac

        self.assertEqual(ac.thinking_param("enabled:8000", "claude-sonnet-5-5"), {"type": "adaptive", "display": "summarized"})
        self.assertEqual(ac.thinking_param("enabled:8000", "claude-haiku-4-5")["type"], "enabled")
        self.assertEqual(ac.thinking_param("adaptive", "claude-sonnet-5-5"), {"type": "adaptive", "display": "summarized"})

    def test_an_older_model_gets_a_budget_and_a_refusing_one_is_asked_again_without(self):
        from dclab_rnd.models import anthropic_client as ac

        self.assertEqual(ac.thinking_param("adaptive", "claude-haiku-4-5")["type"], "enabled")
        self.assertEqual(ac.thinking_param("adaptive", "claude-sonnet-4-20250514")["type"], "enabled")  # a dated 4.0 id
        self.assertEqual(ac.thinking_param("adaptive", "claude-opus-5-5")["type"], "adaptive")
        sent = []

        class Refusing:
            def create(self, **kwargs):
                sent.append(kwargs)
                if "thinking" in kwargs:
                    error = Exception('"thinking.type.enabled" is not supported for this model.')
                    error.status_code = 400
                    raise error
                return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="ok")], stop_reason="end_turn",
                                             usage=types.SimpleNamespace(input_tokens=1, output_tokens=1))
        client = ac.AnthropicClient(model="claude-test-refuses", api_key="sk-ant-test-placeholder")
        client._client = types.SimpleNamespace(messages=Refusing())
        client.thinking = "adaptive"
        self.addCleanup(ac.NO_THINKING_MODELS.discard, "claude-test-refuses")
        out = client.complete([{"role": "user", "content": "go"}], None, 100)
        self.assertEqual(out["content"], "ok")
        self.assertEqual(["thinking" in k for k in sent], [True, False])  # asked again without, once
        client.complete([{"role": "user", "content": "go"}], None, 100)
        self.assertFalse("thinking" in sent[-1])  # never asked twice

    def test_effort_goes_in_output_config_beside_a_format(self):
        from dclab_rnd.models import anthropic_client as ac

        with mock.patch.dict(os.environ, {"DCLAB_THINKING_EFFORT": "xhigh"}):
            out, _ = ac._options({"thinking": "adaptive", "response_format": {"type": "json_schema", "schema": {"type": "object"}}}, "", "claude-sonnet-5-5")
        self.assertEqual(out["extra_body"]["output_config"]["effort"], "xhigh")
        self.assertIn("format", out["extra_body"]["output_config"])
        with mock.patch.dict(os.environ, {"DCLAB_THINKING_EFFORT": "bogus"}):
            out, _ = ac._options({"thinking": "adaptive"}, "", "claude-sonnet-5-5")
        self.assertNotIn("extra_body", out)  # an unknown level is never sent


if __name__ == "__main__":
    unittest.main()
