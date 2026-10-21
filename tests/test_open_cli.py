"""The `dclab` command of the open edition: the doctor's probe, the switches it sets, `init`'s settings file, and no vendor
model sent to an endpoint that never named one."""

from __future__ import annotations

import io
import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from dclab_rnd import gates, models, open_cli
from dclab_rnd.extensions import optional
from dclab_rnd.models import settings as model_settings
from dclab_rnd.models.gateway import Gateway

CLEAN = {k: "" for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY",
                         "ANTHROPIC_MODEL", "DCLAB_INTERN_MODEL", "DCLAB_TIER_STANDARD_MODEL", "DCLAB_TIER_STANDARD_BASE_URL",
                         "DCLAB_TIER_CHEAP_MODEL", "DCLAB_TIER_CHEAP_BASE_URL")}


class Scripted:
    def __init__(self, replies):
        self.replies = replies

    def complete(self, messages, tools=None, max_tokens=1800, **options):
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def reply(content="", calls=None):
    return {"content": content, "tool_calls": calls or [], "usage": {"input_tokens": 3, "output_tokens": 2}}


class CliTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        env = mock.patch.dict(os.environ, {**CLEAN, "DCLAB_AGENT_HOME": str(self.home), "OPENAI_API_KEY": "sk-test-never-shown",
                                           "OPENAI_MODEL": "a-model", "DCLAB_NO_LIVE_MODELS": ""})
        env.start()
        self.addCleanup(env.stop)
        self.before = models.installed()
        self.addCleanup(models.install, self.before)

    def use(self, replies):
        gateway = Gateway(None, transport=lambda tier, purpose: Scripted(replies))
        models.install(gateway)
        patch = mock.patch("dclab_rnd.models.gateway.for_workspace", return_value=gateway)  # the doctor probes through the workspace's gateway
        patch.start()
        self.addCleanup(patch.stop)

    def run_cli(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = open_cli.main(list(argv))
        return code, out.getvalue()

    def test_a_model_calling_tools_natively_passes_the_probe_and_serves_the_tool_using_agents(self):
        self.use([reply("OK"), reply(calls=[{"id": "c1", "name": "record_answer", "arguments": {"answer": 5}}])])
        code, out = self.run_cli("doctor")
        self.assertIn("calls tools natively", out)
        self.assertNotIn("sk-test-never-shown", out)
        self.assertTrue(gates.read(self.home)["probe"]["tools"])
        self.assertEqual(gates.why("built_chat", self.home), "model (probed)")
        self.assertEqual(code, 0, out)

    def test_a_model_writing_its_tool_call_as_text_fails_the_probe_and_the_agents_stay_on_their_rules(self):
        self.use([reply("OK"), reply(json.dumps({"name": "record_answer", "arguments": {"answer": 5}}))])
        code, out = self.run_cli("doctor")
        self.assertEqual(code, 1)
        self.assertIn("wrote the call as text", out)
        self.assertFalse(gates.passed("built_chat", self.home))

    def test_a_failed_probe_after_a_passed_one_puts_the_agents_back_on_their_rules(self):  # the review of 2026-10-21
        self.use([reply("OK"), reply(calls=[{"id": "c1", "name": "record_answer", "arguments": {"answer": 5}}])])
        self.run_cli("doctor")
        self.assertEqual(gates.why("built_chat", self.home), "model (probed)")
        self.use([RuntimeError("APIConnectionError: connection refused")])
        self.assertEqual(self.run_cli("doctor")[0], 1)
        self.assertEqual(gates.why("built_chat", self.home), "rules")

    def test_a_probe_holds_for_the_model_it_tried_only(self):
        self.use([reply("OK"), reply(calls=[{"id": "c1", "name": "record_answer", "arguments": {"answer": 5}}])])
        self.run_cli("doctor")
        with mock.patch.dict(os.environ, {"OPENAI_MODEL": "another-model"}):
            self.assertEqual(gates.why("built_chat", self.home), "rules")  # model B never serves on model A's probe
        self.assertEqual(gates.read(self.home)["probe"]["model"], "a-model")

    def test_a_failing_endpoint_is_reported_with_its_reason_and_fails_the_doctor(self):
        self.use([RuntimeError("APIConnectionError: connection refused")])
        code, out = self.run_cli("doctor")
        self.assertEqual(code, 1)
        self.assertIn("connection refused", out)
        self.assertNotIn("calls tools natively", out)

    def test_trust_switches_a_role_on_and_an_unknown_role_is_refused(self):
        self.assertEqual(self.run_cli("models", "trust", "nope")[0], 2)
        code, out = self.run_cli("models", "trust", "built_chat")
        self.assertEqual(code, 0)
        self.assertEqual(gates.why("built_chat", self.home), "model (trusted by you)")
        if optional("agent_eval") is None:  # the open edition: the switch decides, and the command shows it (the full edition's measurement decides there)
            self.assertIn("model (trusted by you)", out)
        self.assertFalse(gates.passed("map_roles", self.home))

    def test_init_writes_a_private_settings_file_and_keeps_the_lines_it_does_not_own(self):
        cfg = Path(tempfile.mkdtemp()) / "cfg"
        cfg.mkdir()
        (cfg / ".env").write_text("DCLAB_AUTH=password\nOPENAI_MODEL=old\n", encoding="utf-8")
        answers = iter(["2", "https://llm.example.org/v1", "my-model", "n"])
        with mock.patch.object(open_cli, "CONFIG_DIR", cfg), mock.patch.object(open_cli, "CONFIG_ENV", cfg / ".env"), \
                mock.patch.object(open_cli, "ollama_models", return_value=[]), \
                mock.patch("builtins.input", lambda prompt="": next(answers)), \
                mock.patch.object(open_cli.getpass, "getpass", return_value="sk-typed-secret"):
            code, out = self.run_cli("init")
        text = (cfg / ".env").read_text(encoding="utf-8")
        self.assertEqual(code, 0)
        self.assertIn("DCLAB_AUTH=password", text)
        self.assertIn("OPENAI_MODEL=my-model", text)
        self.assertNotIn("OPENAI_MODEL=old", text)
        self.assertIn("OPENAI_API_KEY=sk-typed-secret", text)
        self.assertNotIn("sk-typed-secret", out)
        self.assertEqual(stat.S_IMODE((cfg / ".env").stat().st_mode), 0o600)


class NeutralDefaultsTest(unittest.TestCase):
    def test_an_endpoint_that_named_no_model_gets_no_vendor_model_and_is_not_available(self):
        with mock.patch.dict(os.environ, {**CLEAN, "DCLAB_LLM_BASE_URL": "http://127.0.0.1:11434/v1"}):
            tier = model_settings.public("standard")
        self.assertEqual(tier["model"], "")
        self.assertFalse(tier["available"])
        self.assertFalse(tier["model_configured"])

    def test_a_tier_on_another_server_never_inherits_the_standard_model(self):  # the review of 2026-10-21
        with mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-x", "DCLAB_TIER_CHEAP_BASE_URL": "http://localhost:11434/v1"}):
            cheap = model_settings.public("cheap")
        self.assertEqual(cheap["model"], "")
        self.assertFalse(cheap["available"])
        with mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-x", "OPENAI_MODEL": "m1", "DCLAB_TIER_CHEAP_MODEL": "small",
                                          "DCLAB_TIER_CHEAP_BASE_URL": "http://localhost:11434/v1"}):
            self.assertEqual(model_settings.tier("cheap")["model"], "small")
        with mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-x", "OPENAI_MODEL": "m1"}):
            self.assertEqual(model_settings.tier("cheap")["model"], "m1")  # the same server: the standard model, as before

    def test_the_vendor_endpoint_keeps_its_default_model(self):
        with mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-x"}):
            self.assertTrue(model_settings.tier("standard")["model"])


if __name__ == "__main__":
    unittest.main()


class AgentStatusTest(unittest.TestCase):
    def test_with_no_model_every_agent_is_on_its_rules_and_says_why(self):
        with mock.patch.dict(os.environ, {**CLEAN, "DCLAB_AGENT_HOME": tempfile.mkdtemp()}):
            rows = gates.status()
        self.assertEqual({r["role"] for r in rows}, set(gates.ROLES))
        self.assertTrue(all(r["serves"] == "rules" and "no model" in r["why"] for r in rows))

    def test_the_page_says_what_the_chat_itself_decides(self):
        with mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-x", "OPENAI_MODEL": "m", "DCLAB_AGENT_HOME": tempfile.mkdtemp()}):
            for forced in ("model", "rules"):
                with mock.patch.dict(os.environ, {"DCLAB_BUILT_CHAT_SERVE": forced}):
                    from dclab_rnd.draft.orchestrator import _model_serves

                    row = next(r for r in gates.status() if r["role"] == "built_chat")
                    self.assertEqual(row["serves"], "model" if _model_serves() else "rules")
                    self.assertEqual(row["serves"], forced)

    def test_a_reader_switched_off_says_so_on_the_page_and_in_the_command(self):  # the review of 2026-10-21: it said "model (probed)"
        home = tempfile.mkdtemp()
        with mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-x", "OPENAI_MODEL": "m", "DCLAB_AGENT_HOME": home,
                                          "DCLAB_CONVERSATION_READER": "", "DCLAB_MAP_ROLES": ""}):
            gates.trust(["all"], Path(home))  # the model would serve every role it is asked for
            rows = gates.status()
            out = io.StringIO()
            with redirect_stdout(out):
                code = open_cli.main(["models", "status"])
        for role, switch in (("conversation_reader", "DCLAB_CONVERSATION_READER"), ("map_roles", "DCLAB_MAP_ROLES")):
            row = next(r for r in rows if r["role"] == role)
            self.assertEqual(row["serves"], "rules", row)  # what the product does: draft.reader and draft.map ask their switch
            self.assertIn(f"switched off ({switch}=1", row["why"])
        self.assertTrue(all(r["why"].startswith(r["serves"]) for r in rows), rows)  # a reason never says model when the rules serve
        self.assertEqual(code, 0)
        self.assertRegex(out.getvalue(), r"conversation_reader\s+rules: switched off")
