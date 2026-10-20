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
from dclab_rnd.models import settings as model_settings
from dclab_rnd.models.gateway import Gateway

CLEAN = {k: "" for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY",
                         "ANTHROPIC_MODEL", "DCLAB_INTERN_MODEL", "DCLAB_TIER_STANDARD_MODEL", "DCLAB_TIER_STANDARD_BASE_URL")}


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
        models.install(Gateway(None, transport=lambda tier, purpose: Scripted(replies)))

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

    def test_the_vendor_endpoint_keeps_its_default_model(self):
        with mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-x"}):
            self.assertTrue(model_settings.tier("standard")["model"])


if __name__ == "__main__":
    unittest.main()
