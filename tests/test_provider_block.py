"""Claude only (the owner, 2026-10-25): a provider blocked in .env (DCLAB_BLOCK_PROVIDERS=openai) is never asked, its
key kept; a live run that never read .env reaches the owner's model, never a key that sits in the shell."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.models import live_guard, settings  # noqa: E402


class BlockTest(unittest.TestCase):
    def test_a_blocked_provider_is_known_by_its_hosts_and_a_local_server_never(self):
        with mock.patch.dict(os.environ, {live_guard.BLOCK: "openai"}):
            self.assertEqual(live_guard.blocked("https://api.openai.com/v1"), "openai")
            self.assertEqual(live_guard.blocked("https://eu.api.openai.com/v1"), "openai")
            for url in ("https://api.anthropic.com", "http://localhost:11434/v1", "http://127.0.0.1:8080/v1", "", "https://notopenai.com/v1"):
                self.assertIsNone(live_guard.blocked(url), url)
        with mock.patch.dict(os.environ, {live_guard.BLOCK: ""}):
            self.assertIsNone(live_guard.blocked("https://api.openai.com/v1"))

    def test_a_tier_pointed_at_a_blocked_provider_is_unavailable_and_sends_no_key(self):
        env = {live_guard.BLOCK: "openai", "DCLAB_LLM_PROVIDER": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1",
               "OPENAI_API_KEY": "sk-test", "OPENAI_MODEL": "gpt-x"}
        with mock.patch.dict(os.environ, env):
            t = settings.tier("standard")
        self.assertEqual((t["base_url"], t["model"], t["api_key"], t["blocked"]), ("", "", "", "openai"))

    def test_no_client_to_a_blocked_provider_is_ever_created(self):
        from dclab_rnd.models.client import ChatClient

        with mock.patch.dict(os.environ, {live_guard.BLOCK: "openai"}):
            with self.assertRaises(live_guard.ProviderBlocked):
                _ = ChatClient(model="gpt-x", base_url="https://api.openai.com/v1", api_key="sk-test").client
            try:
                from dclab_rnd.agentic import agents
            except ModuleNotFoundError:  # the research campaign's libraries live in .venv-agent only
                return
            # the owner's .env (litellm loads it on import) never makes this the Claude path
            with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": "", "DCLAB_LLM_PROVIDER": "", "ANTHROPIC_API_KEY": ""}), self.assertRaises(live_guard.ProviderBlocked):
                agents._endpoint()  # the research campaign's own client: blocked too

    def test_a_live_run_that_never_read_env_takes_the_owners_settings_and_a_test_never(self):
        env_file = Path(tempfile.mkdtemp()) / ".env"
        env_file.write_text("DCLAB_LLM_PROVIDER=anthropic\nANTHROPIC_API_KEY=sk-ant-test\nDCLAB_BLOCK_PROVIDERS=openai\n")
        clean = {k: v for k, v in os.environ.items() if k not in ("DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", live_guard.BLOCK)}
        with mock.patch.object(live_guard, "ENV_FILE", env_file), mock.patch.dict(os.environ, clean, clear=True):
            live_guard._ENV_CACHE.clear()
            self.assertEqual(live_guard.owner_value("DCLAB_LLM_PROVIDER"), "")  # inside a test: never the owner's .env
            with mock.patch.object(live_guard, "in_test", return_value=False):  # a command-line run
                self.assertEqual(live_guard.owner_value("DCLAB_LLM_PROVIDER"), "anthropic")
                self.assertTrue(settings.anthropic_default())
                self.assertEqual(settings.tier("cheap")["provider"], "anthropic")
                self.assertEqual(live_guard.blocked("https://api.openai.com/v1"), "openai")
                with mock.patch.dict(os.environ, {"DCLAB_LLM_PROVIDER": ""}):
                    self.assertEqual(live_guard.owner_value("DCLAB_LLM_PROVIDER"), "")  # the process's own value, even empty, wins
        live_guard._ENV_CACHE.clear()


if __name__ == "__main__":
    unittest.main()
