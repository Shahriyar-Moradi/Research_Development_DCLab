"""The research campaign's routes on the product's Claude (the owner runs Claude only): a run starts on the provider the
server has, with a priced model, and a run never resumes on another. No request leaves: the launch is stubbed. Kept apart
from test_agentic.py, which needs NOOA (``.venv-agent``); these need the product server (pandas, ``.venv``)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_rnd.agentic import server, services as services_module
except ImportError as error:  # the product server needs pandas
    MISSING = str(error)
else:
    MISSING = ""

CLAUDE = {"DCLAB_LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "anthropic-placeholder", "OPENAI_API_KEY": "", "DCLAB_BLOCK_PROVIDERS": "openai",
          "DCLAB_TIER_STANDARD_BASE_URL": "", "DCLAB_TIER_STANDARD_MODEL": "", "DCLAB_TIER_STANDARD_API_KEY": "", "DCLAB_CAMPAIGN_MODEL": "",
          "DCLAB_NO_LIVE_MODELS": "1"}


@unittest.skipIf(MISSING, MISSING)
class CampaignRouteTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, CLAUDE)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.launched = []
        launch = mock.patch.object(services_module.Services, "launch", lambda s, run_id, resume=False: self.launched.append((run_id, resume)))
        launch.start()
        self.addCleanup(launch.stop)
        self.client = TestClient(server.create_app(Path(tempfile.mkdtemp())))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.token = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def test_a_run_starts_on_claude_with_a_priced_model(self):
        cfg = self.client.get("/api/config").json()
        self.assertTrue(cfg["api_key_configured"])
        made = self.client.post("/api/runs", json={"max_experiments": 1}, headers=self.token)
        self.assertEqual(made.status_code, 201, made.text)
        config = made.json()["config"]
        self.assertEqual((config["provider"], config["model"]), ("anthropic", "claude-sonnet-5"))
        self.assertEqual(self.launched, [(made.json()["id"], False)])
        self.assertNotIn("anthropic-placeholder", made.text, "the key never in a run's record")

    def test_an_unpriced_campaign_model_is_refused_and_a_requested_one_falls_back(self):
        os.environ["DCLAB_CAMPAIGN_MODEL"] = "claude-opus-4-7"
        refused = self.client.post("/api/runs", json={"max_experiments": 1}, headers=self.token)
        self.assertEqual(refused.status_code, 409)
        self.assertIn("no price", refused.json()["detail"])
        os.environ["DCLAB_CAMPAIGN_MODEL"] = ""
        made = self.client.post("/api/runs", json={"max_experiments": 1, "model": "claude-opus-4-7"}, headers=self.token)
        self.assertEqual(made.json()["config"]["model"], "claude-sonnet-5")

    def test_a_run_never_resumes_on_another_provider(self):
        services = self.client.app.state.services
        services.store.create("old", {"model": "gpt-5.6-terra", "max_experiments": 1})  # a run from before: OpenAI's
        services.store.update("old", status="paused")
        refused = self.client.post("/api/runs/old/resume", headers=self.token)
        self.assertEqual(refused.status_code, 409)
        self.assertIn("began on openai", refused.json()["detail"])
        self.assertEqual(self.launched, [])

    def test_a_blocked_provider_alone_is_no_campaign(self):
        os.environ.update({"DCLAB_LLM_PROVIDER": "", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "openai-placeholder"})
        self.assertFalse(self.client.get("/api/config").json()["api_key_configured"])
        self.assertEqual(self.client.post("/api/runs", json={}, headers=self.token).status_code, 409)


if __name__ == "__main__":
    unittest.main()
