"""The provider switch (the owner, 2026-11-11): with OpenRouter every DCLab model call, the lead agent's included, goes to Qwen;
with Anthropic every call goes to Claude again. No model name written in code may cross providers (a Qwen name sent to
Anthropic's endpoint, or a Claude name to OpenRouter), and results record the model actually served."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.models import settings  # noqa: E402

PREFIXES = ("DCLAB_TIER_", "ANTHROPIC_", "OPENAI_", "OPENROUTER_", "DCLAB_LLM_", "DCLAB_INTERN_MODEL", "DCLAB_ROUTE_MODEL",
            "DCLAB_BLOCK_PROVIDERS", "DCLAB_CAMPAIGN_MODEL")
LEAD_AGENT = ("case_agent", "case_specialist", "route_check")


def env(**extra):
    keep = {k: v for k, v in os.environ.items() if not k.startswith(PREFIXES)}
    return mock.patch.dict(os.environ, {**keep, **extra}, clear=True)


OPENROUTER = dict(DCLAB_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="or-placeholder", ANTHROPIC_API_KEY="ant-placeholder",
                  OPENAI_API_KEY="openai-placeholder")
ANTHROPIC = dict(DCLAB_LLM_PROVIDER="anthropic", OPENROUTER_API_KEY="or-placeholder", ANTHROPIC_API_KEY="ant-placeholder",
                 OPENAI_API_KEY="openai-placeholder")


class ProviderSwitchTests(unittest.TestCase):
    def test_on_openrouter_every_purpose_is_served_by_qwen(self):
        with env(**OPENROUTER):
            for name, p in settings.PURPOSES.items():
                t = settings.tier(p.tier)
                self.assertEqual(settings._host(str(t.get("base_url") or "")), settings.OPENROUTER_HOST, name)
                self.assertEqual(settings.served_model(name), "qwen/qwen3.8-flash", name)

    def test_on_anthropic_every_purpose_is_served_by_claude_and_the_other_keys_stay(self):
        with env(**ANTHROPIC):
            for name in settings.PURPOSES:
                self.assertTrue(settings.served_model(name).startswith("claude"), name)

    def test_the_lead_agent_follows_the_provider(self):
        from dclab_rnd.agents import case_agent as A

        with env(**OPENROUTER):
            self.assertEqual({p: settings.served_model(p) for p in LEAD_AGENT}, {p: "qwen/qwen3.8-flash" for p in LEAD_AGENT})
            self.assertEqual(A.route_model(), "qwen/qwen3.8-flash")
        with env(**ANTHROPIC):
            self.assertTrue(A.route_model().startswith("claude"))  # never a Qwen name sent to Anthropic's endpoint

    def test_the_exam_roles_follow_the_provider(self):
        from dclab_rnd.agent_eval import conversation_exam as X

        with env(**OPENROUTER):
            self.assertEqual(set(X.served_models().values()), {"qwen/qwen3.8-flash"})  # Qwen answers, plays the people and judges
        with env(**ANTHROPIC):
            served = X.served_models()
            self.assertEqual((served["person"], served["judge"]), ("claude-haiku-5-5", "claude-opus-5-5"))

    def test_the_reviewer_records_the_served_model(self):
        from dclab_rnd import llm_review

        with env(**OPENROUTER):
            self.assertEqual(llm_review.default_model(), "qwen/qwen3.8-flash")
        with env(**ANTHROPIC):
            self.assertTrue(llm_review.default_model().startswith("claude"))

    def test_the_twin_lane_follows_the_provider_whatever_keys_are_present(self):
        from dclab_rnd.models.anthropic_client import AnthropicClient
        from dclab_rnd.twin import claude

        with env(**ANTHROPIC):  # the OpenRouter key is still there, as the switch keeps it
            client = claude.transport()
            self.assertIsInstance(client, AnthropicClient)
            self.assertTrue(claude.model_name().startswith("claude"))
        with env(**OPENROUTER):
            self.assertNotIsInstance(claude.transport(), AnthropicClient)
            self.assertEqual(claude.model_name(), "qwen/qwen3.8-flash")

    def test_no_code_names_a_model_of_the_other_provider_in_a_live_call(self):
        """The names written in code for a live call: the route check, the exam's person and judge, the journeys' cheap
        seats, the reviewer, the committee and the students' exam read the served model, not a fixed one."""
        for path, banned in (("dclab_rnd/llm_review.py", 'DEFAULT_MODEL = "qwen'),
                             ("dclab_rnd/agent_eval/journeys/__main__.py", 'setdefault("DCLAB_TIER_CHEAP_MODEL", "qwen'),
                             ("dclab_rnd/students_exam.py", 'doc = {"model": "qwen'),
                             ("dclab_rnd/agents/case_agent.py", 'ROUTE_MODEL = "qwen')):
            self.assertNotIn(banned, (ROOT / path).read_text(encoding="utf-8"), path)


if __name__ == "__main__":
    unittest.main()
