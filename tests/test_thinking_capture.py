"""The owner's rule (2026-10-28): the agents whose runs become training data ask the model for its thinking, so the
whole exchange the gateway keeps (32.10) holds it. DCLAB_THINKING turns it off, on for all, or names the purposes."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.models import gateway as gw  # noqa: E402
from dclab_rnd.models import settings  # noqa: E402


class Transport:
    """A scripted transport that, like the Anthropic client, has a thinking setting."""

    thinking = None

    def complete(self, messages, tools=None, max_tokens=None, **options):
        return {"content": "ok", "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1}}


class ThinkingTest(unittest.TestCase):
    def test_the_training_agents_ask_for_thinking_by_default(self):
        with mock.patch.dict(os.environ, {"DCLAB_THINKING": ""}):
            for purpose in ("intern", "home_agent", "solution_planner", "reporter", "solver_answer"):
                self.assertEqual(settings.thinking_for(purpose), "adaptive", purpose)
            for purpose in ("embedding", "parse_pattern", "decision_answer"):
                self.assertIsNone(settings.thinking_for(purpose), purpose)
        self.assertTrue(set(settings.THINKING_DEFAULT) <= set(settings.PURPOSES))

    def test_off_all_and_a_list(self):
        with mock.patch.dict(os.environ, {"DCLAB_THINKING": "off"}):
            self.assertIsNone(settings.thinking_for("intern"))
        with mock.patch.dict(os.environ, {"DCLAB_THINKING": "all"}):
            self.assertEqual(settings.thinking_for("parse_pattern"), "adaptive")
        with mock.patch.dict(os.environ, {"DCLAB_THINKING": "reporter, critic"}):
            self.assertEqual((settings.thinking_for("reporter"), settings.thinking_for("intern")), ("adaptive", None))

    def test_the_gateway_sets_it_on_a_transport_that_takes_it_and_never_overrides_one_set(self):
        with mock.patch.dict(os.environ, {"DCLAB_THINKING": ""}):
            g = gw.Gateway()
            g.transport = lambda tier, p: Transport()
            bound = gw.Bound(g, "intern", settings.purpose("intern"), {"name": "standard", "model": "m"}, None, None)
            self.assertEqual(bound._transport.thinking, "adaptive")
            other = gw.Bound(g, "parse_pattern", settings.purpose("parse_pattern"), {"name": "cheap", "model": "m"}, None, None)
            self.assertIsNone(other._transport.thinking)
        chosen = Transport()
        chosen.thinking = "enabled:2000"  # a caller's own choice (the twin lane's) stays
        g = gw.Gateway()
        g.transport = lambda tier, p: chosen
        if True:
            bound = gw.Bound(g, "intern", settings.purpose("intern"), {"name": "standard", "model": "m"}, None, None)
            self.assertEqual(bound._transport.thinking, "enabled:2000")


class CellsThinkingTest(unittest.TestCase):
    def test_a_purpose_shown_cells_keeps_its_thinking_redacted_in_the_workspace(self):
        import json
        import tempfile

        home = Path(tempfile.mkdtemp())
        g = gw.Gateway()
        g.transport = lambda tier, p: Transport()
        bound = gw.Bound(g, "intern", settings.purpose("intern"), {"name": "standard", "model": "m"}, None, None)
        out = {"content": "done", "thinking": [{"text": "The row with mail ann@example.com looks odd.", "signature": "s"}]}
        with mock.patch.object(gw, "_exchange_home", lambda gateway: home), \
                mock.patch("dclab_rnd.experience.model_text_allowed", return_value=True):
            bound._keep_exchange("r1", [{"role": "user", "content": "rows"}], out, "ok")
        line = json.loads(next((home / "model_exchanges").rglob("*.jsonl")).read_text().splitlines()[0])
        self.assertFalse(line["whole"])  # the prompt held cells: its messages are not kept
        self.assertNotIn("ann@example.com", json.dumps(line["thinking_redacted"]))
        self.assertIn("looks odd", line["thinking_redacted"][0]["text"])


if __name__ == "__main__":
    unittest.main()
