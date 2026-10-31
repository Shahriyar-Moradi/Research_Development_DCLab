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
            self.assertIsNone(settings.thinking_for("embedding"))  # the owner's rule (10-28 night): every other purpose thinks
            for purpose in ("parse_pattern", "decision_answer"):
                self.assertEqual(settings.thinking_for(purpose), "adaptive", purpose)
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
            other = gw.Bound(g, "embedding", settings.purpose("embedding"), {"name": "cheap", "model": "m"}, None, None)
            self.assertIsNone(other._transport.thinking)  # the one purpose with no thinking (settings.NO_THINKING)
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
        with mock.patch.object(gw, "_exchange_home", lambda gateway, live=False: home), \
                mock.patch("dclab_rnd.experience.model_text_allowed", return_value=True):
            bound._keep_exchange("r1", [{"role": "user", "content": "rows"}], out, "ok")
        line = json.loads(next((home / "model_exchanges").rglob("*.jsonl")).read_text().splitlines()[0])
        self.assertFalse(line["whole"])  # the prompt held cells: its messages are not kept
        self.assertNotIn("ann@example.com", json.dumps(line["thinking_redacted"]))
        self.assertIn("looks odd", line["thinking_redacted"][0]["text"])

    def test_cells_of_invented_tables_keep_the_exchange_whole(self):
        """2026-10-31: the practice exams' lead agent saw cells of tables code invented, so 306 of its calls kept only a
        redacted thinking and no words. An invented table hides no one: the exam's own gateway says so and the call is kept
        whole. The review: never a setting, so no .env can loosen the rule for the server that reads people's tables."""
        import json
        import tempfile

        def kept(gateway_attr, env):
            home = Path(tempfile.mkdtemp())
            g = gw.Gateway()
            if gateway_attr:
                g.invented_tables = gateway_attr
            g.transport = lambda tier, p: Transport()
            bound = gw.Bound(g, "case_agent", settings.purpose("case_agent"), {"name": "standard", "model": "m"}, None, None)
            bound.request_cells = True
            out = {"content": "done", "thinking": [{"text": "Plan basic churns more.", "signature": "s"}]}
            with mock.patch.object(gw, "_exchange_home", lambda gateway, live=False: home), \
                    mock.patch.dict(os.environ, {"DCLAB_LOG_PROMPTS": "1", **env}):
                bound._keep_exchange("r1", [{"role": "user", "content": "rows"}], out, "ok")
            return json.loads(next((home / "model_exchanges").rglob("*.jsonl")).read_text().splitlines()[0])

        line = kept("conversation exam dev", {})
        self.assertTrue(line["whole"])
        self.assertEqual((line["messages"][0]["content"], line["thinking"][0]["text"]), ("rows", "Plan basic churns more."))
        self.assertEqual(line["tables"], "invented: conversation exam dev")
        self.assertFalse(kept(None, {})["whole"])  # today's rule without the mark: words not kept
        self.assertFalse(kept(None, {"DCLAB_INVENTED_TABLES": "1"})["whole"])  # a setting never turns it on

    def test_the_exam_marks_only_a_practice_runs_gateway(self):
        from dclab_rnd.agent_eval import conversation_exam as ce

        class G:
            pass

        dev, sealed, test = G(), G(), G()
        ce.mark_invented_tables(dev, "dev")
        ce.mark_invented_tables(sealed, "sealed_v1")  # kept whole in the vault, never in the workspace
        ce.mark_invented_tables(test, "test")
        self.assertEqual(dev.invented_tables, "conversation exam dev")
        self.assertFalse(hasattr(sealed, "invented_tables") or hasattr(test, "invented_tables"))

    def test_an_exams_shape_only_exchange_says_so_and_is_never_counted_as_lost_thinking(self):
        """2026-10-30: 1,020 shape-only exchanges since 28 Oct were all sealed exams' (shape only by the owner's rule), but
        each said the same vague reason and the coverage counted them as thinking lost."""
        import json
        import tempfile

        from dclab_rnd.store import coverage as cov

        home = Path(tempfile.mkdtemp())
        g = gw.Gateway()
        g.transport = lambda tier, p: Transport()
        bound = gw.Bound(g, "home_agent", settings.purpose("home_agent"), {"name": "standard", "model": "m"}, None, None)
        out = {"content": "done", "thinking": [{"text": "The person wants a forecast.", "signature": "s"}]}
        with mock.patch.object(gw, "_exchange_home", lambda gateway, live=False: home), \
                mock.patch.dict(os.environ, {"DCLAB_LOG_PROMPTS": "0", "DCLAB_SEALED_RUN": "sealed_v1"}):
            bound._keep_exchange("r1", [{"role": "user", "content": "words"}], out, "ok")
        line = json.loads(next((home / "model_exchanges").rglob("*.jsonl")).read_text().splitlines()[0])
        self.assertEqual((line["whole"], line["exam"], line["thought"]), (False, "sealed_v1", True))
        self.assertIn("an exam", line["why"])
        self.assertNotIn("forecast", json.dumps(line))  # an exam keeps no thinking text
        from dclab_rnd.store.layout import Layout

        doc = cov.measure(Layout(Path(tempfile.mkdtemp())), home=home)
        self.assertEqual(doc["sources"]["by_purpose"]["home_agent"]["thought_lost"], 0)

    def test_an_exam_run_marks_itself(self):
        from dclab_rnd import experience

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DCLAB_SEALED_RUN", None)
            self.assertTrue(experience.sealed_run("sealed_v2"))
            self.assertEqual((os.environ["DCLAB_SEALED_RUN"], os.environ["DCLAB_LOG_PROMPTS"]), ("sealed_v2", "0"))


if __name__ == "__main__":
    unittest.main()


class InventedTablesRunnersTest(unittest.TestCase):
    """2026-10-31 (the owner: all data captured, thinking included): every runner whose tables are all invented by code marks
    its gateway, so the calls that saw cells keep their words and thinking. A runner that may read a person's table (the
    studio, the intern on projects, twin runs on samples) is never on this list."""

    RUNNERS = ("agent_eval/home/__main__.py", "agent_eval/home/conversation.py", "agent_eval/home/sparse.py",
               "agent_eval/home/tools_probe.py", "agent_eval/home/intent_requests.py", "agent_eval/journeys/__main__.py",
               "agent_eval/journeys/programme.py", "decision_model_compare.py",
               "agent_eval/conversation_exam/__init__.py")

    def test_each_invented_table_runner_marks_its_gateway(self):
        for rel in self.RUNNERS:
            text = (ROOT / "dclab_rnd" / rel).read_text(encoding="utf-8")
            self.assertIn("mark_invented_tables(", text, rel)

    def test_the_mark_is_never_set_by_the_server_or_the_twin(self):
        roots = ["agentic", "draft", "studio", "intern", "twin", "jobs", "agent_eval/investigator"]
        files = [p for r in roots for p in (ROOT / "dclab_rnd" / r).rglob("*.py")] + [ROOT / "dclab_rnd" / "worker.py"]
        self.assertTrue(len(files) > 20)
        for path in files:
            self.assertNotIn("invented_tables", path.read_text(encoding="utf-8"), str(path.relative_to(ROOT)))

    def test_the_helper_marks_the_gateway(self):
        g = gw.Gateway()
        self.assertIs(gw.mark_invented_tables(g, "home eval"), g)
        self.assertEqual(g.invented_tables, "home eval")
        sealed = gw.Gateway()
        with mock.patch.dict(os.environ, {"DCLAB_SEALED_RUN": "sealed_v1"}):
            gw.mark_invented_tables(sealed, "home eval")
        self.assertFalse(hasattr(sealed, "invented_tables"))  # a sealed run is never marked
