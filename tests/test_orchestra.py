"""Package 28.1: the orchestration layer's spike on LangGraph, used thinly: three nodes on a draft, paused for the person,
resumed from a checkpoint in another process, every step and pause in the harness; the model only through the gateway."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import langgraph  # noqa: F401

    from dclab_rnd import experience
    from dclab_rnd.agent_eval import journeys
    from dclab_rnd.orchestra import checkpoints, model, spike
    from tests.test_models_gateway import CLEAN, Scripted, gateway
except ImportError as error:  # pydantic, pandas or langgraph are not in every environment
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class SpikeTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.log = experience.FileLog(self.home / "experience", "wtest")
        experience.install(self.log)
        self.addCleanup(experience.install, None)
        self.journey = next(j for j in journeys.load("dev") if j["id"] == "JD-02")

    def walk(self, run, draft="d1"):
        """A written person on the spike: every question answered from the journey, the confirmation approved."""
        out = run(draft, self.journey["words"])
        asked = []
        while out["status"] == "paused" and out["asks"]["kind"] == "question":
            asked.append(out["asks"]["field"])
            out = run(draft, resume=self.journey["answers"].get(out["asks"]["field"], "I don't know"))
        self.assertEqual(out["asks"]["kind"], "confirmation")
        self.assertIn("family", out["asks"]["summary"])
        out = run(draft, resume=True)
        self.assertEqual((out["status"], out["confirmed"]), ("done", True))
        return asked, out

    def test_the_spike_is_a_session_of_the_orchestra_agent(self):  # the harness's probe
        with checkpoints.open_checkpointer(self.home, database_url="") as saver:
            s = spike.Spike(saver)
            asked, out = self.walk(lambda d, p=None, resume=None: s.run(d, p, resume=resume))
        self.assertTrue(asked)
        self.assertEqual(out["asked"], asked)
        events = self.log.read()
        steps = [e for e in events if e.kind == "agent.step" and e.payload["agent"] == "orchestra"]
        tools = [e.payload.get("tool") for e in steps]
        self.assertIn("interrupt", tools)
        self.assertIn("answer", tools)
        self.assertIn("analyst", tools)
        self.assertIn("confirm", tools)
        self.assertEqual(tools.count("interrupt"), len(asked) + 1)  # each question and the confirmation
        runs = [e for e in events if e.kind == "run.recorded" and e.payload["name"] == "orchestra"]
        self.assertEqual(len(runs), len(asked) + 2)  # the start, each resume
        self.assertEqual({r.payload["status"] for r in runs[:-1]}, {"stopped"})  # every paused walk
        self.assertEqual(runs[-1].payload["status"], "completed")

    def test_a_walk_resumes_in_another_process_from_its_checkpoint(self):
        with checkpoints.open_checkpointer(self.home, database_url="") as saver:
            first = spike.Spike(saver).run("d2", self.journey["words"])
        self.assertEqual((first["status"], first["asks"]["kind"]), ("paused", "question"))
        field = first["asks"]["field"]
        with checkpoints.open_checkpointer(self.home, database_url="") as saver:  # the file reopened: a new process
            later = spike.Spike(saver).run("d2", resume=self.journey["answers"].get(field, "I don't know"))
        self.assertEqual(later["status"], "paused")
        self.assertEqual(later["asked"][:1], [field])  # what was asked before the crash is kept
        self.assertNotEqual(later["asks"].get("field"), field)  # never asked twice
        self.assertEqual(checkpoints.where(self.home, database_url="")["kind"], "sqlite")
        self.assertTrue((self.home / "orchestra" / "checkpoints.sqlite3").exists())

    def test_a_new_walk_needs_the_words_and_a_resume_needs_a_thread(self):
        with checkpoints.open_checkpointer(self.home, database_url="") as saver:
            with self.assertRaises(ValueError):
                spike.Spike(saver).run("d3")
            with self.assertRaises(ValueError):  # a stale client, a wrong id: refused before any node runs
                spike.Spike(saver).run("fresh", resume="whatever")
        events = self.log.read()
        self.assertEqual([e for e in events if e.kind == "run.recorded" and e.payload["name"] == "orchestra"], [])

    def test_a_correction_at_the_confirmation_is_not_an_approval(self):
        with checkpoints.open_checkpointer(self.home, database_url="") as saver:
            s = spike.Spike(saver)
            out = s.run("d4", self.journey["words"])
            while out["status"] == "paused" and out["asks"]["kind"] == "question":
                out = s.run("d4", resume=self.journey["answers"].get(out["asks"]["field"], "I don't know"))
            out = s.run("d4", resume="no, the outcome is not that column")
        self.assertEqual((out["status"], out["confirmed"]), ("done", False))
        confirm = [e for e in self.log.read() if e.kind == "agent.step" and e.payload.get("tool") == "confirm"]
        self.assertEqual(len(confirm), 1)  # the correction is a recorded step of the walk, ended not confirmed

    def test_the_edge_reads_the_state_only_so_a_quiet_card_ends_at_the_analyst(self):  # the review's F3
        class Quiet(spike.Spike):
            def _card(self, state):
                return super()._card(state).model_copy(update={"questions": []})

        with checkpoints.open_checkpointer(self.home, database_url="") as saver:
            out = Quiet(saver).run("d5", self.journey["words"])
        self.assertEqual((out["status"], out["asks"]["kind"], out["asked"]), ("paused", "confirmation", []))

    @unittest.skipUnless(os.environ.get("DCLAB_DATABASE_URL"), "PostgreSQL checkpoints need DCLAB_DATABASE_URL")
    def test_postgres_holds_the_checkpoints_when_the_workspace_runs_on_it(self):
        self.assertEqual(checkpoints.where(self.home)["kind"], "postgres")
        with checkpoints.open_checkpointer(self.home) as saver:
            first = spike.Spike(saver).run("pg1", self.journey["words"])
        self.assertEqual(first["status"], "paused")
        with checkpoints.open_checkpointer(self.home) as saver:
            later = spike.Spike(saver).run("pg1", resume="I don't know")
        self.assertEqual(later["asked"][:1], [first["asks"]["field"]])


@unittest.skipIf(MISSING, MISSING)
class AdapterTests(unittest.TestCase):
    def test_a_node_asks_a_model_through_the_gateway_only(self):
        from dclab_rnd.models import FileUsage, gateway as gw_module

        with mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-never-logged", "OPENAI_MODEL": "std-model"}):
            Scripted.built = []
            gw, _ = gateway(FileUsage(Path(tempfile.mkdtemp()) / "usage.jsonl"), plan=[{"content": "  the outcome is churned ", "tool_calls": [], "usage": {"input_tokens": 3, "output_tokens": 2}}])
            gw_module.install(gw)
            self.addCleanup(gw_module.install, None)
            said = model.text("home_agent", [{"role": "user", "content": "which column?"}], draft_id="d9")
            self.assertEqual(said, "the outcome is churned")
            from dclab_rnd.models import settings

            self.assertEqual([(b.tier["name"], b.purpose) for b in Scripted.built], [("standard", settings.PURPOSES["home_agent"])])
            with self.assertRaises(KeyError):  # a purpose the gateway does not know is refused, never routed
                model.complete("orchestra_nonsense", [{"role": "user", "content": "x"}])

    def test_no_model_means_none_and_the_rules_path(self):
        from dclab_rnd.models import FileUsage, gateway as gw_module

        with mock.patch.dict(os.environ, CLEAN):  # no key for any tier
            gw, _ = gateway(FileUsage(Path(tempfile.mkdtemp()) / "usage.jsonl"))
            gw_module.install(gw)
            self.addCleanup(gw_module.install, None)
            self.assertIsNone(model.complete("home_agent", [{"role": "user", "content": "x"}]))


if __name__ == "__main__":
    unittest.main()
