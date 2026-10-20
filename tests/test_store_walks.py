"""Package 32.5: workflows as records.

A walk from a draft to its project is one WorkflowRecord: its draft events, its graph moves (a refused one kept as refused),
its stage runs, the ledger's decisions about the draft and the project, and the agent runs recorded for them, in time order,
each linking to its own record by id; a draft that never became a project is a walk that stopped in the draft; the walks
keeper keeps each record under ``raw/workflows/walks/``, a new version when the walk moved on.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401
    from fastapi.testclient import TestClient  # noqa: F401

    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class WalksTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.store import Layout

        self.home = Path(tempfile.mkdtemp()) / "agent_runs"
        self.home.mkdir()
        (self.home / "consent.json").write_text(json.dumps({"train": True}), encoding="utf-8")
        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        env = mock.patch.dict(os.environ, {"DCLAB_STORE": str(self.layout.root)})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(create_app(self.home))
        self.s = self.client.app.state.services
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def scripted_walk(self):
        from dclab_rnd.agents import harness
        from dclab_rnd.ledger import capture

        draft = self.s.drafts.create("Predict which orders arrive late")
        self.s.drafts.emit(draft["id"], "chat", {"step": "asked"})
        pid = self.client.post("/api/projects", json={"name": "Late orders", "goal": "late orders"}, headers=self.h).json()["id"]
        self.s.drafts.update(draft["id"], lambda d: d.update(project_id=pid))
        self.s.projects.transition(pid, {"at": "2099-01-01T00:00:01+00:00", "actor": "human", "move": "set_solution", "from": "WF-01", "to": "WF-02",
                                         "status": "allowed", "rules": [], "failed_checks": []})
        self.s.projects.transition(pid, {"at": "2099-01-01T00:00:02+00:00", "actor": "agent", "move": "read_holdout", "from": "WF-02", "to": "WF-02",
                                         "status": "blocked", "rules": ["DCLAB-R17"], "failed_checks": ["holdout_locked"]})
        with experience.bind(project=pid):
            capture.critic_weighed(draft["id"], "target", "rules", True, "the outcome is not named")
            with harness.session("conductor") as session:
                session.step("keep:test", {}, {"ok": True}, 0.0)
        return draft["id"], pid

    def test_a_walk_is_one_record_in_time_order_linking_every_part(self):
        from dclab_rnd.store import workflows

        draft_id, pid = self.scripted_walk()
        lone = self.s.drafts.create("Forecast demand")  # never became a project
        records = {r.draft or r.project: r for r in workflows.walks(self.s)}
        walk = records[draft_id]
        self.assertEqual(walk.project, pid)
        kinds = [s.kind for s in walk.steps]
        for kind in ("draft_event", "move", "refused", "agent_run"):
            self.assertIn(kind, kinds)
        self.assertEqual([s.at for s in walk.steps], sorted(s.at for s in walk.steps))
        refused = next(s for s in walk.steps if s.kind == "refused")
        self.assertEqual((refused.ref, refused.detail["failed_checks"]), ("read_holdout", ["holdout_locked"]))
        if self.s.ledger is not None:
            self.assertTrue(walk.decisions)  # the critic's decision about the draft
            self.assertTrue(set(walk.decisions) <= {s.ref for s in walk.steps if s.kind == "decision"})
        self.assertTrue(walk.agent_runs)
        self.assertIsNotNone(walk.workflow_run)
        self.assertEqual(walk.outcome, "built")
        self.assertEqual(records[lone["id"]].outcome, "stopped_in_draft")
        text = json.dumps(walk.model_dump(mode="json"))
        self.assertNotIn("Predict which orders arrive late", text)  # the person's words stay in their own records

    def test_the_walks_keeper_keeps_each_walk_and_a_new_version_when_it_moved_on(self):
        from dclab_rnd.store import Store, conductor
        from dclab_rnd.store.keepers import standard

        draft_id, pid = self.scripted_walk()
        keepers = {k: v for k, v in standard.install({}).items() if k == "walks"}
        conductor.conduct(self.s, self.layout, keepers)
        store = Store(self.layout)
        kept = [e for e in store.entries("workflows") if e.source == f"walks/{draft_id}.json"]
        self.assertEqual(len(kept), 1)
        self.assertEqual(json.loads(store.read(kept[0]))["project"], pid)
        self.s.projects.transition(pid, {"at": "2099-01-01T00:00:03+00:00", "actor": "human", "move": "approve_gate", "from": "WF-02",
                                         "to": "WF-03", "status": "allowed", "rules": [], "failed_checks": []})
        conductor.conduct(self.s, self.layout, keepers)
        self.assertEqual(len([e for e in store.entries("workflows") if e.source == f"walks/{draft_id}.json"]), 2)

    def test_without_a_server_the_walks_are_read_from_each_workspaces_folder_and_nothing_is_written_there(self):
        from dclab_rnd.store import Layout, Store, conductor
        from dclab_rnd.store.keepers import standard

        draft_id, pid = self.scripted_walk()
        before = sorted(str(p.relative_to(self.home)) for p in self.home.rglob("*") if p.is_dir())
        layout = Layout(Path(tempfile.mkdtemp()) / "store")
        keepers = {k: v for k, v in standard.install({}).items() if k == "walks"}
        conductor.conduct({"agent_runs": self.home, "test_runs": None, "claude_projects": None}, layout, keepers)  # a command-line run
        kept = [e for e in Store(layout).entries("workflows") if e.source == f"walks/{draft_id}.json"]
        self.assertEqual(len(kept), 1)
        walk = json.loads(Store(layout).read(kept[0]))
        self.assertEqual(walk["project"], pid)
        self.assertIn("refused", [s["kind"] for s in walk["steps"]])
        self.assertEqual(sorted(str(p.relative_to(self.home)) for p in self.home.rglob("*") if p.is_dir()), before)


if __name__ == "__main__":
    unittest.main()
