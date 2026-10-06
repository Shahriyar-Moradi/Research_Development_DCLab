"""One audit trail (package 10.4): an append-only log per workspace, on files and PostgreSQL, written where each
governed action happens: data imports with their source and hash, solutions, and the Home agent's refused moves."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pgtest  # noqa: E402
from dclab_rnd import audit  # noqa: E402


class LogContract:
    def log(self):
        raise NotImplementedError

    def test_append_read_filter_and_count(self):
        log = self.log()
        self.assertTrue(log.empty())
        audit.record(log, "move", "agent", project={"id": "p1", "name": "Churn"}, move="run_stage", status="blocked", message="no data")
        audit.record(log, "approval", "human", "Owner", project={"id": "p1", "name": "Churn"}, move="approve_gate", status="allowed")
        audit.record(log, "data_import", "human", draft_id="d1", move="add_asset", status="allowed", sha256="ab" * 32)
        self.assertFalse(log.empty())
        total, rows = log.query(10, 0)
        self.assertEqual((total, [r["kind"] for r in rows]), (3, ["data_import", "approval", "move"]))  # newest first
        self.assertEqual(rows[0]["detail"]["sha256"], "ab" * 32)
        self.assertEqual(log.query(10, 0, project_id="p1")[0], 2)
        self.assertEqual([r["kind"] for r in log.query(10, 0, status="blocked", actor="agent")[1]], ["move"])
        self.assertEqual([r["kind"] for r in log.query(1, 1)[1]], ["approval"])
        self.assertEqual({k: v for k, v in log.counts().items()}, {"move": 1, "approval": 1, "data_import": 1, "status:blocked": 1})


class FileLogTests(LogContract, unittest.TestCase):
    def log(self):
        return audit.FileAudit(Path(tempfile.mkdtemp()) / "audit.jsonl")

    def test_a_line_cut_by_a_crash_does_not_hide_the_rest(self):
        log = self.log()
        audit.record(log, "move", "human", move="a")
        with log.path.open("a") as f:
            f.write('{"cut": ')
        audit.record(log, "move", "human", move="b")  # appended after the broken line
        self.assertEqual([r["move"] for r in log.query(10, 0)[1]], ["b", "a"])


class PgLogTests(LogContract, unittest.TestCase):
    def log(self):
        from dclab_rnd.storage import db

        url = pgtest.require()
        return audit.PgAudit(db.workspace_for(Path(tempfile.mkdtemp()), url), url=url)

    def test_the_database_refuses_to_change_or_delete_an_entry(self):
        import sqlalchemy as sa

        from dclab_rnd.storage.models import audit_event as t

        log = self.log()
        audit.record(log, "policy", "human", move="set_policy", status="allowed")
        for statement in (sa.update(t).values(who="Someone else"), sa.delete(t)):
            with self.assertRaises(sa.exc.DBAPIError) as refused, log.engine.begin() as c:
                c.execute(statement.where(t.c.workspace_id == log.workspace_id))
            self.assertIn("append-only", str(refused.exception))
        self.assertEqual(log.query(10, 0)[0], 1)


class WiringTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
            from test_studio import CHURN_CONTRACT, churn_frame
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.s = self.client.app.state.services
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}
        self.contract, self.frame = CHURN_CONTRACT, churn_frame

    def rows(self, **filters):
        return self.s.audit.query(100, 0, **filters)[1]

    def test_a_projects_table_and_solution_are_audited_with_their_hashes(self):
        from dclab_rnd.studio.graph import solution_hash

        pid = self.client.post("/api/projects", json={"name": "Churn", "goal": "Predict churn"}, headers=self.h).json()["id"]
        self.client.put(f"/api/projects/{pid}/data?filename=c.csv", content=self.frame(300).to_csv(index=False).encode(), headers=self.h)
        project = self.s.projects.get(pid)
        (imported,) = self.rows(kind="data_import")
        self.assertEqual((imported["project_id"], imported["detail"]["sha256"], imported["detail"]["source"]),
                         (pid, project["data"]["sha256"], {"kind": "upload"}))
        self.assertEqual(self.client.put(f"/api/projects/{pid}/solution", json=self.contract, headers=self.h).status_code, 200)
        self.client.put(f"/api/projects/{pid}/solution", json=self.contract, headers=self.h)  # unchanged: nothing new
        (saved,) = self.rows(kind="solution")
        self.assertEqual((saved["detail"]["target"], saved["detail"]["solution_hash"]),
                         (self.contract["target"], solution_hash(self.s.projects.get(pid)["solution"])))
        self.assertEqual([r["move"] for r in self.rows(kind="move")], ["set_solution"])  # the validated move beside it

    def test_a_drafts_file_and_the_home_agents_refusals_are_audited(self):
        from dclab_rnd.agents.registry import Tool
        from dclab_rnd.draft.chat import DraftGuard, Turn

        draft = self.client.post("/api/drafts", json={"problem": "Predict which customers churn next month"}, headers=self.h).json()
        body = b"a,b\n1,2\n3,4\n"
        self.client.put(f"/api/drafts/{draft['id']}/data?filename=c.csv", content=body, headers=self.h)
        (imported,) = self.rows(kind="data_import")
        import hashlib

        self.assertEqual((imported["draft_id"], imported["detail"]["sha256"], imported["detail"]["source"]["kind"]),
                         (draft["id"], hashlib.sha256(body).hexdigest(), "upload"))
        agent = self.s.draft_work.agent(draft["id"])
        tool = mock.Mock(spec=Tool, move="set_pack")
        refused = DraftGuard()(tool, Turn(agent, draft["id"]), {"key": "no-such-pack"}, lambda: self.fail("must not run"))
        self.assertIn("error", refused)
        (move,) = [r for r in self.rows(kind="move") if r["draft_id"] == draft["id"]]
        self.assertEqual((move["actor"], move["who"], move["move"], move["status"]), ("agent", "Home agent", "set_pack", "blocked"))
        self.assertNotIn("args", move["detail"])  # what the model wrote stays out
        self.assertNotIn("no-such-pack", json.dumps(move))

    def test_a_holdout_approval_is_audited_as_used_only_once_the_project_saved_it(self):
        from dclab_rnd.studio import engine, graph

        pid = self.s.projects.create("Holdout", "general", "")["id"]
        p = self.s.projects.get(pid)
        p["approvals"] = [{"gate": "holdout", "by": "owner", "at": "2026-10-01T00:00:00+00:00", "reason": "ok"}]
        self.s.projects.save(p)

        def saved(store, project_id, stage, project, reuse_reason):
            store.save(project)
            return {"setup_summary": "ran"}
        with mock.patch.object(graph, "check", return_value=mock.Mock(allowed=True)), mock.patch.object(graph, "log"):
            with mock.patch.object(engine, "_execute", side_effect=ValueError("the table's hash does not match")), self.assertRaises(ValueError):
                engine.execute(self.s.projects, pid, "final", actor="agent")
            self.assertEqual(self.rows(kind="approval_used"), [])  # the project still holds the approval unused
            with mock.patch.object(engine, "_execute", side_effect=saved):
                engine.execute(self.s.projects, pid, "final", actor="agent")
        (used,) = self.rows(kind="approval_used")
        self.assertEqual((used["actor"], used["detail"]["approved_at"]), ("agent", "2026-10-01T00:00:00+00:00"))

    def test_switches_routing_and_a_failing_audit(self):
        pid = self.client.post("/api/projects", json={"name": "Switches", "goal": "g"}, headers=self.h).json()["id"]
        self.client.patch(f"/api/projects/{pid}", json={"settings": {"share_for_training": True}}, headers=self.h)
        (opt,) = self.rows(kind="policy")
        self.assertEqual((opt["move"], opt["detail"]["args"]["on"]), ("set_training_opt_in", True))
        for _ in range(2):  # the same setting again changes nothing: no row
            self.assertEqual(self.client.post("/api/models/routing", json={"purpose": "intern", "setting": "shadow", "tier": None}, headers=self.h).status_code, 200)
        self.assertEqual(self.rows(kind="routing"), [])
        broken = mock.Mock(append=mock.Mock(side_effect=OSError("disk full")))
        audit.record(broken, "move", "human", move="x")  # reported on stderr; the action it records stands
        self.assertIsNone(audit.for_store(mock.Mock(spec=["home"], home=Path(tempfile.mkdtemp()))))  # a store made outside a workspace writes none

    def test_synthetic_data_the_home_agent_asked_for_is_the_agents(self):
        draft = self.client.post("/api/drafts", json={"problem": "Predict which customers churn next month"}, headers=self.h).json()
        self.s.draft_work.requests(draft["id"], "simulate", {"prompt": "churn", "rows": 200})
        rows = [r for r in self.rows(kind="data_import") if r["draft_id"] == draft["id"]]
        self.assertEqual([(r["actor"], r["who"], r["detail"]["synthetic"]) for r in rows], [("agent", "Home agent", True)])


if __name__ == "__main__":
    unittest.main()
