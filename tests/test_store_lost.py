"""Package 32.4: keepers for what was lost before.

A job run through three states leaves three lines in its history and the tasks keeper keeps them; an intern session saved
five times has five versions; a team's walk is kept with its board; a project deleted through the product is still whole
in the store with its deletion recorded beside it; events past retention are still in the store; the fitted students'
files are kept.
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
class LostTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout

        self.base = Path(tempfile.mkdtemp()) / "agent_runs"
        self.base.mkdir()
        (self.base / "consent.json").write_text(json.dumps({"train": True}), encoding="utf-8")
        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        env = mock.patch.dict(os.environ, {"DCLAB_STORE": str(self.layout.root), "DCLAB_STORE_CAPTURE": "1"})
        env.start()
        self.addCleanup(env.stop)
        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        experience.install(log)
        self.addCleanup(experience.install, None)

    def conduct(self):
        from dclab_rnd.store import conductor
        from dclab_rnd.store.keepers import standard

        return conductor.conduct({"agent_runs": self.base, "test_runs": Path(tempfile.mkdtemp())}, self.layout, standard.install({}))

    def test_a_jobs_states_and_a_sessions_saves_are_each_a_version_and_kept(self):
        from dclab_rnd.intern.sessions import SessionStore
        from dclab_rnd.jobs.store import open_jobs
        from dclab_rnd.store import Store

        jobs = open_jobs(self.base)
        job = jobs.enqueue("stage", "p1:data", {})
        jobs.claim("w1")
        jobs.beat([job["id"]], "w1")  # a heartbeat alone is no new state
        jobs.finish(job["id"], "done", worker="w1")
        history = [json.loads(line) for line in (self.base / "jobs_history.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([h["state"]["status"] for h in history], ["queued", "running", "done"])
        sessions = SessionStore(self.base / "intern")
        s = sessions.create("Look around", mode="standard", model=None)
        for i in range(4):
            s["steps"] = [{"n": n} for n in range(i + 1)]
            sessions.save(s)
        versions = (self.base / "intern" / "versions" / f"{s['id']}.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(versions), 5)
        self.assertEqual(len(sessions.list()), 1)  # the versions are not sessions
        (self.base / "orchestra").mkdir()
        (self.base / "orchestra" / "walks.jsonl").write_text('{"walk": "w1", "node": "understand", "board": {}}\n', encoding="utf-8")
        (self.base / "students").mkdir()
        (self.base / "students" / "register.jsonl").write_text('{"point": "column.time"}\n', encoding="utf-8")
        (self.base / "students" / "model.joblib").write_bytes(b"\x80fitted")
        record = self.conduct()
        self.assertEqual(record["failed"], [])
        store = Store(self.layout)
        self.assertEqual(store.whole("tasks", "dev", "jobs_history.jsonl"), (self.base / "jobs_history.jsonl").read_bytes())
        self.assertEqual(len(store.whole("conversations", "dev", f"intern/versions/{s['id']}.jsonl").splitlines()), 5)
        self.assertIn("orchestra/walks.jsonl", {e.source for e in store.entries("workflows")})
        self.assertEqual(store.whole("students", "dev", "students/model.joblib"), b"\x80fitted")  # the fitted file itself

    def test_a_deleted_project_stays_whole_in_the_store_with_its_deletion_recorded(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.store import Store
        from dclab_rnd.store.keepers.workspaces import discover

        home = self.base / "live_tests" / "run1"
        home.mkdir(parents=True)
        client = TestClient(create_app(home))
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        pid = client.post("/api/projects", json={"name": "Orders", "goal": "late orders"}, headers=h).json()["id"]
        name = next(w.name for w in discover(self.base) if w.root.resolve() == home.resolve())
        self.assertIn(client.delete(f"/api/projects/{pid}", headers=h).status_code, (200, 204))
        store = Store(self.layout)
        self.assertFalse((home / "projects" / pid).exists())  # removed from the workspace
        deletion = [e for e in store.entries("deletions") if e.workspace == name]
        self.assertEqual(len(deletion), 1)
        doc = json.loads(store.read(deletion[0]))
        self.assertEqual((doc["what"], doc["id"]), ("project", pid))
        if os.environ.get("DCLAB_DATABASE_URL"):  # the project was a row: its whole document is in the deletion record
            self.assertEqual((doc["doc"] or {}).get("id"), pid)
        else:  # the project was files: they were kept before they were removed
            self.assertIn(f"projects/{pid}/project.json", {e.source for e in store.entries("workflows") if e.workspace == name})
            self.assertGreaterEqual(doc["kept_first"]["kept"], 1)

    def test_events_past_retention_are_still_in_the_store(self):
        from dclab_rnd.experience import retention
        from dclab_rnd.store import Store

        log = experience.open_log(self.base)
        experience.record("visit.started", "person", {"surface": "home"}, log=log, at="2020-01-01T00:00:00.000000+00:00")
        self.assertTrue(list((self.base / "experience").rglob("*.jsonl")))
        out = retention.sweep_events(log, event_days=30)
        self.assertEqual(out["events_deleted"], 1)
        kept = b"".join(Store(self.layout).read(e) for e in Store(self.layout).entries("sessions"))
        self.assertIn(b"visit.started", kept)


    @unittest.skipUnless(os.environ.get("DCLAB_DATABASE_URL"), "the database keeper reads PostgreSQL")
    def test_on_postgresql_the_tables_are_kept_an_append_only_one_by_its_new_rows(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.store import Store

        home = self.base / "pgws"
        home.mkdir()
        client = TestClient(create_app(home))
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        pid = client.post("/api/projects", json={"name": "Orders", "goal": "late orders"}, headers=h).json()["id"]
        self.assertEqual(self.conduct()["failed"], [])
        store = Store(self.layout)
        projects = [e for e in store.entries("workflows") if e.source == "db/project.jsonl"]
        self.assertTrue(projects)
        self.assertIn(pid.encode(), store.whole("workflows", projects[-1].workspace, "db/project.jsonl"))
        audit = [e for e in store.entries("audit") if e.source == "db/audit_event.jsonl"]
        client.post("/api/projects", json={"name": "Second", "goal": "g"}, headers=h)
        self.conduct()
        after = [e for e in store.entries("audit") if e.source == "db/audit_event.jsonl" and e.workspace == (audit[-1].workspace if audit else None)]
        if audit:
            self.assertTrue(any(e.offset > 0 for e in after) or len(after) == len(audit))  # new rows kept as a tail


if __name__ == "__main__":
    unittest.main()
