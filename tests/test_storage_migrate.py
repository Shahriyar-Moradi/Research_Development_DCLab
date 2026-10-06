"""Moving a file workspace into the database (package 9.4): every page shows the same, the source is not changed,
running it twice duplicates nothing."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402


def snapshot(folder: Path) -> dict[str, str]:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


def pages(home: Path, env: dict[str, str]) -> dict:
    """What the product's pages read, from a server on ``home`` (files, or the database with ``env``)."""
    from fastapi.testclient import TestClient
    from dclab_rnd.agentic.server import create_app

    with mock.patch.dict(os.environ, env), TestClient(create_app(home)) as c:
        projects = c.get("/api/projects").json()  # in the order the page shows them
        detail = {}
        for p in projects:
            d = c.get(f"/api/projects/{p['id']}").json()
            detail[p["id"]] = {"name": d["name"], "solution": d.get("solution"), "data": (d.get("data") or {}).get("sha256"),
                               "stages": {k: v.get("status") for k, v in d["stages"].items()}, "records": sorted(d.get("records") or {}),
                               "transitions": [(t["move"], t["status"], t["at"]) for t in d["transitions"]],
                               "activity": [(a["kind"], a["at"]) for a in d["activity"]], "memory": d.get("memory")}
        drafts = [(d["id"], d.get("problem"), d.get("status")) for d in c.get("/api/drafts").json()]
        from dclab_rnd.storage import open_stores
        store = open_stores(home)[1]  # the events route is a stream: read the same log through the store the server uses
        events = {d[0]: [(e["seq"], e["kind"], e["at"]) for e in store.events(d[0])] for d in drafts}
        sessions = [(s["id"], s.get("status"), s.get("project_id")) for s in c.get("/api/intern/sessions").json()]
        traces = {s[0]: [(r["tool"], r["verdict"]) for r in c.get(f"/api/intern/sessions/{s[0]}").json().get("trace", [])] for s in sessions}
        lessons = sorted(l["id"] for l in c.get("/api/lessons").json()["lessons"])
        usage = (c.get("/api/models").json().get("usage") or {}).get("requests")
    return {"projects": [(p["id"], p["name"]) for p in projects], "detail": detail, "drafts": drafts, "events": events,
            "sessions": sessions, "traces": traces, "lessons": lessons, "usage": usage}


class MigrateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        pgtest.empty(cls.url)
        cls.files_env = {"DCLAB_DATABASE_URL": ""}
        cls.db_env = {"DCLAB_DATABASE_URL": cls.url}
        cls.source = Path(tempfile.mkdtemp()) / "files-workspace"
        with mock.patch.dict(os.environ, cls.files_env):
            cls.build_source(cls.source)
        cls.before = snapshot(cls.source)
        cls.target = Path(tempfile.mkdtemp()) / "db-workspace"
        from dclab_rnd.storage.migrate import migrate
        with mock.patch.dict(os.environ, cls.db_env):
            cls.first = migrate(cls.source, cls.target, cls.url)
            cls.second = migrate(cls.source, cls.target, cls.url)
        cls.after = snapshot(cls.source)  # before any test starts a server on the source (that server writes its own files)

    @staticmethod
    def build_source(home: Path):
        from dclab_rnd.agents.traces import open_traces
        from dclab_rnd.expansion.runner import fast_profile
        from dclab_rnd.intern import Intern
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.lessons import FileLessons
        from dclab_rnd.models.usage import FileUsage
        from dclab_rnd.storage import open_stores

        projects, drafts, sessions = open_stores(home)
        intern = Intern(sessions, Toolbox(projects), None, traces=open_traces(home))
        with fast_profile():
            s = intern.run(intern.start("Predict which telco customers churn")["id"])
        for when, problem in (("2026-09-01T10:00:00+00:00", "An older draft"), ("2026-09-20T10:00:00+00:00", "A middle draft")):
            older = drafts.create(problem)
            path = home / "drafts" / older["id"] / "draft.json"
            doc = json.loads(path.read_text())
            doc["updated"] = when
            path.write_text(json.dumps(doc))
        old_project = projects.create("An older project", "general", "x")
        path = home / "projects" / old_project["id"] / "project.json"
        doc = json.loads(path.read_text())
        doc["updated"] = "2026-09-01T10:00:00+00:00"
        path.write_text(json.dumps(doc))
        d = drafts.create("Predict churn next month")
        drafts.emit(d["id"], "message", {"role": "user", "text": "Predict churn next month"})
        drafts.emit(d["id"], "pipeline", {"step": "structured"})
        FileLessons(home / "lessons.json").save({"id": "abc123def0", "project_id": s["project_id"], "status": "proposed", "claim": "c",
                                                 "against": "a", "next_test": "n", "cites": [], "scope": {}, "synthetic": False, "proposed_at": "2026-10-06T10:00:00+00:00"})
        FileUsage(home / "model_requests.jsonl").record({"at": "2026-10-06T10:00:00+00:00", "purpose": "intern", "tier": "strong", "model": "m",
                                                          "endpoint": "e", "input_tokens": 10, "output_tokens": 2, "seconds": 1.0, "attempts": 1, "outcome": "ok"})

    def test_every_page_shows_the_same_projects_records_and_logs(self):
        self.assertTrue(self.first["ok"], self.first)
        self.assertEqual(len(self.first["projects"]["copied"]), 2)
        self.assertGreater(self.first["logs"]["audit"]["copied"], 0)  # the audit trail moves with the workspace (10.4)
        self.assertEqual(pages(self.source, self.files_env), pages(self.target, self.db_env))

    def test_the_source_is_not_changed(self):
        self.assertEqual(self.after, self.before)  # two migrations read it and wrote nothing
        self.assertFalse((self.source / ".dclab_workspace").exists())

    def test_running_it_twice_duplicates_nothing(self):
        self.assertTrue(self.second["ok"], self.second)
        self.assertEqual((self.second["projects"]["copied"], len(self.second["projects"]["already"])), ([], 2))
        self.assertEqual(self.second["files"]["copied"], 0)  # nothing copied over what the target holds
        self.assertEqual(self.second["sessions"]["copied"], 0)
        self.assertEqual(self.second["logs"]["agent_steps"]["copied"], 0)
        self.assertEqual(self.second["logs"]["audit"]["copied"], 0)
        self.assertEqual(pages(self.source, self.files_env), pages(self.target, self.db_env))  # still the same: nothing doubled

    def test_the_data_files_are_copied_and_recorded_with_their_hash(self):
        from dclab_rnd.storage import open_stores
        from dclab_rnd.studio import data as studio_data

        with mock.patch.dict(os.environ, self.db_env):
            projects = open_stores(self.target)[0]
            project = next(projects.get(p) for p in self.first["projects"]["copied"] if projects.get(p).get("data"))
            self.assertTrue(studio_data.data_path(projects, project).is_file())  # read through file storage, hash checked
        self.assertGreaterEqual(self.first["files"]["copied"], 2)  # the table and the exported notebook and report

    def test_a_missing_copy_is_reported(self):
        from dclab_rnd.storage.migrate import Source, verify
        from dclab_rnd.storage import db

        pid = next(p for p in self.first["projects"]["copied"] if (self.target / "projects" / p / "data").is_dir() and any((self.target / "projects" / p / "data").iterdir()))
        table = next((self.target / "projects" / pid / "data").iterdir())
        original = table.read_bytes()
        try:
            table.unlink()  # the copy is lost
            with mock.patch.dict(os.environ, self.db_env):
                problems = verify(Source(self.source), self.target, db.workspace_for(self.target, self.url), self.url)
            self.assertTrue(any(p.endswith("not in the database workspace") for p in problems), problems)
        finally:
            table.write_bytes(original)

    def test_the_command_refuses_a_target_that_is_the_source_or_inside_it(self):
        from dclab_rnd.storage.migrate import migrate
        for target in (self.source, self.source / "inside", self.source.parent):
            with mock.patch.dict(os.environ, self.db_env), self.assertRaises(SystemExit):
                migrate(self.source, target, self.url)
        self.assertFalse((self.source / "inside").exists())

    def test_another_workspace_s_ids_are_conflicts_not_a_crash(self):
        from dclab_rnd.storage.migrate import migrate
        with mock.patch.dict(os.environ, self.db_env):
            other = migrate(self.source, Path(tempfile.mkdtemp()) / "another", self.url)
        self.assertFalse(other["ok"])
        self.assertEqual(len(other["projects"]["conflict"]), 2)
        self.assertEqual(other["lessons"]["conflict"], ["abc123def0"])
        self.assertEqual(other["failed"], [])

    def test_a_missing_log_is_a_difference_and_a_rerun_copies_it(self):
        import sqlalchemy as sa
        from dclab_rnd.storage import db
        from dclab_rnd.storage.migrate import Source, migrate, verify
        from dclab_rnd.storage.models import agent_step

        wid = db.workspace_for(self.target, self.url)
        with db.engine(self.url).begin() as c:
            c.execute(sa.delete(agent_step).where(agent_step.c.workspace_id == wid))  # a run that stopped before the logs
        with mock.patch.dict(os.environ, self.db_env):
            problems = verify(Source(self.source), self.target, wid, self.url)
            self.assertTrue(any(p.startswith("log agent_steps") for p in problems), problems)
            again = migrate(self.source, self.target, self.url)
        self.assertTrue(again["ok"], again)
        self.assertGreater(again["logs"]["agent_steps"]["copied"], 0)

    def test_zz_a_rerun_after_the_target_was_used_keeps_its_newer_file(self):
        from dclab_rnd.storage import db
        from dclab_rnd.storage.files import LocalFiles, PgRecords
        from dclab_rnd.storage.migrate import migrate

        files = LocalFiles(self.target, PgRecords(db.workspace_for(self.target, self.url), self.url))
        pid = next(p for p in self.first["projects"]["copied"] if (self.target / "projects" / p / "data").is_dir() and any((self.target / "projects" / p / "data").iterdir()))
        table = next((self.target / "projects" / pid / "data").iterdir())
        newer = files.put(files.key_of(table), b"a,b\n1,2\n")  # re-uploaded in the database workspace after the move
        with mock.patch.dict(os.environ, self.db_env):
            again = migrate(self.source, self.target, self.url)
        self.assertEqual(table.read_bytes(), b"a,b\n1,2\n")  # not copied over
        self.assertEqual(files.record(files.key_of(table))["sha256"], newer["sha256"])
        self.assertTrue(again["ok"], again)
        self.assertTrue(any("changed there since the move" in n for n in again["notes"]))


if __name__ == "__main__":
    unittest.main()
