"""Backups, retention and deletion (package 12.5): a backup restores into an empty database and folder with the same
rows and files, and refuses anything else; raw uploads past DCLAB_RETENTION_RAW_DAYS are deleted while the cleaned
table, its hash and every record stay; deleting a project or a draft leaves an audit entry."""
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pgtest  # noqa: E402

CSV = "tenure,monthly,plan,churned\n" + "".join(f"{i % 40},{20 + i % 7}.5,{'ab'[i % 2]},{'yes' if i % 3 == 0 else 'no'}\n" for i in range(60))


def old(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")


class RetentionTests(unittest.TestCase):
    """On file stores; RetentionOnPostgresTests runs the same on PostgreSQL."""

    def database(self) -> str:
        return ""

    def setUp(self):
        try:
            import pandas  # noqa: F401
        except ImportError as error:
            self.skipTest(f"pandas not installed: {error}")
        env = mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": self.database(), "DCLAB_FILES_URL": "", "DCLAB_RETENTION_RAW_DAYS": ""})
        env.start()
        self.addCleanup(env.stop)
        from dclab_rnd.storage import open_stores

        self.home = Path(tempfile.mkdtemp())
        self.projects, self.drafts, _ = open_stores(self.home)

    def ready_asset(self, days: int, name: str = "customers.csv") -> tuple[str, dict]:
        from dclab_rnd.draft import pipeline

        draft = self.drafts.create("Predict churn")
        (self.drafts.data_dir(draft["id"]) / name).write_text(CSV, encoding="utf-8")
        asset = pipeline.new_asset(self.drafts, draft["id"], "upload", name, name)
        final = pipeline.run(self.drafts, draft["id"], asset["id"])
        self.assertEqual(final["status"], "ready")
        pipeline.set_asset(self.drafts, draft["id"], asset["id"], added=old(days))
        return draft["id"], final

    def test_raw_uploads_past_the_days_go_and_the_cleaned_table_and_records_stay(self):
        from dclab_rnd import audit, retention

        stale_id, stale = self.ready_asset(40)
        fresh_id, _ = self.ready_asset(5)
        clean = self.drafts.data_dir(stale_id) / "clean.parquet"
        clean_sha = audit.sha256_of(clean)
        raw_sha = audit.sha256_of(self.drafts.data_dir(stale_id) / stale["filename"])
        self.assertEqual([r["draft_id"] for r in retention.sweep(self.drafts, 30, dry_run=True)], [stale_id])  # a dry run deletes nothing
        self.assertTrue((self.drafts.data_dir(stale_id) / stale["filename"]).exists())

        done = retention.sweep(self.drafts, 30)
        self.assertEqual([(r["draft_id"], r["sha256"]) for r in done], [(stale_id, raw_sha)])
        self.assertFalse((self.drafts.data_dir(stale_id) / stale["filename"]).exists())
        self.assertEqual(audit.sha256_of(clean), clean_sha)  # the cleaned table is untouched
        kept = next(a for a in self.drafts.get(stale_id)["assets"] if a["id"] == stale["id"])
        self.assertEqual((kept["status"], kept["rows"], kept["raw_sha256"]), ("ready", 60, raw_sha))
        self.assertTrue(kept["raw_deleted"])
        self.assertTrue((self.drafts.data_dir(fresh_id) / "customers.csv").exists())  # 5 days old: kept
        self.assertEqual(retention.sweep(self.drafts, 30), [])  # once only
        log = audit.for_store(self.drafts)
        _, rows = log.query(50, 0, kind="retention")
        self.assertEqual([(r["draft_id"], r["detail"]["sha256"]) for r in rows], [(stale_id, raw_sha)])
        _, imports = log.query(50, 0, kind="data_import", draft_id=stale_id)
        self.assertEqual(len(imports), 1)  # the import record stays

    def test_a_name_uploaded_twice_follows_its_newest_upload_and_changed_or_failed_files_stay(self):
        from dclab_rnd import audit, retention
        from dclab_rnd.draft import pipeline

        did, first = self.ready_asset(40)
        (self.drafts.data_dir(did) / "customers.csv").write_text(CSV + "1,1.5,a,no\n", encoding="utf-8")  # the same name uploaded again
        again = pipeline.new_asset(self.drafts, did, "upload", "customers.csv", "customers.csv")
        self.assertEqual(retention.sweep(self.drafts, 30), [])  # the file is the new upload's, still queued
        self.assertTrue((self.drafts.data_dir(did) / "customers.csv").exists())
        pipeline.set_asset(self.drafts, did, again["id"], status="ready", added=old(40))
        self.assertEqual([r["asset"] for r in retention.sweep(self.drafts, 30)], [again["id"]])  # once both are old: deleted once
        self.assertFalse((self.drafts.data_dir(did) / "customers.csv").exists())
        self.assertTrue(all(a.get("raw_deleted") for a in self.drafts.get(did)["assets"]))  # every asset of the name is marked

        changed_id, changed = self.ready_asset(40, "changed.csv")
        (self.drafts.data_dir(changed_id) / "changed.csv").write_text("other content\n", encoding="utf-8")  # not what the asset recorded
        failed_id, failed = self.ready_asset(40, "broken.csv")
        pipeline.set_asset(self.drafts, failed_id, failed["id"], status="failed")  # no cleaned table of its own: a retry needs the raw file
        self.assertEqual(retention.sweep(self.drafts, 30), [])
        self.assertTrue((self.drafts.data_dir(changed_id) / "changed.csv").exists())
        self.assertTrue((self.drafts.data_dir(failed_id) / "broken.csv").exists())

        resumed_id, resumed = self.ready_asset(40, "resumed.csv")  # a sweep that stopped between marking and deleting
        pipeline.set_asset(self.drafts, resumed_id, resumed["id"], raw_deleted="2026-01-01T00:00:00+00:00")
        self.assertEqual([r["asset"] for r in retention.sweep(self.drafts, 30)], [resumed["id"]])
        self.assertFalse((self.drafts.data_dir(resumed_id) / "resumed.csv").exists())
        _, rows = audit.for_store(self.drafts).query(50, 0, kind="retention")
        self.assertTrue(any(r["detail"].get("resumed") for r in rows))

    def test_a_file_is_deleted_only_when_its_content_is_the_expected_one(self):
        from dclab_rnd import retention

        folder = Path(tempfile.mkdtemp())
        (folder / "upload.csv").write_text("a new upload\n", encoding="utf-8")
        self.assertFalse(retention._retire(folder / "upload.csv", "0" * 64))  # not the content the sweep hashed: put back
        self.assertEqual((folder / "upload.csv").read_text(encoding="utf-8"), "a new upload\n")
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["upload.csv"])
        self.assertFalse(retention._retire(folder / "gone.csv", "0" * 64))  # another sweep took it

    def test_a_file_still_being_processed_is_never_touched_and_the_rule_is_off_by_default(self):
        from dclab_rnd import retention
        from dclab_rnd.draft import pipeline

        draft = self.drafts.create("Predict churn")
        (self.drafts.data_dir(draft["id"]) / "queued.csv").write_text(CSV, encoding="utf-8")
        asset = pipeline.new_asset(self.drafts, draft["id"], "upload", "queued.csv", "queued.csv")
        pipeline.set_asset(self.drafts, draft["id"], asset["id"], added=old(90))
        self.assertEqual(retention.sweep(self.drafts, 30), [])
        self.assertIsNone(retention.raw_days())
        env = type("Env", (), {"drafts": self.drafts})()
        self.assertEqual(retention.sweep_due(env, {}, 0.0), [])  # off: nothing swept
        with mock.patch.dict(os.environ, {"DCLAB_RETENTION_RAW_DAYS": "30"}):
            state = {}
            self.assertEqual(retention.raw_days(), 30)
            retention.sweep_due(env, state, 100.0)
            self.assertEqual(state["last"], 100.0)
            retention.sweep_due(env, state, 200.0)  # within the hour: not again
            self.assertEqual(state["last"], 100.0)
        with mock.patch.dict(os.environ, {"DCLAB_RETENTION_RAW_DAYS": "a month"}):
            from dclab_rnd.settings import Settings

            self.assertIsNone(retention.raw_days())
            self.assertTrue(any("DCLAB_RETENTION_RAW_DAYS" in p for p in Settings.load(self.home).problems()))


class RetentionOnPostgresTests(RetentionTests):
    def database(self) -> str:
        url = pgtest.require()
        pgtest.empty(url)
        return url


class DeletionTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        self.app = create_app(Path(tempfile.mkdtemp()))
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def test_deleting_a_project_or_a_draft_removes_its_files_and_leaves_an_audit_entry(self):
        pid = self.client.post("/api/projects", json={"name": "To delete", "goal": "g"}, headers=self.h).json()["id"]
        self.client.put(f"/api/projects/{pid}/data?filename=c.csv", content=CSV.encode(), headers=self.h)
        folder = self.app.state.services.projects.data_dir(pid).parent
        self.assertEqual(self.client.delete(f"/api/projects/{pid}", headers=self.h).status_code, 204)
        self.assertFalse(folder.exists())
        did = self.client.post("/api/drafts", json={"problem": "Predict churn"}, headers=self.h).json()["id"]
        draft_folder = self.app.state.services.drafts.data_dir(did).parent
        self.assertEqual(self.client.delete(f"/api/drafts/{did}", headers=self.h).status_code, 204)
        # no file of it is left (its agent, still starting, may make an empty folder again: an empty folder holds nothing)
        self.assertEqual([p for p in draft_folder.rglob("*") if p.is_file()] if draft_folder.exists() else [], [])
        rows = self.client.get("/api/platform/audit?kind=deletion").json()
        moves = {(e["move"], (e.get("project") or {}).get("id") or e.get("draft_id")) for e in rows["items"]}
        self.assertIn(("delete_project", pid), moves)
        self.assertIn(("delete_draft", did), moves)


class BackupTests(unittest.TestCase):
    """Against the PostgreSQL test database: a backup, then a restore into a new database and folder."""

    def setUp(self):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
            from dclab_rnd.storage import backup
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        self.url = pgtest.require()
        if backup.tool("pg_dump") is None or backup.tool("pg_restore") is None:
            self.skipTest("PostgreSQL's client tools (pg_dump, pg_restore) are not installed")
        pgtest.empty(self.url)
        import sqlalchemy as sa
        from sqlalchemy.engine import make_url

        from dclab_rnd.storage import db

        self.target = make_url(self.url).set(database=f"{make_url(self.url).database}_restore").render_as_string(hide_password=False)
        admin = sa.create_engine(self.url, isolation_level="AUTOCOMMIT")
        name = make_url(self.target).database
        try:
            with admin.connect() as c:
                c.execute(sa.text(f'drop database if exists "{name}"'))
                c.execute(sa.text(f'create database "{name}"'))
        except Exception as exc:  # noqa: BLE001 — a role that cannot create databases
            admin.dispose()
            self.skipTest(f"cannot create a scratch database: {type(exc).__name__}")

        def drop():
            db.engine(self.target).dispose()
            with admin.connect() as c:
                c.execute(sa.text(f'drop database if exists "{name}" with (force)'))
            admin.dispose()
        self.addCleanup(drop)
        env = mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": self.url})
        env.start()
        self.addCleanup(env.stop)
        self.home = Path(tempfile.mkdtemp())
        self.app = create_app(self.home)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}
        self.pid = self.client.post("/api/projects", json={"name": "Backed up", "goal": "Predict churn"}, headers=h).json()["id"]
        self.assertEqual(self.client.put(f"/api/projects/{self.pid}/data?filename=c.csv", content=CSV.encode(), headers=h).status_code, 200)
        self.client.post("/api/drafts", json={"problem": "Predict churn"}, headers=h)

    def test_a_backup_restores_into_an_empty_database_and_folder_with_the_same_rows_and_files(self):
        from dclab_rnd.storage import backup, db, open_stores
        from dclab_rnd.studio import data as studio_data

        made = backup.backup(self.url, self.home, Path(tempfile.mkdtemp()))
        self.assertGreater(made["rows"]["project"], 0)
        self.assertGreater(made["rows"]["audit_event"], 0)
        self.assertGreater(made["files"], 0)
        restored_home = Path(tempfile.mkdtemp()) / "restored"
        report = backup.restore(Path(made["folder"]), self.target, restored_home)
        self.assertEqual(report["differences"], [])
        self.assertEqual(report["rows"], made["rows"])
        self.assertEqual(report["schema"], db.head())
        with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": self.target}):  # the restored workspace works: same id, same table
            projects, _, _ = open_stores(restored_home)
            project = projects.get(self.pid)
            self.assertTrue(studio_data.data_path(projects, project).is_file())
        with self.assertRaises(backup.BackupError) as again:  # never over data
            backup.restore(Path(made["folder"]), self.target, Path(tempfile.mkdtemp()))
        self.assertIn("not empty", str(again.exception))

    def test_linked_folders_are_followed_and_a_failed_backup_leaves_nothing(self):
        from dclab_rnd.storage import backup

        outside = Path(tempfile.mkdtemp())
        (outside / "kept.csv").write_text(CSV, encoding="utf-8")
        (self.home / "linked").symlink_to(outside, target_is_directory=True)  # a folder on another disk
        made = backup.backup(self.url, self.home, Path(tempfile.mkdtemp()))
        restored = Path(tempfile.mkdtemp()) / "r"
        self.assertEqual(backup.restore(Path(made["folder"]), self.target, restored)["differences"], [])
        self.assertEqual((restored / "linked" / "kept.csv").read_text(encoding="utf-8"), CSV)
        self.assertFalse((restored / "linked").is_symlink())
        import sqlalchemy as sa
        from sqlalchemy.engine import make_url

        from dclab_rnd.storage import db

        db.engine(self.target).dispose()
        admin = sa.create_engine(self.url, isolation_level="AUTOCOMMIT")
        name = make_url(self.target).database
        with admin.connect() as c:  # an empty database again: the folder is what must refuse now
            c.execute(sa.text(f'drop database "{name}" with (force)'))
            c.execute(sa.text(f'create database "{name}"'))
        admin.dispose()
        with self.assertRaises(backup.BackupError) as busy:  # never over files either
            backup.restore(Path(made["folder"]), self.target, restored)
        self.assertIn("target folder is not empty", str(busy.exception))
        with self.assertRaises(backup.BackupError) as inside:
            backup.backup(self.url, self.home, self.home / "backups")
        self.assertIn("inside the workspace", str(inside.exception))

        out = Path(tempfile.mkdtemp())
        with mock.patch.object(backup, "_run", side_effect=backup.BackupError("pg_dump failed: no space")):
            with self.assertRaises(backup.BackupError):
                backup.backup(self.url, self.home, out)
        self.assertEqual(list(out.iterdir()), [])  # no half-made backup left looking like one
        with mock.patch.object(backup, "_files", return_value=["gone/meanwhile.csv"]):
            made = backup.backup(self.url, self.home, Path(tempfile.mkdtemp()))
        self.assertEqual((made["files"], made["vanished"]), (0, ["gone/meanwhile.csv"]))

    def test_a_damaged_backup_restores_nothing(self):
        from dclab_rnd.storage import backup

        made = Path(backup.backup(self.url, self.home, Path(tempfile.mkdtemp()))["folder"])
        with open(made / "files.tar.gz", "ab") as f:
            f.write(b"x")
        with self.assertRaises(backup.BackupError) as damaged:
            backup.restore(made, self.target, Path(tempfile.mkdtemp()) / "r")
        self.assertIn("damaged", str(damaged.exception))
        self.assertNotIn("dclab-ci-only", str(damaged.exception))


class ConnectionTests(unittest.TestCase):
    def test_the_newest_client_tools_are_picked(self):
        """A dump needs a pg_dump at least as new as the server: a PATH that points at an older one must not win."""
        try:
            from dclab_rnd.storage import backup
        except ImportError as error:
            self.skipTest(str(error))
        versions = {"/old/pg_dump": 15, "/new/pg_dump": 16}
        with mock.patch.dict(os.environ, {"DCLAB_PG_BIN": ""}), mock.patch.object(backup.shutil, "which", return_value="/old/pg_dump"), \
                mock.patch("glob.glob", side_effect=lambda pattern: ["/new/pg_dump"] if "homebrew" in pattern else []), \
                mock.patch.object(backup, "_major", side_effect=lambda path: versions[path]):
            self.assertEqual(backup.tool("pg_dump"), "/new/pg_dump")
        with mock.patch.object(backup.subprocess, "run", return_value=type("R", (), {"stdout": "pg_dump (PostgreSQL) 16.15 (Homebrew)\n"})()):
            self.assertEqual(backup._major("/any/pg_dump"), 16)

    def test_the_password_and_tls_options_go_in_the_environment(self):
        try:
            from dclab_rnd.storage import backup
        except ImportError as error:
            self.skipTest(str(error))
        env = backup._libpq("postgresql+psycopg://dclab:not-a-real-pass@db.internal:6432/dclab?sslmode=verify-full&sslrootcert=/ca.pem")
        self.assertEqual((env["PGHOST"], env["PGPORT"], env["PGPASSWORD"], env["PGSSLMODE"], env["PGSSLROOTCERT"]),
                         ("db.internal", "6432", "not-a-real-pass", "verify-full", "/ca.pem"))
        env = backup._libpq("postgresql+psycopg://me@/dclab_test?host=/tmp&port=5433")  # a socket folder and a port in the query
        self.assertEqual((env["PGHOST"], env["PGPORT"], env["PGDATABASE"]), ("/tmp", "5433", "dclab_test"))


if __name__ == "__main__":
    unittest.main()
