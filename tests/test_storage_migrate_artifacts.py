"""Review of 96e47f43: moving a file workspace into the database keeps a project's approved model exportable. Its
``artifacts/`` (the captured model, serving record and parity rows) move through file storage with their digests, and its
``secrets/`` (the key that hashes the artifact's categories, the service log's key) move as files only the owner reads."""
import hashlib
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402


class MigrateArtifactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        pgtest.empty(cls.url)
        cls.source = Path(tempfile.mkdtemp()) / "files-workspace"
        with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": ""}):
            from dclab_rnd.storage import open_stores

            projects = open_stores(cls.source)[0]
            cls.pid = projects.create("A served project", "general", "x")["id"]
        folder = cls.source / "projects" / cls.pid
        cls.model = folder / "artifacts" / "run1" / "model.pkl"
        cls.model.parent.mkdir(parents=True)
        cls.model.write_bytes(b"pickled model bytes")
        (cls.model.parent / "serving_meta.json").write_text('{"model_id": "x"}\n')
        secrets = folder / "secrets"
        secrets.mkdir(mode=0o700)
        cls.keys = {"service_artifact_run1.key": "a" * 32, "service_log.key": "b" * 32}
        for name, value in cls.keys.items():
            (secrets / name).write_text(value)
            os.chmod(secrets / name, 0o600)
        cls.target = Path(tempfile.mkdtemp()) / "db-workspace"
        from dclab_rnd.storage.migrate import migrate

        with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": cls.url}):
            cls.first = migrate(cls.source, cls.target, cls.url)
            cls.second = migrate(cls.source, cls.target, cls.url)

    def test_the_artifacts_are_copied_and_recorded_with_their_digest(self):
        from dclab_rnd.storage.files import LocalFiles, PgRecords
        from dclab_rnd.storage import db

        self.assertTrue(self.first["ok"], self.first)
        files = LocalFiles(self.target, PgRecords(db.workspace_for(self.target, self.url), self.url))
        key = f"projects/{self.pid}/artifacts/run1/model.pkl"
        record = files.records.get(key)
        self.assertIsNotNone(record)
        self.assertEqual(record["sha256"], hashlib.sha256(b"pickled model bytes").hexdigest())
        self.assertEqual(files.path(key, expected=record["sha256"]).read_bytes(), b"pickled model bytes")
        self.assertIsNotNone(files.records.get(f"projects/{self.pid}/artifacts/run1/serving_meta.json"))

    def test_the_secrets_are_copied_for_the_owner_only_and_never_recorded(self):
        from dclab_rnd.storage.files import PgRecords
        from dclab_rnd.storage import db

        secrets = self.target / "projects" / self.pid / "secrets"
        self.assertEqual(stat.S_IMODE(secrets.stat().st_mode), 0o700)
        for name, value in self.keys.items():
            path = secrets / name
            self.assertEqual(path.read_text(), value)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            records = PgRecords(db.workspace_for(self.target, self.url), self.url)
            self.assertIsNone(records.get(f"projects/{self.pid}/secrets/{name}"))  # a key is a file, never a stored record
        self.assertEqual(self.first["files"]["secrets"], 2)

    def test_a_rerun_copies_nothing_again_and_keeps_a_target_s_own_key(self):
        self.assertTrue(self.second["ok"], self.second)
        self.assertEqual((self.second["files"]["copied"], self.second["files"].get("secrets", 0)), (0, 0))
        held = self.target / "projects" / self.pid / "secrets" / "service_log.key"
        held.write_text("c" * 32)  # the database workspace made its own since
        from dclab_rnd.storage.migrate import migrate

        with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": self.url}):
            third = migrate(self.source, self.target, self.url)
        self.assertEqual(held.read_text(), "c" * 32)  # never overwritten
        self.assertTrue(any("secret" in d for d in third["differences"]), third["differences"])
        held.write_text("b" * 32)

    def test_a_missing_secret_is_a_difference(self):
        from dclab_rnd.storage import db
        from dclab_rnd.storage.migrate import Source, verify

        moved = self.target / "projects" / self.pid / "secrets" / "service_artifact_run1.key"
        kept = moved.read_bytes()
        moved.unlink()
        try:
            with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": self.url}):
                lacks = verify(Source(self.source), self.target, db.workspace_for(self.target, self.url), self.url)
            self.assertTrue(any("service_artifact_run1.key" in d for d in lacks), lacks)
        finally:
            moved.write_bytes(kept)
            os.chmod(moved, 0o600)


if __name__ == "__main__":
    unittest.main()
