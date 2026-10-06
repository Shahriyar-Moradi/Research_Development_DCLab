"""File storage (package 9.3): tables and artifacts behind one interface, recorded with their hash, checked on read."""
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.storage import files as F, open_stores  # noqa: E402
from dclab_rnd.studio import data as studio_data, engine  # noqa: E402


def table(n=120, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"age": rng.integers(18, 80, n), "spend": np.round(rng.gamma(2, 30, n), 2)})
    frame["churned"] = np.where(rng.random(n) < 0.3, "yes", "no")
    return frame


SOLUTION = {"target": "churned", "task": "binary", "positive_label": "yes", "prediction_moment": "At the monthly snapshot.",
            "forbidden": [], "identifiers": [], "time_column": None, "group_column": None, "text_columns": [], "metric": None, "notes": ""}


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.files = F.LocalFiles(self.root, F.JsonRecords(self.root))

    def test_put_open_exists_and_delete_with_a_record(self):
        rec = self.files.put("projects/abc/data/t.csv", b"a,b\n1,2\n")
        self.assertEqual((rec["key"], rec["size"], rec["content_type"], rec["backend"]), ("projects/abc/data/t.csv", 8, "text/csv", "local"))
        self.assertEqual(len(rec["sha256"]), 64)
        self.assertTrue(self.files.exists("projects/abc/data/t.csv"))
        with self.files.open("projects/abc/data/t.csv") as handle:
            self.assertEqual(handle.read(), b"a,b\n1,2\n")
        self.assertEqual(set(self.files.record("projects/abc/data/t.csv")), {"key", "size", "sha256", "content_type", "backend", "recorded"})  # no credential, no URL
        self.files.delete("projects/abc/data/t.csv")
        self.assertFalse(self.files.exists("projects/abc/data/t.csv"))
        self.assertIsNone(self.files.record("projects/abc/data/t.csv"))

    def test_a_changed_or_missing_file_is_refused(self):
        self.files.put("drafts/d1/data/x.csv", b"a\n1\n")
        (self.root / "drafts/d1/data/x.csv").write_bytes(b"a\n2\n")
        with self.assertRaises(F.FileError) as caught:
            self.files.path("drafts/d1/data/x.csv")
        self.assertIn("SHA-256", str(caught.exception))
        (self.root / "drafts/d1/data/x.csv").unlink()
        with self.assertRaises(F.FileError):
            self.files.path("drafts/d1/data/x.csv")

    def test_the_name_and_size_rules_hold(self):
        for key in ("../etc/passwd", "/abs/x.csv", "projects/../x.csv", "projects//x.csv", "projects/./x.csv", "a\\b/x.csv"):
            with self.assertRaises(F.FileError, msg=key):
                self.files.put(key, b"x")
        with mock.patch.object(F, "MAX_BYTES", 4):
            with self.assertRaises(F.FileError):
                self.files.put("projects/abc/data/big.csv", b"12345")

    def test_every_name_an_upload_keeps_is_a_key(self):
        for name in ("-sales.csv", "my data.csv", "Q1 & Q2.csv"):  # safe_name keeps a leading dash; a store's folder may hold a space
            self.assertEqual(self.files.put(f"projects/abc/data/{name}", b"x\n1\n")["key"], f"projects/abc/data/{name}")

    def test_a_rewrite_with_its_old_time_is_still_caught(self):
        import os
        rec = self.files.put("projects/abc/data/t.csv", b"a,b\n1,2\n")
        path = self.root / "projects/abc/data/t.csv"
        self.files.path("projects/abc/data/t.csv")  # hashed and remembered
        stat = path.stat()
        path.write_bytes(b"a,b\n9,9\n")  # the same size
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # and the old modification time (a restore, cp -p)
        with self.assertRaises(F.FileError):
            self.files.path("projects/abc/data/t.csv")
        self.assertEqual(len(rec["sha256"]), 64)

    def test_a_linked_folder_keeps_its_key(self):
        elsewhere = Path(tempfile.mkdtemp())
        (self.root / "projects").mkdir()
        (self.root / "projects" / "abc").symlink_to(elsewhere, target_is_directory=True)  # a project folder moved to another disk
        path = self.root / "projects" / "abc" / "t.csv"
        path.write_bytes(b"x\n1\n")
        self.assertEqual(self.files.key_of(path), "projects/abc/t.csv")
        with self.assertRaises(F.FileError):
            self.files.key_of(elsewhere / "t.csv")  # outside the workspace: a storage error, not a crash

    def test_forget_removes_every_record_under_a_prefix(self):
        for name in ("a.csv", "b.csv"):
            self.files.put(f"projects/p1/data/{name}", b"x\n1\n")
        self.files.put("projects/p10/data/c.csv", b"x\n1\n")
        self.assertEqual(sorted(self.files.forget("projects/p1")), ["projects/p1/data/a.csv", "projects/p1/data/b.csv"])
        self.assertIsNotNone(self.files.record("projects/p10/data/c.csv"))  # "p1" is not a prefix of "p10"


class MovedWorkspaceTests(unittest.TestCase):
    """The done-when: a project's data can be read after the workspace folder is moved; the hash is checked on read."""

    def test_the_data_is_read_after_the_folder_moves_and_a_changed_file_is_refused(self):
        home = Path(tempfile.mkdtemp()) / "workspace"
        projects = open_stores(home)[0]  # files, or PostgreSQL under test-pg
        pid = projects.create("Churn", "general", "Predict churn")["id"]
        table().to_csv(projects.data_dir(pid) / "t.csv", index=False)
        studio_data.attach_data(projects, pid, "t.csv")
        projects.update(pid, solution=SOLUTION)
        engine._BUNDLES.clear()
        self.assertEqual(len(engine.prepare(projects, projects.get(pid)).frame), 120)

        moved = home.parent / "moved"
        shutil.move(str(home), str(moved))
        projects = open_stores(moved)[0]  # a new process on the moved folder
        engine._BUNDLES.clear()
        project = projects.get(pid)  # the same workspace: the folder's marker, not its path
        self.assertEqual(len(engine.prepare(projects, project).frame), 120)

        path = projects.data_dir(pid) / "t.csv"
        path.write_text(path.read_text().replace("yes", "no", 1))  # the table changed on disk
        engine._BUNDLES.clear()
        with self.assertRaises(studio_data.DataError) as caught:
            engine.prepare(projects, project)
        self.assertIn("SHA-256", str(caught.exception))


class AttachTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.projects = open_stores(self.home)[0]
        self.pid = self.projects.create("t", "general", "Predict churn")["id"]
        table().to_csv(self.projects.data_dir(self.pid) / "good.csv", index=False)
        studio_data.attach_data(self.projects, self.pid, "good.csv")

    def test_a_name_starting_with_a_dash_attaches(self):
        table(seed=1).to_csv(self.projects.data_dir(self.pid) / "-sales.csv", index=False)
        studio_data.attach_data(self.projects, self.pid, "-sales.csv")
        project = self.projects.get(self.pid)
        self.assertEqual(project["data"]["filename"], "-sales.csv")
        self.assertTrue(studio_data.data_path(self.projects, project).is_file())

    def test_a_table_the_storage_refuses_leaves_the_old_one_in_place(self):
        table(seed=2).to_csv(self.projects.data_dir(self.pid) / "new.csv", index=False)
        with mock.patch.object(F.LocalFiles, "put", side_effect=RuntimeError('connection to server at "127.0.0.1", port 5432 failed for user "dclab"')):
            with self.assertRaises(studio_data.DataError) as caught:
                studio_data.attach_data(self.projects, self.pid, "new.csv")
        self.assertEqual(str(caught.exception), "The table could not be stored (RuntimeError)")  # no host, port or user
        project = self.projects.get(self.pid)
        self.assertEqual(project["data"]["filename"], "good.csv")
        self.assertTrue(studio_data.data_path(self.projects, project).is_file())  # still readable, still checked

    def test_a_changed_table_is_a_409_not_a_500(self):
        from fastapi.testclient import TestClient
        from dclab_rnd.agentic.server import create_app

        path = self.projects.data_dir(self.pid) / "good.csv"
        path.write_text(path.read_text().replace("yes", "no", 1))
        with TestClient(create_app(self.home)) as c:
            h = {"X-DCLab-Token": c.get("/api/config").json()["csrf"]}
            r = c.post(f"/api/projects/{self.pid}/solution/proposal", json={"target": "churned"}, headers=h)
            self.assertEqual(r.status_code, 409, r.text)
            self.assertIn("SHA-256", r.text)


class PipelineTests(unittest.TestCase):
    def test_a_storage_that_fails_never_stops_the_draft_pipeline(self):
        from dclab_rnd.draft import pipeline

        drafts = open_stores(Path(tempfile.mkdtemp()))[1]
        with mock.patch.object(F, "files_for", side_effect=RuntimeError("NoCredentialsError")):
            pipeline._record(drafts, Path("/nowhere/x.csv"))  # best effort: no exception


class FakeS3:
    def __init__(self):
        self.objects, self.meta = {}, {}

    def put_object(self, Bucket, Key, Body, ContentType, Metadata):
        self.objects[(Bucket, Key)] = Body.read()
        self.meta[(Bucket, Key)] = (ContentType, Metadata)

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        body = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(body), "ContentLength": len(body)}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {}

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


class S3Tests(unittest.TestCase):
    def setUp(self):
        self.root, self.s3 = Path(tempfile.mkdtemp()), FakeS3()
        self.files = F.S3Files(self.root, "s3://dclab-test/ws1", F.JsonRecords(self.root), client=self.s3)

    def test_a_write_reaches_the_bucket_and_a_missing_local_copy_is_fetched_and_checked(self):
        rec = self.files.put("projects/p1/data/t.csv", b"a\n1\n")
        self.assertEqual(rec["backend"], "s3")
        self.assertEqual(self.s3.meta[("dclab-test", "ws1/projects/p1/data/t.csv")], ("text/csv", {"sha256": rec["sha256"]}))
        (self.root / "projects/p1/data/t.csv").unlink()  # another machine, or a lost local copy
        self.assertEqual(self.files.path("projects/p1/data/t.csv").read_bytes(), b"a\n1\n")
        self.s3.objects[("dclab-test", "ws1/projects/p1/data/t.csv")] = b"a\n9\n"  # the object was changed
        (self.root / "projects/p1/data/t.csv").unlink()
        with self.assertRaises(F.FileError):
            self.files.path("projects/p1/data/t.csv")
        with mock.patch.object(F, "MAX_BYTES", 2):  # an object over the limit is refused before it is read
            self.s3.objects[("dclab-test", "ws1/projects/p1/data/t.csv")] = b"a\n1\n"
            self.assertFalse(self.files._fetch("projects/p1/data/t.csv", self.root / "projects/p1/data/t.csv"))
        self.files.delete("projects/p1/data/t.csv")
        self.assertNotIn(("dclab-test", "ws1/projects/p1/data/t.csv"), self.s3.objects)
        self.assertNotIn("dclab-test", json.dumps(self.files.records.get("projects/p1/data/t.csv")))  # no bucket or URL in a record

    def test_the_bucket_is_named_by_one_setting_and_boto3_is_needed_only_then(self):
        with mock.patch.dict("os.environ", {"DCLAB_FILES_URL": "s3://bucket/prefix"}):
            self.assertIsInstance(F.open_files(self.root), F.S3Files)
        with mock.patch.dict("os.environ", {"DCLAB_FILES_URL": ""}):
            self.assertIsInstance(F.open_files(self.root), F.LocalFiles)
        with self.assertRaises(F.FileError):
            F.S3Files(self.root, "https://not-a-bucket", F.JsonRecords(self.root))


class ExportTests(unittest.TestCase):
    def test_an_exported_notebook_is_recorded_by_key_not_by_absolute_path(self):
        from dclab_rnd.expansion.runner import fast_profile
        from dclab_rnd.intern.tools import Toolbox

        home = Path(tempfile.mkdtemp())
        projects = open_stores(home)[0]
        box = Toolbox(projects)
        pid = box.call("create_project", {"name": "t", "goal": "Predict churn"})["project_id"]
        table().to_csv(projects.data_dir(pid) / "t.csv", index=False)
        studio_data.attach_data(projects, pid, "t.csv")
        saved = box.call("set_solution", {**{k: v for k, v in SOLUTION.items() if k != "notes"}, "project_id": pid})
        self.assertTrue(saved.get("saved"), saved)
        with fast_profile():
            box.call("run_all", {"project_id": pid})
        out = box.call("export_notebook", {"project_id": pid})
        self.assertTrue(out["files"] and all(not f.startswith("/") and f.endswith(("notebook.ipynb", "report.md")) for f in out["files"]), out)
        files = F.files_for(projects)
        self.assertTrue(all(files.record(k) for k in out["files"]))


if __name__ == "__main__":
    unittest.main()
