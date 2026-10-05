"""Data connectors (Kaggle, Hugging Face, databases, cloud storage) and their draft routes. No network: every
remote API is mocked; databases are SQLite files."""
import io
import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
import urllib.error
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dclab_rnd.connectors as base  # noqa: E402
from dclab_rnd.connectors import BadInput, ConnectorError, cloud, db, hf, kaggle  # noqa: E402

KEY = "s3cr3t-kaggle-key-123"
CSV = b"customer_id,tenure,churn\n" + b"".join(f"c{i},{i % 40},{'yes' if i % 3 == 0 else 'no'}\n".encode() for i in range(60))


def tmpdir() -> Path:
    path = Path(tempfile.mkdtemp())
    return path


# ---------------------------------------------------------------------------- Kaggle
class FakeResponse:
    def __init__(self, body: bytes, headers: dict | None = None):
        self._body, self.headers = io.BytesIO(body), headers or {}

    def read(self, n: int = -1) -> bytes:
        return self._body.read(n)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def zipped(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            if name.endswith("/"):
                archive.writestr(zipfile.ZipInfo(name), b"")
            else:
                archive.writestr(name, data)
    return buffer.getvalue()


class KaggleTests(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"KAGGLE_USERNAME": "alice", "KAGGLE_KEY": KEY, "KAGGLE_CONFIG_DIR": str(tmpdir())})
        env.start()
        self.addCleanup(env.stop)
        self.requests = []

    def serve(self, routes):
        """urlopen that answers by URL path; a value can be bytes, (bytes, headers) or an exception."""
        def urlopen(request, timeout=None):
            self.requests.append(request)
            path = request.full_url.replace(kaggle.API, "")
            for prefix, answer in routes.items():
                if path.startswith(prefix):
                    if isinstance(answer, Exception):
                        raise answer
                    body, headers = answer if isinstance(answer, tuple) else (answer, {})
                    return FakeResponse(body, headers)
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, io.BytesIO())
        return mock.patch("urllib.request.urlopen", side_effect=urlopen)

    def test_search_maps_fields_and_sends_basic_auth_unredirected(self):
        listing = [{"ref": "blastchar/telco-customer-churn", "title": "Telco Customer Churn", "totalBytes": 977501,
                    "licenseName": "Data files © Original Authors", "lastUpdated": "2018-02-23T16:38:51Z",
                    "url": "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"},
                   {"ownerRef": "someone", "datasetSlug": "bank", "title": "Bank", "total_bytes": "12", "license_name": "CC0"},
                   {"ref": "../bad", "title": "skipped"}]
        with self.serve({"/datasets/list?": json.dumps(listing).encode()}):
            found = kaggle.search("telco churn", page=2)
        self.assertEqual([d["ref"] for d in found], ["blastchar/telco-customer-churn", "someone/bank"])
        self.assertEqual(found[0]["size_bytes"], 977501)
        self.assertEqual((found[1]["size_bytes"], found[1]["license"]), (12, "CC0"))
        self.assertEqual(found[1]["url"], "https://www.kaggle.com/datasets/someone/bank")
        request = self.requests[0]
        self.assertIn("search=telco+churn", request.full_url)
        self.assertIn("page=2", request.full_url)
        self.assertIn("Authorization", request.unredirected_hdrs)  # never forwarded to the storage redirect
        self.assertNotIn("Authorization", request.headers)
        with self.assertRaises(BadInput):
            kaggle.search("  ")

    def test_files_tolerates_key_casing(self):
        with self.serve({"/datasets/list/o/d": json.dumps({"datasetFiles": [{"name": "a.csv", "totalBytes": 10},
                                                                            {"Name": "b.parquet", "total_bytes": 20}]}).encode()}):
            self.assertEqual(kaggle.files("o/d"), [{"name": "a.csv", "size_bytes": 10}, {"name": "b.parquet", "size_bytes": 20}])
        with self.assertRaises(BadInput):
            kaggle.files("o/../d")

    def test_download_zip_keeps_largest_table_and_ignores_traversal(self):
        archive = zipped({"../evil.csv": CSV * 50, "/abs.csv": CSV * 40, "folder/": b"", "small.csv": b"a,b\n1,2\n",
                          "nested/data.csv": CSV, "__MACOSX/._data.csv": CSV * 30, "readme.md": b"# hi" * 1000})
        meta_json = json.dumps({"licenseName": "CC0: Public Domain", "lastUpdated": "2024-01-02", "currentVersionNumber": 3}).encode()
        root = tmpdir()
        directory = root / "draft" / "data"
        with self.serve({"/datasets/view/o/telco": meta_json, "/datasets/download/o/telco": archive}):
            out = kaggle.download("o/telco", directory)
        self.assertEqual(out["filename"], "telco_data.csv")
        self.assertEqual(Path(out["path"]).read_bytes(), CSV)
        self.assertEqual(sorted(p.name for p in directory.iterdir()), ["telco_data.csv"])  # no scratch file left
        self.assertFalse(any(p.name.endswith("evil.csv") for p in root.rglob("*")))
        src = out["source"]
        self.assertEqual((src["kind"], src["ref"], src["file"], src["license"], src["version"]), ("kaggle", "o/telco", "nested/data.csv", "CC0: Public Domain", 3))
        self.assertEqual(len(src["sha256"]), 64)
        self.assertNotIn(KEY, json.dumps(out))

    def test_download_one_file_raw_or_zipped(self):
        directory = tmpdir()
        with self.serve({"/datasets/download/o/d/train.csv": (CSV, {"Content-Disposition": 'attachment; filename="train.csv"'})}):
            out = kaggle.download("o/d", directory, file="train.csv")
        self.assertEqual((out["filename"], out["source"]["file"]), ("d_train.csv", "train.csv"))
        self.assertIsNone(out["source"]["license"])  # /datasets/view answered 404: the licence is unknown, not an error
        with self.serve({"/datasets/download/o/d/train.csv": zipped({"train.csv": CSV})}):
            again = kaggle.download("o/d", directory, file="train.csv")
        self.assertEqual(again["filename"], "d_train-1.csv")  # never overwrites the earlier import
        with self.serve({"/datasets/download/o/d/x.csv": zipped({"other.csv": CSV, "more.csv": CSV})}):
            with self.assertRaises(ConnectorError):
                kaggle.download("o/d", directory, file="x.csv")

    def test_size_cap_while_streaming(self):
        with mock.patch.object(base, "MAX_BYTES", 100), self.serve({"/datasets/download/o/d": zipped({"big.csv": CSV})}):
            with self.assertRaises(ConnectorError) as caught:
                kaggle.download("o/d", tmpdir())
        self.assertIn("limit", str(caught.exception))

    def test_refused_credentials_never_leak_the_key(self):
        error = urllib.error.HTTPError("https://www.kaggle.com/api/v1/datasets/list", 401, "Unauthorized", {}, io.BytesIO())
        with self.serve({"/datasets/list": error}):
            with self.assertRaises(ConnectorError) as caught:
                kaggle.search("churn")
        self.assertIn("refused the credentials", str(caught.exception))
        self.assertNotIn(KEY, str(caught.exception))
        with self.serve({"/datasets/list": urllib.error.URLError(OSError("no route"))}):
            with self.assertRaises(ConnectorError) as caught:
                kaggle.search("churn")
        self.assertIn("Could not reach Kaggle", str(caught.exception))

    def test_missing_credentials(self):
        with mock.patch.dict(os.environ, {"KAGGLE_USERNAME": "", "KAGGLE_KEY": ""}):
            self.assertIsNone(kaggle.credentials_source())
            self.assertFalse(base.status()["kaggle"]["configured"])
            with self.assertRaises(ConnectorError) as caught:
                kaggle.search("churn")
            self.assertIn("KAGGLE_USERNAME", str(caught.exception))
            Path(os.environ["KAGGLE_CONFIG_DIR"], "kaggle.json").write_text(json.dumps({"username": "bob", "key": KEY}))
            self.assertEqual(kaggle.credentials_source(), "kaggle.json")
            self.assertNotIn(KEY, json.dumps(base.status()))


# ---------------------------------------------------------------------------- Hugging Face
def http_error(cls, status=404):
    import httpx

    response = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/api/datasets/x"))
    return cls("boom", response=response)


class FakeHfApi:
    def __init__(self, branches: dict, errors: dict | None = None):
        self.branches, self.errors, self.calls = branches, errors or {}, []

    def __call__(self, *a, **k):
        return self

    def list_repo_files(self, repo_id, repo_type=None, revision=None, token=None):
        self.calls.append(("list", revision))
        if revision in self.errors:
            raise self.errors[revision]
        if revision not in self.branches:
            from huggingface_hub.errors import RevisionNotFoundError
            raise http_error(RevisionNotFoundError)
        return list(self.branches[revision])

    SHAS = {"main": "abc123", hf.CONVERTED: "conv456"}

    def files_at(self, revision):
        by_sha = {sha: branch for branch, sha in self.SHAS.items()}
        return self.branches[by_sha.get(revision, revision)]

    def get_paths_info(self, repo_id, paths, repo_type=None, revision=None, token=None):
        return [SimpleNamespace(path=p, size=len(self.files_at(revision)[p])) for p in paths]

    def dataset_info(self, repo_id, revision=None, token=None):
        return SimpleNamespace(sha=self.SHAS[revision], card_data={"license": "mit"}, tags=[])


class HfTests(unittest.TestCase):
    def run_fetch(self, branches, **kwargs):
        api = FakeHfApi(branches, kwargs.pop("errors", None))
        downloads = []

        def download(repo_id, filename, repo_type=None, revision=None, token=None, local_dir=None):
            downloads.append((filename, revision))
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(api.files_at(revision)[filename])
            return str(target)
        directory = tmpdir()
        with mock.patch("huggingface_hub.HfApi", api), mock.patch("huggingface_hub.hf_hub_download", side_effect=download):
            out = hf.fetch(kwargs.pop("dataset", "acme/churn"), directory, **kwargs)
        self.assertEqual(sorted(p.name for p in directory.iterdir()), [out["filename"]])  # scratch folder removed
        return out, downloads

    def test_split_match_and_parquet_preferred(self):
        files = {"README.md": b"x", "data/train.csv": CSV, "data/train-00000-of-00001.parquet": b"PAR1" + CSV,
                 "data/test-00000-of-00001.parquet": b"PAR1", "data/constrained.csv": CSV}
        out, downloads = self.run_fetch({"main": files})
        self.assertEqual(downloads, [("data/train-00000-of-00001.parquet", "abc123")])  # pinned to the recorded commit
        src = out["source"]
        self.assertEqual((src["kind"], src["dataset"], src["sha"], src["license"], src["split"]), ("hf", "acme/churn", "abc123", "mit", "train"))
        self.assertEqual(src["url"], "https://huggingface.co/datasets/acme/churn")
        self.assertEqual(out["filename"], "churn_train-00000-of-00001.parquet")
        out, downloads = self.run_fetch({"main": files}, split="test")
        self.assertEqual(downloads[0][0], "data/test-00000-of-00001.parquet")
        self.assertEqual(hf.pick_file(["a/validation.jsonl", "a/validation.json", "a/train.csv"], "validation", None), ("a/validation.jsonl", 1))
        self.assertEqual(hf.pick_file(["full/train-0.csv", "lite/train-0.csv"], "train", "lite"), ("lite/train-0.csv", 1))

    def test_fallback_to_parquet_conversion_takes_first_shard(self):
        branches = {"main": {"README.md": b"x", "churn.py": b"script"},
                    hf.CONVERTED: {"default/train/0000.parquet": b"PAR1a", "default/train/0001.parquet": b"PAR1b", "default/test/0000.parquet": b"PAR1"}}
        out, downloads = self.run_fetch(branches)
        self.assertEqual(downloads, [("default/train/0000.parquet", "conv456")])
        src = out["source"]
        self.assertTrue(src["converted"])
        self.assertEqual((src["shards"], src["config"], src["revision"], src["sha"]), (2, "default", hf.CONVERTED, "conv456"))
        self.assertIn("first of 2", src["note"])
        self.assertEqual(out["filename"], "churn_default_train.parquet")
        self.assertEqual(hf.pick_converted(["en/train/0000.parquet", "fr/train/0000.parquet"], "train", None), ("en/train/0000.parquet", 1, "en"))
        self.assertEqual(hf.pick_converted(["en/partial-train/0000.parquet"], "train", "en")[0], "en/partial-train/0000.parquet")

    def test_no_split_anywhere_is_a_plain_error(self):
        with self.assertRaises(ConnectorError) as caught:
            self.run_fetch({"main": {"README.md": b"x", "data/test.csv": CSV}}, split="validation")
        self.assertIn("No validation split", str(caught.exception))

    def test_errors_are_mapped(self):
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

        cases = [(GatedRepoError, 403, "accept the dataset's terms"), (RepositoryNotFoundError, 404, "has no dataset"),
                 (HfHubHTTPError, 401, "refused the token")]
        for cls, status, text in cases:
            with self.subTest(cls=cls.__name__), self.assertRaises(ConnectorError) as caught:
                self.run_fetch({"main": {}}, errors={"main": http_error(cls, status)})
            self.assertIn(text, str(caught.exception))
        with mock.patch.object(base, "MAX_BYTES", 10), self.assertRaises(ConnectorError) as caught:
            self.run_fetch({"main": {"train.csv": CSV}})
        self.assertIn("limit", str(caught.exception))
        for bad in ("../x", "a/b/c", "a b"):
            with self.subTest(bad=bad), self.assertRaises(BadInput):
                hf.fetch(bad, tmpdir())
        with self.assertRaises(BadInput):
            hf.fetch("acme/churn", tmpdir(), split="../train")


# ---------------------------------------------------------------------------- databases
def sqlite_db(rows: int = 50) -> tuple[str, Path]:
    path = tmpdir() / "warehouse.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE customers (id INTEGER, tenure INTEGER, plan TEXT, churn TEXT)")
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)",
                         [(i, i % 24, "pro" if i % 2 else "basic", "yes" if i % 4 == 0 else "no") for i in range(rows)])
    return f"sqlite:///{path}", path


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.url, self.path = sqlite_db(50)
        env = mock.patch.dict(os.environ, {"DCLAB_DB_WAREHOUSE": self.url, "DCLAB_DB_EMPTY": " "})
        env.start()
        self.addCleanup(env.stop)

    def test_connections_and_status_hide_urls(self):
        self.assertIn("warehouse", db.connections())
        self.assertNotIn("empty", db.connections())
        status = json.dumps(base.status())
        self.assertIn("warehouse", status)
        self.assertNotIn("sqlite:///", status)

    def test_table_and_query(self):
        import pandas as pd

        out = db.fetch("WAREHOUSE", tmpdir(), table="customers")
        frame = pd.read_parquet(out["path"])
        self.assertEqual(frame.shape, (50, 4))
        src = out["source"]
        self.assertEqual((src["kind"], src["connection"], src["table"], src["rows"], src["dialect"]), ("database", "warehouse", "customers", 50, "sqlite"))
        self.assertEqual(out["filename"], "warehouse_customers.parquet")
        q = db.fetch("warehouse", tmpdir(), query="WITH pro AS (SELECT * FROM customers WHERE plan = 'pro') -- the paying ones\n"
                                                    "SELECT id, churn FROM pro;  -- done", limit=7)
        self.assertEqual((q["source"]["rows"], q["source"]["limit"], q["source"]["truncated"]), (7, 7, True))
        self.assertEqual(list(pd.read_parquet(q["path"]).columns), ["id", "churn"])
        literal = db.fetch("warehouse", tmpdir(), query="SELECT 'a :b; DROP' AS s, id FROM customers", limit=1)
        self.assertEqual(pd.read_parquet(literal["path"])["s"][0], "a :b; DROP")
        self.assertEqual(db.fetch("warehouse", tmpdir(), table="customers", limit=10**9)["source"]["limit"], db.MAX_LIMIT)

    def test_rejected_reads(self):
        bad = ["UPDATE customers SET churn = 'no'", "SELECT 1; DROP TABLE customers", "SELECT * FROM customers; SELECT 1",
               "DELETE FROM customers", "WITH x AS (DELETE FROM customers RETURNING *) SELECT * FROM x",
               "SELECT * INTO copy FROM customers", "SELECT '\\'', 1; DROP TABLE t; SELECT '\\''", "SELECT $$; DROP TABLE t; $$", "/* only a comment */"]
        for query in bad:
            with self.subTest(query=query), self.assertRaises(BadInput):
                db.fetch("warehouse", tmpdir(), query=query)
        for table in ("customers; DROP TABLE customers", "a.b.c", "1abc", "customers --"):
            with self.subTest(table=table), self.assertRaises(BadInput):
                db.fetch("warehouse", tmpdir(), table=table)
        with self.assertRaises(BadInput):
            db.fetch("warehouse", tmpdir(), table="customers", query="SELECT 1")
        with self.assertRaises(BadInput):
            db.fetch("warehouse", tmpdir())
        with sqlite3.connect(self.path) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0], 50)  # nothing was changed

    def test_errors_never_show_the_url(self):
        with mock.patch.dict(os.environ, {"DCLAB_DB_PG": "postgresql+psycopg2://reader:hunter2@db.internal/prod"}):
            for call, text in [(lambda: db.fetch("missing", tmpdir(), table="t"), "DCLAB_DB_MISSING"),
                               (lambda: db.fetch("warehouse", tmpdir(), table="nope"), "OperationalError"),
                               (lambda: db.fetch("warehouse", tmpdir(), query="SELECT * FROM customers WHERE id < 0"), "no rows")]:
                with self.assertRaises(ConnectorError) as caught:
                    call()
                self.assertIn(text, str(caught.exception))
                self.assertNotIn("sqlite:///", str(caught.exception))
            try:
                import psycopg2  # noqa: F401
            except ImportError:
                with self.assertRaises(ConnectorError) as caught:
                    db.fetch("pg", tmpdir(), table="t")
                self.assertIn("psycopg2-binary", str(caught.exception))
                self.assertNotIn("hunter2", str(caught.exception))
                self.assertNotIn("db.internal", str(caught.exception))


# ---------------------------------------------------------------------------- cloud
class FakeS3:
    def __init__(self, objects: dict[str, bytes], error=None):
        self.objects, self.error = objects, error

    def head_object(self, Bucket, Key):
        if self.error:
            raise self.error
        if Key not in self.objects:
            raise client_error(404, "404")
        return {"ContentLength": len(self.objects[Key]), "ETag": '"etag-1"'}

    def get_object(self, Bucket, Key, **kwargs):
        return {"Body": io.BytesIO(self.objects[Key])}


def client_error(status: int, code: str):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code, "Message": "x"}, "ResponseMetadata": {"HTTPStatusCode": status}}, "HeadObject")


class CloudTests(unittest.TestCase):
    def test_s3_download(self):
        directory = tmpdir()
        with mock.patch("boto3.client", return_value=FakeS3({"exports/churn.csv": CSV})):
            out = cloud.fetch("s3://acme-data/exports/churn.csv", directory)
        self.assertEqual(Path(out["path"]).read_bytes(), CSV)
        self.assertEqual(out["filename"], "acme-data_churn.csv")
        self.assertEqual(out["source"], {"kind": "cloud", "uri": "s3://acme-data/exports/churn.csv", "etag": "etag-1",
                                         "size_bytes": len(CSV), "sha256": out["source"]["sha256"]})
        self.assertEqual([p.name for p in directory.iterdir()], ["acme-data_churn.csv"])

    def test_s3_size_cap_and_errors(self):
        with mock.patch.object(base, "MAX_BYTES", 10), mock.patch("boto3.client", return_value=FakeS3({"a.csv": CSV})):
            with self.assertRaises(ConnectorError) as caught:
                cloud.fetch("s3://acme-data/a.csv", tmpdir())
        self.assertIn("limit", str(caught.exception))
        for error, text in [(client_error(403, "403"), "Access denied"), (client_error(404, "NoSuchKey"), "was not found")]:
            with self.subTest(text=text), mock.patch("boto3.client", return_value=FakeS3({}, error)):
                with self.assertRaises(ConnectorError) as caught:
                    cloud.fetch("s3://acme-data/a.csv", tmpdir())
                self.assertIn(text, str(caught.exception))

    def test_s3_without_credentials_tries_anonymous_once(self):
        from botocore.exceptions import NoCredentialsError

        clients = [FakeS3({}, NoCredentialsError()), FakeS3({}, client_error(403, "AccessDenied"))]
        with mock.patch("boto3.client", side_effect=clients):
            with self.assertRaises(ConnectorError) as caught:
                cloud.fetch("s3://acme-data/a.csv", tmpdir())
        self.assertIn("no AWS credentials", str(caught.exception))
        with mock.patch("boto3.client", side_effect=[FakeS3({}, NoCredentialsError()), FakeS3({"a.csv": CSV})]):
            self.assertTrue(cloud.fetch("s3://acme-data/a.csv", tmpdir())["source"]["anonymous"])

    def test_bad_uris(self):
        for uri in ("http://x/a.csv", "s3://acme-data", "s3://acme-data/", "s3://Bad_Bucket!/a.csv", "s3://acme-data/../a.csv",
                    "s3://acme-data/a/./b.csv", "s3://acme-data/model.pkl", "gs://acme-data/dir/", "s3://ab/a.csv"):
            with self.subTest(uri=uri), self.assertRaises(BadInput):
                cloud.fetch(uri, tmpdir())

    def test_gcs_needs_a_client_library(self):
        with mock.patch.dict(sys.modules, {"google.cloud.storage": None, "gcsfs": None}):
            with self.assertRaises(ConnectorError) as caught:
                cloud.fetch("gs://acme-data/a.csv", tmpdir())
        self.assertIn("google-cloud-storage", str(caught.exception))

    def test_gcs_with_google_cloud_storage(self):
        try:
            import google.api_core.exceptions  # noqa: F401
            import google.auth.exceptions  # noqa: F401
        except ImportError:
            self.skipTest("google-cloud-storage is not installed")
        blob = mock.MagicMock(size=len(CSV), etag="g-1")
        blob.open.return_value = io.BytesIO(CSV)
        storage = mock.MagicMock()
        storage.Client.return_value.bucket.return_value.blob.return_value = blob
        out = cloud._gcs_client(storage, "acme-data", "a.csv", "gs://acme-data/a.csv", tmpdir() / "part")
        self.assertEqual(out, {"etag": "g-1", "anonymous": False})


# ---------------------------------------------------------------------------- routes
def wait(fn, timeout=30.0, every=0.1):
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        time.sleep(every)
    raise AssertionError("timed out waiting")


class RouteTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        self.url, _ = sqlite_db(50)
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1", "DCLAB_DB_TEST": self.url})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(create_app(tmpdir()))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def test_database_import_becomes_a_ready_asset(self):
        c = self.client
        status = c.get("/api/connectors")
        self.assertEqual(status.status_code, 200)
        self.assertIn("test", status.json()["database"]["connections"])
        self.assertNotIn("sqlite:///", status.text)
        d = c.post("/api/drafts", json={"problem": "Predict which customers churn next month"}, headers=self.h).json()
        did = d["id"]
        self.assertEqual(c.post(f"/api/drafts/{did}/data/database", json={"connection": "test", "table": "customers"}).status_code, 403)  # CSRF
        self.assertEqual(c.post(f"/api/drafts/{did}/data/database", json={"connection": "test"}, headers=self.h).status_code, 422)
        self.assertEqual(c.post(f"/api/drafts/{did}/data/database", json={"connection": "test", "query": "DROP TABLE customers"}, headers=self.h).status_code, 422)
        self.assertEqual(c.post(f"/api/drafts/{did}/data/database", json={"connection": "nope", "table": "customers"}, headers=self.h).status_code, 400)
        self.assertEqual(c.post("/api/drafts/abc123/data/database", json={"connection": "test", "table": "customers"}, headers=self.h).status_code, 404)
        r = c.post(f"/api/drafts/{did}/data/database", json={"connection": "test", "table": "customers"}, headers=self.h)
        self.assertEqual(r.status_code, 200, r.text)
        asset = r.json()
        self.assertEqual((asset["kind"], asset["source"]["rows"]), ("database", 50))

        def ready():
            found = next((a for a in c.get(f"/api/drafts/{did}").json()["assets"] if a["id"] == asset["id"]), None)
            return found if found and found["status"] in ("ready", "failed") else None
        final = wait(ready)
        self.assertEqual(final["status"], "ready", final.get("error"))
        self.assertEqual(final["rows"], 50)
        self.assertNotIn("sqlite:///", json.dumps(final["source"]))
        self.assertNotIn("url", final["source"])

    def test_other_routes_validate_before_any_network(self):
        c = self.client
        did = c.post("/api/drafts", json={"problem": "Predict which customers churn next month"}, headers=self.h).json()["id"]
        self.assertEqual(c.post(f"/api/drafts/{did}/data/kaggle", json={"ref": "../etc"}, headers=self.h).status_code, 422)
        self.assertEqual(c.post(f"/api/drafts/{did}/data/hf", json={"dataset": "a/b/c"}, headers=self.h).status_code, 422)
        self.assertEqual(c.post(f"/api/drafts/{did}/data/cloud", json={"uri": "s3://acme-data/../x.csv"}, headers=self.h).status_code, 422)
        self.assertEqual(c.post("/api/connectors/kaggle/search", json={}, headers=self.h).status_code, 422)
        with mock.patch.dict(os.environ, {"KAGGLE_USERNAME": "", "KAGGLE_KEY": "", "KAGGLE_CONFIG_DIR": str(tmpdir())}):
            r = c.post("/api/connectors/kaggle/search", json={"query": "churn"}, headers=self.h)
        self.assertEqual(r.status_code, 400)
        self.assertIn("KAGGLE_USERNAME", r.json()["detail"])
        with mock.patch("boto3.client", return_value=FakeS3({"churn.csv": CSV})):
            r = c.post(f"/api/drafts/{did}/data/cloud", json={"uri": "s3://acme-data/churn.csv"}, headers=self.h)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual((r.json()["kind"], r.json()["name"]), ("cloud", "s3://acme-data/churn.csv"))


if __name__ == "__main__":
    unittest.main()
