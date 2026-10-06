"""Configuration and secrets (package 12.2): every setting the code reads is documented in .env.example; a secret
comes from the environment or a file (Docker secrets); the server refuses to start, saying why, when a required
setting is missing or the database cannot be reached; the host check takes the names a proxy uses; no secret is
committed."""
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

NAMES = re.compile(r"\b(DCLAB_[A-Z0-9_]+|OPENAI_[A-Z0-9_]+|HF_[A-Z0-9_]+|KAGGLE_[A-Z0-9_]+|AWS_[A-Z0-9_]+)\b")


class ConfigTests(unittest.TestCase):
    def test_every_setting_the_code_reads_is_documented(self):
        import subprocess

        documented = set(NAMES.findall((ROOT / ".env.example").read_text(encoding="utf-8")))
        tracked = subprocess.run(["git", "ls-files", "dclab_rnd", "general_pipeline", "scripts", "docker-compose.yml", "Dockerfile"],
                                 cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        read = set()
        for name in tracked:
            path = ROOT / name
            if (path.suffix in (".py", ".yml") or path.name == "Dockerfile") and path.is_file():
                read |= set(NAMES.findall(path.read_text(encoding="utf-8", errors="ignore")))
        missing = sorted(n for n in read if n not in documented and not n.endswith("_"))  # a name ending in _ is a prefix (DCLAB_TIER_{tier}_...)
        self.assertEqual(missing, [], "document these in .env.example with their safe default")

    def test_a_secret_can_come_from_a_file(self):
        from dclab_rnd.settings import load_secret_files

        secret = Path(tempfile.mkdtemp()) / "openai_api_key"
        secret.write_text("sk-from-a-docker-secret\n")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY_FILE": str(secret), "HOME_FILE": str(secret)}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            self.assertEqual(load_secret_files(), ["OPENAI_API_KEY"])
            self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-from-a-docker-secret")
            self.assertNotIn("HOME", load_secret_files())  # only the product's own settings
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "set directly", "OPENAI_API_KEY_FILE": str(secret)}):
            load_secret_files()
            self.assertEqual(os.environ["OPENAI_API_KEY"], "set directly")  # a value set wins
        with mock.patch.dict(os.environ, {"DCLAB_SECRET_KEY_FILE": "/no/such/file"}), self.assertRaises(RuntimeError):
            load_secret_files()

    def test_the_server_refuses_to_start_and_says_why(self):
        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.settings import Settings

        cases = [({"DCLAB_AUTH": "password", "DCLAB_DATABASE_URL": ""}, "DCLAB_AUTH needs PostgreSQL"),
                 ({"DCLAB_DATABASE_URL": "postgresql+psycopg://nobody@127.0.0.1:1/dclab_nowhere"}, "does not answer"),
                 ({"DCLAB_AUTH": "oidc", "DCLAB_DATABASE_URL": "", "DCLAB_OIDC_ISSUER": ""}, "DCLAB_OIDC_ISSUER"),
                 ({"DCLAB_MODEL_RETRIES": "twice"}, "DCLAB_MODEL_RETRIES must be a number")]
        for env, reason in cases:
            with mock.patch.dict(os.environ, env):
                self.assertTrue(any(reason in p for p in Settings.load(tempfile.mkdtemp()).problems()), (env, reason))
                with self.assertRaises(RuntimeError) as refused:
                    create_app(Path(tempfile.mkdtemp()))
                self.assertIn("cannot start", str(refused.exception))
                self.assertNotIn("nobody@", str(refused.exception))  # a URL never appears: it can hold a password

    def test_the_host_check_takes_a_proxys_name_only_with_sign_in(self):
        from dclab_rnd.settings import Settings

        with mock.patch.dict(os.environ, {"DCLAB_ALLOWED_HOSTS": "https://DCLab.example.com:8443, dclab.internal, *.team.example, a*b"}):
            settings = Settings.load(tempfile.mkdtemp())
            self.assertEqual(settings.hosts()[3:], ["dclab.example.com", "dclab.internal", "*.team.example"])  # as the check compares them
            problems = " ".join(settings.problems())
        self.assertIn("without sign-in anyone who reaches it is the owner", problems)  # other names need DCLAB_AUTH
        self.assertIn("a wildcard is only allowed as *.domain", problems)

    def test_the_host_check_lets_a_proxys_name_through(self):
        import pgtest
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        url = pgtest.require()
        with mock.patch.dict(os.environ, {"DCLAB_ALLOWED_HOSTS": "dclab.example.com", "DCLAB_AUTH": "password", "DCLAB_DATABASE_URL": url}):
            from dclab_rnd.storage import db

            db.upgrade(url)
            client = TestClient(create_app(Path(tempfile.mkdtemp())))
            self.assertEqual(client.get("/api/config", headers={"host": "dclab.example.com"}).status_code, 200)
            self.assertEqual(client.get("/api/config", headers={"host": "evil.example.net"}).status_code, 400)

    def test_a_typo_in_the_sign_in_mode_does_not_leave_a_server_open(self):
        from dclab_rnd.settings import Settings

        with mock.patch.dict(os.environ, {"DCLAB_AUTH": "passwd"}):
            self.assertIn("DCLAB_AUTH is password, oidc or none", " ".join(Settings.load(tempfile.mkdtemp()).problems()))

    def test_a_secret_file_reaches_the_command_line_tools_and_others_are_left_alone(self):
        from dclab_rnd import secret_files
        from dclab_rnd.connectors import db as connectors

        folder = Path(tempfile.mkdtemp())
        (folder / "url").write_text("postgresql+psycopg://reader:s3cret-value@db.host/warehouse\n")
        with mock.patch.dict(os.environ, {"DCLAB_DB_WAREHOUSE_FILE": str(folder / "url"), "AWS_SHARED_CREDENTIALS_FILE": "/no/such/credentials",
                                          "DCLAB_PRICES_FILE": "/no/such/prices.json", "DCLAB_DB_POOL_SIZE": "10"}):
            os.environ.pop("DCLAB_DB_WAREHOUSE", None)
            self.assertEqual(secret_files.load(strict=True), ["DCLAB_DB_WAREHOUSE"])  # AWS's and the prices' own files are not secrets here
            self.assertEqual(connectors.connections(), ["warehouse"])  # not "pool_size", not "warehouse_file"

    def test_the_worker_refuses_to_start_and_says_why(self):
        import contextlib
        import io

        from dclab_rnd import worker

        err = io.StringIO()
        with mock.patch.dict(os.environ, {"DCLAB_AUTH": "password", "DCLAB_DATABASE_URL": ""}), contextlib.redirect_stderr(err):
            self.assertEqual(worker.main(["--home", tempfile.mkdtemp()]), 2)
        self.assertIn("cannot start", err.getvalue())

    def test_a_schema_behind_the_code_is_refused_with_what_to_run(self):
        import secrets

        import pgtest
        import sqlalchemy as sa
        from sqlalchemy.engine import make_url

        from dclab_rnd.settings import Settings
        from dclab_rnd.storage import db

        base = pgtest.require()
        name = f"dclab_schematest_{secrets.token_hex(3)}"
        admin = sa.create_engine(make_url(base).set(database="postgres"), isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            c.execute(sa.text(f'create database "{name}"'))
        url = make_url(base).set(database=name).render_as_string(hide_password=False)
        try:
            db.upgrade(url, "0013")
            with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": url}):
                problems = " ".join(Settings.load(tempfile.mkdtemp()).problems())
            self.assertIn("run python -m dclab_rnd.storage upgrade", problems)
        finally:
            db.engine(url).dispose()
            with admin.connect() as c:
                c.execute(sa.text(f'drop database if exists "{name}" with (force)'))
            admin.dispose()

    def test_no_secret_is_committed(self):
        import scan_secrets

        self.assertEqual(scan_secrets.scan(scan_secrets.tracked(), scan_secrets.ROOT), [])
        folder = Path(tempfile.mkdtemp())
        key = "sk-proj-" + "a1B2c3D4" * 5
        (folder / "settings.yml").write_text(f"api_key: {key}   # put your own here, see example.com\n")  # the comment does not hide it
        self.assertEqual([k for _, _, k, _ in scan_secrets.scan(scan_secrets.everything(folder), folder)], ["OpenAI-style key"])
        self.assertNotIn(key, " ".join(str(f) for f in scan_secrets.scan(scan_secrets.everything(folder), folder)))  # never the value

if __name__ == "__main__":
    unittest.main()
