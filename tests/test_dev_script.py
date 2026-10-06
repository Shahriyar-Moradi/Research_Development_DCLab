"""Local development with one command (package 12.0): the database is created and migrated, a reset needs consent and
only touches dclab_* databases, and the frontend is rebuilt when its sources change."""
import os
import secrets
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import dev  # noqa: E402
import pgtest  # noqa: E402


class SafetyTests(unittest.TestCase):
    def test_only_dclab_databases_are_touched(self):
        for name in ("postgres", "production", "app"):
            with self.assertRaises(dev.DevError):
                dev.check_name(f"postgresql+psycopg://u@/{name}")
        self.assertEqual(dev.check_name("postgresql+psycopg://u@/dclab_dev"), "dclab_dev")

    def test_a_reset_needs_consent(self):
        with mock.patch.dict(os.environ, {}):  # main() reads .env into the environment; restore it afterwards
            self.assertEqual(dev.main(["db-reset"]), 2)  # without --yes nothing is dropped

    def test_another_name_or_another_host_is_refused_before_any_connection(self):
        with mock.patch.object(dev, "server_reachable", side_effect=AssertionError("connected")):
            for url in ("postgresql+psycopg://u@/myapp", 'postgresql+psycopg://u@/dclab_x"; select 1; --',
                        "postgresql+psycopg://u:pw@staging.example.com/dclab_dev"):
                with self.assertRaises(dev.DevError, msg=url):
                    dev.ensure_database(url, reset=True)
            with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": "postgresql+psycopg://u:pw@staging.example.com/dclab_dev"}):
                self.assertEqual(dev.main(["db-reset", "--yes"]), 2)
        self.assertEqual(dev.check_name("postgresql+psycopg://u@localhost/dclab_dev"), "dclab_dev")
        self.assertEqual(dev.check_name("postgresql+psycopg://u@db.example.com/dclab_dev", remote=True), "dclab_dev")  # only when asked

    def test_every_host_source_the_driver_reads_is_checked(self):
        for url, env in (("postgresql+psycopg://u@localhost/dclab_dev?host=staging.example.com", {}),  # the query wins in libpq
                         ("postgresql+psycopg://u@/dclab_dev?hostaddr=10.0.0.5", {}),
                         ("postgresql+psycopg://u@/dclab_dev?service=prod", {}),
                         ("postgresql+psycopg://u@/dclab_dev", {"PGHOST": "staging.example.com"}),
                         ("postgresql+psycopg://u@/dclab_dev", {"PGHOSTADDR": "10.0.0.5"}),
                         ("postgresql+psycopg://u@/dclab_dev", {"PGSERVICE": "prod"})):
            with mock.patch.dict(os.environ, env), self.assertRaises(dev.DevError, msg=(url, env)):
                dev.check_name(url)
        with mock.patch.dict(os.environ, {"PGHOST": "/tmp"}):
            self.assertEqual(dev.check_name("postgresql+psycopg://u@/dclab_dev"), "dclab_dev")  # a socket folder is this machine

    def test_the_server_reached_must_be_this_machine(self):
        class Connection:
            def __init__(self, address):
                self.address = address

            def execute(self, _):
                return mock.Mock(scalar=lambda: self.address)
        for address in (None, "127.0.0.1", "::1"):
            dev.check_server(Connection(address), "dclab_dev")
        with self.assertRaises(dev.DevError):
            dev.check_server(Connection("10.0.0.5"), "dclab_dev")

    def test_an_unreachable_server_says_how_to_start_one_and_creates_nothing(self):
        with self.assertRaises(dev.DevError) as caught:
            dev.ensure_database("postgresql+psycopg://u@127.0.0.1:1/dclab_unreachable")
        self.assertIn("brew services start postgresql", str(caught.exception))

    def test_serve_runs_one_reloading_worker_on_the_checked_database(self):
        with mock.patch.object(dev, "ensure_database", return_value="already at 0008") as ensured, \
             mock.patch.object(dev, "build_frontend"), mock.patch.object(dev, "watch_frontend"), \
             mock.patch.object(dev, "watch_worker"), mock.patch.object(dev.subprocess, "Popen") as popen:
            popen.return_value.wait.return_value = 0
            self.assertEqual(dev.serve("postgresql+psycopg://u@/dclab_dev", 8799), 0)
        calls = {" ".join(c.args[0]): c for c in popen.call_args_list}
        api = next(c for k, c in calls.items() if "uvicorn" in k)
        worker = next(c for k, c in calls.items() if "dclab_rnd.worker" in k)  # the job worker runs on its own (package 10.3)
        command, env = api.args[0], api.kwargs["env"]
        self.assertEqual(env["DCLAB_WORKER"], "external")
        self.assertEqual(worker.kwargs["env"]["DCLAB_DATABASE_URL"], "postgresql+psycopg://u@/dclab_dev")
        self.assertEqual(command[command.index("--port") + 1], "8799")
        self.assertIn("--reload", command)
        self.assertEqual(command[command.index("--workers") + 1], "1")
        self.assertEqual(env["DCLAB_DATABASE_URL"], "postgresql+psycopg://u@/dclab_dev")
        ensured.assert_called_once()

    def test_the_worker_runs_without_reload_when_watchfiles_is_missing(self):
        import importlib.util

        with mock.patch.object(importlib.util, "find_spec", return_value=None):
            self.assertEqual(dev.worker_command(), [sys.executable, "-m", "dclab_rnd.worker"])
        process = mock.Mock(wait=mock.Mock(return_value=1))
        with mock.patch("sys.stderr") as err:
            dev.watch_worker(process, threading.Event())  # the API still runs: say so loudly
        self.assertIn("THE JOB WORKER STOPPED", "".join(c.args[0] for c in err.write.call_args_list))

    def test_the_frontend_is_rebuilt_when_a_source_changes(self):
        folder = Path(tempfile.mkdtemp())
        (folder / "a.js").write_text("1")
        builds, stop = [], threading.Event()
        watcher = threading.Thread(target=dev.watch_frontend, args=(lambda: builds.append(1), stop, 0.05, folder), daemon=True)
        watcher.start()
        time.sleep(0.2)
        self.assertEqual(builds, [])  # nothing changed yet
        later = time.time() + 5
        (folder / "a.js").write_text("2")
        os.utime(folder / "a.js", (later, later))
        deadline = time.time() + 10
        while time.time() < deadline and not builds:
            time.sleep(0.05)
        stop.set()
        self.assertEqual(builds, [1])

    def test_the_watcher_survives_a_failing_build_and_a_vanishing_file(self):
        folder = Path(tempfile.mkdtemp())
        (folder / "a.js").write_text("1")
        calls, stop = [], threading.Event()

        def build():
            calls.append(1)
            raise RuntimeError("a broken source file")
        real = dev.changed_since
        flaky = iter([FileNotFoundError("an editor's temporary file")])

        def scan(f, last):
            for error in flaky:
                raise error
            return real(f, last)
        with mock.patch.object(dev, "changed_since", side_effect=scan):
            watcher = threading.Thread(target=dev.watch_frontend, args=(build, stop, 0.05, folder), daemon=True)
            watcher.start()
            for n in (1, 2):
                later = time.time() + 5 * n
                (folder / "a.js").write_text(str(n))
                os.utime(folder / "a.js", (later, later))
                deadline = time.time() + 10
                while time.time() < deadline and len(calls) < n:
                    time.sleep(0.05)
            self.assertTrue(watcher.is_alive())
            stop.set()
        self.assertEqual(len(calls), 2)  # it rebuilt after a failure, and after a file vanished mid-scan


class DatabaseTests(unittest.TestCase):
    def test_a_database_is_created_migrated_and_reset(self):
        from sqlalchemy.engine import make_url

        from dclab_rnd.storage import db

        base = pgtest.require()
        url = make_url(base).set(database=f"dclab_devtest_{secrets.token_hex(3)}").render_as_string(hide_password=False)
        try:
            self.assertIn("created and migrated", dev.ensure_database(url))
            head = db.current(url)
            self.assertIsNotNone(head)
            self.assertIn("already at", dev.ensure_database(url))  # a second run changes nothing
            self.assertIn("reset and", dev.ensure_database(url, reset=True))
            self.assertEqual(db.current(url), head)
        finally:
            import sqlalchemy as sa
            db.engine(url).dispose()
            admin = sa.create_engine(make_url(url).set(database="postgres"), isolation_level="AUTOCOMMIT")
            with admin.connect() as c:
                c.execute(sa.text(f'drop database if exists "{make_url(url).database}" with (force)'))
            admin.dispose()


if __name__ == "__main__":
    unittest.main()
