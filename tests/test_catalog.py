"""Package 19.13: the archive and its catalog. The catalog lists every dataset with its lineage back to its sessions,
the Admin page reads it (the owner only), and a restore from a backup gives the same catalog."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_rnd import consent, datasets, experience, ledger
    from dclab_rnd.experience import catalog
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""


def workspace(home: Path):
    """A workspace with a consented project: decisions, a built Home conversation and an agent run, then its datasets."""
    from dclab_rnd.agentic.server import create_app

    client = TestClient(create_app(home))
    s = client.app.state.services
    h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
    pid = client.post("/api/projects", json={"name": "Consented", "goal": "g"}, headers=h).json()["id"]
    client.patch(f"/api/projects/{pid}", json={"settings": {"consent": {"train": True, "improve": True}}}, headers=h)
    consent.set_workspace_consent(Path(s.store.home), "train", True)
    rec = ledger.decide("column.is_identifier", ("column", "order_id"), answers=[ledger.answer("rule", "id_like", False)], value_used=False,
                        state={"column": "order_id"}, ledger=s.ledger, project=pid, conversation="d_conv")
    ledger.label(rec, True, "benchmark", "replica:delivery_fraud", ledger=s.ledger)
    for i, (kind, actor, payload) in enumerate([("draft.created", "person", {"problem": "Predict late orders"}),
                                                ("chat.agent", "agent", {"text": "When?", "question_field": "prediction_moment"}),
                                                ("chat.person", "person", {"text": "At the order"}), ("draft.built", "person", {"project": pid})]):
        experience.record(kind, actor, payload, log=s.experience, at=f"2026-10-13T09:0{i}:00+00:00", visit="v_home", conversation="d_conv", project=pid)
    for i in range(2):
        experience.record("agent.step", "agent", {"agent": "intern", "n": i, "tool": "describe_data"}, log=s.experience,
                          at=f"2026-10-13T10:0{i}:00+00:00", visit="v_intern", run="run_1", project=pid)
    datasets.build(s)
    datasets.development(Path(s.store.home) / datasets.FOLDER)
    return client, s, h


@unittest.skipIf(MISSING, MISSING)
class CatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = Path(tempfile.mkdtemp())
        cls.client, cls.s, cls.h = workspace(cls.home)

    def test_every_dataset_with_its_lineage_back_to_its_sessions(self):
        items = {i["path"]: i for i in catalog.catalog(self.s.store.home)}
        self.assertTrue({"decisions/column.is_identifier", "conversations/problem_understanding", "trajectories/agent_runs",
                         "development"} <= set(items), sorted(items))
        self.assertEqual(items["decisions/column.is_identifier"]["sources"], {"decision": 1, "label": 1})
        self.assertGreater(items["development"]["rows"], 100)
        visits = lambda path: {x["visit"] for x in catalog.lineage(self.s.store.home, path, self.s.experience, self.s.ledger)["sessions"]}  # noqa: E731
        self.assertEqual(visits("conversations/problem_understanding"), {"v_home"})
        self.assertEqual(visits("trajectories/agent_runs"), {"v_intern"})
        self.assertEqual(visits("decisions/column.is_identifier"), {"v_home"})  # through its conversation, not its whole project
        with self.assertRaises(KeyError):
            catalog.lineage(self.s.store.home, "../../etc", self.s.experience)

    def test_every_item_says_its_size_version_and_consent(self):
        items = catalog.catalog(self.s.store.home)
        order = [(i["built_at"], i["path"]) for i in items]
        self.assertEqual(order, sorted(order, reverse=True))  # newest first, then by path
        self.assertNotEqual(order, sorted(order))
        one = next(i for i in items if i["path"] == "decisions/column.is_identifier")
        self.assertGreater(one["bytes"], 0)
        self.assertEqual((one["version"], one["use"]), ("1", "train"))
        self.assertIn("train", one["consent"])

    def test_a_broken_manifest_is_listed_and_has_no_lineage(self):
        home = Path(tempfile.mkdtemp())
        broken = home / catalog.FOLDER / "decisions" / "column.broken"
        broken.mkdir(parents=True)
        (broken / "manifest.json").write_text("{half")
        self.assertEqual(catalog.catalog(home), [{"path": "decisions/column.broken", "error": "unreadable manifest"}])
        with self.assertRaises(KeyError):
            catalog.lineage(home, "decisions/column.broken", None)

    def test_the_command_line(self):
        import contextlib
        import io

        from dclab_rnd.experience import __main__ as cli

        with mock.patch.dict(os.environ, {"DCLAB_AGENT_HOME": str(self.s.store.home)}):
            listed = io.StringIO()
            with contextlib.redirect_stdout(listed):
                self.assertEqual(cli.main(["catalog"]), 0)
            self.assertIn("decisions/column.is_identifier", listed.getvalue())
            self.assertIn("consent: train", listed.getvalue())
            one = io.StringIO()
            with contextlib.redirect_stdout(one):
                self.assertEqual(cli.main(["catalog", "--lineage", "conversations/problem_understanding"]), 0)
            self.assertIn("v_home", one.getvalue())

    def test_the_admin_page_reads_it(self):
        r = self.client.get("/api/datasets").json()
        self.assertGreaterEqual(r["total"], 4)
        for item in r["items"]:  # every path the catalog lists opens
            self.assertEqual(self.client.get("/api/datasets/lineage", params={"path": item["path"]}).status_code, 200, item["path"])
        one = self.client.get("/api/datasets/lineage", params={"path": "trajectories/agent_runs"}).json()
        self.assertEqual([x["visit"] for x in one["sessions"]], ["v_intern"])
        self.assertEqual(self.client.get("/api/datasets/lineage", params={"path": "../x"}).status_code, 422)
        self.assertEqual(self.client.get("/api/datasets/lineage", params={"path": "decisions/none.such"}).status_code, 404)
        from dclab_rnd.accounts import guard

        self.assertEqual({guard.permission("GET", p) for p in ("/api/datasets", "/api/datasets/lineage")}, {"admin"})
        from dclab_rnd.accounts.principal import Principal

        with mock.patch.object(guard, "current", return_value=Principal(role="reviewer", user_id="u1", via="session")):
            self.assertEqual(self.client.get("/api/datasets").status_code, 403)  # through the app: only the owner
            self.assertEqual(self.client.get("/api/datasets/lineage", params={"path": "development"}).status_code, 403)


@unittest.skipIf(MISSING, MISSING)
class RestoreTest(unittest.TestCase):
    def test_a_restore_gives_the_same_catalog(self):
        from tests import pgtest
        from dclab_rnd.storage import backup, db

        url = pgtest.require()
        if backup.tool("pg_dump") is None or backup.tool("pg_restore") is None:
            self.skipTest("PostgreSQL's client tools (pg_dump, pg_restore) are not installed")
        pgtest.empty(url)
        import sqlalchemy as sa
        from sqlalchemy.engine import make_url

        target = make_url(url).set(database=f"{make_url(url).database}_catalog").render_as_string(hide_password=False)
        admin = sa.create_engine(url, isolation_level="AUTOCOMMIT")
        name = make_url(target).database
        try:
            with admin.connect() as c:
                c.execute(sa.text(f'drop database if exists "{name}"'))
                c.execute(sa.text(f'create database "{name}"'))
        except Exception as exc:  # noqa: BLE001
            admin.dispose()
            self.skipTest(f"cannot create a scratch database: {type(exc).__name__}")

        def drop():
            db.engine(target).dispose()
            with admin.connect() as c:
                c.execute(sa.text(f'drop database if exists "{name}" with (force)'))
            admin.dispose()
        self.addCleanup(drop)
        with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": url}):
            home = Path(tempfile.mkdtemp())
            client, s, _ = workspace(home)
            before = catalog.catalog(home)
            lineage = catalog.lineage(home, "decisions/column.is_identifier", s.experience, s.ledger)["sessions"]
            made = backup.backup(url, home, Path(tempfile.mkdtemp()))
        restored = Path(tempfile.mkdtemp()) / "restored"
        backup.restore(Path(made["folder"]), target, restored)
        self.assertEqual(catalog.catalog(restored), before)
        with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": target}):
            from dclab_rnd.experience import open_log

            wid = db.workspace_for(restored)
            log, book = open_log(restored, wid), ledger.open_ledger(restored, wid)
            self.assertEqual(catalog.lineage(restored, "decisions/column.is_identifier", log, book)["sessions"], lineage)


if __name__ == "__main__":
    unittest.main()
