"""Problem memory's bases and summaries behind the storage interface (finding 10 of the problem-memory review).

The same behaviour on files and on PostgreSQL: a stale revision is refused (409 over HTTP), a summary built from an older
cursor is never published over a newer one, a purge leaves a tombstone hidden from reads (never a deletion), and
processes racing on one revision leave one winner. On PostgreSQL also: a file workspace's bases and summaries are read
back after ``storage migrate``, and the database refuses to DELETE a base row.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402

from dclab_rnd.memory import bases, open_memory, scope as S, summary as SM, view  # noqa: E402
from dclab_rnd.storage import MemoryBases, open_memory_bases  # noqa: E402
from dclab_rnd.storage.memory_bases import FileMemoryBases, PgMemoryBases  # noqa: E402


def _scope(project_id="p0000000001", problem_id="prob_000000000001", role="data_scientist"):
    return S.AuthorizedScope(workspace_id="local", principal_id="dana", role=role, project_id=project_id, problem_id=problem_id)


RACER = r"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from dclab_rnd.memory import bases, scope as S
home, start, n = Path(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
sc = S.AuthorizedScope(workspace_id="local", principal_id="dana", role="data_scientist", project_id="p0000000001", problem_id="prob_000000000001")
while time.time() < start:
    time.sleep(0.001)
try:
    out = bases.upsert_base_fact(home, sc, subject="problem", field="prediction_moment", text=f"Predicted at moment number {n}.",
                                 source="race", expected_revision=0)
    print(json.dumps({"won": True, "revision": out["revision"]}))
except bases.RevisionConflictError as exc:
    print(json.dumps({"won": False, "current": exc.current_revision}))
"""


class _Contract:
    """Every test here runs once on files and once on PostgreSQL."""

    env: dict = {}
    store_type: type = object

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"DCLAB_AUTH": "", **self.env})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.home = Path(tempfile.mkdtemp()) / "workspace"
        self.home.mkdir()

    def test_the_store_is_chosen_as_the_other_stores_are(self):
        store = open_memory_bases(self.home)
        self.assertIsInstance(store, self.store_type)
        self.assertIsInstance(store, MemoryBases)

    def test_a_stale_revision_is_refused_and_nothing_is_written(self):
        sc = _scope()
        first = bases.upsert_base_fact(self.home, sc, subject="problem", field="prediction_moment", text="After dispatch.",
                                       source="ui", expected_revision=0)
        self.assertEqual(first["revision"], 1)
        for attempt in (lambda: bases.upsert_base_fact(self.home, sc, subject="problem", field="prediction_moment", text="At order creation.",
                                                       source="ui", expected_revision=0),
                        lambda: bases.correct_base_fact(self.home, sc, first["entry"]["entry_id"], "At order creation.", expected_revision=0),
                        lambda: bases.retire_base_fact(self.home, sc, first["entry"]["entry_id"], expected_revision=0)):
            with self.assertRaises(bases.RevisionConflictError) as caught:
                attempt()
            self.assertEqual(caught.exception.to_detail()["current_revision"], 1)
        held = bases.read_base(self.home, sc)
        self.assertEqual((held["revision"], [e["text"] for e in held["active"]]), (1, ["After dispatch."]))
        self.assertEqual([i.text for i in view(open_memory(self.home), sc.chain())], ["After dispatch."])  # the loser wrote no item
        # the project base keeps its own revision; replays and drafts are idempotent on either store
        self.assertEqual(bases.upsert_base_fact(self.home, sc, target="project", subject="glossary", field="sla", text="SLA is 45 minutes.",
                                                source="ui", expected_revision=0)["revision"], 1)
        e1 = bases.record_material_event(self.home, sc, source_type="decision", source_id="d1", source_revision="1", text="Dropped courier_rating.")
        e2 = bases.record_material_event(self.home, sc, source_type="decision", source_id="d1", source_revision="1", text="Dropped courier_rating.")
        self.assertEqual((e1["replayed"], e2["replayed"], e1["entry"]["entry_id"]), (False, True, e2["entry"]["entry_id"]))
        d = bases.propose_draft(self.home, sc, subject="features", field="policy", text="Keep dispatch_delay_min.", source_type="advisor", source_id="a1")
        accepted = bases.confirm_draft(self.home, sc, d["entry"]["entry_id"], expected_revision=1)
        self.assertEqual((accepted["revision"], accepted["entry"]["status"]), (2, "active"))

    def test_a_stale_revision_over_http_is_409(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.accounts.principal import Principal
        from dclab_rnd.agentic.server import create_app

        app = create_app(self.home)
        client = TestClient(app)
        p = app.state.services.projects.create("Delivery", goal="ETA")
        dana = Principal(role="data_scientist", workspace_id=None, user_id="dana", via="token")
        body = {"target": "problem", "subject": "problem", "field": "prediction_moment", "text": "Predicted after dispatch.", "expected_revision": 0}
        with mock.patch("dclab_rnd.agentic.server.identify", return_value=dana):
            made = client.post(f"/api/projects/{p['id']}/memory/base", json=body)
            self.assertEqual(made.status_code, 200, made.text)
            late = client.post(f"/api/projects/{p['id']}/memory/base", json={**body, "text": "Predicted at order creation."})
            self.assertEqual(late.status_code, 409, late.text)
            self.assertEqual(late.json()["detail"]["current_revision"], 1)
            shown = client.get(f"/api/projects/{p['id']}/memory/base").json()
        self.assertEqual([e["text"] for e in shown["active"]], ["Predicted after dispatch."])

    def test_a_summary_from_an_older_cursor_is_never_published(self):
        sc = _scope()
        first = bases.upsert_base_fact(self.home, sc, subject="problem", field="prediction_moment", text="After dispatch.", source="ui",
                                       expected_revision=0)
        slow = SM.build_from_snapshot(SM.capture_snapshot(self.home, sc))  # a worker starts from revision 1
        bases.correct_base_fact(self.home, sc, first["entry"]["entry_id"], "At order creation.", expected_revision=1)
        refused = SM.publish_summary(self.home, sc, slow)
        self.assertEqual((refused["published"], refused["reason"]), (False, "stale_source_cursor"))
        fresh = SM.read_summary(self.home, sc)
        self.assertIn("At order creation.", fresh["text"])
        self.assertNotIn("After dispatch.", fresh["text"])
        again = SM.read_summary(self.home, sc)
        self.assertTrue(again["from_cache"])  # published, and read back from the store

        # the store's own compare and swap, without the live check: an older cursor never replaces a newer one
        store = open_memory_bases(self.home)
        key = "local:p0000000001:prob_000000000001:shared"
        self.assertTrue(store.publish(key, {"project_id": "p0000000001", "problem_id": "prob_000000000001", "cursor": [9, 5, 4, 3, 0], "text": "new"})["published"])
        old = store.publish(key, {"project_id": "p0000000001", "problem_id": "prob_000000000001", "cursor": [9, 5, 3, 9, 9], "text": "old"})
        self.assertEqual((old["published"], old["reason"], old["stored_cursor"]), (False, "older_than_stored_summary", [9, 5, 4, 3, 0]))
        self.assertEqual(store.summary(key)["text"], "new")
        self.assertTrue(store.publish(key, {"project_id": "p0000000001", "problem_id": "prob_000000000001", "cursor": [9, 5, 4, 3, 0], "text": "same"})["published"])
        self.assertEqual(store.invalidate("p0000000001", "prob_000000000001"), 1)
        self.assertIsNone(store.summary(key))

    def test_a_purge_leaves_a_tombstone_hidden_from_reads(self):
        sc = _scope()
        bases.upsert_base_fact(self.home, sc, subject="problem", field="target", text="Target is actual minutes.", source="ui", expected_revision=0)
        bases.upsert_base_fact(self.home, sc, target="project", subject="business", field="sla", text="SLA is 45 minutes.", source="ui",
                               expected_revision=0)
        other = _scope(project_id="p0000000002", problem_id="prob_000000000002")
        bases.upsert_base_fact(self.home, other, subject="problem", field="target", text="Target is churn.", source="ui", expected_revision=0)
        SM.read_summary(self.home, sc)
        out = bases.purge_scope(self.home, "project", "p0000000001", project_id="p0000000001")
        self.assertEqual(out["removed_bases"], 2)
        self.assertEqual(out["removed_summaries"], 1)
        for target in ("problem", "project"):
            held = bases.read_base(self.home, sc, target)
            self.assertEqual((held["revision"], held["active"], held["retired"]), (0, [], []))
        self.assertEqual([e["text"] for e in bases.read_base(self.home, other)["active"]], ["Target is churn."])  # another project's stays
        kept = [doc for _, doc in open_memory_bases(self.home).all_bases() if doc.get("purged")]
        self.assertEqual(len(kept), 2)  # the ledgers are kept, with their tombstone and their words
        self.assertTrue(all(k["purged"]["reason"] == "project deleted" for k in kept))
        self.assertIn("Target is actual minutes.", [e["text"] for k in kept for e in k["entries"]])
        # a new base for the same scope starts again, beside the tombstone
        again = bases.upsert_base_fact(self.home, sc, subject="problem", field="target", text="Target is late or not.", source="ui",
                                       expected_revision=0)
        self.assertEqual(again["revision"], 1)
        self.assertEqual(bases.purge_scope(self.home, "project", "p0000000001", project_id="p0000000001")["removed_bases"], 1)
        self.assertEqual(len([d for _, d in open_memory_bases(self.home).all_bases() if d.get("purged")]), 3)

    def test_racing_processes_on_one_revision_leave_one_winner(self):
        start = time.time() + 2.0  # every process imports first, then all write at once
        procs = [subprocess.Popen([sys.executable, "-c", RACER, str(ROOT), str(self.home), str(start), str(n)], cwd=ROOT,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ, **self.env})
                 for n in range(4)]
        results = []
        for proc in procs:
            out, err = proc.communicate(timeout=120)
            self.assertEqual(proc.returncode, 0, err[-2000:])
            results.append(json.loads(out.strip().splitlines()[-1]))
        self.assertEqual(sum(r["won"] for r in results), 1, results)
        self.assertTrue(all(r["current"] == 1 for r in results if not r["won"]), results)
        held = bases.read_base(self.home, _scope())
        self.assertEqual((held["revision"], len(held["active"]), held["retired"]), (1, 1, []))
        items = view(open_memory(self.home), _scope().chain())
        self.assertEqual([i.text for i in items], [e["text"] for e in held["active"]])  # only the winner's memory item


class FileBasesTests(_Contract, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}
    store_type = FileMemoryBases

    def test_the_files_are_where_they_were(self):
        sc = _scope()
        bases.upsert_base_fact(self.home, sc, subject="problem", field="target", text="Target is minutes.", source="ui", expected_revision=0)
        SM.read_summary(self.home, sc)
        self.assertEqual(len(list((self.home / "memory" / "bases").glob("problem_*.json"))), 1)
        self.assertEqual(len(list((self.home / "memory" / "summaries").glob("sum_*.json"))), 1)
        bases.purge_scope(self.home, "project", "p0000000001", project_id="p0000000001")
        self.assertEqual(list((self.home / "memory" / "bases").glob("*.json")), [])
        self.assertEqual(len(list((self.home / "memory" / "bases" / "purged").glob("*.json"))), 1)


class PgBasesTests(_Contract, unittest.TestCase):
    store_type = PgMemoryBases

    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}

    def test_no_file_is_written_and_a_base_row_cannot_be_deleted(self):
        import sqlalchemy as sa

        from dclab_rnd.storage import db
        from dclab_rnd.storage.models import memory_base

        sc = _scope()
        bases.upsert_base_fact(self.home, sc, subject="problem", field="target", text="Target is minutes.", source="ui", expected_revision=0)
        SM.read_summary(self.home, sc)
        self.assertFalse((self.home / "memory" / "bases").exists())
        self.assertFalse((self.home / "memory" / "summaries").exists())
        wid = db.workspace_for(self.home)
        with self.assertRaises(sa.exc.DBAPIError):
            with db.engine(self.url).begin() as c:
                c.execute(sa.delete(memory_base).where(memory_base.c.workspace_id == wid))
        self.assertEqual(open_memory_bases(self.home).counts(), {"bases": 1, "summaries": 1})

    def test_a_write_after_its_row_changed_underneath_is_refused(self):
        import sqlalchemy as sa

        from dclab_rnd.storage import db
        from dclab_rnd.storage.memory_bases import BaseChanged
        from dclab_rnd.storage.models import memory_base

        store = open_memory_bases(self.home)
        store.change("problem", "x", {"scope": "problem", "scope_id": "x", "revision": 0}, lambda doc: (True, None))

        def meanwhile(doc):
            with db.engine(self.url).begin() as c:  # another writer that took no lock (only a purge could): the version moves
                c.execute(sa.update(memory_base).where(memory_base.c.scope_id == "x", memory_base.c.workspace_id == store.workspace_id)
                          .values(version=memory_base.c.version + 1))
            doc["revision"] = 1
            return True, None

        with self.assertRaises(BaseChanged):
            store.change("problem", "x", {}, meanwhile)
        self.assertEqual(store.read("problem", "x")["revision"], 0)

    def test_reads_after_a_migrate(self):
        from dclab_rnd.storage.migrate import migrate
        from dclab_rnd.studio.store import ProjectStore

        source = Path(tempfile.mkdtemp()) / "files-workspace"
        sc = _scope()
        with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": ""}):
            ProjectStore(source / "projects")
            first = bases.upsert_base_fact(source, sc, subject="problem", field="prediction_moment", text="After dispatch.", source="ui",
                                           expected_revision=0)
            bases.correct_base_fact(source, sc, first["entry"]["entry_id"], "At order creation.", expected_revision=1)
            bases.upsert_base_fact(source, sc, target="project", subject="business", field="sla", text="SLA is 45 minutes.", source="ui",
                                   expected_revision=0)
            gone = _scope(project_id="p0000000009", problem_id="prob_000000000009")
            bases.upsert_base_fact(source, gone, subject="problem", field="target", text="Target is churn.", source="ui", expected_revision=0)
            bases.purge_scope(source, "project", "p0000000009", project_id="p0000000009")
            published = SM.read_summary(source, sc)
            before = {t: bases.read_base(source, sc, t) for t in ("problem", "project")}
        self.assertTrue(published["fresh"])

        report = migrate(source, self.home, self.url)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["logs"]["memory_bases"], {"source": 3, "copied": 3, "already": 0})
        self.assertEqual(report["logs"]["memory_summaries"], {"source": 1, "copied": 1, "already": 0})
        again = migrate(source, self.home, self.url)
        self.assertTrue(again["ok"], again)
        self.assertEqual(again["logs"]["memory_bases"]["copied"], 0)
        self.assertEqual(again["logs"]["memory_summaries"]["copied"], 0)

        for target in ("problem", "project"):
            moved = bases.read_base(self.home, sc, target)
            self.assertEqual({k: moved[k] for k in ("revision", "active", "retired", "drafts")},
                             {k: before[target][k] for k in ("revision", "active", "retired", "drafts")})
        self.assertEqual(bases.read_base(self.home, gone)["active"], [])  # the tombstone came along and still hides it
        self.assertEqual(len([d for _, d in open_memory_bases(self.home).all_bases() if d.get("purged")]), 1)
        cached = SM.read_summary(self.home, sc)
        self.assertTrue(cached["from_cache"])  # the copied summary, at the live cursor
        self.assertIn("At order creation.", cached["text"])
        # and the database workspace goes on from the copied revision
        with self.assertRaises(bases.RevisionConflictError):
            bases.upsert_base_fact(self.home, sc, subject="problem", field="target", text="Target is minutes.", source="ui", expected_revision=0)
        self.assertEqual(bases.upsert_base_fact(self.home, sc, subject="problem", field="target", text="Target is minutes.", source="ui",
                                                expected_revision=2)["revision"], 3)
        self.assertEqual(FileMemoryBases(source).counts(), {"bases": 3, "summaries": 1})  # the source is only read


if __name__ == "__main__":
    unittest.main()
