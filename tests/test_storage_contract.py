"""One set of behaviours, run against the file stores and against PostgreSQL (packages 9.1 and 9.2)."""
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pgtest  # noqa: E402
from dclab_rnd import storage  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402
from dclab_rnd.intern.sessions import SessionStore  # noqa: E402
from dclab_rnd.studio.store import STAGE_KEYS, ProjectStore  # noqa: E402


class Contract:
    """Mixed into one TestCase per backend; ``self.make(home)`` returns (projects, drafts, sessions) for a workspace."""

    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.projects, self.drafts, self.sessions = self.make(self.home)

    # ---- projects
    def test_project_documents(self):
        p = self.projects.create("Churn", "telecom", "Predict who leaves")
        self.assertEqual((p["name"], p["industry"], p["stages"]["data"]), ("Churn", "telecom", {"status": "pending"}))
        got = self.projects.get(p["id"])
        got["goal"] = "changed"
        self.assertEqual(self.projects.get(p["id"])["goal"], "Predict who leaves")  # what get returns is a copy
        self.projects.save(got)
        self.assertEqual(self.projects.get(p["id"])["goal"], "changed")
        self.assertEqual(self.projects.update(p["id"], goal="again")["goal"], "again")
        for bad in ("nope", "../x", "", "0" * 12):
            with self.assertRaises(KeyError):
                self.projects.get(bad)
        a = self.projects.create("A")
        time.sleep(1.1)  # the file store records time to the second: leave a gap so this checks ordering, not a tie
        self.projects.update(a["id"], goal="newest")
        self.assertEqual([x["id"] for x in self.projects.list()][0], a["id"])  # newest change first
        self.assertEqual(len(self.projects.list(limit=1)), 1)
        self.assertEqual(len(self.projects.list(limit=5, offset=1)), 1)
        self.projects.delete(a["id"])
        self.assertEqual([x["id"] for x in self.projects.list()], [p["id"]])
        self.projects.delete(a["id"])  # deleting twice is fine

    def test_old_documents_are_migrated_when_read(self):
        p = self.projects.create("Old")
        doc = self.projects.get(p["id"])
        doc["contract"], doc["solution"] = {"target": "y"}, None
        doc.pop("solution")
        self.projects.save(doc)
        self.assertEqual(self.projects.get(p["id"]).get("solution"), {"target": "y"})
        self.assertNotIn("contract", self.projects.get(p["id"]))

    def test_stage_records(self):
        p = self.projects.create("S")["id"]
        self.assertFalse(self.projects.has_stage(p, "data"))
        self.assertIsNone(self.projects.read_stage(p, "data"))
        with self.assertRaises(ValueError):  # a score that is not a number is a defect to surface, not to store
            self.projects.write_stage(p, "data", {"stage": "data", "score": float("nan")})
        self.assertFalse(self.projects.has_stage(p, "data"))
        self.projects.write_stage(p, "leakage", {"stage": "leakage"})
        self.projects.write_stage(p, "data", {"stage": "data", "n": 4})  # a rerun replaces it
        self.assertEqual(self.projects.read_stage(p, "data")["n"], 4)
        self.assertEqual(list(self.projects.records(p)), ["data", "leakage"])  # in stage order
        with self.assertRaises(KeyError):
            self.projects.write_stage(p, "nonsense", {})
        doc = self.projects.get(p)
        doc["stages"]["leakage"], doc["decisions"] = {"status": "completed"}, {"data": "x", "leakage": "y"}
        self.projects.save(doc)
        self.projects.clear_stages(p, "leakage")
        after = self.projects.get(p)
        self.assertEqual((sorted(self.projects.records(p)), after["stages"]["leakage"], after["decisions"]), (["data"], {"status": "pending"}, {"data": "x"}))

    def test_append_only_logs(self):
        p = self.projects.create("L")["id"]
        for i in range(5):
            self.projects.log(p, "step", {"i": i})
            self.projects.transition(p, {"at": f"2026-10-0{i + 1}", "move": "run_stage", "status": "allowed" if i % 2 == 0 else "blocked", "actor": "human", "n": i})
        self.assertEqual([a["payload"]["i"] for a in self.projects.activity(p, 3)], [2, 3, 4])  # the latest three, oldest first
        self.assertEqual(self.projects.activity(p)[0]["kind"], "created")
        self.assertEqual([t["n"] for t in self.projects.transitions(p, 2)], [3, 4])
        self.assertEqual(self.projects.transitions(p)[0]["status"], "allowed")
        other = self.projects.create("Other")["id"]
        self.assertEqual(self.projects.transitions(other), [])

    def test_files_live_beside_the_workspace(self):
        p = self.projects.create("F")["id"]
        (self.projects.data_dir(p) / "t.csv").write_text("a\n1\n")
        self.assertTrue((self.projects.export_dir(p)).is_dir())
        self.assertTrue(self.projects.location())

    # ---- drafts
    def test_draft_documents_and_events(self):
        d = self.drafts.create("Predict which customers cancel", pack="timeseries")
        self.assertEqual((d["status"], d["pack"]["key"]), ("open", "timeseries"))
        self.assertEqual(self.drafts.events_version(d["id"]), (0, 0))
        e1 = self.drafts.emit(d["id"], "chat", {"text": "hi"})
        e2 = self.drafts.emit(d["id"], "status", {"thinking": True})
        self.assertEqual((e1["seq"], e2["seq"]), (1, 2))
        self.assertEqual([e["seq"] for e in self.drafts.events(d["id"])], [1, 2])
        self.assertEqual([e["kind"] for e in self.drafts.events(d["id"], after=1)], ["status"])
        self.assertNotEqual(self.drafts.events_version(d["id"]), (0, 0))
        self.drafts.update(d["id"], lambda x: x["messages"].append({"text": "m"}))
        self.assertEqual(self.drafts.get(d["id"])["messages"], [{"text": "m"}])
        with self.assertRaises(KeyError):
            self.drafts.get("0" * 12)
        with self.assertRaises(KeyError):
            self.drafts.update("0" * 12, lambda x: None)
        self.drafts.delete(d["id"])
        with self.assertRaises(KeyError):
            self.drafts.get(d["id"])

    def test_concurrent_events_get_distinct_numbers_and_updates_are_not_lost(self):
        d = self.drafts.create("Predict which customers cancel")["id"]

        def work():
            for _ in range(10):
                self.drafts.emit(d, "tick", {})
                self.drafts.update(d, lambda x: x["messages"].append({"n": 1}))
        threads = [threading.Thread(target=work) for _ in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(sorted(e["seq"] for e in self.drafts.events(d)), list(range(1, 61)))
        self.assertEqual(len(self.drafts.get(d)["messages"]), 60)

    def test_agent_turns_are_one_at_a_time_and_re_entrant(self):
        d = self.drafts.create("Predict which customers cancel")["id"]
        order, inside = [], threading.Event()

        def first():
            with self.drafts.turn(d):
                with self.drafts.turn(d):  # the data-ready turn inside a reply
                    order.append("a-start")
                    inside.set()
                    threading.Event().wait(0.3)
                    order.append("a-end")

        def second():
            inside.wait(5)
            with self.drafts.turn(d):
                order.append("b")
        threads = [threading.Thread(target=first), threading.Thread(target=second)]
        [t.start() for t in threads]
        [t.join(10) for t in threads]
        self.assertEqual(order, ["a-start", "a-end", "b"])

    # ---- sessions
    def test_intern_sessions(self):
        s = self.sessions.create("Audit the churn data", mode="standard", model=None, budget={"max_steps": 9999, "max_minutes": 0})
        self.assertEqual((s["budget"]["max_steps"], s["budget"]["max_minutes"], s["status"]), (80, 1, "queued"))
        s["steps"], s["messages"], s["status"] = [{"tool": "a"}, {"tool": "b"}], [{"role": "user"}], "running"
        self.sessions.save(s)
        listed = self.sessions.list()
        self.assertEqual((len(listed), listed[0]["steps"], "messages" in listed[0], listed[0]["status"]), (1, 2, False, "running"))
        self.assertEqual(len(self.sessions.get(s["id"])["messages"]), 1)
        self.assertEqual(self.sessions.list(limit=0), [])
        self.sessions.delete(s["id"])
        with self.assertRaises(KeyError):
            self.sessions.get(s["id"])


class FileStoreContract(Contract, unittest.TestCase):
    def make(self, home):
        return ProjectStore(home / "projects"), DraftStore(home / "drafts"), SessionStore(home / "intern")


class PostgresContract(Contract, unittest.TestCase):
    def setUp(self):
        self.url = pgtest.require()
        pgtest.empty(self.url)
        super().setUp()

    def make(self, home):
        from dclab_rnd.storage.postgres import open_stores
        return open_stores(home, self.url)

    def test_the_stores_satisfy_the_protocols(self):
        for store, protocol in zip((self.projects, self.drafts, self.sessions), (storage.Projects, storage.Drafts, storage.Sessions)):
            self.assertIsInstance(store, protocol)

    def test_a_workspace_never_sees_another_workspaces_rows(self):
        mine, theirs = self.projects, self.make(Path(tempfile.mkdtemp()))[0]
        p = mine.create("Mine")
        self.assertEqual([x["id"] for x in theirs.list()], [])
        with self.assertRaises(KeyError):
            theirs.get(p["id"])
        with self.assertRaises(KeyError):
            theirs.update(p["id"], name="stolen")
        with self.assertRaises(KeyError):
            theirs.log(p["id"], "x", {})
        theirs.delete(p["id"])  # deleting someone else's project deletes nothing
        self.assertEqual(mine.get(p["id"])["name"], "Mine")
        d = self.drafts.create("Mine too")
        other_drafts, other_sessions = self.make(Path(tempfile.mkdtemp()))[1:]
        with self.assertRaises(KeyError):
            other_drafts.get(d["id"])
        with self.assertRaises(KeyError):
            other_drafts.emit(d["id"], "x", {})
        self.assertEqual(other_drafts.events(d["id"]), [])
        s = self.sessions.create("Mine", mode="standard", model=None)
        with self.assertRaises(KeyError):
            other_sessions.get(s["id"])
        with self.assertRaises(KeyError):
            other_sessions.save(dict(s))  # saving over another workspace's id is refused

    def test_the_same_folder_is_the_same_workspace(self):
        again = self.make(self.home)[0]
        p = self.projects.create("Shared")
        self.assertEqual(again.get(p["id"])["name"], "Shared")

    def test_project_documents_may_hold_numbers_json_cannot_write(self):
        p = self.projects.create("N")["id"]
        self.projects.update(p, settings={"x": float("nan"), "y": float("inf"), "ok": 1.5})  # JSONB has no NaN: stored as null
        self.assertEqual(self.projects.get(p)["settings"], {"x": None, "y": None, "ok": 1.5})

    def test_deleting_a_project_deletes_its_rows_and_files(self):
        from dclab_rnd.storage import db
        import sqlalchemy as sa
        p = self.projects.create("D")["id"]
        self.projects.log(p, "x", {})
        self.projects.transition(p, {"move": "m"})
        self.projects.write_stage(p, "data", {})
        (self.projects.data_dir(p) / "t.csv").write_text("a")
        self.projects.delete(p)
        with db.engine(self.url).connect() as c:
            left = [c.execute(sa.text(f"select count(*) from {t}")).scalar() for t in ("activity", "transition", "stage_record")]
        self.assertEqual((left, self.projects.data_dir.__self__._folder(p).exists()), ([0, 0, 0], False))


if __name__ == "__main__":
    unittest.main()
