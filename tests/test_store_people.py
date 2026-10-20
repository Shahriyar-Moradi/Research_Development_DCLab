"""Package 32.32: each person's account, sessions and work, kept for training without anything that identifies or lets
anyone in. The people keeper reads the account tables by named columns and keeps a pseudonymous id, a salted hash of the
email, how they sign in, their role and their sessions; a password, a session cookie or its digest, a request token, an API
token, an email or a name never reaches the store (the plants below), a removed person stays in the log, and every record
that says whose it is carries it (``body.user``, ``provenance.by``). Needs the PostgreSQL test database (skipped without it).
"""

import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import pgtest
    from dclab_rnd.accounts.store import Accounts
    from dclab_rnd.storage import db
    from dclab_rnd.store import Layout, Store, conductor, shapers
    from dclab_rnd.store.keepers import standard
    from dclab_rnd.store.keepers import people as keeper
except ImportError as error:
    raise unittest.SkipTest(str(error))

PASSWORD = "Plant-Password-9F3kq-XYZ"
EMAIL = "plant.person.check@example.org"
NAME = "Plantname Checkperson"
SUBJECT = "idp|plant-subject-7731"
USERNAME = "plant-username-4471"


class PeopleTests(unittest.TestCase):
    def setUp(self):
        self.url = pgtest.require()
        pgtest.empty(self.url)
        self.scratch = Path(tempfile.mkdtemp(prefix="dclab-people-test-"))
        self.base = self.scratch / "agent_runs"
        (self.base / "day").mkdir(parents=True)
        self.layout = Layout(self.scratch / "store")
        env = mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": self.url, "DCLAB_STORE": str(self.layout.root), "DCLAB_STORE_LOOP": "0",
                                           "DCLAB_STORE_COMMITS": "off", "DCLAB_STORE_CLAUDE_PROJECTS": "off"})
        env.start()
        self.addCleanup(env.stop)
        patch = mock.patch("dclab_rnd.store.keepers.workspaces.AGENT_RUNS", self.base)
        patch.start()
        self.addCleanup(patch.stop)
        self.accounts = Accounts(self.url)
        self.workspace = db.workspace_for(self.base / "day")
        self.user = self.accounts.create_user(EMAIL, NAME, password=PASSWORD, username=USERNAME)
        self.accounts.set_member(self.workspace, self.user["id"], "owner")
        self.cookie, self.csrf = self.accounts.start_session(self.user["id"], self.workspace)
        self.api, _ = self.accounts.create_token(self.user["id"], self.workspace, "my laptop plant")
        self.sso = self.accounts.create_user("sso.person@example.org", "Sso Person", subject=SUBJECT)
        self.accounts.set_member(self.workspace, self.sso["id"], "viewer")

    def keep(self):
        from sqlalchemy import text

        with db.engine(self.url).connect() as c:
            self.digest = c.execute(text("select id from user_session")).scalar()
            self.token_hash = c.execute(text("select token_hash from api_token")).scalar()
            self.password_hash = c.execute(text("select password_hash from app_user where email = :e"), {"e": EMAIL}).scalar()
        return conductor.conduct({"agent_runs": self.base, "claude_projects": None, "test_runs": self.scratch / "none"}, self.layout, standard.install({}))

    def texts(self):
        def opened(p: Path) -> bytes:
            data = gzip.decompress(p.read_bytes())
            return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data

        raw = b"".join(opened(p) for p in self.layout.tier("raw").rglob("*.gz"))
        records = b"".join(gzip.decompress(p.read_bytes()) for p in self.layout.tier("records").rglob("*.gz"))
        return raw, records

    def test_people_and_sessions_are_kept_and_nothing_that_identifies_or_lets_in(self):
        record = self.keep()
        self.assertFalse(record["failed"], record["failed"])
        raw, records = self.texts()
        everything = raw + records
        for plant in (PASSWORD, self.cookie, self.csrf, self.digest, self.api, self.token_hash, self.password_hash, EMAIL, USERNAME, NAME, SUBJECT, "my laptop plant", "Sso Person"):
            self.assertNotIn(str(plant).encode(), everything, plant)
        entries = {e.source for e in Store(self.layout).entries("sessions")}
        self.assertIn("people/person.jsonl", entries)
        self.assertIn("people/session.jsonl", entries)
        people = [r for r in shapers.records(self.layout, "sessions") if r.type == "person"]
        by = {r.body["user"]: r for r in people}
        self.assertEqual(set(by), {self.user["id"], self.sso["id"]})
        mine = by[self.user["id"]].body
        self.assertEqual((mine["signs_in"], mine["role"], mine["event"], mine["active"], mine["api_tokens"]), ("password", "owner", "joined", True, 1))
        self.assertEqual(by[self.sso["id"]].body["signs_in"], "sso")
        self.assertEqual(by[self.user["id"]].provenance.by, self.user["id"])  # whose it is
        self.assertEqual((len(mine["email_hash"]), len(mine["username_hash"])), (16, 16))
        sessions = [r for r in shapers.records(self.layout, "sessions") if r.type == "session"]
        self.assertEqual([r.body["user"] for r in sessions], [self.user["id"]])
        self.assertNotEqual(sessions[0].body["session"], self.digest)  # hashed again, salted

    def test_each_person_has_a_record_with_their_sessions_and_work(self):
        from dclab_rnd.store import service

        self.keep()
        out = service.people(self.layout)
        by = {p["user"]: p for p in out["people"]}
        mine = by[self.user["id"]]
        self.assertEqual((mine["role"], mine["signs_in"], mine["sessions"], mine["removed"]), ("owner", "password", 1, False))
        self.assertEqual(by[self.sso["id"]]["signs_in"], "sso")
        self.assertEqual(out["count"], 2)
        self.assertFalse(out["enough"])  # two people: too few to say anything across them
        self.assertNotIn(EMAIL, json.dumps(out))
        service.write(self.layout)
        self.assertTrue(self.layout.service("people").exists())

    def test_the_salt_stays_beside_the_store_never_in_it(self):
        self.keep()
        state = self.layout.root.with_name(self.layout.root.name + ".people") / "state.json"
        self.assertTrue(state.exists())
        salt = json.loads(state.read_text())["salt"]
        raw, records = self.texts()
        self.assertNotIn(salt.encode(), raw + records)
        self.assertFalse(any(self.layout.root in p.parents for p in [state]))
        self.assertEqual(state.stat().st_mode & 0o077, 0)  # the owner's alone

    def test_the_same_run_again_adds_nothing_a_change_and_a_removal_are_events_and_the_person_stays(self):
        self.keep()
        before = len([r for r in shapers.records(self.layout, "sessions") if r.type == "person"])
        self.keep()
        self.assertEqual(len([r for r in shapers.records(self.layout, "sessions") if r.type == "person"]), before)  # unchanged: nothing new
        self.accounts.set_member(self.workspace, self.sso["id"], "reviewer")
        self.accounts.remove_member(self.workspace, self.user["id"])
        self.keep()
        events = [(r.body["user"], r.body["event"], r.body.get("role")) for r in shapers.records(self.layout, "sessions") if r.type == "person"]
        self.assertIn((self.sso["id"], "changed", "reviewer"), events)
        self.assertIn((self.user["id"], "removed", "owner"), events)  # removed from the workspace: its last record stays, said removed
        self.assertIn((self.user["id"], "joined", "owner"), events)

    def test_a_workspace_without_consent_keeps_counts_only(self):
        other = db.workspace_for(self.scratch / "elsewhere")  # a folder that is not an owner's workspace: its consent is operate
        self.accounts.set_member(other, self.user["id"], "viewer")
        self.keep()
        sources = {(e.workspace, e.source) for e in Store(self.layout).entries("sessions") if e.source.startswith("people/")}
        self.assertTrue(any(s.endswith("~counts.json") for _, s in sources), sources)
        counted = [w for w, s in sources if s.endswith("~counts.json")]
        self.assertFalse(any(w in counted for w, s in sources if s == "people/person.jsonl"))


class WhoseItIsTests(unittest.TestCase):
    def test_every_kind_of_record_carries_the_user_it_came_from(self):
        from dclab_core.store import Provenance  # noqa: F401 - the field exists

        body = shapers._body({"kind": "ui.action", "ids": {"user": "u1234", "visit": "v1"}, "at": "2026-10-20T10:00:00+00:00"}, "sessions", "experience/x.jsonl")
        self.assertEqual(body["user"], "u1234")
        row = shapers._body({"user_id": "u9999", "id": 3, "at": "2026-10-20T10:00:00+00:00"}, "sessions", "db/experience_event.jsonl")
        self.assertEqual(row["user"], "u9999")


if __name__ == "__main__":
    unittest.main()
