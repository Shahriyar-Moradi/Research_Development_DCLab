"""Teams and who is in them (W2 of the collaborative workspace), on files and on PostgreSQL.

A team is not a role: a role says what a person may do, a team says whom an agent represents. The same store behaviour
on both: a person in two teams, removal, a name taken once per workspace, any lowercase kind, and another workspace
seeing nothing (not even a team whose id it knows). On PostgreSQL also: a member is an account, and leaving the
workspace removes the person from its teams. Over HTTP the owner manages teams; a viewer and an ML engineer are refused,
and the other workspace's owner can neither see nor change them. The new permissions: nobody may contact customers by
default, only the owner approves a decision, the owner and ML engineers contribute.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pgtest  # noqa: E402

from dclab_rnd.storage import Teams, open_teams  # noqa: E402
from dclab_rnd.storage.teams import FileTeams, NoSuchTeam, PgTeams, TeamError  # noqa: E402

PASSWORD = "a long enough password"


class PermissionTests(unittest.TestCase):
    def test_the_new_permissions_and_who_has_them(self):
        from dclab_rnd.accounts.roles import ROLES, allows

        self.assertEqual({r for r in ROLES if allows(r, "contact_customers")}, set())  # granted, never assumed
        self.assertEqual({r for r in ROLES if allows(r, "approve_decision")}, {"owner"})
        self.assertEqual({r for r in ROLES if allows(r, "contribute")}, {"owner", "data_scientist"})
        self.assertFalse(allows(None, "contribute"))

    def test_the_team_routes_are_the_owners_and_reading_is_every_members(self):
        from dclab_rnd.accounts.guard import permission

        self.assertEqual(permission("GET", "/api/workspace/teams"), "read")
        for method, path in (("POST", "/api/workspace/teams"), ("POST", "/api/workspace/teams/{team_id}/members"),
                             ("DELETE", "/api/workspace/teams/{team_id}/members/{user_id}")):
            self.assertEqual(permission(method, path), "admin", path)


class _Contract:
    """Every test here runs once on files and once on PostgreSQL."""

    env: dict = {}
    store_type: type = object

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"DCLAB_AUTH": "", **self.env})
        patcher.start()
        self.addCleanup(patcher.stop)
        root = Path(tempfile.mkdtemp())
        self.home, self.other_home = root / "workspace", root / "other"
        self.home.mkdir()
        self.other_home.mkdir()

    def person(self, name: str) -> str:
        return name

    def test_the_store_is_chosen_as_the_other_stores_are(self):
        store = open_teams(self.home)
        self.assertIsInstance(store, self.store_type)
        self.assertIsInstance(store, Teams)

    def test_a_person_in_two_teams_and_removal(self):
        store = open_teams(self.home)
        business = store.create_team("Business", "business")
        science = store.create_team("Data Science", "data_science")
        dana, sam = self.person("dana"), self.person("sam")
        row, added = store.add_member(business["id"], dana)
        self.assertTrue(added)
        self.assertEqual((row["team_id"], row["user_id"]), (business["id"], dana))
        self.assertTrue(row["since"])
        self.assertFalse(store.add_member(business["id"], dana)[1])  # twice changes nothing
        store.add_member(science["id"], dana)
        store.add_member(science["id"], sam)
        self.assertEqual({t["name"] for t in store.teams_of(dana)}, {"Business", "Data Science"})
        self.assertEqual([t["name"] for t in store.teams_of(sam)], ["Data Science"])
        self.assertTrue(store.is_member(business["id"], dana))
        self.assertFalse(store.is_member(business["id"], sam))
        shown = {t["name"]: t for t in store.teams()}
        self.assertEqual({m["user_id"] for m in shown["Data Science"]["members"]}, {dana, sam})
        self.assertEqual(shown["Data Science"]["kind_label"], "Data Science")
        self.assertEqual(store.counts(), {"teams": 2, "memberships": 3})

        self.assertTrue(store.remove_member(business["id"], dana))
        self.assertFalse(store.remove_member(business["id"], dana))  # already gone
        self.assertFalse(store.is_member(business["id"], dana))
        self.assertEqual([t["name"] for t in store.teams_of(dana)], ["Data Science"])
        self.assertEqual(store.remove_user(dana), 1)  # left the workspace: every team
        self.assertEqual(store.teams_of(dana), [])
        self.assertEqual(store.counts(), {"teams": 2, "memberships": 1})
        with self.assertRaises(NoSuchTeam):
            store.add_member("team_nosuch", dana)
        with self.assertRaises(NoSuchTeam):
            store.remove_member("team_nosuch", dana)

    def test_names_and_kinds(self):
        store = open_teams(self.home)
        finance = store.create_team("  Finance   team ", "Finance")  # kind is configuration: another word is allowed
        self.assertEqual((finance["name"], finance["kind"], finance["kind_label"]), ("Finance team", "finance", "Finance"))
        self.assertEqual(store.create_team("ML", "ml-engineering")["kind"], "ml_engineering")
        for name, kind in (("finance TEAM", "finance"), ("", "sales"), ("Sales", ""), ("Sales", "9sales"), ("Sales", "sales;drop"), ("x" * 81, "sales")):
            with self.assertRaises(TeamError, msg=(name, kind)):
                store.create_team(name, kind)
        self.assertEqual([t["name"] for t in store.teams()], ["Finance team", "ML"])
        store.create_team("Straße", "sales")
        with self.assertRaises(TeamError):  # the same name on both stores, by casefold
            store.create_team("STRASSE", "sales")

    def test_another_workspace_sees_nothing(self):
        mine, theirs = open_teams(self.home), open_teams(self.other_home)
        sales = mine.create_team("Sales", "sales")
        dana = self.person("dana")
        mine.add_member(sales["id"], dana)
        self.assertEqual(theirs.teams(), [])
        self.assertEqual(theirs.teams_of(dana), [])
        self.assertIsNone(theirs.team(sales["id"]))
        self.assertFalse(theirs.is_member(sales["id"], dana))
        with self.assertRaises(NoSuchTeam):  # knowing the id is not enough
            theirs.add_member(sales["id"], self.person("eve"))
        with self.assertRaises(NoSuchTeam):
            theirs.remove_member(sales["id"], dana)
        self.assertEqual(theirs.remove_user(dana), 0)
        self.assertTrue(mine.is_member(sales["id"], dana))
        self.assertEqual(theirs.create_team("Sales", "sales")["name"], "Sales")  # a name is unique per workspace only
        self.assertEqual(theirs.counts(), {"teams": 1, "memberships": 0})


class FileTeamsTests(_Contract, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}
    store_type = FileTeams

    def test_the_file_is_in_the_workspace_folder_and_a_broken_one_is_never_overwritten(self):
        store = open_teams(self.home)
        store.create_team("Marketing", "marketing")
        path = self.home / "teams" / "teams.json"
        self.assertTrue(path.is_file())
        from dclab_rnd.storage.teams import BrokenTeamsFile

        for broken in ("[not a teams file", '{"teams": "no"}'):
            path.write_text(broken, encoding="utf-8")
            for attempt in (lambda: store.create_team("Sales", "sales"), store.teams, lambda: store.remove_user("dana")):
                with self.assertRaises(BrokenTeamsFile) as caught:
                    attempt()
                self.assertNotIn(str(self.home), str(caught.exception))  # the message reaches an HTTP answer: no path in it
            self.assertEqual(path.read_text(encoding="utf-8"), broken)  # never overwritten

    def test_without_sign_in_teams_are_made_but_people_are_not_placed(self):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            raise unittest.SkipTest(f"Studio dependencies not installed: {error}")
        with TestClient(create_app(self.home)) as client:
            client.headers["X-DCLab-Token"] = client.get("/api/config").json()["csrf"]
            made = client.post("/api/workspace/teams", json={"name": "Business", "kind": "business"})
            self.assertEqual(made.status_code, 201, made.text)
            self.assertEqual(client.post("/api/workspace/teams", json={"name": "business", "kind": "business"}).status_code, 409)
            self.assertEqual(client.post("/api/workspace/teams", json={"name": "Ops", "kind": "ops!"}).status_code, 422)
            placed = client.post(f"/api/workspace/teams/{made.json()['id']}/members", json={"user_id": "owner"})
            self.assertEqual(placed.status_code, 409, placed.text)  # the local owner is the only person: members need accounts
            self.assertEqual(client.post("/api/workspace/teams/team_nosuch/members", json={"user_id": "owner"}).status_code, 404)
            self.assertEqual([t["name"] for t in client.get("/api/workspace/teams").json()], ["Business"])
            (self.home / "teams" / "teams.json").write_text("{", encoding="utf-8")
            for answer in (client.get("/api/workspace/teams"), client.post("/api/workspace/teams", json={"name": "Sales", "kind": "sales"}),
                           client.post(f"/api/workspace/teams/{made.json()['id']}/members", json={"user_id": "owner"})):
                self.assertEqual(answer.status_code, 503, answer.text)
                self.assertNotIn(str(self.home), answer.text)


class PgTeamsTests(_Contract, unittest.TestCase):
    store_type = PgTeams

    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}

    def person(self, name: str) -> str:
        from dclab_rnd.accounts.store import Accounts

        from dclab_rnd.storage import db

        accounts = Accounts()
        user = accounts.create_user(f"{name}-{os.urandom(4).hex()}@example.com", name.title(), password=PASSWORD)["id"]
        accounts.set_member(db.workspace_for(self.home), user, "viewer")  # a member is a person of the workspace
        return user

    def test_a_member_is_an_account_and_leaving_the_workspace_leaves_its_teams(self):
        from dclab_rnd.accounts.store import Accounts
        from dclab_rnd.storage import db

        store = open_teams(self.home)
        sales = store.create_team("Sales", "sales")
        with self.assertRaises(TeamError):
            store.add_member(sales["id"], "u_not_an_account")
        accounts, wid = Accounts(), db.workspace_for(self.home)
        outsider = accounts.create_user(f"out-{os.urandom(4).hex()}@example.com", "Outsider", password=PASSWORD)["id"]
        with self.assertRaises(TeamError):  # an account, but not of this workspace: refused by the store itself
            store.add_member(sales["id"], outsider)
        self.assertEqual(store.teams_of(outsider), [])
        other_wid = db.workspace_for(self.other_home)
        dana = self.person("dana")
        accounts.set_member(wid, dana, "data_scientist")
        accounts.set_member(other_wid, dana, "viewer")
        store.add_member(sales["id"], dana)
        theirs = open_teams(self.other_home)
        marketing = theirs.create_team("Marketing", "marketing")
        theirs.add_member(marketing["id"], dana)
        accounts.remove_member(wid, dana)  # leaves this workspace: this workspace's teams, in the same transaction
        self.assertEqual(store.teams_of(dana), [])
        self.assertEqual([t["name"] for t in theirs.teams_of(dana)], ["Marketing"])  # the other workspace's teams stay
        accounts.sync_memberships(dana, {}, {other_wid})  # the identity provider's groups remove the other one
        self.assertEqual(theirs.teams_of(dana), [])

    def test_storage_migrate_moves_a_folders_teams_once(self):
        from dclab_rnd.storage.migrate import migrate

        source = Path(tempfile.mkdtemp()) / "files"
        (source / "projects").mkdir(parents=True)  # a file workspace
        files = FileTeams(source)
        sales = files.create_team("Sales", "sales")
        files.create_team("Marketing", "marketing")
        files.add_member(sales["id"], "owner")
        target = Path(tempfile.mkdtemp()) / "db"
        first = migrate(source, target, self.url)
        self.assertEqual(first["logs"]["teams"], {"source": 2, "copied": 2, "already": 0, "memberships_not_moved": 1})
        moved = open_teams(target)
        self.assertEqual({(t["id"], t["name"], t["kind"]) for t in moved.teams()},
                         {(t["id"], t["name"], t["kind"]) for t in files.teams()})
        self.assertEqual(migrate(source, target, self.url)["logs"]["teams"]["copied"], 0)  # a second run copies nothing


class TeamRoutesTests(unittest.TestCase):
    """Over HTTP with password sign-in: two workspaces, an owner, an ML engineer and a viewer in the first."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            raise unittest.SkipTest(f"Studio dependencies not installed: {error}")
        url = pgtest.require()
        pgtest.empty(url)
        cls.env = mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": url, "DCLAB_AUTH": "password", "OPENAI_API_KEY": ""})
        cls.env.start()
        from dclab_rnd.accounts.store import Accounts

        cls.TestClient = TestClient
        cls.app = create_app(Path(tempfile.mkdtemp()))
        cls.main = TestClient(cls.app)
        cls.main.__enter__()
        cls.accounts = Accounts()
        cls.w1 = cls.app.state.services.workspace_id
        cls.w2 = cls.accounts.create_workspace("Other company")
        cls.users = {}
        for name, workspace, role in (("owner1", cls.w1, "owner"), ("ds1", cls.w1, "data_scientist"), ("view1", cls.w1, "viewer"),
                                      ("owner2", cls.w2, "owner")):
            user = cls.accounts.create_user(f"{name}-{os.getpid()}@example.com", name.title(), password=PASSWORD)
            cls.accounts.set_member(workspace, user["id"], role)
            cls.users[name] = user
        cls.clients = {name: cls.sign_in(name) for name in cls.users}

    @classmethod
    def tearDownClass(cls):
        cls.main.__exit__(None, None, None)
        cls.env.stop()

    @classmethod
    def sign_in(cls, name):
        client = cls.TestClient(cls.app)
        anonymous = client.get("/api/config").json()["csrf"]
        r = client.post("/api/auth/password", json={"email": cls.users[name]["email"], "password": PASSWORD}, headers={"X-DCLab-Token": anonymous})
        assert r.status_code == 200, r.text
        client.headers["X-DCLab-Token"] = r.json()["csrf"]
        return client

    def test_the_owner_manages_teams_and_others_are_refused(self):
        owner, scientist, viewer, stranger = (self.clients[n] for n in ("owner1", "ds1", "view1", "owner2"))
        ds1, view1, owner2 = (self.users[n]["id"] for n in ("ds1", "view1", "owner2"))
        for client in (viewer, scientist):
            refused = client.post("/api/workspace/teams", json={"name": "Business", "kind": "business"})
            self.assertEqual(refused.status_code, 403, refused.text)
            self.assertEqual(refused.json()["detail"]["error"], "role")
        business = owner.post("/api/workspace/teams", json={"name": "Business", "kind": "business"})
        self.assertEqual(business.status_code, 201, business.text)
        science = owner.post("/api/workspace/teams", json={"name": "Data Science", "kind": "data_science"}).json()
        b = business.json()["id"]
        self.assertEqual(viewer.post(f"/api/workspace/teams/{b}/members", json={"user_id": view1}).status_code, 403)
        for team in (b, science["id"]):  # the ML engineer in two teams
            placed = owner.post(f"/api/workspace/teams/{team}/members", json={"user_id": ds1})
            self.assertEqual((placed.status_code, placed.json()["added"]), (201, True), placed.text)
        self.assertFalse(owner.post(f"/api/workspace/teams/{b}/members", json={"user_id": ds1}).json()["added"])
        self.assertEqual(owner.post(f"/api/workspace/teams/{b}/members", json={"user_id": view1}).status_code, 201)
        self.assertEqual(owner.post(f"/api/workspace/teams/{b}/members", json={"user_id": owner2}).status_code, 404)  # not of this workspace
        self.assertEqual(owner.post(f"/api/workspace/teams/{b}/members", json={"user_id": "u_nobody"}).status_code, 404)
        self.assertEqual({t["name"] for t in viewer.get("/api/workspace/teams", params={"user_id": ds1}).json()}, {"Business", "Data Science"})
        self.assertEqual({m["user_id"] for m in next(t for t in viewer.get("/api/workspace/teams").json() if t["id"] == b)["members"]}, {ds1, view1})
        # the other workspace's owner: sees none of it, and its ids change nothing
        self.assertEqual(stranger.get("/api/workspace/teams").json(), [])
        self.assertEqual(stranger.post(f"/api/workspace/teams/{b}/members", json={"user_id": owner2}).status_code, 404)
        self.assertEqual(stranger.delete(f"/api/workspace/teams/{b}/members/{ds1}").status_code, 404)
        # removal
        self.assertEqual(scientist.delete(f"/api/workspace/teams/{b}/members/{ds1}").status_code, 403)
        self.assertEqual(owner.delete(f"/api/workspace/teams/{b}/members/{ds1}").status_code, 204)
        self.assertEqual(owner.delete(f"/api/workspace/teams/{b}/members/{ds1}").status_code, 404)
        self.assertEqual([t["name"] for t in owner.get("/api/workspace/teams", params={"user_id": ds1}).json()], ["Data Science"])
        # leaving the workspace over HTTP leaves its teams
        leaver = self.accounts.create_user(f"leaver-{os.getpid()}@example.com", "Leaver", password=PASSWORD)["id"]
        self.accounts.set_member(self.w1, leaver, "viewer")
        self.assertEqual(owner.post(f"/api/workspace/teams/{b}/members", json={"user_id": leaver}).status_code, 201)
        self.assertEqual(owner.delete(f"/api/workspace/members/{leaver}").status_code, 204)
        self.assertEqual(owner.get("/api/workspace/teams", params={"user_id": leaver}).json(), [])
        self.assertNotIn(leaver, {m["user_id"] for t in owner.get("/api/workspace/teams").json() for m in t["members"]})
        moves = [e.get("move") for e in owner.get("/api/platform/audit").json()["items"]]
        for move in ("team_create", "team_add_member", "team_remove_member"):
            self.assertIn(move, moves)


if __name__ == "__main__":
    unittest.main()
