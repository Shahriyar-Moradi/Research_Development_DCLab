"""Accounts, workspaces and roles (package 10.2), with password sign-in on PostgreSQL.

Two workspaces and a member of each role. Every write route is called by someone whose role may not, and by a member
of the other workspace with this workspace's ids: neither changes anything. A viewer cannot run a stage or approve a
gate, a user never sees another workspace, and a gate approval names the person who approved it."""
import json
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pgtest  # noqa: E402

PASSWORD = "a long enough password"
WRITES = {"POST", "PUT", "PATCH", "DELETE"}


class AccountsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            raise unittest.SkipTest(f"Studio dependencies not installed: {error}")
        url = pgtest.require()
        pgtest.empty(url)  # this file's workspaces only: a second run starts from nothing
        cls.env = mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": url, "DCLAB_AUTH": "password", "OPENAI_API_KEY": "",
                                               "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"})
        cls.env.start()
        from dclab_rnd.storage import db

        db.upgrade(url)
        from dclab_rnd.accounts.store import Accounts

        cls.TestClient = TestClient
        cls.app = create_app(Path(tempfile.mkdtemp()))
        cls.main = TestClient(cls.app)
        cls.main.__enter__()
        cls.accounts = Accounts()
        cls.w1 = cls.app.state.services.workspace_id
        cls.w2 = cls.accounts.create_workspace("Other team")
        cls.users = {}
        for name, workspace, role in (("owner1", cls.w1, "owner"), ("ds1", cls.w1, "data_scientist"), ("rev1", cls.w1, "reviewer"),
                                      ("view1", cls.w1, "viewer"), ("owner2", cls.w2, "owner"), ("ds2", cls.w2, "data_scientist")):
            user = cls.accounts.create_user(f"{name}-{os.getpid()}@example.com", name.title(), password=PASSWORD)
            cls.accounts.set_member(workspace, user["id"], role)
            cls.users[name] = user
        cls.clients = {name: cls.sign_in(name) for name in cls.users}
        # the first workspace's things, made by its owner
        s1 = cls.app.state.services
        with cls.as_user("owner1"):
            cls.project = s1.projects.create("Churn W1", "general", "Predict churn")
            cls.draft = s1.drafts.create("Predict which customers churn next month")
            cls.session = s1.intern_sessions.create("Review the churn project", mode="standard", model=None, project_id=cls.project["id"])
        cls.job = s1.job_store.enqueue("stage", "w1-key", {"project_id": cls.project["id"], "stages": ["data"]}, worker="w")
        s1.job_store.finish(cls.job["id"], "failed", "boom")
        cls.lesson = s1.lesson_store.save({"id": "lw1abc", "project_id": cls.project["id"], "status": "proposed", "claim": "c", "against": "a",
                                           "next_test": "n", "cites": [], "scope": {}, "synthetic": False, "proposed_at": "2026-10-07T10:00:00+00:00"})
        s1.store.create("runw1", {"goal": "g", "datasets": []})
        token, record = cls.accounts.create_token(cls.users["owner1"]["id"], cls.w1, "laptop")
        cls.token, cls.token_record = token, record

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

    @classmethod
    def as_user(cls, name):
        from dclab_rnd.accounts.principal import Principal, acting

        user = cls.users[name]
        return acting(Principal(role="owner", workspace_id=cls.w1, user_id=user["id"], email=user["email"], name=user["name"], via="test"))

    def ids(self):
        return {"project_id": self.project["id"], "stage": "data", "note_id": "n1", "draft_id": self.draft["id"], "session_id": self.session["id"],
                "job_id": self.job["id"], "lesson_id": self.lesson["id"], "run_id": "runw1", "token_id": self.token_record["id"],
                "user_id": self.users["ds1"]["id"]}

    def snapshot(self):
        s1 = self.app.state.services
        return (json.dumps(s1.projects.get(self.project["id"]), sort_keys=True, default=str),
                json.dumps({k: v for k, v in s1.drafts.get(self.draft["id"]).items() if k != "updated"}, sort_keys=True, default=str),
                json.dumps(s1.intern_sessions.get(self.session["id"]), sort_keys=True, default=str),
                s1.job_store.get(self.job["id"])["status"], s1.lesson_store.get(self.lesson["id"])["status"],
                sorted((m["id"], m["role"]) for m in self.accounts.members(self.w1)), len(self.accounts.tokens(self.users["owner1"]["id"])))

    # ------------------------------------------------------------------ the package's "Done when"
    def test_every_write_route_refuses_the_wrong_role_and_the_wrong_workspace(self):
        from dclab_rnd.accounts.guard import permission
        from dclab_rnd.agentic.api_models import api_routes

        wrong_role = {"write": "view1", "approve_stage": "view1", "approve_gate": "ds1", "review": "ds1", "admin": "ds1"}
        before, ids, checked = self.snapshot(), self.ids(), []
        for route in api_routes(self.app):
            for method in sorted(route.methods & WRITES):
                needed = permission(method, route.path)
                path = re.sub(r"\{(\w+)\}", lambda m: ids.get(m.group(1), "x"), route.path)
                if needed in wrong_role:  # someone in this workspace whose role may not
                    r = self.clients[wrong_role[needed]].request(method, path, json={})
                    self.assertEqual(r.status_code, 403, f"{method} {route.path} as {wrong_role[needed]}: {r.status_code} {r.text[:200]}")
                    self.assertEqual(r.json()["detail"]["error"], "role", route.path)
                if "{" in route.path and needed != "public":  # a member of the other workspace, with this workspace's ids
                    r = self.clients["owner2"].request(method, path, json={})
                    self.assertNotEqual(r.status_code // 100, 2, f"{method} {route.path} from the other workspace: {r.status_code} {r.text[:200]}")
                checked.append(f"{method} {route.path}")
        self.assertEqual(self.snapshot(), before)  # nothing in the first workspace changed
        self.assertGreater(len(checked), 45)

    def test_a_viewer_cannot_run_a_stage_or_approve_a_gate(self):
        pid = self.project["id"]
        viewer, reviewer, scientist = self.clients["view1"], self.clients["rev1"], self.clients["ds1"]
        self.assertEqual(viewer.post(f"/api/projects/{pid}/stages/data/run", json={}).status_code, 403)
        self.assertEqual(viewer.post(f"/api/projects/{pid}/run").status_code, 403)
        self.assertEqual(viewer.post(f"/api/projects/{pid}/approvals", json={"gate": "holdout", "reason": "ok"}).status_code, 403)
        self.assertEqual(scientist.post(f"/api/projects/{pid}/approvals", json={"gate": "holdout", "reason": "ok"}).status_code, 403)
        self.assertEqual(viewer.get(f"/api/projects/{pid}").status_code, 200)  # a viewer reads
        self.assertNotEqual(reviewer.post(f"/api/projects/{pid}/stages/data/approve", json={}).status_code, 403)  # the stage's choice: reviewer too
        # gate switches and the training opt-in are the owner's
        self.assertEqual(scientist.patch(f"/api/projects/{pid}", json={"policy": {"require_holdout_approval": True}}).status_code, 403)
        self.assertEqual(scientist.patch(f"/api/projects/{pid}", json={"name": "Churn W1"}).status_code, 200)

    def test_a_user_never_sees_another_workspace(self):
        other = self.clients["ds2"]
        created = other.post("/api/projects", json={"name": "W2 project", "goal": "g"})
        self.assertEqual(created.status_code, 201, created.text)
        mine = {p["id"] for p in other.get("/api/projects").json()}
        theirs = {p["id"] for p in self.clients["owner1"].get("/api/projects").json()}
        self.assertIn(created.json()["id"], mine)
        self.assertNotIn(self.project["id"], mine)
        self.assertNotIn(created.json()["id"], theirs)
        for path in (f"/api/projects/{self.project['id']}", f"/api/drafts/{self.draft['id']}", f"/api/intern/sessions/{self.session['id']}",
                     f"/api/jobs/{self.job['id']}"):
            self.assertEqual(other.get(path).status_code, 404, path)
        self.assertNotIn(self.draft["id"], {d["id"] for d in other.get("/api/drafts").json()})
        audit = other.get("/api/platform/audit").json()
        self.assertFalse(any((e.get("project") or {}).get("id") == self.project["id"] for e in audit["items"]))

    def test_a_gate_approval_names_the_person(self):
        s1 = self.app.state.services
        p = s1.projects.get(self.project["id"])
        p["solution"] = {"target": "y", "task": "binary"}
        s1.projects.save(p)
        r = self.clients["rev1"].post(f"/api/projects/{self.project['id']}/approvals", json={"gate": "holdout", "reason": "split checked"})
        self.assertEqual(r.status_code, 200, r.text)
        approval = s1.projects.get(self.project["id"])["approvals"][-1]
        self.assertEqual((approval["user_id"], approval["by"], approval["role"]), (self.users["rev1"]["id"], "Rev1", "reviewer"))
        row = next(e for e in self.clients["owner1"].get("/api/platform/audit?kind=approval").json()["items"] if e["at"] == approval["at"])
        self.assertEqual((row["user_id"], row["who"]), (self.users["rev1"]["id"], "Rev1"))

    def test_pages_load_and_the_schema_is_not_served(self):
        anonymous = self.TestClient(self.app)
        for path in ("/", "/guide"):
            self.assertEqual(anonymous.get(path).status_code, 200, path)  # the shell: no data; the sign-in page lives here (part C)
            self.assertEqual(self.clients["view1"].get(path).status_code, 200, path)
        self.assertEqual(anonymous.get("/openapi.json").status_code, 404)  # FastAPI's own routes are outside the role check
        self.assertEqual(anonymous.get("/docs").status_code, 404)
        self.assertEqual(anonymous.get("/healthz").status_code, 200)  # a load balancer's probes carry no session (12.4)
        self.assertEqual(anonymous.get("/readyz").json()["checks"]["database"], "ok")

    def test_the_home_agent_and_the_jobs_of_a_second_workspace_run_there(self):
        other = self.clients["owner2"]
        draft = other.post("/api/drafts", json={"problem": "Predict which invoices are paid late"}).json()
        deadline = time.time() + 120  # a CI runner running six test files at once is slow
        while time.time() < deadline and not other.get(f"/api/drafts/{draft['id']}").json()["messages"]:
            time.sleep(0.2)
        state = other.get(f"/api/drafts/{draft['id']}").json()
        self.assertTrue(state["messages"], {k: state.get(k) for k in ("status", "agent", "events_seq")})  # the agent answered, in this workspace
        self.assertEqual(other.post(f"/api/drafts/{draft['id']}/pack", json={"key": "auto"}).status_code, 200)
        csv = b"days,amount,late\n" + b"".join(f"{i % 60},{100 + i},{i % 4 == 0}\n".encode() for i in range(120))
        asset = other.put(f"/api/drafts/{draft['id']}/data?filename=i.csv", content=csv).json()
        while time.time() < deadline + 120:
            found = next(a for a in other.get(f"/api/drafts/{draft['id']}").json()["assets"] if a["id"] == asset["id"])
            if found["status"] in ("ready", "failed"):
                break
            time.sleep(0.2)
        self.assertEqual(found["status"], "ready")  # its pipeline job ran on this workspace's worker
        self.assertNotIn(draft["id"], {d["id"] for d in self.clients["owner1"].get("/api/drafts").json()})

    # ------------------------------------------------------------------ sessions, tokens, members
    def test_signed_out_requests_are_refused_and_sessions_end(self):
        anonymous = self.TestClient(self.app)
        self.assertEqual(anonymous.get("/api/projects").status_code, 401)
        token = anonymous.get("/api/config").json()["csrf"]
        self.assertEqual(anonymous.post("/api/projects", json={"name": "x"}, headers={"X-DCLab-Token": token}).status_code, 401)
        self.assertEqual(anonymous.post("/api/auth/password", json={"email": self.users["ds1"]["email"], "password": "wrong password!"},
                                        headers={"X-DCLab-Token": token}).status_code, 401)
        client = self.sign_in("ds1")
        self.assertTrue(client.get("/api/auth/me").json()["signed_in"])
        self.assertEqual(client.post("/api/projects", json={"name": "x"}, headers={"X-DCLab-Token": "not the session's"}).status_code, 403)
        client.post("/api/auth/signout")
        self.assertEqual(client.get("/api/projects").status_code, 401)

    def test_an_api_token_acts_as_its_user_without_a_cookie(self):
        bare = self.TestClient(self.app)
        token, record = self.accounts.create_token(self.users["ds1"]["id"], self.w1, "script")
        headers = {"Authorization": f"Bearer {token}"}
        self.assertEqual(bare.get("/api/projects", headers=headers).status_code, 200)
        created = bare.post("/api/projects", json={"name": "From a script", "goal": "g"}, headers=headers)  # no CSRF: not a cookie
        self.assertEqual(created.status_code, 201, created.text)
        self.assertEqual(bare.post(f"/api/projects/{self.project['id']}/approvals", json={"gate": "holdout"}, headers=headers).status_code, 403)
        self.accounts.revoke_token(self.users["ds1"]["id"], record["id"])
        self.assertEqual(bare.get("/api/projects", headers=headers).status_code, 401)

    def test_mcp_acts_in_the_tokens_workspace(self):
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.skipTest("mcp not installed")
        token, _ = self.accounts.create_token(self.users["ds2"]["id"], self.w2, "chat ui")
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        bare = self.TestClient(self.app)
        call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "create_project", "arguments": {"name": "via mcp in W2", "goal": "g"}}}
        self.assertEqual(bare.post("/mcp", json=call, headers=headers).status_code, 401)  # an MCP client sends a token
        made = bare.post("/mcp", json=call, headers={**headers, "Authorization": f"Bearer {token}"})
        self.assertEqual(made.status_code, 200, made.text[:300])
        project_id = json.loads(made.json()["result"]["content"][0]["text"])["project_id"]
        self.assertEqual(self.clients["ds2"].get(f"/api/projects/{project_id}").status_code, 200)  # in the token's workspace
        self.assertEqual(self.clients["owner1"].get(f"/api/projects/{project_id}").status_code, 404)  # and not in the server's own
        viewer, _ = self.accounts.create_token(self.users["view1"]["id"], self.w1, "viewer's")
        self.assertEqual(bare.post("/mcp", json=call, headers={**headers, "Authorization": f"Bearer {viewer}"}).status_code, 403)

    def test_the_owner_manages_members_and_a_workspace_keeps_an_owner(self):
        owner, scientist = self.clients["owner1"], self.clients["ds1"]
        self.assertEqual(scientist.post("/api/workspace/members", json={"email": "new@example.com", "role": "viewer"}).status_code, 403)
        added = owner.post("/api/workspace/members", json={"email": f"new-{os.getpid()}@example.com", "role": "viewer", "password": PASSWORD})
        self.assertEqual(added.status_code, 201, added.text)
        self.assertEqual(owner.patch(f"/api/workspace/members/{added.json()['id']}", json={"role": "reviewer"}).json()["role_label"], "Reviewer")
        self.assertEqual(owner.patch(f"/api/workspace/members/{self.users['owner1']['id']}", json={"role": "viewer"}).status_code, 409)
        self.assertEqual(owner.delete(f"/api/workspace/members/{added.json()['id']}").status_code, 204)
        labels = {m["role_label"] for m in owner.get("/api/workspace/members").json()}
        self.assertTrue(labels <= {"Owner", "ML engineer", "Reviewer", "Business viewer"})  # the Admin page's names

    def test_a_member_of_two_workspaces_switches_between_them(self):
        user = self.accounts.create_user(f"both-{os.getpid()}@example.com", "Both", password=PASSWORD)
        self.accounts.set_member(self.w1, user["id"], "viewer")
        self.accounts.set_member(self.w2, user["id"], "owner")
        self.users["both"] = user
        client = self.sign_in("both")
        me = client.get("/api/auth/me").json()
        self.assertEqual({w["id"] for w in me["workspaces"]}, {self.w1, self.w2})
        target = self.w2 if me["workspace_id"] == self.w1 else self.w1
        self.assertEqual(client.post("/api/auth/workspace", json={"workspace_id": target}).status_code, 200)
        self.assertEqual(client.get("/api/auth/me").json()["workspace_id"], target)
        self.assertEqual(client.post("/api/auth/workspace", json={"workspace_id": "wnotmine"}).status_code, 404)


class PoolTests(unittest.TestCase):
    def test_a_database_away_or_a_broken_workspace_stops_nothing(self):
        from types import SimpleNamespace

        from dclab_rnd.agentic.pool import Pool

        started = []
        fake = SimpleNamespace(default=SimpleNamespace(worker=SimpleNamespace(start=lambda: started.append("own"))),
                               settings=SimpleNamespace(auth="password"))
        fake.workspace_ids = mock.Mock(side_effect=OSError("the database is away"))
        Pool.start_all(fake)  # reported, not raised: the worker process and the server go on
        fake.workspace_ids = lambda: ["own", "w-broken", "w-good"]
        fake.get = lambda w: started.append(w) if w != "w-broken" else (_ for _ in ()).throw(RuntimeError("folder names another workspace"))
        Pool.start_all(fake)
        self.assertEqual(started, ["own", "own", "w-good"])  # the workspace after the broken one still starts

    def test_a_route_with_several_methods_needs_what_each_needs(self):
        import asyncio

        from fastapi import HTTPException

        from dclab_rnd.accounts.guard import require
        from dclab_rnd.accounts.principal import Principal, acting

        check = require("approve_gate", "write")
        with acting(Principal(role="reviewer", user_id="u1")), self.assertRaises(HTTPException):
            asyncio.run(check())  # a reviewer approves gates, but may not write
        with acting(Principal(role="owner", user_id="u1")):
            asyncio.run(check())


class CommandLineTests(unittest.TestCase):
    def test_the_first_owner_is_added_from_the_command_line(self):
        from dclab_rnd.accounts.__main__ import main
        from dclab_rnd.accounts.store import Accounts

        url = pgtest.require()
        email = f"cli-{os.getpid()}-{int(time.time())}@example.com"
        with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": url, "DCLAB_NEW_PASSWORD": PASSWORD, "DCLAB_AGENT_HOME": tempfile.mkdtemp()}):
            from dclab_rnd.storage import db

            db.upgrade(url)
            self.assertEqual(main(["add", "--email", email, "--role", "owner", "--name", "First"]), 0)
            self.assertIsNotNone(Accounts().check_password(email, PASSWORD))
            self.assertIsNone(Accounts().check_password(email, "another long password"))


if __name__ == "__main__":
    unittest.main()
