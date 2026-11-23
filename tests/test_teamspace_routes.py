"""W3 and W5 together through the routes (the joint review of 2026-11-20).

Two signed-in people in their own team threads (Sara in Business, Dan in Data Science) finish the renewal case, steps 1
to 4, through the routes alone: Sara's question opens the case, the agents carry it, Sara's answer reaches Data Science,
and the answer comes back to her thread with the four things a Business member needs (what was found, why, who acts
next, what changed). Also: a reader demoted while a stream is open stops receiving it; a deleted project's threads are
kept and unreadable; a reply goes to the real team's thread once one exists; and ``storage migrate`` moves the team work
and threads of a file workspace into PostgreSQL once. PostgreSQL through ``tests/run_parallel.py --pg``.
"""

import os
import re
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402

from dclab_rnd.memory.scope import create_problem  # noqa: E402
from dclab_rnd.storage import open_teams  # noqa: E402
from dclab_rnd.teamspace import agents, delivery, history_lines  # noqa: E402
from dclab_rnd.teamspace import threads as th  # noqa: E402

PASSWORD = "a long enough password"
BRIEF = ("Customers are leaving. Who should Sales call before renewal? Requested: can the data rank accounts by churn risk "
         "seven days before renewal?")
ANSWER = "Failure to renew by the contractual renewal date."
FEASIBLE = "The sample has renewal dates and mature outcomes. We can score seven days before renewal."


def ids_in(body: str) -> list[int]:
    return [int(n) for n in re.findall(r"^id: (\d+)$", body, flags=re.M)]


def settle(services, limit=60):
    """Deliver everything pending, lane by lane, as the worker's jobs would."""
    work = delivery.store_for(services)
    for _ in range(limit):
        heads = work.pending_heads()
        if not heads:
            return
        head = heads[0]
        for row in delivery.run(services, {"project_id": head["project_id"], "task_id": delivery.lane_of(work, head)}):
            assert row["status"] == "delivered", row
    raise AssertionError("the deliveries never settled")


def settled_by_the_worker(services, seconds=30.0):
    """The app's own worker delivers what the routes queued: wait until nothing is pending."""
    work = delivery.store_for(services)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not work.deliveries(status="pending"):
            assert not [r for r in work.deliveries() if r["status"] != "delivered"], work.deliveries()
            return
        delivery.requeue_undelivered(services)
        time.sleep(0.1)
    raise AssertionError("the worker never delivered everything")


class SignedInTeamsTests(unittest.TestCase):
    """Password sign-in on PostgreSQL: Sara in Business, Dan in Data Science, and an owner."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            raise unittest.SkipTest(f"Studio dependencies not installed: {error}")
        url = pgtest.require()
        pgtest.empty(url)
        cls.patch = mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": url, "DCLAB_AUTH": "password", "OPENAI_API_KEY": "",
                                                "DCLAB_TEAM_THREAD_RECHECK_SECONDS": "0.2", "DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS": "0"})
        cls.patch.start()
        from dclab_rnd.accounts.store import Accounts

        cls.TestClient = TestClient
        cls.app = create_app(Path(tempfile.mkdtemp()))
        cls.main = TestClient(cls.app)
        cls.main.__enter__()
        cls.accounts = Accounts()
        s = cls.app.state.services
        cls.services, cls.w1 = s, s.workspace_id
        cls.users = {}
        for name, role in (("sara", "data_scientist"), ("dan", "data_scientist"), ("boss", "owner")):
            user = cls.accounts.create_user(f"{name}-{os.getpid()}-{os.urandom(3).hex()}@example.com", name.title(), password=PASSWORD)
            cls.accounts.set_member(cls.w1, user["id"], role)
            cls.users[name] = user
        teams = open_teams(s.store.home)
        cls.business = teams.create_team("Business", "business")["id"]
        cls.science = teams.create_team("Data Science", "data_science")["id"]
        teams.add_member(cls.business, cls.users["sara"]["id"])
        teams.add_member(cls.science, cls.users["dan"]["id"])
        cls.clients = {name: cls.sign_in(name) for name in cls.users}
        delivery.set_target(None)
        delivery.set_membership_checker(None)

    @classmethod
    def tearDownClass(cls):
        cls.main.__exit__(None, None, None)
        cls.patch.stop()

    @classmethod
    def sign_in(cls, name):
        client = cls.TestClient(cls.app)
        anonymous = client.get("/api/config").json()["csrf"]
        r = client.post("/api/auth/password", json={"email": cls.users[name]["email"], "password": PASSWORD}, headers={"X-DCLab-Token": anonymous})
        assert r.status_code == 200, r.text
        client.headers["X-DCLab-Token"] = r.json()["csrf"]
        return client

    def project(self, name):
        s = self.services
        pid = s.projects.create(name, goal="Who should Sales call")["id"]
        s.projects.update(pid, team_inputs=["sample:renewal_v1"])  # the invented sample the case may read
        return pid, create_problem(s.projects, pid, name, workspace_id=self.w1)["problem_id"]

    def open(self, who, team, pid, prob):
        r = self.clients[who].post(f"/api/projects/{pid}/problems/{prob}/team-threads", json={"team": team})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["id"]

    def read(self, who, pid, thread_id):
        return self.clients[who].get(f"/api/projects/{pid}/team-threads/{thread_id}")

    def test_a_sales_member_answers_its_agent_in_the_sales_thread_once(self):
        """T3: Sara asks Sales; the Sales agent asks its own team in the Sales thread; Rob answers there with the reply
        button's task id and a response id; the same answer sent twice is one message and one record; Sara gets one reply."""
        pid, prob = self.project("Renewal calls")
        rob = self.accounts.create_user(f"rob-{os.getpid()}-{os.urandom(3).hex()}@example.com", "Rob", password=PASSWORD)
        self.accounts.set_member(self.w1, rob["id"], "data_scientist")
        teams = open_teams(self.services.store.home)
        sales = teams.create_team(f"Sales {os.urandom(2).hex()}", "sales")["id"]
        teams.add_member(sales, rob["id"])
        self.users["rob"] = rob
        self.clients["rob"] = self.sign_in("rob")
        sara_thread = self.open("sara", self.business, pid, prob)
        base = f"/api/projects/{pid}/team-threads"
        asked = self.clients["sara"].post(f"{base}/{sara_thread}/messages", json={"text": "Who should Sales call this week?", "ask": "sales"})
        self.assertEqual(asked.status_code, 201, asked.text)
        settled_by_the_worker(self.services)
        sales_thread = self.open("rob", sales, pid, prob)
        question = self.read("rob", pid, sales_thread).json()["messages"][-1]
        self.assertEqual((question["author"]["id"], question["handoff"]["kind"]), ("sales", "clarification"))
        task_id = question["handoff"]["task_id"]
        body = {"text": "18", "task_id": task_id, "response_id": "rob-answer-1"}
        first = self.clients["rob"].post(f"{base}/{sales_thread}/messages", json=body)
        again = self.clients["rob"].post(f"{base}/{sales_thread}/messages", json={**body, "text": "25"})  # a resend: the saved words count
        self.assertEqual((first.status_code, again.status_code), (201, 201))
        self.assertEqual(first.json()["id"], again.json()["id"])
        settled_by_the_worker(self.services)
        view = delivery.store_for(self.services).task_view(pid, task_id)
        self.assertEqual(view["task"].status, "complete")
        [ask] = [a for a in view["actions"] if a.action == "ask_capacity"]
        self.assertEqual(ask.responded_by.id, rob["id"])
        replies = [m for m in self.read("sara", pid, sara_thread).json()["messages"] if (m.get("handoff") or {}).get("kind") == "reply"]
        self.assertEqual(len(replies), 1)
        self.assertIn("18 calls", replies[0]["text"])
        self.assertEqual(self.read("sara", pid, sales_thread).status_code, 403)  # the Sales thread stays the Sales team's

    def test_sara_and_dan_finish_the_renewal_round_trip_through_their_own_threads(self):
        pid, prob = self.project("Renewal retention")
        sara_thread, dan_thread = self.open("sara", self.business, pid, prob), self.open("dan", self.science, pid, prob)
        base = f"/api/projects/{pid}/team-threads"
        # 1-2: Sara asks in her thread; her agent opens the case for Data Science
        posted = self.clients["sara"].post(f"{base}/{sara_thread}/messages", json={"text": BRIEF, "ask": "data_science"})
        self.assertEqual(posted.status_code, 201, posted.text)
        settled_by_the_worker(self.services)
        # 3: Data Science asks one question; it reaches Sara's thread, and Dan's thread keeps it
        mine = self.read("sara", pid, sara_thread).json()["messages"]
        self.assertEqual([(m["author"]["kind"], (m.get("handoff") or {}).get("kind")) for m in mine], [("person", None), ("agent", "clarification")])
        self.assertEqual(mine[-1]["text"], agents.CLARIFY_TARGET)
        dans = self.read("dan", pid, dan_thread).json()["messages"]
        self.assertEqual([m["handoff"]["kind"] for m in dans], ["task", "clarification"])
        self.assertEqual((dans[0]["text"], dans[1]["text"]), (BRIEF, agents.CLARIFY_TARGET))
        # Sara answers in her own thread: the route hands it to her agent, which carries it to Data Science
        self.assertEqual(self.clients["sara"].post(f"{base}/{sara_thread}/messages", json={"text": ANSWER}).status_code, 201)
        settled_by_the_worker(self.services)
        # 4: the answer and its report come back to Sara's thread, once
        mine = self.read("sara", pid, sara_thread).json()["messages"]
        replies = [m for m in mine if (m.get("handoff") or {}).get("kind") == "reply"]
        self.assertEqual(len(replies), 1)
        found, why, nxt, changed = replies[0]["text"].split("\n")
        self.assertEqual(found, f"Simulated (fixture sample:renewal_v1): {FEASIBLE}")  # what was found (on an invented sample)
        self.assertTrue(why.startswith("Why: schema:sample:renewal_v1"), why)  # why: the evidence and its limit
        self.assertEqual(nxt, "Next: closed, nothing is owed.")  # who acts next
        self.assertTrue(changed.startswith("Changed: first report (") and replies[0]["handoff"]["item_id"] in changed)  # what changed
        dans = self.read("dan", pid, dan_thread).json()["messages"]
        self.assertEqual([m["handoff"]["kind"] for m in dans], ["task", "clarification", "answer"])
        self.assertEqual(dans[-1]["text"], ANSWER)
        # each sees only their own team's thread; every project reader gets the shared history
        self.assertEqual(self.read("sara", pid, dan_thread).status_code, 403)
        self.assertEqual(self.read("dan", pid, sara_thread).status_code, 403)
        lines = self.clients["dan"].get(f"/api/projects/{pid}/problems/{prob}/handoffs").json()["lines"]
        self.assertEqual(lines, [f"1. Business → Data Science · {BRIEF}", f"2. Data Science → Business · {agents.CLARIFY_TARGET}",
                                 f"3. Business → Data Science · {ANSWER}",
                                 f"4. Data Science → Business · Simulated (fixture sample:renewal_v1): {FEASIBLE}"])
        # the people behind the answers are the signed-in ones, never written by an agent
        work = delivery.store_for(self.services)
        [task] = [t for t in work.open_tasks("data_science", pid)] or [work.task(pid, replies[0]["handoff"]["task_id"])]
        humans = [a for a in work.task_view(pid, task.id)["actions"] if a.source.startswith("human:")]
        self.assertEqual({a.responded_by.id for a in humans}, {self.users["sara"]["id"]})
        # Dan cannot answer for Sara: a message from him in his thread is only his message
        self.assertEqual(self.clients["dan"].post(f"{base}/{dan_thread}/messages", json={"text": "Cancellation."}).status_code, 201)
        settled_by_the_worker(self.services)
        self.assertEqual(len(work.task_view(pid, task.id)["reports"]), 1)

    def test_an_owner_demoted_while_reading_a_stream_stops_receiving_it(self):
        pid, prob = self.project("Stream")
        sara_thread = self.open("sara", self.business, pid, prob)
        store = th.open_threads(self.services.store.home)
        boss = self.users["boss"]
        agent = {"kind": "agent", "id": "data_science", "team": "data_science"}

        def later():
            time.sleep(0.5)
            store.post(sara_thread, agent, "seen while an owner")
            time.sleep(0.4)
            self.accounts.set_member(self.w1, boss["id"], "viewer")  # demoted: a viewer in no team reads nothing here
            time.sleep(0.8)  # past the recheck
            store.post(sara_thread, agent, "after the demotion")
        worker = threading.Thread(target=later)
        worker.start()
        try:
            body = self.clients["boss"].get(f"/api/projects/{pid}/team-threads/{sara_thread}/events", params={"wait": 4}).text
        finally:
            worker.join()
            self.accounts.set_member(self.w1, boss["id"], "owner")
        self.assertIn("seen while an owner", body)
        self.assertNotIn("after the demotion", body)


class _Local:
    env: dict = {}

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"DCLAB_AUTH": "", "OPENAI_API_KEY": "", "DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS": "0", **self.env})
        patcher.start()
        self.addCleanup(patcher.stop)
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        self.app = create_app(Path(tempfile.mkdtemp()) / "workspace")
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.headers["X-DCLab-Token"] = self.client.get("/api/config").json()["csrf"]
        self.services = self.app.state.services
        self.ws = th.open_threads(self.services.store.home).workspace
        self.pid = self.services.projects.create("Renewal", goal="Who should Sales call")["id"]
        self.prob = create_problem(self.services.projects, self.pid, "Renewal", workspace_id=self.ws)["problem_id"]

    def test_a_deleted_projects_threads_are_kept_and_unreadable(self):
        thread = self.client.post(f"/api/projects/{self.pid}/problems/{self.prob}/team-threads", json={"team": "business"}).json()
        path = f"/api/projects/{self.pid}/team-threads/{thread['id']}"
        self.assertEqual(self.client.post(f"{path}/messages", json={"text": "our notes"}).status_code, 201)
        self.assertEqual(self.client.delete(f"/api/projects/{self.pid}").status_code, 204)
        for answer in (self.client.get(path), self.client.get(f"{path}/events", params={"wait": 0}),
                       self.client.post(f"{path}/messages", json={"text": "still here?"}),
                       self.client.get(f"/api/projects/{self.pid}/problems/{self.prob}/handoffs")):
            self.assertIn(answer.status_code, (403, 404), answer.text)
            self.assertNotIn("our notes", answer.text)
        self.assertEqual([t["id"] for t in self.client.get("/api/team-threads").json()], [])
        kept = th.open_threads(self.services.store.home)  # the store keeps it (only the owner's audited forget removes)
        self.assertEqual([m["text"] for m in kept.messages(thread["id"])], ["our notes"])

    def test_a_reply_goes_to_the_real_teams_thread_once_one_exists(self):
        store = th.open_threads(self.services.store.home)
        old, _ = store.open_thread(self.pid, self.prob, "business", label="Business")  # before any Business team existed
        team = open_teams(self.services.store.home).create_team("Business", "business")["id"]
        row = {"id": "d-quirk", "kind": "reply", "project_id": self.pid, "problem_id": self.prob, "task_id": "T", "item_id": "T@r1",
               "thread_id": old["id"], "from_team": "data_science", "to_team": "business", "text": "the answer", "initiator": None}
        landed = th.post_delivery(self.services, row)
        self.assertNotEqual(landed["thread_id"], old["id"])
        self.assertEqual(store.thread(landed["thread_id"])["team"], team)
        self.assertEqual(store.messages(old["id"]), [])


class FileLocalTests(_Local, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}


class PgLocalTests(_Local, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}


class MigrateTeamspaceTests(unittest.TestCase):
    """``storage migrate`` moves a file workspace's team work and threads into PostgreSQL, once."""

    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()

    def test_the_team_work_and_threads_move_once_and_read_the_same(self):
        from dclab_rnd import context, experience
        from dclab_rnd.jobs import open_jobs
        from dclab_rnd.storage import open_stores, open_team_work
        from dclab_rnd.storage.migrate import migrate

        source = Path(tempfile.mkdtemp()) / "files"
        source.mkdir()
        with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": "", "DCLAB_AUTH": "", "DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS": "0"}):
            projects, drafts, _ = open_stores(source)
            env = SimpleNamespace(store=SimpleNamespace(home=source), projects=projects, drafts=drafts, workspace_id=None,
                                  experience=experience.open_log(source, None), job_store=open_jobs(source),
                                  teamwork=open_team_work(source, projects=projects))
            token = context.set_services(env)
            try:
                delivery.set_target(None)
                pid = projects.create("Renewal", goal="Who should Sales call")["id"]
                prob = create_problem(projects, pid, "Renewal", workspace_id="local")["problem_id"]
                biz = th.open_threads(source).open_thread(pid, prob, "business", label="Business")[0]["id"]
                agents.open_case(env, pid, prob, task_id="T-renewal", question=BRIEF, thread_id=biz, member="sara",
                                 inputs=("sample:renewal_v1",))
                settle(env)
                agents.person_replies(env, pid, "T-renewal", person="sara", text=ANSWER)
                settle(env)
                held_view = env.teamwork.task_view(pid, "T-renewal")
                held_history = history_lines(env.teamwork.handoff_history(pid, prob))
                held_threads = {t["team"]: [m["text"] for m in th.open_threads(source).messages(t["id"])]
                                for t in th.open_threads(source).threads(project_id=pid)}
            finally:
                context.reset_services(token)
        target = Path(tempfile.mkdtemp()) / "db"
        with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": self.url}):
            first = migrate(source, target, self.url)
            self.assertEqual(first["logs"]["team_work"]["copied"], first["logs"]["team_work"]["source"])
            self.assertEqual(first["logs"]["team_threads"]["copied"], first["logs"]["team_threads"]["source"])
            moved = open_team_work(target)
            view = moved.task_view(pid, "T-renewal")
            self.assertEqual(view["task"].status, "complete")
            self.assertEqual(view["task"].workspace_id, moved.workspace)  # local became the database's workspace
            self.assertEqual([r.answer for r in view["reports"]], [r.answer for r in held_view["reports"]])
            self.assertEqual([a.action_id for a in view["actions"]], [a.action_id for a in held_view["actions"]])
            self.assertEqual(history_lines(moved.handoff_history(pid, prob)), held_history)
            threads = th.open_threads(target)
            self.assertEqual({t["team"]: [m["text"] for m in threads.messages(t["id"])] for t in threads.threads(project_id=pid)},
                             held_threads)
            again = migrate(source, target, self.url)
            self.assertEqual((again["logs"]["team_work"]["copied"], again["logs"]["team_threads"]["copied"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
