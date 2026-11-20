"""W3 of the collaborative workspace (2026-11-20): each team's own thread on a problem, and the shared handoff history.

The store on files and on PostgreSQL alike: one thread a (project, problem, team), messages appended whole and never
rewritten, a delivery posted once whatever the retries, another project or workspace finding nothing. The delivery
outbox's default target posts the renewal round trip into the teams' threads (the reply into the thread it came from),
and the history reads like the owner's example with no thread's private words in it. Over HTTP without sign-in the local
owner reads and posts everywhere; with sign-in (PostgreSQL) two people in different teams each see only their own
thread, a viewer reads and cannot post, the owner reads every thread, and another project or workspace is refused. The
stream sends a new message and resumes from Last-Event-ID. Run on PostgreSQL through ``tests/run_parallel.py --pg``
(never directly: that empties ``dclab_test``).
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

from dclab_core import team_work as tw  # noqa: E402
from dclab_rnd import context, experience  # noqa: E402
from dclab_rnd.jobs import Worker, open_jobs  # noqa: E402
from dclab_rnd.jobs import handlers  # noqa: E402,F401 — registers the kinds, team_delivery among them
from dclab_rnd.memory.scope import create_problem  # noqa: E402
from dclab_rnd.storage import TeamThreads, open_stores, open_team_threads, open_team_work, open_teams  # noqa: E402
from dclab_rnd.teamspace import delivery, history_lines  # noqa: E402
from dclab_rnd.teamspace import threads as th  # noqa: E402

PASSWORD = "a long enough password"
T0, T1 = "2026-11-20T18:00:00+00:00", "2026-11-20T18:01:00+00:00"
QUESTION = "Can we rank accounts seven days before renewal?"
CLARIFY = "Does churn mean cancellation or failure to renew by the contractual date?"
ANSWER = "Confirmed by Business: non-renewal by the contractual date."
PRIVATE = "Between us: the renewal discount budget is only 12 percent this quarter."
DS = tw.Actor(kind="agent", id="data_science", team="data_science")
SARA = tw.Actor(kind="person", id="sara", team="business")
CLOSED = tw.NextStep(owner="business", unblocks="Business reads the finding")
EXAMPLE = [f"1. Business → Data Science · {QUESTION}", f"2. Data Science → Business · {CLARIFY}", f"3. Business → Data Science · {ANSWER}"]


def ids_in(body: str) -> list[int]:
    return [int(n) for n in re.findall(r"^id: (\d+)$", body, flags=re.M)]


def wait_for(fn, seconds=20.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timed out")


class _Store:
    """The store, on files and on PostgreSQL."""

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
        self.projects, _, _ = open_stores(self.home)
        self.store = open_team_threads(self.home)
        self.ws = th.open_threads(self.home).workspace
        self.log = experience.open_log(self.home, self.ws if self.ws != "local" else None)
        token = context.set_services(SimpleNamespace(experience=self.log))
        self.addCleanup(context.reset_services, token)
        self.pid = self.projects.create("Renewal", goal="Who should Sales call")["id"]
        self.pid2 = self.projects.create("Delivery", goal="Dispatch")["id"]
        self.prob = create_problem(self.projects, self.pid, "Renewal", workspace_id=self.ws)["problem_id"]

    def test_the_store_is_chosen_as_the_other_stores_are(self):
        self.assertIsInstance(self.store, self.store_type)
        self.assertIsInstance(self.store, TeamThreads)

    def test_one_thread_a_team_and_problem_with_its_messages_kept_whole_and_numbered(self):
        biz, made = self.store.open_thread(self.pid, self.prob, "team_biz", label="Business", by={"kind": "person", "id": "sara"})
        again, made_again = self.store.open_thread(self.pid, self.prob, "team_biz", label="Business")
        self.assertEqual((made, made_again, again["id"]), (True, False, biz["id"]))
        science, _ = self.store.open_thread(self.pid, self.prob, "team_ds", label="Data Science")
        self.assertNotEqual(science["id"], biz["id"])
        long_words = "Who should Sales call before renewal? " * 400  # far longer than any shown cut: kept whole
        first, added = self.store.post(biz["id"], {"kind": "person", "id": "sara", "team": "team_biz"}, long_words)
        second, _ = self.store.post(biz["id"], {"kind": "agent", "id": "business", "team": "business"}, "Asked Data Science.")
        self.assertTrue(added)
        self.assertEqual((first["seq"], second["seq"]), (1, 2))
        self.assertEqual(first["text"], long_words)
        self.assertEqual((first["visibility"], first["author"]["kind"], second["author"]["kind"]), ("team", "person", "agent"))
        self.assertEqual([m["seq"] for m in self.store.messages(biz["id"])], [1, 2])
        self.assertEqual([m["seq"] for m in self.store.messages(biz["id"], after=1)], [2])
        self.assertEqual(self.store.messages(science["id"]), [])  # another team's thread holds none of it
        self.assertEqual(self.store.thread(biz["id"])["messages"], 2)
        self.assertEqual([t["id"] for t in self.store.threads(teams=["team_ds"])], [science["id"]])
        self.assertEqual(self.store.find_thread(self.pid, self.prob, "team_ds")["id"], science["id"])
        for bad in ({"kind": "robot", "id": "x"}, {"kind": "person", "id": ""}):
            with self.assertRaises(th.ThreadError):
                self.store.post(biz["id"], bad, "words")
        with self.assertRaises(th.ThreadError):
            self.store.post(biz["id"], {"kind": "person", "id": "sara"}, "   ")
        with self.assertRaises(KeyError):
            self.store.post("th_nosuch", {"kind": "person", "id": "sara"}, "words")
        events = [e.payload for e in self.log.read(kinds=["team.handoff"])]
        self.assertEqual([e["what"] for e in events].count("thread.message"), 2)
        self.assertFalse(any(long_words[:40] in str(e) for e in events))  # names only, never the words

    def test_a_delivery_key_is_posted_once_in_a_project(self):
        biz, _ = self.store.open_thread(self.pid, self.prob, "team_biz")
        agent = {"kind": "agent", "id": "data_science", "team": "data_science"}
        one, added = self.store.post(biz["id"], agent, CLARIFY, handoff={"delivery_id": "d1"}, key="delivery:d1")
        two, added_again = self.store.post(biz["id"], agent, CLARIFY, handoff={"delivery_id": "d1"}, key="delivery:d1")
        self.assertEqual((added, added_again, two["id"]), (True, False, one["id"]))
        other, _ = self.store.open_thread(self.pid, self.prob, "team_ds")
        three, added_elsewhere = self.store.post(other["id"], agent, CLARIFY, key="delivery:d1")  # another thread: still once
        self.assertEqual((added_elsewhere, three["id"]), (False, one["id"]))
        self.assertEqual(len(self.store.messages(biz["id"])), 1)
        self.assertEqual(self.store.counts()["messages"], 1)

    def test_another_project_and_another_workspace_find_nothing(self):
        biz, _ = self.store.open_thread(self.pid, self.prob, "team_biz")
        self.store.post(biz["id"], {"kind": "person", "id": "sara"}, PRIVATE)
        self.assertEqual(self.store.threads(project_id=self.pid2), [])
        self.assertIsNone(self.store.find_thread(self.pid2, self.prob, "team_biz"))
        theirs = open_team_threads(self.other_home)
        self.assertIsNone(theirs.thread(biz["id"]))
        self.assertEqual(theirs.messages(biz["id"]), [])
        self.assertEqual(theirs.threads(), [])
        with self.assertRaises(KeyError):
            theirs.post(biz["id"], {"kind": "person", "id": "mallory"}, "hello")
        self.assertEqual(len(self.store.messages(biz["id"])), 1)

    def test_who_may_read_and_post(self):
        teams = SimpleNamespace(is_member=lambda team, user: (team, user) in {("team_biz", "sara"), ("team_biz", "vic")})
        thread = {"team": "team_biz"}
        P = SimpleNamespace
        self.assertEqual(th.access(thread, P(user_id=None, role="owner", via="local"), teams, auth_on=False), "write")
        self.assertIsNone(th.access(thread, P(user_id=None, role=None, via="anonymous"), teams, auth_on=False))
        self.assertEqual(th.access(thread, P(user_id="sara", role="data_scientist"), teams, auth_on=True), "write")
        self.assertEqual(th.access(thread, P(user_id="vic", role="viewer"), teams, auth_on=True), "read")
        self.assertEqual(th.access(thread, P(user_id="olga", role="owner"), teams, auth_on=True), "read")
        self.assertIsNone(th.access(thread, P(user_id="dan", role="data_scientist"), teams, auth_on=True))
        self.assertIsNone(th.access(thread, P(user_id=None, role=None), teams, auth_on=True))

    def test_a_team_agents_name_maps_to_the_workspaces_one_team_of_that_kind(self):
        teams = open_teams(self.home)
        self.assertEqual(th.team_key(teams, "business"), ("business", "Business"))  # no team yet: only the owner reads it
        made = teams.create_team("Business", "business")
        self.assertEqual(th.team_key(teams, "business"), (made["id"], "Business"))
        self.assertEqual(th.team_key(teams, made["id"]), (made["id"], "Business"))
        teams.create_team("Business EU", "business")
        self.assertEqual(th.team_key(teams, "business"), ("business", "Business"))  # two of a kind: never a guess


class FileStoreTests(_Store, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}
    store_type = th.FileThreads

    def test_the_file_is_the_projects_folder_beside_its_work(self):
        biz, _ = self.store.open_thread(self.pid, self.prob, "team_biz")
        self.store.post(biz["id"], {"kind": "person", "id": "sara"}, "hello")
        self.assertTrue(list((self.home / "teamspace").glob(f"*/{th.FILE}")))


class PgStoreTests(_Store, unittest.TestCase):
    store_type = th.PgThreads

    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}

    def test_a_message_is_never_rewritten(self):
        import sqlalchemy as sa

        from dclab_rnd.storage import db

        biz, _ = self.store.open_thread(self.pid, self.prob, "team_biz")
        self.store.post(biz["id"], {"kind": "person", "id": "sara"}, "the words as written")
        with self.assertRaisesRegex(Exception, "append-only"):
            with db.engine(self.url).begin() as c:
                c.execute(sa.text("update team_message set key = 'x' where thread_id = :t"), {"t": biz["id"]})
        self.assertEqual(self.store.messages(biz["id"])[0]["text"], "the words as written")

    def test_two_workers_posting_one_delivery_at_once_leave_one_message(self):
        """The second insert meets the unique key (its check ran before the first committed): the first message stands."""
        biz, _ = self.store.open_thread(self.pid, self.prob, "team_biz")
        agent = {"kind": "agent", "id": "data_science", "team": "data_science"}
        first, _ = self.store.post(biz["id"], agent, CLARIFY, key="delivery:d9")
        real, calls = th.PgThreads._by_key, {"n": 0}

        def late_check(store, c, project_id, key):
            calls["n"] += 1
            return None if calls["n"] == 1 else real(store, c, project_id, key)
        with mock.patch.object(th.PgThreads, "_by_key", late_check):
            again, added = self.store.post(biz["id"], agent, CLARIFY, key="delivery:d9")
        self.assertEqual((added, again["id"], calls["n"]), (False, first["id"], 2))
        self.assertEqual([m["seq"] for m in self.store.messages(biz["id"])], [1])
        self.assertEqual(self.store.thread(biz["id"])["messages"], 1)  # the losing post's number was rolled back


# ---------------------------------------------------------------------------------------------------------- delivery
class _Delivery:
    """The outbox's default target posts the renewal round trip into the teams' threads, each delivery once."""

    env: dict = {}

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"DCLAB_AUTH": "", "DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS": "0.1",
                                               "DCLAB_TEAM_DELIVERY_ATTEMPTS": "3", **self.env})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.home = Path(tempfile.mkdtemp()) / "workspace"
        self.home.mkdir()
        from dclab_rnd.storage import db

        projects, drafts, _ = open_stores(self.home)
        wid = db.workspace_for(self.home) if db.database_url(required=False) else None
        self.env_ = SimpleNamespace(store=SimpleNamespace(home=self.home), projects=projects, drafts=drafts, workspace_id=wid,
                                    experience=experience.open_log(self.home, wid), job_store=open_jobs(self.home),
                                    teamwork=open_team_work(self.home, projects=projects))
        token = context.set_services(self.env_)
        self.addCleanup(context.reset_services, token)
        from dclab_rnd.teamspace import threads as team_threads

        # the default target's thread post alone: these tests move the tasks by hand, so the agents' turn that follows it
        # in the default (W5) is tested by test_teamspace_agents
        previous = delivery.set_target(team_threads.deliver)
        self.addCleanup(delivery.set_target, previous)
        previous_checker = delivery.set_membership_checker(None)
        self.addCleanup(delivery.set_membership_checker, previous_checker)
        self.work = self.env_.teamwork
        self.ws = self.work.workspace
        self.pid = projects.create("Renewal", goal="Who should Sales call")["id"]
        self.prob = create_problem(projects, self.pid, "Renewal", workspace_id=self.ws)["problem_id"]
        self.sara = {"user_id": "sara", "role": "data_scientist", "workspace_id": self.ws}
        teams = open_teams(self.home)
        self.biz_team = teams.create_team("Business", "business")["id"]
        self.ds_team = teams.create_team("Data Science", "data_science")["id"]
        self.threads = open_team_threads(self.home)
        self.biz_thread, _ = self.threads.open_thread(self.pid, self.prob, self.biz_team, label="Business")

    def run_job(self, job):
        self.assertEqual(Worker(self.env_.job_store, self.env_, threads=0).run_here(job, timeout=30)["status"], "done")

    def deliver(self, written):
        self.run_job(delivery.send(self.env_, written))
        return self.work.delivery(written.delivery["id"])

    def relay(self):
        for key in delivery.requeue_undelivered(self.env_):
            job = self.env_.job_store.active(delivery.KIND, key)
            if job is not None:
                self.run_job(job)

    def task(self):
        return tw.TeamTask(id="T-ds", workspace_id=self.ws, project_id=self.pid, problem_id=self.prob, brief_revision=1,
                           requester=tw.Requester(team="business", member="sara", thread_id=self.biz_thread["id"]),
                           owner_team="data_science", question=QUESTION, expected_output="feasibility",
                           done_criteria=["says whether the data can rank accounts"], created_at=T0)

    def test_the_round_trip_lands_in_each_teams_thread_and_the_history_reads_like_the_owners_example(self):
        self.threads.post(self.biz_thread["id"], {"kind": "person", "id": "sara", "team": self.biz_team}, PRIVATE)
        asked = self.deliver(self.work.create_task(self.task(), initiator=self.sara))
        ds_thread = self.threads.find_thread(self.pid, self.prob, self.ds_team)
        self.assertIsNotNone(ds_thread)  # opened by the first delivery to Data Science
        self.assertEqual(asked["receipt"]["thread_id"], ds_thread["id"])
        self.work.move(self.pid, "T-ds", "accepted", DS)
        clarified = self.deliver(self.work.move(self.pid, "T-ds", "waiting_for_person", DS, note=CLARIFY))
        self.assertEqual(clarified["receipt"]["thread_id"], self.biz_thread["id"])  # back into the thread the task came from
        self.deliver(self.work.move(self.pid, "T-ds", "accepted", DS, note=ANSWER))
        science = [(m["author"]["team"], m["text"], m["handoff"]["kind"]) for m in self.threads.messages(ds_thread["id"])]
        self.assertEqual(science, [("business", QUESTION, "task"), ("business", ANSWER, "answer")])
        business = [(m["author"]["kind"], m["text"]) for m in self.threads.messages(self.biz_thread["id"])]
        self.assertEqual(business, [("person", PRIVATE), ("agent", CLARIFY)])
        lines = history_lines(self.work.handoff_history(self.pid, self.prob))
        self.assertEqual(lines, EXAMPLE)
        self.assertFalse(any(PRIVATE in line or "12 percent" in line for line in lines))
        whats = [e.payload["what"] for e in self.env_.experience.read(kinds=["team.handoff"])]
        self.assertIn("thread.opened", whats)
        self.assertEqual(whats.count("thread.message"), 4)

    def test_a_delivery_delivered_twice_is_posted_once(self):
        calls = {"n": 0}

        def crash_after_posting(env, row):  # the target posted, then the worker died before the row was marked delivered
            receipt = th.deliver(env, row)  # the thread post (the agents' turn after it: test_teamspace_agents)
            calls["n"] += 1
            if calls["n"] == 1:
                raise delivery.TemporaryFailure("cut off after posting")
            return receipt
        delivery.set_target(crash_after_posting)
        written = self.work.create_task(self.task(), initiator=self.sara)
        self.run_job(delivery.send(self.env_, written))
        self.assertEqual(self.work.delivery(written.delivery["id"])["status"], "pending")
        wait_for(lambda: (self.relay(), self.work.delivery(written.delivery["id"])["status"] == "delivered")[1])
        self.assertEqual(calls["n"], 2)
        again = th.deliver(self.env_, self.work.delivery(written.delivery["id"]))  # and a third time, by hand
        self.assertFalse(again["posted"])
        ds_thread = self.threads.find_thread(self.pid, self.prob, self.ds_team)
        self.assertEqual([m["text"] for m in self.threads.messages(ds_thread["id"])], [QUESTION])

    def test_a_reply_never_lands_in_another_teams_thread(self):
        """A task whose requester names another team's thread: its reply goes to the requester team's own thread."""
        ds_thread, _ = self.threads.open_thread(self.pid, self.prob, self.ds_team)
        task = self.task().model_copy(update={"requester": tw.Requester(team="business", member="sara", thread_id=ds_thread["id"])})
        self.work.create_task(task, initiator=self.sara)
        self.work.move(self.pid, "T-ds", "accepted", DS)
        row = self.work.move(self.pid, "T-ds", "waiting_for_person", DS, note=CLARIFY).delivery
        receipt = th.deliver(self.env_, row)
        self.assertEqual(receipt["thread_id"], self.biz_thread["id"])
        self.assertEqual([m["text"] for m in self.threads.messages(ds_thread["id"])], [])


    def member(self, name):
        """A person of this workspace: any id on files, an account on PostgreSQL (a team's member must be one there)."""
        from dclab_rnd.storage import db

        if not db.database_url(required=False):
            return name
        from dclab_rnd.accounts.store import Accounts

        accounts = Accounts()
        user = accounts.create_user(f"{name}-{os.urandom(4).hex()}@example.com", name.title(), password=PASSWORD)["id"]
        accounts.set_member(self.ws, user, "data_scientist")
        return user

    def test_with_two_teams_of_a_kind_the_reply_goes_back_to_the_askers_own_team_thread(self):
        teams = open_teams(self.home)
        eu = teams.create_team("Business EU", "business")["id"]
        self.assertEqual(th.team_key(teams, "business"), ("business", "Business"))  # two of a kind: no single team
        sara = self.member("sara")
        teams.add_member(self.biz_team, sara)
        initiator = {"user_id": sara, "role": "data_scientist", "workspace_id": self.ws}
        self.work.create_task(self.task(), initiator=initiator)
        self.work.move(self.pid, "T-ds", "accepted", DS)
        row = self.work.move(self.pid, "T-ds", "waiting_for_person", DS, note=CLARIFY).delivery
        self.assertEqual(th.deliver(self.env_, row)["thread_id"], self.biz_thread["id"])  # Sara's team's thread, which she reads
        eu_thread, _ = self.threads.open_thread(self.pid, self.prob, eu)
        stranger = {**dict(row), "id": "d-other", "thread_id": eu_thread["id"]}  # Sara is not in Business EU: never posted there
        self.assertNotEqual(th.deliver(self.env_, stranger)["thread_id"], eu_thread["id"])
        self.assertEqual(self.threads.messages(eu_thread["id"]), [])

    def test_an_agents_words_are_posted_as_they_travelled_and_nothing_at_all_is_blocked_at_once(self):
        row = self.work.create_task(self.task(), initiator=self.sara).delivery
        spaces = th.deliver(self.env_, {**row, "id": "d-spaces", "text": "   "})  # a report's answer team_work accepted
        ds_thread = self.threads.find_thread(self.pid, self.prob, self.ds_team)
        self.assertEqual((spaces["thread_id"], self.threads.messages(ds_thread["id"])[-1]["text"]), (ds_thread["id"], "   "))
        with self.assertRaises(delivery.PermanentFailure):
            th.deliver(self.env_, {**row, "id": "d-empty", "text": ""})


class FileDeliveryTests(_Delivery, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}


class PgDeliveryTests(_Delivery, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}


# ---------------------------------------------------------------------------------------------------------- HTTP, no sign-in
class _RoutesLocal:
    """Without sign-in: the local owner reads and posts in every thread; another project or workspace is refused."""

    env: dict = {}

    def setUp(self):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            raise unittest.SkipTest(f"Studio dependencies not installed: {error}")
        patcher = mock.patch.dict(os.environ, {"DCLAB_AUTH": "", "OPENAI_API_KEY": "", **self.env})
        patcher.start()
        self.addCleanup(patcher.stop)
        root = Path(tempfile.mkdtemp())
        self.app = create_app(root / "workspace")
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.headers["X-DCLab-Token"] = self.client.get("/api/config").json()["csrf"]
        s = self.app.state.services
        self.services = s
        self.ws = th.open_threads(s.store.home).workspace
        self.pid = s.projects.create("Renewal", goal="Who should Sales call")["id"]
        self.pid2 = s.projects.create("Delivery", goal="Dispatch")["id"]
        self.prob = create_problem(s.projects, self.pid, "Renewal", workspace_id=self.ws)["problem_id"]
        self.prob2 = create_problem(s.projects, self.pid2, "Dispatch", workspace_id=self.ws)["problem_id"]

    def open(self, team="business", pid=None, prob=None):
        r = self.client.post(f"/api/projects/{pid or self.pid}/problems/{prob or self.prob}/team-threads", json={"team": team})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_the_local_owner_opens_reads_and_posts(self):
        thread = self.open()
        self.assertEqual((thread["team"], thread["team_label"], thread["access"], thread["opened"]), ("business", "Business", "write", True))
        self.assertFalse(self.open()["opened"])
        posted = self.client.post(f"/api/projects/{self.pid}/team-threads/{thread['id']}/messages", json={"text": PRIVATE})
        self.assertEqual(posted.status_code, 201, posted.text)
        self.assertEqual((posted.json()["seq"], posted.json()["author"]["id"], posted.json()["text"]), (1, "owner", PRIVATE))
        read = self.client.get(f"/api/projects/{self.pid}/team-threads/{thread['id']}").json()
        self.assertEqual([m["text"] for m in read["messages"]], [PRIVATE])
        self.assertEqual([t["id"] for t in self.client.get("/api/team-threads").json()], [thread["id"]])
        self.assertEqual(self.client.get("/api/team-threads", params={"project_id": self.pid2}).json(), [])
        self.assertEqual(self.client.post(f"/api/projects/{self.pid}/team-threads/{thread['id']}/messages", json={"text": " "}).status_code, 422)
        too_long = "x" * (th.TEXT_MAX + 1)
        self.assertEqual(self.client.post(f"/api/projects/{self.pid}/team-threads/{thread['id']}/messages", json={"text": too_long}).status_code, 422)
        self.assertEqual(self.client.post(f"/api/projects/{self.pid}/problems/{self.prob}/team-threads", json={"team": "nobody"}).status_code, 404)
        whats = [e.payload["what"] for e in self.services.experience.read(kinds=["team.handoff"])]
        self.assertEqual((whats.count("thread.opened"), whats.count("thread.message")), (1, 1))

    def test_another_project_or_workspace_is_refused(self):
        thread = self.open()
        for path in (f"/api/projects/{self.pid2}/team-threads/{thread['id']}", f"/api/projects/{self.pid}/team-threads/th_nosuch",
                     f"/api/projects/{self.pid2}/team-threads/{thread['id']}/events?wait=0"):
            self.assertEqual(self.client.get(path).status_code, 403, path)
        self.assertEqual(self.client.post(f"/api/projects/{self.pid2}/team-threads/{thread['id']}/messages", json={"text": "hi"}).status_code, 403)
        self.assertEqual(self.client.post(f"/api/projects/{self.pid}/problems/{self.prob2}/team-threads", json={"team": "business"}).status_code, 403)
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        with TestClient(create_app(Path(tempfile.mkdtemp()) / "other")) as other:
            other.headers["X-DCLab-Token"] = other.get("/api/config").json()["csrf"]
            self.assertEqual(other.get(f"/api/projects/{self.pid}/team-threads/{thread['id']}").status_code, 403)
            self.assertEqual(other.post(f"/api/projects/{self.pid}/team-threads/{thread['id']}/messages", json={"text": "hi"}).status_code, 403)
            self.assertEqual(other.get("/api/team-threads").json(), [])
            self.assertEqual(other.get(f"/api/projects/{self.pid}/problems/{self.prob}/handoffs").status_code, 403)

    def test_the_stream_sends_a_new_message_and_resumes_from_last_event_id(self):
        thread = self.open()
        base = f"/api/projects/{self.pid}/team-threads/{thread['id']}"
        self.client.post(f"{base}/messages", json={"text": "first"})
        store = th.open_threads(self.services.store.home)

        def later():
            time.sleep(0.6)
            store.post(thread["id"], {"kind": "agent", "id": "data_science", "team": "data_science"}, "a new message")
        poster = threading.Thread(target=later)
        poster.start()
        body = self.client.get(f"{base}/events", params={"wait": 4}).text
        poster.join()
        self.assertEqual(ids_in(body), [1, 2])
        self.assertIn("a new message", body)
        self.assertIn("event: message", body)
        self.client.post(f"{base}/messages", json={"text": "third"})
        resumed = self.client.get(f"{base}/events", params={"wait": 0}, headers={"Last-Event-ID": "1"}).text
        self.assertEqual(ids_in(resumed), [2, 3])
        self.assertEqual(ids_in(self.client.get(f"{base}/events", params={"wait": 0, "after": 2}).text), [3])

    def test_the_handoff_history_reads_like_the_owners_example(self):
        work = open_team_work(self.services.store.home, projects=self.services.projects)
        thread = self.open()
        self.client.post(f"/api/projects/{self.pid}/team-threads/{thread['id']}/messages", json={"text": PRIVATE})
        task = tw.TeamTask(id="T-ds", workspace_id=self.ws, project_id=self.pid, problem_id=self.prob, brief_revision=1,
                           requester=tw.Requester(team="business", member="owner", thread_id=thread["id"]), owner_team="data_science",
                           question=QUESTION, expected_output="feasibility", done_criteria=["says whether"], created_at=T0)
        rows = [work.create_task(task).delivery]
        work.move(self.pid, "T-ds", "accepted", DS)
        rows.append(work.move(self.pid, "T-ds", "waiting_for_person", DS, note=CLARIFY).delivery)
        rows.append(work.move(self.pid, "T-ds", "accepted", DS, note=ANSWER).delivery)
        for row in rows:
            work.delivery_done(row["id"], {"inbox": row["to_team"]}, project_id=self.pid)
        got = self.client.get(f"/api/projects/{self.pid}/problems/{self.prob}/handoffs")
        self.assertEqual(got.status_code, 200, got.text)
        body = got.json()
        self.assertEqual((body["title"], body["count"], body["lines"]), ("Shared handoff history · 3", 3, EXAMPLE))
        self.assertNotIn("12 percent", got.text)
        self.assertEqual(self.client.get(f"/api/projects/{self.pid2}/problems/{self.prob}/handoffs").status_code, 403)


class FileRoutesLocalTests(_RoutesLocal, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}


class PgRoutesLocalTests(_RoutesLocal, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}


# ---------------------------------------------------------------------------------------------------------- HTTP, sign-in
class SignedInRoutesTests(unittest.TestCase):
    """With password sign-in (PostgreSQL): two workspaces; in the first an owner, a Business member, a Data Science member,
    a viewer in Business, and an ML engineer in no team."""

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
                                                "DCLAB_TEAM_THREAD_RECHECK_SECONDS": "0.2"})
        cls.patch.start()
        from dclab_rnd.accounts.store import Accounts

        cls.TestClient = TestClient
        cls.app = create_app(Path(tempfile.mkdtemp()))
        cls.main = TestClient(cls.app)
        cls.main.__enter__()
        cls.accounts = Accounts()
        s = cls.app.state.services
        cls.w1 = s.workspace_id
        cls.w2 = cls.accounts.create_workspace("Other company")
        cls.users = {}
        for name, workspace, role in (("owner1", cls.w1, "owner"), ("biz", cls.w1, "data_scientist"), ("ds", cls.w1, "data_scientist"),
                                      ("viewer", cls.w1, "viewer"), ("loner", cls.w1, "data_scientist"), ("owner2", cls.w2, "owner")):
            user = cls.accounts.create_user(f"{name}-{os.getpid()}-{os.urandom(3).hex()}@example.com", name.title(), password=PASSWORD)
            cls.accounts.set_member(workspace, user["id"], role)
            cls.users[name] = user
        teams = open_teams(s.store.home)
        cls.business = teams.create_team("Business", "business")["id"]
        cls.science = teams.create_team("Data Science", "data_science")["id"]
        for name, team in (("biz", cls.business), ("viewer", cls.business), ("ds", cls.science)):
            teams.add_member(team, cls.users[name]["id"])
        cls.pid = s.projects.create("Renewal", goal="Who should Sales call")["id"]
        cls.pid2 = s.projects.create("Delivery", goal="Dispatch")["id"]
        cls.prob = create_problem(s.projects, cls.pid, "Renewal", workspace_id=cls.w1)["problem_id"]
        cls.clients = {name: cls.sign_in(name) for name in cls.users}
        cls.services = s

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

    def path(self, thread_id, pid=None):
        return f"/api/projects/{pid or self.pid}/team-threads/{thread_id}"

    def open(self, who, team):
        r = self.clients[who].post(f"/api/projects/{self.pid}/problems/{self.prob}/team-threads", json={"team": team})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_each_team_sees_only_its_own_thread(self):
        biz_thread, ds_thread = self.open("biz", self.business), self.open("ds", self.science)
        c = self.clients
        self.assertEqual(c["biz"].post(f"{self.path(biz_thread['id'])}/messages", json={"text": PRIVATE}).status_code, 201)
        self.assertEqual(c["ds"].post(f"{self.path(ds_thread['id'])}/messages", json={"text": "Our own working notes."}).status_code, 201)
        self.assertEqual([t["id"] for t in c["biz"].get("/api/team-threads").json()], [biz_thread["id"]])
        self.assertEqual([t["id"] for t in c["ds"].get("/api/team-threads").json()], [ds_thread["id"]])
        self.assertEqual(c["loner"].get("/api/team-threads").json(), [])
        for who, theirs in (("biz", ds_thread), ("ds", biz_thread), ("loner", biz_thread), ("loner", ds_thread)):
            for answer in (c[who].get(self.path(theirs["id"])), c[who].get(f"{self.path(theirs['id'])}/events", params={"wait": 0}),
                           c[who].post(f"{self.path(theirs['id'])}/messages", json={"text": "let me in"})):
                self.assertEqual(answer.status_code, 403, (who, answer.text))
                self.assertNotIn("12 percent", answer.text)
        refused = c["ds"].post(f"/api/projects/{self.pid}/problems/{self.prob}/team-threads", json={"team": self.business})
        self.assertEqual(refused.status_code, 403)  # nobody opens another team's thread
        mine = c["biz"].get(self.path(biz_thread["id"])).json()  # (the other tests of this class post to it too)
        self.assertEqual((mine["messages"][-1]["text"], mine["access"]), (PRIVATE, "write"))
        self.assertEqual(mine["messages"][-1]["author"]["id"], self.users["biz"]["id"])
        stream = c["biz"].get(f"{self.path(biz_thread['id'])}/events", params={"wait": 0})
        self.assertEqual((stream.status_code, ids_in(stream.text)), (200, list(range(1, len(mine["messages"]) + 1))))
        # another project: the same thread through it is refused
        self.assertEqual(c["biz"].get(self.path(biz_thread["id"], self.pid2)).status_code, 403)
        self.assertEqual(c["biz"].get("/api/team-threads", params={"project_id": self.pid2}).json(), [])

    def test_a_viewer_reads_and_cannot_post_and_the_owner_reads_every_thread(self):
        biz_thread = self.open("biz", self.business)
        self.open("ds", self.science)
        c = self.clients
        self.assertEqual(c["viewer"].get(self.path(biz_thread["id"])).json()["access"], "read")
        refused = c["viewer"].post(f"{self.path(biz_thread['id'])}/messages", json={"text": "may I?"})
        self.assertEqual(refused.status_code, 403, refused.text)
        self.assertEqual(c["viewer"].post(f"/api/projects/{self.pid}/problems/{self.prob}/team-threads", json={"team": self.business}).status_code, 403)
        listed = c["owner1"].get("/api/team-threads").json()
        self.assertEqual({t["team"] for t in listed}, {self.business, self.science})
        self.assertEqual({t["access"] for t in listed}, {"read"})  # the owner is in no team here: reads, does not post
        self.assertEqual(c["owner1"].post(f"{self.path(biz_thread['id'])}/messages", json={"text": "owner here"}).status_code, 403)

    def test_another_workspace_sees_nothing_and_every_project_reader_gets_the_history(self):
        biz_thread = self.open("biz", self.business)
        c = self.clients
        c["biz"].post(f"{self.path(biz_thread['id'])}/messages", json={"text": PRIVATE})
        stranger = c["owner2"]
        self.assertEqual(stranger.get("/api/team-threads").json(), [])
        self.assertEqual(stranger.get(self.path(biz_thread["id"])).status_code, 403)
        self.assertEqual(stranger.get(self.path(biz_thread["id"])).json(), stranger.get(self.path("th_nosuch")).json())  # says nothing
        self.assertEqual(stranger.post(f"{self.path(biz_thread['id'])}/messages", json={"text": "hi"}).status_code, 403)
        self.assertEqual(stranger.get(f"/api/projects/{self.pid}/problems/{self.prob}/handoffs").status_code, 403)
        work = open_team_work(self.services.store.home, projects=self.services.projects)
        biz = self.users["biz"]
        task = tw.TeamTask(id=f"T-{os.urandom(3).hex()}", workspace_id=self.w1, project_id=self.pid, problem_id=self.prob, brief_revision=1,
                           requester=tw.Requester(team="business", member=biz["id"], thread_id=biz_thread["id"]), owner_team="data_science",
                           question=QUESTION, expected_output="feasibility", done_criteria=["says whether"], created_at=T0)
        row = work.create_task(task, initiator={"user_id": biz["id"], "role": "data_scientist", "workspace_id": self.w1}).delivery
        work.delivery_done(row["id"], {"inbox": "data_science"}, project_id=self.pid)
        for who in ("viewer", "loner", "ds", "owner1"):
            got = c[who].get(f"/api/projects/{self.pid}/problems/{self.prob}/handoffs")
            self.assertEqual(got.status_code, 200, (who, got.text))
            self.assertIn(f"1. Business → Data Science · {QUESTION}", got.json()["lines"])
            self.assertNotIn("12 percent", got.text)


class SignedInStreamTests(SignedInRoutesTests):
    """The open stream of a member removed from the team stops receiving the team's messages."""

    def test_a_member_removed_from_the_team_stops_receiving_its_stream(self):
        biz_thread = self.open("biz", self.business)
        teams = open_teams(self.services.store.home)
        mover = self.accounts.create_user(f"mover-{os.urandom(4).hex()}@example.com", "Mover", password=PASSWORD)
        self.accounts.set_member(self.w1, mover["id"], "data_scientist")
        teams.add_member(self.business, mover["id"])
        self.users["mover"] = mover
        client = self.sign_in("mover")
        store = th.open_threads(self.services.store.home)
        agent = {"kind": "agent", "id": "data_science", "team": "data_science"}

        def later():
            time.sleep(0.5)
            store.post(biz_thread["id"], agent, "seen while still a member")
            time.sleep(0.5)
            teams.remove_member(self.business, mover["id"])
            time.sleep(0.6)  # past the recheck
            store.post(biz_thread["id"], agent, "after the member left")
        worker = threading.Thread(target=later)
        worker.start()
        body = client.get(f"{self.path(biz_thread['id'])}/events", params={"wait": 4}).text
        worker.join()
        self.assertIn("seen while still a member", body)
        self.assertNotIn("after the member left", body)
        self.assertEqual(client.get(self.path(biz_thread["id"])).status_code, 403)

    # the inherited tests run once, in SignedInRoutesTests
    test_each_team_sees_only_its_own_thread = None
    test_a_viewer_reads_and_cannot_post_and_the_owner_reads_every_thread = None
    test_another_workspace_sees_nothing_and_every_project_reader_gets_the_history = None


if __name__ == "__main__":
    unittest.main()
