"""W8 of the collaborative workspace (2026-11-20): what the team screens read, and the pages that read it.

The task page's route (``GET /api/projects/{pid}/team-tasks/{task_id}``) gives a task with its actions (simulated, denied
and failed told apart), its reports at every revision with their ids and the supersede links both ways, the children's
joined reports, who owes the next response, and its deliveries by name and state, never their words. It is read like a
thread: by a member of a team on the task's chain or the owner; anyone else, another project's task and a missing one get
the same 403. An agent's job description (``GET /api/team-agents/{name}``) is the directory's, by agent name or team id.
The Teams tab is a project page with its own tab, a sidebar entry, every read through ``DC.client``, the shared states,
the owner's handoff form and the browser check. PostgreSQL through ``tests/run_parallel.py --pg``.
"""

import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402

SRC = ROOT / "dclab_rnd" / "agentic" / "web" / "src"
BRIEF = ("Customers are leaving. Who should Sales call before renewal? Requested: can the data rank accounts by churn risk "
         "seven days before renewal?")
ANSWER = "Failure to renew by the contractual renewal date."
DISPATCH_Q = ("For order 1042 at its first dispatch: when will it arrive, and will a courier accept the first offer? "
              "Estimate the arrival time and the probability of acceptance.")
LATE = "dispatch:1042@18:05"  # a snapshot no specialist has a fixture for: both scores are denied


# ---------------------------------------------------------------------------------------------------------- shared with the browser check
def food_env(home: Path, jobs_home: Path | None = None) -> SimpleNamespace:
    """A workspace folder's team work as the agents use it (files; ``jobs_home`` keeps the jobs elsewhere, so a running
    server's worker never takes them)."""
    from dclab_rnd import experience
    from dclab_rnd.jobs import open_jobs
    from dclab_rnd.storage import open_stores, open_team_work

    projects, drafts, _ = open_stores(home)
    scratch = jobs_home or home
    return SimpleNamespace(store=SimpleNamespace(home=home), projects=projects, drafts=drafts, workspace_id=None,
                           experience=experience.open_log(scratch, None), job_store=open_jobs(scratch),
                           teamwork=open_team_work(home, projects=projects))


def pump(env, limit: int = 120) -> None:
    """Deliver what is pending, lane by lane, as the worker's jobs would."""
    from dclab_rnd.teamspace import delivery

    work = delivery.store_for(env)
    for _ in range(limit):
        heads = work.pending_heads()
        if not heads:
            return
        head = heads[0]
        for row in delivery.run(env, {"project_id": head["project_id"], "task_id": delivery.lane_of(work, head)}):
            assert row["status"] == "delivered", row
    raise AssertionError("the deliveries never settled")


def seed_food(env, pid: str, prob: str, thread_id: str, *, member: str = "owner", settle=None) -> dict[str, str]:
    """The owner's food delivery walkthrough on invented fixtures: ``fixed`` answered (ETA E1, acceptance A1) and then
    corrected (the kitchen adds 8 minutes: the ETA report r2 replaces r1); ``denied`` asked at a snapshot no specialist
    has, so both scores are denied and the brief names ML Engineering as the next owner. ``settle`` delivers what is
    pending (``pump``, by default; the browser check waits for its server's worker instead)."""
    from dclab_rnd import context
    from dclab_rnd.teamspace import agents, delivery, fixtures

    settle = settle or pump
    target, checker = delivery.set_target(None), delivery.set_membership_checker(None)
    token = context.set_services(env)
    try:
        for task_id, snapshot in (("T-fixed", fixtures.DISPATCH), ("T-denied", LATE)):
            out = agents.open_case(env, pid, prob, task_id=task_id, question=DISPATCH_Q, thread_id=thread_id, member=member,
                                   inputs=(snapshot,), expected_output="a dispatch brief: the arrival and the first-offer acceptance estimates")
            assert out["refused"] == [], out
            settle(env)
        out = agents.correct(env, pid, "T-fixed.eta", kitchen_minutes=8)
        assert out["refused"] == [], out
        settle(env)
    finally:
        context.reset_services(token)
        delivery.set_target(target)
        delivery.set_membership_checker(checker)
    return {"fixed": "T-fixed", "eta": "T-fixed.eta", "acceptance": "T-fixed.acceptance", "denied": "T-denied",
            "denied_acceptance": "T-denied.acceptance"}


# ---------------------------------------------------------------------------------------------------------- the routes, without sign-in
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
        from dclab_rnd.memory.scope import create_problem
        from dclab_rnd.teamspace import threads as th

        self.app = create_app(Path(tempfile.mkdtemp()) / "workspace")
        s = self.app.state.services
        self.home = Path(s.store.home)
        ws = th.open_threads(self.home).workspace
        self.pid = s.projects.create("Food delivery", goal="When does order 1042 arrive")["id"]
        self.prob = create_problem(s.projects, self.pid, "Order 1042", workspace_id=ws)["problem_id"]
        self.thread = th.open_threads(self.home).open_thread(self.pid, self.prob, "business", label="Business")[0]["id"]
        # the walkthrough is played before the app's worker starts, so only this test delivers its rows
        self.ids = seed_food(food_env(self.home, Path(tempfile.mkdtemp())), self.pid, self.prob, self.thread)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.headers["X-DCLab-Token"] = self.client.get("/api/config").json()["csrf"]

    def seed(self):
        return self.ids

    def task(self, task_id, pid=None):
        return self.client.get(f"/api/projects/{pid or self.pid}/team-tasks/{task_id}")

    def test_a_task_page_tells_simulated_denied_and_superseded_apart_and_names_who_answers_next(self):
        ids = self.seed()
        denied = self.task(ids["denied"]).json()
        self.assertEqual(denied["task"]["owner_label"], "Data Science")
        self.assertEqual(denied["task"]["requester_label"], "Business")
        self.assertEqual(denied["next_owner"], {"id": "ml_engineering", "kind": "team", "label": "ML Engineering"})
        [brief] = denied["reports"]
        self.assertEqual((brief["status"], brief["id"]), ("partial", "T-denied@r1"))
        self.assertNotIn("%", brief["answer"])  # no invented score
        kids = {r["task_id"]: r for r in denied["child_reports"]}
        self.assertEqual([(l["action"], l["status"]) for l in kids[ids["denied_acceptance"]]["actions"]["failed"]], [("score", "denied")])
        child = self.task(ids["denied_acceptance"]).json()
        self.assertIn("denied", [a["status"] for a in child["actions"]])
        self.assertEqual(child["task"]["owner_label"], "Model (acceptance)")

        eta = self.task(ids["eta"]).json()
        old, new = sorted(eta["reports"], key=lambda r: r["revision"])
        self.assertEqual((old["id"], old["status"], old["superseded_by"]), ("T-fixed.eta@r1", "superseded", "T-fixed.eta@r2"))
        self.assertEqual(new["supersedes"], "T-fixed.eta@r1")  # linked both ways
        self.assertTrue(new["simulated"] and old["simulated"])
        self.assertIn("simulated", [a["status"] for a in eta["actions"]])
        self.assertIsNone(eta["next_owner"])  # closed: nobody owes a response

        fixed = self.task(ids["fixed"]).json()
        self.assertEqual([c["id"] for c in fixed["children"]], [ids["eta"], ids["acceptance"]])
        self.assertEqual(fixed["join"]["status"], "complete")
        self.assertEqual(sorted(r["id"] for r in fixed["reports"]), ["T-fixed@r1", "T-fixed@r2"])
        for d in fixed["deliveries"] + denied["deliveries"]:  # names and states only: never the words, a thread or a person
            self.assertFalse({"text", "thread_id", "initiator", "receipt", "last_error"} & set(d), d)
            self.assertIn(d["from_team"], ("Business", "Data Science", "Model (eta)", "Model (acceptance)"))

    def test_another_projects_task_and_a_missing_one_get_the_same_refusal(self):
        ids = self.seed()
        other = self.app.state.services.projects.create("Elsewhere", goal="x")["id"]
        for answer in (self.task(ids["fixed"], pid=other), self.task("T-nothing")):
            self.assertEqual(answer.status_code, 403, answer.text)
            self.assertEqual(answer.json()["detail"], "Not your team's task")

    def test_an_agents_job_description_by_name_by_specialist_and_by_team_id(self):
        job = self.client.get("/api/team-agents/data_science").json()
        self.assertEqual(job["identity"], "Data Science team")
        self.assertTrue(any("approve a decision" in n for n in job["may_never"]))
        self.assertTrue(job["actions"] and all({"what", "by", "available"} <= set(a) for a in job["actions"]))
        self.assertEqual(self.client.get("/api/team-agents/model:eta").json()["name"], "model")
        from dclab_rnd.storage import open_teams

        team = open_teams(self.home).create_team("Business", "business")["id"]
        self.assertEqual(self.client.get(f"/api/team-agents/{team}").json()["name"], "business")
        self.assertEqual(self.client.get("/api/team-agents/accounting").status_code, 404)


class FileLocalTests(_Local, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}


# ---------------------------------------------------------------------------------------------------------- who may read a task
class SignedInTaskAccessTests(unittest.TestCase):
    """Sara in Business asks; Dan in Data Science answers; Mia in ML Engineering works a delegated task; Sam in Sales sees the history, never the task."""

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
                                                "DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS": "0"})
        cls.patch.start()
        from dclab_rnd.accounts.store import Accounts
        from dclab_rnd.storage import open_teams
        from dclab_rnd.teamspace import delivery

        cls.TestClient = TestClient
        cls.app = create_app(Path(tempfile.mkdtemp()))
        cls.main = TestClient(cls.app)
        cls.main.__enter__()
        accounts = Accounts()
        s = cls.app.state.services
        cls.services, cls.w1 = s, s.workspace_id
        cls.users = {}
        for name, role in (("sara", "data_scientist"), ("dan", "data_scientist"), ("sam", "data_scientist"), ("mia", "data_scientist"), ("boss", "owner")):
            user = accounts.create_user(f"{name}-{os.getpid()}-{os.urandom(3).hex()}@example.com", name.title(), password="a long enough password")
            accounts.set_member(cls.w1, user["id"], role)
            cls.users[name] = user
        teams = open_teams(s.store.home)
        cls.teams = {k: teams.create_team(w, k)["id"] for k, w in (("business", "Business"), ("data_science", "Data Science"), ("sales", "Sales"),
                                                                     ("ml_engineering", "ML Engineering"))}
        for who, team in (("sara", "business"), ("dan", "data_science"), ("sam", "sales"), ("mia", "ml_engineering")):
            teams.add_member(cls.teams[team], cls.users[who]["id"])
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
        r = client.post("/api/auth/password", json={"email": cls.users[name]["email"], "password": "a long enough password"},
                        headers={"X-DCLab-Token": anonymous})
        assert r.status_code == 200, r.text
        client.headers["X-DCLab-Token"] = r.json()["csrf"]
        return client

    def test_the_teams_on_the_chain_and_the_owner_read_a_task_and_another_team_does_not(self):
        from dclab_rnd.memory.scope import create_problem
        from dclab_rnd.teamspace import delivery

        s = self.services
        pid = s.projects.create("Renewal retention", goal="Who should Sales call")["id"]
        s.projects.update(pid, team_inputs=["sample:renewal_v1"])
        prob = create_problem(s.projects, pid, "Renewal", workspace_id=self.w1)["problem_id"]
        thread = self.clients["sara"].post(f"/api/projects/{pid}/problems/{prob}/team-threads", json={"team": self.teams["business"]}).json()["id"]
        r = self.clients["sara"].post(f"/api/projects/{pid}/team-threads/{thread}/messages", json={"text": BRIEF, "ask": "data_science"})
        self.assertEqual(r.status_code, 201, r.text)
        work = delivery.store_for(s)
        deadline = time.monotonic() + 30
        while work.deliveries(status="pending") and time.monotonic() < deadline:
            delivery.requeue_undelivered(s)
            time.sleep(0.1)
        [row] = [d for d in work.deliveries(project_id=pid) if d["kind"] == "task"]
        path = f"/api/projects/{pid}/team-tasks/{row['task_id']}"
        for who in ("sara", "dan", "boss"):
            answer = self.clients[who].get(path)
            self.assertEqual(answer.status_code, 200, (who, answer.text))
            self.assertEqual(answer.json()["task"]["question"], BRIEF)
        refused = self.clients["sam"].get(path)
        self.assertEqual((refused.status_code, refused.json()["detail"]), (403, "Not your team's task"))
        self.assertNotIn(BRIEF, refused.text)
        history = self.clients["sam"].get(f"/api/projects/{pid}/problems/{prob}/handoffs")
        self.assertEqual(history.status_code, 200)  # the shared history stays every reader's
        self.assertTrue(history.json()["lines"][0].startswith("1. Business → Data Science · "))

    def test_a_delegated_task_is_read_up_its_chain_never_down_it_nor_by_another_team(self):
        """Data Science delegates to ML Engineering: Sara (Business, who asked the parent) reads the child; Mia (ML
        Engineering, the child's owner) is refused the parent; Sam (Sales) is refused the child."""
        from datetime import datetime, timezone

        from dclab_core import team_work as tw
        from dclab_rnd.memory.scope import create_problem
        from dclab_rnd.teamspace import delivery

        s = self.services
        pid = s.projects.create("Renewal model", goal="Rank accounts")["id"]
        prob = create_problem(s.projects, pid, "Renewal model", workspace_id=self.w1)["problem_id"]
        work = delivery.store_for(s)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        sara = {"user_id": self.users["sara"]["id"], "role": "data_scientist", "workspace_id": self.w1}
        parent = tw.TeamTask(id="T-chain", workspace_id=work.workspace, project_id=pid, problem_id=prob, brief_revision=1,
                             requester=tw.Requester(team="business", member=sara["user_id"], thread_id="th_chain"), owner_team="data_science",
                             question="Can the data rank accounts before renewal?", expected_output="a feasibility answer",
                             done_criteria=["says whether the data can answer"], budget_eur=1.0, created_at=now)
        work.create_task(parent, initiator=sara)
        ds = tw.Actor(kind="agent", id="data_science", team="data_science")
        work.move(pid, "T-chain", "accepted", ds, initiator=sara)
        work.delegate(pid, "T-chain", id="T-chain.ml", owner_team="ml_engineering", question="Compare a ranking with the baseline.",
                      expected_output="a run reference and a comparison", done_criteria=["a run and its baseline"], by=ds,
                      thread_id="th_chain", created_at=now, initiator=sara)
        base = f"/api/projects/{pid}/team-tasks"
        child = self.clients["sara"].get(f"{base}/T-chain.ml")
        self.assertEqual(child.status_code, 200, child.text)
        self.assertEqual(child.json()["task"]["owner_label"], "ML Engineering")
        for who, task_id in (("mia", "T-chain"), ("sam", "T-chain.ml")):
            refused = self.clients[who].get(f"{base}/{task_id}")
            self.assertEqual((refused.status_code, refused.json()["detail"]), (403, "Not your team's task"), who)
        self.assertEqual(self.clients["mia"].get(f"{base}/T-chain.ml").status_code, 200)  # the child's own team reads it


# ---------------------------------------------------------------------------------------------------------- the page
class TeamPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = (SRC / "views" / "team.html").read_text(encoding="utf-8")
        cls.core = (SRC / "core.js").read_text(encoding="utf-8")

    def test_the_teams_tab_is_a_project_page_with_its_own_tab_and_a_sidebar_entry(self):
        from dclab_rnd.agentic.web import build

        self.assertIn("team", build.VIEWS)
        self.assertRegex(self.core, r"const PROJECT_VIEWS = \[[^\]]*'team'\]")
        self.assertIn("{ id: 'teams', label: 'Teams', views: [['team', 'Team threads']] }", self.core)
        self.assertIn("{ id: 'team', label: 'Team inbox', icon: 'users'", self.core)

    def test_every_read_goes_through_the_client_and_live_updates_through_the_stream(self):
        for call in ("myTeamThreads(", "readTeamThread(", "postTeamMessage(", "problemHandoffs(", "openTeamThread(", "readTeamTask(",
                     "teamAgentJob(", "path.teamThreadEvents("):
            self.assertIn("DC.client." + call, self.page)
        self.assertIn("DC.stream(", self.page)
        self.assertNotRegex(self.page, r"[`'\"]/api/")
        for state in ("DC.states.loading(", "DC.states.empty(", "DC.states.failed("):
            self.assertIn(state, self.page)

    def test_the_history_is_drawn_in_the_owners_words_and_simulated_is_always_marked(self):
        self.assertIn("h.title || `Shared handoff history · ${h.count || 0}`", self.page)
        self.assertIn("' · not delivered: '", self.page)  # the API's mark is kept, and shown
        self.assertIn('class="tm-lines"', self.page)  # the API's numbered lines, as they are (no second numbering)
        for status in ("simulated: ['proof', 'Simulated']", "denied: ['bad', 'Denied']", "failed: ['bad', 'Failed']"):
            self.assertIn(status, self.page)
        self.assertIn("'Replaces'", self.page)
        self.assertIn("'Replaced by'", self.page)
        self.assertNotRegex(self.page, r'(?<![-\w])style\s*=\s*["\']')

    def test_leaving_the_page_ends_its_stream_and_a_read_in_flight_opens_none(self):
        self.assertIn("if (DC.state.view !== 'team') { this.T.gen++; this.stop(); }", self.page)
        listen = self.page[self.page.index("  listen(el, thread) {"):]
        self.assertTrue(listen.split("\n")[1].strip().startswith("if (DC.state.view !== 'team') return;"), listen[:200])


class BrowserCheckTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "scripts" / "browser_e2e.py").exists(), "scripts/browser_e2e.py is not part of this edition")
    def test_the_browser_check_drives_the_four_screens(self):
        text = (ROOT / "scripts" / "browser_e2e.py").read_text(encoding="utf-8")
        body = re.search(r"def check_teamwork\(.*?\n(?=def |CHECKS)", text, flags=re.S).group(0)
        for part in ("#tm-compose", "tm-history-summary", "data-tm-report", "#tm-job", "team-1280", "team-375", "window.__streams"):
            self.assertIn(part, body)


if __name__ == "__main__":
    unittest.main()
