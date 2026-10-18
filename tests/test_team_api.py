"""Package 28.8: the teams in the product: a draft's understanding walk started and resumed through the API, its board and the
pause it waits on read without running anything, a resume with no pause refused, the project's solution team through its
route, and the Admin trace view of every walk (the owner's)."""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import langgraph  # noqa: F401
    from fastapi.testclient import TestClient

    from dclab_rnd.agentic.server import create_app
except ImportError as error:
    MISSING = str(error)
else:
    MISSING = ""


def wait(check, seconds=60.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        got = check()
        if got:
            return got
        time.sleep(0.2)
    raise AssertionError("timed out")


@unittest.skipIf(MISSING, MISSING)
class TeamApiTests(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1", "DCLAB_NO_LIVE_MODELS": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.home = Path(tempfile.mkdtemp())
        self.client = TestClient(create_app(self.home))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def test_a_drafts_walk_starts_pauses_on_a_question_and_resumes_with_the_answer(self):
        c = self.client
        d = c.post("/api/drafts", json={"problem": "Predict which customers cancel next month"}, headers=self.h).json()
        wait(lambda: c.get(f"/api/drafts/{d['id']}").json()["questions"])
        wait(lambda: not c.get(f"/api/drafts/{d['id']}").json().get("agent", {}).get("busy"))
        empty = c.get(f"/api/drafts/{d['id']}/team").json()
        self.assertEqual((empty["board"], empty["waiting"], empty["walk"]), (None, None, None))  # nothing walked yet
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/team", json={"answer": "x"}, headers=self.h).status_code, 409)  # no pause to resume
        out = self.start_walk(d["id"])
        self.assertEqual((out["status"], out["asks"]["kind"]), ("paused", "question"), out)
        seen = c.get(f"/api/drafts/{d['id']}/team").json()
        self.assertEqual((seen["waiting"]["kind"], seen["waiting"]["field"]), ("question", out["asks"]["field"]))  # the pause, read back
        nxt = c.post(f"/api/drafts/{d['id']}/team", json={"answer": "Whether the customer cancels within 30 days"}, headers=self.h).json()
        self.assertIn(nxt["status"], ("paused", "done"))
        self.assertGreaterEqual(c.get(f"/api/drafts/{d['id']}/team").json()["handoffs"], 1)  # the answer moved the walk

    def test_the_teams_switch_and_budgets_start_off_and_every_change_is_audited(self):  # 28.8
        c = self.client
        got = c.get("/api/team/settings").json()
        self.assertEqual(got["settings"], {"on": False, "run_eur": 0.5, "node_eur": 0.2, "max_steps": 40})
        for bad in ({"on": "yes"}, {"run_eur": 0}, {"node_eur": 9}, {"max_steps": 7.5}, {"colour": 1}):
            self.assertEqual(c.put("/api/team/settings", json=bad, headers=self.h).status_code, 422, bad)
        after = c.put("/api/team/settings", json={"on": True, "run_eur": 1.0}, headers=self.h).json()["settings"]
        self.assertEqual((after["on"], after["run_eur"], after["node_eur"]), (True, 1.0, 0.2))  # the other keys kept
        from dclab_rnd import audit

        rows = [r for r in audit.open_audit(self.home).query(500, 0)[1] if r.get("move") == "set_team"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["detail"]["args"]["changed"], {"on": True, "run_eur": 1.0})

    def test_a_walk_stops_at_its_euro_budget_and_at_a_nodes(self):  # 28.8
        c = self.client
        draft_id, _ = self.draft_waiting()
        from dclab_rnd.orchestra import walk as W

        c.put("/api/team/settings", json={"run_eur": 0.1}, headers=self.h)
        with mock.patch.object(W, "_spent", return_value=0.25):  # the walk's requests so far cost more than its budget
            out = c.post(f"/api/drafts/{draft_id}/team", json={"answer": "Whether the customer cancels within 30 days"}, headers=self.h).json()
        self.assertNotEqual(out.get("status"), "paused", out)  # stopped, not waiting for the person
        self.assertIn("euro budget", str(out.get("says") or ""), out)
        state = {"draft_id": draft_id, "steps": [{"action": "answer", "cost_eur": 0.5}], "runs": []}
        walk = W.Walk(mock.MagicMock(), None, max_steps=40, node_eur=0.2)
        walk.c.where.return_value = mock.MagicMock(next="answer", state="problem", says="", blockers=[], model_dump_json=lambda: "{}")
        walk.c.drafts.get.return_value = {"questions": []}
        walk.c.agent.pending.return_value = None
        with mock.patch("dclab_rnd.ledger.capture.team_node"):
            got = walk.situate(state)
        self.assertEqual(got["next"], "stop")
        self.assertIn("over its budget", got["says"])

    def test_a_walk_counts_its_nested_agents_requests_and_caps_each_node(self):  # the review of 28.8, F2 and F3
        from dclab_rnd import experience
        from dclab_rnd.orchestra import walk as W

        log = experience.FileLog(self.home / "experience-x", "w")
        experience.install(log)
        self.addCleanup(experience.install, None)
        with experience.bind(run="walkrun1", conversation="d1"):
            experience.record("model.request", "agent", {"purpose": "understanding_critic", "model": "m", "cost": 0.1, "outcome": "ok"})
            with experience.bind(parent="walkrun1"), experience.bind(run="d1", conversation="d1"):  # the Home agent's own run inside the walk
                experience.record("model.request", "agent", {"purpose": "home_agent", "model": "m", "cost": 0.4, "outcome": "ok"})
        self.assertAlmostEqual(W._spent("d1", ["walkrun1"]), 0.5)
        client = mock.MagicMock(spent_eur=0.3, run_limit_eur=None)
        advisor = mock.MagicMock(cap_eur=None)
        advisor.spent_eur.return_value = 0.05
        walk = W.Walk(mock.MagicMock(), None, max_steps=40, critic_advisor=advisor, node_eur=0.2)
        walk.c.agent.client = client
        walk._cap_node()
        self.assertAlmostEqual(client.run_limit_eur, 0.5)  # what it had spent, plus one node's budget
        self.assertAlmostEqual(advisor.cap_eur, 0.25)

    def test_a_node_is_capped_before_it_acts_and_its_nested_runs_cost_is_its_own(self):  # the second review of 28.8, finding 4
        from dclab_core import coordinator as machine
        from dclab_rnd import experience
        from dclab_rnd.orchestra import checkpoints
        from dclab_rnd.orchestra import walk as W

        log = experience.FileLog(self.home / "experience-n", "w")
        experience.install(log)
        self.addCleanup(experience.install, None)
        coordinator = mock.MagicMock()
        coordinator.agent.client = mock.MagicMock(spent_eur=0.1, run_limit_eur=None)
        coordinator.agent.pending.return_value = None
        coordinator.drafts.get.return_value = {"questions": [], "assets": []}
        limits = []
        coordinator.where.side_effect = lambda d: (machine.Situation(state="built", next="done", says="done") if limits else
                                                   machine.Situation(state="workflow", next="plan_workflow", says="draw it"))

        def act(draft_id, now, person):  # the Home agent's own run inside the node: its request carries the walk as its parent
            limits.append(coordinator.agent.client.run_limit_eur)
            with experience.bind(run="home-run", conversation=draft_id):
                experience.record("model.request", "agent", {"purpose": "home_agent", "model": "m", "cost": 0.07, "outcome": "ok"})
            return {"workflow": True}
        coordinator.act.side_effect = act
        with checkpoints.open_checkpointer(self.home, database_url="") as saver:
            out = W.Walk(coordinator, saver, home=self.home, max_steps=40, critic=False, node_eur=0.2, max_eur=5.0).run("d9")
        self.assertEqual(len(limits), 1)
        self.assertAlmostEqual(limits[0], 0.3)  # capped before acting: what it had spent, plus one node's budget
        step = next(s for s in out["steps"] if s["action"] == "plan_workflow")
        self.assertAlmostEqual(step["cost_eur"], 0.07)  # counted through the parent the walk bound, not the test

    def test_with_the_switch_on_home_builds_through_the_walk_and_says_what_it_asks_first(self):  # 28.8
        from dclab_rnd.orchestra import service as team

        drafts = mock.MagicMock()
        state = {"project_id": None}
        drafts.get.side_effect = lambda d: dict(state)
        calls = []

        def walk(home, ds, projects, agent, d, resume, gateway):
            calls.append(resume)
            if resume is None:
                return {"status": "paused", "asks": {"kind": "approval", "gate": "build", "key": "k-build"}}
            state["project_id"] = "p-built"
            return {"status": "done", "next": "done"}

        with mock.patch.object(team, "pending", return_value=None), mock.patch.object(team, "walk_draft", side_effect=walk):
            self.assertEqual(team.build_draft(self.home, drafts, None, None, "d1"), "p-built")
        self.assertEqual(calls, [None, {"key": "k-build", "answer": True}])  # started, then "build it" answered the build pause by its key
        state["project_id"] = None

        def taken(home, ds, projects, agent, d, resume, gateway):  # another request answered the pause first (the review, F5)
            state["project_id"] = "p-other"
            raise team.NothingToResume("nothing to resume")

        with mock.patch.object(team, "pending", return_value={"kind": "approval", "gate": "build", "key": "k"}), \
                mock.patch.object(team, "walk_draft", side_effect=taken):
            self.assertEqual(team.build_draft(self.home, drafts, None, None, "d1"), "p-other")
        state["project_id"] = None
        from dclab_rnd.studio.data import DataError

        with mock.patch.object(team, "pending", return_value={"kind": "approval", "gate": "build", "key": "k"}), \
                mock.patch.object(team, "walk_draft", side_effect=DataError("the table changed")):
            with self.assertRaises(DataError):  # the second review, finding 2: a real error inside the walk is never taken for a pause answered elsewhere
                team.build_draft(self.home, drafts, None, None, "d1")
        agent = mock.MagicMock()
        agent.pending.return_value = {"field": "target", "text": "Which column?"}  # still the open question: asked, never looked at again
        with mock.patch.object(team, "pending", return_value={"kind": "question", "field": "target", "text": "Which column?"}), \
                mock.patch.object(team, "walk_draft", side_effect=AssertionError("an open question is not resumed")):
            with self.assertRaises(team.TeamWaits) as caught:
                team.build_draft(self.home, drafts, None, agent, "d1")
        self.assertIn("Which column?", str(caught.exception))
        concerns = {"concerns": [{"field": "entity", "question": "What is one row?"}]}
        with mock.patch.object(team, "pending", return_value={"kind": "approval", "gate": "understanding", "summary": concerns}), \
                mock.patch.object(team, "walk_draft", side_effect=AssertionError("the critic's concerns are never approved for the person")):
            with self.assertRaises(team.TeamWaits) as caught:
                team.build_draft(self.home, drafts, None, agent, "d1")
        self.assertIn("What is one row?", str(caught.exception))  # the chat says what the critic asks, and where to decide
        state["project_id"] = None
        with mock.patch.object(team, "pending", return_value=None), \
                mock.patch.object(team, "walk_draft", return_value={"status": "done", "next": "stop", "says": "The team's euro budget was used."}):
            with self.assertRaises(team.TeamWaits) as caught:
                team.build_draft(self.home, drafts, None, None, "d1")
        self.assertEqual(caught.exception.rule, "team stopped")
        c = self.client  # through the route: with the switch on and no table, the build is refused before anything walks
        d = c.post("/api/drafts", json={"problem": "Predict which customers cancel next month"}, headers=self.h).json()
        c.put("/api/team/settings", json={"on": True}, headers=self.h)
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/build/run", headers=self.h).status_code, 409)

    def draft_waiting(self):
        c = self.client
        d = c.post("/api/drafts", json={"problem": "Predict which customers cancel next month"}, headers=self.h).json()
        wait(lambda: c.get(f"/api/drafts/{d['id']}").json()["questions"])
        wait(lambda: not c.get(f"/api/drafts/{d['id']}").json().get("agent", {}).get("busy"))
        out = self.start_walk(d["id"])
        self.assertEqual(out["asks"]["kind"], "question", out)
        return d["id"], out

    def start_walk(self, draft_id):
        """Start the walk once the Home agent's turn has ended: a 409 (the agent's job still listed, under a loaded machine)
        starts nothing, so it is asked again (the gate of 28.8 found the race)."""
        def start():
            r = self.client.post(f"/api/drafts/{draft_id}/team", json={}, headers=self.h)
            return None if r.status_code == 409 and "answering" in r.text else r.json()
        return wait(start)

    def test_an_answer_must_fit_its_pause_and_a_start_while_it_waits_keeps_the_walk(self):  # the review of 28.8, M2 and M3
        c = self.client
        did, out = self.draft_waiting()
        self.assertEqual(c.post(f"/api/drafts/{did}/team", json={"approve": False}, headers=self.h).status_code, 422)  # a yes or no is no answer to a question
        before = c.get(f"/api/drafts/{did}").json().get("understanding") or {}
        self.assertNotIn("False", str(before.get(out["asks"]["field"]) or ""))
        again = c.post(f"/api/drafts/{did}/team", json={}, headers=self.h).json()
        self.assertEqual(again["asks"], out["asks"])  # the same pause: the walk was not thrown away

    def test_an_empty_answer_leaves_no_stale_pause_on_the_page(self):  # the review of 28.8, M1
        c = self.client
        did, _ = self.draft_waiting()
        c.post(f"/api/drafts/{did}/team", json={"answer": ""}, headers=self.h)
        seen = c.get(f"/api/drafts/{did}/team").json()
        if seen["waiting"] is not None:  # another question may follow; never the one just skipped, unanswerable
            self.assertEqual(c.post(f"/api/drafts/{did}/team", json={"answer": "x"} if seen["waiting"]["kind"] == "question" else {"approve": True},
                                    headers=self.h).status_code, 200)

    def test_the_admin_trace_lists_the_walks_and_a_projects_team_route_answers(self):
        c = self.client
        from dclab_rnd.orchestra import service

        with mock.patch.object(service, "team_passed", return_value=False):
            walks = c.get("/api/teams").json()
        self.assertEqual((walks["count"], walks["team_serves_a_model"]), (0, False))
        self.assertEqual(c.get("/api/projects/nope/team").status_code, 404)

    def test_the_solution_team_through_its_route_and_the_holdout_is_the_gates_own_permission(self):  # the review of 28.8, H1
        from dclab_rnd.accounts import guard
        from dclab_rnd.accounts.principal import Principal
        from tests.test_studio_programme import make_project

        c = self.client
        projects = c.app.state.services.projects
        pid, _ = make_project(projects)
        out = c.post(f"/api/projects/{pid}/team", json={}, headers=self.h).json()
        self.assertEqual((out["status"], out["asks"]["gate"]), ("paused", "holdout"), out)
        seen = c.get(f"/api/projects/{pid}/team").json()
        self.assertEqual((seen["waiting"]["kind"], seen["waiting"]["gate"]), ("approval", "holdout"))
        self.assertIn("parts", seen["waiting"]["summary"])  # what the approval approves is shown
        engineer = Principal(role="data_scientist", workspace_id="w", user_id="u1", via="session")
        with mock.patch.object(guard, "current", return_value=engineer):
            self.assertEqual(c.post(f"/api/projects/{pid}/team", json={"approve": True}, headers=self.h).status_code, 403)
        self.assertEqual(int(projects.get(pid).get("holdout_uses") or 0), 0)  # nothing read
        done = c.post(f"/api/projects/{pid}/team", json={"approve": True}, headers=self.h).json()  # the owner (no accounts here)
        self.assertEqual(done["next"], "done", done)
        self.assertEqual(projects.get(pid)["holdout_uses"], 1)


@unittest.skipIf(MISSING, MISSING)
class ChatAnswersTests(unittest.TestCase):
    """The second review of 28.8, finding 1: with the switch on, a question answered in the chat (or a map confirmed on the
    page) while the walk waited is never asked again by "build it"; the walk reads the draft again and goes on."""

    def test_build_it_after_answering_in_the_chat_never_asks_the_answered_question_again(self):
        from tests.test_coordinator import ANSWERS, MISSING as COORD_MISSING, PROBLEM, dfr, upload

        if COORD_MISSING:
            self.skipTest(COORD_MISSING)
        from dclab_rnd import experience
        from dclab_rnd.draft import map as problem_map
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore
        from dclab_rnd.orchestra import service as team
        from dclab_rnd.studio import ProjectStore

        home = Path(tempfile.mkdtemp())
        experience.install(experience.FileLog(home / "experience", "c"))
        self.addCleanup(experience.install, None)
        drafts, projects = DraftStore(home / "drafts"), ProjectStore(home / "projects")
        agent = HomeAgent(drafts, None)
        replica = dfr.generate(seed=0, n_orders=6000, n_bikers=240)
        orders = drafts.create(PROBLEM)["id"]
        agent.start(orders)
        upload(drafts, agent, orders, "part2-dataset.csv", replica.rides)
        draft = drafts.create(PROBLEM)["id"]
        agent.start(draft)
        upload(drafts, agent, draft, "part2-fraud-sample.csv", replica.sample.assign(final_status="DELIVERED"))
        first = team.walk_draft(home, drafts, projects, agent, draft)  # the Team panel started the walk: it waits on a question
        self.assertEqual((first["status"], first["asks"]["kind"]), ("paused", "question"))
        answered, project_id = [], None
        for _ in range(30):
            try:
                project_id = team.build_draft(home, drafts, projects, agent, draft)  # "build it"
                break
            except team.TeamWaits as waits:
                asks = team.pending(home, f"draft:{draft}") or {}
                if asks.get("kind") == "question":
                    self.assertNotIn((asks.get("field"), asks.get("text")), answered, str(waits))  # an answered question is never asked again
                    answered.append((asks.get("field"), asks.get("text")))
                    agent.reply(draft, ANSWERS.get(asks.get("field")) or "I don't know")  # answered in the chat, not on the panel
                elif asks.get("kind") == "confirmation":
                    self.assertIn("confirm the map", str(waits))
                    problem_map.edit(drafts, draft, "confirm", who="person")  # confirmed on the page
                else:
                    self.fail(f"the walk waits on {asks}")
        self.assertTrue(project_id, "the walk never built")
        self.assertEqual(drafts.get(draft)["project_id"], project_id)


if __name__ == "__main__":
    unittest.main()
