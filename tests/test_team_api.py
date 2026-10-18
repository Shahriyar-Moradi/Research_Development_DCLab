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
        out = c.post(f"/api/drafts/{d['id']}/team", json={}, headers=self.h).json()
        self.assertEqual((out["status"], out["asks"]["kind"]), ("paused", "question"), out)
        seen = c.get(f"/api/drafts/{d['id']}/team").json()
        self.assertEqual((seen["waiting"]["kind"], seen["waiting"]["field"]), ("question", out["asks"]["field"]))  # the pause, read back
        nxt = c.post(f"/api/drafts/{d['id']}/team", json={"answer": "Whether the customer cancels within 30 days"}, headers=self.h).json()
        self.assertIn(nxt["status"], ("paused", "done"))
        self.assertGreaterEqual(c.get(f"/api/drafts/{d['id']}/team").json()["handoffs"], 1)  # the answer moved the walk

    def draft_waiting(self):
        c = self.client
        d = c.post("/api/drafts", json={"problem": "Predict which customers cancel next month"}, headers=self.h).json()
        wait(lambda: c.get(f"/api/drafts/{d['id']}").json()["questions"])
        wait(lambda: not c.get(f"/api/drafts/{d['id']}").json().get("agent", {}).get("busy"))
        out = c.post(f"/api/drafts/{d['id']}/team", json={}, headers=self.h).json()
        self.assertEqual(out["asks"]["kind"], "question")
        return d["id"], out

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


if __name__ == "__main__":
    unittest.main()
