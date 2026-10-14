"""Packages 19.2 and 19.3: the studio records every session in one log. One flow through the real API, from the
problem's first sentence on Home to a stage run in the worker and a question to the project's agent, comes back as
one ordered timeline of one visit, every event carrying the ids it should; a second browser is a second visit; an
event the contract refuses is dropped without stopping the person's work; a model's prompt and reply are kept only
when the workspace keeps them. The same test runs on PostgreSQL under ``make test-pg``."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd
    from fastapi.testclient import TestClient

    from dclab_core import experience as contract
    from dclab_rnd import experience
    from dclab_rnd.agentic.server import VISIT_COOKIE, create_app
    from tests.test_draft_flow import wait
except ImportError as error:  # the agent environment has no pandas
    MISSING = str(error)
else:
    MISSING = ""

PROBLEM = "Predict which orders arrive late, at the moment the order is placed"


def orders(n=400, seed=0):
    rng = np.random.default_rng(seed)
    distance = rng.uniform(0.5, 12, n)
    return pd.DataFrame({"order_id": np.arange(10_000, 10_000 + n), "created_at": pd.date_range("2025-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M"),
                         "distance_km": distance.round(2), "fare": (20 + distance * 3 + rng.normal(0, 2, n)).round(1),
                         "late": (rng.random(n) < 0.15 + 0.04 * distance).astype(int)})


# Every route that changes state, and what it leaves in the record (19.3). A route added to the API without a line here
# fails `test_every_write_route_says_what_it_records`: each one is a decision, never an omission.
EVENTS, NONE = "events", "none"
WRITE_ROUTES = {
    "POST /api/drafts": (EVENTS, ["draft.created"]),
    "POST /api/ledger/blind/{record_id}": (NONE, "a blind label is the ledger's own record (decision_label), not a session event"),
    "POST /api/feedback": (EVENTS, ["feedback.given"]),
    "PUT /api/consent": (EVENTS, ["decision.accepted"]),  # the switch, audited as a policy
    "PATCH /api/drafts/{draft_id}": (EVENTS, ["draft.edited"]),
    "DELETE /api/drafts/{draft_id}": (NONE, "the deletion is audited (its tombstone) and what the draft left is removed from the record (19.10)"),
    "PUT /api/drafts/{draft_id}/data": (EVENTS, ["data.uploaded", "data.cleaned", "data.analysed", "run.recorded"]),  # a label file: no analysis of its own
    "POST /api/drafts/{draft_id}/data/sample": (EVENTS, ["data.uploaded"]),
    "POST /api/drafts/{draft_id}/data/from-draft": (EVENTS, ["data.uploaded"]),  # 21.3: another draft's table, brought here
    "POST /api/drafts/{draft_id}/data/synthetic": (EVENTS, ["data.uploaded"]),
    "POST /api/drafts/{draft_id}/data/kaggle": (EVENTS, ["data.uploaded"]),
    "POST /api/drafts/{draft_id}/data/hf": (EVENTS, ["data.uploaded"]),
    "POST /api/drafts/{draft_id}/data/database": (EVENTS, ["data.uploaded"]),
    "POST /api/drafts/{draft_id}/data/cloud": (EVENTS, ["data.uploaded"]),
    "POST /api/drafts/{draft_id}/messages": (EVENTS, ["chat.person", "chat.agent"]),
    "POST /api/drafts/{draft_id}/pack": (NONE, "the pack is in the draft and in the next solution.proposed's kind; no kind of its own yet"),
    "POST /api/drafts/{draft_id}/solution/proposal": (EVENTS, ["solution.proposed"]),
    "PUT /api/drafts/{draft_id}/solution": (EVENTS, ["solution.saved"]),
    "PUT /api/drafts/{draft_id}/settings": (EVENTS, ["decision.accepted"]),
    "POST /api/drafts/{draft_id}/build": (EVENTS, ["draft.built"]),  # or draft.refused, for a list of cases alone
    "POST /api/projects": (NONE, "an empty project; its data, solution and stages are events"),
    "PATCH /api/projects/{project_id}": (EVENTS, ["decision.accepted"]),  # settings, gates, the opt-in; a name or a goal leaves none
    "DELETE /api/projects/{project_id}": (NONE, "the deletion is audited (its tombstone) and the project's events are removed (19.10)"),
    "DELETE /api/experience/sessions/{visit}": (NONE, "the session's events are removed; the audit keeps its tombstone (19.10)"),
    "PUT /api/projects/{project_id}/data": (EVENTS, ["data.uploaded"]),
    "PUT /api/projects/{project_id}/tables/{name}": (EVENTS, ["data.uploaded"]),  # 16.15 C: a related table, its audit row a data import
    "POST /api/projects/{project_id}/pipeline/rerun": (EVENTS, ["job.started", "run.recorded"]),  # 20.8: a step rerun as a job
    "POST /api/projects/{project_id}/data/sample": (EVENTS, ["data.uploaded"]),
    "PUT /api/projects/{project_id}/solution": (EVENTS, ["graph.move", "solution.saved"]),
    "POST /api/projects/{project_id}/solution/proposal": (NONE, "the project page's audit is shown, not saved: the saved solution is the event"),
    "POST /api/projects/{project_id}/stages/{stage}/run": (EVENTS, ["job.started", "stage.started", "stage.finished", "run.recorded", "job.stopped"]),
    "POST /api/projects/{project_id}/run": (EVENTS, ["job.started", "stage.started", "stage.finished", "run.recorded", "job.stopped"]),
    "POST /api/projects/{project_id}/stages/{stage}/approve": (EVENTS, ["graph.move", "stage.approved"]),
    "POST /api/projects/{project_id}/approvals": (EVENTS, ["decision.accepted"]),
    "POST /api/projects/{project_id}/review": (EVENTS, ["graph.move"]),  # 20.7: confirmed alerts become labels, audited as a move
    "POST /api/projects/{project_id}/ask": (EVENTS, ["chat.person", "chat.agent"]),
    "POST /api/projects/{project_id}/graph/check": (NONE, "reads whether a move would be allowed; changes nothing"),
    "POST /api/projects/{project_id}/lessons": (NONE, "proposes lessons for review; the review is the event"),
    "POST /api/projects/{project_id}/review/fixes": (NONE, "a notebook review's fixes, written by a model: its model.request is the event"),
    "DELETE /api/projects/{project_id}/memory/{note_id}": (EVENTS, ["decision.overridden"]),
    "POST /api/lessons/{lesson_id}/review": (EVENTS, ["decision.accepted"]),
    "POST /api/intern/sessions": (EVENTS, ["chat.person", "agent.step", "chat.agent"]),
    "POST /api/intern/sessions/{session_id}/message": (EVENTS, ["chat.person", "chat.agent"]),
    "DELETE /api/intern/sessions/{session_id}": (NONE, "the session's events stay until retention (19.6)"),
    "POST /api/jobs/{job_id}/stop": (EVENTS, ["job.stopped"]),
    "POST /api/jobs/{job_id}/retry": (EVENTS, ["job.started", "job.stopped"]),
    "POST /api/models/routing": (EVENTS, ["decision.accepted"]),
    "POST /api/evidence/ask": (NONE, "a search of the evidence library; changes nothing"),
    "POST /api/connectors/kaggle/search": (NONE, "a search; changes nothing"),
    "POST /api/runs": (NONE, "the legacy research campaign: its own ledger (research.sqlite3)"),
    "POST /api/runs/{run_id}/pause": (NONE, "the legacy research campaign"),
    "POST /api/runs/{run_id}/resume": (NONE, "the legacy research campaign"),
    "POST /api/auth/password": (NONE, "account administration: audited (sign_in, role_change), not a step of solving"),
    "POST /api/auth/signout": (NONE, "account administration"),
    "POST /api/auth/tokens": (NONE, "account administration"),
    "DELETE /api/auth/tokens/{token_id}": (NONE, "account administration"),
    "POST /api/auth/workspace": (NONE, "the next request's workspace; its events are that workspace's"),
    "POST /api/workspace/members": (NONE, "account administration"),
    "PATCH /api/workspace/members/{user_id}": (NONE, "account administration"),
    "DELETE /api/workspace/members/{user_id}": (NONE, "account administration"),
}


@unittest.skipIf(MISSING, MISSING)
class OneSessionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1", "DCLAB_LOG_PROMPTS": ""})
        cls.env.start()
        cls.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        cls.client.__enter__()
        c, h = cls.client, {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}
        cls.h = h
        d = c.post("/api/drafts", json={"problem": PROBLEM}, headers=h).json()
        cls.draft = d["id"]
        c.put(f"/api/drafts/{d['id']}/data?filename=orders.csv", content=orders().to_csv(index=False).encode(), headers=h)
        wait(lambda: any(a["status"] == "ready" for a in c.get(f"/api/drafts/{d['id']}").json()["assets"]))
        proposal = c.post(f"/api/drafts/{d['id']}/solution/proposal", json={"target": "late"}, headers=h).json()
        solution = {"target": "late", "task": "binary", "positive_label": "1", "prediction_moment": "When the order is placed, before dispatch",
                    "forbidden": [], "identifiers": ["order_id"], "time_column": "created_at", "group_column": None, "text_columns": [],
                    "metric": proposal["metric"], "notes": ""}
        assert c.put(f"/api/drafts/{d['id']}/solution", json=solution, headers=h).status_code == 200
        c.put(f"/api/drafts/{d['id']}/settings", json={"split": "time", "folds": 3, "quick": True, "max_rows": 400, "budget": {}}, headers=h)
        built = c.post(f"/api/drafts/{d['id']}/build", json={}, headers=h)
        assert built.status_code == 201, built.text
        cls.project = built.json()["id"]
        ran = c.post(f"/api/projects/{cls.project}/stages/data/run?wait=true", headers=h)
        assert ran.status_code == 200, ran.text
        approved = c.post(f"/api/projects/{cls.project}/stages/data/approve", json={}, headers=h)
        assert approved.status_code == 200, approved.text
        asked = c.post(f"/api/projects/{cls.project}/ask", json={"question": "Which split did the data stage use?"}, headers=h)
        assert asked.status_code == 200, asked.text
        cls.edited = c.post("/api/drafts", json={"problem": "Find the fraud"}, headers=h).json()["id"]
        assert c.patch(f"/api/drafts/{cls.edited}", json={"problem": "Find the fake deliveries among the orders"}, headers=h).status_code == 200
        # what went wrong the owner's morning, recorded: a key refused as the outcome, a list of cases built alone
        assert c.post(f"/api/drafts/{d['id']}/solution/proposal", json={"target": "order_id"}, headers=h).status_code == 422
        from dclab_core.replicas import delivery_fraud as dfr

        cases = dfr.generate(seed=0, n_orders=6000, n_bikers=240).sample.assign(final_status="DELIVERED")
        cls.cases_draft = c.post("/api/drafts", json={"problem": "I want build anomaly detection"}, headers=h).json()["id"]
        c.put(f"/api/drafts/{cls.cases_draft}/data?filename=known.csv", content=cases.to_csv(index=False).encode(), headers=h)
        wait(lambda: any(a["status"] == "ready" for a in c.get(f"/api/drafts/{cls.cases_draft}").json()["assets"]))
        assert c.post(f"/api/drafts/{cls.cases_draft}/build", json={}, headers=h).status_code == 409
        with mock.patch("dclab_rnd.studio.agent.answer", side_effect=RuntimeError("boom")):  # a route that raises
            cls.raised = c.post(f"/api/projects/{cls.project}/ask", json={"question": "Again?"}, headers=h)
        cls.visit = c.cookies.get(VISIT_COOKIE)
        cls.services = cls.client.app.state.services
        cls.events = cls.services.experience.read()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.env.stop()

    def kinds(self, events=None):
        return [e.kind for e in (events if events is not None else self.events)]

    def test_the_flow_is_one_visit_in_order(self):
        self.assertRegex(self.visit or "", r"^v[0-9a-f]{16}$")
        self.assertEqual({e.ids.visit for e in self.events}, {self.visit})
        kinds = self.kinds()
        for kind in ("visit.started", "draft.created", "data.uploaded", "data.analysed", "solution.proposed", "solution.saved",
                     "draft.built", "graph.move", "job.started", "stage.started", "stage.finished", "job.stopped", "chat.person", "chat.agent"):
            self.assertIn(kind, kinds, kind)
        order = [kinds.index(k) for k in ("visit.started", "draft.created", "data.uploaded", "solution.proposed", "draft.built", "stage.started")]
        self.assertEqual(order, sorted(order))  # in the order it happened
        self.assertEqual(kinds.count("visit.started"), 1)

    def test_every_write_route_says_what_it_records(self):
        spec = self.client.get("/openapi.json").json()
        routes = {f"{m.upper()} {p}" for p, ops in spec["paths"].items() for m in ops if m in ("post", "put", "patch", "delete")}
        self.assertEqual(routes, set(WRITE_ROUTES), "a write route must say what it records (or why nothing)")
        for route, (what, detail) in WRITE_ROUTES.items():
            if what == EVENTS:
                self.assertTrue(set(detail) <= set(contract.KINDS), route)
            else:
                self.assertTrue(len(detail) > 20, route)  # a reason, not a shrug
        exercised = ("POST /api/drafts", "PATCH /api/drafts/{draft_id}", "PUT /api/drafts/{draft_id}/data",
                     "POST /api/drafts/{draft_id}/solution/proposal", "PUT /api/drafts/{draft_id}/solution", "POST /api/drafts/{draft_id}/build",
                     "POST /api/projects/{project_id}/stages/{stage}/run", "POST /api/projects/{project_id}/stages/{stage}/approve",
                     "POST /api/projects/{project_id}/ask")
        kinds = set(self.kinds())
        for route in exercised:  # the flow above went through these: their events are in the record
            self.assertTrue(set(WRITE_ROUTES[route][1]) <= kinds, (route, set(WRITE_ROUTES[route][1]) - kinds))
        edit = next(e for e in self.events if e.kind == "draft.edited")
        self.assertEqual((edit.ids.conversation, edit.payload["before"], edit.payload["after"]),
                         (self.edited, "Find the fraud", "Find the fake deliveries among the orders"))

    def test_every_event_carries_the_ids_it_should(self):
        workspace = {e.ids.workspace for e in self.events}
        self.assertEqual(len(workspace), 1)
        for e in self.events:
            if e.kind.startswith(("draft.", "data.uploaded", "data.analysed", "solution.proposed")) and e.ids.project is None:  # Home, not the project's import
                self.assertIn(e.ids.conversation, (self.draft, self.edited, self.cases_draft), e.kind)
            if e.kind.startswith("stage.") or (e.kind.startswith("job.") and e.payload.get("kind") == "stage"):
                self.assertEqual(e.ids.project, self.project, e.kind)
            if e.kind.startswith("job.") and e.payload.get("kind") == "pipeline":  # a draft's data pipeline
                self.assertIn(e.ids.conversation, (self.draft, self.cases_draft), e.kind)
        stage = [e for e in self.events if e.kind in ("job.started", "stage.started", "stage.finished", "job.stopped")]
        self.assertTrue(stage and all(e.ids.job for e in stage))  # the worker's events know their job …
        self.assertTrue(all(e.ids.request for e in stage))  # … and the request that queued it
        self.assertTrue(all(e.ids.request for e in self.events if e.kind in ("draft.created", "solution.saved", "draft.built")))
        finished = next(e for e in self.events if e.kind == "stage.finished")
        self.assertEqual(finished.payload["stage"], "data")
        stopped = next(e for e in self.events if e.kind == "job.stopped")
        self.assertEqual(stopped.payload["status"], "done")

    def test_the_sequence_is_exactly_the_flow_on_either_store(self):
        """Every event of the main draft and its project, in order (the Home agent's replies, which come from a background
        turn, left out). This runs on the file store and, under make test-pg, on PostgreSQL: both must give this."""
        mine = [(e.kind, e.payload.get("stage") or e.payload.get("point") or e.payload.get("move") or e.payload.get("surface") or "")
                for e in self.events if ((e.ids.conversation == self.draft or e.ids.project == self.project) and e.kind != "chat.agent"
                                         or (e.kind == "chat.agent" and e.ids.project == self.project))
                and not (e.kind.startswith("job.") and e.payload.get("kind") == "pipeline")  # a background job's end races the next request
                and not (e.kind == "run.recorded" and e.payload.get("run_kind") in ("agent", "pipeline"))  # background: they race the next request
                and not (e.kind in ("agent.step", "tool.call") and e.ids.run == self.draft)]  # the Home agent's moves: a background turn too
        self.assertEqual(mine, [
            ("draft.created", ""), ("data.uploaded", ""), ("data.cleaned", ""), ("data.analysed", ""), ("solution.proposed", ""), ("solution.saved", ""), ("decision.accepted", "settings"), ("graph.move", "set_solution"),
            ("draft.built", ""), ("job.started", ""), ("stage.started", "data"), ("stage.finished", "data"), ("run.recorded", ""), ("graph.move", "run_stage"),
            ("job.stopped", ""), ("graph.move", "approve_stage"), ("stage.approved", "data"), ("chat.person", "project"), ("chat.agent", "project"),
            ("solution.refused", "")])  # the raising /ask failed before its question was logged; its error is the visit's, not the project's

    def test_the_background_runs_are_on_the_record(self):
        pipeline = [e for e in self.events if e.kind == "run.recorded" and e.payload.get("run_kind") == "pipeline" and e.ids.conversation == self.draft]
        self.assertEqual([e.payload["status"] for e in pipeline], ["completed"])  # the one upload, read and cleaned
        analysed = next(e for e in self.events if e.kind == "data.analysed" and e.ids.conversation == self.draft)
        self.assertEqual(analysed.ids.run, pipeline[0].ids.run)  # the pipeline's events carry its run

    def test_the_home_agents_scripted_moves_are_steps_of_its_conversation(self):
        # 21.1: with no model, the Home agent's questions are steps of the harness like a model's, in its conversation
        moves = [e for e in self.events if e.kind == "agent.step" and e.ids.conversation == self.draft]
        self.assertTrue(moves and all(e.payload["agent"] == "home.script" for e in moves))
        self.assertIn("ask_user", [e.payload["tool"] for e in moves])
        self.assertTrue(all(e.ids.model_request is None for e in moves))  # no model chose them

    def test_one_action_is_one_event(self):
        kinds = [e.kind for e in self.events if e.ids.conversation == self.draft or e.ids.project == self.project]
        self.assertEqual(kinds.count("solution.saved"), 1)  # saved once in the wizard; the build's copy is not a save
        uploads = [e for e in self.events if e.kind == "data.uploaded" and e.ids.conversation == self.draft]
        self.assertEqual(len(uploads), 1)  # not again from the audit, nor from the build's copy
        settings = [e for e in self.events if e.kind == "decision.accepted" and e.payload.get("point") == "settings"]
        self.assertTrue(settings and settings[0].payload["choice"].get("max_rows") == 400)

    def test_what_went_wrong_is_in_the_record(self):
        refused = next(e for e in self.events if e.kind == "solution.refused")
        self.assertEqual((refused.payload["target"], refused.ids.conversation), ("order_id", self.draft))
        cases = [e for e in self.events if e.ids.conversation == self.cases_draft]
        self.assertIn("data.cases_found", [e.kind for e in cases])
        self.assertIn("draft.refused", [e.kind for e in cases])
        self.assertEqual(self.raised.status_code, 500)
        error = next(e for e in self.events if e.kind == "error.raised")
        self.assertEqual((error.payload["code"], error.ids.visit), ("RuntimeError", self.visit))
        self.assertIn(VISIT_COOKIE, self.raised.headers.get("set-cookie", ""))  # the browser keeps its visit through an error

    def test_the_project_agent_answer_is_kept_and_the_card_sums_the_visit(self):
        answer = [e for e in self.events if e.kind == "chat.agent" and e.payload.get("surface") == "project"]
        self.assertTrue(answer and answer[0].payload["text"])  # dropped before 19.3: only the question was logged
        card = contract.session_card(self.events)
        self.assertEqual((card.visit, card.outcome, card.problem), (self.visit, "ran", PROBLEM))
        self.assertIn(self.project, card.projects)
        self.assertIn(self.draft, card.conversations)
        self.assertTrue(experience.lines(self.events))
        self.assertIn(self.visit, [c.visit for c in experience.sessions(self.services.experience)])

    def test_a_second_browser_is_a_second_visit_and_probes_are_none(self):
        other = TestClient(self.client.app)
        h = {"X-DCLab-Token": other.get("/api/config").json()["csrf"]}
        other.post("/api/drafts", json={"problem": "Forecast tomorrow's demand per store"}, headers=h)
        second = other.cookies.get(VISIT_COOKIE)
        self.assertTrue(second and second != self.visit)
        self.assertNotIn("set-cookie", {k.lower() for k in self.client.get("/healthz").headers})
        before = len(self.services.experience.visits())
        for _ in range(3):  # a client that keeps no cookie, polling: no visit is recorded until something happens
            TestClient(self.client.app).get("/api/config")
        self.assertEqual(len(self.services.experience.visits()), before)
        self.assertIn(second, self.services.experience.visits())

    def test_a_refused_event_never_stops_the_work(self):
        before = experience.DROPPED["count"]
        r = self.client.post("/api/drafts", json={"problem": "Predict churn; our key is " + "sk-" + "proj-abcdefghijklmnopqrstuvwxyz0123"}, headers=self.h)
        self.assertEqual(r.status_code, 201)  # the draft exists; its event, holding a credential, was not recorded
        self.assertGreaterEqual(experience.DROPPED["count"], before + 1)  # the Home agent's echo of it is refused too, later
        self.assertFalse(any("sk-proj" in e.model_dump_json() for e in self.services.experience.read()))


@unittest.skipIf(MISSING, MISSING)
class AgentsTest(unittest.TestCase):
    def test_an_intern_turn_is_recorded_with_its_steps(self):
        from dclab_rnd.intern import Intern, SessionStore
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.studio import ProjectStore, data as sd

        home = Path(tempfile.mkdtemp())
        log = experience.FileLog(home / "experience", "wtest")
        projects = ProjectStore(home / "projects")
        project = projects.create("Late orders", "general", "late orders")
        orders().to_csv(projects.data_dir(project["id"]) / "orders.csv", index=False)
        sd.attach_data(projects, project["id"], "orders.csv")
        intern = Intern(SessionStore(home / "sessions"), Toolbox(projects), None, traces=None)
        experience.install(log)
        try:
            with experience.bind(visit="v00000000000000aa"):
                session = intern.start("Predict whether an order is late; the outcome is the late column", project_id=project["id"])
                intern.run(session["id"])
        finally:
            experience.install(None)
        events = log.read(visit="v00000000000000aa")
        kinds = [e.kind for e in events]
        self.assertEqual(kinds[0], "chat.person")
        self.assertIn("tool.call", kinds)
        self.assertIn("agent.step", kinds)
        self.assertIn(kinds[-1], ("chat.agent", "error.raised"))
        self.assertTrue(all(e.ids.conversation == session["id"] for e in events if e.kind in ("chat.person", "chat.agent", "agent.step")))

    def test_a_home_turn_through_the_gateway_joins_its_model_request_to_its_step(self):
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore
        from dclab_rnd.models.usage import FileUsage
        from tests.test_models_gateway import CLEAN, gateway as scripted

        home = Path(tempfile.mkdtemp())
        log = experience.FileLog(home / "experience", "wtest")
        store = DraftStore(home / "drafts")
        with mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-never-logged", "OPENAI_MODEL": "std-model"}):
            call = {"id": "c1", "name": "ask_user", "arguments": {"field": "target", "question": "Which deliveries count as fake?",
                                                                   "why": "It is the outcome.", "options": ["Confirmed fakes", "No label"]}}
            plan = [{"content": "I will ask what counts as fake.", "tool_calls": [call], "usage": {"input_tokens": 50, "output_tokens": 9},
                     "assistant_message": {"role": "assistant", "content": "I will ask what counts as fake.", "tool_calls": [
                         {"id": "c1", "type": "function", "function": {"name": "ask_user", "arguments": '{"field": "target"}'}}]}}]
            gw, _ = scripted(FileUsage(home / "usage.jsonl"), plan)
            experience.install(log)
            try:
                with experience.bind(visit="v00000000000000bb"):
                    d = store.create("I want build anomaly detection")
                    HomeAgent(store, None).start(d["id"])
                    HomeAgent(store, gw.client("home_agent", draft_id=d["id"])).reply(d["id"], "it is anomaly detection")
            finally:
                experience.install(None)
        events = log.read(visit="v00000000000000bb")
        request = next(e for e in events if e.kind == "model.request")
        step = next(e for e in events if e.kind == "agent.step" and e.payload.get("tool") == "ask_user" and e.payload["agent"] == "home")
        script = next(e for e in events if e.kind == "agent.step" and e.payload["agent"] == "home.script")  # the opening question, no model
        self.assertIsNone(script.ids.model_request)
        self.assertTrue(request.ids.model_request and request.ids.model_request == step.ids.model_request)
        self.assertEqual(step.payload["reply"], "I will ask what counts as fake.")  # the model's own words, kept
        self.assertEqual(request.ids.conversation, d["id"])


@unittest.skipIf(MISSING, MISSING)
class ModelTextTest(unittest.TestCase):
    def test_a_prompt_and_a_reply_are_kept_only_when_the_workspace_keeps_them(self):
        from dclab_rnd.experience import hooks

        log = experience.FileLog(Path(tempfile.mkdtemp()), "wtest")
        entry = {"purpose": "home_agent", "model": "m", "input_tokens": 10, "output_tokens": 2, "cost_eur": 0.001, "seconds": 0.5, "outcome": "ok"}
        messages = [{"role": "user", "content": "I want build anomaly detection"}]
        experience.install(log)
        try:
            with experience.bind(visit="v0000000000000001"):
                with mock.patch.dict(os.environ, {"DCLAB_LOG_PROMPTS": ""}):
                    hooks.from_model_request(entry, messages, "ask_user(target)")
                with mock.patch.dict(os.environ, {"DCLAB_LOG_PROMPTS": "1"}):
                    hooks.from_model_request(entry, messages, "ask_user(target)")
        finally:
            experience.install(None)
        first, second = log.read()
        self.assertNotIn("prompt", first.payload)
        self.assertNotIn("reply", first.payload)
        self.assertEqual((first.payload["cost"], first.payload["currency"]), (0.001, "EUR"))
        self.assertIn("anomaly detection", second.payload["prompt"])
        self.assertEqual(second.payload["reply"], "ask_user(target)")
        self.assertEqual(contract.redact(second, contract.SHAREABLE).redacted, ["prompt:model_text", "reply:model_text"])

    def test_a_replayed_activity_entry_is_not_recorded_again(self):
        from dclab_rnd.studio import ProjectStore

        log = experience.FileLog(Path(tempfile.mkdtemp()), "wtest")
        store = ProjectStore(Path(tempfile.mkdtemp()))
        project = store.create("p", "general", "g")
        experience.install(log)
        try:
            store.log(project["id"], "stage_started", {"stage": "data"})  # fresh: recorded
            store.log(project["id"], "stage_started", {"stage": "data"}, at="2026-01-01T00:00:00+00:00")  # an import of history: not again
        finally:
            experience.install(None)
        self.assertEqual([e.kind for e in log.read()], ["stage.started"])

    def test_outside_a_workspace_nothing_is_written(self):
        experience.install(None)
        self.assertIsNone(experience.record("chat.person", "person", {"text": "hello"}))


if __name__ == "__main__":
    unittest.main()
