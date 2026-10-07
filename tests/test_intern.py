"""The DCLab intern: the toolbox, the standard plan, the LLM loop with a scripted model, budgets and the API."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.expansion.runner import fast_profile  # noqa: E402
from dclab_rnd.intern import Intern, SessionStore  # noqa: E402
from dclab_rnd.intern.tools import Toolbox  # noqa: E402
from dclab_rnd.studio import ProjectStore, sft  # noqa: E402


class ScriptedModel:
    """Replays tool calls the way a chat model would, substituting the project id once it exists."""

    model = "scripted"

    def __init__(self, script):
        self.script, self.turn, self.pid = script, 0, None

    def complete(self, messages, tools):
        self.turn += 1
        last = messages[-1]
        if last["role"] == "tool":
            try:
                data = json.loads(last["content"])
            except json.JSONDecodeError:
                data = {}
            if isinstance(data, dict) and data.get("project_id"):
                self.pid = data["project_id"]
        if self.turn > len(self.script):
            return {"content": "All done.", "tool_calls": [], "finish_reason": "stop", "usage": {"input_tokens": 1, "output_tokens": 1},
                    "assistant_message": {"role": "assistant", "content": "All done."}}
        name, args = self.script[self.turn - 1]
        args = {k: (self.pid if v == "PID" else v) for k, v in args.items()}
        call = {"id": f"c{self.turn}", "name": name, "arguments": args}
        return {"content": "", "tool_calls": [call], "finish_reason": "tool_calls", "usage": {"input_tokens": 10, "output_tokens": 5},
                "assistant_message": {"role": "assistant", "content": "", "tool_calls": [{"id": call["id"], "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}


BIKE_SCRIPT = [
    ("write_plan", {"plan": "1 load the bike sample 2 solution 3 stages 4 report"}),
    ("list_samples", {}),
    ("create_project", {"name": "Bike demand", "goal": "forecast daily rentals", "industry": "logistics and delivery"}),
    ("use_sample", {"project_id": "PID", "key": "bike_sharing_daily"}),
    ("set_solution", {"project_id": "PID", "target": "cnt", "task": "regression", "prediction_moment": "Forecast two days ahead; same-day rider counts are unknown.",
                      "forbidden": [{"column": "casual", "reason": "post-outcome"}, {"column": "registered", "reason": "post-outcome"}], "identifiers": ["instant"], "time_column": "dteday"}),
    ("run_all", {"project_id": "PID"}),
    ("get_results", {"project_id": "PID", "stage": "final"}),
    ("finish", {"report": "Bike model built; see the project."}),
]


def make(home):
    projects = ProjectStore(Path(home) / "projects")
    return projects, SessionStore(Path(home) / "intern")


class ToolboxTests(unittest.TestCase):
    def test_schemas_and_guardrails(self):
        projects, _ = make(tempfile.mkdtemp())
        box = Toolbox(projects)
        names = set(box.names())
        self.assertTrue({"search_evidence", "create_project", "use_sample", "set_solution", "run_all", "export_notebook"} <= names)
        self.assertNotIn("audit_columns", names)  # no path-based tools for the model
        self.assertIn("error", box.call("nope", {}))
        self.assertIn("error", box.call("create_project", {"name": "x"}))  # goal required
        self.assertIn("error", box.call("describe_data", {"project_id": "000000000000"}))
        samples = box.call("list_samples", {})["samples"]
        self.assertTrue({"telco_churn", "bike_sharing_daily", "letter_recognition", "credit_card_fraud", "ecommerce_clothing_reviews"} <= {s["key"] for s in samples})


class StandardPlanTests(unittest.TestCase):
    def test_runs_the_fixed_order_and_reports(self):
        projects, sessions = make(tempfile.mkdtemp())
        intern = Intern(sessions, Toolbox(projects))
        with fast_profile():
            session = intern.start("Build a leakage-safe heart disease model and report the honest score.", {"max_steps": 12, "max_minutes": 5})
            session = intern.run(session["id"])
        self.assertEqual(session["status"], "completed")
        self.assertEqual([s["tool"] for s in session["steps"]], ["list_samples", "create_project", "use_sample", "describe_data", "propose_solution", "set_solution", "run_all", "export_notebook"])
        self.assertTrue(all(s["ok"] for s in session["steps"]))
        self.assertIn("Honest score", session["final"])
        self.assertIn("DCLAB-R22", session["final"])
        project = projects.get(session["project_id"])
        self.assertEqual(project["solution"]["target"], "target")
        self.assertEqual(project["data"]["filename"], "heart_disease.csv")
        self.assertTrue(all(v["status"] == "completed" for v in project["stages"].values()))
        self.assertTrue((projects.directory(project["id"]) / "exports" / "notebook.ipynb").exists())
        followup = intern.message(session["id"], "which model was chosen?")
        self.assertEqual(followup["status"], "completed")
        self.assertIn("ranking", followup["final"].lower())
        examples = sft.examples_from_project(project, projects.records(project["id"]))
        self.assertEqual(len(examples), 5)
        self.assertTrue(all(len(e["messages"]) == 3 and e["metadata"]["source"] == "notebook_project" for e in examples))

    def test_unknown_dataset_asks_for_one(self):
        projects, sessions = make(tempfile.mkdtemp())
        intern = Intern(sessions, Toolbox(projects))
        session = intern.run(intern.start("Model the quarterly widget pipeline of my company.", {"max_steps": 5, "max_minutes": 1})["id"])
        self.assertEqual(session["status"], "completed")
        self.assertIn("upload", session["final"])
        self.assertIsNone(session["project_id"])


class LlmLoopTests(unittest.TestCase):
    def test_scripted_model_builds_a_regression_project(self):
        projects, sessions = make(tempfile.mkdtemp())
        intern = Intern(sessions, Toolbox(projects), ScriptedModel(BIKE_SCRIPT))
        with fast_profile():
            session = intern.run(intern.start("Forecast bike rentals", {"max_steps": 20, "max_minutes": 5})["id"])
        self.assertEqual(session["status"], "completed")
        self.assertEqual(session["final"], "Bike model built; see the project.")
        self.assertEqual(session["plan"], "1 load the bike sample 2 solution 3 stages 4 report")
        self.assertEqual([s["tool"] for s in session["steps"]][:4], ["write_plan", "list_samples", "create_project", "use_sample"])
        final = projects.read_stage(session["project_id"], "final")["evidence"]
        self.assertIn("mae", final["holdout_metrics"])
        self.assertEqual(projects.read_stage(session["project_id"], "leakage")["evidence"]["declared_leakage_features"], ["casual", "registered"])
        self.assertGreater(session["used"]["input_tokens"], 0)

    def test_budget_is_enforced(self):
        projects, sessions = make(tempfile.mkdtemp())
        intern = Intern(sessions, Toolbox(projects), ScriptedModel(BIKE_SCRIPT))
        with fast_profile():
            session = intern.run(intern.start("Forecast bike rentals", {"max_steps": 3, "max_minutes": 5})["id"])
        self.assertEqual(session["status"], "budget_exhausted")
        self.assertEqual(session["used"]["steps"], 3)
        self.assertIn("budget", session["final"])


    def test_a_refused_plan_or_report_does_not_break_the_session(self):
        projects, sessions = make(tempfile.mkdtemp())
        script = [("write_plan", {}), ("finish", {}), ("finish", {"report": "Nothing to build yet."})]
        intern = Intern(sessions, Toolbox(projects), ScriptedModel(script))
        session = intern.run(intern.start("Look around first", {"max_steps": 10, "max_minutes": 5})["id"])
        self.assertEqual((session["status"], session["final"]), ("completed", "Nothing to build yet."))
        self.assertEqual([(s["tool"], s["ok"]) for s in session["steps"]], [("write_plan", False), ("finish", False), ("finish", True)])


class StreamingClientTests(unittest.TestCase):
    """ChatClient.stream joins a streamed reply into the shape complete() returns, without a network or a key."""

    @staticmethod
    def chunk(content=None, calls=None, finish=None, usage=None):
        from types import SimpleNamespace as NS
        delta = NS(content=content, tool_calls=calls)
        return NS(choices=[NS(delta=delta, finish_reason=finish)] if (content is not None or calls or finish) else [], usage=usage)

    def client(self, chunks, seen=None, fail=False):
        from types import SimpleNamespace as NS
        from dclab_rnd.intern.llm import ChatClient

        def create(**kwargs):
            if seen is not None:
                seen.update(kwargs)
            if fail:
                raise ValueError("secret-header-value")
            return iter(chunks)
        client = ChatClient(model="m", base_url="http://x", api_key="k")
        client._client = NS(chat=NS(completions=NS(create=create)))
        return client

    def test_text_and_split_tool_arguments_are_joined(self):
        from types import SimpleNamespace as NS
        fn = lambda name=None, args=None: NS(name=name, arguments=args)  # noqa: E731
        chunks = [self.chunk("Hel"), self.chunk("lo."),
                  self.chunk(calls=[NS(index=0, id="c1", function=fn("record", '{"field": "tar'))]),
                  self.chunk(calls=[NS(index=0, id=None, function=fn(None, 'get", "value": "churn"}'))]),
                  self.chunk(calls=[NS(index=1, id="c2", function=fn("ask_user", "{oops"))]),
                  self.chunk(finish="tool_calls"),
                  self.chunk(usage=NS(prompt_tokens=11, completion_tokens=7))]
        heard, seen = [], {}
        out = self.client(chunks, seen).stream([{"role": "user", "content": "hi"}], [{"type": "function"}], on_text=heard.append)
        self.assertEqual(heard, ["Hel", "lo."])
        self.assertEqual(out["content"], "Hello.")
        self.assertEqual([(c["id"], c["name"], c["arguments"]) for c in out["tool_calls"]],
                         [("c1", "record", {"field": "target", "value": "churn"}), ("c2", "ask_user", {"_raw": "{oops"})])
        self.assertEqual((out["finish_reason"], out["usage"]), ("tool_calls", {"input_tokens": 11, "output_tokens": 7}))
        self.assertEqual(out["assistant_message"]["tool_calls"][0]["function"]["name"], "record")
        self.assertTrue(seen["stream"] and seen["tool_choice"] == "auto")

    def test_a_refusal_gets_a_safe_reason_and_no_credit_pauses_further_requests(self):
        from types import SimpleNamespace as NS
        from dclab_rnd.intern import llm

        llm.resume()
        self.addCleanup(llm.resume)

        class Refused(Exception):
            def __init__(self, status, code):
                super().__init__("Authorization: Bearer sk-live-secret")  # provider text can carry a key: never shown
                self.status_code, self.body = status, {"error": {"code": code, "type": "x", "message": "sk-live-secret"}}
        cases = [(Refused(429, "insufficient_quota"), "the model account has no credit left", True), (Refused(401, "invalid_api_key"), "the provider refused the key", True),
                 (Refused(404, "model_not_found"), "the model name is not available to this key", True), (Refused(429, "rate_limit_exceeded"), "the provider is limiting requests for now", False),
                 (Refused(503, "server_error"), "the provider had an error", False), (ValueError("x"), "the model request failed", False)]
        self.assertEqual([llm.failure(e) for e, _, _ in cases], [(reason, pointless) for _, reason, pointless in cases])
        sent = []

        def create(**kwargs):
            sent.append(1)
            raise Refused(429, "insufficient_quota")
        client = llm.ChatClient(model="m", base_url="http://x", api_key="k")
        client._client = NS(chat=NS(completions=NS(create=create)))
        with self.assertRaises(RuntimeError) as first:
            client.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(str(first.exception), "Refused: the model account has no credit left")
        self.assertNotIn("sk-live-secret", str(first.exception))
        for call in (client.complete, client.stream):
            with self.assertRaises(RuntimeError) as again:
                call([{"role": "user", "content": "hi"}])
            self.assertEqual(str(again.exception), "ModelPaused: the model account has no credit left")
        self.assertEqual(len(sent), 1)  # the paused requests never reached the provider
        other = llm.ChatClient(model="other", base_url="http://x", api_key="k")  # another model is not paused
        other._client = NS(chat=NS(completions=NS(create=lambda **k: (_ for _ in ()).throw(Refused(429, "rate_limit_exceeded")))))
        with self.assertRaises(RuntimeError) as limited:
            other.complete([{"role": "user", "content": "hi"}])
        self.assertIn("limiting requests", str(limited.exception))
        with self.assertRaises(RuntimeError) as limited_again:
            other.complete([{"role": "user", "content": "hi"}])
        self.assertNotIn("ModelPaused", str(limited_again.exception))  # a rate limit passes by itself: keep trying
        llm.resume()
        with self.assertRaises(RuntimeError) as after:
            client.complete([{"role": "user", "content": "hi"}])
        self.assertNotIn("ModelPaused", str(after.exception))

    def test_errors_surface_by_type_only(self):
        with self.assertRaises(RuntimeError) as caught:
            self.client([], fail=True).stream([{"role": "user", "content": "hi"}])
        self.assertIn("ValueError", str(caught.exception))
        self.assertNotIn("secret-header-value", str(caught.exception))


class StandardPlanSolutionTests(unittest.TestCase):
    def test_time_column_is_not_also_forbidden(self):
        """The fraud sample blocks elapsed Time and also orders the split by it; the plan must not list it twice."""
        projects, sessions = make(tempfile.mkdtemp())
        intern = Intern(sessions, Toolbox(projects), None)
        with fast_profile():
            session = intern.run(intern.start("Audit the credit-card fraud data for leakage", {"max_steps": 7, "max_minutes": 5})["id"])
        saved = next((s for s in session["steps"] if s["tool"] == "set_solution"), None)
        # on failure, say how the session ended: CI once saw it stop after two steps, never reproduced on a laptop
        why = {"status": session.get("status"), "error": session.get("error"), "steps": [(s["tool"], str(s.get("result", ""))[:160]) for s in session["steps"]]}
        self.assertIsNotNone(saved, why)
        self.assertNotIn("error", json.dumps(saved.get("result", {}))[:200].lower(), saved)
        solution = projects.get(session["project_id"])["solution"]
        self.assertNotIn(solution["time_column"], [f["column"] for f in solution["forbidden"]])


class ApiTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        # The server loads .env, so a developer's real key would send these tests to a live model.
        # Without a key the intern follows the deterministic standard plan, the same as in CI.
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        self.headers = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def test_sessions_over_http(self):
        c = self.client
        info = c.get("/api/intern").json()
        self.assertEqual(info["mode"], "standard")
        self.assertTrue(info["examples"])
        self.assertEqual(c.post("/api/intern/sessions", json={"task": "x"}, headers=self.headers).status_code, 422)
        with fast_profile():
            s = c.post("/api/intern/sessions?wait=true", json={"task": "Forecast daily bike rentals and say whether tuning helped.", "budget": {"max_steps": 12, "max_minutes": 5}}, headers=self.headers).json()
        self.assertEqual(s["status"], "completed")
        self.assertNotIn("messages", s)
        self.assertTrue(s["project_id"])
        self.assertEqual(c.get(f"/api/projects/{s['project_id']}").json()["stages"]["final"]["status"], "completed")
        rows = [json.loads(line) for line in c.get(f"/api/projects/{s['project_id']}/export/sft").text.splitlines() if line]
        self.assertEqual(len(rows), 6)  # five stages plus the solution example (two forbidden columns)
        self.assertEqual(c.get(f"/api/projects/{s['project_id']}").json()["solution"]["forbidden"][0]["column"], "casual")
        self.assertEqual(c.get("/api/intern/sessions").json()[0]["id"], s["id"])
        self.assertEqual(c.delete(f"/api/intern/sessions/{s['id']}", headers=self.headers).status_code, 204)


if __name__ == "__main__":
    unittest.main()
