"""The DCLab intern: the toolbox, the standard plan, the LLM loop with a scripted model, budgets and the API."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

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
    ("write_plan", {"plan": "1 load the bike sample 2 contract 3 stages 4 report"}),
    ("list_samples", {}),
    ("create_project", {"name": "Bike demand", "goal": "forecast daily rentals", "industry": "logistics and delivery"}),
    ("use_sample", {"project_id": "PID", "key": "bike_sharing_daily"}),
    ("set_contract", {"project_id": "PID", "target": "cnt", "task": "regression", "prediction_moment": "Forecast two days ahead; same-day rider counts are unknown.",
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
        self.assertTrue({"search_evidence", "create_project", "use_sample", "set_contract", "run_all", "export_notebook"} <= names)
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
        self.assertEqual([s["tool"] for s in session["steps"]], ["list_samples", "create_project", "use_sample", "describe_data", "propose_contract", "set_contract", "run_all", "export_notebook"])
        self.assertTrue(all(s["ok"] for s in session["steps"]))
        self.assertIn("Honest score", session["final"])
        self.assertIn("DCLAB-R22", session["final"])
        project = projects.get(session["project_id"])
        self.assertEqual(project["contract"]["target"], "target")
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
        self.assertEqual(session["plan"], "1 load the bike sample 2 contract 3 stages 4 report")
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


class ApiTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        self.headers = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def test_sessions_over_http(self):
        c = self.client
        info = c.get("/api/intern").json()
        self.assertIn(info["mode"], ("llm", "standard"))
        self.assertTrue(info["examples"])
        self.assertEqual(c.post("/api/intern/sessions", json={"task": "x"}, headers=self.headers).status_code, 422)
        with fast_profile():
            s = c.post("/api/intern/sessions?wait=true", json={"task": "Forecast daily bike rentals and say whether tuning helped.", "budget": {"max_steps": 12, "max_minutes": 5}}, headers=self.headers).json()
        self.assertEqual(s["status"], "completed")
        self.assertNotIn("messages", s)
        self.assertTrue(s["project_id"])
        self.assertEqual(c.get(f"/api/projects/{s['project_id']}").json()["stages"]["final"]["status"], "completed")
        rows = [json.loads(line) for line in c.get(f"/api/projects/{s['project_id']}/export/sft").text.splitlines() if line]
        self.assertEqual(len(rows), 6)  # five stages plus the contract example (two forbidden columns)
        self.assertEqual(c.get(f"/api/projects/{s['project_id']}").json()["contract"]["forbidden"][0]["column"], "casual")
        self.assertEqual(c.get("/api/intern/sessions").json()[0]["id"], s["id"])
        self.assertEqual(c.delete(f"/api/intern/sessions/{s['id']}", headers=self.headers).status_code, 204)


if __name__ == "__main__":
    unittest.main()
