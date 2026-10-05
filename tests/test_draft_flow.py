"""Home's draft flow: store and events, pack detection, the solution workflow, the agent, and the HTTP routes."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.draft import pack, workflow  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent, MAX_QUESTIONS  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402


def wait(fn, timeout=60.0, every=0.1):
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        time.sleep(every)
    raise AssertionError("timed out waiting")


class StoreTests(unittest.TestCase):
    def test_create_update_events(self):
        store = DraftStore(Path(tempfile.mkdtemp()))
        d = store.create("Predict which customers leave next month", "tabular")
        self.assertEqual(d["pack"]["source"], "user")
        store.update(d["id"], lambda x: x["messages"].append({"text": "hi"}))
        self.assertEqual(store.get(d["id"])["messages"][0]["text"], "hi")
        a, b = store.emit(d["id"], "chat", {"x": 1}), store.emit(d["id"], "status", {})
        self.assertEqual((a["seq"], b["seq"]), (1, 2))
        self.assertEqual([e["seq"] for e in store.events(d["id"], after=1)], [2])
        with self.assertRaises(KeyError):
            store.get("../etc")


class PackTests(unittest.TestCase):
    def test_user_choice_wins_then_text_then_data(self):
        self.assertEqual(pack.detect("forecast daily demand", chosen="vision")["key"], "vision")
        self.assertEqual(pack.detect("Forecast tomorrow's bike rentals")["key"], "timeseries")
        self.assertEqual(pack.detect("Find card fraud at authorization time")["key"], "imbalanced")
        self.assertEqual(pack.detect("Predict churn", {"profile": {"text_candidates": ["review"]}, "summary": {"kinds": {}}})["key"], "text")
        d = pack.detect("Predict churn")
        self.assertEqual((d["key"], d["source"]), ("tabular", "default"))


class WorkflowTests(unittest.TestCase):
    def test_template_has_gates_and_revisits_from_code(self):
        wf = workflow.template("timeseries", {"target": "rentals tomorrow"})
        self.assertEqual([n["wf"] for n in wf["nodes"]], [f"WF-{i:02d}" for i in range(1, 11)])
        self.assertEqual({g["gate"] for g in wf["gates"]}, {"solution", "holdout"})
        self.assertTrue(any(n["label"] == "Time split + backtests" for n in wf["nodes"]))
        self.assertIn("rentals tomorrow", wf["nodes"][0]["detail"])

    def test_model_workflow_is_validated(self):
        good = {"title": "x", "nodes": [{"wf": w, "label": l} for w, l in [("WF-01", "Solution"), ("WF-02", "Label delay"), ("WF-03", "Split by store"),
                                                                           ("WF-05", "Leakage"), ("WF-07", "Screen"), ("WF-09", "Holdout")]]}
        wf, problems = workflow.validate(good, "tabular")
        self.assertEqual(problems, [])
        self.assertEqual(wf["source"], "model")
        self.assertEqual(len(wf["gates"]), 1)  # no WF-04 step, so only the holdout gate
        bad_order = {"nodes": [{"wf": "WF-01", "label": "a"}, {"wf": "WF-05", "label": "b"}, {"wf": "WF-03", "label": "c"}, {"wf": "WF-09", "label": "d"}]}
        self.assertIsNone(workflow.validate(bad_order, None)[0])
        missing = {"nodes": [{"wf": "WF-01", "label": "a"}, {"wf": "WF-07", "label": "b"}]}
        self.assertIn("missing required steps", " ".join(workflow.validate(missing, None)[1]))
        self.assertIsNone(workflow.validate({"nodes": [{"wf": "WF-99", "label": "x"}]}, None)[0])


class ScriptedClient:
    """Plays tool calls like a chat model, then stops."""

    def __init__(self, turns):
        self.turns, self.i = turns, 0

    def complete(self, messages, tools=None, max_tokens=1800):
        self.i += 1
        if self.i > len(self.turns):
            return {"content": "Thanks, that is clear.", "tool_calls": [], "assistant_message": {"role": "assistant", "content": "Thanks, that is clear."}}
        name, args = self.turns[self.i - 1]
        call = {"id": f"c{self.i}", "name": name, "arguments": args}
        return {"content": "", "tool_calls": [call], "assistant_message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": call["id"], "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.store = DraftStore(Path(tempfile.mkdtemp()))

    def test_script_asks_few_questions_then_summarises(self):
        requests = []
        agent = HomeAgent(self.store, None, on_request=lambda d, what, args: requests.append(what))
        d = self.store.create("Predict which customers cancel their subscription")
        agent.start(d["id"])
        d = self.store.get(d["id"])
        self.assertEqual(d["pack"]["key"], "tabular")
        self.assertEqual(d["workflow"]["nodes"][0]["state"], "current")
        self.assertEqual(d["questions"][0]["field"], "target")
        for answer in ["Customers who cancel within 30 days", "At a fixed snapshot each month", "We contact the top cases", "Simulate data"]:
            agent.reply(d["id"], answer)
        d = self.store.get(d["id"])
        self.assertLessEqual(len(d["questions"]), MAX_QUESTIONS)
        self.assertEqual(d["understanding"]["prediction_moment"], "At a fixed snapshot each month")
        self.assertEqual(requests, ["simulate"])
        self.assertEqual(d["workflow"]["nodes"][0]["state"], "done")
        self.assertEqual(d["messages"][-1]["kind"], "summary")
        kinds = [e["kind"] for e in self.store.events(d["id"])]
        self.assertIn("workflow", kinds)
        self.assertIn("chat", kinds)

    def test_model_turn_uses_tools_and_code_validates_the_workflow(self):
        d = self.store.create("Forecast tomorrow's demand per store")
        HomeAgent(self.store, None).start(d["id"])
        client = ScriptedClient([
            ("record", {"field": "target", "value": "units sold per store tomorrow"}),
            ("propose_workflow", {"title": "Store demand", "nodes": [{"wf": "WF-01", "label": "x"}, {"wf": "WF-07", "label": "y"}]}),  # rejected
            ("set_pack", {"key": "timeseries", "why": "daily forecast"}),
            ("ask_user", {"field": "prediction_moment", "question": "When is the forecast made?", "options": ["The day before"]}),
        ])
        HomeAgent(self.store, client).reply(d["id"], "Units per store")
        d = self.store.get(d["id"])
        self.assertEqual(d["understanding"]["target"], "units sold per store tomorrow")
        self.assertEqual(d["workflow"]["source"], "template")  # the invalid proposal was refused
        self.assertEqual(d["pack"]["key"], "timeseries")
        self.assertEqual(d["messages"][-1]["kind"], "question")
        rejected = [e for e in self.store.events(d["id"]) if e["kind"] == "workflow" and e["data"].get("rejected")]
        self.assertTrue(rejected)

    def test_model_failure_falls_back_to_the_script(self):
        class Broken:
            def complete(self, *a, **k):
                raise RuntimeError("RateLimitError: the model request failed")
        d = self.store.create("Predict which customers cancel")
        HomeAgent(self.store, None).start(d["id"])
        HomeAgent(self.store, Broken()).reply(d["id"], "Customers who cancel within 30 days")
        d = self.store.get(d["id"])
        self.assertEqual(d["understanding"]["target"], "Customers who cancel within 30 days")
        notes = [e["data"].get("note") for e in self.store.events(d["id"]) if e["kind"] == "status" and e["data"].get("note")]
        self.assertTrue(any("RateLimitError" in n for n in notes))


class BooleanFeatureTests(unittest.TestCase):
    """Cleaning turns yes/no columns into booleans; the engine must accept them as features (and as the target)."""

    def test_data_stage_runs_with_boolean_columns(self):
        import numpy as np
        import pandas as pd
        from dclab_rnd.studio import ProjectStore, data, engine

        rng = np.random.default_rng(3)
        n = 400
        frame = pd.DataFrame({"tenure": rng.integers(1, 72, n), "paperless": pd.array(rng.random(n) > 0.5, dtype="boolean"),
                              "partner": rng.random(n) > 0.4, "charges": rng.normal(60, 20, n).round(2)})
        frame.loc[3, "paperless"] = pd.NA
        frame["churned"] = (rng.random(n) < 0.2 + 0.2 * frame["partner"]).astype(bool)
        store = ProjectStore(Path(tempfile.mkdtemp()))
        pid = store.create("bools", "general", "who leaves")["id"]
        frame.to_parquet(store.data_dir(pid) / "t.parquet", index=False)
        data.attach_data(store, pid, "t.parquet")
        project = store.get(pid)
        project["solution"] = {"target": "churned", "task": "binary", "positive_label": "True", "prediction_moment": "At the monthly snapshot of each customer.",
                               "forbidden": [], "identifiers": [], "time_column": None, "group_column": None, "text_columns": [], "metric": "roc_auc", "notes": ""}
        project["settings"]["quick"] = True
        store.save(project)
        record = engine.execute(store, pid, "data", "human")
        rate = record["evidence"]["target_summary"]["positive_rate_train"]
        self.assertTrue(0.15 < rate < 0.6, rate)  # True was read as the positive class, not "nothing is positive"


class ApiTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def draft(self, did):
        return self.client.get(f"/api/drafts/{did}").json()

    def test_full_flow_sample_then_build(self):
        c = self.client
        self.assertEqual(len(c.get("/api/packs").json()), len(pack.PACKS))
        self.assertEqual(c.post("/api/drafts", json={"problem": "x"}, headers=self.h).status_code, 422)
        d = c.post("/api/drafts", json={"problem": "Rank clients for the term-deposit call campaign"}, headers=self.h).json()
        wait(lambda: self.draft(d["id"])["questions"])
        r = c.post(f"/api/drafts/{d['id']}/data/sample", json={"key": "bank_marketing"}, headers=self.h)
        self.assertEqual(r.status_code, 200, r.text)
        ready = wait(lambda: next((a for a in self.draft(d["id"])["assets"] if a["status"] in ("ready", "failed")), None))
        self.assertEqual(ready["status"], "ready", ready.get("error"))
        draft = self.draft(d["id"])
        self.assertGreater(draft["analysis"]["summary"]["rows"], 1000)
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/messages", json={"text": ""}, headers=self.h).status_code, 422)
        target = draft["analysis"]["profile"]["target_candidates"][0] if draft["analysis"]["profile"]["target_candidates"] else "y"
        wait(lambda: c.post(f"/api/drafts/{d['id']}/messages", json={"text": target}, headers=self.h).status_code == 202)
        wait(lambda: self.draft(d["id"])["understanding"].get("target") == target)
        # wizard: the solution draft from the column audit, then split and budget
        y = ready["suggestion"]["target"]
        proposal = c.post(f"/api/drafts/{d['id']}/solution/proposal", json={"target": y}, headers=self.h)
        self.assertEqual(proposal.status_code, 200, proposal.text)
        prop = proposal.json()
        self.assertIn("duration", {f["column"] for f in prop["forbidden"]})  # the R&D's own solution for bank_marketing
        solution = {"target": y, "task": prop["task"], "positive_label": str(prop["positive_label"]) if prop["task"] == "binary" else None,
                    "prediction_moment": "Immediately before the marketing call is placed.",
                    "forbidden": [{"column": "duration", "reason": "known only after the call"}], "identifiers": [], "time_column": None,
                    "group_column": None, "text_columns": [], "metric": prop["metric"], "notes": ""}
        self.assertEqual(c.put(f"/api/drafts/{d['id']}/solution", json={**solution, "target": "nope"}, headers=self.h).status_code, 422)
        self.assertEqual(c.put(f"/api/drafts/{d['id']}/solution", json=solution, headers=self.h).status_code, 200)
        settings = c.put(f"/api/drafts/{d['id']}/settings", json={"split": "stratified", "quick": True, "max_rows": 999999, "budget": {"calls": 500}}, headers=self.h).json()["settings"]
        self.assertEqual((settings["max_rows"], settings["budget"]["max_steps"]), (200000, 80))  # clamped
        project = c.post(f"/api/drafts/{d['id']}/build", json={}, headers=self.h)
        self.assertEqual(project.status_code, 201, project.text)
        p = project.json()
        self.assertEqual(p["data"]["rows"], ready["rows"])
        self.assertFalse(p["data"]["synthetic"])
        self.assertEqual(p["draft"]["id"], d["id"])
        self.assertEqual(p["solution"]["forbidden"][0]["column"], "duration")
        self.assertEqual(p["budget"]["max_steps"], 80)
        moves = c.get(f"/api/projects/{p['id']}").json()["transitions"]
        self.assertEqual([(m["move"], m["status"]) for m in moves], [("set_solution", "allowed")])
        self.assertEqual(self.draft(d["id"])["status"], "built")
        ws = c.get("/api/workspace").json()
        self.assertEqual(ws["stats"]["projects"], 1)

    def test_upload_and_events_stream(self):
        c = self.client
        d = c.post("/api/drafts", json={"problem": "Which support tickets get escalated?"}, headers=self.h).json()
        rows = "\n".join(f"{i},{'yes' if i % 3 == 0 else 'no'},{i * 1.5},  team {i % 4} " for i in range(80))
        r = c.put(f"/api/drafts/{d['id']}/data?filename=../../tickets.csv", content=("id,escalated,hours,team\n" + rows).encode(), headers=self.h)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["filename"], "tickets.csv")  # the path part is dropped
        wait(lambda: next((a for a in self.draft(d["id"])["assets"] if a["status"] == "ready"), None))
        with c.stream("GET", f"/api/drafts/{d['id']}/events?wait=0.5") as s:
            lines = []
            for line in s.iter_lines():
                lines.append(line)
                if line.startswith("event: analysis"):
                    break
        self.assertTrue(any(line.startswith("event: chat") for line in lines))
        self.assertTrue(any(line.startswith("id: ") for line in lines))

    def test_synthetic_data_is_labelled(self):
        c = self.client
        d = c.post("/api/drafts", json={"problem": "Detect card fraud at authorization time"}, headers=self.h).json()
        r = c.post(f"/api/drafts/{d['id']}/data/synthetic", json={"prompt": "card fraud", "rows": 1000}, headers=self.h)
        self.assertEqual(r.status_code, 202, r.text)
        asset = wait(lambda: next((a for a in self.draft(d["id"])["assets"] if a["status"] in ("ready", "failed")), None))
        self.assertEqual(asset["status"], "ready", asset.get("error"))
        self.assertTrue(asset["synthetic"])
        p = c.post(f"/api/drafts/{d['id']}/build", json={}, headers=self.h).json()
        self.assertTrue(p["data"]["synthetic"])
        self.assertTrue(p["draft"]["synthetic"])


if __name__ == "__main__":
    unittest.main()
