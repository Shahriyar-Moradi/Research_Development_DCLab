"""The DCLab notebook: projects, solutions, the five-stage engine on user data, exports and the API."""

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.expansion.runner import fast_profile  # noqa: E402
from dclab_rnd.studio import ProjectStore, agent, solution as sc, data as sd, engine, export  # noqa: E402


def churn_frame(n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "signup_date": pd.date_range("2022-01-01", periods=n, freq="6h").strftime("%Y-%m-%d %H:%M:%S"),
        "age": rng.integers(18, 80, n), "plan": rng.choice(["basic", "plus", "pro"], n),
        "monthly_fee": rng.uniform(10, 120, n).round(2), "support_calls": rng.poisson(1.2, n),
    })
    logit = -2 + 0.03 * (frame.age - 45) + 0.5 * frame.support_calls - 0.01 * frame.monthly_fee + (frame.plan == "basic") * 0.8
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    frame["churned"] = np.where(y == 1, "yes", "no")
    frame["final_refund_amount"] = np.where(y == 1, rng.uniform(5, 50, n), 0.0)  # known only after churn
    return frame


def new_project(store: ProjectStore, frame: pd.DataFrame, solution: dict) -> dict:
    project = store.create("Test project", "telecom", "Predict churn before renewal")
    path = store.data_dir(project["id"]) / "data.csv"
    frame.to_csv(path, index=False)
    loaded = sd.load_table(path)
    project["data"] = {"filename": "data.csv", "rows": len(loaded), "columns": list(loaded.columns), "sha256": sd.sha256(path), "profile": sd.profile_table(loaded)}
    project["solution"] = sc.Solution(**solution).model_dump()
    project["settings"]["quick"] = True
    return store.save(project)


CHURN_CONTRACT = {"target": "churned", "task": "binary", "positive_label": "yes", "identifiers": ["customer_id"], "time_column": "signup_date",
                  "prediction_moment": "At renewal time, before the customer decides; refunds are only known afterwards.",
                  "forbidden": [{"column": "final_refund_amount", "reason": "paid after churn"}]}


class DataAndContractTests(unittest.TestCase):
    def test_profile_and_proposal_find_the_planted_leak(self):
        frame = churn_frame()
        profile = sd.profile_table(frame)
        self.assertEqual(profile["target_candidates"][0], "churned")
        self.assertIn("signup_date", profile["time_candidates"])
        self.assertIn("customer_id", profile["id_candidates"])
        proposal = sc.propose(frame, profile, "churned")
        self.assertEqual(proposal["task"], "binary")
        self.assertEqual(proposal["positive_label"], "yes")
        self.assertIn("final_refund_amount", [f["column"] for f in proposal["forbidden"]])
        self.assertIn("customer_id", proposal["identifiers"])

    def test_solution_rejects_incoherent_forms(self):
        with self.assertRaises(ValueError):
            sc.Solution(target="y", task="binary", prediction_moment="Before the outcome is known", forbidden=[{"column": "y"}])
        with self.assertRaises(ValueError):
            sc.Solution(target="y", task="regression", prediction_moment="Before the outcome is known", metric="roc_auc")
        c = sc.Solution(target="y", task="binary", prediction_moment="Before the outcome is known")
        with self.assertRaises(ValueError):
            c.check_columns(["x"])

    def test_sample_catalog_carries_known_solutions(self):
        catalog = {s["key"]: s for s in sd.sample_catalog()}
        self.assertIn("final_customer_fare", catalog["hyperack"]["blocked"])
        self.assertIn("duration", catalog["bank_marketing"]["blocked"])


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = tempfile.mkdtemp()
        cls.store = ProjectStore(Path(cls.home))
        cls.project = new_project(cls.store, churn_frame(), CHURN_CONTRACT)
        with fast_profile():
            cls.records = {stage: engine.execute(cls.store, cls.project["id"], stage) for stage in ("data", "leakage", "features", "models", "final")}

    def test_every_stage_has_the_campaign_record_shape(self):
        for stage, record in self.records.items():
            for key in ("experiment_id", "setup_summary", "evidence", "claims", "notes", "provenance", "primary_metric", "decision_time_rule"):
                self.assertIn(key, record, stage)
            for claim in record["claims"]:
                self.assertEqual(set(claim), {"claim_id", "kind", "statement", "evidence", "limitations"})
            for note in record["notes"]:
                self.assertTrue(note["proof"], f"{stage}: {note['title']} has no proof")

    def test_leak_is_measured_and_time_order_is_kept(self):
        ev = self.records["leakage"]["evidence"]
        self.assertEqual(ev["declared_leakage_features"], ["final_refund_amount"])
        self.assertGreater(ev["apparent_lift"], 0.05)
        self.assertIn("random_vs_time_cv", ev)
        self.assertTrue(any(n["severity"] == "high" for n in self.records["leakage"]["notes"]))
        self.assertIn("TimeSeriesSplit", self.records["data"]["evidence"]["cv_protocol"])
        self.assertEqual(self.records["data"]["evidence"]["column_roles"]["customer_id"], "identifier")

    def test_holdout_is_consumed_once_and_choices_flow_downstream(self):
        final = self.records["final"]["evidence"]
        self.assertTrue(final["holdout_consumed"])
        self.assertEqual(final["holdout_uses_in_this_project"], 1)
        self.assertEqual(final["feature_recipe"], self.records["features"]["decision"]["chosen"])
        self.assertEqual(final["model"], self.records["models"]["decision"]["chosen"])
        self.assertFalse(final["production_approved"])
        project = self.store.get(self.project["id"])
        self.assertTrue(all(v["status"] in ("completed", "approved") for v in project["stages"].values()))

    def test_approve_with_override_clears_downstream(self):
        options = [o["id"] for o in self.records["models"]["decision"]["options"]]
        other = next(o for o in options if o != self.records["models"]["decision"]["selected"])
        project = engine.approve(self.store, self.project["id"], "models", other)
        self.assertEqual(project["stages"]["models"]["status"], "approved")
        self.assertEqual(project["stages"]["final"]["status"], "pending")
        self.assertTrue(self.store.read_stage(self.project["id"], "models")["decision"]["overridden"])
        from dclab_rnd.studio import graph
        with self.assertRaises(graph.GraphBlocked) as blocked:  # the holdout was used once already
            engine.execute(self.store, self.project["id"], "final")
        self.assertEqual(blocked.exception.verdict.status, "needs_approval")
        with self.assertRaises(graph.GraphBlocked) as agent:  # the intern may never reuse it
            engine.execute(self.store, self.project["id"], "final", actor="agent", reuse_reason="want a better score")
        self.assertEqual(agent.exception.verdict.status, "blocked")
        with fast_profile():
            record = engine.execute(self.store, self.project["id"], "final", reuse_reason="the owner chose a more explainable model")
        self.assertEqual(record["evidence"]["holdout_reuse_reason"], "the owner chose a more explainable model")
        statuses = [t["status"] for t in self.store.transitions(self.project["id"])]
        self.assertIn("needs_approval", statuses)
        self.assertIn("blocked", statuses)
        self.assertEqual(record["evidence"]["model"], other)
        self.assertEqual(record["evidence"]["holdout_uses_in_this_project"], 2)
        self.assertTrue(any(n["severity"] == "high" and "used 2 times" in n["title"] for n in record["notes"]))

    def test_exports_are_valid(self):
        project = self.store.get(self.project["id"])
        records = self.store.records(self.project["id"])
        nb = export.notebook(project, records)
        self.assertEqual(nb["nbformat"], 4)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
        json.loads(export.dumps_notebook(nb))
        report = export.report(project, records)
        self.assertIn("final_refund_amount", report)
        self.assertIn("DCLAB-R22", report)
        self.assertNotIn("Synthetic data", report + json.dumps(nb))
        # the data file is read by its type, and synthetic data says so in both exports
        synthetic = {**project, "data": {**project["data"], "filename": "sim.parquet", "synthetic": True}}
        nb = export.notebook(synthetic, records)
        self.assertIn("frame = pd.read_parquet('sim.parquet')", json.dumps(nb))
        self.assertIn(export.SYNTHETIC_NOTE, nb["cells"][0]["source"])
        self.assertIn(export.SYNTHETIC_NOTE, export.report(synthetic, records))
        self.assertEqual(export._read_code("t.tsv"), "pd.read_csv('t.tsv', sep='\\t', low_memory=False)")

    def test_answers_cite_project_facts_and_evidence(self):
        answer = agent.answer("is final_refund_amount a leak?", self.store.get(self.project["id"]), self.store.records(self.project["id"]), "binary_imbalanced")
        self.assertIn("final_refund_amount", answer["answer"])
        self.assertTrue(answer["proof"])


class OtherTaskTests(unittest.TestCase):
    def test_multiclass_and_grouped_regression(self):
        rng = np.random.default_rng(1)
        n = 600
        mc = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "cat": rng.choice(list("xyz"), n)})
        mc["kind"] = np.where(mc.a + mc.b > 0.7, "high", np.where(mc.a + mc.b < -0.7, "low", "mid"))
        rg = pd.DataFrame({"store_id": rng.choice([f"S{i}" for i in range(25)], n), "x1": rng.normal(size=n), "promo": rng.choice([0, 1], n)})
        rg["sales"] = 50 + 10 * rg.x1 + 20 * rg.promo + rng.normal(0, 5, n)
        rg["units"] = rg.sales / 5
        store = ProjectStore(Path(tempfile.mkdtemp()))
        with fast_profile():
            p1 = new_project(store, mc, {"target": "kind", "task": "multiclass", "prediction_moment": "At observation time, all inputs known."})
            p2 = new_project(store, rg, {"target": "sales", "task": "regression", "group_column": "store_id", "forbidden": [{"column": "units", "reason": "post-outcome"}],
                                         "prediction_moment": "Before the week starts; units sold are known after."})
            for project in (p1, p2):
                for stage in ("data", "leakage", "features", "models", "final"):
                    engine.execute(store, project["id"], stage)
        self.assertIn("macro_f1", store.read_stage(p1["id"], "final")["evidence"]["holdout_metrics"])
        self.assertEqual(store.read_stage(p1["id"], "data")["evidence"]["target_summary"]["classes"], 3)
        final = store.read_stage(p2["id"], "final")["evidence"]
        self.assertIn("mae", final["holdout_metrics"])
        self.assertIn("GroupKFold", store.read_stage(p2["id"], "data")["evidence"]["cv_protocol"])
        self.assertGreater(store.read_stage(p2["id"], "leakage")["evidence"]["apparent_lift"], 0)


class ApiTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        self.home = tempfile.mkdtemp()
        self.client = TestClient(create_app(Path(self.home)))
        self.token = self.client.get("/api/config").json()["csrf"]
        self.headers = {"X-DCLab-Token": self.token}

    def test_project_flow_over_http(self):
        c = self.client
        self.assertEqual(c.post("/api/projects", json={"name": "x"}).status_code, 403)
        created = c.post("/api/projects", json={"name": "HTTP churn", "industry": "telecom", "goal": "Predict churn"}, headers=self.headers)
        self.assertEqual(created.status_code, 201)
        pid = created.json()["id"]
        csv = churn_frame(400).to_csv(index=False).encode()
        self.assertEqual(c.put(f"/api/projects/{pid}/data?filename=../evil.csv", content=csv, headers=self.headers).status_code, 200)
        project = c.get(f"/api/projects/{pid}").json()
        self.assertEqual(project["data"]["filename"], "evil.csv")
        self.assertIn("churned", project["data"]["profile"]["target_candidates"])
        proposal = c.post(f"/api/projects/{pid}/solution/proposal", json={"target": "churned"}, headers=self.headers).json()
        self.assertIn("final_refund_amount", [f["column"] for f in proposal["forbidden"]])
        bad = c.put(f"/api/projects/{pid}/solution", json={**CHURN_CONTRACT, "target": "nope"}, headers=self.headers)
        self.assertEqual(bad.status_code, 422)
        self.assertEqual(c.put(f"/api/projects/{pid}/solution", json=CHURN_CONTRACT, headers=self.headers).status_code, 200)
        c.patch(f"/api/projects/{pid}", json={"settings": {"quick": True}}, headers=self.headers)
        with fast_profile():
            project = c.post(f"/api/projects/{pid}/run?wait=true", headers=self.headers).json()
        self.assertEqual({k: v["status"] for k, v in project["stages"].items()}, dict.fromkeys(project["stages"], "completed"))
        self.assertIn("final", project["records"])
        answer = c.post(f"/api/projects/{pid}/ask", json={"question": "which model is best?"}, headers=self.headers).json()
        self.assertIn("ranking", answer["answer"].lower())
        nb = c.get(f"/api/projects/{pid}/export/notebook")
        self.assertEqual(nb.status_code, 200)
        self.assertEqual(json.loads(nb.text)["nbformat"], 4)
        self.assertIn("Solution draft", c.get(f"/api/projects/{pid}/export/report").text)
        self.assertEqual(c.get("/api/evidence/DCLAB-R04").json()["record_id"], "DCLAB-R04")
        self.assertEqual(c.get("/api/evidence/NOPE").status_code, 404)
        self.assertEqual(c.post(f"/api/projects/{pid}/stages/models/approve", json={"choice": "logistic_regression"}, headers=self.headers).status_code, 200)
        self.assertEqual(c.get(f"/api/projects/{pid}").json()["stages"]["final"]["status"], "pending")
        self.assertEqual(c.delete(f"/api/projects/{pid}", headers=self.headers).status_code, 204)
        self.assertEqual(c.get(f"/api/projects/{pid}").status_code, 404)

    def test_samples_and_capabilities(self):
        self.assertTrue(any(s["key"] == "telco_churn" for s in self.client.get("/api/samples").json()))
        status = self.client.get("/api/studio").json()
        self.assertEqual(status["tasks"], ["binary", "multiclass", "regression"])


if __name__ == "__main__":
    unittest.main()
