"""Tests for solving business, data science, and ML problems without uploaded data."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from dclab_core import case as C
from dclab_rnd.agents import case_agent as A
from dclab_rnd.draft import case_home, chat as draft_chat, no_data, store as draft_store

try:
    from fastapi.testclient import TestClient

    from dclab_rnd.agentic.server import create_app
except ImportError as error:
    MISSING = str(error)
else:
    MISSING = ""


class NoDataUnitTests(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "OPENAI_API_KEY": ""})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_natural_stance_detection_english_and_persian(self):
        en = no_data.detect_no_data_stance("We want to predict customer churn, but we don't have any data yet")
        self.assertTrue(en["matched"])
        self.assertIn("no_data_yet", en["reasons"])

        priv = no_data.detect_no_data_stance("Please design a fraud solution without uploading our confidential data")
        self.assertTrue(priv["matched"])
        self.assertTrue(priv["withholding_privacy"])

        fa = no_data.detect_no_data_stance("هنوز دیتا ندارم، یک راه حل بدون آپلود فایل طراحی کن")
        self.assertTrue(fa["matched"])

        normal = no_data.detect_no_data_stance("I uploaded churn.csv with 50,000 rows", {"assets": [{"status": "ready"}]})
        self.assertFalse(normal["matched"])

    def test_deterministic_economics_and_virtual_catalog_blueprint(self):
        sol = no_data.solve_without_data(
            "Predict customer churn next month without data yet. 20000 customers, 10% churn rate, intervention cost $20, LTV $400.",
        )
        econ = sol["economics"]
        self.assertEqual(econ["inputs"]["monthly_volume"], 20000)

        # Regression guard: "50,000 monthly transactions" must never multiply by "m" from "monthly"
        sol_m = no_data.solve_without_data("Predict credit card fraud across 50,000 monthly transactions without data yet")
        self.assertEqual(sol_m["economics"]["inputs"]["monthly_volume"], 50000)

        # Persian digits & percent normalization
        sol_fa = no_data.solve_without_data("دیتا ندارم، ۲۰٬۰۰۰ مشتری داریم و نرخ ریزش ۱۰٪ است")
        self.assertEqual(sol_fa["economics"]["inputs"]["monthly_volume"], 20000)
        self.assertAlmostEqual(sol_fa["economics"]["inputs"]["base_rate"], 0.10, places=3)
        self.assertAlmostEqual(econ["inputs"]["base_rate"], 0.10, places=3)
        self.assertAlmostEqual(econ["inputs"]["action_cost"], 20.0, places=2)
        self.assertAlmostEqual(econ["inputs"]["value_per_positive"], 400.0, places=2)
        self.assertGreater(econ["break_even_precision"], 0.0)
        self.assertTrue(sol["as_said"][0].startswith('{"no_data_economics":'))
        ds = sol["data_science"]
        self.assertTrue(ds["virtual_tables"])
        self.assertTrue(ds["forbidden_leakage_columns"])
        ml = sol["ml"]
        self.assertEqual(ml["catalog_plan"]["schema_source"], "virtual_blueprint")
        self.assertTrue(sol["simulation"]["requires_user_approval"])
        self.assertGreaterEqual(len(sol["simulation"]["questions"]), 3)

    def test_case_agent_tool_and_case_home_card_persistence(self):
        home_dir = Path(tempfile.mkdtemp())
        store = draft_store.DraftStore(home_dir)
        d = store.create("Reduce subscription churn without data yet")
        state = A.CaseState(case=C.new(f"draft-{d['id']}"))
        state.case = C.hear(state.case, "We want to reduce churn but do not have a dataset yet and don't want synthetic data unless asked.")
        out = A._design_solution_without_data(state, monthly_volume=5000, base_rate=0.12, action_cost=10, value_saved=250)
        self.assertTrue(out["blueprint_ready"])
        self.assertTrue(any(str(c).startswith('{"no_data_economics":') for c in state.computed))
        self.assertEqual(state.cards[-1]["kind"], "no_data_solution")

        # When no_data stance exists and table is None, _start guides agent to design_solution_without_data instead of upload refusal
        err = A._start(state, "prediction", "ml_project")
        self.assertIn("design_solution_without_data", err["error"])

        # CaseHome._card handles no_data_solution without failing QueryPlan checks
        ch = case_home.CaseHome(store, None)
        ch._card(d["id"], state.cards[-1])
        saved = store.get(d["id"])
        self.assertIn("no_data_solution", saved)
        self.assertEqual(saved["messages"][-1]["kind"], "no_data_solution")

    def test_rules_first_home_solves_without_data_and_never_auto_simulates(self):
        home_dir = Path(tempfile.mkdtemp())
        store = draft_store.DraftStore(home_dir)
        ha = draft_chat.HomeAgent(store, None)
        d = store.create("Predict telecom churn before we collect historical data")
        ha.start(d["id"])
        saved = store.get(d["id"])
        self.assertIn("no_data_solution", saved)
        self.assertEqual(saved["assets"], [])  # Never auto-generates synthetic data
        self.assertTrue(any(m.get("kind") == "no_data_solution" for m in saved["messages"]))

        # Replying with privacy/withholding also works when asking an intent/data question
        d2 = store.create("Predict customer retention")
        ha.start(d2["id"])
        ha.reply(d2["id"], "I don't want to upload my private database; just give me business, data science, and ML plan.")
        saved2 = store.get(d2["id"])
        self.assertIn("no_data_solution", saved2)
        self.assertEqual(saved2["assets"], [])

        # Asking to simulate only asks the user how simulated data should look first (synthetic_way)
        ha.reply(d2["id"], "Actually, can we simulate synthetic data to prototype?")
        saved3 = store.get(d2["id"])
        self.assertEqual(saved3["assets"], [])  # Still zero auto-created data until user designs/approves
        self.assertTrue(any(m.get("kind") == "synthetic_way" for m in saved3["messages"]))


@unittest.skipIf(MISSING, MISSING)
class NoDataApiTests(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_NO_LIVE_MODELS": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.home = Path(tempfile.mkdtemp())
        self.client = TestClient(create_app(self.home))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def test_no_data_solution_endpoint_and_build_unchanged(self):
        r = self.client.post("/api/drafts", json={"problem": "Predict fraud risk next quarter without uploading data yet"}, headers=self.h)
        self.assertEqual(r.status_code, 201, r.text)
        did = r.json()["id"]

        sol_res = self.client.post(
            f"/api/drafts/{did}/no-data-solution",
            json={"refresh": True, "economics": {"monthly_volume": 12000, "base_rate": 0.04, "action_cost": 8, "value_saved": 300}},
            headers=self.h,
        )
        self.assertEqual(sol_res.status_code, 200, sol_res.text)
        body = sol_res.json()
        self.assertEqual(body["solution"]["family"], "fraud")
        self.assertEqual(body["solution"]["economics"]["inputs"]["monthly_volume"], 12000)
        self.assertEqual(body["draft"]["assets"], [])

        # Confirm Build project remains unchanged and still enforces having a table before building a project
        time.sleep(0.1)
        build_res = self.client.post(f"/api/drafts/{did}/build", json={}, headers=self.h)
        self.assertIn(build_res.status_code, (400, 409, 422))
