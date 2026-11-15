"""Tests for solving business, data science, and ML problems without uploaded data."""

import json
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

    def test_the_stance_is_a_stated_lack_of_data_never_a_nearby_word(self):
        """The review of 22ad1bf8: "no data quality issues", "no data scientist", "without data leakage", a privacy word or
        the word "simulate" alone all matched; several plain statements of having no data did not."""
        for words in ("I have no data quality issues, the table is clean", "There is no data scientist on the team",
                      "The file is confidential, I will upload it now", "This is sensitive: please compare the two regions in my table",
                      "I cannot share the private key, here is the csv", "Can you design an A/B test without data leakage?",
                      "simulate what happens if price rises 5%", "کیفیت داده نیست مشکل ما"):
            self.assertFalse(no_data.detect_no_data_stance(words)["matched"], words)
        for words in ("we lack historical data", "our dataset does not exist yet", "I've got zero data at the moment",
                      "we are still gathering the data", "هیچ داده‌ای جمع نکرده‌ایم", "هنوز اطلاعاتی نداریم", "دیتایی در دسترس نیست",
                      "We can't share our customer data, it is confidential"):
            self.assertTrue(no_data.detect_no_data_stance(words)["matched"], words)
        self.assertTrue(no_data.detect_no_data_stance("simulate what happens if price rises 5%")["wants_simulation"])
        # a table that is still being read is data: only a failed upload leaves the person without one
        self.assertFalse(no_data.detect_no_data_stance("I have no data yet", {"assets": [{"status": "cleaning"}]})["matched"])
        self.assertTrue(no_data.detect_no_data_stance("I have no data yet", {"assets": [{"status": "failed"}]})["matched"])

    def test_a_model_blueprint_only_when_the_words_name_something_to_predict(self):
        """The review of 22ad1bf8: a route plan, a why-question, an A/B test's design and a report all got "binary
        classification, predict outcome_label"."""
        for words in ("I have 12 vans and 300 orders a day and no data exported yet. How should I plan the routes to cut fuel cost?",
                      "We have no data yet. Complaints rose since March. Why, and what should we do first?",
                      "I don't have data. How should we design an A/B test for our new checkout page?",
                      "I have no data. Can you explain the difference between precision and recall?",
                      "No data yet. I need a weekly dashboard of sales by region.", "no data yet, help me"):
            fit = no_data.blueprint_fit(words)
            self.assertFalse(fit["fits"], words)
        fit = no_data.blueprint_fit("No data yet. I want to predict which customers will churn next month.")
        self.assertEqual((fit["fits"], fit["family"], fit["family_source"]), (True, "churn", "words"))
        self.assertTrue(no_data.blueprint_fit("We have no data yet; we want to detect fraud in card transactions.")["fits"])
        self.assertEqual(no_data.blueprint_fit("no data yet, help me")["family_source"], "default")  # nothing pointed anywhere: said, not hidden

    def test_a_number_is_the_persons_or_it_is_marked_assumed(self):
        """The review of 22ad1bf8: defaults (10,000 units, 8%, cost 15, value 180, 35%) were shown as facts and were a source
        for the lead agent's number check; "rate" was read inside "operate"; a correction lost to the first number."""
        econ = no_data.compute_decision_economics("We operate at 95% uptime. Accuracy must be above 90%; churn unknown. "
                                                  "300 enterprise accounts churn; last year 12 customers left. We generate $2M revenue")
        inputs = econ["inputs"]
        self.assertEqual((inputs["monthly_volume"], inputs["monthly_volume_assumed"]), (300, False))  # the unit named first, not a later one
        self.assertTrue(inputs["base_rate_assumed"] and inputs["action_cost_assumed"] and inputs["value_per_positive_assumed"])
        self.assertEqual(econ["assumed"], ["base rate", "cost per action", "value per caught case", "share of actions that work"])
        said = " ".join(econ["as_said"])
        self.assertEqual(json.loads(econ["as_said"][0]), {"no_data_stated_inputs": {"monthly_volume": 300}})
        for assumed in ("0.08", "15.0", "180.0", "0.35", "break_even"):
            self.assertNotIn(assumed, said)  # an assumed figure, or one computed from it, is never a source
        # a correction wins; a phrase's number is its own ("a $50 intervention cost, and $2,500 saved" is cost 50)
        fixed = no_data.compute_decision_economics("20000 customers, 10% churn. Correction: we have 50000 customers and the churn rate is 4%.")["inputs"]
        self.assertEqual((fixed["monthly_volume"], fixed["base_rate"]), (50000, 0.04))
        phrase = no_data.compute_decision_economics("5,000 enterprise accounts, an 8% annual churn rate, a $50 CSM intervention cost, and $2,500 saved per retained account.")["inputs"]
        self.assertEqual((phrase["action_cost"], phrase["value_per_positive"]), (50.0, 2500.0))
        # a typed number out of range is refused, never computed with
        for bad in ({"base_rate": "abc"}, {"base_rate": 5}, {"action_cost": -3}, {"monthly_volume": -100}):
            with self.assertRaises(ValueError):
                no_data.compute_decision_economics("x", overrides=bad)

    def test_the_card_says_assumed_and_cites_only_records_that_exist(self):
        sol = no_data.solve_without_data("No data yet. I want to predict which customers will churn next month.")
        self.assertIn("5 of 5 inputs below are DCLab's placeholders", sol["markdown"])
        self.assertIn("assumed by DCLab", sol["markdown"])
        self.assertNotIn("23.8", sol["summary"])  # the summary quotes no figure computed from placeholders
        self.assertIn("placeholders", sol["summary"])
        from dclab_rnd import evidence_index

        index = evidence_index.EvidenceIndex.load()
        for pit in sol["data_science"]["pitfall_checklist"]:
            self.assertNotIn(pit["cite"], ("DCLAB-R01", "DCLAB-L01"))  # no made-up fallback id
            if pit["cite"]:
                self.assertIsNotNone(index.get(pit["cite"]), pit["cite"])
        self.assertTrue(sol["citations"])  # the search's records are read by their own id
        # one split on the card: the rules' plan repeats the validation split, never a second, different one
        self.assertIn("DCLab's rules' plan (no model debate ran)", sol["markdown"])
        self.assertNotIn("stratified", sol["markdown"])
        self.assertNotIn("Planner-Challenger-Judge Debate Verdict", sol["markdown"])

    def test_no_model_is_called_when_no_gateway_is_given(self):
        """The review of 22ad1bf8: the "no model" path took the process's own gateway for the task planner."""
        from dclab_rnd import models as M

        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "0"}), mock.patch.object(M, "installed", side_effect=AssertionError("the process's gateway was taken")):
            sol = no_data.solve_without_data("No data yet. I want to predict which customers will churn next month.")
        self.assertEqual(sol["ml"]["catalog_plan"]["routed_to"], "deterministic")
        self.assertEqual(sol["harness_report"]["mode"], "deterministic_floor")

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
        self.assertTrue(sol["as_said"][0].startswith('{"no_data_stated_inputs":'))  # the person's own numbers, and only those
        self.assertEqual(econ["assumed"], ["share of actions that work"])
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
        state.case = C.hear(state.case, "We want to predict churn but do not have a dataset yet and don't want synthetic data unless asked. "
                                        "We have 5,000 customers and 12% of them leave each year.")
        # the model passes two numbers the person said and two it made up: only the person's are used
        out = A._design_solution_without_data(state, problem="Reduce churn across 50,000 customers with 30% churn", monthly_volume=5000,
                                              base_rate=0.12, action_cost=10, value_saved=250)
        self.assertTrue(out["blueprint_ready"])
        self.assertEqual(out["ignored_numbers"], ["action_cost", "value_saved"])
        inputs = out["economics"]["inputs"]
        self.assertEqual((inputs["monthly_volume"], inputs["base_rate"], inputs["action_cost_assumed"], inputs["value_per_positive_assumed"]), (5000, 0.12, True, True))
        self.assertIn("cost per action", out["assumed_inputs"])
        self.assertTrue(any(str(c).startswith('{"no_data_stated_inputs":') for c in state.computed))
        self.assertFalse(any("break_even" in str(c) or "250" in str(c) for c in state.computed))  # nothing assumed or made up becomes a source
        self.assertEqual(state.cards[-1]["kind"], "no_data_solution")
        # once a turn; never with a table; never past the case's cap; never for a question that is not a prediction
        self.assertIn("already built", A._design_solution_without_data(state)["error"])
        state.no_data_built = False
        state.spent_eur, state.cap_eur = 0.6, 0.5
        self.assertIn("budget", A._design_solution_without_data(state)["error"])
        state.spent_eur = 0.0
        with_table = A.CaseState(case=state.case, table=A.table_of("customers.csv", [{"name": "customer_id", "kind": "text", "noticed": []}], 10))
        self.assertIn("a table is shared", A._design_solution_without_data(with_table)["error"])
        other = A.CaseState(case=C.hear(C.new("draft-x"), "We have no data yet. Complaints rose since March. Why, and what should we do first?"))
        refused = A._design_solution_without_data(other)
        self.assertIn("no model blueprint fits", refused["error"])
        self.assertEqual(other.cards, [])

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
        # "I will not upload" is no answer to "what should the model predict": that question stays open for the real answer
        target = next(q for q in saved2["questions"] if q["field"] == "target")
        self.assertIsNone(target["answered"])
        cards = sum(1 for m in saved2["messages"] if m.get("kind") == "no_data_solution")
        ha.reply(d2["id"], "Whether a customer cancels in the next 30 days")
        after = store.get(d2["id"])
        self.assertIn("cancels in the next 30 days", str((after.get("understanding") or {}).get("target")))
        self.assertEqual(sum(1 for m in after["messages"] if m.get("kind") == "no_data_solution"), cards)  # an answer is no request for another card

        # Asking to simulate only asks the user how simulated data should look first (synthetic_way)
        ha.reply(d2["id"], "Actually, can we simulate synthetic data to prototype?")
        saved3 = store.get(d2["id"])
        self.assertEqual(saved3["assets"], [])  # Still zero auto-created data until user designs/approves
        self.assertTrue(any(m.get("kind") == "synthetic_way" for m in saved3["messages"]))


class NoDataRouteTests(unittest.TestCase):
    """The review of 22ad1bf8: a word pattern set the route to prediction and recorded it as the person's confirmed choice."""

    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "OPENAI_API_KEY": ""})
        env.start()
        self.addCleanup(env.stop)
        self.store = draft_store.DraftStore(Path(tempfile.mkdtemp()))
        self.ha = draft_chat.HomeAgent(self.store, None)

    def kinds_of(self, draft_id):
        return [m.get("kind") for m in self.store.get(draft_id)["messages"]]

    def test_a_person_with_no_data_is_never_routed_to_a_prediction_by_a_word_pattern(self):
        d = self.store.create("Help me with my customers")
        self.ha.start(d["id"])
        self.ha.reply(d["id"], "There is no data scientist on our team, so I need a simple weekly report of sales by region")
        saved = self.store.get(d["id"])
        intent = saved.get("intent") or {}
        self.assertFalse(intent.get("source") == "user" and intent.get("track") == "prediction", intent)  # never "the person's own choice"
        self.assertNotIn("no_data_solution", saved)
        self.assertNotIn("no_data_solution", self.kinds_of(d["id"]))

    def test_no_blueprint_on_a_question_that_is_not_a_prediction(self):
        for problem in ("I don't have data. Can you explain the difference between precision and recall?",
                        "How should we design an A/B test for our new checkout page? We have no data yet.",
                        "We have no data yet. Complaints rose since March. Why, and what should we do first?"):
            d = self.store.create(problem)
            self.ha.start(d["id"])
            self.assertNotIn("no_data_solution", self.kinds_of(d["id"]), problem)
            self.assertNotIn("no_data_solution", self.store.get(d["id"]), problem)

    def test_a_prediction_with_nothing_named_gets_a_plain_note_not_a_template(self):
        d = self.store.create("Predict something useful for us; we have no data yet")
        self.ha.start(d["id"])
        saved = self.store.get(d["id"])
        if (saved.get("intent") or {}).get("track") == "prediction" and (saved.get("intent") or {}).get("status") != "asked":
            self.assertTrue("no_data_solution" in self.kinds_of(d["id"]) or "no_data_note" in self.kinds_of(d["id"]))
        d2 = self.store.create("Build me a prediction model. We have no data yet.")
        self.ha.start(d2["id"])
        self.assertEqual(self.store.get(d2["id"])["assets"], [])

    def test_the_blueprint_is_drawn_again_only_when_asked_or_given_a_number(self):
        d = self.store.create("Predict telecom churn before we collect historical data")
        self.ha.start(d["id"])
        cards = self.kinds_of(d["id"]).count("no_data_solution")
        self.assertEqual(cards, 1)
        def settle(draft):  # nothing is left open, so the next messages are free notes
            for q in draft.get("questions") or []:
                q["answered"] = q.get("answered") or "settled"

        self.store.update(d["id"], settle)
        base = self.kinds_of(d["id"]).count("no_data_solution")
        self.ha.script_turn(d["id"], "thanks, what does AUC mean?")
        self.assertEqual(self.kinds_of(d["id"]).count("no_data_solution"), base)
        self.store.update(d["id"], settle)  # Home asked its next question after the note: settled too
        self.ha.script_turn(d["id"], "We have 50000 customers and the churn rate is 4%")
        self.assertEqual(self.kinds_of(d["id"]).count("no_data_solution"), base + 1)
        inputs = self.store.get(d["id"])["no_data_solution"]["economics"]["inputs"]
        self.assertEqual((inputs["monthly_volume"], inputs["base_rate"]), (50000, 0.04))


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

        # a typed number that is no number, or out of range, is a 422: never a 500, never a negative break-even
        for bad in ({"base_rate": "abc"}, {"base_rate": 5, "action_cost": -3, "monthly_volume": -100}):
            r = self.client.post(f"/api/drafts/{did}/no-data-solution", json={"economics": bad}, headers=self.h)
            self.assertEqual(r.status_code, 422, r.text)
        # a draft that names nothing to predict gets no model plan from this route
        other = self.client.post("/api/drafts", json={"problem": "Predict something for us; no data yet"}, headers=self.h).json()["id"]
        time.sleep(0.2)
        r = self.client.post(f"/api/drafts/{other}/no-data-solution", json={}, headers=self.h)
        if r.status_code == 200:
            self.assertTrue(r.json()["solution"] is None or r.json()["fit"]["fits"])
        else:
            self.assertEqual(r.status_code, 409, r.text)

        # Confirm Build project remains unchanged and still enforces having a table before building a project
        time.sleep(0.1)
        build_res = self.client.post(f"/api/drafts/{did}/build", json={}, headers=self.h)
        self.assertIn(build_res.status_code, (400, 409, 422))
