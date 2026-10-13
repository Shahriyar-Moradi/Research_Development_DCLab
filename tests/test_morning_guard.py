"""The guard against the owner's morning of 2026-10-13 (package 20.0). Two files of a delivery-fraud challenge went to
two drafts, both "I want build anomaly detection": the orders, and the known frauds (every row DELIVERED, a reason per
case, the order id different on every row). The frauds' draft became a project whose target was customer_id, a
column picked because it had few values, while the labels question was still open. Here the same morning runs on the
invented replica: no key, constant or row number is ever an outcome, nothing picks the outcome for the person, the
list of cases is recognised and pointed at the orders, it is never built alone, and with the orders it becomes their
labels. The model may ask about any field still open; the plan's order is a suggestion."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd

    from dclab_core.replicas import delivery_fraud as dfr
    from dclab_rnd.draft import pipeline
    from dclab_rnd.draft.api import BuildRefused, build_project
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.store import DraftStore
    from dclab_rnd.studio import ProjectStore, graph
    from dclab_rnd.studio.data import profile_table, target_refusal
    from tests.test_draft_flow import ScriptedClient
except ImportError as error:  # pydantic and pandas are not in every environment
    MISSING = str(error)
else:
    MISSING = ""

PROBLEM = "I want build anomaly detection"
MOMENT = "After delivery, in the daily audit of the day's orders"


def upload(store, agent, draft_id, name, frame):
    frame.to_csv(store.data_dir(draft_id) / name, index=True)  # saved with its row numbers, as the owner's files were
    asset = pipeline.new_asset(store, draft_id, "upload", name, name)
    pipeline.run(store, draft_id, asset["id"], agent=agent)


@unittest.skipIf(MISSING, MISSING)
class NoGuessedOutcomeTest(unittest.TestCase):
    def test_a_key_a_constant_or_a_row_number_is_never_an_outcome_and_a_key_name_is_never_proposed(self):
        n = 300
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"Unnamed: 0": np.arange(n), "customer_id": rng.choice([2, 7, 9], n), "order_id": np.arange(10_000, 10_000 + n),
                              "final_status": "DELIVERED", "fare": rng.normal(size=n), "is_fraud": rng.integers(0, 2, n)})
        profile = profile_table(frame)
        columns = {c["name"]: c for c in profile["columns"]}
        self.assertEqual(profile["target_candidates"][0], "is_fraud")
        self.assertFalse({"Unnamed: 0", "customer_id", "order_id", "final_status"} & set(profile["target_candidates"]))
        for name, reason in (("Unnamed: 0", "row number"), ("order_id", "key"), ("final_status", "one value")):
            self.assertIn(reason, target_refusal(columns[name]) or "", name)
        self.assertIsNone(target_refusal(columns["customer_id"]))  # few values: a person may still name it, it is never guessed


@unittest.skipIf(MISSING, MISSING)
class TheGuardRefusesOnlyWhatItShouldTest(unittest.TestCase):
    def test_whole_number_amounts_and_binary_names_ending_in_id_stay_outcomes_and_text_does_not(self):
        rng = np.random.default_rng(3)
        n = 200
        frame = pd.DataFrame({"salary": rng.choice(np.arange(30_000, 200_000), n, replace=False), "price": np.round(rng.uniform(1, 9_000, n)),
                              "counter": np.arange(1, n + 1), "is_paid": rng.integers(0, 2, n), "valid": rng.integers(0, 2, n),
                              "note": [f"customer wrote message number {i} about the delivery being late and the box opened" for i in range(n)]})
        profile = profile_table(frame)
        columns = {c["name"]: c for c in profile["columns"]}
        for name in ("salary", "price", "is_paid", "valid"):
            self.assertIsNone(target_refusal(columns[name]), name)
        self.assertIn("is_paid", profile["target_candidates"])  # "id" at the end of a word is not a key's name
        self.assertIn("key", target_refusal(columns["counter"]))  # 1, 2, 3, …
        self.assertIn("free text", target_refusal(columns["note"]) or "")

    def test_an_outcome_is_taken_only_when_a_person_or_the_table_names_it(self):
        from dclab_rnd.studio.data import named_outcome

        rng = np.random.default_rng(6)
        n = 200
        frame = pd.DataFrame({"customer_id": rng.choice([2, 7, 9], n), "order_id": np.arange(n), "late": rng.integers(0, 2, n),
                              "fare": rng.normal(size=n)})
        profile = profile_table(frame)
        self.assertIsNone(named_outcome(profile, "I want build anomaly detection"))  # nothing named: DCLab asks
        self.assertEqual(named_outcome(profile, "predict whether an order is late"), "late")
        self.assertIsNone(named_outcome(profile, "group by order_id"))  # a key named in a sentence is still a key
        self.assertEqual(named_outcome(profile_table(frame.rename(columns={"late": "target"})), "anything"), "target")

    def test_the_chat_never_offers_a_key_named_column(self):
        from dclab_rnd.draft.chat import rank_targets

        rng = np.random.default_rng(4)
        frame = pd.DataFrame({"delivery_id": rng.integers(0, 160, 300), "biker_id": rng.integers(0, 60, 300), "fare": rng.normal(size=300),
                              "abused": rng.integers(0, 2, 300)})
        ranked = rank_targets(profile_table(frame), "Find the bikers who abuse deliveries")
        self.assertEqual(ranked[0], "abused")
        self.assertFalse({"delivery_id", "biker_id"} & set(ranked))

    def test_a_labelled_table_named_fraud_is_a_table_to_learn_from(self):
        from dclab_rnd.studio.labels import cases_table

        rng = np.random.default_rng(5)
        n = 400
        frame = pd.DataFrame({"transaction_id": np.arange(n), "transaction_type": rng.choice(list("abcd"), n), "amount": rng.normal(size=n),
                              "status": "SETTLED", "chargeback": rng.integers(0, 2, n)})
        self.assertIsNone(cases_table(profile_table(frame), "fraud_transactions.csv"))  # chargeback 0/1 is its outcome


@unittest.skipIf(MISSING, MISSING)
class TheMorningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        replica = dfr.generate(seed=0, n_orders=6000, n_bikers=240)
        cls.orders = replica.rides
        cls.cases = replica.sample.assign(final_status="DELIVERED")  # every known case was delivered, as in the owner's file

    def setUp(self):
        self.store = DraftStore(Path(tempfile.mkdtemp()))
        self.agent = HomeAgent(self.store, None)
        self.orders_draft = self.store.create(PROBLEM)
        self.agent.start(self.orders_draft["id"])
        upload(self.store, self.agent, self.orders_draft["id"], "part2-dataset.csv", self.orders)
        self.cases_draft = self.store.create(PROBLEM)
        self.agent.start(self.cases_draft["id"])
        upload(self.store, self.agent, self.cases_draft["id"], "part2-fraud-sample.csv", self.cases)

    def test_the_list_of_cases_is_recognised_pointed_at_the_orders_and_never_built_alone(self):
        draft = self.store.get(self.cases_draft["id"])
        cases = draft["analysis"]["cases_table"]
        self.assertIn("order_id", cases["keys"])
        self.assertIn("reason", cases["case_kinds"])
        self.assertEqual(cases["same_on_every_row"]["final_status"], "DELIVERED")
        found = cases["tables"][0]
        present = int(self.cases["order_id"].astype(str).isin(set(self.orders["order_id"].astype(str))).sum())  # the replica plants a few absent
        self.assertEqual((found["draft_id"], found["key"], found["matched"], found["of"]), (self.orders_draft["id"], "order_id", present, len(self.cases)))
        said = draft["messages"][-1]["text"] if draft["messages"][-1].get("kind") == "analysis" else \
            next(m["text"] for m in reversed(draft["messages"]) if m.get("kind") == "analysis")
        self.assertIn("list of known cases", said)
        self.assertIn("part2-dataset.csv", said)
        self.assertNotIn("customer", " ".join(draft["analysis"]["profile"]["target_candidates"]))
        with self.assertRaises(BuildRefused):
            build_project(self.store, ProjectStore(Path(tempfile.mkdtemp())), self.cases_draft["id"])

    def test_a_kind_of_case_named_with_fraud_is_still_a_kind_not_an_outcome(self):
        from dclab_rnd.studio.labels import cases_table

        renamed = self.cases.rename(columns={"reason": "fraud_reason"})
        self.assertIsNotNone(cases_table(profile_table(renamed), "sample.csv"))
        no_state = self.cases.drop(columns=["status", "final_status"])  # the file's name alone does not make a list of cases
        self.assertIsNone(cases_table(profile_table(no_state), "known_frauds.csv"))

    def test_naming_an_outcome_lets_the_person_build_what_they_chose(self):
        self.store.update(self.cases_draft["id"], lambda d: d["understanding"].update(target="biker_fare", prediction_moment=MOMENT))
        project = build_project(self.store, ProjectStore(Path(tempfile.mkdtemp())), self.cases_draft["id"])
        self.assertTrue(project["id"])

    def test_with_the_orders_in_the_same_draft_the_cases_become_their_labels(self):
        upload(self.store, self.agent, self.cases_draft["id"], "part2-dataset.csv", self.orders)
        self.store.update(self.cases_draft["id"], lambda d: d["understanding"].update(prediction_moment=MOMENT))
        draft = self.store.get(self.cases_draft["id"])
        labels = next(a for a in draft["assets"] if a.get("kind") == "labels")
        self.assertEqual(labels["label_review"]["key"], ["order_id"])
        project = build_project(self.store, ProjectStore(Path(tempfile.mkdtemp())), self.cases_draft["id"])
        self.assertEqual(project["labels"]["key"], ["order_id"])

    def test_a_key_saved_as_the_solution_is_refused_by_the_graph(self):
        upload(self.store, self.agent, self.cases_draft["id"], "part2-dataset.csv", self.orders)
        self.store.update(self.cases_draft["id"], lambda d: d["understanding"].update(prediction_moment=MOMENT))
        projects = ProjectStore(Path(tempfile.mkdtemp()))
        project = build_project(projects=projects, drafts=self.store, draft_id=self.cases_draft["id"])
        for target, allowed in (("Unnamed: 0", False), ("order_id", False), ("is_case", True)):
            verdict = graph.check(projects.get(project["id"]), "set_solution", "human", target=target)
            self.assertEqual(verdict.allowed, allowed, target)


@unittest.skipIf(MISSING, MISSING)
class TheWizardNeverPicksTheOutcomeTest(unittest.TestCase):
    def test_the_proposal_waits_for_a_named_outcome_and_refuses_a_key(self):
        import os
        from unittest import mock

        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app
        from tests.test_draft_flow import wait

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"}), \
                TestClient(create_app(Path(tempfile.mkdtemp()))) as c:
            h = {"X-DCLab-Token": c.get("/api/config").json()["csrf"]}
            d = c.post("/api/drafts", json={"problem": PROBLEM}, headers=h).json()
            rng = np.random.default_rng(1)
            table = pd.DataFrame({"customer_id": rng.choice([2, 7, 9], 400), "order_id": np.arange(400), "fare": rng.normal(size=400),
                                  "fake": rng.integers(0, 2, 400)})
            c.put(f"/api/drafts/{d['id']}/data?filename=orders.csv", content=table.to_csv(index=False).encode(), headers=h)
            wait(lambda: any(a["status"] == "ready" for a in c.get(f"/api/drafts/{d['id']}").json()["assets"]))
            nothing = c.post(f"/api/drafts/{d['id']}/solution/proposal", headers=h)
            self.assertEqual(nothing.status_code, 422)  # before: the profile's first candidate was taken
            self.assertIn("Choose the outcome column", nothing.text)
            self.assertEqual(c.post(f"/api/drafts/{d['id']}/solution/proposal", json={"target": "order_id"}, headers=h).status_code, 422)
            self.assertEqual(c.post(f"/api/drafts/{d['id']}/solution/proposal", json={"target": "fake"}, headers=h).status_code, 200)
            solution = {"target": "order_id", "task": "regression", "prediction_moment": "After each delivery, in the daily audit",
                        "forbidden": [], "identifiers": [], "time_column": None, "group_column": None, "text_columns": [], "metric": None, "notes": ""}
            refused = c.put(f"/api/drafts/{d['id']}/solution", json=solution, headers=h)
            self.assertEqual(refused.status_code, 422)
            self.assertIn("key", refused.text)

    def test_the_build_route_says_what_is_missing(self):
        import os
        from unittest import mock

        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app
        from tests.test_draft_flow import wait

        cases = dfr.generate(seed=0, n_orders=6000, n_bikers=240).sample.assign(final_status="DELIVERED")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"}), \
                TestClient(create_app(Path(tempfile.mkdtemp()))) as c:
            h = {"X-DCLab-Token": c.get("/api/config").json()["csrf"]}
            d = c.post("/api/drafts", json={"problem": PROBLEM}, headers=h).json()
            c.put(f"/api/drafts/{d['id']}/data?filename=known.csv", content=cases.to_csv(index=False).encode(), headers=h)
            wait(lambda: any(a["status"] == "ready" for a in c.get(f"/api/drafts/{d['id']}").json()["assets"]))
            built = c.post(f"/api/drafts/{d['id']}/build", json={}, headers=h)
            self.assertEqual(built.status_code, 409, built.text)
            self.assertIn("list of known cases", built.text)


@unittest.skipIf(MISSING, MISSING)
class TheModelMayAskAnyOpenFieldTest(unittest.TestCase):
    def test_an_open_field_out_of_the_plans_order_is_asked_and_a_known_one_is_not(self):
        store = DraftStore(Path(tempfile.mkdtemp()))
        d = store.create(PROBLEM)
        HomeAgent(store, None).start(d["id"])
        self.assertEqual(store.get(d["id"])["questions"][0]["field"], "target")  # the plan's first suggestion
        client = ScriptedClient([  # the morning's refused question: what is done with a flag, before the plan reached it
            ("ask_user", {"field": "action", "question": "What happens to a delivery once it is flagged?",
                          "why": "Reviewing and blocking cost different mistakes.", "options": ["Sent for review", "Blocked"]}),
            ("record", {"field": "prediction_moment", "value": "After delivery", "quote": "after delivery"}),
            ("ask_user", {"field": "prediction_moment", "question": "When is it flagged?", "why": "w", "options": ["a", "b"]}),
        ])
        HomeAgent(store, client).reply(d["id"], "it is anomaly detection, flagged after delivery")
        draft = store.get(d["id"])
        texts = [q["text"] for q in draft["questions"]]
        self.assertNotEqual(HomeAgent(store, None).next_question(draft)[0], "action")
        self.assertIn("What happens to a delivery once it is flagged?", texts)  # refused for the order before
        self.assertNotIn("When is it flagged?", texts)  # known now: not asked again

    def test_the_outcome_comes_first_and_a_field_this_draft_does_not_use_is_not_asked(self):
        store = DraftStore(Path(tempfile.mkdtemp()))
        d = store.create("Predict which customers cancel their subscription")
        HomeAgent(store, None).start(d["id"])
        store.update(d["id"], lambda x: x["questions"].clear() or x["messages"].clear())  # the outcome not asked yet
        client = ScriptedClient([
            ("ask_user", {"field": "action", "question": "What do you do with each prediction?", "why": "w", "options": ["a", "b"]}),
            ("ask_user", {"field": "target", "question": "Which customers count as cancelled?", "why": "w", "options": ["a", "b"]}),
        ])
        HomeAgent(store, client).reply(d["id"], "Hello")
        texts = [q["text"] for q in store.get(d["id"])["questions"]]
        self.assertNotIn("What do you do with each prediction?", texts)  # before the outcome
        self.assertIn("Which customers count as cancelled?", texts)
        client = ScriptedClient([("ask_user", {"field": "budget", "question": "How many alerts a day?", "why": "w", "options": ["a", "b"]})])
        HomeAgent(store, client).reply(d["id"], "Customers who cancel within 30 days")
        self.assertNotIn("How many alerts a day?", [q["text"] for q in store.get(d["id"])["questions"]])  # a fraud pack's field


if __name__ == "__main__":
    unittest.main()
