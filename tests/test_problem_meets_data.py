"""Package 20.4: the problem meets the data. Each planted mismatch in a suite (the unit is not the grain, the outcome
is an identifier or is absent, the label file is not joined, the moment needs a time the table does not have, the
outcome does not fit the family) is found, with a question and its evidence; a matching pair raises nothing."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd

    from dclab_core import meet
    from dclab_core.card import problem_card
    from dclab_core.contracts import ProblemBrief, TextFact
    from dclab_core.datacard import card_of
    from dclab_core.replicas import delivery_fraud, delivery_time, store_demand, subscription_churn
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""


def said(value):
    return TextFact(value=value, status="stated", quote=value, source="person")


def problem(words, family=None, **facts):
    brief = ProblemBrief(**{k: said(v) for k, v in facts.items()})
    return problem_card(brief, words=words, family=family)


@unittest.skipIf(MISSING, MISSING)
class SuiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        churn = subscription_churn.generate(seed=0).subscribers
        orders = delivery_time.generate(seed=0).orders
        fraud = delivery_fraud.generate(seed=0, n_orders=3000)
        sales = store_demand.generate(seed=0).sales
        moment = "when the subscriber signs up"
        cls.matching = [
            ("churn", problem("Predict which subscribers churn", "classification", target_column="churned", unit="a subscriber",
                              prediction_moment=moment), card_of({"subscribers": churn}, target="churned")),
            ("delivery time", problem("Predict delivery minutes", "regression", target_column="actual_minutes", unit="an order",
                                      prediction_moment="when the order is placed"), card_of({"orders": orders}, target="actual_minutes")),
            ("fraud with its file", problem("Find the frauds; a file lists the known ones", "rare_event", unit="a ride",
                                            prediction_moment="after the ride"),
             card_of({"rides": fraud.rides, "known_frauds": fraud.sample})),
            ("demand", problem("Forecast units sold per store and product", "forecasting", target_column="units", unit="a store",
                               prediction_moment="a week ahead"), card_of({"sales": sales}, target="units")),  # a unit and a two-column key
        ]
        no_time = churn.drop(columns=["signup_time"])
        per_customer = pd.DataFrame({"customer_id": [f"C{i % 200:03d}" for i in range(1000)], "order_id": range(1000),
                                     "amount": np.random.default_rng(0).normal(50, 10, 1000), "churned": [i % 2 for i in range(1000)]})
        cls.planted = [  # (code, the problem, the data)
            ("target_absent", problem("Predict churn", "classification", target_column="will_leave", prediction_moment=moment),
             card_of({"subscribers": churn})),
            ("target_refused", problem("Predict churn", "classification", target_column="subscriber_id", prediction_moment=moment),
             card_of({"subscribers": churn}, target="subscriber_id")),
            ("target_family", problem("Predict the fee", "regression", target_column="churned", prediction_moment=moment),
             card_of({"subscribers": churn}, target="churned")),
            ("unit_grain", problem("Predict which customers churn", "classification", target_column="churned", unit="a customer",
                                   prediction_moment="at month end"), card_of({"orders": per_customer}, target="churned")),
            ("labels_not_joined", problem("Find the frauds; a file lists the known ones", "rare_event", prediction_moment="after the ride"),
             card_of({"rides": fraud.rides})),
            ("time_absent", problem("Predict churn", "classification", target_column="churned", time_column="signup_time",
                                    prediction_moment=moment), card_of({"subscribers": no_time}, target="churned")),
            ("time_absent", problem("Forecast units", "forecasting", target_column="units", prediction_moment="a week ahead"),
             card_of({"sales": sales.drop(columns=["date"])}, target="units")),
            ("labels_contradict", problem("Find the unusual subscribers, nothing is labelled", "anomaly", prediction_moment=moment),
             card_of({"subscribers": churn})),
        ]

    def test_each_planted_mismatch_is_found(self):
        for code, card, data in self.planted:
            with self.subTest(code=code):
                found = meet.check(card, data)
                self.assertIn(code, [m.code for m in found.mismatches], [m.question for m in found.mismatches])
                m = next(m for m in found.mismatches if m.code == code)
                self.assertTrue(m.question and m.evidence)
                if code != "labels_contradict":
                    self.assertFalse(found.ready)
                else:
                    self.assertTrue(found.ready)  # a contradiction is shown, never a block

    def test_a_matching_pair_raises_nothing(self):
        for name, card, data in self.matching:
            with self.subTest(pair=name):
                found = meet.check(card, data)
                self.assertEqual([m.code for m in found.mismatches], [], [m.question for m in found.mismatches])
                self.assertTrue(found.ready)

    def test_the_evidence_holds_names_and_counts_never_values(self):
        churn = subscription_churn.generate(seed=0).subscribers
        for code, card, data in self.planted:
            text = meet.check(card, data).model_dump_json()
            for value in churn["subscriber_id"].astype(str).head(50):
                self.assertNotIn(value, text, code)

    def test_a_family_read_from_words_gives_way_to_the_named_outcome(self):
        churn = subscription_churn.generate(seed=0).subscribers
        inferred = problem("Predict the monthly fee of each subscriber", target_column="churned", prediction_moment="at signup")
        self.assertNotEqual(inferred.family.status, "stated")
        found = meet.check(inferred, card_of({"subscribers": churn}, target="churned"))
        if found.mismatches:
            self.assertEqual([(m.code, m.blocking) for m in found.mismatches], [("target_family", False)])
        self.assertTrue(found.ready)

    def test_what_must_not_block(self):
        from dclab_core.contracts import ProblemBrief, TextFact

        n = 600
        rng = np.random.default_rng(1)
        orders = pd.DataFrame({"order_id": range(n), "customer_id": [f"C{i % 60:03d}" for i in range(n)], "delivery_zone_id": [f"Z{i % 9}" for i in range(n)],
                               "delivery_id": range(n), "amount": rng.normal(size=n), "is_fraud": (rng.random(n) < 0.05).astype(int),
                               "created_at": pd.date_range("2025-01-01", periods=n, freq="h")})
        known = orders[orders.is_fraud == 1].head(8)[["order_id"]].assign(reason="fake")
        files = card_of({"orders": orders.drop(columns=["is_fraud"]), "labels": known})
        fraud = problem("Find the frauds; a file lists the known ones", "rare_event", prediction_moment="after the order")
        self.assertNotIn("labels_not_joined", [m.code for m in meet.check(fraud, files, joined=True).mismatches])  # the host joined it
        unjoined = [m for m in meet.check(fraud, files, joined=False).mismatches if m.code == "labels_not_joined"]
        self.assertEqual([m.blocking for m in unjoined], [False])  # the file is here: a question about its key
        inferred = problem_card(ProblemBrief(target_column=said("is_fraud"), prediction_moment=said("at order"),
                                             unit=TextFact(value="customer", status="inferred", reason="the file's name")),
                                words="", family="classification")
        found = meet.check(inferred, card_of({"customer_transactions": orders}, target="is_fraud"))
        self.assertTrue(found.ready)  # a unit read from a file's name is a question at most
        delivery = problem("Predict whether each delivery is late", "classification", target_column="is_fraud", unit="a delivery",
                           prediction_moment="at order")
        self.assertNotIn("unit_grain", [m.code for m in meet.check(delivery, card_of({"t": orders}, target="is_fraud")).mismatches])
        rambling = problem("Find the frauds", "classification", target_column="is_fraud", prediction_moment="at order",
                           unit="One customer, usually; " + "they place many orders over the months " * 12)
        meet.check(rambling, card_of({"t": orders}, target="is_fraud"))  # never raises: what is inserted is clipped

    def test_customer_id_named_as_the_outcome_blocks(self):
        frame = pd.DataFrame({"customer_id": [f"C{i % 8}" for i in range(400)], "spend": range(400)})
        found = meet.check(problem("Predict the customer", "classification", target_column="customer_id", prediction_moment="now"),
                           card_of({"t": frame}, target="customer_id"))
        self.assertEqual([m.code for m in found.mismatches if m.blocking], ["target_refused"])
        self.assertEqual(meet.ProblemDataCheck.read(found.model_dump_json()), found)


@unittest.skipIf(MISSING, MISSING)
class StudioTest(unittest.TestCase):
    def test_a_draft_build_is_refused_on_a_blocking_mismatch_and_the_card_says_why(self):
        import tempfile
        import time

        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        churn = subscription_churn.generate(seed=0).subscribers.head(600)
        d = client.post("/api/drafts", json={"problem": "Predict which subscribers churn"}, headers=h).json()["id"]
        client.put(f"/api/drafts/{d}/data", params={"filename": "subscribers.csv"}, content=churn.to_csv(index=False).encode(), headers=h)
        deadline = time.time() + 60
        while time.time() < deadline and client.get(f"/api/drafts/{d}").json()["assets"][0]["status"] not in ("ready", "failed"):
            time.sleep(0.2)
        store = client.app.state.services.drafts
        store.update(d, lambda x: x["understanding"].update(target="subscriber_id", prediction_moment="at signup", labels="A column marks them"))
        from dclab_rnd.draft.card import meeting

        broken = {"id": d, "assets": [{"id": "a0", "status": "ready", "clean_file": "gone.parquet", "name": "t"}], "active_asset": "a0"}
        failed = meeting(broken, client.app.state.services.drafts)  # a check that cannot run never lets a build through
        self.assertEqual([(m.code, m.blocking) for m in failed.mismatches], [("not_checked", True)])
        r = client.post(f"/api/drafts/{d}/build", headers=h)
        self.assertEqual(r.status_code, 409, r.text)
        self.assertIn("cannot be the outcome", r.text)
        card = client.get(f"/api/drafts/{d}/card").json()
        self.assertEqual([m["code"] for m in card["meets_data"]["mismatches"] if m["blocking"]], ["target_refused"])
        store.update(d, lambda x: x["understanding"].update(target="churned"))
        self.assertEqual(client.post(f"/api/drafts/{d}/build", headers=h).status_code, 201)


if __name__ == "__main__":
    unittest.main()
