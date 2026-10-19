"""Package 30.13: Problem Card v2 and the fit verdict. The framework's fields (the decision and who owns it, what each
mistake costs, the team's capacity, the success numbers, today's baseline, the horizon) are filled from the person's
answers or left open with who can answer them, and never block; a v1 card and a v3 data card still read; G2's verdict is
one of three: fits, fits with changes (each change a card edit with why, which a person accepts and the check then
passes), or does not fit (the missing data named)."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401

    HAVE_ML = True
except ImportError:  # the agent environment
    HAVE_ML = False


def _card(words: str, answers=None, columns=None, family=None):
    from dclab_core.card import problem_card
    from dclab_core.understand import understand

    brief = understand(words, {"table": columns} if columns else None)
    return problem_card(brief, words=words, data={"columns": bool(columns), "tables": 1} if columns else None, family=family, answers=answers)


class CardTests(unittest.TestCase):
    def test_the_new_fields_are_filled_from_answers_or_owned_and_never_block(self):
        card = _card("Detect card fraud at authorization time. The outcome is is_fraud.", columns=["amount", "is_fraud"],
                     answers={"capacity": "About 80 a day", "costs": "A missed fraud costs the money; a false alarm annoys a good customer. "
                                                                    "A missed fraud costs more than a false alarm.", "decision_owner": "The risk lead"})
        self.assertEqual((card.capacity.k, card.capacity.period), (80, "day"))
        self.assertEqual(card.error_costs.worse, "miss")
        self.assertEqual(card.decision_owner.value, "The risk lead")
        owned = {f.field: f.owner for f in card.open_fields}
        self.assertNotIn("capacity", owned)
        self.assertIn("baseline_today", owned)
        self.assertTrue(all(owned.values()))  # every field nobody filled has who can answer it
        bare = _card("Detect card fraud at authorization time. The outcome is is_fraud.", columns=["amount", "is_fraud"])
        self.assertEqual((bare.ready, bare.blocking), (card.ready, card.blocking))  # the new fields change nothing a build waits for
        self.assertEqual(card.complete, card.ready)

    def test_a_capacity_is_the_number_a_team_acts_on_and_never_a_year_or_out_of_range(self):
        from dclab_core.card import capacity_said

        cases = {"About 80 a day": (80, "day"), "We get about 15,000,000 transactions a month and can review 300 a day": (300, "day"),
                 "None yet: 0 checked daily": (None, None), "40,000 orders a day; two analysts check about 150 a day": (150, "day"),
                 "Since 2019 we check some daily": (None, None), "2k a day": (2000, "day"), "1.5k a day": (1500, "day"),
                 "About 200 calls a month": (200, "month"), "a few hundred orders a day": (None, None)}
        for text, want in cases.items():
            got = capacity_said(text)
            self.assertEqual((got.k, got.period), want, text)
            self.assertEqual(got.said.value, text)  # the words always kept whole

    def test_which_mistake_is_worse_is_read_from_the_comparison_itself(self):
        from dclab_core.card import costs_said

        self.assertEqual(costs_said("A missed fraud costs more than a false alarm.").worse, "miss")
        self.assertEqual(costs_said("A miss costs less than a false alarm, which hurts more").worse, "false_alarm")
        self.assertEqual(costs_said("false alarms annoy customers; a missed case costs money").worse, None)
        self.assertTrue(costs_said("false alarms annoy customers").false_alarm.known)  # the plural too
        card = _card("We fill in the missing delivery estimates on the app.")
        self.assertIn("error_costs", {f.field for f in card.open_fields})  # an action is never read as a cost

    def test_a_v1_card_still_reads(self):
        from dclab_core.card import ProblemCard

        card = _card("Predict which customers churn next month.")
        stored = json.loads(card.model_dump_json(include={"brief", "family", "in_play", "labels", "outcome", "questions", "ready", "blocking"}))
        stored["schema_version"] = "dclab_core.problem_card.v1"
        read = ProblemCard.read(stored)
        self.assertEqual((read.schema_version, read.open_fields, read.complete), ("dclab_core.problem_card.v2", [], False))


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class FitTests(unittest.TestCase):
    def test_a_named_outcome_that_is_not_a_column_fits_with_a_change_and_the_change_makes_it_fit(self):
        import numpy as np
        import pandas as pd

        from dclab_core.datacard import card_of
        from dclab_core.meet import fit

        rng = np.random.default_rng(0)
        table = pd.DataFrame({"amount": rng.normal(50, 10, 300), "is_fraud": rng.choice([0, 1], 300, p=[0.9, 0.1])})
        words = "Detect fraud. The outcome is fraud_flag."
        card = _card(words, columns=list(table.columns))
        card = card.model_copy(update={"brief": card.brief.model_copy(update={"target_column": card.brief.target_column.model_copy(
            update={"value": "fraud_flag", "status": "stated", "quote": "fraud_flag", "source": "person"})})})
        verdict = fit(card, card_of({"t": table}))
        self.assertEqual(verdict.verdict, "fits_with_changes")
        edit = verdict.edits[0]
        self.assertEqual((edit.field, edit.value, verdict.back_to), ("target_column", "is_fraud", "problem"))
        fixed = card.model_copy(update={"brief": card.brief.model_copy(update={"target_column": card.brief.target_column.model_copy(update={"value": edit.value, "quote": edit.value})})})
        self.assertEqual(fit(fixed, card_of({"t": table}, target="is_fraud")).verdict, "fits")  # accepted, the check passes

    def test_noisy_labels_ask_to_be_judged_on_the_confirmed_ones_and_a_v3_card_reads(self):
        from dclab_core.datacard import DataCard, card_of
        from dclab_core.meet import fit
        from dclab_core.replicas import retail_returns as rr

        table = rr.generate(seed=0).returns
        table = table.drop(columns=[c for c in table.columns if c.startswith("_")])
        data = card_of({"returns": table}, target="label")
        self.assertTrue(any(d.name == "noisy_labels" and d.applies for d in data.detected_modifiers))
        self.assertEqual({c.name: c.known_at_prediction for c in data.tables[0].columns}["review_status"], "no")
        card = _card("Find staff return fraud. The outcome is label.", columns=list(table.columns))
        verdict = fit(card, data)
        self.assertIn("noisy_labels", {e.code for e in verdict.edits})
        stored = json.loads(data.model_dump_json(exclude={"detected_modifiers", "fit"}))
        for t in stored["tables"]:
            for c in t["columns"]:
                c.pop("known_at_prediction", None)
        stored["schema_version"] = "dclab_core.data_card.v3"
        self.assertEqual(DataCard.read(stored).schema_version, "dclab_core.data_card.v4")

    def test_each_edit_accepted_clears_its_mismatch(self):
        import numpy as np
        import pandas as pd

        from dclab_core.datacard import card_of
        from dclab_core.meet import check, fit

        rng = np.random.default_rng(2)
        n = 400
        orders = pd.DataFrame({"order_id": np.arange(n), "customer_id": rng.integers(0, 60, n), "amount": rng.normal(40, 8, n),
                               "is_fraud": rng.choice([0, 1], n, p=[0.95, 0.05]), "churned": rng.choice([0, 1], n)})
        data = card_of({"orders": orders}, target="is_fraud")
        # the person said anomaly detection while the known cases are here: the edit is the family, and it clears the mismatch
        anomaly = _card("Find anomalies in the orders. The outcome is is_fraud.", columns=list(orders.columns), family="anomaly")
        edit = next(e for e in fit(anomaly, data).edits if e.code == "labels_contradict")
        self.assertEqual((edit.field, edit.value), ("family", "rare_event"))
        fixed = _card("Find anomalies in the orders. The outcome is is_fraud.", columns=list(orders.columns), family="rare_event")
        self.assertNotIn("labels_contradict", {m.code for m in check(fixed, data).mismatches if m.blocking})
        # an outcome that can never be one (an entity) asks for another column, never "does not fit"
        from dclab_core.contracts import TextFact

        refused = _card("Predict churn.", columns=list(orders.columns))
        refused = refused.model_copy(update={"brief": refused.brief.model_copy(update={"target_column": TextFact(
            value="customer_id", status="stated", quote="customer_id", source="person")})})
        verdict = fit(refused, card_of({"orders": orders}, target="customer_id"))
        self.assertEqual(verdict.verdict, "fits_with_changes")
        self.assertIn(next(e.value for e in verdict.edits if e.field == "target_column"), {"is_fraud", "churned"})

    def test_the_unit_edit_names_the_tables_key_and_predicting_per_row_clears_the_mismatch(self):
        import numpy as np
        import pandas as pd

        from dclab_core.contracts import TextFact
        from dclab_core.datacard import card_of
        from dclab_core.meet import check, fit

        rng = np.random.default_rng(3)
        n = 400
        orders = pd.DataFrame({"order_id": np.arange(n), "customer_id": rng.integers(0, 50, n), "is_fraud": rng.choice([0, 1], n, p=[0.9, 0.1])})
        data = card_of({"orders": orders}, target="is_fraud")
        card = _card("Detect fraud for each customer. The outcome is is_fraud.", columns=list(orders.columns))
        card = card.model_copy(update={"brief": card.brief.model_copy(update={"unit": TextFact(value="One customer", status="stated", quote="One customer", source="person")})})
        edit = next(e for e in fit(card, data).edits if e.field == "unit")
        self.assertTrue(edit.value.startswith("one row per") and "order_id" in edit.value, edit.value)
        per_row = card.model_copy(update={"brief": card.brief.model_copy(update={"unit": TextFact(value=edit.value, status="stated", quote=edit.value, source="person")})})
        self.assertNotIn("unit_grain", {m.code for m in check(per_row, data).mismatches})

    def test_a_date_written_after_the_outcome_is_never_marked_known_and_a_label_source_is_not(self):
        import numpy as np
        import pandas as pd

        from dclab_core.datacard import card_of

        rng = np.random.default_rng(4)
        n = 300
        table = pd.DataFrame({"created_at": pd.date_range("2024-01-01", periods=n, freq="h"), "chargeback_date": pd.date_range("2024-02-01", periods=n, freq="h"),
                              "amount": rng.normal(20, 5, n), "is_fraud": rng.choice([0, 1], n), "label_source": rng.choice(["rule", "expert"], n)})
        known = {c.name: c.known_at_prediction for c in card_of({"t": table}, target="is_fraud").tables[0].columns}
        self.assertEqual((known["chargeback_date"], known["label_source"], known["is_fraud"]), ("unknown", "no", None))

    def test_a_forecast_with_no_dates_does_not_fit_and_names_what_is_missing(self):
        import numpy as np
        import pandas as pd

        from dclab_core.datacard import card_of
        from dclab_core.meet import fit

        table = pd.DataFrame({"store": np.arange(200) % 7, "units": np.random.default_rng(1).poisson(30, 200)})
        card = _card("Forecast the units each store sells next week. The outcome is units.", columns=list(table.columns), family="forecasting")
        verdict = fit(card, card_of({"t": table}, target="units"))
        self.assertEqual((verdict.verdict, verdict.back_to), ("does_not_fit", "data"))
        self.assertIn("a column that says when each row happened", verdict.missing)



@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class DraftTests(unittest.TestCase):
    def test_on_a_draft_the_verdict_proposes_its_edit_and_accepting_it_clears_it(self):
        import tempfile

        from dclab_core.replicas import retail_returns as rr
        from dclab_rnd.draft import pipeline
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.card import accept_fit, card_of, meeting
        from dclab_rnd.draft.store import DraftStore

        store = DraftStore(Path(tempfile.mkdtemp()))
        agent = HomeAgent(store, None)
        d = store.create("Find staff return fraud from the returns log")["id"]
        table = rr.generate(seed=0).returns
        table.drop(columns=[c for c in table.columns if c.startswith("_")]).to_csv(store.data_dir(d) / "returns.csv", index=False)
        asset = pipeline.new_asset(store, d, "upload", "returns.csv", "returns.csv")
        pipeline.run(store, d, asset["id"], agent=agent)
        store.update(d, lambda x: x.setdefault("understanding", {}).update(target="label", budget="5 returns a day"))
        _, verdict = meeting(store.get(d), store, with_fit=True)
        self.assertEqual(verdict.verdict, "fits_with_changes")
        self.assertEqual(card_of(store.get(d)).capacity.k, 5)  # Home's budget answer is the team's capacity
        store.update(d, lambda x: accept_fit(x, [e.model_dump() for e in verdict.edits]))
        _, after = meeting(store.get(d), store, with_fit=True)
        self.assertNotIn("noisy_labels", {e.code for e in after.edits})  # accepted once, never proposed again
        self.assertEqual(store.get(d)["fit_accepted"][0]["code"], verdict.edits[0].code)


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class LedgerTests(unittest.TestCase):
    def test_the_verdict_is_recorded_from_what_it_read_and_the_persons_acceptance_beside_it(self):
        import tempfile

        from dclab_core.ledger import check_state
        from dclab_core.meet import CardEdit, FitVerdict, Mismatch, ProblemDataCheck
        from dclab_rnd import ledger
        from dclab_rnd.ledger import capture

        book = ledger.FileLedger(Path(tempfile.mkdtemp()), "w")
        ledger.install(book)
        self.addCleanup(ledger.install, None)
        checked = ProblemDataCheck(mismatches=[Mismatch(code="unit_grain", blocking=True, question="per customer?", evidence={})], ready=False)
        verdict = FitVerdict(verdict="fits_with_changes", edits=[CardEdit(field="unit", value="one row per order_id", why="x", code="unit_grain")],
                             reasons=["unit_grain: per customer?"], back_to="problem")
        card = _card("Detect fraud. The outcome is is_fraud.")
        capture.fit_verdict({"id": "d1"}, verdict, checked, card)
        capture.fit_verdict({"id": "d1"}, verdict, checked, card, accepted=True)
        first, second = [r for r in book.records() if r.point == "data.fit_verdict"]
        self.assertEqual((first.used, second.used, second.answer("person").value), ("rule", "person", "fits_with_changes"))
        self.assertEqual(first.state["blocking"], 1)  # read from the mismatches, never from the verdict's own edits
        self.assertFalse({"answered"} & set(first.state))
        self.assertEqual(check_state(dict(first.state)), [])


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class RouteTests(unittest.TestCase):
    def setUp(self):
        import os
        import tempfile
        from unittest import mock

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

    def test_the_accept_route_says_why_it_refuses_and_writes_what_it_accepts(self):
        from dclab_core.replicas import retail_returns as rr
        from dclab_rnd.draft import pipeline

        c = self.client
        d = c.post("/api/drafts", json={"problem": "Find staff return fraud from the returns log"}, headers=self.h).json()
        none = c.post(f"/api/drafts/{d['id']}/fit/accept", json={}, headers=self.h)
        self.assertEqual(none.status_code, 409)
        self.assertIn("no table", none.json()["detail"])
        store = c.app.state.drafts
        table = rr.generate(seed=0).returns
        table.drop(columns=[x for x in table.columns if x.startswith("_")]).to_csv(store.data_dir(d["id"]) / "returns.csv", index=False)
        asset = pipeline.new_asset(store, d["id"], "upload", "returns.csv", "returns.csv")
        pipeline.run(store, d["id"], asset["id"])
        store.update(d["id"], lambda x: x.setdefault("understanding", {}).update(target="label"))
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/fit/accept", json={"fields": "unit"}, headers=self.h).status_code, 422)
        card = c.get(f"/api/drafts/{d['id']}/card").json()
        self.assertEqual(card["fit"]["verdict"], "fits_with_changes")
        done = c.post(f"/api/drafts/{d['id']}/fit/accept", json={"fields": ["target_definition"]}, headers=self.h)
        self.assertEqual(done.status_code, 200, done.text)
        self.assertEqual(done.json()["accepted"], ["target_definition"])
        self.assertNotIn("noisy_labels", {e["code"] for e in done.json()["fit"]["edits"]})
        recorded = c.app.state.services.ledger.records(point="data.fit_verdict", draft=d["id"])
        self.assertTrue(recorded and recorded[-1].used == "person")


if __name__ == "__main__":
    unittest.main()
