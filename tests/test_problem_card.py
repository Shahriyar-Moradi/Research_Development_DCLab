"""Package 20.1: the problem card. The family of a problem, its label situation, the questions still open (blocking or
not) and whether a project can be built yet, in dclab_core so the main product reads the same card; and on Home, the
build refused while the card is not ready, with the reasons in plain words. The owner's morning of 2026-10-13 ("I want
build anomaly detection", built into a model of customer ids while the labels question was open) cannot happen again."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core.card import LABEL_WORDS, ProblemCard, problem_card
    from dclab_core.contracts import TextFact
    from dclab_core.understand import understand
except ImportError as error:  # the agent environment may lack pydantic
    MISSING = str(error)
else:
    MISSING = ""

MOMENT = "After delivery, in the daily audit of the day's orders"


def card(text, files=None, data=None, **stated):
    brief = understand(text, files)
    if stated:
        brief = brief.model_copy(update={k: TextFact(value=v, status="stated", quote=v, source="person") for k, v in stated.items()})
    return problem_card(brief, words=text, data=data)


@unittest.skipIf(MISSING, MISSING)
class CardTest(unittest.TestCase):
    def test_the_morning_leaves_two_families_in_play_and_a_blocking_labels_question(self):
        c = card("I want build anomaly detection")
        self.assertIsNone(c.family.value)
        self.assertEqual(c.in_play, ["rare_event", "anomaly"])
        self.assertFalse(c.ready)
        labels = next(q for q in c.questions if q.field == "labels")
        self.assertTrue(labels.blocks)
        self.assertIn("Nothing is labelled", labels.options)
        self.assertTrue(any("confirmed" in b for b in c.blocking))

    def test_the_label_situation_settles_the_family(self):
        self.assertEqual(card("I want build anomaly detection. Nothing is labelled.").family.value, "anomaly")
        self.assertEqual(card("Find the fake deliveries; we have known frauds in a separate file.").family.value, "rare_event")
        files = {"orders.csv": ["order_request_id", "fare"], "known.csv": ["order_request_id", "reason"]}
        c = card("Find the fake deliveries in the orders", files, data={"label_file": True}, prediction_moment=MOMENT)
        self.assertEqual((c.family.value, c.labels.value, c.labels.source), ("rare_event", "file", "data"))
        self.assertEqual(c.outcome, "the cases listed in the file of known cases")
        self.assertTrue(c.ready, c.blocking)  # the unit and the action are asked, but no build waits for them
        self.assertTrue({q.field for q in c.questions if not q.blocks} <= {"unit", "action"})

    def test_a_named_outcome_and_a_moment_make_a_classification_ready(self):
        c = card("Predict which customers churn next month", {"t.csv": ["churned", "tenure"]}, target_column="churned", prediction_moment=MOMENT)
        c = problem_card(c.brief.model_copy(update={"task": c.brief.task.model_copy(update={"value": "binary", "status": "inferred", "reason": "two values"})}),
                         words="Predict which customers churn next month")
        self.assertEqual((c.family.value, c.ready), ("classification", True))

    def test_what_blocks_and_what_does_not(self):
        c = card("Forecast daily demand per store")
        self.assertEqual(c.family.value, "forecasting")
        self.assertEqual({q.field for q in c.questions if q.blocks}, {"outcome", "prediction_moment"})
        self.assertEqual(card("Group our customers into segments").family.value, "segmentation")
        unknown = card("Help me with my data")
        self.assertIn("family", {q.field for q in unknown.questions if q.blocks})
        alone = card("Find the fake deliveries; known frauds are in a file", data={"cases_only": True}, prediction_moment=MOMENT)
        self.assertIn("data", {q.field for q in alone.questions if q.blocks})

    def test_a_no_comes_before_a_yes_in_the_label_words(self):
        self.assertEqual(LABEL_WORDS[0][0], "none")
        self.assertEqual(card("Nothing is labelled, there is no file of known cases.").labels.value, "none")

    def test_the_card_is_a_published_versioned_contract(self):
        import jsonschema

        from dclab_core import contracts, schemas

        self.assertIn("problem_card", contracts.CONTRACTS)
        c = card("I want build anomaly detection")
        self.assertEqual(ProblemCard.read(c.model_dump_json()), c)
        jsonschema.Draft202012Validator(schemas.generate()["problem_card"]).validate(json.loads(c.model_dump_json()))


@unittest.skipIf(MISSING, MISSING)
class RulesTest(unittest.TestCase):
    """What the review found: words that invented an outcome, overrode the person, or matched by accident."""

    def test_a_file_spoken_of_is_never_an_outcome(self):
        c = card("I want build anomaly detection", data={"tables": 1}, prediction_moment=MOMENT)
        c = problem_card(c.brief, words="I want build anomaly detection. A file lists the known ones", data={"tables": 1})
        self.assertIsNone(c.outcome)
        self.assertIn("data", {q.field for q in c.questions if q.blocks})  # add the file: the draft has none
        self.assertIsNone(card("The bikers are in a separate file").labels.value)

    def test_what_the_person_said_beats_a_word(self):
        from dclab_core.contracts import TaskFact

        brief = understand("Predict which customers will churn next month; the market segment column may help")
        brief = brief.model_copy(update={"task": TaskFact(value="binary", status="stated", quote="binary", source="person")})
        c = problem_card(brief, words="Predict which customers will churn next month; the market segment column may help")
        self.assertEqual(c.family.value, "classification")
        self.assertIn("prediction_moment", {q.field for q in c.questions if q.blocks})
        self.assertEqual(problem_card(brief, words="x", family="uplift").family.value, "uplift")  # a stated kind wins over all
        forecast_words = "Predict which customers churn; the forecast team will read the scores"
        self.assertEqual(problem_card(understand(forecast_words), words=forecast_words).family.value, "forecasting")  # words alone
        self.assertEqual(problem_card(brief, words=forecast_words).family.value, "classification")  # a stated task beats them

    def test_a_kind_chosen_on_home_is_the_cards_family(self):
        from dclab_rnd.draft.card import card_of

        draft = {"problem": "Find the fraud in our forecasts", "messages": [], "assets": [], "analysis": {},
                 "understanding": {}, "solution": {"kind": "forecasting", "target": "units", "prediction_moment": MOMENT}}
        self.assertEqual(card_of(draft).family.value, "forecasting")  # the words say fraud; the person chose forecasting
        word_families = {"ranking", "time_to_event", "uplift", "segmentation", "rare_event", "anomaly"}
        for words in ("Rank telecom customers by their risk of leaving", "Predict customer lifetime value", "Use the default settings",
                      "incremental loads of orders", "I would recommend a quick model", "split by market segment"):
            c = card(words)
            self.assertNotIn(c.family.value, word_families, words)  # no word decides these by accident
            self.assertEqual(c.in_play, [], words)
        self.assertEqual(card("Customers who default on their loans").in_play, ["rare_event", "anomaly"])

    def test_a_plain_no_is_no_labels(self):
        for answer in ("We have no confirmed cases yet", "No", "we don't have labels", "Nothing is labelled"):
            self.assertEqual(card(answer).labels.value, "none", answer)

    def test_the_card_as_the_main_products_proposals(self):
        from dclab_core.adapters import main_product as mp

        ready = card("Find the fake deliveries in the orders", {"o.csv": ["order_request_id"]}, data={"label_file": True}, prediction_moment=MOMENT)
        out = mp.problem_card(ready)
        self.assertEqual(list(out["ProblemSpec"]["fields"]), mp.fields_of("ProblemSpec"))
        self.assertEqual(out["ProblemSpec"]["fields"]["task_type"], "binary")
        self.assertEqual(set(out["decision_points"]), {"spec.objective", "target.column"})
        self.assertEqual((out["lock"], out["blocking"]), (True, []))
        open_ = mp.problem_card(card("I want build anomaly detection"))
        self.assertFalse(open_["lock"])
        self.assertTrue(open_["blocking"])
        self.assertEqual(open_["decision_points"]["spec.objective"]["proposal"]["in_play"], ["rare_event", "anomaly"])
        anomaly = mp.problem_card(card("Find unusual orders; nothing is labelled", prediction_moment=MOMENT))
        self.assertIsNone(anomaly["ProblemSpec"]["fields"]["task_type"])
        self.assertIn("confirmed cases make it binary", anomaly["ProblemSpec"]["notes"]["task_type"])


@unittest.skipIf(MISSING, MISSING)
class HomeTest(unittest.TestCase):
    """The morning again, on Home with the replica: the build waits for what the card blocks on, and says why."""

    def setUp(self):
        from dclab_core.replicas import delivery_fraud as dfr
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore
        from tests.test_morning_guard import upload

        self.store = DraftStore(Path(tempfile.mkdtemp()))
        self.agent = HomeAgent(self.store, None)
        self.draft = self.store.create("I want build anomaly detection")["id"]
        self.agent.start(self.draft)
        self.replica = dfr.generate(seed=0, n_orders=6000, n_bikers=240)
        upload(self.store, self.agent, self.draft, "part2-dataset.csv", self.replica.rides)

    def build(self):
        from dclab_rnd.draft.api import build_project
        from dclab_rnd.studio import ProjectStore

        return build_project(self.store, ProjectStore(Path(tempfile.mkdtemp())), self.draft)

    def answer(self, **values):
        self.store.update(self.draft, lambda d: d["understanding"].update(values))

    def test_the_build_waits_for_the_labels_and_the_moment_and_says_why(self):
        from dclab_rnd.draft.api import BuildRefused
        from dclab_rnd.draft.card import card_of

        self.answer(target="it is anomaly detection")  # the morning's answer: names no column
        with self.assertRaises(BuildRefused) as refused:
            self.build()
        self.assertIn("confirmed", str(refused.exception))
        self.assertEqual(card_of(self.store.get(self.draft)).in_play, ["rare_event", "anomaly"])
        self.assertIsNone(card_of(self.store.get(self.draft)).outcome)  # the method named is not an outcome
        self.answer(labels="Nothing is labelled")
        self.assertEqual(card_of(self.store.get(self.draft)).family.value, "anomaly")
        with self.assertRaises(BuildRefused) as refused:
            self.build()
        self.assertIn("prediction made", str(refused.exception))
        self.answer(prediction_moment=MOMENT, entity="The biker")
        project = self.build()  # 20.7: ready, and built as what was said: anomaly detection, with no outcome column
        solution = project["solution"]
        self.assertEqual((solution["kind"], solution["target"], solution["labels"]), ("anomaly", None, "none"))
        self.assertEqual(solution["kind_fields"]["entity_column"], "biker_id")
        self.assertEqual(solution["kind_fields"]["time_column"], "created_date")

    def test_with_the_known_frauds_it_is_a_rare_event_problem_and_builds(self):
        from tests.test_morning_guard import upload

        upload(self.store, self.agent, self.draft, "part2-fraud-sample.csv", self.replica.sample.assign(final_status="DELIVERED"))
        self.answer(labels="A sample of the cases we found", prediction_moment=MOMENT)
        from dclab_rnd.draft.card import card_of

        c = card_of(self.store.get(self.draft))
        self.assertEqual((c.family.value, c.labels.value, c.ready), ("rare_event", "file", True))
        project = self.build()
        self.assertEqual(project["labels"]["key"], ["order_id"])

    def test_a_draft_without_data_is_not_built_while_its_family_is_open(self):
        from dclab_rnd.draft.api import BuildRefused, build_project
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.studio import ProjectStore

        d = self.store.create("Find the unusual orders")["id"]  # a tabular-pack sentence: the labels question comes all the same
        HomeAgent(self.store, None).start(d)
        self.assertEqual(self.store.get(d)["questions"][0]["field"], "target")  # the outcome first …
        self.assertEqual(HomeAgent(self.store, None).next_question(self.store.get(d))[0], "labels")  # … then the labels
        with self.assertRaises(BuildRefused) as refused:
            build_project(self.store, ProjectStore(Path(tempfile.mkdtemp())), d)
        self.assertEqual(refused.exception.rule, "problem card not ready")

    def test_the_built_project_keeps_its_card(self):
        from tests.test_morning_guard import upload

        upload(self.store, self.agent, self.draft, "part2-fraud-sample.csv", self.replica.sample.assign(final_status="DELIVERED"))
        self.answer(labels="A sample of the cases we found", prediction_moment=MOMENT)
        project = self.build()
        self.assertEqual(project["draft"]["card"]["family"]["value"], "rare_event")

    def test_the_home_question_lets_a_person_say_nothing_is_labelled(self):
        from dclab_rnd.draft.chat import HomeAgent

        self.answer(target="the fake deliveries", prediction_moment=MOMENT, entity="The biker")
        asked = None
        for _ in range(8):
            pending = self.agent.pending(self.store.get(self.draft))
            if pending is None:
                break
            if pending["field"] == "labels":
                asked = pending
                break
            self.agent.reply(self.draft, "Upload a sample")
        self.assertIsNotNone(asked)
        self.assertIn("Nothing is labelled", asked.get("options") or [])


@unittest.skipIf(MISSING, MISSING)
class PeoplesOwnWordsTest(unittest.TestCase):
    """20.10 v2: the families said without their technical names, by the person's words, by the shape of an answer, and
    by what the data holds; and the words that must not move a family."""

    def test_less_common_families_said_in_plain_words(self):
        from dclab_core.card import WORDS

        said = {
            "uplift": "Which members would only renew because of the voucher, rather than the ones who renew regardless?",
            "time_to_event": "Estimate the likely tenure of each new tenant, many leases have not ended yet",
            "ranking": "Personalise the homepage carousel: the articles each reader is most likely to read",
            "streams": "We track vibration on every pump and want an early warning",
            "entity_tables": "Will each account make no purchases in the next 14 days? We have their transaction history and their support tickets",
            "text_or_images": "Something that reads the report itself and tells the severity",
        }
        for family, text in said.items():
            self.assertIn(family, [f for f, p in WORDS.items() if p.search(text)], text)
        for text in ("Forecast weekly sales per store; promotions are planned weeks before",
                     "Predict churn from the churned column"):
            self.assertNotIn("time_to_event", [f for f, p in WORDS.items() if p.search(text)], text)

    def test_censored_lifetimes_win_over_a_plan(self):
        c = card("For the budget plan, the expected lifetime of each subscriber, most have not cancelled yet")
        self.assertEqual(c.family.value, "time_to_event")

    def test_an_answer_that_only_gives_the_shape_keeps_the_family_the_words_name(self):
        from dclab_core.card import refine_family

        self.assertEqual(refine_family("classification", "triage new tickets with a model that reads the ticket itself"), "text_or_images")
        self.assertEqual(refine_family("regression", "how long until they cancel, many still have not cancelled yet"), "time_to_event")
        self.assertEqual(refine_family("classification", "predict churn from the churned column"), "classification")
        self.assertEqual(refine_family("anomaly", "reads the note itself"), "anomaly")  # only a shape is refined

    def test_a_negated_history_of_cases_is_nothing_labelled(self):
        for text in ("We have no record of confirmed fraud so far", "We have never confirmed any case"):
            self.assertEqual(next(v for v, p in LABEL_WORDS if p.search(text)), "none", text)
        self.assertEqual(next(v for v, p in LABEL_WORDS if p.search("We have a file of confirmed frauds")), "file")
        c = card("No history of confirmed fraud yet; a daily shortlist of the most unusual rides")
        self.assertEqual(c.family.value, "anomaly")

    def test_what_the_data_holds_names_the_family(self):
        c = card("Which customers will stop ordering?", data={"tables": 3, "columns": True}, target_definition="no order in the following month")
        self.assertEqual((c.family.value, c.family.source), ("entity_tables", "data"))
        c = card("Which pumps fail? We read their sensors", data={"tables": 2, "columns": True}, target_definition="a failure within a day")
        self.assertNotEqual(c.family.value, "entity_tables")
        c = card("Which tickets are urgent?", data={"columns": True, "text_only": True})
        self.assertEqual((c.family.value, c.family.source), ("text_or_images", "data"))

    def test_what_must_not_move_a_family(self):
        """The review's cases (20.10 v2): words in passing never override a family or an outcome column."""
        from dclab_core.card import WORDS, label_said, refine_family

        def hits(text):
            return [f for f, p in WORDS.items() if p.search(text)]

        for text in ("Predict which customers will churn next month. Customers who haven't cancelled yet are still active.",
                     "Forecast demand for each store next week; censored sales when stock ran out",
                     "Predict the expected lifetime value of each customer", "Predict how long the delivery lasts, in minutes"):
            self.assertNotIn("time_to_event", hits(text), text)
        self.assertNotIn("streams", hits("An early-warning score for students at risk of failing"))
        self.assertNotIn("streams", hits("Predict the house price; we log the temperature of each room"))
        self.assertIn("streams", hits("We log the vibration of every pump and want an early warning"))
        self.assertNotIn("uplift", hits("Delivery times, when the zones change their mind about routing"))
        # a stated regression or classification is kept; only an answer that gave a shape alone is refined
        self.assertEqual(refine_family("regression", "how long each delivery lasts, in minutes", shape_only=False), "regression")
        self.assertEqual(refine_family("regression", "Predict the expected lifetime value of each customer"), "regression")
        self.assertEqual(refine_family("classification", "we track the current balance"), "classification")
        # a file of known cases said apart from a "no ... of" is a file
        self.assertEqual(label_said("We have no record of past chargebacks, but a file of known frauds exists"), "file")
        self.assertEqual(label_said("no history of confirmed fraud yet"), "none")
        # several tables never override a stated family; a label file is not a related table
        c = problem_card(understand("Flag risky orders before dispatch"), words="Flag risky orders before dispatch",
                         data={"tables": 2, "label_file": True, "columns": True}, family="classification")
        self.assertEqual(c.family.value, "classification")
        # a lifetime said beside a named yes or no column is asked, never inferred
        brief = understand("Which customers stay? the expected tenure, many have not cancelled yet", {"t.csv": ["churned", "tenure"]})
        brief = brief.model_copy(update={"target_column": TextFact(value="churned", status="stated", quote="churned", source="person"),
                                         "task": brief.task.model_copy(update={"value": "binary", "status": "inferred", "reason": "two values"})})
        c = problem_card(brief, words="Which customers stay? the expected tenure, many have not cancelled yet")
        self.assertIsNone(c.family.value)
        self.assertIn("time_to_event", c.in_play)

    def test_a_shape_only_answer_is_marked_and_a_named_family_is_not(self):
        from dclab_core.conversation import read_answer

        self.assertEqual(read_answer("family", "a yes or no for each note").get("family_shape_only"), "yes")
        self.assertIsNone(read_answer("family", "a classification, yes or no").get("family_shape_only"))

    def test_a_text_only_table_is_read_from_its_data_card(self):
        from dclab_rnd.draft.card import _text_only

        cols = lambda *p: {"tables": [{"name": "t", "columns": [{"name": n, "purpose": q} for n, q in p]}]}
        self.assertTrue(_text_only(cols(("id", "identifier"), ("author", "entity"), ("body", "text"), ("urgent", "target_candidate")), "urgent"))
        self.assertFalse(_text_only(cols(("body", "text"), ("price", "feature"), ("urgent", "target_candidate")), "urgent"))
        self.assertFalse(_text_only(cols(("id", "identifier"), ("urgent", "target_candidate")), "urgent"))


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(MISSING, MISSING)
class RouteTest(unittest.TestCase):
    def test_the_card_route(self):
        import os
        from unittest import mock

        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"}), \
                TestClient(create_app(Path(tempfile.mkdtemp()))) as c:
            h = {"X-DCLab-Token": c.get("/api/config").json()["csrf"]}
            d = c.post("/api/drafts", json={"problem": "I want build anomaly detection"}, headers=h).json()
            got = c.get(f"/api/drafts/{d['id']}/card").json()
            self.assertEqual((got["ready"], got["in_play"]), (False, ["rare_event", "anomaly"]))
            self.assertEqual(c.get("/api/drafts/nope0000/card").status_code, 404)
