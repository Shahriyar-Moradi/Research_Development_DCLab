"""Home reads the kind of help first (package H2): the rules, the model's reading and its checks, the question on a
tie, the tracks that are not a prediction, the person's choice (a label on the reading), the guard and the route.

The sentences here are written for these tests; none is a request of the H1 suite (dev or sealed).
"""
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

from dclab_rnd import ledger  # noqa: E402  (the readings are decisions in the ledger on main)
from dclab_rnd.agents import default_registry  # noqa: E402
from dclab_rnd.draft import intent, plan  # noqa: E402
from dclab_rnd.draft.chat import DraftGuard, HomeAgent, Turn  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402

BY_TRACK = {
    "prediction": ["Predict which subscribers will cancel next quarter", "Forecast weekly orders for each warehouse"],
    "analysis": ["What was our average basket size last year?", "How many tickets did each team close in March?"],
    "preparation": ["Merge the two customer exports and remove duplicate rows", "Clean up the missing values in my survey file"],
    "reporting": ["Build a dashboard of daily signups for the team", "I need a weekly report of refunds per region"],
    "business": ["Why did profit drop in the second quarter?", "Should we open a second store or expand online?"],
    "advice": ["What is the difference between precision and recall?", "Is it ok to use the test set to pick features?"],
}
TIE = "Predict churn and build a dashboard of it"  # prediction and reporting, equally strong: Home asks


class Scripted:
    """A model client that answers with fixed replies and keeps what it was told."""

    def __init__(self, *replies):
        self.replies, self.seen, self.outputs = list(replies), [], []

    def complete(self, messages, tools=None):
        self.seen.append(messages)
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        return {"content": reply}

    def output(self, ok, note=""):
        self.outputs.append(ok)


def wait(fn, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("timed out waiting")


class RuleTests(unittest.TestCase):
    def test_each_track_is_read_from_ordinary_requests(self):
        for track, sentences in BY_TRACK.items():
            for sentence in sentences:
                reading = intent.read(sentence)
                self.assertEqual(reading["track"], track, sentence)
                self.assertEqual((reading["status"], reading["source"]), ("stated", "rules"))
                self.assertIn(reading["quote"].lower(), sentence.lower())  # the words behind the reading are the person's
                self.assertTrue(plan.quoted(reading["quote"], [sentence]))  # and a valid quote by the product's own rule

    def test_a_request_with_no_signal_starts_as_a_prediction_and_says_so(self):
        reading = intent.read("Help with my customers")
        self.assertEqual((reading["track"], reading["status"], reading["source"]), ("prediction", "inferred", "default"))
        self.assertIsNone(reading["quote"])

    def test_a_real_tie_is_asked(self):
        reading = intent.read(TIE)
        self.assertEqual(reading["status"], "asked")
        self.assertEqual(set(reading["tracks"][:2]), {"prediction", "reporting"})

    def test_an_answer_is_read_by_its_label_or_its_words(self):
        labels = {t: intent.LABELS[t] for t in ("prediction", "reporting")}
        self.assertEqual(intent.from_answer(intent.LABELS["reporting"], labels), "reporting")
        self.assertEqual(intent.from_answer("a dashboard please", labels), "reporting")
        self.assertIsNone(intent.from_answer("Something else", labels))  # names nothing: not a choice
        self.assertIsNone(intent.from_answer("I'm not sure", labels))

    def test_prediction_requests_that_count_or_join_stay_predictions(self):
        for sentence in ("Predict how many units we sell per store per week", "How many bikes will be rented tomorrow?",
                         "How much will each customer spend next month?", "Flag loans that will default and show the number of defaults by region",
                         "Detect fraudulent transactions and merge the card and merchant tables", "Build a churn model and report the results to the board",
                         "Forecast how many tickets we get each day", "Which support tickets get escalated?",
                         "Rank clients for the term-deposit call campaign",
                         "Estimate how many minutes a food delivery will take, to show the customer when they place the order."):
            reading = intent.read(sentence)
            self.assertEqual((reading["track"], reading["status"]), ("prediction", "stated"), sentence)

    def test_problem_descriptions_of_predictions_stay_predictions(self):
        """The problem analyst's dev briefs (several sentences each, all predictions) never leave the prediction track,
        and none is asked: a model's description that says "report the F1" or "decide on" is still a prediction."""
        briefs = json.loads((ROOT / "dclab_rnd/agent_eval/analyst/briefs_dev.json").read_text())
        self.assertEqual(len(briefs), 12)
        for brief in briefs:
            reading = intent.read(brief["description"])
            self.assertEqual(reading["track"], "prediction", brief["id"])
            self.assertNotEqual(reading["status"], "asked", brief["id"])
        for sentence in ("Help me decide which leads to call first", "Monitor machines and alert before a failure"):
            self.assertEqual(intent.read(sentence)["track"], "prediction", sentence)

    def test_a_report_riding_on_a_prediction_is_asked_not_taken(self):
        for sentence in ("Build a churn model and send me a weekly report of its scores",
                         "Find customers who will upgrade, and give me a monthly report"):
            reading = intent.read(sentence)
            self.assertEqual(reading["status"], "asked", sentence)
            self.assertIn("prediction", reading["tracks"][:2])

    def test_a_request_too_short_to_quote_is_not_a_rules_reading(self):
        reading = intent.read("Predict.")
        self.assertEqual(reading["source"], "default")  # never a reading whose words fail the quote rule


class ModelReadingTests(unittest.TestCase):
    def test_the_model_reads_an_unclear_request_and_its_quote_is_checked(self):
        client = Scripted('{"tracks": ["analysis"], "quote": "my customers", "confidence": 0.7}')
        reading = intent.read("Help with my customers", client)
        self.assertEqual((reading["track"], reading["source"], reading["quote"]), ("analysis", "model", "my customers"))
        self.assertEqual(client.outputs, [True])

    def test_a_quote_the_person_did_not_write_is_refused(self):
        client = Scripted('{"tracks": ["business"], "quote": "grow revenue fast", "confidence": 0.9}')
        reading = intent.read("Help with my customers", client)
        self.assertEqual(reading["source"], "default")
        self.assertEqual(client.outputs, [False])

    def test_an_unknown_track_or_a_failure_leaves_the_rules_reading(self):
        self.assertEqual(intent.read("Help with my customers", Scripted('{"tracks": ["poetry"], "quote": "my customers"}'))["source"], "default")
        self.assertEqual(intent.read("Help with my customers", Scripted(RuntimeError("down")))["source"], "default")
        self.assertEqual(intent.read(TIE, Scripted("not json"))["status"], "asked")

    def test_a_clear_request_never_calls_the_model(self):
        client = Scripted()
        intent.read(BY_TRACK["business"][0], client)
        self.assertEqual(client.seen, [])


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.store = DraftStore(self.home)
        self.book = ledger.FileLedger(self.home / "ledger", "wtest")  # the readings and the person's labels are kept here
        ledger.install(self.book)
        self.addCleanup(ledger.install, None)

    def readings(self):
        return self.book.records(point="request.intent")

    def labels(self):
        return self.book.labels(point="request.intent")

    def start(self, problem, **kw):
        d = self.store.create(problem)
        HomeAgent(self.store, None, **kw).start(d["id"])
        return self.store.get(d["id"])

    def test_a_prediction_runs_as_before(self):
        d = self.start(BY_TRACK["prediction"][0])
        self.assertEqual(d["intent"]["track"], "prediction")
        self.assertTrue(d["workflow"])
        self.assertTrue(d["questions"])
        self.assertIsNotNone(d.get("pack"))

    def test_another_track_gets_no_pack_workflow_or_ml_question(self):
        for track in ("analysis", "preparation", "reporting", "business", "advice"):
            d = self.start(BY_TRACK[track][0])
            self.assertEqual(d["intent"]["track"], track)
            self.assertFalse(d.get("workflow"), track)
            self.assertEqual([q for q in d["questions"] if not q.get("brief")], [], track)  # H4: only the track's brief asks
            said = " ".join(m.get("text") or "" for m in d["messages"] if m.get("role") == "agent")  # H4: a brief question may follow
            self.assertIn(intent.NEXT[track], said)
            if track in ("preparation", "reporting", "business"):  # H7: analysis is answered from the table now
                self.assertIn("not built yet", said)  # never a promise of what does not exist

    def test_a_chosen_pack_means_a_prediction(self):
        d = self.store.create(BY_TRACK["analysis"][0], pack="imbalanced")
        HomeAgent(self.store, None).start(d["id"])
        self.assertEqual(self.store.get(d["id"])["intent"]["source"], "user")
        self.assertEqual(self.store.get(d["id"])["intent"]["track"], "prediction")

    def test_a_tie_is_asked_and_the_answer_opens_the_track(self):
        d = self.start(TIE)
        q = d["questions"][0]
        self.assertEqual(q["field"], "intent")
        self.assertIn(intent.LABELS["reporting"], q["options"])
        HomeAgent(self.store, None).reply(d["id"], intent.LABELS["reporting"])
        d = self.store.get(d["id"])
        self.assertEqual((d["intent"]["track"], d["intent"]["source"]), ("reporting", "user"))
        self.assertFalse(d.get("workflow"))
        reading = self.readings()[0]
        self.assertEqual([(lab.record, lab.value) for lab in self.labels()], [(reading.id, "reporting")])

    def test_something_else_is_not_a_choice(self):
        d = self.start(TIE)
        HomeAgent(self.store, None).reply(d["id"], "Something else")
        d = self.store.get(d["id"])
        self.assertEqual((d["intent"]["source"], d["intent"]["status"]), ("rules", "inferred"))
        self.assertEqual(self.labels(), [])
        self.assertTrue(any("change the type of help above" in (m.get("text") or "") for m in d["messages"]))

    def test_while_undecided_data_draws_no_workflow_and_the_answer_settles_it(self):
        import numpy as np
        import pandas as pd

        from dclab_rnd.draft import analyze

        d = self.start(TIE)
        frame = pd.DataFrame({"churned": np.arange(40) % 2, "region": ["a", "b"] * 20})
        self.store.update(d["id"], lambda x: x.update(analysis=analyze.analyze(frame), assets=[{"id": "a1", "status": "ready", "name": "t.csv"}], active_asset="a1"))
        agent = HomeAgent(self.store, None)
        agent.data_ready(d["id"], self.store.get(d["id"])["assets"][0])
        self.assertFalse(self.store.get(d["id"]).get("workflow"))
        self.assertEqual([r for r in self.book.records() if r.point.startswith("workflow.template")], [])  # no ML workflow drawn
        agent.reply(d["id"], intent.LABELS["reporting"])
        d = self.store.get(d["id"])
        self.assertEqual(d["intent"]["track"], "reporting")
        self.assertFalse(d.get("workflow"))

    def test_a_label_never_lands_on_another_drafts_reading(self):
        a = self.start(BY_TRACK["business"][0])  # read as business: a sample
        b = self.store.create(BY_TRACK["analysis"][0], pack="imbalanced")  # a chosen pack: no reading, so no sample
        agent = HomeAgent(self.store, None)
        agent.start(b["id"])
        agent.choose_intent(b["id"], "analysis")
        reading_a = next(r for r in self.readings() if r.draft == a["id"])
        self.assertEqual(self.book.labels(record=reading_a.id), [])  # b had no reading of its own: nothing labelled anywhere
        self.assertEqual(self.labels(), [])

    def test_the_persons_earlier_choice_survives_start(self):
        d = self.store.create(BY_TRACK["business"][1])
        self.store.update(d["id"], lambda x: x.update(intent=intent.chosen("prediction")))
        HomeAgent(self.store, None).start(d["id"])
        d = self.store.get(d["id"])
        self.assertEqual((d["intent"]["track"], d["intent"]["source"]), ("prediction", "user"))
        self.assertTrue(d["workflow"])

    def test_editing_the_problem_rereads_a_reading_but_keeps_a_choice(self):
        d = self.start(BY_TRACK["analysis"][0])
        agent = HomeAgent(self.store, None)
        self.store.update(d["id"], lambda x: x.update(problem=BY_TRACK["prediction"][0]))
        agent.replan(d["id"])
        d = self.store.get(d["id"])
        self.assertEqual(d["intent"]["track"], "prediction")
        self.assertTrue(d["workflow"])
        agent.choose_intent(d["id"], "analysis")
        self.store.update(d["id"], lambda x: x.update(problem=BY_TRACK["prediction"][1]))
        agent.replan(d["id"])
        d = self.store.get(d["id"])
        self.assertEqual((d["intent"]["track"], d["intent"]["source"]), ("analysis", "user"))
        self.assertFalse(d.get("workflow"))

    def test_editing_the_problem_never_calls_a_model(self):
        class Counting:
            calls = 0

            def complete(self, *a, **k):
                Counting.calls += 1
                return {"content": "", "tool_calls": []}
        d = self.start(BY_TRACK["analysis"][0])
        self.store.update(d["id"], lambda x: x.update(problem="Predict which customers will cancel next month"))
        HomeAgent(self.store, Counting(), intent_client=Counting()).replan(d["id"])
        self.assertEqual(Counting.calls, 0)
        self.assertEqual(self.store.get(d["id"])["intent"]["track"], "prediction")

    def test_a_pack_chosen_while_start_reads_still_makes_a_prediction(self):
        import threading

        gate, started = threading.Event(), threading.Event()

        class Slow:
            def complete(self, *a, **k):
                started.set()
                gate.wait(10)
                return {"content": '{"tracks": ["analysis"], "quote": "my customers", "confidence": 0.8}'}
        d = self.store.create("Help with my customers")
        reader = threading.Thread(target=HomeAgent(self.store, None, intent_client=Slow()).start, args=(d["id"],))
        reader.start()
        self.assertTrue(started.wait(10))
        agent = HomeAgent(self.store, None)
        agent.set_pack(d["id"], "tabular", "user", "You chose this pack.")
        chooser = threading.Thread(target=agent.pack_chosen, args=(d["id"],))
        chooser.start()
        gate.set()
        reader.join(10)
        chooser.join(10)
        d = self.store.get(d["id"])
        self.assertEqual((d["intent"]["track"], d["intent"]["source"]), ("prediction", "user"))
        self.assertTrue(d["workflow"])

    def test_a_model_set_intent_to_the_same_track_writes_nothing(self):
        d = self.start("Help with my customers and their orders")
        from dclab_rnd.agents import default_registry
        tool = default_registry().get("set_intent")
        before = len(self.readings())
        self.assertTrue(tool.handler(Turn(HomeAgent(self.store, None), d["id"]), track="prediction", quote="their orders")["unchanged"])
        self.assertEqual(len(self.readings()), before)

    def test_the_intent_question_does_not_take_an_ml_question(self):
        d = self.start(TIE)
        HomeAgent(self.store, None).reply(d["id"], intent.LABELS["prediction"])
        d = self.store.get(d["id"])
        ml = [q for q in d["questions"] if q["field"] != "intent"]
        self.assertTrue(ml)  # the prediction path asks its first question as usual
        self.assertTrue(d["workflow"])

    def test_choosing_another_track_closes_the_prediction_and_labels_the_reading(self):
        d = self.start(BY_TRACK["prediction"][1])
        agent = HomeAgent(self.store, None)
        agent.choose_intent(d["id"], "analysis")
        d = self.store.get(d["id"])
        self.assertEqual((d["intent"]["track"], d["intent"]["source"], d["intent"]["rule_track"]), ("analysis", "user", "prediction"))
        self.assertIsNone(d["workflow"])
        self.assertTrue(all(q.get("answered") for q in d["questions"] if not q.get("brief")))  # H4: the new track's brief may ask
        labels = self.labels()
        self.assertEqual([(lab.value, lab.source) for lab in labels], [("analysis", "user_decision_exposed")])
        agent.choose_intent(d["id"], "prediction")  # and back: the workflow returns
        self.assertTrue(self.store.get(d["id"])["workflow"])

    def test_confirming_the_reading_is_an_approval(self):
        d = self.start(BY_TRACK["business"][0])
        HomeAgent(self.store, None).choose_intent(d["id"], "business")
        reading = next(r for r in self.readings() if r.draft == d["id"])  # keyed by its own draft
        self.assertEqual([(lab.record, lab.value) for lab in self.labels()], [(reading.id, "business")])  # the rule's choice, approved

    def test_the_sample_holds_counts_and_names_never_the_words(self):
        problem = "Merge the two customer exports and remove duplicate rows"
        d = self.start(problem)
        rows = self.readings()
        self.assertEqual(len(rows), 1)
        self.assertNotIn("customer exports", json.dumps(rows[0].model_dump(mode="json")))  # counts and codes, never the words
        self.assertEqual(rows[0].state["words"], len(problem.split()))
        self.assertEqual(rows[0].draft, d["id"])

    def test_build_words_off_a_prediction_explain_instead_of_building(self):
        d = self.start(BY_TRACK["analysis"][1])
        HomeAgent(self.store, None).reply(d["id"], "let's build the solution")
        d = self.store.get(d["id"])
        self.assertIn("change the type of help", d["messages"][-1]["text"])
        self.assertIsNone(d.get("project_id"))

    def test_data_on_another_track_is_described_without_an_outcome_question(self):
        import numpy as np
        import pandas as pd

        from dclab_rnd.draft import analyze

        d = self.start(BY_TRACK["analysis"][0])
        frame = pd.DataFrame({"basket": np.arange(50.0), "region": ["a", "b"] * 25})
        self.store.update(d["id"], lambda x: x.update(analysis=analyze.analyze(frame), assets=[{"id": "a1", "status": "ready", "name": "t.csv"}], active_asset="a1"))
        HomeAgent(self.store, None).data_ready(d["id"], self.store.get(d["id"])["assets"][0])
        d = self.store.get(d["id"])
        self.assertIn("analysis", [m.get("kind") for m in d["messages"]])  # the data described (H7: an answer may follow it)
        self.assertEqual([q for q in d["questions"] if not q.get("brief")], [])  # H4: only the track's brief asks
        self.assertFalse(d.get("workflow"))


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.store = DraftStore(Path(tempfile.mkdtemp()))

    def tool(self, name):
        return default_registry().get(name)

    def test_ml_moves_are_refused_off_the_prediction_track(self):
        d = self.store.create(BY_TRACK["reporting"][0])
        HomeAgent(self.store, None).start(d["id"])
        turn, guard = Turn(HomeAgent(self.store, None), d["id"]), DraftGuard()
        self.assertIn("not a prediction", guard.refusal(self.tool("set_pack"), turn, {"key": "tabular"})["error"])
        self.assertIn("not a prediction", guard.refusal(self.tool("propose_workflow"), turn, {"nodes": []})["error"])

    def test_set_intent_needs_the_persons_words_and_respects_their_choice(self):
        d = self.store.create("Help with my customers and their orders")
        agent = HomeAgent(self.store, None)
        agent.start(d["id"])
        turn, guard, tool = Turn(agent, d["id"]), DraftGuard(), self.tool("set_intent")
        self.assertIn("quote", guard.refusal(tool, turn, {"track": "analysis", "quote": "sum the revenue"})["error"])
        self.assertIn("Unknown", guard.refusal(tool, turn, {"track": "poetry", "quote": "my customers"})["error"])
        self.assertIsNone(guard.refusal(tool, turn, {"track": "analysis", "quote": "their orders"}))
        tool.handler(turn, track="analysis", quote="their orders")
        d2 = self.store.get(d["id"])
        self.assertEqual((d2["intent"]["track"], d2["intent"]["source"]), ("analysis", "model"))
        self.assertFalse(d2.get("workflow"))
        agent.choose_intent(d["id"], "prediction")
        self.assertIn("chose", guard.refusal(tool, turn, {"track": "analysis", "quote": "their orders"})["error"])


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

    def test_the_intent_route(self):
        c = self.client
        d = c.post("/api/drafts", json={"problem": BY_TRACK["business"][1]}, headers=self.h).json()
        wait(lambda: (c.get(f"/api/drafts/{d['id']}").json().get("intent") or {}).get("track"))
        self.assertEqual(c.get(f"/api/drafts/{d['id']}").json()["intent"]["track"], "business")
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/intent", json={"track": "magic"}, headers=self.h).status_code, 422)
        out = c.post(f"/api/drafts/{d['id']}/intent", json={"track": "prediction"}, headers=self.h)
        self.assertEqual(out.status_code, 200)
        self.assertEqual((out.json()["intent"]["track"], out.json()["intent"]["source"]), ("prediction", "user"))
        wait(lambda: c.get(f"/api/drafts/{d['id']}").json().get("workflow"))

    def test_the_brief_route(self):  # H4: the person corrects one field of what DCLab understood
        c = self.client
        d = c.post("/api/drafts", json={"problem": BY_TRACK["reporting"][1]}, headers=self.h).json()
        wait(lambda: ((c.get(f"/api/drafts/{d['id']}").json().get("track_brief") or {}).get("reporting")))
        out = c.post(f"/api/drafts/{d['id']}/brief", json={"track": "reporting", "field": "audience", "value": "the finance team"}, headers=self.h)
        self.assertEqual(out.status_code, 200)
        entry = out.json()["track_brief"]["reporting"]["audience"]
        self.assertEqual((entry["value"], entry["source"]), ("the finance team", "user"))
        bad = c.post(f"/api/drafts/{d['id']}/brief", json={"track": "reporting", "field": "target", "value": "x"}, headers=self.h)
        self.assertEqual(bad.status_code, 422)  # a prediction's field is not on this sheet

    def test_ml_routes_refuse_another_track_and_a_pack_means_a_prediction(self):
        c = self.client
        d = c.post("/api/drafts", json={"problem": BY_TRACK["reporting"][1]}, headers=self.h).json()
        wait(lambda: (c.get(f"/api/drafts/{d['id']}").json().get("intent") or {}).get("track") == "reporting")
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/build", headers=self.h).status_code, 409)
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/solution/proposal", json={}, headers=self.h).status_code, 409)
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/map", json={}, headers=self.h).status_code, 409)  # the review of the port: routes
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/map/edit", json={"action": "confirm"}, headers=self.h).status_code, 409)
        out = c.post(f"/api/drafts/{d['id']}/pack", json={"key": "tabular"}, headers=self.h).json()
        self.assertEqual((out["intent"]["track"], out["intent"]["source"]), ("prediction", "user"))
        self.assertTrue(out["workflow"])

    def test_the_wizard_asks_for_a_prediction(self):
        c = self.client
        self.assertEqual(c.post("/api/drafts", json={"problem": BY_TRACK["analysis"][1], "intent": "magic"}, headers=self.h).status_code, 422)
        d = c.post("/api/drafts", json={"problem": BY_TRACK["analysis"][1], "intent": "prediction"}, headers=self.h).json()
        wait(lambda: c.get(f"/api/drafts/{d['id']}").json().get("workflow"))
        self.assertEqual(c.get(f"/api/drafts/{d['id']}").json()["intent"]["source"], "user")


class ReviewOfThePortTests(unittest.TestCase):
    """The review of the H2 port to main (2026-10-28): eight findings, each with its test."""

    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.store = DraftStore(self.home)
        self.book = ledger.FileLedger(self.home / "ledger", "wtest")
        ledger.install(self.book)
        self.addCleanup(ledger.install, None)

    def test_a_model_reading_keeps_the_rules_own_answer(self):  # finding 1
        class Model:
            def complete(self, *a, **k):
                return {"content": '{"tracks": ["analysis"], "quote": "my customers", "confidence": 0.6}'}
        d = self.store.create("Help with my customers and their orders")
        HomeAgent(self.store, None, intent_client=Model()).start(d["id"])
        rec = self.book.records(point="request.intent")[0]
        answers = {a.by: a.value for a in rec.answers}
        self.assertEqual((answers.get("model_role"), answers.get("rule")), ("analysis", None))  # the rules read nothing: they abstain

    def test_an_edit_never_throws_away_a_models_reading_for_the_default(self):  # finding 2
        class Model:
            def complete(self, *a, **k):
                return {"content": '{"tracks": ["analysis"], "quote": "my customers", "confidence": 0.6}'}
        d = self.store.create("Help with my customers and their orders")
        agent = HomeAgent(self.store, None, intent_client=Model())
        agent.start(d["id"])
        self.store.update(d["id"], lambda x: x.update(problem="Help with my customers and their order lines"))
        agent.replan(d["id"])
        d = self.store.get(d["id"])
        self.assertEqual((d["intent"]["track"], d["intent"]["source"]), ("analysis", "model"))
        self.assertFalse(d.get("workflow"))
        self.assertEqual([q for q in d["questions"] if not q.get("brief")], [])  # H4: only the track's brief asks

    def test_off_a_prediction_no_question_of_the_prediction_path_is_asked(self):  # finding 3 (the critic asks through ask)
        d = self.store.create(BY_TRACK["reporting"][0])
        agent = HomeAgent(self.store, None)
        agent.start(d["id"])
        self.assertIsNone(agent.ask(d["id"], "target", "Which column holds the outcome?", source="critic"))
        from dclab_rnd.draft.coordinator import Coordinator
        from dclab_rnd.studio import ProjectStore

        out = Coordinator(self.store, ProjectStore(self.home / "projects"), agent).run(d["id"], None)
        self.assertIn("not a prediction", out["says"])
        self.assertEqual([q for q in self.store.get(d["id"])["questions"] if not q.get("brief")], [])  # H4: only the brief asks

    def test_a_draft_from_before_h2_stays_a_prediction(self):  # finding 6
        from dclab_rnd.draft.chat import DraftGuard

        d = self.store.create("Help with my customers and their orders")  # no intent key: made before H2
        agent = HomeAgent(self.store, None)
        tool = type("T", (), {"move": "set_intent", "name": "set_intent"})()
        turn = type("Turn", (), {"agent": agent, "draft_id": d["id"]})()
        refused = DraftGuard().refusal(tool, turn, {"track": "analysis", "quote": "my customers"})
        self.assertIn("prediction", refused["error"])
        self.assertEqual(json.loads(agent.context(self.store.get(d["id"])))["intent"]["track"], "prediction")

    def test_an_external_chat_cannot_choose_the_type_of_help(self):  # finding 5
        from dclab_rnd import chat_agents

        self.assertIsNotNone(chat_agents._refusal_of("POST", "/api/drafts", {"problem": "Show me sales by region", "intent": "analysis"}))
        self.assertIsNone(chat_agents._refusal_of("POST", "/api/drafts", {"problem": "Show me sales by region"}))

    def test_the_coordinator_does_not_count_the_intent_question(self):  # finding 8
        from dclab_rnd.draft.coordinator import Coordinator
        from dclab_rnd.studio import ProjectStore

        d = self.start_tie()
        agent = HomeAgent(self.store, None)
        agent.reply(d["id"], intent.LABELS["prediction"])
        draft = self.store.get(d["id"])
        c = Coordinator(self.store, ProjectStore(self.home / "projects"), agent)
        self.assertEqual(len([q for q in draft["questions"] if q["field"] == "intent"]), 1)
        from dclab_rnd.draft.chat import counted
        self.assertNotIn("intent", [q["field"] for q in counted(draft)])
        c._open_field(draft)  # runs with the intent question uncounted

    def test_a_model_reply_off_a_prediction_states_no_number_nobody_gave(self):  # HOME-003: "a churn rate of 2.5%"
        d = self.store.create("Show me revenue by region for last quarter")
        agent = HomeAgent(self.store, None)
        agent.start(d["id"])
        draft = self.store.get(d["id"])
        said = agent.checked_reply(draft, "Revenue usually varies by about 2.5% between regions.")
        self.assertNotIn("2.5", said)
        self.assertIn(intent.NEXT["analysis"], said)
        self.assertEqual(agent.checked_reply(draft, "I can describe your table once you upload it."), "I can describe your table once you upload it.")

    def start_tie(self):
        d = self.store.create(TIE)
        HomeAgent(self.store, None).start(d["id"])
        return self.store.get(d["id"])


if __name__ == "__main__":
    unittest.main()
