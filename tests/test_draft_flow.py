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
from dclab_rnd.draft.chat import HomeAgent, MAX_QUESTIONS, Turn  # noqa: E402
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

    def test_a_horizon_alone_does_not_make_a_classification_a_time_series(self):
        for sentence in ("Predict which subscribers will cancel next month so support can call them first.",
                         "Rank telecom customers by their risk of leaving next month", "Who will renew their plan next year?",
                         "Which daily active users will churn?"):
            self.assertEqual(pack.detect(sentence)["key"], "tabular", sentence)
        for sentence in ("How many units will each store sell next week?", "Forecast daily unit sales per store", "Predict tomorrow's energy load",
                         "Estimate hourly call volume", "Forecast which products run out next month"):
            self.assertEqual(pack.detect(sentence)["key"], "timeseries", sentence)
        self.assertEqual(pack.detect("Flag fraudulent payments daily")["key"], "imbalanced")  # the more specific pack still wins


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

    def test_a_message_waits_for_the_data_ready_turn_to_finish(self):
        """The answer to "which column?" must meet the question asked with the columns, not the gap before it."""
        import threading

        d = self.store.create("Predict which customers cancel")
        agent = HomeAgent(self.store, None)
        agent.start(d["id"])  # asks for the outcome before any data exists

        def with_data(x):
            x["assets"] = [{"id": "a1", "name": "c.csv", "status": "ready"}]
            x["active_asset"] = "a1"
            x["analysis"] = {"summary": {"rows": 10, "columns": 2}, "columns": [{"name": "left"}, {"name": "plan"}], "profile": {"target_candidates": ["left", "plan"]}}
        self.store.update(d["id"], with_data)
        in_gap, release = threading.Event(), threading.Event()
        original = agent.next_turn

        def slow_next_turn(draft_id, opening=False):  # the old question is withdrawn, the new one not asked yet
            in_gap.set()
            release.wait(60)
            return original(draft_id, opening)
        agent.next_turn = slow_next_turn
        ready = threading.Thread(target=agent.data_ready, args=(d["id"], {"id": "a1"}))
        ready.start()
        self.assertTrue(in_gap.wait(60))
        answer = threading.Thread(target=HomeAgent(self.store, None).reply, args=(d["id"], "left"))
        answer.start()
        answer.join(0.3)
        self.assertTrue(answer.is_alive())  # the message waits instead of landing in the notes
        release.set()
        ready.join(60)
        answer.join(60)
        self.assertFalse(ready.is_alive() or answer.is_alive())
        draft = self.store.get(d["id"])
        self.assertEqual(draft["understanding"].get("target"), "left")
        self.assertNotIn("notes", draft["understanding"])

    def test_a_sentence_that_states_the_outcome_and_the_moment_is_not_asked_them_again(self):
        agent = HomeAgent(self.store, None)
        d = self.store.create("Predict which subscribers cancel within 30 days, scored on the first day of each month")
        agent.start(d["id"])
        d = self.store.get(d["id"])
        plan = d["plan"]
        self.assertEqual((plan["target"]["status"], plan["target"]["source"], plan["target"]["quote"]),
                         ("stated", "problem", "which subscribers cancel within 30 days"))
        self.assertEqual((plan["prediction_moment"]["status"], plan["prediction_moment"]["quote"]), ("stated", "on the first day of each month"))
        self.assertEqual(plan["action"]["status"], "unknown")
        self.assertEqual([q["field"] for q in d["questions"]], ["action"])  # the moment before the cost; the stated ones are not asked
        self.assertEqual(d["understanding"]["prediction_moment"], "on the first day of each month")
        for answer in ["The team calls the top 200; a missed leaver costs a year of fees", "No data yet, plan only"]:
            agent.reply(d["id"], answer)
        d = self.store.get(d["id"])
        self.assertEqual(len(d["questions"]), 2)  # four before: no field was lost on the way
        self.assertEqual({f: d["plan"][f]["status"] for f in ("target", "prediction_moment", "action", "data_plan")},
                         {"target": "stated", "prediction_moment": "stated", "action": "stated", "data_plan": "stated"})
        self.assertEqual(d["plan"]["next"], None)
        self.assertEqual(d["messages"][-1]["kind"], "summary")

    def test_an_outcome_without_a_window_is_still_asked(self):
        d = self.store.create("Predict which customers cancel their subscription")
        HomeAgent(self.store, None).start(d["id"])
        d = self.store.get(d["id"])
        self.assertEqual(d["plan"]["target"]["status"], "unknown")
        self.assertEqual((d["questions"][0]["field"], d["plan"]["next"]), ("target", "target"))

    def test_an_answer_that_also_states_the_moment_fills_it(self):
        agent = HomeAgent(self.store, None)
        d = self.store.create("Predict which customers cancel their subscription")
        agent.start(d["id"])
        agent.reply(d["id"], "Customers who cancel within 30 days, scored every Monday morning")
        d = self.store.get(d["id"])
        self.assertEqual((d["plan"]["prediction_moment"]["status"], d["plan"]["prediction_moment"]["source"]), ("stated", "answer"))
        self.assertEqual([q["field"] for q in d["questions"]], ["target", "action"])

    def test_the_model_proposes_the_plan_and_code_checks_its_quotes(self):
        verdicts = []

        class Planner:
            def complete(self, messages, tools=None, max_tokens=1800):
                content = json.dumps({"target": {"status": "stated", "value": "cancels within 30 days", "quote": "cancel within 30 days"},
                                      "prediction_moment": {"status": "inferred", "value": "before the renewal date", "quote": "before renewal"},
                                      "action": {"status": "stated", "value": "call them", "quote": "the retention team calls them"}})  # not in the text
                return {"content": content, "tool_calls": [], "assistant_message": {"role": "assistant", "content": content}}

            def output(self, ok, reason=""):
                verdicts.append((ok, reason))
        d = self.store.create("Which members cancel within 30 days, checked before renewal?")
        HomeAgent(self.store, Planner()).start(d["id"])
        d = self.store.get(d["id"])
        self.assertEqual({f: (d["plan"][f]["status"], d["plan"][f]["source"]) for f in ("target", "prediction_moment", "action")},
                         {"target": ("stated", "model"), "prediction_moment": ("inferred", "model"), "action": ("unknown", None)})
        self.assertEqual(d["understanding"]["prediction_moment"], "before the renewal date")
        self.assertNotIn("action", d["understanding"])  # a quote that is not in the user's words is dropped
        self.assertEqual([q["field"] for q in d["questions"]], ["action"])
        self.assertEqual(verdicts, [(False, "a quote is not in the user's text")])

    def test_a_model_record_needs_the_users_words_or_the_open_question(self):
        d = self.store.create("Predict which customers cancel their subscription")
        HomeAgent(self.store, None).start(d["id"])  # the outcome question is open
        client = ScriptedClient([
            ("record", {"field": "prediction_moment", "value": "monthly"}),  # neither quoted nor asked: refused
            ("record", {"field": "prediction_moment", "value": "at renewal", "quote": "renewal time"}),  # not the user's words: refused
            ("record", {"field": "target", "value": "cancels within 30 days"}),  # answers the open question
        ])
        agent = HomeAgent(self.store, client)
        agent.reply(d["id"], "Cancels within 30 days")
        d = self.store.get(d["id"])
        self.assertNotIn("prediction_moment", d["understanding"])
        self.assertEqual((d["plan"]["target"]["status"], d["plan"]["target"]["source"]), ("stated", "answer"))

    def test_the_keyword_rules_claim_only_what_the_sentence_sets_out(self):
        from dclab_rnd.draft.plan import extract

        for sentence in ("Predict next month's revenue per store", "Forecast next week demand", "Predict the next 3 months of sales",
                         "Score every lead within 24 hours of signup", "Predict which users abandon their cart before checkout",
                         "Predict which customers will cancel before renewal", "Customers who cancel before renewal",
                         "Predict whether a purchase is made at checkout", "Predict which users run every day"):
            with self.subTest(sentence):
                self.assertEqual(extract(sentence), {})  # a window alone is not an outcome; "before renewal" here is part of it
        self.assertEqual(extract("Predict which orders arrive late within 2 days, 2 days before dispatch")["prediction_moment"],
                         "2 days before dispatch")  # the offset is kept: dropping it would move the moment later
        self.assertEqual(extract("Which subscribers cancel within 30 days, scored every Monday and the team calls them")["prediction_moment"], "every Monday")

    def test_the_guard_wants_whole_words_that_support_the_value_and_a_real_answer(self):
        d = self.store.create("Predict which customers cancel their subscription")
        HomeAgent(self.store, None).start(d["id"])
        self.store.update(d["id"], lambda x: x["questions"].clear() or x["messages"].clear())  # nothing open
        client = ScriptedClient([
            ("record", {"field": "prediction_moment", "value": "after the account is closed", "quote": "cus"}),  # a fragment of a word
            ("record", {"field": "action", "value": "after the account is closed", "quote": "their subscription"}),  # says something else
            ("ask_user", {"field": "action", "question": "What do you do with each prediction?"}),  # the outcome comes first
            ("ask_user", {"field": "target", "question": "What is predicted?"}),
        ])
        HomeAgent(self.store, client).reply(d["id"], "Hello")
        d = self.store.get(d["id"])
        self.assertEqual(d["understanding"], {})
        self.assertEqual([q["field"] for q in d["questions"]], ["target"])
        # a question the model asks and "answers" in the same reply was never answered by the user
        self.store.update(d["id"], lambda x: x["questions"].clear() or x["understanding"].update(target="cancels within 30 days"))
        agent = HomeAgent(self.store, None)
        out = agent.registry.call("ask_user", {"field": "prediction_moment", "question": "When?"}, Turn(agent, d["id"]))
        self.assertTrue(out["asked"])
        refused = agent.registry.call("record", {"field": "prediction_moment", "value": "after the account is closed"}, Turn(agent, d["id"]))
        self.assertIn("has not answered", refused["error"])

    def test_a_new_sentence_keeps_the_answers_and_asks_what_became_open(self):
        agent = HomeAgent(self.store, None)
        d = self.store.create("Predict which subscribers cancel within 30 days, scored every Monday")
        agent.start(d["id"])
        for answer in ["We call the top 200; a missed leaver costs a year of fees", "No data yet, plan only"]:
            agent.reply(d["id"], answer)
        self.assertEqual(self.store.get(d["id"])["messages"][-1]["kind"], "summary")
        self.store.update(d["id"], lambda x: x.update(problem="Predict which subscribers cancel within 30 days"))
        agent.replan(d["id"])
        d = self.store.get(d["id"])
        self.assertEqual(d["understanding"]["action"], "We call the top 200; a missed leaver costs a year of fees")  # answers stay
        self.assertEqual((d["plan"]["prediction_moment"]["status"], d["questions"][-1]["field"]), ("unknown", "prediction_moment"))

    def test_the_column_question_past_the_limit_ends_in_a_summary(self):
        agent = HomeAgent(self.store, None)
        d = self.store.create("Predict which customers cancel next month")
        agent.start(d["id"])
        for answer in ["At the monthly snapshot", "We call them", "Upload a sample"]:
            agent.reply(d["id"], answer)
        self.store.update(d["id"], lambda x: x["questions"].append({"id": "q9", "field": "notes", "text": "x", "options": [], "answered": "y"}))

        def with_data(x):
            x["assets"] = [{"id": "a1", "name": "c.csv", "status": "ready"}]
            x["active_asset"] = "a1"
            x["analysis"] = {"summary": {"rows": 10, "columns": 2}, "columns": [{"name": "left"}, {"name": "plan"}], "profile": {"target_candidates": ["left", "plan"]}}
        self.store.update(d["id"], with_data)
        agent.data_ready(d["id"], {"id": "a1", "name": "c.csv"})
        self.assertEqual(self.store.get(d["id"])["messages"][-1]["kind"], "summary")  # four questions asked: no fifth, no silence

    def test_a_failing_or_unreadable_planner_leaves_the_keyword_rules(self):
        class Broken:
            def complete(self, *a, **k):
                raise RuntimeError("BudgetExceeded: the cap is reached")

        class Chatty:
            def complete(self, *a, **k):
                return {"content": "Sure! The outcome is churn.", "tool_calls": [], "assistant_message": {"role": "assistant", "content": "x"}}
        for client in (Broken(), Chatty()):
            d = self.store.create("Predict which subscribers cancel within 30 days, scored every Monday")
            HomeAgent(self.store, client).start(d["id"])
            d = self.store.get(d["id"])
            self.assertEqual({f: d["plan"][f]["source"] for f in ("target", "prediction_moment")}, {"target": "problem", "prediction_moment": "problem"})

    def test_a_source_that_knows_its_outcome_column_is_offered_first(self):
        d = self.store.create("Forecast daily unit sales per store")

        def with_data(x):
            x["assets"] = [{"id": "a1", "name": "Synthetic · Daily store demand (synthetic)", "status": "ready", "synthetic": True, "suggestion": {"target": "units_sold"}}]
            x["active_asset"] = "a1"
            x["analysis"] = {"summary": {"rows": 10, "columns": 4}, "columns": [{"name": n} for n in ("unit_price", "store", "promotion", "units_sold")],
                             "profile": {"target_candidates": ["unit_price", "store", "promotion", "units_sold"]}}
        self.store.update(d["id"], with_data)
        HomeAgent(self.store, None).next_turn(d["id"])
        question = self.store.get(d["id"])["questions"][-1]
        self.assertEqual((question["field"], question["options"][0]), ("target", "units_sold"))
        self.assertEqual((len(question["options"]), question["options"][-1]), (4, "Something else"))

    def test_columns_the_problem_sentence_names_are_offered_first(self):
        from dclab_rnd.draft.chat import rank_targets

        col = lambda name, unique=3, **kw: {"name": name, "kind": "categorical", "unique": unique, "id_like": False, "target_name_like": False, "low_cardinality": unique <= 20, **kw}  # noqa: E731
        profile = {"target_candidates": ["plan", "monthly_fee", "paperless"],
                   "columns": [col("customer", 400, id_like=True), col("tenure_months", 60), col("plan"), col("monthly_fee", 7), col("paperless", 2), col("cancelled", 2),
                               col("cancel_reason_text", 300, kind="text")]}
        ranked = rank_targets(profile, "Predict which subscribers will cancel next month so support can call them first.")
        self.assertEqual(ranked[0], "cancelled")  # "cancel" names it; "month" and "customers" name the setting, not the outcome
        self.assertEqual(ranked[1:], ["plan", "monthly_fee", "paperless"])
        self.assertNotIn("customer", ranked)  # identifiers and free text are never outcomes
        self.assertNotIn("cancel_reason_text", ranked)
        fraud = {"target_candidates": ["amount"], "columns": [col("paymentAmount", 900), col("is_fraud", 2, target_name_like=True), col("amount", 800)]}
        self.assertEqual(rank_targets(fraud, "Flag fraudulent card payments before they are approved")[0], "is_fraud")
        self.assertEqual(rank_targets({"target_candidates": ["a", "b"], "columns": []}, ""), ["a", "b"])

    def test_the_model_reads_column_summaries_and_findings_but_never_values(self):
        d = self.store.create("Predict which customers cancel")
        agent = HomeAgent(self.store, None)
        self.assertIn("error", agent.run_tool(d["id"], "get_profile", {}))  # nothing prepared yet

        def with_data(x):
            x["analysis"] = {"summary": {"rows": 500, "columns": 45}, "correlations": [{"a": "c1", "b": "c2", "rho": 0.9}],
                             "columns": [{"name": f"c{i}", "kind": "categorical", "missing_rate": 0.0, "unique": 3, "top": [["SECRET-VALUE", 9]]} for i in range(45)],
                             "highlights": [{"severity": "info", "title": "Duplicate rows", "text": "12 rows (2.4%) repeat another row exactly.", "columns": []}],
                             "profile": {"target_candidates": ["c44"], "time_candidates": [], "id_candidates": ["c0"], "text_candidates": []}}
        self.store.update(d["id"], with_data)
        page = agent.run_tool(d["id"], "get_profile", {"offset": 40})
        self.assertEqual((page["total_columns"], [c["name"] for c in page["columns"]], page["target_candidates"]), (45, ["c40", "c41", "c42", "c43", "c44"], ["c44"]))
        self.assertEqual(set(page["columns"][0]), {"name", "kind", "missing_rate", "unique"})
        named = agent.run_tool(d["id"], "get_profile", {"columns": ["c7", "nope"], "offset": "x"})
        self.assertEqual([c["name"] for c in named["columns"]], ["c7"])
        findings = agent.run_tool(d["id"], "get_analysis", {})
        self.assertEqual((findings["summary"]["rows"], findings["highlights"][0]["title"], findings["correlations"][0]["rho"]), (500, "Duplicate rows", 0.9))
        everything = json.dumps([page, named, findings, agent.context(self.store.get(d["id"]))])
        self.assertNotIn("SECRET-VALUE", everything)  # top values are data: they stay on this machine
        self.assertLessEqual(len(json.loads(agent.context(self.store.get(d["id"])))["data"]["columns"]), 40)

    def test_the_home_agent_runs_on_the_runtime_with_a_replayable_trace(self):
        from dclab_rnd.agents import FileTraces, replay

        traces = FileTraces(Path(tempfile.mkdtemp()) / "agent_steps.jsonl")
        d = self.store.create("Predict which customers cancel")
        HomeAgent(self.store, None).start(d["id"])
        client = ScriptedClient([
            ("record", {"field": "outcome", "value": "cancels"}),  # not a field: the schema refuses it and the model reads why
            ("record", {"field": "target", "value": "cancels within 30 days", "extra": 1}),  # an unknown key is refused too
            ("record", {"field": "target", "value": "cancels within 30 days"}),
            ("ask_user", {"question": "When is the prediction made?"}),  # no field: allowed, the answer goes to the notes
        ])
        agent = HomeAgent(self.store, client, traces=traces)
        agent.reply(d["id"], "Whether they cancel")
        rows = traces.steps(d["id"])
        self.assertEqual([(r["tool"], r["verdict"]) for r in rows], [("record", "error"), ("record", "error"), ("record", "ok"), ("ask_user", "ok")])
        self.assertEqual([r["state"] for r in rows], ["-----", "-----", "-----", "x----"])  # what the agent saw before each move
        self.assertEqual(self.store.get(d["id"])["understanding"]["target"], "cancels within 30 days")
        self.assertEqual(self.store.get(d["id"])["messages"][-1]["kind"], "question")
        self.assertEqual(client.i, 4)  # the question ended the turn: no fifth request
        self.assertTrue(replay(rows, agent.policy(), agent.registry)["same"])

    def test_a_refused_question_or_proposal_is_corrected_and_reported(self):
        verdicts = []
        d = self.store.create("Predict which customers cancel")
        HomeAgent(self.store, None).start(d["id"])
        client = ScriptedClient([
            ("ask_user", {"question": "When is the prediction made?", "field": "prediction_time"}),  # not a field: refused, the turn goes on
            ("propose_workflow", {"title": "x"}),  # no nodes: refused by the schema, so the gateway hears it here
            ("ask_user", {"question": "When is the prediction made?", "field": "prediction_moment"}),
        ])
        client.output = lambda ok, reason="": verdicts.append(ok)
        HomeAgent(self.store, client).reply(d["id"], "Whether they cancel")
        self.assertEqual(client.i, 3)
        self.assertEqual(self.store.get(d["id"])["messages"][-1]["text"], "When is the prediction made?")
        self.assertEqual(verdicts, [False, False, True])

    def test_the_model_can_start_a_simulation_and_the_turn_ends_there(self):
        asked = []
        d = self.store.create("Predict which customers cancel")
        HomeAgent(self.store, None).start(d["id"])
        client = ScriptedClient([("simulate_data", {"description": "5,000 subscribers with tenure and plan; 20% cancel", "rows": 999999}),
                                 ("ask_user", {"field": "action", "question": "never reached"})])
        HomeAgent(self.store, client, on_request=lambda did, what, args: asked.append((what, args))).reply(d["id"], "Please simulate the data")
        self.assertEqual([(w, a["rows"]) for w, a in asked], [("simulate", 200000)])  # capped
        self.assertIn("20% cancel", asked[0][1]["prompt"])
        self.assertEqual(client.i, 1)  # the second scripted call was never requested
        self.assertIn("labelled synthetic", self.store.get(d["id"])["messages"][-1]["text"])

    def test_streaming_client_sends_token_chunks_then_the_message_replaces_them(self):
        class Streaming:
            def stream(self, messages, tools=None, max_tokens=1800, on_text=None):
                text = "Thanks, that is clear. " * 6
                for i in range(0, len(text), 5):
                    on_text(text[i:i + 5])
                return {"content": text, "tool_calls": [], "assistant_message": {"role": "assistant", "content": text}}
        d = self.store.create("Predict which customers cancel")
        HomeAgent(self.store, None).start(d["id"])
        HomeAgent(self.store, Streaming()).reply(d["id"], "Customers who cancel within 30 days")
        events = self.store.events(d["id"])
        tokens = [e["data"] for e in events if e["kind"] == "token"]
        last = next(m for m in self.store.get(d["id"])["messages"] if tokens and m["id"] == tokens[0]["id"])  # the streamed reply
        self.assertTrue(tokens and all(t["id"] == last["id"] for t in tokens))
        self.assertLess(len(tokens), 12)  # chunks, not one event per delta
        self.assertEqual("".join(t["delta"] for t in tokens).strip(), last["text"])
        order = [e["kind"] for e in events if e["kind"] in ("token", "chat")]
        self.assertEqual(order[-1], "chat")  # the message follows its tokens
        self.assertFalse([t for t in tokens if t.get("drop")])

    def test_a_streamed_step_that_ends_in_tool_calls_or_fails_drops_its_bubble(self):
        class ToolsAfterText:
            def stream(self, messages, tools=None, max_tokens=1800, on_text=None):
                on_text("Let me note that down. ")
                call = {"id": "c1", "name": "record", "arguments": {"field": "target", "value": "churn"}}
                return {"content": "Let me note that down. ", "tool_calls": [call], "assistant_message": {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "record", "arguments": "{}"}}]}}
        d = self.store.create("Predict which customers cancel")
        HomeAgent(self.store, None).start(d["id"])
        HomeAgent(self.store, ToolsAfterText()).complete(d["id"], [])
        kinds = [(e["data"].get("id") and e["data"].get("drop"), e["kind"]) for e in self.store.events(d["id"]) if e["kind"] == "token"]
        self.assertEqual([k[0] for k in kinds][-1], True)

        class Broken:
            def stream(self, messages, tools=None, max_tokens=1800, on_text=None):
                on_text("Partial ")
                raise RuntimeError("APIError: the model request failed")
        with self.assertRaises(RuntimeError):
            HomeAgent(self.store, Broken()).complete(d["id"], [])
        self.assertTrue([e for e in self.store.events(d["id"]) if e["kind"] == "token"][-1]["data"].get("drop"))

    def test_model_failure_falls_back_to_the_script(self):
        class Broken:
            def complete(self, *a, **k):
                raise RuntimeError("RateLimitError: the model request failed")
        d = self.store.create("Predict which customers cancel")
        HomeAgent(self.store, None).start(d["id"])
        HomeAgent(self.store, Broken()).reply(d["id"], "Customers who cancel within 30 days")
        d = self.store.get(d["id"])
        self.assertEqual(d["understanding"]["target"], "Customers who cancel within 30 days")
        HomeAgent(self.store, Broken()).reply(d["id"], "At the weekly snapshot, before the retention call.")
        notes = [e["data"].get("note") for e in self.store.events(d["id"]) if e["kind"] == "status" and e["data"].get("note")]
        self.assertEqual(notes, ["The model was unavailable (the model request failed); DCLab continues with its standard questions."])  # said once, not per turn

    def test_a_model_that_only_talks_does_not_stall_the_conversation(self):
        class Talker:  # never calls a tool, as a small local model did in the live check
            def complete(self, messages, tools=None, max_tokens=1800):
                return {"content": "Great, that makes sense.", "tool_calls": [], "assistant_message": {"role": "assistant", "content": "Great, that makes sense."}}
        d = self.store.create("Predict which customers pause or cancel")
        HomeAgent(self.store, None).start(d["id"])
        agent = HomeAgent(self.store, Talker())
        for answer in ("Lost means paused or cancelled within 30 days.", "Every Monday morning.", "The team calls the riskiest 200.", "No data yet, plan only"):
            agent.reply(d["id"], answer)
        draft = self.store.get(d["id"])
        self.assertEqual({k: draft["understanding"].get(k) for k in ("target", "prediction_moment", "action")},
                         {"target": "Lost means paused or cancelled within 30 days.", "prediction_moment": "Every Monday morning.", "action": "The team calls the riskiest 200."})
        self.assertEqual(draft["messages"][-1]["kind"], "summary")  # the user always has a next step
        self.assertEqual(draft["agent"]["mode"], "model")

    def test_the_summary_is_said_once_and_quotes_the_answers(self):
        d = self.store.create("Predict which customers cancel")
        self.store.update(d["id"], lambda x: x["understanding"].update(target="A customer counts as lost if they pause or cancel within 30 days.",
                                                                       prediction_moment="Every Monday morning", action="The team calls the riskiest customers"))
        agent = HomeAgent(self.store, None)
        agent.summarize(d["id"])
        agent.summarize(d["id"])  # the data-ready turn and the reply that started a simulation both end here
        summaries = [m for m in self.store.get(d["id"])["messages"] if m.get("kind") == "summary"]
        self.assertEqual(len(summaries), 1)
        self.assertIn("Outcome: A customer counts as lost if they pause or cancel within 30 days. Prediction moment: Every Monday morning. "
                      "What happens with each prediction: The team calls the riskiest customers. Pack: Tabular.", summaries[0]["text"])
        self.store.update(d["id"], lambda x: x["assets"].append({"id": "a1", "status": "ready"}))
        agent.summarize(d["id"])  # something changed (the data is ready), so it is worth saying again
        self.assertEqual(len([m for m in self.store.get(d["id"])["messages"] if m.get("kind") == "summary"]), 2)


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
        self.assertIn("StratifiedKFold(3,", json.dumps(record))  # no choice made: three folds

    def test_the_wizard_folds_reach_the_cross_validation_and_the_solution_decides_the_split(self):
        import numpy as np
        import pandas as pd
        from dclab_rnd.studio import ProjectStore, data, engine

        self.assertEqual([engine.cv_folds({"settings": s}) for s in ({}, {"folds": 5}, {"folds": 99}, {"folds": 1}, {"folds": "x"})], [3, 5, 10, 2, 3])
        self.assertEqual([engine.split_for(s) for s in ({}, {"group_column": "g"}, {"time_column": "t", "group_column": "g"})], ["stratified", "group", "time"])
        rng = np.random.default_rng(5)
        n = 400
        frame = pd.DataFrame({"tenure": rng.integers(1, 72, n), "charges": rng.normal(60, 20, n).round(2), "plan": rng.choice(["a", "b", "c"], n)})
        frame["churned"] = (rng.random(n) < 0.15 + 0.004 * frame["tenure"]).astype(int)
        store = ProjectStore(Path(tempfile.mkdtemp()))
        pid = store.create("folds", "general", "who leaves")["id"]
        frame.to_parquet(store.data_dir(pid) / "t.parquet", index=False)
        data.attach_data(store, pid, "t.parquet")
        project = store.get(pid)
        project["solution"] = {"target": "churned", "task": "binary", "positive_label": "1", "prediction_moment": "At the monthly snapshot of each customer.",
                               "forbidden": [], "identifiers": [], "time_column": None, "group_column": None, "text_columns": [], "metric": "roc_auc", "notes": ""}
        project["settings"].update(quick=True, folds=5)
        store.save(project)
        record = engine.execute(store, pid, "data", "human")
        text = json.dumps(record)
        self.assertIn("StratifiedKFold(5,", text)
        self.assertNotIn("StratifiedKFold(3,", text)


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
        self.assertEqual(self.draft(d["id"])["trace"], [])  # the Home agent's model steps; the script makes none
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
        # a missing or malformed body is the caller's mistake (422), never a server error; the proposal's body is optional
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/messages", content=b"not json", headers=self.h).status_code, 422)
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/data/sample", content=b"[1]", headers=self.h).status_code, 422)
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/solution/proposal", headers=self.h).json()["target"], target)  # the chat's target
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/solution/proposal", json={"target": "nope"}, headers=self.h).status_code, 422)
        # wizard: the solution draft from the column audit, then split and budget
        y = ready["suggestion"]["target"]
        proposal = c.post(f"/api/drafts/{d['id']}/solution/proposal", json={"target": y}, headers=self.h)
        self.assertEqual(proposal.status_code, 200, proposal.text)
        prop = proposal.json()
        self.assertIn("duration", {f["column"] for f in prop["forbidden"]})  # the R&D's own solution for bank_marketing
        review = {i["column"]: i for i in prop["review"]["items"]}  # the leakage reviewer (A3.2), without a model here
        self.assertEqual((prop["review"]["mode"], review["duration"]["apply"]), ("rules", True))
        self.assertTrue(all(i["records"] and i["reason"] for i in prop["review"]["items"]))
        edited = c.post(f"/api/drafts/{d['id']}/solution/proposal", json={"target": y, "prediction_moment": "Before the call; the pdays counter is not available then"},
                        headers=self.h).json()  # the moment as the sheet shows it, edited: the review reads it
        self.assertEqual(edited["prediction_moment"], "Before the call; the pdays counter is not available then")
        self.assertIn(("pdays", "moment", False), [(i["column"], i["source"], i["apply"]) for i in edited["review"]["items"]])
        self.assertEqual(prop["prediction_moment"], ready["suggestion"]["prediction_moment"])  # a real moment: the sample's own
        self.assertTrue(prop["prediction_moment_hint"].startswith("Describe the moment"))  # guidance, never a value
        solution = {"target": y, "task": prop["task"], "positive_label": str(prop["positive_label"]) if prop["task"] == "binary" else None,
                    "prediction_moment": "Immediately before the marketing call is placed.",
                    "forbidden": [{"column": "duration", "reason": "known only after the call"}], "identifiers": [], "time_column": None,
                    "group_column": None, "text_columns": [], "metric": prop["metric"], "notes": ""}
        self.assertEqual(c.put(f"/api/drafts/{d['id']}/solution", json={**solution, "target": "nope"}, headers=self.h).status_code, 422)
        self.assertEqual(c.put(f"/api/drafts/{d['id']}/solution", json=solution, headers=self.h).status_code, 200)
        settings = c.put(f"/api/drafts/{d['id']}/settings", json={"split": "time", "folds": 5, "quick": True, "max_rows": 999999, "budget": {"calls": 500}}, headers=self.h).json()["settings"]
        self.assertEqual((settings["max_rows"], settings["budget"]["max_steps"]), (200000, 80))  # clamped
        self.assertEqual(settings["split"], "stratified")  # the solution names no time column, so "time" cannot be what runs
        project = c.post(f"/api/drafts/{d['id']}/build", json={}, headers=self.h)
        self.assertEqual(project.status_code, 201, project.text)
        p = project.json()
        self.assertEqual(p["data"]["rows"], ready["rows"])
        self.assertFalse(p["data"]["synthetic"])
        self.assertEqual(p["draft"]["id"], d["id"])
        self.assertEqual(p["solution"]["forbidden"][0]["column"], "duration")
        self.assertEqual([(n["kind"], n["who"], n["move"]) for n in p["memory"]], [("decision", "human", "set_solution")])  # A5.2: the project remembers it
        self.assertEqual(p["budget"]["max_steps"], 80)
        self.assertEqual((p["settings"]["folds"], p["draft"]["settings"]["split"]), (5, "stratified"))
        again = c.post(f"/api/projects/{p['id']}/solution/proposal", json={"target": y}, headers=self.h)  # the Solution page's "audit again"
        self.assertEqual(again.status_code, 200, again.text)
        patched = c.patch(f"/api/projects/{p['id']}", json={"settings": {"folds": 40}}, headers=self.h).json()
        self.assertEqual(patched["settings"]["folds"], 10)  # capped
        moves = c.get(f"/api/projects/{p['id']}").json()["transitions"]
        self.assertEqual([(m["move"], m["status"]) for m in moves], [("set_solution", "allowed")])
        self.assertEqual(self.draft(d["id"])["status"], "built")
        ws = c.get("/api/workspace").json()
        self.assertEqual(ws["stats"]["projects"], 1)

    def test_editing_the_sentence_reads_the_plan_again(self):
        c = self.client
        d = c.post("/api/drafts", json={"problem": "Predict which subscribers cancel within 30 days, scored every Monday"}, headers=self.h).json()
        wait(lambda: self.draft(d["id"])["questions"])
        self.assertEqual(self.draft(d["id"])["plan"]["prediction_moment"]["quote"], "every Monday")
        edited = c.patch(f"/api/drafts/{d['id']}", json={"problem": "Predict which subscribers cancel within 30 days"}, headers=self.h).json()
        self.assertEqual((edited["plan"]["prediction_moment"]["status"], edited["plan"]["target"]["status"]), ("unknown", "stated"))
        self.assertNotIn("prediction_moment", edited["understanding"])  # read from the old sentence, gone with it

    def test_a_project_built_from_an_upload_can_be_audited_again(self):
        """Its suggestion holds only what the chat established (no forbidden list), which once crashed the proposal route."""
        c = self.client
        d = c.post("/api/drafts", json={"problem": "Predict which customers cancel next month"}, headers=self.h).json()
        rows = "tenure,plan,charges,left\n" + "\n".join(f"{i % 60},{'ab'[i % 2]},{20 + i % 70},{int(i % 5 == 0)}" for i in range(300))
        self.assertEqual(c.put(f"/api/drafts/{d['id']}/data?filename=c.csv", content=rows.encode(), headers=self.h).status_code, 200)
        wait(lambda: next((a for a in self.draft(d["id"])["assets"] if a["status"] == "ready"), None))
        # the sentence gives the outcome in words ("cancel next month"): its column is still asked, before or after the
        # moment depending on whether the table was ready when the opening turn ran; answer whichever is open
        self.assertEqual(self.draft(d["id"])["plan"]["target"]["status"], "stated")
        answers = {"target": "left", "prediction_moment": "At the monthly snapshot"}
        for _ in range(2):
            q = wait(lambda: next((q for q in self.draft(d["id"])["questions"] if not q.get("answered") and q["field"] in answers), None))
            if q["field"] == "target":
                self.assertIn("left", q["options"])
            text = answers.pop(q["field"])
            wait(lambda: c.post(f"/api/drafts/{d['id']}/messages", json={"text": text}, headers=self.h).status_code == 202)
            wait(lambda: any(x["id"] == q["id"] and x.get("answered") for x in self.draft(d["id"])["questions"]))
        self.assertEqual(self.draft(d["id"])["understanding"].get("target"), "left")
        p = c.post(f"/api/drafts/{d['id']}/build", json={}, headers=self.h).json()
        self.assertEqual(p["suggestion"], {"target": "left", "prediction_moment": "At the monthly snapshot"})
        proposal = c.post(f"/api/projects/{p['id']}/solution/proposal", json={"target": "left"}, headers=self.h)
        self.assertEqual(proposal.status_code, 200, proposal.text)
        self.assertEqual(proposal.json()["task"], "binary")

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

    def test_synthetic_templates_are_listed_chosen_and_explained(self):
        c = self.client
        listing = c.get("/api/synthetic/templates").json()
        self.assertFalse(listing["model"])  # no key in this test environment
        keys = [t["key"] for t in listing["templates"]]
        self.assertIn("fraud", keys)
        self.assertTrue(all(t["name"] and "(synthetic)" not in t["name"] and t["columns"] > 3 for t in listing["templates"]))
        d = c.post("/api/drafts", json={"problem": "Predict which members of a fitness app will cancel next month"}, headers=self.h).json()
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/data/synthetic", json={"template": "nope"}, headers=self.h).status_code, 422)
        # no template named: keywords pick one, and the user is told it does not follow the description
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/data/synthetic", json={"prompt": "gym members and weekly workouts", "rows": 300}, headers=self.h).status_code, 202)
        first = wait(lambda: next((a for a in self.draft(d["id"])["assets"] if a["status"] in ("ready", "failed")), None))
        self.assertEqual((first["status"], first["template"], first["spec_source"]), ("ready", "churn", "template"))
        self.assertEqual(first["suggestion"], {"target": "churned"})  # the generator knows its outcome column
        self.assertIn("No model is configured", first["template_note"])
        fresh = c.post(f"/api/drafts/{d['id']}/solution/proposal", headers=self.h).json()
        self.assertIsNone(fresh["prediction_moment"])  # nobody stated the moment yet, so there is no value to prefill
        self.assertIn("does not follow", first["template_note"])
        # a template the user picked
        wait(lambda: c.post(f"/api/drafts/{d['id']}/data/synthetic", json={"template": "fraud", "rows": 300}, headers=self.h).status_code == 202)
        second = wait(lambda: next((a for a in self.draft(d["id"])["assets"] if a.get("template") == "fraud" and a["status"] in ("ready", "failed")), None))
        self.assertEqual((second["status"], second["synthetic"], second["rows"]), ("ready", True, 300))
        self.assertIn("as you chose", second["template_note"])
        notes = [e["data"]["note"] for e in self.client.app.state.drafts.events(d["id"]) if e["kind"] == "status" and e["data"].get("note")]
        self.assertEqual(sum("template" in n for n in notes), 2)

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
