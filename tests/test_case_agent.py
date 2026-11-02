"""Packages 37.2 to 37.4 (docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): the context builder and the lead agent. Claude leads the
conversation around the case file; code guards the edges (sources of facts, readiness, options, numbers). Scripted
models only: the live measurement is the conversation exam's arm "case"."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import case as C  # noqa: E402
from dclab_rnd.agents import case_agent as A  # noqa: E402
from dclab_rnd.agents import context as ctx  # noqa: E402
from dclab_rnd.agents import default_registry  # noqa: E402


class Script:
    """A model client replying in order: a list of tool calls (name, arguments) and the words beside them."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def complete(self, messages, tools=None, *args, **kwargs):
        self.seen.append(json.loads(json.dumps(messages)))
        text, calls = self.replies.pop(0) if self.replies else ("", [])
        calls = [{"id": f"c{len(self.seen)}-{i}", "name": f"case_{n}", "arguments": a} for i, (n, a) in enumerate(calls)]
        msg = {"role": "assistant", "content": text, **({"tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]} if calls else {})}
        return {"content": text, "tool_calls": calls, "assistant_message": msg, "usage": {"input_tokens": 1, "output_tokens": 1}}


VIEW = [{"name": "customer_id", "kind": "text", "noticed": ["identifier"]}, {"name": "plan", "kind": "category", "noticed": []},
        {"name": "churned_next_month", "kind": "numeric", "noticed": ["outcome-like name"]}]


def state(table=True):
    return A.CaseState(case=C.new("case-t"), table=A.table_of("customers.csv", VIEW, 1200) if table else None)


FULL = {"gets": "a list your team can act on", "tradeoff": "needs past outcomes", "effort": "a day"}  # 37.5: written in full
OPTIONS = [{"title": "Rank customers by risk", "what": "a weekly ranked list", "track": "prediction", "tool": "ml_project", "columns": ["churned_next_month"], **FULL},
           {"title": "Compare churn by plan", "what": "churn per plan", "track": "analysis", "tool": "query", "columns": ["plan"], **FULL}]


class ContextTests(unittest.TestCase):
    def test_a_long_conversation_fits_the_budget_and_keeps_every_fact(self):
        case = C.new("c", ["we lose customers every month and it hurts", "my retention team calls people"])
        case = C.apply(case, "goal", C.CaseFact(value="keep customers", status="said", quote="we lose customers"), by="agent", why="w")
        case = C.apply(case, "user", C.CaseFact(value="the retention team", status="said", quote="retention team calls"), by="agent", why="w")
        turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "words " * 400} for i in range(60)]
        built = ctx.build(case, turns, A.table_of("customers.csv", VIEW, 1200), budget=12_000)
        self.assertLessEqual(built["chars"], 12_000)
        self.assertGreater(built["dropped"], 0)
        for fact in ("keep customers", "the retention team", "churned_next_month"):
            self.assertIn(fact, built["state"])  # older turns go; the case file never does
        self.assertEqual(built["turns"][-1]["content"], turns[-1]["content"])

    def test_the_latest_words_always_fit_and_the_state_rides_under_them(self):
        built = ctx.build(C.new("c", ["hi"]), [{"role": "user", "content": "x" * 50_000}], budget=1000)
        self.assertEqual(len(built["turns"]), 1)
        msgs = ctx.messages("SYSTEM", built)
        self.assertEqual(msgs[0], {"role": "system", "content": "SYSTEM"})  # the same instructions every turn (cached)
        self.assertIn("The case file", msgs[-1]["content"])


class TurnTests(unittest.TestCase):
    def test_it_reads_the_table_notes_what_was_said_and_asks_one_question(self):
        s = state()
        client = Script(("", [("read_table", {})]),
                        ("You are losing customers.", [("note", {"field": "goal", "value": "keep customers", "status": "said", "quote": "losing customers"}),
                                                        ("ask", {"question": "Who will act on the list of customers at risk?"})]))
        out = A.turn(s, client, "we are losing customers and I have their table")
        self.assertIn("Who will act on the list", out["text"])
        self.assertTrue(out["asking"])
        self.assertEqual(s.case.goal.quote, "losing customers")
        self.assertEqual(s.case.data.status, "computed")  # code wrote the data fact when the table was read
        self.assertEqual(s.case.asked, 1)

    def test_a_refused_fact_comes_back_as_the_tools_result_and_the_agent_goes_on(self):
        s = state(table=False)
        client = Script(("", [("note", {"field": "decision", "value": "call them", "status": "said", "quote": "call the riskiest first"})]),
                        ("", [("ask", {"question": "What will you do with the result?"})]))
        A.turn(s, client, "we are losing customers")
        tool_turns = [m for m in client.seen[1] if m.get("role") == "tool"]
        self.assertIn("not the person's words", tool_turns[0]["content"])
        self.assertIsNone(s.case.decision)

    def test_options_wait_for_readiness_unless_the_person_asks_to_see_them(self):
        s = state()
        client = Script(("", [("propose", {"options": OPTIONS})]), ("", [("ask", {"question": "What is your goal?"})]))
        out = A.turn(s, client, "can you help with our customers")
        self.assertEqual(out["options"], [])
        self.assertIn("not yet", json.dumps(client.seen[1]))
        client = Script(("Here are two ways.", [("propose", {"options": OPTIONS})]))
        out = A.turn(s, client, "just show me what you can do")
        self.assertEqual([o.title for o in out["options"]], ["Rank customers by risk", "Compare churn by plan"])

    def test_the_persons_pick_is_the_route_and_start_waits_for_a_choice_or_a_ready_case(self):
        s = state()
        A.turn(s, Script(("", [("start", {"track": "prediction", "tool": "ml_project"})]), ("", [("ask", {"question": "What do you need?"})])), "hello")
        self.assertIsNone(s.case.route)
        A.turn(s, Script(("Two ways.", [("propose", {"options": OPTIONS})])), "show me options")
        out = A.pick(s, Script(("I will rank them.", [])), "Compare churn by plan")
        self.assertEqual((s.case.route.track, s.case.route.by, s.case.chosen), ("analysis", "person", "Compare churn by plan"))
        self.assertEqual(out["text"], "I will rank them.")

    def test_an_advice_answer_starts_without_a_ready_case_and_a_start_is_followed_by_words(self):  # CEX-005: d-36 looped, owner-telco got no words
        s = state(table=False)
        client = Script(("Report PR-AUC first: with rare positives it shows how useful the flags are.", [("start", {"track": "advice", "tool": "cited_answer"})]),
                        ("Say it in one line to your manager.", []))
        out = A.turn(s, client, "should I report PR-AUC or ROC-AUC to my manager")
        self.assertEqual(s.case.route.track, "advice")
        self.assertIn("one line", out["text"])  # the run went on after the start, so the person reads what happens next
        self.assertEqual(len(client.seen), 2)

    def test_a_number_nobody_gave_is_dropped_from_what_the_person_reads(self):
        s = state()
        out = A.turn(s, Script(("", [("read_table", {})]), ("Your table has 1200 rows. About 14% of customers leave.", [("ask", {"question": "Is that right?"})])), "hi")
        self.assertIn("1200 rows", out["text"])  # read by code
        self.assertNotIn("14%", out["text"])

    def test_a_dropped_sentence_goes_back_to_the_agent_to_write_again_never_a_silent_turn(self):  # CEX-006 d-36
        s = state(table=False)
        client = Script(("Only about 7 in 50 customers leave, so lead with PR-AUC.", []), ("Few customers leave, so lead with PR-AUC.", []))
        out = A.turn(s, client, "churn with 3% positives: which metric for my manager")
        self.assertEqual(out["text"], "Few customers leave, so lead with PR-AUC.")
        self.assertIn("not shown to the person", json.dumps(client.seen[1]))
        out = A.turn(state(table=False), Script(("About 3 in 100 customers leave.", []), ("About 3% of customers leave.", [])), "we have 3% positives")
        self.assertEqual(out["text"], "About 3% of customers leave.")  # rewritten in the person's own number (the exam reads 100 as new)
        out = A.turn(state(table=False), Script(("About 14% leave.", []), ("Roughly 15% leave.", [])), "help with churn")
        self.assertIn("only numbers that come from you", out["text"])  # a failed rewrite: code's line, never silence (the review)

    def test_the_rewrite_runs_no_tool_and_a_question_counts_once(self):  # the review: the rewrite asked again
        s = state(table=False)
        client = Script(("", [("ask", {"question": "Do about 14% of customers leave each month?"})]),
                        ("", [("ask", {"question": "Do many customers leave each month?"})]))
        out = A.turn(s, client, "help with churn")
        self.assertEqual(s.case.asked, 1)
        self.assertNotIn("14%", out["text"])
        self.assertTrue(out["text"])

    def test_the_route_the_person_picked_stays_theirs(self):  # the review: a start or new options after the pick flipped it
        s = state()
        A.turn(s, Script(("", [("propose", {"options": OPTIONS})]), ("Two ways.", [])), "show me options")
        A.pick(s, Script(("Starting.", [("start", {"track": "prediction", "tool": "ml_project"})]), ("Done.", [])), "Compare churn by plan")
        self.assertEqual((s.case.route.track, s.case.route.by), ("analysis", "person"))
        A.turn(s, Script(("", [("propose", {"options": OPTIONS})]), ("ok", [])), "can I see the base rate first?")
        self.assertEqual(s.case.route.track, "analysis")
        A.turn(s, Script(("", [("start", {"track": "prediction", "tool": "ml_project", "quote": "a prediction instead"})]), ("Switched.", [])),
               "actually I want a prediction instead")
        self.assertEqual((s.case.route.track, s.case.route.by), ("prediction", "person"))  # the person asked, in their words

    def test_answers_that_mention_passing_or_not_knowing_are_not_declines(self):  # the review
        for words in ("we pass the list to the sales team every Monday", "I don't know who will leave, that is the problem", "customers skip payments"):
            self.assertFalse(A.declines(words), words)
        for words in ("not sure", "pass", "no idea really"):
            self.assertTrue(A.declines(words), words)

    def test_a_question_passed_over_counts_toward_the_ceiling(self):
        s = state(table=False)
        A.turn(s, Script(("", [("ask", {"question": "Who uses the result?"})])), "help with churn")
        A.turn(s, Script(("", [("ask", {"question": "What will you do?"})])), "not sure really")
        self.assertEqual(s.case.declined, 1)

    def test_no_request_holds_a_cell_value(self):  # the purpose's data limit: names, kinds and notices, never a value
        from dclab_rnd.agent_eval.conversation_exam import arms as exam_arms
        from dclab_rnd.agent_eval.home import tables

        view = exam_arms.table_view("customers")
        s = A.CaseState(case=C.new("c"), table=A.table_of("customers.csv", view, len(tables.build("customers"))))
        client = Script(("", [("read_table", {})]), ("", [("ask", {"question": "What is the goal?"})]))
        A.turn(s, client, "help me with my customers")
        sent = json.dumps(client.seen)
        frame = tables.build("customers")
        for value in list(frame["customer_id"].head(50)) + ["Dubai", "Sharjah", "premium"]:
            self.assertNotIn(f'"{value}"', sent)
            self.assertNotIn(f" {value},", sent)

    def test_the_tools_are_registered_once_with_their_moves_and_a_guard(self):
        reg = default_registry()
        self.assertEqual(sorted(reg.names("case")), sorted(A.TOOL_NAMES))
        self.assertIn("case", reg.guards)
        self.assertEqual(reg.guards["case"](reg.get("case_ask"), None, {}, lambda: {"ok": True}), {"error": "the case tools need a case: run them through the lead agent's turn"})


class GuardToolTests(unittest.TestCase):  # 37.6: the guard layer's tools, each run and checked by code
    def frame_state(self):
        from dclab_rnd.agent_eval.home import tables

        s = state()
        s.frame = tables.build("customers")
        return s

    def test_a_query_computes_aggregates_and_a_missing_column_is_named(self):
        s = self.frame_state()
        client = Script(("", [("query", {"group_by": ["plan"], "measures": [{"name": "customers", "op": "count"}]}),
                              ("query", {"measures": [{"name": "life", "op": "mean", "column": "lifetime_spend"}]})]),
                        ("", [("ask", {"question": "Which plan matters most to you?"})]))
        A.turn(s, client, "how many customers per plan?")
        results = [m for m in client.seen[1] if m.get("role") == "tool"]
        self.assertIn("customers", results[0]["content"])
        self.assertIn("lifetime_spend", results[1]["content"])  # refused with the column's name: the agent can say it is missing
        self.assertTrue(any("basic" in c for c in s.computed))  # computed: its numbers may be said

    def test_a_blanket_drop_of_empty_rows_needs_the_persons_yes(self):  # the silent-drop trap
        s = self.frame_state()
        client = Script(("", [("preview_step", {"kind": "drop_missing", "columns": ["monthly_spend"]})]), ("", [("ask", {"question": "Shall I?"})]))
        A.turn(s, client, "just delete every row with an empty cell")
        result = json.loads(next(m for m in client.seen[1] if m.get("role") == "tool")["content"])
        self.assertFalse(result["applied"])
        self.assertIn("rows_before", result)

    def test_the_leakage_check_runs_on_the_outcome(self):
        s = self.frame_state()
        out = A._check_leakage(s, "churned_next_month")
        self.assertIn("suspects", out)
        self.assertIn("error", A._check_leakage(s, "no_such_column"))

    def test_a_build_needs_a_project_route_and_the_persons_words(self):
        s = self.frame_state()
        self.assertIn("error", A._request_build(s, "yes build it"))
        A.turn(s, Script(("", [("propose", {"options": OPTIONS})]), ("Two ways.", [])), "show me options")
        A.pick(s, Script(("Started.", [])), "Rank customers by risk")
        self.assertIn("error", A._request_build(s, "yes build it"))  # not their words
        s.case = C.hear(s.case, "No, please don't build it yet, I want to think")
        self.assertIn("error", A._request_build(s, "build it"))  # the review: a no is never a yes
        s.case = C.hear(s.case, "yes please build it now")
        out = A._request_build(s, "build it now")
        self.assertEqual(out["recorded"], "ml_project")
        self.assertIn("the build starts now", out["next"])
        self.assertIn("no result or number", out["next"])

    def test_the_person_reads_each_option_in_full_and_claude_alone_has_the_same_data_tools(self):
        from dclab_rnd.agent_eval.conversation_exam import arms as exam_arms

        text = exam_arms._option_text("a ranked list", "a list your team can act on", "needs past outcomes", "a day")
        self.assertIn("You get: a list your team can act on", text)
        self.assertIn("Effort: a day", text)
        self.assertEqual({t["function"]["name"] for t in exam_arms.DATA_TOOLS}, {"query_table", "check_leakage", "preview_step"})

    def test_a_request_never_holds_a_record_and_at_most_the_purposes_cell_values(self):
        from dclab_rnd.agent_eval.home import tables
        from dclab_rnd.models import settings

        s = self.frame_state()
        client = Script(("", [("query", {"group_by": ["city", "plan"], "measures": [{"name": "n", "op": "count"}]})]), ("", [("ask", {"question": "ok?"})]))
        A.turn(s, client, "customers by city and plan")
        sent = json.dumps(client.seen)
        frame = tables.build("customers")
        self.assertFalse(any(v in sent for v in frame["customer_id"].head(200)))
        values = {str(v) for col in ("segment", "city", "plan") for v in frame[col].unique()}
        self.assertLessEqual(sum(v in sent for v in values), settings.PURPOSES["case_agent"].cell_values)


class GuardEdgeTests(unittest.TestCase):  # the review of 37.6
    def frame(self):
        from dclab_rnd.agent_eval.home import tables

        return tables.build("customers")

    def test_a_preview_filter_follows_the_query_rules_and_small_counts_are_said_as_fewer_than(self):
        from dclab_rnd.agents import data_tools

        f = self.frame()
        out = data_tools.preview_step(f, "filter", filters=[{"column": "customer_id", "op": "eq", "value": str(f["customer_id"].iloc[0])}])
        self.assertIn("error", out)
        out = data_tools.preview_step(f, "filter", filters=[{"column": "tenure_months", "op": "gt", "value": 58}])
        self.assertTrue(isinstance(out["rows_after"], str) or out["rows_after"] >= 5)

    def test_a_query_that_cannot_run_is_a_refusal_and_cuts_are_said(self):
        from dclab_rnd.agents import data_tools

        f = self.frame()
        self.assertIn("error", data_tools.query(f, filters=[{"column": "plan", "op": "gt", "value": 3}]))
        self.assertIn("error", data_tools.query(f, group_by=["segment", "city", "plan"]))  # at most two columns
        out = data_tools.query(f, group_by=["city", "plan"], sort=[{"by": "rows", "desc": True}])
        self.assertIn("groups_not_shown", out)
        self.assertIn("no holdout", out["note"])
        self.assertTrue(out["exposure"])

    def test_a_case_sees_at_most_forty_group_names(self):
        from dclab_rnd.agents import data_tools

        seen = {f"name-{i}" for i in range(38)}
        self.assertIn("error", data_tools.query(self.frame(), group_by=["city"], seen=seen))

    def test_claude_alone_reads_a_wrong_argument_as_a_refusal(self):
        from dclab_rnd.agent_eval.conversation_exam import arms as exam_arms

        client = Script(("", []))
        arm = exam_arms.PlainArm(client)
        arm.upload, arm.frame = "customers", self.frame()
        arm.messages.append({"role": "user", "content": "hi"})

        class C2:
            calls = [{"content": "", "tool_calls": [{"id": "q1", "name": "query_table", "arguments": {"limit": 3, "group_by": ["plan"]}}],
                      "assistant_message": {"role": "assistant", "content": ""}}, {"content": "done", "tool_calls": [], "assistant_message": {"role": "assistant", "content": "done"}}]

            def complete(self, *a, **k):
                return self.calls.pop(0)

        arm.client = C2()
        turn = arm._turn()
        self.assertEqual(turn["text"], "done")  # an extra argument dropped, never a crash


class EndToEndTests(unittest.TestCase):
    def test_it_plays_every_dev_person_to_a_route_with_scripted_models(self):  # 37.3's done-when, scripted
        from dclab_rnd.agent_eval import conversation_exam as X
        from dclab_rnd.agent_eval.conversation_exam import arms as exam_arms

        class Person:
            def __init__(self):
                self.n = 0

            def answer(self, transcript):
                self.n += 1
                return {"say": "show me what you can do", "pick": None} if self.n == 1 else {"say": "the first", "pick": 1}

        class Keeps(Script):
            """A scripted agent that, told once a route doubt (the plan's step 1), keeps its options: it may."""

            def complete(self, messages, tools=None, *args, **kwargs):
                last = messages[-1] if messages else {}
                if last.get("role") == "tool" and "may call for" in str(last.get("content")):
                    self.doubted = getattr(self, "doubted", 0) + 1
                    self.replies.insert(0, ("", [("propose", {"options": [dict(o, columns=[]) for o in OPTIONS]})]))
                return super().complete(messages, tools, *args, **kwargs)

        def client(person):
            first = [("", [("query", {"measures": [{"name": "rows", "op": "count"}]})])] if person.get("upload") and C.wants_figures([person["opening"]]) else []
            return Keeps(*first, ("Tell me more about what you need.", [("ask", {"question": "What would you decide with the result?"})]),  # a person who
                         ("", [("propose", {"options": [dict(o, columns=[]) for o in OPTIONS]})]), ("Here are two ways.", []),  # asks for figures of
                         ("Good, I'll start.", []))  # their table gets them first (2026-11-02): a query comes before the question

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        for p in X.load("dev"):
            conv = X.play(p, exam_arms.CaseArm(client(p)), Person())
            self.assertEqual(conv["route"], "prediction", p["id"])
            self.assertEqual(conv["ended"], "routed", p["id"])
            self.assertIn("case_file", conv["notes"])


class ActFirstTests(unittest.TestCase):
    """2026-11-02 (the owner's Telco session: nine questions and no figure for twelve messages): a person who asks for figures
    of the table they shared gets them first; the decision and the user are not asked before; a bare yes after the agent's
    question, and an order to show, are go-aheads."""

    ASKED = ["I face losing my customer I want know which segment of my customer I lost more", "give me a report of segment of my customer",
             "now show my report and give dashboard", "give plot", "no just show me the report and dashboard what I sked",
             "how many customers left by contract?", "compare churn by plan"]

    def test_figures_are_asked_for_in_these_words_and_not_in_a_request_for_a_model(self):
        for w in self.ASKED:
            self.assertTrue(C.asks_for_figures(w), w)
        for w in ("we are losing customers", "I want a model that predicts who will leave", "should we drop the basic plan"):
            self.assertFalse(C.asks_for_figures(w), w)

    def ready_case(self, words):
        case = C.new("case-f")
        for w in words:
            case = C.hear(case, w)
        case = C.apply(case, "goal", C.CaseFact(value=" ".join(words[0].split()[:6]), status="said", quote=" ".join(words[0].split()[:6])), by="agent", why="noted")
        return C.apply(case, "data", C.CaseFact(value="t.csv: 7043 rows, 21 columns", status="computed", computed="profile"), by="code", why="read")

    def test_a_figures_request_on_a_shared_table_is_ready_with_only_the_goal_and_the_data(self):
        r = C.readiness(self.ready_case([self.ASKED[0]]))
        self.assertTrue(r.ready and r.may_propose, r.reasons)

    def test_without_a_figures_request_the_decision_and_the_user_are_still_needed(self):
        r = C.readiness(self.ready_case(["we are losing customers"]))
        self.assertFalse(r.ready)
        self.assertEqual(sorted(r.missing), ["decision", "user"])

    def test_a_prediction_request_still_needs_the_decision_and_the_user(self):
        r = C.readiness(self.ready_case(["which customers will leave, and give me a report"]))
        self.assertFalse(r.ready)

    def test_an_unknown_about_the_decision_does_not_block_a_figures_request(self):
        case = self.ready_case([self.ASKED[0]])
        case = C.apply(case, "unknowns", C.Unknown(field="decision", question="What will you do with it?", blocks="all", rank=1), by="agent", why="open")
        self.assertTrue(C.readiness(case).ready)

    def test_an_order_to_show_and_a_bare_yes_to_the_agents_question_are_go_aheads(self):
        for w in ("now show my report and give dashboard", "give plot", "no just show me the report and dashboard what I sked"):
            self.assertTrue(A.goes_ahead(w), w)
        self.assertTrue(A.goes_ahead("yes right", asking=True))
        self.assertTrue(A.goes_ahead("go", asking=True))
        self.assertFalse(A.goes_ahead("yes right", asking=False))  # a yes to nothing is not an order
        self.assertFalse(A.goes_ahead("don't show the report yet", asking=False))

    def test_a_question_before_the_figures_is_refused_and_allowed_after_a_query_ran(self):
        s = state()
        s.frame = __import__("pandas").DataFrame(QueryCapTests.FRAME)
        q = ("query", {"group_by": ["plan"], "measures": [{"name": "customers", "op": "count"}]})
        client = Script(("", [("ask", {"question": "What will you do with the answer?"})]), ("", [q]),
                        ("Plan x has 3 customers and plan y has 3.", [("ask", {"question": "Which plan do you want to look at next?"})]))
        out = A.turn(s, client, "which plan loses more? show me")
        first = [m["content"] for m in client.seen[1] if m.get("role") == "tool"][0]
        self.assertIn("ask the table now", first)
        self.assertEqual(s.case.asked, 1)  # only the question after the figures was counted
        self.assertIn("Plan x has 3 customers", out["text"])
    def test_a_chart_drawn_in_block_characters_never_reaches_the_person(self):
        s = state()
        text = "Here is a quick plot.\nMonth-to-month \u2588\u2588\u2588\u2588\u2588\u2588 One year \u2588\u2588\u2588 Two year \u2588\nMonth-to-month customers leave the most."
        self.assertEqual(A.checked(text, s), "Here is a quick plot. Month-to-month customers leave the most.")
        self.assertEqual(A.checked("Contract \u2588\u2588 matters most to this group of customers across the whole table of data.", s),
                         "Contract matters most to this group of customers across the whole table of data.")

    def test_the_pages_create_a_project_message_is_a_yes_to_building_and_a_go_ahead(self):
        self.assertTrue(A.says_yes("Create a project from this chat"))
        self.assertTrue(A.goes_ahead("Create a project from this chat"))
        self.assertFalse(A.says_yes("Don't create a project yet"))
        for question in ("Can I create a project later?", "How do I create a project?", "create a project from this chat when it is ready?"):
            self.assertFalse(A.says_yes(question), question)  # a question is no yes (the review of 37.11j)
            self.assertFalse(A.goes_ahead(question), question)

    def test_requests_that_are_not_figures_stay_with_the_model_and_the_decision_questions(self):
        for w in ("Rank my customers by risk of leaving", "Compare random forest and svm for imbalanced data",
                  "How many folds should I use for cross validation?", "Which plan should I pick for hosting my data?"):
            self.assertFalse(C.wants_figures(w), w)

    def test_a_personal_column_is_refused_before_any_number_is_seen(self):
        s = state()
        s.frame = __import__("pandas").DataFrame(QueryCapTests.FRAME)
        s.personal = {"plan"}
        client = Script(("", [("query", {"group_by": ["plan"], "measures": [{"name": "n", "op": "count"}]}),
                              ("dashboard", {"tiles": [{"group_by": ["plan"], "measures": [{"name": "n", "op": "count"}]}]})]), ("I cannot use that column.", []))
        A.turn(s, client, "which plan loses more?")
        results = [m["content"] for m in client.seen[1] if m.get("role") == "tool"]
        self.assertTrue(all("holds personal data" in r for r in results), results)
        self.assertEqual(s.cards, [])

    def test_a_dashboard_after_an_answer_moves_the_route_so_the_board_can_be_refreshed(self):
        s = state()
        s.frame = __import__("pandas").DataFrame(QueryCapTests.FRAME)
        a = ("answer", {"group_by": ["plan"], "measures": [{"name": "customers", "op": "count"}]})
        d = ("dashboard", {"tiles": [{"group_by": ["plan"], "measures": [{"name": "customers", "op": "count"}]}]})
        A.turn(s, Script(("", [a]), ("Plan x has 3 customers.", [])), "which plan loses more? show me")
        self.assertEqual(s.case.route.tool, "query")
        A.turn(s, Script(("", [d]), ("The dashboard shows customers by plan.", [])), "now give dashboard")
        self.assertEqual((s.case.route.track, s.case.route.tool), ("reporting", "dashboard"))

    def test_a_decision_question_is_not_asked_twice_nor_after_an_order_for_figures(self):
        q = "What will you do with this? For example offer them a deal."
        self.assertEqual(A.without_decision_questions("Consumer lost 174. " + q), "Consumer lost 174.")
        self.assertEqual(A.without_decision_questions("Who will read the report?"), "")
        s = state()
        s.turns = [{"role": "user", "content": "x"}, {"role": "assistant", "content": q}]
        out = A.turn(s, Script((q + " Consumer lost the most.", [])), "just show me the report")
        self.assertNotIn("What will you do", out["text"])
        self.assertIn("Consumer lost the most.", out["text"])

    def test_the_first_reply_to_a_figures_request_does_not_ask_what_they_will_do(self):
        s = state()
        out = A.turn(s, Script(("Consumer lost the most. What will you do with this?", [])), "which segment of my customer I lost more")
        self.assertEqual(out["text"], "Consumer lost the most.")

    def test_the_figures_shown_are_the_work_and_the_case_says_so_only_for_a_person_who_asked_for_them(self):
        s = state()
        s.frame = __import__("pandas").DataFrame(QueryCapTests.FRAME)
        q = ("answer", {"group_by": ["plan"], "measures": [{"name": "customers", "op": "count"}]})
        A.turn(s, Script(("", [q]), ("Plan x has 3 customers.", [])), "which plan loses more? show me")
        self.assertEqual((s.case.route.track, s.case.route.tool, s.case.route.by), ("analysis", "query", "agent"))
        t = state()
        t.frame = __import__("pandas").DataFrame(QueryCapTests.FRAME)
        A.turn(t, Script(("", [q]), ("Plan x has 3 customers.", [])), "I need a model that will predict who will leave")
        self.assertIsNone(t.case.route)  # a model was asked for: showing a count is not the work


class OrdersTests(unittest.TestCase):
    """2026-11-02 (the owner's second Telco session): five messages to start one build. "yes" could not be quoted (two words were
    needed), "build the model" was not read as a yes, and the agent asked the person to type a phrase."""

    def routed(self, words):
        s = state()
        A.turn(s, Script(("", [("propose", {"options": OPTIONS})]), ("Two ways.", [])), "show me options")
        A.pick(s, Script(("Started.", [])), "Rank customers by risk")
        s.case = C.hear(s.case, words)
        return s

    def test_a_bare_yes_to_the_agents_question_is_a_yes_and_can_be_quoted_whole(self):
        s = self.routed("yes")
        s.turns = [{"role": "assistant", "content": "A model would score each customer. Shall I build it now?"}, {"role": "user", "content": "yes"}]
        self.assertEqual(A._request_build(s, "yes")["recorded"], "ml_project")

    def test_a_bare_yes_to_a_question_about_something_else_is_not_a_yes_to_building(self):
        s = self.routed("yes")
        s.turns = [{"role": "assistant", "content": "Want me to also show churn by contract?"}, {"role": "user", "content": "yes"}]
        self.assertIn("not about building", A._request_build(s, "yes")["error"])

    def test_negated_conditional_and_reported_orders_are_not_orders(self):
        for w in ("I'd rather not build a model", "there is no need to build a model", "I won't build a model until you show me churn by contract",
                  "We should not build a model yet.", "Not now. Build a model after the report", "build the model once I confirm",
                  "Build a model only after the leakage check", "build a model tomorrow", "my colleague said: build a model",
                  "if we build a model what happens", "I won't train a model", "make a model of the business", "create a model of costs in excel",
                  "build a model without the contract column", "no model please, whatever you think"):
            self.assertFalse(A.orders_model(w), w)
            self.assertFalse(A.says_yes(w), w)
            self.assertFalse(A.goes_ahead(w), w)

    def test_a_report_is_not_blocked_by_a_mention_of_risk_or_a_yes_to_an_offered_chart(self):
        for words in ("Which contract has the highest churn risk? chart it", "show a report of customers likely to leave", "yes please"):
            s = state()
            s.frame = __import__("pandas").DataFrame(QueryCapTests.FRAME)
            client = Script(("", [("answer", {"group_by": ["plan"], "measures": [{"name": "customers", "op": "count"}]})]), ("Plans x and y have 3 each.", []))
            A.turn(s, client, words)
            first = [m["content"] for m in client.seen[1] if m.get("role") == "tool"][0]
            self.assertNotIn("a model is the work now", first, words)

    def test_the_outcome_needs_the_persons_own_outcome_words_in_the_same_sentence_as_the_column(self):
        case = C.new("c")
        case = C.hear(case, "Which contract type loses most customers? Is it different by gender?")
        self.assertEqual(C.grounded_columns(case, ["gender", "Contract"]), ["Contract"])  # gender is only mentioned
        case = C.hear(case, "the churn_flag column says who left")
        self.assertEqual(C.grounded_columns(case, ["gender", "churn_flag"]), ["churn_flag"])

    def test_a_long_message_still_needs_its_own_two_word_quote(self):
        s = self.routed("yes, but only after I have asked my team about the weekly list size")
        self.assertIn("error", A._request_build(s, "yes"))  # one word of a long message is not a quote
        self.assertEqual(A._request_build(s, "yes, but only after I have asked my team")["recorded"], "ml_project")

    def test_an_order_to_build_a_model_is_a_yes_and_a_question_about_building_is_not(self):
        for w in ("build the model", "now build model", "make prediction models", "train a model please", "Create a churn model"):
            self.assertTrue(A.says_yes(w), w)
            self.assertTrue(A.goes_ahead(w), w)
        for w in ("how do I build a model?", "can you build a model later", "should I build a model", "what does a model need?", "don't build the model yet"):
            self.assertFalse(A.says_yes(w), w)
            self.assertFalse(A.goes_ahead(w), w)

    def test_you_decide_is_a_go_ahead(self):
        for w in ("you decide", "your call", "up to you", "whatever you think"):
            self.assertTrue(A.goes_ahead(w), w)

    def test_the_outcome_column_is_taken_only_when_the_persons_words_name_it(self):
        s = self.routed("yes build it now")
        s.table = A.table_of("t.csv", [{"name": "churn_flag", "kind": "category", "noticed": []}, {"name": "plan", "kind": "category", "noticed": []}], 100)
        self.assertIn("error", A._request_build(s, "yes build it now", outcome="churn_flag"))  # nobody named it
        s.case = C.hear(s.case, "yes build it now, the churn_flag column says who left")
        out = A._request_build(s, "yes build it now", outcome="churn_flag")
        self.assertEqual(s.build["outcome"], "churn_flag")
        self.assertIn("error", A._request_build(s, "yes build it now", outcome="no_such_column"))

    def test_no_report_is_drawn_while_the_work_is_a_model_or_the_person_handed_the_choice_over(self):
        s = state()
        s.frame = __import__("pandas").DataFrame(QueryCapTests.FRAME)
        q = ("answer", {"group_by": ["plan"], "measures": [{"name": "customers", "op": "count"}]})
        for words in ("now build model", "you decide", "make prediction models"):
            client = Script(("", [q]), ("Understood.", []))
            A.turn(s, client, words)
            first = [m["content"] for m in client.seen[1] if m.get("role") == "tool"][0]
            self.assertIn("a model is the work now", first, words)
        self.assertEqual(s.cards, [])
        client = Script(("", [q]), ("Plan x has 3 customers.", []))
        A.turn(s, client, "also show me a chart of customers by plan")  # figures asked for: drawn even while a model is the work
        self.assertEqual(len(s.cards), 1)



if __name__ == "__main__":
    unittest.main()


class CaptureTests(unittest.TestCase):
    def test_the_lead_agents_thinking_summary_is_stored_whole(self):  # rule 8: a new purpose keeps its thinking
        import os
        from unittest import mock

        from dclab_rnd import experience
        from dclab_rnd.models import gateway as gw

        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, True)

        class T:
            thinking = {"type": "adaptive"}

            def complete(self, messages, tools=None, max_tokens=1800, **options):
                return {"content": "What will you decide?", "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1},
                        "thinking": [{"text": "the decision is still unknown"}], "assistant_message": {"role": "assistant", "content": "What will you decide?"}}

        clean = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY",
                                 "DCLAB_NO_LIVE_MODELS", "DCLAB_THINKING")} | {"OPENAI_API_KEY": "k", "DCLAB_AGENT_HOME": str(home)}
        transport = T()
        with mock.patch.object(gw, "_transport", lambda tier, purpose: transport), mock.patch.dict(os.environ, clean), \
                mock.patch.object(experience, "current_log", lambda: None):
            client = gw.Gateway(transport=gw._transport).client(A.PURPOSE, draft_id="exam:test:case-p")
            A.turn(state(table=False), client, "we keep losing customers")
        docs = [json.loads(line) for p in sorted(home.rglob("model_exchanges/**/*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(docs and all(d["purpose"] == "case_agent" for d in docs))
        self.assertTrue(docs[0]["whole"])  # no query answer in it: kept whole, messages and all (the review of 37.6)
        self.assertEqual(docs[0]["thinking"], [{"text": "the decision is still unknown"}])
        self.assertTrue(docs[0]["messages"])

    def test_the_persons_words_stay_in_the_state_when_old_turns_are_dropped(self):  # the review: quotes need their words
        case = C.new("c", ["first message with the goal in it", "second"])
        built = ctx.build(case, [{"role": "user", "content": "x " * 40_000}] * 5, budget=2000)
        self.assertIn("first message with the goal in it", built["state"])


class SpecialistTests(unittest.TestCase):  # 37.7
    def test_a_specialist_is_sent_only_when_its_trigger_holds_and_within_the_cases_budget(self):
        from dclab_rnd.agents import specialists as S

        case = C.new("c", ["help"])
        self.assertIn("no table", S.allowed("data_investigator", case, None, []))
        self.assertIsNone(S.allowed("data_investigator", case, A.table_of("t.csv", VIEW, 10), []))
        self.assertIn("nothing has started", S.allowed("solution_designer", case, None, []))
        self.assertIn("already", S.allowed("critic", case, None, ["critic"]))
        self.assertIn("consults", S.allowed("critic", case, None, ["a", "b", "c"]))

    def test_its_findings_go_into_the_case_file_and_it_never_sees_a_group_name(self):
        s = state()
        s.computed = ['{"columns": ["plan", "n"], "rows": [["premium", 3]], "groups_not_shown": 0}', '{"suspects": []}']
        spec = Script(json.dumps({"findings": [{"kind": "risk", "text": "No date of cancellation: a month ahead must be assumed.", "field": "data"},
                                               {"kind": "nonsense", "text": "x"}]}))

        class Holder:  # the specialist's client sees plain text; the Script returns its reply as content
            def __init__(self, inner):
                self.inner, self.seen = inner, []

            def complete(self, messages, tools=None, **k):
                self.seen.append(json.dumps(messages))
                return {"content": self.inner.replies.pop(0)}

        s.specialist_client = Holder(spec)
        out = A._consult(s, "data_investigator", "What can this table answer about churn?")
        self.assertEqual(len(out["findings"]), 1)  # the finding of no known kind is dropped
        self.assertEqual((s.case.findings[0].by, s.case.findings[0].kind), ("data_investigator", "risk"))
        self.assertEqual(s.case.changes[-1].by, "specialist:data_investigator")
        self.assertNotIn("premium", s.specialist_client.seen[0])  # a query's group names never reach it
        self.assertIn("data_investigator found", ctx.case_text(s.case))

    def test_the_specialists_thinking_is_stored_whole(self):  # rule 8
        import os
        from unittest import mock

        from dclab_rnd import experience
        from dclab_rnd.models import gateway as gw

        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, True)

        class T:
            thinking = {"type": "adaptive"}

            def complete(self, messages, tools=None, max_tokens=1800, **options):
                body = json.dumps({"findings": [{"kind": "fact", "text": "One row is one customer."}]})
                return {"content": body, "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1},
                        "thinking": [{"text": "the grain is a customer"}], "assistant_message": {"role": "assistant", "content": body}}

        clean = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY",
                                 "DCLAB_NO_LIVE_MODELS", "DCLAB_THINKING")} | {"OPENAI_API_KEY": "k", "DCLAB_AGENT_HOME": str(home)}
        transport = T()
        with mock.patch.object(gw, "_transport", lambda tier, purpose: transport), mock.patch.dict(os.environ, clean), \
                mock.patch.object(experience, "current_log", lambda: None):
            s = state()
            s.specialist_client = gw.Gateway(transport=gw._transport).client("case_specialist", draft_id="exam:test:case-s")
            A._consult(s, "data_investigator", "What is one row?")
        docs = [json.loads(line) for p in sorted(home.rglob("model_exchanges/**/*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(docs[0]["purpose"], "case_specialist")
        self.assertTrue(docs[0]["whole"])
        self.assertEqual(docs[0]["thinking"], [{"text": "the grain is a customer"}])


class SessionTests(unittest.TestCase):  # 37.8
    def test_a_checkpoint_after_every_turn_resumes_with_nothing_lost(self):
        from dclab_rnd.agents import case_session

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        s = state()
        s.checkpoint = folder / "case.json"
        A.turn(s, Script(("", [("read_table", {})]), ("", [("note", {"field": "goal", "value": "keep customers", "status": "said", "quote": "losing customers"}),
                                                          ("ask", {"question": "Who acts on it?"})])), "we are losing customers")
        back = case_session.load(s.checkpoint, frame_of=lambda name: None)
        self.assertEqual(back.case, s.case)
        self.assertEqual((back.turns, back.computed, back.asking), (s.turns, s.computed, s.asking))
        self.assertEqual(back.trace.run_id, s.trace.run_id)  # the same conversation in the record
        A.turn(back, Script(("", [("ask", {"question": "And the data?"})])), "the retention team")
        self.assertEqual(back.case.version, s.case.version + 2)  # carried on from where it stopped

    def test_a_case_at_its_cap_stops_and_asks_and_a_yes_goes_on(self):
        s = state(table=False)
        s.spent_eur, s.cap_eur = 5.2, 5.0
        client = Script(("never called", []))
        out = A.turn(s, client, "and then?")
        self.assertIn("budget", out["text"])
        self.assertEqual(client.seen, [])  # no model work past the cap
        out = A.turn(s, Script(("Going on.", [])), "yes, go on")
        self.assertEqual(out["text"], "Going on.")
        self.assertEqual(s.cap_eur, 10.2)  # a yes buys a whole cap beyond what was already spent (the review)


class RecorderTests(unittest.TestCase):  # 37.12
    def test_every_change_is_an_event_when_it_is_made_and_the_case_rebuilds_from_them(self):
        from dclab_rnd import experience

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        log = experience.FileLog(folder / "experience", "case-test")
        previous = experience._INSTALLED.get("log")
        experience.install(log)
        self.addCleanup(experience.install, previous)
        s = state()
        A.turn(s, Script(("", [("read_table", {})]), ("", [("note", {"field": "goal", "value": "keep customers", "status": "said", "quote": "losing customers"}),
                                                          ("add_unknown", {"field": "outcome", "question": "How do you know one left?", "blocks": "some"}),
                                                          ("ask", {"question": "Who acts on it?"})])), "we are losing customers")
        events = [json.loads(line) for p in sorted(folder.rglob("*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]
        changes = [e for e in events if e.get("kind") == "case.changed"]
        self.assertEqual(len(changes), s.case.version)  # one event per version, none missed
        rebuilt = C.rebuild(s.case.case_id, [{"version": e["payload"]["version"], "field": e["payload"]["field"], "op": e["payload"]["op"],
                                              "by": e["payload"]["by"], "why": "", "after": json.loads(e["payload"]["after"])} for e in changes])
        for field in ("goal", "data", "unknowns", "words", "asked", "version"):
            self.assertEqual(getattr(rebuilt, field), getattr(s.case, field), field)

    def test_a_resumed_case_records_no_old_change_again(self):
        s = state()
        A.turn(s, Script(("", [("ask", {"question": "Who?"})])), "help")
        again = A.CaseState(case=s.case)
        self.assertEqual(again.recorded, s.case.version)


class PrivacyEdgeTests(unittest.TestCase):  # the re-review of 37.5 to 37.8
    def test_once_a_case_has_seen_group_names_every_request_holds_cells_persian_names_too(self):
        s = state()
        s.group_names = {"تهران"}

        class Inner:
            request_cells = None

            def complete(self, messages, *a, **k):
                self.flag = self.request_cells
                return {"content": "ok", "tool_calls": [], "assistant_message": {"role": "assistant", "content": "ok"}}

        inner = Inner()
        A._Cells(inner, s).complete([{"role": "user", "content": "and the next turn"}])
        self.assertIs(inner.flag, True)
        fresh = A._Cells(Inner(), state())
        fresh.complete([{"role": "user", "content": "hello"}])
        self.assertIs(fresh._inner.flag, False)

    def test_a_specialist_asked_about_a_group_by_name_has_its_request_marked(self):
        s = state()
        s.group_names = {"Shiraz"}

        class Holder:
            request_cells = None

            def complete(self, messages, tools=None, **k):
                self.flag = self.request_cells
                return {"content": json.dumps({"findings": [{"kind": "fact", "text": "x"}]})}

        s.specialist_client = Holder()
        A._consult(s, "data_investigator", "Why does Shiraz churn most?")
        self.assertIs(s.specialist_client.flag, True)

    def test_query_groups_never_reach_a_trace_row(self):
        from dclab_rnd.agents import traces

        kept = json.dumps(traces.summary({"rows": [["Shiraz", 30]], "columns": ["city", "n"]}))
        self.assertNotIn("Shiraz", kept)

    def test_the_cap_follows_the_setting_and_a_checkpoint_keeps_names_exposure_and_the_tracers_place(self):
        import os
        from unittest import mock

        from dclab_rnd.agents import case_session

        with mock.patch.dict(os.environ, {"DCLAB_CASE_CAP_EUR": "1"}):
            self.assertEqual(A.CaseState(case=C.new("c")).cap_eur, 1.0)
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        s = state()
        s.group_names, s.exposure = {"basic", "plus"}, [{"columns_related": [["plan", "churned_next_month"]]}]
        A.turn(s, Script(("", [("ask", {"question": "Who?"})])), "help")
        path = case_session.save(s, folder / "c.json")
        back = case_session.load(path)
        self.assertEqual((back.group_names, back.exposure, back.checkpoint), (s.group_names, s.exposure, path))
        self.assertEqual((back.trace.n, back.trace.last_reply), (s.trace.n, s.trace.last_reply))

    def test_the_leakage_check_says_what_it_related_on_every_row(self):
        from dclab_rnd.agent_eval.home import tables
        from dclab_rnd.agents import data_tools

        out = data_tools.leakage(tables.build("customers"), "churned_next_month")
        self.assertEqual(out["exposure"]["rows"], "all")
        self.assertIn(["plan", "churned_next_month"], out["exposure"]["columns_related"])

    def test_the_critic_and_the_domain_expert_have_their_triggers(self):
        from dclab_rnd.agents import specialists as S

        case = C.new("c", ["we lose customers every month"])
        self.assertIn("missing", S.allowed("critic", case, None, []))
        self.assertIn("goal", S.allowed("domain_expert", case, None, []))
        case = C.apply(case, "goal", C.CaseFact(value="keep customers", status="said", quote="lose customers"), by="agent", why="w")
        self.assertIsNone(S.allowed("domain_expert", case, None, []))
        self.assertIn("missing", S.allowed("critic", case, None, []))
        self.assertIn("leakage cannot arise", S.allowed("leakage_auditor", case, A.table_of("t.csv", [{"name": "x", "kind": "numeric", "noticed": []}], 3), []))

    def test_a_version_one_case_file_reads_as_version_two(self):
        doc = json.loads(C.new("c", ["hi there"]).model_dump_json())
        doc["schema_version"] = "dclab_core.case_file.v1"
        doc.pop("findings"); doc.pop("build")
        self.assertEqual(C.CaseFile.read(doc).schema_version, "dclab_core.case_file.v2")


class NeverSilentTests(unittest.TestCase):
    def test_a_turn_spent_on_tools_is_asked_once_for_its_words_with_no_move(self):  # CEX-011 d-04
        s = state()
        calls = [("", [("read_table", {})])] * 8
        client = Script(*calls, ("Your table has 1200 customers. Who will act on the list?", [("ask", {"question": "ignored: no move now"})]))
        out = A.turn(s, client, "help me with my customers")
        self.assertIn("Who will act on the list?", out["text"])
        self.assertEqual(s.case.asked, 0)  # the forced message ran no move


class SpeedAndGoAheadTests(unittest.TestCase):
    """2026-10-31 (the sealed run: 34 s a turn, 17 repeated questions): the table is read by code before the agent's first
    call, so the agent does not spend a call on read_table; a confirmation question right after the person said go ahead is
    refused, so the agent acts."""

    def test_code_reads_the_shared_table_before_the_first_model_call(self):
        s = state()
        client = Script(("Your table has 1200 rows. Who will act on the list?", [("ask", {"question": "Who will act on the list?"})]))
        out = A.turn(s, client, "help me with my customers")
        self.assertIsNotNone(s.case.data)  # the data fact is code's, without a read_table call
        self.assertEqual(s.case.data.status, "computed")
        self.assertIn("1200 rows", out["text"])  # the row count may be said: code computed it
        self.assertEqual(len(client.seen), 1)  # one model call for the whole turn

    def test_a_confirmation_after_go_ahead_is_refused_and_the_agent_acts(self):
        s = state()
        A.turn(s, Script(("", [("ask", {"question": "Should I rank customers by risk?"})])), "help me keep customers")
        client = Script(("", [("ask", {"question": "Does that match what you need?"})]),
                        ("Starting now: DCLab will rank your customers by risk.", [("start", {"track": "prediction", "tool": "ml_project", "quote": "go ahead"})]))
        out = A.turn(s, client, "Yes, go ahead with it.")
        refused = [m for m in client.seen[1] if m.get("role") == "tool"]
        self.assertIn("go ahead", refused[0]["content"])
        self.assertNotIn("Does that match", out["text"])

    def test_after_go_ahead_a_question_about_something_missing_is_still_allowed(self):
        s = state()
        A.turn(s, Script(("", [("ask", {"question": "What is your goal?"})])), "help me")
        out = A.turn(s, Script(("", [("ask", {"question": "Which column says a customer left?"})])), "go ahead")
        self.assertIn("Which column says a customer left?", out["text"])

    def test_stop_asking_counts_as_go_ahead(self):
        self.assertTrue(A.goes_ahead("Stop asking me for the export date. Run the check now."))
        self.assertTrue(A.goes_ahead("Go ahead with the build."))
        self.assertFalse(A.goes_ahead("Don't go ahead yet, wait for my manager."))
        self.assertFalse(A.goes_ahead("We want to keep members."))


class QueryCapTests(unittest.TestCase):
    """2026-10-31 (CEX-014: 98 table queries in 10 conversations once the table was read by code): a cap on the queries that
    ran in a turn. 2026-11-02 (the owner's Telco session): the cap is four, a refused query is not counted (two refusals had
    closed the turn and the agent said "next turn"), and three refusals end the tries with words to say now."""

    FRAME = {"customer_id": ["a", "b", "c", "d", "e", "f"], "plan": ["x", "x", "x", "y", "y", "y"], "churned_next_month": [1, 0, 0, 1, 1, 0]}

    def frame_state(self):
        s = state()
        s.frame = __import__("pandas").DataFrame(self.FRAME)
        return s

    def test_a_fifth_query_in_one_turn_is_refused(self):
        s = self.frame_state()
        q = ("query", {"group_by": ["plan"], "measures": [{"name": "customers", "op": "count"}]})
        client = Script(("", [q] * 5), ("Plans x and y each have 3 customers. Who acts on this?", [("ask", {"question": "Who acts on this?"})]))
        A.turn(s, client, "how do my plans compare?")
        results = [m["content"] for m in client.seen[1] if m.get("role") == "tool"]
        self.assertNotIn("error", results[3])
        self.assertIn("asked 4 times this turn", results[4])
        self.assertNotIn("next turn", results[4])
        A.turn(s, Script(("", [q]), ("ok", [("ask", {"question": "Next?"})])), "and now?")  # a new turn may query again
        self.assertEqual(s.queries, 1)

    def test_a_refused_query_is_not_counted_so_it_can_be_repaired_in_the_same_turn(self):
        s = self.frame_state()
        bad = ("query", {"group_by": ["plan"], "measures": [{"name": "share", "op": "rate", "column": "plan"}]})
        good = ("query", {"group_by": ["plan"], "measures": [{"name": "left", "op": "count", "where": {"column": "churned_next_month", "op": "eq", "value": 1}}]})
        client = Script(("", [bad, bad]), ("", [good]), ("Plan x lost 1 and plan y lost 2.", [("ask", {"question": "What will you do about plan y?"})]))
        A.turn(s, client, "which plan loses more?")
        self.assertEqual((s.queries, s.query_errors), (1, 2))
        self.assertTrue(any('"left"' in c or "as_said" in c for c in s.computed))  # the repaired query ran and its numbers may be said

    def test_three_refused_queries_end_the_tries_with_words_to_say_now_not_a_promise_of_later(self):
        s = self.frame_state()
        bad = ("query", {"group_by": ["plan"], "measures": [{"name": "share", "op": "rate", "column": "plan"}]})
        client = Script(("", [bad, bad, bad, bad]), ("I could not work that out; here is what I know.", [("ask", {"question": "Shall I try another way?"})]))
        A.turn(s, client, "which plan loses more?")
        results = [m["content"] for m in client.seen[1] if m.get("role") == "tool"]
        self.assertIn("three questions of the table were refused", results[3])
        self.assertIn("do not say you will try again later", results[3])

    def test_send_it_now_is_a_go_ahead(self):
        self.assertTrue(A.goes_ahead("You still haven't sent the table I asked for. Send it now."))


class YesAfterYesTests(unittest.TestCase):
    """2026-11-01 (sealed v2, blind pairs, a second labeller): the lead agent asked for a yes after the person said yes. A
    long yes ("Yes, start the build ... hold the holdout until ...") failed the yes check on the word "hold", so the build
    request was refused and the agent asked again; and it wrote "Shall I build now?" in its text after a go-ahead."""

    def test_a_long_yes_with_conditions_counts(self):
        for w in ("Yes, start the build. Begin with the holdout churner count, and hold the holdout until we have locked the margin.",
                  "Yes, build it that way. Three things before you start. Go ahead with the build on the extract for now.",
                  "Go ahead: start with the churner count. Keep the holdout closed until the build is done.",
                  "Sure, build it."):
            self.assertTrue(A.says_yes(w), w)

    def test_a_refusal_still_refuses(self):
        for w in ("No, do not build yet.", "Wait, not yet.", "Yes, but don't build it until Monday.", "Please hold off on the build.",
                  "Sure, but wait until my manager agrees before you start."):
            self.assertFalse(A.says_yes(w), w)

    def test_a_confirmation_question_in_the_text_after_a_go_ahead_is_dropped(self):
        s = state()
        A.turn(s, Script(("", [("ask", {"question": "Should I rank customers by risk?"})])), "help me keep customers")
        out = A.turn(s, Script(("Your yes is noted: DCLab starts ranking customers by risk now. Shall I build now?", [])), "Yes, go ahead and build it.")
        self.assertIn("starts ranking", out["text"])
        self.assertNotIn("Shall I build", out["text"])


class NumberCutTests(unittest.TestCase):
    """2026-11-01 (CEX-018 d-36, d-15): the number check dropped the heart of an answer and its rewrite was cut again, so only
    'Does that cover what you need?' reached the person. The rewrite now names the numbers allowed, and a rewrite left with
    no statement (questions only) or mostly cut gets one stricter try."""

    def test_a_rewrite_cut_again_to_a_bare_question_gets_one_stricter_try(self):
        s = state(table=False)
        client = Script(("Report PR-AUC: ROC-AUC can read 0.91 while precision is 0.12. Does that cover it?", []),
                        ("Use PR-AUC; ROC can show 0.9 when few flags are right. Does that cover it?", []),
                        ("Report PR-AUC, with your churn rate beside it as the baseline: ROC-AUC can look strong when most flagged customers are wrong.", []))
        out = A.turn(s, client, "churn with 3% positives: PR-AUC or ROC-AUC for my manager?")
        self.assertIn("Report PR-AUC", out["text"])
        self.assertNotIn("0.91", out["text"])
        self.assertEqual(len(client.seen), 3)

    def test_the_rewrite_names_the_numbers_allowed(self):
        self.assertIn("exactly as the tools returned", A.REWRITE)
        self.assertTrue(A.has_statement("Report PR-AUC. Does that cover it?"))
        self.assertFalse(A.has_statement("Does that cover what you need, or would you like help wording it?"))


class LongAnswerCostTests(unittest.TestCase):
    """2026-11-01 (CEX-026 r-13): a data scientist asked for a full script; the number check withheld every sentence holding a
    setting nobody gave and each withheld answer was written again at full length (five of fourteen calls, EUR 0.73, no code).
    The check is unchanged; a message is capped at SHOWN_WORDS and a long reply that loses sentences is not written again."""

    LONG = " ".join(f"Step {w} explains one idea about resampling in plain words without any figure." for w in
                    ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
                     "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twentyone", "twentytwo", "twentythree", "twentyfour"))

    def test_a_message_over_the_cap_is_cut_at_a_sentence_with_a_note_and_no_rewrite(self):
        self.assertGreater(A.word_count(self.LONG), A.SHOWN_WORDS)
        client = Script((self.LONG, []))
        out = A.turn(state(table=False), client, "explain resampling for fraud models")
        self.assertEqual(len(client.seen), 1)  # nothing sent back
        self.assertLessEqual(A.word_count(out["text"].replace(A.CUT_NOTE, "")), A.SHOWN_WORDS)
        self.assertTrue(out["text"].startswith("Step one explains"))
        self.assertIn(A.CUT_NOTE, out["text"])
        self.assertTrue(out["text"].replace(A.CUT_NOTE, "").strip().endswith("."))  # cut at a sentence

    def test_a_long_reply_that_loses_sentences_is_not_written_again_and_never_shows_the_number(self):
        body = self.LONG[: self.LONG.index("Step eleven")]
        long_reply = body + " Use 300 rounds and 5 folds for the run. " + body
        self.assertGreater(A.word_count(long_reply), A.REWRITE_WORDS)
        client = Script((long_reply, []), ("This second call must never be made.", []))
        out = A.turn(state(table=False), client, "explain resampling for fraud models")
        self.assertEqual(len(client.seen), 1)  # no rewrite: the saving
        self.assertNotIn("300 rounds", out["text"])  # the check still withholds the number nobody gave
        self.assertNotIn("5 folds", out["text"])
        self.assertIn("Step one explains", out["text"])  # what survived reaches the person
        self.assertIn(A.WITHHELD_NOTE, out["text"])

    def test_a_long_reply_with_nothing_left_gets_one_short_words_only_rewrite_not_the_long_ones(self):
        """The review: a script with a number on every line left only 'tell me more'; one short rewrite gives the person words."""
        long_reply = " ".join(f"Use {n}00 rounds in setting {n}." for n in range(1, 40))
        client = Script((long_reply, []), ("First resample inside each training fold, then fit, then score on data it has not seen.", []),
                        ("never asked", []))
        out = A.turn(state(table=False), client, "give me the settings")
        self.assertEqual(len(client.seen), 2)  # one rewrite, never the strict second
        self.assertIn("at most 120 words", json.dumps(client.seen[1]))
        self.assertIn("resample inside each training fold", out["text"])
        self.assertNotIn("00 rounds", out["text"])

    def test_a_short_rewrite_that_is_cut_again_ends_in_code_s_line_after_one_call_only(self):
        long_reply = " ".join(f"Use {n}00 rounds in setting {n}." for n in range(1, 40))
        client = Script((long_reply, []), ("Use 300 rounds and 5 folds.", []), ("never asked", []))
        out = A.turn(state(table=False), client, "give me the settings")
        self.assertEqual(len(client.seen), 2)
        self.assertIn("only numbers that come from you", out["text"])
        self.assertNotIn("300 rounds", out["text"])

    def test_a_long_reply_whose_survivors_are_filler_is_rewritten_once_short(self):
        filler = "Sure, happy to help with that. " + " ".join(f"Set value {n} to {n}00 here." for n in range(1, 35))
        self.assertGreater(A.word_count(filler), A.REWRITE_WORDS)
        client = Script((filler, []), ("Use class weights first; resampling comes second and only inside each fold.", []))
        out = A.turn(state(table=False), client, "which settings should I use")
        self.assertEqual(len(client.seen), 2)
        self.assertIn("class weights first", out["text"])

    def test_a_script_without_sentence_ends_is_cut_at_a_line_and_its_fence_closed(self):
        script = "Here is the script. ```python\n" + "\n".join(f"step_{n} = run(stage_{n})" for n in range(1, 120)) + "\n```"
        self.assertGreater(A.word_count(script), A.SHOWN_WORDS)
        out, cut = A.shortened(script)
        self.assertTrue(cut)
        self.assertLess(A.word_count(out), A.word_count(script))
        self.assertLessEqual(A.word_count(out), A.SHOWN_WORDS + 5)
        self.assertTrue(out.startswith("Here is the script."))
        self.assertIn("step_1 = run(stage_1)", out)  # the code is not dropped after being announced
        self.assertEqual(out.count("```") % 2, 0)

    def test_a_block_that_cannot_be_made_shorter_is_not_marked_cut(self):
        one_line = "word " * 400
        self.assertEqual(A.shortened(one_line), (one_line.strip(), False))

    def test_the_cap_holds_after_a_rewrite_and_the_note_follows_the_text_shown(self):
        client = Script(("Only about 7 in 50 customers leave, so lead with PR-AUC.", []), (self.LONG, []))
        out = A.turn(state(table=False), client, "churn with 3% positives: which metric for my manager")
        self.assertLessEqual(A.word_count(out["text"].replace(A.CUT_NOTE, "")), A.SHOWN_WORDS + 5)
        self.assertIn(A.CUT_NOTE, out["text"])
        client = Script(("Only about 7 in 50 customers leave, so lead with PR-AUC.", []), ("Few customers leave, so lead with PR-AUC.", []))
        out = A.turn(state(table=False), client, "churn with 3% positives: which metric for my manager")
        self.assertNotIn(A.CUT_NOTE, out["text"])  # the first draft was short: no note

    def test_a_note_the_agent_echoes_from_its_history_is_not_shown_twice(self):
        client = Script((f"Here is the short answer. {A.CUT_NOTE}", []))
        out = A.turn(state(table=False), client, "explain resampling")
        self.assertEqual(out["text"].count("short version"), 0)
        self.assertIn("Here is the short answer.", out["text"])

    def test_a_short_reply_still_gets_its_rewrite_as_before(self):
        client = Script(("Only about 7 in 50 customers leave, so lead with PR-AUC.", []), ("Few customers leave, so lead with PR-AUC.", []))
        out = A.turn(state(table=False), client, "churn with 3% positives: which metric for my manager")
        self.assertEqual(out["text"], "Few customers leave, so lead with PR-AUC.")
        self.assertEqual(len(client.seen), 2)

    def test_shortened_keeps_the_first_sentence_and_closes_an_open_code_fence(self):
        self.assertEqual(A.shortened("Short one."), ("Short one.", False))
        text, cut = A.shortened("First sentence is here. " + "```python\nx = 1.\n" + "word " * 30 + ". More.", limit=8)
        self.assertTrue(cut)
        self.assertTrue(text.startswith("First sentence is here."))
        self.assertEqual(text.count("```") % 2, 0)


class RouteDriftTests(unittest.TestCase):
    """2026-11-01 (sealed v2: the lead agent reached the right route for 15 of 19, Claude alone 19; it talked 4 to 8 turns and
    drifted into advice): advice no longer opens before the case is ready when there is data work; after three turns with
    the goal and the data known it must propose or start; a cheap second reading of the person's words checks a start once."""

    def test_advice_before_the_case_is_ready_is_refused_when_a_table_is_shared(self):
        s = state()
        client = Script(("", [("start", {"track": "advice", "tool": "cited_answer"})]), ("", [("ask", {"question": "Who will use the list?"})]))
        A.turn(s, client, "We lose customers. Can you help?")
        refused = [m["content"] for m in client.seen[1] if m.get("role") == "tool"]
        self.assertIn("data work", refused[0])
        self.assertIsNone(s.case.route)

    def test_advice_with_no_table_and_a_plain_question_still_starts(self):
        s = state(table=False)
        out = A.turn(s, Script(("Report PR-AUC.", [("start", {"track": "advice", "tool": "cited_answer"})])), "quick q: PR-AUC or ROC-AUC? no data work, just the answer")
        self.assertEqual(s.case.route.track, "advice")
        self.assertIn("PR-AUC", out["text"])

    def test_after_three_turns_with_goal_and_data_known_asking_is_refused_and_starting_allowed(self):
        s = state()
        for i, w in enumerate(("We want to keep members.", "The table is all our members.", "Mostly the front desk.")):
            A.turn(s, Script(("", [("ask", {"question": "And what else?"})])), w)
            if i == 0:
                s.case = C.apply(s.case, "goal", C.CaseFact(value="keep members", status="said", quote="keep members"), by="agent", why="t")
        client = Script(("", [("ask", {"question": "Anything else?"})]), ("Starting a ranked list of members to call.", [("start", {"track": "prediction", "tool": "ml_project"})]))
        out = A.turn(s, client, "That is all I know.")
        refused = [m["content"] for m in client.seen[1] if m.get("role") == "tool"]
        self.assertIn("propose or start", refused[0])
        self.assertEqual(s.case.route.track, "prediction")

    def test_a_start_the_second_reading_disagrees_with_is_refused_once(self):
        s = state()
        s.case = C.hear(s.case, "every Monday we want a weekly view of cancellations by destination")
        s.budget_reached = True  # the turn budget lets the agent start on its best reading
        s.route_client = Script(("reporting", []))
        first = A._start(s, "analysis", "query")
        self.assertIn("reporting", first.get("error", ""))
        second = A._start(s, "analysis", "query")  # the agent may keep its pick: the check asks once
        self.assertIn("started", second)

    def test_the_route_check_sees_only_the_persons_words_and_column_names(self):
        s = state()
        s.case = C.hear(s.case, "we want a weekly dashboard of cancellations")
        client = Script(("reporting", []))
        self.assertEqual(A.route_reading(client, s), "reporting")
        sent = json.dumps(client.seen[0])
        self.assertIn("weekly dashboard", sent)
        self.assertIn("churned_next_month", sent)


class Counting(Script):
    def __init__(self, reading):
        super().__init__()
        self.reading, self.calls = reading, 0

    def complete(self, messages, tools=None, *args, **kwargs):
        self.calls += 1
        return {"content": self.reading, "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1}}


class RouteStepOneTests(unittest.TestCase):
    """The plan of 2026-11-01, step 1 (route): the kind of help the person's own words name is offered; the cheap second reading
    checks options as well as a start, read again only on new words; each doubt is told once, across turns; a clear person
    may be proposed to at once; Home gets the route check the exam measured."""

    def test_the_words_name_a_kind_of_help_only_when_they_ask_for_it(self):
        named = C.named_tracks
        self.assertEqual(set(named(["which bookings are going to get cancelled next month"])), {"prediction"})
        self.assertEqual(set(named(["send me the customer ids on the premium plan with 3 tickets"])), {"analysis"})
        self.assertEqual(set(named(["i want to drop the worst 3 destinations"])), {"business"})
        self.assertEqual(set(named(["people say AI can predict no-shows. is that real or hype?"])), {"advice"})
        for words in ("I need a churn prediction for the board", "should I report PR-AUC or ROC-AUC to my manager",
                      "we lose customers every month", "is an AI demand forecast needed for my bakeries"):
            self.assertNotIn("prediction", named([words]), words)
            self.assertNotIn("business", named([words]), words)
            self.assertNotIn("reporting", named([words]), words)

    def test_options_missing_the_named_kind_of_help_are_refused_once_with_the_persons_words(self):
        s = state()
        s.case = C.hear(s.case, "send me the customers who churned, with their ids")
        s.case = C.apply(s.case, "goal", C.CaseFact(value="the customers who churned", status="said", quote="customers who churned"), by="agent", why="t")
        two = [dict(o) for o in OPTIONS if o["track"] == "prediction"] + [dict(OPTIONS[0], title="Groups", tool="segmentation_project")]
        first = A._propose(s, two)
        self.assertIn("analysis", first["error"])
        self.assertIn("with their ids", first["error"])
        self.assertEqual(s.case.options, [])
        self.assertIn("shown", A._propose(s, two))  # told once: the agent may keep its options

    def test_the_second_reading_checks_options_and_is_read_again_only_on_new_words(self):
        s = state()
        s.case = C.hear(s.case, "every Monday the team needs the cancellations by plan")
        s.case = C.apply(s.case, "goal", C.CaseFact(value="every Monday the cancellations", status="said", quote="every Monday"), by="agent", why="t")
        s.route_client = Counting("reporting")
        s.budget_reached = True
        out = A._propose(s, OPTIONS)
        self.assertIn("second reading", out["error"])
        self.assertIn("reporting", out["error"])
        self.assertIn("shown", A._propose(s, OPTIONS))
        self.assertEqual(s.route_client.calls, 1)  # no new words: no second call
        self.assertIn("error", A._start(s, "analysis", "query"))  # a start is its own move: told once too
        self.assertEqual(s.route_client.calls, 1)

    def test_a_told_doubt_is_kept_across_turns_as_home_rebuilds_its_state(self):
        from dclab_rnd.agents import case_session

        s = state()
        s.route_told = ["start:reporting"]
        doc = case_session.snapshot(s)
        again = A.CaseState(case=s.case)
        case_session.restore(again, doc)
        self.assertEqual(again.route_told, ["start:reporting"])

    def test_a_clear_person_may_be_proposed_to_at_once(self):
        case = C.hear(C.new("c"), "which members are likely to cancel next month? my team calls 300 a month")
        self.assertFalse(C.readiness(case).may_propose)  # the goal is not recorded yet
        case = C.apply(case, "goal", C.CaseFact(value="members likely to cancel", status="said", quote="likely to cancel"), by="agent", why="t")
        r = C.readiness(case)
        self.assertTrue(r.may_propose)
        self.assertFalse(r.ready)
        self.assertTrue(any("named the help" in x for x in r.reasons))
        vague = C.apply(C.hear(C.new("c"), "we lose customers, help"), "goal", C.CaseFact(value="lose customers less", status="said", quote="lose customers"), by="agent", why="t")
        self.assertFalse(C.readiness(vague).may_propose)

    def test_a_clear_person_is_proposed_to_but_never_started_on_before_the_case_is_ready(self):
        """The second review of step 1: "send me the customers who churned ... call list by friday" is both a go-ahead and a
        named analysis on a real-prediction trap; an early start locked it."""
        s = state()
        s.case = C.hear(s.case, "send me the customers who churned in the last 3 months, my retention team needs a call list by friday")
        s.case = C.apply(s.case, "goal", C.CaseFact(value="the customers who churned", status="said", quote="customers who churned"), by="agent", why="t")
        s.go_ahead = True
        self.assertTrue(C.readiness(s.case).may_propose)
        self.assertIn("error", A._start(s, "analysis", "query"))
        self.assertIsNone(s.case.route)

    def test_advice_words_do_not_silence_the_second_reading(self):
        """The second review of step 1: "is that a good idea?" named advice only, and the reading that would catch the data work
        behind it was never asked."""
        s = state()
        s.case = C.hear(s.case, "Is that a good idea? we keep losing members and I want to call them before it is too late")
        s.budget_reached = True
        s.route_client = Counting("prediction")
        out = A._start(s, "advice", "cited_answer")
        self.assertIn("prediction", out.get("error", ""))
        self.assertEqual(s.route_client.calls, 1)
        case = C.apply(s.case, "goal", C.CaseFact(value="call them before it is too late", status="said", quote="call them before"), by="agent", why="t")
        self.assertFalse(C.readiness(case).may_propose)  # advice words alone never make a person clear

    def test_a_short_table_value_does_not_switch_the_route_check_off(self):
        s = state()
        s.group_names = {"No", "Gold Plus"}
        s.case = C.hear(s.case, "no idea, which members will cancel next month?")
        client = Counting("prediction")
        self.assertEqual(A.route_reading(client, s), "prediction")
        s.case = C.hear(s.case, "what about the gold plus ones")
        self.assertIsNone(A.route_reading(Counting("prediction"), s))

    def test_method_questions_and_relative_clauses_name_no_kind_of_help(self):
        """The review of step 1: these were read as business or prediction."""
        for words in ("Should I drop the duplicate rows before training?", "Should I raise the threshold to 0.7",
                      "What should I do about leakage in my features", "Why is the model accuracy so high?", "What do I do with missing values before cross-validation?",
                      "I have to decide whether to use SMOTE or class weights", "My manager, who will present it to the board, wants last month's churn by plan",
                      "which threshold will work best", "the customers who will receive the email", "before they leave town"):
            self.assertEqual(C.named_tracks([words]), {}, words)
        for words, track in (("which customers will churn next quarter", "prediction"), ("who is likely to cancel", "prediction"),
                             ("should we close the Leeds store", "business"), ("what do i do about falling sales", "business")):
            self.assertIn(track, C.named_tracks([words]), words)

    def test_two_kinds_of_help_named_never_start_before_the_case_is_ready(self):
        """The review of step 1: "a dashboard of the bookings going to get cancelled, go ahead" started a dashboard (a trap)."""
        s = state()
        s.case = C.hear(s.case, "can you make me a dashboard of the bookings that are going to get cancelled? go ahead")
        s.case = C.apply(s.case, "goal", C.CaseFact(value="the bookings going to get cancelled", status="said", quote="going to get cancelled"), by="agent", why="t")
        s.go_ahead = True
        self.assertIn("error", A._start(s, "reporting", "dashboard"))
        self.assertIsNone(s.case.route)

    def test_the_second_reading_never_pushes_another_kind_of_help_when_the_named_one_is_offered(self):
        """The review of step 1 (r-10, a rows request): named analysis offered, the reading said prediction, the options refused."""
        s = state()
        s.case = C.hear(s.case, "send me the customer ids on the premium plan with 3+ tickets")
        s.case = C.apply(s.case, "goal", C.CaseFact(value="the customer ids on the premium plan", status="said", quote="customer ids"), by="agent", why="t")
        s.route_client = Counting("prediction")
        rows = [dict(OPTIONS[1]), dict(OPTIONS[0], title="A weekly view", track="reporting", tool="dashboard", columns=[])]
        self.assertIn("shown", A._propose(s, rows))
        self.assertEqual(s.route_client.calls, 0)  # their own words outrank the second reading: no call

    def test_a_doubt_is_told_once_a_move_for_what_was_said_however_the_options_change(self):
        s = state()
        s.case = C.hear(s.case, "every Monday the team needs the cancellations by plan")
        s.route_client = Counting("prediction")
        s.budget_reached = True
        s.case = C.apply(s.case, "goal", C.CaseFact(value="every Monday the cancellations", status="said", quote="every Monday"), by="agent", why="t")
        self.assertIn("error", A._propose(s, [dict(OPTIONS[1]), dict(OPTIONS[0], title="Clean", track="preparation", tool="recipe", columns=[])]))
        self.assertIn("shown", A._propose(s, [dict(OPTIONS[1]), dict(OPTIONS[0], title="Groups", tool="segmentation_project", columns=[])]))

    def test_the_route_check_is_not_asked_when_the_words_repeat_a_shown_table_value(self):
        """The review of step 1: route_check holds no cell value, but a repeated group name reached its usage log."""
        s = state()
        s.group_names = {"Gold Plus"}
        s.case = C.hear(s.case, "the Gold Plus customers, which of them will churn?")
        client = Counting("prediction")
        self.assertIsNone(A.route_reading(client, s))
        self.assertEqual(client.calls, 0)

    def test_home_gives_the_lead_agent_the_route_check(self):
        from dclab_rnd.draft.case_home import CaseHome
        from dclab_rnd.draft.store import DraftStore

        asked = []

        class Models:
            def client(self, purpose, draft_id=None, **k):
                asked.append((purpose, k.get("model")))
                return None

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        store = DraftStore(folder / "drafts")
        d = store.create("predict which members cancel")
        CaseHome(store, Models())._state(store.get(d["id"]))
        self.assertIn(("route_check", A.ROUTE_MODEL), asked)


class RouteDriftReviewTests(unittest.TestCase):
    """The review of 2026-11-01: a real advice question with a table shared still gets advice; the turn budget switches off
    once work has started."""

    def test_a_yes_or_no_method_question_with_a_table_still_gets_advice(self):
        s = state()
        out = A.turn(s, Script(("Not on its own: tenure counts up to cancellation, so check when it was measured.", [("start", {"track": "advice", "tool": "cited_answer"})])),
                     "is tenure_months leaky for churned_next_month? yes or no")
        self.assertEqual(s.case.route.track, "advice")

    def test_the_turn_budget_switches_off_once_work_has_started(self):
        s = state()
        for i, w in enumerate(("We want to keep members.", "The table is all our members.", "Mostly the front desk.")):
            A.turn(s, Script(("", [("ask", {"question": "And?"})])), w)
            if i == 0:
                s.case = C.apply(s.case, "goal", C.CaseFact(value="keep members", status="said", quote="keep members"), by="agent", why="t")
        A.turn(s, Script(("Starting.", [("start", {"track": "prediction", "tool": "ml_project"})])), "go")
        A.turn(s, Script(("", [("ask", {"question": "Shall the list hold 200 names or fewer?"})])), "Can we change the list size?")
        self.assertFalse(s.budget_reached)
        self.assertTrue(s.asking)
