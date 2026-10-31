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
        self.assertIn("nothing is built yet", out["next"])

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

        def client():
            return Script(("Tell me more about what you need.", [("ask", {"question": "What would you decide with the result?"})]),
                          ("", [("propose", {"options": [dict(o, columns=[]) for o in OPTIONS]})]), ("Here are two ways.", []),
                          ("Good, I'll start.", []))

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        for p in X.load("dev"):
            conv = X.play(p, exam_arms.CaseArm(client()), Person())
            self.assertEqual(conv["route"], "prediction", p["id"])
            self.assertEqual(conv["ended"], "routed", p["id"])
            self.assertIn("case_file", conv["notes"])


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
