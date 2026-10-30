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


OPTIONS = [{"title": "Rank customers by risk", "what": "a weekly ranked list", "track": "prediction", "tool": "ml_project", "columns": ["churned_next_month"]},
           {"title": "Compare churn by plan", "what": "churn per plan", "track": "analysis", "tool": "query", "columns": ["plan"]}]


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
        self.assertEqual(docs[0]["thinking"], [{"text": "the decision is still unknown"}])

    def test_the_persons_words_stay_in_the_state_when_old_turns_are_dropped(self):  # the review: quotes need their words
        case = C.new("c", ["first message with the goal in it", "second"])
        built = ctx.build(case, [{"role": "user", "content": "x " * 40_000}] * 5, budget=2000)
        self.assertIn("first message with the goal in it", built["state"])
