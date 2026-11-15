"""Stage 2 of the plan of 2026-11-15 (more room for the model): exact fixes to the lead agent's rules, no redesign.

Each test is one row of the stage's table, found in the day's live runs and in the review of the harness:

2.1 an order to build a model starts the prediction it names (it was refused as "not ready" while the build said "start first");
2.2 a go-ahead never leaves only refusals (start said "not ready", propose "not yet, ask", ask "do not ask");
2.3 in every state a move that decides what the person gets is open;
2.4 a build's yes is read by code from the person's message (a quote one word off was refused like no yes at all);
2.5 a quote copied with its quote marks is the person's; a refusal says which rule failed;
2.6 several facts are recorded in one call;
2.7 working notes beside a tool call are not the message; an empty first reply is asked again;
2.8 the number check drops the sentence or the line that holds the number, never the list around it;
2.9 the kind of work follows the tool; a refusal names the valid pairs;
2.10 a message that only says go ahead does not raise the second reading again.
The team walk's fallback (2.11) is tested where its failing test lives: tests/test_chat_agents.py.
"""

import copy
import itertools
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import case as C  # noqa: E402
from dclab_rnd.agents import case_agent as A  # noqa: E402

VIEW = [{"name": "customer_id", "kind": "text", "noticed": ["identifier"]}, {"name": "plan", "kind": "category", "noticed": []},
        {"name": "churned_next_month", "kind": "numeric", "noticed": ["outcome-like name"]}]
FULL = {"gets": "a list your team can act on", "tradeoff": "needs past outcomes", "effort": "a day"}
OPTIONS = [{"title": "Rank customers by risk", "what": "a weekly ranked list", "track": "prediction", "tool": "ml_project", "columns": ["churned_next_month"], **FULL},
           {"title": "Compare churn by plan", "what": "churn per plan", "track": "analysis", "tool": "query", "columns": ["plan"], **FULL}]
BRIEF = ("I lead retention at a subscription company and I want to know which customers will leave next month, so my team can "
         "call them. The table has one row per customer with their plan and whether they churned the next month.")


class Script:
    """A model client replying in order: its words and its case tool calls (name, arguments)."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def complete(self, messages, tools=None, *args, **kwargs):
        self.seen.append(json.loads(json.dumps(messages)))
        text, calls = self.replies.pop(0) if self.replies else ("", [])
        calls = [{"id": f"c{len(self.seen)}-{i}", "name": f"case_{n}", "arguments": a} for i, (n, a) in enumerate(calls)]
        msg = {"role": "assistant", "content": text,
               **({"tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]} if calls else {})}
        return {"content": text, "tool_calls": calls, "assistant_message": msg, "usage": {"input_tokens": 1, "output_tokens": 1}}


def state(table=True):
    return A.CaseState(case=C.new("case-s2"), table=A.table_of("customers.csv", VIEW, 1200) if table else None)


def said(s, *messages, agent=None):
    """The person's messages heard as a turn hears them (the flags of the latest message), with the agent's message between."""
    for i, words in enumerate(messages):
        if i and agent:
            s.turns.append({"role": "assistant", "content": agent})
        A.heard(s, words)
    return s


def ready(s):
    """The goal, the decision and the user noted from BRIEF, as a lead agent would: the case is ready."""
    for name, value, quote in (("goal", "know which customers will leave", "which customers will leave next month"),
                               ("decision", "my team can call them", "so my team can call them"), ("user", "my team", "my team can call")):
        s.case = C.apply(s.case, name, C.CaseFact(value=value, status="said", quote=quote), by="agent", why="noted")
    return s


def started(s, track="prediction", tool="ml_project"):
    """Work started on a ready case (a start is refused on any other, unless the person chose or ordered it)."""
    if BRIEF in s.case.words and not C.readiness(s.case).ready:
        ready(s)
    s.case = C.apply(s.case, "route", C.CaseRoute(track=track, tool=tool, by="agent"), by="agent", why="work started")
    return s


class BuildYesTests(unittest.TestCase):
    def test_the_yes_is_read_from_their_message_whatever_the_quote(self):
        for quote in (None, "Yes!! please build it...", "yes please build it now", "the person agreed"):
            s = started(said(state(), BRIEF, "yes please build it now"))
            out = A._request_build(s, quote) if quote else A._request_build(s)
            self.assertEqual(out.get("recorded"), "ml_project", (quote, out))
            self.assertEqual(s.case.build.quote, "yes please build it now")  # their own words are the record of their yes

    def test_a_pick_is_not_a_yes_and_the_refusal_says_so(self):
        s = started(said(state(), BRIEF, "I pick: Rank customers by risk"))
        error = A._request_build(s, "I pick: Rank customers by risk")["error"]
        self.assertIn("a pick is not a yes", error)
        self.assertIn("ask once", error)
        self.assertIsNone(s.case.build)

    def test_no_yes_is_refused_as_no_yes_and_a_no_stays_a_no(self):
        for words in ("Tell me more about how it works first.", "No, please don't build it yet, I want to think"):
            s = started(said(state(), BRIEF, words))
            error = A._request_build(s)["error"]
            self.assertIn("holds no yes to building", error, words)
            self.assertIsNone(s.case.build)

    def test_a_bare_yes_answers_the_question_it_was_asked(self):
        s = started(said(state(), BRIEF, "yes", agent="Which plan do most of your customers have?"))
        self.assertIn("not about building", A._request_build(s)["error"])
        s = started(said(state(), BRIEF, "yes", agent="Shall I build the model on this table?"))
        self.assertEqual(A._request_build(s)["recorded"], "ml_project")

    def test_proceed_after_the_question_about_building_is_their_yes(self):
        s = started(said(state(), BRIEF, "Proceed.", agent="Shall I build the model on this table?"))
        self.assertTrue(s.go_ahead)
        self.assertEqual(A._request_build(s)["recorded"], "ml_project")
        s = started(said(state(), BRIEF, "Proceed.", agent="Here is what I understood so far."))
        self.assertIn("holds no yes to building", A._request_build(s)["error"])  # a go-ahead to something else is not a yes to a build

    def test_a_go_ahead_about_something_else_is_never_a_yes_to_building(self):
        """The review of stage 2: with the build read as "any go-ahead after any question that names a model", each of these
        recorded a build. Only a message that is nothing but a go-ahead answers a question that is about building."""
        asked = "Shall I build the model now, or would you like to see the churn rate per plan first?"
        for agent, words in ((asked, "Show me the churn rate per plan first."), ("Shall I build the model on this table?", "proceed with the analysis only, no project"),
                             ("Shall I build the model on this table?", "send me the list of plans"), ("Shall I build the model on this table?", "you decide"),
                             ("I started the prediction project. Which column says a customer left?", "Proceed."),
                             ("Should I start with the churn per plan?", "run it"), (asked, "Proceed.")):
            s = started(said(state(), BRIEF, words, agent=agent))
            out = A._request_build(s)
            self.assertIn("error", out, (agent, words))
            self.assertIsNone(s.case.build, (agent, words))
        for words in ("Proceed.", "ok, go ahead", "run it please", "Go ahead."):
            s = started(said(state(), BRIEF, words, agent="I can rank your customers by risk. Shall I build the model on this table?"))
            self.assertEqual(A._request_build(s).get("recorded"), "ml_project", words)

    def test_the_outcome_is_still_never_guessed(self):
        s = started(said(state(), BRIEF, "yes please build it now"))
        self.assertIn("has not named", A._request_build(s, None, "customer_id")["error"])


class DeadEndTests(unittest.TestCase):
    def test_an_order_to_build_starts_the_prediction_it_names(self):
        s = said(state(), "Build the model")
        self.assertFalse(C.readiness(s.case).ready)
        self.assertEqual(A._start(s, "prediction", "ml_project").get("started"), "prediction/ml_project")
        out = A._request_build(s)  # and their order is their yes: the same turn requests the build
        self.assertEqual(out.get("recorded"), "ml_project", out)
        slow = said(state(), "I won't build a model until I understand my data")
        self.assertIn("not ready", A._start(slow, "prediction", "ml_project")["error"])  # no order: the case must be ready

    def test_a_go_ahead_opens_the_options_with_what_is_open_as_assumptions(self):
        s = said(state(), "Something about our customers.", "Go ahead with whatever you think is best.")
        self.assertTrue(s.go_ahead)
        self.assertIn("not ready", A._start(copy.deepcopy(s), "prediction", "ml_project")["error"])
        self.assertIn("Propose now", A._start(copy.deepcopy(s), "prediction", "ml_project")["error"])  # the refusal names the open move
        self.assertIn("shown", A._propose(s, OPTIONS))
        waiting = said(state(), "Something about our customers.")
        self.assertIn("not yet", A._propose(waiting, OPTIONS)["error"])  # without a go-ahead the questions still come first

    def test_the_one_question_about_building_is_allowed_after_a_go_ahead(self):
        s = started(said(state(), BRIEF, "Proceed.", agent="Here is what I understood so far."))
        self.assertIn("holds no yes", A._request_build(copy.deepcopy(s))["error"])
        self.assertEqual(A._ask(s, "Shall I build the model on this table now?"), {"asked": True})
        again = started(said(state(), BRIEF, "Proceed.", agent="Shall I build the model on this table?"))
        self.assertIn("do not ask them to confirm again", A._ask(again, "Shall I build the model now?")["error"])  # asked once: their go-ahead answered it
        other = started(said(state(), BRIEF, "Proceed.", agent="Here is what I understood so far."))
        self.assertIn("do not ask them to confirm again", A._ask(other, "Does that sound good to you?")["error"])


class AlwaysAWayForwardTests(unittest.TestCase):
    """2.3: in every state reached by these words, facts, chips and routes, a move that decides what the person gets is
    allowed. After a go-ahead a question does not count (the person said to act), except the one question about building."""

    WORDS = ("Build the model", "Proceed.", "Go ahead with whatever you think is best.", "yes", BRIEF, "Help me with my customers.",
             "What was our churn per plan last month?", "I pick: Rank customers by risk", "Is it a good idea to use a model here?")
    QUESTIONS = ("Which column records that a customer left?", "Shall I build the model on this table now?")

    def moves(self, s):
        yield "propose", lambda x: A._propose(x, OPTIONS)
        yield "start prediction", lambda x: A._start(x, "prediction", "ml_project")
        yield "start analysis", lambda x: A._start(x, "analysis", "query")
        yield "start advice", lambda x: A._start(x, "advice", "cited_answer")
        yield "request_build", lambda x: A._request_build(x)
        yield "ask about building", lambda x: A._ask(x, self.QUESTIONS[1])
        if s.table is None:  # what every start says with no table: "ask them to share the table it runs on"
            yield "ask for the table", lambda x: A._ask(x, "Could you share the table this should run on?")
        if not s.go_ahead:
            yield "ask", lambda x: A._ask(x, self.QUESTIONS[0])

    def test_no_state_refuses_every_deciding_move(self):
        stuck = []
        for first, table, chip, facts, route, last in itertools.product(
                (BRIEF, "Something about our customers."), (True, False), (None, "prediction", "advice"), ("none", "all"), (None, "prediction"), self.WORDS):
            s = state(table)
            s.chip = chip
            said(s, first)
            if facts == "all":
                own = " ".join(first.rstrip(".").split()[:4])  # a run of their own words, as value and as quote
                for name in ("goal", "decision", "user"):
                    s.case = C.apply(s.case, name, C.CaseFact(value=own, status="said", quote=own), by="agent", why="noted")
            if route:
                if not table:
                    continue  # a prediction never started without a table
                started(s)
            said(s, last, agent="Here is what I understood so far.")
            allowed = [name for name, move in self.moves(s) if "error" not in move(copy.deepcopy(s))]
            if not allowed:
                stuck.append({"first": first[:30], "table": table, "chip": chip, "facts": facts, "route": route, "last": last[:40]})
        self.assertEqual(stuck, [], f"{len(stuck)} states refuse every deciding move")


class QuoteTests(unittest.TestCase):
    WORDS = ["I lead retention and my team calls customers who are about to leave."]

    def test_a_quote_copied_with_its_marks_is_the_persons(self):
        for quote in ('"my team calls customers"', "“my team calls customers”", "'my team calls customers.'", "my team calls customers"):
            self.assertTrue(C.quote_ok(quote, "the retention team calls customers", self.WORDS), quote)
        self.assertFalse(C.quote_ok('"the board decides"', "the board decides", self.WORDS))  # still theirs, or refused

    def test_a_refusal_says_which_rule_failed(self):
        self.assertEqual(C.quote_problem("my team calls customers", "the team calls them", self.WORDS), None)
        self.assertEqual(C.quote_problem("the board decides", "the board decides", self.WORDS), "not_found")
        self.assertEqual(C.quote_problem("about to leave", "revenue", self.WORDS), "unrelated")
        self.assertEqual(C.quote_problem("leave", "customers leave", self.WORDS), "short")
        case = C.hear(C.new("q"), self.WORDS[0])
        unrelated = C.check_fact(C.CaseFact(value="revenue", status="said", quote="about to leave"), case)[0]
        self.assertIn("is the person's words, but shares no word with your value", unrelated)
        missing = C.check_fact(C.CaseFact(value="the board decides", status="said", quote="the board decides"), case)[0]
        self.assertIn("is not the person's words", missing)
        self.assertIn("without quote marks", missing)


class NoteTests(unittest.TestCase):
    def test_several_facts_are_recorded_in_one_call(self):
        s = said(state(), BRIEF)
        out = A._note(s, notes=[{"field": "goal", "value": "know which customers will leave", "status": "said", "quote": "which customers will leave next month"},
                                {"field": "user", "value": "my team calls them", "status": "said", "quote": "so my team can call them"},
                                {"field": "decision", "value": "the board decides", "status": "said", "quote": "the board decides"},
                                {"field": "assumptions", "value": "a call is made once a month", "status": "assumed"}])
        self.assertEqual(out["noted"], ["goal", "user", "assumptions"])
        self.assertIn("is not the person's words", out["refused"]["decision"])  # one refused fact does not stop the others
        self.assertTrue(C.known(s.case.goal) and C.known(s.case.user) and not C.known(s.case.decision))
        self.assertIn("error", A._note(s, notes=[]))
        self.assertIn("error", A._note(s, notes=[{"field": "goal", "value": "x", "status": "said", "quote": "never said this"}]))

    def test_one_fact_is_recorded_as_before(self):
        s = said(state(), BRIEF)
        self.assertEqual(A._note(s, "goal", "know which customers will leave", "said", "which customers will leave next month")["noted"], "goal")
        self.assertIn("error", A._note(s, "goal", "x", "guessed"))


class ReplyTests(unittest.TestCase):
    def test_working_notes_beside_a_tool_call_are_not_the_message(self):
        call = [{"id": "c", "type": "function", "function": {"name": "case_note", "arguments": "{}"}}]
        plan = ("The real question is who can still be kept, not who will leave. I would rank customers by risk, check the ranking on "
                "later months, and start with a small weekly list. Step one: fix what leaving means. Step two: build the list. "
                "Step three: check it against last quarter.")
        messages = [{"role": "assistant", "content": "I'll start by checking the table.", "tool_calls": call},
                    {"role": "tool", "content": "{}"},
                    {"role": "assistant", "content": "Now let me record the goal and the user.", "tool_calls": call},
                    {"role": "assistant", "content": plan, "tool_calls": call},  # a plan written beside the options is the message
                    {"role": "assistant", "content": "Which of these two fits you better?"}]
        self.assertEqual(A.spoken(messages), [plan, "Which of these two fits you better?"])
        self.assertEqual(A.spoken([{"role": "assistant", "content": "I'll send you the list on Friday."}]), ["I'll send you the list on Friday."])  # no call beside it: an answer

    def test_words_beside_options_or_a_question_are_for_the_person(self):
        """The review of stage 2: a short answer beside the options that began with "I'll" was dropped, and the person got the
        fallback line under their options."""
        def beside(tool):
            return [{"id": "c", "type": "function", "function": {"name": f"case_{tool}", "arguments": "{}"}}]

        answer = "I'll rank your customers by risk and send your team the list each Monday."
        self.assertEqual(A.spoken([{"role": "assistant", "content": answer, "tool_calls": beside("propose")}]), [answer])
        self.assertEqual(A.spoken([{"role": "assistant", "content": "Let me say what I understood: you want fewer cancellations.", "tool_calls": beside("ask")}]),
                         ["Let me say what I understood: you want fewer cancellations."])
        only = [{"role": "assistant", "content": "I'll check the table first.", "tool_calls": beside("query")}]
        self.assertEqual(A.spoken(only), ["I'll check the table first."])  # the turn's only words stay: never the fallback line instead

    def test_an_empty_first_reply_is_asked_again_with_its_tools(self):
        s = state()
        client = Script(("", []), ("", [("ask", {"question": "Which column records that a customer left?"})]))
        out = A.turn(s, client, BRIEF)
        self.assertIn("Which column records", out["text"])
        self.assertNotIn(A.FALLBACK, out["text"])  # HAB-011: a full brief was answered "could you tell me a little more"

    def test_a_rewrites_refusal_says_what_to_do(self):
        s = state()
        s.rewriting = True
        from dclab_rnd.agents import default_registry

        out = default_registry().call("case_propose", {"options": OPTIONS}, s)
        self.assertIn("write your message to the person now", out["error"])


class NumberCheckTests(unittest.TestCase):
    def test_one_number_costs_its_line_not_the_list(self):
        s = said(state(), BRIEF)
        text = ("Here is the plan:\n- Fix what leaving means for you\n- Expect about 20% of customers to be flagged\n- Check the list on later months\n\n"
                "Does that match what you need?")
        dropped = []
        kept = A.checked(text, s, dropped)
        self.assertEqual(kept, "Here is the plan:\n- Fix what leaving means for you\n- Check the list on later months\n\nDoes that match what you need?")
        self.assertEqual(len(dropped), 1)
        self.assertIn("20%", dropped[0])

    def test_a_sentence_inside_a_line_is_still_the_unit(self):
        s = said(state(), BRIEF)
        kept = A.checked("I would rank customers by risk. About 35% of them will leave. Then we check the ranking.", s, [])
        self.assertEqual(kept, "I would rank customers by risk. Then we check the ranking.")
        self.assertEqual(A.checked("1. Fix the outcome\n2. Build the list", s, []), "1. Fix the outcome\n2. Build the list")  # a step's number is no claim


class OptionTests(unittest.TestCase):
    def test_the_kind_of_work_follows_the_tool(self):
        s = said(state(), BRIEF, "show me options")
        wrong = [{**OPTIONS[0]}, {**OPTIONS[1], "track": "reporting"}]  # query belongs to analysis
        out = A._propose(s, wrong)
        self.assertEqual(out.get("shown"), ["Rank customers by risk", "Compare churn by plan"], out)
        self.assertEqual([o.track for o in s.case.options], ["prediction", "analysis"])

    def test_a_refusal_names_the_valid_pairs(self):
        s = said(state(), BRIEF, "show me options")
        self.assertIn("prediction/ml_project", A._propose(s, [{**OPTIONS[0], "tool": "model_builder"}, OPTIONS[1]])["error"])
        self.assertIn("analysis/query", A._start(s, "analysis", "dashboard")["error"])
        self.assertIn("reporting", A._start(s, "analysis", "dashboard")["error"])  # where the named tool belongs
        one = A._propose(s, [OPTIONS[0], {**OPTIONS[0], "title": "Another ranking"}])
        self.assertIn("start it (start) instead of proposing", one["error"])


class ChipTests(unittest.TestCase):
    def test_an_option_cannot_leave_the_persons_pick_by_naming_another_kind(self):
        """The review of stage 2: the kind of work was derived from the tool after the chip was checked, so an option that
        called itself reporting and named the model's tool was shown under a reporting pick."""
        s = state()
        s.chip = "reporting"
        said(s, "Give me a report of churn by plan.")
        sneaked = [{**OPTIONS[0], "track": "reporting"}, {"title": "A churn dashboard", "what": "churn by plan", "track": "reporting", "tool": "dashboard",
                                                          "columns": ["plan"], **FULL}]
        out = A._propose(s, sneaked)
        self.assertIn("offer only reporting", out["error"])
        self.assertEqual(s.case.options, [])


class SecondReadingTests(unittest.TestCase):
    def test_a_go_ahead_does_not_hide_a_doubt_that_was_never_told(self):
        """The review of stage 2: a report was asked for, the agent asked a question, the person said go ahead, and a
        prediction started with no word of their own request."""
        s = said(state(), "I need a monthly report of churn by plan for the board.", "go ahead", agent="Which month should it start from?")
        self.assertEqual(s.route_told, [])
        self.assertIn("reporting", A.route_doubt(s, "start", {"prediction"}) or "")

    def test_a_message_that_only_says_go_ahead_raises_no_new_doubt(self):
        class Second:
            def complete(self, *args, **kwargs):
                return {"content": "business", "tool_calls": [], "usage": {}}

        s = said(state(), "Help me understand why customers leave.")
        s.route_client = Second()
        self.assertIn("may call for business", A.route_doubt(s, "start", {"analysis"}))  # raised once, on what they said
        said(s, "Everything I know is in what I wrote. Choose what you judge best and go ahead.")
        self.assertTrue(s.go_ahead)
        self.assertIsNone(A.route_doubt(s, "start", {"analysis"}))  # their go-ahead says nothing new about the kind of help
        fresh = said(state(), "Help me understand why customers leave.")
        fresh.route_client = Second()
        self.assertIn("send the same start again", A.route_doubt(fresh, "start", {"analysis"}))


if __name__ == "__main__":
    unittest.main()
