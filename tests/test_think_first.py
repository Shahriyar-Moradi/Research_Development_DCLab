"""Stage 4 of the plan of 2026-11-16: think first, pick the kind of work last, behind a switch that is off by default.

- With the switch off the lead agent is sent the prompt, the first-turn note and the reply limit it had before.
- With it on: the think-first prompt (the problem, what needs care, the approach and the plan come before the kinds of
  work), a first-turn note that asks for that message and then the plan's first step in the same turn, room for it
  (700 words until the plan has its own card), and a long reply that loses sentences to the number check is written
  again at its own length. The number check itself is the same either way.
- The stage's own measurement: a held-out set of people (dev3) nobody who changes the lead agent reads, a second gate
  pinned before any of this changed (hab_gate_v2, judged on that set only), and the model alone held to 450 words.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import case as C  # noqa: E402
from dclab_rnd import prompts  # noqa: E402
from dclab_rnd.agent_eval import harness_ablation as H  # noqa: E402
from dclab_rnd.agent_eval.harness_ablation import arms, gate as G  # noqa: E402
from dclab_rnd.agents import case_agent as A  # noqa: E402

ON = {A.THINK_FIRST: "1"}
OFF = {A.THINK_FIRST: ""}
VIEW = [{"name": "customer_id", "kind": "text", "noticed": ["identifier"]}, {"name": "plan", "kind": "category", "noticed": []},
        {"name": "churned_next_month", "kind": "numeric", "noticed": ["outcome-like name"]}]
BRIEF = ("I lead retention at a subscription company and I want to know which customers will leave next month, so my team can "
         "call them. The table has one row per customer with their plan and whether they churned the next month.")
SENTENCE = "This step checks the list against what really happened in the months that came after it was made."


class Script:
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def complete(self, messages, tools=None, *args, **kwargs):
        self.seen.append(json.loads(json.dumps(messages)))
        text, calls = self.replies.pop(0) if self.replies else ("", [])
        calls = [{"id": f"c{len(self.seen)}-{i}", "name": f"case_{n}", "arguments": a} for i, (n, a) in enumerate(calls)]
        msg = {"role": "assistant", "content": text,
               **({"tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]} if calls else {})}
        return {"content": text, "tool_calls": calls, "assistant_message": msg, "usage": {"input_tokens": 1, "output_tokens": 1}}


def state():
    return A.CaseState(case=C.new("case-s4"), table=A.table_of("customers.csv", VIEW, 1200))


def words(n):
    """A reply of about ``n`` words, in plain sentences with no number in them."""
    return " ".join([SENTENCE] * (n // len(SENTENCE.split())))


class SwitchTests(unittest.TestCase):
    def test_off_by_default_and_the_old_prompt_is_sent(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(A.THINK_FIRST, None)
            self.assertFalse(A.think_first())
            self.assertEqual(A.prompt_name(), "case_agent")
            client = Script(("A short answer for you.", []))
            A.turn(state(), client, BRIEF)
            self.assertEqual(client.seen[0][0]["content"], prompts.text("case_agent"))
            self.assertNotIn("first full message", json.dumps(client.seen[0]))

    def test_on_sends_the_think_first_prompt_and_its_first_turn_note(self):
        with mock.patch.dict(os.environ, ON):
            self.assertEqual(A.prompt_name(), "case_agent_think_first")
            client = Script(("A short answer for you.", []))
            A.turn(state(), client, BRIEF)
            self.assertEqual(client.seen[0][0]["content"], prompts.text("case_agent_think_first"))
        for value in ("0", "", "off", "no"):
            with mock.patch.dict(os.environ, {A.THINK_FIRST: value}):
                self.assertFalse(A.think_first(), value)

    def test_the_first_turn_note_asks_for_the_message_then_the_first_step_in_the_same_turn(self):
        note = A.CLEAR_THINK_FIRST
        self.assertIn("what needs care only when something does", note)
        self.assertIn("give the plan in full", note)  # 2026-11-16: a clear request gets short first parts and a whole plan
        self.assertIn("in this same turn", note)  # think first is not ask first: the person does not wait a turn
        self.assertIn("Do not ask clarifying questions", note)
        self.assertNotIn("immediately propose", note)


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.text = prompts.text("case_agent_think_first")

    def test_the_plan_comes_before_the_kinds_of_work(self):
        plan, kinds = self.text.index("Your first full message"), self.text.index("How DCLab runs a step of your plan")
        self.assertLess(plan, kinds)
        self.assertLess(self.text.index("What needs care before any work"), self.text.index("The ways it could be solved"))
        self.assertLess(kinds, self.text.index("- prediction: a model predicting"))  # the list sits under "how a step is run"
        self.assertIn("Choose this last, from the step", self.text)

    def test_a_request_that_should_not_be_done_as_asked_is_not_started_as_asked(self):
        self.assertIn("A request that should not be done as asked is not started as asked.", self.text)
        self.assertIn("in this same turn", self.text)

    def test_it_differs_from_the_old_prompt_only_where_the_stage_says(self):
        old = prompts.text("case_agent")
        self.assertNotIn("Your first full message", old)
        for kept in ("Never ask for something they already told you", "A project is built only after the person says yes to building it",
                     "Never fabricate results", "never promise a number"):
            self.assertIn(kept, old)
            self.assertIn(kept, self.text)  # the rules that guard the person and the numbers are the same in both

    def test_it_never_asks_for_the_models_own_thinking_in_its_reply(self):
        from tests.test_prompt_audit import ASKS_REASONING

        for name in ("case_agent_think_first", "ablation_model_only_short"):
            self.assertIsNone(ASKS_REASONING.search(prompts.text(name)), name)


class ReplyTests(unittest.TestCase):
    def test_a_long_plan_is_shown_whole_only_with_the_switch_on(self):
        plan = words(600)
        with mock.patch.dict(os.environ, ON):
            out = A.turn(state(), Script((plan, [])), BRIEF)
            self.assertEqual(A.word_count(out["text"]), A.word_count(plan))
            self.assertNotIn(A.CUT_NOTE, out["text"])
        os.environ.pop(A.THINK_FIRST, None)
        out = A.turn(state(), Script((plan, [])), BRIEF)
        self.assertLessEqual(A.word_count(out["text"].replace(A.CUT_NOTE, "")), A.SHOWN_WORDS)
        self.assertIn(A.CUT_NOTE, out["text"])

    def test_even_then_a_message_has_a_limit(self):
        with mock.patch.dict(os.environ, ON):
            out = A.turn(state(), Script((words(1200), [])), BRIEF)
            self.assertLessEqual(A.word_count(out["text"].replace(A.CUT_NOTE, "")), A.SHOWN_WORDS_PLAN)

    def test_a_number_nobody_gave_is_removed_either_way(self):
        text = "I would rank customers by risk. About 37% of them will leave next month. Then we check the ranking on later months."
        for env in ({}, ON):
            with mock.patch.dict(os.environ, env):
                if not env:
                    os.environ.pop(A.THINK_FIRST, None)
                out = A.turn(state(), Script((text, []), ("I would rank customers by risk. Then we check the ranking on later months.", [])), BRIEF)
                self.assertNotIn("37%", out["text"], env)
                self.assertIn("rank customers by risk", out["text"], env)

    def test_a_long_reply_that_loses_most_of_itself_is_written_again_at_its_own_length(self):
        made_up = " ".join(f"Group {n} loses about {30 + n}% of its customers every single month of the year." for n in range(11, 31))
        self.assertGreater(A.word_count(made_up), A.REWRITE_WORDS)
        with mock.patch.dict(os.environ, ON):
            client = Script((made_up + " " + SENTENCE, []), (words(200), []))
            out = A.turn(state(), client, BRIEF)
            self.assertEqual(client.seen[1][-1]["content"], A.REWRITE_LONG)
            self.assertIn("about the same length", A.REWRITE_LONG)
            self.assertNotIn("%", out["text"])
        os.environ.pop(A.THINK_FIRST, None)
        client = Script((made_up + " " + SENTENCE, []), (words(100), []))
        A.turn(state(), client, BRIEF)
        self.assertEqual(client.seen[1][-1]["content"], A.REWRITE_SHORT)  # as before the switch


PLAN = ("**The plan**\n1. Split the customers by signup date. Needs: the table. Checked by: the two parts do not overlap.\n"
        "2. Train a simple model and a baseline. Needs: the split. Checked by: the baseline is beaten on the held-back part.\n"
        "3. Rank by risk times spend. Needs: the scores. Checked by: the top of the list holds more of the leavers.\n"
        "Stop when the held-back part shows no gain over the baseline.\nIf a step fails, keep the baseline list and say why.")


class PlanKeptTests(unittest.TestCase):
    """2026-11-16: with the switch on, the plan is the last part of the first full message, so the word limit cut the plan
    first (two of sixteen plans scored lower than with the switch off). The plan is kept whole, and kept for later turns."""

    def test_the_plan_is_read_as_its_steps_with_their_heading_and_closing_lines(self):
        before, plan, after = A.plan_block("The problem is a ranking. 1. This is one thing.\n\n" + PLAN + "\n\nI have started the first step.")
        self.assertEqual((before, plan, after), ("The problem is a ranking. 1. This is one thing.", PLAN, "I have started the first step."))
        self.assertEqual(A.plan_block("One point. 1. Two things in a sentence.")[1], "")  # numbers inside a sentence are no plan
        self.assertEqual(A.plan_block("Options:\n1. A\n2. B\n\nfour\nlines\nof\ntext\nhere\n\n" + PLAN)[1], PLAN)  # the last list is the plan

    def test_a_plan_written_as_one_paragraph_becomes_a_list(self):
        inline = ("**The plan** 1. Split the customers by signup date. 2. Train a simple model and a baseline. 3. Rank by risk times spend. "
                  "Stop when the held-back part shows no gain. If a step fails, keep the baseline list.")
        listed = A.plan_lines(inline)
        self.assertEqual(listed.split("\n"), ["**The plan**", "1. Split the customers by signup date.", "2. Train a simple model and a baseline.",
                                               "3. Rank by risk times spend.", "Stop when the held-back part shows no gain.",
                                               "If a step fails, keep the baseline list."])
        self.assertEqual(A.plan_block(listed)[1], listed)
        for whole in ("We lost 1. That is one thing, and 2. is another.", "1. A single step on its line.", "Compare version 1. with version 2. only."):
            self.assertEqual(A.plan_lines(whole), whole)  # a sentence with numbers in it is not a plan

    def test_a_step_the_number_check_takes_out_leaves_no_gap_and_an_aside_goes_before_a_step_does(self):
        text = ("The problem is a ranking.\n\n**The plan**\n1. Split the customers by signup date.\n2. Keep the top 37% for the calls.\n"
                "3. Train a simple model (about 500 trees) and a baseline.\n4. Rank by risk times spend.\nStop when nothing improves.\n"
                "If a step fails, keep the baseline list.")
        with mock.patch.dict(os.environ, ON):
            s = state()
            out = A.turn(s, Script((text, [])), BRIEF)
        self.assertNotIn("37%", out["text"])
        self.assertNotIn("500 trees", out["text"])
        self.assertIn("1. Split the customers by signup date.\n2. Train a simple model and a baseline.\n3. Rank by risk times spend.", out["text"])
        self.assertEqual(s.plan.count("\n"), 5)  # the plan kept for later turns is the one the person saw

    def test_a_long_first_message_loses_words_around_the_plan_never_the_plan(self):
        long_message = words(900) + "\n\n" + PLAN
        with mock.patch.dict(os.environ, ON):
            s = state()
            out = A.turn(s, Script((long_message, [])), BRIEF)
        self.assertIn(PLAN, out["text"])  # whole, though the words before it were cut at the limit
        self.assertLess(len(out["text"].split()), A.SHOWN_WORDS_PLAN + A.PLAN_WORDS)
        self.assertEqual(s.plan, PLAN)
        with mock.patch.dict(os.environ, OFF):
            off = A.turn(state(), Script((long_message, [])), BRIEF)
        self.assertNotIn("If a step fails", off["text"])  # with the switch off nothing changes: the limit cuts the end

    def test_the_plan_is_shown_back_to_the_model_on_the_next_turn_and_survives_a_checkpoint(self):
        from dclab_rnd.agents import case_session

        with mock.patch.dict(os.environ, ON):
            s = state()
            A.turn(s, Script(("The problem is a ranking.\n\n" + PLAN, [])), BRIEF)
            again = Script(("I am on step two.", []))
            A.turn(s, again, "Go ahead.")
            self.assertIn("Your plan for this person", json.dumps(again.seen[0]))
            self.assertIn("Rank by risk times spend", json.dumps(again.seen[0]))
            kept = case_session.snapshot(s)
            fresh = state()
            case_session.restore(fresh, kept)
            self.assertEqual(fresh.plan, PLAN)
        with mock.patch.dict(os.environ, OFF):
            quiet = Script(("Fine.", []))
            s.plan = PLAN
            A.turn(s, quiet, "And then?")
            self.assertNotIn("Your plan for this person", json.dumps(quiet.seen[0]))  # only with the switch on

    def test_the_first_full_message_has_room_for_its_plan(self):
        seen = {}

        class Room(Script):
            def complete(self, messages, tools=None, *args, **kwargs):
                seen.setdefault("max_tokens", kwargs.get("max_tokens"))
                return super().complete(messages, tools, *args, **kwargs)

        with mock.patch.dict(os.environ, ON):
            A.turn(state(), Room(("A short answer for you.", [])), BRIEF)
        self.assertEqual(seen["max_tokens"], A.THINK_FIRST_TOKENS)
        self.assertGreater(A.THINK_FIRST_TOKENS, A.POLICY.max_tokens)

    def test_the_prompt_asks_for_options_weighed_and_a_plan_with_its_stop_and_its_failure(self):
        text = " ".join(prompts.text("case_agent_think_first").split())  # the prompt's own line breaks are not part of its words
        for words_ in ("The ways it could be solved", "the one you choose", "as many as the work needs", "how its result is checked",
                       "each on its own line", '"Stop when"', '"If a step fails"'):
            self.assertIn(words_, text)
        self.assertNotIn("three to five numbered steps", text)


class MeasurementTests(unittest.TestCase):
    def tearDown(self):
        H.use_set("dev")

    def test_the_held_out_set_is_frozen_and_apart(self):
        seen = set()
        for name in ("dev", "dev2"):
            H.use_set(name)
            seen |= {c["id"] for c in H.load()[0]}
        H.use_set("dev3")
        cases, key = H.load()  # checked for its shape by code; its people are read by no test and by nobody who changes the lead agent
        self.assertIsNone(H.frozen())
        self.assertEqual((len(cases), len(key)), (8, 8))
        self.assertFalse(seen & set(key))
        self.assertIn("dev3", H.HELD_OUT)

    def test_the_second_gate_is_pinned_and_judges_the_held_out_set_only(self):
        self.assertIsNone(G.pinned(G.GATE_V2))
        self.assertIsNone(G.pinned())  # the first gate is as it was
        self.assertEqual((G.gate_of("dev3"), G.gate_of("dev"), G.gate_of("dev2")), (G.GATE_V2, G.GATE_ID, G.GATE_ID))
        self.assertEqual(G.TERMS_V2["sets"], ["dev3"])
        self.assertIn("forbidden_done", G.ZERO_V2)

    def rows(self, planning, understanding, **counted):
        zero = {name: 0 for name in G.ZERO_V2}
        return [{"case": f"hab-2{i}", "judged": True, "planning": planning, "understanding": understanding, **zero, **counted} for i in range(1, 9)]

    def test_both_judged_checks_are_relative_to_the_same_model_alone(self):
        model = "qwen/qwen3.8-flash"
        judge = G.JUDGE_OF[model]
        self.assertTrue(G.verdict(model, judge, self.rows(3.5, 3.5), self.rows(4, 4), gate=G.GATE_V2)["passed"])
        self.assertFalse(G.verdict(model, judge, self.rows(3, 4), self.rows(4, 4), gate=G.GATE_V2)["passed"])  # the plan a point behind
        self.assertFalse(G.verdict(model, judge, self.rows(4, 3), self.rows(4, 4), gate=G.GATE_V2)["passed"])  # understanding a point behind
        low = G.verdict(model, judge, self.rows(2, 2), self.rows(2, 2), gate=G.GATE_V2)
        self.assertTrue(low["passed"])  # no floor from a tuned set: a model that is weak alone is not the harness's fault
        done = G.verdict(model, judge, self.rows(4, 4, forbidden_done=1), self.rows(4, 4), gate=G.GATE_V2)
        self.assertEqual([c["check"] for c in done["checks"] if not c["ok"]], ["forbidden done"])
        self.assertTrue(G.verdict(model, judge, self.rows(4, 4, forbidden_done=1), self.rows(4, 4))["passed"])  # the first gate never counted it

    def test_the_model_alone_can_be_held_to_the_lead_agents_length(self):
        self.assertEqual(H.ARMS, ("model_only", "open", "plain", "case"))  # the default run is as it was
        self.assertIn("model_only_short", H.ALL_ARMS)
        self.assertIn("model_only_short", H.PLAN_ONLY)
        long_, short = prompts.text(arms.MODEL_ONLY_PROMPT), prompts.text(arms.MODEL_ONLY_SHORT_PROMPT)
        self.assertEqual(short.replace(", in at most 450 words", ""), long_)  # the same prompt but for its length

        class Once:
            def __init__(self):
                self.seen = []

            def complete(self, messages, tools=None, **kwargs):
                self.seen.append(messages)
                return {"content": "A short plan.", "tool_calls": []}

        client = Once()
        arm = arms.ModelOnlyShortArm(client)
        self.assertEqual(arm.open({"id": "x", "opening": "predict churn", "upload": None})["text"], "A short plan.")
        self.assertEqual((arm.name, client.seen[0][0]["content"]), ("model_only_short", short))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------------------------------------------------
# The review of stage 4 (2026-11-16): one test per fix.

class ReviewFixTests(unittest.TestCase):
    OPTIONS = [{"title": "Rank customers by risk", "what": "a weekly ranked list", "track": "prediction", "tool": "ml_project",
                "columns": ["churned_next_month"], "gets": "a list your team can act on", "tradeoff": "needs past outcomes", "effort": "a day"},
               {"title": "Compare churn by plan", "what": "churn per plan", "track": "analysis", "tool": "query", "columns": ["plan"],
                "gets": "a table of rates", "tradeoff": "explains, does not predict", "effort": "an hour"}]

    def first_request(self, words, env):
        client = Script(("A short answer for you.", []))
        with mock.patch.dict(os.environ, env):
            if not env:
                os.environ.pop(A.THINK_FIRST, None)
            A.turn(state(), client, words)
        return json.dumps(client.seen[0])

    def test_fix_1_a_chart_of_their_table_is_still_answered_with_the_table_first(self):
        chart = "Which plan loses the most customers? Show me a chart."
        s = state()
        A.heard(s, chart)
        self.assertTrue(s.wants_figures)
        on = self.first_request(chart, ON)
        self.assertIn("start/query/answer", on)
        self.assertNotIn(A.CLEAR_THINK_FIRST, on)
        self.assertIn(A.CLEAR_THINK_FIRST, self.first_request(BRIEF, ON))  # a need to work out gets the new note
        self.assertIn("is answered first, with the table", prompts.text("case_agent_think_first"))

    def test_fix_2_a_message_already_written_is_not_asked_for_again(self):
        def proposed(env):
            s = state()
            A.heard(s, BRIEF)
            A.heard(s, "show me options")
            with mock.patch.dict(os.environ, env):
                if not env:
                    os.environ.pop(A.THINK_FIRST, None)
                return A._propose(s, self.OPTIONS)["next"]

        self.assertIn("it stands", proposed(ON))
        self.assertIn("do not write the plan", proposed(ON))
        self.assertIn("now write that message", proposed({}))  # unchanged with the switch off
        self.assertIn("it stands", A.WRITTEN_NEXT)

    def test_fix_3_the_sound_way_is_offered_as_two_options_and_comes_first(self):
        text = prompts.text("case_agent_think_first")
        self.assertIn("offer two sound\nways", text)
        self.assertIn("start the sound\nversion of it instead", text)
        self.assertIn("This comes before every rule below about\nstarting or building at once", text)

    def doc(self, name, flag, set_="dev"):
        rows = [{"case": f"x{i}", "judged": True, "understanding": 3, "solution_design": 3, "planning": 3, "evidence_discipline": 3,
                 "accepted": True, "forbidden_done": 0, "safety_failures": 0, "seconds": 10, "assistant_turns": 2, "spent_eur": 0.0}
                for i in range(8)]
        convs = [{"id": f"x{i}", "arm": "case", "ended": "answered", "transcript": [], "safety": {}} for i in range(8)]
        return {"id": name, "split": "dev", "set": set_, "label": "stage-4", "models": {"arms": "test/model", "judge": "test/judge"},
                "manifest": {"flags": {"DCLAB_CASE_THINK_FIRST": flag}}, "scores": {"case": rows}, "conversations": convs}

    def built(self, *docs):
        import tempfile

        from dclab_rnd.agent_eval.harness_ablation import benchmark

        folder = Path(tempfile.mkdtemp())
        for d in docs:
            (folder / f"{d['id']}_dev.json").write_text(json.dumps(d), encoding="utf-8")
        with mock.patch.object(benchmark, "RESULTS", folder):
            return benchmark, benchmark.build()

    def test_fix_4_a_run_with_the_switch_on_stays_apart_from_one_with_it_off(self):
        benchmark, data = self.built(self.doc("HAB-901", "1"), self.doc("HAB-902", None))
        self.assertEqual(sorted((r["id"], r["think_first"]) for r in data["runs"]), [("HAB-901", True), ("HAB-902", False)])
        self.assertTrue(benchmark.think_first_of(self.doc("a", "on")))
        self.assertFalse(benchmark.think_first_of(self.doc("a", "0")))
        self.assertIn("think first", benchmark._row_name({"model": "m", "judge": "j", "think_first": True}))

    def test_fix_5_each_gate_is_drawn_under_its_own_headings(self):
        from dclab_rnd.agent_eval.harness_ablation import benchmark

        def rows(planning):
            zero = {name: 0 for name in G.ZERO_V2}
            return [{"case": f"x{i}", "judged": True, "planning": planning, "understanding": 3, **zero} for i in range(8)]

        model = "qwen/qwen3.8-flash"
        one = {"run": "R1", "label": None, "set": "dev", **G.verdict(model, G.JUDGE_OF[model], rows(3), rows(3))}
        two = {"run": "R2", "label": None, "set": "dev3", **G.verdict(model, G.JUDGE_OF[model], rows(3), rows(3), gate=G.GATE_V2)}
        data = benchmark.build()
        data["gate"] = [one, two]
        text = benchmark.markdown(data)
        self.assertIn("## The gate (`hab_gate_v1`", text)
        self.assertIn("## The gate (`hab_gate_v2`", text)
        v2 = text[text.index("`hab_gate_v2`"):text.index("`hab_gate_v1`")]
        self.assertIn("forbidden done", v2)
        self.assertIn("understanding: lead agent minus the model alone", v2)
        self.assertIn("hab_gate_v2", benchmark.page(data))

    def test_fix_6_a_held_out_run_is_kept_to_its_means(self):
        benchmark, data = self.built(self.doc("HAB-903", None, set_="dev3"))
        run = next(r for r in data["runs"] if r["id"] == "HAB-903")
        self.assertTrue(run["held_out"])
        self.assertNotIn("by_case", run["arms"]["case"])
        self.assertNotIn("interventions_by_case", run)
        self.assertNotIn("counted", run)
        self.assertNotIn("x1", json.dumps(run))  # no person's id
        self.assertEqual(run["arms"]["case"]["planning"], 3)  # the means stay

    def test_the_test_runner_blanks_the_switch(self):
        sys.path.insert(0, str(ROOT / "tests"))
        import run_parallel

        self.assertIn(A.THINK_FIRST, run_parallel.OWNER_MEMORY_MODEL)
        self.assertEqual(run_parallel.child_env("/tmp")[A.THINK_FIRST], "")
