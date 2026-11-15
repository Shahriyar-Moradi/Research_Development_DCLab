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
        self.assertIn("what needs care before any work", note)
        self.assertIn("in this same turn", note)  # think first is not ask first: the person does not wait a turn
        self.assertIn("Do not ask clarifying questions", note)
        self.assertNotIn("immediately propose", note)


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.text = prompts.text("case_agent_think_first")

    def test_the_plan_comes_before_the_kinds_of_work(self):
        plan, kinds = self.text.index("Your first full message"), self.text.index("How DCLab runs a step of your plan")
        self.assertLess(plan, kinds)
        self.assertLess(self.text.index("What needs care before any work"), self.text.index("The approach you would take"))
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
