"""The harness ablation (campaign harness_ablation_v1): the same model in four arms on written cases, a judge blind to the
arm, and the first divergence of the lead agent read from the events a run keeps. Scripted models only."""
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import experience, prompts  # noqa: E402
from dclab_rnd.agent_eval import harness_ablation as H  # noqa: E402
from dclab_rnd.agent_eval.conversation_exam import arms as exam_arms  # noqa: E402
from dclab_rnd.agent_eval.harness_ablation import arms, first_divergence as F, gate as G, judge  # noqa: E402
from dclab_rnd.models import gateway as gw  # noqa: E402
from dclab_rnd.models import settings as model_settings  # noqa: E402
from dclab_rnd.models.usage import FileUsage  # noqa: E402


class Script:
    """A model client that returns the given replies in order and keeps what it was sent."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def complete(self, messages, tools=None, *args, **kwargs):
        self.seen.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        r = self.replies.pop(0) if self.replies else "Here is the plan."
        if isinstance(r, dict) and "tool_calls" in r:
            msg = {"role": "assistant", "content": r.get("content") or "", "tool_calls": r["tool_calls"]}
            return {"content": r.get("content") or "", "tool_calls": r["tool_calls"], "assistant_message": msg}
        text = r if isinstance(r, str) else json.dumps(r)
        return {"content": text, "tool_calls": [], "assistant_message": {"role": "assistant", "content": text}}


class CaseScript:
    """The lead agent's scripted model: each reply is its words and its case tool calls (name, arguments)."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def complete(self, messages, tools=None, *args, **kwargs):
        self.seen.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        text, calls = self.replies.pop(0) if self.replies else ("I will go on with what you wrote.", [])
        calls = [{"id": f"c{len(self.seen)}-{i}", "name": f"case_{n}", "arguments": a} for i, (n, a) in enumerate(calls)]
        msg = {"role": "assistant", "content": text,
               **({"tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]} if calls else {})}
        return {"content": text, "tool_calls": calls, "assistant_message": msg, "usage": {"input_tokens": 1, "output_tokens": 1}}


def verdict(key, **over):
    return {"understood": [True] * len(key["must_understand"]), "forbidden_done": [False] * len(key["forbidden"]), "understanding": 3,
            "solution_design": 3, "planning": 2, "evidence_discipline": 4, "adaptation": None, "accepted": True, **over}


class CasesTests(unittest.TestCase):
    def test_the_cases_are_complete_and_the_key_is_apart(self):
        cases, key = H.load()
        self.assertEqual(len(cases), 8)
        self.assertEqual(H.check_cases(cases, key), [])
        for c in cases:
            self.assertEqual(set(c), {"id", "stratum", "upload", "opening"})
        self.assertGreaterEqual(sum(not k["right_is_model"] for k in key.values()), 3)  # a third or more: the right answer is no model
        self.assertGreaterEqual(sum(not k["in_six_tracks"] for k in key.values()), 2)  # outside the lead agent's six kinds of work
        self.assertTrue(any(k["route"] == "prediction" and k["right_is_model"] for k in key.values()))  # a control the harness should win or tie

    def test_a_case_that_carries_a_field_of_its_key_is_refused(self):
        cases, key = H.load()
        bad = [dict(cases[0], must_understand=["x", "y"]), *cases[1:]]
        self.assertTrue(any("key stays apart" in p for p in H.check_cases(bad, key)))

    def test_a_run_is_refused_until_the_cases_are_frozen_and_the_code_is_one_commit(self):
        with mock.patch.object(H, "PIN", Path(tempfile.mkdtemp()) / "none.sha256"):
            self.assertIn("not frozen", H.frozen())
            with mock.patch.object(H, "code_state", lambda: {"commit": "abc", "dirty": [], "diff_sha256": None}), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                self.assertEqual(H.main(["--yes", "--cap-eur", "1"]), 2)
            self.assertIn("not frozen", err.getvalue())
        pin = Path(tempfile.mkdtemp()) / "cases.sha256"
        pin.write_text(H.fingerprint() + "\n", encoding="utf-8")
        with mock.patch.object(H, "PIN", pin):
            self.assertIsNone(H.frozen())
            with mock.patch.object(H, "code_state", lambda: {"commit": "abc", "dirty": ["dclab_rnd/agents/case_agent.py"], "diff_sha256": "x"}), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                self.assertEqual(H.main(["--yes", "--cap-eur", "1"]), 2)
            self.assertIn("uncommitted", err.getvalue())
            state = H.code_state()
            if state["dirty"]:  # a run on uncommitted code names what it ran on: the diff's hash, and the untracked files' hash
                self.assertTrue(state["diff_sha256"] or state["untracked_sha256"])
            pin.write_text("0" * 64 + "\n", encoding="utf-8")
            self.assertIn("changed after", H.frozen())

    def test_without_yes_it_prints_the_plan_and_sends_nothing(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(H.main([]), 0)
        plan = json.loads(out.getvalue().split("nothing was sent")[0])
        self.assertEqual((plan["cases"], plan["conversations"]), (8, 8 * 4))  # each case once by default: a second repeat is asked for
        self.assertGreater(plan["estimate_eur"], 0)


class ArmsTests(unittest.TestCase):
    def test_the_model_only_arm_sends_one_request_with_the_table_view_and_no_tools(self):
        client = Script("A plan.")
        arm = arms.ModelOnlyArm(client)
        turn = arm.open({"id": "x", "opening": "predict churn", "upload": "customers"})
        self.assertEqual((turn["text"], turn["choices"], len(client.seen)), ("A plan.", [], 1))
        self.assertIsNone(client.seen[0]["tools"])
        sent = client.seen[0]["messages"][-1]["content"]
        self.assertIn("churned_next_month", sent)  # the column view the other arms can read
        self.assertNotIn("C-00001", sent)  # never a value
        self.assertEqual(arm.outcome()["route"], None)
        with self.assertRaises(RuntimeError):
            arm.send("more")

    def test_the_open_arm_is_given_no_kind_of_work_to_choose(self):
        names = [t["function"]["name"] for t in arms.READ_TOOLS]
        self.assertEqual(names, ["profile_table", "query_table", "check_leakage", "preview_step"])
        text = prompts.text(arms.OPEN_PROMPT) + prompts.text(arms.MODEL_ONLY_PROMPT)
        for word in (*(t for t in exam_arms.TOOLS if t != "query"), "offer_options", "start_work", '"query"', '"prediction"', '"advice"'):
            self.assertNotIn(word, text)  # "query" alone is part of the read tool's name, query_table
        self.assertNotIn("enum", json.dumps([t for t in arms.READ_TOOLS if t["function"]["name"] == "profile_table"]))

    def test_the_open_arm_runs_the_read_tools_and_ends_on_words(self):
        client = Script({"tool_calls": [{"id": "c1", "name": "profile_table", "arguments": {}}]},
                        {"tool_calls": [{"id": "c2", "name": "start_work", "arguments": {"track": "prediction", "tool": "ml_project"}}]}, "Here is what I would do.")
        arm = arms.OpenArm(client)
        turn = arm.open({"id": "x", "opening": "predict churn", "upload": "customers"})
        self.assertEqual(turn["text"], "Here is what I would do.")
        self.assertEqual(arm.notes()["tool_calls"], [{"tool": "profile_table", "ok": True}, {"tool": "start_work", "ok": False}])
        self.assertEqual(len(arm.sources()), 1)
        self.assertIsNone(Script().complete([], None)["tool_calls"] or None)
        no_table = arms.OpenArm(Script("An answer."))
        no_table.open({"id": "y", "opening": "what should we do?", "upload": None})
        self.assertIsNone(no_table.client.seen[0]["tools"])

    def test_the_open_arm_is_told_when_its_last_request_comes(self):
        """HAB-006: a model still looking at the table on the last request, sent with no tools and no word about it, answered
        with a tool call written as text. The runner now says it was the last look."""
        look = {"tool_calls": [{"id": "c", "name": "profile_table", "arguments": {}}]}
        client = Script(*([look] * (arms.MAX_CALLS - 1)), "My answer, from what I saw.")
        arm = arms.OpenArm(client)
        turn = arm.open({"id": "x", "opening": "predict churn", "upload": "customers"})
        self.assertEqual(turn["text"], "My answer, from what I saw.")
        last = client.seen[-1]
        self.assertIsNone(last["tools"])
        self.assertEqual(last["messages"][-1], {"role": "user", "content": arms.LAST_REQUEST})
        self.assertNotIn(arms.LAST_REQUEST, json.dumps(client.seen[-2]["messages"]))  # said once, on the last request only

    def test_no_new_prompt_asks_for_reasoning_in_its_answer(self):
        from tests.test_prompt_audit import ASKS_REASONING

        for name in (arms.MODEL_ONLY_PROMPT, arms.OPEN_PROMPT, judge.PROMPT):
            self.assertIsNone(ASKS_REASONING.search(prompts.text(name)), name)


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.cases, self.key = H.load()
        self.case, self.k = self.cases[0], self.key[self.cases[0]["id"]]
        self.transcript = [{"role": "person", "text": self.case["opening"], "choices": []}, {"role": "assistant", "text": "A plan.", "choices": []}]

    def test_the_judge_reads_the_key_and_the_conversation_and_never_the_arm(self):
        text = judge.case_text(self.case, self.k, self.transcript)
        self.assertIn(self.k["must_understand"][0], text)
        self.assertIn(self.k["forbidden"][0], text)
        self.assertIsNone(re.search(r"\b(model_only|plain|open arm|lead agent|arm)\b", text))
        self.assertNotIn("right_is_model", text)
        self.assertNotIn("in_six_tracks", text)

    def test_an_answer_of_the_wrong_shape_is_invalid_and_never_repaired(self):
        self.assertEqual(judge.check(verdict(self.k), self.k), [])
        self.assertTrue(judge.check(verdict(self.k, understanding=5), self.k))
        self.assertTrue(judge.check(verdict(self.k, adaptation=3), self.k))  # no adaptation check in the case: it must be null
        self.assertTrue(judge.check(verdict(self.k, understood=[True]), self.k))
        self.assertTrue(judge.check(verdict(self.k, accepted="yes"), self.k))
        with_check = dict(self.k, adaptation_check="changes course when the table shows no fall")
        self.assertTrue(judge.check(verdict(self.k), with_check))
        self.assertEqual(judge.check(verdict(self.k, adaptation=2), with_check), [])
        got = judge.judged(Script("not json", verdict(self.k)), self.case, self.k, self.transcript)
        self.assertTrue(got["valid"])  # asked once more, then taken as it is
        got = judge.judged(Script("not json", "still not"), self.case, self.k, self.transcript)
        self.assertFalse(got["valid"])
        self.assertIn("first_problems", got)


class FirstDivergenceTests(unittest.TestCase):
    def test_every_fragment_is_still_a_message_of_the_code_it_names(self):
        for reason, _, fragment, path in F.FRAGMENTS:
            if path is not None:  # None: a library writes the message (the registry's schema check)
                self.assertIn(fragment, (ROOT / path).read_text(encoding="utf-8"), f"{reason}: the message changed; update FRAGMENTS")

    def tool(self, name, verdict_, result, **arguments):
        return {"what": "tool", "tool": name, "verdict": verdict_, "arguments": arguments, "result": result}

    def test_a_refused_start_a_second_reading_and_a_route_set_by_code_are_told_apart(self):
        refused = self.tool("case_start", "error", "error: the person has not chosen and the case is not ready: decision is not known yet", track="preparation", tool="recipe")
        self.assertEqual(F.reason_of(refused), "start_refused_not_ready")
        doubt = self.tool("case_start", "error", "error: the person's need may call for analysis (a second reading of their words), not prediction.")
        self.assertEqual(F.reason_of(doubt), "route_overruled")
        route = {"what": "case", "field": "route", "by": "agent", "after": '{"track": "analysis", "tool": "query"}'}
        self.assertEqual(F.reason_of(route, self.tool("case_answer", "ok", "{}")), "route_set_by_code")  # _worked: no start was called
        self.assertIsNone(F.reason_of(route, self.tool("case_start", "ok", '{"started": "analysis/query"}')))  # the model's own start
        dropped = self.tool("case_propose", "ok", '{"shown": ["a", "b"], "refused": {"Delete every row": "the same kind as an option already kept"}}')
        self.assertEqual(F.reason_of(dropped), "options_dropped")
        self.assertIsNone(F.reason_of(self.tool("case_propose", "ok", '{"shown": ["a", "b"], "refused": {}}')))
        kept = self.tool("case_propose", "ok", '{"shown": ["a", "b"], "refused": {"A randomised pilot (kept)": "shown without the sentences holding numbers nobody gave (23%)"}}')
        self.assertEqual(F.reason_of(kept), "options_trimmed")  # the option stays: no longer a dropped one
        self.assertEqual(F.reason_of(self.tool("case_propose", "error", "error: the work has started: tell the person what happens next")), "guard_refused")
        self.assertEqual(F.reason_of(self.tool("case_start", "error", "error: DCLab cannot run research/experiment")), "closed_list")
        self.assertIsNone(F.reason_of(self.tool("case_note", "error", "error: the quote is not the person's words")))
        # a kind of work outside the list is refused by the registry's schema check, before the lead agent's own code
        schema = self.tool("case_start", "error", "error: Invalid argument for case_start: tool: 'ab_test' is not one of ['ml_project', 'query']")
        self.assertEqual(F.reason_of(schema), "closed_list")
        self.assertEqual(F.reason_of(self.tool("case_start", "error", "error: the work has started as analysis/query. Change it only when")), "route_change_refused")
        self.assertEqual(F.reason_of(self.tool("case_start", "error", "error: the person has no data yet or prefers not to upload it: call")), "no_data_redirect")
        # the table tools: a question the query cannot express, and the per-turn caps on questions
        self.assertEqual(F.reason_of(self.tool("case_query", "error", "error: measure q4: a condition (where) belongs to a count or a share, not to sum: Fix")), "query_cannot_express")
        self.assertEqual(F.reason_of(self.tool("case_answer", "error", "error: measure last: max needs a numeric column; order_date is categorical")), "query_cannot_express")
        self.assertEqual(F.reason_of(self.tool("case_query", "error", "error: the table has been asked 4 times this turn: write to the person now")), "query_cap")
        self.assertEqual(F.reason_of(self.tool("case_answer", "error", "error: The arguments for case_answer were not valid JSON; send")), "bad_arguments")
        # a refusal nobody listed is kept as one, never read as "no gate fired"
        self.assertEqual(F.reason_of(self.tool("case_query", "error", "error: some new message nobody listed")), "other_refusal")

    def test_a_conversation_no_code_touched_is_an_error_only_when_its_route_is_wrong(self):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        person = {"id": "p", "route": "prediction", "also_right": []}
        conv = {"ended": "answered", "route": "prediction", "tool": "ml_project", "routed_by": "self"}
        self.assertEqual(F.read(folder, person, conv)["first"]["reason"], "no_gate")  # the right route, no refusal: nothing to explain
        self.assertEqual(F.read(folder, person, dict(conv, route="advice"))["first"]["reason"], "model_error")
        self.assertEqual(F.read(folder, person, dict(conv, route="advice", routed_by="option"))["first"]["reason"], "picked_wrong_option")
        self.assertEqual(F.read(folder, person, dict(conv, route=None))["first"]["reason"], "no_route")
        self.assertEqual(F.read(folder, {"id": "p", "route": None}, dict(conv, route="advice"))["first"]["reason"], "no_gate")  # no written route to be wrong against

    def test_a_turn_ended_by_its_reply_limit_is_a_budget_ending(self):
        self.assertTrue(F._out_of_budget({"state": {"agent": "case", "step": 8, "steps_left": 6}, "answers": [{"by": "rule", "answerer": "rule:budget"}]}))
        self.assertTrue(F._out_of_budget({"state": {"agent": "case", "step": 14, "steps_left": 0}, "answers": [{"by": "model_role", "answerer": "case"}]}))
        self.assertFalse(F._out_of_budget({"state": {"agent": "case", "step": 4, "steps_left": 10}, "answers": [{"by": "model_role", "answerer": "case"}]}))

    def test_it_says_whether_the_refused_move_was_the_right_one(self):
        person = {"route": "preparation", "also_right": []}
        self.assertTrue(F._right({"track": "preparation", "tool": "recipe"}, person))
        self.assertFalse(F._right({"tracks": ["advice", "analysis"]}, person))
        self.assertIsNone(F._right({"track": "advice"}, {"route": None}))  # a case outside the six kinds has no route to compare

    def test_the_kept_dev_runs_are_read_without_a_sealed_one(self):
        found = F.recorded_dev()
        self.assertTrue(all("sealed" not in str(r["split"]) for r in found["runs"]))
        self.assertEqual(sum(found["first_reason_counts"].values()), found["lost"])
        stored = json.loads((H.RESULTS / "HAB-000_recorded_dev.json").read_text(encoding="utf-8"))
        self.assertEqual({k: stored[k] for k in ("lost", "won", "lost_people")},
                         {k: found[k] for k in ("lost", "won", "lost_people")})  # the stored result is what the code reads
        # the first reasons are read from the conversations each run kept under agent_runs/, a folder git does not carry: a
        # clean checkout (the commit's gate) has none of them and names every lost pair "not_kept"
        if set(found["first_reason_counts"]) != {"not_kept"}:
            self.assertEqual(stored["first_reason_counts"], found["first_reason_counts"])
        self.assertTrue(all(r["plain_right_case_wrong"] is not None and r["people"] for r in found["runs"]))


class SecondSetTests(unittest.TestCase):
    """The eight people of 2026-11-15 written by an independent agent (dev2): frozen, apart from the first eight, and a run
    on them is judged with their own key."""

    def tearDown(self):
        H.use_set("dev")

    def test_the_second_set_is_frozen_and_shares_no_person_with_the_first(self):
        first = {c["id"] for c in H.load()[0]}
        H.use_set("dev2")
        cases, key = H.load()
        self.assertIsNone(H.frozen())
        self.assertEqual((len(cases), set(key)), (8, {c["id"] for c in cases}))
        self.assertFalse(first & set(key))
        self.assertGreaterEqual(sum(c["upload"] is None for c in cases), 2)
        self.assertEqual(H.SET, "dev2")
        with self.assertRaises(ValueError):
            H.use_set("sealed")

    def test_the_benchmark_keeps_the_two_sets_apart(self):
        from dclab_rnd.agent_eval.harness_ablation import benchmark

        def run(name, set_, label=None):
            rows = {"case": {"by_case": {f"{name}-{i}": {"planning": 2, "understanding": 3} for i in range(8)}}}
            return {"id": name, "model": "m", "judge": "j", "label": label, "set": set_, "arms": rows, "interventions": {}, "counted": {}}

        made = benchmark.before_after([{**run("a", "dev"), "arms": {"case": {"by_case": {}, "seconds": 1, "assistant_turns": 1, "numbers_nobody_gave": 0,
                                                                               "spent_eur": 0, "accepted": 0, "cases": 0, **{d: None for d in benchmark.DIMENSIONS}}}},
                                       {**run("b", "dev2", "stage-1"), "arms": {"case": {"by_case": {}, "seconds": 1, "assistant_turns": 1, "numbers_nobody_gave": 0,
                                                                                         "spent_eur": 0, "accepted": 0, "cases": 0, **{d: None for d in benchmark.DIMENSIONS}}}}])
        self.assertEqual(made, [])  # a run on other people is no "before" of this one
        self.assertLess(benchmark._order("after-three-fixes"), benchmark._order("stage-1"))


class GateTests(unittest.TestCase):
    """The gate of the repair (stage 0 of the plan of 2026-11-15): pinned before any behaviour changed, an AND of its checks."""

    QWEN = "qwen/qwen3.8-flash"

    def rows(self, planning, understanding=4, **counted):
        zero = {name: 0 for name in G.ZERO}
        return [{"case": f"hab-0{i}", "judged": True, "planning": planning, "understanding": understanding, **zero, **counted} for i in range(1, 9)]

    def test_the_gate_is_the_pinned_one(self):
        self.assertIsNone(G.pinned())
        self.assertEqual(G.TERMS["id"], "hab_gate_v1")  # other terms are another gate: a new id and a new pin
        self.assertEqual(set(G.QUICK_SET) - {c["id"] for c in H.load()[0]}, set())

    def test_every_check_must_hold(self):
        judge_ = G.JUDGE_OF[self.QWEN]
        self.assertTrue(G.verdict(self.QWEN, judge_, self.rows(3), self.rows(3.5))["passed"])  # half a point below the model alone
        self.assertFalse(G.verdict(self.QWEN, judge_, self.rows(2), self.rows(3.5))["passed"])
        self.assertFalse(G.verdict(self.QWEN, judge_, self.rows(3, understanding=3), self.rows(3.5))["passed"])  # below the floor
        for name in G.ZERO:
            failed = G.verdict(self.QWEN, judge_, self.rows(3, **{name: 1}), self.rows(3.5))
            self.assertFalse(failed["passed"], name)
            self.assertEqual([c["check"] for c in failed["checks"] if not c["ok"]], [name.replace("_", " ")])

    def test_what_cannot_be_read_is_not_passed(self):
        judge_ = G.JUDGE_OF[self.QWEN]
        uncounted = [{k: v for k, v in r.items() if k != "empty_answers"} for r in self.rows(3)]
        self.assertEqual(G.verdict(self.QWEN, judge_, uncounted, self.rows(3.5))["not_read"], ["empty answers"])
        self.assertFalse(G.verdict(self.QWEN, "qwen/qwen3.8-27b", self.rows(3), self.rows(3.5))["passed"])  # a judge of the arm's own family
        self.assertFalse(G.verdict(self.QWEN, judge_, self.rows(3)[:3], self.rows(3.5)[:3])["passed"])  # the quick set is no verdict
        self.assertFalse(G.verdict("some/other-model", "a judge", self.rows(3), self.rows(3.5))["passed"])  # no floor was fixed for it

    def test_a_dead_end_is_refusals_in_a_row_of_the_moves_that_decide(self):
        def move(tool, ok=False):
            return {"what": "tool", "tool": tool, "verdict": "ok" if ok else "refused"}

        self.assertEqual(F.blocked_runs([move("case_start"), move("case_request_build")]), [["case_start", "case_request_build"]])
        self.assertEqual(F.blocked_runs([move("case_start")] * 3), [["case_start"] * 3])
        self.assertEqual(F.blocked_runs([move("case_start"), move("case_start")]), [])  # one move tried twice is a retry
        self.assertEqual(F.blocked_runs([move("case_start"), move("case_propose", ok=True), move("case_request_build")]), [])
        self.assertEqual(F.blocked_runs([move("case_query")] * 5), [])  # a table question asked in another shape is no dead end


class RunTests(unittest.TestCase):
    """One case through every arm with scripted models: the key never reaches an arm, the lead agent's refused start is found
    in the events the run kept, and the scorecard, the manifest and the receipt are written."""

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, True)
        cases, self.key = H.load()
        self.cases = [c for c in cases if c["id"] == "hab-08"]
        self.clients: dict[str, list] = {a: [] for a in H.ARMS}
        self.judges: list[Script] = []

    def arm_for(self):
        def keep(arm, client):
            self.clients[arm].append(client)
            return client

        return {"model_only": lambda name, folder: arms.ModelOnlyArm(keep("model_only", Script("A plan with steps."))),
                "open": lambda name, folder: arms.OpenArm(keep("open", Script("Another plan."))),
                "plain": lambda name, folder: exam_arms.PlainArm(keep("plain", Script("A third plan."))),
                "case": lambda name, folder: exam_arms.CaseArm(keep("case", CaseScript(
                    ("", [("start", {"track": "prediction", "tool": "ml_project"})]), ("I will look at what the model needs first.", []))), None, folder)}

    def judge_for(self, name, arm):
        self.judges.append(Script(verdict(self.key["hab-08"], accepted=arm != "case", planning=1 if arm == "case" else 3)))
        return self.judges[-1]

    def test_one_case_through_every_arm(self):
        result = H.measure(self.cases, self.key, self.arm_for(), self.judge_for, repeats=1, keep=self.folder, run="HAB-T")
        self.assertIsNone(result["incomplete"])
        self.assertEqual({a: len(rows) for a, rows in result["scores"].items()}, {a: 1 for a in H.ARMS})
        # the key never reaches an arm's model
        secret = [*self.key["hab-08"]["must_understand"], *self.key["hab-08"]["acceptable_approaches"], *self.key["hab-08"]["forbidden"]]
        for arm, clients in self.clients.items():
            sent = json.dumps([c.seen for c in clients])
            for phrase in secret:
                self.assertNotIn(phrase, sent, arm)
        # the judge reads the key, and the same words whichever arm spoke
        for j in self.judges:
            self.assertIn(secret[0], j.seen[0]["messages"][-1]["content"])
        # the lead agent asked to start the right work and its readiness gate refused: read from the events the run kept
        case_conv = next(c for c in result["conversations"] if c["arm"] == "case")
        first = case_conv["first_divergence"]["first"]
        self.assertEqual((first["reason"], first["wanted"], first["wanted_right"], first["safety"]),
                         ("start_refused_not_ready", {"track": "prediction", "tool": "ml_project"}, True, False))
        self.assertEqual(result["first_divergence_counts"], {"start_refused_not_ready": 1})
        row = result["scores"]["case"][0]
        self.assertEqual((row["first_divergence"], row["wanted_right"], row["accepted"], row["planning"]), ("start_refused_not_ready", True, False, 1))
        self.assertIsNone(result["scores"]["open"][0]["first_divergence"])
        # each dimension by itself, and the pairs by case
        self.assertEqual(result["summary"]["open"]["planning"], 3.0)
        self.assertNotIn("score", result["summary"]["open"])
        self.assertEqual(result["paired"]["case_minus_open"]["planning"]["difference"], -2.0)
        self.assertNotIn("case_minus_model_only_execution", result["paired"])
        # kept whole, with the script's marker, a scorecard and a receipt of every file
        self.assertTrue((self.folder / "scripted.json").exists())
        kept = json.loads((self.folder / "case" / "hab-08-r1" / "conversation.json").read_text(encoding="utf-8"))
        self.assertEqual(kept["case"]["id"], "hab-08")
        self.assertNotIn("must_understand", kept["case"])
        H.write_scorecard(self.folder / "scorecard.csv", result["scores"])
        lines = (self.folder / "scorecard.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual((lines[0].split(","), len(lines)), (list(H.FIELDS), 5))
        got = H.receipt(self.folder)
        self.assertEqual(got["files"], len(got["file_sha256"]))
        self.assertIn("scorecard.csv", got["file_sha256"])
        self.assertTrue((self.folder / "archive_receipt.json").exists())

    def test_the_script_clicks_an_option_the_key_accepts_and_says_go_ahead_otherwise(self):
        class Shows:
            name = "case"

            def __init__(self, tracks):
                self.tracks, self.sent, self.picks = tracks, [], []

            def _turn(self, text, options=True):
                return {"text": text, "choices": [{"title": f"Do {t}", "what": "", "track": t, "tool": "x", "kind": "option"} for t in self.tracks] if options else []}

            def open(self, person):
                self.person = dict(person)
                return self._turn("Here are two ways.")

            def pick(self, i):
                self.picks.append(i)
                return self._turn("Started.", options=False)

            def send(self, words):
                self.sent.append(words)
                return self._turn("Which one?") if len(self.sent) < 9 else self._turn("Done.", options=False)

            def outcome(self):
                return {"route": None, "tool": None, "routed_by": None}

            def sources(self):
                return []

            def notes(self):
                return {}

        case, key = self.cases[0], self.key["hab-08"]  # the key's route is prediction
        arm = Shows(["analysis", "prediction"])
        conv = H.play(case, arm, key=key)
        self.assertEqual((arm.picks, arm.sent, conv["picked"], conv["options_unpicked"], conv["assistant_turns"]), ([1], [], 1, 0, 2))
        self.assertEqual(set(arm.person), {"id", "opening", "upload"})  # the key decides the click; the arm never gets it
        arm = Shows(["analysis", "advice"])  # no option fits: the person says go ahead, up to the last turn
        conv = H.play(case, arm, key=key)
        self.assertEqual((arm.picks, arm.sent, conv["options_unpicked"], conv["assistant_turns"], conv["ended"]),
                         ([], [H.GO_AHEAD] * 3, 3, H.MAX_TURNS, "asks"))
        from dclab_rnd.agents import case_agent

        self.assertGreater(H.MAX_TURNS, case_agent.TURN_BUDGET)  # the lead agent's own turn budget can be reached

    def test_a_conversation_the_cap_stopped_is_kept_out_of_the_scores(self):
        def capped(name, folder):
            raise RuntimeError("BudgetExceeded: the run's cap is reached")

        result = H.measure(self.cases, self.key, {"open": capped}, self.judge_for, repeats=1)
        self.assertEqual((result["incomplete"], result["scores"]["open"], result["conversations"], result["paired"]), ("cap reached", [], [], None))

        def refused(name, folder):  # a key's own spending limit: the provider refuses it (HAB-003)
            raise RuntimeError("the provider refused the key")

        result = H.measure(self.cases, self.key, {"open": refused}, self.judge_for, repeats=1)
        self.assertEqual((result["incomplete"], result["scores"]["open"]), ("the provider refused the key", []))

    def test_the_arms_can_be_played_now_and_judged_later_from_what_was_kept(self):
        played = H.measure(self.cases, self.key, self.arm_for(), None, repeats=1, keep=self.folder, run="HAB-T")
        self.assertTrue(all(not r["judged"] and not r["failed"] and r["accepted"] is None for rows in played["scores"].values() for r in rows))
        self.assertIsNone(played["paired"])  # no scores to compare yet
        self.assertEqual(played["first_divergence_counts"], {"start_refused_not_ready": 1})  # the trace needs no judge
        before = (self.folder / "case" / "hab-08-r1" / "conversation.json").read_bytes()
        judged = H.judge_kept(self.folder, self.key, self.judge_for, run="HAB-T")
        self.assertEqual(len(self.judges), 4)
        self.assertTrue(all(r["judged"] for rows in judged["scores"].values() for r in rows))
        self.assertEqual(judged["paired"]["case_minus_open"]["planning"]["difference"], -2.0)
        self.assertEqual(judged["scores"]["case"][0]["first_divergence"], "start_refused_not_ready")
        self.assertEqual((self.folder / "case" / "hab-08-r1" / "conversation.json").read_bytes(), before)  # a kept conversation is never rewritten
        self.assertTrue((self.folder / "case" / "hab-08-r1" / "judged.json").exists())
        again = H.judge_kept(self.folder, self.key, self.judge_for, run="HAB-T")  # already judged: read, never asked again
        self.assertEqual(len(self.judges), 4)
        self.assertEqual(again["summary"], judged["summary"])

    def test_an_arm_that_breaks_is_scored_zero_and_never_dropped(self):
        def broken(name, folder):
            raise RuntimeError("the model did not answer")

        result = H.measure(self.cases, self.key, {"open": broken}, self.judge_for, repeats=1)
        row = result["scores"]["open"][0]
        self.assertEqual((row["failed"], row["understanding"], row["accepted"]), (True, 0, False))
        self.assertEqual(result["summary"]["open"]["failed"], 1)

    def test_the_manifest_pins_the_code_the_models_the_prompts_and_the_cases(self):
        info = H.manifest("HAB-T", list(H.ARMS), 2, 3.0, {"arms": "qwen/qwen3.8-flash", "judge": H.JUDGE_MODEL})
        self.assertTrue(info["code"]["commit"])
        self.assertEqual(set(info["prompt_sha256"]), set(H.PROMPTS))
        self.assertTrue(all(info["prompt_sha256"].values()))
        self.assertTrue(all(info["tool_schema_sha256"].values()))
        self.assertEqual(info["cases_sha256"], H.fingerprint())
        self.assertIn("DCLAB_CASE_AGENT", info["flags"])
        self.assertNotEqual(info["models"]["arms"], info["models"]["judge"])  # the judge is never the arms' model


class ThinkingKeptTests(unittest.TestCase):
    """Every model call of the run keeps its API thinking summary, under the run's own folder: the two new arms through the
    exam's plain purpose, and the judge through its own endpoint."""

    def test_the_arms_and_the_judge_keep_their_thinking_in_the_run_folder(self):
        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, True)
        cases, key = H.load()

        class T:
            def __init__(self):
                self.thinking = {"type": "adaptive"}

            def complete(self, messages, tools=None, max_tokens=1800, **options):
                text = json.dumps(verdict(key["hab-07"])) if "judge" in messages[0]["content"].split(".")[0] else "A plan."
                return {"content": text, "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1}, "thinking": [{"text": "the summary"}],
                        "assistant_message": {"role": "assistant", "content": text}}

        transport = T()
        clean = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "DCLAB_NO_LIVE_MODELS", "DCLAB_THINKING",
                                 "DCLAB_BLOCK_PROVIDERS", "OPENROUTER_BASE_URL")} | {"OPENAI_API_KEY": "k", "OPENROUTER_API_KEY": "o", "ANTHROPIC_API_KEY": "", "DCLAB_AGENT_HOME": str(home / "elsewhere")}
        with mock.patch.object(gw, "_transport", lambda tier, purpose: transport), mock.patch.dict(os.environ, clean), \
                mock.patch.object(experience, "current_log", lambda: None):
            g = gw.mark_invented_tables(gw.Gateway(FileUsage(home / "usage.jsonl"), transport=gw._transport), "harness ablation test")
            g.home = home
            case = next(c for c in cases if c["id"] == "hab-07")
            a = arms.ModelOnlyArm(g.client(H.PURPOSE["model_only"])).open(case)
            b = arms.OpenArm(g.client(H.PURPOSE["open"])).open(case)
            tier = H.judge_tier()
            self.assertEqual((tier["provider"], tier["model"], model_settings._host(tier["base_url"])), ("openai", H.JUDGE_MODEL, model_settings.OPENROUTER_HOST))
            self.assertNotIn("claude", H.JUDGE_MODEL)  # the owner's rule: no Claude model in any test, arm or judge
            bound = gw.Bound(g, H.PURPOSE["judge"], model_settings.purpose(H.PURPOSE["judge"]), tier, None, "exam:HAB-T:judge")
            got = judge.judged(bound, case, key["hab-07"], [{"role": "person", "text": case["opening"], "choices": []},
                                                            {"role": "assistant", "text": a["text"], "choices": []}])
        self.assertEqual((a["text"], b["text"], got["valid"]), ("A plan.", "A plan.", True))
        docs = [json.loads(line) for p in sorted(home.rglob("model_exchanges/**/*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sorted(d["purpose"] for d in docs), ["exam_judge", "exam_plain", "exam_plain"])
        # the judge's request is sent on the judge's own model: a tier of another name was moved to the arms' model on the first request
        self.assertEqual([d["model"] for d in docs if d["purpose"] == "exam_judge"], [H.JUDGE_MODEL])
        self.assertTrue(all(d["model"] != H.JUDGE_MODEL for d in docs if d["purpose"] == "exam_plain"))
        self.assertTrue(all(d["thinking"] == [{"text": "the summary"}] for d in docs))
        self.assertTrue(all(d["whole"] is True for d in docs))
        self.assertFalse(list((home / "elsewhere").rglob("*.jsonl")))  # nothing outside the run's folder
        kept = H.capture(home)
        self.assertEqual((kept["ok"], kept["with_thinking"], kept["without_thinking"]), (3, 3, []))

    def test_a_live_run_end_to_end_with_a_scripted_transport(self):
        """The --yes path whole: the run folder, the judge on its own model, the result, the capture, the receipt."""
        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, True)
        _, key = H.load()

        class T:
            def __init__(self):
                self.thinking = {"type": "adaptive"}

            def complete(self, messages, tools=None, max_tokens=1800, **options):
                text = json.dumps(verdict(key["hab-07"])) if messages[0]["content"].startswith("You judge") else "Route planning is an optimisation problem."
                return {"content": text, "tool_calls": [], "usage": {"input_tokens": 10, "output_tokens": 10}, "thinking": [{"text": "the summary"}],
                        "assistant_message": {"role": "assistant", "content": text}}

        transport = T()

        def scripted_gateway(run_dir):
            g = gw.mark_invented_tables(gw.Gateway(FileUsage(Path(run_dir) / "usage.jsonl"), transport=lambda tier, purpose: transport), "harness ablation test")
            g.home, g.shadows = Path(run_dir), None
            return g

        clean = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "DCLAB_NO_LIVE_MODELS", "DCLAB_THINKING",
                                 "DCLAB_BLOCK_PROVIDERS", "OPENROUTER_BASE_URL")} | {"OPENAI_API_KEY": "k", "OPENROUTER_API_KEY": "o", "ANTHROPIC_API_KEY": "", "DCLAB_AGENT_HOME": str(home / "elsewhere")}
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, clean), mock.patch("dotenv.load_dotenv", lambda *a, **k: None), mock.patch.object(experience, "current_log", lambda: None), \
                mock.patch.object(H, "frozen", lambda: None), mock.patch.object(H, "code_state", lambda: {"commit": "abc", "dirty": [], "diff_sha256": None}), \
                mock.patch.object(H, "owner_allows", lambda cap: None), mock.patch.object(H, "run_gateway", scripted_gateway), \
                mock.patch.object(H, "KEEP_ROOT", home / "live_tests"), mock.patch.object(H, "RESULTS", home / "results"), \
                redirect_stdout(out), redirect_stderr(err):
            code = H.main(["--only", "hab-07", "--arms", "model_only,open", "--yes", "--cap-eur", "1"])
        self.assertEqual(code, 0, err.getvalue())
        # the same cases on another model: every arm's request names it, and the result says so
        seen_models: list[str] = []

        class Named(T):
            def complete(self, messages, tools=None, max_tokens=1800, **options):
                return super().complete(messages, tools, max_tokens, **options)

        def other_gateway(run_dir):
            g = gw.mark_invented_tables(gw.Gateway(FileUsage(Path(run_dir) / "usage.jsonl"), transport=lambda tier, purpose: (seen_models.append(tier["model"]), Named())[1]), "t")
            g.home, g.shadows = Path(run_dir), None
            return g

        with mock.patch.dict(os.environ, clean), mock.patch("dotenv.load_dotenv", lambda *a, **k: None), mock.patch.object(experience, "current_log", lambda: None), \
                mock.patch.object(H, "frozen", lambda: None), mock.patch.object(H, "code_state", lambda: {"commit": "abc", "dirty": [], "diff_sha256": None}), \
                mock.patch.object(H, "owner_allows", lambda cap: None), mock.patch.object(H, "run_gateway", other_gateway), \
                mock.patch.object(H, "price_snapshot", lambda run_dir, model: None), \
                mock.patch.object(H, "KEEP_ROOT", home / "live_tests"), mock.patch.object(H, "RESULTS", home / "results"), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(H.main(["--only", "hab-07", "--arms", "model_only,open", "--no-judge", "--arm-model", "other/model-x", "--yes", "--cap-eur", "1"]), 0)
        self.assertEqual(set(seen_models), {"other/model-x"})
        other = json.loads((home / "results" / "HAB-002_dev_unjudged.json").read_text(encoding="utf-8"))
        self.assertEqual(other["models"]["arms"], "other/model-x")
        record = json.loads((home / "results" / "HAB-001_dev.json").read_text(encoding="utf-8"))
        self.assertEqual((record["arms"], record["incomplete"], record["models"]["judge"]), (["model_only", "open"], None, H.JUDGE_MODEL))
        self.assertTrue(all(r["judged"] and r["accepted"] for rows in record["scores"].values() for r in rows))
        self.assertIsNotNone(record["spent_eur"])  # the run's total: the arms' and the judge's (the judge here is a free model)
        run_dir = next(p for p in sorted((home / "live_tests").iterdir()) if p.name.endswith("HAB-001"))
        docs = [json.loads(line) for p in sorted(run_dir.rglob("model_exchanges/**/*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sorted({(d["purpose"], d["model"] == H.JUDGE_MODEL) for d in docs}), [("exam_judge", True), ("exam_plain", False)])
        self.assertEqual((record["capture"]["ok"], record["capture"]["with_thinking"], record["capture"]["without_thinking"]), (4, 4, []))
        for name in ("manifest.json", "scorecard.csv", "usage.jsonl", "scripted.json"):
            self.assertTrue((run_dir / name).exists(), name)
        # the receipt says archived only when every file is in the snapshot with the same hash
        self.assertIsNone(H.receipt(run_dir)["archived"])
        snapshot = home / "snapshot"
        shutil.copytree(run_dir, snapshot / "live_tests" / run_dir.name, ignore=shutil.ignore_patterns("archive_receipt.json"))
        self.assertTrue(H.receipt(run_dir, snapshot)["archived"])
        (snapshot / "live_tests" / run_dir.name / "scorecard.csv").write_text("changed", encoding="utf-8")
        got = H.receipt(run_dir, snapshot)
        self.assertEqual((got["archived"], got["not_in_snapshot"]), (False, ["scorecard.csv"]))

    def test_a_run_no_model_serves_is_refused_and_stores_nothing(self):
        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, True)
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "OPENROUTER_API_KEY": "o", "DCLAB_BLOCK_PROVIDERS": ""}), mock.patch("dotenv.load_dotenv", lambda *a, **k: None), \
                mock.patch.object(H, "frozen", lambda: None), mock.patch.object(H, "code_state", lambda: {"commit": "abc", "dirty": [], "diff_sha256": None}), \
                mock.patch.object(H, "owner_allows", lambda cap: None), mock.patch.object(H, "KEEP_ROOT", home / "live_tests"), \
                mock.patch.object(H, "RESULTS", home / "results"), redirect_stdout(io.StringIO()), redirect_stderr(err):
            self.assertEqual(H.main(["--only", "hab-07", "--yes", "--cap-eur", "1"]), 2)
        self.assertIn("no model serves", err.getvalue())
        self.assertFalse((home / "results").exists())
        self.assertFalse(list((home / "live_tests").glob("*")) if (home / "live_tests").exists() else [])

    def test_the_owners_switch_and_monthly_cap_are_read_before_anything_is_sent(self):
        from dclab_rnd import control

        with mock.patch.object(control, "live_allowed", lambda home=None: False):
            self.assertIn("switched live model calls off", H.owner_allows(1.0))
        with mock.patch.object(control, "live_allowed", lambda home=None: True), mock.patch.dict(os.environ, {"DCLAB_WORKSPACE_MONTHLY_EUR": "0.000001"}):
            self.assertIn("would pass its cap", H.owner_allows(1.0))

    def test_a_judge_named_as_openrouter_names_it_runs_on_openrouter(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "o", "ANTHROPIC_API_KEY": "", "DCLAB_BLOCK_PROVIDERS": "", "OPENROUTER_BASE_URL": ""}):
            tier = H.judge_tier("qwen/qwen3.8-27b")
            self.assertEqual((tier["model"], tier["provider"], model_settings._host(tier["base_url"]), tier["name"]),
                             ("qwen/qwen3.8-27b", "openai", model_settings.OPENROUTER_HOST, model_settings.purpose("exam_judge").tier))
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "ANTHROPIC_API_KEY": "a"}):
            self.assertIsNone(H.judge_tier("qwen/qwen3.8-27b"))

    def test_no_judge_when_the_owner_blocked_the_provider_or_no_key_is_set(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "DCLAB_BLOCK_PROVIDERS": ""}):
            self.assertIsNone(H.judge_tier())
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "a", "DCLAB_BLOCK_PROVIDERS": "anthropic"}):
            self.assertIsNone(H.judge_tier("some-model-on-anthropic"))

    def test_a_claude_model_is_refused_as_an_arm_or_a_judge(self):
        """The owner's rule (2026-11-15): no Claude model in any test, arm or judge, at any provider."""
        for extra in (["--arm-model", "anthropic/claude-sonnet-5.5"], ["--judge-model", "claude-opus-5-5"]):
            err = io.StringIO()
            with mock.patch.object(H, "frozen", lambda: None), mock.patch.object(H, "code_state", lambda: {"commit": "abc", "dirty": [], "diff_sha256": None}), \
                    mock.patch.object(H, "owner_allows", lambda cap: None), mock.patch("dotenv.load_dotenv", lambda *a, **k: None), \
                    mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "o", "ANTHROPIC_API_KEY": "a", "DCLAB_BLOCK_PROVIDERS": ""}), \
                    mock.patch.object(H, "KEEP_ROOT", Path(tempfile.mkdtemp())), redirect_stdout(io.StringIO()), redirect_stderr(err):
                self.assertEqual(H.main(["--only", "hab-07", "--yes", "--cap-eur", "1", *extra]), 2)
            self.assertIn("no Claude model", err.getvalue())

    def test_thinking_kept_redacted_counts_as_thinking_kept(self):
        """Live, 2026-11-16: 31 of 59 calls had seen table values, kept their thinking redacted, and were counted as none."""
        with tempfile.TemporaryDirectory() as tmp:
            day = Path(tmp) / "model_exchanges" / "2026" / "11"
            day.mkdir(parents=True)
            rows = [{"purpose": "exam_plain", "outcome": "ok", "whole": True, "thinking": [{"text": "a summary"}]},
                    {"purpose": "intern", "outcome": "ok", "whole": False, "thought": True, "thinking_redacted": [{"text": "a redacted summary"}]},
                    {"purpose": "reporter", "outcome": "ok", "whole": False, "thought": False},
                    {"purpose": "intern", "outcome": "the provider refused the key"}]
            (day / "16.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            kept = H.capture(Path(tmp))
        self.assertEqual((kept["exchanges"], kept["ok"], kept["whole"]), (4, 3, 1))
        self.assertEqual((kept["with_thinking"], kept["thinking_redacted"], kept["without_thinking"]), (2, 1, ["reporter"]))


if __name__ == "__main__":
    unittest.main()
