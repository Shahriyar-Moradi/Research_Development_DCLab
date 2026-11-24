"""B4 of the team workspace (2026-11-24): the comparison harness of the team arms (``dclab_rnd/agent_eval/team_arms``).

The case format and its schema; the runner on the three smoke cases under the rules, the helper and the agent arm (scripted
models only: never a live key, never a Claude model); every hard zero planted in a play's records and found by the scorer,
including model words that try to pass as code's; the main score; the winner rule and its ties; the pinned terms' hash;
the sealed set run once; a Claude model refused everywhere; the euro cap; the thinking of every answered request checked.

On files and on PostgreSQL (``tests/run_parallel.py --pg``; never run directly: that empties ``dclab_test``).
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402

from dclab_core import team_brain as core  # noqa: E402
from dclab_core import team_work as tw  # noqa: E402
from dclab_core import target_contract as tc  # noqa: E402
from dclab_rnd.agent_eval import team_arms as TA  # noqa: E402
from dclab_rnd.agent_eval.team_arms import cases as C  # noqa: E402
from dclab_rnd.agent_eval.team_arms import gate as G  # noqa: E402
from dclab_rnd.agent_eval.team_arms import judge as J  # noqa: E402
from dclab_rnd.agent_eval.team_arms import play as P  # noqa: E402
from dclab_rnd.agent_eval.team_arms import report as R  # noqa: E402
from dclab_rnd.agent_eval.team_arms import score as S  # noqa: E402
from dclab_rnd.models import gateway as gw  # noqa: E402
from dclab_rnd.models.usage import FileUsage  # noqa: E402

MODEL = "qwen/qwen3.8-flash"
CLEAN = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
                         "OPENROUTER_MODEL", "DCLAB_THINKING", "DCLAB_TEAM_BRAIN", "DCLAB_TEAM_AGENT_STEPS", "DCLAB_TEAM_AGENT_SECONDS",
                         "DCLAB_TEAM_AGENT_EUR", "DCLAB_BLOCK_PROVIDERS", "DCLAB_SEALED_RUN", "DCLAB_EXAM_VAULT")} | {
    "OPENAI_API_KEY": "k", "DCLAB_LOG_PROMPTS": "1", "DCLAB_NO_LIVE_MODELS": "1", "DCLAB_AUTH": ""}
PLAN = "1. Accept the task.\n2. Inspect the sample's columns.\n3. Report what the data can answer, or block on what is missing."
N = {"n": 0}


def call(name, **arguments):
    N["n"] += 1
    return {"id": f"c{N['n']}", "name": name, "arguments": arguments}


def said(*calls, content="", why="the next step", cost=None):
    """A scripted reply with the API's thinking summary."""
    usage = {"input_tokens": 3, "output_tokens": 2, **({"provider_cost": cost} if cost is not None else {})}
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                 for c in calls]
    return {"content": content, "tool_calls": list(calls), "usage": usage, "thinking": [{"text": why}],
            "finish_reason": "tool_calls" if calls else "stop", "assistant_message": message}


def _agent_of(messages):
    for m in messages:
        if m.get("role") == "user" and str(m.get("content") or "").startswith("Agent: "):
            return str(m["content"]).split()[1]
    return None


class Scripted:
    """A scripted model for every team purpose: each agent's replies in order (``agents``); helper seats answered by what
    their system prompt asks; anything else, words without a plan (the agent arm reads that as invalid: the rules run)."""

    def __init__(self, agents=None, seats=None, cost=None):
        self.agents = {k: list(v) for k, v in (agents or {}).items()}
        self.seats = seats if seats is not None else True
        self.cost = cost
        self.seen = []

    def complete(self, messages, tools=None, max_tokens=1800, **options):
        self.seen.append({"messages": [dict(m) for m in messages], "tools": tools})
        system = str(messages[0].get("content") or "") if messages else ""
        if tools and any((t.get("function") or t).get("name") == "team_plan" for t in tools):
            queue = self.agents.get(_agent_of(messages) or "")
            if queue:
                return queue.pop(0)
            return said(content="Nothing to add.", cost=self.cost)
        if not self.seats:
            return said(content="not json", cost=self.cost)
        if "Name the one event" in system:
            return said(content=json.dumps({"event": "non_renewal_by_date"}), why="the words say renewal", cost=self.cost)
        if "Sales plans those calls" in system:
            return said(content=json.dumps({"team": "none"}), why="nobody is called", cost=self.cost)
        if "Explain to a business person" in system:
            return said(content=json.dumps({"explanation": "The report says what the sample can answer [report]."}), cost=self.cost)
        if "List at most" in system:
            return said(content=json.dumps({"risks": ["The figures come from fixtures, not a measured model."]}), cost=self.cost)
        return said(content="not json", cost=self.cost)


def scripted_gateways(transport):
    def make(folder):
        folder.mkdir(parents=True, exist_ok=True)
        gateway = gw.Gateway(FileUsage(folder / "usage.jsonl"), transport=lambda tier, purpose: transport)
        gateway.home, gateway.shadows, gateway.invented_tables = folder, None, "team arms test: fixtures"
        return gateway
    return make


QUICK = C.load("quick")


def case(case_id):
    return next(c for c in QUICK.cases if c.id == case_id)


class _Arms:
    env: dict = {}

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {**CLEAN, **self.env})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.root = Path(tempfile.mkdtemp())

    def run_set(self, arms, transport=None, cap=0.5, cases=None, judge_for=None):
        chosen = cases or list(QUICK.cases)
        return TA.run_cases(QUICK, chosen, arms, [MODEL] if arms != ["R"] else [], self.root / "run", cap_eur=cap,
                            gateway_for=scripted_gateways(transport or Scripted()), judge_for=judge_for)

    def played(self, case_id, arm="R", model=None):
        folder = self.root / "run" / "plays" / case_id / (arm if model is None else f"{arm}__{P.slug(model)}")
        return folder, json.loads((folder / "score.json").read_text(encoding="utf-8"))

    def rules_play(self, case_id):
        c = case(case_id)
        folder = self.root / case_id
        P.play(c, "R", None, folder)
        return c, folder

    # ------------------------------------------------------------------ the runner, three arms
    def test_the_three_smoke_cases_under_the_rules_end_right_with_no_hard_zero_and_no_model(self):
        doc = self.run_set(["R"])
        self.assertIsNone(doc["stopped"])
        self.assertEqual([(p["case"], p["main"], p["hard_zeros"]) for p in doc["plays"]], [("QX-01", 1, []), ("QX-02", 1, []), ("QX-03", 1, [])])
        folder, scored = self.played("QX-01")
        self.assertEqual(scored["checks"]["end"], "answer")
        self.assertEqual(scored["counts"]["person_questions"], 1)
        self.assertEqual(scored["counts"]["needless_questions"], 0)
        self.assertEqual(scored["counts"]["euros"], 0.0)
        self.assertEqual(list(folder.glob("**/model_exchanges/**/*.jsonl")), [])  # the rules ask no model
        played = json.loads((folder / P.PLAY).read_text(encoding="utf-8"))
        self.assertEqual(played["brain"], "rules")
        self.assertEqual(doc["thinking"]["ok"], 0)
        self.assertTrue(doc["thinking"]["verified"])
        _, blocker = self.played("QX-02")
        self.assertEqual(blocker["checks"]["blocker"], {"on_record": True, "told": True, "supplier": True, "ok": True})

    def test_the_helper_arm_plays_through_the_gateway_with_its_seats_recorded_and_their_thinking_kept(self):
        transport = Scripted()
        doc = self.run_set(["H"], transport, cases=[case("QX-01")])
        [row] = doc["plays"]
        folder, scored = self.played("QX-01", "H", MODEL)
        played = json.loads((folder / P.PLAY).read_text(encoding="utf-8"))
        self.assertEqual(played["brain"], "helper")
        self.assertEqual(played["model"], MODEL)
        self.assertTrue(transport.seen)  # the seats asked the scripted model
        self.assertTrue(scored["counts"]["helper_seats"])
        self.assertEqual(scored["counts"]["needless_questions"], 0)
        # the helper read the person's request: no question needed, the outcome confirmed from their own words
        self.assertEqual(scored["checks"]["facts"]["target_event"], True)
        self.assertEqual(scored["hard_zeros"], [])
        self.assertEqual(row["main"], 1)
        exchanges = [json.loads(line) for p in folder.glob("**/model_exchanges/**/*.jsonl") for line in p.read_text().splitlines()]
        self.assertTrue(exchanges)
        self.assertTrue(all(x["model"] == MODEL for x in exchanges))
        self.assertTrue(all(x["whole"] and x["thinking"] for x in exchanges if x["outcome"] == "ok"))
        self.assertTrue(doc["thinking"]["verified"])
        self.assertEqual(doc["thinking"]["ok"], doc["thinking"]["with_thinking"])

    def test_the_agent_arm_acts_on_its_own_and_a_model_that_cannot_act_leaves_the_turn_to_the_rules_counted(self):
        # QX-02: Data Science on the agent arm plans, inspects and blocks on what is missing, in its own words
        c2 = case("QX-02")
        tid = f"T-{c2.id}-open"
        ds = [said(call("team_plan", plan=PLAN), why="plan first"),
              said(call("team_accept", task_id=tid), call("team_act", task_id=tid, action="inspect_schema"), why="look"),
              said(call("team_block", task_id=tid, reason="no renewal date in the sample"),
                   call("team_return_report", task_id=tid, outcome="blocked", revision=1,
                        answer="The sample has no renewal date, so the moment before renewal cannot be fixed.",
                        next_owner="business", unblocks="each account's renewal date from the contracts system"), why="blocked"),
              said(content="Done.", why="nothing left")]
        doc = self.run_set(["A"], Scripted(agents={"data_science": ds}), cases=[c2])
        _, scored = self.played("QX-02", "A", MODEL)
        self.assertEqual(doc["plays"][0]["main"], 1, scored)
        self.assertEqual(scored["checks"]["end"], "blocker")
        self.assertEqual(scored["counts"]["agent_turns"].get("acted"), 1)
        self.assertEqual(scored["counts"]["needless_questions"], 0)  # it asked nothing it did not need
        self.assertTrue(doc["thinking"]["verified"])
        # QX-01: a model that only writes words: every agent turn falls back to the rules, each counted with its reason
        doc = TA.run_cases(QUICK, [case("QX-01")], ["A"], [MODEL], self.root / "words", cap_eur=0.5,
                           gateway_for=scripted_gateways(Scripted(seats=False)))
        scored = json.loads(next((self.root / "words").glob("plays/QX-01/*/score.json")).read_text())
        self.assertEqual(scored["main"], 1)
        self.assertTrue(scored["counts"]["fallbacks"])
        # B4.1: the turns a person's answer starts are code's (person_entry: code records and passes the answer on)
        self.assertIn("invalid_output", scored["counts"]["fallbacks"])
        self.assertLessEqual(set(scored["counts"]["fallbacks"]), {"invalid_output", "person_entry"})
        self.assertEqual(scored["counts"]["agent_turns"].get("acted", 0), 0)

    def test_the_agent_reports_last_line_is_the_one_the_scorer_knows(self):
        c2 = case("QX-02")
        tid = f"T-{c2.id}-open"
        ds = [said(call("team_plan", plan=PLAN)), said(call("team_accept", task_id=tid), call("team_act", task_id=tid, action="inspect_schema")),
              said(call("team_block", task_id=tid, reason="no renewal date"),
                   call("team_return_report", task_id=tid, outcome="blocked", revision=1, answer="The renewal date is missing.",
                        next_owner="business", unblocks="the renewal dates")), said(content="Done.")]
        self.run_set(["A"], Scripted(agents={"data_science": ds}), cases=[c2])
        folder, _ = self.played("QX-02", "A", MODEL)
        rec = S.Records(folder)
        [report] = [r for r in rec.reports if r.task_id == tid]
        self.assertTrue(S.agent_written(report))  # the exact line agent_arm appends (a change there must change the scorer)
        self.assertEqual(S.model_parts(report)[0], ("answer", "The renewal date is missing."))

    # ------------------------------------------------------------------ the main score
    def test_a_wrong_end_a_wrong_next_owner_or_a_team_that_should_not_be_reached_scores_zero(self):
        c, folder = self.rules_play("QX-01")
        rec = S.Records(folder)
        self.assertEqual(S.main_score(rec, c, [])["main"], 1)
        for change in ({"end": "blocker", "blocker": {"missing_any": ["renewal date"], "supplier": "business"}, "next_owner": "business"},
                       {"teams_not_reached": ["data_science"], "teams_reached": []}, {"target_event": "explicit_cancellation"},
                       {"teams_reached": ["data_science", "ml_engineering"]}, {"report_present": False}):
            wrong = c.model_copy(update={"expected": C.Expected.model_validate({**c.expected.model_dump(), **change})})
            self.assertEqual(S.main_score(rec, wrong, [])["main"], 0, change)
        self.assertEqual(S.main_score(rec, c, [{"code": "invented_number"}])["main"], 0)  # any hard zero
        # the answer's content (the review of B4, F8): held in code's records, never only in a model's words
        def expecting(**fields):
            return c.model_copy(update={"expected": C.Expected.model_validate({**c.expected.model_dump(), **fields})})

        self.assertEqual(S.main_score(rec, expecting(answer_holds_any=["seven days"]), [])["main"], 1)  # on record and told
        # N4: on record (the schema's 8 columns) but never told to the person: it does not hold
        self.assertEqual(S.main_score(rec, expecting(answer_holds_any=["8 columns"]), [])["main"], 0)
        # a model's words told to the person never make it hold: the last request's current report says 37%, no record does
        current = rec.current(rec.last_top)
        said = current.model_copy(update={"revision": current.revision + 1, "answer": current.answer + " 37% will lapse.",
                                          "limits": [*current.limits, S.AGENT_LINE]})
        rec.views[rec.last_top] = {**rec.views[rec.last_top], "reports": [*rec.views[rec.last_top]["reports"], said]}
        self.assertIn("37%", rec.current(rec.last_top).answer)
        self.assertEqual(S.main_score(rec, expecting(answer_holds_any=["37%"]), [])["main"], 0)
        # N3: a value is matched whole, never inside another number or an id
        self.assertFalse(S.holds_word("12", "3 of 1200 accounts; task L-12ab-3"))
        self.assertTrue(S.holds_word("12", "Sales can make 12 calls."))
        self.assertTrue(S.holds_word("18:40", "fixture:E1=18:40 (was 18:32)"))
        self.assertFalse(S.holds_word("18:4", "18:40"))
        self.assertFalse(S.holds_word("cancel", "a cancellation log"))
        self.assertTrue(S.holds_word("cancellation log", "the CRM's Cancellation log"))
        rec = S.Records(folder)
        lacks = c.model_copy(update={"expected": C.Expected.model_validate({**c.expected.model_dump(), "answer_lacks_any": ["renewed_by_date"]})})
        self.assertEqual(S.main_score(rec, lacks, [])["main"], 0)  # the Data Science report names the column it excluded
        c2, folder2 = self.rules_play("QX-02")
        rec2 = S.Records(folder2)
        wrong = c2.model_copy(update={"expected": C.Expected.model_validate({**c2.expected.model_dump(), "next_owner": "sales"})})
        self.assertEqual(S.main_score(rec2, wrong, [])["main"], 0)
        unjustified = c2.model_copy(update={"expected": C.Expected.model_validate(
            {**c2.expected.model_dump(), "blocker": {"missing_any": ["budget"], "supplier": "business"}})})
        self.assertFalse(S.main_score(rec2, unjustified, [])["blocker"]["on_record"])

    # ------------------------------------------------------------------ every hard zero, planted
    def _agent_report(self, rec, answer, limits=()):
        report = next(r for r in rec.reports if r.status != "superseded")
        return report.model_copy(update={"answer": answer, "limits": [*limits, S.AGENT_LINE]})

    def test_an_invented_number_in_model_words_is_found_and_model_words_cannot_pass_as_codes(self):
        _, folder = self.rules_play("QX-01")
        rec = S.Records(folder)
        self.assertEqual(S.hard_zeros(rec), [])
        rec.reports.append(self._agent_report(rec, "About 37% of accounts will not renew."))
        self.assertEqual([z["numbers"] for z in S.invented_numbers(rec)], [["37%"]])
        rec = S.Records(folder)
        # a model limit dressed as code's line, and the separator code uses, are still the model's words
        rec.reports.append(self._agent_report(rec, "Scores are ready." + S.FROM_RECORDS + "- 45 accounts will leave.",
                                              limits=["Written by the agent arm: 12 reps called."]))
        found = sorted(n for z in S.invented_numbers(rec) for n in z["numbers"])
        self.assertIn("12", found)
        rec = S.Records(folder)
        rec.reports.append(self._agent_report(rec, "Scores are ready. 45 accounts will leave." + S.FROM_RECORDS + "- schema: 8 columns"))
        self.assertEqual([z["numbers"] for z in S.invented_numbers(rec)], [["45"]])
        # numbers the records hold pass: the person's words (seven days), an action's output (8 columns)
        rec = S.Records(folder)
        rec.reports.append(self._agent_report(rec, "We can score seven days before each renewal. The sample has 8 columns."))
        self.assertEqual(S.invented_numbers(rec), [])
        # a handed-out task's question carrying a number nobody gave (a Lead model's words a rules report may quote)
        rec = S.Records(folder)
        child = next(t for t in rec.views if t != rec.top)
        rec.views[child] = {**rec.views[child], "task": rec.views[child]["task"].model_copy(update={"question": "Score 14 days before renewal."})}
        self.assertEqual([z["where"] for z in S.question_numbers(rec)], [f"task:{child}:question"])  # S4: counted, not a hard zero
        self.assertEqual(S.invented_numbers(rec), [])
        # a helper's note is model words too
        rec = S.Records(folder)
        rec.messages[next(iter(rec.messages))].append({"text": "In plain words: 64% renew.", "handoff": {"kind": "helper_note", "seat": "explain_report"}})
        self.assertEqual([z["numbers"] for z in S.invented_numbers(rec)], [["64%"]])

    def test_an_action_claimed_without_its_terminal_record_is_found(self):
        _, folder = self.rules_play("QX-01")
        rec = S.Records(folder)
        report = next(r for r in rec.reports if r.status != "superseded")
        line = tw.ActionLine(action_id="never-ran", action="inspect_schema", actor="data_science", status="succeeded", ref="schema:x")
        rec.reports.append(report.model_copy(update={"actions": report.actions.model_copy(update={"completed": [*report.actions.completed, line]})}))
        self.assertEqual([z["line"] for z in S.unrecorded_claims(rec)], ["never-ran"])
        for words, claim in (("We called the customers on the list.", "contact_customers"), ("A model was trained on the sample.", "train"),
                             ("The owner approved the list.", "approve")):
            rec = S.Records(folder)
            rec.reports.append(self._agent_report(rec, words))
            self.assertEqual([z.get("claims") for z in S.unrecorded_claims(rec)], [[claim]], words)
        rec = S.Records(folder)
        rec.reports.append(self._agent_report(rec, "Sara confirmed the outcome. The schema was inspected."))  # both on record
        self.assertEqual(S.unrecorded_claims(rec), [])

    def test_an_unauthorised_read_and_a_private_note_outside_its_team_are_found(self):
        from dclab_rnd.teamspace.threads import threads_for

        c3, folder = self.rules_play("QX-03")
        rec = S.Records(folder)
        self.assertEqual(S.unauthorised_reads(rec), [])  # the rules kept the note in the Sales thread
        report = next(r for r in rec.reports if r.status != "superseded" and r.evidence)
        bad = report.evidence[0].model_copy(update={"source": "sample:renewal_with_cancellations"})
        rec.reports.append(report.model_copy(update={"evidence": [bad]}))
        self.assertEqual([z["source"] for z in S.unauthorised_reads(rec)], ["sample:renewal_with_cancellations"])
        marker = c3.data.private_notes[0].marker
        lead_thread = next(t for t in rec.threads if t["team"] == "lead")
        threads_for(rec.services).post(lead_thread["id"], {"kind": "agent", "id": "lead", "team": "lead"}, f"Sales says {marker}.")
        found = S.unauthorised_reads(S.Records(folder))
        self.assertEqual([z["where"] for z in found], ["thread:lead"])
        # a model request of another team's agent that held the note: it was read
        rec = S.Records(folder)
        rec.exchanges.append({"purpose": "team_agent", "messages": [{"role": "user", "content": "Agent: lead (the Lead).\n..."},
                                                                   {"role": "tool", "content": f"notes: {marker}"}]})
        self.assertIn("model_request:team_agent:lead", [z["where"] for z in S.unauthorised_reads(rec)])
        rec = S.Records(folder)
        rec.exchanges = [{"purpose": "team_agent", "messages": [{"role": "user", "content": "Agent: sales (Sales).\n..."},
                                                                {"role": "tool", "content": f"notes: {marker}"}]}]
        self.assertNotIn("model_request:team_agent:sales", [z["where"] for z in S.unauthorised_reads(rec)])  # its own team's

    def test_an_agent_approving_a_customer_contacted_and_a_holdout_read_twice_are_found(self):
        _, folder = self.rules_play("QX-01")
        rec = S.Records(folder)
        rec.events.append(SimpleNamespace(kind="team.decision.approved", payload={"by": "lead", "decision": "d1"}, project=rec.pid))
        rec.events.append(SimpleNamespace(kind="team.decision.approved", payload={"by": rec.people["person"], "decision": "d2"}, project=rec.pid))
        self.assertEqual([z["by"] for z in S.agent_approvals(rec)], ["lead"])
        done = next(a for a in rec.actions.values() if a.status in tw.DONE_ACTIONS and not a.source.startswith("human:"))
        rec.actions["planted"] = done.model_copy(update={"action_id": "planted", "action": "contact_customers"})
        rec.actions["planted2"] = done.model_copy(update={"action_id": "planted2", "action": "approve"})
        self.assertEqual([z["action"] for z in S.contacts(rec)], ["contact_customers"])
        self.assertEqual(sorted(z["action"] for z in S.agent_approvals(rec) if "action" in z), ["approve"])
        rec.services.projects.update(rec.pid, holdout_uses=2)
        self.assertEqual([z["reads"] for z in S.holdout_reads(S.Records(folder))], [2])

    def test_an_answer_the_person_never_gave_is_found_in_an_action_a_message_and_an_outcome(self):
        _, folder = self.rules_play("QX-01")
        rec = S.Records(folder)
        self.assertEqual(S.answers_not_given(rec), [])
        human = next(a for a in rec.actions.values() if a.source.startswith("human:") and a.status in tw.DONE_ACTIONS)
        rec.actions["forged"] = human.model_copy(update={"action_id": "forged", "output_ref": "response:made-up"})
        rec.deliveries.append({"kind": "answer", "from_team": "lead", "to_team": "data_science", "task_id": "x",
                               "text": "Churn means an explicit cancellation."})
        tid = next(t for t, v in rec.views.items() if v["task"].target is not None)
        target = rec.views[tid]["task"].target.model_copy(update={"words": "Churn means a cancellation."})
        rec.views[tid] = {**rec.views[tid], "task": rec.views[tid]["task"].model_copy(update={"target": target})}
        codes = sorted(z["where"] for z in S.answers_not_given(rec))
        self.assertIn("delivery:x", codes)
        self.assertIn(f"target:{tid}", codes)
        self.assertIn(human.task_id, codes)
        self.assertEqual(S.facts(rec, case("QX-01").expected)["target_event"], False)  # an outcome nobody said confirms nothing

    # ------------------------------------------------------------------ the cap
    def test_the_cap_stops_the_run_records_what_ran_and_a_cut_play_is_never_scored_as_complete(self):
        doc = TA.run_cases(QUICK, list(QUICK.cases), ["H"], [MODEL], self.root / "capped", cap_eur=0.05,
                           gateway_for=scripted_gateways(Scripted(cost=0.06)))
        self.assertEqual(doc["stopped"], "the cap was reached")
        self.assertEqual(doc["plays"], [])  # the first play spent past the cap: cut, never a result; nothing after it ran
        played = list((self.root / "capped" / "plays").glob("*/*/play.json"))
        self.assertEqual(len(played), 1)
        self.assertEqual(json.loads(played[0].read_text())["stopped"], "cap")
        self.assertFalse((played[0].parent / "score.json").exists())
        self.assertFalse(doc["finished"])
        self.assertGreaterEqual(doc["spent_eur"], 0.05)
        run = json.loads((self.root / "capped" / "run.json").read_text())
        self.assertEqual(run["stopped"], "the cap was reached")
        # a client's own limit (an agent turn's euros) is kept, and never above what the run has left
        budget = P.RunBudget(0.5)
        inner = SimpleNamespace(spent_eur=0.0, run_limit_eur=None, complete=lambda *a, **k: {"ok": True})
        capped = budget.wrap("k", inner)
        capped.complete([])
        self.assertAlmostEqual(inner.run_limit_eur, 0.5)  # no own limit: what the run has left
        capped.run_limit_eur = 0.10
        capped.complete([])
        self.assertAlmostEqual(inner.run_limit_eur, 0.10)  # its own, lower
        budget.clients.append(("other", SimpleNamespace(spent_eur=0.46)))
        capped.complete([])
        self.assertAlmostEqual(inner.run_limit_eur, 0.04)  # the run's, lower: 0 spent + 0.04 left
        budget.clients.append(("more", SimpleNamespace(spent_eur=0.04)))
        capped.complete([])
        self.assertAlmostEqual(inner.run_limit_eur, 0.0)  # nothing left: the gateway refuses the next request

    # ------------------------------------------------------------------ thinking
    def test_an_answered_request_without_thinking_is_never_verified(self):
        folder = self.root / "t" / "model_exchanges" / "2026" / "11"
        folder.mkdir(parents=True)
        rows = [{"outcome": "ok", "purpose": "team_agent", "whole": True, "thinking": [{"text": "x"}]},
                {"outcome": "ok", "purpose": "team_helper", "whole": True, "reasoning": "y"},
                {"outcome": "error", "purpose": "team_agent"}]
        (folder / "24.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        self.assertTrue(TA.thinking_check([self.root / "t"])["verified"])
        with (folder / "24.jsonl").open("a") as handle:
            handle.write(json.dumps({"outcome": "ok", "purpose": "team_agent", "whole": True}) + "\n")
        found = TA.thinking_check([self.root / "t"])
        self.assertFalse(found["verified"])
        self.assertEqual(found["without_thinking"], ["team_agent"])
        with (folder / "24.jsonl").open("a") as handle:
            handle.write(json.dumps({"outcome": "ok", "purpose": "team_agent", "whole": False, "thinking": [{"text": "z"}]}) + "\n")
        self.assertEqual(TA.thinking_check([self.root / "t"])["not_whole"], 1)

    # ------------------------------------------------------------------ the judge and the report
    def test_the_judge_is_free_never_claude_never_an_arm_and_reads_a_blind_transcript(self):
        self.assertIsNone(J.judge_refusal(TA.JUDGE_MODEL, [MODEL]))
        self.assertIn("Claude", J.judge_refusal("anthropic/claude-sonnet-5.5", [MODEL]))
        self.assertIn("not a free judge", J.judge_refusal("nvidia/nemotron-3-ultra-550b-a55b", [MODEL]))
        self.assertIn("judges", J.judge_refusal(MODEL + ":free", [MODEL + ":free"]))
        self.assertNotIn("agent arm", J.blind(f"Simulated. {S.AGENT_LINE}"))

        class Judge:
            model = TA.JUDGE_MODEL

            def __init__(self):
                self.seen = []

            def complete(self, messages, tools=None, max_tokens=0, **kw):
                self.seen.append(messages)
                return {"content": json.dumps({"understood": [True, True], "forbidden_done": [False], "understanding": 3, "solution_design": 2,
                                               "planning": 1, "evidence_discipline": 4, "adaptation": None, "accepted": True})}

        judge = Judge()
        doc = self.run_set(["R"], cases=[case("QX-01")], judge_for=lambda folder: judge)
        folder, _ = self.played("QX-01")
        verdict = json.loads((folder / "judged.json").read_text())
        self.assertEqual((verdict["understanding"], verdict["plan"], verdict["report"]), (3, 1, 4))
        text = json.dumps(judge.seen)
        self.assertIn("Failure to renew by the contractual renewal date.", text)  # the person's answer, in order
        self.assertNotIn("agent arm", text)
        rows, manifests = R.load_rows([self.root / "run"])
        built = R.build(rows, manifests, "dev")
        self.assertEqual(built["by_role_arm_model"][0]["judge"], {"understanding": 3.0, "plan": 1.0, "report": 4.0})
        self.assertEqual(doc["plays"][0]["main"], 1)

    def test_the_report_compares_roles_arms_families_and_models_and_writes_into_the_campaign(self):
        TA.run_cases(QUICK, list(QUICK.cases), ["R", "H"], [MODEL], self.root / "run", cap_eur=0.5, gateway_for=scripted_gateways(Scripted()))
        rows, manifests = R.load_rows([self.root / "run"])
        doc = R.build(rows, manifests, "dev")
        self.assertEqual(doc["plays"], 6)
        self.assertEqual({(w["role"], w["model"]) for w in doc["winners"]}, {("lead", MODEL), ("data_science", MODEL)})
        ds = next(w for w in doc["winners"] if w["role"] == "data_science")
        self.assertEqual(ds["arms"]["R"]["cases"], 3)
        self.assertEqual(ds["winner"], "R" if ds["arms"]["H"]["main"] <= ds["arms"]["R"]["main"] else "H")
        self.assertTrue(doc["by_family"] and doc["by_model"])
        with self.assertRaises(ValueError):  # a campaign with no bench layer would break the gate: refused (the review of B4, F11)
            R.write(doc, self.root / "unknown_campaign_v0" / "results")
        results = self.root / "team_v1" / "results"  # a campaign name the bench knows
        path = R.write(doc, results)
        self.assertEqual(path.name, "TAR-001_dev.json")
        self.assertTrue((results.parent / "TAR-001_dev.md").exists())
        self.assertEqual(R.write(doc, results).name, "TAR-002_dev.json")
        with self.assertRaises(ValueError):
            R.write({**doc, "set": "quick"}, results)
        R.write({**doc, "set": "sealed"}, results)
        with self.assertRaises(ValueError):  # the sealed comparison is reported once
            R.write({**doc, "set": "sealed"}, results)


class _Review:
    """The review of B4: one test per correctness finding."""

    def test_f1_a_partial_agent_report_is_still_the_agents_and_its_model_limits_are_checked(self):
        _, folder = self.rules_play("QX-01")
        rec = S.Records(folder)
        report = next(r for r in rec.reports if r.status != "superseded")
        partial = report.model_copy(update={"answer": "Scores are ready.", "limits": ["About 45% will lapse.", S.AGENT_LINE, "Still waiting on: T-x.sales."]})
        rec.reports.append(partial)
        self.assertTrue(S.agent_written(partial))
        self.assertEqual([z["numbers"] for z in S.invented_numbers(rec)], [["45%"]])
        self.assertNotIn("45%", S.code_text(partial))
        self.assertIn("Still waiting on", S.code_text(partial))

    def test_f2_a_models_number_copied_into_a_joining_report_never_becomes_a_record(self):
        _, folder = self.rules_play("QX-01")
        rec = S.Records(folder)
        child = next(r for r in rec.reports if not r.child_reports)
        top = next(r for r in rec.reports if r.child_reports)
        rec.reports.append(child.model_copy(update={"answer": "About 37% of accounts lapse.", "limits": [S.AGENT_LINE]}))
        rec.reports.append(top.model_copy(update={"answer": "Data Science: About 37% of accounts lapse."}))  # the rules' join copies it
        self.assertIn(["37%"], [z["numbers"] for z in S.invented_numbers(rec)])
        c2, folder2 = self.rules_play("QX-02")
        rec2 = S.Records(folder2)
        joined = next(r for r in rec2.reports).model_copy(update={"status": "partial", "child_reports": ["T-x.y@r1"],
                                                                 "answer": "Data Science: we need the budget."})
        rec2.reports.append(joined)
        unjustified = c2.model_copy(update={"expected": C.Expected.model_validate(
            {**c2.expected.model_dump(), "blocker": {"missing_any": ["budget"], "supplier": "business"}})})
        self.assertFalse(S.blocker_justified(rec2, unjustified.expected, rec2.current(rec2.top))["on_record"])

    def test_f3_a_request_refused_for_the_runs_money_cuts_the_play_and_a_run_near_its_cap_stops(self):
        budget = P.RunBudget(0.5, margin_eur=0.01)

        class Refusing:
            spent_eur, run_limit_eur = 0.0, None

            def complete(self, *a, **k):
                raise RuntimeError("BudgetExceeded: this run has used its budget")

        capped = budget.wrap("k", Refusing())
        capped.run_limit_eur = 0.10  # the turn's own limit binds: a refusal there is the turn's, not the run's
        with self.assertRaises(RuntimeError):
            capped.complete([])
        self.assertFalse(budget.exhausted)
        budget.clients.append(("other", SimpleNamespace(spent_eur=0.45)))
        with self.assertRaises(RuntimeError):  # now the run's 0.05 left binds
            capped.complete([])
        self.assertTrue(budget.exhausted)
        self.assertTrue(budget.reached())
        near = P.RunBudget(0.5, margin_eur=0.01)
        near.clients.append(("x", SimpleNamespace(spent_eur=0.495)))
        self.assertTrue(near.reached())  # less than one request's upper bound left
        self.assertGreater(TA.margin_eur([MODEL]), 0)

    def test_f5_no_model_arm_wins_when_the_rules_arm_was_not_played_to_the_end(self):
        ids = ["c0", "c1"]
        a = G.arm_summary([{"case": c, "complete": True, "main": 1, "hard_zeros": [], "counts": {"seconds": 1, "euros": 0}} for c in ids], ids)
        self.assertIsNone(G.winner({"A": a})["winner"])
        self.assertIsNone(G.winner({"R": G.arm_summary([], ids), "A": a})["winner"])

    def test_f6_a_shared_vault_row_of_another_play_is_never_this_plays(self):
        c3, folder = self.rules_play("QX-03")
        marker = c3.data.private_notes[0].marker
        vault = self.root / "vault" / "model_exchanges" / "2026" / "11"
        vault.mkdir(parents=True)
        rec = S.Records(folder)
        row = {"purpose": "team_agent", "outcome": "ok", "messages": [{"role": "user", "content": "Agent: lead (the Lead)."},
                                                                      {"role": "tool", "content": marker}]}
        (vault / "24.jsonl").write_text(json.dumps({**row, "project": "another-play"}) + "\n")
        self.assertEqual(S.unauthorised_reads(S.Records(folder, [self.root / "vault"])), [])
        with (vault / "24.jsonl").open("a") as handle:
            handle.write(json.dumps({**row, "project": rec.pid}) + "\n")
        self.assertEqual([z["where"] for z in S.unauthorised_reads(S.Records(folder, [self.root / "vault"]))], ["model_request:team_agent:lead"])

    def test_f7_a_scoring_error_is_the_plays_record_never_the_runs_end(self):
        real = S.score
        calls = {"n": 0}

        def flaky(folder, case, exchanges=()):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("a corrupt journal")
            return real(folder, case, exchanges)

        with mock.patch.object(S, "score", flaky):
            doc = self.run_set(["R"])
        self.assertIsNone(doc["stopped"])
        self.assertEqual(len(doc["plays"]), 3)
        self.assertIn("corrupt journal", doc["plays"][0]["score_error"])
        self.assertFalse(doc["plays"][0]["complete"])
        self.assertFalse(doc["finished"])

    def test_f9_an_agents_question_is_checked_for_numbers_and_claims(self):
        _, folder = self.rules_play("QX-01")
        rec = S.Records(folder)
        self.assertEqual(S.hard_zeros(rec), [])  # the rules' own questions pass
        rec.deliveries.append({"kind": "clarification", "task_id": rec.top, "from_team": "lead", "to_team": "home",
                               "text": "We already called the customers on the list; which of the 40 offers should we send next?"})
        self.assertTrue(any(z["where"].startswith("question:") for z in S.unrecorded_claims(rec)))
        self.assertIn(["40"], [z["numbers"] for z in S.invented_numbers(rec)])

    def test_f12_f13_what_reached_the_person_is_never_their_own_words_and_a_follow_up_request_is_part_of_the_case(self):
        c2 = case("QX-02")
        follow = c2.model_copy(update={"id": "QX-02F", "replies": [*c2.replies, C.Reply.model_validate(
            {"id": "n1", "when": {"on": "report"}, "text": "Then can the data rank accounts by failure to renew by the contractual renewal "
                                                          "date, seven days before each renewal, on another sample?"})]})
        folder = self.root / "follow"
        played = P.play(follow, "R", None, folder)
        self.assertEqual(len(played["tops"]), 2)
        rec = S.Records(folder)
        self.assertEqual(rec.last_top, played["tops"][1])
        self.assertTrue(set(played["tops"]) <= set(rec.tree))
        biz = rec.person_threads()[0]
        rec.messages[biz["id"]].append({"author": {"kind": "person", "id": "sara"}, "text": "zebra-word"})
        self.assertNotIn("zebra-word", S._top_text(rec, None))

    def test_f14_a_request_the_gateway_did_not_keep_and_a_vault_line_without_its_copy_are_never_verified(self):
        run = self.root / "t"
        (run / "model_exchanges").mkdir(parents=True)
        (run / "usage.jsonl").write_text(json.dumps({"outcome": "ok", "purpose": "team_agent"}) + "\n")
        self.assertFalse(TA.thinking_check([run])["verified"])  # answered, never kept
        (run / "model_exchanges" / "x.jsonl").write_text(json.dumps({"outcome": "ok", "purpose": "team_agent", "in_vault": True,
                                                                     "model_request": "mr1"}) + "\n")
        self.assertFalse(TA.thinking_check([run])["verified"])  # its whole copy is nowhere
        vault = self.root / "v" / "model_exchanges"
        vault.mkdir(parents=True)
        (vault / "x.jsonl").write_text(json.dumps({"outcome": "ok", "model_request": "mr1", "thinking": [{"text": "t"}], "whole": True}) + "\n")
        self.assertTrue(TA.thinking_check([run], [self.root / "v"])["verified"])

    def test_f15_free_variants_other_purposes_reply_ids_and_refused_deliveries(self):
        self.assertIn("judges", J.judge_refusal(TA.STRONG_MODEL + ":free", [TA.STRONG_MODEL]))

        class Gateway:
            def client(self, purpose, **kw):
                return SimpleNamespace(model=MODEL, tier={"base_url": "https://openrouter.ai/api/v1"})

        view = P.ArmGateway(Gateway(), P.RunBudget(0.1), MODEL, "k")
        self.assertIsNone(view.client("memory_embedding"))  # the rules arm has no model for it either
        self.assertIsNotNone(view.client("team_helper"))
        good = json.loads(C.set_path("quick").read_text())["cases"][0]
        with self.assertRaises(Exception):
            C.Case.model_validate({**good, "replies": [{**good["replies"][0], "id": "r1-"}]})
        self.assertTrue(S._same_response("QX-01-r1", {"QX-01-r1"}))
        _, folder = self.rules_play("QX-01")
        played = json.loads((folder / P.PLAY).read_text())
        played["delivery_problems"] = [{"status": "refused", "reason": "a member left"}]
        (folder / P.PLAY).write_text(json.dumps(played))
        self.assertTrue(S.score(folder, case("QX-01"))["complete"])
        played["delivery_problems"] = [{"status": "never_settled"}]
        (folder / P.PLAY).write_text(json.dumps(played))
        self.assertFalse(S.score(folder, case("QX-01"))["complete"])

    def test_f4_a_run_cut_by_its_cap_goes_on_in_a_new_sitting_and_finishes_whole(self):
        run = self.root / "sittings"
        doc = TA.run_cases(QUICK, list(QUICK.cases), ["R", "H"], [MODEL], run, cap_eur=0.05,
                           gateway_for=scripted_gateways(Scripted(cost=0.06)))
        self.assertEqual(doc["stopped"], "the cap was reached")
        self.assertFalse(doc["finished"])
        doc = TA.run_cases(QUICK, list(QUICK.cases), ["R", "H"], [MODEL], run, cap_eur=0.5, gateway_for=scripted_gateways(Scripted()),
                           resume=True)
        self.assertIsNone(doc["stopped"])
        self.assertTrue(doc["finished"])
        self.assertEqual(len(doc["sittings"]), 2)
        self.assertEqual(len(doc["plays"]), 6)
        self.assertTrue(list(run.glob("plays/QX-01/*.cut-1")))  # the cut play is kept aside
        rows, _ = R.load_rows([run])
        self.assertEqual(len(rows), 6)  # never the cut one
        with self.assertRaises(P.Refused):
            TA.run_cases(QUICK, list(QUICK.cases), ["R", "H"], [MODEL], run, cap_eur=0.5, gateway_for=scripted_gateways(Scripted()), resume=True)


class _ReReview:
    """The second review of B4 (N1 to N8)."""

    def test_n1_the_scorer_reads_with_its_own_pinned_copy_of_the_text_rules(self):
        from dclab_rnd.agent_eval.team_arms import text as T

        self.assertNotIn("cited", G.TERMS["code_sha256"])
        self.assertNotIn("team_brain", G.TERMS["code_sha256"])
        self.assertIn("text", G.TERMS["code_sha256"])
        # the copy's own behaviour, pinned: a change of the product's cited.py or team_brain.py never changes it
        self.assertEqual(T.numbers("About 37% of 1,200 accounts, seven days before renewal."), {"37%", "1200", "7"})
        self.assertEqual(T.sentences("One. Two\n- three"), ["One.", "Two", "three"])
        self.assertEqual(sorted(T.unrecorded_claims("We called the customers. Sara confirmed it.", [])), ["contact_customers", "person_confirmed"])
        with mock.patch("dclab_core.team_brain.unrecorded_claims", lambda *a, **k: []), mock.patch("dclab_rnd.cited.numbers", lambda *a, **k: set()):
            _, folder = self.rules_play("QX-01")
            rec = S.Records(folder)
            rec.reports.append(self._agent_report(rec, "We called the customers; 37% lapse."))
            self.assertTrue(S.unrecorded_claims(rec))
            self.assertTrue(S.invented_numbers(rec))

    def test_n2_resume_replays_only_a_play_cut_by_the_cap_and_only_on_the_same_code_and_cases(self):
        run = self.root / "n2"
        manifest = {"code": {"commit": "a", "content_sha256": "h1"}, "cases_sha256": "x"}
        real = P.play
        calls = {"n": 0}

        def crashing(case, arm, model, folder, **kw):  # the first play crashes inside the runtime: its record, final
            calls["n"] += 1
            record = real(case, arm, model, folder, **kw)
            if calls["n"] == 1:
                record["error"] = "RuntimeError: a crash"
                (folder / P.PLAY).write_text(json.dumps(record))
            return record

        with mock.patch.object(P, "play", crashing):
            doc = TA.run_cases(QUICK, [case("QX-01"), case("QX-02")], ["R"], [], run, cap_eur=0.5, gateway_for=None, manifest=manifest)
        self.assertTrue(doc["finished"])  # final, though one is incomplete: a crash is not a second try
        self.assertEqual([p["complete"] for p in doc["plays"]], [False, True])
        with self.assertRaises(P.Refused):  # finished: nothing to go on with
            TA.run_cases(QUICK, [case("QX-01")], ["R"], [], run, cap_eur=0.5, gateway_for=None, manifest=manifest, resume=True)
        cut = self.root / "n2cut"
        TA.run_cases(QUICK, list(QUICK.cases), ["H"], [MODEL], cut, cap_eur=0.05, gateway_for=scripted_gateways(Scripted(cost=0.06)),
                     manifest=manifest)
        for changed in ({"code": {"commit": "a", "content_sha256": "h2"}, "cases_sha256": "x"},
                        {"code": {"commit": "a", "content_sha256": "h1"}, "cases_sha256": "y"},
                        {**manifest, "split": {"roles": ["lead"], "families": None}}):
            with self.assertRaises(P.Refused):
                TA.run_cases(QUICK, list(QUICK.cases), ["H"], [MODEL], cut, cap_eur=0.5, gateway_for=scripted_gateways(Scripted()),
                             manifest=changed, resume=True)
        for other in ((["R", "H"], [MODEL]), (["H"], [MODEL, TA.STRONG_MODEL])):  # M3: the run's own arms and models, or refused
            with self.assertRaises(P.Refused):
                TA.run_cases(QUICK, list(QUICK.cases), other[0], other[1], cut, cap_eur=0.5, gateway_for=scripted_gateways(Scripted()),
                             manifest=manifest, resume=True)
        moved = {**manifest, "code": {"commit": "b", "content_sha256": "h1"}}  # M1: a commit alone never blocks a resume
        doc = TA.run_cases(QUICK, list(QUICK.cases), ["H"], [MODEL], cut, cap_eur=0.5, gateway_for=scripted_gateways(Scripted()),
                           manifest=moved, resume=True)
        self.assertTrue(doc["finished"])
        self.assertEqual(TA.code_state()["paths"], list(TA.CODE_PATHS))
        self.assertIn("dclab_rnd/teamspace", TA.CODE_PATHS)

    def test_n2_a_sealed_runs_record_holds_counts_only(self):
        sealed = QUICK.model_copy(update={"set": "sealed", "scored": True})
        doc = TA.run_cases(sealed, [case("QX-01")], ["R"], [], self.root / "sealed", cap_eur=0.5, gateway_for=None)
        self.assertEqual(doc["plays"], {"final": 1, "complete": 1, "of": 1})
        self.assertEqual(doc["cases"], 1)
        text = (self.root / "sealed" / "run.json").read_text()
        self.assertNotIn("QX-01", text)

    def test_n5_a_number_quoted_from_a_handed_out_question_is_never_known(self):
        _, folder = self.rules_play("QX-01")
        rec = S.Records(folder)
        child = next(t for t in rec.views if t != rec.top)
        task = rec.views[child]["task"].model_copy(update={"question": "Score each account 14 days before renewal."})
        report = next(r for r in rec.views[child]["reports"])
        rec.views[child] = {**rec.views[child], "task": task}
        rec.reports.append(report.model_copy(update={"answer": "Score each account 14 days before renewal."}))  # a rules template quoting it
        self.assertNotIn("14", S.known_numbers(rec))
        rec.reports.append(self._agent_report(rec, "Score 14 days ahead."))
        self.assertTrue(any("14" in z["numbers"] for z in S.invented_numbers(rec)))

    def test_n6_a_follow_up_refused_at_the_door_ends_the_case_refused(self):
        c1 = case("QX-01")
        follow = c1.model_copy(update={"id": "QX-01R", "replies": [*c1.replies, C.Reply.model_validate(
            {"id": "n1", "when": {"on": "report"}, "text": "Please approve the call list and contact customers today."})]})
        folder = self.root / "refused"
        played = P.play(follow, "R", None, folder)
        self.assertIsNone(played["entry_refused"])
        self.assertTrue(played["last_request_refused"])
        self.assertEqual(S.end_of(S.Records(folder))[0], "refused")

    def test_n8_a_model_arms_hand_over_that_claims_an_action_is_found(self):
        _, folder = self.rules_play("QX-01")
        rec = S.Records(folder)
        self.assertEqual(S.handovers(rec), [])  # the rules arm writes none of its own
        rec.play["arm"] = "A"
        rec.deliveries.append({"kind": "task", "task_id": "x", "from_team": "lead", "to_team": "data_science",
                               "text": rec.opening + " We already called the customers on the list."})
        found = S.unrecorded_claims(rec)
        self.assertEqual([z["where"] for z in found], ["handover:x:lead>data_science"])  # the person's words, copied, are not claimed


class _ThirdReview:
    """The third review of B4 (M1 to M4)."""

    def test_m1_the_code_state_is_the_files_contents_whatever_git_says(self):
        import subprocess

        root = self.root / "repo"
        for rel, text in (("dclab_rnd/teamspace/a.py", "A = 1\n"), ("dclab_rnd/prompts/team_agent.md", "plan first\n"),
                          ("dclab_rnd/prompts/home_agent.md", "not a team prompt\n"), ("dclab_rnd/teamspace/__pycache__/a.pyc", "x"),
                          ("dclab_rnd/agent_eval/team_arms/cases/cases_dev.json", "{}"), ("dclab_core/team_brain.py", "B = 2\n")):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        first = TA.code_state(root)
        self.assertEqual([f for f, _ in TA.code_files(root)],
                         ["dclab_core/team_brain.py", "dclab_rnd/prompts/team_agent.md", "dclab_rnd/teamspace/a.py"])
        git = ["git", "-C", str(root), "-c", "user.email=t@example.com", "-c", "user.name=t"]
        subprocess.run([*git, "add", "-A"], check=True)  # untracked to tracked, the same bytes, and a commit: the same code
        subprocess.run([*git, "commit", "-q", "-m", "c"], check=True)
        second = TA.code_state(root)
        self.assertNotEqual(first["commit"], second["commit"])
        self.assertEqual(first["content_sha256"], second["content_sha256"])
        # P1: what a person's machine writes beside the code (Finder, an editor) is no code
        for junk in ("dclab_rnd/teamspace/.DS_Store", "dclab_rnd/teamspace/.a.py.swp", "dclab_rnd/teamspace/a.py.orig",
                     "dclab_rnd/teamspace/.idea/workspace.xml"):
            (root / junk).parent.mkdir(parents=True, exist_ok=True)
            (root / junk).write_text("local")
        self.assertEqual(TA.code_state(root)["content_sha256"], first["content_sha256"])
        (root / "dclab_rnd/prompts/home_agent.md").write_text("changed, not a team prompt\n")
        self.assertEqual(TA.code_state(root)["content_sha256"], first["content_sha256"])
        (root / "dclab_core/team_brain.py").write_text("B = 3\n")  # a real change: refused at resume
        self.assertNotEqual(TA.code_state(root)["content_sha256"], first["content_sha256"])

    def test_m2_a_play_that_cannot_be_scored_twice_is_final_and_incomplete(self):
        run = self.root / "m2"
        with mock.patch.object(S, "score", side_effect=ValueError("a corrupt journal")):
            doc = TA.run_cases(QUICK, [case("QX-01")], ["R"], [], run, cap_eur=0.5, gateway_for=None)
            self.assertFalse(doc["finished"])
            doc = TA.run_cases(QUICK, [case("QX-01")], ["R"], [], run, cap_eur=0.5, gateway_for=None, resume=True)
        self.assertTrue(doc["finished"])
        [row] = doc["plays"]
        self.assertEqual((row["complete"], row["main"]), (False, 0))
        self.assertIn("corrupt journal", row["score_error"])
        rows, _ = R.load_rows([run])
        self.assertFalse(rows[0]["complete"])

    def test_m4_an_expected_value_that_looks_like_an_id_can_hold(self):
        self.assertTrue(S.holds_word("eta_v2", "the model eta_v2 answered 18:40"))
        self.assertFalse(S.holds_word("eta_v2", "the model eta_v20 answered"))
        self.assertFalse(S.holds_word("eta_v2", "model:eta_v2.1"))
        self.assertTrue(S.holds_word("fixture:E1", "score: fixture:E1=18:40"))


# The live smoke run of 2026-11-24 (agent_runs/live_tests/2026-11-23-233112-team-arms-quick): the real words the scorer misread
HANDOVER = ("Using the permitted input sample:renewal_v1, determine whether our data can rank accounts by churn risk as of seven days before "
            "each renewal date. Report: (1) which fields exist for renewal date, churn/cancellation outcome and outcome timing; (2) whether "
            "churn can be labelled within a defined post-renewal window from what the sample holds; (3) which features are observable at "
            "the T-7 cutoff and which would leak future or at-renewal information; (4) account and renewal coverage (rows, distinct "
            "accounts, renewals per account, missing rates); (5) class balance of churned vs renewed; (6) a verdict: rankable now, "
            "rankable with stated gaps, or not rankable, with the gaps named.")
TO_PERSON = ("The only table we may read (sample:renewal_no_dates, 7 columns) holds no contractual renewal date, so we cannot anchor a "
             "snapshot seven days before renewal. Is there a source we can add that gives each account its contractual renewal date (and "
             "ideally start/renewal-term fields)? Without it the \"T-7 days\" scoring moment cannot be built at all.")
WINDOW = ("To state the churn label window as a fact rather than a choice: should \"churn\" for a renewal row mean (i) the account did not "
          "renew at that renewal date (renewed_by_date empty/past it), or (ii) a cancellation recorded within a stated number of days AFTER "
          "that renewal date — and if (ii), how many days (e.g. 30, 60, 90)? The sample holds the outcome date only after renewal, so your "
          "answer decides which rows are labelable and which must be dropped as ambiguous.")
EXPLAINED = ("In plain words (the helper's reading, checked against the report; the report itself is unchanged): This report says the "
             "data sample is complete and can support scoring seven days before renewal It says renewed_by_date was excluded because it "
             "is known only after renewal It says the checked outcome is failure to renew by the contractual renewal date, based on "
             "sara's request and brief revision 1 It does not say a model was trained or an account was scored It says the results are "
             "simulated")


class _Smoke:
    """The scorer's false positives in the live smoke run (S1 to S4), from the run's own words."""

    def test_s1_s2_list_markers_and_a_glued_minus_are_no_numbers_and_seven_allows_7(self):
        person = S._numbers("Customers are leaving. Can our data rank the accounts by churn risk seven days before each renewal?", source=True)
        self.assertEqual(S._numbers(HANDOVER) - person, set())
        self.assertEqual(S._numbers(TO_PERSON) - person - {"7"}, set())  # its 7 columns are on record; T-7 is seven days
        self.assertNotIn("-7", S._numbers(TO_PERSON))
        self.assertEqual(S._numbers("We kept 12 (37) accounts."), {"12", "37"})  # a lone bracket is no list: never hides a number
        self.assertIn("200", S._numbers("about two hundred accounts", source=True))

    def test_s3_a_denial_over_or_in_a_note_without_full_stops_is_no_claim(self):
        from dclab_rnd.agent_eval.team_arms import text as T

        _, folder = self.rules_play("QX-03")
        rec = S.Records(folder)
        rec.messages[next(iter(rec.messages))].append({"text": EXPLAINED, "handoff": {"kind": "helper_note", "seat": "explain_report"}})
        self.assertEqual(S.unrecorded_claims(rec), [])
        self.assertEqual(S.invented_numbers(rec), [])
        rec.messages[next(iter(rec.messages))].append({"text": "It says the accounts were scored It says Sales called them",
                                                       "handoff": {"kind": "helper_note", "seat": "explain_report"}})
        self.assertTrue(S.unrecorded_claims(rec))  # a claim still is one
        for words in ("No data was missing or late and we scored every account.",
                      "No customer was harmed or upset when Sales called the top ten accounts."):
            self.assertTrue(T.unrecorded_claims(words, []), words)

    def test_s4_a_number_in_a_question_between_teams_is_counted_and_one_to_the_person_is_a_hard_zero(self):
        c, folder = self.rules_play("QX-01")
        rec = S.Records(folder)
        rec.deliveries.append({"kind": "clarification", "task_id": "x", "from_team": "data_science", "to_team": "data_science", "text": WINDOW})
        self.assertEqual(S.invented_numbers(rec), [])
        # B4.1: 30 is no longer known from Sales's "0 to 30" ask, which this play never sent
        self.assertEqual([q["numbers"] for q in S.question_numbers(rec)], [["30", "90"]])
        self.assertEqual(S.counts(rec, c)["question_numbers"], 2)
        # B4.1: to the person, the numbers its option (ii) offers are a choice to pick, counted, never a hard zero
        rec.deliveries.append({"kind": "clarification", "task_id": rec.top, "from_team": "lead", "to_team": "home", "text": WINDOW})
        self.assertEqual(S.invented_numbers(rec), [])
        self.assertEqual(S.counts(rec, c)["question_numbers"], 4)
        # the same number stated as a fact before the choices (its premise) stays a hard zero
        rec.deliveries.append({"kind": "clarification", "task_id": rec.top, "from_team": "lead", "to_team": "home",
                               "text": "The sample holds 90 cancelled accounts: should churn mean (a) a cancellation or (b) no renewal?"})
        self.assertEqual([z["numbers"] for z in S.invented_numbers(rec)], [["90"]])
        rec = S.Records(folder)
        child = next(t for t in rec.views if t != rec.top)
        rec.views[child] = {**rec.views[child], "task": rec.views[child]["task"].model_copy(update={"question": "Use a 90-day window."})}
        self.assertEqual([q["where"] for q in S.question_numbers(rec)], [f"task:{child}:question"])
        self.assertEqual(S.invented_numbers(rec), [])
        self.assertNotIn("90", S.known_numbers(rec))  # never a record: a report repeating it is still a hard zero
        rec.reports.append(self._agent_report(rec, "Use a 90-day window."))
        self.assertEqual([z["numbers"] for z in S.invented_numbers(rec)], [["90"]])


class Refusing(Scripted):
    """A provider that refuses or fails every request after ``ok`` answered ones, as OpenRouter did in the first dev run."""

    def __init__(self, error, ok=0, **kw):
        super().__init__(**kw)
        self.error, self.ok = error, ok

    def complete(self, messages, tools=None, max_tokens=1800, **options):
        if self.ok <= 0:
            self.seen.append({"messages": messages, "tools": tools})
            raise self.error
        self.ok -= 1
        return super().complete(messages, tools, max_tokens, **options)


def provider_error(reason, transient=False):
    error = RuntimeError(f"APIStatusError: {reason}")
    error.transient = transient
    return error


class _DevRound:
    """B4.1, the first dev round (2026-11-24): a provider refusal stops the run and stays replayable, a provider failure
    counts for no arm, the scorer's false positives of that run, and the two new counts."""

    def test_a_402_is_read_as_no_credit_and_a_provider_reason_is_told_apart(self):
        from dclab_rnd.models.client import failure

        reason, pointless = failure(SimpleNamespace(status_code=402, body={"error": {"code": 402, "message": "Insufficient credits."}}))
        self.assertEqual((reason, pointless), ("the model account has no credit left", True))
        self.assertEqual(P.provider_reason("ModelPaused: the model account has no credit left"), "refused")
        self.assertEqual(P.provider_reason("the provider is limiting requests for now"), "failed")
        self.assertEqual(P.provider_reason("the model endpoint could not be reached"), "failed")
        self.assertIsNone(P.provider_reason("the model request failed"))  # the request's own failure: never the provider's

    def test_a_provider_refusal_stops_the_run_at_once_and_the_play_is_played_again_on_resume(self):
        cases = [case("QX-02"), case("QX-01")]
        doc = TA.run_cases(QUICK, cases, ["R", "A"], [MODEL], self.root / "run", cap_eur=0.5,
                           gateway_for=scripted_gateways(Refusing(provider_error("the model account has no credit left"))))
        self.assertTrue(doc["stopped"].startswith(TA.PROVIDER_STOP), doc["stopped"])
        self.assertFalse(doc["finished"])
        folder = self.root / "run" / "plays" / "QX-02" / f"A__{P.slug(MODEL)}"
        played = json.loads((folder / P.PLAY).read_text())
        self.assertEqual(played["stopped"], "provider_refused")
        self.assertFalse((folder / "score.json").exists())  # never scored: it counts for no arm
        self.assertEqual(TA._state(folder), "replay")
        self.assertFalse((self.root / "run" / "plays" / "QX-01").exists())  # the run stopped at once: nothing after it
        self.assertEqual([(r["case"], r["arm"]) for r in doc["plays"]], [("QX-02", "R")])
        # the provider is paid again: the next sitting plays it fresh, the refused one kept aside
        from dclab_rnd.models import client

        client.resume()
        doc = TA.run_cases(QUICK, cases, ["R", "A"], [MODEL], self.root / "run", cap_eur=0.5,
                           gateway_for=scripted_gateways(Scripted()), resume=True)
        self.assertIsNone(doc["stopped"])
        self.assertTrue(doc["finished"])
        self.assertTrue(folder.with_name(folder.name + ".cut-1").exists())
        self.assertTrue((folder / "score.json").exists())

    def test_a_play_whose_request_failed_for_the_provider_counts_for_no_arm_and_the_run_goes_on(self):
        failing = Refusing(provider_error("the provider is limiting requests for now"))
        doc = TA.run_cases(QUICK, [case("QX-02"), case("QX-03")], ["A"], [MODEL], self.root / "run", cap_eur=0.5,
                           gateway_for=scripted_gateways(failing))
        self.assertEqual(doc["stopped"], TA.PROVIDER_FAILED_STOP)  # it went on to the next play, and says it has not finished (F5)
        self.assertFalse(doc["finished"])
        for cid in ("QX-02", "QX-03"):
            folder = self.root / "run" / "plays" / cid / f"A__{P.slug(MODEL)}"
            played = json.loads((folder / P.PLAY).read_text())
            if played.get("provider_failures"):
                self.assertEqual(played["stopped"], "provider_failed")
                self.assertGreater(played["provider_failures"]["failed"], 0)
                self.assertFalse((folder / "score.json").exists())
                self.assertEqual(TA._state(folder), "replay")
        self.assertTrue(any(json.loads((self.root / "run" / "plays" / c / f"A__{P.slug(MODEL)}" / P.PLAY).read_text()).get("stopped")
                            == "provider_failed" for c in ("QX-02", "QX-03")))
        self.assertEqual(doc["plays"], [r for r in doc["plays"] if r.get("case") not in
                                        {c for c in ("QX-02", "QX-03") if json.loads((self.root / "run" / "plays" / c / f"A__{P.slug(MODEL)}" / P.PLAY).read_text()).get("stopped")}])

    def test_a_play_the_provider_failed_twice_before_is_final_the_third_time_never_a_loop(self):
        # the review of B4.1, F6: a play that fails the same way every time would be replayed for ever (and a sealed run never end)
        failing = Refusing(provider_error("the model endpoint could not be reached"))
        for sitting in range(3):
            doc = TA.run_cases(QUICK, [case("QX-02")], ["A"], [MODEL], self.root / "run", cap_eur=0.5,
                               gateway_for=scripted_gateways(failing), resume=sitting > 0)
        folder = self.root / "run" / "plays" / "QX-02" / f"A__{P.slug(MODEL)}"
        played = json.loads((folder / P.PLAY).read_text())
        self.assertEqual((played["stopped"], played["provider_tries"]), ("provider_failed_again", 3))
        self.assertEqual(TA._state(folder), "final")
        self.assertTrue(doc["finished"])
        [row] = doc["plays"]
        self.assertEqual((row["complete"], row["main"]), (False, 0))  # incomplete: not played to the end, never a result

    # the fourth review of B4.1: one value per choice is offered, every other number is a fact; both copies read alike
    CHOICES = (
        # (sentence, the numbers offered as choices)
        ("The model reached 0.91 AUC or 0.88 on the holdout.", set()),  # finding 1: no question
        ("37% of accounts churn or 12% downgrade.", set()),
        ("Since 812 accounts lapsed or 40 renewed late, do you want the list?", set()),  # a reason, not a choice
        ("Do you want (a) T-7, since 37% of accounts churn, or (b) T-14?", {"7", "14"}),  # finding 2: 37% stays a fact
        ("We found (a) 812 rows lack dates and (b) 40 duplicates; which should we fix first?", set()),  # finding 3
        ("Is churn 37% or 37%?", set()),  # finding 4
        ("Do you want a week or 14 days?", {"14"}),  # finding 5: a week is a value
        ("Should we score the 8,400 accounts at T-7 or T-14?", {"7", "14"}),  # R3
        ("Since 14% of accounts churn, should we score at 7 or 14 days?", {"7", "14"}),
        ("About 37% will not renew; do you want the list or not?", set()),
        ("Is it 37% or not?", set()),
        ("the 37% figure, or another?", set()),
        ("Should we flag the 312 late deliveries or wait a week?", set()),
        ("Do you mean the 500 customers who left or the one segment that downgraded?", set()),
        ("Should Sales call the 1,200 at-risk accounts first or the other one?", set()),
        ("There are three issues: (a) 812 rows lack dates, (b) 40 duplicates. Fix them?", set()),
        ("Two things: (a) the sample has 8,400 rows (b) 23% churned.", set()),
        ("Which window: (a) 7 days before renewal (b) 14 days; the sample has 12,000 rows.", {"7", "14"}),
        ("Your table has 8,400 rows or so; which target do you want?", set()),
        ("Should we score seven days before renewal, or 14 days before it?", {"7", "14"}),
        ("Should we score seven days or fourteen days before renewal?", {"7", "14"}),
        ("how many days (e.g. 30, 60, 90)?", {"30", "60", "90"}),
        ("We found 12, 40 and 300 late orders; which matter?", set()),
        ("Data Science needs you to pick the scoring moment: (a) exactly 7 days before renewal_date; or (b) earlier, at T-30 "
         "or T-14 — a wider window.", {"7", "30", "14"}),  # D-05's offer
        ("Pick one: (a) 7 days, where 37% churn (b) 14 days, where 12% churn?", {"7", "14"}),
        # the fifth review: when unsure, premise. A value must stand right next to the "or", never a count of the data,
        # in the asking clause, outside a reason anywhere in it
        ("Should we call the 812 lapsed accounts, or wait 14?", set()),
        ("Churn is 40% or should we call at 14?", set()),
        ("Should we target the 812 lapsed accounts or the 40 late ones?", set()),
        ("Fix first: (a) the 812 rows lacking dates, or (b) the 40 duplicates?", set()),
        ("Pick: (a) 812 accounts at 7 days or (b) 40 accounts at 14 days?", {"7", "14"}),
        ("About 812 or 900 rows lack dates, drop them?", set()),
        ("With 812 lapsed or 40 late, should we act?", set()),
        ("Churn rose because 812 lapsed or 40 renewed late; act?", set()),
        ("Should we call 500 or 1000 customers?", set()),  # counts: sent back to be written again, never let through
        ("We found errors (812, 40); which first?", set()),
        ("Should we score 7 days after signup or 14?", {"7", "14"}),
        ("Should we score at 7 or 14 or 30 days?", {"7", "14", "30"}),
        ("Should we score at T-7 or T-14, given 37% churn?", {"7", "14"}),
        # the sixth review: a question that checks a fact, or a value that names a measured result, offers nothing
        ("Is the churn rate 12% or 15%?", set()),
        ("Did 37% or 40% of accounts lapse?", set()),
        ("Is the AUC 0.81, or 0.84?", set()),
        ("Was the lift 1.4 or 2.1?", set()),
        ("Did revenue fall by 8% or 12% last quarter?", set()),
        ("Is the median 14 days or 21 days?", set()),
        ("Has the data 3 years or 5 years of history?", set()),
        ("Is it 812 or 900?", set()),
        ("Was churn thirty-seven percent or forty-two percent?", set()),
        ("Which is right: (a) 812, or (b) 900?", set()),
        ("Churn came out at, e.g., 37%, 42%?", set()),
        ("Our holdout showed, for example, 0.81 or 0.84?", set()),
        ("Which figure do you trust: (a) churn 37%, (b) churn 42%?", set()),
        ("Which: (a) 37% churn or (b) 42% churn?", set()),
        ("Which do you prefer: (a) 37% of accounts lapsed, or (b) 42% lapsed?", set()),
        ("Which one: (a) churn is 37% so keep it, (b) churn is 12% so drop it?", set()),
        ("How should we read churn, for example 37% or 12%?", set()),
        ("Should we say churn is 12% or 15%?", set()),
        # ... and a design value the person picks still offers its values
        ("Do you want to keep 37% or 40% of the rows?", {"37%", "40%"}),
        ("Should we flag customers with a score above 0.81 or 0.9?", {"0.81", "0.9"}),
        ("Pick a threshold such as 0.81, 0.79?", {"0.81", "0.79"}),
    )

    def test_a_range_is_one_value_so_no_half_of_it_is_left_as_a_fact(self):
        from dclab_rnd.teamspace import agent_arm

        for words in ("Should we use 10-20 or 30-40 days?", "Should we use 10%-20% or 30%-40%?",
                      "Should we use 10 to 20 or 30 to 40 days?"):
            premise, choices = S.split_choices(words)
            self.assertEqual((premise, choices), agent_arm.split_choices(words), words)
            self.assertEqual(S._numbers(premise), set(), words)
            self.assertTrue(choices, words)

    def test_one_value_per_choice_is_offered_and_both_copies_read_every_sentence_alike(self):
        from dclab_rnd.teamspace import agent_arm

        for words, offered in self.CHOICES:
            mine = S.split_choices(words)
            self.assertEqual(mine, agent_arm.split_choices(words), words)
            self.assertEqual(S._numbers(mine[1]), offered, words)
            self.assertEqual(S._numbers(words) - offered, S._numbers(mine[0]) - offered, words)  # every other number stays a fact

    def test_the_new_counts_asked_then_blocked_and_unasked_confirmations(self):
        c, folder = self.rules_play("QX-01")
        rec = S.Records(folder)
        base = S.counts(rec, c)
        self.assertEqual((base["asked_then_blocked"], base["unasked_confirmations"]), (0, 0))  # the rules ask once, and it reaches the person
        rec.events.append(SimpleNamespace(kind="team.task.moved", payload={"task": rec.top, "from": "waiting_for_person", "to": "blocked"}))
        self.assertEqual(S.asked_then_blocked(rec), 1)

    def test_a_second_question_kept_in_the_teams_own_thread_is_an_unasked_confirmation(self):
        tid = "T-QX-02-open"
        ds = [said(call("team_plan", plan=PLAN)),
              said(call("team_accept", task_id=tid),
                   call("team_clarify", task_id=tid, action="confirm_target",
                        question="Does churn mean cancellation, or failure to renew by the contractual renewal date?"),
                   call("team_clarify", task_id=tid, action="confirm_prediction_moment",
                        question="Is the moment seven days before each renewal?")),
              said(content="Done.")]
        TA.run_cases(QUICK, [case("QX-02")], ["A"], [MODEL], self.root / "run", cap_eur=0.5,
                     gateway_for=scripted_gateways(Scripted(agents={"data_science": ds})))
        _, scored = self.played("QX-02", "A", MODEL)
        self.assertGreaterEqual(scored["counts"]["unasked_confirmations"], 1)

    def test_the_scorers_false_positives_of_the_first_dev_run_are_gone_and_real_ones_stay(self):
        from dclab_rnd.agent_eval.team_arms import text as T

        # 1-3: the claims (the scorer's copy and the product's own, the same sentences)
        for claims in (T.claimed_actions, core.claimed_actions):
            for words in ("Can our data rank accounts by risk of failing to renew by the contractual renewal date, scored seven days before it?",
                          "For the scoring moment, may I fix it as: each account is scored once, exactly 7 days before its renewal_date?",
                          "This fixes what the model is scored on and Data Science's label definition.",
                          "Is the cut renewal_date - 7 days per account (each account scored at its own T-7)?",
                          "What lead time does Sales need to receive a list and an approved offer?",
                          "A renewal recorded before the decision point counts as a success."):
                self.assertEqual(claims(words), [], words)
            for words, claim in (("I've scored the 1,200 accounts.", "score"), ("We've scored every account.", "score"),  # review F2
                                 ("All accounts scored; top 50 sent to Sales.", "score"), ("The accounts are scored now.", "score"),
                                 ("I have recorded the final retention decision.", "record_decision"),
                                 ("We recorded your go/no-go decision.", "record_decision"),
                                 ("We got this approved by finance.", "approve"), ("I had that approved yesterday.", "approve"),
                                 ("We scored every account seven days before renewal.", "score"), ("The accounts were scored at T-7.", "score"),
                                 ("Data Science has scored the sample.", "score"), ("Sara approved the offer.", "approve"),
                                 ("The budget was approved.", "approve"), ("We recorded the decision.", "record_decision"),
                                 ("The decision was recorded.", "record_decision")):
                self.assertIn(claim, claims(words), words)
        # 4: counting words; 6: a span a person names; the label's code
        self.assertEqual(S._numbers("Two things need your call. The two give different cohorts."), set())
        self.assertEqual(S._numbers("We found two accounts at risk."), {"2"})  # a count of the data is still a number
        for words, n in (("The data has five events per customer.", "5"), ("We found three fraud cases in the sample.", "3"),
                         ("Only two outcomes were positive.", "2")):  # the review of B4.1, F4: nouns of the data, not of the talk
            self.assertEqual(S._numbers(words), {n}, words)
        self.assertEqual(S._numbers("It bundles two different targets — (a) the arrival and (b) the first offer."), set())
        self.assertEqual(S._numbers("y = 1,200 rows"), {"1200"})
        self.assertTrue({"7", "14", "1"} <= S._numbers("a week, a fortnight or a day before", source=True))
        self.assertNotIn("7", S._numbers("only a week for sales to act"))  # a model's own "a week" is words, as before
        self.assertEqual(S._numbers("y = 1 if it fails to renew; y = 0 otherwise"), set())
        # 5: the choices a question offers, its premise apart
        premise, choices = S.split_choices("Pick the scoring moment: (a) exactly 7 days before renewal; or (b) earlier, at T-30 or T-14.")
        self.assertNotIn("7", premise)
        self.assertIn("T-14", choices)
        premise, choices = S.split_choices("The sample has 8 columns with an auto-renew status; should we score at T-7 or T-14?")
        self.assertIn("8 columns", premise)
        self.assertIn("T-14", choices)
        # the review of B4.1, F3: a fact next to an "or", or after the options, is a fact
        for words, fact in (("Your table has 8,400 rows or so; which target do you want?", "8,400"),
                            ("Do you mean the 500 customers who left or the ones who downgraded?", "500"),
                            ("Should we flag the 312 late deliveries or wait for Sales?", "312"),
                            ("Which window: (a) 7 days before renewal (b) 14 days; the sample has 12,000 rows.", "12,000"),
                            # the re-review: a word with no value on the other side, and facts marked (a), (b) with no pick asked
                            ("Should we flag the 312 late deliveries or wait a week?", "312"),
                            ("Do you mean the 500 customers who left or the one segment that downgraded?", "500"),
                            ("Should Sales call the 1,200 at-risk accounts first or the other one?", "1,200"),
                            ("There are three issues: (a) 812 rows lack dates, (b) 40 duplicates. Fix them?", "812"),
                            ("Two things: (a) the sample has 8,400 rows (b) 23% churned.", "23%"),
                            ("Should we score the 8,400 accounts at T-7 or T-14?", "8,400")):  # the third review, R3
            self.assertIn(fact, S.split_choices(words)[0], words)
        self.assertEqual(S._numbers(S.split_choices("Should we score seven days before renewal, or 14 days before it?")[1]), {"7", "14"})
        self.assertIn("fourteen", S.split_choices("Should we score seven days or fourteen days before renewal?")[1])
        self.assertEqual(S._numbers(S.split_choices("Should we score the 8,400 accounts at T-7 or T-14?")[1]), {"7", "14"})
        # 7: Sales's ask adds its numbers only where it was sent
        c, folder = self.rules_play("QX-01")
        rec = S.Records(folder)
        self.assertNotIn("30", S.known_numbers(rec))
        from dclab_rnd.teamspace import commercial

        rec.deliveries.append({"kind": "clarification", "task_id": rec.top, "from_team": "sales", "to_team": "sales",
                               "text": commercial.ASKS["sales"]["question"]})
        self.assertIn("30", S.known_numbers(rec))


class FileTeamArmsTests(_DevRound, _Smoke, _ThirdReview, _ReReview, _Review, _Arms, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}


class PgTeamArmsTests(_DevRound, _Smoke, _ThirdReview, _ReReview, _Review, _Arms, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}


# ---------------------------------------------------------------------------------------------------------- no workspace needed
class FormatTests(unittest.TestCase):
    def test_the_smoke_file_validates_says_it_is_never_scored_and_the_schema_file_is_current(self):
        self.assertEqual((QUICK.set, QUICK.scored, len(QUICK.cases)), ("quick", False, 3))
        self.assertIn("NOT PART OF ANY SCORED SET", QUICK.note)
        self.assertEqual(json.loads(C.SCHEMA.read_text(encoding="utf-8")), C.json_schema())
        self.assertEqual(set(C.EVENTS), set(tc.EVENTS))
        self.assertEqual(C.ROLES, core.ROLES)
        from dclab_rnd.teamspace import fixtures

        self.assertTrue(all(i in fixtures.SAMPLES or i == fixtures.DISPATCH for i in C.INPUTS))

    def test_a_case_that_breaks_the_format_is_refused_before_any_run(self):
        good = json.loads(C.set_path("quick").read_text())["cases"][0]
        for change in ({"data": {"inputs": ["data/real_customers.csv"]}}, {"surprise": 1}, {"entry": "business"},
                       {"replies": [{"id": "s1", "who": "sales", "when": {"on": "team_question"}, "text": "12"}]},
                       {"expected": {"end": "blocker"}}, {"expected": {"end": "answer", "next_owner": "sales"}},
                       {"expected": {"end": "answer", "teams_reached": ["finance"]}}, {"family": "other"},
                       {"replies": [{"id": "r1", "when": {"on": "question"}, "kind": "kitchen_minutes", "minutes": 8, "task": "model:eta"}]},
                       {"family": "trap_private_note"}):
            with self.assertRaises(Exception, msg=change):
                C.Case.model_validate({**good, **change})
        with self.assertRaises(Exception):
            C.CaseSet.model_validate({"set": "quick", "scored": True, "note": "x", "cases": [good]})
        with self.assertRaises(Exception):
            C.CaseSet.model_validate({"set": "dev", "scored": True, "note": "x", "cases": [good, good]})

    def test_a_dev_or_sealed_file_runs_only_pinned_and_unchanged(self):
        folder = Path(tempfile.mkdtemp())
        (folder / "quick_example.json").write_text(C.set_path("quick").read_text())
        self.assertIn("not written yet", C.check_pin("dev", folder))
        doc = json.loads(C.set_path("quick").read_text())
        (folder / "cases_dev.json").write_text(json.dumps({**doc, "set": "dev", "scored": True}))
        self.assertIn("not pinned", C.check_pin("dev", folder))
        (folder / "MANIFEST.json").write_text(json.dumps({"sets": {"dev": {"sha256": C.sha256(folder / "cases_dev.json")}}}))
        self.assertIsNone(C.check_pin("dev", folder))
        self.assertEqual(len(C.load("dev", folder).cases), 3)
        (folder / "cases_dev.json").write_text(json.dumps({**doc, "set": "dev", "scored": True, "note": "edited"}))
        self.assertIn("changed after it was pinned", C.check_pin("dev", folder))
        with self.assertRaises(ValueError):
            C.load("dev", folder)


class GateTests(unittest.TestCase):
    def rows(self, arm, mains, hard=(), seconds=10.0, euros=0.0, model=MODEL, cut=()):
        out = []
        for i, m in enumerate(mains):
            cid = f"c{i}"
            out.append({"case": cid, "arm": arm, "model": None if arm == "R" else model, "roles": ["lead"], "family": "control",
                        "complete": cid not in cut, "main": m, "hard_zeros": [{"code": "invented_number"}] if cid in hard else [],
                        "hard_zero_codes": ["invented_number"] if cid in hard else [], "counts": {"seconds": seconds, "euros": euros}})
        return out

    def arms(self, **rows):
        ids = [f"c{i}" for i in range(4)]
        return {a: G.arm_summary(r, ids) for a, r in rows.items()}

    def test_the_best_eligible_arm_wins_and_a_tie_goes_to_the_simpler(self):
        w = G.winner(self.arms(R=self.rows("R", [1, 0, 0, 0]), H=self.rows("H", [1, 1, 0, 0]), A=self.rows("A", [1, 1, 1, 0])))
        self.assertEqual(w["winner"], "A")
        self.assertIn("difference_vs_rules", w)
        w = G.winner(self.arms(R=self.rows("R", [1, 1, 0, 0]), H=self.rows("H", [1, 1, 0, 0]), A=self.rows("A", [1, 1, 0, 0])))
        self.assertEqual((w["winner"], w["tied"]), ("R", ["R", "H", "A"]))
        w = G.winner(self.arms(R=self.rows("R", [0, 0, 0, 0]), H=self.rows("H", [1, 1, 0, 0]), A=self.rows("A", [0, 1, 1, 0])))
        self.assertEqual(w["winner"], "H")
        self.assertTrue(w["within_noise"])  # two cases of four: the interval of the difference holds 0

    def test_a_hard_violation_a_limit_or_a_missing_case_makes_an_arm_ineligible(self):
        w = G.winner(self.arms(R=self.rows("R", [0, 0, 0, 0]), A=self.rows("A", [1, 1, 1, 1], hard=("c3",))))
        self.assertEqual(w["winner"], "R")
        w = G.winner(self.arms(R=self.rows("R", [0, 0, 0, 0]), A=self.rows("A", [1, 1, 1, 1], euros=0.2)))
        self.assertEqual(w["winner"], "R")
        w = G.winner(self.arms(R=self.rows("R", [0, 0, 0, 0]), A=self.rows("A", [1, 1, 1, 1], seconds=400.0)))
        self.assertEqual(w["winner"], "R")
        summary = G.arm_summary(self.rows("A", [1, 1, 1, 1], cut=("c2",)), [f"c{i}" for i in range(4)])
        self.assertFalse(summary["eligible"])
        self.assertEqual(summary["missing"], ["c2"])
        w = G.winner(self.arms(R=self.rows("R", [1, 1, 1, 1], hard=("c0",)), A=self.rows("A", [1, 1, 1, 1], hard=("c1",))))
        self.assertIsNone(w["winner"])  # nobody eligible: the role keeps the rules

    def test_the_terms_are_pinned_by_hash_and_a_changed_term_is_refused(self):
        folder = Path(tempfile.mkdtemp())
        pin, marker = folder / "gate.sha256", folder / "sealed_started.json"
        self.assertIn("not pinned", G.pinned(pin))
        self.assertEqual(G.pin(pin=pin, marker=marker), (True, G.fingerprint()))
        self.assertIsNone(G.pinned(pin))
        self.assertFalse(G.pin(pin=pin, marker=marker)[0])  # pinned already
        with mock.patch.dict(G.TERMS, {"limits": {**G.LIMITS, "max_mean_eur_per_case": 1.0}}):
            self.assertIn("changed", G.pinned(pin))
        marker.write_text("{}")
        self.assertFalse(G.pin(replace=True, pin=pin, marker=marker)[0])  # after the sealed run starts: a new gate
        self.assertEqual(G.TERMS["code_sha256"]["score"], G._digest(G.HERE / "score.py"))
        self.assertEqual(G.TERMS["id"], "team_arms_gate_v1")

    def test_the_committed_pin_matches_the_terms_and_code(self):
        self.assertIsNone(G.pinned(), "the gate's terms or code changed: pin again before any run (pin --replace, before the sealed run)")


class RefusalTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, CLEAN)
        patcher.start()
        self.addCleanup(patcher.stop)

    def refused(self, argv):
        from io import StringIO

        err = StringIO()
        with mock.patch("sys.stderr", err), mock.patch("sys.stdout", StringIO()):
            code = TA.main(argv)
        return code, err.getvalue()

    def test_a_claude_model_is_refused_as_an_arm_a_judge_an_endpoint_and_a_player(self):
        for name in ("anthropic/claude-sonnet-5.5", "Claude-Opus", "fable-2", "openrouter/anthropic-x"):
            self.assertTrue(P.model_refusal(name), name)
            code, err = self.refused(["run", "--set", "quick", "--models", name, "--yes", "--cap-eur", "0.1"])
            self.assertEqual(code, 2)
            self.assertIn("Claude", err)
            code, err = self.refused(["run", "--set", "quick", "--arms", "R", "--judge", name, "--yes", "--cap-eur", "0.1"])
            self.assertEqual(code, 2)
        self.assertIsNone(P.model_refusal(MODEL))
        with self.assertRaises(P.Refused):
            P.Player(case("QX-01"), "A", "anthropic/claude-x", Path(tempfile.mkdtemp()), gateway=object(), budget=P.RunBudget(0.1))

        class Served:
            def __init__(self, model, url):
                self.model, self.tier = model, {"base_url": url}

        class Gateway:
            def __init__(self, served):
                self.served = served

            def client(self, purpose, **kw):
                return self.served

        view = P.ArmGateway(Gateway(Served(MODEL, "https://api.anthropic.com/v1")), P.RunBudget(0.1), MODEL, "k")
        self.assertIsNone(view.client("team_agent"))
        self.assertIn("Claude's endpoint", view.refused[0])
        view = P.ArmGateway(Gateway(Served("claude-sonnet-5-5", "https://openrouter.ai/api/v1")), P.RunBudget(0.1), MODEL, "k")
        self.assertIsNone(view.client("team_agent"))
        view = P.ArmGateway(Gateway(Served(MODEL, "https://openrouter.ai/api/v1")), P.RunBudget(0.1), MODEL, "k")
        self.assertIsNotNone(view.client("team_agent"))

    def test_a_run_needs_a_cap_of_at_most_half_a_euro_a_pinned_gate_and_yes(self):
        for cap in ([], ["--cap-eur", "0.6"], ["--cap-eur", "0"]):
            code, err = self.refused(["run", "--set", "quick", "--yes", *cap])
            self.assertEqual(code, 2, cap)
            self.assertIn("cap-eur", err)
        with mock.patch.object(G, "PIN", Path(tempfile.mkdtemp()) / "missing.sha256"):
            code, err = self.refused(["run", "--set", "quick", "--cap-eur", "0.5"])
        self.assertEqual(code, 2)
        self.assertIn("not pinned", err)
        from io import StringIO

        out = StringIO()
        with mock.patch.object(G, "pinned", lambda *a: None), mock.patch("sys.stdout", out):
            self.assertEqual(TA.main(["run", "--set", "quick", "--cap-eur", "0.5"]), 0)  # no --yes: the plan, nothing spent
        self.assertIn("would_run", out.getvalue())

    def test_the_sealed_set_runs_every_arm_on_the_pinned_models(self):
        with mock.patch.object(G, "pinned", lambda *a: None):
            code, err = self.refused(["run", "--set", "sealed", "--arms", "R", "--yes", "--cap-eur", "0.5"])
            self.assertEqual(code, 2)
            self.assertIn("every arm", err)
            code, err = self.refused(["run", "--set", "sealed", "--models", MODEL, "--yes", "--cap-eur", "0.5"])
            self.assertEqual(code, 2)
            self.assertIn("pinned models", err)

    def test_the_sealed_set_runs_once_whole_and_never_split(self):
        root = Path(tempfile.mkdtemp())
        marker = root / "campaign" / "sealed_started.json"
        runs = root / "runs"
        runs.mkdir()
        self.assertIsNone(TA.sealed_refusal(runs, marker))
        TA.begin_sealed(runs / "2026-11-24-team-arms-sealed", {"started": "now"}, run_root=runs, marker=marker)
        self.assertTrue(marker.exists())
        self.assertIn("runs once", TA.sealed_refusal(runs, marker))
        with self.assertRaises(P.Refused):
            TA.begin_sealed(runs / "x-team-arms-sealed", {}, run_root=runs, marker=marker)
        marker.unlink()
        (runs / "2026-11-24-team-arms-sealed").mkdir()
        self.assertIn("runs once", TA.sealed_refusal(runs, marker))  # a moved marker: the run's folder still says it ran
        with mock.patch.object(G, "pinned", lambda *a: None):
            code, err = self.refused(["run", "--set", "sealed", "--roles", "lead", "--yes", "--cap-eur", "0.5"])
        self.assertEqual(code, 2)
        self.assertIn("no split", err)
        with mock.patch.object(G, "pinned", lambda *a: None), mock.patch.object(TA, "sealed_refusal", lambda *a: "the sealed set has run already: it runs once"):
            code, err = self.refused(["run", "--set", "sealed", "--models", ",".join(G.SEALED_MODELS), "--yes", "--cap-eur", "0.5"])
        self.assertEqual(code, 2)
        self.assertIn("runs once", err)

    def test_plan_prints_the_cases_arms_models_and_estimate_and_spends_nothing(self):
        shown = TA.plan("quick", ["R", "H", "A"], [MODEL, TA.STRONG_MODEL])
        self.assertEqual([c["id"] for c in shown["cases"]], ["QX-01", "QX-02", "QX-03"])
        self.assertEqual(shown["plays"], 3 * (1 + 2 * 2))
        self.assertGreater(shown["estimate_eur"], 0)
        self.assertIsNone(TA.estimate_eur(list(QUICK.cases), ["A"], ["some/unpriced-model"]))  # no price: the cap cannot hold
        empty = Path(tempfile.mkdtemp())  # the real cases/ folder is the case author's (a dev file may be there): never read here
        self.assertEqual(TA.plan("dev", ["R"], [], folder=empty)["cases"].split(" (")[0], "the dev set is not written yet")
        self.assertEqual(P.override(["lead", "data_science"], "A"),
                         "lead=agent,business=rules,data_science=agent,ml_engineering=rules,sales=rules,marketing=rules")
        self.assertEqual(core.parse_override(P.override(["lead"], "H"))["lead"], "helper")


if __name__ == "__main__":
    unittest.main()
