"""B3 of the team workspace (2026-11-24): the agent arm acts.

A role switched to ``agent`` runs a model loop (``agents/runtime.run``) with the team tools of the one registry, inside
the strict layer: the same guard, task store and report builder as the rules. The model writes its plan first (kept
whole), then chooses its actions, their order, its questions and hand-overs; the Lead breaks a request into team tasks
itself. Code attaches the report's numbers from the records and refuses any sentence whose number no record holds, or
that claims an action without its terminal record. Budgets (steps, seconds, euros) end a turn cleanly. A model that
cannot answer before the first action leaves the turn to the rules; after an action the rules never run on top. A
delivery handed over again during or after its agent turn never repeats an action. Every model call keeps its thinking.

Scripted transports only (never a live key). On files and on PostgreSQL (``tests/run_parallel.py --pg``; never run
directly: that empties ``dclab_test``).
"""

import json
import os
import sys
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402
from test_teamspace_agents import ANSWER, BRIEF, _Contract  # noqa: E402

from dclab_core import team_brain as core  # noqa: E402
from dclab_core import team_work as tw  # noqa: E402
from dclab_rnd.models import gateway as gw  # noqa: E402
from dclab_rnd.models.usage import FileUsage  # noqa: E402
from dclab_rnd.teamspace import agent_arm, agents, brains, delivery, fixtures, lead  # noqa: E402

CLEAN = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY",
                         "DCLAB_THINKING", "DCLAB_TEAM_BRAIN", "DCLAB_TEAM_AGENT_STEPS", "DCLAB_TEAM_AGENT_SECONDS",
                         "DCLAB_TEAM_AGENT_EUR", "DCLAB_NO_LIVE_MODELS")} | {"OPENAI_API_KEY": "k", "DCLAB_LOG_PROMPTS": "1"}
PLAN = ("1. Accept the task: it is ours.\n2. Inspect the sample's columns (inspect_schema).\n3. Check which columns leak "
        "(check_leakage), then name the baseline.\n4. Report what the data can answer, or block on what is missing.\n"
        "Stop when the report is returned. If a step is refused, read why and adjust.")
N = {"n": 0}


def call(name, **arguments):
    N["n"] += 1
    return {"id": f"c{N['n']}", "name": name, "arguments": arguments}


def said(*calls, content="", why="the plan says what comes next", cost=None):
    """A scripted model reply with its thinking (the API's thinking summary)."""
    usage = {"input_tokens": 3, "output_tokens": 2, **({"provider_cost": cost} if cost is not None else {})}
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                 for c in calls]
    return {"content": content, "tool_calls": list(calls), "usage": usage, "thinking": [{"text": why}],
            "finish_reason": "tool_calls" if calls else "stop", "assistant_message": message}


DONE = said(content="Done.")


class Scripted:
    """Replies in order, then ``rest`` for ever (default: words without a plan, which the agent arm reads as invalid)."""

    def __init__(self, *replies, rest=None, delay=0.0, gate=None, delays=None):
        self.replies, self.rest, self.delay, self.gate, self.seen, self.thinking = list(replies), rest, delay, gate, [], None
        self.delays = list(delays or [])  # a delay per request, in order (then ``delay``)

    def complete(self, messages, tools=None, max_tokens=1800, **options):
        self.seen.append({"messages": [dict(m) for m in messages], "tools": tools, "options": options})
        if self.gate is not None:
            self.gate.wait(20)
        wait = self.delays.pop(0) if self.delays else self.delay
        if wait:
            import time

            time.sleep(wait)
        reply = self.replies.pop(0) if self.replies else (self.rest or said(content="Nothing to add."))
        if isinstance(reply, BaseException):
            raise reply
        return reply


def exchanges(home):
    return [json.loads(line) for p in sorted(Path(home).rglob("model_exchanges/**/*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]


class _Agent(_Contract):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ, CLEAN)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(brains.drain, 10)
        from dclab_rnd import audit
        from dclab_rnd.ledger import open_ledger
        from dclab_rnd.models.routing import open_routing

        self.services.ledger = open_ledger(self.home, self.services.workspace_id)
        self.routing = open_routing(self.home)
        self.audit = audit.open_audit(self.home)
        self.draft = self.services.drafts.create("Renewal on Home")["id"]

    def scripted(self, *replies, **kw):
        transport = Scripted(*replies, **kw)
        self.services.gateway = gw.Gateway(FileUsage(self.home / "model_requests.jsonl"), transport=lambda tier, purpose: transport,
                                           routing=self.routing)
        return transport

    def switch(self, role, arm="agent"):
        return brains.switch(self.routing, self.audit, role, arm, "owner", "compare the arms on the dev set")

    def events(self, kind):
        return [e.payload for e in self.services.experience.read(kinds=[kind])]

    def turns(self, agent=None):
        return [p for p in self.events("team.agent.turn") if agent is None or p["agent"] == agent]

    def answered(self, agent=None):
        return [p for p in self.events("team.brain.answered") if agent is None or p["agent"] == agent]

    def runs(self, name):
        return [e for e in self.services.experience.read(kinds=["run.recorded"]) if e.payload.get("name") == name]

    def ds_row(self, task_id="T-renewal"):
        return next(r for r in self.work.deliveries(project_id=self.pid) if r["task_id"] == task_id and r["to_team"] == "data_science"
                    and r["kind"] == "task")

    def mine(self, task_id, agent="data_science"):
        return [a.action for a in self.view(task_id)["actions"] if a.actor.id == agent]

    def schema_ref(self):
        return f"schema:sample:renewal_v1:{len(fixtures.sample('sample:renewal_v1')['columns'])} columns"

    # ------------------------------------------------------------------ 1. the plan first, kept whole
    def test_the_plan_is_recorded_whole_before_the_first_action_and_a_write_before_it_is_refused(self):
        self.switch("data_science")
        transport = self.scripted(said(call("team_accept", task_id="T-renewal")),  # a write before the plan: refused
                                  said(call("team_plan", plan=PLAN)),
                                  said(call("team_accept", task_id="T-renewal")), DONE)
        self.open_renewal()
        self.pump(only={"T-renewal"})
        self.assertEqual(self.view("T-renewal")["task"].status, "accepted")
        steps = [(s.payload.get("tool"), s.payload.get("verdict")) for s in self.steps("teamspace.data_science")]
        self.assertEqual([t for t, _ in steps], ["team_accept", "team_plan", "team_accept", None])
        self.assertNotEqual(steps[0][1], "ok")  # refused: nothing ran before the plan
        [turn] = self.turns("data_science")
        self.assertEqual((turn["outcome"], turn["stopped"], turn["writes"], turn["refused"]), ("acted", "answered", 1, 1))
        self.assertEqual(turn["plan_chars"], len(PLAN))
        self.assertEqual(turn["plan_steps"], 4)
        record = agent_arm.journal(self.services, turn["run"])
        self.assertEqual(record["plan"], PLAN)  # whole, never cut
        self.assertEqual([s["tool"] for s in record["steps"]], ["team_accept", "team_plan", "team_accept"])
        [ds] = self.answered("data_science")
        self.assertEqual((ds["arm"], ds["acted"], ds["model_tool"], ds["rule_tool"]), ("agent", "agent", "team_accept", "team_accept"))
        self.assertEqual(ds["request"], turn["request"])
        # its prompt asks for a plan (an answer), never for its reasoning: that comes from the API's thinking
        system, user = transport.seen[0]["messages"][0]["content"], transport.seen[0]["messages"][1]["content"]
        for words in ("reasoning", "explain your", "step by step", "<thinking>", "think aloud"):
            self.assertNotIn(words, (system + user).lower())
        self.assertIn("team_plan", system)
        offered = {(t.get("function") or t)["name"] for t in transport.seen[0]["tools"]}
        self.assertTrue({"team_plan", "team_read_task", "team_read_notes", "team_accept", "team_return_report"} <= offered)
        self.assertEqual(transport.seen[0]["options"].get("_single_attempt"), True)
        record = next(r for r in self.services.ledger.records(point="team.brain") if r.state.get("team") == "data_science")
        self.assertEqual((record.used, record.value_used), ("model_role", "agent"))

    # ------------------------------------------------------------------ 2. the Lead breaks the request into team tasks itself
    def test_a_hand_over_to_a_team_that_never_owns_handed_out_work_is_refused_so_no_case_hangs(self):
        # the live smoke run of 2026-11-24 (QX-03): an agent Lead handed a ranking to Business, the task stayed open for ever
        self.switch("lead")
        out = lead.ask(self.services, self.pid, self.prob, question=BRIEF, track="analyze", person="sara", draft_id=self.draft,
                       initiator=self.sara, inputs=("sample:renewal_v1",))
        tid = out["task_id"]
        transport = self.scripted(
            said(call("team_plan", plan="1. Accept.\n2. Business ranks the accounts.")),
            said(call("team_accept", task_id=tid),
                 call("team_delegate", action="delegate", task_id=tid, owner_team="business",
                      question="Rank the accounts by the risk of not renewing.", expected_output="a ranking",
                      done_criteria=["a ranking"])),
            DONE, rest=DONE)
        self.pump(only={tid})
        view = self.view(tid)
        self.assertEqual([c.owner_team for c in view["children"]], [])  # nothing handed to Business
        [turn] = self.turns("lead")
        self.assertEqual(turn["writes"], 1)  # the accept ran; the refused hand-over was neither run nor counted
        told = json.dumps(transport.seen[-1]["messages"])  # the model is told why, so it can hand the work to a team that takes it
        self.assertIn("does not take handed-out work", told)

    def test_the_lead_on_the_agent_arm_opens_team_tasks_itself_and_the_work_reaches_the_teams(self):
        self.switch("lead")
        plan = ("1. Accept and check the prerequisites.\n2. Data Science confirms the outcome and checks the label.\n"
                "3. In parallel Sales gives its capacity.\n4. I join the replies into one report.")
        out = lead.ask(self.services, self.pid, self.prob, question=BRIEF, track="analyze", person="sara", draft_id=self.draft,
                       initiator=self.sara, inputs=("sample:renewal_v1",))
        tid = out["task_id"]
        transport = self.scripted(
            said(call("team_plan", plan=plan)),
            said(call("team_accept", task_id=tid), call("team_act", task_id=tid, action="check_prerequisites")),
            said(call("team_delegate", action="delegate", task_id=tid, new_task_id="anything-the-model-likes", owner_team="data_science",
                      question="Can the sample rank accounts by the risk of not renewing, seven days before renewal?",
                      expected_output="a feasibility answer or a precise blocker", done_criteria=["says whether the data can answer"],
                      allowed_inputs=["sample:renewal_v1", "help:analyze", "sample:renewal_with_cancellations"],
                      thread_id="someone-elses-thread", budget_eur=50),
                 call("team_delegate", action="delegate", task_id=tid, new_task_id="x2", owner_team="sales",
                      question="How many calls can Sales make before renewal, and who must not be contacted?",
                      expected_output="capacity and exclusions", done_criteria=["names the capacity or what blocks it"])),
            DONE, rest=DONE)  # later Lead turns: words without a plan, so the rules carry the replies
        self.pump(only={tid})
        view = self.view(tid)
        children = {c.owner_team: c for c in view["children"]}
        self.assertEqual(set(children), {"data_science", "sales"})
        self.assertEqual(children["data_science"].id, f"{tid}.data_science")  # code names a child task: one per team
        self.assertEqual(children["sales"].id, f"{tid}.sales")
        self.assertEqual(children["data_science"].allowed_inputs, ["sample:renewal_v1"])  # never wider than the Lead's, never help:
        self.assertIn("rank accounts by the risk of not renewing", children["data_science"].question)
        # code's, never the model's: the child's reply thread and its budget (the review of B3, F6)
        self.assertNotEqual(children["data_science"].requester.thread_id, "someone-elses-thread")
        self.assertEqual(children["data_science"].budget_eur, 0)
        self.assertEqual(sorted(a.action for a in view["actions"] if a.actor.id == "lead"), ["check_prerequisites", "delegate", "delegate"])
        [turn] = self.turns("lead")
        self.assertEqual((turn["outcome"], turn["writes"]), ("acted", 4))
        self.assertEqual(agent_arm.journal(self.services, turn["run"])["plan"], plan)
        rows = {r["to_team"] for r in self.work.deliveries(project_id=self.pid) if r["kind"] == "task" and r["task_id"] in
                (children["data_science"].id, children["sales"].id)}
        self.assertEqual(rows, {"data_science", "sales"})
        self.pump()  # the teams (on the rules) take their tasks; the Lead's later turns fall back to the rules (no plan)
        ds = self.view(children["data_science"].id)
        self.assertIn(ds["task"].status, ("waiting_for_person", "accepted", "complete", "blocked"))
        self.assertTrue([s for s in self.steps("teamspace.data_science") if s.payload.get("tool") == "team_accept"])
        later = self.turns("lead")[1:]
        self.assertTrue(later and all((t["outcome"], t["fallback"]) == ("fell_back", "invalid_output") for t in later), later)
        self.assertEqual(len(transport.replies), 0)

    # ------------------------------------------------------------------ 3. the guard refuses; the turn goes on
    def test_forbidden_actions_are_refused_by_the_guard_recorded_and_the_turn_goes_on(self):
        self.switch("data_science")
        other = agents.open_case(self.services, self.pid, self.prob, task_id="T-other", question="How many calls can Sales make?",
                                 thread_id=self.biz_thread, member="sara", initiator=self.sara, owner="sales")  # Sales's case
        self.assertEqual(other["refused"], [])
        self.scripted(said(call("team_plan", plan=PLAN)),
                      said(call("team_act", task_id="T-renewal", action="approve"),
                           call("team_act", task_id="T-renewal", action="contact_customers"),
                           call("team_read_notes", team="sales"),
                           call("team_read_task", task_id="T-other"),
                           call("team_delegate", action="delegate_specialist", new_task_id="T-new", owner_team="sales",
                                question="Open a case", expected_output="x", done_criteria=["x"])),
                      said(call("team_read_notes", team="data_science"), call("team_accept", task_id="T-renewal")), DONE)
        self.open_renewal()
        self.pump(only={"T-renewal"})
        self.assertEqual(self.view("T-renewal")["task"].status, "accepted")
        acts = {a.action for a in self.view("T-renewal")["actions"]}
        self.assertFalse(acts & {"approve", "contact_customers"})
        verdicts = [(s.payload.get("tool"), s.payload.get("verdict")) for s in self.steps("teamspace.data_science")]
        refused = [t for t, v in verdicts if t and v != "ok"]
        self.assertEqual(refused, ["team_act", "team_act", "team_read_notes", "team_read_task", "team_delegate"])
        [turn] = [t for t in self.turns("data_science") if t["task"] == "T-renewal"]
        self.assertEqual((turn["outcome"], turn["refused"], turn["writes"]), ("acted", 5, 1))
        journal = agent_arm.journal(self.services, turn["run"])
        errors = [s["error"] for s in journal["steps"] if s.get("error")]
        self.assertTrue(any("never approves" in e for e in errors), errors)
        self.assertTrue(any("contact customers" in e for e in errors), errors)
        self.assertTrue(any("stay in that team" in e for e in errors), errors)
        self.assertTrue(any("a new case is opened by its person" in e for e in errors), errors)
        self.assertTrue(any("neither owns nor requested" in e for e in errors), errors)
        row = self.ds_row()
        self.assertEqual(row["status"], "delivered")  # a refused call is the guard doing its job: the delivery is not blocked
        self.assertIsNone(self.work.task(self.pid, "T-new"))

    # ------------------------------------------------------------------ 4, 5. the report: numbers from the records, no claim without a record
    def _feasibility(self, answer, limits=(), extra=()):
        return [said(call("team_plan", plan=PLAN)),
                said(call("team_accept", task_id="T-renewal"), call("team_act", task_id="T-renewal", action="inspect_schema"), *extra),
                said(call("team_return_report", task_id="T-renewal", outcome="complete", revision=7, answer=answer,
                          limits=list(limits), evidence=[{"source": "made-up", "ref": "x", "input_revision": "y"}])),
                DONE]

    def test_a_number_no_record_holds_is_refused_and_code_attaches_the_numbers_from_the_records(self):
        self.switch("data_science")
        columns = len(fixtures.sample("sample:renewal_v1")["columns"])
        answer = (f"The sample has {columns} columns. About 37% of accounts will not renew. "
                  "I recommend scoring each account seven days before renewal.")
        self.scripted(*self._feasibility(answer, limits=["Only 12 accounts were looked at.", "Checked on the sample only."]))
        self.open_renewal()
        self.pump(only={"T-renewal"})
        view = self.view("T-renewal")
        [report] = view["reports"]
        self.assertEqual((report.revision, view["task"].status), (1, "complete"))  # code numbers the revision
        self.assertIn(f"The sample has {columns} columns.", report.answer)
        self.assertIn("seven days before renewal", report.answer)
        self.assertNotIn("37%", report.answer)
        self.assertIn(self.schema_ref(), report.answer)  # code attaches what the action recorded
        self.assertNotIn("Only 12 accounts", " ".join(report.limits))
        self.assertIn("Checked on the sample only.", report.limits)
        self.assertEqual([e.source for e in report.evidence], ["sample:renewal_v1"])  # built by code, never the model's
        self.assertTrue(report.simulated)
        [turn] = self.turns("data_science")
        self.assertEqual((turn["sentences_kept"], turn["sentences_refused"]), (3, 2))
        self.assertEqual(turn["refusals"], ["number_not_in_records", "number_not_in_records"])
        journal = agent_arm.journal(self.services, turn["run"])
        [checked] = journal["reports"]
        self.assertIn("About 37% of accounts will not renew.", [r["text"] for r in checked["refused"]])
        [record] = self.services.ledger.records(point="team.agent_report")
        self.assertEqual((record.value_used, record.state["sentences_refused"], record.state["report_refusal"]),
                         ("refused", 2, "number_not_in_records"))

    def test_a_number_the_model_sent_itself_or_an_ids_digits_never_count_as_a_record(self):
        """The review of B3, F1 and F2: a number the model put in a refused call's arguments, a question it asked, or the
        digits of a task id's digest are no record; the person's words and code's outputs are."""
        self.switch("data_science")
        tid = "T-3f9a20c7b1e4"
        answer = ("About 37% of accounts will not renew. 20 accounts are at risk. "
                  "I recommend scoring each account seven days before renewal.")
        self.scripted(said(call("team_plan", plan=PLAN)),
                      said(call("team_read_task", task_id="churn-37%"), call("team_accept", task_id=tid),
                           call("team_act", task_id=tid, action="inspect_schema")),
                      said(call("team_return_report", task_id=tid, outcome="complete", revision=1, answer=answer,
                                limits="Checked on the sample only.")), DONE)
        self.open_renewal(tid)
        self.pump(only={tid})
        [report] = self.view(tid)["reports"]
        self.assertNotIn("37%", report.answer)
        self.assertNotIn("20 accounts", report.answer)
        self.assertIn("seven days before renewal", report.answer)  # the person's own words hold the number
        self.assertIn("Checked on the sample only.", report.limits)  # limits sent as text are one limit (F7)
        [turn] = self.turns("data_science")
        self.assertEqual(turn["refusals"], ["number_not_in_records", "number_not_in_records"])

    def test_a_row_handed_over_again_after_a_fallback_is_the_rules_never_a_model_on_top(self):
        """The review of B3, F3: after a fallback the rules may have written; the same row again runs the rules (they read
        the stored state and repeat nothing), never the model."""
        self.switch("data_science")
        transport = self.scripted(said(content="I would accept it."), said(call("team_plan", plan=PLAN)),
                                  said(call("team_act", task_id="T-renewal", action="inspect_schema")), DONE)
        self.open_renewal()
        row = self.ds_row()
        agents.deliver(self.services, row)  # the model answers without a plan: the rules accept and ask
        self.assertEqual(self.view("T-renewal")["task"].status, "waiting_for_person")
        actions = self.mine("T-renewal")
        agents.deliver(self.services, self.work.delivery(row["id"], self.pid))  # a worker crashed before delivery_done
        self.assertEqual(len(transport.seen), 1)
        self.assertEqual(self.mine("T-renewal"), actions)
        self.assertEqual(self.view("T-renewal")["task"].status, "waiting_for_person")
        self.assertEqual([t["outcome"] for t in self.turns("data_science")], ["fell_back"])
        self.assertEqual([a["fallback"] for a in self.answered("data_science")], ["invalid_output", "invalid_output"])

    def test_a_model_never_records_or_carries_an_answer_the_person_did_not_give(self):
        """The second review of B3, G1: a Business agent on a delivered row records "sara's" answer and carries words of its
        own to Data Science. Both are refused: a person's answer is recorded only from the words that person gave."""
        self.switch("business")
        transport = self.scripted(
            said(call("team_plan", plan="1. Record the answer.\n2. Carry it to Data Science.")),
            said(call("team_record_human_response", task_id="T-renewal", action="confirm_terms", response_id="r-forged", responded_by="sara")),
            said(call("team_answer", task_id="T-renewal", response_id="r-forged", text="It means not renewing; about 37% will not renew.")),
            DONE, rest=DONE)
        self.open_renewal()  # Business opens the case (code, the person's own message)
        self.pump(only={"T-renewal"})  # Data Science (rules) asks what churn means: Business's agent turn on that row
        self.assertTrue(transport.seen)
        view = self.view("T-renewal")
        self.assertFalse([a for a in view["actions"] if a.output_ref == "response:r-forged"])
        self.assertFalse([d for d in view["deliveries"] if d["kind"] == "answer"])
        self.assertEqual(view["task"].status, "waiting_for_person")  # still the person's to answer
        [turn] = [t for t in self.turns("business") if t["outcome"] == "fell_back" and t.get("fallback") == "no_action"]
        journal = agent_arm.journal(self.services, turn["run"])
        self.assertTrue(any("only from the words that person gave" in (s.get("error") or "") for s in journal["steps"]), journal["steps"])
        # every write refused, so the work does not wait on it: the rules took the turn and asked the person the real question
        self.assertTrue([a for a in view["actions"] if a.actor.id == "business" and a.action == "confirm_terms"])
        self.assertEqual([m["text"] for m in self.said("business", "clarification")], [agents.CLARIFY_TARGET])
        self.assertIn("no_action", [a.get("fallback") for a in self.answered("business")])
        # the person's own answer still goes through, as before (code, from their message)
        out = agents.person_replies(self.services, self.pid, "T-renewal", person="sara", text=ANSWER, initiator=self.sara)
        self.assertEqual(out["refused"], [])
        [said_answer] = [d for d in self.view("T-renewal")["deliveries"] if d["kind"] == "answer"]
        self.assertEqual(said_answer["text"], ANSWER)

    def test_sales_and_marketing_agents_never_give_a_tools_values_the_member_did_not_say(self):
        """The third review of B3, R1: a commercial tool reads the member's recorded answer, its values rebuilt from their own
        words; a model that never asked a member cannot make up a capacity or an offer's cost and then report it."""
        for team, action, subject, claim in (
                ("sales", "draft_contact_list", {"r": "none", "capacity": 400}, "Sales can make 400 calls this week."),
                ("marketing", "check_cost", {"r": "none", "amount": 10, "budget": 4000}, "The budget of EUR 4000 funds 400 offers.")):
            with self.subTest(team=team):
                self.switch(team)
                tid = f"T-{team}"
                self.scripted(said(call("team_plan", plan="1. Accept.\n2. Act.\n3. Report.")),
                              said(call("team_accept", task_id=tid), call("team_act", task_id=tid, action=action, subject=json.dumps(subject))),
                              said(call("team_block", task_id=tid, reason="waiting")),
                              said(call("team_return_report", task_id=tid, outcome="blocked", revision=1, answer=claim,
                                        next_owner=team, unblocks="a member answers")), DONE, rest=DONE)
                out = agents.open_case(self.services, self.pid, self.prob, task_id=tid, question="Who should we call, and at what offer?",
                                       thread_id=self.biz_thread, member="sara", initiator=self.sara, owner=team, inputs=("sample:renewal_v1",))
                self.assertEqual(out["refused"], [])
                self.pump(only={tid})
                view = self.view(tid)
                self.assertFalse([a for a in view["actions"] if a.action == action])  # refused: no member's answer named
                [report] = view["reports"]
                self.assertNotIn("400", report.answer)
                [turn] = [t for t in self.turns(team) if t["task"] == tid]
                self.assertIn("number_not_in_records", turn["refusals"])
                journal = agent_arm.journal(self.services, turn["run"])
                self.assertTrue(any("recorded answer" in (s.get("error") or "") for s in journal["steps"]), journal["steps"])

    def test_more_wordings_of_a_claim_and_the_unblocks_text_are_checked(self):
        """The third review of B3, R2 and R3."""
        for sentence, claim in (("Sara confirmed that churn means not renewing.", "person_confirmed"),
                                ("The decision owner signed off on the plan.", "approve"), ("Approval was given by the owner.", "approve"),
                                ("We sent the offer to the accounts.", "contact_customers"),
                                ("The customers have been told about the offer.", "contact_customers"),
                                ("I drafted the call list.", "draft_contact_list"),
                                ("No list was drafted, but we called the customers.", "contact_customers")):
            self.assertIn(claim, core.unrecorded_claims(sentence, ["inspect_schema"]), sentence)
        self.assertEqual(core.unrecorded_claims("Sara confirmed that churn means not renewing.", [core.PERSON]), [])
        self.assertEqual(core.claimed_actions("Data Science confirmed the sample has renewal dates."), [])
        self.assertEqual(core.claimed_actions("Data Science sent its report to the Lead."), [])
        self.switch("data_science")
        self.scripted(said(call("team_plan", plan=PLAN)),
                      said(call("team_accept", task_id="T-renewal"), call("team_block", task_id="T-renewal", reason="the outcome")),
                      said(call("team_return_report", task_id="T-renewal", outcome="blocked", revision=1, answer="The outcome is not confirmed.",
                                next_owner="business", unblocks="About 37% of accounts will not renew; Sara confirmed it.")), DONE)
        self.open_renewal()
        self.pump(only={"T-renewal"})
        [report] = self.view("T-renewal")["reports"]
        self.assertEqual(report.next_task.unblocks, "what this report names as missing")  # code's words, not the model's
        [turn] = self.turns("data_science")
        self.assertEqual(turn["sentences_refused"], 1)

    def test_every_fallback_marks_its_row_so_a_redelivery_never_runs_a_model_on_the_rules_work(self):
        """The third review of B3, R4: no model at first, so the rules write; the worker crashes before the row is marked
        delivered; the model is back. The row is the rules' again. And a row a rules-arm attempt already ran stays the rules'
        after the role is switched to the agent arm."""
        self.switch("data_science")
        self.open_renewal()
        row = self.ds_row()
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            self.services.gateway = None
            agents.deliver(self.services, row)
        actions = self.mine("T-renewal")
        self.assertEqual(self.work.delivery(row["id"], self.pid)["agent_turn"]["state"], "fell_back")
        transport = self.scripted(said(call("team_plan", plan=PLAN)), said(call("team_act", task_id="T-renewal", action="inspect_schema")), DONE)
        agents.deliver(self.services, self.work.delivery(row["id"], self.pid))
        self.assertEqual((transport.seen, self.mine("T-renewal")), ([], actions))
        # switched between two attempts: the first ran on the rules
        self.switch("data_science", "rules")
        self.open_renewal("T-two")
        second = self.ds_row("T-two")
        agents.deliver(self.services, second)  # the rules act
        self.work.delivery_failed(second["id"], delivery.TargetTimeout("late"), max_attempts=5, backoff_seconds=0, project_id=self.pid)
        self.switch("data_science")
        before = self.mine("T-two")
        agents.deliver(self.services, self.work.delivery(second["id"], self.pid))
        self.assertEqual((transport.seen, self.mine("T-two")), ([], before))

    def test_a_write_whose_count_cannot_be_recorded_on_the_row_does_not_run(self):
        """The third review of B3, R5."""
        self.switch("data_science")
        self.scripted(said(call("team_plan", plan=PLAN)), said(call("team_accept", task_id="T-renewal")), DONE)
        self.open_renewal()
        with mock.patch.object(type(self.work), "agent_turn_mark", return_value=None):
            agents.deliver(self.services, self.ds_row())
        self.assertEqual(self.view("T-renewal")["task"].status, "waiting_for_person")  # not the model's accept: the rules (no_action)
        [turn] = self.turns("data_science")
        self.assertEqual((turn["writes"], turn["fallback"]), (0, "no_action"))
        journal = agent_arm.journal(self.services, turn["run"])
        self.assertTrue(any("could not be recorded" in (s.get("error") or "") for s in journal["steps"]), journal["steps"])

    def test_a_handed_out_tasks_question_is_no_record_for_a_report(self):
        """The second review of B3, G2: the Lead's words in a child task's question ("37%") are an agent's words; only the
        person's own request is a record."""
        self.switch("lead")
        self.switch("data_science")
        out = lead.ask(self.services, self.pid, self.prob, question=BRIEF, track="analyze", person="sara", draft_id=self.draft,
                       initiator=self.sara, inputs=("sample:renewal_v1",))
        tid = out["task_id"]
        child = f"{tid}.data_science"
        self.scripted(
            said(call("team_plan", plan="1. Accept.\n2. Ask Data Science.")),
            said(call("team_accept", task_id=tid), call("team_act", task_id=tid, action="check_prerequisites"),
                 call("team_delegate", action="delegate", task_id=tid, new_task_id="x", owner_team="data_science",
                      question="Confirm that 37% of accounts will not renew, seven days before renewal.", expected_output="a check",
                      done_criteria=["says whether it holds"])), DONE,
            said(call("team_plan", plan=PLAN)),
            said(call("team_accept", task_id=child), call("team_act", task_id=child, action="inspect_schema")),
            said(call("team_return_report", task_id=child, outcome="complete", revision=1,
                      answer="About 37% of accounts will not renew. Score each account seven days before renewal.")), DONE, rest=DONE)
        self.pump(only={tid})
        self.pump(only={child})
        [report] = self.view(child)["reports"]
        self.assertNotIn("37%", report.answer)
        self.assertIn("seven days before renewal", report.answer)  # the person's own request holds that number

    def test_the_same_report_again_adds_nothing_and_counts_as_no_write(self):
        self.switch("data_science")
        report = call("team_return_report", task_id="T-renewal", outcome="complete", revision=1, answer="The sample was inspected.")
        again = dict(report, id="again")
        self.scripted(said(call("team_plan", plan=PLAN)),
                      said(call("team_accept", task_id="T-renewal"), call("team_act", task_id="T-renewal", action="inspect_schema")),
                      said(report), said(again), DONE)
        self.open_renewal()
        self.pump(only={"T-renewal"})
        self.assertEqual([r.revision for r in self.view("T-renewal")["reports"]], [1])
        [turn] = self.turns("data_science")
        self.assertEqual(turn["writes"], 3)  # accept, inspect, the report; the second report stood in for nothing
        journal = agent_arm.journal(self.services, turn["run"])
        self.assertTrue(journal["steps"][-1].get("stood_in"), journal["steps"])

    def test_a_report_cannot_claim_an_action_without_its_terminal_record(self):
        self.switch("data_science")
        answer = ("I inspected the schema of the sample. I checked leakage and found no late column. We contacted the "
                  "customers on the list. All at-risk accounts were called yesterday. Nothing was trained and nobody was contacted.")
        self.scripted(*self._feasibility(answer))
        self.open_renewal()
        self.pump(only={"T-renewal"})
        [report] = self.view("T-renewal")["reports"]
        self.assertIn("I inspected the schema of the sample.", report.answer)  # its record ended: kept
        self.assertNotIn("checked leakage", report.answer)  # no check_leakage record: refused
        self.assertNotIn("We contacted", report.answer)  # no tool contacts anyone: always refused
        self.assertIn("Nothing was trained and nobody was contacted.", report.answer)  # a denial is no claim
        [turn] = self.turns("data_science")
        self.assertNotIn("were called", report.answer)  # the passive too (the review of B3, F4)
        self.assertEqual(turn["refusals"], ["claims_unrecorded_action"] * 3)
        self.assertEqual(core.claimed_actions("Sales should call them before renewal."), [])
        self.assertEqual(core.claimed_actions("We called the customers."), ["contact_customers"])
        self.assertEqual(core.unrecorded_claims("I checked leakage.", ["check_leakage"]), [])

    # ------------------------------------------------------------------ 6. budgets end the turn cleanly
    def test_the_step_budget_ends_the_turn_with_what_was_done_and_never_runs_the_rules_on_top(self):
        self.switch("data_science")
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_AGENT_STEPS": "3"}):
            self.scripted(said(call("team_plan", plan=PLAN)), said(call("team_accept", task_id="T-renewal")),
                          said(call("team_act", task_id="T-renewal", action="inspect_schema"),
                               call("team_act", task_id="T-renewal", action="check_leakage")), DONE)
            self.open_renewal()
            self.pump(only={"T-renewal"})
        [turn] = self.turns("data_science")
        self.assertEqual((turn["outcome"], turn["stopped"], turn["writes"]), ("ended", "steps", 2))
        self.assertEqual(self.mine("T-renewal"), ["inspect_schema"])  # the 4th call never ran
        self.assertEqual(len(self.runs("teamspace.data_science")), 1)  # no rules turn after it
        self.assertEqual(self.ds_row()["status"], "delivered")
        [ds] = self.answered("data_science")
        self.assertEqual((ds["acted"], ds.get("fallback")), ("agent", None))

    def test_the_seconds_budget_ends_the_turn_cleanly(self):
        self.switch("data_science")
        # the plan and the accept answer at once; only the third request outlasts the budget. Every request waiting the
        # same time left the accept a margin a busy PostgreSQL gate overran twice on 2026-11-24, whatever the margin
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_AGENT_SECONDS": "6"}):
            self.scripted(said(call("team_plan", plan=PLAN)), said(call("team_accept", task_id="T-renewal")),
                          said(call("team_act", task_id="T-renewal", action="inspect_schema")), DONE, delays=[0, 0, 9.0])
            self.open_renewal()
            self.pump(only={"T-renewal"})
        [turn] = self.turns("data_science")
        self.assertEqual((turn["outcome"], turn["stopped"], turn["writes"]), ("ended", "time", 1))
        self.assertEqual(self.mine("T-renewal"), [])

    def test_the_euro_budget_ends_the_turn_cleanly(self):
        self.switch("data_science")
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_AGENT_EUR": "0.05"}):
            transport = self.scripted(said(call("team_plan", plan=PLAN), cost=0.03), said(call("team_accept", task_id="T-renewal"), cost=0.03),
                                      said(call("team_act", task_id="T-renewal", action="inspect_schema")), DONE)
            self.open_renewal()
            self.pump(only={"T-renewal"})
        [turn] = self.turns("data_science")
        self.assertEqual((turn["outcome"], turn["stopped"], turn["writes"]), ("ended", "euros", 1))
        self.assertAlmostEqual(turn["euros"], 0.06)
        self.assertEqual(len(transport.seen), 2)  # no request past the budget

    def test_a_budget_that_ends_before_any_action_leaves_the_turn_to_the_rules(self):
        self.switch("data_science")
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_AGENT_STEPS": "1"}):
            self.scripted(said(call("team_plan", plan=PLAN)), said(call("team_accept", task_id="T-renewal")))
            self.open_renewal()
            self.pump(only={"T-renewal"})
        [turn] = self.turns("data_science")
        self.assertEqual((turn["outcome"], turn["fallback"], turn["writes"]), ("fell_back", "budget", 0))
        self.assertEqual(self.view("T-renewal")["task"].status, "waiting_for_person")  # the rules accepted and asked
        [ds] = self.answered("data_science")
        self.assertEqual((ds["arm"], ds["acted"], ds["fallback"]), ("agent", "rules", "budget"))

    # ------------------------------------------------------------------ 7. the fallback: before an action only
    def test_a_model_that_cannot_answer_before_any_action_leaves_the_turn_to_the_rules_and_says_why(self):
        self.switch("data_science")
        cases = [({"DCLAB_NO_LIVE_MODELS": "1"}, (), "no_live_models"),
                 ({}, (TimeoutError("late"),), "timeout"),
                 ({}, (said(content="I would accept it."),), "invalid_output"),
                 ({}, (RuntimeError("BudgetExceeded: the project has used its budget"),), "refused"),
                 ({}, (said(call("team_plan", plan=PLAN)), RuntimeError("the endpoint failed")), "error")]
        for n, (env, replies, why) in enumerate(cases):
            with self.subTest(why=why), mock.patch.dict(os.environ, env):
                tid = f"T-fb{n}"
                self.scripted(*replies)
                if not replies:  # no model is configured at all
                    self.services.gateway = None
                self.open_renewal(tid)
                self.pump(only={tid})
                self.assertEqual(self.view(tid)["task"].status, "waiting_for_person")  # the rules accepted and asked
                [turn] = [t for t in self.turns("data_science") if t["task"] == tid]
                self.assertEqual((turn["outcome"], turn["fallback"], turn["writes"]), ("fell_back", why, 0))
                [ds] = [a for a in self.answered("data_science") if a["task"] == tid]
                self.assertEqual((ds["acted"], ds["fallback"]), ("rules", why))

    def test_a_failure_after_an_action_never_runs_the_rules_on_top(self):
        self.switch("data_science")
        self.scripted(said(call("team_plan", plan=PLAN)), said(call("team_accept", task_id="T-renewal")), RuntimeError("the endpoint failed"))
        self.open_renewal()
        self.pump(only={"T-renewal"})
        view = self.view("T-renewal")
        self.assertEqual((view["task"].status, self.mine("T-renewal")), ("accepted", []))  # the rules never asked their question
        [turn] = self.turns("data_science")
        self.assertEqual((turn["outcome"], turn["stopped"], turn["writes"]), ("ended", "error", 1))
        self.assertEqual(len(self.runs("teamspace.data_science")), 1)
        self.assertEqual(self.ds_row()["status"], "delivered")

    def test_a_turn_a_persons_own_message_starts_is_codes(self):
        self.switch("business")
        transport = self.scripted()
        self.open_renewal()  # Business opens the case in the person's words
        self.assertEqual(transport.seen, [])
        [biz] = self.answered("business")
        self.assertEqual((biz["arm"], biz["acted"], biz["fallback"]), ("agent", "rules", "person_entry"))

    # ------------------------------------------------------------------ 8. a delivery handed over again never repeats an action
    def test_a_redelivery_during_and_after_a_running_agent_turn_never_repeats_an_action(self):
        self.switch("data_science")
        gate = threading.Event()
        transport = self.scripted(said(call("team_plan", plan=PLAN)), said(call("team_accept", task_id="T-renewal")), DONE, gate=gate)
        self.open_renewal()
        row = self.ds_row()
        done = {}
        import contextvars

        here = contextvars.copy_context()  # the worker runs in this test's workspace, as a job does in its worker's
        worker = threading.Thread(target=lambda: here.run(lambda: done.setdefault(
            "rows", delivery.run(self.services, {"project_id": self.pid, "task_id": "T-renewal"}))))
        worker.start()
        try:
            for _ in range(200):  # the turn holds its lease once its first request is out
                if transport.seen and (self.work.delivery(row["id"], self.pid).get("agent_turn") or {}).get("state") == "running":
                    break
                threading.Event().wait(0.05)
            held = self.work.delivery(row["id"], self.pid)
            self.assertEqual(held["agent_turn"]["state"], "running")
            self.assertFalse(delivery.is_due(held))  # not due while the turn runs: the relay skips it
            self.assertEqual(delivery.run(self.services, {"project_id": self.pid, "task_id": "T-renewal"}), [])  # a second deliverer waits
            with self.assertRaises(delivery.TemporaryFailure):  # handed over directly: the running turn holds it
                agents.deliver(self.services, held)
            failed = self.work.delivery_failed(row["id"], delivery.TargetTimeout("no answer"), max_attempts=5, backoff_seconds=0,
                                               project_id=self.pid)  # the delivery's own limit: no retry while the lease holds
            self.assertGreaterEqual(datetime.fromisoformat(failed["next_attempt_at"]), datetime.fromisoformat(held["agent_turn"]["until"]))
            self.assertEqual(len(transport.seen), 1)
        finally:
            gate.set()
            worker.join(30)
        self.assertEqual(self.work.delivery(row["id"], self.pid)["agent_turn"]["state"], "done")
        requests = len(transport.seen)
        actions = len(self.view("T-renewal")["actions"])
        status = self.view("T-renewal")["task"].status
        out = agents.deliver(self.services, self.work.delivery(row["id"], self.pid))  # after: the same row again
        self.assertTrue(out["skipped"])
        self.assertEqual((len(transport.seen), len(self.view("T-renewal")["actions"]), self.view("T-renewal")["task"].status),
                         (requests, actions, status))
        self.assertEqual(len([t for t in self.turns("data_science") if t["outcome"] in ("acted", "ended")]), 1)

    def test_a_turn_cut_off_by_a_crash_after_an_action_is_never_run_again(self):
        self.switch("data_science")
        self.open_renewal()
        row = self.ds_row()
        past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(timespec="milliseconds")
        self.work.agent_turn_claim(row["id"], "crashed", "teamspace-dead", seconds=60, project_id=self.pid)
        self.work.agent_turn_mark(row["id"], "crashed", project_id=self.pid, writes=1, until=past)
        transport = self.scripted(said(call("team_plan", plan=PLAN)), DONE)
        self.pump(only={"T-renewal"})
        self.assertEqual(transport.seen, [])  # no second model turn
        self.assertEqual(self.view("T-renewal")["task"].status, "open")  # and no rules on top
        [turn] = self.turns("data_science")
        self.assertEqual((turn["outcome"], turn["stopped"]), ("ended", "interrupted"))
        self.assertEqual(self.ds_row()["status"], "delivered")

    def test_a_turn_cut_off_before_any_action_runs_again(self):
        self.switch("data_science")
        self.open_renewal()
        row = self.ds_row()
        past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(timespec="milliseconds")
        self.work.agent_turn_claim(row["id"], "crashed", "teamspace-dead", seconds=60, project_id=self.pid)
        self.work.agent_turn_mark(row["id"], "crashed", project_id=self.pid, until=past)
        self.scripted(said(call("team_plan", plan=PLAN)), said(call("team_accept", task_id="T-renewal")), DONE)
        self.pump(only={"T-renewal"})
        self.assertEqual(self.view("T-renewal")["task"].status, "accepted")

    # ------------------------------------------------------------------ 9. thinking kept for every agent call
    def test_every_agent_request_keeps_its_thinking_whole(self):
        self.switch("data_science")
        self.scripted(said(call("team_plan", plan=PLAN), why="the plan first"), said(call("team_accept", task_id="T-renewal"), why="ours"),
                      said(content="Done.", why="nothing left"))
        self.open_renewal()
        self.pump(only={"T-renewal"})
        docs = [d for d in exchanges(self.home) if d["purpose"] == "team_agent"]
        self.assertEqual([d["thinking"] for d in docs], [[{"text": "the plan first"}], [{"text": "ours"}], [{"text": "nothing left"}]])
        self.assertTrue(all(d["whole"] for d in docs))
        self.assertIn(json.dumps(PLAN)[1:-1], json.dumps(docs))  # the plan travels whole in the exchange too

    # ------------------------------------------------------------------ the agent arm's own delivery limit
    def test_an_agent_turn_gets_its_own_delivery_limit_and_the_rules_keep_theirs(self):
        self.open_renewal()
        row = self.ds_row()
        base = delivery.timeout_seconds(self.services, row)
        self.switch("data_science")
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_AGENT_SECONDS": "100"}):
            self.assertGreater(delivery.timeout_seconds(self.services, row), base + 100)
        self.switch("data_science", "rules")
        self.assertEqual(delivery.timeout_seconds(self.services, row), base)


for _name in [n for n in dir(_Contract) if n.startswith("test_")]:  # the fixture only: test_teamspace_agents runs its own tests
    setattr(_Agent, _name, None)


class FileTeamAgentTests(_Agent, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}


class PgTeamAgentTests(_Agent, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}


class ClaimsTests(unittest.TestCase):
    def test_a_denial_carries_over_or_and_nor_but_not_over_and(self):
        """The team-arms smoke run (2026-11-24): a helper's "it does not say a model was trained or an account was scored"
        was read as claiming the scoring."""
        self.assertEqual(core.claimed_actions("It does not say a model was trained or an account was scored"), [])
        self.assertEqual(core.claimed_actions("Nothing was drafted, and neither a model was trained nor the accounts scored."), [])
        self.assertEqual(core.claimed_actions("We did not train a model, and we scored the accounts."), ["score"])
        self.assertEqual(core.claimed_actions("We drafted the list or called them."), ["draft_contact_list", "contact_customers"])
        # the review of the fix: a new clause inside the list's item ends the denial's reach
        self.assertEqual(core.claimed_actions("No data was missing or late and we scored every account."), ["score"])
        self.assertEqual(core.claimed_actions("No customer was harmed or upset when Sales called the top ten accounts."), ["contact_customers"])
        self.assertEqual(core.claimed_actions("Neither the label nor the dates were missing when we trained the model."), ["train"])

    def test_claims_denials_and_promises(self):
        self.assertEqual(core.claimed_actions("I inspected the schema and checked for leakage."), ["inspect_schema", "check_leakage"])
        self.assertEqual(core.claimed_actions("No model was trained."), [])
        self.assertEqual(core.claimed_actions("A model was trained on the sample."), ["train"])
        for passive, action in (("The customers on the list have been contacted.", "contact_customers"),
                                ("All at-risk accounts were called yesterday.", "contact_customers"),
                                ("A baseline was defined.", "define_baseline"), ("The schema was inspected.", "inspect_schema")):
            self.assertEqual(core.claimed_actions(passive), [action], passive)
        self.assertEqual(core.claimed_actions("No customers were contacted."), [])
        self.assertEqual(core.claimed_actions("The accounts on the list were not called."), [])
        # B3 review H1: "not just", "not only" and "no fewer than" widen a claim, they never deny it; "no customers" does
        for widened in ("We emailed not just the top accounts but all customers.",
                        "We called not only the at-risk accounts but every customer on the list.",
                        "We contacted no fewer than 30 customers."):
            self.assertEqual(core.unrecorded_claims(widened, []), ["contact_customers"], widened)
        self.assertEqual(core.claimed_actions("We called no customers."), [])
        self.assertEqual(core.unrecorded_claims("We approved the plan.", ["approve"]), ["approve"])  # never recorded by a tool
        self.assertEqual(core.unrecorded_claims("I asked Data Science for a check.", ["delegate"]), [])
        self.assertTrue(core.promises_action("I will call the customers tomorrow."))
        self.assertFalse(core.promises_action("Sales should call the accounts on the list."))
        rec = core.AgentTurnRecord(role="lead", agent="lead", outcome="fell_back", fallback="budget")
        self.assertEqual(rec.schema_version, "dclab_core.team_agent_turn.v1")
        with self.assertRaises(Exception):
            core.AgentTurnRecord(role="lead", agent="lead", outcome="fell_back", fallback="budget", writes=1)
        with self.assertRaises(Exception):
            core.AgentTurnRecord(role="lead", agent="lead", outcome="acted", fallback="budget")


if __name__ == "__main__":
    unittest.main()
