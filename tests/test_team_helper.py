"""B2 of the team workspace (2026-11-23): the helper arm.

Code decides everything; a model helps at named seats, and its answer is used only where a code check passes. Each seat
is tested for: the model's answer used when its check passes; refused when it fails; ``timeout``, ``no_model`` and an
invalid answer falling back to the rules; its thinking stored; and a role on the rules arm running exactly as before.
The owner's example: "customers who quietly leave" is read as non-renewal on a renewal table, and no question is asked.

Scripted transports only: no live call, and no Claude model as a test model. On files and on PostgreSQL
(``tests/run_parallel.py --pg``; never run directly: that empties ``dclab_test``).
"""

import json
import os
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402
from test_teamspace_agents import ANSWER, BRIEF, _Contract  # noqa: E402

from dclab_core import team_brain as core  # noqa: E402
from dclab_rnd import audit as audit_log  # noqa: E402
from dclab_rnd.models import gateway as gw  # noqa: E402
from dclab_rnd.models.usage import FileUsage  # noqa: E402
from dclab_rnd.teamspace import agents, brains, delivery, lead, seats  # noqa: E402

CLEAN = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY",
                         "DCLAB_THINKING", "DCLAB_TEAM_BRAIN", "DCLAB_TEAM_HELPER_SEAT_SECONDS")} | {"OPENAI_API_KEY": "k", "DCLAB_LOG_PROMPTS": "1"}
QUIET = ("Customers who quietly leave: who should Sales call before renewal? Eligible: accounts renewing in the next 60 days. "
         "Decision date: seven days before each renewal.")
VISIT = ("Customers are drifting. Which accounts deserve a personal visit from the account team before renewal? Decision date: "
         "seven days before each renewal.")
PLAIN = "Do you mean customers who cancel, or customers who do not renew by their renewal date?"
SAMPLE = "sample:renewal_v1"
SEAT_OF = {"Seat: read the person's words.": "read_target", "Seat: the teams a request goes to.": "route_teams",
           "Seat: a clarifying question for the person.": "word_question", "Seat: explain the report in plain words.": "explain_report",
           "Seat: the risks you see.": "list_risks"}


def said(content, why="the words name a renewal that does not happen"):
    """A scripted model reply with its thinking (the API's thinking summary)."""
    return {"content": content, "tool_calls": [], "usage": {"input_tokens": 3, "output_tokens": 2}, "thinking": [{"text": why}],
            "finish_reason": "stop", "assistant_message": {"role": "assistant", "content": content}}


class BySeat:
    """A scripted transport that answers by the seat its system prompt names (after-seats run on threads of their own, so
    their order is not fixed). A seat with no reply left raises, as a failing model would."""

    def __init__(self, delay=0.0, **replies):
        self.replies, self.delay, self.seen, self.thinking = {k: list(v) for k, v in replies.items()}, delay, [], None

    def complete(self, messages, tools=None, max_tokens=1800, **options):
        system = messages[0]["content"]
        seat = next((v for k, v in SEAT_OF.items() if k in system), "?")
        self.seen.append({"seat": seat, "messages": messages, "options": options})
        if self.delay:
            import time

            time.sleep(self.delay)
        left = self.replies.get(seat) or []
        if not left:
            raise RuntimeError(f"no scripted reply for {seat}")
        reply = left.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return said(reply, why=f"the {seat} thought") if isinstance(reply, str) else reply


def exchanges(home):
    return [json.loads(line) for p in sorted(Path(home).rglob("model_exchanges/**/*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]


def _join_seat_threads():
    seats.drain(20)
    for t in list(threading.enumerate()):
        if t.name == "dclab-team-helper-seat":
            t.join(15)


class CoreSeatTests(unittest.TestCase):
    def test_the_seats_each_role_has_and_the_portable_checks(self):
        self.assertEqual(set(core.BEFORE_SEATS) | set(core.AFTER_SEATS), set(core.SEATS))
        self.assertEqual(set(core.ROLE_SEATS), set(core.ROLES))
        self.assertIn("read_target", core.ROLE_SEATS["business"])
        self.assertIn("route_teams", core.ROLE_SEATS["lead"])
        self.assertIn("word_question", core.ROLE_SEATS["data_science"])
        self.assertIsNone(core.check_question(agents.CLARIFY_TARGET, options=agents.CLARIFY_OPTIONS))  # the rules' own wording passes
        self.assertIsNone(core.check_question(PLAIN, options=agents.CLARIFY_OPTIONS))
        for text, why in (("", "empty"), ("Do you mean customers who cancel?", "drops_an_option"),
                          ("Within 30 days, do you mean cancel or not renew?", "has_number"),
                          ("Do you mean cancel, or not renew? I will call them to check.", "not_a_question"),
                          ("We will call them: do you mean cancel, or not renew?", "promises_action"),
                          ("Do you mean cancel or not renew", "not_a_question"), ("Do you mean cancel or not renew? " * 10, "too_long")):
            self.assertEqual(core.check_question(text, options=agents.CLARIFY_OPTIONS), why, text)
        self.assertEqual((core.check_code("sales", ("sales", "none")), core.check_code("marketing", ("sales", "none")),
                          core.check_code("UNSURE", ("sales",))), (None, "not_allowed", "unsure"))
        r = core.SeatRecord(seat="read_target", role="business", agent="data_science", task="T-1", outcome="used", used="model",
                            answer="non_renewal_by_date", request="mr-1", seconds=0.5)
        self.assertEqual(core.SeatRecord.read(r.model_dump(mode="json")), r)
        with self.assertRaises(Exception):
            core.SeatRecord(seat="read_target", role="business", agent="data_science", outcome="used", used="model", answer="Non renewal!")

    def test_the_explanation_and_the_risks_are_checked_by_code(self):
        source = "Status: complete.\nAnswer: The sample has renewal dates and mature outcomes. We can score seven days before renewal."
        shown, why = seats.check_explanation("The data can rank accounts seven days before renewal [report].", source)
        self.assertEqual((shown, why), ("The data can rank accounts seven days before renewal.", None))
        for text, why in (("About 40% of accounts will leave [report].", "number_not_in_report"),
                          ("The model is production-ready [report].", "overclaims"), ("It can rank accounts.", "no_citation"),
                          ("It can rank accounts [report]. About 40% leave [report].", "number_not_in_report")):
            self.assertEqual(seats.check_explanation(text, source), (None, why), text)
        self.assertEqual(seats.check_risks(["The outcomes are a sample, not every account."], source)[1], None)
        for items, why in (([], "empty"), ("one risk", "empty"), (["x"] * 6, "too_many"), (["Churn may reach 40%."], "number_not_in_report"),
                           (["This proves churn falls."], "overclaims"), (["We will call each account."], "promises_action")):
            self.assertEqual(seats.check_risks(items, source), (None, why), items)


class _Helper(_Contract):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ, CLEAN)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(_join_seat_threads)  # a late seat request never writes into the next test's records
        self.addCleanup(brains.drain, 10)
        from dclab_rnd.ledger import open_ledger
        from dclab_rnd.models.routing import open_routing

        self.services.ledger = open_ledger(self.home, self.services.workspace_id)
        self.routing = open_routing(self.home)
        self.audit = audit_log.open_audit(self.home)
        self.draft = self.services.drafts.create("Renewal on Home")["id"]

    def scripted(self, delay=0.0, **replies):
        transport = BySeat(delay, **replies)
        self.services.gateway = gw.Gateway(FileUsage(self.home / "model_requests.jsonl"), transport=lambda tier, purpose: transport,
                                           routing=self.routing)
        return transport

    def switch(self, role, arm="helper"):
        return brains.switch(self.routing, self.audit, role, arm, "owner", "compare the helper arm on the dev set")

    def seat_rows(self, seat=None):
        return [e.payload for e in self.services.experience.read(kinds=["team.helper.seat"]) if seat is None or e.payload["seat"] == seat]

    def open_case(self, question, task_id="T-renewal", sample=SAMPLE):
        out = agents.open_case(self.services, self.pid, self.prob, task_id=task_id, question=question, thread_id=self.biz_thread,
                               member="sara", initiator=self.sara, inputs=(sample,))
        self.assertEqual(out["refused"], [])
        return out

    def tools(self, agent):
        return [s.payload.get("tool") for s in self.steps(f"teamspace.{agent}")]

    # ------------------------------------------------------------------ seat 1: read the person's words
    def test_customers_who_quietly_leave_are_read_as_non_renewal_and_no_question_is_asked(self):
        self.switch("business")
        transport = self.scripted(read_target=['{"event": "non_renewal_by_date"}'])
        self.open_case(QUIET)
        self.pump()
        self.assertTrue(brains.drain(10))
        view = self.view("T-renewal")
        self.assertEqual(view["task"].status, "complete")
        self.assertNotIn("team_clarify", self.tools("data_science"))  # no needless question
        target = view["task"].target
        self.assertEqual((target.event, target.confirmed_by, target.words), ("non_renewal_by_date", "sara", QUIET))
        self.assertTrue(seats.read_by_helper(target))
        [report] = view["reports"]
        said_in = " ".join(a for e in report.evidence for a in e.assumptions) + " ".join(report.limits)
        self.assertIn("read from sara's request by the helper", said_in)  # the report says who read it, and that sara can correct it
        [seen] = transport.seen
        shown = seen["messages"][0]["content"] + seen["messages"][1]["content"]
        self.assertIn("quietly leave", shown)
        self.assertNotIn("renewed_by_date", shown)  # the helper never sees which event the table records: it could only echo it
        for words in ("reasoning", "explain your", "your steps", "step by step", "<thinking>"):
            self.assertNotIn(words, shown.lower())
        [row] = self.seat_rows()
        self.assertEqual((row["seat"], row["role"], row["agent"], row["outcome"], row["used"], row["answer"]),
                         ("read_target", "business", "data_science", "used", "model", "non_renewal_by_date"))
        [doc] = exchanges(self.home)  # the thinking is kept whole beside the answer
        self.assertEqual((doc["purpose"], doc["whole"], doc["model_request"], doc["thinking"]),
                         ("team_helper", True, row["request"], [{"text": "the read_target thought"}]))
        [record] = self.services.ledger.records(point="team.helper_seat")
        self.assertEqual((record.used, record.value_used, record.state["helper_seat"], record.state["seat_outcome"]),
                         ("model_role", "model", "read_target", "used"))
        confirmed = [e.payload for e in self.services.experience.read(kinds=["team.target.confirmed"])]
        self.assertEqual([c.get("read_by") for c in confirmed], ["helper"])
        ds = [e.payload for e in self.services.experience.read(kinds=["team.brain.answered"]) if e.payload["agent"] == "data_science"]
        self.assertTrue(all(r["arm"] == "rules" for r in ds))  # Data Science stays on the rules: only Business's helper read the words

    def test_a_reading_the_table_does_not_record_or_an_answer_not_allowed_is_refused_and_the_rules_ask(self):
        self.switch("business")
        self.scripted(read_target=['{"event": "explicit_cancellation"}', '{"event": "churn"}'])
        self.open_case(QUIET)
        self.pump()
        self.open_case(QUIET, task_id="T-second")
        self.pump()
        for tid in ("T-renewal", "T-second"):
            self.assertEqual(self.view(tid)["task"].status, "waiting_for_person")  # the rules asked, as before
            self.assertIsNone(self.view(tid)["task"].target)
        rows = self.seat_rows("read_target")
        self.assertEqual([(r["outcome"], r["used"], r.get("check")) for r in rows],
                         [("check_failed", "rule", "label_event_mismatch"), ("check_failed", "rule", "not_allowed")])
        self.assertEqual(len(exchanges(self.home)), 2)  # their exchanges and thinking are kept all the same

    def test_unsure_no_model_invalid_output_and_a_timeout_leave_the_rules_to_ask(self):
        self.switch("business")
        self.scripted(read_target=['{"event": "unsure"}', "it is probably non-renewal"])
        self.open_case(QUIET, task_id="T-a")
        self.open_case(QUIET, task_id="T-b")
        self.pump()
        self.services.gateway = None  # the installed gateway: none serves the purpose with live models off
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            self.open_case(QUIET, task_id="T-c")
            self.pump()
        # a model slower than the seat's seconds: the rules' answer, recorded as a timeout; the delivery is not held for it
        self.scripted(delay=6.0, read_target=['{"event": "non_renewal_by_date"}'])
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_HELPER_SEAT_SECONDS": "2"}):
            self.open_case(QUIET, task_id="T-d")
            self.pump()
        for tid in ("T-a", "T-b", "T-c", "T-d"):
            self.assertEqual(self.view(tid)["task"].status, "waiting_for_person", tid)
        self.assertTrue(seats.drain(15))  # the late request ends; only then is its seat recorded, with the request id (review F8)
        rows = {r["task"]: r for r in self.seat_rows("read_target")}
        self.assertEqual({t: (r["outcome"], r["used"]) for t, r in rows.items()},
                         {"T-a": ("unsure", "rule"), "T-b": ("invalid_output", "rule"), "T-c": ("no_live_models", "rule"),
                          "T-d": ("timeout", "rule")})
        self.assertLess(rows["T-d"]["seconds"], 5.0)
        late = [d for d in exchanges(self.home) if d["model_request"] == rows["T-d"].get("request")]
        self.assertEqual([d["thinking"] for d in late], [[{"text": "the read_target thought"}]])  # its exchange, joined to its seat
        self.assertEqual(self.services.ledger.records(point="team.helper_seat")[0].used, "rule")

    def test_the_answer_the_keyword_reader_cannot_read_is_read_by_the_helper(self):
        self.switch("business")
        self.scripted(read_target=['{"event": "unsure"}', '{"event": "non_renewal_by_date"}'])
        self.open_case(BRIEF)
        self.pump()
        self.assertEqual(self.view("T-renewal")["task"].status, "waiting_for_person")
        words = "the ones who just quietly drift away at contract end"
        out = agents.person_replies(self.services, self.pid, "T-renewal", person="sara", text=words, initiator=self.sara)
        self.assertEqual(out["refused"], [])
        self.pump()
        task = self.view("T-renewal")["task"]
        self.assertEqual((task.status, task.target.event, task.target.words, task.target.confirmed_by),
                         ("complete", "non_renewal_by_date", words, "sara"))
        self.assertFalse(seats.read_by_helper(task.target))  # the person answered: their words, the helper's reading of them
        self.assertEqual(seats.reading(task.target), "answer")
        [report] = self.view("T-renewal")["reports"]
        limits = " ".join(report.limits)
        self.assertIn("as the helper read sara's answer", limits)  # review F7: never "as sara confirmed" for the helper's reading
        self.assertNotIn("as sara confirmed", limits)
        self.assertEqual([r["outcome"] for r in self.seat_rows("read_target")], ["unsure", "used"])
        confirmed = [e.payload for e in self.services.experience.read(kinds=["team.target.confirmed"])]
        self.assertEqual([c.get("read_by") for c in confirmed], ["helper"])

    # ------------------------------------------------------------------ seat 2: the clarifying question
    def test_the_question_is_worded_plainly_where_its_check_passes_and_the_rules_word_it_otherwise(self):
        self.switch("data_science")
        transport = self.scripted(word_question=[json.dumps({"question_id": "confirm_target", "question": PLAIN}),
                                                 json.dumps({"question_id": "confirm_target", "question": "Within 30 days, cancel or not renew?"}),
                                                 json.dumps({"question_id": "approve_now", "question": PLAIN})])
        for tid in ("T-a", "T-b", "T-c"):  # one at a time: each turn takes the next scripted wording
            self.open_case(BRIEF, task_id=tid)
            self.pump()
        asked = {t: [a for a in self.view(t)["actions"] if a.action == "confirm_target" and a.actor.id == "data_science"] for t in ("T-a", "T-b", "T-c")}
        questions = {m["handoff"]["task_id"]: m["text"] for m in self.said("business", "clarification")}
        self.assertEqual(questions, {"T-a": PLAIN, "T-b": agents.CLARIFY_TARGET, "T-c": agents.CLARIFY_TARGET})
        self.assertTrue(all(asked.values()))
        rows = {r["task"]: r for r in self.seat_rows("word_question")}
        self.assertEqual({t: (r["outcome"], r.get("check")) for t, r in rows.items()},
                         {"T-a": ("used", None), "T-b": ("check_failed", "has_number"), "T-c": ("check_failed", "not_allowed")})
        self.assertEqual({s["seat"] for s in transport.seen}, {"word_question"})
        self.assertEqual(len(exchanges(self.home)), 3)
        self.assertTrue(all(d["thinking"] == [{"text": "the word_question thought"}] for d in exchanges(self.home)))
        acted = [e.payload for e in self.services.experience.read(kinds=["team.brain.answered"])
                 if e.payload["agent"] == "data_science" and e.payload.get("request")]
        self.assertEqual(sorted(r["acted"] for r in acted), ["helper", "rules", "rules"])  # the helper acted where its wording was used

    def test_the_question_falls_back_to_the_rules_with_no_model_or_a_timeout(self):
        self.switch("data_science")
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            self.open_case(BRIEF, task_id="T-a")
            self.pump()
        self.scripted(delay=6.0, word_question=[json.dumps({"question_id": "confirm_target", "question": PLAIN})])
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_HELPER_SEAT_SECONDS": "2"}):
            self.open_case(BRIEF, task_id="T-b")
            self.pump()
        questions = {m["handoff"]["task_id"]: m["text"] for m in self.said("business", "clarification")}
        self.assertEqual(questions, {"T-a": agents.CLARIFY_TARGET, "T-b": agents.CLARIFY_TARGET})
        self.assertTrue(seats.drain(15))
        self.assertEqual({r["task"]: r["outcome"] for r in self.seat_rows("word_question")}, {"T-a": "no_live_models", "T-b": "timeout"})

    # ------------------------------------------------------------------ the delivery's limit: the seats' seconds, for the helper only
    def test_a_helper_turn_gets_the_seats_seconds_on_top_of_the_delivery_limit_and_the_rules_keep_theirs(self):
        row = {"project_id": self.pid, "task_id": "T-renewal", "to_team": "data_science"}
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_DELIVERY_TIMEOUT_SECONDS": "2", "DCLAB_TEAM_HELPER_SEAT_SECONDS": "6"}):
            self.open_case(BRIEF)
            self.assertEqual(delivery.timeout_seconds(self.services, row), 2.0)  # every role on the rules: unchanged
            self.switch("sales")
            self.assertEqual(delivery.timeout_seconds(self.services, row), 2.0)  # Sales has no before-seat in this turn
            self.switch("business")  # the requester's helper may read the person's words in Data Science's turn
            self.assertEqual(delivery.timeout_seconds(self.services, row), 8.0)
            # review F6: Business's own turns never wait on a seat, so a row to Business keeps the rules' limit
            self.assertEqual(delivery.timeout_seconds(self.services, {**row, "to_team": "business"}), 2.0)
            self.switch("business", "rules")
            self.switch("data_science")
            self.assertEqual(delivery.timeout_seconds(self.services, row), 8.0)
            # a helper slower than the rules' limit but inside the seats' seconds: the delivery is not cut off, its answer is used
            self.scripted(delay=3.0, word_question=[json.dumps({"question_id": "confirm_target", "question": PLAIN})])
            self.pump()
        rows = [r for r in self.work.deliveries(project_id=self.pid) if r["to_team"] == "data_science"]
        self.assertTrue(rows and all(r["status"] == "delivered" and not r.get("attempts") for r in rows), rows)
        self.assertEqual([m["text"] for m in self.said("business", "clarification")], [PLAIN])

    # ------------------------------------------------------------------ the Lead: the request's teams
    def lead_ask(self, question):
        out = lead.ask(self.services, self.pid, self.prob, question=question, track="analyze", person="sara", draft_id=self.draft,
                       initiator=self.sara, inputs=(SAMPLE,))
        self.assertEqual(out["refused"], [])
        return out

    def test_the_lead_adds_sales_where_the_helper_reads_outreach_the_keyword_rule_missed(self):
        self.switch("lead")
        transport = self.scripted(route_teams=['{"team": "sales"}', '{"team": "marketing"}'],
                                  read_target=['{"event": "unsure"}', '{"event": "unsure"}'])
        first = self.lead_ask(VISIT)
        self.pump()
        view = self.view(first["task_id"])
        self.assertEqual([c.owner_team for c in view["children"]], ["data_science", "sales"])
        second = self.lead_ask(VISIT + " Please be quick.")
        self.pump()
        self.assertEqual([c.owner_team for c in self.view(second["task_id"])["children"]], ["data_science"])  # "marketing" is not allowed
        rows = [r for r in self.seat_rows("route_teams")]
        self.assertEqual([(r["outcome"], r.get("answer"), r.get("check")) for r in rows],
                         [("used", "sales", None), ("check_failed", "marketing", "not_allowed")])
        self.assertEqual([s["seat"] for s in transport.seen].count("route_teams"), 2)  # asked once per request, never again on later turns
        self.assertEqual(self.view(f"{first['task_id']}.sales")["task"].status, "blocked")  # Sales plans within its capacity; contacts nobody
        self.assertIn([{"text": "the route_teams thought"}], [d["thinking"] for d in exchanges(self.home) if d["purpose"] == "team_helper"])

    def test_the_lead_on_the_rules_routes_as_before_and_asks_no_model(self):
        transport = self.scripted(route_teams=['{"team": "sales"}'])
        out = self.lead_ask(VISIT)
        self.pump()
        self.assertEqual([c.owner_team for c in self.view(out["task_id"])["children"]], ["data_science"])
        self.assertEqual((transport.seen, self.seat_rows(), exchanges(self.home)), ([], [], []))

    # ------------------------------------------------------------------ seats 3 and 4: the report explained, the risks
    def test_the_report_is_explained_and_its_risks_listed_after_the_turn_and_the_report_is_unchanged(self):
        self.switch("data_science")
        explanation = "The sample can rank accounts seven days before renewal, on its columns and dates only [report]."
        self.scripted(word_question=[json.dumps({"question_id": "confirm_target", "question": PLAIN})],
                      explain_report=[json.dumps({"explanation": explanation}), json.dumps({"explanation": "About 40% will leave [report]."})],
                      list_risks=[json.dumps({"risks": ["The outcomes come from a sample, not every account."]}),
                                  json.dumps({"risks": ["The ranking is production-ready."]})])
        self.renewal_to_the_end()
        self.assertTrue(brains.drain(15))
        [report] = self.view("T-renewal")["reports"]
        notes = self.said("business", "helper_note")
        self.assertEqual(sorted(m["handoff"]["seat"] for m in notes), ["explain_report", "list_risks"])
        texts = {m["handoff"]["seat"]: m["text"] for m in notes}
        self.assertIn("seven days before renewal", texts["explain_report"])
        self.assertNotIn("[report]", texts["explain_report"])
        self.assertTrue(texts["list_risks"].startswith("Risks the helper sees (advice only"))
        self.assertTrue(all(m["handoff"]["item_id"] == report.id for m in notes))
        rows = {r["seat"]: r for r in self.seat_rows() if r["seat"] in core.AFTER_SEATS}
        self.assertEqual({s: (r["outcome"], r["report"]) for s, r in rows.items()},
                         {"explain_report": ("used", report.id), "list_risks": ("used", report.id)})
        self.assertEqual(self.view("T-renewal")["reports"], [report])  # the report itself is never changed
        # a second report whose explanation writes a number the report does not, and whose risk overclaims: both dropped
        out = agents.confirm_outcome(self.services, self.pid, "T-renewal", person="sara", text="Only an explicit cancellation counts.",
                                     initiator=self.sara)
        self.assertTrue(out["changed"])
        self.pump()
        self.assertTrue(brains.drain(15))
        later = [r for r in self.seat_rows() if r["seat"] in core.AFTER_SEATS and r["report"] != report.id]
        self.assertEqual(sorted((r["seat"], r["outcome"], r.get("check")) for r in later),
                         [("explain_report", "check_failed", "number_not_in_report"), ("list_risks", "check_failed", "overclaims")])
        self.assertEqual(len(self.said("business", "helper_note")), 2)  # nothing more was shown
        kinds = {d["thinking"][0]["text"] for d in exchanges(self.home)}
        self.assertTrue({"the explain_report thought", "the list_risks thought"} <= kinds)

    def test_an_explanation_or_a_risk_that_claims_an_action_never_done_is_dropped(self):
        """B4's smoke run (2026-11-24): the after-seats checked numbers, never claims. A helper's words claim nothing the
        task tree's finished actions do not back; a denial over "or" is no claim."""
        self.switch("data_science")
        denial = "It does not say a model was trained or an account was scored [report]."
        self.scripted(word_question=[json.dumps({"question_id": "confirm_target", "question": PLAIN})],
                      explain_report=[json.dumps({"explanation": "We called the customers on the list [report]."})],
                      list_risks=[json.dumps({"risks": ["The model was trained on the sample, so it may overfit."]})])
        self.renewal_to_the_end()
        self.assertTrue(brains.drain(15))
        rows = sorted((r["seat"], r["outcome"], r.get("check")) for r in self.seat_rows() if r["seat"] in core.AFTER_SEATS)
        self.assertEqual(rows, [("explain_report", "check_failed", "claims_unrecorded_action"),
                                ("list_risks", "check_failed", "claims_unrecorded_action")])
        self.assertEqual(self.said("business", "helper_note"), [])  # nothing was shown
        view = self.view("T-renewal")
        done = seats.done_actions(self.work, self.pid, view)
        self.assertIn("inspect_schema", done)
        source = seats._report_text(view["reports"][0])
        self.assertEqual(seats.check_explanation(denial, source, done)[1], None)
        self.assertEqual(seats.check_explanation("The schema was inspected [report].", source, done)[1], None)  # on record
        self.assertEqual(seats.check_risks(["An account was scored with a model."], source, done)[1], "claims_unrecorded_action")

    def test_after_seats_with_no_model_or_invalid_output_show_nothing(self):
        self.switch("data_science")
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            self.renewal_to_the_end()
            self.assertTrue(brains.drain(15))
        rows = [r for r in self.seat_rows() if r["seat"] in core.AFTER_SEATS]
        self.assertEqual(sorted((r["seat"], r["outcome"]) for r in rows), [("explain_report", "no_live_models"), ("list_risks", "no_live_models")])
        self.assertEqual(self.said("business", "helper_note"), [])

    def test_after_seats_past_the_cap_ask_nothing_and_are_recorded_busy_once_each(self):
        self.switch("data_science")
        held = [object() for _ in range(brains.MAX_LIVE)]
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            self.open_renewal()
            self.pump()
            with brains._LIVE_LOCK:
                brains._LIVE.update(held)
            try:
                out = agents.person_replies(self.services, self.pid, "T-renewal", person="sara", text=ANSWER, initiator=self.sara)
                self.assertEqual(out["refused"], [])
                self.pump()
            finally:
                with brains._LIVE_LOCK:
                    brains._LIVE.difference_update(held)
        self.assertEqual(self.view("T-renewal")["task"].status, "complete")  # the report stands
        rows = [r for r in self.seat_rows() if r["seat"] in core.AFTER_SEATS]
        self.assertEqual(sorted((r["seat"], r["outcome"]) for r in rows), [("explain_report", "busy"), ("list_risks", "busy")])

    # ------------------------------------------------------------------ the review of B2
    def test_the_events_are_listed_in_one_order_whatever_the_table_records(self):
        """F1: the table's events come first in ``fixtures.offered``; a prompt in that order would tell the model which event
        the table records, and a model that always names the first would pass the label check on every table."""
        self.switch("business")
        transport = self.scripted(read_target=['{"event": "unsure"}', '{"event": "unsure"}'])
        self.open_case(QUIET, task_id="T-a", sample="sample:renewal_v1")
        self.open_case(QUIET, task_id="T-b", sample="sample:renewal_no_dates")
        self.pump()
        prompts = [s["messages"] for s in transport.seen]
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0], prompts[1])

    def test_a_reading_is_neither_asked_nor_saved_for_a_member_who_lost_access(self):
        """F3: the helper's save is no tool call, so it makes the guard's check itself, before asking and before writing."""
        from dclab_rnd.teamspace import tools

        self.switch("business")
        transport = self.scripted(read_target=['{"event": "non_renewal_by_date"}'])
        self.open_case(QUIET)
        turn = self.turn("data_science", "T-renewal", row={"kind": "task"}, initiator=self.sara)
        seats.attach(turn, "rules")
        with mock.patch.object(tools, "_authority", return_value="Refused: lost access."):
            seats.before_rules(turn)
        self.assertIsNone(self.view("T-renewal")["task"].target)
        self.assertEqual((transport.seen, self.seat_rows()), ([], []))

    def test_the_lead_keeps_sales_after_a_stop_between_two_delegations(self):
        """F4: the helper answered "sales" and the Lead delegated Data Science, then its run stopped before Sales. On the next
        turn the children exist, the seat is not asked again, and its earlier answer still puts Sales in the route."""
        from types import SimpleNamespace

        self.switch("lead")
        transport = self.scripted(route_teams=['{"team": "sales"}'])
        task = SimpleNamespace(id="L-1", brief_revision=1, question=VISIT, allowed_inputs=("help:analyze",))
        view = {"task": task, "children": []}
        first = self.turn("lead", "L-1")
        seats.attach(first, "helper")
        self.assertEqual(agents._route(first, view), ("data_science", "sales"))
        later = self.turn("lead", "L-1")
        seats.attach(later, "helper")
        stopped = {"task": task, "children": [SimpleNamespace(id="L-1.data_science")]}
        self.assertEqual(agents._route(later, stopped), ("data_science", "sales"))
        self.assertEqual(len(transport.seen), 1)
        other = self.turn("lead", "L-2")
        seats.attach(other, "helper")
        self.assertEqual(agents._route(other, {"task": SimpleNamespace(id="L-2", brief_revision=1, question=VISIT,
                                                                      allowed_inputs=("help:analyze",)),
                                               "children": [SimpleNamespace(id="L-2.data_science")]}), ("data_science",))

    def test_the_route_falls_back_to_the_rules_with_no_model_or_a_timeout(self):
        self.switch("lead")
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            first = self.lead_ask(VISIT)
            self.pump()
        self.scripted(delay=6.0, route_teams=['{"team": "sales"}'])
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_HELPER_SEAT_SECONDS": "2"}):
            second = self.lead_ask(VISIT + " Please be quick.")
            self.pump()
        for out in (first, second):
            self.assertEqual([c.owner_team for c in self.view(out["task_id"])["children"]], ["data_science"])
        self.assertTrue(seats.drain(15))
        self.assertEqual([r["outcome"] for r in self.seat_rows("route_teams")], ["no_live_models", "timeout"])

    def test_the_answer_read_falls_back_to_the_unsure_blocker_on_a_timeout(self):
        self.switch("business")
        self.scripted(delay=6.0, read_target=['{"event": "unsure"}', '{"event": "non_renewal_by_date"}'])
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_HELPER_SEAT_SECONDS": "2"}):
            self.open_case(BRIEF)
            self.pump()
            self.assertTrue(seats.drain(15))
            out = agents.person_replies(self.services, self.pid, "T-renewal", person="sara", text="the ones who quietly drift away",
                                        initiator=self.sara)
            self.assertEqual(out["refused"], [])
            self.pump()
        task = self.view("T-renewal")["task"]
        self.assertEqual((task.status, task.target), ("blocked", None))  # the rules' "unsure": a blocker naming who confirms it
        self.assertTrue(seats.drain(15))
        self.assertEqual([r["outcome"] for r in self.seat_rows("read_target")], ["timeout", "timeout"])

    def test_a_reading_by_the_leads_helper_is_never_called_confirmed_in_ml_engineerings_report(self):
        """F2: ML Engineering says how the outcome was settled: the helper read the request; nobody confirmed it."""
        self.switch("lead")
        self.scripted(read_target=['{"event": "non_renewal_by_date"}'])
        out = lead.ask(self.services, self.pid, self.prob, question=QUIET, track="model", person="sara", draft_id=self.draft,
                       initiator=self.sara, inputs=(SAMPLE,))
        self.assertEqual(out["refused"], [])
        self.pump()
        self.assertTrue(brains.drain(15))
        ml = self.view(f"{out['task_id']}.ml_engineering")
        self.assertIsNotNone(ml, "ML Engineering was asked after Data Science answered")
        [report] = ml["reports"]
        self.assertIn("as the helper read sara's request (not confirmed by sara", report.answer)
        self.assertNotIn("as confirmed with Data Science", report.answer)
        science = self.view(f"{out['task_id']}.data_science")["task"]
        self.assertTrue(seats.read_by_helper(science.target))

    def test_a_reading_by_another_roles_helper_is_not_credited_to_data_sciences_arm(self):
        """F5: Business's reading is asked in Data Science's turn; Data Science's turn record says what its own arm did."""
        self.switch("business")
        self.switch("data_science")
        self.scripted(read_target=['{"event": "non_renewal_by_date"}'])
        self.open_case(QUIET)
        self.pump()
        self.assertTrue(brains.drain(15))
        ds = [e.payload for e in self.services.experience.read(kinds=["team.brain.answered"]) if e.payload["agent"] == "data_science"]
        self.assertTrue(ds)
        self.assertTrue(all(r["acted"] == "rules" for r in ds), ds)  # its own seats did not act; Business's helper did
        self.assertEqual([r["role"] for r in self.seat_rows("read_target")], ["business"])

    # ------------------------------------------------------------------ the rules arm: exactly as before
    def test_a_rules_role_reaches_no_seat_and_asks_nothing(self):
        transport = self.scripted(read_target=['{"event": "non_renewal_by_date"}'], word_question=['{"question_id": "confirm_target"}'])
        self.open_case(QUIET)
        self.pump()
        self.assertEqual(self.view("T-renewal")["task"].status, "waiting_for_person")
        self.assertEqual([m["text"] for m in self.said("business", "clarification")], [agents.CLARIFY_TARGET])
        self.assertEqual((transport.seen, self.seat_rows(), exchanges(self.home)), ([], [], []))
        self.assertEqual(self.services.ledger.records(point="team.helper_seat"), [])
        out = agents.person_replies(self.services, self.pid, "T-renewal", person="sara", text=ANSWER, initiator=self.sara)
        self.assertEqual(out["refused"], [])
        self.pump()
        self.assertEqual(self.view("T-renewal")["task"].status, "complete")
        self.assertEqual(transport.seen, [])


for _name in [n for n in dir(_Contract) if n.startswith("test_")]:  # the fixture only: test_teamspace_agents runs its own tests
    setattr(_Helper, _name, None)


class FileTeamHelperTests(_Helper, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}


class PgTeamHelperTests(_Helper, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}


if __name__ == "__main__":
    unittest.main()
