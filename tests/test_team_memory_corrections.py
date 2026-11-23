"""W6 of the collaborative workspace (2026-11-20): a team's own memory, and corrections across teams.

The owner's plan (``Dev_Plans/Collaborative_Agentic_Workspace_2026-11-20.md``, "What the agents remember"): Sales'
working notes and draft exceptions stay with Sales (another team and the shared summary never see them) until Sales shares
a fact on purpose, with its source. The correction test: Sales confirms capacity for 50 calls, then corrects it to 30;
the task, its report and the decision that rested on 50 become stale, Business is alerted in its own thread, the decision
waits for its owner's review, and work that did not rest on it is untouched, in this session and the next. Old memory
items (v1, v2) read unchanged. On files and on PostgreSQL (``tests/run_parallel.py --pg``; never run directly: that empties
``dclab_test``).
"""

import contextvars
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402

from dclab_core import memory as core_mem  # noqa: E402
from dclab_core import team_work as tw  # noqa: E402
from dclab_rnd import context, experience  # noqa: E402
from dclab_rnd.memory import FileMemory, Refused, open_memory, view  # noqa: E402
from dclab_rnd.memory import bases, summary  # noqa: E402
from dclab_rnd.memory.scope import TEAM_STAMP, AccessDenied, AuthorizedScope, create_problem, resolve_scope  # noqa: E402
from dclab_rnd.storage import open_stores, open_team_work  # noqa: E402
from dclab_rnd.teamspace import store as ts  # noqa: E402
from dclab_rnd.teamspace.threads import open_threads  # noqa: E402

T0, T1 = "2026-11-20T09:00:00+00:00", "2026-11-20T09:05:00+00:00"
SALES, MARKETING, BUSINESS = "team_sales01", "team_mkt0001", "team_biz0001"
MEMBERS = {"sam": (SALES,), "mia": (MARKETING,), "sara": (BUSINESS,), "dana": (BUSINESS,)}
SALES_AGENT = tw.Actor(kind="agent", id="sales", team="sales")
MKT_AGENT = tw.Actor(kind="agent", id="marketing", team="marketing")
SARA = tw.Actor(kind="person", id="sara", team="business")
DANA = tw.Actor(kind="person", id="dana", team="business")
T2 = "2026-11-20T09:10:00+00:00"
NOTE = "Territory North is excluded from outreach this quarter."
FIFTY, THIRTY = "Sales can call 50 accounts this week.", "Sales can call 30 accounts this week."
PASSWORD = "a long enough password"


class _Teams:
    """The W2 store's one call the scope reads (``teams_of``); PostgreSQL's teams need accounts, so both runs use this."""

    def teams_of(self, user_id):
        return [{"id": t} for t in MEMBERS.get(user_id, ())]


class _Contract:
    env: dict = {}

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"DCLAB_AUTH": "", **self.env})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.home = Path(tempfile.mkdtemp()) / "workspace"
        self.home.mkdir()
        from dclab_rnd.storage import db

        self.projects, self.drafts, _ = open_stores(self.home)
        self.work = open_team_work(self.home, projects=self.projects)
        self.ws = self.work.workspace
        wid = db.workspace_for(self.home) if db.database_url(required=False) else None
        self.log = experience.open_log(self.home, wid)
        from dclab_rnd.ledger import open_ledger

        self.ledger = open_ledger(self.home, wid)  # a real ledger: a refused record would otherwise go unseen (the review of W7)
        token = context.set_services(SimpleNamespace(experience=self.log, ledger=self.ledger))
        self.addCleanup(context.reset_services, token)
        self.pid = self.projects.create("Renewal retention", goal="Who should Sales call")["id"]
        self.prob = create_problem(self.projects, self.pid, "Renewal", workspace_id=self.ws)["problem_id"]
        self.threads = open_threads(self.home)
        self.biz_thread = self.threads.open_thread(self.pid, self.prob, "business", label="Business")[0]["id"]
        self.sales_thread = self.threads.open_thread(self.pid, self.prob, "sales", label="Sales")[0]["id"]

    def scope(self, who, role="data_scientist"):
        person = SimpleNamespace(user_id=who, role=role, via="local")
        return resolve_scope(workspace_id=self.ws, principal=person, projects=self.projects, project_id=self.pid,
                             problem_id=self.prob, teams=_Teams())

    # ------------------------------------------------------------------ helpers
    def task(self, id, owner, inputs=()):
        return tw.TeamTask(id=id, workspace_id=self.ws, project_id=self.pid, problem_id=self.prob, brief_revision=1,
                           requester=tw.Requester(team="business", member="sara", thread_id=self.biz_thread), owner_team=owner,
                           question=f"{id}: what can your team do this week?", expected_output="a plan within the confirmed numbers",
                           done_criteria=["names its numbers and their source"], allowed_inputs=list(inputs), created_at=T0)

    def finished_task(self, id, owner, actor, action, inputs, evidence):
        self.work.create_task(self.task(id, owner, inputs))
        self.work.move(self.pid, id, "accepted", actor)
        source = tw.DIRECTORY[owner].binding(action)
        run = tw.TeamAction(task_id=id, action_id=f"{id}-a1", actor=actor, action=action, source=source, input_revision="r1",
                            status="running", started_at=T0, retry_key=tw.retry_key(id, action, "r1"))
        self.work.record_action(self.pid, run)
        self.work.finish_action(self.pid, id, run.action_id, "succeeded", ended_at=T1, output_ref=f"draft:{id}")
        return self.work.report_task(self.pid, id, revision=1, current_brief_revision=1, answer=f"{id} answered.",
                                     next_task=tw.NextStep(closed=True), move_to="complete", by=actor, evidence=evidence).item

    def decision(self, id, constraints, evidence):
        return tw.TeamDecision(id=id, problem_id=self.prob, decision_owner=tw.Actor(kind="person", id="dana", team="business"),
                               option=f"{id}: call the shortlisted accounts", constraint_revisions=list(constraints),
                               evidence_revisions=list(evidence))

    def approved(self, id, constraints=(), evidence=(), when=T1, **over):
        """The decision as dana approved it."""
        doc = {**self.decision(id, constraints, evidence).model_dump(), "approval": {"who": DANA.model_dump(), "when": when}, **over}
        return tw.TeamDecision.model_validate(doc)

    def owner(self):
        """dana, holding the role that may approve a team's decision."""
        return self.scope("dana", role="owner")

    def events(self, *kinds):
        return self.log.read(kinds=list(kinds))

    def stored(self, id):
        return next(d["decision"] for d in ts.decisions(self.home, self.scope("sara")) if d["decision"].id == id)

    def biz_messages(self):
        return self.threads.messages(self.biz_thread)

    # ------------------------------------------------------------------ a team's own memory
    def test_a_team_note_is_invisible_to_another_team_and_to_the_shared_summary(self):
        sam, mia, sara = self.scope("sam"), self.scope("mia"), self.scope("sara")
        self.assertEqual(sam.team_ids, (SALES,))
        note = bases.upsert_base_fact(self.home, sam, target=bases.team_target(SALES), subject="territory", field="exception",
                                      text=NOTE, source="thread:sales")["entry"]
        draft = bases.propose_draft(self.home, sam, target=bases.team_target(SALES), subject="territory", field="exception",
                                    text="Territory South may be excluded too.", source_type="sales_agent", source_id="d1")["entry"]
        self.assertEqual(draft["status"], "draft")
        bases.upsert_base_fact(self.home, sara, target="problem", subject="churn", field="target", text="Failure to renew.",
                               source="thread:business")
        # Sales reads its own base, summary and memory
        mine = bases.read_base(self.home, sam, bases.team_target(SALES))
        self.assertEqual([e["text"] for e in mine["active"]], [NOTE])
        self.assertIsNone(mine["project_id"])  # a team's base belongs to no one project
        team_summary = summary.read_team_summary(self.home, sam, SALES)
        self.assertIn(NOTE, team_summary["text"])
        self.assertIn("Territory South", team_summary["text"])
        self.assertEqual(team_summary["visibility_class"], f"team:{SALES}")
        self.assertTrue(summary.read_team_summary(self.home, sam, SALES)["from_cache"])
        store = open_memory(self.home)
        self.assertIn(note["item_id"], [i.id for i in view(store, sam.team_chain(SALES))])
        self.assertNotIn(note["item_id"], [i.id for i in view(store, sam.chain())])  # only in the team's own conversation
        self.assertNotIn(note["item_id"], [i.id for i in view(store, sam.shared_chain())])
        # Marketing, and Business, see none of it
        for other in (mia, sara):
            with self.assertRaises(AccessDenied):
                bases.read_base(self.home, other, bases.team_target(SALES))
            with self.assertRaises(AccessDenied):
                summary.read_team_summary(self.home, other, SALES)
            with self.assertRaises(AccessDenied):
                bases.upsert_base_fact(self.home, other, target=bases.team_target(SALES), subject="x", field="y", text="Forged.", source="x")
            self.assertNotIn(note["item_id"], [i.id for i in view(store, other.chain())])
            self.assertNotIn(("team", core_mem.team_scope_id(TEAM_STAMP, SALES)), other.chain(include_teams=True))
            with self.assertRaises(AccessDenied):
                other.team_chain(SALES)
        # the shared summary never holds a team's fact, whoever reads it
        for reader in (sam, mia, sara):
            shared = summary.read_summary(self.home, reader)
            self.assertNotIn("Territory", shared["text"])
            self.assertNotIn(note["item_id"], shared["active_fact_ids"])
            self.assertIn("Failure to renew.", shared["text"])
            self.assertNotIn("team", [s for s, _ in reader.shared_chain()])
        no_teams = AuthorizedScope(workspace_id=self.ws, principal_id="sam", role="data_scientist", project_id=self.pid,
                                   problem_id=self.prob)
        self.assertNotIn("team", [s for s, _ in no_teams.chain()])  # a scope resolved without the teams store: no team memory

    def test_a_deliberate_share_appears_in_the_problem_base_with_its_source(self):
        sam, mia = self.scope("sam"), self.scope("mia")
        note = bases.upsert_base_fact(self.home, sam, target=bases.team_target(SALES), subject="territory", field="exception",
                                      text=NOTE, source="thread:sales")["entry"]
        draft = bases.propose_draft(self.home, sam, target=bases.team_target(SALES), subject="territory", field="draft",
                                    text="Territory South may be excluded too.", source_type="sales_agent", source_id="d1")["entry"]
        with self.assertRaises(Refused):
            bases.share_team_fact(self.home, sam, SALES, draft["entry_id"])
        with self.assertRaises(AccessDenied):
            bases.share_team_fact(self.home, mia, SALES, note["entry_id"])
        shared = bases.share_team_fact(self.home, sam, SALES, note["entry_id"])["entry"]
        self.assertEqual(shared["text"], NOTE)
        self.assertEqual(shared["source"], f"team:{SALES}:{note['item_id']}@r1")
        problem = bases.read_base(self.home, mia, "problem")
        self.assertEqual([(e["text"], e["source"]) for e in problem["active"]], [(NOTE, shared["source"])])
        self.assertIn(NOTE, summary.read_summary(self.home, mia)["text"])
        self.assertNotIn("Territory South", summary.read_summary(self.home, mia)["text"])  # the draft stays with Sales
        team = bases.read_base(self.home, sam, bases.team_target(SALES))
        self.assertEqual(([e["entry_id"] for e in team["active"]], team["revision"]), ([note["entry_id"]], 1))  # unchanged by the share (a draft moves no revision)

    # ------------------------------------------------------------------ the owner's correction test
    def test_capacity_50_corrected_to_30_marks_exactly_the_dependent_work_stale_and_alerts_business(self):
        sam, sara = self.scope("sam"), self.scope("sara")
        fifty = bases.upsert_base_fact(self.home, sam, target="problem", subject="sales", field="capacity", text=FIFTY,
                                       source="thread:sales")["entry"]
        cite = ts.fact_citation(fifty)
        self.assertEqual(cite, f"fact:{fifty['entry_id']}@r1")
        self.finished_task("T-capacity", "sales", SALES_AGENT, "draft_contact_list", [cite], [ts.fact_evidence(fifty)])
        offer = tw.EvidenceRef(source="sample:offer_v1", ref="offer:v1", input_revision="v1")
        self.finished_task("T-offer", "marketing", MKT_AGENT, "check_eligibility", ["sample:offer_v1"], [offer])
        calls = ts.record_decision(self.home, sara, self.decision("D-calls", [cite], ["T-capacity@r1"]), actor=SARA, thread_id=self.biz_thread)
        ts.record_decision(self.home, sara, self.decision("D-offer", ["offer:v1"], ["T-offer@r1"]), actor=SARA, thread_id=self.biz_thread)
        self.assertFalse(calls["stale"])
        self.assertEqual(ts.stale_marks(self.home, sara), [])
        before = {t: self.work.task_view(self.pid, t) for t in ("T-capacity", "T-offer")}
        posted_before = len(self.biz_messages())

        out = bases.correct_base_fact(self.home, sam, fifty["entry_id"], THIRTY, reason="Two callers are out this week.")

        stale = {f"team_task:{self.pid}/T-capacity", f"work_report:{self.pid}/T-capacity@r1", "team_decision:D-calls"}
        self.assertEqual(set(out["team_work"]["stale"]), stale)
        self.assertEqual({d["ref"] for d in ts.stale_marks(self.home, sara)}, stale)
        by_id = {d["decision"].id: d for d in ts.decisions(self.home, sara)}
        self.assertTrue(by_id["D-calls"]["stale"] and by_id["D-calls"]["needs_review"])
        self.assertFalse(by_id["D-offer"].get("stale") or by_id["D-offer"].get("needs_review"))
        for d in ts.stale_marks(self.home, sara):
            self.assertIn(fifty["entry_id"], [c["old_entry_id"] for c in d["changed_sources"]])
            self.assertIn(fifty["entry_id"], d["alerted"])
        # Business is told once, in its own thread, by the Sales agent; nothing else is posted anywhere
        new = self.biz_messages()[posted_before:]
        self.assertEqual(len(new), 1)
        alert = new[0]
        self.assertEqual((alert["author"]["kind"], alert["author"]["id"]), ("agent", "sales"))
        for words in ("50 accounts", "30 accounts", "T-capacity", "T-capacity@r1", "D-calls", "needs dana's review"):
            self.assertIn(words, alert["text"])
        self.assertNotIn("T-offer", alert["text"])
        self.assertNotIn("D-offer", alert["text"])
        self.assertEqual(self.threads.messages(self.sales_thread), [])
        # the team work itself is not rewritten: the stale mark is the problem's, the records keep their snapshots
        for t in ("T-capacity", "T-offer"):
            after = self.work.task_view(self.pid, t)
            self.assertEqual((after["task"], after["reports"], after["actions"]), (before[t]["task"], before[t]["reports"], before[t]["actions"]))
        self.assertEqual(bases.read_base(self.home, sara, "problem")["active"][0]["text"], THIRTY)

        # a new session: new stores read the same marks, and reconciling again tells nobody twice
        fresh_threads, fresh_work = open_threads(self.home), open_team_work(self.home)
        again = ts.mark_fact_dependents_stale(self.home, self.scope("sara"), work=fresh_work, threads=fresh_threads)
        self.assertEqual((set(again["stale"]), again["alerts"]), (stale, []))
        self.assertEqual(len(fresh_threads.messages(self.biz_thread)), posted_before + 1)
        self.assertEqual({d["ref"] for d in ts.stale_marks(self.home, self.scope("dana"))}, stale)

        # a late task still citing the old revision is stale on arrival, and Business hears of it once more
        self.work.create_task(self.task("T-late", "sales", [cite]))
        late = ts.mark_fact_dependents_stale(self.home, sara)
        self.assertEqual(set(late["stale"]), stale | {f"team_task:{self.pid}/T-late"})
        self.assertEqual([a["refs"] for a in late["alerts"]], [[f"team_task:{self.pid}/T-late"]])
        self.assertEqual(len(self.biz_messages()), posted_before + 2)

    def test_a_task_citing_the_new_revision_is_untouched_and_a_second_correction_marks_it_in_the_same_write(self):
        sam, sara = self.scope("sam"), self.scope("sara")
        fifty = bases.upsert_base_fact(self.home, sam, target="problem", subject="sales", field="capacity", text=FIFTY, source="s")["entry"]
        thirty = bases.correct_base_fact(self.home, sam, fifty["entry_id"], THIRTY)["entry"]
        self.work.create_task(self.task("T-thirty", "sales", [ts.fact_citation(thirty)]))
        self.assertEqual(ts.mark_fact_dependents_stale(self.home, sara)["stale"], [])  # linked, not stale
        out = bases.correct_base_fact(self.home, sam, thirty["entry_id"], "Sales can call 25 accounts this week.")
        self.assertEqual(out["stale_dependents"], [f"team_task:{self.pid}/T-thirty"])  # the correction's own write marked it
        self.assertEqual(out["team_work"]["stale"], [f"team_task:{self.pid}/T-thirty"])
        self.assertEqual(len(out["team_work"]["alerts"]), 1)

    def test_a_decision_is_recorded_only_by_a_writer_on_its_own_problem(self):
        viewer = AuthorizedScope(workspace_id=self.ws, principal_id="vic", role="viewer", project_id=self.pid, problem_id=self.prob)
        with self.assertRaises(AccessDenied):
            ts.record_decision(self.home, viewer, self.decision("D-x", [], []), actor=tw.Actor(kind="person", id="vic"))
        other = create_problem(self.projects, self.pid, "Another", workspace_id=self.ws)["problem_id"]
        with self.assertRaises(AccessDenied):
            ts.record_decision(self.home, self.scope("sara"), tw.TeamDecision(
                id="D-y", problem_id=other, decision_owner=tw.Actor(kind="person", id="dana"), option="x"), actor=SARA)
        kept = ts.record_decision(self.home, self.scope("sara"), self.decision("D-z", [], []), actor=SARA)
        self.assertEqual((kept["kind"], kept["stale"], kept["cites"]), ("team_decision", False, []))
        fact = bases.upsert_base_fact(self.home, self.scope("sam"), target="problem", subject="sales", field="capacity", text=FIFTY,
                                      source="s")["entry"]
        with self.assertRaisesRegex(ValueError, "refused citation"):  # a forged revision keeps its own error, and is recorded
            ts.record_decision(self.home, self.scope("sara"), self.decision("D-f", [f"fact:{fact['entry_id']}@r9"], []), actor=SARA)
        self.assertEqual([e.payload["rule"] for e in self.events("team.decision.refused")], ["no_write", "other_problem", "bad_citation"])
        self.assertEqual([d["decision"].id for d in ts.decisions(self.home, self.scope("sara"))], ["D-z"])


    # ------------------------------------------------------------------ the review of 2026-11-20 (findings 2, 3, 5, 6, 7)
    def test_staleness_carries_every_corrected_fact_down_a_chain_to_the_decision(self):
        """Finding 2: child report -> the parent's report joining it -> the decision on that report; a decision already
        stale from one correction still picks up a second correction's fact, and the alert names it for review."""
        sam, sara, mia = self.scope("sam"), self.scope("sara"), self.scope("mia")
        BIZ = tw.Actor(kind="agent", id="business", team="business")
        cap = bases.upsert_base_fact(self.home, sam, target="problem", subject="sales", field="capacity", text=FIFTY, source="s")["entry"]
        offer = bases.upsert_base_fact(self.home, mia, target="problem", subject="marketing", field="offer",
                                       text="The renewal offer is 10 percent off.", source="m")["entry"]
        self.work.create_task(self.task("T-plan", "business"))
        self.work.move(self.pid, "T-plan", "accepted", BIZ)
        self.work.delegate(self.pid, "T-plan", id="T-cap", owner_team="sales", question="How many calls this week?",
                           expected_output="a capacity", done_criteria=["names its source"], by=BIZ, thread_id=self.biz_thread,
                           created_at=T0, allowed_inputs=[ts.fact_citation(cap)])
        self.work.move(self.pid, "T-cap", "accepted", SALES_AGENT)
        run = tw.TeamAction(task_id="T-cap", action_id="c1", actor=SALES_AGENT, action="draft_contact_list", source="tool:draft_list",
                            input_revision="r1", status="running", started_at=T0, retry_key=tw.retry_key("T-cap", "draft_contact_list", "r1"))
        self.work.record_action(self.pid, run)
        self.work.finish_action(self.pid, "T-cap", "c1", "succeeded", ended_at=T1, output_ref="draft:list")
        self.work.report_task(self.pid, "T-cap", revision=1, current_brief_revision=1, answer="50 calls.", next_task=tw.NextStep(closed=True),
                              move_to="complete", by=SALES_AGENT, evidence=[ts.fact_evidence(cap)])
        join = tw.TeamAction(task_id="T-plan", action_id="p1", actor=BIZ, action="reconcile_replies", source="tool:join_replies",
                             input_revision="r1", status="running", started_at=T0, retry_key=tw.retry_key("T-plan", "reconcile_replies", "r1"))
        self.work.record_action(self.pid, join)
        self.work.finish_action(self.pid, "T-plan", "p1", "succeeded", ended_at=T1, output_ref="brief:1")
        parent = self.work.report_task(self.pid, "T-plan", revision=1, current_brief_revision=1, answer="Call 50 accounts.",
                                       next_task=tw.NextStep(closed=True), move_to="complete", by=BIZ).item
        self.assertEqual(parent.child_reports, ["T-cap@r1"])
        ts.record_decision(self.home, sara, self.decision("D-plan", [ts.fact_citation(offer)], ["T-plan@r1"]), actor=SARA, thread_id=self.biz_thread)
        first = bases.correct_base_fact(self.home, mia, offer["entry_id"], "The renewal offer is 5 percent off.")
        self.assertEqual(first["team_work"]["stale"], ["team_decision:D-plan"])
        second = bases.correct_base_fact(self.home, sam, cap["entry_id"], THIRTY)
        chain = {f"team_task:{self.pid}/T-cap", f"work_report:{self.pid}/T-cap@r1", f"work_report:{self.pid}/T-plan@r1",
                 f"team_task:{self.pid}/T-plan", "team_decision:D-plan"}
        self.assertEqual(set(second["team_work"]["stale"]), chain)
        decision = next(d for d in ts.stale_marks(self.home, sara) if d["ref"] == "team_decision:D-plan")
        self.assertEqual({c["old_entry_id"] for c in decision["changed_sources"] if c.get("old_entry_id")},
                         {offer["entry_id"], cap["entry_id"]})
        alert = second["team_work"]["alerts"]
        self.assertEqual(len(alert), 1)
        self.assertEqual(set(alert[0]["refs"]), chain)
        text = self.threads.messages(self.biz_thread)[-1]["text"]
        self.assertIn("decision D-plan (needs dana's review)", text)
        self.assertIn("30 accounts", text)

    # ------------------------------------------------------------------ T5a (2026-11-23): only the decision owner approves
    def test_only_the_decision_owner_holding_the_permission_records_an_approval(self):
        """Finding 3 of 2026-11-20, inverted by T5a: sara recording dana's approval was accepted. Now the approval is the
        acting person's own: the decision owner, holding ``approve_decision``, and never an agent."""
        sara, mia = self.scope("sara"), self.scope("mia")
        first = ts.record_decision(self.home, sara, self.decision("D-1", [], []), actor=SARA, thread_id=self.biz_thread)
        self.assertEqual((first["doc"]["revision"], first["doc"]["schema_version"]), (1, "dclab_core.team_decision.v2"))
        refusals = [(sara, SARA, "not_owner"),                         # another person writes down dana's approval
                    (sara, DANA, "actor_mismatch"),                    # sara's session naming dana as the one acting
                    (self.scope("dana"), DANA, "no_permission"),       # dana, whose role may not approve a team's decision
                    (self.owner(), tw.Actor(kind="agent", id="business", team="business"), "agent"),  # an agent, in dana's own session
                    (self.owner(), tw.Actor(kind="agent", id="dana", team="business"), "agent")]
        for scope, actor, rule in refusals:
            with self.subTest(rule=rule, actor=f"{actor.kind}:{actor.id}"):
                with self.assertRaises(ts.DecisionRefused) as caught:
                    ts.record_decision(self.home, scope, self.approved("D-1"), actor=actor, thread_id=self.biz_thread)
                self.assertEqual(caught.exception.rule, rule)
                self.assertIsInstance(caught.exception, AccessDenied)
                self.assertIsNone(self.stored("D-1").approval)  # nothing was written
        kept = ts.record_decision(self.home, self.owner(), self.approved("D-1"), actor=DANA, thread_id=self.biz_thread)
        self.assertEqual((kept["doc"]["revision"], kept["doc"]["approval"]["who"]["id"]), (2, "dana"))
        fresh = self.stored("D-1")
        self.assertEqual((fresh.approval.who.id, fresh.revision), ("dana", 2))
        other = tw.TeamDecision(id="D-1", problem_id=self.prob, decision_owner=tw.Actor(kind="person", id="mia"), option="call nobody")
        with self.assertRaisesRegex(ts.DecisionRefused, "dana's decision"):
            ts.record_decision(self.home, mia, other, actor=tw.Actor(kind="person", id="mia", team="marketing"))
        self.assertEqual(self.stored("D-1").option, "D-1: call the shortlisted accounts")
        # what is kept for training: one event per accepted write, one per refusal with its rule, and the approval as a decision
        self.assertEqual([(e.payload["decision"], e.payload["what"], e.payload["revision"]) for e in self.events("team.decision.recorded")],
                         [("D-1", "proposed", 1)])
        done = self.events("team.decision.approved")
        self.assertEqual([(e.actor, e.payload["decision"], e.payload["by"], e.payload["revision"]) for e in done], [("person", "D-1", "dana", 2)])
        refused = self.events("team.decision.refused")
        self.assertEqual([e.payload["rule"] for e in refused], [r for _, _, r in refusals] + ["owner_changed"])
        self.assertEqual([e.actor for e in refused], ["person", "person", "person", "agent", "agent", "person"])
        self.assertNotIn("shortlisted", json.dumps([e.payload for e in done + refused + self.events("team.decision.recorded")]))
        point = self.ledger.records(point="team.decision_approve")
        self.assertEqual([(r.subject.key, r.value_used, r.used, r.answers[0].answerer) for r in point],
                         [(f"{self.pid}:D-1", True, "person", "person:dana")])
        self.assertEqual(dict(point[0].state), {"evidence_reports": 0, "constraints": 0, "replaces_earlier": False})

    def test_a_stale_click_is_a_conflict_and_changes_nothing(self):
        sara = self.scope("sara")
        ts.record_decision(self.home, sara, self.decision("D-1", [], []), actor=SARA)
        changed = self.decision("D-1", [], []).model_copy(update={"option": "D-1: call only the top 30"})
        second = ts.record_decision(self.home, sara, changed, actor=SARA, expected_revision=1)
        self.assertEqual(second["doc"]["revision"], 2)
        with self.assertRaises(bases.RevisionConflictError) as caught:  # dana still looks at revision 1
            ts.record_decision(self.home, self.owner(), self.approved("D-1"), actor=DANA, expected_revision=1)
        self.assertEqual((caught.exception.expected_revision, caught.exception.current_revision), (1, 2))
        self.assertEqual(caught.exception.to_detail()["error"], "revision_conflict")
        self.assertEqual((self.stored("D-1").approval, self.stored("D-1").revision), (None, 2))
        with self.assertRaises(bases.RevisionConflictError):  # "I expect no such decision yet"
            ts.record_decision(self.home, sara, self.decision("D-1", [], []), actor=SARA, expected_revision=0)
        self.assertEqual([e.payload["rule"] for e in self.events("team.decision.refused")], ["revision_conflict", "revision_conflict"])
        self.assertEqual(self.events("team.decision.approved"), [])
        seen = self.approved("D-1", option="D-1: call only the top 30")  # having read revision 2, dana approves it
        self.assertEqual(ts.record_decision(self.home, self.owner(), seen, actor=DANA, expected_revision=2)["doc"]["revision"], 3)

    def test_two_approvals_racing_leave_one_approval(self):
        ts.record_decision(self.home, self.scope("sara"), self.decision("D-1", [], []), actor=SARA)
        owner, gate, results = self.owner(), threading.Barrier(2), {}

        def click(n, when):
            gate.wait(timeout=10)
            try:
                results[n] = ts.record_decision(self.home, owner, self.approved("D-1", when=when), actor=DANA, expected_revision=1,
                                                request_id=f"click-{n}")
            except Exception as exc:  # noqa: BLE001 - the loser's refusal is the result
                results[n] = exc
        clicks = [threading.Thread(target=contextvars.copy_context().run, args=(click, n, when)) for n, when in ((1, T1), (2, T2))]
        for t in clicks:
            t.start()
        for t in clicks:
            t.join(timeout=30)
        won = [r for r in results.values() if isinstance(r, dict)]
        lost = [r for r in results.values() if isinstance(r, bases.RevisionConflictError)]
        self.assertEqual((len(won), len(lost)), (1, 1), results)
        self.assertEqual((self.stored("D-1").revision, self.stored("D-1").approval.when), (2, won[0]["doc"]["approval"]["when"]))
        self.assertEqual(len(self.events("team.decision.approved")), 1)
        self.assertEqual(len(self.ledger.records(point="team.decision_approve")), 1)

    def test_a_retry_with_the_same_request_returns_the_stored_entry_and_records_nothing_twice(self):
        ts.record_decision(self.home, self.scope("sara"), self.decision("D-1", [], []), actor=SARA, request_id="req-propose")
        first = ts.record_decision(self.home, self.owner(), self.approved("D-1"), actor=DANA, expected_revision=1, request_id="req-approve")
        again = ts.record_decision(self.home, self.owner(), self.approved("D-1", when=T2), actor=DANA, expected_revision=1,
                                   request_id="req-approve")  # the same click, sent again after its answer was lost
        self.assertEqual(again, first)
        self.assertEqual((again["doc"]["revision"], again["doc"]["approval"]["when"], again["doc"]["request_id"]),
                         (2, tw.Approval(who=DANA, when=T1).when, "req-approve"))
        late = ts.record_decision(self.home, self.scope("sara"), self.decision("D-1", [], []), actor=SARA, request_id="req-propose")
        self.assertEqual(late, first)  # an older request's retry: the decision as it stands now, never a second write
        self.assertEqual(len(self.events("team.decision.approved")), 1)
        self.assertEqual(len(self.events("team.decision.recorded")), 1)
        self.assertEqual(self.events("team.decision.refused"), [])
        self.assertEqual(len(self.ledger.records(point="team.decision_approve")), 1)
        # the review of T5a: the same request id on another write (another person's, another content) is refused aloud,
        # never answered with the stored entry as if it had been kept
        ts.record_decision(self.home, self.scope("sara"), self.decision("D-2", [], []), actor=SARA, request_id="req-2")
        for scope, actor, doc in ((self.owner(), DANA, self.approved("D-2")),
                                  (self.scope("sara"), SARA, self.decision("D-2", [], []).model_copy(update={"option": "D-2: call nobody"}))):
            with self.assertRaises(ts.DecisionRefused) as caught:
                ts.record_decision(self.home, scope, doc, actor=actor, request_id="req-2")
            self.assertEqual(caught.exception.rule, "request_reused")
        self.assertEqual([e.payload["rule"] for e in self.events("team.decision.refused")], ["request_reused", "request_reused"])
        self.assertEqual((self.stored("D-2").approval, self.stored("D-2").revision), (None, 1))

    def test_a_decisions_record_reaches_the_ledger_only_through_the_guarded_write(self):
        """The review of T5a: ``bases.link_dependents`` called with a decision's document and no guard stored an approval
        with no check, no event and no revision."""
        forged = {"ref": "team_decision:D-6", "kind": "team_decision", "doc": json.loads(self.approved("D-6").model_dump_json())}
        with self.assertRaisesRegex(ValueError, "record_decision"):
            bases.link_dependents(self.home, self.scope("sara"), [forged])
        self.assertEqual(ts.decisions(self.home, self.scope("sara")), [])
        ts.record_decision(self.home, self.scope("sara"), self.decision("D-6", [], []), actor=SARA)
        with self.assertRaisesRegex(ValueError, "record_decision"):
            bases.link_dependents(self.home, self.scope("sara"), [forged])
        self.assertIsNone(self.stored("D-6").approval)

    def test_a_decision_supersedes_only_its_own_owners_earlier_decision(self):
        """The review of T5a: "the owner never changes" was bypassed in one hop, by a new decision of another person's
        naming dana's approved decision as the one it replaces."""
        ts.record_decision(self.home, self.owner(), self.approved("D-1"), actor=DANA)
        mia = tw.Actor(kind="person", id="mia", team="marketing")
        theirs = tw.TeamDecision(id="D-2", problem_id=self.prob, decision_owner=mia, option="D-2: call nobody", supersedes="D-1",
                                 approval=tw.Approval(who=mia, when=T2))
        with self.assertRaises(ts.DecisionRefused) as caught:
            ts.record_decision(self.home, self.scope("mia", role="owner"), theirs, actor=mia)
        self.assertEqual(caught.exception.rule, "owner_changed")
        self.assertIn("dana's decision", str(caught.exception))
        self.assertEqual([d["decision"].id for d in ts.decisions(self.home, self.scope("sara"))], ["D-1"])
        self.assertEqual(self.events("team.decision.refused")[-1].payload["rule"], "owner_changed")

    def test_an_approved_decisions_option_changes_only_as_a_new_decision_naming_what_it_replaces(self):
        sam = self.scope("sam")
        fifty = bases.upsert_base_fact(self.home, sam, target="problem", subject="sales", field="capacity", text=FIFTY, source="s")["entry"]
        ts.record_decision(self.home, self.owner(), self.approved("D-1", [ts.fact_citation(fifty)]), actor=DANA, thread_id=self.biz_thread)
        thirty = bases.correct_base_fact(self.home, sam, fifty["entry_id"], THIRTY)["entry"]
        self.assertTrue(next(d for d in ts.decisions(self.home, sam) if d["decision"].id == "D-1")["stale"])
        for change in ({"option": "D-1: call 30 accounts"}, {"constraint_revisions": [ts.fact_citation(thirty)]},
                       {"evidence_revisions": ["T-capacity@r2"]}):
            with self.subTest(change=sorted(change)):
                with self.assertRaises(ts.DecisionRefused) as caught:  # even its owner does not rewrite what was approved
                    ts.record_decision(self.home, self.owner(), self.approved("D-1", [ts.fact_citation(fifty)], **change), actor=DANA)
                self.assertEqual(caught.exception.rule, "immutable")
        kept = self.stored("D-1")
        self.assertEqual((kept.option, kept.constraint_revisions, kept.revision),
                         ("D-1: call the shortlisted accounts", [ts.fact_citation(fifty)], 1))
        new = self.approved("D-2", [ts.fact_citation(thirty)], when=T2, option="D-2: call 30 accounts", supersedes="D-1")
        with self.assertRaises(ts.DecisionRefused) as caught:
            ts.record_decision(self.home, self.owner(), new.model_copy(update={"supersedes": "D-0"}), actor=DANA)
        self.assertEqual(caught.exception.rule, "unknown_superseded")
        made = ts.record_decision(self.home, self.owner(), new, actor=DANA, thread_id=self.biz_thread)
        self.assertEqual((made["doc"]["supersedes"], made["doc"]["revision"], made["stale"]), ("D-1", 1, False))
        self.assertEqual(self.stored("D-1").option, "D-1: call the shortlisted accounts")  # the earlier decision keeps its snapshot
        approvals = self.ledger.records(point="team.decision_approve")
        self.assertEqual([(r.subject.key, dict(r.state)["replaces_earlier"]) for r in approvals],
                         [(f"{self.pid}:D-1", False), (f"{self.pid}:D-2", True)])
        outcome = {"outcome": "22 of 30 renewed", "outcome_at": T2, "outcome_by": SARA.model_dump(), "outcome_measure": "renewed_within_30_days"}
        logged = ts.record_decision(self.home, self.scope("sara"), self.approved("D-2", [ts.fact_citation(thirty)], when=T2,
                                    option="D-2: call 30 accounts", supersedes="D-1", **outcome), actor=SARA)
        self.assertEqual((logged["doc"]["revision"], logged["doc"]["outcome_by"]["id"]), (2, "sara"))
        self.assertEqual([e.payload["what"] for e in self.events("team.decision.recorded")], ["outcome"])
        self.assertNotIn("22 of 30", json.dumps([e.payload for e in self.events("team.decision.recorded")]))

    def test_without_sign_in_the_local_owner_approves_its_own_decision(self):
        """No accounts: the one person on the machine is the owner (``principal_id`` "owner") and holds the permission."""
        from dclab_rnd.accounts.principal import LOCAL_OWNER

        local = resolve_scope(workspace_id=self.ws, principal=LOCAL_OWNER, projects=self.projects, project_id=self.pid,
                              problem_id=self.prob)
        me = tw.Actor(kind="person", id=local.principal_id)
        mine = tw.TeamDecision(id="D-local", problem_id=self.prob, decision_owner=me, option="call the shortlist",
                               approval=tw.Approval(who=me, when=T1))
        kept = ts.record_decision(self.home, local, mine, actor=me, expected_revision=0)
        self.assertEqual((local.principal_id, kept["doc"]["approval"]["who"]["id"], kept["doc"]["revision"]), ("owner", "owner", 1))
        with self.assertRaises(ts.DecisionRefused) as caught:  # dana's decision is still dana's, even to the local owner
            ts.record_decision(self.home, local, self.approved("D-dana"), actor=me)
        self.assertEqual(caught.exception.rule, "not_owner")

    def test_the_correction_writer_and_kinds_t2_calls_keep_codes_and_revisions_only(self):
        """T5a declares, for T2, the ledger point ``team.correction_accept`` and the kinds ``team.correction.applied`` and
        ``team.correction.refused``: each verdict is kept with the two revisions, an unknown verdict is not kept."""
        from dclab_rnd.ledger import capture

        for verdict, base, current in (("apply", 1, 1), ("duplicate", 1, 2), ("stale_base", 1, 3)):
            self.assertTrue(capture.team_correction_accepted(self.pid, "T-eta", verdict, base_revision=base, current_revision=current))
        self.assertIsNone(capture.team_correction_accepted(self.pid, "T-eta", "overwrite", base_revision=1, current_revision=3))
        kept = self.ledger.records(point="team.correction_accept")
        self.assertEqual([(r.value_used, r.used, r.answers[0].answerer, dict(r.state)) for r in kept],
                         [("apply", "rule", "rule:correction_write", {"base_revision": 1, "current_revision": 1}),
                          ("duplicate", "rule", "rule:correction_write", {"base_revision": 1, "current_revision": 2}),
                          ("stale_base", "rule", "rule:correction_write", {"base_revision": 1, "current_revision": 3})])
        self.assertEqual({r.subject.key for r in kept}, {f"{self.pid}:T-eta@r1"})
        done = experience.record("team.correction.applied", "agent", {"task": "T-eta", "problem": self.prob, "by": "data_science",
                                 "by_kind": "agent", "verdict": "apply", "base_revision": 1, "revision": 2, "report": "T-eta@r2"},
                                 log=self.log, project=self.pid)
        late = experience.record("team.correction.refused", "agent", {"task": "T-eta", "problem": self.prob, "by": "data_science",
                                 "by_kind": "agent", "verdict": "stale_base", "base_revision": 1, "revision": 3,
                                 "reason": "based on revision 1; the answer is at revision 3"}, log=self.log, project=self.pid)
        self.assertEqual((done.schema_version, late.payload["verdict"]), ("dclab_core.experience_event.v21", "stale_base"))
        self.assertTrue(all(c in ("meta", "count") for e in (done, late) for c in e.classes.values()))

    def test_a_v1_decision_stored_before_t5a_reads_and_takes_its_next_write(self):
        """A decision kept by the earlier release (``team_decision.v1``, no revision) reads as revision 1, and its owner
        approves it at revision 2."""
        sara = self.scope("sara")
        ts.record_decision(self.home, sara, self.decision("D-old", [], []), actor=SARA)
        v1 = {k: v for k, v in self.decision("D-old", [], []).model_dump(mode="json").items()
              if k not in ("revision", "request_id", "supersedes", "outcome_at", "outcome_by", "outcome_measure", "policy_ref")}
        v1["schema_version"] = "dclab_core.team_decision.v1"
        store, key = bases._store(self.home), sara.problem_key
        store.change("problem", key, {}, lambda doc: (doc["dependents"]["team_decision:D-old"].update(doc=v1, requests={}) or True, None))
        self.assertEqual((self.stored("D-old").revision, self.stored("D-old").schema_version), (1, "dclab_core.team_decision.v2"))
        kept = ts.record_decision(self.home, self.owner(), self.approved("D-old"), actor=DANA, expected_revision=1)
        self.assertEqual((kept["doc"]["revision"], kept["doc"]["schema_version"]), (2, "dclab_core.team_decision.v2"))

    def test_a_team_draft_is_reviewed_only_by_a_member_through_its_base(self):
        """Finding 5."""
        from dclab_rnd.memory import writing

        sam, mia = self.scope("sam"), self.scope("mia")
        draft = bases.propose_draft(self.home, sam, target=bases.team_target(SALES), subject="territory", field="exception",
                                    text="Territory South may be excluded too.", source_type="sales_agent", source_id="d1")["entry"]
        store = open_memory(self.home)
        for who, role in (("mia", "owner"), ("vic", "reviewer"), ("sam", "data_scientist")):
            with self.assertRaises(PermissionError):
                writing.review(store, draft["item_id"], "accept", by=who, role=role)
        with self.assertRaises(AccessDenied):
            bases.confirm_draft(self.home, mia, draft["entry_id"], target=bases.team_target(SALES))
        done = bases.confirm_draft(self.home, sam, draft["entry_id"], target=bases.team_target(SALES))
        self.assertEqual(done["entry"]["status"], "active")

    def test_a_share_never_replaces_a_problem_fact_unless_it_names_it(self):
        """Finding 6."""
        sam, sara = self.scope("sam"), self.scope("sara")
        held = bases.upsert_base_fact(self.home, sara, target="problem", subject="territory", field="exception",
                                      text="No exclusions this quarter.", source="thread:business")["entry"]
        note = bases.upsert_base_fact(self.home, sam, target=bases.team_target(SALES), subject="territory", field="exception",
                                      text=NOTE, source="thread:sales")["entry"]
        with self.assertRaisesRegex(Refused, "supersede"):
            bases.share_team_fact(self.home, sam, SALES, note["entry_id"])
        self.assertEqual([e["text"] for e in bases.read_base(self.home, sara, "problem")["active"]], ["No exclusions this quarter."])
        with self.assertRaises(Refused):
            bases.share_team_fact(self.home, sam, SALES, note["entry_id"], supersede="ent_0000000000")
        out = bases.share_team_fact(self.home, sam, SALES, note["entry_id"], supersede=held["entry_id"])
        self.assertEqual((out["entry"]["supersedes"], out["entry"]["text"]), (held["entry_id"], NOTE))

    def test_a_person_in_two_teams_reads_one_team_per_conversation(self):
        """Finding 7c."""
        MEMBERS["lee"] = (SALES, MARKETING)
        self.addCleanup(MEMBERS.pop, "lee")
        lee = self.scope("lee")
        sales = bases.upsert_base_fact(self.home, lee, target=bases.team_target(SALES), subject="t", field="note", text=NOTE, source="s")["entry"]
        mkt = bases.upsert_base_fact(self.home, lee, target=bases.team_target(MARKETING), subject="o", field="note",
                                     text="The spring campaign is not announced yet.", source="m")["entry"]
        store = open_memory(self.home)
        seen = {t: {i.id for i in view(store, lee.team_chain(t))} for t in (SALES, MARKETING)}
        self.assertIn(sales["item_id"], seen[SALES])
        self.assertNotIn(mkt["item_id"], seen[SALES])
        self.assertIn(mkt["item_id"], seen[MARKETING])
        self.assertNotIn(sales["item_id"], seen[MARKETING])
        default = {i.id for i in view(store, lee.chain())}
        self.assertFalse({sales["item_id"], mkt["item_id"]} & default)

    def test_the_correction_through_the_route_a_forged_revision_and_a_failed_reconciliation(self):
        """Finding 7a, 7b, 7d: the real route; a citation with a forged revision refused and marking nothing; a task linked
        when it is made is marked stale by the correction's own write even when the reconciliation after it fails."""
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        owner = self.scope("owner", role="owner")
        fifty = bases.upsert_base_fact(self.home, owner, target="problem", subject="sales", field="capacity", text=FIFTY, source="s")["entry"]
        with self.assertRaisesRegex(ValueError, "revision 9"):
            self.work.create_task(self.task("T-forged", "sales", [f"fact:{fifty['entry_id']}@r9"]))
        self.assertIsNone(self.work.task(self.pid, "T-forged"))
        with self.assertRaisesRegex(ValueError, "refused citation"):
            bases.link_dependents(self.home, owner, [{"ref": "team_task:x/T", "kind": "team_task", "cites": [f"fact:{fifty['entry_id']}@r9"]}])
        self.work.create_task(self.task("T-capacity", "sales", [ts.fact_citation(fifty)]))
        linked = bases.read_base(self.home, owner, "problem")["dependents"]
        self.assertEqual([(d["ref"], d["stale"]) for d in linked], [(f"team_task:{self.pid}/T-capacity", False)])  # linked when made
        viewer = AuthorizedScope(workspace_id=self.ws, principal_id="vic", role="viewer", project_id=self.pid, problem_id=self.prob)
        with self.assertRaises(AccessDenied):
            ts.mark_fact_dependents_stale(self.home, viewer)

        app = create_app(self.home)
        client = TestClient(app, headers={"x-dclab-token": app.state.services.csrf})
        with mock.patch("dclab_rnd.teamspace.store.mark_fact_dependents_stale", side_effect=RuntimeError("down")):
            res = client.post(f"/api/projects/{self.pid}/memory/base/{fifty['entry_id']}/correct",
                              json={"text": THIRTY, "expected_revision": 1, "problem_id": self.prob})
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["stale_dependents"], [f"team_task:{self.pid}/T-capacity"])  # the correction's own write
        self.assertEqual((body["team_work"]["error"], body["team_work"]["reconciled"]), ("RuntimeError", False))
        self.assertEqual({d["ref"] for d in ts.stale_marks(self.home, owner)}, {f"team_task:{self.pid}/T-capacity"})
        before = len(self.biz_messages())
        again = client.post(f"/api/projects/{self.pid}/memory/base/reconcile", params={"problem_id": self.prob})  # the retry route
        self.assertEqual(again.status_code, 200, again.text)
        self.assertEqual(len(again.json()["alerts"]), 1)  # it finishes the work: Business is told once
        self.assertEqual(len(self.biz_messages()), before + 1)
        third = client.post(f"/api/projects/{self.pid}/memory/base/reconcile", params={"problem_id": self.prob})
        self.assertEqual(third.json()["alerts"], [])  # idempotent: a second retry changes nothing

    def test_a_teams_memory_base_through_its_routes(self):
        """W6 follow-up: the team base is read and written through its own routes, and never reaches the problem base.
        Without sign-in the local owner is the only person and cannot be placed in a team (members need accounts), so the
        routes treat them as every team's owner (review of 2026-11-22: they answered 403 to everyone; no mock of the
        person's teams here). A team that does not exist is a 404."""
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        from dclab_rnd.storage import open_teams

        teams = open_teams(self.home)
        sales = teams.create_team("Sales", "sales")["id"]
        other = teams.create_team("Marketing", "marketing")["id"]
        app = create_app(self.home)
        client = TestClient(app, headers={"x-dclab-token": app.state.services.csrf})
        made = client.post(f"/api/workspace/teams/{sales}/memory/base", json={"subject": "sales", "field": "capacity",
                                                                            "text": "12 calls a day per rep", "expected_revision": 0})
        self.assertEqual(made.status_code, 200, made.text)
        entry = made.json()["entry"]
        fixed = client.post(f"/api/workspace/teams/{sales}/memory/base/{entry['entry_id']}/correct",
                            json={"text": "10 calls a day per rep", "expected_revision": made.json()["revision"]})
        self.assertEqual(fixed.status_code, 200, fixed.text)
        view = client.get(f"/api/workspace/teams/{sales}/memory/base").json()
        self.assertEqual((view["scope"], view["team_id"]), ("team", sales))
        self.assertEqual([e["text"] for e in view["active"]], ["10 calls a day per rep"])
        self.assertEqual(len(view["retired"]), 1)  # the history is kept
        problem = client.get(f"/api/projects/{self.pid}/memory/base", params={"problem_id": self.prob}).json()
        self.assertNotIn("10 calls a day per rep", str(problem))  # a team's note never reaches the shared base
        theirs = client.get(f"/api/workspace/teams/{other}/memory/base")
        self.assertEqual(theirs.status_code, 200, theirs.text)  # the owner reads every team's memory
        self.assertEqual((theirs.json()["team_id"], theirs.json()["active"]), (other, []))  # and Sales' note is not Marketing's
        for answer in (client.get("/api/workspace/teams/team_nosuch/memory/base"),
                       client.post("/api/workspace/teams/team_nosuch/memory/base",
                                   json={"subject": "s", "field": "f", "text": "t", "expected_revision": 0}),
                       client.post(f"/api/workspace/teams/team_nosuch/memory/base/{entry['entry_id']}/correct",
                                   json={"text": "t", "expected_revision": 1})):
            self.assertEqual(answer.status_code, 404, answer.text)
        teams.delete_team(sales)  # a removed team: its memory stays in the records, its routes answer 404
        self.assertEqual(client.get(f"/api/workspace/teams/{sales}/memory/base").status_code, 404)


class OldItemsTests(unittest.TestCase):
    def test_v1_and_v2_items_read_unchanged_and_a_team_item_is_v3(self):
        for version, scope, sid in (("v1", "project", "p1"), ("v2", "problem", "local:p1:prob_1")):
            text = f"Target is churn within 30 days ({version})."
            doc = {"schema_version": f"dclab_core.memory_item.{version}", "id": core_mem.item_id("fact", scope, sid, text),
                   "kind": "fact", "scope": scope, "scope_id": sid, "text": text, "source": "note:1", "who": "person:owner",
                   "status": "active", "correction": False, "created_at": "2026-10-28T09:00:00+00:00"}
            loaded = core_mem.MemoryItem.read(doc)
            self.assertEqual((loaded.id, loaded.scope, loaded.schema_version), (doc["id"], scope, "dclab_core.memory_item.v3"))
            home = Path(tempfile.mkdtemp())
            (home / "memory").mkdir()
            (home / "memory" / "items.jsonl").write_text(json.dumps(doc) + "\n", encoding="utf-8")
            self.assertEqual([i.id for i in FileMemory(home).log()], [doc["id"]])
        with self.assertRaises(ValueError):
            core_mem.MemoryItem.read({**doc, "schema_version": "dclab_core.memory_item.v4"})
        key = core_mem.team_scope_id("w1", "team_sales")
        self.assertEqual(key, "w1:team_sales")
        for bad in (("", "t"), ("w", ""), ("w:x", "t"), ("w", "t:x")):
            with self.assertRaises(ValueError):
                core_mem.team_scope_id(*bad)
        item = core_mem.new("fact", "team", key, NOTE, "thread:sales", "person:sam", "2026-11-20T09:00:00+00:00")
        self.assertEqual(core_mem.MemoryItem.read(item.model_dump_json()), item)
        self.assertEqual(core_mem.SCOPES.index("team"), core_mem.SCOPES.index("person") + 1)
        self.assertEqual(core_mem.SCOPES.index("team") + 1, core_mem.SCOPES.index("project"))
        project = core_mem.new("fact", "project", "p1", "Goal: keep renewals.", "n", "person:sam", "2026-11-20T09:00:00+00:00")
        self.assertEqual([i.id for i in core_mem.resolve([item, project], [("project", "p1")])], [project.id])  # not this reader's team
        self.assertEqual([i.id for i in core_mem.resolve([item, project], [("team", key), ("project", "p1")])], [project.id, item.id])


class FileTeamMemoryTests(_Contract, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}

    def test_the_teams_store_gives_the_person_their_teams(self):
        from dclab_rnd.storage import open_teams

        teams = open_teams(self.home)
        sales = teams.create_team("Sales", "sales")
        teams.add_member(sales["id"], "sam")
        person = SimpleNamespace(user_id="sam", role="data_scientist", via="local")
        scope = resolve_scope(workspace_id=self.ws, principal=person, projects=self.projects, project_id=self.pid,
                              problem_id=self.prob, teams=teams)
        self.assertEqual(scope.team_ids, (sales["id"],))
        self.assertEqual(scope.team_chain(sales["id"])[-1], ("team", core_mem.team_scope_id(TEAM_STAMP, sales["id"])))
        self.assertEqual(scope.to_record()["team_ids"], [sales["id"]])
        teams.remove_member(sales["id"], "sam")
        self.assertEqual(scope.with_teams(teams).team_ids, ())  # a removed member reads the team's memory no more


class PgTeamMemoryTests(_Contract, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}

    def test_team_memory_survives_a_move_from_files_to_postgresql(self):
        """Finding 1: the team key does not carry the workspace id the store answers with (``local`` on files, ``w…`` on
        PostgreSQL), so a migrated team keeps its notes."""
        from dclab_rnd.storage.migrate import migrate
        from dclab_rnd.studio.store import ProjectStore

        source = Path(tempfile.mkdtemp()) / "files-workspace"
        on_files = AuthorizedScope(workspace_id="local", principal_id="sam", role="data_scientist", team_ids=(SALES,))
        with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": ""}):
            ProjectStore(source / "projects")
            note = bases.upsert_base_fact(source, on_files, target=bases.team_target(SALES), subject="territory", field="exception",
                                          text=NOTE, source="thread:sales")["entry"]
        report = migrate(source, self.home, self.url)
        self.assertTrue(report["ok"], report)
        moved = self.scope("sam")
        self.assertNotEqual(moved.workspace_id, "local")
        base = bases.read_base(self.home, moved, bases.team_target(SALES))
        self.assertEqual([e["entry_id"] for e in base["active"]], [note["entry_id"]])
        self.assertIn(note["item_id"], [i.id for i in view(open_memory(self.home), moved.team_chain(SALES))])
        self.assertIn(NOTE, summary.read_team_summary(self.home, moved, SALES)["text"])


class SignedInTeamMemoryRoutesTests(unittest.TestCase):
    """With password sign-in on PostgreSQL (review of 2026-11-22, no mock of the person's teams): the owner who is not in
    a team reads and writes its memory (the owner's role writes), a member reads and writes it, and a member of the
    workspace who is not in the team is refused (403), as a viewer member is when writing."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            raise unittest.SkipTest(f"Studio dependencies not installed: {error}")
        url = pgtest.require()
        pgtest.empty(url)
        cls.env = mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": url, "DCLAB_AUTH": "password", "OPENAI_API_KEY": ""})
        cls.env.start()
        from dclab_rnd.accounts.store import Accounts

        cls.TestClient = TestClient
        cls.app = create_app(Path(tempfile.mkdtemp()))
        cls.main = TestClient(cls.app)
        cls.main.__enter__()
        accounts = Accounts()
        w1 = cls.app.state.services.workspace_id
        cls.users = {}
        for name, role in (("owner1", "owner"), ("ds_in", "data_scientist"), ("ds_out", "data_scientist"), ("view_in", "viewer")):
            user = accounts.create_user(f"{name}-{os.getpid()}@example.com", name.title(), password=PASSWORD)
            accounts.set_member(w1, user["id"], role)
            cls.users[name] = user
        cls.clients = {name: cls.sign_in(name) for name in cls.users}

    @classmethod
    def tearDownClass(cls):
        cls.main.__exit__(None, None, None)
        cls.env.stop()

    @classmethod
    def sign_in(cls, name):
        client = cls.TestClient(cls.app)
        anonymous = client.get("/api/config").json()["csrf"]
        r = client.post("/api/auth/password", json={"email": cls.users[name]["email"], "password": PASSWORD},
                        headers={"X-DCLab-Token": anonymous})
        assert r.status_code == 200, r.text
        client.headers["X-DCLab-Token"] = r.json()["csrf"]
        return client

    def test_the_owner_reads_every_team_and_a_non_member_is_refused(self):
        owner, ds_in, ds_out, view_in = (self.clients[n] for n in ("owner1", "ds_in", "ds_out", "view_in"))
        sales = owner.post("/api/workspace/teams", json={"name": "Sales", "kind": "sales"}).json()["id"]
        for name in ("ds_in", "view_in"):
            placed = owner.post(f"/api/workspace/teams/{sales}/members", json={"user_id": self.users[name]["id"]})
            self.assertEqual(placed.status_code, 201, placed.text)
        fact = {"subject": "sales", "field": "capacity", "text": "12 calls a day per rep", "expected_revision": 0}
        made = ds_in.post(f"/api/workspace/teams/{sales}/memory/base", json=fact)
        self.assertEqual(made.status_code, 200, made.text)  # a member whose role writes
        read = owner.get(f"/api/workspace/teams/{sales}/memory/base")
        self.assertEqual(read.status_code, 200, read.text)  # the owner, not a member, reads it
        self.assertEqual([e["text"] for e in read.json()["active"]], ["12 calls a day per rep"])
        fixed = owner.post(f"/api/workspace/teams/{sales}/memory/base/{made.json()['entry']['entry_id']}/correct",
                           json={"text": "10 calls a day per rep", "expected_revision": made.json()["revision"]})
        self.assertEqual(fixed.status_code, 200, fixed.text)  # and writes where the owner's role may write
        self.assertEqual(view_in.get(f"/api/workspace/teams/{sales}/memory/base").status_code, 200)  # a viewer member reads
        self.assertEqual(view_in.post(f"/api/workspace/teams/{sales}/memory/base", json={**fact, "field": "x"}).status_code, 403)
        self.assertEqual(ds_out.get(f"/api/workspace/teams/{sales}/memory/base").status_code, 403)  # not in the team
        self.assertEqual(ds_out.post(f"/api/workspace/teams/{sales}/memory/base", json={**fact, "field": "y"}).status_code, 403)
        self.assertEqual(owner.get("/api/workspace/teams/team_nosuch/memory/base").status_code, 404)


if __name__ == "__main__":
    unittest.main()
