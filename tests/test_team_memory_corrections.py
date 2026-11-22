"""W6 of the collaborative workspace (2026-11-20): a team's own memory, and corrections across teams.

The owner's plan (``Dev_Plans/Collaborative_Agentic_Workspace_2026-11-20.md``, "What the agents remember"): Sales'
working notes and draft exceptions stay with Sales (another team and the shared summary never see them) until Sales shares
a fact on purpose, with its source. The correction test: Sales confirms capacity for 50 calls, then corrects it to 30;
the task, its report and the decision that rested on 50 become stale, Business is alerted in its own thread, the decision
waits for its owner's review, and work that did not rest on it is untouched, in this session and the next. Old memory
items (v1, v2) read unchanged. On files and on PostgreSQL (``tests/run_parallel.py --pg``; never run directly: that empties
``dclab_test``).
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
        token = context.set_services(SimpleNamespace(experience=self.log))
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
        calls = ts.record_decision(self.home, sara, self.decision("D-calls", [cite], ["T-capacity@r1"]), thread_id=self.biz_thread)
        ts.record_decision(self.home, sara, self.decision("D-offer", ["offer:v1"], ["T-offer@r1"]), thread_id=self.biz_thread)
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
            ts.record_decision(self.home, viewer, self.decision("D-x", [], []))
        other = create_problem(self.projects, self.pid, "Another", workspace_id=self.ws)["problem_id"]
        with self.assertRaises(AccessDenied):
            ts.record_decision(self.home, self.scope("sara"), tw.TeamDecision(
                id="D-y", problem_id=other, decision_owner=tw.Actor(kind="person", id="dana"), option="x"))
        kept = ts.record_decision(self.home, self.scope("sara"), self.decision("D-z", [], []))
        self.assertEqual((kept["kind"], kept["stale"], kept["cites"]), ("team_decision", False, []))


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
        ts.record_decision(self.home, sara, self.decision("D-plan", [ts.fact_citation(offer)], ["T-plan@r1"]), thread_id=self.biz_thread)
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

    def test_recording_a_decision_again_keeps_its_approval_and_never_changes_its_owner(self):
        """Finding 3."""
        sara, mia = self.scope("sara"), self.scope("mia")
        ts.record_decision(self.home, sara, self.decision("D-1", [], []), thread_id=self.biz_thread)
        approved = self.decision("D-1", [], []).model_copy(update={"approval": tw.Approval(
            who=tw.Actor(kind="person", id="dana", team="business"), when=T1)})
        ts.record_decision(self.home, sara, tw.TeamDecision.model_validate(approved.model_dump()), thread_id=self.biz_thread)
        fresh = {d["decision"].id: d["decision"] for d in ts.decisions(self.home, self.scope("dana"))}
        self.assertEqual(fresh["D-1"].approval.who.id, "dana")
        other = tw.TeamDecision(id="D-1", problem_id=self.prob, decision_owner=tw.Actor(kind="person", id="mia"), option="call nobody")
        with self.assertRaisesRegex(ValueError, "dana's decision"):
            ts.record_decision(self.home, mia, other)
        self.assertEqual(ts.decisions(self.home, sara)[0]["decision"].option, "D-1: call the shortlisted accounts")

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
