"""W1 of the collaborative workspace (2026-11-20): team tasks, actions, reports and decisions as contracts and rules.

The owner's examples from ``Dev_Plans/Collaborative_Agentic_Workspace_2026-11-20.md``: the renewal round trip, the
food delivery two-branch join, a denied specialist action, the kitchen's +8 minutes, late replies, private notes, and
an agent that may not approve.
"""

import json
import unittest

from dclab_core import team_work as tw
from dclab_core.contracts import CONTRACTS

T0 = "2026-11-20T18:00:00+00:00"
T1 = "2026-11-20T18:01:00+00:00"
T2 = "2026-11-20T18:02:00+00:00"

BUSINESS = tw.Actor(kind="agent", id="business", team="business")
DS = tw.Actor(kind="agent", id="data_science", team="data_science")
ETA = tw.Actor(kind="agent", id="model:eta", team="model:eta")
ACC = tw.Actor(kind="agent", id="model:acceptance", team="model:acceptance")
SARA = tw.Actor(kind="person", id="sara", team="business")


def task(id="T-ds", owner="data_science", parent="T-biz", requester_team="business", status="accepted", **kw):
    return tw.TeamTask(id=id, workspace_id="W1", project_id="P-renewal", problem_id="PB-1", brief_revision=1,
                       parent_task_id=parent, requester=tw.Requester(team=requester_team, member="sara", thread_id="TH-biz"),
                       owner_team=owner, question=kw.pop("question", "Who should Sales call before renewal?"),
                       expected_output="feasibility", done_criteria=["says whether the data can rank accounts"],
                       status=status, created_at=kw.pop("created_at", T0), **kw)


def action(task_id, aid, name, actor, status="succeeded", source=None, **kw):
    binding = None
    for cfg in tw.DIRECTORY.values():
        if cfg.name == actor.id.split(":")[0]:
            binding = cfg.binding(name)
    source = source or (binding if binding and binding.startswith("tool:") else "human:resp-1")
    times = {} if status == "planned" else {"started_at": T0} if status == "running" else {"started_at": T0, "ended_at": T1}
    refs = {}
    if status in tw.DONE_ACTIONS:
        refs["output_ref"] = kw.pop("output_ref", f"out:{aid}")
    elif status in tw.FAILED_ACTIONS:
        refs["failure_ref"] = kw.pop("failure_ref", f"fail:{aid}")
    elif status == "skipped":
        refs["reason"] = kw.pop("reason", "not needed for this question")
    if source.startswith("human:") and status in tw.DONE_ACTIONS:
        refs["responded_by"] = kw.pop("responded_by", SARA)
    return tw.TeamAction(task_id=task_id, action_id=aid, actor=actor, action=name, source=source, input_revision="brief-1",
                         status=status, retry_key=tw.retry_key(task_id, name, "brief-1"), **times, **refs, **kw)


def closed():
    return tw.NextStep(closed=True)


class RenewalRoundTripTest(unittest.TestCase):
    def test_a_clarification_keeps_the_task_open_and_the_reply_joins_once(self):
        parent = task(id="T-biz", owner="business", parent=None, requester_team="business")
        parent = tw.validate_transition(parent, "waiting_for_team", BUSINESS)
        child = task(status="open")
        child = tw.validate_transition(child, "accepted", DS)
        child = tw.validate_transition(child, "waiting_for_person", DS)  # churn = cancellation or no renewal?
        self.assertTrue(tw.is_open(child))
        self.assertEqual(tw.next_owner(child, []), "sara")
        child = tw.validate_transition(child, "accepted", DS)  # Sara: failure to renew by the renewal date
        acts = [action("T-ds", "A1", "inspect_schema", DS), action("T-ds", "A2", "confirm_target", DS, source="human:resp-7"),
                action("T-ds", "A3", "check_leakage", DS)]
        self.assertTrue(all(tw.may_take(tw.DIRECTORY["data_science"], a) for a in acts))
        child = tw.validate_transition(child, "complete", DS)
        report = tw.build_report(child, acts, [], answer="Renewal outcomes and data seven days before renewal can rank accounts.",
                                 evidence=[tw.EvidenceRef(source="crm.renewals", ref="schema:v3", input_revision="brief-1")],
                                 next_task=tw.NextStep(owner="business", unblocks="Business reads the finding"))
        self.assertEqual(report.status, "complete")
        self.assertEqual([a.action for a in report.actions.completed], ["inspect_schema", "confirm_target", "check_leakage"])
        first = tw.accept_reply(parent, report, 1)
        self.assertTrue(first.ok)
        again = tw.accept_reply(parent, report, 1, accepted=[report])
        self.assertEqual((again.ok, again.kind), (False, "duplicate"))
        join = tw.join_status(parent, [report, report], children=[child])
        self.assertEqual((join.status, join.reports), ("complete", ["T-ds@r1"]))

    def test_only_the_owner_moves_a_task_and_only_the_requester_cancels(self):
        child = task(status="open")
        with self.assertRaisesRegex(ValueError, "only the owner"):
            tw.validate_transition(child, "accepted", BUSINESS)
        with self.assertRaisesRegex(ValueError, "requesting team"):
            tw.validate_transition(child, "cancelled", DS)
        self.assertEqual(tw.validate_transition(child, "cancelled", SARA).status, "cancelled")
        done = tw.validate_transition(tw.validate_transition(child, "accepted", DS), "complete", DS)
        with self.assertRaisesRegex(ValueError, "cannot move from complete to accepted"):
            tw.validate_transition(done, "accepted", DS)
        with self.assertRaisesRegex(ValueError, "names its replacement"):
            tw.validate_transition(done, "superseded", DS)
        self.assertEqual(tw.validate_transition(done, "superseded", DS, superseded_by="T-ds2").superseded_by, "T-ds2")


class FoodDeliveryTest(unittest.TestCase):
    def setUp(self):
        self.parent = task(id="T-dispatch", owner="data_science", parent="T-biz-food", question="ETA and first-offer acceptance for order 1042?",
                           budget_eur=1.0)
        self.eta = tw.delegate(self.parent, id="T-eta", owner_team="model:eta", question="Arrival estimate for order 1042 at dispatch.",
                               expected_output="minutes", done_criteria=["an ETA with its fixture"], by=DS, thread_id="TH-ds", created_at=T0)
        self.acc = tw.delegate(self.parent, id="T-acc", owner_team="model:acceptance", question="First-offer acceptance for order 1042.",
                               expected_output="probability", done_criteria=["a probability with its fixture"], by=DS, thread_id="TH-ds",
                               created_at=T0)
        self.eta = tw.validate_transition(tw.validate_transition(self.eta, "accepted", ETA), "complete", ETA)
        self.acc = tw.validate_transition(tw.validate_transition(self.acc, "accepted", ACC), "complete", ACC)
        self.eta_report = tw.build_report(self.eta, [action("T-eta", "E-1", "score", ETA, status="simulated", output_ref="fixture:E1")], [],
                                          answer="Arrival 18:32.", evidence=[tw.EvidenceRef(source="fixture", ref="E1", input_revision="order-1042@dispatch")],
                                          next_task=closed())
        self.acc_report = tw.build_report(self.acc, [action("T-acc", "C-1", "score", ACC, status="simulated", output_ref="fixture:A1")], [],
                                          answer="First-offer acceptance 72%.", evidence=[tw.EvidenceRef(source="fixture", ref="A1", input_revision="order-1042@dispatch")],
                                          next_task=closed())

    def test_children_join_once_in_either_order_and_the_parent_names_the_one_outstanding(self):
        self.assertEqual((self.eta.parent_task_id, self.eta.requester.team), ("T-dispatch", "data_science"))
        with self.assertRaisesRegex(ValueError, "does not delegate"):
            tw.delegate(task(id="T-new", status="open"), id="T-x", owner_team="model:eta", question="q", expected_output="o",
                        done_criteria=["d"], by=DS, thread_id="TH-ds", created_at=T0)
        with self.assertRaisesRegex(ValueError, "only the parent's owner"):
            tw.delegate(self.parent, id="T-x", owner_team="model:eta", question="q", expected_output="o",
                        done_criteria=["d"], by=BUSINESS, thread_id="TH-ds", created_at=T0)
        self.assertTrue(self.eta_report.simulated and self.acc_report.simulated)
        children = [self.eta, self.acc]
        for first, second in ((self.eta_report, self.acc_report), (self.acc_report, self.eta_report)):
            with self.subTest(first=first.task_id):
                accepted = []
                self.assertTrue(tw.accept_reply(self.parent, first, 1, accepted).ok)
                accepted.append(first)
                half = tw.join_status(self.parent, accepted, children=children)
                self.assertEqual(half.status, "partial")
                self.assertEqual(half.outstanding, [second.task_id])
                waiting = tw.validate_transition(self.parent, "waiting_for_team", DS)
                self.assertEqual(tw.next_owner(waiting, accepted, children=children), "model:eta" if second is self.eta_report else "model:acceptance")
                partial = tw.build_report(waiting, [], accepted, children=children, answer="One of two estimates is back.",
                                          next_task=tw.NextStep(owner=second.owner, unblocks="the other estimate"))
                self.assertEqual((partial.status, partial.outstanding), ("partial", [second.task_id]))
                self.assertIn(second.task_id, " ".join(partial.limits))
                self.assertTrue(tw.accept_reply(self.parent, second, 1, accepted).ok)
                accepted.append(second)
                self.assertFalse(tw.accept_reply(self.parent, second, 1, accepted).ok)  # a retried delivery
                done = tw.validate_transition(waiting, "complete", DS)
                brief = tw.build_report(done, [action("T-dispatch", "D-1", "delegate_specialist", DS),
                                               action("T-dispatch", "D-2", "review_evidence", DS)],
                                        accepted + [second], children=children,
                                        answer="At 18:00 the estimated arrival was 18:32; first-offer acceptance 72%.",
                                        next_task=tw.NextStep(owner="business", unblocks="Business reviews the dispatch brief"))
                self.assertEqual(brief.status, "complete")
                self.assertEqual(sorted(brief.child_reports), ["T-acc@r1", "T-eta@r1"])
                self.assertTrue(brief.simulated)
                self.assertEqual(tw.WorkReport.read(brief.model_dump_json()), brief)

    def test_kitchen_plus_eight_supersedes_the_eta_report_and_leaves_acceptance_untouched(self):
        children = [self.eta, self.acc]
        acc_before = self.acc_report.model_dump_json()
        revised = tw.build_report(self.eta, [action("T-eta", "E-2", "score", ETA, status="simulated", output_ref="fixture:E1+8")], [],
                                  answer="Arrival 18:40 (kitchen added 8 minutes; fixed route assumed).", revision=2, brief_revision=2,
                                  evidence=[tw.EvidenceRef(source="fixture", ref="E1", input_revision="order-1042@prep+8",
                                                           assumptions=["the route does not change"])],
                                  next_task=closed())
        old, new = tw.supersede(self.eta_report, revised)
        self.assertEqual((old.status, old.superseded_by, new.supersedes), ("superseded", "T-eta@r2", "T-eta@r1"))
        self.assertTrue(tw.accept_reply(self.parent, new, 2, accepted=[self.eta_report, self.acc_report]).ok)
        join = tw.join_status(self.parent, [old, new, self.acc_report], children=children)
        self.assertEqual((join.status, sorted(join.reports)), ("complete", ["T-acc@r1", "T-eta@r2"]))
        self.assertEqual(self.acc_report.model_dump_json(), acc_before)
        self.assertEqual(self.acc_report.evidence[0].input_revision, "order-1042@dispatch")
        with self.assertRaisesRegex(ValueError, "already superseded"):
            tw.supersede(old, tw.build_report(self.eta, [action("T-eta", "E-3", "score", ETA, status="simulated")], [],
                                              answer="Arrival 18:45.", revision=3, next_task=closed()))
        with self.assertRaisesRegex(ValueError, "later revision"):
            tw.supersede(revised, self.eta_report)

    def test_a_late_reply_to_another_branch_or_an_obsolete_revision_is_refused(self):
        other_parent = task(id="T-other", owner="data_science", parent="T-biz-food")
        verdict = tw.accept_reply(other_parent, self.eta_report, 1)
        self.assertEqual((verdict.ok, verdict.kind), (False, "other_branch"))
        verdict = tw.accept_reply(self.parent, self.acc_report, 2)
        self.assertEqual((verdict.ok, verdict.kind), (False, "obsolete"))
        cancelled = tw.validate_transition(self.parent, "cancelled", tw.Actor(kind="agent", id="business", team="business"))
        self.assertEqual(tw.accept_reply(cancelled, self.eta_report, 1).kind, "closed")
        with self.assertRaisesRegex(ValueError, "another branch"):
            tw.build_report(other_parent, [], [self.eta_report], answer="x", next_task=closed())
        stray = self.eta_report.model_copy(update={"parent_task_id": "T-other"})
        self.assertEqual(tw.join_status(self.parent, [stray], children=[self.eta, self.acc]).returned, [])


class HonestReportTest(unittest.TestCase):
    def test_a_denied_specialist_action_names_the_failed_attempt_and_the_next_owner(self):
        eta = tw.validate_transition(task(id="T-eta", owner="model:eta", parent="T-dispatch", requester_team="data_science"), "blocked", ETA)
        denied = action("T-eta", "E-1", "score", ETA, status="denied", failure_ref="deny:no-scope-model-eta",
                        reason="no permission to score this version")
        report = tw.build_report(eta, [denied], [], answer="No ETA: scoring with version E1 was denied.",
                                 next_task=tw.NextStep(owner="ml_engineering", unblocks="permission to score ETA version E1"))
        self.assertEqual(report.status, "blocked")
        self.assertEqual([(l.action, l.status, l.ref) for l in report.actions.failed], [("score", "denied", "deny:no-scope-model-eta")])
        self.assertEqual((report.actions.completed, report.evidence, report.child_reports), ([], [], []))
        self.assertEqual(tw.next_owner(eta, [report]), "ml_engineering")
        with self.assertRaisesRegex(ValueError, "complete only when its task is"):
            tw.build_report(eta, [denied], [], answer="ETA 18:32.", status="complete", next_task=closed())
        done = tw.TeamTask.model_validate({**eta.model_dump(), "status": "complete"})
        with self.assertRaisesRegex(ValueError, "completed action or a returned child"):
            tw.build_report(done, [denied], [], answer="ETA 18:32.", status="complete", next_task=closed())
        with self.assertRaises(ValueError):
            tw.NextStep()  # an open next step names who and what

    def test_a_planned_or_running_action_is_never_listed_as_completed(self):
        ds = task()
        planned = action("T-ds", "A1", "check_leakage", DS, status="planned")
        running = action("T-ds", "A2", "define_baseline", DS, status="running")
        done = action("T-ds", "A3", "inspect_schema", DS)
        self.assertFalse(tw.can_claim(planned) or tw.can_claim(running))
        self.assertTrue(tw.can_claim(done) and tw.can_claim(action("T-ds", "A4", "score", ETA, status="timed_out")))
        report = tw.build_report(ds, [planned, running, done], [], answer="Schema read; checks under way.",
                                 next_task=tw.NextStep(owner="data_science", unblocks="the leakage check finishes"))
        self.assertEqual(report.status, "partial")
        self.assertEqual([l.action_id for l in report.actions.completed], ["A3"])
        listed = {l.action_id for bucket in (report.actions.completed, report.actions.failed, report.actions.skipped) for l in bucket}
        self.assertNotIn("A1", listed)
        self.assertNotIn("A2", listed)
        self.assertIn("check_leakage (planned)", " ".join(report.limits))
        complete = tw.TeamTask.model_validate({**ds.model_dump(), "status": "complete"})
        with self.assertRaisesRegex(ValueError, "partial"):
            tw.build_report(complete, [planned, done], [], answer="Done.", status="complete", next_task=closed())
        with self.assertRaises(ValueError):
            tw.ReportActions(completed=[tw.ActionLine(action_id="A1", action="check_leakage", actor="data_science", status="planned")])
        with self.assertRaises(ValueError):
            tw.TeamAction(task_id="T-ds", action_id="A9", actor=DS, action="check_leakage", source="tool:leakage_check",
                          input_revision="brief-1", status="planned", output_ref="out:x", retry_key="k")

    def test_private_notes_never_appear_in_a_report(self):
        sales = task(id="T-sales", owner="sales", parent="T-biz")
        seller = tw.Actor(kind="agent", id="sales", team="sales")
        note = "north territory reps are unreliable"
        ok = action("T-sales", "S-1", "check_exclusions", seller)
        report = tw.build_report(sales, [ok], [], answer="Capacity for 50 calls this week.",
                                 evidence=[tw.EvidenceRef(source="sales team", ref="capacity:v1", input_revision="brief-1"),
                                           tw.EvidenceRef(source="team notes", ref="note:7", input_revision="brief-1", private=True)],
                                 limits=["Capacity confirmed for this week only.", note], private_notes=[note],
                                 next_task=tw.NextStep(owner="business", unblocks="Business approves the draft"))
        text = report.model_dump_json()
        self.assertNotIn(note, text)
        self.assertNotIn("note:7", text)
        self.assertEqual([e.ref for e in report.evidence], ["capacity:v1"])
        base = dict(answer="Capacity 50.", private_notes=[note], next_task=tw.NextStep(owner="business", unblocks="approval"))
        leaks = {
            "answer": dict(base, answer=f"Call 50, {note}."),
            "action reason": dict(base, actions=[ok, action("T-sales", "S-2", "draft_contact_list", seller, status="skipped", reason=note)]),
            "action failure": dict(base, actions=[ok, action("T-sales", "S-3", "check_exclusions", seller, status="failed",
                                                             failure_ref=note, reason="x")]),
            "action output": dict(base, actions=[action("T-sales", "S-4", "check_exclusions", seller, output_ref=f"out:{note}")]),
            "next owner": dict(base, next_task=tw.NextStep(owner=note, unblocks="approval")),
            "next unblocks": dict(base, next_task=tw.NextStep(owner="business", unblocks=f"Business decides: {note}")),
            "evidence input revision": dict(base, evidence=[tw.EvidenceRef(source="s", ref="r", input_revision=note)]),
            "evidence assumption": dict(base, evidence=[tw.EvidenceRef(source="s", ref="r", input_revision="1", assumptions=[f"skip: {note}"])]),
            "a limit quoting it": dict(base, limits=[f"Remember: {note}"]),
        }
        for where, kw in leaks.items():
            with self.subTest(field=where):
                acts = kw.pop("actions", [ok])
                with self.assertRaisesRegex(ValueError, "private note"):
                    tw.build_report(sales, acts, [], **kw)
        with self.assertRaisesRegex(ValueError, "private note"):
            tw.WorkReport.model_validate({**report.model_dump(), "evidence": [{"source": "s", "ref": "r", "input_revision": "1",
                                                                             "private": True}]})


class AuthorityTest(unittest.TestCase):
    def test_an_agent_cannot_approve_for_the_decision_owner(self):
        decision = tw.TeamDecision(id="D-1", problem_id="PB-1", decision_owner=SARA, option="Call the 30 riskiest accounts",
                                   evidence_revisions=["T-ds@r1"], constraint_revisions=["capacity@r2"], follow_up_date="2026-12-20")
        self.assertFalse(tw.may_approve(BUSINESS, decision))
        self.assertFalse(tw.may_approve(tw.Actor(kind="team", id="business"), decision))
        self.assertFalse(tw.may_approve(tw.Actor(kind="person", id="omid", team="business"), decision))
        self.assertFalse(tw.may_approve(tw.Actor(kind="agent", id="sara"), decision))
        self.assertTrue(tw.may_approve(SARA, decision))
        with self.assertRaisesRegex(ValueError, "only the decision owner"):
            tw.TeamDecision(**{**decision.model_dump(), "approval": {"who": BUSINESS.model_dump(), "when": T2}})
        with self.assertRaisesRegex(ValueError, "only the decision owner"):  # an agent named like the owner
            tw.TeamDecision(**{**decision.model_dump(), "approval": {"who": {"kind": "agent", "id": "sara"}, "when": T2}})
        with self.assertRaisesRegex(ValueError, "a person"):
            tw.TeamDecision(id="D", problem_id="PB-1", decision_owner=BUSINESS, option="o",
                            approval={"who": BUSINESS.model_dump(), "when": T2})
        approved = tw.TeamDecision(**{**decision.model_dump(), "approval": {"who": SARA.model_dump(), "when": T2},
                                      "outcome": "22 of 30 renewed (simulated)", "simulated_follow_up": True})
        self.assertTrue(approved.simulated_follow_up)
        for cfg in tw.DIRECTORY.values():
            self.assertIn("approve", cfg.forbidden)
            self.assertIn("contact_customers", cfg.forbidden)

    def test_an_action_outside_the_directory_or_from_the_wrong_source_is_refused(self):
        ds = tw.DIRECTORY["data_science"]
        self.assertFalse(tw.may_take(ds, "run_training"))
        self.assertTrue(tw.may_take(tw.DIRECTORY["ml_engineering"], "run_training"))
        self.assertFalse(tw.may_take(ds, action("T-ds", "A1", "check_leakage", DS, source="tool:inspect_schema")))
        self.assertFalse(tw.may_take(ds, action("T-ds", "A1", "inspect_schema", BUSINESS)))
        self.assertFalse(tw.may_take(ds, action("T-ds", "A1", "confirm_target", DS, source="tool:confirm")))
        self.assertTrue(tw.may_take(tw.DIRECTORY["model"], action("T-eta", "E1", "score", ETA)))

    def test_the_actor_kind_and_the_responding_person_are_checked(self):
        ds, biz = tw.DIRECTORY["data_science"], tw.DIRECTORY["business"]
        impostor = tw.Actor(kind="person", id="data_science", team="data_science")
        self.assertFalse(tw.may_take(ds, action("T-ds", "A1", "inspect_schema", impostor)))
        self.assertFalse(tw.may_take(ds, action("T-ds", "A1", "inspect_schema", tw.Actor(kind="team", id="data_science", team="data_science"))))
        recorded = action("T-biz", "B1", "record_decision", BUSINESS, source="human:resp-9", responded_by=SARA)
        self.assertTrue(tw.may_take(biz, recorded))
        with self.assertRaisesRegex(ValueError, "names the person who responded"):
            tw.TeamAction(**{**recorded.model_dump(), "responded_by": None})
        with self.assertRaisesRegex(ValueError, "responded_by is a person"):
            tw.TeamAction(**{**recorded.model_dump(), "responded_by": BUSINESS.model_dump()})
        with self.assertRaisesRegex(ValueError, "responded_by is a person"):
            tw.TeamAction(**{**action("T-ds", "A1", "inspect_schema", DS).model_dump(), "responded_by": SARA.model_dump()})
        self.assertTrue(tw.may_take(biz, action("T-biz", "B2", "record_decision", BUSINESS, status="planned", source="human:pending")))


class ReviewFindingsTest(unittest.TestCase):
    """The independent review of W1 (2026-11-20): each finding reproduced, then refused."""

    def setUp(self):
        fd = FoodDeliveryTest()
        fd.setUp()
        self.fd = fd

    def test_a_parent_never_completes_with_a_child_outstanding(self):
        fd = self.fd
        waiting = tw.validate_transition(fd.parent, "waiting_for_team", DS)
        children = [fd.eta, fd.acc]
        with self.assertRaisesRegex(ValueError, "still out"):
            tw.validate_transition(waiting, "complete", DS, children=children, child_reports=[fd.eta_report])
        done = tw.validate_transition(waiting, "complete", DS, children=children, child_reports=[fd.eta_report, fd.acc_report])
        with self.assertRaisesRegex(ValueError, "pass children"):
            tw.build_report(done, [action("T-dispatch", "D-1", "delegate_specialist", DS)], [fd.eta_report], answer="ETA 18:32",
                            next_task=closed())
        partial_child = tw.WorkReport.model_validate({**fd.eta_report.model_dump(), "status": "partial",
                                                      "next_task": {"owner": "model:eta", "unblocks": "the score"}})
        join = tw.join_status(done, [partial_child, fd.acc_report], children=children)
        self.assertEqual((join.status, join.outstanding), ("partial", ["T-eta"]))
        report = tw.build_report(waiting, [], [partial_child, fd.acc_report], children=children, answer="Acceptance is back.",
                                 next_task=tw.NextStep(owner="model:eta", unblocks="the ETA"))
        self.assertEqual((report.status, report.child_reports), ("partial", ["T-acc@r1"]))

    def test_a_superseded_or_older_revision_reply_is_obsolete(self):
        fd = self.fd
        revised = tw.build_report(fd.eta, [action("T-eta", "E-2", "score", ETA, status="simulated")], [], answer="18:40",
                                  revision=2, next_task=closed())
        verdict = tw.accept_reply(fd.parent, fd.eta_report, 1, accepted=[revised])
        self.assertEqual((verdict.ok, verdict.kind), (False, "obsolete"))
        self.assertEqual(tw.accept_reply(fd.parent, fd.eta_report, 1, accepted=["T-eta@r2"]).kind, "obsolete")
        old, _ = tw.supersede(fd.eta_report, revised)
        self.assertEqual(tw.accept_reply(fd.parent, old, 1).kind, "obsolete")
        self.assertEqual(tw.accept_reply(fd.parent, fd.eta_report, 1, accepted=["T-eta@r1"]).kind, "duplicate")
        self.assertTrue(tw.accept_reply(fd.parent, revised, 1, accepted=["T-eta@r1"]).ok)

    def test_an_action_is_listed_once_and_one_side_effect_is_never_claimed_twice(self):
        ds = tw.TeamTask.model_validate({**task().model_dump(), "status": "complete"})
        a = action("T-ds", "A1", "inspect_schema", DS)
        report = tw.build_report(ds, [a, a], [], answer="Schema read.", next_task=closed())
        self.assertEqual([l.action_id for l in report.actions.completed], ["A1"])
        planned = action("T-ds", "A1", "inspect_schema", DS, status="planned")
        self.assertEqual([l.action_id for l in tw.build_report(ds, [planned, a], [], answer="Schema read.",
                                                                next_task=closed()).actions.completed], ["A1"])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            tw.build_report(ds, [a, action("T-ds", "A1", "inspect_schema", DS, status="failed")], [], answer="x", next_task=closed())
        with self.assertRaisesRegex(ValueError, "retry key"):
            tw.build_report(ds, [a, action("T-ds", "A1b", "inspect_schema", DS)], [], answer="x", next_task=closed())

    def test_supersede_never_leaves_a_link_pointing_two_ways(self):
        fd = self.fd
        r2 = tw.build_report(fd.eta, [action("T-eta", "E-2", "score", ETA, status="simulated")], [], answer="18:40", revision=2, next_task=closed())
        r3 = tw.build_report(fd.eta, [action("T-eta", "E-3", "score", ETA, status="simulated")], [], answer="18:45", revision=3, next_task=closed())
        old2, new3 = tw.supersede(r2, r3)
        with self.assertRaisesRegex(ValueError, "already replaces"):
            tw.supersede(fd.eta_report, new3)
        with self.assertRaisesRegex(ValueError, "itself superseded"):
            tw.supersede(fd.eta_report, old2)

    def test_a_report_is_complete_only_when_its_task_is(self):
        a = action("T-ds", "A1", "inspect_schema", DS)
        self.assertEqual(tw.build_report(task(), [a], [], answer="x", next_task=tw.NextStep(owner="data_science", unblocks="more checks")).status,
                         "partial")
        blocked = tw.validate_transition(task(), "blocked", DS)
        with self.assertRaisesRegex(ValueError, "complete only when its task is"):
            tw.build_report(blocked, [a], [], answer="x", status="complete", next_task=closed())

    def test_a_stored_report_keeps_its_own_rules(self):
        eta_report = self.fd.eta_report
        with self.assertRaisesRegex(ValueError, "marked simulated"):
            tw.WorkReport.model_validate({**eta_report.model_dump(), "simulated": False})
        with self.assertRaisesRegex(ValueError, "completed action or a returned child"):
            tw.WorkReport.model_validate({**eta_report.model_dump(), "actions": {}, "simulated": False})

    def test_only_permitted_evidence_and_tools_are_reported(self):
        t = tw.TeamTask.model_validate({**task().model_dump(), "status": "complete", "allowed_inputs": ["crm.renewals"],
                                        "allowed_tools": ["inspect_schema"]})
        a = action("T-ds", "A1", "inspect_schema", DS)
        ok = tw.build_report(t, [a], [], answer="x", next_task=closed(),
                             evidence=[tw.EvidenceRef(source="crm.renewals", ref="schema:v3", input_revision="brief-1")])
        self.assertEqual(ok.status, "complete")
        with self.assertRaisesRegex(ValueError, "source the task does not allow"):
            tw.build_report(t, [a], [], answer="x", next_task=closed(),
                            evidence=[tw.EvidenceRef(source="billing.invoices", ref="inv:1", input_revision="brief-1")])
        with self.assertRaisesRegex(ValueError, "tool the task does not allow"):
            tw.build_report(t, [a, action("T-ds", "A2", "check_leakage", DS)], [], answer="x", next_task=closed())

    def test_an_older_brief_child_is_refused_and_children_share_the_parent_budget(self):
        parent = tw.TeamTask.model_validate({**task(id="T-p2").model_dump(), "brief_revision": 2, "status": "complete"})
        child = tw.TeamTask.model_validate({**task(id="T-c", parent="T-p2", owner="model:eta", requester_team="data_science").model_dump(),
                                            "status": "complete"})
        old = tw.build_report(child, [action("T-c", "X", "score", ETA, status="simulated")], [], answer="old", next_task=closed())
        with self.assertRaisesRegex(ValueError, "older brief revision"):
            tw.join_status(parent, [old], children=[child])
        with self.assertRaisesRegex(ValueError, "older brief revision"):
            tw.build_report(parent, [], [old], children=[child], answer="x", next_task=closed())
        p = task(budget_eur=1.0)
        kw = dict(owner_team="model:eta", question="q", expected_output="o", done_criteria=["d"], by=DS, thread_id="TH", created_at=T0)
        c1 = tw.delegate(p, id="C1", budget_eur=0.6, **kw)
        with self.assertRaisesRegex(ValueError, "budgets together"):
            tw.delegate(p, id="C2", budget_eur=0.6, siblings=[c1], **kw)
        c2 = tw.delegate(p, id="C2", budget_eur=0.4, siblings=[c1], **kw)
        self.assertTrue(tw.budget_fits(p, [c1, c2]))
        self.assertFalse(tw.budget_fits(p, [c1, c2, tw.delegate(p, id="C3", budget_eur=0.1, **kw)]))
        with self.assertRaisesRegex(ValueError, "already exists"):
            tw.delegate(p, id="C1", siblings=[c1], **kw)


class DirectoryTest(unittest.TestCase):
    def test_every_directory_entry_binds_its_actions_to_a_tool_or_a_person(self):
        names = {"business", "data_science", "ml_engineering", "sales", "marketing", "project", "problem", "model", "home",
                 "lead", "advisor"}
        self.assertEqual(set(tw.DIRECTORY), names)
        for name, cfg in tw.DIRECTORY.items():
            with self.subTest(agent=name):
                self.assertEqual(cfg.name, name)
                self.assertTrue(cfg.action_directory)
                for entry in cfg.action_directory:
                    self.assertRegex(entry.bound_to, r"^(tool|human):[a-z_]+$")
                self.assertTrue(cfg.required_report and cfg.memory_scope)
                self.assertEqual(tw.AgentConfig.read(cfg.model_dump_json()), cfg)
        kinds = {n: c.kind for n, c in tw.DIRECTORY.items()}
        self.assertTrue(all(kinds[n] == "team_agent" for n in ("business", "data_science", "ml_engineering", "sales", "marketing")))
        with self.assertRaisesRegex(ValueError, "approve"):
            tw.AgentConfig(name="rogue", kind="specialist", identity="x", responsibility="x", required_report=["x"],
                           memory_scope=["team"], action_directory=[{"action": "go", "bound_to": "tool:go", "description": "x"}])
        with self.assertRaises(ValueError):
            tw.DirectoryAction(action="go", bound_to="go", description="unbound")


class ContractTest(unittest.TestCase):
    def test_the_records_are_registered_versioned_and_round_trip(self):
        self.assertEqual(tw.CONTRACT, "dclab_core.team_work.v1")
        for name in ("team_task", "team_action", "work_report", "team_decision", "agent_config", "team_work"):
            self.assertIn(name, CONTRACTS)
        child = task()
        work = tw.TeamWork(problem_id="PB-1", tasks=[child], actions=[action("T-ds", "A1", "inspect_schema", DS)])
        self.assertEqual(tw.TeamWork.read(json.loads(work.model_dump_json())), work)
        with self.assertRaisesRegex(ValueError, "problem"):
            tw.TeamWork(problem_id="PB-2", tasks=[child])
        with self.assertRaisesRegex(ValueError, "zone"):
            task(created_at="2026-11-20T18:00:00")


if __name__ == "__main__":
    unittest.main()
