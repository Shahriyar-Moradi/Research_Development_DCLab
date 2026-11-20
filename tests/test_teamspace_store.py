"""W4 of the collaborative workspace (2026-11-20): tasks, actions, reports and the delivery outbox, kept on files and on
PostgreSQL alike.

The owner's examples (``Dev_Plans/Collaborative_Agentic_Workspace_2026-11-20.md``): the renewal round trip with its
clarification, the food delivery's two branches finished in opposite orders and the kitchen's +8 minutes, a denied
action, a retried call adding nothing, a task saved with its delivery or not at all, and another project or workspace
reading nothing. Run on PostgreSQL through ``tests/run_parallel.py --pg`` (never directly: that empties ``dclab_test``).
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402

from dclab_core import team_work as tw  # noqa: E402
from dclab_rnd import context, experience  # noqa: E402
from dclab_rnd.memory.scope import AccessDenied, create_problem  # noqa: E402
from dclab_rnd.storage import TeamWork, open_stores, open_team_work  # noqa: E402
from dclab_rnd.teamspace import FileTeamWork, PgTeamWork, history_lines  # noqa: E402
from dclab_rnd.teamspace import store as ts  # noqa: E402

T0, T1, T2 = "2026-11-20T18:00:00+00:00", "2026-11-20T18:01:00+00:00", "2026-11-20T18:02:00+00:00"
BUSINESS = tw.Actor(kind="agent", id="business", team="business")
DS = tw.Actor(kind="agent", id="data_science", team="data_science")
ETA = tw.Actor(kind="agent", id="model:eta", team="model:eta")
ACC = tw.Actor(kind="agent", id="model:acceptance", team="model:acceptance")
SARA = tw.Actor(kind="person", id="sara", team="business")
QUESTION = "Can we rank accounts seven days before renewal?"
CLARIFY = "Does churn mean cancellation or failure to renew by the contractual date?"
ANSWER = "Failure to renew by the contractual renewal date."


RACER = r"""
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from dclab_core import team_work as tw
from dclab_rnd.teamspace import open_team_work
home, pid, prob, ws, start, n = Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), sys.argv[7]
work = open_team_work(home)
while time.time() < start:
    time.sleep(0.001)
for i in range(12):
    work.create_task(tw.TeamTask(id=f"T-{n}-{i}", workspace_id=ws, project_id=pid, problem_id=prob, brief_revision=1,
                                 requester=tw.Requester(team="business", member="sara", thread_id="TH-biz"), owner_team="data_science",
                                 question=f"Question {n}-{i}?", expected_output="x", done_criteria=["d"], created_at="2026-11-20T18:00:00+00:00"),
                     initiator={"user_id": "sara", "role": "data_scientist", "workspace_id": ws})
"""


class _Contract:
    """Every test here runs once on files and once on PostgreSQL."""

    env: dict = {}
    store_type: type = object

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"DCLAB_AUTH": "", **self.env})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.home = Path(tempfile.mkdtemp()) / "workspace"
        self.home.mkdir()
        self.projects, self.drafts, _ = open_stores(self.home)
        self.work = open_team_work(self.home)
        self.ws = self.work.workspace  # memory.scope.workspace_of's name for it: the database's id, or "local" on files
        self.log = experience.open_log(self.home, self.ws if isinstance(self.work, PgTeamWork) else None)
        token = context.set_services(type("Env", (), {"experience": self.log})())
        self.addCleanup(context.reset_services, token)
        self.pid = self.projects.create("Renewal", goal="Who should Sales call")["id"]
        self.sara = {"user_id": "sara", "role": "data_scientist", "workspace_id": self.ws}
        self.prob = self.problem("Renewal")

    def problem(self, name, project=None):
        return create_problem(self.projects, project or self.pid, name, workspace_id=self.ws)["problem_id"]

    # ------------------------------------------------------------------ helpers
    def task(self, id="T-ds", owner="data_science", requester="business", question=QUESTION, project=None, budget=0.0, problem=None):
        return tw.TeamTask(id=id, workspace_id=self.ws, project_id=project or self.pid, problem_id=problem or self.prob, brief_revision=1,
                           requester=tw.Requester(team=requester, member="sara", thread_id="TH-biz"), owner_team=owner, question=question,
                           expected_output="feasibility", done_criteria=["says whether the data can rank accounts"], budget_eur=budget,
                           created_at=T0)

    def act(self, task_id, aid, name, actor, status="succeeded", source=None, input_revision="brief-1", **kw):
        cfg = tw.DIRECTORY[actor.id.split(":")[0]]
        binding = cfg.binding(name)
        source = source or (binding if binding.startswith("tool:") else "human:resp-1")
        refs = {}
        if status in ("planned",):
            times = {}
        elif status == "running":
            times = {"started_at": T0}
        else:
            times = {"started_at": T0, "ended_at": T1}
        if status in tw.DONE_ACTIONS:
            refs["output_ref"] = kw.pop("output_ref", f"out:{aid}")
        elif status in tw.FAILED_ACTIONS:
            refs["failure_ref"] = kw.pop("failure_ref", f"fail:{aid}")
        if source.startswith("human:") and status in tw.DONE_ACTIONS:
            refs["responded_by"] = SARA
        return tw.TeamAction(task_id=task_id, action_id=aid, actor=actor, action=name, source=source, input_revision=input_revision,
                             status=status, retry_key=tw.retry_key(task_id, name, input_revision), **times, **refs, **kw)

    def done(self, task_id, aid, name, actor, **kw):
        """An action recorded running, then finished: the two records the store keeps."""
        a = self.act(task_id, aid, name, actor, status="running", **{k: v for k, v in kw.items() if k == "input_revision"})
        self.work.record_action(self.pid, a)
        status = kw.get("status", "succeeded")
        extra = {"output_ref": kw.get("output_ref", f"out:{aid}")} if status in tw.DONE_ACTIONS else {"failure_ref": f"fail:{aid}"}
        if a.source.startswith("human:") and status in tw.DONE_ACTIONS:
            extra["responded_by"] = SARA
        return self.work.finish_action(self.pid, task_id, aid, status, ended_at=T1, **extra).item

    def handoffs(self):
        return [e.payload.get("what") for e in self.log.read(kinds=["team.handoff"])]

    # ------------------------------------------------------------------ tests
    def test_the_store_is_chosen_as_the_other_stores_are(self):
        self.assertIsInstance(self.work, self.store_type)
        self.assertIsInstance(self.work, TeamWork)

    def test_the_renewal_round_trip_keeps_the_clarification_open_and_reads_like_the_owners_example(self):
        made = self.work.create_task(self.task(), initiator=self.sara)
        self.assertEqual((made.delivery["kind"], made.delivery["status"], made.delivery["to_team"]), ("task", "pending", "data_science"))
        self.assertEqual(made.delivery["initiator"], self.sara)
        again = self.work.create_task(self.task(), initiator=self.sara)  # a retried request adds nothing
        self.assertTrue(again.duplicate)
        self.assertEqual(again.delivery["id"], made.delivery["id"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.work.create_task(self.task(question="Another question?"))

        self.work.move(self.pid, "T-ds", "accepted", DS)
        asked = self.work.move(self.pid, "T-ds", "waiting_for_person", DS, note=CLARIFY)
        self.assertEqual((asked.delivery["kind"], asked.delivery["thread_id"], asked.delivery["to_team"]), ("clarification", "TH-biz", "business"))
        self.assertTrue(tw.is_open(asked.item))
        self.assertEqual([t.id for t in self.work.open_tasks("data_science")], ["T-ds"])  # a clarification keeps it open
        self.assertEqual(self.work.task_view(self.pid, "T-ds")["next_owner"], "sara")
        with self.assertRaisesRegex(ValueError, "only the owner"):
            self.work.move(self.pid, "T-ds", "accepted", BUSINESS)
        answered = self.work.move(self.pid, "T-ds", "accepted", DS, note=ANSWER)
        self.assertEqual((answered.delivery["kind"], answered.delivery["to_team"]), ("answer", "data_science"))

        self.done("T-ds", "A1", "inspect_schema", DS)
        self.done("T-ds", "A2", "confirm_target", DS, source="human:resp-7")
        self.done("T-ds", "A3", "check_leakage", DS)
        self.work.move(self.pid, "T-ds", "complete", DS)
        reply = self.work.store_report(self.pid, "T-ds", current_brief_revision=1, answer="Yes: renewal outcomes and data seven days before renewal can rank accounts.",
                                       evidence=[tw.EvidenceRef(source="crm.renewals", ref="schema:v3", input_revision="brief-1")],
                                       next_task=tw.NextStep(owner="business", unblocks="Business reads the finding"))
        self.assertEqual((reply.item.id, reply.item.status), ("T-ds@r1", "complete"))
        self.assertEqual([a.action for a in reply.item.actions.completed], ["inspect_schema", "confirm_target", "check_leakage"])
        self.assertEqual((reply.delivery["kind"], reply.delivery["thread_id"], reply.delivery["item_id"]), ("reply", "TH-biz", "T-ds@r1"))
        self.assertIsNone(reply.verdict)  # a top-level task answers its requester's thread

        retried = self.work.store_report(self.pid, "T-ds", current_brief_revision=1, answer="A different wording from a retried step.",
                                         next_task=tw.NextStep(owner="business", unblocks="Business reads the finding"))
        self.assertTrue(retried.duplicate)
        self.assertEqual((retried.item.answer, retried.delivery["id"]), (reply.item.answer, reply.delivery["id"]))  # deduplicated by (task, revision)

        view = self.work.task_view(self.pid, "T-ds")
        self.assertEqual(([r.id for r in view["reports"]], view["next_owner"], view["task"].status), (["T-ds@r1"], "business", "complete"))
        self.assertTrue(all(tw.can_claim(a) for a in view["actions"]))
        self.assertEqual(self.work.open_tasks("data_science"), [])
        history = self.work.handoff_history(self.pid, self.prob)
        self.assertEqual(history_lines(history)[0],  # saved, not delivered yet: the history says so
                         "1. Business → Data Science · Can we rank accounts seven days before renewal? · not delivered: pending")
        for d in self.work.deliveries(project_id=self.pid):
            self.work.delivery_done(d["id"])
        history = self.work.handoff_history(self.pid, self.prob)
        self.assertEqual(history_lines(history)[:3], [
            "1. Business → Data Science · Can we rank accounts seven days before renewal?",
            "2. Data Science → Business · Does churn mean cancellation or failure to renew by the contractual date?",
            "3. Business → Data Science · Failure to renew by the contractual renewal date."])
        self.assertEqual(history[3]["from_team"], "Data Science")
        self.assertTrue(history[3]["text"].startswith("Yes: renewal outcomes"))
        self.assertEqual(set(history[0]), {"n", "from_team", "to_team", "text", "state"})  # no thread id: other teams may read it
        self.assertEqual(self.work.counts(), {"tasks": 1, "actions": 3, "reports": 1, "deliveries": 4})
        whats = self.handoffs()
        self.assertEqual((whats.count("task.created"), whats.count("action.finished"), whats.count("report.stored")), (1, 3, 1))
        self.assertEqual(whats.count("task.moved"), 4)  # accepted, waiting for Sara, accepted, complete

    def test_food_delivery_branches_join_once_in_either_order(self):
        for first, second in (("T-eta", "T-acc"), ("T-acc", "T-eta")):
            with self.subTest(first=first):
                suffix = first[-3:]
                parent_id = f"T-dispatch-{suffix}"
                ids = {"T-eta": f"T-eta-{suffix}", "T-acc": f"T-acc-{suffix}"}
                self.work.create_task(self.task(id=parent_id, question="ETA and first-offer acceptance for order 1042?", budget=1.0,
                                                problem=self.problem(f"Food {suffix}")), initiator=self.sara)
                self.work.move(self.pid, parent_id, "accepted", DS)
                eta = self.work.delegate(self.pid, parent_id, id=ids["T-eta"], owner_team="model:eta", question="Arrival estimate for order 1042.",
                                         expected_output="minutes", done_criteria=["an ETA with its fixture"], by=DS, thread_id="TH-ds",
                                         created_at=T0, budget_eur=0.4)
                self.assertEqual((eta.item.parent_task_id, eta.delivery["kind"], eta.delivery["initiator"]), (parent_id, "task", self.sara))
                self.work.delegate(self.pid, parent_id, id=ids["T-acc"], owner_team="model:acceptance", question="First-offer acceptance for order 1042.",
                                   expected_output="probability", done_criteria=["a probability with its fixture"], by=DS, thread_id="TH-ds",
                                   created_at=T0, budget_eur=0.4)
                with self.assertRaisesRegex(ValueError, "fit inside"):  # the siblings come from the store
                    self.work.delegate(self.pid, parent_id, id=f"T-x-{suffix}", owner_team="model:eta", question="Again?", expected_output="o",
                                       done_criteria=["d"], by=DS, thread_id="TH-ds", created_at=T0, budget_eur=0.5)
                self.work.move(self.pid, parent_id, "waiting_for_team", DS)
                actors = {"T-eta": (ETA, "fixture:E1", "Arrival 18:32."), "T-acc": (ACC, "fixture:A1", "First-offer acceptance 72%.")}

                def finish(branch):
                    tid, (actor, ref, answer) = ids[branch], actors[branch]
                    self.work.move(self.pid, tid, "accepted", actor)
                    self.done(tid, f"S-{tid}", "score", actor, status="simulated", output_ref=ref)
                    self.work.move(self.pid, tid, "complete", actor)
                    return self.work.store_report(self.pid, tid, current_brief_revision=1, answer=answer, next_task=tw.NextStep(closed=True),
                                                  evidence=[tw.EvidenceRef(source="fixture", ref=ref[8:], input_revision="order-1042@dispatch")])

                one = finish(first)
                self.assertEqual((one.verdict.kind, one.delivery["to_team"], one.delivery["thread_id"]), ("accepted", "data_science", "TH-ds"))
                view = self.work.task_view(self.pid, parent_id)
                self.assertEqual((view["join"].status, view["join"].outstanding), ("partial", [ids[second]]))
                with self.assertRaisesRegex(ValueError, "still out"):  # the children come from the store
                    self.work.move(self.pid, parent_id, "complete", DS)
                partial = self.work.store_report(self.pid, parent_id, current_brief_revision=1, answer="One estimate so far.",
                                                 next_task=tw.NextStep(owner=actors[second][0].id, unblocks="its estimate returns"))
                self.assertEqual((partial.item.status, partial.item.outstanding), ("partial", [ids[second]]))
                self.assertIn(f"Still waiting on: {ids[second]}.", partial.item.limits)
                self.assertEqual(self.work.task_view(self.pid, parent_id)["next_owner"], actors[second][0].id)

                two = finish(second)
                self.assertTrue(two.verdict.ok)
                dup = self.work.store_report(self.pid, ids[second], current_brief_revision=1, answer="again", next_task=tw.NextStep(closed=True))
                self.assertTrue(dup.duplicate)  # a retried child reply adds nothing
                self.work.move(self.pid, parent_id, "complete", DS)
                brief = self.work.supersede_report(self.pid, parent_id, current_brief_revision=1, answer="At 18:00 the estimated arrival was 18:32. "
                                                   "The first-offer acceptance estimate was 72%.", next_task=tw.NextStep(owner="business",
                                                   unblocks="Business reviews the dispatch brief"))
                self.assertEqual(brief.item.status, "complete")
                self.assertEqual(sorted(brief.item.child_reports), sorted([f"{ids['T-eta']}@r1", f"{ids['T-acc']}@r1"]))  # each exactly once
                self.assertTrue(brief.item.simulated)
                self.assertEqual((brief.replaced.status, brief.replaced.superseded_by, brief.item.supersedes),
                                 ("superseded", f"{parent_id}@r2", f"{parent_id}@r1"))
                stored = {r.id: r for r in self.work.task_view(self.pid, parent_id)["reports"]}
                self.assertEqual((stored[f"{parent_id}@r1"].superseded_by, stored[f"{parent_id}@r2"].supersedes),
                                 (f"{parent_id}@r2", f"{parent_id}@r1"))  # both ways, as stored

        # the kitchen adds 8 minutes: the ETA report is corrected, linked back; the acceptance report keeps its snapshot
        eta_id, acc_id = "T-eta-eta", "T-acc-eta"
        acc_before = self.work.task_view(self.pid, acc_id)["reports"]
        self.done(eta_id, "S-eta-2", "score", ETA, status="simulated", output_ref="fixture:E2", input_revision="order-1042@prep+8")
        fixed = self.work.supersede_report(self.pid, eta_id, current_brief_revision=1, answer="Arrival 18:40.", next_task=tw.NextStep(closed=True),
                                           evidence=[tw.EvidenceRef(source="fixture", ref="E2", input_revision="order-1042@prep+8")])
        self.assertEqual((fixed.item.id, fixed.item.supersedes, fixed.verdict.kind), (f"{eta_id}@r2", f"{eta_id}@r1", "accepted"))
        self.assertEqual(fixed.delivery["kind"], "reply")
        self.assertEqual(self.work.task_view(self.pid, acc_id)["reports"], acc_before)  # not rerun, not rewritten
        parent = self.work.task_view(self.pid, "T-dispatch-eta")
        self.assertEqual(sorted(r.id for r in tw._latest(parent["child_reports"]).values()), sorted([f"{eta_id}@r2", f"{acc_id}@r1"]))

    def test_a_denied_action_is_reported_with_the_next_owner_and_endings_are_recorded_once(self):
        self.work.create_task(self.task(), initiator=self.sara)
        self.work.move(self.pid, "T-ds", "accepted", DS)
        self.done("T-ds", "A1", "inspect_schema", DS)
        running = self.act("T-ds", "A2", "check_leakage", DS, status="running")
        self.work.record_action(self.pid, running)
        self.assertTrue(self.work.record_action(self.pid, running).duplicate)  # the same record again is the same record
        denied = self.work.finish_action(self.pid, "T-ds", "A2", "denied", ended_at=T1, failure_ref="permission:crm.renewals")
        self.assertEqual(denied.item.status, "denied")
        again = self.work.finish_action(self.pid, "T-ds", "A2", "denied", ended_at=T1, failure_ref="permission:crm.renewals")
        self.assertTrue(again.duplicate)
        with self.assertRaisesRegex(ValueError, "second terminal record"):
            self.work.finish_action(self.pid, "T-ds", "A2", "succeeded", ended_at=T2, output_ref="out:A2")
        with self.assertRaisesRegex(ValueError, "already recorded"):
            self.work.record_action(self.pid, self.act("T-ds", "A2", "check_leakage", DS, status="planned"))
        # the same side effect once: a second success under one retry key is refused (a failed attempt then a success is a retry)
        with self.assertRaisesRegex(ValueError, "retry key"):
            self.done("T-ds", "A1-again", "inspect_schema", DS)
        retried = self.done("T-ds", "A2-retry", "check_leakage", DS, status="failed")
        self.assertEqual(retried.retry_key, denied.item.retry_key)

        self.work.move(self.pid, "T-ds", "blocked", DS)
        report = self.work.store_report(self.pid, "T-ds", current_brief_revision=1, answer="The leakage check was denied: no feasibility answer yet.",
                                        next_task=tw.NextStep(owner="business", unblocks="Business grants read access to crm.renewals"))
        self.assertEqual(report.item.status, "blocked")
        self.assertEqual([(a.action_id, a.status, a.ref) for a in report.item.actions.failed],
                         [("A2", "denied", "permission:crm.renewals"), ("A2-retry", "failed", "fail:A2-retry")])
        self.assertEqual([a.action_id for a in report.item.actions.completed], ["A1"])  # no invented success
        self.assertEqual(self.work.task_view(self.pid, "T-ds")["next_owner"], "business")

    def test_a_task_and_its_delivery_are_saved_together_or_not_at_all(self):
        target = ts._PgTx if isinstance(self.work, PgTeamWork) else ts._FileTx
        with mock.patch.object(target, "insert_delivery", side_effect=RuntimeError("the outbox is away")):
            with self.assertRaises(RuntimeError):
                self.work.create_task(self.task(), initiator=self.sara)
        self.assertIsNone(self.work.task(self.pid, "T-ds"))
        self.assertEqual(self.work.deliveries(), [])
        made = self.work.create_task(self.task(), initiator=self.sara)
        self.assertEqual([d["id"] for d in self.work.deliveries(status="pending")], [made.delivery["id"]])
        with self.assertRaisesRegex(ValueError, "delegate"):  # a child task is made only through delegate
            self.work.create_task(tw.TeamTask.model_validate({**self.task(id="T-child").model_dump(), "parent_task_id": "T-ds"}))

    def test_another_project_or_workspace_reads_nothing(self):
        self.work.create_task(self.task(), initiator=self.sara)
        other = self.projects.create("Another", goal="x")["id"]
        self.assertIsNone(self.work.task(other, "T-ds"))
        self.assertIsNone(self.work.task_view(other, "T-ds"))
        self.assertEqual(self.work.open_tasks("data_science", project_id=other), [])
        self.assertEqual(self.work.handoff_history(other, self.prob), [])
        self.assertEqual(self.work.deliveries(project_id=other), [])
        with self.assertRaises(KeyError):
            self.work.move(other, "T-ds", "accepted", DS)
        with self.assertRaises(KeyError):
            self.work.record_action(other, self.act("T-ds", "A1", "inspect_schema", DS, status="planned"))
        with self.assertRaisesRegex(ValueError, "belongs to workspace"):
            self.work.create_task(tw.TeamTask.model_validate({**self.task(id="T-w").model_dump(), "workspace_id": "w-elsewhere"}))
        elsewhere = Path(tempfile.mkdtemp()) / "workspace-b"
        elsewhere.mkdir()
        theirs = open_team_work(elsewhere)
        self.assertIsNone(theirs.task(self.pid, "T-ds"))
        self.assertEqual((theirs.deliveries(), theirs.open_tasks("data_science"), theirs.handoff_history(self.pid, self.prob)), ([], [], []))
        self.assertEqual(theirs.counts(), {"tasks": 0, "actions": 0, "reports": 0, "deliveries": 0})

    def test_every_change_is_saved_with_its_delivery_or_not_at_all(self):
        target = ts._PgTx if isinstance(self.work, PgTeamWork) else ts._FileTx
        broken = mock.patch.object(target, "insert_delivery", side_effect=RuntimeError("the outbox is away"))
        self.work.create_task(self.task(budget=1.0), initiator=self.sara)
        self.work.move(self.pid, "T-ds", "accepted", DS)
        with broken, self.assertRaises(RuntimeError):
            self.work.move(self.pid, "T-ds", "waiting_for_person", DS, note=CLARIFY)
        self.assertEqual(self.work.task(self.pid, "T-ds").status, "accepted")  # the move went with its clarification
        with broken, self.assertRaises(RuntimeError):
            self.work.delegate(self.pid, "T-ds", id="T-child", owner_team="model:eta", question="ETA?", expected_output="minutes",
                               done_criteria=["an ETA"], by=DS, thread_id="TH-ds", created_at=T0, budget_eur=0.5)
        self.assertIsNone(self.work.task(self.pid, "T-child"))
        self.assertEqual(self.work.task_view(self.pid, "T-ds")["children"], [])
        self.done("T-ds", "A1", "inspect_schema", DS)
        self.work.move(self.pid, "T-ds", "complete", DS)
        with broken, self.assertRaises(RuntimeError):
            self.work.store_report(self.pid, "T-ds", current_brief_revision=1, answer="Yes.", next_task=tw.NextStep(closed=True))
        self.assertEqual(self.work.task_view(self.pid, "T-ds")["reports"], [])  # no report without its reply
        self.assertEqual([d["kind"] for d in self.work.deliveries(project_id=self.pid)], ["task"])
        self.assertEqual(self.work.store_report(self.pid, "T-ds", current_brief_revision=1, answer="Yes.",
                                                next_task=tw.NextStep(closed=True)).delivery["kind"], "reply")

    def test_a_retried_task_with_a_new_time_is_the_same_task_and_its_problem_must_be_the_projects(self):
        first = self.work.create_task(self.task(), initiator=self.sara)
        later = tw.TeamTask.model_validate({**self.task().model_dump(), "created_at": T2})
        again = self.work.create_task(later, initiator=self.sara)
        self.assertTrue(again.duplicate)
        self.assertEqual((again.item.created_at, again.delivery["id"]), (first.item.created_at, first.delivery["id"]))
        with self.assertRaises(AccessDenied):  # a problem this project does not have
            self.work.create_task(self.task(id="T-elsewhere", problem="prob_not_here"), initiator=self.sara)
        other = self.projects.create("Another", goal="x")["id"]
        theirs = self.problem("Theirs", project=other)
        with self.assertRaises(AccessDenied):  # another project's problem
            self.work.create_task(self.task(id="T-cross", problem=theirs), initiator=self.sara)
        self.assertEqual(len(self.work.deliveries()), 1)

    def test_a_withdrawn_clarification_is_not_an_answer(self):
        self.work.create_task(self.task(), initiator=self.sara)
        self.work.move(self.pid, "T-ds", "accepted", DS)
        self.work.move(self.pid, "T-ds", "waiting_for_person", DS, note=CLARIFY)
        gone = self.work.move(self.pid, "T-ds", "cancelled", SARA, note="We no longer need this.")
        self.assertIsNone(gone.delivery)
        self.assertEqual([d["kind"] for d in self.work.deliveries(project_id=self.pid)], ["task", "clarification"])
        self.assertIn("task.moved", self.handoffs())

    def test_words_are_kept_whole_and_only_the_history_cuts_them(self):
        long_question = "Which accounts should Sales call, and why? " * 80  # about 3,400 characters
        self.work.create_task(self.task(question=long_question[:1990]), initiator=self.sara)
        self.work.move(self.pid, "T-ds", "accepted", DS)
        note = "Churn means " + "no renewal by the contractual date, " * 100
        asked = self.work.move(self.pid, "T-ds", "waiting_for_person", DS, note=note)
        self.assertEqual(self.work.delivery(asked.delivery["id"])["text"], note)  # whole in the row
        shown = self.work.handoff_history(self.pid, self.prob)[1]["text"]
        self.assertLess(len(shown), len(note))
        self.assertTrue(shown.endswith("…") and note.startswith(shown[:-1]))

    def test_a_private_note_never_reaches_the_history(self):
        secret = "Internal: the renewal table has known gaps for Q3"
        self.work.create_task(self.task(), initiator=self.sara)
        self.work.move(self.pid, "T-ds", "accepted", DS)
        self.done("T-ds", "A1", "inspect_schema", DS)
        self.work.move(self.pid, "T-ds", "complete", DS)
        with self.assertRaisesRegex(ValueError, "private note"):
            self.work.store_report(self.pid, "T-ds", current_brief_revision=1, answer=f"Yes. {secret}", private_notes=[secret],
                                   next_task=tw.NextStep(closed=True))
        self.work.store_report(self.pid, "T-ds", current_brief_revision=1, answer="Yes: the data can rank accounts.", private_notes=[secret],
                               limits=[secret], evidence=[tw.EvidenceRef(source="team-notes", ref="note-1", input_revision="b1", private=True)],
                               next_task=tw.NextStep(closed=True))
        shown = " ".join(history_lines(self.work.handoff_history(self.pid, self.prob)))
        self.assertNotIn("known gaps", shown)
        self.assertNotIn("known gaps", json.dumps(self.work.deliveries(project_id=self.pid)))
        self.assertNotIn("known gaps", json.dumps([r.model_dump() for r in self.work.task_view(self.pid, "T-ds")["reports"]]))

    def test_concurrent_writers_lose_nothing_and_one_budget_is_spent_once(self):
        start = time.time() + 2.0  # every process imports first, then all write at once
        procs = [subprocess.Popen([sys.executable, "-c", RACER, str(ROOT), str(self.home), self.pid, self.prob, self.ws, str(start), str(n)],
                                  cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=dict(os.environ))
                 for n in range(3)]
        for proc in procs:
            out, err = proc.communicate(timeout=180)
            self.assertEqual(proc.returncode, 0, err[-2000:])
        self.assertEqual(self.work.counts()["tasks"], 3 * 12)
        self.assertEqual(len(self.work.deliveries(status="pending", project_id=self.pid)), 3 * 12)
        self.assertEqual(len({d["seq"] for d in self.work.deliveries(project_id=self.pid)}), 3 * 12)  # one order, no two rows on one place

        self.work.create_task(self.task(id="T-parent", budget=1.0), initiator=self.sara)
        self.work.move(self.pid, "T-parent", "accepted", DS)
        made, refused = [], []
        barrier = threading.Barrier(4)

        def ask(k):
            barrier.wait()
            try:
                self.work.delegate(self.pid, "T-parent", id=f"T-c{k}", owner_team="model:eta", question=f"Estimate {k}?",
                                   expected_output="minutes", done_criteria=["an ETA"], by=DS, thread_id="TH-ds", created_at=T0,
                                   budget_eur=0.6)
                made.append(k)
            except ValueError:
                refused.append(k)
        threads = [threading.Thread(target=ask, args=(k,)) for k in range(4)]
        [t.start() for t in threads]
        [t.join(60) for t in threads]
        self.assertEqual((len(made), len(refused)), (1, 3))  # the siblings are read under the lock: the budget is spent once
        self.assertEqual(len(self.work.task_view(self.pid, "T-parent")["children"]), 1)

class FileTeamWorkTests(_Contract, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}
    store_type = FileTeamWork

    def test_the_work_is_one_file_per_project_in_the_workspace_folder(self):
        self.work.create_task(self.task(), initiator=self.sara)
        other = self.projects.create("Another", goal="x")["id"]
        self.work.create_task(self.task(project=other, problem=self.problem("Theirs", project=other)), initiator=self.sara)
        self.assertEqual(sorted(p.name for p in (self.home / "teamspace").iterdir() if not p.name.startswith(".")), sorted([self.pid, other]))
        self.assertTrue((self.home / "teamspace" / self.pid / "work.json").is_file())
        self.assertEqual(self.work.counts()["tasks"], 2)  # the same id in two projects is two tasks


class PgTeamWorkTests(_Contract, unittest.TestCase):
    store_type = PgTeamWork

    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}

    def test_no_file_is_written_and_the_database_refuses_a_second_success(self):
        import sqlalchemy as sa

        from dclab_rnd.storage import db
        from dclab_rnd.storage.models import team_action

        self.work.create_task(self.task(), initiator=self.sara)
        self.work.move(self.pid, "T-ds", "accepted", DS)
        done = self.done("T-ds", "A1", "inspect_schema", DS)
        self.assertFalse((self.home / "teamspace").exists())
        twin = tw.TeamAction.model_validate({**done.model_dump(), "action_id": "A1-twin"})
        with self.assertRaises(sa.exc.IntegrityError):  # the partial unique index, even for a writer that skips the store's check
            with db.engine(self.url).begin() as c:
                c.execute(sa.insert(team_action).values(workspace_id=self.ws, task_id="T-ds", action_id="A1-twin", project_id=self.pid,
                                                        retry_key=twin.retry_key, status="succeeded", doc=twin.model_dump(mode="json")))

    def test_a_task_changed_underneath_is_refused(self):
        import sqlalchemy as sa

        from dclab_rnd.storage import db
        from dclab_rnd.storage.models import team_task

        self.work.create_task(self.task(), initiator=self.sara)
        real = ts._PgTx.task

        def stale(tx, task_id):  # another writer that took no lock moves the version between the read and the write
            row = real(tx, task_id)
            with db.engine(self.url).begin() as c:
                c.execute(sa.update(team_task).where(team_task.c.id == task_id).values(version=team_task.c.version + 1))
            return row

        with mock.patch.object(ts._PgTx, "task", stale):
            with self.assertRaises(ts.Conflict):
                self.work.move(self.pid, "T-ds", "accepted", DS)
        self.assertEqual(self.work.task(self.pid, "T-ds").status, "open")


if __name__ == "__main__":
    unittest.main()
