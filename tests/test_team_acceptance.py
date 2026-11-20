"""The owner's acceptance checks of the collaborative workspace (W7, 2026-11-21), end to end, one test per check, with the
capture W7 adds: every task, action and report recorded as an event where it is saved, and each task's owning team as a
ledger decision.

1. Business ↔ Data Science round trip: one answer and one linked report; every claimed action has a terminal record, an
   actor and permitted evidence; the handoff history reads like the owner's example.
2. Two delivery branches finished in opposite orders: one dispatch brief holds both results exactly once; while one is out,
   the parent says which result is outstanding.
3. A specialist action denied: the report names the failed attempt and the next owner; no invented score; a retry adds no
   duplicate.
4. The kitchen adds 8 minutes after completion: the revised ETA report links to the superseded one; acceptance keeps its
   snapshot.
5. Restart, another team, another project or workspace, a revoked member.
6. Authority: no agent approves for the decision owner, no role contacts customers by default.
7. The four questions, from the Business thread alone.

The scripted team agents prove the plumbing, never judgment. On files and on PostgreSQL (``tests/run_parallel.py --pg``;
never run directly: that empties ``dclab_test``).
"""

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
from test_teamspace_agents import ANSWER, BRIEF, FEASIBLE, LABEL, _Contract  # noqa: E402

from dclab_core import team_work as tw  # noqa: E402
from dclab_rnd.teamspace import agents, delivery, fixtures, history_lines  # noqa: E402
from dclab_rnd.teamspace.threads import access  # noqa: E402

W7_TASK = ("team.task.created", "team.task.moved", "team.task.delivered")
W7_WORK = ("team.action.started", "team.action.ended", "team.report.written")


class _Acceptance(_Contract):
    def setUp(self):
        super().setUp()
        from dclab_rnd.ledger import open_ledger

        # a real ledger (the review of W7: a patched recorder hid that every routing record was refused)
        self.services.ledger = open_ledger(self.home, self.services.workspace_id)

    @property
    def routed(self):
        """The stored team.task_owner decisions, as (subject, value used, used by, state)."""
        return [(r.subject.key, r.value_used, r.used, dict(r.state)) for r in self.services.ledger.records(point="team.task_owner")]

    def events(self, *kinds):
        return self.services.experience.read(kinds=kinds or None)

    def four_questions(self, message):
        found, why, nxt, changed = message["text"].split("\n")
        self.assertTrue(why.startswith("Why: ") and nxt.startswith("Next: ") and changed.startswith("Changed: "), message["text"])
        return {"found": found, "why": why[len("Why: "):], "next": nxt[len("Next: "):], "changed": changed[len("Changed: "):]}

    # ------------------------------------------------------------------ 1
    def test_1_the_round_trip_has_one_answer_one_linked_report_and_reads_like_the_owners_example(self):
        self.renewal_to_the_end()
        view = self.view("T-renewal")
        [report] = view["reports"]
        stored = {a.action_id: a for a in view["actions"]}
        for line in report.actions.completed + report.actions.failed + report.actions.skipped:
            action = stored[line.action_id]
            self.assertTrue(tw.can_claim(action), line)  # a terminal record behind every claimed action
            self.assertTrue(line.actor)
            if action.actor.id in tw.DIRECTORY:  # permitted: an action of its agent's own directory, bound to a tool or a person
                self.assertIsNotNone(tw.DIRECTORY[action.actor.id].binding(action.action), (action.actor.id, action.action))
        task = view["task"]
        shared = [e for e in report.evidence if not e.private]
        self.assertTrue(shared)
        for e in shared:  # permitted evidence: from what the task was allowed to read
            self.assertIn(e.source, task.allowed_inputs, e)
        answers = self.said("business", "reply")
        self.assertEqual([m["handoff"]["item_id"] for m in answers], [report.id])  # one answer, linking its one report
        self.assertEqual(history_lines(self.work.handoff_history(self.pid, self.prob))[0], f"1. Business → Data Science · {BRIEF}")
        # W7: the work recorded where it was saved, and the owning team as a decision
        kinds = {e.kind for e in self.events()}
        for kind in W7_TASK + W7_WORK:
            self.assertIn(kind, kinds, kind)
        created = [e for e in self.events("team.task.created") if e.payload["task"] == "T-renewal"]
        self.assertEqual([(e.payload["owner_team"], e.payload["requester_team"], e.payload["routed_by"]) for e in created],
                         [("data_science", "business", "requester")])
        written = [e for e in self.events("team.report.written") if e.payload["task"] == "T-renewal"]
        self.assertEqual([(e.payload["report"], e.payload["status"]) for e in written], [(report.id, "complete")])
        ended = {e.payload["action"] for e in self.events("team.action.ended")}
        self.assertTrue({line.action_id for line in report.actions.completed} <= ended)
        self.assertIn((f"{self.pid}:T-renewal", "data_science", "rule", {"from_team": "business", "child": False, "routed_by": "requester"}),
                      self.routed)  # stored, on files and PostgreSQL
        for e in self.events(*W7_TASK, *W7_WORK):  # never a cell or a private note: ids, names, states, counts, an agent's answer
            self.assertLessEqual(set(e.classes.values()), {"meta", "count", "agent_text"}, e.kind)

    # ------------------------------------------------------------------ 2
    def test_2_two_branches_in_opposite_orders_join_once_and_the_parent_says_what_is_out(self):
        for n, (first, second) in enumerate((("eta", "acceptance"), ("acceptance", "eta"))):
            task_id = f"T-dispatch-{n}"
            with self.subTest(first=first):
                self.food(task_id, first, second)
                self.assertEqual(self.view(task_id)["join"].outstanding, [f"{task_id}.{second}"])
                self.assertIn(f"Still waiting on: {task_id}.{second}", agents.preview_report(self.services, self.pid, task_id).answer)
                self.pump()
                [report] = self.view(task_id)["reports"]
                self.assertEqual(sorted(report.child_reports), sorted([f"{task_id}.eta@r1", f"{task_id}.acceptance@r1"]))
                self.assertEqual(len(self.said("business", "reply", task_id)), 1)
                children = {r[0] for r in self.routed if r[3]["child"] is True}
                self.assertTrue({f"{self.pid}:{task_id}.eta", f"{self.pid}:{task_id}.acceptance"} <= children)  # routed by the parent's owner

    # ------------------------------------------------------------------ 3
    def test_3_a_denied_action_names_the_failed_attempt_and_the_next_owner_with_no_invented_score(self):
        with mock.patch.dict(fixtures.MODELS, clear=False):
            del fixtures.MODELS[("acceptance", fixtures.DISPATCH)]
            self.food("T-dispatch", "acceptance", "eta")
            self.pump()
        [denied] = self.view("T-dispatch.acceptance")["reports"]
        self.assertEqual([(l.action, l.status) for l in denied.actions.failed], [("score", "denied")])
        [report] = self.view("T-dispatch")["reports"]
        self.assertEqual(report.next_task.owner, "ml_engineering")
        for text in (report.answer, denied.answer, *report.limits):
            self.assertNotIn("%", text)
        reply = next(d for d in self.work.deliveries(project_id=self.pid, task_id="T-dispatch.acceptance") if d["kind"] == "reply")
        before = len(self.view("T-dispatch")["actions"])
        delivery.default_target(self.services, reply)  # the retry
        self.assertEqual(len(self.view("T-dispatch")["actions"]), before)
        self.assertIn("denied", {e.payload["status"] for e in self.events("team.action.ended") if e.payload["task"] == "T-dispatch.acceptance"})

    # ------------------------------------------------------------------ 4
    def test_4_the_kitchens_eight_minutes_supersede_eta_and_acceptance_keeps_its_snapshot(self):
        self.food("T-dispatch", "eta", "acceptance")
        self.pump()
        acceptance = self.view("T-dispatch.acceptance")["reports"]
        self.assertEqual(agents.correct(self.services, self.pid, "T-dispatch.eta", kitchen_minutes=8)["refused"], [])
        old, new = sorted(self.view("T-dispatch.eta")["reports"], key=lambda r: r.revision)
        self.assertEqual((old.superseded_by, new.supersedes), (new.id, old.id))
        self.pump()
        self.assertEqual(self.view("T-dispatch.acceptance")["reports"], acceptance)
        superseded = [(e.payload["report"], e.payload["by_report"]) for e in self.events("team.report.superseded")]
        self.assertIn((old.id, new.id), superseded)

    # ------------------------------------------------------------------ 5
    def test_5_a_restart_delivers_once_and_other_teams_projects_workspaces_and_revoked_members_are_kept_out(self):
        self.renewal_to_the_end()
        before = self.all_messages()
        for row in self.work.deliveries(project_id=self.pid):  # a restart hands every row over again
            self.assertTrue(delivery.default_target(self.services, row)["agent"]["skipped"], row["kind"])
        self.assertEqual(self.all_messages(), before)
        self.assertEqual(len(self.said("business", "reply")), 1)
        # another team: the history only, never the thread; another project or workspace: nothing
        thread = self.threads.thread(self.biz_thread)
        teams = SimpleNamespace(is_member=lambda team, uid: (team, uid) == ("business", "sara"))
        outsider = SimpleNamespace(user_id="dev", role="data_scientist", via="session")
        self.assertIsNone(access(thread, outsider, teams, auth_on=True))
        self.assertEqual(access(thread, SimpleNamespace(user_id="sara", role="data_scientist", via="session"), teams, auth_on=True), "write")
        self.assertTrue(self.work.handoff_history(self.pid, self.prob))  # the shared history is every reader's of the project
        # another project with its own problem and work: neither reads the other's rows
        from dclab_rnd.memory.scope import create_problem

        pid2 = self.services.projects.create("Other project", goal="Elsewhere")["id"]
        prob2 = create_problem(self.services.projects, pid2, "Elsewhere", workspace_id=self.ws)["problem_id"]
        thread2 = self.threads.open_thread(pid2, prob2, "business", label="Business")[0]["id"]
        agents.open_case(self.services, pid2, prob2, task_id="T-other", question=BRIEF, thread_id=thread2, member="sara",
                         initiator=self.sara, inputs=("sample:renewal_v1",))
        self.assertEqual({d["task_id"] for d in self.work.deliveries(project_id=pid2)}, {"T-other"})
        self.assertNotIn("T-other", {d["task_id"] for d in self.work.deliveries(project_id=self.pid)})
        self.assertEqual([t["id"] for t in self.threads.threads(project_id=pid2)], [thread2])
        self.assertNotIn(thread2, [t["id"] for t in self.threads.threads(project_id=self.pid)])
        self.assertEqual(len(self.work.handoff_history(pid2, prob2)), 1)
        self.assertIsNone(self.work.task(self.pid, "T-other"))
        other = self.home.parent / "other-workspace"
        other.mkdir()
        from dclab_rnd.storage import open_team_work

        self.assertEqual(open_team_work(other).deliveries(project_id=self.pid), [])
        # a revoked member's delivery is refused
        self.open_renewal("T-again")
        self.pump(only={"T-again"})
        delivery.set_membership_checker(lambda uid, wid: None if uid == "sara" else "data_scientist")
        out = agents.person_replies(self.services, self.pid, "T-again", person="sara", text=ANSWER, initiator=self.sara)
        self.assertIn("can no longer change work in this project", out["refused"][0])

    # ------------------------------------------------------------------ 6
    def test_6_no_agent_approves_for_the_decision_owner_and_no_role_contacts_customers(self):
        from dclab_rnd.accounts.roles import ROLES, allows

        for config in tw.DIRECTORY.values():
            self.assertIn("approve", config.forbidden, config.name)
            self.assertFalse([a.action for a in config.action_directory if a.action.startswith("approve")], config.name)
        self.assertFalse([r for r in ROLES if allows(r, "contact_customers")])  # no role by default
        self.renewal_to_the_end()
        approvals = [a for a in self.view("T-renewal")["actions"] if a.action in ("approve_decision", "record_decision")]
        self.assertFalse([a for a in approvals if a.actor.kind != "person"])

    # ------------------------------------------------------------------ 7
    def test_7_a_business_member_answers_the_four_questions_from_the_thread_alone(self):
        self.renewal_to_the_end()
        [answer] = self.said("business", "reply")
        four = self.four_questions(answer)
        self.assertEqual(four["found"], f"{LABEL} {FEASIBLE}")  # what did we decide
        self.assertIn("schema:sample:renewal_v1", four["why"])  # why
        self.assertEqual(four["next"], "closed, nothing is owed.")  # who acts next
        self.assertEqual(four["changed"], "first report (T-renewal@r1).")  # what changed since yesterday


for _name in [n for n in dir(_Contract) if n.startswith("test")]:  # the W5 walkthroughs run in their own file, not again here
    setattr(_Acceptance, _name, None)


class FileTeamAcceptanceTests(_Acceptance, unittest.TestCase):
    pass


class PgTeamAcceptanceTests(_Acceptance, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}


if __name__ == "__main__":
    unittest.main()
