"""Home's lead agent and the team agents, connected through the workspace's Lead (2026-11-21), end to end.

1. Home → Lead → Data Science → one joined report back in the Home chat, every claimed action behind a terminal record.
2. A team's question reaches the person in Home, and their answer goes back to that team once.
3. A restart delivers once: a row handed over again adds no step and no second message.
4. Another person, team or project sees nothing it should not: the Home thread is its person's; the owner reads it.
5. No agent approves or contacts customers; the Lead hands work to team agents only; Home opens cases for the Lead only.
6. The hand-over is refused without a project, without the person's words, or for a kind of help the Lead does not take.

The Lead and Home are scripted rules: these tests prove the plumbing, never judgment. On files and on PostgreSQL
(``tests/run_parallel.py --pg``; never run directly: that empties ``dclab_test``).
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402
from test_teamspace_agents import ANSWER, BRIEF, _Contract  # noqa: E402

from dclab_core import team_work as tw  # noqa: E402
from dclab_rnd.teamspace import agents, delivery, lead, threads  # noqa: E402
from dclab_rnd.teamspace.tools import TeamGuard, TeamTurn  # noqa: E402

SAMPLE = "sample:renewal_v1"


class _Lead(_Contract):
    def setUp(self):
        super().setUp()
        from dclab_rnd.ledger import open_ledger

        self.services.ledger = open_ledger(self.home, self.services.workspace_id)
        self.draft = self.services.drafts.create("Renewal on Home")["id"]

    def ask(self, question=BRIEF, track="analyze", inputs=(SAMPLE,)):
        out = lead.ask(self.services, self.pid, self.prob, question=question, track=track, person="sara", draft_id=self.draft,
                       initiator=self.sara, inputs=inputs)
        self.assertEqual(out["refused"], [])
        return out

    def chat(self):
        return [m for m in self.services.drafts.get(self.draft)["messages"] if m.get("team")]

    def to_the_end(self, **ask):
        out = self.ask(**ask)
        self.pump()
        [question] = self.chat()
        self.assertEqual(question["team"]["kind"], "clarification")
        waiting = lead.waiting(self.services, self.pid, self.draft)
        self.assertEqual(waiting.id, out["task_id"])
        said = lead.answer(self.services, self.pid, out["task_id"], person="sara", text=ANSWER, initiator=self.sara)
        self.assertEqual(said["refused"], [])
        self.pump()
        return out

    # ------------------------------------------------------------------ 1
    def test_1_home_to_lead_to_data_science_and_one_joined_report_back_in_home(self):
        out = self.to_the_end()
        task = self.work.task(self.pid, out["task_id"])
        self.assertEqual((task.owner_team, task.requester.team, task.requester.member), ("lead", "home", "sara"))
        self.assertEqual(task.status, "partial")  # the brief asks who Sales should call: Sales waits for its capacity (0.168.0)
        view = self.view(task.id)
        self.assertEqual([c.owner_team for c in view["children"]], ["data_science", "sales"])
        [report] = view["reports"]
        self.assertTrue(report.answer.startswith("Data Science: "), report.answer)
        self.assertIn("\nSales: The calls can be planned", report.answer)
        self.assertEqual(report.next_task.owner, "sales")
        stored = {a.action_id: a for a in view["actions"]}
        for line in report.actions.completed + report.actions.failed + report.actions.skipped:
            self.assertTrue(tw.can_claim(stored[line.action_id]), line)
        child = self.work.task(self.pid, f"{task.id}.data_science")
        self.assertNotIn("help:analyze", child.allowed_inputs)  # the kind of help is the Lead's to read, never a team's input
        self.assertIn(SAMPLE, child.allowed_inputs)
        for e in report.evidence:
            self.assertIn(e.source, child.allowed_inputs)
        cards = self.chat()
        self.assertEqual([c["team"]["kind"] for c in cards], ["clarification", "reply"])
        *found, why, nxt, changed = cards[-1]["text"].split("\n")  # one line a team, then why, next and what changed
        self.assertEqual([line.split(":")[0].split(" ")[-1] if line.startswith("Simulated") else line.split(":")[0] for line in found][1:], ["Sales"])
        self.assertIn("Data Science:", found[0])
        self.assertIn("Sales", nxt)  # who acts next: the Sales team, with its capacity
        self.assertTrue(why.startswith("Why: ") and nxt.startswith("Next: ") and changed.startswith("Changed: "))
        self.assertNotIn("@r", changed)  # the report's revision id stays on the task page
        routed = {(r.subject.key.split(":", 1)[1], r.value_used, dict(r.state)["routed_by"])
                  for r in self.services.ledger.records(point="team.task_owner")}
        self.assertIn((task.id, "lead", "requester"), routed)
        self.assertIn((child.id, "data_science", "parent_owner"), routed)

    # ------------------------------------------------------------------ more teams (0.168.0)
    RANK = "Can our data rank the accounts by churn risk seven days before each renewal?"

    def _never_forbidden(self, view):
        for child in view["children"]:
            acts = {a.action for a in self.view(child.id)["actions"]}
            self.assertFalse(acts & {"approve", "train", "deploy", "contact_customers", "run_training", "register_version"}, (child.owner_team, acts))

    def test_a_model_goes_to_data_science_and_ml_engineering_and_one_report_names_both(self):
        out = self.to_the_end(question=self.RANK, track="model")
        view = self.view(out["task_id"])
        self.assertEqual([c.owner_team for c in view["children"]], ["data_science", "ml_engineering"])
        self.assertEqual(view["task"].status, "complete")
        [report] = view["reports"]
        self.assertTrue(report.answer.startswith("Data Science: "), report.answer)
        self.assertIn("\nML Engineering: The experiment can be set up: score each account seven days before each renewal", report.answer)
        self.assertIn("Nothing was trained", report.answer)
        ml = self.view(f"{out['task_id']}.ml_engineering")
        self.assertEqual([(a.action, a.status) for a in ml["actions"]], [("validate_contract", "simulated")])
        self.assertTrue(ml["reports"][0].simulated)
        self.assertIn("renewed_by_date", ml["actions"][0].output_ref)  # the outcome named as excluded, from the sample's columns
        self._never_forbidden(view)
        self.assertEqual([c["team"]["kind"] for c in self.chat()], ["clarification", "reply"])  # one question, one report

    def test_a_model_about_calls_goes_to_three_teams_and_sales_blocks_on_its_capacity(self):
        out = self.to_the_end(track="model")  # the brief: who should Sales call before renewal
        view = self.view(out["task_id"])
        self.assertEqual([c.owner_team for c in view["children"]], ["data_science", "sales", "ml_engineering"])  # ML Engineering after Data Science answered
        self.assertEqual(view["task"].status, "partial")
        sales = self.view(f"{out['task_id']}.sales")
        self.assertEqual(sales["task"].status, "blocked")
        [said] = sales["reports"]
        self.assertIn("do-not-contact", said.answer)
        self.assertIn("nobody is contacted", said.answer)
        self.assertEqual((said.next_task.owner, said.next_task.unblocks), ("sales", agents.CAPACITY))
        [report] = view["reports"]
        self.assertEqual(sorted(r.split("@")[0] for r in report.child_reports), sorted(f"{out['task_id']}.{t}" for t in ("data_science", "ml_engineering", "sales")))
        self.assertEqual([line.split(":")[0] for line in report.answer.split("\n")], ["Data Science", "ML Engineering", "Sales"])
        self.assertLess(report.answer.index("Data Science: "), report.answer.index("ML Engineering: "))
        self.assertLess(report.answer.index("ML Engineering: "), report.answer.index("Sales: "))  # in the route's order
        self.assertEqual(report.next_task.owner, "sales")
        self._never_forbidden(view)

    def test_ml_engineering_is_not_asked_while_data_science_is_blocked_and_sales_blocks_on_its_own_reason(self):
        """Review F4: ML Engineering comes after Data Science has answered; review F5: Sales's blocker is its own."""
        out = self.ask(track="model", inputs=("sample:renewal_no_dates",))  # the brief (who should Sales call), no renewal dates
        self.pump()
        lead.answer(self.services, self.pid, out["task_id"], person="sara", text=ANSWER, initiator=self.sara)
        self.pump()
        view = self.view(out["task_id"])
        self.assertEqual(sorted(c.owner_team for c in view["children"]), ["data_science", "sales"])  # no ML Engineering
        self.assertEqual(view["task"].status, "partial")
        [said] = self.view(f"{out['task_id']}.sales")["reports"]
        self.assertIn("no do-not-contact list", said.answer)
        self.assertNotIn("renewal date", said.answer)

    def test_ml_engineering_reads_the_outcome_the_person_confirmed_to_data_science(self):
        """Review F4: it stated renewed_by_date as the outcome even when the person said churn meant a cancellation."""
        out = self.ask(question=self.RANK, track="model")
        self.pump()
        lead.answer(self.services, self.pid, out["task_id"], person="sara", text="Cancellation of the subscription.", initiator=self.sara)
        self.pump()
        ml = self.view(f"{out['task_id']}.ml_engineering")
        self.assertIn("Data Science found:", ml["task"].question)
        [said] = ml["reports"]
        self.assertIn('the outcome is "Cancellation of the subscription", as confirmed with Data Science', said.answer)

    def test_a_crash_between_ml_engineerings_two_writes_resumes_and_the_lead_finishes(self):
        """Review F1: the action written running, the process gone before it finished: the redelivery runs it again."""
        work = self.services.teamwork
        real, failed = work.finish_action, []

        def once(project_id, task_id, action_id, status, **kw):
            if task_id.endswith(".ml_engineering") and not failed:
                failed.append(action_id)
                raise RuntimeError("the process stopped here")
            return real(project_id, task_id, action_id, status, **kw)

        work.finish_action = once
        try:
            out = self.ask(question=self.RANK, track="model")
            self.pump()
            lead.answer(self.services, self.pid, out["task_id"], person="sara", text=ANSWER, initiator=self.sara)
            try:
                self.pump()
            except AssertionError:
                pass  # the delivery that crashed is retried
        finally:
            work.finish_action = real
        self.assertTrue(failed)
        self.pump()
        ml = self.view(f"{out['task_id']}.ml_engineering")
        self.assertEqual([(a.action, a.status) for a in ml["actions"]], [("validate_contract", "simulated")])
        self.assertEqual(self.work.task(self.pid, out["task_id"]).status, "complete")

    def test_asking_again_after_a_partial_report_is_a_new_request(self):
        """Review F3: Sales blocks on its capacity, so the request ends partial; asking again said "the teams are working"."""
        first = self.to_the_end()
        self.assertEqual(self.work.task(self.pid, first["task_id"]).status, "partial")
        again = self.ask()
        self.assertNotEqual(again["task_id"], first["task_id"])
        self.assertFalse(again["already_open"])

    def test_the_lead_never_sends_a_report_or_preparation_to_sales(self):
        self.assertEqual(tw.lead_route("report", "who should Sales call"), ("data_science",))
        self.assertEqual(tw.lead_route("analyze", "who should Sales call before renewal?"), ("data_science", "sales"))
        self.assertEqual(tw.lead_route("model", self.RANK), ("data_science", "ml_engineering"))

    # ------------------------------------------------------------------ 2
    def test_2_a_teams_question_reaches_the_person_in_home_and_their_answer_goes_back_once(self):
        out = self.to_the_end()
        child = self.view(f"{out['task_id']}.data_science")
        answers = [d for d in child["deliveries"] if d["kind"] == "answer"]
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0]["text"], ANSWER)
        humans = [a for a in child["actions"] if a.source.startswith("human:") and a.status in tw.DONE_ACTIONS]
        self.assertTrue(humans)
        self.assertEqual({a.responded_by.id for a in humans}, {"sara"})  # every recorded answer is the person's, never an agent's
        again = lead.answer(self.services, self.pid, out["task_id"], person="sara", text=ANSWER, initiator=self.sara)
        self.assertEqual(again["steps"], [])  # nothing waits for an answer now: nothing is recorded twice

    def test_a_teams_question_during_homes_turn_follows_homes_answer(self):
        """The live run of 2026-11-21: the worker delivered the team's question while Home was still writing its answer, so
        the question came first. A card that comes during a turn waits; the turn's end posts it after the answer, once."""
        import threading

        from dclab_rnd.draft.chat import turn

        drafts, draft = self.services.drafts, self.draft
        holding, done = threading.Event(), threading.Event()

        class Home:
            store = drafts

            @turn
            def answer(self, draft_id):  # Home's turn: it holds the draft until its answer is saved
                holding.set()
                done.wait(30)
                drafts.update(draft_id, lambda d: d["messages"].append({"id": "home-answer", "role": "agent", "text": "Handed to the Lead."}))

        t = threading.Thread(target=Home().answer, args=(draft,))
        t.start()
        self.assertTrue(holding.wait(30))
        self.ask()
        self.pump()  # the Lead and Data Science run now, while Home's turn holds the draft
        self.assertEqual(self.chat(), [], "the team's question waits for Home's answer")
        done.set()
        t.join(30)
        ids = [m.get("id") for m in drafts.get(draft)["messages"]]
        [card] = self.chat()
        self.assertEqual(card["team"]["kind"], "clarification")
        self.assertLess(ids.index("home-answer"), ids.index(card["id"]))
        self.assertFalse(drafts.get(draft).get(lead.HELD))
        self.assertEqual(lead.flush_held(drafts, draft), 0)  # posted once
        self.assertIsNotNone(lead.waiting(self.services, self.pid, draft))  # and the person may answer it, as before

    # ------------------------------------------------------------------ 3
    def test_3_a_row_handed_over_again_adds_no_step_and_no_second_message(self):
        self.to_the_end()
        rows = [r for r in self.work.deliveries(project_id=self.pid) if r["to_team"] == "home"]
        self.assertTrue(rows)
        before = len(self.services.drafts.get(self.draft)["messages"])
        for row in rows:
            receipt = agents.deliver(self.services, row)
            self.assertEqual(receipt["steps"], [], row["kind"])
        self.assertEqual(len(self.services.drafts.get(self.draft)["messages"]), before)

    # ------------------------------------------------------------------ 4
    def test_4_the_home_thread_is_its_persons_and_the_owner_reads_it(self):
        out = self.ask()
        thread = self.threads.thread(out["thread_id"])
        self.assertEqual(threads.draft_of(thread), self.draft)
        teams = SimpleNamespace(is_member=lambda team, uid: False)
        who = lambda uid, role: SimpleNamespace(user_id=uid, role=role)  # noqa: E731
        self.assertEqual(threads.access(thread, who("sara", "data_scientist"), teams, auth_on=True), "write")
        self.assertIsNone(threads.access(thread, who("omar", "data_scientist"), teams, auth_on=True))
        self.assertEqual(threads.access(thread, who("owner", "owner"), teams, auth_on=True), "read")
        other = self.services.projects.create("Another project", goal="x")["id"]
        self.assertIsNone(lead.waiting(self.services, other, self.draft))
        self.assertEqual(self.threads.threads(project_id=other), [])

    # ------------------------------------------------------------------ 5
    def test_5_no_agent_approves_or_contacts_customers_and_each_hands_work_only_where_it_may(self):
        for name in ("home", "lead"):
            config = tw.DIRECTORY[name]
            self.assertIn("approve", config.forbidden)
            self.assertIn("contact_customers", config.forbidden)
        self.assertEqual(tw.DIRECTORY["home"].kind, "entry")
        self.assertEqual(tw.DIRECTORY["lead"].kind, "coordinator")
        self.assertIsNone(tw.opening_refusal(tw.DIRECTORY["home"], "lead"))
        self.assertIsNotNone(tw.opening_refusal(tw.DIRECTORY["home"], "data_science"))  # Home always goes through the Lead
        self.assertIsNotNone(tw.opening_refusal(tw.DIRECTORY["lead"], "data_science"))  # the Lead opens no case of its own
        self.assertIsNotNone(tw.delegation_refusal(tw.DIRECTORY["lead"], "model:eta"))  # a team asks its own specialists
        self.assertIsNone(tw.delegation_refusal(tw.DIRECTORY["lead"], "data_science"))
        from dclab_rnd.agents import default_registry

        registry = default_registry()
        turn = TeamTurn(self.services, self.work, self.pid, "home", task_id="X", initiator=self.sara, person="sara")
        tool = registry.get("team_delegate") if hasattr(registry, "get") else registry.tools["team_delegate"]
        why = TeamGuard.refusal(tool, turn, {"action": "open_case", "new_task_id": "X", "owner_team": "data_science"})
        self.assertIn("only to a coordinator", why)
        out = self.ask()
        self.pump()
        lead_turn = TeamTurn(self.services, self.work, self.pid, "lead", task_id=out["task_id"], initiator=self.sara)
        why = TeamGuard.refusal(registry.tools["team_act"] if hasattr(registry, "tools") else tool, lead_turn,
                                {"task_id": out["task_id"], "action": "approve"})
        self.assertIn("never approves", why)

    # ------------------------------------------------------------------ 6
    def test_6_the_hand_over_is_refused_without_a_project_words_or_a_kind_of_help_the_lead_takes(self):
        with self.assertRaises(lead.TeamsRefused):
            lead.ask(self.services, "", self.prob, question=BRIEF, track="analyze", person="sara", draft_id=self.draft)
        with self.assertRaises(lead.TeamsRefused):
            lead.ask(self.services, self.pid, self.prob, question="  ", track="analyze", person="sara", draft_id=self.draft)
        with self.assertRaises(lead.TeamsRefused):
            lead.ask(self.services, self.pid, self.prob, question=BRIEF, track="advice", person="sara", draft_id=self.draft)
        self.assertEqual(self.work.deliveries(project_id=self.pid), [])

    def test_a_request_with_nothing_to_read_is_blocked_by_the_lead_before_any_team_is_asked(self):
        out = self.ask(inputs=())
        self.pump()
        view = self.view(out["task_id"])
        self.assertEqual(view["task"].status, "blocked")
        self.assertEqual(view["children"], [])
        [report] = view["reports"]
        self.assertEqual(report.next_task.owner, "home")
        [card] = self.chat()
        self.assertIn("Blocked before any team was asked", card["text"])

    def test_the_same_words_from_the_same_draft_are_one_task(self):
        first = self.ask()
        second = self.ask()
        self.assertEqual(first["task_id"], second["task_id"])
        self.assertFalse(second["opened"])
        self.assertEqual(len([r for r in self.work.deliveries(project_id=self.pid) if r["to_team"] == "lead"]), 1)


    # ------------------------------------------------------------------ Home's lead agent and the built chat
    def case(self, words, project=True):
        from dclab_core import case as C
        from dclab_rnd.agents import case_agent as A

        state = A.CaseState(case=C.new("lead-case", list(words)))
        state.draft_id, state.workspace_home = self.draft, self.home
        if project:
            state.project_id, state.problem_id = self.pid, self.prob
        return state

    def test_the_lead_agents_tool_hands_the_persons_words_to_the_lead(self):
        from dclab_rnd.agents import case_agent as A

        self.services.projects.update(self.pid, team_inputs=[SAMPLE])
        out = A._ask_teams(self.case([BRIEF]), "analyze")
        self.assertEqual(out.get("sent_to"), "lead", out)
        task = self.work.task(self.pid, out["task_id"])
        self.assertEqual(task.question, BRIEF)  # the person's own words, never the agent's rewording
        self.assertIn(SAMPLE, task.allowed_inputs)
        self.assertIn("case_ask_teams", A.TOOL_NAMES)
        self.assertEqual(A.MOVES["ask_teams"], "case.ask_teams")

    def test_the_lead_agents_tool_is_refused_without_a_project_words_or_with_an_approval(self):
        from dclab_rnd.agents import case_agent as A

        self.assertIn("project", A._ask_teams(self.case([BRIEF], project=False), "analyze")["error"])
        self.assertIn("approve", A._ask_teams(self.case(["Please approve the churn campaign for these accounts"]), "analyze")["error"])
        self.assertIn("customers", A._ask_teams(self.case(["Email our customers the renewal offer this week"]), "report")["error"])
        self.assertIn("advice", A._ask_teams(self.case([BRIEF]), "advice")["error"])
        self.assertEqual(self.work.deliveries(project_id=self.pid), [])

    def test_the_built_chat_hands_a_request_to_the_lead_and_the_answer_route_reaches_the_team(self):
        from dclab_rnd.draft import orchestrator as orch

        self.services.projects.update(self.pid, team_inputs=[SAMPLE], default_problem_id=self.prob)
        self.services.drafts.update(self.draft, lambda d: d["messages"].append({"id": "u1", "role": "user", "kind": "text", "text": BRIEF}))
        turn = orch.Turn(self.services, self.pid, self.draft, agent=None, text=BRIEF)
        out = orch.ask_teams(turn, track="analyze", request="can the data rank accounts by churn risk?")
        self.assertEqual(out.get("sent_to"), "lead", out)
        self.assertEqual(self.work.task(self.pid, out["task_id"]).question, "can the data rank accounts by churn risk?")
        self.assertIn("ask_teams", orch.HOME_MOVES)
        made_up = orch.ask_teams(turn, track="analyze", request="words the person never wrote")
        self.assertEqual(self.work.task(self.pid, made_up["task_id"]).question, BRIEF)  # not their words: their own message instead


    def test_a_message_in_the_home_thread_answers_the_leads_open_question(self):
        out = self.ask()
        self.pump()
        thread = self.threads.thread(out["thread_id"])
        self.assertIsNone(agents.from_thread(self.services, thread, person="omar", text=ANSWER, initiator=None))  # not the case's person
        said = agents.from_thread(self.services, thread, person="sara", text=ANSWER, initiator=self.sara)
        self.assertEqual(said["refused"], [])
        self.pump()
        self.assertEqual(self.work.task(self.pid, out["task_id"]).status, "partial")  # Data Science answered; Sales waits for its capacity


    # ------------------------------------------------------------------ the review of 2026-11-21
    def test_r1_a_second_message_before_the_worker_ran_is_not_a_second_answer(self):
        out = self.ask()
        self.pump()
        lead.answer(self.services, self.pid, out["task_id"], person="sara", text=ANSWER, initiator=self.sara)
        self.assertIsNone(lead.waiting(self.services, self.pid, self.draft, person="sara"))  # answered: the next message is a message
        later = lead.answer(self.services, self.pid, out["task_id"], person="sara", text="Actually show me the leakage report",
                            initiator=self.sara)
        self.assertEqual(later["steps"], [])
        self.pump()
        child = self.view(f"{out['task_id']}.data_science")
        self.assertEqual([d["text"] for d in child["deliveries"] if d["kind"] == "answer"], [ANSWER])

    def test_r2_only_the_person_asked_answers_and_another_persons_draft_message_is_not_taken(self):
        self.ask()
        self.pump()
        self.assertIsNotNone(lead.waiting(self.services, self.pid, self.draft, person="sara"))
        self.assertIsNone(lead.waiting(self.services, self.pid, self.draft, person="omar"))

    def test_r4_two_people_asking_from_one_draft_each_have_their_own_thread(self):
        first = self.ask()
        omar = {"user_id": "omar", "role": "data_scientist", "workspace_id": self.ws}
        second = lead.ask(self.services, self.pid, self.prob, question="Can the data rank accounts by renewal value?", track="analyze",
                          person="omar", draft_id=self.draft, initiator=omar, inputs=(SAMPLE,))
        self.assertNotEqual(first["thread_id"], second["thread_id"])
        self.assertEqual(threads.draft_of(self.threads.thread(second["thread_id"])), self.draft)

    def test_r5_a_blocked_request_asked_again_is_a_new_task_and_an_open_one_is_not_asked_twice(self):
        blocked = self.ask(inputs=())
        self.pump()
        self.assertEqual(self.work.task(self.pid, blocked["task_id"]).status, "blocked")
        again = self.ask()
        self.assertNotEqual(again["task_id"], blocked["task_id"])
        self.assertTrue(again["opened"])
        self.assertFalse(again["already_open"])
        twice = self.ask()
        self.assertEqual(twice["task_id"], again["task_id"])
        self.assertTrue(twice["already_open"])

    def test_r6_the_tools_say_plainly_when_the_teams_cannot_read_the_projects_data(self):
        from dclab_rnd.agents import case_agent as A
        from dclab_rnd.draft import orchestrator as orch

        self.assertEqual(A._ask_teams(self.case([BRIEF]), "analyze")["error"], lead.NO_DATA_YET)
        self.services.projects.update(self.pid, default_problem_id=self.prob)
        turn = orch.Turn(self.services, self.pid, self.draft, agent=None, text=BRIEF)
        self.assertEqual(orch.ask_teams(turn, track="analyze", request=BRIEF)["error"], lead.NO_DATA_YET)
        self.assertEqual(self.work.deliveries(project_id=self.pid), [])

    def test_r7_only_orders_to_approve_or_contact_customers_are_refused(self):
        for words in ("Predict which credit card applications are approved", "Which loans will be approved next quarter?",
                      "Rank accounts by churn risk so we know whom to email customers about", "What drives approval rates?"):
            self.assertIsNone(lead.refusal(project_id="p", problem_id="b", question=words, track="analyze"), words)
        for words in ("Please approve the churn campaign for these accounts", "Approve the offer.", "Email our customers the renewal offer",
                      "Rank the accounts, then call the customers on the list", "I want the teams to approve it"):
            self.assertIsNotNone(lead.refusal(project_id="p", problem_id="b", question=words, track="analyze"), words)

    def test_r9_a_team_agent_never_opens_or_delegates_a_task_for_the_lead_or_home(self):
        for agent in ("business", "data_science"):
            config = tw.DIRECTORY[agent]
            for owner in ("lead", "home"):
                self.assertIsNotNone(tw.opening_refusal(config, owner), (agent, owner))
                self.assertIsNotNone(tw.delegation_refusal(config, owner), (agent, owner))
        self.assertIsNone(tw.opening_refusal(tw.DIRECTORY["business"], "data_science"))
        self.assertIsNone(tw.delegation_refusal(tw.DIRECTORY["data_science"], "model:eta"))


    def test_the_built_chats_rules_hand_a_request_naming_the_teams_to_the_lead(self):
        from dclab_rnd.draft import orchestrator as orch

        said = "Please ask the teams: can our data rank the accounts by churn risk seven days before each renewal?"
        self.assertEqual(orch.plan(said, None), [("ask_teams", {"track": "model", "request": said})])
        self.assertEqual(orch.teams_track("Ask the teams to clean and merge the renewal tables"), "prepare")
        self.assertEqual(orch.teams_track("check with the data science team why renewals dropped"), "analyze")
        self.assertNotIn("ask_teams", [n for n, _ in orch.plan("show the models trained", None)])


    def test_while_the_teams_question_waits_only_an_answer_is_taken_as_the_answer(self):
        """The live run of 2026-11-21 (S5, S6): the same request again and a question of the person's own were both recorded
        as the answer to "does churn mean cancellation or failure to renew?"."""
        out = self.ask()
        self.pump()
        task = lead.waiting(self.services, self.pid, self.draft)
        self.assertEqual(task.id, out["task_id"])
        self.assertTrue(lead.is_answer(ANSWER, task))
        self.assertTrue(lead.is_answer("cancellation", task))
        self.assertFalse(lead.is_answer(task.question, task))  # the request again
        self.assertFalse(lead.is_answer("Please ask the teams: " + task.question, task))
        self.assertFalse(lead.is_answer("Wait, before that: which model did the last run choose and what was its holdout score?", task))
        self.assertFalse(lead.is_answer("Also ask the teams to look at the discount history", task))
        self.assertFalse(lead.is_answer("   ", task))
        # review F2: an answer may quote part of the request, or start with "tell the team"
        self.assertTrue(lead.is_answer("seven days before each renewal", task))
        self.assertTrue(lead.is_answer("Tell the team: failure to renew", task))

    def test_a_card_held_under_a_lock_no_turn_flushes_is_posted_before_the_next_card(self):
        """Review F4: a card held while a lock holder without the turn decorator held the draft was never posted, and the
        report came in ahead of it. The next card posted directly posts the held ones first."""
        drafts = self.services.drafts
        early = {"id": "team-early", "role": "agent", "text": "held question", "team": {"from": "lead", "kind": "clarification"}}
        drafts.update(self.draft, lambda d: d.setdefault(lead.HELD, []).append(early))
        out = self.ask()
        self.pump()
        ids = [m.get("id") for m in drafts.get(self.draft)["messages"]]
        self.assertIn("team-early", ids)
        later = [m for m in drafts.get(self.draft)["messages"] if m.get("team") and m.get("id") != "team-early"]
        self.assertTrue(later)
        self.assertLess(ids.index("team-early"), ids.index(later[0]["id"]))
        self.assertFalse(drafts.get(self.draft).get(lead.HELD))
        self.assertTrue(out["task_id"])

    def test_the_same_request_in_other_capitals_or_punctuation_is_one_task(self):
        """Live S5: the model quoted the same request as "can our data …?" and then "Can our data …?": two tasks, two questions."""
        self.assertEqual(lead.task_id_for("d1", "Can our data rank accounts?", "sara"), lead.task_id_for("d1", "can our data  rank accounts", "sara"))
        self.assertNotEqual(lead.task_id_for("d1", "can our data rank accounts", "sara"), lead.task_id_for("d1", "can our data rank products", "sara"))

    def test_the_built_chats_rules_say_what_they_did_with_the_request(self):
        """Live 2026-11-21: on the rules path (the product today) the chat said only "Done." after handing a request to the Lead."""
        from dclab_rnd.draft import orchestrator as orch

        self.services.projects.update(self.pid, team_inputs=[SAMPLE], default_problem_id=self.prob)
        said = "Please ask the teams: can our data rank the accounts by churn risk seven days before each renewal?"
        self.services.drafts.update(self.draft, lambda d: d.update(project_id=self.pid, problem_id=self.prob))
        self.services.drafts.update(self.draft, lambda d: d["messages"].append({"id": "u1", "role": "user", "kind": "text", "text": said}))
        for again in (False, True):
            turn = orch.Turn(self.services, self.pid, self.draft, agent=None, text=said)
            orch.ask_teams(turn, track="model", request=said)
            words = orch.Orchestrator._rules_words(orch.Orchestrator.__new__(orch.Orchestrator), turn)
            self.assertNotEqual(words.strip(), "Done.")
            self.assertIn("already has this request" if again else "Workspace Lead, who asks the teams", words)

    def test_the_request_asked_again_inside_other_words_is_the_open_task(self):
        """Live S5 (final run): "Please ask the teams: can our data …?" then "can our data …?" opened two tasks."""
        first = self.ask(question="Please ask the teams: can our data rank the accounts by churn risk before renewal?")
        again = self.ask(question="Can our data rank the accounts by churn risk before renewal")
        self.assertEqual(again["task_id"], first["task_id"])
        self.assertTrue(again["already_open"])
        other = self.ask(question="Can our data explain why the renewal discounts were given?")
        self.assertNotEqual(other["task_id"], first["task_id"])
        narrower = self.ask(question="Can our data rank the accounts by churn risk before renewal for enterprise customers only?")
        self.assertNotEqual(narrower["task_id"], first["task_id"])  # review F3: a narrower request is its own task

    # ------------------------------------------------------------------ the review of 2026-11-22
    def test_r22_3_the_built_chats_model_never_reads_another_persons_lead_card(self):
        """A shared draft with sign-in on: Omar's model read Sara's card from the Lead in the conversation's history."""
        import os
        from unittest import mock

        from dclab_rnd.accounts.principal import Principal, acting
        from dclab_rnd.draft import orchestrator as orch

        self.services.drafts.update(self.draft, lambda d: d["messages"].append({"id": "u-first", "role": "user", "kind": "text", "text": BRIEF}))
        self.ask()
        self.pump()
        [card] = self.chat()
        self.services.drafts.update(self.draft, lambda d: d["messages"].extend(
            [{"id": "u-next", "role": "user", "kind": "text", "text": "What does this project predict?"},
             {"id": "u-last", "role": "user", "kind": "text", "text": "and its score?"}]))
        home = orch.Orchestrator(self.services, SimpleNamespace(store=self.services.drafts))
        said = lambda: "\n".join(m["content"] for m in home.history(self.draft))  # noqa: E731
        ws = self.services.workspace_id
        with mock.patch.dict(os.environ, {"DCLAB_AUTH": "password"}):
            with acting(Principal(role="data_scientist", workspace_id=ws, user_id="omar", via="session")):
                self.assertNotIn(card["text"][:40], said())
            with acting(Principal(role="data_scientist", workspace_id=ws, user_id="sara", via="session")):
                self.assertIn(card["text"][:40], said())
        self.assertIn(card["text"][:40], said())  # no sign-in: the local owner reads it

    def test_r22_4_an_answer_home_skipped_is_not_said_to_have_reached_the_teams(self):
        out = self.ask()
        self.pump()
        first = lead.answer(self.services, self.pid, out["task_id"], person="sara", text=ANSWER, initiator=self.sara)
        self.assertIn("has given your answer to the teams", lead.answered_words(first))
        again = lead.answer(self.services, self.pid, out["task_id"], person="sara", text="Cancellation", initiator=self.sara)
        self.assertEqual((again["refused"], again["steps"]), ([], []))  # Home's rules skipped it: already answered
        words = lead.answered_words(again)
        self.assertNotIn("has given your answer", words)
        self.assertIn("already answered", words)
        self.assertIn("could not be passed", lead.answered_words({"refused": ["no"], "steps": []}))

    def test_r22_7_ml_engineering_takes_the_outcome_data_science_confirmed_not_the_persons_first_words(self):
        """The person's own words come first in the question: "The outcome is: …" there is not what Data Science confirmed."""
        out = self.ask(question=self.RANK + "\nThe outcome is: a late payment.", track="model")
        self.pump()
        lead.answer(self.services, self.pid, out["task_id"], person="sara", text="Cancellation of the subscription.", initiator=self.sara)
        self.pump()
        [said] = self.view(f"{out['task_id']}.ml_engineering")["reports"]
        self.assertIn('the outcome is "Cancellation of the subscription", as confirmed with Data Science', said.answer)
        self.assertNotIn("late payment", said.answer)

    def test_r22_the_teams_question_still_waits_is_said_once(self):
        out = self.ask()
        self.pump()
        [card] = self.chat()
        task = self.work.task(self.pid, out["task_id"])
        messages = self.services.drafts.get(self.draft)["messages"]
        self.assertTrue(lead.still_waiting_due(messages, task))
        said = {"id": "m-wait", "role": "agent", "text": "still waits", "team": {"from": "lead", "kind": "still_waiting", "task_id": task.id}}
        self.assertFalse(lead.still_waiting_due(messages + [said], task))
        again = {"id": "m-q2", "role": "agent", "text": "another question", "team": {"from": "lead", "kind": "clarification", "task_id": task.id}}
        self.assertTrue(lead.still_waiting_due(messages + [said, again], task))  # a new question: said again, once
        self.assertEqual(card["team"]["kind"], "clarification")

    def test_r22_the_lead_agents_tool_is_refused_without_a_home_conversation(self):
        from dclab_rnd.agents import case_agent as A

        self.services.projects.update(self.pid, team_inputs=[SAMPLE])
        state = self.case([BRIEF])
        state.draft_id = None
        self.assertIn("Home conversation", A._ask_teams(state, "analyze")["error"])
        self.assertEqual(self.work.deliveries(project_id=self.pid), [])

    def test_r22_a_quote_of_the_request_is_a_real_span_not_two_words_or_a_negation(self):
        from dclab_rnd.agents import case_agent as A

        self.services.projects.update(self.pid, team_inputs=[SAMPLE])
        words = ["Do not rank the accounts yet. First explain why renewals dropped last quarter in the north region."]
        out = A._ask_teams(self.case(words), "analyze", quote="rank the accounts")
        self.assertEqual(self.work.task(self.pid, out["task_id"]).question, words[0])  # a negated span: their whole message instead
        self.assertFalse(lead.quote_is_request("the accounts", words))  # two words
        self.assertFalse(lead.quote_is_request("rank the accounts yet", words))  # inside "do not"
        self.assertTrue(lead.quote_is_request("explain why renewals dropped last quarter", words))
        self.assertTrue(lead.quote_is_request("Rank accounts by churn", ["Rank accounts by churn"]))  # the whole short request

    def test_the_no_data_words_are_for_the_person_and_whole(self):
        """Live S4: the rules showed this text cut at 200 characters, ending in an order meant for the model ("Say so to the ")."""
        self.assertLess(len(lead.NO_DATA_YET), 200)
        self.assertNotIn("Say so", lead.NO_DATA_YET)
        self.assertTrue(lead.NO_DATA_YET.endswith("."))


class LeadRulesReviewTests(unittest.TestCase):
    """The review of 2026-11-22: words, not an understanding; each case a sentence the review ran."""

    def names(self, said):
        from dclab_rnd.draft import orchestrator as orch

        return [n for n, _ in orch.plan(said, None)]

    def test_r22_1_an_order_to_ask_the_teams_only_when_it_is_one(self):
        for said in ("Please don't ask the team, just show me the data", "I will ask my team about the horizon tomorrow",
                     "Do not ask the teams yet; show the models", "Never hand it to the lead", "We'll check with the team later",
                     "I'm going to ask the teams tomorrow", "Show the models, but don't ask the teams",
                     # reviewer round 2
                     "Please don't ask the team", "Never send this to the teams", "Can you not ask the teams yet",
                     "I will ask my team later", "No need to ask the team", "Ask the team? No, show me the data."):
            self.assertNotIn("ask_teams", self.names(said), said)
        self.assertIn("show_data", self.names("Ask the team? No, show me the data."))
        self.assertIn("show_data", self.names("Please don't ask the team, just show me the data"))
        for said in ("Please ask the teams: can our data rank the accounts by churn risk seven days before each renewal?",
                     "Ask the teams to clean and merge the renewal tables", "check with the data science team why renewals dropped",
                     "Customers are leaving. Please ask the teams to rank the accounts by churn risk.",
                     "Can you ask the teams whether the renewal dates are complete?",
                     # reviewer round 2: a comma starts an order too (handed over before the first fix)
                     "Yes, ask the teams", "OK, ask the teams", "Please, ask the teams",
                     "Good idea, ask the teams to check the tables"):
            self.assertIn("ask_teams", self.names(said), said)
        self.assertEqual(self.names("Sure, ask Data Science"), ["ask_teams"])

    def test_r22_1_show_the_data_and_then_ask_the_teams_does_both(self):
        from dclab_rnd.draft import orchestrator as orch

        said = "Show the data and then ask the teams whether it works"
        planned = orch.plan(said, None)
        self.assertEqual([n for n, _ in planned], ["show_data", "ask_teams"])
        self.assertEqual(planned[-1][1]["request"], said)  # the teams get the person's whole words
        self.assertEqual(self.names("Customers are leaving. Please ask the teams: can our data rank accounts?"), ["ask_teams"])

    def test_r22_8_the_live_scripts_refuse_any_claude_model_however_it_is_configured(self):
        import os
        from unittest import mock

        sys.path.insert(0, str(ROOT / "scripts"))
        from no_claude import claude_settings

        blank = {k: "" for k in os.environ if k.startswith(("DCLAB_TIER_", "DCLAB_LLM_", "OPENROUTER_", "OPENAI_", "ANTHROPIC_"))}
        blank.update(DCLAB_INTERN_MODEL="")
        cases = {"provider": {"DCLAB_LLM_PROVIDER": "anthropic"},
                 "openrouter": {"DCLAB_LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "x", "OPENROUTER_MODEL": "anthropic/claude-sonnet-4.5"},
                 "tier": {"DCLAB_TIER_STRONG_MODEL": "claude-opus-4"},
                 "endpoint": {"DCLAB_TIER_CHEAP_BASE_URL": "https://api.anthropic.com/v1", "DCLAB_TIER_CHEAP_MODEL": "x"},
                 "intern": {"DCLAB_INTERN_MODEL": "Claude-3-haiku"}}
        for name, env in cases.items():
            with mock.patch.dict(os.environ, {**blank, **env}):
                found = claude_settings()
                self.assertTrue(found, name)
                self.assertFalse(any("x" == f.split("=")[-1] for f in found), "a key's value is never named")
        with mock.patch.dict(os.environ, {**blank, "DCLAB_LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "x",
                                          "OPENROUTER_MODEL": "qwen/qwen3.8-flash", "ANTHROPIC_MODEL": "claude-sonnet-5.5"}):
            self.assertEqual(claude_settings(), [])  # Claude's name kept for another provider, unused here: the run may go on
        # reviewer round 2: blocking Anthropic is no Claude model, and a blocked tier serves nothing
        with mock.patch.dict(os.environ, {**blank, "DCLAB_LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "x",
                                          "OPENROUTER_MODEL": "qwen/qwen3.8-flash", "DCLAB_BLOCK_PROVIDERS": "openai,anthropic",
                                          "DCLAB_TIER_CHEAP_BASE_URL": "https://api.anthropic.com", "DCLAB_TIER_CHEAP_API_KEY": "x"}):
            self.assertEqual(claude_settings(), [])
        for script in ("live_lead_scenarios.py", "live_lead_teams.py"):
            text = (ROOT / "scripts" / script).read_text(encoding="utf-8")
            self.assertIn("refuse_claude(", text, script)
            self.assertIn("mark_invented_tables", text, script)
            self.assertIn("thinking_record(", text, script)
            self.assertLess(text.index("load_dotenv(ROOT"), text.index("refuse_claude("), script)  # after .env is read

    def test_r22_8_a_run_without_a_model_is_not_refused_and_reaches_none(self):
        """Reviewer round 2: --rules and --report call no model; a Claude setting then turns live models off, never refuses."""
        import os
        from unittest import mock

        sys.path.insert(0, str(ROOT / "scripts"))
        from no_claude import refuse_claude

        with mock.patch.dict(os.environ, {"DCLAB_INTERN_MODEL": "claude-3-haiku", "DCLAB_NO_LIVE_MODELS": ""}):
            with self.assertRaises(SystemExit):
                refuse_claude()
            refuse_claude(model_used=False)
            self.assertEqual(os.environ["DCLAB_NO_LIVE_MODELS"], "1")

    def test_r22_8_only_calls_that_answered_must_keep_their_thinking(self):
        """Reviewer round 2: a timeout or a 429 was counted as a call without its thinking and failed the run (rule 9: the
        non-error exchanges)."""
        import json
        import tempfile

        sys.path.insert(0, str(ROOT / "scripts"))
        from no_claude import thinking_record

        home = Path(tempfile.mkdtemp())
        (home / "model_exchanges" / "p").mkdir(parents=True)
        rows = [{"outcome": "ok", "purpose": "home", "thinking": "summary"}, {"outcome": "error: TimeoutError", "purpose": "home"},
                {"outcome": "error: RateLimit", "purpose": "home"}]
        (home / "model_exchanges" / "p" / "a.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        out = thinking_record(home)
        self.assertEqual((out["model_calls"], out["answered_calls"], out["errored_calls"], out["calls_with_thinking"]), (3, 1, 2, 1))
        self.assertTrue(out["thinking_kept"])
        self.assertNotIn("failed", out)
        with open(home / "model_exchanges" / "p" / "a.jsonl", "a", encoding="utf-8") as f:
            f.write("\n" + json.dumps({"outcome": "ok", "purpose": "home"}))
        out = thinking_record(home)
        self.assertFalse(out["thinking_kept"])
        self.assertIn("1 answered model call(s) without their thinking", out["failed"])


for _name in [n for n in dir(_Contract) if n.startswith("test")]:  # the W5 walkthroughs run in their own file, not again here
    setattr(_Lead, _name, None)


class LeadWorkspaceTests(_Lead, unittest.TestCase):
    pass


class PgLeadWorkspaceTests(_Lead, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}


class SignedInLeadRouteTests(unittest.TestCase):
    """Password sign-in on PostgreSQL: Sara asks on Home, the teams ask her back, and her answer in the Home chat reaches them
    as hers (the principal of her request is the initiator the outbox rechecks)."""

    @classmethod
    def setUpClass(cls):
        import os
        import tempfile
        from unittest import mock

        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            raise unittest.SkipTest(f"Studio dependencies not installed: {error}")
        url = pgtest.require()
        pgtest.empty(url)
        cls.patch = mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": url, "DCLAB_AUTH": "password", "OPENAI_API_KEY": "",
                                                "DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS": "0", "DCLAB_LLM_PROVIDER": ""})
        cls.patch.start()
        from dclab_rnd.accounts.store import Accounts

        cls.app = create_app(Path(tempfile.mkdtemp()))
        cls.main = TestClient(cls.app)
        cls.main.__enter__()
        s = cls.services = cls.app.state.services
        accounts = Accounts()
        cls.sara = accounts.create_user(f"sara-{os.getpid()}-{os.urandom(3).hex()}@example.com", "Sara", password="a-long-test-password-1")
        accounts.set_member(s.workspace_id, cls.sara["id"], "data_scientist")
        cls.client = TestClient(cls.app)
        csrf = cls.client.get("/api/config").json()["csrf"]
        r = cls.client.post("/api/auth/password", json={"email": cls.sara["email"], "password": "a-long-test-password-1"},
                            headers={"X-DCLab-Token": csrf})
        assert r.status_code == 200, r.text
        cls.client.headers["X-DCLab-Token"] = r.json()["csrf"]
        cls.omar = accounts.create_user(f"omar-{os.getpid()}-{os.urandom(3).hex()}@example.com", "Omar", password="a-long-test-password-2")
        accounts.set_member(s.workspace_id, cls.omar["id"], "data_scientist")
        cls.other = TestClient(cls.app)
        csrf = cls.other.get("/api/config").json()["csrf"]
        r = cls.other.post("/api/auth/password", json={"email": cls.omar["email"], "password": "a-long-test-password-2"},
                           headers={"X-DCLab-Token": csrf})
        assert r.status_code == 200, r.text
        cls.other.headers["X-DCLab-Token"] = r.json()["csrf"]
        delivery.set_target(None)
        delivery.set_membership_checker(None)

    @classmethod
    def tearDownClass(cls):
        cls.main.__exit__(None, None, None)
        cls.patch.stop()

    def pump(self, work):
        """Wait for the app's own worker to deliver what is pending (a second deliverer here would race it)."""
        import time

        deadline = time.time() + 90
        while time.time() < deadline:
            if not work.pending_heads():
                stuck = [r for r in work.deliveries() if r["status"] not in ("delivered", "pending")]
                self.assertEqual(stuck, [], [(r["reason"], r.get("last_error")) for r in stuck])
                return
            time.sleep(0.2)
        raise AssertionError("the deliveries never settled")

    def test_her_answer_in_the_home_chat_reaches_the_team_as_hers(self):
        import time

        from dclab_rnd.memory.scope import create_problem
        from dclab_rnd.storage import open_team_work

        s = self.services
        pid = s.projects.create("Renewal retention", goal="Who should Sales call")["id"]
        prob = create_problem(s.projects, pid, "Renewal", workspace_id=s.workspace_id)["problem_id"]
        draft = s.drafts.create("Renewal on Home")["id"]
        s.drafts.update(draft, lambda d: d.update(project_id=pid, problem_id=prob))
        work = open_team_work(s.store.home, projects=s.projects)
        initiator = {"user_id": self.sara["id"], "role": "data_scientist", "workspace_id": s.workspace_id}
        out = lead.ask(s, pid, prob, question=BRIEF, track="analyze", person=self.sara["id"], draft_id=draft, initiator=initiator,
                       inputs=(SAMPLE,))
        self.assertEqual(out["refused"], [])
        self.pump(work)
        self.assertIsNotNone(lead.waiting(s, pid, draft))
        r = self.other.post(f"/api/drafts/{draft}/messages", json={"text": "What does this project predict?"})
        self.assertEqual(r.status_code, 202, r.text)
        time.sleep(1.0)
        self.assertIsNotNone(lead.waiting(s, pid, draft, person=self.sara["id"]))  # Omar's message was not taken as her answer
        lead_cards = lambda client: [m for m in client.get(f"/api/drafts/{draft}").json()["messages"]  # noqa: E731
                                     if isinstance(m.get("team"), dict) and m["team"].get("from") == "lead"]
        self.assertEqual(lead_cards(self.other), [])  # her cards are hers
        self.assertTrue(lead_cards(self.client))
        deadline = time.time() + 60
        while time.time() < deadline and (s.drafts.get(draft).get("status") == "answering"):
            time.sleep(0.5)
        for _ in range(40):  # the agent's turn on Omar's message ends before hers is posted
            r = self.client.post(f"/api/drafts/{draft}/messages", json={"text": ANSWER})
            if r.status_code != 409:
                break
            time.sleep(0.5)
        self.assertEqual(r.status_code, 202, r.text)
        deadline = time.time() + 60
        while time.time() < deadline and not any(isinstance(m.get("team"), dict) and m["team"].get("kind") == "answered"
                                                 for m in s.drafts.get(draft)["messages"]):
            time.sleep(0.2)
        told = [m for m in s.drafts.get(draft)["messages"] if isinstance(m.get("team"), dict) and m["team"].get("kind") == "answered"]
        self.assertEqual(len(told), 1, s.drafts.get(draft)["messages"][-3:])
        self.assertIn("given your answer", told[0]["text"])
        self.pump(work)
        task = work.task(pid, out["task_id"])
        self.assertEqual(task.status, "partial")  # Data Science answered; Sales waits for its capacity
        child = work.task_view(pid, f"{out['task_id']}.data_science")
        humans = [a for a in child["actions"] if a.source.startswith("human:") and a.status in tw.DONE_ACTIONS]
        self.assertEqual({a.responded_by.id for a in humans}, {self.sara["id"]})
        self.assertEqual([m["team"]["kind"] for m in s.drafts.get(draft)["messages"] if isinstance(m.get("team"), dict)],
                         ["clarification", "answered", "reply"])
        # the workspace page: Sara sees her conversation with the Lead and reads its task page; Omar sees neither
        mine = self.client.get("/api/workspace/overview").json()
        self.assertEqual([c["task_id"] for c in mine["conversations"]], [out["task_id"]])
        self.assertEqual(mine["conversations"][0]["draft_id"], draft)
        self.assertEqual(self.client.get(f"/api/projects/{pid}/team-tasks/{out['task_id']}").status_code, 200)
        theirs = self.other.get("/api/workspace/overview").json()
        self.assertEqual(theirs["conversations"], [])
        self.assertEqual(self.other.get(f"/api/projects/{pid}/team-tasks/{out['task_id']}").status_code, 403)
        listed = [t["id"] for t in self.client.get("/api/team-threads").json()]
        self.assertIn(out["thread_id"], listed)
        self.assertNotIn(out["thread_id"], [t["id"] for t in self.other.get("/api/team-threads").json()])


if __name__ == "__main__":
    unittest.main()
