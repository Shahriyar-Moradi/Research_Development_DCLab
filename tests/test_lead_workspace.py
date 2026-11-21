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

    def to_the_end(self):
        out = self.ask()
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
        self.assertEqual(task.status, "complete")
        view = self.view(task.id)
        self.assertEqual([c.owner_team for c in view["children"]], ["data_science"])
        [report] = view["reports"]
        self.assertTrue(report.answer.startswith("Data Science: "), report.answer)
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
        found, why, nxt, changed = cards[-1]["text"].split("\n")
        self.assertIn("Data Science:", found)
        self.assertTrue(why.startswith("Why: ") and nxt.startswith("Next: ") and changed.startswith("Changed: "))
        self.assertNotIn("@r", changed)  # the report's revision id stays on the task page
        routed = {(r.subject.key.split(":", 1)[1], r.value_used, dict(r.state)["routed_by"])
                  for r in self.services.ledger.records(point="team.task_owner")}
        self.assertIn((task.id, "lead", "requester"), routed)
        self.assertIn((child.id, "data_science", "parent_owner"), routed)

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
        self.assertEqual(self.work.task(self.pid, out["task_id"]).status, "complete")


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

    def test_the_no_data_words_are_for_the_person_and_whole(self):
        """Live S4: the rules showed this text cut at 200 characters, ending in an order meant for the model ("Say so to the ")."""
        self.assertLess(len(lead.NO_DATA_YET), 200)
        self.assertNotIn("Say so", lead.NO_DATA_YET)
        self.assertTrue(lead.NO_DATA_YET.endswith("."))


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
        self.assertEqual(task.status, "complete")
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
