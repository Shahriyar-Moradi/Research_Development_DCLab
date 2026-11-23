"""T3 of the team workspace (2026-11-23): Sales and Marketing ask their own teams.

Sales reads the dated do-not-contact list, asks a member of the Sales team (in the Sales thread) for the call capacity and
returns a call list capped at that number as a restricted reference: counts only, no account row. A capacity of 0 is a
blocker, words with no number a hint (the question stays open), an answer by someone outside the team is refused, the
same answer twice is recorded once, and a member removed before the reply is delivered blocks that delivery. Marketing
asks for the offer's terms and returns them checked, or a blocker naming what is missing. Sales and Marketing answer
independently of each other; words that could answer two open questions answer none and the agent asks which; a task for
an agent with no rules is answered "unsupported" at once. On files and on PostgreSQL (``tests/run_parallel.py --pg``).
"""

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

from dclab_rnd import context, experience  # noqa: E402
from dclab_rnd.jobs import open_jobs  # noqa: E402
from dclab_rnd.memory.scope import create_problem  # noqa: E402
from dclab_rnd.storage import open_stores, open_team_work, open_teams  # noqa: E402
from dclab_rnd.teamspace import agents, commercial, delivery  # noqa: E402
from dclab_rnd.teamspace.threads import threads_for  # noqa: E402

CALLS = "Which accounts renewing in the next 60 days should Sales call this week?"
OFFER = "Which retention offer may we propose to accounts renewing in December?"
PASSWORD = "commercial-test-pass-1"  # a test account's password on the test database only
TERMS = ("Offer: EUR 10 credit per account, budget EUR 1000, the campaign runs from 2026-12-01 to 2026-12-31, "
         "eligible: accounts renewing in December with contact consent.")


class _Commercial:
    env: dict = {}

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"DCLAB_AUTH": "", "DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS": "0", **self.env})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.home = Path(tempfile.mkdtemp()) / "workspace"
        self.home.mkdir()
        from dclab_rnd.storage import db

        projects, drafts, _ = open_stores(self.home)
        wid = db.workspace_for(self.home) if db.database_url(required=False) else None
        self.services = SimpleNamespace(store=SimpleNamespace(home=self.home), projects=projects, drafts=drafts, workspace_id=wid,
                                        experience=experience.open_log(self.home, wid), job_store=open_jobs(self.home),
                                        teamwork=open_team_work(self.home, projects=projects))
        self.work = self.services.teamwork
        self.pid = projects.create("Renewal calls", goal="Who should Sales call")["id"]
        projects.update(self.pid, team_inputs=["sample:renewal_v1"])  # the invented sample the case may read
        self.prob = create_problem(projects, self.pid, "Renewal", workspace_id=self.work.workspace)["problem_id"]
        target = delivery.set_target(None)
        self.addCleanup(delivery.set_target, target)
        checker = delivery.set_membership_checker(None)
        self.addCleanup(delivery.set_membership_checker, checker)
        token = context.set_services(self.services)
        self.addCleanup(context.reset_services, token)
        self.sara, self.raha, self.omid = (self.member(n) for n in ("sara", "raha", "omid"))
        self.teams = open_teams(self.home)
        business = self.teams.create_team("Business", "business")
        self.teams.add_member(business["id"], self.sara)
        self.sales_team = self.teams.create_team("Sales", "sales")
        self.teams.add_member(self.sales_team["id"], self.raha)
        marketing = self.teams.create_team("Marketing", "marketing")
        self.teams.add_member(marketing["id"], self.omid)
        self.threads = threads_for(self.services)
        self.biz = self.threads.open_thread(self.pid, self.prob, business["id"], label="Business")[0]

    # ------------------------------------------------------------------ helpers
    def member(self, name):
        """A person of this workspace: any id on files, an account on PostgreSQL (a team's member must be one there)."""
        from dclab_rnd.storage import db

        if not db.database_url(required=False):
            return name
        from dclab_rnd.accounts.store import Accounts

        accounts = Accounts()
        user = accounts.create_user(f"{name}-{os.urandom(4).hex()}@example.com", name.title(), password=PASSWORD)["id"]
        accounts.set_member(self.work.workspace, user, "data_scientist")
        return user

    def pump(self, limit=80):
        for _ in range(limit):
            heads = self.work.pending_heads()
            if not heads:
                return
            head = heads[0]
            for row in delivery.run(self.services, {"project_id": head["project_id"], "task_id": delivery.lane_of(self.work, head)}):
                self.assertEqual(row["status"], "delivered", row.get("reason"))
        raise AssertionError("the deliveries never settled")

    def ask(self, team, text, message_id):
        out = agents.from_thread(self.services, self.biz, person=self.sara, text=text, team_kind="business", ask=team,
                                 message_id=message_id)
        self.assertEqual(out["refused"], [])
        self.pump()
        return out["task_id"]

    def thread(self, kind):
        team = next(t for t in self.teams.teams() if t["kind"] == kind)
        return self.threads.find_thread(self.pid, self.prob, team["id"])

    def answer(self, kind, person, text, **kw):
        out = agents.from_thread(self.services, self.thread(kind), person=person, text=text, team_kind=kind, **kw)
        self.pump()
        return out

    def view(self, task_id):
        return self.work.task_view(self.pid, task_id)

    def said(self, kind, handoff):
        return [m for m in self.threads.messages(self.thread(kind)["id"]) if (m.get("handoff") or {}).get("kind") == handoff]

    def report(self, task_id):
        reports = [r for r in self.view(task_id)["reports"] if r.status != "superseded"]
        self.assertEqual(len(reports), 1, reports)
        return reports[0]

    # ------------------------------------------------------------------ Sales
    def test_sales_asks_its_own_team_and_returns_a_capped_restricted_list(self):
        task = self.ask("sales", CALLS, "m-1")
        view = self.view(task)
        self.assertEqual(view["task"].status, "accepted")  # it waits on its own team, never on the person who asked
        [asked] = self.said("sales", "clarification")
        self.assertIn("0 to 30", asked["text"])
        self.assertEqual(commercial.awaiting(view)["action"], "ask_capacity")
        self.assertIn("as of 2026-11-20", next(a for a in view["actions"] if a.action == "check_exclusions").output_ref)
        out = self.answer("sales", self.raha, "18, Raha is out on Friday", message_id="s-1")
        self.assertEqual(out["refused"], [])
        view = self.view(task)
        self.assertEqual(view["task"].status, "complete")
        [ask] = [a for a in view["actions"] if a.action == "ask_capacity"]
        self.assertEqual(ask.responded_by.id, self.raha)
        report = self.report(task)
        self.assertIn("18 calls", report.answer)
        self.assertIn("18 of the 102 eligible accounts", report.answer)
        self.assertIn("restricted:sales:", report.answer)
        self.assertNotIn("Raha is out", report.answer)  # the member's words stay in the Sales thread
        [reply] = [m for m in self.threads.messages(self.biz["id"]) if (m.get("handoff") or {}).get("kind") == "reply"]
        self.assertIn("18 calls", reply["text"])

    def test_a_capacity_of_zero_is_a_blocker_and_nobody_is_contacted(self):
        task = self.ask("sales", CALLS, "m-1")
        self.answer("sales", self.raha, "0 this week", message_id="s-1")
        self.assertEqual(self.view(task)["task"].status, "blocked")
        report = self.report(task)
        self.assertEqual(report.status, "blocked")
        self.assertIn("capacity 0", report.answer)

    def test_words_with_no_number_get_a_hint_and_the_question_stays_open(self):
        task = self.ask("sales", CALLS, "m-1")
        out = self.answer("sales", self.raha, "a few, I think", message_id="s-1")
        self.assertEqual(out["did"], "hint")
        self.assertEqual(len(self.said("sales", "hint")), 1)
        self.assertIsNotNone(commercial.awaiting(self.view(task)))
        self.answer("sales", self.raha, "12", message_id="s-2")
        self.assertEqual(self.view(task)["task"].status, "complete")

    def test_someone_outside_the_sales_team_cannot_answer_for_it(self):
        task = self.ask("sales", CALLS, "m-1")
        out = self.answer("sales", self.sara, "25", message_id="s-1")
        self.assertTrue(out["refused"] and "not in the workspace's sales team" in out["refused"][0], out)
        self.assertIsNotNone(commercial.awaiting(self.view(task)))

    def test_the_same_answer_twice_is_recorded_once(self):
        task = self.ask("sales", CALLS, "m-1")
        self.answer("sales", self.raha, "18", response_id="once")
        self.answer("sales", self.raha, "18", response_id="once")
        actions = self.view(task)["actions"]
        self.assertEqual(len([a for a in actions if a.action == "ask_capacity"]), 1)
        self.assertEqual(len([a for a in actions if a.action == "draft_contact_list"]), 1)
        self.assertEqual(len(self.view(task)["reports"]), 1)

    def test_a_member_removed_before_delivery_blocks_the_reply(self):
        task = self.ask("sales", CALLS, "m-1")
        with mock.patch.object(delivery, "run", lambda *a, **k: []):  # the reply is written, not yet delivered
            agents.from_thread(self.services, self.thread("sales"), person=self.raha, text="18", team_kind="sales", message_id="s-1")
        self.teams.remove_member(self.sales_team["id"], self.raha)
        head = next(h for h in self.work.pending_heads() if h["kind"] == "reply")
        rows = delivery.run(self.services, {"project_id": self.pid, "task_id": delivery.lane_of(self.work, head)})
        self.assertEqual([r["status"] for r in rows if r["kind"] == "reply"], ["blocked"])
        self.assertFalse([m for m in self.threads.messages(self.biz["id"]) if (m.get("handoff") or {}).get("kind") == "reply"])
        self.assertEqual(self.view(task)["task"].status, "complete")

    def test_without_a_sales_team_the_capacity_is_a_blocker_naming_who_supplies_it(self):
        self.teams.delete_team(self.sales_team["id"])
        task = self.ask("sales", CALLS, "m-1")
        self.assertEqual(self.view(task)["task"].status, "blocked")
        self.assertEqual(self.report(task).next_task.unblocks, commercial.CAPACITY)

    def test_a_number_is_read_only_when_it_is_the_only_one(self):
        self.assertEqual(commercial.capacity_of("18, Raha is out on Friday"), 18)
        self.assertIsNone(commercial.capacity_of("2 of us are off, so 18"))
        self.assertIsNone(commercial.capacity_of("Mon 24 to Fri 28 we can do 15"))
        task = self.ask("sales", CALLS, "m-1")
        self.assertEqual(self.answer("sales", self.raha, "2 of us are off, so 18", message_id="s-1")["did"], "hint")
        self.assertIsNotNone(commercial.awaiting(self.view(task)))

    def test_a_resent_answer_never_lands_on_a_task_opened_since(self):
        first = self.ask("sales", CALLS, "m-1")
        self.answer("sales", self.raha, "18", response_id="r-1")
        self.assertEqual(self.view(first)["task"].status, "complete")
        second = self.ask("sales", "And for the week after?", "m-2")
        out = self.answer("sales", self.raha, "18", response_id="r-1", resend=True)  # the same message, sent again
        self.assertIsNone(out)
        self.assertIsNotNone(commercial.awaiting(self.view(second)))
        self.assertFalse([a for a in self.view(second)["actions"] if a.action == "ask_capacity" and a.responded_by])

    def test_a_question_lost_in_a_crash_is_asked_again_once(self):
        from dclab_rnd.teamspace.tools import TeamTurn

        with mock.patch("dclab_rnd.teamspace.tools.post_own", return_value=None):  # written running, never posted
            task = self.ask("sales", CALLS, "m-1")
        self.assertEqual(self.said("sales", "clarification"), [])  # the task arrived in the Sales thread; the question did not
        for _ in range(2):
            agents.run_turn(TeamTurn(self.services, self.work, self.pid, "sales", task_id=task))
        self.assertEqual(len(self.said("sales", "clarification")), 1)
        self.assertEqual(len([a for a in self.view(task)["actions"] if a.action == "ask_capacity"]), 1)
        self.answer("sales", self.raha, "10", message_id="s-1")
        self.assertEqual(self.view(task)["task"].status, "complete")

    def test_the_lead_never_joins_a_reply_whose_member_left_the_team(self):
        from dclab_rnd.ledger import open_ledger
        from dclab_rnd.teamspace import lead

        self.services.ledger = open_ledger(self.home, self.services.workspace_id)
        draft = self.services.drafts.create("Renewal on Home")["id"]
        sara = {"user_id": self.sara, "role": "data_scientist", "workspace_id": self.work.workspace}
        brief = ("Customers are leaving. Who should Sales call before renewal? Eligible: accounts renewing in the next 60 days. "
                 "Decision date: seven days before each renewal. Requested: can the data rank accounts by churn risk?")
        out = lead.ask(self.services, self.pid, self.prob, question=brief, track="analyze", person=self.sara, draft_id=draft,
                       initiator=sara, inputs=("sample:renewal_v1",))
        self.assertEqual(out["refused"], [])
        self.pump()
        sales = f"{out['task_id']}.sales"
        self.assertIsNotNone(commercial.awaiting(self.view(sales)))
        self.answer("sales", self.raha, "18", message_id="s-1")
        self.assertEqual(self.view(sales)["task"].status, "complete")
        self.teams.remove_member(self.sales_team["id"], self.raha)  # removed before Data Science's reply lets the Lead join
        said = lead.answer(self.services, self.pid, out["task_id"], person=self.sara, text="Failure to renew by the contractual renewal date.",
                           initiator=sara)
        self.assertEqual(said["refused"], [])
        self.pump()
        [report] = [r for r in self.view(out["task_id"])["reports"] if r.status != "superseded"]
        self.assertEqual(report.status, "partial")
        self.assertNotIn("18 calls", report.answer)
        self.assertIn("Sales: not joined", report.answer)
        self.assertEqual(report.next_task.owner, "sales")

    # ------------------------------------------------------------------ Marketing
    def test_amounts_with_separators_and_a_budget_that_funds_nothing(self):
        self.assertEqual(commercial.terms_of("budget EUR 1,000, EUR 10 credit per account")["budget"], 1000.0)
        self.assertEqual(commercial.terms_of("budget EUR 1.000, EUR 10.50 credit per account")["amount"], 10.5)
        self.assertEqual(commercial.terms_of("budget EUR 1.000, EUR 10.50 credit per account")["budget"], 1000.0)
        task = self.ask("marketing", OFFER, "m-1")
        self.answer("marketing", self.omid, "EUR 10 credit per account, budget EUR 5, from 2026-12-01 to 2026-12-31, "
                    "eligible: December renewals", message_id="k-1")
        self.assertEqual(self.view(task)["task"].status, "blocked")
        self.assertIn("funds no offer", self.report(task).answer)

    def test_marketing_returns_checked_terms(self):
        task = self.ask("marketing", OFFER, "m-1")
        self.assertEqual(self.view(task)["task"].status, "accepted")
        self.answer("marketing", self.omid, TERMS, message_id="k-1")
        self.assertEqual(self.view(task)["task"].status, "complete")
        answer = self.report(task).answer
        self.assertIn("EUR 10 an offer; a budget of EUR 1000 funds 100 offers", answer)
        self.assertIn("2026-12-01 to 2026-12-31", answer)
        self.assertIn(f"confirmed by {self.omid}", answer)

    def test_marketing_names_what_is_missing(self):
        task = self.ask("marketing", OFFER, "m-1")
        self.answer("marketing", self.omid, "EUR 10 credit per account, from 2026-12-01 to 2026-12-31, eligible: December renewals",
                    message_id="k-1")
        self.assertEqual(self.view(task)["task"].status, "blocked")
        self.assertIn("the budget", self.report(task).answer)

    # ------------------------------------------------------------------ both, and the thread
    def test_sales_and_marketing_answer_independently(self):
        calls = self.ask("sales", CALLS, "m-1")
        offer = self.ask("marketing", OFFER, "m-2")
        self.answer("marketing", self.omid, TERMS, message_id="k-1")  # Sales's member is offline
        self.assertEqual(self.view(offer)["task"].status, "complete")
        self.assertEqual(self.view(calls)["task"].status, "accepted")
        self.answer("sales", self.raha, "20", message_id="s-1")
        self.assertEqual(self.view(calls)["task"].status, "complete")

    def test_words_that_could_answer_two_tasks_answer_none_until_one_is_named(self):
        first = self.ask("sales", CALLS, "m-1")
        second = self.ask("sales", "And for the week after?", "m-2")
        out = self.answer("sales", self.raha, "15", message_id="s-1")
        self.assertEqual(out["did"], "which")
        self.assertEqual(sorted(out["which"]), sorted([first, second]))
        self.assertEqual(len(self.said("sales", "which_task")), 1)
        self.answer("sales", self.raha, "15", message_id="s-2", task_id=second)
        self.assertEqual(self.view(second)["task"].status, "complete")
        self.assertIsNotNone(commercial.awaiting(self.view(first)))

    def test_an_agent_with_no_rules_says_unsupported_at_once(self):
        with mock.patch.dict(agents.BRAINS):
            del agents.BRAINS["marketing"]
            task = self.ask("marketing", OFFER, "m-1")
        self.assertEqual(self.view(task)["task"].status, "blocked")
        self.assertIn("Unsupported", self.report(task).answer)
        self.assertEqual([a["name"] for a in agents.askable()], ["data_science", "ml_engineering", "sales", "marketing"])


class FileCommercialTests(_Commercial, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}


class PgCommercialTests(_Commercial, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}


if __name__ == "__main__":
    unittest.main()
