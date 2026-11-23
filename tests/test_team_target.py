"""T1 of the team workspace (2026-11-23, dclab_core 0.170.0): a feasibility claim means the outcome Business confirmed.

The regression pair on one renewal table: "failure to renew by the contractual date" passes on mature outcomes, "explicit
cancellation only" blocks with the missing record, why it matters, who supplies it and which task resumes. Around it:
paraphrases, an answer that does not settle the outcome (it stays open, then a clear one resumes the task), immature
outcomes, a late feature on a table that does record cancellations, the outcome in the Business report and in the stored
contract, and a changed outcome after feasibility (a new brief revision, the earlier report superseded, the same event
in other words changing nothing). On files and on PostgreSQL (``tests/run_parallel.py --pg``; never run directly: that
empties ``dclab_test``).
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402

from dclab_rnd.teamspace import agents  # noqa: E402
from test_teamspace_agents import ANSWER, FEASIBLE, _Contract  # noqa: E402

CANCEL = "Explicit cancellation only, not non-renewal."
LAPSE_WORDS = "failure to renew by the contractual renewal date"


class _Target(_Contract):
    def answer(self, text, sample="sample:renewal_v1", task_id="T-renewal"):
        self.open_renewal(task_id, sample)
        self.pump()
        out = agents.person_replies(self.services, self.pid, task_id, person="sara", text=text, initiator=self.sara)
        self.assertEqual(out["refused"], [])
        self.pump()
        return self.view(task_id)

    def events(self, kind):
        return [e.payload for e in self.services.experience.read(kinds=[kind])]

    def current(self, view):
        return [r for r in view["reports"] if r.status != "superseded"]

    # ------------------------------------------------------------------ the regression pair
    def test_failure_to_renew_passes_on_mature_outcomes_and_is_kept_as_the_tasks_contract(self):
        view = self.answer(ANSWER)
        task = view["task"]
        self.assertEqual(task.status, "complete")
        t = task.target
        self.assertEqual((t.event, t.words, t.confirmed_by, t.brief_revision, t.label_source),
                         ("non_renewal_by_date", ANSWER, "sara", 1, "sample:renewal_v1"))
        self.assertEqual((t.entity, t.prediction_time), ("account", "seven days before each renewal"))
        self.assertIn("renewal date has passed", t.maturity_rule)
        [report] = view["reports"]
        self.assertEqual(report.answer, FEASIBLE)
        # the outcome is in the Business report, as confirmed, with the column that records it
        self.assertIn(f"The outcome checked is {LAPSE_WORDS}, as sara confirmed (brief revision 1): renewed_by_date records it",
                      " ".join(report.limits))
        self.assertEqual(report.evidence[0].assumptions, [f"The outcome is: {ANSWER} (confirmed by sara)."])
        [confirmed] = self.events("team.target.confirmed")
        self.assertEqual((confirmed["event"], confirmed["changed"], confirmed["brief_revision"]), ("non_renewal_by_date", False, 1))
        self.assertNotIn(ANSWER, str(confirmed))  # codes and ids only: never the person's words
        self.assertEqual(self.events("team.target.mismatch"), [])

    def test_explicit_cancellation_on_the_renewal_table_blocks_with_what_is_missing_who_supplies_it_and_what_resumes(self):
        from dclab_rnd.ledger import capture

        seen, real = [], capture._put

        def keep(point, subject, answers, value, used, **kw):
            if point == "team.target_agreement":
                seen.append((value, kw.get("state")))
            return real(point, subject, answers, value, used, **kw)

        with mock.patch.object(capture, "_put", keep):
            view = self.answer(CANCEL)
        self.assertEqual(view["task"].status, "blocked")
        self.assertEqual(view["task"].target.event, "explicit_cancellation")
        [report] = view["reports"]
        self.assertEqual(report.status, "blocked")
        self.assertIn("the outcome confirmed is an explicit cancellation, but sample:renewal_v1 holds no record of it", report.answer)
        self.assertIn("renewed_by_date records failure to renew", report.answer)  # what the table does record
        self.assertIn("(missing field: a record of an explicit cancellation)", report.answer)
        self.assertIn("Why it matters: a model trained on what the table records would learn failure to renew", report.answer)
        self.assertIn("The Business team can supply it (the CRM's cancellation log", report.answer)
        self.assertIn("work resumes on T-renewal when it is attached to sample:renewal_v1", report.answer)
        self.assertNotIn("score seven days", report.answer)
        self.assertEqual((report.next_task.owner, report.next_task.unblocks),
                         ("business", "attach a record of an explicit cancellation to sample:renewal_v1"))
        [mismatch] = self.events("team.target.mismatch")
        self.assertEqual((mismatch["verdict"], mismatch["missing"], mismatch["recorded"], mismatch["supplier"]),
                         ("event_mismatch", "explicit_cancellation", ["non_renewal_by_date"], "business"))
        self.assertEqual([v for v, _ in seen], ["event_mismatch"])  # one judgment, where Data Science inspected the table
        self.assertEqual(seen[0][1]["target_event"], "explicit_cancellation")
        self.assertFalse(seen[0][1]["event_recorded"])

    def test_paraphrases_read_as_their_event(self):
        for n, (words, status) in enumerate([("Accounts that let the contract expire.", "complete"),
                                             ("Only when the customer cancels the subscription.", "blocked")]):
            with self.subTest(words=words):
                self.assertEqual(self.answer(words, task_id=f"T-{n}")["task"].status, status)

    # ------------------------------------------------------------------ an open question, immature outcomes, a late feature
    def test_an_answer_that_does_not_settle_it_stays_open_and_a_clear_one_resumes_the_task(self):
        view = self.answer("Either a cancellation or not renewing, I guess.")
        self.assertIsNone(view["task"].target)  # never a guess
        self.assertEqual(view["task"].status, "blocked")
        [report] = view["reports"]
        self.assertIn("the answer does not settle what the outcome means", report.answer)
        self.assertIn("work resumes on T-renewal when they do", report.answer)
        self.assertEqual(report.next_task.owner, "business")
        unsure = agents.confirm_outcome(self.services, self.pid, "T-renewal", person="sara", text="Maybe both?", initiator=self.sara)
        self.assertEqual((unsure["outcome"], unsure["changed"]), ("unsure", False))

        out = agents.confirm_outcome(self.services, self.pid, "T-renewal", person="sara", text="Failure to renew by the renewal date.",
                                     initiator=self.sara)
        self.assertEqual((out["outcome"], out["changed"], out["revised"], out["stale"]),
                         ("non_renewal_by_date", True, ["T-renewal"], ["T-renewal@r1"]))
        self.pump()
        view = self.view("T-renewal")
        self.assertEqual((view["task"].status, view["task"].brief_revision, view["task"].target.brief_revision), ("complete", 2, 2))
        [now] = self.current(view)
        self.assertEqual((now.revision, now.brief_revision, now.answer), (2, 2, FEASIBLE))

    def test_a_person_other_than_the_cases_member_cannot_confirm_the_outcome(self):
        self.answer(ANSWER)
        out = agents.confirm_outcome(self.services, self.pid, "T-renewal", person="omar", text=CANCEL)
        self.assertIn("only sara answers it", out["refused"][0])
        omar = {**self.sara, "user_id": "omar"}  # the review of T1: a call acting for omar cannot speak for sara
        out = agents.confirm_outcome(self.services, self.pid, "T-renewal", person="sara", text=CANCEL, initiator=omar)
        self.assertIn("acts for omar", out["refused"][0])
        self.assertEqual(self.view("T-renewal")["task"].target.event, "non_renewal_by_date")

    def test_while_the_question_is_open_the_outcome_is_answered_in_the_thread_not_set_aside(self):
        self.open_renewal()
        self.pump()
        self.assertEqual(self.view("T-renewal")["task"].status, "waiting_for_person")
        out = agents.confirm_outcome(self.services, self.pid, "T-renewal", person="sara", text=ANSWER, initiator=self.sara)
        self.assertIn("answer it in the thread", out["refused"][0])
        self.assertIsNone(self.view("T-renewal")["task"].target)

    def test_a_brief_that_names_the_outcome_still_has_it_confirmed_and_judged(self):
        """The review of T1: "who will cancel" named no vague word, so nothing was asked and the renewal column passed."""
        out = agents.open_case(self.services, self.pid, self.prob, task_id="T-cancel", thread_id=self.biz_thread, member="sara",
                               question="Which accounts will cancel their subscription? Decision date: seven days before each renewal.",
                               initiator=self.sara, inputs=("sample:renewal_v1",))
        self.assertEqual(out["refused"], [])
        self.pump()
        self.assertEqual(self.view("T-cancel")["task"].status, "waiting_for_person")
        agents.person_replies(self.services, self.pid, "T-cancel", person="sara", text="Explicit cancellation only.", initiator=self.sara)
        self.pump()
        view = self.view("T-cancel")
        self.assertEqual((view["task"].status, view["task"].target.event), ("blocked", "explicit_cancellation"))

    def test_a_crash_between_the_answer_and_its_outcome_saves_the_clear_words_on_the_retry(self):
        """The review of T1: the answer recorded, the process gone before the outcome was saved: the retry read the claimed
        question as an unsettled answer and blocked."""
        from dclab_rnd.teamspace import tools

        self.open_renewal()
        self.pump()
        agents.person_replies(self.services, self.pid, "T-renewal", person="sara", text=ANSWER, initiator=self.sara)
        with mock.patch.object(tools, "save_outcome", lambda *a, **kw: None):  # the process stops before the outcome is saved
            self.pump()
        self.assertIsNone(self.view("T-renewal")["task"].target)
        [row] = [d for d in self.work.deliveries(project_id=self.pid) if d["kind"] == "answer"]
        agents.deliver(self.services, row)  # the row handed over again after the restart
        self.pump()
        view = self.view("T-renewal")
        self.assertEqual((view["task"].status, view["task"].target.event), ("complete", "non_renewal_by_date"))
        self.assertEqual(view["reports"][0].answer, FEASIBLE)

    def test_immature_outcomes_block_until_they_are_known(self):
        view = self.answer(ANSWER, sample="sample:renewal_recent")
        self.assertEqual(view["task"].status, "blocked")
        [report] = view["reports"]
        self.assertIn("not known yet", report.answer)
        self.assertIn("(missing field: renewed_by_date)", report.answer)
        self.assertEqual(self.events("team.target.mismatch")[0]["verdict"], "immature")

    def test_a_table_that_records_cancellations_answers_and_keeps_the_late_feature_out(self):
        view = self.answer(CANCEL, sample="sample:renewal_with_cancellations")
        self.assertEqual(view["task"].status, "complete")
        [report] = view["reports"]
        self.assertIn("mature outcomes for the outcome confirmed (cancelled_at)", report.answer)
        limits = " ".join(report.limits)
        self.assertIn("cancellation_reason is written after it, a late feature", limits)
        self.assertIn("Excluded as inputs: renewed_by_date, cancelled_at, cancellation_reason", limits)
        self.assertIn("The outcome checked is an explicit cancellation, as sara confirmed (brief revision 1): cancelled_at records it",
                      limits)

    # ------------------------------------------------------------------ a changed outcome after feasibility
    def test_a_changed_outcome_revises_the_brief_supersedes_the_report_and_the_same_event_changes_nothing(self):
        self.answer(ANSWER)
        same = agents.confirm_outcome(self.services, self.pid, "T-renewal", person="sara",
                                      text="Not renewing by the contractual date.", initiator=self.sara)
        self.assertEqual((same["outcome"], same["changed"], same["revised"]), ("non_renewal_by_date", False, []))
        self.assertEqual(self.view("T-renewal")["task"].brief_revision, 1)

        out = agents.confirm_outcome(self.services, self.pid, "T-renewal", person="sara", text=CANCEL, initiator=self.sara)
        self.assertEqual((out["changed"], out["revised"], out["stale"]), (True, ["T-renewal"], ["T-renewal@r1"]))
        view = self.view("T-renewal")
        self.assertEqual((view["task"].status, view["task"].brief_revision), ("accepted", 2))
        self.assertEqual([r.status for r in view["reports"]], ["complete"])  # the earlier answer stands until the next one
        self.pump()
        view = self.view("T-renewal")
        self.assertEqual(view["task"].status, "blocked")
        [now] = self.current(view)
        self.assertEqual((now.revision, now.brief_revision, now.supersedes), (2, 2, "T-renewal@r1"))
        self.assertIn("a record of an explicit cancellation", now.answer)
        replies = [d["item_id"] for d in self.work.deliveries(project_id=self.pid) if d["kind"] == "reply"]
        self.assertEqual(replies, ["T-renewal@r1", "T-renewal@r2"])  # Business hears the revised answer once
        confirmed = self.events("team.target.confirmed")
        self.assertEqual([(c["event"], c["changed"]) for c in confirmed],
                         [("non_renewal_by_date", False), ("explicit_cancellation", True)])
        self.assertEqual((confirmed[-1]["previous"], confirmed[-1]["stale"]), ("non_renewal_by_date", 1))


for _name in [n for n in dir(_Contract) if n.startswith("test")]:  # the W5 walkthroughs run in their own file, not again here
    setattr(_Target, _name, None)


class FileTeamTargetTests(_Target, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}


class PgTeamTargetTests(_Target, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}


if __name__ == "__main__":
    unittest.main()
