"""W4 of the collaborative workspace (2026-11-20): what travels between teams is delivered by a job, once, in order.

The renewal round trip delivered end to end (the reply posted to the Business thread once, the history reading like the
owner's example), a repeated delivery adding nothing, a failure counted with its next time and retried by the relay
(never by sleeping in a worker), a hung recipient cut off, a reason that never carries an error's text, a task's rows
delivered oldest first, a restart delivering what was saved before it, a job that failed or stopped picked up again, a
revoked or demoted member's delivery refused and left out of the history, and another workspace queueing nothing. On
files and on PostgreSQL (``tests/run_parallel.py --pg``; never run directly: that empties ``dclab_test``).
"""

import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402

from dclab_core import team_work as tw  # noqa: E402
from dclab_rnd import context, experience  # noqa: E402
from dclab_rnd.jobs import Worker, open_jobs  # noqa: E402
from dclab_rnd.jobs import handlers  # noqa: E402,F401 — registers the kinds, team_delivery among them
from dclab_rnd.memory.scope import create_problem  # noqa: E402
from dclab_rnd.storage import open_stores, open_team_work  # noqa: E402
from dclab_rnd.teamspace import delivery, history_lines  # noqa: E402

T0, T1 = "2026-11-20T18:00:00+00:00", "2026-11-20T18:01:00+00:00"
DS = tw.Actor(kind="agent", id="data_science", team="data_science")
CLOSED = tw.NextStep(owner="business", unblocks="Business reads the finding")


class Recipient:
    """A scripted recipient (W5 brings the agents): fails ``fail`` times, then takes each row and says so."""

    def __init__(self):
        self.fail, self.error, self.got, self.calls, self.sleep = 0, delivery.TemporaryFailure, [], 0, 0.0

    def __call__(self, env, row):
        self.calls += 1
        if self.sleep:
            time.sleep(self.sleep)
        if self.fail:
            self.fail -= 1
            raise self.error("the agent is busy")
        self.got.append((row["id"], row["kind"], row["to_team"], row.get("thread_id")))
        return {"seen": row["id"]}


def wait_for(fn, seconds=20.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timed out")


class _Contract:
    env: dict = {}

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"DCLAB_AUTH": "", "DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS": "0.2",
                                               "DCLAB_TEAM_DELIVERY_ATTEMPTS": "3", **self.env})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.home = Path(tempfile.mkdtemp()) / "workspace"
        self.home.mkdir()
        self.services = self.make_env(self.home)
        self.work = self.services.teamwork
        self.pid = self.services.projects.create("Renewal", goal="Who should Sales call")["id"]
        self.ws = self.work.workspace
        self.prob = create_problem(self.services.projects, self.pid, "Renewal", workspace_id=self.ws)["problem_id"]
        self.sara = {"user_id": "sara", "role": "data_scientist", "workspace_id": self.ws}
        self.recipient = Recipient()
        previous = delivery.set_target(self.recipient)
        self.addCleanup(delivery.set_target, previous)
        previous_checker = delivery.set_membership_checker(None)
        self.addCleanup(delivery.set_membership_checker, previous_checker)
        token = context.set_services(self.services)
        self.addCleanup(context.reset_services, token)

    @staticmethod
    def make_env(home):
        from dclab_rnd.storage import db

        projects, drafts, _ = open_stores(home)
        wid = db.workspace_for(home) if db.database_url(required=False) else None
        return SimpleNamespace(store=SimpleNamespace(home=home), projects=projects, drafts=drafts, workspace_id=wid,
                               experience=experience.open_log(home, wid), job_store=open_jobs(home),
                               teamwork=open_team_work(home, projects=projects))

    def task(self, id="T-ds", member="sara", question="Can we rank accounts seven days before renewal?"):
        return tw.TeamTask(id=id, workspace_id=self.ws, project_id=self.pid, problem_id=self.prob, brief_revision=1,
                           requester=tw.Requester(team="business", member=member, thread_id="TH-biz"), owner_team="data_science",
                           question=question, expected_output="feasibility", done_criteria=["says whether the data can rank accounts"],
                           created_at=T0)

    def run_job(self, job, env=None):
        env = env or self.services
        ended = Worker(env.job_store, env, threads=0).run_here(job, timeout=30)
        return ended

    def deliver(self, written):
        """Queue the saved row's task and run its job here, as a worker would; the row as it ends."""
        job = delivery.send(self.services, written)
        self.assertIsNotNone(job)
        self.assertEqual(self.run_job(job)["status"], "done")
        return self.work.delivery(written.delivery["id"])

    def relay(self, env=None):
        """What a worker's recovery tick does: queue every task whose next row is due, and run those jobs here."""
        env = env or self.services
        for key in delivery.requeue_undelivered(env):
            job = env.job_store.active(delivery.KIND, key)
            if job is not None:
                self.run_job(job, env)

    def finished_task(self, id="T-ds"):
        written = self.work.create_task(self.task(id=id), initiator=self.sara)
        self.work.move(self.pid, id, "accepted", DS)
        a = tw.TeamAction(task_id=id, action_id="A1", actor=DS, action="inspect_schema", source="tool:inspect_schema",
                          input_revision="brief-1", status="running", started_at=T0, retry_key=tw.retry_key(id, "inspect_schema", "brief-1"))
        self.work.record_action(self.pid, a)
        self.work.finish_action(self.pid, id, "A1", "succeeded", ended_at=T1, output_ref="schema:v3")
        self.work.move(self.pid, id, "complete", DS)
        return written

    # ------------------------------------------------------------------ tests
    def test_the_renewal_round_trip_is_delivered_and_the_answer_posted_once(self):
        rows = [self.deliver(self.work.create_task(self.task(), initiator=self.sara))]
        self.work.move(self.pid, "T-ds", "accepted", DS)
        rows.append(self.deliver(self.work.move(self.pid, "T-ds", "waiting_for_person", DS,
                                                note="Does churn mean cancellation or failure to renew by the contractual date?")))
        rows.append(self.deliver(self.work.move(self.pid, "T-ds", "accepted", DS, note="Failure to renew by the contractual renewal date.")))
        a = tw.TeamAction(task_id="T-ds", action_id="A1", actor=DS, action="inspect_schema", source="tool:inspect_schema",
                          input_revision="brief-1", status="running", started_at=T0, retry_key=tw.retry_key("T-ds", "inspect_schema", "brief-1"))
        self.work.record_action(self.pid, a)
        self.work.finish_action(self.pid, "T-ds", "A1", "succeeded", ended_at=T1, output_ref="schema:v3")
        self.work.move(self.pid, "T-ds", "complete", DS)
        reply = self.work.store_report(self.pid, "T-ds", current_brief_revision=1, next_task=CLOSED,
                                       answer="Yes: the data can rank accounts seven days before renewal.")
        rows.append(self.deliver(reply))
        self.assertEqual([r["status"] for r in rows], ["delivered"] * 4)
        self.assertEqual([r["kind"] for r in rows], ["task", "clarification", "answer", "reply"])
        self.assertEqual((rows[3]["posted"]["thread_id"], rows[1]["posted"]["thread_id"], rows[0]["posted"]), ("TH-biz", "TH-biz", None))
        self.assertEqual([(kind, to) for _, kind, to, _ in self.recipient.got],
                         [("task", "data_science"), ("clarification", "business"), ("answer", "data_science"), ("reply", "business")])
        self.assertEqual(history_lines(self.work.handoff_history(self.pid, self.prob)), [
            "1. Business → Data Science · Can we rank accounts seven days before renewal?",
            "2. Data Science → Business · Does churn mean cancellation or failure to renew by the contractual date?",
            "3. Business → Data Science · Failure to renew by the contractual renewal date.",
            "4. Data Science → Business · Yes: the data can rank accounts seven days before renewal."])

        # the same reply again: stored once, delivered once, posted once
        again = self.work.store_report(self.pid, "T-ds", current_brief_revision=1, answer="again", next_task=CLOSED)
        self.assertIsNone(delivery.send(self.services, again))  # nothing pending: no job
        delivery.run(self.services, {"project_id": self.pid, "task_id": "T-ds"})  # even a stray job adds nothing
        self.relay()
        self.assertEqual(self.recipient.calls, 4)
        self.assertEqual(self.work.delivery(reply.delivery["id"])["posted"], rows[3]["posted"])
        self.assertEqual(len([d for d in self.work.deliveries() if d["kind"] == "reply"]), 1)
        whats = [e.payload.get("what") for e in self.services.experience.read(kinds=["team.handoff"])]
        for what in ("task.created", "task.delivered", "clarification.delivered", "answer.delivered", "reply.delivered", "action.finished",
                     "report.stored"):
            self.assertEqual(whats.count(what), 1, what)

    def test_a_failure_is_counted_with_its_next_time_and_the_relay_retries_it_when_due(self):
        self.recipient.fail = 2
        written = self.work.create_task(self.task(), initiator=self.sara)
        first = delivery.send(self.services, written)
        self.assertEqual(delivery.send(self.services, written)["id"], first["id"])  # one active job per task
        began = time.monotonic()
        self.assertEqual(self.run_job(first)["status"], "done")
        self.assertLess(time.monotonic() - began, 5)  # the job ends at the failure: nothing waits in the worker
        row = self.work.delivery(written.delivery["id"])
        wait = (datetime.fromisoformat(row["next_attempt_at"]) - datetime.fromisoformat(row["failed_at"])).total_seconds()
        self.assertEqual((row["status"], row["attempts"]), ("pending", 1))
        self.assertAlmostEqual(wait, 0.2, delta=0.002)  # the first wait: the backoff, from the recorded failure (a busy machine cannot shrink it)
        if (datetime.now(timezone.utc) - datetime.fromisoformat(row["failed_at"])).total_seconds() < 0.15:  # a busy machine may be past it
            self.assertEqual(delivery.requeue_undelivered(self.services), [])  # not due yet: the relay leaves it

        restarted = self.make_env(self.home)  # attempts and the next time survive a restart
        held = restarted.teamwork.delivery(written.delivery["id"])
        self.assertEqual((held["attempts"], held["next_attempt_at"]), (1, row["next_attempt_at"]))
        time.sleep(0.25)
        self.relay(restarted)
        row = self.work.delivery(written.delivery["id"])
        wait = (datetime.fromisoformat(row["next_attempt_at"]) - datetime.fromisoformat(row["failed_at"])).total_seconds()
        self.assertEqual((row["status"], row["attempts"]), ("pending", 2))
        self.assertAlmostEqual(wait, 0.4, delta=0.002)  # doubled
        time.sleep(0.45)
        self.relay(restarted)
        row = self.work.delivery(written.delivery["id"])
        self.assertEqual((row["status"], row["attempts"], len(self.recipient.got)), ("delivered", 2, 1))  # two failures, one delivery

    def test_the_limit_blocks_with_a_reason_that_carries_no_error_text(self):
        def leaky(env, row):
            raise RuntimeError("(psycopg.OperationalError) connection to server at 'db.internal' user=dclab password=s3cret "
                               "[SQL: INSERT ...] [parameters: {'customer_email': 'jane@example.com'}]")
        delivery.set_target(leaky)
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS": "0"}):
            stuck = self.work.create_task(self.task(), initiator=self.sara)
            for _ in range(3):
                self.relay()
        row = self.work.delivery(stuck.delivery["id"])
        self.assertEqual((row["status"], row["attempts"]), ("blocked", 3))
        self.assertEqual(row["reason"], "Not delivered to Data Science after 3 attempts (RuntimeError).")
        for leak in ("s3cret", "jane@example.com", "db.internal", "psycopg"):
            self.assertNotIn(leak, row["reason"])  # the requester sees fixed words and the class
        for leak in ("s3cret", "jane@example.com"):
            self.assertNotIn(leak, row["last_error"])  # kept for the server, scrubbed
        self.assertIn("password=[SECRET]", row["last_error"])
        self.assertEqual(self.work.task_view(self.pid, "T-ds")["deliveries"][0]["reason"], row["reason"])
        self.assertEqual(delivery.requeue_undelivered(self.services), [])  # a blocked row is not retried
        self.assertEqual(history_lines(self.work.handoff_history(self.pid, self.prob)),
                         ["1. Business → Data Science · Can we rank accounts seven days before renewal? · not delivered: blocked"])

        delivery.set_target(self.recipient)
        self.recipient.fail, self.recipient.error = 1, delivery.PermanentFailure
        refused = self.work.create_task(self.task(id="T-perm"), initiator=self.sara)
        row = self.deliver(refused)
        self.assertEqual((row["status"], row["attempts"], row["reason"]),
                         ("blocked", 0, "Not delivered: Data Science cannot take it (PermanentFailure)."))

    def test_a_hung_recipient_is_cut_off(self):
        self.recipient.sleep = 3.0  # far beyond the 0.1 s limit, so a slow machine still shows the cut-off
        with mock.patch.dict(os.environ, {"DCLAB_TEAM_DELIVERY_TIMEOUT_SECONDS": "0.1"}):
            written = self.work.create_task(self.task(), initiator=self.sara)
            began = time.monotonic()
            row = self.deliver(written)
        self.assertLess(time.monotonic() - began, 2.0)  # cut off near 0.1 s, never the recipient's 3 s: a busy machine gets the margin between
        self.assertEqual((row["status"], row["attempts"]), ("pending", 1))
        self.assertTrue(row["last_error"].startswith("TargetTimeout"))

    def test_a_tasks_rows_are_delivered_oldest_first(self):
        # a wait of a second (not 0.2 s): on a loaded machine r1 must still be waiting when r2's job runs (the gate of 2026-11-21)
        patcher = mock.patch.dict(os.environ, {"DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        written = self.finished_task()
        self.deliver(written)  # the task itself
        r1 = self.work.store_report(self.pid, "T-ds", current_brief_revision=1, answer="Eligible accounts: 1,240.", next_task=CLOSED)
        self.recipient.fail = 1
        self.assertEqual(self.deliver(r1)["status"], "pending")  # r1 failed once: waiting for its next attempt
        r2 = self.work.supersede_report(self.pid, "T-ds", current_brief_revision=1, answer="Eligible accounts: 1,180 after exclusions.",
                                        next_task=CLOSED)
        self.deliver(r2)  # r2's job runs now, and stops behind r1
        self.assertEqual(self.work.delivery(r2.delivery["id"])["status"], "pending")
        self.assertEqual([kind for _, kind, _, _ in self.recipient.got], ["task"])
        time.sleep(1.1)
        self.relay()
        replies = [rid for rid, kind, _, _ in self.recipient.got if kind == "reply"]
        self.assertEqual(replies, [r1.delivery["id"], r2.delivery["id"]])  # the thread ends on r2, the correction
        posted = sorted((self.work.delivery(d)["posted"]["at"], d) for d in replies)
        self.assertEqual(posted[-1][1], r2.delivery["id"])

    def test_a_restart_delivers_what_was_saved_before_it_once(self):
        saved = self.work.create_task(self.task(), initiator=self.sara)  # saved, then the server stopped: no job, no worker
        self.assertEqual(self.services.job_store.list(kind=delivery.KIND), [])
        after = self.make_env(self.home)  # the process starts again
        worker = Worker(after.job_store, after, threads=1, poll=0.05, beat=0.05).start()
        try:
            row = wait_for(lambda: (r := after.teamwork.delivery(saved.delivery["id"])) and r["status"] == "delivered" and r)
        finally:
            worker.stop()
        self.assertEqual((len(self.recipient.got), row["attempts"]), (1, 0))
        self.assertEqual(Worker(after.job_store, after, threads=0).requeue_deliveries(), [])  # a second start finds nothing
        self.assertEqual(len(self.recipient.got), 1)

        # a worker that died holding a delivery: the recovery marks its job interrupted, and the relay queues the task again
        lost = self.work.create_task(self.task(id="T-lost"), initiator=self.sara)
        job = delivery.send(self.services, lost)
        self.services.job_store.claim("deadhost:1:abc", job_id=job["id"])
        self.services.job_store.interrupt(job["id"], "deadhost:1:abc", "gone")
        self.relay()
        self.assertEqual(self.work.delivery(lost.delivery["id"])["status"], "delivered")
        self.assertEqual(len(self.recipient.got), 2)

    def test_a_job_that_failed_is_counted_and_picked_up_by_the_running_workers_relay(self):
        real_authorize, real_done = delivery.authorize, self.work.delivery_done
        calls = {"authorize": 0, "done": 0}

        def flaky_authorize(env, row):  # the accounts database blips once, inside the job
            calls["authorize"] += 1
            if calls["authorize"] == 1:
                raise RuntimeError("could not connect to server")
            return real_authorize(env, row)

        def flaky_done(delivery_id, *a, **kw):  # the store blips once after the recipient took T-done's row
            if self.work.delivery(delivery_id)["task_id"] == "T-done":
                calls["done"] += 1
                if calls["done"] == 1:
                    raise RuntimeError("the store is away")
            return real_done(delivery_id, *a, **kw)

        with mock.patch.dict(os.environ, {"DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS": "0.05"}), \
                mock.patch.object(delivery, "authorize", flaky_authorize), mock.patch.object(self.work, "delivery_done", flaky_done):
            worker = Worker(self.services.job_store, self.services, threads=1, poll=0.05, beat=0.05, stale=0.3).start()
            try:
                one = self.work.create_task(self.task(id="T-auth"), initiator=self.sara)
                delivery.send(self.services, one)
                row = wait_for(lambda: (r := self.work.delivery(one.delivery["id"])) and r["status"] == "delivered" and r)
                self.assertEqual(row["attempts"], 1)  # the failed job counted an attempt (settle), then the relay delivered it
                two = self.work.create_task(self.task(id="T-done"), initiator=self.sara)
                delivery.send(self.services, two)
                row = wait_for(lambda: (r := self.work.delivery(two.delivery["id"])) and r["status"] == "delivered" and r)
            finally:
                worker.stop()
        self.assertEqual(row["attempts"], 1)
        self.assertEqual([rid for rid, _, _, _ in self.recipient.got].count(two.delivery["id"]), 2)  # at least once at the target
        self.assertIsNone(row["posted"])  # a task row goes to the owner's agent: no thread to post to

    def test_a_revoked_or_demoted_members_delivery_is_refused_and_left_out_of_the_history(self):
        roles = {"omar": None, "vera": "viewer"}
        delivery.set_membership_checker(lambda uid, wid: roles.get(uid, "data_scientist"))
        for member in ("omar", "vera"):
            who = {"user_id": member, "role": "data_scientist", "workspace_id": self.ws}
            written = self.work.create_task(self.task(id=f"T-{member}", member=member, question=f"{member.title()}'s question?"),
                                            initiator=who)
            job = delivery.send(self.services, written)
            self.assertEqual(job["user"]["user_id"], member)  # the job acts for the member who started the work
            self.run_job(job)
            row = self.work.delivery(written.delivery["id"])
            self.assertEqual((row["status"], row["reason"]), ("refused", delivery.REFUSED))  # fixed words: no workspace id
            self.assertNotIn(str(self.ws), row["reason"])
        self.assertEqual(self.recipient.calls, 0)  # never handed over
        self.assertEqual(delivery.requeue_undelivered(self.services), [])
        ok = self.work.create_task(self.task(id="T-sara"), initiator=self.sara)
        self.assertEqual(self.deliver(ok)["status"], "delivered")
        self.assertEqual(history_lines(self.work.handoff_history(self.pid, self.prob)),
                         ["1. Business → Data Science · Can we rank accounts seven days before renewal?"])  # the refused never left

        # with sign-in on, the worker itself refuses the job; settling it refuses the row with the same words
        with mock.patch.dict(os.environ, {"DCLAB_AUTH": "password"}):
            late = self.work.create_task(self.task(id="T-late", member="omar"),
                                         initiator={"user_id": "omar", "role": "data_scientist", "workspace_id": self.ws})
            ended = self.run_job(delivery.send(self.services, late))
        row = self.work.delivery(late.delivery["id"])
        self.assertEqual(row["status"], "refused", ended.get("error"))
        self.assertEqual(self.recipient.calls, 1)

    def test_another_workspace_queues_and_reads_nothing(self):
        self.work.create_task(self.task(), initiator=self.sara)
        other_home = Path(tempfile.mkdtemp()) / "workspace-b"
        other_home.mkdir()
        other = self.make_env(other_home)
        self.assertEqual(delivery.requeue_undelivered(other), [])
        self.assertEqual(other.teamwork.deliveries(), [])
        self.assertEqual(other.job_store.list(kind=delivery.KIND), [])
        self.assertEqual(delivery.requeue_undelivered(SimpleNamespace()), [])  # no workspace at all: quiet


class FileDeliveryTests(_Contract, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}


class PgDeliveryTests(_Contract, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}


if __name__ == "__main__":
    unittest.main()
