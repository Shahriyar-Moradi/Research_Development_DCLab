"""Durable jobs (package 10.3): the job table on files and PostgreSQL, the worker (claims, checkpoints, stop,
heartbeat, recovery), and the server: a stage run, an intern turn and a data pipeline are jobs that a person can stop
from the Compute page, and a job a dead worker left running is interrupted on start and can be retried."""
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pgtest  # noqa: E402
from dclab_rnd.jobs import ActiveJob, FileJobs, PgJobs, Stopped, Worker, checkpoint, progress  # noqa: E402
from dclab_rnd.jobs import worker as jw  # noqa: E402


def wait_for(fn, seconds=20.0, every=0.05):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(every)
    raise AssertionError("timed out")


class StoreContract:
    def store(self):
        raise NotImplementedError

    def setUp(self):
        self.jobs = self.store()

    def test_one_active_job_per_key_and_claims_in_order(self):
        a = self.jobs.enqueue("stage", "p1", {"n": 1})
        b = self.jobs.enqueue("stage", "p2", {"n": 2})
        with self.assertRaises(ActiveJob) as held:
            self.jobs.enqueue("stage", "p1", {"n": 3})
        self.assertEqual(held.exception.job["id"], a["id"])
        self.jobs.enqueue("intern", "p1", {})  # another kind may hold the same key
        self.assertEqual(self.jobs.claim("w1")["id"], a["id"])  # oldest first
        self.assertEqual(self.jobs.claim("w1", job_id=b["id"])["id"], b["id"])
        got = self.jobs.get(a["id"])
        self.assertEqual((got["status"], got["worker"], got["attempts"]), ("running", "w1", 1))
        self.assertIsNone(self.jobs.claim("w2", job_id=a["id"]))  # a running job is never claimed again
        self.assertEqual(self.jobs.finish(a["id"], "done")["status"], "done")
        self.assertEqual(self.jobs.finish(a["id"], "failed")["status"], "done")  # the first ending stands
        self.jobs.enqueue("stage", "p1", {"n": 4})  # the key is free again
        with self.assertRaises(KeyError):
            self.jobs.get("j000000000000")

    def test_cancel_retry_and_interrupt(self):
        queued = self.jobs.enqueue("pipeline", "d:a1", {})
        self.assertEqual(self.jobs.cancel(queued["id"])["status"], "cancelled")
        running = self.jobs.enqueue("pipeline", "d:a2", {}, worker="w1")  # claimed at once (?wait=true)
        self.assertEqual(running["status"], "running")
        asked = self.jobs.cancel(running["id"])
        self.assertEqual((asked["status"], asked["cancel_requested"]), ("running", True))
        with self.assertRaises(ValueError):
            self.jobs.retry(running["id"])  # still running
        again = self.jobs.retry(queued["id"])
        self.assertEqual((again["status"], again["cancel_requested"], again["error"]), ("queued", False, None))
        self.assertEqual(self.jobs.claim("w2", job_id=queued["id"])["attempts"], 1)
        # only the worker that holds a job may be found stale and interrupted, once
        self.jobs._change(running["id"], lambda j: j.update(heartbeat=0) or True)
        self.assertIn(running["id"], [j["id"] for j in self.jobs.stale(30)])
        self.assertIsNone(self.jobs.interrupt(running["id"], "someone-else", "x"))
        self.assertEqual(self.jobs.interrupt(running["id"], "w1", "gone")["status"], "interrupted")
        self.assertIsNone(self.jobs.interrupt(running["id"], "w1", "gone"))
        self.assertEqual(self.jobs.retry(running["id"])["status"], "queued")
        self.jobs.enqueue("pipeline", "d:a3", {})
        blocker = self.jobs.enqueue("stage", "p9", {})
        self.jobs.claim("w1", job_id=blocker["id"])
        self.jobs.finish(blocker["id"], "failed", "boom")
        self.jobs.enqueue("stage", "p9", {})
        with self.assertRaises(ActiveJob):  # a retry may not take a key another job holds now
            self.jobs.retry(blocker["id"])


class FileStoreTests(StoreContract, unittest.TestCase):
    def store(self):
        return FileJobs(Path(tempfile.mkdtemp()) / "jobs.json")


class PgStoreTests(StoreContract, unittest.TestCase):
    def store(self):
        from dclab_rnd.storage import db

        url = pgtest.require()
        return PgJobs(db.workspace_for(Path(tempfile.mkdtemp()), url), url)

    def test_two_workers_never_run_the_same_job(self):
        ran, lock = [], threading.Lock()

        def run(env, payload):
            time.sleep(0.05)
            with lock:
                ran.append(payload["n"])
        jw.handler("t-share", run, lambda *a: None)
        self.addCleanup(jw.HANDLERS.pop, "t-share", None)
        ids = [self.jobs.enqueue("t-share", f"k{n}", {"n": n})["id"] for n in range(12)]
        workers = [Worker(self.jobs, None, threads=3, poll=0.05).start() for _ in range(2)]
        try:
            wait_for(lambda: all(self.jobs.get(i)["status"] == "done" for i in ids))
        finally:
            for w in workers:
                w.stop()
        self.assertEqual(sorted(ran), list(range(12)))  # each once
        self.assertEqual({self.jobs.get(i)["attempts"] for i in ids}, {1})


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.jobs = FileJobs(Path(tempfile.mkdtemp()) / "jobs.json")
        self.settled = []
        self.gate = threading.Event()

        def steps(env, payload):
            for n in range(payload.get("steps", 3)):
                checkpoint()
                progress(step=n + 1)
                if payload.get("hold") and n == 0:
                    self.gate.wait(10)
            if payload.get("fail"):
                raise ValueError("the data is broken")
        jw.handler("t-steps", steps, lambda env, payload, reason, job: self.settled.append(reason))
        self.addCleanup(jw.HANDLERS.pop, "t-steps", None)
        self.worker = Worker(self.jobs, None, threads=2, poll=0.05, beat=0.05, stale=0.5).start()
        self.addCleanup(self.worker.stop, 1.0)

    def ended(self, job_id):
        return wait_for(lambda: (j := self.jobs.get(job_id))["status"] not in ("queued", "running") and j)

    def test_a_job_runs_records_progress_and_ends(self):
        job = self.jobs.enqueue("t-steps", "a", {"steps": 3})
        self.worker.notify()
        done = self.ended(job["id"])
        self.assertEqual((done["status"], done["progress"], done["attempts"]), ("done", {"step": 3}, 1))
        self.assertEqual(self.settled, [])

    def test_a_failure_is_recorded_and_settled(self):
        job = self.jobs.enqueue("t-steps", "b", {"fail": True})
        self.worker.notify()
        done = self.ended(job["id"])
        self.assertEqual(done["status"], "failed")
        self.assertIn("the data is broken", done["error"])
        self.assertEqual(self.settled, ["failed"])

    def test_a_stop_takes_effect_at_the_next_checkpoint(self):
        job = self.jobs.enqueue("t-steps", "c", {"hold": True, "steps": 5})
        self.worker.notify()
        wait_for(lambda: self.jobs.get(job["id"])["progress"].get("step") == 1)
        self.jobs.cancel(job["id"])
        self.gate.set()
        done = self.ended(job["id"])
        self.assertEqual((done["status"], done["progress"]), ("cancelled", {"step": 1}))  # step 1 finished; 2 never started
        self.assertEqual(self.settled, ["stopped"])

    def test_a_stopping_worker_interrupts_and_another_recovers_a_dead_one(self):
        job = self.jobs.enqueue("t-steps", "d", {"hold": True})
        self.worker.notify()
        wait_for(lambda: self.jobs.get(job["id"])["progress"].get("step") == 1)
        stopping = threading.Thread(target=self.worker.stop, args=(5.0,))
        stopping.start()
        self.gate.set()
        stopping.join()
        self.assertEqual(self.jobs.get(job["id"])["status"], "interrupted")
        # a job claimed by a worker that died (no heartbeat): another worker marks it interrupted and settles it
        dead = self.jobs.enqueue("t-steps", "e", {}, worker="dead-host:1:x")
        self.jobs._change(dead["id"], lambda j: j.update(heartbeat=time.time() - 60) or True)
        other = Worker(self.jobs, None, threads=1, poll=0.05, stale=30).start()
        self.addCleanup(other.stop, 1.0)
        self.assertEqual(self.jobs.get(dead["id"])["status"], "interrupted")
        self.assertEqual(self.settled[-1], "interrupted")

    def test_a_live_worker_whose_job_was_taken_stops_at_its_next_checkpoint(self):
        job = self.jobs.enqueue("t-steps", "f", {"hold": True, "steps": 5})
        self.worker.notify()
        wait_for(lambda: self.jobs.get(job["id"])["progress"].get("step") == 1)
        # a beat that landed since the job was found stale wins: nothing is interrupted
        self.assertIsNone(self.jobs.interrupt(job["id"], self.worker.id, "x", older_than=30))
        # another worker marked it interrupted (this one looked dead) and retried it elsewhere: this one stops, and writes nothing
        self.jobs.interrupt(job["id"], self.worker.id, "x")
        self.jobs.retry(job["id"])
        self.jobs.claim("other-worker", job_id=job["id"])
        self.gate.set()
        wait_for(lambda: job["id"] not in self.worker.running)
        row = self.jobs.get(job["id"])
        self.assertEqual((row["status"], row["worker"], row["progress"]), ("running", "other-worker", {}))
        self.assertEqual(self.settled, [])  # the displaced worker settles nothing: the run that took over is left alone

    def test_a_job_of_a_dead_process_on_this_machine_is_recovered_at_once(self):
        import socket
        import subprocess

        gone = subprocess.Popen([sys.executable, "-c", "pass"])
        gone.wait()
        job = self.jobs.enqueue("t-steps", "g", {}, worker=f"{socket.gethostname()}:{gone.pid}:x")
        self.assertEqual(self.worker.recover(), [])  # a fresh heartbeat: maybe a container with the same host name
        self.jobs._change(job["id"], lambda j: j.update(heartbeat=time.time() - 20) or True)  # a few beats missed, far from 120 s
        self.assertEqual([j["id"] for j in self.worker.recover()], [job["id"]])
        self.assertEqual(self.jobs.get(job["id"])["status"], "interrupted")
        alive = self.jobs.enqueue("t-steps", "h", {}, worker=f"{socket.gethostname()}:{os.getppid()}:x")
        self.worker.recover()
        self.assertEqual(self.jobs.get(alive["id"])["status"], "running")  # a live process's job is left alone

    def test_a_stage_job_taken_over_during_its_last_stage_leaves_the_project_alone(self):
        from dclab_rnd.jobs import handlers

        class Env:
            projects = mock.Mock()
        p = {"stages": {"final": {"status": "running"}}, "running": "final"}
        Env.projects.get.return_value = p
        job = self.jobs.enqueue("stage", "p1", {"project_id": "p1", "stages": ["final"]}, worker=self.worker.id)

        def last_stage(projects, pid, stage, reuse_reason=None):
            self.jobs.interrupt(job["id"], self.worker.id, "x")  # found stale while it ran its last stage, and retried elsewhere
            self.jobs.retry(job["id"])
            self.jobs.claim("other-worker", job_id=job["id"])
        with mock.patch("dclab_rnd.studio.engine.execute", side_effect=last_stage):
            self.worker.env = Env
            self.worker.execute(job)
        Env.projects.save.assert_not_called()  # the run that took over owns the project now
        self.assertEqual(self.jobs.get(job["id"])["worker"], "other-worker")
        self.assertIs(jw.HANDLERS["stage"].run, handlers._stage_run)  # the real stage handler ran

    def test_checkpoint_outside_a_job_does_nothing(self):
        checkpoint()
        progress(step=1)
        self.assertTrue(issubclass(Stopped, BaseException) and not issubclass(Stopped, Exception))


class ServerTests(unittest.TestCase):
    """The server's jobs: Stop on the Compute page, and recovery after a server that died mid-run."""

    def setUp(self):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
            from test_studio import CHURN_CONTRACT, churn_frame, new_project
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        self.TestClient, self.create_app = TestClient, create_app
        self.home = Path(tempfile.mkdtemp())
        self.client = TestClient(create_app(self.home))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.s = self.client.app.state.services
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}
        self.pid = new_project(self.s.projects, churn_frame(300), CHURN_CONTRACT)["id"]
        self.gate, self.ran = threading.Event(), []

    def fake_execute(self, hold_first=True):
        def execute(projects, pid, stage, reuse_reason=None, **_):
            self.ran.append(stage)
            if hold_first and len(self.ran) == 1:
                self.gate.wait(10)
            p = projects.get(pid)
            p["stages"][stage] = {"status": "completed"}
            projects.save(p)
        return mock.patch("dclab_rnd.studio.engine.execute", side_effect=execute)

    def job(self, job_id):
        return self.client.get(f"/api/jobs/{job_id}").json()

    def test_stop_on_the_compute_page_ends_a_stage_run_between_stages(self):
        with self.fake_execute():
            r = self.client.post(f"/api/projects/{self.pid}/run", headers=self.h)
            self.assertEqual(r.status_code, 200)
            wait_for(lambda: self.ran)
            self.assertEqual(self.client.post(f"/api/projects/{self.pid}/run", headers=self.h).status_code, 409)  # one run per project
            listing = self.client.get("/api/ops/jobs").json()["jobs"]
            queued = next(j for j in listing if j["kind"] == "stage" and j["status"] == "queued")
            detail = self.client.get(f"/api/ops/jobs/{queued['id']}").json()
            self.assertIsNotNone(detail["stop"])
            self.assertEqual(self.client.post(detail["stop"]["href"], headers=self.h).status_code, 200)
            self.gate.set()
            job_id = detail["stop"]["job_id"]
            wait_for(lambda: self.job(job_id)["status"] == "cancelled")
        p = self.s.projects.get(self.pid)
        self.assertEqual(self.ran, ["data"])  # the stage that started finished; the next never started
        self.assertEqual({k: v["status"] for k, v in p["stages"].items()}, {"data": "completed", **dict.fromkeys(("leakage", "features", "models", "final"), "pending")})
        self.assertIsNone(p["running"])
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/stop", headers=self.h).status_code, 409)  # it has ended
        self.assertEqual(self.client.get("/api/jobs/jnope").status_code, 404)

    def test_a_run_a_dead_server_left_is_interrupted_on_start_and_retried(self):
        payload = {"project_id": self.pid, "stages": ["data", "leakage"], "reuse_reason": None}
        self.s.mark_queued(self.pid, payload["stages"])
        p = self.s.projects.get(self.pid)
        p["stages"]["data"] = {"status": "running", "started": "2026-10-06T10:00:00+00:00"}
        self.s.projects.save(p)
        dead = self.s.job_store.enqueue("stage", self.pid, payload, worker="killed-server:1:x")
        self.s.job_store._change(dead["id"], lambda j: j.update(heartbeat=time.time() - 3600) or True)
        with self.TestClient(self.create_app(self.home)) as restarted:  # the server starts again on the same workspace
            s2 = restarted.app.state.services
            self.assertEqual(s2.job_store.get(dead["id"])["status"], "interrupted")
            p = s2.projects.get(self.pid)
            self.assertEqual((p["stages"]["data"]["status"], p["stages"]["leakage"]["status"], p["running"]), ("failed", "pending", None))
            self.assertIn("Interrupted", p["stages"]["data"]["error"])
            from dclab_rnd.agentic.pages import ops

            controls = ops.Jobs(restarted.app.state.services and _ctx(s2))._controls(
                {"kind": "stage", "project_id": self.pid, "stage": "data", "status": "failed", "current": True, "id": f"stage-{self.pid}-data-1"})
            self.assertEqual(controls[1]["job_id"], dead["id"])  # the Compute page offers Retry
            with self.fake_execute(hold_first=False):
                r = restarted.post(f"/api/jobs/{dead['id']}/retry", headers={"X-DCLab-Token": restarted.get("/api/config").json()["csrf"]})
                self.assertEqual(r.status_code, 200, r.text)
                wait_for(lambda: s2.job_store.get(dead["id"])["status"] == "done")
            job = s2.job_store.get(dead["id"])
            self.assertEqual(job["attempts"], 2)
            self.assertEqual({k: s2.projects.get(self.pid)["stages"][k]["status"] for k in ("data", "leakage")}, {"data": "completed", "leakage": "completed"})

    def test_an_intern_turn_and_a_data_pipeline_stop_too(self):
        from dclab_rnd.intern.tools import Toolbox

        real = Toolbox.call
        entered = threading.Event()

        def slow(toolbox, name, arguments):
            entered.set()
            if not self.gate.is_set():
                self.gate.wait(10)
            return real(toolbox, name, arguments)
        no_model = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"})  # the standard plan
        with no_model, mock.patch.object(Toolbox, "call", slow):
            session = self.client.post("/api/intern/sessions", json={"task": "Review the churn project and report", "project_id": self.pid}, headers=self.h).json()
            job = wait_for(lambda: self.s.job_store.active("intern", session["id"]))
            wait_for(entered.is_set)  # the first tool call is under way (a stop while queued is tested with the stage run)
            self.assertEqual(self.client.post(f"/api/jobs/{job['id']}/stop", headers=self.h).status_code, 200)
            self.gate.set()
            wait_for(lambda: self.job(job["id"])["status"] == "cancelled")
            self.assertEqual(session["mode"], "standard")
            self.assertEqual(self.s.intern_sessions.get(session["id"])["status"], "stopped")
            self.assertEqual(len(self.s.intern_sessions.get(session["id"])["steps"]), 1)  # the step that started finished; no other ran
            self.assertEqual(self.client.post(f"/api/jobs/{job['id']}/retry", headers=self.h).status_code, 200)  # a stopped turn can run again
            wait_for(lambda: self.job(job["id"])["attempts"] == 2)
            self.client.post(f"/api/jobs/{job['id']}/stop", headers=self.h)  # 409 when it already ended: either way it ends
            wait_for(lambda: self.job(job["id"])["status"] not in ("queued", "running"), seconds=120)
            self.assertNotIn("text", self.job(job["id"])["payload"])  # no free text in a job response

        self.gate.clear()
        from dclab_rnd.draft import structure

        real_table = structure.to_table

        def slow_table(*a, **k):
            self.gate.wait(10)
            return real_table(*a, **k)
        draft = self.client.post("/api/drafts", json={"problem": "Predict which customers churn next month"}, headers=self.h).json()
        with mock.patch.object(structure, "to_table", slow_table):
            asset = self.client.put(f"/api/drafts/{draft['id']}/data?filename=c.csv", content=b"a,b\n1,2\n3,4\n", headers=self.h).json()
            job = wait_for(lambda: self.s.job_store.latest("pipeline", f"{draft['id']}:{asset['id']}"))
            self.client.post(f"/api/jobs/{job['id']}/stop", headers=self.h)
            self.gate.set()
            wait_for(lambda: self.job(job["id"])["status"] == "cancelled")
        a = next(x for x in self.s.drafts.get(draft["id"])["assets"] if x["id"] == asset["id"])
        self.assertEqual((a["status"], a["error"]), ("failed", jw.TEXT["stopped"]))


class ControlTests(ServerTests):
    def test_a_synthetic_tables_pipeline_offers_stop(self):
        from dclab_rnd.agentic.pages import ops

        row = self.s.job_store.enqueue("synthetic", "d1:synthetic", {"draft_id": "d1"}, worker="w")
        stop, _ = ops.Jobs(_ctx(self.s))._controls({"kind": "data", "id": "data-d1-a1234abcd", "draft_id": "d1", "status": "running"})
        self.assertEqual(stop["job_id"], row["id"])

    def test_a_run_that_loses_the_race_puts_the_project_back(self):
        from dclab_rnd.jobs import ActiveJob

        p = self.s.projects.get(self.pid)
        p["stages"]["data"] = {"status": "completed", "finished": "t"}
        self.s.projects.save(p)
        with mock.patch.object(self.s.job_store, "enqueue", side_effect=ActiveJob(None)):  # another instance inserted first
            self.assertEqual(self.client.post(f"/api/projects/{self.pid}/run", headers=self.h).status_code, 409)
        p = self.s.projects.get(self.pid)
        self.assertEqual((p["stages"]["data"], p["stages"]["leakage"]["status"], p["running"]), ({"status": "completed", "finished": "t"}, "pending", None))

    test_stop_on_the_compute_page_ends_a_stage_run_between_stages = None
    test_a_run_a_dead_server_left_is_interrupted_on_start_and_retried = None
    test_an_intern_turn_and_a_data_pipeline_stop_too = None


def _ctx(s):
    from dclab_rnd.agentic.pages import Context

    return Context(store=s.store, projects=s.projects, drafts=s.drafts, intern_sessions=s.intern_sessions,
                   jobs=s.jobs, intern_jobs=s.intern_jobs, draft_jobs=s.draft_jobs, job_store=s.job_store)


if __name__ == "__main__":
    unittest.main()
