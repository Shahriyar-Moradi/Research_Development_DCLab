"""Rate limits and per-user quotas (package 10.6), on PostgreSQL with accounts on and small limits: a person over a
limit gets 429 saying what was exceeded and when it resets, and the others are unaffected."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pgtest  # noqa: E402

PASSWORD = "a long enough password"
LIMITS = {"DCLAB_LIMIT_REQUESTS_PER_MINUTE": "40", "DCLAB_LIMIT_WORKSPACE_REQUESTS_PER_MINUTE": "100000", "DCLAB_LIMIT_UPLOAD_MB_PER_DAY": "1",
          "DCLAB_LIMIT_JOBS_PER_USER": "1", "DCLAB_LIMIT_JOBS_PER_WORKSPACE": "50", "DCLAB_LIMIT_USER_MONTHLY_EUR": "1"}


class LimitsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            raise unittest.SkipTest(f"Studio dependencies not installed: {error}")
        url = pgtest.require()
        pgtest.empty(url)
        cls.env = mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": url, "DCLAB_AUTH": "password", "OPENAI_API_KEY": "",
                                               "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1", **LIMITS})
        cls.env.start()
        from dclab_rnd.accounts.store import Accounts
        from dclab_rnd.storage import db

        db.upgrade(url)
        cls.TestClient = TestClient
        cls.app = create_app(Path(tempfile.mkdtemp()))
        cls.main = TestClient(cls.app)
        cls.main.__enter__()
        cls.accounts = Accounts()
        cls.w = cls.app.state.services.workspace_id
        cls.users = {}
        for name in ("ann", "ben", "cat", "dan", "eve", "fay", "gus", "hal", "ivy"):
            user = cls.accounts.create_user(f"{name}-{os.getpid()}@example.com", name.title(), password=PASSWORD)
            cls.accounts.set_member(cls.w, user["id"], "data_scientist")
            cls.users[name] = user

    @classmethod
    def tearDownClass(cls):
        cls.main.__exit__(None, None, None)
        cls.env.stop()

    def sign_in(self, name):
        client = self.TestClient(self.app)
        token = client.get("/api/config").json()["csrf"]
        r = client.post("/api/auth/password", json={"email": self.users[name]["email"], "password": PASSWORD}, headers={"X-DCLab-Token": token})
        self.assertEqual(r.status_code, 200, r.text)
        client.headers["X-DCLab-Token"] = r.json()["csrf"]
        return client

    def test_requests_a_minute_and_the_others_are_unaffected(self):
        ann, ben = self.sign_in("ann"), self.sign_in("ben")
        for n in range(100):  # a fixed one-minute window: a run that crosses the minute starts counting again
            over = ann.get("/api/projects")
            if over.status_code == 429:
                break
        self.assertEqual(over.status_code, 429)
        self.assertLessEqual(n, 82)
        body = over.json()["detail"]
        self.assertEqual((over.status_code, body["limit"], body["scope"], body["allowed"]), (429, "requests", "user", 40))
        self.assertTrue(1 <= int(over.headers["Retry-After"]) <= 60)
        self.assertIn("resets at", body["message"])
        self.assertEqual(ben.get("/api/projects").status_code, 200)  # another person in the same workspace

    def test_bytes_brought_in_a_day(self):
        cat, dan = self.sign_in("cat"), self.sign_in("dan")
        pid = cat.post("/api/projects", json={"name": "Uploads", "goal": "g"}).json()["id"]
        chunk = b"a,b\n" + b"1,2\n" * 150_000  # about 0.6 MB
        first = cat.put(f"/api/projects/{pid}/data?filename=one.csv", content=chunk)
        self.assertNotEqual(first.status_code, 429, first.text[:200])
        second = cat.put(f"/api/projects/{pid}/data?filename=two.csv", content=chunk)
        self.assertEqual((second.status_code, second.json()["detail"]["limit"]), (429, "upload_bytes"))
        self.assertFalse((self.app.state.services.projects.data_dir(pid) / "two.csv").exists())  # nothing was stored
        draft = dan.post("/api/drafts", json={"problem": "Predict which customers churn next month"}).json()
        self.assertNotEqual(dan.put(f"/api/drafts/{draft['id']}/data?filename=d.csv", content=chunk).status_code, 429)  # dan's own allowance

    def test_jobs_at_once(self):
        from dclab_rnd.accounts.principal import Principal, acting

        ben = self.sign_in("ben")
        s = self.app.state.services
        user = self.users["ben"]
        with acting(Principal(role="data_scientist", workspace_id=self.w, user_id=user["id"], email=user["email"], name="Ben")):
            s.job_store.enqueue("intern", "ben-running", {"session_id": "x"}, worker="elsewhere")  # one of his, running
        refused = ben.post("/api/intern/sessions", json={"task": "Review the churn project and write a report"})
        body = refused.json()["detail"]
        self.assertEqual((refused.status_code, body["limit"], body["scope"]), (429, "jobs", "user"))
        self.assertIn("frees when one of them ends", body["message"])
        self.assertNotEqual(self.sign_in("ann").post("/api/intern/sessions", json={"task": "Review the churn project and write a report"}).status_code, 429)

    def test_model_spend_a_month(self):
        from dclab_rnd.accounts.principal import Principal, acting
        from dclab_rnd.models.usage import now

        gateway = self.app.state.services.gateway
        dan = self.users["dan"]
        as_dan = Principal(role="data_scientist", workspace_id=self.w, user_id=dan["id"], email=dan["email"], name="Dan")
        with acting(as_dan):
            gateway.usage.record({"at": now(), "purpose": "intern", "tier": "standard", "model": "m", "endpoint": "e", "outcome": "ok", "cost_eur": 0.9,
                                  "input_tokens": 10, "output_tokens": 2, "seconds": 0.5, "attempts": 1})
            refusal, release = gateway.reserve(None, None, 0.2)  # would pass his euro a month
            self.assertIn("your monthly model budget", refusal or "")
        with acting(Principal(role="data_scientist", workspace_id=self.w, user_id=self.users["cat"]["id"], via="test")):
            refusal, release = gateway.reserve(None, None, 0.2)
            self.assertIsNone(refusal)  # cat has spent nothing
            release()

    def test_the_admin_page_shows_usage_against_the_limits(self):
        quotas = self.sign_in("eve").get("/api/platform/quotas").json()  # eve has used nothing else
        self.assertTrue(quotas["on"])
        kinds = {q["kind"]: q for q in quotas["limits"]}
        self.assertEqual(set(kinds), {"requests", "upload_bytes", "jobs", "model_eur"})
        self.assertEqual(kinds["requests"]["user"]["limit"], 40)
        self.assertGreaterEqual(kinds["requests"]["user"]["used"], 1)
        self.assertEqual(kinds["upload_bytes"]["user"]["limit"], 1024 * 1024)


    def busy(self, name):
        """One job of this person's, running elsewhere: they are at the job limit of one."""
        from dclab_rnd.accounts.principal import Principal, acting

        user = self.users[name]
        with acting(Principal(role="data_scientist", workspace_id=self.w, user_id=user["id"], email=user["email"], name=name)):
            return self.app.state.services.job_store.enqueue("intern", f"{name}-running", {"session_id": "x"}, worker="elsewhere")

    def test_a_stage_run_or_a_retry_at_the_job_limit_is_refused_and_changes_nothing(self):
        fay = self.sign_in("fay")
        s = self.app.state.services
        from test_studio import CHURN_CONTRACT, churn_frame, new_project

        pid = new_project(s.projects, churn_frame(300), CHURN_CONTRACT)["id"]  # data and a solution: the workflow lets a stage run
        self.busy("fay")
        refused = fay.post(f"/api/projects/{pid}/stages/data/run", json={})
        self.assertEqual((refused.status_code, refused.json()["detail"]["limit"]), (429, "jobs"))
        p = s.projects.get(pid)
        self.assertEqual((p["running"], p["stages"]["data"]["status"]), (None, "pending"))  # not left "queued"
        old = s.job_store.enqueue("stage", f"{pid}-old", {"project_id": pid, "stages": ["data"]}, worker="w")
        s.job_store.finish(old["id"], "failed", "boom")
        self.assertEqual(fay.post(f"/api/jobs/{old['id']}/retry").status_code, 429)  # a retry counts like a new job

    def test_a_draft_upload_at_the_job_limit_stores_and_charges_nothing(self):
        from dclab_rnd import limits

        gus = self.sign_in("gus")
        draft = gus.post("/api/drafts", json={"problem": "Predict which customers churn next month"}).json()
        before = len(self.app.state.services.drafts.get(draft["id"])["assets"])
        self.busy("gus")
        refused = gus.put(f"/api/drafts/{draft['id']}/data?filename=d.csv", content=b"a,b\n1,2\n")
        self.assertEqual((refused.status_code, refused.json()["detail"]["limit"]), (429, "jobs"))
        self.assertEqual(len(self.app.state.services.drafts.get(draft["id"])["assets"]), before)
        self.assertEqual(limits.used("upload_bytes", self.users["gus"]["id"], self.w)["user"], 0)

    def test_requests_at_once_never_take_more_than_the_job_limit(self):
        import threading

        from dclab_rnd import limits
        from dclab_rnd.accounts.principal import Principal, acting

        s, user = self.app.state.services, self.users["hal"]
        taken, refused = [], []

        def queue(n):
            with acting(Principal(role="data_scientist", workspace_id=self.w, user_id=user["id"], email=user["email"], name="Hal")):
                try:
                    s.queue_job("intern", f"hal-{n}", {"session_id": "none"}, here=True)
                    taken.append(n)
                except limits.LimitExceeded:
                    refused.append(n)
        threads = [threading.Thread(target=queue, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual((len(taken), len(refused)), (1, 7))

    def test_a_refused_request_is_not_counted_and_a_person_spends_across_workspaces(self):
        from dclab_rnd import limits
        from dclab_rnd.models.usage import PgUsage, month_start, now

        ivy = self.users["ivy"]
        for _ in range(40):
            limits.charge("requests", 1, ivy["id"], self.w, now=1_000_000.0)
        with self.assertRaises(limits.LimitExceeded) as over:
            limits.charge("requests", 1, ivy["id"], self.w, now=1_000_000.0)
        self.assertIn("reached the limit of 40 requests a minute", str(over.exception))
        self.assertEqual(limits.used("requests", ivy["id"], self.w, now=1_000_000.0)["user"], 40)  # the refused one was not added
        other = self.accounts.create_workspace("Ivy's other team")
        row = {"at": now(), "purpose": "intern", "tier": "standard", "model": "m", "endpoint": "e", "outcome": "ok", "cost_eur": 0.6,
               "input_tokens": 1, "output_tokens": 1, "seconds": 0.1, "attempts": 1, "user_id": ivy["id"]}
        PgUsage(self.w).record(row)
        PgUsage(other).record(row)
        self.assertAlmostEqual(PgUsage(self.w).spent(month_start(), user_id=ivy["id"]), 1.2)  # a person's limit counts every workspace


class LimitsSettingsTests(unittest.TestCase):
    def test_a_bad_workspace_budget_does_not_break_the_limits(self):
        from dclab_rnd import limits

        with mock.patch.dict(os.environ, {"DCLAB_WORKSPACE_MONTHLY_EUR": "lots"}):
            self.assertIsNone(limits.limits()["model_eur"]["workspace"])

    def test_bad_values_fall_back_to_safe_defaults_and_zero_switches_a_limit_off(self):
        from dclab_rnd import limits

        with mock.patch.dict(os.environ, {"DCLAB_LIMIT_REQUESTS_PER_MINUTE": "lots", "DCLAB_LIMIT_JOBS_PER_USER": "0"}):
            values = limits.limits()
        self.assertEqual(values["requests"]["user"], 600)
        self.assertIsNone(values["jobs"]["user"])


if __name__ == "__main__":
    unittest.main()
