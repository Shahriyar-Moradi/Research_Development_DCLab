"""The Admin page's Control tab (A3 of the 2026-11-03 plan): the owner's switches kept in the workspace, read by the live-model
guard and the gateway's cap, an environment setting winning, the password asked again, and every change audited."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import control  # noqa: E402
from dclab_rnd.accounts import admin_login  # noqa: E402

PASSWORD = "a long admin password"


class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def test_defaults_then_a_change_then_a_refusal(self):
        self.assertEqual((control.live_allowed(self.home), control.monthly_cap(self.home)), (True, None))
        control.set_values(self.home, live_models=False, workspace_monthly_eur=20)
        self.assertEqual((control.live_allowed(self.home), control.monthly_cap(self.home)), (False, 20.0))
        with self.assertRaises(ValueError):
            control.set_values(self.home, workspace_monthly_eur=-1)
        (self.home / control.FILE).write_text("{broken")
        self.assertTrue(control.live_allowed(self.home))  # a broken file is the defaults, never an error

    def test_the_live_guard_refuses_a_paid_provider_when_stopped_and_lets_a_local_model_through(self):
        from dclab_rnd.models import live_guard

        control.set_values(self.home, live_models=False)
        with mock.patch.dict(os.environ, {"DCLAB_AGENT_HOME": str(self.home), live_guard.ALLOW: "1"}):
            with self.assertRaises(live_guard.ProviderBlocked):
                live_guard.check("anthropic", "https://api.anthropic.com")
            live_guard.check("ollama", "http://127.0.0.1:11434")  # a model on this machine is not a paid call

    def test_the_environment_cap_wins_over_the_switch(self):
        from dclab_rnd.models.gateway import workspace_cap

        control.set_values(self.home, workspace_monthly_eur=20)
        with mock.patch.dict(os.environ, {"DCLAB_WORKSPACE_MONTHLY_EUR": ""}):
            self.assertEqual(workspace_cap(self.home), 20.0)
        with mock.patch.dict(os.environ, {"DCLAB_WORKSPACE_MONTHLY_EUR": "5"}):
            self.assertEqual(workspace_cap(self.home), 5.0)


class RouteTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        self.home = Path(tempfile.mkdtemp())
        self.client = TestClient(create_app(self.home))
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def test_the_state_reads_and_a_change_needs_the_password_when_a_login_is_set(self):
        r = self.client.get("/api/admin/control").json()
        self.assertIn("disk_free_gb", r)
        self.assertEqual(r["switches"]["live_models"], True)
        admin_login.set_login(self.home, "owner", PASSWORD)
        token = admin_login.sign_in(self.home, "owner", PASSWORD)
        self.client.cookies.set(admin_login.COOKIE, token)
        bad = self.client.post("/api/admin/control", json={"live_models": False, "password": "wrong password!!"}, headers=self.h)
        self.assertEqual(bad.status_code, 403)
        self.assertTrue(control.live_allowed(self.home))
        ok = self.client.post("/api/admin/control", json={"live_models": False, "password": PASSWORD}, headers=self.h)
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertFalse(control.live_allowed(self.home))
        _, rows = self.client.app.state.services.audit.query(100, 0)
        moves = [(r.get("move"), r.get("status")) for r in rows if r.get("kind") == "admin"]
        self.assertIn(("control_set", "allowed"), moves)
        self.assertIn(("control_set", "refused"), moves)
        self.assertNotIn(PASSWORD, json.dumps(rows, default=str))

    def test_resume_and_stop_every_job(self):
        self.assertEqual(self.client.post("/api/admin/control/resume-models", headers=self.h).json(), {"resumed": True})
        self.assertIn("stopped", self.client.post("/api/admin/control/stop-jobs", json={}, headers=self.h).json())


if __name__ == "__main__":
    unittest.main()
