"""The Admin area's own sign-in (A1 and A2 of the 2026-11-03 plan): one username and password, a hashed session that ends,
a lockout after wrong passwords, and the owner-only routes closed to a browser without that session."""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.accounts import admin_login as al  # noqa: E402

PASSWORD = "a long admin password"


class LoginTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        al.set_login(self.home, "owner", PASSWORD)

    def test_the_file_keeps_a_hash_never_the_password_and_only_the_owner_may_read_it(self):
        text = (self.home / al.FILE).read_text()
        self.assertNotIn(PASSWORD, text)
        self.assertTrue(json.loads(text)["hash"].startswith("$argon2id$"))
        self.assertEqual((self.home / al.FILE).stat().st_mode & 0o777, 0o600)
        with self.assertRaises(ValueError):
            al.set_login(self.home, "owner", "short")
        with self.assertRaises(ValueError):
            al.set_login(self.home, "a b", PASSWORD)

    def test_a_right_password_opens_a_session_kept_only_as_its_hash(self):
        token = al.sign_in(self.home, "owner", PASSWORD)
        self.assertTrue(al.session(self.home, token))
        self.assertNotIn(token, (self.home / al.SESSIONS).read_text())
        self.assertFalse(al.session(self.home, token + "x"))
        self.assertFalse(al.session(self.home, None))
        al.sign_out(self.home, token)
        self.assertFalse(al.session(self.home, token))

    def test_a_wrong_username_or_password_is_refused(self):
        with self.assertRaises(ValueError):
            al.sign_in(self.home, "owner", "wrong password!!")
        with self.assertRaises(ValueError):
            al.sign_in(self.home, "someone", PASSWORD)

    def test_five_wrong_passwords_lock_the_login_even_for_the_right_one(self):
        for _ in range(al.FAILURES - 1):
            with self.assertRaises(ValueError):
                al.sign_in(self.home, "owner", "wrong password!!")
        with self.assertRaises(al.Locked):
            al.sign_in(self.home, "owner", "wrong password!!")
        with self.assertRaises(al.Locked):
            al.sign_in(self.home, "owner", PASSWORD)
        later = datetime.now(timezone.utc) + timedelta(minutes=al.LOCK_MINUTES + 1)
        with mock.patch.object(al, "_now", return_value=later):
            self.assertTrue(al.sign_in(self.home, "owner", PASSWORD))  # open again after the lock
        self.assertEqual(json.loads((self.home / al.FILE).read_text())["locks"], 0)

    def test_a_session_ends_when_idle_and_after_its_longest_life(self):
        token = al.sign_in(self.home, "owner", PASSWORD)
        idle = datetime.now(timezone.utc) + timedelta(minutes=al.IDLE_MINUTES + 1)
        with mock.patch.object(al, "_now", return_value=idle):
            self.assertFalse(al.session(self.home, token))
        token = al.sign_in(self.home, "owner", PASSWORD)
        start = datetime.now(timezone.utc)
        for minutes in range(20, al.MAX_HOURS * 60, 20):  # used every 20 minutes: never idle
            with mock.patch.object(al, "_now", return_value=start + timedelta(minutes=minutes)):
                self.assertTrue(al.session(self.home, token), minutes)
        with mock.patch.object(al, "_now", return_value=start + timedelta(hours=al.MAX_HOURS, minutes=1)):
            self.assertFalse(al.session(self.home, token))

    def test_a_password_change_needs_the_current_one_and_ends_every_session(self):
        token = al.sign_in(self.home, "owner", PASSWORD)
        with self.assertRaises(ValueError):
            al.change_password(self.home, "not the password", "another long password")
        al.change_password(self.home, PASSWORD, "another long password")
        self.assertFalse(al.session(self.home, token))
        self.assertTrue(al.sign_in(self.home, "owner", "another long password"))

    def test_the_state_the_page_reads(self):
        self.assertEqual(al.state(self.home, None)["required"], True)
        self.assertEqual(al.state(self.home, None)["signed_in"], False)
        self.assertTrue(al.state(self.home, al.sign_in(self.home, "owner", PASSWORD))["signed_in"])
        self.assertTrue(al.remove(self.home))
        self.assertEqual(al.state(self.home, None)["required"], False)



class RouteTests(unittest.TestCase):
    """A2: with an admin login set, the owner-only routes need its session; without one nothing changes."""

    def setUp(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        self.home = Path(tempfile.mkdtemp())
        self.client = TestClient(create_app(self.home))
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def sign_in(self, password=PASSWORD):
        return self.client.post("/api/admin/signin", json={"username": "owner", "password": password}, headers=self.h)

    def test_without_an_admin_login_the_owner_routes_stay_open(self):
        self.assertEqual(self.client.get("/api/admin/bench").status_code, 200)
        self.assertEqual(self.client.get("/api/admin/me").json()["required"], False)

    def test_with_an_admin_login_the_owner_routes_need_its_session(self):
        al.set_login(self.home, "owner", PASSWORD)
        for path in ("/api/admin/bench", "/api/students", "/api/experience/sessions"):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 401, path)
            self.assertEqual(r.json()["detail"]["error"], "admin_login")
        self.assertEqual(self.client.get("/api/config").status_code, 200)  # the rest of DCLab is as before
        self.assertEqual(self.client.get("/api/drafts").status_code, 200)
        r = self.sign_in()
        self.assertEqual(r.status_code, 200)
        cookie = r.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=strict", cookie)
        self.assertEqual(self.client.get("/api/admin/bench").status_code, 200)
        self.assertTrue(self.client.get("/api/admin/me").json()["signed_in"])
        self.client.post("/api/admin/signout", headers=self.h)
        self.assertEqual(self.client.get("/api/admin/bench").status_code, 401)

    def test_a_sign_in_needs_the_page_request_token(self):
        al.set_login(self.home, "owner", PASSWORD)
        r = self.client.post("/api/admin/signin", json={"username": "owner", "password": PASSWORD})
        self.assertEqual(r.status_code, 403)

    def test_wrong_passwords_lock_the_sign_in_and_every_attempt_is_audited_without_the_password(self):
        al.set_login(self.home, "owner", PASSWORD)
        codes = [self.sign_in("wrong password!!").status_code for _ in range(al.FAILURES)]
        self.assertEqual(codes, [401] * (al.FAILURES - 1) + [429])
        self.assertEqual(self.sign_in().status_code, 429)  # the right one too, while locked
        text = (self.home / "audit.jsonl").read_text()
        rows = [json.loads(line) for line in text.splitlines()]
        admin = [r for r in rows if r.get("kind") == "admin"]
        self.assertGreaterEqual(len(admin), al.FAILURES + 1)
        self.assertNotIn(PASSWORD, text)
        self.assertNotIn("wrong password!!", text)

    def test_a_password_change_needs_the_current_one_and_signs_out(self):
        al.set_login(self.home, "owner", PASSWORD)
        self.sign_in()
        r = self.client.post("/api/admin/password", json={"current": "nope nope nope", "new": "another long password"}, headers=self.h)
        self.assertEqual(r.status_code, 422)
        r = self.client.post("/api/admin/password", json={"current": PASSWORD, "new": "another long password"}, headers=self.h)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get("/api/admin/bench").status_code, 401)


class AccountLockoutTests(unittest.TestCase):
    """A2: the account sign-in (DCLAB_AUTH=password) had no limit on wrong passwords."""

    def setUp(self):
        from dclab_rnd.accounts import api

        self.api = api
        api._FAILED.clear()
        api._LOCKED.clear()

    def test_five_wrong_passwords_in_the_window_lock_the_login_then_it_opens(self):
        api, now = self.api, 1000.0
        for i in range(api.FAILURES - 1):
            self.assertIsNone(api._failed("owner", now + i))
        self.assertIsNotNone(api._failed("owner", now + 5))
        self.assertTrue(api._locked_until("owner", now + 6))
        self.assertIsNone(api._locked_until("someone else", now + 6))  # one login's lock is its own
        self.assertIsNone(api._locked_until("owner", now + 6 + api.LOCK_SECONDS))

    def test_failures_spread_beyond_the_window_never_lock(self):
        api = self.api
        for i in range(api.FAILURES * 2):
            self.assertIsNone(api._failed("owner", 1000.0 + i * (api.WINDOW_SECONDS / (api.FAILURES - 1) + 1)))


class ReviewFixTests(unittest.TestCase):
    """The review of 2026-11-03: a burst skipped the lockout, the table grew without end, review switches stayed open,
    the username leaked, a broken file broke every request, and the lock's doubling was untested."""

    def test_a_burst_of_guesses_is_counted_before_any_is_checked(self):
        from dclab_rnd.accounts import api

        api._FAILED.clear(); api._LOCKED.clear()
        refused = [bool(api._locked_until("owner", 1000.0) or api._attempt("owner", 1000.0)) for _ in range(200)]
        self.assertEqual(refused.index(True), api.FAILURES - 1)  # the fifth attempt locks: at most four are checked
        self.assertTrue(all(refused[api.FAILURES - 1:]))
        api._succeeded("someone")

    def test_the_failure_table_stays_bounded(self):
        from dclab_rnd.accounts import api

        api._FAILED.clear(); api._LOCKED.clear()
        for i in range(api.MAX_LOGINS + 50):
            api._attempt(f"login-{i}" * 40, 1000.0 + i)
        self.assertLessEqual(len(api._FAILED), api.MAX_LOGINS + 1)
        self.assertTrue(all(len(k) <= 254 for k in [api._key("x" * 1000)]))

    def test_the_lock_doubles_while_it_repeats(self):
        home = Path(tempfile.mkdtemp())
        al.set_login(home, "owner", PASSWORD)
        start = datetime.now(timezone.utc)
        for round_, minutes in ((0, al.LOCK_MINUTES), (1, 2 * al.LOCK_MINUTES)):
            at = start + timedelta(hours=round_)
            with mock.patch.object(al, "_now", return_value=at):
                for _ in range(al.FAILURES - 1):
                    with self.assertRaises(ValueError):
                        al.sign_in(home, "owner", "wrong password!!")
                with self.assertRaises(al.Locked) as caught:
                    al.sign_in(home, "owner", "wrong password!!")
            until = datetime.fromisoformat(caught.exception.until)
            self.assertEqual(round((until - at).total_seconds() / 60), minutes)

    def test_a_broken_session_file_is_no_session_never_an_error(self):
        home = Path(tempfile.mkdtemp())
        al.set_login(home, "owner", PASSWORD)
        token = al.sign_in(home, "owner", PASSWORD)
        doc = json.loads((home / al.SESSIONS).read_text())
        for entry in doc.values():
            entry["seen"] = "2026-11-03T10:00:00"  # no time zone: a hand edit
        (home / al.SESSIONS).write_text(json.dumps(doc))
        self.assertFalse(al.session(home, token))
        self.assertEqual(al.state(home, token)["signed_in"], False)

    def test_the_username_is_never_given_before_sign_in(self):
        home = Path(tempfile.mkdtemp())
        al.set_login(home, "owner", PASSWORD)
        self.assertIsNone(al.state(home, None)["username"])
        self.assertEqual(al.state(home, al.sign_in(home, "owner", PASSWORD))["username"], "owner")


class SwitchTests(RouteTests):
    """Review finding 3: the routing switches (review) and the audit trail were open without the admin session."""

    def test_the_routing_switches_and_the_audit_need_the_admin_session(self):
        al.set_login(self.home, "owner", PASSWORD)
        for method, path in (("POST", "/api/models/routing"), ("POST", "/api/ledger/routing"), ("GET", "/api/platform/audit")):
            r = self.client.request(method, path, json={}, headers=self.h)
            self.assertEqual(r.status_code, 401, path)
        self.sign_in()
        self.assertEqual(self.client.get("/api/platform/audit").status_code, 200)


if __name__ == "__main__":
    unittest.main()
