"""Sign-in through the company's identity provider (package 10.2, part B), against a fake provider: discovery, an
authorization redirect with PKCE, the code exchanged for an ID token signed with the provider's key, and groups mapped
to roles. A wrong state, a token for another client, a forged signature and a person without a mapped group are refused."""
import base64
import hashlib
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pgtest  # noqa: E402

ISSUER = "https://login.example.test/realms/acme/"  # with the trailing slash some providers keep in their tokens


class FakeProvider:
    """Answers like an identity provider; ``groups`` and ``audience`` say what the next ID token carries."""

    def __init__(self):
        import jwt
        from cryptography.hazmat.primitives.asymmetric import rsa

        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.jwk = {**__import__("json").loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key())), "kid": "k1", "use": "sig", "alg": "RS256"}
        self.groups, self.audience, self.subject, self.signer = ["ml-team"], "dclab", "sub-123", self.key
        self.verified, self.keys_down = True, False
        self.codes: dict[str, dict] = {}

    def authorize(self, url: str, email: str) -> str:
        """The provider's login page: remembers the nonce and the PKCE challenge, and sends the browser back with a code."""
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        code = "code-" + q["state"][:8]
        self.codes[code] = {"nonce": q["nonce"], "challenge": q["code_challenge"], "email": email}
        return f"/api/auth/oidc/callback?code={code}&state={q['state']}"

    def handler(self, request):
        import httpx
        import jwt

        if request.url.path.endswith("/.well-known/openid-configuration"):
            return httpx.Response(200, json={"issuer": ISSUER, "authorization_endpoint": ISSUER + "/auth", "token_endpoint": ISSUER + "/token",
                                             "jwks_uri": ISSUER + "/certs"})
        if request.url.path.endswith("/certs"):
            return httpx.Response(503) if self.keys_down else httpx.Response(200, json={"keys": [self.jwk]})
        if request.url.path.endswith("/token"):
            form = parse_qs(request.content.decode())
            pending = self.codes.pop(form["code"][0], None)
            verifier = form["code_verifier"][0]
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
            if pending is None or challenge != pending["challenge"]:
                return httpx.Response(400, json={"error": "invalid_grant"})
            now = int(time.time())
            claims = {"iss": ISSUER, "aud": self.audience, "sub": self.subject, "iat": now, "exp": now + 300, "nonce": pending["nonce"],
                      "email": pending["email"], "email_verified": self.verified, "name": "Sam Lee", "groups": self.groups}
            return httpx.Response(200, json={"id_token": jwt.encode(claims, self.signer, algorithm="RS256", headers={"kid": "k1"}),
                                             "access_token": "a", "token_type": "Bearer"})
        return httpx.Response(404)


class OidcTests(unittest.TestCase):
    def setUp(self):
        try:
            import httpx
            from fastapi.testclient import TestClient

            from dclab_rnd.accounts import oidc
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"dependencies not installed: {error}")
        url = pgtest.require()
        pgtest.empty(url)
        self.provider = FakeProvider()
        env = mock.patch.dict(os.environ, {
            "DCLAB_DATABASE_URL": url, "DCLAB_AUTH": "oidc", "OPENAI_API_KEY": "", "DCLAB_OIDC_ISSUER": ISSUER, "DCLAB_OIDC_CLIENT_ID": "dclab",
            "DCLAB_OIDC_CLIENT_SECRET": "not-a-real-secret", "DCLAB_SECRET_KEY": "a test signing key, not a real one",
            "DCLAB_OIDC_ROLES": "admins=owner,ml-team=data_scientist,risk=reviewer,everyone=viewer"})
        self.env_patch = env
        env.start()
        self.addCleanup(env.stop)
        from dclab_rnd.storage import db

        db.upgrade(url)
        oidc._DISCOVERY.clear()
        transport = httpx.MockTransport(self.provider.handler)
        patched = mock.patch.object(oidc, "client", lambda **kw: httpx.Client(transport=transport, **kw))
        patched.start()
        self.addCleanup(patched.stop)
        self.app = create_app(Path(tempfile.mkdtemp()))
        self.client = TestClient(self.app, follow_redirects=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def sign_in(self, email="sam@example.com", client=None, state=None):
        client = client or self.client
        start = client.get("/api/auth/oidc/start")
        self.assertEqual(start.status_code, 302, start.text)
        back = self.provider.authorize(start.headers["location"], email)
        if state is not None:
            back = back.split("&state=")[0] + f"&state={state}"
        return client.get(back)

    def test_groups_give_the_role_and_the_session_works(self):
        done = self.sign_in()
        self.assertEqual((done.status_code, done.headers["location"]), (302, "/"))
        me = self.client.get("/api/auth/me").json()
        self.assertEqual((me["signed_in"], me["email"], me["role"], me["role_label"]), (True, "sam@example.com", "data_scientist", "ML engineer"))
        csrf = self.client.get("/api/config").json()["csrf"]
        self.assertEqual(self.client.post("/api/projects", json={"name": "From SSO", "goal": "g"}, headers={"X-DCLab-Token": csrf}).status_code, 201)
        self.assertEqual(self.client.post("/api/auth/password", json={"email": "sam@example.com", "password": "x" * 12},
                                          headers={"X-DCLab-Token": csrf}).status_code, 404)  # this server signs in through the provider only

    def test_a_group_change_changes_the_role_at_the_next_sign_in(self):
        self.sign_in()
        self.provider.groups = ["risk"]
        from fastapi.testclient import TestClient

        again = TestClient(self.app, follow_redirects=False)
        self.sign_in(client=again)
        self.assertEqual(again.get("/api/auth/me").json()["role"], "reviewer")
        self.assertEqual(self.client.get("/api/auth/me").json()["role"], "reviewer")  # the earlier session reads the role again too

    def refused(self, reason, **kwargs):
        answer = self.sign_in(**kwargs)
        self.assertEqual(answer.status_code, 403, answer.text[:300])
        self.assertIn(reason, answer.text)
        return answer

    def test_refusals(self):
        self.provider.groups = ["marketing"]  # no group maps to a role
        self.refused("none of your groups")
        self.assertFalse(self.client.get("/api/auth/me").json()["signed_in"])
        self.provider.groups = ["ml-team"]
        self.refused("did not start here", state="forged-state")  # not the login this browser started
        self.refused("did not start here", state="é-non-ascii")
        self.provider.audience = "another-client"
        self.refused("InvalidAudienceError")  # a token for another client
        self.provider.audience = "dclab"
        from cryptography.hazmat.primitives.asymmetric import rsa

        self.provider.signer = rsa.generate_private_key(public_exponent=65537, key_size=2048)  # not the published key
        self.refused("InvalidSignatureError")
        self.provider.signer = self.provider.key
        self.provider.keys_down = True
        self.refused("could not finish the sign-in")  # the provider is down: the refusal page, not a server error
        self.provider.keys_down = False
        self.assertEqual(self.sign_in().status_code, 302)

    def test_an_email_links_an_account_only_when_the_provider_verified_it(self):
        from dclab_rnd.accounts.store import Accounts

        accounts = Accounts()
        boss = accounts.create_user("boss@example.com", "Boss", password="a long enough password")
        accounts.set_member(self.app.state.services.workspace_id, boss["id"], "owner")
        self.provider.groups, self.provider.verified, self.provider.subject = ["everyone"], False, "attacker"
        self.sign_in(email="boss@example.com")  # an email the person typed at the provider, not proven theirs
        me = self.client.get("/api/auth/me").json()
        self.assertNotEqual(me["email"], "boss@example.com")
        self.assertEqual(me["role"], "viewer")  # a new account, with what the group gives
        self.assertEqual(accounts.role_of(boss["id"], self.app.state.services.workspace_id), "owner")  # the real account is untouched
        self.assertIsNone(accounts.user(boss["id"])["subject"])
        self.provider.verified, self.provider.subject = True, "boss-at-the-provider"
        from fastapi.testclient import TestClient

        boss_client = TestClient(self.app, follow_redirects=False)
        self.sign_in(email="boss@example.com", client=boss_client)  # verified: the account is linked
        self.assertEqual(accounts.user(boss["id"])["subject"], "boss-at-the-provider")
        self.provider.subject = "someone-else"  # another identity with the same verified email: never re-linked
        self.refused("linked to another sign-in", email="boss@example.com", client=TestClient(self.app, follow_redirects=False))
        self.assertEqual(accounts.user(boss["id"])["subject"], "boss-at-the-provider")

    def test_leaving_a_group_takes_the_role_away_and_sign_ins_are_audited(self):
        from dclab_rnd.accounts.store import Accounts

        accounts = Accounts()
        second = accounts.create_workspace("Risk team")
        with mock.patch.dict(os.environ, {"DCLAB_OIDC_ROLES": f"ml-team=data_scientist,risk=reviewer,risk={second}:reviewer"}):
            self.provider.groups = ["ml-team", "risk"]
            self.sign_in()
            user = accounts.by_subject("sub-123")
            self.assertEqual(accounts.role_of(user["id"], second), "reviewer")
            self.provider.groups = ["ml-team"]
            from fastapi.testclient import TestClient

            self.sign_in(client=TestClient(self.app, follow_redirects=False))
            self.assertIsNone(accounts.role_of(user["id"], second))  # left the group: no longer a member there
        own = self.app.state.services.workspace_id
        from dclab_rnd import audit

        kinds = [r["kind"] for r in audit.PgAudit(own).query(50, 0)[1]]
        self.assertIn("sign_in", kinds)
        changes = [r for r in audit.PgAudit(second).query(50, 0)[1] if r["kind"] == "role_change"]
        self.assertEqual([r["detail"]["args"]["role"] for r in changes], [None, "reviewer"])  # newest first: removed, then given

    def test_a_mapping_that_names_no_workspace_is_refused_before_anyone_is_sent_away(self):
        with mock.patch.dict(os.environ, {"DCLAB_OIDC_ROLES": "ml-team=wnosuchworkspace:viewer"}):
            answer = self.client.get("/api/auth/oidc/start")
            self.assertEqual(answer.status_code, 503)
            self.assertIn("wnosuchworkspace", answer.text)

    def test_the_mapping_is_read_strictly(self):
        from dclab_rnd.accounts import oidc

        self.assertEqual(oidc.parse_roles("a=owner, b=w123:viewer"), [("a", None, "owner"), ("b", "w123", "viewer")])
        with self.assertRaises(oidc.OidcError):
            oidc.parse_roles("a=admin")
        self.assertEqual(oidc.roles_for(["a", "b", "c"], [("a", None, "viewer"), ("c", None, "reviewer"), ("b", "w2", "owner")], "w1"),
                         {"w1": "reviewer", "w2": "owner"})


if __name__ == "__main__":
    unittest.main()
