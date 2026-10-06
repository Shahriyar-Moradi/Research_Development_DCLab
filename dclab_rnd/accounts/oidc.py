"""Sign-in through the company's identity provider (package 10.2, part B): OpenID Connect, authorization code with PKCE.

    DCLAB_AUTH=oidc
    DCLAB_OIDC_ISSUER        https://login.example.com/realms/acme  (its /.well-known/openid-configuration is read)
    DCLAB_OIDC_CLIENT_ID     the client registered for DCLab
    DCLAB_OIDC_CLIENT_SECRET its secret (from the environment or a Docker secret; never in a file in the repository)
    DCLAB_OIDC_REDIRECT      https://dclab.example.com/api/auth/oidc/callback (default: this server's own callback URL)
    DCLAB_OIDC_ROLES         group=role pairs, comma separated: "dclab-owners=owner,ml=data_scientist,risk=reviewer,all=viewer";
                             "group=workspace_id:role" puts a group in another workspace than the server's own
    DCLAB_OIDC_GROUPS_CLAIM  the claim that lists a person's groups (default "groups")
    DCLAB_SECRET_KEY         signs the short-lived cookie that carries the login's state, nonce and PKCE verifier

On each sign-in the person's memberships in the workspaces the mapping names follow their groups: a group they left
takes the role away; the highest role their groups give is theirs. A person none of whose groups is mapped is refused.
The ID token is checked: signature against the provider's published keys, issuer, audience, expiry and nonce.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any
from urllib.parse import urlencode

from .roles import ROLES

COOKIE = "dclab_oidc"
RANK = {r: n for n, r in enumerate(reversed(ROLES))}  # viewer < reviewer < data_scientist < owner
_DISCOVERY: dict[str, tuple[float, dict[str, Any]]] = {}


class OidcError(Exception):
    pass


def config() -> dict[str, Any]:
    issuer = (os.environ.get("DCLAB_OIDC_ISSUER") or "").rstrip("/")
    missing = [k for k in ("DCLAB_OIDC_ISSUER", "DCLAB_OIDC_CLIENT_ID", "DCLAB_OIDC_CLIENT_SECRET", "DCLAB_SECRET_KEY", "DCLAB_OIDC_ROLES")
               if not os.environ.get(k)]
    if missing:
        raise OidcError("sign-in through the identity provider needs " + ", ".join(missing))
    return {"issuer": issuer, "client_id": os.environ["DCLAB_OIDC_CLIENT_ID"], "client_secret": os.environ["DCLAB_OIDC_CLIENT_SECRET"],
            "redirect": os.environ.get("DCLAB_OIDC_REDIRECT") or "", "groups_claim": os.environ.get("DCLAB_OIDC_GROUPS_CLAIM") or "groups",
            "roles": parse_roles(os.environ["DCLAB_OIDC_ROLES"]), "secret": os.environ["DCLAB_SECRET_KEY"].encode()}


def parse_roles(text: str) -> list[tuple[str, str | None, str]]:
    """``group=role`` or ``group=workspace:role`` pairs → (group, workspace or None for the server's own, role)."""
    out = []
    for part in (p.strip() for p in text.split(",") if p.strip()):
        group, _, target = part.partition("=")
        workspace, _, role = target.rpartition(":")
        if not group.strip() or role.strip() not in ROLES:
            raise OidcError(f"DCLAB_OIDC_ROLES: {part!r} is not group=role with a role among {', '.join(ROLES)}")
        out.append((group.strip(), workspace.strip() or None, role.strip()))
    return out


def client(**kwargs: Any):
    """The HTTP client for the provider (a test replaces it with one that answers like a provider)."""
    import httpx

    return httpx.Client(timeout=15, **kwargs)


def _safely(what: str, fn):
    """Run a step against the provider; anything it answers wrongly (down, malformed, not JSON) is an OidcError."""
    import httpx

    try:
        return fn()
    except OidcError:
        raise
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise OidcError(f"the identity provider could not {what} ({type(exc).__name__})") from None


def discovery(issuer: str) -> dict[str, Any]:
    cached = _DISCOVERY.get(issuer)
    if cached and cached[0] > time.time():
        return cached[1]

    def read():
        with client() as http:
            doc = http.get(issuer + "/.well-known/openid-configuration").raise_for_status().json()
        if not isinstance(doc, dict) or str(doc.get("issuer", "")).rstrip("/") != issuer:
            raise OidcError("the provider's discovery document names another issuer")
        for key in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
            if not isinstance(doc.get(key), str):
                raise OidcError(f"the provider's discovery document has no {key}")
        return doc
    doc = _safely("be reached", read)
    _DISCOVERY[issuer] = (time.time() + 3600, doc)
    return doc


def _same(a: str, b: str) -> bool:
    return hmac.compare_digest(str(a or "").encode(), str(b or "").encode())


# ---------------------------------------------------------------------- the login's state, carried in a signed cookie

def _sign(secret: bytes, data: dict[str, Any]) -> str:
    body = base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode()).decode()
    return body + "." + hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()


def _unsign(secret: bytes, value: str) -> dict[str, Any] | None:
    body, _, mac = (value or "").rpartition(".")
    if not body or not _same(mac, hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(body.encode()))
    except ValueError:
        return None
    return data if isinstance(data, dict) and data.get("exp", 0) > time.time() else None


def start(redirect: str) -> tuple[str, str]:
    """Where to send the browser, and the signed cookie that remembers this login (10 minutes)."""
    cfg = config()
    _known_workspaces(cfg["roles"])
    doc = discovery(cfg["issuer"])
    state, nonce, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(24), secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    query = {"response_type": "code", "client_id": cfg["client_id"], "redirect_uri": cfg["redirect"] or redirect,
             "scope": "openid email profile", "state": state, "nonce": nonce, "code_challenge": challenge, "code_challenge_method": "S256"}
    cookie = _sign(cfg["secret"], {"state": state, "nonce": nonce, "verifier": verifier, "redirect": cfg["redirect"] or redirect, "exp": time.time() + 600})
    return doc["authorization_endpoint"] + "?" + urlencode(query), cookie


def finish(code: str, state: str, cookie: str) -> dict[str, Any]:
    """Exchange the code, check the ID token, and return the person's claims: sub, email (and whether the provider
    verified it), name and groups."""
    from urllib.parse import quote

    cfg = config()
    pending = _unsign(cfg["secret"], cookie)
    if pending is None or not _same(pending.get("state"), state):
        raise OidcError("the sign-in expired or did not start here; try again")
    doc = discovery(cfg["issuer"])

    def exchange():
        with client() as http:  # client_secret_basic, the id and the secret form-encoded first (RFC 6749 2.3.1)
            answer = http.post(doc["token_endpoint"], data={"grant_type": "authorization_code", "code": code, "redirect_uri": pending["redirect"],
                                                            "code_verifier": pending["verifier"]},
                               auth=(quote(cfg["client_id"], safe=""), quote(cfg["client_secret"], safe="")))
            if answer.status_code != 200:
                raise OidcError("the identity provider refused the sign-in")
            body = answer.json()
            return (body.get("id_token") if isinstance(body, dict) else None), http.get(doc["jwks_uri"]).raise_for_status().json()
    id_token, keys = _safely("finish the sign-in", exchange)
    claims = verify(id_token, keys, str(doc["issuer"]), cfg["client_id"])  # the issuer exactly as the provider names it
    if not _same(claims.get("nonce"), pending.get("nonce")):
        raise OidcError("the sign-in's nonce does not match")
    groups = claims.get(cfg["groups_claim"]) or []
    return {"sub": str(claims["sub"]), "email": str(claims.get("email") or "").strip().lower(), "email_verified": claims.get("email_verified") is True,
            "name": str(claims.get("name") or claims.get("preferred_username") or "")[:200],
            "groups": [str(g) for g in groups] if isinstance(groups, list) else [str(groups)]}


def verify(id_token: str | None, jwks: dict[str, Any], issuer: str, audience: str) -> dict[str, Any]:
    import jwt

    if not id_token or not isinstance(jwks, dict):
        raise OidcError("the identity provider sent no ID token")
    try:
        header = jwt.get_unverified_header(id_token)
        key = next((k for k in jwks.get("keys", []) if isinstance(k, dict) and k.get("kid") == header.get("kid")), None)
        if key is None:
            raise OidcError("the ID token is signed with a key the provider does not publish")
        signing = jwt.PyJWK(key).key
        claims = jwt.decode(id_token, signing, algorithms=["RS256", "ES256", "PS256"], audience=audience, issuer=issuer,
                            options={"require": ["exp", "iat", "sub", "iss", "aud"]})
    except (jwt.PyJWTError, TypeError, ValueError) as exc:  # a malformed key or token from a broken provider: refused, not a server error
        raise OidcError(f"the ID token is not valid ({type(exc).__name__})") from None
    if isinstance(claims.get("aud"), list) and len(claims["aud"]) > 1 and claims.get("azp") != audience:
        raise OidcError("the ID token was issued to another client")
    return claims


def roles_for(groups: list[str], mapping: list[tuple[str, str | None, str]], default_workspace: str) -> dict[str, str]:
    """workspace → the highest role the person's groups give there."""
    out: dict[str, str] = {}
    for group, workspace, role in mapping:
        if group in groups:
            target = workspace or default_workspace
            if target not in out or RANK[role] > RANK[out[target]]:
                out[target] = role
    return out


def _known_workspaces(mapping: list[tuple[str, str | None, str]]) -> None:
    """Every workspace the mapping names exists (a typo would otherwise fail each sign-in half-way)."""
    from .store import Accounts

    accounts = Accounts()
    for _, workspace, _ in mapping:
        if workspace is not None:
            try:
                accounts.workspace(workspace)
            except KeyError:
                raise OidcError(f"DCLAB_OIDC_ROLES names the workspace {workspace}, which does not exist") from None


def _placeholder(sub: str) -> str:
    """The email of a user whose provider gives no verified email: unique to the subject, never a real address."""
    return "sso-" + hashlib.sha256(sub.encode()).hexdigest()[:24] + "@sso.invalid"


def sign_in(accounts: Any, claims: dict[str, Any], default_workspace: str) -> tuple[dict[str, Any], dict[str, str], list[tuple[str, str | None, str | None]]]:
    """The user for these claims and their memberships in the mapped workspaces, set from their groups in one transaction.

    A user is found by the provider's subject. Else an existing account with the same email is linked, only when the
    provider verified that email and the account is not linked to another subject (an email a person can type is not
    proof of who they are). Else a new user is made, with the email when verified. Returns (user, workspace → role,
    the membership changes as (workspace, old role, new role)); refuses a person whose groups give no role."""
    from sqlalchemy.exc import IntegrityError

    from .store import AccountError

    cfg = config()
    _known_workspaces(cfg["roles"])
    roles = roles_for(claims["groups"], cfg["roles"], default_workspace)
    if not roles:
        raise OidcError("none of your groups gives you a role in DCLab; ask its owner")
    try:
        user = accounts.by_subject(claims["sub"])
        if user is None and claims.get("email_verified") and claims["email"]:
            found = accounts.by_email(claims["email"])
            if found is not None:
                if found.get("subject") not in (None, claims["sub"]):
                    raise OidcError("an account with this email is linked to another sign-in; ask the owner")
                if not found["active"]:
                    raise OidcError("this account is switched off")
                accounts.update_user(found["id"], subject=claims["sub"])
                user = accounts.user(found["id"])
        if user is None:
            email = claims["email"] if claims.get("email_verified") and claims["email"] else _placeholder(claims["sub"])
            user = accounts.create_user(email, claims["name"], subject=claims["sub"])
        if not user.get("active", True):
            raise OidcError("this account is switched off")
        managed = {workspace or default_workspace for _, workspace, _ in cfg["roles"]}
        changes = accounts.sync_memberships(user["id"], {w: r for w, r in roles.items() if w in managed}, managed)
    except (AccountError, IntegrityError) as exc:
        raise OidcError(f"the account could not be set up ({type(exc).__name__})") from None
    return user, roles, changes
