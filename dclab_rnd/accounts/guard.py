"""Who is asking, and whether they may (package 10.2): one role check for every route.

``permission(method, path)`` names what a route needs (``roles.PERMISSIONS``): reading is ``read``, writing is
``write``, and the few routes that approve, review or administer say so below. ``agentic.api_models.Router`` adds
``require(permission)`` to every route it builds, so a new route is checked without anyone remembering to. A route
that changes only some fields with more at stake (a project's gate switches) checks those itself with ``demand``.

``identify`` reads the session cookie or an API token. Without sign-in configured every request is the local owner.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from .principal import ANONYMOUS, LOCAL_OWNER, Principal, current
from .roles import allows, label

COOKIE = "dclab_session"

PUBLIC = {("GET", "/healthz"), ("GET", "/readyz"),  # a load balancer's probes (12.4): no data, no session
          ("GET", "/"), ("GET", "/guide"), ("GET", "/classic"), ("GET", "/favicon.ico"),  # the page shells: no data; their API calls are checked
          ("GET", "/api/config"), ("POST", "/api/auth/password"), ("POST", "/api/auth/signout"), ("GET", "/api/auth/me"),
          ("GET", "/api/auth/oidc/start"), ("GET", "/api/auth/oidc/callback")}
SPECIAL = {
    ("POST", "/api/projects/{project_id}/approvals"): "approve_gate",
    ("POST", "/api/projects/{project_id}/stages/{stage}/approve"): "approve_stage",
    ("POST", "/api/lessons/{lesson_id}/review"): "review",
    ("POST", "/api/models/routing"): "review",
    # questions spend the model budget but change nothing: every role may ask (package 10.6 limits the spend)
    ("POST", "/api/projects/{project_id}/ask"): "read",
    ("POST", "/api/projects/{project_id}/graph/check"): "read",
    ("POST", "/api/evidence/ask"): "read",
    # signed in, any role: switching workspace, one's own API tokens
    ("POST", "/api/auth/workspace"): "read",
    ("GET", "/api/auth/tokens"): "read",
    ("POST", "/api/auth/tokens"): "read",
    ("DELETE", "/api/auth/tokens/{token_id}"): "read",
    # the owner's: members and their roles
    ("GET", "/api/workspace/members"): "read",
    ("POST", "/api/workspace/members"): "admin",
    ("PATCH", "/api/workspace/members/{user_id}"): "admin",
    ("DELETE", "/api/workspace/members/{user_id}"): "admin",
}


def permission(method: str, path: str) -> str:
    method = method.upper()
    if (method, path) in PUBLIC:
        return "public"
    if (method, path) in SPECIAL:
        return SPECIAL[(method, path)]
    return "read" if method in ("GET", "HEAD", "OPTIONS") else "write"


def demand(needed: str, what: str = "this") -> Principal:
    """Refuse unless the person may ``needed``: 401 when signed out, 403 when their role may not."""
    who = current() or LOCAL_OWNER
    if not allows(who.role, needed):
        if not who.signed_in:
            raise HTTPException(401, "Sign in first")
        raise HTTPException(403, {"error": "role", "message": f"A {label(who.role)} may not {what}", "role": who.role, "needs": needed})
    return who


def require(*needed: str):
    async def check() -> None:
        for each in needed:
            if each != "public":
                demand(each, ROUTE_WORDS.get(each, "do this"))
    check.__name__ = "require_" + "_".join(needed)
    return check


ROUTE_WORDS = {"read": "read this workspace", "write": "change this workspace", "approve_stage": "approve a stage",
               "approve_gate": "approve a gate", "review": "review this", "admin": "change this setting"}


# ---------------------------------------------------------------------- identifying a request

def identify(headers: dict[str, str], cookies: dict[str, str], settings: Any, default_workspace: str | None) -> Principal:
    """The person behind a request: an API token (``Authorization: Bearer dclab_…``), else the session cookie; the
    local owner when no sign-in is configured; anonymous otherwise."""
    if settings.auth == "none":
        return Principal(role="owner", workspace_id=default_workspace, via="local")
    from .store import Accounts

    accounts = Accounts()
    bearer = headers.get("authorization", "")
    if bearer.lower().startswith("bearer "):
        found = accounts.token(bearer[7:].strip())
        if found is None:
            return ANONYMOUS
        return _principal(accounts, found["user_id"], found["workspace_id"], via="token")
    token = cookies.get(COOKIE)
    if token:
        found = accounts.session(token)
        if found is not None:
            return _principal(accounts, found["user_id"], found["workspace_id"], via="session", session_id=found["id"], csrf=found["csrf"])
    return ANONYMOUS


def _principal(accounts: Any, user_id: str, workspace_id: str, **extra: Any) -> Principal:
    try:
        user = accounts.user(user_id)
    except KeyError:
        return ANONYMOUS
    if not user["active"]:
        return ANONYMOUS
    spaces = tuple(accounts.workspaces_of(user_id))
    role = next((r for w, _, r in spaces if w == workspace_id), None)  # removed from the workspace: no role there now
    return Principal(role=role, workspace_id=workspace_id, user_id=user_id, email=user["email"], name=user["name"],
                     workspaces=spaces, **extra)
