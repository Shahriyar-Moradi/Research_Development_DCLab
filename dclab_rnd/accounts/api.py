"""Sign-in, sessions, workspaces, members and API tokens over HTTP (package 10.2).

    POST   /api/auth/password {login, password}      sign in with a username or an email; sets the session cookie (HttpOnly, SameSite=Lax)
    POST   /api/auth/signout                         ends the session
    GET    /api/auth/me                              who is signed in, their workspaces and role
    POST   /api/auth/workspace {workspace_id}        act in another workspace one belongs to
    GET    /api/workspace/members                    the workspace's members (every role)
    POST   /api/workspace/members {username?, email?, role, name?, password?}   add a member (the owner): a username, an email, or both
    PATCH  /api/workspace/members/{user_id} {role}   change a role (the owner)
    DELETE /api/workspace/members/{user_id}          remove a member (the owner)
    GET    /api/auth/tokens                          one's own API tokens (never the tokens themselves)
    POST   /api/auth/tokens {name}                   a new token for this workspace, shown once
    DELETE /api/auth/tokens/{token_id}               revoke one
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

from ..agentic.api_models import AnyObject, Doc, Page, Router
from .guard import COOKIE
from .. import context
from .principal import current
from .roles import LABELS, ROLES
from .store import SESSION_MAX_DAYS, AccountError, Accounts, digest

router = Router()


def _settings(request: Request):
    return request.app.state.settings


def _on(request: Request) -> Accounts:
    if _settings(request).auth == "none":
        raise HTTPException(404, "Accounts are not switched on for this server (DCLAB_AUTH)")
    return Accounts()


def _body(payload: Any) -> dict[str, Any]:
    return payload.model_dump(exclude_unset=True) if payload is not None else {}


def _signed_in() -> Any:
    who = current()
    if who is None or not who.user_id:
        raise HTTPException(401, "Sign in first")
    return who


@router.post("/api/auth/password", response_model=Page)
async def sign_in(payload: AnyObject, request: Request):
    accounts = _on(request)
    if _settings(request).auth != "password":
        raise HTTPException(404, "This server signs in through the company's identity provider")
    body = _body(payload)
    login = str(body.get("login") or body.get("username") or body.get("email") or "")  # a username, or an email
    user = await asyncio.to_thread(accounts.check_password, login, str(body.get("password") or ""))
    if user is None:
        await asyncio.sleep(0.3)  # a wrong password costs the guesser time
        _auth_event(getattr(context.services(), "workspace_id", None), "auth.sign_in_failed", {"reason": "bad_credentials", "via": "password"}, request=request)
        raise HTTPException(401, "The username (or email) or the password is not right")
    spaces = await asyncio.to_thread(accounts.workspaces_of, user["id"])
    if not spaces:
        _auth_event(getattr(context.services(), "workspace_id", None), "auth.sign_in_failed", {"reason": "no_workspace", "via": "password"}, user["id"], request)
        raise HTTPException(403, "This account is not a member of any workspace yet")
    token, csrf = await asyncio.to_thread(accounts.start_session, user["id"], spaces[0][0])
    _audit_account(spaces[0][0], "sign_in", "sign_in", user, {"via": "password"})
    _auth_event(spaces[0][0], "auth.sign_in", {"via": "password", "session": digest(token)}, user["id"], request)  # the session's id is the token's digest
    response = JSONResponse({"signed_in": True, "csrf": csrf, "workspace_id": spaces[0][0]})
    response.set_cookie(COOKIE, token, max_age=int(SESSION_MAX_DAYS * 86400), httponly=True, samesite="lax",
                        secure=bool(_settings(request).cookie_secure), path="/")
    return response


@router.post("/api/auth/signout", response_model=Page)
async def sign_out(request: Request):
    token = request.cookies.get(COOKIE)
    if token and _settings(request).auth != "none":
        who = current()
        if who is not None and who.signed_in and who.via == "session":  # a bearer caller's cookie is not its own session
            _auth_event(who.workspace_id, "auth.sign_out", {"session": getattr(who, "session_id", None)}, who.user_id)
        await asyncio.to_thread(Accounts().end_session, token)
    response = JSONResponse({"signed_in": False})
    response.delete_cookie(COOKIE, path="/")
    return response


@router.get("/api/auth/me", response_model=Page)
async def me(request: Request):
    who = current()
    return {"auth": _settings(request).auth, **(who.public() if who is not None else {"signed_in": False}),
            "roles": [{"key": r, "label": LABELS[r]} for r in ROLES]}


@router.post("/api/auth/workspace", response_model=Page)
async def switch_workspace(payload: AnyObject, request: Request):
    accounts, who = _on(request), _signed_in()
    target = str(_body(payload).get("workspace_id") or "")
    if target not in {w for w, _, _ in who.workspaces}:
        raise HTTPException(404, "Not a workspace you belong to")
    if who.via != "session":
        raise HTTPException(409, "An API token acts in its own workspace")
    await asyncio.to_thread(accounts.switch_session, request.cookies[COOKIE], target)
    _auth_event(who.workspace_id, "auth.workspace_switched", {"from": who.workspace_id, "to": target}, who.user_id)
    return {"workspace_id": target}


# ---------------------------------------------------------------------- members (the owner's)
@router.get("/api/workspace/members", response_model=list[Doc])
async def members(request: Request):
    accounts, who = _on(request), _signed_in()
    rows = await asyncio.to_thread(accounts.members, who.workspace_id)
    return [{**r, "role_label": LABELS[r["role"]]} for r in rows]


@router.post("/api/workspace/members", status_code=201, response_model=Doc)
async def add_member(payload: AnyObject, request: Request):
    accounts, who = _on(request), _signed_in()
    body = _body(payload)
    role = str(body.get("role") or "")
    if role not in ROLES:
        raise HTTPException(422, f"role is one of {', '.join(ROLES)}")
    email = str(body.get("email") or "").strip().lower()
    username = str(body.get("username") or "").strip().lower()

    own = bool(body.get("own_workspace"))  # the person gets a workspace of their own: their projects stay apart from everyone's

    def add():
        found = accounts.by_username(username) if username else accounts.by_email(email)
        if own and found is not None:  # a workspace of their own is for a new person: an existing account is never placed in one by someone else
            raise HTTPException(409, "This username or email already has an account: add them to this workspace instead, or they make their own")
        user = found or accounts.create_user(email or None, str(body.get("name") or ""), password=body.get("password") or None, username=username or None)
        if own:
            label = str(body.get("name") or "").strip() or username or email.split("@")[0]
            workspace_id = accounts.create_workspace(f"{label}'s lab")
            accounts.set_member(workspace_id, user["id"], "owner")  # they own their space
            if who.user_id:
                accounts.set_member(workspace_id, who.user_id, "owner")  # and the inviter can open it from the switcher
            return {"id": user["id"], "email": user["email"], "username": user.get("username"), "role": "owner", "role_label": LABELS["owner"], "created": found is None,
                    "workspace_id": workspace_id, "workspace": f"{label}'s lab"}
        accounts.set_member(who.workspace_id, user["id"], role)
        return {"id": user["id"], "email": user["email"], "username": user.get("username"), "role": role, "role_label": LABELS[role], "created": found is None}
    try:
        added = await asyncio.to_thread(add)
    except (AccountError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None
    _audit_member("add_member_own_workspace" if own else "add_member", added["id"], added["role"])
    return added


@router.patch("/api/workspace/members/{user_id}", response_model=Doc)
async def change_member(user_id: str, payload: AnyObject, request: Request):
    accounts, who = _on(request), _signed_in()
    role = str(_body(payload).get("role") or "")
    if role not in ROLES:
        raise HTTPException(422, f"role is one of {', '.join(ROLES)}")
    if await asyncio.to_thread(accounts.role_of, user_id, who.workspace_id) is None:
        raise HTTPException(404, "Not a member of this workspace")
    await _keep_an_owner(accounts, who.workspace_id, user_id, role)
    await asyncio.to_thread(accounts.set_member, who.workspace_id, user_id, role)
    _audit_member("change_role", user_id, role)
    return {"id": user_id, "role": role, "role_label": LABELS[role]}


@router.delete("/api/workspace/members/{user_id}", status_code=204, response_class=Response)
async def remove_member(user_id: str, request: Request):
    accounts, who = _on(request), _signed_in()
    if await asyncio.to_thread(accounts.role_of, user_id, who.workspace_id) is None:
        raise HTTPException(404, "Not a member of this workspace")
    await _keep_an_owner(accounts, who.workspace_id, user_id, None)
    await asyncio.to_thread(accounts.remove_member, who.workspace_id, user_id)
    _audit_member("remove_member", user_id, None)
    return Response(status_code=204)


async def _keep_an_owner(accounts: Accounts, workspace_id: str, user_id: str, role: str | None) -> None:
    """A workspace never loses its last owner (nobody could then manage its members)."""
    owners = [m["id"] for m in await asyncio.to_thread(accounts.members, workspace_id) if m["role"] == "owner"]
    if owners == [user_id] and role != "owner":
        raise HTTPException(409, "A workspace keeps at least one owner")


_FAILURES_SEEN: dict[tuple[Any, Any], float] = {}


def _first_failure_this_minute(net: Any, agent: Any, now: float | None = None) -> bool:
    """23.3: whether a failed sign-in from this address and browser is the first in a minute (the rest are not recorded)."""
    import time

    now = time.time() if now is None else now
    if len(_FAILURES_SEEN) > 1000:
        for key in [k for k, t in _FAILURES_SEEN.items() if now - t >= 60]:
            del _FAILURES_SEEN[key]
    last = _FAILURES_SEEN.get((net, agent))
    if last is not None and now - last < 60:
        return False
    _FAILURES_SEEN[(net, agent)] = now
    return True


def _auth_event(workspace_id: str | None, kind: str, payload: dict[str, Any], user_id: str | None = None, request: Any = None) -> None:
    """23.3: a sign-in or session event in the record of ``workspace_id`` (a sign-in has no request principal yet: the
    workspace is the user's first), with the user agent hashed and the address cut, never a token or a password."""
    from dclab_core import experience as contract

    from .. import context, experience

    if request is not None and kind in ("auth.sign_in", "auth.sign_in_failed"):
        payload = {**payload, "agent": contract.hash_user_agent(request.headers.get("user-agent")),
                   "net": contract.truncate_ip(getattr(getattr(request, "client", None), "host", None))}
    if kind == "auth.sign_in_failed" and not _first_failure_this_minute(payload.get("net"), payload.get("agent")):
        return  # a flood of wrong passwords is one event a minute per address and browser, not a table of them
    log = None
    if workspace_id:
        services = context.services()
        log = getattr(services, "experience", None) if services is not None and getattr(services, "workspace_id", None) == workspace_id else None
        if log is None and request is not None:  # another workspace's record, opened whole: its stamp is its own (23.1)
            try:
                log = getattr(request.app.state.pool.get(workspace_id), "experience", None)
            except Exception:  # noqa: BLE001 - no pool (a bare router), or the workspace is gone
                log = None
        if log is None:
            try:
                log = experience.PgLog(workspace_id)  # no folder known: stamped operate, never leaves
            except Exception:  # noqa: BLE001 - no database: no record of sign-ins (files have no sign-in)
                log = None
    try:
        experience.record(kind, "person", payload, log=log, **({"user": user_id} if user_id else {}))
    except Exception:  # noqa: BLE001 - never stop a sign-in because the record is away
        pass


def _audit_account(workspace_id: str, kind: str, move: str, user: dict[str, Any], args: dict[str, Any]) -> None:
    """A sign-in, or a role the provider's groups set, in that workspace's audit (no request principal exists yet)."""
    from .. import audit

    audit.record(audit.PgAudit(workspace_id), kind, "human", user.get("name") or user.get("username") or user.get("email"), move=move, status="allowed",
                 user_id=user["id"], args={**args, "member": user["id"]})


def _audit_member(move: str, user_id: str, role: str | None) -> None:
    from .. import audit, context

    services = context.services()
    audit.record(getattr(services, "audit", None), "role_change", "human", move=move, status="allowed",
                 args={"member": user_id, "role": role, "role_label": LABELS.get(role or "")})
    _auth_event(getattr(services, "workspace_id", None), "auth.role_changed", {"member": user_id, "role": role, "by": "owner"})


# ---------------------------------------------------------------------- API tokens (one's own)
@router.get("/api/auth/tokens", response_model=list[Doc])
async def tokens(request: Request):
    accounts, who = _on(request), _signed_in()
    return await asyncio.to_thread(accounts.tokens, who.user_id)


@router.post("/api/auth/tokens", status_code=201, response_model=Doc)
async def new_token(payload: AnyObject, request: Request):
    accounts, who = _on(request), _signed_in()
    token, record = await asyncio.to_thread(accounts.create_token, who.user_id, who.workspace_id, str(_body(payload).get("name") or ""))
    _auth_event(who.workspace_id, "auth.token_created", {"token": record["id"], "name": record.get("name")}, who.user_id)
    return {**record, "token": token, "note": "Shown once: copy it now. It acts as you in this workspace, with your role."}


@router.delete("/api/auth/tokens/{token_id}", status_code=204, response_class=Response)
async def revoke_token(token_id: str, request: Request):
    accounts, who = _on(request), _signed_in()
    if not await asyncio.to_thread(accounts.revoke_token, who.user_id, token_id):
        raise HTTPException(404, "No such token")
    _auth_event(who.workspace_id, "auth.token_revoked", {"token": token_id}, who.user_id)
    return Response(status_code=204)



# ---------------------------------------------------------------------- the company's identity provider (part B)
@router.get("/api/auth/oidc/start", response_class=Response, include_in_schema=False)
async def oidc_start(request: Request):
    """Send the browser to the identity provider; a signed cookie remembers this login for ten minutes."""
    from fastapi.responses import RedirectResponse

    from . import oidc

    if _settings(request).auth != "oidc":
        raise HTTPException(404, "This server does not sign in through an identity provider")
    if _settings(request).cookie_secure and not os.environ.get("DCLAB_OIDC_REDIRECT"):  # behind a TLS proxy the request's own URL says http://
        raise HTTPException(503, "Set DCLAB_OIDC_REDIRECT to this server's https:// callback address")
    try:
        location, cookie = await asyncio.to_thread(oidc.start, str(request.url_for("oidc_callback")))
    except oidc.OidcError as exc:
        raise HTTPException(503, str(exc)) from None
    response = RedirectResponse(location, status_code=302)
    response.set_cookie(oidc.COOKIE, cookie, max_age=600, httponly=True, samesite="lax", secure=bool(_settings(request).cookie_secure),
                        path="/api/auth/oidc")
    return response


@router.get("/api/auth/oidc/callback", response_class=Response, include_in_schema=False)
async def oidc_callback(request: Request, code: str = "", state: str = ""):
    """The provider sends the browser back here: check the login, sign the person in, and open the product."""
    from fastapi.responses import HTMLResponse, RedirectResponse

    from . import oidc

    accounts = _on(request)
    if _settings(request).auth != "oidc":
        raise HTTPException(404, "This server does not sign in through an identity provider")
    try:
        claims = await asyncio.to_thread(oidc.finish, code, state, request.cookies.get(oidc.COOKIE, ""))
        user, roles, changes = await asyncio.to_thread(oidc.sign_in, accounts, claims, request.app.state.services.workspace_id)
    except oidc.OidcError as exc:
        page = HTMLResponse(f"<!doctype html><meta charset=utf-8><title>Sign-in refused</title><p>Sign-in refused: {_html(str(exc))}.</p>"
                            "<p><a href=\"/\">Back to DCLab</a></p>", status_code=403)
        page.delete_cookie(oidc.COOKIE, path="/api/auth/oidc")
        return page
    own = request.app.state.services.workspace_id
    workspace = own if own in roles else next(iter(roles))  # the server's own workspace first, when the person has a role there
    token, _ = await asyncio.to_thread(accounts.start_session, user["id"], workspace)
    for changed, old, new in changes:  # the provider's groups changed a membership: that workspace's audit says so
        _audit_account(changed, "role_change", "set_role_from_groups", user, {"was": old, "role": new, "role_label": LABELS.get(new or "")})
        _auth_event(changed, "auth.role_changed", {"member": user["id"], "was": old, "role": new, "by": "identity_provider"}, user["id"])
    _audit_account(workspace, "sign_in", "sign_in", user, {"via": "identity provider"})
    _auth_event(workspace, "auth.sign_in", {"via": "identity_provider", "session": digest(token)}, user["id"], request)
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(COOKIE, token, max_age=int(SESSION_MAX_DAYS * 86400), httponly=True, samesite="lax",
                        secure=bool(_settings(request).cookie_secure), path="/")
    response.delete_cookie(oidc.COOKIE, path="/api/auth/oidc")
    return response


def _html(text: str) -> str:
    import html

    return html.escape(text)
