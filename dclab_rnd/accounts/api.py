"""Sign-in, sessions, workspaces, members and API tokens over HTTP (package 10.2).

    POST   /api/auth/password {email, password}      sign in; sets the session cookie (HttpOnly, SameSite=Lax)
    POST   /api/auth/signout                         ends the session
    GET    /api/auth/me                              who is signed in, their workspaces and role
    POST   /api/auth/workspace {workspace_id}        act in another workspace one belongs to
    GET    /api/workspace/members                    the workspace's members (every role)
    POST   /api/workspace/members {email, role, name?, password?}   add a member (the owner)
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
from .principal import current
from .roles import LABELS, ROLES
from .store import SESSION_MAX_DAYS, AccountError, Accounts

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
    user = await asyncio.to_thread(accounts.check_password, str(body.get("email") or ""), str(body.get("password") or ""))
    if user is None:
        await asyncio.sleep(0.3)  # a wrong password costs the guesser time
        raise HTTPException(401, "The email or the password is not right")
    spaces = await asyncio.to_thread(accounts.workspaces_of, user["id"])
    if not spaces:
        raise HTTPException(403, "This account is not a member of any workspace yet")
    token, csrf = await asyncio.to_thread(accounts.start_session, user["id"], spaces[0][0])
    _audit_account(spaces[0][0], "sign_in", "sign_in", user, {"via": "password"})
    response = JSONResponse({"signed_in": True, "csrf": csrf, "workspace_id": spaces[0][0]})
    response.set_cookie(COOKIE, token, max_age=SESSION_MAX_DAYS * 86400, httponly=True, samesite="lax",
                        secure=bool(_settings(request).cookie_secure), path="/")
    return response


@router.post("/api/auth/signout", response_model=Page)
async def sign_out(request: Request):
    token = request.cookies.get(COOKIE)
    if token and _settings(request).auth != "none":
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

    def add():
        found = accounts.by_email(email)
        user = found or accounts.create_user(email, str(body.get("name") or ""), password=body.get("password") or None)
        accounts.set_member(who.workspace_id, user["id"], role)
        return {"id": user["id"], "email": user["email"], "role": role, "role_label": LABELS[role], "created": found is None}
    try:
        added = await asyncio.to_thread(add)
    except (AccountError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None
    _audit_member("add_member", added["id"], role)
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


def _audit_account(workspace_id: str, kind: str, move: str, user: dict[str, Any], args: dict[str, Any]) -> None:
    """A sign-in, or a role the provider's groups set, in that workspace's audit (no request principal exists yet)."""
    from .. import audit

    audit.record(audit.PgAudit(workspace_id), kind, "human", user.get("name") or user.get("email"), move=move, status="allowed",
                 user_id=user["id"], args={**args, "member": user["id"]})


def _audit_member(move: str, user_id: str, role: str | None) -> None:
    from .. import audit, context

    services = context.services()
    audit.record(getattr(services, "audit", None), "role_change", "human", move=move, status="allowed",
                 args={"member": user_id, "role": role, "role_label": LABELS.get(role or "")})


# ---------------------------------------------------------------------- API tokens (one's own)
@router.get("/api/auth/tokens", response_model=list[Doc])
async def tokens(request: Request):
    accounts, who = _on(request), _signed_in()
    return await asyncio.to_thread(accounts.tokens, who.user_id)


@router.post("/api/auth/tokens", status_code=201, response_model=Doc)
async def new_token(payload: AnyObject, request: Request):
    accounts, who = _on(request), _signed_in()
    token, record = await asyncio.to_thread(accounts.create_token, who.user_id, who.workspace_id, str(_body(payload).get("name") or ""))
    return {**record, "token": token, "note": "Shown once: copy it now. It acts as you in this workspace, with your role."}


@router.delete("/api/auth/tokens/{token_id}", status_code=204, response_class=Response)
async def revoke_token(token_id: str, request: Request):
    accounts, who = _on(request), _signed_in()
    if not await asyncio.to_thread(accounts.revoke_token, who.user_id, token_id):
        raise HTTPException(404, "No such token")
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
    _audit_account(workspace, "sign_in", "sign_in", user, {"via": "identity provider"})
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(COOKIE, token, max_age=SESSION_MAX_DAYS * 86400, httponly=True, samesite="lax",
                        secure=bool(_settings(request).cookie_secure), path="/")
    response.delete_cookie(oidc.COOKIE, path="/api/auth/oidc")
    return response


def _html(text: str) -> str:
    import html

    return html.escape(text)
