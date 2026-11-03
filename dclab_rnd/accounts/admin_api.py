"""The Admin area's own sign-in over HTTP (A1 of the 2026-11-03 plan; ``accounts.admin_login``).

    GET    /api/admin/me                          whether an admin login is set, whether this browser is signed in, a lock's end
    POST   /api/admin/signin {username, password}  a session in the ``dclab_admin`` cookie (HttpOnly, SameSite=Strict)
    POST   /api/admin/signout                     ends this browser's admin session
    POST   /api/admin/password {current, new}     a new password; every admin session ends (sign in again)

Every sign-in, failure, lock, sign-out and password change is in the workspace's audit trail (kind ``admin``); the
password and the token never are. A browser's write carries the page's request token as every other write does.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .. import audit, context
from ..agentic.api_models import AnyObject, Page, Router
from . import admin_login

router = Router()


def _home(request: Request):
    return request.app.state.settings.agent_home


def _body(payload: Any) -> dict[str, Any]:
    return payload.model_dump(exclude_unset=True) if payload is not None else {}


def _audit(move: str, status: str = "allowed", **values: Any) -> None:
    audit.record(getattr(context.services(), "audit", None), "admin", "human", "Owner", move=move, status=status, **values)


def _cookie(response: JSONResponse, request: Request, token: str | None) -> None:
    secure = bool(request.app.state.settings.cookie_secure)
    if token is None:
        response.delete_cookie(admin_login.COOKIE, path="/", httponly=True, samesite="strict", secure=secure)
    else:
        response.set_cookie(admin_login.COOKIE, token, max_age=admin_login.MAX_HOURS * 3600, httponly=True, samesite="strict",
                            secure=secure, path="/")


def _accounts_on(request: Request) -> bool:
    """With DCLAB_AUTH the owner role is the gate: a leftover admin login file neither asks nor answers."""
    return request.app.state.settings.auth != "none"


@router.get("/api/admin/me", response_model=Page)
async def admin_me(request: Request):
    if _accounts_on(request):
        return {"required": False, "signed_in": False, "username": None, "locked_until": None, "accounts": True}
    return await asyncio.to_thread(admin_login.state, _home(request), request.cookies.get(admin_login.COOKIE))


@router.post("/api/admin/signin", response_model=Page)
async def admin_signin(payload: AnyObject, request: Request):
    if _accounts_on(request):
        raise HTTPException(404, "This server signs in with accounts (DCLAB_AUTH): the owner role guards the Admin area")
    body = _body(payload)
    try:
        token = await asyncio.to_thread(admin_login.sign_in, _home(request), str(body.get("username") or ""), str(body.get("password") or ""))
    except admin_login.Locked as locked:
        _audit("admin_sign_in", "refused", reason="locked", until=locked.until)
        raise HTTPException(429, {"error": "locked", "message": f"Too many wrong passwords: the Admin sign-in opens again at {locked.until}. On this machine, `python -m dclab_rnd.accounts admin-set` sets a new password at once", "until": locked.until})
    except ValueError as exc:
        await asyncio.sleep(0.3)  # a wrong password costs the guesser time
        _audit("admin_sign_in", "refused", reason="bad_credentials" if "wrong" in str(exc) else "no_login")
        raise HTTPException(401, "The username or the password is not right" if "wrong" in str(exc) else str(exc))
    _audit("admin_sign_in", "allowed")
    response = JSONResponse({"signed_in": True})
    _cookie(response, request, token)
    return response


@router.post("/api/admin/signout", response_model=Page)
async def admin_signout(request: Request):
    await asyncio.to_thread(admin_login.sign_out, _home(request), request.cookies.get(admin_login.COOKIE))
    _audit("admin_sign_out")
    response = JSONResponse({"signed_in": False})
    _cookie(response, request, None)
    return response


@router.post("/api/admin/password", response_model=Page)
async def admin_password(payload: AnyObject, request: Request):
    body = _body(payload)
    try:
        await asyncio.to_thread(admin_login.change_password, _home(request), str(body.get("current") or ""), str(body.get("new") or ""))
    except ValueError as exc:
        _audit("admin_password", "refused", reason=str(exc)[:80])
        raise HTTPException(422, str(exc))
    _audit("admin_password", "allowed")
    response = JSONResponse({"changed": True, "signed_in": False})
    _cookie(response, request, None)
    return response
