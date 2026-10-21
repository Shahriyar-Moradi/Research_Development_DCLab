"""Claude beside the system (phase 3, package 25.4): the owner's switch and the twin sessions.

    GET  /api/twin/settings          the workspace's switch: on, sample, session_eur, monthly_eur (all off by default)
    PUT  /api/twin/settings          {"on": bool, "sample": bool, "session_eur": n, "monthly_eur": n}: the owner's, audited
    GET  /api/twin/sessions          the Home sessions Claude ran beside (newest first), with what each spent
    GET  /api/budget                 25.10: the signed-in person's monthly model limit, what they spent, what is left
    PUT  /api/workspace/budgets/{user_id}   {"monthly_eur": n | null}: the owner sets one person's monthly limit; audited
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Body, HTTPException

from ..api_models import Page, Router


def register(app, ctx) -> None:
    router = Router()

    def home() -> Any:
        return ctx.store.home

    @router.get("/api/twin/settings", response_model=Page)
    async def twin_settings():
        from ...twin import home as twin_home

        return {"settings": twin_home.settings(home()), "defaults": twin_home.DEFAULTS, "limits": twin_home.LIMITS}

    @router.put("/api/twin/settings", response_model=Page)
    async def set_twin_settings(body: Any = Body(None)):
        from ... import audit
        from ...twin import home as twin_home

        body = body if isinstance(body, dict) else {}
        before = twin_home.settings(home())
        try:
            after = await asyncio.to_thread(twin_home.set_settings, home(), body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        changed = {k: after[k] for k in after if before.get(k) != after[k]}
        if changed:
            from ...experience import hooks

            audit.record(ctx.audit, "policy", "human", move="set_twin", status="allowed", args={"policy": "twin", "changed": changed,
                                                                                               "was": {k: before.get(k) for k in changed}})
            hooks.settings_changed({f"twin_{k}": v for k, v in changed.items()})
        return {"settings": after}

    @router.get("/api/twin/sessions", response_model=Page)
    async def twin_sessions(limit: int = 100):
        from ...twin import home as twin_home

        rows = await asyncio.to_thread(twin_home.Sessions(home()).list, max(1, min(limit, 500)))
        return {"sessions": rows, "settings": twin_home.settings(home())}

    app.include_router(router)
