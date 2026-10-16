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

    @router.get("/api/budget", response_model=Page)
    async def my_budget():
        """The person's own month: their model limit (15 unless the owner set another), what they spent, what is left."""
        from ... import limits
        from ...accounts.principal import current
        from ...models.usage import month_start

        who = current()
        if not limits.on() or who is None or not who.user_id:
            return {"on": False, "monthly_eur": limits.limits()["model_eur"]["user"], "note": "Limits apply when accounts are on (DCLAB_AUTH)."}
        usage = getattr(getattr(ctx, "models", None), "usage", None) or getattr(getattr(ctx, "gateway", None), "usage", None)
        spent = float(usage.spent(month_start(), user_id=who.user_id)) if usage is not None else 0.0
        cap = limits.user_monthly(who.user_id, home())
        return {"on": True, "monthly_eur": cap, "spent_eur": round(spent, 4), "left_eur": None if cap is None else round(max(0.0, cap - spent), 4)}

    @router.get("/api/workspace/budgets", response_model=Page)
    async def member_budgets():
        """The people's own monthly model limits the owner set, and the default (15)."""
        from ... import limits

        return {"users": limits.user_budgets(home()), "default": limits.limits()["model_eur"]["user"]}

    @router.put("/api/workspace/budgets/{user_id}", response_model=Page)
    async def set_member_budget(user_id: str, body: Any = Body(None)):
        """The owner sets one person's monthly model limit ({"monthly_eur": n}, or null for the default); audited."""
        import re

        from ... import audit, limits

        if not re.fullmatch(r"[0-9a-zA-Z_-]{1,80}", user_id):
            raise HTTPException(422, "a person's id")
        body = body if isinstance(body, dict) else {}
        if "monthly_eur" not in body:  # a typo never clears a person's limit
            raise HTTPException(422, 'send {"monthly_eur": n}, or {"monthly_eur": null} to go back to the default')
        before = limits.user_budgets(home()).get(user_id)
        try:
            budgets = await asyncio.to_thread(limits.set_user_monthly, home(), user_id, body.get("monthly_eur"))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        audit.record(ctx.audit, "policy", "human", move="set_member_budget", status="allowed",
                     args={"user": user_id, "monthly_eur": budgets.get(user_id), "was": before})
        from ... import experience

        experience.record("decision.accepted", "person", {"point": "budget.member", "choice": {"monthly_eur": budgets.get(user_id)}, "by": "person"})
        return {"users": budgets, "default": limits.limits()["model_eur"]["user"]}

    app.include_router(router)
