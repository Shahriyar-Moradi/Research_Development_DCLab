"""The people's monthly model budgets (with accounts on): a person's own month, and the limits the owner sets per member.

    GET /api/budget                         the person's own month: limit, spent, left
    GET /api/workspace/budgets              the limits the owner set, and the default
    PUT /api/workspace/budgets/{user_id}    the owner sets one person's limit ({"monthly_eur": n} or null); audited

Moved out of the twin's page (2026-10-21): budgets belong to accounts and every edition has them.
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
