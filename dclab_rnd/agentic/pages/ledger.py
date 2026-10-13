"""Decisions: the ledger on the Admin page (packages 19.7, 19.8).

    GET  /api/ledger/summary              how many decisions by point, kind and who decided; labels by source
    GET  /api/ledger/blind/next           the next decision drawn for a blind label: its question and state only
    POST /api/ledger/blind/{record_id}    a blind reviewer's answer, kept with its sampling weight

Only the owner and reviewers read or label them (``accounts.guard``: ``review``). A blind reviewer never sees an
answer, the value used or who decided (``dclab_core.ledger.blind_view``).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from fastapi import Body, HTTPException

from ..api_models import Page, Router

ID = re.compile(r"[0-9a-zA-Z_.:-]{6,80}")


def register(app, ctx) -> None:
    router = Router()

    def store() -> Any:
        return getattr(ctx, "ledger", None)

    @router.get("/api/ledger/summary", response_model=Page)
    async def ledger_summary():
        """Decisions by point, kind and who decided; labels by source; how many are drawn for blind labels."""
        from ...ledger import labels

        return await asyncio.to_thread(labels.summary, store())

    @router.get("/api/ledger/blind/next", response_model=Page)
    async def ledger_blind_next(point: str | None = None):
        """The next decision drawn for a blind label (its question, subject, state and options), or none left."""
        from ...ledger import labels

        if point is not None and not re.fullmatch(r"[a-z_.]{3,60}", point):
            raise HTTPException(422, "point is a decision point")
        item = await asyncio.to_thread(labels.next_blind, store(), "blind-v1", point)
        return {"item": item}

    @router.post("/api/ledger/blind/{record_id}", response_model=Page)
    async def ledger_blind_label(record_id: str, body: Any = Body(None)):
        """A blind reviewer's answer: {"value": ...}; kept with the weight of how the decision was drawn."""
        from ...accounts.principal import current
        from ...ledger import labels

        if not ID.fullmatch(record_id):
            raise HTTPException(422, "record_id is a decision's id")
        value = (body or {}).get("value") if isinstance(body, dict) else None
        if value is None or not isinstance(value, (bool, int, float, str)):
            raise HTTPException(422, "value is a flag, a number or a short text")
        who = current()
        by = f"reviewer:{who.user_id}" if who is not None and who.user_id else "reviewer"
        try:
            out = await asyncio.to_thread(labels.blind_label, record_id, value, by, "blind-v1", store())
        except KeyError:
            raise HTTPException(404, "No such decision") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return {"label": out.model_dump(mode="json")}

    app.include_router(router)
