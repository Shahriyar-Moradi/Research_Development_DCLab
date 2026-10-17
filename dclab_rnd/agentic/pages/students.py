"""Students: the register of the decision models behind every choice, on the Admin page (package 29.1).

    GET /api/students    one row per decision point: recorded decisions, labels by source, its homework (the generated
                         set and its sealed rows), its student (family, dev score beside the rule's), the sealed verdict,
                         the level it serves at, and the labelled disagreements still needed before it may act

The owner's (``accounts.guard``: ``admin``): the register names every choice DCLab makes and how far each is trusted.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..api_models import Page, Router


def register(app, ctx) -> None:
    router = Router()

    def read() -> dict[str, Any]:
        from ... import students

        routing_doc = None
        models = getattr(ctx, "models", None)
        if models is not None and getattr(models, "routing", None) is not None:
            try:
                routing_doc = models.routing.get()
            except Exception:  # noqa: BLE001 - no routing document yet
                routing_doc = None
        return students.read(getattr(ctx, "ledger", None), routing_doc)  # the command's own read: the same numbers

    @router.get("/api/students", response_model=Page)
    async def students_register():
        """The students' register: every decision point and where its student stands in the loop."""
        return await asyncio.to_thread(read)

    app.include_router(router)
