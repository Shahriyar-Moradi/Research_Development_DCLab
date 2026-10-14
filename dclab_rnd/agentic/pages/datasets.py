"""Datasets: the catalog of the training and evaluation sets on the Admin page (package 19.13).

    GET /api/datasets                  every set `make datasets` built: builder, name, version, consent, rows, splits, size
    GET /api/datasets/lineage?path=    the sessions one set came from

The owner's (``accounts.guard``: ``admin``): the sets hold redacted text from consented projects, never cells.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from fastapi import HTTPException

from ..api_models import Page, Router

PATH = re.compile(r"[a-z_]{3,20}(/[A-Za-z0-9_.]{1,120})?")  # as the catalog lists them: decisions/column.is_identifier, development


def register(app, ctx) -> None:
    router = Router()

    def home() -> Any:
        return ctx.store.home

    @router.get("/api/datasets", response_model=Page)
    async def datasets_catalog():
        """Every dataset of this workspace, newest first, with where it came from in counts."""
        from ...experience import catalog

        items = await asyncio.to_thread(catalog.catalog, home())
        return {"items": items, "total": len(items), "rows": sum(i.get("rows") or 0 for i in items)}

    @router.get("/api/datasets/lineage", response_model=Page)
    async def datasets_lineage(path: str):
        """The sessions (visits) one dataset's sources came from."""
        from ...experience import catalog

        if not PATH.fullmatch(path):
            raise HTTPException(422, "path is a dataset's builder/name, as the catalog lists it")
        try:
            return await asyncio.to_thread(catalog.lineage, home(), path, getattr(ctx, "experience", None), getattr(ctx, "ledger", None))
        except KeyError:
            raise HTTPException(404, "No such dataset in this workspace") from None

    app.include_router(router)
