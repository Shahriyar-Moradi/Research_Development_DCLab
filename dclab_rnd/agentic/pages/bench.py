"""DCLab Bench on the Admin page (B3 of the 2026-11-03 plan): every evaluation of DCLab by layer, the owner's.

    GET  /api/admin/bench     each layer's state (red, amber, thin, green, empty) with its reasons, the latest result of
                              each of its suites, the recorded runs as history, and the campaigns with no layer

Read only: running the suites is ``python -m dclab_rnd.bench run`` (``make bench``), never a page.
"""

from __future__ import annotations

import asyncio

from ..api_models import Page, Router


def register(app, ctx) -> None:
    router = Router()

    @router.get("/api/admin/bench", response_model=Page)
    async def admin_bench():
        from ... import bench

        return await asyncio.to_thread(bench.summary)

    @router.get("/api/admin/bench/versions", response_model=Page)
    async def admin_bench_versions():
        from ...bench import versions

        return {"agents": await asyncio.to_thread(versions.timeline)}

    @router.get("/api/admin/bench/monitors", response_model=Page)
    async def admin_bench_monitors(days: int = 7):
        from ...bench import monitors

        return await asyncio.to_thread(monitors.summary, ctx.store.home, max(1, min(days, 90)), getattr(ctx.models, "usage", None))

    app.include_router(router)
