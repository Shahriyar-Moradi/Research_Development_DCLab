"""The evaluation dashboard (package 25.8): Claude and the system, and every model's precision, recall, F1 and accuracy.

    GET  /api/evaluation      the board (every answerer by decision point, against its truth, with n and intervals),
                              the twin matrix (Claude against the system per step, the mode of twin_rule_v1), the sources

The owner's and reviewers' (``accounts.guard``: ``review``), as the decision ledger.
"""

from __future__ import annotations

import asyncio

from ..api_models import Page, Router


def register(app, ctx) -> None:
    router = Router()

    @router.get("/api/evaluation", response_model=Page)
    async def evaluation_board():
        from ... import evaluation

        result = await asyncio.to_thread(evaluation.board, ctx.store.home, getattr(ctx, "ledger", None))
        return {**result, "summary": evaluation.summary(result)}

    app.include_router(router)
