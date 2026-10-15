"""Consent and deletion of the record, on the Admin page (package 19.10).

    GET    /api/consent                        the workspace's switches (use for improving DCLab, use for training)
    PUT    /api/consent                        {"use": improve | train, "on": true | false}: the owner's, audited
    DELETE /api/experience/sessions/{visit}    a session's events, with a tombstone in the audit (its decisions go with their project or draft)

A project's switches are its settings (``PATCH /api/projects/{id}`` with ``settings.consent``); deleting a project or
a draft removes what it left in the record and the ledger too (``experience.retention.forget``).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from fastapi import Body, HTTPException, Response

from ..api_models import Page, Router

VISIT = re.compile(r"[0-9a-zA-Z_-]{1,80}")


def register(app, ctx) -> None:
    router = Router()

    def home() -> Any:
        return ctx.store.home

    @router.get("/api/consent", response_model=Page)
    async def read_consent():
        """The workspace's switches; both off until the owner turns them on."""
        from ... import consent

        counts = await asyncio.to_thread(consent.stamped_counts, getattr(ctx, "experience", None))
        return {"workspace": consent.workspace_consent(home()), "uses": list(consent.USES), "notice_version": consent.NOTICE_VERSION,
                "stamped": counts, "level": consent.current_level(home=home())}  # 23.1: what the record holds under each level

    @router.put("/api/consent", response_model=Page)
    async def set_consent(body: Any = Body(None)):
        """{"use": "improve" | "train", "on": bool}: one switch, changed by the owner and audited."""
        from ... import audit, consent

        body = body if isinstance(body, dict) else {}
        use, on = body.get("use"), body.get("on")
        if use not in consent.USES or not isinstance(on, bool):
            raise HTTPException(422, f"use is one of {', '.join(consent.USES)} and on is true or false")
        before = consent.workspace_consent(home())
        after = await asyncio.to_thread(consent.set_workspace_consent, home(), use, on)
        if before.get(use) != on:
            audit.record(ctx.audit, "policy", "human", move="set_consent", status="allowed",
                         args={"policy": f"consent_{use}", "label": audit.POLICY_NAMES[f"consent_{use}"], "on": on, "was": before.get(use)})
            from dclab_core import consent as core

            from ... import experience

            for change in core.changes(before, after, "workspace"):  # 23.1: the history of consent is in the record
                experience.record("consent.changed", "person", change)
            consent.invalidate()
        return {"workspace": after, "uses": list(consent.USES), "notice_version": consent.NOTICE_VERSION}

    @router.delete("/api/experience/sessions/{visit}", status_code=204, response_class=Response)
    async def delete_session(visit: str):
        """Every event a session left; the audit keeps that it was deleted, by whom and when, never what it held. Its
        decisions belong to their project or draft, and are deleted with them."""
        from ... import audit
        from ...experience import retention

        if not VISIT.fullmatch(visit):
            raise HTTPException(422, "visit is a session's id")
        log = getattr(ctx, "experience", None)
        if log is None or not await asyncio.to_thread(lambda: log.read(visit=visit, limit=1)):
            raise HTTPException(404, "No session with this id in this workspace")
        gone = await asyncio.to_thread(retention.forget, visit=visit, log=log)
        audit.record(ctx.audit, "deletion", "human", move="delete_session", status="allowed", visit=visit, **gone)
        return Response(status_code=204)

    app.include_router(router)
