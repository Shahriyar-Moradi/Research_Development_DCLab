"""The teams on the project and Admin pages (package 28.8).

    GET  /api/projects/{id}/team   the solution team on this project as it stands: its last board, the pause waiting for
                                   the person, its last record; nothing runs
    POST /api/projects/{id}/team   start the solution team, or resume it (``{"answer": ...}`` or ``{"approve": true|false}``):
                                   it runs until its next pause or its end
    GET  /api/teams                every finished walk of the workspace for the Admin trace view: team, status, nodes,
                                   hand-offs, pauses, cost, the models asked (the owner's: ``accounts.guard``)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import Body, HTTPException

from ..api_models import AnyObject, Doc, Page, Router


def register(app, ctx) -> None:
    router = Router()

    def home() -> Path:
        from ... import context

        live = context.services() or ctx
        store = getattr(live, "store", None)
        if store is not None and getattr(store, "home", None) is not None:
            return Path(store.home)
        from ...settings import Settings

        return Path(Settings().agent_home)

    def projects():
        from ... import context

        return getattr(context.services() or ctx, "projects", None) or ctx.projects

    def project(project_id: str) -> dict[str, Any]:
        try:
            return projects().get(project_id)
        except KeyError:
            raise HTTPException(404, "Project not found") from None

    @router.get("/api/projects/{project_id}/team", response_model=Doc)
    async def project_team(project_id: str):
        """The solution team on this project as it stands; nothing runs."""
        project(project_id)
        from ...orchestra import service as team

        return await asyncio.to_thread(team.view, home(), f"project:{project_id}")

    @router.post("/api/projects/{project_id}/team", response_model=Doc)
    async def project_team_walk(project_id: str, payload: AnyObject | None = Body(None)):
        """Start the solution team, or resume it with the person's answer; until its next pause or its end."""
        p = project(project_id)
        if p.get("running"):
            raise HTTPException(409, f"{p['running']} is running on this project: try again when it is done")
        body = payload.model_dump(exclude_unset=True) if payload is not None else {}
        from ...accounts.guard import demand
        from ...orchestra import service as team

        thread = f"project:{project_id}"
        try:
            waiting = await asyncio.to_thread(team.pending, home(), thread)
            if "approve" in body and waiting is not None and waiting.get("kind") == "approval" and waiting.get("gate") == "holdout":
                demand("approve_gate", "approve reading the holdout")  # the gate's own permission (the review of 28.8): an owner or a reviewer
            resume = await asyncio.to_thread(team.resume_value, home(), thread, body)
            return await asyncio.to_thread(team.walk_project, home(), projects(), project_id, resume)
        except team.WrongAnswer as exc:
            raise HTTPException(422, str(exc)[:300]) from None
        except team.Busy as exc:
            raise HTTPException(409, str(exc)[:300]) from None
        except ValueError as exc:  # nothing to resume
            raise HTTPException(409, str(exc)[:300]) from None

    @router.get("/api/teams", response_model=Page)
    async def team_walks():
        """Every finished walk of both teams, newest first: the Admin trace view."""
        from ...orchestra import service as team

        return await asyncio.to_thread(team.walks, home())

    app.include_router(router)
