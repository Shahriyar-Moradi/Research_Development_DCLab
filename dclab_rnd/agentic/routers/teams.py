"""Teams over HTTP (W2 of the collaborative workspace): the owner makes teams and places people in them.

    GET    /api/workspace/teams[?user_id=]                     the workspace's teams with their members (every role); one person's
    POST   /api/workspace/teams {name, kind}                   a new team (the owner)
    POST   /api/workspace/teams/{team_id}/members {user_id}    a member of this workspace joins it (the owner)
    DELETE /api/workspace/teams/{team_id}/members/{user_id}    they leave it (the owner)
    PATCH  /api/workspace/teams/{team_id} {name}               a new name (the owner; W7)
    DELETE /api/workspace/teams/{team_id}                      the team removed with its memberships (the owner; W7)

A team says whom an agent represents; a role says what a person may do (``accounts/roles.py``), and joining a team
changes no role. A member is a person with an account in this workspace, so placing people needs sign-in
(``DCLAB_AUTH``); without it the local owner is the only person, and teams are still made for the agents. Every change is
in the workspace's audit trail. The routes' permissions are in ``accounts/guard.SPECIAL``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.responses import Response

from ...storage.teams import KINDS, BrokenTeamsFile, NoSuchTeam, TeamError, open_teams
from ..api_models import Router, TeamCreate, TeamDoc, TeamMemberAdd, TeamMembershipDoc, TeamRename
from ..services import Services, services

router = Router()


def _store(s: Services) -> Any:
    return open_teams(s.store.home)


EVENTS = {"team_create": "team.created", "team_rename": "team.renamed", "team_delete": "team.removed",
          "team_add_member": "team.member_added", "team_remove_member": "team.member_removed"}


def _audit(s: Services, move: str, args: dict[str, Any]) -> None:
    """A team change in the workspace's audit, beside the members' role changes (who is placed where is a membership), and
    its W7 event (ids and kinds only: a team's name stays in the audit, never in the event; a kind outside ``KINDS`` is
    free text the owner typed, so the event says "other")."""
    from ... import audit, experience
    from ...accounts.principal import current

    who = current()
    by = getattr(who, "user_id", None) or "owner"
    audit.record(getattr(s, "audit", None), "role_change", "human", move=move, status="allowed", args={**args, "by": by})
    kind = args.get("kind")
    if kind is not None and kind not in KINDS:  # a kind the owner typed stays in the audit; the event keeps a known one
        kind = "other"
    payload = {"team": args.get("team"), "by": by, "kind": kind, "member": args.get("member"), "members": args.get("members")}
    experience.record(EVENTS[move], "person", {k: v for k, v in payload.items() if v is not None})


def _member_of_workspace(request: Request, user_id: str) -> None:
    """The person must be a member of the workspace the request acts in (404 otherwise, as for the members routes)."""
    from ...accounts.principal import current
    from ...accounts.store import Accounts

    if request.app.state.settings.auth == "none":
        raise HTTPException(409, "Team members are people with accounts: switch sign-in on (DCLAB_AUTH) and add them as members first")
    who = current()
    if who is None or not who.workspace_id or Accounts().role_of(user_id, who.workspace_id) is None:
        raise HTTPException(404, "Not a member of this workspace")


@router.get("/api/workspace/teams", response_model=list[TeamDoc])
async def list_teams(user_id: str | None = None, s: Services = Depends(services)):
    store = _store(s)
    try:
        return await asyncio.to_thread(store.teams_of, user_id) if user_id else await asyncio.to_thread(store.teams)
    except BrokenTeamsFile as exc:
        raise HTTPException(503, str(exc)) from None


@router.post("/api/workspace/teams", status_code=201, response_model=TeamDoc)
async def create_team(payload: TeamCreate, s: Services = Depends(services)):
    try:
        team = await asyncio.to_thread(_store(s).create_team, payload.name, payload.kind)
    except BrokenTeamsFile as exc:
        raise HTTPException(503, str(exc)) from None
    except TeamError as exc:
        raise HTTPException(409 if "already has" in str(exc) else 422, str(exc)) from None
    _audit(s, "team_create", {"team": team["id"], "name": team["name"], "kind": team["kind"]})
    return team


@router.patch("/api/workspace/teams/{team_id}", response_model=TeamDoc)
async def rename_team(team_id: str, payload: TeamRename, s: Services = Depends(services)):
    try:
        team = await asyncio.to_thread(_store(s).rename_team, team_id, payload.name)
    except NoSuchTeam:
        raise HTTPException(404, "No such team") from None
    except BrokenTeamsFile as exc:
        raise HTTPException(503, str(exc)) from None
    except TeamError as exc:
        raise HTTPException(409 if "already has" in str(exc) else 422, str(exc)) from None
    _audit(s, "team_rename", {"team": team_id, "name": team["name"]})
    return team


@router.delete("/api/workspace/teams/{team_id}", status_code=204, response_class=Response)
async def delete_team(team_id: str, s: Services = Depends(services)):
    """The team and its memberships removed. Its threads and its memory stay in the workspace's records (the store keeps
    what was written), but no route shows a removed team's memory any more (they answer 404). Refused (409) while the
    team has open work, or a row still on its way to or from it (a reply queued with its completed task): it would land
    in a thread no member can read any more. The check and the removal hold the team work's lock, so no task is made
    between them (``teamspace.store.remove_team_if_idle``)."""
    from ...storage import open_team_work
    from ...teamspace.store import TeamBusy

    store = _store(s)
    try:
        team = await asyncio.to_thread(store.team, team_id)
        if team is None:
            raise NoSuchTeam(team_id)
        names = [team_id] + ([team["kind"]] if [t["id"] for t in await asyncio.to_thread(store.teams) if t["kind"] == team["kind"]] == [team_id] else [])
        work = open_team_work(s.store.home, projects=s.projects)
        members = await asyncio.to_thread(work.remove_team_if_idle, names, lambda: store.delete_team(team_id))
    except TeamBusy as busy:
        raise HTTPException(409, f"{team['name']} still has open work or deliveries on their way ({busy}): "
                                 "finish it, hand it to another team, or wait for its deliveries first") from None
    except TimeoutError as exc:
        raise HTTPException(409, f"{exc}; try again") from None
    except NoSuchTeam:
        raise HTTPException(404, "No such team") from None
    except BrokenTeamsFile as exc:
        raise HTTPException(503, str(exc)) from None
    _audit(s, "team_delete", {"team": team_id, "members": members})
    return Response(status_code=204)


@router.post("/api/workspace/teams/{team_id}/members", status_code=201, response_model=TeamMembershipDoc)
async def add_team_member(team_id: str, payload: TeamMemberAdd, request: Request, s: Services = Depends(services)):
    user_id = str(payload.user_id or "").strip()
    if not user_id:
        raise HTTPException(422, "user_id is the member's id (GET /api/workspace/members)")
    store = _store(s)
    try:
        if await asyncio.to_thread(store.team, team_id) is None:
            raise NoSuchTeam(team_id)
        await asyncio.to_thread(_member_of_workspace, request, user_id)
        row, added = await asyncio.to_thread(store.add_member, team_id, user_id)
    except NoSuchTeam:
        raise HTTPException(404, "No such team") from None
    except BrokenTeamsFile as exc:
        raise HTTPException(503, str(exc)) from None
    except TeamError as exc:  # no account, or not a member of this workspace (checked again in the insert's transaction)
        raise HTTPException(404, str(exc)) from None
    if added:
        _audit(s, "team_add_member", {"team": team_id, "member": user_id})
    return {**row, "added": added}


@router.delete("/api/workspace/teams/{team_id}/members/{user_id}", status_code=204, response_class=Response)
async def remove_team_member(team_id: str, user_id: str, s: Services = Depends(services)):
    try:
        removed = await asyncio.to_thread(_store(s).remove_member, team_id, user_id)
    except NoSuchTeam:
        raise HTTPException(404, "No such team") from None
    except BrokenTeamsFile as exc:
        raise HTTPException(503, str(exc)) from None
    if not removed:
        raise HTTPException(404, "Not in this team")
    _audit(s, "team_remove_member", {"team": team_id, "member": user_id})
    return Response(status_code=204)
