"""Team threads and the shared handoff history over HTTP (W3 of the collaborative workspace, 2026-11-20).

    GET  /api/team-threads[?project_id=]                                    my team threads (the owner: every thread)
    POST /api/projects/{project_id}/problems/{problem_id}/team-threads {team}  open my team's thread on this problem
    GET  /api/projects/{project_id}/team-threads/{thread_id}[?after=]       the thread and its messages
    POST /api/projects/{project_id}/team-threads/{thread_id}/messages {text, ask?} post to it (W5: an answer to the
                                                                            thread agent's open question reaches it; ``ask``
                                                                            opens a question for that team)
    GET  /api/projects/{project_id}/team-threads/{thread_id}/events         its new messages, as server-sent events
    GET  /api/projects/{project_id}/problems/{problem_id}/handoffs          the problem's shared handoff history

A thread is its team's (``teamspace/threads.py``, :func:`threads.access`): its members read it and, when their role may
write, post to it; the owner reads every one. Anyone else, and any thread of another project or workspace, gets 403 (a
thread that does not exist gets the same answer, so a refusal says nothing about what exists). Every route first checks
the project and problem as the memory routes do (``memory.scope.resolve_scope``: the workspace, a project's own member
list). The handoff history is for every reader of the project: who asked whom and what came back, as the owner writes it
(``1. Business → Data Science · Can we rank accounts …?``), never a thread's own messages. The routes' permissions are in
``accounts/guard.SPECIAL``. Without sign-in the local owner reads and posts everywhere, as on every other route.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

from fastapi import Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...memory import scope as mem_scope
from ...teamspace import stream, threads as th
from ..api_models import HandoffHistoryDoc, Router, TeamMessageDoc, TeamMessagePost, TeamThreadDoc, TeamThreadOpen, TeamThreadView
from ..services import Services, services

router = Router()
REFUSED = "Not your team's thread"


def _who() -> Any:
    from ...accounts.principal import LOCAL_OWNER, current

    return current() or LOCAL_OWNER


def _auth_on() -> bool:
    return mem_scope.auth_enabled()


def _threads(s: Services) -> Any:
    return th.open_threads(s.store.home)


def _teams(s: Services) -> Any:
    from ...storage import open_teams

    return open_teams(s.store.home)


def _scope(s: Services, project_id: str, problem_id: Optional[str] = None, *, write: bool = False) -> mem_scope.AuthorizedScope:
    """The project (and problem) as this person may reach them: 403 otherwise, whatever the reason."""
    try:
        return mem_scope.resolve_scope(workspace_id=mem_scope.workspace_of(s.projects), projects=s.projects, drafts=s.drafts,
                                       project_id=project_id, problem_id=problem_id, require_write=write)
    except (mem_scope.AccessDenied, mem_scope.AmbiguousProblemError, KeyError) as exc:
        detail = "Not yours to change" if write and "write permission" in str(exc) else "Not yours to read"
        raise HTTPException(403, detail) from None


def _thread(s: Services, project_id: str, thread_id: str, *, write: bool = False) -> tuple[Any, dict[str, Any], str]:
    """The store, the thread and what this person may do on it; 403 for another project's, another workspace's, a missing
    one, or one that is not their team's."""
    try:
        _scope(s, project_id, write=write)  # the project first: an outsider learns nothing about which threads exist
    except HTTPException as refused:
        raise HTTPException(403, REFUSED if refused.detail == "Not yours to read" else refused.detail) from None
    store = _threads(s)
    thread = store.thread(thread_id)
    if thread is None or thread["project_id"] != project_id:
        raise HTTPException(403, REFUSED)
    _scope(s, project_id, thread["problem_id"], write=write)
    can = th.access(thread, _who(), _teams(s), auth_on=_auth_on())
    if can is None or (write and can != "write"):
        raise HTTPException(403, REFUSED if can is None else "You may read this thread, not post to it")
    return store, thread, can


def _recheck_seconds() -> float:
    try:
        return max(0.05, float(os.environ.get("DCLAB_TEAM_THREAD_RECHECK_SECONDS") or 15.0))
    except ValueError:
        return 15.0


def _author() -> dict[str, Any]:
    who = _who()
    return {"kind": "person", "id": str(who.user_id or "owner"), "name": who.who}


def _live(who: Any) -> Any:
    """The reader as the accounts database knows them now (their live role in the workspace), for a recheck that outlives
    the request: a person removed or demoted since the stream opened is checked with the role they have now. Without
    sign-in the local owner stays as they are; with it, no role found (or no database to ask) refuses."""
    import dataclasses

    if not _auth_on():
        return who
    checker = mem_scope.live_membership()
    role = checker(str(who.user_id), str(who.workspace_id)) if checker is not None and who.user_id and who.workspace_id else None
    if role is None:
        raise mem_scope.AccessDenied("no live membership")
    return dataclasses.replace(who, role=role)


@router.get("/api/team-threads", response_model=list[TeamThreadDoc])
async def my_team_threads(project_id: Optional[str] = None, s: Services = Depends(services)):
    """The threads of my teams (every thread for the owner, and without sign-in), oldest first, each with what I may do."""
    def run() -> list[dict[str, Any]]:
        who, auth_on, teams = _who(), _auth_on(), _teams(s)
        store = _threads(s)
        if not auth_on or who.role == "owner":
            found = store.threads(project_id=project_id)
        else:
            mine = [t["id"] for t in teams.teams_of(str(who.user_id))] if who.user_id else []
            found = store.threads(project_id=project_id, teams=mine)
        allowed: dict[str, bool] = {}
        out = []
        for thread in found:
            pid = thread["project_id"]
            if pid not in allowed:
                try:
                    _scope(s, pid)
                    allowed[pid] = True
                except HTTPException:
                    allowed[pid] = False
            can = th.access(thread, who, teams, auth_on=auth_on) if allowed[pid] else None
            if can is not None:
                out.append({**thread, "access": can})
        return out
    return await asyncio.to_thread(run)


@router.post("/api/projects/{project_id}/problems/{problem_id}/team-threads", response_model=TeamThreadDoc)
async def open_team_thread(project_id: str, problem_id: str, payload: TeamThreadOpen, s: Services = Depends(services)):
    """My team's thread on this problem, opened the first time (``opened`` says whether it was opened now). ``team`` is a
    team's id (GET /api/workspace/teams), or a team agent's name (business, data_science, …)."""
    from dclab_core import team_work as tw

    name = str(payload.team or "").strip()
    if not name:
        raise HTTPException(422, "team is a team's id (GET /api/workspace/teams)")

    def run() -> dict[str, Any]:
        _scope(s, project_id, problem_id, write=True)
        teams = _teams(s)
        found = teams.team(name)
        if found is not None:
            key, label = found["id"], found["name"]
        elif name in tw.DIRECTORY or any(t.get("kind") == name for t in teams.teams()):
            key, label = th.team_key(teams, name)
        else:
            raise HTTPException(404, "No such team")
        if th.access({"team": key}, _who(), teams, auth_on=_auth_on()) != "write":
            raise HTTPException(403, "Only the team's members may open its thread")
        thread, made = _threads(s).open_thread(project_id, problem_id, key, label=label, by=_author())
        if made:
            th.record_event("thread.opened", thread, {"kind": "person", "team": key}, "open")
        return {**thread, "access": "write", "opened": made}
    return await asyncio.to_thread(run)


@router.get("/api/projects/{project_id}/team-threads/{thread_id}", response_model=TeamThreadView)
async def read_team_thread(project_id: str, thread_id: str, after: int = 0, s: Services = Depends(services)):
    """The thread and its messages after ``after`` (whole, oldest first)."""
    def run() -> dict[str, Any]:
        store, thread, can = _thread(s, project_id, thread_id)
        return {"thread": {**thread, "access": can}, "messages": store.messages(thread_id, max(0, int(after))), "access": can}
    return await asyncio.to_thread(run)


@router.post("/api/projects/{project_id}/team-threads/{thread_id}/messages", status_code=201, response_model=TeamMessageDoc)
async def post_team_message(project_id: str, thread_id: str, payload: TeamMessagePost, s: Services = Depends(services)):
    """A message from me to my team's thread, kept whole (too long a message is refused, never cut)."""
    text = payload.text if isinstance(payload.text, str) else ""
    if not text.strip():
        raise HTTPException(422, "text is what you say")
    if len(text) > th.TEXT_MAX:
        raise HTTPException(422, f"A message is at most {th.TEXT_MAX} characters: nothing was cut or saved")

    def run() -> dict[str, Any]:
        store, thread, _ = _thread(s, project_id, thread_id, write=True)
        author = {**_author(), "team": thread["team"]}
        message, made = store.post(thread_id, author, text)
        if made:
            _to_agent(s, thread, message, text, ask=(str(payload.ask).strip() or None) if payload.ask else None)
        return message
    return await asyncio.to_thread(run)


def _to_agent(s: Services, thread: dict[str, Any], message: dict[str, Any], text: str, ask: Optional[str] = None) -> None:
    """W5: the thread's team agent takes the person's message: in a Business thread, their answer to an open question asked
    of them there, or (with ``ask``) a new question for that team. The message is saved first and stays whatever the agent
    does; a refusal is the agent's record (its run), never this route's error."""
    from ...teamspace import agents

    who = _who()
    teams = _teams(s)
    found = teams.team(str(thread["team"]))
    kind = found.get("kind") if found is not None else str(thread["team"])
    initiator = {"user_id": str(who.user_id), "role": str(who.role), "workspace_id": str(who.workspace_id or mem_scope.workspace_of(s.projects))} \
        if who.user_id else None
    agents.from_thread(s, thread, person=str(who.user_id or "owner"), text=text, initiator=initiator, message_id=str(message["id"]),
                       team_kind=kind, ask=ask)


@router.get("/api/projects/{project_id}/team-threads/{thread_id}/events", response_class=StreamingResponse,
            responses={200: {"content": {"text/event-stream": {}}, "description": "The thread's new messages, as server-sent events"}})
async def team_thread_events(project_id: str, thread_id: str, request: Request, after: int = 0, wait: float = stream.STREAM_SECONDS,
                             s: Services = Depends(services)):
    """Every message after Last-Event-ID (or ``after``), then each new one as it is posted (``event: message``, ``id``: its
    number). The access is checked when the stream opens and again every ``DCLAB_TEAM_THREAD_RECHECK_SECONDS`` (15) while
    it is open: a member who left the team (or the project) stops receiving within that time."""
    store, _thread_doc, _can = await asyncio.to_thread(_thread, s, project_id, thread_id)
    listening = store.wait_for_events()
    every = _recheck_seconds()
    checked, who = {"at": time.monotonic()}, _who()  # the reader, kept: the body is sent after this request's context ends

    def read(seq: int) -> list[dict[str, Any]]:
        """The new messages; the reader's access checked again every few seconds, the stream ended when it is gone."""
        if time.monotonic() - checked["at"] >= every:
            from ...accounts.principal import acting

            try:
                with acting(_live(who)):
                    _thread(s, project_id, thread_id)
            except (HTTPException, mem_scope.AccessDenied):
                raise stream.Closed() from None
            checked["at"] = time.monotonic()
        return store.messages(thread_id, seq)
    body = stream.numbered_events(request, read, start=stream.start_from(request, after),
                                  wait=wait, listening=listening, key=th.NOTIFY_PREFIX + thread_id, kind="message")
    return StreamingResponse(body, media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})


@router.get("/api/projects/{project_id}/problems/{problem_id}/handoffs", response_model=HandoffHistoryDoc)
async def problem_handoffs(project_id: str, problem_id: str, s: Services = Depends(services)):
    """The problem's shared handoff history, for every reader of the project: one entry per thing that travelled between
    teams (``n``, ``from_team``, ``to_team``, ``text``, ``state``) and the same as lines, in the owner's words."""
    def run() -> dict[str, Any]:
        from ...teamspace import history_lines
        from ...teamspace.store import open_team_work

        _scope(s, project_id, problem_id)
        entries = open_team_work(s.store.home, projects=s.projects).handoff_history(project_id, problem_id)
        return {"project_id": project_id, "problem_id": problem_id, "count": len(entries),
                "title": f"Shared handoff history · {len(entries)}", "entries": entries, "lines": history_lines(entries)}
    return await asyncio.to_thread(run)
