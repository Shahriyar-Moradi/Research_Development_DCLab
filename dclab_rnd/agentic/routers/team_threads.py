"""Team threads and the shared handoff history over HTTP (W3 of the collaborative workspace, 2026-11-20).

    GET  /api/team-threads[?project_id=]                                    my team threads (the owner: every thread)
    POST /api/projects/{project_id}/problems/{problem_id}/team-threads {team}  open my team's thread on this problem
    GET  /api/projects/{project_id}/team-threads/{thread_id}[?after=]       the thread and its messages
    POST /api/projects/{project_id}/team-threads/{thread_id}/messages {text, ask?} post to it (W5: an answer to the
                                                                            thread agent's open question reaches it; ``ask``
                                                                            opens a question for that team)
    GET  /api/projects/{project_id}/team-threads/{thread_id}/events         its new messages, as server-sent events
    GET  /api/projects/{project_id}/problems/{problem_id}/handoffs          the problem's shared handoff history
    GET  /api/projects/{project_id}/team-tasks/{task_id}                    W8: a task's page (actions, reports, next owner)
    GET  /api/team-agents/{name}                                            W8: a team agent's job description

A thread is its team's (``teamspace/threads.py``, :func:`threads.access`): its members read it and, when their role may
write, post to it; the owner reads every one. Anyone else, and any thread of another project or workspace, gets 403 (a
thread that does not exist gets the same answer, so a refusal says nothing about what exists). Every route first checks
the project and problem as the memory routes do (``memory.scope.resolve_scope``: the workspace, a project's own member
list). The handoff history is for every reader of the project: who asked whom and what came back, as the owner writes it
(``1. Business → Data Science · Can we rank accounts …?``), never a thread's own messages. A task's page (W8) is read
like a thread: by a member of a team on the task's chain (who asked it, who owns it, and the same of every task above
it, so Business opens the specialist tasks its brief joined) or the owner; its deliveries are shown by name and state,
never their words. The routes' permissions are in
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
from ..api_models import (HandoffHistoryDoc, Router, TeamAgentJobDoc, TeamBrainDoc, TeamBrainsDoc, TeamBrainSet, TeamMessageDoc,
                          TeamMessagePost, TeamTaskView, TeamThreadDoc, TeamThreadOpen, TeamThreadView, WorkspaceOverview)
from ..services import Services, services

router = Router()
REFUSED = "Not your team's thread"
TASK_REFUSED = "Not your team's task"
CHAIN_MAX = 20  # how many tasks above a task its readers are looked for in


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
                                       project_id=project_id, problem_id=problem_id, require_write=write,
                                       teams=mem_scope.teams_store(s.store.home))  # W6: the person's teams; a thread reads team_chain(its team) only
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
            found += [t for t in store.threads(project_id=project_id) if th.is_home(t)]  # 2026-11-21: their Home conversations with the Lead
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

    task_id = str(payload.task_id).strip() if isinstance(payload.task_id, str) and payload.task_id.strip() else None
    response_id = str(payload.response_id).strip()[:80] if isinstance(payload.response_id, str) and payload.response_id.strip() else None

    def run() -> dict[str, Any]:
        from ...teamspace.tools import safe_id

        store, thread, _ = _thread(s, project_id, thread_id, write=True)
        author = {**_author(), "team": thread["team"]}
        key = f"person:{thread_id}:{author.get('id') or 'owner'}:{safe_id(response_id)}" if response_id else None
        message, made = store.post(thread_id, author, text, key=key)
        if made or response_id:  # the same answer again: the agent's turn is idempotent on it and resumes where it stopped
            _to_agent(s, thread, message, str(message["text"]), ask=(str(payload.ask).strip() or None) if payload.ask else None,
                      task_id=task_id, response_id=response_id, resend=not made)  # the words as saved, never a resend's
        return message
    return await asyncio.to_thread(run)


def _to_agent(s: Services, thread: dict[str, Any], message: dict[str, Any], text: str, ask: Optional[str] = None,
              task_id: Optional[str] = None, response_id: Optional[str] = None, resend: bool = False) -> None:
    """W5: the thread's team agent takes the person's message: in a Business thread, their answer to an open question asked
    of them there, or (with ``ask``) a new question for that team; T3: in a Sales or Marketing thread, a member's answer to
    the question its agent asked the team. The message is saved first and stays whatever the agent does; a refusal is the
    agent's record (its run), never this route's error."""
    from ...teamspace import agents

    who = _who()
    teams = _teams(s)
    found = teams.team(str(thread["team"]))
    kind = found.get("kind") if found is not None else str(thread["team"])
    initiator = {"user_id": str(who.user_id), "role": str(who.role), "workspace_id": str(who.workspace_id or mem_scope.workspace_of(s.projects))} \
        if who.user_id else None
    agents.from_thread(s, thread, person=str(who.user_id or "owner"), text=text, initiator=initiator, message_id=str(message["id"]),
                       team_kind=kind, ask=ask, task_id=task_id, response_id=response_id, resend=resend)


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


# ---------------------------------------------------------------------------------------------------------- W8: the screens' reads
def _chain_teams(work: Any, project_id: str, task: Any) -> list[str]:
    """The teams on a task's chain: who asked it and who owns it, and the same of every task above it."""
    names: list[str] = []
    seen: set[str] = set()
    current = task
    for _ in range(CHAIN_MAX):
        for name in (current.requester.team, current.owner_team):
            if name not in names:
                names.append(name)
        parent = current.parent_task_id
        if not parent or parent in seen:
            break
        seen.add(parent)
        current = work.task(project_id, parent)
        if current is None:
            break
    return names


def _asked_from_my_home(work: Any, project_id: str, task: Any, who: Any, teams: Any, auth_on: bool, s: Services) -> bool:
    """Whether a task on the chain was asked from a Home conversation this person may read (2026-11-21: the person who asked
    the Lead on Home reads the Lead's task and the teams' tasks below it, as a team reads its own chain)."""
    store, seen, current = _threads(s), set(), task
    for _ in range(CHAIN_MAX):
        if current.requester.team == th.HOME:
            thread = store.thread(current.requester.thread_id)
            return thread is not None and th.access(thread, who, teams, auth_on=auth_on) is not None
        parent = current.parent_task_id
        if not parent or parent in seen:
            return False
        seen.add(parent)
        current = work.task(project_id, parent)
        if current is None:
            return False
    return False


def _who_label(name: Any, task: Any) -> Optional[dict[str, Any]]:
    """Who owes the next response, for people: a team by its name, or the person who asked."""
    from ...teamspace import team_label

    if not name:
        return None
    if task is not None and name == task.requester.member and name != task.requester.team:
        return {"id": name, "kind": "person", "label": f"The person who asked ({team_label(task.requester.team)})"}
    return {"id": name, "kind": "team", "label": team_label(str(name))}


def _report_doc(report: Any) -> dict[str, Any]:
    from ...teamspace import team_label

    return {**report.model_dump(mode="json"), "id": report.id, "owner_label": team_label(report.owner)}


def _task_doc(task: Any) -> dict[str, Any]:
    from ...teamspace import team_label

    return {**task.model_dump(mode="json"), "owner_label": team_label(task.owner_team), "requester_label": team_label(task.requester.team)}


DELIVERY_SHOWN = ("id", "kind", "item_id", "status", "reason", "attempts", "created_at", "delivered_at")


@router.get("/api/projects/{project_id}/team-tasks/{task_id}", response_model=TeamTaskView)
async def read_team_task(project_id: str, task_id: str, s: Services = Depends(services)):
    """W8: a task's page: what was asked and what counts as done, every action with its status, its reports at every revision
    (the answer first; superseded revisions linked both ways), its children and their joined reports, and who owes the next
    response. A member of a team on the task's chain reads it, and the owner; anyone else, and another project's or a missing
    task, gets the same 403."""
    def run() -> dict[str, Any]:
        from ...teamspace import team_label
        from ...teamspace.store import open_team_work

        try:
            _scope(s, project_id)
        except HTTPException as refused:
            raise HTTPException(403, TASK_REFUSED if refused.detail == "Not yours to read" else refused.detail) from None
        work = open_team_work(s.store.home, projects=s.projects)
        task = work.task(project_id, task_id)
        if task is None:
            raise HTTPException(403, TASK_REFUSED)
        try:
            _scope(s, project_id, task.problem_id)
        except HTTPException:  # a problem deleted or out of reach: the same answer as a task that does not exist
            raise HTTPException(403, TASK_REFUSED) from None
        teams, who, auth_on = _teams(s), _who(), _auth_on()
        keys = [th.team_key(teams, name)[0] for name in _chain_teams(work, project_id, task)]
        if not any(th.access({"team": key}, who, teams, auth_on=auth_on) is not None for key in keys) \
                and not _asked_from_my_home(work, project_id, task, who, teams, auth_on, s):
            raise HTTPException(403, TASK_REFUSED)
        view = work.task_view(project_id, task_id)
        if view is None:
            raise HTTPException(403, TASK_REFUSED)
        deliveries = [{**{k: row.get(k) for k in DELIVERY_SHOWN}, "from_team": team_label(str(row.get("from_team"))),
                       "to_team": team_label(str(row.get("to_team")))} for row in view["deliveries"]]
        join = view["join"]
        return {"project_id": project_id, "task": _task_doc(view["task"]), "children": [_task_doc(c) for c in view["children"]],
                "actions": [a.model_dump(mode="json") for a in view["actions"]],
                "reports": [_report_doc(r) for r in view["reports"]], "child_reports": [_report_doc(r) for r in view["child_reports"]],
                "join": join.model_dump(mode="json") if join is not None else None,
                "next_owner": _who_label(view["next_owner"], view["task"]), "deliveries": deliveries}
    return await asyncio.to_thread(run)


@router.get("/api/workspace/overview", response_model=WorkspaceOverview)
async def workspace_overview(project_id: Optional[str] = None, s: Services = Depends(services)):
    """2026-11-21: the team workspace as its own place, across projects: the conversations with the Lead this person may read
    (theirs; every one for the owner), and the open team tasks of the projects they may read, each with who owes the next
    response. Words are first lines only (the handoff history shows the same); a task's page holds the rest."""
    def run() -> dict[str, Any]:
        from ...teamspace import lead, team_label
        from ...teamspace.store import open_team_work

        who, auth_on, teams, store = _who(), _auth_on(), _teams(s), _threads(s)
        work = open_team_work(s.store.home, projects=s.projects)
        readable: dict[str, bool] = {}

        def may_read(pid: str) -> bool:
            if pid not in readable:
                try:
                    _scope(s, pid)
                    readable[pid] = True
                except HTTPException:
                    readable[pid] = False
            return readable[pid]

        names: dict[str, str] = {}

        def project_name(pid: str) -> str:
            if pid not in names:
                try:
                    names[pid] = str((s.projects.get(pid) or {}).get("name") or pid)
                except KeyError:
                    names[pid] = pid
            return names[pid]

        conversations = []
        for task in lead.lead_tasks(s, project_id):
            if not may_read(task.project_id):
                continue
            thread = store.thread(task.requester.thread_id)
            if thread is None or th.access(thread, who, teams, auth_on=auth_on) is None:
                continue
            view = work.task_view(task.project_id, task.id) or {}
            children = [{"id": c.id, "team": team_label(c.owner_team), "status": c.status} for c in view.get("children") or []]
            conversations.append({"task_id": task.id, "project_id": task.project_id, "project": project_name(task.project_id),
                                  "problem_id": task.problem_id, "question": _first_line(task.question), "status": task.status,
                                  "next_owner": _who_label(view.get("next_owner"), task), "teams": children,
                                  "thread_id": thread["id"], "draft_id": th.draft_of(thread), "created_at": task.created_at})
        open_tasks = []
        seen: set[str] = set()
        for row in work.deliveries(project_id=project_id):
            if row["kind"] != "task" or row["task_id"] in seen or not may_read(row["project_id"]):
                continue
            seen.add(row["task_id"])
            task = work.task(row["project_id"], row["task_id"])
            if task is None or not tw_open(task) or task.owner_team == "lead":
                continue
            view = work.task_view(task.project_id, task.id) or {}
            open_tasks.append({"task_id": task.id, "project_id": task.project_id, "project": project_name(task.project_id),
                               "owner": task.owner_team, "owner_label": team_label(task.owner_team), "status": task.status,
                               "question": _first_line(task.question), "next_owner": _who_label(view.get("next_owner"), task),
                               "created_at": task.created_at})
        return {"conversations": conversations, "open_tasks": open_tasks,
                "projects": sorted({(c["project_id"], c["project"]) for c in conversations + open_tasks}, key=lambda x: x[1])}
    return await asyncio.to_thread(run)


def tw_open(task: Any) -> bool:
    from dclab_core import team_work as tw

    return tw.is_open(task)


def _first_line(text: Any, limit: int = 200) -> str:
    line = str(text or "").strip().split("\n", 1)[0]
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


@router.get("/api/team-agents/{name}", response_model=TeamAgentJobDoc)
async def team_agent_job(name: str, s: Services = Depends(services)):
    """W8: a team agent's job description, by the agent's name (``data_science``, ``model:eta``) or a team's id (its kind's
    agent): its responsibility, the actions it may take and what performs each, the report it owes and what it may never do."""
    from dclab_core import team_work as tw

    def run() -> dict[str, Any]:
        from ...teamspace.agents import job_description

        key = str(name or "").strip()
        base = key.split(":")[0]
        if base not in tw.DIRECTORY:
            found = _teams(s).team(key)
            base = key = str((found or {}).get("kind") or "")
            if base not in tw.DIRECTORY:
                raise HTTPException(404, "No such team agent")
        return job_description(key)
    return await asyncio.to_thread(run)


@router.get("/api/team-brains", response_model=TeamBrainsDoc)
async def team_brains(s: Services = Depends(services)):
    """B1: every team role's brain (rules, helper, agent), where it came from (default, stored, or the DCLAB_TEAM_BRAIN
    override), and the last changes. Any member of the workspace reads it."""
    from ...teamspace import brains

    return await asyncio.to_thread(brains.overview, s)


@router.put("/api/team-brains/{role}", response_model=TeamBrainDoc)
async def set_team_brain(role: str, payload: TeamBrainSet, s: Services = Depends(services)):
    """B1: one audited switch of a team role's brain, the owner's (``accounts/guard.SPECIAL``: admin), with a reason; the
    rules are the default. The change and its audit row are written together; the same arm again changes nothing."""
    from ...accounts.principal import LOCAL_OWNER, current
    from ...teamspace import brains

    who = current() or LOCAL_OWNER

    def run() -> dict[str, Any]:
        try:
            brains.switch(s.gateway.routing, s.audit, role, str(payload.arm or ""), who.role, str(payload.reason or ""))
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from None
        except brains.BrainSwitchError as exc:
            raise HTTPException(422, str(exc)) from None
        return next(r for r in brains.overview(s)["roles"] if r["role"] == role)
    return await asyncio.to_thread(run)
