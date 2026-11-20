"""The team agents' tools and the guard every one of them passes (W5 of the collaborative workspace, 2026-11-20).

Registered once, in scope ``team`` of the one registry (``agents.GROUPS["team"]``): the team agents (``agents.py``) reach
them only through ``runtime.run``, so every call is checked against its schema, passes :class:`TeamGuard`, and is a traced
step of the agent's run. Two kinds of tool:

- **Task moves** (``team_accept``, ``team_block``; ``team_return_report`` completes the task when it reports): the task's
  owner moves its own task, through ``team_work.validate_transition`` (the store calls it).
- **Directory actions**: ``team_delegate``, ``team_clarify``, ``team_record_human_response``, ``team_act``,
  ``team_model_call``. Each names an ``action`` of the acting agent's directory (``team_work.DIRECTORY``) and is the
  mechanism of its binding (``tool:delegate`` is ``team_delegate``; a ``human:`` binding is asked with ``team_clarify`` and
  recorded with ``team_record_human_response``; a bound tool this release provides is ``team_act`` or, for a model
  version, ``team_model_call``). Each records its ``TeamAction`` through the W4 store: written ``running``, then finished
  once with its terminal status, under the retry key ``(task, action, input revision)``. Its action id is derived from that
  key, so a retried call finds the record and adds nothing. (The store has no separate start: a ``planned`` record would
  need a second record to become ``running``, which the store refuses; the attempt is written when it starts.)
- ``team_answer`` carries a person's recorded answer from the requesting agent to the task's owner, as an ``answer``
  delivery; it runs only after that agent recorded the person's response.

The guard refuses, before any handler runs: an action outside the acting agent's directory (``team_work.may_take``); a
forbidden one (an approval: no agent approves for the decision owner, ``team_work.may_approve``; contacting customers: no
role has that permission by default); an action through the wrong tool; a human-bound response without the person who
gave it, or from a person who does not hold the bound role; a task outside the project, or one the agent neither owns nor
requested; and a move by an agent that does not own the task.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from dclab_core import team_work as tw

from ..agents.registry import Registry, Tool
from . import fixtures

SCOPE = "team"
Row = dict[str, Any]

# The bound tools this release provides (the rest of the directory is bound to tools of later releases: R2, R3).
ACTS = ("inspect_schema", "leakage_check", "define_baseline", "explain_uncertainty", "review_evidence", "join_replies",
        "present_options")
MODEL_TOOLS = ("model_score", "model_contract", "model_evaluation")
ACTION_MOVES = ("delegate", "clarify", "record_human_response", "act", "model_call")
MOVES = ("accept", "block", "answer", "return_report") + ACTION_MOVES  # every write of the team scope, as its guard reads it
OWNER_MOVES = ("accept", "block", "return_report", "model_call")
NEVER = {"approve": "approve a decision for its owner", "train": "train a model", "deploy": "deploy a model",
         "contact_customers": "contact customers"}
_ID_CHARS = re.compile(r"[^A-Za-z0-9_.:@-]+")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _digest(*parts: Any) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12]


def safe_id(text: str) -> str:
    """A string as a team_work id (letters, digits and ``_.:@-``, at most 120 characters)."""
    cleaned = _ID_CHARS.sub("-", str(text)).strip("-.:@_") or "x"
    return cleaned if len(cleaned) <= 120 else cleaned[:107] + "-" + _digest(cleaned)


def action_id(task_id: str, action: str, input_revision: str) -> str:
    """The id of an attempt, derived from its retry key: the same attempt asked twice is the same record."""
    return f"{action}-{_digest(tw.retry_key(task_id, action, input_revision))}"


def _threads(env: Any) -> Optional[tuple[Any, Any]]:
    """The workspace's team threads and teams (W3, W2), or None for an env with no workspace."""
    from . import threads

    if getattr(env, "team_threads", None) is None and _home(env) is None:
        return None
    return threads.threads_for(env), threads._teams_for(env)


def team_thread(env: Any, team: str, project_id: str, problem_id: str) -> str:
    """The thread a child task's reply goes back to: the delegating team's own thread on the problem (W3: opened the first
    time). An env with no workspace gets a stable name, which ``threads.post_delivery`` maps to that team's thread."""
    found = _threads(env)
    if found is None:
        return safe_id(f"team.{team}.{problem_id}")
    from .threads import team_key

    store, teams = found
    key, label = team_key(teams, team)
    thread, _ = store.open_thread(project_id, problem_id, key, label=label, by={"kind": "agent", "id": team, "team": team})
    return thread["id"]


def post_own(ctx: TeamTurn, task: tw.TeamTask, text: str, *, key: str, kind: str) -> Optional[Row]:
    """A message by the acting agent in its own team's thread on the task's problem, once per ``key`` (W3's post): what the
    agent did for the team that reads it (a question it sent another team)."""
    found = _threads(ctx.env)
    if found is None:
        return None
    store = found[0]
    thread_id = team_thread(ctx.env, ctx.agent, ctx.project_id, task.problem_id)
    handoff = {"kind": kind, "task_id": task.id, "item_id": task.id, "from_team": ctx.agent, "to_team": task.requester.team}
    message, _ = store.post(thread_id, {"kind": "agent", "id": ctx.agent, "team": ctx.agent}, text, handoff=handoff, key=key)
    return message


# ---------------------------------------------------------------------------------------------------------- the turn
@dataclass
class TeamTurn:
    """What a tool runs in: the acting agent (a directory name, or a qualified specialist such as ``model:eta``), its
    project and the task in focus, the member it acts for (``initiator``: the outbox keeps it on every row), and what started
    the turn (a delivered row, a person's words, a correction)."""

    env: Any
    work: Any
    project_id: str
    agent: str
    task_id: Optional[str] = None
    initiator: Optional[Row] = None
    person: Optional[str] = None
    text: Optional[str] = None
    row: Optional[Row] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def config(self) -> Optional[tw.AgentConfig]:
        return tw.DIRECTORY.get(self.agent.split(":")[0])

    @property
    def actor(self) -> tw.Actor:
        return tw.Actor(kind="agent", id=self.agent, team=self.agent)


def _home(env: Any) -> Optional[Path]:
    for home in (getattr(getattr(env, "store", None), "home", None), getattr(env, "home", None)):
        if home:
            return Path(home)
    return None


# ---------------------------------------------------------------------------------------------------------- reads
def _task(ctx: TeamTurn, task_id: Optional[str]) -> tw.TeamTask:
    task = ctx.work.task(ctx.project_id, str(task_id or ""))
    if task is None:
        raise KeyError(f"no task {task_id} in project {ctx.project_id}")
    return task


def _actions(ctx: TeamTurn, task_id: str) -> list[tw.TeamAction]:
    view = ctx.work.task_view(ctx.project_id, task_id)
    return list(view["actions"]) if view else []


def case_member(ctx: TeamTurn, task: tw.TeamTask) -> tuple[Optional[str], str]:
    """The person a case answers to and their team: the nearest requester with a member, up the delegation chain."""
    seen: set[str] = set()
    current: Optional[tw.TeamTask] = task
    while current is not None and current.id not in seen:
        seen.add(current.id)
        if current.requester.member:
            return current.requester.member, current.requester.team
        current = ctx.work.task(ctx.project_id, current.parent_task_id) if current.parent_task_id else None
    return None, task.requester.team


def _team_kinds(ctx: TeamTurn, person: str) -> Optional[set[str]]:
    """The kinds of the workspace's teams the person is in (W2), or None when the workspace has no teams set up."""
    held = getattr(ctx.env, "teams", None)
    teams = held if held is not None and hasattr(held, "teams_of") else None
    if teams is None:
        home = _home(ctx.env)
        if home is None:
            return None
        from ..storage import open_teams

        teams = open_teams(home)
    if not teams.teams():
        return None
    return {t["kind"] for t in teams.teams_of(person)}


def holds_role(ctx: TeamTurn, task: tw.TeamTask, binding: str, person: str) -> Optional[str]:
    """Why ``person`` may not answer for a ``human:<role>`` binding, or None. The requester and the decision owner are the
    case's member (the person who asked); any other role is a member of a team of the acting agent's kind. When the
    workspace has teams set up (W2), the person is also in a team of the kind they answer for."""
    role = binding.split(":", 1)[1]
    if role in ("requester", "decision_owner"):
        member, team = case_member(ctx, task)
        if member is None:
            return f"this case has no member to answer as its {role.replace('_', ' ')}"
        if person != member:
            return f"{person} is not this case's {role.replace('_', ' ')}: only {member} answers it"
    else:
        team = ctx.agent.split(":")[0]
    kinds = _team_kinds(ctx, person)
    if kinds is None:
        return None if role in ("requester", "decision_owner") else f"no team is set up to answer as {role.replace('_', ' ')}"
    if team.split(":")[0] not in kinds:
        return f"{person} is not in the workspace's {team} team"
    return None


# ---------------------------------------------------------------------------------------------------------- the guard
def _forbidden(ctx: TeamTurn, action: str) -> Optional[str]:
    config = ctx.config
    if action == "approve":
        placeholder = tw.TeamDecision(id="approval-check", problem_id="approval-check", option="x",
                                      decision_owner=tw.Actor(kind="person", id="decision-owner"))
        if not tw.may_approve(ctx.actor, placeholder):
            return ("Refused: an agent never approves. Only the decision owner, a person, approves a decision "
                    "(team_work.may_approve); prepare the options and let the owner decide.")
    if action == "contact_customers":
        return (f"Refused: {ctx.agent} has no permission to contact customers. No agent or role contacts customers by "
                "default; a person with that permission does it outside this workspace.")
    if action in config.forbidden or action in NEVER:
        return f"Refused: {ctx.agent} may never {NEVER.get(action, action)}."
    return None


def _authority(ctx: TeamTurn) -> Optional[str]:
    """The member the turn acts for may still change work in this project, now (rechecked at every tool call, as the
    delivery rechecks it before handing a row over): a member who lost access since gets nothing done in their name."""
    from ..memory.scope import AccessDenied
    from . import delivery

    try:
        delivery.authorize(ctx.env, {"initiator": ctx.initiator, "project_id": ctx.project_id})
    except AccessDenied as exc:
        return f"Refused: the member this work acts for can no longer change work in this project ({type(exc).__name__})."
    return None


def _responder(ctx: TeamTurn, task: tw.TeamTask, action: str, person: str, response_id: str) -> Optional[str]:
    """Why ``person``'s answer may not be recorded here, or None. An agent never writes down a person's answer that this
    turn did not carry:

    - the turn acts for a member (``initiator``): the person is that member, and the turn's own person (if named);
    - the task's owner records an answer relayed to it: the requesting agent already recorded the same response from the
      same person (a finished human-bound action with that response id), so the owner only repeats what was given."""
    who = (ctx.initiator or {}).get("user_id")
    if who is not None and str(who) != person:
        return f"Refused: this turn acts for {who}; it cannot record an answer by {person}."
    if ctx.person is not None and ctx.person != person:
        return f"Refused: this turn carries {ctx.person}'s words; it cannot record an answer by {person}."
    if task.owner_team == ctx.agent and task.requester.team != ctx.agent:
        given = [a for a in _actions(ctx, task.id) if a.actor.id == task.requester.team and a.source.startswith("human:")
                 and a.status in tw.DONE_ACTIONS and a.responded_by is not None and a.responded_by.id == person
                 and a.output_ref == f"response:{response_id}"]
        if not given:
            return (f"Refused: no answer {response_id or '(none)'} by {person} was recorded by {task.requester.team}; the owner "
                    "records only an answer its requester carried.")
    return None


def _binding_fits(move: str, binding: str) -> bool:
    if move == "delegate":
        return binding == "tool:delegate"
    if move in ("clarify", "record_human_response"):
        return binding.startswith("human:")
    if move == "act":
        return binding.startswith("tool:") and binding[5:] in ACTS
    if move == "model_call":
        return binding.startswith("tool:") and binding[5:] in MODEL_TOOLS
    return False


class TeamGuard:
    """Every write of the team scope passes here first (see the module's docstring for what it refuses)."""

    def __call__(self, tool: Tool, context: Any, arguments: dict[str, Any], proceed: Callable[[], Any]) -> Any:
        why = self.refusal(tool, context, arguments)
        return {"error": why} if why else proceed()

    @staticmethod
    def refusal(tool: Tool, ctx: Any, args: dict[str, Any]) -> Optional[str]:
        if not isinstance(ctx, TeamTurn) or ctx.config is None or ctx.work is None:
            return "Team tools run only in a team agent's turn, with its workspace's team work."
        config, move = ctx.config, tool.move
        action = args.get("action")
        why = _authority(ctx)
        if why:
            return why
        if move in ACTION_MOVES:
            if not action:
                return f"{tool.name} names the directory action it performs."
            why = _forbidden(ctx, action)
            if why:
                return why
            if not tw.may_take(config, action):
                return f"Refused: {action} is not in {config.name}'s action directory ({', '.join(a.action for a in config.action_directory)})."
            binding = config.binding(action) or ""
            if not _binding_fits(move, binding):
                return f"Refused: {action} is bound to {binding}, not to {tool.name}."
        if move == "delegate" and not args.get("task_id"):
            return None if config.kind == "team_agent" else "Only a team agent opens a new top-level task."
        try:
            task = _task(ctx, args.get("task_id"))
        except KeyError as exc:
            return f"Refused: {exc.args[0]}."
        mine = task.owner_team == ctx.agent
        if move in OWNER_MOVES or move == "delegate" and args.get("task_id"):
            if not mine:
                return f"Refused: {task.id} is {task.owner_team}'s task; only its owner {move.replace('_', ' ')}s it."
        elif not mine and task.requester.team != ctx.agent:
            return f"Refused: {ctx.agent} neither owns nor requested {task.id}."
        if move == "record_human_response":
            person = str(args.get("responded_by") or "").strip()
            if not person:
                return ("Refused: a human-bound action is recorded only with the person who responded (responded_by): "
                        "an agent cannot answer for a person.")
            return _responder(ctx, task, action, person, str(args.get("response_id") or "")) or \
                holds_role(ctx, task, config.binding(action) or "", person)
        if move == "answer":
            if task.requester.team != ctx.agent:
                return f"Refused: only the requesting team ({task.requester.team}) carries its person's answer."
            answered = [a for a in _actions(ctx, task.id) if a.actor.id == ctx.agent and a.source.startswith("human:")
                        and a.status in tw.DONE_ACTIONS and a.responded_by is not None]
            if not answered:
                return "Refused: record the person's response first (team_record_human_response); an answer is never made up."
        return None


# ---------------------------------------------------------------------------------------------------------- recording
def record(ctx: TeamTurn, task: tw.TeamTask, action: str, *, input_revision: str, status: Optional[str] = None,
           output_ref: Optional[str] = None, failure_ref: Optional[str] = None, reason: Optional[str] = None,
           responded_by: Optional[str] = None, source: Optional[str] = None) -> tuple[tw.TeamAction, bool]:
    """One attempt of a directory action: written running, then (``status`` given) finished once. Returns the record and
    whether it was already there (a retried call adds nothing)."""
    config = ctx.config
    binding = config.binding(action) or ""
    aid = action_id(task.id, action, input_revision)
    held = next((a for a in _actions(ctx, task.id) if a.action_id == aid), None)
    if held is not None and tw.can_claim(held):
        return held, True
    if held is None:
        if source is None:
            source = binding if binding.startswith("tool:") else f"human:{safe_id('q-' + _digest(task.id, action))}"
        held = tw.TeamAction(task_id=task.id, action_id=aid, actor=ctx.actor, action=action, source=source,
                             input_revision=input_revision, status="running", started_at=now(),
                             retry_key=tw.retry_key(task.id, action, input_revision))
        if not tw.may_take(config, held):
            raise ValueError(f"{action} by {ctx.agent} through {source} is not an action of its directory")
        ctx.work.record_action(ctx.project_id, held)
    if status is None:
        return held, False
    person = tw.Actor(kind="person", id=responded_by) if responded_by else None
    ended = ctx.work.finish_action(ctx.project_id, task.id, aid, status, ended_at=now(), output_ref=output_ref,
                                   failure_ref=failure_ref, reason=reason, responded_by=person).item
    if not tw.may_take(config, ended):
        raise ValueError(f"{action} by {ctx.agent} ended without what its binding needs")
    return ended, False


def _send(ctx: TeamTurn, written: Any) -> None:
    """Queue the delivery a change saved (``written.delivery``, if any)."""
    _queue(ctx, getattr(written, "delivery", None))


def _queue(ctx: TeamTurn, row: Optional[Row]) -> None:
    """Queue a saved delivery row. A failure here loses nothing: the row is saved, and the relay
    (``delivery.requeue_undelivered``) queues it on the worker's next recovery tick."""
    if not row:
        return
    from . import delivery

    try:
        delivery.enqueue(ctx.env, row)
    except Exception:  # noqa: BLE001 — see above: the relay picks the saved row up
        pass


def _line(a: tw.TeamAction) -> Row:
    return {"action_id": a.action_id, "action": a.action, "status": a.status, "ref": a.output_ref or a.failure_ref}


# ---------------------------------------------------------------------------------------------------------- handlers
def _move(ctx: TeamTurn, task_id: str, status: str, **kw: Any) -> Any:
    """The owner's move, once: a task already in that status (another turn of the same agent moved it first) is the same
    move, not a refusal. Returns what the store wrote, or None when nothing had to move."""
    if _task(ctx, task_id).status == status:
        return None
    try:
        return ctx.work.move(ctx.project_id, task_id, status, ctx.actor, initiator=ctx.initiator, **kw)
    except ValueError:
        if _task(ctx, task_id).status == status:
            return None
        raise


def _accept(ctx: TeamTurn, task_id: str) -> Row:
    moved = _move(ctx, task_id, "accepted")
    return {"task_id": task_id, "status": "accepted", "duplicate": moved is None}


def _block(ctx: TeamTurn, task_id: str, reason: str) -> Row:
    moved = _move(ctx, task_id, "blocked")
    return {"task_id": task_id, "status": "blocked", "reason": reason, "duplicate": moved is None}


def _clarify(ctx: TeamTurn, task_id: str, action: str, question: str) -> Row:
    """Ask a person. The owner's question moves its task to waiting_for_person and travels to the requester as a
    clarification (the task stays open); the requesting agent that receives it records that it asked its person (the
    delivery posts the question into the person's thread)."""
    task = _task(ctx, task_id)
    held, duplicate = record(ctx, task, action, input_revision=f"brief-{task.brief_revision}")
    if tw.can_claim(held):
        return {"task_id": task.id, "action": _line(held), "duplicate": True}
    if task.owner_team == ctx.agent and task.status != "waiting_for_person":
        _send(ctx, _move(ctx, task.id, "waiting_for_person", note=question))
    if task.owner_team == ctx.agent:  # the owner's own thread keeps the question it sent (the delivery posts it to the requester's)
        post_own(ctx, task, question, key=f"asked:{held.action_id}:{task.id}", kind="clarification")
    return {"task_id": task.id, "asked": question, "action": _line(held), "duplicate": duplicate}


def _record_human_response(ctx: TeamTurn, task_id: str, action: str, response_id: str, responded_by: Optional[str] = None) -> Row:
    task = _task(ctx, task_id)
    ended, duplicate = record(ctx, task, action, input_revision=f"brief-{task.brief_revision}", status="succeeded",
                              output_ref=f"response:{response_id}", responded_by=responded_by,
                              source=f"human:{safe_id(response_id)}")
    return {"task_id": task.id, "action": _line(ended), "responded_by": responded_by, "duplicate": duplicate}


def _answer(ctx: TeamTurn, task_id: str, response_id: str, text: str) -> Row:
    """The person's answer to the owner, as an ``answer`` delivery saved once per response (``store.send_answer``). The
    owner takes its task up again when the row reaches it."""
    task = _task(ctx, task_id)
    written = ctx.work.send_answer(ctx.project_id, task.id, response_id=safe_id(response_id), text=str(text), initiator=ctx.initiator)
    if not written.duplicate:
        _send(ctx, written)
    return {"task_id": task.id, "delivery": written.delivery["id"], "duplicate": written.duplicate}


def _delegate(ctx: TeamTurn, action: str, new_task_id: str, owner_team: str, question: str, expected_output: str,
              done_criteria: list[str], task_id: Optional[str] = None, problem_id: Optional[str] = None,
              thread_id: Optional[str] = None, allowed_inputs: Optional[list[str]] = None, budget_eur: float = 0.0,
              brief_revision: int = 1) -> Row:
    """A task for another team: a new top-level task (``task_id`` omitted: the requesting team agent asks for its person,
    who is the case's member) or a child of the acting agent's task (``team_work.delegate``). Saved with its delivery;
    the delegation is the acting agent's action, recorded on the task it was made from."""
    work = ctx.work
    if task_id is None:
        if not problem_id or not thread_id:
            raise ValueError("a new top-level task names its problem and the thread its answer goes back to")
        made = work.create_task(tw.TeamTask(id=new_task_id, workspace_id=str(work.workspace), project_id=ctx.project_id,
                                            problem_id=problem_id, brief_revision=brief_revision,
                                            requester=tw.Requester(team=ctx.agent, member=ctx.person, thread_id=thread_id),
                                            owner_team=owner_team, question=question, expected_output=expected_output,
                                            done_criteria=done_criteria, allowed_inputs=list(allowed_inputs or []),
                                            budget_eur=budget_eur, created_at=now()), initiator=ctx.initiator)
        _send(ctx, made)
        on = made.item
    else:
        parent = _task(ctx, task_id)
        made = work.delegate(ctx.project_id, parent.id, id=new_task_id, owner_team=owner_team, question=question,
                             expected_output=expected_output, done_criteria=done_criteria, by=ctx.actor,
                             thread_id=thread_id or team_thread(ctx.env, ctx.agent, ctx.project_id, parent.problem_id), created_at=now(),
                             allowed_inputs=list(allowed_inputs or []), budget_eur=budget_eur, initiator=ctx.initiator,
                             parent_waits=True)
        _send(ctx, made)
        on = parent
    done, _ = record(ctx, on, action, input_revision=new_task_id, status="succeeded", output_ref=f"task:{new_task_id}")
    return {"task_id": new_task_id, "owner_team": owner_team, "action": _line(done), "duplicate": made.duplicate}


def _act(ctx: TeamTurn, task_id: str, action: str, subject: Optional[str] = None) -> Row:
    """A directory action bound to a tool this release provides, computed by code from what the task may read."""
    task = _task(ctx, task_id)
    tool = (ctx.config.binding(action) or "")[5:]
    status, ref, revision = "succeeded", None, f"brief-{task.brief_revision}"
    sample_ref = next((i for i in task.allowed_inputs if i.startswith("sample:")), None)
    if tool == "inspect_schema":
        found = fixtures.sample(sample_ref) if sample_ref else None
        if found is None:
            status, ref = "failed", f"no permitted sample for {task.id}"
        else:
            check = fixtures.feasibility(sample_ref)
            ref = f"schema:{sample_ref}:{len(found['columns'])} columns" + ("" if check["ok"] else f"; missing {check['missing']}")
    elif tool == "leakage_check":
        check = fixtures.feasibility(sample_ref or "")
        ref = "excluded:" + (",".join(check.get("excluded") or ()) or "none") + " (known only after renewal)"
    elif tool == "define_baseline":
        ref = "baseline:rank by days since last login (the simple rule a model must beat)"
    elif tool == "explain_uncertainty":
        ref = "uncertainty:feasibility from the sample's columns only; no account was scored"
    elif tool in ("review_evidence", "join_replies", "present_options"):
        if not subject:
            raise ValueError(f"{action} names the report it reads (subject)")
        reports = [r for r in ctx.work.task_view(ctx.project_id, task.id)["reports"] if r.id == subject]
        if tool == "review_evidence":
            view = ctx.work.task_view(ctx.project_id, task.id)
            reports = [r for r in view["child_reports"] if r.id == subject]
        if not reports:
            raise ValueError(f"{subject} is not a report {ctx.agent} may read on {task.id}")
        revision, ref = subject, f"{'reviewed' if tool == 'review_evidence' else 'presented'}:{subject}"
    else:
        raise ValueError(f"{tool} is not a tool this release provides")
    if status == "succeeded" and tool in ("inspect_schema", "leakage_check", "define_baseline", "explain_uncertainty") \
            and sample_ref in fixtures.SAMPLES:
        status = "simulated"  # a check on an invented sample is labelled as such, so the report says it is simulated
    ended, duplicate = record(ctx, task, action, input_revision=revision, status=status,
                              output_ref=ref if status in tw.DONE_ACTIONS else None,
                              failure_ref=ref if status in tw.FAILED_ACTIONS else None)
    return {"task_id": task.id, "action": _line(ended), "duplicate": duplicate}


def _model_call(ctx: TeamTurn, task_id: str, action: str, correction_minutes: Optional[int] = None) -> Row:
    """The fixture model specialist: a version's estimate for the task's dispatch snapshot, labelled simulated. A
    correction (the kitchen's added minutes) is a new attempt on a new input revision; a missing version is a denial."""
    task = _task(ctx, task_id)
    kind = ctx.agent.split(":", 1)[1] if ":" in ctx.agent else ""
    snapshot = next((i for i in task.allowed_inputs if i.startswith("dispatch:")), "")
    revision = snapshot + (f";kitchen+{int(correction_minutes)}m" if correction_minutes else "")
    found = fixtures.model_fixture(kind, snapshot) if snapshot else None
    if action != "score":
        ended, duplicate = record(ctx, task, action, input_revision=revision or "none", status="simulated" if found else "denied",
                                  output_ref=f"fixture:{found['id']}:contract ok" if found else None,
                                  failure_ref=None if found else f"no permitted {kind or 'model'} version for {snapshot or 'this task'}")
        return {"task_id": task.id, "action": _line(ended), "duplicate": duplicate}
    if found is None:
        ended, duplicate = record(ctx, task, action, input_revision=revision or "none", status="denied",
                                  failure_ref=f"no permitted {kind or 'model'} version for {snapshot or 'this task'}")
        return {"task_id": task.id, "action": _line(ended), "denied": True, "duplicate": duplicate}
    if kind == "eta":
        value = fixtures.eta_after(found["arrival"], correction_minutes or 0)
    else:
        value = f"{round(float(found['probability']) * 100)}%"
    ended, duplicate = record(ctx, task, action, input_revision=revision, status="simulated",
                              output_ref=f"fixture:{found['id']}={value}")
    return {"task_id": task.id, "fixture": found["id"], "value": value, "simulated": True, "action": _line(ended),
            "duplicate": duplicate}


def _return_report(ctx: TeamTurn, task_id: str, answer: str, outcome: str, revision: int, next_owner: Optional[str] = None,
                   unblocks: Optional[str] = None, evidence: Optional[list[dict[str, Any]]] = None,
                   limits: Optional[list[str]] = None) -> Row:
    """The owner's report at ``revision``, built by ``team_work.build_report`` from the stored actions and joined child
    reports (the store builds it), with its reply to the requester. ``complete`` completes the task (refused while a child
    is out), ``partial`` marks it partial, ``blocked`` needs the task blocked first; the move and the report are one
    transaction (``store.report_task``). Revision 1 is the first report; a later one supersedes the current report (a
    correction). The same revision again returns the stored report, also when another turn stored it first."""
    work, task = ctx.work, _task(ctx, task_id)
    if outcome not in ("complete", "partial", "blocked"):
        raise ValueError("a report's outcome is complete, partial or blocked")
    view = work.task_view(ctx.project_id, task.id)
    held = next((r for r in view["reports"] if r.revision == revision), None)
    if held is not None:
        return {"task_id": task.id, "report": held.id, "status": held.status, "duplicate": True}
    current = [r for r in view["reports"] if r.status != "superseded"]
    if revision != 1 and (not current or max(r.revision for r in current) != revision - 1):
        raise ValueError(f"revision {revision} follows the current report's revision")
    move_to = None
    if outcome == "complete" and task.status != "complete":
        move_to = "complete"
    elif outcome == "partial" and task.status in ("accepted", "waiting_for_team"):
        move_to = "partial"
    elif outcome == "blocked" and task.status != "blocked":
        raise ValueError("block the task (team_block) before reporting the blocker")
    step = tw.NextStep(closed=True) if not next_owner else tw.NextStep(owner=next_owner, unblocks=unblocks or "the next step")
    written = work.report_task(ctx.project_id, task.id, revision=revision, current_brief_revision=task.brief_revision, answer=answer,
                               next_task=step, move_to=move_to, by=ctx.actor,
                               evidence=[tw.EvidenceRef.model_validate(e) for e in evidence or []], limits=list(limits or []),
                               status=None if outcome != "partial" else "partial", initiator=ctx.initiator)
    _send(ctx, written)
    return {"task_id": task.id, "report": written.item.id, "status": written.item.status, "simulated": written.item.simulated,
            "supersedes": written.item.supersedes, "joined": None if written.verdict is None else written.verdict.ok,
            "duplicate": written.duplicate}


# ---------------------------------------------------------------------------------------------------------- registration
def _errors(exc: Exception) -> Optional[dict[str, Any]]:
    """An expected failure (a rule refused the move, a task is missing, access was denied) as an error the run records."""
    if isinstance(exc, (ValueError, KeyError, LookupError, PermissionError)):
        return {"error": f"{type(exc).__name__}: {exc}"}
    return None


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


_S = {"type": "string"}
_TASK = {"type": "string", "description": "The task, by its id in this project."}
_ACTION = {"type": "string", "description": "The directory action this call performs."}

TOOLS = (
    Tool("team_accept", "Accept a task this team owns.", _schema({"task_id": _TASK}, ["task_id"]), _accept,
         effect="write", scope=SCOPE, move="accept", takes_context=True, errors=_errors),
    Tool("team_block", "Block a task this team owns: its report will name what unblocks it.",
         _schema({"task_id": _TASK, "reason": _S}, ["task_id", "reason"]), _block,
         effect="write", scope=SCOPE, move="block", takes_context=True, errors=_errors),
    Tool("team_clarify", "Ask a person one question for a human-bound directory action; the task stays open.",
         _schema({"task_id": _TASK, "action": _ACTION, "question": _S}, ["task_id", "action", "question"]), _clarify,
         effect="write", scope=SCOPE, move="clarify", takes_context=True, errors=_errors),
    Tool("team_record_human_response", "Record the person's response to a human-bound directory action.",
         _schema({"task_id": _TASK, "action": _ACTION, "response_id": _S,
                  "responded_by": {"type": "string", "description": "The person who responded, by their id."}},
                 ["task_id", "action", "response_id"]), _record_human_response,
         effect="write", scope=SCOPE, move="record_human_response", takes_context=True, errors=_errors),
    Tool("team_answer", "Carry the person's recorded answer to the task's owner.",
         _schema({"task_id": _TASK, "response_id": _S, "text": _S}, ["task_id", "response_id", "text"]), _answer,
         effect="write", scope=SCOPE, move="answer", takes_context=True, errors=_errors),
    Tool("team_delegate", "Open a task for another team: a new top-level task, or a child of a task this team owns.",
         _schema({"action": _ACTION, "new_task_id": _S, "owner_team": _S, "question": _S, "expected_output": _S,
                  "done_criteria": {"type": "array", "items": _S, "minItems": 1}, "task_id": _TASK, "problem_id": _S,
                  "thread_id": _S, "allowed_inputs": {"type": "array", "items": _S}, "budget_eur": {"type": "number", "minimum": 0},
                  "brief_revision": {"type": "integer", "minimum": 1}},
                 ["action", "new_task_id", "owner_team", "question", "expected_output", "done_criteria"]), _delegate,
         effect="write", scope=SCOPE, move="delegate", takes_context=True, errors=_errors),
    Tool("team_act", "Perform a directory action bound to a tool this workspace provides.",
         _schema({"task_id": _TASK, "action": _ACTION, "subject": {"type": "string", "description": "The report it reads."}},
                 ["task_id", "action"]), _act,
         effect="write", scope=SCOPE, move="act", takes_context=True, errors=_errors),
    Tool("team_model_call", "Ask one model version (a fixture in this release, labelled simulated) for its estimate.",
         _schema({"task_id": _TASK, "action": _ACTION, "correction_minutes": {"type": "integer"}}, ["task_id", "action"]),
         _model_call, effect="write", scope=SCOPE, move="model_call", takes_context=True, errors=_errors),
    Tool("team_return_report", "Return the task's report, built from its action log and its children's reports.",
         _schema({"task_id": _TASK, "answer": _S, "outcome": {"type": "string", "enum": ["complete", "partial", "blocked"]},
                  "revision": {"type": "integer", "minimum": 1}, "next_owner": _S, "unblocks": _S,
                  "evidence": {"type": "array", "items": {"type": "object"}}, "limits": {"type": "array", "items": _S}},
                 ["task_id", "answer", "outcome", "revision"]), _return_report,
         effect="write", scope=SCOPE, move="return_report", takes_context=True, errors=_errors),
)


def register(registry: Registry) -> None:
    for tool in TOOLS:
        registry.register(tool)
    registry.guards[SCOPE] = TeamGuard()


def is_error(content: Any) -> bool:
    try:
        value = json.loads(content) if isinstance(content, str) else content
    except (TypeError, ValueError):
        return False
    return isinstance(value, dict) and "error" in value


__all__ = ["ACTS", "MODEL_TOOLS", "MOVES", "SCOPE", "TOOLS", "TeamGuard", "TeamTurn", "action_id", "case_member", "holds_role", "is_error",
           "post_own", "record", "register", "safe_id", "team_thread"]
