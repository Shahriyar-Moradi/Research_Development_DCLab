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
from dclab_core.team_brain import PLAN_CHARS as _PLAN_CHARS

from ..agents.registry import Registry, Tool
from . import fixtures

SCOPE = "team"
Row = dict[str, Any]

# The bound tools this release provides (the rest of the directory is bound to tools of later releases: R2, R3).
ACTS = ("inspect_schema", "leakage_check", "define_baseline", "explain_uncertainty", "review_evidence", "join_replies",
        "present_options", "check_prerequisites", "validate_contract", "check_exclusions")
# T3 (2026-11-23): the Sales and Marketing tools, computed in ``teamspace/commercial.py`` (check_exclusions moved there too)
COMMERCIAL_ACTS = ("check_exclusions", "draft_list", "share_reference", "offer_rules", "campaign_calendar", "offer_cost")
ACTS = ACTS + tuple(a for a in COMMERCIAL_ACTS if a not in ACTS)


def own_team_binding(binding: str) -> bool:
    """T3: whether a human-bound action asks a member of the acting agent's own team (``human:sales_team``), not the person
    who asked (``human:requester``) or the decision's owner: that question is posted in the agent's own thread and answered
    there by its team, and the task does not wait on its requester."""
    role = binding.split(":", 1)[1] if binding.startswith("human:") else ""
    return bool(role) and role not in ("requester", "decision_owner")
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


def post_own(ctx: TeamTurn, task: tw.TeamTask, text: str, *, key: str, kind: str, to_team: Optional[str] = None) -> Optional[Row]:
    """A message by the acting agent in its own team's thread on the task's problem, once per ``key`` (W3's post): what the
    agent did for the team that reads it (a question it sent another team; T3: ``to_team`` its own, a question for its team)."""
    found = _threads(ctx.env)
    if found is None:
        return None
    store = found[0]
    thread_id = team_thread(ctx.env, ctx.agent, ctx.project_id, task.problem_id)
    handoff = {"kind": kind, "task_id": task.id, "item_id": task.id, "from_team": ctx.agent, "to_team": to_team or task.requester.team}
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
    held = tw.DIRECTORY.get(team.split(":")[0])
    if held is not None and held.kind in ("entry", "coordinator"):
        return None  # asked through Home and the Lead: the case's own member answers, whatever team they are in
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
      same person (a finished human-bound action with that response id), so the owner only repeats what was given;
    - B3 (the review's G1): the requester, or an owner that asked its own team, records only from that person's own words
      (:func:`answer_words`: this turn's message, or an answer already recorded on the task it was handed from), so a model
      can never record an answer nobody gave."""
    who = (ctx.initiator or {}).get("user_id")
    if who is not None and str(who) != person:
        return f"Refused: this turn acts for {who}; it cannot record an answer by {person}."
    if ctx.person is not None and ctx.person != person:
        return f"Refused: this turn carries {ctx.person}'s words; it cannot record an answer by {person}."
    own_team = own_team_binding(ctx.config.binding(action) or "")
    if not own_team and task.owner_team == ctx.agent and task.requester.team != ctx.agent:
        given = [a for a in _actions(ctx, task.id) if a.actor.id == task.requester.team and a.source.startswith("human:")
                 and a.status in tw.DONE_ACTIONS and a.responded_by is not None and a.responded_by.id == person
                 and a.output_ref == f"response:{response_id}"]
        if not given:
            return (f"Refused: no answer {response_id or '(none)'} by {person} was recorded by {task.requester.team}; the owner "
                    "records only an answer its requester carried.")
        return None
    # T3: the owner asked its own team (holds_role checks the team); or the requester records its person's answer. Either way
    # only from that person's own words: those this turn carries, or (a coordinator carrying an answer down) those already
    # recorded on the task it was handed from. B3, the review's G1: never an answer a model made up
    if answer_words(ctx, task, response_id, person) is None:
        return (f"Refused: {person} gave no answer {response_id or '(none)'} here; an agent records a person's answer only from "
                "the words that person gave (this turn's message, or an answer already recorded on the task it was handed from).")
    return None


def answer_words(ctx: TeamTurn, task: tw.TeamTask, response_id: str, person: Optional[str] = None) -> Optional[str]:
    """The person's own words for ``response_id``, or None (B3, the review's G1): the words this turn carries (a person's
    message, run by code) when its response is that one; else, for a task handed out from another, the answer already
    recorded on that task (its ``answer`` delivery, and a person's response recorded with the same id). ``person`` None:
    whoever gave it."""
    rid = safe_id(response_id or "")
    if ctx.person is not None and ctx.text is not None and (person is None or ctx.person == person):
        carried = ctx.extra.get("response_id")
        if carried is None or safe_id(str(carried)) == rid:
            return str(ctx.text)
    if not task.parent_task_id:
        return None
    view = ctx.work.task_view(ctx.project_id, task.parent_task_id)
    if view is None:
        return None
    recorded = any(a.source.startswith("human:") and a.status in tw.DONE_ACTIONS and a.responded_by is not None
                   and (person is None or a.responded_by.id == person) and a.output_ref == f"response:{response_id}" for a in view["actions"])
    words = next((d["text"] for d in view["deliveries"] if d["kind"] == "answer" and d["key"] == f"answer:{task.parent_task_id}:r-{rid}"), None)
    return words if recorded and words else None


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
    """Every write of the team scope passes here first (see the module's docstring for what it refuses). B3: the team's
    reads (``team_read_task``, ``team_read_notes``) ask :meth:`read_refusal` before they read anything."""

    def __call__(self, tool: Tool, context: Any, arguments: dict[str, Any], proceed: Callable[[], Any]) -> Any:
        why = self.refusal(tool, context, arguments)
        return {"error": why} if why else proceed()

    @staticmethod
    def read_refusal(ctx: Any, *, task_id: Optional[str] = None, team: Optional[str] = None) -> Optional[str]:
        """Why the acting agent may not read this, or None. The member it acts for may still change work in the project
        (rechecked at every read, as at every write); a task it reads is one it owns or asked for (a Lead reads the tasks it
        handed out: it asked for them); a team's notes (its own thread) stay in that team: an agent reads only its own."""
        if not isinstance(ctx, TeamTurn) or ctx.config is None or ctx.work is None:
            return "Team tools run only in a team agent's turn, with its workspace's team work."
        why = _authority(ctx)
        if why:
            return why
        if team is not None and str(team).split(":")[0] != ctx.agent.split(":")[0]:
            return (f"Refused: {team}'s notes stay in that team; {ctx.agent} reads only its own team's notes. Ask that team "
                    "through a task instead.")
        if task_id is not None:
            task = ctx.work.task(ctx.project_id, str(task_id))
            if task is None:
                return f"Refused: no task {task_id} in project {ctx.project_id}."
            if task.owner_team != ctx.agent and task.requester.team != ctx.agent:
                return f"Refused: {ctx.agent} neither owns nor requested {task.id}."
        return None

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
            why = tw.opening_refusal(config, str(args.get("owner_team") or ""))
            return f"Refused: {why}." if why else None
        if move == "delegate":
            why = tw.delegation_refusal(config, str(args.get("owner_team") or ""))
            if why:
                return f"Refused: {why}."
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
            response = str(args.get("response_id") or "")
            answered = [a for a in _actions(ctx, task.id) if a.actor.id == ctx.agent and a.source.startswith("human:")
                        and a.status in tw.DONE_ACTIONS and a.responded_by is not None and a.output_ref == f"response:{response}"]
            if not answered:
                return "Refused: record the person's response first (team_record_human_response); an answer is never made up."
            if answer_words(ctx, task, response, answered[-1].responded_by.id) is None:  # B3, the review's G1
                return "Refused: the answer carried is the person's own words, and this turn holds none for that response."
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
        try:
            ctx.work.record_action(ctx.project_id, held)
        except ValueError:
            # another turn of the same agent (a second deliverer: the relay and a worker) wrote this attempt first: it is the
            # same attempt (its id is the retry key's), so this turn goes on from the stored record and adds nothing
            again = next((a for a in _actions(ctx, task.id) if a.action_id == aid), None)
            if again is None:
                raise
            held = again
            if tw.can_claim(held):
                return held, True
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


# B4.2: the values a question offers the person to pick travel as a list (``team_clarify.options``), never guessed from its
# words. Code writes them under the question in one fixed form, and reads them back from it (``read_choices``): the question's
# own words are its premise, every number in them must be on record; each option is a value the person may pick.
CHOICES_HEAD = "Choices for you to pick:"
_CHOICE_SEP = " · "
OPTION_LIMIT, OPTIONS_MIN, OPTIONS_MAX = 120, 2, 6
_CHOICE_LABEL = re.compile(r"^\(([a-f])\) ")


def render_choices(question: str, options: Optional[list[str]]) -> str:
    """The question as the person reads it: its words, then on a line of their own the options code lists, (a) to (f)."""
    if not options:
        return question
    listed = _CHOICE_SEP.join(f"({chr(97 + i)}) {o.strip()}" for i, o in enumerate(options))
    return f"{question.rstrip()}\n{CHOICES_HEAD} {listed}"


def read_choices(text: str) -> tuple[str, list[str]]:
    """(the question's own words, the options code listed under it): the inverse of :func:`render_choices`. A text with no
    code-written line of choices is all premise."""
    text = str(text or "")
    at = text.rfind("\n" + CHOICES_HEAD + " ")
    if at < 0:
        return text, []
    question, line = text[:at], text[at + len(CHOICES_HEAD) + 2:]
    parts = line.split(_CHOICE_SEP)
    options = []
    for i, part in enumerate(parts):
        m = _CHOICE_LABEL.match(part)
        if not m or m.group(1) != chr(97 + i):
            return text, []  # not the form code writes: all premise
        options.append(part[m.end():])
    return question, options


def options_problem(question: str, options: Optional[list[str]]) -> Optional[str]:
    """Why a question's options cannot be sent, or None."""
    if CHOICES_HEAD in str(question or ""):
        return f"the question may not hold the words {CHOICES_HEAD!r}: code writes that line from options"
    if not options:
        return None
    if not OPTIONS_MIN <= len(options) <= OPTIONS_MAX:
        return f"offer {OPTIONS_MIN} to {OPTIONS_MAX} options, or none"
    for o in options:
        o = str(o or "")
        if not o.strip() or len(o) > OPTION_LIMIT or "\n" in o or _CHOICE_SEP.strip() in o or CHOICES_HEAD in o:
            return (f"each option is one line of at most {OPTION_LIMIT} characters, not empty, without "
                    f"{_CHOICE_SEP.strip()!r} or {CHOICES_HEAD!r}")
    if len(render_choices(question, options)) > tw.TEXT:
        return f"the question and its options together pass {tw.TEXT} characters"
    return None


def _clarify(ctx: TeamTurn, task_id: str, action: str, question: str, options: Optional[list[str]] = None) -> Row:
    """Ask a person. The owner's question moves its task to waiting_for_person and travels to the requester as a
    clarification (the task stays open); the requesting agent that receives it records that it asked its person (the
    delivery posts the question into the person's thread). B4.2: ``options``, the values offered to pick, are written by
    code under the question."""
    if not options and CHOICES_HEAD in str(question or ""):
        question, options = read_choices(question)  # a question relayed whole (the requester's rules ask what they received)
    problem = options_problem(question, options)
    if problem:
        return {"error": f"Not sent: {problem}."}
    question = render_choices(question, options)
    task = _task(ctx, task_id)
    held, duplicate = record(ctx, task, action, input_revision=f"brief-{task.brief_revision}")
    if tw.can_claim(held):
        return {"task_id": task.id, "action": _line(held), "duplicate": True}
    if task.owner_team == ctx.agent and own_team_binding(ctx.config.binding(action) or ""):
        # T3: a question for the agent's own team (Sales asks its members for their capacity): asked in its own thread, the
        # task stays with it (accepted) and nothing travels to the requester; a member's answer there is recorded as theirs
        post_own(ctx, task, question, key=f"asked:{held.action_id}:{task.id}", kind="clarification", to_team=ctx.agent)
        return {"task_id": task.id, "asked": question, "of": "own_team", "action": _line(held), "duplicate": duplicate}
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
    out = {"task_id": task.id, "action": _line(ended), "responded_by": responded_by, "duplicate": duplicate}
    if action == TARGET_ACTION and task.owner_team == ctx.agent and responded_by:
        # T1: the person's answer to "what does the outcome mean?" is the task's confirmed outcome when its words are clear
        key = f"answer:{task.id}:r-{safe_id(response_id)}"
        words = next((d["text"] for d in ctx.work.task_view(ctx.project_id, task.id)["deliveries"] if d["key"] == key), None)
        target = confirmed_outcome(task, words, responded_by, response_id, turn=ctx)
        out["outcome"] = target.event if target is not None else "unsure"
        if target is not None:
            from . import seats

            save_outcome(ctx.env, ctx.work, ctx.project_id, task, target, by=ctx.actor, initiator=ctx.initiator,
                         read_by="helper" if seats.reading(target) else None)  # B2: the keyword reader could not: the helper read it
    return out


# ---------------------------------------------------------------------------------------------------------- T1: the outcome confirmed
TARGET_ACTION = "confirm_target"  # Data Science's human-bound action: the person says what the outcome means


def sample_of(task: tw.TeamTask) -> Optional[str]:
    return next((i for i in task.allowed_inputs if i.startswith("sample:")), None)


def confirmed_outcome(task: tw.TeamTask, words: Optional[str], person: str, response_id: str, turn: Any = None) -> Optional[Any]:
    """The person's words as the task's confirmed outcome (``target_contract.TargetContract``), or None when they do not
    settle it (``target_contract.read_event``: two events, none, one both ways, a hedge). The event is read from the events
    the task's table records and the others DCLab reads, so a person can name one the table lacks (that is then a
    blocker, never a silent swap). The rest of the contract is the sample's (``fixtures.terms``), on the brief revision
    ``team_work.target_revision`` gives.

    B2: inside a turn (``turn``), where the keyword reader cannot read the words, the requester's helper may (``seats``:
    only on the helper arm, and its event is used only when it is allowed and the table's label records it); else None."""
    from dclab_core import target_contract as tc

    event = tc.read_event(words, fixtures.offered(sample_of(task)))
    if event is None and turn is not None:
        from . import seats

        event = seats.of(turn).read_target(task, words, where="answer")
        if event is not None:  # the review of B2, F7: marked, so a report says the helper read the person's answer
            return contract_for(task, event, words, person, f"{seats.HELPER_ANSWER}{response_id}")
    if event is None:
        return None
    return contract_for(task, event, words, person, response_id)


def contract_for(task: tw.TeamTask, event: str, words: Optional[str], person: str, response_id: str) -> Any:
    """The confirmed outcome of ``event`` in the person's ``words``, with the rest of the sample's contract."""
    from dclab_core import target_contract as tc

    sample = sample_of(task)
    terms = {k: v for k, v in fixtures.terms(sample, task.question).items() if v}
    return tc.TargetContract(event=event, words=str(words)[:tc.TEXT], brief_revision=tw.target_revision(task, event),
                             confirmed_by=str(person)[:120], response=safe_id(response_id), **terms)


def save_outcome(env: Any, work: Any, project_id: str, task: tw.TeamTask, target: Any, *, by: tw.Actor, initiator: Any = None,
                 read_by: Optional[str] = None) -> Any:
    """Save a confirmed outcome (``store.confirm_target``), queue the answer rows it sent (one per task that works again on
    the new revision), and record it (``team.target.confirmed``: codes, ids and counts, never the person's words; B2:
    ``read_by`` ``helper`` when the helper read the event from the person's words and code checked it)."""
    before = task.target
    written = work.confirm_target(project_id, task.id, target, by=by, initiator=initiator)
    if written.duplicate:
        return written
    from . import delivery

    for row in written.rows:
        try:
            delivery.enqueue(env, row)
        except Exception:  # noqa: BLE001 — as in _queue: the row is saved, and the relay queues it on its next tick
            pass
    try:
        from .. import experience

        payload = {"task": task.id, "problem": task.problem_id, "event": target.event, "by": target.confirmed_by,
                   "brief_revision": written.item.brief_revision, "source": target.label_source, "changed": bool(written.changed),
                   "previous": before.event if before is not None else None, "stale": len(written.stale), "response": target.response,
                   "read_by": read_by}
        experience.record("team.target.confirmed", "agent" if read_by else "person", {k: v for k, v in payload.items() if v is not None},
                          project=project_id)
    except Exception:  # noqa: BLE001 — the record is best effort: the outcome stands
        pass
    return written


def judge_outcome(ctx: TeamTurn, task: tw.TeamTask, check: Row) -> None:
    """Record where a team judged its table against the confirmed outcome (``inspect_schema``, ``validate_contract``): the
    ledger point ``team.target_agreement`` and, on a mismatch, ``team.target.mismatch`` (codes and counts only)."""
    verdict = check.get("verdict")
    if task.target is None or verdict is None:
        return
    from ..ledger import capture

    try:
        capture.team_target_agreement(ctx.project_id, task.id, verdict.kind, event=verdict.event, label_columns=len(verdict.columns),
                                      brief_revision=task.brief_revision, changed=task.brief_revision > 1, by=ctx.agent)
    except Exception:  # noqa: BLE001 — the ledger is best effort here: the check stands
        pass
    if verdict.ok:
        return
    try:
        from .. import experience

        payload = {"task": task.id, "problem": task.problem_id, "event": verdict.event, "verdict": verdict.kind, "missing": check.get("field"),
                   "recorded": list(verdict.recorded), "columns": len(verdict.columns), "source": sample_of(task),
                   "supplier": check.get("supplier"), "by": ctx.agent, "brief_revision": task.brief_revision}
        experience.record("team.target.mismatch", "agent", {k: v for k, v in payload.items() if v is not None}, project=ctx.project_id)
    except Exception:  # noqa: BLE001
        pass


def _answer(ctx: TeamTurn, task_id: str, response_id: str, text: str) -> Row:
    """The person's answer to the owner, as an ``answer`` delivery saved once per response (``store.send_answer``). The
    owner takes its task up again when the row reaches it."""
    task = _task(ctx, task_id)
    words = answer_words(ctx, task, response_id)  # B3, the review's G1: the person's own words, never the caller's
    written = ctx.work.send_answer(ctx.project_id, task.id, response_id=safe_id(response_id), text=str(words if words is not None else text),
                                   initiator=ctx.initiator)
    if not written.duplicate:
        _send(ctx, written)
    return {"task_id": task.id, "delivery": written.delivery["id"], "duplicate": written.duplicate}


def _delegate(ctx: TeamTurn, action: str, new_task_id: str, owner_team: str, question: str, expected_output: str,
              done_criteria: list[str], task_id: Optional[str] = None, problem_id: Optional[str] = None,
              thread_id: Optional[str] = None, allowed_inputs: Optional[list[str]] = None, budget_eur: float = 0.0,
              brief_revision: int = 1, target_from: Optional[str] = None) -> Row:
    """A task for another team: a new top-level task (``task_id`` omitted: the requesting team agent asks for its person,
    who is the case's member) or a child of the acting agent's task (``team_work.delegate``). Saved with its delivery;
    the delegation is the acting agent's action, recorded on the task it was made from. ``target_from`` (T1): a child
    carries the outcome confirmed on that task (the parent or a sibling), copied by the store as stored."""
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
                             parent_waits=True, target_from=target_from)
        _send(ctx, made)
        on = parent
    done, _ = record(ctx, on, action, input_revision=new_task_id, status="succeeded", output_ref=f"task:{new_task_id}")
    return {"task_id": new_task_id, "owner_team": owner_team, "action": _line(done), "duplicate": made.duplicate}


def _act(ctx: TeamTurn, task_id: str, action: str, subject: Optional[str] = None) -> Row:
    """A directory action bound to a tool this release provides, computed by code from what the task may read."""
    task = _task(ctx, task_id)
    tool = (ctx.config.binding(action) or "")[5:]
    if tool in COMMERCIAL_ACTS:  # T3: Sales and Marketing compute their own (teamspace/commercial.py), recorded the same way
        from . import commercial

        return commercial.act(ctx, task, action, tool, subject)
    status, ref, revision = "succeeded", None, f"brief-{task.brief_revision}"
    sample_ref = next((i for i in task.allowed_inputs if i.startswith("sample:")), None)
    if tool == "inspect_schema":
        found = fixtures.sample(sample_ref) if sample_ref else None
        dispatch = next((i for i in task.allowed_inputs if i.startswith("dispatch:")), None)
        if found is None and sample_ref is None and dispatch:  # B4.1: what it reads instead, so the agent knows where to go
            status, ref = "failed", (f"no table for {task.id}: it reads the dispatch snapshot {dispatch}, which the model specialists "
                                     "estimate from (model:eta, model:acceptance)")
        elif found is None:
            status, ref = "failed", f"no permitted sample for {task.id}"
        else:
            check = fixtures.feasibility(sample_ref, task.target)  # T1: against the outcome the person confirmed, when one is
            ref = f"schema:{sample_ref}:{len(found['columns'])} columns" + ("" if check["ok"] else f"; missing {check['missing']}")
            judge_outcome(ctx, task, check)
    elif tool in ("leakage_check", "define_baseline", "explain_uncertainty") and sample_ref is None:
        # B4.1 (D-03, D-07, D-21 of the first dev run): no table to read (a dispatch snapshot, or nothing): never a renewal
        # table's words. A dispatch is estimated by the model specialists; the checks a table answers do not apply
        dispatch = next((i for i in task.allowed_inputs if i.startswith("dispatch:")), None)
        what = (f"the task reads the dispatch snapshot {dispatch}, not a table: its estimates come from the model specialists "
                "(model:eta for the arrival, model:acceptance for the first offer)") if dispatch else f"no permitted sample for {task.id}"
        if tool == "explain_uncertainty":
            ref = (f"uncertainty:{what}; their results are fixtures, not measured" if dispatch else f"uncertainty:{what}")
        else:
            status, ref = "failed", f"no table to {'check for leakage' if tool == 'leakage_check' else 'name a baseline for'}: {what}"
    elif tool == "leakage_check":
        check = fixtures.feasibility(sample_ref or "", task.target)
        ref = "excluded:" + (",".join(check.get("excluded") or ()) or "none") + " (known only after renewal)"
    elif tool == "define_baseline":
        ref = "baseline:rank by days since last login (the simple rule a model must beat)"
    elif tool == "explain_uncertainty":
        ref = "uncertainty:feasibility from the sample's columns only; no account was scored"
    elif tool == "check_prerequisites":
        # what a coordinated task needs before teams are asked: its question and at least one permitted input to read
        readable = [i for i in task.allowed_inputs if not i.startswith("help:")]  # help:<track> is the Lead's own, not data
        if not readable:
            status, ref = "failed", "no permitted input: the project declares nothing the teams may read"
        else:
            ref = f"prerequisites:{len(readable)} permitted inputs; question of {len(task.question)} characters"
    elif tool == "validate_contract":  # ML Engineering (0.168.0): the experiment's contract from the sample's columns, never a run
        check = fixtures.feasibility(sample_ref or "", task.target)  # T1: the outcome Data Science confirmed, carried as a field
        judge_outcome(ctx, task, check)
        if not check["ok"]:
            status, ref = "failed", f"contract incomplete: {check['why']}"
        else:
            ref = (f"contract:{sample_ref}: cutoff from {check['renewal_date']}; outcome {check['outcome']}; time-ordered holdout; "
                   f"excluded {','.join(check['excluded']) or 'none'}")
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
    if status == "succeeded" and tool in ("inspect_schema", "leakage_check", "define_baseline", "explain_uncertainty", "validate_contract") \
            and sample_ref in fixtures.SAMPLES:
        status = "simulated"  # a check on an invented sample is labelled as such, so the report says it is simulated
    ended, duplicate = record(ctx, task, action, input_revision=revision, status=status,
                              output_ref=ref if status in tw.DONE_ACTIONS else None,
                              failure_ref=ref if status in tw.FAILED_ACTIONS else None)
    return {"task_id": task.id, "action": _line(ended), "duplicate": duplicate}


CORRECTED = "kitchen_minutes"  # T2: the field a correction of the arrival estimate changes (the kitchen's minutes, 0: none)
MAX_MINUTES = 240  # a kitchen's delay is 0 to four hours: beyond, the fixed-route estimate means nothing (and wraps past midnight)


def _tag(correction_id: Optional[str]) -> Optional[str]:
    return f"correction {safe_id(correction_id)}" if correction_id else None


def _tagged(action: tw.TeamAction, tag: Optional[str]) -> bool:
    # ": " ends the tag (a safe id has no space; it may have ':', so id "k" must not match the reason of id "k:2")
    return bool(tag) and bool(action.reason) and (action.reason == tag or action.reason.startswith(tag + ": "))


def judge_correction(work: Any, project_id: str, task: tw.TeamTask, minutes: int, base_revision: Optional[int] = None,
                     correction_id: Optional[str] = None) -> Row:
    """T2: what a correction of the specialist's answer is, judged on the stored task (``team_work.accept_correction``).

    The correction is based on ``base_revision`` (default: the current report's revision) and becomes the score's input
    revision ``team_work.correction_revision(snapshot, kitchen_minutes, minutes, base)``. It is a ``duplicate`` when a
    report already carries that revision, when its ``correction_id`` was already applied (the same correction delivered
    again, whatever base it now names), or when it would not change the current answer (the same minutes on the current
    report: a zero with no delay to withdraw); ``stale_base`` when it names an older report than the current one; else
    ``apply``. A score of it left running by a crash is not an application: it is finished, never scored twice. Refused
    (ValueError): no report to correct, minutes outside 0 to :data:`MAX_MINUTES`, a base that names no report, no base
    (and no known correction id) once the report has been revised (a late resend would otherwise land on the newest
    answer it never saw), a correction id reused for another correction."""
    view = work.task_view(project_id, task.id)
    snapshot = next((i for i in task.allowed_inputs if i.startswith("dispatch:")), "")
    if not snapshot:
        raise ValueError(f"{task.id} reads no dispatch snapshot: there is no estimate to correct")
    current = max((r for r in view["reports"] if r.status != "superseded"), key=lambda r: r.revision, default=None)
    if current is None:
        raise ValueError(f"{task.id} has no report to correct yet")
    minutes = int(minutes)
    if not 0 <= minutes <= MAX_MINUTES:
        raise ValueError(f"the kitchen's minutes are between 0 and {MAX_MINUTES}, not {minutes}")
    tag = _tag(correction_id)
    earlier = next((a for a in view["actions"] if a.action == "score" and _tagged(a, tag)), None)
    if earlier is not None:  # the same correction id: the correction it named then, never a new one
        read = tw.read_correction(earlier.input_revision)
        if read is None or int(read[1]) != minutes or (base_revision is not None and int(base_revision) != read[2]):
            raise ValueError(f"correction {correction_id} was already given for another correction of {task.id}")
        base, revision = read[2], earlier.input_revision
    else:
        if base_revision is None and current.revision > 1:
            raise ValueError(f"{task.id} was revised (now {task.id}@r{current.revision}): a correction names the revision "
                             "it is based on")
        base = current.revision if base_revision is None else int(base_revision)
        tw.accept_correction(current.revision, base)  # a base that names no report: refused before anything is built
        revision = tw.correction_revision(snapshot, CORRECTED, minutes, base)
    carried = {e.input_revision for r in view["reports"] for e in r.evidence}
    now = tw.read_correction(current.evidence[0].input_revision) if current.evidence else None
    unchanged = earlier is None and base == current.revision and minutes == (int(now[1]) if now else 0)
    verdict = tw.accept_correction(current.revision, base, already_applied=revision in carried or unchanged)
    return {"verdict": verdict, "revision": revision, "base": base, "current": current.revision, "minutes": minutes,
            "id": safe_id(correction_id) if correction_id else None, "tag": tag, "report": current.id,
            "unchanged": unchanged and revision not in carried}


def _reconcile(ctx: TeamTurn, task: tw.TeamTask, judged: Row, action: tw.TeamAction) -> Optional[Row]:
    """A correction on an older report: the requester reads, in its thread, that nothing changed and what to reconcile
    against (once per refused attempt)."""
    found = _threads(ctx.env)
    if found is None:
        return None
    store = found[0]
    current = f"{task.id}@r{judged['current']}"
    text = (f"Please reconcile: a correction of {task.id} (the kitchen's {judged['minutes']} minutes) is based on "
            f"{task.id}@r{judged['base']}, but the current report is {current}. Nothing was changed; if it still applies, "
            f"send it again against {current} as a new correction (with a new correction id).")
    author = {"kind": "agent", "id": ctx.agent, "team": ctx.agent}
    handoff = {"kind": "reconcile", "task_id": task.id, "item_id": current, "from_team": ctx.agent, "to_team": task.requester.team}
    key = f"reconcile:{task.id}:{action.action_id}"
    try:
        message, _ = store.post(task.requester.thread_id, author, text, handoff=handoff, key=key)
    except KeyError:  # the requester's thread is not held here: its team's thread on the problem (opened the first time)
        thread_id = team_thread(ctx.env, task.requester.team, ctx.project_id, task.problem_id)
        message, _ = store.post(thread_id, author, text, handoff=handoff, key=key)
    return message


def _model_call(ctx: TeamTurn, task_id: str, action: str, correction_minutes: Optional[int] = None,
                base_revision: Optional[int] = None, correction_id: Optional[str] = None) -> Row:
    """The fixture model specialist: a version's estimate for the task's dispatch snapshot, labelled simulated. A
    correction (the kitchen's minutes, 0 included: the delay withdrawn) is a new attempt on the input revision that names
    its base report (:func:`judge_correction`); one already applied writes nothing; one on an older report than the
    current one is a skipped attempt and a "please reconcile" reply to the requester. A missing version is a denial."""
    task = _task(ctx, task_id)
    kind = ctx.agent.split(":", 1)[1] if ":" in ctx.agent else ""
    snapshot = next((i for i in task.allowed_inputs if i.startswith("dispatch:")), "")
    revision, tag = snapshot, None
    if correction_minutes is not None and action == "score":
        judged = judge_correction(ctx.work, ctx.project_id, task, correction_minutes, base_revision, correction_id)
        revision, tag = judged["revision"], judged["tag"]
        if judged["verdict"] == "duplicate":
            return {"task_id": task.id, "verdict": "duplicate", "revision": revision, "duplicate": True}
        if judged["verdict"] == "stale_base":
            why = (f"based on {task.id}@r{judged['base']}; the current report is {task.id}@r{judged['current']}: "
                   "please reconcile")
            ended, duplicate = record(ctx, task, action, input_revision=revision, status="skipped",
                                      reason=f"{tag}: {why}" if tag else why)
            _reconcile(ctx, task, judged, ended)
            return {"task_id": task.id, "verdict": "stale_base", "action": _line(ended), "duplicate": duplicate}
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
        value = fixtures.eta_after(found["arrival"], int(correction_minutes) if correction_minutes is not None else 0)
    else:
        value = f"{round(float(found['probability']) * 100)}%"
    ended, duplicate = record(ctx, task, action, input_revision=revision, status="simulated",
                              output_ref=f"fixture:{found['id']}={value}", reason=tag)  # tag: the correction's id, if any
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


# ---------------------------------------------------------------------------------------------------------- B3: plan and read
PLAN_STEP = re.compile(r"^\s*(?:\*\*)?(?:\d+[.)]|[-*•])(?:\*\*)?\s+\S")
TEXT_SHOWN = 1500  # what a read shows of a question, an answer or a message


def plan_steps(plan: str) -> int:
    """The steps a plan lists: its numbered or bulleted lines (0 when it lists none)."""
    return sum(1 for line in str(plan or "").splitlines() if PLAN_STEP.match(line))


def _plan(ctx: TeamTurn, plan: str) -> Row:
    """The agent's plan for this turn (B3): written before its first action, kept whole by the agent arm (its turn's
    journal and the model's exchange). It changes no team work: a plan is an answer, never an action."""
    from dclab_core.team_brain import PLAN_CHARS

    said = str(plan or "").strip()
    if not said:
        raise ValueError("a plan says what this turn will do, in order")
    if len(said) > PLAN_CHARS:
        raise ValueError(f"a plan is at most {PLAN_CHARS} characters: write a shorter one")
    return {"plan": "kept", "chars": len(said), "steps": plan_steps(said)}


def _cut(text: Any, chars: int = TEXT_SHOWN) -> str:
    said = str(text or "")
    return said if len(said) <= chars else said[:chars] + "…"


def _read_task(ctx: TeamTurn, task_id: str) -> Row:
    """What the acting agent may see of a task it owns or asked for: its brief, its state, its action log, its reports, the
    tasks it handed out with their reports, and what travelled on it (the person's answers included). The team's records
    only: never a table, a row or a value."""
    why = TeamGuard.read_refusal(ctx, task_id=task_id)
    if why:
        return {"error": why}
    view = ctx.work.task_view(ctx.project_id, str(task_id))
    task = view["task"]
    out: Row = {"task": {"id": task.id, "status": task.status, "owner": task.owner_team, "requester": task.requester.team,
                         "parent": task.parent_task_id, "brief_revision": task.brief_revision, "question": _cut(task.question),
                         "expected_output": task.expected_output, "done_criteria": list(task.done_criteria),
                         "allowed_inputs": list(task.allowed_inputs), "target": task.target.event if task.target is not None else None},
                "actions": [{"action_id": a.action_id, "action": a.action, "by": a.actor.id, "status": a.status, "input_revision": a.input_revision,
                             "ref": a.output_ref or a.failure_ref, "reason": a.reason,
                             "responded_by": a.responded_by.id if a.responded_by is not None else None} for a in view["actions"]],
                "reports": [{"id": r.id, "revision": r.revision, "status": r.status, "answer": _cut(r.answer), "limits": list(r.limits),
                             "next_owner": r.next_task.owner, "unblocks": r.next_task.unblocks, "simulated": r.simulated,
                             "brief_revision": r.brief_revision, "child_reports": list(r.child_reports)} for r in view["reports"]],
                "children": [{"id": c.id, "owner": c.owner_team, "status": c.status} for c in view["children"]],
                "child_reports": [{"id": r.id, "task": r.task_id, "owner": r.owner, "status": r.status, "answer": _cut(r.answer),
                                   "limits": list(r.limits), "next_owner": r.next_task.owner, "simulated": r.simulated}
                                  for r in view["child_reports"]],
                "travelled": [{"kind": d["kind"], "from": d["from_team"], "to": d["to_team"], "status": d["status"], "key": d["key"],
                               "text": _cut(d.get("text"), 600)} for d in view["deliveries"]]}
    if view.get("join") is not None:
        out["outstanding"] = list(view["join"].outstanding)
    return out


def _read_notes(ctx: TeamTurn, team: str, task_id: Optional[str] = None) -> Row:
    """The acting team's own notes on the task's problem: its team thread's latest messages. Another team's notes stay in
    that team (refused)."""
    why = TeamGuard.read_refusal(ctx, team=team, task_id=task_id)
    if why:
        return {"error": why}
    task = ctx.work.task(ctx.project_id, str(task_id or ctx.task_id or ""))
    if task is None:
        return {"team": team, "messages": []}
    found = _threads(ctx.env)
    if found is None:
        return {"team": team, "messages": []}
    from .threads import team_key

    store, teams = found
    thread = store.find_thread(ctx.project_id, task.problem_id, team_key(teams, ctx.agent)[0])
    messages = store.messages(thread["id"]) if thread else []
    return {"team": team, "problem": task.problem_id,
            "messages": [{"seq": m.get("seq"), "by": (m.get("author") or {}).get("id"), "kind": (m.get("handoff") or {}).get("kind"),
                          "task": (m.get("handoff") or {}).get("task_id"), "text": _cut(m.get("text"), 600)} for m in messages[-20:]]}


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


def _text(limit: int, description: Optional[str] = None) -> dict[str, Any]:
    """A text field with the store's own limit (B4.1: the first dev run had 18 texts refused for their length, a limit the
    model never saw): the model reads it in the schema, and the registry refuses a longer one before anything runs. Only on
    a field the store itself limits: never on a person's own words (``team_answer.text``), which travel whole."""
    return {"type": "string", "maxLength": limit, **({"description": description} if description else {})}
_ACTION = {"type": "string", "description": "The directory action this call performs."}

TOOLS = (
    Tool("team_accept", "Accept a task this team owns.", _schema({"task_id": _TASK}, ["task_id"]), _accept,
         effect="write", scope=SCOPE, move="accept", takes_context=True, errors=_errors),
    Tool("team_block", "Block a task this team owns: its report will name what unblocks it.",
         _schema({"task_id": _TASK, "reason": _text(tw.SHORT, "What blocks it, in one sentence.")}, ["task_id", "reason"]), _block,
         effect="write", scope=SCOPE, move="block", takes_context=True, errors=_errors),
    Tool("team_clarify", "Ask a person one question for a human-bound directory action; the task stays open.",
         _schema({"task_id": _TASK, "action": _ACTION,
                  "question": _text(tw.TEXT, "Your question. Every number in it must be on record; put values you offer to pick "
                                             "in options."),
                  "options": {"type": "array", "minItems": OPTIONS_MIN, "maxItems": OPTIONS_MAX,
                              "items": _text(OPTION_LIMIT),
                              "description": "Optional: the values the person may pick, one short line each (for example "
                                             "\"7 days before renewal\", \"14 days before renewal\"). Code lists them "
                                             "under your question."}},
                 ["task_id", "action", "question"]), _clarify,
         effect="write", scope=SCOPE, move="clarify", takes_context=True, errors=_errors),
    Tool("team_record_human_response", "Record the person's response to a human-bound directory action.",
         _schema({"task_id": _TASK, "action": _ACTION, "response_id": _S,
                  "responded_by": {"type": "string", "description": "The person who responded, by their id."}},
                 ["task_id", "action", "response_id"]), _record_human_response,
         effect="write", scope=SCOPE, move="record_human_response", takes_context=True, errors=_errors),
    Tool("team_answer", "Carry the person's recorded answer to the task's owner.",
         _schema({"task_id": _TASK, "response_id": _S, "text": _S}, ["task_id", "response_id", "text"]), _answer,  # the person's own words: never cut short (B4.1 review, F1)
         effect="write", scope=SCOPE, move="answer", takes_context=True, errors=_errors),
    Tool("team_delegate", "Open a task for another team: a new top-level task, or a child of a task this team owns.",
         _schema({"action": _ACTION, "new_task_id": _S, "owner_team": _S, "question": _text(tw.TEXT),
                  "expected_output": _text(tw.SHORT, "What the team returns, in a short phrase."),
                  "done_criteria": {"type": "array", "items": _S, "minItems": 1}, "task_id": _TASK, "problem_id": _S,
                  "thread_id": _S, "allowed_inputs": {"type": "array", "items": _S}, "budget_eur": {"type": "number", "minimum": 0},
                  "brief_revision": {"type": "integer", "minimum": 1},
                  "target_from": {"type": "string", "description": "A task whose confirmed outcome the child carries."}},
                 ["action", "new_task_id", "owner_team", "question", "expected_output", "done_criteria"]), _delegate,
         effect="write", scope=SCOPE, move="delegate", takes_context=True, errors=_errors),
    Tool("team_act", "Perform a directory action bound to a tool this workspace provides.",
         _schema({"task_id": _TASK, "action": _ACTION, "subject": {"type": "string", "description": "The report it reads."}},
                 ["task_id", "action"]), _act,
         effect="write", scope=SCOPE, move="act", takes_context=True, errors=_errors),
    Tool("team_model_call", "Ask one model version (a fixture in this release, labelled simulated) for its estimate.",
         _schema({"task_id": _TASK, "action": _ACTION, "correction_minutes": {"type": "integer", "minimum": 0, "maximum": MAX_MINUTES},
                  "base_revision": {"type": "integer", "minimum": 1, "description": "The report revision a correction is based on."},
                  "correction_id": {"type": "string", "description": "The correction's own id: the same correction again adds nothing."}},
                 ["task_id", "action"]),
         _model_call, effect="write", scope=SCOPE, move="model_call", takes_context=True, errors=_errors),
    Tool("team_return_report", "Return the task's report, built from its action log and its children's reports.",
         _schema({"task_id": _TASK, "answer": _text(tw.TEXT), "outcome": {"type": "string", "enum": ["complete", "partial", "blocked"]},
                  "revision": {"type": "integer", "minimum": 1}, "next_owner": _S,
                  "unblocks": _text(tw.SHORT, "What unblocks it, in a short phrase."),
                  "evidence": {"type": "array", "items": {"type": "object"}}, "limits": {"type": "array", "items": _S}},
                 ["task_id", "answer", "outcome", "revision"]), _return_report,
         effect="write", scope=SCOPE, move="return_report", takes_context=True, errors=_errors),
    # B3 (2026-11-24): what an agent-arm turn adds to the team's tools. Reads and a plan: none changes team work, so none is
    # a write; each read asks the guard (TeamGuard.read_refusal) before it reads anything
    Tool("team_plan", "Write your plan for this turn before your first action: the steps in order, the tool each uses, whom "
                      "you hand work to, and when you stop. It is kept whole. Call it again to say how the plan changed.",
         _schema({"plan": {"type": "string", "maxLength": _PLAN_CHARS, "description": "The plan, one numbered step a line."}}, ["plan"]), _plan,
         scope=SCOPE, takes_context=True, errors=_errors),
    Tool("team_read_task", "Read a task this team owns or asked for: its brief, state, action log, reports, the tasks it handed "
                           "out and what travelled on it.",
         _schema({"task_id": _TASK}, ["task_id"]), _read_task, scope=SCOPE, takes_context=True, errors=_errors),
    Tool("team_read_notes", "Read this team's own notes on the task's problem (its team thread). Another team's notes stay in "
                            "that team.",
         _schema({"team": {"type": "string", "description": "The team whose notes to read: this team's own name."}, "task_id": _TASK},
                 ["team"]), _read_notes, scope=SCOPE, takes_context=True, errors=_errors),
)
AGENT_TOOLS = ("team_plan", "team_read_task", "team_read_notes")  # B3: offered to an agent-arm turn beside its directory's tools


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


__all__ = ["ACTS", "AGENT_TOOLS", "COMMERCIAL_ACTS", "MODEL_TOOLS", "MOVES", "SCOPE", "TOOLS", "TeamGuard", "TeamTurn", "action_id", "case_member",
           "holds_role", "is_error", "own_team_binding", "post_own", "record", "register", "safe_id", "team_thread",
           "TARGET_ACTION", "answer_words", "confirmed_outcome", "judge_outcome", "plan_steps", "sample_of", "save_outcome"]
