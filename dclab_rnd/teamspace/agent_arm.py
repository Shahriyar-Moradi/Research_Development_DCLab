"""The agent arm acts (B3 of the team workspace, 2026-11-24): a model runs a team agent's turn inside the strict layer.

A role switched to ``agent`` (``brains.switch``; every role defaults to the rules) no longer answers in shadow: on a
delivered row, its model runs the turn on the one agent loop (``agents.runtime.run``) with the team tools of the one
registry (``agents.default_registry``: the agent's directory tools, as the rules are offered them, and the three B3 adds:
``team_plan``, ``team_read_task``, ``team_read_notes``). Every write passes the same guard the rules pass
(``tools.TeamGuard``: the action directory, permissions, the task state machine, only the owner approves, nobody contacts
customers); every read asks the guard too (a team's notes stay in that team).

**Plan first, kept whole.** No write runs before the model's plan (``team_plan``): a write asked first is refused and the
model is told to plan. The plan is an answer (what it will do, in order), never its reasoning: that comes only from the
API's thinking (the gateway asks for it, ``models/settings.thinking_for``, and keeps each exchange whole with its
thinking summary). The plan's words are kept whole in the turn's journal (``<workspace>/teamspace/agent_turns/<run>.json``)
and in the model's exchange; the records hold its length, steps and hash.

**Free choice inside the strict layer.** The model chooses its actions, their order, its questions and hand-overs. Code
fixes what must not depend on a model's words: a child task is ``<task>.<team>`` (one per team per task, so the same
hand-over again is the same task), it reads no input its parent may not read (never the Lead's ``help:``), a new case is
opened only by its person's own message, and a report's revision is the next one (the same report again, with nothing new
behind it, is the stored one).

**Reports.** The model writes the findings and the recommendation (``answer``) and may add limits. Code keeps a sentence
only when every number in it is a number of a record code or a person wrote (an action's output, a returned report, the
person's own request (the top task's question: a handed-out task's question is an agent's words), a person's answer: ``cited.numbers``; an identifier's digits are no number, and a model's own words read back, such
as a question it asked or an id it sent, are no record), it claims no action without its terminal record (``team_brain.unrecorded_claims``), and it
neither overclaims (``cited.overclaims``) nor promises an action. Code then attaches the numbers itself: each returned
team's answer and each recorded action's output, and builds the evidence from the action log and the returned reports
(the model's evidence is never used). The kept and refused sentences are in the journal; their counts and reasons in the
records (``team.agent.turn``, the ledger point ``team.agent_report``).

**Budgets** (``DCLAB_TEAM_AGENT_STEPS`` 16 tool calls, ``DCLAB_TEAM_AGENT_SECONDS`` 180, ``DCLAB_TEAM_AGENT_EUR`` 0.10 per
turn), checked by code: the loop's own steps and seconds, each request given only the seconds left, no request once the
euros are spent, and a reply that arrives after the seconds ended is refused. When one ends the turn ends with what was
done recorded (``ended``).

**Fallback, before the first action only.** No model (none configured, ``DCLAB_NO_LIVE_MODELS=1``), a refused, late,
invalid or failing request, a budget that ends before any write, or a plan followed by no write on a task still open
(``no_action``: every write refused, or none made; the work never waits on a model that did nothing): the turn returns :class:`Fallback` and the rules run it
(``agents.run_turn``), the reason recorded (``team_brain.FALLBACKS``). After a write the rules never run on top (they would
repeat the work): the turn ends and its partial state is recorded. A turn a person's own message starts (their case opened
in their words, their answer recorded) is code's (``person_entry``).

**One turn per delivery** (the B1 review's risk: a slow turn missed the delivery's limit and the delivery retried while
the first attempt ran). The turn takes a lease on its delivery row (``store.agent_turn_claim``): while it runs, the row is
not due (the relay and a second deliverer leave it), a failed attempt waits for the lease's end, and a row handed over
directly raises ``delivery.TurnHeld`` (nothing is started). The writes a turn starts are counted on the row *before* each
runs; a turn cut off after one (a crash) is marked ended and never run again, by the model or the rules; one cut off
before any write runs again. A row handed over after its turn ended starts nothing (``repeat``); one handed over after its
turn fell back to the rules is the rules' again (they read the stored state, so they repeat nothing), never a model's. The agent-arm row gets
its own delivery limit (:func:`turn_seconds`); the rules keep theirs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from dclab_core import team_brain as core
from dclab_core import team_work as tw

PURPOSE = "team_agent"
PROMPT = "team_agent"
ENV_STEPS, ENV_SECONDS, ENV_EUR = "DCLAB_TEAM_AGENT_STEPS", "DCLAB_TEAM_AGENT_SECONDS", "DCLAB_TEAM_AGENT_EUR"
DEFAULT_STEPS, DEFAULT_SECONDS, DEFAULT_EUR = 16, 180.0, 0.10
MAX_TOKENS = 4000  # room for the thinking and one reply
DELIVERY_MARGIN = 30.0  # on top of the turn's seconds: its records, a slow store
LEASE_MARGIN = 120.0  # the lease outlives the turn and the delivery's own limit
MODEL_CHARS = 900  # the model's findings in a report answer, at most (code's lines from the records follow)
ERROR_CHARS = 300
JOURNAL = "agent_turns"
BUDGET_STOPS = ("steps", "time", "euros", "replies")
_NO_FINDING = ("task:", "reviewed:", "presented:", "response:")  # an action's record that names what it touched, not what it found
NO_ANSWER_STOPS = ("max_output_tokens", "tools_as_text")


# ---------------------------------------------------------------------------------------------------------- the budgets
def _number(name: str, default: float, low: float) -> float:
    try:
        value = float(os.environ.get(name) or default)
    except ValueError:
        return default
    return value if value >= low else default


def budgets() -> dict[str, float]:
    """Steps (tool calls), seconds and euros one agent turn may use (``DCLAB_TEAM_AGENT_*``; a value that does not parse is
    the default, and ``settings.Settings.problems`` names it)."""
    return {"steps": int(_number(ENV_STEPS, DEFAULT_STEPS, 1)), "seconds": _number(ENV_SECONDS, DEFAULT_SECONDS, 0.1),
            "euros": _number(ENV_EUR, DEFAULT_EUR, 0.0)}


def turn_seconds() -> float:
    """What an agent-arm delivery adds to the rules' limit: the turn's seconds and a margin (``delivery.timeout_seconds``)."""
    return budgets()["seconds"] + DELIVERY_MARGIN


def lease_seconds() -> float:
    return turn_seconds() + LEASE_MARGIN


def acts_on(env: Any, row: Optional[Mapping[str, Any]]) -> bool:
    """Whether a model acts on this delivered row: its recipient is a switchable role with rules to fall back to, on the
    agent arm."""
    if not row:
        return False
    base = str(row.get("to_team") or "").split(":")[0]
    if base not in core.ROLES:
        return False
    from . import brains
    from .agents import BRAINS

    return base in BRAINS and brains.arm_for(env, base)[0] == "agent"


# ---------------------------------------------------------------------------------------------------------- one turn
@dataclass
class Fallback:
    """The agent arm did not act before its first write: the rules run the turn. ``model`` is the brain's part of
    ``team.brain.answered``."""

    reason: str
    run: Optional[str] = None
    request: Optional[str] = None

    @property
    def model(self) -> dict[str, Any]:
        return {"purpose": PURPOSE, "tool": None, "request": self.request, "fallback": self.reason}


@dataclass
class _State:
    turn: Any
    budget: dict[str, float]
    run: Optional[str] = None
    delivery: Optional[str] = None
    token: Optional[str] = None
    until: Optional[float] = None  # the lease's end, as time.monotonic()
    started: float = field(default_factory=time.monotonic)
    plan: Optional[str] = None
    plans: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    writes: int = 0  # writes that ran
    begun: int = 0  # writes started (counted on the row before each runs; a refused one is taken back)
    refused: int = 0
    first_write: Optional[str] = None
    reports: list[dict[str, Any]] = field(default_factory=list)
    called: Optional[dict[str, Any]] = None  # what the registry's last call was (a write, run or stood in for, a plan)
    ended: Optional[str] = None  # a budget the model's client ended the turn on
    failure: Optional[str] = None  # why the model's request failed
    request: Optional[str] = None
    requests: list[str] = field(default_factory=list)
    euros: float = 0.0

    def seconds(self) -> float:
        return time.monotonic() - self.started


def _failure(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError) or type(exc).__name__ in ("APITimeoutError", "ReadTimeout", "Timeout"):
        return "timeout"
    if isinstance(exc, RuntimeError) and str(exc).startswith("BudgetExceeded"):
        return "refused"
    return "error"


def _stop(text: str) -> dict[str, Any]:
    return {"content": text, "tool_calls": [], "finish_reason": "stop", "usage": {"input_tokens": 0, "output_tokens": 0},
            "assistant_message": {"role": "assistant", "content": text}}


class _Model:
    """The model's client as the loop sees it: one attempt per request, given only the seconds left; no request once the
    euros are spent; a reply that arrives after the seconds ended is refused (the loop then ends on its budget)."""

    def __init__(self, client: Any, state: _State):
        self.client, self.state = client, state
        self.last_request: Optional[str] = None

    def complete(self, messages: list[dict[str, Any]], tools: Any = None, **_: Any) -> dict[str, Any]:
        from ..models import settings as model_settings

        state, budget = self.state, self.state.budget
        if budget["euros"] > 0 and float(getattr(self.client, "spent_eur", 0.0) or 0.0) >= budget["euros"]:
            state.ended = "euros"
            return _stop("[DCLab's code] The turn's euro budget is spent.")
        left = budget["seconds"] - state.seconds()
        if left <= 0:
            state.ended = "time"
            return _stop("[DCLab's code] The turn's time is up.")
        try:
            wait = min(float(model_settings.purpose(PURPOSE).timeout), left)
        except Exception:  # noqa: BLE001
            wait = left
        try:
            out = self.client.complete(messages, tools, MAX_TOKENS, single_attempt=True, timeout_seconds=max(0.05, wait))
        except RuntimeError as exc:
            if str(exc).startswith("BudgetExceeded: this run has used its budget"):  # the gateway's hold on the turn's euros
                state.ended = "euros"
                return _stop("[DCLab's code] The turn's euro budget is spent.")
            state.failure = _failure(exc)
            raise
        except BaseException as exc:
            state.failure = _failure(exc)
            raise
        finally:
            self.last_request = getattr(self.client, "last_request", None)
            if self.last_request and self.last_request != state.request:
                state.request = self.last_request
                state.requests.append(self.last_request)
            state.euros = float(getattr(self.client, "spent_eur", 0.0) or 0.0)
        if state.seconds() > budget["seconds"]:  # late: never acted on
            state.ended = "time"
            return _stop("[DCLab's code] The reply came after the turn's time was up.")
        return out if isinstance(out, dict) else _stop("")


class _Tools:
    """The one registry as the agent's loop reads it: every call goes to ``registry.call`` (its schema check, its guard, its
    handler); before it, code holds the agent arm's own rules (plan first, a child's name and inputs, a report's revision,
    numbers and claims) and counts each write on the delivery row before it runs."""

    def __init__(self, registry: Any, state: _State):
        self.registry, self.state = registry, state

    def get(self, name: str) -> Any:
        return self.registry.get(name)

    def schemas(self, scope: Any = None, names: Any = None) -> list[dict[str, Any]]:
        return self.registry.schemas(scope, names)

    def call(self, name: str, arguments: Optional[dict[str, Any]] = None, context: Any = None, scope: Any = None, check: str = "schema") -> Any:
        state, turn = self.state, self.state.turn
        tool = self.registry.get(name)
        args = dict(arguments or {})
        write = tool is not None and tool.effect == "write"
        pending: Optional[dict[str, Any]] = None
        value: Any = None
        if write and state.plan is None:
            value = {"error": "Not run: write your plan first (team_plan): the steps of this turn in order. No action runs before it."}
        elif write and state.until is not None and time.monotonic() > state.until:
            value = {"error": "Not run: this turn's hold on its delivery has ended."}
        elif write and tool.move == "delegate":
            value = _delegate_args(turn, args)
        elif write and tool.move == "return_report":
            value, pending = _report_args(turn, state, args)
        if value is None and write and not _mark(state, writes=state.begun + 1):
            # the review of B3, R5: a write whose count is not on the row could be repeated after a crash: it does not run
            value = {"error": "Not run: this turn's hold on its delivery could not be recorded."}
        ran = value is None
        if ran:
            if write:
                state.begun += 1
            value = self.registry.call(name, args, context, scope, check)
        ok = not (isinstance(value, dict) and "error" in value)
        if ran and write and not ok:  # the guard or a rule refused it: nothing changed
            state.begun -= 1
            _mark(state, writes=state.begun)
        info: dict[str, Any] = {"write": write, "ran": ran}
        if ran and write and ok:
            state.writes += 1
            state.first_write = state.first_write or name
            if pending is not None:
                _keep_report(turn, state, pending, value)
        if ok and name == "team_plan":
            plan = str(args.get("plan") or "").strip()
            state.plans.append(plan)
            if state.plan is None:
                state.plan = plan
            info["plan"] = len(state.plans)
        state.called = info
        return value


def _on_step(state: _State) -> Any:
    """Every step of the loop, in the turn's journal: a call this registry ran, refused or stood in for, and one the loop
    refused before it (a tool the agent is not offered, arguments cut at the reply's limit)."""
    def seen(step: Any) -> None:
        info, state.called = state.called or {"write": False, "ran": False}, None
        row: dict[str, Any] = {"n": step.n, "tool": step.tool, "ok": bool(step.ok), "write": bool(info.get("write"))}
        if not info.get("ran", True) and step.ok:
            row["stood_in"] = True  # code answered for it (the same report again): nothing was written
        if "plan" in info:
            row["plan"] = info["plan"]
        if not step.ok:
            state.refused += 1
            row["error"] = str((step.result or {}).get("error") if isinstance(step.result, dict) else step.result)[:ERROR_CHARS]
        state.tools.append(str(step.tool))
        state.steps.append(row)
    return seen


def _mark(turn_state: _State, /, **fields: Any) -> bool:
    """The turn's state on its delivery row; whether it was written (a turn with no row: True). A row no longer pending, or
    held by another, is not changed (False)."""
    if not turn_state.delivery or not turn_state.token:
        return True
    try:
        return turn_state.turn.work.agent_turn_mark(turn_state.delivery, turn_state.token, project_id=turn_state.turn.project_id,
                                                    **fields) is not None
    except Exception as exc:  # noqa: BLE001
        print(f"teamspace: the agent turn on {turn_state.delivery} was not marked ({type(exc).__name__})", file=sys.stderr, flush=True)
        return False


# ---------------------------------------------------------------------------------------------------------- code's part of a call
def _delegate_args(turn: Any, args: dict[str, Any]) -> Optional[dict[str, Any]]:
    """A hand-over as code fixes it (changes ``args`` in place; returns a refusal or None): a child of a task, named
    ``<task>.<team>``, reading only what its parent may read (never ``help:``). A new case is its person's to open."""
    from .tools import safe_id

    if not args.get("task_id"):
        return {"error": "Refused: a new case is opened by its person (their own message), never by an agent. Hand work out as a "
                         "child of your task (team_delegate with task_id)."}
    parent = turn.work.task(turn.project_id, str(args["task_id"]))
    if parent is None:
        return None  # the registry's guard names it
    owner = str(args.get("owner_team") or "")
    args["new_task_id"] = safe_id(f"{parent.id}.{owner}")
    for key in ("thread_id", "problem_id", "brief_revision", "budget_eur"):  # code's: the child's reply thread, its problem and
        args.pop(key, None)  # brief are its parent's, and it spends nothing the parent did not give (the review of B3, F6)
    readable = [i for i in parent.allowed_inputs if not str(i).startswith("help:")]
    asked = args.get("allowed_inputs")
    args["allowed_inputs"] = [i for i in readable if i in asked] if isinstance(asked, list) else readable
    return None


def _sources(turn: Any, state: _State, view: Mapping[str, Any]) -> tuple[list[str], set[str]]:
    """The records a report's numbers may come from, as texts, and the directory actions with a terminal record (on the
    task and on every task it handed out)."""
    task = view["task"]
    row = turn.row or {}
    # the person's words and what code wrote: never a model's own words read back (a question it asked, a reason it gave,
    # an id it sent), so a number the model made up never becomes a record (the review of B3, F1)
    top, seen = task, {task.id}  # the person's own question is the request's: a handed-out task's question is an agent's words
    while top.parent_task_id and top.parent_task_id not in seen:  # (the review's G2: a Lead could write a number into it)
        parent = turn.work.task(turn.project_id, top.parent_task_id)
        if parent is None:
            break
        top = parent
        seen.add(top.id)
    texts = [str(top.question)] if not top.parent_task_id else []
    if row.get("kind") in ("answer", "reply"):  # a person's answer; a team's reply, built by code from its report
        texts.append(str(row.get("text") or ""))
    texts += [str(d.get("text") or "") for d in view["deliveries"] if d.get("kind") == "answer"]
    done: set[str] = set()
    for a in view["actions"]:
        texts.append(" ".join(str(x) for x in (a.action, a.output_ref, a.failure_ref, a.reason) if x))
        if a.status in tw.DONE_ACTIONS:
            done.add(a.action)
            if a.source.startswith("human:") and a.responded_by is not None:
                done.add(core.PERSON)  # a person's answer is on record: "Sara confirmed ..." may stand
    for r in view["child_reports"]:
        texts.append(r.answer)
        texts += list(r.limits)
        texts += [" ".join(str(x) for x in (e.source, e.ref, e.input_revision, e.uncertainty, *e.assumptions) if x) for e in r.evidence]
    for child in view["children"]:
        cview = turn.work.task_view(turn.project_id, child.id)
        for a in (cview or {}).get("actions") or ():
            texts.append(" ".join(str(x) for x in (a.action, a.output_ref, a.failure_ref) if x))
            if a.status in tw.DONE_ACTIONS:
                done.add(a.action)
                if a.source.startswith("human:") and a.responded_by is not None:
                    done.add(core.PERSON)
    return texts, done


# An identifier, not a quantity: a token with letters and digits joined by the characters ids use (T-3f9a20c7b1e4,
# L-d1-abc123, renewal_v1, fixture:E1; a part of digits only is a count, as in schema:...:8 columns), or a bare hex digest. Its digits are never numbers on either side of the check
# (the review of B3, F2: a task id's digest made "20" a known number).
_ID = re.compile(r"(?<![\w.%])(?:[A-Za-z][\w]*(?:[-_.:@/](?=\w*[A-Za-z])[\w]+)+|[A-Za-z]+\d[\w]*|(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{6,})(?![\w%])")


def _no_ids(text: str) -> str:
    return _ID.sub(lambda m: m.group(0) if not re.search(r"\d", m.group(0)) else " ", str(text or ""))


def check_sentence(sentence: str, known: set[str], spoken: set[str], done: set[str]) -> Optional[str]:
    """Why a sentence of the model's report may not stand, as a code of ``team_brain.REPORT_REFUSALS``, or None."""
    from .. import cited

    if not sentence.strip():
        return "empty"
    if not cited.numbers(_no_ids(sentence)) <= known:
        return "number_not_in_records"
    if not cited.number_words(sentence) <= spoken:
        return "number_word_not_in_records"
    if core.unrecorded_claims(sentence, sorted(done)):
        return "claims_unrecorded_action"
    if cited.overclaims(sentence):
        return "overclaims"
    if core.promises_action(sentence):
        return "promises_action"
    return None


def _report_args(turn: Any, state: _State, args: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """The report as code builds it from the model's words (changes ``args`` in place). Returns (a result that stands in
    for the call, or None to make it) and the check to keep once the report is written."""
    from .. import cited
    from .store import team_label

    task = turn.work.task(turn.project_id, str(args.get("task_id") or ""))
    view = turn.work.task_view(turn.project_id, task.id) if task is not None else None
    if task is None or view is None or task.owner_team != turn.agent:
        return None, None  # the guard refuses it
    current = [r for r in view["reports"] if r.status != "superseded"]
    latest = max(current, key=lambda r: r.revision) if current else None
    join = view.get("join")
    joined = sorted(join.reports) if join is not None else []
    terminal = {a.action_id for a in view["actions"] if tw.can_claim(a)}
    if latest is not None and latest.brief_revision >= task.brief_revision and sorted(latest.child_reports) == joined:
        listed = {line.action_id for line in (*latest.actions.completed, *latest.actions.failed, *latest.actions.skipped)}
        if terminal <= listed:  # the same report again, with nothing new behind it: the stored one
            return {"task_id": task.id, "report": latest.id, "status": latest.status, "duplicate": True}, None
    args["revision"] = 1 if latest is None else latest.revision + 1
    texts, done = _sources(turn, state, view)
    known = set().union(*(cited.numbers(_no_ids(t), source=True) for t in texts)) if texts else set()
    spoken = set().union(*(cited.words_of(t) for t in texts)) if texts else set()
    kept: list[str] = []
    refused: list[dict[str, str]] = []
    for sentence in cited.sentences(cited.CITE.sub("", str(args.get("answer") or ""))):
        why = check_sentence(sentence, known, spoken, done)
        (refused.append({"text": sentence, "reason": why, "where": "answer"}) if why else kept.append(sentence))
    limits: list[str] = []
    given = args.get("limits")
    given = [given] if isinstance(given, str) else (given if isinstance(given, list) else [])  # the review of B3, F7
    for limit in given:
        said = " ".join(str(limit or "").split())
        why = check_sentence(said, known, spoken, done) if said else "empty"
        (refused.append({"text": said, "reason": why, "where": "limit"}) if why else limits.append(said))
    model_part, used = [], 0
    for sentence in kept:
        if used + len(sentence) + 1 > MODEL_CHARS:
            break
        model_part.append(sentence)
        used += len(sentence) + 1
    # code's part: what each returned team answered and what each action recorded
    returned = [r for r in view["child_reports"] if r.id in joined and r.status != "superseded"]
    lines = [f"{team_label(r.owner)}: {' '.join(r.answer.split())[:400]}" for r in returned]
    own = [a for a in view["actions"] if a.actor.id == turn.agent]  # this team's own work on the task (a hand-over's record is a task)
    for a in own:
        if a.status in tw.DONE_ACTIONS and a.output_ref and not str(a.output_ref).startswith(_NO_FINDING):
            lines.append(f"{a.action} ({a.status}): {a.output_ref}")
        elif a.status in tw.FAILED_ACTIONS:
            lines.append(f"{a.action} ({a.status}): {a.failure_ref or 'no output'}")
    answer = " ".join(model_part) or "No sentence of the agent's findings held against the records."
    if lines:
        answer += "\n\nFrom the records:\n" + "\n".join(f"- {line}" for line in lines)
    args["answer"] = answer if len(answer) <= tw.TEXT else answer[:tw.TEXT - 1] + "…"
    args["limits"] = [*limits, "Written by the agent arm: the findings and the recommendation are a model's, checked by code; every "
                               "number in them is from the records."]
    # the evidence is the records': every returned team's, and each recorded action's output on what the task may read
    readable = [i for i in task.allowed_inputs if not str(i).startswith("help:")]
    evidence = [e.model_dump(mode="json") for r in returned for e in r.evidence]
    if readable:
        evidence += [{"source": readable[0], "ref": str(a.output_ref)[:tw.SHORT], "input_revision": a.input_revision,
                      "uncertainty": "the action's own record"} for a in own if a.status in tw.DONE_ACTIONS and a.output_ref
                     and a.input_revision and not str(a.output_ref).startswith(_NO_FINDING)]
    args["evidence"] = evidence
    if args.get("outcome") in ("partial", "blocked") and not args.get("next_owner"):
        args["next_owner"] = task.requester.team
    said = " ".join(str(args.get("unblocks") or "").split())
    if said:  # the review of B3, R3: shown to the requester as written, so checked like every sentence of the model's
        why = check_sentence(said, known, spoken, done)
        if why:
            refused.append({"text": said, "reason": why, "where": "unblocks"})
            args["unblocks"] = None
    if args.get("next_owner") and not args.get("unblocks"):
        args["unblocks"] = "what this report names as missing"
    check = {"task": task.id, "kept": [*kept, *limits], "refused": refused, "attached": len(lines)}
    return None, check


def _keep_report(turn: Any, state: _State, check: dict[str, Any], value: Any) -> None:
    """A report the agent returned, with the sentences code kept and refused: in the journal, and the ledger point."""
    report = value.get("report") if isinstance(value, Mapping) else None
    if isinstance(value, Mapping) and value.get("duplicate"):
        return
    state.reports.append({**check, "report": report})
    try:
        from ..ledger import capture

        first = check["refused"][0]["reason"] if check["refused"] else None
        capture.team_agent_report(turn.project_id, check["task"], turn.agent, kept=len(check["kept"]), refused=len(check["refused"]),
                                  refusal=first, report=report, run=state.run)
    except Exception as exc:  # noqa: BLE001 - the record never fails the turn
        print(f"teamspace: the agent's report on {check['task']} was not recorded ({type(exc).__name__})", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------- what the model reads
def brief(turn: Any, offered: tuple[str, ...], budget: Mapping[str, float]) -> str:
    """The turn as the model first reads it: who it is and its job, its action directory, the task and the person's
    question, what arrived on this turn, the tools it may call and its budget. The team's own records: never a table."""
    from . import brains
    from .agents import config_of

    lines = [brains._brief(turn, offered)]
    row = turn.row or {}
    arrived = f"What arrived: a {row.get('kind') or 'turn'} from {row.get('from_team') or 'nobody'}"
    if row.get("item_id"):
        arrived += f" (item {row['item_id']})"
    if row.get("task_id") and row.get("task_id") != turn.task_id:
        arrived += f" on {row['task_id']}"
    text = " ".join(str(row.get("text") or "").split())
    lines.append(arrived + (f": {text[:600]}" if text else "."))
    config = config_of(turn.agent)
    directory = "; ".join(f"{a.action} ({a.bound_to}, through {config.tools.get(a.action) or 'no tool in this release'})"
                          for a in config.config.action_directory)
    lines.append(f"Your action directory: {directory}.")
    lines.append("You may never: " + ", ".join(sorted(set(config.forbidden) | {"approve", "contact_customers"})) + ".")
    lines.append(f"Your budget this turn: {int(budget['steps'])} tool calls and {int(budget['seconds'])} seconds.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------- the turn
def _client(turn: Any) -> tuple[Any, Optional[str]]:
    from . import brains

    try:
        gateway = brains._gateway(turn)
        client = gateway.client(PURPOSE, project_id=turn.project_id) if gateway is not None else None
    except Exception:  # noqa: BLE001 - a store that cannot say whether the project is private: the rules' turn
        return None, "error"
    if client is None:
        return None, "no_live_models" if os.environ.get("DCLAB_NO_LIVE_MODELS") == "1" else "no_model"
    client.request_cells = False  # the team's records, never a table's cells: the exchange is kept whole with its thinking
    return client, None


def act(turn: Any, registry: Any, *, base: str, source: str, name: str, trace: Any, brain: Any) -> Any:
    """The agent arm's turn: its result (the shape of ``agents.run_turn``'s), or :class:`Fallback` for the rules. Raises
    ``delivery.TurnHeld`` when another agent turn holds the row (nothing is started)."""
    from . import delivery

    budget = budgets()
    state = _State(turn=turn, budget=budget)
    rule = _rule_first(turn, brain)
    row = turn.row or {}
    if not row:
        return _fell_back(turn, state, base, "person_entry")
    state.run = trace.run_id
    if row.get("id"):  # the lease first (the review of B3, R4): every way this turn ends is then marked on its row
        state.delivery, state.token = str(row["id"]), secrets.token_hex(8)
        claimed, held = turn.work.agent_turn_claim(state.delivery, state.token, state.run, seconds=lease_seconds(), project_id=turn.project_id)
        if claimed == "fell_back":  # the rules own this turn (they may have written): never a model on top (the review of B3, F3)
            before = held.get("agent_turn") or {}
            reason = before.get("fallback") if before.get("fallback") in core.FALLBACKS else "stopped"
            return Fallback(reason=reason, run=before.get("run"))
        if claimed == "held":
            raise delivery.TurnHeld(f"an agent turn ({(held.get('agent_turn') or {}).get('run')}) is running on {state.delivery}")
        if claimed in ("finished", "delivered", "interrupted"):
            before = held.get("agent_turn") or {}
            _record(turn, state, base, "ended" if claimed == "interrupted" else "repeat", stopped="interrupted" if claimed == "interrupted" else None,
                    run=before.get("run"))
            return _skipped(turn, base, source, claimed)
        state.until = time.monotonic() + lease_seconds()
    client, why = _client(turn)
    if client is None:
        return _fell_back(turn, state, base, why or "no_model")
    if budget["euros"] > 0:
        client.run_limit_eur = budget["euros"]  # the gateway refuses a request that could pass it, too
    from ..agents.runtime import Policy, run as run_policy
    from .. import prompts
    from .agents import config_of
    from .tools import AGENT_TOOLS

    offered = tuple(dict.fromkeys((*config_of(turn.agent).policy_tools, *AGENT_TOOLS)))
    policy = Policy(name=name, system=prompts.text(PROMPT), tools=offered, max_steps=int(budget["steps"]), max_seconds=float(budget["seconds"]),
                    max_replies=int(budget["steps"]) + 2, max_tokens=MAX_TOKENS)
    messages = [{"role": "system", "content": policy.system}, {"role": "user", "content": brief(turn, offered, budget)}]
    model = _Model(client, state)
    result, error = None, None
    try:
        result = run_policy(policy, model, _Tools(registry, state), messages, context=turn, trace=trace, on_step=_on_step(state))
    except Exception as exc:  # noqa: BLE001 - a failing model or tool: before a write the rules act, after one the turn ends
        error = exc
    except BaseException:  # a person's Stop: never the rules on top; the turn ends, recorded
        _finish(turn, state, base, source, rule, "ended", "interrupted")
        raise
    stopped = state.ended or (state.failure if error is not None else None) or (getattr(result, "stopped", None) if result else "error")
    if stopped not in core.AGENT_STOPS:
        stopped = "error" if error is not None else "answered"
    if state.begun == 0:  # no write started: the rules may act without repeating anything
        reason = None
        if error is not None:
            reason = state.failure or "error"
        elif stopped in BUDGET_STOPS:
            reason = "budget"
        elif state.plan is None or stopped in NO_ANSWER_STOPS:
            reason = "invalid_output"
        elif _still_open(turn):  # planned, then wrote nothing (every write refused, or none made): the work never waits on it
            reason = "no_action"
        if reason is not None:
            return _fell_back(turn, state, base, reason, stopped=stopped)
    if error is not None and state.failure is None:
        print(f"teamspace: {turn.agent}'s agent turn failed after an action ({type(error).__name__}: {str(error)[:200]})", file=sys.stderr,
              flush=True)
    outcome = "acted" if stopped == "answered" and error is None else "ended"
    _finish(turn, state, base, source, rule, outcome, stopped)
    final = getattr(result, "final", None) if result is not None else None
    return {"agent": turn.agent, "task_id": turn.task_id, "skipped": False, "run": state.run,
            "steps": [{"tool": s["tool"], "ok": s["ok"], "result": None} for s in state.steps], "stopped": stopped, "refused": [],
            "guard_refused": [s["error"] for s in state.steps if s.get("error")], "final": final,
            "brain": {"arm": "agent", "source": source, "acted": "agent", "asked": True},
            "agent_turn": {"outcome": outcome, "stopped": stopped, "writes": state.writes, "refused": state.refused, "plan": state.plan is not None}}


def _still_open(turn: Any) -> bool:
    """Whether the turn's task is still open: a turn that wrote nothing on it leaves it to the rules (``no_action``)."""
    try:
        task = turn.work.task(turn.project_id, turn.task_id) if turn.task_id else None
    except Exception:  # noqa: BLE001 - unknown: the rules decide, as they read the stored task themselves
        return True
    return task is not None and task.status not in tw.CLOSED_TASKS


def _rule_first(turn: Any, brain: Any) -> tuple[str, dict[str, Any]]:
    """The rules' first call on the same stored state (read only), to record beside the model's."""
    try:
        first = brain(turn)
    except Exception:  # noqa: BLE001
        first = None
    return first if first is not None else ("none", {})


def _skipped(turn: Any, base: str, source: str, why: str) -> dict[str, Any]:
    return {"agent": turn.agent, "task_id": turn.task_id, "skipped": True, "steps": [], "refused": [], "run": None, "repeat": why,
            "brain": {"arm": "agent", "source": source, "acted": "agent", "asked": False}}


def _fell_back(turn: Any, state: _State, base: str, reason: str, stopped: Optional[str] = None) -> Fallback:
    if state.delivery:  # the rules own the row's turn from now: a retry is theirs again, never a model's on top of them
        _mark(state, state="fell_back", stopped=stopped, fallback=reason)
    _record(turn, state, base, "fell_back", stopped=stopped, fallback=reason)
    _journal(turn, state, "fell_back", stopped, reason)
    return Fallback(reason=reason, run=state.run, request=state.request)


def _finish(turn: Any, state: _State, base: str, source: str, rule: tuple[str, dict[str, Any]], outcome: str, stopped: str) -> None:
    _mark(state, state="done" if outcome == "acted" else "ended", stopped=stopped)
    _journal(turn, state, outcome, stopped, None)
    _record(turn, state, base, outcome, stopped=stopped)
    from . import brains

    brains.record(turn, "agent", source, rule, {"purpose": PURPOSE, "tool": state.first_write or "none", "request": state.request,
                                                "fallback": None}, state.run, acted="agent")


# ---------------------------------------------------------------------------------------------------------- the records
def journal_path(home: Any, run: str) -> Path:
    from .tools import safe_id

    return Path(home) / "teamspace" / JOURNAL / f"{safe_id(run)}.json"


def journal(env: Any, run: Optional[str]) -> Optional[dict[str, Any]]:
    """One agent turn's journal (its plan whole, its steps, its reports' kept and refused sentences), or None."""
    from .tools import _home

    home = _home(env)
    if home is None or not run:
        return None
    path = journal_path(home, run)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _journal(turn: Any, state: _State, outcome: str, stopped: Optional[str], fallback: Optional[str]) -> None:
    """The turn whole, in the workspace (the team's own records; never a table): kept before anything is cut for a record."""
    from .tools import _home

    home = _home(turn.env)
    if home is None or not state.run:
        return
    doc = {"run": state.run, "agent": turn.agent, "task": turn.task_id, "project": turn.project_id, "delivery": state.delivery,
           "outcome": outcome, "stopped": stopped, "fallback": fallback, "plan": state.plan, "plans": state.plans, "steps": state.steps,
           "reports": state.reports, "writes": state.writes, "refused": state.refused, "euros": round(state.euros, 6),
           "seconds": round(state.seconds(), 3), "requests": state.requests, "budget": state.budget}
    try:
        path = journal_path(home, state.run)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        os.replace(temporary, path)
    except Exception as exc:  # noqa: BLE001
        print(f"teamspace: the agent turn's journal was not written ({type(exc).__name__})", file=sys.stderr, flush=True)


def _record(turn: Any, state: _State, base: str, outcome: str, *, stopped: Optional[str] = None, fallback: Optional[str] = None,
            run: Optional[str] = None) -> None:
    """``team.agent.turn`` (``dclab_core.team_brain.AgentTurnRecord``). Codes, ids and measures. Never fails the turn."""
    try:
        plan = state.plan or ""
        kept = sum(len(r["kept"]) for r in state.reports)
        refused = [x["reason"] for r in state.reports for x in r["refused"]]
        record = core.AgentTurnRecord(
            role=base, agent=str(turn.agent)[:80], task=turn.task_id, run=run or state.run, delivery=state.delivery or (turn.row or {}).get("id"),
            outcome=outcome, stopped=stopped, fallback=fallback, plan_chars=len(plan), plan_steps=_plan_steps(plan),
            plan_sha256=hashlib.sha256(plan.encode("utf-8")).hexdigest() if plan else None, steps=len(state.steps), writes=state.writes,
            refused=state.refused, tools=[t for t in state.tools if re.fullmatch(r"[a-z0-9_]{1,60}", t)][:200], sentences_kept=kept,
            sentences_refused=len(refused), refusals=refused[:50], euros=round(state.euros, 6) if state.euros else None,
            seconds=round(state.seconds(), 3), request=state.request)
        try:
            task = turn.work.task(turn.project_id, turn.task_id) if turn.task_id else None
        except Exception:  # noqa: BLE001
            task = None
        payload = {**record.model_dump(mode="json", exclude={"schema_version"}), "problem": getattr(task, "problem_id", None)}
        from .. import experience

        experience.record("team.agent.turn", "agent", {k: v for k, v in payload.items() if v is not None and v != []}, project=turn.project_id,
                          run=record.run)
    except Exception as exc:  # noqa: BLE001
        print(f"teamspace: team.agent.turn not recorded ({type(exc).__name__}: {str(exc)[:200]})", file=sys.stderr, flush=True)


def _plan_steps(plan: str) -> int:
    from .tools import plan_steps

    return min(200, plan_steps(plan))


__all__ = ["DEFAULT_EUR", "DEFAULT_SECONDS", "DEFAULT_STEPS", "ENV_EUR", "ENV_SECONDS", "ENV_STEPS", "Fallback", "PURPOSE", "act", "acts_on",
           "brief", "budgets", "check_sentence", "journal", "journal_path", "lease_seconds", "turn_seconds"]
