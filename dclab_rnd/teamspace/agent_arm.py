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
NO_HANDED_WORK = frozenset({"business", "lead", "home"})  # the requester's side: they ask, they never own a handed-out task


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
    passed: list[dict[str, Any]] = field(default_factory=list)  # B4.1: code's steps that recorded and passed a person's answer
    checked: list[dict[str, Any]] = field(default_factory=list)  # B4.1: questions and hand-overs refused for their words

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
            value = _delegate_args(turn, args) or _check_words(turn, state, args, "delegate")
        elif write and tool.move == "clarify":
            value = _check_words(turn, state, args, "clarify")
        elif write and tool.move == "block":
            value = _block_refusal(turn, args)
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
    args["owner_team"] = team_id(args.get("owner_team"))  # B4.1: "Data Science", "ds" are data_science
    owner = str(args.get("owner_team") or "")
    if owner.split(":")[0] in NO_HANDED_WORK:  # the live smoke run of 2026-11-24 (QX-03): an agent Lead handed a ranking to
        # Business, whose rules never take handed-out work, and the case hung with the task open for ever
        return {"error": f"Refused: {owner} does not take handed-out work. Hand analysis to data_science or ml_engineering and "
                         "outreach limits to sales or marketing; ask the person through a question or your report."}
    args["new_task_id"] = safe_id(f"{parent.id}.{owner}")
    for key in ("thread_id", "problem_id", "brief_revision", "budget_eur"):  # code's: the child's reply thread, its problem and
        args.pop(key, None)  # brief are its parent's, and it spends nothing the parent did not give (the review of B3, F6)
    readable = [i for i in parent.allowed_inputs if not str(i).startswith("help:")]
    asked = args.get("allowed_inputs")
    chosen = [i for i in readable if i in asked] if isinstance(asked, list) else []
    # B4.1 (D-13: tasks handed out with inputs=[] could read nothing): a hand-over that names no input its parent may read
    # takes the parent's readable inputs, never wider
    args["allowed_inputs"] = chosen or readable
    return None


# B4.1 (the first dev run: 19 hand-overs refused for a team written "data science", "Data Science", "ds", "data-science"):
# the team ids an agent hands work to, and the words a model writes for them
TEAM_IDS = ("data_science", "ml_engineering", "sales", "marketing", "business", "lead", "model:eta", "model:acceptance")
_TEAM_ALIASES = {"ds": "data_science", "datascience": "data_science", "data_scientist": "data_science", "data_scientists": "data_science",
                 "ml": "ml_engineering", "mle": "ml_engineering", "ml_eng": "ml_engineering", "ml_engineer": "ml_engineering",
                 "ml_engineers": "ml_engineering", "machine_learning": "ml_engineering", "machine_learning_engineering": "ml_engineering",
                 "eta": "model:eta", "model_eta": "model:eta", "eta_model": "model:eta", "acceptance": "model:acceptance",
                 "model_acceptance": "model:acceptance", "acceptance_model": "model:acceptance"}


def team_id(name: Any) -> str:
    """A team's id as the directory names it, from the words a model wrote: case, spaces and dashes folded ("Data Science",
    "data-science" are data_science) and the usual short names ("ds", "ml", "eta"). A name it cannot place is returned as
    written, folded, so the guard refuses it and says why."""
    raw = str(name or "").strip()
    if raw.lower().startswith("model:"):
        return "model:" + re.sub(r"[\s-]+", "_", raw.split(":", 1)[1].strip().lower())
    folded = re.sub(r"[\s-]+", "_", raw.lower()).strip("_")
    folded = re.sub(r"_team$", "", folded)
    return _TEAM_ALIASES.get(folded, folded)


def _block_refusal(turn: Any, args: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """B4.1 (D-02, D-14, D-18 of the first dev run: a question was asked, then the task blocked before the answer came): no
    block while a question to the person on that task is still open."""
    task = turn.work.task(turn.project_id, str(args.get("task_id") or ""))
    if task is not None and task.status == "waiting_for_person":
        return {"error": "Not run: a question to the person on this task is still open. Wait for their answer: end this turn "
                         "(answer Done.). Their answer comes back to you and the work goes on from it; block only on what no answer can give."}
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


# B4.1: what reads like a number in a question and is none, the same rules the comparison's scorer reads with (its own copy):
# a counting word of the conversation ("two things", "the two"), a minus glued to a letter (T-7) and a binary label's code
# (y = 0); and what a person's words make known: a span of days ("a week" is 7)
# only nouns of the conversation itself (the review of B4.1, F4: "five events per customer", "three fraud cases" and "two
# outcomes were positive" count the data, so events, cases, outcomes, items and cohorts are never here)
_TALK = (r"things|questions|options|choices|points|ways|steps|readings|meanings|definitions|interpretations|answers|teams|"
         r"alternatives|possibilities|targets\s+(?:to|you|we|for\s+(?:the|this))|decisions\s+(?:to|for\s+you)")
# ... or one that counts the options it then lists: "two different targets — (a) ... and (b) ..."
_COUNTING = re.compile(rf"\b(?:two|three|four|five)(?=\s+(?:[a-z-]+\s+)?(?:{_TALK})\b|[^.;?\d]{{0,40}}?[:—–]\s*\(?a\))", re.I)
_COUNTING_PRONOUN = re.compile(r"\b(the|these|those|both|all)\s+(?:two|three|four|five)\b(?=\s*(?:[.,;:!?)]|$|of\b|give\b|are\b|were\b|"
                               r"have\b|differ|mean\b|would\b|could\b|can\b|will\b|is\b|lead\b|change\b|need\b|point\b))", re.I)
_GLUED_MINUS = re.compile(r"(?<=[A-Za-z])-(?=\d)")
_LABEL_CODE = re.compile(r"\b(?:y|label|target|class|outcome)\s*=\s*[01](?![\d.,]\d?)\b", re.I)
_SPAN = re.compile(r"\b(a|an|one|two|three|four|\d+)\s+(day|week|fortnight)s?\b", re.I)
_SPAN_DAYS = {"day": 1, "week": 7, "fortnight": 14}
_SMALL = {w: n for n, w in enumerate("zero one two three four five six seven eight nine ten".split())} | {"a": 1, "an": 1}
_OPTION = re.compile(r"(?:(?<=^)|(?<=[\s:;,(]))\(?(?:[a-e]|i{1,3}|[1-5])\)(?=\s)")
_BOUNDARY = re.compile(r"[;:—–?(]|,\s|\s-\s")
_SEGMENT_END = re.compile(r"[;?]|\.(?:\s|$)")
_VALUE = r"(?:T-)?\d+(?:\.\d+)?%?"  # a value as written in a list ("8,400" is one number: a list's comma is followed by a space)
# a choice holds a real number, read as the checks read numbers (the re-review of B4.1, R1: "or wait a week", "or the other
# one" offer no value, so the number on the other side stays a fact)
def _cited_numbers(text: str) -> set[str]:
    from .. import cited

    return cited.numbers(text)


def _has_number(text: str) -> bool:
    return bool(_cited_numbers(_no_ids(_read(text)))) or bool(_SPAN.search(str(text or "")))  # the fourth review, 5: "a week" is a value too (7 days)


# a word that asks the person to pick, before the first option marker (the fourth review, 3: a pick word after the options,
# "(a) 812 rows lack dates and (b) 40 duplicates; which should we fix first?", lists facts)
_PICK = re.compile(r"\b(?:which|pick|choose|choice|prefer|either|should\s+(?:we|i|it|the)|do\s+you\s+want|would\s+you)\b", re.I)
def _same_value(a: str, b: str) -> bool:
    return _cited_numbers(_no_ids(_read(a))) == _cited_numbers(_no_ids(_read(b)))


# a clause that gives a reason states what holds ("Since 812 accounts lapsed or 40 renewed late, ...", "Churn rose because 812
# lapsed or 40 renewed late"): never choices. "since", "because" and "given" anywhere before the "or" in its clause; "as",
# "after" and "when" only at the clause's start ("score 7 days after signup or 14?" is a choice)
_GIVEN = re.compile(r"\b(?:since|because|given)\b", re.I)
_GIVEN_START = re.compile(r"^\W*(?:as|after|when)\b", re.I)
# where a number may stand in a sentence: a range read as one value ("10-20", "10%-20%", "10 to 20"), digits (with a sign glued
# to a letter, a unit, a percent), number words, spans of days
_DIGITS = r"(?:\b[A-Za-z]-)?\d+(?:[.,]\d+)*%?"
_NUMBER_AT = re.compile(rf"{_DIGITS}(?:\s?[-–]\s?|\s+to\s+)\d+(?:[.,]\d+)*%?|{_DIGITS}|"
                        r"\b(?:a|an|one|two|three|four|\d+)\s+(?:day|week|fortnight)s?\b|"
                        r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                        r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[-\s](?:one|two|three|four|"
                        r"five|six|seven|eight|nine))?(?:\s?(?:%|percent))?\b", re.I)


def _values(text: str, start: int = 0, end: Optional[int] = None) -> list[tuple[int, int]]:
    """The spans of ``text[start:end]`` that hold a number, as the checks read one ("the other one" holds none)."""
    end = len(text) if end is None else end
    return [(m.start(), m.end()) for m in _NUMBER_AT.finditer(text, start, end) if _has_number(m.group(0))]


_TIME_UNIT = r"(?:days?|weeks?|months?|hours?|minutes?|years?)"
_TIME_AFTER = re.compile(rf"\s*{_TIME_UNIT}\b", re.I)


def _kind(text: str, span: tuple[int, int]) -> str:
    """What a value counts: ``time`` (a span of days, T-7, 14 days), ``percent``, ``bare`` (nothing after it: "7 or 14 days"),
    or ``count`` (a number of something: "312 late deliveries")."""
    value, after = text[span[0]:span[1]], text[span[1]:]
    if _SPAN.fullmatch(value.strip()) or re.match(r"[A-Za-z]-", value) or _TIME_AFTER.match(after):
        return "time"
    if value.endswith("%") or re.match(r"\s*(?:%|percent)", after, re.I):
        return "percent"
    return "bare" if re.match(r"\s*(?:or\b|[,;:?!.)—–]|$)", after, re.I) else "count"


def _alike(text: str, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Two values offered as choices of one thing: never a count of the data ("the 812 accounts or the 40"), and of one kind
    or one of them bare ("at 7 or 14 days")."""
    ka, kb = _kind(text, a), _kind(text, b)
    return "count" not in (ka, kb) and (ka == kb or "bare" in (ka, kb))


# the value right before an "or": the value, its time unit, and for a time a short "before ..." ("seven days before renewal,
# or 14 days"); right after an "or": a preposition at most, then the value ("or at T-14", "or 14 days"). Anything else
# between a value and the "or" ("the 812 lapsed accounts, or wait 14") leaves the number a fact.
_LEFT_TAIL = re.compile(rf"(?:\s+{_TIME_UNIT})?(?:\s+(?:before|after|prior\s+to)(?:\s+[\w-]+){{0,2}})?\s*,?\s*$", re.I)
_RIGHT_HEAD = re.compile(r"\s*(?:(?:at|in|after|within|by|to)\s+)?", re.I)
# only a list offered as examples ("(e.g. 30, 60, 90)", "such as 7 or 14"): "we found 12, 40 and 300" states facts
_NUMBER_LIST = re.compile(rf"(?:\b(?:e\.g\.,?|for example,?|such as)\s+)"
                          rf"({_VALUE}(?:\s*(?:,\s+|\s+or\s+){_VALUE}){{1,}})(?![\w.,]\d)", re.I)


def _or_pair(text: str, at: re.Match) -> Optional[list[tuple[int, int]]]:
    """The two values an "or" offers, or None: each must stand right next to the "or", both must be of one kind (never a
    count), differ, sit in a clause that gives no reason, and the "or" must be in the asking clause, not a statement
    before it ("About 812 or 900 rows lack dates, drop them?")."""
    left = re.sub(r"[,\s]+$", "", text[:at.start()])
    cut = max((b.end() for b in _BOUNDARY.finditer(left)), default=0)
    clause = left[cut:]
    if _GIVEN.search(clause) or _GIVEN_START.match(clause):
        return None
    a = next((s for s in reversed(_values(text, cut, len(left)))
              if _LEFT_TAIL.fullmatch(text, s[1], at.start()) and (s[1] == len(left) or _kind(text, s) == "time")), None)
    head = _RIGHT_HEAD.match(text, at.end())
    m = _NUMBER_AT.match(text, head.end())
    b = (m.start(), m.end()) if m and _has_number(m.group(0)) else None
    if a is not None and _STATES.search(text[:a[0]]):
        return None  # "should we say churn is 12% or 15%?" states a value
    if a is None or b is None or not _alike(text, a, b) or _same_value(text[a[0]:a[1]], text[b[0]:b[1]]):
        return None
    rest = text[b[1]:]
    stop = _CLAUSE_END.search(rest)
    if stop and "?" in rest[stop.end():] and not _REASON_AFTER.match(rest[stop.end():]):
        return None  # a statement clause, a question after it ("at T-7 or T-14, given 37% churn?" still asks)
    return [a, b]


_CLAUSE_END = re.compile(r"[;:]|,\s")
_REASON_AFTER = re.compile(r"\W*(?:given|since|because|as|with|where)\b", re.I)


# B4.1, the sixth review: a question that checks a fact ("Is the churn rate 12% or 15%?", "Did 37% or 40% lapse?", "Which is
# right: (a) 812, or (b) 900?", "Our holdout showed, for example, 0.81 or 0.84?") offers no choice: its numbers are claims.
# Only a question that asks the person to pick a design value (a window, a moment, what to use) offers its values.
_FACT_START = re.compile(r"^\W*(?:is|are|was|were|did|does|has|have|had|do(?!\s+you\s+(?:want|prefer|need|mean)\b))\b", re.I)
_FACT_WORDS = re.compile(r"\b(?:was|were|did|had|right|correct|true|accurate|trust|actual(?:ly)?|really|show(?:s|ed|n)?|came\s+out|"
                         r"rose|fell|dropped|grew|reached|measured|observed|on\s+average|equals?)\b", re.I)
_PICK_ANY = re.compile(r"\b(?:which|pick|choose|choice|prefer|either|should\s+(?:we|i|it|the)|do\s+you\s+want|would\s+you|"
                       r"how\s+many|how\s+long|use|set|score|keep|flag)\b", re.I)
_STRONG = re.compile(r"[;:?—–]")
_STATES = re.compile(r"\b(?:is|are|=)\s*$", re.I)


_MEASURE = (r"(?:churn\w*|laps\w*|renew\w*|left|downgrad\w*|auc|lift|rate|revenue|accuracy|precision|recall|mae|rmse|"
            r"median|mean|average|share|\w+ed)")
_MEASURED_AFTER = re.compile(rf"(?:\s+(?:of\s+)?[\w-]+){{0,2}}?\s+{_MEASURE}\b", re.I)
_MEASURED_BEFORE = re.compile(rf"\b{_MEASURE}\b(?:\W+\w+){{0,3}}?\W*$", re.I)


def _measured(text: str, span: tuple[int, int]) -> bool:
    """A value that names a measured result ("37% churn", "37% of accounts lapsed", "churn is 37%", "read churn, for example
    37%"): a fact, never a choice. A time is a design value ("7 days before renewal"); a measured time is caught as a
    question that checks a fact ("Is the median 14 days or 21?")."""
    if _kind(text, span) == "time":
        return False
    before = text[max(0, span[0] - 40):span[0]]
    if _STATES.search(before) or _MEASURED_BEFORE.search(re.split(r"[;:?(]|\b[a-e]\)", before)[-1]):
        return True
    return bool(_MEASURED_AFTER.match(re.split(r"[,;:?)(]|\bor\b", text[span[1]:])[0]))


def split_choices(sentence: str) -> tuple[str, str]:
    """A question's sentence as (its premise, the values it offers the person to pick). When unsure, a number is premise: a
    false choice would let an invented number pass, a false premise only sends the question back to be written again
    (the reviews of B4.1: F3, R1 to R3, the fourth and the fifth review). Offered are only:

    - option markers ("(a)", "b)"): two or more, a word asking for a pick before the first one (or an "or" before the last
      one, in a question), and at least two parts offering a value: the first value of each part that is not a count of the
      data ("(a) 812 accounts at 7 days" offers 7); everything else is premise;
    - an "or" in a question with a value right next to it on each side, of one kind and never a count ("at T-7 or T-14",
      "seven days before renewal, or 14 days", "10-20 or 30-40 days"), outside a reason ("since", "because", "given") and
      in the asking clause;
    - a list of example values in a question ("how many days (e.g. 30, 60, 90)?").
    """
    text = str(sentence or "")
    asks = text.rstrip().endswith("?")
    offered: list[tuple[int, int]] = []
    markers = list(_OPTION.finditer(text))
    picks = len(markers) >= 2 and bool(_PICK.search(text[:markers[0].start()])
                                       or (asks and re.search(r"\bor\s*$", text[:markers[-1].start()], re.I)))
    asks = asks or picks  # "pick the moment: (a) ...; or (b) at T-30 or T-14." asks too (D-05)
    if _FACT_START.match(text) or not (picks or _PICK_ANY.search(text)):
        return text, ""  # it checks a fact, or asks for no pick: every number is premise

    def checks_fact(group: list[tuple[int, int]], lead: Optional[int] = None) -> bool:
        """A fact word ("showed", "right", "did") in the clause that holds the values (from the last ";", ":", "?" or dash
        before them), or before marked options: their values are claims ("Which is right: (a) 812, or (b) 900?")."""
        if not group:
            return False
        first, last = min(a for a, _ in group), max(b for _, b in group)
        start = lead if lead is not None else max((m.end() for m in _STRONG.finditer(text, 0, first)), default=0)
        return bool(_FACT_WORDS.search(text, start, last)) or any(_measured(text, g) for g in group)
    if picks:
        stop = _SEGMENT_END.search(text, markers[-1].end())
        end = stop.start() if stop else len(text)
        firsts = []
        for i, m in enumerate(markers):
            part = _values(text, m.end(), markers[i + 1].start() if i + 1 < len(markers) else end)
            first = next((s for s in part if _kind(text, s) != "count"), None)
            if first:
                firsts.append(first)
        if len(firsts) >= 2 and not checks_fact(firsts, lead=0):
            offered += firsts  # one measured option makes them all facts ("(a) 37% churn or (b) 42% churn")
    if asks:
        for m in _NUMBER_LIST.finditer(text):
            listed = _values(text, m.start(1), m.end(1))
            if not checks_fact(listed) and not _measured(text, (m.start(), m.start())):
                offered += listed  # "read churn, for example 37% or 12%": the list names a measured result
        for m in re.finditer(r"\bor\b", text, re.I):
            pair = _or_pair(text, m) or []
            if not checks_fact(pair):
                offered += pair
    if not offered:
        return text, ""
    premise, choices, at = [], [], 0
    for start, end in sorted(set(offered)):
        if start < at:
            continue
        premise.append(text[at:start])
        choices.append(text[start:end])
        at = end
    premise.append(text[at:])
    return " ".join(premise), " ".join(choices)


WORD_CHECKS = ("clarify", "delegate")


def _read(text: str) -> str:
    return _LABEL_CODE.sub(" ", _GLUED_MINUS.sub(" ", _COUNTING_PRONOUN.sub(r"\1 ", _COUNTING.sub(" ", str(text or "")))))


def _spans(text: str) -> set[str]:
    out: set[str] = set()
    for count, unit in _SPAN.findall(str(text or "")):
        n = int(count) if count.isdigit() else _SMALL.get(count.lower(), 1)
        value = str(n * _SPAN_DAYS[unit.lower()])
        out |= {value, "+" + value}
    return out


def check_words(sentence: str, known: set[str], spoken: set[str], done: set[str], *, choices: bool = False) -> Optional[str]:
    """Why a sentence of a question or a hand-over may not be sent, as a code of ``team_brain.REPORT_REFUSALS``, or None: a
    number (or a number word) no record holds, or an action claimed without its terminal record. With ``choices`` (a question
    to the person), a number in a choice it offers ("7 days, or T-14?") is allowed; one in its premise is not."""
    from .. import cited

    premise = split_choices(sentence)[0] if choices else sentence
    if not cited.numbers(_no_ids(_read(premise))) <= known:
        return "number_not_in_records"
    if not cited.number_words(_read(premise)) <= spoken:
        return "number_word_not_in_records"
    if core.unrecorded_claims(sentence, sorted(done)):
        return "claims_unrecorded_action"
    return None


def _check_words(turn: Any, state: _State, args: dict[str, Any], kind: str) -> Optional[dict[str, Any]]:
    """B4.1: a question (``team_clarify``) or a hand-over's question (``team_delegate``) checked like a report's sentences,
    before it is sent: its numbers from the records (the person's words, code's records), no action claimed without its
    terminal record. Refused whole, with the sentence and why, so the model writes it again."""
    from .. import cited

    task = turn.work.task(turn.project_id, str(args.get("task_id") or ""))
    view = turn.work.task_view(turn.project_id, task.id) if task is not None else None
    if task is None or view is None:
        return None  # the guard names it
    texts, done = _sources(turn, state, view)
    known = set().union(*(cited.numbers(_no_ids(t), source=True) | _spans(t) for t in texts)) if texts else set()
    spoken = set().union(*(cited.words_of(t) for t in texts)) if texts else set()
    for sentence in cited.sentences(str(args.get("question") or "")):
        why = check_words(sentence, known, spoken, done, choices=kind == "clarify")
        if why:
            state.checked.append({"tool": f"team_{kind}", "task": task.id, "text": sentence, "reason": why})
            said = {"number_not_in_records": "it states a number no record holds (only the person's words and code's records hold numbers; a "
                                             "choice you offer the person may carry one)",
                    "number_word_not_in_records": "it states an amount in words no record holds",
                    "claims_unrecorded_action": "it says an action was done that no record says ended"}[why]
            return {"error": f"Not sent: {said}: \"{sentence[:300]}\". Write it again without that, from what the records hold."}
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
def brief(turn: Any, offered: tuple[str, ...], budget: Mapping[str, float], passed: Optional[list[dict[str, Any]]] = None) -> str:
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
    return "\n".join(lines + passed_lines(passed))


def passed_lines(passed: Optional[list[dict[str, Any]]]) -> list[str]:
    """What the brief says about code's passing of the person's answer: only the steps that ran (the review of B4.1, F7), and
    a refused one by name."""
    done = [x["tool"] for x in passed or () if x.get("ok")]
    failed = [x["tool"] for x in passed or () if not x.get("ok")]
    lines = []
    if done:
        lines.append("DCLab's code has already recorded the person's answer on this turn and passed it to the teams that asked "
                     f"({', '.join(dict.fromkeys(done))}): never record or carry it again. Go on with your task from it.")
    if failed:
        lines.append(f"DCLab's code could not finish passing the person's answer on ({', '.join(dict.fromkeys(failed))} was refused): "
                     "read the task (team_read_task) to see where the answer stands.")
    return lines


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
    if rule[0] in ANSWER_CALLS:  # B4.1: a person's answer arrived: code records it and passes it on, as the rules do
        passed = _pass_answer(turn, registry, brain, state, name)
        if passed is not None and not _work_left(turn):
            # nothing is left for this agent until a team replies or the person answers again: the turn was code's (the
            # person's own message, as person_entry), the model is not asked, and the rules never run on top
            fell = _fell_back(turn, state, base, "person_entry")
            brains_record(turn, source, rule, fell.model, passed.get("run"))
            return {**passed, "brain": {"arm": "agent", "source": source, "acted": "rules", "asked": False}}
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
    messages = [{"role": "system", "content": policy.system}, {"role": "user", "content": brief(turn, offered, budget, state.passed)}]
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


# B4.1 (D-05, D-09, D-10, D-16 of the first dev run: 33 calls to record or carry the person's answer were refused, and turns
# ran out of time right after the answer): a person's answer is code's to record and pass on, never the model's to retype
ANSWER_CALLS = ("team_record_human_response", "team_answer")


def _answer_brain(brain: Any) -> Any:
    """The rules' calls that record a person's answer and pass it on (to each team that asked), and the one accept that takes
    the task up again after it; then nothing, so the model goes on from there (G1: only code records a person's answer, from
    that person's own words: the guard checks every one of these calls as it checks the rules')."""
    took_up: list[str] = []

    def answer_only(turn: Any) -> Any:
        call = brain(turn)
        if call is None:
            return None
        name, args = call
        if name in ANSWER_CALLS:
            return call
        if name == "team_accept" and not took_up:
            task = turn.work.task(turn.project_id, str(args.get("task_id") or ""))
            if task is not None and task.status == "waiting_for_person":
                took_up.append(task.id)
                return call
        return None

    return answer_only


def _pass_answer(turn: Any, registry: Any, brain: Any, state: _State, name: str) -> Optional[dict[str, Any]]:
    """Run the rules' answer calls on a run of their own (traced as the rules', never the model's), before the model's turn.
    Returns the run's result (the shape of ``agents.run_turn``'s), or None when nothing ran."""
    from ..agents.harness import tracer
    from ..agents.runtime import Policy, run as run_policy
    from .agents import MAX_STEPS, SYSTEM, ScriptedChoices, _rules_policy, config_of

    trace = tracer(name)
    policy = _rules_policy(Policy)(name=name, system=SYSTEM, tools=config_of(turn.agent).policy_tools, max_steps=MAX_STEPS, max_seconds=60.0)
    result = run_policy(policy, ScriptedChoices(_answer_brain(brain), turn, None), registry,
                        [{"role": "user", "content": f"answer on {turn.task_id}"}], context=turn, trace=trace)
    steps = [{"tool": x.tool, "ok": bool(x.ok)} for x in result.steps if x.tool]
    state.passed = [{**x, "by": "code", "run": trace.run_id} for x in steps]
    if not any(x["ok"] for x in steps):
        return None
    return {"agent": turn.agent, "task_id": turn.task_id, "skipped": False, "run": trace.run_id,
            "steps": [{"tool": x.tool, "ok": x.ok, "result": x.result} for x in result.steps], "stopped": result.stopped,
            "refused": [], "guard_refused": [x.result.get("error") for x in result.steps if not x.ok and isinstance(x.result, dict)],
            "final": result.final}


def _work_left(turn: Any) -> bool:
    """Whether the agent has work on its task now that the answer is passed on: not while the task waits on the person
    again or on teams that have not replied (their reply is its next turn); yes when it was taken up with nothing out."""
    try:
        view = turn.work.task_view(turn.project_id, turn.task_id) if turn.task_id else None
    except Exception:  # noqa: BLE001 - unknown: the model decides
        return True
    if view is None:
        return False
    task, join = view["task"], view.get("join")
    if task.status in tw.CLOSED_TASKS or task.status == "waiting_for_person":
        return False
    return not (join is not None and join.outstanding)


def brains_record(turn: Any, source: str, rule: tuple[str, dict[str, Any]], model: dict[str, Any], run: Optional[str]) -> None:
    from . import brains

    brains.record(turn, "agent", source, rule, model, run, acted="rules")


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
           "passed_answer": state.passed, "checked_words": state.checked,
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
