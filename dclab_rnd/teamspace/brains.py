"""The brain a team agent runs with, and the one audited switch that chooses it per role (B1 of the team workspace, 2026-11-23).

Every team agent acts through one strict layer (the ``team`` tools, their guard, the task store, the report builder);
only its *brain* differs (``dclab_core.team_brain``):

- ``rules``: today's agents (``agents.BRAINS`` through ``agents.ScriptedChoices``), unchanged, the default of every role;
- ``helper``: code decides and a model helps at named seats, its answer used only where a code check passes (B2:
  ``teamspace/seats.py``; the seats each role has are ``dclab_core.team_brain.ROLE_SEATS``);
- ``agent``: a model chooses the agent's next call inside the strict layer (the full loop is B3's).

**B2, the helper arm.** B1's helper placeholder (which tool next, in shadow) is replaced by the seats. The rules still
choose every call. Where they reach a seat whose answer changes what they do next (reading the person's words, the teams a
request goes to, a question's wording), the turn waits for the helper within its bounded seconds and goes on with the
helper's answer only when code accepts it; the seats that change nothing the rules do (the report explained, the risks)
are asked after the turn on the thread below. :func:`finish` records the turn's summary (``acted`` is ``helper`` when one
of its answers was used) and starts the after-seats. A helper-arm turn's delivery gets the seats' seconds on top of the
rules' limit (``delivery.timeout_seconds``); a role on the rules keeps its limit.

**The agent arm in B1 always lets the rules act, and never waits on a model.** A role switched to ``agent`` also asks its
model once per turn which tool comes next. Its request is *built* before the rules act, from the same stored state the rules
read (so the two answers are comparable), and *sent* only after the rules' run has ended, on a thread of its own outside
the delivery (:func:`finish`): a slow model never holds, fails, repeats or delays a delivery (the review of B1, F1). The
answer is checked by code (a tool the agent may call, or "none") and recorded beside the rules' first call; it is never
executed. When no model serves the purpose (none configured, ``DCLAB_NO_LIVE_MODELS=1``), the turn stopped before the
rules finished, the request is refused (a cap, the Admin page's stop), times out or fails, or its answer fails the check,
the turn falls back to the rules and the reason is recorded (``dclab_core.team_brain.FALLBACKS``). Every model request
goes through the gateway with thinking on (``models/settings.thinking_for``: summarized thinking on Anthropic), and the
exchange is kept whole with its thinking: the team purposes are shown team records, never a table's cells. The request
runs in the turn's context: the same run id, and a span of the turn's trace (``team_turn``).

**B3** (the agent arm acts): its loop runs inside the delivery and needs longer than the rules' 30 s. The deadline belongs
to the arm then: ``delivery.timeout_seconds`` reads the recipient's arm (``arm_for``) and gives an acting model arm its
own limit, the rules keeping theirs. B2 made that seam for the helper: a turn where a helper before-seat can run gets the
rules' limit plus the seats' seconds (``seats.waits_in``, ``seats.seat_seconds``).

**The switch** (:func:`switch`): one audited change per role, by a reviewer or the owner, with a reason; stored in the
workspace's routing document (``models/routing.py``, the same file or PostgreSQL row the model purposes use) under
``team_brains``, read on every turn (:func:`arm_for`). ``DCLAB_TEAM_BRAIN`` (``helper``, or ``lead=agent,sales=helper``)
overrides it for tests and evaluations; an override wins, as the environment does over ``control.json``.

**The record**: every turn that ran writes ``team.brain.answered`` (experience) and the ledger point ``team.brain``
(``ledger/capture.team_brain_answered``): the arm, where it came from, the arm that acted, the rules' first tool, the
model's, whether they agreed, and the model request whose exchange holds the thinking.
"""

from __future__ import annotations

import atexit
import os
import re
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from dclab_core import team_brain as core

Call = tuple[str, dict[str, Any]]
ENV = "DCLAB_TEAM_BRAIN"
SECTION = "team_brains"
PURPOSES = {"helper": "team_helper", "agent": "team_agent"}
QUESTION_CHARS = 1500  # the brief's words a model reads (the person's question, never a table)
MAX_TOKENS = 4000  # room for the thinking and a one-tool answer


class BrainSwitchError(ValueError):
    """A brain switch that is refused; the reason says why."""


# ---------------------------------------------------------------------------------------------------------- the switch
def _home(env: Any) -> Any:
    from .tools import _home as home_of

    return home_of(env)


def store_of(env: Any) -> Any:
    """The workspace's routing store: the gateway's (the server's), else the workspace folder's (a worker, a test)."""
    found = getattr(getattr(env, "gateway", None), "routing", None)
    if found is not None:
        return found
    home = _home(env)
    if home is None:
        return None
    from ..models.routing import open_routing

    return open_routing(home)


def settings(store: Any) -> dict[str, dict[str, Any]]:
    """Each switched role's setting (``BrainSetting`` as a dict), by role; a role not in it runs on the rules."""
    if store is None:
        return {}
    doc = store.get() or {}
    return dict(((doc.get(SECTION) or {}).get("roles")) or {})


def override() -> dict[str, str]:
    """``DCLAB_TEAM_BRAIN`` as ``{role: arm}`` (empty: none). A value that does not parse is no override: the turn keeps
    the stored setting, and the server names the mistake before it starts (``settings.Settings.problems``)."""
    try:
        return core.parse_override(os.environ.get(ENV, ""))
    except ValueError as exc:
        print(f"teamspace: {ENV} ignored ({exc})", file=sys.stderr, flush=True)
        return {}


def arm_for(env: Any, role: str) -> tuple[str, str]:
    """The arm ``role`` runs with on this turn and where it came from (``override``, ``stored``, ``default``). A routing
    document that cannot be read leaves the rules (``default``): a turn never stops on its switch."""
    base = str(role).split(":")[0]
    if base not in core.ROLES:
        return core.DEFAULT_ARM, "default"
    try:
        stored = settings(store_of(env))
    except Exception as exc:  # noqa: BLE001
        print(f"teamspace: the team brains could not be read ({type(exc).__name__}); the rules answer", file=sys.stderr, flush=True)
        stored = {}
    return core.arm_of(stored, base, override())


def switch(store: Any, audit_log: Any, role: str, arm: str, by: Optional[str], reason: str = "") -> dict[str, Any]:
    """One audited change of ``role``'s brain: a reviewer or the owner, a reason, the setting written into the routing
    document and its audit row in the same call. ``rules`` is the default (the role's setting is removed). The same arm
    again changes nothing and writes no row. Returns the role's setting now (``None`` for the rules)."""
    if role not in core.ROLES:
        raise BrainSwitchError(f"{role!r} is not a team role a brain is chosen for ({', '.join(core.ROLES)})")
    if arm not in core.ARMS:
        raise BrainSwitchError(f"a team brain is one of {', '.join(core.ARMS)}, not {arm!r}")
    if by not in core.REVIEWERS:
        raise PermissionError("Only a reviewer or the owner switches a team agent's brain")
    reason = " ".join(str(reason or "").split())[:core.SHORT]
    if not reason:
        raise BrainSwitchError("Say why: the reason is kept in the audit trail")
    if store is None:
        raise BrainSwitchError("this workspace has no routing document to keep the switch in")
    from ..models.routing import _now, atomic

    made: dict[str, Any] = {}

    def apply(doc: dict[str, Any]) -> Optional[dict[str, Any]]:  # read, changed and written as one locked step
        section = dict(doc.get(SECTION) or {})
        roles = dict(section.get("roles") or {})
        before = (roles.get(role) or {}).get("arm") or core.DEFAULT_ARM
        if before == arm:
            return None
        at = _now()
        setting = core.BrainSetting(role=role, arm=arm, previous=None if before == core.DEFAULT_ARM else before, by=by, reason=reason,
                                    since=at).model_dump(mode="json")
        if arm == core.DEFAULT_ARM:
            roles.pop(role, None)
        else:
            roles[role] = setting
        made.update(at=at, role=role, **{"from": before, "to": arm}, by=by, reason=reason)
        section.update(roles=roles, history=[*(section.get("history") or []), dict(made)])
        return {**doc, SECTION: section}

    doc = atomic(store, apply)
    if made and audit_log is not None:
        from .. import audit

        audit.record(audit_log, "routing", "human", str(by).capitalize(), move="set_team_brain", status="allowed",
                     args={"purpose": f"team_brain:{role}", "role": role, "from": made["from"], "to": arm}, message=reason)
    return (((doc.get(SECTION) or {}).get("roles")) or {}).get(role)


def overview(env: Any) -> dict[str, Any]:
    """Every switchable role's arm now and where it came from, the arms, and the last changes (for the team pages)."""
    store = store_of(env)
    try:
        doc = store.get() if store is not None else {}
    except Exception:  # noqa: BLE001
        doc = {}
    section = (doc or {}).get(SECTION) or {}
    stored, forced = dict(section.get("roles") or {}), override()
    roles = []
    for role in core.ROLES:
        arm, source = core.arm_of(stored, role, forced)
        roles.append({"role": role, "arm": arm, "source": source, "setting": stored.get(role)})
    return {"roles": roles, "arms": list(core.ARMS), "default": core.DEFAULT_ARM, "override": bool(forced),
            "history": list(section.get("history") or [])[-50:]}




# ---------------------------------------------------------------------------------------------------------- the brains
@dataclass
class Asked:
    """A model request built before the rules act (the stored state they read), sent once they have ended."""

    brain: "_ModelBrain"
    client: Any
    messages: list[dict[str, Any]]
    tools: Optional[list[dict[str, Any]]]
    offered: tuple[str, ...]


class Brain(Protocol):
    """One arm. ``prepare`` reads a turn before the rules act, on the same stored state, and returns what to ask: None for the
    rules (no model), an :class:`Asked` to send once the rules' run has ended, or what to record at once (a fallback such as
    ``no_model``). It never acts and never sends a request."""

    arm: str
    purpose: Optional[str]

    def prepare(self, turn: Any, first: Call, offered: tuple[str, ...], registry: Any) -> Any: ...


class RulesBrain:
    """Today's agents: the rules choose every call (``agents.ScriptedChoices``); no model is asked, nothing is added."""

    arm, purpose = "rules", None

    def prepare(self, turn: Any, first: Call, offered: tuple[str, ...], registry: Any) -> Any:
        return None


def _gateway(turn: Any) -> Any:
    found = getattr(turn.env, "gateway", None)
    if found is not None:
        return found
    from ..models.gateway import installed_only

    return installed_only()


def _brief(turn: Any, offered: tuple[str, ...]) -> str:
    """What a model is shown of a turn: the agent's job, the task's state, the person's question, the turn's reason and
    the tools it may call. The team's own records only: never a table, a row or a value."""
    from dclab_core import team_work as tw

    base = turn.agent.split(":")[0]
    config = tw.DIRECTORY.get(base)
    task = turn.work.task(turn.project_id, turn.task_id) if turn.task_id else None
    lines = [f"Agent: {turn.agent} ({config.identity if config else base}). Job: {config.responsibility if config else 'a team agent'}."]
    if task is not None:
        lines.append(f"Task: {task.id}, status {task.status}, brief revision {task.brief_revision}, owned by {task.owner_team}, "
                     f"asked by {task.requester.team}.")
        lines.append("Question: " + str(task.question)[:QUESTION_CHARS])
    else:
        question = str(turn.extra.get("question") or "")[:QUESTION_CHARS]
        lines.append(f"Task: {turn.task_id or 'none yet'}, not opened yet." + (f"\nQuestion: {question}" if question else ""))
    why = (turn.row or {}).get("kind") or turn.extra.get("mode") or "turn"
    lines.append(f"This turn: {why}.")
    lines.append("Tools you may call: " + ", ".join(offered) + ".")
    return "\n".join(lines)


def _code(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    return text if re.fullmatch(r"[a-z0-9_]{1,80}", text) else None


class _ModelBrain:
    """A model asked once, beside the rules, through the gateway: what it answers is checked by :meth:`read` and never acted on."""

    arm: str = ""
    purpose: Optional[str] = None
    prompt: str = ""

    def messages(self, turn: Any, first: Call, offered: tuple[str, ...]) -> list[dict[str, Any]]:
        from .. import prompts

        return [{"role": "system", "content": prompts.text(self.prompt)}, {"role": "user", "content": _brief(turn, offered)}]

    def tools(self, offered: tuple[str, ...], registry: Any) -> Optional[list[dict[str, Any]]]:
        return None

    def read(self, out: dict[str, Any], offered: tuple[str, ...]) -> Optional[str]:
        raise NotImplementedError

    def said(self, **values: Any) -> dict[str, Any]:
        return {"purpose": self.purpose, "tool": None, "request": None, "fallback": None, **values}

    def prepare(self, turn: Any, first: Call, offered: tuple[str, ...], registry: Any) -> Any:
        """The request on the state the rules are about to read (the brief, the tools offered), or the fallback when there is
        no model to ask. Nothing is sent here."""
        try:  # a store that cannot say whether the project is private never fails the rules' turn
            gateway = _gateway(turn)
            client = gateway.client(self.purpose, project_id=turn.project_id) if gateway is not None else None
        except Exception:  # noqa: BLE001
            return self.said(fallback="error")
        if client is None:
            return self.said(fallback="no_live_models" if os.environ.get("DCLAB_NO_LIVE_MODELS") == "1" else "no_model")
        try:
            return Asked(self, client, self.messages(turn, first, offered), self.tools(offered, registry), tuple(offered))
        except Exception:  # noqa: BLE001 - a brief that cannot be built: the rules' turn as it is
            return self.said(fallback="error")

    def send(self, asked: Asked) -> dict[str, Any]:
        """Send the request once (no retry), within the purpose's seconds; the checked answer, or why it fell back."""
        from ..models import settings as model_settings

        out, client = self.said(), asked.client
        client.request_cells = False  # 37.6: the team's records, never a table's cells: the exchange is kept whole
        try:
            reply = client.complete(asked.messages, asked.tools, MAX_TOKENS, single_attempt=True,
                                    timeout_seconds=float(model_settings.purpose(self.purpose).timeout))
        except TimeoutError:
            out["fallback"] = "timeout"
        except RuntimeError as exc:
            out["fallback"] = "refused" if str(exc).startswith("BudgetExceeded") else "error"
        except Exception:  # noqa: BLE001 - a failing model leaves the rules' turn as it is
            out["fallback"] = "error"
        else:
            tool = self.read(reply if isinstance(reply, dict) else {}, asked.offered)
            if tool is None:
                out["fallback"] = "invalid_output"
            else:
                out["tool"] = tool
        out["request"] = getattr(client, "last_request", None)
        return out

    def beside(self, turn: Any, first: Call, offered: tuple[str, ...], registry: Any) -> dict[str, Any]:
        """Prepare and send in one call, in line (a caller outside a turn: a measurement, a test)."""
        asked = self.prepare(turn, first, offered, registry)
        return self.send(asked) if isinstance(asked, Asked) else asked


class HelperBrain:
    """B2's helper: no request of its own per turn. The turn's seats (``seats.Seats``, attached to the turn before the rules
    run) are asked where the rules reach them; ``prepare`` hands them to :func:`finish`, which records the turn's summary
    and starts the after-seats."""

    arm, purpose = "helper", "team_helper"

    def prepare(self, turn: Any, first: Call, offered: tuple[str, ...], registry: Any) -> Any:
        from . import seats

        return seats.of(turn)


class AgentBrain(_ModelBrain):
    """B1's agent: offered the agent's own tools, it calls the one that should come next (or none). Its call is never run in
    B1; B3 runs the loop on ``agents.runtime.run`` with the same tools and guard."""

    arm, purpose, prompt = "agent", "team_agent", "team_agent"

    def tools(self, offered: tuple[str, ...], registry: Any) -> Optional[list[dict[str, Any]]]:
        return registry.schemas(names=[n for n in offered if registry.get(n) is not None])

    def read(self, out: dict[str, Any], offered: tuple[str, ...]) -> Optional[str]:
        calls = out.get("tool_calls") or []
        if not calls:
            return "none" if str(out.get("content") or "").strip() else None
        first = calls[0] if isinstance(calls[0], dict) else {}
        name = _code(first.get("name") or (first.get("function") or {}).get("name"))
        return name if name in offered else None


BRAINS: dict[str, Brain] = {"rules": RulesBrain(), "helper": HelperBrain(), "agent": AgentBrain()}


def brain(arm: str) -> Brain:
    return BRAINS.get(arm) or BRAINS[core.DEFAULT_ARM]


# ---------------------------------------------------------------------------------------------------------- after the rules
_LIVE: set[threading.Thread] = set()
_LIVE_LOCK = threading.Lock()
MAX_LIVE = 4  # shadow requests out at once in this process; a turn past it asks no model and is recorded as ``busy``


def _thread(job: Callable[[], None]) -> None:
    def go() -> None:
        try:
            job()
        finally:
            with _LIVE_LOCK:
                _LIVE.discard(threading.current_thread())

    thread = threading.Thread(target=go, name="dclab-team-brain", daemon=True)
    with _LIVE_LOCK:  # the cap is checked and the slot taken at once: two turns never both pass at MAX_LIVE - 1
        if len(_LIVE) >= MAX_LIVE:
            raise Busy()
        _LIVE.add(thread)
    try:
        thread.start()
    except BaseException:
        with _LIVE_LOCK:  # a thread that never started never holds a slot
            _LIVE.discard(thread)
        raise


class Busy(RuntimeError):
    """Too many shadow requests are out in this process: the turn asks no model."""


_RUNNER: dict[str, Callable[[Callable[[], None]], None]] = {"run": _thread}


def set_runner(fn: Optional[Callable[[Callable[[], None]], None]]) -> Callable[[Callable[[], None]], None]:
    """How a model arm's request is run once the rules have ended (default: its own daemon thread). Returns the one replaced."""
    previous, _RUNNER["run"] = _RUNNER["run"], fn or _thread
    return previous


def drain(seconds: float = 30.0) -> bool:
    """Wait up to ``seconds`` for the requests still out (a test, a command-line run before it exits); True when none is."""
    import time

    deadline = time.monotonic() + seconds
    while True:
        with _LIVE_LOCK:
            live = list(_LIVE)
        if not live:
            return True
        left = deadline - time.monotonic()
        if left <= 0:
            return False
        live[0].join(left)


# A process that ends (a command-line run, a worker restart) waits for the requests still out, so a sent request's exchange,
# its thinking and the turn's record are never cut off; at most the agent purpose's seconds and a margin.
atexit.register(drain, 130.0)


def finish(turn: Any, arm: str, source: str, first: Call, prepared: Any, run: Optional[str], completed: bool, steps: Any = ()) -> None:
    """After the rules' run, whether it completed or raised (a Stop, a failing tool): record which brain answered. A prepared
    request is sent only when the rules' run completed, on a thread of its own in this turn's context (its run id and its
    span), so the delivery never waits on it; a turn that did not complete asks no model (``stopped``). B2: a helper turn
    records its before-seats' summary, and a completed one that returned a report starts its after-seats on that thread
    (``busy`` when the cap is reached, ``error`` when no thread starts). Never raises."""
    try:
        from . import seats

        if isinstance(prepared, seats.Seats):
            _finish_helper(turn, arm, source, first, prepared, run, completed, steps)
            return
        if not isinstance(prepared, Asked):
            record(turn, arm, source, first, prepared, run)
            return
        if not completed:
            record(turn, arm, source, first, prepared.brain.said(fallback="stopped"), run)
            return
        import contextvars

        from .. import experience

        def job() -> None:
            with experience.bind(run=run):
                said = prepared.brain.send(prepared)
                record(turn, arm, source, first, said, run)
        context = contextvars.copy_context()
        try:
            _RUNNER["run"](lambda: context.run(job))
        except Busy:  # a burst of deliveries on a switched role never starts a thread each
            record(turn, arm, source, first, prepared.brain.said(fallback="busy"), run)
        except Exception:  # noqa: BLE001 - no thread to send it: the turn is still recorded
            record(turn, arm, source, first, prepared.brain.said(fallback="error"), run)
    except Exception as exc:  # noqa: BLE001 - the record of a turn never fails the turn
        print(f"teamspace: the brain of {turn.agent}'s turn was not recorded ({type(exc).__name__})", file=sys.stderr, flush=True)


def _finish_helper(turn: Any, arm: str, source: str, first: Call, held: Any, run: Optional[str], completed: bool, steps: Any) -> None:
    from . import seats

    model = held.summary()
    if model is None and not completed:
        model = {"purpose": "team_helper", "tool": None, "request": None, "fallback": "stopped"}
    record(turn, arm, source, first, model, run, acted="helper" if held.acted() else core.DEFAULT_ARM)
    if not completed:
        return
    import contextvars

    from .. import experience

    role = str(turn.agent).split(":")[0]
    for _task, report, seat, job in seats.after(turn, seats.returned(steps), run):
        def go(job: Callable[[], None] = job) -> None:
            with experience.bind(run=run):
                job()
        context = contextvars.copy_context()
        try:
            _RUNNER["run"](lambda go=go, context=context: context.run(go))
        except Busy:  # a burst of reports on a switched role never starts a thread each: the report stands, unexplained
            seats.record_seat(turn, role, seat, "busy", rule="none", report=report, run=run)
        except Exception:  # noqa: BLE001 - no thread to ask it on: the report stands, and the seat is recorded
            seats.record_seat(turn, role, seat, "error", rule="none", report=report, run=run)


# ---------------------------------------------------------------------------------------------------------- the record
def record(turn: Any, arm: str, source: str, first: Call, model: Optional[dict[str, Any]], run: Optional[str],
           acted: str = core.DEFAULT_ARM) -> None:
    """Which brain answered the turn, in the experience record and the ledger. Best effort: a store that cannot be read or
    written is reported and the turn stands. Codes and ids."""
    try:
        try:
            task = turn.work.task(turn.project_id, turn.task_id) if turn.task_id else None
        except Exception as exc:  # noqa: BLE001 - the problem id is a convenience; the record is written without it
            print(f"teamspace: {turn.task_id} could not be read for its brain record ({type(exc).__name__})", file=sys.stderr, flush=True)
            task = None
        mode = _code((turn.row or {}).get("kind") or turn.extra.get("mode") or "turn")
        payload = {"task": turn.task_id, "problem": getattr(task, "problem_id", None) or turn.extra.get("problem_id"), "agent": turn.agent,
                   "arm": arm, "source": source, "acted": acted, "rule_tool": first[0], "mode": mode}
        if model is not None:  # B2: a helper turn names no tool (its seats are their own records): no agreement to state
            payload.update(fallback=model.get("fallback"), model_tool=model.get("tool"), request=model.get("request"),
                           agreed=None if model.get("fallback") or model.get("tool") is None else model.get("tool") == first[0])
        try:
            from .. import experience

            experience.record("team.brain.answered", "agent", {k: v for k, v in payload.items() if v is not None}, project=turn.project_id,
                              run=run)
        except Exception as exc:  # noqa: BLE001
            print(f"teamspace: team.brain.answered not recorded ({type(exc).__name__})", file=sys.stderr, flush=True)
        from ..ledger import capture

        capture.team_brain_answered(turn.project_id, str(turn.task_id or "none"), turn.agent, arm=arm, source=source, acted=acted,
                                    rule_tool=first[0], model=model, run=run)
    except Exception as exc:  # noqa: BLE001
        print(f"teamspace: the brain record of {getattr(turn, 'agent', '?')} failed ({type(exc).__name__})", file=sys.stderr, flush=True)


__all__ = ["BRAINS", "ENV", "PURPOSES", "AgentBrain", "Asked", "Brain", "BrainSwitchError", "HelperBrain", "RulesBrain", "arm_for", "brain",
           "drain", "finish", "overview", "record", "set_runner", "settings", "store_of", "switch"]
