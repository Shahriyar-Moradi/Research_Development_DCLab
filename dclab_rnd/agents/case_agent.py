"""The lead agent (packages 37.3 and 37.4, docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): one agent holds a case from the
person's first sentence to the route, around the case file (``dclab_core.case``), on the shared runtime (``agents.run``).

Claude thinks first, with the whole case in view (the context builder, ``agents.context``): it listens, says back what it
understood, explores with the table, and proposes when the case is ready. Code guards the edges: the case file refuses a
fact without its source, readiness and the question ceiling are code, options are checked before the person sees them,
and every sentence to the person with a number nobody gave or code computed is dropped (as Home's own check does). A
refusal comes back to the agent as the tool's result, so it can think again.

Its tools are registered once, in scope "case" (``register``); the state of one case is :class:`CaseState`, the context
every tool receives. Since 37.11 (the owner, 2026-11-01) it leads Home by default wherever a model serves its purpose; the
switch ``DCLAB_CASE_AGENT`` set to 0 turns it off, and with no model the rules-first Home serves (the fallback and the open
edition's path). The conversation exam measures it as the arm "case".
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from dclab_core import case as C

from .. import prompts
from . import context as ctx
from .registry import Registry, Tool
from .runtime import Policy

PROMPT = "case_agent"
PURPOSE = "case_agent"
SWITCH = "DCLAB_CASE_AGENT"
TOOLS = ("read_table", "query", "check_leakage", "preview_step", "note", "add_unknown", "resolve", "confirm_assumption", "ask", "propose",
         "start", "request_build", "consult")  # 37.6: the guard layer's tools; 37.7: consult sends a specialist
BUILDS = ("ml_project", "segmentation_project")  # what a build request starts: a project
ENDS_TURN = ("ask",)  # a question ends the turn; after options or a start the agent writes its message (CEX-005: both came bare)
NO_DATA_WORK = ("advice",)  # an answer with no data work starts without a ready case: nothing is built
# 2026-11-01 (sealed v2: four route errors ended as advice): advice opens before the case is ready only when nothing points to
# data work; with a table shared, or a list, report, model or cleaning asked for, it waits for readiness like the rest
DATA_WORK = re.compile(r"\b(?:build|make|give|send|create|prepare|set up|get)\b[^.?!]{0,40}\b(?:list|report|dashboard|model|forecast|"
                       r"chart|ranking|file)s?\b|\bclean (?:up )?(?:my|the|our|this)\b|\b(?:rank|segment|group) (?:my|our|the) \w+|"
                       r"\bpredict (?:which|who|whether)\b|\bforecast (?:the|our|next)\b|\bdedup\w*|\bmerge (?:the|our|these|my)\b", re.I)
JUST_ANSWER = re.compile(r"\b(?:no data work|just (?:the )?answer|quick q(?:uestion)?|in general|in theory)\b", re.I)
HELP = re.compile(r"\b(?:can you help|help me|help us|what can you do|how can you help|i need help)\b", re.I)
TURN_BUDGET = 3  # the person's turns after which, with the goal and the data known, the agent proposes or starts
ROUTE_MODEL = "claude-haiku-5-5"  # the cheap second reading of the route


def data_work(state: "CaseState") -> bool:
    """Something in the case asks for work on data: the person asks for a list, report, model or cleaning; or a table is shared
    and the person asks for help in general rather than a specific question about method (the review of 2026-11-01: a shared
    table alone did not make d-40's yes-or-no leakage question data work)."""
    words = " ".join(state.case.words)
    if JUST_ANSWER.search(words):
        return False
    if DATA_WORK.search(words):
        return True
    if state.table is None:
        return False
    asks_method = "?" in words and bool(re.search(r"\b(?:leak\w*|metric|which|should (?:i|we)|is (?:it|this|that|\w+_\w+)|difference|versus|vs\.?|yes or no|how do i|why)\b", words, re.I))
    return bool(HELP.search(words)) or not asks_method


def route_reading(client: Any, state: "CaseState") -> Optional[str]:
    """A second, cheap reading of which kind of help the person needs: the person's own words and the table's column names
    only (never the assistant's messages, which may quote group names), answered with one track; None when it cannot."""
    if client is None:
        return None
    cols = ", ".join(c["name"] for c in (state.table or {}).get("columns") or [])
    words = "\n".join(f"- {w}" for w in state.case.words[-8:])
    try:
        if hasattr(client, "request_cells"):  # the person may repeat a group name the agent showed: then the request holds cells
            client.request_cells = bool(state.group_names) and any(g in " ".join(state.case.words) for g in state.group_names)
        out = client.complete([{"role": "system", "content": prompts.text("route_check")},
                               {"role": "user", "content": f"The person's messages:\n{words}\n\nTheir table's columns: {cols or 'no table shared'}"}],
                              None, max_tokens=1500)
    except Exception:  # noqa: BLE001 - a second reading that fails never blocks the agent
        return None
    text = str((out or {}).get("content") or "").lower()
    found = [t for t in C.TRACKS if re.search(rf"\b{t}\b", text)] if hasattr(C, "TRACKS") else []
    return found[0] if len(found) == 1 else None
DECLINES = re.compile(r"\b(?:not sure|no idea|don'?t know|do not know|can'?t say|rather not say|no clue)\b", re.I)
WHOLE_DECLINE = re.compile(r"^\s*(?:pass|skip|skip it|next|no idea|dunno)\W*$", re.I)
DECLINE_WORDS = 6  # the review: "I don't know who will leave, that is the problem" is an answer, not a pass
# the person asks to see what DCLab can do: the agent may propose then, with what is open stated (the prompt says so)
WANTS_OPTIONS = re.compile(r"\b(?:what can you do|show me (?:the |some |your )?options|just show me|"
                           r"show me what you can do|give me (?:some )?options|what are my options)\b", re.I)


YES_BUILD = re.compile(r"\b(?:yes|yeah|yep|go ahead|build it|please build|do it|start it|ok(?:ay)?|sure|let'?s go)\b", re.I)
NOT_NOW = re.compile(r"\b(?:no|not|don'?t|do not|wait|hold|later|stop|never|cancel)\b", re.I)


# 2026-10-31 (the sealed run: 17 questions about what the person had said, most right after "go ahead" or "run it")
GO_AHEAD = re.compile(r"\b(?:go ahead|send (?:it|the|me|them)|run (?:it|the|this|that)|do it|just do|stop asking|proceed|build it|start (?:it|now|the)|"
                      r"let'?s go|yes,? please|now,? please)\b", re.I)
HOLD = re.compile(r"\b(?:don'?t|do not|not yet|wait|hold on|hold off|later|never)\b", re.I)
CONFIRMS = re.compile(r"\b(?:does (?:that|this|it) (?:match|sound|work|fit)|can you confirm|could you confirm|do you want me to|"
                      r"would you like me to|want me to|shall i|should i|is (?:that|this) (?:right|ok|okay|correct|what you)|"
                      r"sound good|are you happy|ready (?:for me )?to|okay to|ok to)\b", re.I)


def goes_ahead(words: str) -> bool:
    """The person tells the agent to act now: a go-ahead with no hold in the same message ("stop asking" counts)."""
    w = str(words or "")
    return bool(GO_AHEAD.search(w)) and not HOLD.search(w)


# 2026-11-01: a long yes with conditions ("Yes, start the build ... hold the holdout until ...") failed on a word such as "hold"
# or "not" anywhere in it, so the build was refused and the agent asked for a yes again. A yes now leads the message or names
# the build; only a refusal aimed at building (or a leading no or wait) still refuses.
LEADING_YES = re.compile(r"^\W*(?:yes|yeah|yep|sure|ok(?:ay)?|go ahead|please (?:build|start|go ahead))\b", re.I)
BUILD_YES = re.compile(r"\b(?:start the build|go ahead with the build|build it|start building|build now|start it now)\b", re.I)
REFUSE_BUILD = re.compile(r"\b(?:don'?t|do not|not yet|wait|hold off|later|never|cancel)\b[^.!?]{0,40}\b(?:build|start|go ahead)|"
                          r"\bhold off\b|^\W*(?:no|not yet|wait)\b", re.I)


def says_yes(words: str) -> bool:
    """The person's yes to building: a leading yes, or the build named, or a yes word with no delay in the message; never a
    refusal aimed at building (the review of 37.6; 2026-11-01)."""
    w = str(words or "")
    if REFUSE_BUILD.search(w):
        return False
    return bool(LEADING_YES.search(w) or BUILD_YES.search(w)) or (bool(YES_BUILD.search(w)) and not NOT_NOW.search(w))


def declines(words: str) -> bool:
    """A short message that only passes over the question."""
    w = str(words or "")
    return bool(WHOLE_DECLINE.search(w)) or (len(w.split()) <= DECLINE_WORDS and bool(DECLINES.search(w)))


def on(models: Any = None) -> bool:
    """Whether the lead agent leads Home. ``DCLAB_CASE_AGENT`` decides when set (1 on, 0 off). Unset, it is the default where
    ``models`` serves its purpose (37.11: the owner's decision of 2026-11-01, made before a sealed exam passed: CEX-013 and
    CEX-017 failed their gates); with no model the rules-first Home serves."""
    value = os.environ.get(SWITCH, "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    available = getattr(models, "available", None)
    try:
        return bool(available(PURPOSE)) if callable(available) else False
    except Exception:  # noqa: BLE001 - a gateway that cannot answer serves no model: the fallback
        return False


@dataclass
class CaseState:
    """One case as the agent works on it: the case file, the table's view (never a value), and this turn's outputs."""

    case: C.CaseFile
    table: Optional[dict[str, Any]] = None
    computed: list[str] = field(default_factory=list)  # texts code computed: the numbers they hold may be said
    turns: list[dict[str, str]] = field(default_factory=list)  # the conversation as the agent saw it
    said: list[str] = field(default_factory=list)  # this turn's question, if the agent asked one
    shown: list[C.CaseOption] = field(default_factory=list)  # this turn's options, after code's checks
    asking: bool = False  # the last turn ended on a question
    wants_options: bool = False  # the person asked, in this message, to see what DCLab can do
    go_ahead: bool = False  # the person told the agent, in this message, to go ahead: no confirmation question (2026-10-31)
    queries: int = 0  # the table queries in this turn (at most QUERIES_A_TURN)
    budget_reached: bool = False  # 2026-11-01: three turns with the goal and the data known: propose or start, no more asking
    route_client: Any = None  # 2026-11-01: the cheap second reading of the route (purpose route_check); None: no check
    route_checked: bool = False  # the second reading asks once a case
    rewriting: bool = False  # the agent is rewriting dropped sentences: no move may run (the review: a question counted twice)
    trace: Any = None  # one tracer for the whole case, so its turns are one conversation in the record
    frame: Any = None  # the shared table itself (never sent to a model; the data tools read it)
    build: Optional[dict[str, Any]] = None  # 37.6: the build the person said yes to (their words), never one without
    group_names: set[str] = field(default_factory=set)  # every group name the case was shown (at most data_tools.MAX_NAMES)
    exposure: list[dict[str, Any]] = field(default_factory=list)  # what the answers looked at (H8): carried to a build
    specialist_client: Any = None  # 37.7: the client specialists are sent with (purpose case_specialist); None: none may be sent
    consulted: list[str] = field(default_factory=list)  # 37.7: the specialists already sent on this case
    spent_eur: float = 0.0  # 37.8: what this case's model work has cost so far
    cap_eur: float = field(default_factory=lambda: __import__("dclab_rnd.agents.case_session", fromlist=["cap_eur"]).cap_eur())  # 37.8: the soft cap (DCLAB_CASE_CAP_EUR), raised by the person's yes
    waiting_on_budget: bool = False  # 37.8: the case stopped at its cap and asked
    checkpoint: Any = None  # 37.8: where the state is saved after every turn (None: not saved)

    def __post_init__(self) -> None:
        object.__setattr__(self, "recorded", self.case.version)  # what the case already held is not a new event (a resumed case)
        object.__setattr__(self, "_ready", True)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name == "case" and getattr(self, "_ready", False):
            record_changes(self)

    def refuse(self, error: Exception) -> dict[str, Any]:
        return {"error": str(error)}


ACTORS = {"person": "person", "code": "system", "card": "system"}  # everyone else (the agent, a specialist) is "agent"


def record_changes(state: CaseState) -> None:
    """37.12: every new version of the case file is an event the moment it is made (case.changed), under the case's id and the
    turn's trace, so a kill mid-turn loses no change and a case rebuilds from its events alone (dclab_core.case.rebuild)."""
    import json

    from .. import experience

    for change in state.case.changes:
        if change.version <= getattr(state, "recorded", 0):
            continue
        experience.record("case.changed", ACTORS.get(change.by.split(":")[0], "agent"),
                          {"case": state.case.case_id, "version": change.version, "field": change.field, "op": change.op, "by": change.by,
                           "after": json.dumps(change.after, ensure_ascii=False, default=str)},
                          conversation=state.case.case_id)
        object.__setattr__(state, "recorded", change.version)


def _read_table(state: CaseState) -> dict[str, Any]:
    if not state.table:
        return {"error": "No table has been shared. Ask the person about their data, or work without one."}
    if state.case.data is None:  # what code read becomes the case's data fact (by code, computed)
        state.case = C.apply(state.case, "data", C.CaseFact(value=f"{state.table['table']}: {state.table['rows']} rows, "
                                                                 f"{len(state.table['columns'])} columns", status="computed",
                                                         computed=f"the upload's profile ({state.table['table']})"),
                             by="code", why="the table was read")
    out = {"table": state.table["table"], "rows": state.table["rows"], "columns": state.table["columns"]}
    import json

    state.computed.append(json.dumps(out))  # what code read: its numbers may be said
    return out


QUERIES_A_TURN = 2  # the prompt's "one or two questions of the table", kept by code


def _data(state: CaseState, name: str, **arguments: Any) -> dict[str, Any]:
    """37.6: one of the guard layer's data tools on the shared table; what it computes may be said to the person."""
    import json

    from . import data_tools

    if state.frame is None:
        return {"error": "No table has been shared: ask the person to share it, or work without numbers."}
    if name == "query":
        if state.queries >= QUERIES_A_TURN:  # CEX-014: queries took over the turns once the table was read by code
            return {"error": f"you have asked the table twice this turn: write to the person now with what you found, and ask "
                             f"more of the table next turn if it is still needed"}
        state.queries += 1
        arguments["seen"] = state.group_names
    out = data_tools.TOOLS[name](state.frame, **arguments)
    if "error" not in out:
        if out.get("exposure"):
            state.exposure.append(out["exposure"])
        state.computed.append(json.dumps(out, default=str))
    return out


def _query(state: CaseState, group_by: Optional[list[str]] = None, measures: Optional[list[dict[str, Any]]] = None,
           filters: Optional[list[dict[str, Any]]] = None, sort: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    return _data(state, "query", group_by=group_by, measures=measures, filters=filters, sort=sort)


def _check_leakage(state: CaseState, outcome: str) -> dict[str, Any]:
    return _data(state, "leakage", outcome=outcome)


def _preview_step(state: CaseState, kind: str, columns: Optional[list[str]] = None, filters: Optional[list[dict[str, Any]]] = None,
                  keys: Optional[list[str]] = None) -> dict[str, Any]:
    return _data(state, "preview_step", kind=kind, columns=columns, filters=filters, keys=keys)


def _request_build(state: CaseState, quote: str) -> dict[str, Any]:
    """A build only on a project route, a ready case (or the person's own choice), and the person's yes in their latest words."""
    route = state.case.route
    if route is None or route.tool not in BUILDS:
        return {"error": "a build starts a project: start a prediction or segmentation first, with the person's choice"}
    r = C.readiness(state.case)
    if not r.ready and not state.case.chosen:
        return {"error": "the case is not ready: " + "; ".join(r.reasons)}
    latest = state.case.words[-1] if state.case.words else ""
    if latest.startswith("I pick:") or not says_yes(latest) or not (quote and C.quote_ok(quote, quote, [latest])):
        return {"error": "a build needs the person's yes to building, in their latest message: ask them plainly, and pass their words as quote"}
    state.build = {"tool": route.tool, "quote": quote}
    try:  # the case file keeps the person's yes with their words (v2)
        state.case = C.apply(state.case, "build", C.CaseBuild(tool=route.tool, quote=quote), by="person", why="the person said yes to the build")
    except C.CaseRefused as error:
        return state.refuse(error)
    # the live check of 37.11 (2026-11-01): "nothing is built yet" was said word for word while Home started the build
    return {"recorded": route.tool, "next": "tell the person in a sentence that their yes is recorded and the build starts now; "
                                            "give no result or number before it has run"}


def _consult(state: CaseState, specialist: str, question: str) -> dict[str, Any]:
    """37.7: send a specialist with the case; its findings go into the case file for the lead agent, never to the person."""
    from . import context as ctx_
    from . import specialists as S

    if state.specialist_client is None:
        return {"error": "no specialist can be sent in this session"}
    why = S.allowed(specialist, state.case, state.table, state.consulted)
    if why:
        return {"error": why}
    plain = [c for c in state.computed if '"groups_not_shown"' not in c]  # never a query's answers
    case_text = ctx_.case_text(state.case)
    named = any(n and (n in question or n in case_text) for n in state.group_names)  # the review: names reach it in words too
    try:
        state.specialist_client.request_cells = named
    except AttributeError:
        pass
    out = S.consult(state.specialist_client, specialist, question, case_text, ctx_.table_text(state.table), plain)
    state.consulted.append(specialist)
    if "error" in out:
        return out
    findings = [C.CaseFinding(by=specialist, **f) for f in out["findings"]]
    state.case = C.apply(state.case, "findings", findings, by=f"specialist:{specialist}", why=question[:200])
    return {"findings": out["findings"], "next": "use what helps; ask the person about any question it raised"}


def _note(state: CaseState, field: str, value: str, status: str, quote: Optional[str] = None) -> dict[str, Any]:
    if status not in ("said", "assumed"):
        return {"error": "status is said (the person's words, quoted) or assumed (your supposition, shown to them)"}
    try:
        state.case = C.apply(state.case, field, C.CaseFact(value=value, status=status, quote=quote if status == "said" else None),
                             by="agent", why="noted by the lead agent")
    except (C.CaseRefused, ValueError) as error:
        return state.refuse(error)
    return {"noted": field, "readiness": C.readiness(state.case).model_dump(mode="json")}


def _add_unknown(state: CaseState, field: str, question: str, blocks: str = "some", rank: int = 0) -> dict[str, Any]:
    try:
        state.case = C.apply(state.case, "unknowns", C.Unknown(field=field, question=question, blocks=blocks, rank=rank), by="agent", why="open")
    except (C.CaseRefused, ValueError) as error:
        return state.refuse(error)
    return {"open": [u.field for u in state.case.unknowns]}


def _resolve(state: CaseState, field: str) -> dict[str, Any]:
    state.case = C.apply(state.case, "resolved", field, by="agent", why="answered")
    return {"open": [u.field for u in state.case.unknowns]}


def _confirm(state: CaseState, index: int, quote: str, reject: bool = False) -> dict[str, Any]:
    try:
        state.case = C.confirm(state.case, int(index), quote=quote, by="person", reject=bool(reject))
    except (C.CaseRefused, IndexError, ValueError) as error:
        return state.refuse(error)
    return {"assumption": state.case.assumptions[int(index)].model_dump(mode="json")}


def _ask(state: CaseState, question: str) -> dict[str, Any]:
    if not str(question).strip():
        return {"error": "the question is empty"}
    if state.budget_reached and not any(u.blocks == "all" for u in state.case.unknowns):
        return {"error": f"after {TURN_BUDGET} turns you know the goal and the data: propose or start now, stating anything still "
                         "open as an assumption the person can correct"}
    if state.go_ahead and CONFIRMS.search(str(question)):
        return {"error": "the person told you to go ahead: act now (start the work, or request_build if they said yes to building) "
                         "and tell them what you did; do not ask them to confirm again. Ask only about something the work cannot "
                         "start without."}
    state.case = C.asked(state.case)
    state.said.append(str(question).strip())
    state.asking = True
    return {"asked": True}


def _propose(state: CaseState, options: list[dict[str, Any]]) -> dict[str, Any]:
    r = C.readiness(state.case)
    if not r.may_propose and not state.case.chosen and not state.wants_options and not state.budget_reached:
        return {"error": "not yet: " + "; ".join(r.reasons) + ". Ask about what is missing, or propose once the ceiling is reached."}
    try:
        offered = [C.CaseOption(**{k: v for k, v in o.items() if k in C.CaseOption.model_fields}) for o in options or []]
    except ValueError as error:
        return state.refuse(error)
    columns = {c["name"] for c in (state.table or {}).get("columns") or []}
    kept, refused = C.check_options(offered, state.case, columns=columns, computed=state.computed)
    if len(kept) < 2:
        return {"error": "fewer than two options passed the checks", "refused": refused}
    state.case = C.apply(state.case, "options", kept, by="agent", why="proposed")
    state.shown = kept
    return {"shown": [o.title for o in kept], "refused": refused,
            "next": "the person sees these under your message: now write that message (a sentence or two leading into them), then stop"}


def _start(state: CaseState, track: str, tool: str, quote: Optional[str] = None) -> dict[str, Any]:
    if C.TRACK_OF_TOOL.get(tool) != track:
        return {"error": f"DCLab cannot run {track}/{tool}"}
    route = state.case.route
    if route is not None:  # the review of 37.3: a start after the person's pick rewrote their route as the agent's
        if (route.track, route.tool) == (track, tool):
            return {"started": f"{track}/{tool}", "next": "already under way: tell the person what happens next, then stop"}
        latest = state.case.words[-1:] if state.case.words else []
        if not (quote and C.quote_ok(quote, quote, latest)):
            return {"error": f"the work has started as {route.track}/{route.tool}. Change it only when the person asks to in their "
                             "latest message: pass their words as quote. A first step on the way (a base rate) is part of the same work."}
        state.case = C.apply(state.case, "route", C.CaseRoute(track=track, tool=tool, by="person"), by="person", why=f"the person asked: {quote}")
        return {"started": f"{track}/{tool}", "next": "tell the person the work changed as they asked"}
    r = C.readiness(state.case)
    if track in NO_DATA_WORK and not state.case.chosen and not r.ready and not state.budget_reached and data_work(state):
        return {"error": "the person's need points to data work (a shared table, or a list, report, model or cleaning): advice "
                         "answers a question with no work on data. Propose or start the work their need calls for."}
    if not state.case.chosen and not r.ready and not state.budget_reached and track not in NO_DATA_WORK:
        return {"error": "the person has not chosen and the case is not ready: " + "; ".join(r.reasons)}
    if not state.case.chosen and state.route_client is not None and not state.route_checked:
        state.route_checked = True
        second = route_reading(state.route_client, state)
        if second and second != track:
            return {"error": f"a second reading of the person's own words points to {second}, not {track}. Read their words again; "
                             f"start {second} unless their words clearly call for {track}."}
    state.case = C.apply(state.case, "route", C.CaseRoute(track=track, tool=tool, by="agent"), by="agent", why="work started")
    return {"started": f"{track}/{tool}", "next": "now tell the person in a few sentences what happens next, then stop"}


MOVES = {"consult": "case.consult", "request_build": "case.build", "note": "case.note", "add_unknown": "case.unknown", "resolve": "case.resolve", "confirm_assumption": "case.confirm",
         "ask": "case.ask", "propose": "case.propose", "start": "case.start"}


class CaseGuard:
    """The case scope's writes change one case file. The case file refuses a fact without its source and code decides
    readiness and the options; this guard refuses a call that names no case, a move it does not know, and any move but a
    question once the work has started (the route is the person's or the ready case's, never changed silently)."""

    def __call__(self, tool: Tool, context: Any, arguments: dict[str, Any], proceed):
        if not isinstance(context, CaseState):
            return {"error": "the case tools need a case: run them through the lead agent's turn"}
        if tool.move not in MOVES.values():
            return {"error": f"{tool.move!r} is not a move on a case"}
        if context.rewriting:
            return {"error": "only rewrite your message now: no tool runs while you rewrite"}
        if context.case.route is not None and tool.move == "case.propose":  # new options after the start flipped a route (CEX-007)
            return {"error": "the work has started: tell the person what happens next, or ask what they want changed"}
        return proceed()


def register(registry: Registry) -> None:
    S = {"type": "string"}
    from . import data_tools

    defs = [
        ("query", data_tools.DESCRIPTIONS["query"], data_tools.SCHEMAS["query"]["properties"], [], _query, "read"),
        ("check_leakage", data_tools.DESCRIPTIONS["leakage"], data_tools.SCHEMAS["leakage"]["properties"], ["outcome"], _check_leakage, "read"),
        ("preview_step", data_tools.DESCRIPTIONS["preview_step"], data_tools.SCHEMAS["preview_step"]["properties"], ["kind"], _preview_step, "read"),
        ("request_build", "Ask DCLab to build the project the person chose, once they said yes: pass their words as quote.",
         {"quote": S}, ["quote"], _request_build, "write"),
        ("consult", "Send a specialist with the case when a question needs depth: data_investigator (what the table can answer), "
                    "leakage_auditor (columns known only after the decision), critic (what is weak before you propose), "
                    "domain_expert (what matters in this kind of business), solution_designer (how started work should be set up). "
                    "Its findings come back to you and go into the case file. Code decides whether it may be sent now.",
         {"specialist": {"type": "string", "enum": ["data_investigator", "leakage_auditor", "critic", "domain_expert", "solution_designer"]},
          "question": S}, ["specialist", "question"], _consult, "write"),
        ("read_table", "Read the shared table's columns: names, kinds and what DCLab noticed about each, and its row count. No values.", {}, [], _read_table, "read"),
        ("note", "Record a fact in the case file: goal, decision, user (who uses the result), data, cannot_answer, constraints or "
                 "assumptions. status said needs the person's exact words as quote; status assumed is your supposition.",
         {"field": {"type": "string", "enum": [*C.SINGLE, *C.LISTS]}, "value": S, "status": {"type": "string", "enum": ["said", "assumed"]}, "quote": S},
         ["field", "value", "status"], _note, "write"),
        ("add_unknown", "Record something still unknown and which options wait for it.",
         {"field": S, "question": S, "blocks": {"type": "string", "enum": ["all", "some", "none"]}, "rank": {"type": "integer"}},
         ["field", "question", "blocks"], _add_unknown, "write"),
        ("resolve", "Mark an unknown answered.", {"field": S}, ["field"], _resolve, "write"),
        ("confirm_assumption", "The person confirmed (or rejected) assumption number index, in these words of theirs.",
         {"index": {"type": "integer"}, "quote": S, "reject": {"type": "boolean"}}, ["index", "quote"], _confirm, "write"),
        ("ask", "Ask the person one question. It ends your turn.", {"question": S}, ["question"], _ask, "write"),
        ("propose", "Show the person two or three options that differ in kind. Code checks them; then write your message leading into them.",
         {"options": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "object", "required": ["title", "what", "gets", "tradeoff", "effort", "track", "tool"], "properties": {
             "title": S, "what": S, "gets": S, "tradeoff": S, "effort": S, "track": {"type": "string", "enum": list(C.TRACKS)},
             "tool": {"type": "string", "enum": list(C.TOOLS)}, "columns": {"type": "array", "items": S}}}}},
         ["options"], _propose, "write"),
        ("start", "Start the work the person chose, or that their clear need calls for (an advice answer too); then tell them what happens next. "
                  "To change work already started, quote the person's latest words asking for the change.",
         {"track": {"type": "string", "enum": list(C.TRACKS)}, "tool": {"type": "string", "enum": list(C.TOOLS)}, "quote": S}, ["track", "tool"], _start, "write"),
    ]
    for name, description, props, required, handler, effect in defs:
        registry.register(Tool(name=f"case_{name}", description=description, parameters={"type": "object", "properties": props, "required": required},
                               handler=handler, effect=effect, scope="case", takes_context=True, move=MOVES.get(name)))
    registry.guards["case"] = CaseGuard()


TOOL_NAMES = tuple(f"case_{t}" for t in TOOLS)
POLICY = Policy(name="case", system="", tools=TOOL_NAMES, max_steps=14, max_replies=8, stop_after=tuple(f"case_{t}" for t in ENDS_TURN))


def unconfirmed(text: str) -> str:
    """A message without its sentences that ask the person to confirm again (said after they told the agent to go ahead)."""
    parts = re.split(r"(?<=[.?!])\s+", str(text or "").strip())
    kept = [p for p in parts if not (p.rstrip().endswith("?") and CONFIRMS.search(p)) and not re.search(r"\ba plain yes\b", p, re.I)]
    return " ".join(kept) if kept else ""


def checked(text: str, state: CaseState, dropped: Optional[list[str]] = None) -> str:
    """A message to the person with every sentence holding a number nobody gave or code computed dropped (and listed in
    ``dropped``, so the agent can be told and write it again). The rule is cited.unsaid, the conversation exam's own."""
    from .. import cited

    sources = [*state.case.words, *state.computed]
    kept = []
    for s in re.split(r"(?<=[.?!])\s+", str(text or "").strip()):
        if not s:
            continue
        unsaid = cited.unsaid(s, sources)
        if unsaid:
            if dropped is not None:
                dropped.append(f"{s} (numbers nobody gave: {', '.join(unsaid)})")
            continue
        kept.append(s)
    return " ".join(kept)


SAY_NOW = ("[DCLab's code] You have not written to the person this turn. Write your message now: what you understood or found, "
           "then one question or the next step. No tool runs now.")
FALLBACK = "Could you tell me a little more about what you need?"
FALLBACK_NUMBERS = ("I want to give you only numbers that come from you or from your data, so let me put that differently: "
                    "could you tell me a little more about what you need?")
REWRITE = ("[DCLab's code] These sentences were not shown to the person, because their numbers come from nobody: the person did "
           "not give them and no tool computed them:\n{dropped}\nWrite your whole message again, keeping its answer. Quote numbers "
           "exactly as the tools returned them (no rounding, no adding up) or as the person gave them; otherwise say it without "
           "digits. Reply with the message only.")
REWRITE_STRICT = ("[DCLab's code] Your message was cut again and the person would get no answer. Write the answer itself now, in "
                  "words: no digits at all unless they are copied exactly from a tool's result or the person's words. Reply with the "
                  "message only.")


def has_statement(text: str) -> bool:
    """A message that says something, not only asks (2026-11-01: a cut answer left a bare follow-up question)."""
    parts = [p for p in re.split(r"(?<=[.?!])\s+", str(text or "").strip()) if p.strip()]
    return any(not p.rstrip().endswith("?") for p in parts)


def turn(state: CaseState, client: Any, words: str, registry: Optional[Registry] = None) -> dict[str, Any]:
    """One turn: the person's words heard, the context built, the agent run until it asks, proposes, starts or answers.
    Returns {"text", "options", "route"}."""
    from . import default_registry
    from .runtime import run as run_policy  # the harness's loop: every step traced and recorded (21.1)

    registry = registry or default_registry()
    from . import case_session

    stop = case_session.ask_to_continue(state, str(words))  # 37.8: at the cap, ask before any more model work
    if stop is not None:
        state.case = C.hear(state.case, words)
        state.turns += [{"role": "user", "content": str(words)}, {"role": "assistant", "content": stop}]
        if state.checkpoint:
            case_session.save(state, state.checkpoint)
        return {"text": stop, "options": [], "route": state.case.route, "asking": True}
    if state.asking and declines(str(words)):
        state.case = C.declined(state.case)
    state.case = C.hear(state.case, words)
    state.wants_options = bool(WANTS_OPTIONS.search(str(words)))  # this message only (the review: it stayed on for the case)
    state.go_ahead = goes_ahead(str(words))  # this message only
    state.queries = 0
    if state.table and not any(str(c).startswith('{"table": ') for c in state.computed):  # 2026-10-31: code reads the shared
        _read_table(state)  # table (again after a restore), so no model call is spent on it and its counts may be said
    if state.trace is None:
        from .harness import tracer

        state.trace = tracer("case")
    state.turns.append({"role": "user", "content": str(words)})
    state.said, state.shown, state.asking = [], [], False
    person_turns = sum(1 for t in state.turns if t.get("role") == "user")
    state.budget_reached = (person_turns > TURN_BUDGET and C.known(state.case.goal) and C.known(state.case.data)
                            and state.case.route is None and not state.case.chosen)  # the review: off once work started or was picked
    built = ctx.build(state.case, state.turns, state.table)
    messages = ctx.messages(prompts.text(PROMPT), built)
    start = len(messages)
    before = _spent(client) + _spent(state.specialist_client)
    client = _Cells(client, state)  # 37.6: a request holds cell values only when a query's answer is in it
    result = run_policy(POLICY, client, registry, messages, context=state, trace=state.trace)
    texts = [str(m.get("content") or "").strip() for m in result.messages[start:] if m.get("role") == "assistant" and str(m.get("content") or "").strip()]
    if not texts and not state.said:  # CEX-011 d-04: every reply spent on queries, no word for the person: asked once, no move allowed
        again_from = len(result.messages)
        result.messages.append({"role": "user", "content": SAY_NOW})
        once = Policy(name="case", system="", tools=TOOL_NAMES, max_steps=POLICY.max_steps, max_replies=1, stop_after=POLICY.stop_after)
        state.rewriting = True
        try:
            run_policy(once, client, registry, result.messages, context=state, trace=state.trace)
        finally:
            state.rewriting = False
        texts = [str(m.get("content") or "").strip() for m in result.messages[again_from + 1:] if m.get("role") == "assistant" and str(m.get("content") or "").strip()]
    dropped: list[str] = []
    words_out = _unique([checked(t, state, dropped) for t in texts] + [checked(q, state, dropped) for q in state.said])
    if state.go_ahead or says_yes(str(words)):  # 2026-11-01: after a go-ahead, no question asking the person to confirm again
        trimmed = [unconfirmed(w) for w in words_out]
        if any(trimmed):  # a message that was only the question is kept rather than replaced by code's fallback line
            words_out = trimmed
    if dropped:  # a refusal comes back to the agent so it can write again (the plan's rule), once: never a silent turn
        again_from = len(result.messages)
        result.messages.append({"role": "user", "content": REWRITE.format(dropped="\n".join(f"- {d}" for d in dropped))})
        once = Policy(name="case", system="", tools=TOOL_NAMES, max_steps=POLICY.max_steps, max_replies=1, stop_after=POLICY.stop_after)
        state.rewriting = True  # the tools stay offered (the API needs them beside earlier tool calls) but none may run
        try:
            run_policy(once, client, registry, result.messages, context=state, trace=state.trace)
        finally:
            state.rewriting = False
        rewritten = [str(m.get("content") or "").strip() for m in result.messages[again_from + 1:] if m.get("role") == "assistant"]
        fixed = [w for w in (checked(t, state) for t in rewritten if t) if w]
        kept = sum(len(w) for w in fixed) / max(1, sum(len(t) for t in rewritten if t))
        if not any(has_statement(w) for w in fixed) or kept < 0.6:  # cut again: one stricter try, so the answer reaches the person
            strict_from = len(result.messages)
            result.messages.append({"role": "user", "content": REWRITE_STRICT})
            state.rewriting = True
            try:
                run_policy(once, client, registry, result.messages, context=state, trace=state.trace)
            finally:
                state.rewriting = False
            strict = [str(m.get("content") or "").strip() for m in result.messages[strict_from + 1:] if m.get("role") == "assistant"]
            again = [w for w in (checked(t, state) for t in strict if t) if w]
            if any(has_statement(w) for w in again):
                fixed = again
        if fixed:
            words_out = _unique(fixed)
    if not any(words_out):  # never a silent turn (the review): code's own line, and the question if one was asked
        words_out = [FALLBACK_NUMBERS if dropped else FALLBACK] + ([q for q in (checked(q, state) for q in state.said) if q] or [])
    text = "\n\n".join(w for w in words_out if w)
    state.spent_eur += max(0.0, _spent(client) + _spent(state.specialist_client) - before)  # 37.8: what this turn cost
    state.turns.append({"role": "assistant", "content": text + ("\n(Options: " + "; ".join(o.title for o in state.shown) + ")" if state.shown else "")})
    if state.checkpoint:  # 37.8: saved after every turn, so a stop or a crash loses nothing
        case_session.save(state, state.checkpoint)
    return {"text": text, "options": list(state.shown), "route": state.case.route, "asking": state.asking}


def _spent(client: Any) -> float:
    try:
        return float(getattr(client, "spent_eur", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


class _Cells:
    """A client that tells the gateway, request by request, whether the messages hold a query's answer (group names: cell
    values). Requests without one are kept whole (the review of 37.6: the purpose's limit had stopped every request's text
    from being kept)."""

    def __init__(self, inner: Any, state: "CaseState"):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_state", state)

    def _mark(self, messages: list[dict[str, Any]]) -> None:
        # the review: a later turn repeats the names in the agent's own words, and a non-ASCII name never matched its stored
        # text: once the case has been shown group names, every request of it holds cells
        held = bool(self._state.group_names) or any(
            m.get("role") == "tool" and '"groups_not_shown"' in str(m.get("content") or "") for m in messages)
        try:
            self._inner.request_cells = held
        except AttributeError:
            pass

    def complete(self, messages: list[dict[str, Any]], *args: Any, **kwargs: Any) -> Any:
        self._mark(messages)
        return self._inner.complete(messages, *args, **kwargs)

    def stream(self, messages: list[dict[str, Any]], *args: Any, **kwargs: Any) -> Any:
        self._mark(messages)
        return self._inner.stream(messages, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)


def _unique(parts: list[str]) -> list[str]:
    """Each paragraph once (CEX-006: the question was also the message's words, and showed twice)."""
    seen, out = set(), []
    for p in parts:
        key = re.sub(r"\s+", " ", p).strip().lower()
        if key and not any(key in s or s in key for s in seen):
            seen.add(key)
            out.append(p)
    return out


def pick(state: CaseState, client: Any, title: str, registry: Optional[Registry] = None) -> dict[str, Any]:
    """The person picked an option: their choice is recorded (by the person) and the agent answers it."""
    option = next((o for o in state.case.options if o.title == title), None)
    state.case = C.apply(state.case, "chosen", title, by="person", why="picked an option")
    if option is not None:
        state.case = C.apply(state.case, "route", C.CaseRoute(track=option.track, tool=option.tool, by="person"), by="person", why="picked an option")
    return turn(state, client, f"I pick: {title}", registry)


def table_of(name: str, view: list[Mapping[str, Any]], rows: int) -> dict[str, Any]:
    return {"table": name, "rows": rows, "columns": [dict(c) for c in view]}


__all__ = ["CaseState", "POLICY", "PROMPT", "PURPOSE", "SWITCH", "TOOL_NAMES", "checked", "on", "pick", "register", "table_of", "turn"]
