"""The lead agent (packages 37.3 and 37.4, docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): one agent holds a case from the
person's first sentence to the route, around the case file (``dclab_core.case``), on the shared runtime (``agents.run``).

Claude thinks first, with the whole case in view (the context builder, ``agents.context``): it listens, says back what it
understood, explores with the table, and proposes when the case is ready. Code guards the edges: the case file refuses a
fact without its source, readiness and the question ceiling are code, options are checked before the person sees them,
and every sentence to the person with a number nobody gave or code computed is dropped (as Home's own check does). A
refusal comes back to the agent as the tool's result, so it can think again.

Its tools are registered once, in scope "case" (``register``); the state of one case is :class:`CaseState`, the context
every tool receives. The switch is ``DCLAB_CASE_AGENT`` (off by default); the conversation exam measures it as the arm
"case" before it serves anyone.
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
TOOLS = ("read_table", "note", "add_unknown", "resolve", "confirm_assumption", "ask", "propose", "start")
ENDS_TURN = ("ask",)  # a question ends the turn; after options or a start the agent writes its message (CEX-005: both came bare)
NO_DATA_WORK = ("advice",)  # an answer with no data work starts without a ready case: nothing is built
DECLINES = re.compile(r"\b(?:not sure|no idea|don'?t know|do not know|can'?t say|rather not say|no clue)\b", re.I)
WHOLE_DECLINE = re.compile(r"^\s*(?:pass|skip|skip it|next|no idea|dunno)\W*$", re.I)
DECLINE_WORDS = 6  # the review: "I don't know who will leave, that is the problem" is an answer, not a pass
# the person asks to see what DCLab can do: the agent may propose then, with what is open stated (the prompt says so)
WANTS_OPTIONS = re.compile(r"\b(?:what can you do|show me (?:the |some |your )?options|just show me|"
                           r"show me what you can do|give me (?:some )?options|what are my options)\b", re.I)


def declines(words: str) -> bool:
    """A short message that only passes over the question."""
    w = str(words or "")
    return bool(WHOLE_DECLINE.search(w)) or (len(w.split()) <= DECLINE_WORDS and bool(DECLINES.search(w)))


def on() -> bool:
    return os.environ.get(SWITCH, "").strip().lower() in ("1", "true", "yes", "on")


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
    rewriting: bool = False  # the agent is rewriting dropped sentences: no move may run (the review: a question counted twice)
    trace: Any = None  # one tracer for the whole case, so its turns are one conversation in the record

    def refuse(self, error: Exception) -> dict[str, Any]:
        return {"error": str(error)}


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
    state.case = C.asked(state.case)
    state.said.append(str(question).strip())
    state.asking = True
    return {"asked": True}


def _propose(state: CaseState, options: list[dict[str, Any]]) -> dict[str, Any]:
    r = C.readiness(state.case)
    if not r.may_propose and not state.case.chosen and not state.wants_options:
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
    if not state.case.chosen and not r.ready and track not in NO_DATA_WORK:
        return {"error": "the person has not chosen and the case is not ready: " + "; ".join(r.reasons)}
    state.case = C.apply(state.case, "route", C.CaseRoute(track=track, tool=tool, by="agent"), by="agent", why="work started")
    return {"started": f"{track}/{tool}", "next": "now tell the person in a few sentences what happens next, then stop"}


MOVES = {"note": "case.note", "add_unknown": "case.unknown", "resolve": "case.resolve", "confirm_assumption": "case.confirm",
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
    defs = [
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
         {"options": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "object", "required": ["title", "what", "track", "tool"], "properties": {
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


FALLBACK = "Could you tell me a little more about what you need?"
FALLBACK_NUMBERS = ("I want to give you only numbers that come from you or from your data, so let me put that differently: "
                    "could you tell me a little more about what you need?")
REWRITE = ("[DCLab's code] These sentences were not shown to the person, because their numbers come from nobody: the person did "
           "not give them and no tool computed them:\n{dropped}\nWrite your message again without those numbers (use the person's "
           "own numbers, or none). Reply with the message only.")


def turn(state: CaseState, client: Any, words: str, registry: Optional[Registry] = None) -> dict[str, Any]:
    """One turn: the person's words heard, the context built, the agent run until it asks, proposes, starts or answers.
    Returns {"text", "options", "route"}."""
    from . import default_registry
    from .runtime import run as run_policy  # the harness's loop: every step traced and recorded (21.1)

    registry = registry or default_registry()
    if state.asking and declines(str(words)):
        state.case = C.declined(state.case)
    state.case = C.hear(state.case, words)
    state.wants_options = bool(WANTS_OPTIONS.search(str(words)))  # this message only (the review: it stayed on for the case)
    if state.trace is None:
        from .harness import tracer

        state.trace = tracer("case")
    state.turns.append({"role": "user", "content": str(words)})
    state.said, state.shown, state.asking = [], [], False
    built = ctx.build(state.case, state.turns, state.table)
    messages = ctx.messages(prompts.text(PROMPT), built)
    start = len(messages)
    result = run_policy(POLICY, client, registry, messages, context=state, trace=state.trace)
    texts = [str(m.get("content") or "").strip() for m in result.messages[start:] if m.get("role") == "assistant" and str(m.get("content") or "").strip()]
    dropped: list[str] = []
    words_out = _unique([checked(t, state, dropped) for t in texts] + [checked(q, state, dropped) for q in state.said])
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
        if fixed:
            words_out = _unique(fixed)
    if not any(words_out):  # never a silent turn (the review): code's own line, and the question if one was asked
        words_out = [FALLBACK_NUMBERS if dropped else FALLBACK] + ([q for q in (checked(q, state) for q in state.said) if q] or [])
    text = "\n\n".join(w for w in words_out if w)
    state.turns.append({"role": "assistant", "content": text + ("\n(Options: " + "; ".join(o.title for o in state.shown) + ")" if state.shown else "")})
    return {"text": text, "options": list(state.shown), "route": state.case.route, "asking": state.asking}


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
