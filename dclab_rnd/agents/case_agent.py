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
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional

from dclab_core import case as C

from .. import prompts
from . import context as ctx
from .registry import Registry, Tool
from .runtime import Policy

PROMPT = "case_agent"
# Stage 4 of the plan of 2026-11-16 (think first, pick the kind of work last), behind a switch that is off by default:
# with it off, the prompt, the first-turn note and the reply's limit are exactly what they were.
THINK_FIRST = "DCLAB_CASE_THINK_FIRST"
THINK_FIRST_PROMPT = "case_agent_think_first"


def think_first() -> bool:
    """Whether the lead agent thinks first: its first full message says the problem, what needs care, the ways to solve it
    and the plan, and the plan is kept whole and shown back to it. On unless ``DCLAB_CASE_THINK_FIRST`` is 0.

    The owner's decision of 2026-11-16, made with the gate not passed. On the eight held-out people (HAB-060, HAB-061; the
    product's model) the lead agent went from 2.75 / 2.38 / 2.38 (understanding / solution / plan), accepted for 3 of 8,
    to 3.88 / 3.75 / 3.62, accepted for 7 of 8, against the same model alone at 3.62 / 3.62 / 3.88 and 6 of 8. It failed
    ``hab_gate_v2`` on one check of eleven: one forbidden thing was done where the gate allows none (the model alone did
    two). With a weaker main model (Nemotron 3.5 Super, HAB-062) the gain did not carry. 0 puts the first prompt back."""
    return os.environ.get(THINK_FIRST, "").strip().lower() not in ("0", "off", "false", "no")


def prompt_name() -> str:
    """The lead agent's prompt file: the think-first one when its switch is on."""
    return THINK_FIRST_PROMPT if think_first() else PROMPT
PURPOSE = "case_agent"
SWITCH = "DCLAB_CASE_AGENT"
TOOLS = ("read_table", "query", "answer", "dashboard", "check_leakage", "preview_step", "note", "add_unknown", "resolve", "confirm_assumption", "ask", "propose",
         "start", "request_build", "consult", "design_solution_without_data")  # 37.6: the guard layer's tools; 37.7: consult sends a specialist
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
# the cheap second reading of the route when the product runs on Anthropic (2026-11-11: on OpenRouter the cheap tier's model
# serves it, so a Qwen name is never sent to Anthropic's endpoint, nor a Claude name to OpenRouter)
ROUTE_MODEL = "claude-haiku-5-5"


def route_model() -> str:
    """The model for the second reading of the route: DCLAB_ROUTE_MODEL when set, the cheap tier's model when running on
    OpenRouter (qwen/qwen3.8-flash by default), else ROUTE_MODEL (Anthropic's Haiku)."""
    from ..models import settings as model_settings

    explicit = os.environ.get("DCLAB_ROUTE_MODEL", "").strip()
    if explicit:
        return explicit
    cheap = model_settings.tier("cheap")
    if model_settings.openrouter_default() or model_settings._host(str(cheap.get("base_url") or "")) == model_settings.OPENROUTER_HOST:
        return str(cheap.get("model") or model_settings.OPENROUTER_DEFAULT_MODEL)
    return ROUTE_MODEL


def route_now(state: "CaseState") -> Optional[str]:
    """The second reading of the route for the person's words so far, read again only when they said something new."""
    if state.route_client is None:
        return None
    thread, state.route_thread = state.route_thread, None
    if thread is not None:  # read ahead at the turn's start (route_ahead): waited for here, never asked twice
        thread.join()
    heard = len(state.case.words)
    if state.route_seen is None or state.route_seen[0] != heard:
        state.route_seen = (heard, route_reading(state.route_client, state))
    return state.route_seen[1]


ROUTE_WAIT = 3.0  # seconds a finished turn waits for a second reading it did not need (the review of stage 1: never a ready answer held)


def route_ahead(state: "CaseState") -> None:
    """Start the second reading when the turn starts, beside the lead model's own work (stage 1 of 2026-11-15: read inline at a
    propose or a start it held the turn for about 14 seconds). The same request with the same thinking: only the waiting
    changes. Not started where it would not be read: a chip or a choice decides, or the person's own words name the help."""
    if os.environ.get("DCLAB_ROUTE_AHEAD", "1").strip().lower() in ("0", "off", "false"):
        return
    if state.route_thread is not None and not state.route_thread.is_alive():
        state.route_thread = None  # one left by a turn that failed: finished, and no longer in the way
    if state.route_client is None or state.case.chosen or state.chip or state.route_thread is not None:
        return
    heard = len(state.case.words)
    if state.route_seen is not None and state.route_seen[0] == heard:
        return
    if any(t != "advice" for t in C.named_tracks(state.case.words)):
        return
    # only where a propose or a start could reach the second reading this turn: a turn that can only ask is not charged for it
    if not (C.readiness(state.case, state.table).may_propose or state.wants_options or state.budget_reached or state.clear_on_arrival or state.go_ahead):
        return
    import contextvars
    import threading

    from dclab_rnd import experience

    run_id = getattr(state.trace, "run_id", None)

    def read() -> None:
        with experience.bind(run=run_id, conversation=run_id):  # the turn's run is bound after this thread's context is copied
            state.route_seen = (heard, route_reading(state.route_client, state))

    state.route_thread = threading.Thread(target=contextvars.copy_context().run, args=(read,), daemon=True)  # the turn's ids go with it
    state.route_thread.start()


def route_doubt(state: "CaseState", move: str, tracks: set[str]) -> Optional[str]:
    """The plan's step 1 (2026-11-01): before options are shown or work starts, the kind of help the person's own words name
    (code) and the cheap second reading (route_check) are compared with what the agent chose; a kind of help missing from it
    is told once a move, with the person's words quoted, and the agent decides again (AI first, the check second)."""
    named = C.named_tracks(state.case.words)
    key = f"{move}:{len(state.case.words)}"  # once a move for what the person has said so far (the review: not once per set of points)
    if key in state.route_told:
        return None
    latest_only = state.case.words[-1] if state.case.words else ""
    if state.go_ahead and not C.named_tracks([latest_only]) and any(k.startswith(move + ":") for k in state.route_told):
        return None  # Stage 2: "go ahead" says nothing new about the kind of help; a doubt already told for this move is not raised again on it
    strong = {t: p for t, p in named.items() if t != "advice"}  # advice words ("is that a good idea") are the weakest sign: they hide data work
    if set(strong) & tracks:  # the review: what their words name is covered; the second reading may not push a model onto a rows request
        return None
    second = None if strong else route_now(state)  # their own words outrank the second reading
    missing = {t: p for t, p in named.items() if t not in tracks}
    points = {**missing, **({second: ""} if second and second not in tracks else {})}
    # The gates must agree (HAB-002, the delivery-routes case: the second reading named advice, advice was then refused as data
    # work, analysis was refused for lack of a table, and no kind of help was left). A kind of help the next gate would refuse is
    # never asked for here: advice while the words ask for data work, and work on a table while none is shared.
    points = {t: p for t, p in points.items() if not ((t in NO_DATA_WORK and data_work(state)) or (t in CHIP_NEEDS_TABLE and state.table is None))}
    if not points:
        return None
    state.route_told.append(key)
    latest = state.case.words[-1] if state.case.words else ""
    said = "; ".join(f"{t}" + (f' (their words: "{p}")' if p else " (a second reading of their words)") for t, p in points.items())
    chosen = ", ".join(sorted(tracks))
    if move == "propose":
        return (f"the person's need may call for {said}, which none of your options offers ({chosen}). Their latest words: "
                f"\"{latest[:300]}\". Read them again and include that kind of help among the options unless their words clearly rule it out.")
    return (f"the person's need may call for {said}, not {chosen}. Their latest words: \"{latest[:300]}\". Read them again; start "
            f"that kind of help, or propose it beside yours, unless their words clearly call for {chosen}: then send the same start "
            "again, and it goes through (this is said once).")


def data_work(state: "CaseState") -> bool:
    """Something in the case asks for work on data: the person asks for a list, report, model or cleaning; or a table is shared
    and the person asks for help in general rather than a specific question about method (the review of 2026-11-01: a shared
    table alone did not make d-40's yes-or-no leakage question data work)."""
    words = " ".join(state.case.words)
    if JUST_ANSWER.search(words):
        return False
    if DATA_WORK.search(words) or (C.CLEAR_GOAL_OR_ROUTE.search(words) and C.CLEAR_SUBJECT.search(words)):
        return True
    if state.chip in CHIP_NEEDS_TABLE:
        return True
    named = set(C.named_tracks(state.case.words)) - {"advice"}
    if named:
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
    said = " ".join(state.case.words)
    # whole words, any case; a value under three characters ("No", "A") names nothing and would switch the check off for good
    if any(len(g) >= 3 and re.search(rf"(?<!\w){re.escape(g)}(?!\w)", said, re.I) for g in state.group_names or ()):
        return None  # the review of step 1: their words repeat a table value the case was shown; route_check holds no cell value
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
HOLD = re.compile(r"\b(?:don'?t|do not|not yet|not now|wait|hold on|hold off|later|never|until|once|after|tomorrow|next week|won'?t|can'?t|cannot|rather not|no need)\b", re.I)
CONFIRMS = re.compile(r"\b(?:does (?:that|this|it) (?:match|sound|work|fit)|can you confirm|could you confirm|do you want me to|"
                      r"would you like me to|want me to|shall i|should i|is (?:that|this) (?:right|ok|okay|correct|what you)|"
                      r"sound good|are you happy|ready (?:for me )?to|okay to|ok to)\b", re.I)


ASKS_NOW = re.compile(r"\b(?:show|give|send|make|draw|do)\b|\bi (?:asked|sked)\b|\bnow\b", re.I)  # an order for a figure, a chart or a report
BARE_YES = re.compile(r"^\W*(?:yes|yeah|yep|ok(?:ay)?|sure|go|go on|do it|right|yes,? right|yes,? please|please)\W*$", re.I)


def goes_ahead(words: str, asking: bool = False) -> bool:
    """The person tells the agent to act now: a go-ahead with no hold in the same message ("stop asking" counts). Since
    2026-11-02 also an order for figures ("show me the report", "give plot", "now show my dashboard") and a bare yes or go
    that answers the agent's own question."""
    w = str(words or "")
    if HOLD.search(w) or REFUSES_BUILDING.search(w):  # a delegation or an order inside a refusal ("no model please, whatever you think") is none
        return False
    return bool(PAGE_BUILD.match(w)) or bool(GO_AHEAD.search(w)) or orders_model(w) or bool(DELEGATE.search(w)) or (C.wants_figures(w) and bool(ASKS_NOW.search(w))) or (asking and bool(BARE_YES.match(w)))


# 2026-11-01: a long yes with conditions ("Yes, start the build ... hold the holdout until ...") failed on a word such as "hold"
# or "not" anywhere in it, so the build was refused and the agent asked for a yes again. A yes now leads the message or names
# the build; only a refusal aimed at building (or a leading no or wait) still refuses.
LEADING_YES = re.compile(r"^\W*(?:yes|yeah|yep|sure|ok(?:ay)?|go ahead|please (?:build|start|go ahead))\b", re.I)
BUILD_YES = re.compile(r"\b(?:start the build|go ahead with the build|build it|start building|build now|start it now)\b", re.I)
# 2026-11-02 (the owner's second Telco session): an order to build a model is the person's yes ("now build model", "make prediction models"),
# and "you decide" is a decision handed over; a question about building is neither.
ORDER_MODEL = re.compile(r"^\W*(?:(?:please|now|ok(?:ay)?|yes|then|just|so|and|alright)[\s,]+)*(?:build|make|create|train|run|start)(?: me| us)?(?: a| the| my| some)?"
                         r"(?: prediction| predictive| churn| ml| machine learning)?(?: model| models| classifier| predictor)\b(?!\s+(?:of|in|from|using|with)\b)", re.I)
NEGATING = re.compile(r"\b(?:not|no|never|cannot|rather|need|until|once|after|before|if|without|neither|nor|tomorrow|next|instead|skip|said|asked|later|when|unless|maybe|perhaps|first|only|but)\b|n't", re.I)
ASKING = re.compile(r"^\W*(?:how|can|could|should|what|why|is|are|do you|would|will)\b", re.I)
DELEGATE = re.compile(r"\b(?:you decide|your call|up to you|whatever you (?:think|prefer|like)|choose for me|you choose|as you see fit|i don'?t mind)\b", re.I)


def orders_model(words: str) -> bool:
    """An order to build a model: the message opens with the verb, is short, and holds no question, no negation and no delay
    (the review of 37.11k: "I won't build a model until ..." and "build a model tomorrow" are not orders)."""
    w = str(words or "")
    return (len(w.split()) <= 9 and bool(ORDER_MODEL.match(w)) and "?" not in w and not ASKING.match(w) and not NEGATING.search(w) and not HOLD.search(w))

PAGE_BUILD = re.compile(r"^\W*create a project from this chat\W*$", re.I)  # the page's "Build this chat as a project" button sends exactly this
REFUSE_AIMED = (r"\b(?:don'?t|do not|not yet|not now|wait|hold off|later|never|cancel|tomorrow|next week|won'?t|can'?t|cannot|rather not|no need)\b"
                r"[^.!?]{0,40}\b(?:build|start|go ahead|create|train|model)|\bno (?:model|models)\b|\bhold off\b")  # a delay or a no aimed at building
REFUSE_BUILD = re.compile(REFUSE_AIMED + r"|^\W*(?:no|not yet|not now|wait)\b", re.I)
REFUSES_BUILDING = re.compile(REFUSE_AIMED, re.I)


def says_yes(words: str) -> bool:
    """The person's yes to building: a leading yes, or the build named, or a yes word with no delay in the message; never a
    refusal aimed at building (the review of 37.6; 2026-11-01)."""
    w = str(words or "")
    if REFUSE_BUILD.search(w):
        return False
    return (bool(PAGE_BUILD.match(w) or LEADING_YES.search(w) or BUILD_YES.search(w) or orders_model(w) or BARE_YES.match(w))
            or (bool(YES_BUILD.search(w)) and not NOT_NOW.search(w)))


NO_MORE = re.compile(r"\b(?:no more questions|stop asking|i (?:told|said) you|i don'?t need|just show|don'?t ask)\b", re.I)


def says_no_more(words: str) -> bool:
    """The person has said they want no more questions (or that they already answered)."""
    return bool(NO_MORE.search(str(words or "")))


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
    personal: set[str] = field(default_factory=set)  # the columns the studio marks as personal: no query, answer or dashboard tile uses them
    cards: list[dict[str, Any]] = field(default_factory=list)  # 2026-11-02: this turn's answers and dashboards, drawn by Home under the message
    wants_figures: bool = False  # the person asked, in this message, for figures of the shared table: no question before they are shown
    queries: int = 0  # the table queries that ran in this turn (at most QUERIES_A_TURN): a refused one is never counted
    query_errors: int = 0  # the table queries refused in this turn (at most QUERY_ERRORS_A_TURN)
    budget_reached: bool = False  # 2026-11-01: three turns with the goal and the data known: propose or start, no more asking
    route_client: Any = None  # 2026-11-01: the cheap second reading of the route (purpose route_check); None: no check
    chip: Optional[str] = None  # 2026-11-04: the kind of help the person picked on Home before writing (a track): the route
    chip_intent: Optional[str] = None  # the draft's chosen track the chip was last read from (a press, not every difference, resets it)
    chip_checked: bool = False  # the one question asked when their words pointed elsewhere (asked once, then their pick decides)
    clear_on_arrival: bool = False  # request is crystal-clear on arrival (goal, subject/target and table known)
    route_told: list[str] = field(default_factory=list)  # the plan's step 1: each route doubt (a move and what it pointed to) told once
    route_seen: Optional[tuple[int, Optional[str]]] = None  # the second reading for the words heard so far: read again on new words only
    route_thread: Any = None  # the second reading under way since the turn's start (route_ahead); never kept in a checkpoint
    only_option: Optional[tuple[str, str]] = None  # this turn: the one option that passed the checks when a propose needed two
    rewriting: bool = False  # the agent is rewriting dropped sentences: no move may run (the review: a question counted twice)
    trace: Any = None  # one tracer for the whole case, so its turns are one conversation in the record
    frame: Any = None  # the shared table itself (never sent to a model; the data tools read it)
    build: Optional[dict[str, Any]] = None  # 37.6: the build the person said yes to (their words), never one without
    group_names: set[str] = field(default_factory=set)  # every group name the case was shown (at most data_tools.MAX_NAMES)
    exposure: list[dict[str, Any]] = field(default_factory=list)  # what the answers looked at (H8): carried to a build
    specialist_client: Any = None  # 37.7: the client specialists are sent with (purpose case_specialist); None: none may be sent
    models: Any = None  # gateway for multi-model no-data understanding & harness
    draft_id: Optional[str] = None  # draft id for gateway telemetry
    no_data_solution: Optional[dict[str, Any]] = None  # latest no-data blueprint for virtual-schema specialist consults
    plan: str = ""  # 2026-11-16: the plan the agent wrote for this person (its numbered steps and closing lines), kept across turns
    no_data_built: bool = False  # the blueprint was built in this turn (once a turn: one pass is a dozen model calls)
    consulted: list[str] = field(default_factory=list)  # 37.7: the specialists already sent on this case
    spent_eur: float = 0.0  # 37.8: what this case's model work has cost so far
    cap_eur: float = field(default_factory=lambda: __import__("dclab_rnd.agents.case_session", fromlist=["cap_eur"]).cap_eur())  # 37.8: the soft cap (DCLAB_CASE_CAP_EUR), raised by the person's yes
    waiting_on_budget: bool = False  # 37.8: the case stopped at its cap and asked
    checkpoint: Any = None  # 37.8: where the state is saved after every turn (None: not saved)
    workspace_home: Any = None  # A1: workspace folder for lead_choice.json switch
    decisions: list[dict[str, Any]] = field(default_factory=list)  # A1-A2: bounded choice and observation log
    decision_effects: dict[str, dict[str, Any]] = field(default_factory=dict)  # A2: durable idempotency table
    decision_revisions: int = 0  # A2: plan revisions performed in this case
    decision_experiments: int = 0  # A2: training-fold comparisons performed in this case
    approvals_pending: bool = False  # A1-A2: pending approval blocks gated execution
    problem_revision: Optional[Any] = None  # A5-A6: active problem summary revision when memory is attached
    project_id: Optional[str] = None  # A6: bound project id when resuming or linked to a project
    problem_id: Optional[str] = None  # A6: bound problem id when linked to a project problem
    authorized_scope: Any = None  # A3-A6: host-resolved AuthorizedScope (never persisted in checkpoint)
    problem_summary: Optional[dict[str, Any]] = None  # A5-A6: latest active versioned resumable summary
    pending_recommendations: list[dict[str, Any]] = field(default_factory=list)  # A6: Advisor recommendations with pinned dependencies

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
    fact_val = f"{state.table['table']}: {state.table['rows']} rows, {len(state.table['columns'])} columns"
    if state.case.data is None or state.case.data.status != "computed" or state.case.data.value != fact_val:  # what code read becomes the case's data fact (by code, computed)
        state.case = C.apply(state.case, "data", C.CaseFact(value=fact_val, status="computed",
                                                         computed=f"the upload's profile ({state.table['table']})"),
                             by="code", why="the table was read")
    out = {"table": state.table["table"], "rows": state.table["rows"], "columns": state.table["columns"]}
    import json

    dumped = json.dumps(out)
    if dumped not in state.computed:
        state.computed.append(dumped)  # what code read: its numbers may be said
    return out


# Questions of the table that ran in one turn, kept by code (the owner's Telco session, 2026-11-02: two was too few to repair a
# refusal and still answer; HAB-002, 2026-11-15: four stopped an analysis of a fall by region, channel and category half way, ten
# times in eight conversations: eight now). A refused question is not counted, but four refusals end the tries.
QUERIES_A_TURN = 8
QUERY_ERRORS_A_TURN = 4


def _data(state: CaseState, name: str, **arguments: Any) -> dict[str, Any]:
    """37.6: one of the guard layer's data tools on the shared table; what it computes may be said to the person."""
    import json

    from . import data_tools

    if state.frame is None:
        return {"error": "No table has been shared: ask the person to share it, or work without numbers."}
    if name == "query":
        if state.queries >= QUERIES_A_TURN:  # CEX-014: queries took over the turns once the table was read by code
            return {"error": f"the table has been asked {state.queries} times this turn: write to the person now with what you found"}
        if state.query_errors >= QUERY_ERRORS_A_TURN:
            return {"error": f"{state.query_errors} questions of the table were refused this turn: tell the person in plain words what did not work "
                             "and what you found without it; do not say you will try again later"}
        arguments["seen"] = state.group_names
        arguments["personal"] = state.personal
    out = data_tools.TOOLS[name](state.frame, **arguments)
    if name == "query":
        if "error" in out:
            state.query_errors += 1
        else:
            state.queries += 1
    if "error" not in out:
        if out.get("exposure"):
            state.exposure.append(out["exposure"])
        state.computed.append(json.dumps(out, default=str))
    return out


def _query(state: CaseState, group_by: Optional[list[str]] = None, measures: Optional[list[dict[str, Any]]] = None,
           filters: Optional[list[dict[str, Any]]] = None, sort: Optional[list[dict[str, Any]]] = None,
           period: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return _data(state, "query", group_by=group_by, measures=measures, filters=filters, sort=sort, period=period)


MAX_TILES = 6  # a dashboard's tiles: a page the person can read


def _worked(state: CaseState, track: str, tool: str) -> None:
    """The figures asked for were shown: that is the work, and the case says so (the route the panel and the record read). Only for a
    person whose words ask for figures and no model, and never over a route already started or picked."""
    route = state.case.route
    moves = route is not None and tool == "dashboard" and route.by == "agent" and route.tool == "query"  # a board after an answer: the Refresh button reads this
    if state.chip and track != state.chip:
        return  # 2026-11-04: the person picked another kind of help on Home; figures shown on the way do not change it
    figures_wanted = state.wants_figures or (bool(state.table) and C.wants_figures(state.case.words) and not _model_work(state))
    if (route is None or moves) and not state.case.chosen and figures_wanted:
        state.case = C.apply(state.case, "route", C.CaseRoute(track=track, tool=tool, by="agent"), by="agent", why="the figures asked for were shown")


def _model_work(state: CaseState) -> bool:
    """The person's latest message is about a model (an order to build, a delegation, a model's words) and does not ask for figures, or a
    build is the work and this message does not ask for figures: no report is drawn again (the owner's second Telco session)."""
    latest = state.case.words[-1] if state.case.words else ""
    ordered = orders_model(latest) or bool(DELEGATE.search(latest))  # an order or a delegation, not a mention of risk or a score (the review of 37.11k)
    building = state.case.route is not None and state.case.route.tool in BUILDS
    return (ordered or building) and not state.wants_figures


NOT_NOW_CARD = {"error": "a model is the work now: tell the person what happens next in words; draw a report or a dashboard only when they ask for numbers"}


def _answer(state: CaseState, group_by: Optional[list[str]] = None, measures: Optional[list[dict[str, Any]]] = None,
            filters: Optional[list[dict[str, Any]]] = None, sort: Optional[list[dict[str, Any]]] = None,
            period: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Ask the table and show the person the answer: a chart and its numbers, drawn by code from this very result (the owner's
    Telco session, 2026-11-02: a report was promised and nothing was ever shown). The numbers come back to the agent too."""
    if _model_work(state):
        return NOT_NOW_CARD
    kept: list[Any] = []
    out = _data(state, "query", group_by=group_by, measures=measures, filters=filters, sort=sort, keep=kept, period=period)
    if "error" in out or not kept:
        return out
    state.cards.append({"kind": "answer", "plan": kept[-1].model_dump(mode="json")})
    _worked(state, "analysis", "query")
    return {**out, "shown": "a chart and the numbers are drawn under your message; now say in two or three sentences what they show, "
                            "with the numbers exactly as as_said gives them, and do not draw anything yourself"}


def _dashboard(state: CaseState, tiles: Optional[list[dict[str, Any]]] = None, title: Optional[str] = None) -> dict[str, Any]:
    """A dashboard of up to six tiles, each a question of the table; code runs every tile and draws its chart. The agent sees no
    number from it: ask the table (query) for the numbers it will say."""
    from . import data_tools

    if state.frame is None:
        return {"error": "No table has been shared: ask the person to share it first."}
    if _model_work(state):
        return NOT_NOW_CARD
    if not tiles or len(tiles) > MAX_TILES:
        return {"error": f"a dashboard has one to {MAX_TILES} tiles, each {{group_by, measures}}: for example one tile per segment column"}
    plans = []
    for n, tile in enumerate(tiles, 1):
        plan, _, refused = data_tools.plan_of(state.frame, (tile or {}).get("group_by"), (tile or {}).get("measures"), (tile or {}).get("filters"),
                                              (tile or {}).get("sort"), state.personal, (tile or {}).get("period"))
        if plan is None:
            return {"error": f"tile {n}: {refused['error']}", **({"columns": refused["columns"]} if "columns" in refused else {})}
        plans.append(plan.model_dump(mode="json"))
    state.cards.append({"kind": "dashboard", "title": str(title or "Your dashboard")[:120], "plans": plans})
    _worked(state, "reporting", "dashboard")
    return {"shown": f"a dashboard of {len(plans)} tiles is drawn under your message; say which numbers it holds, and quote a figure "
                     "only if a query gave it to you"}


def _check_leakage(state: CaseState, outcome: str) -> dict[str, Any]:
    return _data(state, "leakage", outcome=outcome)


def _preview_step(state: CaseState, kind: str, columns: Optional[list[str]] = None, filters: Optional[list[dict[str, Any]]] = None,
                  keys: Optional[list[str]] = None) -> dict[str, Any]:
    return _data(state, "preview_step", kind=kind, columns=columns, filters=filters, keys=keys)


# The review of stage 2 (2026-11-15): "any build word anywhere beside any question mark" read "I started the project. Which
# column says a customer left?" as a question about building. The build is named inside the question's own sentence.
BUILD_QUESTION = re.compile(r"\b(?:build|train)\b|\b(?:create|start|make|set up|go ahead with)\b[^.?!]{0,40}\b(?:model|project|build)\b", re.I)
# a message that is nothing but a go-ahead: "Proceed.", "ok, go ahead", "run it please". Never one that asks for something
# ("show me the churn rate first", "proceed with the analysis only", "you decide")
ONLY_GO = re.compile(r"^\W*(?:(?:ok(?:ay)?|yes|yeah|sure|alright|please|now|then)[\s,]+)*(?:proceed|go ahead|go on|go|run it|do it|start it|build it|let'?s go)"
                     r"(?:[\s,]+(?:please|now|then))*\W*$", re.I)


def _asked_to_build(state: CaseState) -> bool:
    """The agent's last message before this one asked about building (a model, a project): one of its questions names the build."""
    before = [t for t in state.turns[:-1] if t.get("role") == "assistant"]
    last = str(before[-1].get("content") or "") if before else ""
    return any(s.rstrip().endswith("?") and BUILD_QUESTION.search(s) and not re.search(r"\bor\b", s, re.I)  # "build it, or see the rate first?" is a choice, not a yes-or-no
               for s in re.split(r"(?<=[.?!])\s+", last))


def _request_build(state: CaseState, quote: Optional[str] = None, outcome: Optional[str] = None) -> dict[str, Any]:
    """A build only on a project route, a ready case (or the person's own choice), and the person's yes in their latest words (an
    order to build a model counts, and so does a bare yes: the quote may be the whole of a short message). ``outcome``: the column
    the model learns, accepted only when the table has it and the person's words (or an assumption they confirmed) name it."""
    route = state.case.route
    if route is None or route.tool not in BUILDS:
        return {"error": "a build starts a project: start a prediction or segmentation first (start, quoting the person's latest words)"}
    r = C.readiness(state.case)
    latest = state.case.words[-1] if state.case.words else ""
    if not r.ready and not state.case.chosen and not orders_model(latest):  # an order to build a model needs no interview first (the owner's Telco session)
        return {"error": "the case is not ready: " + "; ".join(r.reasons)}
    # Stage 2 of 2026-11-15: the yes is read here, from the person's latest message itself. Before, the model had to pass it
    # as a word-for-word quote: a quote with "..." or one changed word was refused with the same words as no yes at all, 18
    # times in the day's runs. The rule is unchanged (no build without their yes); each refusal now says which case it is.
    if latest.startswith("I pick:"):
        return {"error": "the person picked an option; a pick is not a yes to building. Tell them in a sentence or two what will be "
                         "built and from which columns, then ask once whether to build it (ask). Request the build after their yes."}
    bare = bool(BARE_YES.match(latest))  # "yes", "ok", "go": a yes only to the question it answers
    if bare and not orders_model(latest) and not _asked_to_build(state):  # a yes to a question about something else is not a yes to building
        return {"error": "their yes answers your last question, which was not about building: ask whether to build, then request it on their yes"}
    if not (says_yes(latest) or (ONLY_GO.match(latest) and _asked_to_build(state))):  # "proceed" or "run it", and nothing else, after you asked about building is their yes
        return {"error": "their latest message holds no yes to building. Say what will be built, ask once whether to build it (ask), and "
                         "request the build after their yes. Never ask them to type a particular phrase."}
    if not (quote and (C.quote_ok(quote, quote, [latest]) or C.whole_short_message(quote, [latest]))):
        quote = latest if len(latest) <= C.MAX_TEXT else latest[:C.MAX_TEXT].rsplit(" ", 1)[0]  # their own message is the record of their yes
    column = None
    if outcome:
        names = [str(c.get("name")) for c in (state.table or {}).get("columns") or []]
        column = next((n for n in names if n.casefold() == str(outcome).casefold()), None)
        if column is None:
            return {"error": f"the table has no column {outcome!r}: the columns are {', '.join(names[:30])}"}
        if column not in C.grounded_columns(state.case, [column]):
            return {"error": f"the person has not named {column}: ask which column is the outcome, or leave outcome out"}
    state.build = {"tool": route.tool, "quote": quote, **({"outcome": column} if column else {})}
    try:  # the case file keeps the person's yes with their words (v2)
        state.case = C.apply(state.case, "build", C.CaseBuild(tool=route.tool, quote=quote), by="person", why="the person said yes to the build")
    except C.CaseRefused as error:
        return state.refuse(error)
    # the live check of 37.11 (2026-11-01): "nothing is built yet" was said word for word while Home started the build
    return {"recorded": route.tool, "next": "tell the person in one short sentence that DCLab is checking the remaining build "
                                            "requirements now; give no result or number before it has run"}


def _consult(state: CaseState, specialist: str, question: str) -> dict[str, Any]:
    """37.7: send a specialist with the case; its findings go into the case file for the lead agent, never to the person."""
    from . import context as ctx_
    from . import specialists as S

    if state.specialist_client is None:
        return {"error": "no specialist can be sent in this session"}
    nd_sol = state.no_data_solution or next(
        (c.get("solution") for c in reversed(state.cards) if c.get("kind") == "no_data_solution" and isinstance(c.get("solution"), dict)),
        None,
    )
    if state.table is not None:
        nd_sol = None  # a real table is shared: its columns are what a specialist reads, and its own rules decide who may be sent
    nd_ds = (nd_sol or {}).get("data_science") or {}
    vtable_specs = list(nd_ds.get("virtual_table_specs") or [])
    vcols = [
        str(col)
        for cols in (nd_ds.get("virtual_tables") or {}).values()
        for col in (cols or [])
    ]
    why = S.allowed(
        specialist,
        state.case,
        state.table,
        state.consulted,
        virtual_columns=vcols or None,
        has_blueprint=bool(nd_sol),
    )
    if why:
        return {"error": why}
    plain = [c for c in state.computed if '"groups_not_shown"' not in c]  # never a query's answers
    case_text = ctx_.case_text(state.case)
    named = any(n and (n in question or n in case_text) for n in state.group_names)  # the review: names reach it in words too
    try:
        state.specialist_client.request_cells = named
    except AttributeError:
        pass
    if state.table is not None:
        t_txt = ctx_.table_text(state.table)
    elif vtable_specs:
        pred_mom = str((((nd_sol or {}).get("understanding") or {}).get("fields") or {}).get("prediction_moment", {}).get("value") or nd_ds.get("prediction_time") or "")
        t_txt = S.virtual_table_text(vtable_specs, pred_mom)
    else:
        t_txt = ctx_.table_text(state.table)
    out = S.consult(state.specialist_client, specialist, question, case_text, t_txt, plain)
    state.consulted.append(specialist)
    if "error" in out:
        return out
    findings = [C.CaseFinding(by=specialist, **f) for f in out["findings"]]
    state.case = C.apply(state.case, "findings", findings, by=f"specialist:{specialist}", why=question[:200])
    return {"findings": out["findings"], "next": "use what helps; ask the person about any question it raised"}


MAX_NOTES = 8  # facts one note call may hold


def _note(state: CaseState, field: Optional[str] = None, value: Optional[str] = None, status: Optional[str] = None, quote: Optional[str] = None,
          notes: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """One fact, or several in one call (stage 2 of 2026-11-15: 188 notes against 117 table questions in the day's runs; each
    was a step of the turn). Each fact is checked by the case file as before; one that is refused does not stop the others."""
    if notes is None:
        if status not in ("said", "assumed"):
            return {"error": "status is said (the person's words, quoted) or assumed (your supposition, shown to them)"}
        try:
            state.case = C.apply(state.case, field, C.CaseFact(value=value, status=status, quote=quote if status == "said" else None),
                                 by="agent", why="noted by the lead agent")
        except (C.CaseRefused, ValueError) as error:
            return state.refuse(error)
        return {"noted": field, "readiness": C.readiness(state.case).model_dump(mode="json")}
    if not isinstance(notes, list) or not notes or len(notes) > MAX_NOTES:
        return {"error": f"notes is a list of one to {MAX_NOTES} facts, each with field, value, status and (for said) quote"}
    noted, refused = [], {}
    for item in notes:
        item = item if isinstance(item, dict) else {}
        name = str(item.get("field") or "")
        if item.get("status") not in ("said", "assumed") or name not in (*C.SINGLE, *C.LISTS) or not str(item.get("value") or "").strip():
            refused[name or "?"] = "each fact needs a field of the case file, a value, and status said or assumed"
            continue
        try:
            state.case = C.apply(state.case, name, C.CaseFact(value=str(item["value"]), status=item["status"],
                                                             quote=item.get("quote") if item["status"] == "said" else None),
                                 by="agent", why="noted by the lead agent")
            noted.append(name)
        except (C.CaseRefused, ValueError) as error:
            refused[name] = str(state.refuse(error).get("error") or error)[:300]
    out: dict[str, Any] = {"noted": noted, "readiness": C.readiness(state.case).model_dump(mode="json")}
    if refused:
        out["refused"] = refused
    return out if noted else {"error": "no fact was recorded", "refused": refused}


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
    if state.wants_figures and state.queries == 0 and not state.cards and state.query_errors < QUERY_ERRORS_A_TURN and not any(u.blocks == "all" for u in state.case.unknowns):
        return {"error": "the person asked for figures of their table: ask the table now (query) and write what it shows; "
                         "ask a question only after the figures are in your message, and only one"}
    if state.go_ahead and DECISION_Q.search(str(question)):
        return {"error": "the person told you to go ahead or decided for you: do it now, state what you assumed about what they will do with "
                         "it, and ask nothing about it"}
    if state.go_ahead and CONFIRMS.search(str(question)):
        # Stage 2 (a dead end, verified): "proceed" is a go-ahead and no yes to building, so request_build said "ask whether to
        # build" and this said "do not ask". The one question about building is allowed when a build is the work, their words
        # hold no yes and it was not asked yet.
        latest = state.case.words[-1] if state.case.words else ""
        route = state.case.route
        building = route is not None and route.tool in BUILDS and bool(BUILD_QUESTION.search(str(question)))
        if not (building and not says_yes(latest) and not _asked_to_build(state)):
            return {"error": "the person told you to go ahead: act now (start the work, propose options with what is open stated as "
                             "assumptions, or request_build when they said yes to building) and tell them what you did; do not ask them "
                             "to confirm again. Ask only about something the work cannot start without."}
    state.case = C.asked(state.case)
    state.said.append(str(question).strip())
    state.asking = True
    return {"asked": True}


CHIP_NEEDS_TABLE = ("prediction", "analysis", "reporting", "preparation")  # the kinds of help that run on a shared table
# what a start is followed by (stage 2 of 2026-11-15: "a few sentences, then stop" left no room for the plan the prompt asks for)
STARTED_NEXT = ("now tell the person what you understood, your approach and the plan (three to five steps, in order, each with what it "
                "checks), then stop. No number that no tool returned.")
# with the think-first switch on the message is written before the move (the review of stage 4: "now write that message" made
# the model write its plan a second time, and the limit then cut the copy)
WRITTEN_NEXT = ("the person sees this under your message. If you already wrote your message in this turn (the problem, what needs "
                "care, your approach, the plan), it stands: add one or two sentences on {what}, then stop; do not write the plan "
                "again. If you have not written it yet, write it now, then stop. No number that no tool returned.")
PAIRS = ", ".join(f"{track}/{tool}" for tool, track in C.TRACK_OF_TOOL.items())  # every kind of help with its tool, as a refusal names them


def changed_mind(state: CaseState) -> Optional[str]:
    """The kind of help the person's latest message asks for instead of their chip, from their second message on (the first
    message is the one the chip was picked with: a conflict there is asked once, never taken as a change of mind), only before
    any work started and only when that message says it is a change (instead, actually, rather ...)."""
    if not state.chip or len(state.case.words) < 2 or state.case.route is not None:
        return None  # once work has started, a change goes through the route's own quote rule
    latest = state.case.words[-1]
    if not C.CHANGE_WORDS.search(latest):
        return None  # "first, what was churn last month?" or "also a chart" is part of the same work, never a change of mind
    return C.chip_conflict(state.chip, [latest])


def chip_note(state: CaseState) -> Optional[str]:
    """What the agent is told about a chip: the route, and the one check when the words point elsewhere. Once work has
    started the route's own quote rule governs: no note (the third review, 2026-11-04)."""
    if not state.chip or state.case.chosen or state.case.route is not None:
        return None
    other = C.chip_conflict(state.chip, state.case.words)
    label = C.CHIPS.get(state.chip, state.chip)
    latest = changed_mind(state)
    if latest:
        return (f"the person picked {label} ({state.chip}) on Home, but their latest words ask for {latest}: start {latest} with their "
                "words as the quote, or propose the two once if unsure")
    if state.chip == "advice" and data_work(state) and not state.chip_checked:
        return "the person picked Learn (advice) on Home, but their words ask for work on data: propose advice and that work once"
    if other and not state.chip_checked:
        return (f"the person picked {label} ({state.chip}) on Home, but their words point to {other}: propose exactly two options, one "
                f"{state.chip} and one {other}, and let them pick; ask nothing else")
    return f"the person picked {label} ({state.chip}) on Home: that is the route. Start it now; never ask which kind of help they want"


def _propose(state: CaseState, options: list[dict[str, Any]]) -> dict[str, Any]:
    options = [o for o in options or [] if isinstance(o, dict)]
    unknown = sorted({str(o.get("tool")) for o in options if o.get("tool") not in C.TRACK_OF_TOOL})
    if unknown:
        return {"error": f"DCLab has no tool {', '.join(unknown)}. Each option names one of: " + PAIRS}
    # Stage 2: the kind of work follows from the tool (they are one to one); a pair that did not match was refused with no
    # hint of the right one ("analysis/dashboard"). Derived before anything reads it, so a chip is checked on the real kind.
    options = [{**o, "track": C.TRACK_OF_TOOL[o["tool"]]} for o in options]
    r = C.readiness(state.case)
    chip_other = C.chip_conflict(state.chip, state.case.words) if state.chip and not state.case.chosen and state.case.route is None else None
    if state.chip and not state.case.chosen:  # 2026-11-04: a chip is the person's choice; options only inside it, or the one check
        latest_other = changed_mind(state)
        advice_data = {"analysis", "prediction", "reporting", "preparation"} if state.chip == "advice" and data_work(state) else set()
        allowed = {state.chip} | ({chip_other} if chip_other and not state.chip_checked else set()) | ({latest_other} if latest_other else set()) | advice_data
        tracks = {str(o.get("track")) for o in options or []}
        if not tracks <= allowed:
            return {"error": f"the person picked {C.CHIPS.get(state.chip, state.chip)}: offer only {', '.join(sorted(allowed))}"}
    # Stage 2 (a dead end, verified): after "go ahead" with the decision or the user unsaid, a start was refused as not ready,
    # this said "not yet, ask", and a question was refused as "do not ask, act now". A go-ahead opens the options, with what
    # is open stated as assumptions: the person still chooses.
    if (not r.may_propose and not state.case.chosen and not state.wants_options and not state.budget_reached and not state.chip
            and not state.clear_on_arrival and not state.go_ahead):
        return {"error": "not yet: " + "; ".join(r.reasons) + ". Ask about what is missing, or propose once the ceiling is reached."}
    try:
        offered = [C.CaseOption(**{k: v for k, v in o.items() if k in C.CaseOption.model_fields}) for o in options]
    except ValueError as error:
        return state.refuse(error)
    columns = {c["name"] for c in (state.table or {}).get("columns") or []}
    kept, refused = C.check_options(offered, state.case, columns=columns, computed=state.computed)
    if len(kept) < 2:
        # HAB-024 (Nemotron 3 Ultra, the delivery-routes person, 2026-11-16): one option passed, this said "start it instead",
        # and the start was refused as not ready, twice each way. The one option that passed is remembered for this turn: after
        # a go-ahead its start is allowed, with what is open stated as assumptions.
        state.only_option = (kept[0].track, kept[0].tool) if len(kept) == 1 else None
        only = f" The one that passed is {kept[0].track}/{kept[0].tool}: start it (start) instead of proposing." if kept else \
            " When only one kind of help fits, start it (start) instead of proposing."
        return {"error": "fewer than two options passed the checks. Options must be of different kinds (" + PAIRS + ")." + only, "refused": refused}
    doubt = None if (state.case.chosen or state.chip) else route_doubt(state, "propose", {o.track for o in kept})
    if doubt:
        return {"error": doubt}
    if state.chip and {o.track for o in kept} != {state.chip}:
        state.chip_checked = True  # the one question about the chip is this choice: asked once
    state.case = C.apply(state.case, "options", kept, by="agent", why="proposed")
    state.shown = kept
    return {"shown": [o.title for o in kept], "refused": refused,
            "next": WRITTEN_NEXT.format(what="the options") if think_first() else
                    "the person sees these under your message: now write that message. When their brief is enough, write what you "
                    "understood of the problem, your approach and a plan of three to five steps, then lead into the options; otherwise a "
                    "sentence or two. Then stop."}


def _start(state: CaseState, track: str, tool: str, quote: Optional[str] = None) -> dict[str, Any]:
    if tool not in C.TRACK_OF_TOOL:
        return {"error": f"DCLab has no tool {tool!r}. A start names one of: " + PAIRS}
    if C.TRACK_OF_TOOL.get(tool) != track:
        return {"error": f"DCLab cannot run {track}/{tool}: {tool} belongs to {C.TRACK_OF_TOOL[tool]}. The pairs are: " + PAIRS}
    route = state.case.route
    if route is not None:  # the review of 37.3: a start after the person's pick rewrote their route as the agent's
        if (route.track, route.tool) == (track, tool):
            return {"started": f"{track}/{tool}", "next": "already under way: tell the person what happens next (the steps, in order), then stop"}
        latest = state.case.words[-1:] if state.case.words else []
        if not (quote and C.quote_ok(quote, quote, latest)):
            return {"error": f"the work has started as {route.track}/{route.tool}. Change it only when the person asks to in their "
                             "latest message: pass their words as quote. A first step on the way (a base rate) is part of the same work."}
        state.case = C.apply(state.case, "route", C.CaseRoute(track=track, tool=tool, by="person"), by="person", why=f"the person asked: {quote}")
        return {"started": f"{track}/{tool}", "next": "tell the person the work changed as they asked"}
    if state.chip and not state.case.chosen:  # 2026-11-04: the person picked the kind of help on Home: it is the route
        latest = changed_mind(state)
        if latest and track == latest and quote and C.quote_ok(quote, quote, state.case.words[-1:]):
            state.chip, state.chip_checked = track, True  # they changed their mind in their latest words: that is their choice now
        elif state.chip == "advice" and track == "advice" and data_work(state):
            return {"error": "they picked Learn, but their words ask for work on data: propose the two once (advice and the work they "
                             "describe) and let them pick"}
    if state.chip and not state.case.chosen:
        if track in CHIP_NEEDS_TABLE and track == state.chip and state.table is None:
            from ..draft import no_data
            if _no_data_v2() and _no_table_said(state):
                return {"error": NO_TABLE_DONE if state.no_data_built else NO_TABLE_GO}
            if no_data.detect_no_data_stance(" ".join(state.case.words)).get("matched") and no_data.blueprint_fit(" ".join(state.case.words))["fits"]:
                return {"error": "the person has no data yet or prefers not to upload it: call design_solution_without_data to give the full 3-layer Business, Data Science, and ML blueprint instead of demanding an upload."}
            return {"error": f"they want {C.CHIPS.get(state.chip, state.chip)} ({state.chip}): ask them to share the table it runs on, then start it"}
        if track != state.chip:
            return {"error": f"the person picked {C.CHIPS.get(state.chip, state.chip)} ({state.chip}): start {state.chip}, or, when their words "
                             "point elsewhere, propose the two kinds of help once"}
        other = C.chip_conflict(state.chip, state.case.words)
        if other and not state.chip_checked:
            return {"error": f"their words point to {other}, not the {C.CHIPS.get(state.chip, state.chip)} they picked: propose the two once "
                             f"({state.chip} and {other}) and let them pick"}
        state.case = C.apply(state.case, "route", C.CaseRoute(track=track, tool=tool, by="person"), by="person",
                             why=f"the person picked {C.CHIPS.get(state.chip, state.chip)} on Home")
        if tool in ("query", "dashboard"):
            return {"started": f"{track}/{tool}", "next": "nothing is shown yet: ask the table (query), show it with answer or dashboard, then "
                                                          "say what it shows; do not say anything is ready before a tool returned it"}
        return {"started": f"{track}/{tool}", "next": WRITTEN_NEXT.format(what="what happens next") if think_first() else STARTED_NEXT}
    r = C.readiness(state.case)
    if track in NO_DATA_WORK and not state.case.chosen and not r.ready and not state.budget_reached and data_work(state):
        return {"error": "the person's need points to data work (a shared table, or a list, report, model or cleaning): advice "
                         "answers a question with no work on data. Propose or start the work their need calls for."}
    if track in CHIP_NEEDS_TABLE and state.table is None:
        from ..draft import no_data
        if _no_data_v2() and _no_table_said(state):
            return {"error": NO_TABLE_DONE if state.no_data_built else NO_TABLE_GO}
        if track == "prediction" and no_data.detect_no_data_stance(" ".join(state.case.words)).get("matched") and no_data.blueprint_fit(" ".join(state.case.words))["fits"]:
            return {"error": "the person has no data yet or prefers not to upload it: call design_solution_without_data first to give their full 3-layer Business, Data Science, and ML blueprint."}
        return {"error": f"they want {track}: ask them to share or upload the table it runs on, then start it"}
    # a clear person is proposed to at once (readiness), never started on before the case is ready: "send me the customers who
    # churned ... call list by friday" is both a go-ahead and a named analysis, on a real-prediction trap (the second review of step 1)
    # Stage 2 (a dead end, verified): "Build the model" was refused here as not ready while request_build said "start a
    # prediction first". An order to build a model needs no interview first (request_build already reads it so): it starts the
    # project it names; the build itself still waits for the checks request_build makes.
    ordered = tool in BUILDS and orders_model(state.case.words[-1] if state.case.words else "")
    sole = state.go_ahead and state.only_option == (track, tool)  # the one option that passed the checks, after the person said go ahead
    if not state.case.chosen and not r.ready and not state.budget_reached and track not in NO_DATA_WORK and not ordered and not sole:
        return {"error": "the person has not chosen and the case is not ready: " + "; ".join(r.reasons)
                         + ". Propose now (two kinds of help, what is open stated as assumptions), or ask the one thing that is missing."}
    doubt = None if state.case.chosen else route_doubt(state, "start", {track})
    if doubt:
        return {"error": doubt}
    state.case = C.apply(state.case, "route", C.CaseRoute(track=track, tool=tool, by="agent"), by="agent", why="work started")
    if tool in ("query", "dashboard"):  # 2026-11-02: "work started" showed nothing; the figures are shown by this very turn
        return {"started": f"{track}/{tool}", "next": "nothing is shown yet: ask the table (query), show it with answer or dashboard, then "
                                                      "say what it shows; do not say anything is ready before a tool returned it"}
    return {"started": f"{track}/{tool}", "next": WRITTEN_NEXT.format(what="what happens next") if think_first() else STARTED_NEXT}


NO_TABLE_GO = ("the person has no data to share (their own words say so): do not ask for a table. Call design_solution_without_data now; it "
               "writes what they need without data.")


NO_TABLE_DONE = ("the person has no data to share, and the solution without data was already handled in this turn: do not ask for a "
                 "table and do not start this work; write to the person now")


def _no_data_v2() -> bool:
    from ..nodata import product

    return product.enabled()


def _no_table_said(state: "CaseState") -> bool:
    """The person's own words say they have no table to share (none yet, private, not ready), read by the rebuilt design's
    patterns in English and Persian, or by the first design's."""
    from ..draft import no_data
    from ..nodata import stance

    words = " ".join(state.case.words)
    latest = str(state.case.words[-1]) if state.case.words else ""
    if state.table is not None or stance.has_now(latest):  # they have it now, or are about to share it: asking for it is right again
        return False
    said = stance.NOT_A_STANCE.sub(" ", words)  # "without data leakage" and "no data scientist" say nothing about having a table
    return bool(stance.read(said)["no_table"] or no_data.detect_no_data_stance(said).get("matched"))


NO_SOLUTION_NEXT = ("No solution could be written by the tool this time. Answer the person yourself, now, in your own words: what the "
                    "decision is, the options and what each costs and gives, and a short plan they can start without data. State no "
                    "number they did not give. Do not ask for a file, and do not call this tool again in this turn.")


NO_DATA_TOOL_V2 = (
    "When no table is shared and the person has no data yet, or cannot or will not share it: write the solution they need without "
    "it, in this same turn. It reads their own words and writes what fits: a business answer when they face a decision, what to "
    "record when they ask that, a full solution with a model when they want one, or the questions to ask before any made-up data. "
    "Never ask for an upload again, and never say you cannot go on without data. No data is created: made-up data needs the "
    "person's answers on how it should look and their yes to a design first.")


def _design_v2(state: CaseState, nodata_v2: Any) -> dict[str, Any]:
    """The rebuilt no-data design for the lead agent: the person's own words in, the solution as a card out. Every
    number in it is theirs or a sum code checked; those sums become sources of this turn's reply."""
    words = " ".join(state.case.words).strip()
    if not words:
        return {"error": "the person has not described their problem yet: ask what they need, in one question"}
    left = float(state.cap_eur or 0.5) - float(state.spent_eur or 0.0)
    if left <= 0:
        return {"error": "this case has used its model budget: tell the person what you have, with no new solution"}
    try:
        sol = nodata_v2.solve_for_product(words, gateway=state.models, draft_id=state.draft_id, cap_eur=min(0.12, left))
    except Exception:  # noqa: BLE001 - the person is still answered: the lead model writes it itself
        state.no_data_built = True
        return {"solution_ready": False, "next": NO_SOLUTION_NEXT}
    state.no_data_built = True
    state.spent_eur += float(sol.get("cost_eur") or (sol.get("harness_report") or {}).get("cost_eur") or 0.0)
    if nodata_v2.is_note(sol):  # no model wrote one and the rules see nothing to predict: no card, never an empty "solution"
        return {"solution_ready": False, "next": NO_SOLUTION_NEXT}
    state.no_data_solution = sol
    for item in sol.get("as_said") or []:
        if item not in state.computed:
            state.computed.append(item)
    state.cards.append({"kind": "no_data_solution", "solution": sol})
    if state.case.data is None:
        try:
            state.case = C.apply(state.case, "data", C.CaseFact(value="No table is shared; a solution was designed without data.", status="assumed"),
                                 by="agent", why="solution designed without data")
        except (C.CaseRefused, ValueError):
            pass
    card = sol.get("card") or {}
    if sol.get("version") != "nodata_v2":  # the first design's rules answered (no model wrote one): its card, its placeholders said
        return {"solution_ready": True, "written_by": "rules", "family": sol.get("family"),
                "assumed_inputs": list((sol.get("economics") or {}).get("assumed") or []),
                "next": ("A blueprint written by DCLab's rules is now shown to the person as a card (no model was available to write "
                         "one for them). Write a short reply (under 120 words). State a number only if the person said it: the inputs "
                         "listed in assumed_inputs are DCLab's placeholders, so say they are placeholders and ask for the real ones. "
                         "Do not ask for a file.")}
    return {"solution_ready": True, "need": sol.get("need"), "parts": [p["title"] for p in card.get("parts") or []],
            "questions_for_the_person": list(card.get("questions") or [])[:3], "written_by": sol.get("mode"),
            "next": ("The solution is now shown to the person as a card: do not repeat it. Write a short reply (under 120 words): "
                     "say in one or two sentences what the card holds and what to do first, then ask at most one of the questions "
                     "above. State no number the person did not give. Do not ask for a file. If they want made-up data, ask how "
                     "it should look before anything is designed.")}


def _design_solution_without_data(
    state: CaseState,
    problem: Optional[str] = None,
    monthly_volume: Optional[float] = None,
    base_rate: Optional[float] = None,
    action_cost: Optional[float] = None,
    value_saved: Optional[float] = None,
    action_success_rate: Optional[float] = None,
    target_hint: Optional[str] = None,
) -> dict[str, Any]:
    """Build an honest 3-layer Business + Data Science + ML blueprint when no dataset is shared yet."""
    from ..draft import no_data

    if state.table is not None:
        return {"error": "a table is shared: work on it (read it, query it, propose or start). This tool is for a person with no data."}
    if state.no_data_built:
        return {"error": "the blueprint was already built in this turn: write to the person now"}
    from ..nodata import product as nodata_v2

    if nodata_v2.enabled():  # the rebuilt design (2026-11-16), behind DCLAB_NO_DATA_V2: any need, not only a prediction
        return _design_v2(state, nodata_v2)
    # the person's own words decide: the problem, whether it is a prediction at all, and every number (the model's wording
    # of the problem is never a source of numbers: the review of 22ad1bf8)
    problem_text = " ".join(state.case.words).strip()
    if not problem_text:
        return {"error": "describe the person's problem first before building a no-data solution blueprint"}
    fit = no_data.blueprint_fit(problem_text)
    if not fit["fits"]:
        return {"error": f"no model blueprint fits here: {fit['why']}. Do not show a model plan. Answer what they asked (propose "
                         "or start the work it calls for), or ask which outcome they want predicted."}
    left = float(state.cap_eur or 0.5) - float(state.spent_eur or 0.0)
    if left <= 0:
        return {"error": "this case has used its model budget: tell the person what you have, with no new blueprint"}
    given = {"monthly_volume": monthly_volume, "base_rate": base_rate, "action_cost": action_cost, "value_saved": value_saved,
             "action_success_rate": action_success_rate}
    overrides = {k: v for k, v in given.items() if v is not None and _said_number(v, problem_text, share=k in ("base_rate", "action_success_rate"))}
    ignored = sorted(k for k, v in given.items() if v is not None and k not in overrides)
    case_facts = {}
    for fname in (*C.SINGLE, *C.LISTS):
        val = getattr(state.case, fname, None)
        if isinstance(val, list):
            case_facts[fname] = [v.model_dump(mode="json") if hasattr(v, "model_dump") else v for v in val]
        elif val is not None:
            case_facts[fname] = val.model_dump(mode="json") if hasattr(val, "model_dump") else val
    stance = no_data.detect_no_data_stance(" ".join(state.case.words))
    try:
        overrides = no_data.checked_overrides(overrides)
    except ValueError as error:
        return {"error": str(error)}
    sol = no_data.solve_without_data(
        problem_text,
        case_facts=case_facts,
        economics_overrides=overrides or None,
        withholding_privacy=bool(stance.get("withholding_privacy")),
        gateway=state.models,
        models=state.models,
        draft_id=state.draft_id,
        cap_eur=min(0.25, left),  # never more than the case has left
        target_hint=target_hint,
    )
    state.no_data_built = True
    state.no_data_solution = sol
    state.spent_eur += float((sol.get("harness_report") or {}).get("cost_eur") or 0.0)
    for item in sol.get("as_said") or []:
        if item not in state.computed:
            state.computed.append(item)
    state.cards.append({"kind": "no_data_solution", "solution": sol})
    if state.case.data is None:
        try:
            state.case = C.apply(
                state.case,
                "data",
                C.CaseFact(value="No uploaded table yet; 3-layer pre-data solution blueprint created.", status="assumed"),
                by="agent",
                why="pre-data solution design",
            )
        except (C.CaseRefused, ValueError):
            pass
    return {
        "blueprint_ready": True,
        "family": sol["family"],
        "assumed_inputs": list(sol["economics"].get("assumed") or []),
        **({"ignored_numbers": ignored, "why_ignored": "the person did not say these numbers: they are not used"} if ignored else {}),
        "understanding": sol.get("understanding"),
        "economics": sol["economics"],
        "virtual_tables": sol["data_science"]["virtual_tables"],
        "ml_architecture": sol["ml"]["architecture"],
        "debate": sol["ml"].get("debate"),
        "harness_report": sol.get("harness_report"),
        "simulation_questions": sol["simulation"]["questions"],
        "next": (
            "A 3-layer Business, Data Science, and ML blueprint card is now shown to the person. Write a short, helpful reply "
            "(<180 words). State a number only if the person said it: the inputs listed in assumed_inputs are DCLab's placeholders, "
            "so say they are placeholders and ask for the real ones; never quote a figure computed from them. If they "
            "might want synthetic data, ask them how many rows, columns, and outcome rate they want BEFORE offering to design/simulate it."
        ),
    }


def _said_number(value: Any, words: str, share: bool = False) -> bool:
    """Whether the person's own words hold this number (a share also as its percent)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    forms = {f"{v:g}", f"{v:,.0f}" if v == int(v) else f"{v:,.2f}"}
    if v == int(v):
        forms.add(str(int(v)))
    if share:
        forms |= {f"{v * 100:g}%", f"{v * 100:g} %", f"{v * 100:g} percent"}
    text = str(words or "").translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩٬٫٪", "01234567890123456789,.%"))
    return any(re.search(r"(?<![\d.,])" + re.escape(form) + r"(?![\d])", text) for form in forms)


MOVES = {"consult": "case.consult", "request_build": "case.build", "note": "case.note", "add_unknown": "case.unknown", "resolve": "case.resolve", "confirm_assumption": "case.confirm",
         "ask": "case.ask", "propose": "case.propose", "start": "case.start", "design_solution_without_data": "case.no_data_solution"}


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
            return {"error": "no tool runs in this reply: write your message to the person now, in words, with what you have"}
        if context.case.route is not None and tool.move == "case.propose":  # new options after the start flipped a route (CEX-007)
            return {"error": "the work has started: tell the person what happens next, or ask what they want changed"}
        return proceed()


def register(registry: Registry) -> None:
    S = {"type": "string"}
    from . import data_tools

    defs = [
        ("query", data_tools.DESCRIPTIONS["query"], data_tools.SCHEMAS["query"]["properties"], [], _query, "read"),
        ("answer", "Ask the table one aggregate question and SHOW the person the answer as a chart with its numbers table (a report: "
                   "which segment loses most, how many by group). Same arguments as query, and the numbers come back to you. Use it "
                   "when they ask to see, show, plot or report something.",
         data_tools.SCHEMAS["query"]["properties"], [], _answer, "read"),
        ("dashboard", "Show the person a dashboard: one to six tiles, each a question of the table ({group_by, measures}, as in query), "
                      "drawn as charts and kept so it can be refreshed. You see no number from it: query for the numbers you will say.",
         {"title": S, "tiles": {"type": "array", "minItems": 1, "maxItems": MAX_TILES, "items": data_tools.SCHEMAS["query"]}}, ["tiles"], _dashboard, "read"),
        ("check_leakage", data_tools.DESCRIPTIONS["leakage"], data_tools.SCHEMAS["leakage"]["properties"], ["outcome"], _check_leakage, "read"),
        ("preview_step", data_tools.DESCRIPTIONS["preview_step"], data_tools.SCHEMAS["preview_step"]["properties"], ["kind"], _preview_step, "read"),
        ("request_build", "Ask DCLab to build the project the person chose, once they said yes or ordered it in their latest message (code "
                          "reads their message itself; quote is optional). A pick of an option is not a yes: ask once whether to build. "
                          "If the person named the outcome column (or confirmed it), pass it as outcome.",
         {"quote": S, "outcome": S}, [], _request_build, "write"),
        ("consult", "Send a specialist with the case when a question needs depth: data_investigator (what the table can answer), "
                    "leakage_auditor (columns known only after the decision), critic (what is weak before you propose), "
                    "domain_expert (what matters in this kind of business), solution_designer (how started work should be set up). "
                    "Its findings come back to you and go into the case file. Code decides whether it may be sent now.",
         {"specialist": {"type": "string", "enum": ["data_investigator", "leakage_auditor", "critic", "domain_expert", "solution_designer"]},
          "question": S}, ["specialist", "question"], _consult, "write"),
        ("read_table", "Read the shared table's columns: names, kinds and what DCLab noticed about each, and its row count. No values.", {}, [], _read_table, "read"),
        ("note", "Record facts in the case file: goal, decision, user (who uses the result), data, cannot_answer, constraints or "
                 "assumptions. status said needs the person's exact words as quote (a run of their words, no quote marks); status assumed "
                 "is your supposition. Record several facts in one call with notes: a list of {field, value, status, quote}.",
         {"field": {"type": "string", "enum": [*C.SINGLE, *C.LISTS]}, "value": S, "status": {"type": "string", "enum": ["said", "assumed"]}, "quote": S,
          "notes": {"type": "array", "maxItems": MAX_NOTES, "items": {"type": "object", "properties": {
              "field": {"type": "string", "enum": [*C.SINGLE, *C.LISTS]}, "value": S, "status": {"type": "string", "enum": ["said", "assumed"]}, "quote": S},
              "required": ["field", "value", "status"]}}},
         [], _note, "write"),
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
        ("design_solution_without_data",
         NO_DATA_TOOL_V2 if _no_data_v2() else
         "Only when no table is shared, the person has no data yet (or will not upload it), AND they want a prediction, a forecast, a score or groups: "
         "build a 3-layer Business + Data Science + ML blueprint (economics, Minimum Viable Data Spec, leakage traps, and pre-data task plan). "
         "Not for a why-question, a test's design, a route plan, a report or an explanation: answer those as what they are. "
         "Pass a number only if the person said it. "
         "Never fabricate results or auto-create synthetic data; ask the user first how simulated data should look if they want a prototype.",
         {
             "problem": S,
             "monthly_volume": {"type": "number"},
             "base_rate": {"type": "number"},
             "action_cost": {"type": "number"},
             "value_saved": {"type": "number"},
             "action_success_rate": {"type": "number"},
             "target_hint": S,
         },
         [],
         _design_solution_without_data,
         "write"),
    ]
    for name, description, props, required, handler, effect in defs:
        registry.register(Tool(name=f"case_{name}", description=description, parameters={"type": "object", "properties": props, "required": required},
                               handler=handler, effect=effect, scope="case", takes_context=True, move=MOVES.get(name)))
    registry.guards["case"] = CaseGuard()


TOOL_NAMES = tuple(f"case_{t}" for t in TOOLS)
POLICY = Policy(name="case", system="", tools=TOOL_NAMES, max_steps=20, max_replies=12, stop_after=tuple(f"case_{t}" for t in ENDS_TURN),
                max_tokens=3000, warn_last=True)  # 2026-11-15 (stage 1): a reply of 1,800 tokens cut two calls mid-argument; the last step is announced  # 2026-11-15: 14 and 8 ended three of eight turns (HAB-002)


DECISION_Q = re.compile(r"\b(?:what (?:will|would|do) you (?:do|want to do)|are you deciding|what (?:decision|is the decision)|how will you use|"
                        r"what will (?:this|the answer|it) be used for|do you already offer|who (?:will|would) (?:use|read|act on))\b", re.I)


def without_decision_questions(text: str) -> str:
    """The message without its questions about what the person will do with the answer (the owner's Telco session: asked twice and
    declined, the figures wanted first). Said after an order for figures or once already, such a question is not asked again."""
    parts = re.split(r"(?<=[.?!])\s+", str(text or "").strip())
    kept, dropped_last = [], False
    for p in parts:
        asks = p.rstrip().endswith("?") and bool(DECISION_Q.search(p))
        if asks or (dropped_last and re.match(r"(?:for example|e\.g\.|such as)\b", p, re.I)):  # its examples go with it
            dropped_last = True
            continue
        dropped_last = False
        kept.append(p)
    return " ".join(kept)


def unconfirmed(text: str) -> str:
    """A message without its sentences that ask the person to confirm again (said after they told the agent to go ahead)."""
    parts = re.split(r"(?<=[.?!])\s+", str(text or "").strip())
    kept = [p for p in parts if not (p.rstrip().endswith("?") and CONFIRMS.search(p)) and not re.search(r"\ba plain yes\b", p, re.I)]
    return " ".join(kept) if kept else ""


BARS = re.compile(r"[\u2580-\u259f]{2,}")  # blocks drawn as a chart: the model's own, never code's (a chart comes from answer or dashboard)


def without_bars(text: str) -> str:
    """The message without charts drawn in characters (the owner's Telco session: black blocks beside collapsed labels). A line
    that is only such a chart goes; a sentence that merely holds blocks loses them."""
    out = []
    for line in str(text or "").split("\n"):
        if BARS.search(line):
            line = re.sub(r" {2,}", " ", BARS.sub("", line))
            if len(line.strip()) <= 60:  # only labels were left: "Month-to-month  One year  Two year"
                continue
        out.append(line)
    return "\n".join(out)


def checked(text: str, state: CaseState, dropped: Optional[list[str]] = None) -> str:
    """A message to the person with every sentence holding a number nobody gave or code computed dropped (and listed in
    ``dropped``, so the agent can be told and write it again). The rule is cited.unsaid, the conversation exam's own."""
    from .. import cited

    sources = [*state.case.words, *state.computed]
    # Stage 2 of 2026-11-15: the unit is a sentence inside a line. Before, a text was split only at ". ", "? " and "! " and
    # joined with spaces: a list of steps with no full stops was one unit, so one number nobody gave removed the whole list
    # and the question after it, and every line break of a plan was lost. The check itself is unchanged.
    lines_out: list[str] = []
    for line in without_bars(text).strip().split("\n"):
        if not line.strip():
            lines_out.append("")  # a paragraph break stays one
            continue
        lead = line[:len(line) - len(line.lstrip())]
        kept = []
        for s in re.split(r"(?<=[.?!])\s+", line.strip()):
            if not s:
                continue
            unsaid = cited.unsaid(s, sources)
            if unsaid and think_first() and PLAN_STEP.match(line):
                # a step of the plan loses the aside that states the number, not the step: a plan with a step missing is
                # a broken plan (HAB-055: a plan reached the person as steps 2, 3 and 4)
                lighter = re.sub(r"\s*\([^()]*\)", lambda m: "" if cited.unsaid(m.group(0), sources) else m.group(0), s)
                if lighter.strip() and not cited.unsaid(lighter, sources):
                    if dropped is not None:
                        dropped.append(f"an aside in a step of the plan (numbers nobody gave: {', '.join(unsaid)})")
                    kept.append(lighter)
                    continue
            if unsaid:
                if dropped is not None:
                    dropped.append(f"{s} (numbers nobody gave: {', '.join(unsaid)})")
                continue
            kept.append(s)
        if kept and not re.fullmatch(r"(?:[-*•]|\d+[.)])", " ".join(kept).strip()):  # a bullet left with only its mark is dropped
            lines_out.append(lead + " ".join(kept))
    out = "\n".join(lines_out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


NARRATION = re.compile(r"^\W*(?:(?:now|first|next|then|ok(?:ay)?|good|great|alright)[,.!]?\s+)*(?:i'?ll|i will|i'?m going to|i am going to|let me|let's|"
                       r"i need to|i should|i'?ll start|i (?:can|must) (?:now )?(?:check|look|ask|record|note|query)|checking|looking at|recording)\b", re.I)
NARRATION_WORDS = 40
# the calls a working note sits beside: looking and writing the case down. Words beside options, a question, a start or a
# build are for the person ("I'll rank your customers by risk and send the list each Monday" beside the options)
BOOKKEEPING = frozenset(f"case_{t}" for t in ("read_table", "query", "check_leakage", "preview_step", "note", "add_unknown", "resolve",
                                              "confirm_assumption", "consult"))


def spoken(messages: list[dict[str, Any]]) -> list[str]:
    """What the model wrote for the person in one turn: its texts, without the short working notes it wrote beside a tool call
    ("I'll start by checking the table", "Now let me record the goal"). Stage 2 of 2026-11-15: those were joined into the
    message and used up its words. A text beside a tool call that is longer, or is no note of its own work, stays: a plan
    written in the same reply as the options is the message."""
    out, every = [], []
    for m in messages:
        text = str(m.get("content") or "").strip()
        if m.get("role") != "assistant" or not text:
            continue
        every.append(text)
        names = {str((c.get("function") or {}).get("name") or c.get("name") or "") for c in m.get("tool_calls") or []}
        if names and names <= BOOKKEEPING and word_count(text) <= NARRATION_WORDS and NARRATION.match(text):
            continue
        out.append(text)
    return out or every  # the review: a turn whose only words were such a note keeps them, never the fallback line


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


# 2026-11-01 (CEX-026 r-13): a data scientist asked for a full script; the number check withheld every sentence with a setting nobody
# gave, each withheld answer was written again at full length, five of fourteen calls were such rewrites and one conversation cost
# EUR 0.73 and ended without the code. The check is unchanged (it is the 0 against 139 invented numbers); what is capped is what it
# costs: a message holds at most SHOWN_WORDS words, and a reply longer than REWRITE_WORDS that loses sentences to the check is not
# written again by the model (what survived reaches the person, with a plain note).
SHOWN_WORDS = 450  # 2026-11-15: a person who gave a full brief gets the problem, the approach and the plan in one message (HAB-002: 300 cut them)
REWRITE_WORDS = 150
SHOWN_WORDS_PLAN = 700  # with the think-first switch on: the problem, what needs care, the approach and the plan, until the plan has its own card (stage 3)
CLEAR_THINK_FIRST = ("The person's request is clear on arrival (the goal, the subject and the data are known). Do not ask clarifying "
                     "questions on this turn. Write your first full message, then do the plan's first step in this same turn: start "
                     "it, or propose it when the person should choose. For a clear request keep the first parts short (the problem in "
                     "two or three sentences; what needs care only when something does) and give the plan in full: it is what they "
                     "will act on. If the request should not be done as asked, say why and propose the sound way instead.")
THINK_FIRST_TOKENS = 6000  # a first full message with its plan, after the model's own thinking: at 3,000 three of sixteen came back empty or cut
REWRITE_LONG = ("[DCLab's code] Part of your message was not shown to the person, because its numbers come from nobody: the person did "
                "not give them and no tool computed them. Write the whole message again at about the same length, keeping its parts "
                "and its plan: no digits at all unless they are copied exactly from a tool's result or the person's words. Reply with "
                "the message only.")
CUT_NOTE = "(That is the short version. Ask me about any part and I will go deeper.)"
WITHHELD_NOTE = ("(I left out the parts whose numbers I cannot source from you or your data. Ask me about any part in words "
                 "and I will go through it.)")


def word_count(text: str) -> int:
    return len(str(text or "").split())


# The plan is kept whole (2026-11-16). With the think-first switch on the plan is the last part of the first full message,
# so the word limit cut the plan first: two of sixteen plans scored lower than without the switch, one squeezed by a long
# "what needs care", one cut. The numbered steps and their closing lines (when to stop, what to do if a step fails) are
# taken out before the limit is applied to the rest, kept in the case's state, and shown back to the model on later turns.
PLAN_STEP = re.compile(r"^\s*(?:\*\*)?\d+[.)](?:\*\*)?\s+\S")
PLAN_TAIL = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?(?:stop\b|when to stop|i(?:'ll| will| would) stop|we stop|if (?:a|any|one|this|that) step fails|if a step|if it fails)", re.I)
PLAN_HEAD = re.compile(r"\bplan\b|\bsteps\b", re.I)
PLAN_WORDS = 450  # a plan longer than this is prose with numbers in it: it is shortened like the rest
PLAN_NOTE = ("Your plan for this person, as you wrote it (it is on their screen):\n{plan}\nKeep to it. Say which step you are on and do "
             "that step. Change the plan only when something new makes it wrong, and then say what changed and why. Do not write "
             "the whole plan again.")


INLINE_STEP = re.compile(r"(?:(?<=\s)|^)(?:\*\*)?(\d+)[.)](?:\*\*)?\s+(?=\S)")
PLAN_CLOSE = re.compile(r"\s+(?=(?:Stop when|If a step fails)\b)")


def plan_lines(text: str) -> str:
    """A plan written as one paragraph ("1. ... 2. ... 3. ... Stop when ...") as a list, a step to a line (HAB-052 to
    HAB-055: four of sixteen plans were written so, and were neither kept whole nor easy to read). A line is split only
    when it holds at least three step numbers that rise from one or two, so a sentence with a number in it stays whole."""
    out: list[str] = []
    for line in str(text or "").split("\n"):
        run: list[tuple[int, int]] = []
        for m in INLINE_STEP.finditer(line):
            n = int(m.group(1))
            if (not run and n <= 2) or (run and run[-1][1] < n <= run[-1][1] + 2):
                run.append((m.start(), n))
        if len(run) < 3 or (run[0][0] == 0 and len(run) == 1):
            out.append(line)
            continue
        head = line[:run[0][0]].rstrip()
        if head:
            out.append(head)
        for i, (pos, _) in enumerate(run):
            step = line[pos:(run[i + 1][0] if i + 1 < len(run) else len(line))].strip()
            if i + 1 == len(run):  # the closing lines after the last step get lines of their own
                step, *closing = PLAN_CLOSE.split(step)
                out.append(step.strip())
                out.extend(c.strip() for c in closing if c.strip())
            else:
                out.append(step)
    return "\n".join(out)


def renumbered(text: str) -> str:
    """The plan's steps numbered in order again: a step the number check took out leaves no gap ("1, 2, 3, 5")."""
    before, plan, after = plan_block(text)
    if not plan:
        return str(text or "")
    n, lines = 0, []
    for line in plan.split("\n"):
        if PLAN_STEP.match(line):
            n += 1
            line = re.sub(r"^(\s*(?:\*\*)?)\d+", lambda m: f"{m.group(1)}{n}", line, count=1)
        lines.append(line)
    return "\n\n".join(p for p in (before, "\n".join(lines), after) if p.strip())


def plan_block(text: str) -> tuple[str, str, str]:
    """A message as what comes before its plan, the plan, and what comes after. The plan is the last run of at least two
    numbered steps (a step may run over a few lines), with a short heading line that names it and the lines that say when
    to stop and what to do if a step fails. No such run: the whole message, and no plan."""
    lines = str(text or "").split("\n")
    numbered = [i for i, line in enumerate(lines) if PLAN_STEP.match(line)]
    if len(numbered) < 2:
        return str(text or ""), "", ""
    start = end = numbered[-1]
    for i in reversed(numbered[:-1]):
        if start - i > 4:  # more than three lines between two numbers: another list, further up
            break
        start = i
    if sum(1 for i in numbered if start <= i <= end) < 2:
        return str(text or ""), "", ""
    stop = end + 1
    while stop < len(lines) and stop - end <= 8:
        line = lines[stop]
        if not line.strip() or line.startswith((" ", "\t")) or PLAN_TAIL.match(line):
            stop += 1
            continue
        break
    while stop > end + 1 and not lines[stop - 1].strip():
        stop -= 1
    head = start - 1
    while head >= 0 and not lines[head].strip():
        head -= 1
    if head >= 0 and len(lines[head].split()) <= 12 and PLAN_HEAD.search(lines[head]) and not PLAN_STEP.match(lines[head]):
        start = head
    return "\n".join(lines[:start]).rstrip(), "\n".join(lines[start:stop]).strip(), "\n".join(lines[stop:]).strip()


def shortened(text: str, limit: int = SHOWN_WORDS) -> tuple[str, bool]:
    """A message cut once it passes ``limit`` words, and whether it really got shorter. It is cut at a sentence; a long block
    with line breaks and few sentence ends (a script) is cut at a line. The first piece always stays, and a code fence left
    open by the cut is closed."""
    text = str(text or "").strip()
    if word_count(text) <= limit:
        return text, False
    units: list[str] = []
    for n, piece in enumerate(re.split(r"(?<=[.?!])(\s+)", text)):  # sentence, whitespace, sentence, ...
        if n % 2:
            units[-1] += piece
        elif word_count(piece) > max(20, limit // 2) and "\n" in piece:
            units.extend(re.findall(r"[^\n]*\n|[^\n]+$", piece))
        else:
            units.append(piece)
    kept, used = [], 0
    for unit in units:
        size = word_count(unit)
        if kept and used + size > limit:
            break
        kept.append(unit)
        used += size
    out = "".join(kept).rstrip()
    if out.count("```") % 2:
        out += "\n```"
    return (out, True) if word_count(out) < word_count(text) else (text, False)


REWRITE_SHORT = ("[DCLab's code] Your message was long and most of it was not shown to the person, because its numbers come from nobody: "
                 "the person did not give them and no tool computed them. Write the answer itself now, in words, in at most 120 words: "
                 "no digits at all unless they are copied exactly from a tool's result or the person's words. Reply with the message only.")


def has_statement(text: str) -> bool:
    """A message that says something, not only asks (2026-11-01: a cut answer left a bare follow-up question)."""
    parts = [p for p in re.split(r"(?<=[.?!])\s+", str(text or "").strip()) if p.strip()]
    return any(not p.rstrip().endswith("?") for p in parts)


def heard(state: CaseState, words: str) -> None:
    """One message of the person, heard: the case file gets their words and every flag that holds for this message only is
    set (a go-ahead, a wish for options or figures, the turn budget). The turn starts with it; a test of the gates starts
    from it too, so the states it checks are the ones a turn really makes (stage 2 of 2026-11-15)."""
    if state.asking and declines(str(words)):
        state.case = C.declined(state.case)
    state.case = C.hear(state.case, words)
    state.wants_options = bool(WANTS_OPTIONS.search(str(words)))  # this message only (the review: it stayed on for the case)
    state.go_ahead = goes_ahead(str(words), asking=state.asking)  # this message only (asking: the last turn ended on a question)
    state.wants_figures = (bool(state.table) and C.wants_figures([str(words)])  # this message only: figures first, unless it is a model or a decision
                           and not ({"prediction", "business"} & set(C.named_tracks([str(words)]))))  # asked for (what was said before does not count)
    state.queries = state.query_errors = 0
    state.no_data_built = False
    state.cards = []
    state.only_option = None
    if state.table and (state.case.data is None or state.case.data.status != "computed"
                        or not any(str(c).startswith(f'{{"table": "{state.table["table"]}"') for c in state.computed)):  # 2026-10-31: code reads the shared
        _read_table(state)  # table (again after a restore, or when a table arrives after an earlier assumption), so no model call is spent on it and its counts may be said
    state.turns.append({"role": "user", "content": str(words)})
    state.said, state.shown, state.asking = [], [], False
    person_turns = sum(1 for t in state.turns if t.get("role") == "user")
    state.clear_on_arrival = C.is_clear_on_arrival(state.case, state.table)
    state.budget_reached = (person_turns > TURN_BUDGET and C.known(state.case.goal) and C.known(state.case.data)
                            and state.case.route is None and not state.case.chosen)  # the review: off once work started or was picked


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
    heard(state, words)
    if state.trace is None:
        from .harness import tracer

        state.trace = tracer("case")
    person_turns = sum(1 for t in state.turns if t.get("role") == "user")
    # the review of stage 4: a number, a ranking or a chart of their table is answered first with the table, whatever the
    # switch says (the owner's rule of 2026-11-02): only a need that has to be worked out gets the think-first note
    clear_note = ((CLEAR_THINK_FIRST if think_first() and not state.wants_figures else
                   "The person's request is crystal-clear on arrival (goal, subject/target, and data are already established). "
                   "Do NOT ask clarifying questions or reflective questions on Turn 1; propose actionable options (propose) "
                   "or start the requested work (start/query/answer) in this turn, and write in your message what you understood of "
                   "the problem, your approach and the plan's steps.")) if (state.clear_on_arrival and person_turns == 1) else None
    plan_note = PLAN_NOTE.format(plan=state.plan) if think_first() and state.plan and person_turns > 1 else None
    built = ctx.build(state.case, state.turns, state.table, notes=[n for n in (chip_note(state), clear_note, plan_note) if n])
    messages = ctx.messages(prompts.text(prompt_name()), built)
    from . import decision_loop

    if decision_loop.is_enabled(state.workspace_home):
        return decision_loop.run_bounded_turn(state, _Cells(client, state) if client is not None else None, registry, messages)
    start = len(messages)
    before = _spent(client) + _spent(state.specialist_client) + _spent(state.route_client)  # the review: the route check counts too
    client = _Cells(client, state)  # 37.6: a request holds cell values only when a query's answer is in it
    route_ahead(state)
    policy = replace(POLICY, max_tokens=THINK_FIRST_TOKENS) if think_first() else POLICY
    result = run_policy(policy, client, registry, messages, context=state, trace=state.trace)
    if not result.steps and not state.said and not any(str(m.get("content") or "").strip() for m in result.messages[start:] if m.get("role") == "assistant"):
        # Stage 2 (HAB-011): the first reply came back with no word and no call, the one rewrite could not run a tool, and a
        # full brief was answered "could you tell me a little more". The turn is run once more instead, with its tools.
        del result.messages[start:]
        result = run_policy(policy, client, registry, result.messages, context=state, trace=state.trace)
    texts = spoken(result.messages[start:])
    if not texts and not state.said:  # CEX-011 d-04: every reply spent on queries, no word for the person: asked once, no move allowed
        again_from = len(result.messages)
        result.messages.append({"role": "user", "content": SAY_NOW})
        once = Policy(name="case", system="", tools=TOOL_NAMES, max_steps=POLICY.max_steps, max_replies=1, stop_after=POLICY.stop_after)
        state.rewriting = True
        try:
            run_policy(once, client, registry, result.messages, context=state, trace=state.trace)
        finally:
            state.rewriting = False
        texts = spoken(result.messages[again_from + 1:])
    dropped: list[str] = []

    def again(instruction: str) -> list[str]:
        """One more model reply, the tools offered (the API needs them beside earlier calls) but none allowed to run."""
        from_at = len(result.messages)
        result.messages.append({"role": "user", "content": instruction})
        once = Policy(name="case", system="", tools=TOOL_NAMES, max_steps=POLICY.max_steps, max_replies=1, stop_after=POLICY.stop_after)
        state.rewriting = True
        try:
            run_policy(once, client, registry, result.messages, context=state, trace=state.trace)
        finally:
            state.rewriting = False
        return [str(m.get("content") or "").strip() for m in result.messages[from_at + 1:] if m.get("role") == "assistant"]

    def capped(parts: list[str]) -> tuple[list[str], bool]:
        """The messages of a turn within SHOWN_WORDS in all (an echoed note of code's is taken out first), and whether any was cut."""
        out, was, left = [], False, (SHOWN_WORDS_PLAN if think_first() else SHOWN_WORDS)
        for part in parts:
            part = str(part or "").replace(CUT_NOTE, "").replace(WITHHELD_NOTE, "").strip()
            if not part:
                continue
            if think_first():
                part = plan_lines(part)  # a plan written as one paragraph becomes a list, so it can be kept whole and read
            before, plan, after = plan_block(part) if think_first() else (part, "", "")
            if plan and word_count(plan) <= PLAN_WORDS:  # the plan stays whole: the limit is for the words around it
                kept = []
                for piece in (before, after):
                    if piece.strip():
                        short, c = shortened(piece, max(left, 40))
                        was, left = was or c, left - word_count(short)
                        kept.append(short)
                    else:
                        kept.append("")
                state.plan = plan
                out.append("\n\n".join(p for p in (kept[0], plan, kept[1]) if p.strip()))
                continue
            short, c = shortened(part, max(left, 40))
            was, left = was or c, left - word_count(short)
            out.append(short)
        return out, was

    shorts, cut = capped(texts)  # the length cap comes first: a sentence cut for length is never sent back for a rewrite
    written = sum(word_count(s) for s in shorts)  # how long the reply was as written, not what the check left of it
    long_reply = written > REWRITE_WORDS
    answers = [checked(s, state, dropped) for s in shorts]
    words_out = _unique(answers + [checked(q, state, dropped) for q in state.said])
    if state.go_ahead or says_yes(str(words)):  # 2026-11-01: after a go-ahead, no question asking the person to confirm again
        trimmed = [unconfirmed(w) for w in words_out]
        if any(trimmed):  # a message that was only the question is kept rather than replaced by code's fallback line
            words_out = trimmed
    asked_before = any(t.get("role") == "assistant" and DECISION_Q.search(str(t.get("content") or "")) for t in state.turns)
    if state.go_ahead or state.wants_figures or asked_before and says_no_more(str(words)):  # 2026-11-02: the figures first; never asked again
        trimmed = [without_decision_questions(w) for w in words_out]
        if any(trimmed):
            words_out = trimmed
    withheld = False
    if dropped:  # a refusal comes back to the agent so it can write again (the plan's rule), once: never a silent turn
        survivors = sum(word_count(w) for w in answers)
        if long_reply and any(has_statement(w) for w in answers) and survivors >= 0.4 * written:
            withheld = True  # a long reply with most of its answer left is not written again (CEX-026): what survived reaches the person
        else:
            rewritten = again((REWRITE_LONG if think_first() else REWRITE_SHORT) if long_reply
                              else REWRITE.format(dropped="\n".join(f"- {d}" for d in dropped)))
            parts, rcut = capped(rewritten)
            fixed = [w for w in (checked(s, state) for s in parts) if w]
            kept = sum(len(w) for w in fixed) / max(1, sum(len(r) for r in rewritten if r))
            if not long_reply and (not any(has_statement(w) for w in fixed) or kept < 0.6):
                # cut again: one stricter try, so the answer reaches the person (short replies only: a long one has had its one short try)
                strict = again(REWRITE_STRICT)
                parts, scut = capped(strict)
                again_fixed = [w for w in (checked(s, state) for s in parts) if w]
                if any(has_statement(w) for w in again_fixed):
                    fixed, rcut = again_fixed, scut
            if fixed and (not long_reply or any(has_statement(w) for w in fixed)):
                words_out, cut = _unique(fixed), rcut  # the note follows the text that is shown, not the first draft
            elif long_reply and any(words_out):
                withheld = True
    notes = [n for n, on in ((CUT_NOTE, cut), (WITHHELD_NOTE, withheld)) if on]
    if not any(words_out):  # never a silent turn (the review): code's own line, and the question if one was asked
        words_out, notes = [FALLBACK_NUMBERS if dropped else FALLBACK] + ([q for q in (checked(q, state) for q in state.said) if q] or []), []
    text = "\n\n".join([w for w in words_out if w] + notes)
    if think_first():  # the plan as the person sees it, after the number check: numbered without a gap, and kept for later turns
        text = renumbered(text)
        shown_plan = plan_block(text)[1]
        if shown_plan:
            state.plan = shown_plan
    thread, state.route_thread = state.route_thread, None
    if thread is not None:  # a second reading nobody waited for: given a moment so its cost is this turn's, never waited on for long
        thread.join(timeout=ROUTE_WAIT)
    state.spent_eur += max(0.0, _spent(client) + _spent(state.specialist_client) + _spent(state.route_client) - before)  # 37.8: what this turn cost
    state.turns.append({"role": "assistant", "content": text + ("\n(Options: " + "; ".join(o.title for o in state.shown) + ")" if state.shown else "")})
    if state.checkpoint:  # 37.8: saved after every turn, so a stop or a crash loses nothing
        case_session.save(state, state.checkpoint)
    return {"text": text, "options": list(state.shown), "route": state.case.route, "asking": state.asking, "cards": list(state.cards)}


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
