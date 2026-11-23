"""Which "brain" a team agent runs with, and the one audited switch that chooses it (B1 of the team workspace, 2026-11-23).

A team agent (the Lead, Business, Data Science, ML Engineering, Sales, Marketing) always acts through the same strict
layer: the same tools, guard, task store and report builder. Only what chooses its next step differs, its *arm*:

- ``rules``: code chooses every step from the stored task (the first release; the default for every role);
- ``helper``: code decides, and a model helps at named seats (its output used only where a code check passes);
- ``agent``: a model plans first and chooses its tool calls inside the strict layer.

A host keeps one :class:`BrainSetting` per role that a reviewer or the owner set, with a reason. :func:`arm_of` reads the
arm a turn runs with: a host's override (a test's, an evaluation's) first, then the stored setting, then ``rules``. A
role a host does not switch (a specialist, the person's Home entry) is always ``rules``. When a model arm cannot answer
(no model, a refused or failed request, an answer its check refuses), the turn falls back to the rules and the host
records why, as one of :data:`FALLBACKS`. Codes only: never a person's words or a table's value.

**The helper's seats** (B2, 0.172.0). In the ``helper`` arm code still decides everything; a model helps at named seats
(:data:`SEATS`), and its output is used only where a code check passes, else the rules' own answer stands and the
fallback is recorded (:class:`SeatRecord`). Which seats a role has is :data:`ROLE_SEATS`. Three seats change what the
rules do next (:data:`BEFORE_SEATS`: reading the person's words, the teams a request goes to, a clarifying question's
wording), so they are asked *before* the rules act on that choice, within a bounded number of seconds per turn; the
other two change nothing the rules do (:data:`AFTER_SEATS`: a report explained in plain words, the risks the helper
sees) and are asked after the turn, never waited on. :func:`check_code` and :func:`check_question` are the portable code
checks; the explanation and the risks are checked by the host's citation check.

**The agent arm acts** (B3, 0.173.0). In the ``agent`` arm a model chooses the turn's calls inside the strict layer (the
same tools, guard, task store and report builder as the rules). It writes its plan first (kept whole by the host), then
chooses its actions, their order, its questions and hand-overs. A host owes it four rules, each checked by code:

- budgets per turn (steps, seconds, euros): when one ends, the turn ends with what was done recorded (:data:`AGENT_STOPS`);
- a fallback only *before* the first action: a model that is missing, refused, late, invalid or failing, or a budget that
  ends before any write, leaves the turn to the rules (one of :data:`FALLBACKS`). After an action the rules never run on
  top (they would repeat work): the turn ends and its partial state is recorded (``ended``);
- one turn per delivery: a delivery handed over again while its turn runs, or after it ended, starts no second turn;
- a report keeps the model's sentences only where they hold: every number traces to a record (an action's output, a
  tool's result, a returned report or the brief), no sentence claims an action without its terminal record
  (:func:`claimed_actions`), overclaims or promises an action (:data:`REPORT_REFUSALS`). Code attaches the numbers from
  the records itself.

:class:`AgentTurnRecord` is one agent turn's record. Codes, ids and measures: the plan's words stay with the host (its
exchange and its turn journal), never in the record.

Contracts ``dclab_core.team_brain_setting.v1``, ``dclab_core.team_helper_seat.v1`` and ``dclab_core.team_agent_turn.v1``.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping, Optional, Sequence, Union

from pydantic import Field, model_validator

from dclab_core.contracts import CONTRACTS, Versioned, _version

CONTRACT = "dclab_core.team_brain_setting.v1"
ARMS = ("rules", "helper", "agent")
DEFAULT_ARM = "rules"
ROLES = ("lead", "business", "data_science", "ml_engineering", "sales", "marketing")  # the team agents a host may switch
REVIEWERS = ("reviewer", "owner")  # who may switch a role's brain
SOURCES = ("default", "stored", "override")  # where a turn's arm came from
# why a model arm's turn fell back to the rules ("stopped": the turn ended before the rules finished, so no model was asked).
# 0.172.0 (B2): a helper seat also falls back when the model said "unsure" or its answer failed the seat's code check.
# 0.173.0 (B3): an agent-arm turn also falls back when a budget ended before its first action (``budget``), and a turn a
# person's own message starts (their case opened in their words, their answer recorded) is code's, never a model's
# (``person_entry``): code writes down exactly what the person gave; a turn that planned and then wrote nothing on a task still
# open (every write refused, or none made) falls back too (``no_action``), so the work is never left waiting, and the arm's
# comparison counts it as the agent's failure
FALLBACKS = ("no_model", "no_live_models", "refused", "timeout", "invalid_output", "error", "stopped", "busy", "unsure",
             "check_failed", "budget", "person_entry", "no_action")
SHORT = 300

# ---------------------------------------------------------------------------------------------- 0.172.0 (B2): the seats
SEATS = ("read_target", "route_teams", "word_question", "explain_report", "list_risks")
BEFORE_SEATS = ("read_target", "route_teams", "word_question")  # change what the rules do: asked before they act, bounded
AFTER_SEATS = ("explain_report", "list_risks")  # change nothing the rules do: asked after the turn, never waited on
# Which seats each role's helper has. A person's words are read by the agent that speaks for that person (Business for its
# member, the Lead for the person on Home), in the turn of the team that owns the outcome; that team's code checks them.
ROLE_SEATS: dict[str, tuple[str, ...]] = {
    "lead": ("read_target", "route_teams", "explain_report", "list_risks"),
    "business": ("read_target",),
    "data_science": ("word_question", "explain_report", "list_risks"),
    "ml_engineering": ("explain_report", "list_risks"),
    "sales": ("explain_report", "list_risks"),
    "marketing": ("explain_report", "list_risks"),
}
UNSURE = "unsure"  # what a code seat answers when the words do not settle it: the rules then ask, as before
# "used": the check passed and the model's answer was used (0.173.0: written out, so B3's fallbacks are no seat's outcome)
SEAT_OUTCOMES = ("used", "no_model", "no_live_models", "refused", "timeout", "invalid_output", "error", "stopped", "busy", "unsure",
                 "check_failed")
QUESTION_CHARS = 240  # a clarifying question's wording, at most
_NUMBER = re.compile(r"\d|\b(?:zero|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|"
                     r"sixty|seventy|eighty|ninety|hundred|thousand|million|billion|half|percent)\b", re.I)
# a promise of an action: the agent says it will do something (an agent acts only through its tools, never by its words)
_PROMISE = re.compile(r"\b(?:i|we)\s*(?:will|'ll|shall|am going to|are going to)\b|\b(?:i|we)'ll\b|\bpromise\w*\b|\bguarantee\w*\b|"
                      r"\bwill\s+(?:then\s+)?(?:call|contact|email|e-mail|send|approve|train|deploy|launch|start|run|build|score)\b",
                      re.I)
QUESTION_CHECKS = ("empty", "too_long", "has_number", "promises_action", "not_a_question", "drops_an_option", "unknown_question")

Arm = Literal["rules", "helper", "agent"]
Role = Literal["lead", "business", "data_science", "ml_engineering", "sales", "marketing"]


class BrainSetting(Versioned):
    """One role's brain as a reviewer or the owner set it: the arm, since when, by whom, why, and what it replaced."""

    schema_version: Literal["dclab_core.team_brain_setting.v1"] = _version("team_brain_setting", 1)
    role: Role
    arm: Arm
    previous: Optional[Arm] = Field(default=None, description="The arm it replaced (None: the default, rules).")
    by: Literal["reviewer", "owner"]
    reason: str = Field(min_length=1, max_length=SHORT, description="Why, in the reviewer's words: kept in the audit trail.")
    since: str = Field(description="ISO 8601, UTC.")


def parse_override(text: Optional[str]) -> dict[str, str]:
    """A host's override as ``{role: arm}``: one arm for every role (``helper``), or ``role=arm`` pairs separated by commas
    (``lead=agent,sales=helper``). Empty: no override. An unknown role or arm is an error, never ignored."""
    raw = str(text or "").strip().lower()
    if not raw:
        return {}
    if "=" not in raw:
        if raw not in ARMS:
            raise ValueError(f"a team brain is one of {', '.join(ARMS)}, not {raw!r}")
        return {role: raw for role in ROLES}
    out: dict[str, str] = {}
    for part in (p.strip() for p in raw.split(",") if p.strip()):
        role, _, arm = (x.strip() for x in part.partition("="))
        if role not in ROLES:
            raise ValueError(f"{role!r} is not a team role a brain is chosen for ({', '.join(ROLES)})")
        if arm not in ARMS:
            raise ValueError(f"a team brain is one of {', '.join(ARMS)}, not {arm!r}")
        out[role] = arm
    return out


def arm_of(settings: Mapping[str, Union[BrainSetting, Mapping[str, Any]]], role: str,
           override: Optional[Mapping[str, str]] = None) -> tuple[str, str]:
    """The arm ``role`` runs with and where it came from (:data:`SOURCES`): the override, else the stored setting, else
    ``rules``. A role outside :data:`ROLES` is always ``rules`` (``default``)."""
    if role not in ROLES:
        return DEFAULT_ARM, "default"
    if override and override.get(role) in ARMS:
        return str(override[role]), "override"
    found = (settings or {}).get(role)
    arm = getattr(found, "arm", None) if found is not None and not isinstance(found, Mapping) else (found or {}).get("arm")
    if arm in ARMS:
        return str(arm), "stored"
    return DEFAULT_ARM, "default"


Seat = Literal["read_target", "route_teams", "word_question", "explain_report", "list_risks"]
SeatOutcome = Literal["used", "no_model", "no_live_models", "refused", "timeout", "invalid_output", "error", "stopped", "busy",
                      "unsure", "check_failed"]


class SeatRecord(Versioned):
    """One helper seat on one turn (0.172.0, B2): whose answer the host used, and why. The model's answer is kept here only
    as a code (an event, a team, a question id); a text seat's words stay in the model's exchange (``request``) and, when
    its check passed, in what the host showed. Never the person's words, a prompt or a table's value."""

    schema_version: Literal["dclab_core.team_helper_seat.v1"] = _version("team_helper_seat", 1)
    seat: Seat
    role: Role = Field(description="The role whose helper sat: its arm is the one switched to helper.")
    agent: str = Field(min_length=1, max_length=80, description="The agent whose turn asked (Data Science reads Business's person's words).")
    task: Optional[str] = Field(default=None, max_length=200)
    outcome: SeatOutcome
    used: Literal["model", "rule"] = Field(description="Whose answer the turn went on with: the model's only when outcome is used.")
    answer: Optional[str] = Field(default=None, pattern=r"^[a-z0-9_]{1,60}$", description="A code seat's answer (an event, a team, a question id).")
    check: Optional[str] = Field(default=None, pattern=r"^[a-z0-9_]{1,60}$", description="Why the code check refused it, as a code.")
    request: Optional[str] = Field(default=None, max_length=120, description="The model request whose exchange keeps the reply and its thinking.")
    seconds: Optional[float] = Field(default=None, ge=0, description="How long the turn waited on the seat (a before-seat only).")


def check_code(answer: Any, allowed: Sequence[str]) -> Optional[str]:
    """Why a code seat's answer may not be used, or None: it must be one of ``allowed``. ``unsure`` is never an error
    (the caller treats it as "the rules ask"), so it returns ``unsure`` for the caller to tell apart."""
    code = str(answer or "").strip().lower()
    if code == UNSURE:
        return UNSURE
    return None if code in allowed else "not_allowed"


def check_question(text: Any, *, options: Sequence[str] = (), limit: int = QUESTION_CHARS) -> Optional[str]:
    """Why a clarifying question's wording may not be shown to a person, or None (one of :data:`QUESTION_CHECKS`): it has
    words, fits in ``limit`` characters, is one question ending with a question mark, holds no number (a number in a
    question invites a number nobody measured), promises no action (an agent acts through its tools, never by its words),
    and still names every option the rules' question offers (``options``: a regular expression of the phrases that name
    each, so "cancel" names a cancellation)."""
    said = " ".join(str(text or "").split())
    if not said:
        return "empty"
    if len(said) > limit:
        return "too_long"
    if not said.endswith("?") or said.count("?") != 1:
        return "not_a_question"
    if _NUMBER.search(said):
        return "has_number"
    if _PROMISE.search(said):
        return "promises_action"
    if any(not re.search(rf"\b(?:{o})\b", said, re.I) for o in options):
        return "drops_an_option"
    return None


# ---------------------------------------------------------------------------------------------- 0.173.0 (B3): the agent acts
# How an agent-arm turn ended: it acted (and ended on its own words), fell back to the rules before any action, ended
# after an action without finishing (a budget, a failing model, a crash), or was a delivery handed over again (no turn)
AGENT_OUTCOMES = ("acted", "fell_back", "ended", "repeat")
# Why its run stopped: the model answered (its own end), a budget (steps, seconds, euros; replies: the requests' bound),
# a reply cut twice at its length limit, calls written as text, the model failing (error, timeout, refused), a turn cut
# off by a crash and not run again (interrupted), or no plan before the first action (no_plan)
AGENT_STOPS = ("answered", "steps", "time", "euros", "replies", "max_output_tokens", "tools_as_text", "error", "timeout", "refused",
               "interrupted", "no_plan")
# Why code refused one of the model's report sentences (or limits)
REPORT_REFUSALS = ("number_not_in_records", "number_word_not_in_records", "claims_unrecorded_action", "overclaims",
                   "promises_action", "unknown_source", "empty")
PLAN_CHARS = 4000  # a plan's words, at most (kept whole: the host never cuts one)

_WORDS = r"(?:\s+[\w'-]+){0,4}\s+"
# A sentence that says an action was done. Each claim is checked against the action log: an action with no terminal
# record ("succeeded" or "simulated") on the task or a task it handed out is a claim code refuses. The last four have no
# tool in any release of the team workspace, so a claim of them is always refused (unless the sentence denies it).
_BEEN = r"(?:\s+[\w'-]+){0,4}\s+(?:was|were|has been|have been|had been|got|is now|are now)\s+"  # the passive: "the schema was inspected"
CLAIMS: dict[str, "re.Pattern[str]"] = {name: re.compile(pattern, re.I) for name, pattern in (
    ("inspect_schema", rf"\b(?:inspected|examined|read|reviewed|looked at|went through)\b{_WORDS}(?:schema|columns?)\b|"
                       rf"\b(?:schema|columns?)\b{_BEEN}(?:inspected|examined|read|reviewed|looked at)\b"),
    ("check_leakage", rf"\b(?:checked|tested|screened|audited|ran)\b(?:{_WORDS})?(?:for\s+)?(?:\w+\s+)?leak\w*|"
                      rf"\bleak\w*\b{_BEEN}(?:checked|tested|screened|audited|ruled out)\b"),
    ("define_baseline", rf"\b(?:defined|set|named|chose|picked|fixed)\b{_WORDS}baseline\b|\bbaseline\b{_BEEN}(?:defined|set|named|chosen|picked|fixed)\b"),
    ("validate_contract", rf"\bvalidated\b{_WORDS}contract\b|\bcontract\b{_BEEN}validated\b"),
    ("review_evidence", rf"\breviewed\b{_WORDS}(?:evidence|repl(?:y|ies)|reports?|answers?)\b|"
                        rf"\b(?:evidence|repl(?:y|ies)|reports?|answers?)\b{_BEEN}reviewed\b"),
    ("delegate", rf"\b(?:delegated|handed|assigned|opened)\b{_WORDS}(?:tasks?|work|it|this|request)\b|"
                 r"\basked\s+(?:the\s+)?(?:data science|ml engineering|sales|marketing|business)\b|"
                 rf"\b(?:tasks?|work|request)\b{_BEEN}(?:delegated|handed|assigned|opened)\b"),
    ("score", r"\bscored\b"),
    ("train", r"\b(?:trained|retrained|fitted|fit)\b(?:\s+[\w'-]+){0,3}\s+models?\b|\bmodels?\b(?:\s+[\w'-]+){0,4}\s+(?:was|were|has been|have been|had been|got|is now)\s+(?:trained|retrained|fitted)\b"),
    ("deploy", r"\bdeployed\b"),
    ("approve", r"\bapproved\b|\bsigned[- ]off\b|\bapproval\b(?:\s+[\w'-]+){0,3}\s+(?:was|has been|had been)\s+(?:given|granted)\b|"
                r"\bgave\s+(?:the\s+|their\s+|his\s+|her\s+|its\s+)?(?:approval|sign-?off|go-ahead)\b"),
    # 0.173.0, the review of B3 (R2): every directory action a team performs, and what a person said
    ("draft_contact_list", rf"\b(?:drafted|prepared|built|made|compiled|produced)\b{_WORDS}(?:call |contact )?list\b|"
                           rf"\b(?:call |contact )?list\b{_BEEN}(?:drafted|prepared|built|made|compiled)\b"),
    ("check_exclusions", rf"\b(?:removed|excluded|filtered out|took out)\b{_WORDS}(?:do-not-contact|exclusions?)\b|"
                         rf"\bchecked\b{_WORDS}(?:exclusions?|do-not-contact)\b|\bexclusions?\b{_BEEN}(?:checked|applied|removed)\b"),
    ("check_eligibility", rf"\bchecked\b{_WORDS}eligib\w*|\beligib\w*\b{_BEEN}checked\b"),
    ("check_campaign_dates", rf"\bchecked\b{_WORDS}(?:campaign(?:'s)? )?dates\b|\bdates\b{_BEEN}checked\b"),
    ("check_cost", rf"\bchecked\b{_WORDS}(?:cost|budget)\b|\b(?:cost|budget)\b{_BEEN}checked\b"),
    ("check_prerequisites", rf"\bchecked\b{_WORDS}prerequisites?\b|\bprerequisites?\b{_BEEN}checked\b"),
    ("present_options", rf"\bpresented\b{_WORDS}(?:options?|report|findings)\b|\b(?:options?|report)\b{_BEEN}presented\b"),
    ("record_decision", rf"\brecorded\b{_WORDS}decision\b|\bdecision\b{_BEEN}recorded\b"),
    ("run_evaluation", rf"\bevaluated\b{_WORDS}models?\b|\bmodels?\b{_BEEN}evaluated\b"),
    ("register_version", rf"\bregistered\b{_WORDS}(?:models?|versions?)\b|\b(?:models?|versions?)\b{_BEEN}registered\b"),
    ("compare_baseline", rf"\bcompared\b{_WORDS}(?:with |to |against )?(?:the |a )?baseline\b"),
    ("person_confirmed", r"\b(?P<who>[A-Z][a-z]+|the (?:person|owner|decision owner|member|requester|customer|client))\b(?:\s+[\w'-]+){0,3}?\s+"
                         r"(?:has |have |had )?(?:confirmed|agreed|accepted|said)\b|\b(?:confirmed|agreed|accepted)\s+by\s+"
                         r"(?:[A-Z][a-z]+|the (?:person|owner|decision owner|member|requester|customer|client))\b"),
    ("contact_customers", rf"\b(?:called|rang|contacted|e-?mailed|phoned|messaged|texted|reached out to|sent|told|notified|informed|offered)\b"
                          r"(?:\s+[\w'-]+){0,5}?\s+(?:to\s+)?(?:the\s+|our\s+|all\s+)?(?:[\w-]+\s+)?(?:customers?|accounts?|clients?|users?|people|them)\b|"
                          rf"\b(?:customers?|accounts?|clients?|users?|people|they)\b{_BEEN}(?:called|rang|contacted|e-?mailed|phoned|messaged|"
                          r"texted|reached|sent|told|notified|informed|offered)\b"),
)}
NEVER_RECORDED = ("train", "deploy", "approve", "contact_customers")  # no team tool performs them
# a name that is a team or the agent itself, never a person who confirmed something ("Sales confirmed the capacity")
_NOT_PEOPLE = frozenset({"i", "we", "it", "this", "that", "they", "he", "she", "you", "data", "science", "sales", "marketing", "business",
                         "lead", "engineering", "ml", "code", "dclab", "the", "a", "an", "our", "its", "their", "team", "both", "each", "all"})
PERSON = "human_response"  # what meets a person's confirmation: a finished human-bound action (a person's recorded answer)
# The directory actions each claim is met by (any of them, with a terminal record)
CLAIM_ACTIONS: dict[str, tuple[str, ...]] = {
    "delegate": ("delegate", "delegate_specialist", "delegate_analysis", "request_sales_input", "request_marketing_input", "open_case"),
    "review_evidence": ("review_evidence", "reconcile_replies", "present_options", "present_report"),
    "draft_contact_list": ("draft_contact_list", "revise_on_correction"),
    "present_options": ("present_options", "present_report"),
    "person_confirmed": (PERSON,),
}
# A claim is no claim when the words just before it deny it, ask for it, or put it in the future or a condition:
# "nothing was trained", "Sales should call them", "before anyone contacted them"
_NOT_DONE = re.compile(r"\b(?:no|not|never|nothing|none|neither|nor|without|cannot|can't|won't|n't|before|until|if|unless|"
                       r"will|would|should|could|can|may|might|must|needs?|asks?|asked to|wants?|plans?|once)\b", re.I)
# A denial inside a claim's own words ("we called no customers"). "Not just", "not only" and "no fewer than" widen the
# claim instead of denying it ("we emailed not just the top accounts but all customers"), so they never count.
_INNER_DENIAL = re.compile(r"\b(?:not|never|no)\b(?!\s+(?:just|only|merely|simply|fewer|less|more|later)\b)", re.I)


def claimed_actions(sentence: Any) -> list[str]:
    """The actions a sentence says were done (keys of :data:`CLAIMS`), in their order. A claim the words just before it
    deny, ask for or put in the future ("nothing was trained", "Sales should call them") is no claim. Words only: whether
    the action was done is the host's to check against its action log."""
    text = " ".join(str(sentence or "").split())
    out: list[str] = []
    for name, pattern in CLAIMS.items():
        for match in pattern.finditer(text):
            # a denial counts in its own clause only: "No list was drafted, but we called the customers" claims the call
            head = text[:match.start()]
            cut = max((m.end() for m in re.finditer(r"[;,:]|\bbut\b", head, re.I)), default=0)
            before = head[cut:].split()[-6:]
            if _NOT_DONE.search(" ".join(before)) or _INNER_DENIAL.search(match.group(0)):
                continue  # denied before it, or inside it ("the accounts on the list were never ... called")
            who = (match.groupdict().get("who") or "") if name == "person_confirmed" else ""
            if who and who.split()[-1].lower() in _NOT_PEOPLE:
                continue  # a team or the agent confirmed it: its own action, not a person's answer
            if name not in out:
                out.append(name)
            break
    return out


def unrecorded_claims(sentence: Any, done: Sequence[str]) -> list[str]:
    """The actions ``sentence`` claims that ``done`` (the directory actions with a terminal record; :data:`PERSON` when a
    person's answer is recorded) does not hold."""
    have = set(done)
    return [c for c in claimed_actions(sentence)
            if c in NEVER_RECORDED or not (have & ({c} | set(CLAIM_ACTIONS.get(c, ()))))]


def promises_action(sentence: Any) -> bool:
    """Whether a sentence promises an action in the agent's own name ("I will call them"): an agent acts through its tools."""
    return bool(_PROMISE.search(" ".join(str(sentence or "").split())))


AgentOutcome = Literal["acted", "fell_back", "ended", "repeat"]
AgentStop = Literal["answered", "steps", "time", "euros", "replies", "max_output_tokens", "tools_as_text", "error", "timeout", "refused",
                    "interrupted", "no_plan"]
Fallback = Literal["no_model", "no_live_models", "refused", "timeout", "invalid_output", "error", "stopped", "busy", "unsure",
                   "check_failed", "budget", "person_entry", "no_action"]
_CODE = r"^[a-z0-9_]{1,60}$"


class AgentTurnRecord(Versioned):
    """One turn of a team agent on the agent arm (0.173.0, B3): how it ended, why it stopped, its plan as a measure (the
    words stay with the host), what it did, what code refused, what its report kept, and what it cost. Codes, ids and
    measures: never the person's words, a prompt, the plan or a reply."""

    schema_version: Literal["dclab_core.team_agent_turn.v1"] = _version("team_agent_turn", 1)
    role: Role
    agent: str = Field(min_length=1, max_length=80)
    task: Optional[str] = Field(default=None, max_length=200)
    run: Optional[str] = Field(default=None, max_length=120, description="The agent's run: its steps, each an agent.step.")
    delivery: Optional[str] = Field(default=None, max_length=120, description="The delivery this turn answered (one turn per delivery).")
    outcome: AgentOutcome
    stopped: Optional[AgentStop] = None
    fallback: Optional[Fallback] = Field(default=None, description="Why the rules acted instead (outcome fell_back only).")
    plan_chars: int = Field(default=0, ge=0, le=PLAN_CHARS)
    plan_steps: int = Field(default=0, ge=0, le=200)
    plan_sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$", description="Of the plan's words, kept whole by the host.")
    steps: int = Field(default=0, ge=0, description="Tool calls the model made, the plan and reads included.")
    writes: int = Field(default=0, ge=0, description="Calls that wrote team work and ran (the guard let them through).")
    refused: int = Field(default=0, ge=0, description="Calls the guard, a check or a schema refused: nothing changed.")
    tools: list[str] = Field(default_factory=list, max_length=200, description="The tools called, in order, as codes.")
    sentences_kept: int = Field(default=0, ge=0)
    sentences_refused: int = Field(default=0, ge=0)
    refusals: list[str] = Field(default_factory=list, max_length=50, description="Each refused sentence's reason, a code of REPORT_REFUSALS.")
    euros: Optional[float] = Field(default=None, ge=0)
    seconds: Optional[float] = Field(default=None, ge=0)
    request: Optional[str] = Field(default=None, max_length=120, description="The last model request: its exchange keeps the thinking.")

    @model_validator(mode="after")
    def _rules(self):
        if (self.outcome == "fell_back") != (self.fallback is not None):
            raise ValueError("a turn names its fallback exactly when the rules acted instead")
        if self.outcome == "fell_back" and self.writes:
            raise ValueError("a turn falls back to the rules only before its first action")
        if any(not re.fullmatch(_CODE, t) for t in self.tools):
            raise ValueError("a tool is named by its code")
        if any(r not in REPORT_REFUSALS for r in self.refusals):
            raise ValueError(f"a refused sentence's reason is one of {REPORT_REFUSALS}")
        return self


for _model in (BrainSetting, SeatRecord, AgentTurnRecord):
    CONTRACTS[_model.model_fields["schema_version"].default.split(".")[1]] = _model

__all__ = ["AFTER_SEATS", "AGENT_OUTCOMES", "PERSON", "AGENT_STOPS", "ARMS", "AgentTurnRecord", "BEFORE_SEATS", "BrainSetting", "CLAIMS", "CLAIM_ACTIONS",
           "CONTRACT", "DEFAULT_ARM", "FALLBACKS", "NEVER_RECORDED", "PLAN_CHARS", "QUESTION_CHARS", "QUESTION_CHECKS", "REPORT_REFUSALS",
           "REVIEWERS", "ROLES", "ROLE_SEATS", "SEATS", "SEAT_OUTCOMES", "SOURCES", "SeatRecord", "UNSURE", "arm_of", "check_code",
           "check_question", "claimed_actions", "parse_override", "promises_action", "unrecorded_claims"]
