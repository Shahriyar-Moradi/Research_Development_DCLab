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

Contracts ``dclab_core.team_brain_setting.v1`` and ``dclab_core.team_helper_seat.v1``.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping, Optional, Sequence, Union

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Versioned, _version

CONTRACT = "dclab_core.team_brain_setting.v1"
ARMS = ("rules", "helper", "agent")
DEFAULT_ARM = "rules"
ROLES = ("lead", "business", "data_science", "ml_engineering", "sales", "marketing")  # the team agents a host may switch
REVIEWERS = ("reviewer", "owner")  # who may switch a role's brain
SOURCES = ("default", "stored", "override")  # where a turn's arm came from
# why a model arm's turn fell back to the rules ("stopped": the turn ended before the rules finished, so no model was asked).
# 0.172.0 (B2): a helper seat also falls back when the model said "unsure" or its answer failed the seat's code check
FALLBACKS = ("no_model", "no_live_models", "refused", "timeout", "invalid_output", "error", "stopped", "busy", "unsure",
             "check_failed")
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
SEAT_OUTCOMES = ("used", *FALLBACKS)  # "used": the check passed and the model's answer was used
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


for _model in (BrainSetting, SeatRecord):
    CONTRACTS[_model.model_fields["schema_version"].default.split(".")[1]] = _model

__all__ = ["AFTER_SEATS", "ARMS", "BEFORE_SEATS", "BrainSetting", "CONTRACT", "DEFAULT_ARM", "FALLBACKS", "QUESTION_CHARS", "QUESTION_CHECKS",
           "REVIEWERS", "ROLES", "ROLE_SEATS", "SEATS", "SEAT_OUTCOMES", "SOURCES", "SeatRecord", "UNSURE", "arm_of", "check_code",
           "check_question", "parse_override"]
