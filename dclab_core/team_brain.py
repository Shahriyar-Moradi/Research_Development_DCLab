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

Contract ``dclab_core.team_brain_setting.v1``.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional, Union

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Versioned, _version

CONTRACT = "dclab_core.team_brain_setting.v1"
ARMS = ("rules", "helper", "agent")
DEFAULT_ARM = "rules"
ROLES = ("lead", "business", "data_science", "ml_engineering", "sales", "marketing")  # the team agents a host may switch
REVIEWERS = ("reviewer", "owner")  # who may switch a role's brain
SOURCES = ("default", "stored", "override")  # where a turn's arm came from
# why a model arm's turn fell back to the rules ("stopped": the turn ended before the rules finished, so no model was asked)
FALLBACKS = ("no_model", "no_live_models", "refused", "timeout", "invalid_output", "error", "stopped", "busy")
SHORT = 300

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


CONTRACTS[BrainSetting.model_fields["schema_version"].default.split(".")[1]] = BrainSetting

__all__ = ["ARMS", "BrainSetting", "CONTRACT", "DEFAULT_ARM", "FALLBACKS", "REVIEWERS", "ROLES", "SOURCES", "arm_of", "parse_override"]
