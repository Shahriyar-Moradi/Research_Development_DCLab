"""The model switches when the measurement suites are not here (the open edition, 2026-10-21).

In the full edition a model serves a role only after a sealed measurement passed (dclab_rnd.agent_eval). The open edition
has no sealed sets; there a role that acts through tools starts on its rules, and serves the configured model when:

- ``DCLAB_AGENT_MODE=model`` (or ``rules``) says so for every role, or
- the person trusted it: ``dclab models trust <role|all>`` writes ``gates.local.json`` in the workspace folder, or
- ``dclab doctor``'s probe found the model calls tools natively (``"probe": {"tools": true, "model": ..., "endpoint": ...}``
  in the same file), for the model and endpoint configured now.

Nothing here is a measurement: the studio shows each role as "model (probed)", "model (trusted by you)" or "rules".
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

FILE = "gates.local.json"
ROLES = ("built_chat", "map_roles", "conversation_reader")


def _home(home: Optional[Path] = None) -> Path:
    if home is not None:
        return Path(home)
    from .settings import Settings

    return Path(Settings.load().agent_home)


def read(home: Optional[Path] = None) -> dict[str, Any]:
    try:
        doc = json.loads((_home(home) / FILE).read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def write(doc: dict[str, Any], home: Optional[Path] = None) -> Path:
    path = _home(home) / FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def why(role: str, home: Optional[Path] = None) -> str:
    """How the role is served here: "model (all, by DCLAB_AGENT_MODE)", "model (trusted by you)", "model (probed)" or "rules"."""
    mode = os.environ.get("DCLAB_AGENT_MODE", "").strip().lower()
    if mode in ("model", "rules"):
        return f"{mode} (DCLAB_AGENT_MODE)"
    doc = read(home)
    trusted = set(doc.get("trusted") or [])
    if role in trusted or "all" in trusted:
        return "model (trusted by you)"
    probe = doc.get("probe") or {}
    if probe.get("tools") is True and probe.get("model") and {k: probe.get(k) for k in ("model", "endpoint")} == probed_as():
        return "model (probed)"  # a probe holds for the model and endpoint it tried, never for the next one configured
    return "rules"


def probed_as() -> dict[str, Any]:
    """The model a probe is about: the standard tier's model and endpoint (never its key)."""
    try:
        from .models import settings as model_settings

        tier = model_settings.public("standard")
        return {"model": tier.get("model") or "", "endpoint": tier.get("endpoint") or ""}
    except Exception:  # noqa: BLE001 - no model layer: nothing was probed
        return {"model": "", "endpoint": ""}


def passed(role: str, home: Optional[Path] = None) -> bool:
    """Whether the model serves this role here (see the module's doc)."""
    try:
        return why(role, home).startswith("model")
    except Exception:  # noqa: BLE001 - a switch that cannot be read: the rules serve
        return False


def trust(roles: list[str], home: Optional[Path] = None) -> dict[str, Any]:
    doc = read(home)
    doc["trusted"] = sorted(set(doc.get("trusted") or []) | set(roles))
    write(doc, home)
    return doc


def record_probe(probe: dict[str, Any], home: Optional[Path] = None) -> dict[str, Any]:
    doc = read(home)
    doc["probe"] = probe
    write(doc, home)
    return doc


LABELS = {"built_chat": "The project chat's team", "map_roles": "The problem map's roles", "conversation_reader": "Home's conversation reader"}
SWITCHES = {"map_roles": "DCLAB_MAP_ROLES", "conversation_reader": "DCLAB_CONVERSATION_READER"}  # roles that also need their own switch


def status(home: Optional[Path] = None) -> list[dict[str, Any]]:
    """For each agent that acts through tools: does it run on the configured model now, and why. It asks the same functions
    the agents ask (so the page never disagrees with them): the sealed measurements where they exist (the full edition),
    else this module's switches."""
    from .extensions import optional
    from .models import settings as model_settings

    configured = bool(model_settings.public("standard").get("available"))
    measured = optional("agent_eval") is not None
    out = []
    for role in ROLES:
        switched_off = False
        try:
            if role == "built_chat":
                from .draft.orchestrator import _model_serves

                serves = _model_serves()
            elif role == "map_roles":
                from .draft import map as problem_map

                switched_off = not problem_map.switched_on()  # draft.map.advisor_for asks the switch and the measurement
                serves = not switched_off and problem_map.roles_passed()
            else:
                from .draft import reader

                switched_off = not reader.switched_on()
                serves = not switched_off and (passed(role, home) if not measured else optional("agent_eval.journeys").reader_passed())
        except Exception:  # noqa: BLE001 - a switch that cannot be read: the rules serve
            serves = False
        if not configured:
            serves, reason = False, "rules: no model is configured"
        elif switched_off:  # off until it is switched on, whatever a probe or a measurement says
            reason = f"rules: switched off ({SWITCHES[role]}=1 turns it on)"
        elif measured:
            reason = "model: it passed its sealed measurement" if serves else "rules: it has not passed its sealed measurement"
        else:
            reason = why(role, home)
            if not serves and reason.startswith("model"):  # the reason never says model when the rules serve
                reason = "rules"
        out.append({"role": role, "label": LABELS.get(role, role), "serves": "model" if serves else "rules", "why": reason})
    return out


__all__ = ["ROLES", "passed", "why", "trust", "record_probe", "read", "probed_as", "status"]
