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


__all__ = ["ROLES", "passed", "why", "trust", "record_probe", "read", "probed_as"]
