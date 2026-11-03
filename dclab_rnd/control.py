"""The owner's control switches, kept in the workspace (A3 of the 2026-11-03 plan): stop every live model call, and a monthly euro cap.

Until now these were environment settings only (``DCLAB_NO_LIVE_MODELS``, ``DCLAB_WORKSPACE_MONTHLY_EUR``): changing one meant
editing ``.env`` and restarting. ``<workspace>/control.json`` holds the same two switches so the Admin page's Control tab can
change them at once; an environment setting, when present, still wins (a deployment pins what it pins).

- ``live_models`` false: no client to a real provider is created (``models.live_guard.check`` refuses it); the rules serve,
  as with no model configured.
- ``workspace_monthly_eur``: the gateway refuses a request that would pass it (``models.gateway.workspace_cap``).

Every change is audited by the route that makes it (kind ``admin``). Read with the file's modification time as a cache key, so a
change is seen by every process on its next request.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

FILE = "control.json"
DEFAULTS: dict[str, Any] = {"live_models": True, "workspace_monthly_eur": None}
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def home_of(home: Any = None) -> Optional[Path]:
    if home is not None:
        return Path(home)
    raw = os.environ.get("DCLAB_AGENT_HOME")
    if raw:
        return Path(raw)
    try:
        from .settings import default_home

        return Path(default_home())
    except Exception:  # noqa: BLE001
        return None


def get(home: Any = None) -> dict[str, Any]:
    folder = home_of(home)
    path = folder / FILE if folder else None
    if path is None or not path.is_file():
        return dict(DEFAULTS)
    stamp = path.stat().st_mtime
    hit = _CACHE.get(str(path))
    if hit and hit[0] == stamp:
        return dict(hit[1])
    try:
        doc = {**DEFAULTS, **json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        doc = dict(DEFAULTS)  # a broken file is the defaults, never an error on every request
    _CACHE[str(path)] = (stamp, doc)
    return dict(doc)


def set_values(home: Any, **values: Any) -> dict[str, Any]:
    """Change the switches given (``live_models`` a flag, ``workspace_monthly_eur`` a positive number or None)."""
    folder = home_of(home)
    if folder is None:
        raise ValueError("no workspace folder")
    doc = get(folder)
    if "live_models" in values:
        doc["live_models"] = bool(values["live_models"])
    if "workspace_monthly_eur" in values:
        cap = values["workspace_monthly_eur"]
        if cap is not None and (not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap <= 0):
            raise ValueError("the monthly cap is a positive number of euro, or empty for none")
        doc["workspace_monthly_eur"] = float(cap) if cap is not None else None
    doc["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{FILE}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    tmp.replace(folder / FILE)
    return doc


def live_allowed(home: Any = None) -> bool:
    return bool(get(home).get("live_models", True))


def monthly_cap(home: Any = None) -> Optional[float]:
    cap = get(home).get("workspace_monthly_eur")
    return float(cap) if isinstance(cap, (int, float)) and not isinstance(cap, bool) and cap > 0 else None


__all__ = ["get", "set_values", "live_allowed", "monthly_cap", "FILE"]
