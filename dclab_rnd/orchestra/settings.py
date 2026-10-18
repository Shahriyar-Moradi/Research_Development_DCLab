"""The team's switch and budgets (package 28.8, from 28.3): one workspace file, changed only by an admin, every change audited.

``on``: Home's Build (the button, "build it" in the chat, the model's build tool) goes through the understanding team's
walk, the build being the walk's approval; off, Home builds as before and the team is reached only through its own API.
The wizard's build (``POST /api/drafts/{id}/build``) stays direct whatever the switch: there the person sets the solution
field by field, and a tester's CLI calling it builds what it was told (the second review of 28.8: said, not implied).
``run_eur``: the most a walk (or a solution team's run) may spend on model requests, across every resume; ``node_eur``:
the most one node may spend; ``max_steps``: the walk's step budget. A walk that reaches a budget stops with a blocker the
person sees; a node whose model reaches its cap falls back to the rules. Defaults are off and small: the team serves only
when the owner turns it on, and its models only after its sealed gate (``service.team_passed``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SETTINGS_FILE = "team.json"
DEFAULTS: dict[str, Any] = {"on": False, "run_eur": 0.5, "node_eur": 0.2, "max_steps": 40}
LIMITS = {"run_eur": (0.01, 20.0), "node_eur": (0.01, 5.0), "max_steps": (5, 200)}


def settings(home: Path) -> dict[str, Any]:
    """The workspace's team settings, a key the file lacks (or holds with the wrong type) at its default."""
    path = Path(home) / SETTINGS_FILE
    try:
        stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except ValueError:
        stored = {}
    out = {}
    for key, default in DEFAULTS.items():
        value = stored.get(key, default)
        if isinstance(default, bool):
            out[key] = value if isinstance(value, bool) else default
        elif isinstance(default, int) and not isinstance(default, bool):
            out[key] = value if isinstance(value, int) and not isinstance(value, bool) else default
        else:
            out[key] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
    return out


def set_settings(home: Path, changes: dict[str, Any]) -> dict[str, Any]:
    """Change some settings: only the known keys, a switch as true or false, a budget within its bounds (ValueError otherwise).
    Every other key in the file is kept."""
    current = settings(home)
    for key, value in changes.items():
        if key not in DEFAULTS:
            raise ValueError(f"{key} is not a team setting; one of {', '.join(DEFAULTS)}")
        default = DEFAULTS[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} is true or false")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} is a number")
            low, high = LIMITS[key]
            if not low <= float(value) <= high:
                raise ValueError(f"{key} is between {low} and {high}")
            if isinstance(default, int) and float(value) != int(value):
                raise ValueError(f"{key} is a whole number")
            value = int(value) if isinstance(default, int) else float(value)
        current[key] = value
    path = Path(home) / SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2), encoding="utf-8")
    tmp.replace(path)
    return current
