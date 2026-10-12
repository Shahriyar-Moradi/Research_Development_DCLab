"""Which of dclab_core's roles the product uses by default (package 18.9), and the advisor it gives them.

A role is on only where the result its gate was judged on passed (``roles_judged.json``, made by ``make roles-judged``
from the recorded measurements); the others stay available and off. The owner can override: ``DCLAB_ROLES`` is
``judged`` (the default), ``none``, or a comma-separated list of roles to switch on (``problem_analyst,reporter``).
A role gets an advisor only when it is on, a model serves its purpose, and the data is not private; otherwise it runs
its rules, as every role does without a model.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

JUDGED = Path(__file__).resolve().parent / "roles_judged.json"
CALL_CAP_EUR = 0.05  # each product call of a role (one understanding, one plan adoption, one report) has its own small cap
# the roles the product's flows ask (18.9); the critic and the brief checker failed their gates and are not wired
WIRED = ("problem_analyst", "data_investigator", "solution_planner", "feature_engineer", "reporter")


@lru_cache(maxsize=1)
def judged() -> dict[str, bool]:
    try:
        rows = json.loads(JUDGED.read_text(encoding="utf-8")).get("roles", [])
    except (OSError, ValueError):
        return {}
    return {r["role"]: bool(r.get("on")) for r in rows}


def on(role: str) -> bool:
    """Whether ``role`` is on: the owner's override (``DCLAB_ROLES``), else its gate's verdict."""
    chosen = (os.environ.get("DCLAB_ROLES") or "judged").strip().lower()
    if chosen == "none":
        return False
    if chosen != "judged":
        return role in {r.strip() for r in chosen.split(",") if r.strip()}
    return judged().get(role, False)


def advisor(role: str, *, project: Optional[dict[str, Any]] = None, draft_id: Optional[str] = None,
            cap_eur: Optional[float] = None) -> Any:
    """The gateway advisor for ``role``, or None (the role is off, the project is private, or no model serves it)."""
    if not on(role):
        return None
    if project is not None and (project.get("settings") or {}).get("private"):
        return None
    from .core_advisor import GatewayAdvisor
    from .models import installed

    gateway = installed()
    if not gateway.available(role):
        return None
    return GatewayAdvisor(gateway, project_id=(project or {}).get("id"), draft_id=draft_id, max_tokens=3000, cap_eur=cap_eur)


def status() -> list[dict[str, Any]]:
    """Each role, whether it is on and why (for the Admin and Learn pages)."""
    chosen = (os.environ.get("DCLAB_ROLES") or "judged").strip().lower()
    return [{"role": role, "on": on(role) and role in WIRED, "judged_on": passed, "wired": role in WIRED,
             "why": ("not wired into the product's flows (its gate did not pass)" if role not in WIRED else
                     "its gate passed" if passed and chosen == "judged" else "its gate, or the whole layer's, did not pass"
                     if chosen == "judged" else f"DCLAB_ROLES={chosen}")}
            for role, passed in sorted(judged().items()) if role not in ("team", "layer")]
