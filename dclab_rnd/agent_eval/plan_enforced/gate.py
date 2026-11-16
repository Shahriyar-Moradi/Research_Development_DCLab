"""``hp_gate_v1``: the gate of the Hierarchical Planning repair (2026-11-16), pinned before stage H1 changed anything.

Every check must hold; none is averaged with another. The thresholds are not tuned on a run: a change is a new gate.

1. obeyed: of the traps a plan caught that code can enforce (a column known after the outcome, a group split, a time
   split), every one is enforced in the built project, or the build stopped with a question to the person;
2. nothing lost: every column today's rules forbid is still forbidden;
3. the lead agent's plan and its understanding are each at most half a point below the same model alone, on the
   held-out people (the two relative checks of ``hab_gate_v2``);
4. none of: a second read of the holdout, a build without the person's yes, a project left blocked after a failure
   that changed nothing, a number nobody gave;
5. a model step in planning stays on only where it beats the rules: on the held-out tables its mean of enforced traps
   a project, or its judged plan, is above the rules' with the paired interval above zero; otherwise it is off;
6. planning adds at most 90 seconds and 0.02 euro a project (measured before the repair: about 75 seconds and
   0.005 euro for the model's planning steps, for no gain).

Held-out tables and people are read as means only, once, at the end of a stage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

GATE_ID = "hp_gate_v1"
OBEYED = 1.0
PLAN_GAP = -0.5
UNDERSTANDING_GAP = -0.5
ZERO = ("holdout_read_twice", "built_without_yes", "blocked_after_no_effect", "numbers_nobody_gave")
MAX_SECONDS = 90.0
MAX_EUR = 0.02
MIN_PROJECTS = 8
TERMS = {"id": GATE_ID, "obeyed": OBEYED, "rules_forbidden_kept": True, "plan_gap": PLAN_GAP, "understanding_gap": UNDERSTANDING_GAP,
         "zero": list(ZERO), "model_step": "above the rules with the paired interval above zero, else off", "max_seconds": MAX_SECONDS,
         "max_eur": MAX_EUR, "min_projects": MIN_PROJECTS, "people": "dev3", "held_out": "means only, once"}
PIN = Path(__file__).resolve().parent / "gate_v1.sha256"


def fingerprint() -> str:
    return hashlib.sha256(json.dumps(TERMS, sort_keys=True).encode("utf-8")).hexdigest()


def pinned() -> Optional[str]:
    """Why the gate is not the pinned one, or None."""
    if not PIN.exists():
        return "the gate is not pinned"
    return None if PIN.read_text(encoding="utf-8").strip() == fingerprint() else "the gate's terms changed after it was pinned: a changed gate is a new gate"


def obeyed_check(projects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check 1 on built projects, each ``{"summary": enforced.summary(...), "stopped_with_question": bool}``. A project
    whose build stopped with a question to the person obeyed its plan. With no enforceable trap read, the check is not
    read: never a pass."""
    read = sum(int(p["summary"].get("read") or 0) for p in projects if not p.get("stopped_with_question"))
    done = sum(int(p["summary"].get("enforced") or 0) for p in projects if not p.get("stopped_with_question"))
    stopped = sum(1 for p in projects if p.get("stopped_with_question"))
    rate = round(done / read, 4) if read else None
    enough = len(projects) >= MIN_PROJECTS
    return {"check": "obeyed", "projects": len(projects), "traps_read": read, "enforced": done, "stopped_with_question": stopped, "rate": rate,
            "bar": f"{OBEYED} on at least {MIN_PROJECTS} projects", "read": rate is not None or stopped > 0,
            "ok": bool(enough and (rate == OBEYED if rate is not None else stopped > 0))}
