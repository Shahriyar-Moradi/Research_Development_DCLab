"""``hp_gate_v3``: the planning exam after the practice set's fixes (2026-11-19), terms fixed before its run.

v2's six checks, judged by v2's own functions (their code is pinned here by its hash), on fresh held-out tables and
people written by an independent session, plus two checks for what v2 could not see:

7. caught: the plan that serves enforces at least half of the traps the case authors planted, over every built
   project of its arm (v2: 0.29 for the rules; the practice set after the fixes: 0.58, read by the fixer);
8. no needless entity split: no project keeps an entity apart where the case says the same entities recur and the
   brief asks for no new ones (an entity split there throws away rows and tests a question nobody asked).

Every check must hold; none is averaged with another. The committed result holds counts and means only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..planning_gate_v2 import gate as v2

GATE_ID = "hp_gate_v3"
CAUGHT = 0.5
MIN_PROJECTS = v2.MIN_PROJECTS
people = v2.people  # check 3, read as hab_gate_v2 reads a set
HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "planning_gate_v2"


def _digest(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


TERMS = {"id": GATE_ID, "checks_1_to_6": "hp_gate_v2's functions", "checks_sha256": _digest(V2 / "gate.py"),
         "builder_sha256": _digest(V2 / "run.py"), "reader_sha256": _digest(HERE / "run.py"),
         # the review: v3's own checks and commands, and the code that decides enforced, obeyed and a made-up number
         "gate_sha256": _digest(HERE / "gate.py"), "cli_sha256": _digest(HERE / "__init__.py"),
         "enforced_sha256": _digest(HERE.parents[1] / "tasks" / "enforced.py"), "unsaid_sha256": _digest(HERE.parent / "e2e" / "score.py"),
         "people_gate_sha256": _digest(HERE.parent / "harness_ablation" / "gate.py"),
         "held_tables_pin": _digest(HERE / "held" / "cases.sha256"),
         "held_people_pin": _digest(HERE.parent / "harness_ablation" / "cases_plan_held_v3.sha256"), "people": "plan_held_v3",
         "obeyed": v2.OBEYED, "plan_gap": v2.PLAN_GAP, "understanding_gap": v2.UNDERSTANDING_GAP, "zero": list(v2.ZERO),
         "max_seconds": v2.MAX_SECONDS, "max_eur": v2.MAX_EUR, "min_projects": v2.MIN_PROJECTS, "bootstrap": v2.BOOTSTRAP,
         "caught": CAUGHT, "needless_group": 0,
         "held_out": "read once, at the run; nobody who changes planning reads a case or a person"}
PIN = HERE / "gate_v3.sha256"


def fingerprint() -> str:
    return hashlib.sha256(json.dumps(TERMS, sort_keys=True).encode("utf-8")).hexdigest()


def pinned() -> Optional[str]:
    if not PIN.exists():
        return "the gate is not pinned"
    return None if PIN.read_text(encoding="utf-8").strip() == fingerprint() else "the gate's terms changed after it was pinned: a changed gate is a new gate"


def caught(projects: Sequence[Mapping[str, Any]], serves: Optional[str]) -> dict[str, Any]:
    """Check 7: of every planted trap in the serving arm's built projects, the share enforced; and by kind, for both arms."""
    def share(arm: str) -> tuple[int, int]:
        rows = [p for p in projects if p["arm"] == arm and not p.get("stopped_with_question")]
        return sum(int(p.get("planted_enforced") or 0) for p in rows), sum(int(p.get("planted") or 0) for p in rows)

    by_kind: dict[str, dict[str, list[int]]] = {}
    for p in projects:
        if p.get("stopped_with_question"):
            continue
        for kind, (n, k) in (p.get("planted_by_kind") or {}).items():
            got = by_kind.setdefault(kind, {}).setdefault(p["arm"], [0, 0])
            got[0] += k
            got[1] += n
    out: dict[str, Any] = {"by_kind": {k: {arm: (round(a / b, 4) if b else None) for arm, (a, b) in arms.items()} for k, arms in by_kind.items()},
                           "bar": f"at least {CAUGHT} for the serving plan"}
    for arm in ("rules", "model"):
        done, total = share(arm)
        out[arm] = round(done / total, 4) if total else None
    rate = out.get(serves) if serves else None
    out.update(serves=serves, met=None if rate is None else bool(rate >= CAUGHT))
    return out


def needless_group(projects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check 8, read only on projects whose case names a column that needs no entity split (none: not judged)."""
    applies = [p for p in projects if p.get("needless_group") is not None and p.get("no_group_case")]
    n = sum(int(p.get("needless_group") or 0) for p in applies)
    return {"projects_with_a_needless_group": n, "projects_it_applies_to": len(applies), "projects": len(projects),
            "met": (n == 0) if applies else None}


def judge(projects: Sequence[Mapping[str, Any]], people_check: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    verdict = v2.judge(projects, people_check)  # checks 1 to 6, as v2 judges them
    serves = verdict["checks"]["5_model_step"].get("serves")
    verdict["checks"]["7_caught"] = caught(projects, serves)
    verdict["checks"]["8_no_needless_group"] = needless_group(projects)
    why = pinned()
    verdict.update(gate=GATE_ID, pinned=why is None)
    verdict.pop("pin_problem", None)
    if why:
        verdict["pin_problem"] = why
    verdict["not_judged"] = sorted(name for name, c in verdict["checks"].items() if c.get("met") is None)
    verdict["passed"] = why is None and not verdict["not_judged"] and all(c["met"] for c in verdict["checks"].values())
    return verdict


def public(verdict: Mapping[str, Any]) -> dict[str, Any]:
    out = v2.public(verdict)
    out["checks"]["7_caught"] = {k: v for k, v in (verdict["checks"].get("7_caught") or {}).items()}  # rates by kind and arm: no case
    return out
