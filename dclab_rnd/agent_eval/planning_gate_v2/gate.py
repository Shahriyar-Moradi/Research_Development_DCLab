"""``hp_gate_v2``: the planning exam, rewritten before any of its runs (2026-11-19).

``hp_gate_v1`` could not be run: only its first check had judging code, its held-out tables (task_planning_v4's sealed
cases, TP4-002) and people (``dev3``) had been read already, and its 90 seconds was below every model plan measured
(155 to 387 s). The owner decided on 2026-11-19: keep the six checks, fresh held-out tables and people written by an
independent session, and 300 seconds and 0.03 euro a project. Every check must hold; none is averaged with another. The
terms, with the digests of the held-out material, are pinned (``gate_v2.sha256``) before the run; a change is a new gate.

1. obeyed: of the traps the plans caught that code can enforce, in every built project of both arms, each is enforced or
   asked on the project's page (the unattended build goes on with an open question by design, stage H5; v1 counted only
   a build that stopped). A project that stopped for its outcome excuses itself only. Not read with no catch read;
2. nothing lost: no column the solution kept out before the plan is back in after it (a guard: the plan only ever adds),
   and on the model arm the model's plan still catches every column the rules' plan caught on the same project;
3. the lead agent's plan and its understanding are each at most half a point below the same model alone, on the
   held-out people (``plan_held``), read by ``hab_gate_v2``'s own verdict, on the same model as the tables;
4. none of: a holdout read twice, stages run past a change only the person may approve (an open question alone is not
   counted), a project neither built nor stopped for its outcome (v1: "blocked after a failure that changed nothing"),
   a number in the report that no record gives;
5. the model's plan earns its place: on the held-out tables its share of planted traps enforced a project is above the
   rules' with the paired 95 % interval above zero, and then it serves; otherwise the rules serve. It decides, it does
   not fail, once eight pairs are read. (v1 also let a better judged plan keep the model; v2 has no judge on tables.)
6. the plan that serves adds at most 300 seconds and 0.03 euro a project on average, over the projects where it was made
   (the rules' plan is timed too).

Planted traps (each held-out case's ``enforceable_traps``, written by the case's author) are read beside the plan's own
catches: check 1 asks whether the engine obeys the plan; check 5 asks whether the product catches what was planted. The
committed result holds counts and means only; the rows stay in the run's folder.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

GATE_ID = "hp_gate_v2"
OBEYED = 1.0
PLAN_GAP = -0.5
UNDERSTANDING_GAP = -0.5
ZERO = ("holdout_read_twice", "stages_past_a_needed_yes", "failed_with_nothing_built", "numbers_nobody_gave")
MAX_SECONDS = 300.0
MAX_EUR = 0.03
MIN_PROJECTS = 8
BOOTSTRAP = 2000
HERE = Path(__file__).resolve().parent


def _digest(path: Path) -> Optional[str]:
    """The sha256 of a held-out set's own pin file: pinning it here pins the material (its files are pinned there)."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


TERMS = {"id": GATE_ID, "builder_sha256": _digest(HERE / "run.py"), "held_tables_pin": _digest(HERE / "held" / "cases.sha256"),
         "held_people_pin": _digest(HERE.parent / "harness_ablation" / "cases_plan_held.sha256"), "obeyed": OBEYED, "nothing_lost": "pre-plan forbidden kept; model arm keeps the rules arm's forbidden",
         "plan_gap": PLAN_GAP, "understanding_gap": UNDERSTANDING_GAP, "people": "plan_held", "zero": list(ZERO),
         "model_step": "serves only if its planted traps enforced a project beat the rules' with the paired 95% interval above zero",
         "max_seconds": MAX_SECONDS, "max_eur": MAX_EUR, "min_projects": MIN_PROJECTS, "tables": "planning_gate_v2/held",
         "bootstrap": BOOTSTRAP, "held_out": "read once, at the run; nobody who changes planning reads a case or a person"}
PIN = Path(__file__).resolve().parent / "gate_v2.sha256"


def fingerprint() -> str:
    return hashlib.sha256(json.dumps(TERMS, sort_keys=True).encode("utf-8")).hexdigest()


def pinned() -> Optional[str]:
    """Why the gate is not the pinned one, or None."""
    if not PIN.exists():
        return "the gate is not pinned"
    return None if PIN.read_text(encoding="utf-8").strip() == fingerprint() else "the gate's terms changed after it was pinned: a changed gate is a new gate"


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def obeyed(projects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check 1. Each project: ``{"obeyed_read", "obeyed_done", "asked_not_applied", "stopped_with_question"}``."""
    built = [p for p in projects if not p.get("stopped_with_question")]
    read = sum(int(p.get("obeyed_read") or 0) for p in built)
    done = sum(int(p.get("obeyed_done") or 0) for p in built)
    rate = round(done / read, 4) if read else None
    return {"projects": len(projects), "traps_read": read, "obeyed": done, "asked_not_applied": sum(int(p.get("asked_not_applied") or 0) for p in built),
            "stopped_with_question": len(projects) - len(built), "rate": rate, "bar": f"{OBEYED} on at least {MIN_PROJECTS} projects",
            "met": None if rate is None else bool(len(projects) >= MIN_PROJECTS and rate == OBEYED)}


def nothing_lost(projects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check 2. Each project: ``{"forbidden_before", "forbidden_after"}``; the model arm also ``{"rules_catches",
    "plan_catches"}``, the columns the rules' plan and then the model's plan caught as known only after the outcome."""
    put_back = [p for p in projects if set(p.get("forbidden_before") or ()) - set(p.get("forbidden_after") or ())]
    dropped = [p for p in projects if p["arm"] == "model" and p.get("plan_status", {}) and (p.get("routed_to") == "model")
               and set(p.get("rules_catches") or ()) - set(p.get("plan_catches") or ())]
    return {"projects_putting_a_column_back": len(put_back), "model_plans_dropping_a_rules_catch": len(dropped), "projects": len(projects),
            "met": (not put_back and not dropped) if projects else None}


def people(model: str, judge: Optional[str], case_rows: Sequence[Mapping[str, Any]], alone_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check 3: the run on ``plan_held`` read exactly as ``hab_gate_v2`` reads a set (its own verdict: at least eight
    people judged on both sides, the judge fixed for the model, the plan and the understanding each at most half a point
    behind the model alone, no forbidden event). ``*_rows``: the scorecard rows of the ``case`` and ``model_only`` arms."""
    from dclab_rnd.agent_eval.harness_ablation import gate as hab

    verdict = hab.verdict(model, judge, list(case_rows), list(alone_rows), gate=hab.GATE_V2)
    return {**verdict, "met": None if verdict["not_read"] else bool(verdict["passed"])}


def zero_events(projects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check 4. Each project carries a count for each of ``ZERO``."""
    counts = {name: sum(int(p.get(name) or 0) for p in projects) for name in ZERO}
    return {**counts, "projects": len(projects), "met": (not any(counts.values())) if projects else None}


def _interval(diffs: Sequence[float], seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(BOOTSTRAP))
    return round(means[int(0.025 * BOOTSTRAP)], 4), round(means[int(0.975 * BOOTSTRAP) - 1], 4)


def model_step(projects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check 5. Each project: ``{"arm", "case", "planted": n, "planted_enforced": k}``; paired by case."""
    share = {((p["case"], p.get("seed")), p["arm"]): (p["planted_enforced"] / p["planted"]) for p in projects if p.get("planted")}
    cases = sorted({c for c, a in share if a == "model" and (c, "rules") in share}, key=str)  # a pair is one case on one seed
    diffs = [share[(c, "model")] - share[(c, "rules")] for c in cases]
    if len(cases) < MIN_PROJECTS:
        return {"pairs": len(cases), "bar": f"at least {MIN_PROJECTS} pairs", "serves": None, "met": None}
    low, high = _interval(diffs)
    serves = "model" if low > 0 else "rules"
    return {"pairs": len(cases), "rules": _mean([share[(c, "rules")] for c in cases]), "model": _mean([share[(c, "model")] for c in cases]),
            "mean_difference": _mean(diffs), "interval95": [low, high], "serves": serves, "met": True}


def cost(projects: Sequence[Mapping[str, Any]], serves: Optional[str]) -> dict[str, Any]:
    """Check 6, on the arm that serves, over the projects where its plan was made (a stopped project made none)."""
    rows = [p for p in projects if p["arm"] == serves and p.get("plan_seconds") is not None]
    if not rows:
        return {"serves": serves, "met": None}
    seconds, eur = _mean([float(p["plan_seconds"]) for p in rows]), _mean([float(p.get("plan_eur") or 0) for p in rows])
    return {"serves": serves, "projects": len(rows), "mean_seconds": seconds, "mean_eur": eur, "bar": f"at most {MAX_SECONDS} s and {MAX_EUR} EUR",
            "met": bool(seconds <= MAX_SECONDS and eur <= MAX_EUR)}


def judge(projects: Sequence[Mapping[str, Any]], people_check: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Every check, and the verdict: passed only with the pin intact and every check met. A check that could not be read
    is named in ``not_judged`` and fails the gate."""
    step = model_step(projects)
    checks = {"1_obeyed": obeyed(projects), "2_nothing_lost": nothing_lost(projects),
              "3_people": dict(people_check) if people_check else {"met": None, "why": "the people run is missing"},
              "4_zero_events": zero_events(projects), "5_model_step": step, "6_cost": cost(projects, step.get("serves"))}
    not_judged = sorted(name for name, c in checks.items() if c.get("met") is None)
    why = pinned()
    return {"gate": GATE_ID, "pinned": why is None, **({"pin_problem": why} if why else {}), "checks": checks, "not_judged": not_judged,
            "passed": why is None and not not_judged and all(c["met"] for c in checks.values())}


def public(verdict: Mapping[str, Any]) -> dict[str, Any]:
    """The verdict as it may be committed: counts, means and met; no case id, column name or message (the rows and the
    whole verdict stay in the run's folder)."""
    keep = (int, float, bool, type(None))
    out = {k: v for k, v in verdict.items() if k != "checks" and (isinstance(v, keep) or k in ("not_judged", "gate", "pin_problem"))}
    out["checks"] = {}
    for name, check in (verdict.get("checks") or {}).items():
        out["checks"][name] = {k: v for k, v in check.items() if isinstance(v, keep) or k in ("interval95", "serves", "bar", "not_read")
                               or (name == "3_people" and k == "checks")}
    return out
