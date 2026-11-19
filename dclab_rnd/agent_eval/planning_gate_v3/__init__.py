"""The planning exam ``hp_gate_v3`` (2026-11-19): v2's checks plus two, fresh held-out tables and people, one run.

    python -m dclab_rnd.agent_eval.planning_gate_v3 plan                       # the terms, the pin, the size, the cost
    python -m dclab_rnd.agent_eval.planning_gate_v3 pin                        # pin the terms (once, before any run)
    python -m dclab_rnd.agent_eval.planning_gate_v3 tables --yes --cap-eur 0.5 # build every held-out case with both arms, once
    python -m dclab_rnd.agent_eval.harness_ablation --set plan_held_v3 --arms case,model_only --yes --cap-eur 0.5   # the people
    python -m dclab_rnd.agent_eval.planning_gate_v3 judge --people HAB-0NN     # the verdict, into the campaign

The held-out cases are read only by ``tables``, once. Its rows (case by case) stay in the run's folder under
``agent_runs/live_tests/``, never committed, and the result keeps the checks' counts and means only, so nobody who
changes the planning reads a held-out case. No Claude model, at any provider (the owner, 2026-11-15).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import gate

CAMPAIGN = "planning_gate_v3"
MODEL = "qwen/qwen3.8-flash"  # the product's model (the owner's switch to Qwen, 2026-11-11)
ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "evidence" / "campaigns" / CAMPAIGN / "results"
RUN_ROOT = ROOT / "agent_runs" / "live_tests"


def _runs() -> list[Path]:
    return sorted(RUN_ROOT.glob("*-planning-gate-v3-tables"))


def _cmd_plan() -> int:
    from .run import load_held

    held = Path(__file__).resolve().parent / "held"
    state: dict[str, Any] = {"gate": gate.GATE_ID, "terms": gate.TERMS, "pin": gate.pinned() or "pinned", "model": MODEL,
                             "tables_runs": [p.name for p in _runs()], "results": sorted(p.name for p in RESULTS.glob("*.json"))}
    if (held / "cases.json").exists():
        cases, seeds, _ = load_held(held)  # counts only: no case is shown
        state.update(held_cases=len(cases), seeds=len(seeds), projects=2 * len(cases) * len(seeds),
                     estimate_eur=round(0.015 * len(cases) * len(seeds), 2), estimate_minutes=round(len(cases) * len(seeds) * (2 * 3 + 5)))
    else:
        state["held_cases"] = "not written yet"
    print(json.dumps(state, indent=1))
    return 0


def _cmd_pin() -> int:
    if not (gate.TERMS.get("held_tables_pin") and gate.TERMS.get("held_people_pin")):
        print("refused: the held-out tables and people are not written yet; the pin must cover them", file=sys.stderr)
        return 2
    why = _people_registered()
    if why:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    if gate.PIN.exists():
        print("refused: the gate is pinned already; a change is a new gate", file=sys.stderr)
        return 2
    gate.PIN.write_text(gate.fingerprint() + "\n", encoding="utf-8")
    print(f"pinned {gate.GATE_ID}: {gate.fingerprint()}")
    return 0


MARKER = ROOT / "evidence" / "campaigns" / CAMPAIGN / "tables_started.json"  # committed: the run-once guard survives a moved run folder


def _people_registered() -> Optional[str]:
    from ..harness_ablation import HELD_OUT, SETS

    entry = SETS.get("plan_held_v3")
    if not entry or "plan_held_v3" not in HELD_OUT or entry[2] != "cases_plan_held_v3.sha256":
        return "the held-out people (plan_held_v3) are not registered in harness_ablation's SETS and HELD_OUT"
    return None


def preflight(cases: list[dict[str, Any]], seeds: list[int], tables: Any) -> Optional[str]:
    """What would waste the one run, by counts only: a case without its table function, a trap kind the reader does not
    know, or too few pairs with a planted trap for check 5."""
    from .run import TRAP_NAME

    missing = sum(not callable(getattr(tables, f"make_{c['id']}", None)) for c in cases)
    bad_group = 0  # the review: check 8 must name a real column, or it passes with nothing tested
    for c in cases:
        needless = c.get("no_group_needed")
        if needless is not None and not missing:
            columns = {str(col) for f in getattr(tables, f"make_{c['id']}")(seeds[0]).values() for col in f.columns}
            bad_group += int(not isinstance(needless, str) or needless not in columns)
    if bad_group or not any(c.get("no_group_needed") for c in cases):
        return f"{bad_group} cases whose no_group_needed is not a column; {sum(bool(c.get('no_group_needed')) for c in cases)} cases name one (at least one)"
    unknown = sum(t.get("kind") not in TRAP_NAME for c in cases for t in c.get("enforceable_traps") or [])
    pairs = sum(bool(c.get("enforceable_traps")) for c in cases) * len(seeds)
    if missing or unknown or pairs < gate.MIN_PROJECTS:
        return f"{missing} cases without a table function, {unknown} unknown trap kinds, {pairs} pairs with a planted trap (at least {gate.MIN_PROJECTS})"
    return None


def _cmd_tables(model: str, cap: float, yes: bool) -> int:
    """Every held-out case on both of its seeds, both arms, once. Refused on a second run (a run folder or a result), an
    unpinned gate, a Claude model, a preflight problem, or without --yes. A model request that fails stops the run."""
    if "claude" in model.lower() or "anthropic" in model.lower():
        print("refused: a Claude model is never an arm here (the owner, 2026-11-15)", file=sys.stderr)
        return 2
    why = gate.pinned()
    if why:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    from dclab_rnd.settings import Settings as _Settings

    if _runs() or list(RESULTS.glob("HPG3-*.json")) or MARKER.exists() or (Path(_Settings().agent_home) / "exam_vault" / "HPG3-001_tables").exists():
        print("refused: the held-out tables were run already; they are read once", file=sys.stderr)
        return 2
    why = _people_registered()
    if why:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    from .run import build, load_held

    cases, seeds, tables = load_held()
    problem = preflight(cases, seeds, tables)
    if problem:
        print(f"refused: {problem}", file=sys.stderr)
        return 2
    if not yes:
        print(json.dumps({"would_run": "every held-out case, both seeds, the rules arm and the model arm", "projects": 2 * len(cases) * len(seeds),
                          "model": model, "cap_eur": cap}, indent=1))
        return 0
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    for name in ("DCLAB_DATABASE_URL", "DCLAB_STUDIO_HOME", "DCLAB_DRAFT_HOME", "DCLAB_INTERN_HOME"):
        os.environ.pop(name, None)  # the held-out projects stay in the run's folder, never in a shared database or workspace
    from dclab_rnd import experience
    from dclab_rnd.settings import Settings

    # held-out: its calls keep their shape in the log and are kept whole, with their thinking, in the exam vault, outside
    # every folder the training store reads (as a sealed conversation exam is)
    experience.sealed_run("sealed_planning_gate_v3_held", vault=Path(Settings().agent_home) / "exam_vault" / "HPG3-001_tables")
    from ..conversation_exam import Budget
    from ..harness_ablation import _ModelView, _stopped, owner_allows, price_snapshot, run_gateway

    refused = owner_allows(cap)
    if refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 2
    stamp = datetime.now(timezone.utc)
    run_dir = RUN_ROOT / f"{stamp:%Y-%m-%d}-planning-gate-v3-tables"
    run_dir.mkdir(parents=True)
    from ..harness_ablation import code_state

    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(json.dumps({"started": stamp.isoformat(timespec="seconds"), "run": run_dir.name, "code": code_state()}, indent=1) + "\n",
                      encoding="utf-8")
    price_snapshot(run_dir, model)
    gateway = run_gateway(run_dir)
    budget = Budget(gateway, cap)
    rows: list[dict[str, Any]] = []
    stopped: Optional[str] = "interrupted"  # cleared only when every case has run: a run cut short is never read as whole
    try:
        for seed in seeds:
            for case in cases:
                frames = getattr(tables, f"make_{case['id']}")(seed)
                for arm in ("rules", "model"):
                    view = _ModelView(budget, f"planning-gate-v3:{case['id']}:{seed}", model) if arm == "model" else None
                    row = build(case, frames, arm, run_dir / "projects" / f"{case['id']}-{seed}-{arm}", gateway=view, spent=budget.spent_eur)
                    rows.append({**row, "seed": seed})
                    print(f"  {len(rows)} of {2 * len(cases) * len(seeds)} built", flush=True)  # no case content on screen
                    if row.get("model_errors") or budget.spent_eur() >= cap:
                        stopped = "the cap was reached" if budget.spent_eur() >= cap else f"a model request failed ({row['model_errors'][0]})"
                        break
                if stopped and stopped != "interrupted":
                    break
            if stopped and stopped != "interrupted":
                break
        else:
            stopped = None
    except Exception as error:  # noqa: BLE001
        stopped = _stopped(error) or type(error).__name__
    finally:
        (run_dir / "rows.json").write_text(json.dumps({"rows": rows, "stopped": stopped, "spent_eur": round(budget.spent_eur(), 4), "model": model,
                                                        "seeds": seeds, "of": 2 * len(cases) * len(seeds), "evaluation_only": True,
                                                        "code": json.loads(MARKER.read_text(encoding="utf-8")).get("code")}, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"built": len(rows), "of": 2 * len(cases) * len(seeds), "stopped": stopped, "spent_eur": round(budget.spent_eur(), 4), "run": run_dir.name}, indent=1))
    return 0 if stopped is None else 1


PEOPLE = ROOT / "evidence" / "campaigns" / "harness_ablation_v1" / "results"


def _people_rows(result_id: str) -> tuple[str, Optional[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """The people run, refused unless it is the only run ever made on plan_held_v3 (no picking the better of two), on the
    pinned people, after the gate's pin (the review)."""
    found = [p for p in PEOPLE.glob(f"{result_id}_*.json") if not p.name.endswith("_unjudged.json")]
    if len(found) != 1:
        raise ValueError(f"{len(found)} judged results named {result_id}")
    doc = json.loads(found[0].read_text(encoding="utf-8"))
    if doc.get("set") != "plan_held_v3":
        raise ValueError(f"{result_id} is not a run on plan_held_v3")
    if doc.get("incomplete"):
        raise ValueError(f"{result_id} stopped before every person was played")
    runs = set()
    for other in PEOPLE.glob("HAB-*.json"):
        try:
            if json.loads(other.read_text(encoding="utf-8")).get("set") == "plan_held_v3":
                runs.add(other.name.split("_", 1)[0])
        except (OSError, ValueError):
            continue
    if runs != {result_id}:
        raise ValueError(f"{len(runs)} runs on plan_held_v3: the held-out people are played once")
    pinned_people = (ROOT / "dclab_rnd" / "agent_eval" / "harness_ablation" / "cases_plan_held_v3.sha256").read_text(encoding="utf-8").split()[0]
    if str((doc.get("manifest") or {}).get("cases_sha256") or "").split()[:1] != [pinned_people]:
        raise ValueError(f"{result_id} did not run on the pinned people")
    if datetime.fromisoformat(str(doc.get("at"))).timestamp() < gate.PIN.stat().st_mtime:
        raise ValueError(f"{result_id} ran before the gate was pinned")
    return str(doc["models"]["arms"]), doc["models"].get("judge"), list(doc["scores"].get("case") or []), list(doc["scores"].get("model_only") or [])


def _cmd_judge(people_id: Optional[str]) -> int:
    """The verdict, once. The whole verdict stays in the run's folder; the committed result holds counts and means."""
    runs = _runs()
    if len(runs) != 1:
        print(f"refused: {len(runs)} held-out table runs", file=sys.stderr)
        return 2
    if list(RESULTS.glob("HPG3-*.json")):
        print("refused: the gate was judged already; a second reading could pick its people run", file=sys.stderr)
        return 2
    kept = json.loads((runs[0] / "rows.json").read_text(encoding="utf-8"))
    if not people_id:
        print("refused: name the judged people run (--people HAB-0NN): the gate is read once, with all its checks", file=sys.stderr)
        return 2
    people = None
    if people_id:
        model, judge_model, case_rows, alone_rows = _people_rows(people_id)
        if model != kept.get("model"):
            print(f"refused: the people run used {model}, the tables {kept.get('model')}", file=sys.stderr)
            return 2
        people = gate.people(model, judge_model, case_rows, alone_rows)
    verdict = gate.judge(kept["rows"], people)
    if kept.get("stopped") or len(kept["rows"]) != kept.get("of"):
        verdict["passed"] = False
        verdict["stopped"] = True
    (runs[0] / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str) + "\n", encoding="utf-8")
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_id = "HPG3-001"
    doc = {"id": run_id, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tables_run": runs[0].name, "people_result": people_id,
           "model": kept.get("model"), "spent_eur_tables": kept.get("spent_eur"), "evaluation_only": True, **gate.public(verdict)}
    path = RESULTS / f"{run_id}_gate.json"
    path.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"passed": doc["passed"], "not_judged": doc["not_judged"], "result": path.name}, indent=1))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.planning_gate_v3", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("pin")
    tables = sub.add_parser("tables")
    tables.add_argument("--model", default=MODEL)
    tables.add_argument("--cap-eur", type=float, default=0.5)
    tables.add_argument("--yes", action="store_true")
    judged = sub.add_parser("judge")
    judged.add_argument("--people", default=None, help="the judged harness_ablation result on plan_held_v3 (HAB-0NN)")
    args = parser.parse_args(argv)
    if args.command == "plan":
        return _cmd_plan()
    if args.command == "pin":
        return _cmd_pin()
    if args.command == "tables":
        return _cmd_tables(args.model, args.cap_eur, args.yes)
    return _cmd_judge(args.people)
