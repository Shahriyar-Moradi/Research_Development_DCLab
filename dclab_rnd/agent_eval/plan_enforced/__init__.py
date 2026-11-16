"""Did the engine obey the plan? The measurement of Hierarchical Planning's stage H0 (campaign ``plan_enforced_v1``).

It reads built projects that a run kept (their solution, their stage records, and the traps their plan caught) and counts,
by code, how many of the traps code can enforce were enforced (``dclab_rnd.tasks.enforced``). Nothing is run and no
model is asked: the first result, PE-001, is the baseline from a live run kept before the repair.

    python -m dclab_rnd.agent_eval.plan_enforced --kept agent_runs/live_tests/<run> [--store PE-00N --note "..."]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dclab_rnd.tasks import enforced

from . import gate

CAMPAIGN = "plan_enforced_v1"
ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "evidence" / "campaigns" / CAMPAIGN / "results"


def _traps(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        return [str(v) for v in ast.literal_eval(str(value))]
    except (ValueError, SyntaxError):
        return []


def read_kept(run_dir: Path) -> list[dict[str, Any]]:
    """Each scenario of a kept studio run: what its plan caught, and what the built project did about it."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    out = []
    for scenario in summary.get("scenarios") or []:
        folder = run_dir / "projects" / str(scenario.get("project_id"))
        if not (folder / "project.json").exists():
            continue
        solution = json.loads((folder / "project.json").read_text(encoding="utf-8")).get("solution") or {}
        stages = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((folder / "stages").glob("*.json"))}
        profiles = ((stages.get("data") or {}).get("evidence") or {}).get("feature_profiles") or []
        columns = sorted({str(f.get("feature")) for f in profiles} | {str(c) for c in solution.get("all_columns") or []} | set(enforced.final_features(stages)))
        records = enforced.obeyed(_traps(scenario.get("caught_traps")), solution, stages, columns)
        out.append({"scenario": scenario.get("scenario_id"), "passed_as_reported": bool(scenario.get("passed")), "routed_to": scenario.get("routed_to"),
                    "holdout_score": scenario.get("holdout_score"), "holdout_split": enforced.split_of(stages), "forbidden": list(solution.get("forbidden") or []),
                    "final_features": enforced.final_features(stages), "summary": enforced.summary(records),
                    "traps": [r for r in records if r["enforceable"]],
                    # a build that stopped with a question obeyed its plan; one that paused, went on and built did not stop
                    "paused_with_question": bool(scenario.get("saw_clarify_pause")),
                    "stopped_with_question": bool(scenario.get("saw_clarify_pause")) and "final" not in stages})
    return out


def result(run_dir: Path, projects: list[dict[str, Any]], result_id: str, note: str = "") -> dict[str, Any]:
    check = gate.obeyed_check(projects)
    return {"schema": "dclab_rnd.plan_enforced.v1", "campaign_id": CAMPAIGN, "id": result_id, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gate": gate.GATE_ID, "gate_pinned": gate.pinned() is None, "source": {"kind": "kept live run", "run": run_dir.name, "live_model_calls": 0},
            "note": note, "totals": {"projects": len(projects), "reported_passed": sum(p["passed_as_reported"] for p in projects),
                                     "traps_caught": sum(p["summary"]["caught"] for p in projects),
                                     "enforceable": sum(p["summary"]["enforceable"] for p in projects),
                                     "enforced": sum(p["summary"]["enforced"] for p in projects)},
            "check_obeyed": check, "projects": projects,
            "limits": ["five invented tables of 180 rows from one run and one model", "a trap is read from its name and the column it ends with",
                       "the other checks of the gate are not read here"]}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.plan_enforced", description=__doc__)
    parser.add_argument("--kept", required=True, help="a kept studio run folder (summary.json and projects/)")
    parser.add_argument("--store", default=None, help="a result id (PE-00N): write the result into the campaign")
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)
    run_dir = Path(args.kept)
    if not (run_dir / "summary.json").exists():
        print(f"refused: {run_dir} holds no summary.json", file=sys.stderr)
        return 2
    doc = result(run_dir, read_kept(run_dir), args.store or "unstored", args.note)
    print(json.dumps({k: doc[k] for k in ("totals", "check_obeyed")}, indent=1))
    for p in doc["projects"]:
        print(f"  {p['scenario']}: reported passed {p['passed_as_reported']}; enforceable {p['summary']['enforceable']}, enforced {p['summary']['enforced']}; broken {p['summary']['broken']}")
    if args.store:
        path = RESULTS / f"{args.store}.json"
        if path.exists():
            print(f"refused: {path.name} exists (results are never rewritten)", file=sys.stderr)
            return 2
        RESULTS.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"stored {path.relative_to(ROOT)}")
    return 0
