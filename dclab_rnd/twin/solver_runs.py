"""The rules' standard plan on many studied problems, for the M5 solver set (owner, 2026-10-21: reach 10, then 20 clean runs).

Each run is a twin run with the workflow lane only: the sample seeded **blind** (its known solution hidden, so the rules
must find the leaks themselves), the outcome **named** in the task as a person names it (DCLab never picks the outcome:
the owner's rule of 2026-10-13, and why a blind task stopped the standard plan after describing the data), judged against
the planted truth (``compare.planted``: the target and the R&D's leak columns) into ``steps.jsonl``. The layout is a twin
run's, so the store keeps it and the M5 builder reads its score and verdict like any other; no ``compare.json`` is
written, so the twin evaluation (25.8: Claude against the rules) never reads a run Claude was not in. No model is called.

    python -m dclab_rnd.twin.solver_runs                 # every studied sample, once
    python -m dclab_rnd.twin.solver_runs --samples telco_churn,hyperack --full
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

OUTCOME_NAMED = "The outcome to predict is the column"  # the M5 builder reads this phrase: the outcome was given, not chosen
TASK = ("Build a leakage-safe model for the project's table: understand the problem, describe the data, propose and set the solution, "
        "run the stages and report the honest holdout score. " + OUTCOME_NAMED + " `{target}`.")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_one(key: str, home: Optional[Path] = None, quick: bool = True, max_steps: int = 30, max_minutes: int = 40,
            run_id: Optional[str] = None) -> dict[str, Any]:
    """One blind standard-plan run on a studied sample, as ``<home>/twin/runs/<run_id>/``; its manifest."""
    from .. import context
    from ..expansion.runner import fast_profile
    from ..studio import data as sd
    from . import _workflow_gateway, compare as comparing, runs_home
    from .lane import Lane

    _frame, known = sd.load_sample(key)
    target = known.get("target")
    if not target:
        raise ValueError(f"{key}: the sample names no outcome, so a person's request cannot name it")
    task = TASK.format(target=target)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    root = runs_home(home) / run_id
    root.mkdir(parents=True, exist_ok=False)
    lane = Lane("workflow", root / "workflow", _workflow_gateway("off", root / "workflow"),
                label="the current system: rules, validator, standard plan, no model")
    manifest: dict[str, Any] = {"run_id": run_id, "seed": f"sample:{key}", "task": task, "model": None, "started": now(), "finished": None,
                                "blind": True, "system_only": True, "outcome_named": True, "quick": quick, "workflow_models": "off",
                                "caps": {"max_steps": max_steps, "max_minutes": max_minutes},
                                "lanes": {"workflow": {"label": lane.label}},
                                "notes": ["the workflow lane only (no Claude): an M5 solver run; the outcome named in the task, the leaks hidden"]}
    (root / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with context.using(lane.services):
        lane.seed_sample(key, task[:300], blind=True)
    manifest["lanes"]["workflow"].update(project_id=lane.project_id)
    if quick:
        with fast_profile():
            summary = lane.run_project(task, None, max_steps, max_minutes)
    else:
        summary = lane.run_project(task, None, max_steps, max_minutes)
    (root / "workflow" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    manifest["lanes"]["workflow"].update(status=lane.status, error=lane.error, started=lane.started, finished=lane.finished, session_id=lane.session_id)
    truth = comparing.planted("sample", key, lane)
    manifest["truth"] = truth
    absent = {"data_sha256": summary.get("data_sha256"), "given_solution": summary.get("given_solution"), "status": "not_run",
              "session": {"model": "not run (a system-only solver run)"}}  # the same input, abstaining: never read as Claude's answer
    steps = comparing.write_steps(root / "steps.jsonl", run_id, summary, absent, truth)
    judged = [s for s in steps if s.step == "solution.columns" and s.verdict is not None]
    manifest["verdict"] = judged[0].verdict.system if judged else None
    manifest["finished"] = now()
    (root / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    from ..studio import data as sd

    p = argparse.ArgumentParser(prog="python -m dclab_rnd.twin.solver_runs", description=__doc__.split("\n\n")[0])
    p.add_argument("--samples", default="all", help="comma-separated sample keys, or all (default)")
    p.add_argument("--home", type=Path, default=None, help="the workspace folder (default DCLAB_AGENT_HOME or agent_runs/)")
    p.add_argument("--full", action="store_true", help="the engine's full profile instead of the quick one")
    args = p.parse_args(argv)
    keys = [s["key"] for s in sd.sample_catalog()] if args.samples == "all" else [k.strip() for k in args.samples.split(",") if k.strip()]
    passed = 0
    for key in keys:
        try:
            m = run_one(key, args.home, quick=not args.full)
        except Exception as exc:  # noqa: BLE001 — one sample that cannot run never stops the others
            print(f"{key:28s} error: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            continue
        verdict = m.get("verdict") or {}
        ok = bool(verdict) and all(v is True for v in verdict.values())
        passed += ok
        print(f"{key:28s} {m['lanes']['workflow'].get('status')!s:10s} truth {'passed' if ok else 'failed' if verdict else 'not judged'} "
              f"{json.dumps(verdict)}  {m['run_id']}", flush=True)
    print(f"{passed} of {len(keys)} passed their planted truth", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
