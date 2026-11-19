"""The planning practice set: why the planner misses planted traps (2026-11-19).

hp_gate_v2 showed both plans enforce about a third of the traps the case authors planted, and its rows may not be read.
This practice set is the open twin: cases written by an independent session that never read the planner's code, with
every answer readable. Each case is built on the product's own unattended path (``run.build``, the exam's builder,
unchanged) and every planted trap is read into one line: was it caught before the plan, did the plan catch it, was it
enforced in the end, and what the built project did instead.

    python -m dclab_rnd.agent_eval.planning_gate_v2.practice             # the rules arm on every case and seed (free)
    python -m dclab_rnd.agent_eval.planning_gate_v2.practice --case pr03 # one case

The rules arm calls no model. Rows and a report go to ``agent_runs/practice/<date>-planning-practice/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
FOLDER = HERE / "practice"
ROOT = HERE.parents[2]
RUN_ROOT = ROOT / "agent_runs" / "practice"
KIND = {"late_column": "forbid", "group_split": "group", "time_split": "time_split"}  # a planted trap's kind, as a catch reads it


def _plan_caught(plan: Mapping[str, Any], kind: str, column: str, columns: Sequence[str]) -> bool:
    """The plan named this trap: a catch of the same kind on the same column (any time catch, for a time split)."""
    from dclab_rnd.tasks import enforced

    for trap in plan.get("caught_traps") or []:
        if enforced.kind_of(str(trap)) == KIND[kind] and (kind == "time_split" or enforced.column_of(str(trap), columns) == column):
            return True
    return False


def explain(trap: Mapping[str, Any], row: Mapping[str, Any], solution: Mapping[str, Any], plan: Mapping[str, Any],
            stages: Mapping[str, Any], columns: Sequence[str]) -> dict[str, Any]:
    """One planted trap, read whole: where it was caught (or not) and what the project did instead. Pure."""
    from dclab_rnd.tasks import enforced

    kind, column = str(trap["kind"]), str(trap["column"])
    done = bool(trap.get("enforced"))
    before = column in (row.get("forbidden_before") or [])
    caught = _plan_caught(plan, kind, column, columns)
    asked = [a for a in (plan.get("enforcement") or {}).get("asked") or [] if a.get("kind") == KIND[kind]
             and (kind == "time_split" or str(a.get("value")) == column)]
    split = enforced.split_of(stages)
    if kind == "late_column":
        used = column in enforced.final_features(stages)
        did = f"forbidden: {column in enforced.forbidden_columns(solution)}; a final feature: {used}"
    elif kind == "time_split":
        did = f"split: {split or 'not run'}; time column: {solution.get('time_column')}"
    else:
        did = f"split: {split or 'not run'}; group column: {solution.get('group_column')}"
    where = ("enforced before the plan" if done and before else "enforced by the plan" if done and caught
             else "enforced without a plan catch" if done else "asked, not applied" if asked else "caught, not enforced" if caught
             else "missed")
    return {"case": row.get("case"), "seed": row.get("seed"), "kind": kind, "column": column, "enforced": done,
            "before_plan": before, "plan_caught": caught, "asked": bool(asked), "where": where, "did": did}


def _one(job: tuple[dict[str, Any], int, str]) -> dict[str, Any]:
    """Build one case on one seed with the rules arm and read every planted trap (a worker: imports inside)."""
    from dclab_rnd.storage import open_stores
    from dclab_rnd.studio import engine
    from dclab_rnd.tasks import path as task_path

    from .run import build, load_held

    case, seed, run_dir = job
    _, _, tables = load_held(FOLDER)
    frames = getattr(tables, f"make_{case['id']}")(seed)
    folder = Path(run_dir) / "projects" / f"{case['id']}-{seed}-rules"
    try:
        row = {**build(case, frames, "rules", folder), "seed": seed}
    except Exception as error:  # noqa: BLE001 - a practice run reports a crash and goes on
        return {"case": case["id"], "seed": seed, "error": f"{type(error).__name__}: {error}"[:300], "traps": []}
    projects = open_stores(folder)[0]
    project = projects.get(row["project_id"])
    stages = {s: projects.read_stage(row["project_id"], s) for s in engine.STAGE_KEYS}
    stages = {k: v for k, v in stages.items() if v is not None}
    solution, plan = project.get("solution") or {}, task_path.stored(project) or {}
    columns = [c for f in frames.values() for c in f.columns]
    row["traps"] = [explain(t, row, solution, plan, stages, columns) for t in row.get("planted_traps") or []]
    row["solution"] = {k: solution.get(k) for k in ("target", "task", "time_column", "group_column")}
    row["plan_caught_traps"] = list(plan.get("caught_traps") or [])
    return row


def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per trap kind: planted, enforced, and where each was caught or lost."""
    by: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for t in row.get("traps") or []:
            by[t["kind"]]["planted"] += 1
            by[t["kind"]]["enforced"] += int(t["enforced"])
            by[t["kind"]][t["where"]] += 1
    traps = [t for r in rows for t in r.get("traps") or []]
    return {"projects": len(rows), "crashed": sum(bool(r.get("error")) for r in rows), "planted": len(traps),
            "enforced": sum(t["enforced"] for t in traps), "by_kind": {k: dict(v) for k, v in sorted(by.items())}}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.planning_gate_v2.practice", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", default=None, help="only these case ids")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args(argv)
    os.environ["DCLAB_NO_LIVE_MODELS"] = "1"  # the rules arm: no model, ever
    from .run import load_held

    cases, seeds, _ = load_held(FOLDER)
    cases = [c for c in cases if not args.case or c["id"] in args.case]
    run_dir = RUN_ROOT / f"{datetime.now(timezone.utc):%Y-%m-%dT%H%M%S}-planning-practice"
    run_dir.mkdir(parents=True)
    jobs = [(c, s, str(run_dir)) for s in seeds for c in cases]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for row in pool.map(_one, jobs):
            rows.append(row)
            print(f"  {len(rows)} of {len(jobs)}: {row['case']} seed {row['seed']}" + (f" crashed: {row['error']}" if row.get("error") else
                  "  " + ", ".join(f"{t['kind']}:{t['where']}" for t in row["traps"])), flush=True)
    report = {"set": "planning_practice_v1", "arm": "rules", "summary": summary(rows), "rows": rows}
    (run_dir / "rows.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    print(f"kept {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
