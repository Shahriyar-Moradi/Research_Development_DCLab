"""Does the Leakage Advisor help? A live check on the planning practice set (2026-11-19).

Each practice case (written by an independent session, every answer readable) is built as a project with its outcome,
task, time column and the brief as the prediction moment, and nothing kept out. The leakage stage then runs twice: the
advisor's rules alone (no gateway installed) and the advisor with the product's model (the run's own gateway, capped,
every exchange kept whole with its thinking under the run's folder). Each column the case author marked is scored:
a planted late column should be blocked or asked about; an honest column that only looks suspicious should be kept.

    python -m dclab_rnd.agent_eval.leakage_advisor_live --yes --cap-eur 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
MODEL = "qwen/qwen3.8-flash"


def score(advice: Mapping[str, Any], leaks: list[str], honest: list[str]) -> dict[str, Any]:
    """Pure: each marked column's action (block, ask, keep, or absent)."""
    acts = {str(r.get("column")): str(r.get("action")) for r in advice.get("recommendations") or []}
    return {"leaks": {c: acts.get(c, "absent") for c in leaks}, "honest": {c: acts.get(c, "absent") for c in honest}}


def summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    leak, honest = Counter(), Counter()
    for r in rows:
        for a in r[arm]["leaks"].values():
            leak[a] += 1
        for a in r[arm]["honest"].values():
            honest[a] += 1
    n_leak, n_honest = sum(leak.values()), sum(honest.values())
    return {"leaks": dict(leak), "leaks_flagged": round((leak["block"] + leak["ask"]) / n_leak, 3) if n_leak else None,
            "leaks_blocked": round(leak["block"] / n_leak, 3) if n_leak else None,
            "honest": dict(honest), "honest_blocked": round(honest["block"] / n_honest, 3) if n_honest else None,
            "honest_flagged": round((honest["block"] + honest["ask"]) / n_honest, 3) if n_honest else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.leakage_advisor_live")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--cap-eur", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    from .planning_gate_v2.practice import FOLDER
    from .planning_gate_v2.run import load_held

    cases, _, tables = load_held(FOLDER)
    if not args.yes:
        print(json.dumps({"would_run": len(cases), "stages": "data, leakage x2", "model": MODEL, "cap_eur": args.cap_eur}))
        return 0
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    for name in ("DCLAB_DATABASE_URL", "DCLAB_STUDIO_HOME", "DCLAB_DRAFT_HOME", "DCLAB_INTERN_HOME"):
        os.environ.pop(name, None)
    from dclab_rnd.models import gateway as gw_mod
    from dclab_rnd.storage import open_stores
    from dclab_rnd.studio import data as sd
    from dclab_rnd.studio import engine

    from .harness_ablation import price_snapshot, run_gateway

    run_dir = ROOT / "agent_runs" / "live_tests" / f"{datetime.now(timezone.utc):%Y-%m-%dT%H%M%S}-leakage-advisor-live"
    run_dir.mkdir(parents=True)
    price_snapshot(run_dir, MODEL)
    gateway = run_gateway(run_dir)

    def spent() -> float:
        path = run_dir / "usage.jsonl"
        return sum(float(json.loads(l).get("cost_eur") or json.loads(l).get("eur") or 0) for l in path.read_text().splitlines() if l.strip()) if path.exists() else 0.0

    rows: list[dict[str, Any]] = []
    for case in cases:
        if spent() >= args.cap_eur:
            print("stopped at the cap", file=sys.stderr)
            break
        frames = getattr(tables, f"make_{case['id']}")(args.seed)
        leaks = [t["column"] for t in case.get("enforceable_traps") or [] if t["kind"] == "late_column"]
        honest = [c for c in case.get("honest_columns_that_look_suspicious") or [] if isinstance(c, str)]
        if not leaks and not honest:
            continue
        projects = open_stores(run_dir / "projects" / case["id"])[0]
        pid = projects.create(case["id"], "general", case["brief"][:300])["id"]
        for i, (name, frame) in enumerate(frames.items()):
            frame.to_csv(projects.data_dir(pid) / name, index=False)
            if i == 0:
                sd.attach_data(projects, pid, name, actor="human")
        time = next((t["column"] for t in case.get("enforceable_traps") or [] if t["kind"] == "time_split"), None)
        project = projects.get(pid)
        project["solution"] = {"target": case["known_facts"]["target"], "task": "regression" if case.get("task") == "regression" else "binary",
                               "prediction_moment": case["brief"], "forbidden": [], "identifiers": [], "time_column": time, "labels": "complete"}
        project["settings"].update(quick=True)
        projects.save(project)
        row: dict[str, Any] = {"case": case["id"]}
        try:
            engine.execute(projects, pid, "data")
            with __import__("unittest.mock").mock.patch.object(gw_mod, "_INSTALLED", None):
                rules = engine.execute(projects, pid, "leakage")["evidence"]["leakage_advisor"]
            with __import__("unittest.mock").mock.patch.object(gw_mod, "_INSTALLED", gateway):
                model = engine.execute(projects, pid, "leakage")["evidence"]["leakage_advisor"]
            row.update(rules=score(rules, leaks, honest), model=score(model, leaks, honest), model_used=bool(model.get("model_used", model.get("source"))))
        except Exception as error:  # noqa: BLE001 - a case that fails is reported, the run goes on
            row["error"] = f"{type(error).__name__}: {error}"[:300]
        rows.append(row)
        print(f"  {len(rows)} of {len(cases)}: {case['id']}" + (f" failed: {row['error']}" if "error" in row else ""), flush=True)
    ok = [r for r in rows if "error" not in r]
    exchanges = [json.loads(l) for f in run_dir.glob("**/model_exchanges/**/*.jsonl") for l in f.read_text().splitlines() if l.strip()]
    report = {"cases": len(rows), "failed": len(rows) - len(ok), "rules": summary(ok, "rules"), "model": summary(ok, "model"),
              "spent_eur": round(spent(), 4), "exchanges": len(exchanges),
              "with_thinking": sum(bool(e.get("thinking") or e.get("reasoning")) for e in exchanges if e.get("outcome") == "ok"), "rows": rows}
    (run_dir / "result.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    print(f"kept {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
