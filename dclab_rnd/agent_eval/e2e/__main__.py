"""python -m dclab_rnd.agent_eval.e2e plan|run|rules.

plan   prints what a run would do and costs nothing.
rules  runs only DCLab's rules arm on the dev cases (no model, free) and scores it.
run    the three arms on the named dev cases, live (--yes and --cap-eur needed; the product's model only, never Claude).
The held-out seeds are never read here: the gate run is a separate, later step the owner approves."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import CAMPAIGN, MODEL

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "evidence" / "campaigns" / CAMPAIGN / "results"


def next_id() -> str:
    RESULTS.mkdir(parents=True, exist_ok=True)
    taken = [int(p.name[4:7]) for p in RESULTS.glob("E2E-[0-9][0-9][0-9]*.json")]
    return f"E2E-{max(taken, default=0) + 1:03d}"


def refuse_model(model: str) -> str | None:
    low = model.lower()
    if "claude" in low or "anthropic" in low:
        return "a Claude model is never an arm here (the owner, 2026-11-15)"
    return None


def rules_arm(case: Any, folder: Path) -> dict[str, Any]:
    """DCLab with no model: the twin's workflow lane alone, its standard plan."""
    from dclab_rnd import context
    from dclab_rnd.storage import open_stores
    from dclab_rnd.studio import engine
    from dclab_rnd.twin.claude import NoModels  # NoModels is defined alongside TwinGateway in twin.claude and re-exported in twin
    from dclab_rnd.twin.lane import Lane

    from . import dclab

    pid = dclab.seed_source(case, folder / "source")
    lane = Lane("workflow", folder / "rules", NoModels(), label="the rules")
    with context.using(lane.services):
        lane.seed_project(open_stores(folder / "source")[0], pid, blind=True)
        lane.run_project(case.request(), None, max_steps=40, max_minutes=20)
    project = lane.projects.get(lane.project_id) if lane.project_id else None
    records = {s: lane.projects.read_stage(lane.project_id, s) for s in engine.STAGE_KEYS} if lane.project_id else {}
    records = {k: v for k, v in records.items() if v is not None}
    return dclab.outcome("rules", project, records)


def _select_cases(spec: str | None) -> list[Any]:
    from . import cases

    available = cases.dev_cases()
    if not spec:
        return available
    by_id = {c.id: c for c in available}
    wanted = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [w for w in wanted if w not in by_id]
    if unknown:
        raise ValueError(f"unknown or non-dev case ids {unknown}; available dev cases: {', '.join(by_id)}")
    return [by_id[w] for w in wanted]


def _run_folder(run_id: str) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder = ROOT / "agent_runs" / "live_tests" / f"{day}-e2e-{run_id}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _elapsed_seconds(started: str | None, finished: str | None, fallback: float | None = None) -> float | None:
    if started and finished:
        try:
            delta = datetime.fromisoformat(finished) - datetime.fromisoformat(started)
            return round(delta.total_seconds(), 1)
        except ValueError:
            pass
    return round(fallback, 1) if fallback is not None else None


def _lane_outcome(arm: str, twin_root: Path, lane_name: str, manifest: dict[str, Any], wall: float) -> dict[str, Any]:
    from dclab_rnd.studio import engine
    from dclab_rnd.studio.store import ProjectStore

    from . import dclab

    store = ProjectStore(twin_root / lane_name / "projects")
    lane_info = (manifest.get("lanes") or {}).get(lane_name) or {}
    pid = lane_info.get("project_id")
    if not pid:
        listed = store.list()
        pid = listed[0]["id"] if listed else None
    project = store.get(pid) if pid else None
    records = {s: store.read_stage(pid, s) for s in engine.STAGE_KEYS} if pid else {}
    records = {k: v for k, v in records.items() if v is not None}
    out = dclab.outcome(arm, project, records)
    out["seconds"] = _elapsed_seconds(lane_info.get("started"), lane_info.get("finished"), wall)
    out["euros"] = 0.0 if arm == "rules" else round(float((manifest.get("model_usage") or {}).get("eur") or 0.0), 6)
    if lane_info.get("error"):
        out["error"] = lane_info["error"]
    return out


def _scored_row(case: Any, out: dict[str, Any]) -> dict[str, Any]:
    from . import score

    row = score.score(case, out)
    if "euros" in out:
        row["euros"] = out["euros"]
    if "seconds" in out:
        row["seconds"] = out["seconds"]
    if "error" in out:
        row["error"] = out["error"]
    return row


def _plan_doc(selected: list[Any], model: str, cap: float | None) -> dict[str, Any]:
    return {"campaign": CAMPAIGN, "cases": [c.id for c in selected], "arms": ["rules", "dclab_agent", "plain"],
            "model": model, "cap_eur": cap}


def _cmd_rules(selected: list[Any]) -> int:
    run_id = next_id()
    run_dir = _run_folder(run_id)
    scores: list[dict[str, Any]] = []
    for case in selected:
        t0 = time.monotonic()
        out = rules_arm(case, run_dir / case.id)
        out["seconds"] = round(time.monotonic() - t0, 1)
        out["euros"] = 0.0
        row = _scored_row(case, out)
        scores.append(row)
        print(f"{case.id:20s} finished={row['finished']} leaks_kept_out={row.get('leaks_kept_out')} "
              f"claimed={row.get('claimed')} real={row.get('real')} inside={row.get('real_inside_interval')} "
              f"holdout_uses={row.get('holdout_uses')} seconds={row.get('seconds')}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{run_id}_rules_dev.json"
    doc = {"id": run_id, "arms": ["rules"], "model": None, "cases": [c.id for c in selected], "scores": scores,
           "run_dir": str(run_dir.relative_to(ROOT))}
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(path))
    return 0


def _cmd_run(selected: list[Any], model: str, cap: float | None, yes: bool) -> int:
    reason = refuse_model(model)
    if reason:
        print(f"refused: {reason}", file=sys.stderr)
        return 2
    if not yes:
        print(json.dumps(_plan_doc(selected, model, cap), indent=1))
        print("refused: nothing was sent; pass --yes and --cap-eur above zero to run live", file=sys.stderr)
        return 2
    if cap is None or cap <= 0:
        print("refused: --cap-eur above zero is required", file=sys.stderr)
        return 2
    if os.environ.get("DCLAB_NO_LIVE_MODELS") == "1":
        print("refused: DCLAB_NO_LIVE_MODELS=1 is set", file=sys.stderr)
        return 2
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    from dclab_rnd import twin
    from dclab_rnd.agent_eval import harness_ablation
    from dclab_rnd.agent_eval.conversation_exam import Budget  # Budget lives in conversation_exam and is imported there by harness_ablation
    from dclab_rnd.models.gateway import mark_invented_tables

    from . import dclab, plain

    why = harness_ablation.owner_allows(cap)
    if why:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    run_id = next_id()
    run_dir = _run_folder(run_id)
    harness_ablation.price_snapshot(run_dir, model)
    spent = 0.0
    scores: list[dict[str, Any]] = []
    plain_calls: dict[str, list[dict[str, Any]]] = {}
    ran_cases: list[str] = []
    not_run: list[str] = []
    for index, case in enumerate(selected):
        if spent >= cap:
            not_run = [c.id for c in selected[index:]]
            print(f"stopped at cap EUR {cap:.4f} (spent EUR {spent:.4f}); did not run: {', '.join(not_run)}")
            break
        ran_cases.append(case.id)
        case_dir = run_dir / case.id
        pid = dclab.seed_source(case, case_dir / "source")
        t_twin = time.monotonic()
        manifest = twin.run(f"project:{pid}", home=case_dir, source_home=case_dir / "source", task=case.request(), model=model,
                            cap_eur=max(0.0, cap - spent), blind=True, design=False, roles="judged", quick=True,
                            invented_tables="e2e comparison: replicas made by code")
        twin_wall = time.monotonic() - t_twin
        twin_root = case_dir / "twin" / "runs" / manifest["run_id"]
        spent += float((manifest.get("model_usage") or {}).get("eur") or 0.0)
        for arm_name, lane_name in (("rules", "workflow"), ("dclab_agent", "ai_first")):
            scores.append(_scored_row(case, _lane_outcome(arm_name, twin_root, lane_name, manifest, twin_wall)))
        if spent >= cap:
            plain_out: dict[str, Any] = {"arm": "plain", "finished": None, "seen": [], "calls": [],
                                          "error": f"cap EUR {cap:.4f} reached before plain arm", "euros": 0.0, "seconds": 0.0}
        else:
            gateway = harness_ablation.run_gateway(case_dir / "plain")
            mark_invented_tables(gateway, "e2e comparison: replicas made by code")
            budget = Budget(gateway, max(0.0, cap - spent))
            client = budget.client(f"e2e:{case.id}", plain.PURPOSE, model)
            t_plain = time.monotonic()
            if client is None:
                plain_out = {"arm": "plain", "finished": None, "seen": [], "calls": [], "error": "model client unavailable"}
            else:
                try:
                    plain_out = plain.run(case, client, case_dir / "plain" / "box")
                except Exception as exc:  # noqa: BLE001 — keep partial run records if budget or transport fails mid-case
                    plain_out = {"arm": "plain", "finished": None, "seen": [], "calls": [],
                                 "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            plain_spent = float(budget.spent_eur())
            spent += plain_spent
            plain_out["euros"] = round(plain_spent, 6)
            plain_out["seconds"] = round(time.monotonic() - t_plain, 1)
        plain_calls[case.id] = list(plain_out.get("calls") or [])
        scores.append(_scored_row(case, plain_out))
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{run_id}_dev.json"
    doc = {"id": run_id, "arms": ["rules", "dclab_agent", "plain"], "model": model, "cap": cap, "spent": round(spent, 6),
           "cases": ran_cases, "not_run": not_run, "scores": scores, "plain_calls": plain_calls,
           "run_dir": str(run_dir.relative_to(ROOT))}
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"id": run_id, "spent": round(spent, 6), "not_run": not_run, "result": str(path)}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.e2e")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "rules", "run"):
        p = sub.add_parser(name)
        p.add_argument("--cases", default=None, help="comma-separated dev case ids (default: all dev cases)")
        p.add_argument("--model", default=MODEL)
        p.add_argument("--cap-eur", type=float, default=None)
        p.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    try:
        selected = _select_cases(args.cases)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    if args.command == "plan":
        print(json.dumps(_plan_doc(selected, args.model, args.cap_eur), indent=1))
        return 0
    if args.command == "rules":
        return _cmd_rules(selected)
    return _cmd_run(selected, args.model, args.cap_eur, args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
