"""DCLab Bench (B3 of the 2026-11-03 plan): every evaluation of DCLab read into one shape, by layer, with a history by commit.

- :func:`evidence_rows` reads every result in ``evidence/campaigns/*/results`` into ``dclab_core.bench.BenchRow`` rows,
  each campaign in its layer (``LAYER_OF``; a campaign with no layer is listed, never guessed).
- :func:`run` runs the free suites of the layers asked for (``COMMANDS``: scripted, no live model, never a sealed or test
  split), times each, and records the run once in ``evidence/bench/runs/<time>_<commit>.json`` (refused if it exists).
- :func:`summary` says each layer's state from its suites' latest results (``dclab_core.bench.layer_status``: red, amber,
  thin, green, empty, with the reasons), the latest result of every suite, and the history of the recorded runs.

Sealed results are read and shown with their date and verdict; nothing here ever runs one. Nothing is generated for the
repository: the summary is computed when it is asked for, so a new result never leaves a stale file behind.

    python -m dclab_rnd.bench read [--json]               # the layers from what is recorded (no run)
    python -m dclab_rnd.bench run [--layers agents,...]   # run the free suites, record the run, then read
    GET /api/admin/bench                                   # the same, on the Admin page's Benchmark tab (the owner's)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from dclab_core import bench as core

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS = ROOT / "evidence" / "campaigns"
RUNS = ROOT / "evidence" / "bench" / "runs"

# every campaign in one layer; a new campaign without a line here is reported as unassigned (and fails a test)
LAYER_OF: dict[str, str] = {
    **{c: "agents" for c in ("agent_eval_v1", "agent_verification_v1", "benchmark_v2", "benchmark_v3", "benchmark_v4", "harness_v1",
                             "journeys_v1", "journeys_programme_v1", "journeys_built_v1", "analyst_v1", "investigator_v1",
                             "brief_checker_v1", "map_roles_v1", "conversation_v1", "learning_v1", "metric_change_v1")},
    **{c: "lead_agent" for c in ("conversation_exam_v1", "home_v1")},
    **{c: "llm_roles" for c in ("roles_v1", "solver_v1", "decision_llm_v1")},
    **{c: "team" for c in ("team_v1", "orchestra_v1", "department_v1")},
    **{c: "decision_models" for c in ("decision_models_v1", "decision_r3_v1", "decision_shadow_v1", "decision_workflow_v1",
                                      "workflow_models_v1", "understand_v1", "students_exam_v1", "committee_v1", "curriculum_v1",
                                      "active_learning_v1", "real_homework_v1", "category_codes_v1", "ability_v1")},
    **{c: "retrieval_memory" for c in ("retrieval_v1", "memory_retrieval_v1", "memory_value_v1", "memory_value_v2")},
    **{c: "data_science" for c in ("automl_v1", "bakeoff_v1", "expansion_v1", "kinds_v1", "model_building_50_v1", "pitfalls_v1",
                                   "pitfalls_by_kind_v1", "search_v1", "search_v2", "search_v3", "fraud_decision_v1",
                                   "fraud_decision_v2", "fraud_decision_v3", "time_decision_v1", "time_decision_v2",
                                   "time_decision_v3", "time_decision_v4", "time_decision_v5", "daily_centre_v1", "daily_ties_v1",
                                   "certification_v1", "foundation_v1", "cascade_v1")},
    **{c: "system_tasks" for c in ("real_problems_v1", "twin_v1", "system_tasks_v1", "task_planning_v1",
                                    "task_planning_v2", "task_planning_v3")},
}

# the free suites a run may execute, by layer: scripted, no live model, never a sealed or test split
COMMANDS: dict[str, list[tuple[str, list[str]]]] = {
    "software": [("product-e2e", ["make", "product-e2e"])],
    "agents": [("agent-eval", ["make", "agent-eval"]), ("harness-eval", ["make", "harness-eval"]),
               ("journeys-dev", ["make", "journeys", "SPLIT=dev"])],
    "retrieval_memory": [("retrieval-eval", ["make", "retrieval-eval"])],
    "data_science": [("kinds-check", ["make", "kinds-check"])],
    "system_tasks": [
        ("system-tasks-dev", [sys.executable, "-m", "dclab_rnd.agent_eval.system_tasks", "--split", "dev"]),
        ("task-planning-offline", [sys.executable, "-m", "dclab_rnd.agent_eval.task_planning_v3", "run-offline"]),
    ],
}
TIMEOUT = 1800  # seconds a command may take before it is stopped and counted as failed


def _compat_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Normalize campaign result shapes whose gate outcome lives under ``judgment`` (read-only view)."""
    judgment = doc.get("judgment")
    if doc.get("verdict") is not None or not isinstance(judgment, dict):
        return doc
    out = dict(doc)
    v1, v2 = judgment.get("verdict"), judgment.get("verdict_v2")
    if isinstance(v1, str) and isinstance(v2, str):
        out["verdict"] = {"gate_v1": v1, "gate_v2": v2}
    elif isinstance(v2 or v1, str):
        out["verdict"] = v2 or v1
    if "gate" not in out:
        out["gate"] = "tp06.gate.v2" if isinstance(v2, str) else "tp06.gate.v1"
    if not out.get("model") and not out.get("models") and out.get("live_model"):
        out["model"] = out["live_model"]
    scores = doc.get("scores")
    if out.get("spent_eur") is None and isinstance(scores, dict):
        out["spent_eur"] = round(sum(
            float(row.get("cost_eur") or 0)
            for arm_rows in scores.values() if isinstance(arm_rows, list)
            for row in arm_rows if isinstance(row, dict)
        ), 4)
    if isinstance(scores, dict) and isinstance(scores.get("C"), list) and scores["C"]:
        c_rows = [row for row in scores["C"] if isinstance(row, dict)]
        if c_rows:
            out.setdefault("arm_c_correct", [sum(bool(row.get("correct")) for row in c_rows), len(c_rows)])
            out.setdefault("arm_c_completed", [sum(bool(row.get("completed")) for row in c_rows), len(c_rows)])
    return out


def evidence_rows(campaigns: Path = CAMPAIGNS) -> tuple[list[core.BenchRow], list[str]]:
    """Every result as rows, and the campaigns that have no layer (left out, named)."""
    rows: list[core.BenchRow] = []
    unassigned: set[str] = set()
    for path in sorted(Path(campaigns).glob("*/results/*.json")):
        suite = path.parent.parent.name
        layer = LAYER_OF.get(suite)
        if layer is None:
            unassigned.add(suite)
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            doc = _compat_doc(doc)
            at = str(doc.get("at") or doc.get("completed_at") or doc.get("created") or "")
            rows += core.rows_of(doc, suite=suite, ref=path.stem, layer=layer, at=at)
    return rows, sorted(unassigned)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def commit() -> tuple[str, bool]:
    """The code's commit and whether the tree had changes (a run on a dirty tree says so)."""
    return _git("rev-parse", "--short=12", "HEAD"), bool(_git("status", "--porcelain", "--untracked-files=no"))


def run(layers: Optional[Iterable[str]] = None, runs: Path = RUNS, execute: Any = None, campaigns: Path = CAMPAIGNS) -> Path:
    """Run the free suites of ``layers`` (all with a command when None), one after another, and record the run once.
    ``execute(argv) -> exit code`` replaces the subprocess (tests). A layer without free suites is refused before anything runs."""
    chosen = list(layers) if layers else list(COMMANDS)
    unknown = [layer for layer in chosen if layer not in COMMANDS]
    if unknown or not chosen:
        raise ValueError(f"no free suites for {', '.join(unknown) or 'nothing'}: choose from {', '.join(COMMANDS)}")
    sha, dirty = commit()
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    target = Path(runs) / f"{stamp}_{sha or 'nocommit'}_{os.getpid()}.json"
    if target.exists():
        raise FileExistsError(f"{target.name} exists: a recorded run is never rewritten")
    done = []
    for layer in chosen:
        for name, argv in COMMANDS[layer]:
            t0 = time.monotonic()
            code = execute(argv) if execute else _execute(argv)
            done.append({"layer": layer, "suite": name, "argv": argv, "exit": code, "seconds": round(time.monotonic() - t0, 1)})
    rows = [core.BenchRow(suite=f"cmd:{d['suite']}", ref=f"{stamp}_{d['suite']}", layer=d["layer"], passed=d["exit"] == 0,
                          verdict="pass" if d["exit"] == 0 else f"fail: exit {d['exit']}", seconds=d["seconds"], commit=sha,
                          at=started.isoformat(timespec="seconds")) for d in done]
    record = {"schema": "dclab_rnd.bench_run.v1", "at": started.isoformat(timespec="seconds"), "commit": sha, "dirty": dirty,
              "layers": chosen, "commands": done, "rows": [json.loads(r.model_dump_json()) for r in rows],
              "machine": {"memory_gb": _memory_gb(), "free_disk_gb": round(shutil.disk_usage(ROOT).free / 1024 ** 3, 1)},
              "states": {layer: s["status"] for layer, s in summary(campaigns=campaigns, runs=runs, extra=rows)["layers"].items()}}
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8") as fh:  # "x": never over an existing record
        json.dump(record, fh, indent=1, ensure_ascii=False)
    return target


def _execute(argv: list[str]) -> int:
    env = {**os.environ, "DCLAB_NO_LIVE_MODELS": "1"}
    try:
        return subprocess.run(argv, cwd=ROOT, env=env, timeout=TIMEOUT).returncode
    except subprocess.TimeoutExpired:
        return 124


def _memory_gb() -> Optional[float]:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def run_rows(runs: Path = RUNS) -> list[dict[str, Any]]:
    out = []
    for path in sorted(Path(runs).glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def summary(campaigns: Path = CAMPAIGNS, runs: Path = RUNS, extra: Iterable[core.BenchRow] = ()) -> dict[str, Any]:
    """Each layer's state and reasons, the latest result of each of its suites, the recorded runs as history, and the
    campaigns with no layer."""
    rows, unassigned = evidence_rows(campaigns)
    recorded = run_rows(runs)
    for doc in recorded:
        rows += [core.BenchRow(**r) for r in doc.get("rows", [])]
    rows += list(extra)
    by_layer: dict[str, list[core.BenchRow]] = defaultdict(list)
    for r in rows:
        by_layer[r.layer].append(r)
    layers = {}
    for layer in core.LAYERS:
        state = core.layer_status(by_layer.get(layer, []))
        layers[layer] = {**state, "suite_count": state["suites"], "suites": _latest(by_layer.get(layer, []))}  # the list replaces the count
    history = [{"at": d.get("at"), "commit": d.get("commit"), "dirty": d.get("dirty"), "states": d.get("states", {}),
                "seconds": round(sum(c.get("seconds") or 0 for c in d.get("commands", [])), 1)} for d in recorded]
    return {"layers": layers, "history": history, "unassigned": unassigned, "results": len({(r.suite, r.ref) for r in rows}),
            "spent_eur": round(sum(r.cost_eur or 0 for r in rows if r.metric == "verdict"), 2)}


def _latest(rows: list[core.BenchRow]) -> list[dict[str, Any]]:
    """Per track of each suite (``dclab_core.bench.track_of``: its split and point), its latest result: verdict, split,
    date, cost, the result's largest count, and how many results the track has."""
    by_track: dict[tuple[str, tuple[str, str]], list[core.BenchRow]] = defaultdict(list)
    for r in rows:
        by_track[(r.suite, core.track_of(r.ref, r.split))].append(r)
    out = []
    for (suite, track), rs in sorted(by_track.items()):
        verdicts = [r for r in rs if r.metric == "verdict"]
        if not verdicts:
            continue
        last = max(verdicts, key=core.order_of)
        counts = [r for r in rs if r.ref == last.ref and r.n]
        main = max(counts, key=lambda r: r.n) if counts else None
        out.append({"suite": suite, "track": " ".join(x for x in track if x and x != "unknown"), "ref": last.ref, "at": last.at, "split": last.split, "passed": last.passed,
                    "verdict": last.verdict, "gate": last.gate, "cost_eur": last.cost_eur, "seconds": last.seconds,
                    "results": len(verdicts), "sealed_results": sum(r.split == "sealed" for r in verdicts),
                    "count": None if main is None else {"metric": main.metric, "k": main.k, "n": main.n,
                                                       "lower": main.lower, "upper": main.upper}})
    return out


__all__ = ["LAYER_OF", "COMMANDS", "evidence_rows", "run", "summary", "commit"]
