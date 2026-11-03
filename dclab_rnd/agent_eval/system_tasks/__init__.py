"""System tasks (B7 of the 2026-11-03 plan): the whole system on one problem, per kind, under a gate fixed before any run.

Each task is one problem kind's acceptance (``kinds.acceptance.run``): a project built from a seeded replica with the kind's
traps planted, the five stages run by the engine under the workflow graph's gates, and every decision a careful data
scientist reaches on it checked (the holdout produced and read once, the planted leaks and identifiers kept out, the
baselines computed). Nothing calls a model; no euro is spent.

The kinds campaign (``kinds_v1``, 16.14) already ran every kind on seeds 0, 1 and 2 without a gate. This suite adds:

- **the kinds that count** are those supported in ``kinds/capability.json`` when this suite was written (``KINDS``, fixed here);
- **dev seeds** 3 and 4, new to every earlier run, for practice;
- **sealed seeds** 731 and 732, read once (``SEALED_SHA`` pins them; a second sealed run is refused);
- ``system_gate_v1``, fixed here before any run: a task passes when every decision of its kind holds on its seed; the suite
  passes when the share of tasks passed has a 95% Wilson lower bound of 0.8 or more and no kind fails on every seed.

    python -m dclab_rnd.agent_eval.system_tasks --split dev
    python -m dclab_rnd.agent_eval.system_tasks --split sealed --yes      # once
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from dclab_core import bench as core

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "evidence" / "campaigns" / "system_tasks_v1"
KINDS = ("rare_event", "entity_tables", "forecasting", "matching", "ranking", "segmentation", "time_to_event", "uplift",
         "streams", "places", "text_tabular", "anomaly", "binary", "multiclass", "regression")
SPLITS = {"dev": (3, 4), "sealed": (731, 732)}
SEALED_SHA = hashlib.sha256(json.dumps([list(KINDS), list(SPLITS["sealed"])]).encode()).hexdigest()
GATE = {"version": "system_gate_v1", "task": "every decision of the kind holds on the seed",
        "suite_lower_bound": 0.8, "no_kind_fails_every_seed": True}


def _runner(kind_key: str, seed: int) -> dict[str, Any]:
    from dclab_rnd import kinds
    from dclab_rnd.kinds import acceptance

    kind = next(k for k in kinds.all_kinds() if k.key == kind_key)
    return acceptance.run(kind, seed)


def tasks(split: str) -> list[tuple[str, int]]:
    return [(k, s) for k in KINDS for s in SPLITS[split]]


def judge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The gate on the tasks' rows: the share passed with its Wilson interval, and the kinds that failed on every seed."""
    n, k = len(rows), sum(1 for r in rows if r["passed"])
    lo, hi = core.wilson(k, n)
    by_kind: dict[str, list[bool]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r["passed"])
    always = sorted(kind for kind, results in by_kind.items() if results and not any(results))
    passed = bool(n) and lo is not None and lo >= GATE["suite_lower_bound"] and not always
    return {"passed": passed, "tasks": [k, n], "tasks_ci95": [lo, hi], "kinds_failing_every_seed": always,
            "verdict": ("pass" if passed else "fail") + f": {k} of {n} tasks (95% {lo} to {hi})"
                       + (f"; failing on every seed: {', '.join(always)}" if always else "")}


def _next_number() -> int:
    numbers = [int(p.stem.split("-")[1].split("_")[0]) for p in (CAMPAIGN / "results").glob("SYS-*.json")]
    return max(numbers, default=0) + 1


def run(split: str = "dev", yes: bool = False, runner: Optional[Callable[[str, int], dict[str, Any]]] = None,
        campaign: Path = CAMPAIGN) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split is one of {sorted(SPLITS)}")
    results = Path(campaign) / "results"
    if split == "sealed":
        if not yes:
            raise ValueError("the sealed seeds are read once: pass --yes when the dev runs are what they should be")
        if any(results.glob("SYS-*_sealed.json")) or (results / "SEALED_STARTED").exists():
            raise FileExistsError("the sealed seeds were already run (or a run was started and stopped): a second sealed run is refused")
        results.mkdir(parents=True, exist_ok=True)
        (results / "SEALED_STARTED").write_text(datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n")  # a stopped run still counts
    runner = runner or _runner
    rows = []
    for kind, seed in tasks(split):
        started = time.perf_counter()
        try:
            out = runner(kind, seed)
            missing = list(out.get("missing") or [])
            rows.append({"kind": kind, "seed": seed, "passed": out.get("status") == "supported" and not missing,
                         "checks": [len([c for c in out.get("checks", []) if c.get("ok")]), len(out.get("checks", []))],
                         "missing": missing[:20], "seconds": round(time.perf_counter() - started, 1)})
        except Exception as exc:  # noqa: BLE001 - a task that crashes fails, with why
            rows.append({"kind": kind, "seed": seed, "passed": False, "checks": [0, 0], "missing": [f"crashed: {type(exc).__name__}: {str(exc)[:200]}"],
                         "seconds": round(time.perf_counter() - started, 1)})
    verdict = judge(rows)
    results.mkdir(parents=True, exist_ok=True)
    number = len(list(results.glob("SYS-*.json"))) + 1 if campaign != CAMPAIGN else _next_number()
    doc = {"id": f"SYS-{number:03d}", "campaign_id": "system_tasks_v1", "kind": "system_tasks", "status": "completed",
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": split, "seeds": list(SPLITS[split]),
           "kinds": list(KINDS), "gate": GATE, "sealed_sha256": SEALED_SHA if split == "sealed" else None, "spent_eur": 0.0,
           "tasks": verdict["tasks"], "tasks_ci95": verdict["tasks_ci95"], "verdict": verdict["verdict"],
           "kinds_failing_every_seed": verdict["kinds_failing_every_seed"], "rows": rows,
           "rule": "The engine's own decisions on seeded replicas with planted traps, no model: evidence that DCLab decides well on "
                   "these invented problems, not about real data nor production readiness."}
    target = results / f"{doc['id']}_{split}.json"
    with open(target, "x", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    return target


__all__ = ["run", "judge", "tasks", "KINDS", "SPLITS", "GATE", "SEALED_SHA"]
