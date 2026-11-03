"""A against B (B6 of the 2026-11-03 plan): any two arms on the same cases, the lead agent against Claude alone made general.

The conversation exam already runs several arms on the same people (``plain``: Claude alone; ``case``: the lead agent; Home's
``rules``, ``model`` and ``plain``) and keeps every person's row per arm. This reads two arms of one result, pairs them by case,
and says, per metric, how B did against A: both means, B's difference with its 95% bootstrap interval over the cases, wins,
losses and ties, the exact McNemar p-value for yes/no metrics, and euro and seconds per case (``dclab_core.bench.paired``).

A gate may be given (``--gate FILE``: ``{"metric": ..., "min_difference": ..., "lower_bound_above": 0}``); the record keeps the
gate file's hash and its modification time beside the result's own date, so whether the gate was written before the run can be
checked rather than claimed. Each comparison is written once to ``evidence/bench/compare/`` (never over an existing one), and
sent to MLflow as a run when ``DCLAB_TRACE_EXPORT`` names it. Rows a judge did not score, or that failed to run, are left out
and counted.

    python -m dclab_rnd.bench compare --result conversation_exam_v1/CEX-032_sealed_v3 --a plain --b case
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dclab_core import bench as core

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS = ROOT / "evidence" / "campaigns"
OUT = ROOT / "evidence" / "bench" / "compare"
# metric -> higher is better; the per-case rows of the exam and Home carry these
METRICS = {"understanding": True, "route_right": True, "trap_pass": True, "safety_failures": False, "asked": False,
           "already_answered": False, "spent_eur": False, "seconds_per_turn": False}


def arms_of(doc: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    scores = doc.get("scores")
    return {k: v for k, v in scores.items() if isinstance(v, list)} if isinstance(scores, dict) else {}


def values(rows: list[dict[str, Any]], metric: str) -> tuple[dict[str, float], int]:
    """Case id -> the metric as a number (a flag as 1 or 0), for rows judged and run; and how many rows were left out."""
    out, left = {}, 0
    for r in rows:
        v = r.get(metric)
        if r.get("failed") or r.get("unjudged") or r.get("judged") is False or v is None or not r.get("id"):
            left += 1
            continue
        out[str(r["id"])] = float(v) if not isinstance(v, bool) else (1.0 if v else 0.0)
    return out, left


def compare(doc: dict[str, Any], a: str, b: str, metrics: Optional[list[str]] = None) -> dict[str, Any]:
    arms = arms_of(doc)
    if a not in arms or b not in arms:
        raise ValueError(f"the result has arms {sorted(arms)}; asked for {a} and {b}")
    out = {}
    for metric in metrics or [m for m in METRICS if any(m in r for r in arms[a])]:
        va, left_a = values(arms[a], metric)
        vb, left_b = values(arms[b], metric)
        out[metric] = {**core.paired(va, vb, higher_is_better=METRICS.get(metric, True)), "left_out_a": left_a, "left_out_b": left_b}
    return out


def judge(by_metric: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    m = by_metric.get(gate.get("metric"))
    if m is None or m["interval"][0] is None:
        return {"passed": None, "why": "the gate's metric has no paired cases"}
    checks = [("difference", m["difference"] >= float(gate.get("min_difference", 0)))]
    if "lower_bound_above" in gate:
        checks.append(("lower_bound", m["interval"][0] > float(gate["lower_bound_above"])))
    return {"passed": all(ok for _, ok in checks), "checks": dict(checks)}


def run(result: str, a: str, b: str, gate_file: Optional[str] = None, metrics: Optional[list[str]] = None,
        campaigns: Path = CAMPAIGNS, out: Path = OUT) -> Path:
    path = Path(campaigns) / f"{result}.json" if not str(result).endswith(".json") else Path(result)
    if not path.is_file():
        path = Path(campaigns) / result.split("/", 1)[0] / "results" / f"{result.split('/', 1)[-1]}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    by_metric = compare(doc, a, b, metrics)
    record: dict[str, Any] = {"schema": "dclab_rnd.bench_compare.v1", "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "result": path.stem, "suite": path.parent.parent.name, "result_at": doc.get("at"),
                              "split": core.split_of(doc, path.stem), "a": a, "b": b, "metrics": by_metric}
    if gate_file:
        g = Path(gate_file)
        gate = json.loads(g.read_text(encoding="utf-8"))
        record["gate"] = {**gate, "file_sha256": hashlib.sha256(g.read_bytes()).hexdigest(),
                          "file_modified": datetime.fromtimestamp(g.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")}
        record["verdict"] = judge(by_metric, gate)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{record['at'][:19].replace(':', '').replace('-', '')}_{path.stem}_{b}_vs_{a}_{os.getpid()}.json"
    with open(target, "x", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    try:
        from .. import watch_mlflow

        watch_mlflow.log_run("dclab-compare", f"{path.stem}: {b} vs {a}", {"result": path.stem, "a": a, "b": b, "split": record["split"]},
                             {f"{m}.difference": v["difference"] for m, v in by_metric.items() if v["difference"] is not None},
                             tags={"dclab.kind": "compare"})
    except Exception:  # noqa: BLE001 - MLflow is a viewer; the record above is the result
        pass
    return target


__all__ = ["compare", "run", "judge", "values", "METRICS"]
