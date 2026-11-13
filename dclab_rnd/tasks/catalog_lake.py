"""P7: Outcome-labelled planning gold dataset builder (dev benchmarks + live adaptive studio trajectories; NEVER sealed)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "dclab_rnd.planning_gold.v1"


def _sealed_case_ids() -> set[str]:
    path = ROOT / "dclab_rnd" / "agent_eval" / "task_planning_v4" / "cases_sealed.json"
    if not path.is_file():
        return set()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return {str(c.get("id")) for c in (doc.get("cases") or ()) if isinstance(c, dict) and c.get("id")}
    except Exception:
        return set()


def _index_exchanges(roots: list[Path]) -> dict[str, dict[str, Any]]:
    exchanges: dict[str, dict[str, Any]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("**/model_exchanges/**/*.jsonl")):
            if "sealed" in path.as_posix().lower():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line)
                except ValueError:
                    continue
                rid = doc.get("model_request")
                if rid and doc.get("whole") is True:
                    exchanges[str(rid)] = doc
    return exchanges


def build_planning_gold_set(
    *,
    root: Path | None = None,
    out_dataset: Path | None = None,
    out_lake: Path | None = None,
) -> dict[str, Any]:
    """Assemble outcome-labelled P7 planning gold examples from dev + live studio trajectories (sealed strictly excluded)."""
    base = Path(root) if root else ROOT
    sealed_ids = _sealed_case_ids()

    dev_jsonl = base / "training_records" / "datasets" / "task_planning_v4_dev.jsonl"
    results_dir = base / "evidence" / "campaigns" / "task_planning_v4" / "results"
    live_root = base / "agent_runs" / "live_tests"

    dev_result_scores: dict[str, dict[str, Any]] = {}
    if results_dir.is_dir():
        for res_path in sorted(results_dir.glob("*_dev*.json")):
            if "sealed" in res_path.name.lower():
                continue
            try:
                doc = json.loads(res_path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            by_arm = doc.get("scores") or {}
            b0_map = {r["id"]: r for r in (by_arm.get("B0") or ()) if isinstance(r, dict) and r.get("id")}
            b1_map = {r["id"]: r for r in (by_arm.get("B1") or ()) if isinstance(r, dict) and r.get("id")}
            for r in by_arm.get("C") or ():
                if not isinstance(r, dict) or not r.get("id"):
                    continue
                rid = str(r["id"])
                b0 = b0_map.get(rid) or {}
                b1 = b1_map.get(rid) or {}
                dev_result_scores[rid] = {
                    "result_file": res_path.name,
                    "holdout_delta_vs_b0": (
                        round(float(r["holdout_score"]) - float(b0["holdout_score"]), 4)
                        if r.get("holdout_score") is not None and b0.get("holdout_score") is not None
                        else None
                    ),
                    "holdout_delta_vs_b1": (
                        round(float(r["holdout_score"]) - float(b1["holdout_score"]), 4)
                        if r.get("holdout_score") is not None and b1.get("holdout_score") is not None
                        else None
                    ),
                }

    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    if dev_jsonl.is_file():
        for line in dev_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            case_id = str(rec.get("case_id") or "")
            if not case_id or case_id in sealed_ids:
                continue
            key_id = f"{case_id}:{rec.get('seed')}"
            meta = dev_result_scores.get(key_id, {})
            cov = float(rec.get("required_coverage") or 0.0)
            trap = float(rec.get("trap_catch_rate") or 0.0)
            sub = float(rec.get("composite_kind_match") or 0.0)
            passed = cov >= 0.99 and trap >= 0.99 and sub >= 0.99
            sig = hashlib.sha256(f"dev|{key_id}|{rec.get('reply') or ''}".encode("utf-8")).hexdigest()[:16]
            if sig in seen_keys:
                continue
            seen_keys.add(sig)
            rows.append(
                {
                    "schema": SCHEMA,
                    "source": "tp08_dev_live",
                    "case_id": case_id,
                    "seed": rec.get("seed"),
                    "stratum": rec.get("stratum"),
                    "label": "positive" if passed else "negative",
                    "required_coverage": cov,
                    "trap_catch_rate": trap,
                    "composite_kind_match": sub,
                    "holdout_score": rec.get("holdout_score"),
                    "holdout_delta_vs_b0": meta.get("holdout_delta_vs_b0"),
                    "holdout_delta_vs_b1": meta.get("holdout_delta_vs_b1"),
                    "messages": rec.get("messages"),
                    "reply": rec.get("reply"),
                    "reasoning": rec.get("reasoning"),
                    "sealed_excluded": True,
                }
            )

    studio_dirs = sorted(live_root.glob("adaptive_studio_live_*")) if live_root.is_dir() else []
    exchanges = _index_exchanges(studio_dirs)
    for run_dir in studio_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for sc in summary.get("scenarios") or ():
            if not isinstance(sc, dict):
                continue
            sid = str(sc.get("scenario_id") or sc.get("project_id") or "")
            if not sid or sid in sealed_ids:
                continue
            mreq = str(sc.get("model_request") or "")
            ex = exchanges.get(mreq)
            if not ex:
                continue
            passed = bool(sc.get("passed"))
            sig = hashlib.sha256(f"studio|{sid}|{mreq}".encode("utf-8")).hexdigest()[:16]
            if sig in seen_keys:
                continue
            seen_keys.add(sig)
            rows.append(
                {
                    "schema": SCHEMA,
                    "source": "adaptive_studio_live",
                    "case_id": sid,
                    "seed": sc.get("seed", 0),
                    "stratum": sc.get("stratum", "studio_e2e"),
                    "label": "positive" if passed else "negative",
                    "required_coverage": 1.0 if passed else 0.0,
                    "trap_catch_rate": sc.get("trap_catch_rate"),
                    "composite_kind_match": sc.get("composite_kind_match"),
                    "holdout_score": sc.get("holdout_score"),
                    "holdout_delta_vs_b0": sc.get("holdout_delta_vs_b0"),
                    "holdout_delta_vs_b1": None,
                    "messages": ex.get("messages"),
                    "reply": ex.get("reply"),
                    "reasoning": ex.get("reasoning") or ex.get("thinking"),
                    "sealed_excluded": True,
                }
            )

    dataset_path = out_dataset or (base / "training_records" / "datasets" / "planning_gold_v1.jsonl")
    lake_path = out_lake or (base / "lake" / "gold" / "planning" / "dev.jsonl")
    for p in (dataset_path, lake_path):
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    positives = sum(1 for r in rows if r["label"] == "positive")
    negatives = sum(1 for r in rows if r["label"] == "negative")
    return {
        "schema": SCHEMA,
        "total": len(rows),
        "positive": positives,
        "negative": negatives,
        "sources": sorted({str(r["source"]) for r in rows}),
        "sealed_excluded": True,
        "dataset_path": str(dataset_path),
        "lake_path": str(lake_path),
    }


__all__ = ["SCHEMA", "build_planning_gold_set"]
