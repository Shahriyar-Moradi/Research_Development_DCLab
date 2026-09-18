"""Turn normalized runs into cross-dataset evidence and an experiment backlog."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

from .record import ExperimentRecord


def _eligible(records: Iterable[ExperimentRecord]) -> list[ExperimentRecord]:
    return [
        record
        for record in records
        if record.status == "completed"
        and record.roc_auc is not None
        and record.deployment_eligible is True
    ]


def _champions(records: list[ExperimentRecord]) -> list[dict]:
    by_dataset: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for record in _eligible(records):
        by_dataset[record.dataset].append(record)

    champions = []
    for dataset, group in sorted(by_dataset.items()):
        winner = max(group, key=lambda item: (item.roc_auc or -1.0, item.f1 or -1.0))
        champions.append({
            "dataset": dataset,
            "run_id": winner.run_id,
            "suite": winner.suite,
            "experiment": winner.experiment_name,
            "model": winner.model,
            "model_family": winner.model_family,
            "roc_auc": winner.roc_auc,
            "f1": winner.f1,
            "recall": winner.recall,
            "source_path": winner.source_path,
        })
    return champions


def _model_evidence(records: list[ExperimentRecord]) -> list[dict]:
    # One best result per dataset/family prevents large experiment suites from dominating.
    best: dict[tuple[str, str], ExperimentRecord] = {}
    for record in _eligible(records):
        key = (record.dataset, record.model_family)
        previous = best.get(key)
        if previous is None or (record.roc_auc or -1) > (previous.roc_auc or -1):
            best[key] = record

    ranks: dict[str, list[float]] = defaultdict(list)
    aucs: dict[str, list[float]] = defaultdict(list)
    wins: dict[str, int] = defaultdict(int)
    for dataset in sorted({dataset for dataset, _ in best}):
        rows = sorted(
            (record for (ds, _), record in best.items() if ds == dataset),
            key=lambda item: item.roc_auc or -1,
            reverse=True,
        )
        for rank, record in enumerate(rows, start=1):
            ranks[record.model_family].append(float(rank))
            aucs[record.model_family].append(float(record.roc_auc))
            if rank == 1:
                wins[record.model_family] += 1

    evidence = []
    for family, values in aucs.items():
        evidence.append({
            "model_family": family,
            "dataset_count": len(values),
            "wins": wins[family],
            "mean_roc_auc": mean(values),
            "mean_rank": mean(ranks[family]),
            "min_roc_auc": min(values),
            "max_roc_auc": max(values),
        })
    return sorted(evidence, key=lambda item: (-item["dataset_count"], item["mean_rank"], -item["mean_roc_auc"]))


def _leakage_gaps(records: list[ExperimentRecord]) -> list[dict]:
    by_dataset_mode: dict[tuple[str, str], list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        if record.status == "completed" and record.roc_auc is not None:
            by_dataset_mode[(record.dataset, record.mode)].append(record)

    output = []
    for dataset in sorted({record.dataset for record in records if record.has_known_leakage}):
        safe = by_dataset_mode.get((dataset, "safe"), [])
        unsafe = by_dataset_mode.get((dataset, "unsafe"), [])
        if not safe or not unsafe:
            continue
        safe_best = max(safe, key=lambda item: item.roc_auc or -1)
        unsafe_best = max(unsafe, key=lambda item: item.roc_auc or -1)
        output.append({
            "dataset": dataset,
            "safe_roc_auc": safe_best.roc_auc,
            "unsafe_roc_auc": unsafe_best.roc_auc,
            "apparent_leakage_lift": (unsafe_best.roc_auc or 0) - (safe_best.roc_auc or 0),
            "safe_source": safe_best.source_path,
            "unsafe_source": unsafe_best.source_path,
        })
    return output


def _recommendations(records: list[ExperimentRecord], champions: list[dict], model_evidence: list[dict]) -> list[dict]:
    completed = [record for record in records if record.status == "completed"]
    recs: list[dict] = []
    provenance_rate = sum(record.has_provenance for record in completed) / max(len(completed), 1)
    if provenance_rate < 0.95:
        recs.append({
            "priority": 1,
            "title": "Replicate champions with full provenance",
            "why": f"Only {provenance_rate:.1%} of completed historical runs contain code, data, seed, and environment provenance.",
            "experiment": "Re-run each deployment champion with the new provenance-enabled runners; require identical data hashes and report score drift.",
            "success": "All active champions are reproducible from a clean checkout and differ by no more than 0.001 ROC-AUC.",
        })

    metric_keys = {key for record in completed for key in (record.strategy + " " + record.notes).lower().split()}
    if not any(token in metric_keys for token in {"brier", "ece", "logloss", "log_loss"}):
        recs.append({
            "priority": 1,
            "title": "Add probability-quality benchmarks",
            "why": "The registry is rich in ranking metrics, but it has no systematic Brier score, log loss, or calibration error evidence.",
            "experiment": "Evaluate the top three safe model families with Brier score, log loss, ECE, and calibration curves on the locked split.",
            "success": "Select a calibrated champion that preserves ROC-AUC while improving Brier score and expected business cost.",
        })

    recs.append({
        "priority": 1,
        "title": "Measure winner stability, not only point estimates",
        "why": "Most stored results expose one locked split and no confidence interval, so small leaderboard differences may be noise.",
        "experiment": "Run repeated stratified 5-fold CV (5 repeats) for the top three deployment-eligible families and bootstrap paired score differences.",
        "success": "Promote a winner only when its median lift is positive and the 95% paired interval is decision-useful.",
    })

    if any(item["dataset"] == "hyperack" for item in champions):
        recs.append({
            "priority": 2,
            "title": "Add temporal and operational holdouts for HyperAck",
            "why": "A random locked split can overestimate production performance when pricing, geography, or demand changes over time.",
            "experiment": "Backtest the safe champion on later-time, unseen-zone, and high-demand slices; compare AUC, recall, calibration, and coverage.",
            "success": "Document the worst-slice floor and define a retraining or rollback threshold before deployment.",
        })

    if model_evidence:
        leader = min(
            (item for item in model_evidence if item["dataset_count"] >= 3),
            key=lambda item: (item["mean_rank"], -item["mean_roc_auc"]),
            default=model_evidence[0],
        )
        recs.append({
            "priority": 3,
            "title": f"Challenge the cross-dataset leader: {leader['model_family']}",
            "why": f"It currently covers {leader['dataset_count']} datasets with mean rank {leader['mean_rank']:.2f}; that is evidence, not a universal guarantee.",
            "experiment": "Add three datasets from different row-count, imbalance, and categorical-cardinality regimes and compare rank stability against GBDT and ensemble baselines.",
            "success": "The preferred default is based on broad rank stability plus runtime/cost, with documented exceptions by dataset profile.",
        })
    return sorted(recs, key=lambda item: (item["priority"], item["title"]))


def _regressions(root: Path, champions: list[dict]) -> list[dict]:
    baseline_path = root / "knowledge" / "approved_baseline.json"
    if not baseline_path.exists():
        return []
    import json

    try:
        baseline = json.loads(baseline_path.read_text())
    except (OSError, json.JSONDecodeError):
        return [{"dataset": "*", "message": "approved_baseline.json is invalid"}]
    tolerance = float(baseline.get("tolerance", 0.002))
    current = {item["dataset"]: item["roc_auc"] for item in champions}
    output = []
    for dataset, expected in baseline.get("champions", {}).items():
        actual = current.get(dataset)
        if actual is None:
            output.append({"dataset": dataset, "expected": expected, "actual": None, "message": "champion missing"})
        elif actual < float(expected) - tolerance:
            output.append({
                "dataset": dataset,
                "expected": expected,
                "actual": actual,
                "delta": actual - float(expected),
                "message": "ROC-AUC regression exceeds tolerance",
            })
    return output


def build_evidence(root: Path, records: list[ExperimentRecord], issues: list[dict[str, str]]) -> dict:
    champions = _champions(records)
    model_evidence = _model_evidence(records)
    status_counts: dict[str, int] = defaultdict(int)
    for record in records:
        status_counts[record.status] += 1
    provenance_count = sum(record.has_provenance for record in records)
    return {
        "schema_version": 1,
        "summary": {
            "experiments": len(records),
            "datasets": len({record.dataset for record in records}),
            "suites": len({record.suite for record in records}),
            "deployment_eligible_completed": len(_eligible(records)),
            "with_provenance": provenance_count,
            "status_counts": dict(sorted(status_counts.items())),
        },
        "champions": champions,
        "model_evidence": model_evidence,
        "leakage_gaps": _leakage_gaps(records),
        "recommendations": _recommendations(records, champions, model_evidence),
        "regressions": _regressions(root, champions),
        "quality_issues": issues,
    }

