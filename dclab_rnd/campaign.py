"""Manifest, orchestration, and durable agent memory for DCLab R&D campaigns."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

CAMPAIGN_ID = "model_building_50_v1"
CAMPAIGN_DIR = Path("campaigns") / CAMPAIGN_ID
DATASETS = (
    ("adult", "Adult (Census Income)", "https://archive.ics.uci.edu/dataset/2/adult"),
    (
        "bank_marketing",
        "Bank Marketing",
        "https://archive.ics.uci.edu/dataset/222/bank+marketing",
    ),
    (
        "breast_cancer",
        "Breast Cancer Wisconsin (Diagnostic)",
        "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
    ),
    (
        "heart_disease",
        "Heart Disease",
        "https://archive.ics.uci.edu/dataset/45/heart+disease",
    ),
    (
        "credit_default",
        "Default of Credit Card Clients",
        "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
    ),
    (
        "german_credit",
        "Statlog German Credit",
        "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
    ),
    ("mushroom", "Mushroom", "https://archive.ics.uci.edu/dataset/73/mushroom"),
    ("spambase", "Spambase", "https://archive.ics.uci.edu/dataset/94/spambase"),
    (
        "online_shoppers",
        "Online Shoppers Purchasing Intention",
        "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset",
    ),
    (
        "wine_quality",
        "Wine Quality (Binary)",
        "https://archive.ics.uci.edu/dataset/186/wine+quality",
    ),
)
EXPERIMENT_TYPES = (
    {
        "kind": "data_understanding",
        "question": "What does the dataset contain, and which quality or production-availability risks must be resolved before modeling?",
        "hypothesis": "A structured train-only profile will expose reliability risks that a model leaderboard hides.",
    },
    {
        "kind": "leakage_audit",
        "question": "Which columns violate the prediction-time contract or behave like suspicious target proxies?",
        "hypothesis": "Decision-time policy plus deterministic heuristics will detect known leakage and produce review candidates without auto-deleting ambiguous features.",
    },
    {
        "kind": "feature_engineering",
        "question": "Which train-fitted feature recipe improves validation evidence without feature dilution?",
        "hypothesis": "The smallest feature set near the best CV score will generalize more reliably than unconstrained feature generation.",
    },
    {
        "kind": "model_selection",
        "question": "Which algorithm family provides the strongest stable training-CV evidence for this dataset profile?",
        "hypothesis": "Algorithm winners vary by dataset; stability-aware multi-family screening is safer than a universal default.",
    },
    {
        "kind": "optimization_reliability",
        "question": "Does optimization produce a meaningful CV gain, and how does the preselected recipe perform on the once-consumed holdout?",
        "hypothesis": "Conservative optimization and one final holdout evaluation will separate repeatable improvements from leaderboard noise.",
    },
)


def campaign_path(root: Path) -> Path:
    return root / CAMPAIGN_DIR


def build_manifest(root: Path) -> dict[str, Any]:
    """Build the deterministic 10-dataset × 5-question campaign manifest."""
    available = {
        item.name
        for item in (root / "external_data").iterdir()
        if item.is_dir()
        and (item / "X.parquet").exists()
        and (item / "y.parquet").exists()
    }
    datasets = [item for item in DATASETS if item[0] in available]
    experiments = []
    number = 0
    for key, name, url in datasets:
        for experiment_type in EXPERIMENT_TYPES:
            number += 1
            experiments.append(
                {
                    "campaign_id": CAMPAIGN_ID,
                    "experiment_id": f"EXP-{number:03d}",
                    "dataset": key,
                    "dataset_name": name,
                    "source": "UCI",
                    "source_url": url,
                    **experiment_type,
                    "status": "planned",
                }
            )
    if len(experiments) != 50:
        raise RuntimeError(
            f"campaign requires exactly 50 experiments; discovered {len(experiments)}"
        )
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "objective": "Learn and preserve an evidence-backed workflow for data understanding, leakage control, feature engineering, model selection, optimization, and production reliability.",
        "scientific_contract": {
            "selection_data": "training-only repeated/stratified cross-validation",
            "final_holdout": "consumed once in optimization_reliability after feature/model selection",
            "leakage": "deterministic rules propose candidates; human decision-time semantics confirm exclusions",
            "llm_authority": "critic/proposer only; metrics and decisions must cite deterministic evidence",
        },
        "datasets": len(datasets),
        "experiment_count": len(experiments),
        "experiments": experiments,
    }


def write_manifest(root: Path) -> Path:
    path = campaign_path(root) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(build_manifest(root), indent=2, sort_keys=True) + "\n"
    if not path.exists() or path.read_text() != content:
        path.write_text(content)
    return path


def _result_path(root: Path, task: dict[str, Any]) -> Path:
    return (
        campaign_path(root)
        / "results"
        / f"{task['experiment_id']}_{task['dataset']}_{task['kind']}.json"
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def load_results(root: Path) -> dict[str, dict[str, Any]]:
    results = {}
    for path in sorted((campaign_path(root) / "results").glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        experiment_id = payload.get("experiment_id")
        if experiment_id:
            payload["_path"] = path.relative_to(root).as_posix()
            results[str(experiment_id)] = payload
    return results


def select_tasks(
    manifest: dict[str, Any],
    *,
    datasets: Iterable[str] | None = None,
    experiments: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    dataset_filter = {item.strip() for item in datasets or [] if item.strip()}
    experiment_filter = {item.strip() for item in experiments or [] if item.strip()}
    tasks = []
    for task in manifest["experiments"]:
        if dataset_filter and task["dataset"] not in dataset_filter:
            continue
        if (
            experiment_filter
            and task["experiment_id"] not in experiment_filter
            and task["kind"] not in experiment_filter
        ):
            continue
        tasks.append(task)
    return tasks


def campaign_status(root: Path) -> dict[str, Any]:
    manifest = build_manifest(root)
    results = load_results(root)
    counts = Counter(result.get("status", "unknown") for result in results.values())
    completed_ids = {
        key for key, value in results.items() if value.get("status") == "completed"
    }
    return {
        "campaign_id": CAMPAIGN_ID,
        "planned": manifest["experiment_count"],
        "completed": len(completed_ids),
        "failed": counts.get("failed", 0),
        "pending": manifest["experiment_count"] - len(completed_ids),
        "datasets_with_final_holdout": len(
            {
                result.get("dataset")
                for result in results.values()
                if result.get("kind") == "optimization_reliability"
                and result.get("status") == "completed"
            }
        ),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _relative_evidence(result: dict[str, Any], suffix: str) -> str:
    return f"{result.get('_path', result.get('experiment_id'))}#{suffix}"


def render_campaign_outputs(root: Path) -> dict[str, str]:
    manifest = build_manifest(root)
    results = load_results(root)
    status = campaign_status(root)
    by_dataset: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in results.values():
        by_dataset[result.get("dataset", "unknown")][result.get("kind", "unknown")] = (
            result
        )
    analyzed_rows = [
        group.get("data_understanding", {}).get("evidence", {}).get("analyzed_rows")
        for group in by_dataset.values()
    ]
    analyzed_rows = [value for value in analyzed_rows if isinstance(value, int)]
    max_analyzed_rows = max(analyzed_rows, default=0)

    report = [
        "# DCLab 50-Experiment Model-Building Campaign",
        "",
        f"Status: **{status['completed']}/{status['planned']} completed**, **{status['failed']} failed**, **{status['pending']} pending**.",
        "",
        "This campaign is intentionally organized as 10 real UCI datasets × 5 scientific questions. Model and feature choices use training-only CV; the final holdout is consumed once per dataset.",
        "",
        "## Dataset decisions",
        "",
        "| Dataset | Leakage exclusions | Feature recipe | CV model candidate | Final recipe | Holdout ROC-AUC | Brier | Evidence |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    model_wins: Counter[str] = Counter()
    stage_wins: Counter[str] = Counter()
    final_rows = []
    for dataset in sorted({task["dataset"] for task in manifest["experiments"]}):
        group = by_dataset.get(dataset, {})
        leak = group.get("leakage_audit", {})
        feature = group.get("feature_engineering", {})
        model = group.get("model_selection", {})
        final = group.get("optimization_reliability", {})
        leak_cols = leak.get("evidence", {}).get("declared_leakage_features")
        stage = feature.get("evidence", {}).get("selected_stage", "—")
        model_name = model.get("evidence", {}).get("selected_model", "—")
        final_evidence = final.get("evidence", {})
        final_recipe = "—"
        if final_evidence:
            final_recipe = f"{final_evidence.get('model')}/{final_evidence.get('selected_optimization')}"
            model_wins[str(final_evidence.get("model"))] += 1
            stage_wins[str(final_evidence.get("feature_stage"))] += 1
            final_rows.append(final)
        metrics = final_evidence.get("holdout_metrics", {})
        evidence_link = f"`{final.get('_path', 'pending')}`" if final else "pending"
        leakage_text = (
            "pending"
            if leak_cols is None
            else (", ".join(leak_cols) or "none declared")
        )
        report.append(
            f"| {dataset} | {leakage_text} | {stage} | {model_name} | {final_recipe} | "
            f"{_fmt(metrics.get('roc_auc'))} | {_fmt(metrics.get('brier'))} | {evidence_link} |"
        )

    report += [
        "",
        "## Data-understanding evidence",
        "",
        "Profiles use the training partition for target-aware statements. Random-split PSI is only a smoke test; it does not replace temporal or operational drift validation.",
        "",
        "| Dataset | Source rows | Analyzed rows | Features | Positive rate | Missing cells | Duplicate rows | Features needing reliability review |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted(by_dataset):
        evidence = by_dataset[dataset].get("data_understanding", {}).get("evidence", {})
        if not evidence:
            continue
        report.append(
            f"| {dataset} | {evidence.get('source_rows', '—')} | {evidence.get('analyzed_rows', '—')} | "
            f"{evidence.get('feature_count', '—')} | {_fmt(evidence.get('positive_rate_train'), 3)} | "
            f"{_fmt(evidence.get('missing_cell_rate_train'), 3)} | {_fmt(evidence.get('duplicate_row_rate_train'), 3)} | "
            f"{evidence.get('risk_feature_count', '—')} |"
        )

    report += [
        "",
        "## Leakage evidence",
        "",
        "The apparent lift is measured on training CV only. A heuristic flag is a request for semantic review, not automatic proof of leakage.",
        "",
        "| Dataset | Declared exclusions | Review candidates | Detector canary | Safe CV ROC-AUC | Unsafe CV ROC-AUC | Apparent leakage lift |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for dataset in sorted(by_dataset):
        evidence = by_dataset[dataset].get("leakage_audit", {}).get("evidence", {})
        if not evidence:
            continue
        comparison = evidence.get("safe_vs_unsafe_training_cv") or {}
        safe_auc = (
            comparison.get("safe", {}).get("metrics", {}).get("roc_auc", {}).get("mean")
        )
        unsafe_auc = (
            comparison.get("unsafe", {})
            .get("metrics", {})
            .get("roc_auc", {})
            .get("mean")
        )
        report.append(
            f"| {dataset} | {', '.join(evidence.get('declared_leakage_features') or []) or 'none'} | "
            f"{len(evidence.get('heuristic_review_candidates') or [])} | {evidence.get('detector_canary_passed')} | "
            f"{_fmt(safe_auc)} | {_fmt(unsafe_auc)} | {_fmt(comparison.get('apparent_auc_lift'))} |"
        )

    report += [
        "",
        "## Feature-engineering evidence",
        "",
        "The selection rule chooses the smallest matrix within 0.002 mean CV ROC-AUC of the best stage. This makes feature dilution a first-class negative result.",
        "",
        "| Dataset | Raw CV ROC-AUC | Selected stage | Selected CV ROC-AUC | Selected − raw | Interpretation |",
        "|---|---:|---|---:|---:|---|",
    ]
    for dataset in sorted(by_dataset):
        evidence = (
            by_dataset[dataset].get("feature_engineering", {}).get("evidence", {})
        )
        if not evidence:
            continue
        delta = evidence.get("selected_vs_raw_auc")
        interpretation = (
            "generated features helped"
            if delta is not None and delta >= 0.002
            else "raw was sufficient within tolerance"
        )
        report.append(
            f"| {dataset} | {_fmt(evidence.get('raw_auc_mean'))} | {evidence.get('selected_stage')} | "
            f"{_fmt(evidence.get('selected_auc_mean'))} | {_fmt(delta)} | {interpretation} |"
        )

    report += [
        "",
        "## Algorithm and optimization evidence",
        "",
        "| Dataset | Selected family | Model-screen CV ROC-AUC | CV std | Accepted configuration | Accepted CV lift vs baseline | Holdout ROC-AUC | Bootstrap 95% interval | ECE |",
        "|---|---|---:|---:|---|---:|---:|---|---:|",
    ]
    for dataset in sorted(by_dataset):
        model_evidence = (
            by_dataset[dataset].get("model_selection", {}).get("evidence", {})
        )
        final_evidence = (
            by_dataset[dataset].get("optimization_reliability", {}).get("evidence", {})
        )
        if not model_evidence or not final_evidence:
            continue
        configurations = final_evidence.get("configuration_results") or []
        baseline_auc = (
            configurations[0].get("metrics", {}).get("roc_auc", {}).get("mean")
            if configurations
            else None
        )
        selected_name = final_evidence.get("selected_optimization")
        selected_row = next(
            (
                item
                for item in configurations
                if item.get("optimization") == selected_name
            ),
            {},
        )
        selected_auc = selected_row.get("metrics", {}).get("roc_auc", {}).get("mean")
        accepted_lift = (
            selected_auc - baseline_auc
            if selected_auc is not None and baseline_auc is not None
            else None
        )
        holdout = final_evidence.get("holdout_metrics", {})
        interval = final_evidence.get("holdout_roc_auc_bootstrap_ci", {})
        interval_text = (
            f"{_fmt(interval.get('low'))}–{_fmt(interval.get('high'))}"
            if interval
            else "—"
        )
        report.append(
            f"| {dataset} | {model_evidence.get('selected_model')} | {_fmt(model_evidence.get('selected_auc_mean'))} | "
            f"{_fmt(model_evidence.get('selected_auc_std'))} | {selected_name} | {_fmt(accepted_lift)} | "
            f"{_fmt(holdout.get('roc_auc'))} | {interval_text} | {_fmt(holdout.get('ece_10'))} |"
        )

    report += [
        "",
        "## Feature-reliability evidence",
        "",
        "These are importance-stability candidates, not automatically approved production features. Names still require decision-time, privacy, availability, and monitoring review.",
        "",
        "| Dataset | Most stable CV features (top-10 frequency) |",
        "|---|---|",
    ]
    for dataset in sorted(by_dataset):
        evidence = (
            by_dataset[dataset].get("optimization_reliability", {}).get("evidence", {})
        )
        stable = evidence.get("cv_feature_stability") or []
        summary = ", ".join(
            f"`{item['feature']}` ({item['top10_frequency']:.0%})"
            for item in stable[:5]
        )
        report.append(f"| {dataset} | {summary or 'importance unavailable'} |")

    report += [
        "",
        "## Cross-dataset evidence",
        "",
        "The counts below describe this campaign only. They are empirical defaults to challenge on new data, not universal laws.",
        "",
        "### Final model-family selections",
        "",
    ]
    if model_wins:
        report.extend(
            f"- `{name}`: {count} dataset(s)"
            for name, count in model_wins.most_common()
        )
    else:
        report.append("- No final selections yet.")
    report += ["", "### Selected feature recipes", ""]
    if stage_wins:
        report.extend(
            f"- `{name}`: {count} dataset(s)"
            for name, count in stage_wins.most_common()
        )
    else:
        report.append("- No feature selections yet.")

    report += [
        "",
        "## Interpretation boundaries",
        "",
        "1. UCI datasets are real public data, but they are benchmarks—not substitutes for a production pilot's prediction-time contract.",
        "2. Cached categoricals were factorized by the earlier downloader. Future ingestion should preserve raw categorical values and fit encoders on training data only.",
        "3. Feature importance is not causality and not proof of production availability.",
        "4. Small score differences require uncertainty and cost/latency interpretation before promotion.",
        "5. The LLM review queue may challenge and propose; it may not rewrite observed metrics or approve deployment.",
        "",
        "## Next evidence priorities",
        "",
        f"1. Re-run close decisions with repeated 5×5 CV and full available rows; this first completed pass analyzed up to {max_analyzed_rows:,} rows per dataset with 3-fold CV.",
        "2. Re-ingest raw categorical values and compare train-fitted encoders or native categorical handling against the legacy globally factorized cache.",
        "3. Add temporal, geographic, source-system, and out-of-domain holdouts where the data-generating process supports them.",
        "4. Add explicit cost matrices and select thresholds on out-of-fold predictions, never on the final holdout.",
        "5. Investigate calibration when ECE or Brier loss is operationally material; Heart Disease is an immediate calibration-review candidate in this pass.",
        "6. Add subgroup/fairness reviews for demographic, credit, and health datasets before any real-world use.",
        "7. Send the 50 bounded tasks in `llm_review_queue.jsonl` to a capable LLM critic, then store only cited, reviewed conclusions—not free-form chat—as durable memory.",
        "",
    ]

    memory_lines = []
    review_lines = []
    for result in sorted(
        results.values(), key=lambda item: item.get("experiment_id", "")
    ):
        for claim in result.get("claims", []):
            memory_lines.append(
                json.dumps(
                    {
                        "schema_version": 1,
                        "campaign_id": CAMPAIGN_ID,
                        "experiment_id": result.get("experiment_id"),
                        "dataset": result.get("dataset"),
                        "kind": result.get("kind"),
                        "claim": claim,
                        "source_path": result.get("_path"),
                        "provenance": result.get("provenance"),
                    },
                    sort_keys=True,
                )
            )
        if (
            result.get("status") == "completed"
            and result.get("llm_review", {}).get("status") == "pending"
        ):
            review_lines.append(
                json.dumps(
                    {
                        "task": "DCLab evidence critic",
                        "experiment_id": result.get("experiment_id"),
                        "dataset": result.get("dataset"),
                        "question": result.get("question"),
                        "claims": result.get("claims", []),
                        "evidence_path": result.get("_path"),
                        "instructions": result.get("llm_review", {}).get(
                            "required_outputs", []
                        ),
                        "rules": [
                            "Separate observed facts, inferences, hypotheses, and recommendations.",
                            "Cite the experiment ID and exact evidence path for every factual statement.",
                            "Never infer causal effects from predictive evidence.",
                            "Never approve a feature without decision-time and production-availability review.",
                        ],
                    },
                    sort_keys=True,
                )
            )

    agent = [
        "# Agent Research Context",
        "",
        "Use this as a retrieval index, not as authority over the underlying result JSON.",
        "",
        "## Operating contract",
        "",
        "- Start with the prediction moment and target definition.",
        "- Treat leakage heuristics as review candidates; confirm semantics with a human/domain source.",
        "- Fit imputers, encoders, feature selection, calibration, and tuning inside training folds.",
        "- Select with training CV and consume the locked holdout once.",
        "- Prefer the simplest recipe within a predeclared tolerance of the best score.",
        "- Evaluate ranking, threshold metrics, calibration, uncertainty, runtime, and production availability.",
        "- Preserve negative results and limitations.",
        "- Never convert importance into causality.",
        "",
        "## Current campaign state",
        "",
        f"- Completed: {status['completed']}/{status['planned']}",
        f"- Datasets with final holdout evidence: {status['datasets_with_final_holdout']}/10",
        "- Machine-readable claims: `agent_memory.jsonl`",
        "- Pending critic tasks: `llm_review_queue.jsonl`",
        "- Full synthesis: `CAMPAIGN_REPORT.md`",
        "",
        "## Mandatory answer format for an LLM",
        "",
        "1. Observed evidence with citations.",
        "2. Interpretation and uncertainty.",
        "3. Decision or recommendation.",
        "4. Risks and missing evidence.",
        "5. Smallest falsifiable next experiment.",
        "",
    ]

    patterns = """# Reusable Model-Building Workflow

These are the code flows that the campaign executes and tests. They are patterns, not copy/paste-only recipes.

```text
prediction contract
  → immutable source + provenance
  → locked holdout
  → train-only EDA and leakage review
  → fold-fitted feature ablation
  → stability-aware algorithm screen
  → conservative optimization
  → one final holdout evaluation
  → calibration/uncertainty/feature reliability
  → evidence claims + LLM critic queue
```

## Train-only feature fitting

```python
engineer.fit(X_fold_train, y_fold_train)
X_fit = engineer.transform(X_fold_train)
X_valid = engineer.transform(X_fold_valid)
model.fit(X_fit, y_fold_train)
```

## Feature promotion rule

```text
decision-time available
AND stable across validation folds
AND reproducibly generated
AND contractually obtainable in production
AND monitored for missingness and drift
AND improves a predeclared metric/cost objective
```

## Model selection rule

```text
Compare multiple families on identical folds.
Rank by validation evidence and stability, then runtime/cost.
Optimize only the selected family.
Use the final holdout once after all choices are locked.
```

Implementation: `dclab_rnd/science.py`; orchestration: `dclab_rnd/campaign.py`.
"""

    return {
        "CAMPAIGN_REPORT.md": "\n".join(report),
        "AGENT_CONTEXT.md": "\n".join(agent),
        "MODEL_BUILDING_WORKFLOW.md": patterns,
        "agent_memory.jsonl": "\n".join(memory_lines) + ("\n" if memory_lines else ""),
        "llm_review_queue.jsonl": "\n".join(review_lines)
        + ("\n" if review_lines else ""),
    }


def sync_campaign_outputs(root: Path, *, check: bool = False) -> list[str]:
    base = campaign_path(root)
    base.mkdir(parents=True, exist_ok=True)
    changed = []
    for name, content in render_campaign_outputs(root).items():
        path = base / name
        if not path.exists() or path.read_text() != content:
            changed.append(path.relative_to(root).as_posix())
            if not check:
                path.write_text(content)
    return changed


def validate_campaign(root: Path) -> list[str]:
    """Validate completeness, final evidence, provenance, and generated memory."""
    issues: list[str] = []
    manifest = build_manifest(root)
    expected = {item["experiment_id"]: item for item in manifest["experiments"]}
    results = load_results(root)
    missing = sorted(set(expected) - set(results))
    extra = sorted(set(results) - set(expected))
    if missing:
        issues.append(f"missing result IDs: {', '.join(missing)}")
    if extra:
        issues.append(f"unknown result IDs: {', '.join(extra)}")

    final_datasets = set()
    for experiment_id, result in sorted(results.items()):
        if result.get("status") != "completed":
            issues.append(f"{experiment_id}: status={result.get('status')}")
            continue
        if not isinstance(result.get("provenance"), dict):
            issues.append(f"{experiment_id}: missing provenance")
        claims = result.get("claims")
        if not isinstance(claims, list) or not claims:
            issues.append(f"{experiment_id}: missing evidence claims")
        if result.get("kind") == "optimization_reliability":
            final_datasets.add(result.get("dataset"))
            roc_auc = result.get("metrics", {}).get("roc_auc")
            if not isinstance(roc_auc, (float, int)) or not 0 <= roc_auc <= 1:
                issues.append(f"{experiment_id}: invalid final ROC-AUC")
            if result.get("evidence", {}).get("holdout_consumed") is not True:
                issues.append(f"{experiment_id}: final holdout evidence is missing")
    if len(final_datasets) != 10:
        issues.append(
            f"expected final evidence for 10 datasets; found {len(final_datasets)}"
        )

    manifest_path = campaign_path(root) / "manifest.json"
    expected_manifest = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if not manifest_path.exists() or manifest_path.read_text() != expected_manifest:
        issues.append("manifest.json is stale")
    stale = sync_campaign_outputs(root, check=True)
    if stale:
        issues.append("stale generated campaign artifacts: " + ", ".join(stale))
    return issues


def run_campaign(
    root: Path,
    *,
    datasets: Iterable[str] | None = None,
    experiments: Iterable[str] | None = None,
    force: bool = False,
    quick: bool = False,
    max_rows: int | None = 10_000,
    fail_fast: bool = False,
) -> int:
    """Run selected tasks with resumable, atomic evidence writes."""
    manifest = build_manifest(root)
    write_manifest(root)
    tasks = select_tasks(manifest, datasets=datasets, experiments=experiments)
    if not tasks:
        raise ValueError("no campaign tasks matched the requested filters")

    from dclab_rnd.science import run_task

    failures = 0
    results_dir = campaign_path(root) / "results"
    for index, task in enumerate(tasks, start=1):
        path = _result_path(root, task)
        if path.exists() and not force:
            try:
                existing = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                existing = {}
            if existing.get("status") == "completed":
                print(
                    f"[{index}/{len(tasks)}] SKIP {task['experiment_id']} {task['dataset']}/{task['kind']}",
                    flush=True,
                )
                continue
        print(
            f"[{index}/{len(tasks)}] RUN  {task['experiment_id']} {task['dataset']}/{task['kind']}",
            flush=True,
        )
        try:
            result = run_task(
                root,
                task,
                results_dir=results_dir,
                max_rows=max_rows,
                quick=quick,
            )
        except Exception as exc:  # noqa: BLE001 - each task must preserve failure evidence and let the campaign continue
            failures += 1
            result = {
                "schema_version": 2,
                "campaign_id": CAMPAIGN_ID,
                "experiment_id": task["experiment_id"],
                "dataset": task["dataset"],
                "kind": task["kind"],
                "question": task["question"],
                "hypothesis": task["hypothesis"],
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            _write_json_atomic(path, result)
            sync_campaign_outputs(root)
            if fail_fast:
                return 1
            continue
        _write_json_atomic(path, result)
        sync_campaign_outputs(root)
        print(f"  SAVED {path.relative_to(root)}", flush=True)

    changed = sync_campaign_outputs(root)
    current = campaign_status(root)
    print(
        f"Campaign: {current['completed']}/{current['planned']} completed, "
        f"{current['failed']} failed, {current['pending']} pending."
    )
    if changed:
        print("Refreshed: " + ", ".join(changed))
    return 1 if failures else 0
