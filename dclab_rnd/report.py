"""Deterministic generated artifacts for the DCLab R&D knowledge base."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .record import ExperimentRecord


def _fmt(value: object, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _registry_csv(records: list[ExperimentRecord]) -> str:
    if not records:
        return ""
    output = io.StringIO()
    fieldnames = list(records[0].to_dict())
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(record.to_dict())
    return output.getvalue()


def _knowledge_markdown(evidence: dict) -> str:
    summary = evidence["summary"]
    lines = [
        "# DCLab R&D Knowledge Base",
        "",
        "This file is generated from the repository's experiment JSON. Run `python -m dclab_rnd sync` after every experiment cycle.",
        "",
        "## Evidence inventory",
        "",
        f"- **{summary['experiments']}** normalized experiment records across **{summary['datasets']}** datasets and **{summary['suites']}** suites.",
        f"- **{summary['deployment_eligible_completed']}** completed results are eligible for deployment comparisons.",
        f"- **{summary['with_provenance']}** records contain full code/data/runtime provenance.",
        "",
        "## Deployment-eligible champions",
        "",
        "| Dataset | Model family | Experiment | ROC-AUC | F1 | Recall | Evidence |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in evidence["champions"]:
        lines.append(
            f"| {row['dataset']} | {row['model_family']} | {row['experiment']} | {_fmt(row['roc_auc'])} | "
            f"{_fmt(row['f1'])} | {_fmt(row['recall'])} | `{row['source_path']}` |"
        )

    lines += [
        "",
        "## Cross-dataset model evidence",
        "",
        "Each model family contributes at most one best deployment-eligible result per dataset, preventing large experiment suites from dominating the conclusion.",
        "",
        "| Model family | Datasets | Wins | Mean rank | Mean ROC-AUC | Range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in evidence["model_evidence"]:
        lines.append(
            f"| {row['model_family']} | {row['dataset_count']} | {row['wins']} | {_fmt(row['mean_rank'], 2)} | "
            f"{_fmt(row['mean_roc_auc'])} | {_fmt(row['min_roc_auc'])}–{_fmt(row['max_roc_auc'])} |"
        )

    lines += [
        "",
        "## Leakage evidence",
        "",
        "| Dataset | Best safe ROC-AUC | Best unsafe ROC-AUC | Apparent lift |",
        "|---|---:|---:|---:|",
    ]
    for row in evidence["leakage_gaps"]:
        lines.append(
            f"| {row['dataset']} | {_fmt(row['safe_roc_auc'])} | {_fmt(row['unsafe_roc_auc'])} | "
            f"{row['apparent_leakage_lift']:+.4f} |"
        )
    if not evidence["leakage_gaps"]:
        lines.append("| — | — | — | — |")

    lines += [
        "",
        "## Interpretation rules",
        "",
        "1. Deployment comparisons exclude known post-outcome feature leakage.",
        "2. A point-estimate win is a hypothesis until repeated-CV or bootstrap uncertainty supports it.",
        "3. Cross-dataset mean rank is stronger evidence for a default algorithm than one dataset's highest ROC-AUC.",
        "4. Runtime, calibration, recall, and business cost remain selection constraints even when ROC-AUC is primary.",
        "",
        "See [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md) for the evidence-driven backlog and [DATA_QUALITY.md](DATA_QUALITY.md) for registry warnings.",
        "",
    ]
    return "\n".join(lines)


def _next_markdown(evidence: dict) -> str:
    lines = [
        "# Next Experiments",
        "",
        "Generated from gaps in the current evidence registry. Complete Priority 1 items before adding more unconstrained model searches.",
        "",
    ]
    for item in evidence["recommendations"]:
        lines += [
            f"## P{item['priority']} — {item['title']}",
            "",
            f"**Why:** {item['why']}",
            "",
            f"**Experiment:** {item['experiment']}",
            "",
            f"**Success gate:** {item['success']}",
            "",
        ]
    return "\n".join(lines)


def _quality_markdown(evidence: dict) -> str:
    issues = evidence["quality_issues"]
    errors = [item for item in issues if item["severity"] == "error"]
    warnings = [item for item in issues if item["severity"] == "warning"]
    regressions = evidence["regressions"]
    lines = [
        "# R&D Data Quality",
        "",
        f"- Errors: **{len(errors)}**",
        f"- Warnings: **{len(warnings)}**",
        f"- Benchmark regressions: **{len(regressions)}**",
        "",
    ]
    if regressions:
        lines += ["## Regressions", ""]
        for item in regressions:
            lines.append(f"- `{item['dataset']}`: {item['message']}")
        lines.append("")
    if issues:
        lines += ["## Registry issues", ""]
        for item in issues:
            lines.append(f"- **{item['severity'].upper()}** `{item['path']}` — {item['message']}")
        lines.append("")
    else:
        lines += ["No registry issues detected.", ""]
    return "\n".join(lines)


def render_outputs(records: list[ExperimentRecord], evidence: dict) -> dict[str, str]:
    return {
        "experiment_registry.csv": _registry_csv(records),
        "evidence.json": json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        "KNOWLEDGE_BASE.md": _knowledge_markdown(evidence),
        "NEXT_EXPERIMENTS.md": _next_markdown(evidence),
        "DATA_QUALITY.md": _quality_markdown(evidence),
    }


def sync_outputs(root: Path, outputs: dict[str, str], *, check: bool = False) -> list[str]:
    knowledge = root / "knowledge"
    changed = []
    for name, content in outputs.items():
        path = knowledge / name
        if not path.exists() or path.read_text() != content:
            changed.append(path.relative_to(root).as_posix())
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
    return changed

