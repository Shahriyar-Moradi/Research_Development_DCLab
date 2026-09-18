"""Command-line control plane for DCLab R&D evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import build_evidence
from .cycle import build_cycle_command, run_cycle
from .registry import collect_registry
from .report import render_outputs, sync_outputs


def _load(root: Path):
    records, issues = collect_registry(root)
    evidence = build_evidence(root, records, issues)
    return records, evidence


def _summary(evidence: dict) -> None:
    summary = evidence["summary"]
    print(
        f"DCLab R&D: {summary['experiments']} experiments | "
        f"{summary['datasets']} datasets | {summary['deployment_eligible_completed']} eligible results"
    )
    for champion in evidence["champions"]:
        print(
            f"  {champion['dataset']:<18} {champion['model_family']:<24} "
            f"ROC-AUC={champion['roc_auc']:.4f} ({champion['experiment']})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DCLab R&D experiment registry and knowledge automation")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    commands = parser.add_subparsers(dest="command", required=True)
    sync = commands.add_parser("sync", help="rebuild registry, evidence, and research backlog")
    sync.add_argument("--check", action="store_true", help="fail if generated knowledge is stale")
    validate = commands.add_parser("validate", help="validate experiment records and regression gates")
    validate.add_argument("--strict", action="store_true", help="treat warnings as failures")
    commands.add_parser("status", help="show the current deployment-eligible champions")
    baseline = commands.add_parser("baseline", help="approve current champion ROC-AUC values as regression baseline")
    baseline.add_argument("--tolerance", type=float, default=0.002)
    cycle = commands.add_parser("cycle", help="run a controlled experiment suite, then refresh research memory")
    cycle.add_argument("suite", choices=["hyperack", "external", "playbook"])
    cycle.add_argument("--dataset", help="dataset key (required for external/playbook)")
    cycle.add_argument("--model", default="logistic_regression", help="model name or comma-list accepted by the selected runner")
    cycle.add_argument("--mode", choices=["safe", "unsafe", "all"], default="safe")
    cycle.add_argument("--optimization", choices=["baseline", "optimized", "all"], default="baseline")
    cycle.add_argument("--reports-only", action="store_true", help="playbook: regenerate reports without training")
    cycle.add_argument("--dry-run", action="store_true", help="print the exact command without running it")
    args = parser.parse_args(argv)

    root = args.root.resolve()

    if args.command == "cycle":
        try:
            command = build_cycle_command(
                suite=args.suite,
                dataset=args.dataset,
                model=args.model,
                mode=args.mode,
                optimization=args.optimization,
                reports_only=args.reports_only,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print("Experiment command:", " ".join(command))
        if args.dry_run:
            return 0
        code = run_cycle(root, command)
        if code:
            print(f"Experiment cycle failed with exit code {code}; knowledge was not refreshed.")
            return code
        records, evidence = _load(root)
        changed = sync_outputs(root, render_outputs(records, evidence))
        print(f"Experiment completed. Refreshed {len(changed)} knowledge artifact(s).")
        _summary(evidence)
        return 0

    records, evidence = _load(root)

    if args.command == "status":
        _summary(evidence)
        return 0

    if args.command == "sync":
        changed = sync_outputs(root, render_outputs(records, evidence), check=args.check)
        _summary(evidence)
        if changed:
            verb = "Stale" if args.check else "Updated"
            print(f"{verb}: {', '.join(changed)}")
        else:
            print("Knowledge artifacts are current.")
        return 1 if args.check and changed else 0

    if args.command == "validate":
        errors = [item for item in evidence["quality_issues"] if item["severity"] == "error"]
        warnings = [item for item in evidence["quality_issues"] if item["severity"] == "warning"]
        regressions = evidence["regressions"]
        print(f"Validation: {len(errors)} errors, {len(warnings)} warnings, {len(regressions)} regressions")
        for item in errors + regressions:
            print(f"  ERROR: {item}")
        if args.strict:
            for item in warnings:
                print(f"  WARNING: {item}")
        return 1 if errors or regressions or (args.strict and warnings) else 0

    if args.command == "baseline":
        if args.tolerance < 0:
            parser.error("--tolerance must be non-negative")
        payload = {
            "schema_version": 1,
            "metric": "roc_auc",
            "tolerance": args.tolerance,
            "champions": {row["dataset"]: row["roc_auc"] for row in evidence["champions"]},
        }
        path = root / "knowledge" / "approved_baseline.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Approved {len(payload['champions'])} dataset baselines in {path.relative_to(root)}")
        return 0

    return 2
