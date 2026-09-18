"""Command-line control plane for DCLab R&D evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import build_evidence
from .campaign import (
    campaign_status,
    run_campaign,
    sync_campaign_outputs,
    validate_campaign,
    write_manifest,
)
from .llm_review import DEFAULT_MODEL, run_llm_reviews
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
    parser = argparse.ArgumentParser(
        description="DCLab R&D experiment registry and knowledge automation"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    commands = parser.add_subparsers(dest="command", required=True)
    sync = commands.add_parser(
        "sync", help="rebuild registry, evidence, and research backlog"
    )
    sync.add_argument(
        "--check", action="store_true", help="fail if generated knowledge is stale"
    )
    validate = commands.add_parser(
        "validate", help="validate experiment records and regression gates"
    )
    validate.add_argument(
        "--strict", action="store_true", help="treat warnings as failures"
    )
    commands.add_parser("status", help="show the current deployment-eligible champions")
    baseline = commands.add_parser(
        "baseline",
        help="approve current champion ROC-AUC values as regression baseline",
    )
    baseline.add_argument("--tolerance", type=float, default=0.002)
    cycle = commands.add_parser(
        "cycle", help="run a controlled experiment suite, then refresh research memory"
    )
    cycle.add_argument("suite", choices=["hyperack", "external", "playbook"])
    cycle.add_argument("--dataset", help="dataset key (required for external/playbook)")
    cycle.add_argument(
        "--model",
        default="logistic_regression",
        help="model name or comma-list accepted by the selected runner",
    )
    cycle.add_argument("--mode", choices=["safe", "unsafe", "all"], default="safe")
    cycle.add_argument(
        "--optimization", choices=["baseline", "optimized", "all"], default="baseline"
    )
    cycle.add_argument(
        "--reports-only",
        action="store_true",
        help="playbook: regenerate reports without training",
    )
    cycle.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact command without running it",
    )
    campaign = commands.add_parser(
        "campaign", help="plan, run, and summarize the 50-experiment knowledge campaign"
    )
    campaign_actions = campaign.add_subparsers(dest="campaign_action", required=True)
    campaign_actions.add_parser(
        "plan", help="write the deterministic 50-experiment manifest and empty reports"
    )
    campaign_actions.add_parser("status", help="show resumable campaign progress")
    campaign_actions.add_parser(
        "report", help="rebuild campaign report and LLM-agent memory"
    )
    campaign_actions.add_parser(
        "verify",
        help="validate all results, provenance, final evidence, and generated memory",
    )
    campaign_review = campaign_actions.add_parser(
        "review",
        help="run pending LLM critic reviews from llm_review_queue.jsonl via OpenAI",
    )
    campaign_review.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL})",
    )
    campaign_review.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max number of pending reviews to process",
    )
    campaign_review.add_argument(
        "--experiment",
        default=None,
        help="comma-separated experiment IDs (e.g. EXP-001,EXP-002)",
    )
    campaign_review.add_argument(
        "--force",
        action="store_true",
        help="re-review all completed experiments (ignore pending queue)",
    )
    campaign_run = campaign_actions.add_parser(
        "run", help="execute campaign tasks and capture evidence"
    )
    campaign_run.add_argument(
        "--dataset", default="all", help="dataset key, comma-list, or all"
    )
    campaign_run.add_argument(
        "--experiment",
        default="all",
        help="experiment ID, kind, comma-list, or all",
    )
    campaign_run.add_argument(
        "--max-rows",
        type=int,
        default=10_000,
        help="deterministic analysis cap per dataset",
    )
    campaign_run.add_argument(
        "--quick",
        action="store_true",
        help="one CV repeat for a faster evidence-building pass",
    )
    campaign_run.add_argument(
        "--force", action="store_true", help="re-run completed matching tasks"
    )
    campaign_run.add_argument(
        "--fail-fast", action="store_true", help="stop after the first failed task"
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()

    if args.command == "campaign":
        if args.campaign_action == "plan":
            path = write_manifest(root)
            changed = sync_campaign_outputs(root)
            print(f"Planned 50 experiments in {path.relative_to(root)}")
            if changed:
                print("Created/refreshed: " + ", ".join(changed))
            return 0
        if args.campaign_action == "status":
            current = campaign_status(root)
            print(
                f"{current['campaign_id']}: {current['completed']}/{current['planned']} completed | "
                f"{current['failed']} failed | {current['pending']} pending | "
                f"{current['datasets_with_final_holdout']}/10 final holdouts"
            )
            return 1 if current["failed"] else 0
        if args.campaign_action == "report":
            write_manifest(root)
            changed = sync_campaign_outputs(root)
            print(
                "Campaign knowledge refreshed."
                if changed
                else "Campaign knowledge is current."
            )
            for path in changed:
                print(f"  {path}")
            return 0
        if args.campaign_action == "verify":
            issues = validate_campaign(root)
            if issues:
                print(f"Campaign verification failed with {len(issues)} issue(s):")
                for issue in issues:
                    print(f"  ERROR: {issue}")
                return 1
            print(
                "Campaign verification passed: 50 results, 10 final holdouts, "
                "provenance and generated memory are current."
            )
            return 0
        if args.campaign_action == "review":
            experiment_ids = (
                None
                if not args.experiment
                else [part.strip() for part in args.experiment.split(",") if part.strip()]
            )
            try:
                return run_llm_reviews(
                    root,
                    model=args.model,
                    limit=args.limit,
                    experiment_ids=experiment_ids,
                    force=args.force,
                )
            except RuntimeError as exc:
                print(f"ERROR: {exc}")
                return 1
        datasets = None if args.dataset == "all" else args.dataset.split(",")
        experiments = None if args.experiment == "all" else args.experiment.split(",")
        if args.max_rows < 100:
            parser.error("--max-rows must be at least 100")
        try:
            return run_campaign(
                root,
                datasets=datasets,
                experiments=experiments,
                force=args.force,
                quick=args.quick,
                max_rows=args.max_rows,
                fail_fast=args.fail_fast,
            )
        except ValueError as exc:
            parser.error(str(exc))

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
            print(
                f"Experiment cycle failed with exit code {code}; knowledge was not refreshed."
            )
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
        changed = sync_outputs(
            root, render_outputs(records, evidence), check=args.check
        )
        _summary(evidence)
        if changed:
            verb = "Stale" if args.check else "Updated"
            print(f"{verb}: {', '.join(changed)}")
        else:
            print("Knowledge artifacts are current.")
        return 1 if args.check and changed else 0

    if args.command == "validate":
        errors = [
            item for item in evidence["quality_issues"] if item["severity"] == "error"
        ]
        warnings = [
            item for item in evidence["quality_issues"] if item["severity"] == "warning"
        ]
        regressions = evidence["regressions"]
        print(
            f"Validation: {len(errors)} errors, {len(warnings)} warnings, {len(regressions)} regressions"
        )
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
            "champions": {
                row["dataset"]: row["roc_auc"] for row in evidence["champions"]
            },
        }
        path = root / "knowledge" / "approved_baseline.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(
            f"Approved {len(payload['champions'])} dataset baselines in {path.relative_to(root)}"
        )
        return 0

    return 2
