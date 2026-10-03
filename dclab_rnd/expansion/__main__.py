"""CLI: ``python3 -m dclab_rnd.expansion {run,status,report}``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dclab_rnd.expansion.datasets import DATASET_ORDER, default_root
from dclab_rnd.expansion.runner import STAGES, run_campaign, status, write_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m dclab_rnd.expansion", description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root(), help="repository root")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run experiments (resumable; skips completed results unless --force)")
    run.add_argument("--dataset", action="append", choices=DATASET_ORDER, help="limit to dataset (repeatable)")
    run.add_argument("--stage", action="append", choices=STAGES, help="limit to stage kind (repeatable)")
    run.add_argument("--force", action="store_true", help="rerun even if a completed result exists")
    sub.add_parser("status", help="show which experiments are completed")
    sub.add_parser("report", help="regenerate manifest.json, CAMPAIGN_REPORT.md, agent_memory.jsonl")
    args = parser.parse_args(argv)
    root = args.root.resolve()

    if args.command == "run":
        started = time.perf_counter()
        executed = run_campaign(root, datasets=args.dataset, stages=args.stage, force=args.force)
        print(f"executed {len(executed)} experiment(s) in {time.perf_counter() - started:.1f}s")
        return 0
    if args.command == "status":
        rows = status(root)
        for row in rows:
            elapsed = f"{row['elapsed_seconds']:.1f}s" if row["elapsed_seconds"] is not None else ""
            print(f"{row['experiment_id']}  {row['status']:<10} {row['dataset']:<28} {row['kind']:<26} {elapsed}")
        done = sum(row["status"] == "completed" for row in rows)
        print(f"{done}/{len(rows)} completed")
        return 0
    paths = write_outputs(root)
    for name, path in paths.items():
        print(f"wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
