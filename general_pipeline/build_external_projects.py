"""Scaffold HyperAck-style folders for every external dataset and run experiments."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from general_pipeline.external_catalog import DATASET_CATALOG  # noqa: E402
from general_pipeline.external_project import (  # noqa: E402
    PROJECTS_ROOT,
    generate_project_benchmark,
    run_project,
    scaffold_project,
    write_master_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build & run per-dataset external projects")
    parser.add_argument("--dataset", default="all", help="Dataset key or 'all'")
    parser.add_argument("--scaffold-only", action="store_true")
    parser.add_argument("--run", action="store_true", help="Run experiments (default if not scaffold-only)")
    parser.add_argument("--benchmark-only", action="store_true", help="Only rebuild plots/reports from existing results")
    parser.add_argument("--include-transformer", action="store_true")
    parser.add_argument(
        "--model",
        default="all",
        help="Comma-separated models or 'all' for default zoo",
    )
    args = parser.parse_args()

    keys = [s.key for s in DATASET_CATALOG] if args.dataset == "all" else [d.strip() for d in args.dataset.split(",")]
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)

    models = None if args.model == "all" else [m.strip() for m in args.model.split(",")]
    do_run = args.run or (not args.scaffold_only and not args.benchmark_only)

    t0 = time.perf_counter()
    index_lines = ["# External Projects Index", "", "HyperAck-style folders for each public dataset.", ""]

    for key in keys:
        print("\n" + "=" * 72)
        print(f"PROJECT: {key}")
        print("=" * 72)
        pdir = scaffold_project(key)
        if args.benchmark_only:
            df = generate_project_benchmark(key)
            write_master_report(key, df)
        elif do_run and not args.scaffold_only:
            run_project(
                key,
                models=models,
                include_transformer=args.include_transformer,
                save_report=True,
            )
        index_lines.append(f"- [`{key}_exp`]({key}_exp/) — {next(s.name for s in DATASET_CATALOG if s.key == key)}")

    (PROJECTS_ROOT / "README.md").write_text("\n".join(index_lines) + "\n")
    print(f"\nDone in {time.perf_counter() - t0:.1f}s. Projects root: {PROJECTS_ROOT}")


if __name__ == "__main__":
    main()
