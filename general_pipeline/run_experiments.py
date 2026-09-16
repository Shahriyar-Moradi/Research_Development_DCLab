"""Master execution script: Runs the complete 4-way tabular benchmark across all algorithms."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from general_pipeline.benchmark import generate_benchmark_artifacts
from general_pipeline.models import MODEL_REGISTRY
from general_pipeline.runner import run_experiment, run_quadrant


def main():
    parser = argparse.ArgumentParser(description="Run General Tabular Pipeline Benchmark")
    parser.add_argument("--mode", choices=["safe", "unsafe", "all"], default="all", help="Feature mode")
    parser.add_argument("--optimization", choices=["baseline", "optimized", "all"], default="all", help="Optimization level")
    parser.add_argument("--model", choices=MODEL_REGISTRY + ["all"], default="all", help="Specific model to run")
    args = parser.parse_args()

    modes = ["safe", "unsafe"] if args.mode == "all" else [args.mode]
    opts = ["baseline", "optimized"] if args.optimization == "all" else [args.optimization]
    models = MODEL_REGISTRY if args.model == "all" else [args.model]

    t_start = time.perf_counter()
    print(f"{'#'*80}")
    print(f"GENERAL TABULAR PIPELINE: 4-WAY BENCHMARK EXECUTION")
    print(f"Modes: {modes} | Optimizations: {opts}")
    print(f"Models ({len(models)}): {models}")
    print(f"{'#'*80}\n")

    total_runs = len(modes) * len(opts) * len(models)
    completed = 0

    for mode in modes:
        for opt in opts:
            print(f"\n>>> EXECUTING QUADRANT: mode={mode.upper()} | optimization={opt.upper()}")
            for m in models:
                completed += 1
                print(f"[{completed}/{total_runs}] Running {m} ({mode}, {opt})...")
                try:
                    run_experiment(m, mode=mode, optimization=opt, save=True)
                except Exception as e:
                    print(f"FAILED: {m} on {mode}/{opt}: {e}")

    total_time = time.perf_counter() - t_start
    print(f"\n{'='*80}")
    print(f"All experiments completed in {total_time:.1f}s ({total_time/60:.2f} minutes).")
    print(f"Compiling benchmark tables and generating diagnostic plots...")
    print(f"{'='*80}\n")

    generate_benchmark_artifacts()


if __name__ == "__main__":
    main()
