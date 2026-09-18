"""Run classification experiments across downloaded external datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hyperack_exp"))

from shared.protocol import evaluate  # noqa: E402
from dclab_rnd.provenance import capture_provenance  # noqa: E402
from general_pipeline.external_catalog import DATASET_CATALOG  # noqa: E402
from general_pipeline.external_dataset import list_available_datasets, load_external_dataset  # noqa: E402
from general_pipeline.models import get_model  # noqa: E402

RESULTS_DIR = ROOT / "general_pipeline" / "results_external"
OUTPUTS_DIR = ROOT / "benchmark_outputs" / "external"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
# Ensure output dir exists before any shell redirection (e.g. tee) from callers.

# Keep runtime tractable across 10 datasets × 2 optimization levels
DEFAULT_MODELS = [
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
    "catboost",
]


def _serialize(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items() if k not in {"y_true", "y_pred", "y_prob"}}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def run_one(
    dataset_key: str,
    model_name: str,
    optimization: str = "baseline",
    *,
    save: bool = True,
) -> Dict[str, Any]:
    X_train, y_train, X_test, y_test, cat_cols, num_cols, meta = load_external_dataset(dataset_key)
    model = get_model(
        model_name=model_name,
        optimization=optimization,
        mode="safe",  # use safe-tuned hyperparams for external tabular tasks
        cat_cols=cat_cols,
        num_cols=num_cols,
    )
    t0 = time.perf_counter()
    metrics = evaluate(model, X_train, y_train, X_test, y_test)
    elapsed = time.perf_counter() - t0

    payload = {
        "schema_version": 1,
        "dataset": dataset_key,
        "dataset_name": meta.get("name", dataset_key),
        "model_name": model_name,
        "mode": "external",
        "optimization": optimization,
        "feature_count": int(X_train.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "pos_rate": float(y_train.mean()),
        "metrics": _serialize(metrics),
        "total_elapsed_seconds": round(elapsed, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_url": meta.get("url"),
        "provenance": capture_provenance(
            ROOT,
            data_paths=[
                ROOT / "external_data" / dataset_key / "X.parquet",
                ROOT / "external_data" / dataset_key / "y.parquet",
                ROOT / "external_data" / dataset_key / "meta.json",
            ],
            random_state=42,
        ),
    }

    if save:
        out = RESULTS_DIR / f"{dataset_key}__{optimization}__{model_name}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")

    auc = metrics.get("roc_auc", 0.0)
    f1 = metrics.get("f1", 0.0)
    print(
        f"[{dataset_key:<16}] [{optimization:<9}] {model_name:<24} "
        f"| ROC-AUC: {auc:.4f} | F1: {f1:.4f} | ({elapsed:.1f}s)",
        flush=True,
    )
    return payload


def aggregate_results() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        m = data.get("metrics", {})
        rows.append(
            {
                "dataset": data.get("dataset"),
                "dataset_name": data.get("dataset_name"),
                "model_name": data.get("model_name"),
                "optimization": data.get("optimization"),
                "feature_count": data.get("feature_count"),
                "train_rows": data.get("train_rows"),
                "pos_rate": data.get("pos_rate"),
                "accuracy": m.get("accuracy"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "roc_auc": m.get("roc_auc"),
                "avg_precision": m.get("avg_precision"),
                "elapsed_s": data.get("total_elapsed_seconds"),
                "source_url": data.get("source_url"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    csv_path = OUTPUTS_DIR / "external_multidataset_benchmark.csv"
    df.sort_values(["dataset", "optimization", "roc_auc"], ascending=[True, True, False]).to_csv(
        csv_path, index=False
    )
    print(f"Saved: {csv_path}")

    # Pivot: best ROC per dataset × optimization
    best = (
        df.sort_values("roc_auc", ascending=False)
        .groupby(["dataset", "optimization"], as_index=False)
        .first()[["dataset", "optimization", "model_name", "roc_auc", "f1", "accuracy", "recall", "precision"]]
    )
    best_path = OUTPUTS_DIR / "external_best_by_dataset.csv"
    best.to_csv(best_path, index=False)
    print(f"Saved: {best_path}")

    pivot = df.pivot_table(
        index=["dataset", "model_name"],
        columns="optimization",
        values="roc_auc",
        aggfunc="max",
    )
    pivot_path = OUTPUTS_DIR / "external_pivot_roc_auc.csv"
    pivot.to_csv(pivot_path)
    print(f"Saved: {pivot_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-dataset tabular classification runner")
    parser.add_argument("--dataset", default="all", help="Dataset key or 'all'")
    parser.add_argument("--optimization", choices=["baseline", "optimized", "all"], default="all")
    parser.add_argument(
        "--model",
        default="all",
        help=f"Model name, comma-list, or 'all' (default core: {DEFAULT_MODELS})",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    if args.aggregate_only:
        df = aggregate_results()
        print(df.groupby("dataset")["roc_auc"].max() if not df.empty else "No results yet")
        return

    available = list_available_datasets()
    if not available:
        raise SystemExit("No external datasets found. Run: python general_pipeline/download_external_datasets.py")

    datasets = available if args.dataset == "all" else [d.strip() for d in args.dataset.split(",")]
    for d in datasets:
        if d not in available:
            raise SystemExit(f"Unknown/unavailable dataset '{d}'. Available: {available}")

    if args.model == "all":
        models = DEFAULT_MODELS
    else:
        models = [m.strip() for m in args.model.split(",")]

    opts = ["baseline", "optimized"] if args.optimization == "all" else [args.optimization]

    catalog_names = {s.key: s.name for s in DATASET_CATALOG}
    print("#" * 80)
    print("MULTI-DATASET TABULAR CLASSIFICATION BENCHMARK")
    print(f"Datasets ({len(datasets)}): {[catalog_names.get(d, d) for d in datasets]}")
    print(f"Models ({len(models)}): {models}")
    print(f"Optimizations: {opts}")
    print("#" * 80)

    total = len(datasets) * len(opts) * len(models)
    done = 0
    t0 = time.perf_counter()
    for ds in datasets:
        for opt in opts:
            for model in models:
                done += 1
                print(f"[{done}/{total}] {ds} / {opt} / {model}")
                try:
                    run_one(ds, model, opt, save=True)
                except Exception as exc:
                    print(f"FAILED: {ds}/{opt}/{model}: {exc}", flush=True)

    print(f"\nCompleted in {time.perf_counter() - t0:.1f}s. Aggregating...")
    df = aggregate_results()
    if not df.empty:
        print("\nBest ROC-AUC by dataset (any model/opt):")
        print(
            df.sort_values("roc_auc", ascending=False)
            .groupby("dataset", as_index=False)
            .first()[["dataset", "model_name", "optimization", "roc_auc", "f1"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
