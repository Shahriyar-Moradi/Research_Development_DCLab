"""Execution engine for the General Tabular Pipeline."""

from __future__ import annotations

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
HYPERACK_DIR = ROOT / "hyperack_exp"
sys.path.insert(0, str(HYPERACK_DIR))
sys.path.insert(0, str(ROOT))

from shared.protocol import evaluate  # noqa: E402
from general_pipeline.dataset import load_dataset  # noqa: E402
from general_pipeline.models import MODEL_REGISTRY, get_model  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _serialize(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def run_experiment(
    model_name: str,
    mode: str = "safe",
    optimization: str = "baseline",
    save: bool = True,
) -> Dict[str, Any]:
    """Run a single controlled experiment on the locked dataset split.

    Args:
        model_name: Name of model in MODEL_REGISTRY.
        mode: 'safe' or 'unsafe'.
        optimization: 'baseline' or 'optimized'.
        save: Whether to save result JSON.

    Returns:
        Result dictionary containing metrics, metadata, and timings.
    """
    mode = mode.lower()
    optimization = optimization.lower()
    if mode not in ["safe", "unsafe"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'safe' or 'unsafe'.")
    if optimization not in ["baseline", "optimized"]:
        raise ValueError(f"Invalid optimization: {optimization}. Must be 'baseline' or 'optimized'.")

    # Load data
    X_train, y_train, X_test, y_test, cat_cols, num_cols = load_dataset(mode)

    # Instantiate model
    model = get_model(
        model_name=model_name,
        optimization=optimization,
        mode=mode,
        cat_cols=cat_cols,
        num_cols=num_cols,
    )

    t0 = time.perf_counter()
    metrics = evaluate(model, X_train, y_train, X_test, y_test)
    total_time = time.perf_counter() - t0

    payload = {
        "model_name": model_name,
        "mode": mode,
        "optimization": optimization,
        "feature_count": X_train.shape[1],
        "metrics": _serialize(metrics),
        "total_elapsed_seconds": round(total_time, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if save:
        filename = f"{mode}_{optimization}_{model_name}.json"
        out_path = RESULTS_DIR / filename
        out_path.write_text(json.dumps(payload, indent=2) + "\n")

    tag = f"[{mode.upper()} {optimization.upper()}]"
    auc = metrics.get("roc_auc", 0.0)
    rec = metrics.get("recall", 0.0)
    f1 = metrics.get("f1", 0.0)
    print(f"{tag:<17} {model_name:<24} | ROC-AUC: {auc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ({total_time:.1f}s)", flush=True)

    return payload


def run_quadrant(
    mode: str,
    optimization: str,
    models: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run all or specified models for a given quadrant."""
    if models is None:
        models = MODEL_REGISTRY

    print(f"\n{'='*70}\nSTARTING QUADRANT: mode={mode.upper()} | optimization={optimization.upper()}\n{'='*70}", flush=True)
    results = []
    for m in models:
        try:
            res = run_experiment(m, mode=mode, optimization=optimization)
            results.append(res)
        except Exception as e:
            print(f"Error running {m} ({mode}, {optimization}): {e}", flush=True)

    return results
