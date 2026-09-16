"""Per-dataset external project toolkit (HyperAck-style folder layout).

Each project lives at:
  external_projects/<dataset>_exp/
    README.md
    run_all.py
    data/                 -> symlink to ../../external_data/<dataset>
    results/              JSON experiment records
    benchmark_outputs/    CSVs + PNG plots
    MASTER_BENCHMARK_REPORT.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hyperack_exp"))

from shared.protocol import evaluate  # noqa: E402
from general_pipeline.external_catalog import DATASET_CATALOG  # noqa: E402
from general_pipeline.models import MODEL_REGISTRY, get_model  # noqa: E402

PROJECTS_ROOT = ROOT / "external_projects"
EXTERNAL_DATA = ROOT / "external_data"
RANDOM_STATE = 42

# Bank Marketing: call duration is post-outcome leakage (analogous to HyperAck final fares).
LEAKAGE_DROP = {
    "bank_marketing": ["duration"],
}

DEFAULT_MODELS = [
    "logistic_regression",
    "linear_svc",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
    "catboost",
    "stacking_ensemble",
    "soft_vote_blend",
]

# Optional heavier model — include when --full
FULL_EXTRA_MODELS = ["tabular_transformer"]


def project_dir(key: str) -> Path:
    return PROJECTS_ROOT / f"{key}_exp"


def catalog_entry(key: str):
    for spec in DATASET_CATALOG:
        if spec.key == key:
            return spec
    raise KeyError(key)


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


def load_project_xy(
    key: str,
    *,
    mode: str = "safe",
    test_size: float = 0.2,
    max_rows: int | None = 20000,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str], dict]:
    """Load locked split. For datasets with leakage cols, mode safe drops them."""
    ds_dir = EXTERNAL_DATA / key
    X = pd.read_parquet(ds_dir / "X.parquet")
    y = pd.read_parquet(ds_dir / "y.parquet")["target"].astype(int)
    meta = json.loads((ds_dir / "meta.json").read_text())

    leak_cols = LEAKAGE_DROP.get(key, [])
    if mode == "safe" and leak_cols:
        X = X.drop(columns=[c for c in leak_cols if c in X.columns], errors="ignore")
        meta["dropped_leakage"] = leak_cols
    elif mode == "unsafe":
        meta["dropped_leakage"] = []

    if max_rows is not None and len(X) > max_rows:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=RANDOM_STATE)
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        meta["subsampled_to"] = max_rows

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Persist split indices for reproducibility (HyperAck pattern)
    return X_train, y_train, X_test, y_test, [], list(X_train.columns), meta


def scaffold_project(key: str) -> Path:
    """Create HyperAck-style project folder for one dataset."""
    spec = catalog_entry(key)
    pdir = project_dir(key)
    (pdir / "results").mkdir(parents=True, exist_ok=True)
    (pdir / "benchmark_outputs").mkdir(parents=True, exist_ok=True)

    data_link = pdir / "data"
    target = (EXTERNAL_DATA / key).resolve()
    if data_link.is_symlink() or data_link.exists():
        if data_link.is_symlink():
            data_link.unlink()
        elif data_link.is_dir() and not any(data_link.iterdir()):
            data_link.rmdir()
    if not data_link.exists():
        try:
            data_link.symlink_to(os.path.relpath(target, pdir), target_is_directory=True)
        except OSError:
            # Fallback: write a pointer file
            data_link.mkdir(exist_ok=True)
            (data_link / "SOURCE.txt").write_text(f"{target}\n")

    modes_note = (
        "Supports **safe** (drop leakage cols) vs **unsafe** (keep all) plus baseline/optimized."
        if key in LEAKAGE_DROP
        else "Baseline vs optimized across the shared model zoo (no domain leakage split)."
    )
    readme = f"""# {spec.name} experiments

HyperAck-style tabular classification project for **{spec.name}**.

- Source: [{spec.url}]({spec.url})
- Cached data: `data/` → `external_data/{key}/`
- Locked split: stratified 80/20, seed `{RANDOM_STATE}`
- {modes_note}

## Run

```bash
.venv/bin/python external_projects/{key}_exp/run_all.py
# or from toolkit:
.venv/bin/python general_pipeline/build_external_projects.py --dataset {key} --run
```

## Layout

| Path | Purpose |
|---|---|
| `data/` | Features + target parquet + meta |
| `results/` | Per-experiment JSON metrics |
| `benchmark_outputs/` | CSVs + diagnostic PNGs |
| `MASTER_BENCHMARK_REPORT.md` | Project master report |

## Models

{', '.join(DEFAULT_MODELS)}
"""
    (pdir / "README.md").write_text(readme)

    run_all = f'''#!/usr/bin/env python3
"""Run full experiment suite + benchmark for {key}."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from general_pipeline.external_project import run_project  # noqa: E402

if __name__ == "__main__":
    run_project("{key}", models=None, include_transformer=False, save_report=True)
'''
    (pdir / "run_all.py").write_text(run_all)
    return pdir


def run_experiment(
    key: str,
    model_name: str,
    *,
    mode: str,
    optimization: str,
    save: bool = True,
) -> Dict[str, Any]:
    pdir = project_dir(key)
    results_dir = pdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, cat_cols, num_cols, meta = load_project_xy(key, mode=mode)
    model = get_model(
        model_name=model_name,
        optimization=optimization,
        mode="safe",
        cat_cols=cat_cols,
        num_cols=num_cols,
    )
    t0 = time.perf_counter()
    metrics = evaluate(model, X_train, y_train, X_test, y_test)
    elapsed = time.perf_counter() - t0

    payload = {
        "dataset": key,
        "dataset_name": meta.get("name", key),
        "model_name": model_name,
        "mode": mode,
        "optimization": optimization,
        "feature_count": int(X_train.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "pos_rate": float(y_train.mean()),
        "metrics": _serialize(metrics),
        "total_elapsed_seconds": round(elapsed, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_url": meta.get("url"),
        "dropped_leakage": meta.get("dropped_leakage", []),
    }
    if save:
        out = results_dir / f"{mode}_{optimization}_{model_name}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"[{key}] [{mode}/{optimization}] {model_name:<24} "
        f"| AUC {metrics.get('roc_auc', 0):.4f} | F1 {metrics.get('f1', 0):.4f} | ({elapsed:.1f}s)",
        flush=True,
    )
    return payload


def modes_for(key: str) -> List[str]:
    return ["safe", "unsafe"] if key in LEAKAGE_DROP else ["safe"]


def run_project(
    key: str,
    *,
    models: Optional[List[str]] = None,
    include_transformer: bool = False,
    optimizations: Optional[List[str]] = None,
    save_report: bool = True,
) -> pd.DataFrame:
    scaffold_project(key)
    models = list(models or DEFAULT_MODELS)
    if include_transformer:
        for m in FULL_EXTRA_MODELS:
            if m not in models:
                models.append(m)
    opts = optimizations or ["baseline", "optimized"]
    mode_list = modes_for(key)

    total = len(mode_list) * len(opts) * len(models)
    done = 0
    for mode in mode_list:
        for opt in opts:
            for model in models:
                done += 1
                print(f"[{done}/{total}] {key} / {mode} / {opt} / {model}", flush=True)
                try:
                    run_experiment(key, model, mode=mode, optimization=opt, save=True)
                except Exception as exc:
                    print(f"FAILED {key}/{mode}/{opt}/{model}: {exc}", flush=True)

    df = generate_project_benchmark(key)
    if save_report:
        write_master_report(key, df)
    return df


def load_project_results(key: str) -> pd.DataFrame:
    results_dir = project_dir(key) / "results"
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        d = json.loads(path.read_text())
        m = d.get("metrics", {})
        rows.append(
            {
                "dataset": d.get("dataset", key),
                "mode": d.get("mode", "safe"),
                "optimization": d.get("optimization"),
                "model_name": d.get("model_name"),
                "quadrant": f"{d.get('mode', 'safe')}_{d.get('optimization')}",
                "feature_count": d.get("feature_count"),
                "train_rows": d.get("train_rows"),
                "pos_rate": d.get("pos_rate"),
                "accuracy": m.get("accuracy"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "roc_auc": m.get("roc_auc"),
                "avg_precision": m.get("avg_precision"),
                "elapsed_s": d.get("total_elapsed_seconds"),
            }
        )
    return pd.DataFrame(rows)


def _annotate_bars(ax, bars, vals, horizontal: bool = True):
    for b, val in zip(bars, vals):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if horizontal:
            ax.text(float(val) + 0.002, b.get_y() + b.get_height() / 2, f"{float(val):.3f}",
                    va="center", ha="left", fontsize=7, color="#334155", clip_on=False)
        else:
            ax.text(b.get_x() + b.get_width() / 2, float(val) + 0.002, f"{float(val):.3f}",
                    va="bottom", ha="center", fontsize=7, color="#334155", rotation=90, clip_on=False)


def generate_project_benchmark(key: str) -> pd.DataFrame:
    """Generate HyperAck-like CSVs + plots inside the project folder."""
    df = load_project_results(key)
    out = project_dir(key) / "benchmark_outputs"
    out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print(f"No results for {key}")
        return df

    sns.set_theme(style="whitegrid")
    spec = catalog_entry(key)

    csv_path = out / "complete_benchmark.csv"
    df.sort_values(["mode", "optimization", "roc_auc"], ascending=[True, True, False]).to_csv(csv_path, index=False)

    pivot = df.pivot_table(index="model_name", columns=["mode", "optimization"], values="roc_auc")
    pivot.to_csv(out / "pivot_roc_auc.csv")

    models = sorted(df["model_name"].unique().tolist())
    # Sort by best optimized safe ROC
    safe_opt = df[(df["mode"] == "safe") & (df["optimization"] == "optimized")].set_index("model_name")["roc_auc"]
    models = sorted(models, key=lambda m: float(safe_opt.get(m, 0.0)))

    # --- Plot 1: baseline vs optimized (and unsafe if present) ---
    fig, ax = plt.subplots(figsize=(12, max(6, 0.45 * len(models) + 2)))
    y = np.arange(len(models))
    height = 0.2
    quads = []
    colors = {
        "safe_baseline": "#D97706",
        "safe_optimized": "#16845B",
        "unsafe_baseline": "#EF4444",
        "unsafe_optimized": "#7F1D1D",
    }
    for mode in sorted(df["mode"].unique()):
        for opt in ["baseline", "optimized"]:
            sub = df[(df["mode"] == mode) & (df["optimization"] == opt)].set_index("model_name")["roc_auc"]
            if sub.empty:
                continue
            quads.append((f"{mode}_{opt}", [float(sub.get(m, np.nan)) for m in models], colors.get(f"{mode}_{opt}", "#2563EB")))

    nq = max(len(quads), 1)
    for i, (label, vals, color) in enumerate(quads):
        offset = (i - (nq - 1) / 2) * height
        bars = ax.barh(y + offset, vals, height, label=label.replace("_", " ").title(), color=color, alpha=0.92)
        _annotate_bars(ax, bars, vals)

    ax.set_yticks(y)
    ax.set_yticklabels([m.replace("_", " ").title() for m in models], fontsize=10)
    ax.set_xlabel("ROC-AUC", fontweight="bold")
    finite = df["roc_auc"].dropna()
    lo = max(0.5, float(finite.min()) - 0.05) if len(finite) else 0.5
    ax.set_xlim(lo, min(1.02, float(finite.max()) + 0.06) if len(finite) else 1.02)
    ax.set_title(f"{spec.name}: Baseline vs Optimized ROC-AUC", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out / "roc_baseline_vs_optimized.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: 4-metric profile (safe optimized) ---
    so = df[(df["mode"] == "safe") & (df["optimization"] == "optimized")].set_index("model_name")
    if not so.empty:
        metric_cols = ["accuracy", "precision", "recall", "f1"]
        metric_colors = {"accuracy": "#2563EB", "precision": "#0D9488", "recall": "#D97706", "f1": "#16845B"}
        sorted_m = sorted(so.index.tolist(), key=lambda m: float(so.loc[m, "f1"]))
        pos = np.arange(len(sorted_m))
        w = 0.2
        fig, ax = plt.subplots(figsize=(13, max(6, 0.4 * len(sorted_m) + 2)))
        for i, mc in enumerate(metric_cols):
            vals = [float(so.loc[m, mc]) for m in sorted_m]
            bars = ax.barh(pos + (i - 1.5) * w, vals, w, label=mc.capitalize(), color=metric_colors[mc], alpha=0.92)
            _annotate_bars(ax, bars, vals)
        ax.set_yticks(pos)
        ax.set_yticklabels([m.replace("_", " ").title() for m in sorted_m])
        ax.set_xlabel("Score")
        ax.set_xlim(0.0, 1.05)
        ax.set_title(f"{spec.name}: Safe Optimized — Acc / Prec / Rec / F1", fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(out / "safe_optimized_4metrics_profile.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # vertical twin
        fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(sorted_m) + 2), 6.5))
        for i, mc in enumerate(metric_cols):
            vals = [float(so.loc[m, mc]) for m in sorted_m]
            bars = ax.bar(pos + (i - 1.5) * w, vals, w, label=mc.capitalize(), color=metric_colors[mc], alpha=0.92)
            _annotate_bars(ax, bars, vals, horizontal=False)
        ax.set_xticks(pos)
        ax.set_xticklabels([m.replace("_", " ").title() for m in sorted_m], rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.08)
        ax.set_title(f"{spec.name}: Safe Optimized — Acc / Prec / Rec / F1 (Vertical)", fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(out / "safe_optimized_4metrics_profile_vertical.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- Plot 3: heatmap ---
    heat = so[metric_cols + ["roc_auc"]] if not so.empty else pd.DataFrame()
    if not heat.empty:
        heat = heat.copy()
        heat.index = [i.replace("_", " ").title() for i in heat.index]
        fig, ax = plt.subplots(figsize=(9, max(5, 0.4 * len(heat) + 2)))
        sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.5, vmax=1.0, ax=ax)
        ax.set_title(f"{spec.name}: Safe Optimized Scorecard", fontweight="bold")
        plt.tight_layout()
        fig.savefig(out / "metrics_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- Plot 4: Pareto ROC vs Recall ---
    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {"safe_baseline": "o", "safe_optimized": "s", "unsafe_baseline": "^", "unsafe_optimized": "D"}
    for quad, group in df.groupby("quadrant"):
        ax.scatter(
            group["roc_auc"], group["recall"],
            label=quad.replace("_", " ").title(),
            marker=markers.get(quad, "o"),
            color=colors.get(quad, "#2563EB"),
            s=80, alpha=0.9, edgecolors="black", linewidth=0.7,
        )
        for _, row in group.iterrows():
            if row["model_name"] in {"lightgbm", "catboost", "stacking_ensemble", "soft_vote_blend", "extra_trees"}:
                ax.annotate(
                    f"{row['model_name'].replace('_', ' ')}\nAUC={row['roc_auc']:.3f}",
                    (row["roc_auc"], row["recall"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=7.5,
                )
    ax.set_xlabel("ROC-AUC", fontweight="bold")
    ax.set_ylabel("Recall", fontweight="bold")
    ax.set_title(f"{spec.name}: ROC-AUC vs Recall Frontier", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    fig.savefig(out / "pareto_roc_vs_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 5: optimization lift ---
    lifts = []
    for m in models:
        b = df[(df["mode"] == "safe") & (df["optimization"] == "baseline") & (df["model_name"] == m)]["roc_auc"]
        o = df[(df["mode"] == "safe") & (df["optimization"] == "optimized") & (df["model_name"] == m)]["roc_auc"]
        if len(b) and len(o):
            lifts.append((m, float(o.iloc[0] - b.iloc[0])))
    if lifts:
        lifts = sorted(lifts, key=lambda x: x[1])
        fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(lifts) + 2)))
        vals = [v for _, v in lifts]
        bars = ax.barh([m.replace("_", " ").title() for m, _ in lifts], vals, color="#2563EB", alpha=0.9)
        _annotate_bars(ax, bars, vals)
        ax.axvline(0, color="#64748B", lw=1)
        ax.set_xlabel("Δ ROC-AUC (Optimized − Baseline)", fontweight="bold")
        ax.set_title(f"{spec.name}: Optimization Lift", fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(out / "optimization_lift.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[{key}] benchmark artifacts -> {out}")
    return df


def write_master_report(key: str, df: pd.DataFrame) -> Path:
    spec = catalog_entry(key)
    pdir = project_dir(key)
    if df.empty:
        path = pdir / "MASTER_BENCHMARK_REPORT.md"
        path.write_text(f"# {spec.name}\n\nNo results yet.\n")
        return path

    best = df.sort_values("roc_auc", ascending=False).iloc[0]
    safe_opt = df[(df["mode"] == "safe") & (df["optimization"] == "optimized")].sort_values("roc_auc", ascending=False)
    champion = safe_opt.iloc[0] if not safe_opt.empty else best

    table_rows = []
    show = df.sort_values(["mode", "optimization", "roc_auc"], ascending=[True, True, False])
    for _, r in show.iterrows():
        table_rows.append(
            f"| {r['mode']} | {r['optimization']} | {r['model_name']} | {r['roc_auc']:.4f} | "
            f"{r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |"
        )

    leak_note = ""
    if key in LEAKAGE_DROP:
        leak_note = (
            f"\n## Leakage protocol\n"
            f"Safe mode drops `{LEAKAGE_DROP[key]}` (post-outcome). "
            f"Unsafe mode keeps all features — analogous to HyperAck final fares.\n"
        )

    md = f"""# {spec.name} — Master Benchmark Report

HyperAck-style project folder: `external_projects/{key}_exp/`

- **Source:** [{spec.url}]({spec.url})
- **Description:** {spec.description}
- **Split:** stratified 80/20, seed {RANDOM_STATE}
- **Champion (safe optimized):** `{champion['model_name']}` — ROC-AUC **{champion['roc_auc']:.4f}**, F1 **{champion['f1']:.4f}**
- **Overall best:** `{best['model_name']}` ({best['mode']}/{best['optimization']}) — ROC-AUC **{best['roc_auc']:.4f}**
{leak_note}
## Full results

| Mode | Opt | Model | ROC-AUC | Acc | Prec | Rec | F1 |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## Figures

![ROC baseline vs optimized](benchmark_outputs/roc_baseline_vs_optimized.png)

![Safe optimized 4 metrics](benchmark_outputs/safe_optimized_4metrics_profile.png)

![Metrics heatmap](benchmark_outputs/metrics_heatmap.png)

![Pareto ROC vs Recall](benchmark_outputs/pareto_roc_vs_recall.png)

![Optimization lift](benchmark_outputs/optimization_lift.png)

## Reproduce

```bash
.venv/bin/python external_projects/{key}_exp/run_all.py
```
"""
    path = pdir / "MASTER_BENCHMARK_REPORT.md"
    path.write_text(md)
    print(f"[{key}] wrote {path}")
    return path
