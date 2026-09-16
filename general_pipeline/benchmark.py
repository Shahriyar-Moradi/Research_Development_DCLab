"""Analysis, reporting, and visualization generator for the General Pipeline Benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUTS_DIR = ROOT / "benchmark_outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_results() -> pd.DataFrame:
    """Load all JSON results from general_pipeline/results/."""
    records = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            m = d.get("metrics", {})
            records.append({
                "mode": d.get("mode"),
                "optimization": d.get("optimization"),
                "model_name": d.get("model_name"),
                "quadrant": f"{d.get('mode')}_{d.get('optimization')}",
                "roc_auc": m.get("roc_auc"),
                "recall": m.get("recall"),
                "precision": m.get("precision"),
                "f1": m.get("f1"),
                "accuracy": m.get("accuracy"),
                "avg_precision": m.get("avg_precision"),
                "fit_seconds": m.get("fit_seconds"),
                "total_seconds": d.get("total_elapsed_seconds"),
                "feature_count": d.get("feature_count"),
            })
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return pd.DataFrame(records)


def generate_benchmark_artifacts():
    """Produce comparison CSVs and high-resolution diagnostic plots."""
    df = load_all_results()
    if df.empty:
        print("No results found in", RESULTS_DIR)
        return

    # Save complete raw benchmark table
    csv_path = OUTPUTS_DIR / "complete_4way_benchmark.csv"
    df.sort_values(["mode", "optimization", "roc_auc"], ascending=[True, True, False]).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Pivot table: Model x Quadrant for ROC-AUC and Recall
    pivot_roc = df.pivot_table(index="model_name", columns=["mode", "optimization"], values="roc_auc")
    pivot_rec = df.pivot_table(index="model_name", columns=["mode", "optimization"], values="recall")

    pivot_csv = OUTPUTS_DIR / "pivot_roc_auc_by_quadrant.csv"
    pivot_roc.to_csv(pivot_csv)
    print(f"Saved: {pivot_csv}")

    # -------------------------------------------------------------
    # Plot 1: 4-Way Quadrant Comparison (Grouped Bar Chart)
    # -------------------------------------------------------------
    # Focus on key models present across quadrants
    key_models = [
        "logistic_regression",
        "linear_svc",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "xgboost",
        "catboost",
        "lightgbm",
        "tabular_transformer",
        "stacking_ensemble",
        "soft_vote_blend",
    ]
    present_models = [m for m in key_models if m in df["model_name"].unique()]

    # Sort models by safe_optimized performance
    safe_opt_series = df[(df["mode"] == "safe") & (df["optimization"] == "optimized")].set_index("model_name")["roc_auc"]
    sorted_models = sorted(present_models, key=lambda m: safe_opt_series.get(m, 0.0))

    fig, ax = plt.subplots(figsize=(14, 9))
    y = np.arange(len(sorted_models))
    height = 0.20

    # Colors
    c_safe_base = "#D97706"     # Amber
    c_safe_opt = "#16845B"      # Emerald Green
    c_unsafe_base = "#DC2626"   # Light Red
    c_unsafe_opt = "#7F1D1D"    # Deep Crimson

    def get_vals(mode, opt):
        sub = df[(df["mode"] == mode) & (df["optimization"] == opt)].set_index("model_name")["roc_auc"]
        return [sub.get(m, np.nan) for m in sorted_models]

    vals_safe_base = get_vals("safe", "baseline")
    vals_safe_opt = get_vals("safe", "optimized")
    vals_unsafe_base = get_vals("unsafe", "baseline")
    vals_unsafe_opt = get_vals("unsafe", "optimized")

    ax.barh(y - 1.5 * height, vals_safe_base, height, label="Safe Baseline", color=c_safe_base, alpha=0.9)
    ax.barh(y - 0.5 * height, vals_safe_opt, height, label="Safe Optimized", color=c_safe_opt)
    ax.barh(y + 0.5 * height, vals_unsafe_base, height, label="Unsafe Baseline", color=c_unsafe_base, alpha=0.8)
    ax.barh(y + 1.5 * height, vals_unsafe_opt, height, label="Unsafe Optimized", color=c_unsafe_opt)

    ax.set_yticks(y)
    ax.set_yticklabels([m.replace("_", " ").title() for m in sorted_models], fontsize=11, fontweight="medium")
    ax.set_xlabel("ROC-AUC Score", fontsize=12, fontweight="bold")
    ax.set_xlim(0.85, 0.995)
    ax.set_title("Master 4-Way Tabular Benchmark: Safe vs Unsafe (Baseline vs Optimized)", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)

    # Highlight Tabular Transformer
    if "tabular_transformer" in sorted_models:
        idx = sorted_models.index("tabular_transformer")
        ax.get_yticklabels()[idx].set_color("#2563EB")
        ax.get_yticklabels()[idx].set_fontweight("bold")

    plt.tight_layout()
    p1 = OUTPUTS_DIR / "master_4way_quadrant_roc.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved: {p1}")

    # -------------------------------------------------------------
    # Plot 2: Deep Dive: Tabular Transformer vs GBDTs vs Forests
    # -------------------------------------------------------------
    comp_models = ["tabular_transformer", "lightgbm", "xgboost", "catboost", "extra_trees", "random_forest"]
    comp_df = df[df["model_name"].isin(comp_models)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Left: Safe Regime
    safe_data = comp_df[comp_df["mode"] == "safe"].pivot(index="model_name", columns="optimization", values="roc_auc").reindex(comp_models)
    labels = [m.replace("_", " ").title() for m in comp_models]
    y_comp = np.arange(len(comp_models))

    axes[0].barh(y_comp - 0.18, safe_data.get("baseline", 0), 0.35, label="Baseline", color="#9A6500")
    axes[0].barh(y_comp + 0.18, safe_data.get("optimized", 0), 0.35, label="Optimized", color="#16845B")
    axes[0].set_yticks(y_comp)
    axes[0].set_yticklabels(labels, fontsize=11, fontweight="semibold")
    axes[0].set_xlabel("ROC-AUC", fontsize=11, fontweight="bold")
    axes[0].set_xlim(0.88, 0.96)
    axes[0].set_title("Safe Regime (Honest Pre-Outcome Features)", fontsize=12, fontweight="bold")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right")

    # Right: Unsafe Regime
    unsafe_data = comp_df[comp_df["mode"] == "unsafe"].pivot(index="model_name", columns="optimization", values="roc_auc").reindex(comp_models)
    axes[1].barh(y_comp - 0.18, unsafe_data.get("baseline", 0), 0.35, label="Baseline", color="#EF4444")
    axes[1].barh(y_comp + 0.18, unsafe_data.get("optimized", 0), 0.35, label="Optimized", color="#7F1D1D")
    axes[1].set_xlabel("ROC-AUC", fontsize=11, fontweight="bold")
    axes[1].set_xlim(0.88, 0.995)
    axes[1].set_title("Unsafe Regime (Leaked Final Fares Included)", fontsize=12, fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right")

    fig.suptitle("Tabular Transformer (FT-Transformer) vs GBDTs & Random Forests", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p2 = OUTPUTS_DIR / "tabular_transformer_vs_gbdts.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------
    # Plot 3: Pareto Frontier (ROC-AUC vs Recall)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8))
    markers = {"safe_baseline": "o", "safe_optimized": "s", "unsafe_baseline": "^", "unsafe_optimized": "D"}
    colors = {"safe_baseline": "#D97706", "safe_optimized": "#16845B", "unsafe_baseline": "#EF4444", "unsafe_optimized": "#7F1D1D"}

    for quad, group in df.groupby("quadrant"):
        ax.scatter(
            group["roc_auc"],
            group["recall"],
            label=quad.replace("_", " ").title(),
            marker=markers.get(quad, "o"),
            color=colors.get(quad, "#2563EB"),
            s=80,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.8,
        )
        # Label select star models
        for _, row in group.iterrows():
            if row["model_name"] in ["tabular_transformer", "soft_vote_blend", "stacking_ensemble", "lightgbm"]:
                ax.annotate(
                    f"{row['model_name'].replace('_', ' ')}",
                    (row["roc_auc"], row["recall"]),
                    xytext=(5, 3),
                    textcoords="offset points",
                    fontsize=8.5,
                    alpha=0.85,
                )

    ax.set_xlabel("ROC-AUC (Discrimination Quality)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall (True Positive Rate)", fontsize=12, fontweight="bold")
    ax.set_title("Classification Frontier: ROC-AUC vs Recall across all Quadrants", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    p3 = OUTPUTS_DIR / "pareto_roc_vs_recall.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"Saved: {p3}")

    # -------------------------------------------------------------
    # Plot 4: Optimization Delta Lift (Optimized - Baseline)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    safe_deltas = []
    unsafe_deltas = []
    models_with_deltas = []

    for m in present_models:
        sb = df[(df["mode"] == "safe") & (df["optimization"] == "baseline") & (df["model_name"] == m)]["roc_auc"].values
        so = df[(df["mode"] == "safe") & (df["optimization"] == "optimized") & (df["model_name"] == m)]["roc_auc"].values
        ub = df[(df["mode"] == "unsafe") & (df["optimization"] == "baseline") & (df["model_name"] == m)]["roc_auc"].values
        uo = df[(df["mode"] == "unsafe") & (df["optimization"] == "optimized") & (df["model_name"] == m)]["roc_auc"].values

        if len(sb) and len(so) and len(ub) and len(uo):
            models_with_deltas.append(m)
            safe_deltas.append(so[0] - sb[0])
            unsafe_deltas.append(uo[0] - ub[0])

    if models_with_deltas:
        y_d = np.arange(len(models_with_deltas))
        ax.barh(y_d - 0.18, safe_deltas, 0.35, label="Safe Optimization Lift", color="#16845B")
        ax.barh(y_d + 0.18, unsafe_deltas, 0.35, label="Unsafe Optimization Lift", color="#B42318")
        ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
        ax.set_yticks(y_d)
        ax.set_yticklabels([m.replace("_", " ").title() for m in models_with_deltas], fontsize=10)
        ax.set_xlabel("$\Delta$ ROC-AUC (Optimized - Baseline)", fontsize=11, fontweight="bold")
        ax.set_title("Optimization Return on Investment by Algorithm Family", fontsize=13, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")

        plt.tight_layout()
        p4 = OUTPUTS_DIR / "optimization_lift_by_model.png"
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        print(f"Saved: {p4}")

    print("All benchmark visualizations successfully generated.")


if __name__ == "__main__":
    generate_benchmark_artifacts()
