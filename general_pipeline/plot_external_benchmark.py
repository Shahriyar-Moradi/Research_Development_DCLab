"""Plots for the multi-dataset external classification benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "benchmark_outputs" / "external"
CSV_PATH = OUTPUTS_DIR / "external_multidataset_benchmark.csv"


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH}. Run run_multi_dataset.py first.")

    df = pd.read_csv(CSV_PATH)
    sns.set_theme(style="whitegrid")

    # Best ROC per dataset (optimized preferred if available)
    best = (
        df.sort_values("roc_auc", ascending=False)
        .groupby("dataset", as_index=False)
        .first()
        .sort_values("roc_auc")
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(best["dataset"], best["roc_auc"], color="#16845B", alpha=0.9)
    for b, row in zip(bars, best.itertuples()):
        ax.text(
            row.roc_auc + 0.005,
            b.get_y() + b.get_height() / 2,
            f"{row.roc_auc:.3f} ({row.model_name})",
            va="center",
            fontsize=8,
            color="#334155",
        )
    ax.set_xlim(0.5, 1.05)
    ax.set_xlabel("Best ROC-AUC")
    ax.set_title("External Datasets — Best Model ROC-AUC")
    plt.tight_layout()
    out1 = OUTPUTS_DIR / "external_best_roc_by_dataset.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # Heatmap dataset × model (optimized)
    opt = df[df["optimization"] == "optimized"].copy()
    if opt.empty:
        opt = df.copy()
    pivot = opt.pivot_table(index="dataset", columns="model_name", values="roc_auc", aggfunc="max")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.6, vmax=1.0, ax=ax)
    ax.set_title("External Datasets ROC-AUC Heatmap (Optimized)")
    plt.tight_layout()
    out2 = OUTPUTS_DIR / "external_roc_heatmap.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    # Baseline vs optimized lift for LightGBM
    if {"baseline", "optimized"}.issubset(set(df["optimization"])):
        lgb = df[df["model_name"] == "lightgbm"].pivot_table(
            index="dataset", columns="optimization", values="roc_auc"
        )
        if {"baseline", "optimized"}.issubset(lgb.columns):
            lgb = lgb.assign(lift=lgb["optimized"] - lgb["baseline"]).sort_values("lift")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(lgb.index, lgb["lift"], color="#2563EB", alpha=0.9)
            ax.axvline(0, color="#94A3B8", lw=1)
            ax.set_xlabel("ROC-AUC Lift (Optimized − Baseline)")
            ax.set_title("LightGBM Optimization Lift Across External Datasets")
            plt.tight_layout()
            out3 = OUTPUTS_DIR / "external_lightgbm_opt_lift.png"
            fig.savefig(out3, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out3}")


if __name__ == "__main__":
    main()
