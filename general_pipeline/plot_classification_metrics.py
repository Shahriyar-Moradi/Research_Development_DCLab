"""Generate Accuracy / Precision / Recall / F1 plots in both horizontal and vertical layouts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "benchmark_outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUTS_DIR / "complete_4way_benchmark.csv"

COLORS = {
    "safe_baseline": "#D97706",
    "safe_optimized": "#16845B",
    "unsafe_baseline": "#EF4444",
    "unsafe_optimized": "#7F1D1D",
}

METRIC_COLORS = {
    "accuracy": "#2563EB",
    "precision": "#0D9488",
    "recall": "#D97706",
    "f1": "#16845B",
}

MODEL_ORDER = [
    "logistic_regression",
    "linear_svc",
    "tabular_transformer",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "catboost",
    "soft_vote_blend",
    "stacking_ensemble",
    "lightgbm",
]

QUADRANT_SPECS = [
    ("safe", "baseline", "Safe Baseline", COLORS["safe_baseline"]),
    ("safe", "optimized", "Safe Optimized", COLORS["safe_optimized"]),
    ("unsafe", "baseline", "Unsafe Baseline", COLORS["unsafe_baseline"]),
    ("unsafe", "optimized", "Unsafe Optimized", COLORS["unsafe_optimized"]),
]

METRIC_SPECS = [
    ("accuracy", "Accuracy (Overall Correctness)", (0.75, 0.95)),
    ("precision", "Precision (Positive Predictive Value)", (0.65, 0.98)),
    ("recall", "Recall (True Positive Rate)", (0.65, 0.95)),
    ("f1", "F1-Score (Harmonic Mean)", (0.75, 0.95)),
]


def clean_name(m: str) -> str:
    mapping = {
        "logistic_regression": "Logistic Regression",
        "linear_svc": "Linear SVC",
        "tabular_transformer": "Tabular Transformer",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "hist_gradient_boosting": "HistGradientBoosting",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
        "soft_vote_blend": "Soft-Vote Blend",
        "stacking_ensemble": "Stacking Ensemble",
        "lightgbm": "LightGBM",
    }
    return mapping.get(m, m.replace("_", " ").title())


def highlight_transformer(ax, models: list[str], labels) -> None:
    if "tabular_transformer" not in models:
        return
    idx = models.index("tabular_transformer")
    if idx < len(labels):
        labels[idx].set_color("#2563EB")
        labels[idx].set_fontweight("bold")


def annotate_barh_values(
    ax,
    bars,
    vals: list[float],
    *,
    fontsize: float = 6.5,
    decimals: int = 3,
    x_pad: float = 0.0025,
) -> None:
    for b, val in zip(bars, vals):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        ax.text(
            float(val) + x_pad,
            b.get_y() + b.get_height() / 2,
            f"{float(val):.{decimals}f}",
            va="center",
            ha="left",
            fontsize=fontsize,
            color="#334155",
            clip_on=False,
        )


def annotate_bar_values(
    ax,
    bars,
    vals: list[float],
    *,
    fontsize: float = 6.5,
    decimals: int = 3,
    y_pad: float = 0.0025,
    rotate: bool = True,
) -> None:
    for b, val in zip(bars, vals):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            float(val) + y_pad,
            f"{float(val):.{decimals}f}",
            va="bottom",
            ha="center",
            fontsize=fontsize,
            color="#334155",
            rotation=90 if rotate else 0,
            clip_on=False,
        )


def axis_lim_with_label_room(lim: tuple[float, float], *, horizontal: bool, extra: float = 0.045) -> tuple[float, float]:
    """Extend the value-axis upper bound so bar value labels are not clipped."""
    return (lim[0], lim[1] + extra)


def plot_2x2_grid(df: pd.DataFrame, models: list[str], orientation: str) -> Path:
    """2x2 Accuracy/Precision/Recall/F1 across 4 quadrants — horizontal or vertical bars."""
    is_h = orientation == "horizontal"
    fig, axes = plt.subplots(2, 2, figsize=(18, 14) if is_h else (18, 12))
    n = len(models)
    pos = np.arange(n)
    width = 0.20
    labels = [clean_name(m) for m in models]

    for ax_idx, (metric_col, metric_title, lim) in enumerate(METRIC_SPECS):
        ax = axes[ax_idx // 2, ax_idx % 2]
        for q_idx, (mode, opt, q_label, color) in enumerate(QUADRANT_SPECS):
            sub = df[(df["mode"] == mode) & (df["optimization"] == opt)].set_index("model_name")
            vals = [float(sub.loc[m, metric_col]) if m in sub.index else np.nan for m in models]
            offset = (q_idx - 1.5) * width
            if is_h:
                bars = ax.barh(pos + offset, vals, width, label=q_label, color=color, alpha=0.92)
                annotate_barh_values(ax, bars, vals, fontsize=5.8, x_pad=0.002)
            else:
                bars = ax.bar(pos + offset, vals, width, label=q_label, color=color, alpha=0.92)
                annotate_bar_values(ax, bars, vals, fontsize=5.8, y_pad=0.0015, rotate=True)

        if is_h:
            ax.set_yticks(pos)
            ax.set_yticklabels(labels, fontsize=10.5, fontweight="medium")
            ax.set_xlabel(metric_col.upper(), fontsize=11, fontweight="bold")
            ax.set_xlim(axis_lim_with_label_room(lim, horizontal=True))
            ax.grid(axis="x", linestyle="--", alpha=0.6)
            if ax_idx % 2 == 0:
                highlight_transformer(ax, models, ax.get_yticklabels())
        else:
            ax.set_xticks(pos)
            ax.set_xticklabels(labels, fontsize=9, fontweight="medium", rotation=35, ha="right")
            ax.set_ylabel(metric_col.upper(), fontsize=11, fontweight="bold")
            ax.set_ylim(axis_lim_with_label_room(lim, horizontal=False, extra=0.035))
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            highlight_transformer(ax, models, ax.get_xticklabels())

        ax.set_title(metric_title, fontsize=13, fontweight="bold", pad=10)
        if ax_idx == 0:
            ax.legend(loc="lower right" if is_h else "upper left", fontsize=9.5, framealpha=0.95)
        ax.margins(x=0.02 if not is_h else 0.0, y=0.02 if is_h else 0.0)

    fig.suptitle(
        f"Master Classification Metrics ({orientation.title()} Bars)\n"
        "Accuracy, Precision, Recall & F1 across All Models and 4 Quadrants",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTPUTS_DIR / f"classification_metrics_2x2_grid_{orientation}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_safe_profile(df: pd.DataFrame, models: list[str], orientation: str) -> Path:
    """Safe-optimized multi-metric profile — horizontal or vertical."""
    is_h = orientation == "horizontal"
    safe_opt = df[(df["mode"] == "safe") & (df["optimization"] == "optimized")].set_index("model_name")
    sorted_models = sorted(
        models,
        key=lambda m: float(safe_opt.loc[m, "f1"]) if m in safe_opt.index else 0.0,
        reverse=not is_h,
    )
    labels = [clean_name(m) for m in sorted_models]
    pos = np.arange(len(sorted_models))
    w = 0.20

    fig, ax = plt.subplots(figsize=(15, 8.5) if is_h else (16, 7.5))
    for idx, (m_key, color) in enumerate(METRIC_COLORS.items()):
        vals = [float(safe_opt.loc[m, m_key]) for m in sorted_models]
        offset = (idx - 1.5) * w
        if is_h:
            bars = ax.barh(pos + offset, vals, w, label=m_key.capitalize(), color=color, alpha=0.92)
            annotate_barh_values(ax, bars, vals, fontsize=7.0, x_pad=0.003)
        else:
            bars = ax.bar(pos + offset, vals, w, label=m_key.capitalize(), color=color, alpha=0.92)
            annotate_bar_values(ax, bars, vals, fontsize=6.5, y_pad=0.003, rotate=True)

    if is_h:
        ax.set_yticks(pos)
        ax.set_yticklabels(labels, fontsize=11, fontweight="medium")
        ax.set_xlabel("Metric Score (0.0 to 1.0)", fontsize=12, fontweight="bold")
        ax.set_xlim(0.68, 1.02)
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        highlight_transformer(ax, sorted_models, ax.get_yticklabels())
    else:
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, fontsize=10, fontweight="medium", rotation=30, ha="right")
        ax.set_ylabel("Metric Score (0.0 to 1.0)", fontsize=12, fontweight="bold")
        ax.set_ylim(0.68, 1.02)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        highlight_transformer(ax, sorted_models, ax.get_xticklabels())

    ax.set_title(
        f"Safe Optimized Regime ({orientation.title()} Bars): Accuracy, Precision, Recall, F1\n"
        "Zero Leakage Honest Production Models",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower right" if is_h else "upper left", fontsize=11, framealpha=0.95)
    plt.tight_layout()
    out = OUTPUTS_DIR / f"safe_optimized_4metrics_profile_{orientation}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_single_metric_panels(df: pd.DataFrame, models: list[str], orientation: str) -> list[Path]:
    """One plot per metric comparing all 4 quadrants."""
    paths = []
    is_h = orientation == "horizontal"
    labels = [clean_name(m) for m in models]
    pos = np.arange(len(models))
    width = 0.20

    for metric_col, metric_title, lim in METRIC_SPECS:
        fig, ax = plt.subplots(figsize=(12, 8) if is_h else (14, 6.5))
        for q_idx, (mode, opt, q_label, color) in enumerate(QUADRANT_SPECS):
            sub = df[(df["mode"] == mode) & (df["optimization"] == opt)].set_index("model_name")
            vals = [float(sub.loc[m, metric_col]) if m in sub.index else np.nan for m in models]
            offset = (q_idx - 1.5) * width
            if is_h:
                bars = ax.barh(pos + offset, vals, width, label=q_label, color=color, alpha=0.92)
                annotate_barh_values(ax, bars, vals, fontsize=6.5, x_pad=0.002)
            else:
                bars = ax.bar(pos + offset, vals, width, label=q_label, color=color, alpha=0.92)
                annotate_bar_values(ax, bars, vals, fontsize=6.5, y_pad=0.0015, rotate=True)

        if is_h:
            ax.set_yticks(pos)
            ax.set_yticklabels(labels, fontsize=11, fontweight="medium")
            ax.set_xlabel(metric_col.upper(), fontsize=12, fontweight="bold")
            ax.set_xlim(axis_lim_with_label_room(lim, horizontal=True))
            ax.grid(axis="x", linestyle="--", alpha=0.6)
            highlight_transformer(ax, models, ax.get_yticklabels())
        else:
            ax.set_xticks(pos)
            ax.set_xticklabels(labels, fontsize=10, fontweight="medium", rotation=30, ha="right")
            ax.set_ylabel(metric_col.upper(), fontsize=12, fontweight="bold")
            ax.set_ylim(axis_lim_with_label_room(lim, horizontal=False, extra=0.035))
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            highlight_transformer(ax, models, ax.get_xticklabels())

        ax.set_title(
            f"{metric_title} — {orientation.title()} Bars\nAll Models × 4 Quadrants",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax.legend(loc="lower right" if is_h else "upper left", fontsize=10, framealpha=0.95)
        plt.tight_layout()
        out = OUTPUTS_DIR / f"{metric_col}_by_quadrant_{orientation}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def plot_pr_frontier(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12, 8.5))
    p_grid = np.linspace(0.60, 0.99, 200)
    r_grid = np.linspace(0.60, 0.99, 200)
    P, R = np.meshgrid(p_grid, r_grid)
    F1 = 2 * (P * R) / (P + R)
    contours = ax.contour(
        P,
        R,
        F1,
        levels=[0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94],
        colors="#CBD5E1",
        linewidths=1.0,
        linestyles="dashed",
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="F1=%.2f", colors="#94A3B8")

    markers = {
        "safe_baseline": "o",
        "safe_optimized": "s",
        "unsafe_baseline": "^",
        "unsafe_optimized": "D",
    }
    for quad_name, (mode, opt, q_label, color) in zip(
        ["safe_baseline", "safe_optimized", "unsafe_baseline", "unsafe_optimized"],
        QUADRANT_SPECS,
    ):
        sub = df[(df["mode"] == mode) & (df["optimization"] == opt)]
        ax.scatter(
            sub["precision"],
            sub["recall"],
            label=q_label,
            color=color,
            marker=markers[quad_name],
            s=95,
            edgecolors="black",
            linewidth=0.8,
            alpha=0.9,
            zorder=5,
        )
        for _, r in sub.iterrows():
            if r["model_name"] in {
                "lightgbm",
                "stacking_ensemble",
                "tabular_transformer",
                "catboost",
            }:
                p, rec = float(r["precision"]), float(r["recall"])
                ax.annotate(
                    f"{clean_name(r['model_name'])}\nP={p:.3f} R={rec:.3f}",
                    (p, rec),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=7.5,
                    fontweight="bold" if r["model_name"] == "tabular_transformer" else "normal",
                    color="#1E293B",
                    alpha=0.9,
                    zorder=6,
                )

    ax.set_xlabel("Precision (Accuracy of Positive Predictions)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall (Coverage of Actual Positives)", fontsize=12, fontweight="bold")
    ax.set_xlim(0.68, 0.98)
    ax.set_ylim(0.68, 0.95)
    ax.set_title(
        "Precision vs Recall Frontier with F1 Iso-Curves",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=11, framealpha=0.95)
    plt.tight_layout()
    out = OUTPUTS_DIR / "precision_vs_recall_f1_frontier.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_heatmap(df: pd.DataFrame) -> Path:
    models = [m for m in MODEL_ORDER if m in df["model_name"].unique()]
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    safe_heat = (
        df[(df["mode"] == "safe") & (df["optimization"] == "optimized")]
        .set_index("model_name")[metric_cols]
        .reindex(models)
    )
    unsafe_heat = (
        df[(df["mode"] == "unsafe") & (df["optimization"] == "optimized")]
        .set_index("model_name")[metric_cols]
        .reindex(models)
    )
    safe_heat.index = [clean_name(m) for m in safe_heat.index]
    unsafe_heat.index = [clean_name(m) for m in unsafe_heat.index]

    sns.heatmap(
        safe_heat,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.70,
        vmax=0.98,
        cbar=False,
        ax=axes[0],
        linewidths=0.5,
        linecolor="#E2E8F0",
    )
    axes[0].set_title("Safe Optimized Regime (Zero Leakage)", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xticklabels(metric_names, fontsize=10.5, fontweight="semibold")

    sns.heatmap(
        unsafe_heat,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.70,
        vmax=0.98,
        cbar=True,
        ax=axes[1],
        linewidths=0.5,
        linecolor="#E2E8F0",
        cbar_kws={"label": "Metric Score"},
    )
    axes[1].set_title("Unsafe Optimized Regime (With Final Fares)", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xticklabels(metric_names, fontsize=10.5, fontweight="semibold")

    fig.suptitle(
        "Comprehensive Model Scorecard Heatmap: Accuracy, Precision, Recall, F1 & ROC-AUC",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    out = OUTPUTS_DIR / "classification_metrics_heatmap.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} records from {CSV_PATH}")

    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.sans-serif"] = [
        "SF Pro Text",
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "sans-serif",
    ]
    plt.rcParams["axes.edgecolor"] = "#E2E8F0"
    plt.rcParams["axes.linewidth"] = 0.8

    models = [m for m in MODEL_ORDER if m in df["model_name"].unique()]
    outputs: list[Path] = []

    for orientation in ("horizontal", "vertical"):
        print(f"\n=== Generating {orientation} plots ===")
        outputs.append(plot_2x2_grid(df, models, orientation))
        print(f"Saved: {outputs[-1]}")
        outputs.append(plot_safe_profile(df, models, orientation))
        print(f"Saved: {outputs[-1]}")
        for p in plot_single_metric_panels(df, models, orientation):
            outputs.append(p)
            print(f"Saved: {p}")

    # Keep canonical filenames pointing to horizontal versions for report embeds
    for src_suffix, dest in [
        ("classification_metrics_2x2_grid_horizontal.png", "classification_metrics_2x2_grid.png"),
        ("safe_optimized_4metrics_profile_horizontal.png", "safe_optimized_4metrics_profile.png"),
    ]:
        src = OUTPUTS_DIR / src_suffix
        dest_path = OUTPUTS_DIR / dest
        dest_path.write_bytes(src.read_bytes())
        print(f"Synced canonical: {dest_path}")

    outputs.append(plot_pr_frontier(df))
    print(f"Saved: {outputs[-1]}")
    outputs.append(plot_heatmap(df))
    print(f"Saved: {outputs[-1]}")

    print(f"\nGenerated {len(outputs)} figure files (horizontal + vertical).")


if __name__ == "__main__":
    main()
