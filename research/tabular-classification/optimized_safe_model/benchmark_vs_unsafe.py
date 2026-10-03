"""Visual benchmark: optimized-safe vs unsafe (same exp_id where available)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OPT = ROOT / "results"
UNSAFE = ROOT.parent / "hyperack_exp" / "results"
SAFE = ROOT.parent / "safe_leakage_exp" / "results"

# Friendly labels for matched ladder (01-15)
LABELS = {
    "01": "01 baseline",
    "02": "02 lgbm baseline",
    "03": "03 pricing FE",
    "04": "04 time FE",
    "05": "05 geo FE",
    "06": "06 full FE",
    "07": "07 feature select",
    "08": "08 xgboost tuned",
    "09": "09 lightgbm tuned",
    "10": "10 histgb",
    "12": "12 calibration",
    "13": "13 stacking",
    "14": "14 safe champion slot",
    "15": "15 soft-vote / blend",
}


def load(folder: Path, side: str) -> pd.DataFrame:
    rows = []
    for path in sorted(folder.glob("[0-9][0-9]_*.json")):
        rec = json.loads(path.read_text())
        m = rec.get("metrics", {})
        if m.get("roc_auc") is None:
            continue
        rows.append(
            {
                "exp_id": str(rec["exp_id"]).zfill(2),
                "name": rec["name"],
                "side": side,
                "roc_auc": m["roc_auc"],
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "accuracy": m.get("accuracy"),
                "features": rec.get("feature_count"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    opt = load(OPT, "optimized_safe")
    unsafe = load(UNSAFE, "unsafe")
    safe = load(SAFE, "safe")
    if opt.empty or unsafe.empty:
        raise SystemExit("Need optimized_safe_model/results and hyperack_exp/results.")

    # Matched same-model ladder (shared exp_ids)
    m = opt.merge(
        unsafe,
        on="exp_id",
        how="inner",
        suffixes=("_opt", "_unsafe"),
    )
    m = m.merge(
        safe[["exp_id", "roc_auc", "recall", "f1"]].rename(
            columns={
                "roc_auc": "roc_auc_prev_safe",
                "recall": "recall_prev_safe",
                "f1": "f1_prev_safe",
            }
        ),
        on="exp_id",
        how="left",
    )
    m["delta_roc_vs_unsafe"] = m["roc_auc_opt"] - m["roc_auc_unsafe"]
    m["delta_recall_vs_unsafe"] = m["recall_opt"] - m["recall_unsafe"]
    m["label"] = m["exp_id"].map(LABELS).fillna(
        m["exp_id"] + " " + m["name_opt"].astype(str).str[:18]
    )
    m = m.sort_values("roc_auc_opt", ascending=True)

    out_csv = OPT / "optimized_vs_unsafe_comparison.csv"
    m.to_csv(out_csv, index=False)

    # --- Plot 1: side-by-side ROC + recall (matched models) ---
    y = np.arange(len(m))
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

    axes[0].barh(y - 0.2, m["roc_auc_unsafe"], height=0.4, color="#B42318", label="Unsafe (final fares)")
    axes[0].barh(y + 0.2, m["roc_auc_opt"], height=0.4, color="#16845B", label="Optimized safe")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(m["label"])
    axes[0].set_xlabel("ROC-AUC")
    axes[0].set_title("ROC-AUC: unsafe vs optimized safe")
    axes[0].set_xlim(0.92, 0.985)
    axes[0].legend(loc="lower right")

    axes[1].barh(y - 0.2, m["recall_unsafe"], height=0.4, color="#B42318", label="Unsafe")
    axes[1].barh(y + 0.2, m["recall_opt"], height=0.4, color="#16845B", label="Optimized safe")
    axes[1].set_xlabel("Recall")
    axes[1].set_title("Recall: unsafe vs optimized safe")
    axes[1].set_xlim(0.72, 0.94)
    axes[1].legend(loc="lower right")

    fig.suptitle(
        "Optimized safe vs unsafe (same exp_id, same test split)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    png1 = OPT / "optimized_vs_unsafe_side_by_side.png"
    fig.savefig(png1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: delta gap (how much ROC drops when staying safe) ---
    fig2, ax = plt.subplots(figsize=(10, 7))
    gap = -m["delta_roc_vs_unsafe"]  # positive = unsafe ahead
    colors = ["#B42318" if g > 0.001 else "#16845B" for g in gap]
    ax.barh(y, gap, color=colors, height=0.7)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(m["label"])
    ax.set_xlabel("ROC-AUC gap (unsafe − optimized safe)")
    ax.set_title("Leakage cost: how much ROC the unsafe final-fare features buy")
    ax.set_xlim(0, max(0.05, float(gap.max()) + 0.005))
    fig2.tight_layout()
    png2 = OPT / "optimized_vs_unsafe_roc_gap.png"
    fig2.savefig(png2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # --- Plot 3: champions overview ---
    best_opt = opt.loc[opt["roc_auc"].idxmax()]
    best_unsafe = unsafe.loc[unsafe["roc_auc"].idxmax()]
    best_prev_safe = safe.loc[safe["roc_auc"].idxmax()] if not safe.empty else None

    champ_names = [
        f"Best unsafe\n{best_unsafe['exp_id']} {best_unsafe['name'][:22]}",
        f"Best prev safe\n{best_prev_safe['exp_id']} {best_prev_safe['name'][:22]}"
        if best_prev_safe is not None
        else "Best prev safe",
        f"Best optimized safe\n{best_opt['exp_id']} {best_opt['name'][:22]}",
    ]
    champ_roc = [
        best_unsafe["roc_auc"],
        best_prev_safe["roc_auc"] if best_prev_safe is not None else np.nan,
        best_opt["roc_auc"],
    ]
    champ_rec = [
        best_unsafe["recall"],
        best_prev_safe["recall"] if best_prev_safe is not None else np.nan,
        best_opt["recall"],
    ]
    champ_colors = ["#B42318", "#9A6500", "#16845B"]

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(3)
    axes3[0].bar(x, champ_roc, color=champ_colors, width=0.65)
    axes3[0].set_xticks(x)
    axes3[0].set_xticklabels(champ_names, fontsize=9)
    axes3[0].set_ylabel("ROC-AUC")
    axes3[0].set_ylim(0.93, 0.985)
    axes3[0].set_title("Champions — ROC-AUC")
    for i, v in enumerate(champ_roc):
        if pd.notna(v):
            axes3[0].text(i, v + 0.0015, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

    axes3[1].bar(x, champ_rec, color=champ_colors, width=0.65)
    axes3[1].set_xticks(x)
    axes3[1].set_xticklabels(champ_names, fontsize=9)
    axes3[1].set_ylabel("Recall")
    axes3[1].set_ylim(0.70, 0.95)
    axes3[1].set_title("Champions — Recall")
    for i, v in enumerate(champ_rec):
        if pd.notna(v):
            axes3[1].text(i, v + 0.008, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

    fig3.suptitle(
        "HyperAck champions: unsafe vs previous safe vs optimized safe",
        fontsize=12,
        fontweight="bold",
    )
    fig3.tight_layout()
    png3 = OPT / "optimized_vs_unsafe_champions.png"
    fig3.savefig(png3, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # --- Plot 4: top-12 optimized safe overall (includes 16+) with unsafe best line ---
    top = opt.sort_values("roc_auc", ascending=True).tail(12)
    fig4, ax4 = plt.subplots(figsize=(10, 7))
    ax4.barh(
        np.arange(len(top)),
        top["roc_auc"],
        color="#16845B",
        height=0.7,
        label="Optimized safe",
    )
    ax4.axvline(
        best_unsafe["roc_auc"],
        color="#B42318",
        linestyle="--",
        linewidth=1.6,
        label=f"Best unsafe ({best_unsafe['roc_auc']:.4f})",
    )
    if best_prev_safe is not None:
        ax4.axvline(
            best_prev_safe["roc_auc"],
            color="#9A6500",
            linestyle="--",
            linewidth=1.4,
            label=f"Best prev safe ({best_prev_safe['roc_auc']:.4f})",
        )
    ax4.set_yticks(np.arange(len(top)))
    ax4.set_yticklabels(top["exp_id"] + " " + top["name"].astype(str).str[:26])
    ax4.set_xlabel("ROC-AUC")
    ax4.set_xlim(0.938, 0.982)
    ax4.set_title("Top optimized-safe models vs unsafe / prior-safe ceilings")
    ax4.legend(loc="lower right")
    fig4.tight_layout()
    png4 = OPT / "optimized_top_vs_unsafe_ceiling.png"
    fig4.savefig(png4, dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # Report snippet
    avg_gap = float((-m["delta_roc_vs_unsafe"]).mean())
    lines = [
        "# Optimized Safe vs Unsafe Benchmark",
        "",
        "Same locked test split. Optimized-safe models never use `final_customer_fare` / `final_biker_fare`.",
        "",
        "## Champions",
        f"- **Best unsafe:** {best_unsafe['exp_id']} {best_unsafe['name']} "
        f"(ROC={best_unsafe['roc_auc']:.4f}, recall={best_unsafe['recall']:.4f})",
        f"- **Best previous safe:** {best_prev_safe['exp_id']} {best_prev_safe['name']} "
        f"(ROC={best_prev_safe['roc_auc']:.4f}, recall={best_prev_safe['recall']:.4f})"
        if best_prev_safe is not None
        else "- **Best previous safe:** n/a",
        f"- **Best optimized safe:** {best_opt['exp_id']} {best_opt['name']} "
        f"(ROC={best_opt['roc_auc']:.4f}, recall={best_opt['recall']:.4f})",
        f"- **Avg ROC gap (matched 01–15):** unsafe ahead by {avg_gap:.4f}",
        "",
        "## Matched ladder (optimized safe vs unsafe)",
        "",
        "| Exp | Optimized safe | ROC opt | ROC unsafe | ΔROC | Recall opt | Recall unsafe |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in m.sort_values("roc_auc_opt", ascending=False).iterrows():
        lines.append(
            f"| {r['exp_id']} | {r['name_opt']} | {r['roc_auc_opt']:.4f} | "
            f"{r['roc_auc_unsafe']:.4f} | {r['delta_roc_vs_unsafe']:+.4f} | "
            f"{r['recall_opt']:.4f} | {r['recall_unsafe']:.4f} |"
        )
    lines += [
        "",
        "## Plots",
        f"- {png1}",
        f"- {png2}",
        f"- {png3}",
        f"- {png4}",
        f"- {out_csv}",
        "",
    ]
    report = ROOT / "OPTIMIZED_VS_UNSAFE_BENCHMARK.md"
    report.write_text("\n".join(lines) + "\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {png1}")
    print(f"Wrote {png2}")
    print(f"Wrote {png3}")
    print(f"Wrote {png4}")
    print(f"Wrote {report}")
    print(
        m.sort_values("roc_auc_opt", ascending=False)[
            ["exp_id", "name_opt", "roc_auc_opt", "roc_auc_unsafe", "delta_roc_vs_unsafe"]
        ]
        .head(8)
        .to_string(index=False)
    )
    prev = best_prev_safe["roc_auc"] if best_prev_safe is not None else float("nan")
    print(
        f"Champions: unsafe={best_unsafe['roc_auc']:.4f}  "
        f"opt={best_opt['roc_auc']:.4f}  prev_safe={prev:.4f}"
    )


if __name__ == "__main__":
    main()
