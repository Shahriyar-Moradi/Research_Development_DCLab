"""Compare optimized-safe vs previous safe (and unsafe) HyperAck results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OPT = ROOT / "results"
SAFE = ROOT.parent / "safe_leakage_exp" / "results"
UNSAFE = ROOT.parent / "hyperack_exp" / "results"


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
    safe = load(SAFE, "safe")
    unsafe = load(UNSAFE, "unsafe")
    if opt.empty:
        raise SystemExit("No optimized results. Run run_all_optimized.py first.")

    # Merge on exp_id where possible
    m = opt.merge(safe, on="exp_id", how="left", suffixes=("_opt", "_safe"))
    m = m.merge(
        unsafe[["exp_id", "roc_auc", "recall", "f1"]].rename(
            columns={
                "roc_auc": "roc_auc_unsafe",
                "recall": "recall_unsafe",
                "f1": "f1_unsafe",
            }
        ),
        on="exp_id",
        how="left",
    )
    m["delta_vs_safe_roc"] = m["roc_auc_opt"] - m["roc_auc_safe"]
    m["delta_vs_safe_recall"] = m["recall_opt"] - m["recall_safe"]
    m = m.sort_values("roc_auc_opt", ascending=False)
    out_csv = OPT / "optimized_vs_safe_comparison.csv"
    m.to_csv(out_csv, index=False)

    # Plot: top optimized models + matched safe pairs
    prior_best = float(safe["roc_auc"].max()) if not safe.empty else float("nan")
    top = m.head(15).sort_values("roc_auc_opt")
    labels = top["exp_id"] + " " + top["name_opt"].astype(str).str[:22]
    y = range(len(top))
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True)
    axes[0].barh(list(y), top["roc_auc_opt"], height=0.7, color="#16845B", label="Optimized safe")
    if pd.notna(prior_best):
        axes[0].axvline(prior_best, color="#9A6500", linestyle="--", linewidth=1.5, label=f"Prior best safe ({prior_best:.4f})")
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("ROC-AUC")
    axes[0].set_title("Top optimized-safe ROC-AUC")
    axes[0].set_xlim(0.93, 0.9505)
    axes[0].legend(loc="lower right")

    axes[1].barh(list(y), top["recall_opt"], height=0.7, color="#2457D6", label="Optimized recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_title("Recall for same top models")
    axes[1].set_xlim(0.72, 0.90)
    axes[1].legend(loc="lower right")
    fig.suptitle("Optimized leakage-safe models (same test split, no final fares)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    png = OPT / "optimized_vs_safe_side_by_side.png"
    fig.savefig(png, dpi=140, bbox_inches="tight")

    # Matched-pair plot when available
    pair = m.dropna(subset=["roc_auc_safe", "roc_auc_opt"]).sort_values("roc_auc_opt")
    if not pair.empty:
        fig2, ax = plt.subplots(figsize=(10, 6))
        yy = range(len(pair))
        ax.barh([i - 0.2 for i in yy], pair["roc_auc_safe"], height=0.4, color="#9A6500", label="Previous safe")
        ax.barh([i + 0.2 for i in yy], pair["roc_auc_opt"], height=0.4, color="#16845B", label="Optimized safe")
        ax.set_yticks(list(yy))
        ax.set_yticklabels(pair["exp_id"] + " " + pair["name_opt"].astype(str).str[:18])
        ax.set_xlabel("ROC-AUC")
        ax.set_xlim(0.92, 0.96)
        ax.legend(loc="lower right")
        ax.set_title("Matched exp_id: previous safe vs optimized")
        fig2.tight_layout()
        png2 = OPT / "optimized_vs_safe_matched_pairs.png"
        fig2.savefig(png2, dpi=140, bbox_inches="tight")
        print(f"Wrote {png2}")

    best = m.iloc[0]
    prior_best = float(safe["roc_auc"].max()) if not safe.empty else float("nan")
    prior_best_name = (
        safe.loc[safe["roc_auc"].idxmax(), "name"] if not safe.empty else "n/a"
    )
    lift_vs_best = (
        best["roc_auc_opt"] - prior_best if pd.notna(prior_best) else float("nan")
    )
    lines = [
        "# Optimized Safe Models Report",
        "",
        "Goal: improve HyperAck models **without** using unsafe final fares.",
        "",
        "## What we changed (still leakage-safe)",
        "- Kept dropping `final_customer_fare` / `final_biker_fare`",
        "- Added richer **safe** FE: first-fare ratios, time flags (morning/night), route ratios, interactions, train-only geo clusters + quantile bins",
        "- Used class imbalance handling (`scale_pos_weight` / `class_weight=balanced`)",
        "- Stronger tuning (more trials) for XGB/LGBM",
        "- Recovery passes: replay prior winner, local search around it, seed bags, soft votes",
        "- Better ensembles: 4-model stacking + 3-model soft vote",
        "- Threshold tuned for F1+recall blend (exp 12)",
        "",
        "## Headline",
        f"- **Best optimized safe:** {best['exp_id']} - {best['name_opt']} "
        f"(ROC-AUC={best['roc_auc_opt']:.4f}, recall={best['recall_opt']:.4f})",
        f"- **Prior best safe:** {prior_best_name} (ROC-AUC={prior_best:.4f})",
        f"- **Lift vs prior best safe ROC:** {lift_vs_best:+.4f}",
        f"- **Avg Δ vs matched safe exp:** "
        f"ROC {m['delta_vs_safe_roc'].mean():.4f}, recall {m['delta_vs_safe_recall'].mean():.4f}",
        "",
        "## Leaderboard (optimized safe)",
        "",
        "| Rank | Exp | Name | ROC-AUC | Recall | F1 | vs prev safe ΔROC |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for i, (_, r) in enumerate(m.iterrows(), 1):
        d = r["delta_vs_safe_roc"]
        d_s = f"{d:+.4f}" if pd.notna(d) else "n/a"
        lines.append(
            f"| {i} | {r['exp_id']} | {r['name_opt']} | {r['roc_auc_opt']:.4f} | "
            f"{r['recall_opt']:.4f} | {r['f1_opt']:.4f} | {d_s} |"
        )
    lines += [
        "",
        "## Files",
        f"- {out_csv}",
        f"- {png}",
        "- results/*.json",
        "",
        "## Simple takeaway",
        "Leakage-safe ceiling on this split is ~0.9450 ROC-AUC (replay of prior safe LightGBM winner).",
        "Richer FE, class weights, CatBoost, Optuna, TE, ablation, and ensembles did not beat that peak,",
        "but several models improve recall vs the prior winner (0.7707) while staying safe (no final fares).",
        "Champion for ROC: exp 21 `lgbm_prior_winner_replay`. See plots for the full ladder.",
        "",
    ]
    report = ROOT / "OPTIMIZED_SAFE_REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {png}")
    print(f"Wrote {report}")
    print(m[["exp_id", "name_opt", "roc_auc_opt", "recall_opt", "delta_vs_safe_roc"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
