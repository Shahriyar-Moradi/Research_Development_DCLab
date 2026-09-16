"""Compare leakage-safe runs vs original (possibly unsafe) HyperAck results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
UNSAFE_DIR = ROOT.parent / "hyperack_exp" / "results"
SAFE_DIR = ROOT / "results"


def load_side(folder: Path, side: str) -> pd.DataFrame:
    rows = []
    for path in sorted(folder.glob("[0-9][0-9]_*.json")):
        rec = json.loads(path.read_text())
        m = rec.get("metrics", {})
        if m.get("roc_auc") is None:
            rows.append(
                {
                    "exp_id": rec["exp_id"],
                    "name": rec["name"],
                    "side": side,
                    "roc_auc": None,
                    "recall": None,
                    "f1": None,
                    "accuracy": None,
                    "features": rec.get("feature_count"),
                    "status": m.get("status", "skipped"),
                }
            )
            continue
        rows.append(
            {
                "exp_id": rec["exp_id"],
                "name": rec["name"],
                "side": side,
                "roc_auc": m.get("roc_auc"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "accuracy": m.get("accuracy"),
                "avg_precision": m.get("avg_precision"),
                "features": rec.get("feature_count"),
                "fit_seconds": m.get("fit_seconds"),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    unsafe = load_side(UNSAFE_DIR, "unsafe")
    safe = load_side(SAFE_DIR, "safe")
    if safe.empty:
        raise SystemExit("No safe results found. Run run_all_safe.py first.")

    merged = unsafe.merge(
        safe,
        on=["exp_id", "name"],
        suffixes=("_unsafe", "_safe"),
        how="outer",
    )
    for col in ["roc_auc", "recall", "f1", "accuracy"]:
        u, s = f"{col}_unsafe", f"{col}_safe"
        if u in merged.columns and s in merged.columns:
            merged[f"delta_{col}"] = merged[s] - merged[u]

    merged = merged.sort_values("exp_id").reset_index(drop=True)
    out_csv = SAFE_DIR / "safe_vs_unsafe_comparison.csv"
    merged.to_csv(out_csv, index=False)

    ok = merged.dropna(subset=["roc_auc_safe", "roc_auc_unsafe"]).copy()
    ok = ok.sort_values("roc_auc_safe", ascending=False)

    lines = []
    lines.append("# Safe vs Unsafe HyperAck Report\n")
    lines.append(
        "This report compares models trained **without final fares** "
        "(`safe_leakage_exp`) against the original suite that **allowed final fares** "
        "(`hyperack_exp`). Both use the same locked train/test split.\n"
    )
    lines.append("## What changed in the safe suite\n")
    lines.append("- Dropped unsafe columns: `final_customer_fare`, `final_biker_fare`")
    lines.append("- Kept safe signals: first fare, distance, time, geo, category, products")
    lines.append("- Same experiment strategies / model families as before\n")

    if not ok.empty:
        best_safe = ok.iloc[0]
        best_unsafe = merged.dropna(subset=["roc_auc_unsafe"]).sort_values(
            "roc_auc_unsafe", ascending=False
        ).iloc[0]
        lines.append("## Headline results\n")
        lines.append(
            f"- **Best safe model:** {best_safe['exp_id']} - {best_safe['name']} "
            f"(ROC-AUC={best_safe['roc_auc_safe']:.4f}, recall={best_safe['recall_safe']:.4f})"
        )
        lines.append(
            f"- **Best unsafe model:** {best_unsafe['exp_id']} - {best_unsafe['name']} "
            f"(ROC-AUC={best_unsafe['roc_auc_unsafe']:.4f}, recall={best_unsafe['recall_unsafe']:.4f})"
        )
        lines.append(
            f"- **Gap (best unsafe − best safe) ROC-AUC:** "
            f"{best_unsafe['roc_auc_unsafe'] - best_safe['roc_auc_safe']:.4f}"
        )
        lines.append(
            f"- **Average ROC-AUC drop when going safe:** "
            f"{(-ok['delta_roc_auc']).mean():.4f}"
        )
        lines.append("")

    lines.append("## Per-experiment comparison (sorted by safe ROC-AUC)\n")
    lines.append(
        "| Exp | Name | Unsafe ROC-AUC | Safe ROC-AUC | Δ ROC-AUC | Unsafe Recall | Safe Recall | Δ Recall |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in ok.iterrows():
        lines.append(
            f"| {r['exp_id']} | {r['name']} | "
            f"{r['roc_auc_unsafe']:.4f} | {r['roc_auc_safe']:.4f} | {r['delta_roc_auc']:+.4f} | "
            f"{r['recall_unsafe']:.4f} | {r['recall_safe']:.4f} | {r['delta_recall']:+.4f} |"
        )

    lines.append("\n## How to read this\n")
    lines.append(
        "- **Negative Δ** means the safe model is worse on that metric (expected if final fares were helpful but leaky)."
    )
    lines.append(
        "- Prefer **safe** models for production if final fares are not known at prediction time."
    )
    lines.append(
        "- Unsafe scores are an optimistic lab ceiling, not a deployable guarantee.\n"
    )
    lines.append("## Files\n")
    lines.append(f"- Comparison CSV: `{out_csv}`")
    lines.append("- Safe results: `safe_leakage_exp/results/`")
    lines.append("- Unsafe results: `hyperack_exp/results/`\n")

    report = ROOT / "SAFE_VS_UNSAFE_REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {report}")
    if not ok.empty:
        print(ok[["exp_id", "name", "roc_auc_unsafe", "roc_auc_safe", "delta_roc_auc"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
