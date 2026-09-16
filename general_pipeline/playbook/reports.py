"""Detailed HyperAck-style reports + comparison plots for external projects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from general_pipeline.external_catalog import DATASET_CATALOG
from general_pipeline.external_project import project_dir
from general_pipeline.playbook.ladder import results_frame
from general_pipeline.playbook.policy import get_policy


def _spec(key: str):
    return next(s for s in DATASET_CATALOG if s.key == key)


def _annotate(ax, bars, vals):
    for b, v in zip(bars, vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ax.text(float(v) + 0.002, b.get_y() + b.get_height() / 2, f"{float(v):.3f}",
                va="center", fontsize=7, color="#334155", clip_on=False)


def plot_ladder_comparisons(key: str, df: pd.DataFrame) -> None:
    out = project_dir(key) / "benchmark_outputs"
    out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    sns.set_theme(style="whitegrid")
    name = _spec(key).name

    # Safe vs Unsafe for each exp_id (leakage cost)
    safe = df[df["mode"] == "safe"].set_index("exp_name")
    unsafe = df[df["mode"] == "unsafe"].set_index("exp_name")
    common = sorted(set(safe.index) & set(unsafe.index), key=lambda n: int(safe.loc[n, "exp_id"]) if "exp_id" in safe.columns else n)

    if common:
        fig, ax = plt.subplots(figsize=(12, max(6, 0.4 * len(common) + 2)))
        y = np.arange(len(common))
        svals = [float(safe.loc[n, "roc_auc"]) for n in common]
        uvals = [float(unsafe.loc[n, "roc_auc"]) for n in common]
        b1 = ax.barh(y - 0.18, svals, 0.35, label="Safe", color="#16845B")
        b2 = ax.barh(y + 0.18, uvals, 0.35, label="Unsafe", color="#B42318")
        _annotate(ax, b1, svals)
        _annotate(ax, b2, uvals)
        ax.set_yticks(y)
        ax.set_yticklabels(common)
        ax.set_xlabel("ROC-AUC")
        ax.set_title(f"{name}: Safe vs Unsafe by Experiment")
        ax.legend()
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(out / "safe_vs_unsafe_by_experiment.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        gaps = pd.DataFrame({"exp_name": common, "safe": svals, "unsafe": uvals})
        gaps["leakage_gap"] = gaps["unsafe"] - gaps["safe"]
        gaps.to_csv(out / "leakage_gap_by_experiment.csv", index=False)

    # FE stage lift (safe mode)
    fe_order = ["raw", "logs", "ratios", "interactions", "full_fe", "selected"]
    fe_df = df[(df["mode"] == "safe") & (df["exp_name"].isin([
        "baseline_lightgbm", "fe_log_transforms", "fe_ratios", "fe_interactions",
        "fe_full_clusters_bins", "feature_selection_mi",
    ]))]
    if not fe_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        fe_df = fe_df.sort_values("exp_id")
        bars = ax.bar(fe_df["exp_name"], fe_df["roc_auc"], color="#2563EB", alpha=0.9)
        for b, v in zip(bars, fe_df["roc_auc"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=8, rotation=90)
        ax.set_ylabel("ROC-AUC")
        ax.set_title(f"{name}: Feature Engineering Ladder (Safe)")
        ax.set_ylim(max(0.5, fe_df["roc_auc"].min() - 0.05), min(1.02, fe_df["roc_auc"].max() + 0.05))
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        fig.savefig(out / "fe_ladder_safe.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Optimization methods comparison (safe)
    opt_names = [
        "baseline_lightgbm", "fe_full_clusters_bins", "tuned_xgboost", "tuned_lightgbm",
        "hist_gradient_boosting", "extra_trees_strong", "calibrated_lgbm_threshold",
        "stacking_ensemble", "soft_vote_blend",
    ]
    opt_df = df[(df["mode"] == "safe") & (df["exp_name"].isin(opt_names))].sort_values("roc_auc")
    if not opt_df.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        bars = ax.barh(opt_df["exp_name"], opt_df["roc_auc"], color="#16845B", alpha=0.9)
        _annotate(ax, bars, opt_df["roc_auc"].tolist())
        ax.set_xlabel("ROC-AUC")
        ax.set_title(f"{name}: Optimization Methods (Safe)")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(out / "optimization_methods_safe.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Quadrant summary: safe/unsafe × baseline/optimized proxies
    # baseline ~ exp 02, optimized ~ best of tuned/stack/softvote
    summary_rows = []
    for mode in df["mode"].unique():
        sub = df[df["mode"] == mode]
        base = sub[sub["exp_name"] == "baseline_lightgbm"]
        opt = sub[sub["exp_name"].isin(["tuned_lightgbm", "tuned_xgboost", "stacking_ensemble", "soft_vote_blend"])]
        if len(base):
            summary_rows.append({"mode": mode, "arm": "baseline", "roc_auc": float(base.iloc[0]["roc_auc"]), "exp": "baseline_lightgbm"})
        if len(opt):
            best = opt.sort_values("roc_auc", ascending=False).iloc[0]
            summary_rows.append({"mode": mode, "arm": "optimized", "roc_auc": float(best["roc_auc"]), "exp": best["exp_name"]})
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(out / "quadrant_summary.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        labels, vals, colors = [], [], []
        cmap = {"safe_baseline": "#D97706", "safe_optimized": "#16845B", "unsafe_baseline": "#EF4444", "unsafe_optimized": "#7F1D1D"}
        for _, r in sdf.iterrows():
            lab = f"{r['mode']}_{r['arm']}"
            labels.append(lab)
            vals.append(r["roc_auc"])
            colors.append(cmap.get(lab, "#2563EB"))
        bars = ax.bar(labels, vals, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_ylabel("ROC-AUC")
        ax.set_title(f"{name}: Safe/Unsafe × Baseline/Optimized")
        plt.xticks(rotation=15)
        plt.tight_layout()
        fig.savefig(out / "quadrant_safe_unsafe_opt.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def write_detailed_reports(key: str, df: Optional[pd.DataFrame] = None) -> None:
    df = results_frame(key) if df is None else df
    pdir = project_dir(key)
    out = pdir / "benchmark_outputs"
    out.mkdir(parents=True, exist_ok=True)
    plot_ladder_comparisons(key, df)

    spec = _spec(key)
    policy = get_policy(key)
    meta = json.loads((pdir / "data" / "meta.json").read_text()) if (pdir / "data" / "meta.json").exists() else {}
    # data may be symlink
    from general_pipeline.playbook.ladder import EXTERNAL_DATA
    meta = json.loads((EXTERNAL_DATA / key / "meta.json").read_text())

    if df.empty:
        (pdir / "FEATURE_ENGINEERING_REPORT.md").write_text(f"# {spec.name}\n\nNo ladder results.\n")
        return

    safe = df[df["mode"] == "safe"].sort_values("exp_id")
    unsafe = df[df["mode"] == "unsafe"].sort_values("exp_id")
    best_safe = safe.sort_values("roc_auc", ascending=False).iloc[0]
    best_unsafe = unsafe.sort_values("roc_auc", ascending=False).iloc[0] if not unsafe.empty else best_safe
    best_opt = safe[safe["optimization_method"] != "none"].sort_values("roc_auc", ascending=False)
    best_opt_row = best_opt.iloc[0] if not best_opt.empty else best_safe

    # FE stage table
    fe_rows = []
    for _, r in safe.iterrows():
        fe_rows.append(
            f"| {int(r['exp_id']):02d} | {r['exp_name']} | {r['fe_stage']} | {int(r['feature_count'])} | "
            f"{r['roc_auc']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['optimization_method']} |"
        )

    # Leakage comparison table
    leak_rows = []
    if not unsafe.empty:
        for name in sorted(set(safe["exp_name"]) & set(unsafe["exp_name"])):
            s = safe[safe["exp_name"] == name].iloc[0]
            u = unsafe[unsafe["exp_name"] == name].iloc[0]
            leak_rows.append(
                f"| {name} | {s['roc_auc']:.4f} | {u['roc_auc']:.4f} | {u['roc_auc']-s['roc_auc']:+.4f} | "
                f"{s['recall']:.4f} | {u['recall']:.4f} |"
            )

    # Optimization ranking
    opt_rank = safe.sort_values("roc_auc", ascending=False)
    opt_lines = []
    for i, (_, r) in enumerate(opt_rank.iterrows(), 1):
        note = str(r.get("notes") or "").replace("|", "/").replace("\n", " ")[:80]
        opt_lines.append(
            f"| {i} | {r['exp_name']} | {r['optimization_method']} | {r['roc_auc']:.4f} | {r['f1']:.4f} | {note} |"
        )

    fe_report = f"""# {spec.name} — Feature Engineering & Selection Report

**Project:** `external_projects/{key}_exp/`  
**Source:** [{spec.url}]({spec.url})  
**Rows / features (raw):** {meta.get('n_rows')} / {meta.get('n_features')}  
**Positive rate:** {meta.get('pos_rate', 0):.3f}  
**Protocol:** stratified 80/20, seed 42; all stateful FE fit on **train only**

---

## 1. Problem & decision-time policy

{spec.description}

### Leakage / safe vs unsafe
- **Has classic leakage columns:** {policy.has_leakage}
- **Unsafe-only features:** `{policy.unsafe_only_features or '[]'}`
- **Rationale:** {policy.rationale}

When leakage exists, **safe** drops post-outcome columns (HyperAck analog of final fares).  
When it does not, safe and unsafe matrices are identical — the ladder still documents FE and optimization lifts.

---

## 2. Feature engineering ladder (what we tried)

| Stage | Idea | Why (HyperAck playbook) |
|---|---|---|
| raw | Original numeric matrix | Floor score |
| logs | `log1p` on skewed non-negative cols | Tame heavy tails |
| ratios | Pairwise intensity ratios | Scale-normalized signals |
| interactions | Products of top-MI features | Non-linear combinations |
| full_fe | + KMeans clusters + quantile bins | Spatial/soft thresholds (train-fit) |
| selected | Top mutual-information subset | Test dilution vs compact set |

---

## 3. Safe-mode experiment results

| ID | Experiment | FE stage | #Feats | ROC-AUC | Recall | F1 | Optimization |
|---|---|---|---:|---:|---:|---:|---|
{chr(10).join(fe_rows)}

**Best safe model:** `{best_safe['exp_name']}` — ROC-AUC **{best_safe['roc_auc']:.4f}**, F1 **{best_safe['f1']:.4f}**  
**Best optimization method (safe):** `{best_opt_row['optimization_method']}` via `{best_opt_row['exp_name']}` (AUC {best_opt_row['roc_auc']:.4f})

### FE stage chart
![FE ladder](benchmark_outputs/fe_ladder_safe.png)

---

## 4. Feature selection findings

Experiment `07 feature_selection_mi` keeps a top-MI subset of the full FE matrix.  
Compare its ROC-AUC against `06 fe_full_clusters_bins`:

"""
    sel = safe[safe["exp_name"] == "feature_selection_mi"]
    full = safe[safe["exp_name"] == "fe_full_clusters_bins"]
    if len(sel) and len(full):
        delta = float(sel.iloc[0]["roc_auc"] - full.iloc[0]["roc_auc"])
        fe_report += (
            f"- Full FE AUC: **{full.iloc[0]['roc_auc']:.4f}** ({int(full.iloc[0]['feature_count'])} feats)\n"
            f"- Selected AUC: **{sel.iloc[0]['roc_auc']:.4f}** ({int(sel.iloc[0]['feature_count'])} feats)\n"
            f"- Δ: **{delta:+.4f}** — "
            + ("selection helped or matched." if delta >= -0.002 else "full FE retained more signal (dilution not an issue / selection too aggressive).")
            + "\n"
        )

    fe_report += f"""

---

## 5. Figures
- `benchmark_outputs/fe_ladder_safe.png`
- `benchmark_outputs/safe_vs_unsafe_by_experiment.png`
- `benchmark_outputs/ladder_results.csv`
"""
    (pdir / "FEATURE_ENGINEERING_REPORT.md").write_text(fe_report)

    opt_report = f"""# {spec.name} — Optimization Methods Report

**Question:** Which optimization methods lift ROC-AUC the most on this dataset?

---

## 1. Methods tested

| Method | Experiment | Description |
|---|---|---|
| none (baseline) | 02 baseline_lightgbm | Default LightGBM on raw features |
| FE only | 03–06 | Feature engineering without hyperparameter search |
| mutual_info_selection | 07 | Compact MI feature subset |
| RandomizedSearchCV | 08 tuned_xgboost, 09 tuned_lightgbm | 3-fold CV maximizing ROC-AUC |
| hand_tuned | 11 extra_trees_strong | Playbook ExtraTrees region |
| isotonic_calibration | 12 calibrated_lgbm | Probability calibration |
| oof_stacking | 13 stacking_ensemble | ET+LGBM+XGB → logistic meta |
| soft_probability_blend | 14 soft_vote_blend | Weighted probability average |

---

## 2. Safe-mode ranking (best → worst)

| Rank | Experiment | Method | ROC-AUC | F1 | Notes |
|---:|---|---|---:|---:|---|
{chr(10).join(opt_lines)}

**Winner:** `{best_safe['exp_name']}` with method `{best_safe['optimization_method']}`  
**Best pure search method:** `{best_opt_row['exp_name']}` (`{best_opt_row['optimization_method']}`)

![Optimization methods](benchmark_outputs/optimization_methods_safe.png)

---

## 3. Safe vs Optimized Safe

| Arm | Experiment | ROC-AUC | Recall | F1 |
|---|---|---:|---:|---:|
| Safe baseline | baseline_lightgbm | {float(safe[safe['exp_name']=='baseline_lightgbm'].iloc[0]['roc_auc']) if len(safe[safe['exp_name']=='baseline_lightgbm']) else float('nan'):.4f} | — | — |
| Safe optimized (best) | {best_safe['exp_name']} | {best_safe['roc_auc']:.4f} | {best_safe['recall']:.4f} | {best_safe['f1']:.4f} |

Lift (best − baseline LGBM): see ranking table and `quadrant_safe_unsafe_opt.png`.

---

## 4. Unsafe vs Optimized Unsafe / Safe vs Unsafe

"""
    if policy.has_leakage and leak_rows:
        opt_report += f"""Leakage columns `{policy.unsafe_only_features}` inflate unsafe scores.

| Experiment | Safe AUC | Unsafe AUC | Δ (leakage) | Safe Rec | Unsafe Rec |
|---|---:|---:|---:|---:|---:|
{chr(10).join(leak_rows)}

**Best unsafe:** `{best_unsafe['exp_name']}` AUC **{best_unsafe['roc_auc']:.4f}**  
**Leakage cost (best unsafe − best safe):** {best_unsafe['roc_auc'] - best_safe['roc_auc']:+.4f}

![Safe vs Unsafe](benchmark_outputs/safe_vs_unsafe_by_experiment.png)
![Quadrants](benchmark_outputs/quadrant_safe_unsafe_opt.png)
"""
    else:
        opt_report += (
            "No post-outcome leakage columns for this dataset — safe and unsafe feature matrices are "
            "**identical**. Differences across modes (if any) are numerical noise only. "
            "Focus optimization comparisons on the safe ranking above.\n"
        )

    opt_report += """

---

## 5. Recommendation for production

1. Prefer **safe** features only when leakage columns exist.
2. Start from **full FE** then test MI selection.
3. Apply **RandomizedSearchCV** on LightGBM/XGBoost before stacking.
4. Use **soft-vote / stacking** only if they beat the best single tuned model on the locked test set.
"""
    (pdir / "OPTIMIZATION_REPORT.md").write_text(opt_report)

    # Master combined report
    master = f"""# {spec.name} — Complete Master Report

HyperAck-style end-to-end study for **{spec.name}**.

## Executive summary

| Item | Value |
|---|---|
| Best safe ROC-AUC | **{best_safe['roc_auc']:.4f}** (`{best_safe['exp_name']}`) |
| Best unsafe ROC-AUC | **{best_unsafe['roc_auc']:.4f}** (`{best_unsafe['exp_name']}`) |
| Best optimization method | `{best_opt_row['optimization_method']}` |
| Leakage present | {policy.has_leakage} |
| Leakage columns | `{policy.unsafe_only_features}` |

```
Unsafe ceiling (if leakage)  ──▶  {best_unsafe['roc_auc']:.4f}
        │ drop leakage
        ▼
Safe baseline / FE ladder    ──▶  see FEATURE_ENGINEERING_REPORT.md
        │ RandomizedSearch / ensembles
        ▼
Safe optimized champion      ──▶  {best_safe['roc_auc']:.4f}  ({best_safe['exp_name']})
```

## Detailed documents

1. [FEATURE_ENGINEERING_REPORT.md](FEATURE_ENGINEERING_REPORT.md) — FE stages, selection, safe results  
2. [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) — tuning/ensembling methods & which won  
3. [MASTER_BENCHMARK_REPORT.md](MASTER_BENCHMARK_REPORT.md) — model-zoo benchmark  
4. Notebooks in `notebooks/` — step-by-step reproducible narrative  

## Key figures

![Quadrants](benchmark_outputs/quadrant_safe_unsafe_opt.png)

![FE ladder](benchmark_outputs/fe_ladder_safe.png)

![Optimization](benchmark_outputs/optimization_methods_safe.png)

![Safe vs Unsafe](benchmark_outputs/safe_vs_unsafe_by_experiment.png)

## Reproduce

```bash
.venv/bin/python general_pipeline/playbook/run_playbook.py --dataset {key}
.venv/bin/python external_projects/{key}_exp/run_all.py
```

## Future datasets

Use the same playbook pipeline:

```bash
# 1) add dataset to external_catalog + download
# 2) define leakage in playbook/policy.py
# 3) run:
.venv/bin/python general_pipeline/playbook/run_playbook.py --dataset <new_key>
```
"""
    (pdir / "COMPLETE_MASTER_REPORT.md").write_text(master)
    print(f"[{key}] wrote FEATURE_ENGINEERING_REPORT, OPTIMIZATION_REPORT, COMPLETE_MASTER_REPORT")
