"""
workflow_samples.py
===================

SFT samples for the four other DCLab workflow stages (the leakage audit has its own
generator in `leakage_audit_samples.py`):

    data_understanding      -> what must happen before anything else (duplicates, imbalance,
                               missingness, small data, or proceed)
    feature_engineering     -> keep the raw features or adopt an engineered recipe
    model_selection         -> which model family to carry forward
    optimization_reliability-> calibrate or not; accept the final holdout or investigate a gap

How labels are made
-------------------
* data_understanding: one problem is PLANTED in a real dataset (duplicate rows, positives
  removed, values blanked, rows cut). The checker confirms the planted problem is the only
  one that crosses the stated thresholds.
* the other three stages: every candidate is actually TRAINED and EVALUATED on paired folds
  of real data (sometimes with a simulated target built from real columns, to get variety,
  which the dataset card says openly). The label is the repo's own rule applied to the
  numbers shown in the input (DCLAB-R07/R13/R14/R16/R17/R18/R19), so the answer is always
  recoverable from the evidence.

Every answer cites only numbers that appear in its input, and splits are by dataset.

Usage
-----
    python research/llm-fine-tuning/experiments/sft_factory/workflow_samples.py                       # all stages, all datasets
    python research/llm-fine-tuning/experiments/sft_factory/workflow_samples.py --stages model_selection --variants 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from leakage_audit_samples import CATALOG, HERE, SYSTEM_PROMPT, TARGETS, load, univariate_auc

# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------

MODELS = {
    "logistic_regression": lambda s: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                   LogisticRegression(max_iter=3000)),
    "random_forest": lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                             RandomForestClassifier(150, min_samples_leaf=2, n_jobs=-1, random_state=s)),
    "extra_trees": lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                           ExtraTreesClassifier(150, min_samples_leaf=2, n_jobs=-1, random_state=s)),
    "hist_gradient_boosting": lambda s: HistGradientBoostingClassifier(max_iter=120, random_state=s),
}


# Calibration stage also sees class-weighted variants: reweighting classes is a common, real
# cause of distorted probabilities, so these are where calibration genuinely pays off.
CAL_MODELS = {
    **MODELS,
    "logistic_regression_balanced_weights": lambda s: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    "random_forest_balanced_weights": lambda s: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(150, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=s)),
}


def safe_base(key, X):
    blocked = [c for c in CATALOG[key].get("blocked", []) if c in X.columns]
    return X.drop(columns=blocked)


def folds(y, k, seed):
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def fold_aucs(model_name, X, y, splits, seed):
    t = time.time()
    out = []
    for tr, te in splits:
        m = MODELS[model_name](seed).fit(X.iloc[tr], y.iloc[tr])
        out.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return np.array(out), time.time() - t


def card(key, meta, X, y, note=None):
    lines = [
        f"DATASET: {meta['name']} ({key}). Predicts {TARGETS[key]}.",
        f"DECISION MOMENT: {CATALOG[key]['decision']}",
    ]
    if note:
        lines.append(f"NOTE: {note}")
    lines.append(f"ROWS USED: {len(X):,} | FEATURES: {X.shape[1]} | POSITIVE RATE: {y.mean():.1%}")
    return "\n".join(lines)


def make_sample(stage, stage_card, dcard, body, actions, guide, decision, why, rule, confidence, nxt, meta):
    user = (f"STAGE: {stage}\n{stage_card}\n\n{dcard}\n\n{body}\n\n"
            f"ALLOWED ACTIONS: {' | '.join(actions)}\n({guide})")
    assistant = f"DECISION: {decision}\nWHY: {why}\nRULE: {rule}\nCONFIDENCE: {confidence}\nNEXT: {nxt}"
    assert decision in actions
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}],
            "metadata": {**meta, "stage": stage, "action": decision}}


def simulated_target(X, rng, kind, return_cols=False):
    """Outcome built from real columns, so a known mechanism drives it."""
    cont = [c for c in X.columns if X[c].nunique() > 10] or list(X.columns)
    skewed = [c for c in cont if X[c].min() >= 0 and abs(X[c].skew()) > 1]
    if kind == "log" and skewed:
        cols = list(rng.choice(skewed, size=min(3, len(skewed)), replace=False))
        lg = sum(np.log1p(X[c]).to_numpy() / (np.log1p(X[c]).std() + 1e-9) for c in cols)
        score = lg + rng.normal(0, 0.5, len(X))
        y = pd.Series((score > np.median(score)).astype(int))
        return (y, cols) if return_cols else y
    a, b = rng.choice(cont, size=2, replace=False) if len(cont) >= 2 else (cont[0], cont[0])
    if kind == "ratio":
        r = (X[a] / (X[b].abs() + 1e-3)).rank(pct=True).to_numpy()
        score = r * 4 + rng.normal(0, 0.5, len(X))
        y = pd.Series((score > np.median(score)).astype(int))
        return (y, [a, b]) if return_cols else y
    za = X[a].rank(pct=True).to_numpy() - 0.5
    zb = X[b].rank(pct=True).to_numpy() - 0.5
    noise = rng.normal(0, 1, len(X))
    if kind == "interaction":
        score = za * zb * 12 + noise * 0.6
    elif kind == "linear":
        cols = rng.choice(cont, size=min(5, len(cont)), replace=False)
        score = sum(rng.uniform(0.5, 2) * (X[c].rank(pct=True).to_numpy() - 0.5) for c in cols) * 3 + noise * 0.7
    elif kind == "threshold":  # axis-aligned XOR: easy for trees, hard for a linear model
        score = ((za > 0) ^ (zb > 0)).astype(float) * 3 + noise * 0.6
    else:
        raise ValueError(kind)
    y = pd.Series((score > np.median(score)).astype(int))
    return (y, [a, b]) if return_cols else y


# ----------------------------------------------------------------------------
# Stage 1: data understanding (planted problems)
# ----------------------------------------------------------------------------

DU_ACTIONS = ["proceed", "deduplicate_before_split", "use_imbalance_aware_evaluation",
              "impute_within_folds", "small_data_protocol"]
DU_CARD = (
    "Data understanding: profile the dataset and decide the first action before any split or model. "
    "Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | "
    "fewer than 500 rows or fewer than 20 rows per feature."
)
DU_GUIDE = (
    "proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | "
    "use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | "
    "impute_within_folds = fit imputation (plus missing indicators) inside training folds only | "
    "small_data_protocol = repeated stratified CV, simple models first, wide uncertainty"
)


def du_profile(X, y):
    miss = X.isna().mean().sort_values(ascending=False)
    top = [(c, float(v)) for c, v in miss.head(3).items()]
    return {
        "rows": len(X), "features": X.shape[1], "pos": float(y.mean()),
        "minority": float(min(y.mean(), 1 - y.mean())),
        "dup": float(pd.concat([X, y.rename("__y")], axis=1).duplicated().mean()),
        "top_missing": top, "max_missing": top[0][1] if top else 0.0,
        "rows_per_feature": len(X) / max(X.shape[1], 1),
        "near_constant": int(sum(X[c].value_counts(normalize=True, dropna=False).iloc[0] > 0.995 for c in X.columns)),
    }


def du_flags(p):
    return {
        "deduplicate_before_split": p["dup"] > 0.01,
        "use_imbalance_aware_evaluation": p["minority"] < 0.10,
        "impute_within_folds": p["max_missing"] > 0.05,
        "small_data_protocol": p["rows"] < 500 or p["rows_per_feature"] < 20,
    }


def du_body(p):
    miss = ", ".join(f"`{c}` {v:.1%}" for c, v in p["top_missing"])
    return (
        "PROFILE:\n"
        f"- rows: {p['rows']:,} | features: {p['features']} | rows per feature: {p['rows_per_feature']:.1f}\n"
        f"- positive rate: {p['pos']:.1%} (minority class {p['minority']:.1%})\n"
        f"- exact duplicate rows (features + target): {p['dup']:.1%}\n"
        f"- highest missing rates: {miss}\n"
        f"- near-constant columns (one value in >99.5% of rows): {p['near_constant']}"
    )


def du_why(action, p, kind):
    if action == "proceed":
        return (f"No threshold is crossed: duplicates {p['dup']:.1%}, minority class {p['minority']:.1%}, highest missing "
                f"rate {p['max_missing']:.1%}, {p['rows']:,} rows ({p['rows_per_feature']:.1f} per feature). The data is "
                f"ready for the leakage audit.")
    if action == "deduplicate_before_split":
        return (f"{p['dup']:.1%} of rows are exact duplicates, above the 1% threshold. If they are split randomly, copies "
                f"of the same row land in both training and test folds, so the test score partly measures memory. "
                f"Duplicates must be removed, or grouped so all copies stay in one fold, before any split.")
    if action == "use_imbalance_aware_evaluation":
        return (f"The minority class is only {p['minority']:.1%} of rows, below the 10% threshold. Accuracy and even ROC-AUC "
                f"can look good while the model misses most positives, so evaluation needs stratified folds, PR-AUC, and a "
                f"threshold chosen from the cost of each error.")
    if action == "impute_within_folds":
        c, v = p["top_missing"][0]
        return (f"`{c}` is missing in {v:.1%} of rows, above the 5% threshold. Filling it with statistics from the whole "
                f"dataset would leak test-fold information into training, so imputation (and a missing-value indicator, "
                f"since absence may carry meaning) must be fitted inside each training fold.")
    return (f"There are only {p['rows']:,} rows ({p['rows_per_feature']:.1f} per feature). Single splits on data this small "
            f"swing widely, so use repeated stratified CV, start with simple regularized models, and report wide "
            f"uncertainty instead of single scores.")


DU_RULES = {
    "proceed": "DCLAB-R01 (contract written, profile clean)",
    "deduplicate_before_split": "DCLAB-R04 (duplicate/entity leakage) and DCLAB-R02 (split by the independence unit)",
    "use_imbalance_aware_evaluation": "DCLAB-R18 (metrics and thresholds from the decision cost)",
    "impute_within_folds": "DCLAB-R03 (fit imputers on training partitions only) and DCLAB-R12 (missingness as a state)",
    "small_data_protocol": "DCLAB-R16 (paired, stable comparisons) and DCLAB-R13 (start from a transparent baseline)",
}
DU_NEXT = {
    "proceed": "Write the profile to the decision ledger and start the leakage audit.",
    "deduplicate_before_split": "Deduplicate (or define groups), re-profile, then start the leakage audit.",
    "use_imbalance_aware_evaluation": "Fix the evaluation plan (stratified folds, PR-AUC, cost-based threshold) before the leakage audit.",
    "impute_within_folds": "Add in-fold imputation and missing indicators to the pipeline, then start the leakage audit.",
    "small_data_protocol": "Switch to repeated stratified CV with a linear baseline, then start the leakage audit.",
}


def stage_data_understanding(key, variants, seed):
    rng = np.random.default_rng(seed)
    X0, y0, meta = load(key, 4000, seed)
    X0 = safe_base(key, X0)
    keep = ~pd.concat([X0, y0.rename("__y")], axis=1).duplicated()
    X0, y0 = X0[keep].reset_index(drop=True), y0[keep].reset_index(drop=True)
    samples, drops = [], Counter()
    plants = ["proceed", "deduplicate_before_split", "use_imbalance_aware_evaluation",
              "impute_within_folds", "small_data_protocol"]
    for _ in range(variants):
        for plant in plants:
            n = int(rng.integers(min(1500, len(X0)), len(X0) + 1))
            idx = rng.choice(len(X0), size=n, replace=False)
            X, y = X0.iloc[idx].reset_index(drop=True), y0.iloc[idx].reset_index(drop=True)
            if plant == "deduplicate_before_split":
                k = int(len(X) * rng.uniform(0.03, 0.2))
                d = rng.choice(len(X), size=k, replace=False)
                X, y = pd.concat([X, X.iloc[d]], ignore_index=True), pd.concat([y, y.iloc[d]], ignore_index=True)
            elif plant == "use_imbalance_aware_evaluation":
                pos = np.flatnonzero(y.to_numpy() == 1)
                target = rng.uniform(0.01, 0.07)
                n_pos = max(5, int(target * (len(y) - len(pos)) / (1 - target)))
                drop_pos = rng.choice(pos, size=max(0, len(pos) - n_pos), replace=False)
                X, y = X.drop(index=drop_pos).reset_index(drop=True), y.drop(index=drop_pos).reset_index(drop=True)
            elif plant == "impute_within_folds":
                X = X.copy()
                for c in rng.choice(X.columns, size=min(int(rng.integers(1, 4)), X.shape[1]), replace=False):
                    X.loc[rng.random(len(X)) < rng.uniform(0.12, 0.6), c] = np.nan
            elif plant == "small_data_protocol":
                m = int(rng.integers(120, 480))
                sub = rng.choice(len(X), size=min(m, len(X)), replace=False)
                X, y = X.iloc[sub].reset_index(drop=True), y.iloc[sub].reset_index(drop=True)
            p = du_profile(X, y)
            fired = [a for a, on in du_flags(p).items() if on]
            ok = (fired == [] if plant == "proceed" else fired == [plant])
            if not ok:
                drops[plant] += 1
                continue
            samples.append(make_sample(
                "data_understanding", DU_CARD, card(key, meta, X, y), du_body(p), DU_ACTIONS, DU_GUIDE,
                plant, du_why(plant, p, plant), DU_RULES[plant], "high", DU_NEXT[plant],
                {"dataset": key, "generator": f"du_{plant}", "source": "planted"}))
    return samples, drops


# ----------------------------------------------------------------------------
# Stage 3: feature engineering (paired comparison of recipes, rule R07/R16)
# ----------------------------------------------------------------------------

FE_ACTIONS = ["keep_raw", "adopt_log_transforms", "adopt_ratios", "adopt_interactions"]
FE_CARD = (
    "Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. "
    "Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. "
    "If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features."
)
FE_GUIDE = ("keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = "
            "adopt that recipe for model selection")
FE_TOL = 0.005


def fe_recipes(X, rng, force=()):
    cont = [c for c in X.columns if X[c].nunique() > 10]
    pick = list(rng.choice(cont, size=min(4, len(cont)), replace=False)) if len(cont) >= 2 else []
    pick = list(dict.fromkeys([c for c in force if c in cont] + pick))[:4]
    out = {"raw": X}
    skewed = [c for c in cont if X[c].min() >= 0 and abs(X[c].skew()) > 1]
    if skewed:
        out["log_transforms"] = X.assign(**{f"log1p_{c}": np.log1p(X[c]) for c in skewed})
    if len(pick) >= 2:
        pairs = [(a, b) for i, a in enumerate(pick) for b in pick[i + 1:]]
        out["ratios"] = X.assign(**{f"{a}_over_{b}": X[a] / (X[b].abs() + 1e-3) for a, b in pairs})
        z = {c: (X[c] - X[c].mean()) / (X[c].std() + 1e-9) for c in pick}
        out["interactions"] = X.assign(**{f"{a}_x_{b}": z[a] * z[b] for a, b in pairs})
    return out


def fe_decide(res):
    """Rule R07/R16 on paired fold results {recipe: {"folds", "n"}}; adds "gain"/"wins" to res."""
    raw = res["raw"]["folds"]
    rows, qualified = [], []
    for name, r in res.items():
        gain = r["folds"] - raw
        r["gain"], r["wins"] = float(gain.mean()), int((gain > 0).sum())
        gain_txt = "—" if name == "raw" else f"{r['gain']:+.4f}"
        wins_txt = "—" if name == "raw" else f"{r['wins']}/5"
        rows.append(f"| {name} | {r['n']} | {r['folds'].mean():.4f} | {r['folds'].std():.4f} | {gain_txt} | {wins_txt} |")
        if name != "raw" and r["gain"] >= FE_TOL and r["wins"] == 5:
            qualified.append(name)
    if qualified:
        qualified.sort(key=lambda n: -res[n]["gain"])
        best = qualified[0]
        if len(qualified) > 1 and res[best]["gain"] - res[qualified[1]]["gain"] <= 0.002 \
                and res[qualified[1]]["n"] < res[best]["n"]:
            best = qualified[1]
        action = f"adopt_{best}"
        r = res[best]
        why = (f"`{best}` improves on raw by {r['gain']:+.4f} mean paired ROC-AUC and wins on {r['wins']}/5 folds, "
               f"which clears the +0.005 bar consistently, not by luck on one fold. "
               + (f"Other recipes that also qualified gained less. " if len(qualified) > 1 else "")
               + f"It adds {r['n'] - res['raw']['n']} features with a recorded formula, so its lineage stays auditable.")
        rule, nxt = ("DCLAB-R07 (promote the smallest defensible set within tolerance) and DCLAB-R16 (paired margin)",
                     f"Carry the raw + {best} feature set into model selection and log each new feature's formula.")
    else:
        action, best = "keep_raw", "raw"
        cand = [n for n in res if n != "raw"]
        bestg = max(cand, key=lambda n: res[n]["gain"]) if cand else None
        why = ("No engineered recipe clears the bar. "
               + (f"The closest, `{bestg}`, changes ROC-AUC by {res[bestg]['gain']:+.4f} and wins on "
                  f"{res[bestg]['wins']}/5 folds, which is not a consistent gain of at least +0.005. " if bestg else "")
               + "Extra features without a real gain only add maintenance and drift risk, so the raw baseline stays.")
        rule, nxt = ("DCLAB-R07 (start from a strong raw baseline) and DCLAB-R16 (demand a predeclared margin)",
                     "Carry the raw feature set into model selection.")
    return {"rows": rows, "action": action, "recipe": best, "why": why, "rule": rule, "next": nxt}


def stage_feature_engineering(key, variants, seed, sims_only=False):
    rng = np.random.default_rng(seed)
    Xf, yf, meta = load(key, 3000, seed)
    Xf = safe_base(key, Xf)
    samples, drops = [], Counter()
    for v in range(variants):
        model = ["logistic_regression", "hist_gradient_boosting"][v % 2]
        if sims_only:
            model = ["logistic_regression", "logistic_regression", "hist_gradient_boosting"][v % 3]
            sim = ["interaction", "ratio", "log"][v % 3]
            if sim == "log" and not any(Xf[c].nunique() > 10 and Xf[c].min() >= 0 and abs(Xf[c].skew()) > 1
                                        for c in Xf.columns):
                sim = "interaction"  # no skewed column to build a log mechanism from
        else:
            sim = [None, None, "interaction", None][v % 4] if v >= 2 else None
        force = ()
        if sim:
            y, force = simulated_target(Xf, rng, sim, return_cols=True)
        else:
            y = yf
        note = ("Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; "
                "the mechanism is not disclosed.") if sim else None
        recipes = fe_recipes(Xf, rng, force)
        splits = folds(y, 5, seed + v)
        res = {}
        for name, Xr in recipes.items():
            a, t = fold_aucs(model, Xr, y, splits, seed)
            res[name] = {"folds": a, "n": Xr.shape[1], "t": t}
        d = fe_decide(res)
        rows, action, why, rule, nxt = d["rows"], d["action"], d["why"], d["rule"], d["next"]
        body = (f"MODEL USED FOR THE COMPARISON: {model} | 5 stratified folds, identical for every recipe\n"
                f"RESULTS:\n| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |\n"
                f"|---|---|---|---|---|---|\n" + "\n".join(rows))
        samples.append(make_sample("feature_engineering", FE_CARD, card(key, meta, Xf, y, note), body, FE_ACTIONS,
                                   FE_GUIDE, action, why, rule, "high", nxt,
                                   {"dataset": key, "generator": f"fe_{'sim_' + sim if sim else 'real'}_{model}",
                                    "source": "simulated_target" if sim else "real"}))
    return samples, drops


# ----------------------------------------------------------------------------
# Stage 4: model selection (rule R13/R14)
# ----------------------------------------------------------------------------

MS_ACTIONS = list(MODELS)
MS_CARD = (
    "Model selection: screen a linear baseline and several nonlinear families on identical folds. "
    "Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and "
    "one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep "
    "the higher score but mark the choice as low confidence."
)
MS_GUIDE = "choose the one model family to carry into optimization"


def ms_decide(res):
    """Rule R13/R14 on {model: {"mean", "std", "score", "t"}}."""
    order = sorted(res, key=lambda n: -res[n]["score"])
    best, second = order[0], order[1]
    tie = res[best]["score"] - res[second]["score"] <= 0.002
    fast_ratio = max(res[best]["t"], res[second]["t"]) / max(min(res[best]["t"], res[second]["t"]), 1e-3)
    speed_tie_break = tie and fast_ratio >= 1.5
    if speed_tie_break and res[second]["t"] < res[best]["t"]:
        best, second = second, best
    rows = [f"| {n} | {res[n]['mean']:.4f} | {res[n]['std']:.4f} | {res[n]['score']:.4f} | {res[n]['t']:.2f} |"
            for n in res]
    b, s_ = res[best], res[second]
    if speed_tie_break:
        why = (f"`{best}` and `{second}` are effectively tied on stability-adjusted score ({b['score']:.4f} vs "
               f"{s_['score']:.4f}, within 0.002), so the clearly faster one goes forward: `{best}` takes {b['t']:.2f}s vs "
               f"{s_['t']:.2f}s. A score gap this small is noise, while a speed gap this large is real.")
    elif tie:
        why = (f"`{best}` and `{second}` are effectively tied: stability-adjusted scores {b['score']:.4f} vs "
               f"{s_['score']:.4f} (within 0.002) and similar runtimes ({b['t']:.2f}s vs {s_['t']:.2f}s). Nothing here "
               f"separates them meaningfully, so `{best}` goes forward on its slightly higher score, as a low-confidence "
               f"choice that later evidence (calibration, stability on new folds) can overturn.")
    else:
        why = (f"`{best}` has the highest stability-adjusted score ({b['score']:.4f}: mean {b['mean']:.4f}, std "
               f"{b['std']:.4f}), ahead of `{second}` at {s_['score']:.4f}. "
               + ("The linear baseline wins here, which is exactly why it is always screened: no family is best on every dataset."
                  if best == "logistic_regression" else
                  f"The linear baseline scores {res['logistic_regression']['score']:.4f}, so the nonlinear family earns its extra complexity."))
    return {"rows": rows, "best": best, "second": second, "why": why, "tie": tie, "speed_tie_break": speed_tie_break,
            "confidence": "low" if (tie and not speed_tie_break) else "medium" if tie else "high",
            "next": f"Carry `{best}` into optimization; keep `{second}` as the paired comparison when tuning."}


def stage_model_selection(key, variants, seed):
    rng = np.random.default_rng(seed)
    Xf, yf, meta = load(key, 4000, seed)
    Xf = safe_base(key, Xf)
    samples, drops = [], Counter()
    for v in range(variants):
        size = [None, 800, 300, None, None, 1500][v % 6]
        sim = [None, None, None, "linear", "threshold", None][v % 6]
        X, y = Xf, yf
        if sim:
            y = simulated_target(Xf, rng, sim)
        if size and size < len(X):
            idx = rng.choice(len(X), size=size, replace=False)
            X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
        if y.nunique() < 2 or y.value_counts().min() < 10:
            drops["too_few_positives"] += 1
            continue
        splits = folds(y, 3, seed + v)
        res = {}
        for name in MODELS:
            a, t = fold_aucs(name, X, y, splits, seed)
            res[name] = {"mean": a.mean(), "std": a.std(), "score": a.mean() - 0.25 * a.std(), "t": t}
        d = ms_decide(res)
        rows, best, second, why, tie, speed_tie_break = d["rows"], d["best"], d["second"], d["why"], d["tie"], d["speed_tie_break"]
        note = None
        if sim:
            note = ("Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; "
                    "the mechanism is not disclosed.")
        body = (f"SCREEN: 3 stratified folds, identical for every model, same raw feature set\n"
                f"| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |\n|---|---|---|---|---|\n"
                + "\n".join(rows))
        samples.append(make_sample(
            "model_selection", MS_CARD, card(key, meta, X, y, note), body, MS_ACTIONS, MS_GUIDE, best, why,
            "DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)",
            d["confidence"],
            d["next"],
            {"dataset": key, "generator": f"ms_{'sim_' + sim if sim else 'real'}_{size or 'full'}",
             "source": "simulated_target" if sim else "real"}))
    return samples, drops


# ----------------------------------------------------------------------------
# Stage 5: optimization & reliability (calibration R14/R18; holdout check R17/R19)
# ----------------------------------------------------------------------------

OPT_CAL_ACTIONS = ["keep_uncalibrated", "calibrate_sigmoid", "calibrate_isotonic"]
OPT_HO_ACTIONS = ["accept_result", "investigate_distribution_shift", "investigate_gap_without_shift"]
OPT_CAL_CARD = (
    "Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw "
    "model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 "
    "AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the "
    "training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE."
)
OPT_HO_CARD = (
    "Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if "
    "|CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between "
    "training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift."
)


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def psi(a, b, bins=10):
    a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    if a.nunique() < 2 or len(b) < 20:
        return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ha = np.clip(np.histogram(a, edges)[0] / len(a), 1e-6, None)
    hb = np.clip(np.histogram(b, edges)[0] / len(b), 1e-6, None)
    return float(np.sum((ha - hb) * np.log(ha / hb)))


def cal_decide(st, model, train_rows):
    """Rules R14/R18 on {"raw"|"sigmoid"|"isotonic": {"ece", "auc", "brier"}}."""
    raw = st["raw"]
    ok = {m: raw["ece"] > 0.03 and st[m]["ece"] <= 0.75 * raw["ece"] and st[m]["auc"] >= raw["auc"] - 0.002
          for m in ("sigmoid", "isotonic")}
    if not any(ok.values()):
        action = "keep_uncalibrated"
        why = (f"Raw `{model}` probabilities are already usable or calibration does not pay: raw ECE is {raw['ece']:.4f}"
               + (" (at or below the 0.03 bar)" if raw["ece"] <= 0.03 else
                  f", and neither method cuts it by 25% without losing ROC-AUC (sigmoid {st['sigmoid']['ece']:.4f}, "
                  f"isotonic {st['isotonic']['ece']:.4f})")
               + ". Adding a calibration layer would add complexity for no measured benefit.")
    else:
        cands = [m for m in ("sigmoid", "isotonic") if ok[m]]
        pick = min(cands, key=lambda m: st[m]["ece"])
        if pick == "isotonic" and ok["sigmoid"] and (train_rows < 1000 or st["sigmoid"]["ece"] - st["isotonic"]["ece"] < 0.005):
            pick = "sigmoid"
        action = f"calibrate_{pick}"
        why = (f"Raw `{model}` probabilities are off by an ECE of {raw['ece']:.4f}, above the 0.03 bar. {pick.capitalize()} "
               f"calibration fitted inside the training folds brings ECE to {st[pick]['ece']:.4f} with ROC-AUC "
               f"{st[pick]['auc']:.4f} vs {raw['auc']:.4f} raw, so ranking is kept while probabilities become trustworthy. "
               + ((f"Sigmoid is preferred over isotonic because the training folds have only about {train_rows:,} rows "
                   f"(under 1,000), where isotonic tends to overfit." if train_rows < 1000 else
                   f"Sigmoid is preferred because isotonic is better by less than 0.005 ECE "
                   f"({st['isotonic']['ece']:.4f} vs {st['sigmoid']['ece']:.4f}), not enough to justify the more flexible method.")
                  if pick == "sigmoid" and ok["isotonic"] else ""))
    nxt = ("Lock the uncalibrated model and move to the final holdout check." if action == "keep_uncalibrated"
           else f"Add {action.split('_')[1]} calibration inside the pipeline, lock it, and move to the final holdout check.")
    return {"action": action, "why": why, "next": nxt}


HOLDOUT_NEXT = {
    "accept_result": "Report CV and holdout together with uncertainty, and do not tune anything on the holdout.",
    "investigate_distribution_shift": "Do not tune on the holdout. Slice results by the shifted feature, document the gap, and decide with the data owner.",
    "investigate_gap_without_shift": "Do not tune on the holdout. Put a bootstrap interval on the holdout score, review which earlier choices were made on CV, and report both numbers with the explanation.",
}


def holdout_decide(cv, hauc, psis, n_holdout):
    """Rules R17/R19: cv = fold AUCs on training, hauc = holdout AUC scored once, psis = [(column, psi)] sorted."""
    gap = cv.mean() - hauc
    if abs(gap) <= 0.02:
        action = "accept_result"
        why = (f"CV ROC-AUC {cv.mean():.4f} and holdout ROC-AUC {hauc:.4f} differ by {gap:+.4f}, inside the ±0.02 band, "
               f"so the CV estimate held up on untouched data. "
               + (f"`{psis[0][0]}` does shift (PSI {psis[0][1]:.3f}), so it goes on the monitoring list, but it did not hurt "
                  f"performance here." if psis[0][1] > 0.2 else "No feature shifted meaningfully between training and holdout."))
    elif psis[0][1] > 0.2:
        action = "investigate_distribution_shift"
        why = (f"The holdout ROC-AUC ({hauc:.4f}) is {abs(gap):.4f} {'below' if gap > 0 else 'above'} CV ({cv.mean():.4f}), "
               f"outside the ±0.02 band, and `{psis[0][0]}` has a PSI of {psis[0][1]:.3f} (above 0.2): the holdout "
               f"population is different from training. The CV number does not describe this population, so the shift "
               f"must be explained before any claim is made.")
    else:
        action = "investigate_gap_without_shift"
        why = (f"CV ({cv.mean():.4f}) and holdout ({hauc:.4f}) differ by {gap:+.4f}, outside the ±0.02 band, yet no feature "
               f"shifts much (highest PSI {psis[0][1]:.3f}). "
               + (f"The holdout scored higher than CV, which usually means an easier or luckier sample of "
                  f"{n_holdout:,} rows rather than a better model, so the higher number must not be reported as the expected performance."
                  if gap < 0 else
                  f"The holdout scored lower than CV without a population change, which points to choices fitted to CV noise "
                  f"(optimistic CV) or an unstable estimate on {n_holdout:,} rows, so the gap needs explaining before results are reported."))
    return {"action": action, "why": why, "next": HOLDOUT_NEXT[action]}


def stage_optimization(key, variants, seed):
    rng = np.random.default_rng(seed)
    Xf, yf, meta = load(key, 4000, seed)
    Xf = safe_base(key, Xf)
    samples, drops = [], Counter()
    # --- calibration samples
    for v in range(variants):
        model = list(CAL_MODELS)[v % 6]
        size = [None, 900][v // 6 % 2]
        X, y = Xf, yf
        if size and size < len(X):
            idx = rng.choice(len(X), size=size, replace=False)
            X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
        splits = folds(y, 3, seed + v)
        preds = {k: np.zeros(len(y)) for k in ("raw", "sigmoid", "isotonic")}
        for tr, te in splits:
            base = CAL_MODELS[model](seed)
            preds["raw"][te] = base.fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
            for meth in ("sigmoid", "isotonic"):
                c = CalibratedClassifierCV(CAL_MODELS[model](seed), method=meth, cv=3).fit(X.iloc[tr], y.iloc[tr])
                preds[meth][te] = c.predict_proba(X.iloc[te])[:, 1]
        st = {k: {"ece": ece(y, p), "auc": roc_auc_score(y, p),
                  "brier": float(np.mean((p - y.to_numpy()) ** 2))} for k, p in preds.items()}
        train_rows = int(len(y) * 2 / 3)
        d = cal_decide(st, model, train_rows)
        action, why = d["action"], d["why"]
        body = (f"MODEL: {model} | 3 stratified folds | training rows per fold: about {train_rows:,}\n"
                f"| probabilities | ECE | ROC-AUC | Brier |\n|---|---|---|---|\n"
                + "\n".join(f"| {k} | {s['ece']:.4f} | {s['auc']:.4f} | {s['brier']:.4f} |" for k, s in st.items()))
        samples.append(make_sample(
            "optimization_reliability", OPT_CAL_CARD, card(key, meta, X, y), body, OPT_CAL_ACTIONS,
            "keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic", action, why.strip(),
            "DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)",
            "high", d["next"],
            {"dataset": key, "generator": f"opt_calibration_{model}", "source": "real"}))
    # --- holdout samples
    cont = [c for c in Xf.columns if Xf[c].nunique() > 10] or list(Xf.columns)
    for v in range(variants):
        shifted = v % 2 == 1
        if shifted:
            col = rng.choice(cont)
            q = Xf[col].rank(method="first", pct=True)
            ho = (q > 0.75).to_numpy() if rng.random() < 0.5 else (q <= 0.25).to_numpy()
        else:
            ho = rng.random(len(Xf)) < 0.25
        Xtr, ytr, Xho, yho = Xf[~ho].reset_index(drop=True), yf[~ho].reset_index(drop=True), Xf[ho], yf[ho]
        if yho.nunique() < 2 or yho.value_counts().min() < 10:
            drops["holdout_one_class"] += 1
            continue
        model = "hist_gradient_boosting"
        cv, _ = fold_aucs(model, Xtr, ytr, folds(ytr, 3, seed + v), seed)
        m = MODELS[model](seed).fit(Xtr, ytr)
        hauc = roc_auc_score(yho, m.predict_proba(Xho)[:, 1])
        psis = sorted(((c, psi(Xtr[c], Xho[c])) for c in Xf.columns), key=lambda t: -t[1])[:3]
        gap = cv.mean() - hauc
        d = holdout_decide(cv, hauc, psis, len(yho))
        action, why = d["action"], d["why"]
        body = (f"MODEL: {model} (locked) | training rows: {len(Xtr):,} | holdout rows: {len(Xho):,} "
                f"(holdout positive rate {yho.mean():.1%})\n"
                f"- 3-fold CV ROC-AUC on training: {cv.mean():.4f} (std {cv.std():.4f})\n"
                f"- holdout ROC-AUC (scored once): {hauc:.4f}\n"
                f"- difference, CV minus holdout: {gap:+.4f}\n"
                f"- largest feature shifts, training vs holdout (PSI): "
                + ", ".join(f"`{c}` {p:.3f}" for c, p in psis))
        samples.append(make_sample(
            "optimization_reliability", OPT_HO_CARD, card(key, meta, Xf, yf), body, OPT_HO_ACTIONS,
            "accept_result | investigate_distribution_shift | investigate_gap_without_shift", action, why,
            "DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)", "high",
            d["next"],
            {"dataset": key, "generator": f"opt_holdout_{'shifted' if shifted else 'random'}", "source": "real"}))
    return samples, drops


STAGES = {
    "data_understanding": stage_data_understanding,
    "feature_engineering": stage_feature_engineering,
    "feature_engineering_balance": lambda k, v, s: stage_feature_engineering(k, v, s + 500, sims_only=True),
    "model_selection": stage_model_selection,
    "optimization_reliability": stage_optimization,
}
ALL_ACTIONS = set(DU_ACTIONS + FE_ACTIONS + MS_ACTIONS + OPT_CAL_ACTIONS + OPT_HO_ACTIONS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="*", default=list(STAGES))
    ap.add_argument("--datasets", nargs="*", default=list(TARGETS))
    ap.add_argument("--variants", type=int, default=12)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--val-datasets", nargs="*", default=["wine_quality"])
    ap.add_argument("--test-datasets", nargs="*", default=["online_shoppers", "german_credit"])
    ap.add_argument("--out", type=Path, default=HERE / "out")
    args = ap.parse_args()
    split_of = {**{k: "val" for k in args.val_datasets}, **{k: "test" for k in args.test_datasets}}

    for stage in args.stages:
        t0, rows, drops, seen = time.time(), [], Counter(), set()
        for i, key in enumerate(args.datasets):
            t = time.time()
            s, d = STAGES[stage](key, args.variants, args.seed + i)
            drops.update(d)
            for x in s:
                h = hashlib.sha256(x["messages"][1]["content"].encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    rows.append(x)
            print(f"{stage:26s} {key:16s} {len(s):4d} samples  {time.time() - t:6.1f}s", flush=True)
        splits = {"train": [], "val": [], "test": []}
        for x in rows:
            splits[split_of.get(x["metadata"]["dataset"], "train")].append(x)
        random.Random(args.seed).shuffle(splits["train"])
        out = args.out / stage
        out.mkdir(parents=True, exist_ok=True)
        for name, part in splits.items():
            with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
        manifest = {"stage": stage, "total": len(rows), "splits": {k: len(v) for k, v in splits.items()},
                    "by_action": dict(Counter(r["metadata"]["action"] for r in rows)),
                    "by_generator": dict(Counter(r["metadata"]["generator"] for r in rows)),
                    "dropped_by_checker": dict(drops), "seconds": round(time.time() - t0)}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
