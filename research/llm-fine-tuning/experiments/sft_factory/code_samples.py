"""
code_samples.py
===============

Code samples for the workflow agent, so fine-tuning keeps (and points) its coding skill instead of wearing it down.
Two families, both VERIFIED by running the code here on the real data:

1. code_then_decide (three turns): the step asks for the evidence; the assistant writes the pandas / scikit-learn
   code; the code is EXECUTED and its real output comes back; the assistant decides with the step's rule.
   Steps covered: WF-03 pre-split profile, WF-06 feature ladder, WF-07 algorithm screen, WF-08 calibration,
   WF-09 holdout check. The decision is computed by the same rule functions as the single-step stages
   (workflow_samples.du_* / fe_decide / ms_decide / cal_decide / holdout_decide) from the numbers the code printed,
   so the code, its output and the decision always agree.

2. code_review (one turn): a snippet with a planted leakage bug (preprocessing fitted on all rows, selection or
   oversampling before cross-validation, target encoding on all rows, tuning on the test set, deduplicating after
   the split, a random split on time-ordered rows), or the same snippet written correctly. The answer says
   fix_leak or no_issue, why, the rule, and, for a bug, the fixed code. Both versions are executed; a sample is kept
   only if both run.

Every sample carries its WF step header and its route, like the rest of the factory.

Usage
-----
    python research/llm-fine-tuning/experiments/sft_factory/code_samples.py
    python research/llm-fine-tuning/experiments/sft_factory/code_samples.py --datasets heart_disease --variants 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

import numpy as np

import workflow_map as M
import workflow_samples as W
from leakage_audit_samples import HERE, ROOT, SYSTEM_PROMPT, TARGETS
from problem_to_plan_samples import J, numbers_ok
from splits import split_of

PUBLIC = list(TARGETS)
RUN_TIMEOUT = 600

CODE_CARD = ("Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named "
             "below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the "
             "result as JSON. The code will be run and its output returned to you.")
REVIEW_CARD = ("Code review: read the snippet and decide whether it leaks information across the train/test boundary "
               "or tunes on data it later reports on. fix_leak = it does; give the fixed code. no_issue = it does not.")
REVIEW_ACTIONS = ["fix_leak", "no_issue"]


def run_code(code: str) -> tuple[bool, str]:
    try:
        p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    return p.returncode == 0, (p.stdout if p.returncode == 0 else p.stderr[-2000:])


def load_lines(key: str, n: int, seed: int) -> str:
    return textwrap.dedent(f"""\
        X = pd.read_parquet("data/public/{key}/X.parquet")
        y = pd.read_parquet("data/public/{key}/y.parquet").iloc[:, 0].astype(int)
        if len(X) > {n}:  # a stratified sample keeps the class balance
            keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac={n} / len(X), random_state={seed})).index
            X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
        """)


def py_block(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def sample(turns, meta):
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + [{"role": r, "content": c} for r, c in turns],
            "metadata": {**meta, "routed": True}}


# ============================================================================ 1. code, then decide

ECE_FN = textwrap.dedent("""\
    def ece(y, p, bins=10):
        y, p = np.asarray(y), np.asarray(p)
        edges = np.linspace(0, 1, bins + 1)
        total = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
            if m.any():
                total += m.mean() * abs(y[m].mean() - p[m].mean())
        return float(total)
    """)

PSI_FN = textwrap.dedent("""\
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
    """)

MODEL_DEFS = {
    "logistic_regression": '    "logistic_regression": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000)),',
    "random_forest": '    "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(150, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),',
    "hist_gradient_boosting": '    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=120, random_state=SEED),',
}
MODEL_IMPORTS = {
    "logistic_regression": ["from sklearn.impute import SimpleImputer", "from sklearn.linear_model import LogisticRegression",
                            "from sklearn.pipeline import make_pipeline", "from sklearn.preprocessing import StandardScaler"],
    "random_forest": ["from sklearn.ensemble import RandomForestClassifier", "from sklearn.impute import SimpleImputer",
                      "from sklearn.pipeline import make_pipeline"],
    "hist_gradient_boosting": ["from sklearn.ensemble import HistGradientBoostingClassifier"],
}
STD = ["import json", "import time"]
THIRD = ["import numpy as np", "import pandas as pd"]


def header(models=(), std=("json",), extra=(), seed=None, third=("numpy", "pandas")):
    """Only the imports a snippet uses, grouped the usual way, then SEED and the MODELS it needs."""
    lines = sorted(f"import {m}" for m in std) + [""] + [l for l in THIRD if l.split()[1] in third]
    sk = sorted({l for m in models for l in MODEL_IMPORTS[m]} | set(extra))
    merged = {}
    for l in sk:  # merge "from x import a" lines that share a module
        mod, names = l[len("from "):].split(" import ")
        merged.setdefault(mod, []).extend(n.strip() for n in names.split(","))
    lines += [f"from {mod} import {', '.join(sorted(set(ns)))}" for mod, ns in sorted(merged.items())]
    out = "\n".join(lines) + "\n\n"
    if seed is not None:
        out += f"SEED = {seed}\n"
    if models:
        out += "MODELS = {\n" + "\n".join(MODEL_DEFS[m] for m in models) + "\n}\n"
    return out + "\n"


def drop_blocked(blocked) -> str:
    return f"X = X.drop(columns={blocked!r})  # the dataset contract blocks these\n\n" if blocked else ""


def task_presplit(key, seed, rng):
    ask = (f"Profile `data/public/{key}` (X.parquet, y.parquet; target: {TARGETS[key]}) BEFORE any split: rows, "
           f"features, rows per feature, positive rate and minority share, exact duplicate rows (features + target), "
           f"the three highest missing rates, and near-constant columns (one value in more than 99.5% of rows).")
    code = header(third=("pandas",)) + textwrap.dedent(f"""\
        X = pd.read_parquet("data/public/{key}/X.parquet")
        y = pd.read_parquet("data/public/{key}/y.parquet").iloc[:, 0].astype(int)

        missing = X.isna().mean().sort_values(ascending=False).head(3)
        report = {{
            "rows": len(X),
            "features": X.shape[1],
            "rows_per_feature": round(len(X) / X.shape[1], 1),
            "positive_rate": round(float(y.mean()), 4),
            "minority_share": round(float(min(y.mean(), 1 - y.mean())), 4),
            "duplicate_share": round(float(pd.concat([X, y.rename("__y")], axis=1).duplicated().mean()), 4),
            "top_missing": {{c: round(float(v), 4) for c, v in missing.items()}},
            "near_constant_columns": int(sum(X[c].value_counts(normalize=True, dropna=False).iloc[0] > 0.995 for c in X.columns)),
        }}
        print(json.dumps(report, indent=1))
        """)

    def decide(out):
        p = {"rows": out["rows"], "features": out["features"], "pos": out["positive_rate"], "minority": out["minority_share"],
             "dup": out["duplicate_share"], "top_missing": list(out["top_missing"].items()),
             "max_missing": max(out["top_missing"].values(), default=0.0), "rows_per_feature": out["rows_per_feature"],
             "near_constant": out["near_constant_columns"]}
        fired = [a for a in ["deduplicate_before_split", "small_data_protocol", "use_imbalance_aware_evaluation",
                             "impute_within_folds"] if W.du_flags(p)[a]]
        action = fired[0] if fired else "proceed"
        why = W.du_why(action, p, action)
        return ("WF-03", "data_understanding", W.DU_CARD + " If several thresholds are crossed, act on the first in this order: "
                "deduplicate_before_split, small_data_protocol, use_imbalance_aware_evaluation, impute_within_folds.",
                W.DU_ACTIONS, action, why, W.DU_RULES[action], "high", W.DU_NEXT[action], M.route("continue", "WF-04"))
    return "WF-03", ask, code, decide


def task_feature_ladder(key, seed, rng):
    from leakage_audit_samples import load
    X, y, _ = load(key, 3000, seed)
    X = W.safe_base(key, X)
    cont = [c for c in X.columns if X[c].nunique() > 10]
    if len(cont) < 2:
        return None
    pick = list(rng.choice(cont, size=min(3, len(cont)), replace=False))
    model = ["logistic_regression", "hist_gradient_boosting"][seed % 2]
    ask = (f"Feature ladder for `data/public/{key}` (target: {TARGETS[key]}; use a stratified sample of 3,000 rows, "
           f"seed {seed}, and drop the columns the dataset's contract blocks: {', '.join(W.CATALOG[key].get('blocked', [])) or 'none'}). "
           f"Compare the raw columns with three recipes on the SAME 5 stratified folds and the same model ({model}): "
           f"log1p of non-negative skewed columns (|skew| > 1, more than 10 distinct values), pairwise ratios and pairwise "
           f"products of standardized values of {', '.join(f'`{c}`' for c in pick)}. For each recipe report the feature count, "
           f"mean and std ROC-AUC, the mean paired gain over raw and how many folds improved.")
    blocked = W.CATALOG[key].get("blocked", [])
    code = header((model,), extra=("from sklearn.metrics import roc_auc_score", "from sklearn.model_selection import StratifiedKFold"),
                  seed=seed) + load_lines(key, 3000, seed) + drop_blocked(blocked) + textwrap.dedent(f"""\
        pick = {pick!r}
        cont = [c for c in X.columns if X[c].nunique() > 10]
        skewed = [c for c in cont if X[c].min() >= 0 and abs(X[c].skew()) > 1]
        pairs = [(a, b) for i, a in enumerate(pick) for b in pick[i + 1:]]

        def recipes(frame, stats):
            out = {{"raw": frame}}
            if skewed:
                out["log_transforms"] = frame.assign(**{{f"log1p_{{c}}": np.log1p(frame[c]) for c in skewed}})
            out["ratios"] = frame.assign(**{{f"{{a}}_over_{{b}}": frame[a] / (frame[b].abs() + 1e-3) for a, b in pairs}})
            z = {{c: (frame[c] - stats[c][0]) / (stats[c][1] + 1e-9) for c in pick}}
            out["interactions"] = frame.assign(**{{f"{{a}}_x_{{b}}": z[a] * z[b] for a, b in pairs}})
            return out

        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
        aucs, n_features = {{}}, {{}}
        for tr, te in folds:
            stats = {{c: (X.iloc[tr][c].mean(), X.iloc[tr][c].std()) for c in pick}}  # training fold only
            train_sets, test_sets = recipes(X.iloc[tr], stats), recipes(X.iloc[te], stats)
            for name in train_sets:
                n_features[name] = train_sets[name].shape[1]
                m = MODELS["{model}"]().fit(train_sets[name], y.iloc[tr])
                aucs.setdefault(name, []).append(roc_auc_score(y.iloc[te], m.predict_proba(test_sets[name])[:, 1]))

        raw = np.array(aucs["raw"])
        report = {{}}
        for name, a in aucs.items():
            a = np.array(a)
            report[name] = {{"features": n_features[name],
                            "mean_roc_auc": round(float(a.mean()), 6), "std": round(float(a.std()), 6),
                            "mean_paired_gain": None if name == "raw" else round(float((a - raw).mean()), 6),
                            "folds_improved": None if name == "raw" else int(((a - raw) > 0).sum())}}
        print(json.dumps(report, indent=1))
        """)

    def decide(out):
        res = {}
        for name, r in out.items():
            mean, gain = r["mean_roc_auc"], r["mean_paired_gain"] or 0.0
            # rebuild a fold vector with the printed mean gain and wins, so fe_decide applies the same rule
            res[name] = {"folds": None, "n": r["features"], "mean": mean, "std": r["std"], "gain": gain,
                         "wins": r["folds_improved"] or 0}
        d = fe_decide_printed(res)
        return ("WF-06", "feature_engineering", W.FE_CARD, W.FE_ACTIONS, d["action"], d["why"], d["rule"], "high",
                d["next"], M.route("continue", "WF-07"))
    return "WF-06", ask, code, decide


def fe_decide_printed(res):
    """workflow_samples.fe_decide on printed summaries (mean, std, paired gain, folds improved)."""
    qualified = [n for n in res if n != "raw" and res[n]["gain"] >= W.FE_TOL and res[n]["wins"] == 5]
    if qualified:
        qualified.sort(key=lambda n: -res[n]["gain"])
        best = qualified[0]
        if len(qualified) > 1 and res[best]["gain"] - res[qualified[1]]["gain"] <= 0.002 and res[qualified[1]]["n"] < res[best]["n"]:
            best = qualified[1]
        r = res[best]
        why = (f"`{best}` improves on raw by {r['gain']:+.4f} mean paired ROC-AUC and wins on {r['wins']}/5 folds, "
               f"which clears the +0.005 bar consistently, not by luck on one fold. "
               + ("Other recipes that also qualified gained less. " if len(qualified) > 1 else "")
               + f"It adds {r['n'] - res['raw']['n']} features with a recorded formula, so its lineage stays auditable.")
        return {"action": f"adopt_{best}", "why": why,
                "rule": "DCLAB-R07 (promote the smallest defensible set within tolerance) and DCLAB-R16 (paired margin)",
                "next": f"Carry the raw + {best} feature set into model selection and log each new feature's formula."}
    cand = [n for n in res if n != "raw"]
    bestg = max(cand, key=lambda n: res[n]["gain"]) if cand else None
    why = ("No engineered recipe clears the bar. "
           + (f"The closest, `{bestg}`, changes ROC-AUC by {res[bestg]['gain']:+.4f} and wins on "
              f"{res[bestg]['wins']}/5 folds, which is not a consistent gain of at least +0.005. " if bestg else "")
           + "Extra features without a real gain only add maintenance and drift risk, so the raw baseline stays.")
    return {"action": "keep_raw", "why": why,
            "rule": "DCLAB-R07 (start from a strong raw baseline) and DCLAB-R16 (demand a predeclared margin)",
            "next": "Carry the raw feature set into model selection."}


def task_screen(key, seed, rng):
    blocked = W.CATALOG[key].get("blocked", [])
    ask = (f"Algorithm screen for `data/public/{key}` (target: {TARGETS[key]}; stratified sample of 3,000 rows, seed "
           f"{seed}; drop the contract's blocked columns: {', '.join(blocked) or 'none'}): logistic_regression, "
           f"random_forest and hist_gradient_boosting on the SAME 3 stratified folds. Report mean and std ROC-AUC, the "
           f"stability-adjusted score (mean − 0.25 × std) and fit+predict seconds.")
    code = header(tuple(MODEL_DEFS), std=("json", "time"), extra=("from sklearn.metrics import roc_auc_score",
                  "from sklearn.model_selection import StratifiedKFold"), seed=seed) + load_lines(key, 3000, seed) + \
        drop_blocked(blocked) + textwrap.dedent(f"""\
        folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y))
        report = {{}}
        for name, make in MODELS.items():
            start, aucs = time.time(), []
            for tr, te in folds:
                m = make().fit(X.iloc[tr], y.iloc[tr])
                aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
            aucs = np.array(aucs)
            report[name] = {{"mean_roc_auc": round(float(aucs.mean()), 6), "std": round(float(aucs.std()), 6),
                            "stability_adjusted": round(float(aucs.mean() - 0.25 * aucs.std()), 6),
                            "seconds": round(time.time() - start, 2)}}
        print(json.dumps(report, indent=1))
        """)

    def decide(out):
        res = {n: {"mean": r["mean_roc_auc"], "std": r["std"], "score": r["stability_adjusted"], "t": r["seconds"]}
               for n, r in out.items()}
        d = W.ms_decide(res)
        return ("WF-07", "model_selection", W.MS_CARD, list(res), d["best"], d["why"],
                "DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)",
                d["confidence"], d["next"], M.route("continue", "WF-08"))
    return "WF-07", ask, code, decide


def task_calibration(key, seed, rng):
    blocked = W.CATALOG[key].get("blocked", [])
    model = ["logistic_regression", "random_forest", "hist_gradient_boosting"][seed % 3]
    ask = (f"Calibration check for `{model}` on `data/public/{key}` (target: {TARGETS[key]}; stratified sample of 3,000 "
           f"rows, seed {seed}; drop the contract's blocked columns: {', '.join(blocked) or 'none'}): out-of-fold "
           f"probabilities on 3 stratified folds, raw and with sigmoid and isotonic calibration fitted inside each "
           f"training fold. Report ECE (10 bins), ROC-AUC and Brier for each, and the training rows per fold.")
    code = header((model,), extra=("from sklearn.calibration import CalibratedClassifierCV", "from sklearn.metrics import roc_auc_score",
                  "from sklearn.model_selection import StratifiedKFold"), seed=seed) + "\n" + ECE_FN + "\n\n" + \
        load_lines(key, 3000, seed) + drop_blocked(blocked) + textwrap.dedent(f"""\
        folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y))
        preds = {{k: np.zeros(len(y)) for k in ("raw", "sigmoid", "isotonic")}}
        for tr, te in folds:
            preds["raw"][te] = MODELS["{model}"]().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
            for method in ("sigmoid", "isotonic"):  # calibration learns only from the training fold
                cal = CalibratedClassifierCV(MODELS["{model}"](), method=method, cv=3).fit(X.iloc[tr], y.iloc[tr])
                preds[method][te] = cal.predict_proba(X.iloc[te])[:, 1]

        report = {{"training_rows_per_fold": int(len(y) * 2 / 3)}}
        for k, p in preds.items():
            report[k] = {{"ece": round(ece(y, p), 6), "roc_auc": round(float(roc_auc_score(y, p)), 6),
                         "brier": round(float(np.mean((p - y.to_numpy()) ** 2)), 6)}}
        print(json.dumps(report, indent=1))
        """)

    def decide(out):
        st = {k: {"ece": out[k]["ece"], "auc": out[k]["roc_auc"], "brier": out[k]["brier"]} for k in ("raw", "sigmoid", "isotonic")}
        d = W.cal_decide(st, model, out["training_rows_per_fold"])
        return ("WF-08", "optimization_reliability", W.OPT_CAL_CARD, W.OPT_CAL_ACTIONS, d["action"], d["why"].strip(),
                "DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)",
                "high", d["next"], M.route("continue", "WF-09"))
    return "WF-08", ask, code, decide


def task_holdout(key, seed, rng):
    blocked = W.CATALOG[key].get("blocked", [])
    from leakage_audit_samples import load
    X, _, _ = load(key, 3000, seed)
    cont = [c for c in W.safe_base(key, X).columns if X[c].nunique() > 10]
    shifted = seed % 2 == 1 and cont
    col = str(rng.choice(cont)) if shifted else None
    how = (f"the holdout is the top quarter of rows by `{col}` (a deliberately shifted holdout)" if shifted else
           "the holdout is a random quarter of the rows")
    ask = (f"Final holdout check for hist_gradient_boosting on `data/public/{key}` (target: {TARGETS[key]}; stratified "
           f"sample of 3,000 rows, seed {seed}; drop the contract's blocked columns: {', '.join(blocked) or 'none'}); {how}. "
           f"Report 3-fold CV ROC-AUC on the training part (mean, std), the holdout ROC-AUC scored once, the difference "
           f"CV minus holdout and its absolute value, and the three largest feature shifts (PSI) between training and holdout.")
    split_line = (f'holdout = (X["{col}"].rank(method="first", pct=True) > 0.75).to_numpy()' if shifted else
                  "holdout = np.random.default_rng(SEED).random(len(X)) < 0.25")
    code = header(("hist_gradient_boosting",), extra=("from sklearn.metrics import roc_auc_score",
                  "from sklearn.model_selection import StratifiedKFold"), seed=seed) + "\n" + PSI_FN + "\n\n" + \
        load_lines(key, 3000, seed) + drop_blocked(blocked) + textwrap.dedent(f"""\
        {split_line}
        Xtr, ytr = X[~holdout].reset_index(drop=True), y[~holdout].reset_index(drop=True)
        Xho, yho = X[holdout], y[holdout]

        cv = []
        for tr, te in StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(Xtr, ytr):
            m = MODELS["hist_gradient_boosting"]().fit(Xtr.iloc[tr], ytr.iloc[tr])
            cv.append(roc_auc_score(ytr.iloc[te], m.predict_proba(Xtr.iloc[te])[:, 1]))
        cv = np.array(cv)
        final = MODELS["hist_gradient_boosting"]().fit(Xtr, ytr)  # locked: every choice is made before this line
        holdout_auc = roc_auc_score(yho, final.predict_proba(Xho)[:, 1])  # scored once
        shifts = sorted(((c, psi(Xtr[c], Xho[c])) for c in X.columns), key=lambda t: -t[1])[:3]
        print(json.dumps({{
            "training_rows": len(ytr), "holdout_rows": int(len(yho)),
            "cv_roc_auc_mean": round(float(cv.mean()), 6), "cv_roc_auc_std": round(float(cv.std()), 6),
            "holdout_roc_auc": round(float(holdout_auc), 6),
            "cv_minus_holdout": round(float(cv.mean() - holdout_auc), 6),
            "abs_difference": round(float(abs(cv.mean() - holdout_auc)), 6),
            "largest_psi": {{c: round(p, 4) for c, p in shifts}},
        }}, indent=1))
        """)

    def decide(out):
        cv = np.array([out["cv_roc_auc_mean"]])  # holdout_decide reads the mean only
        psis = list(out["largest_psi"].items())
        d = W.holdout_decide(cv, out["holdout_roc_auc"], psis, out["holdout_rows"])
        return ("WF-09", "optimization_reliability", W.OPT_HO_CARD, W.OPT_HO_ACTIONS, d["action"], d["why"],
                "DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)", "high", d["next"],
                M.single_route("holdout", d["action"]))
    return "WF-09", ask, code, decide


TASKS = {"presplit": task_presplit, "feature_ladder": task_feature_ladder, "screen": task_screen,
         "calibration": task_calibration, "holdout": task_holdout}


def code_then_decide(key, task, seed, report):
    rng = np.random.default_rng(int(hashlib.sha256(f"{key}{task}{seed}".encode()).hexdigest()[:8], 16))
    got = TASKS[task](key, seed, rng)
    if got is None:
        report["dropped"]["task_not_applicable"] += 1
        return None
    step, ask, code, decide = got
    ok, out = run_code(code)
    if not ok:
        report["dropped"]["code_failed"] += 1
        report.setdefault("failures", []).append({"dataset": key, "task": task, "error": out[-400:]})
        return None
    printed = json.loads(out)
    _, stage, card, actions, action, why, rule, conf, nxt, r = decide(printed)
    u1 = M.header(step) + f"STAGE: {stage} (code)\n{CODE_CARD}\n\nTASK: {ask}"
    a1 = py_block(code)
    u2 = (M.header(step) + f"EXECUTION OUTPUT:\n```json\n{out.strip()}\n```\n\nNow decide.\n{card}\n\n"
          f"ALLOWED ACTIONS: {' | '.join(actions)}")
    a2 = M.with_route_text(f"DECISION: {action}\nWHY: {why}\nRULE: {rule}\nCONFIDENCE: {conf}\nNEXT: {nxt}", r)
    assert action in actions
    bad = numbers_ok(a2, u1 + "\n" + u2)
    if bad:
        report["dropped"]["number_not_in_input"] += 1
        report.setdefault("number_examples", []).append({"dataset": key, "task": task, "numbers": bad[:3]})
        return None
    return sample([("user", u1), ("assistant", a1), ("user", u2), ("assistant", a2)],
                  {"dataset": key, "scenario": key, "stage": "code_then_decide", "generator": f"code_{task}", "action": action,
                   "step": step, "route": r["action"], "source": "executed"})


# ============================================================================ 2. code review

def _header(key, n, seed, extra=""):
    lines = ["import json", "", "import numpy as np", "import pandas as pd",
             "from sklearn.impute import SimpleImputer",
             "from sklearn.linear_model import LogisticRegression",
             "from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split",
             "from sklearn.pipeline import make_pipeline",
             "from sklearn.preprocessing import StandardScaler"]
    lines += [x for x in extra.split("\n") if x]
    return "\n".join(lines) + "\n\n" + load_lines(key, n, seed)


REVIEW = {}


def review(name, step, rule, why, nxt):
    def deco(fn):
        REVIEW[name] = dict(fn=fn, step=step, rule=rule, why=why, next=nxt)
        return fn
    return deco


@review("scaler_before_cv", "WF-06", "DCLAB-R03 (fit every preprocessing step inside training folds only)",
        "`StandardScaler` is fitted on all rows before cross-validation, so every test fold's mean and spread have "
        "already shaped the training data. The effect is often small, but the score is no longer an honest estimate, "
        "and the same habit with stronger steps (imputation, selection, encoding) leaks much more.",
        "Re-run the comparison with the scaler inside the pipeline, so each fold fits it on its own training rows.")
def _scaler(key, seed, bug):
    h = _header(key, 3000, seed)
    if bug:
        body = """\
            X_scaled = StandardScaler().fit_transform(X.fillna(X.median()))
            scores = cross_val_score(LogisticRegression(max_iter=3000), X_scaled, y,
                                     cv=StratifiedKFold(5, shuffle=True, random_state={seed}), scoring="roc_auc")
            print(json.dumps({{"cv_roc_auc": round(scores.mean(), 4)}}))
            """
    else:
        body = """\
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000))
            scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state={seed}), scoring="roc_auc")
            print(json.dumps({{"cv_roc_auc": round(scores.mean(), 4)}}))
            """
    return h + textwrap.dedent(body).format(seed=seed)


@review("imputer_before_cv", "WF-06", "DCLAB-R03 (fit imputers on training partitions only) and DCLAB-R12 (missingness as a state)",
        "The median used to fill missing values is computed on all rows, test folds included, before the folds are "
        "made. Imputation learns from data, so it must be fitted inside each training fold; a missing-value indicator "
        "keeps the information that a value was absent.",
        "Move the imputer (with an indicator) into the pipeline and re-run the cross-validation.")
def _imputer(key, seed, bug):
    h = _header(key, 3000, seed)
    if bug:
        body = """\
            X_filled = X.fillna(X.median())
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            scores = cross_val_score(model, X_filled, y, cv=StratifiedKFold(5, shuffle=True, random_state={seed}), scoring="roc_auc")
            print(json.dumps({{"cv_roc_auc": round(scores.mean(), 4)}}))
            """
    else:
        body = """\
            model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(max_iter=3000))
            scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state={seed}), scoring="roc_auc")
            print(json.dumps({{"cv_roc_auc": round(scores.mean(), 4)}}))
            """
    return h + textwrap.dedent(body).format(seed=seed)


@review("selection_before_cv", "WF-06", "DCLAB-R03 (selection inside training folds only) and DCLAB-R07 (the feature ladder decides on paired folds)",
        "`SelectKBest` ranks columns by their link to `y` using all rows, then the model is cross-validated on the "
        "chosen columns. The test folds already voted on which columns survive, so the score is optimistic, and with "
        "many weak or noisy columns it can be far too high.",
        "Put the selector inside the pipeline so each fold selects on its own training rows, then compare again.")
def _selection(key, seed, bug):
    h = _header(key, 3000, seed, "from sklearn.feature_selection import SelectKBest, f_classif")
    k = "min(10, X.shape[1])"
    if bug:
        body = f"""\
            X_filled = X.fillna(X.median())
            X_best = SelectKBest(f_classif, k={k}).fit_transform(X_filled, y)
            scores = cross_val_score(LogisticRegression(max_iter=3000), X_best, y,
                                     cv=StratifiedKFold(5, shuffle=True, random_state={{seed}}), scoring="roc_auc")
            print(json.dumps({{{{"cv_roc_auc": round(scores.mean(), 4)}}}}))
            """
    else:
        body = f"""\
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SelectKBest(f_classif, k={k}),
                                  LogisticRegression(max_iter=3000))
            scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state={{seed}}), scoring="roc_auc")
            print(json.dumps({{{{"cv_roc_auc": round(scores.mean(), 4)}}}}))
            """
    return h + textwrap.dedent(body).format(seed=seed)


@review("oversample_before_cv", "WF-06", "DCLAB-R03 (resampling inside training folds only) and DCLAB-R18 (evaluate on the real class balance)",
        "Minority rows are copied before the folds are made, so copies of the same row land in both training and test "
        "folds and the model is tested on rows it memorised. The test folds also no longer have the real class "
        "balance, so the score describes neither the model nor production.",
        "Oversample only the training rows inside each fold, keep every test fold untouched, and re-run.")
def _oversample(key, seed, bug):
    h = _header(key, 3000, seed, "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import roc_auc_score")
    if bug:
        body = """\
            rng = np.random.default_rng({seed})
            minority = int(y.value_counts().idxmin())
            extra = rng.choice(np.flatnonzero(y.to_numpy() == minority), size=int((y != minority).sum() - (y == minority).sum()))
            X_bal = pd.concat([X, X.iloc[extra]], ignore_index=True)
            y_bal = pd.concat([y, y.iloc[extra]], ignore_index=True)
            model = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(200, random_state={seed}, n_jobs=-1))
            scores = cross_val_score(model, X_bal, y_bal, cv=StratifiedKFold(5, shuffle=True, random_state={seed}), scoring="roc_auc")
            print(json.dumps({{"cv_roc_auc": round(scores.mean(), 4)}}))
            """
    else:
        body = """\
            rng = np.random.default_rng({seed})
            minority = int(y.value_counts().idxmin())
            scores = []
            for tr, te in StratifiedKFold(5, shuffle=True, random_state={seed}).split(X, y):
                ytr = y.iloc[tr]
                pool = tr[ytr.to_numpy() == minority]
                extra = rng.choice(pool, size=int((ytr != minority).sum() - (ytr == minority).sum()))
                idx = np.concatenate([tr, extra])  # copies come from the training fold only
                model = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(200, random_state={seed}, n_jobs=-1))
                model.fit(X.iloc[idx], y.iloc[idx])
                scores.append(roc_auc_score(y.iloc[te], model.predict_proba(X.iloc[te])[:, 1]))
            print(json.dumps({{"cv_roc_auc": round(float(np.mean(scores)), 4)}}))
            """
    return h + textwrap.dedent(body).format(seed=seed)


@review("target_encoding_all_rows", "WF-05", "DCLAB-R04 (cross-split aggregate leakage) and DCLAB-R03 (encoders fitted inside training folds)",
        "Each category is replaced by the mean of `y` over all rows, so every row's own label is inside its encoded "
        "value, and test folds were used to compute it. With rare categories this nearly copies the target.",
        "Use an encoder that is fitted inside each training fold (scikit-learn's TargetEncoder in the pipeline cross-fits), then re-run.")
def _target_enc(key, seed, bug):
    h = _header(key, 3000, seed, "from sklearn.preprocessing import TargetEncoder\nfrom sklearn.compose import make_column_transformer")
    pre = """\
        cat = X.columns[X.nunique().argmax()]  # the column with the most distinct values, treated as a category
        X[cat] = X[cat].astype(str)
        """
    if bug:
        body = textwrap.dedent(pre) + textwrap.dedent("""\
            X[cat] = X[cat].map(y.groupby(X[cat]).mean())
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000))
            scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state={seed}), scoring="roc_auc")
            print(json.dumps({{"cv_roc_auc": round(scores.mean(), 4)}}))
            """)
    else:
        body = textwrap.dedent(pre) + textwrap.dedent("""\
            others = [c for c in X.columns if c != cat]
            prep = make_column_transformer((TargetEncoder(random_state={seed}), [cat]),
                                           (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), others))
            model = make_pipeline(prep, LogisticRegression(max_iter=3000))
            scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state={seed}), scoring="roc_auc")
            print(json.dumps({{"cv_roc_auc": round(scores.mean(), 4)}}))
            """)
    return h + textwrap.dedent(body).format(seed=seed)


@review("tuned_on_test", "WF-09", "DCLAB-R17 (the holdout is scored once, never tuned on)",
        "The regularisation strength is chosen by its score on the test set, and that same test score is reported. The "
        "test set took part in the choice, so the reported number is the best of several tries on it, not an "
        "estimate for new data.",
        "Choose C by cross-validation on the training part only, refit, and score the test set once.")
def _tuned(key, seed, bug):
    h = _header(key, 3000, seed, "from sklearn.metrics import roc_auc_score\nfrom sklearn.model_selection import GridSearchCV")
    if bug:
        body = """\
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state={seed})
            best = (None, -1.0)
            for C in [0.01, 0.1, 1.0, 10.0]:
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X_tr, y_tr)
                auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
                if auc > best[1]:
                    best = (C, auc)
            print(json.dumps({{"best_C": best[0], "test_roc_auc": round(best[1], 4)}}))
            """
    else:
        body = """\
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state={seed})
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000))
            search = GridSearchCV(pipe, {{"logisticregression__C": [0.01, 0.1, 1.0, 10.0]}}, scoring="roc_auc",
                                  cv=StratifiedKFold(5, shuffle=True, random_state={seed})).fit(X_tr, y_tr)
            test_auc = roc_auc_score(y_te, search.predict_proba(X_te)[:, 1])  # the test set is used once, at the end
            print(json.dumps({{"best_C": search.best_params_["logisticregression__C"], "test_roc_auc": round(test_auc, 4)}}))
            """
    return h + textwrap.dedent(body).format(seed=seed)


@review("dedupe_after_split", "WF-03", "DCLAB-R04 (duplicate/entity leakage) and DCLAB-R02 (split by the independence unit)",
        "Duplicates are removed from the training part only, after the split. Copies of the same row can still sit on "
        "both sides of the split, so the test score partly measures memory. Exact duplicates must go before splitting.",
        "Drop duplicate rows from the whole table first, then split and re-run.")
def _dedupe(key, seed, bug):
    h = _header(key, 3000, seed, "from sklearn.metrics import roc_auc_score")
    if bug:
        body = """\
            data = pd.concat([X, y.rename("target")], axis=1)
            train, test = train_test_split(data, test_size=0.25, stratify=data["target"], random_state={seed})
            train = train.drop_duplicates()
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000))
            model.fit(train.drop(columns="target"), train["target"])
            auc = roc_auc_score(test["target"], model.predict_proba(test.drop(columns="target"))[:, 1])
            print(json.dumps({{"test_roc_auc": round(auc, 4), "train_rows": len(train), "test_rows": len(test)}}))
            """
    else:
        body = """\
            data = pd.concat([X, y.rename("target")], axis=1).drop_duplicates()  # before any split
            train, test = train_test_split(data, test_size=0.25, stratify=data["target"], random_state={seed})
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000))
            model.fit(train.drop(columns="target"), train["target"])
            auc = roc_auc_score(test["target"], model.predict_proba(test.drop(columns="target"))[:, 1])
            print(json.dumps({{"test_roc_auc": round(auc, 4), "train_rows": len(train), "test_rows": len(test)}}))
            """
    return h + textwrap.dedent(body).format(seed=seed)


@review("random_split_on_time", "WF-03", "DCLAB-R02 (split by time when rows arrive over time) and DCLAB-R17 (an honest holdout)",
        "The rows arrive over time (`signup_time`), but they are shuffled into a random split, so the model trains on "
        "later subscribers and is tested on earlier ones. The score describes a mix of periods, not the future period "
        "the model will meet.",
        "Sort by `signup_time`, train on the earliest 75% and test on the latest 25%, then re-run.")
def _time(key, seed, bug):
    h = textwrap.dedent(f"""\
        import json

        import pandas as pd
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        from dclab_core.replicas import subscription_churn

        df = subscription_churn.generate(seed={seed}).subscribers
        df["signup_time"] = pd.to_datetime(df["signup_time"])
        features = ["monthly_fee", "tenure_months", "support_calls", "late_payments"]
        y = (df["churned"] == "yes").astype(int)
        """)
    if bug:
        body = """\
            X_tr, X_te, y_tr, y_te = train_test_split(df[features], y, test_size=0.25, stratify=y, random_state={seed})
            model = HistGradientBoostingClassifier(random_state={seed}).fit(X_tr, y_tr)
            print(json.dumps({{"test_roc_auc": round(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]), 4)}}))
            """
    else:
        body = """\
            order = df["signup_time"].sort_values(kind="stable").index
            cut = int(len(order) * 0.75)
            tr, te = order[:cut], order[cut:]  # earlier rows train, later rows test
            model = HistGradientBoostingClassifier(random_state={seed}).fit(df.loc[tr, features], y.loc[tr])
            print(json.dumps({{"test_roc_auc": round(roc_auc_score(y.loc[te], model.predict_proba(df.loc[te, features])[:, 1]), 4)}}))
            """
    return h + textwrap.dedent(body).format(seed=seed)


def code_review(name, key, seed, report):
    spec = REVIEW[name]
    buggy, fixed = spec["fn"](key, seed, True), spec["fn"](key, seed, False)
    ok_b, out_b = run_code(buggy)
    ok_f, out_f = run_code(fixed)
    if not (ok_b and ok_f):
        report["dropped"]["code_failed"] += 1
        report.setdefault("failures", []).append({"dataset": key, "review": name, "error": (out_b if not ok_b else out_f)[-400:]})
        return []
    out = []
    data_line = (f"DATA: `data/public/{key}` (target: {TARGETS[key]})" if name != "random_split_on_time" else
                 "DATA: the subscription_churn replica (rows arrive over time; `signup_time` says when)")
    for bug, code, printed in ((True, buggy, out_b), (False, fixed, out_f)):
        step = spec["step"]
        user = (M.header(step) + f"STAGE: code_review\n{REVIEW_CARD}\n\n{data_line}\n\nCODE:\n{py_block(code)}\n\n"
                f"ITS OUTPUT:\n```json\n{printed.strip()}\n```\n\nALLOWED ACTIONS: {' | '.join(REVIEW_ACTIONS)}")
        if bug:
            why = spec["why"]
            ans = (f"DECISION: fix_leak\nWHY: {why}\nFIX:\n{py_block(fixed)}\nRULE: {spec['rule']}\nCONFIDENCE: high\n"
                   f"NEXT: {spec['next']}")
            r = M.route("continue", step, "re-run with the fixed code before any result is used")
        else:
            why = ("Everything that learns from the data is fitted on training rows only, and the score is computed on "
                   "rows that took no part in any choice, so the number is an honest estimate.")
            ans = f"DECISION: no_issue\nWHY: {why}\nRULE: {spec['rule']}\nCONFIDENCE: high\nNEXT: Use the result as it is."
            r = M.route("continue", step)
        ans = M.with_route_text(ans, r)
        out.append(sample([("user", user), ("assistant", ans)],
                          {"dataset": key if name != "random_split_on_time" else "subscription_churn",
                           "scenario": key if name != "random_split_on_time" else "subscription_churn",
                           "stage": "code_review", "generator": f"review_{name}", "action": "fix_leak" if bug else "no_issue",
                           "step": step, "route": r["action"], "source": "executed"}))
    return out


# ============================================================================ build

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=PUBLIC)
    ap.add_argument("--variants", type=int, default=4, help="seeds per dataset for code-then-decide")
    ap.add_argument("--review-variants", type=int, default=2, help="seeds per dataset for code review")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "out")
    args = ap.parse_args()
    report = {"dropped": Counter()}
    samples = []
    for key in args.datasets:
        t0, n0 = time.time(), len(samples)
        for v in range(args.variants):
            seed = 1 + v + args.seed
            for task in TASKS:
                s = code_then_decide(key, task, seed, report)
                if s:
                    samples.append(s)
            for name in REVIEW if v < args.review_variants else []:
                if name == "random_split_on_time" and key != args.datasets[0]:
                    continue  # one replica, reviewed once per variant
                samples.extend(code_review(name, key, seed, report))
        print(f"{key:16s} {len(samples) - n0:4d} samples  {time.time() - t0:6.1f}s", flush=True)

    seen, stages = set(), {"code_then_decide": {"train": [], "val": [], "test": []}, "code_review": {"train": [], "val": [], "test": []}}
    for s in samples:
        h = hashlib.sha256(J(s["messages"][1:]).encode()).hexdigest()
        if h in seen:
            report["dropped"]["duplicate"] += 1
            continue
        seen.add(h)
        stages[s["metadata"]["stage"]][split_of(s["metadata"]["scenario"])].append(s)
    summary = {}
    for stage, splits in stages.items():
        random.Random(args.seed).shuffle(splits["train"])
        out = args.out / stage
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in splits.items():
            with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
                f.writelines(J(r) + "\n" for r in rows)
        summary[stage] = {k: len(v) for k, v in splits.items()}
        (out / "manifest.json").write_text(json.dumps({"stage": stage, "splits": summary[stage],
                                                       "by_action": dict(Counter(r["metadata"]["action"] for v in splits.values() for r in v)),
                                                       "by_generator": dict(Counter(r["metadata"]["generator"] for v in splits.values() for r in v))},
                                                      indent=2))
    report["dropped"] = dict(report["dropped"])
    report["stages"] = summary
    report["by_action"] = dict(Counter(s["metadata"]["action"] for s in samples))
    (args.out / "code_samples_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
