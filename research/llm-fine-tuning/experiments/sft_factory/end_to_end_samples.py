"""
end_to_end_samples.py
=====================

Full end-to-end trajectories on the unified DCLab workflow WF-01…WF-10: ONE multi-turn conversation that runs a
table through every step, each step reading what the earlier ones decided and routing the work (continue, go back
to the step the evidence reopens, or ask the owner).

    WF-01 problem text ──► problem contract
    WF-02 contract + real table ──► contract check (proceed, or fix the contract → back to WF-01 → revised contract)
    WF-03 whole table, before any split ──► pre-split data checks (deduplicate first if needed); the split follows
    WF-04 training partition ──► findings ──► solution plan (the solution gate)
    WF-05 plan + evidence ──► leakage audit (a leak the plan missed → back to WF-01 → the leak recorded)
    WF-06 feature ladder ──► WF-07 algorithm screen over the plan's families ──► WF-08 calibration
    WF-09 locked pipeline ──► holdout, scored once (a shift → back to WF-03; the run stops there)
    WF-10 the final report, every number taken from the steps above

Where every answer comes from
-----------------------------
* WF-01, WF-02, WF-04 reuse problem_to_plan_samples and contract_check_samples (contract known by construction,
  checks computed on the table, findings and plan from ``dclab_core`` with the gold contract).
* WF-03 applies the data-understanding thresholds of workflow_samples to the whole table.
* WF-05 applies the plan and the declared decision moment; evidence is computed on the training partition.
* WF-06…WF-09 call the same rule functions as the single-step stages (fe_decide, ms_decide, cal_decide,
  holdout_decide) on the partition the plan fixed; the holdout is the rest of the table, touched once.
* Routes only use the moves the repository's graph allows (``dclab_rnd.studio.graph.REVISITS``).

Two variants per scenario: a clean run, and (on every third seed) a run whose labels are partly blank, so the
contract check corrects the contract and the work goes back to WF-01 before continuing on the labelled rows.

Binary scenarios only (the downstream rules are written for a yes/no outcome in ROC-AUC). Every assistant turn
after the contract passes the number check against everything before it.

Training note: train these one assistant turn at a time with the earlier answers as context (build_training_mix
does this), so the model sees exactly what it sees in use.

Usage
-----
    python research/llm-fine-tuning/experiments/sft_factory/end_to_end_samples.py
    python research/llm-fine-tuning/experiments/sft_factory/end_to_end_samples.py --scenarios heart_disease --seeds 1 --variants 1
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

import problem_to_plan_samples as PP
import workflow_map as M
import workflow_samples as W
from contract_check_samples import ACTIONS as CC_ACTIONS
from contract_check_samples import CC_CARD, answer as cc_answer, kind_of, profile, run_checks
from leakage_audit_samples import ACTION_GUIDE, ACTIONS as LA_ACTIONS, CATALOG, REAL_DICTIONARY, _auc, compute_evidence, cv_auc, univariate_auc
from problem_to_plan_samples import (DA_CARD, HERE, J, PU_CARD, SCENARIOS, SD_CARD, SYSTEM_PROMPT, A, P, build_case,
                                     check_contract, compact, files_line, findings_in, findings_out, numbers_ok, plan_json,
                                     table_for, too_close, training_index)
from splits import split_of

E2E_KEYS = [k for k, s in SCENARIOS.items() if s["task"] == "binary" and k != "product_reviews"]  # reviews: the signal is text
FAMILY = {"logistic_regression": "logistic_regression", "random_forest": "random_forest",
          "gradient_boosting": "hist_gradient_boosting", "extra_trees": "extra_trees"}

LA_CARD = (
    "Leakage audit: decide, column by column, what may enter a model that predicts at the contract's prediction "
    "moment. Exclude a column the solution plan forbids, a column the declared decision moment rules out, and an "
    "identifier; flag a column the plan holds for confirmation; keep a column with no leakage evidence. A strong score "
    "alone starts a review, it does not prove leakage. Columns not listed keep their plan status. A leak the plan did "
    "not know reopens the solution (back to WF-01). Output JSON."
)
DU_E2E_CARD = (W.DU_CARD + " If several thresholds are crossed, act on the first in this order: deduplicate_before_split, "
               "small_data_protocol, use_imbalance_aware_evaluation, impute_within_folds; the others are carried to the plan.")
DU_ORDER = ["deduplicate_before_split", "small_data_protocol", "use_imbalance_aware_evaluation", "impute_within_folds"]
REVISIT_CARD = ("Solution revisit (WF-01): the step that sent the work back found something the solution did not know. "
                "Record exactly that change, say what stands, and route the work back to where it continues. Output JSON.")
REPORT_CARD = ("Knowledge capture (WF-10): write the final report from the decisions above only. Every number must be one "
               "already shown in this run. Output JSON.")


# ----------------------------------------------------------------------------------------------- data helpers

def positive_label(y: pd.Series):
    vals = sorted(y.dropna().unique().tolist(), key=str)
    for p in ("yes", "Yes", "true", "True", 1, True):
        if p in vals:
            return p
    return vals[-1]


def numeric_frame(df: pd.DataFrame, cols: list[str], codes: dict) -> pd.DataFrame:
    out = {}
    for c in cols:
        s = df[c]
        out[c] = s.map(codes[c]).astype(float) if c in codes else pd.to_numeric(s, errors="coerce").astype(float)
    return pd.DataFrame(out, index=df.index)


def codes_for(df: pd.DataFrame, cols: list[str]) -> dict:
    return {c: {v: i for i, v in enumerate(pd.unique(df[c].dropna()))} for c in cols if kind_of(df[c]) in ("text", "bool")}


def recipe_frame(X: pd.DataFrame, recipe: str, pick: list[str], skewed: list[str], stats: dict) -> pd.DataFrame:
    """The same formulas as workflow_samples.fe_recipes, with every statistic taken from the training partition."""
    if recipe == "raw":
        return X
    if recipe == "log_transforms":
        return X.assign(**{f"log1p_{c}": np.log1p(X[c]) for c in skewed})
    pairs = [(a, b) for i, a in enumerate(pick) for b in pick[i + 1:]]
    if recipe == "ratios":
        return X.assign(**{f"{a}_over_{b}": X[a] / (X[b].abs() + 1e-3) for a, b in pairs})
    z = {c: (X[c] - stats[c][0]) / (stats[c][1] + 1e-9) for c in pick}
    return X.assign(**{f"{a}_x_{b}": z[a] * z[b] for a, b in pairs})


def stage_turn(step, stage, card, body, actions=None, guide=None):
    return (M.header(step) + f"STAGE: {stage}\n{card}\n\n{body}"
            + (f"\n\nALLOWED ACTIONS: {' | '.join(actions)}" if actions else "")
            + (f"\n({guide})" if guide else ""))


def decision_text(action, why, rule, confidence, nxt, route):
    return M.with_route_text(f"DECISION: {action}\nWHY: {why}\nRULE: {rule}\nCONFIDENCE: {confidence}\nNEXT: {nxt}", route)


# ----------------------------------------------------------------------------------------------- WF-03

def presplit_turn(df, target, y_all):
    """Pre-split checks on the whole table. Returns (user, answer, action)."""
    cols = [c for c in df.columns if c != target]
    X = numeric_frame(df, cols, codes_for(df, cols))
    p = W.du_profile(X, y_all)
    fired = [a for a in DU_ORDER if W.du_flags(p)[a]]
    action = fired[0] if fired else "proceed"
    why = W.du_why(action, p, action)
    others = {"small_data_protocol": f"{p['rows']:,} rows ({p['rows_per_feature']:.1f} per feature)",
              "use_imbalance_aware_evaluation": f"minority class {p['minority']:.1%}",
              "impute_within_folds": f"highest missing rate {p['max_missing']:.1%}",
              "deduplicate_before_split": f"duplicates {p['dup']:.1%}"}
    if len(fired) > 1:
        why += " Also crossed, and carried to the plan: " + "; ".join(others[a] for a in fired[1:]) + "."
    user = stage_turn("WF-03", "data_understanding", DU_E2E_CARD, W.du_body(p), W.DU_ACTIONS, W.DU_GUIDE)
    ans = decision_text(action, why, W.DU_RULES[action], "high", W.DU_NEXT[action], M.route("continue", "WF-04"))
    return user, ans, action


# ----------------------------------------------------------------------------------------------- WF-05

def compact_evidence(c, kind, plan_says, dictionary, ev) -> str:
    return (f"- `{c}` | {kind} | plan says: {plan_says} | data dictionary: {dictionary or 'no entry'} | "
            f"missing {ev.missing_rate:.1%} (positives {ev.missing_pos:.1%}, negatives {ev.missing_neg:.1%}) | "
            f"unique {ev.unique_ratio:.1%} of rows | univariate ROC-AUC {_auc(ev)} | target copy: {'yes' if ev.exact_target_copy else 'no'} | "
            f"suspicious name: {'yes' if ev.suspicious_name else 'no'} | row-number pattern: {'yes' if ev.consecutive_integers else 'no'} | "
            f"3-fold CV ROC-AUC with {ev.cv_with:.3f}, without {ev.cv_without:.3f}, change {ev.cv_change:+.3f}")


def leakage_turn(key, spec, df, plan, train_idx, y_tr, rng, seed):
    """Column decisions. Returns (user, answer_dict, feature_set, codes, moment_only_leaks)."""
    target, time_col = spec["target"], spec["time"]
    d = {x.key: x.choice for x in plan.decisions}
    forbidden = {x["column"]: x["reason"] for x in d.get("features.forbidden", [])}
    confirm = {x["column"]: x["question"] for x in d.get("features.to_confirm", [])}
    identifiers = list(d.get("features.identifiers", []))
    blocked = [c for c in CATALOG.get(key, {}).get("blocked", []) if c in df.columns and c not in forbidden]
    cols = [c for c in df.columns if c != target]
    kinds = {c: kind_of(df[c]) for c in cols}
    not_raw = [c for c in cols if kinds[c] == "date" or c == time_col]
    codes = codes_for(df, cols)
    out_cols = set(forbidden) | set(confirm) | set(identifiers) | set(blocked) | set(not_raw)
    feature_set = [c for c in cols if c not in out_cols]
    tr = df.loc[train_idx]
    base_X = numeric_frame(tr, feature_set, codes).reset_index(drop=True)
    base_auc = cv_auc(base_X, y_tr, seed)
    aucs = {c: univariate_auc(base_X[c], y_tr) or 0.5 for c in feature_set}
    top = sorted(feature_set, key=lambda c: -aucs[c])[:3]
    rest = [c for c in feature_set if c not in top]
    keep = top + ([str(rng.choice(rest))] if rest else [])

    moment = CATALOG.get(key, {}).get("decision")
    lines, decisions, seen = [], [], set()
    audit = ([(c, "forbidden") for c in forbidden] + [(c, "blocked") for c in blocked] +
             [(c, "identifier") for c in identifiers if c not in forbidden] + [(c, "confirm") for c in confirm] +
             [(c, "keep") for c in keep])
    for c, why_kind in audit:
        if c in seen or c not in df.columns:
            continue
        seen.add(c)
        values = numeric_frame(tr, [c], codes)[c].reset_index(drop=True)
        ev = compute_evidence(c, values.to_numpy(), y_tr, base_X, base_auc, in_base=c in feature_set)
        plan_says = {"forbidden": f"forbidden ({forbidden.get(c, '')})", "identifier": "identifier",
                     "confirm": f"to confirm ({confirm.get(c, '')})"}.get(why_kind, "not named")
        dictionary = REAL_DICTIONARY.get(key, {}).get(c)
        lines.append(compact_evidence(c, kinds[c], plan_says, dictionary, ev))
        if why_kind == "forbidden":
            dec, rule = "exclude", "DCLAB-R04 (post-outcome leakage)"
            why = (f"The solution plan forbids `{c}`: {forbidden[c]} Its measured signal (univariate ROC-AUC {_auc(ev)}, "
                   f"CV change {ev.cv_change:+.3f}) would not exist when the prediction is made.")
        elif why_kind == "blocked":
            dec, rule = "exclude", "DCLAB-R04 (post-outcome leakage)"
            why = (f"The declared decision moment rules `{c}` out"
                   + (f" ({dictionary.rstrip('.').lower()})" if dictionary else "")
                   + f": it describes what happens during or after the event being predicted. Its signal (univariate "
                   f"ROC-AUC {_auc(ev)}, CV change {ev.cv_change:+.3f}) would not be available in production. The plan's "
                   f"checks on name and power did not catch it, so the solution must record it.")
        elif why_kind == "identifier":
            dec, rule = "exclude", "DCLAB-R04 (identifier / entity leakage)"
            why = (f"`{c}` takes a different value on {ev.unique_ratio:.1%} of rows: it names the row instead of "
                   f"describing it, so a model could only memorise it.")
        elif why_kind == "confirm":
            dec, rule = "flag_for_review", "DCLAB-R05 (a strong or suspicious column starts a review, it does not prove leakage)"
            why = (f"The plan holds `{c}` until a person confirms when it is written ({confirm[c]}). "
                   f"Until then it stays out of every model.")
        else:
            strong = ev.auc is not None and ev.auc >= 0.8
            dec = "keep"
            rule = "DCLAB-R05 (high AUC alone does not prove leakage)" if strong else "DCLAB-R04 (checked: no leakage pattern)"
            why = (f"Neither the plan nor the decision moment marks `{c}` as written after the prediction moment. "
                   + (f"Its univariate ROC-AUC of {_auc(ev)} is high, but a strong score alone is a reason to review, not "
                      f"proof of leakage, and nothing else points to leakage (not a target copy"
                      + (", no outcome-linked missingness)." if abs(ev.missing_pos - ev.missing_neg) < 0.1 else ").")
                      if strong else
                      f"The evidence shows no leakage pattern (univariate ROC-AUC {_auc(ev)}, CV change {ev.cv_change:+.3f})."))
        decisions.append({"column": c, "decision": dec, "why": why, "rule": rule})
    unlisted = [c for c in feature_set if c not in seen]
    body = (f"DECISION MOMENT (dataset contract): {moment or 'not recorded; use the problem contract'}\n"
            f"EVIDENCE: computed on the training partition only ({len(y_tr):,} rows); CV uses the plan's other usable "
            f"columns as the base.\nOTHER COLUMNS, not audited one by one: {', '.join(f'`{c}`' for c in unlisted) or 'none'}"
            + (f"\nNOT RAW FEATURES: {', '.join(f'`{c}`' for c in not_raw)} (date or time columns)" if not_raw else "")
            + "\nCOLUMNS:\n" + "\n".join(lines))
    user = stage_turn("WF-05", "leakage_audit", LA_CARD, body, LA_ACTIONS, ACTION_GUIDE)
    moment_leaks = [x for x in decisions if x["column"] in blocked]
    if moment_leaks:
        r = M.route("back", "WF-01", M.REVISITS[("WF-05", "WF-01")] + ": " +
                    ", ".join(f"`{x['column']}`" for x in moment_leaks) + " must be recorded as forbidden")
    else:
        r = M.route("continue", "WF-06")
    ans = {"columns": decisions, "feature_set": feature_set,
           "not_raw_features": [{"column": c, "why": "a date or time column orders the split; it is not a raw feature"}
                                for c in not_raw],
           "next": "Build the feature matrix from the feature_set columns on the training partition and compare feature recipes.",
           "route": r}
    return user, ans, feature_set, codes, [x["column"] for x in moment_leaks]


def revisit_for_leaks(columns: list[str]):
    user = M.header("WF-01") + f"STAGE: solution_revisit\n{REVISIT_CARD}\n\nSENT BACK FROM: WF-05 Leakage audit"
    ans = {"change": {"features.forbidden": {"add": [{"column": c, "reason": "ruled out by the declared decision moment; found by the leakage audit"} for c in columns]}},
           "stands": "Every other decision of the solution stands; the audit already kept these columns out of the feature set.",
           "route": M.route("continue", "WF-06", "the audit is complete; continue with the feature ladder")}
    return user, J(ans)


def revisit_for_contract(changes: list[dict]):
    user = M.header("WF-01") + f"STAGE: solution_revisit\n{REVISIT_CARD}\n\nSENT BACK FROM: WF-02 Source and lineage"
    ans = {"change": {ch["field"]: {"from": ch["from"], "to": ch["to"], "reason": ch["reason"]} for ch in changes},
           "stands": "Every stated fact of the contract stands; only the inferred fact the data contradicted changes.",
           "data_rule": "Rows without a label leave the table before the split; they are neither training nor test rows.",
           "route": M.route("continue", "WF-03", "the contract is corrected; design the split on the labelled rows")}
    return user, J(ans)


# ----------------------------------------------------------------------------------------------- WF-06…WF-09

def modelling_turns(plan, df, train_idx, hold_idx, feature_set, codes, y_tr, y_ho, rng, seed):
    turns = []
    Xtr = numeric_frame(df.loc[train_idx], feature_set, codes).reset_index(drop=True)
    Xho = numeric_frame(df.loc[hold_idx], feature_set, codes).reset_index(drop=True)
    pos_line = f"TRAINING PARTITION: {len(Xtr):,} rows | FEATURES: {Xtr.shape[1]} | POSITIVE RATE: {y_tr.mean():.1%}"

    # WF-06 feature ladder
    state = copy.deepcopy(rng)
    recipes = W.fe_recipes(Xtr, rng)
    cont = [c for c in Xtr.columns if Xtr[c].nunique() > 10]
    pick = list(state.choice(cont, size=min(4, len(cont)), replace=False)) if len(cont) >= 2 else []
    skewed = [c for c in cont if Xtr[c].min() >= 0 and abs(Xtr[c].skew()) > 1]
    stats = {c: (Xtr[c].mean(), Xtr[c].std()) for c in pick}
    for name, R in recipes.items():  # the holdout transform must be the same formula as the training one
        pd.testing.assert_frame_equal(recipe_frame(Xtr, name, pick, skewed, stats), R)
    fe_model = ["logistic_regression", "hist_gradient_boosting"][seed % 2]
    splits = W.folds(y_tr, 5, seed)
    res = {}
    for name, R in recipes.items():
        a, t = W.fold_aucs(fe_model, R, y_tr, splits, seed)
        res[name] = {"folds": a, "n": R.shape[1], "t": t}
    fe = W.fe_decide(res)
    body = (f"{pos_line}\nMODEL USED FOR THE COMPARISON: {fe_model} | 5 stratified folds, identical for every recipe\n"
            f"RESULTS:\n| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |\n"
            f"|---|---|---|---|---|---|\n" + "\n".join(fe["rows"]))
    turns.append(("feature_engineering", stage_turn("WF-06", "feature_engineering", W.FE_CARD, body, W.FE_ACTIONS, W.FE_GUIDE),
                  decision_text(fe["action"], fe["why"], fe["rule"], "high", fe["next"], M.route("continue", "WF-07"))))
    recipe = fe["recipe"]
    Xtr_f, Xho_f = recipe_frame(Xtr, recipe, pick, skewed, stats), recipe_frame(Xho, recipe, pick, skewed, stats)

    # WF-07 algorithm screen, over the plan's families
    families = [f for f in next(x.choice for x in plan.decisions if x.key == "model.families") if f != "dummy"]
    names = [FAMILY[f] for f in families if f in FAMILY]
    assert "logistic_regression" in names
    splits = W.folds(y_tr, 3, seed)
    res = {}
    for n in names:
        a, t = W.fold_aucs(n, Xtr_f, y_tr, splits, seed)
        res[n] = {"mean": a.mean(), "std": a.std(), "score": a.mean() - 0.25 * a.std(), "t": t}
    ms = W.ms_decide(res)
    body = (f"{pos_line}\nFEATURE SET: raw" + (f" + {recipe}" if recipe != "raw" else "") + f" ({Xtr_f.shape[1]} features)\n"
            f"FAMILIES FROM THE SOLUTION PLAN: {', '.join(families)} (gradient_boosting runs as hist_gradient_boosting; "
            f"the dummy baseline predicts the training prevalence, so its ROC-AUC is 0.5 by construction)\n"
            f"SCREEN: 3 stratified folds, identical for every model\n"
            f"| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |\n|---|---|---|---|---|\n"
            + "\n".join(ms["rows"]))
    turns.append(("model_selection", stage_turn("WF-07", "model_selection", W.MS_CARD, body, names, W.MS_GUIDE),
                  decision_text(ms["best"], ms["why"],
                                "DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)",
                                ms["confidence"], ms["next"], M.route("continue", "WF-08"))))
    best = ms["best"]

    # WF-08 calibration
    splits = W.folds(y_tr, 3, seed + 1)
    preds = {k: np.zeros(len(y_tr)) for k in ("raw", "sigmoid", "isotonic")}
    for tr, te in splits:
        preds["raw"][te] = W.MODELS[best](seed).fit(Xtr_f.iloc[tr], y_tr.iloc[tr]).predict_proba(Xtr_f.iloc[te])[:, 1]
        for meth in ("sigmoid", "isotonic"):
            c = CalibratedClassifierCV(W.MODELS[best](seed), method=meth, cv=3).fit(Xtr_f.iloc[tr], y_tr.iloc[tr])
            preds[meth][te] = c.predict_proba(Xtr_f.iloc[te])[:, 1]
    st = {k: {"ece": W.ece(y_tr, p), "auc": roc_auc_score(y_tr, p), "brier": float(np.mean((p - y_tr.to_numpy()) ** 2))}
          for k, p in preds.items()}
    train_rows = int(len(y_tr) * 2 / 3)
    cal = W.cal_decide(st, best, train_rows)
    body = (f"{pos_line}\nMODEL: {best} | 3 stratified folds | training rows per fold: about {train_rows:,}\n"
            f"| probabilities | ECE | ROC-AUC | Brier |\n|---|---|---|---|\n"
            + "\n".join(f"| {k} | {s['ece']:.4f} | {s['auc']:.4f} | {s['brier']:.4f} |" for k, s in st.items()))
    turns.append(("calibration", stage_turn("WF-08", "optimization_reliability", W.OPT_CAL_CARD, body, W.OPT_CAL_ACTIONS),
                  decision_text(cal["action"], cal["why"].strip(),
                                "DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)",
                                "high", cal["next"], M.route("continue", "WF-09"))))

    # WF-09 the holdout, scored once with the locked pipeline
    def locked():
        if cal["action"] == "keep_uncalibrated":
            return W.MODELS[best](seed)
        return CalibratedClassifierCV(W.MODELS[best](seed), method=cal["action"].split("_")[1], cv=3)
    cv = []
    for tr, te in W.folds(y_tr, 3, seed + 2):
        m = locked().fit(Xtr_f.iloc[tr], y_tr.iloc[tr])
        cv.append(roc_auc_score(y_tr.iloc[te], m.predict_proba(Xtr_f.iloc[te])[:, 1]))
    cv = np.array(cv)
    hauc = roc_auc_score(y_ho, locked().fit(Xtr_f, y_tr).predict_proba(Xho_f)[:, 1])
    psis = sorted(((c, W.psi(Xtr[c], Xho[c])) for c in Xtr.columns), key=lambda t: -t[1])[:3]
    ho = W.holdout_decide(cv, hauc, psis, len(y_ho))
    lock = best + ("" if cal["action"] == "keep_uncalibrated" else f" + {cal['action'].split('_')[1]} calibration")
    body = (f"LOCKED PIPELINE: raw" + (f" + {recipe}" if recipe != "raw" else "") + f" features → {lock}\n"
            f"training rows: {len(y_tr):,} | holdout rows: {len(y_ho):,} (holdout positive rate {y_ho.mean():.1%}); the "
            f"holdout is the part of the table the plan locked at the start\n"
            f"- 3-fold CV ROC-AUC on training: {cv.mean():.4f} (std {cv.std():.4f})\n"
            f"- holdout ROC-AUC (scored once): {hauc:.4f}\n"
            f"- difference, CV minus holdout: {cv.mean() - hauc:+.4f} (absolute {abs(cv.mean() - hauc):.4f})\n"
            f"- largest feature shifts, training vs holdout (PSI): " + ", ".join(f"`{c}` {p:.3f}" for c, p in psis))
    ho_route = M.single_route("holdout", ho["action"])
    turns.append(("holdout", stage_turn("WF-09", "optimization_reliability", W.OPT_HO_CARD, body, W.OPT_HO_ACTIONS),
                  decision_text(ho["action"], ho["why"], "DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)",
                                "high", ho["next"], ho_route)))
    summary = {"recipe": fe["action"], "model": best, "calibration": cal["action"], "holdout": ho["action"]}
    facts = {"recipe": recipe, "model": lock, "cv": f"{cv.mean():.4f}", "cv_std": f"{cv.std():.4f}", "holdout": f"{hauc:.4f}",
             "holdout_decision": ho["action"], "shifts": [(c, f"{p:.3f}") for c, p in psis if p > 0.1],
             "ms_confidence": ms["confidence"], "cal": cal["action"]}
    return turns, summary, facts, ho_route


def report_turn(brief, rows, train_n, hold_n, deduped, la, facts, revisits):
    user = M.header("WF-10") + f"STAGE: report\n{REPORT_CARD}"
    excluded = [x["column"] for x in la["columns"] if x["decision"] == "exclude"]
    flagged = [x["column"] for x in la["columns"] if x["decision"] == "flag_for_review"]
    limits = []
    if facts["holdout_decision"] == "investigate_gap_without_shift":
        limits.append("CV and holdout disagree beyond the ±0.02 band without a population shift; report both numbers with a bootstrap interval, not the higher one")
    if facts["ms_confidence"] == "low":
        limits.append("the model family was a low-confidence choice between near-tied families")
    if flagged:
        limits.append("columns held for confirmation stay out until the owner says when they are written")
    ans = {
        "problem": {"goal": brief.goal.value if brief.goal.known else None, "target": brief.target_column.value,
                    "prediction_moment": brief.prediction_moment.value},
        "data": {"rows": rows, "training_rows": train_n, "holdout_rows": hold_n, "deduplicated_before_split": deduped},
        "solution_revisits": revisits,
        "features": {"excluded": excluded, "held_for_confirmation": flagged, "recipe": facts["recipe"]},
        "model": facts["model"],
        "results": {"cv_roc_auc": facts["cv"], "cv_std": facts["cv_std"], "holdout_roc_auc": facts["holdout"],
                    "holdout_decision": facts["holdout_decision"]},
        "monitor": [{"feature": c, "psi": p} for c, p in facts["shifts"]],
        "limits": limits,
        "status": "complete",
    }
    return user, J(ans)


# ----------------------------------------------------------------------------------------------- build

def build(keys, seeds, variants, seed0):
    bench = PP.benchmark_texts()
    samples, report = [], {"dropped": Counter(), "decisions": defaultdict(Counter), "max_benchmark_similarity": 0.0,
                           "routes": Counter()}
    for key in keys:
        spec = SCENARIOS[key]
        t0, n0 = time.time(), len(samples)
        rng = np.random.default_rng(int(hashlib.sha256(("e2e" + key).encode()).hexdigest()[:8], 16) + seed0)
        for seed in seeds:
            df0, table = table_for(key, spec, seed)
            files = files_line(spec, df0)
            plans = [("clean", v) for v in range(variants)] + ([("labels_missing", 0)] if seed % 3 == 0 else [])
            cache = {}
            made = Counter()
            for _attempt in range(len(plans) * 8):
                todo = [p for p in plans if made[p] == 0]
                if not todo:
                    break
                variant = todo[0]
                kind = variant[0]
                description, brief, connected = build_case(key, spec, rng, list(df0.columns), True)
                sim = too_close(description, bench)
                report["max_benchmark_similarity"] = max(report["max_benchmark_similarity"], round(sim, 3))
                if sim >= 0.5 or check_contract(description, brief) or not connected:
                    report["dropped"]["description"] += 1
                    continue
                df = df0.copy()
                if kind == "labels_missing":
                    if not (brief.label_coverage.known and brief.label_coverage.status == "inferred"):
                        report["dropped"]["labels_variant_needs_inferred_coverage"] += 1
                        continue
                    blank = np.random.default_rng(seed).random(len(df)) < 0.2
                    df[spec["target"]] = df[spec["target"]].astype(object).where(~blank, None)
                checks = run_checks(df, brief)
                fails = [c for c, v in checks.items() if v == "fail"]
                if fails != ([] if kind == "clean" else ["target_labelled"]):
                    report["dropped"]["contract_check_unexpected"] += 1
                    continue

                chain = [("user", M.header("WF-01") + f"STAGE: problem_understanding\n{PU_CARD}\n\nDESCRIPTION:\n{description}\n\n"
                                  f"FILES (headers only):\n{files}"),
                         ("assistant", M.with_route_json(J(compact(brief)), M.route("continue", "WF-02")))]
                cc = cc_answer("clean" if kind == "clean" else "labels_missing", df, brief, checks)
                cc_route = M.single_route("contract_check", cc["decision"])
                chain += [("user", M.header("WF-02") + f"STAGE: contract_check\n{CC_CARD}\n\nCheck the contract above against the table.\n\n"
                                   f"DATA PROFILE:\n{profile(df, table)}\n\nALLOWED DECISIONS: {' | '.join(CC_ACTIONS)}"),
                          ("assistant", M.with_route_json(J(cc), cc_route))]
                revisits = []
                if kind == "labels_missing":
                    u, a = revisit_for_contract(cc["contract_changes"])
                    chain += [("user", u), ("assistant", a)]
                    revisits.append("WF-02 → WF-01: label coverage corrected to partial")
                    brief = brief.model_copy(update={"label_coverage": PP.CoverageFact(value="partial", status="inferred",
                                                                                     reason=cc["contract_changes"][0]["reason"])})
                    df = df[df[spec["target"]].notna()].reset_index(drop=True)
                    df[spec["target"]] = df[spec["target"]].astype(df0[spec["target"]].dtype)

                pos = positive_label(df[spec["target"]])
                y_all = (df[spec["target"]] == pos).astype(int)
                u, a, du_action = presplit_turn(df, spec["target"], y_all)
                chain += [("user", u), ("assistant", a)]
                deduped = du_action == "deduplicate_before_split"
                if deduped:
                    keep = ~df.duplicated()
                    df, y_all = df[keep].reset_index(drop=True), y_all[keep].reset_index(drop=True)
                train_idx, partition = training_index(df, spec, np.random.default_rng(seed))
                tset = set(train_idx)
                hold_idx = [i for i in df.index if i not in tset]
                y_tr, y_ho = y_all.loc[train_idx].reset_index(drop=True), y_all.loc[hold_idx].reset_index(drop=True)

                ck = (kind, J(compact(brief)))
                if ck not in cache:
                    findings = A.analyse({table: df}, target=spec["target"], brief=brief, train_index=train_idx, main=table)
                    plan = P.plan(brief, findings)
                    cache[ck] = (findings, plan, P.validate(plan, findings))
                findings, plan, issues = cache[ck]
                metric = next(x.choice for x in plan.decisions if x.key == "metric.primary")
                if issues or metric != "roc_auc":  # the downstream stage rules are written in ROC-AUC
                    report["dropped"]["plan_invalid" if issues else "metric_not_roc_auc"] += 1
                    continue
                table_line = f"TABLE: {table} — {len(df):,} rows, {df.shape[1]} columns ({', '.join(map(str, df.columns))})"
                chain += [
                    ("user", M.header("WF-04") + f"STAGE: data_analysis\n{DA_CARD}\n\n{table_line}\nTRAINING PARTITION: {partition}\n\n"
                             f"MEASUREMENTS:\n{findings_in(findings)}"),
                    ("assistant", M.with_route_json(J({"task": findings.task, "target": findings.target,
                                                       "training_rows": findings.training_rows, "findings": findings_out(findings)}),
                                                    M.single_route("data_analysis", ""))),
                    ("user", M.header("WF-04") + f"STAGE: solution_design\n{SD_CARD}\n\nUse the contract and the findings above."),
                    ("assistant", M.with_route_json(J(plan_json(plan)), M.single_route("solution_design", ""))),
                ]
                mk = (kind, J([d.model_dump() for d in plan.decisions if d.key.startswith("features") or d.key == "model.families"]))
                if ("model", mk) not in cache:
                    mrng = np.random.default_rng(int(hashlib.sha256((key + str(seed) + J(mk)).encode()).hexdigest()[:8], 16))
                    la_user, la, feature_set, codes, leaks = leakage_turn(key, spec, df, plan, train_idx, y_tr, mrng, seed)
                    tail = [("user", la_user), ("assistant", J(la))]
                    rev = []
                    if leaks:
                        u, a = revisit_for_leaks(leaks)
                        tail += [("user", u), ("assistant", a)]
                        rev.append("WF-05 → WF-01: " + ", ".join(leaks) + " recorded as forbidden")
                    turns, summary, facts, ho_route = modelling_turns(plan, df, train_idx, hold_idx, feature_set, codes,
                                                                      y_tr, y_ho, mrng, seed)
                    for _s, u, a in turns:
                        tail += [("user", u), ("assistant", a)]
                    cache[("model", mk)] = (tail, summary, facts, ho_route, la, rev)
                tail, summary, facts, ho_route, la, rev = cache[("model", mk)]
                chain += tail
                all_revisits = revisits + rev
                if ho_route["action"] == "continue":
                    u, a = report_turn(brief, len(df), len(train_idx), len(hold_idx), deduped, la, facts, all_revisits)
                    chain += [("user", u), ("assistant", a)]
                bad = []
                for i, (role, text) in enumerate(chain):
                    if role == "assistant" and i > 1:  # the contract turn is checked by the contract rules instead
                        bad += numbers_ok(text, SYSTEM_PROMPT + "\n" + "\n".join(t for _, t in chain[:i]))
                if bad:
                    report["dropped"]["number_not_in_input"] += 1
                    report.setdefault("number_examples", []).append({"scenario": key, "seed": seed, "numbers": bad[:4]})
                    continue
                for k, v in summary.items():
                    report["decisions"][k][v] += 1
                report["decisions"]["presplit"][du_action] += 1
                routes = [json.loads(t)["route"]["action"] if t.lstrip().startswith("{") and "route" in json.loads(t)
                          else (t.split("NEXT: [", 1)[1].split()[0] if "NEXT: [" in t else "none")
                          for r_, t in chain if r_ == "assistant"]
                report["routes"].update(routes)
                chars = sum(len(t) for _, t in chain) + len(SYSTEM_PROMPT)
                samples.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT}] +
                                            [{"role": r, "content": c} for r, c in chain],
                                "metadata": {"scenario": key, "dataset": key, "seed": seed, "source": spec["source"],
                                             "stage": "end_to_end", "action": "end_to_end", "variant": kind,
                                             "turns": len(chain) // 2, "chars": chars, "routed": True,
                                             "revisits": len(all_revisits), "stopped_at": "WF-10" if ho_route["action"] == "continue" else "WF-09",
                                             **summary}})
                made[variant] += 1
        print(f"{key:20s} {len(samples) - n0:4d} trajectories  {time.time() - t0:6.1f}s", flush=True)
    return samples, report



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="*", default=E2E_KEYS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--variants", type=int, default=3, help="clean descriptions per scenario and seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "out")
    args = ap.parse_args()
    samples, report = build(args.scenarios, args.seeds, args.variants, args.seed)
    seen, splits = set(), {"train": [], "val": [], "test": []}
    for s in samples:
        h = hashlib.sha256(J(s["messages"][1:]).encode()).hexdigest()
        if h in seen:
            report["dropped"]["duplicate"] += 1
            continue
        seen.add(h)
        splits[split_of(s["metadata"]["scenario"])].append(s)
    random.Random(args.seed).shuffle(splits["train"])
    out = args.out / "end_to_end"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(J(r) + "\n" for r in rows)
    chars = [s["metadata"]["chars"] for s in samples]
    report.update({
        "dropped": dict(report["dropped"]), "decisions": {k: dict(v) for k, v in report["decisions"].items()},
        "routes": dict(report["routes"]),
        "variants": dict(Counter(s["metadata"]["variant"] for s in samples)),
        "stopped_at": dict(Counter(s["metadata"]["stopped_at"] for s in samples)),
        "with_revisits": sum(1 for s in samples if s["metadata"]["revisits"]),
        "splits": {k: len(v) for k, v in splits.items()},
        "by_scenario": dict(Counter(s["metadata"]["scenario"] for s in samples)),
        "length": {"chars_median": int(np.median(chars)) if chars else 0, "chars_max": max(chars, default=0),
                   "approx_tokens_median": int(np.median(chars) / 3.5) if chars else 0,
                   "approx_tokens_max": int(max(chars, default=0) / 3.5)},
    })
    (out / "manifest.json").write_text(json.dumps({"stage": "end_to_end", "splits": report["splits"],
                                                   "by_scenario": report["by_scenario"]}, indent=2))
    (args.out / "end_to_end_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
