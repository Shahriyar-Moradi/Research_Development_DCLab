"""
end_to_end_samples.py
=====================

Full end-to-end trajectories: ONE multi-turn conversation that runs the whole DCLab workflow on one table, each step
reading what the earlier steps decided.

    problem text ──► problem_understanding ──► contract
    contract + real table ──► contract_check ──► proceed
    contract + training partition ──► data_analysis ──► findings
    contract + findings ──► solution_design ──► plan (WF-01…WF-10)
    plan + evidence ──► leakage_audit ──► column decisions and the feature set
    feature set ──► feature_engineering ──► recipe          (rule R07/R16)
    plan's model families ──► model_selection ──► family    (rule R13/R14)
    chosen family ──► calibration ──► keep or calibrate     (rule R14/R18)
    locked pipeline ──► final holdout, scored once ──► accept or investigate   (rule R17/R19)

Where every answer comes from
-----------------------------
* The first four turns reuse problem_to_plan_samples (contract known by construction; findings and plan are the
  library's own output with the gold contract) and contract_check_samples (checks computed on the table).
* The leakage audit follows the plan and the declared decision moment: a column the plan forbids, a column the
  dataset's decision moment rules out, and an identifier are excluded; a column the plan holds for confirmation is
  flagged; the strongest remaining columns, and one more, are kept. The evidence for each is computed.
* Feature engineering, model selection, calibration and the holdout decision call the SAME rule functions as the
  single-step stages in workflow_samples.py (fe_decide, ms_decide, cal_decide, holdout_decide), on the training
  partition the plan fixed. The holdout is the rest of the table: it is touched once, at the last step.

Only binary scenarios run end to end (the downstream stages are written for a yes/no outcome): the subscription_churn
replica and the ten public datasets. Every assistant turn passes the number check against everything before it.

Training note: these conversations need loss on EVERY assistant turn, and they are long (see the report's token
estimate); train them with a larger max_length or not at all.

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
import workflow_samples as W
from contract_check_samples import ACTIONS as CC_ACTIONS
from contract_check_samples import CC_CARD, answer as cc_answer, kind_of, profile, run_checks
from leakage_audit_samples import ACTION_GUIDE, ACTIONS as LA_ACTIONS, CATALOG, REAL_DICTIONARY, _auc, compute_evidence, cv_auc, univariate_auc
from problem_to_plan_samples import (DA_CARD, HERE, J, PU_CARD, SCENARIOS, SD_CARD, SPLIT_OF, SYSTEM_PROMPT, A, P, build_case,
                                     check_contract, compact, files_line, findings_in, findings_out, numbers_ok, plan_json,
                                     table_for, too_close, training_index)

E2E_KEYS = [k for k, s in SCENARIOS.items() if s["task"] == "binary" and k != "product_reviews"]  # reviews: the signal is text
FAMILY = {"logistic_regression": "logistic_regression", "random_forest": "random_forest",
          "gradient_boosting": "hist_gradient_boosting", "extra_trees": "extra_trees"}

LA_CARD = (
    "Leakage audit: decide, column by column, what may enter a model that predicts at the contract's prediction "
    "moment. Exclude a column the solution plan forbids, a column the declared decision moment rules out, and an "
    "identifier; flag a column the plan holds for confirmation; keep a column with no leakage evidence. A strong score "
    "alone starts a review, it does not prove leakage. Columns not listed keep their plan status. Output JSON."
)


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


def stage_turn(stage, card, body, actions, guide=None):
    return (f"STAGE: {stage}\n{card}\n\n{body}\n\nALLOWED ACTIONS: {' | '.join(actions)}"
            + (f"\n({guide})" if guide else ""))


def decision_text(action, why, rule, confidence, nxt):
    return f"DECISION: {action}\nWHY: {why}\nRULE: {rule}\nCONFIDENCE: {confidence}\nNEXT: {nxt}"


# ----------------------------------------------------------------------------------------------- leakage audit

def leakage_turn(key, spec, df, plan, train_idx, y_tr, rng, seed):
    """Column decisions for the audit turn. Returns (user_body, answer_dict, feature_set)."""
    target, time_col = spec["target"], spec["time"]
    d = {x.key: x.choice for x in plan.decisions}
    forbidden = {x["column"]: x["reason"] for x in d.get("features.forbidden", [])}
    confirm = {x["column"]: x["question"] for x in d.get("features.to_confirm", [])}
    identifiers = list(d.get("features.identifiers", []))
    blocked = [c for c in CATALOG.get(key, {}).get("blocked", []) if c in df.columns and c not in forbidden]
    cols = [c for c in df.columns if c != target]
    kinds = {c: kind_of(df[c]) for c in cols}
    not_raw = [c for c in cols if kinds[c] == "date" or c == time_col]
    codes = {c: {v: i for i, v in enumerate(pd.unique(df[c].dropna()))} for c in cols if kinds[c] in ("text", "bool")}
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
    blocks, decisions = [], []
    audit = ([(c, "forbidden") for c in forbidden] + [(c, "blocked") for c in blocked] +
             [(c, "identifier") for c in identifiers if c not in forbidden] + [(c, "confirm") for c in confirm] +
             [(c, "keep") for c in keep])
    seen = set()
    for c, why_kind in audit:
        if c in seen or c not in df.columns:
            continue
        seen.add(c)
        values = numeric_frame(tr, [c], codes)[c].reset_index(drop=True)
        ev = compute_evidence(c, values.to_numpy(), y_tr, base_X, base_auc, in_base=c in feature_set)
        plan_says = {"forbidden": f"forbidden ({forbidden.get(c, '')})", "identifier": "identifier",
                     "confirm": f"to confirm ({confirm.get(c, '')})"}.get(why_kind, "not named")
        if why_kind == "identifier" and c in confirm:
            plan_says = "identifier"
        dictionary = REAL_DICTIONARY.get(key, {}).get(c)
        blocks.append(f"COLUMN: `{c}` | kind: {kinds[c]} | plan says: {plan_says} | data dictionary: "
                      f"{dictionary or 'no entry'}\n{ev.render()}")
        if why_kind == "forbidden":
            dec, rule = "exclude", "DCLAB-R04 (post-outcome leakage)"
            why = (f"The solution plan forbids `{c}`: {forbidden[c]} Its measured signal (univariate ROC-AUC {_auc(ev)}, "
                   f"CV change {ev.cv_change:+.3f}) would not exist when the prediction is made.")
        elif why_kind == "blocked":
            dec, rule = "exclude", "DCLAB-R04 (post-outcome leakage)"
            why = (f"The declared decision moment rules `{c}` out"
                   + (f" ({dictionary.rstrip('.').lower()})" if dictionary else "")
                   + f": it describes what happens during or after the event being predicted. Its signal (univariate "
                   f"ROC-AUC {_auc(ev)}, CV change {ev.cv_change:+.3f}) would not be available in production, even "
                   f"though the plan's checks on name and power did not catch it.")
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
            + "\n\n" + "\n\n".join(blocks))
    ans = {"columns": decisions, "feature_set": feature_set,
           "not_raw_features": [{"column": c, "why": "a date or time column orders the split; it is not a raw feature"}
                                for c in not_raw],
           "next": "Build the feature matrix from the feature_set columns on the training partition and compare feature recipes."}
    return body, ans, feature_set, codes


# ----------------------------------------------------------------------------------------------- the modelling turns

def modelling_turns(key, spec, df, plan, train_idx, hold_idx, feature_set, codes, y_tr, y_ho, rng, seed):
    turns = []
    Xtr = numeric_frame(df.loc[train_idx], feature_set, codes).reset_index(drop=True)
    Xho = numeric_frame(df.loc[hold_idx], feature_set, codes).reset_index(drop=True)
    pos_line = f"TRAINING PARTITION: {len(Xtr):,} rows | FEATURES: {Xtr.shape[1]} | POSITIVE RATE: {y_tr.mean():.1%}"

    # feature engineering
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
    turns.append(("feature_engineering", stage_turn("feature_engineering", W.FE_CARD, body, W.FE_ACTIONS, W.FE_GUIDE),
                  decision_text(fe["action"], fe["why"], fe["rule"], "high", fe["next"])))
    recipe = fe["recipe"]
    Xtr_f, Xho_f = recipe_frame(Xtr, recipe, pick, skewed, stats), recipe_frame(Xho, recipe, pick, skewed, stats)

    # model selection, over the plan's families
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
    turns.append(("model_selection", stage_turn("model_selection", W.MS_CARD, body, names, W.MS_GUIDE),
                  decision_text(ms["best"], ms["why"],
                                "DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)",
                                ms["confidence"], ms["next"])))
    best = ms["best"]

    # calibration
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
    turns.append(("calibration", stage_turn("optimization_reliability", W.OPT_CAL_CARD, body, W.OPT_CAL_ACTIONS),
                  decision_text(cal["action"], cal["why"].strip(),
                                "DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)",
                                "high", cal["next"])))

    # final holdout, scored once with the locked pipeline
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
    turns.append(("holdout", stage_turn("optimization_reliability", W.OPT_HO_CARD, body, W.OPT_HO_ACTIONS),
                  decision_text(ho["action"], ho["why"], "DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)",
                                "high", ho["next"])))
    summary = {"recipe": fe["action"], "model": best, "calibration": cal["action"], "holdout": ho["action"]}
    return turns, summary


# ----------------------------------------------------------------------------------------------- build

def build(keys, seeds, variants, seed0):
    bench = PP.benchmark_texts()
    samples, report = [], {"dropped": Counter(), "decisions": defaultdict(Counter), "max_benchmark_similarity": 0.0}
    for key in keys:
        spec = SCENARIOS[key]
        t0, n0 = time.time(), len(samples)
        rng = np.random.default_rng(int(hashlib.sha256(("e2e" + key).encode()).hexdigest()[:8], 16) + seed0)
        for seed in seeds:
            df, table = table_for(key, spec, seed)
            pos = positive_label(df[spec["target"]])
            y_all = (df[spec["target"]] == pos).astype(int)
            train_idx, partition = training_index(df, spec, rng)
            hold_idx = [i for i in df.index if i not in set(train_idx)]
            y_tr, y_ho = y_all.loc[train_idx].reset_index(drop=True), y_all.loc[hold_idx].reset_index(drop=True)
            files = files_line(spec, df)
            cache = {}
            made = 0
            for _attempt in range(variants * 8):
                if made >= variants:
                    break
                description, brief, connected = build_case(key, spec, rng, list(df.columns), True)
                sim = too_close(description, bench)
                report["max_benchmark_similarity"] = max(report["max_benchmark_similarity"], round(sim, 3))
                if sim >= 0.5 or check_contract(description, brief) or not connected:
                    report["dropped"]["description"] += 1
                    continue
                checks = run_checks(df, brief)
                if any(v == "fail" for v in checks.values()):
                    report["dropped"]["contract_check_failed"] += 1
                    continue
                ck = J(compact(brief))
                if ck not in cache:
                    findings = A.analyse({table: df}, target=spec["target"], brief=brief, train_index=train_idx, main=table)
                    plan = P.plan(brief, findings)
                    cache[ck] = (findings, plan, P.validate(plan, findings))
                findings, plan, issues = cache[ck]
                metric = next(x.choice for x in plan.decisions if x.key == "metric.primary")
                if issues or metric != "roc_auc":  # the downstream stage rules are written in ROC-AUC
                    report["dropped"]["plan_invalid" if issues else "metric_not_roc_auc"] += 1
                    continue
                pu_user = (f"STAGE: problem_understanding\n{PU_CARD}\n\nDESCRIPTION:\n{description}\n\n"
                           f"FILES (headers only):\n{files}")
                cc = cc_answer("clean", df, brief, checks)
                table_line = f"TABLE: {table} — {len(df):,} rows, {df.shape[1]} columns ({', '.join(map(str, df.columns))})"
                chain = [
                    ("user", pu_user), ("assistant", J(compact(brief))),
                    ("user", f"STAGE: contract_check\n{CC_CARD}\n\nCheck the contract above against the table.\n\n"
                             f"DATA PROFILE:\n{profile(df, table)}\n\nALLOWED DECISIONS: {' | '.join(CC_ACTIONS)}"),
                    ("assistant", J(cc)),
                    ("user", f"STAGE: data_analysis\n{DA_CARD}\n\n{table_line}\nTRAINING PARTITION: {partition}\n\n"
                             f"MEASUREMENTS:\n{findings_in(findings)}"),
                    ("assistant", J({"task": findings.task, "target": findings.target, "training_rows": findings.training_rows,
                                     "findings": findings_out(findings)})),
                    ("user", f"STAGE: solution_design\n{SD_CARD}\n\nUse the contract and the findings above."),
                    ("assistant", J(plan_json(plan))),
                ]
                mk = J([d.model_dump() for d in plan.decisions if d.key.startswith("features") or d.key == "model.families"])
                mkey = ("model", mk)
                if mkey not in cache:
                    mrng = np.random.default_rng(int(hashlib.sha256((key + str(seed) + mk).encode()).hexdigest()[:8], 16))
                    body, la, feature_set, codes = leakage_turn(key, spec, df, plan, train_idx, y_tr, mrng, seed)
                    la_user = stage_turn("leakage_audit", LA_CARD, body, LA_ACTIONS, ACTION_GUIDE)
                    turns, summary = modelling_turns(key, spec, df, plan, train_idx, hold_idx, feature_set, codes,
                                                     y_tr, y_ho, mrng, seed)
                    cache[mkey] = ([("leakage_audit", la_user, J(la))] + turns, summary)
                tail, summary = cache[mkey]
                for _stage, u, a in tail:
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
                chars = sum(len(t) for _, t in chain) + len(SYSTEM_PROMPT)
                samples.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT}] +
                                            [{"role": r, "content": c} for r, c in chain],
                                "metadata": {"scenario": key, "dataset": key, "seed": seed, "source": spec["source"],
                                             "stage": "end_to_end", "action": "end_to_end", "turns": len(chain) // 2,
                                             "chars": chars, **summary}})
                made += 1
        print(f"{key:20s} {len(samples) - n0:4d} trajectories  {time.time() - t0:6.1f}s", flush=True)
    return samples, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="*", default=E2E_KEYS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--variants", type=int, default=2, help="descriptions per scenario and seed")
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
        splits[SPLIT_OF.get(s["metadata"]["scenario"], "train")].append(s)
    random.Random(args.seed).shuffle(splits["train"])
    out = args.out / "end_to_end"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(J(r) + "\n" for r in rows)
    chars = [s["metadata"]["chars"] for s in samples]
    report.update({
        "dropped": dict(report["dropped"]), "decisions": {k: dict(v) for k, v in report["decisions"].items()},
        "splits": {k: len(v) for k, v in splits.items()},
        "by_scenario": dict(Counter(s["metadata"]["scenario"] for s in samples)),
        "length": {"chars_median": int(np.median(chars)) if chars else 0, "chars_max": max(chars, default=0),
                   "approx_tokens_median": int(np.median(chars) / 3.5) if chars else 0,
                   "approx_tokens_max": int(max(chars, default=0) / 3.5)},
    })
    (out / "manifest.json").write_text(json.dumps({"stage": "end_to_end", "splits": report["splits"],
                                                   "by_scenario": report["by_scenario"], "split_by_scenario": SPLIT_OF}, indent=2))
    (args.out / "end_to_end_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
