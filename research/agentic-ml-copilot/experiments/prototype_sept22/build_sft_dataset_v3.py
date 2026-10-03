#!/usr/bin/env python3
"""
build_sft_dataset_v3.py
=========================
ADDITIVE — fixes a real gap in v1/v2, does not delete or replace them.

THE PROBLEM THIS FIXES
-----------------------
v1's `agent_memory`-sourced examples (and, to a lesser extent, v2's
reasoning traces) asked things like:

    "What did experiment EXP-023 on the 'credit_default' dataset find
     during the 'feature engineering' stage?"

...and then answered with a bare conclusion. A model fine-tuned on that
learns to recite "EXP-023 said X" without ever being told what EXP-023
actually WAS: what credit_default even predicts, what "feature
engineering stage" means methodologically, or what was actually compared
in that run (which feature recipes, which model, how many folds, what
scores). That's memorization of opaque ID -> fact pairs, not learning
the underlying reasoning pattern — exactly the failure mode this project
is trying to avoid everywhere else in its own methodology.

THE FIX
-------
Every example built here is self-contained. The user turn always
includes, in plain language:
  1. A DATASET CARD — what this dataset is and what it predicts, pulled
     from data/public/<name>/meta.json where available, plus a short
     factual description of the well-known public dataset.
  2. A STAGE CARD — what this pipeline stage does methodologically and
     why it exists, derived from the repo's own workflow_blocks.json.
  3. A SETUP SUMMARY — the actual configurations/models/features compared
     in THIS run, extracted directly from the experiment's `evidence`
     block (never invented) — e.g. "6 feature recipes were compared on
     lightgbm with 3-fold CV: raw (14 features, mean ROC-AUC 0.8955),
     ratios (37 features, 0.8944), ...".
Only then does the question get asked, and only then is the answer
(observed evidence -> interpretation -> decision -> risks -> next test)
grounded in something the model can actually reason from.

Usage
-----
    python build_sft_dataset_v3.py \
        --results-dir evidence/campaigns/model_building_50_v1/results \
        --meta-dir data/public \
        --out-dir sft_out_v3
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a senior data-science / ML engineering research assistant trained on "
    "DCLab's evidence-first model-building methodology. You are given the dataset, "
    "the pipeline stage, and the exact experimental setup for a run. Answer using "
    "exactly this structure: (1) Observed evidence, citing what supports each point; "
    "(2) Interpretation and uncertainty; (3) Decision or recommendation; (4) Risks "
    "and missing evidence; (5) The smallest falsifiable next experiment. Never invent "
    "numbers that were not given to you in the setup."
)

# --- 1. Dataset cards: what each dataset actually IS and predicts. -----------
# Short, factual descriptions of these well-documented public datasets.
DATASET_CARDS = {
    "adult": "US Census Bureau income extract. Predicts whether a person's annual "
             "income exceeds $50K from demographic and employment attributes "
             "(age, education, occupation, hours worked, etc.).",
    "bank_marketing": "A Portuguese bank's phone-based term-deposit marketing campaign "
                       "records. Predicts whether a contacted client subscribes to a "
                       "term deposit.",
    "breast_cancer": "Wisconsin Diagnostic Breast Cancer dataset. Predicts whether a "
                      "tumor is malignant or benign from cell-nucleus measurements "
                      "taken from a biopsy image.",
    "credit_default": "Taiwanese credit-card client records. Predicts whether a client "
                       "will default on their credit card payment next month, using "
                       "payment history and bill/payment amounts.",
    "german_credit": "German Credit dataset. Classifies a loan applicant as good or bad "
                      "credit risk from financial and demographic attributes.",
    "heart_disease": "Cleveland Heart Disease dataset. Predicts presence of heart "
                      "disease from clinical measurements (cholesterol, chest pain "
                      "type, resting ECG, etc.).",
    "hyperack": "DCLab's proprietary ride/delivery order dataset. Predicts whether a "
                "driver will accept an order at the moment it is offered.",
    "mushroom": "UCI Mushroom dataset. Classifies a mushroom as edible or poisonous "
                "from physical characteristics (cap shape, odor, gill size, etc.).",
    "online_shoppers": "Online Shoppers Purchasing Intention dataset — session-level web "
                        "analytics. Predicts whether a browsing session ends in a "
                        "purchase.",
    "spambase": "Spambase dataset — word/character frequency features extracted from "
                "emails. Classifies an email as spam or not spam.",
    "wine_quality": "Wine Quality dataset — physicochemical tests on wine samples "
                     "(acidity, sugar, alcohol, etc.). Predicts a quality rating.",
}

# --- 2. Stage cards: what each pipeline stage does, from workflow_blocks.json -
STAGE_CARDS = {
    "data_understanding": (
        "This stage answers 'what is actually in the data before any modeling?'. "
        "It runs a schema/type audit, profiles the target rate and missingness, "
        "checks for duplicate or overlapping entities between splits, looks at "
        "distribution slices, and proposes association hypotheses WITHOUT yet "
        "promoting any of them into a model."
    ),
    "leakage_audit": (
        "This stage builds a semantic availability matrix (is each column knowable "
        "at the declared prediction moment?), checks lineage/timestamps, reviews "
        "target-proxy candidates, audits the split and preprocessing for leakage, "
        "and runs a safe-vs-unsafe ablation to measure how much apparent score is "
        "being bought by information that shouldn't be available."
    ),
    "feature_engineering": (
        "This stage runs a feature ladder: starting from a raw baseline, it tests "
        "single-domain transforms, ratios/differences, and bounded interactions one "
        "rung at a time, each compared to the previous rung on identical folds, "
        "keeping a new feature set only if it produces a reproducible lift."
    ),
    "model_selection": (
        "This stage screens algorithm families side by side on identical folds — "
        "typically a dummy floor, a regularized linear model, bagged trees, boosting "
        "families, and a domain-appropriate specialist — then ranks them together "
        "rather than tuning any single model in isolation."
    ),
    "optimization_reliability": (
        "This stage takes the screened model, runs a small explicit hyperparameter "
        "search validated with paired-fold comparisons, consumes the locked holdout "
        "exactly once, then stress-tests the result: calibration, threshold/cost "
        "analysis, behavior under missing inputs, subgroup performance floors, and "
        "production-readiness of the top features."
    ),
}


def dataset_card(dataset: str, meta_dir: Path) -> str:
    card = DATASET_CARDS.get(dataset, f"({dataset} — no standard description on file; treat as an unfamiliar dataset and rely only on the setup summary below.)")
    meta_path = meta_dir / dataset / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        size_bits = []
        if meta.get("n_rows"):
            size_bits.append(f"{meta['n_rows']:,} rows")
        if meta.get("n_features"):
            size_bits.append(f"{meta['n_features']} features")
        if meta.get("pos_rate") is not None:
            size_bits.append(f"positive rate {meta['pos_rate']:.3f}")
        if size_bits:
            card += " (" + ", ".join(size_bits) + ")."
    return card


def _fmt_pct(x):
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)


def summarize_setup(result: dict) -> str:
    """Pull the ACTUAL run configuration out of `evidence`, per stage kind.
    Every number here comes straight from the experiment JSON — nothing invented."""
    kind = result.get("kind")
    ev = result.get("evidence", {}) or {}
    lines = []

    if kind == "data_understanding":
        lines.append(f"Rows analyzed: {ev.get('analyzed_rows')} (train {ev.get('train_rows')}, holdout {ev.get('holdout_rows')}, source {ev.get('source_rows')}).")
        lines.append(f"Feature count: {ev.get('feature_count')}; flagged as decision-time-risky: {ev.get('risk_feature_count')}.")
        lines.append(f"Target positive rate (train): {_fmt_pct(ev.get('positive_rate_train'))}. Missing-cell rate (train): {_fmt_pct(ev.get('missing_cell_rate_train'))}. Duplicate-row rate (train): {_fmt_pct(ev.get('duplicate_row_rate_train'))}.")

    elif kind == "leakage_audit":
        lines.append(f"Synthetic exact-target canary caught by the detector: {ev.get('detector_canary_passed')}.")
        declared = ev.get("declared_leakage_features") or []
        lines.append(f"Declared decision-time exclusions going in: {declared if declared else 'none'}.")
        if ev.get("policy_rationale"):
            lines.append(f"Policy rationale: {ev['policy_rationale']}")
        svu = ev.get("safe_vs_unsafe_training_cv")
        if svu and isinstance(svu, dict):
            safe_auc = (svu.get("safe") or {}).get("metrics", {}).get("roc_auc", {}).get("mean")
            unsafe_auc = (svu.get("unsafe") or {}).get("metrics", {}).get("roc_auc", {}).get("mean")
            if safe_auc is not None and unsafe_auc is not None:
                lines.append(f"Safe-vs-unsafe training CV ROC-AUC: safe={_fmt_pct(safe_auc)}, unsafe={_fmt_pct(unsafe_auc)} (apparent lift {_fmt_pct(unsafe_auc - safe_auc)}).")

    elif kind == "feature_engineering":
        stages = ev.get("stage_results", [])
        if stages:
            model = stages[0].get("model", "?")
            folds = stages[0].get("folds", "?")
            lines.append(f"{len(stages)} feature recipes compared on {model} with {folds}-fold CV:")
            for s in stages:
                auc = s.get("metrics", {}).get("roc_auc", {}).get("mean")
                lines.append(f"  - {s.get('stage')}: {s.get('feature_count_mean')} features, mean ROC-AUC {_fmt_pct(auc)}")
        lines.append(f"Selection rule: {ev.get('selection_rule')}")
        lines.append(f"Selected recipe: {ev.get('selected_stage')} (vs. raw baseline delta: {_fmt_pct(ev.get('selected_vs_raw_auc'))}).")

    elif kind == "model_selection":
        models = ev.get("model_results", [])
        if models:
            feat_stage = ev.get("feature_stage", "?")
            lines.append(f"{len(models)} model families screened on the '{feat_stage}' feature set, identical folds:")
            for m in models:
                auc = m.get("metrics", {}).get("roc_auc", {}).get("mean")
                lines.append(f"  - {m.get('model')}: {m.get('feature_count_mean')} features, mean ROC-AUC {_fmt_pct(auc)}, {_fmt_pct(m.get('elapsed_seconds'))}s")
        lines.append(f"Selection rule: {ev.get('selection_rule')}")
        lines.append(f"Selected model: {ev.get('selected_model')} (mean ROC-AUC {_fmt_pct(ev.get('selected_auc_mean'))} +/- {_fmt_pct(ev.get('selected_auc_std'))}).")

    elif kind == "optimization_reliability":
        lines.append(f"Model under optimization: {ev.get('model')}, feature set: {ev.get('feature_stage')}.")
        if ev.get("optimized_vs_baseline_cv_auc") is not None:
            lines.append(f"Optimized vs. baseline training-CV ROC-AUC delta: {_fmt_pct(ev.get('optimized_vs_baseline_cv_auc'))}.")
        lines.append(f"Holdout consumed: {ev.get('holdout_consumed')}.")
        hm = ev.get("holdout_metrics") or {}
        if hm.get("roc_auc") is not None:
            lines.append(f"Final holdout ROC-AUC: {_fmt_pct(hm.get('roc_auc'))} (accuracy {_fmt_pct(hm.get('accuracy'))}, F1 {_fmt_pct(hm.get('f1'))}, Brier {_fmt_pct(hm.get('brier'))}).")
        ci = ev.get("holdout_roc_auc_bootstrap_ci") or {}
        if ci:
            lines.append(f"Bootstrap 95% CI on holdout ROC-AUC: [{_fmt_pct(ci.get('low'))}, {_fmt_pct(ci.get('high'))}] over {ci.get('draws')} draws.")
        top_feats = ev.get("top_final_features") or []
        if top_feats:
            top_str = ", ".join(f"{f['feature']} ({_fmt_pct(f['normalized_importance'])})" for f in top_feats[:5])
            lines.append(f"Top features by normalized importance: {top_str}.")
        if ev.get("production_feature_rule"):
            lines.append(f"Production feature rule applied: {ev['production_feature_rule']}")

    else:
        lines.append("(No structured setup extractor for this stage kind yet — fell back to claims only.)")

    return "\n".join(lines)


def format_reasoning_answer(review: dict) -> str:
    parts = []
    evidence = review.get("observed_evidence") or []
    if evidence:
        parts.append("Observed evidence:\n" + "\n".join(f"- {e['statement']}" for e in evidence if e.get("statement")))
    if review.get("interpretation"):
        parts.append(f"Interpretation and uncertainty:\n{review['interpretation']}")
    if review.get("decision"):
        parts.append(f"Decision / recommendation:\n{review['decision']}")
    risks = review.get("risks_and_missing_evidence") or []
    if risks:
        parts.append("Risks and missing evidence:\n" + "\n".join(f"- {r}" for r in risks))
    nxt = review.get("next_experiment") or {}
    if nxt:
        parts.append(
            "Smallest falsifiable next experiment:\n"
            f"Title: {nxt.get('title', '')}\n"
            f"Hypothesis: {nxt.get('hypothesis', '')}\n"
            f"Success gate: {nxt.get('success_gate', '')}\n"
            f"Why it's the smallest useful step: {nxt.get('why_smallest', '')}"
        )
    return "\n\n".join(parts).strip()


def claims_answer(claims: list) -> str:
    facts = [c["statement"] for c in claims if c.get("kind") == "fact"]
    decisions = [c["statement"] for c in claims if c.get("kind") in ("decision", "recommendation")]
    limitations = []
    for c in claims:
        limitations.extend(c.get("limitations", []))
    parts = []
    if facts:
        parts.append("Observed evidence:\n" + "\n".join(f"- {s}" for s in facts))
    if decisions:
        parts.append("Decision / recommendation:\n" + "\n".join(f"- {s}" for s in decisions))
    if limitations:
        parts.append("Known limitations:\n" + "\n".join(f"- {s}" for s in sorted(set(limitations))))
    return "\n\n".join(parts).strip()


def build_example(result: dict, meta_dir: Path):
    dataset = result.get("dataset", "unknown")
    kind = result.get("kind", "unknown")
    question = (result.get("question") or "").strip()
    hypothesis = (result.get("hypothesis") or "").strip()

    d_card = dataset_card(dataset, meta_dir)
    s_card = STAGE_CARDS.get(kind, f"(No standard description on file for stage '{kind}'.)")
    setup = summarize_setup(result)

    review_block = result.get("llm_review") or {}
    if review_block.get("status") == "completed" and review_block.get("review"):
        answer = format_reasoning_answer(review_block["review"])
        source = "llm_reasoning_trace_grounded"
    else:
        answer = claims_answer(result.get("claims") or [])
        source = "claims_grounded"

    if not answer:
        return None

    user = (
        f"Dataset ({dataset}): {d_card}\n\n"
        f"Pipeline stage ({kind.replace('_', ' ')}): {s_card}\n\n"
        f"This run's setup:\n{setup}\n\n"
        f"Question: {question}"
    )
    if hypothesis:
        user += f"\nWorking hypothesis going in: {hypothesis}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "source": source,
            "experiment_id": result.get("experiment_id"),
            "dataset": dataset,
            "stage": kind,
        },
    }


def dedup(examples):
    seen, unique = set(), []
    for ex in examples:
        digest = hashlib.sha256(json.dumps(ex["messages"], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(ex)
    return unique


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("evidence/campaigns/model_building_50_v1/results"))
    parser.add_argument("--meta-dir", type=Path, default=Path("data/public"))
    parser.add_argument("--out-dir", type=Path, default=Path("research/llm-fine-tuning/experiments/sft/prototype_out_v3"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result_files = sorted(args.results_dir.glob("*.json"))
    if not result_files:
        raise SystemExit(f"No result files found under {args.results_dir}")

    examples = []
    skipped = 0
    for fp in result_files:
        result = json.loads(fp.read_text(encoding="utf-8"))
        ex = build_example(result, args.meta_dir)
        if ex:
            examples.append(ex)
        else:
            skipped += 1

    examples = dedup(examples)
    random.seed(args.seed)
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_fraction)) if examples else 0
    val_set, train_set = examples[:n_val], examples[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "sft_train_v3.jsonl", "w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(args.out_dir / "sft_val_v3.jsonl", "w", encoding="utf-8") as f:
        for ex in val_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Result files scanned : {len(result_files)}")
    print(f"Examples built       : {len(examples)} (skipped {skipped})")
    print(f"Train / Val          : {len(train_set)} / {len(val_set)}")
    print(f"Written to: {args.out_dir}/sft_train_v3.jsonl and {args.out_dir}/sft_val_v3.jsonl")


if __name__ == "__main__":
    main()
