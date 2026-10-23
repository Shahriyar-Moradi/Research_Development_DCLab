#!/usr/bin/env python3
"""
build_sft_dataset_v2.py
========================
ADDITIVE upgrade — does not replace or remove `build_sft_dataset.py` (v1).
Run both; concatenate their JSONL outputs, or mix them at training time.

What v1 did
-----------
Turned the *summary* knowledge artifacts (model_building_rules.jsonl,
workflow_blocks.json, agent_memory.jsonl claims) into clean Q&A pairs.
Good for teaching facts and rules.

What v2 adds
------------
Reads the FULL per-experiment result files under
`campaigns/model_building_50_v1/results/*.json` and extracts the genuine
five-part reasoning trace DCLab's own LLM critic already produced for each
one (the `llm_review.review` block), in exactly the structure this repo's
own `AGENT_CONTEXT.md` mandates:

    1. Observed evidence (with citations)
    2. Interpretation and uncertainty
    3. Decision or recommendation
    4. Risks and missing evidence
    5. Smallest falsifiable next experiment

This is the single highest-value addition for fine-tuning an SLM: it
teaches the REASONING PATTERN (how to move from evidence to a cautious,
falsifiable decision), not just isolated facts. For experiments that don't
yet have a completed `llm_review` (fresh runs, or before
`make rd-campaign-review` has been run on them), it falls back to a
claims-only example so no experiment's evidence is wasted.

Usage
-----
    python build_sft_dataset_v2.py \
        --results-dir campaigns/model_building_50_v1/results \
        --out-dir sft_out_v2

Re-run after every `make rd-campaign-review` pass, and after Round 2
datasets are added — this script needs no changes to pick up new
experiments; it just globs the results directory.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a senior data-science / ML engineering research assistant trained on "
    "DCLab's evidence-first model-building methodology. For every question, answer "
    "using exactly this structure: (1) Observed evidence, citing what supports each "
    "point; (2) Interpretation and uncertainty; (3) Decision or recommendation; "
    "(4) Risks and missing evidence; (5) The smallest falsifiable next experiment."
)


def format_reasoning_answer(review: dict) -> str:
    parts = []

    evidence = review.get("observed_evidence") or []
    if evidence:
        lines = [f"- {e['statement']}" for e in evidence if e.get("statement")]
        if lines:
            parts.append("Observed evidence:\n" + "\n".join(lines))

    interpretation = review.get("interpretation")
    if interpretation:
        parts.append(f"Interpretation and uncertainty:\n{interpretation}")

    decision = review.get("decision")
    if decision:
        parts.append(f"Decision / recommendation:\n{decision}")

    risks = review.get("risks_and_missing_evidence") or []
    if risks:
        parts.append("Risks and missing evidence:\n" + "\n".join(f"- {r}" for r in risks))

    next_exp = review.get("next_experiment") or {}
    if next_exp:
        block = (
            f"Title: {next_exp.get('title', '')}\n"
            f"Hypothesis: {next_exp.get('hypothesis', '')}\n"
            f"Success gate: {next_exp.get('success_gate', '')}\n"
            f"Why it's the smallest useful step: {next_exp.get('why_smallest', '')}"
        )
        parts.append("Smallest falsifiable next experiment:\n" + block)

    return "\n\n".join(parts).strip()


def reasoning_example(result: dict):
    review_block = result.get("llm_review") or {}
    if review_block.get("status") != "completed":
        return None
    review = review_block.get("review")
    if not review:
        return None
    answer = format_reasoning_answer(review)
    if not answer:
        return None

    dataset = result.get("dataset", "unknown")
    stage = (result.get("kind") or "unknown").replace("_", " ")
    question = (result.get("question") or "").strip()
    hypothesis = (result.get("hypothesis") or "").strip()

    user = f"Dataset: {dataset}. Stage: {stage}.\nQuestion: {question}"
    if hypothesis:
        user += f"\nWorking hypothesis going in: {hypothesis}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "source": "llm_reasoning_trace",
            "experiment_id": result.get("experiment_id"),
            "dataset": dataset,
            "stage": result.get("kind"),
        },
    }


def claims_fallback_example(result: dict):
    """Used only when no completed llm_review exists yet for this experiment."""
    claims = result.get("claims") or []
    if not claims:
        return None

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

    answer = "\n\n".join(parts).strip()
    if not answer:
        return None

    dataset = result.get("dataset", "unknown")
    stage = (result.get("kind") or "unknown").replace("_", " ")
    question = (result.get("question") or "").strip()
    user = f"Dataset: {dataset}. Stage: {stage}.\nQuestion: {question}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "source": "claims_fallback",
            "experiment_id": result.get("experiment_id"),
            "dataset": dataset,
            "stage": result.get("kind"),
        },
    }


def dedup(examples):
    seen, unique = set(), []
    for ex in examples:
        digest = hashlib.sha256(
            json.dumps(ex["messages"], sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(ex)
    return unique


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("campaigns/model_building_50_v1/results"))
    parser.add_argument("--out-dir", type=Path, default=Path("sft_out_v2"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result_files = sorted(args.results_dir.glob("*.json"))
    if not result_files:
        raise SystemExit(f"No result files found under {args.results_dir}")

    examples = []
    n_with_review = n_fallback = n_skipped = 0
    for fp in result_files:
        result = json.loads(fp.read_text(encoding="utf-8"))
        ex = reasoning_example(result)
        if ex:
            examples.append(ex)
            n_with_review += 1
            continue
        ex = claims_fallback_example(result)
        if ex:
            examples.append(ex)
            n_fallback += 1
        else:
            n_skipped += 1

    examples = dedup(examples)
    random.seed(args.seed)
    random.shuffle(examples)

    n_val = max(1, int(len(examples) * args.val_fraction)) if examples else 0
    val_set, train_set = examples[:n_val], examples[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "sft_train_v2.jsonl", "w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(args.out_dir / "sft_val_v2.jsonl", "w", encoding="utf-8") as f:
        for ex in val_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Result files scanned     : {len(result_files)}")
    print(f"Full reasoning-trace exs : {n_with_review}")
    print(f"Claims-only fallback exs : {n_fallback}")
    print(f"Skipped (no usable data) : {n_skipped}")
    print(f"Total after dedup        : {len(examples)}")
    print(f"Train / Val              : {len(train_set)} / {len(val_set)}")
    print(f"Written to: {args.out_dir}/sft_train_v2.jsonl and {args.out_dir}/sft_val_v2.jsonl")


if __name__ == "__main__":
    main()
