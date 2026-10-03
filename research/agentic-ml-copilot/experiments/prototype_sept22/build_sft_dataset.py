#!/usr/bin/env python3
"""
build_sft_dataset.py
=====================

Converts DCLab's existing evidence artifacts (rules, workflow blocks, and
per-experiment claims) into a CLEAN instruction-tuning (SFT) dataset that a
small language model (SLM) can actually be fine-tuned on.

Why this exists
----------------
The repo already produces excellent *retrieval* artifacts:
  - evidence/knowledge/model_building_rules.jsonl   (methodological + empirical rules)
  - evidence/knowledge/workflow_blocks.json         (reusable process flows)
  - evidence/campaigns/model_building_50_v1/agent_memory.jsonl (per-experiment claims)

These are great for RAG / an agent's long-term memory, but they are NOT yet
in a shape a fine-tuning job expects: paired (question, answer) turns with no
provenance noise (git commit hashes, package versions, file paths) sitting
inside the text the model has to learn to imitate.

This script:
  1. Reads the three source files.
  2. Turns each rule into 2 clean Q&A pairs (the rule itself, and a
     "how would I notice I broke this rule" framing).
  3. Turns each workflow block into 1 clean Q&A pair.
  4. Groups per-experiment claims by (experiment_id, dataset, stage) and
     turns each group into 1 clean Q&A pair, stripping provenance/hashes.
  5. Deduplicates, shuffles (seed=42), and writes a 90/10 train/val split
     in OpenAI-style chat JSONL — the format most SFT frameworks
     (OpenAI fine-tuning, Axolotl, LLaMA-Factory, TRL/SFTTrainer) accept
     directly or with a one-line loader change.

Usage
-----
    python build_sft_dataset.py \
        --rules evidence/knowledge/model_building_rules.jsonl \
        --blocks evidence/knowledge/workflow_blocks.json \
        --claims evidence/campaigns/model_building_50_v1/agent_memory.jsonl \
        --out-dir sft_out

Run it from the root of the Research_Development_DCLab checkout, or pass
explicit paths. Re-run after every new experiment / campaign cycle so the
fine-tuning corpus grows alongside the evidence registry.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a senior data-science / ML engineering research assistant. "
    "You answer using the DCLab evidence-first model-building framework: "
    "state the observed evidence, your interpretation and its uncertainty, "
    "a concrete decision or recommendation, remaining risks or missing "
    "evidence, and the smallest next falsifiable test."
)

NEUTRAL_EXCEPTION_TEXT = "none; apply judgment to the prediction contract."


def load_jsonl(path: Path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def chat_example(user: str, assistant: str, meta: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "metadata": meta,
    }


def rule_examples(rules):
    examples = []
    for r in rules:
        category = r["category"].replace("_", " ")
        statement = r["statement"]
        why = r["why"]
        failure_signal = r["failure_signal"]
        next_test = r["next_test"]
        confidence = r["confidence"]
        exceptions = r.get("exceptions", "")

        # Variant 1: "what is the rule and why"
        q1 = (
            f"In the '{category}' stage of building a tabular ML model, "
            "what does evidence-based practice recommend, and why does it matter?"
        )
        a1 = (
            f"Rule ({confidence} confidence): {statement}\n\n"
            f"Why it matters: {why}\n\n"
            f"How you'd know it was violated: {failure_signal}\n\n"
            f"How to test it: {next_test}"
        )
        if exceptions and exceptions.strip().lower() != NEUTRAL_EXCEPTION_TEXT:
            a1 += f"\n\nExceptions: {exceptions}"
        examples.append(
            chat_example(
                q1, a1,
                {"source": "model_building_rules", "rule_id": r["rule_id"], "category": r["category"]},
            )
        )

        # Variant 2: diagnostic framing (useful for an agent debugging a pipeline)
        q2 = (
            f"I suspect my pipeline is cutting corners on '{category}'. "
            "What would that look like in practice, and how do I check?"
        )
        a2 = f"{failure_signal} To check it directly: {next_test} Underlying rule: {statement}"
        examples.append(
            chat_example(
                q2, a2,
                {"source": "model_building_rules_diagnostic", "rule_id": r["rule_id"], "category": r["category"]},
            )
        )
    return examples


def workflow_examples(blocks):
    examples = []
    for b in blocks:
        steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(b["flow"]))
        q = f"What is the reusable workflow for the '{b['name']}' stage of model building?"
        a = f"The '{b['name']}' stage ({b['id']}) follows this flow:\n{steps}"
        examples.append(chat_example(q, a, {"source": "workflow_blocks", "id": b["id"]}))
    return examples


def claim_examples(claims):
    grouped = defaultdict(list)
    for row in claims:
        key = (row["experiment_id"], row["dataset"], row["kind"])
        grouped[key].append(row["claim"])

    examples = []
    for (exp_id, dataset, stage), claim_list in grouped.items():
        facts = [c["statement"] for c in claim_list if c.get("kind") == "fact"]
        risks = [c["statement"] for c in claim_list if c.get("kind") == "risk"]
        other = [
            c["statement"] for c in claim_list
            if c.get("kind") not in ("fact", "risk")
        ]
        limitations = []
        for c in claim_list:
            limitations.extend(c.get("limitations", []))

        parts = []
        if facts:
            parts.append("Observed evidence:\n" + "\n".join(f"- {s}" for s in facts))
        if risks:
            parts.append("Risks / caveats flagged:\n" + "\n".join(f"- {s}" for s in risks))
        if other:
            parts.append("Other findings:\n" + "\n".join(f"- {s}" for s in other))
        if limitations:
            uniq = sorted(set(limitations))
            parts.append("Known limitations:\n" + "\n".join(f"- {s}" for s in uniq))

        answer = "\n\n".join(parts).strip()
        if not answer:
            continue

        stage_readable = stage.replace("_", " ")
        question = (
            f"What did experiment {exp_id} on the '{dataset}' dataset find "
            f"during the '{stage_readable}' stage?"
        )
        examples.append(
            chat_example(
                question, answer,
                {"source": "agent_memory", "experiment_id": exp_id, "dataset": dataset, "stage": stage},
            )
        )
    return examples


def dedup(examples):
    seen = set()
    unique = []
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
    parser.add_argument("--rules", type=Path, default=Path("evidence/knowledge/model_building_rules.jsonl"))
    parser.add_argument("--blocks", type=Path, default=Path("evidence/knowledge/workflow_blocks.json"))
    parser.add_argument("--claims", type=Path, default=Path("evidence/campaigns/model_building_50_v1/agent_memory.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("research/llm-fine-tuning/experiments/sft/prototype_out"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rules = load_jsonl(args.rules) if args.rules.exists() else []
    blocks = json.loads(args.blocks.read_text(encoding="utf-8")) if args.blocks.exists() else []
    claims = load_jsonl(args.claims) if args.claims.exists() else []

    examples = rule_examples(rules) + workflow_examples(blocks) + claim_examples(claims)
    examples = dedup(examples)

    random.seed(args.seed)
    random.shuffle(examples)

    n_val = max(1, int(len(examples) * args.val_fraction)) if examples else 0
    val_set, train_set = examples[:n_val], examples[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "sft_train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(args.out_dir / "sft_val.jsonl", "w", encoding="utf-8") as f:
        for ex in val_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    by_source = defaultdict(int)
    for ex in examples:
        by_source[ex["metadata"]["source"]] += 1

    print(f"Total examples : {len(examples)}")
    print(f"Train / Val    : {len(train_set)} / {len(val_set)}")
    print("By source:")
    for src, n in sorted(by_source.items()):
        print(f"  {src:30s} {n}")
    print(f"\nWritten to: {args.out_dir}/sft_train.jsonl and {args.out_dir}/sft_val.jsonl")


if __name__ == "__main__":
    main()
