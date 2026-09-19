#!/usr/bin/env python3
"""
build_sft_dataset.py
=====================

Converts DCLab's existing evidence artifacts into a CLEAN instruction-tuning
(SFT) dataset that a small language model (SLM) can be fine-tuned on.

Sources (all optional if the path is missing)
---------------------------------------------
  A. Knowledge / campaign (this script's original job)
     - knowledge/model_building_rules.jsonl
     - knowledge/workflow_blocks.json
     - campaigns/model_building_50_v1/agent_memory.jsonl

  B. Agentic Research Studio (additive; does not replace A)
     - agent_runs/clean_exports/sft_chat.jsonl
       Tasks: plan_agenda, review_data_leakage, propose_experiment,
       critique_experiment, synthesize_lessons
     Generate / refresh with:
       .venv-agent/bin/python -m dclab_rnd.agentic export-clean

These are complementary:
  A teaches *rules, workflows, and campaign claims* (what to believe).
  B teaches *live research loop behavior* (how to propose/critique/synthesize
  from compact trial evidence). Raw trial-*/OOF files are never copied into
  the SFT text.

Usage
-----
    python cluad_initial_reports/build_sft_dataset.py
    python cluad_initial_reports/build_sft_dataset.py --out-dir sft_out
    python cluad_initial_reports/build_sft_dataset.py --no-studio
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
                q1,
                a1,
                {"source": "model_building_rules", "rule_id": r["rule_id"], "category": r["category"]},
            )
        )

        q2 = (
            f"I suspect my pipeline is cutting corners on '{category}'. "
            "What would that look like in practice, and how do I check?"
        )
        a2 = f"{failure_signal} To check it directly: {next_test} Underlying rule: {statement}"
        examples.append(
            chat_example(
                q2,
                a2,
                {
                    "source": "model_building_rules_diagnostic",
                    "rule_id": r["rule_id"],
                    "category": r["category"],
                },
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
        other = [c["statement"] for c in claim_list if c.get("kind") not in ("fact", "risk")]
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
                question,
                answer,
                {
                    "source": "agent_memory",
                    "experiment_id": exp_id,
                    "dataset": dataset,
                    "stage": stage,
                },
            )
        )
    return examples


def studio_examples(records):
    """Normalize agentic clean_exports/sft_chat.jsonl into the same chat schema."""
    examples = []
    for row in records:
        messages = row.get("messages") or []
        if len(messages) < 3:
            continue
        user = next((m["content"] for m in messages if m.get("role") == "user"), None)
        assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), None)
        if not user or not assistant:
            continue
        if isinstance(user, (dict, list)):
            user = json.dumps(user, ensure_ascii=False)
        if isinstance(assistant, (dict, list)):
            assistant = json.dumps(assistant, ensure_ascii=False)
        source = row.get("source") or {}
        examples.append(
            chat_example(
                user,
                assistant,
                {
                    "source": f"studio_{row.get('task') or 'agentic'}",
                    "task": row.get("task"),
                    "run_id": source.get("run_id"),
                    "label": source.get("label"),
                    "datasets": source.get("datasets"),
                    "id": row.get("id"),
                },
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
    parser.add_argument("--rules", type=Path, default=ROOT / "knowledge/model_building_rules.jsonl")
    parser.add_argument("--blocks", type=Path, default=ROOT / "knowledge/workflow_blocks.json")
    parser.add_argument(
        "--claims",
        type=Path,
        default=ROOT / "campaigns/model_building_50_v1/agent_memory.jsonl",
    )
    parser.add_argument(
        "--studio-sft",
        type=Path,
        default=ROOT / "agent_runs/clean_exports/sft_chat.jsonl",
        help="Agentic Research Studio clean SFT JSONL (from export-clean)",
    )
    parser.add_argument("--no-studio", action="store_true", help="Skip studio agentic examples")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "sft_out")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rules = load_jsonl(args.rules) if args.rules.exists() else []
    blocks = json.loads(args.blocks.read_text(encoding="utf-8")) if args.blocks.exists() else []
    claims = load_jsonl(args.claims) if args.claims.exists() else []
    studio = []
    if not args.no_studio and args.studio_sft.exists():
        studio = load_jsonl(args.studio_sft)

    examples = (
        rule_examples(rules)
        + workflow_examples(blocks)
        + claim_examples(claims)
        + studio_examples(studio)
    )
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

    manifest = {
        "total": len(examples),
        "train": len(train_set),
        "val": len(val_set),
        "by_source": dict(sorted(by_source.items())),
        "inputs": {
            "rules": str(args.rules) if args.rules.exists() else None,
            "blocks": str(args.blocks) if args.blocks.exists() else None,
            "claims": str(args.claims) if args.claims.exists() else None,
            "studio_sft": str(args.studio_sft)
            if (not args.no_studio and args.studio_sft.exists())
            else None,
        },
    }
    (args.out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Total examples : {len(examples)}")
    print(f"Train / Val    : {len(train_set)} / {len(val_set)}")
    print("By source:")
    for src, n in sorted(by_source.items()):
        print(f"  {src:40s} {n}")
    print(f"\nWritten to: {args.out_dir}/sft_train.jsonl and {args.out_dir}/sft_val.jsonl")


if __name__ == "__main__":
    main()
