#!/usr/bin/env python3
"""
grounded_review.py
====================
The "use it today, no training" path. Combines the retrieval index
(build_rag_index.py) with a prompt template matching your own
AGENT_CONTEXT.md answer contract, and either prints the ready-to-send
prompt (dry run — no API key needed, safe to demo right now) or sends it
to an OpenAI-compatible chat endpoint if OPENAI_API_KEY is set.

This is a drop-in replacement for wherever your pipeline currently builds
an LLM-checkpoint prompt by hand: same inputs (dataset, stage, evidence),
richer output (grounded in retrieved rules + similar past experiments),
zero model training required.

Usage
-----
    # Dry run — prints the assembled prompt, no API call, no key needed
    python grounded_review.py --dataset bank_marketing --stage leakage_audit \
        --question "Is the duration column safe to use in production?"

    # Live call (needs OPENAI_API_KEY in the environment, and `pip install openai`)
    OPENAI_API_KEY=sk-... python grounded_review.py --dataset bank_marketing \
        --stage leakage_audit --question "..." --live --model gpt-4o-mini
"""

import argparse
import json
import os
from pathlib import Path

from build_rag_index import RetrievalIndex, load_all

ANSWER_CONTRACT = (
    "You are DCLab's evidence-first model-building critic. Using ONLY the retrieved "
    "context below plus the question, answer in exactly this structure: "
    "(1) Observed evidence, citing which retrieved item supports each point; "
    "(2) Interpretation and uncertainty; (3) Decision or recommendation; "
    "(4) Risks and missing evidence; (5) The smallest falsifiable next experiment. "
    "If the retrieved context doesn't cover something, say so explicitly rather than guessing."
)


def assemble_prompt(index: RetrievalIndex, dataset: str, stage: str, question: str, k: int = 5):
    hits = []
    # Rule 1: anything tagged for this exact dataset+stage first (dataset-specific evidence beats general rules).
    hits += index.search(question, filters={"dataset": dataset, "stage": stage} if dataset and stage else {"dataset": dataset} if dataset else {}, k=k)
    # Rule 2: general rules/workflow blocks regardless of dataset, to fill in methodology context.
    hits += index.search(question, filters={"kind": "rule"}, k=3)
    hits += index.search(question, filters={"kind": "workflow_block"}, k=2)

    seen_ids = set()
    context_blocks = []
    for record, score in hits:
        if record.id in seen_ids:
            continue
        seen_ids.add(record.id)
        context_blocks.append(f"(relevance={score:.3f})\n{record.display}")

    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no matching records found — say so in your answer)"

    user_message = (
        f"Dataset: {dataset or 'unspecified'}. Stage: {stage or 'unspecified'}.\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        f"Question: {question}"
    )
    return ANSWER_CONTRACT, user_message


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--stage", default="")
    parser.add_argument("--question", required=True)
    parser.add_argument("--rules", type=Path, default=Path("evidence/knowledge/model_building_rules.jsonl"))
    parser.add_argument("--claims", type=Path, default=Path("evidence/campaigns/model_building_50_v1/agent_memory.jsonl"))
    parser.add_argument("--blocks", type=Path, default=Path("evidence/knowledge/workflow_blocks.json"))
    parser.add_argument("--live", action="store_true", help="Actually call an OpenAI-compatible chat endpoint instead of a dry run.")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    records = load_all(args.rules, args.claims, args.blocks)
    index = RetrievalIndex(records)
    system_msg, user_msg = assemble_prompt(index, args.dataset, args.stage, args.question)

    if not args.live:
        print("=== DRY RUN (no API call) — this is exactly what would be sent ===\n")
        print("[SYSTEM]\n" + system_msg)
        print("\n[USER]\n" + user_msg)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("--live requires OPENAI_API_KEY to be set in the environment.")

    from openai import OpenAI  # pip install openai
    client = OpenAI()
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
