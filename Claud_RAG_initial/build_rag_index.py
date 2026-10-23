#!/usr/bin/env python3
"""
build_rag_index.py
====================
A working retrieval demo over DCLab's existing evidence files — structured
retrieval, not naive document-chunk RAG. Runs immediately, offline (TF-IDF,
no model download needed) so you can verify the *retrieval design* today;
swap in real embeddings later (one function to change, see bottom).

WHY STRUCTURED RETRIEVAL, NOT "CHUNK THE DOCS AND EMBED THEM"
---------------------------------------------------------------
Naive RAG chunks big documents into ~500-token windows and finds the
closest ones by embedding similarity. That's the right tool when you have
thousands of pages of unstructured text and don't know what shape the
answer will take.

Your knowledge base isn't that. Every unit of knowledge here is already a
small, well-typed record: a rule has a category, a statement, a why, a
confidence. A claim has a dataset, a stage, a kind, an evidence list. A
workflow block has a name and an ordered flow. Chunking these into 500-
token windows would slice a rule's "statement" away from its "why" and
"failure_signal" mid-record — actively destroying structure you already
have for free.

The right retrieval unit here is ONE RECORD (one rule, one claim-group,
one workflow block), each carrying real metadata. That means retrieval
is two steps, not one:

  1. FILTER (deterministic, cheap, exact): dataset=? stage=? category=?
     confidence=? — this alone often narrows thousands of records to a
     handful, with zero ambiguity.
  2. RANK (fuzzy, similarity-based) only within what survives the filter.

This is "hybrid retrieval": exact metadata filtering first, embedding-
style similarity second. For a KB this structured, skipping step 1 and
doing pure vector search over everything is strictly worse: it can surface
a highly-worded but irrelevant record from the wrong dataset ahead of an
exact match filtered on dataset alone.

Usage
-----
    python build_rag_index.py --rules knowledge/model_building_rules.jsonl \
        --claims campaigns/model_building_50_v1/agent_memory.jsonl \
        --blocks knowledge/workflow_blocks.json

Then, in your own code:
    records = load_all(...)
    index = RetrievalIndex(records)
    hits = index.search("is duration safe to use for bank_marketing", filters={"dataset": "bank_marketing"}, k=3)
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Record:
    id: str
    text: str                     # what gets embedded / matched against
    display: str                  # what gets shown to the agent/model once retrieved
    metadata: dict = field(default_factory=dict)


def load_jsonl(path: Path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def rules_to_records(rules) -> list:
    out = []
    for r in rules:
        text = f"{r['category']} {r['statement']} {r['why']} {r['failure_signal']}"
        display = (
            f"[Rule {r['rule_id']} | {r['category']} | confidence={r['confidence']}]\n"
            f"{r['statement']}\nWhy: {r['why']}\nFailure signal: {r['failure_signal']}\n"
            f"Next test: {r['next_test']}"
        )
        out.append(Record(
            id=r["rule_id"], text=text, display=display,
            metadata={"kind": "rule", "category": r["category"], "confidence": r["confidence"]},
        ))
    return out


def blocks_to_records(blocks) -> list:
    out = []
    for b in blocks:
        text = f"{b['name']} " + " ".join(b["flow"])
        display = f"[Workflow {b['id']} | {b['name']}]\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(b["flow"]))
        out.append(Record(
            id=b["id"], text=text, display=display,
            metadata={"kind": "workflow_block", "name": b["name"]},
        ))
    return out


def claims_to_records(claims) -> list:
    grouped = defaultdict(list)
    for c in claims:
        grouped[(c["experiment_id"], c["dataset"], c["kind"])].append(c["claim"])

    out = []
    for (exp_id, dataset, stage), claim_list in grouped.items():
        statements = [c["statement"] for c in claim_list]
        text = f"{dataset} {stage} " + " ".join(statements)
        display = (
            f"[Experiment {exp_id} | dataset={dataset} | stage={stage}]\n"
            + "\n".join(f"- ({c['kind']}) {c['statement']}" for c in claim_list)
        )
        out.append(Record(
            id=exp_id, text=text, display=display,
            metadata={"kind": "experiment_claim", "dataset": dataset, "stage": stage},
        ))
    return out


class RetrievalIndex:
    def __init__(self, records: list):
        self.records = records
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([r.text for r in records])

    def _passes_filters(self, record: Record, filters: dict) -> bool:
        return all(record.metadata.get(k) == v for k, v in filters.items())

    def search(self, query: str, filters: dict = None, k: int = 5):
        filters = filters or {}
        candidate_idx = [i for i, r in enumerate(self.records) if self._passes_filters(r, filters)]
        if not candidate_idx:
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix[candidate_idx]).flatten()
        ranked = sorted(zip(candidate_idx, sims), key=lambda x: x[1], reverse=True)[:k]
        return [(self.records[i], float(score)) for i, score in ranked]


def load_all(rules_path: Path, claims_path: Path, blocks_path: Path) -> list:
    records = []
    if rules_path.exists():
        records += rules_to_records(load_jsonl(rules_path))
    if blocks_path.exists():
        records += blocks_to_records(json.loads(blocks_path.read_text(encoding="utf-8")))
    if claims_path.exists():
        records += claims_to_records(load_jsonl(claims_path))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rules", type=Path, default=Path("knowledge/model_building_rules.jsonl"))
    parser.add_argument("--claims", type=Path, default=Path("campaigns/model_building_50_v1/agent_memory.jsonl"))
    parser.add_argument("--blocks", type=Path, default=Path("knowledge/workflow_blocks.json"))
    args = parser.parse_args()

    records = load_all(args.rules, args.claims, args.blocks)
    print(f"Loaded {len(records)} retrievable records "
          f"({sum(1 for r in records if r.metadata['kind']=='rule')} rules, "
          f"{sum(1 for r in records if r.metadata['kind']=='workflow_block')} workflow blocks, "
          f"{sum(1 for r in records if r.metadata['kind']=='experiment_claim')} experiment-claim groups)")

    index = RetrievalIndex(records)

    demo_queries = [
        ("is it safe to use call duration as a feature", {"dataset": "bank_marketing"}),
        ("how do I know a feature is production ready", {}),
        ("what does the leakage audit stage actually check", {"kind": "workflow_block"}),
    ]
    for query, filters in demo_queries:
        print(f"\n=== Query: '{query}' | filters={filters} ===")
        for record, score in index.search(query, filters=filters, k=2):
            print(f"  score={score:.3f} | {record.display.splitlines()[0]}")


if __name__ == "__main__":
    main()
