"""
check_and_merge.py
==================

Runs the same quality checks on every stage's samples, then merges all stages into one
training set at sft_factory/out/all_stages/{train,val,test}.jsonl plus report.json.

Checks
------
1. decision is one of the allowed actions listed in that sample's input
2. every number cited in WHY appears in the input (no invented evidence)
3. no duplicate inputs
4. each dataset appears in exactly one split (train / val / test)
5. answer not given away: the evidence section never contains the chosen action word
   (skipped for model_selection, where the candidates ARE the options being compared)
6. balance: share of the most common decision per stage
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
STAGES = ["data_understanding", "leakage_audit", "feature_engineering", "feature_engineering_balance",
          "model_selection", "optimization_reliability"]
NUM = re.compile(r"[+-]?\d[\d,]*\.\d+%?|[+-]?\d+/5")


def load(stage, split):
    p = OUT / stage / f"{split}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def check(sample):
    user = sample["messages"][1]["content"]
    ans = sample["messages"][2]["content"]
    decision = re.search(r"DECISION: (\S+)", ans).group(1)
    allowed = re.search(r"ALLOWED ACTIONS: (.*)", user).group(1).split(" | ")
    why = ans.split("WHY:")[1].split("RULE:")[0]
    problems = []
    if decision not in allowed:
        problems.append("decision_not_allowed")
    missing = [n for n in NUM.findall(why) if n.lstrip("+") not in user and n not in user]
    if missing:
        problems.append(f"number_not_in_input:{missing[:3]}")
    evidence = user.split("ALLOWED ACTIONS:")[0].split("\n\n", 2)[-1]
    stage = sample["metadata"]["stage"] if "stage" in sample["metadata"] else "leakage_audit"
    if stage != "model_selection" and decision in evidence:
        problems.append("answer_in_evidence")
    return problems


def main():
    report, merged = {}, {"train": [], "val": [], "test": []}
    seen, ds_split = set(), defaultdict(set)
    for stage in STAGES:
        stats = Counter()
        actions = Counter()
        bad_examples = []
        for split in merged:
            for s in load(stage, split):
                s["metadata"].setdefault("stage", stage)
                h = hashlib.sha256(s["messages"][1]["content"].encode()).hexdigest()
                if h in seen:
                    stats["duplicate"] += 1
                    continue
                probs = check(s)
                if probs:
                    stats["rejected"] += 1
                    bad_examples.append((s["metadata"].get("generator"), probs))
                    continue
                seen.add(h)
                ds_split[s["metadata"]["dataset"]].add(split)
                merged[split].append(s)
                actions[s["metadata"]["action"]] += 1
                stats[split] += 1
        total = sum(actions.values())
        report[stage] = {
            "samples": total, "train": stats["train"], "val": stats["val"], "test": stats["test"],
            "rejected": stats["rejected"], "duplicates": stats["duplicate"],
            "by_decision": dict(actions.most_common()),
            "largest_decision_share": round(actions.most_common(1)[0][1] / total, 3) if total else None,
            "rejection_examples": bad_examples[:5],
        }
    report["datasets_in_more_than_one_split"] = [d for d, sp in ds_split.items() if len(sp) > 1]
    report["totals"] = {k: len(v) for k, v in merged.items()}

    rnd = random.Random(0)
    (OUT / "all_stages").mkdir(parents=True, exist_ok=True)
    for split, rows in merged.items():
        rnd.shuffle(rows)
        with open(OUT / "all_stages" / f"{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (OUT / "all_stages" / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
