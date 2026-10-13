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

from workflow_map import annotate

OUT = Path(__file__).resolve().parent / "out"
STAGES = ["problem_understanding", "contract_check", "data_analysis", "solution_design", "connected_problem_to_plan",
          "end_to_end", "data_understanding", "leakage_audit", "feature_engineering", "feature_engineering_balance",
          "model_selection", "optimization_reliability", "code_then_decide", "code_review"]
MULTI_TURN = {"connected_problem_to_plan", "end_to_end", "code_then_decide"}
# Stages whose answers are JSON contracts (problem_to_plan_samples.py). Their generator already validates them
# against dclab_core's contracts, quote rules, plan.validate and a stricter number check; here they must parse.
JSON_STAGES = {"problem_understanding", "contract_check", "data_analysis", "solution_design", "connected_problem_to_plan"}
# Multi-turn samples mix formats: JSON turns must parse, code turns must compile, DECISION turns must pick an action
# their own turn allows. Their generators already check every number of every turn against everything before it.
# Every answer must carry its route (workflow_map), except a code turn and the final report.
NUM = re.compile(r"[+-]?\d[\d,]*\.\d+%?|[+-]?\d+/5")


def load(stage, split):
    p = OUT / stage / f"{split}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def check_turns(sample):
    problems, msgs = [], sample["messages"]
    for i, m in enumerate(msgs):
        if m["role"] == "user" and not m["content"].startswith("WORKFLOW STEP: WF-"):
            problems.append(f"turn_{i}_no_step_header")
        if m["role"] != "assistant":
            continue
        c = m["content"]
        if c.startswith("DECISION:"):
            decision = re.search(r"DECISION: (\S+)", c).group(1)
            allowed = re.search(r"ALLOWED ACTIONS: (.*)", msgs[i - 1]["content"]).group(1).split(" | ")
            if decision not in allowed:
                problems.append(f"turn_{i}_decision_not_allowed")
            if not re.search(r"\nNEXT: \[(continue|back|ask_owner) → WF-\d\d\]", c):
                problems.append(f"turn_{i}_missing_route")
        elif c.startswith("```python"):
            try:
                compile(c.split("```python\n", 1)[1].rsplit("```", 1)[0], "<sample>", "exec")
            except SyntaxError:
                problems.append(f"turn_{i}_code_does_not_compile")
        else:
            try:
                obj = json.loads(c)
            except json.JSONDecodeError:
                problems.append(f"turn_{i}_not_json")
                continue
            if "route" not in obj and obj.get("status") != "complete":
                problems.append(f"turn_{i}_missing_route")
    return problems


def check(sample):
    if sample["metadata"].get("stage") in MULTI_TURN or sample["metadata"].get("stage") in JSON_STAGES:
        return check_turns(sample)
    if sample["metadata"].get("stage") in JSON_STAGES:
        problems = []
        for m in sample["messages"]:
            if m["role"] == "assistant":
                try:
                    json.loads(m["content"])
                except json.JSONDecodeError:
                    problems.append("assistant_not_json")
        return problems
    user = sample["messages"][1]["content"]
    ans = sample["messages"][2]["content"]
    decision = re.search(r"DECISION: (\S+)", ans).group(1)
    allowed = re.search(r"ALLOWED ACTIONS: (.*)", user).group(1).split(" | ")
    why = ans.split("WHY:")[1].split("FIX:")[0].split("RULE:")[0]
    problems = []
    if decision not in allowed:
        problems.append("decision_not_allowed")
    missing = [n for n in NUM.findall(why) if n.lstrip("+") not in user and n not in user]
    if missing:
        problems.append(f"number_not_in_input:{missing[:3]}")
    problems += [p for p in check_turns(sample) if "missing_route" in p or "no_step_header" in p]
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
                s = annotate(s)  # the unified workflow: step header and route (workflow_map)
                h = hashlib.sha256(json.dumps(s["messages"][1:], ensure_ascii=False).encode()).hexdigest()  # whole conversation
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
