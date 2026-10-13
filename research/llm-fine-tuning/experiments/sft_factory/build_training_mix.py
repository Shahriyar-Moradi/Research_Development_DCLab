"""
build_training_mix.py
=====================

Turns the merged, checked samples (out/all_stages) into TRAINING ITEMS: one item per assistant answer, with the
conversation before it as context. This is how the model is used (one step at a time, earlier answers in the
history, no earlier thinking in the history: Qwen's own rule), so it is also how it is trained.

    out/all_stages/{train,val,test}.jsonl  ──►  out/mix/items_{train,val,test}.jsonl + out/mix/mix_report.json

Each item says
* ``context``  the messages before the answer (system, user turns, earlier gold answers without thinking);
* ``answer``   the gold answer (the label);
* ``think_mode`` think (about 70%) or no_think (about 30%), so both of Qwen's modes keep working;
* ``trace``    how its <think> reasoning will be made on the Mac (mac/make_think_traces.py):
               ``sample`` (Qwen reasons freely; kept only if it reaches the gold decision and route),
               ``rationalize`` (answers too long to match exactly: Qwen writes the reasoning for the gold answer,
               checked for invented numbers), or ``none`` (no_think items);
* ``match``    what a sampled trace must agree with (decision, route, contract fields, column decisions);
* ``weight``   how many times it is repeated in training: rare decisions and small stages are upsampled
               (at most 4×), so the model learns them instead of the majority answer.

Long trajectories are not trained turn by turn in full: from each end-to-end run the turns that go back or ask,
the revisit and report turns, and a few others at random are kept (``--e2e-turns``), so the long contexts do not
crowd out everything else.

Usage
-----
    python research/llm-fine-tuning/experiments/sft_factory/build_training_mix.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

THINK_SHARE = 0.7
MAX_WEIGHT = 4


def h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def answer_kind(ans: str) -> dict:
    """What a freely sampled answer must agree with, or that it must be rationalized."""
    if ans.startswith("DECISION:"):
        m = re.search(r"\nNEXT: (\[[^\]]+\])", ans)
        return {"kind": "decision", "decision": re.search(r"DECISION: (\S+)", ans).group(1), "route": m.group(1) if m else None}
    if ans.startswith("```python"):
        return {"kind": "rationalize", "why": "code"}
    try:
        obj = json.loads(ans)
    except json.JSONDecodeError:
        return {"kind": "rationalize", "why": "free text"}
    route = obj.get("route", {})
    r = [route.get("action"), (route.get("to") or "").split(" ")[0]] if route else None
    if "decision" in obj and "checks" in obj:
        return {"kind": "json_decision", "decision": obj["decision"], "route": r}
    if "columns" in obj and "feature_set" in obj:
        return {"kind": "columns", "decisions": {c["column"]: c["decision"] for c in obj["columns"]}, "route": r}
    if "unknown" in obj and "open_questions" in obj:
        fields = {f: obj[f]["value"] for f in ("task", "target_column", "prediction_moment", "time_column") if f in obj}
        return {"kind": "contract", "fields": fields, "unknown": sorted(obj["unknown"]), "route": r}
    return {"kind": "rationalize", "why": "long structured answer"}


def items_from(sample: dict, rng: random.Random, e2e_turns: int) -> list[dict]:
    msgs, md = sample["messages"], sample["metadata"]
    answer_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    if md.get("stage") == "end_to_end":
        keep = set()
        for i in answer_idx:
            a = msgs[i]["content"]
            if "[back →" in a or '"action": "back"' in a or "ask_owner" in a or '"status": "complete"' in a \
                    or "solution_revisit" in msgs[i - 1]["content"]:
                keep.add(i)
        others = [i for i in answer_idx if i not in keep and i > 3]  # the first answers repeat the front-stage samples
        keep |= set(rng.sample(others, min(e2e_turns, len(others))))
        answer_idx = sorted(keep)
    elif md.get("stage") == "connected_problem_to_plan":
        answer_idx = [rng.choice(answer_idx[1:])]  # the turn whose context is the earlier answers
    out = []
    for i in answer_idx:
        context = [{"role": m["role"], "content": m["content"]} for m in msgs[:i]]
        ans = msgs[i]["content"]
        user = msgs[i - 1]["content"]
        step = re.match(r"WORKFLOW STEP: (WF-\d\d)", user)
        stage = re.search(r"STAGE: (\S+)", user)
        match = answer_kind(ans)
        item = {
            "context": context, "answer": ans,
            "stage": md.get("stage"),
            "turn_stage": "code_writing" if ans.startswith("```python") else (stage.group(1) if stage else md.get("stage")),
            "step": step.group(1) if step else md.get("step"),
            "action": match.get("decision") or md.get("action"), "dataset": md.get("dataset"),
            "match": match,
        }
        item["id"] = h({"c": context, "a": ans})[:20]
        think = (int(item["id"][:8], 16) % 100) < THINK_SHARE * 100
        item["think_mode"] = "think" if think else "no_think"
        item["trace"] = ("sample" if match["kind"] != "rationalize" else "rationalize") if think else "none"
        out.append(item)
    return out


def weights(items: list[dict]) -> None:
    """Upsample rare (stage, decision) pairs and small stages, at most MAX_WEIGHT times."""
    by_pair = Counter((it["turn_stage"], it["action"]) for it in items)
    by_stage = defaultdict(list)
    for (st, _a), n in by_pair.items():
        by_stage[st].append(n)
    stage_sizes = Counter(it["turn_stage"] for it in items)
    median_stage = sorted(stage_sizes.values())[len(stage_sizes) // 2]
    for it in items:
        st = it["turn_stage"]
        pair_target = max(by_stage[st])  # balance decisions within a stage
        w_pair = pair_target / by_pair[(st, it["action"])]
        w_stage = median_stage / stage_sizes[st]  # lift small stages towards the median stage
        it["weight"] = int(min(MAX_WEIGHT, max(1, round(math.sqrt(w_pair) * max(1.0, w_stage) ** 0.5))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=OUT / "all_stages")
    ap.add_argument("--out", type=Path, default=OUT / "mix")
    ap.add_argument("--e2e-turns", type=int, default=3, help="extra random turns kept per end-to-end run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in ("train", "val", "test"):
        rows = [json.loads(l) for l in open(args.src / f"{split}.jsonl", encoding="utf-8")]
        items = [it for r in rows for it in items_from(r, rng, args.e2e_turns)]
        if split == "train":
            weights(items)
        else:
            for it in items:
                it["weight"] = 1
        seen, uniq = set(), []
        for it in items:
            if it["id"] not in seen:
                seen.add(it["id"])
                uniq.append(it)
        with open(args.out / f"items_{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps({**it, "split": split}, ensure_ascii=False) + "\n" for it in uniq)
        chars = [sum(len(m["content"]) for m in it["context"]) + len(it["answer"]) for it in uniq]
        report[split] = {
            "items": len(uniq), "weighted_items": sum(it["weight"] for it in uniq),
            "by_stage": dict(Counter(it["turn_stage"] for it in uniq)),
            "by_trace": dict(Counter(it["trace"] for it in uniq)),
            "by_match": dict(Counter(it["match"]["kind"] for it in uniq)),
            "approx_tokens_per_epoch": int(sum(c * it["weight"] for c, it in zip(chars, uniq)) / 3.5),
            "approx_tokens_max_item": int(max(chars, default=0) / 3.5),
            "weights": dict(Counter(it["weight"] for it in uniq)),
        }
    (args.out / "mix_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
