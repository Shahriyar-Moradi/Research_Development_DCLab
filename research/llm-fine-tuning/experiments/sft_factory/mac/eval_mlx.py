"""
eval_mlx.py  (runs on the Mac, with MLX)
=======================================

Generation-based evaluation: the model ANSWERS each validation or test item (thinking mode, Qwen's recommended
sampling), and the answers are scored against the gold labels. Loss alone does not say whether decisions are right.

Scores
* decision accuracy and route accuracy, overall, per stage and per decision (rare decisions shown separately);
* audit columns (every column decided right) and contract fields (target, task, moment, time column);
* format: every DECISION answer has WHY, RULE, CONFIDENCE and a routed NEXT; JSON answers parse; code compiles;
* honesty: numbers in the answer that are not in the input (should be 0), and in the thinking (differences, sums and
  ratios of shown numbers allowed);
* general skill (replay items): code that passes its tests, numeric answers within tolerance;
* length of the thinking.

Use it on the base model first (no adapter), then on each epoch's adapter; ``--compare`` prints them side by side,
and ``--pick`` names the best epoch: highest decision accuracy among those whose replay pass rate is within 2 points
of the base model's and whose invented-number rate is not higher.

Usage
-----
    python mac/eval_mlx.py --model models/qwen35-9b-8bit --split val --out eval/base_val.json
    python mac/eval_mlx.py --model models/qwen35-9b-8bit --adapter adapters/dclab-qwen35/epoch_2 --split val --out eval/e2_val.json
    python mac/eval_mlx.py --compare eval/base_val.json eval/e1_val.json eval/e2_val.json eval/e3_val.json --pick
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from common import (MIX, extract_code, final_number, matches, numbers_ok, parse_answer, read_jsonl, run_code_tests,
                    split_generation)

DECISION_PARTS = ("\nWHY: ", "\nRULE: ", "\nCONFIDENCE: ", "\nNEXT: [")


def score_item(row, think, answer, tasks):
    s = {"stage": row["stage"], "source": row["source"], "think_words": len(think.split())}
    ctx = "\n".join(m["content"] for m in row["context"])
    m = row["match"]
    if row["source"] == "replay":
        task = tasks.get(row.get("task", ""))
        if task and task["kind"] == "code":
            code = extract_code(answer)
            s["replay_ok"] = bool(code) and run_code_tests(code, task["tests"])
        elif task:
            v = final_number(answer)
            s["replay_ok"] = v is not None and abs(v - task["answer"]) <= task["tolerance"]
        return s
    gold = parse_answer(row["answer"])
    got = parse_answer(answer)
    if m["kind"] in ("decision", "json_decision"):
        s["gold_decision"] = m["decision"]
        s["decision_ok"] = got.get("decision") == m["decision"]
        if m.get("route"):
            s["route_ok"] = got.get("route") == m["route"]
    elif m["kind"] == "columns":
        gd, pd_ = m["decisions"], (got.get("decisions") or {})
        s["columns_ok"] = sum(pd_.get(c) == d for c, d in gd.items()) / max(len(gd), 1)
        s["decision_ok"] = pd_ == gd
        s["route_ok"] = got.get("route") == m["route"]
    elif m["kind"] == "contract":
        s["decision_ok"] = matches(m, answer)
        s["fields_ok"] = got.get("fields") == m["fields"]
        s["route_ok"] = got.get("route") == m["route"]
    if row["answer"].startswith("DECISION:"):
        s["format_ok"] = answer.startswith("DECISION:") and all(p in answer for p in DECISION_PARTS)
    elif row["answer"].startswith("```python"):
        code = extract_code(answer)
        try:
            compile(code or "", "<answer>", "exec")
            s["format_ok"] = bool(code)
        except SyntaxError:
            s["format_ok"] = False
    else:
        s["format_ok"] = gold.get("kind") == "invalid_json" or got.get("kind") not in ("none", "invalid_json")
    s["invented_numbers_answer"] = len(numbers_ok(re.sub(r"```python.*?```", "", answer, flags=re.S), ctx))
    s["invented_numbers_think"] = len(numbers_ok(think, ctx, arithmetic=True))
    return s


def summarise(scores):
    def mean(key, rows):
        vals = [r[key] for r in rows if key in r]
        return round(sum(vals) / len(vals), 4) if vals else None
    wf = [s for s in scores if s["source"] == "workflow"]
    by_stage = defaultdict(list)
    by_decision = defaultdict(list)
    for s in wf:
        by_stage[s["stage"]].append(s)
        if "gold_decision" in s:
            by_decision[s["gold_decision"]].append(s)
    rp = [s for s in scores if s["source"] == "replay"]
    return {
        "items": len(scores),
        "decision_accuracy": mean("decision_ok", wf), "route_accuracy": mean("route_ok", wf),
        "format_ok": mean("format_ok", wf),
        "answers_with_invented_numbers": round(sum(s.get("invented_numbers_answer", 0) > 0 for s in wf) / max(len(wf), 1), 4),
        "thinking_with_invented_numbers": round(sum(s.get("invented_numbers_think", 0) > 0 for s in wf) / max(len(wf), 1), 4),
        "think_words_mean": mean("think_words", wf),
        "replay_pass_rate": mean("replay_ok", rp), "replay_items": len(rp),
        "by_stage": {k: {"n": len(v), "decision": mean("decision_ok", v), "route": mean("route_ok", v),
                         "format": mean("format_ok", v)} for k, v in sorted(by_stage.items())},
        "by_decision": {k: {"n": len(v), "accuracy": mean("decision_ok", v)} for k, v in sorted(by_decision.items())},
    }


def run(args):
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = load(args.model, adapter_path=args.adapter) if args.adapter else load(args.model)
    rows = read_jsonl(Path(args.data) / f"final_{args.split}.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    tasks = {t["id"]: t for t in read_jsonl(Path(args.data) / "replay_tasks.jsonl")}
    sampler = make_sampler(temp=args.temperature, top_p=0.95, top_k=20)
    thinking = args.mode == "think"
    prompts = [tokenizer.apply_chat_template(r["context"], add_generation_prompt=True, enable_thinking=thinking, tokenize=True)
               for r in rows]
    scores, gens = [], []
    for i in range(0, len(rows), args.batch):
        texts = batch_generate(model, tokenizer, prompts[i:i + args.batch], max_tokens=args.max_tokens, sampler=sampler,
                               completion_batch_size=args.batch, prefill_batch_size=min(args.batch, 4)).texts
        for r, text in zip(rows[i:i + args.batch], texts):
            think, answer = split_generation(text) if thinking else ("", text.replace("<|im_end|>", "").strip())
            scores.append(score_item(r, think, answer, tasks))
            gens.append({"id": r["id"], "stage": r["stage"], "think": think, "answer": answer, "gold": r["answer"]})
        print(f"{min(i + args.batch, len(rows))}/{len(rows)}", flush=True)
    report = {"model": args.model, "adapter": args.adapter, "split": args.split, "mode": args.mode,
              "summary": summarise(scores)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    with open(out.with_suffix(".generations.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(g, ensure_ascii=False) + "\n" for g in gens)
    print(json.dumps(report["summary"], indent=2))


def compare(paths, pick):
    reps = [json.loads(Path(p).read_text()) for p in paths]
    keys = ["decision_accuracy", "route_accuracy", "format_ok", "answers_with_invented_numbers",
            "thinking_with_invented_numbers", "replay_pass_rate", "think_words_mean"]
    names = [Path(p).stem for p in paths]
    print("| metric | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for k in keys:
        print(f"| {k} | " + " | ".join(str(r["summary"].get(k)) for r in reps) + " |")
    if pick:
        base = reps[0]["summary"]
        ok = [(n, r["summary"]) for n, r in zip(names[1:], reps[1:])
              if (r["summary"]["replay_pass_rate"] or 0) >= (base["replay_pass_rate"] or 0) - 0.02
              and r["summary"]["answers_with_invented_numbers"] <= max(base["answers_with_invented_numbers"], 0.0)]
        if ok:
            best = max(ok, key=lambda t: (t[1]["decision_accuracy"] or 0, t[1]["route_accuracy"] or 0))
            print(f"\nbest: {best[0]} (decision accuracy {best[1]['decision_accuracy']}, keeps general skill and honesty)")
        else:
            print("\nno candidate keeps general skill and honesty at the base model's level: add replay data or lower the "
                  "learning rate before choosing")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--data", default=str(MIX))
    ap.add_argument("--split", default="val")
    ap.add_argument("--mode", choices=["think", "no_think"], default="think")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--out", default="eval/report.json")
    ap.add_argument("--compare", nargs="*")
    ap.add_argument("--pick", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare, args.pick)
    else:
        run(args)


if __name__ == "__main__":
    main()
