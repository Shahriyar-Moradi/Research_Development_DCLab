"""
make_replay.py  (runs on the Mac, with MLX)
==========================================

The "replay" share of the training mix: Qwen's OWN answers (with its thinking) to general coding and ML-arithmetic
tasks, kept only when they are right. Training on them keeps the model's general coding and reasoning where they
were while it learns the workflow (self-distillation, verified):

* a code answer is kept only if its function passes every test (expected values from a reference implementation);
* a numeric answer is kept only if its last line ``ANSWER: <number>`` is within the tolerance.

Resumable: finished tasks are appended to the output and skipped on the next run.

Usage (after replay_tasks.py)
-----
    python mac/make_replay.py --model models/qwen35-9b-8bit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import MIX, append_jsonl, extract_code, final_number, read_jsonl, run_code_tests, split_generation


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mix", type=Path, default=MIX)
    ap.add_argument("--tasks", type=Path, default=None, help="default: <mix>/replay_tasks.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="default: <mix>/replay_items.jsonl")
    ap.add_argument("--per-task", type=int, default=2, help="different correct answers kept per task")
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-think-words", type=int, default=1500)
    args = ap.parse_args()

    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    args.tasks = args.tasks or args.mix / "replay_tasks.jsonl"
    args.out = args.out or args.mix / "replay_items.jsonl"
    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)
    done = {r["task"] for r in read_jsonl(args.out)}
    tasks = [t for t in read_jsonl(args.tasks) if t["id"] not in done]
    print(f"{len(tasks)} replay tasks to do", flush=True)
    t0 = time.time()
    for c in range(0, len(tasks), args.batch):
        chunk = tasks[c:c + args.batch]
        kept = {t["id"]: [] for t in chunk}
        for _ in range(args.tries):
            todo = [t for t in chunk if len(kept[t["id"]]) < args.per_task]
            if not todo:
                break
            prompts = [tokenizer.apply_chat_template([{"role": "user", "content": t["prompt"]}], add_generation_prompt=True,
                                                     enable_thinking=True, tokenize=True) for t in todo]
            texts = batch_generate(model, tokenizer, prompts, max_tokens=args.max_tokens, sampler=sampler,
                                   completion_batch_size=args.batch, prefill_batch_size=min(args.batch, 4)).texts
            for t, text in zip(todo, texts):
                think, answer = split_generation(text)
                if not answer or len(think.split()) > args.max_think_words:
                    continue
                if t["kind"] == "code":
                    code = extract_code(answer)
                    ok = bool(code) and run_code_tests(code, t["tests"])
                else:
                    v = final_number(answer)
                    ok = v is not None and abs(v - t["answer"]) <= t["tolerance"]
                if ok and answer not in [k["answer"] for k in kept[t["id"]]]:
                    kept[t["id"]].append({"think": think, "answer": answer})
        for t in chunk:
            for j, k in enumerate(kept[t["id"]]):
                append_jsonl(args.out, {"id": f"replay-{t['id']}-{j}", "task": t["id"], "kind": t["kind"],
                                        "context": [{"role": "user", "content": t["prompt"]}],
                                        "answer": k["answer"], "think": k["think"], "think_mode": "think"})
            if not kept[t["id"]]:
                append_jsonl(args.out, {"id": f"replay-{t['id']}-none", "task": t["id"], "kind": t["kind"], "failed": True})
        n = c + len(chunk)
        print(f"{n}/{len(tasks)} tasks | {(time.time() - t0) / n:.1f}s per task", flush=True)


if __name__ == "__main__":
    main()
