"""
make_think_traces.py  (runs on the Mac, with MLX)
===============================================

Gives each thinking item its <think> reasoning, written by Qwen itself, so fine-tuning keeps Qwen's own way of
reasoning and only points it at our rules.

For every item whose ``trace`` is ``sample``: Qwen answers the step freely, in thinking mode, up to ``--tries``
times. A try is kept only if its final answer reaches the GOLD decision and route (and, for a contract or an audit,
the same fields or column decisions) and its reasoning passes the checks: no number that is not in the
conversation (a difference, sum or ratio of two shown numbers is allowed), no mention of being given the answer,
and not too long. The kept reasoning is paired with the GOLD answer.

For ``rationalize`` items (answers too long to match exactly: findings, plans, reports, code) and for sampled
items that never reached the gold answer, Qwen sees the gold answer and thinks its way to it; the same checks
apply. Items that fail everything are written as failed; export_mlx.py turns them into non-thinking items.

Resumable: every finished item is appended to the output at once, and items already there are skipped.

Usage (on the Mac, after build_training_mix.py)
-----
    python mac/make_think_traces.py --model models/qwen35-9b-8bit --splits train val
    python mac/make_think_traces.py --model models/qwen35-9b-8bit --limit 20      # a quick look first
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import MIX, append_jsonl, matches, read_jsonl, split_generation, trace_problems

RATIONALIZE_NOTE = (
    "\n\n---\nREVIEWED ANSWER (correct): the answer below is the right one for this step. Think through the evidence "
    "above step by step, the way you would without it, so that your reasoning reaches this answer on its own terms. "
    "Do not mention in your thinking that an answer was provided.\n\n{answer}"
)


def context_text(item) -> str:
    return "\n".join(m["content"] for m in item["context"])


def prompt_tokens(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True, tokenize=True)


def rationalize_messages(item):
    msgs = [dict(m) for m in item["context"]]
    msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + RATIONALIZE_NOTE.format(answer=item["answer"])}
    return msgs


def generate(model, tokenizer, batch_generate, sampler, prompts, max_tokens, batch):
    out = []
    for i in range(0, len(prompts), batch):
        res = batch_generate(model, tokenizer, prompts[i:i + batch], max_tokens=max_tokens, sampler=sampler,
                             completion_batch_size=batch, prefill_batch_size=min(batch, 4))
        out.extend(res.texts)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local MLX model folder (mlx_lm.convert output)")
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--mix", type=Path, default=MIX, help="folder with items_*.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="default: <mix>/traces.jsonl")
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--rationalize-tries", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8, help="sequences generated together (continuous batching)")
    ap.add_argument("--max-tokens", type=int, default=3072, help="cap on thinking + answer tokens per try")
    ap.add_argument("--max-think-words", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="only this many items (a quick look)")
    ap.add_argument("--chunk", type=int, default=64, help="items handled per round, so progress is saved often")
    args = ap.parse_args()

    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    args.out = args.out or args.mix / "traces.jsonl"
    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p, top_k=args.top_k)
    done = {r["id"] for r in read_jsonl(args.out)}
    items = [it for s in args.splits for it in read_jsonl(args.mix / f"items_{s}.jsonl")
             if it["trace"] in ("sample", "rationalize") and it["id"] not in done]
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} items need a trace ({len(done)} already done)", flush=True)
    t0, kept, failed = time.time(), 0, 0
    for c in range(0, len(items), args.chunk):
        chunk = items[c:c + args.chunk]
        pending = [it for it in chunk if it["trace"] == "sample"]
        tries = {it["id"]: 0 for it in chunk}
        reasons = {it["id"]: [] for it in chunk}
        for _ in range(args.tries):
            if not pending:
                break
            texts = generate(model, tokenizer, batch_generate, sampler,
                             [prompt_tokens(tokenizer, it["context"]) for it in pending], args.max_tokens, args.batch)
            still = []
            for it, text in zip(pending, texts):
                tries[it["id"]] += 1
                think, answer = split_generation(text)
                probs = ([] if matches(it["match"], answer) else ["answer_differs"]) + \
                    trace_problems(think, context_text(it), args.max_think_words)
                if probs:
                    reasons[it["id"]].append(probs[0])
                    still.append(it)
                else:
                    append_jsonl(args.out, {"id": it["id"], "think": think, "method": "sampled", "tries": tries[it["id"]]})
                    kept += 1
            pending = still
        rational = pending + [it for it in chunk if it["trace"] == "rationalize"]
        for _ in range(args.rationalize_tries):
            if not rational:
                break
            texts = generate(model, tokenizer, batch_generate, sampler,
                             [prompt_tokens(tokenizer, rationalize_messages(it)) for it in rational], args.max_tokens, args.batch)
            still = []
            for it, text in zip(rational, texts):
                tries[it["id"]] += 1
                think, _answer = split_generation(text)
                probs = trace_problems(think, context_text(it), args.max_think_words)
                if probs:
                    reasons[it["id"]].append(probs[0])
                    still.append(it)
                else:
                    append_jsonl(args.out, {"id": it["id"], "think": think, "method": "rationalized", "tries": tries[it["id"]]})
                    kept += 1
            rational = still
        for it in rational:
            append_jsonl(args.out, {"id": it["id"], "think": None, "method": "failed", "tries": tries[it["id"]],
                                    "reasons": reasons[it["id"]][-3:]})
            failed += 1
        n = c + len(chunk)
        rate = (time.time() - t0) / max(n, 1)
        print(f"{n}/{len(items)} items | kept {kept} | failed {failed} | {rate:.1f}s per item | "
              f"about {(len(items) - n) * rate / 3600:.1f} h left", flush=True)


if __name__ == "__main__":
    main()
