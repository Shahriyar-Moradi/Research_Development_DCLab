"""
train_mlx.py  (runs on the Mac, with MLX)
========================================

LoRA fine-tuning of Qwen3.5-9B (or another MLX model) on out/mix/final_*.jsonl, built for long, careful runs on
an M1 Max with 64 GB:

* the loss covers exactly what the model writes: its <think> reasoning and the answer of the LAST assistant turn
  (earlier turns are context, without thinking, as in use), in thinking or non-thinking mode per item;
* logits are computed only for those positions, so a 12k-token item does not need 12k × 248k logits;
* Qwen3.5's linear-attention layers train with the chunked gated delta rule (gated_delta_chunked.py) instead of a
  token-by-token loop, which keeps 12k-token items inside memory;
* gradient checkpointing per layer, batch 1 with gradient accumulation, AdamW with warm-up and cosine decay;
* every ``save_every`` optimizer steps: adapter, optimizer state and position are saved, so a stopped run resumes
  where it stopped (``--resume``) with the same data order and learning rate;
* a copy of the adapter at the end of every epoch, for choosing the best one with eval_mlx.py;
* validation loss on workflow items and on replay items (is general skill drifting?), logged to train_log.jsonl
  with tokens per second and peak memory.

Items longer than ``max_seq_length`` are skipped and counted, never cut: a cut item would teach a half answer.

Usage
-----
    python mac/train_mlx.py --config mac/config.yaml
    python mac/train_mlx.py --config mac/config.yaml --resume
    python mac/train_mlx.py --config mac/config.yaml --bench 2048 4096 8192 12288   # memory and speed per length
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from common import MIX, read_jsonl, render

DEFAULTS = {
    "model": None, "adapter_path": "adapters/dclab-qwen35", "data": str(MIX),
    "lora": {"rank": 16, "scale": 2.0, "dropout": 0.05, "num_layers": -1},
    "epochs": 3, "learning_rate": 1.0e-4, "warmup_share": 0.05, "min_lr_share": 0.1, "weight_decay": 0.0,
    "grad_accumulation": 8, "max_seq_length": 16384, "grad_checkpoint": True,
    "save_every": 250, "eval_every": 250, "eval_items": 120, "seed": 7,
    "wired_limit_gb": 56, "cache_limit_gb": 4, "chunked_gated_delta": True,
}


# ----------------------------------------------------------------------------------------------- model

def backbone_and_head(model):
    lm = getattr(model, "language_model", model)
    inner = lm.model
    tied = getattr(getattr(lm, "args", None), "tie_word_embeddings", False)
    head = inner.embed_tokens.as_linear if tied or not hasattr(lm, "lm_head") else lm.lm_head
    return inner, head


def setup_model(cfg):
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers
    model, tokenizer = load(cfg["model"])
    if cfg["chunked_gated_delta"] and getattr(model, "model_type", "") in ("qwen3_5", "qwen3_5_moe"):
        from gated_delta_chunked import patch_qwen3_5
        patch_qwen3_5()
    model.freeze()
    lora = cfg["lora"]
    linear_to_lora_layers(model, lora["num_layers"], {"rank": lora["rank"], "scale": lora["scale"], "dropout": lora["dropout"]})
    if cfg["grad_checkpoint"]:
        grad_checkpoint(model.layers[0])
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    n_all = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"trainable parameters: {n_train / 1e6:.1f}M of {n_all / 1e6:.0f}M ({100 * n_train / n_all:.2f}%)", flush=True)
    return model, tokenizer


def adapter_config(cfg):
    lora = cfg["lora"]
    return {"fine_tune_type": "lora", "num_layers": lora["num_layers"],
            "lora_parameters": {"rank": lora["rank"], "scale": lora["scale"], "dropout": lora["dropout"]},
            "model": cfg["model"]}


# ----------------------------------------------------------------------------------------------- data

def encode(tokenizer, row):
    prompt, full = render(tokenizer, row["context"], row["answer"], row.get("think"), row["mode"])
    ids = tokenizer.encode(full, add_special_tokens=False)
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    offset = len(p_ids)
    if ids[:offset] != p_ids:  # a token merged across the boundary: fall back to the common prefix
        offset = next((i for i, (a, b) in enumerate(zip(ids, p_ids)) if a != b), min(len(ids), len(p_ids)))
    return ids, offset


def load_rows(cfg, tokenizer, split):
    rows = read_jsonl(Path(cfg["data"]) / f"final_{split}.jsonl")
    out, skipped = [], 0
    for r in rows:
        ids, off = encode(tokenizer, r)
        if len(ids) > cfg["max_seq_length"] or off >= len(ids) - 1:
            skipped += 1
            continue
        out.append({"ids": ids, "offset": off, "weight": int(r.get("weight", 1)), "source": r.get("source", "workflow"),
                    "id": r["id"]})
    return out, skipped


def epoch_order(rows, epoch, seed):
    expanded = [i for i, r in enumerate(rows) for _ in range(r["weight"])]
    random.Random(seed * 1000 + epoch).shuffle(expanded)
    return expanded


# ----------------------------------------------------------------------------------------------- loss

def make_loss(model):
    inner, head = backbone_and_head(model)

    def loss_fn(model, ids, offset, length):
        h = inner(ids[:, :-1])
        h = h[:, offset - 1:length - 1]  # positions that predict the answer tokens only
        logits = head(h).astype(mx.float32)
        targets = ids[:, offset:length]
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    return loss_fn


def evaluate(model, loss_fn, rows):
    model.eval()
    totals = {}
    for r in rows:
        ids = mx.array([r["ids"]])
        loss = loss_fn(model, ids, r["offset"], len(r["ids"]))
        mx.eval(loss)
        totals.setdefault(r["source"], []).append(loss.item())
    model.train()
    return {k: float(np.mean(v)) for k, v in totals.items()}


# ----------------------------------------------------------------------------------------------- checkpoints

def save_checkpoint(model, opt, state, cfg, path: Path, tag: str | None = None):
    path.mkdir(parents=True, exist_ok=True)
    adapters = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path / "adapters.safetensors"), adapters)
    mx.save_safetensors(str(path / "optimizer.safetensors"), dict(tree_flatten(opt.state)))
    (path / "adapter_config.json").write_text(json.dumps(adapter_config(cfg), indent=1))
    (path / "train_state.json").write_text(json.dumps(state, indent=1))
    if tag:
        keep = path / tag
        keep.mkdir(exist_ok=True)
        shutil.copy(path / "adapters.safetensors", keep / "adapters.safetensors")
        shutil.copy(path / "adapter_config.json", keep / "adapter_config.json")


def load_checkpoint(model, opt, path: Path):
    model.load_weights(str(path / "adapters.safetensors"), strict=False)
    opt.state = tree_unflatten(list(mx.load(str(path / "optimizer.safetensors")).items()))
    return json.loads((path / "train_state.json").read_text())


# ----------------------------------------------------------------------------------------------- main

def schedule(cfg, total_steps):
    lr = cfg["learning_rate"]
    warm = max(1, int(total_steps * cfg["warmup_share"]))
    return optim.join_schedules(
        [optim.linear_schedule(0.0, lr, warm),
         optim.cosine_decay(lr, max(1, total_steps - warm), end=lr * cfg["min_lr_share"])], [warm])


def bench(cfg, lengths):
    model, tokenizer = setup_model(cfg)
    loss_fn = make_loss(model)
    vg = nn.value_and_grad(model, loss_fn)
    vocab = getattr(tokenizer, "vocab_size", 1000)
    for n in lengths:
        ids = mx.array(np.random.default_rng(0).integers(10, min(vocab, 30000), (1, n)).astype(np.int32))
        mx.reset_peak_memory()
        times = []
        for _ in range(3):
            t = time.time()
            loss, grads = vg(model, ids, n - 1500 if n > 3000 else n // 2, n)
            mx.eval(loss, grads)
            times.append(time.time() - t)
        print(f"length {n:6d}: {np.median(times):7.1f} s per step, peak memory {mx.get_peak_memory() / 1e9:5.1f} GB, "
              f"{n / np.median(times):6.0f} tokens/s", flush=True)


def train(cfg, resume: bool, stop_after: int = 0):
    out = Path(cfg["adapter_path"])
    model, tokenizer = setup_model(cfg)
    train_rows, skipped = load_rows(cfg, tokenizer, "train")
    val_rows, _ = load_rows(cfg, tokenizer, "val")
    val_rows = sorted(val_rows, key=lambda r: r["id"])
    val_sel = [r for r in val_rows if r["source"] == "workflow"][: cfg["eval_items"]] + \
              [r for r in val_rows if r["source"] == "replay"][: max(10, cfg["eval_items"] // 4)]
    per_epoch = sum(r["weight"] for r in train_rows)
    total_steps = math.ceil(per_epoch * cfg["epochs"] / cfg["grad_accumulation"])
    tokens_epoch = sum(len(r["ids"]) * r["weight"] for r in train_rows)
    print(f"train items {len(train_rows)} ({per_epoch} with repeats, {skipped} skipped as longer than "
          f"{cfg['max_seq_length']}), ~{tokens_epoch / 1e6:.1f}M tokens per epoch, {total_steps} optimizer steps", flush=True)
    opt = optim.AdamW(learning_rate=schedule(cfg, total_steps), weight_decay=cfg["weight_decay"])
    loss_fn = make_loss(model)
    vg = nn.value_and_grad(model, loss_fn)
    state = {"epoch": 0, "pos": 0, "micro": 0, "opt_step": 0, "tokens": 0, "skipped_long": skipped}
    if resume and (out / "train_state.json").exists():
        state = load_checkpoint(model, opt, out)
        print(f"resumed at epoch {state['epoch']}, item {state['pos']}, optimizer step {state['opt_step']}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a")
    model.train()
    if state["opt_step"] == 0:
        ev = evaluate(model, loss_fn, val_sel)
        log.write(json.dumps({"opt_step": 0, "val": ev}) + "\n")
        log.flush()
        print(f"step 0 | val {ev}", flush=True)
    grads_acc, n_acc, t0, tok0, losses = None, 0, time.time(), 0, []
    while state["epoch"] < cfg["epochs"]:
        order = epoch_order(train_rows, state["epoch"], cfg["seed"])
        while state["pos"] < len(order):
            r = train_rows[order[state["pos"]]]
            ids = mx.array([r["ids"]])
            loss, grads = vg(model, ids, r["offset"], len(r["ids"]))
            grads_acc = grads if grads_acc is None else tree_map(lambda a, b: a + b, grads_acc, grads)
            mx.eval(loss, grads_acc)
            losses.append(loss.item())
            n_acc += 1
            state["pos"] += 1
            state["micro"] += 1
            state["tokens"] += len(r["ids"])
            tok0 += len(r["ids"])
            if n_acc == cfg["grad_accumulation"] or state["pos"] == len(order):
                opt.update(model, tree_map(lambda g: g / n_acc, grads_acc))
                mx.eval(model.parameters(), opt.state)
                grads_acc, n_acc = None, 0
                state["opt_step"] += 1
                mx.clear_cache()
                if state["opt_step"] % 10 == 0:
                    dt = time.time() - t0
                    rec = {"opt_step": state["opt_step"], "epoch": state["epoch"], "loss": float(np.mean(losses)),
                           "lr": float(opt.learning_rate.item()), "tokens_per_s": tok0 / max(dt, 1e-9),
                           "peak_gb": mx.get_peak_memory() / 1e9}
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(f"step {rec['opt_step']}/{total_steps} | epoch {rec['epoch'] + 1} | loss {rec['loss']:.4f} | "
                          f"lr {rec['lr']:.2e} | {rec['tokens_per_s']:.0f} tok/s | peak {rec['peak_gb']:.1f} GB", flush=True)
                    losses, t0, tok0 = [], time.time(), 0
                if state["opt_step"] % cfg["eval_every"] == 0:
                    ev = evaluate(model, loss_fn, val_sel)
                    log.write(json.dumps({"opt_step": state["opt_step"], "val": ev}) + "\n")
                    log.flush()
                    print(f"step {state['opt_step']} | val {ev}", flush=True)
                if state["opt_step"] % cfg["save_every"] == 0:
                    save_checkpoint(model, opt, state, cfg, out)
                if stop_after and state["opt_step"] >= stop_after:
                    print(f"stopped after optimizer step {state['opt_step']} (last checkpoint: step "
                          f"{state['opt_step'] - state['opt_step'] % cfg['save_every']}); continue with --resume", flush=True)
                    return
        state["epoch"] += 1
        state["pos"] = 0
        ev = evaluate(model, loss_fn, val_sel)
        log.write(json.dumps({"opt_step": state["opt_step"], "epoch_end": state["epoch"], "val": ev}) + "\n")
        log.flush()
        print(f"epoch {state['epoch']} done | val {ev}", flush=True)
        save_checkpoint(model, opt, state, cfg, out, tag=f"epoch_{state['epoch']}")
    print(f"done: adapters in {out} (one copy per epoch in {out}/epoch_N)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--bench", type=int, nargs="*")
    ap.add_argument("--set", nargs="*", default=[], help="override, e.g. epochs=2 model=models/x")
    ap.add_argument("--stop-after", type=int, default=0, help="stop after this many optimizer steps (resume later)")
    args = ap.parse_args()
    cfg = json.loads(json.dumps(DEFAULTS))
    cfg.update(yaml.safe_load(args.config.read_text()) or {})
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    if mx.metal.is_available():
        mx.set_wired_limit(int(cfg["wired_limit_gb"] * 1e9))
        mx.set_cache_limit(int(cfg["cache_limit_gb"] * 1e9))
    mx.random.seed(cfg["seed"])
    if args.bench:
        bench(cfg, args.bench)
    else:
        train(cfg, args.resume, args.stop_after)


if __name__ == "__main__":
    main()
