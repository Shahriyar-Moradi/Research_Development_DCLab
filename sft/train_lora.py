#!/usr/bin/env python3
"""LoRA supervised fine-tuning of a small open model on the DCLab v3 corpus.

This is the "SFT job can train on it directly" path. It reads the chat-format
files written by ``sft/build_sft_dataset_v3.py`` and trains a LoRA adapter with
Hugging Face TRL. Run it on a machine with a GPU; it is not run in CI.

    pip install "trl>=0.12" "peft>=0.13" "transformers>=4.46" datasets accelerate
    python sft/train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --out sft/runs/qwen15b-dclab
    python sft/eval_sft.py generate --adapter sft/runs/qwen15b-dclab --model Qwen/Qwen2.5-1.5B-Instruct

Defaults are deliberately small (short epochs, rank-16 LoRA) because the corpus is
a few hundred examples: the goal is to teach the answer *format* and the habit
of reasoning from given evidence, not to memorize numbers. Facts that change stay
in retrieval (``dclab_rnd.evidence_index``), not in weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "sft" / "out_v3"


def load_chat(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append({"messages": json.loads(line)["messages"]})  # drop metadata; TRL expects messages only
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "sft" / "runs" / "dclab-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    train = Dataset.from_list(load_chat(args.data_dir / "train.chat.jsonl"))
    val = Dataset.from_list(load_chat(args.data_dir / "val.chat.jsonl"))
    config = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        assistant_only_loss=True,          # learn the answers, not the long evidence prompts
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        bf16=True,
        report_to="none",
        seed=42,
    )
    peft = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=train, eval_dataset=val, peft_config=peft)
    trainer.train()
    trainer.save_model(str(args.out))
    print(f"Adapter saved to {args.out}. Evaluate with: python sft/eval_sft.py generate --adapter {args.out} --model {args.model}")


if __name__ == "__main__":
    main()
