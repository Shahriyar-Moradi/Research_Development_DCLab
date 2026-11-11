"""Fine-tune GLiNER2.5-Decide. Starts with --smoke before a full run."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, default=Path("research/churn-prediction/experiments/gliner_decide/processed"))
    p.add_argument("--out", type=Path, default=Path("research/churn-prediction/experiments/gliner_decide/runs/gliner"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--smoke", action="store_true", help="Use only 64 training rows and 1 epoch")
    p.add_argument("--allow-cpu", action="store_true", help="Slow; opt in only if GPU acceleration unavailable")
    args = p.parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available(),
          "Apple MPS:", torch.backends.mps.is_available())
    if not (torch.cuda.is_available() or torch.backends.mps.is_available() or args.allow_cpu):
        raise SystemExit("No accelerator detected. Use CUDA, MPS, or explicitly --allow-cpu for a tiny smoke test.")
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
    args.out.mkdir(parents=True, exist_ok=True)
    src = args.processed / "train.jsonl"
    if args.smoke:
        lines = src.read_text(encoding="utf-8").splitlines()[:64]
        src = args.out / "smoke_train.jsonl"
        src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Loading GLiNER checkpoint...")
    model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
    config = TrainingConfig(output_dir=str(args.out), num_epochs=1 if args.smoke else args.epochs,
                            batch_size=args.batch_size)
    print("Training on", src)
    ExtractorTrainer(model, config).train(train_data=str(src))
    from common import dump
    dump(args.out / "training_manifest.json", {"checkpoint": "fastino/GLiNER2.5-Decide",
        "train_file": str(src), "seed": 42, "epochs": 1 if args.smoke else args.epochs,
        "batch_size": args.batch_size, "smoke": args.smoke,
        "torch": torch.__version__, "gpu_cuda": torch.cuda.is_available(), "gpu_mps": torch.backends.mps.is_available()})
    print("Model output:", args.out / "final")
    print("SMOKE MODE: not a usable churn model" if args.smoke else "Training completed; evaluate DEV before final TEST")

if __name__ == "__main__":
    main()
