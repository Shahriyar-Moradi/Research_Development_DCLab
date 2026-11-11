"""Convert IBM Telco rows into GLiNER2 classification JSONL, with frozen group-safe splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import FEATURES, format_example, read_data, serialize_customer, dump


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    p.add_argument("--out", type=Path, default=Path("research/churn-prediction/experiments/gliner_decide/processed"))
    args = p.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"Dataset not found: {args.csv}. Run from the DCLab repository root.")
    args.out.mkdir(parents=True, exist_ok=True)
    df = read_data(args.csv)
    texts = df.apply(serialize_customer, axis=1)
    # Identical feature records must never appear in different splits.
    groups = texts.map(lambda t: hashlib.sha256(t.encode()).hexdigest()).to_numpy()
    y = (df["Churn"] == "Yes").astype(int).to_numpy()
    indices = np.arange(len(df))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    remaining, test = next(splitter.split(indices, y, groups=groups))
    val_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    tr_local, dev_local = next(val_splitter.split(remaining, y[remaining], groups=groups[remaining]))
    train, dev = remaining[tr_local], remaining[dev_local]
    assigned = {"train": train.tolist(), "dev": dev.tolist(), "test": test.tolist()}
    assert len(set(train) | set(dev) | set(test)) == len(df)
    assert not (set(groups[train]) & set(groups[dev]))
    assert not (set(groups[train]) & set(groups[test]))
    assert not (set(groups[dev]) & set(groups[test]))
    split_name = np.empty(len(df), dtype=object)
    for name, ids in assigned.items():
        split_name[ids] = name
        rows = [format_example(texts.iloc[i], "churn" if y[i] else "not_churn") for i in ids]
        write_jsonl(args.out / f"{name}.jsonl", rows)
    # Identifiers and target never become model input; source row index only for joining splits.
    pd.DataFrame({"source_index": indices, "split": split_name, "target": y}).to_csv(args.out / "splits.csv", index=False)
    source_hash = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    manifest = {
        "dataset": "IBM Telco Customer Churn", "source_file": str(args.csv), "source_sha256": source_hash,
        "n": len(df), "trainable_features": FEATURES, "excluded": ["customerID", "Churn"],
        "label_definition": "Churn Yes/No from original cross-sectional sample; no verified prospective prediction horizon",
        "splitting": "StratifiedGroupKFold twice on identical serialized feature strings; random states 42/43",
        "splits": {k: {"rows": len(v), "churn": int(sum(y[v])), "rate": round(float(np.mean(y[v])), 4)} for k, v in assigned.items()},
        "warning": "Test is sealed for one final comparison. No time-based split possible with this public snapshot."
    }
    dump(args.out / "manifest.json", manifest)
    print(json.dumps({"output": str(args.out), "splits": manifest["splits"]}, indent=2))
    print("Example:", texts.iloc[int(train[0])][:440], "...", sep="\n")

if __name__ == "__main__":
    main()
