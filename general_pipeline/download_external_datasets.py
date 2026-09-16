"""Download and cache the 10 external UCI tabular classification datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from general_pipeline.external_catalog import DATASET_CATALOG, prepare_xy  # noqa: E402

DATA_DIR = ROOT / "external_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _pick_target_column(features: pd.DataFrame, targets: pd.DataFrame, key: str) -> pd.Series:
    if targets.shape[1] == 1:
        return targets.iloc[:, 0]
    # Prefer common target names
    preferred = {
        "adult": ["income"],
        "bank_marketing": ["y"],
        "breast_cancer": ["Diagnosis", "diagnosis"],
        "heart_disease": ["num", "target"],
        "credit_default": ["Y", "default payment next month", "default.payment.next.month"],
        "german_credit": ["class"],
        "mushroom": ["poisonous", "class"],
        "spambase": ["Class", "class"],
        "online_shoppers": ["Revenue"],
        "wine_quality": ["quality"],
    }.get(key, [])
    for name in preferred:
        if name in targets.columns:
            return targets[name]
    # fallback first column
    return targets.iloc[:, 0]


def download_one(key: str, uci_id: int) -> dict:
    print(f"Fetching UCI id={uci_id} ({key})...", flush=True)
    ds = fetch_ucirepo(id=uci_id)
    X_raw = ds.data.features.copy()
    y_raw = _pick_target_column(X_raw, ds.data.targets.copy(), key)

    # Wine quality may be red/white combined already via features
    X, y = prepare_xy(X_raw, y_raw, key)

    out_dir = DATA_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    X.to_parquet(out_dir / "X.parquet", index=False)
    y.to_frame("target").to_parquet(out_dir / "y.parquet", index=False)

    meta = {
        "key": key,
        "uci_id": uci_id,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "pos_rate": float(y.mean()),
        "feature_names": list(X.columns),
        "name": next(s.name for s in DATASET_CATALOG if s.key == key),
        "url": next(s.url for s in DATASET_CATALOG if s.key == key),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"  saved {key}: rows={meta['n_rows']} feats={meta['n_features']} pos={meta['pos_rate']:.3f}",
        flush=True,
    )
    return meta


def main() -> None:
    summary = []
    errors = []
    for spec in DATASET_CATALOG:
        try:
            if spec.uci_id is None:
                raise ValueError("missing uci_id")
            summary.append(download_one(spec.key, spec.uci_id))
        except Exception as exc:
            msg = f"{spec.key}: {exc}"
            print(f"FAILED {msg}", flush=True)
            errors.append(msg)

    index = {"datasets": summary, "errors": errors}
    (DATA_DIR / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\nDownloaded {len(summary)}/{len(DATASET_CATALOG)} datasets -> {DATA_DIR}")
    if errors:
        print("Errors:", errors)
        sys.exit(1)


if __name__ == "__main__":
    main()
