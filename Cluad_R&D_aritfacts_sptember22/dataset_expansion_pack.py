#!/usr/bin/env python3
"""
dataset_expansion_pack.py
==========================
ADDITIVE. Does not touch, remove, or re-derive anything in your existing
`external_data/` datasets (adult, bank_marketing, breast_cancer, ...). It
only adds four NEW datasets that plug into the same
`external_data/<name>/{X.parquet, y.parquet, meta.json}` convention your
`dclab_rnd` pipeline already reads (see `provenance.data_sha256` paths in
any existing result JSON, e.g. `external_data/adult/X.parquet`).

Once converted, a new dataset should run through the exact same commands
you already use, e.g.:

    .venv/bin/python -m dclab_rnd campaign run --dataset credit_card_fraud

Why these four
---------------
Your first 50-experiment campaign is 10 datasets, all UCI, all binary
classification. These four were chosen specifically to stress DIFFERENT
failure modes than that first round did:

| Dataset (slug)              | Category          | New failure mode it stresses                                   |
|------------------------------|-------------------|------------------------------------------------------------------|
| credit_card_fraud            | imbalanced fraud   | ~0.17% positive rate — tests metric choice (PR-AUC vs ROC-AUC), resampling vs. cost-weighting, and calibration under extreme imbalance |
| otto_group_products          | multiclass         | 9-class target with anonymized count features — first non-binary target in the registry |
| rossmann_store_sales         | time-series        | Per-store daily sales; `Open`/`Promo` change over time — a real test of temporal splitting and window/future leakage |
| ecommerce_clothing_reviews   | text + tabular     | Free-text `Review Text` alongside tabular columns — forces an explicit decision about whether/how embeddings become "features" and whether they pass the same leakage/availability tests as any other column |

Where to get the real files (run these on your own machine — this
sandbox cannot reach kaggle.com or huggingface.co):

    # Kaggle CLI (requires `pip install kaggle` and an API token in ~/.kaggle/kaggle.json)
    kaggle datasets download -d mlg-ulb/creditcardfraud -p raw/ --unzip
    kaggle competitions download -c otto-group-product-classification-challenge -p raw/ --unzip
    kaggle competitions download -c rossmann-store-sales -p raw/ --unzip
    kaggle datasets download -d nicapotato/womens-ecommerce-clothing-reviews -p raw/ --unzip

Then convert each with this script, e.g.:

    python dataset_expansion_pack.py --dataset credit_card_fraud \
        --csv raw/creditcard.csv --out-dir external_data

IMPORTANT — this script was built and smoke-tested against tiny
HAND-WRITTEN samples that match each dataset's PUBLISHED column schema
(see `raw_samples/` if you generated one), because this environment
cannot download the real files. Verify your real CSV's column names
match the `target_column` / `id_columns` below before your first real
run — schemas occasionally change between Kaggle dataset versions.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATASET_SPECS = {
    "credit_card_fraud": {
        "category": "imbalanced_fraud",
        "source": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "problem_type": "binary_classification",
        "target_column": "Class",
        "id_columns": [],
        # PCA-anonymized V1..V28 are legitimately safe (no lineage risk),
        # but Time/Amount deserve an explicit decision-time note.
        "declared_risk_columns": {
            "Time": "Seconds since the first transaction in the extract — fine as a recency feature, "
                    "but must be recomputed relative to a real prediction moment in production, not "
                    "relative to this static file's first row.",
        },
        "notes": (
            "Positive rate is ~0.17% in the real file. Use PR-AUC and cost-weighted metrics "
            "alongside ROC-AUC (DCLAB-R18). Stratify every split and every CV fold on Class."
        ),
    },
    "otto_group_products": {
        "category": "multiclass",
        "source": "https://www.kaggle.com/competitions/otto-group-product-classification-challenge",
        "problem_type": "multiclass_classification",
        "target_column": "target",
        "id_columns": ["id"],
        "declared_risk_columns": {},
        "notes": (
            "9-class target (Class_1..Class_9), 93 anonymized integer count features in the real "
            "file. Good first multiclass case: features are already anonymized counts, so leakage "
            "review can focus purely on split/preprocessing discipline (DCLAB-R02, DCLAB-R03) "
            "rather than semantic column meaning."
        ),
    },
    "rossmann_store_sales": {
        "category": "time_series",
        "source": "https://www.kaggle.com/competitions/rossmann-store-sales",
        "problem_type": "regression_time_series",
        "target_column": "Sales",
        "id_columns": [],
        "declared_risk_columns": {
            "Customers": "Only known AFTER a day closes (it's a same-day outcome, like Sales itself) "
                         "— post-outcome leakage if used to predict same-day Sales. Exclude from "
                         "the feature set; this is a canonical post-outcome-leakage teaching case.",
            "Open": "0 on days the store is closed, which deterministically forces Sales to 0. Keep "
                    "as a feature but ALWAYS split by Date, never randomly, or the model 'peeks' at "
                    "future closures.",
        },
        "notes": (
            "Split by Date (train on early dates, evaluate on later dates) — never a random "
            "row split. This is the repo's first dataset where DCLAB-R02 (split by time "
            "direction) and DCLAB-R19 (temporal generalization slices) become the central test, "
            "not a side note."
        ),
    },
    "ecommerce_clothing_reviews": {
        "category": "text_plus_tabular",
        "source": "https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews",
        "problem_type": "binary_classification",
        "target_column": "Recommended IND",
        "id_columns": ["Clothing ID", "Title"],
        "declared_risk_columns": {
            "Review Text": "Free text written by the same reviewer at the same time as the "
                           "recommendation decision — almost certainly available at the "
                           "prediction moment, but confirm the real production use case (e.g. "
                           "predicting recommendation from a review being TYPED vs. already "
                           "SUBMITTED) before treating it as safe.",
            "Positive Feedback Count": "Accumulates AFTER the review is posted (other users voting "
                                       "it helpful) — a classic post-outcome leakage candidate if "
                                       "the prediction moment is 'at review submission time'.",
        },
        "notes": (
            "First dataset with a free-text column. Forces a concrete decision: does 'Review Text' "
            "get embedded and treated as N numeric features (then it must pass the SAME "
            "leakage/availability/reliability tests as any other column, per DCLAB-R10), or is it "
            "out of scope for this modeling round? Decide and record it explicitly — don't let it "
            "slide in through vectorization without review."
        ),
    },
}


def build_external_dataset(csv_path: Path, dataset_slug: str, out_dir: Path) -> None:
    if dataset_slug not in DATASET_SPECS:
        raise SystemExit(f"Unknown dataset '{dataset_slug}'. Known: {list(DATASET_SPECS)}")
    spec = DATASET_SPECS[dataset_slug]

    df = pd.read_csv(csv_path)
    target_col = spec["target_column"]
    if target_col not in df.columns:
        raise SystemExit(
            f"Target column '{target_col}' not found in {csv_path}. "
            f"Columns present: {list(df.columns)}. Check the schema note in DATASET_SPECS."
        )

    id_cols = [c for c in spec["id_columns"] if c in df.columns]
    y = df[[target_col]].copy()
    X = df.drop(columns=[target_col] + id_cols)

    dest = out_dir / dataset_slug
    dest.mkdir(parents=True, exist_ok=True)
    X.to_parquet(dest / "X.parquet", index=False)
    y.to_parquet(dest / "y.parquet", index=False)

    meta = {
        "dataset": dataset_slug,
        "category": spec["category"],
        "problem_type": spec["problem_type"],
        "source": spec["source"],
        "target_column": target_col,
        "dropped_id_columns": id_cols,
        "row_count": int(len(df)),
        "feature_count": int(X.shape[1]),
        "declared_risk_columns": spec["declared_risk_columns"],
        "notes": spec["notes"],
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "converted_from": str(csv_path),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{dataset_slug}] wrote {X.shape[0]} rows x {X.shape[1]} features to {dest}/")
    print(f"[{dataset_slug}] declared risk columns: {list(spec['declared_risk_columns'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASET_SPECS), help="Dataset slug to convert.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to the downloaded raw CSV.")
    parser.add_argument("--out-dir", type=Path, default=Path("external_data"), help="Root external_data directory.")
    args = parser.parse_args()

    build_external_dataset(args.csv, args.dataset, args.out_dir)


if __name__ == "__main__":
    main()
