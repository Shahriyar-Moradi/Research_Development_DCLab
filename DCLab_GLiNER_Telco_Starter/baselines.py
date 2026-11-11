"""Logistic regression and HistGradientBoosting baselines on exactly the GLiNER split."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import FEATURES, NUMERIC, CATEGORICAL, read_data, metrics, dump


def preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC),
        ("category", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ], remainder="drop")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    p.add_argument("--processed", type=Path, default=Path("research/churn-prediction/experiments/gliner_decide/processed"))
    p.add_argument("--out", type=Path, default=Path("research/churn-prediction/experiments/gliner_decide/runs/baselines"))
    args = p.parse_args()
    df = read_data(args.csv)
    splits = pd.read_csv(args.processed / "splits.csv").sort_values("source_index").reset_index(drop=True)
    assert len(df) == len(splits) and (splits.source_index.to_numpy() == np.arange(len(df))).all()
    X, y = df[FEATURES], (df["Churn"] == "Yes").astype(int)
    train = splits["split"] == "train"
    dev = splits["split"] == "dev"
    models = {
        "logistic": LogisticRegression(C=0.2, max_iter=1500, random_state=42),
        "histboost": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, l2_regularization=1, random_state=42),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    scores = {}
    for name, clf in models.items():
        pipeline = Pipeline([("preprocess", preprocessor()), ("model", clf)])
        pipeline.fit(X.loc[train], y.loc[train])
        proba = pipeline.predict_proba(X.loc[dev])[:, 1]
        scores[name] = metrics(y.loc[dev].tolist(), proba.tolist())
        joblib.dump(pipeline, args.out / f"{name}.joblib")
    dump(args.out / "dev_metrics.json", scores)
    print("Development baselines (NOT the final test):")
    print(pd.DataFrame(scores).T.to_string())

if __name__ == "__main__":
    main()
