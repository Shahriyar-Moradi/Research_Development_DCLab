"""Pitfalls campaign: measure what common notebook mistakes actually cost.

The notebook copilot (``dclab_rnd.copilot``) flags methodology mistakes. A flag is
only trustworthy if we know how much the mistake matters, so each pitfall here is
run both the WRONG way and the RIGHT way on identical data, and the difference in
the reported score is recorded as evidence.

Pitfalls measured (``campaigns/pitfalls_v1/``):

PIT-001  Scaling / imputation fitted on all rows before the split
PIT-002  Feature selection fitted on all rows before cross-validation
PIT-003  Oversampling the minority class before the split
PIT-004  Target-mean encoding computed on all rows before the split
PIT-005  Random split on data that arrives over time
PIT-006  Choosing among many configurations by their holdout score

CLI::

    python -m dclab_rnd.pitfalls run [--only PIT-002] [--force]
    python -m dclab_rnd.pitfalls report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "pitfalls_v1"
OUT = ROOT / "campaigns" / CAMPAIGN_ID
RESULTS = OUT / "results"
SEEDS = (11, 22, 33)
ROW_CAP = 8000

BLOCKED = {
    "bank_marketing": ["duration"],
    "online_shoppers": [
        "PageValues", "Administrative", "Administrative_Duration", "Informational",
        "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates",
    ],
}


# --------------------------------------------------------------------------- data


def load_external(key: str, cap: int = ROW_CAP, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.read_parquet(ROOT / "external_data" / key / "X.parquet")
    y = pd.read_parquet(ROOT / "external_data" / key / "y.parquet").iloc[:, 0].to_numpy().astype(int)
    X = X.drop(columns=[c for c in BLOCKED.get(key, []) if c in X.columns])
    # LightGBM rejects JSON characters in names; keep names unique after cleaning.
    X.columns = [
        str(c) if re.fullmatch(r"[0-9A-Za-z_.-]+", str(c)) else f"f{i}_{re.sub(r'[^0-9A-Za-z_]', '_', str(c))}"
        for i, c in enumerate(X.columns)
    ]
    if len(X) > cap:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(X), size=cap, replace=False))
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    return X.astype(float), y


def data_paths(keys: list[str]) -> list[Path]:
    paths = []
    for key in keys:
        if key == "telco_churn":
            paths.append(ROOT / "data" / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
        elif key == "hyperack":
            paths.append(ROOT / "data" / "hyperack" / "hyper_ackt-dataset.csv")
        else:
            paths += [ROOT / "external_data" / key / "X.parquet", ROOT / "external_data" / key / "y.parquet"]
    return paths


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max())}


# --------------------------------------------------------------------------- pitfalls


def pit_preprocess_before_split() -> dict[str, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    datasets = ["adult", "bank_marketing", "credit_default", "spambase", "german_credit", "online_shoppers"]
    rows = []
    for key in datasets:
        deltas, wrong_scores, right_scores = [], [], []
        for seed in SEEDS:
            X, y = load_external(key, seed=seed)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
            # WRONG: statistics from every row, including the future test rows
            prep = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()).fit(X)
            wrong = LogisticRegression(max_iter=2000).fit(prep.transform(Xtr), ytr)
            w = _auc(yte, wrong.predict_proba(prep.transform(Xte))[:, 1])
            # RIGHT: statistics from the training rows only
            right = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
            r = _auc(yte, right.predict_proba(Xte)[:, 1])
            wrong_scores.append(w); right_scores.append(r); deltas.append(w - r)
        rows.append({"dataset": key, "wrong_auc": _summary(wrong_scores), "right_auc": _summary(right_scores), "inflation": _summary(deltas)})
    worst = max(rows, key=lambda r: abs(r["inflation"]["mean"]))
    return {
        "pitfall": "preprocess_before_split",
        "question": "How much does fitting an imputer and scaler on all rows before the train/test split inflate the holdout score?",
        "hypothesis": "For row-wise scaling and mean imputation the inflation is tiny; the habit matters because the same mistake with target-aware or high-capacity transforms is large.",
        "setup_summary": f"Logistic regression with mean imputation + standard scaling on {len(datasets)} datasets (≤{ROW_CAP} rows each), {len(SEEDS)} random stratified 80/20 splits; wrong = preprocessing fitted on all rows, right = fitted on training rows only.",
        "evidence": {"datasets": rows, "largest_abs_inflation": {"dataset": worst["dataset"], "mean": worst["inflation"]["mean"]}},
        "claims": [
            ("fact", f"Fitting mean-imputation and scaling before the split changed holdout ROC-AUC by at most {abs(worst['inflation']['mean']):.4f} on average ({worst['dataset']}).", ["evidence.datasets"]),
            ("recommendation", "Still fit every preprocessing step inside the training data (a Pipeline): the measured cost is small here, but the identical mistake with feature selection, oversampling or target encoding is large (PIT-002, PIT-003, PIT-004).", ["evidence.datasets"]),
        ],
        "limitations": ["Simple, unsupervised, row-wise transforms only; learned or target-aware transforms behave very differently."],
        "datasets_used": datasets,
    }


def pit_selection_before_cv() -> dict[str, Any]:
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def run(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> tuple[float, float]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        selected = SelectKBest(f_classif, k=k).fit(X, y).transform(X)  # WRONG: uses every row's label
        wrong = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), selected, y, cv=cv, scoring="roc_auc").mean()
        right = cross_val_score(make_pipeline(StandardScaler(), SelectKBest(f_classif, k=k), LogisticRegression(max_iter=2000)), X, y, cv=cv, scoring="roc_auc").mean()
        return float(wrong), float(right)

    rows = []
    wrong_s, right_s = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(200, 2000))
        y = rng.integers(0, 2, size=200)
        w, r = run(X, y, 20, seed)
        wrong_s.append(w); right_s.append(r)
    rows.append({"dataset": "noise_canary (200 rows, 2000 pure-noise features, random labels)", "k": 20,
                 "wrong_auc": _summary(wrong_s), "right_auc": _summary(right_s),
                 "inflation": _summary([a - b for a, b in zip(wrong_s, right_s)])})
    for key, k in (("spambase", 10), ("german_credit", 5), ("credit_default", 5)):
        wrong_s, right_s = [], []
        for seed in SEEDS:
            X, y = load_external(key, cap=4000, seed=seed)
            w, r = run(X.fillna(X.mean()).to_numpy(), y, k, seed)
            wrong_s.append(w); right_s.append(r)
        rows.append({"dataset": key, "k": k, "wrong_auc": _summary(wrong_s), "right_auc": _summary(right_s),
                     "inflation": _summary([a - b for a, b in zip(wrong_s, right_s)])})
    canary = rows[0]
    real_max = max(r["inflation"]["mean"] for r in rows[1:])
    return {
        "pitfall": "selection_before_cv",
        "question": "How much does selecting features on all rows before cross-validation inflate the CV score?",
        "hypothesis": "With many candidate features and few rows, selection before CV manufactures signal from noise; with few features and many rows the inflation is small.",
        "setup_summary": f"SelectKBest(f_classif) + logistic regression, 5-fold CV, {len(SEEDS)} seeds. A pure-noise canary (random labels, 2000 features, 200 rows) plus three real datasets (≤4000 rows). Wrong = selection fitted once on all rows; right = selection inside each training fold.",
        "evidence": {"datasets": rows},
        "claims": [
            ("fact", f"On pure noise with random labels, selecting 20 of 2000 features before CV produced ROC-AUC {canary['wrong_auc']['mean']:.3f}; doing it inside the folds gave {canary['right_auc']['mean']:.3f} (chance is 0.5).", ["evidence.datasets"]),
            ("fact", f"On the real datasets (few features, thousands of rows) the inflation was at most {real_max:+.4f}.", ["evidence.datasets"]),
            ("recommendation", "Put feature selection inside the cross-validated pipeline. The danger grows with the number of candidate features relative to rows, so wide data (text, genomics, many engineered features) is where this mistake fabricates results.", ["evidence.datasets"]),
        ],
        "limitations": ["Univariate filter selection only; wrapper/model-based selection can inflate more."],
        "datasets_used": ["spambase", "german_credit", "credit_default"],
    }


def pit_oversample_before_split() -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    def oversample(X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(seed)
        minority = int(np.argmin(np.bincount(y)))
        idx_min = np.flatnonzero(y == minority)
        extra = rng.choice(idx_min, size=int((y != minority).sum() - len(idx_min)), replace=True)
        idx = np.concatenate([np.arange(len(y)), extra])
        return X.iloc[idx].reset_index(drop=True), y[idx]

    datasets = ["bank_marketing", "credit_default", "online_shoppers", "german_credit"]
    rows = []
    for key in datasets:
        wrong_s, right_s = [], []
        for seed in SEEDS:
            X, y = load_external(key, cap=6000, seed=seed)
            X = X.fillna(X.median())
            # WRONG: duplicate minority rows first, then split -> copies land in both train and test
            Xo, yo = oversample(X, y, seed)
            Xtr, Xte, ytr, yte = train_test_split(Xo, yo, test_size=0.2, stratify=yo, random_state=seed)
            m = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
            wrong_s.append(_auc(yte, m.predict_proba(Xte)[:, 1]))
            # RIGHT: split first, oversample the training part only, evaluate on untouched rows
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
            Xtr_o, ytr_o = oversample(Xtr.reset_index(drop=True), ytr, seed)
            m = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed).fit(Xtr_o, ytr_o)
            right_s.append(_auc(yte, m.predict_proba(Xte)[:, 1]))
        rows.append({"dataset": key, "wrong_auc": _summary(wrong_s), "right_auc": _summary(right_s),
                     "inflation": _summary([a - b for a, b in zip(wrong_s, right_s)])})
    best = max(rows, key=lambda r: r["inflation"]["mean"])
    return {
        "pitfall": "oversample_before_split",
        "question": "How much does oversampling the minority class before the split inflate the holdout score?",
        "hypothesis": "Duplicated minority rows end up in both train and test, so a flexible model recognizes them and the holdout score is inflated.",
        "setup_summary": f"Random minority oversampling to a 50/50 balance, random forest (200 trees), {len(datasets)} imbalanced datasets (≤6000 rows), {len(SEEDS)} seeds. Wrong = oversample then split; right = split, oversample the training rows only.",
        "evidence": {"datasets": rows},
        "claims": [
            ("fact", f"Oversampling before the split inflated holdout ROC-AUC by up to {best['inflation']['mean']:+.4f} on average ({best['dataset']}: {best['wrong_auc']['mean']:.4f} reported vs {best['right_auc']['mean']:.4f} honest).", ["evidence.datasets"]),
            ("recommendation", "Resample only inside the training data (imblearn Pipeline or after the split), and evaluate on untouched, naturally imbalanced rows with a metric that respects imbalance such as average precision.", ["evidence.datasets"]),
        ],
        "limitations": ["Random duplication oversampling; SMOTE interpolates new points and leaks through near-duplicates instead of exact copies."],
        "datasets_used": datasets,
    }


def pit_target_encoding_before_split() -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, train_test_split

    def encode(train_keys: pd.Series, train_y: np.ndarray, apply_keys: pd.Series, smoothing: float = 10.0) -> np.ndarray:
        prior = float(train_y.mean())
        stats = pd.DataFrame({"k": train_keys.to_numpy(), "y": train_y}).groupby("k")["y"].agg(["sum", "count"])
        enc = (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)
        return apply_keys.map(enc).fillna(prior).to_numpy()

    cases = []
    telco = pd.read_csv(ROOT / "data" / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="coerce")
    cases.append(("telco_churn", "customerID (unique per row)", telco[["tenure", "MonthlyCharges", "TotalCharges"]],
                  telco["customerID"], (telco["Churn"] == "Yes").astype(int).to_numpy()))
    X, y = load_external("adult", seed=0)
    cases.append(("adult", "occupation (~15 levels)", X.drop(columns=["occupation"]), X["occupation"].astype(str), y))
    hyper = pd.read_csv(ROOT / "data" / "hyperack" / "hyper_ackt-dataset.csv").dropna(subset=["total_distance"])
    num = hyper[["total_distance", "sum_product", "first_customer_fare", "weekday"]]
    cases.append(("hyperack", "deliverey_category_id", num, hyper["deliverey_category_id"].astype(str), hyper["hyper_ack"].to_numpy().astype(int)))

    rows = []
    for key, column, base, keys, target in cases:
        wrong_s, right_s = [], []
        for seed in SEEDS:
            idx_tr, idx_te = train_test_split(np.arange(len(target)), test_size=0.2, stratify=target, random_state=seed)
            full = encode(keys, target, keys, smoothing=1.0)  # WRONG: every row's own label is inside its encoding
            Xw = base.assign(enc=full)
            m = HistGradientBoostingClassifier(random_state=seed).fit(Xw.iloc[idx_tr], target[idx_tr])
            wrong_s.append(_auc(target[idx_te], m.predict_proba(Xw.iloc[idx_te])[:, 1]))
            # RIGHT: training rows get out-of-fold encodings (never their own label); test rows use train statistics
            tr_keys, tr_y = keys.iloc[idx_tr].reset_index(drop=True), target[idx_tr]
            tr_enc = np.zeros(len(idx_tr))
            for fit_idx, enc_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(tr_keys, tr_y):
                tr_enc[enc_idx] = encode(tr_keys.iloc[fit_idx], tr_y[fit_idx], tr_keys.iloc[enc_idx], smoothing=1.0)
            te_enc = encode(tr_keys, tr_y, keys.iloc[idx_te], smoothing=1.0)
            m = HistGradientBoostingClassifier(random_state=seed).fit(base.iloc[idx_tr].assign(enc=tr_enc), target[idx_tr])
            right_s.append(_auc(target[idx_te], m.predict_proba(base.iloc[idx_te].assign(enc=te_enc))[:, 1]))
        rows.append({"dataset": key, "encoded_column": column, "wrong_auc": _summary(wrong_s), "right_auc": _summary(right_s),
                     "inflation": _summary([a - b for a, b in zip(wrong_s, right_s)])})
    worst = max(rows, key=lambda r: r["inflation"]["mean"])
    low = min(rows, key=lambda r: r["inflation"]["mean"])
    return {
        "pitfall": "target_encoding_before_split",
        "question": "How much does computing target-mean encodings on all rows before the split inflate the holdout score?",
        "hypothesis": "The fewer rows per category, the more each encoding contains the row's own label; for identifier-like columns the leak is near total.",
        "setup_summary": "Mean-target encoding (smoothing 1) of one categorical column plus a few numeric columns, histogram gradient boosting, 80/20 stratified split, 3 seeds. Wrong = encoding computed on all rows; right = out-of-fold encodings for training rows, training statistics applied to test rows.",
        "evidence": {"datasets": rows},
        "claims": [
            ("fact", f"Encoding {worst['encoded_column']} with the target before the split gave ROC-AUC {worst['wrong_auc']['mean']:.4f} on {worst['dataset']} versus {worst['right_auc']['mean']:.4f} honestly (inflation {worst['inflation']['mean']:+.4f}).", ["evidence.datasets"]),
            ("fact", f"For a low-cardinality column ({low['encoded_column']} on {low['dataset']}) the inflation was {low['inflation']['mean']:+.4f}.", ["evidence.datasets"]),
            ("recommendation", "Compute target encodings out-of-fold on training data only, and never target-encode identifiers; an identifier that looks predictive is memorizing labels. Even encoding training rows with their own label (without out-of-fold) teaches the model to trust a column that is useless on new rows.", ["evidence.datasets"]),
        ],
        "limitations": ["One encoding scheme and smoothing value; out-of-fold encoders reduce but do not remove the risk for rare categories."],
        "datasets_used": ["telco_churn", "adult", "hyperack"],
    }


def pit_random_vs_time_split() -> dict[str, Any]:
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(ROOT / "data" / "hyperack" / "hyper_ackt-dataset.csv").dropna(subset=["total_distance"])
    ts = pd.to_datetime(df["first_created_at"], errors="coerce", utc=True)
    df = df.assign(_ts=ts, hour=ts.dt.hour).dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
    features = ["deliverey_category_id", "weekday", "time_bucket", "total_distance", "sum_product", "source_latitude",
                "source_longitude", "destination_latitude", "destination_longitude", "first_customer_fare", "hour"]
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = df["hyper_ack"].to_numpy().astype(int)

    def fit_eval(tr: np.ndarray, te: np.ndarray, seed: int) -> float:
        m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1)
        m.fit(X.iloc[tr], y[tr])
        return _auc(y[te], m.predict_proba(X.iloc[te])[:, 1])

    random_scores = []
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        random_scores.append(fit_eval(tr, te, seed))
    cut = int(len(y) * 0.8)
    time_score = fit_eval(np.arange(cut), np.arange(cut, len(y)), SEEDS[0])
    gap = float(np.mean(random_scores) - time_score)
    period = (str(df["_ts"].iloc[0].date()), str(df["_ts"].iloc[cut].date()), str(df["_ts"].iloc[-1].date()))
    return {
        "pitfall": "random_split_on_time_data",
        "question": "On data that arrives over time, how different is a random holdout from a train-on-past, test-on-future holdout?",
        "hypothesis": "A random split mixes future rows into training, so it estimates interpolation and is usually more optimistic than a forward-in-time holdout.",
        "setup_summary": f"HyperAck orders ({len(y):,} rows, {period[0]} to {period[2]}), 11 leakage-safe features, LightGBM. Random = stratified 80/20 over {len(SEEDS)} seeds; time-ordered = train on the first 80% by timestamp, test on the last 20% (from {period[1]}).",
        "evidence": {"random_split_auc": _summary(random_scores), "time_ordered_auc": time_score, "optimism_gap": gap,
                     "rows": int(len(y)), "time_range": list(period), "features": features},
        "claims": [
            ("fact", f"A random split reported ROC-AUC {np.mean(random_scores):.4f}; training on the past and testing on the last 20% of time gave {time_score:.4f} (random minus time-ordered {gap:+.4f})."
             + (" Here the random split was NOT optimistic: the process was stable enough over this window that a forward-in-time test scored as well or better." if gap <= 0.002 else " The random split was optimistic about future performance."),
             ["evidence.random_split_auc", "evidence.time_ordered_auc"]),
            ("recommendation", "When rows have timestamps and the model will score future rows, validate forward in time (TimeSeriesSplit or a time cut-off) and report that number; a random split answers a different question. Do not assume the direction of the gap: measure it, because a stable process can show none.", ["evidence.optimism_gap"]),
        ],
        "limitations": ["One dataset and one cut-off; the gap depends on how fast the process drifts."],
        "datasets_used": ["hyperack"],
    }


def pit_holdout_reuse() -> dict[str, Any]:
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split

    datasets = ["bank_marketing", "credit_default", "spambase", "german_credit", "online_shoppers"]
    n_configs = 30
    rows = []
    for key in datasets:
        biases, rank_on_other = [], []
        for seed in SEEDS:
            X, y = load_external(key, cap=6000, seed=seed)
            idx = np.arange(len(y))
            tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
            a, b = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
            rng = np.random.default_rng(seed)
            scores = []
            for _ in range(n_configs):
                params = {"n_estimators": int(rng.integers(50, 400)), "learning_rate": float(10 ** rng.uniform(-2.3, -0.5)),
                          "num_leaves": int(rng.integers(4, 128)), "min_child_samples": int(rng.integers(5, 100)),
                          "colsample_bytree": float(rng.uniform(0.4, 1.0))}
                m = LGBMClassifier(random_state=seed, verbose=-1, **params).fit(X.iloc[tr], y[tr])
                scores.append((_auc(y[a], m.predict_proba(X.iloc[a])[:, 1]), _auc(y[b], m.predict_proba(X.iloc[b])[:, 1])))
            arr = np.asarray(scores)
            # Select on one holdout, evaluate on the other, in both directions. Subtract the average gap
            # between the two holdouts so that one holdout simply being easier does not count as bias.
            for sel, other in ((0, 1), (1, 0)):
                pick = int(np.argmax(arr[:, sel]))
                biases.append(float((arr[pick, sel] - arr[pick, other]) - (arr[:, sel] - arr[:, other]).mean()))
                rank_on_other.append(int((arr[:, other] > arr[pick, other]).sum()) + 1)
        rows.append({"dataset": key, "configs": n_configs, "selection_bias": _summary(biases),
                     "picked_config_rank_on_untouched_holdout": _summary(rank_on_other)})
    avg = float(np.mean([r["selection_bias"]["mean"] for r in rows]))
    worst = max(rows, key=lambda r: r["selection_bias"]["mean"])
    avg_rank = float(np.mean([r["picked_config_rank_on_untouched_holdout"]["mean"] for r in rows]))
    return {
        "pitfall": "holdout_reuse_for_selection",
        "question": "If we pick the best of many configurations by their holdout score, how optimistic is that holdout score?",
        "hypothesis": "Selecting the maximum of noisy scores selects favorable noise; the chosen model's holdout score overstates its performance on fresh data.",
        "setup_summary": f"{n_configs} random LightGBM configurations per run on a 60/20/20 split into train and two holdouts, {len(datasets)} datasets (≤6000 rows), {len(SEEDS)} seeds, both selection directions. Bias = the picked model's score drop from the selection holdout to the untouched one, minus the average drop across all configurations.",
        "evidence": {"datasets": rows, "average_selection_bias": avg, "average_rank_of_pick_on_untouched_holdout": avg_rank},
        "claims": [
            ("fact", f"Choosing the best of {n_configs} configurations by holdout score overstated ROC-AUC by {avg:+.4f} on average and by {worst['selection_bias']['mean']:+.4f} on {worst['dataset']}; on the untouched holdout the pick ranked {avg_rank:.1f} of {n_configs} on average.", ["evidence.datasets", "evidence.average_selection_bias"]),
            ("recommendation", "Tune with cross-validation on training data, then touch the holdout once. If the holdout was used to choose, the reported score is biased upward; confirm on a fresh split or use nested CV. The bias grows with the number of configurations tried and shrinks with holdout size.", ["evidence.average_selection_bias"]),
        ],
        "limitations": ["Configurations from one model family are highly correlated, which keeps the bias small; comparing many unrelated pipelines inflates more."],
        "datasets_used": datasets,
    }


PITFALLS: dict[str, tuple[str, Callable[[], dict[str, Any]]]] = {
    "PIT-001": ("preprocess_before_split", pit_preprocess_before_split),
    "PIT-002": ("selection_before_cv", pit_selection_before_cv),
    "PIT-003": ("oversample_before_split", pit_oversample_before_split),
    "PIT-004": ("target_encoding_before_split", pit_target_encoding_before_split),
    "PIT-005": ("random_split_on_time_data", pit_random_vs_time_split),
    "PIT-006": ("holdout_reuse_for_selection", pit_holdout_reuse),
}


# --------------------------------------------------------------------------- persistence


def result_path(pit_id: str) -> Path:
    return RESULTS / f"{pit_id}_{PITFALLS[pit_id][0]}.json"


def run(only: list[str] | None = None, force: bool = False) -> None:
    from dclab_rnd.provenance import capture_provenance

    RESULTS.mkdir(parents=True, exist_ok=True)
    for pit_id, (name, fn) in PITFALLS.items():
        if only and pit_id not in only:
            continue
        path = result_path(pit_id)
        if path.exists() and not force:
            print(f"skip {pit_id} (exists)")
            continue
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        t0 = time.perf_counter()
        body = fn()
        claims = [
            {"claim_id": f"{pit_id}-C{i}", "kind": kind, "statement": text, "evidence": refs,
             "limitations": body.get("limitations", [])}
            for i, (kind, text, refs) in enumerate(body.pop("claims"), 1)
        ]
        used = body.pop("datasets_used")
        payload = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "experiment_id": pit_id,
            "kind": "pitfall",
            "pitfall": body.pop("pitfall"),
            "dataset": "multiple",
            "datasets": used,
            "question": body.pop("question"),
            "hypothesis": body.pop("hypothesis"),
            "setup_summary": body.pop("setup_summary"),
            "evidence": body.pop("evidence"),
            "claims": claims,
            "status": "completed",
            "started_at": started,
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "provenance": capture_provenance(ROOT, data_paths=data_paths(used), random_state=list(SEEDS)),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{pit_id} {name}: {claims[0]['statement']} ({payload['elapsed_seconds']}s)")
    write_report()


def load_results() -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("PIT-*.json"))]


def write_report() -> None:
    results = load_results()
    lines = [
        "# Pitfalls campaign: what common notebook mistakes really cost",
        "",
        "Each mistake was run the wrong way and the right way on identical data. The difference is how much the "
        "reported score lies. These measurements back the severity and the proof shown by the notebook copilot.",
        "",
        "| ID | Mistake | Measured effect |",
        "|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['experiment_id']} | {r['pitfall'].replace('_', ' ')} | {r['claims'][0]['statement']} |")
    lines.append("")
    for r in results:
        lines += [f"## {r['experiment_id']} · {r['pitfall'].replace('_', ' ')}", "", f"**Question.** {r['question']}", "",
                  f"**Setup.** {r['setup_summary']}", ""]
        lines += [f"- **{c['kind']}**: {c['statement']}" for c in r["claims"]]
        if r["claims"] and r["claims"][0].get("limitations"):
            lines += ["", "Limitations: " + " ".join(r["claims"][0]["limitations"])]
        lines += ["", f"Evidence: `campaigns/{CAMPAIGN_ID}/results/{result_path(r['experiment_id']).name}`", ""]
    lines += ["Re-run: `python -m dclab_rnd.pitfalls run --force`.", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PITFALLS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.pitfalls", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--only", help="comma-separated PIT ids")
    r.add_argument("--force", action="store_true")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.only.split(",") if args.only else None, force=args.force)
    else:
        write_report()
        print(f"Wrote {OUT / 'PITFALLS_REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
