"""Benchmark TabPFN against XGBoost, LightGBM, and sklearn baselines.

Follows the Prior Labs protocol:
  https://docs.priorlabs.ai/benchmarking
  https://github.com/PriorLabs/TabPFN/blob/main/examples/benchmarking_tabpfn.py

Same stratified 80/20 split for every model. Primary metric is ROC-AUC.
Two train sizes:
  - context: TabPFN-2's 1,024-row budget (fair small-data comparison)
  - full: entire train split (production trees; TabPFN-2 still capped at 1,024)
"""

from __future__ import annotations

import os

# LightGBM/XGBoost OpenMP runtimes can SIGSEGV when mixed with PyTorch MPS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from tab_transformer_model import run_ft_transformer, run_tab_transformer
from train_tabpfn import (
    PART1_PATH,
    ROOT,
    TELCO_PATH,
    THINKING_VERSION,
    default_max_train_rows,
    fit_random_forest,
    load_part1,
    load_telco,
    make_tabpfn_classifier,
    maybe_subsample_train,
    metrics,
    pick_device,
    pick_version,
)

import lightgbm as lgb
import xgboost as xgb

OUT_JSON = ROOT / "results" / "benchmark.json"
OUT_CSV = ROOT / "results" / "benchmark.csv"

XGB_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "auc",
    "objective": "binary:logistic",
    "seed": 42,
}


def cat_num_cols(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = list(X.select_dtypes(include=["category", "object", "bool"]).columns)
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols


def ordinal_matrix(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numeric matrix for GBT wheels that crash on pandas categoricals (macOS)."""
    cat_cols, num_cols = cat_num_cols(X_train)
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                cat_cols,
            ),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ]
    )
    Xtr = pd.DataFrame(pre.fit_transform(X_train), index=X_train.index)
    Xte = pd.DataFrame(pre.transform(X_test), index=X_test.index)
    return Xtr, Xte


def pack(name: str, y_true, y_pred, y_proba, fit_s: float, pred_s: float, notes: str) -> dict:
    out = metrics(y_true, y_pred, y_proba)
    out.update(
        {
            "model": name,
            "fit_seconds": round(fit_s, 3),
            "predict_seconds": round(pred_s, 3),
            "notes": notes,
        }
    )
    return out


def run_tabpfn(
    X_train,
    y_train,
    X_test,
    y_test,
    version: str,
    device: str,
    thinking_effort: str = "high",
    thinking_metric: str = "accuracy",
    thinking_timeout_s: float | None = None,
) -> dict:
    clf = make_tabpfn_classifier(
        version,
        device,
        n_estimators=None,
        thinking_effort=thinking_effort,
        thinking_metric=thinking_metric,
        thinking_timeout_s=thinking_timeout_s,
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    proba = clf.predict_proba(X_test)[:, 1]
    pred_s = time.perf_counter() - t1
    pred = (proba >= 0.5).astype(int)
    notes = "raw DataFrame, no scaling / one-hot"
    if version == THINKING_VERSION:
        notes = (
            f"tabpfn-client thinking_mode=True effort={thinking_effort} "
            f"metric={thinking_metric}"
        )
    return pack(
        f"TabPFN-{version}",
        y_test,
        pred,
        proba,
        fit_s,
        pred_s,
        notes,
    )


def run_xgb(X_train, y_train, X_test, y_test, *, rounds: int, name: str, notes: str) -> dict:
    t0 = time.perf_counter()
    Xtr, Xte = ordinal_matrix(X_train, X_test)
    dtrain = xgb.DMatrix(Xtr, label=np.asarray(y_train))
    dtest = xgb.DMatrix(Xte, label=np.asarray(y_test))
    booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=rounds)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    proba = booster.predict(dtest)
    pred_s = time.perf_counter() - t1
    pred = (proba >= 0.5).astype(int)
    return pack(name, y_test, pred, proba, fit_s, pred_s, notes)


def run_xgb_cv(X_train, y_train, X_test, y_test) -> dict:
    t_cv = time.perf_counter()
    Xtr, Xte = ordinal_matrix(X_train, X_test)
    dtrain = xgb.DMatrix(Xtr, label=np.asarray(y_train))
    cv_result = xgb.cv(
        XGB_PARAMS,
        dtrain,
        num_boost_round=2000,
        nfold=5,
        stratified=True,
        early_stopping_rounds=50,
        seed=42,
    )
    cv_s = time.perf_counter() - t_cv
    best_rounds = max(1, int(round(len(cv_result) * 0.8)))
    t0 = time.perf_counter()
    dtest = xgb.DMatrix(Xte, label=np.asarray(y_test))
    booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=best_rounds)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    proba = booster.predict(dtest)
    pred_s = time.perf_counter() - t1
    pred = (proba >= 0.5).astype(int)
    row = pack(
        "XGBoost (CV-tuned n_estimators)",
        y_test,
        pred,
        proba,
        fit_s,
        pred_s,
        f"5-fold CV early stopping, best_rounds={best_rounds}, cv={cv_s:.1f}s",
    )
    row["cv_seconds"] = round(cv_s, 2)
    row["best_rounds"] = best_rounds
    return row


def run_lgbm(X_train, y_train, X_test, y_test) -> dict:
    # Native pandas-category path segfaults on this macOS LightGBM wheel.
    cat_cols, num_cols = cat_num_cols(X_train)
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                cat_cols,
            ),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ]
    )
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    pred_s = time.perf_counter() - t1
    return pack(
        "LightGBM (defaults)",
        y_test,
        pred,
        proba,
        fit_s,
        pred_s,
        "n_estimators=100, ordinal cats (native cats crash on this wheel)",
    )


def run_hgb(X_train, y_train, X_test, y_test) -> dict:
    cat_cols, num_cols = cat_num_cols(X_train)
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                cat_cols,
            ),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ]
    )
    cat_idx = list(range(len(cat_cols)))
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        random_state=42,
        categorical_features=cat_idx if cat_cols else None,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    pred_s = time.perf_counter() - t1
    return pack(
        "HistGradientBoosting",
        y_test,
        pred,
        proba,
        fit_s,
        pred_s,
        "ordinal cats + median impute, max_iter=200",
    )


def run_logreg(X_train, y_train, X_test, y_test) -> dict:
    cat_cols, num_cols = cat_num_cols(X_train)
    pre = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
        ]
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    pred_s = time.perf_counter() - t1
    return pack(
        "LogisticRegression",
        y_test,
        pred,
        proba,
        fit_s,
        pred_s,
        "one-hot + StandardScaler, class_weight=balanced",
    )


def run_dummy(X_train, y_train, X_test, y_test) -> dict:
    clf = DummyClassifier(strategy="prior", random_state=42)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    proba = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)
    pred_s = time.perf_counter() - t1
    return pack("Dummy (prior)", y_test, pred, proba, fit_s, pred_s, "class-prior baseline")


def run_lucid_transformers(X_train, y_train, X_test, y_test, device: str) -> list[dict]:
    rows = []
    print("    TabTransformer (lucidrains) ...", flush=True)
    rows.append(run_tab_transformer(X_train, y_train, X_test, y_test, device=device))
    print("    FTTransformer (lucidrains) ...", flush=True)
    rows.append(run_ft_transformer(X_train, y_train, X_test, y_test, device=device))
    return rows


def run_suite(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    include_tabpfn: bool,
    version: str,
    device: str,
    only_tab_transformer: bool = False,
) -> list[dict]:
    if only_tab_transformer:
        return run_lucid_transformers(X_train, y_train, X_test, y_test, device)
    rows: list[dict] = []
    if include_tabpfn:
        print("    TabPFN ...", flush=True)
        rows.append(run_tabpfn(X_train, y_train, X_test, y_test, version, device))
    print("    Dummy ...", flush=True)
    rows.append(run_dummy(X_train, y_train, X_test, y_test))
    print("    LogisticRegression ...")
    rows.append(run_logreg(X_train, y_train, X_test, y_test))
    print("    RandomForest ...")
    rf = fit_random_forest(X_train, y_train, X_test, y_test)
    rf["model"] = "RandomForest"
    rf["notes"] = "n_estimators=300, ordinal cats, class_weight=balanced"
    rows.append(rf)
    print("    HistGradientBoosting ...")
    rows.append(run_hgb(X_train, y_train, X_test, y_test))
    print("    LightGBM ...")
    rows.append(run_lgbm(X_train, y_train, X_test, y_test))
    print("    XGBoost defaults ...")
    rows.append(
        run_xgb(
            X_train,
            y_train,
            X_test,
            y_test,
            rounds=100,
            name="XGBoost (sensible defaults)",
            notes="n_estimators=100, ordinal cats (native cats crash on this wheel)",
        )
    )
    print("    XGBoost CV ...")
    rows.append(run_xgb_cv(X_train, y_train, X_test, y_test))
    rows.extend(run_lucid_transformers(X_train, y_train, X_test, y_test, device))
    return rows


def slim(row: dict) -> dict:
    keep = [
        "model",
        "roc_auc",
        "avg_precision",
        "accuracy",
        "f1",
        "fit_seconds",
        "predict_seconds",
        "notes",
        "best_rounds",
        "cv_seconds",
    ]
    return {k: row[k] for k in keep if k in row}


def run_dataset(
    task_id: str,
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    version: str,
    device: str,
    test_size: float,
    context_rows: int,
    include_thinking: bool = False,
    thinking_effort: str = "high",
    thinking_metric: str = "accuracy",
    thinking_timeout_s: float | None = None,
    only_tab_transformer: bool = False,
    protocols: tuple[str, ...] = ("context", "full"),
) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_ctx, y_ctx = maybe_subsample_train(X_train, y_train, context_rows)

    context_rows_out: list[dict] = []
    if "context" in protocols:
        print(f"\n=== {name} | context n={len(X_ctx)} ===")
        context_rows_out = run_suite(
            X_ctx,
            y_ctx,
            X_test,
            y_test,
            include_tabpfn=not only_tab_transformer,
            version=version,
            device=device,
            only_tab_transformer=only_tab_transformer,
        )
        print(pd.DataFrame([slim(r) for r in context_rows_out]).to_string(index=False))

    full_rows: list[dict] = []
    if "full" in protocols:
        print(f"\n=== {name} | full train n={len(X_train)} ===")
        full_rows = run_suite(
            X_train,
            y_train,
            X_test,
            y_test,
            include_tabpfn=False,
            version=version,
            device=device,
            only_tab_transformer=only_tab_transformer,
        )
        print(pd.DataFrame([slim(r) for r in full_rows]).to_string(index=False))

    if include_thinking and version != THINKING_VERSION and "full" in protocols:
        print(f"\n=== {name} | TabPFN thinking on full train n={len(X_train)} ===")
        thinking_row = run_tabpfn(
            X_train,
            y_train,
            X_test,
            y_test,
            THINKING_VERSION,
            device,
            thinking_effort=thinking_effort,
            thinking_metric=thinking_metric,
            thinking_timeout_s=thinking_timeout_s,
        )
        full_rows.append(thinking_row)
        print(pd.DataFrame([slim(thinking_row)]).to_string(index=False))

    return {
        "task_id": task_id,
        "name": name,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "n_test": int(len(X_test)),
        "n_train_full": int(len(X_train)),
        "n_train_context": int(len(X_ctx)),
        "context": [slim(r) for r in context_rows_out],
        "full": [slim(r) for r in full_rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v2", "v3", THINKING_VERSION], default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument(
        "--include-thinking",
        action="store_true",
        help="Add TabPFN-3-Plus thinking mode (tabpfn-client API) on the full train split.",
    )
    parser.add_argument("--thinking-effort", choices=["medium", "high"], default="high")
    parser.add_argument(
        "--thinking-metric",
        choices=["accuracy", "log_loss", "roc_auc"],
        default="accuracy",
    )
    parser.add_argument("--thinking-timeout-s", type=float, default=None)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["telco", "part1"],
        choices=["telco", "part1"],
    )
    parser.add_argument(
        "--only-tab-transformer",
        action="store_true",
        help=(
            "Train only lucidrains TabTransformer and FTTransformer, then merge "
            "those rows into results/benchmark.csv (keeps TabPFN thinking scores)."
        ),
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["context", "full"],
        choices=["context", "full"],
    )
    args = parser.parse_args()

    device = pick_device()
    version = pick_version(args.version)
    if args.max_train_rows is not None:
        context_rows = args.max_train_rows
    elif args.only_tab_transformer:
        # Match the TabPFN-2 context protocol already stored in benchmark.csv.
        context_rows = 1024
    else:
        context_rows = default_max_train_rows(version, device)
    print(
        f"device={device} version={version} context_rows={context_rows} "
        f"torch={torch.__version__} sklearn={sklearn.__version__} "
        f"xgb={xgb.__version__} lgb={lgb.__version__}"
    )

    payload: dict = {
        "device": device,
        "version": version,
        "context_rows": context_rows,
        "test_size": args.test_size,
        "seed": 42,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "lightgbm": lgb.__version__,
        "protocol": "https://docs.priorlabs.ai/benchmarking",
        "datasets": [],
    }

    loaders = {
        "telco": ("Telco customer churn", lambda: load_telco(TELCO_PATH)),
        "part1": ("Delivery hyper_ack", lambda: load_part1(PART1_PATH)),
    }
    for task in args.tasks:
        name, loader = loaders[task]
        X, y = loader()
        payload["datasets"].append(
            run_dataset(
                task,
                name,
                X,
                y,
                version,
                device,
                args.test_size,
                context_rows,
                include_thinking=args.include_thinking,
                thinking_effort=args.thinking_effort,
                thinking_metric=args.thinking_metric,
                thinking_timeout_s=args.thinking_timeout_s,
                only_tab_transformer=args.only_tab_transformer,
                protocols=tuple(args.protocols),
            )
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for ds in payload["datasets"]:
        for protocol, key in [("context", "context"), ("full", "full")]:
            train_n = ds["n_train_context"] if protocol == "context" else ds["n_train_full"]
            for row in ds[key]:
                flat.append(
                    {
                        "dataset": ds["name"],
                        "protocol": protocol,
                        "train_n": train_n,
                        "test_n": ds["n_test"],
                        **row,
                    }
                )
    new_df = pd.DataFrame(flat)
    if args.only_tab_transformer and OUT_CSV.exists() and not new_df.empty:
        old = pd.read_csv(OUT_CSV)
        keys = set(zip(new_df["dataset"], new_df["protocol"], new_df["model"]))
        keep = [
            (row["dataset"], row["protocol"], row["model"]) not in keys
            for _, row in old.iterrows()
        ]
        new_df = pd.concat([old.loc[keep], new_df], ignore_index=True)
        payload["merged_into_existing_csv"] = True
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    new_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
