"""Binary classification with TabPFN on Telco Churn and delivery hyper_ack.

Follows the Prior Labs local demo pattern:
  https://github.com/PriorLabs/TabPFN
  https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb

TabPFN is a tabular foundation model. Do not scale or one-hot encode features.
Pass mixed numeric/categorical pandas DataFrames; missing values are allowed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

ROOT = Path(__file__).resolve().parent
TELCO_PATH = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
_PART1_CANDIDATES = (
    ROOT / "hyper_ackt-dataset.csv",
    ROOT / "part1-dataset.csv",
)
PART1_PATH = next((p for p in _PART1_CANDIDATES if p.exists()), _PART1_CANDIDATES[0])
RESULTS_PATH = ROOT / "results" / "metrics.json"

THINKING_VERSION = "thinking"


def _token_from_env_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("TABPFN_TOKEN="):
            return stripped.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def load_tabpfn_token() -> bool:
    """Load TABPFN_TOKEN from .env into the process. Never prints the secret."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    token = (os.environ.get("TABPFN_TOKEN") or "").strip() or _token_from_env_file(ROOT / ".env")
    if not token:
        return False
    os.environ["TABPFN_TOKEN"] = token
    try:
        import tabpfn_client

        tabpfn_client.set_access_token(token)
    except Exception:
        pass
    return True


load_tabpfn_token()


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_version(requested: str | None = None) -> str:
    if requested:
        return requested
    # TabPFN-3 (Colab default) needs TABPFN_TOKEN from https://ux.priorlabs.ai/account
    if os.environ.get("TABPFN_TOKEN"):
        return "v3"
    return "v2"


def make_tabpfn_classifier(
    version: str,
    device: str,
    n_estimators: int | str | None,
    thinking_effort: str = "high",
    thinking_metric: str = "accuracy",
    thinking_timeout_s: float | None = None,
):
    if version == THINKING_VERSION:
        from tabpfn_client import TabPFNClassifier as ClientTabPFNClassifier

        # Hosted TabPFN-3-Plus. Thinking is API-only; local `tabpfn` cannot do this.
        return ClientTabPFNClassifier(
            thinking_mode=True,
            thinking_effort=thinking_effort,
            thinking_metric=thinking_metric,
            **({"thinking_timeout_s": thinking_timeout_s} if thinking_timeout_s else {}),
        )
    kwargs: dict = {
        "device": device,
        "ignore_pretraining_limits": True,
        "fit_mode": "fit_with_cache",
    }
    if n_estimators is not None:
        kwargs["n_estimators"] = n_estimators
    if version == "v3":
        return TabPFNClassifier(**kwargs)
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2, **kwargs)


def default_max_train_rows(version: str, device: str) -> int:
    if version == "v2":
        return 1024  # TabPFN-2 recommended context size
    if version == THINKING_VERSION:
        return 100_000  # API TabPFN-3-Plus; thinking uses the full train split
    if device == "cpu":
        return 5000  # TabPFN-3 CPU guardrail
    return 100_000


def load_telco(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")
    X["SeniorCitizen"] = X["SeniorCitizen"].astype("category")
    return X, y


def load_part1(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    # Unique timestamps act like IDs; weekday/time_bucket already encode time-of-day.
    df = df.drop(columns=["first_created_at"])
    df["created_date"] = pd.to_datetime(df["created_date"], format="mixed")
    df["created_dayofyear"] = df["created_date"].dt.dayofyear.astype(np.int16)
    df = df.drop(columns=["created_date"])
    df["total_distance"] = pd.to_numeric(df["total_distance"], errors="coerce")
    y = df["hyper_ack"].astype(int)
    X = df.drop(columns=["hyper_ack"])
    for col in ["deliverey_category_id", "weekday"]:
        X[col] = X[col].astype("category")
    return X, y


def metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


def maybe_subsample_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_train_rows: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X_train) <= max_train_rows:
        return X_train, y_train
    X_sub, _, y_sub, _ = train_test_split(
        X_train,
        y_train,
        train_size=max_train_rows,
        stratify=y_train,
        random_state=42,
    )
    print(f"  subsample train: {len(X_train)} -> {len(X_sub)} (cap={max_train_rows})")
    return X_sub, y_sub


def fit_tabpfn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    device: str,
    n_estimators: int | str | None,
    version: str,
    thinking_effort: str = "high",
    thinking_metric: str = "accuracy",
    thinking_timeout_s: float | None = None,
) -> dict:
    clf = make_tabpfn_classifier(
        version,
        device,
        n_estimators,
        thinking_effort=thinking_effort,
        thinking_metric=thinking_metric,
        thinking_timeout_s=thinking_timeout_s,
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0
    t1 = time.perf_counter()
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    t_pred = time.perf_counter() - t1
    out = metrics(y_test, pred, proba)
    out["fit_seconds"] = round(t_fit, 2)
    out["predict_seconds"] = round(t_pred, 2)
    return out


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    cat_cols = list(X_train.select_dtypes(include=["category", "object"]).columns)
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ]
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0
    t1 = time.perf_counter()
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    t_pred = time.perf_counter() - t1
    out = metrics(y_test, pred, proba)
    out["fit_seconds"] = round(t_fit, 2)
    out["predict_seconds"] = round(t_pred, 2)
    return out


def run_task(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    device: str,
    n_estimators: int | str | None,
    test_size: float,
    max_train_rows: int,
    version: str,
    thinking_effort: str = "high",
    thinking_metric: str = "accuracy",
    thinking_timeout_s: float | None = None,
) -> dict:
    print(f"\n=== {name} ===")
    print(f"rows={len(X)} features={X.shape[1]} positive_rate={float(y.mean()):.3f}")
    print(f"dtypes:\n{X.dtypes}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train_tab, y_train_tab = maybe_subsample_train(X_train, y_train, max_train_rows)
    print(
        f"tabpfn_train={len(X_train_tab)} rf_train={len(X_train)} "
        f"test={len(X_test)} device={device} version={version}"
    )

    print("Fitting TabPFNClassifier ...")
    tabpfn = fit_tabpfn(
        X_train_tab,
        y_train_tab,
        X_test,
        y_test,
        device,
        n_estimators,
        version,
        thinking_effort=thinking_effort,
        thinking_metric=thinking_metric,
        thinking_timeout_s=thinking_timeout_s,
    )
    print(
        f"  TabPFN  ROC-AUC={tabpfn['roc_auc']:.4f}  "
        f"acc={tabpfn['accuracy']:.4f}  f1={tabpfn['f1']:.4f}  "
        f"fit={tabpfn['fit_seconds']}s  predict={tabpfn['predict_seconds']}s"
    )
    print(tabpfn["report"])

    print("Fitting RandomForest on the same TabPFN context rows ...")
    rf_same = fit_random_forest(X_train_tab, y_train_tab, X_test, y_test)
    print(
        f"  RF-same ROC-AUC={rf_same['roc_auc']:.4f}  "
        f"acc={rf_same['accuracy']:.4f}  f1={rf_same['f1']:.4f}"
    )

    print("Fitting RandomForest on the full train split ...")
    rf_full = fit_random_forest(X_train, y_train, X_test, y_test)
    print(
        f"  RF-full ROC-AUC={rf_full['roc_auc']:.4f}  "
        f"acc={rf_full['accuracy']:.4f}  f1={rf_full['f1']:.4f}  "
        f"fit={rf_full['fit_seconds']}s"
    )
    print(rf_full["report"])

    return {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "n_train_tabpfn": int(len(X_train_tab)),
        "n_train_full": int(len(X_train)),
        "n_test": int(len(X_test)),
        "device": device,
        "version": version,
        "n_estimators": n_estimators,
        "thinking_effort": thinking_effort if version == THINKING_VERSION else None,
        "thinking_metric": thinking_metric if version == THINKING_VERSION else None,
        "tabpfn": tabpfn,
        "random_forest_same_context": rf_same,
        "random_forest_full_train": rf_full,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        choices=["v2", "v3", THINKING_VERSION],
        default=None,
        help="v3 needs TABPFN_TOKEN. thinking uses tabpfn-client (API TabPFN-3-Plus). "
        "Default: v3 if token is set, else v2.",
    )
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument(
        "--thinking-effort",
        choices=["medium", "high"],
        default="high",
    )
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
    args = parser.parse_args()

    device = pick_device()
    version = pick_version(args.version)
    n_estimators = args.n_estimators
    max_train_rows = args.max_train_rows or default_max_train_rows(version, device)
    print(
        f"torch={torch.__version__} device={device} version={version} "
        f"max_train_rows={max_train_rows}"
    )
    if version in {"v3", THINKING_VERSION} and not os.environ.get("TABPFN_TOKEN"):
        print(
            "TABPFN_TOKEN is not set. Get a key at "
            "https://ux.priorlabs.ai/account/api-keys"
        )

    thinking_kw = dict(
        thinking_effort=args.thinking_effort,
        thinking_metric=args.thinking_metric,
        thinking_timeout_s=args.thinking_timeout_s,
    )
    results: dict = {
        "device": device,
        "torch": torch.__version__,
        "version": version,
        "n_estimators": n_estimators,
        "max_train_rows": max_train_rows,
        **({k: v for k, v in thinking_kw.items()} if version == THINKING_VERSION else {}),
    }
    if "telco" in args.tasks:
        X, y = load_telco(TELCO_PATH)
        results["telco_churn"] = run_task(
            "Telco customer churn",
            X,
            y,
            device,
            n_estimators,
            args.test_size,
            max_train_rows,
            version,
            **thinking_kw,
        )
    if "part1" in args.tasks:
        X, y = load_part1(PART1_PATH)
        results["delivery_hyper_ack"] = run_task(
            "Delivery hyper_ack",
            X,
            y,
            device,
            n_estimators,
            args.test_size,
            max_train_rows,
            version,
            **thinking_kw,
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
