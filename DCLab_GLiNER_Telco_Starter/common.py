"""Shared data and metrics for the first GLiNER Telco churn pilot."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    classification_report, f1_score, log_loss, roc_auc_score,
)

FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [x for x in FEATURES if x not in NUMERIC]
TASK = "churn_status"
LABELS = ["not_churn", "churn"]


def read_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set([*FEATURES, "Churn", "customerID"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required IBM Telco fields: {sorted(missing)}")
    df = df.reset_index(drop=True)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    if not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Unexpected churn labels; expected Yes and No")
    return df


def serialize_customer(row: pd.Series) -> str:
    """Same exact feature order and wording at train and inference time."""
    lines = ["Telecom customer subscription profile at the recorded snapshot:"]
    for col in FEATURES:
        value = row[col]
        if pd.isna(value) or str(value).strip() == "":
            display = "unknown"
        elif col == "SeniorCitizen":
            display = "yes" if int(value) == 1 else "no"
        elif col in ("MonthlyCharges", "TotalCharges"):
            display = f"{float(value):.2f}"
        elif col == "tenure":
            display = str(int(value))
        else:
            display = str(value).strip()
        lines.append(f"{col}: {display}")
    return "\n".join(lines)


def format_example(text: str, label: str) -> dict:
    return {
        "input": text,
        "output": {
            "classifications": [
                {"task": TASK, "labels": LABELS, "true_label": [label]}
            ]
        },
    }


def metrics(y_true, prob_churn, labels=None) -> dict:
    y_true = list(map(int, y_true))
    out = {"n": len(y_true), "churn_rate": round(sum(y_true) / len(y_true), 5)}
    if labels is None:
        labels = [int(p >= 0.5) for p in prob_churn]
    out["accuracy"] = round(float(accuracy_score(y_true, labels)), 5)
    out["f1"] = round(float(f1_score(y_true, labels, zero_division=0)), 5)
    out["recall"] = round(sum(a == b == 1 for a, b in zip(y_true, labels)) / max(1, sum(y_true)), 5)
    if prob_churn is not None and len(set(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, prob_churn)), 5)
        out["average_precision"] = round(float(average_precision_score(y_true, prob_churn)), 5)
        out["log_loss"] = round(float(log_loss(y_true, prob_churn, labels=[0, 1])), 5)
        out["brier"] = round(float(brier_score_loss(y_true, prob_churn)), 5)
    return out


def dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
