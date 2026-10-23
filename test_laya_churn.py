#!/usr/bin/env python3

"""
Research-grade evaluation of Laya-MLX for Telco churn classification.

Evaluates four formulations:

1. CHOICE normal:
       churn, stay

2. CHOICE reversed:
       stay, churn

3. NOUL churn:
       "This customer will churn."

4. NOUL stay:
       "This customer will remain subscribed."
       converted into P(churn) = 1 - P(stay)

Metrics
-------
Classification:
    Accuracy
    Balanced Accuracy
    Precision
    Recall / Sensitivity
    Specificity
    NPV
    F1
    F0.5
    F2
    MCC
    Cohen Kappa
    FPR
    FNR

Ranking:
    ROC-AUC
    Average Precision
    Trapezoidal PR-AUC

Probabilistic:
    Log Loss / NLL
    Brier Score
    Brier Skill Score

Calibration:
    Event ECE, 15 bins
    Event MCE
    Decision ECE, 15 bins

Selective prediction:
    Accuracy @ 25% coverage
    Accuracy @ 50% coverage
    Accuracy @ 75% coverage
    Accuracy @ 100% coverage

Robustness:
    Choice option-order flip rate
    Choice probability drift
    NOUL complement consistency

Uncertainty:
    Bootstrap 95% confidence intervals

Performance:
    P50 request latency
    P95 request latency
    Mean latency
    Rows/sec
    P50 cost per decision
    P95 cost per decision

Install:
    pip install -U laya-mlx pandas numpy scikit-learn tqdm

Example:

    python test_laya_churn.py \
      "/path/to/telco_churn_head_499.csv"

Typed-decisions checkpoint:

    python test_laya_churn.py \
      "/path/to/telco_churn_head_499.csv" \
      --model aac6fef/laya-typed-decisions-mlx
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm

import laya_mlx as laya

from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_MODEL = "aac6fef/laya-mlx"

TARGET = "Churn"

DEFAULT_THRESHOLD = 0.50

# Laya's own evaluation harness uses 15 bins.
ECE_BINS = 15

BOOTSTRAP_SEED = 42

QUESTIONS_PER_ROW = 4


EXPECTED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


# ============================================================
# DATA PREPARATION
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return "missing"

    value = str(value).strip()

    if value == "":
        return "missing"

    return value


def to_float_or_missing(value):

    if pd.isna(value):
        return "missing"

    value = str(value).strip()

    if value == "":
        return "missing"

    try:
        return float(value)

    except ValueError:
        return value


def semantic_state(row):

    senior = row["SeniorCitizen"]

    if pd.isna(senior):
        senior_value = "missing"
    else:
        senior_value = (
            "Yes"
            if int(senior) == 1
            else "No"
        )

    return {

        "customer_gender":
            clean_text(row["gender"]),

        "is_senior_citizen":
            senior_value,

        "has_partner":
            clean_text(row["Partner"]),

        "has_dependents":
            clean_text(row["Dependents"]),

        "tenure_months":
            (
                int(row["tenure"])
                if not pd.isna(row["tenure"])
                else "missing"
            ),

        "has_phone_service":
            clean_text(row["PhoneService"]),

        "multiple_phone_lines":
            clean_text(row["MultipleLines"]),

        "internet_service_type":
            clean_text(row["InternetService"]),

        "online_security":
            clean_text(row["OnlineSecurity"]),

        "online_backup":
            clean_text(row["OnlineBackup"]),

        "device_protection":
            clean_text(row["DeviceProtection"]),

        "technical_support":
            clean_text(row["TechSupport"]),

        "streaming_tv":
            clean_text(row["StreamingTV"]),

        "streaming_movies":
            clean_text(row["StreamingMovies"]),

        "contract_type":
            clean_text(row["Contract"]),

        "paperless_billing":
            clean_text(row["PaperlessBilling"]),

        "payment_method":
            clean_text(row["PaymentMethod"]),

        "monthly_charges":
            (
                float(row["MonthlyCharges"])
                if not pd.isna(row["MonthlyCharges"])
                else "missing"
            ),

        "total_charges":
            to_float_or_missing(
                row["TotalCharges"]
            ),
    }


# ============================================================
# QUESTIONS
# ============================================================

QUESTIONS = {

    # --------------------------------------------------------
    # Normal option order
    # --------------------------------------------------------

    "choice_normal": {

        "type": "choice",

        "instructions": (
            "Based only on the customer's account, contract, billing, "
            "and service attributes, predict the customer's future "
            "subscription outcome."
        ),

        "criteria": {

            "churn": (
                "The customer will cancel or discontinue "
                "the telecommunications service."
            ),

            "stay": (
                "The customer will remain subscribed "
                "to the telecommunications service."
            ),
        },
    },

    # --------------------------------------------------------
    # Reversed option order
    # --------------------------------------------------------

    "choice_reversed": {

        "type": "choice",

        "instructions": (
            "Based only on the customer's account, contract, billing, "
            "and service attributes, predict the customer's future "
            "subscription outcome."
        ),

        "criteria": {

            "stay": (
                "The customer will remain subscribed "
                "to the telecommunications service."
            ),

            "churn": (
                "The customer will cancel or discontinue "
                "the telecommunications service."
            ),
        },
    },

    # --------------------------------------------------------
    # NOUL churn proposition
    # --------------------------------------------------------

    "noul_churn": {

        "type": "noul",

        "instructions": (
            "This customer will churn, meaning the customer "
            "will cancel or discontinue the telecommunications service."
        ),

        "criteria": {

            "false": (
                "The customer will remain subscribed."
            ),

            "true": (
                "The customer will cancel or discontinue the service."
            ),
        },
    },

    # --------------------------------------------------------
    # Opposite NOUL proposition
    # --------------------------------------------------------

    "noul_stay": {

        "type": "noul",

        "instructions": (
            "This customer will remain subscribed to the "
            "telecommunications service."
        ),

        "criteria": {

            "false": (
                "The customer will cancel or discontinue the service."
            ),

            "true": (
                "The customer will remain subscribed."
            ),
        },
    },
}


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(df):

    missing = sorted(
        EXPECTED_COLUMNS
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    target_values = set(
        df[TARGET]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    unexpected = (
        target_values
        - {"Yes", "No"}
    )

    if unexpected:

        raise ValueError(
            f"Unexpected target values: "
            f"{sorted(unexpected)}"
        )


# ============================================================
# CALIBRATION
# ============================================================

def calibration_stats(
    probability,
    outcomes,
    n_bins=15,
):

    probability = np.asarray(
        probability,
        dtype=float,
    )

    outcomes = np.asarray(
        outcomes,
        dtype=float,
    )

    probability = np.clip(
        probability,
        0.0,
        1.0,
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    mce = 0.0

    rows = []

    n = len(probability)

    for index in range(n_bins):

        lower = edges[index]
        upper = edges[index + 1]

        if index == n_bins - 1:

            mask = (
                (probability >= lower)
                & (probability <= upper)
            )

        else:

            mask = (
                (probability >= lower)
                & (probability < upper)
            )

        count = int(
            np.sum(mask)
        )

        if count == 0:
            continue

        mean_probability = float(
            np.mean(
                probability[mask]
            )
        )

        observed_rate = float(
            np.mean(
                outcomes[mask]
            )
        )

        gap = abs(
            mean_probability
            - observed_rate
        )

        weight = (
            count / n
        )

        ece += (
            weight * gap
        )

        mce = max(
            mce,
            gap,
        )

        rows.append(
            {
                "bin": index,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_probability":
                    mean_probability,
                "observed_rate":
                    observed_rate,
                "absolute_gap":
                    float(gap),
            }
        )

    return (
        float(ece),
        float(mce),
        rows,
    )


# ============================================================
# SELECTIVE ACCURACY
# ============================================================

def selective_accuracy(
    y_true,
    prediction,
    probability,
    coverage,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    prediction = np.asarray(
        prediction,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    # Probability assigned to selected decision.
    answer_confidence = np.maximum(
        probability,
        1.0 - probability,
    )

    count = max(
        1,
        int(
            np.ceil(
                len(y_true)
                * coverage
            )
        ),
    )

    order = np.argsort(
        -answer_confidence
    )

    selected = (
        order[:count]
    )

    return float(
        accuracy_score(
            y_true[selected],
            prediction[selected],
        )
    )


# ============================================================
# METRIC HELPERS
# ============================================================

def safe_roc_auc(
    y_true,
    probability,
):

    if len(
        np.unique(y_true)
    ) < 2:

        return float("nan")

    return float(
        roc_auc_score(
            y_true,
            probability,
        )
    )


def safe_average_precision(
    y_true,
    probability,
):

    if np.sum(y_true) == 0:
        return float("nan")

    return float(
        average_precision_score(
            y_true,
            probability,
        )
    )


def trapezoidal_pr_auc(
    y_true,
    probability,
):

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            probability,
        )
    )

    return float(
        auc(
            recall,
            precision,
        )
    )


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_confidence_intervals(
    y_true,
    probability,
    threshold=0.5,
    n_bootstrap=1000,
    seed=42,
):

    if n_bootstrap <= 0:
        return {}

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    positive_indices = np.where(
        y_true == 1
    )[0]

    negative_indices = np.where(
        y_true == 0
    )[0]

    if (
        len(positive_indices) == 0
        or len(negative_indices) == 0
    ):
        return {}

    rng = np.random.default_rng(
        seed
    )

    values = {

        "accuracy": [],
        "f1": [],
        "roc_auc": [],
        "average_precision": [],
        "brier_score": [],
    }

    for _ in range(n_bootstrap):

        sampled_positive = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True,
        )

        sampled_negative = rng.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True,
        )

        indices = np.concatenate(
            [
                sampled_positive,
                sampled_negative,
            ]
        )

        rng.shuffle(
            indices
        )

        yt = y_true[indices]
        pp = probability[indices]

        yp = (
            pp >= threshold
        ).astype(int)

        values["accuracy"].append(
            accuracy_score(
                yt,
                yp,
            )
        )

        values["f1"].append(
            f1_score(
                yt,
                yp,
                zero_division=0,
            )
        )

        values["roc_auc"].append(
            roc_auc_score(
                yt,
                pp,
            )
        )

        values["average_precision"].append(
            average_precision_score(
                yt,
                pp,
            )
        )

        values["brier_score"].append(
            brier_score_loss(
                yt,
                pp,
            )
        )

    result = {}

    for metric, samples in values.items():

        samples = np.asarray(
            samples,
            dtype=float,
        )

        result[metric] = {

            "lower_95":
                float(
                    np.percentile(
                        samples,
                        2.5,
                    )
                ),

            "upper_95":
                float(
                    np.percentile(
                        samples,
                        97.5,
                    )
                ),
        }

    return result


# ============================================================
# COMPLETE METRIC SET
# ============================================================

def calculate_metrics(
    y_true,
    probability,
    threshold=0.5,
    bootstrap_iterations=1000,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    prediction = (
        probability
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1],
        )
        .ravel()
    )

    precision = (
        precision_score(
            y_true,
            prediction,
            zero_division=0,
        )
    )

    recall = (
        recall_score(
            y_true,
            prediction,
            zero_division=0,
        )
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else float("nan")
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else float("nan")
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else float("nan")
    )

    roc_auc = (
        safe_roc_auc(
            y_true,
            probability,
        )
    )

    average_precision = (
        safe_average_precision(
            y_true,
            probability,
        )
    )

    pr_auc = (
        trapezoidal_pr_auc(
            y_true,
            probability,
        )
    )

    safe_probability = np.clip(
        probability,
        1e-7,
        1.0 - 1e-7,
    )

    nll = log_loss(
        y_true,
        safe_probability,
        labels=[0, 1],
    )

    brier = (
        brier_score_loss(
            y_true,
            probability,
        )
    )

    prevalence = float(
        np.mean(y_true)
    )

    # Baseline: always predict observed prevalence.
    baseline_probability = np.full(
        len(y_true),
        prevalence,
    )

    baseline_brier = (
        brier_score_loss(
            y_true,
            baseline_probability,
        )
    )

    if baseline_brier > 0:

        brier_skill = (
            1.0
            - (
                brier
                / baseline_brier
            )
        )

    else:

        brier_skill = float("nan")

    # --------------------------------------------------------
    # EVENT CALIBRATION
    #
    # Is P(churn) calibrated to actual churn frequency?
    # --------------------------------------------------------

    (
        event_ece,
        event_mce,
        calibration_bins,
    ) = calibration_stats(

        probability,
        y_true,
        n_bins=ECE_BINS,
    )

    # --------------------------------------------------------
    # DECISION CALIBRATION
    #
    # Closest analogue to Laya's generic answer-confidence ECE.
    #
    # Confidence = probability assigned to selected answer.
    # Outcome = whether selected answer was correct.
    # --------------------------------------------------------

    answer_confidence = np.maximum(
        probability,
        1.0 - probability,
    )

    decision_correct = (
        prediction == y_true
    ).astype(int)

    (
        decision_ece,
        _,
        decision_calibration_bins,
    ) = calibration_stats(

        answer_confidence,
        decision_correct,
        n_bins=ECE_BINS,
    )

    # --------------------------------------------------------
    # Selective prediction
    # --------------------------------------------------------

    selective = {

        "accuracy_at_25pct_coverage":
            selective_accuracy(
                y_true,
                prediction,
                probability,
                0.25,
            ),

        "accuracy_at_50pct_coverage":
            selective_accuracy(
                y_true,
                prediction,
                probability,
                0.50,
            ),

        "accuracy_at_75pct_coverage":
            selective_accuracy(
                y_true,
                prediction,
                probability,
                0.75,
            ),

        "accuracy_at_100pct_coverage":
            selective_accuracy(
                y_true,
                prediction,
                probability,
                1.00,
            ),
    }

    bootstrap = (
        bootstrap_confidence_intervals(
            y_true,
            probability,
            threshold=threshold,
            n_bootstrap=bootstrap_iterations,
            seed=BOOTSTRAP_SEED,
        )
    )

    metrics = {

        "samples":
            int(len(y_true)),

        "positives":
            int(np.sum(y_true)),

        "negatives":
            int(
                len(y_true)
                - np.sum(y_true)
            ),

        "prevalence":
            prevalence,

        "threshold":
            float(threshold),

        # Classification

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    prediction,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    prediction,
                )
            ),

        "precision":
            float(precision),

        "recall_sensitivity":
            float(recall),

        "specificity":
            float(specificity),

        "negative_predictive_value":
            float(npv),

        "f1":
            float(
                f1_score(
                    y_true,
                    prediction,
                    zero_division=0,
                )
            ),

        "f0_5":
            float(
                fbeta_score(
                    y_true,
                    prediction,
                    beta=0.5,
                    zero_division=0,
                )
            ),

        "f2":
            float(
                fbeta_score(
                    y_true,
                    prediction,
                    beta=2.0,
                    zero_division=0,
                )
            ),

        "mcc":
            float(
                matthews_corrcoef(
                    y_true,
                    prediction,
                )
            ),

        "cohen_kappa":
            float(
                cohen_kappa_score(
                    y_true,
                    prediction,
                )
            ),

        "false_positive_rate":
            float(fpr),

        "false_negative_rate":
            float(fnr),

        # Ranking

        "roc_auc":
            float(roc_auc),

        "average_precision":
            float(
                average_precision
            ),

        "pr_auc_trapezoidal":
            float(pr_auc),

        # Probability

        "log_loss_nll":
            float(nll),

        "brier_score":
            float(brier),

        "baseline_brier":
            float(
                baseline_brier
            ),

        "brier_skill_score":
            float(
                brier_skill
            ),

        # Calibration

        "event_ece_15":
            float(event_ece),

        "event_mce_15":
            float(event_mce),

        "decision_ece_15":
            float(
                decision_ece
            ),

        # Distribution

        "mean_p_churn":
            float(
                np.mean(
                    probability
                )
            ),

        "median_p_churn":
            float(
                np.median(
                    probability
                )
            ),

        "std_p_churn":
            float(
                np.std(
                    probability
                )
            ),

        "min_p_churn":
            float(
                np.min(
                    probability
                )
            ),

        "max_p_churn":
            float(
                np.max(
                    probability
                )
            ),

        # Confusion matrix

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),

        # Selective prediction

        "selective_accuracy":
            selective,

        # Bootstrap

        "bootstrap_95_ci":
            bootstrap,

        # Calibration bins

        "event_calibration_bins":
            calibration_bins,

        "decision_calibration_bins":
            decision_calibration_bins,
    }

    return (
        metrics,
        prediction,
    )


# ============================================================
# PRINT
# ============================================================

def print_metrics(
    name,
    y_true,
    probability,
    threshold,
    bootstrap_iterations,
):

    metrics, prediction = (
        calculate_metrics(
            y_true,
            probability,
            threshold=threshold,
            bootstrap_iterations=
                bootstrap_iterations,
        )
    )

    print("\n")
    print("=" * 88)
    print(name)
    print("=" * 88)

    print(
        f"Samples:                     "
        f"{metrics['samples']}"
    )

    print(
        f"Churn prevalence:            "
        f"{metrics['prevalence']:.4f}"
    )

    print(
        f"Decision threshold:          "
        f"{metrics['threshold']:.4f}"
    )

    print("\n--- Classification ---")

    print(
        f"Accuracy:                    "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy:           "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision / PPV:             "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall / Sensitivity:        "
        f"{metrics['recall_sensitivity']:.4f}"
    )

    print(
        f"Specificity:                 "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"NPV:                         "
        f"{metrics['negative_predictive_value']:.4f}"
    )

    print(
        f"F1:                          "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"F0.5:                        "
        f"{metrics['f0_5']:.4f}"
    )

    print(
        f"F2:                          "
        f"{metrics['f2']:.4f}"
    )

    print(
        f"MCC:                         "
        f"{metrics['mcc']:.4f}"
    )

    print(
        f"Cohen Kappa:                 "
        f"{metrics['cohen_kappa']:.4f}"
    )

    print("\n--- Ranking ---")

    print(
        f"ROC-AUC:                     "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Average Precision:           "
        f"{metrics['average_precision']:.4f}"
    )

    print(
        f"PR-AUC trapezoidal:          "
        f"{metrics['pr_auc_trapezoidal']:.4f}"
    )

    print("\n--- Probability quality ---")

    print(
        f"Log Loss / NLL:              "
        f"{metrics['log_loss_nll']:.4f}"
    )

    print(
        f"Brier Score:                 "
        f"{metrics['brier_score']:.4f}"
    )

    print(
        f"Baseline Brier:              "
        f"{metrics['baseline_brier']:.4f}"
    )

    print(
        f"Brier Skill Score:           "
        f"{metrics['brier_skill_score']:.4f}"
    )

    print("\n--- Calibration ---")

    print(
        f"Event ECE (15 bins):         "
        f"{metrics['event_ece_15']:.4f}"
    )

    print(
        f"Event MCE (15 bins):         "
        f"{metrics['event_mce_15']:.4f}"
    )

    print(
        f"Decision ECE (15 bins):      "
        f"{metrics['decision_ece_15']:.4f}"
    )

    print("\n--- Selective accuracy ---")

    for key, value in (
        metrics[
            "selective_accuracy"
        ].items()
    ):

        print(
            f"{key:28s} "
            f"{value:.4f}"
        )

    print("\n--- Confusion matrix ---")

    print(
        np.array(
            [
                [
                    metrics["TN"],
                    metrics["FP"],
                ],
                [
                    metrics["FN"],
                    metrics["TP"],
                ],
            ]
        )
    )

    print("\n--- Classification report ---")

    print(
        classification_report(
            y_true,
            prediction,
            target_names=[
                "stay",
                "churn",
            ],
            digits=4,
            zero_division=0,
        )
    )

    ci = (
        metrics[
            "bootstrap_95_ci"
        ]
    )

    if ci:

        print(
            "\n--- Bootstrap 95% CI ---"
        )

        for metric, interval in (
            ci.items()
        ):

            print(
                f"{metric:22s} "
                f"["
                f"{interval['lower_95']:.4f}, "
                f"{interval['upper_95']:.4f}"
                f"]"
            )

    return (
        metrics,
        prediction,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv_path",
        type=str,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=
            "laya_churn_predictions.csv",
    )

    parser.add_argument(
        "--metrics-output",
        type=str,
        default=
            "laya_churn_metrics.json",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    csv_path = (
        Path(args.csv_path)
        .expanduser()
        .resolve()
    )

    if not csv_path.exists():

        raise FileNotFoundError(
            csv_path
        )

    print(
        f"Loading CSV: "
        f"{csv_path}"
    )

    df = pd.read_csv(
        csv_path
    )

    validate_dataset(
        df
    )

    if args.limit is not None:

        df = (
            df
            .head(args.limit)
            .copy()
        )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: "
        f"{len(df.columns)}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        df[TARGET]
        .value_counts()
    )

    y_true = (

        df[TARGET]

        .astype(str)

        .str.strip()

        .map(
            {
                "No": 0,
                "Yes": 1,
            }
        )

        .to_numpy(
            dtype=int
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        f"\nLoading model: "
        f"{args.model}"
    )

    agent = laya.load(

        args.model,

        dtype="float16",

        device="gpu",

        batch_size=16,
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    normal_choice_label = []
    reversed_choice_label = []

    p_choice_normal = []
    p_choice_reversed = []

    p_noul_churn = []
    p_noul_stay = []

    laya_choice_entropy_confidence = []
    laya_noul_confidence = []

    input_tokens = []

    row_latencies = []

    first_result = None

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    if len(df):

        agent.predict(
            semantic_state(
                df.iloc[0]
            ),
            QUESTIONS,
        )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    for _, row in tqdm(

        df.iterrows(),

        total=len(df),

        desc="Laya inference",

        unit="row",
    ):

        state = (
            semantic_state(row)
        )

        start = (
            time.perf_counter()
        )

        result = agent.predict(
            state,
            QUESTIONS,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        row_latencies.append(
            elapsed
        )

        if first_result is None:

            first_result = result

        # ----------------------------------------------------
        # Normal choice
        # ----------------------------------------------------

        normal = (
            result[
                "answers"
            ][
                "choice_normal"
            ]
        )

        normal_choice_label.append(
            normal["choice"]
        )

        p_choice_normal.append(
            float(
                normal[
                    "probabilities"
                ][
                    "churn"
                ]
            )
        )

        laya_choice_entropy_confidence.append(
            float(
                normal[
                    "confidence"
                ]
            )
        )

        # ----------------------------------------------------
        # Reversed choice
        # ----------------------------------------------------

        reversed_answer = (
            result[
                "answers"
            ][
                "choice_reversed"
            ]
        )

        reversed_choice_label.append(
            reversed_answer[
                "choice"
            ]
        )

        p_choice_reversed.append(
            float(
                reversed_answer[
                    "probabilities"
                ][
                    "churn"
                ]
            )
        )

        # ----------------------------------------------------
        # NOUL churn
        # ----------------------------------------------------

        churn_noul = (
            result[
                "answers"
            ][
                "noul_churn"
            ]
        )

        p_noul_churn.append(
            float(
                churn_noul[
                    "noul"
                ]
            )
        )

        laya_noul_confidence.append(
            float(
                churn_noul[
                    "confidence"
                ]
            )
        )

        # ----------------------------------------------------
        # NOUL stay
        # ----------------------------------------------------

        stay_noul = (
            result[
                "answers"
            ][
                "noul_stay"
            ]
        )

        p_noul_stay.append(
            float(
                stay_noul[
                    "noul"
                ]
            )
        )

        input_tokens.append(
            int(
                result[
                    "usage"
                ][
                    "input_tokens"
                ]
            )
        )

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    p_choice_normal = np.asarray(
        p_choice_normal,
        dtype=float,
    )

    p_choice_reversed = np.asarray(
        p_choice_reversed,
        dtype=float,
    )

    p_noul_churn = np.asarray(
        p_noul_churn,
        dtype=float,
    )

    p_noul_stay = np.asarray(
        p_noul_stay,
        dtype=float,
    )

    # Convert stay proposition into an independent churn probability.
    p_noul_stay_complement = (
        1.0
        - p_noul_stay
    )

    # --------------------------------------------------------
    # Raw example
    # --------------------------------------------------------

    print(
        "\nExample raw result:"
    )

    print(
        json.dumps(
            first_result,
            indent=2,
            ensure_ascii=False,
        )
    )

    # ========================================================
    # MAIN METRICS
    # ========================================================

    (
        metrics_choice_normal,
        prediction_choice_normal,
    ) = print_metrics(

        "CHOICE NORMAL ORDER",

        y_true,

        p_choice_normal,

        args.threshold,

        args.bootstrap,
    )

    (
        metrics_choice_reversed,
        prediction_choice_reversed,
    ) = print_metrics(

        "CHOICE REVERSED ORDER",

        y_true,

        p_choice_reversed,

        args.threshold,

        args.bootstrap,
    )

    (
        metrics_noul_churn,
        prediction_noul_churn,
    ) = print_metrics(

        "NOUL: CUSTOMER WILL CHURN",

        y_true,

        p_noul_churn,

        args.threshold,

        args.bootstrap,
    )

    (
        metrics_noul_stay,
        prediction_noul_stay,
    ) = print_metrics(

        "NOUL COMPLEMENT: 1 - P(STAY)",

        y_true,

        p_noul_stay_complement,

        args.threshold,

        args.bootstrap,
    )

    # ========================================================
    # OPTION ORDER ROBUSTNESS
    # ========================================================

    normal_labels = np.asarray(
        normal_choice_label
    )

    reversed_labels = np.asarray(
        reversed_choice_label
    )

    order_flip_rate = float(
        np.mean(
            normal_labels
            != reversed_labels
        )
    )

    order_probability_mae = float(
        np.mean(
            np.abs(
                p_choice_normal
                - p_choice_reversed
            )
        )
    )

    order_probability_max = float(
        np.max(
            np.abs(
                p_choice_normal
                - p_choice_reversed
            )
        )
    )

    # ========================================================
    # NOUL COMPLEMENT CONSISTENCY
    # ========================================================

    direct_vs_complement_mae = float(
        np.mean(
            np.abs(
                p_noul_churn
                - p_noul_stay_complement
            )
        )
    )

    direct_vs_complement_max = float(
        np.max(
            np.abs(
                p_noul_churn
                - p_noul_stay_complement
            )
        )
    )

    # Both propositions considered true.
    contradiction_rate = float(
        np.mean(
            (
                p_noul_churn >= 0.5
            )
            &
            (
                p_noul_stay >= 0.5
            )
        )
    )

    # Both propositions considered false.
    neither_rate = float(
        np.mean(
            (
                p_noul_churn < 0.5
            )
            &
            (
                p_noul_stay < 0.5
            )
        )
    )

    print("\n")
    print("=" * 88)
    print("JEV/LAYA ROBUSTNESS DIAGNOSTICS")
    print("=" * 88)

    print(
        f"Choice option-order flip rate:       "
        f"{order_flip_rate:.4f}"
    )

    print(
        f"Choice P(churn) mean abs drift:      "
        f"{order_probability_mae:.4f}"
    )

    print(
        f"Choice P(churn) max drift:           "
        f"{order_probability_max:.4f}"
    )

    print(
        f"NOUL direct/complement mean drift:   "
        f"{direct_vs_complement_mae:.4f}"
    )

    print(
        f"NOUL direct/complement max drift:    "
        f"{direct_vs_complement_max:.4f}"
    )

    print(
        f"NOUL contradiction rate:             "
        f"{contradiction_rate:.4f}"
    )

    print(
        f"NOUL neither-true rate:              "
        f"{neither_rate:.4f}"
    )

    # ========================================================
    # LATENCY
    # ========================================================

    latency_ms = (
        np.asarray(
            row_latencies
        )
        * 1000
    )

    total_seconds = float(
        np.sum(
            row_latencies
        )
    )

    latency = {

        "mean_request_ms":
            float(
                np.mean(
                    latency_ms
                )
            ),

        "p50_request_ms":
            float(
                np.percentile(
                    latency_ms,
                    50,
                )
            ),

        "p95_request_ms":
            float(
                np.percentile(
                    latency_ms,
                    95,
                )
            ),

        "rows_per_second":
            float(
                len(df)
                / total_seconds
            ),

        "p50_cost_per_decision_ms":
            float(
                np.percentile(
                    latency_ms
                    / QUESTIONS_PER_ROW,
                    50,
                )
            ),

        "p95_cost_per_decision_ms":
            float(
                np.percentile(
                    latency_ms
                    / QUESTIONS_PER_ROW,
                    95,
                )
            ),

        "mean_input_tokens_per_row":
            float(
                np.mean(
                    input_tokens
                )
            ),
    }

    print("\n")
    print("=" * 88)
    print("LATENCY")
    print("=" * 88)

    for key, value in latency.items():

        print(
            f"{key:32s} "
            f"{value:.4f}"
        )

    # ========================================================
    # SAVE ROW RESULTS
    # ========================================================

    output = df.copy()

    output["y_true"] = y_true

    output[
        "choice_normal_label"
    ] = normal_choice_label

    output[
        "choice_normal_p_churn"
    ] = p_choice_normal

    output[
        "choice_normal_prediction"
    ] = prediction_choice_normal

    output[
        "choice_reversed_label"
    ] = reversed_choice_label

    output[
        "choice_reversed_p_churn"
    ] = p_choice_reversed

    output[
        "choice_reversed_prediction"
    ] = prediction_choice_reversed

    output[
        "noul_churn_p"
    ] = p_noul_churn

    output[
        "noul_churn_prediction"
    ] = prediction_noul_churn

    output[
        "noul_stay_p"
    ] = p_noul_stay

    output[
        "noul_stay_complement_p_churn"
    ] = p_noul_stay_complement

    output[
        "noul_stay_complement_prediction"
    ] = prediction_noul_stay

    output[
        "choice_order_probability_delta"
    ] = np.abs(
        p_choice_normal
        - p_choice_reversed
    )

    output[
        "choice_order_flipped"
    ] = (
        normal_labels
        != reversed_labels
    )

    output[
        "noul_complement_delta"
    ] = np.abs(
        p_noul_churn
        - p_noul_stay_complement
    )

    output[
        "laya_choice_entropy_confidence"
    ] = laya_choice_entropy_confidence

    output[
        "laya_noul_confidence"
    ] = laya_noul_confidence

    output[
        "request_latency_ms"
    ] = latency_ms

    output_path = (

        Path(args.output)

        .expanduser()

        .resolve()
    )

    output.to_csv(
        output_path,
        index=False,
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {

        "dataset": {

            "path":
                str(csv_path),

            "rows":
                int(len(df)),

            "churn_count":
                int(
                    np.sum(y_true)
                ),

            "stay_count":
                int(
                    len(y_true)
                    - np.sum(y_true)
                ),

            "prevalence":
                float(
                    np.mean(
                        y_true
                    )
                ),
        },

        "model": {

            "model_id":
                args.model,

            "threshold":
                float(
                    args.threshold
                ),

            "ece_bins":
                ECE_BINS,

            "bootstrap_iterations":
                int(
                    args.bootstrap
                ),
        },

        "choice_normal":
            metrics_choice_normal,

        "choice_reversed":
            metrics_choice_reversed,

        "noul_churn":
            metrics_noul_churn,

        "noul_stay_complement":
            metrics_noul_stay,

        "robustness": {

            "choice_option_order_flip_rate":
                order_flip_rate,

            "choice_probability_mean_abs_drift":
                order_probability_mae,

            "choice_probability_max_drift":
                order_probability_max,

            "noul_complement_mean_abs_drift":
                direct_vs_complement_mae,

            "noul_complement_max_drift":
                direct_vs_complement_max,

            "noul_contradiction_rate":
                contradiction_rate,

            "noul_neither_true_rate":
                neither_rate,
        },

        "latency":
            latency,
    }

    metrics_path = (

        Path(
            args.metrics_output
        )

        .expanduser()

        .resolve()
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n")
    print("=" * 88)
    print("FILES")
    print("=" * 88)

    print(
        f"Predictions:\n"
        f"{output_path}"
    )

    print(
        f"\nMetrics:\n"
        f"{metrics_path}"
    )


if __name__ == "__main__":

    main()