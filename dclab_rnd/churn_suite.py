"""Reproducible Telco Churn development benchmark.

This is a fixed 15-experiment campaign. It deliberately reports adaptive
development CV, never an untouched test or deployment claim.
"""
import argparse
import json
from pathlib import Path
from .agentic.catalog import ROOT
from .agentic.worker import evaluate

RESULTS = ROOT / "churn_exp" / "results"

def feature(name, operation, inputs, rationale):
    return {"name": name, "operation": operation, "inputs": inputs, "rationale": rationale}

def plan(title, model, parameters=None, features=None, stress=None):
    return {"dataset": "telco_churn", "title": title,
            "hypothesis": f"{title} provides a discriminating, reproducible comparison on the same grouped development folds.",
            "model": model, "parameters": parameters or {}, "features": features or [],
            "drop_columns": [], "stress_columns": stress or [], "evidence_ids": [],
            "reference_evidence_id": None, "expected_learning": "Measure ranking, probability quality, robustness and the cost of this modeling choice."}

CHARGE_FEATURES = [
    feature("fe_log_tenure", "signed_log", ["tenure"], "Compress tenure while retaining monotonic customer-age information."),
    feature("fe_total_per_tenure", "ratio", ["TotalCharges", "tenure"], "Approximate realized monthly spend; tenure zero becomes missing and is imputed inside each fold."),
    feature("fe_monthly_x_tenure", "product", ["MonthlyCharges", "tenure"], "Compare billed total with a simple expected cumulative charge."),
]
RICH_FEATURES = CHARGE_FEATURES + [
    feature("fe_charge_gap", "difference", ["TotalCharges", "fe_monthly_x_tenure"], "Expose billing deviation without using the churn target."),
    feature("fe_log_total", "signed_log", ["TotalCharges"], "Reduce the long-tailed scale of cumulative charges."),
]

PLANS = [
    plan("Prior-only floor", "dummy"),
    plan("Logistic baseline C=1", "logistic_regression", {"C": 1.0}, stress=["tenure", "MonthlyCharges", "TotalCharges"]),
    plan("Regularized logistic C=0.2", "logistic_regression", {"C": 0.2}, stress=["tenure", "MonthlyCharges", "TotalCharges"]),
    plan("Flexible logistic C=3", "logistic_regression", {"C": 3.0}, stress=["tenure", "MonthlyCharges", "TotalCharges"]),
    plan("Extra Trees baseline", "extra_trees", {"n_estimators": 140, "min_samples_leaf": 2}, stress=["tenure", "MonthlyCharges", "TotalCharges"]),
    plan("Regularized Extra Trees", "extra_trees", {"n_estimators": 220, "max_depth": 12, "min_samples_leaf": 10}, stress=["tenure", "MonthlyCharges", "TotalCharges"]),
    plan("Random Forest baseline", "random_forest", {"n_estimators": 140, "min_samples_leaf": 2}, stress=["tenure", "MonthlyCharges", "TotalCharges"]),
    plan("Regularized Random Forest", "random_forest", {"n_estimators": 220, "max_depth": 10, "min_samples_leaf": 12}, stress=["tenure", "MonthlyCharges", "TotalCharges"]),
    plan("Histogram gradient boosting", "hist_gradient_boosting", {"learning_rate": 0.08, "min_samples_leaf": 20, "max_depth": 6}, stress=["tenure", "MonthlyCharges", "TotalCharges"]),
    plan("Regularized histogram boosting", "hist_gradient_boosting", {"learning_rate": 0.04, "min_samples_leaf": 35, "max_depth": 4}, stress=["tenure", "MonthlyCharges", "TotalCharges"]),
    plan("LightGBM baseline", "lightgbm", {"n_estimators": 140, "learning_rate": 0.05, "max_depth": 6, "min_samples_leaf": 20}, stress=["tenure", "MonthlyCharges", "TotalCharges"]),
    plan("Regularized LightGBM", "lightgbm", {"n_estimators": 220, "learning_rate": 0.03, "max_depth": 4, "min_samples_leaf": 35}, stress=["tenure", "MonthlyCharges", "TotalCharges"]),
    plan("XGBoost baseline", "xgboost", {"n_estimators": 160, "learning_rate": 0.05, "max_depth": 4}, stress=["tenure", "MonthlyCharges", "TotalCharges"]),
    plan("Logistic with charge features", "logistic_regression", {"C": 0.2}, CHARGE_FEATURES, ["tenure", "MonthlyCharges", "TotalCharges"]),
    plan("LightGBM with charge features", "lightgbm", {"n_estimators": 220, "learning_rate": 0.03, "max_depth": 4, "min_samples_leaf": 35}, RICH_FEATURES, ["tenure", "MonthlyCharges", "TotalCharges"]),
]

def run(repeats=2, max_rows=7043):
    RESULTS.mkdir(parents=True, exist_ok=True)
    completed = []
    for index, candidate in enumerate(PLANS, 1):
        exp_id = f"CHURN-{index:03d}"
        directory = RESULTS / exp_id
        result_file = directory / "result.json"
        if result_file.is_file():
            result = json.loads(result_file.read_text())
            if result["plan"] != candidate or len(result["folds"]) != repeats * 3:
                raise RuntimeError(f"Immutable result conflict at {exp_id}; use a fresh revision/directory")
        else:
            result = evaluate(candidate, max_rows, repeats, directory)
        completed.append({"id": exp_id, "title": candidate["title"], "model": candidate["model"], "metrics": result["metrics"], "fold_standard_deviation": result["fold_standard_deviation"], "wall_seconds": result["wall_seconds"], "result": result})
        print(f"{exp_id} {candidate['title']}: AUC={result['metrics']['roc_auc']:.4f}", flush=True)
    baseline = completed[1]["result"]
    for item in completed:
        result = item.pop("result")
        if result["split_hash"] == baseline["split_hash"] and result["data_hashes"] == baseline["data_hashes"]:
            item["paired_auc_delta_vs_logistic"] = sum(a["roc_auc"] - b["roc_auc"] for a, b in zip(result["folds"], baseline["folds"])) / len(result["folds"])
    leaderboard = sorted(completed, key=lambda row: row["metrics"]["roc_auc"], reverse=True)
    summary = {"project": "telco_churn", "planned": len(PLANS), "completed": len(completed), "evaluation": "2x3-fold adaptive stratified duplicate-group development CV", "production_approved": False, "baseline_id": "CHURN-002", "best": leaderboard[0], "leaderboard": leaderboard}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = ["# Telco Churn: 15-experiment benchmark", "", "> Development evidence only. The same CV is reused across candidates; no independent test or deployment approval.", "", "| Rank | ID | Experiment | Model | ROC-AUC | AP | Log loss | Δ AUC vs logistic |", "|---:|---|---|---|---:|---:|---:|---:|"]
    for rank, item in enumerate(leaderboard, 1):
        m = item["metrics"]
        lines.append(f"| {rank} | {item['id']} | {item['title']} | {item['model']} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} | {m['log_loss']:.4f} | {item['paired_auc_delta_vs_logistic']:+.4f} |")
    lines += ["", "## Protocol", "", "- `customerID` is excluded before modeling.", "- Blank `TotalCharges` at zero tenure becomes zero; all other missing numeric values are imputed inside each fold.", "- Categorical encoding, numeric imputation/scaling and derived features are fitted/executed inside each fold.", "- Exact duplicate allowed-input rows share a fold. The source has no verified household/account-history grouping or event-time split.", "- Every result includes OOF predictions, calibration, missing-input stress tests, sensitivity, data/code hashes and a replay recipe.", "", "## Interpretation", "", "The highest score is a candidate for independent temporal/external confirmation, not a universal best model. Prefer probability quality, robustness, operating constraints and stability—not ROC-AUC alone."]
    (ROOT / "churn_exp" / "CHURN_BENCHMARK.md").write_text("\n".join(lines) + "\n")
    return summary

def status():
    found = sum((RESULTS / f"CHURN-{i:03d}" / "result.json").is_file() for i in range(1, len(PLANS)+1))
    print(json.dumps({"planned": len(PLANS), "completed": found, "summary": str(RESULTS / "summary.json")}, indent=2))

def main():
    parser = argparse.ArgumentParser(description="DCLab Telco Churn benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run"); execute.add_argument("--repeats", type=int, choices=[1, 2, 3], default=2); execute.add_argument("--rows", type=int, default=7043)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "run": print(json.dumps(run(args.repeats, args.rows), indent=2))
    else: status()

if __name__ == "__main__": main()
