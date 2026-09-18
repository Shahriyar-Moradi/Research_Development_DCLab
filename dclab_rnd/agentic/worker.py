"""Deterministic ML worker. No LLM, arbitrary code, shell, or network tools.

Runs in the existing ML environment, separately from the NOOA runtime. All
scores are adaptive DEVELOPMENT CV, never a pristine held-out confirmation.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .catalog import ROOT, DATASETS, catalog

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source_paths(key, root=ROOT):
    if key == "hyperack": return [root / "hyper_ackt-dataset.csv"]
    if key == "telco_churn": return [root / "WA_Fn-UseC_-Telco-Customer-Churn.csv"]
    directory = root / "external_data" / key
    return [directory / "X.parquet", directory / "y.parquet", directory / "meta.json"]

def _load_custom(key, root):
    if key == "hyperack":
        path = source_paths(key, root)[0]
        frame = pd.read_csv(path)
        y = frame.pop("hyper_ack").astype(int)
        created = pd.to_datetime(frame.pop("created_date"), errors="coerce", utc=True)
        first = pd.to_datetime(frame.pop("first_created_at"), errors="coerce", utc=True)
        frame["created_month"] = created.dt.month
        frame["created_day_of_month"] = created.dt.day
        frame["created_hour"] = first.dt.hour
        frame["created_minute"] = first.dt.minute
        return frame, y
    path = source_paths(key, root)[0]
    frame = pd.read_csv(path, dtype={"SeniorCitizen": "string"})
    y = frame.pop("Churn").map({"No": 0, "Yes": 1}).astype(int)
    frame = frame.drop(columns=["customerID"])
    total = pd.to_numeric(frame["TotalCharges"].astype(str).str.strip(), errors="coerce")
    frame["TotalCharges"] = total.mask(total.isna() & frame["tenure"].eq(0), 0.0)
    return frame, y

def load(key, max_rows, root=ROOT):
    if key not in DATASETS:
        raise ValueError("Unknown dataset")
    directory = root / "external_data" / key
    if key in ("hyperack", "telco_churn"):
        X, y = _load_custom(key, root)
    else:
        X = pd.read_parquet(directory / "X.parquet")
        y = pd.read_parquet(directory / "y.parquet")["target"].astype(int)
    if len(X) != len(y) or set(y.unique()) != {0, 1}:
        raise ValueError("Expected aligned binary target")
    # Fixed label-independent sampling. Never tune the sample from outcomes.
    if len(X) > max_rows:
        rows = np.sort(np.random.default_rng(42).choice(len(X), max_rows, replace=False))
        X, y = X.iloc[rows], y.iloc[rows]
    row_ids = X.index.to_numpy()
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    categorical = DATASETS[key]["categorical"]
    cats = list(X.columns) if categorical == "ALL" else [c for c in categorical if c in X]
    # Factorized category codes must not acquire an ordinal numeric interpretation.
    for col in cats:
        X[col] = X[col].map(lambda value: str(value) if pd.notna(value) and value != -1 else np.nan)
    paths = source_paths(key, root)
    return X, y, cats, row_ids, {p.name: sha(p) for p in paths}

def profile(key, max_rows):
    X, y, cats, _, hashes = load(key, max_rows)
    columns = []
    for col in X:
        series = X[col]
        item = {"name": col, "kind": "categorical" if col in cats else "numeric", "missing_fraction": float(series.isna().mean()), "unique": int(series.nunique()), "blocked": col in DATASETS[key]["blocked"]}
        if col not in cats:
            item["quantiles"] = {str(k): float(v) if pd.notna(v) else None for k, v in series.quantile([0, .25, .5, .75, 1]).items()}
            item["target_abs_correlation"] = float(abs(series.corr(y))) if series.nunique() > 1 else None
        columns.append(item)
    warnings = ["Target associations are exploratory signals, not proof of leakage.", "Availability must be confirmed from event lineage and prediction timestamps.", "Only aggregate profiles are sent to the LLM; no source rows."]
    if key == "hyperack": warnings.append("The 0.9793 historical ceiling used post-outcome final fares; those columns are policy-blocked here.")
    if key == "telco_churn": warnings.append("customerID is removed before profiling/modeling; the dataset lacks entity history and a verified outcome-window timestamp.")
    return {"dataset": key, "rows": len(X), "positive_fraction": float(y.mean()), "exact_duplicate_fraction": float(X.duplicated().mean()), "columns": columns, "policy": next(c for c in catalog() if c["key"] == key), "hashes": hashes, "warnings": warnings}

class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Row-local transforms preserve lineage; no statistics are fitted on test rows."""
    def __init__(self, features=()):
        self.features = features
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        out = X.copy()
        for spec in self.features:
            a = pd.to_numeric(out[spec["inputs"][0]], errors="coerce")
            op = spec["operation"]
            if op == "signed_log":
                value = np.sign(a) * np.log1p(abs(a))
            elif op == "missing_indicator":
                value = a.isna().astype(float)
            else:
                b = pd.to_numeric(out[spec["inputs"][1]], errors="coerce")
                if op == "ratio":
                    value = a / b.where(abs(b) > 1e-9)
                elif op == "difference":
                    value = a - b
                elif op == "product":
                    value = a * b
                else:
                    raise ValueError("Unsupported transformation")
            out[spec["name"]] = value.replace([np.inf, -np.inf], np.nan)
        return out

def validate_plan(plan, X, cats):
    blocked = set(DATASETS[plan["dataset"]]["blocked"])
    drops = set(plan["drop_columns"])
    if drops - set(X) or set(plan["stress_columns"]) - set(X):
        raise ValueError("Unknown dropped/stressed column")
    model_raw = set(X) - blocked - drops
    if not model_raw and not plan["features"]:
        raise ValueError("No usable features")
    # A safe raw column may feed a derived feature while being excluded from
    # the final model matrix. This supports representation-replacement tests.
    known = set(X) - blocked
    required_sources = set()
    derived_names = set()
    for spec in plan["features"]:
        if spec["name"] in known or spec["name"] in derived_names:
            raise ValueError("Feature name collision")
        available = known | derived_names
        if not set(spec["inputs"]) <= available:
            raise ValueError("Derived feature has blocked, unavailable, or unknown inputs")
        if set(spec["inputs"]) & set(cats):
            raise ValueError("Arithmetic on categorical codes is forbidden")
        required_sources.update(i for i in spec["inputs"] if i in X.columns)
        derived_names.add(spec["name"])
    effective_inputs = model_raw | required_sources
    if set(plan["stress_columns"]) - effective_inputs:
        raise ValueError("Stress columns must be model inputs or derived-feature sources")
    return {
        "model_raw": [c for c in X if c in model_raw],
        "pipeline_inputs": [c for c in X if c in effective_inputs],
        "derived": [spec["name"] for spec in plan["features"]],
    }

def model_for(name, params):
    params = {k: v for k, v in params.items() if v is not None}
    allowed = {
        "dummy": set(), "logistic_regression": {"C"},
        "extra_trees": {"n_estimators", "max_depth", "min_samples_leaf"},
        "random_forest": {"n_estimators", "max_depth", "min_samples_leaf"},
        "hist_gradient_boosting": {"max_depth", "min_samples_leaf", "learning_rate"},
        "lightgbm": {"n_estimators", "max_depth", "min_samples_leaf", "learning_rate"},
        "xgboost": {"n_estimators", "max_depth", "learning_rate"},
    }
    if name not in allowed or set(params) - allowed[name]:
        raise ValueError("Unsupported model parameters")
    if name == "dummy": return DummyClassifier(strategy="prior")
    if name == "logistic_regression": return LogisticRegression(max_iter=1200, random_state=42, **params)
    if name in ("extra_trees", "random_forest"):
        cls = ExtraTreesClassifier if name == "extra_trees" else RandomForestClassifier
        return cls(random_state=42, n_jobs=1, **{"n_estimators": 100, "min_samples_leaf": 2, **params})
    if name == "hist_gradient_boosting": return HistGradientBoostingClassifier(random_state=42, max_iter=100, **params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        if "min_samples_leaf" in params: params["min_child_samples"] = params.pop("min_samples_leaf")
        return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **{"n_estimators": 100, **params})
    from xgboost import XGBClassifier
    return XGBClassifier(random_state=42, n_jobs=1, tree_method="hist", eval_metric="logloss", **{"n_estimators": 100, "max_depth": 4, **params})

def build_pipeline(plan, columns, cats):
    # Accept a plain raw-column list for focused unit tests; production passes
    # the fully validated lineage map from validate_plan().
    if isinstance(columns, list):
        model_raw, derived = columns, [f["name"] for f in plan["features"]]
    else:
        model_raw, derived = columns["model_raw"], columns["derived"]
    numeric = [c for c in model_raw if c not in cats] + derived
    categorical = [c for c in model_raw if c in cats]
    pre = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=3))]), categorical),
    ], remainder="drop")
    return Pipeline([("features", DerivedFeatures(plan["features"])), ("preprocess", pre), ("model", model_for(plan["model"], plan["parameters"]))])

def metrics(y, p):
    return {"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p))}

def evaluate(plan, max_rows, repeats, output):
    start = time.monotonic()
    key = plan["dataset"]
    X, y, cats, row_ids, hashes = load(key, max_rows)
    columns = validate_plan(plan, X, cats)
    model_raw, pipeline_inputs = columns["model_raw"], columns["pipeline_inputs"]
    # Group on ALL policy-allowed original inputs, independent of each candidate's
    # drops/derived features. This fixes the split across paired ablations.
    group_cols = [c for c in X if c not in DATASETS[key]["blocked"]]
    groups = pd.util.hash_pandas_object(X[group_cols], index=False).to_numpy()
    folds, predictions, stresses, importance = [], [], [], []
    split_hasher = hashlib.sha256()
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42 + repeat)
        for fold, (train, test) in enumerate(splitter.split(X, y, groups)):
            if len(set(y.iloc[train])) < 2 or len(set(y.iloc[test])) < 2:
                raise ValueError("A group fold contains one class; dataset needs another evaluation protocol")
            assert not set(groups[train]) & set(groups[test])
            split_hasher.update(np.asarray(test, dtype=np.int64).tobytes())
            pipe = build_pipeline(plan, columns, cats)
            before = time.monotonic()
            pipe.fit(X.iloc[train][pipeline_inputs], y.iloc[train])
            p = pipe.predict_proba(X.iloc[test][pipeline_inputs])[:, 1]
            score = metrics(y.iloc[test], p)
            folds.append({"repeat": repeat, "fold": fold, **score, "fit_predict_seconds": time.monotonic() - before})
            predictions.extend({"row_id": int(row_ids[i]), "repeat": repeat, "fold": fold, "target": int(y.iloc[i]), "probability": float(prob)} for i, prob in zip(test, p))
            for column in plan["stress_columns"]:
                stressed = X.iloc[test][pipeline_inputs].copy()
                stressed[column] = np.nan
                sp = pipe.predict_proba(stressed)[:, 1]
                stressed_metrics = metrics(y.iloc[test], sp)
                stresses.append({"repeat": repeat, "fold": fold, "column": column, **stressed_metrics,
                    "metric_deltas_clean_minus_stressed": {metric: score[metric] - stressed_metrics[metric] for metric in score}})
            # Explanatory permutation sensitivity, first fold only, all raw inputs.
            # Regenerates derived features when a parent changes. Not causal impact.
            if repeat == fold == 0:
                rng = np.random.default_rng(867)
                for col in pipeline_inputs:
                    permuted = X.iloc[test][pipeline_inputs].copy()
                    permuted[col] = rng.permutation(permuted[col].to_numpy())
                    changed = pipe.predict_proba(permuted)[:, 1]
                    importance.append({"column": col, "auc_drop": score["roc_auc"] - float(roc_auc_score(y.iloc[test], changed))})
    mean = {m: float(np.mean([f[m] for f in folds])) for m in metrics(y.iloc[test], p)}
    spread = {m: float(np.std([f[m] for f in folds], ddof=1)) for m in mean}
    frame = pd.DataFrame(predictions)
    repeat_summary = []
    for repeat in range(repeats):
        repeat_frame = frame[frame.repeat == repeat]
        repeat_summary.append({"repeat": repeat, **metrics(repeat_frame.target, repeat_frame.probability)})
    calibration = []
    # Report each repeat independently; repeated OOF predictions are correlated
    # and must not be represented as twice as many independent observations.
    for repeat in range(repeats):
        repeat_frame = frame[frame.repeat == repeat]
        for lower in np.arange(0, 1, .2):
            section = repeat_frame[(repeat_frame.probability >= lower) & (repeat_frame.probability < lower + .2 if lower < .8 else repeat_frame.probability <= 1)]
            if len(section): calibration.append({"repeat": repeat, "lower": float(lower), "upper": float(lower + .2), "count": len(section), "predicted": float(section.probability.mean()), "observed": float(section.target.mean())})
    import importlib.metadata
    result = {"dataset": key, "plan": plan, "rows": len(X), "feature_count": len(model_raw) + len(plan["features"]), "retained_columns": model_raw, "pipeline_source_columns": pipeline_inputs, "derived_feature_lineage": plan["features"], "excluded_by_policy": DATASETS[key]["blocked"], "folds": folds, "repeat_summary": repeat_summary, "metrics": mean, "fold_standard_deviation": spread, "split_hash": split_hasher.hexdigest(), "data_hashes": hashes, "worker_hash": sha(Path(__file__)), "catalog_hash": sha(Path(__file__).with_name("catalog.py")), "python": sys.version.split()[0], "packages": {p: importlib.metadata.version(p) for p in ["scikit-learn", "numpy", "pandas"]}, "wall_seconds": time.monotonic() - start, "stress_tests": stresses, "input_sensitivity": sorted(importance, key=lambda i: -i["auc_drop"]), "output_calibration": calibration, "confusion_matrix_at_0_5_by_repeat": [{"repeat": repeat, "matrix": confusion_matrix(section.target, section.probability >= .5, labels=[0, 1]).tolist()} for repeat, section in frame.groupby("repeat")], "evaluation": "Adaptive repeated stratified GROUP development CV; exact duplicate allowed-input rows share a fold. No unbiased holdout or causal/production claim.", "limitations": ["Repeated folds are correlated; fold spread is not a confidence interval.", "Hyperparameter/feature selection uses the same development CV; requires fresh external confirmation.", "No entity/time IDs: duplicate grouping cannot guarantee entity or temporal separation.", "Permutation sensitivity is a single-fold diagnostic, not causal importance.", "Legacy category mappings and source availability require verification."], "production_approved": False}
    output.mkdir(parents=True, exist_ok=True)
    frame.to_json(output / "oof_predictions.jsonl", orient="records", lines=True)
    recipe = {"schema_version": 1, "plan": plan, "max_rows": max_rows, "repeats": repeats, "data_hashes": hashes, "worker_hash": result["worker_hash"], "catalog_hash": result["catalog_hash"], "blocks": ["load fingerprinted public data", "decision-time policy gate", "fixed duplicate-group folds", "fold-local feature/preprocessing pipeline", "fit and predict OOF", "stress and permutation diagnostics", "paired development comparison; external confirmation pending"]}
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2))
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result

def main():
    request = json.load(sys.stdin)
    if request["action"] == "profile":
        result = [profile(k, request["max_rows"]) for k in request["datasets"]]
    elif request["action"] == "experiment":
        result = evaluate(request["plan"], request["max_rows"], request["repeats"], Path(request["output"]))
    else:
        raise ValueError("Unknown worker action")
    print(json.dumps(result, allow_nan=False))

if __name__ == "__main__":
    main()
