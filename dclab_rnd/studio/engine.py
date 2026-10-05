"""The five-stage workflow on a project's own data.

Every stage is deterministic code built from the expansion campaign's primitives
(locked holdout, training-only folds, fit-fold recipes, explicit selection rules).
The same code that produced the R&D evidence runs the person's project, so each
record has the same shape as a campaign result and the agent can cite like with like.
"""

from __future__ import annotations

import copy
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dclab_rnd.categoricals import HEURISTIC_RULE, category_code_columns
from dclab_rnd.expansion import datasets as ds
from dclab_rnd.expansion import runner as rn

from . import agent, graph
from .solution import Solution
from .data import column_kind, load_table, sha256
from .store import STAGE_KEYS, ProjectStore

TASK_TYPE = {"binary": "binary_imbalanced", "multiclass": "multiclass", "regression": "timeseries_regression"}
STAGES: list[dict[str, str]] = [
    {"key": "data", "kind": "data_understanding", "title": "Understand the data", "workflow": "WF-04",
     "question": "What is in the data, and what could make a score untrustworthy?"},
    {"key": "leakage", "kind": "leakage_audit", "title": "Audit leakage", "workflow": "WF-05",
     "question": "Which columns would not exist at the prediction moment, and how much would they inflate the score?"},
    {"key": "features", "kind": "feature_engineering", "title": "Climb the feature ladder", "workflow": "WF-06",
     "question": "Which feature recipe earns its place on identical folds?"},
    {"key": "models", "kind": "model_selection", "title": "Screen algorithms", "workflow": "WF-07",
     "question": "Which model family leads on identical folds, once fold spread is counted?"},
    {"key": "final", "kind": "optimization_reliability", "title": "Tune and confirm on the holdout", "workflow": "WF-08",
     "question": "Does careful tuning help, and what is the honest score on rows the model never saw?"},
]
STAGE_BY_KEY = {s["key"]: s for s in STAGES}
QUICK_ROWS = 3000
DEFAULT_FOLDS, FOLD_CAPS = 3, (2, 10)


def cv_folds(project: dict[str, Any]) -> int:
    """Training folds for this project: the wizard's choice (project settings), 3 when none was made."""
    try:
        folds = int((project.get("settings") or {}).get("folds") or DEFAULT_FOLDS)
    except (TypeError, ValueError):
        folds = DEFAULT_FOLDS
    return max(FOLD_CAPS[0], min(folds, FOLD_CAPS[1]))


def split_for(solution: Any) -> str:
    """The split the engine uses. The solution decides it, not a setting: a declared time column means the holdout
    is the latest period (DCLAB-R02); otherwise a declared group column keeps each group on one side; otherwise
    a stratified random split. ``solution`` is a Solution or its dict."""
    get = solution.get if isinstance(solution, dict) else lambda key: getattr(solution, key, None)
    return "time" if get("time_column") else "group" if get("group_column") else "stratified"
_BUNDLES: dict[tuple[str, str, str], Any] = {}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------- prepare


class Prepared:
    def __init__(self, frame: pd.DataFrame, solution: Solution, bundle: ds.TaskBundle, roles: dict[str, str],
                 class_labels: list[str] | None, recipes: dict[str, dict[str, Any]], sampling: dict[str, Any], code_rule: str = ""):
        self.frame, self.solution, self.bundle, self.roles = frame, solution, bundle, roles
        self.class_labels, self.recipes, self.sampling = class_labels, recipes, sampling
        self.code_rule = code_rule  # how bundle.spec.categorical_columns was decided

    @property
    def task(self) -> str:
        return self.solution.task

    @property
    def metric(self) -> str:
        return self.bundle.spec.primary_metric


def _target_series(frame: pd.DataFrame, solution: Solution) -> tuple[pd.Series, list[str] | None]:
    raw = frame[solution.target]
    if solution.task == "binary":
        positive = solution.positive_label
        if positive is None:
            values = sorted(raw.dropna().unique(), key=str)
            positive = str(values[-1])
        y = (raw.astype(str).str.strip() == str(positive)).astype(int)
        return y, None
    if solution.task == "multiclass":
        labels = sorted(raw.dropna().astype(str).unique(), key=str)
        mapping = {label: i for i, label in enumerate(labels)}
        return raw.astype(str).map(mapping).astype(int), labels
    return pd.to_numeric(raw, errors="coerce").astype(float), None


def prepare(store: ProjectStore, project: dict[str, Any]) -> Prepared:
    """Load the table, apply the solution, lock the holdout and register the feature recipes."""
    if not project.get("data") or not project.get("solution"):
        raise ValueError("The project needs data and a solution first")
    solution = Solution(**project["solution"])
    path = store.data_dir(project["id"]) / project["data"]["filename"]
    declared = project["data"].get("categorical")  # the R&D's list for one of its samples; None for an uploaded table
    key = (project["id"], project["data"]["sha256"], _hash(project["solution"]) + str(project["settings"]) + str(declared))
    if key in _BUNDLES:
        return _BUNDLES[key]
    frame = load_table(path)
    solution.check_columns(list(frame.columns))
    # Yes/no columns arrive as booleans (cleaned uploads, CSVs with True/False); numpy cannot take quantiles or
    # differences of booleans, so features become 0/1 with missing kept missing. The target is compared as text.
    for column in frame.columns:
        if column != solution.target and (pd.api.types.is_bool_dtype(frame[column]) or str(frame[column].dtype) == "boolean"):
            frame[column] = pd.to_numeric(frame[column].astype("object").map({True: 1.0, False: 0.0}), errors="coerce")
    source_rows = len(frame)
    frame = frame[frame[solution.target].notna()].reset_index(drop=True)
    y, class_labels = _target_series(frame, solution)
    keep = y.notna()
    frame, y = frame[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    X = frame.drop(columns=[solution.target])

    roles: dict[str, str] = {}
    identifiers: dict[str, str] = {c: "declared identifier (never a feature)" for c in solution.identifiers if c in X}
    text_columns = tuple(c for c in solution.text_columns if c in X)
    task_type = TASK_TYPE[solution.task]
    if text_columns and solution.task == "binary":
        task_type = "text_tabular_binary"
    elif text_columns:
        for column in text_columns:
            identifiers[column] = "free text is modeled for binary targets only; excluded here"
        text_columns = ()
    time_column = solution.time_column
    if time_column:
        if column_kind(X[time_column]) == "datetime":
            stamps = pd.to_datetime(X[time_column], errors="coerce")
            valid = stamps.notna()
            X, y, frame = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True), frame[valid].reset_index(drop=True)
            X[time_column] = pd.to_datetime(X[time_column], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            identifiers[time_column] = "time key: orders the split; only calendar features are derived from it"
            datetime_time = True
        else:
            X[time_column] = pd.to_numeric(X[time_column], errors="coerce")
            identifiers[time_column] = "time key: orders the split; a dataset-relative counter is not a production feature"
            datetime_time = False
    else:
        datetime_time = False
    for column in X.columns:
        if column not in identifiers and column != solution.group_column and column not in text_columns and column_kind(X[column]) == "datetime":
            identifiers[column] = "datetime column that is not the declared time column; excluded to avoid one-hot dates"
    blocked = {f.column: (f.reason or "declared unavailable at the prediction moment") for f in solution.forbidden if f.column in X}
    for column in X.columns:
        roles[column] = ("blocked" if column in blocked else "identifier" if column in identifiers and column != time_column
                         else "time" if column == time_column else "group" if column == solution.group_column
                         else "text" if column in text_columns else "feature")

    settings = project.get("settings") or {}
    max_rows = int(settings.get("max_rows") or 20000)
    if settings.get("quick"):
        max_rows = min(max_rows, QUICK_ROWS)
    sampling: dict[str, Any] = {"source_rows": source_rows, "rows_with_target": int(len(X)), "max_rows": max_rows}
    if len(X) > max_rows:
        if time_column:
            order = np.argsort(ds._time_key(X[time_column]), kind="stable")[-max_rows:]
            sampling["rule"] = f"kept the latest {max_rows} rows by `{time_column}` (label-independent)"
        else:
            order = np.sort(np.random.default_rng(ds.RANDOM_STATE).choice(len(X), max_rows, replace=False))
            sampling["rule"] = f"random sample of {max_rows} rows, seed {ds.RANDOM_STATE} (label-independent)"
        X, y = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
    sampling["rows_used"] = int(len(X))

    positive_rate = float(y.mean()) if solution.task == "binary" else None
    metric = solution.resolved_metric(positive_rate)
    split = split_for(solution)
    regression = solution.task == "regression"
    spec = ds.DatasetSpec(
        key=f"project_{project['id']}", name=project["name"], task_type=task_type, primary_metric=metric,
        target=solution.target, url=f"upload://{project['data']['filename']}", filename=project["data"]["filename"],
        sha256=project["data"]["sha256"], description=project.get("goal") or project["name"],
        decision_time_contract=solution.prediction_moment, split_strategy=split, source_rows=source_rows,
        feature_count=int(X.shape[1]), blocked_features=blocked, identifier_columns=identifiers,
        text_columns=text_columns, time_column=time_column, group_column=solution.group_column,
        positive_rate=positive_rate,
        settings={"precision_target": 0.9, "fe_tolerance": 0.01 if regression else 0.002, "fe_tolerance_relative": regression,
                  "tuning_margin": 0.01 if regression else 0.001, "tuning_margin_relative": regression},
    )
    bundle = ds.prepare_bundle(spec, X, y, data_paths=[path], source_rows=source_rows, sampling=sampling,
                               class_labels=class_labels, cv_folds=cv_folds(project))
    # Numeric columns that stand for categories get no log or product feature (DCLAB-R11). Decided once,
    # on training rows: the declaration when there is one, otherwise the conservative heuristic.
    features = [c for c, role in roles.items() if role == "feature"]
    codes = category_code_columns(bundle.X_train[features], declared)
    bundle.spec = replace(spec, categorical_columns=tuple(codes))
    code_rule = "declared in the R&D dataset catalog for this sample" if declared is not None else HEURISTIC_RULE
    recipes = _register_recipes(bundle.spec, X, roles, datetime_time)
    prepared = Prepared(frame, solution, bundle, roles, class_labels, recipes, sampling, code_rule)
    _BUNDLES.clear()
    _BUNDLES[key] = prepared
    return prepared


def _hash(value: Any) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _register_recipes(spec: ds.DatasetSpec, X: pd.DataFrame, roles: dict[str, str], datetime_time: bool) -> dict[str, dict[str, Any]]:
    features = [c for c, role in roles.items() if role == "feature"]
    codes = set(spec.categorical_columns)
    numeric = [c for c in features if c not in codes and pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 2]
    left_out = f" ({len(codes)} category-code column{'' if len(codes) == 1 else 's'} left out)" if codes else ""
    if spec.task_type == "text_tabular_binary":
        recipes = {
            "tabular": {"description": f"{len(features)} structured columns only, no text", "tabular": True},
            "tfidf_word": {"description": "word 1-2gram TF-IDF of the text columns", "word": True},
            "combined": {"description": "word TF-IDF + structured columns + text-length/missing meta", "word": True, "tabular": True, "meta": True},
        }
        rn.LEAKAGE_RECIPE[spec.key] = "combined"
    else:
        recipes = {"raw": {"description": f"the {len(features)} safe input columns as given (categories and category codes one-hot, fitted per fold)", "derive": ()}}
        if numeric:
            recipes["log_numeric"] = {"description": f"raw + signed log of the {len(numeric)} numeric inputs{left_out}", "derive": ("log_numeric",)}
        if datetime_time:
            recipes["calendar"] = {"description": "raw + calendar features from the time column (day-of-year sin/cos, day of month, ISO week, weekend)", "derive": ("calendar",)}
        if 2 <= len(numeric) <= 8:
            pairs = len(numeric) * (len(numeric) - 1) // 2
            recipes["poly2"] = {"description": f"raw + all {pairs} pairwise products of the numeric inputs{left_out}", "derive": ("poly2",)}
        if spec.task_type != "timeseries_regression" and len(features) >= 10:
            k = max(5, int(round(0.6 * len(features))))
            recipes["selected_mi"] = {"description": f"top-{k} inputs by fit-fold mutual information", "derive": (), "select_k": k}
        rn.LEAKAGE_RECIPE[spec.key] = "raw"
    rn.RECIPES[spec.key] = recipes
    return recipes


# ---------------------------------------------------------------------- stage executors


def _claim(cid: str, kind: str, statement: str, evidence: list[str], limitations: list[str] | None = None) -> dict[str, Any]:
    return rn._claim(cid, kind, statement, evidence, limitations)


def _label(metric: str) -> str:
    return rn.METRIC_LABEL.get(metric, metric)


def stage_data(p: Prepared, eid: str) -> tuple[dict, list, str]:
    bundle, spec = p.bundle, p.bundle.spec
    X_tr, y_tr, X_te = bundle.X_train, bundle.y_train, bundle.X_test
    profiles = []
    for column in X_tr.columns:
        series = X_tr[column]
        missing = float(series.isna().mean())
        unique_ratio = float(series.nunique(dropna=False) / max(len(series), 1))
        top_frequency = float(series.value_counts(dropna=False, normalize=True).iloc[0]) if len(series) else 0.0
        psi = rn._psi(series, X_te[column]) if pd.api.types.is_numeric_dtype(series) else None
        risks = []
        if missing > 0.05:
            risks.append("missingness")
        if rn._identifier_like(series, unique_ratio) and p.roles.get(column) == "feature":
            risks.append("identifier_like")
        if top_frequency > 0.995:
            risks.append("near_constant")
        if psi is not None and psi > 0.20:
            risks.append("train_holdout_feature_shift")
        profiles.append({"feature": str(column), "role": p.roles.get(column, "feature"), "dtype": str(series.dtype),
                         "missing_rate": missing, "unique_count": int(series.nunique(dropna=False)), "unique_ratio": unique_ratio,
                         "top_value_frequency": top_frequency, "train_vs_holdout_psi": psi, "production_risks": risks})
    feature_cols = [c for c, role in p.roles.items() if role == "feature"]
    evidence: dict[str, Any] = {
        "source_rows": bundle.source_rows, "raw_feature_count": int(X_tr.shape[1]), "usable_feature_count": len(feature_cols),
        "train_rows": int(len(X_tr)), "holdout_rows": int(len(X_te)), "holdout_split": bundle.holdout_description,
        "cv_protocol": bundle.cv_description, "sampling": bundle.sampling,
        "missing_cell_rate_train": float(X_tr.isna().sum().sum() / max(X_tr.size, 1)),
        "duplicate_feature_row_rate_train": float(X_tr[feature_cols].duplicated().mean()) if feature_cols else 0.0,
        "duplicate_feature_rows_train": int(X_tr[feature_cols].duplicated().sum()) if feature_cols else 0,
        "feature_profiles": profiles,
        "risk_features": {pr["feature"]: pr["production_risks"] for pr in profiles if pr["production_risks"]},
        "column_roles": dict(p.roles),
        "note": "Holdout feature distributions (never labels) are compared via PSI; the holdout labels stay sealed until the final stage.",
    }
    if p.task == "binary":
        positives, rate = int(y_tr.sum()), float(y_tr.mean())
        evidence["target_summary"] = {"positive_rate_train": rate, "positives_train": positives, "negatives_train": int(len(y_tr) - positives),
                                      "imbalance_ratio_neg_per_pos": float((len(y_tr) - positives) / max(positives, 1)),
                                      "majority_baseline_accuracy_train": float(max(rate, 1 - rate)), "positive_label": p.solution.positive_label}
        target_text = f"a positive rate of {rate:.1%} ({positives} positives)"
    elif p.task == "multiclass":
        counts = y_tr.value_counts().sort_index()
        share = counts / counts.sum()
        evidence["target_summary"] = {"classes": int(len(counts)),
                                      "class_counts_train": {str(p.class_labels[i] if p.class_labels else i): int(v) for i, v in counts.items()},
                                      "smallest_class_count": int(counts.min()), "largest_class_count": int(counts.max()),
                                      "imbalance_ratio_max_min": float(counts.max() / max(counts.min(), 1)),
                                      "majority_baseline_accuracy_train": float(share.max())}
        target_text = f"{len(counts)} classes with {counts.min()}–{counts.max()} training rows each"
    else:
        evidence["target_summary"] = {"mean_train": float(y_tr.mean()), "std_train": float(y_tr.std()), "min_train": float(y_tr.min()),
                                      "median_train": float(y_tr.median()), "max_train": float(y_tr.max()),
                                      "coefficient_of_variation_train": float(y_tr.std() / y_tr.mean()) if y_tr.mean() else None}
        target_text = f"a target mean of {y_tr.mean():.4g} (std {y_tr.std():.4g}, range {y_tr.min():.4g}–{y_tr.max():.4g})"
    if spec.time_column:
        evidence["time_range"] = {"train": [str(X_tr[spec.time_column].iloc[0]), str(X_tr[spec.time_column].iloc[-1])],
                                  "holdout": [str(X_te[spec.time_column].iloc[0]), str(X_te[spec.time_column].iloc[-1])]}
    if spec.group_column:
        sizes = X_tr[spec.group_column].value_counts()
        evidence["group_stats"] = {"train_groups": int(len(sizes)), "holdout_groups": int(X_te[spec.group_column].nunique()),
                                   "largest_group_share": float(sizes.iloc[0] / sizes.sum()), "median_rows_per_group": float(sizes.median())}
    if spec.text_columns:
        evidence["text_stats"] = {c: {"missing_rate": float(X_tr[c].isna().mean()),
                                      "median_words": float(X_tr[c].dropna().astype(str).str.split().str.len().median())} for c in spec.text_columns}
    risks = evidence["risk_features"]
    claims = [
        _claim(f"{eid}-C1", "fact", f"{bundle.source_rows} source rows and {X_tr.shape[1]} columns; {len(feature_cols)} are usable inputs under the solution. "
               f"The locked holdout is {bundle.holdout_description} ({len(X_te)} rows); {len(X_tr)} training rows remain with {target_text}.",
               ["evidence.source_rows", "evidence.usable_feature_count", "evidence.holdout_split", "evidence.target_summary"]),
        _claim(f"{eid}-C2", "risk", f"{len(risks)} of {X_tr.shape[1]} columns triggered a review rule (missingness > 5%, identifier-like, near-constant, or train/holdout PSI > 0.20): "
               + ("; ".join(f"{k}: {', '.join(v)}" for k, v in list(risks.items())[:12]) or "none") + ".",
               ["evidence.feature_profiles", "evidence.risk_features"],
               ["PSI on feature distributions is a smoke check; it uses no holdout labels and does not prove production stability."]),
    ]
    if p.task == "binary":
        ts = evidence["target_summary"]
        claims.append(_claim(f"{eid}-C3", "risk", f"Predicting the majority class for every row already scores {ts['majority_baseline_accuracy_train']:.1%} accuracy, "
                             f"so accuracy alone cannot judge this model; the solution scores {_label(p.metric)}.",
                             ["evidence.target_summary"]))
    elif p.task == "multiclass":
        ts = evidence["target_summary"]
        claims.append(_claim(f"{eid}-C3", "risk", f"The smallest class has {ts['smallest_class_count']} training rows (largest/smallest ratio {ts['imbalance_ratio_max_min']:.1f}); "
                             f"macro-F1 weights every class equally, so rare classes decide the score.", ["evidence.target_summary"]))
    else:
        ts = evidence["target_summary"]
        claims.append(_claim(f"{eid}-C3", "fact", f"The target's coefficient of variation is {ts['coefficient_of_variation_train'] or 0:.2f}; "
                             + (f"rows are ordered by `{spec.time_column}` so the holdout is strictly later than training." if spec.time_column
                                else "no time column was declared, so rows are split at random; if these rows are events in time, declare the time column."),
                             ["evidence.target_summary", "evidence.holdout_split"]))
    summary = (f"Profiled {len(X_tr)} training rows × {X_tr.shape[1]} columns (holdout: {bundle.holdout_description}); "
               f"per-column missingness, uniqueness and PSI; {len(risks)} columns flagged for review.")
    return evidence, claims, summary


def stage_leakage(p: Prepared, eid: str) -> tuple[dict, list, str]:
    bundle, spec, task = p.bundle, p.bundle.spec, p.bundle.task_type
    metric, higher = spec.primary_metric, spec.higher_is_better
    X_tr, y_tr = bundle.X_train, bundle.y_train
    declared = [c for c in spec.blocked_features if c in X_tr.columns]
    skip = [c for c, role in p.roles.items() if role in ("identifier", "time", "group", "text")]
    sample_X, sample_y = X_tr, y_tr
    if len(X_tr) > 60000:
        sample_X = X_tr.sample(60000, random_state=ds.RANDOM_STATE)
        sample_y = y_tr.loc[sample_X.index]
    findings = rn.suspicious_features(sample_X, sample_y, task, declared=declared, skip=skip)
    probe = [c for c in sample_X.columns if c not in skip][:3]
    canary = rn._canary_check(sample_X[probe], sample_y, task, skip) if probe else {"passed": True, "note": "no probe columns"}
    ranking = rn.univariate_ranking(sample_X, sample_y, task, skip=skip)
    audit_model = rn._fallback_family("lightgbm") if task != "text_tabular_binary" else "logistic_regression"
    recipe = rn.LEAKAGE_RECIPE[spec.key]
    safe = rn.cross_validate(bundle, recipe, audit_model, config_id="safe")
    comparison: dict[str, Any] = {"model": audit_model, "recipe": recipe, "metric": metric, "safe": safe}
    lift = 0.0
    if declared:
        unsafe = rn.cross_validate(bundle, recipe, audit_model, include_blocked=declared, config_id="unsafe_all_blocked")
        lift = rn.oriented_gain(rn._mean(unsafe, metric), rn._mean(safe, metric), higher)
        comparison["unsafe"], comparison["apparent_lift"] = unsafe, lift
        if 1 < len(declared) <= 6:
            comparison["per_column"] = {}
            for column in declared:
                single = rn.cross_validate(bundle, recipe, audit_model, include_blocked=[column], config_id=f"unsafe_{column}")
                comparison["per_column"][column] = {"metric_mean": rn._mean(single, metric),
                                                    "apparent_lift": rn.oriented_gain(rn._mean(single, metric), rn._mean(safe, metric), higher)}
    non_declared = [f for f in findings if "declared_post_outcome_or_contested" not in f["reasons"]]
    evidence: dict[str, Any] = {
        "declared_leakage_features": declared,
        "policy_rationale": " ".join(f"`{c}`: {r}" for c, r in spec.blocked_features.items()) or "No column is blocked: " + spec.decision_time_contract,
        "identifier_exclusions": dict(spec.identifier_columns), "decision_time_contract": spec.decision_time_contract,
        "heuristic_review_candidates": findings,
        "heuristic_rules": {"strong_univariate_signal": rn.STRONG_SIGNAL[task], "extreme_univariate_signal": rn.EXTREME_SIGNAL[task],
                            "identifier_like": "unique_ratio >= 0.98 and non-numeric or integer-valued", "rows_scanned": int(len(sample_X))},
        "top_univariate_signals": ranking, "detector_canary": canary, "detector_canary_passed": bool(canary.get("passed", False)),
        "safe_vs_unsafe_training_cv": comparison, "apparent_lift": float(lift), "apparent_lift_metric": metric,
        "cv_protocol": bundle.cv_description,
        "warning": "Name, univariate-signal, identity and uniqueness rules propose a review; only decision-time semantics confirm leakage.",
    }
    claims = [_claim(f"{eid}-C1", "fact", f"The {task} leakage detector caught its synthetic canaries: {canary}.",
                     ["evidence.detector_canary", "evidence.detector_canary_passed"])]
    if declared:
        per = comparison.get("per_column", {})
        per_text = (" Per column: " + ", ".join(f"{c} {rn._signed(v['apparent_lift'], metric)}" for c, v in per.items()) + ".") if per else ""
        claims.append(_claim(f"{eid}-C2", "decision",
                             f"Forbidden under the solution: {declared}. Including them moves training-CV {_label(metric)} ({audit_model}, {recipe} recipe) "
                             f"from {rn._ms(safe, metric)} to {rn._ms(comparison['unsafe'], metric)}, "
                             + (f"an apparent lift of {rn._signed(lift, metric)} that would not exist in production." if lift > 0
                                else f"no apparent lift ({rn._signed(lift, metric)}); the exclusion rests on decision-time semantics, not on CV inflation.") + per_text,
                             ["evidence.declared_leakage_features", "evidence.policy_rationale", "evidence.safe_vs_unsafe_training_cv", "evidence.apparent_lift"],
                             [evidence["warning"]]))
    else:
        claims.append(_claim(f"{eid}-C2", "decision", f"No column is forbidden under the solution; the safe {audit_model} {recipe} baseline scores "
                             f"{rn._ms(safe, metric)} training-CV {_label(metric)}, apparent lift 0 by construction.",
                             ["evidence.declared_leakage_features", "evidence.policy_rationale", "evidence.safe_vs_unsafe_training_cv"], [evidence["warning"]]))
    flagged = ", ".join(f"{f['feature']} ({'/'.join(f['reasons'])})" for f in non_declared) or "none"
    claims.append(_claim(f"{eid}-C3", "risk", f"Heuristics flagged {len(non_declared)} non-declared column(s) for human review: {flagged}. Strongest single-feature signal: "
                         + (f"{ranking[0]['feature']} ({ranking[0]['signal']:.3f})." if ranking else "n/a."),
                         ["evidence.heuristic_review_candidates", "evidence.top_univariate_signals", "evidence.heuristic_rules"],
                         ["Strong signal alone is not leakage; a legitimate lag or a real driver is expected to be predictive."]))
    parts = [f"Scanned {len(sample_X)} training rows with {task} heuristics and synthetic canaries;"]
    if declared:
        parts.append(f"compared {audit_model} on the {recipe} recipe with vs without {declared}: {_label(metric)} {rn._ms(safe, metric)} safe vs "
                     f"{rn._ms(comparison['unsafe'], metric)} unsafe (apparent lift {rn._signed(lift, metric)}).")
    else:
        parts.append(f"no forbidden columns; safe {audit_model} {recipe} baseline {_label(metric)} {rn._ms(safe, metric)}.")
    if spec.split_strategy == "time":
        from sklearn.model_selection import KFold

        folds = len(bundle.cv_splits)  # the same number of folds as the time-ordered protocol it is compared with
        random_splits = list(KFold(n_splits=folds, shuffle=True, random_state=ds.RANDOM_STATE).split(np.arange(len(X_tr))))
        random_cv = rn.cross_validate(bundle, recipe, audit_model, splits=random_splits, config_id="safe_random_kfold", split_label=f"KFold({folds}, shuffle) — ignores time order")
        gap = rn.oriented_gain(rn._mean(random_cv, metric), rn._mean(safe, metric), higher)
        evidence["random_vs_time_cv"] = {"time_ordered": {"metric_mean": rn._mean(safe, metric), "protocol": bundle.cv_description},
                                         "random_kfold": random_cv, "optimism_gap": gap}
        claims.append(_claim(f"{eid}-C4", "risk", f"Random KFold(3) reports {_label(metric)} {rn._ms(random_cv, metric)} for the same safe recipe versus "
                             f"{rn._ms(safe, metric)} with time-ordered CV: random splits look {rn._signed(gap, metric)} better because they see the future.",
                             ["evidence.random_vs_time_cv"]))
        parts.append(f"Random vs time-ordered CV gap: {rn._signed(gap, metric)}.")
    return evidence, claims, " ".join(parts)


def stage_features(p: Prepared, eid: str) -> tuple[dict, list, str]:
    bundle, spec = p.bundle, p.bundle.spec
    metric, higher = spec.primary_metric, spec.higher_is_better
    model = rn._fallback_family("lightgbm") if spec.task_type != "text_tabular_binary" else "logistic_regression"
    tolerance = float(spec.settings.get("fe_tolerance", 0.002))
    relative = bool(spec.settings.get("fe_tolerance_relative", False))
    rows = []
    for name, config in p.recipes.items():
        row = rn.cross_validate(bundle, name, model, config_id=name)
        row["description"] = config["description"]
        rows.append(row)
    selected = rn.select_within_tolerance(rows, metric, higher, tolerance, relative=relative)
    best = rn.rank_rows(rows, metric, higher)[0]
    first = rows[0]
    rule = (f"Choose the recipe with the fewest features (then fastest) whose mean training-CV {_label(metric)} is within "
            + (f"{tolerance:.0%} (relative)" if relative else f"{tolerance}") + f" of the best; all recipes use {model} with default parameters on identical folds ({bundle.cv_description}).")
    evidence = {"stage_results": rows, "stage_model": model, "selected_recipe": selected["recipe"], "selected_recipe_description": selected["description"],
                "selection_rule": rule, "best_recipe": best["recipe"], "selected_metric_mean": rn._mean(selected, metric), "best_metric_mean": rn._mean(best, metric),
                "reference_recipe": first["recipe"], "reference_metric_mean": rn._mean(first, metric),
                "selected_vs_reference_gain": rn.oriented_gain(rn._mean(selected, metric), rn._mean(first, metric), higher), "cv_protocol": bundle.cv_description,
                "category_code_columns": list(spec.categorical_columns), "category_code_rule": p.code_rule,
                "category_code_encoding": "one-hot, levels learned on each fit fold (40 most frequent)" if rn.encoded_codes(spec) else "passed as numbers"}
    table = "; ".join(f"{r['recipe']} ({r['feature_count_mean']:.0f} feats, {rn._ms(r, metric)})" for r in rows)
    claims = [
        _claim(f"{eid}-C1", "decision", f"Selected recipe `{selected['recipe']}` ({selected['feature_count_mean']:.0f} features, {rn._ms(selected, metric)} {_label(metric)}); "
               f"best mean was `{best['recipe']}` at {rn._ms(best, metric)}. Ladder: {table}.",
               ["evidence.stage_results", "evidence.selection_rule", "evidence.selected_recipe"],
               ["Recipe differences within fold noise are not evidence that one representation is better."]),
    ]
    gain = evidence["selected_vs_reference_gain"]
    claims.append(_claim(f"{eid}-C2", "fact", f"Versus the first rung (`{first['recipe']}`), the selected recipe changes {_label(metric)} by {rn._signed(gain, metric)} on identical folds.",
                         ["evidence.selected_vs_reference_gain", "evidence.stage_results"]))
    codes = evidence["category_code_columns"]
    if codes:
        names = ", ".join(f"`{c}`" for c in codes[:12]) + (f" and {len(codes) - 12} more" if len(codes) > 12 else "")
        claims.append(_claim(f"{eid}-C3", "fact", f"Treated as category codes ({p.code_rule}): {names}. "
                             f"Every recipe encodes them as categories ({evidence['category_code_encoding']}) and builds no log or product "
                             "feature from them, because their order and scale are arbitrary (DCLAB-R11).",
                             ["evidence.category_code_columns", "evidence.category_code_rule", "evidence.category_code_encoding"],
                             ["Levels unseen in a fit fold, and levels beyond the 40 most frequent, encode as all zeros.",
                              "A table that stores codes without their labels (as the R&D samples do) yields one-hot columns named by code, not by category."]))
    summary = f"Compared {len(rows)} feature recipes with {model} on {bundle.cv_description}: {table}. Selected `{selected['recipe']}`."
    return evidence, claims, summary


def stage_models(p: Prepared, eid: str, recipe: str) -> tuple[dict, list, str]:
    bundle, spec = p.bundle, p.bundle.spec
    metric, higher = spec.primary_metric, spec.higher_is_better
    rows = [rn.cross_validate(bundle, recipe, family, config_id=family) for family in rn.model_families(bundle.task_type)]
    for row in rows:
        row["adjusted_score"] = rn.adjusted_score(rn._mean(row, metric), rn._std(row, metric), higher)
    ranked = rn.rank_rows(rows, metric, higher)
    selected, runner_up = ranked[0], (ranked[1] if len(ranked) > 1 else ranked[0])
    sign = "-" if higher else "+"
    evidence = {"feature_recipe": recipe, "model_results": rows,
                "ranking": [{"model": r["model"], "mean": rn._mean(r, metric), "std": rn._std(r, metric), "adjusted_score": r["adjusted_score"],
                             "elapsed_seconds": r["elapsed_seconds"], "feature_count_mean": r["feature_count_mean"]} for r in ranked],
                "selected_model": selected["model"],
                "selection_rule": f"Rank by mean training-CV {_label(metric)} {sign} 0.25×fold std ({'higher' if higher else 'lower'} is better), then runtime; identical folds ({bundle.cv_description}); default parameters.",
                "selected_metric_mean": rn._mean(selected, metric), "selected_metric_std": rn._std(selected, metric), "cv_protocol": bundle.cv_description}
    table = "; ".join(f"{r['model']} {rn._ms(r, metric)} ({r['elapsed_seconds']:.1f}s)" for r in ranked)
    spread = abs(rn._mean(selected, metric) - rn._mean(runner_up, metric))
    claims = [
        _claim(f"{eid}-C1", "recommendation", f"{selected['model']} leads on the `{recipe}` recipe ({rn._ms(selected, metric)} {_label(metric)}, adjusted {selected['adjusted_score']:.4f}); "
               f"runner-up {runner_up['model']} at {rn._ms(runner_up, metric)}. Ranking: {table}.",
               ["evidence.model_results", "evidence.ranking", "evidence.selection_rule"],
               ["A cross-validation leader is a candidate, not a universal winner, and has not yet seen the locked holdout."]),
        _claim(f"{eid}-C2", "fact", f"The gap between leader and runner-up is {spread:.4f} against a leader fold std of {rn._std(selected, metric):.4f}"
               + ("; the two are within fold noise." if spread <= rn._std(selected, metric) else "; the lead exceeds one fold std."),
               ["evidence.ranking"]),
    ]
    summary = f"Screened {len(rows)} model families on the `{recipe}` recipe and {bundle.cv_description}: {table}."
    return evidence, claims, summary


def stage_final(p: Prepared, eid: str, recipe: str, family: str) -> tuple[dict, list, str]:
    bundle, spec, task = p.bundle, p.bundle.spec, p.bundle.task_type
    metric, higher = spec.primary_metric, spec.higher_is_better
    n_classes = rn._n_classes(bundle)
    baseline = rn.cross_validate(bundle, recipe, family, config_id=f"{family}_baseline")
    candidates = rn.optimization_candidates(family, task)
    tuned = [rn.cross_validate(bundle, recipe, family, c["params"], config_id=c["candidate_id"]) for c in candidates]
    margin = float(spec.settings.get("tuning_margin", 0.001))
    relative = bool(spec.settings.get("tuning_margin_relative", False))
    decision = rn.tuning_decision(baseline, tuned, metric, higher, margin, relative=relative)
    selected = decision.pop("selected")
    decision["selected_config"] = selected["config_id"]
    started = time.perf_counter()
    final = rn.fit_and_predict(bundle, recipe, family, selected["model_params"] or None, bundle.X_train, bundle.y_train, bundle.X_test)
    final_seconds = time.perf_counter() - started
    predictions, y_test = final["predictions"], bundle.y_test
    holdout = rn.compute_metrics(task, y_test, predictions, n_classes=n_classes, detail=True, precision_target=float(spec.settings.get("precision_target", 0.9)))
    holdout["fit_seconds"] = float(final_seconds)
    if task == "multiclass" and p.class_labels:
        for item in holdout.get("worst_3_classes", []):
            item["source_label"] = p.class_labels[item["class_index"]]
    metric_fn = rn.primary_metric_fn(task, metric, n_classes)
    ci = rn.bootstrap_ci(y_test.to_numpy(), predictions, metric_fn, groups=bundle.groups("test") if spec.split_strategy == "group" else None)
    baselines = _baselines(p, predictions)
    importance = rn._importance(final["model"], final["names"])
    top_features = [{"feature": k, "normalized_importance": v} for k, v in sorted(importance.items(), key=lambda kv: -kv[1])[:20]]
    cv_mean, holdout_value = rn._mean(selected, metric), holdout[metric]
    evidence: dict[str, Any] = {
        "feature_recipe": recipe, "model": family, "configuration_results": [baseline, *tuned], "optimization_search_space": candidates,
        "tuning_decision": decision, "selected_optimization": selected["config_id"], "selected_model_params": selected["model_params"],
        "selection_rule": (f"Evaluate {len(candidates)} explicit parameter candidates on training CV ({bundle.cv_description}); take the best by mean "
                           f"{'-' if higher else '+'} 0.25×std and accept it only if its mean {_label(metric)} beats the default by at least "
                           + (f"{margin:.0%} (relative)" if relative else f"{margin}") + ", otherwise keep the default configuration."),
        "holdout_consumed": True, "holdout_split": bundle.holdout_description, "holdout_rows": int(len(y_test)), "train_rows": int(len(bundle.y_train)),
        "holdout_metrics": holdout, "holdout_primary_metric_ci": ci, "cv_selected_metric_mean": cv_mean,
        "cv_to_holdout_gap": rn.oriented_gain(holdout_value, cv_mean, higher), "baselines": baselines, "top_final_features": top_features,
        "sampling": bundle.sampling, "production_approved": False,
        "production_feature_rule": "Promote a feature only if it is decision-time available, stable across folds, obtainable in production and monitored for drift.",
    }
    if task in ("binary_imbalanced", "text_tabular_binary"):
        rows = []
        y_arr = y_test.to_numpy()
        for t in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
            flags = predictions >= t
            tp = int((flags & (y_arr == 1)).sum())
            rows.append({"threshold": t, "alerts": int(flags.sum()), "true_positives": tp,
                         "precision": float(tp / flags.sum()) if flags.sum() else None, "recall": float(tp / max(int(y_arr.sum()), 1))})
        evidence["threshold_analysis"] = rows
        bins = []
        for lower in np.arange(0, 1, 0.2):
            mask = (predictions >= lower) & ((predictions < lower + 0.2) if lower < 0.8 else (predictions <= 1))
            if mask.any():
                bins.append({"lower": float(lower), "upper": float(lower + 0.2), "count": int(mask.sum()), "predicted": float(predictions[mask].mean()), "observed": float(y_arr[mask].mean())})
        evidence["calibration_bins"] = bins
    label = _label(metric)
    claims = [
        _claim(f"{eid}-C1", "fact", f"{family}/{selected['config_id']} on the `{recipe}` recipe scored {label} {rn._f(holdout_value)} on the once-consumed holdout "
               f"({bundle.holdout_description}; {ci['method']} 95% interval {rn._f(ci['low'])}–{rn._f(ci['high'])}); training-CV mean was {rn._f(cv_mean)}.",
               ["evidence.holdout_metrics", "evidence.holdout_primary_metric_ci", "evidence.holdout_consumed", "evidence.cv_selected_metric_mean"],
               ["A single holdout is one draw; CV models see fewer rows than the final model, so the holdout can legitimately differ from the CV mean."]),
        _claim(f"{eid}-C2", "decision", f"Tuning {'accepted' if decision['accepted'] else 'rejected'}: best candidate {decision['best_candidate']} changed mean training-CV {label} by "
               f"{decision['gain']:+.4f} (required ≥ {decision['required_gain']:.4f}), so {selected['config_id']} was used.",
               ["evidence.tuning_decision", "evidence.configuration_results", "evidence.selection_rule"],
               ["Only a handful of explicit candidates were tried; this is a sanity check, not an exhaustive search."]),
    ]
    base_text = "; ".join(f"{k}: " + ", ".join(f"{m} {rn._f(v)}" for m, v in b.items() if isinstance(v, (int, float))) for k, b in baselines.items())
    claims.append(_claim(f"{eid}-C3", "fact", f"Trivial baselines on the same holdout: {base_text}.", ["evidence.baselines"]))
    summary = (f"Tried {len(candidates)} explicit {family} parameter candidates on {bundle.cv_description} (tuning {'accepted' if decision['accepted'] else 'rejected'}), "
               f"fitted {selected['config_id']} on all {len(bundle.y_train)} training rows and scored the sealed holdout once: {label} {rn._f(holdout_value)} "
               f"({rn._f(ci['low'])}–{rn._f(ci['high'])}).")
    return evidence, claims, summary


def _baselines(p: Prepared, predictions: np.ndarray) -> dict[str, Any]:
    bundle, task = p.bundle, p.bundle.task_type
    y_test, y_train = bundle.y_test, bundle.y_train
    if task in ("binary_imbalanced", "text_tabular_binary"):
        rate = float(y_test.mean())
        return {"majority_class": {"accuracy": float(max(rate, 1 - rate)), "roc_auc": 0.5, "average_precision_random_ranking": rate}}
    if task == "multiclass":
        from sklearn.metrics import f1_score

        majority = int(y_train.value_counts().idxmax())
        n_classes = rn._n_classes(bundle)
        return {"majority_class": {"macro_f1": float(f1_score(y_test, np.full(len(y_test), majority), average="macro", labels=list(range(n_classes)), zero_division=0)),
                                   "accuracy": float((y_test == majority).mean())}}
    out = {}
    for name, value in (("training_mean", float(y_train.mean())), ("training_median", float(y_train.median()))):
        out[name] = rn.regression_metrics(y_test, np.full(len(y_test), value))
    return out


# ---------------------------------------------------------------------- orchestration


def chosen(record: dict[str, Any] | None, fallback_key: str) -> str | None:
    if not record:
        return None
    decision = record.get("decision") or {}
    return decision.get("chosen") or record["evidence"].get(fallback_key)


def execute(store: ProjectStore, project_id: str, stage: str, actor: str = "human", reuse_reason: str | None = None) -> dict[str, Any]:
    """Run one stage now (blocking) and write its record.

    The move is validated by the workflow graph first and logged either way; a move the
    graph does not allow raises ``graph.GraphBlocked`` with the verdict.
    """
    project = store.get(project_id)
    verdict = graph.check(project, "run_stage", actor, stage=stage, reuse_reason=reuse_reason)
    if not verdict.allowed:
        graph.log(store, project_id, verdict, project)
        raise graph.GraphBlocked(verdict)
    state_before = copy.deepcopy(project)  # the state the move was proposed in, for the log
    index = STAGE_KEYS.index(stage)
    later = [s for s in STAGE_KEYS[index + 1:] if project["stages"][s].get("status") in ("completed", "approved")]
    if stage != "final" and project["stages"][stage].get("status") in ("completed", "approved") and later:
        store.clear_stages(project_id, STAGE_KEYS[index + 1])  # a rerun makes every later result stale
        project = store.get(project_id)
    if stage == "final" and actor == "agent":
        graph.consume_holdout_approval(project)
    try:
        record = _execute(store, project_id, stage, project, reuse_reason)
    except Exception as exc:
        graph.log(store, project_id, verdict, state_before, outcome=f"failed: {type(exc).__name__}: {str(exc)[:200]}")
        raise
    graph.log(store, project_id, verdict, state_before, outcome="done: " + record["setup_summary"][:200])
    return record


def _execute(store: ProjectStore, project_id: str, stage: str, project: dict[str, Any], reuse_reason: str | None) -> dict[str, Any]:
    p = prepare(store, project)
    meta = STAGE_BY_KEY[stage]
    eid = f"PRJ-{project_id[:6]}-{stage}"
    project["stages"][stage] = {"status": "running", "started": _now()}
    project["running"] = stage
    store.save(project)
    store.log(project_id, "stage_started", {"stage": stage})
    clock = time.perf_counter()
    try:
        decision: dict[str, Any] | None = None
        if stage == "data":
            evidence, claims, summary = stage_data(p, eid)
        elif stage == "leakage":
            evidence, claims, summary = stage_leakage(p, eid)
        elif stage == "features":
            evidence, claims, summary = stage_features(p, eid)
            decision = {"kind": "recipe", "selected": evidence["selected_recipe"], "chosen": evidence["selected_recipe"], "rule": evidence["selection_rule"],
                        "options": [{"id": r["recipe"], "label": r["recipe"], "description": r["description"], "features": r["feature_count_mean"],
                                     "mean": rn._mean(r, p.metric), "std": rn._std(r, p.metric), "elapsed_seconds": r["elapsed_seconds"]} for r in evidence["stage_results"]]}
        elif stage == "models":
            recipe = chosen(store.read_stage(project_id, "features"), "selected_recipe")
            evidence, claims, summary = stage_models(p, eid, recipe)
            decision = {"kind": "model", "selected": evidence["selected_model"], "chosen": evidence["selected_model"], "rule": evidence["selection_rule"],
                        "options": [{"id": r["model"], "label": r["model"], "mean": r["mean"], "std": r["std"], "adjusted": r["adjusted_score"],
                                     "elapsed_seconds": r["elapsed_seconds"], "features": r["feature_count_mean"]} for r in evidence["ranking"]]}
        else:
            recipe = chosen(store.read_stage(project_id, "features"), "selected_recipe")
            family = chosen(store.read_stage(project_id, "models"), "selected_model")
            project["holdout_uses"] = int(project.get("holdout_uses", 0)) + 1
            evidence, claims, summary = stage_final(p, eid, recipe, family)
            evidence["holdout_uses_in_this_project"] = project["holdout_uses"]
            if reuse_reason:
                evidence["holdout_reuse_reason"] = str(reuse_reason)[:500]
            decision = {"kind": "final", "selected": evidence["selected_optimization"], "chosen": evidence["selected_optimization"], "rule": evidence["selection_rule"], "options": []}
        elapsed = time.perf_counter() - clock
        evidence["elapsed_seconds"] = float(elapsed)
        record = {
            "stage": stage, "kind": meta["kind"], "title": meta["title"], "workflow": meta["workflow"], "question": meta["question"],
            "experiment_id": eid, "status": "completed", "started_at": project["stages"][stage]["started"], "completed_at": _now(),
            "elapsed_seconds": float(elapsed), "task": p.task, "task_type": p.bundle.task_type, "primary_metric": p.metric,
            "decision_time_rule": p.solution.prediction_moment, "holdout_policy": p.bundle.spec.holdout_policy(),
            "setup_summary": summary, "evidence": evidence, "claims": claims, "decision": decision,
            "provenance": {"data_sha256": project["data"]["sha256"], "filename": project["data"]["filename"], "sampling": p.sampling,
                           "cv": p.bundle.cv_description, "holdout": p.bundle.holdout_description, "random_state": ds.RANDOM_STATE},
        }
        record = rn._jsonable(record)
        record["notes"] = agent.narrate(stage, record, p.solution.model_dump(), p.bundle.task_type, project_id)
        store.write_stage(project_id, stage, record)
        project = store.get(project_id)
        project["stages"][stage] = {"status": "completed", "started": record["started_at"], "finished": record["completed_at"], "elapsed_seconds": float(elapsed)}
        if stage == "final":
            project["holdout_uses"] = evidence["holdout_uses_in_this_project"]
        project["running"] = None
        store.save(project)
        store.log(project_id, "stage_completed", {"stage": stage, "elapsed_seconds": round(elapsed, 1), "summary": summary[:300]})
        return record
    except Exception as exc:  # noqa: BLE001 — the failure itself is the record
        project = store.get(project_id)
        project["stages"][stage] = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:600]}", "finished": _now()}
        project["running"] = None
        store.save(project)
        store.log(project_id, "stage_failed", {"stage": stage, "error": project["stages"][stage]["error"]})
        raise


def run_all(store: ProjectStore, project_id: str, start: str = "data") -> list[str]:
    """Run every remaining stage in order; stops at the first failure."""
    done = []
    for stage in STAGE_KEYS[STAGE_KEYS.index(start):]:
        execute(store, project_id, stage)
        done.append(stage)
    return done


def approve(store: ProjectStore, project_id: str, stage: str, choice: str | None = None, actor: str = "human") -> dict[str, Any]:
    """Sign-off on a stage, optionally overriding the deterministic choice; downstream results are cleared."""
    project = store.get(project_id)
    verdict = graph.check(project, "approve_stage", actor, stage=stage, choice=choice)
    graph.log(store, project_id, verdict, project)
    if not verdict.allowed:
        raise graph.GraphBlocked(verdict)
    record = store.read_stage(project_id, stage)
    if not record or project["stages"][stage].get("status") not in ("completed", "approved"):
        raise ValueError("Only a completed stage can be approved")
    if choice:
        options = {o["id"] for o in (record.get("decision") or {}).get("options", [])}
        if choice not in options:
            raise ValueError("Unknown option")
        record["decision"]["chosen"] = choice
        record["decision"]["overridden"] = choice != record["decision"]["selected"]
        store.write_stage(project_id, stage, record)
        project["decisions"][stage] = choice
    index = STAGE_KEYS.index(stage)
    for later in STAGE_KEYS[index + 1:]:
        if project["stages"][later].get("status") != "pending":
            store.save(project)
            store.clear_stages(project_id, later)
            project = store.get(project_id)
            break
    project["stages"][stage]["status"] = "approved"
    project["stages"][stage]["approved_at"] = _now()
    store.save(project)
    store.log(project_id, "stage_approved", {"stage": stage, "choice": choice})
    return project


def capabilities() -> dict[str, Any]:
    return {
        "tasks": ["binary", "multiclass", "regression"],
        "text_columns": "binary targets only",
        "split_strategies": ["time (declared time column)", "group (declared group column)", "stratified random"],
        "model_families": {t: rn.model_families(tt) for t, tt in TASK_TYPE.items()},
        "lightgbm": rn._have("lightgbm"), "xgboost": rn._have("xgboost"),
        "llm": agent.llm_available(),
    }
