"""HyperAck-style experiment ladder for external dataset projects."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hyperack_exp"))

from shared.protocol import evaluate  # noqa: E402
from general_pipeline.external_catalog import DATASET_CATALOG  # noqa: E402
from general_pipeline.external_project import PROJECTS_ROOT, project_dir, scaffold_project  # noqa: E402
from general_pipeline.models import SoftVotingClassifier, get_model  # noqa: E402
from general_pipeline.playbook.features import build_feature_matrix  # noqa: E402
from general_pipeline.playbook.policy import get_policy  # noqa: E402

RANDOM_STATE = 42
EXTERNAL_DATA = ROOT / "external_data"


def _serialize(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items() if k not in {"y_true", "y_pred", "y_prob"}}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def load_raw_xy(key: str, mode: str, max_rows: int | None = 20000):
    policy = get_policy(key)
    ds = EXTERNAL_DATA / key
    X = pd.read_parquet(ds / "X.parquet")
    y = pd.read_parquet(ds / "y.parquet")["target"].astype(int)
    meta = json.loads((ds / "meta.json").read_text())

    drop = policy.unsafe_only_features if mode == "safe" else []
    if drop:
        X = X.drop(columns=[c for c in drop if c in X.columns], errors="ignore")
    meta["mode"] = mode
    meta["dropped_leakage"] = drop
    meta["has_leakage"] = policy.has_leakage
    meta["leakage_rationale"] = policy.rationale

    if max_rows and len(X) > max_rows:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=RANDOM_STATE)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return (
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
        meta,
    )


def _pipe(model, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _save(key: str, payload: dict) -> Path:
    results = project_dir(key) / "results" / "ladder"
    results.mkdir(parents=True, exist_ok=True)
    name = f"{payload['exp_id']:02d}_{payload['mode']}_{payload['exp_name']}.json"
    path = results / name
    path.write_text(json.dumps(_serialize(payload), indent=2) + "\n")
    return path


def _run_eval(model, Xtr, ytr, Xte, yte) -> Tuple[dict, float]:
    t0 = time.perf_counter()
    metrics = evaluate(model, Xtr, ytr, Xte, yte)
    return metrics, time.perf_counter() - t0


def run_ladder_experiment(
    key: str,
    exp_id: int,
    exp_name: str,
    mode: str,
    stage: str,
    model_factory,
    *,
    notes: str = "",
    optimization_method: str = "none",
) -> dict:
    X_train, y_train, X_test, y_test, meta = load_raw_xy(key, mode)
    Xtr, Xte, fe_meta = build_feature_matrix(X_train, y_train, X_test, stage=stage)
    model = model_factory()
    metrics, elapsed = _run_eval(model, Xtr, y_train, Xte, y_test)
    payload = {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "dataset": key,
        "mode": mode,
        "fe_stage": stage,
        "optimization_method": optimization_method,
        "notes": notes,
        "feature_count": int(Xtr.shape[1]),
        "train_rows": int(len(Xtr)),
        "test_rows": int(len(Xte)),
        "pos_rate": float(y_train.mean()),
        "fe_meta": fe_meta,
        "dropped_leakage": meta.get("dropped_leakage", []),
        "has_leakage": meta.get("has_leakage", False),
        "metrics": _serialize(metrics),
        "total_elapsed_seconds": round(elapsed, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = _save(key, payload)
    m = metrics
    print(
        f"[{key}] {exp_id:02d}/{mode}/{exp_name:<28} stage={stage:<12} "
        f"feats={Xtr.shape[1]:3d} AUC={m['roc_auc']:.4f} F1={m['f1']:.4f} ({elapsed:.1f}s) -> {path.name}",
        flush=True,
    )
    return payload


def _tuned_lgbm():
    base = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
        ]
    )
    dist = {
        "model__n_estimators": [200, 300, 450, 600],
        "model__learning_rate": [0.02, 0.03, 0.05, 0.08],
        "model__num_leaves": [16, 24, 31, 48],
        "model__min_child_samples": [20, 40, 60, 80],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.9],
        "model__reg_lambda": [0.1, 0.5, 1.0, 2.0],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return RandomizedSearchCV(
        base, dist, n_iter=16, scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=-1, refit=True
    )


def _tuned_xgb():
    base = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
                ),
            ),
        ]
    )
    dist = {
        "model__n_estimators": [200, 400, 600, 800],
        "model__learning_rate": [0.02, 0.03, 0.05],
        "model__max_depth": [4, 5, 6, 7],
        "model__min_child_weight": [1, 3, 5],
        "model__subsample": [0.75, 0.85, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.85],
        "model__reg_lambda": [1.0, 2.0, 4.0],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return RandomizedSearchCV(
        base, dist, n_iter=12, scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=-1, refit=True
    )


def ladder_specs(mode: str) -> List[dict]:
    """HyperAck-inspired ordered experiments for one mode."""
    return [
        dict(
            exp_id=1,
            exp_name="baseline_logistic",
            stage="raw",
            optimization_method="none",
            notes="Floor: logistic regression on raw features.",
            model_factory=lambda: _pipe(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), scale=True),
        ),
        dict(
            exp_id=2,
            exp_name="baseline_lightgbm",
            stage="raw",
            optimization_method="none",
            notes="Strong GBDT baseline on raw features (HyperAck exp 02).",
            model_factory=lambda: _pipe(LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
        ),
        dict(
            exp_id=3,
            exp_name="fe_log_transforms",
            stage="logs",
            optimization_method="none",
            notes="Add log1p transforms for skewed non-negative columns.",
            model_factory=lambda: _pipe(LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
        ),
        dict(
            exp_id=4,
            exp_name="fe_ratios",
            stage="ratios",
            optimization_method="none",
            notes="Add pairwise intensity ratios among high-variance features.",
            model_factory=lambda: _pipe(LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
        ),
        dict(
            exp_id=5,
            exp_name="fe_interactions",
            stage="interactions",
            optimization_method="none",
            notes="Multiplicative interactions among top-MI features.",
            model_factory=lambda: _pipe(LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
        ),
        dict(
            exp_id=6,
            exp_name="fe_full_clusters_bins",
            stage="full_fe",
            optimization_method="none",
            notes="Full FE: logs+ratios+interactions+KMeans clusters+quantile bins (train-fit).",
            model_factory=lambda: _pipe(LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
        ),
        dict(
            exp_id=7,
            exp_name="feature_selection_mi",
            stage="selected",
            optimization_method="mutual_info_selection",
            notes="Keep top-MI subset of full FE matrix (HyperAck exp 07).",
            model_factory=lambda: _pipe(LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
        ),
        dict(
            exp_id=8,
            exp_name="tuned_xgboost",
            stage="full_fe",
            optimization_method="RandomizedSearchCV_roc_auc",
            notes="RandomizedSearchCV on XGBoost over full FE (HyperAck exp 08).",
            model_factory=_tuned_xgb,
        ),
        dict(
            exp_id=9,
            exp_name="tuned_lightgbm",
            stage="full_fe",
            optimization_method="RandomizedSearchCV_roc_auc",
            notes="RandomizedSearchCV on LightGBM over full FE (HyperAck exp 09).",
            model_factory=_tuned_lgbm,
        ),
        dict(
            exp_id=10,
            exp_name="hist_gradient_boosting",
            stage="full_fe",
            optimization_method="none",
            notes="Sklearn HistGB on full FE (HyperAck exp 10).",
            model_factory=lambda: _pipe(HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ),
        dict(
            exp_id=11,
            exp_name="extra_trees_strong",
            stage="full_fe",
            optimization_method="hand_tuned",
            notes="Strong ExtraTrees (n=600, depth=14) on full FE.",
            model_factory=lambda: _pipe(
                ExtraTreesClassifier(
                    n_estimators=600,
                    max_depth=14,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ),
        dict(
            exp_id=12,
            exp_name="calibrated_lgbm_threshold",
            stage="full_fe",
            optimization_method="isotonic_calibration",
            notes="Isotonic-calibrated LightGBM; threshold kept at 0.5 for ranking metrics.",
            model_factory=lambda: _pipe(
                CalibratedClassifierCV(
                    LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=-1),
                    method="isotonic",
                    cv=3,
                )
            ),
        ),
        dict(
            exp_id=13,
            exp_name="stacking_ensemble",
            stage="full_fe",
            optimization_method="oof_stacking",
            notes="ET+LGBM+XGB stacked with logistic meta-learner (HyperAck exp 13).",
            model_factory=lambda: get_model("stacking_ensemble", optimization="optimized", mode="safe"),
        ),
        dict(
            exp_id=14,
            exp_name="soft_vote_blend",
            stage="full_fe",
            optimization_method="soft_probability_blend",
            notes="Soft-vote blend ET+LGBM+XGB (HyperAck exp 15 / optimized safe champion style).",
            model_factory=lambda: get_model("soft_vote_blend", optimization="optimized", mode="safe"),
        ),
        dict(
            exp_id=15,
            exp_name="leakage_honesty_check",
            stage="full_fe",
            optimization_method="none",
            notes=(
                "Honesty marker: LightGBM on full FE under this mode. "
                "Compare safe vs unsafe of this id to measure leakage cost."
            ),
            model_factory=lambda: _pipe(
                LGBMClassifier(n_estimators=313, learning_rate=0.02, num_leaves=24, random_state=RANDOM_STATE, verbosity=-1)
            ),
        ),
    ]


def run_project_ladder(key: str, modes: Optional[List[str]] = None) -> pd.DataFrame:
    scaffold_project(key)
    policy = get_policy(key)
    if modes is None:
        modes = ["safe", "unsafe"] if policy.has_leakage else ["safe"]
        # Always also run unsafe when leakage exists; when not, run unsafe once as identical matrix marker
        if not policy.has_leakage:
            modes = ["safe", "unsafe"]  # identical features; documents equivalence

    rows = []
    for mode in modes:
        for spec in ladder_specs(mode):
            try:
                payload = run_ladder_experiment(
                    key,
                    spec["exp_id"],
                    spec["exp_name"],
                    mode,
                    spec["stage"],
                    spec["model_factory"],
                    notes=spec["notes"],
                    optimization_method=spec["optimization_method"],
                )
                rows.append(payload)
            except Exception as exc:
                print(f"FAILED {key}/{mode}/{spec['exp_name']}: {exc}", flush=True)
    return results_frame(key)


def results_frame(key: str) -> pd.DataFrame:
    results = project_dir(key) / "results" / "ladder"
    rows = []
    if not results.exists():
        return pd.DataFrame()
    for path in sorted(results.glob("*.json")):
        d = json.loads(path.read_text())
        m = d.get("metrics", {})
        rows.append(
            {
                "exp_id": d.get("exp_id"),
                "exp_name": d.get("exp_name"),
                "mode": d.get("mode"),
                "fe_stage": d.get("fe_stage"),
                "optimization_method": d.get("optimization_method"),
                "feature_count": d.get("feature_count"),
                "roc_auc": m.get("roc_auc"),
                "accuracy": m.get("accuracy"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "avg_precision": m.get("avg_precision"),
                "elapsed_s": d.get("total_elapsed_seconds"),
                "notes": d.get("notes"),
                "has_leakage": d.get("has_leakage"),
                "dropped_leakage": ",".join(d.get("dropped_leakage") or []),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        out = project_dir(key) / "benchmark_outputs" / "ladder_results.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values(["mode", "exp_id"]).to_csv(out, index=False)
    return df


def run_all_ladders(keys: Optional[List[str]] = None) -> None:
    keys = keys or [s.key for s in DATASET_CATALOG]
    for key in keys:
        print("\n" + "#" * 72)
        print(f"LADDER: {key}")
        print("#" * 72)
        run_project_ladder(key)
