"""Telco churn classification with tabular Transformers.

Pipeline:
    CSV -> pandas -> PyTorch Tabular -> FT-Transformer / TabTransformer
                                    -> SAINT (custom; not in PyTorch Tabular)
                                    -> churn probability

Run (conda env llm10):
    /opt/miniconda3/envs/llm10/bin/python tabular_transformers/tabular_transformer_churn.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib
from saint_tabular import SaintConfig, predict_proba as saint_predict_proba, train_saint

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DEFAULT_CSV = REPO / "data" / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
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
]
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COL = "Churn"
ID_COL = "customerID"
FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    data = data.apply(lambda s: s.str.strip() if s.dtype == "object" else s)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    # Blank TotalCharges are new customers (tenure == 0); 0 is the correct value.
    data["TotalCharges"] = data["TotalCharges"].fillna(0.0)
    data[NUMERICAL_COLS] = data[NUMERICAL_COLS].astype("float64")
    data["SeniorCitizen"] = data["SeniorCitizen"].astype(str)
    for col in CATEGORICAL_COLS:
        data[col] = data[col].astype(str)
    data[TARGET_COL] = data[TARGET_COL].astype(str)
    return data


def split_frames(
    data: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdout = test_size + val_size
    train, temp = train_test_split(
        data, test_size=holdout, stratify=data[TARGET_COL], random_state=seed
    )
    relative_test = test_size / holdout
    val, test = train_test_split(
        temp, test_size=relative_test, stratify=temp[TARGET_COL], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _binary_labels(series: pd.Series) -> np.ndarray:
    return (series.astype(str) == "Yes").to_numpy(dtype=np.int64)


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_yes": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_yes": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_yes": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    if thresholds.size == 0:
        return 0.5
    return float(thresholds[int(np.nanargmax(f1[:-1]))])


def _accelerator() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


def _churn_probability_from_pred_df(pred_df: pd.DataFrame) -> np.ndarray:
    for col in ("Churn_Yes_probability", "Churn_1_probability"):
        if col in pred_df.columns:
            return pred_df[col].to_numpy(dtype=np.float64)
    prob_cols = [c for c in pred_df.columns if c.endswith("_probability")]
    yes_cols = [c for c in prob_cols if "Yes" in c or c.endswith("_1_probability")]
    if yes_cols:
        return pred_df[yes_cols[0]].to_numpy(dtype=np.float64)
    raise KeyError(f"Could not find churn probability column in {list(pred_df.columns)}")


def sklearn_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    numeric_steps: List[Tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), NUMERICAL_COLS),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )


def _tree_feature_importance(preprocessor: ColumnTransformer, importances: np.ndarray) -> pd.DataFrame:
    names = preprocessor.get_feature_names_out()
    frame = pd.DataFrame({"feature": names, "importance": importances})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def train_logreg(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Tuple[np.ndarray, Dict[str, float], object]:
    pipe = Pipeline(
        [
            ("preprocessor", sklearn_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    pipe.fit(train[FEATURE_COLS], _binary_labels(train[TARGET_COL]))
    val_prob = pipe.predict_proba(val[FEATURE_COLS])[:, 1]
    test_prob = pipe.predict_proba(test[FEATURE_COLS])[:, 1]
    threshold = best_f1_threshold(_binary_labels(val[TARGET_COL]), val_prob)
    metrics = evaluate_probs(_binary_labels(test[TARGET_COL]), test_prob, threshold=threshold)
    metrics["model"] = "Logistic Regression"
    metrics["library"] = "scikit-learn"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "logistic_regression.joblib")
    metrics["saved_to"] = str(MODEL_DIR / "logistic_regression.joblib")
    return test_prob, metrics, pipe


def train_random_forest(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, seed: int = 42
) -> Tuple[np.ndarray, Dict[str, float], object]:
    preprocessor = sklearn_preprocessor(scale_numeric=False)
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        min_samples_split=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(train[FEATURE_COLS], _binary_labels(train[TARGET_COL]))
    val_prob = pipe.predict_proba(val[FEATURE_COLS])[:, 1]
    test_prob = pipe.predict_proba(test[FEATURE_COLS])[:, 1]
    threshold = best_f1_threshold(_binary_labels(val[TARGET_COL]), val_prob)
    metrics = evaluate_probs(_binary_labels(test[TARGET_COL]), test_prob, threshold=threshold)
    metrics["model"] = "Random Forest"
    metrics["library"] = "scikit-learn"
    importance = _tree_feature_importance(pipe.named_steps["preprocessor"], pipe.named_steps["classifier"].feature_importances_)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(OUTPUT_DIR / "random_forest_feature_importance.csv", index=False)
    metrics["top_feature"] = str(importance.iloc[0]["feature"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "random_forest.joblib")
    metrics["saved_to"] = str(MODEL_DIR / "random_forest.joblib")
    return test_prob, metrics, pipe


def train_xgboost(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, seed: int = 42
) -> Tuple[np.ndarray, Dict[str, float], object]:
    from xgboost import XGBClassifier

    preprocessor = sklearn_preprocessor(scale_numeric=False)
    y_train = _binary_labels(train[TARGET_COL])
    y_val = _binary_labels(val[TARGET_COL])
    x_train = preprocessor.fit_transform(train[FEATURE_COLS])
    x_val = preprocessor.transform(val[FEATURE_COLS])
    x_test = preprocessor.transform(test[FEATURE_COLS])
    n_neg = max(int((y_train == 0).sum()), 1)
    n_pos = max(int((y_train == 1).sum()), 1)
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=n_neg / n_pos,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        early_stopping_rounds=40,
    )
    clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    val_prob = clf.predict_proba(x_val)[:, 1]
    test_prob = clf.predict_proba(x_test)[:, 1]
    threshold = best_f1_threshold(y_val, val_prob)
    metrics = evaluate_probs(_binary_labels(test[TARGET_COL]), test_prob, threshold=threshold)
    metrics["model"] = "XGBoost"
    metrics["library"] = "xgboost"
    importance = _tree_feature_importance(preprocessor, clf.feature_importances_)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(OUTPUT_DIR / "xgboost_feature_importance.csv", index=False)
    metrics["top_feature"] = str(importance.iloc[0]["feature"])
    bundle = {"preprocessor": preprocessor, "classifier": clf}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / "xgboost.joblib")
    metrics["saved_to"] = str(MODEL_DIR / "xgboost.joblib")
    return test_prob, metrics, bundle


def _tabular_configs(model_name: str, epochs: int, batch_size: int, seed: int, ckpt_dir: Path):
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import FTTransformerConfig, TabTransformerConfig
    from pytorch_tabular.models.common.heads import LinearHeadConfig

    data_config = DataConfig(
        target=[TARGET_COL],
        continuous_cols=NUMERICAL_COLS,
        categorical_cols=CATEGORICAL_COLS,
        normalize_continuous_features=True,
        num_workers=0,
        pin_memory=False,
    )
    trainer_config = TrainerConfig(
        batch_size=batch_size,
        max_epochs=epochs,
        min_epochs=1,
        early_stopping="valid_loss",
        early_stopping_mode="min",
        early_stopping_patience=5,
        checkpoints="valid_loss",
        checkpoints_path=str(ckpt_dir),
        checkpoints_mode="min",
        load_best=True,
        accelerator=_accelerator(),
        devices=1,
        auto_lr_find=False,
    )
    optimizer_config = OptimizerConfig()
    common_metrics = dict(
        task="classification",
        learning_rate=1e-3,
        metrics=["accuracy", "auroc"],
        metrics_prob_input=[False, True],
        metrics_params=[{}, {"num_classes": 2}],
        seed=seed,
        embedding_dropout=0.1,
    )
    if model_name == "ft_transformer":
        head = LinearHeadConfig(layers="", dropout=0.1).__dict__
        model_config = FTTransformerConfig(
            input_embed_dim=32,
            num_heads=4,
            num_attn_blocks=3,
            attn_dropout=0.1,
            ff_dropout=0.1,
            add_norm_dropout=0.1,
            transformer_activation="GEGLU",
            attn_feature_importance=True,
            head="LinearHead",
            head_config=head,
            **common_metrics,
        )
    elif model_name == "tab_transformer":
        head = LinearHeadConfig(layers="64", dropout=0.1, use_batch_norm=True).__dict__
        model_config = TabTransformerConfig(
            input_embed_dim=32,
            num_heads=4,
            num_attn_blocks=4,
            attn_dropout=0.1,
            ff_dropout=0.1,
            add_norm_dropout=0.1,
            transformer_activation="GEGLU",
            head="LinearHead",
            head_config=head,
            **common_metrics,
        )
    else:
        raise ValueError(model_name)
    return data_config, trainer_config, optimizer_config, model_config


def train_pytorch_tabular(
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float], object]:
    from pytorch_tabular import TabularModel

    save_dir = MODEL_DIR / model_name
    ckpt_dir = MODEL_DIR / f"{model_name}_ckpt"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_config, trainer_config, optimizer_config, model_config = _tabular_configs(
        model_name, epochs, batch_size, seed, ckpt_dir
    )
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        verbose=True,
        suppress_lightning_logger=True,
    )

    n_no = int((train[TARGET_COL] == "No").sum())
    n_yes = int((train[TARGET_COL] == "Yes").sum())
    # LabelEncoder sorts alphabetically: No=0, Yes=1
    loss_weight = torch.tensor([1.0, n_no / max(n_yes, 1)], dtype=torch.float32)
    loss = nn.CrossEntropyLoss(weight=loss_weight)

    fit_cols = CATEGORICAL_COLS + NUMERICAL_COLS + [TARGET_COL]
    tabular_model.fit(
        train=train[fit_cols],
        validation=val[fit_cols],
        loss=loss,
        seed=seed,
    )

    val_pred = tabular_model.predict(val[fit_cols])
    test_pred = tabular_model.predict(test[fit_cols])
    val_prob = _churn_probability_from_pred_df(val_pred)
    test_prob = _churn_probability_from_pred_df(test_pred)
    threshold = best_f1_threshold(_binary_labels(val[TARGET_COL]), val_prob)
    metrics = evaluate_probs(_binary_labels(test[TARGET_COL]), test_prob, threshold=threshold)
    pretty = {
        "ft_transformer": "FT-Transformer",
        "tab_transformer": "TabTransformer",
    }[model_name]
    metrics["model"] = pretty
    metrics["library"] = "pytorch-tabular"
    try:
        tabular_model.save_model(str(save_dir))
        metrics["saved_to"] = str(save_dir)
    except Exception as exc:  # noqa: BLE001
        metrics["save_error"] = str(exc)
    try:
        fi = tabular_model.feature_importance()
        if fi is not None and len(fi):
            fi_path = OUTPUT_DIR / f"{model_name}_feature_importance.csv"
            fi.sort_values("importance", ascending=False).to_csv(fi_path, index=False)
            metrics["top_feature"] = str(fi.sort_values("importance", ascending=False).iloc[0]["Features"])
    except Exception:
        pass
    return test_prob, metrics, tabular_model


def train_saint_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float], object]:
    ckpt = MODEL_DIR / "saint.pt"
    config = SaintConfig(
        dim=32,
        depth=3,
        num_heads=4,
        dropout=0.1,
        lr=1e-3,
        batch_size=batch_size,
        max_epochs=epochs,
        patience=5,
        seed=seed,
        use_intersample=True,
    )
    model, preprocessor, history = train_saint(
        train_frame=train,
        val_frame=val,
        categorical_cols=CATEGORICAL_COLS,
        numerical_cols=NUMERICAL_COLS,
        target_col=TARGET_COL,
        config=config,
        checkpoint_path=ckpt,
    )
    x_cat_va, x_num_va = preprocessor.transform(val)
    x_cat_te, x_num_te = preprocessor.transform(test)
    device = next(model.parameters()).device
    val_prob = saint_predict_proba(model, x_cat_va, x_num_va, device, batch_size=batch_size)
    test_prob = saint_predict_proba(model, x_cat_te, x_num_te, device, batch_size=batch_size)
    threshold = best_f1_threshold(_binary_labels(val[TARGET_COL]), val_prob)
    metrics = evaluate_probs(_binary_labels(test[TARGET_COL]), test_prob, threshold=threshold)
    metrics["model"] = "SAINT"
    metrics["library"] = "custom (not in pytorch-tabular)"
    metrics["best_val_auc"] = history.get("best_val_auc")
    metrics["saved_to"] = str(ckpt)
    return test_prob, metrics, (model, preprocessor)


def save_roc_plot(y_true: np.ndarray, curves: Dict[str, np.ndarray], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, prob in curves.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Telco churn ROC — trees vs tabular Transformers")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular Transformers for Telco churn")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--models",
        default="logreg,random_forest,xgboost,ft_transformer,tab_transformer,saint",
        help="Comma-separated: logreg,random_forest,xgboost,ft_transformer,tab_transformer,saint",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="2 epochs, for a smoke test")
    return parser.parse_args(argv)


def run(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    args = parse_args(argv)
    epochs = 2 if args.fast else args.epochs
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv}")
    data = load_and_prepare(args.csv)
    train, val, test = split_frames(data, seed=args.seed)
    print(
        f"Rows {len(data)} | train {len(train)} | val {len(val)} | test {len(test)} | "
        f"churn rate {data[TARGET_COL].eq('Yes').mean():.1%} | device {_accelerator()}"
    )

    y_test = _binary_labels(test[TARGET_COL])
    curves: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, float]] = []
    pred_frame = test[[ID_COL, TARGET_COL]].copy() if ID_COL in test.columns else test[[TARGET_COL]].copy()
    pred_frame["y_true"] = y_test

    trainers = {
        "logreg": lambda: train_logreg(train, val, test),
        "random_forest": lambda: train_random_forest(train, val, test, seed=args.seed),
        "xgboost": lambda: train_xgboost(train, val, test, seed=args.seed),
        "ft_transformer": lambda: train_pytorch_tabular(
            "ft_transformer", train, val, test, epochs, args.batch_size, args.seed
        ),
        "tab_transformer": lambda: train_pytorch_tabular(
            "tab_transformer", train, val, test, epochs, args.batch_size, args.seed
        ),
        "saint": lambda: train_saint_model(train, val, test, epochs, args.batch_size, args.seed),
    }

    for name in names:
        if name not in trainers:
            raise ValueError(f"Unknown model {name}. Choose from {list(trainers)}")
        print("\n" + "=" * 60)
        print(f"Training {name}")
        print("=" * 60)
        test_prob, metrics, _fitted = trainers[name]()
        curves[metrics["model"]] = test_prob
        pred_frame[f"prob_{name}"] = test_prob
        rows.append(metrics)
        print(classification_report(y_test, (test_prob >= metrics["threshold"]).astype(int), digits=3))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})

    metrics_df = pd.DataFrame(rows)
    preferred = [
        "model",
        "library",
        "roc_auc",
        "avg_precision",
        "f1_yes",
        "recall_yes",
        "precision_yes",
        "accuracy",
        "threshold",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    ordered = [c for c in preferred if c in metrics_df.columns] + [
        c for c in metrics_df.columns if c not in preferred
    ]
    metrics_df = metrics_df[ordered].sort_values("roc_auc", ascending=False)
    metrics_path = OUTPUT_DIR / "tabular_transformer_metrics.csv"
    pred_path = OUTPUT_DIR / "tabular_transformer_test_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False)
    pred_frame.to_csv(pred_path, index=False)
    save_roc_plot(y_test, curves, OUTPUT_DIR / "tabular_transformer_roc.png")

    summary = {
        "n_rows": int(len(data)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "churn_rate": float(data[TARGET_COL].eq("Yes").mean()),
        "device": _accelerator(),
        "best_model": str(metrics_df.iloc[0]["model"]),
        "best_roc_auc": float(metrics_df.iloc[0]["roc_auc"]),
        "metrics_csv": str(metrics_path),
    }
    (OUTPUT_DIR / "tabular_transformer_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nLeaderboard")
    print(metrics_df.to_string(index=False))
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {pred_path}")
    return metrics_df


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    run()
