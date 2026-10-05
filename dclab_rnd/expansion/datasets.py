"""Dataset adapters for the ``expansion_v1`` campaign.

Each adapter pins a public source URL and the SHA-256 of the raw file, declares
the decision-time contract (which columns are blocked and why), and produces a
:class:`TaskBundle` with one deterministic locked holdout plus training-only CV
folds that respect the task's structure (time order, entity groups, or class
strata).

Raw files live in ``data/downloads/<dataset>/`` and are never committed.  Nothing
in this module downloads data at import time; :func:`dataset_cards` is purely
static metadata.
"""

from __future__ import annotations

import os
import shutil
import ssl
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from dclab_rnd.provenance import file_sha256

RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.20
CV_FOLDS = 3

TASK_TYPES = (
    "binary_imbalanced",
    "multiclass",
    "timeseries_regression",
    "text_tabular_binary",
)
HIGHER_IS_BETTER = {
    "average_precision": True,
    "roc_auc": True,
    "macro_f1": True,
    "mae": False,
}


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one dataset and its decision-time contract."""

    key: str
    name: str
    task_type: str
    primary_metric: str
    target: str
    url: str
    filename: str
    sha256: str
    description: str
    decision_time_contract: str
    split_strategy: str  # "time" | "stratified" | "group"
    source_rows: int
    feature_count: int
    blocked_features: dict[str, str] = field(default_factory=dict)
    identifier_columns: dict[str, str] = field(default_factory=dict)
    text_columns: tuple[str, ...] = ()
    # Numeric columns that stand for categories (DCLAB-R11): kept as inputs, never logged or multiplied.
    categorical_columns: tuple[str, ...] = ()
    time_column: str | None = None
    group_column: str | None = None
    positive_rate: float | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def higher_is_better(self) -> bool:
        return HIGHER_IS_BETTER[self.primary_metric]

    def holdout_policy(self) -> str:
        if self.split_strategy == "time":
            return (
                f"Locked holdout = last {HOLDOUT_FRACTION:.0%} of rows in `{self.time_column}` order; "
                f"recipe/model/parameter choices use only expanding-window {CV_FOLDS}-fold time-ordered CV "
                "on the earlier rows. The holdout is consumed once, in optimization_reliability."
            )
        if self.split_strategy == "group":
            return (
                f"Locked holdout = one of 5 stratified group folds by `{self.group_column}` (~{HOLDOUT_FRACTION:.0%} of rows, "
                f"no {self.group_column} shared with training); selection uses {CV_FOLDS}-fold stratified group CV "
                "on training rows only. The holdout is consumed once, in optimization_reliability."
            )
        return (
            f"Locked holdout = stratified random {HOLDOUT_FRACTION:.0%}; selection uses {CV_FOLDS}-fold stratified CV "
            "on training rows only. The holdout is consumed once, in optimization_reliability."
        )


SPECS: dict[str, DatasetSpec] = {
    "credit_card_fraud": DatasetSpec(
        key="credit_card_fraud",
        name="Credit Card Fraud Detection (ULB / Kaggle)",
        task_type="binary_imbalanced",
        primary_metric="average_precision",
        target="Class",
        url="https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv",
        filename="creditcard.csv",
        sha256="33a178be65174943a07d1736acf5898f6c8e1fa80950dbbe598c17eda968ec59",
        description=(
            "Two days of September-2013 European card transactions (28 PCA components V1-V28, Amount, "
            "elapsed Time) used to predict whether a transaction is fraudulent (Class=1, 0.17% of rows)."
        ),
        decision_time_contract=(
            "Score each transaction at authorization time using only that transaction's own attributes "
            "(V1-V28, Amount, and time-of-day derived from its timestamp); the dataset-relative elapsed "
            "`Time` counter itself is not a production feature."
        ),
        split_strategy="time",
        source_rows=284807,
        feature_count=30,
        blocked_features={
            "Time": (
                "Seconds elapsed since the first transaction in this two-day extract. It is a dataset-relative "
                "counter that grows without bound in production, so a model trained on it extrapolates outside "
                "its training range; only cyclic hour-of-day derived from it is decision-time safe."
            )
        },
        time_column="Time",
        positive_rate=492 / 284807,
        settings={
            "fit_negative_fraction": 0.25,
            "precision_target": 0.90,
            "fe_stage_model": "lightgbm",
            "fe_tolerance": 0.005,
            "fe_tolerance_relative": False,
            "tuning_margin": 0.005,
            "tuning_margin_relative": False,
        },
    ),
    "letter_recognition": DatasetSpec(
        key="letter_recognition",
        name="Letter Recognition (UCI via PMLB)",
        task_type="multiclass",
        primary_metric="macro_f1",
        target="target",
        url="https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/letter/letter.tsv.gz",
        filename="letter.tsv.gz",
        sha256="f8ffe4f9920dc63fb85b864cc1afffb66e45c6e302d175905fd7db04ba67b8c6",
        description=(
            "20,000 distorted capital-letter glyph images summarized by 16 integer shape statistics, used to "
            "predict which of the 26 letters (A-Z) the glyph shows."
        ),
        decision_time_contract=(
            "All 16 shape statistics are computed from the glyph image itself before classification, so every "
            "column is decision-time available; no column is blocked."
        ),
        split_strategy="stratified",
        source_rows=20000,
        feature_count=16,
        read_kwargs={"sep": "\t"},
        settings={
            "fe_stage_model": "lightgbm",
            "fe_tolerance": 0.002,
            "fe_tolerance_relative": False,
            "tuning_margin": 0.002,
            "tuning_margin_relative": False,
        },
    ),
    "bike_sharing_daily": DatasetSpec(
        key="bike_sharing_daily",
        name="Bike Sharing (Capital Bikeshare, daily; UCI via interpretable-ml-book)",
        task_type="timeseries_regression",
        primary_metric="mae",
        target="cnt",
        url="https://raw.githubusercontent.com/christophM/interpretable-ml-book/master/data/bike.csv",
        filename="bike.csv",
        sha256="a6bed1577f6e4de56dcfae6d8315985073cc0ad3446c8c99ed60c86a0b5ad9dc",
        description=(
            "728 days (2011-01-03 to 2012-12-31) of Washington D.C. Capital Bikeshare usage with calendar and "
            "weather attributes, used to predict the total daily rental count (cnt)."
        ),
        decision_time_contract=(
            "Forecast day d's total rentals at the end of day d-2 (the horizon implied by the provided "
            "`cnt_2d_bfr` lag): calendar fields, the day's weather (treated as a forecast proxy), and rental "
            "counts from day d-2 or earlier are allowed; same-day casual/registered counts are not."
        ),
        split_strategy="time",
        source_rows=728,
        feature_count=17,
        blocked_features={
            "casual": "Same-day count of casual riders; casual + registered == cnt exactly, so it is a post-outcome component of the target.",
            "registered": "Same-day count of registered riders; casual + registered == cnt exactly, so it is a post-outcome component of the target.",
        },
        identifier_columns={
            "instant": "Row/time index (duplicates days_since_2011 + 1); not a predictive attribute.",
            "dteday": "Calendar date key; used only to derive calendar and past-only lag features.",
        },
        time_column="dteday",
        settings={
            "fe_stage_model": "lightgbm",
            "fe_tolerance": 0.01,
            "fe_tolerance_relative": True,
            "tuning_margin": 0.01,
            "tuning_margin_relative": True,
            "min_lag_days": 2,
            "block_length": 7,
        },
    ),
    "ecommerce_clothing_reviews": DatasetSpec(
        key="ecommerce_clothing_reviews",
        name="Women's E-Commerce Clothing Reviews (Kaggle)",
        task_type="text_tabular_binary",
        primary_metric="roc_auc",
        target="Recommended IND",
        url="https://raw.githubusercontent.com/AFAgarap/ecommerce-reviews-analysis/master/Womens%20Clothing%20E-Commerce%20Reviews.csv",
        filename="Womens Clothing E-Commerce Reviews.csv",
        sha256="bd93cc515747ad1f87b8bc863c9e40da758509e6db6506a96d06976e61e36ee0",
        description=(
            "23,486 customer reviews of women's clothing (title, free text, reviewer age, product taxonomy) "
            "used to predict whether the reviewer recommends the product (Recommended IND=1, 82% of rows)."
        ),
        decision_time_contract=(
            "Predict the recommendation flag from the review title/text, reviewer age, and product taxonomy "
            "at submission time, for products not seen in training; the star rating (written in the same form "
            "as the recommendation) and helpful-vote counts (accrued after publication) are not available."
        ),
        split_strategy="group",
        source_rows=23486,
        feature_count=10,
        blocked_features={
            "Rating": "1-5 star rating entered by the reviewer in the same submission as the recommendation flag; a post-outcome proxy of the target, not an input available before the reviewer states the outcome.",
            "Positive Feedback Count": "Number of other shoppers who later marked the review helpful; it accrues after publication and is unavailable at submission time.",
        },
        identifier_columns={
            "Unnamed: 0": "Row index written by the CSV export; carries no product or reviewer meaning.",
            "Clothing ID": "High-cardinality product identifier (1,206 values); used as the grouping key so holdout/CV products are unseen, never as a raw feature.",
        },
        text_columns=("Title", "Review Text"),
        group_column="Clothing ID",
        positive_rate=0.8223622583666865,
        settings={
            "fe_stage_model": "logistic_regression",
            "fe_tolerance": 0.002,
            "fe_tolerance_relative": False,
            "tuning_margin": 0.001,
            "tuning_margin_relative": False,
        },
    ),
}
DATASET_ORDER = tuple(SPECS)


def dataset_cards() -> list[dict[str, Any]]:
    """Static per-dataset metadata for indexes; never downloads or reads data."""
    cards = []
    for spec in SPECS.values():
        card: dict[str, Any] = {
            "key": spec.key,
            "name": spec.name,
            "description": spec.description,
            "task_type": spec.task_type,
            "rows": int(spec.source_rows),
            "features": int(spec.feature_count),
            "source": spec.url,
            "decision_time_contract": spec.decision_time_contract,
            "blocked_features": list(spec.blocked_features),
        }
        if spec.task_type in {"binary_imbalanced", "text_tabular_binary"}:
            card["positive_rate"] = float(spec.positive_rate or 0.0)
        cards.append(card)
    return cards


@dataclass
class TaskBundle:
    """Data plus the locked split and training-only CV folds for one dataset."""

    spec: DatasetSpec
    X: pd.DataFrame
    y: pd.Series
    train_idx: np.ndarray
    test_idx: np.ndarray
    cv_splits: list[tuple[np.ndarray, np.ndarray]]
    cv_description: str
    holdout_description: str
    data_paths: list[Path] = field(default_factory=list)
    source_rows: int = 0
    sampling: dict[str, Any] = field(default_factory=dict)
    class_labels: list[Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def task_type(self) -> str:
        return self.spec.task_type

    @property
    def primary_metric(self) -> str:
        return self.spec.primary_metric

    @property
    def X_train(self) -> pd.DataFrame:
        return self.X.iloc[self.train_idx].reset_index(drop=True)

    @property
    def y_train(self) -> pd.Series:
        return self.y.iloc[self.train_idx].reset_index(drop=True)

    @property
    def X_test(self) -> pd.DataFrame:
        return self.X.iloc[self.test_idx].reset_index(drop=True)

    @property
    def y_test(self) -> pd.Series:
        return self.y.iloc[self.test_idx].reset_index(drop=True)

    def groups(self, part: str) -> np.ndarray | None:
        column = self.spec.group_column
        if not column:
            return None
        index = self.train_idx if part == "train" else self.test_idx
        return self.X[column].iloc[index].to_numpy()


def prepare_bundle(
    spec: DatasetSpec,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    data_paths: list[Path] | None = None,
    source_rows: int | None = None,
    sampling: dict[str, Any] | None = None,
    class_labels: list[Any] | None = None,
    cv_folds: int = CV_FOLDS,
) -> TaskBundle:
    """Create the locked holdout and training-only CV folds for ``spec``."""
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedGroupKFold,
        StratifiedKFold,
        TimeSeriesSplit,
        train_test_split,
    )

    regression = spec.task_type == "timeseries_regression"
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    n = len(X)
    if spec.split_strategy == "time":
        order = np.argsort(_time_key(X[spec.time_column]), kind="stable")
        X = X.iloc[order].reset_index(drop=True)
        y = y.iloc[order].reset_index(drop=True)
        n_test = int(round(HOLDOUT_FRACTION * n))
        train_idx = np.arange(n - n_test)
        test_idx = np.arange(n - n_test, n)
        splitter = TimeSeriesSplit(n_splits=cv_folds)
        cv_splits = [(a, b) for a, b in splitter.split(train_idx)]
        cv_description = f"expanding-window TimeSeriesSplit({cv_folds}) on time-ordered training rows"
        holdout_description = f"last {n_test} of {n} rows by `{spec.time_column}`"
    elif spec.split_strategy == "group":
        groups = X[spec.group_column].to_numpy()
        # Regression targets cannot be stratified: fall back to plain group folds.
        outer_cls = GroupKFold if regression else StratifiedGroupKFold
        outer_kw = {} if regression else {"shuffle": True, "random_state": RANDOM_STATE}
        train_idx, test_idx = next(outer_cls(n_splits=5, **outer_kw).split(X, y, groups))
        inner = outer_cls(n_splits=cv_folds, **outer_kw)
        cv_splits = [
            (a, b)
            for a, b in inner.split(train_idx, y.iloc[train_idx], groups[train_idx])
        ]
        cv_description = f"{outer_cls.__name__}({cv_folds}) by `{spec.group_column}` on training rows"
        holdout_description = (
            f"{len(test_idx)} rows from {len(np.unique(groups[test_idx]))} `{spec.group_column}` groups "
            "never seen in training (first of 5 stratified group folds)"
        )
    else:
        indices = np.arange(n)
        train_idx, test_idx = train_test_split(
            indices, test_size=HOLDOUT_FRACTION, stratify=None if regression else y, random_state=RANDOM_STATE
        )
        train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)
        inner_cls = KFold if regression else StratifiedKFold
        inner = inner_cls(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_splits = [(a, b) for a, b in inner.split(train_idx, y.iloc[train_idx])]
        cv_description = f"{inner_cls.__name__}({cv_folds}, shuffle, random_state={RANDOM_STATE}) on training rows"
        holdout_description = f"{'random' if regression else 'stratified random'} {len(test_idx)} of {n} rows"
    return TaskBundle(
        spec=spec,
        X=X,
        y=y,
        train_idx=np.asarray(train_idx),
        test_idx=np.asarray(test_idx),
        cv_splits=cv_splits,
        cv_description=cv_description,
        holdout_description=holdout_description,
        data_paths=list(data_paths or []),
        source_rows=int(source_rows if source_rows is not None else n),
        sampling=dict(sampling or {}),
        class_labels=class_labels,
    )


def _time_key(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float)
    return pd.to_datetime(series).to_numpy(dtype="datetime64[ns]").astype("int64")


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_path(root: Path, spec: DatasetSpec) -> Path:
    return root / "data" / "downloads" / spec.key / spec.filename


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    try:
        context = ssl.create_default_context(cafile=cafile) if cafile else None
        with urllib.request.urlopen(url, timeout=300, context=context) as response:
            with temp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except Exception:  # fall back to curl, which honours the proxy CA bundle
        if shutil.which("curl") is None:
            raise
        subprocess.run(["curl", "-sSfL", "-o", str(temp), url], check=True, timeout=900)
    temp.replace(destination)


def ensure_raw_file(root: Path, spec: DatasetSpec, *, verify: bool = True) -> Path:
    """Return the pinned raw file, downloading it only when missing."""
    path = raw_path(root, spec)
    if not path.is_file():
        _download(spec.url, path)
    if verify:
        digest = file_sha256(path)
        if digest != spec.sha256:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {spec.sha256}, got {digest}. "
                "Delete the file to re-download, or update the pinned hash after review."
            )
    return path


def _load_fraud(root: Path, spec: DatasetSpec) -> TaskBundle:
    path = ensure_raw_file(root, spec)
    frame = pd.read_csv(path)
    y = frame.pop(spec.target).astype(int)
    return prepare_bundle(
        spec,
        frame,
        y,
        data_paths=[path],
        source_rows=len(frame),
        sampling={
            "rows_used": len(frame),
            "fit_negative_fraction": spec.settings["fit_negative_fraction"],
            "note": (
                "All rows and all positives are used for validation and holdout scoring. Only model *fits* keep all "
                f"positives plus a seeded {spec.settings['fit_negative_fraction']:.0%} sample of negatives; scores are "
                "prior-corrected back to the true base rate before threshold and calibration metrics."
            ),
        },
    )


def _load_letter(root: Path, spec: DatasetSpec) -> TaskBundle:
    path = ensure_raw_file(root, spec)
    frame = pd.read_csv(path, **spec.read_kwargs)
    raw_target = frame.pop(spec.target)
    labels = sorted(raw_target.unique().tolist())
    mapping = {value: index for index, value in enumerate(labels)}
    y = raw_target.map(mapping).astype(int)
    return prepare_bundle(
        spec,
        frame,
        y,
        data_paths=[path],
        source_rows=len(frame),
        sampling={"rows_used": len(frame), "note": "No subsampling."},
        class_labels=[str(value) for value in labels],
    )


def _load_bike(root: Path, spec: DatasetSpec) -> TaskBundle:
    path = ensure_raw_file(root, spec)
    frame = pd.read_csv(path)
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype(object)
    y = frame.pop(spec.target).astype(float)
    return prepare_bundle(
        spec,
        frame,
        y,
        data_paths=[path],
        source_rows=len(frame),
        sampling={"rows_used": len(frame), "note": "No subsampling; 2011-03-10 is absent from the source file."},
    )


def _load_reviews(root: Path, spec: DatasetSpec) -> TaskBundle:
    path = ensure_raw_file(root, spec)
    frame = pd.read_csv(path)
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype(object)
    y = frame.pop(spec.target).astype(int)
    return prepare_bundle(
        spec,
        frame,
        y,
        data_paths=[path],
        source_rows=len(frame),
        sampling={"rows_used": len(frame), "note": "No subsampling."},
    )


LOADERS: dict[str, Callable[[Path, DatasetSpec], TaskBundle]] = {
    "credit_card_fraud": _load_fraud,
    "letter_recognition": _load_letter,
    "bike_sharing_daily": _load_bike,
    "ecommerce_clothing_reviews": _load_reviews,
}


def load_bundle(root: Path, key: str) -> TaskBundle:
    """Load (downloading if needed) one dataset and build its locked split."""
    spec = SPECS[key]
    return LOADERS[key](Path(root), spec)
