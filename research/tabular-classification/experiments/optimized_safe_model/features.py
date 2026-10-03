"""Optimized leakage-safe feature builders (no final fares)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

import sys
from pathlib import Path

HYPERACK = Path(__file__).resolve().parents[1] / "hyperack_exp"
sys.path.insert(0, str(HYPERACK))

from shared.protocol import (  # noqa: E402
    add_geo_features,
    add_pricing_features,
    add_time_features,
    make_xy,
)

UNSAFE = {"final_customer_fare", "final_biker_fare"}


def _drop_unsafe(X: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in X.columns if c in UNSAFE or c.startswith("final_")]
    return X.drop(columns=cols, errors="ignore")


def add_optimized_safe_features(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Extra safe-only transforms to recover performance without leakage."""
    X = X.copy()
    distance = X["total_distance"].clip(lower=0.05)
    fare = X["first_customer_fare"].clip(lower=1)

    X["log_first_fare"] = np.log1p(X["first_customer_fare"])
    X["sqrt_distance"] = np.sqrt(X["total_distance"].clip(lower=0))
    X["products_per_km"] = X["sum_product"] / distance
    X["fare_x_distance"] = fare * X["total_distance"]
    X["fare_x_products"] = fare * X["sum_product"].clip(lower=0)
    X["distance_x_products"] = X["total_distance"] * X["sum_product"].clip(lower=0)

    if "hour" in X.columns:
        hour = X["hour"]
        X["is_morning"] = hour.between(6, 10).astype(int)
        X["is_night"] = ((hour >= 22) | (hour <= 5)).astype(int)
        X["fare_x_hour"] = fare * hour
        X["distance_x_hour"] = X["total_distance"] * hour
        X["category_x_hour"] = X["deliverey_category_id"] * hour
        X["category_x_weekend"] = X["deliverey_category_id"] * X.get("is_weekend", 0)

    if "haversine_km" in X.columns:
        X["route_vs_reported"] = X["haversine_km"] / distance
        X["abs_lat_delta"] = X["latitude_delta"].abs()
        X["abs_lon_delta"] = X["longitude_delta"].abs()

    # Coarse city-ish magnitude features (still pre-decision).
    X["source_lat_abs"] = df["source_latitude"].abs()
    X["source_lon_abs"] = df["source_longitude"].abs()
    return X.replace([np.inf, -np.inf], np.nan)


def build_optimized_safe_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Full optimized safe feature matrix for one dataframe slice."""
    X, y = make_xy(
        df,
        include_final_fares=False,
        pricing=True,
        time_features=True,
        geo=True,
    )
    X = _drop_unsafe(X)
    X = add_optimized_safe_features(df, X)
    return X, y


def add_train_only_geo_clusters(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_clusters: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    geo_cols = [
        "source_latitude",
        "source_longitude",
        "destination_latitude",
        "destination_longitude",
    ]
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=n_clusters, n_init=20, random_state=42),
            ),
        ]
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["geo_cluster_opt"] = pipe.fit_predict(train_df[geo_cols])
    X_test["geo_cluster_opt"] = pipe.predict(test_df[geo_cols])
    return X_train, X_test


def add_train_only_bins(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_test = X_test.copy()
    bin_cols = ["total_distance", "first_customer_fare", "sum_product"]
    bin_cols = [c for c in bin_cols if c in X_train.columns]
    binning = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "bin",
                KBinsDiscretizer(
                    n_bins=10,
                    encode="ordinal",
                    strategy="quantile",
                    quantile_method="averaged_inverted_cdf",
                ),
            ),
        ]
    )
    tr = binning.fit_transform(X_train[bin_cols])
    te = binning.transform(X_test[bin_cols])
    for i, name in enumerate(bin_cols):
        X_train[f"{name}_opt_bin"] = tr[:, i]
        X_test[f"{name}_opt_bin"] = te[:, i]
    return X_train, X_test


def prepare_optimized_xy(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Build the full optimized safe train/test matrices."""
    X_train, y_train = build_optimized_safe_matrix(train_df)
    X_test, y_test = build_optimized_safe_matrix(test_df)
    X_train, X_test = add_train_only_geo_clusters(train_df, test_df, X_train, X_test)
    X_train, X_test = add_train_only_bins(X_train, X_test)
    assert not any("final_" in c for c in X_train.columns)
    return X_train, y_train, X_test, y_test
