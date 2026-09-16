"""Reusable feature-engineering stages for external tabular datasets (HyperAck ladder).

All stateful decisions are fit on train only, then applied to test with aligned columns.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer


def _numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _align(X: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = X.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = np.nan
    return out[list(columns)]


class FeatureEngineer:
    """Train-fitted FE pipeline producing aligned train/test matrices."""

    def __init__(self, stage: str = "full_fe", random_state: int = 42):
        self.stage = stage
        self.random_state = random_state
        self.log_cols_: List[str] = []
        self.ratio_pairs_: List[Tuple[str, str]] = []
        self.interaction_pairs_: List[Tuple[str, str]] = []
        self.mi_order_: List[str] = []
        self.kmeans_: Optional[KMeans] = None
        self.kmeans_cols_: List[str] = []
        self.bins_: Optional[KBinsDiscretizer] = None
        self.bin_cols_: List[str] = []
        self.output_cols_: List[str] = []
        self.selected_k_: Optional[int] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureEngineer":
        Xn = _numeric_frame(X)
        # log cols from train skew
        self.log_cols_ = []
        if self.stage in {"logs", "ratios", "interactions", "full_fe", "selected"}:
            for c in Xn.columns:
                s = Xn[c]
                if s.notna().sum() < 5:
                    continue
                if float(s.min(skipna=True)) >= 0 and float(s.std(skipna=True) or 0) > 0:
                    if abs(float(s.skew(skipna=True))) > 0.8:
                        self.log_cols_.append(c)

        base = self._stateless(Xn)

        # ratios from train variance
        self.ratio_pairs_ = []
        if self.stage in {"ratios", "interactions", "full_fe", "selected"}:
            variances = base.var(numeric_only=True).sort_values(ascending=False)
            cols = [c for c in variances.index.tolist()[:6] if c in base.columns]
            for i, a in enumerate(cols):
                for b in cols[i + 1 :]:
                    self.ratio_pairs_.append((a, b))

        with_ratios = self._apply_ratios(base)

        # MI on current matrix
        filled = with_ratios.fillna(with_ratios.median())
        cols = list(filled.columns)
        if len(cols) and len(y):
            mi = mutual_info_classif(filled[cols], y, random_state=self.random_state)
            self.mi_order_ = [c for c, _ in sorted(zip(cols, mi), key=lambda t: t[1], reverse=True)]
        else:
            self.mi_order_ = cols

        self.interaction_pairs_ = []
        if self.stage in {"interactions", "full_fe", "selected"}:
            top = self.mi_order_[:8]
            n = 0
            for i, a in enumerate(top):
                for b in top[i + 1 :]:
                    self.interaction_pairs_.append((a, b))
                    n += 1
                    if n >= 8:
                        break
                if n >= 8:
                    break

        with_inter = self._apply_interactions(with_ratios)

        if self.stage in {"full_fe", "selected"}:
            filled2 = with_inter.fillna(with_inter.median())
            var_cols = filled2.var().sort_values(ascending=False).index.tolist()[: min(6, filled2.shape[1])]
            self.kmeans_cols_ = var_cols
            if len(var_cols) >= 2 and len(filled2) >= 50:
                k = 5 if len(filled2) >= 200 else 3
                self.kmeans_ = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                self.kmeans_.fit(filled2[var_cols])

            # refresh MI after interactions
            mi2 = mutual_info_classif(filled2, y, random_state=self.random_state)
            self.mi_order_ = [c for c, _ in sorted(zip(filled2.columns, mi2), key=lambda t: t[1], reverse=True)]
            self.bin_cols_ = self.mi_order_[: min(4, len(self.mi_order_))]
            if self.bin_cols_ and len(filled2) >= 50:
                self.bins_ = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile", subsample=None)
                try:
                    self.bins_.fit(filled2[self.bin_cols_])
                except Exception:
                    self.bins_ = None

            if self.stage == "selected":
                self.selected_k_ = max(8, min(20, with_inter.shape[1] // 2))

        # freeze output schema from train transform
        train_out = self.transform(X)
        self.output_cols_ = list(train_out.columns)
        return self

    def _stateless(self, Xn: pd.DataFrame) -> pd.DataFrame:
        out = Xn.copy()
        if self.stage == "raw":
            return out
        for c in self.log_cols_:
            if c in out.columns:
                out[f"log1p_{c}"] = np.log1p(out[c].clip(lower=0))
        return out

    def _apply_ratios(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for a, b in self.ratio_pairs_:
            if a in out.columns and b in out.columns:
                denom = out[b].replace(0, np.nan)
                out[f"ratio_{a}_div_{b}"] = out[a] / denom
        return out.replace([np.inf, -np.inf], np.nan)

    def _apply_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for a, b in self.interaction_pairs_:
            if a in out.columns and b in out.columns:
                out[f"inter_{a}_x_{b}"] = out[a] * out[b]
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xn = _numeric_frame(X)
        out = self._stateless(Xn)
        if self.stage in {"ratios", "interactions", "full_fe", "selected"}:
            out = self._apply_ratios(out)
        if self.stage in {"interactions", "full_fe", "selected"}:
            out = self._apply_interactions(out)
        if self.stage in {"full_fe", "selected"}:
            filled = out.fillna(out.median())
            if self.kmeans_ is not None and self.kmeans_cols_:
                cols = [c for c in self.kmeans_cols_ if c in filled.columns]
                if len(cols) == len(self.kmeans_cols_):
                    out["kmeans_cluster"] = self.kmeans_.predict(filled[cols])
            if self.bins_ is not None and self.bin_cols_:
                cols = [c for c in self.bin_cols_ if c in filled.columns]
                if len(cols) == len(self.bin_cols_):
                    binned = self.bins_.transform(filled[cols])
                    for i, c in enumerate(cols):
                        out[f"qbin_{c}"] = binned[:, i]
            if self.stage == "selected" and self.selected_k_ is not None:
                keep = [c for c in self.mi_order_[: self.selected_k_] if c in out.columns]
                extra = [c for c in out.columns if c.startswith("kmeans_") or c.startswith("qbin_")]
                keep = list(dict.fromkeys(keep + extra))
                out = out[keep]
        if self.output_cols_:
            out = _align(out, self.output_cols_)
        return out.replace([np.inf, -np.inf], np.nan)

    def meta(self) -> dict:
        return {
            "stage": self.stage,
            "top_mi": self.mi_order_[:15],
            "log_cols": self.log_cols_,
            "n_ratio_pairs": len(self.ratio_pairs_),
            "n_interaction_pairs": len(self.interaction_pairs_),
            "selected_k": self.selected_k_,
            "output_cols": len(self.output_cols_),
        }


def build_feature_matrix(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    stage: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    fe = FeatureEngineer(stage=stage)
    fe.fit(X_train, y_train)
    return fe.transform(X_train), fe.transform(X_test), fe.meta()
