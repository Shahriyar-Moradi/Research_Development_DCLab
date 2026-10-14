"""Anomaly detection without labels (package 20.7): detectors, their evaluation on planted anomalies, a review list.

With no label there is nothing to score a detector against but what is planted. The rows are split by time (whole
days: the latest are the rows to review now, a validation period before them); each detector is fitted on the earlier
rows, and the validation rows receive planted anomalies of three generic shapes (a value beyond the column's range; the
two most correlated columns broken against their direction; a row's entity counts pushed far above the rest); the detector
that finds most of them at the review budget is chosen. It is then fitted on every training row and ranks the latest
rows: the review list, as many rows as the people reviewing can take. Its measure is a measure of what it finds among
planted shapes, never a precision on the real anomalies, which nobody has labelled; that limit is said with it.

Detectors (each scores higher for more unusual, fitted on the training rows only): ``rules`` (the largest robust
deviation of any feature), ``isolation_forest``, ``nearest_neighbours`` (mean distance to the nearest training rows)
and ``entity`` (a row against its entity's own usual). Features: the numeric columns, robust-scaled (skewed ones on a
log scale), and, with an entity, its behaviour (``behaviour.EntityBehaviour`` without labels: counts per entity and day,
trip shapes, repeated places, a row against its entity's mean). Confirmed alerts become labels (the host's review
loop); with labels the problem is a rare-event one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

DETECTORS = ("rules", "isolation_forest", "nearest_neighbours", "entity")
SHAPES = ("far_value", "broken_pair", "busy_entity")
ID_LIKE = ("_id", "_ref", "_key", "_uuid", "phone", "email")


def _robust(x: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    med = float(x.median()) if len(x) else 0.0
    mad = float((x - med).abs().median()) * 1.4826 if len(x) else 0.0
    return med, (mad if mad > 0 else float(x.std() or 0.0) or 1.0)


class Features:
    """What the detectors read, learned from the training rows: numbers robust-scaled (skewed ones logged) and, with an
    entity, its behaviour. ``transform`` builds the same columns for any rows, with the training rows' scales."""

    def __init__(self, columns: Sequence[str], entity: Optional[str] = None, time: Optional[str] = None):
        self.columns, self.entity, self.time = list(columns), entity, time

    def fit(self, train: pd.DataFrame) -> "Features":
        self.behaviour = None
        if self.entity and self.entity in train.columns:
            from dclab_core.behaviour import EntityBehaviour

            self.behaviour = EntityBehaviour([self.entity], time_column=self.time, mode="audit", numeric=self.columns,
                                             grouped=[self.entity]).fit(train, np.zeros(len(train)))
        raw = self._raw(train)
        self.logged = [c for c in raw.columns if raw[c].min() >= 0 and float(raw[c].skew() or 0) > 2]
        raw = self._log(raw)
        keep = [c for c in raw.columns if raw[c].nunique() > 1]  # a constant says nothing (place rates with no labels)
        self.scale = {c: _robust(raw[c]) for c in keep}
        return self

    def _raw(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[self.columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        if self.behaviour is not None:
            extra = self.behaviour.transform(frame).reset_index(drop=True)
            out = pd.concat([out, extra[[c for c in extra.columns if c not in out.columns]]], axis=1)
        return out.astype(float)

    def _log(self, raw: pd.DataFrame) -> pd.DataFrame:
        for c in getattr(self, "logged", []):
            if c in raw:
                raw[c] = np.log1p(raw[c].clip(lower=0))
        return raw

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        raw = self._log(self._raw(frame))
        cols = list(self.scale)
        raw = raw.reindex(columns=cols)
        med = pd.Series({c: self.scale[c][0] for c in cols})
        z = (raw - med) / pd.Series({c: self.scale[c][1] for c in cols})
        return z.fillna(0.0).clip(-50, 50)


def with_history(feats: Features, history: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """``rows``' features computed with the earlier rows as their history (an entity's counts and usual places reach back
    as far as the training rows' do), then only theirs kept. In audit mode each row still sees nothing after its own day."""
    both = feats.transform(pd.concat([history, rows]))
    return both.iloc[len(history):].reset_index(drop=True)


def fit_detector(name: str, Z: pd.DataFrame, entity: Optional[pd.Series] = None, seed: int = 0) -> Any:
    """A detector fitted on the training rows' features ``Z``; ``score(Z2, entity2)`` is higher for more unusual rows."""
    if name == "rules":
        return lambda Z2, e=None: Z2.abs().max(axis=1).to_numpy()
    if name == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        forest = IsolationForest(n_estimators=150, max_samples=min(1024, len(Z)), random_state=seed).fit(Z)
        return lambda Z2, e=None: -forest.score_samples(Z2)
    if name == "nearest_neighbours":
        from sklearn.neighbors import NearestNeighbors

        rng = np.random.default_rng(seed)
        sample = Z.iloc[rng.choice(len(Z), size=min(len(Z), 20000), replace=False)]
        nn = NearestNeighbors(n_neighbors=min(10, len(sample))).fit(sample)
        return lambda Z2, e=None: nn.kneighbors(Z2)[0].mean(axis=1)
    if name == "entity":
        if entity is None:
            return None
        centre = Z.groupby(entity.to_numpy()).median()

        def score(Z2: pd.DataFrame, e: Optional[pd.Series] = None) -> np.ndarray:
            if e is None:
                return Z2.abs().max(axis=1).to_numpy()
            own = centre.reindex(e.to_numpy()).fillna(0.0).to_numpy()  # an entity not seen: its usual is everyone's
            return np.abs(Z2.to_numpy() - own).max(axis=1)
        return score
    raise ValueError(f"unknown detector {name}")


def plant(Z: pd.DataFrame, shape: str, k: int, rng: np.random.Generator, entity: Optional[pd.Series] = None,
          busy: Optional[list[str]] = None) -> tuple[pd.DataFrame, np.ndarray]:
    """A copy of ``Z`` with ``k`` rows made anomalous in ``shape``; returns (planted, truth). ``far_value``: one column
    beyond its range, each at its own distance; ``broken_pair``: the two most correlated columns against their
    direction; ``busy_entity``: the entity's count columns (``busy``) pushed far beyond the rest."""
    planted = Z.copy()
    rows = rng.choice(len(Z), size=min(k, len(Z)), replace=False)
    cols = list(Z.columns)
    if shape == "far_value":
        for r in rows:
            c = cols[int(rng.integers(len(cols)))]
            far = 8.0 + float(rng.uniform(0, 8))
            planted.iloc[r, planted.columns.get_loc(c)] = (Z[c].max() + far) if rng.integers(2) else (Z[c].min() - far)
    elif shape == "broken_pair":
        signed = Z.corr(method="spearman").where(~np.eye(len(cols), dtype=bool)).stack()
        if not len(signed) or signed.abs().max() < 0.5:
            return Z, np.zeros(len(Z), dtype=int)
        a, b = signed.abs().idxmax()
        planted.iloc[rows, planted.columns.get_loc(a)] = Z[a].quantile(0.1)
        planted.iloc[rows, planted.columns.get_loc(b)] = Z[b].quantile(0.9 if signed[(a, b)] > 0 else 0.1)
    elif shape == "busy_entity":
        counts = [c for c in (busy or []) if c in Z.columns]
        if not counts:
            return Z, np.zeros(len(Z), dtype=int)
        for c in counts:
            planted.iloc[rows, planted.columns.get_loc(c)] = Z[c].max() + 8.0 + rng.uniform(0, 8, len(rows))
    else:
        raise ValueError(f"unknown shape {shape}")
    truth = np.zeros(len(Z), dtype=int)
    truth[rows] = 1
    return planted, truth


def _days(frame: pd.DataFrame, time: Optional[str]) -> Optional[pd.Series]:
    if not time or time not in frame.columns:
        return None
    raw = frame[time]
    if pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_datetime64_any_dtype(raw):
        return None
    stamps = pd.to_datetime(raw, errors="coerce", format="mixed")
    return stamps.dt.floor("D") if stamps.notna().mean() > 0.9 else None


def split(frame: pd.DataFrame, time: Optional[str] = None, holdout: float = 0.2, validation: float = 0.2, seed: int = 0) -> dict[str, Any]:
    """The rows to review now (the latest days, never cut inside a day), and inside the training rows a validation period
    before them. Without a time column, a seeded share of rows."""
    days = _days(frame, time)
    if days is not None and days.nunique() >= 3:
        order = sorted(days.dropna().unique())
        n = len(order)
        h = max(1, int(round(n * holdout)))
        v = max(1, int(round((n - h) * validation)))
        hold_from, val_from = order[n - h], order[n - h - v]
        holdout_idx = frame.index[days >= hold_from]
        train_idx = frame.index[days < hold_from]
        valid_idx = frame.index[(days >= val_from) & (days < hold_from)]
        info = {"by": "day", "holdout_from": str(pd.Timestamp(hold_from).date()), "validation_from": str(pd.Timestamp(val_from).date())}
    else:
        rng = np.random.default_rng(seed)
        idx = frame.index.to_numpy()
        rng.shuffle(idx)
        h = int(len(idx) * holdout)
        holdout_idx, train_idx = pd.Index(idx[:h]), pd.Index(idx[h:])
        valid_idx = pd.Index(train_idx[: int(len(train_idx) * validation)])
        info = {"by": "row", "holdout_from": None, "validation_from": None}
    fit_idx = train_idx.difference(valid_idx)
    return {"train": frame.loc[train_idx], "fit": frame.loc[fit_idx], "validation": frame.loc[valid_idx], "holdout": frame.loc[holdout_idx],
            "split": {**info, "train": int(len(train_idx)), "fit": int(len(fit_idx)), "validation": int(len(valid_idx)),
                      "holdout": int(len(holdout_idx))}}


def budget_rows(rows: int, budget: Optional[float]) -> int:
    """The review budget as a number of rows: a share of the rows (0 < budget < 1), a count, or 1% by default."""
    if budget is None or budget <= 0:
        return max(1, int(round(0.01 * rows)))
    return max(1, int(round(budget * rows))) if budget < 1 else max(1, min(rows, int(budget)))


def _recall(score: np.ndarray, truth: np.ndarray, k: int) -> float:
    top = np.argsort(-score)[:k]
    return float(truth[top].sum() / max(truth.sum(), 1))


def choose(parts: dict[str, Any], columns: Sequence[str], entity: Optional[str] = None, time: Optional[str] = None,
           budget: Optional[float] = None, detectors: Sequence[str] = DETECTORS, seed: int = 0) -> dict[str, Any]:
    """Each detector fitted on the fit rows and judged on the validation rows with planted anomalies: its recall at the
    budget per shape (as many planted rows as the budget). Chosen: the best mean recall; a tie to the simpler detector."""
    feats = Features(columns, entity, time).fit(parts["fit"])
    Zf, Zv = feats.transform(parts["fit"]), with_history(feats, parts["fit"], parts["validation"])
    if Zf.shape[1] == 0:
        raise ValueError("No numeric column and no entity to read: a detector has nothing to score")
    ef = parts["fit"][entity].reset_index(drop=True) if entity and entity in parts["fit"] else None
    ev = parts["validation"][entity].reset_index(drop=True) if entity and entity in parts["validation"] else None
    k = budget_rows(len(Zv), budget)
    busy = [c for c in Zv.columns if c.endswith("_count")]
    rng = np.random.default_rng(seed)
    planted = {s: plant(Zv, s, k, rng, ev, busy) for s in SHAPES}
    planted = {s: p for s, p in planted.items() if p[1].sum()}
    results: dict[str, Any] = {}
    for name in detectors:
        det = fit_detector(name, Zf, ef, seed)
        if det is None:
            continue
        per = {s: round(_recall(det(P, ev), truth, k), 4) for s, (P, truth) in planted.items()}
        results[name] = {"recall_at_budget": per, "mean": round(float(np.mean(list(per.values()))), 4) if per else 0.0}
    # stability: the same detector fitted on each half of the fit rows ranks the validation rows alike
    order = list(DETECTORS)
    best = max(results, key=lambda n: (results[n]["mean"], -order.index(n)))
    halves = np.array_split(np.arange(len(Zf)), 2)
    if min(len(h) for h in halves) >= 50:
        a = fit_detector(best, Zf.iloc[halves[0]], ef.iloc[halves[0]] if ef is not None else None, seed)(Zv, ev)
        b = fit_detector(best, Zf.iloc[halves[1]], ef.iloc[halves[1]] if ef is not None else None, seed)(Zv, ev)
        top_a, top_b = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
        stability = round(len(top_a & top_b) / k, 4)
    else:
        stability = None
    return {"validation": results, "chosen": best, "budget_rows": k, "shapes": sorted(planted), "stability": stability,
            "features": list(Zf.columns), "chosen_by": "the best mean recall at the budget on planted anomalies of the validation rows"}


def review(parts: dict[str, Any], chosen: str, columns: Sequence[str], entity: Optional[str] = None, time: Optional[str] = None,
           budget: Optional[float] = None, key: Optional[str] = None, seed: int = 0) -> dict[str, Any]:
    """The chosen detector fitted on every training row, ranking the rows to review now: the review list (key, entity,
    score, the features that stand out), and the same detector's recall on planted anomalies of those rows."""
    feats = Features(columns, entity, time).fit(parts["train"])
    Zt, Zh = feats.transform(parts["train"]), with_history(feats, parts["train"], parts["holdout"])
    et = parts["train"][entity].reset_index(drop=True) if entity and entity in parts["train"] else None
    eh = parts["holdout"][entity].reset_index(drop=True) if entity and entity in parts["holdout"] else None
    det = fit_detector(chosen, Zt, et, seed)
    base = fit_detector("rules", Zt, et, seed)
    score = det(Zh, eh)
    k = budget_rows(len(Zh), budget)
    top = np.argsort(-score)[:k]
    rows = parts["holdout"].reset_index(drop=True)
    listed = []
    for i in top:
        z = Zh.iloc[i]
        why = [c for c in z.abs().sort_values(ascending=False).index[:3] if abs(z[c]) >= 2]
        listed.append({"key": None if key is None or key not in rows else _label(rows.loc[i, key]),
                       "entity": None if eh is None else _label(eh.iloc[i]), "score": round(float(score[i]), 4), "stands_out": why})
    busy = [c for c in Zh.columns if c.endswith("_count")]
    rng = np.random.default_rng(seed + 1)
    planted = {s: plant(Zh, s, k, rng, eh, busy) for s in SHAPES}
    measured = {s: {"chosen": round(_recall(det(P, eh), t, k), 4), "rules": round(_recall(base(P, eh), t, k), 4)}
                for s, (P, t) in planted.items() if t.sum()}
    return {"review_list": listed, "budget_rows": k, "rows_scored": int(len(Zh)), "planted": measured,
            "scores": score, "limits": [
                "Measured on anomalies of three generic shapes planted in a copy of the rows to review (a value beyond its range, a "
                "correlated pair broken, a row's entity counts pushed high): never a precision on the real anomalies, which nobody has labelled.",
                "An anomaly that leaves no trace in these columns (a fraud known only from a column the table does not have) is not "
                "found by any detector here.",
                "Each reviewed alert is a label: once enough are confirmed, the problem is a rare-event one, judged on its labels."]}


def _label(value: Any) -> str:
    """A key or an entity as a person wrote it: 921270, never 921270.0 (a cleaned column of ids may hold floats)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def evaluate(frame: pd.DataFrame, columns: Sequence[str], entity: Optional[str] = None, time: Optional[str] = None,
             budget: Optional[float] = None, key: Optional[str] = None, seed: int = 0) -> dict[str, Any]:
    """Split, choose on the validation period, rank the rows to review once."""
    parts = split(frame, time, seed=seed)
    chosen = choose(parts, columns, entity, time, budget, seed=seed)
    listed = review(parts, chosen["chosen"], columns, entity, time, budget, key, seed)
    return {"split": parts["split"], "choice": chosen, "review": {k: v for k, v in listed.items() if k != "scores"},
            "scores": listed["scores"], "holdout_index": list(parts["holdout"].index)}


def numeric_columns(frame: pd.DataFrame, exclude: Sequence[Optional[str]] = ()) -> list[str]:
    """The numeric columns a detector may read: not excluded, not an identifier by name or by being unique on every row."""
    out = []
    for c in frame.columns:
        if c in exclude or any(t in str(c).lower() for t in ID_LIKE) or str(c).lower() in ("id", "index", "unnamed: 0"):
            continue
        unique_count = pd.api.types.is_integer_dtype(frame[c]) and frame[c].nunique() == len(frame)  # a row number, not a measure
        if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c]) and frame[c].nunique() > 2 and not unique_count:
            out.append(c)
    return out


__all__ = ["DETECTORS", "SHAPES", "Features", "fit_detector", "plant", "split", "choose", "review", "evaluate", "budget_rows",
           "numeric_columns"]
