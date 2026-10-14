"""Several tables (package 16.8): targets and features per entity, as of a cutoff time.

A project often holds an entity table (customers), event tables (orders, support tickets) and sometimes a label table,
related by keys. The question is asked about an entity at a moment: "will this customer stop ordering in the 30 days
after the 1st of March?". Every row of the training table is then an (entity, cutoff) pair, and everything about it
must be computed from events before its cutoff; its target from events after it.

- :func:`check_join`: a join that would repeat the left table's rows (the right key is not unique) is refused, with the
  numbers: rows before, rows after, keys repeated.
- :func:`snapshot_rows`: the (entity, cutoff) pairs at regular cutoffs, each entity from its first appearance on.
- :func:`window_features`: per row, from its entity's events strictly before its cutoff: counts and sums over windows
  (7, 30 and 90 days by default), days since the last event, and the trend (the last 30 days against the 30 before).
  Each row records its cutoff and the last event time it read.
- :func:`window_target`: per row, whether an event of the kind named happens in the horizon after its cutoff (or does
  not, for churn); a row whose horizon runs past the data's end is left out, not called negative.
- :func:`label_join`: a label table by key (15.2) is the same thing with the label given: joined by key, checked for
  repeats, its rows without a label left out and counted.

Pure: numpy and pandas.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd


class DuplicatingJoin(ValueError):
    """A join that would repeat rows of the left table."""


def _when(values: Any) -> pd.Series:
    """Times as naive UTC timestamps (a time zone is converted, not dropped)."""
    t = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    return t.dt.tz_localize(None)


def check_join(left: pd.DataFrame, right: pd.DataFrame, key: str, *, name: str = "the right table") -> dict[str, Any]:
    """Whether joining ``right`` onto ``left`` by ``key`` keeps one row per left row; raises with the numbers if not. The
    count is the join's own (a missing key matches a missing key, as pandas joins do)."""
    counts = right[key].value_counts(dropna=False)
    repeated = counts[counts > 1]
    after = int(len(left[[key]].merge(right[[key]], on=key, how="left")))
    numbers = {"rows_before": int(len(left)), "rows_after": after, "keys_repeated": int(len(repeated)),
               "worst_key_rows": int(repeated.max()) if len(repeated) else 1}
    if after != len(left):
        raise DuplicatingJoin(f"joining {name} by {key!r} would turn {len(left)} rows into {after}: {len(repeated)} keys appear more than "
                              f"once (one of them {numbers['worst_key_rows']} times). Aggregate it per {key!r} first, or join it as events.")
    return numbers


def snapshot_rows(events: pd.DataFrame, *, key: str, time: str, cutoffs: Sequence[str]) -> pd.DataFrame:
    """(entity, cutoff) rows: every entity with an event strictly before a cutoff, at each cutoff (each cutoff once)."""
    first = events.assign(_t=_when(events[time]).to_numpy()).groupby(key)["_t"].min()
    rows = [pd.DataFrame({key: first.index[first < c], "cutoff": c}) for c in sorted({pd.Timestamp(c) for c in cutoffs})]
    return pd.concat(rows, ignore_index=True)


def _pairs(rows: pd.DataFrame, events: pd.DataFrame, key: str, time: str, extra: Sequence[str] = ()) -> pd.DataFrame:
    """Each row's events, by row position (``_row``), with their times."""
    ev = events[[key, *extra]].assign(_t=_when(events[time]).to_numpy())
    return rows[[key, "cutoff"]].reset_index(drop=True).assign(_row=np.arange(len(rows))).merge(ev, on=key, how="left")


def window_features(rows: pd.DataFrame, events: pd.DataFrame, *, key: str, time: str, prefix: str, sums: Sequence[str] = (),
                    windows: Sequence[int] = (7, 30, 90)) -> pd.DataFrame:
    """Per row, from its entity's events strictly before its cutoff: counts and sums per window, days since the last
    event, the trend (events in the last 30 days minus those in the 30 before), and the last event time read (for the
    check: it must be before the cutoff)."""
    merged = _pairs(rows, events, key, time, sums)
    g = merged[merged["_t"] < merged["cutoff"]]  # strictly before: an event at the cutoff itself is not known yet
    g = g.assign(_age=((g["cutoff"] - g["_t"]).dt.total_seconds() / 86400).to_numpy())
    n = len(rows)
    out = pd.DataFrame(index=rows.index)

    def per_row(frame, how="size", column=None):
        grouped = frame.groupby("_row")
        values = grouped.size() if how == "size" else grouped[column].sum() if how == "sum" else grouped["_t"].max()
        return values.reindex(range(n), fill_value=0 if how != "max" else None).to_numpy()
    for w in windows:
        inside = g[g["_age"] <= w]
        out[f"{prefix}_count_{w}d"] = per_row(inside)
        for column in sums:
            out[f"{prefix}_{column}_sum_{w}d"] = per_row(inside, "sum", column).astype(float)
    out[f"{prefix}_count_all"] = per_row(g)
    last = pd.to_datetime(pd.Series(per_row(g, "max")))
    out[f"{prefix}_days_since_last"] = ((rows["cutoff"].to_numpy() - last.to_numpy()) / np.timedelta64(1, "D")).astype(float)
    out[f"{prefix}_trend_30d"] = per_row(g[g["_age"] <= 30]) - per_row(g[(g["_age"] > 30) & (g["_age"] <= 60)])
    out[f"_{prefix}_last_read"] = last.to_numpy()
    return out


def window_target(rows: pd.DataFrame, events: pd.DataFrame, *, key: str, time: str, horizon_days: int, data_end: Optional[str] = None,
                  absent: bool = True) -> pd.DataFrame:
    """Per row: 1 when no event (``absent=True``: churn) or some event (``absent=False``) happens in the horizon
    [cutoff, cutoff + horizon), the cutoff's own moment included (the features stop just before it). Rows whose horizon
    passes the data's end are unknown (None), never negative. Each row records the last event time the target read."""
    merged = _pairs(rows, events, key, time)
    end = pd.Timestamp(data_end) if data_end else merged["_t"].max()
    stop = merged["cutoff"] + pd.Timedelta(days=horizon_days)
    inside = merged[(merged["_t"] >= merged["cutoff"]) & (merged["_t"] < stop)]
    n = len(rows)
    happened = inside.groupby("_row").size().reindex(range(n), fill_value=0).to_numpy() > 0
    target = (~happened if absent else happened).astype(float)
    horizon_end = (rows["cutoff"] + pd.Timedelta(days=horizon_days)).to_numpy()
    known = horizon_end <= np.datetime64(end)
    read = inside.groupby("_row")["_t"].max().reindex(range(n)).to_numpy()
    return pd.DataFrame({"target": np.where(known, target, np.nan), "horizon_end": horizon_end, "target_last_read": read}, index=rows.index)


def label_join(rows: pd.DataFrame, labels: pd.DataFrame, *, key: str, label: str, label_time: Optional[str] = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A label table joined by key (15.2's label file as one more table): refused when keys repeat. With ``label_time``
    (when the label was decided) only rows whose cutoff is before it are kept, since a later row's features would read
    behaviour after the outcome; without it, a key with rows at several cutoffs is refused. Rows without a label are
    left out and counted."""
    numbers = check_join(rows, labels, key, name="the label table")
    joined = rows.merge(labels[[key, label, *([label_time] if label_time else [])]], on=key, how="left")
    if label_time:
        decided = _when(joined[label_time]).to_numpy()
        late = ~(joined["cutoff"].to_numpy() < decided)
        numbers["after_the_label"] = int((late & joined[label].notna().to_numpy()).sum())
        joined = joined[~late]
    elif rows.groupby(key)["cutoff"].nunique().max() > 1:
        raise ValueError(f"the label table has no time, and some {key!r} have rows at several cutoffs: a later row would read behaviour "
                         "after the outcome. Give the label's time, or one cutoff per entity.")
    numbers["without_label"] = int(joined[label].isna().sum())
    return joined[joined[label].notna()], numbers


def _dates(s: pd.Series) -> Optional[pd.Series]:
    if pd.api.types.is_numeric_dtype(s):
        return None
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # mixed formats are parsed one by one: fine for a check
        stamps = pd.to_datetime(s, errors="coerce")
    return stamps if stamps.notna().mean() > 0.9 else None


def entity_check(rows: pd.DataFrame, entities: pd.DataFrame, tables: dict[str, dict[str, Any]], *, key: str) -> dict[str, Any]:
    """Columns of the entity table written after a row's cutoff: a date later than the cutoff of the rows it would join
    (on more than 5% of them), or a number that follows (|correlation| at least 0.9, over the entities that have both) a
    total of an event table over the whole log, or the entity's last event time over the whole log (a date kept as a
    number, a "days since the last order" taken at the export)."""
    out: dict[str, Any] = {"after_the_cutoff": [], "why": {}}
    joined = rows[[key, "cutoff"]].merge(entities, on=key, how="left")
    index = entities[key]
    marks: dict[str, pd.Series] = {}
    for name, spec in tables.items():
        ev = spec["frame"].assign(_t=_when(spec["frame"][spec["time"]]).to_numpy())
        marks[f"count of {name}"] = ev.groupby(key).size().reindex(index, fill_value=0).astype(float)
        last = ev.groupby(key)["_t"].max().reindex(index)
        marks[f"last {name} time"] = (last - pd.Timestamp("1970-01-01")).dt.total_seconds()
    for column in [c for c in entities.columns if c != key]:
        stamps = _dates(joined[column])
        if stamps is not None:
            late = float((stamps > joined["cutoff"]).mean())
            if late > 0.05:
                out["after_the_cutoff"].append(column)
                out["why"][column] = f"on {late:.0%} of the rows its date is after the row's cutoff: written later"
            continue
        if not pd.api.types.is_numeric_dtype(entities[column]):
            continue
        values = pd.Series(entities[column].to_numpy(float), index=index)
        for what, mark in marks.items():
            r = values.corr(mark)  # pairwise: an entity missing either is left out, not a reason to pass
            if pd.notna(r) and abs(r) >= 0.9:
                out["after_the_cutoff"].append(column)
                out["why"][column] = f"it follows the entity's {what} over the whole log (correlation {r:.2f}): taken at the end, not at a cutoff"
                break
    return out


def build(entities: pd.DataFrame, tables: dict[str, dict[str, Any]], *, key: str, cutoffs: Sequence[str], target_table: str,
          horizon_days: int, absent: bool = True, windows: Sequence[int] = (7, 30, 90), data_end: Optional[str] = None,
          entity_columns: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """The training table: one row per (entity, cutoff), the entity table's own columns (joined, checked), window
    features from every event table, and the target from ``target_table`` in the horizon after each cutoff.
    ``entity_columns``: the entity columns to keep, decided elsewhere (the holdout keeps exactly what training kept)."""
    first = tables[target_table]
    rows = snapshot_rows(first["frame"], key=key, time=first["time"], cutoffs=cutoffs)
    joins = {"entities": check_join(rows, entities, key, name="the entity table")}
    written_later = entity_check(rows, entities, tables, key=key)
    kept = ([key, *(c for c in entity_columns if c != key)] if entity_columns is not None
            else [c for c in entities.columns if c not in written_later["after_the_cutoff"]])
    table = rows.merge(entities[kept], on=key, how="left")
    last_read, window_columns = [], []
    for name, spec in tables.items():
        feats = window_features(rows, spec["frame"], key=key, time=spec["time"], prefix=name, sums=spec.get("sums", ()), windows=windows)
        last_read.append(feats.pop(f"_{name}_last_read"))
        window_columns += list(feats.columns)
        table = pd.concat([table, feats], axis=1)
    target = window_target(rows, first["frame"], key=key, time=first["time"], horizon_days=horizon_days, absent=absent, data_end=data_end)
    table = pd.concat([table, target], axis=1)
    read = pd.concat(last_read, axis=1).max(axis=1)
    unknown = int(table["target"].isna().sum())
    table = table[table["target"].notna()].reset_index(drop=True)
    return {"table": table, "joins": joins, "entity_columns_kept_out": written_later, "cutoffs": [str(pd.Timestamp(c).date()) for c in cutoffs],
            "rows_left_out_unknown": unknown, "events_after_cutoff_read": int((read >= rows["cutoff"]).sum()),
            "last_read": read, "entity_columns": [c for c in kept if c != key], "window_columns": window_columns}


def evaluate(built: dict[str, Any], *, key: str, seed: int = 0) -> dict[str, Any]:
    """A model on the (entity, cutoff) rows: trained on the earlier cutoffs whose horizon ends by the latest cutoff (a
    training target read after the holdout's cutoff would be the holdout's own future), scored once on the latest;
    ROC-AUC and average precision against a recency-only baseline."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    t = built["table"]
    last = t["cutoff"].max()
    train, test = t[(t["cutoff"] < last) & (t["horizon_end"] <= last)], t[t["cutoff"] == last]
    if train.empty or test["target"].nunique() < 2:
        raise ValueError("too few cutoffs: no training row ends its horizon before the latest cutoff, or the holdout has one outcome")
    features = [c for c in t.columns if c not in (key, "cutoff", "target", "horizon_end", "target_last_read") and pd.api.types.is_numeric_dtype(t[c])]
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=seed).fit(train[features], train["target"])
    p = model.predict_proba(test[features])[:, 1]
    recency = [c for c in features if c.endswith("_days_since_last")][:1]
    base = HistGradientBoostingClassifier(max_iter=100, random_state=seed).fit(train[recency], train["target"]).predict_proba(test[recency])[:, 1]
    y = test["target"].to_numpy()
    read = pd.to_datetime(train["target_last_read"]).max()
    return {"train_rows": int(len(train)), "holdout_rows": int(len(test)), "train_cutoffs": sorted({str(c.date()) for c in train["cutoff"]}),
            "train_horizon_until": str(train["horizon_end"].max().date()), "train_targets_read_until": str(read) if pd.notna(read) else None,
            "holdout_cutoff": str(last.date()), "features": features,
            "positive_rate": round(float(y.mean()), 3),
            "model": {"roc_auc": round(float(roc_auc_score(y, p)), 4), "average_precision": round(float(average_precision_score(y, p)), 4)},
            "recency_only": {"roc_auc": round(float(roc_auc_score(y, base)), 4), "average_precision": round(float(average_precision_score(y, base)), 4)}}


# ---------------------------------------------------------------------------- 16.15 C: in the engine, split / choose / score

def data_end_of(tables: dict[str, dict[str, Any]], target_table: str, declared: Optional[str] = None) -> pd.Timestamp:
    """Where the log ends: the date the person states, else the last event of the target's table."""
    if declared:
        return pd.Timestamp(declared)
    spec = tables[target_table]
    return _when(spec["frame"][spec["time"]]).max()


def split(tables: dict[str, dict[str, Any]], *, cutoffs: Sequence[str], target_table: str, horizon_days: int,
          data_end: Optional[str] = None) -> dict[str, Any]:
    """Which cutoffs train, validate and test, read from dates only (never an outcome): the holdout is the latest cutoff
    whose horizon the log reaches; training cutoffs end their horizon by it; the validation cutoff is the latest of
    those, fitted on the ones whose horizon ends by it. Cutoffs whose horizon passes the holdout's or the log's end are
    left out, and said."""
    end = data_end_of(tables, target_table, data_end)
    horizon = pd.Timedelta(days=horizon_days)
    every = sorted({pd.Timestamp(c) for c in cutoffs})
    known = [c for c in every if c + horizon <= end]
    if len(known) < 3:
        raise ValueError(f"too few cutoffs whose {horizon_days}-day horizon the log reaches (by {end.date()}): at least 3 are needed")
    holdout = known[-1]
    train = [c for c in known if c + horizon <= holdout]
    if len(train) < 2:
        raise ValueError("too few training cutoffs end their horizon before the holdout's: add earlier cutoffs")
    validation = train[-1]
    fit = [c for c in train if c + horizon <= validation]
    if not fit:
        raise ValueError("no cutoff ends its horizon before the validation cutoff: add earlier cutoffs")
    day = lambda c: str(c.date())  # noqa: E731
    return {"holdout": day(holdout), "train": [day(c) for c in train], "validation": day(validation), "fit": [day(c) for c in fit],
            "data_end": str(end.date()), "left_out": [day(c) for c in every if c not in train and c != holdout],
            "why_left_out": "their horizon passes the log's end, or overlaps the holdout's"}


def clip(tables: dict[str, dict[str, Any]], until: str) -> dict[str, dict[str, Any]]:
    """Every event table cut to its events strictly before ``until``: the view the training rows are built from."""
    stop = pd.Timestamp(until)
    out = {}
    for name, spec in tables.items():
        frame = spec["frame"]
        out[name] = {**spec, "frame": frame[_when(frame[spec["time"]]).to_numpy() < np.datetime64(stop)]}
    return out


def training_view(entities: pd.DataFrame, tables: dict[str, dict[str, Any]], parts: dict[str, Any], *, key: str, target_table: str,
                  horizon_days: int, absent: bool = True) -> dict[str, Any]:
    """The training rows at the training cutoffs, built from the events before the holdout's cutoff only: features and
    targets alike (a horizon ending by the holdout's cutoff is read in full there)."""
    return build(entities, clip(tables, parts["holdout"]), key=key, cutoffs=parts["train"], target_table=target_table,
                 horizon_days=horizon_days, absent=absent, data_end=parts["holdout"])


def feature_columns(t: pd.DataFrame, key: str, exclude: Sequence[str] = ()) -> list[str]:
    """The columns a model reads: numeric, neither the key, the cutoff, the target nor its bookkeeping, nor ``exclude``
    (what the leakage audit kept out: forbidden, identifiers, columns the prediction moment rules out)."""
    out = set(exclude)
    return [c for c in t.columns if c not in (key, "cutoff", "target", "horizon_end", "target_last_read") and c not in out
            and pd.api.types.is_numeric_dtype(t[c])]


CANDIDATES = ("hist_gradient_boosting", "logistic_regression", "recency_only")


def _model(name: str, seed: int = 0) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "logistic_regression":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    return HistGradientBoostingClassifier(max_iter=200 if name == "hist_gradient_boosting" else 100, learning_rate=0.06, random_state=seed)


def _columns(name: str, features: list[str]) -> list[str]:
    return [c for c in features if c.endswith("_days_since_last")][:1] if name == "recency_only" else features


def choose(view: dict[str, Any], parts: dict[str, Any], *, key: str, exclude: Sequence[str] = (), seed: int = 0) -> dict[str, Any]:
    """Every candidate fitted on the fit cutoffs' rows and scored by ROC-AUC on the validation cutoff's rows; the best
    is chosen (a tie to the simpler one). The holdout's rows are not built here."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    t = view["table"]
    fit = t[t["cutoff"].isin(pd.to_datetime(parts["fit"]))]
    val = t[t["cutoff"] == pd.Timestamp(parts["validation"])]
    if fit["target"].nunique() < 2 or val["target"].nunique() < 2:
        raise ValueError("the fit or the validation rows hold one outcome only: add cutoffs")
    features = feature_columns(t, key, exclude)
    scores = {}
    for name in CANDIDATES:
        cols = _columns(name, features)
        p = _model(name, seed).fit(fit[cols], fit["target"]).predict_proba(val[cols])[:, 1]
        scores[name] = {"roc_auc": round(float(roc_auc_score(val["target"], p)), 4),
                        "average_precision": round(float(average_precision_score(val["target"], p)), 4)}
    order = list(CANDIDATES)[::-1]  # recency first: the simplest wins a tie
    best = max(scores, key=lambda n: (scores[n]["roc_auc"], order.index(n)))
    return {"validation": scores, "chosen": best, "features": features, "fit_rows": int(len(fit)), "validation_rows": int(len(val)),
            "chosen_by": "the best ROC-AUC on the validation cutoff's rows, fitted on the cutoffs whose horizon ends by it"}


def score(entities: pd.DataFrame, tables: dict[str, dict[str, Any]], view: dict[str, Any], parts: dict[str, Any], chosen: str, *,
          key: str, target_table: str, horizon_days: int, absent: bool = True, exclude: Sequence[str] = (), seed: int = 0) -> dict[str, Any]:
    """The chosen model fitted on every training row and scored once on the holdout cutoff's rows (built from the whole
    log: features from events before the cutoff, the target from the horizon after it), beside recency alone."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    train = view["table"]
    features = feature_columns(train, key, exclude)
    built = build(entities, tables, key=key, cutoffs=[parts["holdout"]], target_table=target_table, horizon_days=horizon_days,
                  absent=absent, data_end=parts["data_end"], entity_columns=view["entity_columns"])  # what training kept, no more
    held = built["table"]
    # a kept column that follows the log after the holdout's cutoff could not be seen from the training period: said
    later = [c for c in built["entity_columns_kept_out"]["after_the_cutoff"] if c in features]
    lost = [c for c in features if c not in held.columns]
    if lost:
        raise ValueError(f"the holdout's rows lack training features: {', '.join(lost)}")
    test = held
    y = test["target"].to_numpy()
    if len(set(y)) < 2:
        raise ValueError("the holdout cutoff's rows hold one outcome only")
    out = {}
    for name in dict.fromkeys([chosen, "recency_only"]):
        cols = _columns(name, features)
        p = _model(name, seed).fit(train[cols], train["target"]).predict_proba(test[cols])[:, 1]
        out[name] = {"roc_auc": round(float(roc_auc_score(y, p)), 4), "average_precision": round(float(average_precision_score(y, p)), 4)}
    seen = set(train[key])
    return {"holdout": out, "holdout_rows": int(len(test)), "train_rows": int(len(train)), "positive_rate": round(float(np.mean(y)), 4),
            "kept_but_written_later_at_the_holdout": later,
            "holdout_entities": int(test[key].nunique()), "holdout_entities_seen_in_training": int(test[key].isin(seen).sum()),
            "beats_recency": out[chosen]["roc_auc"] > out["recency_only"]["roc_auc"] if chosen != "recency_only" else False}
