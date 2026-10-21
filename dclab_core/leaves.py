"""The leaves of a programme (package 24.5): what runs one block of a solution map and hands its scores to the composition.

A leaf never scores the holdout. It fits on the rows before it, chooses what it needs to choose on the validation rows, and returns a score per row for four
sets of rows: ``fit`` and ``validation`` (from a model fitted on ``fit`` alone: this is what the composition is chosen on) and ``train`` and ``holdout`` (from
the same model refitted on ``train`` = ``fit`` + ``validation``: this is what the composition is scored on, once). A leaf that reads another leaf's score (a
classifier that reads the screen's) reads it under the same protocol, so a score it saw in training is never one made with the rows it is scored on.

Three runners, by the approach the router chose (``solutions.CATALOGUE``): ``anomaly`` (a screen: a detector chosen on planted anomalies of the validation rows, no
label), ``rare_event`` / ``binary`` (a classifier on the confirmed cases, reading the scores of the leaves it depends on) and ``entity_tables`` (accounts that
share a phone or a device and the people who serve them, from a related table, no label). An approach with no runner here is a blocker the programme says, not a
leaf silently skipped.

Rows with no measure at all (every numeric column zero or empty: a cancelled ride) are not scored by a screen: they rank lowest, and the leaf says how many.
Only column names, counts and scores leave a leaf; no value of a row is kept in its record.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from dclab_core.times import read_times

from dclab_core import anomaly
from dclab_core.compose import MIN_POSITIVES, budget_rows, recall_at

PERIODS = ("fit", "validation", "train", "holdout")
COORDINATE = re.compile(r"(^|_)(lat|lon|lng|latitude|longitude)(_|$)|(lat|lon|lng)$", re.I)
MIN_FIT_POSITIVES = 8
LINK_MAX_GROUP = 30       # a phone shared by more accounts than this is a company's, not a ring's
LINK_MIN_SHARE = 0.002    # at least this share of the accounts must share the value with another for the column to be a link
LINK_MAX_SHARE = 0.6
HISTORY_FOR_FULL_WEIGHT = 5


class PartitionError(ValueError):
    pass


@dataclass
class Parts:
    """The rows of each period: ``fit`` and ``validation`` inside ``train``, then the ``holdout`` (the latest)."""

    fit: pd.Index
    validation: pd.Index
    holdout: pd.Index
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def train(self) -> pd.Index:
        return self.fit.union(self.validation, sort=False)

    def of(self, period: str) -> pd.Index:
        return {"fit": self.fit, "validation": self.validation, "train": self.train, "holdout": self.holdout}[period]


@dataclass
class LeafScores:
    """One leaf's output: a score per row for each period, and the record of what it did (``record`` is a dict the programme turns into a ``LeafRun``)."""

    id: str
    scores: dict[str, pd.Series]
    record: dict[str, Any]


@dataclass
class Context:
    frame: pd.DataFrame
    parts: Parts
    time: Optional[str] = None
    entities: Sequence[str] = ()
    labels: Optional[pd.Series] = None              # confirmed cases, 1; every other row 0 (unknown, not clean)
    known_at: Optional[pd.Series] = None            # when each confirmed case was confirmed (NaT: not a case, or no date): a model never learns a case before then
    tables: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    exclude: Sequence[str] = ()                     # columns no leaf reads: identifiers, the label, anything known only afterwards
    budget: Optional[float] = None
    seed: int = 0
    done: dict[str, LeafScores] = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------------------------------ the shared split

def _group_split(frame: pd.DataFrame, key: str, share: float, seed: int, exclude: Optional[pd.Index] = None) -> pd.Index:
    """Whole groups of ``key`` taken at random until they hold ``share`` of the rows."""
    rows = frame if exclude is None else frame.loc[frame.index.difference(exclude, sort=False)]
    sizes = rows.groupby(key).size()
    order = np.random.default_rng(seed).permutation(len(sizes))
    taken, total = [], 0
    for i in order:
        if total >= share * len(rows):
            break
        taken.append(sizes.index[i])
        total += int(sizes.iloc[i])
    return rows.index[rows[key].isin(taken)]


def partition(frame: pd.DataFrame, *, moment: Optional[str] = None, keys: Sequence[str] = (), share: float = 0.2, validation_share: float = 0.2,
              seed: int = 0) -> Parts:
    """The three sets of rows from a split design: by the moment (the holdout is the latest days, a validation period before it, no day cut in two), or whole
    groups of the first key, or a seeded draw. The same call gives the same sets. A design that names a moment is never answered by another split: when the column is
    missing, a number, not dates, or spans fewer than three days, it is refused with the reason (a random split would let the future into the past without saying so).
    Leaves fit once on ``fit`` and are judged once on ``validation``; they run no folds inside, so the entities that repeat across periods (a time split by entity)
    are the time split's own, and the design's note says so."""
    if moment:
        if moment not in frame.columns:
            raise PartitionError(f"the split follows {moment!r}, which the table does not have")
        days = anomaly._days(frame, moment)
        if days is None or days.nunique() < 3:
            raise PartitionError(f"the split follows {moment!r}, which does not read as dates over at least three days (a number, text that is not a date, or a single day): "
                                 "give the moment as dates")
        parts = anomaly.split(frame, moment, holdout=share, validation=validation_share, seed=seed)
        out = Parts(fit=parts["fit"].index, validation=parts["validation"].index, holdout=parts["holdout"].index, info={**parts["split"], "moment": moment})
        lost = len(frame) - len(out.fit) - len(out.validation) - len(out.holdout)
        if lost > 0.05 * len(frame):
            raise PartitionError(f"{lost} rows ({lost / len(frame):.0%}) have no usable {moment}: they belong to no period, and that is more than a split may leave out")
        out.info["no_period"] = int(lost)
        _check(out)
        return out
    if keys and keys[0] in frame.columns:
        holdout = _group_split(frame, keys[0], share, seed)
        validation = _group_split(frame, keys[0], validation_share, seed + 1, exclude=holdout)
        fit = frame.index.difference(holdout, sort=False).difference(validation, sort=False)
        out = Parts(fit=fit, validation=validation, holdout=holdout, info={"by": "group", "key": keys[0], "no_period": 0})
        _check(out)
        return out
    idx = np.random.default_rng(seed).permutation(frame.index.to_numpy())
    h = int(len(idx) * share)
    holdout, rest = pd.Index(idx[:h]), idx[h:]
    v = int(len(rest) * validation_share)
    out = Parts(fit=pd.Index(rest[v:]), validation=pd.Index(rest[:v]), holdout=holdout, info={"by": "row", "no_period": 0})
    _check(out)
    return out


def _check(parts: Parts) -> None:
    if not (len(parts.fit) and len(parts.validation) and len(parts.holdout)):
        raise PartitionError("the table is too small or too short in time to hold a fit, a validation and a holdout period")


def mark_cases(frame: pd.DataFrame, cases: pd.DataFrame, key: Sequence[str]) -> pd.Series:
    """1 for the rows whose ``key`` (one column or several) is in the confirmed cases, 0 for every other row: unknown, never clean. Keys are compared as the analysis
    compares them (``analyse._key_text``: 921270 and 921270.0 are the same case)."""
    from dclab_core.analyse import _key_text

    key = list(key)
    missing = [c for c in key if c not in frame.columns or c not in cases.columns]
    if not key or missing:
        raise ValueError(f"the key {', '.join(missing) or '(none)'} is not in both the table and the confirmed cases")
    wanted = cases[key].dropna().apply(_key_text)
    have = frame[key].apply(_key_text)
    if len(key) == 1:
        return have[key[0]].isin(set(wanted[key[0]])).astype(int).rename("confirmed")
    index = pd.MultiIndex.from_frame(have)
    return pd.Series(index.isin(pd.MultiIndex.from_frame(wanted)).astype(int), index=frame.index, name="confirmed")


CONFIRM_NAME = re.compile(r"(confirm|verif|resolv|decid|decision|closed|reviewed|investigat)", re.I)
DATE_NAME = re.compile(r"(date|time|_at$|^at_|when)", re.I)


def confirmation_column(cases: pd.DataFrame, key: Sequence[str]) -> Optional[str]:
    """The column of the confirmed cases that says when each was confirmed: a column other than the key that reads as dates (more than 90% of it), preferring a name that
    says confirmed or verified. None when the file has none: the cases then carry no date, and the limit is said."""
    found = []
    for c in cases.columns:
        if c in key or pd.api.types.is_numeric_dtype(cases[c]) and not pd.api.types.is_datetime64_any_dtype(cases[c]):
            continue
        if not (CONFIRM_NAME.search(str(c)) or DATE_NAME.search(str(c))):
            continue
        when = read_times(cases[c])
        if when.notna().mean() > 0.9:
            found.append((0 if CONFIRM_NAME.search(str(c)) else 1, str(c)))
    return min(found)[1] if found else None


def case_dates(frame: pd.DataFrame, cases: pd.DataFrame, key: Sequence[str], column: str) -> pd.Series:
    """When each row's case was confirmed (the earliest, if the file lists it twice); NaT for a row that is no case, or a case with no readable date."""
    from dclab_core.analyse import _key_text

    key = list(key)
    when = read_times(cases[column]).reset_index(drop=True)
    wanted = cases[key].reset_index(drop=True).apply(_key_text)
    have = frame[key].apply(_key_text)
    if len(key) == 1:
        by_key = pd.Series(when.to_numpy(), index=wanted[key[0]].to_numpy()).groupby(level=0).min()
        return have[key[0]].map(by_key).rename("confirmed_at")
    by_key = pd.Series(when.to_numpy(), index=pd.MultiIndex.from_frame(wanted)).groupby(level=list(range(len(key)))).min()
    return pd.Series(by_key.reindex(pd.MultiIndex.from_frame(have)).to_numpy(), index=frame.index, name="confirmed_at")


def confirmed_labels(frame: pd.DataFrame, cases: pd.DataFrame, key: str) -> pd.Series:
    """1 for the rows whose ``key`` is in the confirmed cases, 0 for every other row: unknown, never clean."""
    return mark_cases(frame, cases, [key])


def measure_columns(frame: pd.DataFrame, exclude: Sequence[str] = ()) -> list[str]:
    """The numeric columns a leaf may read: not an identifier, not excluded, not a bare coordinate (a place is read through what it does, not where it is)."""
    skip = set(exclude)
    return [c for c in anomaly.numeric_columns(frame, exclude=list(skip)) if not COORDINATE.search(str(c))]


def _no_measure(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return (frame[list(columns)].apply(pd.to_numeric, errors="coerce").fillna(0) == 0).all(axis=1)


# ------------------------------------------------------------------------------------------------------------------------ the screen

def run_anomaly(block: Any, ctx: Context) -> LeafScores:
    cols = measure_columns(ctx.frame, [*ctx.exclude, *ctx.entities, *( [ctx.time] if ctx.time else [] )])
    if not cols:
        raise ValueError("no numeric column to screen")
    empty = _no_measure(ctx.frame, cols)

    def rows(period: str) -> pd.DataFrame:
        idx = ctx.parts.of(period)
        return ctx.frame.loc[idx][~empty.loc[idx].to_numpy()]

    chosen = anomaly.choose({"fit": rows("fit"), "validation": rows("validation")}, cols, None, ctx.time, ctx.budget, seed=ctx.seed)
    out: dict[str, pd.Series] = {}
    for fit_on, scored in (("fit", ("fit", "validation")), ("train", ("train", "holdout"))):
        base = rows(fit_on)
        feats = anomaly.Features(cols, None, ctx.time).fit(base)
        Zb = feats.transform(base)
        det = anomaly.fit_detector(chosen["chosen"], Zb, None, ctx.seed)
        for period in scored:
            part = base if period == fit_on else rows(period)
            Z = Zb if period == fit_on else anomaly.with_history(feats, base, part)
            s = pd.Series(det(Z, None), index=part.index, dtype=float)
            idx = ctx.parts.of(period)
            floor = float(s.min()) - 1.0
            out[period] = s.reindex(idx).fillna(floor)
    left = int(empty.loc[ctx.frame.index].sum())
    record = {"approach": "anomaly", "chosen": chosen["chosen"], "validation": {k: v["mean"] for k, v in chosen["validation"].items()},
              "columns": list(cols), "left_out": f"{left} row(s) with no measure at all (every one of {', '.join(cols[:4])} is zero or empty): not scored, ranked lowest" if left else None,
              "notes": [f"Detector chosen on planted anomalies of the validation rows: {chosen['chosen_by']}."],
              "limits": ["Unusual is not fraud. The score says how unlike the rest a row looks in these columns; a fraud that leaves no trace in them is not found."]}
    return LeafScores(block.id, out, record)


# ------------------------------------------------------------------------------------------------------------------------ the classifier

def _read_ids(block: Any, ctx: Context, reads: Sequence[str]) -> list[str]:
    return [r for r in reads if r in ctx.done]


def _design(ctx: Context, cols: Sequence[str], reads: Sequence[str], period: str) -> pd.DataFrame:
    """The columns a classifier reads on the rows of ``period``: the measures, and the scores of the leaves it reads, from the model of the same protocol."""
    idx = ctx.parts.of(period)
    X = ctx.frame.loc[idx, list(cols)].apply(pd.to_numeric, errors="coerce")
    for r in reads:
        X[f"read_{r}"] = ctx.done[r].scores[period].reindex(idx)
    return X.fillna(0.0)


def _models(seed: int) -> dict[str, Callable[[], Any]]:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {"logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")),
            "gradient_boosting": lambda: GradientBoostingClassifier(random_state=seed, n_estimators=120, max_depth=3)}


def _period_start(ctx: Context, period: str) -> Optional[pd.Timestamp]:
    t = _times(ctx.frame, ctx.time)
    return None if t is None else t.loc[ctx.parts.of(period)].min()


def _known_by(ctx: Context, y: pd.Series, start: Optional[pd.Timestamp], rows: pd.Index) -> tuple[pd.Series, int]:
    """The labels a model could have known when the period starting at ``start`` began: a case confirmed after that is, to the model, a ride nobody has marked. A case with
    no date counts as known. Returns the labels and how many of the cases among ``rows`` (the rows the model learns from) were held back."""
    if ctx.known_at is None or start is None:
        return y, 0
    at = ctx.known_at.reindex(y.index)
    late = (y == 1) & at.notna() & (at >= start)
    return y.where(~late, 0), int(late.loc[rows].sum())


def run_classifier(block: Any, ctx: Context, reads: Sequence[str] = ()) -> LeafScores:
    if ctx.labels is None:
        raise ValueError("no confirmed cases: a classifier has nothing to learn from")
    y = ctx.labels.reindex(ctx.frame.index).fillna(0).astype(int)
    y_fit, held_a = _known_by(ctx, y, _period_start(ctx, "validation"), ctx.parts.fit)   # protocol A learns from the fit rows, with what was known when validation began
    y_train, held_b = _known_by(ctx, y, _period_start(ctx, "holdout"), ctx.parts.train)     # protocol B learns from the train rows, with what was known when the holdout began
    fit_pos = int(y_fit.loc[ctx.parts.fit].sum())
    if fit_pos < MIN_FIT_POSITIVES:
        raise ValueError(f"only {fit_pos} confirmed case(s) in the fit rows" + (" known by the time validation starts" if held_a else "") + f" (at least {MIN_FIT_POSITIVES} are needed)")
    reads = _read_ids(block, ctx, reads)
    cols = measure_columns(ctx.frame, [*ctx.exclude, *ctx.entities, *([ctx.time] if ctx.time else []), y.name or "confirmed"])
    X = {(p, proto): _design(ctx, cols, reads, p) for p, proto in (("fit", "A"), ("validation", "A"), ("train", "B"), ("holdout", "B"))}
    k_val = budget_rows(len(ctx.parts.validation), ctx.budget)
    yv = y.loc[ctx.parts.validation].to_numpy()                              # judged on the cases that were eventually confirmed, as the holdout is
    judged: dict[str, float] = {}
    for name, make in _models(ctx.seed).items():
        m = make().fit(X[("fit", "A")], y_fit.loc[ctx.parts.fit])
        judged[name] = round(recall_at(m.predict_proba(X[("validation", "A")])[:, 1], yv, k_val), 6)
    order = list(judged)
    best = max(judged, key=lambda n: (judged[n], -order.index(n)))          # a tie goes to the simpler model
    out: dict[str, pd.Series] = {}
    for fit_on, scored in (("fit", ("fit", "validation")), ("train", ("train", "holdout"))):
        proto = "A" if fit_on == "fit" else "B"
        m = _models(ctx.seed)[best]().fit(X[(fit_on, proto)], (y_fit if proto == "A" else y_train).loc[ctx.parts.of(fit_on)])
        for period in scored:
            out[period] = pd.Series(m.predict_proba(X[(period, proto)])[:, 1], index=ctx.parts.of(period), dtype=float)
    limits = ["Only some frauds were confirmed, and more of the visible kind: an unmarked row is unknown, not clean, and the model learns what was easy to confirm.",
              f"{fit_pos} confirmed case(s) in the fit rows."]
    if ctx.known_at is not None:
        limits.append(f"A model learns a case only from the day it was confirmed: {held_a} case(s) of the fit rows were held back from the model that scores validation, "
                      f"{held_b} of the train rows from the one that scores the holdout.")
    else:
        limits.append("The confirmed cases carry no date of confirmation: a model may learn a case that was only confirmed after the period it scores began.")
    record = {"approach": block.approach, "chosen": best, "validation": judged, "reads": list(reads), "columns": [*cols, *[f"read_{r}" for r in reads]],
              "notes": [f"Model chosen on the validation rows by recall at the review budget ({k_val} rows); a tie to the simpler model."] +
                       (["Reads the score of " + ", ".join(reads) + " from the same protocol: a score it saw in training was made by a model that had not seen the rows it is judged on, "
                         "except on the rows the model itself was fitted on."] if reads else []),
              "limits": limits}
    return LeafScores(block.id, out, record)


# ------------------------------------------------------------------------------------------------------------------------ the links

SIGNUP_NAME = re.compile(r"(sign.?up|regist|joined|created|opened|enrol)", re.I)


def _times(frame: pd.DataFrame, column: Optional[str]) -> Optional[pd.Series]:
    """A column as timestamps, or None when it is not one: missing, a number (read as nanoseconds it would put every row on one day), or mostly not dates."""
    if not column or column not in frame.columns:
        return None
    raw = frame[column]
    if pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_datetime64_any_dtype(raw):
        return None
    t = read_times(raw)
    return t if t.notna().mean() > 0.9 else None


def link_columns(table: pd.DataFrame, key: str) -> list[str]:
    """The columns of a related table whose value some accounts share with a few others (a phone, a device), and not a column everyone shares (a city) or nobody does (an
    identifier): between LINK_MIN_SHARE and LINK_MAX_SHARE of the rows share a value in a group of two up to LINK_MAX_GROUP. A value shared by more accounts than that (a
    placeholder, a company's switchboard) links nobody and is left out value by value; it does not hide the column's real groups."""
    out = []
    for c in table.columns:
        if c == key or pd.api.types.is_float_dtype(table[c]) or pd.api.types.is_bool_dtype(table[c]):
            continue
        if re.search(r"(date|time|_at$|created|signup)", str(c), re.I):
            continue
        size = table.groupby(c)[c].transform("size")
        shared = float(((size >= 2) & (size <= LINK_MAX_GROUP)).mean())
        if LINK_MIN_SHARE <= shared <= LINK_MAX_SHARE:
            out.append(str(c))
    return out


def signup_column(table: pd.DataFrame, key: str) -> Optional[str]:
    """The column that says when each account of a related table was opened (a name that says so, reading as dates), or None."""
    for c in table.columns:
        if c != key and SIGNUP_NAME.search(str(c)) and _times(table, str(c)) is not None:
            return str(c)
    return None


def find_table(frame: pd.DataFrame, tables: Mapping[str, pd.DataFrame]) -> Optional[tuple[str, str]]:
    """The related table and the column that joins it: a column of the table that is unique in it and holds (nearly) every value of the same-named column of the frame."""
    for name, t in tables.items():
        for c in t.columns:
            if c in frame.columns and t[c].is_unique and frame[c].isin(set(t[c])).mean() >= 0.9:
                return name, c
    return None


def _as_of_share(frame: pd.DataFrame, entity: str, partner: str, t: pd.Series) -> tuple[pd.Series, pd.Series]:
    """For each row: the share of the entity's rides up to and including this one that went to the row's partner, and how many rides that is. Rides are ordered by their
    moment (ties by their place in the table), so a ride never reads a later one, not even one of the same day. A row with no usable moment gets a share of 0."""
    n = len(frame)
    order = np.lexsort((np.arange(n), t.to_numpy()))        # a missing moment sorts last
    ordered = frame.iloc[order][[entity, partner]]
    pair_n = ordered.groupby([entity, partner]).cumcount().to_numpy() + 1
    ent_n = ordered.groupby(entity).cumcount().to_numpy() + 1
    share, count = np.empty(n), np.empty(n)
    share[order], count[order] = pair_n / ent_n, ent_n
    unknown = t.isna().to_numpy()
    share[unknown], count[unknown] = 0.0, 0.0
    return pd.Series(share, index=frame.index), pd.Series(count, index=frame.index)


def _shared_as_of(frame: pd.DataFrame, key: str, table: pd.DataFrame, links: Sequence[str], signup: Optional[str], t: Optional[pd.Series]) -> pd.Series:
    """For each ride: the most accounts that share one contact value with the ride's account, counted as of the ride's moment when the table says when accounts were opened
    (an account opened later is not yet part of the group), else as the table stands. A value shared by more than LINK_MAX_GROUP accounts counts as one."""
    best = pd.Series(1.0, index=frame.index)
    by_account = table.set_index(key)
    opened = read_times(table[signup]).fillna(pd.Timestamp("1900-01-01")).to_numpy() if signup else None
    for c in links:
        value = frame[key].map(by_account[c])
        if opened is None or t is None:
            size = table.groupby(c)[c].transform("size")
            static = pd.Series(np.where(size <= LINK_MAX_GROUP, size, 1), index=table[key].to_numpy())
            best = np.maximum(best, frame[key].map(static).fillna(1.0))
            continue
        acc = pd.DataFrame({"_v": table[c].to_numpy(), "_s": opened}).dropna(subset=["_v"]).sort_values("_s", kind="stable")
        acc["_n"] = acc.groupby("_v").cumcount() + 1
        acc = acc[acc.groupby("_v")["_n"].transform("max") <= LINK_MAX_GROUP]
        rides = pd.DataFrame({"_v": value.to_numpy(), "_t": t.to_numpy(), "_i": np.arange(len(frame))}).dropna(subset=["_v", "_t"]).sort_values("_t", kind="stable")
        got = pd.merge_asof(rides, acc, left_on="_t", right_on="_s", by="_v", direction="backward")
        size = pd.Series(got["_n"].to_numpy(), index=got["_i"].to_numpy()).reindex(np.arange(len(frame)))
        best = np.maximum(best, size.fillna(1.0).to_numpy())
    return pd.Series(np.asarray(best, dtype=float), index=frame.index)


def run_links(block: Any, ctx: Context) -> LeafScores:
    found = find_table(ctx.frame, ctx.tables)
    if not found:
        raise ValueError("no related table joins the rides to their accounts: attach the table of accounts by name")
    name, key = found
    table = ctx.tables[name]
    links = link_columns(table, key)
    if not links:
        raise ValueError(f"no column of {name} is shared by a few accounts (a phone, a device): nothing links them")
    partners = [e for e in ctx.entities if e != key and e in ctx.frame.columns]
    if not partners:
        raise ValueError("no second entity (the people who serve the accounts) to link the accounts to")
    t = _times(ctx.frame, ctx.time)
    if ctx.time and t is None:
        raise ValueError(f"the moment {ctx.time!r} does not read as dates: the links are counted as of each ride's moment, which needs them")
    signup = signup_column(table, key)
    shared = _shared_as_of(ctx.frame, key, table, links, signup, t)
    best = pd.Series(0.0, index=ctx.frame.index)
    for partner in partners:
        if t is None:  # the design names no moment: the rides are not ordered, and the score reads the whole table (said)
            history = ctx.frame.groupby(key)[key].transform("size").astype(float)
            share = ctx.frame.groupby([key, partner])[key].transform("size") / history
        else:
            share, history = _as_of_share(ctx.frame, key, partner, t)
        best = np.maximum(best, share * np.minimum(1.0, history / HISTORY_FOR_FULL_WEIGHT))
    as_of = "the ride's own moment" if t is not None else "the whole table (the design names no moment)"
    score = (shared - 1.0).clip(lower=0.0) * best
    out = {p: score.loc[ctx.parts.of(p)] for p in PERIODS}
    linked = int((shared >= 2).sum())
    group_note = (f"how many accounts share the value, counted as of the ride (the table says when its accounts were opened: {signup})" if signup and ctx.time
                  else "how many accounts share the value, as the table stands now (it does not say when its accounts were opened, so an account opened later is counted for earlier rides)")
    record = {"approach": "entity_tables", "chosen": "shared contact x concentration on one partner", "columns": [*links, *partners],
              "validation": {}, "notes": [f"Accounts are linked by {', '.join(links)} of the table {name}; the score is {group_note}, times how much of the account's rides "
                                         f"so far went to the same {' or '.join(partners)} (history counted to {as_of}: no later ride is read, not even one of the same day). No label is used."],
              "left_out": (f"{int(t.isna().sum())} row(s) with no usable moment score 0" if t is not None and t.isna().any() else None),
              "limits": ["Families and offices share a phone honestly: one shared value alone is not a ring, so the score needs the rides to concentrate on one person too.",
                         f"{linked} row(s) belong to an account that shares a value with another."]}
    return LeafScores(block.id, out, record)


RUNNERS: dict[str, str] = {"anomaly": "anomaly", "rare_event": "classifier", "binary": "classifier", "entity_tables": "links"}


def run_leaf(block: Any, ctx: Context, reads: Sequence[str] = ()) -> LeafScores:
    """Run the block's approach, or say there is no runner for it."""
    kind = RUNNERS.get(block.approach or "")
    if kind is None:
        raise ValueError(f"the programme has no runner for the approach {block.approach!r} yet")
    if kind == "anomaly":
        return run_anomaly(block, ctx)
    if kind == "classifier":
        return run_classifier(block, ctx, reads)
    return run_links(block, ctx)


__all__ = ["PERIODS", "Parts", "LeafScores", "Context", "PartitionError", "RUNNERS", "partition", "mark_cases", "confirmed_labels", "measure_columns", "link_columns", "signup_column", "find_table", "confirmation_column", "case_dates",
           "run_anomaly", "run_classifier", "run_links", "run_leaf"]
