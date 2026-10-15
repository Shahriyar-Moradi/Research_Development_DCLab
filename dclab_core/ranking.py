"""Ranking and recommendation (package 16.5): which items to show each user, from what users did before a moment.

Interactions (user, item, time, event) up to the recommendation moment are all a model may read; the items it may
recommend are those created by the moment. :func:`evaluate` works at two origins, each a moment followed by a window:

- the validation origin (one window before the moment): every model is scored there and the best one, not popularity,
  is chosen before the moment's window is looked at;
- the moment itself: every model scored once on what users did in the window after it (the relevant events: a
  purchase, or a set of events such as a purchase and a high rating).

At an origin the models are the most popular items, the items popular lately (two weeks before the origin), matrix
factorisation of the user-item matrix (truncated SVD of log counts; popularity for users with no history), and a
gradient-boosted ranker, pointwise, on candidates (the union of the others' top items and the user's own items),
trained on the window before the origin with features computed before that window. Item-table columns reach the
ranker only as fixed attributes: a column that tracks interactions is a snapshot the table holds at one moment only, so
it is kept out of the ranker (it would hold the training window's own labels). Scores: NDCG@k and recall@k (each
against popularity, with a paired 90% bootstrap interval over users), catalogue coverage and novelty, for users with
and without a history.

:func:`item_check` finds item columns that know what happened after the moment (beyond everything the log knew of an
item then, they still follow its later counts, on the raw or the log scale) and the items created after the moment.
Every run records the last interaction each part read. Splits by held-out users are not built: every user is cut at the
same moment. Pure: numpy, pandas and scikit-learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

RECENT_DAYS = 14
CANDIDATES = 50
FACTORS = 16
MAX_CELLS = 20_000_000  # users x items for the dense matrix (scipy is not a dependency of dclab_core)
TRACKS = 0.5  # |correlation| with interaction counts above which an item column is a count, not a fixed attribute


def _r(x: Any, digits: int = 4) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if np.isfinite(v) else None


def _set(relevant: Union[str, Sequence[str]]) -> set:
    return {relevant} if isinstance(relevant, str) else set(relevant)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else 0.0


def item_check(items: pd.DataFrame, interactions: pd.DataFrame, *, user: str, item: str, time: str, event: str,
               relevant: Union[str, Sequence[str]], moment: str, created: Optional[str] = None) -> dict[str, Any]:
    """Item columns written with the future, and items that did not exist at the moment. What the log knew of an item at
    the moment is its counts of each event (and of all), over its whole past and its last 7, 14 and 30 days, and its age.
    A column is counted later when what it holds beyond that still follows the counts after the moment (partial
    correlation at least 0.3, for some event): no column computed by the moment can know them."""
    log = interactions.assign(_t=pd.to_datetime(interactions[time]))
    at = pd.Timestamp(moment)
    groups = {f"{event} = {v}": log[event] == v for v in log[event].dropna().unique()}
    groups["all events"] = pd.Series(True, index=log.index)

    def count(mask: pd.Series) -> np.ndarray:
        return log[mask].groupby(item).size().reindex(items[item], fill_value=0).to_numpy(float)

    known, after = [np.ones(len(items))], {}
    for name, mask in groups.items():
        known.append(count(mask & (log["_t"] <= at)))
        for days in (7, 14, 30):
            known.append(count(mask & (log["_t"] <= at) & (log["_t"] > at - pd.Timedelta(days=days))))
        after[name] = count(mask & (log["_t"] > at))
    if created and created in items.columns:
        known.append((at - pd.to_datetime(items[created])).dt.days.clip(lower=0).to_numpy(float))
    K = np.column_stack(known)

    def beyond(v: np.ndarray) -> np.ndarray:  # what v holds beyond what the log knew at the moment
        return v - K @ np.linalg.lstsq(K, v, rcond=None)[0]

    Klog = np.column_stack([K[:, :1], np.log1p(K[:, 1:])])

    def beyond_log(v: np.ndarray) -> np.ndarray:  # the same on log scale: a later count times a price is a sum there
        return v - Klog @ np.linalg.lstsq(Klog, v, rcond=None)[0]

    after_beyond = {name: (beyond(a), beyond_log(np.log1p(a))) for name, a in after.items()}
    out: dict[str, Any] = {"with_the_future": [], "why": {}, "created_after": 0}
    for column in [c for c in items.columns if c != item and pd.api.types.is_numeric_dtype(items[c])]:
        values = pd.to_numeric(items[column], errors="coerce").fillna(0).to_numpy(float)
        rest = beyond(values)
        if np.std(values) == 0 or np.std(rest) < 1e-9 * np.std(values):
            continue  # nothing beyond what the log knew: computed by the moment
        rest_log = beyond_log(np.log1p(values - values.min()))
        name, r = max(((n, max(_corr(rest, a), _corr(rest_log, al))) for n, (a, al) in after_beyond.items()), key=lambda t: t[1])
        if r >= 0.3:
            out["with_the_future"].append(column)
            out["why"][column] = (f"beyond everything the log knew of an item at the moment, it still follows the counts of {name} after "
                                  f"the moment (partial correlation {r:.2f}): it was counted later")
    if created and created in items.columns:
        out["created_after"] = int((pd.to_datetime(items[created]) > at).sum())
    return out


def _matrix(history: pd.DataFrame, users: list, items: list, user: str, item: str, event: str, relevant: Union[str, Sequence[str]]) -> np.ndarray:
    """The user-item matrix of log weighted counts (a relevant event 1, any other 0.3), dense: refused above MAX_CELLS."""
    if len(users) * len(items) > MAX_CELLS:
        raise ValueError(f"{len(users)} users by {len(items)} items is too large for the dense factorisation here (at most {MAX_CELLS} cells)")
    u = pd.Categorical(history[user], categories=users).codes
    i = pd.Categorical(history[item], categories=items).codes
    w = np.where(history[event].isin(_set(relevant)), 1.0, 0.3)
    keep = (u >= 0) & (i >= 0)
    m = np.zeros((len(users), len(items)))
    np.add.at(m, (u[keep], i[keep]), w[keep])
    return np.log1p(m)


class _State:
    """What is known at one moment: the interactions up to it, the items created by it, and each model's scores."""

    def __init__(self, log: pd.DataFrame, at: pd.Timestamp, items: pd.DataFrame, c: dict[str, Any]):
        from sklearn.decomposition import TruncatedSVD

        self.history = log[log["_t"] <= at]
        if self.history.empty:
            raise ValueError(f"no interaction up to {at}: nothing is known at that moment")
        self.read_until = self.history["_t"].max()
        exists = (pd.to_datetime(items[c["created"]]) <= at) if c.get("created") else items[c["item"]].isin(self.history[c["item"]])
        self.items = list(items.loc[exists, c["item"]])
        self.users = sorted(self.history[c["user"]].unique())
        self.uindex = {u: n for n, u in enumerate(self.users)}
        h = self.history[self.history[c["item"]].isin(self.items)]
        weight = np.where(h[c["event"]].isin(c["relevant"]), 1.0, 0.3)
        self.pop = pd.Series(weight, index=h[c["item"]]).groupby(level=0).sum().reindex(self.items, fill_value=0.0).to_numpy()
        recent = (h["_t"] > at - pd.Timedelta(days=RECENT_DAYS)).to_numpy()
        self.recent = pd.Series(weight[recent], index=h.loc[recent, c["item"]]).groupby(level=0).sum().reindex(self.items, fill_value=0.0).to_numpy()
        self.m = _matrix(h, self.users, self.items, c["user"], c["item"], c["event"], c["relevant"])
        k = max(1, min(FACTORS, min(self.m.shape) - 1))
        svd = TruncatedSVD(n_components=k, random_state=0).fit(self.m)
        self.U, self.V = svd.transform(self.m), svd.components_.T
        cats = items.set_index(c["item"]).reindex(self.items)
        self.category = cats[c["category"]].astype(str).to_numpy() if c.get("category") else np.zeros(len(self.items), dtype=str)
        self.age = ((at - pd.to_datetime(cats[c["created"]])).dt.days.to_numpy(float) if c.get("created") else np.zeros(len(self.items)))
        self.item_num = cats[list(c["attributes"])].apply(pd.to_numeric, errors="coerce").to_numpy(float).reshape(len(self.items), -1)

    def scores(self, model: str, u: str) -> np.ndarray:
        if model == "popularity":
            return self.pop
        if model == "recent_popularity":
            return self.recent + 1e-6 * self.pop
        n = self.uindex.get(u)
        if n is None:  # no history: popularity
            return self.pop
        return self.U[n] @ self.V.T + 1e-6 * self.pop

    def features(self, u: str, cand: np.ndarray) -> np.ndarray:
        n = self.uindex.get(u)
        row = self.m[n] if n is not None else np.zeros(len(self.items))
        mf = self.scores("mf", u)
        if n is not None and row.sum() > 0:
            share = pd.Series(row, index=self.category).groupby(level=0).sum()
            cat_share = share.reindex(self.category).to_numpy() / row.sum()
        else:
            cat_share = np.zeros(len(self.items))
        return np.column_stack([np.log1p(self.pop[cand]), np.log1p(self.recent[cand]), mf[cand], row[cand], cat_share[cand],
                                self.age[cand], np.full(len(cand), row.sum()), self.item_num[cand]])

    def candidates(self, u: str) -> np.ndarray:
        picks = [np.argsort(-self.scores(m, u))[:CANDIDATES] for m in ("popularity", "recent_popularity", "mf")]
        n = self.uindex.get(u)
        if n is not None:
            picks.append(np.flatnonzero(self.m[n]))
        return np.unique(np.concatenate(picks))


def _ndcg(ranked: Sequence[str], relevant: set, k: int) -> float:
    """DCG of the top k (gain 1 for a relevant item at rank r: 1/log2(r + 1)) over the best possible DCG."""
    gains = [1.0 / np.log2(r + 2) for r, x in enumerate(ranked[:k]) if x in relevant]
    ideal = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(relevant))))
    return sum(gains) / ideal if ideal else 0.0


def _labels(log: pd.DataFrame, start: pd.Timestamp, days: int, c: dict[str, Any]) -> tuple[dict[str, set], Optional[pd.Timestamp]]:
    window = log[(log["_t"] > start) & (log["_t"] <= start + pd.Timedelta(days=days)) & log[c["event"]].isin(c["relevant"])]
    return window.groupby(c["user"])[c["item"]].apply(set).to_dict(), (window["_t"].max() if len(window) else None)


def _origin(log: pd.DataFrame, items: pd.DataFrame, at: pd.Timestamp, c: dict[str, Any], window_days: int, k: int, seed: int) -> dict[str, Any]:
    """Every model's lists at one origin, scored on the window after it; the ranker trained on the window before it."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    state = _State(log, at, items, c)
    truth, _ = _labels(log, at, window_days, c)
    if not truth:
        raise ValueError(f"no relevant event in the {window_days} days after {at.date()}: nothing to score the lists on")
    before = at - pd.Timedelta(days=window_days)
    ranker, trained = None, {"why_not": None}
    seen = log[log["_t"] <= at]  # the ranker learns on (before, at], from what was known on `before`
    train_truth, labels_until = _labels(seen, before, window_days, c)
    if not (seen["_t"] <= before).any():
        trained["why_not"] = f"no interaction before {before.date()}: the ranker has no earlier window to learn on"
    elif not train_truth:
        trained["why_not"] = f"no relevant event between {before.date()} and {at.date()}: the ranker has nothing to learn"
    else:
        past = _State(seen, before, items, c)
        X, y = [], []
        for u, bought in train_truth.items():
            cand = past.candidates(u)
            X.append(past.features(u, cand))
            y.append(np.isin(np.array(past.items)[cand], list(bought)).astype(int))
        yy = np.concatenate(y)
        if len(np.unique(yy)) < 2:
            trained["why_not"] = "no candidate was bought in the training window: the ranker has nothing to learn"
        else:
            ranker = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed).fit(np.vstack(X), yy)
            trained = {"why_not": None, "features_until": str(past.read_until), "labels_from": str(before), "labels_until": str(labels_until),
                       "rows": int(len(yy))}
    rows = []
    for u, bought in truth.items():
        lists = {m: [state.items[j] for j in np.argsort(-state.scores(m, u), kind="stable")[:k]] for m in ("popularity", "recent_popularity", "mf")}
        if ranker is not None:
            cand = state.candidates(u)
            p = ranker.predict_proba(state.features(u, cand))[:, 1]
            lists["ranker"] = [state.items[j] for j in cand[np.argsort(-p, kind="stable")][:k]]
        for model, ranked in lists.items():
            rows.append({"user": u, "model": model, "history": u in state.uindex, "ndcg": _ndcg(ranked, bought, k),
                         "recall": len(set(ranked) & bought) / len(bought), "items": ranked})
    return {"state": state, "truth": truth, "frame": pd.DataFrame(rows), "ranker": trained}


def _scores(found: dict[str, Any], k: int, seed: int) -> dict[str, Any]:
    state, frame = found["state"], found["frame"]
    share = state.pop / state.pop.sum() if state.pop.sum() else np.full(len(state.items), 1 / max(len(state.items), 1))
    novelty_of = dict(zip(state.items, -np.log2(np.clip(share, 1e-9, 1))))  # rarer items are more novel
    users = sorted(found["truth"])
    rng = np.random.default_rng(seed)
    boot = [rng.integers(0, len(users), len(users)) for _ in range(300)]  # the same resampled users for every model: paired
    base = frame[frame.model == "popularity"].set_index("user").loc[users, "ndcg"].to_numpy()
    out: dict[str, Any] = {}
    for model, g in frame.groupby("model"):
        g = g.set_index("user").loc[users]
        recommended = {x for lst in g["items"] for x in lst}
        diff = g["ndcg"].to_numpy() - base
        out[model] = {
            f"ndcg@{k}": _r(g["ndcg"].mean()), f"recall@{k}": _r(g["recall"].mean()),
            "coverage": _r(len(recommended) / max(len(state.items), 1), 3),  # of the items that existed at the origin
            "novelty": _r(np.mean([novelty_of[x] for lst in g["items"] for x in lst]), 3),
            "with_history": {f"ndcg@{k}": _r(g.loc[g["history"], "ndcg"].mean()), "users": int(g["history"].sum())},
            "without_history": {f"ndcg@{k}": _r(g.loc[~g["history"], "ndcg"].mean()), "users": int((~g["history"]).sum())},
            "against_popularity": {"ndcg_gain": _r(diff.mean()), "ci90": [_r(np.quantile([diff[b].mean() for b in boot], q)) for q in (0.05, 0.95)]},
        }
    return out


def choose(scores: dict[str, Any], k: int) -> Optional[str]:
    """The model to use: the best NDCG@k that is not the popularity baseline it is judged against (None if there is none)."""
    return max((m for m in scores if m != "popularity"), key=lambda m: scores[m][f"ndcg@{k}"] or 0, default=None)


def setup(interactions: pd.DataFrame, items: pd.DataFrame, *, user: str, item: str, time: str, event: str,
          relevant: Union[str, Sequence[str]], moment: str, window_days: int = 14, created: Optional[str] = None,
          category: Optional[str] = None, item_features: Sequence[str] = ()) -> dict[str, Any]:
    """What every step reads: the log with its times, the moment and the validation origin one window before it, and the
    item columns the ranker may read (a column tracking the items' counts up to two windows before the moment is kept
    out and named). Nothing after the moment is read here."""
    log = interactions.assign(_t=pd.to_datetime(interactions[time]))
    at = pd.Timestamp(moment)
    validation_at = at - pd.Timedelta(days=window_days)
    relevant_set = _set(relevant)
    upto = log[(log["_t"] <= validation_at - pd.Timedelta(days=window_days))]
    counts = upto.groupby(item).size().reindex(items[item], fill_value=0).to_numpy(float)
    attributes, kept_out = [], {}
    for column in [f for f in item_features if f in items.columns]:
        r = _corr(pd.to_numeric(items[column], errors="coerce").fillna(0).to_numpy(float), counts)
        if abs(r) >= TRACKS:
            kept_out[column] = (f"it tracks the items' interaction counts (correlation {r:.2f}): the table holds it at one moment only, so in "
                                "the ranker's training window it would hold that window's own labels")
        else:
            attributes.append(column)
    c = {"user": user, "item": item, "event": event, "relevant": relevant_set, "created": created, "category": category, "attributes": attributes}
    return {"log": log, "items": items, "at": at, "validation_at": validation_at, "c": c, "kept_out": kept_out, "moment": moment,
            "window_days": window_days}


def validate(ctx: dict[str, Any], *, k: int = 10, seed: int = 0) -> dict[str, Any]:
    """Every model at the validation origin, from the log up to the moment only: the scores and the model chosen."""
    validation = _origin(ctx["log"][ctx["log"]["_t"] <= ctx["at"]], ctx["items"], ctx["validation_at"], ctx["c"], ctx["window_days"], k, seed)
    scores = _scores(validation, k, seed)
    return {"origin": str(ctx["validation_at"]), "scores": scores, "ranker": validation["ranker"], "chosen": choose(scores, k),
            "chosen_by": f"NDCG@{k} at the validation origin, before the moment's window"}


def score(ctx: dict[str, Any], *, k: int = 10, seed: int = 0) -> dict[str, Any]:
    """Every model scored once on the window after the moment (the holdout), with what it recommended."""
    at, items, c = ctx["at"], ctx["items"], ctx["c"]
    final = _origin(ctx["log"], items, at, c, ctx["window_days"], k, seed)  # the moment's window, read once
    scores = _scores(final, k, seed)
    state = final["state"]
    created = c["created"]
    created_at = pd.to_datetime(items.set_index(c["item"])[created]) if created else None
    lists = {m: g.set_index("user")["items"].to_dict() for m, g in final["frame"].groupby("model")}
    late = {m: int(sum(1 for lst in ls.values() for x in lst if created_at is not None and x in created_at.index and created_at[x] > at))
            for m, ls in lists.items()}
    return {"users_scored": len(final["truth"]), "items_existing": len(state.items), "read_until": str(state.read_until),
            "read_after_moment": bool(state.read_until > at), "ranker_trained_on": final["ranker"], "scores": scores,
            "beats_popularity": {m: (s["against_popularity"]["ci90"][0] or 0) > 0 for m, s in scores.items() if m != "popularity"},
            "created_after_recommended": late, "lists": lists}


def evaluate(interactions: pd.DataFrame, items: pd.DataFrame, *, user: str, item: str, time: str, event: str,
             relevant: Union[str, Sequence[str]], moment: str, window_days: int = 14, k: int = 10, created: Optional[str] = None,
             category: Optional[str] = None, item_features: Sequence[str] = (), seed: int = 0) -> dict[str, Any]:
    """The model chosen at the validation origin (one window before ``moment``), then every model scored once on the
    window after ``moment``. ``item_features``: numeric item columns the ranker may read; run :func:`item_check` first.
    A column among them that tracks interactions is kept out of the ranker and named. ``setup``, ``validate`` and
    ``score`` are its three steps (16.15 B: the engine runs them stage by stage)."""
    ctx = setup(interactions, items, user=user, item=item, time=time, event=event, relevant=relevant, moment=moment,
                window_days=window_days, created=created, category=category, item_features=item_features)
    chosen = validate(ctx, k=k, seed=seed)
    held = score(ctx, k=k, seed=seed)
    return {"moment": moment, "window_days": window_days, "k": k, "relevant": sorted(ctx["c"]["relevant"]), "users_scored": held["users_scored"],
            "items_existing": held["items_existing"], "read_until": held["read_until"], "read_after_moment": held["read_after_moment"],
            "item_features": ctx["c"]["attributes"], "kept_out_of_ranker": ctx["kept_out"],
            "validation": {"origin": chosen["origin"], "scores": chosen["scores"], "ranker": chosen["ranker"]},
            "chosen": chosen["chosen"], "chosen_by": chosen["chosen_by"], "ranker_trained_on": held["ranker_trained_on"], "scores": held["scores"],
            "beats_popularity": held["beats_popularity"], "created_after_recommended": held["created_after_recommended"], "lists": held["lists"]}
