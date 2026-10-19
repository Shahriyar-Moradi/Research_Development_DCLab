"""Matching: which records are the same thing (package 30.16, the owner's pattern 4: copied bills, the same customer under
two names, one product in two catalogues).

One row of the problem is a pair of records. The method, in its order:

1. **Entities and the split.** The known matches join records into entities (a copy and its original, the two catalogues'
   entries of one product); an entity goes whole to one side, so no record is learned from on one side and scored on the
   other. Records are cut into training, validation and holdout by entity, seeded. In one table every two records of one
   entity match (two copies of one bill are the same bill too).
2. **Exact links first.** The fields marked as keys must all agree for the exact rule to link a pair (a flag on either
   record counts: a bill its seller returned); a link the data proves needs no model, and is the baseline a model must beat.
3. **Blocking.** A part's records are compared with every record the part may see: the training records with each other,
   the validation records with the training and validation ones, the holdout records with every record (a real check
   compares a new record with all the others). Pairs form only among records that agree on the block columns (inside the
   time window), or between a record and its nearest neighbours by its text (character n-grams, TF-IDF, cosine, both
   ways). The share of the part's true pairs blocking keeps is measured: a pair it drops no model can find.
4. **Similarity features** per compared field: a text's TF-IDF cosine and token overlap, a number's relative difference,
   a time's minutes apart, a category's equality, a flag on either and on both records.
5. **Models chosen on the validation records:** the exact rule, the single best field, logistic regression and gradient
   boosting, each fitted on the training pairs and its threshold set on every score of the validation pairs; the best F1
   wins (a tie to the simpler).
6. **The holdout, read once:** the chosen model, as fitted and with its threshold, scores the holdout's pairs; pair
   precision, recall and F1, average precision and blocking's recall, beside every baseline, with a bootstrap over
   entities of the F1 difference against each.

Nothing here reads the holdout's matches before ``score``; ``setup`` uses the matches only to form entities and fix the
split before any pair is formed.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

Kind = Literal["text", "number", "time", "category", "flag"]
PARTS = ("train", "validation", "holdout")
SEES = {"train": ("train",), "validation": ("train", "validation"), "holdout": PARTS}  # the records a part's records are compared with
SHARES = (0.6, 0.2, 0.2)  # of the entities, seeded; unmatched records are cut the same way
MODELS = ("exact", "single_field", "logistic", "boosting")  # simplest first: a tie goes to the earlier
TOP_K = 10           # nearest neighbours by text, per record
MAX_PAIRS = 1_500_000  # candidate pairs per part at most (a block too large is said, never silently cut)
NEGATIVES_PER_POSITIVE = 30  # training pairs: every true one, and up to this many false ones per true one, seeded
MAX_MISSING = 0.05   # known matches whose ids are not records: above this share the matches are refused
BOOTSTRAP = 200


def normal(value: Any) -> str:
    """A text as compared: accents and case folded, punctuation as spaces, one space between words."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _id(value: Any) -> str:
    """An id as text: a whole number read as a float ("12.0") is the same id as "12"."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text[:-2] if re.fullmatch(r"-?\d+\.0", text) else text


def _number(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    m = re.search(r"-?\d[\d,]*\.?\d*", str(value or "").replace(",", ""))
    return float(m.group(0)) if m else float("nan")


class _Sets:
    """Union-find over record keys: the entities the known matches make."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _part_of(key: str, seed: int) -> str:
    u = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
    return PARTS[0] if u < SHARES[0] else PARTS[1] if u < SHARES[0] + SHARES[1] else PARTS[2]


def setup(left: pd.DataFrame, right: Optional[pd.DataFrame] = None, *, id_left: str, id_right: Optional[str] = None,
          fields: Sequence[Mapping[str, Any]], matches: pd.DataFrame, block_equal: Sequence[str] = (), window: Optional[Mapping[str, Any]] = None,
          block_text: Optional[str] = None, top_k: int = TOP_K, seed: int = 0) -> dict[str, Any]:
    """The problem fixed before any pair is formed: the records (one table, or a left and a right one), the compared
    ``fields`` (``{"column", "kind", "right"?, "key"?}``), the known ``matches`` (two columns: a left id and a right id, or
    for one table the duplicate's id and its original's), the blocking (``block_equal`` columns, a time ``window`` of
    ``{"column", "minutes"}``, or ``block_text`` for nearest neighbours by a text field) and each record's part by entity.
    Refused: a repeated id, matches whose ids are mostly not records (the columns swapped?), a part without a match."""
    one = right is None
    right = left if one else right
    id_right = id_left if one else (id_right or id_left)
    L = left.reset_index(drop=True).copy()
    R = right.reset_index(drop=True).copy()
    L["__key"] = "L:" + L[id_left].map(_id)
    R["__key"] = ("L:" if one else "R:") + R[id_right].map(_id)
    if L["__key"].duplicated().any() or (not one and R["__key"].duplicated().any()):
        raise ValueError("a record's id is repeated: each record needs its own id to be matched")
    known = set(L["__key"]) | set(R["__key"])
    a, b = list(matches.columns[:2])
    sets, given, missing = _Sets(), 0, 0
    for x, y in zip(matches[a].map(_id), matches[b].map(_id)):
        ka, kb = "L:" + x, ("L:" if one else "R:") + y
        given += 1
        if ka not in known or kb not in known:
            missing += 1
            continue
        if ka != kb:
            sets.union(ka, kb)
    if given and missing / given > MAX_MISSING:
        raise ValueError(f"{missing} of {given} known matches name an id that is not a record: are the matches' two columns "
                         f"({a}, {b}) the left table's and the right table's ids, in that order?")
    groups: dict[str, list[str]] = {}
    for k in known:
        groups.setdefault(sets.find(k), []).append(k)
    pairs: set[tuple[str, str]] = set()
    for members in groups.values():  # every two records of one entity match (across the two sides when there are two)
        if len(members) < 2:
            continue
        ls, rs = ([m for m in members if m.startswith("L:")], [m for m in members if m.startswith("R:")]) if not one else (members, members)
        for x in ls:
            for y in rs:
                if x != y:
                    pairs.add(tuple(sorted((x, y))) if one else (x, y))
    part = {k: _part_of(sets.find(k), seed) for k in known}  # an entity (and an unmatched record, its own) goes whole to one side
    for frame in (L, R):
        frame["__part"] = frame["__key"].map(part)
        frame["__entity"] = frame["__key"].map(sets.find)
    split = {p: {"left": int((L["__part"] == p).sum()), "right": int((R["__part"] == p).sum()) if not one else None,
                 "matches": sum(1 for x, y in pairs if part[x] == p)} for p in PARTS}
    empty = [p for p in PARTS if split[p]["matches"] == 0]
    if empty:
        raise ValueError(f"no known match falls in the {', '.join(empty)} records: too few matches to choose and judge a model")
    return {"one": one, "left": L, "right": R, "id_left": id_left, "id_right": id_right, "fields": [dict(f) for f in fields],
            "matches": pairs, "block_equal": list(block_equal), "window": dict(window) if window else None, "block_text": block_text,
            "top_k": int(top_k), "seed": int(seed), "split": split, "entities": len(groups), "matches_missing": missing}


def _text_vectors(ctx: dict[str, Any], column_left: str, column_right: str, frame_l: pd.DataFrame, frame_r: pd.DataFrame) -> tuple[Any, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
    train = pd.concat([ctx["left"].loc[ctx["left"]["__part"] == "train", column_left].map(normal),
                       ctx["right"].loc[ctx["right"]["__part"] == "train", column_right].map(normal)])
    vec.fit(train if train.str.len().sum() else pd.Series(["empty"]))  # the vocabulary from the training records only
    return vec.transform(frame_l[column_left].map(normal)), vec.transform(frame_r[column_right].map(normal))


def candidates(ctx: dict[str, Any], part: str) -> dict[str, Any]:
    """The candidate pairs of one part: its records (on the left) against every record the part may see (``SEES``), by
    the blocking; their positions, keys, and whether each is a known match. A pair across parts is never a match (an
    entity is whole on one side), so comparing with earlier parts' records reads no label."""
    seen = SEES[part]
    L = ctx["left"][ctx["left"]["__part"] == part].reset_index(drop=True)
    R = (ctx["left"] if ctx["one"] else ctx["right"])
    R = R[R["__part"].isin(seen)].reset_index(drop=True)
    same = ctx["one"]
    lk, rk = L["__key"].to_numpy(), R["__key"].to_numpy()
    found: set[tuple[int, int]] = set()

    def add(i: int, j: int) -> None:
        if same and lk[i] == rk[j]:
            return
        found.add((i, j))

    if ctx["block_text"]:
        from sklearn.neighbors import NearestNeighbors

        column = ctx["block_text"]
        field = next((f for f in ctx["fields"] if f["column"] == column), {"column": column})
        vl, vr = _text_vectors(ctx, column, field.get("right") or column, L, R)
        k = min(ctx["top_k"] + (1 if same else 0), vr.shape[0])
        if k:
            _, idx = NearestNeighbors(n_neighbors=k, metric="cosine").fit(vr).kneighbors(vl)
            for i, row in enumerate(idx):
                for j in row:
                    add(i, int(j))
            # both ways: a record that is another's nearest is paired with it even when that one is not its own nearest
            k2 = min(ctx["top_k"] + (1 if same else 0), vl.shape[0])
            if k2:
                _, back = NearestNeighbors(n_neighbors=k2, metric="cosine").fit(vl).kneighbors(vr)
                for j, row in enumerate(back):
                    for i in row:
                        add(int(i), j)
    if ctx["block_equal"] or ctx["window"]:
        keys_l = L[ctx["block_equal"]].astype(str).agg("|".join, axis=1) if ctx["block_equal"] else pd.Series([""] * len(L))
        keys_r = R[ctx["block_equal"]].astype(str).agg("|".join, axis=1) if ctx["block_equal"] else pd.Series([""] * len(R))
        window = ctx["window"]
        tl = pd.to_datetime(L[window["column"]], errors="coerce") if window else None
        tr = pd.to_datetime(R[window["column"]], errors="coerce") if window else None
        by_key: dict[str, list[int]] = {}
        for j, key in enumerate(keys_r):
            by_key.setdefault(key, []).append(j)
        left_by_key: dict[str, list[int]] = {}
        for i, key in enumerate(keys_l):
            left_by_key.setdefault(key, []).append(i)
        for key, is_ in left_by_key.items():
            js = np.asarray(by_key.get(key, []), dtype=int)
            if not len(js):
                continue
            if window:
                order = js[np.argsort(tr.iloc[js].to_numpy())]
                times = tr.iloc[order].to_numpy()
            for i in is_:
                if window:
                    t = tl.iloc[i]
                    if pd.isna(t):
                        continue
                    lo = np.searchsorted(times, np.datetime64(t - pd.Timedelta(minutes=window["minutes"])), side="left")
                    hi = np.searchsorted(times, np.datetime64(t + pd.Timedelta(minutes=window["minutes"])), side="right")
                    near = order[lo:hi]
                else:
                    near = js
                for j in near:
                    add(int(i), int(j))
                if len(found) > MAX_PAIRS:
                    raise ValueError(f"blocking makes more than {MAX_PAIRS} pairs for the {part} records: block on more columns or a shorter window")
    if same:  # one table: a pair of two records of this part is kept once (i against j, never j against i as well)
        once: dict[tuple[str, str], tuple[int, int]] = {}
        for i, j in sorted(found):
            once.setdefault(tuple(sorted((lk[i], rk[j]))), (i, j))
        found = set(once.values())
    pairs = sorted(found)
    li = np.asarray([p[0] for p in pairs], dtype=int)
    ri = np.asarray([p[1] for p in pairs], dtype=int)
    kl = lk[li] if len(li) else np.array([])
    kr = rk[ri] if len(ri) else np.array([])
    truth = np.asarray([((a, b) if not same else tuple(sorted((a, b)))) in ctx["matches"] for a, b in zip(kl, kr)], dtype=int)
    known = ctx["split"][part]["matches"]
    found_true = len({tuple(sorted((a, b))) if same else (a, b) for a, b, t in zip(kl, kr, truth) if t})
    entity = L["__entity"].to_numpy()[li] if len(li) else np.array([])
    return {"part": part, "L": L, "R": R, "li": li, "ri": ri, "y": truth, "entity": entity, "pairs": len(pairs), "true_in_blocks": found_true,
            "true_pairs": known, "blocking_recall": round(found_true / known, 4) if known else None}


def features(ctx: dict[str, Any], cand: dict[str, Any]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Each candidate pair's similarity per field, the names, and the exact rule's verdict (every key field agrees; a key
    flag is agreed when either record carries it)."""
    L, R, li, ri = cand["L"], cand["R"], cand["li"], cand["ri"]
    cols, names = [], []
    exact = np.ones(len(li), dtype=bool)
    for f in ctx["fields"]:
        cl, cr, kind = f["column"], f.get("right") or f["column"], f.get("kind", "text")
        if kind == "text":
            vl, vr = _text_vectors(ctx, cl, cr, L, R)
            cos = np.asarray(vl[li].multiply(vr[ri]).sum(axis=1)).ravel() if len(li) else np.array([])
            tl = [set(normal(v).split()) for v in L[cl].to_numpy()[li]]
            tr = [set(normal(v).split()) for v in R[cr].to_numpy()[ri]]
            jac = np.asarray([len(a & b) / len(a | b) if (a | b) else 0.0 for a, b in zip(tl, tr)])
            same = np.asarray([a == b and bool(a) for a, b in zip(tl, tr)], dtype=bool)
            cols += [cos, jac]
            names += [f"{cl}:cosine", f"{cl}:tokens"]
        elif kind == "number":
            xl = np.asarray([_number(v) for v in L[cl].to_numpy()[li]])
            xr = np.asarray([_number(v) for v in R[cr].to_numpy()[ri]])
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.abs(xl - xr) / np.maximum(np.maximum(np.abs(xl), np.abs(xr)), 1e-9)
            missing = np.isnan(rel)
            cols += [np.where(missing, 1.0, np.minimum(rel, 1.0)), missing.astype(float)]
            names += [f"{cl}:relative_difference", f"{cl}:missing"]
            same = (~missing) & (rel < 1e-9)
        elif kind == "time":
            tl = pd.to_datetime(pd.Series(L[cl].to_numpy()[li]), errors="coerce")
            tr = pd.to_datetime(pd.Series(R[cr].to_numpy()[ri]), errors="coerce")
            minutes = ((tr - tl).dt.total_seconds() / 60.0).to_numpy()
            cols += [np.where(np.isnan(minutes), 1e6, np.abs(minutes))]
            names += [f"{cl}:minutes_apart"]
            same = np.abs(np.nan_to_num(minutes, nan=1e9)) < 1.0
        elif kind == "flag":
            fl = pd.to_numeric(pd.Series(L[cl].to_numpy()[li]), errors="coerce").fillna(0).to_numpy() > 0
            fr = pd.to_numeric(pd.Series(R[cr].to_numpy()[ri]), errors="coerce").fillna(0).to_numpy() > 0
            cols += [(fl | fr).astype(float), (fl & fr).astype(float)]
            names += [f"{cl}:either", f"{cl}:both"]
            same = fl | fr
        else:
            al = pd.Series(L[cl].to_numpy()[li]).astype(str).map(normal).to_numpy()
            ar = pd.Series(R[cr].to_numpy()[ri]).astype(str).map(normal).to_numpy()
            same = (al == ar) & (al != "")
            cols += [same.astype(float)]
            names += [f"{cl}:equal"]
        if f.get("key", kind == "text"):
            exact &= same
    X = np.column_stack(cols) if cols else np.zeros((len(li), 0))
    return X, names, exact


def _metrics(y: np.ndarray, flagged: np.ndarray, total_true: int) -> dict[str, Any]:
    tp = int((flagged & (y == 1)).sum())
    precision = tp / int(flagged.sum()) if flagged.sum() else None
    recall = tp / total_true if total_true else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"precision": round(precision, 4) if precision is not None else None, "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4), "flagged": int(flagged.sum()), "true_found": tp}


def _sample(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keep = rng.choice(neg, min(len(neg), max(NEGATIVES_PER_POSITIVE * max(len(pos), 1), 200)), replace=False) if len(neg) else neg
    return np.sort(np.concatenate([pos, keep]))


def _fit(name: str, X: np.ndarray, y: np.ndarray, seed: int) -> Any:
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X, y)
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed).fit(X, y)


def _best_threshold(y: np.ndarray, s: np.ndarray, total_true: int) -> tuple[Optional[float], dict[str, Any]]:
    """The threshold with the best F1 over every distinct score (recall over every known pair of the part, blocked or not)."""
    if not len(s) or not y.sum():
        return None, {"precision": None, "recall": None, "f1": 0.0, "flagged": 0, "true_found": 0}
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], y[order]
    tp = np.cumsum(yy)
    flagged = np.arange(1, len(ss) + 1)
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]  # the last row of each run of equal scores: a threshold flags them all
    precision, recall = tp[last] / flagged[last], tp[last] / max(total_true, 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    best = int(np.argmax(f1))
    t = float(ss[last[best]])
    return t, _metrics(y, s >= t, total_true)


def _models(ctx: dict[str, Any], tr: dict[str, Any], Xt: np.ndarray) -> dict[str, Any]:
    fitted = {}
    if tr["y"].sum() and (tr["y"] == 0).sum():
        rows = _sample(tr["y"], ctx["seed"])
        for name in ("logistic", "boosting"):
            fitted[name] = _fit(name, Xt[rows], tr["y"][rows], ctx["seed"])
    return fitted


def _single(X: np.ndarray, names: list[str], y: np.ndarray, total: int) -> tuple[int, int, Optional[float], dict[str, Any]]:
    """The one field, its direction and its threshold with the best F1 on these pairs (a similarity high, a difference low)."""
    best = (-1.0, 0, 1, None, {"f1": 0.0})
    for c in range(X.shape[1]):
        for sign in (1, -1):
            t, m = _best_threshold(y, sign * X[:, c], total)
            if t is not None and m["f1"] > best[0]:
                best = (m["f1"], c, sign, t, m)
    return best[1], best[2], best[3], best[4]


def validate(ctx: dict[str, Any]) -> dict[str, Any]:
    """Every model fitted on the training pairs and judged on the validation pairs (never the holdout), its threshold set
    there: the best F1 wins, a tie to the simpler."""
    tr, va = candidates(ctx, "train"), candidates(ctx, "validation")
    Xt, names, _ = features(ctx, tr)
    Xv, _, exv = features(ctx, va)
    total = va["true_pairs"]
    scores: dict[str, Any] = {"exact": {**_metrics(va["y"], exv, total), "threshold": None}}
    c, sign, t, m = _single(Xv, names, va["y"], total)
    scores["single_field"] = {**m, "field": names[c] if names else None, "direction": "high" if sign == 1 else "low", "threshold": t, "column": c, "sign": sign}
    for name, model in _models(ctx, tr, Xt).items():
        sv = model.predict_proba(Xv)[:, 1] if len(Xv) else np.array([])
        thr, mm = _best_threshold(va["y"], sv, total)
        scores[name] = {**mm, "threshold": thr}
    chosen = max((m for m in MODELS if m in scores), key=lambda m: (round(scores[m]["f1"], 4), -MODELS.index(m)))
    return {"scores": scores, "chosen": chosen, "names": names, "blocking": {p: c_["blocking_recall"] for p, c_ in (("train", tr), ("validation", va))},
            "pairs": {"train": tr["pairs"], "validation": va["pairs"]},
            "chosen_by": "the best pair F1 on the validation records, each model's threshold set on every validation score (a tie to the simpler: "
                         "exact, single field, logistic, boosting)"}


def _bootstrap(y: np.ndarray, entity: np.ndarray, mine: np.ndarray, other: Optional[np.ndarray], total_by_entity: Mapping[str, int], seed: int,
               level: float = 0.90) -> Optional[tuple[float, float]]:
    """An interval over holdout entities drawn again: of the F1 difference (the chosen model's minus another's), or of the
    chosen model's F1 alone when ``other`` is None."""
    keys = sorted(total_by_entity)
    if len(keys) < 5:
        return None
    rng = np.random.default_rng(seed)
    position = {k: i for i, k in enumerate(keys)}
    pos = np.asarray([position.get(e, -1) for e in entity])
    true = np.asarray([total_by_entity[k] for k in keys])
    diffs = []
    for _ in range(BOOTSTRAP):
        w = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)
        wp = np.where(pos >= 0, w[np.maximum(pos, 0)], 0.0)
        t = float((w * true).sum())
        f = []
        for flags in (mine, other) if other is not None else (mine,):
            tp, fl = float((wp * (flags & (y == 1))).sum()), float((wp * flags).sum())
            p, r = (tp / fl if fl else 0.0), (tp / t if t else 0.0)
            f.append(2 * p * r / (p + r) if p + r else 0.0)
        diffs.append(f[0] - f[1] if other is not None else f[0])
    tail = (1 - level) / 2
    return round(float(np.quantile(diffs, tail)), 4), round(float(np.quantile(diffs, 1 - tail)), 4)


def score(ctx: dict[str, Any], chosen: str, threshold: Optional[float] = None, single: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """The holdout read once: the chosen model, fitted on the training pairs as in ``validate`` and with the threshold set
    there, scores the holdout's pairs (its records against every record), beside the exact rule and the single best field
    (``single``: its column, direction and threshold from ``validate``), each with a bootstrap over entities of the F1
    difference."""
    from sklearn.metrics import average_precision_score

    tr, ho = candidates(ctx, "train"), candidates(ctx, "holdout")
    Xt, names, _ = features(ctx, tr)
    Xh, _, exh = features(ctx, ho)
    total = ho["true_pairs"]
    flags = {"exact": exh}
    if single and single.get("threshold") is not None and Xh.shape[1]:
        flags["single_field"] = single["sign"] * Xh[:, single["column"]] >= single["threshold"]
    else:
        flags["single_field"] = np.zeros(len(ho["y"]), dtype=bool)
    s = exh.astype(float)
    if chosen in ("logistic", "boosting"):
        model = _models(ctx, tr, Xt).get(chosen)
        s = model.predict_proba(Xh)[:, 1] if model is not None and len(Xh) else np.zeros(len(ho["y"]))
        flags[chosen] = s >= float(threshold if threshold is not None else 0.5)
    elif chosen == "single_field":
        s = single["sign"] * Xh[:, single["column"]] if single and Xh.shape[1] else s
    out = {m: _metrics(ho["y"], f, total) for m, f in flags.items()}
    entity_of = dict(zip(ctx["left"]["__key"], ctx["left"]["__entity"]))
    part_of = dict(zip(ctx["left"]["__key"], ctx["left"]["__part"]))
    entities = {e: 0 for e in ho["L"]["__entity"]}  # every holdout entity, with its true pairs (zero for a record with no match)
    for a, _ in ctx["matches"]:
        if part_of.get(a) == "holdout" and entity_of.get(a) in entities:
            entities[entity_of[a]] += 1
    mine = flags[chosen]
    for name in ("exact", "single_field"):
        if name != chosen:
            out[name]["f1_difference_ci90"] = _bootstrap(ho["y"], ho["entity"], mine, flags[name], entities, ctx["seed"])
    ap = float(average_precision_score(ho["y"], s)) * (ho["true_in_blocks"] / total) if total and ho["y"].sum() and len(set(ho["y"])) > 1 else None
    out[chosen]["f1_ci95"] = _bootstrap(ho["y"], ho["entity"], mine, None, entities, ctx["seed"], level=0.95)
    me = out[chosen]
    beats = {n: (me["f1"] > out[n]["f1"] and (out[n].get("f1_difference_ci90") or (0, 0))[0] > 0) for n in ("exact", "single_field") if n != chosen}
    return {"chosen": chosen, "threshold": threshold, "scores": out, "pair_f1": me["f1"], "pair_f1_ci95": me.get("f1_ci95"),
            "pair_precision": me["precision"], "pair_recall": me["recall"],
            "average_precision": round(ap, 4) if ap is not None else None, "blocking_recall": ho["blocking_recall"], "holdout_pairs": ho["pairs"],
            "holdout_true_pairs": total, "beats": beats, "beats_exact": beats.get("exact", False), "beats_single_field": beats.get("single_field", False)}


def link(ctx: dict[str, Any], chosen: str, threshold: Optional[float] = None, single: Optional[Mapping[str, Any]] = None) -> list[tuple[str, str, float]]:
    """30.23: the pairs the chosen model links over every record (each part's records against what they may see), each with its score:
    what a data foundation resolves into one entity. Fitted on the training pairs and thresholded as ``validate`` chose; the holdout is
    scored here as everywhere else, never read for a choice."""
    tr = candidates(ctx, "train")
    Xt, _, _ = features(ctx, tr)
    model = _models(ctx, tr, Xt).get(chosen) if chosen in ("logistic", "boosting") else None
    seen: dict[tuple[str, str], float] = {}
    for part in PARTS:
        c = tr if part == "train" else candidates(ctx, part)
        X, _, exact = features(ctx, c)
        if model is not None and len(X):
            s = model.predict_proba(X)[:, 1]
            keep = s >= float(threshold if threshold is not None else 0.5)
        elif chosen == "single_field" and single and single.get("threshold") is not None and X.shape[1]:
            s = single["sign"] * X[:, single["column"]]
            keep = s >= single["threshold"]
        else:
            s, keep = exact.astype(float), exact.astype(bool)
        a, b = c["L"]["__key"].to_numpy()[c["li"]], c["R"]["__key"].to_numpy()[c["ri"]]
        for x, y, score_, k in zip(a, b, s, keep):
            if k and x != y:
                pair = tuple(sorted((str(x), str(y))))
                seen[pair] = max(seen.get(pair, 0.0), float(score_))
    return [(x, y, round(v, 6)) for (x, y), v in sorted(seen.items())]


def evaluate(left: pd.DataFrame, right: Optional[pd.DataFrame] = None, **kw: Any) -> dict[str, Any]:
    """setup, validate and score in one call (a kind's own check of the library)."""
    ctx = setup(left, right, **kw)
    picked = validate(ctx)
    threshold = picked["scores"][picked["chosen"]].get("threshold")
    return {"split": ctx["split"], "validation": picked, "holdout": score(ctx, picked["chosen"], threshold, picked["scores"]["single_field"])}


__all__ = ["link", "PARTS", "SEES", "MODELS", "normal", "setup", "candidates", "features", "validate", "score", "evaluate"]
