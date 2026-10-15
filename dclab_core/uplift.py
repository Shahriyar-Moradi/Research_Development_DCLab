"""Uplift (package 16.7): who responds to an action, not who would have bought anyway.

Rows carry a treatment (two values: the action was taken or not), an outcome (two values) and the person's statement of
whether the treatment was assigned at random. The question is the difference the action makes for each row; the rows
that buy anyway, or never, are not the ones to target.

- :func:`overlap`: a propensity model (the chance of treatment from the columns, out of fold, gradient-boosted so a
  rule like "the youngest and the oldest" is seen) and the standardised differences of the columns between arms. With
  random assignment the propensity should say nothing (AUC near 0.5); without it, rows whose propensity is outside
  the band have no comparable rows in the other arm.
- :func:`evaluate`: rows split once, stratified by arm; a model chosen on a validation cut of the training rows, then
  every model refitted on the training rows and scored once on the holdout: a T-learner (one model per arm), an
  X-learner (imputed effects, mixed by the propensity) and an uplift tree (a regression tree on the transformed
  outcome), against a response model (who buys when treated) and random targeting. Scores: the Qini coefficient (the
  area between the Qini curve and the random line, per treated row) and the AUUC (the area under the uplift curve
  above the random line), each with a paired 90% bootstrap interval over rows, and the uplift by decile of predicted
  uplift. When assignment was not random, every comparison of arms is weighted by the inverse propensity (fitted on
  the rows the models are fitted on), so a model is not chosen for finding who was targeted.
- :func:`wording`: how the record may speak. Randomised, and the columns do not predict the arm: effects. Not random but
  overlapping and balanced: effects only under the assumption that no unmeasured column drives both the action and the
  outcome. Overlapping but imbalanced: differences between arms that differ on named columns. Not overlapping:
  associations only, never effects, said in the record.

Pure: numpy, pandas and scikit-learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

TRIM = 0.05  # propensities outside the band (at least [TRIM, 1 - TRIM]) have no comparable rows in the other arm
MIN_OVERLAP = 0.9  # the share of rows inside that band below which the arms do not overlap
MAX_SMD = 0.25  # a standardised mean difference between arms above which a column is imbalanced
YES = {"1", "1.0", "true", "yes", "y", "treated", "t"}
NO = {"0", "0.0", "false", "no", "n", "control", "f"}


def _r(x: Any, digits: int = 4) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if np.isfinite(v) else None


def binary(column: pd.Series, name: str) -> np.ndarray:
    """A two-valued column as 0/1: booleans, 0/1, yes/no, true/false, treated/control. Anything else is refused by name."""
    values = column.dropna().astype(str).str.strip().str.lower()
    seen = set(values.unique())
    if column.isna().any() or not seen or not seen <= YES | NO:
        raise ValueError(f"{name!r} must hold two values that read as yes and no (1/0, true/false, yes/no); it holds "
                         f"{sorted(seen)[:6]}{' and missing values' if column.isna().any() else ''}")
    return values.isin(YES).to_numpy(int)


def _design(train: pd.DataFrame, frames: Sequence[pd.DataFrame], features: Sequence[str]) -> list[np.ndarray]:
    """Numbers as they are (missing filled with the training median), categories one-hot on the training levels."""
    cats = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
    nums = [c for c in features if c not in cats]
    med = train[nums].median()
    levels = {c: sorted(train[c].astype(str).unique()) for c in cats}

    def one(f):
        parts = [f[nums].fillna(med).to_numpy(float)] if nums else []
        parts += [np.column_stack([(f[c].astype(str) == v).to_numpy(float) for v in levels[c]]) for c in cats]
        return np.hstack(parts) if parts else np.zeros((len(f), 0))
    return [one(f) for f in frames]


def _propensity_model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=50, random_state=0)


def overlap(frame: pd.DataFrame, *, treatment: str, features: Sequence[str], seed: int = 0) -> dict[str, Any]:
    """How alike the two arms are: out-of-fold propensities (gradient-boosted), their AUC, the share inside the overlap
    band, and the columns whose means differ between arms."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    t = binary(frame[treatment], treatment)
    p = np.zeros(len(frame))
    for fit, held in StratifiedKFold(5, shuffle=True, random_state=seed).split(frame, t):
        Xf, Xh = _design(frame.iloc[fit], [frame.iloc[fit], frame.iloc[held]], features)
        p[held] = _propensity_model().fit(Xf, t[fit]).predict_proba(Xh)[:, 1]
    X, = _design(frame, [frame], features)
    names = [c for c in features if pd.api.types.is_numeric_dtype(frame[c])] + [f"{c}={v}" for c in features if not pd.api.types.is_numeric_dtype(frame[c])
                                                                                for v in sorted(frame[c].astype(str).unique())]
    pooled = np.sqrt((X[t == 1].var(axis=0) + X[t == 0].var(axis=0)) / 2) + 1e-9
    smd = (X[t == 1].mean(axis=0) - X[t == 0].mean(axis=0)) / pooled
    share = float(t.mean())
    low, high = min(TRIM, share / 2), max(1 - TRIM, 1 - (1 - share) / 2)  # a 95/5 trial is not "outside" by design
    inside = float(np.mean((p >= low) & (p <= high)))
    worst = sorted(zip(names, smd), key=lambda z: -abs(z[1]))[:5]
    return {"propensity_auc": _r(roc_auc_score(t, p), 3), "inside_band": _r(inside, 3), "band": [_r(low, 3), _r(high, 3)],
            "imbalanced": [{"column": n, "smd": _r(v, 3)} for n, v in worst if abs(v) > MAX_SMD],
            "overlaps": inside >= MIN_OVERLAP, "treated_share": _r(share, 3), "propensity": p}


def wording(randomised: bool, ov: dict[str, Any]) -> dict[str, Any]:
    """What the record may call the differences it measures."""
    if randomised and (ov["propensity_auc"] or 0.5) <= 0.6 and ov["overlaps"]:
        return {"causal": True, "say": "effects: the action was assigned at random, and the columns do not predict who got it"}
    if randomised:
        return {"causal": False, "say": f"associations: the action was said to be random, but the columns predict who got it "
                                        f"(propensity AUC {ov['propensity_auc']}); check how it was assigned before calling these effects"}
    if ov["overlaps"] and not ov["imbalanced"]:
        return {"causal": False, "assumed": True, "say": "effects only under an assumption: the action was not random, so these are effects "
                                                        "only if no unmeasured column drives both the action and the outcome"}
    if ov["overlaps"]:
        return {"causal": False, "say": "differences between arms that differ on " + ", ".join(i["column"] for i in ov["imbalanced"])
                                        + ": the arms overlap, but the action was not random and they are not alike, so these are not effects"}
    return {"causal": False, "say": f"associations, not effects: the action was not assigned at random and the arms do not overlap "
                                    f"({ov['inside_band']:.0%} of rows have a comparable row in the other arm; imbalanced: "
                                    + (", ".join(i["column"] for i in ov["imbalanced"]) or "none") + ")"}


def _hgb(kind: str):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    cls = HistGradientBoostingClassifier if kind == "classifier" else HistGradientBoostingRegressor
    return cls(max_iter=100, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=100, random_state=0)


def _propensity(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The chance of treatment of the rows a model is fitted on, from those rows only."""
    from sklearn.linear_model import LogisticRegression

    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    return LogisticRegression(max_iter=2000).fit((X - mu) / sd, t).predict_proba((X - mu) / sd)[:, 1]


class TLearner:
    def fit(self, X, t, y):
        self.m1, self.m0 = _hgb("classifier").fit(X[t == 1], y[t == 1]), _hgb("classifier").fit(X[t == 0], y[t == 0])
        return self

    def uplift(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


class XLearner:
    """Kunzel et al.: in the treated arm the effect is Y - m0(X), in the control arm m1(X) - Y; each regressed on the
    columns (g1, g0) and mixed by the propensity e: tau = e g0 + (1 - e) g1 (the arm with fewer rows weighs less)."""

    def fit(self, X, t, y):
        from sklearn.linear_model import LogisticRegression

        base = TLearner().fit(X, t, y)
        self.g1 = _hgb("regressor").fit(X[t == 1], y[t == 1] - base.m0.predict_proba(X[t == 1])[:, 1])
        self.g0 = _hgb("regressor").fit(X[t == 0], base.m1.predict_proba(X[t == 0])[:, 1] - y[t == 0])
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        self.prop = LogisticRegression(max_iter=2000).fit((X - self.mu) / self.sd, t)
        return self

    def mix(self, e: np.ndarray, g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
        return e * g0 + (1 - e) * g1

    def uplift(self, X):
        e = np.clip(self.prop.predict_proba((X - self.mu) / self.sd)[:, 1], TRIM, 1 - TRIM)
        return self.mix(e, self.g0.predict(X), self.g1.predict(X))


def transformed_outcome(y: np.ndarray, t: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Z = Y (T - e) / (e (1 - e)): its expectation given the columns is the uplift."""
    return y * (t - e) / (e * (1 - e))


class UpliftTree:
    """A regression tree on the transformed outcome, whose mean in a leaf is the uplift there."""

    def fit(self, X, t, y):
        from sklearn.tree import DecisionTreeRegressor

        e = np.clip(_propensity(X, t), TRIM, 1 - TRIM)
        self.tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=max(50, len(X) // 60), random_state=0).fit(X, transformed_outcome(y, t, e))
        return self

    def uplift(self, X):
        return self.tree.predict(X)


class Response:
    """The usual baseline: target who is most likely to buy when treated (not who changes because of it)."""

    def fit(self, X, t, y):
        self.m = _hgb("classifier").fit(X[t == 1], y[t == 1])
        return self

    def uplift(self, X):
        return self.m.predict_proba(X)[:, 1]


MODELS = {"t_learner": TLearner, "x_learner": XLearner, "uplift_tree": UpliftTree, "response_model": Response}


def _curves(score: np.ndarray, t: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None, points: int = 20):
    """Targeting the top share by score: the (weighted) treated and control counts and outcomes in it."""
    w = np.ones(len(t)) if w is None else w
    order = np.argsort(-score, kind="stable")
    t, y, w = t[order], y[order], w[order]
    shares = np.linspace(0, 1, points + 1)
    rows = []
    for s in shares:
        n = int(round(s * len(t)))
        tt, yy, ww = t[:n], y[:n], w[:n]
        nt, nc = (ww * tt).sum(), (ww * (1 - tt)).sum()
        rows.append((nt, nc, (ww * tt * yy).sum(), (ww * (1 - tt) * yy).sum()))
    return shares, np.array(rows)


def qini(score: np.ndarray, t: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None, points: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """The Qini curve: targeting the top share of rows by score, the extra outcomes the treated ones show over the
    control ones scaled to the treated count: Y_T(top) - Y_C(top) * N_T(top) / N_C(top) (weighted when ``w`` is given)."""
    shares, c = _curves(score, t, y, w, points)
    nt, nc, yt, yc = c.T
    return shares, yt - np.where(nc > 0, yc * nt / np.where(nc > 0, nc, 1), 0.0)


def qini_coefficient(score: np.ndarray, t: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """The area between the Qini curve and the random-targeting line, per treated row (0: no better than random)."""
    shares, gains = qini(score, t, y, w)
    nt_all = (w * t).sum() if w is not None else t.sum()
    return float(_trapz(gains - gains[-1] * shares, shares) / max(nt_all, 1e-9))


def auuc(score: np.ndarray, t: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None, points: int = 20) -> float:
    """The area under the uplift curve above the random line: the uplift curve at share s is (mean outcome of the treated
    minus of the control in the top s) times s; random targeting gives the overall uplift times s."""
    shares, c = _curves(score, t, y, w, points)
    nt, nc, yt, yc = c.T
    diff = np.where((nt > 0) & (nc > 0), yt / np.where(nt > 0, nt, 1) - yc / np.where(nc > 0, nc, 1), 0.0)
    curve = diff * shares
    return float(_trapz(curve - diff[-1] * shares, shares))


def _score(s: np.ndarray, t: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], draws: list) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, f in (("qini_coefficient", qini_coefficient), ("auuc", auuc)):
        sample = [f(s[d], t[d], y[d], None if w is None else w[d]) for d in draws]
        out[name] = _r(f(s, t, y, w))
        out[f"{name}_ci90"] = [_r(np.quantile(sample, 0.05)), _r(np.quantile(sample, 0.95))]
        out[f"{name}_ci95"] = [_r(np.quantile(sample, 0.025)), _r(np.quantile(sample, 0.975))]  # 16.15 B: the readers' interval, the same draws
    deciles = []
    for part in np.array_split(np.argsort(-s, kind="stable"), 10):
        tp, yp = t[part], y[part]
        wp = np.ones(len(part)) if w is None else w[part]
        ok = tp.any() and (1 - tp).any()
        deciles.append(_r((wp * tp * yp).sum() / (wp * tp).sum() - (wp * (1 - tp) * yp).sum() / (wp * (1 - tp)).sum(), 3) if ok else None)
    out.update({"uplift_by_decile": deciles, "qini": [_r(g, 2) for g in qini(s, t, y, w)[1]]})
    return out


def choose(validation: dict[str, Any]) -> Optional[str]:
    """The uplift model to use: the best validation Qini coefficient, never the response model or random targeting it is judged against."""
    return max((n for n in validation if n not in ("response_model", "random")), key=lambda n: validation[n] if validation[n] is not None else -np.inf,
               default=None)


def split(frame: pd.DataFrame, *, treatment: str, holdout: float = 0.3, validation: float = 0.25, seed: int = 0) -> dict[str, Any]:
    """The rows, by arm only (never the outcome): a holdout stratified by arm, and inside the training rows a validation
    cut stratified the same way; with the holdout's fingerprint, which a later stage checks before it reads it."""
    import hashlib

    from sklearn.model_selection import train_test_split

    data = frame.reset_index(drop=True)
    t = binary(data[treatment], treatment)
    if len(set(t)) < 2:
        raise ValueError("every row is in one arm: there is nothing to compare the action with")
    train, held = train_test_split(np.arange(len(data)), test_size=holdout, stratify=t, random_state=seed)
    fit_rows, val_rows = train_test_split(train, test_size=validation, stratify=t[train], random_state=seed)
    return {"train": train, "fit": fit_rows, "validation": val_rows, "holdout": held, "seed": seed,
            "fingerprint": hashlib.sha256(",".join(map(str, sorted(held.tolist()))).encode()).hexdigest()[:16],
            "sizes": {"train": int(len(train)), "validation": int(len(val_rows)), "holdout": int(len(held)), "stratified_by": treatment}}


def _run(data: pd.DataFrame, t: np.ndarray, y: np.ndarray, fit_idx: np.ndarray, eval_idx: np.ndarray, features: Sequence[str],
         randomised: bool, seed: int, rng: Any, boot: int, only: Optional[Sequence[str]] = None) -> tuple[dict, dict]:
    Xf, Xe = _design(data.iloc[fit_idx], [data.iloc[fit_idx], data.iloc[eval_idx]], features)
    out = {name: make().fit(Xf, t[fit_idx], y[fit_idx]).uplift(Xe) for name, make in MODELS.items() if only is None or name in only}
    out["random"] = np.random.default_rng(seed + 1).uniform(size=len(eval_idx))
    w = None
    if not randomised:  # who was targeted must not pass for who responds: weight each arm by 1 / its propensity
        e = np.clip(_propensity_model().fit(Xf, t[fit_idx]).predict_proba(Xe)[:, 1], TRIM, 1 - TRIM)
        w = np.where(t[eval_idx] == 1, 1 / e, 1 / (1 - e))
    draws = [rng.integers(0, len(eval_idx), len(eval_idx)) for _ in range(boot)]  # the same rows for every model: paired
    return out, {name: _score(sc, t[eval_idx], y[eval_idx], w, draws) for name, sc in out.items()}


def validate(frame: pd.DataFrame, parts: dict[str, Any], *, treatment: str, outcome: str, features: Sequence[str], randomised: bool,
             boot: int = 200) -> dict[str, Any]:
    """Every model fitted on the training rows' fit cut and scored on their validation cut (the holdout's outcomes are
    never read): the validation Qini coefficient by model and the model chosen."""
    data = frame.reset_index(drop=True)
    t = binary(data[treatment], treatment)
    rows = np.concatenate([parts["fit"], parts["validation"]])
    y = np.zeros(len(data), dtype=int)
    y[rows] = binary(data[outcome].iloc[rows], outcome)  # the training rows' outcomes only
    if len(set(y[rows])) < 2:
        raise ValueError("every training row has the same outcome: there is no response to compare")
    _, on_validation = _run(data, t, y, parts["fit"], parts["validation"], features, randomised, parts["seed"],
                            np.random.default_rng(parts["seed"]), boot)
    scores = {n: v["qini_coefficient"] for n, v in on_validation.items()}
    return {"validation": scores, "details": on_validation, "best": choose(scores),
            "chosen_by": "the Qini coefficient on a validation cut of the training rows, before the holdout"}


def score(frame: pd.DataFrame, parts: dict[str, Any], *, treatment: str, outcome: str, features: Sequence[str], randomised: bool,
          best: Optional[str], boot: int = 200) -> dict[str, Any]:
    """Every model refitted on all training rows and scored once on the holdout, paired draws as ``evaluate`` makes them
    (the validation's draws are consumed first, so the numbers are the library's own)."""
    data = frame.reset_index(drop=True)
    t, y = binary(data[treatment], treatment), binary(data[outcome], outcome)
    rng = np.random.default_rng(parts["seed"])
    for _ in range(boot):  # the validation's draws, as evaluate takes them before the holdout's
        rng.integers(0, len(parts["validation"]), len(parts["validation"]))
    scores, result = _run(data, t, y, parts["train"], parts["holdout"], features, randomised, parts["seed"], rng, boot)
    lower = result[best]["qini_coefficient_ci90"][0] if best else None
    return {"models": result, "best_beats_random": lower is not None and lower > 0,
            "best_beats_response_model": bool(best) and (result[best]["qini_coefficient"] or 0) > (result["response_model"]["qini_coefficient"] or 0),
            "holdout_rows": parts["holdout"].tolist(), "holdout_uplift": {n: s.tolist() for n, s in scores.items() if n in MODELS}}


def evaluate(frame: pd.DataFrame, *, treatment: str, outcome: str, features: Sequence[str], randomised: bool, holdout: float = 0.3,
             validation: float = 0.25, seed: int = 0, boot: int = 200) -> dict[str, Any]:
    """The model chosen on a validation cut of the training rows, then every model refitted on all training rows and
    scored once on a holdout split stratified by arm: the Qini coefficient and the AUUC with paired 90% intervals, the
    uplift by decile of predicted uplift; with the overlap check and the wording the record may use. When assignment was
    not random, the arms are compared with inverse-propensity weights fitted on the rows the models are fitted on.
    ``split``, ``validate`` and ``score`` are its three steps (16.15: the engine runs them stage by stage)."""
    data = frame.reset_index(drop=True)
    y = binary(data[outcome], outcome)
    parts = split(data, treatment=treatment, holdout=holdout, validation=validation, seed=seed)
    if len(set(y)) < 2:
        raise ValueError("every row has the same outcome: there is no response to compare")
    ov = overlap(data, treatment=treatment, features=features, seed=seed)  # columns and arms only, never the outcome
    chosen = validate(data, parts, treatment=treatment, outcome=outcome, features=features, randomised=randomised, boot=boot)
    held = score(data, parts, treatment=treatment, outcome=outcome, features=features, randomised=randomised, best=chosen["best"], boot=boot)
    return {"randomised": randomised, "overlap": {k: v for k, v in ov.items() if k != "propensity"}, "wording": wording(randomised, ov),
            "split": parts["sizes"], "weighted": not randomised, "validation": chosen["validation"], "validation_rows": parts["validation"].tolist(),
            "best": chosen["best"], "chosen_by": chosen["chosen_by"], **held}


_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed it
