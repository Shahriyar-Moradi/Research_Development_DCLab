"""A wider search, chosen honestly (34.7 with 34.6): Optuna on the screen's leading families, under a time and trial budget,
and the winner confirmed once on a selection fold set aside before the search.

Off unless the project's settings ask for it (``settings.search = {"minutes": 10, "trials": 30, "families": 2}``): the
engine's default runs stay as they were. When on:

1. ``prepare`` sets a selection fold aside from the training rows before any stage scores (``set_aside``), cut the way
   the project splits (``dclab_core.selection``: the latest rows for time, whole groups for entities, by class
   otherwise; on a time split with entities, the selection fold also drops the rows of any entity the search rows
   hold), and the project's folds are cut again over the search rows only: the features stage, the screen, the review
   and the search never see the selection fold (33.2's protocol). A classification needs every class on both sides and
   a group cut at least two groups in the search rows and at most half the rows set aside: a cut that fails is drawn
   again with the next seed (never for time, whose cut is fixed), and when none holds the search is skipped with the
   reason and the folds stay as they were;
2. the models stage, after its screen and review, searches the stage's own choice first (the review's family with the change it kept as the base of every trial) and
   then the screen's next families, with Optuna (TPE, seeded) on folds of the search rows only (cut as the project's
   folds are), every trial a candidate scored by its pooled out-of-fold score, the starting settings included;
3. keeps the best by that score and scores it once on the selection fold (fitted on every search row), beside the
   stage's choice without a search fitted and scored the same way: the record keeps both scores, the gain of the
   search over no search on the selection fold with a paired interval, the number of attempts and the measured
   optimism. The final stage then fits the chosen family with the chosen settings on every training row and reads
   the holdout once, as always.

Nothing here reads the holdout. Every attempt is kept in the record in the order it was made (its settings and score,
never a row).
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from ..expansion import runner as rn

DEFAULT = {"minutes": 10.0, "trials": 30, "families": 2}
SEED = 7
SHARE = 0.2  # the selection fold's share of the training rows
CUT_TRIES = 20  # seeds tried for a cut that holds every class on both sides (group and stratified cuts)
MIN_SELECTION = 10  # rows left in the selection fold, after a time split drops the entities the search rows hold
BOOTSTRAP = 200


def settings(project: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The project's search budget, or None when the search is off."""
    asked = (project.get("settings") or {}).get("search")
    if not asked:
        return None
    if asked is True:
        asked = {}
    try:
        out = {"minutes": float(asked.get("minutes", DEFAULT["minutes"])), "trials": int(asked.get("trials", DEFAULT["trials"])),
               "families": int(asked.get("families", DEFAULT["families"]))}
    except (TypeError, ValueError, AttributeError):
        return None
    out["minutes"] = min(max(out["minutes"], 0.5), 120.0)
    out["trials"] = min(max(out["trials"], 1), 500)
    out["families"] = min(max(out["families"], 1), 5)
    return out


def space(trial: Any, family: str, task_type: str, *, weighted: bool = False) -> Optional[dict[str, Any]]:
    """A family's settings as Optuna draws them; None for a family with no space here (it keeps its defaults).
    ``weighted``: the base already weights the classes (the review's kept change), so no class weight is drawn on top."""
    regression = task_type == "timeseries_regression"
    if family == "logistic_regression":
        params = {"model__C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
        if not regression and not weighted and trial.suggest_categorical("balanced", [False, True]):
            params["model__class_weight"] = "balanced"
        return params
    if family == "ridge":
        return {"model__alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
    if family == "extra_trees":
        return {"model__n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
                "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "model__max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.8, 1.0])}
    if family == "hist_gradient_boosting":
        return {"model__learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "model__max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
                "model__max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
                "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
                "model__l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True)}
    if family == "lightgbm":
        return {"model__n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
                "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "model__num_leaves": trial.suggest_int("num_leaves", 7, 127),
                "model__min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "model__reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}
    if family == "xgboost":
        return {"model__n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "model__max_depth": trial.suggest_int("max_depth", 3, 10),
                "model__subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "model__min_child_weight": trial.suggest_int("min_child_weight", 1, 10)}
    return None


def cut(n: int, *, y: Optional[np.ndarray], entities: Optional[np.ndarray], split: str) -> tuple[np.ndarray, np.ndarray, str] | str:
    """The search rows and the selection fold (positions into the training rows), or the reason no fair cut exists.
    ``y``: the classes of a classification (None for a regression); ``entities``: the group column's values on a group or
    time-with-entity split."""
    from dclab_core import selection

    kind = "time" if split in ("time", "time_group") else "group" if split == "group" else "stratified"
    classes = set(np.unique(y).tolist()) if y is not None else None
    keys = entities.astype(str) if entities is not None else None
    reason = "no cut was tried"
    for attempt in range(1 if kind == "time" else CUT_TRIES):
        try:
            search, select, how = selection.selection_split(n, y=y if kind == "stratified" else None, groups=keys if kind == "group" else None,
                                                            order=np.arange(n) if kind == "time" else None, share=SHARE, seed=SEED + attempt)
        except ValueError as exc:
            reason = str(exc)
            if kind == "time":
                return reason
            continue
        if split == "time_group" and keys is not None:  # an entity the search rows hold never scores the selection fold
            select = select[~np.isin(keys[select], keys[search])]
        if kind == "group" and (len(set(keys[search].tolist())) < 2 or len(select) > 0.5 * n):  # one large group can tip a greedy cut
            reason = f"no cut of {CUT_TRIES} seeds left at least two groups to search and at most half the rows set aside"
            continue
        if len(select) < MIN_SELECTION:
            reason = (f"the selection fold keeps {len(select)} rows once the entities the search rows hold are dropped, fewer than "
                      f"{MIN_SELECTION}")
            continue
        if classes is not None:
            missing = [c for side in (search, select) for c in classes if c not in set(np.unique(y[side]).tolist())]
            if missing:
                reason = ("the latest rows set aside for the selection fold miss a class" if kind == "time" else
                          f"no cut of {CUT_TRIES} seeds kept every class on both the search rows and the selection fold")
                continue
        return search, select, how
    return reason


def _folds(bundle: Any, rows: np.ndarray, split: str, n_folds: int, entities: Optional[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds over the search rows only, cut as the project's folds are (positions into ``rows``)."""
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit

    from ..expansion import datasets as ds

    n = len(rows)
    y = bundle.y_train.to_numpy()[rows]
    if split in ("time", "time_group"):
        folds = [(a, b) for a, b in TimeSeriesSplit(n_splits=n_folds).split(np.arange(n))]
        if split == "time_group" and entities is not None:  # as the project's own folds: a fit row's entity never scores
            keys = entities.astype(str)[rows]
            folds = [(a, b[~np.isin(keys[b], keys[a])]) for a, b in folds]
            folds = [(a, b) for a, b in folds if len(b) >= 10]
        return folds
    classify = bundle.task_type != "timeseries_regression"
    if split == "group" and entities is not None:
        g = entities.astype(str)[rows]
        k = min(n_folds, len(set(g.tolist())))
        if k < 2:
            return []
        if classify and np.unique(y, return_counts=True)[1].min() >= k:
            return [(a, b) for a, b in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=ds.RANDOM_STATE).split(np.arange(n), y, g)]
        return [(a, b) for a, b in GroupKFold(n_splits=k).split(np.arange(n), groups=g)]
    if classify and np.unique(y, return_counts=True)[1].min() >= n_folds:
        return [(a, b) for a, b in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=ds.RANDOM_STATE).split(np.arange(n), y)]
    return [(a, b) for a, b in KFold(n_splits=n_folds, shuffle=True, random_state=ds.RANDOM_STATE).split(np.arange(n))]


def set_aside(bundle: Any) -> dict[str, Any]:
    """The selection fold set aside before any stage scores (step 1 of the module): the project's folds are cut again over
    the search rows, in place on ``bundle`` (prepare's own, for this project and these settings). Returns the cut, or
    ``{"skipped": reason}`` with the folds left as they were."""
    split = bundle.spec.split_strategy
    classify = bundle.task_type != "timeseries_regression"
    entities = bundle.groups("train") if split in ("group", "time_group") else None
    found = cut(len(bundle.y_train), y=bundle.y_train.to_numpy() if classify else None, entities=entities, split=split)
    if isinstance(found, str):
        return {"skipped": found}
    search_rows, select_rows, how = found
    folds = _folds(bundle, search_rows, split, max(2, len(bundle.cv_splits)), entities)
    if len(folds) < 2:
        return {"skipped": "the search rows leave fewer than two usable folds"}
    bundle.cv_splits = [(search_rows[a], search_rows[b]) for a, b in folds]  # positions into the training rows, as before
    bundle.cv_description += (f"; over {len(search_rows)} search rows only: a selection fold of {len(select_rows)} training rows is set "
                              "aside for the search (34.7) and never enters these folds")
    return {"search_rows": search_rows, "select_rows": select_rows, "how": how, "folds": len(folds)}


def skipped(budget: dict[str, Any], why: str) -> dict[str, Any]:
    return {"skipped": why[:500], "budget": budget,
            "rule": f"The search was asked for but skipped ({why[:300]}); the screen's and the review's choice stands."}


def _cluster_interval(statistic: Any, n: int, groups: Optional[np.ndarray]) -> list[Optional[float]]:
    """90% bootstrap interval of ``statistic(rows)`` over ``n`` rows, drawing whole entities when ``groups`` is given."""
    rng = np.random.default_rng(SEED)
    members = None
    if groups is not None:
        codes = np.unique(np.asarray(groups).astype(str), return_inverse=True)[1]
        order = np.argsort(codes, kind="stable")
        members = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    values = []
    for _ in range(BOOTSTRAP):
        rows = (np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]) if members is not None
                else rng.integers(0, n, n))
        try:
            value = statistic(rows)
        except ValueError:
            continue
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if len(values) < 20:
        return [None, None]
    return [round(float(np.quantile(values, 0.05)), 6), round(float(np.quantile(values, 0.95)), 6)]


def _paired_gain(y: np.ndarray, chosen: np.ndarray, without: np.ndarray, fn: Any, higher: bool, groups: Optional[np.ndarray]) -> list[Optional[float]]:
    """90% bootstrap interval of the chosen one's gain over the choice without a search on the same selection rows (oriented
    so a positive gain is better), over whole entities when the split keeps them."""
    sign = 1.0 if higher else -1.0
    return _cluster_interval(lambda r: sign * (fn(y[r], chosen[r]) - fn(y[r], without[r])), len(y), groups)


def run(p: Any, recipe: str, screen: list[dict[str, Any]], budget: dict[str, Any], *, base: Optional[dict[str, Any]] = None,
        checkpoint: Any = None) -> dict[str, Any]:
    """The search and its selection record (see the module); ``screen``: the screen's ranked rows (best first); ``base``:
    the stage's choice before the search (``{"family", "params"}``: the review's family and the change it kept). The
    selection fold is the one ``prepare`` set aside (``p.search_cut``); called without one, it is set aside here, after
    which the bundle's folds are the search rows' (a caller outside the engine)."""
    try:
        import optuna
    except ImportError:
        return skipped(budget, "Optuna is not installed here")
    from dclab_core import selection

    from .data_value import _scorer

    bundle, spec = p.bundle, p.bundle.spec
    held_aside = getattr(p, "search_cut", None) or set_aside(bundle)
    if held_aside.get("skipped"):
        return skipped(budget, held_aside["skipped"])
    search_rows, select_rows, how = held_aside["search_rows"], held_aside["select_rows"], held_aside["how"]
    metric, higher = spec.primary_metric, spec.higher_is_better
    score_fn, _ = _scorer(bundle)
    X, y = bundle.X_train, bundle.y_train
    labels = y.to_numpy()
    split = spec.split_strategy
    entities = bundle.groups("train") if split in ("group", "time_group") else None
    folds = [(np.asarray(a), np.asarray(b)) for a, b in bundle.cv_splits]  # the search rows' folds, positions into the training rows
    if set(np.concatenate([np.concatenate([a, b]) for a, b in folds]).tolist()) & set(select_rows.tolist()):
        raise RuntimeError("a fold holds a row of the selection fold: the selection fold was not set aside before the folds")
    started, deadline = time.perf_counter(), time.perf_counter() + 60.0 * budget["minutes"]
    attempts: list[dict[str, Any]] = []
    failed: list[str] = []

    def cv_score(family: str, params: Optional[dict[str, Any]]) -> float:
        parts_y, parts_p = [], []
        for fit_rows, valid_rows in folds:
            if checkpoint is not None:
                checkpoint()
            out = rn.fit_and_predict(bundle, recipe, family, dict(params) if params else None, X.iloc[fit_rows].reset_index(drop=True),
                                     y.iloc[fit_rows].reset_index(drop=True), X.iloc[valid_rows].reset_index(drop=True), seed=rn.RANDOM_STATE + 1)
            parts_y.append(labels[valid_rows])
            parts_p.append(np.asarray(out["predictions"]))
        stacked = np.concatenate(parts_p) if parts_p[0].ndim == 1 else np.vstack(parts_p)
        value = float(score_fn(np.concatenate(parts_y), stacked))
        if not np.isfinite(value):
            raise ValueError("the pooled out-of-fold score is not a number")
        return value

    first = (base or {}).get("family")
    start_params = {str(k): v for k, v in ((base or {}).get("params") or {}).items()}
    order = ([first] if first else []) + [r["model"] for r in screen]
    families = list(dict.fromkeys(f for f in order if f))[: budget["families"]]

    def start(family: str) -> dict[str, Any]:
        return dict(start_params) if family == first else {}

    for family in families:  # each family's starting settings first: a candidate like any other (the first is the stage's own choice)
        if time.perf_counter() > deadline:
            break
        name = f"{family}:{'reviewed' if start(family) else 'default'}"
        try:
            attempts.append({"index": len(attempts), "name": name, "family": family, "params": start(family), "score": cv_score(family, start(family))})
        except ValueError as exc:
            failed.append(f"{name}: {type(exc).__name__}")
    per_family = max(1, budget["trials"] // max(1, len(families)))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for family in families:
        if time.perf_counter() > deadline:
            break
        study = optuna.create_study(direction="maximize" if higher else "minimize", sampler=optuna.samplers.TPESampler(seed=SEED))

        def objective(trial: Any, family: str = family) -> float:
            drawn = space(trial, family, bundle.task_type, weighted="review__class_weight" in start(family))
            if drawn is None:
                raise optuna.TrialPruned()
            params = {**start(family), **drawn}
            value = cv_score(family, params)
            attempts.append({"index": len(attempts), "name": f"{family}:trial{trial.number}", "family": family, "params": params, "score": value})
            return value

        remaining = max(1.0, deadline - time.perf_counter())
        study.optimize(objective, n_trials=per_family, timeout=remaining, catch=(ValueError,))
    if not attempts:
        return skipped(budget, "no candidate could be scored within the budget" + (f" ({'; '.join(failed[:3])})" if failed else ""))
    by_name = {a["name"]: a for a in attempts}
    held: dict[str, np.ndarray] = {}
    y_select = labels[select_rows]

    def predict_selection(name: str) -> np.ndarray:
        if name not in held:  # fitted on every search row, scored once on the selection fold
            a = by_name[name]
            out = rn.fit_and_predict(bundle, recipe, a["family"], dict(a["params"]) or None, X.iloc[search_rows].reset_index(drop=True),
                                     y.iloc[search_rows].reset_index(drop=True), X.iloc[select_rows].reset_index(drop=True), seed=rn.RANDOM_STATE + 1)
            held[name] = np.asarray(out["predictions"])
        return held[name]

    def on_selection(name: str, rows: Optional[np.ndarray]) -> Optional[float]:
        predictions = predict_selection(name)
        if rows is not None:  # the bootstrap's draws reuse the one fit's predictions
            try:
                return float(score_fn(y_select[rows], predictions[rows]))
            except ValueError:
                return None
        return float(score_fn(y_select, predictions))

    record = selection.choose({a["name"]: a["score"] for a in attempts}, on_selection, metric=metric, higher_is_better=higher, top_k=1,
                              split=how, search_rows=len(search_rows), selection_rows=len(select_rows), bootstrap=BOOTSTRAP)
    groups_select = entities[select_rows] if entities is not None else None
    if groups_select is not None:  # the selection fold's entities, drawn whole (choose draws rows)
        winner = predict_selection(record.chosen)
        record = record.model_copy(update={
            "selection_interval": _cluster_interval(lambda r: float(score_fn(y_select[r], winner[r])), len(y_select), groups_select),
            "note": (record.note + " Its interval draws whole entities of the selection fold.")[:1000]})
    chosen = by_name[record.chosen]
    without = by_name.get(f"{families[0]}:{'reviewed' if start(families[0]) else 'default'}")  # the stage's own choice, unsearched
    against = None
    if without is not None:
        sign = 1.0 if higher else -1.0
        same = without["name"] == chosen["name"]
        score_without = on_selection(without["name"], None)
        against = {"name": without["name"], "family": without["family"], "selection_score": round(score_without, 6),
                   "gain": 0.0 if same else round(sign * (record.selection_score - score_without), 6),
                   "gain_interval": [0.0, 0.0] if same else _paired_gain(y_select, predict_selection(chosen["name"]), predict_selection(without["name"]),
                                                                         score_fn, higher, groups_select)}
    pooled = "pooled out-of-fold"
    return {"family": chosen["family"], "params": chosen["params"], "budget": budget, "minutes": round((time.perf_counter() - started) / 60, 2),
            "attempts": len(attempts), "failed": failed[:20], "selection": record.model_dump(mode="json"), "without_search": against,
            "search_score_kind": pooled, "every_attempt": [{k: a[k] for k in ("index", "name", "family", "params", "score")} for a in attempts],
            "candidates": sorted(({k: a[k] for k in ("index", "name", "family", "params", "score")} for a in attempts),
                                 key=lambda a: -a["score"] if higher else a["score"])[:20],
            "rule": (f"Optuna (TPE, seed {SEED}) on {len(families)} families (the stage's own choice first, then the screen's next), on "
                     f"{len(folds)} folds of {len(search_rows)} search rows; the best by its {pooled} {metric} scored once on a selection fold "
                     f"of {len(select_rows)} rows set aside before any stage scored (34.6), beside the stage's choice without a search "
                     "scored the same way; the holdout untouched.")}
