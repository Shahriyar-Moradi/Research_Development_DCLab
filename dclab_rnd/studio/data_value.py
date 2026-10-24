"""Would more rows help this project's model? (package 34.13) A separate analysis on the training folds only.

``measure`` draws the learning curve of the model the models stage chose (its family and the features stage's recipe,
default parameters, as screened) on the project's **training folds only**: each fold's fit rows cut to 25, 50, 75 and
100 %, the way the split is made (whole groups for a grouped split, the latest rows for a timed one, within each class
otherwise), scored on the fold's validation rows. ``dclab_core.data_value`` turns the curve into the answer (more rows
likely help, unlikely to help, cannot tell) with its interval and its assumption, and the same per segment where the
errors concentrate (the low-cardinality feature columns; a column that holds a person's data is never a segment).

It changes no stage record: the result is the project's own ``data_value`` field, and the held-back cases are never
read (the bundle's training rows and folds only). The answer is captured as the ledger point ``data.more_rows`` and a
``decision.proposed`` event; neither holds a table's value (a segment's level stays in the project's record).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..expansion import runner as rn
from . import engine

POINT = "data.more_rows"
MAX_SEGMENT_LEVELS = 20  # a column with more levels than this is not read as segments
NOTE = ("Measured on the training folds only, with the model the models stage chose (default parameters, as screened); the held-back cases "
        "were not read. The curve says what more rows of the same kind would do; it cannot say what a new column or cleaner labels would do.")


class DataValueError(ValueError):
    """The project cannot be measured here (no models stage yet, or a kind that runs its own machine)."""


def _scorer(bundle: Any) -> tuple[Any, bool]:
    """The primary metric as a fast function of (labels, predictions), and whether it is the fast one (the bootstrap is cut otherwise)."""
    task, metric = bundle.task_type, bundle.spec.primary_metric
    budget = bundle.spec.settings.get("alert_budget")
    n_classes = rn._n_classes(bundle)
    if metric == "recall_at_budget" and budget:
        return rn.budget_metric_fn(float(budget)), True
    try:
        return rn.primary_metric_fn(task, metric, n_classes), True
    except ValueError:
        precision_target = float(bundle.spec.settings.get("precision_target", 0.9))

        def slow(y: Any, p: Any) -> float:
            value = rn.compute_metrics(task, y, p, n_classes=n_classes, precision_target=precision_target, alert_budget=budget).get(metric)
            return float(value) if isinstance(value, (int, float)) else float("nan")
        return slow, False


def _segment_columns(p: Any, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Feature columns with 2 to MAX_SEGMENT_LEVELS levels on the training rows, never one that holds a person's data."""
    out = {}
    for column, role in p.roles.items():
        if role != "feature" or column not in X or column in p.personal:
            continue
        levels = X[column].nunique(dropna=True)
        if 2 <= levels <= MAX_SEGMENT_LEVELS and (not pd.api.types.is_numeric_dtype(X[column]) or levels <= 10):
            out[column] = X[column].to_numpy(dtype=object)
    return out


def measure(store: Any, project_id: str, *, actor: str = "human", repeats: int = 2, bootstrap: int = 200, log: Any = None) -> dict[str, Any]:
    """Draw the curve, keep the project's ``data_value`` record and capture the answer; returns the record. Raises
    :class:`DataValueError` with why when the project cannot be measured here."""
    from dclab_core import data_value as core

    from ..jobs import checkpoint, progress

    say = log or (lambda *_: None)
    project = store.get(project_id)
    models, features = store.read_stage(project_id, "models"), store.read_stage(project_id, "features")
    if not (models and features):
        raise DataValueError("Run the project's stages up to the models stage first: the curve is drawn for the model it chose.")
    solution = engine.Solution(**project["solution"])
    if solution.problem_kind(engine.pack_of(project)).own_machine() is not None:
        raise DataValueError("This kind of project runs its own machine; its learning curve is not measured here yet.")
    p = engine.prepare(store, project)
    bundle, spec = p.bundle, p.bundle.spec
    if not bundle.cv_splits:
        raise DataValueError("The project has no training folds to draw a curve on.")
    family = engine.chosen(models, "selected_model")
    recipe = engine.chosen(features, "selected_recipe") or (models.get("evidence") or {}).get("feature_recipe")
    X_train, y_train = bundle.X_train, bundle.y_train  # the training rows only: the held-back rows are never taken from the bundle
    y_values = y_train.to_numpy()
    score, fast = _scorer(bundle)
    split = spec.split_strategy
    order = np.arange(len(X_train)) if split in ("time", "time_group") else None  # a timed bundle's rows are in time order
    groups = bundle.groups("train") if split == "group" else None
    strata = y_values if p.kind.base_task != "regression" else None
    fits = {"n": 0}
    total = len(bundle.cv_splits) * (1 + (len(core.FRACTIONS) - 1) * (1 if order is not None else max(1, repeats)))

    def fit_predict(fit_rows: np.ndarray, valid_rows: np.ndarray) -> np.ndarray:
        checkpoint()  # Stop works between fits
        fits["n"] += 1
        progress(step=fits["n"], of=total, way="learning curve")
        out = rn.fit_and_predict(bundle, recipe, family, None, X_train.iloc[fit_rows].reset_index(drop=True),
                                 y_train.iloc[fit_rows].reset_index(drop=True), X_train.iloc[valid_rows].reset_index(drop=True),
                                 seed=rn.RANDOM_STATE + 1)
        return np.asarray(out["predictions"])

    started = time.perf_counter()
    say(f"The learning curve of {family} on the `{recipe}` features, on the training folds ({bundle.cv_description})")
    result = core.learning_curve(y_values, bundle.cv_splits, fit_predict, score, metric=spec.primary_metric, higher_is_better=spec.higher_is_better,
                                 groups=groups, order=order, strata=strata, segments=_segment_columns(p, X_train), repeats=repeats,
                                 bootstrap=bootstrap if fast else min(bootstrap, 60), seed=rn.RANDOM_STATE)
    record = {"ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "document": result.model_dump(mode="json"),
              "model": family, "recipe": recipe, "params": "default (as screened)", "split": split, "cv": bundle.cv_description,
              "rows_used": int(len(X_train)), "data_sha256": (project.get("data") or {}).get("sha256"), "fits": fits["n"],
              "seconds": round(time.perf_counter() - started, 1), "note": NOTE}
    store.update(project_id, data_value=record)  # its own field only: no stage record changes meaning
    store.log(project_id, "data_value", {"answer": result.answer, "metric": result.metric, "gain": result.slope.gain if result.slope else None,
                                         "segments": len(result.segments), "seconds": record["seconds"]})
    _capture(project_id, p, result, actor)
    return record


def _capture(project_id: str, p: Any, result: Any, actor: str) -> None:
    from ..experience import hooks
    from ..ledger import capture

    hooks.data_value_measured(project_id, result, actor)
    capture.more_rows(project_id, result, rows=int(len(p.bundle.train_idx)), columns=sum(1 for r in p.roles.values() if r == "feature"),
                      positive_rate=p.bundle.spec.positive_rate)


def view(project: dict[str, Any]) -> dict[str, Any]:
    """The last measurement's record, or ``{}``."""
    return project.get("data_value") or {}


def main(argv: Optional[list[str]] = None) -> int:
    """``python -m dclab_rnd.studio.data_value PROJECT_ID --projects DIR``: measure a project of a file workspace and print the answer."""
    import argparse
    import json

    from .store import ProjectStore

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("project")
    parser.add_argument("--projects", required=True, help="the workspace's projects folder")
    args = parser.parse_args(argv)
    record = measure(ProjectStore(args.projects), args.project, log=lambda line: print(line, flush=True))
    doc = record["document"]
    print(json.dumps({"answer": doc["answer"], "plain": doc["plain"], "assumption": doc["assumption"],
                      "segments": [{k: s[k] for k in ("column", "level", "answer", "reason")} for s in doc["segments"]]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
