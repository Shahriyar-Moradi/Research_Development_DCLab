"""What to collect next, for a project (34.14): ``dclab_core.collect.advise`` with the project's own readings.

What each column group adds is measured on the training folds with the model the models stage chose: the out-of-fold
predictions with every column, then with each group's columns made constant in turn (a constant column carries nothing,
and the recipe keeps its columns), read with the project's measure and a paired bootstrap on the same rows (over whole
entities when the split keeps them together). The holdout is never read. The domain comes from the project's words (with
no known domain each column is weighed on its own); the segments where more rows still helped from its last learning curve
(34.13), when that curve was drawn on the same table; the open data questions from the confirmed problem map's needs the
data card does not hold. The advice is kept on the project and its answer recorded.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ..expansion import runner as rn
from . import engine
from .data_value import _scorer


class CollectError(ValueError):
    """The project cannot be read here, with why."""


def _constant(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = 0.0 if pd.api.types.is_numeric_dtype(out[column]) else "__same__"
    return out


def _needs(project: dict[str, Any]) -> list[dict[str, Any]]:
    """The confirmed problem map's needs the project's data card does not hold (``mapping.coverage``)."""
    plan = project.get("programme_plan") or {}
    card = (project.get("data") or {}).get("card")
    if not plan.get("problem_map") or not card:
        return []
    from dclab_core import mapping
    from dclab_core.problem_map import ProblemMap

    try:
        return mapping.coverage(ProblemMap.model_validate(plan["problem_map"]), card)[:10]
    except (ValueError, TypeError, KeyError):
        return []


def measure(store: Any, project_id: str, *, actor: str = "human") -> dict[str, Any]:
    """Weigh the groups, write the advice to the project's ``collect_next`` and record it; returns the record."""
    from dclab_core import collect

    from ..jobs import checkpoint, progress

    project = store.get(project_id)
    models, features = store.read_stage(project_id, "models"), store.read_stage(project_id, "features")
    if not (models and features):
        raise CollectError("Run the project's stages up to the models stage first: the groups are weighed with the model it chose.")
    solution = engine.Solution(**project["solution"])
    if solution.problem_kind(engine.pack_of(project)).own_machine() is not None:
        raise CollectError("This kind of project runs its own machine; its column groups are not weighed here yet.")
    p = engine.prepare(store, project)
    bundle = p.bundle
    if not bundle.cv_splits:
        raise CollectError("The project has no training folds to weigh the groups on.")
    family = engine.chosen(models, "selected_model")
    recipe = engine.chosen(features, "selected_recipe") or (models.get("evidence") or {}).get("feature_recipe")
    X_train, y_train = bundle.X_train, bundle.y_train  # the training rows only
    columns = [c for c, role in p.roles.items() if role == "feature" and c in X_train]
    words = " ".join(str(x or "") for x in (project.get("goal"), (project.get("draft") or {}).get("problem"), project.get("name")))
    domain = collect.domain_of(words)
    metric_fn, fast = _scorer(bundle)
    rows = np.sort(np.concatenate([valid for _, valid in bundle.cv_splits]))  # every row some fold predicts, in one order
    position = {int(r): i for i, r in enumerate(rows)}
    planned = len(bundle.cv_splits) * (1 + len(collect.groups(columns, domain)))
    fits = {"n": 0}

    def predict_without(dropped: list[str]) -> np.ndarray:
        X = _constant(X_train, dropped) if dropped else X_train
        out = None
        for fit_rows, valid_rows in bundle.cv_splits:
            checkpoint()  # Stop works between fits
            fits["n"] += 1
            progress(step=fits["n"], of=max(planned, fits["n"]), way="what to collect next")
            got = np.asarray(rn.fit_and_predict(bundle, recipe, family, None, X.iloc[fit_rows].reset_index(drop=True),
                                                y_train.iloc[fit_rows].reset_index(drop=True), X.iloc[valid_rows].reset_index(drop=True),
                                                seed=rn.RANDOM_STATE + 1)["predictions"])
            if out is None:
                out = np.full((len(rows), *got.shape[1:]), np.nan)
            out[[position[int(r)] for r in valid_rows]] = got
        return out

    split = bundle.spec.split_strategy
    entities = bundle.groups("train")[rows] if split in ("group", "time_group") else None
    value = project.get("data_value") or {}
    fresh_value = value.get("document") if value.get("data_sha256") == (project.get("data") or {}).get("sha256") else None
    started = time.perf_counter()
    advice = collect.advise(columns, domain=domain, y=y_train.to_numpy()[rows], predict_without=predict_without, metric_fn=metric_fn,
                            higher_is_better=bundle.spec.higher_is_better, metric=bundle.spec.primary_metric, groups_of_rows=entities,
                            data_value=fresh_value, data_needs=_needs(project), bootstrap=200 if fast else 60)
    record = {"ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "document": advice.model_dump(mode="json"),
              "model": family, "recipe": recipe, "cv": bundle.cv_description, "fits": fits["n"], "seconds": round(time.perf_counter() - started, 1),
              "data_sha256": (project.get("data") or {}).get("sha256"), "learning_curve_used": fresh_value is not None}
    store.update(project_id, collect_next=record)
    store.log(project_id, "collect_next", {"domain": domain, "missing": [m.kind for m in advice.missing][:5],
                                           "adds": [r.group for r in advice.groups if r.verdict == "adds"][:5], "seconds": record["seconds"]})
    from ..experience import hooks

    hooks.collect_advised(project_id, advice, actor)
    return record


def view(project: dict[str, Any]) -> dict[str, Any]:
    return project.get("collect_next") or {}
