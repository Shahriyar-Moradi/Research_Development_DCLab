"""Improve a finished model, practically (the owner, 2026-10-20: "improve the model for me": show the ways, then do it).

``run`` tries, on the project's **training folds only** (the held-back cases are never touched), several ways to do better than
the model the project chose, each cross-validated on the same folds as the current model so the comparison is paired:

- **model settings:** the chosen family's other tuning candidates are already in the final stage's record, so none is run twice;
  what is run is what was not: the same family with balanced class weights where it can;
- **other model families:** the next families of the screen, each with its own tuning candidates;
- **features:** another recipe of the features stage, on the chosen family;
- **a blend:** the average of the best models' out-of-fold scores (no refit).

A way is called a gain only when ``dclab_core.review.adopt`` says so: its paired gain on the folds beats the tuning margin set
in advance and the one-sided 90% bound of the fold differences. Because the ways are chosen on the very folds that score them,
a gain here is optimistic: the report says so, and a model adopted from it is scored on the held-back cases only by the final
stage, which reads them a second time (a written reason, a holdout that is no longer fresh).

The result is the project's ``improvements`` record (the latest run, which the chat and the adoption read) and the same record
appended to ``improvement_runs``, every run the project had, numbered, newest last (the owner, 2026-11-03: keep every improvement
and show them on the Models page): counts, rates and parameters, never a row.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from ..expansion import runner as rn
from . import engine

MAX_OTHER_FAMILIES = 2
MAX_PER_FAMILY = 2
MAX_RUNS = 50  # the runs a project keeps; the oldest goes first
FULL_RUNS = 5  # the latest runs kept whole; older ones without their per-fold lists, so the project's document stays small
NOTE = ("Each way was cross-validated on the training folds only, the same folds as the current model, and called a gain only when its paired "
        "gain beats the margin set in advance and the one-sided 90% bound of the fold differences. The ways were chosen on the folds that score "
        "them, so a gain here is optimistic. The held-back cases were not touched; a model adopted from this is scored on them only by the "
        "final stage, which reads them a second time and needs a written reason.")


def _scores(row: dict[str, Any], metric: str) -> list[float]:
    return [float(f[metric]) for f in row["fold_metrics"] if f.get(metric) is not None]


def _blend(bundle: Any, rows: list[dict[str, Any]], metric: str, task: str, n_classes: Optional[int], budget: Optional[float]) -> list[float] | None:
    """Per-fold scores of the mean of the rows' out-of-fold predictions (no refit); None when they cannot be paired."""
    if len(rows) < 2 or any("_oof_pred" not in r or "_oof_index" not in r for r in rows):
        return None
    sizes = [int(f["validation_rows"]) for f in rows[0]["fold_metrics"]]
    first = np.asarray(rows[0]["_oof_index"])
    if any(not np.array_equal(first, np.asarray(r["_oof_index"])) for r in rows[1:]):
        return None  # not the same rows in the same order
    mean = np.mean([np.asarray(r["_oof_pred"], dtype=float) for r in rows], axis=0)
    y = bundle.y_train.to_numpy()[first]
    score = rn.budget_metric_fn(budget) if metric == "recall_at_budget" else rn.primary_metric_fn(task, metric, n_classes)
    out, start = [], 0
    for size in sizes:
        out.append(float(score(y[start:start + size], mean[start:start + size])))
        start += size
    return out


def run(store: Any, project_id: str, *, budget_seconds: float = 300.0, log: Any = None) -> dict[str, Any]:
    """Try the ways; keep and return the project's ``improvements`` record. Raises ValueError with why when the project cannot be improved
    here (no finished models and final stages, or a kind that runs its own machine)."""
    from dclab_core.review import adopt

    from ..jobs import checkpoint, progress

    say = log or (lambda *_: None)
    project = store.get(project_id)
    models, final = store.read_stage(project_id, "models"), store.read_stage(project_id, "final")
    features = store.read_stage(project_id, "features")
    if not (models and final and features):
        raise ValueError("Run the project's stages up to the final one first: there is no model to improve yet.")
    p = engine.prepare(store, project)
    if not hasattr(p, "bundle") or getattr(p.bundle, "cv_splits", None) is None or len(p.bundle.cv_splits) < 3:
        raise ValueError("This kind of project runs its own machine, or has too few folds, so the ways cannot be compared here.")
    bundle, spec = p.bundle, p.bundle.spec
    metric, higher, task = p.metric, spec.higher_is_better, bundle.task_type
    n_classes = rn._n_classes(bundle)
    recipe = engine.chosen(features, "selected_recipe")
    family = engine.chosen(models, "selected_model")
    evidence = (final.get("evidence") or {})
    adopted_before = project.get("adopted_improvement") or {}
    base_params = dict(adopted_before["params"]) if adopted_before.get("family") == family and isinstance(adopted_before.get("params"), dict) \
        else dict(evidence.get("selected_model_params") or {})
    margin = float(spec.settings.get("tuning_margin", 0.001))
    relative = bool(spec.settings.get("tuning_margin_relative", False))
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []  # every cross-validation done, to blend and to report

    say(f"The current model ({family}, recipe {recipe}) on the training folds")
    base = rn.cross_validate(bundle, recipe, family, base_params or None, config_id="current")
    base_scores = _scores(base, metric)
    needed = abs(rn._mean(base, metric)) * margin if relative else margin
    results: list[dict[str, Any]] = []
    skipped: list[str] = []

    def judge(way: str, aspect: str, label: str, fam: str, params: dict[str, Any] | None, row: dict[str, Any], recipe_name: str, adoptable: bool = True) -> None:
        scores = _scores(row, metric)
        verdict = adopt(base_scores, scores, higher, needed) if len(scores) == len(base_scores) else {"adopted": False, "why": "the folds do not pair", "mean_gain": None}
        results.append({"id": way, "aspect": aspect, "label": label, "family": fam, "recipe": recipe_name, "params": params or {}, "adoptable": adoptable,
                        "mean": rn._mean(row, metric), "std": rn._std(row, metric), "fold_scores": [round(s, 6) for s in scores],
                        "gain": verdict.get("mean_gain"), "fold_gains": verdict.get("fold_gains"), "folds_better": verdict.get("folds_better"),
                        "clears": bool(verdict["adopted"]), "why": verdict["why"], "seconds": row.get("elapsed_seconds")})

    def out_of_time(what: str) -> bool:
        if time.perf_counter() - started > budget_seconds:
            skipped.append(f"{what}: the time budget of {int(budget_seconds)} s was used")
            return True
        return False

    steps = []  # (way id, aspect, label, family, params, recipe)
    if rn.change_params({"class_weight": "balanced"}, family, task) and "review__class_weight" not in base_params:
        steps.append(("weights", "Class balance", f"{family} with balanced class weights", family,
                      {**base_params, **rn.change_params({"class_weight": "balanced"}, family, task)}, recipe))
    screened = {r["model"]: r for r in (models.get("evidence") or {}).get("model_results", [])}
    ranking = [r["model"] for r in (models.get("evidence") or {}).get("ranking", []) if r["model"] != family][:MAX_OTHER_FAMILIES]
    for other in ranking:
        for cand in rn.optimization_candidates(other, task)[:MAX_PER_FAMILY]:
            steps.append((f"{other}:{cand['candidate_id']}", "Other model families", f"{other} ({cand['candidate_id']})", other, dict(cand["params"]), recipe))
    others = [r for r in (features.get("evidence") or {}).get("stage_results", []) if r.get("recipe") and r["recipe"] != recipe]
    if others:
        alt = max(others, key=lambda r: rn._mean(r, metric) * (1 if higher else -1))["recipe"]
        steps.append((f"recipe:{alt}", "Features", f"{family} on the `{alt}` features", family, dict(base_params), alt))

    for i, (way, aspect, label, fam, params, recipe_name) in enumerate(steps, 1):
        checkpoint()
        if out_of_time(label):
            continue
        progress(step=i, of=len(steps), way=label)
        say(f"Trying: {label}")
        try:
            row = rn.cross_validate(bundle, recipe_name, fam, params or None, config_id=way)
        except Exception as exc:  # noqa: BLE001 - a way that cannot run (a missing library, a family that rejects the data) is said, never the run's end
            skipped.append(f"{label}: could not run ({type(exc).__name__})")
            continue
        rows.append(row)
        judge(way, aspect, label, fam, params, row, recipe_name, adoptable=fam == family and recipe_name == recipe)
    # a blend of the current model and the best of the others that were cross-validated here (out-of-fold, no refit)
    checkpoint()
    best_rows = sorted(rows, key=lambda r: rn._mean(r, metric) * (1 if higher else -1), reverse=True)[:2]
    blend = _blend(bundle, [base, *best_rows], metric, task, n_classes, spec.settings.get("alert_budget")) if best_rows else None
    if blend is not None:
        judge("blend", "A blend of models", "the average of the current model and the best " + str(len(best_rows)) + " of these", family, None,
              {"model": "blend", "metrics": {metric: {"mean": float(np.mean(blend)), "std": float(np.std(blend, ddof=1)) if len(blend) > 1 else 0.0}},
               "fold_metrics": [{metric: v} for v in blend], "elapsed_seconds": None}, recipe, adoptable=False)
    elif best_rows:
        skipped.append("a blend: the predictions could not be paired")

    results.sort(key=lambda r: (r["gain"] is None, -(r["gain"] if r["gain"] is not None else 0)))  # the gain is oriented: more is better, whichever way the metric points
    best = next((r for r in results if r["clears"] and r["adoptable"]), None)
    aspects: dict[str, dict[str, Any]] = {}
    for r in results:
        a = aspects.setdefault(r["aspect"], {"aspect": r["aspect"], "tried": 0, "clear": 0, "best_gain": None})
        a["tried"] += 1
        a["clear"] += 1 if r["clears"] else 0
        if r["gain"] is not None and (a["best_gain"] is None or r["gain"] > a["best_gain"]):
            a["best_gain"] = r["gain"]
    record = {"ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "metric": metric, "higher_is_better": bool(higher), "margin": needed,
              "folds": len(base_scores), "current": {"family": family, "recipe": recipe, "params": base_params, "mean": rn._mean(base, metric),
                                                     "std": rn._std(base, metric), "fold_scores": [round(s, 6) for s in base_scores]},
              "candidates": results, "aspects": list(aspects.values()), "best": best["id"] if best else None, "skipped": skipped, "adopted": None,
              "seconds": round(time.perf_counter() - started, 1), "note": NOTE}
    runs = history(store.get(project_id))  # read now: a run adopted meanwhile keeps its mark
    record["run"] = (max((int(r.get("run") or 0) for r in runs), default=0) + 1)
    runs = _compact((runs + [record])[-MAX_RUNS:])
    store.update(project_id, improvements=record, improvement_runs=runs)  # their own fields only: a cell or a note written meanwhile stays
    store.log(project_id, "improve_run", {"tried": len(results), "clear": sum(r["clears"] for r in results), "best": record["best"], "seconds": record["seconds"]})
    return record


def _compact(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Runs older than the latest FULL_RUNS keep their ways' scores, gains and verdicts, not the per-fold lists (the page shows the
    means; the latest runs keep everything)."""
    out = []
    for i, r in enumerate(runs):
        if i < len(runs) - FULL_RUNS:
            r = {**r, "current": {k: v for k, v in (r.get("current") or {}).items() if k != "fold_scores"},
                 "candidates": [{k: v for k, v in c.items() if k not in ("fold_scores", "fold_gains")} for c in r.get("candidates") or []]}
        out.append(r)
    return out


def history(project: dict[str, Any]) -> list[dict[str, Any]]:
    """Every improvement run the project kept, oldest first, each with its number. A project improved before runs were kept has its
    one latest run as run 1."""
    runs = [dict(r) for r in project.get("improvement_runs") or [] if isinstance(r, dict)]
    if not runs and isinstance(project.get("improvements"), dict) and project["improvements"].get("candidates") is not None:
        runs = [{**project["improvements"], "run": project["improvements"].get("run") or 1}]
    for i, r in enumerate(runs, 1):
        r.setdefault("run", i)
    return runs


def adopt_candidate(store: Any, project_id: str, candidate: str) -> dict[str, Any]:
    """The person adopts a way that cleared: the project keeps its parameters for the final stage to use (the same hook the review's
    kept changes use). A way of another family or recipe is not adoptable here: choose it in Models first."""
    project = store.get(project_id)
    rec = project.get("improvements") or {}
    found = next((c for c in rec.get("candidates") or [] if c["id"] == candidate), None)
    if found is None:
        raise KeyError("That way is not in the last improvement run.")
    if not found["clears"]:
        raise ValueError("That way did not clear the margin on the folds, so it is not offered for adoption.")
    if project.get("running") == "improve":
        raise ValueError("An improvement run is going on: adopt once it has finished, from its new report.")
    now = {"family": engine.chosen(store.read_stage(project_id, "models"), "selected_model"),
           "recipe": engine.chosen(store.read_stage(project_id, "features"), "selected_recipe")}
    cur = rec.get("current") or {}
    if (cur.get("family"), cur.get("recipe")) != (now["family"], now["recipe"]):
        raise ValueError(f"That run improved {cur.get('family')} on the {cur.get('recipe')} features, but the project now uses {now['family']} on the "
                         f"{now['recipe']} features: run the improvement again for the current model.")
    if not found["adoptable"]:
        raise ValueError("That way changes the model family, the features or is a blend: another family is chosen on the Models page (which re-runs the final stage); other features and a blend cannot be adopted yet.")
    project["adopted_improvement"] = {"id": candidate, "family": found["family"], "recipe": found["recipe"], "params": found["params"], "gain": found["gain"],
                                      "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    rec["adopted"] = candidate
    project["improvements"] = rec
    runs = history(project)
    for r in runs:  # the run the way came from carries the mark too (the latest run is the one adopted from)
        if r.get("run") == rec.get("run", runs[-1]["run"] if runs else None):
            r["adopted"] = candidate
    project["improvement_runs"] = _compact(runs)
    store.save(project)
    return project["adopted_improvement"]


__all__ = ["run", "adopt_candidate", "history", "NOTE"]
