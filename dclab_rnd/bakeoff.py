"""Package 34.5: the model bake-off. DCLab's engine against FLAML on the same rows, the same folds, the same metric.

Each dataset becomes one studio project, made the way the product makes it from a sample (``studio.data.use_sample``
and the R&D's solution for it). Then two engines run on it, each in its own process (so its time and peak memory are
its own):

- ``dclab``: the studio engine as it is, all five stages (``studio.engine.run_all``), no model offered to any purpose.
  Its holdout predictions are taken from the one fit of its final stage and must reproduce the number in its record.
- ``flaml``: FLAML's AutoML on the same prepared table (``engine.prepare``: the same sample of rows, the same locked
  holdout), given the columns the engine's own tabular recipes start from (no identifier, time key, group, text or
  forbidden column), searching on the engine's own training folds (passed as a fixed splitter, so FLAML never
  reshuffles them), scored by the same metric, within a time budget. It sees the holdout rows once, to predict.

The holdout is read once per engine; a compare step then scores both from the saved predictions on the same holdout
rows (checked: the same rows, the same labels), with each engine's bootstrap interval and a paired bootstrap of the
difference. The rule that decides package 34.8 is fixed in the campaign's manifest before the first run, with its
hash; a run refuses to go on under a changed rule.

    python -m dclab_rnd.bakeoff plan                 # what would run, the rule, what is done
    python -m dclab_rnd.bakeoff run --yes            # every dataset, resumable (a finished engine is never rerun)
    python -m dclab_rnd.bakeoff report               # the table and, when every dataset is done, the decision

Results: ``evidence/campaigns/bakeoff_v1/results/BKO-NNN_<dataset>.json`` (written once, never replaced) and
``BKO-100_decision.json``. The runs' workspaces, FLAML's search logs and the predictions are kept under
``agent_runs/bakeoff_v1/`` (not committed). Make target: ``make bakeoff`` (plan) and ``make bakeoff ARGS="run --yes"``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "bakeoff_v1"
CAMPAIGN = ROOT / "evidence" / "campaigns" / CAMPAIGN_ID
WORK = ROOT / "agent_runs" / CAMPAIGN_ID
ENGINES = ("dclab", "flaml")
DEFAULT_BUDGET_S = 300  # FLAML's search time per dataset: about five minutes, as the brief asks on this 8 GB machine
PAIRED_DRAWS = 2000
SEED = 42
MIN_FREE_BYTES = 2 * 1024**3  # stop starting runs below 2 GB of free disk (the owner's rule)
ENGINE_TIMEOUT_S = 90 * 60
FLAML_EXTRA_S = 30 * 60  # FLAML's refit on all training rows and its prediction come after its search budget
FLAML_SETTINGS = {"eval_method": "cv", "seed": 42, "auto_augment": False, "retrain_full": True, "verbose": 0}

# smallest first, so a problem shows early; the numbers BKO-001.. follow this order and never change
DATASETS: tuple[str, ...] = (
    "heart_disease", "breast_cancer", "german_credit", "spambase", "wine_quality", "telco_churn", "mushroom",
    "online_shoppers", "credit_default", "adult", "bank_marketing", "letter_recognition", "ecommerce_clothing_reviews",
    "credit_card_fraud",
)
LEFT_OUT = {
    "hyperack": "known leaks (the final fares); the brief allows it only with its leaks removed, in a later version",
    "bike_sharing_daily": "a daily time series: the engine derives past-only lags that a tabular search has no equal of; "
                          "not one of the expansion samples the brief names (imbalanced fraud, multiclass, text plus tabular)",
}

# The rule that decides package 34.8, fixed before the first run (its hash goes into the manifest).
RULE: dict[str, Any] = {
    "id": "bakeoff_rule_v1",
    "decides": "package 34.8: FLAML as one more family in the engine's models stage",
    "arm": "flaml (FLAML's AutoML on the engine's prepared table and training folds, the time budget below)",
    "statistic": (f"per dataset, a paired bootstrap of the holdout difference on the same holdout rows ({PAIRED_DRAWS} draws, "
                  "whole groups when the split is grouped, rows otherwise), oriented so that a positive difference means FLAML "
                  "scored better on the dataset's primary metric; the 95% percentile interval"),
    "ahead": "FLAML is ahead on a dataset when the interval's lower end is above 0",
    "behind": "FLAML is behind on a dataset when the interval's upper end is below 0",
    "go": ("34.8 goes ahead when FLAML is ahead on at least a third of the datasets where both engines finished "
           "(rounded up), and ahead on more datasets than it is behind"),
    "not_judged": {"ecommerce_clothing_reviews": "FLAML sees no text (the engine's text recipe is DCLab's), while 34.8 would give "
                                                 "FLAML the recipe's features: reported, never counted"},
    "failures": ("a dataset where FLAML failed after its retry counts as FLAML behind; one where only DCLab failed is left out "
                 "and listed in the decision; the decision is written only when every dataset was compared or failed after its retry"),
    "min_datasets": 10,
    "too_few": "fewer than 10 datasets judged: undecided, 34.8 waits",
    "time_and_memory": "reported beside the scores, never in the rule",
}


def rule_sha256(rule: dict[str, Any] | None = None) -> str:
    return hashlib.sha256(json.dumps(rule or RULE, sort_keys=True).encode()).hexdigest()


def decide(rows: list[dict[str, Any]], rule: dict[str, Any] | None = None, failures: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """The rule applied to the compared datasets (each with ``paired``: low and high, positive = FLAML better) and to the
    datasets where an engine failed after its retry (``failures``: dataset -> the engines that failed)."""
    rule = rule or RULE
    skip = set(rule.get("not_judged") or {})
    done = [r for r in rows if r["dataset"] not in skip and r.get("paired") and r["paired"].get("low") is not None]
    failed = {d: sorted(e) for d, e in (failures or {}).items() if d not in skip}
    flaml_failed = sorted(d for d, e in failed.items() if "flaml" in e)
    ahead = sorted(r["dataset"] for r in done if r["paired"]["low"] > 0)
    behind = sorted([r["dataset"] for r in done if r["paired"]["high"] < 0] + flaml_failed)
    judged = len(done) + len(flaml_failed)
    need = math.ceil(judged / 3)
    if judged < rule["min_datasets"]:
        outcome, go = "undecided", None
    else:
        go = len(ahead) >= need and len(ahead) > len(behind)
        outcome = "go" if go else "no go"
    return {"rule_id": rule["id"], "rule_sha256": rule_sha256(rule), "datasets_judged": judged, "needed_ahead": need,
            "flaml_ahead": ahead, "flaml_behind": behind, "no_clear_difference": sorted(r["dataset"] for r in done
                                                                                      if r["dataset"] not in ahead + behind),
            "not_judged": sorted(r["dataset"] for r in rows if r["dataset"] in skip),
            "left_out_dclab_failed": sorted(d for d, e in failed.items() if e == ["dclab"]),
            "outcome": outcome, "package_34_8": {True: "goes ahead", False: "does not go ahead", None: "waits"}[go]}


# ---------------------------------------------------------------------------------------------- the campaign's manifest


def protocol(budget_s: int, datasets: tuple[str, ...] | list[str]) -> dict[str, Any]:
    from dclab_rnd.expansion import datasets as ds, runner as rn

    return {
        "datasets": list(datasets), "left_out": LEFT_OUT, "flaml_budget_seconds": int(budget_s),
        "holdout": f"the studio engine's locked holdout ({ds.HOLDOUT_FRACTION:.0%}: the latest rows under a time split, "
                   "unseen groups under a group split, stratified random otherwise), read once by each engine",
        "folds": "the engine's training folds (bundle.cv_splits): FLAML searches on the same folds through a fixed splitter",
        "rows": "the engine's own sample (at most 20,000 rows, label-independent, its seed): both engines see the same rows",
        "metric": "the engine's primary metric for the dataset (its kind's default), both scored by the same function",
        "intervals": f"each engine: the engine's own bootstrap ({rn.BOOTSTRAP_DRAWS} draws, 95%, clustered when grouped); "
                     f"the difference: paired bootstrap, {PAIRED_DRAWS} draws",
        "time_and_memory": "wall minutes and the process's peak resident memory, each engine in its own process",
        "engine_budget": "the engine has no time knob: it runs as it is and its minutes are measured",
        "flaml_settings": FLAML_SETTINGS, "flaml_metric": FLAML_METRIC, "flaml_version": _flaml_version(),
        "paired_draws": PAIRED_DRAWS, "seed": SEED,
        "code_sha256": _code_sha256(),  # the functions that apply the rule and run the arms: an edit is a new campaign
    }


PINNED_CODE = ("decide", "paired_bootstrap", "compare", "run_dclab", "run_flaml", "flaml_columns", "_flaml_frame", "_metric_fn",
               "solution_from_suggestion", "FixedFolds")


def _code_sha256() -> str:
    import inspect

    module = sys.modules[__name__]
    return hashlib.sha256("\n".join(inspect.getsource(getattr(module, name)) for name in PINNED_CODE).encode()).hexdigest()


def _flaml_version() -> str | None:
    try:
        import flaml
    except ImportError:
        return None
    return flaml.__version__


def frozen(budget_s: int, datasets: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """What is fixed before the run: the rule and the protocol, with their hashes."""
    proto = protocol(budget_s, datasets)
    return {"rule": RULE, "rule_sha256": rule_sha256(), "protocol": proto,
            "protocol_sha256": hashlib.sha256(json.dumps(proto, sort_keys=True).encode()).hexdigest()}


def freeze(campaign: Path, budget_s: int, datasets: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Write the manifest once, before any run; afterwards check that the rule and the protocol are still the same."""
    fixed = frozen(budget_s, datasets)
    path = campaign / "manifest.json"
    if path.is_file():
        manifest = json.loads(path.read_text())
        for key in ("rule_sha256", "protocol_sha256"):
            if manifest.get(key) != fixed[key]:
                raise SystemExit(f"{path}: the {key.split('_')[0]} changed after it was fixed before the run. A changed rule or "
                                 "protocol is a new campaign version (bakeoff_v2), never an edit of this one.")
        return manifest
    import flaml

    manifest = {
        "campaign_id": campaign.name, "package": "34.5",
        "objective": ("The model bake-off: DCLab's studio engine as it is against FLAML's AutoML on the same rows, holdout, "
                      "training folds and metric; per dataset the holdout score with its interval, minutes and peak memory. "
                      "The rule below decides package 34.8 and was fixed before the first run."),
        "frozen_at": _now(), "git_commit": _git_commit(), "flaml_version": flaml.__version__, "python": sys.version.split()[0],
        **fixed,
        "command": "python -m dclab_rnd.bakeoff run --yes",
        "limits": ["One holdout per dataset: one draw of the data, not a distribution over datasets.",
                   "FLAML gets the engine's raw tabular columns, not its engineered features or its text recipe: on the text "
                   "dataset it sees no text.", "Measured on an 8 GB MacBook Air with a five-minute FLAML budget; a longer "
                   "budget or AutoGluon (later, on the M1 Max) may change the result.",
                   "Never a production approval: a benchmark result alone says nothing about deployment."],
    }
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign / "results").mkdir(exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:  # never replaced
        handle.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ---------------------------------------------------------------------------------------------- the projects


def solution_from_suggestion(suggestion: dict[str, Any], frame: Any) -> dict[str, Any]:
    """The Solution a person saves from the R&D's suggestion for a sample (the time column is unticked under
    Forbidden, as the form asks; reasons cut to the field's length)."""
    target, time_column = suggestion["target"], suggestion.get("time_column")
    task = suggestion.get("task") or "binary"
    fields: dict[str, Any] = {
        "target": target, "task": task, "prediction_moment": suggestion["prediction_moment"],
        "forbidden": [{"column": f["column"], "reason": str(f.get("reason") or "")[:600]} for f in suggestion.get("forbidden") or []
                      if f["column"] not in (time_column, target)],
        "identifiers": [c for c in suggestion.get("identifiers") or [] if c != time_column],
        "time_column": time_column, "group_column": suggestion.get("group_column"),
        "text_columns": list(suggestion.get("text_columns") or []),
    }
    if task == "binary":
        values = {str(v) for v in frame[target].dropna().unique()}
        fields["positive_label"] = suggestion.get("positive_label") or ("Yes" if values == {"Yes", "No"} else "1")
    return fields


def _save_solution(store: Any, project: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    """The person's Solution page, saved through the move ``set_solution`` (checked and logged by the workflow graph)."""
    from dclab_rnd.studio import graph as studio_graph
    from dclab_rnd.studio.solution import Solution

    solution = Solution(**fields)
    solution.check_columns(list(project["data"]["columns"]))
    verdict = studio_graph.check(project, "set_solution", "human", target=solution.target)
    if not verdict.allowed:
        raise ValueError(f"the Solution page was refused: {verdict.message}")
    project["solution"] = solution.model_dump()
    project = store.save(project)
    studio_graph.log(store, project["id"], verdict, store.get(project["id"]), outcome="done: the R&D's solution for the sample")
    store.log(project["id"], "solution_saved", {"target": solution.target, "task": solution.task, "by": "bakeoff"})
    return store.get(project["id"])


def _write_pointer(work: Path, store: Any, project: dict[str, Any], dataset: str) -> dict[str, Any]:
    pointer = {"dataset": dataset, "project_id": project["id"], "data_sha256": project["data"]["sha256"], "made_at": _now()}
    (work / "project.json").write_text(json.dumps(pointer, indent=2) + "\n")
    return pointer


def prepare_sample(work: Path, key: str) -> dict[str, Any]:
    """The dataset's project in ``work/workspace``, made once (both engines open it)."""
    if (work / "project.json").is_file():
        return json.loads((work / "project.json").read_text())
    from dclab_rnd.studio import ProjectStore, data as sd

    work.mkdir(parents=True, exist_ok=True)
    store = ProjectStore(work / "workspace")
    catalog = {s["key"]: s for s in sd.sample_catalog()}
    entry = catalog[key]
    project = store.create(entry["name"][:60], entry.get("industry") or "general", entry.get("goal") or entry["name"])
    project = sd.use_sample(store, project["id"], key)
    frame = sd.load_table(sd.data_path(store, project))
    project = _save_solution(store, project, solution_from_suggestion(project["suggestion"], frame))
    return _write_pointer(work, store, project, key)


def prepare_frame(work: Path, name: str, frame: Any, fields: dict[str, Any]) -> dict[str, Any]:
    """A project from a frame and its solution's fields (the tests' tiny table)."""
    from dclab_rnd.studio import ProjectStore, data as sd

    if (work / "project.json").is_file():
        return json.loads((work / "project.json").read_text())

    work.mkdir(parents=True, exist_ok=True)
    store = ProjectStore(work / "workspace")
    project = store.create(name, "general", name)
    frame.to_csv(store.data_dir(project["id"]) / "table.csv", index=False)
    project = sd.attach_data(store, project["id"], "table.csv")
    project = _save_solution(store, project, fields)
    return _write_pointer(work, store, project, name)


def _open(work: Path) -> tuple[Any, str]:
    from dclab_rnd.studio import ProjectStore

    pointer = json.loads((work / "project.json").read_text())
    store = ProjectStore(work / "workspace")
    project = store.get(pointer["project_id"])
    if project["data"]["sha256"] != pointer["data_sha256"]:
        raise ValueError("the project's table changed since the bake-off made it")
    return store, pointer["project_id"]


# ---------------------------------------------------------------------------------------------- the engines


def _split_record(prepared: Any) -> dict[str, Any]:
    bundle = prepared.bundle
    rows = np.asarray(prepared.source_rows)[np.asarray(bundle.test_idx)]
    y_test = np.asarray(bundle.y_test)
    return {"strategy": bundle.spec.split_strategy, "holdout": bundle.holdout_description, "folds": bundle.cv_description,
            "train_rows": int(len(bundle.train_idx)), "holdout_rows": int(len(bundle.test_idx)),
            "holdout_rows_sha256": hashlib.sha256(np.ascontiguousarray(rows.astype(np.int64)).tobytes()).hexdigest(),
            "holdout_labels_sha256": hashlib.sha256(np.ascontiguousarray(y_test.astype(float)).tobytes()).hexdigest(),
            "fold_sizes": [[int(len(a)), int(len(b))] for a, b in bundle.cv_splits],
            "folds_sha256": hashlib.sha256(b"|".join(np.ascontiguousarray(np.asarray(part, dtype=np.int64)).tobytes()
                                                    for fold in bundle.cv_splits for part in fold)).hexdigest(),
            "model_columns": flaml_columns(prepared), "sampling": bundle.sampling}


def _metric_fn(prepared: Any) -> tuple[str, bool, Callable[[Any, Any], float]]:
    from dclab_rnd.expansion import runner as rn

    bundle, spec = prepared.bundle, prepared.bundle.spec
    metric = spec.primary_metric
    if metric == "recall_at_budget":
        return metric, True, rn.budget_metric_fn(spec.settings["alert_budget"])
    return metric, spec.higher_is_better, rn.primary_metric_fn(bundle.task_type, metric, rn._n_classes(bundle))


def _save(work: Path, engine: str, info: dict[str, Any], prepared: Any, predictions: np.ndarray) -> dict[str, Any]:
    bundle = prepared.bundle
    groups = bundle.groups("test")
    np.savez_compressed(work / f"{engine}.npz", predictions=np.asarray(predictions, dtype=float), y_test=np.asarray(bundle.y_test),
                        holdout_rows=np.asarray(prepared.source_rows)[np.asarray(bundle.test_idx)],
                        groups=np.asarray(groups if groups is not None else [], dtype=str))
    info = {**info, "engine": engine, "status": "ok", "finished_at": _now(), "split": _split_record(prepared),
            "peak_memory_mb": _peak_mb()}
    (work / f"{engine}.json").write_text(json.dumps(info, indent=2, default=_plain) + "\n")
    return info


def run_dclab(work: Path) -> dict[str, Any]:
    """The studio engine's five stages; the predictions of its final stage's one holdout fit, checked against its record."""
    from unittest import mock

    from dclab_rnd.expansion import runner as rn
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import engine

    store, pid = _open(work)
    started = time.perf_counter()
    last: dict[str, Any] = {}
    real = rn.fit_and_predict

    def recording(*args: Any, **kwargs: Any) -> dict[str, Any]:
        out = real(*args, **kwargs)
        last.clear()
        last.update(out=out, rows=len(args[6]) if len(args) > 6 else len(kwargs["X_eval"]))
        return out

    with private_scope(True):  # no model is offered to any purpose: the engine's own rules decide
        for stage in engine.STAGE_KEYS[:-1]:
            engine.execute(store, pid, stage)
        with mock.patch.object(rn, "fit_and_predict", recording):  # the final stage's last fit is its one holdout fit
            engine.execute(store, pid, engine.STAGE_KEYS[-1])
    seconds = time.perf_counter() - started
    final = store.read_stage(pid, engine.STAGE_KEYS[-1])
    prepared = engine.prepare(store, store.get(pid))
    metric, higher, fn = _metric_fn(prepared)
    predictions = np.asarray(last["out"]["predictions"])
    if last.get("rows") != len(prepared.bundle.test_idx):
        raise RuntimeError("the final stage's last fit did not predict the holdout rows")
    recorded = final["evidence"]["holdout_metrics"][metric]
    got = fn(np.asarray(prepared.bundle.y_test), predictions)
    if recorded is None or not math.isclose(got, recorded, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(f"the captured predictions give {metric} {got}, the engine's record says {recorded}")
    models = store.read_stage(pid, "models") or {}
    info = {"seconds": seconds, "metric": metric, "higher_is_better": higher, "holdout_score": got,
            "engine_interval": final["evidence"].get("holdout_primary_metric_ci"),
            "model": final["evidence"].get("model"), "recipe": final["evidence"].get("feature_recipe"),
            "config": final["evidence"].get("selected_optimization"), "params": final["evidence"].get("selected_model_params"),
            "screen": (models.get("evidence") or {}).get("ranking"), "cv_selected_mean": final["evidence"].get("cv_selected_metric_mean"),
            "project": str(store.directory(pid).relative_to(work))}
    return _save(work, "dclab", info, prepared, predictions)


class FixedFolds:
    """The engine's training folds as a splitter FLAML accepts (``split_type``): it never reshuffles them."""

    shuffle = False

    def __init__(self, splits: list[tuple[Any, Any]]):
        self.splits = [(np.asarray(a), np.asarray(b)) for a, b in splits]

    def split(self, X: Any = None, y: Any = None, groups: Any = None):
        yield from self.splits

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return len(self.splits)


FLAML_METRIC = {"average_precision": "ap", "roc_auc": "roc_auc", "macro_f1": "macro_f1", "mae": "mae"}


def flaml_columns(prepared: Any) -> list[str]:
    """The columns the engine's tabular recipes start from: no identifier, time key, group, text or forbidden column."""
    from dclab_rnd.expansion import runner as rn

    return rn.FrameRecipe(prepared.bundle, "raw").base_columns(prepared.bundle.X)


def _flaml_frame(frame: Any, codes: tuple[str, ...]) -> Any:
    """Columns the dataset declares as category codes go in as text, so FLAML treats them as categories (the engine
    one-hot encodes them); every other column as it is."""
    import pandas as pd

    frame = frame.copy()
    for column in codes:
        if column in frame:
            frame[column] = frame[column].map(lambda v: "__NAN__" if pd.isna(v) else str(v)).astype(object)
    # 2026-10-25: LightGBM refuses JSON characters in a feature's name (spambase's "char_freq_;"): FLAML failed twice on
    # it for the harness's sake, not its own; positions as names, the same for training and holdout rows
    frame.columns = [f"c{i}" for i in range(frame.shape[1])]
    return frame


def run_flaml(work: Path, budget_s: int, n_jobs: int | None = None) -> dict[str, Any]:
    """FLAML's AutoML on the engine's prepared table and training folds; the holdout rows are given to it once, to predict."""
    from flaml import AutoML
    import flaml

    from dclab_rnd.studio import engine

    store, pid = _open(work)
    started = time.perf_counter()
    prepared = engine.prepare(store, store.get(pid))
    bundle, spec = prepared.bundle, prepared.bundle.spec
    metric, higher, fn = _metric_fn(prepared)
    if metric not in FLAML_METRIC:
        raise ValueError(f"no FLAML metric for {metric}")
    columns = flaml_columns(prepared)
    codes = tuple(spec.categorical_columns)
    X_train = _flaml_frame(bundle.X_train[columns], codes)
    y_train = bundle.y_train
    classification = bundle.task_type != "timeseries_regression"
    n_jobs = n_jobs or max(1, int((os.cpu_count() or 2) * 0.8))  # 80% of the machine (the owner's rule)
    settings = {"task": "classification" if classification else "regression", "metric": FLAML_METRIC[metric],
                "time_budget": int(budget_s), "n_splits": len(bundle.cv_splits), "n_jobs": n_jobs, **FLAML_SETTINGS}
    automl = AutoML()
    log = work / "flaml_search.log"  # every trial FLAML tried, kept whole beside the result
    search_started = time.perf_counter()
    automl.fit(X_train=X_train, y_train=y_train, split_type=FixedFolds(bundle.cv_splits), log_file_name=str(log), **settings)
    search_seconds = time.perf_counter() - search_started
    X_test = _flaml_frame(bundle.X_test[columns], codes)  # the one read of the holdout's rows: after the search, to predict
    if not classification:
        predictions = np.asarray(automl.predict(X_test), dtype=float)
    else:
        raw = np.asarray(automl.predict_proba(X_test), dtype=float)
        classes = [int(c) for c in automl.classes_]
        if bundle.task_type == "multiclass":
            from dclab_rnd.expansion import runner as rn

            predictions = np.zeros((len(X_test), int(rn._n_classes(bundle))))
            for j, c in enumerate(classes):
                predictions[:, c] = raw[:, j]
        else:
            predictions = raw[:, classes.index(1)]
    score = fn(np.asarray(bundle.y_test), predictions)
    trials = sum(1 for line in log.read_text().splitlines() if line.strip().startswith("{")) if log.is_file() else None
    info = {"seconds": time.perf_counter() - started, "search_seconds": search_seconds, "metric": metric, "higher_is_better": higher,
            "holdout_score": score, "flaml_version": flaml.__version__, "settings": settings, "columns": columns,
            "category_code_columns": [c for c in codes if c in columns], "best_estimator": automl.best_estimator,
            "best_config": automl.best_config, "best_cv_loss": automl.best_loss, "trials": trials,
            "search_log": log.name}
    return _save(work, "flaml", info, prepared, predictions)


def run_one(work: Path, engine_name: str, budget_s: int) -> dict[str, Any]:
    if engine_name == "dclab":
        return run_dclab(work)
    if engine_name == "flaml":
        return run_flaml(work, budget_s)
    raise ValueError(engine_name)


# ---------------------------------------------------------------------------------------------- comparing


def paired_bootstrap(y: Any, a: Any, b: Any, fn: Callable[[Any, Any], float], higher: bool, groups: Any = None,
                     draws: int = PAIRED_DRAWS, seed: int = SEED) -> dict[str, Any]:
    """The difference a − b on the same resampled rows (whole groups when given), oriented so positive means a is better."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    sign = 1.0 if higher else -1.0
    rng = np.random.default_rng(seed)
    n = len(y)
    if groups is not None and len(groups):
        groups = np.asarray(groups, dtype=str)
        unique = np.unique(groups)
        positions = {g: np.flatnonzero(groups == g) for g in unique}
        method = f"paired cluster bootstrap over {len(unique)} groups"
    else:
        groups, method = None, "paired iid row bootstrap"
    values = []
    for _ in range(draws):
        if groups is not None:
            index = np.concatenate([positions[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        else:
            index = rng.integers(0, n, size=n)
        try:
            d = sign * (fn(y[index], a[index]) - fn(y[index], b[index]))
        except ValueError:  # a draw with one class cannot score a ranking metric
            continue
        if np.isfinite(d):
            values.append(float(d))
    point = sign * (fn(y, a) - fn(y, b))
    if not values:
        return {"point": float(point), "low": None, "high": None, "draws": 0, "method": method}
    return {"point": float(point), "low": float(np.quantile(values, 0.025)), "high": float(np.quantile(values, 0.975)),
            "draws": len(values), "method": method, "share_of_draws_flaml_better": float(np.mean(np.asarray(values) > 0))}


def compare(work: Path) -> dict[str, Any]:
    """Both engines' saved predictions scored on the same holdout rows: each engine's interval and the paired difference."""
    from dclab_rnd.expansion import runner as rn
    from dclab_rnd.studio import engine

    infos = {e: json.loads((work / f"{e}.json").read_text()) for e in ENGINES}
    arrays = {e: np.load(work / f"{e}.npz", allow_pickle=False) for e in ENGINES}
    for key in ("holdout_rows", "y_test", "groups"):
        if not np.array_equal(arrays["dclab"][key], arrays["flaml"][key]):
            raise ValueError(f"the two engines did not score the same holdout ({key} differ): no comparison")
    for key in ("holdout_rows_sha256", "holdout_labels_sha256", "fold_sizes", "folds_sha256", "model_columns", "train_rows"):
        if infos["dclab"]["split"][key] != infos["flaml"]["split"][key]:
            raise ValueError(f"the two engines did not use the same split ({key} differ): no comparison")
    store, pid = _open(work)
    prepared = engine.prepare(store, store.get(pid))
    metric, higher, fn = _metric_fn(prepared)
    y = arrays["dclab"]["y_test"]
    groups = arrays["dclab"]["groups"] if prepared.bundle.spec.split_strategy in ("group", "time_group") else None
    groups = groups if groups is not None and len(groups) else None
    engines = {}
    for e in ENGINES:
        p = arrays[e]["predictions"]
        ci = rn.bootstrap_ci(y, p, fn, groups=groups)
        engines[e] = {"holdout_score": float(fn(y, p)), "interval_95": {"low": ci["low"], "high": ci["high"], "method": ci["method"],
                                                                         "draws": ci["draws"]},
                      "minutes": round(infos[e]["seconds"] / 60, 2), "peak_memory_mb": infos[e]["peak_memory_mb"],
                      **{k: infos[e].get(k) for k in ("model", "recipe", "config", "params", "best_estimator", "best_config",
                                                      "best_cv_loss", "trials", "search_seconds", "settings", "columns",
                                                      "category_code_columns", "flaml_version", "cv_selected_mean")
                         if infos[e].get(k) is not None}}
    paired = paired_bootstrap(y, arrays["flaml"]["predictions"], arrays["dclab"]["predictions"], fn, higher, groups)
    comparison = ("no comparison" if paired["low"] is None else "flaml ahead" if paired["low"] > 0
                  else "flaml behind" if paired["high"] < 0 else "no clear difference")
    return {"metric": metric, "higher_is_better": higher, "split": infos["dclab"]["split"], "same_split": True,
            "engines": engines, "paired": paired, "comparison": comparison}


# ---------------------------------------------------------------------------------------------- the campaign


def result_path(campaign: Path, dataset: str, datasets: tuple[str, ...] | list[str]) -> Path:
    return campaign / "results" / f"BKO-{list(datasets).index(dataset) + 1:03d}_{dataset}.json"


def store_result(campaign: Path, dataset: str, datasets: tuple[str, ...] | list[str], compared: dict[str, Any]) -> Path:
    """The dataset's result, written once (a file that exists is never replaced: rerun into a new campaign version)."""
    path = result_path(campaign, dataset, datasets)
    eid = path.stem.split("_")[0]
    doc = {"id": eid, "experiment_id": eid, "campaign_id": campaign.name, "kind": "model_bakeoff", "dataset": dataset,
           "at": _now(), "status": "completed", "rule_sha256": rule_sha256(), **compared,
           "limits": ["One holdout: the interval covers resampling of these rows, not another draw of the data.",
                      "Not a production approval."]}
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(doc, indent=2, default=_plain) + "\n")
    return path


def load_results(campaign: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((campaign / "results").glob("BKO-0*.json")) if not p.stem.startswith("BKO-100")]


def render_report(campaign: Path, manifest: dict[str, Any], rows: list[dict[str, Any]], decision: dict[str, Any] | None) -> str:
    def score(e: dict[str, Any]) -> str:
        i = e["interval_95"]
        return f"{e['holdout_score']:.4f} ({i['low']:.4f}–{i['high']:.4f})"

    lines = [f"# {campaign.name}: the model bake-off (package 34.5)", "",
             "Generated by `python -m dclab_rnd.bakeoff report`; do not edit.", "",
             f"Rule `{manifest['rule']['id']}` fixed before the run at {manifest['frozen_at']} (sha256 `{manifest['rule_sha256'][:12]}`): "
             + manifest["rule"]["go"] + ".", "",
             f"FLAML {manifest.get('flaml_version')} with a {manifest['protocol']['flaml_budget_seconds']} s search budget; "
             "the DCLab engine as it is. Same rows, holdout, training folds and metric; 95% intervals.", "",
             "| Dataset | Metric | Holdout rows | DCLab engine | FLAML | FLAML − DCLab (paired) | Comparison | Minutes (DCLab / FLAML) | Peak MB (DCLab / FLAML) |",
             "|---|---|---|---|---|---|---|---|---|"]
    not_judged = set((decision or {}).get("not_judged") or [])
    for r in rows:
        d, f, p = r["engines"]["dclab"], r["engines"]["flaml"], r["paired"]
        diff = f"{p['point']:+.4f} ({p['low']:+.4f} to {p['high']:+.4f})" if p.get("low") is not None else "n/a"
        comparison = f"not judged by the rule ({r['comparison']})" if r["dataset"] in not_judged else r["comparison"]
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['split']['holdout_rows']} | {score(d)} {d.get('model', '')} | {score(f)} "
                     f"{f.get('best_estimator', '')} | {diff} | {comparison} | {d['minutes']} / {f['minutes']} | "
                     f"{d['peak_memory_mb']:.0f} / {f['peak_memory_mb']:.0f} |")
    for dataset, engines in sorted(((decision or {}).get("failed") or {}).items()):  # a failure is a row too, never a silent gap
        who = " and ".join("FLAML" if e == "flaml" else "DCLab" for e in engines)
        lines.append(f"| {dataset} | | | | | | {who} failed after its retry"
                     + (" (counted as FLAML behind by the rule)" if engines == ["flaml"] else "") + " | | |")
    lines += ["", "Differences are oriented so that positive means FLAML scored better (for MAE, lower error).", ""]
    if decision:
        lines += [f"**Decision ({decision['outcome']}):** package 34.8 {decision['package_34_8']}. FLAML ahead on "
                  f"{len(decision['flaml_ahead'])} of {decision['datasets_judged']} datasets judged (needed {decision['needed_ahead']}), "
                  f"behind on {len(decision['flaml_behind'])}.", ""]
    else:
        missing = [d for d in manifest["protocol"]["datasets"] if d not in {r["dataset"] for r in rows}]
        lines += [f"No decision yet: {len(missing)} dataset(s) still to run ({', '.join(missing)}).", ""]
    lines += ["Limits: " + " ".join(manifest.get("limits") or []), ""]
    return "\n".join(lines)


def failures(work_root: Path, datasets: list[str]) -> dict[str, list[str]]:
    """The datasets where an engine (or the dataset itself) failed after its retry: dataset -> the engines that failed."""
    out: dict[str, list[str]] = {}
    for dataset in datasets:
        work = work_root / dataset
        engines = [e for e in ENGINES if (work / f"{e}.failed.json").is_file() and not (work / f"{e}.json").is_file()]
        if (work / "dataset.failed.json").is_file() and not engines:
            engines = list(json.loads((work / "dataset.failed.json").read_text()).get("engines") or ENGINES)
        if engines:
            out[dataset] = engines
    return out


def report(campaign: Path = CAMPAIGN, final: bool = False, work_root: Path = WORK) -> dict[str, Any] | None:
    """Render the report; write the decision once every dataset is compared, or with ``final`` once every dataset not
    compared failed after its retry (never to stop early on a good score)."""
    manifest = json.loads((campaign / "manifest.json").read_text())
    rows = load_results(campaign)
    decision_path = campaign / "results" / "BKO-100_decision.json"
    decision = json.loads(decision_path.read_text()) if decision_path.is_file() else None
    missing = sorted(set(manifest["protocol"]["datasets"]) - {r["dataset"] for r in rows})
    failed = failures(work_root, missing)
    if final and set(failed) != set(missing):
        raise SystemExit(f"no decision yet: {', '.join(sorted(set(missing) - set(failed)))} neither compared nor failed after a retry")
    if decision is None and (not missing or final):
        decision = {"id": "BKO-100", "experiment_id": "BKO-100", "campaign_id": campaign.name, "kind": "model_bakeoff_decision",
                    "at": _now(), "status": "completed", **decide(rows, manifest["rule"], failed), "failed": failed}
        if decision["rule_sha256"] != manifest["rule_sha256"]:
            raise SystemExit("the rule in the manifest does not match its hash")
        with decision_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(decision, indent=2) + "\n")
    (campaign / "CAMPAIGN_REPORT.md").write_text(render_report(campaign, manifest, rows, decision))
    return decision


# ---------------------------------------------------------------------------------------------- running it safely


BUSY = (("tests/run_parallel.py", "the test suite (a gate)"), ("make check-", "a gate"), ("dclab_rnd.store", "a store run"),
        ("dclab_rnd.archive", "an archive run"), ("ollama runner", "a loaded local model"), ("llama-server", "a loaded local model"),
        ("mlx_lm", "a loaded local model"))


def busy(ps_output: str | None = None) -> str | None:
    """Why the machine is not free for a run now (a gate, a store run, a loaded local model, low disk, red memory), or None."""
    if ps_output is None:
        try:
            ps_output = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10).stdout
        except Exception:  # noqa: BLE001 — no ps: nothing to say
            ps_output = ""
    me = str(os.getpid())
    for line in ps_output.splitlines():
        pid, _, command = line.strip().partition(" ")
        if pid == me or "dclab_rnd.bakeoff" in command:
            continue
        for needle, what in BUSY:
            if needle in command:
                return f"{what} is running (pid {pid})"
    if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
        return f"less than {MIN_FREE_BYTES // 1024**3} GB of free disk"
    try:
        level = int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True,
                                   timeout=2).stdout.strip() or 1)
    except Exception:  # noqa: BLE001 — not macOS
        level = 1
    if level >= 4:
        return "macOS memory pressure is red"
    return None


def wait_until_free(poll_s: int = 60, max_hours: float = 12.0, check: Callable[[], str | None] = busy) -> None:
    deadline = time.time() + max_hours * 3600
    said = None
    while True:
        why = check()
        if why is None:
            return
        if why != said:
            print(f"[bakeoff] waiting: {why}", flush=True)
            said = why
        if time.time() > deadline:
            raise SystemExit(f"[bakeoff] stopped: still busy after {max_hours} h ({why})")
        time.sleep(poll_s)


def _child(work: Path, engine_name: str, budget_s: int) -> dict[str, Any]:
    env = {**os.environ, "DCLAB_NO_LIVE_MODELS": "1", "DCLAB_DECISION_SHADOW": "0", "DCLAB_LLM_PROVIDER": "", "DCLAB_DATABASE_URL": "",
           "DCLAB_AGENT_HOME": str(work / "home"), "PYTHONUNBUFFERED": "1"}
    timeout = ENGINE_TIMEOUT_S if engine_name == "dclab" else budget_s + FLAML_EXTRA_S
    log = work / f"{engine_name}.log"
    started = time.perf_counter()
    with log.open("w") as handle:
        try:
            done = subprocess.run([sys.executable, "-m", "dclab_rnd.bakeoff", "one", "--work", str(work), "--engine", engine_name,
                                   "--budget-s", str(budget_s)], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                  timeout=timeout)
            code = done.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    if code == 0 and (work / f"{engine_name}.json").is_file():
        return json.loads((work / f"{engine_name}.json").read_text())
    failure = {"engine": engine_name, "status": "failed", "exit": code, "seconds": time.perf_counter() - started,
               "at": _now(), "log_tail": log.read_text()[-3000:] if log.is_file() else ""}
    (work / f"{engine_name}.failed.json").write_text(json.dumps(failure, indent=2) + "\n")
    return failure


def run(budget_s: int, only: list[str] | None = None, datasets: tuple[str, ...] | list[str] = DATASETS, campaign: Path = CAMPAIGN,
        work_root: Path = WORK, wait: bool = True, prepare: Callable[[Path, str], dict[str, Any]] = prepare_sample,
        runner: Callable[[Path, str, int], dict[str, Any]] | None = None, retries: int = 1) -> dict[str, Any]:
    """Every dataset in order (or those in ``only``: the protocol still names them all); an engine that finished is never
    rerun, a failed one is tried again ``retries`` times."""
    manifest = freeze(campaign, budget_s, datasets)
    runner = runner or _child
    summary: dict[str, Any] = {}
    for dataset in [d for d in datasets if only is None or d in only]:
        if result_path(campaign, dataset, manifest["protocol"]["datasets"]).is_file():
            summary[dataset] = "stored before"
            continue
        work = work_root / dataset
        work.mkdir(parents=True, exist_ok=True)
        try:
            summary[dataset] = _run_dataset(work, dataset, budget_s, campaign, manifest, wait, prepare, runner, retries)
        except Exception as exc:  # noqa: BLE001 — one dataset's failure is recorded; the campaign goes on
            (work / "dataset.failed.json").write_text(json.dumps({"dataset": dataset, "at": _now(), "error": f"{type(exc).__name__}: {exc}"[:3000],
                                                                  "engines": [e for e in ENGINES if not (work / f"{e}.json").is_file()]},
                                                                 indent=2) + "\n")
            print(f"[bakeoff] {dataset}: failed ({type(exc).__name__}: {str(exc)[:300]}); going on", flush=True)
            summary[dataset] = f"failed: {type(exc).__name__}"
        report(campaign, work_root=work_root)
    return summary


def _fresh(work: Path) -> None:
    """A new project for a retry: a killed stage leaves its project marked running, and a final stage that ran leaves the
    holdout used, so the graph would refuse the rerun. FLAML's outputs go too (it must read the same project)."""
    for name in ("workspace", "home"):
        shutil.rmtree(work / name, ignore_errors=True)
    for name in ("project.json", "flaml.json", "flaml.npz", "flaml_search.log"):
        (work / name).unlink(missing_ok=True)


def _run_dataset(work: Path, dataset: str, budget_s: int, campaign: Path, manifest: dict[str, Any], wait: bool,
                 prepare: Callable[[Path, str], dict[str, Any]], runner: Callable[[Path, str, int], dict[str, Any]], retries: int) -> str:
    (work / "dataset.failed.json").unlink(missing_ok=True)
    if wait:
        wait_until_free()
    if not (work / "dclab.json").is_file() and (work / "project.json").is_file():
        _fresh(work)  # the engine never finished on this project: it starts again on a new one
    prepare(work, dataset)
    states = {}
    for engine_name in ENGINES:
        out: dict[str, Any] = {"status": "ok"}
        for attempt in range(0 if (work / f"{engine_name}.json").is_file() else 1 + retries):
            if engine_name == "dclab" and attempt > 0:
                _fresh(work)
                prepare(work, dataset)
            if wait:
                wait_until_free()
            print(f"[bakeoff] {dataset}: {engine_name} (attempt {attempt + 1}) {_now()}", flush=True)
            out = runner(work, engine_name, budget_s)
            if out.get("status") == "ok":
                print(f"[bakeoff] {dataset}: {engine_name} {out['metric']} {out['holdout_score']:.4f} in "
                      f"{out['seconds'] / 60:.1f} min, peak {out['peak_memory_mb']:.0f} MB", flush=True)
                break
            print(f"[bakeoff] {dataset}: {engine_name} failed ({out.get('exit')}); see {work / (engine_name + '.log')}", flush=True)
        states[engine_name] = "ok" if (work / f"{engine_name}.json").is_file() else out.get("status", "failed")
        if states[engine_name] != "ok":
            break  # FLAML reads the engine's project: without the engine's run there is nothing to compare
    if not states or any(s != "ok" for s in states.values()) or len(states) < len(ENGINES):
        return "failed: " + ", ".join(e for e in ENGINES if states.get(e) != "ok")
    for e in ENGINES:
        (work / f"{e}.failed.json").unlink(missing_ok=True)
    path = store_result(campaign, dataset, manifest["protocol"]["datasets"], compare(work))
    stored = json.loads(path.read_text())
    print(f"[bakeoff] {dataset}: {stored['comparison']}, FLAML − DCLab {stored['paired']['point']:+.4f} "
          f"({stored['paired']['low']} to {stored['paired']['high']}) -> {path.name}", flush=True)
    return stored["comparison"]


def plan(datasets: list[str], budget_s: int, campaign: Path = CAMPAIGN) -> str:
    lines = [f"Bake-off {campaign.name}: DCLab's engine against FLAML, {len(datasets)} datasets, FLAML budget {budget_s} s each.",
             f"Rule (fixed before the run): {RULE['go']}. Arm: {RULE['arm']}.", f"Left out: " + "; ".join(f"{k} ({v})" for k, v in LEFT_OUT.items()),
             f"Work folder: {WORK} (kept, not committed). Results: {campaign / 'results'}.", ""]
    for dataset in datasets:
        path = result_path(campaign, dataset, datasets)
        state = "stored" if path.is_file() else ", ".join(f"{e} {'done' if (WORK / dataset / f'{e}.json').is_file() else 'to run'}"
                                                        for e in ENGINES)
        lines.append(f"  {path.stem.split('_')[0]} {dataset}: {state}")
    lines += ["", f"Machine now: {busy() or 'free'}.", "Run: python -m dclab_rnd.bakeoff run --yes (in the background; resumable)."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- small helpers


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _peak_mb() -> float:
    """The process's peak resident memory (and its waited children's, when larger), in MB (macOS reports bytes, Linux KB)."""
    unit = 1 if sys.platform == "darwin" else 1024
    peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return round(peak * unit / 1024**2, 1)


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.bakeoff", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command")
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--datasets", default=",".join(DATASETS))
        p.add_argument("--budget-s", type=int, default=DEFAULT_BUDGET_S)
        if name == "run":
            p.add_argument("--yes", action="store_true", help="run it (without, print the plan)")
            p.add_argument("--no-wait", action="store_true", help="do not wait for a gate, a store run or a loaded model to end")
    one = sub.add_parser("one", help="one engine on one prepared dataset (the run's child process)")
    one.add_argument("--work", required=True)
    one.add_argument("--engine", choices=ENGINES, required=True)
    one.add_argument("--budget-s", type=int, default=DEFAULT_BUDGET_S)
    rep = sub.add_parser("report")
    rep.add_argument("--final", action="store_true", help="write the decision on the datasets that finished")
    args = parser.parse_args(argv)
    if args.command in (None, "plan") or (args.command == "run" and not args.yes):
        print(plan(list(DATASETS), getattr(args, "budget_s", DEFAULT_BUDGET_S)))
        return 0
    if args.command == "one":
        run_one(Path(args.work), args.engine, args.budget_s)
        return 0
    if args.command == "report":
        print(json.dumps(report(CAMPAIGN, final=args.final), indent=2))
        return 0
    only = args.datasets.split(",")
    unknown = [d for d in only if d not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s): {', '.join(unknown)}")
    print(json.dumps(run(args.budget_s, only=only, wait=not args.no_wait), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
