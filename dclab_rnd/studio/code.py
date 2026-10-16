"""Code for each step (package 20.9): a project's pipeline as code a data scientist can read and run.

``export`` writes a folder: ``pipeline.py`` (the steps in order, one block per pipeline step), ``pipeline.ipynb`` (the
same blocks as cells), ``expected.json`` (the holdout number the project recorded), ``data/`` (the person's own tables,
copied as attached, or for a tabular project the rows as the engine prepared them), ``vendor/`` (dclab_core, and the
engine's runner for a tabular project), ``requirements.txt`` and a README. ``python pipeline.py`` computes the holdout
number again and exits 1 when it differs from the record.

A kind with its own machine (forecasting, time to event, anomaly detection, entity tables, streams, long texts) is
written as calls to dclab_core with the recorded choices: every step from the raw tables, the choice made again on
validation and checked against the record. A tabular project (binary, multiclass, regression, rare events, text with a
table, places) is written as the engine's own runner fitting the recorded recipe and model on the rows the engine
prepared (kept losslessly, with which rows trained, which are the holdout and which neither); how the rows were prepared
(the solution applied, labels joined, rows sampled) is said, not re-run. A person's choice over the rule's is scored,
and the rule's own pick is checked beside it.

The folder holds the person's data: the export is a write action and is written to the audit trail. The main product's
``code_snapshots`` are not filled from it yet.
"""

from __future__ import annotations

import json
import shutil
import textwrap

import numpy as np
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[2]
VENDOR = {"dclab_rnd/provenance.py": "dclab_rnd/provenance.py", "dclab_rnd/expansion/__init__.py": "dclab_rnd/expansion/__init__.py",
          "dclab_rnd/expansion/datasets.py": "dclab_rnd/expansion/datasets.py", "dclab_rnd/expansion/runner.py": "dclab_rnd/expansion/runner.py",
          "dclab_rnd/expansion/detectors.py": "dclab_rnd/expansion/detectors.py"}
BASE = ("numpy", "pandas", "scikit-learn", "scipy", "pydantic", "threadpoolctl")


class CodeRefused(ValueError):
    """The project's code cannot be written yet; the reason says why."""


def _positive(project: dict[str, Any], solution: dict[str, Any]) -> Any:
    from ..kinds.binary import as_table_label, profile_booleans  # the positive label as the engine reads it (20.10)

    return as_table_label(solution.get("positive_label"), profile_booleans((project.get("data") or {}).get("profile"), solution.get("target")))


def _c(value: Any) -> str:
    """A recorded value as Python source."""
    return repr(json.loads(json.dumps(value, default=str)))


HEADER = '''\
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE / "vendor"))  # dclab_core (and the engine's runner) as exported with this project
EXPECTED = json.loads((HERE / "expected.json").read_text())


def read_table(name):
    """A table as DCLab read it, by its file's suffix."""
    path = HERE / "data" / name
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\\t", low_memory=False)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path, low_memory=False)'''

FOOTER = '''\
same = value is not None and abs(float(value) - float(EXPECTED["value"])) <= 1e-6 * max(1.0, abs(float(EXPECTED["value"])))
print(json.dumps({"metric": EXPECTED["metric"], "value": value, "expected": EXPECTED["value"], "same": same}))
if not same:
    sys.exit(1)'''


def _ev(records: dict[str, Any], stage: str) -> dict[str, Any]:
    return (records.get(stage) or {}).get("evidence") or {}


def _chosen(records: dict[str, Any], stage: str) -> Any:
    decision = (records.get(stage) or {}).get("decision") or {}
    return decision.get("chosen") if decision.get("chosen") is not None else decision.get("selected")


def _rule(records: dict[str, Any], stage: str) -> Any:
    """The rule's own pick (``selected``): a person's approval may have chosen another option (``chosen``)."""
    decision = (records.get(stage) or {}).get("decision") or {}
    return decision.get("selected") if decision.get("selected") is not None else decision.get("chosen")


def _check(records: dict[str, Any], picked: str) -> str:
    """The line that checks the rule's pick made again, and says when a person chose another option."""
    rule, chosen = _rule(records, "models"), _chosen(records, "models")
    line = f'assert {picked} == {_c(rule)}, "the rule picks again what it picked when the project ran"'
    return line + (f"\n# a person chose {chosen!r} over the rule's {rule!r} when approving the models stage: it is the one scored below"
                   if rule != chosen else "")


# ------------------------------------------------------------------------------------------- the kinds with their own machine

def _entity_tables(project: dict[str, Any], records: dict[str, Any], f: dict[str, Any]) -> list[tuple[str, str]]:
    kept_out, chosen = _ev(records, "leakage").get("kept_out") or [], _chosen(records, "models")
    events = [{"name": t["name"], "time_column": t["time_column"], "sum_columns": t.get("sum_columns", [])} for t in f["event_tables"]]
    common = f'key={_c(f["key_column"])}, target_table={_c(f["target_table"])}, horizon_days={f["horizon_days"]}, absent={f["absent"]}'
    return [
        ("data", f'''from dclab_core import entities

entity_table = read_table({_c(project["data"]["filename"])})
EVENT_TABLES = {_c(events)}
TABLE_FILES = {_c({n: t["filename"] for n, t in (project.get("tables") or {}).items()})}
tables = {{t["name"]: read_table(TABLE_FILES[t["name"]]) for t in EVENT_TABLES}}
specs = {{t["name"]: {{"frame": tables[t["name"]], "time": t["time_column"], "sums": t["sum_columns"]}} for t in EVENT_TABLES}}
# the holdout: the latest cutoff whose horizon the log reaches, from dates alone
parts = entities.split(specs, cutoffs={_c(f["cutoffs"])}, target_table={_c(f["target_table"])}, horizon_days={f["horizon_days"]},
                       data_end={_c(f.get("data_end"))})
assert parts["holdout"] == {_c(_ev(records, "data").get("split", {}).get("holdout"))}, parts["holdout"]
view = entities.training_view(entity_table, specs, parts, {common})  # every training row from the log before the holdout'''),
        ("leakage", f'''KEPT_OUT = {_c(kept_out)}  # the entity columns written after a cutoff, and what the solution forbids'''),
        ("features", f'''FEATURES = entities.feature_columns(view["table"], {_c(f["key_column"])}, KEPT_OUT)  # windows per event table, entity columns
print(len(FEATURES), "features")'''),
        ("models", f'''picked = entities.choose(view, parts, key={_c(f["key_column"])}, exclude=KEPT_OUT)  # on the latest training cutoff
{_check(records, 'picked["chosen"]')}'''),
        ("final", f'''scored = entities.score(entity_table, specs, view, parts, {_c(chosen)}, {common}, exclude=KEPT_OUT)  # the holdout, read once
value = scored["holdout"][{_c(chosen)}]["roc_auc"]'''),
    ]


def _streams(project: dict[str, Any], records: dict[str, Any], f: dict[str, Any]) -> list[tuple[str, str]]:
    held, chosen = _ev(records, "leakage").get("sensors_kept_out") or [], _chosen(records, "models")
    cols = f'unit={_c(f["unit_column"])}, time={_c(f["time_column"])}, sensors={_c(f["sensor_columns"])}'
    return [
        ("data", f'''from dclab_core import streams
from dclab_rnd.expansion import detectors

readings = read_table({_c(project["data"]["filename"])})
incidents = read_table({_c(((project.get("tables") or {}).get(f["incidents_table"]) or {}).get("filename"))})
# the holdout's start, on the time axis of the windows' dates (never an incident)
parts = streams.split(readings, {cols}, window_hours={f["window_hours"]}, horizon_hours={f["horizon_hours"]},
                      label_delay_hours={f["label_delay_hours"]})
assert parts["cut"] == {_c(_ev(records, "data").get("split", {}).get("cut"))}, parts["cut"]
view = streams.training_view(readings, incidents, parts, {cols}, incident_time={_c(f["incident_time_column"])})
_, known = streams.known_by(readings, incidents, parts, time={_c(f["time_column"])}, incident_time={_c(f["incident_time_column"])})'''),
        ("leakage", f'''view = streams.without(view, {_c(held)})  # sensors written knowing the failure, kept out until confirmed'''),
        ("features", '''print(len(view["features"]), "features per window: mean, spread, last value, slope and the change against the machine's own past")'''),
        ("models", f'''DETECTORS = {{"percentile_rules": detectors.PercentileRules, "ecod": detectors.ECOD}}
picked = streams.choose(view, known, parts, unit={_c(f["unit_column"])}, incident_time={_c(f["incident_time_column"])},
                        budget_per_unit_day={f["budget_per_unit_day"]}, detectors=DETECTORS)
{_check(records, 'picked["chosen"]')}'''),
        ("final", f'''scored = streams.score(readings, incidents, view, parts, {_c(chosen)}, {cols}, incident_time={_c(f["incident_time_column"])},
                       budget_per_unit_day={f["budget_per_unit_day"]}, detectors=DETECTORS)  # the holdout's windows, read once
value = scored["holdout"][{_c(chosen)}]["incident_recall"]'''),
    ]


def _anomaly(project: dict[str, Any], records: dict[str, Any], f: dict[str, Any]) -> list[tuple[str, str]]:
    features, chosen = _ev(records, "leakage").get("features") or [], _chosen(records, "models")
    args = f'{_c(features)}, {_c(f["entity_column"])}, {_c(f["time_column"])}, {f["budget"]}'
    return [
        ("data", f'''from dclab_core import anomaly

table = read_table({_c(project["data"]["filename"])})
parts = anomaly.split(table, {_c(f["time_column"])})  # the rows to review: the latest whole days'''),
        ("leakage", f'''FEATURES = {_c(features)}  # the numeric columns read; identifiers and what the moment rules out are kept out'''),
        ("features", f'''built = anomaly.Features(FEATURES, {_c(f["entity_column"])}, {_c(f["time_column"])}).fit(parts["train"])  # fitted on the training rows
print(len(list(built.scale)), "features")'''),
        ("models", f'''picked = anomaly.choose(parts, {args})  # on anomalies planted in the validation period
{_check(records, 'picked["chosen"]')}'''),
        ("final", f'''out = anomaly.review(parts, {_c(chosen)}, {args}, {_c(f["key_column"])})  # the rows to review, ranked once
planted = out["planted"]
value = round(sum(v["chosen"] for v in planted.values()) / max(len(planted), 1), 4)'''),
    ]


def _forecasting(project: dict[str, Any], records: dict[str, Any], f: dict[str, Any]) -> list[tuple[str, str]]:
    from ..kinds.forecasting import ForecastMachine

    solution = project["solution"]
    forbidden = {x["column"] for x in solution.get("forbidden") or []}
    known = [c for c in _ev(records, "leakage").get("kept") or [] if c not in forbidden]
    chosen, time, series, target = _chosen(records, "models"), solution["time_column"], f["series_column"], solution["target"]
    common = (f'series={_c(series)}, time={_c(time)}, target={_c(target)}, horizon={f["horizon"]}, known_ahead=KNOWN_AHEAD, '
              f'season={_c(f["season"])}, gap={f["gap"]}')
    n = ForecastMachine.VALIDATION_ORIGINS
    return [
        ("data", f'''from dclab_core import forecast

table = read_table({_c(project["data"]["filename"])})
table = table.assign(**{{{_c(time)}: pd.to_datetime(table[{_c(time)}], errors="coerce")}}).dropna(subset=[{_c(time)}, {_c(target)}])
origin = forecast.holdout_origin(table, time={_c(time)}, horizon={f["horizon"]}, gap={f["gap"]})  # the last forecast
assert str(origin.date()) == {_c(_ev(records, "data").get("holdout_origin"))}, origin
train = table[table[{_c(time)}] <= origin]'''),
        ("leakage", f'''KNOWN_AHEAD = {_c(known)}  # the columns known for the forecast dates; every other column is kept out'''),
        ("models", f'''validation = forecast.backtest(train, {common}, origins={n})  # rolling origins before the holdout
ranked = sorted(((m, sc["mase"], sc["covers_every_series"]) for m, sc in validation["scores"].items()), key=lambda r: (r[1] is None, r[1]))
rule = next((m for m, mase, every in ranked if m != "seasonal_naive" and mase is not None and every), "seasonal_naive")
{_check(records, "rule")}'''),
        ("final", f'''result = forecast.backtest(table, {common}, origins={n + 1})
value = (forecast.origin_scores(result, str(origin.date())).get({_c(chosen)}) or {{}}).get("mase")  # the last origin: the holdout'''),
    ]


def _time_to_event(project: dict[str, Any], records: dict[str, Any], f: dict[str, Any]) -> list[tuple[str, str]]:
    features, chosen = _ev(records, "leakage").get("features") or [], _chosen(records, "models")
    cols = f'duration={_c(f["duration_column"])}, event={_c(f["event_column"])}'
    return [
        ("data", f'''from dclab_core import survival

table = read_table({_c(project["data"]["filename"])})
parts = survival.split(table, {cols}, start={_c(f["start_column"])})  # the latest starts held out; training censored at their first day'''),
        ("leakage", f'''FEATURES = {_c(features)}  # read at the start; what is written later is kept out'''),
        ("models", f'''picked = survival.choose(parts, {cols}, features=FEATURES, horizons={_c(f["horizons"])})  # a validation cut
{_check(records, 'picked["chosen"]')}'''),
        ("final", f'''scored = survival.score(parts, {_c(chosen)}, {cols}, features=FEATURES, horizons={_c(f["horizons"])}, group={_c(f["group_column"])})
value = (scored["holdout"].get({_c(chosen)}) or {{}}).get("integrated_brier")'''),
    ]


def _embedded(project: dict[str, Any], records: dict[str, Any], f: dict[str, Any]) -> list[tuple[str, str]]:
    solution = project["solution"]
    text, source, target, chosen = f["text_column"], f["source_column"], solution["target"], _chosen(records, "models")
    return [
        ("data", f'''from dclab_core import embeddings

table = read_table({_c(project["data"]["filename"])})
readable = table[{_c(text)}].notna() & table[{_c(source)}].notna()
parts = embeddings.split(table.loc[readable, {_c(source)}], seed=0)  # whole sources, from their names alone
assert parts["holdout_fingerprint"] == {_c(_ev(records, "data").get("split", {}).get("holdout_fingerprint"))}
rows = table[readable & table[{_c(target)}].notna()].reset_index(drop=True)
test = embeddings.holdout_mask(rows[{_c(source)}], parts)
seen = sorted({{str(v) for v in rows.loc[~test, {_c(target)}]}})  # the classes as the training rows have them
positive = {_c(_positive(project, solution))} or seen[-1]
y = (rows[{_c(target)}].astype(str) == str(positive)).astype(int).to_numpy()'''),
        ("features", '''X, _ = embeddings.encode(rows[''' + _c(text) + '''].astype(str).tolist(), embeddings.HashingText())  # a local baseline, not pretrained'''),
        ("models", f'''picked = embeddings.choose(X[~test], y[~test], rows[{_c(source)}][~test])  # folds grouped by source
{_check(records, 'picked["chosen"]')}'''),
        ("final", f'''scored = embeddings.score(X, y, rows[{_c(source)}], parts, {_c(chosen)})  # the held-out sources, read once
value = scored["holdout"]'''),
    ]


MACHINES: dict[str, Callable[[dict, dict, dict], list[tuple[str, str]]]] = {
    "entity_tables": _entity_tables, "streams": _streams, "anomaly": _anomaly, "forecasting": _forecasting,
    "time_to_event": _time_to_event, "embedded": _embedded}


# ------------------------------------------------------------------------------------------------------ tabular projects

def _tabular(prepared: Any, records: dict[str, Any], profile: str) -> list[tuple[str, str]]:
    final = _ev(records, "final")
    recipe, family, params = final.get("feature_recipe"), final.get("model"), final.get("selected_model_params") or None
    bundle = prepared.bundle
    fast = profile == "fast"
    return [
        ("data", f'''from dclab_rnd.expansion import datasets as ds
from dclab_rnd.expansion import runner as rn

# the rows as the engine prepared them: the solution applied, the labels joined, the rows sampled ({bundle.sampling.get("rule") or "every row"})
X = pd.read_pickle(HERE / "data" / "prepared.pkl")  # every value as the engine held it (written by the pinned pandas)
y, part = X.pop("__target__"), X.pop("__part__").to_numpy()
spec = ds.DatasetSpec(**json.loads((HERE / "spec.json").read_text()))
# train, holdout, or neither (a later row of a group already seen in training: in neither, so nothing crosses the line)
bundle = ds.TaskBundle(spec, X, y, np.flatnonzero(part == "train"), np.flatnonzero(part == "holdout"), [], "", {_c(bundle.holdout_description)},
                       class_labels={_c(bundle.class_labels)})'''),
        ("leakage", '''print("never read:", sorted(spec.blocked_features), "| identifiers:", sorted(spec.identifier_columns))  # the recipe leaves them out'''),
        ("features", f'''rn.RECIPES[spec.key] = json.loads((HERE / "recipes.json").read_text())
RECIPE = {_c(recipe)}  # fitted on the training rows only, inside the model's fit'''),
        ("models", f'''FAMILY, PARAMS = {_c(family)}, {_c(params)}  # chosen on the training folds'''
         + (f"  # a person chose it over the rule's {_rule(records, 'models')!r}" if _rule(records, "models") != _chosen(records, "models") else "")),
        ("final", ("rn._PROFILE[\"fast\"] = True  # the project ran with the engine's small model sizes\n" if fast else "")
         + f'''fitted = rn.fit_and_predict(bundle, RECIPE, FAMILY, PARAMS, bundle.X_train, bundle.y_train, bundle.X_test)
metrics = rn.compute_metrics(bundle.task_type, bundle.y_test, fitted["predictions"], n_classes=rn._n_classes(bundle),
                             alert_budget=spec.settings.get("alert_budget"))
value = metrics[EXPECTED["metric"]]'''),
    ]


TITLES = {"data": "The data and the locked holdout", "leakage": "What is kept out", "features": "The features",
          "models": "The choice, made on validation", "final": "The holdout, read once"}


def cells(store: Any, project_id: str) -> dict[str, Any]:
    """The project's code: ``blocks`` (step, title, source), what the folder needs and the number it must reproduce."""
    from dclab_rnd import kinds

    from . import engine
    from .solution import Solution

    project = store.get(project_id)
    records = store.records(project_id)
    final = records.get("final") or {}
    if (final.get("evidence") or {}).get("holdout_metrics") is None:
        raise CodeRefused("Run the final stage first: the code reproduces its holdout number")
    solution = Solution(**project["solution"])
    kind = solution.problem_kind(engine.pack_of(project))
    metric = final.get("primary_metric")
    value = (final["evidence"].get("holdout_metrics") or {}).get(metric)
    if value is None:
        raise CodeRefused(f"The final stage recorded no holdout {metric}: nothing to reproduce")
    run = final.get("run") or {}
    profile = ((run.get("inputs") or {}).get("settings") or {}).get("engine_profile") or "full"
    from ..ledger import use

    if ((run.get("inputs") or {}).get("refs") or {}).get("decision_routing") != use.fingerprint():  # 22.9
        raise CodeRefused("The decision models routed for this workspace changed since the stages ran: run them again, and the code "
                          "reproduces what they decide")
    prepared = None
    if kind.own_machine() is not None:
        maker = MACHINES.get(kind.key)
        if maker is None:
            raise CodeRefused(f"The {kind.title.lower()} kind's code is not written yet")
        blocks = maker(project, records, kind.fields(solution.kind_fields))
        vendor = ["dclab_core"] + (["detectors"] if kind.key == "streams" else [])
    else:
        prepared = engine.prepare(store, project)
        blocks = _tabular(prepared, records, profile)
        vendor = ["dclab_core", "runner"]
    family = str((final.get("evidence") or {}).get("model") or "")
    extra = [p for p in ("lightgbm", "xgboost") if p in family]
    if kind.own_machine() is not None:  # the readers the tables' files need
        names = [project["data"]["filename"], *(t["filename"] for t in (project.get("tables") or {}).values())]
        suffixes = {Path(n).suffix.lower() for n in names}
        extra += ["pyarrow"] if ".parquet" in suffixes else []
        extra += ["openpyxl"] if suffixes & {".xlsx", ".xls"} else []
    return {"kind": kind.key, "metric": metric, "value": value, "profile": profile, "blocks": blocks, "vendor": vendor,
            "requirements": list(BASE) + extra, "prepared": prepared}


def _module(blocks: list[tuple[str, str]], project: dict[str, Any]) -> str:
    parts = [f'"""{project.get("name") or "Project"}: its pipeline as code, exported by DCLab (20.9). `python pipeline.py` computes the '
             'holdout number again and exits 1 when it differs from the record."""', HEADER]
    for step, source in blocks:
        parts.append(f"# ---- {step}: {TITLES.get(step, step)}\n{source}")
    parts.append(FOOTER)
    return "\n\n".join(parts) + "\n"


def _notebook(blocks: list[tuple[str, str]], project: dict[str, Any]) -> dict[str, Any]:
    def cell(kind: str, source: str) -> dict[str, Any]:
        out = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        return {**out, "execution_count": None, "outputs": []} if kind == "code" else out

    out = [cell("markdown", f"# {project.get('name') or 'Project'}\n\nIts pipeline as code, one cell per step; the last cell compares the "
                            "holdout number with the record."), cell("code", HEADER.replace('Path(__file__).resolve().parent if "__file__" in globals() else ', ""))]
    for step, source in blocks:
        out += [cell("markdown", f"## {TITLES.get(step, step)}"), cell("code", source)]
    out.append(cell("code", FOOTER.replace("    sys.exit(1)", "    raise SystemExit('the holdout number differs from the record')")))
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": out}


PROGRAMME = '''\
from dclab_core import leaves, programme
from dclab_core.problem_map import ProblemMap
from dclab_core.solution_map import SolutionMap

MAIN = {main}
TABLES = {tables}
CASES = {cases}
KEY = {key}
EXCLUDE = {exclude}
BUDGET = {budget}
SEED = {seed}
BOOTSTRAPS = {bootstraps}

m = ProblemMap.read((HERE / "problem_map.json").read_text())
sm = SolutionMap.read((HERE / "solution_map.json").read_text())
prior = programme.Programme.read((HERE / "programme.json").read_text())  # the recorded one: the drops a person made are replayed with it

# 1. the tables as attached, and the confirmed cases joined on their key
frame = read_table(MAIN)
tables = {{name: read_table(file) for name, file in TABLES.items()}}
cases = read_table(CASES) if CASES else None
labels = leaves.mark_cases(frame, cases, KEY) if cases is not None else None
column = leaves.confirmation_column(cases, KEY) if cases is not None else None   # when each case was confirmed: a model never learns a case before then
known_at = leaves.case_dates(frame, cases, KEY, column) if column else None

# 2. the one split of the solution map: the holdout is the latest period, a validation period before it
split = sm.holdout.split
parts = leaves.partition(frame, moment=split.moment, keys=split.keys, share=split.share, seed=split.seed)
ctx = leaves.Context(frame=frame, parts=parts, time=split.moment, entities=list(split.keys), labels=labels, known_at=known_at, tables=tables, exclude=EXCLUDE,
                     budget=BUDGET, seed=SEED)

# 3. the leaves, the composition chosen on validation, the holdout read once
start = programme.replan(sm, m, prior, budget=BUDGET, replaying=True)
run = programme.resume(programme.Execution(start), sm, m, ctx, labels, bootstraps=BOOTSTRAPS)
p = run.programme
print(programme.markdown(p))

value = p.holdout.composed.value if p.holdout is not None else None
same = (value is not None and abs(value - EXPECTED["value"]) <= 1e-6 * max(1.0, abs(EXPECTED["value"]))
        and p.holdout.chosen == EXPECTED["chosen"] and p.holdout.best_leaf == EXPECTED["best_leaf"])
print(json.dumps({{"metric": EXPECTED["metric"], "value": value, "expected": EXPECTED["value"], "chosen": EXPECTED["chosen"], "same": same,
                  "fingerprint": p.fingerprint, "recorded_fingerprint": EXPECTED["fingerprint"]}}))
if not same:
    sys.exit(1)'''


def export_programme(store: Any, project: dict[str, Any], out: Path, actor: str = "human") -> dict[str, Any]:
    """The code of a programme (24.5): the maps and the recorded programme as JSON, the tables as attached, and a script that runs ``dclab_core.programme`` on them with the
    recorded split, budget and seed, and exits 1 unless it chooses the same composition and scores the same holdout number. It needs nothing of the studio."""
    from importlib import metadata

    from . import data as sd
    from . import programme as studio_programme
    from ..storage.files import FileError, files_for

    p = studio_programme.stored(project)
    if p is None or p.holdout is None:
        raise CodeRefused("the programme has not read its holdout yet: there is no number for the code to reproduce")
    changed = studio_programme.stale_reasons(project)
    if changed:
        raise CodeRefused("the programme is stale (" + "; ".join(changed) + "): the code would reproduce a number the project's data no longer gives. Run the programme again first.")
    plan = studio_programme.plan_of(project)
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    files = files_for(store)
    copies: dict[str, str] = {}

    def copy(filename: str, sha: Optional[str]) -> str:
        try:
            source = files.path(files.key_of(store.data_dir(project["id"]) / filename), expected=sha)
        except FileError as exc:
            raise sd.DataError(str(exc)) from None
        shutil.copyfile(source, out / "data" / filename)
        copies[filename] = filename
        return filename

    main = copy(project["data"]["filename"], project["data"].get("sha256"))
    tables = {name: copy(t["filename"], t.get("sha256")) for name, t in (project.get("tables") or {}).items()}
    cases = copy(project["labels"]["filename"], project["labels"].get("sha256")) if project.get("labels") else None
    shutil.copytree(ROOT / "dclab_core", out / "vendor" / "dclab_core", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "replicas", "schemas"))
    (out / "problem_map.json").write_text(json.dumps(plan["problem_map"], indent=1) + "\n", encoding="utf-8")
    (out / "solution_map.json").write_text(json.dumps(plan["solution_map"], indent=1) + "\n", encoding="utf-8")
    (out / "programme.json").write_text(p.model_dump_json(indent=1) + "\n", encoding="utf-8")
    h = p.holdout
    (out / "expected.json").write_text(json.dumps({"kind": "programme", "metric": h.metric, "value": h.composed.value, "chosen": h.chosen, "best_leaf": h.best_leaf,
                                                   "fingerprint": p.fingerprint, "difference": h.difference.value, "difference_low": h.difference.low,
                                                   "difference_high": h.difference.high}, indent=1) + "\n", encoding="utf-8")
    script = HEADER + "\n\n" + PROGRAMME.format(main=_c(main), tables=_c(tables), cases=_c(cases), key=_c(list((project.get("labels") or {}).get("key") or [])),
                                              exclude=_c(studio_programme.excluded(project)), budget=_c(studio_programme.budget_of(project)), seed=studio_programme.SEED,
                                              bootstraps=studio_programme.BOOTSTRAPS)
    (out / "programme.py").write_text(script + "\n", encoding="utf-8")
    suffixes = {Path(name).suffix.lower() for name in copies}
    needs = [*BASE, *(["pyarrow"] if ".parquet" in suffixes else []), *(["openpyxl"] if suffixes & {".xlsx", ".xls"} else [])]  # what reading the tables as attached needs
    pins = []
    for package in needs:
        try:
            pins.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            pins.append(package)
    (out / "requirements.txt").write_text("\n".join(pins) + "\n", encoding="utf-8")
    (out / "README.md").write_text(textwrap.dedent(f"""\
        # {project.get('name') or 'Project'}: its programme as code

        Exported by DCLab from the project's programme: a problem of several parts, solved leaf by leaf and composed.
        `python programme.py` runs `dclab_core.programme` on `data/` with the recorded split, budget and seed, prints the report
        (a section per leaf, the composition, the holdout read once) and exits 1 unless it chooses the same composition
        ({h.chosen}) and scores the same holdout {h.metric} ({h.composed.value}).

        `problem_map.json` and `solution_map.json` are the plan the person confirmed; `programme.json` is the recorded run.
        `data/` holds the person's own tables: keep this folder where that data may be kept. The cases are those someone confirmed,
        partial and biased towards what is easy to see: a score here is evidence about the method, not production approval.
        """), encoding="utf-8")
    store.log(project["id"], "code_exported", {"kind": "programme", "metric": h.metric, "folder": out.name})
    import hashlib

    from .. import audit

    digests = {f"data/{name}": hashlib.sha256((out / "data" / name).read_bytes()).hexdigest() for name in sorted(copies)}
    audit.record_for(store, "data_export", actor, project={"id": project["id"], "name": project.get("name")}, move="export_code", status="allowed", files=digests)
    return {"folder": str(out), "kind": "programme", "metric": h.metric, "value": h.composed.value,
            "steps": [l.block for l in p.leaves if l.status == "built"] + ["composition", "holdout"]}


def export(store: Any, project_id: str, out: Path, actor: str = "human") -> dict[str, Any]:
    """Write the project's code folder to ``out`` and return what it holds."""
    from . import data as sd

    project = store.get(project_id)
    if project.get("programme_plan") and not any(((project.get("stages") or {}).get(st) or {}).get("status") in ("completed", "approved") for st in ("data", "final")):
        return export_programme(store, project, out, actor)  # a problem of several parts runs as a programme (24.5), not as the five stages
    made = cells(store, project_id)
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    if made["prepared"] is not None:  # the rows as the engine prepared them, with the holdout it locked
        b = made["prepared"].bundle
        rows = b.X.copy()
        rows["__target__"] = b.y.to_numpy()
        part = np.array(["none"] * len(rows), dtype=object)
        part[np.asarray(b.train_idx, dtype=int)] = "train"
        part[np.asarray(b.test_idx, dtype=int)] = "holdout"
        rows["__part__"] = part
        rows.reset_index(drop=True).to_pickle(out / "data" / "prepared.pkl")
        from dataclasses import asdict

        from ..expansion import runner as rn

        (out / "spec.json").write_text(json.dumps(asdict(b.spec), default=list, indent=1) + "\n", encoding="utf-8")
        (out / "recipes.json").write_text(json.dumps(rn.RECIPES[b.spec.key], default=list, indent=1) + "\n", encoding="utf-8")
    else:  # the tables as attached, byte for byte (checked against their recorded SHA-256 on the way out)
        from ..storage.files import FileError, files_for

        shutil.copyfile(sd.data_path(store, project), out / "data" / project["data"]["filename"])
        files = files_for(store)
        for name, t in (project.get("tables") or {}).items():
            try:
                source = files.path(files.key_of(store.data_dir(project_id) / t["filename"]), expected=t.get("sha256"))
            except FileError as exc:
                raise sd.DataError(str(exc)) from None
            shutil.copyfile(source, out / "data" / t["filename"])
    vendor = out / "vendor"
    shutil.copytree(ROOT / "dclab_core", vendor / "dclab_core", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "replicas", "schemas"))
    wanted = [s for s in VENDOR if "runner" in made["vendor"] or (s.endswith("detectors.py") or s.endswith("__init__.py")) and "detectors" in made["vendor"]]
    for source in wanted:
        (vendor / VENDOR[source]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, vendor / VENDOR[source])
    if wanted:
        (vendor / "dclab_rnd" / "__init__.py").write_text('"""The parts of DCLab R&D this code needs."""\n', encoding="utf-8")
        if "runner" not in made["vendor"]:  # the detectors alone: an empty package for the expansion folder
            (vendor / "dclab_rnd" / "expansion" / "__init__.py").write_text("", encoding="utf-8")
    (out / "pipeline.py").write_text(_module(made["blocks"], project), encoding="utf-8")
    (out / "pipeline.ipynb").write_text(json.dumps(_notebook(made["blocks"], project), indent=1) + "\n", encoding="utf-8")
    (out / "expected.json").write_text(json.dumps({"metric": made["metric"], "value": made["value"], "kind": made["kind"],
                                                   "engine_profile": made["profile"]}, indent=1) + "\n", encoding="utf-8")
    from importlib import metadata

    pins = []
    for package in made["requirements"]:
        try:
            pins.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            pins.append(package)
    (out / "requirements.txt").write_text("\n".join(pins) + "\n", encoding="utf-8")
    (out / "README.md").write_text(textwrap.dedent(f"""\
        # {project.get('name') or 'Project'}: its pipeline as code

        Exported by DCLab from the project's records: one block per step of its pipeline (`pipeline.py`, and the same
        cells in `pipeline.ipynb`). `python pipeline.py` computes the holdout {made['metric']} again and exits 1 when it
        differs from the recorded {made['value']}.

        `data/` holds {"the rows as the engine prepared them (the solution applied, labels joined, rows sampled)" if made['prepared'] is not None
                       else "the project's own tables, as attached"}: keep this folder where that data may be kept. A score is evidence
        about the method, not production approval.
        """), encoding="utf-8")
    store.log(project_id, "code_exported", {"kind": made["kind"], "metric": made["metric"], "folder": out.name})
    import hashlib

    from .. import audit  # the folder holds the person's data: who exported it, and which files, are in the audit trail

    digests = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out / "data").iterdir())}
    audit.record_for(store, "data_export", actor, project={"id": project_id, "name": project.get("name")}, move="export_code", status="allowed",
                     files=digests)
    return {"folder": str(out), "kind": made["kind"], "metric": made["metric"], "value": made["value"], "steps": [s for s, _ in made["blocks"]]}
