"""The five-stage workflow on a project's own data.

Every stage is deterministic code built from the expansion campaign's primitives
(locked holdout, training-only folds, fit-fold recipes, explicit selection rules).
The same code that produced the R&D evidence runs the person's project, so each
record has the same shape as a campaign result and the agent can cite like with like.
"""

from __future__ import annotations

import copy
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dclab_rnd import kinds
from dclab_rnd.categoricals import HEURISTIC_RULE, category_code_columns
from dclab_rnd.expansion import datasets as ds
from dclab_rnd.expansion import runner as rn

from . import agent, graph
from .solution import Solution
from .data import column_kind, data_path, load_table, sha256
from .store import STAGE_KEYS, ProjectStore

TASK_TYPE = {task: kinds.get(task).task_type for task in kinds.BASE_TASKS}  # each task's own kind's machine (16.1)
STAGES: list[dict[str, str]] = [
    {"key": "data", "kind": "data_understanding", "title": "Understand the data", "workflow": "WF-04",
     "question": "What is in the data, and what could make a score untrustworthy?"},
    {"key": "leakage", "kind": "leakage_audit", "title": "Audit leakage", "workflow": "WF-05",
     "question": "Which columns would not exist at the prediction moment, and how much would they inflate the score?"},
    {"key": "features", "kind": "feature_engineering", "title": "Climb the feature ladder", "workflow": "WF-06",
     "question": "Which feature recipe earns its place on identical folds?"},
    {"key": "models", "kind": "model_selection", "title": "Screen algorithms", "workflow": "WF-07",
     "question": "Which model family leads on identical folds, once fold spread is counted?"},
    {"key": "final", "kind": "optimization_reliability", "title": "Tune and confirm on the holdout", "workflow": "WF-08",
     "question": "Does careful tuning help, and what is the honest score on rows the model never saw?"},
]
STAGE_BY_KEY = {s["key"]: s for s in STAGES}
QUICK_ROWS = 3000
SPATIAL_BLOCK = "_spatial_block"  # 16.10: the derived group column of a split by blocks of places
DEFAULT_FOLDS, FOLD_CAPS = 3, (2, 10)


def cv_folds(project: dict[str, Any]) -> int:
    """Training folds for this project: the wizard's choice (project settings), 3 when none was made."""
    try:
        folds = int((project.get("settings") or {}).get("folds") or DEFAULT_FOLDS)
    except (TypeError, ValueError):
        folds = DEFAULT_FOLDS
    return max(FOLD_CAPS[0], min(folds, FOLD_CAPS[1]))


def split_for(solution: Any, pack: str | None = None) -> str:
    """The split the engine uses. The solution decides it, not a setting, through its kind (16.1); every kind today
    keeps the shared rule (kinds.default_split): a declared time column means the holdout is the latest period
    (DCLAB-R02); otherwise a declared group column keeps each group on one side; otherwise a stratified random split;
    time and group together keep both (15.3). ``solution`` is a Solution or its dict, possibly still incomplete."""
    get = solution.get if isinstance(solution, dict) else lambda key: getattr(solution, key, None)
    if get("kind") or get("task") in kinds.BASE_TASKS:
        return kinds.resolve(solution, pack).split(solution)
    return kinds.default_split(solution)


def pack_of(project: dict[str, Any]) -> str | None:
    """The project's Domain pack key (15.6): its own, else the draft's it was built from."""
    return project.get("pack") or ((project.get("draft") or {}).get("pack") or {}).get("key")
_BUNDLES: dict[tuple[str, str, str], Any] = {}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------- prepare


class Prepared:
    def __init__(self, frame: pd.DataFrame, solution: Solution, bundle: ds.TaskBundle, roles: dict[str, str],
                 class_labels: list[str] | None, recipes: dict[str, dict[str, Any]], sampling: dict[str, Any], code_rule: str = ""):
        self.frame, self.solution, self.bundle, self.roles = frame, solution, bundle, roles
        self.class_labels, self.recipes, self.sampling = class_labels, recipes, sampling
        self.code_rule = code_rule  # how bundle.spec.categorical_columns was decided
        self.plan: dict[str, Any] | None = None  # the project's adopted dclab_core plan (17.4), set on every prepare
        self.label_info: dict[str, Any] | None = None  # the project's label file (15.2), set on every prepare
        self.critic: Any = None  # the critic's advisor (18.6) when the project turns it on, set on every prepare
        self.kind: kinds.ProblemKind = kinds.get(solution.task)  # the problem kind (16.1), set on every prepare

    @property
    def task(self) -> str:
        return self.solution.task

    @property
    def metric(self) -> str:
        return self.bundle.spec.primary_metric


def prepare(store: ProjectStore, project: dict[str, Any]) -> Prepared:
    """Load the table, apply the solution, lock the holdout and register the feature recipes."""
    if not project.get("data") or not project.get("solution"):
        raise ValueError("The project needs data and a solution first")
    solution = Solution(**project["solution"])
    kind = solution.problem_kind(pack_of(project))
    declared = project["data"].get("categorical")  # the R&D's list for one of its samples; None for an uploaded table
    key = (project["id"], project["data"]["sha256"], _hash(project["solution"]) + str(project["settings"]) + str(declared)
           + str(project.get("pack")) + _hash(project.get("labels")) + _hash(project.get("feature_programs")))  # each changes what runs
    if key in _BUNDLES:
        _BUNDLES[key].plan, _BUNDLES[key].label_info = project.get("plan"), project.get("labels")
        _BUNDLES[key].critic = _critic_advisor(project)
        return _BUNDLES[key]
    path = data_path(store, project)  # checked against the recorded SHA-256 (package 9.3)
    frame = load_table(path)
    if project.get("labels") and project["labels"].get("target") == solution.target:  # 15.2: the target from the label file
        from . import labels as label_file

        frame = label_file.apply(store, project, frame)
    solution.check_columns(list(frame.columns))
    # Yes/no columns arrive as booleans (cleaned uploads, CSVs with True/False); numpy cannot take quantiles or
    # differences of booleans, so features become 0/1 with missing kept missing. The target is compared as text.
    for column in frame.columns:
        if column != solution.target and (pd.api.types.is_bool_dtype(frame[column]) or str(frame[column].dtype) == "boolean"):
            frame[column] = pd.to_numeric(frame[column].astype("object").map({True: 1.0, False: 0.0}), errors="coerce")
    source_rows = len(frame)
    frame = frame[frame[solution.target].notna()].reset_index(drop=True)
    y, class_labels = kind.target(frame, solution)
    keep = y.notna()
    frame, y = frame[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    X = frame.drop(columns=[solution.target])

    roles: dict[str, str] = {}
    identifiers: dict[str, str] = {c: "declared identifier (never a feature)" for c in solution.identifiers if c in X}
    text_columns, excluded = kind.text_columns(solution, X)
    identifiers.update(excluded)
    task_type = kind.machine(solution, text_columns)
    time_column = solution.time_column
    if time_column:
        if column_kind(X[time_column]) == "datetime":
            stamps = pd.to_datetime(X[time_column], errors="coerce")
            valid = stamps.notna()
            X, y, frame = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True), frame[valid].reset_index(drop=True)
            X[time_column] = pd.to_datetime(X[time_column], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            identifiers[time_column] = "time key: orders the split; only calendar features are derived from it"
            datetime_time = True
        else:
            X[time_column] = pd.to_numeric(X[time_column], errors="coerce")
            identifiers[time_column] = "time key: orders the split; a dataset-relative counter is not a production feature"
            datetime_time = False
    else:
        datetime_time = False
    for column in X.columns:
        if column not in identifiers and column != solution.group_column and column not in text_columns and column_kind(X[column]) == "datetime":
            identifiers[column] = "datetime column that is not the declared time column; excluded to avoid one-hot dates"
    blocked = {f.column: (f.reason or "declared unavailable at the prediction moment") for f in solution.forbidden if f.column in X}
    for column in X.columns:
        roles[column] = ("blocked" if column in blocked else "identifier" if column in identifiers and column != time_column
                         else "time" if column == time_column else "group" if column == solution.group_column
                         else "text" if column in text_columns else "feature")

    group_column = solution.group_column
    if solution.spatial is not None:  # 16.10: blocks of places derived from the coordinates are the groups of every split
        from dclab_core.spatial import blocks

        sp = solution.spatial
        X[SPATIAL_BLOCK] = blocks(pd.to_numeric(X[sp.lat_column], errors="coerce"), pd.to_numeric(X[sp.lon_column], errors="coerce"), sp.block_km)
        roles[SPATIAL_BLOCK] = "group"
        group_column = SPATIAL_BLOCK
        need = max(6, cv_folds(project) + 1)  # the holdout's group folds and the training folds each need whole blocks
        found = int(pd.Series(X[SPATIAL_BLOCK]).nunique())
        if found < need:
            raise ValueError(f"{found} blocks of {sp.block_km:g} km cover these places; the split needs at least {need}: choose smaller blocks")

    settings = project.get("settings") or {}
    max_rows = int(settings.get("max_rows") or 20000)
    if settings.get("quick"):
        max_rows = min(max_rows, QUICK_ROWS)
    sampling: dict[str, Any] = {"source_rows": source_rows, "rows_with_target": int(len(X)), "max_rows": max_rows}
    if len(X) > max_rows:
        if time_column:
            order = np.argsort(ds._time_key(X[time_column]), kind="stable")[-max_rows:]
            sampling["rule"] = f"kept the latest {max_rows} rows by `{time_column}` (label-independent)"
        else:
            order = np.sort(np.random.default_rng(ds.RANDOM_STATE).choice(len(X), max_rows, replace=False))
            sampling["rule"] = f"random sample of {max_rows} rows, seed {ds.RANDOM_STATE} (label-independent)"
        X, y = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
    sampling["rows_used"] = int(len(X))

    positive_rate = kind.positive_rate(y)
    metric = solution.metric or kind.default_metric(solution, positive_rate)  # a pack's defaults live in its kind (15.6, 16.1)
    split = kind.split(solution)
    regression = kind.base_task == "regression"
    spec = ds.DatasetSpec(
        key=f"project_{project['id']}", name=project["name"], task_type=task_type, primary_metric=metric,
        target=solution.target, url=f"upload://{project['data']['filename']}", filename=project["data"]["filename"],
        sha256=project["data"]["sha256"], description=project.get("goal") or project["name"],
        decision_time_contract=solution.prediction_moment, split_strategy=split, source_rows=source_rows,
        feature_count=int(X.shape[1]), blocked_features=blocked, identifier_columns=identifiers,
        text_columns=text_columns, time_column=time_column, group_column=group_column,
        positive_rate=positive_rate,
        settings={"precision_target": 0.9, "fe_tolerance": 0.01 if regression else 0.002, "fe_tolerance_relative": regression,
                  "tuning_margin": 0.01 if regression else 0.001, "tuning_margin_relative": regression,
                  **kind.spec_settings(solution)},
    )
    bundle = kind.bundle(spec, X, y, data_paths=[path], source_rows=source_rows, sampling=sampling,
                         class_labels=class_labels, cv_folds=cv_folds(project))
    # Numeric columns that stand for categories get no log or product feature (DCLAB-R11). Decided once,
    # on training rows: the declaration when there is one, otherwise the conservative heuristic.
    features = [c for c, role in roles.items() if role == "feature"]
    codes = category_code_columns(bundle.X_train[features], declared)
    bundle.spec = replace(spec, categorical_columns=tuple(codes))
    code_rule = "declared in the R&D dataset catalog for this sample" if declared is not None else HEURISTIC_RULE
    recipes = _register_recipes(bundle.spec, X, roles, datetime_time, project.get("feature_programs"), kind)
    prepared = Prepared(frame, solution, bundle, roles, class_labels, recipes, sampling, code_rule)
    prepared.kind = kind
    prepared.plan, prepared.label_info = project.get("plan"), project.get("labels")
    prepared.critic = _critic_advisor(project)
    _BUNDLES.clear()
    _BUNDLES[key] = prepared
    return prepared


def _hash(value: Any) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _behaviour_config(spec: ds.DatasetSpec, X: pd.DataFrame, roles: dict[str, str], numeric: list[str]) -> dict[str, Any] | None:
    """The entity behaviour recipe's settings (15.4): the group column as the entity, the time column (or another
    date column) for days and order, coordinate pairs for places, the mode from the prediction moment."""
    from dclab_core.behaviour import coordinate_pairs, mode_for

    # a block of places groups the folds; it is not an entity whose behaviour is counted
    entities = [spec.group_column] if spec.group_column and spec.group_column in X and spec.group_column != SPATIAL_BLOCK else []
    usable = [c for c, role in roles.items() if role in ("feature", "group", "time", "identifier")]
    if not entities and len(coordinate_pairs(usable)) < 1:
        return None
    time = spec.time_column or next((c for c, role in roles.items() if role == "identifier" and column_kind(X[c]) == "datetime"), None)
    return {"entities": entities, "time_column": time, "mode": mode_for(spec.decision_time_contract), "numeric": numeric[:8],
            "allowed": [c for c, role in roles.items() if role == "feature"],
            "grouped": [spec.group_column] if spec.group_column and spec.split_strategy in ("group", "time_group") else []}


def _programs_config(spec: ds.DatasetSpec, X: pd.DataFrame, roles: dict[str, str], engineered: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """The engineered recipe's settings (18.5): the project's programs, checked again against this table's usable columns."""
    if not engineered:
        return None
    from dclab_core.behaviour import mode_for
    from dclab_core.features import Language

    allowed = [c for c, role in roles.items() if role == "feature"]
    entities = [c for c, role in roles.items() if role == "group"]
    time = spec.time_column or next((c for c, role in roles.items() if role == "identifier" and column_kind(X[c]) == "datetime"), None)
    language = Language(allowed, target=spec.target, time_column=time, entities=entities)
    kept, _ = language.check_all(engineered)
    if not kept:
        return None
    return {"programs": kept, "allowed": allowed, "entities": entities, "time_column": time, "target": spec.target,
            "mode": mode_for(spec.decision_time_contract)}


def _register_recipes(spec: ds.DatasetSpec, X: pd.DataFrame, roles: dict[str, str], datetime_time: bool,
                      engineered: list[dict[str, Any]] | None = None, kind: kinds.ProblemKind | None = None) -> dict[str, dict[str, Any]]:
    features = [c for c, role in roles.items() if role == "feature"]
    codes = set(spec.categorical_columns)
    numeric = [c for c in features if c not in codes and pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 2]
    left_out = f" ({len(codes)} category-code column{'' if len(codes) == 1 else 's'} left out)" if codes else ""
    if spec.task_type == "text_tabular_binary":
        recipes = {
            "tabular": {"description": f"{len(features)} structured columns only, no text", "tabular": True},
            "tfidf_word": {"description": "word 1-2gram TF-IDF of the text columns", "word": True},
            "combined": {"description": "word TF-IDF + structured columns + text-length/missing meta", "word": True, "tabular": True, "meta": True},
        }
        rn.LEAKAGE_RECIPE[spec.key] = "combined"
    else:
        recipes = {"raw": {"description": f"the {len(features)} safe input columns as given (categories and category codes one-hot, fitted per fold)", "derive": ()}}
        if numeric:
            recipes["log_numeric"] = {"description": f"raw + signed log of the {len(numeric)} numeric inputs{left_out}", "derive": ("log_numeric",)}
        if datetime_time:
            recipes["calendar"] = {"description": "raw + calendar features from the time column (day-of-year sin/cos, day of month, ISO week, weekend)", "derive": ("calendar",)}
        if 2 <= len(numeric) <= 8:
            pairs = len(numeric) * (len(numeric) - 1) // 2
            recipes["poly2"] = {"description": f"raw + all {pairs} pairwise products of the numeric inputs{left_out}", "derive": ("poly2",)}
        programs = _programs_config(spec, X, roles, engineered)
        if programs is not None:  # 18.5: the feature engineer's checked programs, screened like any recipe
            recipes["engineered"] = {"description": f"raw + {len(programs['programs'])} proposed feature programs: "
                                                    + ", ".join(p["name"] for p in programs["programs"][:8])
                                                    + f"; mode `{programs['mode']}`", "derive": (), "programs": programs}
        behaviour = _behaviour_config(spec, X, roles, numeric)
        if behaviour is not None:  # 15.4: an entity (or places) to learn behaviour from
            recipes["entity_behaviour"] = {"description": f"raw + behaviour of each {', '.join(behaviour['entities']) or 'place'}: counts per entity, "
                                                          "day, place and route, the row against its entity, shrunk rates from the fit rows; "
                                                          f"mode `{behaviour['mode']}` ({'the rows scored together are known, never their labels' if behaviour['mode'] == 'audit' else 'only earlier rows'})",
                                           "derive": (), "behaviour": behaviour}
        if spec.task_type != "timeseries_regression" and len(features) >= 10:
            k = max(5, int(round(0.6 * len(features))))
            recipes["selected_mi"] = {"description": f"top-{k} inputs by fit-fold mutual information", "derive": (), "select_k": k}
        rn.LEAKAGE_RECIPE[spec.key] = "raw"
    if kind is not None:  # 16.1: the kind's own recipes join the ladder, never replacing a shared one
        recipes.update({k: v for k, v in kind.extra_recipes(spec, X, roles).items() if k not in recipes})
    rn.RECIPES[spec.key] = recipes
    return recipes


# ---------------------------------------------------------------------- stage executors


def _claim(cid: str, kind: str, statement: str, evidence: list[str], limitations: list[str] | None = None) -> dict[str, Any]:
    return rn._claim(cid, kind, statement, evidence, limitations)


def _label(metric: str) -> str:
    return rn.METRIC_LABEL.get(metric, metric)


def stage_data(p: Prepared, eid: str) -> tuple[dict, list, str]:
    bundle, spec = p.bundle, p.bundle.spec
    X_tr, y_tr, X_te = bundle.X_train, bundle.y_train, bundle.X_test
    profiles = []
    for column in X_tr.columns:
        series = X_tr[column]
        missing = float(series.isna().mean())
        unique_ratio = float(series.nunique(dropna=False) / max(len(series), 1))
        top_frequency = float(series.value_counts(dropna=False, normalize=True).iloc[0]) if len(series) else 0.0
        psi = rn._psi(series, X_te[column]) if pd.api.types.is_numeric_dtype(series) else None
        risks = []
        if missing > 0.05:
            risks.append("missingness")
        if rn._identifier_like(series, unique_ratio) and p.roles.get(column) == "feature":
            risks.append("identifier_like")
        if top_frequency > 0.995:
            risks.append("near_constant")
        if psi is not None and psi > 0.20:
            risks.append("train_holdout_feature_shift")
        profiles.append({"feature": str(column), "role": p.roles.get(column, "feature"), "dtype": str(series.dtype),
                         "missing_rate": missing, "unique_count": int(series.nunique(dropna=False)), "unique_ratio": unique_ratio,
                         "top_value_frequency": top_frequency, "train_vs_holdout_psi": psi, "production_risks": risks})
    feature_cols = [c for c, role in p.roles.items() if role == "feature"]
    evidence: dict[str, Any] = {
        "source_rows": bundle.source_rows, "raw_feature_count": int(X_tr.shape[1]), "usable_feature_count": len(feature_cols),
        "train_rows": int(len(X_tr)), "holdout_rows": int(len(X_te)), "holdout_split": bundle.holdout_description,
        "cv_protocol": bundle.cv_description, "sampling": bundle.sampling,
        "missing_cell_rate_train": float(X_tr.isna().sum().sum() / max(X_tr.size, 1)),
        "duplicate_feature_row_rate_train": float(X_tr[feature_cols].duplicated().mean()) if feature_cols else 0.0,
        "duplicate_feature_rows_train": int(X_tr[feature_cols].duplicated().sum()) if feature_cols else 0,
        "feature_profiles": profiles,
        "risk_features": {pr["feature"]: pr["production_risks"] for pr in profiles if pr["production_risks"]},
        "column_roles": dict(p.roles),
        "note": "Holdout feature distributions (never labels) are compared via PSI; the holdout labels stay sealed until the final stage.",
    }
    evidence["problem_kind"] = p.kind.key  # 16.1: the kind whose summary, claim and checks these are
    evidence["target_summary"], target_text = p.kind.summarise_target(y_tr, p.class_labels, p.solution)
    if spec.time_column:
        evidence["time_range"] = {"train": [str(X_tr[spec.time_column].iloc[0]), str(X_tr[spec.time_column].iloc[-1])],
                                  "holdout": [str(X_te[spec.time_column].iloc[0]), str(X_te[spec.time_column].iloc[-1])]}
    if spec.group_column:
        sizes = X_tr[spec.group_column].value_counts()
        evidence["group_stats"] = {"train_groups": int(len(sizes)), "holdout_groups": int(X_te[spec.group_column].nunique()),
                                   "largest_group_share": float(sizes.iloc[0] / sizes.sum()), "median_rows_per_group": float(sizes.median())}
    if p.solution.spatial is not None and spec.group_column == SPATIAL_BLOCK:  # 16.10: how much a random split would flatter here
        from dclab_core import spatial

        sp = p.solution.spatial
        held = set(X_te[SPATIAL_BLOCK])
        evidence["spatial"] = {"block_km": sp.block_km, "lat_column": sp.lat_column, "lon_column": sp.lon_column,
                               "train_blocks": int(X_tr[SPATIAL_BLOCK].nunique()), "holdout_blocks": int(len(held)),
                               "blocks_on_both_sides": int(len(held & set(X_tr[SPATIAL_BLOCK])))}
        numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_tr[c])]
        try:
            head = X_tr.iloc[:5000]  # the training rows only: the holdout stays sealed
            evidence["spatial"]["random_split_optimism"] = spatial.split_optimism(
                head[numeric], np.asarray(y_tr)[:5000], head[sp.lat_column], head[sp.lon_column], block_km=sp.block_km,
                task="binary" if p.kind.base_task == "binary" else "regression")
        except (ValueError, KeyError) as exc:
            evidence["spatial"]["random_split_optimism"] = {"refused": str(exc)[:300]}
    if spec.text_columns:
        evidence["text_stats"] = {c: {"missing_rate": float(X_tr[c].isna().mean()),
                                      "median_words": float(X_tr[c].dropna().astype(str).str.split().str.len().median())} for c in spec.text_columns}
    risks = evidence["risk_features"]
    claims = [
        _claim(f"{eid}-C1", "fact", f"{bundle.source_rows} source rows and {X_tr.shape[1]} columns; {len(feature_cols)} are usable inputs under the solution. "
               f"The locked holdout is {bundle.holdout_description} ({len(X_te)} rows); {len(X_tr)} training rows remain with {target_text}.",
               ["evidence.source_rows", "evidence.usable_feature_count", "evidence.holdout_split", "evidence.target_summary"]),
        _claim(f"{eid}-C2", "risk", f"{len(risks)} of {X_tr.shape[1]} columns triggered a review rule (missingness > 5%, identifier-like, near-constant, or train/holdout PSI > 0.20): "
               + ("; ".join(f"{k}: {', '.join(v)}" for k, v in list(risks.items())[:12]) or "none") + ".",
               ["evidence.feature_profiles", "evidence.risk_features"],
               ["PSI on feature distributions is a smoke check; it uses no holdout labels and does not prove production stability."]),
    ]
    claim_kind, statement, paths = p.kind.target_claim(evidence["target_summary"], _label(p.metric), spec)
    claims.append(_claim(f"{eid}-C3", claim_kind, statement, paths))
    if p.solution.spatial is None and not spec.group_column and not spec.time_column and p.kind.base_task in ("binary", "regression"):
        from dclab_core.analyse import _places  # 16.10: places that cluster under a random split are said, even when no plan set blocks

        found = _places(X_tr, pd.Series(np.asarray(y_tr), index=X_tr.index), p.kind.base_task, None)
        if found and found[0].severity == "warning":
            evidence["places_warning"] = {"statement": found[0].statement, **found[0].evidence}
            claims.append(_claim(f"{eid}-C4", "risk", "The split is random, but " + found[0].statement[0].lower() + found[0].statement[1:]
                                 + " " + found[0].implication, ["evidence.places_warning"]))
    summary = (f"Profiled {len(X_tr)} training rows × {X_tr.shape[1]} columns (holdout: {bundle.holdout_description}); "
               f"per-column missingness, uniqueness and PSI; {len(risks)} columns flagged for review.")
    return evidence, claims, summary


def stage_leakage(p: Prepared, eid: str) -> tuple[dict, list, str]:
    bundle, spec, task = p.bundle, p.bundle.spec, p.bundle.task_type
    metric, higher = spec.primary_metric, spec.higher_is_better
    X_tr, y_tr = bundle.X_train, bundle.y_train
    declared = [c for c in spec.blocked_features if c in X_tr.columns]
    skip = [c for c, role in p.roles.items() if role in ("identifier", "time", "group", "text")]
    sample_X, sample_y = X_tr, y_tr
    if len(X_tr) > 60000:
        sample_X = X_tr.sample(60000, random_state=ds.RANDOM_STATE)
        sample_y = y_tr.loc[sample_X.index]
    findings = rn.suspicious_features(sample_X, sample_y, task, declared=declared, skip=skip)
    probe = [c for c in sample_X.columns if c not in skip][:3]
    canary = rn._canary_check(sample_X[probe], sample_y, task, skip) if probe else {"passed": True, "note": "no probe columns"}
    ranking = rn.univariate_ranking(sample_X, sample_y, task, skip=skip)
    audit_model = rn._fallback_family("lightgbm") if task != "text_tabular_binary" else "logistic_regression"
    recipe = rn.LEAKAGE_RECIPE[spec.key]
    safe = rn.cross_validate(bundle, recipe, audit_model, config_id="safe")
    comparison: dict[str, Any] = {"model": audit_model, "recipe": recipe, "metric": metric, "safe": safe}
    lift = 0.0
    if declared:
        unsafe = rn.cross_validate(bundle, recipe, audit_model, include_blocked=declared, config_id="unsafe_all_blocked")
        lift = rn.oriented_gain(rn._mean(unsafe, metric), rn._mean(safe, metric), higher)
        comparison["unsafe"], comparison["apparent_lift"] = unsafe, lift
        if 1 < len(declared) <= 6:
            comparison["per_column"] = {}
            for column in declared:
                single = rn.cross_validate(bundle, recipe, audit_model, include_blocked=[column], config_id=f"unsafe_{column}")
                comparison["per_column"][column] = {"metric_mean": rn._mean(single, metric),
                                                    "apparent_lift": rn.oriented_gain(rn._mean(single, metric), rn._mean(safe, metric), higher)}
    non_declared = [f for f in findings if "declared_post_outcome_or_contested" not in f["reasons"]]
    evidence: dict[str, Any] = {
        "declared_leakage_features": declared,
        "policy_rationale": " ".join(f"`{c}`: {r}" for c, r in spec.blocked_features.items()) or "No column is blocked: " + spec.decision_time_contract,
        "identifier_exclusions": dict(spec.identifier_columns), "decision_time_contract": spec.decision_time_contract,
        "heuristic_review_candidates": findings,
        "heuristic_rules": {"strong_univariate_signal": rn.STRONG_SIGNAL[task], "extreme_univariate_signal": rn.EXTREME_SIGNAL[task],
                            "identifier_like": "unique_ratio >= 0.98 and non-numeric or integer-valued", "rows_scanned": int(len(sample_X))},
        "top_univariate_signals": ranking, "detector_canary": canary, "detector_canary_passed": bool(canary.get("passed", False)),
        "safe_vs_unsafe_training_cv": comparison, "apparent_lift": float(lift), "apparent_lift_metric": metric,
        "cv_protocol": bundle.cv_description,
        "warning": "Name, univariate-signal, identity and uniqueness rules propose a review; only decision-time semantics confirm leakage.",
    }
    claims = [_claim(f"{eid}-C1", "fact", f"The {task} leakage detector caught its synthetic canaries: {canary}.",
                     ["evidence.detector_canary", "evidence.detector_canary_passed"])]
    if declared:
        per = comparison.get("per_column", {})
        per_text = (" Per column: " + ", ".join(f"{c} {rn._signed(v['apparent_lift'], metric)}" for c, v in per.items()) + ".") if per else ""
        claims.append(_claim(f"{eid}-C2", "decision",
                             f"Forbidden under the solution: {declared}. Including them moves training-CV {_label(metric)} ({audit_model}, {recipe} recipe) "
                             f"from {rn._ms(safe, metric)} to {rn._ms(comparison['unsafe'], metric)}, "
                             + (f"an apparent lift of {rn._signed(lift, metric)} that would not exist in production." if lift > 0
                                else f"no apparent lift ({rn._signed(lift, metric)}); the exclusion rests on decision-time semantics, not on CV inflation.") + per_text,
                             ["evidence.declared_leakage_features", "evidence.policy_rationale", "evidence.safe_vs_unsafe_training_cv", "evidence.apparent_lift"],
                             [evidence["warning"]]))
    else:
        claims.append(_claim(f"{eid}-C2", "decision", f"No column is forbidden under the solution; the safe {audit_model} {recipe} baseline scores "
                             f"{rn._ms(safe, metric)} training-CV {_label(metric)}, apparent lift 0 by construction.",
                             ["evidence.declared_leakage_features", "evidence.policy_rationale", "evidence.safe_vs_unsafe_training_cv"], [evidence["warning"]]))
    flagged = ", ".join(f"{f['feature']} ({'/'.join(f['reasons'])})" for f in non_declared) or "none"
    claims.append(_claim(f"{eid}-C3", "risk", f"Heuristics flagged {len(non_declared)} non-declared column(s) for human review: {flagged}. Strongest single-feature signal: "
                         + (f"{ranking[0]['feature']} ({ranking[0]['signal']:.3f})." if ranking else "n/a."),
                         ["evidence.heuristic_review_candidates", "evidence.top_univariate_signals", "evidence.heuristic_rules"],
                         ["Strong signal alone is not leakage; a legitimate lag or a real driver is expected to be predictive."]))
    parts = [f"Scanned {len(sample_X)} training rows with {task} heuristics and synthetic canaries;"]
    if declared:
        parts.append(f"compared {audit_model} on the {recipe} recipe with vs without {declared}: {_label(metric)} {rn._ms(safe, metric)} safe vs "
                     f"{rn._ms(comparison['unsafe'], metric)} unsafe (apparent lift {rn._signed(lift, metric)}).")
    else:
        parts.append(f"no forbidden columns; safe {audit_model} {recipe} baseline {_label(metric)} {rn._ms(safe, metric)}.")
    if spec.split_strategy == "time":
        from sklearn.model_selection import KFold

        folds = len(bundle.cv_splits)  # the same number of folds as the time-ordered protocol it is compared with
        random_splits = list(KFold(n_splits=folds, shuffle=True, random_state=ds.RANDOM_STATE).split(np.arange(len(X_tr))))
        random_cv = rn.cross_validate(bundle, recipe, audit_model, splits=random_splits, config_id="safe_random_kfold", split_label=f"KFold({folds}, shuffle) — ignores time order")
        gap = rn.oriented_gain(rn._mean(random_cv, metric), rn._mean(safe, metric), higher)
        evidence["random_vs_time_cv"] = {"time_ordered": {"metric_mean": rn._mean(safe, metric), "protocol": bundle.cv_description},
                                         "random_kfold": random_cv, "optimism_gap": gap}
        claims.append(_claim(f"{eid}-C4", "risk", f"Random KFold(3) reports {_label(metric)} {rn._ms(random_cv, metric)} for the same safe recipe versus "
                             f"{rn._ms(safe, metric)} with time-ordered CV: random splits look {rn._signed(gap, metric)} better because they see the future.",
                             ["evidence.random_vs_time_cv"]))
        parts.append(f"Random vs time-ordered CV gap: {rn._signed(gap, metric)}.")
    return evidence, claims, " ".join(parts)


def stage_features(p: Prepared, eid: str) -> tuple[dict, list, str]:
    bundle, spec = p.bundle, p.bundle.spec
    metric, higher = spec.primary_metric, spec.higher_is_better
    model = rn._fallback_family("lightgbm") if spec.task_type != "text_tabular_binary" else "logistic_regression"
    tolerance = float(spec.settings.get("fe_tolerance", 0.002))
    relative = bool(spec.settings.get("fe_tolerance_relative", False))
    rows = []
    for name, config in p.recipes.items():
        row = rn.cross_validate(bundle, name, model, config_id=name)
        row["description"] = config["description"]
        rows.append(row)
    selected = rn.select_within_tolerance(rows, metric, higher, tolerance, relative=relative)
    best = rn.rank_rows(rows, metric, higher)[0]
    first = rows[0]
    rule = (f"Choose the recipe with the fewest features (then fastest) whose mean training-CV {_label(metric)} is within "
            + (f"{tolerance:.0%} (relative)" if relative else f"{tolerance}") + f" of the best; all recipes use {model} with default parameters on identical folds ({bundle.cv_description}).")
    evidence = {"stage_results": rows, "stage_model": model, "selected_recipe": selected["recipe"], "selected_recipe_description": selected["description"],
                "selection_rule": rule, "best_recipe": best["recipe"], "selected_metric_mean": rn._mean(selected, metric), "best_metric_mean": rn._mean(best, metric),
                "reference_recipe": first["recipe"], "reference_metric_mean": rn._mean(first, metric),
                "selected_vs_reference_gain": rn.oriented_gain(rn._mean(selected, metric), rn._mean(first, metric), higher), "cv_protocol": bundle.cv_description,
                "category_code_columns": list(spec.categorical_columns), "category_code_rule": p.code_rule,
                "category_code_encoding": "one-hot, levels learned on each fit fold (40 most frequent)" if rn.encoded_codes(spec) else "passed as numbers"}
    table = "; ".join(f"{r['recipe']} ({r['feature_count_mean']:.0f} feats, {rn._ms(r, metric)})" for r in rows)
    claims = [
        _claim(f"{eid}-C1", "decision", f"Selected recipe `{selected['recipe']}` ({selected['feature_count_mean']:.0f} features, {rn._ms(selected, metric)} {_label(metric)}); "
               f"best mean was `{best['recipe']}` at {rn._ms(best, metric)}. Ladder: {table}.",
               ["evidence.stage_results", "evidence.selection_rule", "evidence.selected_recipe"],
               ["Recipe differences within fold noise are not evidence that one representation is better."]),
    ]
    gain = evidence["selected_vs_reference_gain"]
    claims.append(_claim(f"{eid}-C2", "fact", f"Versus the first rung (`{first['recipe']}`), the selected recipe changes {_label(metric)} by {rn._signed(gain, metric)} on identical folds.",
                         ["evidence.selected_vs_reference_gain", "evidence.stage_results"]))
    codes = evidence["category_code_columns"]
    if codes:
        names = ", ".join(f"`{c}`" for c in codes[:12]) + (f" and {len(codes) - 12} more" if len(codes) > 12 else "")
        claims.append(_claim(f"{eid}-C3", "fact", f"Treated as category codes ({p.code_rule}): {names}. "
                             f"Every recipe encodes them as categories ({evidence['category_code_encoding']}) and builds no log or product "
                             "feature from them, because their order and scale are arbitrary (DCLAB-R11).",
                             ["evidence.category_code_columns", "evidence.category_code_rule", "evidence.category_code_encoding"],
                             ["Levels unseen in a fit fold, and levels beyond the 40 most frequent, encode as all zeros.",
                              "A table that stores codes without their labels (as the R&D samples do) yields one-hot columns named by code, not by category."]))
    summary = f"Compared {len(rows)} feature recipes with {model} on {bundle.cv_description}: {table}. Selected `{selected['recipe']}`."
    return evidence, claims, summary


# simplest first: a rule, then detectors, then linear, bagged and boosted models (15.5's tie-break)
SIMPLICITY = ("percentile_rules", "ecod", "isolation_forest", "logistic_regression", "pu_bagging", "extra_trees",
              "hist_gradient_boosting", "lightgbm", "xgboost")


PRACTICAL = 0.9  # a simpler family is preferred only within 10% of the best's mean, and never when the best wins every fold


def _rare_event(p: Prepared) -> bool:
    return p.kind.rare_screen(p.bundle.task_type, p.bundle.spec.positive_rate, p.solution, p.bundle.spec.settings)


def _rare_choice(ranked: list[dict[str, Any]], metric: str, higher: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """The best mean; then the simplest family whose paired fold gap to the best stays within the one-sided 90% t bound
    (dclab_core.review.adopt: the best does not beat it beyond noise)."""
    from dclab_core.review import adopt

    best = max(ranked, key=lambda r: rn._mean(r, metric) if higher else -rn._mean(r, metric))
    best_mean = rn._mean(best, metric)
    for row in sorted(ranked, key=lambda r: SIMPLICITY.index(r["model"]) if r["model"] in SIMPLICITY else len(SIMPLICITY)):
        if row is best:
            return best, {"best": best["model"], "chosen": best["model"], "why": "no simpler family is within noise of the best"}
        pairs = [(a[metric], b[metric]) for a, b in zip(row["fold_metrics"], best["fold_metrics"])
                 if a.get(metric) is not None and b.get(metric) is not None and np.isfinite(a[metric]) and np.isfinite(b[metric])]
        if len(pairs) < 2:
            continue  # folds without a positive cannot compare: the best stays unless a simpler family is shown close
        mean = rn._mean(row, metric)
        close = mean >= PRACTICAL * best_mean if higher else mean <= best_mean / PRACTICAL
        wins_all = all((b > a) if higher else (b < a) for a, b in pairs)
        gap = adopt([a for a, _ in pairs], [b for _, b in pairs], higher)
        if close and not wins_all and not gap["adopted"]:
            return row, {"best": best["model"], "chosen": row["model"], "paired_gap": gap,
                         "why": f"{row['model']} is simpler and the best ({best['model']}) does not beat it beyond noise: {gap['why']}"}
    return best, {"best": best["model"], "chosen": best["model"]}


def stage_models(p: Prepared, eid: str, recipe: str) -> tuple[dict, list, str]:
    bundle, spec = p.bundle, p.bundle.spec
    metric, higher = spec.primary_metric, spec.higher_is_better
    rare = _rare_event(p)
    # 15.5: the rule for a rare event is fixed before the screen runs, and written into the record
    rare_rule = ("Rare event: rules, anomaly detectors and PU bagging are screened with the classifiers on the same folds; take the "
                 f"best mean training-CV {_label(metric)}, then the simplest family ({' < '.join(SIMPLICITY)}) within {1 - PRACTICAL:.0%} of "
                 "the best's mean whose paired fold gap to the best stays within the one-sided 90% t bound (folds without a positive left "
                 "out), never one the best beats on every fold.") if rare else None
    families = p.kind.model_families(rn.model_families(bundle.task_type), rare)
    rows = [rn.cross_validate(bundle, recipe, family, config_id=family) for family in families]
    for row in rows:
        row["adjusted_score"] = rn.adjusted_score(rn._mean(row, metric), rn._std(row, metric), higher)
    ranked = rn.rank_rows(rows, metric, higher)
    choice = None
    if rare:
        chosen_row, choice = _rare_choice(ranked, metric, higher)
        ranked = [chosen_row] + [r for r in ranked if r is not chosen_row]
    selected, runner_up = ranked[0], (ranked[1] if len(ranked) > 1 else ranked[0])
    sign = "-" if higher else "+"
    evidence = {"feature_recipe": recipe, "model_results": rows,
                "ranking": [{"model": r["model"], "mean": rn._mean(r, metric), "std": rn._std(r, metric), "adjusted_score": r["adjusted_score"],
                             "elapsed_seconds": r["elapsed_seconds"], "feature_count_mean": r["feature_count_mean"]} for r in ranked],
                "selected_model": selected["model"],
                "selection_rule": rare_rule or f"Rank by mean training-CV {_label(metric)} {sign} 0.25×fold std ({'higher' if higher else 'lower'} is better), then runtime; identical folds ({bundle.cv_description}); default parameters.",
                **({"rare_event_choice": choice} if choice else {}),
                "selected_metric_mean": rn._mean(selected, metric), "selected_metric_std": rn._std(selected, metric), "cv_protocol": bundle.cv_description}
    table = "; ".join(f"{r['model']} {rn._ms(r, metric)} ({r['elapsed_seconds']:.1f}s)" for r in ranked)
    spread = abs(rn._mean(selected, metric) - rn._mean(runner_up, metric))
    claims = [
        _claim(f"{eid}-C1", "recommendation", f"{selected['model']} leads on the `{recipe}` recipe ({rn._ms(selected, metric)} {_label(metric)}, adjusted {selected['adjusted_score']:.4f}); "
               f"runner-up {runner_up['model']} at {rn._ms(runner_up, metric)}. Ranking: {table}.",
               ["evidence.model_results", "evidence.ranking", "evidence.selection_rule"],
               ["A cross-validation leader is a candidate, not a universal winner, and has not yet seen the locked holdout."]),
        _claim(f"{eid}-C2", "fact", f"The gap between leader and runner-up is {spread:.4f} against a leader fold std of {rn._std(selected, metric):.4f}"
               + ("; the two are within fold noise." if spread <= rn._std(selected, metric) else "; the lead exceeds one fold std."),
               ["evidence.ranking"]),
    ]
    summary = f"Screened {len(rows)} model families on the `{recipe}` recipe and {bundle.cv_description}: {table}."
    review = _review_rounds(p, recipe, selected, {r["model"]: r for r in ranked})
    if review is not None:
        evidence["review"] = review
        if review["family"] != selected["model"]:  # a round kept a change only another family can make: the review's choice
            evidence["screen_leader"], evidence["selected_model"] = selected["model"], review["family"]
            evidence["selected_metric_mean"], evidence["selected_metric_std"] = review["selected_mean"], review["selected_std"]
            evidence["selection_rule"] += (" A review round may then move to another family, when that family with a kept change beats "
                                           "both the leader and its own screen on the same folds (17.6).")
        for r in review["rounds"]:
            if "variant_mean" not in r:  # a change that could not run: said in the record, not claimed
                continue
            who = r["family"] if r["family"] == r["base_family"] else f"{r['family']} (against the leader {r['base_family']})"
            claims.append(_claim(f"{eid}-R{r['round']}", "decision",
                                 f"Review round {r['round']}: {r['experiment']} on {who} ({r['hypothesis']}) {'kept' if r['adopted'] else 'not kept'}: "
                                 f"{_label(metric)} {rn._f(r['base_mean'])} → {rn._f(r['variant_mean'])} on the same folds; {r['why']}.",
                                 ["evidence.review.rounds"],
                                 ["Measured on training folds only; the holdout is still sealed.",
                                  "The change was chosen from the same folds' errors it is tested on, so the kept training-CV score is optimistic; "
                                  "the holdout, used once, is the honest number."]))
        if review["rounds"]:
            kept = [r["experiment"] for r in review["rounds"] if r["adopted"]]
            summary += f" Review: {len(review['rounds'])} round(s) on the same folds, kept {', '.join(kept) if kept else 'nothing'}."
    return evidence, claims, summary


REVIEW_ROUNDS = 2


import contextvars

_ADVISOR: contextvars.ContextVar[Any] = contextvars.ContextVar("dclab_team_advisor", default=None)


class advisor_scope:
    """Inside this scope (this context only: another thread or job is not affected) the critic uses ``advisor``, the
    team's (18.8) with its euro cap, whatever the project's critic setting; None changes nothing."""

    def __init__(self, advisor: Any):
        self.advisor, self.token = advisor, None

    def __enter__(self) -> "advisor_scope":
        self.token = _ADVISOR.set(self.advisor) if self.advisor is not None else None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.token is not None:
            _ADVISOR.reset(self.token)


def _critic_advisor(project: dict[str, Any]) -> Any:
    """The critic (18.6) ranks the review's experiments only when the project turns it on (off until it is measured),
    or inside a team run's scope; never for a private project."""
    if (project.get("settings") or {}).get("private"):
        return None  # private data: no model sees even its summaries
    if _ADVISOR.get() is not None:
        return _ADVISOR.get()
    if not (project.get("settings") or {}).get("critic"):
        return None
    from ..core_advisor import GatewayAdvisor
    from ..models import installed

    return GatewayAdvisor(installed(), project_id=project.get("id"))


def _review_rounds(p: Prepared, recipe: str, selected: dict[str, Any], screened: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """dclab_core.review on the leader's out-of-fold predictions (training rows only), then at most REVIEW_ROUNDS
    experiments, each a cross-validation on the same folds kept only by dclab_core.review.adopt (package 17.6). A change
    the leader's family cannot make is tried on the best-screened family that can (``screened``, in rank order), and kept
    only if it beats both the leader and that family's own screen row.
    None, logged, when the review cannot run: the stage stands as it was."""
    try:
        from dclab_core.analyse import target_cap
        from dclab_core.review import CATALOGUE, adopt, critique, review as review_of, runnable

        bundle, spec, family = p.bundle, p.bundle.spec, selected["model"]
        metric, higher = spec.primary_metric, spec.higher_is_better
        kind = p.kind.base_task
        rounds_allowed = int(spec.settings.get("review_rounds", REVIEW_ROUNDS))
        margin = float(spec.settings.get("tuning_margin", 0.001))
        relative = bool(spec.settings.get("tuning_margin_relative", False))
        ranking = list(screened)
        names = list(p.class_labels) if kind == "multiclass" and p.class_labels else None
        cap = target_cap(bundle.y_train) if kind == "regression" else None
        entities = [c for c in ([p.solution.group_column] if p.solution.group_column else []) + list(p.solution.identifiers)
                    if c in bundle.X_train and bundle.X_train[c].nunique() < 0.5 * len(bundle.X_train)]
        # only what a model may learn from goes into the segment analysis (never a forbidden, time or text column), and the entities
        safe = [c for c in bundle.X_train.columns if p.roles.get(c) == "feature" or c in entities]
        current, params, tried, rounds = selected, {}, [], []
        last = None
        for number in range(1, rounds_allowed + 2):  # one more review than rounds: what is left after the last round
            index = current["_oof_index"]
            y = bundle.y_train.to_numpy()[index]
            if names:
                y = np.array([names[int(v)] if 0 <= int(v) < len(names) else v for v in y])
            last = review_of(kind, y, current["_oof_pred"], metric=metric, frame=bundle.X_train.iloc[index][safe].reset_index(drop=True),
                             cap=cap, entities=entities, classes=names, tried=tried)
            if number > rounds_allowed:
                break
            reasons = {e.name: (last.hypotheses[e.hypothesis].statement, last.hypotheses[e.hypothesis].cites) for e in last.next_experiments}
            ranked = [e.name for e in runnable(last.next_experiments)]
            critic = None
            if p.critic is not None:  # 18.6: the critic's ranking, each hypothesis checked against the measured weaknesses
                critic = critique(p.critic, last, kind, metric=metric, model=family, score=rn._mean(current, metric), tried=tried)
                for k in critic["kept"]:  # the critic may add a catalogue experiment the rules did not list, citing what it fixes
                    if CATALOGUE[k["experiment"]].runnable and k["experiment"] not in ranked:
                        ranked.append(k["experiment"])
                    # the claim says what the catalogue says the experiment does; the critic's own words stay in the record only
                    gain = CATALOGUE[k["experiment"]].gain
                    reasons.setdefault(k["experiment"], (f"{gain[:1].upper()}{gain[1:]}.", k["cites"]))
                position = {name: i for i, name in enumerate(critic["order"])}
                ranked = sorted(ranked, key=lambda name: position.get(name, len(position)))
            pick = None
            for e in ranked:
                for candidate in [family] + [f for f in ranking if f != family]:
                    change = rn.change_params(CATALOGUE[e].change, candidate, bundle.task_type)
                    if change is not None:
                        pick = (e, candidate, change)
                        break
                if pick:
                    break
            if pick is None:
                break
            name, candidate, change = pick
            tried.append(name)
            # another family keeps only the changes every family makes (the "_" ones), never the leader's own parameters
            trial = {**(params if candidate == family else {k: v for k, v in params.items() if k.startswith("review__")}), **change}
            try:
                row = rn.cross_validate(bundle, recipe, candidate, trial, config_id=f"review-{name}")
            except Exception as exc:  # noqa: BLE001 - this change cannot run here; the rounds before it stand
                rounds.append({"round": number, "experiment": name, "family": candidate, "base_family": family,
                               "adopted": False, "why": f"could not run: {type(exc).__name__}"})
                break
            base_mean = rn._mean(current, metric)
            needed = abs(base_mean) * margin if relative else margin
            decision = adopt([f[metric] for f in current["fold_metrics"]], [f[metric] for f in row["fold_metrics"]], higher, needed)
            if candidate != family and decision["adopted"]:  # the change itself must help that family, not only the switch
                own = adopt([f[metric] for f in screened[candidate]["fold_metrics"]], [f[metric] for f in row["fold_metrics"]], higher,
                            abs(rn._mean(screened[candidate], metric)) * margin if relative else margin)
                decision = {**decision, "adopted": own["adopted"], "own_family_gain": own,
                            "why": decision["why"] + f"; against {candidate}'s own screen: " + own["why"]}
            statement, cites = reasons[name]
            rounds.append({"round": number, "experiment": name, "hypothesis": statement, "cites": cites,
                           "family": candidate, "base_family": family, "change": trial, "base_mean": base_mean,
                           "variant_mean": rn._mean(row, metric), "margin": needed, **decision,
                           **({"critic": {**{k: critic[k] for k in ("source", "reason", "order", "refused")},
                                          "statement": next((k["statement"] for k in critic["kept"] if k["experiment"] == name), None)}}
                              if critic else {})})
            if decision["adopted"]:
                current, params, family = row, trial, candidate
        return {"family": family, "rounds": rounds, "adopted_params": params, "review": last.model_dump(mode="json") if last else None,
                "selected_mean": rn._mean(current, metric), "selected_std": rn._std(current, metric),
                "rule": ("dclab_core.review reads the leader's out-of-fold errors on the training rows; each round cross-validates one "
                         "proposed change on the same folds and keeps it only if its paired gain beats the tuning margin and the one-sided "
                         f"90% t bound of the fold differences; at most {rounds_allowed} rounds; the holdout is not touched.")}
    except Exception as exc:  # noqa: BLE001 - a review the data does not allow leaves the stage as it was
        from ..observe import log

        log.warning("review rounds unavailable: %s", type(exc).__name__)
        return None


def _decision(p: Prepared, selected: dict[str, Any], y_test: pd.Series, predictions: np.ndarray) -> dict[str, Any] | None:
    """The decision on top of the model (package 16.9), when the solution states one: a review budget from the value
    of a case and the cost of a review, chosen on the selected model's out-of-fold scores and read once on the
    holdout; or a price from a model of acceptance by price, fitted and chosen on the training rows only (the
    holdout's labels are not read). None when no decision is stated; a refusal with its reason when one cannot be made."""
    decision = p.solution.decision
    if decision is None:
        return None
    from dclab_core import decide

    bundle = p.bundle
    try:
        if decision.action == "review":
            index = selected["_oof_index"]
            oof_y = bundle.y_train.to_numpy()[index]
            group = p.solution.group_column
            groups = bundle.X_train[group].to_numpy()[index] if group and group in bundle.X_train else None
            rows_per_day = None
            time_column = p.solution.time_column
            if time_column and time_column in bundle.X_train.columns:
                stamps = pd.to_datetime(bundle.X_train[time_column], errors="coerce").dropna()
                days = (stamps.max() - stamps.min()).total_seconds() / 86400 if len(stamps) > 1 else 0
                rows_per_day = len(bundle.X_train) / days if days >= 1 else None
            return decide.review_budget(oof_y, selected["_oof_pred"], value_per_case=decision.value_per_case,
                                        cost_per_review=decision.cost_per_review, groups=groups, partial=p.solution.labels == "partial",
                                        rows_per_day=rows_per_day, y_test=y_test.to_numpy(), score_test=predictions)
        price = decision.price_column
        inputs = [c for c in (price, decision.reference_column) if c]
        refused = [c for c in inputs if p.roles.get(c) != "feature"]
        if decision.unit_value_column and p.roles.get(decision.unit_value_column) == "blocked":
            refused.append(decision.unit_value_column)
        if refused:  # the acceptance model reads only what is known when the offer is made
            return {"decision": "price", "learnable": False,
                    "warning": "The decision reads " + ", ".join(f"`{c}` (its role: {p.roles.get(c)})" for c in refused)
                               + ": a price, its reference and the value of an offer must be known when the offer is made, so none "
                                 "may be forbidden or set aside"}
        X, y = bundle.X_train, bundle.y_train.to_numpy()
        reference = X[decision.reference_column].to_numpy(dtype=float) if decision.reference_column else None
        skip = {price, decision.reference_column, decision.unit_value_column, *decision.test_arm_columns}
        context = X[[c for c, role in p.roles.items() if role == "feature" and c not in skip and c in X]]
        group = p.solution.group_column
        groups = X[group].to_numpy() if group and group in X else None
        unit_value = None
        if decision.side == "pay":
            unit_value = (X[decision.unit_value_column].to_numpy(dtype=float) if decision.unit_value_column
                          else decision.unit_value_factor * reference)
        out = decide.price_curve(X[price].to_numpy(dtype=float), y, reference=reference, context=context,
                                 unit_cost=decision.unit_cost, unit_value=unit_value, groups=groups)
        out["holdout"] = "not read: the price is chosen on the training rows, and the holdout keeps no counterfactual prices"
        return out
    except Exception as exc:  # noqa: BLE001 - a decision the data does not allow leaves the stage standing, said plainly
        from ..observe import log

        log.warning("decision unavailable: %s", type(exc).__name__)
        return {"decision": decision.action, "learnable": False, "warning": f"The decision could not be computed ({type(exc).__name__})"}


def _decision_claim(eid: str, d: dict[str, Any]) -> dict[str, Any]:
    """The decision in one sentence, every number from evidence.decision (16.9)."""
    paths = ["evidence.decision"]
    if not d.get("learnable", True) or d.get("warning"):
        return _claim(f"{eid}-D1", "risk", str(d.get("warning") or "No decision could be made from this data."), paths)
    if d.get("decision") == "review_budget":
        if d.get("note"):  # no review pays: said first, never dressed as a recommendation
            return _claim(f"{eid}-D1", "risk", d["note"], paths, [d["assumption"]])
        lo, hi = d["expected_value_per_row_interval"]
        text = (f"Reviewing the riskiest {d['share']:.2%} of rows maximises expected value at {d['expected_value_per_row']} per row "
                f"(90% interval {lo} to {hi}), for {d['value_per_case']:g} per case found and {d['cost_per_review']:g} per review; "
                f"chosen on out-of-fold scores.")
        if d.get("holdout"):
            text += f" On the holdout it found {d['holdout']['cases_found']} cases in {d['holdout']['reviews']} reviews."
        return _claim(f"{eid}-D1", "decision", text, paths, [d["assumption"]] + ([d["partial_note"]] if d.get("partial_note") else []))
    lo, hi = d["best_multiple_interval"]
    text = (f"The best price is {d['best_multiple']} times the reference (90% interval {lo} to {hi}), with an expected margin of "
            f"{d['expected_margin_per_offer']} per offer against {d['margin_per_offer_as_priced']} as priced; acceptance at it "
            f"{d['acceptance_at_best']:.0%}; chosen on out-of-fold acceptance predictions of the training rows.")
    return _claim(f"{eid}-D1", "decision", text, paths, [d["assumption"]] + ([d["note"]] if d.get("note") else []))


def _task_assessment(p: Prepared, task: str, selected: dict[str, Any], y_test: pd.Series, predictions: np.ndarray) -> dict[str, Any] | None:
    """dclab_core's checks for this task (package 17.5): the alert threshold, calibration, per-class results and
    confusions, or intervals and error by range and segment; chosen on the training rows' out-of-fold predictions and
    read once on the holdout. None, logged, when they cannot be computed: the stage stands without them."""
    try:
        from dclab_core.analyse import target_cap
        from dclab_core.assess import assess, threshold_inputs

        bundle = p.bundle
        kind = p.kind.base_task
        oof_y = bundle.y_train.to_numpy()[selected["_oof_index"]]
        classes = labels = None
        if kind == "multiclass" and p.class_labels:
            names = p.class_labels if isinstance(p.class_labels, (list, tuple)) else [p.class_labels[i] for i in sorted(p.class_labels)]
            classes = list(names)
            to_label = lambda values: [names[int(v)] if 0 <= int(v) < len(names) else v for v in values]  # noqa: E731
            oof_y, labels = to_label(oof_y), to_label(y_test.to_numpy())
        segments = cap = None
        if kind == "regression":  # the columns and the cap are chosen on the training rows
            cats = [c for c in bundle.X_train.columns if not pd.api.types.is_numeric_dtype(bundle.X_train[c]) and 2 <= bundle.X_train[c].nunique() <= 20]
            segments = {c: bundle.X_test[c].tolist() for c in cats[:3]} or None
            cap = target_cap(bundle.y_train)
        counts = pd.Series(oof_y).value_counts()
        tiny = [c for c, n in counts.items() if n < 30 or n / counts.sum() < 0.01] if kind == "multiclass" else None
        span_days = None
        time_column = p.solution.time_column
        if time_column and time_column in bundle.X_train.columns:
            stamps = pd.to_datetime(bundle.X_train[time_column], errors="coerce").dropna()
            span_days = (stamps.max() - stamps.min()).total_seconds() / 86400 if len(stamps) > 1 else None
        inputs = threshold_inputs(p.plan, len(bundle.y_train), span_days)  # the plan's threshold policy and label coverage
        inputs["partial_labels"] = inputs["partial_labels"] or p.solution.labels == "partial"  # the solution says so (15.2)
        if inputs["budget"] is None and p.solution.alert_budget:
            inputs["budget"] = p.solution.alert_budget  # the solution's review budget (15.3)
        out = assess(kind, oof_y, selected["_oof_pred"], labels if labels is not None else y_test.to_numpy(), predictions,
                     budget=inputs["budget"], costs=inputs["costs"], partial_labels=inputs["partial_labels"], classes=classes,
                     tiny_classes=tiny, segments=segments, cap=cap)
        if kind == "binary" and inputs["note"]:
            out["threshold"]["plan_note"] = inputs["note"]
        return out
    except Exception as exc:  # noqa: BLE001 - an assessment the data does not allow leaves the stage as it was
        from ..observe import log

        log.warning("task assessment unavailable: %s", type(exc).__name__)
        return None


def stage_final(p: Prepared, eid: str, recipe: str, family: str, reviewed: dict[str, Any] | None = None) -> tuple[dict, list, str]:
    """``reviewed``: the change the models stage's review kept (17.6), applied when the family is the one it was kept for."""
    bundle, spec, task = p.bundle, p.bundle.spec, p.bundle.task_type
    metric, higher = spec.primary_metric, spec.higher_is_better
    n_classes = rn._n_classes(bundle)
    kept = dict(reviewed or {})
    baseline = rn.cross_validate(bundle, recipe, family, kept or None, config_id=f"{family}_baseline")
    candidates = rn.optimization_candidates(family, task)
    if "review__class_weight" in kept:  # weights kept by the review: a candidate's own class weights would apply them twice
        candidates = [{**c, "params": {k: v for k, v in c["params"].items() if k != "model__class_weight"}} for c in candidates]
    tuned = [rn.cross_validate(bundle, recipe, family, {**c["params"], **kept}, config_id=c["candidate_id"]) for c in candidates]
    margin = float(spec.settings.get("tuning_margin", 0.001))
    relative = bool(spec.settings.get("tuning_margin_relative", False))
    decision = rn.tuning_decision(baseline, tuned, metric, higher, margin, relative=relative)
    selected = decision.pop("selected")
    decision["selected_config"] = selected["config_id"]
    started = time.perf_counter()
    final = rn.fit_and_predict(bundle, recipe, family, selected["model_params"] or None, bundle.X_train, bundle.y_train, bundle.X_test)
    final_seconds = time.perf_counter() - started
    predictions, y_test = final["predictions"], bundle.y_test
    holdout = rn.compute_metrics(task, y_test, predictions, n_classes=n_classes, detail=True, precision_target=float(spec.settings.get("precision_target", 0.9)),
                                 alert_budget=spec.settings.get("alert_budget"))
    holdout["fit_seconds"] = float(final_seconds)
    if task == "multiclass" and p.class_labels:
        for item in holdout.get("worst_3_classes", []):
            item["source_label"] = p.class_labels[item["class_index"]]
    metric_fn = rn.budget_metric_fn(spec.settings["alert_budget"]) if metric == "recall_at_budget" else rn.primary_metric_fn(task, metric, n_classes)
    clustered = spec.split_strategy in ("group", "time_group")
    ci = rn.bootstrap_ci(y_test.to_numpy(), predictions, metric_fn, groups=bundle.groups("test") if clustered else None)
    if spec.settings.get("alert_budget") and metric != "recall_at_budget":  # 15.3: recall at the budget, with its interval too
        holdout["recall_at_budget_ci"] = rn.bootstrap_ci(y_test.to_numpy(), predictions, rn.budget_metric_fn(spec.settings["alert_budget"]),
                                                         groups=bundle.groups("test") if clustered else None)
    baselines = _baselines(p, predictions)
    importance = rn._importance(final["model"], final["names"])
    top_features = [{"feature": k, "normalized_importance": v} for k, v in sorted(importance.items(), key=lambda kv: -kv[1])[:20]]
    cv_mean, holdout_value = rn._mean(selected, metric), holdout[metric]
    evidence: dict[str, Any] = {
        "feature_recipe": recipe, "model": family, "configuration_results": [baseline, *tuned], "optimization_search_space": candidates,
        "tuning_decision": decision, "selected_optimization": selected["config_id"], "selected_model_params": selected["model_params"],
        "review_change": kept,
        "selection_rule": (f"Evaluate {len(candidates)} explicit parameter candidates on training CV ({bundle.cv_description}); take the best by mean "
                           f"{'-' if higher else '+'} 0.25×std and accept it only if its mean {_label(metric)} beats the default by at least "
                           + (f"{margin:.0%} (relative)" if relative else f"{margin}") + ", otherwise keep the default configuration."),
        "holdout_consumed": True, "holdout_split": bundle.holdout_description, "holdout_rows": int(len(y_test)), "train_rows": int(len(bundle.y_train)),
        "holdout_metrics": holdout, "holdout_primary_metric_ci": ci, "cv_selected_metric_mean": cv_mean,
        # a holdout without a case of each class cannot measure a ranking metric: said, never a crash
        "cv_to_holdout_gap": rn.oriented_gain(holdout_value, cv_mean, higher) if holdout_value is not None else None,
        "baselines": baselines, "top_final_features": top_features,
        "sampling": bundle.sampling, "production_approved": False,
        "production_feature_rule": "Promote a feature only if it is decision-time available, stable across folds, obtainable in production and monitored for drift.",
    }
    if task in ("binary_imbalanced", "text_tabular_binary"):
        rows = []
        y_arr = y_test.to_numpy()
        for t in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
            flags = predictions >= t
            tp = int((flags & (y_arr == 1)).sum())
            rows.append({"threshold": t, "alerts": int(flags.sum()), "true_positives": tp,
                         "precision": float(tp / flags.sum()) if flags.sum() else None, "recall": float(tp / max(int(y_arr.sum()), 1))})
        evidence["threshold_analysis"] = rows
        bins = []
        for lower in np.arange(0, 1, 0.2):
            mask = (predictions >= lower) & ((predictions < lower + 0.2) if lower < 0.8 else (predictions <= 1))
            if mask.any():
                bins.append({"lower": float(lower), "upper": float(lower + 0.2), "count": int(mask.sum()), "predicted": float(predictions[mask].mean()), "observed": float(y_arr[mask].mean())})
        evidence["calibration_bins"] = bins
    limits = []
    if p.solution.labels == "partial":
        evidence["precision_note"] = ("The labels are partial (" + (p.solution.label_provenance or "a sample of known cases") + "): an alert "
                                      "without a label may be a case nobody found, so every precision here is a lower bound.")
    info = p.label_info or {}
    if info.get("disagreeing_columns"):
        limits.append("Columns with the same name hold different values in the label file and the table ("
                      + ", ".join(f"{c} {share:.0%} equal" for c, share in list(info["disagreeing_columns"].items())[:6])
                      + "): their meaning in the label file is unknown, so they were never taken from it.")
    if info.get("not_found"):
        limits.append(f"{info['not_found']} label rows are not in the table: they could not be scored, and say the table may miss cases.")
    if limits:
        evidence["limits"] = limits
    assessment = _task_assessment(p, task, selected, y_test, predictions)
    if assessment is not None:
        evidence["task_assessment"] = assessment  # 17.5: thresholds and intervals chosen out of fold, read on the holdout
    chosen_action = _decision(p, selected, y_test, predictions)
    if chosen_action is not None:
        evidence["decision"] = chosen_action  # 16.9: the action on top of the model, by expected value
    label = _label(metric)
    claims = [
        _claim(f"{eid}-C1", "fact", f"{family}/{selected['config_id']} on the `{recipe}` recipe scored {label} {rn._f(holdout_value)} on the once-consumed holdout "
               f"({bundle.holdout_description}; {ci['method']} 95% interval {rn._f(ci['low'])}–{rn._f(ci['high'])}); training-CV mean was {rn._f(cv_mean)}.",
               ["evidence.holdout_metrics", "evidence.holdout_primary_metric_ci", "evidence.holdout_consumed", "evidence.cv_selected_metric_mean"],
               ["A single holdout is one draw; CV models see fewer rows than the final model, so the holdout can legitimately differ from the CV mean."]),
        _claim(f"{eid}-C2", "decision", f"Tuning {'accepted' if decision['accepted'] else 'rejected'}: best candidate {decision['best_candidate']} changed mean training-CV {label} by "
               f"{decision['gain']:+.4f} (required ≥ {decision['required_gain']:.4f}), so {selected['config_id']} was used.",
               ["evidence.tuning_decision", "evidence.configuration_results", "evidence.selection_rule"],
               ["Only a handful of explicit candidates were tried; this is a sanity check, not an exhaustive search."]),
    ]
    base_text = "; ".join(f"{k}: " + ", ".join(f"{m} {rn._f(v)}" for m, v in b.items() if isinstance(v, (int, float))) for k, b in baselines.items())
    claims.append(_claim(f"{eid}-C3", "fact", f"Trivial baselines on the same holdout: {base_text}.", ["evidence.baselines"]))
    if chosen_action is not None:
        claims.append(_decision_claim(eid, chosen_action))
    summary = (f"Tried {len(candidates)} explicit {family} parameter candidates on {bundle.cv_description} (tuning {'accepted' if decision['accepted'] else 'rejected'}), "
               f"fitted {selected['config_id']} on all {len(bundle.y_train)} training rows and scored the sealed holdout once: {label} {rn._f(holdout_value)} "
               f"({rn._f(ci['low'])}–{rn._f(ci['high'])}).")
    return evidence, claims, summary


def _baselines(p: Prepared, predictions: np.ndarray) -> dict[str, Any]:
    bundle, task = p.bundle, p.bundle.task_type
    y_test, y_train = bundle.y_test, bundle.y_train
    if task in ("binary_imbalanced", "text_tabular_binary"):
        rate = float(y_test.mean())
        return {"majority_class": {"accuracy": float(max(rate, 1 - rate)), "roc_auc": 0.5, "average_precision_random_ranking": rate}}
    if task == "multiclass":
        from sklearn.metrics import f1_score

        majority = int(y_train.value_counts().idxmax())
        n_classes = rn._n_classes(bundle)
        return {"majority_class": {"macro_f1": float(f1_score(y_test, np.full(len(y_test), majority), average="macro", labels=list(range(n_classes)), zero_division=0)),
                                   "accuracy": float((y_test == majority).mean())}}
    out = {}
    for name, value in (("training_mean", float(y_train.mean())), ("training_median", float(y_train.median()))):
        out[name] = rn.regression_metrics(y_test, np.full(len(y_test), value))
    return out


# ---------------------------------------------------------------------- orchestration


def chosen(record: dict[str, Any] | None, fallback_key: str) -> str | None:
    if not record:
        return None
    decision = record.get("decision") or {}
    return decision.get("chosen") or record["evidence"].get(fallback_key)


def execute(store: ProjectStore, project_id: str, stage: str, actor: str = "human", reuse_reason: str | None = None) -> dict[str, Any]:
    """Run one stage now (blocking) and write its record.

    The move is validated by the workflow graph first and logged either way; a move the
    graph does not allow raises ``graph.GraphBlocked`` with the verdict.
    """
    project = store.get(project_id)
    verdict = graph.check(project, "run_stage", actor, stage=stage, reuse_reason=reuse_reason)
    if not verdict.allowed:
        graph.log(store, project_id, verdict, project)
        raise graph.GraphBlocked(verdict)
    state_before = copy.deepcopy(project)  # the state the move was proposed in, for the log
    index = STAGE_KEYS.index(stage)
    later = [s for s in STAGE_KEYS[index + 1:] if project["stages"][s].get("status") in ("completed", "approved")]
    if stage != "final" and project["stages"][stage].get("status") in ("completed", "approved") and later:
        store.clear_stages(project_id, STAGE_KEYS[index + 1])  # a rerun makes every later result stale
        project = store.get(project_id)
    used = graph.consume_holdout_approval(project) if stage == "final" and actor == "agent" else None
    try:
        record = _execute(store, project_id, stage, project, reuse_reason)
    except Exception as exc:
        graph.log(store, project_id, verdict, state_before, outcome=f"failed: {type(exc).__name__}: {str(exc)[:200]}")
        _audit_use(store, project_id, used)  # only when the saved project shows the approval used (a failed check saves nothing)
        raise
    graph.log(store, project_id, verdict, state_before, outcome="done: " + record["setup_summary"][:200])
    _audit_use(store, project_id, used)
    return record


def _audit_use(store: ProjectStore, project_id: str, used: dict[str, Any] | None) -> None:
    """The audit row for a holdout approval the final stage used, written once the project saved it as used."""
    if used is None:
        return
    try:
        saved = store.get(project_id)
    except KeyError:
        return
    if not any(a.get("gate") == "holdout" and a.get("at") == used.get("at") and a.get("used") == used.get("used") for a in saved.get("approvals") or []):
        return
    from .. import audit

    audit.record_for(store, "approval_used", "agent", project=saved, move="use_approval", status="allowed", at=used["used"],
                     args={"gate": "holdout"}, message="The final stage used this approval.", rules=["DCLAB-R17"], approved_at=used.get("at"))


def _execute(store: ProjectStore, project_id: str, stage: str, project: dict[str, Any], reuse_reason: str | None) -> dict[str, Any]:
    from ..models.gateway import private_scope

    with private_scope(bool((project.get("settings") or {}).get("private"))):  # 16.13: private data never reaches a model
        return _execute_stage(store, project_id, stage, project, reuse_reason)


def _execute_stage(store: ProjectStore, project_id: str, stage: str, project: dict[str, Any], reuse_reason: str | None) -> dict[str, Any]:
    p = prepare(store, project)
    meta = STAGE_BY_KEY[stage]
    eid = f"PRJ-{project_id[:6]}-{stage}"
    project["stages"][stage] = {"status": "running", "started": _now()}
    project["running"] = stage
    store.save(project)
    store.log(project_id, "stage_started", {"stage": stage})
    clock = time.perf_counter()
    try:
        decision: dict[str, Any] | None = None
        if stage == "data":
            evidence, claims, summary = stage_data(p, eid)
        elif stage == "leakage":
            evidence, claims, summary = stage_leakage(p, eid)
        elif stage == "features":
            evidence, claims, summary = stage_features(p, eid)
            decision = {"kind": "recipe", "selected": evidence["selected_recipe"], "chosen": evidence["selected_recipe"], "rule": evidence["selection_rule"],
                        "options": [{"id": r["recipe"], "label": r["recipe"], "description": r["description"], "features": r["feature_count_mean"],
                                     "mean": rn._mean(r, p.metric), "std": rn._std(r, p.metric), "elapsed_seconds": r["elapsed_seconds"]} for r in evidence["stage_results"]]}
        elif stage == "models":
            recipe = chosen(store.read_stage(project_id, "features"), "selected_recipe")
            evidence, claims, summary = stage_models(p, eid, recipe)
            decision = {"kind": "model", "selected": evidence["selected_model"], "chosen": evidence["selected_model"], "rule": evidence["selection_rule"],
                        "options": [{"id": r["model"], "label": r["model"], "mean": r["mean"], "std": r["std"], "adjusted": r["adjusted_score"],
                                     "elapsed_seconds": r["elapsed_seconds"], "features": r["feature_count_mean"]} for r in evidence["ranking"]]}
        else:
            recipe = chosen(store.read_stage(project_id, "features"), "selected_recipe")
            models = store.read_stage(project_id, "models")
            family = chosen(models, "selected_model")
            review = ((models or {}).get("evidence") or {}).get("review") or {}
            reviewed = review.get("adopted_params") if review.get("family") == family else None  # kept for this family only
            project["holdout_uses"] = int(project.get("holdout_uses", 0)) + 1
            evidence, claims, summary = stage_final(p, eid, recipe, family, reviewed)
            evidence["holdout_uses_in_this_project"] = project["holdout_uses"]
            if reuse_reason:
                evidence["holdout_reuse_reason"] = str(reuse_reason)[:500]
            decision = {"kind": "final", "selected": evidence["selected_optimization"], "chosen": evidence["selected_optimization"], "rule": evidence["selection_rule"], "options": []}
        elapsed = time.perf_counter() - clock
        evidence["elapsed_seconds"] = float(elapsed)
        record = {
            "stage": stage, "kind": meta["kind"], "title": meta["title"], "workflow": meta["workflow"], "question": meta["question"],
            "experiment_id": eid, "status": "completed", "started_at": project["stages"][stage]["started"], "completed_at": _now(),
            "elapsed_seconds": float(elapsed), "task": p.task, "task_type": p.bundle.task_type, "primary_metric": p.metric,
            "decision_time_rule": p.solution.prediction_moment, "holdout_policy": p.bundle.spec.holdout_policy(),
            "setup_summary": summary, "evidence": evidence, "claims": claims, "decision": decision,
            "labels": {"coverage": p.solution.labels, "provenance": p.solution.label_provenance,
                       "file": (project.get("labels") or {}).get("filename")},
            "provenance": {"data_sha256": project["data"]["sha256"], "filename": project["data"]["filename"], "sampling": p.sampling,
                           "cv": p.bundle.cv_description, "holdout": p.bundle.holdout_description, "random_state": ds.RANDOM_STATE},
        }
        record = rn._jsonable(record)
        record["notes"] = agent.narrate(stage, record, p.solution.model_dump(), p.bundle.task_type, project_id)
        explanation = agent.explain_stage(record, project_id)  # a model's words from the record, checked by code (A3.3)
        if explanation:
            record["explanation"] = explanation
        store.write_stage(project_id, stage, record)
        project = store.get(project_id)
        project["stages"][stage] = {"status": "completed", "started": record["started_at"], "finished": record["completed_at"], "elapsed_seconds": float(elapsed)}
        if stage == "final":
            project["holdout_uses"] = evidence["holdout_uses_in_this_project"]
        project["running"] = None
        store.save(project)
        store.log(project_id, "stage_completed", {"stage": stage, "elapsed_seconds": round(elapsed, 1), "summary": summary[:300]})
        return record
    except Exception as exc:  # noqa: BLE001 — the failure itself is the record
        project = store.get(project_id)
        project["stages"][stage] = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:600]}", "finished": _now()}
        project["running"] = None
        store.save(project)
        store.log(project_id, "stage_failed", {"stage": stage, "error": project["stages"][stage]["error"]})
        raise


def run_all(store: ProjectStore, project_id: str, start: str = "data") -> list[str]:
    """Run every remaining stage in order; stops at the first failure."""
    done = []
    for stage in STAGE_KEYS[STAGE_KEYS.index(start):]:
        execute(store, project_id, stage)
        done.append(stage)
    return done


def approve(store: ProjectStore, project_id: str, stage: str, choice: str | None = None, actor: str = "human") -> dict[str, Any]:
    """Sign-off on a stage, optionally overriding the deterministic choice; downstream results are cleared."""
    project = store.get(project_id)
    verdict = graph.check(project, "approve_stage", actor, stage=stage, choice=choice)
    graph.log(store, project_id, verdict, project)
    if not verdict.allowed:
        raise graph.GraphBlocked(verdict)
    record = store.read_stage(project_id, stage)
    if not record or project["stages"][stage].get("status") not in ("completed", "approved"):
        raise ValueError("Only a completed stage can be approved")
    if choice:
        options = {o["id"] for o in (record.get("decision") or {}).get("options", [])}
        if choice not in options:
            raise ValueError("Unknown option")
        record["decision"]["chosen"] = choice
        record["decision"]["overridden"] = choice != record["decision"]["selected"]
        if record["decision"]["overridden"]:
            record.pop("explanation", None)  # a model's words about the rule's selection would sit beside a different decision
        store.write_stage(project_id, stage, record)
        project["decisions"][stage] = choice
    index = STAGE_KEYS.index(stage)
    for later in STAGE_KEYS[index + 1:]:
        if project["stages"][later].get("status") != "pending":
            store.save(project)
            store.clear_stages(project_id, later)
            project = store.get(project_id)
            break
    project["stages"][stage]["status"] = "approved"
    project["stages"][stage]["approved_at"] = _now()
    from . import memory
    memory.stage_approved(project, stage, (record or {}).get("decision"), choice, actor)  # A5.2: kept as a note the agents read
    store.save(project)
    store.log(project_id, "stage_approved", {"stage": stage, "choice": choice})
    return project


def capabilities() -> dict[str, Any]:
    return {
        "tasks": list(kinds.BASE_TASKS),
        "kinds": [k.key for k in kinds.all_kinds()],
        "text_columns": "binary targets only",
        "split_strategies": ["time (declared time column)", "group (declared group column)", "stratified random"],
        "model_families": {t: rn.model_families(tt) for t, tt in TASK_TYPE.items()},
        "lightgbm": rn._have("lightgbm"), "xgboost": rn._have("xgboost"),
        "llm": agent.llm_available(),
    }
