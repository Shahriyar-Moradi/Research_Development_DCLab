"""Category-codes campaign: what encoding factorized category codes as categories changes (rule DCLAB-R11).

The cached UCI tables store categories as factorized codes (bank_marketing: May=0, Jun=1, ...).
Commit 1cf5ae5 kept the codes out of every log, ratio and product feature, and an informal re-check
then showed bank_marketing ``ratios`` and german_credit ``interactions`` losing their lift: trees had
reached real month/poutcome signal through arithmetic on codes. This campaign measures, on identical
training folds only (no locked holdout is read), three treatments of each dataset's declared code columns:

legacy   codes are numbers and feed derived features (the model_building_50_v1 behaviour)
numbers  codes are numbers and feed no derived feature (DCLAB-R11 since 1cf5ae5)
one_hot  codes are one-hot encoded inside each fit fold and feed no derived feature

CAT-001  the playbook ladder (FeatureEngineer + LightGBM, the model_building_50_v1 protocol) and a
         five-family screen on the raw matrix, 3x3 repeated stratified folds; the first repeat is the
         campaign's own folds, so the legacy arm must reproduce its recorded stage results exactly
CAT-002  the DCLab notebook: the recipes its engine registers for each sample, on the engine's own folds

Results are written to ``evidence/campaigns/category_codes_v1/`` and never overwritten without --force.

CLI::

    python -m dclab_rnd.code_encoding run [--only CAT-001] [--force]
    python -m dclab_rnd.code_encoding report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "category_codes_v1"
OUT = ROOT / "evidence/campaigns" / CAMPAIGN_ID
RESULTS = OUT / "results"
RANDOM_STATE = 42
# Every public dataset whose catalog entry declares category codes (dclab_rnd/agentic/catalog.py).
DATASETS = ("bank_marketing", "german_credit", "adult", "credit_default", "heart_disease", "online_shoppers", "mushroom")
MAX_ROWS = 4000  # the model_building_50_v1 protocol: 4,000 rows, 80% training rows, 3-fold CV
FOLDS, REPEATS = 3, 3
LADDER_TOLERANCE = 0.002  # model_building_50_v1 feature-stage rule
ARMS = ("legacy", "numbers", "one_hot")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max())}


def paired(new: list[float], old: list[float], test_train_ratio: float) -> dict[str, Any]:
    """Fold-paired difference ``new - old`` with a naive and a Nadeau-Bengio corrected 95% t-interval.

    Cross-validation folds share training rows, so the naive interval is too narrow; the corrected
    variance ``s^2 (1/n + n_test/n_train)`` (Nadeau and Bengio, 2003) is the one to read.
    """
    from scipy import stats

    d = np.asarray(new, dtype=float) - np.asarray(old, dtype=float)
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else 0.0
    q = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    naive = q * sd / math.sqrt(n) if n > 1 else float("nan")
    corrected = q * sd * math.sqrt(1.0 / n + test_train_ratio) if n > 1 else float("nan")
    return {"mean": mean, "std": sd, "folds": n, "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
            "naive_ci95": [mean - naive, mean + naive], "corrected_ci95": [mean - corrected, mean + corrected],
            "differences": [float(v) for v in d]}


def _verdict(diff: dict[str, Any]) -> str:
    low, high = diff["corrected_ci95"]
    return "better" if low > 0 else "worse" if high < 0 else "within noise"


# --------------------------------------------------------------------------- CAT-001


def _ladder_row(row: dict[str, Any]) -> dict[str, Any]:
    folds = [m["roc_auc"] for m in row["fold_metrics"]]
    return {"stage": row["stage"], "model": row["model"], "feature_count_mean": row["feature_count_mean"],
            "roc_auc": row["metrics"]["roc_auc"], "log_loss": row["metrics"]["log_loss"],
            "fold_roc_auc": folds, "first_repeat_roc_auc_mean": float(np.mean(folds[:FOLDS])),
            "top_features": [f["feature"] for f in row["feature_stability"][:8]],
            "elapsed_seconds": row["elapsed_seconds"]}


def _select_stage(rows: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    """The model_building_50_v1 rule: smallest feature matrix within 0.002 mean ROC-AUC of the best stage."""
    means = {stage: (row["roc_auc"]["mean"] if key == "all" else row["first_repeat_roc_auc_mean"]) for stage, row in rows.items()}
    best = max(means.values())
    eligible = [s for s, m in means.items() if m >= best - LADDER_TOLERANCE]
    selected = min(eligible, key=lambda s: rows[s]["feature_count_mean"])
    return {"selected_stage": selected, "selected_roc_auc_mean": means[selected],
            "best_stage": max(means, key=means.get), "best_roc_auc_mean": best}


def _campaign_record(key: str) -> tuple[str, dict[str, float]]:
    path = next((ROOT / "evidence/campaigns/model_building_50_v1/results").glob(f"EXP-*_{key}_feature_engineering.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    return record["experiment_id"], {r["stage"]: r["metrics"]["roc_auc"]["mean"] for r in record["evidence"]["stage_results"]}


def cat_playbook_ladder() -> dict[str, Any]:
    from dclab_rnd import science

    datasets, used_paths = [], []
    for key in DATASETS:
        bundle = science.load_dataset(ROOT, key, max_rows=MAX_ROWS)
        used_paths += bundle["data_paths"]
        X = science.safe_features(bundle["X_train"], bundle["policy"])  # training rows only; X_test is never read
        y = bundle["y_train"]
        codes = [c for c in (bundle["categorical"] or []) if c in X.columns]
        settings = {"legacy": ([], False), "numbers": (codes, False), "one_hot": (codes, True)}
        ladder: dict[str, dict[str, Any]] = {}
        for arm, (categorical, one_hot) in settings.items():
            ladder[arm] = {
                stage: _ladder_row(science.evaluate_cv(X, y, stage=stage, model_name="lightgbm", folds=FOLDS, repeats=REPEATS,
                                                       categorical=categorical, one_hot_codes=one_hot))
                for stage in science.FEATURE_STAGES
            }
            print(f"  CAT-001 {key} {arm}: " + ", ".join(f"{s} {r['roc_auc']['mean']:.4f}" for s, r in ladder[arm].items()), flush=True)
        screen: dict[str, dict[str, Any]] = {}
        for arm in ("numbers", "one_hot"):  # on the raw stage legacy and numbers build the same matrix
            categorical, one_hot = settings[arm]
            screen[arm] = {model: _ladder_row(science.evaluate_cv(X, y, stage="raw", model_name=model, folds=FOLDS, repeats=REPEATS,
                                                                  categorical=categorical, one_hot_codes=one_hot))
                           for model in science.MODEL_FAMILIES}
        exp_id, recorded = _campaign_record(key)
        reproduction = {stage: {"recorded": recorded[stage], "legacy_first_repeat": ladder["legacy"][stage]["first_repeat_roc_auc_mean"],
                                "abs_diff": abs(recorded[stage] - ladder["legacy"][stage]["first_repeat_roc_auc_mean"])}
                        for stage in recorded}
        selection = {arm: {"all_folds": _select_stage(rows, "all"), "first_repeat": _select_stage(rows, "first")} for arm, rows in ladder.items()}
        stage_paired = {stage: {"one_hot_minus_numbers": paired(ladder["one_hot"][stage]["fold_roc_auc"], ladder["numbers"][stage]["fold_roc_auc"], 1 / (FOLDS - 1)),
                                "one_hot_minus_legacy": paired(ladder["one_hot"][stage]["fold_roc_auc"], ladder["legacy"][stage]["fold_roc_auc"], 1 / (FOLDS - 1))}
                        for stage in science.FEATURE_STAGES}
        model_paired = {model: paired(screen["one_hot"][model]["fold_roc_auc"], screen["numbers"][model]["fold_roc_auc"], 1 / (FOLDS - 1))
                        for model in science.MODEL_FAMILIES}
        # The headline comparison: the stage each arm would select, against what the legacy arm selected.
        chosen = {arm: selection[arm]["all_folds"]["selected_stage"] for arm in ARMS}
        headline = paired(ladder["one_hot"][chosen["one_hot"]]["fold_roc_auc"], ladder["legacy"][chosen["legacy"]]["fold_roc_auc"], 1 / (FOLDS - 1))
        recovery = paired(ladder["one_hot"][chosen["one_hot"]]["fold_roc_auc"], ladder["numbers"][chosen["numbers"]]["fold_roc_auc"], 1 / (FOLDS - 1))
        # The stage the campaign itself selected (legacy arm, first repeat), re-judged on all folds against raw.
        recorded_stage = selection["legacy"]["first_repeat"]["selected_stage"]
        legacy_lift = {"stage": recorded_stage,
                       **paired(ladder["legacy"][recorded_stage]["fold_roc_auc"], ladder["legacy"]["raw"]["fold_roc_auc"], 1 / (FOLDS - 1))}
        one_hot_vs_recorded = {"legacy_stage": recorded_stage,
                               **paired(ladder["one_hot"]["raw"]["fold_roc_auc"], ladder["legacy"][recorded_stage]["fold_roc_auc"], 1 / (FOLDS - 1))}
        datasets.append({
            "dataset": key, "campaign_feature_experiment": exp_id, "train_rows": int(len(X)), "analyzed_rows": int(bundle["analyzed_rows"]),
            "input_columns": int(X.shape[1]), "code_columns": codes, "code_levels": {c: int(X[c].nunique()) for c in codes},
            "ladder": ladder, "model_screen_raw": screen, "selection": selection,
            "stage_paired_roc_auc": stage_paired, "model_paired_roc_auc": model_paired,
            "selected_one_hot_vs_selected_legacy": {"one_hot_stage": chosen["one_hot"], "legacy_stage": chosen["legacy"], **headline},
            "selected_one_hot_vs_selected_numbers": {"one_hot_stage": chosen["one_hot"], "numbers_stage": chosen["numbers"], **recovery},
            "legacy_recorded_stage_minus_raw": legacy_lift,
            "one_hot_raw_minus_legacy_recorded_stage": one_hot_vs_recorded,
            "reproduction_of_campaign": reproduction,
            "reproduced_exactly": bool(max(r["abs_diff"] for r in reproduction.values()) < 1e-9),
        })

    by_key = {d["dataset"]: d for d in datasets}
    focus = [by_key[k] for k in ("bank_marketing", "german_credit") if k in by_key]
    lifts = "; ".join(f"{d['dataset']} `{d['legacy_recorded_stage_minus_raw']['stage']}` minus raw {_ci(d['legacy_recorded_stage_minus_raw'])}, "
                      f"selected over 9 folds: legacy `{d['selection']['legacy']['all_folds']['selected_stage']}`, numbers `{d['selection']['numbers']['all_folds']['selected_stage']}`, "
                      f"one_hot `{d['selection']['one_hot']['all_folds']['selected_stage']}`" for d in focus)
    recovered = "; ".join(f"{d['dataset']} {_ci(d['one_hot_raw_minus_legacy_recorded_stage'])} ({d['one_hot_raw_minus_legacy_recorded_stage']['wins']}/9 folds up)" for d in focus)
    raw_gbm = {k: d["stage_paired_roc_auc"]["raw"]["one_hot_minus_numbers"] for k, d in by_key.items()}
    by_verdict: dict[str, dict[str, list[str]]] = {}
    for model in science.MODEL_FAMILIES:
        groups: dict[str, list[str]] = {"better": [], "worse": [], "within noise": []}
        for k, d in by_key.items():
            diff = d["model_paired_roc_auc"][model]
            groups[_verdict(diff)].append(f"{k} {diff['mean']:+.4f}")
        by_verdict[model] = groups
    screen = "; ".join(f"{model}: better on {', '.join(g['better']) or 'none'}, worse on {', '.join(g['worse']) or 'none'}, within noise on {len(g['within noise'])}"
                       for model, g in by_verdict.items())
    linear_gain = by_verdict["logistic_regression"]["better"] and not by_verdict["logistic_regression"]["worse"]
    reproduced = [d["dataset"] for d in datasets if d["reproduced_exactly"]]
    lift_noise = all(_verdict(d["legacy_recorded_stage_minus_raw"]) == "within noise" for d in focus)
    recover = [_verdict(d["one_hot_raw_minus_legacy_recorded_stage"]) for d in focus]
    recover_text = ("One-hot does not reliably recover what arithmetic on codes appeared to add." if set(recover) == {"within noise"}
                    else "Corrected-interval verdicts: " + ", ".join(f"{d['dataset']} {v}" for d, v in zip(focus, recover)) + ".")
    claims = [
        ("fact", "On the first repeat (the model_building_50_v1 folds) the legacy arm selects the derived stage EXP-008/EXP-028 recorded; over all 9 folds that lift is "
                 + lifts + (". The lift that motivated this campaign is within fold noise." if lift_noise else "."),
         ["evidence.datasets[*].legacy_recorded_stage_minus_raw", "evidence.datasets[*].selection", "evidence.datasets[*].reproduction_of_campaign"]),
        ("fact", "The one-hot raw matrix against the legacy arm's recorded derived stage, LightGBM ROC-AUC over 9 paired folds: " + recovered
                 + ". " + recover_text,
         ["evidence.datasets[*].one_hot_raw_minus_legacy_recorded_stage"]),
        ("fact", "Raw-matrix LightGBM, one_hot minus numbers ROC-AUC: " + "; ".join(f"{k} {v['mean']:+.4f} ({_verdict(v)}, {v['wins']}/{v['folds']} folds up)" for k, v in raw_gbm.items()) + ".",
         ["evidence.datasets[*].stage_paired_roc_auc.raw"]),
        ("fact", "Raw-matrix five-family screen, one_hot minus numbers ROC-AUC (better/worse = corrected 95% interval excludes zero): " + screen + "."
                 + (" Logistic regression, which can only use a code through its arbitrary order, is where the encoding pays." if linear_gain else ""),
         ["evidence.datasets[*].model_paired_roc_auc"]),
        ("fact", f"The legacy arm reproduces the recorded model_building_50_v1 feature-stage ROC-AUC exactly on the first repeat's folds for {len(reproduced)} of {len(datasets)} datasets ({', '.join(reproduced) or 'none'}).",
         ["evidence.datasets[*].reproduction_of_campaign", "evidence.datasets[*].reproduced_exactly"]),
    ]
    return {
        "name": "playbook_ladder",
        "question": "On the playbook ladder, does one-hot encoding the declared category codes inside each fit fold recover the signal that arithmetic on codes used to reach, without breaking DCLAB-R11?",
        "hypothesis": "Codes carry real categorical signal (month, poutcome, checking status); a fold-fitted one-hot raw matrix matches or beats the legacy ladder's best stage, and helps linear models most.",
        "setup_summary": (f"{len(datasets)} public datasets with declared codes; model_building_50_v1 protocol ({MAX_ROWS} stratified rows, 80% training split, blocked columns removed); "
                          f"FeatureEngineer ladder ({', '.join(('raw', 'logs', 'ratios', 'interactions', 'full_fe', 'selected'))}) with LightGBM under three arms (legacy, numbers, one_hot), "
                          f"plus a raw-matrix screen of 5 families under numbers vs one_hot; RepeatedStratifiedKFold({FOLDS}x{REPEATS}, seed {RANDOM_STATE}) on training rows only, "
                          "every arm on identical folds; paired fold differences with Nadeau-Bengio corrected 95% intervals. No holdout row is read."),
        "datasets_used": list(DATASETS), "data_paths": used_paths,
        "evidence": {"protocol": {"max_rows": MAX_ROWS, "folds": FOLDS, "repeats": REPEATS, "random_state": RANDOM_STATE, "stage_model": "lightgbm",
                                  "selection_rule": "smallest feature matrix within 0.002 mean training-CV ROC-AUC of the best stage, per arm",
                                  "arms": {"legacy": "categorical=[] (codes feed derived features)", "numbers": "declared codes excluded from derived features, passed as numbers",
                                           "one_hot": "declared codes excluded from derived features and one-hot encoded on each fit fold (40 most frequent levels)"},
                                  "uncertainty": "paired fold differences; corrected_ci95 uses Nadeau-Bengio variance s^2(1/n + 1/(k-1))"},
                     "datasets": datasets},
        "claims": claims,
        "limitations": ["Training-CV evidence on at most 3,200 rows per dataset; not holdout confirmation and not production approval.",
                        "Repeated folds share training rows; even the corrected interval is approximate.",
                        "The cached tables lost the category labels, so the one-hot columns carry codes; the encoding is equivalent, the interpretation is not."],
    }


# --------------------------------------------------------------------------- CAT-002


def cat_notebook_recipes() -> dict[str, Any]:
    from dclab_rnd.expansion import runner as rn
    from dclab_rnd.studio import ProjectStore, data as sd, engine
    from dclab_rnd.studio.contract import Contract

    store = ProjectStore(Path(tempfile.mkdtemp(prefix="dclab_codes_")))
    datasets, used_paths = [], []
    for key in DATASETS:
        project = store.create(f"codes {key}", "general", "category-code encoding measurement")
        project = sd.use_sample(store, project["id"], key)
        s = project["suggestion"]
        project["contract"] = Contract(target=s["target"], task="binary", positive_label="1", forbidden=s["forbidden"],
                                       prediction_moment=s["prediction_moment"]).model_dump()
        p = engine.prepare(store, store.save(project))  # default settings: up to 20,000 rows, 3 folds
        bundle, spec = p.bundle, p.bundle.spec
        used_paths += [ROOT / "data/public" / key / "X.parquet", ROOT / "data/public" / key / "y.parquet"]
        metric, higher = spec.primary_metric, spec.higher_is_better
        model = rn._fallback_family("lightgbm")
        rows: dict[str, dict[str, dict[str, Any]]] = {}
        try:
            for arm in ("numbers", "one_hot"):
                bundle.spec = replace(spec, settings={**spec.settings, "one_hot_codes": arm == "one_hot"})
                rows[arm] = {}
                configs = [(name, model) for name in p.recipes] + [("raw", family) for family in rn.model_families(spec.task_type) if family != model]
                for name, family in configs:
                    row = rn.cross_validate(bundle, name, family, config_id=f"{name}/{family}")  # bundle.cv_splits: training rows only
                    row.pop("_oof_index"), row.pop("_oof_pred")
                    rows[arm][f"{name}/{family}"] = row
                print(f"  CAT-002 {key} {arm}: " + ", ".join(f"{c} {rn._mean(r, metric):.4f}" for c, r in rows[arm].items()), flush=True)
        finally:
            bundle.spec = spec
        test_train = 1 / (len(bundle.cv_splits) - 1)
        compact = {arm: {cid: {"recipe": r["recipe"], "model": r["model"], "feature_count_mean": r["feature_count_mean"],
                               "metrics": {m: r["metrics"][m] for m in (metric, "roc_auc", "log_loss") if m in r["metrics"]},
                               "fold_metric": [f[metric] for f in r["fold_metrics"]],
                               "top_features": [t["feature"] for t in r["top_features"][:8]]}
                         for cid, r in arm_rows.items()} for arm, arm_rows in rows.items()}
        diffs = {cid: paired(compact["one_hot"][cid]["fold_metric"], compact["numbers"][cid]["fold_metric"], test_train) for cid in compact["one_hot"]}
        if not higher:
            diffs = {cid: {**d, "note": "lower is better"} for cid, d in diffs.items()}
        tolerance = float(spec.settings.get("fe_tolerance", 0.002))
        selection = {arm: rn.select_within_tolerance([r for cid, r in rows[arm].items() if cid.endswith(f"/{model}")], metric, higher, tolerance)["recipe"]
                     for arm in rows}
        datasets.append({
            "dataset": key, "train_rows": int(len(bundle.X_train)), "cv_protocol": bundle.cv_description, "sampling": bundle.sampling,
            "primary_metric": metric, "stage_model": model, "code_columns": list(spec.categorical_columns), "code_rule": p.code_rule,
            "recipes": {name: config["description"] for name, config in p.recipes.items()},
            "results": compact, "paired_one_hot_minus_numbers": diffs,
            "selected_recipe": selection,
        })

    claims_rows = []
    for d in datasets:
        raw = d["paired_one_hot_minus_numbers"][f"raw/{d['stage_model']}"]
        lr = d["paired_one_hot_minus_numbers"]["raw/logistic_regression"]
        claims_rows.append(f"{d['dataset']} LightGBM {raw['mean']:+.4f} ({_verdict(raw)}), logistic {lr['mean']:+.4f} ({_verdict(lr)})")
    families = sorted({cid for d in datasets for cid in d["paired_one_hot_minus_numbers"] if cid.startswith("raw/")})
    family_rows = []
    for cid in families:
        diffs = [d["paired_one_hot_minus_numbers"][cid] for d in datasets if cid in d["paired_one_hot_minus_numbers"]]
        family_rows.append(f"{cid.split('/')[1]} mean {np.mean([x['mean'] for x in diffs]):+.4f} across {len(diffs)} datasets "
                           f"({sum(_verdict(x) == 'better' for x in diffs)} better, {sum(_verdict(x) == 'worse' for x in diffs)} worse)")
    selected_mi = []
    for d in datasets:
        diff = d["paired_one_hot_minus_numbers"].get("selected_mi/" + d["stage_model"])
        if diff:
            selected_mi.append(f"{d['dataset']} {diff['mean']:+.4f}")
    changed = [f"{d['dataset']} ({d['selected_recipe']['numbers']} -> {d['selected_recipe']['one_hot']})" for d in datasets
               if d["selected_recipe"]["numbers"] != d["selected_recipe"]["one_hot"]]
    claims = [
        ("fact", "DCLab notebook raw recipe, one_hot minus numbers training-CV ROC-AUC on the engine's own folds: " + "; ".join(claims_rows) + ".",
         ["evidence.datasets[*].paired_one_hot_minus_numbers", "evidence.datasets[*].results"]),
        ("fact", "Raw recipe by model family, one_hot minus numbers: " + "; ".join(family_rows) + ".",
         ["evidence.datasets[*].paired_one_hot_minus_numbers"]),
        ("fact", "The engine's recipe rule selects a different recipe after the change on " + (", ".join(changed) if changed else "no dataset") + ".",
         ["evidence.datasets[*].selected_recipe"]),
        ("risk", "selected_mi under one_hot minus numbers: " + "; ".join(selected_mi) + ". Its k is 60% of the input columns but it now picks among the encoded "
                 "columns, so it keeps fewer inputs. With one-hot codes the rule selected it on "
                 + (", ".join(d["dataset"] for d in datasets if d["selected_recipe"]["one_hot"] == "selected_mi") or "no dataset")
                 + "; sizing k on encoded columns is a separate change.",
         ["evidence.datasets[*].paired_one_hot_minus_numbers", "evidence.datasets[*].recipes"]),
    ]
    return {
        "name": "notebook_recipes",
        "question": "In the DCLab notebook, how does one-hot encoding the category codes inside each fit fold change the recipes its engine evaluates?",
        "hypothesis": "The change helps linear models clearly and tree models a little, and never needs the holdout to show it.",
        "setup_summary": (f"{len(datasets)} R&D samples loaded into the DCLab notebook with the catalog's contract and code declaration (default settings: up to 20,000 rows, "
                          "stratified 20% holdout locked and unread, 3 stratified training folds). Every registered recipe with the stage model, plus the raw recipe with "
                          "every other notebook model family, under numbers (codes passed as numbers) and one_hot (codes one-hot per fit fold); identical folds; paired fold differences "
                          "with Nadeau-Bengio corrected 95% intervals."),
        "datasets_used": list(DATASETS), "data_paths": used_paths,
        "evidence": {"protocol": {"engine_settings": "defaults (max_rows 20000, cv_folds 3)", "arms": {"numbers": "settings.one_hot_codes = False", "one_hot": "settings.one_hot_codes = True (the default)"},
                                  "uncertainty": "paired fold differences; corrected_ci95 uses Nadeau-Bengio variance s^2(1/n + 1/(k-1))"},
                     "datasets": datasets},
        "claims": claims,
        "limitations": ["Three folds give wide intervals; a direction seen on several datasets is stronger evidence than any single interval.",
                        "Training-CV evidence only; the notebook's holdout stays sealed and nothing here approves a model for production."],
    }


# --------------------------------------------------------------------------- persistence

EXPERIMENTS: dict[str, Callable[[], dict[str, Any]]] = {"CAT-001": cat_playbook_ladder, "CAT-002": cat_notebook_recipes}
NAMES = {"CAT-001": "playbook_ladder", "CAT-002": "notebook_recipes"}


def result_path(cat_id: str) -> Path:
    return RESULTS / f"{cat_id}_{NAMES[cat_id]}.json"


def run(only: list[str] | None = None, force: bool = False) -> None:
    from dclab_rnd.provenance import capture_provenance

    RESULTS.mkdir(parents=True, exist_ok=True)
    for cat_id, fn in EXPERIMENTS.items():
        if only and cat_id not in only:
            continue
        path = result_path(cat_id)
        if path.exists() and not force:
            print(f"skip {cat_id} (exists)")
            continue
        started, t0 = _now(), time.perf_counter()
        body = fn()
        claims = [{"claim_id": f"{cat_id}-C{i}", "kind": kind, "statement": text, "evidence": refs, "limitations": body["limitations"]}
                  for i, (kind, text, refs) in enumerate(body["claims"], 1)]
        payload = {
            "schema_version": 1, "campaign_id": CAMPAIGN_ID, "experiment_id": cat_id, "kind": "category_code_encoding",
            "name": body["name"], "dataset": "multiple", "datasets": body["datasets_used"],
            "question": body["question"], "hypothesis": body["hypothesis"], "setup_summary": body["setup_summary"],
            "holdout_policy": "Training rows only; every locked holdout stays unread.",
            "evidence": body["evidence"], "claims": claims, "status": "completed",
            "started_at": started, "completed_at": _now(), "elapsed_seconds": round(time.perf_counter() - t0, 1),
            "provenance": capture_provenance(ROOT, data_paths=sorted(set(body["data_paths"])), random_state=RANDOM_STATE),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{cat_id}: wrote {path.relative_to(ROOT)} ({payload['elapsed_seconds']}s)")
    write_manifest()
    write_report()


def load_results() -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("CAT-*.json"))]


def write_manifest() -> None:
    results = {r["experiment_id"]: r for r in load_results()}
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "rule": "DCLAB-R11",
        "datasets": list(DATASETS),
        "experiments": [{"experiment_id": cat_id, "name": NAMES[cat_id], "result": f"results/{result_path(cat_id).name}",
                         "status": results[cat_id]["status"] if cat_id in results else "planned"} for cat_id in EXPERIMENTS],
        "command": "python -m dclab_rnd.code_encoding run",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _ci(d: dict[str, Any]) -> str:
    return f"{d['mean']:+.4f} [{d['corrected_ci95'][0]:+.4f}, {d['corrected_ci95'][1]:+.4f}]"


def write_report() -> None:
    results = {r["experiment_id"]: r for r in load_results()}
    lines = ["# Category-codes campaign: encoding factorized codes as categories (DCLAB-R11)", "",
             "Every arm runs on identical training folds; no locked holdout is read. Differences are fold-paired, "
             "shown as mean [Nadeau-Bengio corrected 95% interval]. Training-CV evidence, not production approval.", ""]
    ladder = results.get("CAT-001")
    if ladder:
        lines += ["## CAT-001 · playbook ladder (FeatureEngineer + LightGBM)", "", f"**Setup.** {ladder['setup_summary']}", "",
                  "| Dataset | Legacy stage, 3 folds → 9 folds | Legacy stage − raw | One-hot raw − legacy stage | LightGBM raw: one_hot − numbers | Logistic | Extra Trees | Reproduced |",
                  "|---|---|---|---|---|---|---|---|"]
        for d in ladder["evidence"]["datasets"]:
            sel = d["selection"]["legacy"]
            models = d["model_paired_roc_auc"]
            lines.append(f"| {d['dataset']} | {sel['first_repeat']['selected_stage']} → {sel['all_folds']['selected_stage']} | {_ci(d['legacy_recorded_stage_minus_raw'])} | "
                         f"{_ci(d['one_hot_raw_minus_legacy_recorded_stage'])} | {_ci(d['stage_paired_roc_auc']['raw']['one_hot_minus_numbers'])} | "
                         f"{_ci(models['logistic_regression'])} | {_ci(models['extra_trees'])} | {'yes' if d['reproduced_exactly'] else 'no'} |")
        lines += [""] + [f"- **{c['kind']}**: {c['statement']}" for c in ladder["claims"]] + [""]
    notebook = results.get("CAT-002")
    if notebook:
        lines += ["## CAT-002 · DCLab notebook recipes", "", f"**Setup.** {notebook['setup_summary']}", "",
                  "| Dataset | Metric | Raw LightGBM one_hot − numbers | Logistic | Extra Trees | Selected recipe numbers → one_hot |", "|---|---|---|---|---|---|"]
        for d in notebook["evidence"]["datasets"]:
            pairs = d["paired_one_hot_minus_numbers"]
            lines.append(f"| {d['dataset']} | {d['primary_metric']} | {_ci(pairs['raw/' + d['stage_model']])} | {_ci(pairs['raw/logistic_regression'])} | "
                         f"{_ci(pairs['raw/extra_trees'])} | {d['selected_recipe']['numbers']} → {d['selected_recipe']['one_hot']} |")
        lines += [""] + [f"- **{c['kind']}**: {c['statement']}" for c in notebook["claims"]] + [""]
    for r in results.values():
        lines += [f"Limitations ({r['experiment_id']}): " + " ".join(r["claims"][0]["limitations"]) if r["claims"] else "", ""]
    lines += ["Re-run: `python -m dclab_rnd.code_encoding run --force` (results under `results/`, never edited by hand).", ""]
    (OUT / "CAMPAIGN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.code_encoding", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--only", help="comma-separated CAT ids")
    r.add_argument("--force", action="store_true")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.only.split(",") if args.only else None, force=args.force)
    else:
        write_manifest()
        write_report()
        print(f"Wrote {OUT / 'CAMPAIGN_REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
