"""Pitfalls measured per problem kind (package 16.14): ``make pitfalls-by-kind``.

``pitfalls.py`` measures six notebook mistakes on public binary tables. A kind's own pitfalls are measured here on its
replica (``dclab_rnd.kinds.<kind>.case(seed)``, invented data with its traps planted), the wrong way against the right
way on the same rows, on three seeds:

- **a random split where the kind needs another**: the holdout drawn at random (five draws, averaged) against the split
  the kind uses (the latest rows by time; for places, whole blocks of places, five draws), the same share held out,
  the same model: the score a random split reports minus the score the right split reports is how much it flatters;
- **the leak kept in**: the replica's planted leak columns modelled against left out, on the kind's own split: how
  much a leak inflates the score.

Kinds measured: the tabular kinds the engine models from one table with an outcome column (binary, multiclass,
regression, places, text_tabular). The others' pitfalls are their acceptance decisions (``kinds_v1``: a forecast's
random-split optimism, a stream's incidents after the alert, ...). One result per kind in
``evidence/campaigns/pitfalls_by_kind_v1/results/PIK-<n>_<kind>.json`` (never overwritten), indexed as records of type
``pitfall`` with their kind.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evidence" / "campaigns" / "pitfalls_by_kind_v1" / "results"
KINDS = ("binary", "multiclass", "regression", "places", "text_tabular")
SEEDS = (0, 1, 2)
HOLDOUT = 0.2
BLOCK_KM = 5.0
DRAWS = 5  # random holdouts (and block draws) averaged per seed
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}  # Student's t at 0.975, by degrees of freedom


def _features(frame: pd.DataFrame, target: str, drop: set[str], text: Optional[str] = None, base_time: Optional[pd.Series] = None) -> pd.DataFrame:
    """Numbers as they are, a few-valued text as codes, a long text as its length; keys, the modelled text (read apart)
    and ``drop`` left out; a time as its seconds after the row's own time (``base_time``: an outcome's time is a leak
    only that way), any other time left out."""
    out = {}
    for name in frame.columns:
        if name == target or name in drop or name == text:
            continue
        col = frame[name]
        if pd.api.types.is_datetime64_any_dtype(col):
            if base_time is not None and not col.equals(base_time):
                out[name] = (col - base_time).dt.total_seconds()
            continue
        if pd.api.types.is_bool_dtype(col):
            out[name] = col.astype(float)
        elif pd.api.types.is_numeric_dtype(col):
            if col.nunique() > 0.9 * len(col) and pd.api.types.is_integer_dtype(col):
                continue  # a row key
            out[name] = col.astype(float)
        else:
            strings = col.astype("string")
            if strings.nunique() > 0.9 * len(col):
                continue  # a row key
            out[name] = strings.str.len().astype(float) if strings.nunique() > 50 else strings.astype("category").cat.codes.astype(float)
    return pd.DataFrame(out, index=frame.index)


def _score(task: str, positive: Any, X_train, y_train, X_test, y_test, text_train=None, text_test=None) -> float:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

    if text_train is not None:  # the text read as the kind reads it: words weighted on the training rows only
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        words = TfidfVectorizer(min_df=2, max_features=5000).fit(text_train)
        svd = TruncatedSVD(n_components=30, random_state=0).fit(words.transform(text_train))
        cols = [f"_text_{i}" for i in range(30)]
        X_train = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(svd.transform(words.transform(text_train)), columns=cols)], axis=1)
        X_test = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(svd.transform(words.transform(text_test)), columns=cols)], axis=1)
        y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

    if task == "regression":
        model = HistGradientBoostingRegressor(random_state=0).fit(X_train, y_train)
        return float(mean_absolute_error(y_test, model.predict(X_test)))
    model = HistGradientBoostingClassifier(random_state=0).fit(X_train, y_train)
    if task == "binary":
        classes = list(model.classes_)
        return float(roc_auc_score((y_test == positive).astype(int), model.predict_proba(X_test)[:, classes.index(positive)]))
    return float(accuracy_score(y_test, model.predict(X_test)))


def _blocks_out(block: pd.Series, rng: Any) -> np.ndarray:
    """Whole blocks of places drawn at random until they hold the same share of rows as the random holdout."""
    sizes = block.value_counts()
    order = list(rng.permutation(sorted(sizes.index)))
    chosen, rows = [], 0
    for b in order:
        if rows >= HOLDOUT * len(block):
            break
        chosen.append(b)
        rows += int(sizes[b])
    return block.isin(chosen).to_numpy()


def _higher(task: str) -> bool:
    return task != "regression"


def _times(frame: pd.DataFrame, facts: dict[str, Any]) -> Optional[pd.Series]:
    name = facts.get("time_column") or next((c for c in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[c])), None)
    if name is None:
        return None
    return pd.to_datetime(frame[name], errors="coerce")


def measure(kind_key: str, seed: int) -> dict[str, Any]:
    """One seed of one kind: each pitfall's wrong and right score, and the gap (positive: the wrong way looks better)."""
    from dclab_rnd import kinds

    case = kinds.get(kind_key).case(seed)
    frame, facts = case.table.reset_index(drop=True), dict(case.facts or {})
    target, task = case.target, case.task
    frame = frame[frame[target].notna()].reset_index(drop=True)
    leaks = set(facts.get("leaky_columns") or ([facts["leak"]] if facts.get("leak") else []))
    keys = {facts[k] for k in ("identifier",) if facts.get(k)}
    positive = str(facts.get("positive") or (1 if task == "binary" else None))
    y = frame[target].astype(str) if task != "regression" else frame[target].astype(float)
    if task == "binary":
        positive = positive if positive in set(y) else sorted(set(y))[-1]
    rng = np.random.default_rng(seed)
    n = len(frame)
    random_test = np.zeros(n, dtype=bool)
    random_test[rng.choice(n, int(n * HOLDOUT), replace=False)] = True
    right_name, right_test = "random", random_test
    times = _times(frame, facts)
    if kind_key == "places":
        from dclab_core.spatial import blocks

        block = pd.Series(blocks(frame[facts.get("lat", "lat")].astype(float), frame[facts.get("lon", "lon")].astype(float), BLOCK_KM))
        right_name, right_test = f"whole blocks of {BLOCK_KM:g} km", _blocks_out(block, rng)
    elif times is not None and times.notna().all():
        cut = times.quantile(1 - HOLDOUT)
        right_name, right_test = "the latest rows by time", (times > cut).to_numpy()
    out: dict[str, Any] = {"seed": seed, "rows": n, "task": task, "right_split": right_name, "pitfalls": {}}
    text = facts.get("text") if facts.get("text") in frame.columns else None
    words = frame[text].fillna("").astype(str) if text else None

    def score(X, test):  # the kind's model on these columns, held out at ``test``
        return _score(task, positive, X[~test], y[~test], X[test], y[test],
                      words[~test] if words is not None else None, words[test] if words is not None else None)

    clean = _features(frame, target, leaks | keys, text, times)
    if right_name != "random":  # five draws of each random holdout, averaged: one draw is too noisy to compare
        draws = []
        for _ in range(DRAWS):
            test = np.zeros(n, dtype=bool)
            test[rng.choice(n, int(n * HOLDOUT), replace=False)] = True
            draws.append(score(clean, test))
        wrong = statistics.fmean(draws)
        if kind_key == "places":
            rights = []
            for _ in range(DRAWS):
                test = _blocks_out(block, rng)
                rights.append(score(clean, test))
            right = statistics.fmean(rights)
        else:
            right = score(clean, right_test)
        out["pitfalls"]["random_split"] = {"wrong": round(wrong, 4), "right": round(right, 4),
                                           "gap": round((wrong - right) if _higher(task) else (right - wrong), 4)}
    if leaks:
        leaky = _features(frame, target, keys, text, times)
        wrong, right = score(leaky, right_test), score(clean, right_test)
        modelled = sorted(c for c in leaks if c in leaky.columns)  # a leak counts only as the model saw it
        out["pitfalls"]["leak_kept_in"] = {"wrong": round(wrong, 4), "right": round(right, 4), "leaks": modelled,
                                           "gap": round((wrong - right) if _higher(task) else (right - wrong), 4)}
    return out


def result(kind_key: str, number: int, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    from dclab_rnd import kinds

    kind = kinds.get(kind_key)
    runs = [measure(kind_key, s) for s in seeds]
    metric = {"binary": "ROC-AUC", "multiclass": "accuracy", "regression": "mean absolute error"}[runs[0]["task"]]
    summary, claims = {}, []
    for name, title in (("random_split", "a random split instead of " + runs[0]["right_split"]), ("leak_kept_in", "the planted leak kept in")):
        gaps = [r["pitfalls"][name]["gap"] for r in runs if name in r["pitfalls"]]
        if not gaps:
            continue
        mean = statistics.fmean(gaps)
        half = T975[len(gaps) - 1] * statistics.stdev(gaps) / math.sqrt(len(gaps)) if len(gaps) > 1 else None
        ci = [round(mean - half, 4), round(mean + half, 4)] if half is not None else None
        measured = ci is not None and ci[0] > 0  # the wrong way looks better beyond the seeds' spread
        summary[name] = {"gaps": gaps, "mean": round(mean, 4), "ci95": ci, "seeds": len(gaps), "measured": measured}
        limits = ["Invented replica data with its traps planted, one model family with default settings: the size of the effect on a real "
                  "table differs; the direction is what the replica shows."]
        if measured:
            statement = (f"On the {kind.title} replica, {title} makes the {metric} look better by {mean:.4f} on average over {len(gaps)} seeds "
                         f"(95% interval {ci[0]} to {ci[1]}).")
        else:
            statement = (f"On the {kind.title} replica, {title} showed no difference beyond the spread of {len(gaps)} seeds (mean {mean:.4f}"
                         + (f", 95% interval {ci[0]} to {ci[1]}" if ci else "") + ").")
            limits.append("No difference measured on this replica: it plants no effect of this kind (no drift over time), which says nothing "
                          "about a real table, where a random split is still the wrong one for a time-ordered problem.")
        claims.append({"id": f"C{len(claims) + 1}", "kind": "fact", "statement": statement, "evidence": [f"evidence.{name}"], "limitations": limits})
    ident = f"PIK-{number:03d}"
    return {"experiment_id": ident, "campaign_id": "pitfalls_by_kind_v1", "kind": "pitfall", "problem_kind": kind_key,
            "pitfall": f"{kind_key} pitfalls", "title": f"{kind.title}: what its pitfalls cost on its replica",
            "question": f"On the {kind.title} replica, how much better does a wrong split or a kept leak make the score look?",
            "setup_summary": f"{len(seeds)} seeds of the {kind_key} replica; {int(HOLDOUT * 100)}% held out; histogram gradient boosting with default "
                             f"settings; {metric}; right split: {runs[0]['right_split']}.",
            "evidence": {**summary, "runs": runs}, "claims": claims, "track": kind.evidence_track,
            "provenance": {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seeds": list(seeds)}}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kinds", nargs="*", default=list(KINDS))
    parser.add_argument("--no-store", action="store_true")
    args = parser.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    number = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in RESULTS.glob("PIK-*.json")] or [0])
    for key in args.kinds:
        doc = result(key, number)
        for name, s in {k: v for k, v in doc["evidence"].items() if k != "runs"}.items():
            print(f"{key:13s} {name:14s} gaps {s['gaps']} mean {s['mean']} 95% {s['ci95']}")
        if not args.no_store:
            path = RESULTS / f"{doc['experiment_id']}_{key}.json"
            with path.open("x", encoding="utf-8") as handle:  # never overwritten
                json.dump(doc, handle, indent=2, default=str)
                handle.write("\n")
            number += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
