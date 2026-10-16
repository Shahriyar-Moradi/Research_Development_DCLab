"""Learned AutoML (26.6a): which model family to fit, chosen from a table's meta-features, judged by its regret.

Each verified self-play solve (27.3) kept its table's meta-features (rows, columns, the shares of storage types,
cardinality and missing bands, the task, the outcome's balance; never a value) and every family's cross-validated score
in the engine's screen (training folds only). A meta-model learns from them which family wins, and is judged against
the simple rules a person would use: always XGBoost, always a random forest, always the family that wins most often.

The measure is the **normalised regret** of a pick on a problem: (the best family's score - the picked family's) /
(the best - the worst), 0 when the pick is the best, 1 when it is the worst; a family the problem's screen did not try
counts as the worst. The meta-model is judged leaving one problem kind out at a time, so it is always scored on a kind
of problem it never learned from; intervals are bootstraps over problems.

    python -m dclab_rnd.lake automl [--store]   →  evidence/campaigns/automl_v1/results/AML-NNN.json
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ROOT
from .selfplay import OUT

CAMPAIGN = ROOT / "evidence" / "campaigns" / "automl_v1"
RULE = ("A measurement on verified invented problems of DCLab's engine kinds: how close a family picked from meta-features "
        "comes to the best family of each problem's own screen, judged on problem kinds the meta-model never saw. It says "
        "nothing yet about real tables, nor that any pick beats a tuned search.")
BANDS = {"dtypes": ("integer", "float", "string", "categorical", "datetime", "boolean"),
         "cardinality": ("binary", "low", "medium", "high", "very_high"),
         "nulls": ("none", "under_1pct", "from_1_to_10pct", "from_10_to_50pct", "over_50pct")}


def screens(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or OUT / "screens.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []
    return [r for r in rows if len(r.get("screen") or []) >= 2]


def features(r: dict[str, Any]) -> list[float]:
    cols = max(1, int(r.get("columns") or 1))
    out = [math.log10(max(1, int(r.get("rows") or 1))), float(cols), float(r.get("usable_features") or 0) / cols]
    for key, bands in BANDS.items():
        counts = r.get(key) or {}
        out += [float(counts.get(b, 0)) / cols for b in bands]
    ts = r.get("target_summary") or {}
    rate = ts.get("positive_rate_train")
    out += [1.0 if r.get("task") == t else 0.0 for t in ("binary", "multiclass", "regression")]
    out += [float(rate) if isinstance(rate, (int, float)) else -1.0, math.log10(float(ts.get("imbalance_ratio_neg_per_pos") or 1.0) + 1)]
    return out


def regret(r: dict[str, Any], family: str | None) -> float:
    scores = {s["model"]: float(s["mean"]) for s in r["screen"] if s.get("model") and isinstance(s.get("mean"), (int, float))}
    best, worst = max(scores.values()), min(scores.values())
    if best == worst:
        return 0.0
    got = scores.get(family or "", worst)
    return (best - got) / (best - worst)


def winner(r: dict[str, Any]) -> str:
    return max(((s["model"], float(s["mean"])) for s in r["screen"] if isinstance(s.get("mean"), (int, float))), key=lambda x: x[1])[0]


def _boot(values: list[float], seed: int = 0, n: int = 2000) -> list[float]:
    if not values:
        return [float("nan")] * 2
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)]


def evaluate(rows: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier

    kinds = sorted({r["kind"] for r in rows})
    picks: dict[str, list[float]] = {"meta_model": [], "always_xgboost": [], "always_random_forest": [], "most_frequent_winner": [], "oracle": []}
    for held in kinds:
        train = [r for r in rows if r["kind"] != held]
        test = [r for r in rows if r["kind"] == held]
        if not train or not test:
            continue
        labels = [winner(r) for r in train]
        frequent = Counter(labels).most_common(1)[0][0]
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed).fit([features(r) for r in train], labels)
        predicted = model.predict([features(r) for r in test])
        for r, p in zip(test, predicted):
            picks["meta_model"].append(regret(r, str(p)))
            picks["always_xgboost"].append(regret(r, "xgboost"))
            picks["always_random_forest"].append(regret(r, "random_forest"))
            picks["most_frequent_winner"].append(regret(r, frequent))
            picks["oracle"].append(0.0)
    meta = picks["meta_model"]
    out = {name: {"mean_regret": round(sum(v) / len(v), 4) if v else None, "interval": _boot(v, seed), "best_picked": sum(1 for x in v if x == 0)}
           for name, v in picks.items()}
    paired = {base: {"meta_better": sum(1 for m, b in zip(meta, picks[base]) if m < b), "meta_worse": sum(1 for m, b in zip(meta, picks[base]) if m > b),
                     "difference": round(sum(b - m for m, b in zip(meta, picks[base])) / len(meta), 4) if meta else None,
                     "difference_interval": _boot([b - m for m, b in zip(meta, picks[base])], seed)}
              for base in ("always_xgboost", "always_random_forest", "most_frequent_winner")}
    return {"problems": len(meta), "kinds": kinds, "pickers": out, "paired": paired,
            "families": dict(Counter(winner(r) for r in rows))}


def run(store: bool = False) -> dict[str, Any]:
    rows = screens()
    result = evaluate(rows) if len({r["kind"] for r in rows}) >= 2 else {"problems": len(rows), "note": "fewer than two problem kinds: nothing to judge yet"}
    doc = {"campaign_id": "automl_v1", "kind": "learned_automl", "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source": "lake/bronze/selfplay/screens.jsonl", "result": result, "status": "completed", "rule": RULE}
    if store and result.get("pickers"):
        results = CAMPAIGN / "results"
        results.mkdir(parents=True, exist_ok=True)
        if not (CAMPAIGN / "manifest.json").exists():
            (CAMPAIGN / "manifest.json").write_text(json.dumps({"campaign_id": "automl_v1", "objective": "Package 26.6a: a family picked from a table's "
                "meta-features against always-XGBoost, always-random-forest and the most frequent winner, by normalised regret on held-out problem kinds.",
                "rule": RULE, "command": "python -m dclab_rnd.lake automl --store"}, indent=2) + "\n", encoding="utf-8")
        n = 1 + max([int(p.stem.split("-")[1]) for p in results.glob("AML-*.json")] or [0])
        doc = {"id": f"AML-{n:03d}", "experiment_id": f"AML-{n:03d}", **doc}
        (results / f"AML-{n:03d}.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc
