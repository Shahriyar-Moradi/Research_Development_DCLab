"""DCLab's decision models in the main product's own path for custom decision models (package 22.7).

The main product judges a decision model by an R3 run (``apps/api/app/services/r3_evaluation_service.py`` at the
recorded commit): cases of its three Jev purposes, an answerer called as ``answerer(release, cases)`` returning one
``({"value": ...}, confidence)`` per case (a yes or no's value is the probability of yes, with no confidence; a choice's
value is its label, with its confidence; None is unavailable), each answer read through the purpose's acting band and
the deployed policy, and a run row in ``r3_runs``. This module gives a DCLab answerer that shape and runs the same
reading here:

- ``project(point, state)``: the state as the main product sends it for the purpose (``jev_releases``): fewer fields
  than DCLab's classifiers were trained on (no ``availability`` exists here; the name's tokens are read from the
  column's name, which it sends);
- ``R3Answerer``: a DCLab ``DecisionAnswerer`` as the main product's ``Answerer`` callable;
- ``run(cases, answerer, ...)``: the R3-shaped reading of one answerer against the rule on the same cases, by
  partition and by label source, with the ``ai_off`` ablation, as an ``r3_runs``-shaped row (its digests computed as
  there); ``answer_rows`` and ``label_rows``: the cases as ``semantic_decision_answers`` rows in its label format.

What does not fit, said: a classifier makes no model call, and ``llm_invocation_id`` is NOT NULL there: its rows need
an invocation row of their own, or that column relaxed (the main product's change and its ADR 0008, never made here).
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional

from dclab_core.adapters.main_product import recorded
from dclab_core.adapters.main_product_ledger import JEV_BANDS, LABEL_SOURCES, LEAKAGE_RULE_YES, as_raw, banded
from dclab_core.bands import name_tokens
from dclab_core.ledger import POINTS, Answer

PARTITIONS = {"sealed": "sealed", "dev": "development", "development": "development", "train": "development"}
NO_INVOCATION = ("A classifier makes no model call: semantic_decision_answers.llm_invocation_id is NOT NULL there, so these "
                 "rows need an invocation row of their own or that column relaxed, the main product's change (ADR 0008).")


def releases() -> dict[str, dict[str, Any]]:
    return {k: v for k, v in recorded()["jev_releases"].items() if not k.startswith("_")}


def project(point: str, state: Mapping[str, Any]) -> dict[str, Any]:
    """The state the main product sends for ``point``: its subject and context fields only, and the name's tokens read
    from the column's name when the purpose does not send them (the classifiers read names through tokens)."""
    release = releases()[point]
    keep = set(release["subject_fields"]) | set(release.get("context_fields") or [])
    out = {k: v for k, v in state.items() if k in keep and v is not None}
    if "name_tokens" not in out and out.get("column"):
        out["name_tokens"] = name_tokens(str(out["column"]))
    return out


def r3_answer(point: str, answer: Optional[Answer]) -> Optional[tuple[dict[str, Any], Optional[float]]]:
    if answer is None or answer.value is None:
        return None
    raw, confidence = as_raw(point, answer)
    return {"value": raw}, confidence


class R3Answerer:
    """A DCLab answerer as the main product's ``Answerer``: ``(release, cases) -> [answer]``, a release naming its
    purpose and each case holding its state in ``fields`` (and ``context``) and its options."""

    live = False

    def __init__(self, answerer: Any, projected: bool = True):
        self.answerer, self.projected = answerer, projected
        self.candidate = f"dclab:{answerer.name}@{answerer.version}"

    def __call__(self, release: Any, cases: list[Any]) -> list[Optional[tuple[dict[str, Any], Optional[float]]]]:
        point = release["purpose"] if isinstance(release, Mapping) else release.purpose
        out = []
        for case in cases:
            get = case.get if isinstance(case, Mapping) else lambda k, d=None: getattr(case, k, d)
            state = {k: v for k, v in {"column": get("column"), **(get("fields") or {}), **(get("context") or {})}.items() if v is not None}
            if self.projected:
                state = project(point, state)
            try:
                got = self.answerer.answer(point, state, get("options") or releases()[point].get("choices") or ["yes", "no"])
            except Exception:  # noqa: BLE001 - an answerer that fails is unavailable for that case
                got = None
            out.append(r3_answer(point, got))
        return out


def ai_off(release: Any, cases: list[Any]) -> list[None]:
    """The ablation: no model at all; the policy must then be the rule."""
    return [None] * len(cases)


MODELED_ROLES = ("numeric", "categorical_code")  # a role change between these is the one a model may make at level 2
EXCLUDING_ROLES = ("identifier", "free_text", "other")
ALPHA = 0.05  # the one-sided level of every bound, as there
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782,
       13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729}  # Student's t at 0.95, by degrees of freedom


def _rule(point: str, value: Any) -> Any:
    return (value in LEAKAGE_RULE_YES or value is True) if point == "feature.leakage_suspect" else value


def score(point: str, rule: Any, answer: Optional[tuple[dict[str, Any], Optional[float]]], label: Any,
          accepts: tuple[str, ...] = MODELED_ROLES) -> dict[str, Any]:
    """One case read as the main product reads it: the band; the agreement (unavailable, abstain, agree, disagree); the
    deployed policy at level 2: the model's value when it agrees, or when a disagreement is one a model may make there
    (at semantic_role only between numeric and categorical_code, and only a value the deployed validator ``accepts``),
    else the rule's; and each one's correctness."""
    value = banded(point, answer[0], answer[1]) if answer is not None else None
    agreement = "unavailable" if answer is None else "abstain" if value is None else "agree" if _rule(point, rule) == value else "disagree"
    kind = None
    if point == "column.semantic_role":
        kind = "exclusion" if value in EXCLUDING_ROLES else "role_numeric_categorical" if value in MODELED_ROLES and rule in MODELED_ROLES else None
    eligible = agreement == "disagree" and (point != "column.semantic_role" or (kind == "role_numeric_categorical" and value in accepts))
    used = value if agreement == "agree" or eligible else rule
    truth = label
    raw_correct, confidence = None, None
    if answer is not None:
        raw = answer[0].get("value")
        if releases()[point]["primitive"] == "noul" and isinstance(raw, (int, float)):
            confidence, raw_correct = max(raw, 1 - raw), (raw >= 0.5) == truth
        else:
            confidence, raw_correct = answer[1], raw == truth
    return {"agreement": agreement, "ai": value, "rule_correct": _rule(point, rule) == truth, "policy_correct": _rule(point, used) == truth,
            "ai_correct": (value == truth) if value is not None else None, "eligible": eligible, "confidence": confidence, "raw_correct": raw_correct}


def _wilson(right: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p, z2 = right / n, z * z
    centre, half = (p + z2 / (2 * n)) / (1 + z2 / n), z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _precision(hits: list[tuple[bool, float]]) -> dict[str, Any]:
    """Precision with Kish's effective n (weights are a blind draw's inverse probabilities; 1 here)."""
    if not hits:
        return {"n": 0, "n_eff": 0.0, "errors": 0, "precision": None, "lb": None, "ub": None}
    total = sum(w for _, w in hits)
    n_eff = total * total / sum(w * w for _, w in hits)
    p = sum(w for ok, w in hits if ok) / total
    lb, ub = _wilson(p * n_eff, n_eff)
    return {"n": len(hits), "n_eff": round(n_eff, 2), "errors": sum(1 for ok, _ in hits if not ok), "precision": round(p, 4),
            "lb": round(lb, 4), "ub": round(ub, 4)}


def _t_sf(x: float, df: int) -> float:
    """P(T > x) for Student's t, by integrating its density (no scipy in dclab_core)."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    hi, steps = max(abs(x), 1.0) * 60, 6000
    h = (hi - x) / steps
    f = [c * (1 + (x + i * h) ** 2 / df) ** (-(df + 1) / 2) for i in range(steps + 1)]
    return max(0.0, min(1.0, h / 3 * (f[0] + f[-1] + 4 * sum(f[1:-1:2]) + 2 * sum(f[2:-1:2]))))


def _gap(by_cluster: dict[str, list[float]], seed: int = 0, margins: tuple[float, ...] = (0.05, 0.01), draws: int = 2000) -> dict[str, Any]:
    """The policy's accuracy minus the rule's on the answered cases, with one-sided 95% bounds: a bootstrap over the
    clusters (tables) from 20 of them, else a t interval on the clusters' means."""
    import numpy as np

    clusters = [v for v in by_cluster.values() if v]
    g = len(clusters)
    if g == 0:
        return {"clusters": 0, "point": None, "lb": None, "ub": None, "method": "none", "p_values": {}}
    point = float(np.mean([d for c in clusters for d in c]))
    if g >= 20:
        rng = np.random.default_rng(seed)
        sums, sizes = np.array([sum(c) for c in clusters]), np.array([len(c) for c in clusters])
        picks = rng.integers(0, g, size=(draws, g))
        means = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
        lb, ub = float(np.quantile(means, ALPHA)), float(np.quantile(means, 1 - ALPHA))
        p_values = {str(m): max(float(np.mean(means <= -m)), 1 / (draws + 1)) for m in margins}
        method = "cluster_bootstrap"
    else:
        means = [sum(c) / len(c) for c in clusters]
        centre = sum(means) / g
        se = (math.sqrt(sum((m - centre) ** 2 for m in means) / (g - 1)) / math.sqrt(g)) if g > 1 else float("inf")
        usable = g > 1 and se > 0
        t = T95.get(g - 1, 1.645)
        lb, ub = (centre - t * se, centre + t * se) if usable else (float("nan"), float("nan"))
        p_values = {str(m): (_t_sf((centre + m) / se, g - 1) if usable else 1.0) for m in margins}
        method = "t_cluster_means"
    return {"clusters": g, "point": round(point, 4), "lb": round(lb, 4) if math.isfinite(lb) else None,
            "ub": round(ub, 4) if math.isfinite(ub) else None, "method": method, "p_values": p_values}


def metrics(rows: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """Every statistic on the answered cases (an unavailable answer takes the rule's value and would pull the gap to
    zero); ``total`` and ``unavailable_rate`` keep the rest."""
    total = len(rows)
    answered = [r for r in rows if r["agreement"] != "unavailable"]
    n = len(answered)
    count = {k: sum(r["agreement"] == k for r in rows) for k in ("unavailable", "abstain", "disagree", "agree")}
    base = {"total": total, "n": n, "unavailable_rate": round(count["unavailable"] / total, 4) if total else None,
            "rule_accuracy_all": round(sum(r["rule_correct"] for r in rows) / total, 4) if total else None,
            "ai_policy_accuracy_all": round(sum(r["policy_correct"] for r in rows) / total, 4) if total else None}
    if n == 0:
        return {**base, "datasets": 0}
    in_band = [r for r in answered if r["ai_correct"] is not None]
    by_cluster: dict[str, list[float]] = {}
    for r in answered:
        by_cluster.setdefault(str(r["dataset"]), []).append(float(r["policy_correct"]) - float(r["rule_correct"]))
    weight = lambda r: float(r.get("weight") or 1.0)  # noqa: E731
    calibrated = [r for r in rows if r.get("confidence") is not None]
    return {**base, "datasets": len(by_cluster),
            "rule_accuracy": round(sum(r["rule_correct"] for r in answered) / n, 4),
            "ai_policy_accuracy": round(sum(r["policy_correct"] for r in answered) / n, 4),
            "ai_in_band_accuracy": round(sum(bool(r["ai_correct"]) for r in in_band) / len(in_band), 4) if in_band else None,
            "disagreement_rate": round(count["disagree"] / n, 4), "abstain_rate": round(count["abstain"] / n, 4),
            "in_band": _precision([(bool(r["ai_correct"]), weight(r)) for r in in_band]),
            "bands": {name: _precision([(bool(r["ai_correct"]), weight(r)) for r in in_band if r["ai"] is value]) for name, value in (("yes", True), ("no", False))},
            "deployed_policy": _precision([(bool(r["ai_correct"]), weight(r)) for r in in_band if r["eligible"]]),
            "gap": _gap(by_cluster, seed=seed),
            "calibration_n": len(calibrated)}


def report_digest(report: Mapping[str, Any]) -> str:
    """The report's content digest, as the main product computes it (its run's own fields left out)."""
    body = {k: v for k, v in report.items() if k not in ("run_id", "created_at", "digest", "run_digest", "workspace_evidence")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def run_digest(report: Mapping[str, Any]) -> str:
    return hashlib.sha256(f"{report['digest']}|{report['run_id']}|{report['created_at']}".encode()).hexdigest()


def run(cases: Iterable[Mapping[str, Any]], answerer: Callable[..., list[Any]], *, candidate: str, model_id: str,
        release_version: str = "1", at: Optional[str] = None, run_id: Optional[str] = None, seed: int = 0) -> dict[str, Any]:
    """The R3-shaped reading of ``answerer`` against the rule on ``cases`` (each: point, dataset, partition, column,
    fields, context, options, rule, label, label_source, and optionally accepts and weight), with the ``ai_off``
    ablation, as an ``r3_runs`` row whose report carries its run id, time and digests as the main product verifies them."""
    import uuid

    for name, value, most in (("candidate", candidate, 128), ("model_id", model_id, 64)):
        if len(value) > most:
            raise ValueError(f"{name} is longer than {most} characters: r3_runs keeps it whole, never sliced")
    cases = list(cases)
    by_point: dict[str, list[Mapping[str, Any]]] = {}
    for c in cases:
        by_point.setdefault(c["point"], []).append(c)
    points: dict[str, Any] = {}
    ablation_ok = True
    for point, mine in sorted(by_point.items()):
        release = {"purpose": point, **releases()[point]}
        answers, off = answerer(release, mine), ai_off(release, mine)
        scored = [{**score(point, c["rule"], a, c["label"], tuple(c.get("accepts") or MODELED_ROLES)), "dataset": c["dataset"],
                   "partition": PARTITIONS.get(c["partition"], c["partition"]), "source": c.get("label_source") or "benchmark",
                   "weight": c.get("weight") or 1.0} for c, a in zip(mine, answers)]
        baseline = [score(point, c["rule"], a, c["label"]) for c, a in zip(mine, off)]
        ablation_ok &= all(b["policy_correct"] == b["rule_correct"] for b in baseline)  # no model: the policy is the rule
        sources = {p: metrics([r for r in scored if r["partition"] == p], seed) for p in ("development", "sealed") if any(r["partition"] == p for r in scored)}
        sources["all"] = metrics(scored, seed)
        points[point] = {"key": point, "max_level": release.get("max_level"), "band": JEV_BANDS.get(point), "sources": sources,
                         "by_label_source": {s: metrics([r for r in scored if r["source"] == s], seed) for s in sorted({r["source"] for r in scored})}}
    pair = "+".join(f"{p}@v{releases()[p]['version']}" for p in sorted(points))
    if len(pair) > 512:
        raise ValueError("pair_release is longer than 512 characters: r3_runs keeps it whole, never sliced")
    report: dict[str, Any] = {"schema_version": 1, "run_id": run_id or str(uuid.uuid4()), "created_at": at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "candidate": candidate, "live": False, "partition": "both", "cases": len(cases), "points": points,
                              "pair": {"release": pair, "model_id": model_id}, "ablation": {"ai_off_equals_rule": ablation_ok},
                              "release_version": release_version, "note": NO_INVOCATION}
    report["digest"] = report_digest(report)
    report["run_digest"] = run_digest(report)
    row = {"id": report["run_id"], "candidate": candidate, "live": False, "pair_release": pair, "model_id": model_id,
           "content_digest": report["digest"], "run_digest": report["run_digest"], "cases": len(cases), "report": report,
           "created_at": report["created_at"]}
    columns = set(recorded()["r3"]["runs_columns"])
    if not set(row) <= columns:
        raise ValueError(f"not r3_runs columns: {sorted(set(row) - columns)}")
    return row


def answer_rows(cases: Iterable[Mapping[str, Any]], answerer: Callable[..., list[Any]], *, model_id: str, release_version: str,
                workspace: str) -> list[dict[str, Any]]:
    """The cases as ``semantic_decision_answers`` answer rows (their content; the main product assigns the ids), read
    through its bands at level 0: the rule's value used, the model's beside it."""
    out = []
    cases = list(cases)
    for point in sorted({c["point"] for c in cases}):
        mine = [c for c in cases if c["point"] == point]
        release = {"purpose": point, **releases()[point]}
        for c, a in zip(mine, answerer(release, mine)):
            s = score(point, c["rule"], a, c["label"])
            out.append({"decision_point_key": point, "purpose": point, "release_version": release_version, "model_id": model_id[:64],
                        "question_key": str(c.get("column") or (c.get("fields") or {}).get("column") or "")[:200],
                        "question_digest": hashlib.sha256(json.dumps({**(c.get("fields") or {}), **(c.get("context") or {})}, sort_keys=True).encode()).hexdigest(),
                        "data_class": "metadata", "evidence_partition": "train" if point == "feature.leakage_suspect" else "metadata",
                        "primitive": releases()[point]["primitive"], "answer": a[0] if a else {}, "confidence": a[1] if a else None,
                        "in_acting_band": s["ai"] is not None, "rule_answer": {"value": c["rule"]}, "agreement": s["agreement"], "level": 0,
                        "policy_outcome": "rule", "value_used": {"value": c["rule"]}, "workspace_id": workspace, "dataset_id": None,
                        "llm_invocation_id": None, "labels_version": 0})
    columns = set(recorded()["experience_tables"]["semantic_decision_answers"])
    for r in out:
        if not set(r) <= columns:
            raise ValueError(f"not semantic_decision_answers columns: {sorted(set(r) - columns)}")
    return out


def label_rows(labels: Iterable[Mapping[str, Any]], answer_row: Optional[Mapping[str, Any]] = None,
               answer_id: Optional[str] = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Labels in the main product's label format, in order: a source it has no word for (DCLab's ``measured``) is left
    out and counted first; version 1 supersedes the answer row (``answer_id``: its id there, which the main product
    maps from this record), each later version the label row before it; ``ground_truth`` is ``{"value": ...}``; a label
    without its time is refused. ``answer_row``: the answer row's content, copied into each label row as its table
    requires (when it is not given, the main product copies it from its own answer row)."""
    kept, left = [], {}
    for lab in labels:
        if LABEL_SOURCES.get(str(lab.get("source"))) is None:
            left[str(lab.get("source"))] = left.get(str(lab.get("source")), 0) + 1
        else:
            kept.append(lab)
    out: list[dict[str, Any]] = []
    previous = answer_id
    for version, lab in enumerate(sorted(kept, key=lambda x: str(x.get("at") or "")), start=1):
        if not lab.get("at"):
            raise ValueError("a label needs its time (labeled_at is required there)")
        out.append({**dict(answer_row or {}), "labels_version": version, "ground_truth": {"value": lab.get("value")},
                    "label_source": LABEL_SOURCES[str(lab["source"])], "labeled_at": lab["at"], "supersedes_label_id": previous})
        previous = lab.get("id")
    return out, left


__all__ = ["NO_INVOCATION", "R3Answerer", "ai_off", "answer_rows", "label_rows", "metrics", "project", "r3_answer", "releases", "report_digest",
           "run", "run_digest", "score"]
