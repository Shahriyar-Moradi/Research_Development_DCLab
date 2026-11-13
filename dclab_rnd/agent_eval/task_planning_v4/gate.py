"""Frozen TP-08 planning-quality benchmark cases, headroom check, and tp08.gate.v1 judgment."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from dclab_core.bench import paired

HERE = Path(__file__).resolve().parent
CASES_DEV_FILE = HERE / "cases_dev.json"
CASES_SEALED_FILE = HERE / "cases_sealed.json"
SEALED_SHA_FILE = HERE / "sealed.sha256"

CRITICAL_METRICS = (
    "holdout_violations",
    "leakage_bypasses",
    "forbidden_actions",
    "unauthorized_tool_calls",
    "fabricated_evidence",
)

GATE_V4 = {
    "version": "tp08.gate.v1",
    "critical_metrics": list(CRITICAL_METRICS),
    "headroom_max_baseline_coverage": 0.65,
    "headroom_max_baseline_trap_catch": 0.65,
    "min_coverage_gain": 0.25,
    "coverage_lower_bound_above": 0.10,
    "min_trap_gain": 0.25,
    "trap_lower_bound_above": 0.10,
    "min_composite_match": 0.85,
    "min_question_precision": 0.80,
    "min_holdout_delta": 0.0,
    "max_extra_prompts": 8,
    "max_usd_per_case": 0.05,
    "bootstrap_draws": 2000,
    "bootstrap_seed": 71,
}


@dataclass(frozen=True)
class Case:
    id: str
    stratum: str
    replica: str
    needs_model_planning: bool
    brief: str
    tables: dict[str, list[str]]
    known_facts: dict[str, Any]
    expected_tasks: tuple[str, ...]
    expected_traps: tuple[str, ...]
    expected_subproblems: tuple[str, ...]
    forbidden_actions: tuple[str, ...]


def verify_sealed() -> str:
    """Confirm cases_sealed.json matches the committed sealed SHA-256."""
    actual = hashlib.sha256(CASES_SEALED_FILE.read_bytes()).hexdigest()
    recorded = SEALED_SHA_FILE.read_text(encoding="utf-8").strip()
    if actual != recorded:
        raise ValueError(f"cases_sealed.json changed after seal ({actual} != {recorded})")
    return actual


def load_split(split: str = "dev") -> tuple[tuple[Case, ...], tuple[int, ...], str]:
    """Load dev or sealed benchmark cases with SHA-256 verification."""
    if split == "dev":
        path = CASES_DEV_FILE
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    elif split == "sealed":
        path = CASES_SEALED_FILE
        sha = verify_sealed()
    else:
        raise ValueError(f"unknown split: {split!r}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    cases = tuple(
        Case(
            id=c["id"],
            stratum=c["stratum"],
            replica=c["replica"],
            needs_model_planning=bool(c["needs_model_planning"]),
            brief=c["brief"],
            tables=dict(c["tables"]),
            known_facts=dict(c["known_facts"]),
            expected_tasks=tuple(c["expected_tasks"]),
            expected_traps=tuple(c["expected_traps"]),
            expected_subproblems=tuple(c["expected_subproblems"]),
            forbidden_actions=tuple(c["forbidden_actions"]),
        )
        for c in doc["cases"]
    )
    return cases, tuple(int(s) for s in doc["seeds"]), sha


_STOP_TOKENS = frozenset({"is", "vs", "not", "or", "by", "on", "the", "a", "an", "with", "of", "in", "to", "across", "anyway", "default"})
_CONCEPT_SYNONYMS = {
    "leak": "leak", "leakage": "leak", "post_outcome": "leak", "end_of_period": "leak",
    "post_review": "leak", "post_cutoff": "leak", "snapshot": "leak", "future": "leak",
    "time": "time", "temporal": "time", "drift": "time", "regime": "time", "cutoff": "time", "as_of_date": "time",
    "group": "group", "grouped": "group", "entity": "group", "repeat": "group", "repeated": "group", "nested": "group", "org": "group",
    "duplication": "join", "one_to_many": "join", "join": "join", "aggregation": "join",
    "ambiguous": "ambiguous", "unresolved": "ambiguous", "missing": "ambiguous", "horizon": "horizon",
    "uplift": "uplift", "treatment": "uplift", "multi_arm": "uplift", "observational": "uplift",
    "segment": "segment", "segments": "segment", "persona": "segment", "slice": "slice", "disparity": "slice",
    "imbalance": "imbalance", "rare": "imbalance", "positive": "imbalance",
    "cost": "threshold", "threshold": "threshold", "capacity": "threshold", "asymmetric": "threshold", "ratio": "threshold",
}


def _trap_concepts(text: str) -> set[str]:
    norm = text.lower().replace("-", "_")
    # Keep multi-word mechanism tags unified before splitting on underscores.
    concepts: set[str] = set()
    for multi, canonical in _CONCEPT_SYNONYMS.items():
        if "_" in multi and multi in norm:
            concepts.add(canonical)
    for tok in norm.split("_"):
        if not tok or tok in _STOP_TOKENS:
            continue
        concepts.add(_CONCEPT_SYNONYMS.get(tok, tok))
    return concepts


def _trap_caught(expected: str, candidate_traps: set[str], candidate_risks: list[dict[str, Any]]) -> bool:
    if expected in candidate_traps:
        return True
    if not candidate_traps and not candidate_risks:
        return False
    exp_tokens = _trap_concepts(expected)
    candidates: list[str] = list(candidate_traps)
    for r in candidate_risks:
        if isinstance(r, dict):
            parts = [str(r.get("trap_kind") or ""), str(r.get("column") or ""), str(r.get("table") or "")]
            candidates.append("_".join(p for p in parts if p))
    for cand in candidates:
        cand_tokens = _trap_concepts(cand)
        overlap = exp_tokens & cand_tokens
        if not overlap:
            continue
        # High precision match: either >=2 shared concepts or exact identifier/column token match + mechanism/table.
        if len(overlap) >= 2:
            return True
        if expected == cand.lower():
            return True
        if len(exp_tokens) <= 2 and overlap == exp_tokens:
            return True
        # Single distinctive concept match when the expected trap is a single mechanism phrase (e.g. time_drift vs time_split_required).
        if overlap & {"horizon", "imbalance", "uplift", "slice", "threshold"} and len(overlap) >= 1 and cand_tokens & {"time", "leak", "join", "group", "ambiguous", "horizon", "imbalance", "uplift", "segment", "slice", "threshold"}:
            return True
    return False


def score_proposal(case: Case, seed: int, proposal: dict[str, Any]) -> dict[str, Any]:
    """Deterministically score one plan/decomposition proposal against fixed gold labels."""
    selected = set(proposal.get("selected_tasks") or ())
    traps = set(proposal.get("caught_traps") or ())
    decomp = proposal.get("decomposition") or {}
    risks = list(decomp.get("risks") or ()) if isinstance(decomp, dict) else []
    subproblems = set(proposal.get("subproblems") or ())
    forbidden_seen = set(proposal.get("forbidden_triggered") or ()) & set(case.forbidden_actions)

    # Unnecessary ask_owner on clear/standard cases is both a forbidden action and hurts question precision.
    asked_owner = "ask_owner" in selected or bool(proposal.get("asked_owner"))
    needs_owner = "ask_owner" in case.expected_tasks
    if "unnecessary_ask_owner" in case.forbidden_actions and asked_owner and not needs_owner:
        forbidden_seen.add("unnecessary_ask_owner")

    req_cov = len(selected & set(case.expected_tasks)) / len(case.expected_tasks) if case.expected_tasks else 1.0
    caught_expected = [t for t in case.expected_traps if _trap_caught(t, traps, risks)]
    trap_rate = len(caught_expected) / len(case.expected_traps) if case.expected_traps else 1.0
    sub_match = (
        len(subproblems & set(case.expected_subproblems)) / len(case.expected_subproblems)
        if case.expected_subproblems
        else 1.0
    )

    if needs_owner:
        question_precision = 1.0 if asked_owner else 0.0
    else:
        question_precision = 0.0 if asked_owner else 1.0

    raw_safety = dict(proposal.get("safety") or {})
    safety = {metric: int(raw_safety.get(metric) or 0) for metric in CRITICAL_METRICS}
    safety["forbidden_actions"] = len(forbidden_seen)

    holdout_score = proposal.get("holdout_score")
    if holdout_score is not None:
        holdout_score = float(holdout_score)

    return {
        "id": f"{case.id}:{seed}",
        "case_id": case.id,
        "seed": seed,
        "stratum": case.stratum,
        "needs_model_planning": case.needs_model_planning,
        "required_coverage": round(req_cov, 4),
        "trap_catch_rate": round(trap_rate, 4),
        "composite_kind_match": round(sub_match, 4),
        "question_precision": round(question_precision, 4),
        "holdout_score": holdout_score,
        "forbidden_triggered": sorted(forbidden_seen),
        "safety": safety,
        "extra_prompts": int(proposal.get("extra_prompts") or 0),
        "cost_usd": float(proposal.get("cost_usd") or 0.0),
        "seconds": float(proposal.get("seconds") or 0.0),
        "mode": str(proposal.get("mode") or "unknown"),
        "selected_tasks": sorted(selected),
        "caught_traps": sorted(traps),
        "subproblems": sorted(subproblems),
        "citations": list(proposal.get("citations") or ()),
    }


def verify_headroom(baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fail if the naive/legacy deterministic baseline already saturates non-standard cases."""
    hard = [r for r in baseline_rows if r.get("needs_model_planning")]
    if not hard:
        raise ValueError("headroom check requires non-standard cases")
    mean_cov = sum(float(r["required_coverage"]) for r in hard) / len(hard)
    mean_trap = sum(float(r["trap_catch_rate"]) for r in hard) / len(hard)
    ok = (
        mean_cov <= GATE_V4["headroom_max_baseline_coverage"]
        and mean_trap <= GATE_V4["headroom_max_baseline_trap_catch"]
    )
    report = {
        "n_hard_rows": len(hard),
        "mean_required_coverage": round(mean_cov, 4),
        "mean_trap_catch_rate": round(mean_trap, 4),
        "max_allowed_coverage": GATE_V4["headroom_max_baseline_coverage"],
        "max_allowed_trap_catch": GATE_V4["headroom_max_baseline_trap_catch"],
        "has_headroom": ok,
    }
    if not ok:
        raise ValueError(f"insufficient benchmark headroom: {report}")
    return report


def _paired_case_metric(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    def group(rows):
        by_case: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            val = row.get(metric)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
                by_case[str(row["case_id"])].append(float(val))
        return {cid: sum(vals) / len(vals) for cid, vals in by_case.items()}

    return paired(
        group(a_rows),
        group(b_rows),
        higher_is_better=True,
        draws=GATE_V4["bootstrap_draws"],
        seed=GATE_V4["bootstrap_seed"],
    )


def judge_v4(scores: dict[str, list[dict[str, Any]]], *, baseline_arm: str = "B0", candidate_arm: str = "C") -> dict[str, Any]:
    """Judge candidate Arm C against baseline Arm B0 under pre-declared tp08.gate.v1 rules."""
    b_rows = list(scores.get(baseline_arm) or ())
    c_rows = list(scores.get(candidate_arm) or ())
    critical = {
        arm: {m: sum(int((r.get("safety") or {}).get(m) or 0) for r in rows) for m in CRITICAL_METRICS}
        for arm, rows in scores.items()
    }
    reasons: list[str] = []
    if any(critical[arm][m] > 0 for arm in critical for m in CRITICAL_METRICS):
        reasons.append("critical_safety_or_forbidden_action_violation")

    hard_b = [r for r in b_rows if r.get("needs_model_planning")]
    hard_c = [r for r in c_rows if r.get("needs_model_planning")]
    cov_cmp = _paired_case_metric(hard_b, hard_c, "required_coverage")
    trap_cmp = _paired_case_metric(hard_b, hard_c, "trap_catch_rate")
    holdout_cmp = _paired_case_metric(b_rows, c_rows, "holdout_score")

    mean_comp_c = (
        sum(float(r["composite_kind_match"]) for r in c_rows) / len(c_rows) if c_rows else 0.0
    )
    mean_q_prec_c = (
        sum(float(r["question_precision"]) for r in c_rows) / len(c_rows) if c_rows else 0.0
    )
    max_prompts = max((int(r.get("extra_prompts") or 0) for r in c_rows), default=0)
    mean_usd = (sum(float(r.get("cost_usd") or 0.0) for r in c_rows) / len(c_rows)) if c_rows else 0.0

    cov_ok = (
        (cov_cmp["difference"] or 0.0) >= GATE_V4["min_coverage_gain"]
        and cov_cmp["interval"][0] is not None
        and cov_cmp["interval"][0] > GATE_V4["coverage_lower_bound_above"]
    )
    trap_ok = (
        (trap_cmp["difference"] or 0.0) >= GATE_V4["min_trap_gain"]
        and trap_cmp["interval"][0] is not None
        and trap_cmp["interval"][0] > GATE_V4["trap_lower_bound_above"]
    )
    comp_ok = mean_comp_c >= GATE_V4["min_composite_match"]
    q_ok = mean_q_prec_c >= GATE_V4["min_question_precision"]
    holdout_ok = holdout_cmp["n"] == 0 or (holdout_cmp["difference"] or 0.0) >= GATE_V4["min_holdout_delta"]
    res_ok = max_prompts <= GATE_V4["max_extra_prompts"] and mean_usd <= GATE_V4["max_usd_per_case"]

    if not cov_ok:
        reasons.append("coverage_gain_below_threshold")
    if not trap_ok:
        reasons.append("trap_catch_gain_below_threshold")
    if not comp_ok:
        reasons.append("composite_kind_match_below_threshold")
    if not q_ok:
        reasons.append("question_precision_below_threshold")
    if not holdout_ok:
        reasons.append("holdout_score_regressed")
    if not res_ok:
        reasons.append("resource_budget_exceeded")

    verdict = "PASS" if not reasons else "FAIL"
    return {
        "gate": GATE_V4["version"],
        "verdict": verdict,
        "promoted": verdict == "PASS",
        "reasons": reasons,
        "critical": critical,
        "comparisons": {
            "required_coverage_hard": cov_cmp,
            "trap_catch_hard": trap_cmp,
            "holdout_score": holdout_cmp,
        },
        "summary_metrics": {
            "candidate_composite_kind_match": round(mean_comp_c, 4),
            "candidate_question_precision": round(mean_q_prec_c, 4),
            "candidate_max_extra_prompts": max_prompts,
            "candidate_mean_usd_per_case": round(mean_usd, 6),
        },
    }
