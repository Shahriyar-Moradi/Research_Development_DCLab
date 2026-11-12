"""Versioned TP-06 release-evaluation gate for task_planning_v3 (dual v1 + v2 gate)."""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from dclab_rnd.agent_eval.task_planning_v2.gate import (
    CASES,
    CASE_FILE,
    CASE_SHA256,
    EXPECTED,
    GATE,
    MAX_TOTAL_USD,
    SEEDS,
    BudgetGateway as _V2BudgetGateway,
    DollarCeiling,
    identity_valid,
    judge as judge_v1,
    snapshot_openrouter_prices,
)
from dclab_rnd.models.gateway import mark_invented_tables

GATE_V2 = MappingProxyType({
    **GATE,
    "version": "tp06.gate.v2",
    "min_latency_floor_seconds": 1.0,
    "max_recovered_seconds": 240.0,
})


class BudgetGateway(_V2BudgetGateway):
    """V2 BudgetGateway marked for invented_accounts_v2 so whole prompts + raw reasoning are kept."""

    def __init__(self, usage, ceiling: DollarCeiling, expected_model: str, *, home: Path | None = None,
                 transport=None, invented_tables: str = "invented_accounts_v2"):
        super().__init__(usage, ceiling, expected_model, home=home, transport=transport)
        if invented_tables:
            mark_invented_tables(self, invented_tables)


def judge_v2(scores, *, configured_model=False, c_integrated=False, proposal_probe=None):
    """Evaluate strict tp06.gate.v1 plus pre-declared tp06.gate.v2 comparable-baseline latency check."""
    strict = judge_v1(scores, configured_model=configured_model, c_integrated=c_integrated, proposal_probe=proposal_probe)
    indexed = {
        arm: {r.get("id"): r for r in scores.get(arm, []) if r.get("judged") and identity_valid(r)}
        for arm in ("A", "C")
    }
    a, c = indexed["A"], indexed["C"]
    floor_s = float(GATE_V2["min_latency_floor_seconds"])
    max_ratio = float(GATE_V2["max_latency_ratio"])
    max_rec_s = float(GATE_V2["max_recovered_seconds"])

    comparable_failures = []
    recovery_bound_failures = []
    for key, r in c.items():
        if key not in a:
            continue
        c_sec = r.get("seconds")
        a_sec = a[key].get("seconds")
        if c_sec is None or c_sec < 0 or a_sec is None or a_sec < 0:
            comparable_failures.append({"id": key, "reason": "missing_or_negative_seconds"})
            continue
        if a[key].get("correct") == 1:
            limit = max(max_ratio * float(a_sec), max_ratio * floor_s)
            if float(c_sec) > limit:
                comparable_failures.append({"id": key, "a_seconds": a_sec, "c_seconds": c_sec, "limit": round(limit, 6)})
        else:
            if float(c_sec) > max_rec_s:
                recovery_bound_failures.append({"id": key, "a_seconds": a_sec, "c_seconds": c_sec, "limit": max_rec_s})

    latency_v2_ok = None if not c else (not comparable_failures and not recovery_bound_failures)
    non_latency_reasons = [r for r in strict["reasons"] if r != "gain_or_resource_threshold_not_met"]
    costs_ok = strict["resource_checks"]["cost_and_prompts"]
    gain_ok = strict["gain_passed"]

    reasons_v2 = list(non_latency_reasons)
    if strict["verdict"] == "INSUFFICIENT-EVIDENCE":
        verdict_v2 = "INSUFFICIENT-EVIDENCE"
    elif reasons_v2:
        verdict_v2 = "FAIL"
    elif not gain_ok or not costs_ok or not latency_v2_ok:
        verdict_v2 = "FAIL"
        reasons_v2.append("gain_or_resource_threshold_not_met")
    else:
        verdict_v2 = "PASS"

    return {
        "gate_v1": strict,
        "gate_v2": {
            "version": GATE_V2["version"],
            "verdict": verdict_v2,
            "promoted": verdict_v2 == "PASS",
            "reasons": sorted(set(reasons_v2)),
            "gain_passed": gain_ok,
            "resource_checks": {
                "cost_and_prompts": costs_ok,
                "latency_v2": latency_v2_ok,
                "comparable_failures": comparable_failures,
                "recovery_bound_failures": recovery_bound_failures,
            },
            "thresholds": {
                "min_latency_floor_seconds": floor_s,
                "max_latency_ratio": max_ratio,
                "max_recovered_seconds": max_rec_s,
                "max_extra_prompts": GATE_V2["max_extra_prompts"],
                "max_eur_per_case": GATE_V2["max_eur_per_case"],
            },
        },
        # Top-level verdict mirrors strict v1 alongside explicit v2 field for full audit clarity.
        **strict,
        "verdict_v2": verdict_v2,
        "promoted_v2": verdict_v2 == "PASS",
    }


__all__ = [
    "BudgetGateway",
    "CASES",
    "CASE_FILE",
    "CASE_SHA256",
    "DollarCeiling",
    "EXPECTED",
    "GATE",
    "GATE_V2",
    "MAX_TOTAL_USD",
    "SEEDS",
    "identity_valid",
    "judge_v1",
    "judge_v2",
    "snapshot_openrouter_prices",
]
