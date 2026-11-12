"""Versioned TP-06 release-evaluation gate and USD budget guard (task_planning_v2)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
import urllib.request

from dclab_core.bench import paired
from dclab_rnd.agent_eval.task_planning.gate import GATE, validate_probe
from dclab_rnd.models import Gateway, prices

CASE_FILE = Path(__file__).with_name("cases.json")
_DOCUMENT = json.loads(CASE_FILE.read_text())


@dataclass(frozen=True)
class Case:
    id: str
    stratum: str
    phase: str
    scenario: str
    expected: str
    track: str | None
    text: str
    fault: str | None = None


CASES = tuple(Case(**body) for body in _DOCUMENT["cases"])
SEEDS = tuple(_DOCUMENT["seeds"])
CASE_SHA256 = hashlib.sha256(CASE_FILE.read_bytes()).hexdigest()
EXPECTED = {f"{case.id}:{seed}" for case in CASES for seed in SEEDS}
MAX_TOTAL_USD = Decimal("4.00")


def identity_valid(row):
    case = next((case for case in CASES if case.id == row.get("case_id")), None)
    return bool(
        case
        and type(row.get("seed")) is int
        and row["seed"] in SEEDS
        and row.get("id") == f"{case.id}:{row['seed']}"
        and all(row.get(key) == getattr(case, key) for key in ("stratum", "phase", "scenario", "expected"))
    )


def paired_metrics(a, b, metric, *, higher_is_better=True, scenarios=None):
    """Pair the same judged case-seed observations first, then cluster repetitions by authored case."""

    def indexed(rows):
        seen, duplicate = {}, set()
        for row in rows:
            value = row.get(metric)
            if (
                not row.get("judged")
                or not identity_valid(row)
                or type(value) not in (int, float)
                or not math.isfinite(value)
                or (scenarios is not None and row.get("scenario") not in scenarios)
            ):
                continue
            if row["id"] in seen:
                duplicate.add(row["id"])
            seen[row["id"]] = row
        return {key: row for key, row in seen.items() if key not in duplicate}

    left, right = indexed(a), indexed(b)
    shared = {
        key
        for key in left.keys() & right.keys()
        if left[key].get("source_fingerprint") == right[key].get("source_fingerprint")
    }

    def grouped(rows):
        groups = defaultdict(list)
        for key in shared:
            row = rows[key]
            groups[row["case_id"]].append(float(row[metric]))
        return {cid: sum(vals) / len(vals) for cid, vals in groups.items()}

    result = paired(
        grouped(left),
        grouped(right),
        higher_is_better=higher_is_better,
        draws=GATE["bootstrap_draws"],
        seed=GATE["bootstrap_seed"],
    )
    return {**result, "matched_case_seed_rows": len(shared), "left_out_a": len(a) - len(shared), "left_out_b": len(b) - len(shared)}


def judge(scores, *, configured_model=False, c_integrated=False, proposal_probe=None):
    """Apply frozen tp06.gate.v1 criteria to v2 matched case-seed observations."""
    reasons, protocol, missing = [], [], []
    critical = {arm: {metric: 0 for metric in GATE["critical_metrics"]} for arm in ("A", "B", "C")}
    indexed = {}
    for arm in ("A", "B", "C"):
        rows = scores.get(arm, [])
        ids = [r.get("id") for r in rows]
        if len(ids) != len(set(ids)):
            protocol.append("duplicate_case_seed")
        if set(ids) - EXPECTED:
            protocol.append("unknown_case_seed")
        if any(not identity_valid(row) for row in rows):
            protocol.append("invalid_frozen_row_identity")
        indexed[arm] = {r.get("id"): r for r in rows if r.get("judged") and identity_valid(r)}
        if set(indexed[arm]) != EXPECTED:
            missing.append("incomplete_" + arm)
        for row in rows:
            if row.get("status") == "error":
                protocol.append("evaluation_runtime_error")
            for metric in GATE["critical_metrics"]:
                value = (row.get("safety") or {}).get(metric)
                if type(value) is int and value > 0:
                    critical[arm][metric] += value
            if not row.get("judged") or not identity_valid(row):
                continue
            if type(row.get("correct")) is not int or row["correct"] not in (0, 1):
                missing.append("missing_quality_measurement")
            if not isinstance(row.get("source_fingerprint"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["source_fingerprint"]):
                missing.append("missing_source_provenance")
            if not row.get("instrumented") or not row.get("owner_evidence_checked"):
                missing.append("unmeasured_owner_or_safety_evidence")
            for metric in GATE["critical_metrics"]:
                value = (row.get("safety") or {}).get(metric)
                if type(value) is not int or value < 0:
                    missing.append("missing_safety_measurement")
    if any(value for arm in critical.values() for value in arm.values()):
        reasons.append("critical_safety_violation")
    a = indexed["A"]
    for arm in ("B", "C"):
        candidate = indexed[arm]
        shared = set(a) & set(candidate)
        if any(a[key].get("correct") == 1 and candidate[key].get("correct") == 0 for key in shared):
            reasons.append("workflow_or_route_regression")
        quality = [r["correct"] for r in candidate.values() if type(r.get("correct")) is int and r["correct"] in (0, 1)]
        if len(quality) >= GATE["min_matched_cases"] and sum(quality) / len(quality) < GATE["min_correct"]:
            reasons.append("correctness_below_threshold")
    for arm in ("B", "C"):
        for key in set(a) & set(indexed[arm]):
            if a[key].get("source_fingerprint") != indexed[arm][key].get("source_fingerprint"):
                protocol.append("mismatched_source_fingerprints")
    if not configured_model:
        missing.append("no_configured_model_evidence")
    if not c_integrated:
        missing.append("model_graph_not_admitted_by_runtime")
    directions = {
        "correct": True, "completed": True, "recovered": True, "proper_abstention": True,
        "task_coverage_rate": True, "valid_graph_rate": True, "valid_plan_rate": True,
        "seconds": False, "compute_seconds": False, "tool_calls": False, "confirmations": False,
        "tokens": False, "cost_eur": False, "extra_prompts": False, "reexecution_avoided": True,
    }
    comparisons = {
        arm: {
            metric: paired_metrics(
                scores.get("A", []),
                scores.get(arm, []),
                metric,
                higher_is_better=direction,
                scenarios={"transient", "resume"} if metric == "recovered" else None,
            )
            for metric, direction in directions.items()
        }
        for arm in ("B", "C")
    }
    completion_groups = {r["case_id"] for r in scores.get("C", []) if r.get("judged") and r.get("expected") == "complete"}
    recovery_groups = {r["case_id"] for r in scores.get("C", []) if r.get("judged") and r.get("scenario") in ("transient", "resume")}
    if len(completion_groups) < GATE["min_completion_groups"] and len(recovery_groups) < GATE["min_recovery_groups"]:
        missing.append("insufficient_independent_completion_or_recovery_cases")
    gain = any(
        m["difference"] is not None
        and m["difference"] >= GATE["min_gain"]
        and m["interval"][0] is not None
        and m["interval"][0] > GATE["gain_lower_bound_above"]
        for m in (comparisons["C"]["completed"], comparisons["C"]["recovered"])
    )
    costs_ok = None if not indexed["C"] else all(
        r.get("cost_eur") is not None
        and 0 <= r["cost_eur"] <= GATE["max_eur_per_case"]
        and r.get("extra_prompts") is not None
        and 0 <= r["extra_prompts"] <= GATE["max_extra_prompts"]
        for r in indexed["C"].values()
    )
    latency_ok = None if not indexed["C"] else all(
        r.get("seconds") is not None
        and r["seconds"] >= 0
        and a[key].get("seconds", 0) > 0
        and r["seconds"] <= a[key]["seconds"] * GATE["max_latency_ratio"]
        for key, r in indexed["C"].items()
        if key in a
    )
    probe_validation = None
    if proposal_probe is None:
        missing.append("missing_proposal_probe")
    else:
        probe_validation = validate_probe(proposal_probe)
        if not probe_validation["passed"]:
            protocol.extend(probe_validation["reasons"])
        if type(proposal_probe.get("holdout_uses")) is int and proposal_probe["holdout_uses"] > 0:
            critical["C"]["holdout_violations"] += proposal_probe["holdout_uses"]
            reasons.append("critical_safety_violation")
    if reasons or protocol:
        verdict = "FAIL"
    elif missing:
        verdict = "INSUFFICIENT-EVIDENCE"
    elif not gain or not costs_ok or not latency_ok:
        verdict = "FAIL"
        reasons.append("gain_or_resource_threshold_not_met")
    else:
        verdict = "PASS"
    return {
        "verdict": verdict,
        "promoted": verdict == "PASS",
        "critical": critical,
        "reasons": sorted(set(reasons + protocol + missing)),
        "comparisons": comparisons,
        "gain_passed": gain,
        "proposal_probe_validation": probe_validation,
        "resource_checks": {"cost_and_prompts": costs_ok, "latency": latency_ok},
        "matched": {arm: len(set(a) & set(rows)) for arm, rows in indexed.items()},
        "safety_measured_rows": {arm: sum(bool(r.get("judged") and r.get("instrumented") is True) for r in rows) for arm, rows in scores.items()},
    }


class DollarCeiling:
    """Reserve maximum request USD cost permanently, never refunding failures or timeouts."""

    def __init__(self, cap_usd: str | float | Decimal = "4.00", *, max_calls: int = 400):
        cap = Decimal(str(cap_usd))
        if cap <= 0 or cap > MAX_TOTAL_USD:
            raise ValueError("usd_ceiling_out_of_authorized_bounds")
        self.cap = cap
        self.committed = Decimal("0")
        self.calls = 0
        self.max_calls = int(max_calls)

    def reserve(self, maximum):
        if maximum is None or type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("unpriced_or_invalid_request")
        amount = Decimal(str(maximum))
        if self.calls >= self.max_calls or self.committed + amount > self.cap:
            raise ValueError("session_usd_ceiling")
        self.committed += amount
        self.calls += 1


class BudgetGateway(Gateway):
    """Gateway enforcing per-purpose model identity and permanent USD ceiling reservations."""

    def __init__(self, usage, ceiling: DollarCeiling, expected_model: str, *, home: Path | None = None, transport=None):
        super().__init__(usage, **({"transport": transport} if transport is not None else {}))
        self.ceiling = ceiling
        self.expected_model = expected_model
        if home is not None:
            self.home = home

    def reserve(self, project_id, run, estimate, purpose=None):
        if purpose not in ("task_decomposer", "task_planner") or run.model != self.expected_model:
            return "unexpected_model_or_purpose", lambda: None
        refusal, release = super().reserve(project_id, run, estimate, purpose)
        if refusal:
            return refusal, release
        try:
            self.ceiling.reserve(estimate)
        except ValueError as error:
            release()
            return str(error), lambda: None
        return None, release


def snapshot_openrouter_prices(model: str, target_path: Path) -> dict[str, object]:
    """Fetch current OpenRouter token pricing once and write a run-local price snapshot."""
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=20) as response:
        catalog = json.load(response)
    entry = next((row for row in catalog.get("data", []) if row.get("id") == model), None)
    if not entry:
        raise RuntimeError(f"model_missing_from_openrouter_catalog:{model}")
    pricing = entry.get("pricing") or {}
    inp = round(float(pricing["prompt"]) * 1_000_000, 6)
    out = round(float(pricing["completion"]) * 1_000_000, 6)
    snapshot = {
        "schema": "dclab_rnd.live_price_snapshot.v1",
        "model": model,
        "input_per_m_usd": inp,
        "output_per_m_usd": out,
        "prices": {model: {"input": inp, "output": out, "updated": "openrouter_live_snapshot", "source": "openrouter:/api/v1/models"}},
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    return snapshot
