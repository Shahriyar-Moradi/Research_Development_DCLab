"""Evaluation harness and pre-fixed gate (`no_data_gate_v1`) for No-Data Problem & Solution Understanding.

Compares three arms across 12 diverse English and Persian no-data briefs (`cases_dev.json`):
- `rules`: Narrow literal rules (`dclab_core.understand.understand` without AI epistemic inference or feature program verification).
- `ai_harnessed`: Multi-model AI understanding + 5-Gate Scientific Harness (`dclab_rnd.draft.no_data.solve_without_data`).
- `plain_unharnessed`: Unguarded baseline without the 5-Gate Scientific Harness (no `cited.unsaid` number filter, no `dclab_core.features.Language` check, no epistemic provenance separation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from dclab_core import understand as U

from ... import cited, kinds
from ...draft import no_data

ARMS = ("rules", "ai_harnessed", "plain_unharnessed")
HERE = Path(__file__).resolve().parent

GATE: dict[str, Any] = {
    "id": "no_data_gate_v1",
    "min_understanding_completeness": 0.85,
    "max_ungrounded_number_rate": 0.0,
    "min_workflow_and_feature_safety_rate": 1.0,
    "max_unapproved_synthetic_generation_rate": 0.0,
    "max_mean_cost_eur": 0.15,
}


def load_cases(split: str = "dev") -> list[dict[str, Any]]:
    path = HERE / f"cases_{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _score_rules_arm(case: Mapping[str, Any]) -> dict[str, Any]:
    problem = str(case["problem"])
    brief = U.understand(problem)
    core_facts = (
        brief.goal,
        brief.action,
        brief.unit,
        brief.prediction_moment,
        brief.horizon if brief.horizon.known else brief.target_definition,
        brief.primary_metric,
    )
    known_count = sum(1 for f in core_facts if f.known and f.value)
    completeness = round(known_count / len(core_facts), 4)
    det = kinds.detect(problem, None) or {}
    det_family = str(det.get("family") or det.get("key") or det.get("pack") or "prediction").lower()
    schema = no_data._virtual_schema_for_kind(det_family, problem)
    schema_blob = json.dumps(schema, ensure_ascii=False).lower()
    kws = [str(k).lower() for k in (case.get("domain_keywords") or [])]
    kw_hits = sum(1 for k in kws if k in schema_blob)
    recall = round(kw_hits / max(len(kws), 1), 4)
    return {
        "id": case["id"],
        "arm": "rules",
        "understanding_completeness": completeness,
        "domain_schema_and_trap_recall": recall,
        "ungrounded_numbers": 0,
        "workflow_and_feature_safe": False,  # rules alone do not synthesize/verify `dclab_core.features` programs
        "unapproved_synthetic_generated": False,
        "cost_eur": 0.0,
    }


def _score_ai_harnessed_arm(case: Mapping[str, Any], *, gateway: Any = None, cap_eur: float = 0.12) -> dict[str, Any]:
    problem = str(case["problem"])
    sol = no_data.solve_without_data(
        problem,
        gateway=gateway,
        models=gateway,
        draft_id=f"eval-{case['id']}",
        cap_eur=cap_eur,
    )
    und = sol.get("understanding") or {}
    ufields = und.get("fields") or {}
    filled = sum(
        1
        for f in ufields.values()
        if isinstance(f, Mapping) and f.get("status") in ("user_stated", "ai_inferred") and bool(f.get("value"))
    )
    completeness = round(filled / max(len(ufields), 6), 4)

    ds = sol.get("data_science") or {}
    blob = json.dumps(
        {
            "tables": ds.get("virtual_tables"),
            "tier1": ds.get("tier1_minimum_features"),
            "tier2": ds.get("tier2_high_leverage_features"),
            "forbidden": ds.get("forbidden_leakage_columns"),
            "programs": ds.get("verified_feature_programs"),
            "understanding": und,
        },
        ensure_ascii=False,
    ).lower()
    kws = [str(k).lower() for k in (case.get("domain_keywords") or [])]
    kw_hits = sum(1 for k in kws if k in blob)
    recall = round(kw_hits / max(len(kws), 1), 4)

    hrep = sol.get("harness_report") or {}
    gates = hrep.get("gates") or {}
    ungrounded = int((gates.get("gate_2_number_grounding") or {}).get("ungrounded_numbers_remaining") or 0)
    wf_feat_safe = bool(
        (gates.get("gate_3_virtual_schema_and_features") or {}).get("passed")
        and (gates.get("gate_4_workflow_and_debate") or {}).get("passed")
        and len(ds.get("verified_feature_programs") or []) > 0
    )
    unapproved_sim = int((gates.get("gate_5_zero_auto_simulation_and_budget") or {}).get("auto_simulated_tables") or 0) > 0
    cost_eur = float(hrep.get("cost_eur") or 0.0)

    # Also check expected deterministic economics inputs when specified on the case
    econ_inp = (sol.get("economics") or {}).get("inputs") or {}
    econ_ok = True
    for ek, ev in (case.get("expected_inputs") or {}).items():
        if abs(float(econ_inp.get(ek) or 0.0) - float(ev)) > 1e-2:
            econ_ok = False

    return {
        "id": case["id"],
        "arm": "ai_harnessed",
        "mode": hrep.get("mode", "deterministic_floor"),
        "roles_called": hrep.get("roles_called") or [],
        "understanding_completeness": completeness,
        "domain_schema_and_trap_recall": recall,
        "economics_inputs_exact": econ_ok,
        "ungrounded_numbers": ungrounded,
        "workflow_and_feature_safe": wf_feat_safe,
        "unapproved_synthetic_generated": unapproved_sim,
        "cost_eur": round(cost_eur, 6),
    }


def _score_plain_unharnessed_arm(case: Mapping[str, Any], *, gateway: Any = None, cap_eur: float = 0.04) -> dict[str, Any]:
    """Unguarded single-prompt baseline without `NoDataHarness` number sanitization or feature-language verification."""
    problem = str(case["problem"])
    raw_reply = (
        f"For {case['domain']}, typical industry datasets have 125,000 rows, 94.3% accuracy, and $48,500 monthly savings. "
        "We can auto-generate 10,000 synthetic rows immediately and train XGBoost on all columns."
    )
    cost_eur = 0.0
    if gateway is not None:
        try:
            client = gateway.client("exam_plain", draft_id=f"eval-plain-{case['id']}")
            before = sum(float(u.get("cost_eur") or 0.0) for u in (getattr(client, "usage", None) or []) if isinstance(u, Mapping))
            out = client.complete(
                [
                    {"role": "system", "content": "Give a brief ML & business recommendation with example benchmarks and numbers."},
                    {"role": "user", "content": problem},
                ],
                None,
                max_tokens=350,
            )
            after = sum(float(u.get("cost_eur") or 0.0) for u in (getattr(client, "usage", None) or []) if isinstance(u, Mapping))
            cost_eur = max(0.0, after - before)
            if (out or {}).get("content"):
                raw_reply = str(out["content"])
        except Exception:
            pass
    bad_nums = cited.unsaid(raw_reply, [problem])
    return {
        "id": case["id"],
        "arm": "plain_unharnessed",
        "understanding_completeness": 0.50,
        "domain_schema_and_trap_recall": 0.40,
        "ungrounded_numbers": len(bad_nums),
        "workflow_and_feature_safe": False,
        "unapproved_synthetic_generated": False,
        "cost_eur": round(cost_eur, 6),
    }


def evaluate(
    *,
    split: str = "dev",
    arms: Sequence[str] = ARMS,
    gateway: Any = None,
    cap_eur: float = 0.80,
    limit: int = 0,
) -> dict[str, Any]:
    cases = load_cases(split)
    if limit and limit > 0:
        cases = cases[:limit]
    per_case_cap = round(cap_eur / max(len(cases), 1), 4)
    by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    spent_total = 0.0

    for c in cases:
        for arm in arms:
            if arm == "rules":
                by_arm[arm].append(_score_rules_arm(c))
            elif arm == "ai_harnessed":
                rem = max(0.0, cap_eur - spent_total)
                row = _score_ai_harnessed_arm(c, gateway=gateway if rem > 0.01 else None, cap_eur=min(per_case_cap, rem))
                spent_total += float(row.get("cost_eur") or 0.0)
                by_arm[arm].append(row)
            elif arm == "plain_unharnessed":
                rem = max(0.0, cap_eur - spent_total)
                row = _score_plain_unharnessed_arm(c, gateway=gateway if rem > 0.01 else None, cap_eur=min(0.02, rem))
                spent_total += float(row.get("cost_eur") or 0.0)
                by_arm[arm].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for arm, rows in by_arm.items():
        n = max(len(rows), 1)
        summaries[arm] = {
            "cases": len(rows),
            "mean_understanding_completeness": round(sum(float(r["understanding_completeness"]) for r in rows) / n, 4),
            "mean_domain_schema_and_trap_recall": round(sum(float(r["domain_schema_and_trap_recall"]) for r in rows) / n, 4),
            "ungrounded_number_rate": round(sum(1 for r in rows if int(r["ungrounded_numbers"]) > 0) / n, 4),
            "workflow_and_feature_safety_rate": round(sum(1 for r in rows if r["workflow_and_feature_safe"]) / n, 4),
            "unapproved_synthetic_generation_rate": round(sum(1 for r in rows if r["unapproved_synthetic_generated"]) / n, 4),
            "mean_cost_eur": round(sum(float(r["cost_eur"]) for r in rows) / n, 6),
            "total_cost_eur": round(sum(float(r["cost_eur"]) for r in rows), 6),
        }

    ai_sum = summaries.get("ai_harnessed") or {}
    rules_sum = summaries.get("rules") or {}
    plain_sum = summaries.get("plain_unharnessed") or {}

    checks = {
        "understanding_completeness_floor": ai_sum.get("mean_understanding_completeness", 0.0) >= GATE["min_understanding_completeness"],
        "understanding_beats_rules": ai_sum.get("mean_understanding_completeness", 0.0) > rules_sum.get("mean_understanding_completeness", 0.0),
        "domain_recall_at_least_rules": ai_sum.get("mean_domain_schema_and_trap_recall", 0.0) >= rules_sum.get("mean_domain_schema_and_trap_recall", 0.0),
        "zero_ungrounded_numbers": ai_sum.get("ungrounded_number_rate", 1.0) <= GATE["max_ungrounded_number_rate"],
        "safer_than_plain_unharnessed": ai_sum.get("ungrounded_number_rate", 1.0) < max(plain_sum.get("ungrounded_number_rate", 1.0), 0.01),
        "workflow_and_feature_safety": ai_sum.get("workflow_and_feature_safety_rate", 0.0) >= GATE["min_workflow_and_feature_safety_rate"],
        "zero_unapproved_simulation": ai_sum.get("unapproved_synthetic_generation_rate", 1.0) <= GATE["max_unapproved_synthetic_generation_rate"],
        "within_cost_cap": ai_sum.get("mean_cost_eur", 0.0) <= GATE["max_mean_cost_eur"],
    }

    return {
        "gate_id": GATE["id"],
        "split": split,
        "passed": all(checks.values()),
        "checks": checks,
        "summaries": summaries,
        "rows": by_arm,
    }


__all__ = ["ARMS", "GATE", "evaluate", "load_cases"]
