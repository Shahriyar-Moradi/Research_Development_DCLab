"""The measurement of the no-data blueprint (gate ``no_data_gate_v2``, fixed 2026-11-15 before any run under it).

Twelve written English and Persian briefs of people with no data (``cases_dev.json``), each with the family an expert would
name, the numbers the person stated, and domain words written by hand (never the blueprint's own column names). Three arms,
every one scored by the same code on what it actually returned:

* ``rules``: DCLab's rules alone (``dclab_core.understand`` and the family the words name).
* ``ai_harnessed``: ``draft.no_data.solve_without_data`` (the roles and its gates when a gateway is given; its rules floor when none).
* ``plain``: one model request with a neutral prompt and no harness. Measured only when a model answers: with no model the
  arm is "not measured" (never a constant, never a canned reply).

The first version of this file (gate v1) could not fail: the plain arm's two scores were constants, its fallback reply was a
canned text with invented numbers, two of the harness's "passed" values were constants, and the domain words were the
template's own columns (the review of 22ad1bf8). Nothing here is a constant now, and a check that was not measured is
reported as ``None``, not as passed.

    python -m dclab_rnd.agent_eval.no_data                      # free: the rules and the harness's floor; nothing is sent
    python -m dclab_rnd.agent_eval.no_data --live --yes --cap-eur 0.6   # the roles and the plain arm, kept under agent_runs/live_tests/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from dclab_core import understand as U

from ... import cited, prompts
from ...draft import no_data

ARMS = ("rules", "ai_harnessed", "plain")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "evidence" / "campaigns" / "no_data_v1" / "results"
KEEP_ROOT = ROOT / "agent_runs" / "live_tests"
PLAIN_PROMPT = "no_data_eval_plain"
SAME_FAMILY = ({"classification", "binary", "prediction"},)  # names of one family
UNDERSTOOD = ("user_stated", "ai_inferred", "rules_read")  # a field somebody gave; a DCLab default is not understanding

GATE: dict[str, Any] = {
    "id": "no_data_gate_v2",
    "min_family_right_rate": 0.9,
    "min_economics_inputs_exact_rate": 0.9,
    "max_ungrounded_number_rate": 0.0,
    "min_workflow_and_feature_safety_rate": 1.0,
    "max_mean_cost_eur": 0.15,
}


def load_cases(split: str = "dev") -> list[dict[str, Any]]:
    return json.loads((HERE / f"cases_{split}.json").read_text(encoding="utf-8"))


def family_right(got: Any, expected: Any) -> bool:
    got, expected = str(got or "").lower(), str(expected or "").lower()
    return got == expected or any(got in group and expected in group for group in SAME_FAMILY)


def recall(text: str, case: Mapping[str, Any]) -> float:
    """The share of the case's domain words found in what an arm returned."""
    words = [str(k).lower() for k in case.get("domain_keywords") or []]
    blob = str(text or "").lower()
    return round(sum(1 for w in words if w in blob) / max(len(words), 1), 4)


def _score_rules_arm(case: Mapping[str, Any]) -> dict[str, Any]:
    problem = str(case["problem"])
    brief = U.understand(problem)
    core = (brief.goal, brief.action, brief.unit, brief.prediction_moment, brief.horizon if brief.horizon.known else brief.target_definition, brief.primary_metric)
    family, source = no_data._family_of(problem)
    schema = no_data._virtual_schema_for_kind(family, problem)
    return {"id": case["id"], "arm": "rules", "measured": True,
            "understanding_completeness": round(sum(1 for f in core if f.known and f.value) / len(core), 4),
            "family": family, "family_source": source, "family_right": family_right(family, case.get("expected_family")),
            "domain_recall": recall(json.dumps(schema, ensure_ascii=False), case),
            "ungrounded_numbers": 0, "workflow_and_feature_safe": None, "cost_eur": 0.0}


def _score_ai_harnessed_arm(case: Mapping[str, Any], *, gateway: Any = None, cap_eur: float = 0.12) -> dict[str, Any]:
    problem = str(case["problem"])
    sol = no_data.solve_without_data(problem, gateway=gateway, models=gateway, draft_id=f"eval-{case['id']}", cap_eur=cap_eur)
    und = sol.get("understanding") or {}
    fields = [f for f in (und.get("fields") or {}).values() if isinstance(f, Mapping)]
    ds = sol.get("data_science") or {}
    report = sol.get("harness_report") or {}
    gates = report.get("gates") or {}
    inputs = (sol.get("economics") or {}).get("inputs") or {}
    expected = case.get("expected_inputs") or {}
    shown = json.dumps({"tables": ds.get("virtual_tables"), "tier1": ds.get("tier1_minimum_features"), "tier2": ds.get("tier2_high_leverage_features"),
                        "forbidden": ds.get("forbidden_leakage_columns"), "programs": ds.get("verified_feature_programs"), "understanding": und},
                       ensure_ascii=False)
    return {"id": case["id"], "arm": "ai_harnessed", "measured": True, "mode": report.get("mode"), "roles_called": report.get("roles_called") or [],
            "understanding_completeness": round(sum(1 for f in fields if f.get("status") in UNDERSTOOD and f.get("value")) / max(len(fields), 1), 4),
            "defaults": sum(1 for f in fields if f.get("status") == "dclab_default"),
            "family": sol.get("family"), "family_source": sol.get("family_source"), "family_right": family_right(sol.get("family"), case.get("expected_family")),
            "fits": bool((sol.get("fit") or {}).get("fits")),
            "domain_recall": recall(shown, case),
            "economics_inputs_exact": all(abs(float(inputs.get(k) or 0.0) - float(v)) <= 1e-2 for k, v in expected.items()) if expected else None,
            "assumed_inputs": len((sol.get("economics") or {}).get("assumed") or []),
            "ungrounded_numbers": int((gates.get("gate_2_number_grounding") or {}).get("ungrounded_numbers_remaining") or 0),
            "workflow_and_feature_safe": bool((gates.get("gate_3_virtual_schema_and_features") or {}).get("passed")
                                              and (gates.get("gate_4_workflow_and_debate") or {}).get("passed")),
            "gates_passed": {name: bool(g.get("passed")) for name, g in gates.items()},
            "cost_eur": round(float(report.get("cost_eur") or 0.0), 6)}


def _score_plain_arm(case: Mapping[str, Any], *, gateway: Any = None) -> dict[str, Any]:
    """One model request, a neutral prompt, no harness. With no model: not measured."""
    row: dict[str, Any] = {"id": case["id"], "arm": "plain", "measured": False, "understanding_completeness": None, "family_right": None,
                           "domain_recall": None, "ungrounded_numbers": None, "workflow_and_feature_safe": None, "cost_eur": 0.0}
    client = gateway.client("exam_plain", draft_id=f"exam:no-data-plain-{case['id']}") if gateway is not None else None
    if client is None:
        return row
    try:
        client.request_cells = False  # a written case: no table, no cell
    except AttributeError:
        pass
    before = float(getattr(client, "spent_eur", 0.0) or 0.0)
    out = client.complete([{"role": "system", "content": prompts.text(PLAIN_PROMPT)}, {"role": "user", "content": str(case["problem"])}], None, max_tokens=6000)
    reply = str((out or {}).get("content") or "")
    row["cost_eur"] = round(max(0.0, float(getattr(client, "spent_eur", 0.0) or 0.0) - before), 6)
    if not reply.strip():
        return row
    return {**row, "measured": True, "domain_recall": recall(reply, case), "ungrounded_numbers": len(cited.unsaid(reply, [str(case["problem"])])),
            "reply_chars": len(reply)}


def _rate(rows: Sequence[Mapping[str, Any]], key: str, test: Any = bool) -> Optional[float]:
    kept = [r for r in rows if r.get("measured") and r.get(key) is not None]
    return round(sum(1 for r in kept if test(r[key])) / len(kept), 4) if kept else None


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    kept = [float(r[key]) for r in rows if r.get("measured") and r.get(key) is not None]
    return round(sum(kept) / len(kept), 4) if kept else None


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"cases": len(rows), "measured": sum(1 for r in rows if r.get("measured")),
            "mean_understanding_completeness": _mean(rows, "understanding_completeness"),
            "family_right_rate": _rate(rows, "family_right"),
            "mean_domain_recall": _mean(rows, "domain_recall"),
            "economics_inputs_exact_rate": _rate(rows, "economics_inputs_exact"),
            "ungrounded_number_rate": _rate(rows, "ungrounded_numbers", lambda n: int(n) > 0),
            "workflow_and_feature_safety_rate": _rate(rows, "workflow_and_feature_safe"),
            "mean_cost_eur": round(sum(float(r.get("cost_eur") or 0.0) for r in rows) / max(len(rows), 1), 6),
            "total_cost_eur": round(sum(float(r.get("cost_eur") or 0.0) for r in rows), 6)}


def _at_least(value: Optional[float], floor: float) -> Optional[bool]:
    return None if value is None else value >= floor


def _at_most(value: Optional[float], cap: float) -> Optional[bool]:
    return None if value is None else value <= cap


def verdict(summaries: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Any] = GATE) -> dict[str, Any]:
    """The gate read on the harnessed arm. A check with nothing measured is ``None``; the gate passes only when every check
    is ``True`` (an unmeasured check never counts as passed)."""
    ai = summaries.get("ai_harnessed") or {}
    checks = {"family_right": _at_least(ai.get("family_right_rate"), gate["min_family_right_rate"]),
              "economics_inputs_exact": _at_least(ai.get("economics_inputs_exact_rate"), gate["min_economics_inputs_exact_rate"]),
              "zero_ungrounded_numbers": _at_most(ai.get("ungrounded_number_rate"), gate["max_ungrounded_number_rate"]),
              "workflow_and_feature_safety": _at_least(ai.get("workflow_and_feature_safety_rate"), gate["min_workflow_and_feature_safety_rate"]),
              "within_cost_cap": _at_most(ai.get("mean_cost_eur"), gate["max_mean_cost_eur"])}
    return {"gate_id": gate["id"], "passed": all(v is True for v in checks.values()), "checks": checks}


def evaluate(*, split: str = "dev", arms: Sequence[str] = ARMS, gateway: Any = None, cap_eur: float = 0.80, limit: int = 0,
             cases: Optional[Sequence[Mapping[str, Any]]] = None) -> dict[str, Any]:
    cases = list(cases) if cases is not None else load_cases(split)
    if limit and limit > 0:
        cases = cases[:limit]
    per_case_cap = round(cap_eur / max(len(cases), 1), 4)
    by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    spent = 0.0
    for c in cases:
        for arm in arms:
            left = max(0.0, cap_eur - spent)
            live = gateway if left > 0.01 else None
            if arm == "rules":
                row = _score_rules_arm(c)
            elif arm == "ai_harnessed":
                row = _score_ai_harnessed_arm(c, gateway=live, cap_eur=min(per_case_cap, left))
            else:
                row = _score_plain_arm(c, gateway=live)
            spent += float(row.get("cost_eur") or 0.0)
            by_arm[arm].append(row)
    summaries = {arm: summarize(rows) for arm, rows in by_arm.items()}
    # what the harness adds over the rules and over the model alone, each only where both sides were measured
    ai, rules, plain = (summaries.get(a) or {} for a in ("ai_harnessed", "rules", "plain"))

    def gap(a: Mapping[str, Any], b: Mapping[str, Any], key: str) -> Optional[float]:
        return None if a.get(key) is None or b.get(key) is None else round(float(a[key]) - float(b[key]), 4)

    compare = {"understanding_minus_rules": gap(ai, rules, "mean_understanding_completeness"),
               "domain_recall_minus_rules": gap(ai, rules, "mean_domain_recall"),
               "domain_recall_minus_plain": gap(ai, plain, "mean_domain_recall"),
               "ungrounded_rate_minus_plain": gap(ai, plain, "ungrounded_number_rate")}
    return {**verdict(summaries), "split": split, "live": gateway is not None, "spent_eur": round(spent, 6), "summaries": summaries, "compare": compare, "rows": by_arm}


__all__ = ["ARMS", "GATE", "evaluate", "family_right", "load_cases", "summarize", "verdict"]
