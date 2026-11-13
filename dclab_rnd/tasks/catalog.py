"""P2–P6 runtime: code-owned typed task execution, evidence grounding, router, and bounded re-planning."""
from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

from dclab_core.solution_workflow import FAMILIES
from dclab_core.task_catalog import (
    CATALOG,
    CatalogTaskCall,
    IdentifiedRisk,
    OpenUnknown,
    ProblemDecomposition,
    SubproblemSpec,
    validate_catalog_plan,
    validate_decomposition,
)
from dclab_rnd.tools import POST_OUTCOME_NAME, search_evidence

VERSION = "tp08.catalog.v1"

_SNAPSHOT_OR_OUTCOME_WORDS = re.compile(
    r"(last_order_date|lifetime_orders|payout_issued|closure_reason|final_balance|investigation_outcome|"
    r"resolution_hours|final_refund|returned_flag|chargeback_flag|escalated_label|queue_label)",
    re.I,
)
_TIME_WORDS = re.compile(r"(?:^|_)(time|date|at)$|cutoff|cohort_time|dispatched_at|ordered_at|created_at|recorded_at|measured_at|listed_at|shipped_at|filed_at|originated_at", re.I)
_JOIN_MULTI_TABLES = re.compile(r"(orders|tickets|addresses|sessions|visits|tags|contracts|events|line_items|customer_stats)", re.I)


def grounding_context(brief: str, tables: Mapping[str, list[str]]) -> dict[str, dict[str, Any]]:
    """Retrieve grounded DCLab pitfalls, rules, and precedents for P5 plan justification."""
    cols = " ".join(c for clist in tables.values() for c in clist)
    queries = [brief[:160], f"leakage time split join {cols[:160]}"]
    sources: dict[str, dict[str, Any]] = {}
    for q in queries:
        try:
            res = search_evidence(q, k=4)
        except Exception:
            continue
        for item in res.get("results", []):
            rid = item.get("record_id")
            if rid and rid not in sources:
                sources[rid] = {"record_id": rid, "title": item.get("title", ""), "text": item.get("text", "")}
    return sources


def needs_model_planning(brief: str, tables: Mapping[str, list[str]], known_facts: Mapping[str, Any]) -> bool:
    """P6 router: clean, unambiguous single-table regression/binary stays fast/free; complex/ambiguous routes to model."""
    if known_facts.get("clean_single_table"):
        return False
    if len(tables) > 1:
        return True
    if not known_facts.get("target") or known_facts.get("horizon_days") is None or not known_facts.get("unit"):
        return True
    if known_facts.get("treatment") or known_facts.get("cost_ratio") or known_facts.get("costs") or known_facts.get("daily_review_cap") or known_facts.get("positive_rate") is not None:
        return True
    if known_facts.get("time_column") or known_facts.get("entity_key"):
        return True
    cols = [c for clist in tables.values() for c in clist]
    if any(POST_OUTCOME_NAME.search(c) or _SNAPSHOT_OR_OUTCOME_WORDS.search(c) or _TIME_WORDS.search(c) for c in cols):
        return True
    if re.search(r"\b(uplift|segment\w*|persona\w*|voucher|offer|call vs email|imbalance|rare|counterfeit|fraud|downtime|sensor\w*|telemetry|asymmetric)\b", brief, re.I):
        return True
    return False


def catalog_specs_summary() -> dict[str, dict[str, Any]]:
    """Compact schema description of allowed analysis/planning catalog tasks for prompt grounding."""
    return {
        name: {
            "req": list(spec.required_params),
            "opt": [p for p in spec.allowed_params if p not in spec.required_params],
        }
        for name, spec in CATALOG.items()
        if spec.phase in ("understand", "prepare", "model") and name not in ("data", "leakage", "features", "models")
    }


def _find_col(col: str, frames: Mapping[str, Any] | None, table_hint: str | None = None) -> tuple[str | None, Any]:
    if not frames or not col:
        return None, None
    if table_hint and table_hint in frames and hasattr(frames[table_hint], "columns") and col in frames[table_hint].columns:
        return table_hint, frames[table_hint]
    for tname, df in frames.items():
        if hasattr(df, "columns") and col in df.columns:
            return tname, df
    return None, None


def execute_catalog_task(
    call: CatalogTaskCall,
    *,
    tables: Mapping[str, list[str]],
    known_facts: Mapping[str, Any] | None = None,
    frames: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic code handler for a typed catalog task; inspects schemas/frames on training data only."""
    facts = dict(known_facts or {})
    findings: list[str] = []
    traps: list[str] = []
    followup: list[CatalogTaskCall] = []
    tt = call.task_type
    p = call.params

    if tt == "define_target":
        candidates = [c for cols in tables.values() for c in cols if re.search(r"(churn|return|chargeback|escalat|queue|disput|drop|cancel|stayed|bought|fraud|label)", c, re.I)]
        if facts.get("target") is None:
            findings.append(f"target_unresolved:{candidates}")
            traps.append("unresolved_target_definition")
            followup.append(CatalogTaskCall(task_type="ask_owner", params={"field": "target", "question": f"Confirm target definition (candidates: {candidates or 'event window'})", "blocking": True}, reason="Unconfirmed target"))
    elif tt == "confirm_unit_and_horizon":
        missing_unit = ("unit" in facts and not facts.get("unit"))
        missing_horizon = ("horizon_days" in facts and facts.get("horizon_days") is None)
        if missing_unit or missing_horizon:
            findings.append("missing_unit_or_horizon")
            if missing_horizon:
                traps.append("missing_horizon")
            if missing_unit:
                traps.append("ambiguous_unit")
            followup.append(CatalogTaskCall(task_type="ask_owner", params={"field": "horizon", "question": "Confirm prediction unit and horizon in days.", "blocking": True}, reason="Missing unit or horizon"))
    elif tt == "profile_column":
        col = str(p.get("column") or "")
        t_hint = p.get("table")
        tname, df = _find_col(col, frames, t_hint)
        if POST_OUTCOME_NAME.search(col) or _SNAPSHOT_OR_OUTCOME_WORDS.search(col):
            findings.append(f"profile_suspect_snapshot_or_outcome:{col}")
            traps.append(f"post_outcome_{col}")
            followup.append(CatalogTaskCall(task_type="leakage_probe", params={"table": tname or t_hint, "column": col, "reason": "Verify snapshot/post-outcome timing"}, reason="Profile flagged suspect column"))
        elif df is not None and len(df) > 0:
            s = df[col]
            nunique = int(s.nunique(dropna=True))
            miss = float(s.isna().mean())
            findings.append(f"profile:{tname}.{col}:nunique={nunique}:miss={miss:.3f}")
            if col.endswith("_id") and nunique >= max(10, int(0.8 * len(df))):
                traps.append(f"identifier_{col}")
    elif tt == "check_join_key":
        tname = p.get("table")
        key = p.get("key")
        if frames and tname in frames and key in frames[tname]:
            df = frames[tname]
            max_mult = int(df.groupby(key).size().max()) if len(df) else 1
            if max_mult > 1:
                findings.append(f"join_duplication:{tname}.{key}:max_{max_mult}")
                traps.append(f"{tname}_one_to_many_duplication")
                followup.append(CatalogTaskCall(task_type="grouped_split", params={"table": tname, "entity": key, "reason": "Avoid entity leak across joined rows"}, reason="1:many join detected"))
        elif tname and _JOIN_MULTI_TABLES.search(tname):
            findings.append(f"multi_row_table:{tname}")
            traps.append(f"{tname}_duplication_or_cutoff_risk")
            followup.append(CatalogTaskCall(task_type="leakage_probe", params={"table": tname, "column": key, "reason": "Verify pre-cutoff aggregation"}, reason="Multi-table key check"))
    elif tt == "time_split_check":
        col = str(p.get("column") or "")
        _, df = _find_col(col, frames, p.get("table"))
        if df is not None:
            findings.append(f"temporal_column_present:{col}:rows={len(df)}")
        traps.append("time_split_required")
    elif tt == "grouped_split":
        ent = str(p.get("entity") or "")
        _, df = _find_col(ent, frames, p.get("table"))
        if df is not None and len(df) > 0:
            n_ent = int(df[ent].nunique(dropna=True))
            findings.append(f"grouped_holdout_on:{ent}:entities={n_ent}:rows={len(df)}")
            if n_ent < len(df):
                traps.append(f"repeated_entity_leak_{ent}")
        else:
            findings.append(f"grouped_holdout_on:{ent}")
        traps.append(f"grouped_split_{ent}")
    elif tt == "leakage_probe":
        col = str(p.get("column") or "")
        if POST_OUTCOME_NAME.search(col) or _SNAPSHOT_OR_OUTCOME_WORDS.search(col):
            findings.append(f"leaky_column_dropped:{col}")
            traps.append(f"post_outcome_{col}")
        elif col.endswith("_id") or col in ("subscriber_id", "customer_id", "item_id", "claim_id", "loan_id", "order_id", "device_id", "pump_id"):
            findings.append(f"identifier_column_dropped:{col}")
            traps.append(f"identifier_{col}")
        else:
            findings.append(f"probed_column:{col}")
    elif tt == "segment_subproblem":
        kind = str(p.get("kind") or "")
        findings.append(f"subproblem_verified:{kind}")
        if kind == "uplift":
            if facts.get("randomised") is False:
                traps.append("observational_treatment_bias")
            else:
                traps.append("multi_arm_uplift_policy")
            followup.append(CatalogTaskCall(task_type="threshold_for_cost", params={"cost_ratio": facts.get("cost_ratio") or 1.0}, reason="Optimize net uplift value"))
    elif tt == "calibration":
        pos = facts.get("positive_rate")
        if pos is not None and float(pos) < 0.05:
            traps.append("extreme_class_imbalance")
        findings.append("oof_calibration_enabled")
    elif tt == "threshold_for_cost":
        findings.append("cost_or_capacity_threshold_enabled")
        if facts.get("daily_review_cap"):
            traps.append("capacity_constrained_threshold")
        if facts.get("cost_ratio"):
            traps.append("asymmetric_failure_cost")
    elif tt == "ask_owner":
        findings.append(f"pending_owner_question:{p.get('question', '')[:80]}")
    else:
        findings.append(f"checked:{tt}")

    return {"task_type": tt, "findings": findings, "traps": traps, "followup": followup}


def deterministic_catalog_plan(
    brief: str,
    tables: Mapping[str, list[str]],
    known_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Arm B+: deterministic use of the P2 task catalog via explicit structural rules only."""
    facts = dict(known_facts or {})
    calls: list[CatalogTaskCall] = []
    if facts.get("time_column"):
        calls.append(CatalogTaskCall(task_type="time_split_check", params={"column": str(facts["time_column"])}, reason="Declared time column"))
    if facts.get("entity_key"):
        calls.append(CatalogTaskCall(task_type="grouped_split", params={"entity": str(facts["entity_key"])}, reason="Declared entity key"))
    for tname, cols in tables.items():
        for col in cols:
            if POST_OUTCOME_NAME.search(col) or _SNAPSHOT_OR_OUTCOME_WORDS.search(col):
                calls.append(CatalogTaskCall(task_type="leakage_probe", params={"table": tname, "column": col}, reason="Suspect post-outcome column name"))
    if len(tables) > 1:
        # Pick shared ID column if obvious
        shared = set.intersection(*(set(c) for c in tables.values()))
        key = next((c for c in sorted(shared) if c.endswith("_id")), None)
        if key:
            second = list(tables)[1]
            calls.append(CatalogTaskCall(task_type="check_join_key", params={"table": second, "key": key, "parent_table": list(tables)[0]}, reason="Multi-table key check"))
    if facts.get("positive_rate") is not None and float(facts["positive_rate"]) < 0.05:
        calls.append(CatalogTaskCall(task_type="calibration", params={"positive_rate": float(facts["positive_rate"])}, reason="Rare positive rate"))
    if facts.get("cost_ratio") or facts.get("daily_review_cap"):
        calls.append(CatalogTaskCall(task_type="threshold_for_cost", params={"cost_ratio": facts.get("cost_ratio"), "capacity_limit": facts.get("daily_review_cap")}, reason="Cost or capacity constraint"))
    if not facts.get("target"):
        calls.append(CatalogTaskCall(task_type="define_target", params={}, reason="Target missing"))
    calls.append(CatalogTaskCall(task_type="baseline", params={"metric": "roc_auc"}, reason="Baseline model"))
    valid_calls, _ = validate_catalog_plan(calls, tables)
    return {
        "selected_tasks": [c.task_type for c in valid_calls],
        "calls": [c.model_dump(mode="json") for c in valid_calls],
        "asked_owner": any(c.task_type == "ask_owner" for c in valid_calls),
        "mode": "deterministic_catalog",
    }


def _extract_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.S).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    return json.loads(m.group(0) if m else raw)


def plan_with_model(
    brief: str,
    tables: Mapping[str, list[str]],
    known_facts: Mapping[str, Any] | None = None,
    *,
    project_id: str = "tp08_case",
    gateway: Any = None,
    max_revisions: int = 2,
    cap_eur: float = 0.25,
    frames: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """P3->P2->P5->P4->P6 pipeline: router, ProblemDecomposition, grounded catalog plan, and bounded re-plan."""
    t0 = time.monotonic()
    facts = dict(known_facts or {})
    if not needs_model_planning(brief, tables, facts):
        det = deterministic_catalog_plan(brief, tables, facts)
        target = str(facts.get("target") or "target")
        kind = "regression" if "price" in target or "revenue" in target or "duration" in target else "classification"
        return {
            **det,
            "routed_to": "deterministic",
            "subproblems": [kind],
            "caught_traps": [],
            "decomposition": ProblemDecomposition(
                decision=brief[:200],
                target=target,
                unit=str(facts.get("unit") or "row"),
                horizon_days=int(facts.get("horizon_days") or 0),
                success_metric="rmse" if kind == "regression" else "roc_auc",
                subproblems=(SubproblemSpec(id="main", kind=kind, objective=brief[:200], target=target),),
                source="deterministic",
            ).model_dump(mode="json"),
            "citations": [],
            "revisions": 1,
            "extra_prompts": 0,
            "cost_usd": 0.0,
            "seconds": round(time.monotonic() - t0, 4),
        }

    sources = grounding_context(brief, tables)
    from dclab_rnd import experience, prompts

    if gateway is None:
        from dclab_rnd.models import installed

        gateway = installed()
    client = gateway.client("task_decomposer", project_id=project_id)
    if client is None:
        det = deterministic_catalog_plan(brief, tables, facts)
        return {**det, "routed_to": "fallback_no_client", "subproblems": ["classification"], "caught_traps": [], "citations": [], "revisions": 1, "extra_prompts": 0, "cost_usd": 0.0, "seconds": round(time.monotonic() - t0, 4)}

    from dclab_rnd.models.gateway import Bound

    if isinstance(client, Bound):
        client.run_limit_eur = cap_eur

    system = prompts.text("problem_planner")
    top_sources = dict(list(sources.items())[:5])
    payload = {
        "brief": brief,
        "tables": dict(tables),
        "known_facts": facts,
        "catalog_specs": catalog_specs_summary(),
        "available_families": list(FAMILIES),
        "evidence_sources": {k: v["title"] + " — " + v["text"][:140] for k, v in top_sources.items()},
    }
    messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]

    calls_made = 0
    before_spent = float(getattr(client, "spent_eur", 0.0) or 0.0)
    before_req = experience.last_model_request()
    reply: dict[str, Any] = {}
    try:
        reply = client.complete(messages, None, max_tokens=3800, single_attempt=True, timeout_seconds=90.0, require_price=False, max_total_tokens=20000, extra_body={"reasoning": {"max_tokens": 1000}})
        calls_made += 1
        raw_text = str(reply.get("content") or "").strip()
        try:
            raw_doc = _extract_json(raw_text) if raw_text else {}
        except Exception:
            raw_doc = {}
        if (not raw_text or not raw_doc) and calls_made < max_revisions:
            suffix = "\nYour previous JSON had a syntax error. Fix JSON syntax and respond immediately with raw valid JSON only (3-6 calls)." if raw_text else "\nRespond immediately with compact JSON only (3-6 calls, short reasons)."
            retry_msg = [messages[0], {"role": "user", "content": messages[1]["content"] + suffix}]
            reply = client.complete(retry_msg, None, max_tokens=3800, single_attempt=True, timeout_seconds=90.0, require_price=False, max_total_tokens=20000, extra_body={"reasoning": {"max_tokens": 600}})
            calls_made += 1
            raw_text = str(reply.get("content") or "").strip()
            raw_doc = _extract_json(raw_text or "{}")
    except Exception:
        total_cost = round(max(0.0, float(getattr(client, "spent_eur", 0.0) or 0.0) - before_spent), 6)
        det = deterministic_catalog_plan(brief, tables, facts)
        return {**det, "routed_to": "fallback_invalid_json", "refusals": ["invalid_model_reply"], "subproblems": ["classification"], "caught_traps": [], "citations": [], "revisions": 1, "extra_prompts": calls_made, "cost_usd": total_cost, "seconds": round(time.monotonic() - t0, 4)}

    decomp, decomp_issues = validate_decomposition(raw_doc.get("decomposition") or {}, tables)
    valid_calls, plan_issues = validate_catalog_plan(raw_doc.get("calls") or [], tables)
    revisions = 1

    # Bounded P4 schema repair if first non-empty draft tripped a catalog parameter/column check within max_revisions.
    if (decomp_issues or plan_issues or not valid_calls) and calls_made < max_revisions and str(reply.get("content") or "").strip():
        repair_msg = messages + [
            {"role": "assistant", "content": reply.get("content") or ""},
            {"role": "user", "content": json.dumps({"validation_errors": decomp_issues + plan_issues, "instruction": "Fix only the schema/column issues above while keeping valid fields and strictly obeying catalog_specs."}, ensure_ascii=False)},
        ]
        try:
            rep = client.complete(repair_msg, None, max_tokens=3600, single_attempt=True, timeout_seconds=90.0, require_price=False, max_total_tokens=20000, extra_body={"reasoning": {"max_tokens": 800}})
            calls_made += 1
            revisions += 1
            raw_doc = _extract_json(rep.get("content") or "{}")
            decomp, decomp_issues = validate_decomposition(raw_doc.get("decomposition") or {}, tables)
            valid_calls, plan_issues = validate_catalog_plan(raw_doc.get("calls") or [], tables)
            reply = rep
        except Exception:
            pass

    if decomp_issues or not valid_calls or (plan_issues and not valid_calls):
        total_cost = round(max(0.0, float(getattr(client, "spent_eur", 0.0) or 0.0) - before_spent), 6)
        det = deterministic_catalog_plan(brief, tables, facts)
        return {
            **det,
            "routed_to": "fallback_refused",
            "refusals": decomp_issues + plan_issues,
            "subproblems": [s.kind for s in decomp.subproblems] if decomp else ["classification"],
            "caught_traps": [],
            "citations": [],
            "revisions": revisions,
            "extra_prompts": calls_made,
            "cost_usd": total_cost,
            "seconds": round(time.monotonic() - t0, 4),
        }

    # Execute read/understand catalog tasks locally ($0, pre-holdout) and trigger bounded P4 re-plan if evidence uncovers new followups.
    valid_calls, evidence_rows, caught_traps, decomp, calls_made, revisions, rev_req = _run_evidence_and_replan(
        brief=brief,
        tables=tables,
        facts=facts,
        valid_calls=valid_calls,
        decomp=decomp,
        raw_doc=raw_doc,
        reply_content=str(reply.get("content") or ""),
        messages=messages,
        client=client,
        sources=sources,
        calls_made=calls_made,
        revisions=revisions,
        max_revisions=max_revisions,
        frames=frames,
    )

    total_cost = round(max(0.0, float(getattr(client, "spent_eur", 0.0) or 0.0) - before_spent), 6)
    all_cites = sorted({c for call in valid_calls for c in call.citations if c in sources})
    last_req = rev_req or (experience.last_model_request() if experience.last_model_request() != before_req else None)
    return {
        "routed_to": "model",
        "selected_tasks": sorted({c.task_type for c in valid_calls}),
        "calls": [c.model_dump(mode="json") for c in valid_calls],
        "subproblems": sorted({s.kind for s in decomp.subproblems}) if decomp else ["classification"],
        "caught_traps": caught_traps,
        "asked_owner": any(c.task_type == "ask_owner" for c in valid_calls),
        "decomposition": decomp.model_dump(mode="json") if decomp else None,
        "citations": all_cites,
        "revisions": revisions,
        "extra_prompts": calls_made,
        "cost_usd": round(total_cost, 6),
        "model_request": last_req,
        "seconds": round(time.monotonic() - t0, 4),
        "mode": "adaptive_model",
    }


def _collect_traps(
    model_traps: Iterable[str] | list[str],
    decomp: ProblemDecomposition | None,
    evidence_rows: list[dict[str, Any]],
) -> list[str]:
    risk_traps = [r.trap_kind for r in (decomp.risks if decomp else ())]
    risk_cols = [f"{r.trap_kind}_{r.column}" for r in (decomp.risks if decomp else ()) if r.column]
    return sorted({*model_traps, *risk_traps, *risk_cols, *(t for ev in evidence_rows for t in ev["traps"])})


def _run_evidence_and_replan(
    *,
    brief: str,
    tables: Mapping[str, list[str]],
    facts: Mapping[str, Any],
    valid_calls: list[CatalogTaskCall],
    decomp: ProblemDecomposition | None,
    raw_doc: Mapping[str, Any],
    reply_content: str,
    messages: list[dict[str, Any]],
    client: Any,
    sources: Mapping[str, dict[str, Any]],
    calls_made: int,
    revisions: int,
    max_revisions: int,
    frames: Mapping[str, Any] | None,
) -> tuple[list[CatalogTaskCall], list[dict[str, Any]], list[str], ProblemDecomposition | None, int, int, Any]:
    from dclab_rnd import experience

    evidence_rows = [execute_catalog_task(c, tables=tables, known_facts=facts, frames=frames) for c in valid_calls]
    followups = [f for ev in evidence_rows for f in ev["followup"] if f.task_type not in {c.task_type for c in valid_calls}]
    model_traps = list(raw_doc.get("caught_traps") or ())
    rev_req = None

    if followups and revisions < max_revisions and client is not None:
        clean_evidence = [{"task_type": ev["task_type"], "findings": ev["findings"], "traps": ev["traps"]} for ev in evidence_rows]
        rev_msg = messages + [
            {"role": "assistant", "content": reply_content},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "stage_evidence": clean_evidence,
                        "suggested_followups": [f.model_dump(mode="json") for f in followups],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            before_req = experience.last_model_request()
            reply2 = client.complete(
                rev_msg,
                None,
                max_tokens=3000,
                single_attempt=True,
                timeout_seconds=45.0,
                require_price=False,
                max_total_tokens=16000,
                extra_body={"reasoning": {"max_tokens": 800}},
            )
            calls_made += 1
            revisions += 1
            after_req = experience.last_model_request()
            if after_req != before_req:
                rev_req = after_req
            doc2 = _extract_json(reply2.get("content") or "{}")
            rev_calls, rev_issues = validate_catalog_plan(doc2.get("calls") or [], tables)
            if not rev_issues and rev_calls:
                prev_asks = [c for c in valid_calls if c.task_type == "ask_owner"]
                if prev_asks and not any(c.task_type == "ask_owner" for c in rev_calls):
                    rev_calls = [*rev_calls, *prev_asks]
                valid_calls = rev_calls
                evidence_rows = [execute_catalog_task(c, tables=tables, known_facts=facts, frames=frames) for c in valid_calls]
            model_traps.extend(doc2.get("caught_traps") or ())
            if doc2.get("decomposition"):
                d2, d2_issues = validate_decomposition(doc2["decomposition"], tables)
                if d2 and not d2_issues:
                    decomp = d2
        except Exception:
            pass

    caught_traps = _collect_traps(model_traps, decomp, evidence_rows)
    return valid_calls, evidence_rows, caught_traps, decomp, calls_made, revisions, rev_req


def replan_from_evidence(
    brief: str,
    tables: Mapping[str, list[str]],
    known_facts: Mapping[str, Any] | None,
    current_plan: Mapping[str, Any],
    stage_evidence: Mapping[str, Any] | list[dict[str, Any]],
    *,
    project_id: str = "tp08_case",
    gateway: Any = None,
    sources: Mapping[str, dict[str, Any]] | None = None,
    max_revisions: int = 2,
    cap_eur: float = 0.10,
    frames: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bounded P4 evidence re-planning when data/leakage stage findings surface new traps or followups."""
    t0 = time.monotonic()
    revisions = int(current_plan.get("revisions") or 1)
    if revisions >= max_revisions:
        return {"revised": False, "reason": "revision_limit", "plan": dict(current_plan), "cost_usd": 0.0}

    from dclab_rnd import experience, prompts
    from dclab_rnd.models.gateway import Bound

    if gateway is None:
        from dclab_rnd.models import installed

        gateway = installed()
    client = gateway.client("task_decomposer", project_id=project_id) if gateway is not None else None
    if client is None:
        return {"revised": False, "reason": "no_client", "plan": dict(current_plan), "cost_usd": 0.0}
    if isinstance(client, Bound):
        client.run_limit_eur = cap_eur

    src = dict(sources) if sources is not None else grounding_context(brief, tables)
    facts = dict(known_facts or {})
    raw_calls = current_plan.get("calls") or []
    valid_calls = [c if isinstance(c, CatalogTaskCall) else CatalogTaskCall.model_validate(c) for c in raw_calls]
    decomp_raw = current_plan.get("decomposition")
    decomp = None
    if isinstance(decomp_raw, Mapping):
        try:
            decomp = ProblemDecomposition.model_validate(decomp_raw)
        except Exception:
            decomp = None

    system = prompts.text("problem_planner")
    top_sources = dict(list(src.items())[:5])
    payload = {
        "brief": brief,
        "tables": dict(tables),
        "known_facts": facts,
        "catalog_specs": catalog_specs_summary(),
        "available_families": list(FAMILIES),
        "evidence_sources": {k: v["title"] + " — " + v["text"][:140] for k, v in top_sources.items()},
        "current_plan": {
            "decomposition": decomp_raw,
            "calls": [c.model_dump(mode="json") for c in valid_calls],
            "caught_traps": list(current_plan.get("caught_traps") or ()),
        },
        "stage_evidence": stage_evidence,
        "instruction": "Revise calls and/or decomposition ONLY if stage_evidence reveals new leakage traps, drift, join issues, or unmet constraints.",
    }
    messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    before_spent = float(getattr(client, "spent_eur", 0.0) or 0.0)
    before_req = experience.last_model_request()
    revised = False
    model_traps = list(current_plan.get("caught_traps") or ())

    try:
        reply = client.complete(
            messages,
            None,
            max_tokens=3000,
            single_attempt=True,
            timeout_seconds=45.0,
            require_price=False,
            max_total_tokens=16000,
            extra_body={"reasoning": {"max_tokens": 800}},
        )
        doc2 = _extract_json(reply.get("content") or "{}")
        rev_calls, rev_issues = validate_catalog_plan(doc2.get("calls") or [], tables)
        if not rev_issues and rev_calls:
            valid_calls = rev_calls
            revised = True
        if doc2.get("caught_traps"):
            model_traps.extend(str(t) for t in doc2.get("caught_traps") or () if t)
        if doc2.get("decomposition"):
            d2, d2_issues = validate_decomposition(doc2["decomposition"], tables)
            if d2 and not d2_issues:
                decomp = d2
                revised = True
        revisions += 1
    except Exception:
        total_cost = round(max(0.0, float(getattr(client, "spent_eur", 0.0) or 0.0) - before_spent), 6)
        return {"revised": False, "reason": "replan_failed", "plan": dict(current_plan), "cost_usd": total_cost}

    evidence_rows = [execute_catalog_task(c, tables=tables, known_facts=facts, frames=frames) for c in valid_calls]
    caught_traps = _collect_traps(model_traps, decomp, evidence_rows)
    total_cost = round(max(0.0, float(getattr(client, "spent_eur", 0.0) or 0.0) - before_spent), 6)
    all_cites = sorted({*(current_plan.get("citations") or ()), *(c for call in valid_calls for c in call.citations if c in src)})
    after_req = experience.last_model_request()
    updated = {
        **dict(current_plan),
        "selected_tasks": sorted({c.task_type for c in valid_calls}),
        "calls": [c.model_dump(mode="json") for c in valid_calls],
        "subproblems": sorted({s.kind for s in decomp.subproblems}) if decomp else list(current_plan.get("subproblems") or ["classification"]),
        "caught_traps": caught_traps,
        "asked_owner": any(c.task_type == "ask_owner" for c in valid_calls),
        "decomposition": decomp.model_dump(mode="json") if decomp else decomp_raw,
        "citations": all_cites,
        "revisions": revisions,
        "extra_prompts": int(current_plan.get("extra_prompts") or 0) + 1,
        "cost_usd": round(float(current_plan.get("cost_usd") or 0.0) + total_cost, 6),
        "model_request": after_req if after_req != before_req else current_plan.get("model_request"),
        "seconds": round(float(current_plan.get("seconds") or 0.0) + (time.monotonic() - t0), 4),
    }
    return {"revised": revised, "plan": updated, "cost_usd": total_cost, "model_request": updated.get("model_request")}


def project_catalog_inputs(projects: Any, project: Mapping[str, Any]) -> tuple[str, dict[str, list[str]], dict[str, Any], dict[str, list[str]], dict[str, Any]]:
    """Extract (brief, tables, known_facts, router_tables, frames) from a Studio project without touching holdout."""
    from dclab_rnd.studio import data as studio_data

    pid = str(project.get("id") or "")
    solution = dict(project.get("solution") or {})
    brief = str(project.get("goal") or project.get("name") or pid).strip()

    main_name = Path(str((project.get("data") or {}).get("filename") or "main.csv")).stem or "main"
    tables: dict[str, list[str]] = {}
    frames: dict[str, Any] = {}

    raw_tables = project.get("tables")
    if isinstance(raw_tables, Mapping):
        for tname, tinfo in raw_tables.items():
            if isinstance(tinfo, list):
                tables[str(tname)] = [str(c) for c in tinfo]
            elif isinstance(tinfo, Mapping) and isinstance(tinfo.get("columns"), list):
                tables[str(tname)] = [str(c) for c in tinfo["columns"]]

    data_meta = project.get("data") or {}
    cols = [str(c.get("name") if isinstance(c, Mapping) else c) for c in (data_meta.get("columns") or []) if c]
    if pid and data_meta.get("filename"):
        try:
            path = studio_data.data_path(projects, dict(project))
            df = studio_data.load_table(path)
            if not cols:
                cols = [str(c) for c in df.columns]
            # Keep pre-holdout training slice / cap rows for fast schema+profile grounding only.
            frames[main_name] = df.head(min(len(df), 500)).copy()
        except Exception:
            pass
    if cols and main_name not in tables:
        tables = {main_name: cols, **tables}
    if not tables:
        tables = {main_name: []}

    hints = dict(project.get("catalog_hints") or {})
    if isinstance(hints.get("tables"), Mapping):
        for tname, tcols in hints["tables"].items():
            if isinstance(tcols, list) and str(tname) not in tables:
                tables[str(tname)] = [str(c) for c in tcols]
    known_handled = {
        *(str(x.get("column") if isinstance(x, Mapping) else x) for x in (solution.get("forbidden") or [])),
        *(str(x) for x in (solution.get("identifiers") or [])),
    }
    router_tables = {t: [c for c in clist if c not in known_handled] for t, clist in tables.items()}

    target = solution.get("target")
    unit = hints.get("unit") or solution.get("group_column") or ("row" if target else None)
    horizon = hints.get("horizon_days")
    if horizon is None and target and not hints.get("prediction_moment_unresolved"):
        horizon = 30 if solution.get("time_column") else 0

    facts: dict[str, Any] = {
        "target": target,
        "task": solution.get("task"),
        "unit": unit,
        "horizon_days": horizon,
    }
    for key in ("time_column", "metric", "positive_label", "kind"):
        if solution.get(key) is not None:
            facts[key] = solution.get(key)
    if solution.get("capacity_k") is not None:
        facts["daily_review_cap"] = solution["capacity_k"]
    for key in ("treatment", "cost_ratio", "costs", "daily_review_cap", "positive_rate", "clean_single_table", "time_column"):
        if hints.get(key) is not None:
            facts[key] = hints[key]
    if solution.get("group_column"):
        facts["entity_key"] = solution.get("group_column")
    if solution.get("positive_rate") is not None:
        facts["positive_rate"] = solution.get("positive_rate")
    elif target and main_name in frames and target in frames[main_name].columns:
        try:
            s = frames[main_name][target].dropna()
            if len(s) >= 20 and s.nunique() == 2:
                pos_val = solution.get("positive_label")
                rate = float((s.astype(str) == str(pos_val)).mean()) if pos_val is not None else float(pd_numeric_pos_rate(s))
                if rate < 0.05 or rate > 0.95:
                    facts["positive_rate"] = round(min(rate, 1.0 - rate), 4)
        except Exception:
            pass
    return brief, tables, facts, router_tables, frames


def pd_numeric_pos_rate(series: Any) -> float:
    vals = sorted(series.unique().tolist())
    return float((series == vals[-1]).mean()) if vals else 0.5


def compile_catalog_to_task_graph(
    project_id: str,
    workflow: Any,
    decomp: ProblemDecomposition | Mapping[str, Any],
    calls: list[CatalogTaskCall | Mapping[str, Any]],
    *,
    source: str = "model",
) -> Any:
    """Compile a ProblemDecomposition + validated CatalogTaskCalls into a TP-01B/C signed TaskGraph."""
    from dclab_core.task_fingerprints import graph_fingerprint, metadata_fingerprint, workflow_fingerprint
    from dclab_core.task_graph import AcceptanceCheck, SourceReference, TaskEdge, TaskGraph, TaskNode, TaskProvenance

    parsed_decomp = decomp if isinstance(decomp, ProblemDecomposition) else ProblemDecomposition.read(decomp)
    parsed_calls = [c if isinstance(c, CatalogTaskCall) else CatalogTaskCall.model_validate(c) for c in calls]
    wf_fp = workflow.fingerprint if getattr(workflow, "fingerprint", None) else workflow_fingerprint(workflow)
    sol_fp = metadata_fingerprint(parsed_decomp.model_dump(mode="json"))

    nodes: list[TaskNode] = [
        TaskNode(
            id="root",
            title=f"Solution plan ({parsed_decomp.subproblems[0].kind})",
            objective=parsed_decomp.decision[:240],
            kind="group",
            source_step_ids=tuple(getattr(s, "id", "data") for s in getattr(workflow, "steps", ())[:4]),
        )
    ]
    edges: list[TaskEdge] = []
    prev_id: str | None = None
    for idx, call in enumerate(parsed_calls, 1):
        spec = CATALOG[call.task_type]
        nid = f"task_{idx}_{call.task_type}"
        kind = "gate" if call.task_type in ("ask_owner", "holdout") else "verification"
        check = AcceptanceCheck(
            id=f"chk_{idx}_{call.task_type}",
            kind="approval" if kind == "gate" else "workflow_criterion",
            description=(call.reason or spec.description)[:240],
            evidence_refs=(),
        )
        deps = (prev_id,) if prev_id else ()
        nodes.append(
            TaskNode(
                id=nid,
                parent_id="root",
                title=f"{call.task_type}",
                objective=(call.reason or spec.description)[:300],
                kind=kind,
                dependencies=deps,
                acceptance_checks=(check,),
            )
        )
        edges.append(TaskEdge(source="root", target=nid, kind="grouping"))
        if prev_id:
            edges.append(TaskEdge(source=prev_id, target=nid, kind="dependency"))
        prev_id = nid

    draft_graph = TaskGraph(
        schema_version="dclab_core.task_graph.v1",
        graph_id=f"catalog_{re.sub(r'[^A-Za-z0-9_-]', '_', project_id)[:40]}",
        revision=1,
        objective=parsed_decomp.decision[:240],
        source_refs=(SourceReference(kind="solution_workflow", id="workflow", fingerprint=wf_fp),),
        workflow_fingerprint=wf_fp,
        solution_fingerprint=sol_fp,
        fingerprint="0" * 64,
        nodes=tuple(nodes),
        edges=tuple(edges),
        provenance=TaskProvenance(
            method="model" if source == "model" else "deterministic",
            reference_id="problem_planner",
            producer_version=VERSION,
        ),
    )
    fp = graph_fingerprint(draft_graph)
    return draft_graph.model_copy(update={"fingerprint": fp})


__all__ = [
    "catalog_specs_summary",
    "compile_catalog_to_task_graph",
    "grounding_context",
    "needs_model_planning",
    "execute_catalog_task",
    "deterministic_catalog_plan",
    "plan_with_model",
    "project_catalog_inputs",
    "replan_from_evidence",
]
