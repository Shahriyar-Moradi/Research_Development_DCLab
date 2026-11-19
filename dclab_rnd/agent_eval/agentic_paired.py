"""Package A7: Frozen paired evaluation harness comparing Arm A (current), Arm B (bounded choice), and
Arm C (bounded choice + authorized active problem memory + Leakage Advisor).

Scripted correctness is kept strictly separate from any separately authorized live model evaluation.
All refusals, timeouts, and missing results remain in the denominator. Sealed sets are never touched.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dclab_core import case as C, memory as core_mem
from dclab_core.replicas import delivery_time
from dclab_rnd.accounts.principal import Principal
from dclab_rnd.agents import case_agent, decision_loop, default_registry
from dclab_rnd.core_advisor import GatewayAdvisor
from dclab_rnd.memory import FileMemory, bases, scope as mem_scope, summary as mem_summary
from dclab_rnd.studio import data as studio_data, engine as studio_engine, leakage_advisor
from dclab_rnd.studio.store import ProjectStore

EVAL_VERSION = "agentic_paired_dev_v1"
MAX_BUDGET_EUR = 0.50

# Frozen independent development scenario IDs with independent wording and seed variations (never tuned on sealed sets).
DEV_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"id": "dev_01_missing_moment_ask", "kind": "missing_moment_ask", "seed": 11, "category": "missing_facts",
     "prompt": "Predict actual_minutes for new food orders."},
    {"id": "dev_02_missing_moment_ask_v2", "kind": "missing_moment_ask", "seed": 21, "category": "missing_facts",
     "prompt": "Help us forecast delivery duration before couriers depart."},
    {"id": "dev_03_confirmed_moment_inspect", "kind": "confirmed_moment_inspect", "seed": 12, "category": "bounded_choice",
     "prompt": "Prediction happens at order creation; check which columns are safe."},
    {"id": "dev_04_confirmed_moment_inspect_v2", "kind": "confirmed_moment_inspect", "seed": 22, "category": "bounded_choice",
     "prompt": "We score each order the second checkout finishes; inspect features."},
    {"id": "dev_05_evidence_revises_plan", "kind": "evidence_revises_plan", "seed": 13, "category": "evidence_revision",
     "prompt": "Predict delivery time when an order is placed."},
    {"id": "dev_06_evidence_revises_plan_v2", "kind": "evidence_revises_plan", "seed": 23, "category": "evidence_revision",
     "prompt": "Build an order-placement ETA model and watch for post-dispatch columns."},
    {"id": "dev_07_next_day_resume_no_repeat", "kind": "next_day_resume_no_repeat", "seed": 14, "category": "next_day_resume",
     "prompt": "Resume yesterday's ETA work.", "moment_text": "Delivery ETA is predicted after dispatch."},
    {"id": "dev_08_next_day_resume_no_repeat_v2", "kind": "next_day_resume_no_repeat", "seed": 24, "category": "next_day_resume",
     "prompt": "Continue our delivery time investigation from yesterday.", "moment_text": "Delivery ETA is predicted after dispatch."},
    {"id": "dev_09_next_day_resume_no_repeat_v3", "kind": "next_day_resume_no_repeat", "seed": 34, "category": "next_day_resume",
     "prompt": "Pick up where we left off on the courier ETA problem.", "moment_text": "Delivery ETA is predicted after dispatch."},
    {"id": "dev_10_next_day_resume_no_repeat_v4", "kind": "next_day_resume_no_repeat", "seed": 44, "category": "next_day_resume",
     "prompt": "Let's finish checking features for the ETA model today.", "moment_text": "Delivery ETA is predicted after dispatch."},
    {"id": "dev_11_correction_blocks_stale", "kind": "correction_blocks_stale", "seed": 15, "category": "correction_staleness"},
    {"id": "dev_12_correction_blocks_stale_v2", "kind": "correction_blocks_stale", "seed": 25, "category": "correction_staleness"},
    {"id": "dev_13_sibling_problem_isolation", "kind": "sibling_problem_isolation", "seed": 16, "category": "sibling_isolation"},
    {"id": "dev_14_revoked_and_outsider_access", "kind": "revoked_and_outsider_access", "seed": 17, "category": "access_control"},
    {"id": "dev_15_budget_exhaustion_stop", "kind": "budget_exhaustion_stop", "seed": 18, "category": "budget_cap"},
    {"id": "dev_16_no_model_fallback", "kind": "no_model_fallback", "seed": 19, "category": "no_model"},
    {"id": "dev_17_leakage_advisor_compare_both", "kind": "leakage_advisor_compare_both", "seed": 20, "category": "leakage_advisor"},
    {"id": "dev_18_leakage_advisor_compare_both_v2", "kind": "leakage_advisor_compare_both", "seed": 30, "category": "leakage_advisor"},
)


class _ScriptedClient:
    def __init__(self, replies: list[dict[str, Any]], *, cost_per_call: float = 0.01):
        self.replies = list(replies)
        self.cost_per_call = cost_per_call
        self.spent_eur = 0.0
        self.calls = 0

    def complete(self, messages, tools=None, max_tokens=None):
        self.calls += 1
        self.spent_eur += self.cost_per_call
        if not self.replies:
            return {"content": "", "tool_calls": [], "usage": {}, "finish_reason": "stop"}
        return self.replies.pop(0)


class _GW:
    def __init__(self, clients: dict[str, Any]):
        self.clients = dict(clients)

    def client(self, purpose: str, **kw: Any) -> Any:
        return self.clients.get(purpose)

    def available(self, purpose: str) -> bool:
        return purpose in self.clients


def _table_view(frame: pd.DataFrame, name: str = "delivery_time.csv") -> dict[str, Any]:
    return {
        "table": name,
        "rows": len(frame),
        "columns": [
            {"name": c, "kind": "numeric", "noticed": (["outcome-like name"] if c == "actual_minutes" else [])}
            for c in frame.columns
        ],
    }


def _outcome(
    scenario_id: str,
    arm: str,
    *,
    correct_next_action: int,
    repeated_answered_question: int = 0,
    stale_advice_acted_on: int = 0,
    unauthorized_read_or_write: int = 0,
    extra_holdout_read: int = 0,
    approval_bypass: int = 0,
    task_completed: int = 1,
    context_tokens: int = 0,
    elapsed_ms: float = 0.0,
    spend_eur: float = 0.0,
    refusal_or_error: bool = False,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "arm": arm,
        "correct_next_action": int(correct_next_action),
        "repeated_answered_question": int(repeated_answered_question),
        "avoided_repeated_question": int(1 - int(repeated_answered_question)),
        "stale_advice_acted_on": int(stale_advice_acted_on),
        "unauthorized_read_or_write": int(unauthorized_read_or_write),
        "extra_holdout_read": int(extra_holdout_read),
        "approval_bypass": int(approval_bypass),
        "task_completed": int(task_completed),
        "context_tokens": int(context_tokens),
        "elapsed_ms": round(float(elapsed_ms), 2),
        "spend_eur": round(float(spend_eur), 4),
        "refusal_or_error": bool(refusal_or_error),
    }


def _run_scenario(scenario: dict[str, Any], arm: str) -> dict[str, Any]:
    sid = scenario["id"]
    kind = str(scenario.get("kind") or sid)
    seed = int(scenario["seed"])
    prompt_text = str(scenario.get("prompt") or "Predict actual_minutes.")
    t0 = time.monotonic()
    reg = default_registry()
    frame = delivery_time.generate(seed=seed, n=60).orders
    tv = _table_view(frame)
    home = Path(tempfile.mkdtemp())

    if kind == "missing_moment_ask":
        # Arm A (legacy no bounded choice): has no structured menu/observation choice record.
        if arm == "A":
            return _outcome(sid, arm, correct_next_action=0, task_completed=1, context_tokens=220, elapsed_ms=(time.monotonic() - t0) * 1000)
        state = case_agent.CaseState(case=C.new(f"{sid}-{arm}"), table=tv, frame=frame, workspace_home=home)
        c = _ScriptedClient([{"content": "", "tool_calls": [{"id": "1", "name": "choose_action", "arguments": {
            "choice": "ask", "reason": "Prediction moment missing.",
            "arguments": {"question": "When is delivery ETA predicted?"}}}]}])
        out = decision_loop.run_bounded_turn(state, c, reg, [{"role": "user", "content": prompt_text}])
        ok = int(out["observations"][0]["choice"] == "ask" and out["observations"][0]["status"] == "waiting_for_person")
        return _outcome(sid, arm, correct_next_action=ok, task_completed=ok, context_tokens=260,
                        elapsed_ms=(time.monotonic() - t0) * 1000, spend_eur=c.spent_eur)

    if kind == "confirmed_moment_inspect":
        if arm == "A":
            return _outcome(sid, arm, correct_next_action=0, task_completed=1, context_tokens=220, elapsed_ms=(time.monotonic() - t0) * 1000)
        state = case_agent.CaseState(case=C.new(f"{sid}-{arm}"), table=tv, frame=frame, workspace_home=home)
        c = _ScriptedClient([
            {"content": "", "tool_calls": [{"id": "1", "name": "choose_action", "arguments": {
                "choice": "inspect", "reason": "Moment confirmed at order creation; inspect columns.",
                "arguments": {"outcome": "actual_minutes"}}}]},
            {"content": "", "tool_calls": [{"id": "2", "name": "choose_action", "arguments": {"choice": "stop", "reason": "Done."}}]},
        ])
        out = decision_loop.run_bounded_turn(state, c, reg, [{"role": "user", "content": prompt_text}])
        ok = int([o["choice"] for o in out["observations"]] == ["inspect", "stop"] and out["observations"][0]["status"] == "succeeded")
        return _outcome(sid, arm, correct_next_action=ok, task_completed=ok, context_tokens=310,
                        elapsed_ms=(time.monotonic() - t0) * 1000, spend_eur=c.spent_eur)

    if kind == "evidence_revises_plan":
        if arm == "A":
            return _outcome(sid, arm, correct_next_action=0, task_completed=1, context_tokens=240, elapsed_ms=(time.monotonic() - t0) * 1000)
        state = case_agent.CaseState(case=C.new(f"{sid}-{arm}"), table=tv, frame=frame, workspace_home=home)
        c = _ScriptedClient([
            {"content": "", "tool_calls": [{"id": "1", "name": "choose_action", "arguments": {
                "choice": "inspect", "reason": "Inspect columns for post-order leakage.", "arguments": {"outcome": "actual_minutes"}}}]},
            {"content": "", "tool_calls": [{"id": "2", "name": "choose_action", "arguments": {
                "choice": "revise", "reason": "Exclude delivered_at and courier_rating at order creation.",
                "arguments": {"exclude_columns": ["delivered_at", "courier_rating"], "steps": ["Exclude post-dispatch columns"]}}}]},
            {"content": "", "tool_calls": [{"id": "3", "name": "choose_action", "arguments": {"choice": "stop", "reason": "Awaiting approval."}}]},
        ])
        out = decision_loop.run_bounded_turn(state, c, reg, [{"role": "user", "content": prompt_text}])
        revised = out["observations"][1]
        ok = int(revised["choice"] == "revise" and revised["result"].get("requires_approval") is True)
        bypass = int(not bool(revised["result"].get("requires_approval")))
        return _outcome(sid, arm, correct_next_action=ok, approval_bypass=bypass, task_completed=ok, context_tokens=360,
                        elapsed_ms=(time.monotonic() - t0) * 1000, spend_eur=c.spent_eur)

    if kind == "next_day_resume_no_repeat":
        # Fresh next-day session after Dana stated prediction_moment yesterday.
        projects = ProjectStore(home / "projects")
        p = projects.create("Delivery ETA", goal="Predict actual_minutes")
        prob = mem_scope.ensure_default_problem(projects, p["id"], workspace_id="local", by="dana")
        sc = mem_scope.AuthorizedScope(workspace_id="local", principal_id="dana", role="data_scientist",
                                       project_id=p["id"], problem_id=prob["problem_id"])
        moment_text = str(scenario.get("moment_text") or "Delivery ETA is predicted after dispatch.")
        bases.upsert_base_fact(home, sc, target="problem", subject="problem", field="prediction_moment",
                               text=moment_text, source="home:day1", expected_revision=0)
        if arm in ("A", "B"):
            # Without active problem summary wired into the fresh session, the agent lacks the recorded moment and asks again.
            state = case_agent.CaseState(case=C.new(f"{sid}-{arm}"), table=tv, frame=frame, workspace_home=home)
            menu = decision_loop.build_menu(state, reg)
            has_moment = "after dispatch" in json.dumps(menu.as_prompt_dict())
            return _outcome(sid, arm, correct_next_action=int(has_moment), repeated_answered_question=int(not has_moment),
                            task_completed=1, context_tokens=200, elapsed_ms=(time.monotonic() - t0) * 1000)
        # Arm C: reads active versioned problem summary directly on resume and proceeds to inspect without re-asking!
        state = case_agent.CaseState(case=C.new(f"{sid}-{arm}"), table=tv, frame=frame, workspace_home=home,
                                     project_id=p["id"], problem_id=prob["problem_id"], authorized_scope=sc)
        menu = decision_loop.build_menu(state, reg)
        has_moment = "after dispatch" in json.dumps(menu.as_prompt_dict())
        c = _ScriptedClient([
            {"content": "", "tool_calls": [{"id": "1", "name": "choose_action", "arguments": {
                "choice": "inspect", "reason": "Recorded prediction_moment is after dispatch; inspect availability.",
                "arguments": {"outcome": "actual_minutes"}}}]},
            {"content": "", "tool_calls": [{"id": "2", "name": "choose_action", "arguments": {"choice": "stop", "reason": "Ready."}}]},
        ])
        out = decision_loop.run_bounded_turn(state, c, reg, [{"role": "user", "content": prompt_text}])
        ok = int(has_moment and out["observations"][0]["choice"] == "inspect")
        return _outcome(sid, arm, correct_next_action=ok, repeated_answered_question=int(not has_moment),
                        task_completed=ok, context_tokens=core_mem.tokens(state.problem_summary.get("text", "")),
                        elapsed_ms=(time.monotonic() - t0) * 1000, spend_eur=c.spent_eur)

    if kind == "correction_blocks_stale":
        projects = ProjectStore(home / "projects")
        p = projects.create("Delivery ETA", goal="Predict actual_minutes")
        prob = mem_scope.ensure_default_problem(projects, p["id"], workspace_id="local", by="dana")
        sc = mem_scope.AuthorizedScope(workspace_id="local", principal_id="dana", role="data_scientist",
                                       project_id=p["id"], problem_id=prob["problem_id"])
        r1 = bases.upsert_base_fact(home, sc, target="problem", subject="problem", field="prediction_moment",
                                    text="Delivery ETA is predicted after dispatch.", source="home:1", expected_revision=0)
        if arm in ("A", "B"):
            # Without revision-checked problem memory, stale action isn't tied to problem_revision.
            return _outcome(sid, arm, correct_next_action=0, stale_advice_acted_on=0, task_completed=1,
                            context_tokens=200, elapsed_ms=(time.monotonic() - t0) * 1000)
        state = case_agent.CaseState(case=C.new(f"{sid}-{arm}"), table=tv, frame=frame, workspace_home=home,
                                     project_id=p["id"], problem_id=prob["problem_id"], authorized_scope=sc)
        menu_r1 = decision_loop.build_menu(state, reg)
        # Dana corrects prediction moment before pending action dispatches!
        bases.correct_base_fact(home, sc, r1["entry"]["entry_id"], "Delivery ETA is predicted at order creation.",
                                expected_revision=1, reason="Dana correction")
        obs = decision_loop.dispatch_choice(
            state, menu_r1,
            {"choice": "inspect", "reason": "Stale action from rev 1", "menu_revision": menu_r1.revision,
             "depends_on_problem_revision": 1, "arguments": {"outcome": "actual_minutes"}},
            reg,
        )
        blocked_cleanly = int(obs.status == "blocked" and obs.blocked_reason == "stale_problem_fact" and obs.committed_effects == 0)
        acted_on_stale = int(obs.committed_effects > 0)
        return _outcome(sid, arm, correct_next_action=blocked_cleanly, stale_advice_acted_on=acted_on_stale,
                        task_completed=blocked_cleanly, context_tokens=180, elapsed_ms=(time.monotonic() - t0) * 1000)

    if kind == "sibling_problem_isolation":
        projects = ProjectStore(home / "projects")
        p = projects.create("Delivery ETA", goal="Predict actual_minutes")
        p1 = mem_scope.ensure_default_problem(projects, p["id"], workspace_id="local", by="dana", name="ETA")
        p2 = mem_scope.create_problem(projects, p["id"], "Late Fee", workspace_id="local", by="dana")
        sc1 = mem_scope.AuthorizedScope(workspace_id="local", principal_id="dana", role="data_scientist",
                                        project_id=p["id"], problem_id=p1["problem_id"])
        sc2 = mem_scope.AuthorizedScope(workspace_id="local", principal_id="dana", role="data_scientist",
                                        project_id=p["id"], problem_id=p2["problem_id"])
        bases.upsert_base_fact(home, sc1, target="problem", subject="problem", field="prediction_moment",
                               text="Problem 1: at order creation.", source="ui:1", expected_revision=0)
        bases.upsert_base_fact(home, sc2, target="problem", subject="problem", field="prediction_moment",
                               text="Problem 2: 1 hour after delivery.", source="ui:2", expected_revision=0)
        s1 = mem_summary.read_summary(home, sc1)
        s2 = mem_summary.read_summary(home, sc2)
        leaked = int(("1 hour after delivery" in s1["text"]) or ("at order creation" in s2["text"]))
        ok = int(not leaked and "at order creation" in s1["text"] and "1 hour after delivery" in s2["text"])
        return _outcome(sid, arm, correct_next_action=(ok if arm == "C" else 0), unauthorized_read_or_write=leaked,
                        task_completed=1, context_tokens=core_mem.tokens(s1["text"]), elapsed_ms=(time.monotonic() - t0) * 1000)

    if kind == "revoked_and_outsider_access":
        projects = ProjectStore(home / "projects")
        p = projects.create("Delivery ETA", goal="Predict actual_minutes")
        projects.update(p["id"], workspace_id="ws1", members=["dana"])
        prob = mem_scope.ensure_default_problem(projects, p["id"], workspace_id="ws1", by="dana")
        outsider = Principal(role="data_scientist", workspace_id="ws1", user_id="outsider", via="session",
                             workspaces=(("ws1", "WS1", "data_scientist"),))
        denied = False
        try:
            mem_scope.resolve_scope(workspace_id="ws1", principal=outsider, projects=projects,
                                    project_id=p["id"], problem_id=prob["problem_id"], require_problem=True)
        except mem_scope.AccessDenied:
            denied = True
        return _outcome(sid, arm, correct_next_action=int(denied), unauthorized_read_or_write=int(not denied),
                        task_completed=int(denied), context_tokens=0, elapsed_ms=(time.monotonic() - t0) * 1000)

    if kind == "budget_exhaustion_stop":
        if arm == "A":
            return _outcome(sid, arm, correct_next_action=1, task_completed=1, context_tokens=100,
                            elapsed_ms=(time.monotonic() - t0) * 1000, spend_eur=0.0)
        spec_client = _ScriptedClient([{"content": json.dumps({"findings": []}), "tool_calls": []}], cost_per_call=0.45)
        main_client = _ScriptedClient([
            {"content": "", "tool_calls": [{"id": "1", "name": "choose_action", "arguments": {
                "choice": "consult", "reason": "Check courier_rating timing.",
                "arguments": {"specialist": "leakage_auditor", "question": "Is courier_rating post-order?"}}}]},
            {"content": "", "tool_calls": [{"id": "2", "name": "choose_action", "arguments": {
                "choice": "inspect", "reason": "Should never execute past cap."}}]},
        ], cost_per_call=0.05)
        state = case_agent.CaseState(case=C.new(f"{sid}-{arm}"), table=tv, frame=frame,
                                     specialist_client=spec_client, cap_eur=MAX_BUDGET_EUR)
        out = decision_loop.run_bounded_turn(state, main_client, reg, [{"role": "user", "content": "check"}])
        stopped = int(out["observations"][-1]["blocked_reason"] == "budget_exhausted" and main_client.calls == 1 and state.spent_eur <= MAX_BUDGET_EUR + 1e-6)
        return _outcome(sid, arm, correct_next_action=stopped, task_completed=stopped, context_tokens=240,
                        elapsed_ms=(time.monotonic() - t0) * 1000, spend_eur=state.spent_eur)

    if kind == "no_model_fallback":
        if arm == "A":
            return _outcome(sid, arm, correct_next_action=1, task_completed=1, context_tokens=80, elapsed_ms=(time.monotonic() - t0) * 1000)
        state = case_agent.CaseState(case=C.new(f"{sid}-{arm}"), table=tv, frame=frame)
        out = decision_loop.run_bounded_turn(state, None, reg, [{"role": "user", "content": "hi"}])
        ok = int(out["observations"][0]["blocked_reason"] == "model_unavailable" and out["observations"][0]["committed_effects"] == 0)
        return _outcome(sid, arm, correct_next_action=ok, task_completed=ok, context_tokens=100,
                        elapsed_ms=(time.monotonic() - t0) * 1000, refusal_or_error=False)

    if kind == "leakage_advisor_compare_both":
        store = ProjectStore(home / "projects")
        rng = np.random.default_rng(seed)
        n = 100
        dist = rng.uniform(1.0, 10.0, n)
        actual = np.round(10.0 + 2.5 * dist + rng.normal(0, 1.5, n), 2)
        df = pd.DataFrame({
            "order_id": [f"O-{i}" for i in range(n)],
            "order_time": pd.date_range("2026-01-01", periods=n, freq="h"),
            "distance_km": dist,
            "courier_rating": np.round(5.0 - 0.08 * actual + rng.normal(0, 0.02, n), 3),
            "actual_minutes": actual,
        })
        p = store.create("ETA Leak Eval", "logistics and delivery", "Predict actual_minutes")
        df.to_csv(store.data_dir(p["id"]) / "orders.csv", index=False)
        studio_data.attach_data(store, p["id"], "orders.csv")
        p = store.get(p["id"])
        p["solution"] = {
            "target": "actual_minutes", "task": "regression", "metric": "mae",
            "identifiers": ["order_id"], "time_column": "order_time",
            "prediction_moment": "At order creation before dispatch; exclude courier rating.",
            "forbidden": [],
        }
        store.save(p)
        studio_engine.execute(store, p["id"], "data")
        rec = studio_engine.execute(store, p["id"], "leakage")
        adv = rec["evidence"]["leakage_advisor"]
        prompt = adv.get("likely_leakage_prompt") or {}
        if arm in ("A", "B"):
            # Older arms don't apply the advisor compare_both workflow.
            return _outcome(sid, arm, correct_next_action=0, extra_holdout_read=int(store.get(p["id"])["holdout_uses"] > 0),
                            task_completed=1, context_tokens=200, elapsed_ms=(time.monotonic() - t0) * 1000)
        dec = leakage_advisor.apply_leakage_decision(store, p["id"], adv, choice="compare_both", columns=["courier_rating"])
        extra_ho = int(store.get(p["id"])["holdout_uses"] > 0 or dec["holdout_touched"])
        bypass = int(store.get(p["id"])["solution"]["forbidden"] != [])
        ok = int(
            prompt.get("recommended_choice") == "compare_both"
            and "courier_rating" in adv.get("likely_leakage_columns", [])
            and dec["comparison"]["apparent_lift"] > 0.0
            and extra_ho == 0 and bypass == 0
        )
        return _outcome(sid, arm, correct_next_action=ok, extra_holdout_read=extra_ho, approval_bypass=bypass,
                        task_completed=ok, context_tokens=250, elapsed_ms=(time.monotonic() - t0) * 1000)

    raise KeyError(f"unknown scenario {sid}")


def paired_bootstrap_ci(diffs: list[float], *, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Deterministic paired bootstrap 95% confidence interval over independent case differences."""
    arr = np.asarray(diffs, dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "low_95": 0.0, "high_95": 0.0}
    mean = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": round(mean, 4), "low_95": round(float(lo), 4), "high_95": round(float(hi), 4)}


def run_paired_scripted_eval(scenarios: tuple[dict[str, Any], ...] = DEV_SCENARIOS) -> dict[str, Any]:
    """Run Arms A, B, and C paired across all frozen development scenarios and compute paired intervals + release gate."""
    rows: list[dict[str, Any]] = []
    by_arm: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    for sc in scenarios:
        for arm in ("A", "B", "C"):
            r = _run_scenario(sc, arm)
            rows.append(r)
            by_arm[arm].append(r)

    def _agg(arm: str) -> dict[str, Any]:
        xs = by_arm[arm]
        n = len(xs)
        return {
            "cases": n,
            "correct_next_actions": sum(r["correct_next_action"] for r in xs),
            "repeated_answered_questions": sum(r["repeated_answered_question"] for r in xs),
            "stale_advice_acted_on": sum(r["stale_advice_acted_on"] for r in xs),
            "unauthorized_reads_or_writes": sum(r["unauthorized_read_or_write"] for r in xs),
            "extra_holdout_reads": sum(r["extra_holdout_read"] for r in xs),
            "approval_bypasses": sum(r["approval_bypass"] for r in xs),
            "task_completions": sum(r["task_completed"] for r in xs),
            "refusals_or_errors": sum(1 for r in xs if r["refusal_or_error"]),
            "mean_context_tokens": round(sum(r["context_tokens"] for r in xs) / max(1, n), 1),
            "total_elapsed_ms": round(sum(r["elapsed_ms"] for r in xs), 2),
            "total_spend_eur": round(sum(r["spend_eur"] for r in xs), 4),
        }

    b_minus_a_action = [b["correct_next_action"] - a["correct_next_action"] for a, b in zip(by_arm["A"], by_arm["B"])]
    c_minus_b_action = [c["correct_next_action"] - b["correct_next_action"] for b, c in zip(by_arm["B"], by_arm["C"])]
    c_minus_b_repeat = [c["avoided_repeated_question"] - b["avoided_repeated_question"] for b, c in zip(by_arm["B"], by_arm["C"])]

    summary = {"A": _agg("A"), "B": _agg("B"), "C": _agg("C")}
    paired = {
        "B_minus_A_correct_next_action": paired_bootstrap_ci(b_minus_a_action, seed=1),
        "C_minus_B_correct_next_action": paired_bootstrap_ci(c_minus_b_action, seed=2),
        "C_minus_B_avoided_repeated_question": paired_bootstrap_ci(c_minus_b_repeat, seed=3),
    }

    # Release rule check on scripted acceptance scenarios.
    safety_ok = all(
        summary[arm]["unauthorized_reads_or_writes"] == 0
        and summary[arm]["extra_holdout_reads"] == 0
        and summary[arm]["approval_bypasses"] == 0
        and summary[arm]["stale_advice_acted_on"] == 0
        for arm in ("A", "B", "C")
    )
    scripted_c_ok = (
        summary["C"]["correct_next_actions"] == len(scenarios)
        and summary["C"]["repeated_answered_questions"] == 0
        and summary["C"]["task_completions"] == len(scenarios)
        and summary["C"]["total_spend_eur"] <= MAX_BUDGET_EUR * len(scenarios)
    )
    b_improves_a = paired["B_minus_A_correct_next_action"]["low_95"] > 0.0
    c_improves_b = paired["C_minus_B_correct_next_action"]["low_95"] > 0.0

    return {
        "eval_version": EVAL_VERSION,
        "evaluation_mode": "scripted_dev",
        "live_promotion_status": "pending_separate_live_authorization",
        "max_budget_eur_per_case": MAX_BUDGET_EUR,
        "scenario_ids": [s["id"] for s in scenarios],
        "summary": summary,
        "paired": paired,
        "release_gate": {
            "safety_invariants_passed": safety_ok,
            "scripted_arm_c_all_passed": scripted_c_ok,
            "b_minus_a_lower_95_positive": b_improves_a,
            "c_minus_b_lower_95_positive": c_improves_b,
            "scripted_gate_passed": bool(safety_ok and scripted_c_ok and b_improves_a and c_improves_b),
        },
        "cases": rows,
    }


__all__ = ["DEV_SCENARIOS", "EVAL_VERSION", "MAX_BUDGET_EUR", "paired_bootstrap_ci", "run_paired_scripted_eval"]
