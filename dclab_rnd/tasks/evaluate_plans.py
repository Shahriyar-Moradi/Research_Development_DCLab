"""Repository-only TP-03 scripted DEV comparison. Authored observations are not real execution evidence.

Run DCLAB_NO_LIVE_MODELS=1 python -m dclab_rnd.tasks.evaluate_plans --out evaluation/task_planning_tp03_dev.json.
This compares deterministic plans with deliberately scripted proposals on one binary fixture, not live quality.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from dclab_rnd.models import FileUsage, Gateway
from dclab_rnd.tasks.planner import VERSION, plan_task

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = tuple(("stage_" + stage, stage, "proposed") for stage in ("data", "leakage", "features", "models", "final", "report")) + (
    ("gate_noop", "holdout", "noop"), ("dependency_not_ready", "models", "blocked"),
    ("missing_arguments", "data", "fallback"), ("arbitrary_tool", "data", "fallback"),
    ("wrong_scope", "data", "fallback"), ("write_without_move", "data", "blocked"),
    ("forbidden_value", "data", "blocked"), ("missing_artifact", "data", "blocked"),
    ("invalid_binding", "data", "fallback"), ("stale_capability", "data", "blocked"),
    ("changed_intent", "data", "blocked"), ("consumed_holdout", "final", "blocked"),
    ("overbudget", "data", "fallback"), ("model_refusal", "data", "fallback"),
    ("timeout", "data", "fallback"), ("model_down", "data", "fallback"),
    ("model_disabled", "data", "deterministic"), ("stale_model_context", "data", "fallback"))
GATE = {"version": "tp03_scripted_dev.v1", "fixed_before_run": True, "all_expected_outcomes": True,
        "exact_fallback_on_invalid_proposal": True, "tool_calls": 0, "holdout_accesses": 0,
        "limits": "One authored source fixture with simulated observations; no execution, live quality or rollout verdict."}
FILES = ("dclab_rnd/tasks/planner.py", "dclab_rnd/tasks/evaluate_plans.py", "dclab_rnd/prompts/task_planner.md",
         "tests/task_planning_cases.py", "tests/fixtures/tasks/telco_metadata.json")


def fingerprints():
    from dclab_rnd.models.settings import PURPOSES
    from dclab_rnd.agents.harness import ROLES
    from dclab_core.task_fingerprints import metadata_fingerprint

    return {**{name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES},
            "task_planner_registration": metadata_fingerprint({"purpose": vars(PURPOSES["task_planner"]),
                                                              "harness_role": "task_planner" in ROLES})}


def evaluate():
    from tests.task_planning_cases import case, advance
    from dclab_rnd.agents.registry import Registry
    from tests.test_models_gateway import CLEAN

    rows = []
    env = {**CLEAN, "DCLAB_NO_LIVE_MODELS": "1", "OPENAI_API_KEY": "scripted",
           "OPENAI_MODEL": "scripted-task-plan", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1"}
    remaining = {"max_tool_calls": 2, "max_seconds": 180, "max_cost_eur": 0.5}
    with patch.dict(os.environ, env), tempfile.TemporaryDirectory() as folder, \
            patch("dclab_rnd.models.gateway._exchange_home", return_value=Path(folder)), \
            patch("dclab_rnd.experience.model_text_allowed", return_value=False):
        for ident, task, expected in SCENARIOS:
            project, source, workflow, caps, graph, state = case()
            if ident != "dependency_not_ready":
                for prior in ("data", "leakage", "features", "models", "holdout", "final", "report"):
                    if prior == task:
                        break
                    state = advance(graph, state, prior)
            args = dict(graph=graph, task_id=task, observed_state=state, capabilities=caps, remaining_budget=remaining)
            baseline, base_report = plan_task(workflow, source, project, **args)
            if ident == "write_without_move": caps["run_stage"]["move"] = None
            if ident == "forbidden_value": state["artifacts"]["table"]["value"] = "FORBIDDEN_DEV_VALUE"
            if ident == "missing_artifact": state["artifacts"].pop("table")
            if ident == "stale_capability": caps["run_stage"]["parameters"]["description"] = "Changed tool schema"
            if ident == "changed_intent": project["goal"] = "Another outcome after confirmation"
            if ident == "consumed_holdout": state["holdout_uses"] = 1
            calls = []

            class Scripted:
                def complete(self, messages, tools=None, max_tokens=None, **options):
                    calls.append(1)
                    if tools: raise AssertionError("planner offered execution tools")
                    if ident == "timeout": raise TimeoutError("scripted timeout")
                    if ident == "model_down": raise RuntimeError("scripted outage")
                    body = json.loads(messages[-1]["content"])["baseline"]
                    p = {"version": VERSION, "baseline_fingerprint": body["fingerprint"],
                         **{key: copy.deepcopy(body[key]) for key in ("actions", "budget", "policy")}}
                    if ident == "missing_arguments": p["actions"][0]["arguments"] = []
                    if ident == "arbitrary_tool": p["actions"][0]["capability_name"] = "unregistered_tool"
                    if ident == "wrong_scope": p["actions"][0]["scope"] = "draft"
                    if ident == "invalid_binding": p["actions"][0]["arguments"][0]["value"]["ref_id"] = "missing"
                    if ident == "overbudget": p["budget"]["max_cost_eur"] = 1.0
                    if ident == "stale_model_context": p["baseline_fingerprint"] = "a" * 64
                    return {"content": json.dumps(p), "tool_calls": [],
                            "finish_reason": "refusal" if ident == "model_refusal" else "stop", "usage": {}}

            gateway = Gateway(FileUsage(Path(folder) / "usage.jsonl"), transport=lambda tier, purpose: Scripted())
            original_uses = state["holdout_uses"]
            with patch.object(Registry, "call") as call, patch.object(Registry, "acall") as acall, \
                    patch("dclab_rnd.studio.engine.execute") as execute:
                result, report = plan_task(workflow, source, project, **args, enabled=ident != "model_disabled", gateway=gateway)
                effects = call.call_count + acall.call_count + execute.call_count
            fallback_ok = expected != "fallback" or result == baseline
            passed = (report["status"] == expected and fallback_ok and effects == 0
                      and state["holdout_uses"] == original_uses and not report["execution_authorized"])
            rows.append({"id": ident, "task": task, "expected": expected, "actual": report["status"], "passes": passed,
                         "deterministic_status": base_report["status"], "provider_calls": len(calls), "tool_calls": effects,
                         "extra_holdout_accesses": state["holdout_uses"] - original_uses, "exact_fallback": result == baseline,
                         "plan_status": result.status if result else None, "refusals": report["refusals"],
                         "blockers": report["blockers"]})
    return {"version": VERSION, "split": "dev", "mode": "scripted", "distinct_source_fixtures": 1,
            "fingerprints": fingerprints(), "gate": GATE, "rows": rows,
            "summary": {"scenarios": len(rows), "passed": sum(r["passes"] for r in rows),
                        "scripted_provider_calls": sum(r["provider_calls"] for r in rows), "live_calls": 0, "cost_eur": 0,
                        "tool_calls": sum(r["tool_calls"] for r in rows), "extra_holdout_accesses": 0,
                        "exact_invalid_fallbacks": sum(r["actual"] == "fallback" and r["exact_fallback"] for r in rows)},
            "verdict": "pass" if all(r["passes"] for r in rows) else "fail", "promotion": "not evaluated; TP-06 is separate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verdict": result["verdict"], **result["summary"]}))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
