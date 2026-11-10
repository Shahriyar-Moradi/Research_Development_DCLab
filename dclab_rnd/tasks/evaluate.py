"""TP-02 DEV smoke comparison on authored metadata and deliberately scripted replies; never a live evaluation.

Run: DCLAB_NO_LIVE_MODELS=1 python -m dclab_rnd.tasks.evaluate --out evaluation/task_decomposition_tp02_dev.json
The nine scenarios share one binary fixture. They measure the boundary checker and fallback, not model quality.
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

from dclab_core.contracts import SolutionPlan
from dclab_core.solution_workflow import SolutionWorkflow
from dclab_rnd.agents import build_registry
from dclab_rnd.models import FileUsage, Gateway
from dclab_rnd.tasks.adapter import adapt_workflow, capability_snapshot
from dclab_rnd.tasks.decomposer import VERSION, decompose

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/tasks/telco_metadata.json"
SCENARIOS = ("one_task_per_group", "multi_step_churn", "branch_dependencies", "missing_capability",
             "contradictory_dependency", "hallucinated_tool", "removed_safety_task", "missing_data", "changed_objective")
GATE = {"version": "tp02_scripted_dev.v1", "fixed_before_run": True,
        "expected_outcomes": 9, "preserved_source_coverage": True, "exact_fallback_on_refusal": True,
        "task_execution_allowed": False,
        "limits": "One authored fixture, nine scenarios, deliberately injected failures. No model quality or rollout verdict."}


def _proposal(graph, single=False):
    leaves = [n for n in graph.nodes if n.kind != "group"]
    groups = [[n.id] for n in leaves] if single else [["data", "leakage", "features", "models"], ["holdout", "final", "report"]]
    return {"version": VERSION, "baseline_fingerprint": graph.fingerprint,
            "groups": [{"id": f"group_{i}", "parent_id": None, "members": group} for i, group in enumerate(groups)],
            "dependencies": [{"task_id": n.id, "requires": list(n.dependencies)} for n in leaves]}


def _coverage(graph, workflow):
    required = {s.id for s in workflow.steps}
    covered = {s for n in graph.nodes for s in n.source_step_ids} if graph else set()
    return {"covered": len(covered & required), "required": len(required),
            "meaning": "Structural coverage includes explicit blockers; it is not execution or completion."}


def evaluate():
    fixture = json.loads(FIXTURE.read_text())
    rows = []
    registry = build_registry(("project",))
    # Explicitly local, scripted settings. No real transport, credentials, server or .env is used.
    env = {"DCLAB_NO_LIVE_MODELS": "1", "DCLAB_LLM_PROVIDER": "", "DCLAB_TIER_STANDARD_BASE_URL": "http://127.0.0.1:1/v1",
           "DCLAB_TIER_STANDARD_MODEL": "scripted-task-boundaries", "DCLAB_TIER_STANDARD_API_KEY": "scripted"}
    with patch.dict(os.environ, env), tempfile.TemporaryDirectory() as folder:
        for scenario in SCENARIOS:
            project = copy.deepcopy(fixture["project"])
            plan, workflow = SolutionPlan.read(fixture["source_plan"]), SolutionWorkflow.read(fixture["workflow"])
            caps = capability_snapshot(registry, allowed_tools={"run_stage", "export_notebook"})
            if scenario == "missing_capability":
                caps.pop("run_stage")
            if scenario == "missing_data":
                project.pop("data")
            baseline, mapping = adapt_workflow(workflow, plan, project, capabilities=caps)
            proposal = _proposal(baseline, scenario == "one_task_per_group") if baseline else {}
            if scenario == "branch_dependencies":
                next(d for d in proposal["dependencies"] if d["task_id"] == "models")["requires"].append("data")
            if scenario == "contradictory_dependency":
                next(d for d in proposal["dependencies"] if d["task_id"] == "final")["requires"] = []
            if scenario == "hallucinated_tool":
                proposal["tool"] = "execute_arbitrary_python"
            if scenario == "removed_safety_task":
                proposal["groups"][1]["members"].remove("holdout")
            goal = project["goal"]
            if scenario == "changed_objective":
                project["goal"] = "Predict another outcome after the solution was confirmed."
            calls = []

            class Scripted:
                def complete(self, messages, tools=None, max_tokens=None, **options):
                    if tools:
                        raise AssertionError("tools offered during decomposition")
                    calls.append(1)
                    return {"content": json.dumps(proposal), "tool_calls": [], "finish_reason": "stop",
                            "usage": {"input_tokens": 0, "output_tokens": 0}}

            gateway = Gateway(FileUsage(Path(folder) / "usage.jsonl"), transport=lambda tier, purpose: Scripted())
            gateway.home = Path(folder)
            graph, report = decompose(workflow, plan, project, deterministic_graph=baseline, capabilities=caps,
                                      confirmed_goal=goal, enabled=True, gateway=gateway)
            expected = "proposed" if scenario in SCENARIOS[:3] else "blocked" if scenario in (
                "missing_capability", "missing_data", "changed_objective") else "fallback"
            preserved = report["status"] == "proposed" or graph == baseline
            before, after = _coverage(baseline, workflow), _coverage(graph, workflow)
            rows.append({"id": scenario, "expected": expected, "actual": report["status"],
                         "passes": report["status"] == expected and preserved and before == after and not report["execution_authorized"],
                         "baseline_coverage": before, "proposal_coverage": after, "exact_fallback": graph == baseline,
                         "provider_calls": len(calls), "refusals": report["refusals"], "blockers": report["blockers"],
                         "baseline_fingerprint": baseline.fingerprint if baseline else None,
                         "output_fingerprint": graph.fingerprint if graph else None})
    called = sum(r["provider_calls"] for r in rows)
    invalid = sum(r["actual"] == "fallback" and r["provider_calls"] > 0 for r in rows)
    return {"version": VERSION, "split": "dev", "mode": "scripted", "fixture": str(FIXTURE.relative_to(ROOT)),
            "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), "distinct_source_fixtures": 1,
            "prompt_sha256": hashlib.sha256((ROOT / "dclab_rnd/prompts/task_decomposer.md").read_bytes()).hexdigest(),
            "gate": GATE, "rows": rows,
            "summary": {"scenarios": len(rows), "passed": sum(r["passes"] for r in rows), "provider_calls": called,
                        "deliberately_invalid_proposals": invalid, "invalid_proposal_rate": invalid / called if called else None,
                        "exact_fallbacks": sum(r["actual"] != "proposed" and r["exact_fallback"] for r in rows),
                        "cost_eur": 0, "live_calls": 0, "task_execution": 0},
            "verdict": "pass" if all(r["passes"] for r in rows) else "fail",
            "promotion": "not evaluated; TP-06 is separate"}


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
