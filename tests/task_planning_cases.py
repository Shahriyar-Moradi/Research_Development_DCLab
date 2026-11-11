"""Authored TP-03 host observations, never real approvals or execution evidence."""
import copy
import json
from pathlib import Path

from dclab_core.contracts import SolutionPlan
from dclab_core.solution_workflow import SolutionWorkflow
from dclab_rnd.agents import build_registry
from dclab_rnd.tasks.adapter import adapt_workflow, capability_snapshot


def case():
    fixture = json.loads((Path(__file__).parent / "fixtures/tasks/telco_metadata.json").read_text())
    project = fixture["project"]
    source = SolutionPlan.read(fixture["source_plan"])
    workflow = SolutionWorkflow.read(fixture["workflow"])
    registry = build_registry(("project",))
    caps = capability_snapshot(registry, allowed_tools={"run_stage", "export_notebook"})
    graph, mapping = adapt_workflow(workflow, source, project, capabilities=caps)
    approval = next(ref for ref in graph.source_refs if ref.kind == "approval")
    artifacts = copy.deepcopy(mapping["external_artifacts"])
    artifacts["project_context"]["value"] = project["id"]
    for stage in ("data", "leakage", "features", "models", "final"):
        artifacts["stage_" + stage]["value"] = stage
    state = {"project_id": project["id"], "graph_id": graph.graph_id, "graph_revision": graph.revision,
             "graph_fingerprint": graph.fingerprint, "solution_fingerprint": graph.solution_fingerprint,
             "workflow_fingerprint": graph.workflow_fingerprint, "artifacts": artifacts, "tasks": {},
             "holdout_uses": 0, "approval": {"approved": True, "reference_id": approval.id,
             "fingerprint": approval.fingerprint, "project_id": project["id"],
             "solution_fingerprint": graph.solution_fingerprint, "workflow_fingerprint": graph.workflow_fingerprint,
             "sources": [ref.model_dump(mode="json") for ref in graph.source_refs if ref.kind != "approval"],
             "inputs": {key: artifacts[key]["fingerprint"] for key in ("table", "execution_context")}}}
    return project, source, workflow, caps, graph, state


def advance(graph, state, task_id):
    from tests.task_validation_cases import verified

    state = verified(graph, state, task_id)
    if task_id == "holdout":
        state["tasks"][task_id]["approval_ref"]["fingerprint"] = state["holdout_approval"]["fingerprint"]
    return state
