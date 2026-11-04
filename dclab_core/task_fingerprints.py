"""Canonical metadata identities for task proposals (TP-01C). No I/O and no implicit document rewriting."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from dclab_core.task_graph import TaskGraph
from dclab_core.task_execution_plan import TaskExecutionPlan
from dclab_core.solution_workflow import SolutionWorkflow
from dclab_core.solution_map import SolutionMap


def metadata_fingerprint(value) -> str:
    """SHA-256 of JSON metadata; hosts must supply sanitized metadata, never table rows or model text."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _raw(value) -> dict:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)


def _checks(items):
    return sorted(({**item, "evidence_refs": sorted(item["evidence_refs"])} for item in items), key=lambda item: item["id"])


def graph_fingerprint(graph) -> str:
    """Hash structural content. Node/edge/set-like ordering and creation provenance are immaterial; revision is not.

    A mapping may omit its hash while being built. Only the private copy gets a placeholder for shape checking.
    This computes an identity, not a validation verdict or an approval. Existing documents are never updated.
    """
    body = _raw(graph); body["fingerprint"] = "0" * 64
    body = TaskGraph.read(body).model_dump(mode="json")
    body.pop("fingerprint"); body.pop("provenance")
    body["source_refs"].sort(key=lambda item: (item["kind"], item["id"]))
    for node in body["nodes"]:
        for field in ("source_step_ids", "needs", "produces", "dependencies"):
            node[field].sort()
        node["acceptance_checks"] = _checks(node["acceptance_checks"])
    body["nodes"].sort(key=lambda node: node["id"])
    body["edges"].sort(key=lambda edge: (edge["kind"], edge["source"], edge["target"]))
    return metadata_fingerprint(body)


def plan_fingerprint(plan) -> str:
    """Keep action order, identities, inputs, budgets and stop policy; exclude proposal status, creation record and hash.

    Status is independently checked before readiness; equal hashes are never completion or execution permission.
    """
    body = _raw(plan); body["fingerprint"] = "0" * 64
    body = TaskExecutionPlan.read(body).model_dump(mode="json")
    for field in ("fingerprint", "provenance", "status"):
        body.pop(field)
    for field in ("expected_inputs", "declared_outputs"):
        body[field].sort(key=lambda item: item["id"])
    body["selected_capabilities"].sort(key=lambda item: (item["scope"], item["name"]))
    for action in body["actions"]:
        action["arguments"].sort(key=lambda argument: argument["name"])
    body["acceptance_checks"] = _checks(body["acceptance_checks"])
    return metadata_fingerprint(body)


def workflow_fingerprint(source) -> str:
    """New full SHA-256 identity for ordered source workflow content, or map-level workflows/composition.

    This does not replace SolutionMap's existing authoritative short fingerprint. Titles/criteria and step order
    remain meaningful. Mutable execution results and Programme fingerprints are not accepted as source workflows.
    """
    body = _raw(source)
    if isinstance(source, SolutionMap) or body.get("schema_version") == "dclab_core.solution_map.v1":
        sm = SolutionMap.read(body)
        body = {"schema_version": sm.schema_version, "blocks": [{"id": block.id, "workflow": block.workflow.model_dump(mode="json"),
                "final": block.final, "defers": block.defers} for block in sm.blocks], "steps": [step.model_dump(mode="json") for step in sm.steps],
                "holdout": sm.holdout.model_dump(mode="json")}
    else:
        body = SolutionWorkflow.read(body).model_dump(mode="json")
    return metadata_fingerprint(body)


def capability_fingerprint(capability: Mapping) -> str:
    """Bind the capability schema AND its scope/effect/move/permission/replay policy.

    TP-01B's schema_fingerprint field carries this complete descriptor identity, so a move/permission change cannot
    masquerade as an unchanged argument schema. The host extracts metadata from the real registry, not a model.
    """
    body = {key: capability.get(key) for key in ("name", "scope", "effect", "move", "allowed", "replay_safe")}
    body["parameters"] = capability.get("parameters")
    return metadata_fingerprint(body)


__all__ = ["graph_fingerprint", "plan_fingerprint", "workflow_fingerprint", "capability_fingerprint", "metadata_fingerprint"]
