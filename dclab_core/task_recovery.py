"""Pure TP-05 recovery decisions. They never retry a tool, confirm a source or change an owner decision."""
from __future__ import annotations

from dclab_core.task_graph import TaskGraph
from dclab_core.task_fingerprints import graph_fingerprint, metadata_fingerprint

CATEGORIES = ("transient", "invalid_arguments", "changed_input", "missing_capability", "policy_refusal",
              "missing_approval", "ambiguous_effect", "domain_failure")


def classify_failure(category, *, effect="unknown", attempts=1, max_attempts=2, holdout_consumed=False):
    if category not in CATEGORIES or effect not in ("none", "completed", "unknown"):
        raise ValueError("invalid_failure_classification")
    if type(attempts) is not int or type(max_attempts) is not int or not 1 <= max_attempts <= 3 or attempts < 0:
        raise ValueError("invalid_recovery_bound")
    if effect == "completed":
        action, reason = "verify", "owner_completion_available"
    elif holdout_consumed:
        action, reason = "stop", "holdout_consumed"
    elif category in ("missing_capability", "policy_refusal", "missing_approval", "domain_failure"):
        action, reason = "stop", category
    elif category == "changed_input":
        action, reason = "confirm_sources", "new_confirmed_revision_required"
    elif effect != "none":
        action, reason = "stop", "ambiguous_effect"
    elif attempts >= max_attempts:
        action, reason = "stop", "attempt_limit"
    else:
        action, reason = "replan", "proven_no_handler_entry"
    return {"category": category, "effect": effect, "action": action, "reason": reason,
            "attempts": attempts, "max_attempts": max_attempts}


def task_definition(node):
    body = node.model_dump(mode="json", exclude={"parent_id"})
    for key in ("needs", "produces", "dependencies", "source_step_ids"):
        body[key].sort()
    body["acceptance_checks"] = sorted(({**c, "evidence_refs": sorted(c["evidence_refs"])}
                                       for c in body["acceptance_checks"]), key=lambda c: c["id"])
    return metadata_fingerprint(body)


def invalidated_tasks(graph, *, changed_artifacts=(), changed_tasks=(), verified=()):
    graph = TaskGraph.read(graph.model_dump(mode="json") if hasattr(graph, "model_dump") else graph)
    if graph.fingerprint != graph_fingerprint(graph):
        raise ValueError("invalid_graph_fingerprint")
    nodes = {n.id: n for n in graph.nodes}
    invalid = set(changed_tasks)
    if not invalid <= set(nodes) or not set(verified) <= set(nodes):
        raise ValueError("unknown_task")
    # Grouping is presentation. Only consumed artifacts and execution dependencies invalidate work.
    invalid.update(n.id for n in graph.nodes if set(n.needs) & set(changed_artifacts))
    while True:
        expanded = invalid | {n.id for n in graph.nodes if set(n.dependencies) & invalid}
        if expanded == invalid:
            break
        invalid = expanded
    return {"invalidated": sorted(invalid), "preserved": sorted(set(verified) - invalid)}


def rebase_verified(record, graph, *, inputs, outputs, evidence, confirmed_source_revision=False):
    """Mint a NEW proof identity after host rechecks; never relabel the old verification in place."""
    graph = TaskGraph.read(graph.model_dump(mode='json') if hasattr(graph, 'model_dump') else graph)
    if graph.fingerprint != graph_fingerprint(graph):
        raise ValueError('invalid_graph_fingerprint')
    node = next((n for n in graph.nodes if n.id == record.get("task_id")), None)
    if (record.get("status") != "verified" or node is None or record.get("definition") != task_definition(node)
            or record.get("inputs") != inputs or record.get("outputs") != outputs or record.get("evidence") != evidence):
        raise ValueError("preserved_evidence_changed")
    sources = {(r.kind, r.id): r.fingerprint for r in graph.source_refs}
    before = {(r["kind"], r["id"]): r["fingerprint"] for r in record.get("sources", [])}
    if before != sources and not confirmed_source_revision:
        raise ValueError("source_confirmation_required")
    body = {**record, "graph_revision": graph.revision, "graph_fingerprint": graph.fingerprint,
            "sources": [r.model_dump(mode="json") for r in graph.source_refs], "reused_from": record["id"]}
    body.pop("id", None)
    body["id"] = "verify_" + metadata_fingerprint(body)[:24]
    return body
