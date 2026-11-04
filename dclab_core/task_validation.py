"""Pure task planning checks (TP-01C), never a dispatcher or an authority for scientific effects.

The host supplies current source contracts and sanitized registry/approval/artifact/verification snapshots. These
functions do not fetch them, authenticate signatures, invoke a tool or mark work complete. Runtime owners must
recheck every permission, input, receipt and holdout condition at the actual side-effect boundary.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import json
import re

from pydantic import ValidationError

from dclab_core.task_graph import TaskGraph
from dclab_core.task_execution_plan import TaskExecutionPlan
from dclab_core import solution_workflow as workflows, solution_map as maps
from dclab_core.task_fingerprints import (graph_fingerprint, plan_fingerprint, workflow_fingerprint,
                                          capability_fingerprint, metadata_fingerprint)


# Physical stages are existing domain operations, not a second execution engine. In particular threshold is
# training/OOF work inside final; its conceptual WF-08 location does not authorize a separate final-stage call.
_STAGE_STEPS = {"lock_holdout": "data", "train_only_analysis": "data", "class_balance": "data",
                "leakage_audit": "leakage", "choose_on_validation": "models", "final_once": "final", "threshold": "final"}
_ORDER = ("lock_holdout", "train_only_analysis", "leakage_audit", "choose_on_validation", "holdout_gate", "final_once", "report_limits")


def _snapshot(value):
    """Use a private JSON metadata copy for one check; never retain live registry/store objects."""
    if not isinstance(value, Mapping): raise ValueError("host snapshot must be a metadata mapping")
    try: return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error: raise ValueError("host snapshot must contain JSON metadata only") from error


def _digest(value, lengths=(64,)):
    return isinstance(value, str) and len(value) in lengths and re.fullmatch(r"[0-9a-f]+", value) is not None


def _reference(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}", value) is not None


def _issues(items):
    return sorted(set(items))[:100]


def _read(cls, value):
    # Re-read a dump even for model instances: model_copy/model_construct are not validation boundaries.
    return cls.read(value.model_dump(mode="json") if hasattr(value, "model_dump") else value)


def _shape_problem(error):
    if isinstance(error, ValidationError):
        return ["invalid document field " + ".".join(map(str, item["loc"])) + " (" + item["type"] + ")"
                for item in error.errors(include_input=False, include_context=False)][:20]
    return ["invalid document or schema version"]


def _ancestors(links):
    """Bounded dependency/parent closure; a cycle is explicit and never interpreted as ready."""
    result, cycles = {}, set()
    def visit(ident, path):
        if ident in path:
            cycles.update(path[path.index(ident):]); return set()
        if ident in result: return result[ident]
        found = set()
        for dep in sorted(links.get(ident, ())):
            if dep in links:
                found.add(dep); found.update(visit(dep, [*path, ident]))
        result[ident] = found
        return found
    for ident in sorted(links): visit(ident, [])
    return result, sorted(cycles)


def _capability_problems(node, capabilities):
    if node.kind != "action": return []
    binding = node.capability
    cap = capabilities.get(binding.name)
    if not isinstance(cap, Mapping): return [f"{node.id}: missing capability {binding.name}"]
    out = []
    if cap.get("name") != binding.name or cap.get("scope") != binding.scope or node.scope != binding.scope:
        out.append(f"{node.id}: capability scope/name mismatch")
    if cap.get("allowed") is not True: out.append(f"{node.id}: capability permission is not granted in host snapshot")
    if cap.get("effect") not in ("read", "write") or not isinstance(cap.get("parameters"), Mapping):
        out.append(f"{node.id}: incomplete capability metadata")
    if cap.get("effect") == "write" and (not isinstance(cap.get("move"), str) or not cap["move"]): out.append(f"{node.id}: write capability has no guarded move")
    guarded = {"run_stage": "run_stage", "run_programme": "run_programme", "export_notebook": "capture"}
    if binding.name in guarded and (cap.get("move") != guarded[binding.name] or cap.get("effect") != "write"):
        out.append(f"{node.id}: scientific/export capability must retain its original guarded write authority")
    try:
        if capability_fingerprint(cap) != binding.schema_fingerprint: out.append(f"{node.id}: capability fingerprint mismatch")
    except (ValueError, TypeError): out.append(f"{node.id}: capability metadata is not JSON")
    return out


def _source(graph, source_workflow, source_solution_map):
    """Return admitted source steps, atomic-programme flag and source problems. Never flatten complex leaves."""
    out, steps, atomic = [], {}, False
    if source_workflow is None and source_solution_map is None:
        return {}, False, ["source workflow or solution map is required for provenance validation"]
    if source_workflow is not None and source_solution_map is not None:
        return {}, False, ["supply one authoritative source workflow or solution map"]
    try:
        if source_solution_map is not None:
            sm = _read(maps.SolutionMap, source_solution_map)
            out.extend("source solution map: " + issue for issue in maps.validate(sm))
            refs = [ref for ref in graph.source_refs if ref.kind == "solution_map"]
            if len(refs) != 1 or refs[0].fingerprint != maps.fingerprint(sm): out.append("source solution_map reference fingerprint mismatch")
            atomic = len(sm.blocks) > 1 or sm.holdout.read_by == "composition"
            if atomic:
                if sm.blockers and not any(node.kind == "blocked" for node in graph.nodes): out.append("source solution map blockers require explicit blocked tasks")
                # One node covers the whole validated map through its typed map reference, not repeated leaf IDs.
                calls = [node for node in graph.nodes if node.kind == "action"]
                if len(calls) != 1 or calls[0].capability.name != "run_programme":
                    out.append("complex source requires exactly one atomic run_programme action; no independent leaf/stage execution")
                if any(node.source_step_ids for node in graph.nodes): out.append("atomic programme uses map-level provenance; unqualified leaf source_step_ids are ambiguous")
                gates = [node for node in graph.nodes if node.kind == "gate" and any(check.kind == "approval" for check in node.acceptance_checks)]
                ancestors, _ = _ancestors({node.id: set(node.dependencies) for node in graph.nodes})
                if len(calls) == 1 and not any(gate.id in ancestors[calls[0].id] for gate in gates):
                    out.append("atomic programme requires a separate human gate dependency")
            else:
                steps = {step.id: step for step in sm.blocks[0].workflow.steps}
            expected = workflow_fingerprint(sm)
        else:
            wf = _read(workflows.SolutionWorkflow, source_workflow)
            out.extend("source workflow: " + issue for issue in workflows.validate(wf))
            steps = {step.id: step for step in wf.steps}
            expected = workflow_fingerprint(wf)
            refs = [ref for ref in graph.source_refs if ref.kind == "solution_workflow"]
            if len(refs) != 1 or refs[0].fingerprint != expected: out.append("source solution_workflow reference fingerprint mismatch")
        if graph.workflow_fingerprint != expected: out.append("source workflow fingerprint mismatch")
    except (ValueError, TypeError, AttributeError) as error:
        out.extend("source: " + issue for issue in _shape_problem(error))
    if not any(ref.kind in ("solution", "solution_plan", "solution_map") and ref.fingerprint == graph.solution_fingerprint for ref in graph.source_refs):
        out.append("current solution fingerprint has no matching typed source reference")
    return steps, atomic, out


def validate_graph(graph, source_workflow=None, source_solution_map=None, *, known_capabilities=None, external_artifacts=()) -> list[str]:
    """Validate structure, source coverage, safety ordering, artifact dependencies and declared identity.

    Blocked nodes may honestly represent unsupported requirements; they never become runnable. Actual registry
    membership is checked only when a host snapshot is supplied. Readiness/plan validation require that snapshot.
    """
    try: graph = _read(TaskGraph, graph)
    except (ValueError, TypeError, AttributeError) as error: return _shape_problem(error)
    out = []
    if graph.fingerprint != graph_fingerprint(graph): out.append("graph fingerprint mismatch; create a new revision, never rewrite the stored graph")
    if known_capabilities is not None:
        try: known_capabilities = _snapshot(known_capabilities)
        except ValueError: return _issues([*out, "host capability snapshot must contain JSON metadata only"])
    nodes = {node.id: node for node in graph.nodes}
    links = {node.id: set(node.dependencies) for node in graph.nodes}
    parents = {node.id: ({node.parent_id} if node.parent_id is not None else set()) for node in graph.nodes}
    expected_edges = set()
    for node in graph.nodes:
        if node.parent_id is not None:
            expected_edges.add(("grouping", node.parent_id, node.id))
            parent = nodes.get(node.parent_id)
            if parent is None: out.append(f"{node.id}: absent parent {node.parent_id}")
            elif parent.kind != "group": out.append(f"{node.id}: parent must be a non-executable group")
            if node.parent_id == node.id: out.append(f"{node.id}: self parent")
        if node.kind == "group" and (node.produces or node.needs or node.source_step_ids): out.append(f"{node.id}: group cannot produce/consume artifacts or claim source coverage")
        for dep in node.dependencies:
            expected_edges.add(("dependency", dep, node.id))
            if dep == node.id: out.append(f"{node.id}: self dependency")
            if dep not in nodes: out.append(f"{node.id}: dangling dependency {dep}")
            elif nodes[dep].kind in ("group", "blocked"): out.append(f"{node.id}: dependency on non-satisfiable {nodes[dep].kind} {dep}")
        if known_capabilities is not None:
            out.extend(_capability_problems(node, known_capabilities))
    actual_edges = {(edge.kind, edge.source, edge.target) for edge in graph.edges}
    if actual_edges != expected_edges: out.append("edges must exactly match declared parents and dependencies")
    for edge in graph.edges:
        if edge.source not in nodes or edge.target not in nodes: out.append("edge references an unknown task")
    ancestors, cycles = _ancestors(links)
    _, parent_cycles = _ancestors(parents)
    if cycles: out.append("dependency cycle: " + ", ".join(cycles))
    if parent_cycles: out.append("parent cycle: " + ", ".join(parent_cycles))
    producers = {}
    for node in graph.nodes:
        for artifact in node.produces:
            if artifact in producers: out.append(f"artifact {artifact}: multiple producers")
            producers[artifact] = node.id
    external = set(external_artifacts) | {ref.id for ref in graph.source_refs if ref.kind == "dataset"}
    for node in graph.nodes:
        for artifact in node.needs:
            if artifact in producers:
                owner = producers[artifact]
                if owner not in ancestors[node.id] or nodes[owner].kind not in ("action", "gate", "verification"):
                    out.append(f"{node.id}: artifact {artifact} has no satisfiable producer dependency")
            elif artifact not in external: out.append(f"{node.id}: input artifact {artifact} is undeclared")
    steps, atomic, source_issues = _source(graph, source_workflow, source_solution_map)
    out.extend(source_issues)
    owners = {}
    for node in graph.nodes:
        for sid in node.source_step_ids:
            if sid not in steps: out.append(f"{node.id}: unknown source step {sid}")
            owners.setdefault(sid, []).append(node)
    if not atomic:
        for sid, step in steps.items():
            covered = owners.get(sid, [])
            if len(covered) != 1 or covered[0].kind == "group": out.append(f"source step {sid} needs exactly one non-group owner")
            if step.gate and covered and covered[0].kind not in ("gate", "blocked"): out.append(f"{sid}: human gate must remain a separate gate task")
        def before(a, b, allow_same=False):
            if len(owners.get(a, [])) == len(owners.get(b, [])) == 1:
                left, right = owners[a][0], owners[b][0]
                if (left.id == right.id and not allow_same) or (left.id != right.id and left.id not in ancestors[right.id]): out.append(f"{a} must be a dependency before {b}")
        for left, right in zip(_ORDER, _ORDER[1:]): before(left, right, allow_same=(left, right)==("lock_holdout", "train_only_analysis"))
        order = list(steps)
        lock_index = order.index("lock_holdout") if "lock_holdout" in steps else -1
        for index, (sid, step) in enumerate(steps.items()):
            for owner in owners.get(sid, []):
                if index > lock_index >= 0 and len(owners.get("lock_holdout", [])) == 1:
                    lock = owners["lock_holdout"][0]
                    if owner.id != lock.id and lock.id not in ancestors[owner.id]: out.append(f"{sid}: source step needs lock_holdout before analysis/decisions")
                if step.block != "WF-10" and sid != "final_once" and len(owners.get("final_once", [])) == 1:
                    final = owners["final_once"][0]
                    if owner.id != final.id and owner.id not in ancestors[final.id]: out.append(f"{sid}: source work cannot be placed after final_once")
                if step.block == "WF-10" and len(owners.get("final_once", [])) == 1:
                    final = owners["final_once"][0]
                    if owner.id != final.id and final.id not in ancestors[owner.id]: out.append(f"{sid}: WF-10 reporting/evidence needs final_once first")
                if owner.kind == "action" and sid in _STAGE_STEPS and owner.capability.name != "run_stage":
                    out.append(f"{sid}: physical stage work must use the existing run_stage authority, or a non-executing evidence check")
                if owner.kind == "action" and sid == "report_limits" and owner.capability.name != "export_notebook":
                    out.append("report_limits: action must use the supported registered export authority")
        audits = owners.get("leakage_audit", [])
        finals = owners.get("final_once", [])
        for sid, step in steps.items():
            if step.decision not in ("model.families", "threshold.policy"): continue
            for owner in owners.get(sid, []):
                if len(audits) == 1 and audits[0].id not in ancestors[owner.id]: out.append(f"{sid}: model/threshold choice needs leakage_audit dependency")
                if len(finals) == 1 and owner.id != finals[0].id and owner.id not in ancestors[finals[0].id]:
                    out.append(f"{sid}: selection cannot occur after final_once")
    # Graph bindings omit raw arguments. Critical source steps identify their containing physical stage; v1
    # allows one other run_stage binding (features). An ambiguous second binding must not hide a duplicate fit.
    effects = set()
    for node in graph.nodes:
        if node.kind != "action": continue
        cap = known_capabilities.get(node.capability.name) if known_capabilities is not None else None
        move = cap.get("move") if isinstance(cap, Mapping) else node.capability.name
        if move is not None and not isinstance(move, str): continue
        if move == "run_stage":
            phases = {_STAGE_STEPS[sid] for sid in node.source_step_ids if sid in _STAGE_STEPS}
            if len(phases) > 1: out.append(f"{node.id}: one physical stage effect cannot contain multiple stage phases")
            effect = (node.scope, move, next(iter(phases)) if len(phases) == 1 else "unclassified")
        elif isinstance(cap, Mapping) and (cap.get("effect") == "write" or cap.get("replay_safe") is not True):
            effect = (node.scope, move or node.capability.name)
        else: continue
        if effect in effects: out.append(f"{node.id}: duplicate non-idempotent engine effect")
        effects.add(effect)
    programme = [node for node in graph.nodes if node.kind == "action" and node.capability.name == "run_programme"]
    if len(programme) > 1: out.append("duplicate non-idempotent programme effect")
    if programme and not atomic: out.append("run_programme requires an admitted atomic solution map, not flat stage execution")
    return _issues(out)


def _state_problems(graph, state):
    out = []
    for key, expected in (("graph_id", graph.graph_id), ("graph_revision", graph.revision), ("graph_fingerprint", graph.fingerprint),
                          ("solution_fingerprint", graph.solution_fingerprint), ("workflow_fingerprint", graph.workflow_fingerprint)):
        if state.get(key) != expected or (key == "graph_revision" and type(state.get(key)) is not int): out.append("observed " + key.replace("_", " ") + " is stale")
    artifacts = state.get("artifacts", {})
    if not isinstance(artifacts, Mapping): return out + ["observed artifact metadata is missing"]
    for ident, meta in artifacts.items():
        if not isinstance(meta, Mapping) or not _digest(meta.get("fingerprint")) or meta.get("value_type") not in ("artifact", "string", "integer", "number", "boolean"):
            out.append("host artifact metadata has an invalid fingerprint or value type")
        elif "value" in meta and not _metadata_value(meta): out.append("host input metadata has an invalid typed value or fingerprint")
    for ref in graph.source_refs:
        if ref.kind == "dataset" and (not isinstance(artifacts.get(ref.id), Mapping) or artifacts[ref.id].get("fingerprint") != ref.fingerprint):
            out.append(f"input artifact {ref.id} fingerprint changed or is missing")
    if not isinstance(state.get("project_id"), str) or not state["project_id"]: out.append("host project context is missing")
    return out


def _approval_problems(graph, state, holdout=False, *, historical=False):
    name = "holdout_approval" if holdout else "approval"
    approval = state.get(name)
    if not isinstance(approval, Mapping): return [name + " is missing"]
    out = []
    if approval.get("project_id") != state.get("project_id"): out.append(name + " project binding is stale")
    if approval.get("approved") is not True or approval.get("solution_fingerprint") != graph.solution_fingerprint:
        out.append(name + " is not current for this solution")
    if not _reference(approval.get("reference_id")) or not _digest(approval.get("fingerprint"), (16, 64)): out.append(name + " has no evidence reference")
    if approval.get("workflow_fingerprint") != graph.workflow_fingerprint: out.append(name + " workflow binding is stale")
    sources = approval.get("sources")
    expected_sources = {(ref.kind, ref.id): ref.fingerprint for ref in graph.source_refs if ref.kind != "approval"}
    actual_sources = {}
    if isinstance(sources, (list, tuple)):
        for ref in sources:
            if isinstance(ref, Mapping) and isinstance(ref.get("kind"), str) and isinstance(ref.get("id"), str):
                actual_sources[(ref["kind"], ref["id"])] = ref.get("fingerprint")
    if not isinstance(sources, (list, tuple)) or len(actual_sources) != len(sources) or actual_sources != expected_sources:
        out.append(name + " approved source references are stale or incomplete")
    if not holdout and not any(ref.kind == "approval" and ref.id == approval.get("reference_id") and ref.fingerprint == approval.get("fingerprint") for ref in graph.source_refs):
        out.append("solution approval does not match the declared source reference")
    inputs = approval.get("inputs", {})
    if not isinstance(inputs, Mapping): return out + [name + " input bindings are missing"]
    for ref in graph.source_refs:
        if ref.kind == "dataset" and inputs.get(ref.id) != ref.fingerprint: out.append(name + " input binding is stale")
    for ident, digest in inputs.items():
        meta = state.get("artifacts", {}).get(ident, {})
        if not isinstance(meta, Mapping) or meta.get("fingerprint") != digest: out.append(name + " input artifact binding changed")
    if holdout and not historical and (approval.get("used") is not False or type(state.get("holdout_uses")) is not int or state["holdout_uses"] != 0): out.append("holdout permission is used or consumption is unknown")
    return out


def _verified_tasks(graph, state):
    """Require host verification references and current input/output receipts, never an agent's done flag."""
    verified, out, inflight = set(), [], False
    nodes = {node.id: node for node in graph.nodes}
    artifacts = state.get("artifacts", {})
    records = state.get("tasks", {})
    if not isinstance(records, Mapping): return set(), ["task observations must be host metadata"], False
    for ident, record in records.items():
        if ident not in nodes or not isinstance(record, Mapping): out.append("observation references an unknown task"); continue
        node, status = nodes[ident], record.get("status")
        if status == "pending": continue
        if status in ("failed", "blocked", "stale"):
            inflight = True; continue
        if status == "running": inflight = True; continue
        if status != "verified": out.append(f"{ident}: unknown outcome or unverified completion claim"); continue
        if node.kind in ("group", "blocked"): out.append(f"{ident}: non-executable group/blocked task is not completion evidence"); continue
        if not _reference(record.get("verification_ref")): out.append(f"{ident}: verified state has no verification record")
        if type(record.get("graph_revision")) is not int or record.get("graph_revision") != graph.revision: out.append(f"{ident}: receipt revision is stale")
        if record.get("graph_fingerprint") != graph.fingerprint or record.get("project_id") != state.get("project_id"):
            out.append(f"{ident}: receipt source/project identity is stale")
        inputs, outputs = record.get("inputs", {}), record.get("outputs", {})
        if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping): out.append(f"{ident}: artifact receipts are missing"); continue
        if not set(node.needs) <= set(inputs) or set(outputs) != set(node.produces): out.append(f"{ident}: receipt artifact set is incomplete")
        for field, values in (("input", inputs), ("output", outputs)):
            for artifact, digest in values.items():
                meta = artifacts.get(artifact, {})
                if not isinstance(meta, Mapping) or not _digest(digest) or meta.get("fingerprint") != digest:
                    out.append(f"{ident}: {field} artifact {artifact} fingerprint changed")
                elif field == "output" and meta.get("producer_task") != ident: out.append(f"{ident}: output artifact producer mismatch")
        if node.kind == "gate":
            out.extend(_approval_problems(graph, state, True, historical=True))
            approval = state.get("holdout_approval")
            approval = approval if isinstance(approval, Mapping) else {}
            bindings = approval.get("inputs", {})
            if not isinstance(bindings, Mapping) or any(bindings.get(artifact) != digest for artifact, digest in inputs.items()):
                out.append(f"{ident}: approval does not bind the gate's current prerequisite artifacts")
            ref = record.get("approval_ref", {})
            if not approval.get("approved") or ref != {"id": approval.get("reference_id"), "fingerprint": approval.get("fingerprint")}:
                out.append(f"{ident}: human gate has no matching owner approval evidence")
        verified.add(ident)
    for ident in verified:
        if not set(nodes[ident].dependencies) <= verified: out.append(f"{ident}: verified claim lacks verified dependencies")
    return verified, out, inflight


def ready_tasks(graph, observed_state, *, source_workflow=None, source_solution_map=None) -> list[str]:
    """Return at most ONE deterministic next action, without executing or recording anything.

    Source/state mismatch raises ValueError, rather than being confused with an idle graph. A pending approval,
    blocked graph or in-flight action returns no action. Gates and verifications need independent host evidence.
    """
    graph = _read(TaskGraph, graph)
    state = _snapshot(observed_state)
    capabilities = state.get("capabilities")
    if not isinstance(capabilities, Mapping): raise ValueError("host capability snapshot is required")
    artifacts = state.get("artifacts", {})
    problems = validate_graph(graph, source_workflow, source_solution_map, known_capabilities=capabilities,
                              external_artifacts=artifacts if isinstance(artifacts, Mapping) else ())
    problems.extend(_state_problems(graph, state))
    if problems: raise ValueError("; ".join(_issues(problems)))
    verified, receipt_issues, inflight = _verified_tasks(graph, state)
    problems.extend(receipt_issues)
    for node in graph.nodes:
        if node.kind == "action" and node.id not in verified and any(artifact in artifacts for artifact in node.produces):
            problems.append(f"{node.id}: unverified outputs already exist; reconcile the owner record before retrying")
    if problems: raise ValueError("; ".join(_issues(problems)))
    if inflight or _approval_problems(graph, state) or any(node.kind == "blocked" for node in graph.nodes): return []
    candidates = []
    for node in sorted(graph.nodes, key=lambda item: item.id):
        if node.kind != "action" or node.id in verified or not set(node.dependencies) <= verified: continue
        if not all(artifact in state["artifacts"] for artifact in node.needs): continue
        if ("final_once" in node.source_step_ids or node.capability.name == "run_programme") and _approval_problems(graph, state, True): continue
        candidates.append(node.id)
    return candidates[:1]


def _metadata_value(meta):
    value = meta.get("value")
    kind = meta.get("value_type")
    valid = (kind == "string" and isinstance(value, str) and len(value) <= 1000 or
             kind == "boolean" and type(value) is bool or kind == "integer" and type(value) is int or
             kind == "number" and type(value) in (float, int) and math.isfinite(value))
    return valid and metadata_fingerprint(value) == meta.get("fingerprint")


def validate_execution_plan(plan, graph, known_capabilities, *, source_workflow=None, source_solution_map=None, observed_state=None) -> list[str]:
    """Check a plan against its immutable task, admitted sources and current host-only metadata.

    Full Registry.check argument-schema validation, principal authentication, locks, stop/spend controls, approval
    consumption and engine holdout accounting remain mandatory at execution time. This never calls the registry.
    """
    try:
        plan, graph = _read(TaskExecutionPlan, plan), _read(TaskGraph, graph)
    except (ValueError, TypeError, AttributeError) as error: return _shape_problem(error)
    try:
        known_capabilities = _snapshot(known_capabilities)
        if observed_state is not None: observed_state = _snapshot(observed_state)
    except ValueError: return ["host capability/observed snapshot must contain JSON metadata only"]
    artifacts = observed_state.get("artifacts", {}) if isinstance(observed_state, Mapping) else {}
    out = validate_graph(graph, source_workflow, source_solution_map, known_capabilities=known_capabilities,
                         external_artifacts=artifacts if isinstance(artifacts, Mapping) else ())
    if out: return out
    if plan.fingerprint != plan_fingerprint(plan): out.append("plan fingerprint mismatch; do not rewrite the stored plan")
    if (plan.graph_id, plan.graph_revision, plan.graph_fingerprint) != (graph.graph_id, graph.revision, graph.fingerprint): out.append("plan graph identity/revision is stale")
    node = next((item for item in graph.nodes if item.id == plan.task_id), None)
    if node is None: return _issues([*out, "plan references an unknown task"])
    if plan.status in ("noop", "blocked"): return _issues(out)
    if plan.status != "ready": out.append("plan is not ready; proposal status grants no execution permission")
    if node.kind != "action": return _issues([*out, "non-action task cannot contain execution intents"])
    if not isinstance(observed_state, Mapping): return _issues([*out, "current host observed state and approval are required"])
    state = dict(observed_state); state["capabilities"] = known_capabilities
    out.extend(_state_problems(graph, state))
    if out: return _issues(out)
    out.extend(_approval_problems(graph, state))
    try:
        if plan.task_id not in ready_tasks(graph, state, source_workflow=source_workflow, source_solution_map=source_solution_map): out.append("task dependencies, approval or in-flight state prevent readiness")
    except (ValueError, TypeError, AttributeError) as error: out.append(str(error))
    artifacts = state.get("artifacts", {})
    expected = {item.id: item for item in plan.expected_inputs}
    if not set(node.needs) <= set(expected): out.append("plan omits required task input artifacts")
    for ident, item in expected.items():
        meta = artifacts.get(ident)
        if not isinstance(meta, Mapping) or (meta.get("fingerprint"), meta.get("value_type")) != (item.fingerprint, item.value_type):
            out.append(f"input artifact {ident} is missing, changed or has a different type")
        elif "value" in meta and not _metadata_value(meta): out.append(f"input metadata {ident} fingerprint/type mismatch")
    if set(item.id for item in plan.declared_outputs) != set(node.produces): out.append("plan outputs do not match task artifacts")
    checks = {check.id: check for check in plan.acceptance_checks}
    for required in node.acceptance_checks:
        if checks.get(required.id) != required: out.append(f"plan weakens task acceptance check {required.id}")
    selected = {(binding.name, binding.scope): binding for binding in plan.selected_capabilities}
    primary = (node.capability.name, node.capability.scope)
    if selected.get(primary) != node.capability: out.append("plan does not preserve task capability binding")
    for binding in plan.selected_capabilities:
        cap = known_capabilities.get(binding.name)
        if not isinstance(cap, Mapping): out.append(f"missing capability {binding.name}"); continue
        if cap.get("name") != binding.name or cap.get("scope") != binding.scope or cap.get("allowed") is not True:
            out.append(f"capability {binding.name} scope/permission mismatch")
        if capability_fingerprint(cap) != binding.schema_fingerprint: out.append(f"capability {binding.name} fingerprint mismatch")
    effects, primary_calls = set(), 0
    for action in plan.actions:
        cap = known_capabilities.get(action.capability_name)
        if not isinstance(cap, Mapping): continue
        if (action.capability_name, action.scope) == primary: primary_calls += 1
        elif cap.get("effect") != "read" or cap.get("replay_safe") is not True:
            out.append("additional plan actions must be explicitly replay-safe reads")
        if action.scope != cap.get("scope"): out.append(f"{action.id}: cross-scope tool")
        args, resolved = {argument.name: argument for argument in action.arguments}, {}
        schema = cap.get("parameters", {})
        if set(args) - set(schema.get("properties", {})): out.append(f"{action.id}: unknown tool argument name")
        if set(schema.get("required", ())) - set(args): out.append(f"{action.id}: required tool arguments are missing")
        for name, argument in args.items():
            ref = argument.value
            meta = artifacts.get(ref.ref_id, {}) if ref.kind == "input" else {}
            if ref.kind == "input" and "value" in meta and _metadata_value(meta): resolved[name] = meta["value"]
        if action.scope == "project" and "project_id" in schema.get("properties", {}):
            if resolved.get("project_id") != state.get("project_id"): out.append(f"{action.id}: project argument must match bound host project context")
        move = cap.get("move")
        effect = (action.scope, move or action.capability_name)
        if move == "run_stage":
            stage = resolved.get("stage")
            if stage not in ("data", "leakage", "features", "models", "final"):
                out.append(f"{action.id}: stage requires declared fingerprinted metadata, not an aggregate run_all or deferred guess")
            effect += (stage,)
            required = {"data": "lock_holdout", "leakage": "leakage_audit", "models": "choose_on_validation", "final": "final_once"}.get(stage)
            if required and required not in node.source_step_ids: out.append(f"{action.id}: {stage} effect is not the task's declared source operation")
            if stage == "features":
                ancestors, _ = _ancestors({item.id: set(item.dependencies) for item in graph.nodes})
                if not any(item.id in ancestors[node.id] and "leakage_audit" in item.source_step_ids for item in graph.nodes): out.append("features effect must follow leakage_audit")
            if stage == "final": out.extend(_approval_problems(graph, state, True))
        if move == "run_programme":
            if source_solution_map is None: out.append("programme effect requires an atomic admitted map")
            out.extend(_approval_problems(graph, state, True))
        if cap.get("effect") == "write" or cap.get("replay_safe") is not True:
            if effect in effects: out.append(f"{action.id}: duplicate non-idempotent engine effect")
            effects.add(effect)
            if plan.policy.max_retries: out.append(f"{action.id}: non-idempotent effect cannot carry automatic retry allowance")
    if primary_calls != 1: out.append("task requires exactly one primary capability effect")
    return _issues(out)


__all__ = ["validate_graph", "ready_tasks", "validate_execution_plan"]
