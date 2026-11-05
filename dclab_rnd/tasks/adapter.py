"""TP-01D: one binary workflow compiled to existing guarded operations, without executing them.

The caller supplies current project metadata and a detached registry snapshot. Neither a store nor a runtime
context is accepted. A mapped graph is an inspectable proposal, not execution permission. Missing requirements
are explicit blockers; no dataset, capability, approval or successful output is invented.
"""
from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from typing import Any

import jsonschema
from pydantic import ValidationError

from dclab_core.contracts import SolutionPlan
from dclab_core.card import problem_card
from dclab_core.sensitivity import CLASSES as SENSITIVITY_CLASSES
from dclab_core.solution_workflow import FAMILIES, SolutionWorkflow, validate as validate_workflow
from dclab_core.task_fingerprints import (capability_fingerprint, graph_fingerprint, metadata_fingerprint,
                                          workflow_fingerprint)
from dclab_core.task_graph import TaskGraph
from dclab_core.task_validation import validate_graph
from dclab_rnd.core_adapter import brief_from_project, plan_from_project
from dclab_rnd.studio.graph import solution_hash, solution_signed
from dclab_rnd.studio.solution import Solution
from dclab_rnd.kinds import resolve as resolve_kind

VERSION = "tp01d.v1"
TOOLS = {"run_stage": "run_stage", "export_notebook": "capture"}
ORDER = ("data", "leakage", "features", "models", "holdout", "final", "report")
OWNERS = {"class_balance": "data", "lock_holdout": "data", "train_only_analysis": "data",
          "leakage_audit": "leakage", "choose_on_validation": "models", "holdout_gate": "holdout",
          "threshold": "final", "final_once": "final", "report_limits": "report"}
GROUPS = {"preparation": ("Prepare locked training data", ("data", "leakage", "features")),
          "selection": ("Select on training validation", ("models",)),
          "evaluation": ("Human approval and one final score", ("holdout", "final")),
          "reporting": ("Report the final evidence and limits", ("report",))}
# These are references to existing owner records, not new training/holdout table files.
OUTPUTS = {"data": "data_record", "leakage": "leakage_record", "features": "features_record",
           "models": "models_record", "holdout": "holdout_permission", "final": "final_record", "report": "report_files"}
NEEDS = {"data": ("table", "project_context"),
         "leakage": ("table", "project_context", "data_record"),
         "features": ("table", "project_context", "data_record", "leakage_record"),
         "models": ("table", "project_context", "data_record", "leakage_record", "features_record"),
         "holdout": ("models_record",),
         "final": ("table", "project_context", "data_record", "models_record", "holdout_permission"),
         "report": ("project_context", "data_record", "final_record")}
EVIDENCE = {"data": ["evidence.holdout_split", "evidence.cv_protocol", "evidence.target_summary", "provenance"],
            "leakage": ["evidence", "provenance"], "features": ["evidence", "provenance"],
            "models": ["evidence", "provenance"], "holdout": ["owner human approval record"],
            "final": ["evidence.task_assessment.threshold", "evidence.holdout_metrics", "evidence.holdout_primary_metric_ci",
                      "evidence.holdout_consumed", "evidence.holdout_uses_in_this_project", "run"],
            "report": ["notebook.ipynb", "report.md", "source final record and artifact hashes"]}
ARTIFACTS = {"table": "table", "training rows": "data_record", "holdout": "data_record", "findings": "data_record",
             "columns kept": "leakage_record", "chosen model": "models_record", "approval": "holdout_permission",
             "holdout score": "final_record", "report": "report_files"}


def _json(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise ValueError("task adapter inputs must contain JSON metadata only") from None


def capability_snapshot(registry, *, allowed_tools: Collection[str]) -> dict[str, dict]:
    """Extract canonical tool metadata only. Membership does not grant permission; never invoke a guard/handler.

    The explicit host allowlist is an upper bound for planning, not a replacement for live principal, workflow,
    stop or budget checks. The complete parameter schema comes from the existing registry, never a local copy.
    """
    allowed = set(allowed_tools)
    result = {}
    for name in TOOLS:
        tool = registry.tools.get(name)
        if tool is None:
            continue
        guarded = tool.effect != "write" or tool.scope in registry.guards
        result[name] = _json({"name": tool.name, "scope": tool.scope, "effect": tool.effect, "move": tool.move,
                             "parameters": tool.parameters, "allowed": name in allowed and guarded,
                             "guard_present": guarded, "replay_safe": False})
    return result


def _report(blockers=()):
    return {"adapter_version": VERSION, "status": "blocked" if blockers else "mapped", "blockers": list(blockers),
            "source_mapping": [], "operations": [], "external_artifacts": {}, "topological_order": [],
            "validation_issues": [], "execution_authorized": False,
            "verification_limits": [
                {"source_step_id": "class_balance", "task_id": "data",
                 "limit": "The protocol and target summary do not persist per-fold class counts; a verifier must require measured proof, not infer completion."},
                {"source_step_id": "threshold", "task_id": "final",
                 "limit": "OOF threshold proof is optional in task_assessment; missing proof blocks verification and never authorizes another final read."},
                {"source_step_id": "report_limits", "task_id": "report",
                 "limit": "The registered export writes legacy report.md and notebook.ipynb, not canonical SolutionReport; export success alone does not prove its limits or numeric lineage."}]}


def _plan_problems(plan, project):
    """Only decisions actually represented by the saved studio solution are admitted in this first adapter."""
    expected = plan_from_project(project)
    out = []
    if plan.task != "binary":
        out.append("Source plan is not a binary classification plan")
    for decision in expected.decisions:
        if decision.key == "solution.notes":
            continue
        proposed = plan.decision(decision.key)
        if proposed is None or proposed.choice != decision.choice:
            out.append("Source plan does not preserve saved solution decision " + decision.key)
    for decision in plan.decisions:
        if expected.decision(decision.key) is not None:
            continue
        if decision.key == "task.framing" and isinstance(decision.choice, Mapping) and decision.choice == {"task": "binary"}:
            continue
        if decision.key == "labels.coverage" and decision.choice == "complete":
            continue
        if decision.key == "split.strategy" and decision.choice in ("stratified", {"kind": "stratified"}):
            continue
        out.append("Source plan decision has no adopted studio binding: " + decision.key)
    return out


def _execution_context(project, data):
    """Fingerprint the supplied execution inputs without reading previews, categories, records or workspace state.

    This is a host solution-context identity, not a replacement for the existing signed Solution hash. The runtime
    still owns code/runner/routing identity, actual bytes, current receipts, approvals and concurrency control.
    """
    context = {key: project.get(key) for key in ("settings", "feature_programs", "plan", "brief", "exposure",
                                               "adopted_improvement", "name", "goal", "programme_plan")}
    draft = project.get("draft")
    draft_pack = draft.get("pack") if isinstance(draft, Mapping) else None
    context["pack"] = project.get("pack") or (draft_pack.get("key") if isinstance(draft_pack, Mapping) else None)
    context["categorical"] = data.get("categorical")
    context["columns"] = data.get("columns")
    proposal = project.get("proposal")
    proposal = proposal if isinstance(proposal, Mapping) else {}
    rule = proposal.get("rule") or proposal
    context["proposal"] = {"target": proposal.get("target"), "modifiers": rule.get("modifiers") if isinstance(rule, Mapping) else None}
    profile = data.get("profile")
    columns = profile.get("columns", []) if isinstance(profile, Mapping) else []
    problems, labels = [], []
    if not isinstance(columns, list):
        problems.append("Current profile sensitivity metadata is invalid")
    else:
        for column in columns:
            if not isinstance(column, Mapping) or not isinstance(column.get("name"), str) or column.get("sensitivity") not in SENSITIVITY_CLASSES:
                problems.append("Legacy profile sensitivity needs explicit classification; preview values are not read during planning")
                continue
            labels.append({"name": column["name"], "sensitivity": column["sensitivity"]})
    context["sensitivity"] = sorted(labels, key=lambda c: c["name"])
    return _json(context), problems


def _capability_problems(name, caps, project_id, stage):
    cap = caps.get(name)
    if not isinstance(cap, Mapping):
        return ["Required registered capability is missing: " + name]
    if (cap.get("name"), cap.get("scope"), cap.get("effect"), cap.get("move")) != (name, "project", "write", TOOLS[name]):
        return ["Registered capability scope/effect/move differs from the existing authority: " + name]
    if cap.get("allowed") is not True or cap.get("guard_present") is not True:
        return ["Host policy or missing project guard blocks capability: " + name]
    args = {"project_id": project_id}
    if stage is not None:
        args["stage"] = stage
    try:
        schema = cap["parameters"]
        pending = [schema]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                if any(key in item for key in ("$ref", "$dynamicRef", "$recursiveRef")):
                    return ["Schema references require resolution outside this non-executing adapter: " + name]
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        jsonschema.Draft202012Validator.check_schema(schema)
        errors = list(jsonschema.Draft202012Validator(schema).iter_errors(args))
        if errors or not isinstance(schema, Mapping) or schema.get("type") != "object":
            return ["Registered argument schema cannot represent this existing operation: " + name]
    except (KeyError, TypeError, jsonschema.SchemaError):
        return ["Registered capability has an invalid parameter schema: " + name]
    return []


def adapt_workflow(workflow, source_plan, project_metadata: Mapping[str, Any], *, capabilities=None, revision=1
                   ) -> tuple[TaskGraph | None, dict[str, Any]]:
    """Return an immutable proposal and a JSON mapping report. Never read/write a project or table, or run a tool.

    Invalid workflow contracts/content raise ValueError. Missing identities and unsupported source variants return
    (None, blocked report) rather than fabricated references. With identities present, unavailable capabilities or
    unconfirmed source decisions appear as blocked tasks. Such diagnostic graphs are not admitted for readiness;
    the report retains the existing core validator's issues until their prerequisites can actually be satisfied.
    """
    try:
        wf = SolutionWorkflow.read(workflow.model_dump(mode="json") if isinstance(workflow, SolutionWorkflow) else workflow)
    except (ValueError, TypeError, AttributeError):
        raise ValueError("source workflow has an invalid contract shape") from None
    if wf.family not in FAMILIES:
        return None, _report(["Source workflow family is unknown"])
    problems = validate_workflow(wf)
    if problems:
        raise ValueError("source workflow rejected: " + "; ".join(problems))
    if wf.family != "classification" or set(s.id for s in wf.steps) != set(OWNERS):
        return None, _report(["This adapter supports the basic binary classification workflow only; extra source work needs an explicit mapping"])
    if source_plan is None:
        return None, _report(["Current source SolutionPlan metadata is missing"])
    try:
        plan = SolutionPlan.read(source_plan.model_dump(mode="json") if isinstance(source_plan, SolutionPlan) else source_plan)
    except (ValueError, TypeError, AttributeError):
        raise ValueError("source plan has an invalid contract shape") from None
    if not isinstance(project_metadata, Mapping):
        return None, _report(["Current project metadata is missing"])
    pid = project_metadata.get("id")
    raw_solution, raw_data = project_metadata.get("solution"), project_metadata.get("data")
    if not isinstance(pid, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", pid) is None:
        return None, _report(["Current project identity is missing or invalid"])
    if not isinstance(raw_solution, Mapping) or not raw_solution:
        return None, _report(["Current saved solution metadata is missing"])
    if not isinstance(raw_data, Mapping) or not isinstance(raw_data.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", raw_data["sha256"]) is None:
        return None, _report(["Current recorded dataset fingerprint is missing or invalid"])
    # Do not copy/read profiles, previews, records, paths, credentials or other project state.
    project = {"id": pid, "solution": _json(dict(raw_solution)), "data": {"sha256": raw_data["sha256"]},
               "goal": project_metadata.get("goal") if isinstance(project_metadata.get("goal"), str) else ""}
    raw_signoff = project_metadata.get("signoff")
    if isinstance(raw_signoff, Mapping):
        project["signoff"] = _json({key: raw_signoff.get(key) for key in ("by", "at", "solution_hash")})
    caps = _json(dict(capabilities)) if isinstance(capabilities, Mapping) else {}
    try:
        solution = Solution.model_validate(project["solution"])
    except ValidationError:
        return None, _report(["Current saved solution does not satisfy its existing contract"])
    blockers = _plan_problems(plan, project)
    if not isinstance(project_metadata.get("settings"), Mapping):
        blockers.append("Current execution settings metadata is required; defaults cannot be guessed")
    context, context_issues = _execution_context(project_metadata, raw_data)
    blockers.extend(context_issues)
    context_fp = metadata_fingerprint(context)
    runtime_plan_fp = None
    if project_metadata.get("plan"):
        try:
            runtime_plan = SolutionPlan.read(_json(project_metadata["plan"]))
            runtime_plan_fp = metadata_fingerprint(runtime_plan.model_dump(mode="json"))
            if runtime_plan_fp != metadata_fingerprint(plan.model_dump(mode="json")):
                blockers.append("Supplied source plan differs from the saved runtime threshold/label plan")
        except (ValueError, TypeError, AttributeError):
            blockers.append("Saved runtime plan metadata cannot be verified")
    try:
        kind = resolve_kind(solution, context["pack"])
        basic_kind = kind.key == "binary"
    except (KeyError, ValueError, TypeError):
        basic_kind = False
    if solution.task != "binary" or not basic_kind or solution.text_columns:
        blockers.append("Current saved solution is outside this basic binary adapter")
    if solution.time_column or solution.group_column or solution.spatial or solution.labels != "complete" or project_metadata.get("labels"):
        blockers.append("Time/group/spatial splits or partial/attached labels require a separately reviewed source mapping")
    if any(project_metadata.get(key) for key in ("programme_plan", "programme", "solution_map", "tables")):
        blockers.append("Programme, solution-map and related-table projects require their existing atomic owner; this adapter cannot flatten them")
    columns = raw_data.get("columns")
    if not isinstance(columns, list) or not columns or any(not isinstance(c, str) for c in columns):
        blockers.append("Current dataset column metadata is missing or invalid")
    else:
        try:
            solution.check_columns(columns)
        except ValueError:
            blockers.append("Saved solution columns do not match current dataset metadata")
    signoff = project.get("signoff", {})
    confirmed = solution_signed(project) and all(isinstance(signoff.get(key), str) and signoff[key].strip() for key in ("by", "at"))
    if not confirmed:
        blockers.append("Outcome and prediction contract require current human solution signoff")
    report = _report(blockers)
    s_fp, w_fp = solution_hash(project["solution"]), workflow_fingerprint(wf)
    refs = [{"kind": "solution", "id": "project_solution", "fingerprint": s_fp},
            {"kind": "solution_plan", "id": "source_plan", "fingerprint": metadata_fingerprint(plan.model_dump(mode="json"))},
            {"kind": "solution_workflow", "id": "workflow", "fingerprint": w_fp},
            {"kind": "problem_card", "id": "problem_context", "fingerprint": metadata_fingerprint(problem_card(brief_from_project(project)).model_dump(mode="json"))},
            {"kind": "solution", "id": "execution_context", "fingerprint": context_fp},
            {"kind": "dataset", "id": "table", "fingerprint": raw_data["sha256"]}]
    if confirmed:
        refs.append({"kind": "approval", "id": "solution_signoff", "fingerprint": metadata_fingerprint(signoff)})
    if runtime_plan_fp is not None:
        refs.append({"kind": "solution_plan", "id": "runtime_plan", "fingerprint": runtime_plan_fp})
    report["external_artifacts"] = {"table": {"fingerprint": raw_data["sha256"], "value_type": "artifact"},
                                     "execution_context": {"fingerprint": context_fp, "value_type": "artifact"},
                                     "project_context": {"fingerprint": metadata_fingerprint(pid), "value_type": "string"}}
    nodes = [{"id": "root", "kind": "group", "title": "Binary classification solution",
              "objective": "Solve the confirmed binary outcome using the current saved solution and its guarded workflow."}]
    for group, (title, _) in GROUPS.items():
        nodes.append({"id": group, "parent_id": "root", "kind": "group", "title": title, "objective": title})
    for index, ident in enumerate(ORDER):
        owned = [step for step in wf.steps if OWNERS[step.id] == ident]
        parent = next(group for group, (_, children) in GROUPS.items() if ident in children)
        checks = [{"id": "source_" + step.id, "kind": "approval" if step.gate else "workflow_criterion",
                   "description": step.check, "evidence_refs": [OUTPUTS[ident]]} for step in owned]
        if ident == "features":
            checks.append({"id": "engine_features", "kind": "guard", "description": "The existing features stage selects on training validation only", "evidence_refs": [OUTPUTS[ident]]})
        checks.append({"id": ident + "_lineage", "kind": "lineage", "description": "Current owner evidence binds the source dataset, solution and declared operation", "evidence_refs": [OUTPUTS[ident]]})
        node = {"id": ident, "parent_id": parent, "kind": "gate" if ident == "holdout" else "action", "title": ident,
                "objective": owned[-1].purpose if owned else "Use the existing engine's training-only feature selection stage.",
                "source_step_ids": [step.id for step in owned], "dependencies": list(ORDER[index-1:index]) if index else [],
                "needs": [*NEEDS[ident], "execution_context"], "produces": [OUTPUTS[ident]], "acceptance_checks": checks}
        if ident != "holdout":
            name, stage = ("export_notebook", None) if ident == "report" else ("run_stage", ident)
            local = [*blockers, *_capability_problems(name, caps, pid, stage)]
            if local:
                node.update(kind="blocked", blocker_reason="; ".join(local)[:1000])
                report["blockers"].extend(local)
            else:
                node.update(scope="project", capability={"name": name, "scope": "project", "schema_fingerprint": capability_fingerprint(caps[name])})
            refs_in = {"project_id": "project_context"}
            if stage:
                key = "stage_" + stage
                report["external_artifacts"][key] = {"fingerprint": metadata_fingerprint(stage), "value_type": "string"}
                refs_in["stage"] = key
            report["operations"].append({"task_id": ident, "capability": name, "move": TOOLS[name], "stage": stage,
                                          "status": "mapped" if node["kind"] == "action" else "blocked",
                                          "argument_refs": refs_in, "evidence_paths": EVIDENCE[ident]})
        nodes.append(node)
        for step in owned:
            report["source_mapping"].append({"source_step_id": step.id, "task_id": ident, "block": step.block,
                                              "kind": step.kind, "decision": step.decision, "gate": step.gate,
                                              "inputs": step.inputs, "outputs": step.outputs, "check": step.check,
                                              "rule": step.rule, "rests_on": step.rests_on, "columns": step.columns,
                                              "input_artifact_refs": {role: ARTIFACTS[role] for role in step.inputs},
                                              "output_artifact_refs": {role: ARTIFACTS[role] for role in step.outputs}})
    edges = [{"kind": "grouping", "source": node["parent_id"], "target": node["id"]} for node in nodes if "parent_id" in node]
    edges.extend({"kind": "dependency", "source": dep, "target": node["id"]} for node in nodes for dep in node.get("dependencies", ()))
    body = {"schema_version": "dclab_core.task_graph.v1", "graph_id": "binary_" + metadata_fingerprint(pid)[:16],
            "revision": revision, "objective": nodes[0]["objective"], "source_refs": refs, "solution_fingerprint": s_fp,
            "workflow_fingerprint": w_fp, "nodes": nodes, "edges": edges,
            "provenance": {"method": "deterministic", "reference_id": "binary_workflow_adapter", "producer_version": VERSION}}
    body["fingerprint"] = graph_fingerprint(body)
    graph = TaskGraph.read(body)
    report["topological_order"] = list(ORDER)
    report["blockers"] = sorted(set(report["blockers"]))
    report["validation_issues"] = validate_graph(graph, wf, known_capabilities=caps, external_artifacts=report["external_artifacts"])
    report["status"] = "blocked" if report["blockers"] or report["validation_issues"] else "mapped"
    return graph, report


__all__ = ["adapt_workflow", "capability_snapshot"]
