"""TP-03: bounded plans for existing atomic operations. No tool calls, storage or dispatcher.

The host supplies current metadata and independently verified observations. ``ready`` is a plan shape, never
permission or completion. TP-04 must refresh these inputs, call resolve_action, then use the original registry
and engine guards at the effect boundary. Model planning is explicitly off by default.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from contextlib import ExitStack
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dclab_core.task_execution_plan import PlannedAction, TaskBudget, TaskExecutionPlan, TaskStopPolicy
from dclab_core.task_fingerprints import metadata_fingerprint, plan_fingerprint
from dclab_core.task_graph import Digest, TaskGraph
from dclab_core.task_validation import ready_tasks, validate_execution_plan, validate_graph
from dclab_rnd.tasks.adapter import adapt_workflow
from dclab_rnd.tasks.decomposer import DecompositionBudget, _deadline, _unique_json

VERSION = "tp03.v1"
FAILURES = ("transient", "invalid_arguments", "changed_input", "missing_capability", "policy_refusal",
            "missing_approval", "ambiguous_effect", "domain_failure")


class PlanningBudget(DecompositionBudget):
    """Separate model-planning allowance; remaining_budget bounds the future task's execution.

    Reuse TP-02's one-request byte/token/cost/deadline bounds. No repair call or automatic retry.
    """


class _Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["tp03.v1"]
    baseline_fingerprint: Digest
    actions: tuple[PlannedAction, ...] = Field(min_length=1, max_length=8)
    budget: TaskBudget
    policy: TaskStopPolicy


class _Refused(ValueError):
    pass


def _policy_fingerprint():
    from dclab_rnd import prompts

    # The model's output schema and prompt are inputs too; provenance is excluded by plan_fingerprint.
    return metadata_fingerprint({"version": VERSION, "prompt": prompts.text("task_planner"),
                                 "model_output_schema": _Proposal.model_json_schema()})


def _read(cls, value):
    return cls.read(value.model_dump(mode="json") if hasattr(value, "model_dump") else value)


def _prepare(workflow, source_plan, project, graph, observed_state, capabilities):
    graph = _read(TaskGraph, graph)
    fresh, mapping = adapt_workflow(workflow, source_plan, project, capabilities=capabilities, revision=graph.revision)
    if fresh is None or mapping["status"] != "mapped":
        raise _Refused("source_or_capability_blocked")
    # Admit a TP-02 hierarchy, while preserving every physical operation, source decision and safety check.
    sources = lambda g: sorted((ref.kind, ref.id, ref.fingerprint) for ref in g.source_refs)
    if (sources(graph) != sources(fresh) or graph.solution_fingerprint != fresh.solution_fingerprint
            or graph.workflow_fingerprint != fresh.workflow_fingerprint or graph.graph_id != fresh.graph_id
            or graph.objective != fresh.objective):
        raise _Refused("changed_source_or_intent")

    def atomic(g):
        result = {}
        for n in g.nodes:
            if n.kind == "group":
                continue
            body = n.model_dump(mode="json", exclude={"parent_id", "dependencies"})
            for key in ("source_step_ids", "needs", "produces"):
                body[key].sort()
            body["acceptance_checks"] = sorted(
                ({**c, "evidence_refs": sorted(c["evidence_refs"])} for c in body["acceptance_checks"]), key=lambda c: c["id"])
            result[n.id] = body
        return result

    if atomic(graph) != atomic(fresh):
        raise _Refused("changed_task_authority")
    state = copy.deepcopy(observed_state)
    artifacts = state["artifacts"]
    allowed = set(mapping["external_artifacts"]) | {"planner_policy"}
    allowed.update(a for node in graph.nodes for a in (*node.needs, *node.produces))
    constants = {"project_context": project["id"], **{"stage_" + stage: stage for stage in
                 ("data", "leakage", "features", "models", "final")}}
    for ident, meta in artifacts.items():
        if ident not in allowed:
            raise _Refused("unrelated_artifact")
        if "value" in meta and (ident not in constants or meta["value"] != constants[ident]):
            raise _Refused("forbidden_artifact_value")
        # Metadata only, never paths, previews, rows or opaque result payloads.
        if set(meta) - {"fingerprint", "value_type", "producer_task", "value"}:
            raise _Refused("forbidden_artifact_metadata")
    for ident, expected in mapping["external_artifacts"].items():
        meta = artifacts.get(ident)
        if meta is None or any(meta.get(k) != v for k, v in expected.items()):
            raise _Refused("missing_or_changed_input")
        if ident in constants:
            meta["value"] = constants[ident]
    policy = {"fingerprint": _policy_fingerprint(), "value_type": "artifact"}
    if "planner_policy" in artifacts and artifacts["planner_policy"] != policy:
        raise _Refused("stale_planner_policy")
    artifacts["planner_policy"] = policy
    state["capabilities"] = copy.deepcopy(capabilities)
    if state.get("project_id") != project["id"]:
        raise _Refused("wrong_project")
    if validate_graph(graph, workflow, known_capabilities=capabilities, external_artifacts=artifacts):
        raise _Refused("invalid_graph")
    return graph, state, mapping


def _budget(value):
    return TaskBudget.model_validate(value.model_dump() if isinstance(value, TaskBudget) else value)


def _within(budget, remaining):
    return all(getattr(budget, k) <= getattr(remaining, k) for k in TaskBudget.model_fields)


def _seal(body):
    body["fingerprint"] = plan_fingerprint(body)
    return TaskExecutionPlan.read(body)


def _baseline(graph, node, state, mapping, remaining, revision, status="ready", reason=None):
    op = next((op for op in mapping["operations"] if op["task_id"] == node.id), None)
    refs = list(dict.fromkeys([*node.needs, *(op["argument_refs"].values() if op else ()), "planner_policy"]))
    body = {"schema_version": "dclab_core.task_execution_plan.v1", "graph_id": graph.graph_id,
            "graph_revision": graph.revision, "graph_fingerprint": graph.fingerprint, "task_id": node.id,
            "plan_id": node.id + "_plan", "revision": revision,
            "expected_inputs": [{"id": a, "fingerprint": state["artifacts"][a]["fingerprint"],
                                 "value_type": state["artifacts"][a]["value_type"]} for a in refs if a in state["artifacts"]],
            "acceptance_checks": [c.model_dump(mode="json") for c in node.acceptance_checks],
            "budget": {"max_tool_calls": 1 if status == "ready" else 0,
                       "max_seconds": min(remaining.max_seconds, 120) if status == "ready" else 0,
                       "max_cost_eur": remaining.max_cost_eur if status == "ready" else 0.0},
            "policy": {"on_failure": "escalate", "on_unknown": "block", "max_retries": 0, "compensation": "escalate"},
            "provenance": {"method": "deterministic", "reference_id": "task_execution_planner", "producer_version": VERSION},
            "status": status, "reason": reason}
    if status == "ready":
        body.update(selected_capabilities=[node.capability.model_dump(mode="json")],
                    actions=[{"id": "call", "capability_name": op["capability"], "scope": node.scope,
                              "arguments": [{"name": name, "value": {"kind": "input", "ref_id": ref,
                                            "value_type": state["artifacts"][ref]["value_type"]}}
                                            for name, ref in op["argument_refs"].items()]}],
                    declared_outputs=[{"id": a, "action_id": "call", "value_type": "artifact"} for a in node.produces])
    return _seal(body)


def _arguments(plan, action, state):
    """Resolve only allowlisted fingerprinted metadata. No expressions, files, tables or tool previews."""
    result = {}
    for argument in action.arguments:
        ref = argument.value
        if ref.kind != "input":
            raise _Refused("unresolved_output_binding")
        meta = state["artifacts"].get(ref.ref_id, {})
        if "value" not in meta or metadata_fingerprint(meta["value"]) != meta.get("fingerprint"):
            raise _Refused("unresolved_static_argument")
        result[argument.name] = meta["value"]
    return result


def _schema_check(cap, arguments):
    # Use the exact registry checker without calling any guard or handler. No schema references can resolve
    # remotely: the source adapter rejected them before this point.
    from dclab_rnd.agents.registry import Registry, Tool

    tool = Tool(cap["name"], "metadata-only schema check", cap["parameters"], handler=None,
                scope=cap["scope"], effect=cap["effect"], move=cap["move"])
    if Registry().check(tool, arguments):
        raise _Refused("invalid_tool_arguments")


def _check(plan, graph, workflow, state, caps, remaining):
    if validate_execution_plan(plan, graph, caps, source_workflow=workflow, observed_state=state):
        raise _Refused("plan_validation_failed")
    if not _within(plan.budget, remaining):
        raise _Refused("execution_budget_exceeded")
    if plan.status != "ready":
        return
    if plan.budget.max_seconds <= 0:
        raise _Refused("execution_budget_exhausted")
    # This adapter's leaves are atomic. Never turn one stage into run_all, extra writes or a preview execution.
    if len(plan.actions) != 1 or plan.actions[0].id != "call":
        raise _Refused("atomic_action_required")
    if "planner_policy" not in {i.id for i in plan.expected_inputs}:
        raise _Refused("missing_planner_policy")
    _schema_check(caps[plan.actions[0].capability_name], _arguments(plan, plan.actions[0], state))


def check_plan(plan, workflow, source_plan, project_metadata, *, graph, observed_state, capabilities, remaining_budget):
    """Pure current-source/permission/argument/budget gate; returns code-owned issues, never provider/private text."""
    try:
        graph, state, _ = _prepare(workflow, source_plan, project_metadata, graph, observed_state, capabilities)
        _check(_read(TaskExecutionPlan, plan), graph, workflow, state, capabilities, _budget(remaining_budget))
        return []
    except _Refused as error:
        return [str(error)]
    except Exception:
        return ["invalid_planning_input"]


def resolve_action(plan, action_id, workflow, source_plan, project_metadata, *, graph, observed_state, capabilities, remaining_budget):
    """Refresh and validate immediately before a FUTURE guarded call. Returns arguments only; never executes.

    No authorization token is returned. The dispatcher must still call Registry.call/acall with the current
    principal, scope and context; that guard and the engine own locking, approval consumption and holdout use.
    """
    graph, state, _ = _prepare(workflow, source_plan, project_metadata, graph, observed_state, capabilities)
    plan = _read(TaskExecutionPlan, plan)
    _check(plan, graph, workflow, state, capabilities, _budget(remaining_budget))
    if plan.status != "ready" or action_id != "call":
        raise _Refused("no_executable_action")
    return _arguments(plan, plan.actions[0], state)


def plan_task(workflow, source_plan, project_metadata, *, graph, task_id, observed_state, capabilities,
              remaining_budget, enabled=False, gateway=None, budget=None, past_failures=(), revision=1):
    """Return (typed plan or None, report). Gate/verification is a no-op, never claimed completed.

    remaining_budget is the host's future execution allowance. budget separately caps this planning request;
    the gateway independently enforces its workspace/run spend. Prior failed/unknown effects require their
    existing owner to reconcile before planning another attempt. No task writes or read-tool calls occur.
    """
    from dclab_rnd import experience, prompts
    from dclab_rnd.agents import harness

    began = time.monotonic()
    baseline, client = None, None
    report = {"version": VERSION, "status": "blocked", "blockers": [], "refusals": [], "model_calls": 0,
              "model_request": None, "execution_authorized": False, "failure_classes": list(FAILURES),
              "recovery": "Stop; reconcile owner evidence before any retry. Escalate policy, approval or domain failures."}

    def finish(status, reason=None, plan=None):
        report.update(status=status, seconds=round(time.monotonic() - began, 4))
        if reason:
            report["blockers" if status == "blocked" else "refusals"].append(reason)
        if client is not None and report["model_request"]:
            client.output(status == "proposed", reason or "task plan passed deterministic validation")
        harness.role_step("task_planner", report["seconds"], "ok" if status in ("deterministic", "proposed", "noop") else status,
                          model_request=report["model_request"])
        return plan if plan is not None else baseline, report

    try:
        graph, state, mapping = _prepare(workflow, source_plan, project_metadata, graph, observed_state, capabilities)
        remaining = _budget(remaining_budget)
        bounds = PlanningBudget.model_validate(budget.model_dump() if isinstance(budget, PlanningBudget) else budget or {})
        node = next((n for n in graph.nodes if n.id == task_id), None)
        if node is None:
            return finish("blocked", "unknown_task")
        status, reason = "ready", None
        if node.kind in ("gate", "verification"):
            status, reason = "noop", "Owner approval or independent evidence is required; no task completion is asserted."
        elif node.kind != "action":
            status, reason = "blocked", "non_action_task"
        elif past_failures:
            status, reason = "blocked", "reconciliation_required" if all(f in FAILURES for f in past_failures) else "invalid_failure_metadata"
        elif remaining.max_tool_calls < 1 or remaining.max_seconds <= 0:
            status, reason = "blocked", "execution_budget_exhausted"
        else:
            try:
                if task_id not in ready_tasks(graph, state, source_workflow=workflow):
                    status, reason = "blocked", "task_not_ready"
            except ValueError:
                status, reason = "blocked", "stale_or_unverified_state"
        baseline = _baseline(graph, node, state, mapping, remaining, revision, status, reason)
        if status != "ready":
            return finish(status, reason if status == "blocked" else None)
        _check(baseline, graph, workflow, state, capabilities, remaining)
        if not enabled:
            return finish("deterministic")
        if bounds.cap_eur == 0:
            return finish("fallback", "planning_budget_exhausted")
        messages = [{"role": "system", "content": prompts.text("task_planner")},
                    {"role": "user", "content": json.dumps({"baseline": baseline.model_dump(mode="json"),
                     "allowed_capabilities": {node.capability.name: capabilities[node.capability.name]},
                     "remaining_execution_budget": remaining.model_dump(), "past_failure_codes": list(past_failures)},
                     ensure_ascii=False, allow_nan=False)}]
        if len(json.dumps(messages, ensure_ascii=False).encode()) > bounds.max_input_bytes:
            return finish("fallback", "input_budget")
        with ExitStack() as stack:
            seconds = bounds.max_seconds - (time.monotonic() - began)
            if seconds <= 0:
                return finish("fallback", "planning_timeout")
            stack.enter_context(_deadline(seconds))
            if gateway is None:
                from dclab_rnd.models import installed
                gateway = installed()
            client = gateway.client("task_planner", project_id=project_metadata["id"])
            from dclab_rnd.models.gateway import Bound
            if not isinstance(client, Bound):
                return finish("fallback", "bounded_model_unavailable")
            client.run_limit_eur = bounds.cap_eur
            before = experience.last_model_request()
            try:
                report["model_calls"] = 1
                reply = client.complete(messages, None, max_tokens=bounds.max_tokens, single_attempt=True,
                                        timeout_seconds=max(0.001, bounds.max_seconds - (time.monotonic() - began)),
                                        require_price=True, max_total_tokens=bounds.max_total_tokens)
            finally:
                request = experience.last_model_request()
                report["model_request"] = request if request != before else None
            content = reply.get("content")
            if (reply.get("tool_calls") or reply.get("finish_reason") in ("length", "max_tokens", "tool_calls", "refusal")
                    or not isinstance(content, str) or len(content.encode()) > bounds.max_reply_bytes):
                return finish("fallback", "invalid_model_reply")
            proposal = _Proposal.model_validate(json.loads(content, object_pairs_hook=_unique_json))
            if proposal.baseline_fingerprint != baseline.fingerprint:
                return finish("fallback", "stale_model_context")
            body = baseline.model_dump(mode="json")
            body.update(actions=[a.model_dump(mode="json") for a in proposal.actions], budget=proposal.budget.model_dump(),
                        policy=proposal.policy.model_dump(), provenance={"method": "model", "producer_version": VERSION,
                        "reference_id": report["model_request"] or "proposal_" + metadata_fingerprint(content)[:24]})
            candidate = _seal(body)
            _check(candidate, graph, workflow, state, capabilities, remaining)
            if time.monotonic() - began > bounds.max_seconds:
                return finish("fallback", "planning_timeout")
            return finish("proposed", plan=candidate)
    except _Refused as error:
        return finish("fallback" if baseline is not None else "blocked", str(error))
    except TimeoutError:
        return finish("fallback", "planning_timeout")
    except Exception:
        # Pydantic/provider exceptions may echo private inputs. Only these fixed codes reach reports/traces.
        return finish("fallback" if baseline is not None else "blocked", "proposal_refused" if baseline is not None else "invalid_planning_input")


__all__ = ["PlanningBudget", "plan_task", "check_plan", "resolve_action"]
