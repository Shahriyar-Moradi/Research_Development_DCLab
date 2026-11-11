"""TP-02: one bounded model proposal over TP-01's immutable physical operations.

Only group membership and dependencies are model-controlled. The host copies every atomic task from a freshly
checked adapter result, including its gates, checks, blockers and source bindings. This API has no execution
context, tool registry, project store, scheduler or rollout caller. Fallback returns the supplied graph unchanged.
"""
from __future__ import annotations

import json
import hashlib
import time
import signal
import threading
from contextlib import ExitStack, contextmanager
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dclab_core.task_fingerprints import graph_fingerprint, metadata_fingerprint
from dclab_core.task_graph import Digest, Identifier, TaskGraph
from dclab_core.task_validation import validate_graph
from dclab_rnd.tasks.adapter import adapt_workflow

VERSION = "tp02.v1"


class _Refusal(ValueError):
    """Only code-owned refusal codes can reach task reports or traces."""


@contextmanager
def _deadline(seconds):
    """Interrupt synchronous proposal work, or refuse it. Never leave an uncancellable background request.

    This developer preview supports POSIX main-thread callers with no active real-time alarm. Product worker
    threads and platforms without setitimer keep the deterministic path until a cancellable host owns the call.
    Existing caller signal handlers and timers are not borrowed or overwritten while active.
    """
    if (threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer")
            or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0)):
        raise _Refusal("deadline_unavailable")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        from dclab_rnd.models.gateway import defer_request_deadline

        if not defer_request_deadline():
            raise TimeoutError("planning deadline reached")

    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class DecompositionBudget(BaseModel):
    """Host bounds, never values chosen by the model. No repair or retry in this package."""
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    max_nodes: int = Field(default=32, ge=8, le=64)
    max_depth: int = Field(default=4, ge=2, le=6)
    max_input_bytes: int = Field(default=48_000, ge=1000, le=96_000)
    max_reply_bytes: int = Field(default=16_000, ge=100, le=32_000)
    max_tokens: int = Field(default=1800, ge=100, le=4000)
    max_total_tokens: int = Field(default=9800, ge=100, le=12_000)
    max_seconds: float = Field(default=30.0, gt=0, le=90)
    cap_eur: float = Field(default=0.2, ge=0, le=1)
    model_calls: Literal[1] = 1


GroupID = Annotated[str, Field(pattern=r"^group_(?:[0-9]|[12][0-9]|3[01])$")]


class _Group(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: GroupID
    parent_id: GroupID | None
    members: list[Identifier] = Field(max_length=64)


class _Dependency(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: Identifier
    requires: list[Identifier] = Field(max_length=64)


class _Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["tp02.v1"]
    baseline_fingerprint: Digest
    groups: list[_Group] = Field(min_length=1, max_length=32)
    dependencies: list[_Dependency] = Field(min_length=1, max_length=64)


def _closure(links):
    """Bounded metadata-only closure; missing references and cycles are explicit refusals."""
    found = {}

    def visit(key, path):
        if key not in links or key in path:
            raise _Refusal("cycle_or_unknown_reference")
        if key not in found:
            result = set()
            for dep in links[key]:
                result.add(dep)
                result.update(visit(dep, path | {key}))
            found[key] = result
        return found[key]

    for key in links:
        visit(key, set())
    return found


def _compile(proposal, baseline, workflow, capabilities, external, budget, reference_id):
    """Compile a narrow boundary proposal into the existing TaskGraph contract; no model prose is admitted."""
    p = _Proposal.model_validate(proposal)
    if p.baseline_fingerprint != baseline.fingerprint:
        raise _Refusal("stale_model_context")
    leaves = {n.id: n for n in baseline.nodes if n.kind != "group"}
    groups = {g.id: g for g in p.groups}
    members = [m for g in p.groups for m in g.members]
    deps = {d.task_id: set(d.requires) for d in p.dependencies}
    if (len(groups) != len(p.groups) or set(groups) & (set(leaves) | {"root"})
            or len(members) != len(set(members)) or set(members) != set(leaves)
            or len(deps) != len(p.dependencies) or set(deps) != set(leaves)
            or any(len(d.requires) != len(set(d.requires)) for d in p.dependencies)):
        raise _Refusal("task_membership_or_dependency_mismatch")
    parents = {g.id: {g.parent_id} if g.parent_id else set() for g in p.groups}
    ancestors = _closure(parents)
    if any(len(a) + 3 > budget.max_depth for a in ancestors.values()):
        raise _Refusal("hierarchy_depth_budget")
    if 1 + len(groups) + len(leaves) > budget.max_nodes:
        raise _Refusal("node_budget")
    closure = _closure(deps)
    for ident, leaf in leaves.items():
        if not set(leaf.dependencies) <= closure[ident]:
            raise _Refusal("mandatory_dependency_removed")
    root = next(n for n in baseline.nodes if n.id == "root" and n.kind == "group")
    nodes = [root.model_dump(mode="json")]
    for group in sorted(p.groups, key=lambda g: g.id):
        children = sorted(m for g in p.groups if g.id == group.id or group.id in ancestors[g.id] for m in g.members)
        if not children:
            raise _Refusal("empty_or_ungrounded_group")
        # Derived entirely from immutable source tasks. The model cannot invent objectives or claim new work.
        title = "Coordinate " + ", ".join(children)
        nodes.append({"id": group.id, "kind": "group", "parent_id": group.parent_id or "root",
                      "title": title[:200], "objective": title[:1000]})
    for leaf in leaves.values():
        body = leaf.model_dump(mode="json")
        body.update(parent_id=next(g.id for g in p.groups if leaf.id in g.members), dependencies=sorted(deps[leaf.id]))
        nodes.append(body)
    edges = [{"kind": "grouping", "source": n["parent_id"], "target": n["id"]} for n in nodes if n.get("parent_id")]
    edges.extend({"kind": "dependency", "source": dep, "target": n["id"]} for n in nodes for dep in n.get("dependencies", []))
    body = baseline.model_dump(mode="json")
    body.update(nodes=nodes, edges=edges, revision=baseline.revision + 1,
                provenance={"method": "model", "reference_id": reference_id, "producer_version": VERSION})
    body["fingerprint"] = graph_fingerprint(body)
    candidate = TaskGraph.read(body)
    if validate_graph(candidate, workflow, known_capabilities=capabilities, external_artifacts=external):
        raise _Refusal("tp01_validation_failed")
    return candidate


def _data_metadata(project):
    """A data-card-shaped allowlist built from column metadata, never profiles, previews, labels or values."""
    solution = project["solution"]
    forbidden, identifiers = set(solution.get("forbidden", [])), set(solution.get("identifiers", []))
    return {"family": "classification", "labels": solution.get("labels"),
            "columns": [{"name": name, "role": "outcome" if name == solution["target"] else
                         "identifier" if name in identifiers else "forbidden" if name in forbidden else "candidate"}
                        for name in project["data"]["columns"]]}


def _unique_json(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise _Refusal("duplicate_json_key")
        out[key] = value
    return out


def decompose(workflow, source_plan, project_metadata, *, deterministic_graph, capabilities, confirmed_goal,
              enabled=False, gateway=None, budget=None, unresolved_questions=()):
    """Return (TaskGraph or None, report). Explicit opt-in; a current adapter baseline and confirmed goal are required.

    Recompiling metadata detects changed objectives, data, solution signoff, capability schemas and settings before
    any request. A stale or blocked baseline is retained for inspection with status blocked; it is never authorized.
    Model replies may organize existing physical tasks, but cannot invent stage work or split atomic operations.
    """
    from dclab_rnd import experience, prompts
    from dclab_rnd.agents import harness

    started = time.monotonic()
    budget = DecompositionBudget.model_validate(budget.model_dump() if isinstance(budget, DecompositionBudget) else {} if budget is None else budget)
    report = {"version": VERSION, "status": "fallback", "refusals": [], "blockers": [], "model_calls": 0,
              "execution_authorized": False, "prompt_version": VERSION, "model_request": None}
    fallback = deterministic_graph
    phase, client = "source", None
    deadline = ExitStack()

    def finish(status, *reasons, graph=fallback):
        report.update(status=status, refusals=list(reasons), seconds=round(time.monotonic() - started, 4))
        verdict = "ok" if status == "proposed" else status + ":" + ",".join(reasons)
        if client is not None and report["model_request"]:
            client.output(status == "proposed", ",".join(reasons) or "task boundaries passed TP-01 validation")
        harness.role_step("task_decomposer", report["seconds"], verdict,
                          model_request=report["model_request"])
        return graph, report

    try:
        baseline = TaskGraph.read(fallback.model_dump(mode="json") if isinstance(fallback, TaskGraph) else fallback) if fallback is not None else None
        fresh, mapping = adapt_workflow(workflow, source_plan, project_metadata, capabilities=capabilities,
                                        revision=baseline.revision if baseline is not None else 1)
        report["blockers"] = mapping["blockers"]
        if fresh is None or fallback is None:
            return finish("blocked", "missing_adapter_graph")
        if fresh != baseline:
            return finish("blocked", "stale_deterministic_graph")
        if mapping["status"] != "mapped":
            return finish("blocked", "adapter_blocked")
        if not isinstance(confirmed_goal, str) or not confirmed_goal.strip() or confirmed_goal != project_metadata.get("goal"):
            return finish("blocked", "objective_not_confirmed")
        if not enabled:
            return finish("fallback", "model_disabled")
        # Open questions are blockers, never instructions or permission for a model to guess an answer.
        if unresolved_questions:
            return finish("blocked", "unresolved_questions")
        phase = "proposal"
        evidence = {"confirmed_goal": confirmed_goal, "data_card_metadata": _data_metadata(project_metadata),
                    "validated_workflow": json.loads(workflow.model_dump_json()) if hasattr(workflow, "model_dump_json") else workflow,
                    "task_capability_map": mapping["operations"], "deterministic_graph": baseline.model_dump(mode="json"),
                    "authorized_tool_names": sorted(name for name, cap in capabilities.items() if cap.get("allowed") is True),
                    "budgets": budget.model_dump(), "unresolved_questions": []}
        system = prompts.text("task_decomposer")
        report["prompt_hash"] = hashlib.sha256(system.encode()).hexdigest()[:16]
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, allow_nan=False)}]
        if len(json.dumps(messages, ensure_ascii=False).encode()) > budget.max_input_bytes:
            return finish("fallback", "input_budget")
        if budget.cap_eur == 0 or time.monotonic() - started >= budget.max_seconds:
            return finish("fallback", "planning_budget")
        deadline.enter_context(_deadline(budget.max_seconds - (time.monotonic() - started)))
        if gateway is None:
            from dclab_rnd.models import installed

            gateway = installed()
        client = gateway.client("task_decomposer", project_id=project_metadata["id"])
        if client is None:
            return finish("fallback", "model_unavailable")
        from dclab_rnd.models.gateway import Bound

        if not isinstance(client, Bound):
            return finish("fallback", "bounded_client_unavailable")
        client.run_limit_eur = budget.cap_eur
        remaining = budget.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            return finish("fallback", "planning_budget")
        before = experience.last_model_request()
        try:
            report["model_calls"] = 1
            reply = client.complete(messages, None, max_tokens=budget.max_tokens, single_attempt=True,
                                    timeout_seconds=remaining, require_price=True, max_total_tokens=budget.max_total_tokens)
        finally:
            made = experience.last_model_request()
            report["model_request"] = made if made != before else None
        if time.monotonic() - started > budget.max_seconds:
            return finish("fallback", "planning_timeout")
        content = reply.get("content")
        if (reply.get("tool_calls") or reply.get("finish_reason") in ("length", "max_tokens", "tool_calls", "refusal")
                or not isinstance(content, str) or len(content.encode()) > budget.max_reply_bytes):
            return finish("fallback", "invalid_model_reply")
        proposal = json.loads(content, object_pairs_hook=_unique_json)
        # A private source reference to the gateway exchange, never thinking in a task graph or trace.
        reference_id = report["model_request"] or "proposal_" + metadata_fingerprint(proposal)[:24]
        candidate = _compile(proposal, baseline, workflow, capabilities, mapping["external_artifacts"], budget, reference_id)
        if time.monotonic() - started > budget.max_seconds:
            return finish("fallback", "planning_timeout")
        return finish("proposed", graph=candidate)
    except _Refusal as error:
        return finish("fallback", str(error))
    except (ValidationError, json.JSONDecodeError):
        return finish("blocked" if phase == "source" else "fallback", "invalid_source_shape" if phase == "source" else "invalid_proposal_shape")
    except RuntimeError as error:
        reason = "planning_timeout" if time.monotonic() - started >= budget.max_seconds else (
            "model_budget_refused" if str(error).startswith("BudgetExceeded:") else "model_request_failed")
        return finish("fallback", reason)
    except TimeoutError:
        return finish("fallback", "planning_timeout")
    except Exception:  # a failed provider or malformed source cannot take away the deterministic fallback
        # Never echo model text, source values, provider messages or pydantic's input values into task traces.
        return finish("blocked" if phase == "source" else "fallback", "source_refused" if phase == "source" else "proposal_refused")
    finally:
        deadline.close()


__all__ = ["DecompositionBudget", "decompose"]
