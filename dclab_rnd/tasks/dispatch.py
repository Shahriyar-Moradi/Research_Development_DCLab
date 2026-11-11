"""TP-04: one guarded atomic effect, checkpointed in the solution team's existing saver.

Only owner-operation completion is reconciled here. Scientific acceptance remains pending for TP-05. No tool
reply, checkpoint flag or agent claim is accepted in place of current owner evidence. Unknown effects never retry.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import secrets
import time

from dclab_core.card import problem_card
from dclab_core.lineage import RunRecord
from dclab_core.solution_workflow import solution_workflow
from dclab_core.task_execution_plan import TaskBudget
from dclab_core.task_fingerprints import metadata_fingerprint
from dclab_core.task_graph import TaskGraph
from dclab_rnd.core_adapter import brief_from_project, plan_from_project
from dclab_rnd.tasks.adapter import ORDER, OUTPUTS, adapt_workflow, capability_snapshot
from dclab_rnd.tasks.planner import plan_task, resolve_action

VERSION = "tp04.v1"
NAMESPACE = "task_execution"
STAGES = ("data", "leakage", "features", "models", "final")
TERMINAL = ("completed", "reconciled")


class Blocked(ValueError):
    """Only code-owned reasons may enter task records or traces."""


def _code():
    root = Path(__file__).resolve().parents[1]
    names = ("tasks/dispatch.py", "tasks/planner.py", "studio/engine.py", "studio/graph.py", "studio/lineage.py",
             "intern/tools.py", "agents/registry.py", "tasks/adapter.py", "orchestra/solution.py",
             "studio/store.py", "storage/postgres.py", "storage/files.py")
    return metadata_fingerprint({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names})


def _permission(entry):
    return metadata_fingerprint({k: v for k, v in entry.items() if k != "used"})


class TaskDispatcher:
    def __init__(self, projects, checkpointer, *, enabled=False, registry=None, allowed_tools=None, budget=None):
        from dclab_rnd.agents import build_registry

        self.projects, self.saver, self.enabled = projects, checkpointer, enabled
        self.registry = registry if registry is not None else build_registry(("project",))
        self.allowed = set(allowed_tools if allowed_tools is not None else ("run_stage", "export_notebook"))
        self.budget = TaskBudget.model_validate(budget or {"max_tool_calls": 6, "max_seconds": 3600, "max_cost_eur": 0.0})

    @staticmethod
    def config(project_id):
        return {"configurable": {"thread_id": "project:" + project_id, "checkpoint_ns": NAMESPACE}}

    def state(self, project_id):
        self.projects.get(project_id)  # shared PostgreSQL saver tables are not an authorization boundary
        checkpoint = self.saver.get(self.config(project_id))
        if checkpoint is None:
            return None
        state = copy.deepcopy(checkpoint.get("channel_values", {}).get("task_state"))
        if (not isinstance(state, dict) or not isinstance(state.get("attempts"), list)
                or type(state.get("tool_calls")) is not int or state["tool_calls"] != len(state["attempts"])
                or type(state.get("seconds")) not in (int, float) or not math.isfinite(state["seconds"]) or state["seconds"] < 0):
            raise Blocked("invalid_task_checkpoint")
        tasks, ids = set(), set()
        for attempt in state["attempts"]:
            if (not isinstance(attempt, dict) or attempt.get("task") not in (*STAGES, "report")
                    or attempt.get("status") not in (*TERMINAL, "started", "unknown", "blocked")
                    or not isinstance(attempt.get("id"), str) or attempt["id"] in ids or attempt["task"] in tasks):
                raise Blocked("invalid_task_checkpoint")
            tasks.add(attempt["task"])
            ids.add(attempt["id"])
        TaskBudget.model_validate(state["budget"])
        return state

    def _save(self, pid, state, kind, task=None):
        from langgraph.checkpoint.base import empty_checkpoint
        from dclab_rnd.studio.store import now

        state["revision"] = state.get("revision", 0) + 1
        state.setdefault("transitions", []).append({"kind": kind, "task": task, "at": now(), "revision": state["revision"]})
        checkpoint = empty_checkpoint()
        state["checkpoint"] = checkpoint["id"]
        for attempt in state.get("attempts", []):
            if attempt["task"] == task:
                attempt["checkpoint"] = checkpoint["id"]
        checkpoint["channel_values"] = {"task_state": copy.deepcopy(state)}
        checkpoint["channel_versions"] = {"task_state": state["revision"]}
        prior = self.saver.get_tuple(self.config(pid))
        config = self.saver.put(prior.config if prior else self.config(pid), checkpoint,
                                {"source": "update", "step": state["revision"], "parents": {}},
                                {"task_state": state["revision"]})
        return config["configurable"]["checkpoint_id"]

    def _initial(self, project, caps):
        if any(meta.get("status") != "pending" for meta in project["stages"].values()):
            raise Blocked("existing_work_needs_explicit_adoption")
        source = plan_from_project(project)
        workflow = solution_workflow(problem_card(brief_from_project(project)))
        graph, mapping = adapt_workflow(workflow, source, project, capabilities=caps)
        if graph is None or mapping["status"] != "mapped":
            raise Blocked("source_or_capability_blocked")
        return {"version": VERSION, "graph": graph.model_dump(mode="json"),
                "code": _code(), "attempts": [], "transitions": [],
                "budget": self.budget.model_dump(), "tool_calls": 0, "seconds": 0.0, "revision": 0}

    def _sources(self, project, state, caps):
        if state["version"] != VERSION or state["code"] != _code():
            raise Blocked("runtime_code_changed")
        graph = TaskGraph.read(state["graph"])
        # Rebuild from the authenticated project, then compare immutable source hashes. Source decisions may hold
        # private class values; their complete bodies do not belong in task checkpoints or traces.
        source = plan_from_project(project)
        workflow = solution_workflow(problem_card(brief_from_project(project)))
        fresh, mapping = adapt_workflow(workflow, source, project, capabilities=caps, revision=graph.revision)
        if fresh != graph or mapping["status"] != "mapped":
            raise Blocked("source_or_capability_changed")
        return graph, source, workflow, mapping

    def _stage_evidence(self, pid, task, attempt):
        from dclab_rnd.studio import lineage
        from dclab_rnd.expansion import datasets as ds

        project = self.projects.get(pid)
        record = self.projects.read_stage(pid, task)
        if not record or record.get("status") != "completed" or record.get("stage") != task:
            raise Blocked("owner_completion_unknown")
        run = RunRecord.read(record["run"])
        if (run.parent != attempt["id"] or run.project != pid or run.name != task
                or run.kind != "stage" or run.status != "completed"):
            raise Blocked("owner_run_mismatch")
        if run.seconds is None:
            raise Blocked("owner_duration_unknown")
        previous = {s: self.projects.read_stage(pid, s) for s in STAGES[:STAGES.index(task)]}
        expected = lineage.inputs_of(project, previous, seed=ds.RANDOM_STATE)
        if run.inputs.model_dump(exclude={"sampling"}) != expected.model_dump(exclude={"sampling"}):
            raise Blocked("owner_inputs_changed")
        meta = project["stages"][task]
        if meta.get("run") != run.run or meta.get("status") not in ("completed", "approved"):
            raise Blocked("owner_metadata_incomplete")
        if task == "final":
            from dclab_core.holdout import read_receipt
            receipt = read_receipt(self.projects.data_dir(pid) / "holdout_receipt.json")
            evidence = record.get("evidence", {})
            if (receipt is None or receipt.project_id != pid or receipt.stage != "final" or not receipt.consumed
                    or receipt.reuse_reason or receipt.holdout_uses != 1 or project.get("holdout_uses") != 1
                    or evidence.get("holdout_receipt_id") != receipt.receipt_id
                    or evidence.get("holdout_split_fingerprint") != receipt.split_fingerprint
                    or evidence.get("holdout_uses_in_this_project") != 1):
                raise Blocked("holdout_receipt_unreconciled")
        digest = metadata_fingerprint(record)
        if attempt.get("output_fingerprint") and attempt["output_fingerprint"] != digest:
            raise Blocked("owner_record_changed")
        return {"fingerprint": digest, "run": run.run, "reference": "owner_" + run.run, "seconds": run.seconds}

    def _export_evidence(self, pid, attempt):
        from dclab_rnd.storage.files import files_for

        rows = [r["payload"] for r in self.projects.activity(pid, limit=1000)
                if r["kind"] == "task_exported" and r["payload"].get("attempt") == attempt["id"]]
        if len(rows) != 1:
            raise Blocked("export_completion_unknown")
        evidence = rows[0]
        if evidence["source"] != metadata_fingerprint(self.projects.records(pid)):
            raise Blocked("export_inputs_changed")
        files = files_for(self.projects)
        if len(evidence["files"]) != 2:
            raise Blocked("export_incomplete")
        for artifact in evidence["files"]:
            if not files.record(artifact["key"]) or files.record(artifact["key"])["sha256"] != artifact["sha256"]:
                raise Blocked("export_artifact_changed")
            files.path(artifact["key"], expected=artifact["sha256"])
        digest = metadata_fingerprint(evidence)
        if attempt.get("output_fingerprint") and attempt["output_fingerprint"] != digest:
            raise Blocked("export_artifact_changed")
        return {"fingerprint": digest, "run": None, "reference": "export_" + attempt["id"]}

    def _evidence(self, pid, task, attempt):
        return self._export_evidence(pid, attempt) if task == "report" else self._stage_evidence(pid, task, attempt)

    def _observed(self, project, state, graph, mapping):
        artifacts = copy.deepcopy(mapping["external_artifacts"])
        ref = next(ref for ref in graph.source_refs if ref.kind == "approval")
        approval = {"approved": True, "reference_id": ref.id, "fingerprint": ref.fingerprint,
                    "project_id": project["id"], "solution_fingerprint": graph.solution_fingerprint,
                    "workflow_fingerprint": graph.workflow_fingerprint,
                    "sources": [r.model_dump(mode="json") for r in graph.source_refs if r.kind != "approval"],
                    "inputs": {a: artifacts[a]["fingerprint"] for a in ("table", "execution_context")}}
        observed = {"project_id": project["id"], "graph_id": graph.graph_id, "graph_revision": graph.revision,
                    "graph_fingerprint": graph.fingerprint, "solution_fingerprint": graph.solution_fingerprint,
                    "workflow_fingerprint": graph.workflow_fingerprint, "artifacts": artifacts,
                    "approval": approval, "tasks": {}, "holdout_uses": project.get("holdout_uses")}
        completed = {a["task"]: a for a in state["attempts"] if a["status"] in TERMINAL}
        for task in ORDER:
            node = next(n for n in graph.nodes if n.id == task)
            if task == "holdout":
                saved = state.get("holdout_permission")
                if not saved:
                    continue
                entries = [e for e in project.get("approvals", []) if e.get("gate") == "holdout" and _permission(e) == saved["fingerprint"]]
                if len(entries) != 1 or any(artifacts.get(a, {}).get("fingerprint") != fp for a, fp in saved["inputs"].items()):
                    raise Blocked("holdout_approval_changed")
                permission = {**approval, "reference_id": saved["id"], "fingerprint": saved["fingerprint"],
                              "inputs": {**approval["inputs"], **saved["inputs"]}, "used": bool(entries[0].get("used"))}
                observed["holdout_approval"] = permission
                fingerprint, reference = saved["fingerprint"], saved["id"]
            elif task in completed:
                evidence = self._evidence(project["id"], task, completed[task])
                fingerprint, reference = evidence["fingerprint"], evidence["reference"]
            else:
                continue
            artifacts[OUTPUTS[task]] = {"value_type": "artifact", "fingerprint": fingerprint, "producer_task": task}
            receipt = {"status": "verified", "verification_ref": reference, "project_id": project["id"],
                       "graph_revision": graph.revision, "graph_fingerprint": graph.fingerprint,
                       "inputs": {a: artifacts[a]["fingerprint"] for a in node.needs},
                       "outputs": {OUTPUTS[task]: fingerprint}}
            if task == "holdout":
                receipt["approval_ref"] = {"id": saved["id"], "fingerprint": fingerprint}
            observed["tasks"][task] = receipt
        return observed

    def _gate(self, project, state, graph, mapping):
        if state.get("holdout_permission"):
            return
        models = next((a for a in state["attempts"] if a["task"] == "models" and a["status"] in TERMINAL), None)
        if models is None:
            return
        entries = [e for e in project.get("approvals", []) if e.get("gate") == "holdout" and not e.get("used")
                   and _permission(e) not in models["prior_permissions"]]
        if len(entries) != 1 or project.get("holdout_uses") != 0:
            raise Blocked("fresh_holdout_approval_required")
        observed = self._observed(project, state, graph, mapping)
        node = next(n for n in graph.nodes if n.id == "holdout")
        digest = _permission(entries[0])
        state["holdout_permission"] = {"id": "approval_" + digest[:24], "fingerprint": digest,
                                       "inputs": {a: observed["artifacts"][a]["fingerprint"] for a in node.needs}}
        self._save(project["id"], state, "owner_approval_observed", "holdout")

    def _finish(self, pid, state, attempt, reconciled=False):
        from dclab_rnd import ledger
        from dclab_rnd.studio.store import now

        evidence = self._evidence(pid, attempt["task"], attempt)
        if reconciled:
            reserved = attempt["budget"]["reserved_seconds"]
            duration = evidence.get("seconds", reserved)
            charged = max(reserved, duration)
            state["seconds"] += charged - reserved
            state["budget_overrun"] = state.get("budget_overrun", False) or duration > reserved
            attempt["budget"].update(seconds=duration, recovered_charge_seconds=charged)
        book = ledger.current_ledger()
        ids, capture = [], "not_available"
        if book and evidence["run"]:
            try:
                ids = sorted({r.id for run in (evidence["run"], attempt["id"]) for r in book.records(project=pid, run=run)})
                capture = "linked"
            except Exception:  # ledger capture is best effort; it does not erase conclusive owner completion evidence
                capture = "unavailable"
        attempt.update(status="reconciled" if reconciled else "completed", finished=now(),
                       output_fingerprint=evidence["fingerprint"], owner_run=evidence["run"], evidence_ref=evidence["reference"],
                       decision_ids=ids, decision_capture=capture, acceptance_verification="pending_tp05")
        checkpoint = self._save(pid, state, attempt["status"], attempt["task"])
        return {"status": attempt["status"], "attempt_id": attempt["id"], "checkpoint": checkpoint,
                "owner_run": evidence["run"], "acceptance_verification": "pending_tp05"}

    def _invoke(self, pid, attempt, action, arguments):
        from dclab_rnd import experience
        from dclab_rnd.agents import harness
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.models.gateway import private_scope
        from dclab_rnd.studio.lineage import task_attempt

        clock = time.monotonic()
        with experience.bind(project=pid), harness.session("orchestra.solution", run_id=attempt["id"], params={"task": attempt["task"]}) as session:
            with task_attempt(attempt["id"]), private_scope(True):  # first execution slice is deterministic; no model spend
                try:
                    reply = self.registry.call(action.capability_name, arguments, context=Toolbox(self.projects, registry=self.registry), scope="project")
                except Exception:
                    reply = {"error": "tool_exception"}
            session.step(action.capability_name, arguments, {"status": "refused_or_unknown" if isinstance(reply, dict) and
                         (reply.get("error") or reply.get("blocked")) else "returned"}, time.monotonic() - clock)
        return reply

    def preflight(self, project_id):
        """Check current source and completed owner evidence without dispatching or adopting legacy results."""
        try:
            with self.projects.turn(project_id, require_process=True):
                project = self.projects.get(project_id)
                from dclab_rnd.studio import data
                data.data_path(self.projects, project)
                caps = capability_snapshot(self.registry, allowed_tools=self.allowed)
                state = self.state(project_id)
                if state is None:
                    self._initial(project, caps)
                else:
                    graph, _, _, mapping = self._sources(project, state, caps)
                    self._observed(project, state, graph, mapping)
                return None
        except Blocked as error:
            return str(error)
        except Exception:
            return "task_state_or_evidence_invalid"

    def dispatch(self, project_id, task_id, *, plan=None):
        from dclab_rnd import jobs
        from dclab_rnd.studio import data
        from dclab_rnd.studio.store import now

        if not self.enabled:
            return {"status": "disabled", "reason": "task_execution_disabled"}
        state, attempt = None, None
        with self.projects.turn(project_id, require_process=True):
            jobs.checkpoint()
            try:
                project = self.projects.get(project_id)
                caps = capability_snapshot(self.registry, allowed_tools=self.allowed)
                state = self.state(project_id)
                if state is None:
                    state = self._initial(project, caps)
                    self._save(project_id, state, "initialized")
                graph, source, workflow, mapping = self._sources(project, state, caps)
                data.data_path(self.projects, project)  # owner API checks actual bytes; no table rows are read here
                for pending in state["attempts"]:
                    if pending["status"] == "started":
                        attempt = pending
                        result = self._finish(project_id, state, pending, reconciled=True)
                        attempt = None
                        if pending["task"] == task_id:
                            return result
                if task_id not in (*STAGES, "report"):
                    raise Blocked("non_executable_task")
                previous = next((a for a in state["attempts"] if a["task"] == task_id), None)
                if previous:
                    if previous["status"] in TERMINAL:
                        self._evidence(project_id, task_id, previous)
                        return {"status": "already_completed", "attempt_id": previous["id"], "owner_run": previous.get("owner_run")}
                    if previous["status"] == "started":
                        attempt = previous
                        return self._finish(project_id, state, previous, reconciled=True)
                    raise Blocked("unknown_effect_requires_owner_reconciliation")
                if any(a["status"] not in TERMINAL for a in state["attempts"]):
                    raise Blocked("unresolved_previous_attempt")
                if task_id in ("final", "report"):
                    self._gate(project, state, graph, mapping)
                observed = self._observed(project, state, graph, mapping)
                limit = state["budget"]
                if state.get("budget_overrun"):
                    raise Blocked("execution_budget_exceeded")
                remaining = {"max_tool_calls": min(limit["max_tool_calls"], self.budget.max_tool_calls) - state["tool_calls"],
                             "max_seconds": max(0, int(min(limit["max_seconds"], self.budget.max_seconds) - state["seconds"])),
                             "max_cost_eur": 0.0}
                if remaining["max_tool_calls"] <= 0 or remaining["max_seconds"] <= 0:
                    raise Blocked("execution_budget_exhausted")
                if plan is None:
                    plan, report = plan_task(workflow, source, project, graph=graph, task_id=task_id, observed_state=observed,
                                             capabilities=caps, remaining_budget=remaining)
                    if plan is None or plan.status != "ready":
                        raise Blocked("task_not_ready")
                if plan.task_id != task_id:
                    raise Blocked("wrong_task_plan")
                args = resolve_action(plan, "call", workflow, source, project, graph=graph, observed_state=observed,
                                      capabilities=caps, remaining_budget=remaining)
                attempt = {"id": "task_" + secrets.token_hex(12), "task": task_id, "status": "started", "started": now(),
                           "graph_revision": graph.revision, "graph_fingerprint": graph.fingerprint,
                           "plan_revision": plan.revision, "plan_fingerprint": plan.fingerprint, "plan": plan.model_dump(mode="json"),
                           "inputs": {i.id: i.fingerprint for i in plan.expected_inputs}, "decision_ids": [],
                           "prior_permissions": [_permission(e) for e in project.get("approvals", []) if e.get("gate") == "holdout"],
                           "acceptance_verification": "pending_tp05", "budget": {"tool_calls": 1, "seconds": None, "cost_eur": 0.0}}
                state["attempts"].append(attempt)
                state["tool_calls"] += 1  # charged durably before effect; an ambiguous attempt cannot reclaim its allowance
                state["seconds"] += plan.budget.max_seconds
                attempt["budget"]["reserved_seconds"] = plan.budget.max_seconds
                self._save(project_id, state, "started", task_id)
                began = time.monotonic()
                reply = self._invoke(project_id, attempt, plan.actions[0], args)
                elapsed = time.monotonic() - began
                attempt["budget"]["seconds"] = elapsed
                state["seconds"] += elapsed - plan.budget.max_seconds
                state["budget_overrun"] = elapsed > plan.budget.max_seconds
                if isinstance(reply, dict) and (reply.get("blocked") or reply.get("status") in ("blocked", "needs_approval")):
                    attempt.update(status="blocked", reason="guard_refused")
                    self._save(project_id, state, "guard_refused", task_id)
                    return {"status": "blocked", "reason": "guard_refused", "attempt_id": attempt["id"]}
                result = self._finish(project_id, state, attempt)
                jobs.progress(task=task_id, attempt=attempt["id"], status=result["status"])
                jobs.checkpoint()
                return result
            except Blocked as error:
                reason = str(error)
            except Exception:
                reason = "task_state_or_evidence_invalid"
            if state is not None:
                if attempt is not None:
                    attempt.update(status="unknown", reason=reason)
                self._save(project_id, state, "blocked", task_id)
            return {"status": "blocked", "reason": reason, "attempt_id": attempt["id"] if attempt else None}


__all__ = ["TaskDispatcher"]
