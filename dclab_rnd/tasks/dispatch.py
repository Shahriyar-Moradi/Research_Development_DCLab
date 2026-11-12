"""Guarded task effects, independent measured verification and bounded recovery in the existing saver.

No reply, checkpoint flag or agent claim substitutes for owner evidence. Unknown effects never retry.
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
from dclab_core.task_execution_plan import TaskBudget, TaskExecutionPlan
from dclab_core.task_fingerprints import metadata_fingerprint
from dclab_core.task_graph import TaskGraph
from dclab_rnd.core_adapter import brief_from_project, plan_from_project
from dclab_rnd.tasks.adapter import ORDER, OUTPUTS, adapt_workflow, capability_snapshot
from dclab_rnd.tasks.planner import plan_task, resolve_action

VERSION = "tp05.v1"
NAMESPACE = "task_execution"
STAGES = ("data", "leakage", "features", "models", "final")
TERMINAL = ("completed", "reconciled")


class Blocked(ValueError):
    """Only code-owned reasons may enter task records or traces."""

    def __init__(self, reason, *, verification=None):
        super().__init__(reason)
        self.verification = verification


def _code():
    root = Path(__file__).resolve().parents[1]
    names = ("tasks/dispatch.py", "tasks/verifier.py", "tasks/planner.py", "studio/engine.py", "studio/graph.py", "studio/lineage.py",
             "intern/tools.py", "agents/registry.py", "tasks/adapter.py", "orchestra/solution.py",
             "studio/store.py", "storage/postgres.py", "storage/files.py", "expansion/runner.py", "expansion/datasets.py",
             "studio/solution_report.py")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    for name in ('task_verification.py', 'task_recovery.py', 'holdout.py', 'assess.py', 'report.py'):
        hashes['core/' + name] = hashlib.sha256((root.parent / 'dclab_core' / name).read_bytes()).hexdigest()
    return metadata_fingerprint(hashes)


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
        ids = set()
        for attempt in state["attempts"]:
            if (not isinstance(attempt, dict) or attempt.get("task") not in (*STAGES, "report")
                    or attempt.get("status") not in (*TERMINAL, "started", "unknown", "blocked", "no_effect")
                    or not isinstance(attempt.get("id"), str) or attempt["id"] in ids ):
                raise Blocked("invalid_task_checkpoint")
            ids.add(attempt["id"])
        if state.get('version') == VERSION:
            if any(not isinstance(state.get(key), list) for key in ('verifications', 'replans', 'graph_history', 'source_approvals')):
                raise Blocked('invalid_task_checkpoint')
            active = [a for a in state['attempts'] if not a.get('invalidated_by')]
            if any(sum(a['task'] == task for a in active) > 2 for task in (*STAGES, 'report')):
                raise Blocked('invalid_attempt_history')
            if len(state['replans']) > 2 or len(state['graph_history']) > 2:
                raise Blocked('invalid_recovery_history')
            for attempt in active:
                if attempt['status'] == 'no_effect' and attempt.get('boundary') != {
                        'seen': True, 'phase': 'validation', 'handler_entered': False}:
                    raise Blocked('invalid_no_effect_evidence')
        TaskBudget.model_validate(state["budget"])
        return state

    def _save(self, pid, state, kind, task=None):
        from langgraph.checkpoint.base import empty_checkpoint
        from dclab_rnd.studio.store import now

        state["revision"] = state.get("revision", 0) + 1
        state.setdefault("transitions", []).append({"kind": kind, "task": task, "at": now(), "revision": state["revision"]})
        checkpoint = empty_checkpoint()
        state["checkpoint"] = checkpoint["id"]
        for attempt in reversed(state.get("attempts", [])):
            if attempt["task"] == task:
                attempt["checkpoint"] = checkpoint["id"]
                break
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
                "code": _code(), "external_artifacts": mapping["external_artifacts"], "attempts": [], "transitions": [], "verifications": [], "replans": [], "graph_history": [], "source_approvals": [metadata_fingerprint(project.get("signoff"))], "source_identity": self.source_revision(project["id"]),
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
            decision = record.get('decision') or {}
            choice = decision.get('chosen')
            approved = (meta.get('status') == 'approved' and project.get('decisions', {}).get(task) == choice
                        and any(t.get('actor') == 'human' and t.get('status') == 'allowed'
                                and t.get('move') == 'approve_stage' and t.get('args', {}).get('stage') == task
                                and t.get('args', {}).get('choice') == choice for t in self.projects.transitions(pid)))
            scientific = metadata_fingerprint({k: v for k, v in record.items() if k not in ('decision', 'notes', 'explanation')})
            if not approved or scientific != attempt.get('scientific_fingerprint'):
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
        if len(evidence["files"]) != 3:
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

    def _observed(self, project, state, graph, mapping, *, until=None):
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
        completed = {a["task"]: a for a in state["attempts"] if a["status"] in TERMINAL and not a.get("invalidated_by")}
        for task in ORDER:
            if until is not None and ORDER.index(task) > ORDER.index(until):
                break
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
                attempt = completed[task]
                evidence = self._evidence(project["id"], task, attempt)
                verification = self._verify(project["id"], state, graph, attempt, artifacts, evidence)
                if verification["status"] != "verified":
                    raise Blocked("acceptance_" + verification["status"] + ":" + verification["reasons"][0], verification=verification)
                fingerprint, reference = evidence["fingerprint"], verification["id"]
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

    @staticmethod
    def _inputs(attempt, artifacts):
        from dclab_rnd.tasks.planner import _policy_fingerprint
        known = {key: value['fingerprint'] for key, value in artifacts.items()}
        known.update({'stage_' + stage: metadata_fingerprint(stage) for stage in STAGES})
        known['planner_policy'] = _policy_fingerprint()
        return {key: known.get(key) for key in attempt['inputs']}

    def _verify(self, pid, state, graph, attempt, artifacts, output):
        from dclab_rnd.tasks import verifier
        from dclab_rnd import experience
        from dclab_core.task_recovery import rebase_verified
        original = graph
        if attempt['graph_fingerprint'] != graph.fingerprint:
            original = next((TaskGraph.read(g) for g in state['graph_history']
                             if g['fingerprint'] == attempt['graph_fingerprint']), None)
            if original is None:
                raise Blocked('original_graph_missing')
        inputs = self._inputs(attempt, artifacts)
        result = verifier.verify(self.projects, pid, original, attempt, inputs=inputs, output=output, saver=self.saver, current_graph=graph)
        if original != graph:
            try:
                result = rebase_verified(result, graph, inputs=inputs, outputs=result['outputs'], evidence=result['evidence'],
                                         confirmed_source_revision=True)
            except ValueError:
                raise Blocked('preserved_verification_stale') from None
        if not any(v['id'] == result['id'] for v in state['verifications']):
            state['verifications'].append(result)
            experience.record('agent.step', 'system', {'agent': 'task.verifier', 'n': len(state['verifications']),
                              'state': result['id'], 'reply': None, 'tool': None, 'verdict': result['status'], 'seconds': 0}, project=pid)
        return result

    def verify(self, project_id, task_id):
        """Read-only owner inspection; save only verification metadata in the existing task saver."""
        with self.projects.turn(project_id, require_process=True):
            state = self.state(project_id)
            if state is None:
                return {'status': 'blocked', 'reasons': ['task_not_attempted']}
            state.setdefault('verifications', [])
            try:
                project = self.projects.get(project_id)
                caps = capability_snapshot(self.registry, allowed_tools=self.allowed)
                graph, _, _, mapping = self._sources(project, state, caps)
                self._observed(project, state, graph, mapping, until=task_id)
                result = next(v for v in reversed(state['verifications']) if v['task_id'] == task_id)
            except Exception as error:
                reason = str(error) if isinstance(error, Blocked) else 'owner_evidence_unavailable'
                measured = getattr(error, 'verification', None)
                if measured is not None and measured['task_id'] != task_id:
                    measured = None
                result = measured if reason.startswith('acceptance_') and measured is not None else {'status': 'stale' if reason in ('source_or_capability_changed', 'runtime_code_changed',
                           'owner_record_changed', 'owner_inputs_changed', 'holdout_approval_changed') else 'blocked',
                          'task_id': task_id, 'reasons': [reason], 'evidence': []}
                if 'version' not in result:
                    result.update(version='tp05.verification.v1', graph_fingerprint=state['graph']['fingerprint'],
                                  violated_criteria=[task_id + '_lineage'], observed_source=self.source_revision(project_id))
                    prior = next((a for a in reversed(state['attempts']) if a['task'] == task_id), None)
                    if prior:
                        result.update(plan_fingerprint=prior['plan_fingerprint'], inputs=prior['inputs'],
                                      evidence=[prior['evidence_ref']] if prior.get('evidence_ref') else [])
                from dclab_rnd import experience
                experience.record('agent.step', 'system', {'agent': 'task.verifier', 'n': len(state['verifications']) + 1,
                                  'state': task_id, 'reply': None, 'tool': None, 'verdict': result['status'], 'seconds': 0}, project=project_id)
            if 'id' not in result:
                result['id'] = 'verify_' + metadata_fingerprint(result)[:24]
                if not any(v['id'] == result['id'] for v in state['verifications']):
                    state['verifications'].append(copy.deepcopy(result))
            self._save(project_id, state, 'verification', task_id)
            return copy.deepcopy(result)

    def replan(self, project_id, task_id):
        """A new validated plan, only after trusted no-handler-entry evidence. Never invokes a tool."""
        from dclab_core.task_recovery import classify_failure
        with self.projects.turn(project_id, require_process=True):
            state = self.state(project_id)
            if state is None:
                return {'status': 'blocked', 'reason': 'task_not_attempted'}
            attempts = [a for a in state['attempts'] if a['task'] == task_id and not a.get('invalidated_by')]
            if not attempts:
                return {'status': 'blocked', 'reason': 'task_not_attempted'}
            previous = attempts[-1]
            if previous['status'] in ('started', 'unknown'):
                try:
                    caps = capability_snapshot(self.registry, allowed_tools=self.allowed)
                    self._sources(self.projects.get(project_id), state, caps)
                    self._evidence(project_id, task_id, previous)
                    return self._finish(project_id, state, previous, reconciled=True)
                except Exception:
                    pass  # inconclusive owner evidence can never authorize a replacement effect
            category = previous.get('failure_category', 'ambiguous_effect')
            effect = 'completed' if previous['status'] in TERMINAL else 'none' if previous['status'] == 'no_effect' else 'unknown'
            project = self.projects.get(project_id)
            if not self.enabled:
                return {'status': 'disabled', 'reason': 'task_execution_disabled'}
            if effect == 'none' and (self.projects.read_stage(project_id, task_id) is not None
                                    or project['stages'].get(task_id, {}).get('status', 'pending') != 'pending'
                                    or (self.projects.data_dir(project_id) / 'holdout_receipt.json').exists()):
                return {'status': 'blocked', 'reason': 'owner_effect_requires_reconciliation'}
            decision = classify_failure(category, effect=effect, attempts=len(attempts), holdout_consumed=project.get('holdout_uses', 0) > 0)
            if decision['action'] != 'replan':
                return {'status': 'blocked', **decision}
            if len(state['replans']) >= 2 or any(r['attempt'] == previous['id'] for r in state['replans']):
                return {'status': 'blocked', 'reason': 'replan_limit_or_cycle'}
            try:
                caps = capability_snapshot(self.registry, allowed_tools=self.allowed)
                graph, source, workflow, mapping = self._sources(project, state, caps)
                observed = self._observed(project, state, graph, mapping)
                remaining = {'max_tool_calls': min(self.budget.max_tool_calls, state['budget']['max_tool_calls']) - state['tool_calls'],
                             'max_seconds': max(0, int(min(self.budget.max_seconds, state['budget']['max_seconds']) - state['seconds'])),
                             'max_cost_eur': 0.0}
                plan, _ = plan_task(workflow, source, project, graph=graph, task_id=task_id, observed_state=observed,
                                    capabilities=caps, remaining_budget=remaining, revision=previous['plan_revision'] + 1)
                if plan is None or plan.status != 'ready':
                    raise Blocked('replacement_plan_not_ready')
                resolve_action(plan, 'call', workflow, source, project, graph=graph, observed_state=observed,
                               capabilities=caps, remaining_budget=remaining)
                revision = {'attempt': previous['id'], 'task': task_id, 'decision': decision, 'plan': plan.model_dump(mode='json')}
                state['replans'].append(revision)
                state['retry_authorization'] = revision
                self._save(project_id, state, 'replanned', task_id)
                from dclab_rnd import experience
                experience.record('agent.step', 'system', {'agent': 'task.recovery', 'n': len(state['replans']),
                                  'state': plan.fingerprint, 'reply': None, 'tool': None, 'verdict': 'new_plan_validated', 'seconds': 0}, project=project_id)
                return {'status': 'ready', 'plan_revision': plan.revision, 'plan_fingerprint': plan.fingerprint}
            except Exception as error:
                return {'status': 'blocked', 'reason': str(error) if isinstance(error, Blocked) else 'replacement_plan_invalid'}

    def source_revision(self, project_id):
        """Fingerprint for an explicit source-bound confirmation through the existing owner approval path."""
        project = self.projects.get(project_id)
        from dclab_rnd.tasks.adapter import _execution_context
        context, _ = _execution_context(project, project.get('data') or {})
        if isinstance(context.get('brief'), dict):
            context['brief'] = {k: v for k, v in context['brief'].items() if k != 'approval'}
        return metadata_fingerprint({'context': context, 'data': project.get('data'), 'solution': project.get('solution'),
                                     'tables': project.get('tables'), 'labels': project.get('labels')})

    def adopt_sources(self, project_id):
        """Observe a new human confirmation; never write an approval or clear an owner artifact."""
        from dclab_core.task_recovery import invalidated_tasks, task_definition
        from dclab_rnd.studio import data
        with self.projects.turn(project_id, require_process=True):
            state = self.state(project_id)
            if not self.enabled:
                return {'status': 'disabled', 'reason': 'task_execution_disabled'}
            if state is None:
                return {'status': 'blocked', 'reason': 'task_not_attempted'}
            project = self.projects.get(project_id)
            expected = 'Task source revision ' + self.source_revision(project_id)
            if state.get('code') != _code() or state.get('version') != VERSION:
                return {'status': 'blocked', 'reason': 'runtime_code_changed'}
            if project.get('holdout_uses') or (self.projects.data_dir(project_id) / 'holdout_receipt.json').exists():
                return {'status': 'blocked', 'reason': 'holdout_consumed_or_intended'}
            approval_id = metadata_fingerprint(project.get('signoff'))
            if (project.get('signoff') or {}).get('reason') != expected or approval_id in state['source_approvals']:
                return {'status': 'blocked', 'reason': 'source_bound_human_confirmation_required'}
            if len(state['graph_history']) >= 2 or any(a['status'] in ('started', 'unknown') for a in state['attempts']):
                return {'status': 'blocked', 'reason': 'source_revision_limit_or_ambiguous_effect'}
            try:
                data.data_path(self.projects, project)
                old = TaskGraph.read(state['graph'])
                caps = capability_snapshot(self.registry, allowed_tools=self.allowed)
                source = plan_from_project(project)
                workflow = solution_workflow(problem_card(brief_from_project(project)))
                fresh, mapping = adapt_workflow(workflow, source, project, capabilities=caps, revision=old.revision + 1)
                if fresh is None or mapping['status'] != 'mapped':
                    raise Blocked('replacement_graph_invalid')
                changed = {key for key, value in mapping['external_artifacts'].items()
                           if key not in state.get('external_artifacts', {}) or state['external_artifacts'][key] != value}
                verified = {a['task'] for a in state['attempts'] if a['status'] in TERMINAL and not a.get('invalidated_by')}
                semantic_changed = state['source_identity'] != self.source_revision(project_id)
                definitions = {n.id: task_definition(n) for n in fresh.nodes if n.kind != 'group'}
                changed_tasks = {n.id for n in old.nodes if n.kind != 'group'
                                 and (semantic_changed or definitions.get(n.id) != task_definition(n))}
                result = invalidated_tasks(old, changed_artifacts=changed, changed_tasks=changed_tasks, verified=verified)
                for task in set(result['invalidated']) & set(STAGES):
                    if project['stages'][task].get('status') != 'pending':
                        raise Blocked('owner_must_clear_stale_stages')
                state['source_identity'] = self.source_revision(project_id)
                state['source_approvals'].append(approval_id)
                state['graph_history'].append(state['graph'])
                state['graph'] = fresh.model_dump(mode='json')
                state['external_artifacts'] = mapping['external_artifacts']
                state.pop('holdout_permission', None)
                state.pop('retry_authorization', None)
                for attempt in state['attempts']:
                    if attempt['task'] in result['invalidated'] and not attempt.get('invalidated_by'):
                        attempt['invalidated_by'] = fresh.fingerprint
                self._save(project_id, state, 'source_revision_confirmed')
                return {'status': 'ready', 'graph_revision': fresh.revision, **result}
            except Exception as error:
                return {'status': 'blocked', 'reason': str(error) if isinstance(error, Blocked) else 'source_revision_invalid'}

    def _gate(self, project, state, graph, mapping):
        if state.get("holdout_permission"):
            return
        models = next((a for a in reversed(state["attempts"]) if a["task"] == "models" and a["status"] in TERMINAL and not a.get("invalidated_by")), None)
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
            state["seconds"] += charged - attempt["budget"].get("charged_seconds", reserved)
            state["budget_overrun"] = state.get("budget_overrun", False) or duration > reserved
            attempt["budget"].update(seconds=duration, recovered_charge_seconds=charged, charged_seconds=charged)
        book = ledger.current_ledger()
        ids, capture = [], "not_available"
        if book and evidence["run"]:
            try:
                ids = sorted({r.id for run in (evidence["run"], attempt["id"]) for r in book.records(project=pid, run=run)})
                capture = "linked"
            except Exception:  # ledger capture is best effort; it does not erase conclusive owner completion evidence
                capture = "unavailable"
        project = self.projects.get(pid)
        caps = capability_snapshot(self.registry, allowed_tools=self.allowed)
        graph, _, _, mapping = self._sources(project, state, caps)
        observed = self._observed(project, state, graph, mapping)
        verification = self._verify(pid, state, graph, attempt, observed["artifacts"], evidence)
        attempt.update(status="reconciled" if reconciled else "completed", finished=now(),
                       output_fingerprint=evidence["fingerprint"], owner_run=evidence["run"], evidence_ref=evidence["reference"],
                       decision_ids=ids, decision_capture=capture, acceptance_verification=verification["status"], verification_id=verification["id"])
        if attempt['task'] != 'report':
            attempt['scientific_fingerprint'] = metadata_fingerprint({k: v for k, v in self.projects.read_stage(pid, attempt['task']).items()
                                                                      if k not in ('decision', 'notes', 'explanation')})
        checkpoint = self._save(pid, state, attempt["status"], attempt["task"])
        return {"status": attempt["status"] if verification['status'] == 'verified' else 'blocked',
                "reason": None if verification['status'] == 'verified' else 'acceptance_' + verification['status'], "attempt_id": attempt["id"], "checkpoint": checkpoint,
                "owner_run": evidence["run"], "acceptance_verification": verification["status"], "verification": verification}

    def _invoke(self, pid, attempt, action, arguments):
        from dclab_rnd import experience
        from dclab_rnd.agents import harness
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.models.gateway import private_scope
        from dclab_rnd.studio.lineage import task_attempt

        clock = time.monotonic()
        with experience.bind(project=pid), harness.session("orchestra.solution", run_id=attempt["id"], params={"task": attempt["task"]}) as session:
            with task_attempt(attempt["id"]), private_scope(True):  # first execution slice is deterministic; no model spend
                from dclab_rnd.agents.registry import observe_call
                with observe_call() as observation:
                    try:
                        reply = self.registry.call(action.capability_name, arguments, context=Toolbox(self.projects, registry=self.registry), scope="project")
                    except (TimeoutError, ConnectionError):
                        attempt["failure_category"] = "transient"
                        reply = {"error": "tool_exception"}
                    except Exception:
                        attempt["failure_category"] = "domain_failure"
                        reply = {"error": "tool_exception"}
                    attempt["boundary"] = dict(observation)
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
                previous = next((a for a in reversed(state["attempts"]) if a["task"] == task_id and not a.get("invalidated_by")), None)
                if previous:
                    if previous["status"] in TERMINAL:
                        self._observed(project, state, graph, mapping)
                        return {"status": "already_completed", "attempt_id": previous["id"], "owner_run": previous.get("owner_run")}
                    if previous["status"] == "started":
                        attempt = previous
                        return self._finish(project_id, state, previous, reconciled=True)
                    if previous["status"] != "no_effect" or state.get("retry_authorization", {}).get("attempt") != previous["id"]:
                        raise Blocked("unknown_effect_requires_owner_reconciliation")
                    plan = TaskExecutionPlan.read(state["retry_authorization"]["plan"])
                if any(a["status"] not in (*TERMINAL, "no_effect") and not a.get("invalidated_by") for a in state["attempts"]):
                    raise Blocked("unresolved_previous_attempt")
                if task_id in ("final", "report"):
                    self._gate(project, state, graph, mapping)
                observed = self._observed(project, state, graph, mapping)
                if task_id in ('features', 'models', 'final'):
                    from dclab_rnd.tasks.verifier import science_readiness
                    problem = science_readiness(self.projects, project_id, self.saver)
                    if problem:
                        raise Blocked(problem)
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
                           "acceptance_verification": "not_measured", "budget": {"tool_calls": 1, "seconds": None, "cost_eur": 0.0}}
                state["attempts"].append(attempt)
                state["tool_calls"] += 1  # charged durably before effect; an ambiguous attempt cannot reclaim its allowance
                state["seconds"] += plan.budget.max_seconds
                attempt["budget"]["reserved_seconds"] = plan.budget.max_seconds
                self._save(project_id, state, "started", task_id)
                began = time.monotonic()
                reply = self._invoke(project_id, attempt, plan.actions[0], args)
                elapsed = time.monotonic() - began
                attempt["budget"].update(seconds=elapsed, charged_seconds=elapsed)
                state["seconds"] += elapsed - plan.budget.max_seconds
                state["budget_overrun"] = elapsed > plan.budget.max_seconds
                if isinstance(reply, dict) and (reply.get("blocked") or reply.get("status") in ("blocked", "needs_approval")
                        or (reply.get('error') and attempt.get('boundary', {}).get('phase') == 'guard'
                            and attempt.get('boundary', {}).get('handler_entered') is False)):
                    attempt.update(status="blocked", reason="guard_refused", failure_category='missing_approval' if reply.get('status') == 'needs_approval' else 'policy_refusal')
                    self._save(project_id, state, "guard_refused", task_id)
                    return {"status": "blocked", "reason": "guard_refused", "attempt_id": attempt["id"]}
                boundary = attempt.get("boundary", {})
                if (isinstance(reply, dict) and reply.get("error") and boundary.get("seen") is True
                        and boundary.get("phase") == "validation" and boundary.get("handler_entered") is False):
                    attempt.update(status="no_effect", reason="registry_validation_failed",
                                   failure_category=attempt.get("failure_category", "invalid_arguments"))
                    self._save(project_id, state, "no_effect", task_id)
                    return {"status": "blocked", "reason": "registry_validation_failed", "attempt_id": attempt["id"]}
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
