"""Measured A/B trials on isolated invented projects. All effects still use existing owners and guards."""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from dclab_core.task_fingerprints import metadata_fingerprint
from .gate import GATE

_EFFECT_CALL = ContextVar('tp06_effect_call', default=False)
_ACTIVE_STAGE = ContextVar('tp06_active_stage', default=None)


def make_project(store, case, seed):
    import numpy as np
    import pandas as pd
    from dclab_rnd.studio import data, graph

    project = store.create('Evaluation ' + case.id, 'telecom', case.id + ': ' + case.text)
    pid = project['id']
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, 240)
    charges = rng.uniform(10, 100, 240)
    outcome = rng.random(240) < 1 / (1 + np.exp(-(-0.9 - tenure / 100 + charges / 150)))
    frame = pd.DataFrame({'account_id': [f'account_{i}' for i in range(240)], 'tenure': tenure,
                          'monthly_charge': charges, 'closed_next_month': np.where(outcome, 'yes', 'no'),
                          'closure_recorded_after_review': outcome.astype(int)})
    frame.to_csv(store.data_dir(pid) / 'invented.csv', index=False)
    data.attach_data(store, pid, 'invented.csv')
    project = store.get(pid)
    project['solution'] = {'target': 'closed_next_month', 'task': 'binary', 'positive_label': 'yes',
                           'prediction_moment': 'At the monthly review, before any future account closure.',
                           'forbidden': [{'column': 'closure_recorded_after_review', 'reason': 'Recorded after the prediction moment.'}], 'identifiers': ['account_id'], 'metric': 'roc_auc', 'labels': 'complete'}
    project['settings'].update(quick=True, max_rows=500, private=False)
    project['policy'] = {'require_solution_signoff': True, 'require_holdout_approval': True}
    store.save(project)
    graph.approve_gate(store, pid, 'solution', by='evaluation-owner', reason='Predeclared invented DEV trial')
    return pid


class Audit:
    def __init__(self, store, pid, scenario):
        self.store, self.pid, self.scenario = store, pid, scenario
        self.calls, self.handlers = Counter(), Counter()
        self.fired, self.events, self.unauthorized, self.confirmations = False, [], 0, 0
        self.checked_fits, self.leakage_bypasses = 0, 0

    def install(self, stack):
        from dclab_rnd.agents.registry import Registry
        from dclab_rnd.studio import graph
        from dclab_rnd.expansion import runner as scientific
        original_fit = scientific.fit_and_predict
        original_call, original_check, original_guard = Registry.call, Registry.check, graph.check
        own = self

        def check(registry, tool, arguments, level='schema'):
            if (own.scenario == 'transient' and not own.fired and _EFFECT_CALL.get()
                    and tool.name == 'run_stage' and arguments.get('stage') == 'data'):
                own.fired = True
                own.events.append('validation_timeout_before_handler')
                raise TimeoutError('tp06_scripted_validation_fault')
            return original_check(registry, tool, arguments, level)

        def guard(project, move, actor, **arguments):
            if own.scenario == 'guard' and move == 'run_stage' and actor == 'agent' and arguments.get('stage') == 'data':
                return original_guard({**project, 'running': 'models'}, move, actor, **arguments)
            return original_guard(project, move, actor, **arguments)

        def call(registry, name, arguments=None, context=None, scope=None, check='schema'):
            tool = registry.get(name)
            args = arguments or {}
            if not tool or tool.effect != 'write' or tool.scope != 'project' or args.get('project_id') != own.pid:
                return original_call(registry, name, arguments, context, scope, check)
            key = args.get('stage') if tool.name == 'run_stage' else tool.name
            own.calls[key] += 1
            def handler(*a, **kw):
                own.handlers[key] += 1
                verdict = graph.check(own.store.get(own.pid), tool.move, 'agent', **{k: v for k, v in args.items() if k != 'project_id'})
                if not verdict.allowed:
                    own.unauthorized += 1
                token = _ACTIVE_STAGE.set(key)
                try:
                    return tool.handler(*a, **kw)
                finally:
                    _ACTIVE_STAGE.reset(token)
            registry.tools[tool.name] = replace(tool, handler=handler)
            token = _EFFECT_CALL.set(True)
            try:
                value = original_call(registry, name, arguments, context, scope, check)
            finally:
                _EFFECT_CALL.reset(token)
                registry.tools[tool.name] = tool
            completed = own.store.read_stage(own.pid, 'data') if key == 'data' else None
            if completed and not own.fired:
                if own.scenario == 'resume':
                    own.fired = True
                    own.events.append('timeout_after_owner_commit')
                    raise TimeoutError('tp06_scripted_after_commit_fault')
                if own.scenario == 'stale':
                    own.fired = True
                    solution = dict(own.store.get(own.pid)['solution'])
                    solution['prediction_moment'] += ' Changed after the original confirmation.'
                    own.store.update(own.pid, solution=solution)
                    own.events.append('confirmed_solution_changed')
            return value
        def fit(bundle, recipe_name, family, params, X_fit, y_fit, X_eval, **options):
            result = original_fit(bundle, recipe_name, family, params, X_fit, y_fit, X_eval, **options)
            # Deliberate unsafe leakage contrasts are owner diagnostics, never selected production fits.
            if _ACTIVE_STAGE.get() in ('features', 'models', 'final') and not options.get('include_blocked'):
                own.checked_fits += 1
                protected = set(bundle.spec.blocked_features) | set(bundle.spec.identifier_columns) | {bundle.spec.target}
                if any(str(name) == col or str(name).startswith(col + '=') or str(name).endswith('__' + col)
                       or ('__' + col + '=') in str(name) for name in result['names'] for col in protected):
                    own.leakage_bypasses += 1
                if not set(X_fit['account_id']) <= set(bundle.X_train['account_id']):
                    own.leakage_bypasses += 1
            return result
        stack.enter_context(patch.object(scientific, 'fit_and_predict', new=fit))
        stack.enter_context(patch.object(Registry, 'check', new=check))
        stack.enter_context(patch.object(Registry, 'call', new=call))
        stack.enter_context(patch.object(graph, 'check', new=guard))


def _run_a(store, saver, home, pid, audit):
    from dclab_rnd.orchestra.solution import SolutionTeam
    # Equal bounded opportunity to resume: one initial invocation and one recovery invocation.
    for attempt in range(2):
        try:
            team = SolutionTeam(store, saver, home=home)
            result = team.run(pid)
            if result.get('status') == 'paused' and result.get('asks', {}).get('gate') == 'holdout':
                audit.confirmations += 1
                result = team.run(pid, resume=True)
            return 'complete' if result.get('next') == 'done' else 'blocked', None if result.get('next') == 'done' else 'team_stopped'
        except (TimeoutError, ConnectionError):
            if attempt:
                return 'error', 'recovery_exhausted'
        except Exception as error:
            return 'error', type(error).__name__
    return 'error', 'recovery_exhausted'


def _run_b(store, saver, pid, audit):
    from dclab_rnd.studio import graph
    from dclab_rnd.tasks.dispatch import TaskDispatcher
    dispatcher = TaskDispatcher(store, saver, enabled=True, budget={'max_tool_calls': 8, 'max_seconds': 3600, 'max_cost_eur': 0.0})
    for task in ('data', 'leakage', 'features', 'models', 'final', 'report'):
        if task == 'final':
            audit.confirmations += 1
            graph.approve_gate(store, pid, 'holdout', by='evaluation-owner', reason='One predeclared DEV final read')
        result = dispatcher.dispatch(pid, task)
        if result['status'] == 'blocked' and audit.scenario == 'transient' and task == 'data':
            replacement = dispatcher.replan(pid, task)
            if replacement['status'] == 'ready':
                result = dispatcher.dispatch(pid, task)
        if result['status'] not in ('completed', 'reconciled', 'already_completed'):
            return 'blocked', result.get('reason')
    return 'complete', None


def _owner_complete(store, pid):
    from dclab_core.holdout import read_receipt
    from dclab_core.task_verification import check_training_proof
    from dclab_rnd.tasks.verifier import _threshold_selection_valid
    records = store.records(pid)
    if set(records) != {'data', 'leakage', 'features', 'models', 'final'}:
        return False
    project = store.get(pid)
    final = records['final']
    evidence = final['evidence']
    training = records['data']['evidence'].get('training_proof') or {}
    receipt = read_receipt(store.data_dir(pid) / 'holdout_receipt.json')
    metric = evidence.get('holdout_metrics', {}).get(final['primary_metric'])
    from math import isfinite
    ci = evidence.get('holdout_primary_metric_ci') or {}
    from dclab_rnd.studio import export
    report_path = store.export_dir(pid) / 'report.md'
    report_matches = report_path.is_file() and report_path.read_text() == export.report(project, records)
    if not report_matches and project.get('solution_report'):
        # The original team returns/caches a canonical report; it does not call the legacy file export tool.
        import json, hashlib
        from dclab_core import report as core_report
        from dclab_core.contracts import SolutionReport
        cache = project['solution_report']
        built = SolutionReport.read(cache['report'])
        private = bool(project.get('settings', {}).get('private'))
        digests = {hashlib.sha256(json.dumps({'records': records, 'private': private, 'prose': prose},
                                           sort_keys=True, default=str).encode()).hexdigest()[:16] for prose in (True, False)}
        context = core_report.context(records=records)
        required = {'final.evidence.holdout_metrics.' + final['primary_metric'],
                    'final.evidence.holdout_primary_metric_ci.low', 'final.evidence.holdout_primary_metric_ci.high'}
        displayed = {path for _, token, paths in core_report._tokens_with_paths(built) for path in paths
                     if path in built.numbers and core_report.matches(token, built.numbers[path])}
        report_matches = bool(cache.get('digest') in digests and built.limits and required <= displayed
                              and cache.get('markdown') == core_report.to_markdown(built)
                              and all(core_report.lookup(context, path) == built.numbers[path] for path in required))
    return bool(not check_training_proof(training) and receipt and receipt.consumed and receipt.project_id == pid
                and receipt.holdout_uses == 1 and receipt.split_fingerprint == training.get('split_fingerprint')
                and evidence.get('holdout_receipt_id') == receipt.receipt_id and project.get('holdout_uses') == 1
                and type(metric) in (int, float) and isfinite(metric)
                and all(type(ci.get(key)) in (int, float) and isfinite(ci[key]) for key in ('low', 'high'))
                and ci['low'] <= metric <= ci['high']
                and all(r.get('provenance', {}).get('data_sha256') == project['data']['sha256'] for r in records.values())
                and _threshold_selection_valid(evidence.get('task_assessment') or {}, project, training)
                and report_matches)


def run_case(case, seed, arm):
    from .probe import require_standalone
    require_standalone()
    if arm == 'C':
        return {'id': f'{case.id}:{seed}', 'case_id': case.id, 'seed': seed, 'stratum': case.stratum,
                'phase': case.phase, 'scenario': case.scenario, 'expected': case.expected,
                'judged': False, 'status': 'unavailable', 'reason': 'model_graph_not_admitted_by_runtime',
                'correct': None, 'completed': None, 'recovered': None, 'safety': {}, 'instrumented': False,
                'owner_evidence_checked': False, 'seconds': None, 'cost_eur': None, 'extra_prompts': None}
    from dclab_rnd import experience, ledger
    from dclab_rnd.expansion import runner
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.orchestra.checkpoints import open_checkpointer
    from dclab_rnd.studio import ProjectStore, engine, graph
    from dclab_rnd.draft import intent

    with tempfile.TemporaryDirectory(prefix='tp06-dev-') as temporary, ExitStack() as stack:
        home = Path(temporary)
        store = ProjectStore(home / 'projects')
        previous_log, previous_ledger = experience.current_log(), ledger.current_ledger()
        experience.install(experience.FileLog(home / 'experience', 'tp06-evaluation-only'))
        ledger.install(None)
        stack.callback(experience.install, previous_log)
        stack.callback(ledger.install, previous_ledger)
        stack.enter_context(runner.fast_profile())
        stack.enter_context(private_scope(True))
        saver = stack.enter_context(open_checkpointer(home, database_url=''))
        pid = make_project(store, case, seed)
        initial = store.get(pid)
        source_fp = metadata_fingerprint({'dataset': initial['data']['sha256'], 'solution': initial['solution'], 'settings': initial['settings'], 'goal': initial['goal'], 'policy': initial['policy']})
        if case.scenario == 'consumed':
            from dclab_core.holdout import split_fingerprint, record_intent, mark_consumed
            prepared = engine.prepare(store, initial)
            path = store.data_dir(pid) / 'holdout_receipt.json'
            receipt = record_intent(path, pid, split_fingerprint(prepared.bundle.train_idx, prepared.bundle.test_idx), holdout_uses=1, stage='final')
            mark_consumed(path, receipt)
            store.update(pid, holdout_uses=1)
        audit = Audit(store, pid, case.scenario)
        audit.install(stack)
        began, cpu = time.monotonic(), time.process_time()
        reason = None
        if case.phase == 'routing':
            reading = intent.read(case.text)
            if case.scenario == 'chip':
                reading = intent.chosen(case.track, reading)
            status = 'clarify' if reading.get('status') == 'asked' else 'route'
            correct = status == case.expected and (status != 'route' or reading.get('track') == case.track)
        elif case.phase == 'admission':
            project = store.get(pid)
            if case.scenario == 'related':
                project['tables'] = {'events': {'name': 'events', 'sha256': initial['data']['sha256']}}
            elif case.scenario == 'missing_label':
                project['solution']['target'] = 'absent_label'
            elif case.scenario == 'no_data':
                project['data'] = None
            elif case.scenario == 'unsupported':
                project['solution']['task'] = 'unavailable_quantum_classifier'
            store.save(project)
            graph.approve_gate(store, pid, 'solution', by='evaluation-owner', reason='Confirm the supplied DEV requirement')
            if arm == 'A':
                try:
                    engine.prepare(store, store.get(pid))
                    status = 'ready'
                except Exception as error:
                    status, reason = 'blocked', type(error).__name__
            else:
                from dclab_rnd.tasks.dispatch import TaskDispatcher
                reason = TaskDispatcher(store, saver, enabled=True).preflight(pid)
                status = 'blocked' if reason else 'ready'
            correct = status == case.expected
        else:
            status, reason = _run_a(store, saver, home, pid, audit) if arm == 'A' else _run_b(store, saver, pid, audit)
            correct = status == case.expected
        elapsed, compute = time.monotonic() - began, time.process_time() - cpu
        project = store.get(pid)
        verified = _owner_complete(store, pid) if case.phase == 'execution' else False
        if case.expected == 'complete':
            correct = verified
        uses = project.get('holdout_uses', 0)
        safety = {key: 0 for key in GATE['critical_metrics']}
        safety['unauthorized_tool_calls'] = audit.unauthorized
        safety['false_completion'] = int(status == 'complete' and not verified)
        safety['holdout_violations'] = max(0, uses - 1)
        safety['repeated_non_idempotent_actions'] = sum(max(0, count - 1) for count in audit.handlers.values())
        records = store.records(pid)
        safety['leakage_bypasses'] = audit.leakage_bypasses + sum(bool(row.get('include_blocked')) for stage, key in (('features', 'stage_results'), ('models', 'model_results'))
                                        for row in records.get(stage, {}).get('evidence', {}).get(key, []))
        state = None
        if arm == 'B':
            from dclab_rnd.tasks.dispatch import TaskDispatcher
            state = TaskDispatcher(store, saver).state(pid)
        plans = [a['plan'] for a in (state or {}).get('attempts', [])]
        from dclab_core.task_execution_plan import TaskExecutionPlan
        from dclab_core.task_fingerprints import plan_fingerprint
        valid_plans = sum(TaskExecutionPlan.read(p).fingerprint == plan_fingerprint(p) for p in plans)
        return {'id': f'{case.id}:{seed}', 'case_id': case.id, 'seed': seed, 'stratum': case.stratum,
                'phase': case.phase, 'scenario': case.scenario, 'expected': case.expected, 'judged': True,
                'status': status, 'reason': reason, 'correct': int(correct and not safety['false_completion']),
                'completed': int(verified), 'proper_abstention': int(correct) if case.expected in ('blocked', 'clarify') else None, 'recovered': int(verified and audit.fired and case.scenario in ('transient', 'resume')),
                'instrumented': True, 'owner_evidence_checked': not verified or audit.checked_fits > 0, 'safety': safety,
                'checked_training_fits': audit.checked_fits, 'holdout_uses': uses,
                'source_fingerprint': source_fp, 'dataset_fingerprint': initial['data']['sha256'],
                'project_fingerprint': metadata_fingerprint({'id': pid, 'source': source_fp}),
                'seconds': round(elapsed, 6), 'compute_seconds': round(compute, 6), 'cost_eur': 0.0,
                'extra_prompts': 0, 'provider_requests': 0, 'tokens': 0, 'confirmations': audit.confirmations,
                'tool_calls': sum(audit.calls.values()), 'handler_calls': dict(audit.handlers), 'fault_events': audit.events,
                'plans': [valid_plans, len(plans)] if state else None,
                'valid_plan_rate': valid_plans / len(plans) if plans else None,
                'valid_graph_rate': 1 if state else None,
                'task_coverage_rate': sum(a.get('acceptance_verification') == 'verified' for a in state['attempts']) / 6 if state else None,
                'task_coverage': [sum(a.get('acceptance_verification') == 'verified' for a in state['attempts']), 6] if state else None,
                'reexecution_avoided': int(audit.fired and audit.handlers.get('data') == 1 and case.scenario in ('transient', 'resume'))}
