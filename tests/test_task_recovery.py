"""TP-05 pure recovery policy: no tools, models or holdout data."""
import unittest

from dclab_core.task_recovery import classify_failure, invalidated_tasks
from tests.task_validation_cases import case


class TaskRecoveryPolicyTests(unittest.TestCase):
    def test_transient_retry_requires_trusted_no_handler_entry(self):
        self.assertEqual(classify_failure("transient", effect="none", attempts=1)["action"], "replan")
        self.assertEqual(classify_failure("transient", effect="unknown", attempts=1)["action"], "stop")

    def test_completed_timeout_is_verified_not_repeated(self):
        self.assertEqual(classify_failure("transient", effect="completed", attempts=1)["action"], "verify")
        self.assertEqual(classify_failure("transient", effect="none", attempts=2)["reason"], "attempt_limit")

    def test_changed_inputs_invalidate_only_consumers_and_dependencies(self):
        graph, _, _, _ = case()
        result = invalidated_tasks(graph, changed_artifacts={"features_record"}, verified={"data", "leakage", "features", "models"})
        self.assertEqual(result["invalidated"], ["final", "holdout", "models", "report"])
        self.assertEqual(result["preserved"], ["data", "features", "leakage"])


    def test_policy_capability_approval_and_domain_failures_always_stop(self):
        for category in ('missing_capability', 'policy_refusal', 'missing_approval', 'domain_failure'):
            self.assertEqual(classify_failure(category, effect='none')['action'], 'stop')
        self.assertEqual(classify_failure('changed_input', effect='none')['action'], 'confirm_sources')
        self.assertEqual(classify_failure('invalid_arguments', effect='none')['action'], 'replan')
        self.assertEqual(classify_failure('ambiguous_effect')['action'], 'stop')
        self.assertEqual(classify_failure('transient', effect='none', holdout_consumed=True)['reason'], 'holdout_consumed')

    def test_independent_verification_sibling_survives_scientific_branch_failure(self):
        from dclab_core.task_graph import TaskGraph
        from dclab_core.task_fingerprints import graph_fingerprint
        graph, _, _, _ = case()
        body = graph.model_dump(mode='json')
        body['nodes'].append({'id': 'metadata_check', 'parent_id': 'root', 'kind': 'verification',
                              'title': 'Read the data record', 'objective': 'Check existing metadata without model selection',
                              'needs': ['data_record'], 'dependencies': ['data'],
                              'acceptance_checks': [{'id': 'metadata_lineage', 'kind': 'lineage',
                                                     'description': 'Read the existing data record', 'evidence_refs': ['data_record']}]})
        body['edges'].extend([{'source': 'root', 'target': 'metadata_check', 'kind': 'grouping'},
                              {'source': 'data', 'target': 'metadata_check', 'kind': 'dependency'}])
        body['fingerprint'] = graph_fingerprint(body)
        result = invalidated_tasks(TaskGraph.read(body), changed_tasks={'leakage'}, verified={'data', 'metadata_check', 'models'})
        self.assertEqual(result['preserved'], ['data', 'metadata_check'])
        self.assertNotIn('metadata_check', result['invalidated'])

from unittest.mock import patch
from tests import test_task_dispatch as fixtures


class TaskRecoveryIntegrationTests(unittest.TestCase):
    database_url = ''
    make_store = fixtures.TaskDispatchTests.make_store
    setUp = fixtures.TaskDispatchTests.setUp
    dispatcher = fixtures.TaskDispatchTests.dispatcher

    def test_transient_before_handler_gets_new_validated_plan_and_keeps_history(self):
        d = self.dispatcher()
        original = d.registry.check
        def timeout(tool, arguments, level='schema'):
            raise TimeoutError('private payload must not persist')
        with patch.object(d.registry, 'check', side_effect=timeout):
            first = d.dispatch(self.pid, 'data')
        self.assertEqual(first['reason'], 'registry_validation_failed', first)
        self.assertIsNone(self.store.read_stage(self.pid, 'data'))
        self.assertEqual(d.replan(self.pid, 'data')['status'], 'ready')
        result = d.dispatch(self.pid, 'data')
        self.assertEqual(result.get('acceptance_verification'), 'verified', result)
        state = d.state(self.pid)
        self.assertEqual([a['plan_revision'] for a in state['attempts']], [1, 2])
        self.assertEqual(state['tool_calls'], 2)
        self.assertNotIn('private payload', str(state))
        self.assertEqual(d.dispatch(self.pid, 'data')['status'], 'already_completed')

    def test_two_pre_handler_failures_exhaust_retry_and_do_not_loop(self):
        d = self.dispatcher()
        with patch.object(d.registry, 'check', side_effect=TimeoutError()):
            d.dispatch(self.pid, 'data')
        self.assertEqual(d.replan(self.pid, 'data')['status'], 'ready')
        self.assertEqual(d.replan(self.pid, 'data')['reason'], 'replan_limit_or_cycle')
        with patch.object(d.registry, 'check', side_effect=TimeoutError()):
            d.dispatch(self.pid, 'data')
        self.assertEqual(d.replan(self.pid, 'data')['reason'], 'attempt_limit')
        self.assertIsNone(self.store.read_stage(self.pid, 'data'))

    def test_timeout_after_success_reconciles_without_another_handler(self):
        d = self.dispatcher()
        original = d.registry.tools['run_stage']
        from dataclasses import replace
        calls = []
        def handler(*args, **kwargs):
            calls.append(1)
            original.handler(*args, **kwargs)
            raise TimeoutError('late transport failure')
        d.registry.tools['run_stage'] = replace(original, handler=handler)
        result = d.dispatch(self.pid, 'data')
        self.assertEqual(result.get('acceptance_verification'), 'verified', result)
        self.assertEqual(d.dispatch(self.pid, 'data')['status'], 'already_completed')
        self.assertEqual(len(calls), 1)

    def test_source_change_requires_fresh_source_bound_human_confirmation(self):
        d = self.dispatcher()
        d.dispatch(self.pid, 'data')
        self.store.update(self.pid, goal='Changed task objective')
        self.assertEqual(d.adopt_sources(self.pid)['reason'], 'source_bound_human_confirmation_required')
        self.assertEqual(len(d.state(self.pid)['attempts']), 1)

    def test_changed_table_is_adopted_only_after_owner_clear_and_source_confirmation(self):
        import copy
        import pandas as pd
        from dclab_rnd.studio import data, graph
        d = self.dispatcher(budget={'max_tool_calls': 8, 'max_seconds': 3600, 'max_cost_eur': 0.0})
        d.dispatch(self.pid, 'data')
        old = copy.deepcopy(d.state(self.pid)['attempts'][0])
        project = self.store.get(self.pid)
        solution = copy.deepcopy(project['solution'])
        path = data.data_path(self.store, project)
        frame = pd.read_csv(path)
        frame['tenure'] += 1
        frame.to_csv(path, index=False)
        data.attach_data(self.store, self.pid, path.name)
        project = self.store.get(self.pid)
        project['solution'] = solution
        self.store.save(project)
        graph.approve_gate(self.store, self.pid, 'solution', by='owner', reason='Task source revision ' + d.source_revision(self.pid))
        result = d.adopt_sources(self.pid)
        self.assertEqual(result['status'], 'ready', result)
        self.assertIn('data', result['invalidated'])
        result = d.dispatch(self.pid, 'data')
        self.assertEqual(result.get('acceptance_verification'), 'verified', result)
        state = d.state(self.pid)
        self.assertEqual(state['attempts'][0]['output_fingerprint'], old['output_fingerprint'])
        self.assertEqual(state['attempts'][0]['plan'], old['plan'])
        self.assertEqual(len(state['attempts']), 2)
        self.assertEqual(state['graph']['revision'], 2)
        self.assertEqual(state['tool_calls'], 2)
        self.assertEqual(d.adopt_sources(self.pid)['reason'], 'source_bound_human_confirmation_required')

    def test_confirmation_cannot_authorize_a_later_context_change(self):
        from dclab_rnd.studio import graph
        d = self.dispatcher()
        d.dispatch(self.pid, 'data')
        self.store.update(self.pid, name='A newly named project')
        graph.approve_gate(self.store, self.pid, 'solution', by='owner', reason='Task source revision ' + d.source_revision(self.pid))
        self.store.update(self.pid, exposure={'source': 'changed_after_approval'})
        self.assertEqual(d.adopt_sources(self.pid)['reason'], 'source_bound_human_confirmation_required')

    def test_solution_change_invalidates_even_when_external_artifacts_are_unchanged(self):
        import copy
        from dclab_rnd.studio import graph
        d = self.dispatcher()
        d.dispatch(self.pid, 'data')
        project = self.store.get(self.pid)
        solution = copy.deepcopy(project['solution'])
        solution['prediction_moment'] += ' A newly confirmed constraint.'
        self.store.update(self.pid, solution=solution)
        self.store.clear_stages(self.pid)
        graph.approve_gate(self.store, self.pid, 'solution', by='owner', reason='Task source revision ' + d.source_revision(self.pid))
        result = d.adopt_sources(self.pid)
        self.assertEqual(result['status'], 'ready', result)
        self.assertIn('data', result['invalidated'])

    def test_human_feature_choice_survives_and_is_used_by_models(self):
        from dclab_rnd.studio import engine
        d = self.dispatcher()
        for task in ('data', 'leakage', 'features'):
            self.assertEqual(d.dispatch(self.pid, task).get('acceptance_verification'), 'verified')
        record = self.store.read_stage(self.pid, 'features')
        choice = next(o['id'] for o in record['decision']['options'] if o['id'] != record['decision']['chosen'])
        engine.approve(self.store, self.pid, 'features', choice=choice, actor='human')
        result = d.dispatch(self.pid, 'models')
        self.assertEqual(result.get('acceptance_verification'), 'verified', result)
        self.assertEqual(self.store.read_stage(self.pid, 'models')['evidence']['feature_recipe'], choice)
        self.assertEqual(self.store.get(self.pid)['decisions']['features'], choice)

    def test_old_unused_holdout_approval_does_not_transfer_to_new_models(self):
        import copy
        from dclab_rnd.studio import graph
        d = self.dispatcher(budget={'max_tool_calls': 12, 'max_seconds': 3600, 'max_cost_eur': 0.0})
        for task in ('data', 'leakage', 'features', 'models'):
            d.dispatch(self.pid, task)
        graph.approve_gate(self.store, self.pid, 'holdout', by='owner', reason='Old models only')
        solution = copy.deepcopy(self.store.get(self.pid)['solution'])
        solution['prediction_moment'] += ' A new confirmed constraint.'
        self.store.update(self.pid, solution=solution)
        self.store.clear_stages(self.pid)
        graph.approve_gate(self.store, self.pid, 'solution', by='owner', reason='Task source revision ' + d.source_revision(self.pid))
        self.assertEqual(d.adopt_sources(self.pid)['status'], 'ready')
        for task in ('data', 'leakage', 'features', 'models'):
            self.assertEqual(d.dispatch(self.pid, task).get('acceptance_verification'), 'verified')
        self.assertEqual(d.dispatch(self.pid, 'final')['reason'], 'fresh_holdout_approval_required')
        self.assertEqual(self.store.get(self.pid)['holdout_uses'], 0)
        graph.approve_gate(self.store, self.pid, 'holdout', by='owner', reason='New models only')
        self.assertEqual(d.dispatch(self.pid, 'final').get('acceptance_verification'), 'verified')
        self.assertEqual(self.store.get(self.pid)['holdout_uses'], 1)

    def test_checkpoint_reopen_keeps_replan_and_verification_timeline(self):
        from dclab_rnd.orchestra.checkpoints import open_checkpointer
        from dclab_rnd.tasks.dispatch import TaskDispatcher
        d = self.dispatcher()
        with patch.object(d.registry, 'check', side_effect=TimeoutError()):
            d.dispatch(self.pid, 'data')
        d.replan(self.pid, 'data')
        with open_checkpointer(self.home, database_url=self.database_url) as saver:
            resumed = TaskDispatcher(self.store, saver, enabled=True)
            self.assertEqual(resumed.dispatch(self.pid, 'data').get('acceptance_verification'), 'verified')
            state = resumed.state(self.pid)
            self.assertEqual([r['kind'] for r in state['transitions']], ['initialized', 'started', 'no_effect', 'replanned', 'started', 'completed'])
            self.assertEqual(len(state['verifications']), 1)
            self.assertEqual(state['tool_calls'], 2)


    def test_unchanged_verified_work_survives_an_explicit_approval_revision(self):
        from dclab_rnd.studio import graph
        d = self.dispatcher()
        d.dispatch(self.pid, 'data')
        record = self.store.read_stage(self.pid, 'data')
        original = d.state(self.pid)['verifications'][0]
        graph.approve_gate(self.store, self.pid, 'solution', by='owner', reason='Task source revision ' + d.source_revision(self.pid))
        result = d.adopt_sources(self.pid)
        self.assertEqual(result['status'], 'ready', result)
        self.assertEqual(result['preserved'], ['data'])
        self.assertEqual(d.dispatch(self.pid, 'leakage').get('acceptance_verification'), 'verified')
        state = d.state(self.pid)
        self.assertEqual(self.store.read_stage(self.pid, 'data'), record)
        self.assertEqual(state['verifications'][0], original)
        self.assertTrue(any(v.get('reused_from') == original['id'] for v in state['verifications']))
        self.assertEqual(len([a for a in state['attempts'] if a['task'] == 'data']), 1)

    def test_late_owner_completion_is_reconciled_before_any_new_plan(self):
        from dclab_rnd.studio import engine
        from dclab_rnd.studio.lineage import task_attempt
        from dclab_rnd.models.gateway import private_scope
        d = self.dispatcher()
        with patch.object(d, '_invoke', return_value={'error': 'unknown_transport_state'}):
            self.assertEqual(d.dispatch(self.pid, 'data')['status'], 'blocked')
        attempt = d.state(self.pid)['attempts'][0]
        with task_attempt(attempt['id']), private_scope(True):
            engine.execute(self.store, self.pid, 'data')
        result = d.replan(self.pid, 'data')
        self.assertEqual(result['status'], 'reconciled', result)
        self.assertEqual(result['acceptance_verification'], 'verified')
        self.assertEqual(len(d.state(self.pid)['attempts']), 1)
        self.assertEqual(d.state(self.pid)['replans'], [])
        self.assertGreaterEqual(d.state(self.pid)['seconds'], attempt['budget']['reserved_seconds'])


    def test_human_override_cannot_rewrite_measured_decision_options(self):
        from dclab_rnd.studio import engine
        d = self.dispatcher()
        for task in ('data', 'leakage', 'features'):
            self.assertEqual(d.dispatch(self.pid, task)['acceptance_verification'], 'verified')
        record = self.store.read_stage(self.pid, 'features')
        choice = next(o['id'] for o in record['decision']['options'] if o['id'] != record['decision']['chosen'])
        engine.approve(self.store, self.pid, 'features', choice=choice, actor='human')
        record = self.store.read_stage(self.pid, 'features')
        record['decision']['options'][0]['mean'] = 123.456
        self.store.write_stage(self.pid, 'features', record)
        result = d.dispatch(self.pid, 'models')
        self.assertEqual(result['status'], 'blocked', result)
        self.assertEqual(result['reason'], 'owner_record_changed')
        self.assertIsNone(self.store.read_stage(self.pid, 'models'))
        self.assertEqual(self.store.get(self.pid)['holdout_uses'], 0)


class PostgresTaskRecoveryTests(TaskRecoveryIntegrationTests):
    def setUp(self):
        from tests import pgtest
        self.database_url = pgtest.require()
        pgtest.empty(self.database_url)
        super().setUp()

    make_store = fixtures.PostgresTaskDispatchTests.make_store
