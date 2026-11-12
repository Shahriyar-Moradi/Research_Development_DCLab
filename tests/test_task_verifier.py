"""TP-05 verifies owner evidence independently; no live model or sealed benchmark."""
import unittest

from dclab_core.task_verification import training_proof, check_training_proof


class TaskProofTests(unittest.TestCase):
    def test_fold_proof_records_counts_and_actual_split_without_class_values(self):
        proof = training_proof([0, 1, 2, 3], [4, 5], [([0, 1], [2, 3]), ([2, 3], [0, 1])],
                               ["private_a", "private_b", "private_a", "private_b"])
        self.assertEqual(check_training_proof(proof), [])
        self.assertNotIn("private_a", str(proof))
        self.assertNotIn("private_b", str(proof))

    def test_missing_class_or_overlapping_fold_is_a_failed_proof(self):
        proof = training_proof([0, 1, 2, 3], [4], [([0, 2], [1, 3])], [0, 1, 0, 1])
        self.assertTrue(check_training_proof(proof))
        proof = training_proof([0, 1], [1, 2], [([0], [1])], [0, 1])
        self.assertTrue(check_training_proof(proof))

from unittest.mock import patch
from tests import test_task_dispatch as fixtures
from dclab_rnd.studio import graph


class TaskVerifierIntegrationTests(unittest.TestCase):
    database_url = ''
    make_store = fixtures.TaskDispatchTests.make_store
    setUp = fixtures.TaskDispatchTests.setUp
    dispatcher = fixtures.TaskDispatchTests.dispatcher

    def test_real_chain_has_measured_acceptance_and_single_holdout(self):
        d = self.dispatcher()
        for task in ('data', 'leakage', 'features', 'models'):
            result = d.dispatch(self.pid, task)
            self.assertEqual(result.get('acceptance_verification'), 'verified', result)
        graph.approve_gate(self.store, self.pid, 'holdout', by='owner', reason='One final read')
        for task in ('final', 'report'):
            result = d.dispatch(self.pid, task)
            self.assertEqual(result.get('acceptance_verification'), 'verified', result)
        self.assertEqual(self.store.get(self.pid)['holdout_uses'], 1)
        self.assertEqual(d.verify(self.pid, 'final')['status'], 'verified')
        self.assertEqual(d.dispatch(self.pid, 'final')['status'], 'already_completed')

    def test_agent_done_cannot_replace_missing_measured_proof(self):
        from dclab_rnd.studio import engine
        original = engine.stage_data
        def missing(*args):
            evidence, claims, summary = original(*args)
            return evidence, claims, summary
        # Break the measurement at its owning boundary, before a completed record is written.
        with patch('dclab_core.task_verification.training_proof', return_value={}):
            result = self.dispatcher().dispatch(self.pid, 'data')
        self.assertEqual(result.get('acceptance_verification'), 'blocked', result)
        self.assertEqual(self.dispatcher().dispatch(self.pid, 'leakage')['status'], 'blocked')
        self.assertEqual(self.store.get(self.pid)['holdout_uses'], 0)

    def test_changed_goal_is_stale_and_does_not_execute(self):
        d = self.dispatcher()
        d.dispatch(self.pid, 'data')
        self.store.update(self.pid, goal='A different prediction objective')
        self.assertEqual(d.verify(self.pid, 'data')['status'], 'stale')
        self.assertEqual(d.dispatch(self.pid, 'leakage')['status'], 'blocked')
        self.assertIsNone(self.store.read_stage(self.pid, 'leakage'))

    def test_revoked_solution_approval_does_not_accept_old_completion(self):
        d = self.dispatcher()
        d.dispatch(self.pid, 'data')
        self.store.update(self.pid, signoff=None)
        self.assertEqual(d.verify(self.pid, 'data')['status'], 'stale')
        self.assertEqual(self.store.get(self.pid)['holdout_uses'], 0)

    def test_public_verifier_preserves_failed_criteria_and_evidence(self):
        from dclab_core.task_verification import training_proof
        original = training_proof
        def invalid(*args, **kwargs):
            proof = original(*args, **kwargs)
            proof['split_disjoint'] = False
            return proof
        d = self.dispatcher()
        with patch('dclab_core.task_verification.training_proof', side_effect=invalid):
            result = d.dispatch(self.pid, 'data')
        self.assertEqual(result['verification']['status'], 'failed')
        actual = d.verify(self.pid, 'data')
        self.assertEqual(actual['status'], 'failed', actual)
        self.assertTrue(actual['violated_criteria'])
        self.assertTrue(actual['evidence'])

    def test_changed_actual_bytes_return_stale_measured_verdict(self):
        from dclab_rnd.studio import data
        d = self.dispatcher()
        d.dispatch(self.pid, 'data')
        with data.data_path(self.store, self.store.get(self.pid)).open('a') as out:
            out.write('\n')
        result = d.verify(self.pid, 'data')
        self.assertEqual(result['status'], 'stale', result)
        self.assertIn('source_table_changed', result['reasons'])
        self.assertTrue(result['evidence'])

    def test_strong_leakage_blocks_before_feature_handler_but_existing_team_can_ask(self):
        from dclab_rnd.studio import engine
        from dclab_rnd.orchestra.solution import SolutionTeam
        original = engine.stage_leakage
        def strong(*args):
            evidence, claims, summary = original(*args)
            evidence['heuristic_review_candidates'].append({'feature': 'tenure', 'univariate_auc': 1.0,
                                                           'reasons': ['exact_target_proxy']})
            return evidence, claims, summary
        d = self.dispatcher()
        d.dispatch(self.pid, 'data')
        with patch.object(engine, 'stage_leakage', side_effect=strong):
            self.assertEqual(d.dispatch(self.pid, 'leakage').get('acceptance_verification'), 'verified')
        before = len(d.state(self.pid)['attempts'])
        self.assertEqual(d.dispatch(self.pid, 'features')['reason'], 'strong_leakage_requires_current_human_decision')
        self.assertEqual(len(d.state(self.pid)['attempts']), before)
        self.assertIsNone(self.store.read_stage(self.pid, 'features'))
        first = SolutionTeam(self.store, self.saver, home=self.home, task_execution=True).run(self.pid)
        self.assertEqual(first['asks']['gate'], 'audit', first)
        result = SolutionTeam(self.store, self.saver, home=self.home, task_execution=True).run(self.pid, resume=False)
        self.assertEqual(result['asks']['gate'], 'holdout', result)
        self.assertEqual(d.verify(self.pid, 'features')['status'], 'verified')
        self.assertEqual(self.store.get(self.pid)['holdout_uses'], 0)

    def test_related_tables_remain_blocked_without_dispatch(self):
        self.store.update(self.pid, tables={'unreviewed_events': {'name': 'unreviewed_events'}})
        d = self.dispatcher()
        self.assertEqual(d.dispatch(self.pid, 'data')['status'], 'blocked')
        self.assertIsNone(d.state(self.pid))
        self.assertEqual(self.store.get(self.pid)['holdout_uses'], 0)

    def test_missing_threshold_proof_never_authorizes_another_final_read(self):
        from dclab_rnd.studio import engine
        d = self.dispatcher()
        for task in ('data', 'leakage', 'features', 'models'):
            d.dispatch(self.pid, task)
        graph.approve_gate(self.store, self.pid, 'holdout', by='owner', reason='One final read')
        with patch.object(engine, '_task_assessment', return_value=None):
            result = d.dispatch(self.pid, 'final')
        self.assertEqual(result['acceptance_verification'], 'blocked', result)
        self.assertEqual(self.store.get(self.pid)['holdout_uses'], 1)
        self.assertEqual(d.dispatch(self.pid, 'final')['status'], 'blocked')
        self.assertEqual(d.replan(self.pid, 'final')['status'], 'blocked')
        self.assertEqual(self.store.get(self.pid)['holdout_uses'], 1)


    def test_unavailable_optional_proof_keeps_legacy_engine_compatible(self):
        from dclab_rnd.studio import engine
        from dclab_rnd.models.gateway import private_scope
        with private_scope(True), patch('dclab_core.task_verification.training_proof', side_effect=ValueError('unsupported')):
            record = engine.execute(self.store, self.pid, 'data')
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(record['evidence']['training_proof_unavailable'], 'unsupported_training_evidence')
        self.assertEqual(self.store.get(self.pid)['holdout_uses'], 0)

    def test_report_numbers_hidden_outside_the_sections_do_not_prove_completion(self):
        from dclab_rnd.studio import solution_report
        from dclab_core.contracts import ReportSection
        d = self.dispatcher()
        for task in ('data', 'leakage', 'features', 'models'):
            d.dispatch(self.pid, task)
        graph.approve_gate(self.store, self.pid, 'holdout', by='owner', reason='One final read')
        d.dispatch(self.pid, 'final')
        report = solution_report.build(self.store, self.pid, analyse=False)
        report = report.model_copy(update={'sections': [ReportSection(title='Results', body='See the recorded limits.')]})
        with patch.object(solution_report, 'build', return_value=report):
            result = d.dispatch(self.pid, 'report')
        self.assertEqual(result['acceptance_verification'], 'blocked', result)
        self.assertIn('source_report_limits', result['verification']['violated_criteria'])
        self.assertEqual(self.store.get(self.pid)['holdout_uses'], 1)


    def test_independent_verifier_reads_current_solution_approval_itself(self):
        from dclab_core.task_graph import TaskGraph
        from dclab_rnd.tasks.verifier import verify
        d = self.dispatcher()
        d.dispatch(self.pid, 'data')
        state = d.state(self.pid)
        attempt = state['attempts'][0]
        output = d._evidence(self.pid, 'data', attempt)
        self.store.update(self.pid, signoff=None)
        result = verify(self.store, self.pid, TaskGraph.read(state['graph']), attempt, inputs=attempt['inputs'], output=output)
        self.assertEqual(result['status'], 'stale', result)
        self.assertIn('solution_approval_changed', result['reasons'])


    def test_restored_source_returns_current_proof_without_erasing_stale_history(self):
        d = self.dispatcher()
        first = d.dispatch(self.pid, 'data')['verification']
        original_goal = self.store.get(self.pid)['goal']
        self.store.update(self.pid, goal='A temporarily changed objective')
        stale = d.verify(self.pid, 'data')
        self.assertEqual(stale['status'], 'stale')
        self.store.update(self.pid, goal=original_goal)
        current = d.verify(self.pid, 'data')
        self.assertEqual(current['status'], 'verified', current)
        self.assertEqual(current['id'], first['id'])
        self.assertIn(stale, d.state(self.pid)['verifications'])
        self.assertEqual(len(d.state(self.pid)['attempts']), 1)


    def _final_with_mutated_assessment(self, mutate):
        from dclab_rnd.studio import engine
        d = self.dispatcher()
        for task in ('data', 'leakage', 'features', 'models'):
            self.assertEqual(d.dispatch(self.pid, task)['acceptance_verification'], 'verified')
        graph.approve_gate(self.store, self.pid, 'holdout', by='owner', reason='One final read')
        original = engine._task_assessment
        def altered(*args, **kwargs):
            return mutate(original(*args, **kwargs))
        with patch.object(engine, '_task_assessment', side_effect=altered):
            result = d.dispatch(self.pid, 'final')
        self.assertEqual(self.store.get(self.pid)['holdout_uses'], 1)
        self.assertIn(result['acceptance_verification'], ('blocked', 'failed'), result)
        self.assertIn('source_threshold', result['verification']['violated_criteria'])
        self.assertEqual(d.dispatch(self.pid, 'final')['status'], 'blocked')
        self.assertEqual(self.store.get(self.pid)['holdout_uses'], 1)
        return result

    def test_oof_metadata_without_actual_threshold_cannot_complete(self):
        result = self._final_with_mutated_assessment(lambda out: {'threshold_proof': out['threshold_proof']})
        self.assertEqual(result['acceptance_verification'], 'blocked')

    def test_threshold_value_must_match_its_oof_selection_proof(self):
        def mutate(out):
            out['threshold']['value'] += 0.25
            return out
        self._final_with_mutated_assessment(mutate)

    def test_threshold_policy_is_recomputed_from_current_saved_decisions(self):
        from dclab_core.task_fingerprints import metadata_fingerprint
        def mutate(out):
            out['threshold_policy']['inputs']['budget'] = 0.4
            out['threshold_proof']['policy_fingerprint'] = metadata_fingerprint(out['threshold_policy']['inputs'])
            return out
        self._final_with_mutated_assessment(mutate)


class PostgresTaskVerifierTests(TaskVerifierIntegrationTests):
    def setUp(self):
        from tests import pgtest
        self.database_url = pgtest.require()
        pgtest.empty(self.database_url)
        super().setUp()

    make_store = fixtures.PostgresTaskDispatchTests.make_store
