"""TP-07: reading task progress never runs or reconciles a task."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dclab_rnd.orchestra.checkpoints import open_checkpointer
from dclab_rnd.studio import ProjectStore, graph
from dclab_rnd.tasks.dispatch import TaskDispatcher
from tests.test_task_dispatch import make_project


class TaskPlanViewTests(unittest.TestCase):
    database_url = ''
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.store = ProjectStore(self.home / 'projects')
        self.pid = make_project(self.store)

    def saver(self):
        return open_checkpointer(self.home, database_url=self.database_url)

    def view(self, enabled=True):
        from dclab_rnd.tasks.view import task_plan_view
        return task_plan_view(self.store, self.home, self.pid, enabled=enabled)

    def test_disabled_and_preview_do_not_create_a_checkpoint_or_run_tools(self):
        before = self.store.get(self.pid)
        with patch.object(TaskDispatcher, 'dispatch', side_effect=AssertionError('write')):
            self.assertEqual(self.view(False)['status'], 'disabled')
            out = self.view()
        self.assertEqual(out['status'], 'preview')
        self.assertFalse(out['execution_enabled'])
        self.assertTrue(out['read_only'])
        self.assertEqual(len([n for n in out['nodes'] if n['kind'] == 'action']), 6)
        self.assertEqual(self.store.get(self.pid), before)
        self.assertFalse((self.home / 'orchestra').exists())

    def test_missing_data_or_confirmation_is_a_blocker_not_a_crash(self):
        self.pid = self.store.create('Empty', 'general', 'A private problem')['id']
        out = self.view()
        self.assertEqual(out['status'], 'blocked')
        self.assertFalse(out['nodes'])
        self.assertFalse((self.home / 'orchestra').exists())

    def test_completed_progress_is_read_without_verifier_or_sqlite_setup_writes(self):
        with self.saver() as saver:
            d = TaskDispatcher(self.store, saver, enabled=True)
            d.dispatch(self.pid, 'data')
            before = d.state(self.pid)
            from langgraph.checkpoint.sqlite import SqliteSaver
            with (patch.object(TaskDispatcher, 'verify', side_effect=AssertionError('write')),
                  patch.object(SqliteSaver, 'setup', side_effect=AssertionError('schema write'))):
                out = self.view()
            self.assertEqual(out['status'], 'recorded')
            self.assertEqual(next(n for n in out['nodes'] if n['id'] == 'data')['status'], 'completed')
            self.assertEqual(d.state(self.pid), before)
            self.assertEqual(out['budget']['tool_calls'], 1)

    def test_changed_source_is_stale_and_private_text_is_not_returned(self):
        with self.saver() as saver:
            TaskDispatcher(self.store, saver, enabled=True).dispatch(self.pid, 'data')
            self.store.update(self.pid, goal='PRIVATE_ROW_OR_SECRET_SENTINEL')
            out = self.view()
            self.assertEqual(out['status'], 'stale')
            self.assertNotIn('PRIVATE_ROW_OR_SECRET_SENTINEL', json.dumps(out))
            self.assertEqual(next(n for n in out['nodes'] if n['id'] == 'data')['status'], 'stale')

    def test_unknown_effect_is_blocked_and_never_reconciled_by_reading(self):
        with self.saver() as saver:
            d = TaskDispatcher(self.store, saver, enabled=True)
            d.dispatch(self.pid, 'data')
            state = d.state(self.pid)
            state['attempts'][0].update(status='unknown', reason='PRIVATE_PROVIDER_MESSAGE')
            d._save(self.pid, state, 'test_unknown')
            out = self.view()
            node = next(n for n in out['nodes'] if n['id'] == 'data')
            self.assertEqual(node['status'], 'blocked')
            self.assertNotIn('PRIVATE_PROVIDER_MESSAGE', json.dumps(out))
            self.assertEqual(d.state(self.pid)['attempts'][0]['status'], 'unknown')

    def test_other_workspace_does_not_read_shared_checkpoint(self):
        from dclab_rnd.tasks.view import task_plan_view
        with self.saver() as saver:
            TaskDispatcher(self.store, saver, enabled=True).dispatch(self.pid, 'data')
            other = ProjectStore(self.home / 'other' / 'projects')
            with self.assertRaises(KeyError):
                task_plan_view(other, self.home, self.pid, enabled=True)

    def test_revision_diff_keeps_valid_work_and_read_does_not_write_proofs(self):
        with self.saver() as saver:
            d = TaskDispatcher(self.store, saver, enabled=True)
            d.dispatch(self.pid, 'data')
            graph.approve_gate(self.store, self.pid, 'solution', by='owner', reason='Task source revision ' + d.source_revision(self.pid))
            self.assertEqual(d.adopt_sources(self.pid)['status'], 'ready')
            before = d.state(self.pid)
            out = self.view()
            self.assertTrue(out['revisions'])
            self.assertIn('data', out['revisions'][0]['preserved_tasks'])
            self.assertEqual(before, d.state(self.pid))

    def test_missing_verification_cannot_be_presented_as_completed(self):
        with self.saver() as saver:
            d = TaskDispatcher(self.store, saver, enabled=True)
            d.dispatch(self.pid, 'data')
            state = d.state(self.pid)
            state['verifications'] = []
            d._save(self.pid, state, 'test_lost_proof')
            out = self.view()
            self.assertEqual(next(n for n in out['nodes'] if n['id'] == 'data')['status'], 'blocked')

    def test_new_valid_attempt_is_preserved_after_an_older_attempt_was_invalidated(self):
        with self.saver() as saver:
            d = TaskDispatcher(self.store, saver, enabled=True)
            self.assertEqual(d.dispatch(self.pid, 'data')['status'], 'completed')
            self.store.update(self.pid, goal='A changed prediction contract')
            self.store.clear_stages(self.pid)
            with patch.object(graph, 'now', return_value='2026-11-12T12:00:01Z'):
                graph.approve_gate(self.store, self.pid, 'solution', by='owner',
                                   reason='Task source revision ' + d.source_revision(self.pid))
            self.assertEqual(d.adopt_sources(self.pid)['status'], 'ready')
            self.assertEqual(d.dispatch(self.pid, 'data')['status'], 'completed')
            with patch.object(graph, 'now', return_value='2026-11-12T12:00:02Z'):
                graph.approve_gate(self.store, self.pid, 'solution', by='owner',
                                   reason='Task source revision ' + d.source_revision(self.pid))
            result = d.adopt_sources(self.pid)
            self.assertEqual(result['status'], 'ready')
            self.assertIn('data', result['preserved'])
            before = d.state(self.pid)
            revisions = self.view()['revisions']
            self.assertEqual([(r['from_revision'], r['to_revision']) for r in revisions], [(1, 2), (2, 3)])
            self.assertNotIn('data', revisions[0]['preserved_tasks'])
            self.assertIn('data', revisions[1]['preserved_tasks'])
            self.assertEqual(before, d.state(self.pid))

    def test_raw_checkpoint_labels_are_not_progress_fields(self):
        with self.saver() as saver:
            d = TaskDispatcher(self.store, saver, enabled=True)
            d.dispatch(self.pid, 'data')
            state = d.state(self.pid)
            a = state['attempts'][0]
            a.update(plan_revision='PRIVATE_SENTINEL', acceptance_verification='PRIVATE_SENTINEL')
            state['replans'] = [{'plan': {'revision': 'PRIVATE_SENTINEL', 'task_id': 'PRIVATE_SENTINEL'}}]
            d._save(self.pid, state, 'test_private_fields')
            self.assertNotIn('PRIVATE_SENTINEL', json.dumps(self.view()))

    def test_lost_upstream_proof_blocks_a_previously_completed_chain(self):
        from dclab_rnd.expansion import runner
        with runner.fast_profile(), self.saver() as saver:
            d = TaskDispatcher(self.store, saver, enabled=True)
            for task in ('data', 'leakage', 'features', 'models'):
                self.assertEqual(d.dispatch(self.pid, task)['status'], 'completed')
            graph.approve_gate(self.store, self.pid, 'holdout', by='owner', reason='One invented final read')
            for task in ('final', 'report'):
                self.assertEqual(d.dispatch(self.pid, task)['status'], 'completed')
            state = d.state(self.pid)
            state['verifications'] = [v for v in state['verifications'] if v['task_id'] != 'data']
            d._save(self.pid, state, 'test_upstream_proof_lost')
            out = self.view()
            for node in out['nodes']:
                if node['kind'] != 'group':
                    self.assertEqual(node['status'], 'blocked', node)
                    self.assertNotEqual(node['verification'], 'verified', node)
            self.assertEqual(self.store.get(self.pid)['holdout_uses'], 1)

    def test_missing_owner_permission_is_not_presented_as_approved(self):
        with self.saver() as saver:
            d = TaskDispatcher(self.store, saver, enabled=True)
            d.dispatch(self.pid, 'data')
            state = d.state(self.pid)
            state['holdout_permission'] = {'fingerprint': 'a' * 64}
            d._save(self.pid, state, 'test_permission_not_in_owner')
            gate = next(n for n in self.view()['nodes'] if n['id'] == 'holdout')
            self.assertEqual(gate['approval'], 'awaiting approval')
            self.assertEqual(gate['status'], 'blocked')

    def test_replacement_diff_does_not_call_the_retried_task_preserved(self):
        with self.saver() as saver:
            d = TaskDispatcher(self.store, saver, enabled=True)
            with patch.object(d.registry, 'check', side_effect=TimeoutError('before-handler')):
                d.dispatch(self.pid, 'data')
            self.assertEqual(d.replan(self.pid, 'data')['status'], 'ready')
            self.assertEqual(d.dispatch(self.pid, 'data')['status'], 'completed')
            revision = next(r for r in self.view()['revisions'] if r['kind'] == 'replacement plan')
            self.assertEqual((revision['from_revision'], revision['to_revision']), (1, 2))
            self.assertNotIn('data', revision['preserved_tasks'])

    def test_recorded_running_state_is_not_reconciled_by_view(self):
        with self.saver() as saver:
            d = TaskDispatcher(self.store, saver, enabled=True)
            d.dispatch(self.pid, 'data')
            state = d.state(self.pid)
            state['attempts'][0]['status'] = 'started'
            d._save(self.pid, state, 'test_running')
            self.store.update(self.pid, running='data')
            out = self.view()
            self.assertEqual(next(n for n in out['nodes'] if n['id'] == 'data')['status'], 'running')
            self.assertEqual(d.state(self.pid)['attempts'][0]['status'], 'started')


class PostgresTaskPlanViewTests(TaskPlanViewTests):
    def setUp(self):
        from tests import pgtest
        from dclab_rnd.storage.postgres import open_stores
        self.database_url = pgtest.require()
        pgtest.empty(self.database_url)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.store = open_stores(self.home, self.database_url)[0]
        self.pid = make_project(self.store)


class TaskPlanApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from dclab_rnd.agentic.server import create_app
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.app = create_app(self.home)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.s = self.app.state.services
        self.pid = make_project(self.s.projects)

    def test_route_default_off_and_enabled_preview_are_read_only(self):
        from dclab_rnd.orchestra.settings import set_settings
        path = f'/api/projects/{self.pid}/task-plan'
        out = self.client.get(path)
        self.assertEqual(out.status_code, 200)
        self.assertEqual(out.json()['status'], 'disabled')
        set_settings(self.home, {'task_plan_preview': True})
        before = self.s.projects.get(self.pid)
        out = self.client.get(path)
        self.assertEqual(out.status_code, 200, out.text)
        self.assertEqual(out.json()['status'], 'preview')
        self.assertEqual(self.s.projects.get(self.pid), before)
        self.assertFalse((self.home / 'orchestra' / 'checkpoints.sqlite3').exists())
        csrf = self.client.get('/api/config').json()['csrf']
        self.assertEqual(self.client.post(path, headers={'X-DCLab-Token': csrf}).status_code, 405)

    def test_unknown_project_is_not_disclosed_even_when_disabled(self):
        self.assertEqual(self.client.get('/api/projects/unknown/task-plan').status_code, 404)

    def test_viewer_can_read_but_cannot_enable_and_anonymous_cannot_read(self):
        from dclab_rnd.accounts.principal import Principal, ANONYMOUS
        path = f'/api/projects/{self.pid}/task-plan'
        csrf = self.client.get('/api/config').json()['csrf']
        with patch('dclab_rnd.accounts.guard.current', return_value=Principal(role='viewer', via='test')):
            self.assertEqual(self.client.get(path).status_code, 200)
            self.assertEqual(self.client.put('/api/team/settings', json={'task_plan_preview': True},
                             headers={'X-DCLab-Token': csrf}).status_code, 403)
        with patch('dclab_rnd.accounts.guard.current', return_value=ANONYMOUS):
            self.assertEqual(self.client.get(path).status_code, 401)

    def test_preview_switch_is_audited_reversible_and_cannot_enable_execution(self):
        csrf = self.client.get('/api/config').json()['csrf']
        h = {'X-DCLab-Token': csrf}
        r = self.client.put('/api/team/settings', json={'task_plan_preview': True}, headers=h)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse(r.json()['settings']['on'])
        self.assertFalse(self.client.get(f'/api/projects/{self.pid}/task-plan').json()['execution_enabled'])
        self.assertIn('task_plan_preview', (self.home / 'audit.jsonl').read_text())
        self.assertEqual(self.client.put('/api/team/settings', json={'unknown_flag': True}, headers=h).status_code, 422)
        self.assertEqual(self.client.put('/api/team/settings', json={'admit_proposals': True}, headers=h).status_code, 422)
        self.assertEqual(self.client.put('/api/team/settings', json={'task_plan_preview': False}, headers=h).status_code, 200)
        self.assertEqual(self.client.get(f'/api/projects/{self.pid}/task-plan').json()['status'], 'disabled')

    def test_controlled_task_execution_rollout_switch_and_rollback(self):
        csrf = self.client.get('/api/config').json()['csrf']
        h = {'X-DCLab-Token': csrf}
        on = self.client.put('/api/team/settings', json={'task_execution': True, 'admit_proposals': True}, headers=h)
        self.assertEqual(on.status_code, 200, on.text)
        self.assertTrue(on.json()['settings']['task_execution'])
        self.assertTrue(on.json()['settings']['admit_proposals'])
        plan = self.client.get(f'/api/projects/{self.pid}/task-plan').json()
        self.assertTrue(plan['execution_enabled'])
        self.assertFalse(plan['developer_preview'])
        self.assertIn('task_planning_v3', plan['release'])
        # Instantaneous rollback via owner switch cascades admit_proposals off cleanly.
        off = self.client.put('/api/team/settings', json={'task_execution': False}, headers=h)
        self.assertEqual(off.status_code, 200, off.text)
        self.assertFalse(off.json()['settings']['task_execution'])
        self.assertFalse(off.json()['settings']['admit_proposals'])
        self.assertEqual(self.client.get(f'/api/projects/{self.pid}/task-plan').json()['status'], 'disabled')

    def test_preview_switch_audit_belongs_to_the_selected_workspace(self):
        import asyncio
        from types import SimpleNamespace
        from dclab_rnd import context, audit, experience
        from dclab_rnd.agentic.api_models import api_routes
        other = self.home / 'other-workspace'
        other.mkdir()
        owner = SimpleNamespace(store=SimpleNamespace(home=other), audit=audit.open_audit(other),
                                experience=experience.open_log(other), workspace_id=None)
        endpoint = next(r.endpoint for r in api_routes(self.app) if r.path == '/api/team/settings' and 'PUT' in r.methods)
        with context.using(owner):
            asyncio.run(endpoint(body={'task_plan_preview': True}))
        self.assertIn('task_plan_preview', (other / 'audit.jsonl').read_text())
        self.assertNotIn('task_plan_preview', (self.home / 'audit.jsonl').read_text())
