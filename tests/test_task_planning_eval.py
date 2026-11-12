"""TP-06 frozen offline evaluation: no live models or sealed benchmark access."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dclab_rnd.agent_eval.task_planning.gate import CASES, GATE, SEEDS, judge


def measured_rows(arm):
    return [dict(id=f'{case.id}:{seed}', case_id=case.id, seed=seed, stratum=case.stratum,
                 phase=case.phase, scenario=case.scenario, expected=case.expected, correct=1, completed=int(case.expected == 'complete'), recovered=int(case.scenario in ('transient', 'resume')),
                 judged=True, instrumented=True, source_fingerprint=hashlib.sha256(f'{case.id}:{seed}'.encode()).hexdigest(),
                 safety={name: 0 for name in GATE['critical_metrics']}, seconds=1.0, cost_eur=0.0,
                 extra_prompts=0, owner_evidence_checked=True)
            for case in CASES for seed in SEEDS]


def valid_probe():
    return {'mode': 'scripted_proposal_probe', 'decomposition_status': 'proposed', 'planning_status': 'proposed',
            'dispatch_status': 'blocked', 'attempts': 0, 'owner_records': 0, 'holdout_uses': 0,
            'provider_requests': 0, 'billed_eur': 0.0, 'scripted_requests': 2,
            'exchange_capture_passed': True, 'tool_schemas_offered': False,
            'api_thinking_summaries': [{'label': 'API thinking summary', 'source': 'scripted fixture', 'text': ['Scripted summary.']} for _ in range(2)]}


class FrozenGateTests(unittest.TestCase):
    def test_required_strata_and_numerical_gate_are_frozen(self):
        self.assertEqual(len(CASES), 16)
        self.assertEqual(len({c.stratum for c in CASES}), 16)
        self.assertEqual(len(SEEDS), 3)
        self.assertEqual(GATE['min_matched_cases'], 48)
        self.assertGreater(GATE['min_gain'], 0)
        with self.assertRaises(TypeError):
            GATE['min_gain'] = 0

    def test_missing_model_arm_is_insufficient_not_a_pass(self):
        scores = {'A': measured_rows('A'), 'B': measured_rows('B'), 'C': []}
        result = judge(scores, configured_model=False, c_integrated=False)
        self.assertEqual(result['verdict'], 'INSUFFICIENT-EVIDENCE')
        self.assertFalse(result['promoted'])
        self.assertIsNone(result['resource_checks']['cost_and_prompts'])

    def test_critical_failure_wins_over_missing_model_evidence(self):
        scores = {'A': measured_rows('A'), 'B': measured_rows('B'), 'C': []}
        scores['B'][0]['safety']['holdout_violations'] = 1
        result = judge(scores, configured_model=False, c_integrated=False)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(result['critical']['B']['holdout_violations'], 1)

    def test_missing_safety_measurement_is_never_zero(self):
        scores = {a: measured_rows(a) for a in ('A', 'B', 'C')}
        scores['C'][0]['safety'].pop('false_completion')
        result = judge(scores, configured_model=True, c_integrated=True)
        self.assertEqual(result['verdict'], 'INSUFFICIENT-EVIDENCE')
        self.assertFalse(result['promoted'])

    def test_a_measured_workflow_regression_fails_even_if_c_is_missing(self):
        scores = {'A': measured_rows('A'), 'B': measured_rows('B'), 'C': []}
        scores['B'][0]['correct'] = 0
        self.assertEqual(judge(scores, configured_model=False, c_integrated=False)['verdict'], 'FAIL')

    def test_no_gain_cannot_pass_even_with_a_complete_model_arm(self):
        scores = {a: measured_rows(a) for a in ('A', 'B', 'C')}
        result = judge(scores, configured_model=True, c_integrated=True)
        self.assertNotEqual(result['verdict'], 'PASS')
        self.assertFalse(result['promoted'])

    def test_seed_repeats_are_clustered_by_case_for_uncertainty(self):
        from dclab_rnd.agent_eval.task_planning.gate import paired_metrics
        a, b = measured_rows('A'), measured_rows('B')
        out = paired_metrics(a, b, 'correct')
        self.assertEqual(out['n'], len(CASES))
        self.assertEqual(out['interval'], [0.0, 0.0])

    def test_duplicate_or_mismatched_sources_cannot_promote(self):
        scores = {a: measured_rows(a) for a in ('A', 'B', 'C')}
        scores['B'].append(copy.deepcopy(scores['B'][0]))
        result = judge(scores, configured_model=True, c_integrated=True)
        self.assertFalse(result['promoted'])
        self.assertIn('duplicate_case_seed', result['reasons'])

    def test_unjudged_run_cannot_hide_a_measured_critical_failure(self):
        scores = {'A': measured_rows('A'), 'B': measured_rows('B'), 'C': []}
        scores['B'][0]['judged'] = False
        scores['B'][0]['safety']['false_completion'] = 1
        self.assertEqual(judge(scores, configured_model=False, c_integrated=False)['verdict'], 'FAIL')


    def test_measured_c_regression_fails_even_without_live_evidence(self):
        scores = {arm: measured_rows(arm) for arm in ('A', 'B', 'C')}
        scores['C'][0]['correct'] = 0
        result = judge(scores, configured_model=False, c_integrated=False)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('workflow_or_route_regression', result['reasons'])

    def test_seed_identity_cannot_inflate_independent_groups(self):
        scores = {arm: measured_rows(arm) for arm in ('A', 'B', 'C')}
        for arm, rows in scores.items():
            for row in rows:
                row['case_id'] = row['id']
                row['expected'] = 'complete'
                row['completed'] = int(arm == 'C')
        result = judge(scores, configured_model=True, c_integrated=True)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('invalid_frozen_row_identity', result['reasons'])
        self.assertFalse(result['promoted'])

    def test_pairing_drops_unmatched_seeds_before_grouping(self):
        from dclab_rnd.agent_eval.task_planning.gate import paired_metrics
        a, b = measured_rows('A')[:3], measured_rows('B')[:3]
        a[0]['correct'] = 0
        b[0]['judged'] = False
        result = paired_metrics(a, b, 'correct')
        self.assertEqual(result['mean_a'], 1.0)
        self.assertEqual(result['difference'], 0.0)
        self.assertEqual(result['matched_case_seed_rows'], 2)


    def test_probe_effects_or_lost_summary_cannot_hide_behind_missing_c(self):
        scores = {'A': measured_rows('A'), 'B': measured_rows('B'), 'C': []}
        bad = {**valid_probe(), 'dispatch_status': 'completed', 'attempts': 1, 'owner_records': 1,
               'holdout_uses': 1, 'exchange_capture_passed': False}
        result = judge(scores, configured_model=False, c_integrated=False, proposal_probe=bad)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(result['critical']['C']['holdout_violations'], 1)
        self.assertFalse(result['proposal_probe_validation']['passed'])


class MeasuredRunnerTests(unittest.TestCase):
    def test_probe_ignores_inherited_openrouter_settings(self):
        from unittest.mock import patch
        from dclab_rnd.agent_eval.task_planning.probe import run_probe
        from dclab_rnd.agent_eval.task_planning.gate import validate_probe
        with patch.dict('os.environ', {'DCLAB_LLM_PROVIDER': 'openrouter', 'OPENROUTER_API_KEY': 'scripted-only',
                                     'OPENROUTER_MODEL': 'qwen/qwen3.8-flash'}):
            result = run_probe()
        self.assertTrue(validate_probe(result)['passed'], result)

    def test_probe_does_not_write_to_an_installed_workspace_ledger(self):
        from dclab_rnd import ledger
        from dclab_rnd.agent_eval.task_planning.probe import run_probe
        previous = ledger.current_ledger()
        with tempfile.TemporaryDirectory() as temporary:
            external = ledger.FileLedger(Path(temporary), 'external-workspace')
            ledger.install(external)
            try:
                run_probe()
                self.assertEqual(list(external.records()), [])
                self.assertIs(ledger.current_ledger(), external)
            finally:
                ledger.install(previous)

    def test_embedded_evaluation_is_refused_before_workspace_effects(self):
        from types import SimpleNamespace
        from dclab_rnd import context
        from dclab_rnd.agent_eval.task_planning.probe import run_probe
        with context.using(SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, 'standalone'):
                run_probe()

    def test_b_uses_real_owner_records_and_one_holdout_read(self):
        from dclab_rnd.agent_eval.task_planning.runner import run_case
        from dclab_rnd.agent_eval.task_planning.probe import offline_environment
        with offline_environment():
            row = run_case(CASES[0], SEEDS[0], 'B')
        self.assertEqual(row['correct'], 1, row)
        self.assertEqual(row['completed'], 1)
        self.assertEqual(row['holdout_uses'], 1)
        self.assertEqual(row['handler_calls']['final'], 1)
        self.assertEqual(sum(row['safety'].values()), 0)

    def test_a_resume_preserves_the_completed_data_effect(self):
        from dclab_rnd.agent_eval.task_planning.runner import run_case
        from dclab_rnd.agent_eval.task_planning.probe import offline_environment
        with offline_environment():
            row = run_case(CASES[-1], SEEDS[0], 'A')
        self.assertEqual(row['correct'], 1, row)
        self.assertEqual(row['handler_calls']['data'], 1)
        self.assertEqual(row['holdout_uses'], 1)
        self.assertEqual(sum(row['safety'].values()), 0)

    def test_related_tables_block_b_before_any_handler(self):
        from dclab_rnd.agent_eval.task_planning.runner import run_case
        from dclab_rnd.agent_eval.task_planning.probe import offline_environment
        with offline_environment():
            row = run_case(CASES[5], SEEDS[0], 'B')
        self.assertEqual(row['status'], 'blocked', row)
        self.assertEqual(row['tool_calls'], 0)
        self.assertEqual(row['holdout_uses'], 0)

    def test_c_is_unjudged_and_cannot_be_reported_as_a_completed_run(self):
        from dclab_rnd.agent_eval.task_planning.runner import run_case
        row = run_case(CASES[0], SEEDS[0], 'C')
        self.assertFalse(row['judged'])
        self.assertIsNone(row['completed'])
        self.assertFalse(row['instrumented'])

    def test_scripted_proposals_keep_their_summaries_but_do_not_execute_c(self):
        from dclab_rnd.agent_eval.task_planning.probe import run_probe
        result = run_probe()
        self.assertEqual(result['decomposition_status'], 'proposed', result)
        self.assertEqual(result['planning_status'], 'proposed', result)
        self.assertEqual(result['dispatch_status'], 'blocked')
        self.assertEqual(result['attempts'], 0)
        self.assertEqual(result['owner_records'], 0)
        self.assertEqual(result['provider_requests'], 0)
        self.assertTrue(result['exchange_capture_passed'], result)
        self.assertTrue(all(s['text'] and 'scripted' in s['source'] for s in result['api_thinking_summaries']))
        self.assertFalse(result['tool_schemas_offered'])
        self.assertTrue(all(s['label'] == 'API thinking summary' for s in result['api_thinking_summaries']))


class EvaluationRecordTests(unittest.TestCase):
    def test_freeze_precedes_append_only_results_and_insufficient_is_not_green(self):
        from dclab_rnd.agent_eval.task_planning import run, FREEZE_NAME
        from dclab_rnd.agent_eval.task_planning.runner import run_case
        from dclab_core.bench import verdict_of
        indexed = {arm: {r['id']: r for r in measured_rows(arm)} for arm in ('A', 'B')}
        def fake(case, seed, arm):
            return run_case(case, seed, arm) if arm == 'C' else indexed[arm][f'{case.id}:{seed}']
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / 'task_planning_v1'
            first = run(campaign=campaign, runner=fake, proposal_probe=valid_probe)
            original = first.read_bytes()
            second = run(campaign=campaign, runner=fake, proposal_probe=valid_probe)
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), original)
            doc = json.loads(original)
            frozen = json.loads((campaign / FREEZE_NAME).read_text())
            self.assertLess(frozen['at'], doc['at'])
            self.assertEqual(verdict_of(doc)[0], None)
            self.assertTrue(doc['evaluation_only'])
            self.assertFalse(doc['promoted'])
            from dclab_rnd.bench import evidence_rows
            rows, unassigned = evidence_rows(Path(temporary))
            self.assertEqual(unassigned, [])
            self.assertTrue(rows)
            self.assertTrue(all(r.passed is None for r in rows if r.metric == 'verdict'))
            # Repeated seeds are observations, not 48 independent Bernoulli trials.
            self.assertFalse(any(r.n == len(CASES) * len(SEEDS) for r in rows))
            self.assertEqual(doc['summary']['A_authored_cases'], len(CASES))

    def test_current_commit_is_recorded_separately_from_freeze_commit(self):
        from unittest.mock import patch
        from dclab_rnd.agent_eval.task_planning import freeze, run
        from dclab_rnd.agent_eval.task_planning.runner import run_case
        indexed = {arm: {r['id']: r for r in measured_rows(arm)} for arm in ('A', 'B')}
        def fake(case, seed, arm):
            return run_case(case, seed, arm) if arm == 'C' else indexed[arm][f'{case.id}:{seed}']
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            with patch('dclab_rnd.bench._git', side_effect=lambda *args: 'initial' if args[0] == 'rev-parse' else ''):
                frozen = freeze(campaign)
            with patch('dclab_rnd.bench._git', side_effect=lambda *args: 'later' if args[0] == 'rev-parse' else 'changed'):
                doc = json.loads(run(campaign=campaign, runner=fake, proposal_probe=valid_probe).read_text())
            self.assertEqual(doc['freeze_commit'], 'initial')
            self.assertEqual(doc['commit'], 'later')
            self.assertTrue(doc['dirty'])
            owners = frozen['definitions']['code_sha256']
            for name in ('dclab_rnd/orchestra/solution.py', 'dclab_rnd/expansion/runner.py', 'dclab_rnd/expansion/datasets.py'):
                self.assertIn(name, owners)

    def test_changed_frozen_definitions_require_a_new_version(self):
        from dclab_rnd.agent_eval.task_planning import freeze, FREEZE_NAME
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            freeze(campaign)
            path = campaign / FREEZE_NAME
            changed = json.loads(path.read_text())
            changed['definitions']['gate_sha256'] = 'changed'
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'frozen_definitions_changed'):
                freeze(campaign)


class TrainingExclusionTests(unittest.TestCase):
    def test_frozen_dev_text_and_identifier_cannot_enter_llm_training(self):
        from dclab_rnd.store.sets.llm import Contamination
        guard = Contamination(set(), set())
        self.assertEqual(guard.why([CASES[0].text]), 'evaluation-only task words')
        self.assertEqual(guard.why([CASES[0].id]), 'evaluation-only task words')

    def test_evaluation_campaign_is_kept_with_nontraining_consent(self):
        from types import SimpleNamespace
        from dclab_rnd.store.keepers import campaigns
        from dclab_rnd.store.sets import allowed
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / 'task_planning_v1'
            (base / 'results').mkdir(parents=True)
            (base / 'manifest.json').write_text(json.dumps({'evaluation_only': True}))
            (base / 'results' / 'TPB-001_dev_offline.json').write_text(json.dumps({'evaluation_only': True, 'case': CASES[0].id}))
            sources = list(campaigns.sources({'campaigns': temporary}))
            self.assertTrue(sources)
            for source in sources:
                self.assertEqual(source.consent, 'operate')
                self.assertFalse(allowed(SimpleNamespace(provenance=SimpleNamespace(consent=source.consent)), 'm4_workflow_llm', {}))

    def test_ordinary_campaign_training_consent_is_unchanged(self):
        from dclab_rnd.store.keepers import campaigns
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / 'ordinary_dev' / 'results'
            base.mkdir(parents=True)
            (base / 'RUN-001_dev.json').write_text(json.dumps({'split': 'dev', 'value': 1}))
            self.assertEqual(list(campaigns.sources({'campaigns': temporary}))[0].consent, 'train')
