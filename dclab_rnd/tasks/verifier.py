"""Independent deterministic acceptance checks. No tools, models, fitting or table rows."""
from __future__ import annotations

import json
import math
from dclab_core.task_execution_plan import TaskExecutionPlan
from dclab_core.task_fingerprints import metadata_fingerprint
from dclab_core.task_verification import check_training_proof, verdict


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _threshold_selection_valid(assessment, project, training_proof):
    """Bind an actual training-selected operating point and its policy; never recompute a holdout score."""
    from dclab_core.assess import threshold_inputs
    try:
        artifact = assessment['threshold']
        proof = assessment['threshold_proof']
        policy = assessment['threshold_policy']
        n = training_proof['train_rows']
        span = policy['training_span_days']
        solution = project.get('solution') or {}
        if span is not None and (not _finite(span) or span < 0 or not solution.get('time_column')):
            return False
        resolved = threshold_inputs(project.get('plan'), n, span)
        resolved['partial_labels'] = resolved['partial_labels'] or solution.get('labels') == 'partial'
        if resolved['budget'] is None and solution.get('alert_budget'):
            resolved['budget'] = solution['alert_budget']
        selection = {key: artifact.get(key) for key in ('value', 'chosen_on', 'chosen_by', 'training')}
        counts = artifact['training']
        tp, fp, fn, alerts = (counts[key] for key in ('true_positives', 'false_positives', 'false_negatives', 'alerts'))
        if (len(training_proof['class_counts']) != 2 or not all(type(c) is int and 0 <= c <= n for c in (tp, fp, fn, alerts))
                or tp + fp != alerts or tp + fn != training_proof['class_counts'][1]
                or fp > training_proof['class_counts'][0] or not _finite(counts.get('alert_share'))
                or abs(counts['alert_share'] - alerts / n) > 0.0001):
            return False
        return (_finite(artifact.get('value')) and artifact.get('chosen_on') == 'out-of-fold training predictions'
                and isinstance(artifact.get('chosen_by'), str) and bool(artifact['chosen_by'].strip())
                and proof.get('threshold_fingerprint') == metadata_fingerprint(selection)
                and policy['inputs'] == resolved and proof.get('policy_fingerprint') == metadata_fingerprint(resolved))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def verify(projects, pid, graph, attempt, *, inputs, output, saver=None, current_graph=None):
    from dclab_rnd.studio import data, solution_report
    from dclab_core import report as core_report
    from dclab_core.contracts import SolutionReport
    from dclab_core.holdout import read_receipt
    from dclab_core.lineage import RunRecord
    from dclab_rnd.storage.files import files_for

    plan = TaskExecutionPlan.read(attempt['plan'])
    node = next(n for n in graph.nodes if n.id == attempt['task'])
    checks, issues = {}, []

    def check(ident, condition, reason, *, missing=False):
        checks[ident] = {'status': 'verified' if condition else 'blocked' if missing else 'failed',
                         'reason': 'measured' if condition else reason}

    project = projects.get(pid)
    from dclab_rnd.studio.graph import solution_signed, solution_hash
    from dclab_rnd.tasks.adapter import _execution_context
    current_graph = current_graph or graph
    signoff = {key: (project.get('signoff') or {}).get(key) for key in ('by', 'at', 'solution_hash')}
    approval = next((r for r in current_graph.source_refs if r.kind == 'approval'), None)
    if not solution_signed(project) or approval is None or approval.fingerprint != metadata_fingerprint(signoff):
        issues.append(('stale', 'solution_approval_changed'))
    context, context_problems = _execution_context(project, project.get('data') or {})
    reference = next((r for r in current_graph.source_refs if r.id == 'execution_context'), None)
    if (context_problems or reference is None or reference.fingerprint != metadata_fingerprint(context)
            or current_graph.solution_fingerprint != solution_hash(project.get('solution'))
            or not any(r.kind == 'dataset' and r.fingerprint == (project.get('data') or {}).get('sha256')
                       for r in current_graph.source_refs)):
        issues.append(('stale', 'source_context_changed'))
    try:
        data.data_path(projects, project)
    except Exception:
        issues.append(('stale', 'source_table_changed'))
    if attempt['inputs'] != inputs:
        issues.append(('stale', 'original_inputs_changed'))
    if {c.id for c in plan.acceptance_checks} != {c.id for c in node.acceptance_checks}:
        issues.append(('failed', 'plan_criteria_changed'))
    task = attempt['task']
    if task == 'report':
        rows = [r['payload'] for r in projects.activity(pid, limit=1000)
                if r['kind'] == 'task_exported' and r['payload'].get('attempt') == attempt['id']]
        check('report_lineage', len(rows) == 1 and rows[0].get('source') == metadata_fingerprint(projects.records(pid)),
              'report_sources_changed')
        sound = False
        try:
            files = files_for(projects)
            artifact, = [a for a in rows[0]['files'] if a['key'].endswith('/solution_report.json')]
            report = SolutionReport.read(json.loads(files.path(artifact['key'], expected=artifact['sha256']).read_text()))
            context = solution_report.inputs(projects, pid, analyse=False)
            metric = projects.read_stage(pid, 'final')['primary_metric']
            required = {'final.evidence.holdout_metrics.' + metric,
                        'final.evidence.holdout_primary_metric_ci.low', 'final.evidence.holdout_primary_metric_ci.high'}
            displayed = {path for _, token, paths in core_report._tokens_with_paths(report) for path in paths
                         if path in report.numbers and core_report.matches(token, report.numbers[path])}
            sound = bool(required <= displayed and report.limits and report.sections and required <= set(report.numbers)
                         and report == core_report.build(**context) and not core_report.check(report, core_report.context(**context)))
        except (ValueError, KeyError, TypeError, OSError, IndexError):
            pass
        check('source_report_limits', sound, 'report_limits_or_number_lineage_missing', missing=not sound)
    else:
        record = projects.read_stage(pid, task) or {}
        evidence = record.get('evidence') or {}
        try:
            run = RunRecord.read(record['run'])
            lineage = (run.parent == attempt['id'] and run.project == pid and run.name == task
                       and run.status == 'completed' and metadata_fingerprint(record) == output['fingerprint']
                       and record['provenance']['data_sha256'] == project['data']['sha256'])
        except (ValueError, KeyError, TypeError):
            lineage = False
        check(task + '_lineage', lineage, 'owner_lineage_mismatch')
        proof = evidence.get('training_proof') or {}
        problems = check_training_proof(proof)
        scope = evidence.get('training_scope') or {}
        scope_valid = (scope.get('cv_fingerprint') == metadata_fingerprint(proof)
                       and scope.get('fit_scope') == 'training_folds' and scope.get('selection_scope') == 'training_validation')
        if task == 'data':
            check('source_class_balance', not problems, problems[0] if problems else 'class_balance_unknown', missing=not proof)
            check('source_lock_holdout', not problems and evidence.get('split_fingerprint') == proof.get('split_fingerprint'),
                  'locked_split_not_proven', missing=not proof)
            check('source_train_only_analysis', not problems and scope_valid, 'training_scope_not_proven', missing=not proof or not scope)
        else:
            original = (projects.read_stage(pid, 'data') or {}).get('evidence', {}).get('training_proof')
            if problems or proof != original or not scope_valid:
                issues.append(('blocked' if not proof else 'failed', 'training_partition_changed_or_invalid'))
        if task == 'leakage':
            good = (evidence.get('detector_canary_passed') is True and bool(evidence.get('decision_time_contract'))
                    and set(evidence.get('declared_leakage_features', [])) == set(scope.get('excluded_columns', []))
                    and set(evidence.get('identifier_exclusions', {})) == set(scope.get('identifier_columns', []))
                    and not project.get('tables'))
            check('source_leakage_audit', good, 'leakage_contract_or_related_table_blocked')
        if task in ('features', 'models', 'final'):
            problem = science_readiness(projects, pid, saver)
            if problem:
                issues.append(('blocked', problem))
        if task in ('features', 'models'):
            rows = evidence.get('stage_results' if task == 'features' else 'model_results', [])
            metric = record.get('primary_metric')
            good = bool(rows and evidence.get('selection_rule'))
            for row in rows:
                folds = row.get('fold_metrics', [])
                good = good and not row.get('include_blocked') and len(folds) == len(proof.get('folds', []))
                good = good and all(_finite(f.get(metric)) for f in folds)
            decision = record.get('decision') or {}
            good = good and decision.get('chosen') in {r.get('id') for r in decision.get('options', [])}
            check('engine_features' if task == 'features' else 'source_choose_on_validation', good,
                  'selection_not_measured_on_training_folds')
        if task == 'final':
            receipt = read_receipt(projects.data_dir(pid) / 'holdout_receipt.json')
            once = (receipt is not None and receipt.consumed and receipt.project_id == pid and receipt.stage == 'final'
                    and receipt.holdout_uses == 1 and not receipt.reuse_reason and project.get('holdout_uses') == 1
                    and receipt.split_fingerprint == proof.get('split_fingerprint')
                    and evidence.get('holdout_receipt_id') == receipt.receipt_id)
            check('source_final_once', once, 'holdout_once_or_actual_split_not_proven')
            assessment = evidence.get('task_assessment') or {}
            threshold = assessment.get('threshold_proof') or {}
            n = proof.get('train_rows')
            good = (threshold.get('scope') == 'out_of_fold_training' and threshold.get('cv_fingerprint') == metadata_fingerprint(proof)
                    and threshold.get('count') == n and threshold.get('unique') == n and threshold.get('complete') is True
                    and threshold.get('in_range') is True and threshold.get('finite_scores') is True
                    and threshold.get('configuration_fingerprint') == metadata_fingerprint(evidence.get('selected_optimization'))
                    and _threshold_selection_valid(assessment, project, proof))
            check('source_threshold', good, 'training_oof_threshold_not_proven',
                  missing=not threshold or not assessment.get('threshold') or not assessment.get('threshold_policy'))
            metric = (evidence.get('holdout_metrics') or {}).get(record.get('primary_metric'))
            ci = evidence.get('holdout_primary_metric_ci') or {}
            if not (_finite(metric) and _finite(ci.get('low')) and _finite(ci.get('high')) and ci['low'] <= metric <= ci['high']):
                issues.append(('blocked', 'final_metric_or_uncertainty_missing'))
    return verdict(graph, plan, inputs=inputs, outputs={node.produces[0]: output['fingerprint']},
                   evidence=[output['reference']], checks=checks, issues=issues)


def science_readiness(projects, pid, saver):
    """The leakage stage may finish before the team audit. Selection must wait for unresolved strong flags."""
    from dclab_rnd.studio import leak_audit
    project = projects.get(pid)  # authorization before shared saver access
    evidence = (projects.read_stage(pid, 'leakage') or {}).get('evidence') or {}
    forbidden = {f.get('column') for f in (project.get('solution') or {}).get('forbidden', [])}
    suspects = {f.get('feature') or f.get('column') for f in evidence.get('heuristic_review_candidates', [])
                if set(f.get('reasons', [])) & set(leak_audit.STRONG)
                and 'declared_post_outcome_or_contested' not in f.get('reasons', [])} - forbidden
    last = ((project.get('auditor') or {}).get('last') or {}).get('verdict') or {}
    if last.get('inputs') == leak_audit.single_inputs(project) and last.get('verdict') == 'not_screened':
        return 'leakage_screen_not_measured'
    if not suspects:
        return None
    checkpoint = saver.get({'configurable': {'thread_id': 'project:' + pid, 'checkpoint_ns': ''}}) if saver else None
    state = (checkpoint or {}).get('channel_values') or {}
    if state.get('project_id') != pid or state.get('audited_for') != leak_audit.single_inputs(project):
        return 'strong_leakage_requires_current_human_decision'
    audit, kept = None, False
    for step in state.get('steps', []):
        result = step.get('result') or {}
        if step.get('action') == 'audit':
            kept = False
            audit = result if result.get('inputs') == leak_audit.single_inputs(project) and result == last else None
        if (step.get('action') == 'audit_gate' and audit and audit.get('verdict') == 'send_back'
                and suspects <= set(result.get('kept', []))):
            # The existing team writes this step only after the human pause returns; a tool cannot forge it.
            kept = True
    return None if kept else 'strong_leakage_requires_current_human_decision'
