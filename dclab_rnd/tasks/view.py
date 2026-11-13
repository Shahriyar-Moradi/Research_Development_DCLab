"""TP-07: allowlisted, read-only task progress. No execution, reconciliation or model call."""
from __future__ import annotations

from contextlib import closing, contextmanager
from pathlib import Path
import re
import sqlite3

TITLES = {'root': 'Solution workflow', 'preparation': 'Prepare and check', 'selection': 'Select on training data',
          'evaluation': 'Approve and evaluate', 'reporting': 'Report', 'data': 'Profile training data',
          'leakage': 'Check leakage', 'features': 'Choose features', 'models': 'Choose a model',
          'holdout': 'Owner approval', 'final': 'One final evaluation', 'report': 'Export the report'}
MESSAGES = {'disabled': 'Task Plan developer preview is off. The owner can enable the read-only preview in Admin → Teams.',
            'blocked': 'Attach data and confirm a supported solution to preview its task plan.',
            'preview': 'Proposed tasks only. No task execution has been recorded.',
            'recorded': 'Recorded task progress. Reading this panel does not run or reconcile work.',
            'stale': 'Inputs, permissions or runtime have changed. Recorded completion is historical.',
            'unavailable': 'Task progress could not be read safely. Existing workflow records are unchanged.'}


@contextmanager
def readonly_checkpoint(home, projects):
    """Never create checkpoint storage, migrate it, or run saver.setup()."""
    from dclab_rnd.orchestra.checkpoints import where
    # PostgreSQL stores expose their engine rather than a public URL.
    engine = getattr(projects, 'engine', None)
    if engine is not None:
        import psycopg
        from psycopg.rows import dict_row
        from langgraph.checkpoint.postgres import PostgresSaver
        from dclab_rnd.storage.notify import dsn
        address = engine.url.render_as_string(hide_password=False)
        with psycopg.connect(dsn(address), autocommit=True, row_factory=dict_row,
                             options='-c default_transaction_read_only=on') as conn:
            if conn.execute("SELECT to_regclass('checkpoints') AS present").fetchone()['present'] is None:
                yield None
            else:
                yield PostgresSaver(conn)
        return
    place = where(Path(home), database_url='')
    path = Path(place['path'])
    if not path.is_file():
        yield None
        return
    from langgraph.checkpoint.sqlite import SqliteSaver
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        conn.execute('PRAGMA query_only=ON')
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='checkpoints'").fetchone():
            yield None
            return
        saver = SqliteSaver(conn)
        saver.is_setup = True
        saver._has_task_path = any(row[1] == 'task_path' for row in conn.execute('PRAGMA table_info(writes)'))
        saver.setup = lambda: None
        yield saver


def _number(value, default=0):
    import math
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else default


def _reason(attempt):
    if attempt.get('status') in ('started', 'unknown'):
        return 'The effect needs owner reconciliation before another attempt.'
    if attempt.get('failure_category') == 'transient':
        return 'A transient failure stopped before the handler. A validated replacement plan is required.'
    if attempt.get('reason') in ('guard_refused', 'fresh_holdout_approval_required'):
        return 'The existing guard requires permission or owner approval.'
    return 'This task is stopped. Inspect its recorded evidence before continuing.'


def _project_graph(project, caps):
    from dclab_core.card import problem_card
    from dclab_core.solution_workflow import solution_workflow
    from dclab_rnd.core_adapter import brief_from_project, plan_from_project
    from dclab_rnd.tasks.adapter import adapt_workflow
    workflow = solution_workflow(problem_card(brief_from_project(project)))
    return adapt_workflow(workflow, plan_from_project(project), project, capabilities=caps)[0]


def task_plan_view(projects, home, project_id, *, enabled=False, execution_enabled=False):
    project = projects.get(project_id)  # authenticate the workspace before touching a shared saver
    exec_on = bool(execution_enabled)
    out = {'project_id': project_id, 'read_only': True, 'execution_enabled': exec_on, 'developer_preview': not exec_on,
           'status': 'disabled', 'nodes': [], 'revisions': [], 'budget': None,
           'release': ('Opt-in task execution is enabled in Admin → Teams (task_planning_v3 tp06.gate.v2 PASS).' if exec_on
                       else 'Task execution passed task_planning_v3 (tp06.gate.v2 PASS); off by default until enabled by the owner in Admin → Teams.')}
    def finish(status):
        return {**out, 'status': status, 'message': MESSAGES[status]}
    if not enabled:
        return finish('disabled')
    try:
        from dclab_core.task_graph import TaskGraph
        from dclab_core.task_fingerprints import graph_fingerprint
        from dclab_rnd.tasks.adapter import capability_snapshot
        from dclab_rnd.tasks.dispatch import TaskDispatcher, TERMINAL, _code, _permission
        from dclab_rnd.agents import build_registry
        registry = build_registry(('project',))
        caps = capability_snapshot(registry, allowed_tools={'run_stage', 'export_notebook'})
        with readonly_checkpoint(home, projects) as saver:
            d = TaskDispatcher(projects, saver, registry=registry) if saver is not None else None
            state = d.state(project_id) if d is not None else None
            stale = False
            if state:
                graph = TaskGraph.read(state['graph'])
                if graph_fingerprint(graph) != graph.fingerprint:
                    return finish('unavailable')
                stale = state.get('code') != _code() or state.get('source_identity') != d.source_revision(project_id)
                if not stale:
                    try:
                        d._sources(project, state, caps)  # metadata comparison only; no proof/event writer
                        from dclab_rnd.studio.data import data_path
                        data_path(projects, project)
                    except Exception:
                        stale = True
                status = 'stale' if stale else 'recorded'
                budget = state.get('budget') or {}
                out['budget'] = {'tool_calls': _number(state.get('tool_calls')), 'max_tool_calls': _number(budget.get('max_tool_calls')),
                                 'seconds': _number(state.get('seconds')), 'max_seconds': _number(budget.get('max_seconds')),
                                 'cost_eur': sum(_number((a.get('budget') or {}).get('cost_eur')) for a in state['attempts']),
                                 'max_cost_eur': _number(budget.get('max_cost_eur'))}
            else:
                try:
                    graph = _project_graph(project, caps)
                except Exception:
                    return finish('blocked')
                if graph is None:
                    return finish('blocked')
                status = 'preview'
            if len(graph.nodes) > 64 or any(n.id not in TITLES and not re.fullmatch(r'group_[0-9]{1,2}', n.id) for n in graph.nodes):
                return finish('unavailable')
            ids = {n.id for n in graph.nodes}
            if any(n.parent_id is not None and n.parent_id not in ids or set(n.dependencies) - ids for n in graph.nodes):
                return finish('unavailable')
            out.update(graph_revision=graph.revision, graph_fingerprint=graph.fingerprint,
                       checkpoint_revision=_number(state.get('revision'), None) if state else None)
            attempts = {a['task']: a for a in (state or {}).get('attempts', []) if not a.get('invalidated_by')}
            proofs = {v.get('id'): v for v in (state or {}).get('verifications', []) if isinstance(v, dict)}
            for node in graph.nodes:
                a = attempts.get(node.id)
                value, reason, approval = 'planned', 'Not run by the task runtime.', None
                if node.kind == 'group':
                    reason = 'Logical group; it does not call a tool.'
                elif node.kind in ('gate', 'verification'):
                    permission = (state or {}).get('holdout_permission') or {}
                    approved = node.id == 'holdout' and bool(permission) and sum(
                        _permission(e) == permission.get('fingerprint') for e in project.get('approvals', []) if e.get('gate') == 'holdout') == 1
                    approval = 'approved' if approved and not stale else 'awaiting approval'
                    value, reason = ('completed' if approved and not stale else 'blocked'), 'Owner approval or independent evidence is required; this is a check, not tool work.'
                elif node.kind == 'blocked':
                    value, reason = 'blocked', 'The workflow needs data, permission or a supported capability.'
                if a:
                    proof = proofs.get(a.get('verification_id')) or {}
                    if (a.get('status') in TERMINAL and a.get('acceptance_verification') == 'verified'
                            and proof.get('status') == 'verified' and proof.get('task_id') == node.id
                            and proof.get('plan_fingerprint') == a.get('plan_fingerprint')):
                        value, reason = 'completed', 'Completion was independently verified in the recorded task run.'
                        if not stale:
                            try:
                                d._evidence(project_id, node.id, a)  # read owner fingerprints; never reverify or reconcile
                            except Exception:
                                value, reason = 'stale', 'The recorded owner evidence has changed or is unavailable.'
                    else:
                        value, reason = 'blocked', _reason(a)
                        if a.get('status') in TERMINAL:
                            reason = 'Recorded completion has no matching current verification proof.'
                        if a.get('status') == 'started' and project.get('running') == node.id:
                            value, reason = 'running', 'The existing worker recorded this task as running.'
                if stale and node.kind != 'group':
                    value, reason = 'stale', MESSAGES['stale']
                evidence = None
                if a and a.get('evidence_ref'):
                    evidence = f'/api/projects/{project_id}/lineage'
                out['nodes'].append({'id': node.id, 'parent_id': node.parent_id, 'title': TITLES.get(node.id, 'Task group'),
                      'kind': node.kind, 'dependencies': list(node.dependencies), 'source_step_ids': list(node.source_step_ids),
                      'capability': node.capability.name if node.capability and node.capability.name in ('run_stage', 'export_notebook') else None,
                      'status': value, 'reason': reason, 'plan_revision': _number(a.get('plan_revision'), None) if a else None, 'approval': approval,
                      'evidence_url': evidence, 'verification': ('verified' if value == 'completed' and a else
                                      'stale' if value == 'stale' else 'blocked' if value == 'blocked' and a else 'not measured')})
            for _ in out['nodes']:
                stale_ids = {n['id'] for n in out['nodes'] if n['status'] == 'stale'}
                blocked_ids = {n['id'] for n in out['nodes'] if n['status'] == 'blocked'}
                changed = False
                for node in out['nodes']:
                    if node['status'] != 'stale' and stale_ids.intersection(node['dependencies']):
                        node.update(status='stale', reason='A prerequisite has stale evidence.', verification='stale')
                        changed = True
                    elif node['status'] not in ('stale', 'blocked') and blocked_ids.intersection(node['dependencies']):
                        node.update(status='blocked', reason='A prerequisite needs verification or owner approval.', verification='blocked')
                        changed = True
                if not changed:
                    break
            # Group progress derives only from its descendants, never an independent completion claim.
            by_id = {node['id']: node for node in out['nodes']}
            def group_status(key, seen):
                if key in seen:
                    raise ValueError('parent_cycle')
                item = by_id[key]
                children = [n for n in out['nodes'] if n['parent_id'] == key]
                if item['kind'] == 'group' and children:
                    states = [group_status(c['id'], seen | {key}) for c in children]
                    item['status'] = next((s for s in ('stale', 'blocked', 'running') if s in states),
                                          'completed' if all(s == 'completed' for s in states) else 'planned')
                return item['status']
            for node in out['nodes']:
                group_status(node['id'], set())
            if state:
                versions = [TaskGraph.read(old) for old in state.get('graph_history', [])] + [graph]
                for previous, following in zip(versions, versions[1:]):
                    prior = {n.id: n for n in previous.nodes}
                    current = {n.id: n for n in following.nodes}
                    changed = [key for key in dict.fromkeys([*prior, *current]) if prior.get(key) != current.get(key)]
                    # Describe adjacent transitions using the currently valid attempt. An invalidated older
                    # attempt must not hide its valid replacement, nor can later work predate a transition.
                    preserved = [n['id'] for n in out['nodes'] if n['status'] == 'completed' and n['kind'] != 'group'
                                 and n['id'] in attempts
                                 and _number(attempts[n['id']].get('graph_revision'), float('inf')) <= previous.revision]
                    out['revisions'].append({'kind': 'graph', 'from_revision': previous.revision, 'to_revision': following.revision,
                                             'source_changed': previous.source_refs != following.source_refs,
                                             'changed_tasks': [key for key in changed if key in TITLES], 'preserved_tasks': preserved})
                for revision in state.get('replans', []):
                    plan = revision.get('plan') or {}
                    previous = next((a for a in state['attempts'] if a['id'] == revision.get('attempt')), {})
                    task_id = plan.get('task_id') if plan.get('task_id') in TITLES else None
                    changed = [key for key in ('actions', 'budget', 'policy', 'expected_inputs') if plan.get(key) != (previous.get('plan') or {}).get(key)]
                    out['revisions'].append({'kind': 'replacement plan', 'task_id': task_id,
                                             'from_revision': _number(previous.get('plan_revision'), None),
                                             'to_revision': _number(plan.get('revision'), None), 'changed_fields': changed,
                                             'preserved_tasks': [n['id'] for n in out['nodes'] if n['status'] == 'completed' and n['kind'] == 'action' and n['id'] != task_id]})
            if state and isinstance(state.get('catalog_plan'), dict):
                cp = state['catalog_plan']
                decomp = cp.get('decomposition') if isinstance(cp.get('decomposition'), dict) else {}
                questions = []
                for call in (cp.get('calls') or ()):
                    if isinstance(call, dict) and call.get('task_type') == 'ask_owner':
                        params = call.get('params') if isinstance(call.get('params'), dict) else {}
                        q = str(params.get('question') or '').strip()[:240]
                        if q:
                            questions.append({'field': str(params.get('field') or 'other')[:40], 'question': q})
                for u in (decomp.get('unknowns') or ()):
                    if isinstance(u, dict) and u.get('blocking', True):
                        q = str(u.get('question') or '').strip()[:240]
                        if q and not any(item['question'] == q for item in questions):
                            questions.append({'field': str(u.get('field') or 'other')[:40], 'question': q})
                out['catalog_plan'] = {
                    'routed_to': cp.get('routed_to'),
                    'mode': cp.get('mode'),
                    'decision': str(decomp.get('decision') or '')[:200] or None,
                    'target': str(decomp.get('target') or '')[:80] or None,
                    'unit': str(decomp.get('unit') or '')[:80] or None,
                    'horizon_days': _number(decomp.get('horizon_days'), None),
                    'selected_tasks': [str(t)[:64] for t in (cp.get('selected_tasks') or ())][:24],
                    'subproblems': [str(s)[:64] for s in (cp.get('subproblems') or ())][:12],
                    'caught_traps': [str(t)[:80] for t in (cp.get('caught_traps') or ())][:24],
                    'revisions': _number(cp.get('revisions'), 1),
                    'calls_count': len(cp.get('calls') or ()),
                    'asked_owner': bool(cp.get('asked_owner')),
                    'clarifying_questions': questions[:5],
                    'owner_confirmed': bool(cp.get('owner_confirmed')),
                    'owner_note': str(cp.get('owner_note') or '')[:240] or None,
                }
            if any(n['status'] == 'stale' for n in out['nodes']):
                status = 'stale'
            return finish(status)
    except Exception:
        # No provider/source/pydantic exception text belongs in a progress response.
        out.update(nodes=[], revisions=[], budget=None)
        return finish('unavailable')
