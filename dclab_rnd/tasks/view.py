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


def _catalog_view(cp, statuses):
    """A plan as the page draws it: the breakdown, the questions still open, and the path with each step's state."""
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
    for a in ((cp.get('enforcement') or {}).get('asked') or ()) if isinstance(cp.get('enforcement'), dict) else ():
        q = str((a or {}).get('question') or '').strip()[:400]  # what the plan caught and code could not decide
        if q and not any(item['question'] == q for item in questions):
            questions.append({'field': str(a.get('field') or 'other')[:40], 'question': q})
    return {
        'routed_to': cp.get('routed_to'),
        'mode': cp.get('mode'),
        'decision': str(decomp.get('decision') or '') or None,  # whole: a stored plan's decision, as the planner wrote it (H3)
        'path': _path(cp, statuses),
        'followed': cp.get('followed') if isinstance(cp.get('followed'), dict) else None,
        'stored': cp.get('stored', True),
        # the model's thinking about this plan is the API thinking summary of this request, kept whole by the gateway
        'thinking_request': str(cp.get('model_request'))[:80] if cp.get('model_request') else None,
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


_SHOWN: dict = {}  # a shown-only plan per project state (names and counts, never a row)


def _rules_plan(projects, project, statuses):
    """A project with no task state still shows its path. When its plan was made before its stages (Home's build,
    ``tasks.path``) that stored plan is shown, with what was done about each catch. Otherwise the rules' plan is made
    for display from the names of its columns. Either way the page says whether what was built follows the plan
    (``enforced.obeyed``: a column the plan caught must be kept out of the final model, a split the plan asked for
    must be the split used). Nothing is stored here, nothing is changed, no model is asked."""
    from dclab_core.task_fingerprints import metadata_fingerprint
    from dclab_rnd.tasks import enforced
    from dclab_rnd.tasks import path as task_path
    from dclab_rnd.tasks.catalog import project_catalog_inputs, rules_plan

    kept = task_path.stored(project)
    # the table is read once for a project as it stands (its data, its solution, its stages, its plan), not on every look
    key = (project.get('id'), metadata_fingerprint({'data': (project.get('data') or {}).get('sha256'), 'solution': project.get('solution'),
                                                    'stages': {k: (v or {}).get('run') for k, v in (project.get('stages') or {}).items()},
                                                    'plan': (kept or {}).get('made_at'), 'asked': len(((kept or {}).get('enforcement') or {}).get('asked') or [])}))
    if key in _SHOWN:
        return _catalog_view(_SHOWN[key], statuses)
    brief, tables, facts, _, frames = project_catalog_inputs(projects, project)
    if kept is not None:
        # kept on the project, with what was done about each catch; its open questions are settled by changing the
        # solution on the project's page (no team walk holds this plan), so the page lists them without an answer box
        plan = {**kept, 'owner_confirmed': True, 'stored': False, 'mode': 'made before the stages'}
        solution = project.get('solution') or {}
        out_now = enforced.forbidden_columns(solution)
        record = dict(kept.get('enforcement') or {})
        # a question the person has since settled on the solution's page is no longer open
        record['asked'] = [a for a in record.get('asked') or [] if not (
            (a.get('field') == 'forbidden' and a.get('value') in out_now) or (a.get('field') in ('time_column', 'group_column') and solution.get(a.get('field'))))]
        plan['enforcement'] = record
    else:
        # this view never returns the project's goal text for a plan it only shows (TP-07: free text stays out of a
        # progress response): the shown plan names its one sub-problem by the outcome's column
        plan = {**rules_plan(f"Predict {facts['target']}" if facts.get('target') else "The project's goal", tables, facts, frames=frames),
                'mode': 'shown, not stored', 'stored': False, 'owner_confirmed': True}
    stages = {}
    for name in ('data', 'leakage', 'features', 'models', 'final'):
        record = projects.read_stage(project['id'], name)
        if record:
            stages[name] = record
    columns = [c for cols in tables.values() for c in cols]
    records = enforced.obeyed(plan.get('caught_traps') or [], project.get('solution') or {}, stages, columns)
    summary = enforced.summary(records)
    # a catch that is still a question for the person, or that they decided to leave, is not a broken plan
    record = plan.get('enforcement') if isinstance(plan.get('enforcement'), dict) else {}
    # (a catch code confirmed, waiting only because the solution is the person's to change, is not followed until they act)
    undecided = {str(a.get('value')) for a in record.get('asked') or () if not a.get('needs_person')}
    theirs = {str(w.get('value')) for w in record.get('waived') or ()}
    broken = [r for r in records if r.get('enforceable') and r.get('enforced') is False and str(r.get('column')) not in undecided | theirs]
    plan['followed'] = {'caught': summary['caught'], 'checked': summary['read'], 'followed': summary['enforced'],
                        'open': sum(1 for r in records if r.get('enforceable') and str(r.get('column')) in undecided),
                        'not_followed': [{'trap': r['trap'], 'column': r['column'], 'how': r['how']} for r in broken]}
    if len(_SHOWN) >= 64:
        _SHOWN.clear()
    _SHOWN[key] = plan
    return _catalog_view(plan, statuses)


def _path(cp, statuses):
    """The path from the problem to its solution as the page draws it (H3): each sub-problem, its steps with what each
    found and what the engine did about it (``enforce_plan``), then the build's stages with their state. Names and
    reasons only: a step's raw findings, which may count rows, stay in the record."""
    path = cp.get('path') if isinstance(cp.get('path'), dict) else None
    if not path or not isinstance(path.get('legs'), list):
        return None
    record = cp.get('enforcement') if isinstance(cp.get('enforcement'), dict) else {}

    def did(step):
        on = str(step.get('on') or '')
        for name, say in (('applied', lambda e: f"enforced: {e.get('field')} = {e.get('value')}" + (' (your decision)' if e.get('by') == 'human' else '')),
                          ('asked', lambda e: 'waiting for your answer'), ('waived', lambda e: 'left as it is (your decision)'),
                          ('too_late', lambda e: 'caught after work began')):
            for entry in record.get(name) or ():
                value = str(entry.get('value') or '')
                if value and (on == value or on.endswith('.' + value)):
                    return say(entry)
        return None

    confirmed = bool(cp.get('owner_confirmed'))
    legs = []
    for leg in path['legs'][:16]:
        if not isinstance(leg, dict):
            continue
        steps = []
        for st in leg.get('steps') or ():
            if not isinstance(st, dict):
                continue
            done = did(st)
            person = bool(st.get('for_person'))
            # a check ran when the plan was made; a question waits for the person; a catch may wait for their decision
            state = 'waiting for you' if (person and not confirmed) or done == 'waiting for your answer' else 'done'
            steps.append({'n': st.get('n'), 'task': str(st.get('task') or '')[:64], 'on': st.get('on'), 'why': str(st.get('why') or ''),
                          'for_person': person, 'question': st.get('question'), 'found': [str(t)[:120] for t in st.get('found') or ()][:12],
                          'did': done, 'state': state})
        legs.append({'id': str(leg.get('id') or '')[:61], 'kind': leg.get('kind'), 'objective': str(leg.get('objective') or ''),
                     'why': str(leg.get('why') or ''), 'after': [str(a)[:61] for a in leg.get('after') or ()], 'steps': steps})
    build = [{'n': b.get('n'), 'task': b.get('task'), 'what': b.get('what'), 'why': b.get('why'), 'for_person': bool(b.get('for_person')),
              'status': statuses.get(b.get('task'), 'pending')} for b in path.get('build') or () if isinstance(b, dict)]
    # the next step and its reason: the first question still open, else the first stage of the build not yet done
    waiting = next((st for leg in legs for st in leg['steps'] if st['state'] != 'done'), None)
    if waiting is not None:
        on = str(waiting.get('on') or '')
        asked = next((str(a.get('question')) for a in record.get('asked') or () if a.get('question') and a.get('value')
                      and (on == str(a['value']) or on.endswith('.' + str(a['value'])))), None)  # the question itself, never a tool's name
        nxt = {'n': waiting['n'], 'what': waiting.get('question') or asked or waiting['task'], 'who': 'you',
               'why': waiting['why'] or 'the plan cannot go on until this is settled'}
    else:
        stage = next((b for b in build if b['status'] not in ('completed', 'approved', 'reconciled', 'already_completed')), None)
        nxt = {'n': stage['n'], 'what': stage['what'], 'who': 'you' if stage['for_person'] else 'DCLab', 'why': stage.get('why') or ''} if stage else None
    return {'goal': path.get('goal'), 'legs': legs, 'build': build, 'next': nxt}


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


_SAFE_STATIC_BLOCKER_MAP = {
    'Outcome and prediction contract require current human solution signoff':
        'Solution is not signed yet — sign the solution to unlock this step.',
    'Current saved solution is outside this basic binary adapter':
        'Current saved solution is outside the basic binary adapter.',
    'Time/group/spatial splits or partial/attached labels require a separately reviewed source mapping':
        'Time, group, spatial splits or partial labels require a reviewed source mapping.',
    'Programme, solution-map and related-table projects require their existing atomic owner; this adapter cannot flatten them':
        'Programme, solution-map and related-table projects keep their existing atomic workflow.',
    'Current dataset column metadata is missing or invalid':
        'Dataset column metadata is missing or invalid.',
    'Saved solution columns do not match current dataset metadata':
        'Saved solution columns do not match current dataset columns.',
}
_SAFE_STATIC_BLOCKER_PREFIXES = (
    'Required registered capability is missing: ',
    'Registered capability scope/effect/move differs from the existing authority: ',
    'Host policy or missing project guard blocks capability: ',
    'Schema references require resolution outside this non-executing adapter: ',
    'Registered argument schema cannot represent this existing operation: ',
    'Registered capability has an invalid parameter schema: ',
)


def _static_blocker_reason(node):
    raw = getattr(node, 'blocker_reason', None)
    if not isinstance(raw, str) or not raw.strip():
        return 'The workflow needs data, permission or a supported capability.'
    safe = []
    for part in (p.strip() for p in raw.split(';')):
        if not part:
            continue
        if part in _SAFE_STATIC_BLOCKER_MAP:
            safe.append(_SAFE_STATIC_BLOCKER_MAP[part])
            continue
        for prefix in _SAFE_STATIC_BLOCKER_PREFIXES:
            if part.startswith(prefix):
                cap = part[len(prefix):].strip()
                if cap and re.fullmatch(r'[a-z_][a-z0-9_]*', cap):
                    safe.append(f'{prefix}{cap}.')
                break
    return ' '.join(safe) if safe else 'The workflow needs data, permission or a supported capability.'


def task_plan_view(projects, home, project_id, *, enabled=False, execution_enabled=False):
    from dclab_rnd.studio.graph import solution_signed
    project = projects.get(project_id)  # authenticate the workspace before touching a shared saver
    exec_on = bool(execution_enabled)
    signoff = project.get('signoff') or {}
    needs_signoff = not (
        solution_signed(project)
        and all(isinstance(signoff.get(key), str) and signoff[key].strip() for key in ('by', 'at'))
    )
    out = {'project_id': project_id, 'read_only': True, 'execution_enabled': exec_on, 'developer_preview': not exec_on,
           'needs_signoff': needs_signoff,
           'status': 'disabled', 'nodes': [], 'revisions': [], 'budget': None,
           'release': ('Opt-in task execution is enabled in Admin → Teams (task_planning_v3 tp06.gate.v2 PASS).' if exec_on
                       else 'Task execution passed task_planning_v3 (tp06.gate.v2 PASS); off by default until enabled by the owner in Admin → Teams.')}
    def finish(status):
        if enabled and 'catalog_plan' not in out:
            try:  # every project shows its path, with or without a task state (H3/H5)
                stage_status = {name: 'completed' if (meta or {}).get('status') in ('completed', 'approved') else (meta or {}).get('status') or 'pending'
                                for name, meta in (project.get('stages') or {}).items()}
                if stage_status.get('final') == 'completed':  # the holdout was approved and read, and the run reported
                    stage_status.update(holdout='completed', report='completed')
                # no task state holds this plan, so the project's own stages say what ran (found on a real Home build:
                # five finished stages were drawn as "blocked", the preview graph's word for "not run by the task path")
                out['catalog_plan'] = _rules_plan(projects, project, {**{n['id']: n['status'] for n in out['nodes']}, **stage_status})
            except Exception:
                pass  # the path is an addition to the view: its failure never hides the plan's own state
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
                    value, reason = 'blocked', _static_blocker_reason(node)
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
                out['catalog_plan'] = _catalog_view(state['catalog_plan'], {n['id']: n['status'] for n in out['nodes']})
            if any(n['status'] == 'stale' for n in out['nodes']):
                status = 'stale'
            return finish(status)
    except Exception:
        # No provider/source/pydantic exception text belongs in a progress response.
        out.update(nodes=[], revisions=[], budget=None)
        return finish('unavailable')
