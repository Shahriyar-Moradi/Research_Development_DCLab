"""Scripted C proposal/admission probe. It is not a model-quality or integrated-C evaluation."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch


@contextmanager
def offline_environment():
    require_standalone()
    from dclab_rnd.models import settings
    cleared = {key: '' for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL', 'DCLAB_LLM_BASE_URL',
               'DCLAB_LLM_PROVIDER', 'ANTHROPIC_API_KEY', 'ANTHROPIC_MODEL', 'DCLAB_LOG_PROMPTS',
               'OPENROUTER_API_KEY', 'OPENROUTER_BASE_URL', 'OPENROUTER_MODEL', 'DCLAB_INTERN_MODEL',
               'DCLAB_DATABASE_URL', 'DCLAB_TRACE_EXPORT', 'DCLAB_SEALED_RUN', 'DCLAB_EXAM_VAULT')}
    cleared.update({f'DCLAB_TIER_{tier.upper()}_{field}': '' for tier in settings.TIERS for field in ('BASE_URL', 'MODEL', 'API_KEY')})
    with patch.dict(os.environ, {**cleared, 'DCLAB_NO_LIVE_MODELS': '1', 'DCLAB_DECISION_SHADOW': '0',
                    'OPENAI_API_KEY': 'scripted-fixture', 'OPENAI_MODEL': 'claude-sonnet-5-5',
                    'OPENAI_BASE_URL': 'https://api.anthropic.com/v1', 'DCLAB_THINKING': ''}):
        yield


def require_standalone():
    from dclab_rnd import context
    if context.services() is not None:
        raise RuntimeError('task_planning_evaluation_requires_standalone_process')


def run_probe():
    from dclab_core.card import problem_card
    from dclab_core.solution_workflow import solution_workflow
    from dclab_core.task_graph import TaskGraph
    from dclab_rnd import experience, ledger
    from dclab_rnd.agents import build_registry
    from dclab_rnd.core_adapter import brief_from_project, plan_from_project
    from dclab_rnd.models import Gateway, FileUsage
    from dclab_rnd.orchestra.checkpoints import open_checkpointer
    from dclab_rnd.studio import ProjectStore
    from dclab_rnd.tasks.adapter import adapt_workflow, capability_snapshot
    from dclab_rnd.tasks.decomposer import decompose
    from dclab_rnd.tasks.dispatch import TaskDispatcher
    from dclab_rnd.tasks.planner import plan_task
    from .gate import CASES, SEEDS
    from .runner import make_project

    calls = []
    class Scripted:
        thinking = None
        def complete(self, messages, tools=None, max_tokens=None, **options):
            body = json.loads(messages[-1]['content'])
            baseline = body.get('baseline') or body['deterministic_graph']
            if 'deterministic_graph' in body:
                leaves = [node for node in baseline['nodes'] if node['kind'] != 'group']
                answer = {'version': 'tp02.v1', 'baseline_fingerprint': baseline['fingerprint'],
                          'groups': [{'id': 'group_0', 'parent_id': None, 'members': [node['id'] for node in leaves]}],
                          'dependencies': [{'task_id': node['id'], 'requires': node['dependencies']} for node in leaves]}
            else:
                answer = {'version': 'tp03.v1', 'baseline_fingerprint': baseline['fingerprint'],
                          **{key: baseline[key] for key in ('actions', 'budget', 'policy')}}
            calls.append({'tools': tools, 'thinking_requested': options})
            return {'content': json.dumps(answer), 'finish_reason': 'stop', 'tool_calls': [],
                    'thinking': [{'text': 'SCRIPTED FIXTURE: preserve atomic tasks and existing human safety gates.'}],
                    'usage': {'input_tokens': 10, 'output_tokens': 30}}

    with tempfile.TemporaryDirectory(prefix='tp06-proposal-') as temporary, offline_environment(), ExitStack() as stack:
        home = Path(temporary)
        previous, previous_ledger = experience.current_log(), ledger.current_ledger()
        experience.install(experience.FileLog(home / 'experience', 'tp06-evaluation-only'))
        ledger.install(None)
        stack.callback(experience.install, previous)
        stack.callback(ledger.install, previous_ledger)
        stack.enter_context(patch('dclab_rnd.experience.model_text_allowed', return_value=True))
        store = ProjectStore(home / 'projects')
        pid = make_project(store, CASES[0], SEEDS[0])
        project = store.get(pid)
        registry = build_registry(('project',))
        caps = capability_snapshot(registry, allowed_tools={'run_stage', 'export_notebook'})
        source = plan_from_project(project)
        workflow = solution_workflow(problem_card(brief_from_project(project)))
        graph, mapping = adapt_workflow(workflow, source, project, capabilities=caps)
        gateway = Gateway(FileUsage(home / 'usage.jsonl'), transport=lambda tier, purpose: Scripted(), sleep=lambda _: None)
        candidate, decomposition = decompose(workflow, source, project, deterministic_graph=graph,
                           capabilities=caps, confirmed_goal=project['goal'], enabled=True, gateway=gateway)
        with open_checkpointer(home, database_url='') as saver:
            dispatcher = TaskDispatcher(store, saver, enabled=True, registry=registry)
            observed = dispatcher._observed(project, {'attempts': []}, candidate, mapping)
            plan, planning = plan_task(workflow, source, project, graph=candidate, task_id='data', observed_state=observed,
                                      capabilities=caps, remaining_budget={'max_tool_calls': 2, 'max_seconds': 180, 'max_cost_eur': 0.0},
                                      enabled=True, gateway=gateway)
            result = dispatcher.dispatch(pid, 'data', plan=plan) if (decomposition['status'] == 'proposed' and planning['status'] == 'proposed'
                    and plan is not None) else {'status': 'blocked', 'reason': 'no_model_proposal_admitted'}
            state = dispatcher.state(pid)
            attempts = len((state or {}).get('attempts', []))
        exchanges = [json.loads(line) for path in home.glob('model_exchanges/**/*.jsonl') for line in path.read_text().splitlines()]
        summaries = [{'label': exchange.get('thinking_label'), 'source': 'scripted transport fixture, not model reasoning',
                      'text': [block.get('text') for block in exchange.get('thinking', []) if block.get('text')]}
                     for exchange in exchanges]
        return {'mode': 'scripted_proposal_probe', 'integrated_c': False, 'provider_requests': 0, 'billed_eur': 0.0,
                'scripted_requests': len(calls), 'decomposition_status': decomposition['status'],
                'planning_status': planning['status'], 'decomposition_refusals': decomposition.get('refusals', []), 'proposed_graph_revision': candidate.revision,
                'dispatch_status': result['status'], 'dispatch_reason': result.get('reason'),
                'attempts': attempts, 'owner_records': len(store.records(pid)), 'holdout_uses': store.get(pid)['holdout_uses'],
                'api_thinking_summaries': summaries,
                'exchange_capture_passed': len(exchanges) == len(calls) == 2 and all(e.get('reply') and e.get('thinking') for e in exchanges),
                'tool_schemas_offered': any(c['tools'] for c in calls),
                'reason': 'model_graph_not_admitted_by_runtime'}
