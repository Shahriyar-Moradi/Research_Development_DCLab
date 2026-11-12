"""TP-06: frozen DEV A/B measurements and an honestly incomplete model-assisted arm. No product rollout."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

from dclab_core.task_fingerprints import metadata_fingerprint
from .gate import CASES, SEEDS, GATE, CASE_FILE, CASE_SHA256, judge

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / 'evidence/campaigns/task_planning_v1'
PROTOCOL = ROOT / 'docs/guides/TASK_PLANNING_EVALUATION_PROTOCOL.md'
FREEZE_NAME = 'freeze_v2.json'  # v1 retained: inherited provider settings broke its scripted capture diagnostic
CODE_FILES = ('dclab_rnd/agent_eval/task_planning/gate.py', 'dclab_rnd/agent_eval/task_planning/runner.py',
              'dclab_rnd/agent_eval/task_planning/probe.py', 'dclab_rnd/agent_eval/task_planning/__init__.py',
              'dclab_rnd/tasks/decomposer.py', 'dclab_rnd/tasks/planner.py', 'dclab_rnd/tasks/dispatch.py',
              'dclab_rnd/tasks/verifier.py', 'dclab_rnd/studio/engine.py', 'dclab_rnd/studio/graph.py',
              'dclab_rnd/agents/registry.py', 'dclab_core/task_verification.py', 'dclab_core/task_validation.py',
              'dclab_rnd/tasks/adapter.py', 'dclab_rnd/studio/lineage.py', 'dclab_rnd/studio/store.py',
              'dclab_rnd/studio/solution_report.py', 'dclab_rnd/models/gateway.py',
              'dclab_rnd/store/sets/llm.py', 'dclab_rnd/store/keepers/campaigns.py',
              'dclab_core/bench.py', 'dclab_core/task_fingerprints.py')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definitions():
    from dclab_rnd import prompts
    from dclab_rnd.agents import build_registry
    from dclab_rnd.tasks.adapter import capability_snapshot
    if _sha(CASE_FILE) != CASE_SHA256:
        raise ValueError('case_file_changed_after_import')
    # Reuse the scientific lineage owner's source closure. Only Python source is hashed; no sealed JSON or tables are read.
    from dclab_rnd.studio.lineage import CODE_PATHS
    from importlib.metadata import version, PackageNotFoundError
    code_names = set(CODE_FILES) | {'dclab_rnd/store/sets/benchmarks.txt'}
    code_names.update(path.relative_to(ROOT).as_posix() for prefix in CODE_PATHS for path in (ROOT / prefix).rglob('*.py')
                      if not {'build', 'dist', '__pycache__'} & set(path.parts))
    dependencies = {}
    for package in ('numpy', 'pandas', 'scikit-learn', 'pydantic', 'langgraph', 'langgraph-checkpoint-sqlite', 'lightgbm', 'xgboost'):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return {'case_sha256': CASE_SHA256, 'protocol_sha256': _sha(PROTOCOL),
            'gate': dict(GATE), 'gate_sha256': _sha(Path(__file__).with_name('gate.py')),
            'code_sha256': {name: _sha(ROOT / name) for name in sorted(code_names)}, 'dependency_versions': dependencies,
            'prompt_sha256': {name: hashlib.sha256(prompts.text(name).encode()).hexdigest() for name in ('task_decomposer', 'task_planner')},
            'capability_schema_sha256': metadata_fingerprint(capability_snapshot(build_registry(('project',)), allowed_tools={'run_stage', 'export_notebook'})),
            'seeds': list(SEEDS), 'dataset_version': 'invented_accounts.v1', 'rows': 240,
            'live_cost_cap_eur': 0.0, 'mode': 'dev_offline', 'evaluation_only': True}


def freeze(campaign=CAMPAIGN):
    campaign = Path(campaign)
    campaign.mkdir(parents=True, exist_ok=True)
    target = campaign / FREEZE_NAME
    current = definitions()
    if target.exists():
        record = json.loads(target.read_text())
        if record['definitions'] != json.loads(json.dumps(current)):
            raise ValueError('frozen_definitions_changed_use_a_new_version')
        return record
    from dclab_rnd.bench import _git
    record = {'version': 'tp06.freeze.v2', 'at': datetime.now(timezone.utc).isoformat(timespec='microseconds'),
              'commit': _git('rev-parse', 'HEAD'), 'dirty': bool(_git('status', '--porcelain')),
              'python': platform.python_version(), 'definitions': current}
    try:
        with target.open('x') as output:
            json.dump(record, output, indent=2)
            output.write('\n')
    except FileExistsError:
        return freeze(campaign)
    return record


def report(doc):
    lines = ['# TP-06 DEV task planning comparison', '', doc['verdict'], '',
             'Invented DEV cases only. These measurements do not establish production readiness.', '',
             '| Arm | Judged | Correct | Verified completion | Safety failures |', '|---|---:|---:|---:|---:|']
    for arm, rows in doc['scores'].items():
        judged = [r for r in rows if r.get('judged')]
        lines.append(f"| {arm} | {len(judged)} | {sum(r.get('correct', 0) for r in judged)} | {sum(r.get('completed', 0) for r in judged)} | "
                     + (str(sum(doc['recommendation']['critical'][arm].values())) if judged else 'not measured') + ' |')
    probe_ok = (doc['recommendation'].get('proposal_probe_validation') or {}).get('passed') is True
    lines += ['', 'C is not integrated: a valid model-proposed graph cannot currently be admitted by the dispatcher.' if probe_ok
              else 'The C diagnostic did not establish its refusal/capture protocol; inspect its recorded failures.',
              'The scripted proposal probe does not measure model quality or a complete C execution.', '',
              '## Per-stratum outcomes', '', '| Stratum | A correct | B correct | C judged |', '|---|---:|---:|---:|']
    for case in CASES:
        values = []
        for arm in ('A', 'B', 'C'):
            rows = [r for r in doc['scores'][arm] if r['case_id'] == case.id and r.get('judged')]
            values.append(sum(r.get('correct', 0) for r in rows) if arm != 'C' else len(rows))
        lines.append(f'| {case.stratum} | {values[0]}/{len(SEEDS)} | {values[1]}/{len(SEEDS)} | {values[2]} |')
    lines += ['', '## Paired uncertainty', '', 'Seed repetitions are clustered by authored case; they do not increase the independent sample size.', '',
              '```json', json.dumps(doc['recommendation']['comparisons'], indent=2), '```', '', '## Limits', '',
              '- Model-assisted execution is unavailable and unscored; no live-model request was sent.',
              '- Admission and routing strata are boundary checks, not completed scientific workflows.',
              '- Independent completion/recovery sample requirements are intentionally larger than this DEV smoke.',
              '- No defaults change. No sealed data, real-user outcome or general production claim is included.', '',
              'Reasons: ' + ', '.join(doc['recommendation']['reasons']), '']
    return '\n'.join(lines)


def run(*, campaign=CAMPAIGN, runner=None, proposal_probe=None, split='dev'):
    if split != 'dev':
        raise ValueError('only_frozen_offline_dev_is_available')
    from .runner import run_case
    from .probe import run_probe, offline_environment, require_standalone
    require_standalone()
    campaign = Path(campaign)
    frozen = freeze(campaign)  # durable before any result or scientific effect
    scores = {arm: [] for arm in ('A', 'B', 'C')}
    from dclab_rnd.bench import _git
    current_commit, current_dirty = _git('rev-parse', 'HEAD'), bool(_git('status', '--porcelain'))
    started = datetime.now(timezone.utc).isoformat(timespec='microseconds')
    with offline_environment():
        for case in CASES:
            for seed in SEEDS:
                for arm in scores:
                    try:
                        row = (runner or run_case)(case, seed, arm)
                    except Exception as error:
                        row = {'id': f'{case.id}:{seed}', 'case_id': case.id, 'seed': seed, 'stratum': case.stratum, 'phase': case.phase, 'scenario': case.scenario, 'expected': case.expected,
                               'judged': False, 'status': 'error', 'error_code': type(error).__name__, 'safety': {}}
                    scores[arm].append(row)
        try:
            probe = (proposal_probe or run_probe)()
        except Exception as error:
            probe = {'status': 'error', 'error_code': type(error).__name__}
    # A definition edited during a run invalidates the run; do not write a favorable result from it.
    freeze(campaign)
    recommendation = judge(scores, configured_model=False, c_integrated=False, proposal_probe=probe)
    doc = {'schema': 'dclab_rnd.task_planning_eval.v1', 'campaign_id': 'task_planning_v1', 'status': 'completed',
           'at': started, 'completed_at': datetime.now(timezone.utc).isoformat(timespec='microseconds'),
           'split': 'dev', 'mode': 'dev_offline', 'evaluation_only': True, 'commit': current_commit, 'dirty': current_dirty, 'freeze_commit': frozen['commit'],
           'freeze_ref': FREEZE_NAME, 'freeze_sha256': _sha(campaign / FREEZE_NAME), 'definitions': frozen['definitions'],
           'scores': scores, 'proposal_probe': probe, 'measurement_kind': 'injected_test_runner' if runner else 'real_offline_owner_operations', 'recommendation': recommendation, 'verdict': recommendation['verdict'],
           'promoted': False, 'spent_eur': 0.0, 'provider_requests': 0, 'gate': GATE['version']}
    # Bench treats [k, n] as independent Bernoulli trials. Seeds repeat authored
    # cases, so expose observation counts without requesting a Wilson interval.
    doc['summary'] = {}
    for arm, rows in scores.items():
        judged = [r for r in rows if r.get('judged')]
        doc['summary'].update({f'{arm}_correct_rows': sum(r.get('correct', 0) for r in judged),
                               f'{arm}_judged_rows': len(judged),
                               f'{arm}_authored_cases': len({r['case_id'] for r in judged})})
    results = campaign / 'results'
    results.mkdir(parents=True, exist_ok=True)
    number = max([int(p.name.split('-')[1].split('_')[0]) for p in results.glob('TPB-*_dev_offline.json')] or [0]) + 1
    while True:
        doc['id'] = f'TPB-{number:03d}'
        target = results / f"{doc['id']}_dev_offline.json"
        try:
            with target.open('x') as output:
                json.dump(doc, output, indent=2, allow_nan=False)
                output.write('\n')
            break
        except FileExistsError:
            number += 1
    reports = campaign / 'reports'
    reports.mkdir(exist_ok=True)
    with (reports / f"{target.stem}.md").open('x') as output:
        output.write(report(doc))
    return target
