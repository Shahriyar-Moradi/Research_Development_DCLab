"""Focused installed-wheel check for TP-05; reuse the existing isolated build/install runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import core_wheel_check as wheel
wheel.SMOKE = r'''
import sys, json, re
from importlib import metadata
import dclab_core
from dclab_core.task_verification import training_proof, check_training_proof, threshold_proof
from dclab_core.task_recovery import classify_failure, invalidated_tasks, rebase_verified
from dclab_core.adapters.conformance import problems
assert dclab_core.__file__.startswith(sys.prefix)
assert dclab_core.__version__ == metadata.version('dclab-core')
proof = training_proof([0,1,2,3], [4,5], [([0,1],[2,3]),([2,3],[0,1])], [0,1,0,1])
assert check_training_proof(proof) == []
threshold = threshold_proof(proof, [0,1,2,3], [.1,.8,.2,.9], configuration='default', policy={})
assert threshold['complete'] and threshold['finite_scores']
assert classify_failure('transient', effect='unknown')['action'] == 'stop'
assert problems() == []
assert 'dclab_rnd' not in sys.modules
norm = lambda x: re.sub(r'[-_.]+', '-', x).lower()
closure, todo = set(), ['dclab-core']
while todo:
    name = todo.pop()
    for req in metadata.requires(name) or []:
        if 'extra ==' in req.replace('extra==', 'extra =='):
            continue
        dep = norm(re.match(r'[A-Za-z0-9_.\-]+', req).group(0))
        if dep not in closure:
            closure.add(dep)
            try:
                metadata.distribution(dep)
                todo.append(dep)
            except metadata.PackageNotFoundError:
                pass
installed = {norm(d.metadata['Name']) for d in metadata.distributions()}
print(json.dumps({'python':sys.version.split()[0], 'dclab_core':dclab_core.__version__, 'plan_problems':problems(),
'direct':sorted(norm(re.match(r'[A-Za-z0-9_.\-]+', r).group(0)) for r in metadata.requires('dclab-core') if 'extra ==' not in r),
'unexpected_distributions': sorted(installed - closure - {'dclab-core'}), 'task_verification':'passed', 'task_recovery':'passed'}))
'''
raise SystemExit(wheel.main())
