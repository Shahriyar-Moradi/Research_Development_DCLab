"""TP-01E: the campaign harness must also import without the ML installation."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROBE = """
import importlib.abc
import sys

blocked = {'dclab_core', 'numpy', 'pandas', 'scipy', 'sklearn'}
attempts = []

class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in blocked:
            attempts.append(fullname)
            raise ImportError('TP01E forbidden eager import: ' + fullname)

sys.meta_path.insert(0, NoML())
"""


class AgentImportBoundaryTests(unittest.TestCase):
    def probe(self, body):
        # A child interpreter prevents preloaded ML modules in the suite's fork
        # server from hiding an accidental eager import.
        env = dict(os.environ, DCLAB_NO_LIVE_MODELS="1", DCLAB_LLM_PROVIDER="",
                   ANTHROPIC_API_KEY="", ANTHROPIC_MODEL="")
        return subprocess.run([sys.executable, "-c", PROBE + body], cwd=ROOT,
                              env=env, capture_output=True, text=True, timeout=30)

    def test_harness_and_metrics_need_no_portable_core_or_ml_imports(self):
        result = self.probe("""
from dclab_rnd import agent_metrics
from dclab_rnd.agents import harness

assert 'campaign' in harness.AGENTS
assert agent_metrics.wilson(0, 0) == (0.0, 1.0)
card = agent_metrics.scorecard(rows=[])
assert card['agents'] == [] and card['suites'] == [] and card['rows'] == 0
assert not attempts, attempts
assert not (blocked & {name.split('.', 1)[0] for name in sys.modules})
""")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_probe_detects_the_old_eager_core_import_and_direct_ml_import(self):
        for statement in ("from dclab_core import bench", "import numpy", "import sklearn"):
            with self.subTest(statement=statement):
                result = self.probe(statement)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("TP01E forbidden eager import:", result.stderr)


if __name__ == "__main__":
    unittest.main()
