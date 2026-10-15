"""Package 21.5: a change to a prompt, a tool schema, the graph's validator, a role prompt or any suite's gate is not on
until the harness suite ran on it and passed (``make harness-eval``); its latest result carries the fingerprints of what
it ran with, and they must be what is here now."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval import regression  # noqa: E402


class HarnessGateTest(unittest.TestCase):
    def test_the_suite_ran_on_what_is_here_and_passed(self):
        run = regression.latest()
        self.assertIsNotNone(run, "no run of the harness suite: make harness-eval")
        self.assertTrue(str(run["verdict"]).startswith("pass"), f"{run['id']}: {run['verdict']}")
        self.assertEqual(regression.stale(run), [], f"changed since {run['id']}: run make harness-eval")
        self.assertIn("RG-01", [r["id"] for r in run["rows"]])  # the owner's morning, always


if __name__ == "__main__":
    unittest.main()
