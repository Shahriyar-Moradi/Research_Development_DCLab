"""Package 32.37: the store on a schedule for this Mac: a run only when due, alone, and with 2 GB free; the launchd job's
shape. No test loads anything into launchd or starts a real run."""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("store_scheduled", ROOT / "scripts" / "store_scheduled.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
GB = 1024 ** 3


class StoreScheduledTest(unittest.TestCase):
    def test_a_run_starts_only_when_due_alone_and_with_disk(self):
        self.assertEqual(mod.decide(True, 5 * GB, False), (True, "due"))
        self.assertFalse(mod.decide(True, 5 * GB, True)[0])  # never two runs at once
        go, why = mod.decide(True, int(1.5 * GB), False)
        self.assertFalse(go)
        self.assertIn("2 GB", why)
        self.assertFalse(mod.decide(False, 5 * GB, False)[0])  # the conductor's own hour
        go, why = mod.decide(True, 5 * GB, False, worker=True)
        self.assertFalse(go)
        self.assertIn("worker", why)  # the server's worker runs the store itself

    def test_the_launchd_job(self):
        p = mod.plist()
        self.assertEqual(p["Label"], "ai.dclab.store")
        self.assertEqual(p["StartInterval"], 6 * 3600)
        self.assertFalse(p["RunAtLoad"])
        self.assertEqual(p["ProgramArguments"][1], str(ROOT / "scripts" / "store_scheduled.py"))
        self.assertEqual(p["WorkingDirectory"], str(ROOT))
        self.assertEqual(p["ProcessType"], "Background")


if __name__ == "__main__":
    unittest.main()
