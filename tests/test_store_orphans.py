"""No store run outlives the server that started it (2026-10-25): five `store run` processes kept writing into the owner's
store after their test servers were gone. A run stops when its parent is gone or its folder of agent runs was removed,
and every script that starts a test server turns the worker's own loops off, as the test runner does."""

import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))


class OrphanTests(unittest.TestCase):
    def stopped_when(self, folder, parent):
        from dclab_rnd.store.__main__ import watch

        stopped = threading.Event()
        watch(folder, parent, every=0.02, stop=lambda code: stopped.set())
        return stopped

    def test_a_run_stops_when_its_folder_is_removed(self):
        folder = Path(tempfile.mkdtemp())
        stopped = self.stopped_when(folder, os.getppid())
        self.assertFalse(stopped.wait(0.2))  # the folder is there and the parent alive: the run goes on
        shutil.rmtree(folder)
        self.assertTrue(stopped.wait(2))

    def test_a_run_stops_when_its_parent_is_gone(self):
        self.assertTrue(self.stopped_when(None, -1).wait(2))  # given to another parent: the server that started it is gone

    def test_every_test_server_turns_the_workers_loops_off_as_the_test_runner_does(self):
        import model_paths
        import product_e2e
        import run_parallel

        env = run_parallel.child_env(tempfile.mkdtemp())
        for script in (product_e2e, model_paths):
            for key in ("DCLAB_STORE_LOOP", "DCLAB_STORE_CAPTURE", "DCLAB_COLLECTOR", "DCLAB_STUDENTS_LOOP"):
                self.assertEqual(script.BACKGROUND_OFF.get(key), env[key], (script.__name__, key))
        shots = (ROOT / "scripts" / "user_guide_shots.py").read_text(encoding="utf-8")
        for key in ("DCLAB_STORE_LOOP", "DCLAB_COLLECTOR", "DCLAB_STUDENTS_LOOP"):
            self.assertIn(f'"{key}": "0"', shots)


if __name__ == "__main__":
    unittest.main()
