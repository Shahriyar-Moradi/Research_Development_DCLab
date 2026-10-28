"""One OpenMP thread by default on macOS, for every entry point (live HyperAck through DCLab Code, 2026-10-28).

The server DCLab Code started had two OpenMP runtimes loaded (PyTorch's from miniconda, LightGBM's from Homebrew);
every OpenMP thread waited at a fork barrier and the models stage sat at 0% CPU for 15 minutes. With
OMP_NUM_THREADS=1 it ran in 30 s. The tests already run with one thread (tests/run_parallel.py), so this checks a
fresh interpreter with the variable removed.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = "import dclab_rnd, os; print(os.environ.get('OMP_NUM_THREADS'))"


def fresh(env_extra=None):
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    env.update(env_extra or {})
    done = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().splitlines()[-1]


class OpenMPDefault(unittest.TestCase):
    def test_one_thread_on_macos_when_unset(self):
        self.assertEqual(fresh(), "1" if sys.platform == "darwin" else "None")

    def test_a_persons_own_setting_wins(self):
        self.assertEqual(fresh({"OMP_NUM_THREADS": "4"}), "4")


if __name__ == "__main__":
    unittest.main()
