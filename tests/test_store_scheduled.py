"""Package 32.37: the store on a schedule for this Mac: a run only when due, alone, and with 2 GB free; the launchd job's
shape. No test loads anything into launchd or starts a real run."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

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


class GuardTests(unittest.TestCase):
    """2026-11-03: a whole run that did not fit in memory swapped onto the same disk for hours and ignored the interrupt."""

    def test_the_run_is_interrupted_then_terminated_then_killed_until_it_stops(self):
        import signal as sg

        ss = mod

        class Child:
            def __init__(self, dies_on):
                self.sent, self.dies_on, self.dead = [], dies_on, False

            def poll(self):
                return 0 if self.dead else None

            def send_signal(self, sig):
                self.sent.append(sig)
                self.dead = sig == self.dies_on

        with mock.patch.object(ss.time, "sleep", lambda s: None), mock.patch.object(ss.time, "time", side_effect=lambda: next(ticks)):
            ticks = iter(range(0, 10000, 100))
            child = Child(sg.SIGKILL)
            ss.stop_child(child)
        self.assertEqual(child.sent, [sg.SIGINT, sg.SIGTERM, sg.SIGKILL])
        with mock.patch.object(ss.time, "sleep", lambda s: None):
            first = Child(sg.SIGINT)
            ss.stop_child(first)
        self.assertEqual(first.sent, [sg.SIGINT])  # an interrupt that works is the only signal sent

    def test_swap_in_use_is_read_in_bytes_and_zero_when_unreadable(self):
        ss = mod

        out = mock.Mock(stdout="total = 4096.00M  used = 2501.56M  free = 1594.44M  (encrypted)")
        with mock.patch.object(ss.subprocess, "run", return_value=out):
            self.assertEqual(ss.swap_used(), int(2501.56 * 1024 ** 2))
        with mock.patch.object(ss.subprocess, "run", return_value=mock.Mock(stdout="")):
            self.assertEqual(ss.swap_used(), 0)


if __name__ == "__main__":
    unittest.main()
