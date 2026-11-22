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
        self.assertEqual(p["StartCalendarInterval"], {"Hour": 1, "Minute": 30})
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

    def test_six_gb_of_swap_is_allowed(self):
        self.assertEqual(mod.MAX_SWAP, 6 * 1024 ** 3)

    def test_swap_over_six_gb_pauses_the_run_and_it_goes_on_under_five(self):
        # 2026-11-22: the nightly run was stopped twice while other work held the memory; a pause keeps what it did
        GB, free, st = 1024 ** 3, 50 * 1024 ** 3, {}
        self.assertEqual(mod.guard(st, 3 * GB, free, 0), "go")
        self.assertEqual(mod.guard(st, 7 * GB, free, 10), "pause")
        self.assertEqual(mod.guard(st, 5.5 * GB, free, 100), "wait")  # under 6 but not under 5: still paused, no flapping
        self.assertEqual(mod.guard(st, 4 * GB, free, 200), "resume")
        self.assertEqual(st["resumed_from"], 10)
        self.assertEqual(mod.guard(st, 4 * GB, free, 230), "go")

    def test_a_run_paused_three_hours_is_stopped_and_low_disk_always_stops_first(self):
        GB, free, st = 1024 ** 3, 50 * 1024 ** 3, {}
        self.assertEqual(mod.guard(st, 7 * GB, free, 0), "pause")
        self.assertEqual(mod.guard(st, 7 * GB, free, 3 * 3600 - 1), "wait")
        self.assertEqual(mod.guard(st, 7 * GB, free, 3 * 3600), "stop_swap")
        self.assertEqual(mod.guard({}, 1 * GB, 1 * GB, 0), "stop_disk")
        self.assertEqual(mod.guard({"paused_at": 0}, 7 * GB, 1 * GB, 5), "stop_disk")

    def test_the_watcher_stops_and_continues_the_child_and_logs_its_memory(self):
        import signal as sg

        sent, logs = [], []

        class Child:
            pid = 4242
            def __init__(self):
                self.done = False
            def poll(self):
                return 0 if self.done else None
            def send_signal(self, sig):
                sent.append(sig)
            def wait(self):
                import time as t
                while not self.done:
                    t.sleep(0.01)
                return 0

        child = Child()
        swaps = iter([7 * 1024 ** 3, 4 * 1024 ** 3])

        def swap():
            try:
                return next(swaps)
            except StopIteration:
                child.done = True
                return 0

        with mock.patch.object(mod.subprocess, "Popen", return_value=child), \
                mock.patch.object(mod, "decide", return_value=(True, "due")), \
                mock.patch.object(mod, "other_run_going", return_value=False), \
                mock.patch.object(mod, "worker_running", return_value=False), \
                mock.patch.object(mod, "swap_used", side_effect=swap), \
                mock.patch.object(mod, "rss_of", return_value=2 * 1024 ** 3), \
                mock.patch.object(mod, "log", side_effect=logs.append), \
                mock.patch.object(mod.time, "sleep", lambda s: None), \
                mock.patch("builtins.open", mock.mock_open()), \
                mock.patch.dict(sys.modules, {"dclab_rnd.store.conductor": mock.Mock(due=lambda: True)}):
            import dclab_rnd.store as store_pkg
            with mock.patch.object(store_pkg, "conductor", sys.modules["dclab_rnd.store.conductor"], create=True):
                import threading
                before = set(threading.enumerate())
                mod.run_once()
                child.done = True
                for th in set(threading.enumerate()) - before:
                    th.join(5)
        self.assertEqual(sent[:2], [sg.SIGSTOP, sg.SIGCONT])
        self.assertTrue(any("holds 2.0 GB" in line for line in logs), logs)
        self.assertTrue(any("1 pause(s)" in line for line in logs if line.startswith("ended")), logs)

    def test_swap_in_use_is_read_in_bytes_and_zero_when_unreadable(self):
        ss = mod

        out = mock.Mock(stdout="total = 4096.00M  used = 2501.56M  free = 1594.44M  (encrypted)")
        with mock.patch.object(ss.subprocess, "run", return_value=out):
            self.assertEqual(ss.swap_used(), int(2501.56 * 1024 ** 2))
        with mock.patch.object(ss.subprocess, "run", return_value=mock.Mock(stdout="")):
            self.assertEqual(ss.swap_used(), 0)

    def _child(self, sent, waits=None):
        class Child:
            pid = 4242
            def __init__(self):
                self.done = False
            def poll(self):
                return 0 if self.done else None
            def send_signal(self, sig):
                sent.append(sig)
            def wait(self):
                import time as t
                if waits:
                    raise waits
                for _ in range(500):
                    if self.done:
                        return 0
                    t.sleep(0.01)
                return 0
        return Child()

    def _run(self, child, swap, extra=()):
        logs = []
        patches = [mock.patch.object(mod.subprocess, "Popen", return_value=child),
                   mock.patch.object(mod, "decide", return_value=(True, "due")),
                   mock.patch.object(mod, "other_run_going", return_value=False),
                   mock.patch.object(mod, "worker_running", return_value=False),
                   mock.patch.object(mod, "swap_used", side_effect=swap),
                   mock.patch.object(mod, "rss_of", return_value=1024 ** 3),
                   mock.patch.object(mod, "log", side_effect=logs.append),
                   mock.patch.object(mod.time, "sleep", lambda s: None),
                   mock.patch("builtins.open", mock.mock_open()),
                   mock.patch.dict(sys.modules, {"dclab_rnd.store.conductor": mock.Mock(due=lambda: True)}), *extra]
        import contextlib
        import dclab_rnd.store as store_pkg
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(mock.patch.object(store_pkg, "conductor", sys.modules["dclab_rnd.store.conductor"], create=True))
            before = set(__import__("threading").enumerate())
            try:
                mod.run_once()
            except KeyboardInterrupt:
                pass
            finally:  # the fake child ends, and this test's watcher with it, before the patches go: never into the next test
                import threading
                child.done = True
                for th in set(threading.enumerate()) - before:
                    th.join(5)
        return logs

    def test_an_error_reading_the_swap_while_paused_never_leaves_the_run_stopped(self):
        import signal as sg
        sent = []
        child = self._child(sent)
        calls = iter([7 * 1024 ** 3])

        def swap():
            try:
                return next(calls)
            except StopIteration:
                raise OSError("sysctl could not start")  # the child is still paused here

        import threading
        went_on = threading.Event()
        orig = child.send_signal
        child.send_signal = lambda sig: (orig(sig), went_on.set() if sig == sg.SIGCONT else None)
        child.wait = lambda: (went_on.wait(5), 0)[1]
        self._run(child, swap)
        self.assertEqual(sent[:1], [sg.SIGSTOP])
        self.assertIn(sg.SIGCONT, sent[1:])  # the watcher's finally sent it on

    def test_ctrl_c_while_the_run_is_paused_lets_it_go_on(self):
        import signal as sg
        import threading
        sent = []
        paused = threading.Event()
        child = self._child(sent)
        orig_send = child.send_signal

        def send(sig):
            orig_send(sig)
            if sig == sg.SIGSTOP:
                paused.set()
        child.send_signal = send

        def wait():
            paused.wait(2)
            raise KeyboardInterrupt
        child.wait = wait
        swaps = iter([7 * 1024 ** 3] + [7 * 1024 ** 3] * 1000)
        self._run(child, lambda: next(swaps))
        self.assertEqual(sent[0], sg.SIGSTOP)
        self.assertIn(sg.SIGCONT, sent)  # the script's own end never leaves the child stopped

    def test_a_stop_while_paused_continues_the_child_before_interrupting_it(self):
        import signal as sg
        sent = []
        child = self._child(sent)
        GB = 1024 ** 3
        frees = iter([50 * GB, 50 * GB, 1 * GB])
        stop = mock.patch.object(mod, "stop_child", side_effect=lambda c: (sent.append("stop"), setattr(c, "done", True)))
        disk = mock.patch.object(mod.shutil, "disk_usage", side_effect=lambda p: mock.Mock(free=next(frees, 1 * GB)))
        self._run(child, lambda: 7 * GB, extra=(stop, disk))
        i, j = sent.index(sg.SIGCONT), sent.index("stop")
        self.assertLess(i, j)

if __name__ == "__main__":
    unittest.main()
