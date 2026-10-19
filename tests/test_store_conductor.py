"""Package 32.2: the store's conductor.

A run is a session of the harness agent ``conductor`` with a step per keeper, and one record in ``runs/`` saying what
each keeper read, kept, left unchanged and refused; a keeper that fails is recorded and the others still run; a source a
keeper does not own is refused; two keepers claiming one path cannot both be registered; the run is due at most once an
hour and never with ``DCLAB_STORE_LOOP=0``.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)


def keepers():
    from dclab_rnd.store.keepers import Claim, Keeper, Source, register

    reg = {}

    def good(env):
        yield Source("wtest", "experience/2026/10/20/a.jsonl", b'{"kind": "visit.started"}\n', consent="train")
        yield Source("wtest", "not-mine/b.jsonl", b"x")  # outside its claims: refused

    def broken(env):
        yield Source("wtest", "audit.jsonl", b"{}\n")
        raise RuntimeError("the source went away")

    register(Keeper("sessions", (Claim("sessions", "experience/"),), good, feeds=("m2_workflow_decisions",)), reg)
    register(Keeper("audit", (Claim("audit", "audit"),), broken), reg)
    return reg


@unittest.skipIf(MISSING, MISSING)
class ConductorTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout

        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        self.log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        experience.install(self.log)
        self.addCleanup(experience.install, None)

    def test_a_run_is_a_session_of_the_conductor_with_a_step_per_keeper(self):
        from dclab_rnd.store import Store, conductor

        record = conductor.conduct(None, self.layout, keepers())
        steps = {s["name"]: s for s in record["steps"]}
        self.assertEqual((steps["sessions"]["read"], steps["sessions"]["kept"], len(steps["sessions"]["refused"])), (2, 1, 1))
        self.assertIn("RuntimeError", steps["audit"]["error"])  # recorded, and the other keeper ran
        self.assertEqual(steps["audit"]["kept"], 1)  # what it kept before failing stays kept
        self.assertEqual(record["failed"], ["audit"])
        kept = json.loads((self.layout.runs() / record["started"][:10] / f"{record['id']}.json").read_text(encoding="utf-8"))
        self.assertEqual(kept["id"], record["id"])
        self.assertEqual(len(Store(self.layout).entries("sessions")), 1)
        again = conductor.conduct(None, self.layout, keepers())
        self.assertEqual({s["name"]: s["unchanged"] for s in again["steps"]}["sessions"], 1)  # nothing new the second time
        steps_seen = [e for e in self.log.read() if e.kind == "agent.step" and e.ids.run == record["id"]]
        self.assertEqual(sorted(e.payload.get("tool") for e in steps_seen), ["curate", "keep:audit", "keep:sessions", "shape"])

    def test_two_keepers_on_one_path_are_refused(self):
        from dclab_rnd.store.keepers import Claim, Keeper, register

        reg = keepers()
        with self.assertRaises(ValueError):
            register(Keeper("sessions_too", (Claim("sessions", "experience/2026/"),), lambda env: iter(())), reg)
        register(Keeper("tests", (Claim("tests", "test_runs/"),), lambda env: iter(())), reg)  # another kind: fine

    def test_the_run_is_due_hourly_and_never_when_turned_off(self):
        from dclab_rnd.store import conductor

        with mock.patch.dict(os.environ, {"DCLAB_STORE_LOOP": "1"}):
            self.assertTrue(conductor.due(self.layout))
            conductor.conduct(None, self.layout, {})
            self.assertFalse(conductor.due(self.layout))
            import time

            self.assertTrue(conductor.due(self.layout, clock=time.time() + conductor.EVERY_SECONDS + 1))
        with mock.patch.dict(os.environ, {"DCLAB_STORE_LOOP": "0"}):
            self.assertFalse(conductor.due(self.layout.__class__(Path(tempfile.mkdtemp()))))  # an empty store, but turned off


if __name__ == "__main__":
    unittest.main()
