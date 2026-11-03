"""System tasks B7 (2026-11-03): each supported kind on dev and sealed seeds, judged by system_gate_v1 fixed before any run;
the sealed seeds read once; a crashing task fails with why."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval import system_tasks as st  # noqa: E402


def runner(fail_kind=None, crash_kind=None):
    def run(kind, seed):
        if kind == crash_kind:
            raise RuntimeError("boom")
        ok = kind != fail_kind
        return {"kind": kind, "status": "supported" if ok else "partial", "checks": [{"ok": ok}], "missing": [] if ok else ["a decision"]}
    return run


class SystemTaskTests(unittest.TestCase):
    def setUp(self):
        self.campaign = Path(tempfile.mkdtemp())

    def test_every_kind_passing_passes_the_gate(self):
        doc = json.loads(st.run("dev", runner=runner(), campaign=self.campaign).read_text())
        self.assertEqual(doc["tasks"], [len(st.KINDS) * 2, len(st.KINDS) * 2])
        self.assertTrue(doc["verdict"].startswith("pass"))

    def test_a_kind_failing_on_every_seed_fails_the_suite_whatever_the_share(self):
        doc = json.loads(st.run("dev", runner=runner(fail_kind="uplift"), campaign=self.campaign).read_text())
        self.assertTrue(doc["verdict"].startswith("fail"))
        self.assertEqual(doc["kinds_failing_every_seed"], ["uplift"])

    def test_a_crash_is_a_failed_task_with_its_reason(self):
        doc = json.loads(st.run("dev", runner=runner(crash_kind="ranking"), campaign=self.campaign).read_text())
        row = next(r for r in doc["rows"] if r["kind"] == "ranking")
        self.assertFalse(row["passed"])
        self.assertIn("crashed: RuntimeError", row["missing"][0])

    def test_the_sealed_seeds_need_yes_and_are_read_once(self):
        with self.assertRaises(ValueError):
            st.run("sealed", runner=runner(), campaign=self.campaign)
        st.run("sealed", yes=True, runner=runner(), campaign=self.campaign)
        with self.assertRaises(FileExistsError):
            st.run("sealed", yes=True, runner=runner(), campaign=self.campaign)

    def test_the_gate_and_the_seeds_are_fixed(self):
        self.assertEqual(st.GATE["version"], "system_gate_v1")
        self.assertTrue(set(st.SPLITS["dev"]).isdisjoint({0, 1, 2}))  # the kinds campaign's seeds are never reused
        self.assertTrue(set(st.SPLITS["sealed"]).isdisjoint(st.SPLITS["dev"]))
        self.assertEqual(len(st.SEALED_SHA), 64)


if __name__ == "__main__":
    unittest.main()
