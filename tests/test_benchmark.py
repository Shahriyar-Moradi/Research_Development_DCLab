"""The frozen benchmark (package 14.3): 100+ unseen cases, split by source into dev and a sealed test set whose
fingerprints are committed; each case's expectations name columns its table really has."""
import importlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from dclab_rnd.agent_eval import benchmark
        except ImportError as error:
            raise unittest.SkipTest(f"pandas or scikit-learn not installed: {error}")
        cls.b = benchmark

    def test_size_and_a_split_by_source(self):
        b = self.b
        dev, test = b.split("dev"), b.split("test")
        self.assertGreaterEqual(len(b.CASES), 100)
        self.assertGreaterEqual(len(test), 60)
        self.assertEqual(len(dev) + len(test), len(b.CASES))
        self.assertFalse({c.tags[1] for c in dev} & {c.tags[1] for c in test}, "a base table in both sets: the test set is not held out by source")
        self.assertEqual(len({c.id for c in b.CASES}), len(b.CASES))
        self.assertEqual(len({c.seed for c in b.CASES}), len(b.CASES))
        from dclab_rnd.agent_eval.cases import CASES as SUITE
        self.assertFalse({c.id for c in SUITE} & {c.id for c in b.CASES})  # unseen: not the judgment suite's cases

    def test_the_sealed_test_set_has_not_changed(self):
        sealed = json.loads(self.b.SEALED.read_text(encoding="utf-8"))
        self.assertEqual(sealed["version"], self.b.VERSION)
        self.assertEqual(self.b.sealed_fingerprints(), sealed["cases"],
                         "a sealed case changed: make a new benchmark version instead of editing this one (docs/guides/PRODUCT_BUILD_PLAYBOOK.md 14.3)")

    def test_every_expectation_names_a_column_the_table_has(self):
        for case in self.b.CASES:
            frame = case.table()
            columns = set(frame.columns)
            self.assertIn("target", columns, case.id)
            for name in (*case.leaks, *case.innocent, case.time_column, case.group_column):
                if name:
                    self.assertIn(name, columns, f"{case.id}: {name}")
            self.assertFalse(set(case.leaks) & set(case.innocent), case.id)
            if case.suite == "leakage":
                self.assertTrue(case.leaks, case.id)
            if case.flag_duplicates:
                self.assertTrue(frame.duplicated().any(), case.id)
            self.assertTrue(case.moment.strip(), case.id)

    def test_a_run_on_the_sealed_set_is_always_recorded(self):
        with self.assertRaises(SystemExit):
            self.b.main(["--split", "test"])  # no --output: refused before anything runs

    def test_two_dev_cases_run_and_are_scored_by_family(self):
        suite = importlib.import_module("dclab_rnd.agent_eval.run")
        cases = self.b.split("dev")[:2]
        report = suite.run(("audit",), cases)
        self.assertEqual(len(report["results"]), 2)
        self.assertFalse([r["error"] for r in report["results"] if r["error"]])
        families = suite.by_trap_families(report["results"], {c.id: c.trap for c in cases})
        self.assertEqual(sorted(families["audit"]), sorted({c.trap for c in cases}))


if __name__ == "__main__":
    unittest.main()
