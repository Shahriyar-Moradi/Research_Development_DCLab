"""Every test run is kept whole (the owner's rule: capture everything, never cut before saving): the runner writes the run's header and
each unit's whole output under training_records/test_runs/<day>/, indexes it in runs.jsonl, never overwrites one, and hides a
database password."""

import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import run_parallel  # noqa: E402


class RunCaptureTest(unittest.TestCase):
    def test_a_run_is_kept_whole_and_indexed(self):
        long_output = "x" * 50_000 + "\nFAIL: test_something\n"
        results = [{"file": "test_a.py", "ok": True, "tests": 3, "seconds": 1.234, "failed": 0, "skipped": 1, "output": "OK"},
                   {"file": "test_b.py", "ok": False, "tests": 2, "seconds": 2.0, "failed": 1, "skipped": 0, "output": long_output}]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": "postgresql+psycopg://" + "me" + ":" + "secret" + "@host/db"}):
            path = run_parallel.capture(results, ["test_a", "test_b"], 3.4, False, False, where=Path(tmp))
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh]
            header, units = lines[0], lines[1:]
            self.assertEqual((header["kind"], header["ok"], header["tests"], header["failed"], header["failing_files"]), ("test_run", False, 5, 1, ["test_b.py"]))
            self.assertNotIn("secret", json.dumps(header))  # never a password
            self.assertEqual([u["file"] for u in units], ["test_a.py", "test_b.py"])
            self.assertEqual(units[1]["output"], long_output)  # the whole output, never cut
            index = [json.loads(line) for line in (Path(tmp) / "runs.jsonl").read_text().splitlines()]
            self.assertEqual(index[0]["path"], str(path.relative_to(tmp)))
            again = run_parallel.capture(results, [], 1.0, False, False, where=Path(tmp))
            self.assertNotEqual(again, path)  # a second run is another record
            self.assertEqual(len((Path(tmp) / "runs.jsonl").read_text().splitlines()), 2)

    def test_off_keeps_nothing_and_a_gate_keeps_to_the_main_checkout(self):
        with mock.patch.dict(os.environ, {"DCLAB_TEST_RUNS": "off"}):
            self.assertIsNone(run_parallel.runs_dir())
            self.assertIsNone(run_parallel.capture([], [], 0.0, True, False))
        with mock.patch.dict(os.environ, {"DCLAB_TEST_RUNS": ""}):
            where = run_parallel.runs_dir()
        self.assertEqual(where.parts[-2:], ("training_records", "test_runs"))


if __name__ == "__main__":
    unittest.main()
