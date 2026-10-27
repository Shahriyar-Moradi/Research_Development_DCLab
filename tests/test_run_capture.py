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


class NoLiveModelTest(unittest.TestCase):
    """make check-quick exported no DCLAB_NO_LIVE_MODELS and the runner passed the owner's OpenAI key on: a test's intern
    was served by a live model (2026-10-21). Every test process is now as CI's, whichever gate started the runner."""

    def test_a_test_process_never_reaches_a_live_model(self):
        owner = {"DCLAB_NO_LIVE_MODELS": "", "OPENAI_API_KEY": "sk-owner", "ANTHROPIC_API_KEY": "sk-ant", "DCLAB_LLM_PROVIDER": "anthropic",
                 "DCLAB_WORKSPACE_MONTHLY_EUR": "5", "DCLAB_COLLECTOR_MONTHLY_EUR": "18"}
        with mock.patch.dict(os.environ, owner):
            env = run_parallel.child_env(tempfile.mkdtemp())
        self.assertEqual(env["DCLAB_NO_LIVE_MODELS"], "1")
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DCLAB_LLM_PROVIDER", "DCLAB_WORKSPACE_MONTHLY_EUR", "DCLAB_COLLECTOR_MONTHLY_EUR"):
            self.assertEqual(env[key], "", key)  # empty, so .env (read without override) cannot fill them either

    def test_every_gate_that_runs_tests_switches_live_models_off(self):
        make = (ROOT / "Makefile").read_text(encoding="utf-8")
        line = next(x for x in make.splitlines() if x.endswith(": export DCLAB_NO_LIVE_MODELS = 1") and x.startswith("test "))
        for target in ("test", "test-serial", "check-quick", "check-quick-tests", "agent-test", "rd-check", "test-pg", "check-all"):
            self.assertIn(target, line.split(":")[0].split(), target)


if __name__ == "__main__":
    unittest.main()
