"""The store's hourly job runs the conductor in a process of its own, at a lower priority (2026-10-20: a whole store run in
the worker's own interpreter slowed a models stage and an intern turn running beside it). A run that ends without its
record fails the job, so the next run waits its hour; a run whose keepers partly failed still ran."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StoreJobTest(unittest.TestCase):
    def services(self, home):
        return SimpleNamespace(store=SimpleNamespace(home=home))

    def test_the_store_job_runs_the_conductor_in_its_own_lower_priority_process(self):
        from dclab_rnd.jobs import handlers

        home = Path(tempfile.mkdtemp())
        record = json.dumps({"id": "store-all-x", "failed": ["walks"], "steps": []})
        with mock.patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 1, stdout=record, stderr="")) as run:
            handlers._store_run(self.services(home), {})  # a keeper failed: the run still ran, the job is done
        command = run.call_args.args[0]
        self.assertEqual(command[1:4], ["-m", "dclab_rnd.store", "run"])
        self.assertEqual(command[command.index("--agent-runs") + 1], str(home))
        self.assertIsNotNone(run.call_args.kwargs["preexec_fn"])  # nice 10: the worker's own jobs come first
        self.assertEqual(run.call_args.kwargs["timeout"], handlers.STORE_RUN_SECONDS)

    def test_a_run_without_its_record_fails_the_job(self):
        from dclab_rnd.jobs import handlers

        broken = subprocess.CompletedProcess([], 1, stdout="", stderr="Traceback ... ImportError: no module")
        with mock.patch.object(subprocess, "run", return_value=broken):
            with self.assertRaises(RuntimeError) as caught:
                handlers._store_run(self.services(Path(tempfile.mkdtemp())), {})
        self.assertIn("ImportError", str(caught.exception))

    def test_the_command_runs_on_the_workspaces_folder_it_is_given(self):
        from dclab_rnd.store import __main__ as cli
        from dclab_rnd.store import conductor

        folder = Path(tempfile.mkdtemp())
        with mock.patch.object(conductor, "conduct", return_value={"id": "r", "started": "s", "ended": "e", "failed": [], "source_errors": 0,
                                                                    "steps": []}) as conduct, mock.patch("builtins.print"):
            self.assertEqual(cli.main(["run", "--root", tempfile.mkdtemp(), "--agent-runs", str(folder)]), 0)
        self.assertEqual(conduct.call_args.args[0], {"agent_runs": folder})


if __name__ == "__main__":
    unittest.main()
