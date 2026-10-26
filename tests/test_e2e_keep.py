"""The owner's rule (capture everything for the six models): the end-to-end and browser tests' temporary workspaces are
copied under agent_runs/e2e/<day>/ before they go, so the store collects their sessions, ledgers and projects; the copy
is skipped when DCLAB_E2E_KEEP=0."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import product_e2e  # noqa: E402


class KeepWorkspaceTest(unittest.TestCase):
    def test_the_workspace_is_copied_whole_and_the_copy_can_be_skipped(self):
        home = tempfile.TemporaryDirectory()
        (Path(home.name) / "experience").mkdir()
        (Path(home.name) / "experience" / "events.jsonl").write_text('{"kind": "x"}\n')
        (Path(home.name) / "busy.lock").write_text("")
        root = Path(tempfile.mkdtemp())
        with mock.patch.dict(os.environ, {"DCLAB_E2E_KEEP": "1"}):
            kept = product_e2e.keep_workspace(home, "product_e2e", root)
        self.assertTrue((kept / "experience" / "events.jsonl").is_file())
        self.assertTrue((root / "scripted.json").is_file())  # the lake files no "person" answer from below it
        self.assertFalse((kept / "busy.lock").exists())
        self.assertEqual(kept.parent.parent, root)
        with mock.patch.dict(os.environ, {"DCLAB_E2E_KEEP": "0"}):
            self.assertIsNone(product_e2e.keep_workspace(home, "product_e2e", root))
        self.assertIsNone(product_e2e.keep_workspace(None, "product_e2e", root))
        with mock.patch.dict(os.environ, {"DCLAB_E2E_KEEP": "1"}), mock.patch("shutil.copytree", side_effect=OSError("disk full")):
            self.assertIsNone(product_e2e.keep_workspace(home, "product_e2e", root))  # a failed copy warns, never fails the tests
        home.cleanup()


class KeepResultsTest(unittest.TestCase):
    """Each flow's result is kept whole beside the test suite's runs (training_records/test_runs) and beside the kept
    workspace, so the store holds what passed, its numbers and its time, not only the screen's lines (owner, 2026-10-26)."""

    def test_every_flow_is_kept_whole_with_its_run(self):
        import gzip
        import json

        runs, kept = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        long = "holdout roc_auc 0.8416 (95% 0.8211 to 0.8647), " + "x" * 5000  # never cut
        results = [{"flow": "upload", "ok": True, "detail": long, "seconds": 37.2},
                   {"flow": "synthetic", "ok": False, "detail": "the build kept waiting on the team", "seconds": 12.0}]
        with mock.patch.dict(os.environ, {"DCLAB_TEST_RUNS": str(runs)}):
            path = product_e2e.keep_results(results, ["--only", "upload"], 49.2, kept)
        lines = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
        head, flows = lines[0], lines[1:]
        self.assertEqual([head["kind"], head["ok"], head["flows"], head["failed"], head["failing_flows"], head["argv"]],
                         ["e2e_run", False, 2, 1, ["synthetic"], ["--only", "upload"]])
        self.assertEqual(head["workspace"], str(kept))
        self.assertEqual([(f["kind"], f["flow"], f["ok"]) for f in flows], [("flow", "upload", True), ("flow", "synthetic", False)])
        self.assertEqual(flows[0]["detail"], long)
        index = [json.loads(line) for line in (runs / "runs.jsonl").read_text().splitlines()]
        self.assertEqual([index[-1]["kind"], index[-1]["ok"], index[-1]["path"]], ["e2e_run", False, str(path.relative_to(runs))])
        beside = json.loads((kept / "e2e_result.json").read_text())
        self.assertEqual([beside["run"], len(beside["flows"])], [head["run"], 2])
        with mock.patch.dict(os.environ, {"DCLAB_TEST_RUNS": "off"}):
            self.assertIsNone(product_e2e.keep_results(results, [], 1.0, None))  # off keeps none, as for the suite


if __name__ == "__main__":
    unittest.main()
