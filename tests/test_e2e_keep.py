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


if __name__ == "__main__":
    unittest.main()
