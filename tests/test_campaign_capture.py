"""A campaign run keeps its own ledger and session log in its workspace folder, the rules decide inside it, and the
process's own ledger and log come back afterwards, also after an error."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import campaign_capture, experience, ledger
    from dclab_rnd.ledger import use

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class CampaignCaptureTest(unittest.TestCase):
    def test_the_run_records_in_its_folder_and_the_process_gets_its_own_back(self):
        home = Path(tempfile.mkdtemp()) / "seed0"
        before_log, before_book = experience._INSTALLED.get("log"), ledger._INSTALLED.get("ledger")
        with self.assertRaises(RuntimeError):
            with campaign_capture.recorded(home):
                self.assertEqual(Path(ledger._INSTALLED["ledger"].home), home)
                self.assertEqual(Path(experience._INSTALLED["log"].home), home)
                self.assertEqual(use._SCOPE.get(), {"decisions": {}})  # the rules decide
                raise RuntimeError("a run that fails")
        self.assertIs(experience._INSTALLED.get("log"), before_log)
        self.assertIs(ledger._INSTALLED.get("ledger"), before_book)

    def test_the_code_hash_moves_with_any_file_under_a_folder(self):
        root = Path(tempfile.mkdtemp())
        (root / "pkg" / "sub").mkdir(parents=True)
        (root / "pkg" / "a.py").write_text("x = 1\\n")
        (root / "pkg" / "sub" / "b.py").write_text("y = 2\\n")
        first = campaign_capture.code_sha256(["pkg"], root)
        (root / "pkg" / "sub" / "b.py").write_text("y = 3\\n")
        self.assertNotEqual(campaign_capture.code_sha256(["pkg"], root), first)


if __name__ == "__main__":
    unittest.main()
