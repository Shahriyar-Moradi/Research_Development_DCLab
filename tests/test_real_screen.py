"""Real tables screened by the engine (S7, 2026-10-25): each in its split's folder with the table at a glance and every
family's score, resumable, a sealed table apart, a network failure never marked, room checked before each table."""

import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import real_screen as R  # noqa: E402


class ScreenTest(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "DCLAB_DECISION_SHADOW": "0"})
        env.start()
        self.addCleanup(env.stop)
        from dclab_core import benchgen3

        self.frame, self.target, _ = benchgen3.craft_table(5, rows=400, structure="linear", task="binary")
        self.home = Path(tempfile.mkdtemp())

    def test_a_table_is_screened_into_its_splits_folder_once(self):
        calls = []

        def download(did):
            calls.append(did)
            return self.frame
        with mock.patch("dclab_core.lake.split_of", return_value="dev"):
            got = R.screen_one({"did": 42, "target": self.target}, self.home, download=download)
            again = R.screen_one({"did": 42, "target": self.target}, self.home, download=download)
        self.assertGreaterEqual(got["families"], 2)
        self.assertEqual((again, calls), (got, [42]))  # resumable: read once
        rec = json.loads((self.home / "dev" / "screens.jsonl").read_text().splitlines()[0])
        self.assertEqual(rec["split"], "dev")
        self.assertTrue(rec["first"] and rec["selected"] and len(rec["screen"]) >= 2)
        self.assertTrue({"numeric_columns", "category_columns", "missing_columns"} <= set(rec["state"]))  # the table at a glance
        self.assertEqual((rec["state"]["task"], rec["profile"]), ("binary_classification", "full"))  # its task; the product's settings
        self.assertTrue(any((self.home / "dev" / "ledgers" / "42").rglob("*.jsonl")))  # its decisions kept beside it

    def test_a_sealed_table_sits_apart_and_a_network_failure_leaves_no_mark(self):
        with mock.patch("dclab_core.lake.split_of", return_value="sealed"):
            R.screen_one({"did": 7, "target": self.target}, self.home, download=lambda did: self.frame)
            self.assertTrue((self.home / "sealed" / "screens.jsonl").is_file())
            self.assertFalse((self.home / "train").exists())
            got = R.screen_one({"did": 8, "target": self.target}, self.home, download=lambda did: (_ for _ in ()).throw(TimeoutError("timed out")))
        self.assertIn("timed out", got["failed"])
        self.assertFalse((self.home / "sealed" / "done" / "8.json").exists())  # tried again on the next run
        with mock.patch("dclab_core.lake.split_of", return_value="sealed"):
            bad = R.screen_one({"did": 9, "target": self.target}, self.home,
                               download=lambda did: (_ for _ in ()).throw(ValueError("could not convert string to float: 'Ali Rezaei'")))
        self.assertTrue((self.home / "sealed" / "done" / "9.json").is_file())  # a table that cannot be read is marked once
        self.assertNotIn("Ali", bad["failed"])  # never a cell's value in the record

    def test_the_family_screened_first_is_the_engines_own_first(self):
        from dclab_rnd.ledger import capture

        with mock.patch("dclab_core.lake.split_of", return_value="dev"):
            R.screen_one({"did": 43, "target": self.target}, self.home, download=lambda did: self.frame)
        rec = json.loads((self.home / "dev" / "screens.jsonl").read_text().splitlines()[0])
        self.assertIn(rec["first"], [s["model"] for s in rec["screen"]])
        self.assertIsNone(capture._first_family(object()))  # a kind's own machine, no bundle: nothing recorded

    def test_no_room_means_no_table(self):
        with mock.patch.object(R.os, "getloadavg", return_value=(99.0, 0, 0)):
            self.assertIn("busy", R.room())
        with mock.patch.object(R.os, "getloadavg", return_value=(0.1, 0, 0)), \
                mock.patch.object(R.shutil, "disk_usage", return_value=mock.Mock(free=1e9)):
            self.assertIn("disk", R.room())
        with mock.patch.object(R, "room", return_value="the Mac is busy"), mock.patch.object(R, "tables", return_value=[{"did": 1, "target": "y"}]):
            self.assertEqual(R.run(self.home, wait=False, log=lambda m: None), {"no_room": 1})


if __name__ == "__main__":
    unittest.main()
