"""A collected journey play is kept whole (S8 A2, 2026-10-25): its ledger, drafts, project and events where the store
reads a workspace, each decision naming the person as its dataset, the invented tables left out with their hashes; the
same rows as a play that keeps nothing; and nothing kept unless asked (the gate never asks)."""

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

from dclab_rnd import workflow_models as W  # noqa: E402


class KeptPlayTest(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")
        self.journey = json.loads((W.JOURNEYS / "journeys_dev.json").read_text())[0]
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "DCLAB_DECISION_SHADOW": "0"})
        env.start()
        self.addCleanup(env.stop)

    def test_a_kept_play_holds_its_records_not_its_tables_and_the_store_finds_it(self):
        from dclab_rnd.store.keepers import workspaces

        base = Path(tempfile.mkdtemp())
        with mock.patch.dict(os.environ, {"DCLAB_JOURNEY_PLAYS": str(base)}):
            kept = W.play(("dev", self.journey, 0))
        os.environ.pop("DCLAB_JOURNEY_PLAYS", None)
        plain = W.play(("dev", self.journey, 0))
        self.assertEqual([(r["point"], r["value_raw"]) for r in kept["rows"]], [(r["point"], r["value_raw"]) for r in plain["rows"]])
        folder = base / "dev" / self.journey["id"] / "s0"
        records = [json.loads(line) for p in (folder / "workspace" / "ledger").rglob("*.jsonl") for line in p.read_text().splitlines() if line.strip()]
        decisions = [r for r in records if r.get("point")]
        self.assertTrue(decisions)
        self.assertEqual({r.get("dataset") for r in decisions}, {f"journey:dev:{self.journey['id']}"})  # one person, one group
        self.assertTrue(list((folder / "workspace" / "experience").glob("*/*/*/journey.jsonl")))  # the events, where a workspace keeps them
        self.assertTrue(list((folder / "workspace" / "drafts").rglob("draft.json")))  # every turn of the conversation
        self.assertFalse([p for p in folder.rglob("*") if p.suffix in (".csv", ".parquet")])  # the tables left out ...
        left = json.loads((folder / "LEFT_OUT.json").read_text())["files"]
        self.assertTrue(left and all(len(f["sha256"]) == 64 for f in left))  # ... and named
        found = [w.root for w in workspaces.discover(base)]
        self.assertIn(folder / "workspace", found)

    def test_nothing_is_kept_unless_asked(self):
        os.environ.pop("DCLAB_JOURNEY_PLAYS", None)
        self.assertIsNone(W.kept_folder("dev", "JD-01", 0))
        with mock.patch.dict(os.environ, {"DCLAB_JOURNEY_PLAYS": "off"}):
            self.assertIsNone(W.kept_folder("dev", "JD-01", 0))


if __name__ == "__main__":
    unittest.main()
