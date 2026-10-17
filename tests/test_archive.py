"""The permanent record of everything captured for training (make archive): dated, checksummed, read-only snapshots."""

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src = self.tmp / "bronze"
        (self.src / "answers").mkdir(parents=True)
        (self.src / "answers" / "a.jsonl").write_text('{"x": 1}\n')
        (self.src / "b.json").write_text("{}")
        self.out = self.tmp / "records"

    def snap(self, name):
        from dclab_rnd import archive

        return archive.archive(self.out, database=False, sources={"lake_bronze": [self.src]}, now=name)

    def test_a_snapshot_keeps_every_file_with_its_hash_and_is_read_only(self):
        got = self.snap("S1")
        snap = Path(got["snapshot"])
        manifest = json.loads((snap / "MANIFEST.json").read_text())
        self.assertEqual(sorted(f["path"] for f in manifest["files"]), ["lake_bronze/bronze/answers/a.jsonl", "lake_bronze/bronze/b.json"])
        kept = snap / "lake_bronze" / "bronze" / "answers" / "a.jsonl"
        self.assertEqual(kept.read_text(), '{"x": 1}\n')
        self.assertFalse(kept.stat().st_mode & stat.S_IWUSR)  # read-only once written
        index = json.loads((self.out / "INDEX.json").read_text())
        self.assertEqual([s["name"] for s in index["snapshots"]], ["S1"])
        self.assertTrue((self.out / "README.md").is_file())

    def test_an_unchanged_file_is_linked_to_the_last_snapshot_never_to_the_live_file(self):
        self.snap("S1")
        with (self.src / "answers" / "a.jsonl").open("a") as fh:  # a live file keeps growing
            fh.write('{"x": 2}\n')
        got = self.snap("S2")
        self.assertEqual(got["placed"], {"linked": 1, "copied": 1})
        first = self.out / "snapshots" / "S1" / "lake_bronze" / "bronze" / "answers" / "a.jsonl"
        self.assertEqual(first.read_text(), '{"x": 1}\n')  # the earlier snapshot did not move with the live file
        same = [self.out / "snapshots" / s / "lake_bronze" / "bronze" / "b.json" for s in ("S1", "S2")]
        self.assertEqual(os.stat(same[0]).st_ino, os.stat(same[1]).st_ino)

    def test_a_snapshot_is_never_written_twice(self):
        self.snap("S1")
        with self.assertRaises(ValueError):
            self.snap("S1")


if __name__ == "__main__":
    unittest.main()
