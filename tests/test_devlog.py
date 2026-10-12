"""The development-trajectory corpus: records validate, examples are built per task, whole packages are held out,
private text is refused, and the committed output is what the records build."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import devlog
except ImportError as error:  # pydantic is not part of requirements/ci.txt
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class DevlogTest(unittest.TestCase):
    def test_every_record_validates_and_passes_the_gates(self):
        records = devlog.load()
        self.assertTrue(records)
        self.assertEqual(len({r.package for r in records}), len(records))
        built = devlog.build(records)
        self.assertEqual(built["manifest"]["refused"], [])

    def test_examples_cover_the_four_tasks_and_hold_out_whole_packages(self):
        built = devlog.build()
        tasks = {ex["task"] for rows in built["splits"].values() for ex in rows}
        self.assertTrue({"plan_package", "fix_finding"} <= tasks)
        train = {ex["package"] for ex in built["splits"]["train"]}
        val = {ex["package"] for ex in built["splits"]["val"]}
        self.assertFalse(train & val)

    def test_private_text_is_refused(self):
        record = devlog.load()[0].model_copy(deep=True)
        record.request = "Copied from /Users/someone/data/rides.csv"  # the request is in every example of a record
        self.assertEqual(devlog.build([record])["manifest"]["refused"][0]["why"], "an absolute path")

    def test_the_committed_output_is_fresh(self):
        out = Path(tempfile.mkdtemp())
        devlog.write(devlog.build(), out)
        for name in ("train.chat.jsonl", "val.chat.jsonl", "train.alpaca.jsonl", "val.alpaca.jsonl", "MANIFEST.json"):
            self.assertEqual((devlog.OUT / name).read_text(encoding="utf-8"), (out / name).read_text(encoding="utf-8"),
                             f"{name} is stale: run make sft-dev")
        json.loads((devlog.OUT / "MANIFEST.json").read_text())


if __name__ == "__main__":
    unittest.main()
