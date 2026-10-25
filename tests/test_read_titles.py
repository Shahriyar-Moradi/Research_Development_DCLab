"""A table exported with a title above its header ("Export of 2024-10-01") is read from its header, with a note; a
plain file is read as before (2026-10-25: on planted files the reader turned such an export into one column)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.draft import structure  # noqa: E402


class TitleLinesTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.frame = pd.DataFrame({"id": np.arange(60), "city": rng.choice(["paris", "lyon"], 60), "amount": rng.normal(100, 9, 60).round(2)})
        self.folder = Path(tempfile.mkdtemp())

    def read(self, name: str, text: str):
        path = self.folder / name
        path.write_text(text)
        return structure.to_table(path)

    def test_a_title_and_a_blank_line_above_the_header_are_skipped_and_said(self):
        frame, info = self.read("t.csv", "Export of 2024-10-01\n\n" + self.frame.to_csv(index=False))
        self.assertEqual(frame.shape, self.frame.shape)
        self.assertEqual(list(frame.columns), list(self.frame.columns))
        self.assertTrue(any("Skipped 2 lines above the header" in n for n in info["notes"]))

    def test_a_semicolon_export_with_two_title_lines(self):
        frame, info = self.read("t.txt", "Report: sales\nRegion: north\n" + self.frame.to_csv(index=False, sep=";"))
        self.assertEqual(frame.shape, self.frame.shape)
        self.assertEqual(info["format"], "csv")

    def test_a_plain_file_is_read_as_before(self):
        frame, info = self.read("t.csv", self.frame.to_csv(index=False))
        self.assertEqual(frame.shape, self.frame.shape)
        self.assertFalse(any("above the header" in n for n in info["notes"]))
        self.assertEqual(structure.title_lines(self.frame.to_csv(index=False).encode(), ","), 0)

    def test_a_line_with_fields_of_its_own_is_never_taken_for_a_title(self):
        text = "a,b\n" + self.frame.to_csv(index=False)  # two fields, not one: not a title, and not the table's own count
        self.assertEqual(structure.title_lines(text.encode(), ","), 0)


if __name__ == "__main__":
    unittest.main()
