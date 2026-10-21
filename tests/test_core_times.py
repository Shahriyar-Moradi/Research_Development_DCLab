"""dclab_core.times: text and times read the same way on pandas 2 and 3."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_core.times import is_text, read_times  # noqa: E402


class TimesTest(unittest.TestCase):
    def test_a_time_without_a_year_is_left_missing(self):  # pandas 3 reads "Oct  1 14:32:00" as year 1
        parsed = read_times(pd.Series(["Oct  1 14:32:00", "Oct  2 14:32:01"]))
        self.assertTrue(parsed.isna().all())

    def test_full_dates_and_mixed_formats_still_read(self):
        parsed = read_times(pd.Series(["2024-05-01", "02/05/2024 08:00", None]))
        self.assertEqual(parsed.dt.year.tolist()[:2], [2024, 2024])
        self.assertTrue(pd.isna(parsed.iloc[2]))
        self.assertEqual(read_times(["2024-05-01"]).year.tolist(), [2024])  # a list gives an index, kept as one

    def test_text_is_text_in_either_string_type(self):
        self.assertTrue(is_text(pd.Series(["a", "b"], dtype=object)))
        self.assertTrue(is_text(pd.Series(["a", "b"], dtype="string")))
        self.assertTrue(is_text(pd.Series(["a", "b"])))  # object on pandas 2, str on pandas 3
        self.assertFalse(is_text(pd.Series([1.0, 2.0])))
        self.assertFalse(is_text(pd.Series(pd.to_datetime(["2024-01-01"]))))


if __name__ == "__main__":
    unittest.main()
