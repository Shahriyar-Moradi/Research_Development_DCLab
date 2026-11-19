"""The exported service read 07/08/2025 as 8 July with no word and scored "not a date" as missing in silence
(the end-to-end export check, 2026-11-19). It now reads dates in the training data's own order and says when it guessed."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


class ServiceDateTests(unittest.TestCase):
    def test_the_training_dates_say_their_order(self):
        from dclab_rnd.studio.service import date_order

        self.assertTrue(date_order(pd.Series(["13/02/2025", "01/03/2025"])))  # 13 cannot be a month: day first
        self.assertFalse(date_order(pd.Series(["02/13/2025", "03/01/2025"])))
        self.assertIsNone(date_order(pd.Series(["2025-02-13", "2025-03-01"])))  # ISO: no order to learn
        self.assertIsNone(date_order(pd.Series(["01/02/2025"])))  # never past 12: unknown
        self.assertIsNone(date_order(pd.to_datetime(pd.Series(["2025-02-13"]))))

    def test_a_guessed_or_unread_date_is_said_on_its_row(self):
        from dclab_rnd.studio.service_template.scorer import read_times

        values, said = read_times(pd.Series(["2025-08-07", "07/08/2025", "not a date", None, ""]), None)
        self.assertEqual(values.iloc[0], "2025-08-07 00:00:00")
        self.assertIn("day/month or month/day", said[1][0])
        self.assertIn("could not be read", said[2][0])
        self.assertNotIn(3, said)  # missing is missing, not a mistake
        self.assertNotIn(4, said)

    def test_a_known_day_first_order_is_used_with_no_warning(self):
        from dclab_rnd.studio.service_template.scorer import read_times

        values, said = read_times(pd.Series(["07/08/2025"]), True)
        self.assertEqual((values.iloc[0], said), ("2025-08-07 00:00:00", {}))


if __name__ == "__main__":
    unittest.main()
