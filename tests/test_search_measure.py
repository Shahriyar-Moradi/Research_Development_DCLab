"""34.7's measurement (search_v2): the four gates judge made-up results as they say, and a resume on changed code is
refused (the review of search_v1: an unpinned run could mix two engines in one campaign)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy  # noqa: F401

    from dclab_rnd import search_measure as sm
except ImportError as error:  # pragma: no cover
    MISSING = str(error)
else:
    MISSING = ""


def row(name, point, low, high, selection_gap, search_gap, minutes=2.0, skipped=None):
    return {"dataset": name, "difference": {"point": point, "low": low, "high": high},
            "gaps": {"selection_minus_holdout": selection_gap, "search_minus_holdout": search_gap},
            "search": {"minutes": minutes, "skipped": skipped}, "without": {"seconds": 60}, "with": {"seconds": 240}}


@unittest.skipIf(MISSING, MISSING)
class SearchMeasureTest(unittest.TestCase):
    def test_the_gates(self):
        good = [row(f"d{i}", 0.004, -0.002, 0.01, 0.003, 0.01) for i in range(len(sm.DATASETS))]
        verdict = sm.judge(good)
        self.assertEqual(verdict["verdict"], "pass")
        worse = good[:-2] + [row("w1", -0.02, -0.03, -0.01, 0.003, 0.01), row("w2", -0.02, -0.03, -0.01, 0.003, 0.01)]
        self.assertFalse(sm.judge(worse)["gates"]["never_clearly_worse"])  # two clearly worse
        flat = [row(f"d{i}", -0.001, -0.01, 0.01, 0.003, 0.01) for i in range(len(sm.DATASETS))]
        self.assertFalse(sm.judge(flat)["gates"]["rises_on_average"])
        flattered = [row(f"d{i}", 0.004, -0.002, 0.01, 0.02, 0.01) for i in range(len(sm.DATASETS))]
        self.assertFalse(sm.judge(flattered)["gates"]["honest_selection"])  # the selection fold no closer to the holdout
        slow = good[:-1] + [row("slow", 0.004, -0.002, 0.01, 0.003, 0.01, minutes=1.5 * sm.BUDGET["minutes"] + 0.1)]
        self.assertFalse(sm.judge(slow)["gates"]["bounded"])
        self.assertTrue(sm.judge(good[:3])["verdict"].startswith("not yet"))

    def test_a_resume_on_changed_code_is_refused(self):
        folder = Path(tempfile.mkdtemp()) / "search_v2"
        with mock.patch.object(sm, "CAMPAIGN", folder):
            sm.freeze()
            sm.freeze()  # the same code: a resume
            with mock.patch.object(sm, "CODE_AT_START", "0" * 64), self.assertRaises(SystemExit):
                sm.freeze()
            with mock.patch.object(sm, "BUDGET", {**sm.BUDGET, "trials": 99}), self.assertRaises(SystemExit):
                sm.freeze()


if __name__ == "__main__":
    unittest.main()
