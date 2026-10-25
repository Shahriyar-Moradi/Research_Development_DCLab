"""An entity stored as whole numbers is a group candidate (S8, live 2026-10-25): the fraud people who did not name the
biker got no grouping, because `biker_id` (whole numbers, read as floats where a cell is empty) was left out of the
candidates for being numeric, while its phone, as text, came in its place. A row identifier and a measurement with an
id-like name stay out."""

import os
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class EntityTest(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "DCLAB_DECISION_SHADOW": "0"})
        env.start()
        self.addCleanup(env.stop)

    def test_a_repeating_whole_number_id_is_a_group_candidate(self):
        from dclab_rnd.studio.data import profile_table
        from dclab_rnd.studio.solution import _whole_numbers, propose

        rng = np.random.default_rng(0)
        n = 1200
        biker = rng.integers(1, 60, n).astype(float)
        biker[::97] = np.nan  # an empty cell makes the column floats
        frame = pd.DataFrame({"order_id": np.arange(n), "biker_id": biker, "score_id": rng.normal(size=n).round(3),
                              "fare": rng.gamma(2, 3, n), "y": (rng.random(n) < 0.2).astype(int)})
        self.assertTrue(_whole_numbers(frame["biker_id"]))
        self.assertFalse(_whole_numbers(frame["score_id"]))
        got = propose(frame, profile_table(frame), "y", None, None)["group_candidates"]
        self.assertIn("biker_id", got)
        self.assertNotIn("order_id", got)  # unique per row: a row identifier, never a group
        self.assertNotIn("score_id", got)  # a measurement with an id-like name


if __name__ == "__main__":
    unittest.main()
