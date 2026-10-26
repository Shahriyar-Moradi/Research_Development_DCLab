"""daily_ties_v1: the decision core's day by day interval on invented worlds (no engine), and the gate's next step."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy  # noqa: F401
    import scipy  # noqa: F401

    from dclab_rnd import daily_ties_measure as dt

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class DailyTiesTest(unittest.TestCase):
    def test_a_world_is_read_in_each_arm(self):
        """A world outside the measured ones (never a peek at the campaign's worlds)."""
        self.assertNotIn(7, dt.WORLDS)
        for arm in dt.ARMS:
            r = dt.world(7, arm)
            self.assertIsNotNone(r["realised"], arm)
            self.assertEqual(len(r["interval"]), 2)
            self.assertIsNotNone(r["z"], arm)
        tied = dt.world(7, "ties")
        self.assertGreater(tied["actions"], 0)

    def test_the_ties_arm_holds_ties_within_days(self):
        """The review: read the scores the world uses, not a copy of its rounding."""
        import numpy as np

        def tied_days(arm):
            _, score, days = dt._world_rows(7, arm)
            hold = score[dt.TRAIN:]
            tops = [np.sort(hold[days == d])[::-1][:2] for d in np.unique(days)]
            return sum(1 for top in tops if len(top) == 2 and top[0] == top[1])

        self.assertGreater(tied_days("ties"), 5)  # a day's two best share their score
        self.assertEqual(tied_days("control"), 0)

    def test_the_next_step_fixed_with_the_gate(self):
        right = {"width": "right", "rms_z": 1.0, "interval": [0.92, 1.09], "off_centre": False}
        border = {"width": "right", "rms_z": 1.08, "interval": [0.99, 1.17], "off_centre": False}
        narrow = {"width": "narrow", "rms_z": 1.2, "interval": [1.1, 1.31], "off_centre": False}
        arms = lambda **kw: {"control": right, "ties": right, "big_days": right, "share": right, "share_05": right, **kw}  # noqa: E731
        self.assertEqual(dt.next_step(arms(ties=narrow), None), "break ties at random in the samples")
        self.assertEqual(dt.next_step(arms(ties=narrow, big_days=narrow), [-0.05, 0.1]), "read the worlds")  # not "only" ties
        self.assertEqual(dt.next_step(arms(control=narrow, big_days=narrow), [0.05, 0.2]),
                         "resample whole days or draw m of n, a package of its own")
        self.assertEqual(dt.next_step(arms(control=narrow, big_days=narrow), [-0.02, 0.12]), "read the worlds")  # the paired difference spans 0
        self.assertEqual(dt.next_step(arms(big_days=narrow, share_05=narrow), [-0.02, 0.05]), "a small-count correction, a package of its own")
        self.assertEqual(dt.next_step(arms(), None), "read the replica's seeds")
        self.assertEqual(dt.next_step(arms(control=border), None), "inconclusive: more worlds")  # consistent with the replica's 1.16
        self.assertEqual(dt.next_step(arms(share=dict(right, off_centre=True)), None), "read the worlds")  # a centre, not a width

    def test_the_width_reading(self):
        rows = [{"z": 1.2 if i % 2 else -1.2, "inside": i % 10 != 0} for i in range(200)]
        self.assertEqual(dt.judge(rows)["width"], "narrow")
        self.assertEqual(dt.judge([{"z": 1.0 if i % 2 else -1.0, "inside": True} for i in range(200)])["width"], "right")
        self.assertEqual(dt.judge([{"z": 0.7 if i % 2 else -0.7, "inside": True} for i in range(200)])["width"], "wide")

    def test_a_changed_gate_is_refused(self):
        import json

        scratch = Path(tempfile.mkdtemp())
        (scratch / "c").mkdir()
        (scratch / "c" / "manifest.json").write_text(json.dumps({**dt.manifest(), "sha256": "0" * 64}))
        with mock.patch.object(dt, "CAMPAIGN", scratch / "c"), self.assertRaises(SystemExit):
            dt.freeze()

    def test_the_verdict_is_written_once_when_every_arm_is_in(self):
        import json

        scratch = Path(tempfile.mkdtemp())
        with mock.patch.object(dt, "CAMPAIGN", scratch / "c"), mock.patch.object(dt, "WORLDS", (7, 8)):
            for arm in dt.ARMS:
                dt.run_arm(arm)
            out = dt.report()
            self.assertEqual(out["verdict"], "complete")
            written = json.loads((scratch / "c" / "results" / "DTI-100_decision.json").read_text())
            self.assertEqual(written["experiment_id"], "DTI-100")


if __name__ == "__main__":
    unittest.main()
