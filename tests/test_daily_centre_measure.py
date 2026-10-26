"""daily_centre_v1: the decision interval's centre on invented worlds with a small or a large sampling pool, and the
gate's next step."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy  # noqa: F401

    from dclab_rnd import daily_centre_measure as dc

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class DailyCentreTest(unittest.TestCase):
    def test_a_world_is_read_in_each_arm_and_arms_share_the_world(self):
        """A world outside the measured ones (never a peek at the campaign's worlds)."""
        self.assertNotIn(7, dc.WORLDS)
        self.assertFalse(set(dc.WORLDS) & set(range(1000, 1200)))  # fresh beside daily_ties_v1
        rows = {arm: dc.world(7, arm) for arm in dc.ARMS}
        for arm, r in rows.items():
            self.assertIsNotNone(r["z_mid"], arm)
            self.assertIsNotNone(r["z_mean"], arm)  # 0.120.1: the samples' mean beside the midpoint
        for latest, every in dc.PAIRS.values():  # the same holdout, read the same way; only the samples' pool differs
            self.assertEqual(rows[latest]["realised"], rows[every]["realised"])
            self.assertNotEqual(rows[latest]["interval"], rows[every]["interval"])

    def test_each_decision_gets_its_own_verdict_in_the_gate_s_order(self):
        def arm(bias, by_mean, skew):
            return {"mean_z_mid_interval": bias, "mean_z_mean_interval": by_mean, "skew_interval": skew}

        self.assertEqual(dc.verdict(arm([-0.05, 0.07], [-0.1, 0.1], [0.0, 0.1]), None), "closed: no bias beyond 0.08")
        self.assertEqual(dc.verdict(arm([-0.02, 0.14], [-0.1, 0.1], [0.0, 0.1]), None), "inconclusive: more worlds")  # never closed by failing to reject
        self.assertEqual(dc.verdict(arm([0.04, 0.2], [-0.05, 0.08], [0.06, 0.12]), [0.1, 0.3]),
                         "the midpoint (the samples' skew): centre the interval on the samples' mean")
        self.assertEqual(dc.verdict(arm([0.04, 0.2], [-0.01, 0.18], [-0.01, 0.02]), [0.1, 0.3]),
                         "read the worlds")  # the review: z_mean reaching 0 is not a skew when the mean sits where the midpoint does
        self.assertEqual(dc.verdict(arm([0.04, 0.2], [0.03, 0.18], [-0.01, 0.02]), [-0.3, -0.05]),
                         "the small pool: draw each sample from the latest fold's own rows")
        self.assertEqual(dc.verdict(arm([-0.2, -0.04], [-0.18, -0.03], [-0.02, 0.01]), [-0.3, -0.05]),
                         "read the worlds")  # a bias below 0 is never the pool's
        self.assertEqual(dc.verdict(arm([0.04, 0.2], [0.03, 0.18], [-0.01, 0.02]), [-0.1, 0.2]), "read the worlds")

    def test_the_report_reads_daily_and_share_apart(self):
        """The review: report() itself sends daily to daily_latest and share to share_latest."""
        import json

        rng = __import__("numpy").random.default_rng(3)
        scratch = Path(tempfile.mkdtemp())
        folder = scratch / "c" / "results"
        folder.mkdir(parents=True)
        worlds = tuple(range(60))
        for i, arm in enumerate(dc.ARMS):
            shift = 0.4 if arm == "share_latest" else 0.0  # only share's latest arm sits off
            rows = [{"world": w, "arm": arm, "z_mid": float(rng.normal(shift, 0.1)), "z_mean": float(rng.normal(0, 0.1)),
                     "midpoint": 5.0, "samples_mean": 5.0} for w in worlds]
            (folder / f"DCE-{i + 1:03d}_{arm}.json").write_text(json.dumps({"arm": arm, "worlds": rows}))
        with mock.patch.object(dc, "CAMPAIGN", scratch / "c"), mock.patch.object(dc, "WORLDS", worlds):
            out = dc.report()
        self.assertEqual(out["verdicts"]["daily"], "closed: no bias beyond 0.08")
        self.assertNotEqual(out["verdicts"]["share"], "closed: no bias beyond 0.08")

    def test_the_intervals_of_means_and_the_centre_gap(self):
        low, high = dc.mean_interval([0.3] * 50)
        self.assertAlmostEqual(low, 0.3, places=6)
        self.assertIsNone(dc.mean_interval([0.3] * 5))
        latest = [{"world": w, "samples_mean": 5.0} for w in range(50)]
        every = [{"world": w, "samples_mean": 5.4} for w in range(50)]
        self.assertAlmostEqual(dc.centre_gap(latest, every)[1], -0.4, places=6)

    def test_a_changed_gate_is_refused(self):
        import json

        scratch = Path(tempfile.mkdtemp())
        (scratch / "c").mkdir()
        (scratch / "c" / "manifest.json").write_text(json.dumps({**dc.manifest(), "sha256": "0" * 64}))
        with mock.patch.object(dc, "CAMPAIGN", scratch / "c"), self.assertRaises(SystemExit):
            dc.freeze()

    def test_the_verdict_is_written_once(self):
        import json

        scratch = Path(tempfile.mkdtemp())
        with mock.patch.object(dc, "CAMPAIGN", scratch / "c"), mock.patch.object(dc, "WORLDS", (7, 8)):
            for arm in dc.ARMS:
                dc.run_arm(arm)
            self.assertEqual(dc.report()["verdict"], "complete")
            self.assertEqual(json.loads((scratch / "c" / "results" / "DCE-100_decision.json").read_text())["id"], "DCE-100")


if __name__ == "__main__":
    unittest.main()
