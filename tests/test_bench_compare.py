"""DCLab Bench B6 (2026-11-03): two arms paired on the same cases, the difference with its interval, wins and losses, McNemar for
yes/no metrics, unscored rows left out and counted, each comparison recorded once, and a gate judged on the record."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import bench as core  # noqa: E402
from dclab_rnd.bench import compare  # noqa: E402


class PairedTests(unittest.TestCase):
    def test_only_shared_cases_count_and_the_sign_follows_the_direction(self):
        a = {"p1": 1.0, "p2": 0.0, "p3": 1.0, "p9": 1.0}
        b = {"p1": 1.0, "p2": 1.0, "p3": 1.0}
        m = core.paired(a, b)
        self.assertEqual((m["n"], m["only_a"], m["wins_b"], m["losses_b"], m["ties"]), (3, 1, 1, 0, 2))
        self.assertAlmostEqual(m["difference"], 1 / 3, places=3)
        self.assertEqual(m["mcnemar_p"], 1.0)  # one discordant pair proves nothing
        lower = core.paired({"x": 5.0, "y": 3.0}, {"x": 1.0, "y": 1.0}, higher_is_better=False)
        self.assertEqual(lower["difference"], 3.0)  # fewer failures for B is better for B

    def test_mcnemar_is_the_exact_binomial_on_the_discordant_pairs(self):
        a = {f"p{i}": 0.0 for i in range(10)}
        b = {f"p{i}": 1.0 for i in range(10)}
        self.assertEqual(core.paired(a, b)["mcnemar_p"], round(2 / 2 ** 10, 4))  # 10 of 10 discordant one way

    def test_the_interval_covers_the_difference(self):
        a = {f"p{i}": float(i % 3) for i in range(30)}
        b = {f"p{i}": float(i % 3) + 1.0 for i in range(30)}
        m = core.paired(a, b)
        self.assertEqual(m["interval"], [1.0, 1.0])


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.campaigns = Path(tempfile.mkdtemp())
        rows = lambda arm, vals: [{"id": f"p{i}", "arm": arm, "judged": True, "failed": False, "understanding": v, "route_right": v >= 1}  # noqa: E731
                                  for i, v in enumerate(vals)]
        doc = {"at": "2026-11-01T10:00:00+00:00", "split": "dev", "scores": {
            "plain": rows("plain", [0.5, 0.5, 0.5, 1.0]),
            "case": rows("case", [1.0, 1.0, 0.5, 1.0]) + [{"id": "p9", "judged": False, "understanding": 1.0}]}}
        (self.campaigns / "conversation_exam_v1" / "results").mkdir(parents=True)
        (self.campaigns / "conversation_exam_v1" / "results" / "CEX-900_dev.json").write_text(json.dumps(doc))
        self.out = Path(tempfile.mkdtemp())

    def test_a_comparison_is_recorded_once_with_its_unscored_rows_counted(self):
        gate = Path(tempfile.mkdtemp()) / "gate.json"
        gate.write_text(json.dumps({"metric": "understanding", "min_difference": 0.1, "lower_bound_above": 0}))
        path = compare.run("conversation_exam_v1/CEX-900_dev", "plain", "case", str(gate), campaigns=self.campaigns, out=self.out)
        doc = json.loads(path.read_text())
        u = doc["metrics"]["understanding"]
        self.assertEqual((u["n"], u["wins_b"], u["left_out_b"]), (4, 2, 1))
        self.assertEqual(doc["verdict"], {"passed": False, "checks": {"difference": True, "lower_bound": False}})  # 4 people prove nothing
        self.assertIn("file_sha256", doc["gate"])
        self.assertEqual(len(list(self.out.glob("*.json"))), 1)

    def test_an_arm_the_result_does_not_have_is_refused(self):
        with self.assertRaises(ValueError):
            compare.run("conversation_exam_v1/CEX-900_dev", "plain", "team", campaigns=self.campaigns, out=self.out)


if __name__ == "__main__":
    unittest.main()
