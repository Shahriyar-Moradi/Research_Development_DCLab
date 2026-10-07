"""Package 14.5: the gate's thresholds and its verdicts."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_rnd.agent_eval import gate  # noqa: E402
from dclab_rnd.agent_eval.run import wilson  # noqa: E402


def summary(caught=38, leaks=38, alarms=0, controls=70, valid=1.0, cites=(70, 70), errors=0, cases=70):
    return {"cases": cases, "errors": errors, "leaks_caught": [caught, leaks], "leaks_caught_ci95": wilson(caught, leaks),
            "false_alarms": [alarms, controls], "valid_moves": valid, "citations_exist": list(cites)}


class GateTests(unittest.TestCase):
    def test_the_thresholds_are_the_ones_written_before_the_runs(self):
        """Moving one is a new gate with a new version, never an edit after a result."""
        self.assertEqual(gate.VERSION, "gate_v1")
        self.assertEqual(gate.THRESHOLDS, {
            "leaks_caught_rate_min": 0.90, "leaks_caught_lower_min": 0.80, "false_alarm_rate_max": 0.10, "valid_moves_min": 1.0,
            "citations_exist_min": 1.0, "family_min_cases": 4, "family_leaks_caught_min": 0.50, "scored_share_min": 1.0})

    def test_a_policy_that_meets_every_criterion_passes(self):
        v = gate.verdict(summary(caught=37, alarms=5))  # 37/38 = 0.974, Wilson lower 0.865; 5/70 = 0.071
        self.assertTrue(v["pass"], v["criteria"])
        self.assertIn("not a claim", v["scope"])

    def test_every_criterion_can_fail_alone(self):
        for label, bad in {
            "leaks": summary(caught=33),                    # 0.868 < 0.90
            "lower bound": summary(caught=35, leaks=38),    # 0.921 but the lower bound is 0.79
            "false alarms": summary(alarms=8),              # 0.114 > 0.10
            "valid moves": summary(valid=0.97),
            "citations": summary(cites=(69, 70)),
            "scored": summary(errors=1),
        }.items():
            v = gate.verdict(bad)
            failed = [c["criterion"] for c in v["criteria"] if not c["pass"]]
            self.assertFalse(v["pass"], label)
            self.assertTrue(failed, label)

    def test_an_aggregate_cannot_hide_a_weak_family(self):
        families = {"identifier": {"leaks_caught": [6, 6]}, "missing_is_outcome": {"leaks_caught": [1, 5]}, "tiny": {"leaks_caught": [0, 1]}}
        v = gate.verdict(summary(caught=37, alarms=5), families)
        self.assertFalse(v["pass"])
        weak = next(c for c in v["criteria"] if c["criterion"] == "no weak family")
        self.assertEqual(weak["value"], ["missing_is_outcome"])  # a family of one case is too small to count

    def test_missing_numbers_fail_instead_of_passing(self):
        v = gate.verdict({"cases": 0, "errors": 0, "leaks_caught": [0, 0], "false_alarms": [0, 0], "valid_moves": None, "citations_exist": [0, 0]})
        self.assertFalse(v["pass"])

    def test_the_scripted_baselines_on_the_sealed_set_do_not_pass(self):
        """BEN-002 (recorded before this gate existed, so its numbers did not shape it): the audit rules keep out 28 of 38."""
        import json
        sealed = json.loads((Path(__file__).resolve().parents[1] / "evidence/campaigns/benchmark_v2/results/BEN-002_test_scripted.json").read_text())
        for policy in ("standard", "audit", "bad"):
            self.assertFalse(gate.verdict(sealed["summary"][policy], sealed["by_family"][policy])["pass"], policy)


if __name__ == "__main__":
    unittest.main()
