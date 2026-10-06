"""The judgment suite (package A4.1): seeded planted traps, the standard plan and two scripted reference policies."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

from dclab_rnd.agent_eval import CASES, run, score  # noqa: E402
from dclab_rnd.agent_eval.cases import by_id  # noqa: E402

# What the suite measured when it was frozen. A change to the audit, the standard plan or the validator moves these;
# when the move is intended, rerun `make agent-eval` (it stores a new result) and update the numbers here.
SCORECARD = {
    "standard": {"leaks_caught": [1, 13], "false_alarms": [1, 20], "unsafe_refused": [0, 0], "valid_moves": 1.0},
    "audit": {"leaks_caught": [10, 13], "false_alarms": [3, 20], "unsafe_refused": [0, 0], "valid_moves": 1.0},
    "bad": {"leaks_caught": [0, 13], "false_alarms": [0, 20], "unsafe_refused": [20, 20], "valid_moves": 0.75},
}

class JudgmentSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run()
        cls.by = {(r["case"], r["policy"]): r for r in cls.report["results"]}

    def test_twenty_seeded_cases_each_with_an_expectation(self):
        self.assertGreaterEqual(len(CASES), 20)
        self.assertEqual(len({c.id for c in CASES}), len(CASES))
        for case in CASES:
            self.assertTrue(case.leaks or case.innocent or case.time_column or case.group_column or case.flag_duplicates, case.id)
            self.assertTrue(case.moment and case.clean_inputs, case.id)  # every case says when it predicts, and what must stay usable
        self.assertEqual({c.suite for c in CASES}, {"leakage", "split", "control"})

    def test_the_cases_are_the_ones_the_stored_run_scored(self):
        stored = sorted((ROOT / "evidence/campaigns/agent_eval_v1/results").glob("AEV-*_scripted.json"))
        self.assertTrue(stored)
        prints = {c["id"]: c["fingerprint"] for c in json.loads(stored[-1].read_text())["cases"]}
        self.assertEqual({c.id: c.fingerprint() for c in CASES}, prints)  # a changed table or expectation needs a new run

    def test_every_policy_is_scored_on_every_case_and_cites_only_real_records(self):
        for case in CASES:
            for policy in ("standard", "audit", "bad"):
                r = self.by[(case.id, policy)]
                self.assertIsNone(r["error"], (case.id, policy))
                self.assertTrue(r["citations_exist"], (case.id, policy))

    def test_the_bad_policy_fails_every_case_that_needs_a_decision_and_its_unsafe_moves_are_refused(self):
        for case in CASES:
            r = self.by[(case.id, "bad")]
            if case.leaks or case.time_column or case.group_column:
                self.assertFalse(r["leak_caught"], case.id)  # it forbids nothing and declares nothing
            self.assertTrue(r["unsafe_refused"], case.id)  # a stage before the solution, a skipped stage
            self.assertNotEqual(r["false_alarm"], True, case.id)
        self.assertTrue(all(self.by[("AE-13", p)]["engine_check"] for p in ("standard", "audit", "bad")))  # the data stage flags repeated rows by itself

    def test_the_score_follows_what_the_engine_really_keeps_out(self):
        case = by_id("AE-05")  # the target as text
        as_text = {"solution": {"task": "binary", "forbidden": [], "identifiers": [], "text_columns": ["label_text"]}}
        self.assertFalse(score(case, as_text, {}, [], [], set())["leak_caught"])  # on a binary task text is modelled: still used
        forbidden = {"solution": {"task": "binary", "forbidden": [{"column": "label_text"}], "identifiers": [], "text_columns": []}}
        self.assertTrue(score(case, forbidden, {}, [], [], set())["leak_caught"])
        greedy = {"solution": {"task": "binary", "forbidden": [{"column": c} for c in ("label_text", "age")], "identifiers": [], "text_columns": []}}
        self.assertTrue(score(case, greedy, {}, [], [], set())["false_alarm"])  # an honest base input kept out, in a leak case too
        as_group = {"solution": {"task": "binary", "forbidden": [{"column": "label_text"}], "identifiers": [], "group_column": "plan"}}
        self.assertTrue(score(case, as_group, {}, [], [], set())["false_alarm"])  # a clean column made the group column is lost too
        self.assertIsNone(score(by_id("AE-13"), forbidden, {}, [], [], set())["leak_caught"])  # repeated rows: an engine check only

    def test_the_scorecard_is_what_was_measured(self):
        for policy, expected in SCORECARD.items():
            got = self.report["summary"][policy]
            self.assertEqual({k: got[k] for k in expected}, expected, f"{policy}: the scorecard moved; rerun make agent-eval and update SCORECARD if intended")

    def test_the_report_says_what_the_scores_are_not(self):
        self.assertIn("not about any model's readiness for production", self.report["limitations"][0])
        self.assertEqual(len(self.report["summary"]["standard"]["leaks_caught_ci95"]), 2)  # Wilson interval: 20 cases are few


if __name__ == "__main__":
    unittest.main()
