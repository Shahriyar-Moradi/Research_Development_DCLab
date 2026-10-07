"""Package 14.1: open questions stay open, and a contract must be a valid solution that fits its data."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_rnd.agent_eval import owner_inputs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class OwnerInputsTests(unittest.TestCase):
    def test_the_sheet_ships_with_the_owner_questions_open_except_the_provider_check(self):
        given = owner_inputs.answers()
        self.assertGreaterEqual(len(given), 11)
        self.assertTrue(given["provider.check"].startswith(owner_inputs.RECORDED))  # written from the live runs of 14.2: the owner confirms it
        self.assertEqual(owner_inputs.status()["unconfirmed"], ["provider.check"])  # and the status does not count it as the owner's answer
        for qid in ("hyperack.event", "hyperack.outcome", "churn.snapshot", "churn.action", "review.reviewers", "study.tasks"):
            self.assertEqual(given[qid], "", f"{qid} is the owner's to answer; nothing may fill it in")

    def test_an_unanswered_question_is_open_and_an_answered_one_is_not(self):
        sheet = Path(tempfile.mkdtemp()) / "a.md"
        sheet.write_text("### x.one\nWhat?\n\nANSWER:\n\n### x.two\nWhen?\n\nANSWER: at signup\n")
        s = owner_inputs.status(sheet, Path(tempfile.mkdtemp()))
        self.assertEqual((s["open"], s["answered"]), (["x.one"], ["x.two"]))

    def test_a_contract_must_be_a_valid_solution_that_fits_its_data(self):
        folder = Path(tempfile.mkdtemp())
        telco = "data/project/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        good = {"source": telco, "solution": {"target": "Churn", "task": "binary", "positive_label": "Yes", "identifiers": ["customerID"],
                                              "prediction_moment": "At the month-end snapshot of each customer, before any cancellation request."}}
        (folder / "churn_ok.json").write_text(json.dumps(good))
        (folder / "churn_bad_column.json").write_text(json.dumps({**good, "solution": {**good["solution"], "forbidden": [{"column": "cancellation_date", "reason": "after"}]}}))
        (folder / "churn_short_moment.json").write_text(json.dumps({**good, "solution": {**good["solution"], "prediction_moment": "later"}}))
        (folder / "churn_not_json.json").write_text("{")
        by_name = {c["contract"]: c for c in owner_inputs.status(ROOT / "evaluation/OWNER_ANSWERS.md", folder)["contracts"]}
        self.assertTrue(by_name["churn_ok.json"]["valid"])
        for bad in ("churn_bad_column.json", "churn_short_moment.json", "churn_not_json.json"):
            self.assertFalse(by_name[bad]["valid"], bad)
        self.assertIn("cancellation_date", by_name["churn_bad_column.json"]["why"])
        # a bad contract file is never "ready", even when every question of its source is answered
        sheet = Path(tempfile.mkdtemp()) / "a.md"
        sheet.write_text("### churn.one\nWhen?\n\nANSWER: at month end\n")
        only_bad = Path(tempfile.mkdtemp())
        (only_bad / "churn_bad.json").write_text("{")
        self.assertFalse(owner_inputs.status(sheet, only_bad)["contract_loads"]["churn"])
        (only_bad / "churn_good.json").write_text(json.dumps(good))
        self.assertTrue(owner_inputs.status(sheet, only_bad)["contract_loads"]["churn"])


if __name__ == "__main__":
    unittest.main()
