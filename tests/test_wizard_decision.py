"""34.4's wizard fields (the review of the change): the proposal carries the amounts Home heard for a yes/no outcome
only; a person who unticks the decision in the wizard has declined it, and the build never adds the heard amounts back;
a caller that never sent the field still gets them added, as before."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd.draft import api

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)

SAID = {"costs": "A call costs 5 euro; a lost customer costs 300."}


@unittest.skipIf(MISSING, MISSING)
class WizardDecisionTest(unittest.TestCase):
    def test_the_proposal_carries_the_heard_amounts_for_a_yes_no_outcome_only(self):
        heard = api.proposal_decision(SAID, "binary")
        self.assertEqual((heard["heard_in"], heard["cost_per_action"], heard["cost_false_negative"]), ("costs", 5.0, 300.0))
        self.assertIsNone(api.proposal_decision(SAID, "regression"))
        self.assertIsNone(api.proposal_decision({}, "binary"))

    def test_an_untick_is_a_decline_and_the_build_honours_it(self):
        stated, heard_in = api._decision_said(SAID)
        self.assertTrue(api.declined({"target": "y", "decision": None}, "human"))
        self.assertFalse(api.declined({"target": "y"}, "human"))  # the field never sent: nothing said
        self.assertFalse(api.declined({"target": "y", "decision": None}, "agent"))  # DCLab's own save is no person's decline
        self.assertFalse(api.heard_to_add({"decision_declined": True}, stated, heard_in))
        self.assertTrue(api.heard_to_add({}, stated, heard_in))
        self.assertFalse(api.heard_to_add({}, stated, "action"))  # heard in the answer on what is done: never added on its own


if __name__ == "__main__":
    unittest.main()
