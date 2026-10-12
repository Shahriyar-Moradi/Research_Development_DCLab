"""Package 16.9 through the engine: the fraud replica gets a review budget from the cost of a review; the ride-pricing
replica gets a price curve with an interval; with fees set by a formula it gets the "cannot learn" warning instead."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import kinds
    from dclab_rnd.kinds import acceptance
except ImportError as error:  # pydantic is not part of requirements/ci.txt
    MISSING = str(error)
else:
    MISSING = ""

PAY = {"action": "price", "price_column": "fee", "reference_column": "usual_fee", "side": "pay", "unit_value_factor": 1.6}


def finish(case):
    from dclab_rnd.expansion import runner
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import engine

    root = Path(tempfile.mkdtemp())
    store, pid = acceptance.build(case, root)
    with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
        engine.run_all(store, pid)
    return store, pid, store.read_stage(pid, "final")


def pricing(varied):
    from dclab_core.replicas import ride_pricing as rp

    replica = rp.generate(0, n=4000, varied=varied)
    return acceptance.Case(replica.offers, "accepted", "binary", "When the fee is offered to a courier, before the courier answers.",
                           {"positive_label": "1", "decision": PAY}, facts=replica.facts)


@unittest.skipIf(MISSING, MISSING)
class DecisionTest(unittest.TestCase):
    def test_the_fraud_replica_gets_a_review_budget_from_the_cost_of_a_review(self):
        case = kinds.get("rare_event").case(0)
        case.solution = {**case.solution, "decision": {"action": "review", "value_per_case": 60, "cost_per_review": 1.5}}
        _, _, final = finish(case)
        decision = final["evidence"]["decision"]
        self.assertEqual(decision["decision"], "review_budget")
        self.assertGreater(decision["share"], 0)
        self.assertLess(decision["share"], 0.1)  # a rare event: a small share is worth reviewing
        lo, hi = decision["expected_value_per_row_interval"]
        self.assertLessEqual(lo, decision["expected_value_per_row"] + 1e-9)
        self.assertIn("entities", decision["interval"])  # bootstrapped over bikers, the unit the folds keep apart
        self.assertIn("lower bound", decision["partial_note"])
        self.assertIn("cases_found", decision["holdout"])
        claim = next(c for c in final["claims"] if c["claim_id"].endswith("-D1"))
        self.assertIn("per case found", claim["statement"])

    def test_a_varied_fee_gets_a_price_curve_with_an_interval(self):
        case = pricing(True)
        store, pid, final = finish(case)
        decision = final["evidence"]["decision"]
        self.assertTrue(decision["learnable"], decision.get("warning"))
        lo, hi = decision["best_multiple_interval"]
        truth = case.facts["best_multiple"]
        self.assertLessEqual(lo - 0.05, truth)
        self.assertLessEqual(truth, hi + 0.05)
        self.assertEqual(store.read_stage(pid, "data")["evidence"]["column_roles"]["accepted_at"], "blocked")  # the copy of the label
        self.assertIn("not read", decision["holdout"])

    def test_a_formula_fee_gets_the_cannot_learn_warning(self):
        _, _, final = finish(pricing(False))
        decision = final["evidence"]["decision"]
        self.assertFalse(decision["learnable"])
        self.assertIn("Cannot learn how acceptance responds to price", decision["warning"])
        claim = next(c for c in final["claims"] if c["claim_id"].endswith("-D1"))
        self.assertEqual(claim["kind"], "risk")

    def test_a_decision_is_checked_when_the_solution_is_written(self):
        from dclab_rnd.studio.solution import Solution

        base = {"target": "y", "task": "binary", "prediction_moment": "Before the outcome is known"}
        with self.assertRaises(ValueError):
            Solution(**base, decision={"action": "review", "value_per_case": 10})  # no cost of a review
        with self.assertRaises(ValueError):
            Solution(**base, decision={"action": "price"})  # no price column
        with self.assertRaises(ValueError):
            Solution(**{**base, "task": "regression"}, decision={"action": "review", "value_per_case": 10, "cost_per_review": 1})
        with self.assertRaises(ValueError):
            Solution(**base, decision={"action": "price", "price_column": "p", "side": "pay"})  # paying needs a value
        self.assertEqual(Solution(**base, decision=PAY).decision.side, "pay")
        with self.assertRaises(ValueError):  # a forbidden column never feeds the acceptance model, even as a reference
            Solution(**base, forbidden=[{"column": "final_customer_fare"}],
                     decision={"action": "price", "price_column": "fee", "reference_column": "final_customer_fare"})

    def test_a_review_that_does_not_pay_is_said_plainly(self):
        from dclab_rnd.studio import engine

        claim = engine._decision_claim("X", {"decision": "review_budget", "share": 0.0, "note": "No review pays: at a cost of 5 per review ...",
                                             "assumption": "a review finds the case"})
        self.assertEqual(claim["kind"], "risk")
        self.assertIn("No review pays", claim["statement"])

    def test_a_reference_set_aside_is_refused_with_its_role(self):
        case = pricing(True)
        case.solution = {**case.solution, "identifiers": ["usual_fee"]}  # the reference set aside as an identifier
        _, _, final = finish(case)
        decision = final["evidence"]["decision"]
        self.assertFalse(decision["learnable"])
        self.assertIn("usual_fee", decision["warning"])


if __name__ == "__main__":
    unittest.main()
