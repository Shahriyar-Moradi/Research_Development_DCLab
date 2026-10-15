"""Package 16.7: uplift. The responsive segment is recovered from a randomised campaign (the response model, which
targets who buys anyway, does worse), the Qini curve and AUUC agree with hand-worked values, the model is chosen on
validation, an observational campaign raises the overlap warning and is never called causal, and the kind is refused
by the engine until the engine runs it."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import uplift  # noqa: E402
from dclab_core.replicas import campaign_uplift  # noqa: E402

FEATURES = ["age", "app_user", "past_purchases", "tenure_months", "region"]


class QiniTest(unittest.TestCase):
    def test_the_qini_curve_and_its_coefficient_by_hand(self):
        # four rows: two treated who buy, two controls who do not; the score ranks the treated buyers first
        score, t, y = np.array([4.0, 3, 2, 1]), np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0])
        shares, gains = uplift.qini(score, t, y, points=4)
        np.testing.assert_allclose(gains, [0, 1, 2, 2, 2])  # every treated buyer counts once; the controls bought nothing
        # the curve minus the random line 2 * share is 0, .5, 1, .5, 0: trapezoids of width .25 give .5, per treated row .25
        self.assertAlmostEqual(uplift.qini_coefficient(score, t, y), 0.25)
        self.assertLess(uplift.qini_coefficient(-score, t, y), 0)  # the reverse order is worse than random
        # unequal arms with control buyers: top 3 rows hold 1 treated buyer and 2 controls, one of whom bought
        score, t, y = np.array([6.0, 5, 4, 3, 2, 1]), np.array([1, 0, 0, 1, 1, 0]), np.array([1, 1, 0, 0, 1, 0])
        shares, gains = uplift.qini(score, t, y, points=2)
        np.testing.assert_allclose(gains, [0, 1 - 1 * 1 / 2, 2 - 1 * 3 / 3])  # Y_T - Y_C * N_T / N_C
        weights = np.array([2.0, 1, 1, 1, 1, 1])  # a weighted treated row counts twice
        self.assertAlmostEqual(uplift.qini(score, t, y, weights, points=2)[1][1], 2 - 1 * 2 / 2)

    def test_the_auuc_by_hand(self):
        score, t, y = np.array([4.0, 3, 2, 1]), np.array([1, 0, 1, 0]), np.array([1, 0, 0, 0])
        # top half: treated rate 1, control 0, so the curve is 1 x .5; all rows: .5 - 0, so .5 x 1; the random line .5 x share;
        # curve minus line at shares 0, .5, 1 is 0, .25, 0: the trapezoids give .125
        self.assertAlmostEqual(uplift.auuc(score, t, y, points=2), 0.125)

    def test_the_estimators_parts_by_hand(self):
        x = uplift.XLearner()
        np.testing.assert_allclose(x.mix(np.array([0.2, 0.8]), np.array([1.0, 1.0]), np.array([0.0, 0.0])), [0.2, 0.8])  # e g0 + (1 - e) g1
        z = uplift.transformed_outcome(np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0.25, 0.25, 0.5]))
        np.testing.assert_allclose(z, [1 / 0.25, -1 / 0.75, 0.0])  # Y (T - e) / (e (1 - e))

    def test_the_wording_follows_randomisation_and_overlap(self):
        random_ok = {"propensity_auc": 0.51, "overlaps": True, "imbalanced": [], "inside_band": 1.0}
        self.assertTrue(uplift.wording(True, random_ok)["causal"])
        self.assertFalse(uplift.wording(True, {**random_ok, "propensity_auc": 0.8})["causal"])  # said random, but it is not
        assumed = uplift.wording(False, random_ok)
        self.assertFalse(assumed["causal"])
        self.assertIn("assumption", assumed["say"])
        apart = uplift.wording(False, {**random_ok, "imbalanced": [{"column": "past_purchases"}]})
        self.assertIn("differ on past_purchases", apart["say"])
        self.assertNotIn("do not overlap", apart["say"])  # overlapping but imbalanced is said as such
        poor = uplift.wording(False, {"propensity_auc": 0.97, "overlaps": False, "imbalanced": [{"column": "past_purchases"}], "inside_band": 0.45})
        self.assertFalse(poor["causal"])
        self.assertIn("not effects", poor["say"])
        self.assertEqual(uplift.choose({"t_learner": 0.1, "response_model": 0.5, "random": 0.9}), "t_learner")

    def test_treatment_and_outcome_values_are_read_or_refused(self):
        np.testing.assert_array_equal(uplift.binary(pd.Series(["yes", "no", "Yes"]), "x"), [1, 0, 1])
        np.testing.assert_array_equal(uplift.binary(pd.Series([True, False]), "x"), [1, 0])
        with self.assertRaises(ValueError):
            uplift.binary(pd.Series([1, 2, 1]), "got_offer")


class CampaignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trial = campaign_uplift.generate(0)
        cls.result = uplift.evaluate(cls.trial.customers, treatment="got_offer", outcome="bought", features=FEATURES, randomised=True, boot=100)

    def test_the_responsive_segment_is_recovered_by_every_uplift_learner_and_the_response_model_does_worse(self):
        r = self.result
        responsive = np.array(self.trial.facts["responsive"])[r["holdout_rows"]]
        truth = np.array(self.trial.facts["true_uplift"])[r["holdout_rows"]]

        def top(model):
            s = np.array(r["holdout_uplift"][model])
            return responsive[np.argsort(-s)[: len(s) // 10]].mean()
        for model in ("t_learner", "x_learner", "uplift_tree"):
            self.assertGreater(top(model), 0.75, model)
            self.assertGreater(np.corrcoef(r["holdout_uplift"][model], truth)[0, 1], 0.5, model)
        self.assertGreater(top(r["best"]), top("response_model") + 0.1)
        self.assertAlmostEqual(r["models"][r["best"]]["uplift_by_decile"][0], 0.2, delta=0.06)  # the planted uplift
        self.assertTrue(r["best_beats_random"])
        for key in ("qini_coefficient_ci90", "auuc_ci90"):
            lo, hi = r["models"][r["best"]][key]
            self.assertLess(lo, hi)

    def test_no_effect_means_no_claim(self):
        f = self.trial.customers.copy()
        f["bought"] = np.random.default_rng(5).permutation(f["bought"].to_numpy())  # the offer changes nothing
        r = uplift.evaluate(f, treatment="got_offer", outcome="bought", features=FEATURES, randomised=True, boot=60)
        self.assertFalse(r["best_beats_random"])
        self.assertLessEqual(r["models"][r["best"]]["qini_coefficient_ci90"][0], 0)

    def test_the_model_is_chosen_on_validation_and_the_arms_overlap(self):
        r = self.result
        self.assertEqual(r["best"], uplift.choose(r["validation"]))
        self.assertTrue(set(r["validation_rows"]).isdisjoint(r["holdout_rows"]))
        self.assertEqual(len(r["validation_rows"]), r["split"]["validation"])
        on_holdout = {n: v["qini_coefficient"] for n, v in r["models"].items()}
        self.assertNotEqual(r["validation"], on_holdout)  # the choice was read on other rows than the holdout
        self.assertTrue(r["overlap"]["overlaps"])
        self.assertLess(r["overlap"]["propensity_auc"], 0.6)
        self.assertTrue(r["wording"]["causal"])

    def test_an_observational_campaign_is_never_called_causal(self):
        chosen = campaign_uplift.generate(1, randomised=False)
        r = uplift.evaluate(chosen.customers, treatment="got_offer", outcome="bought", features=FEATURES, randomised=False, boot=50)
        self.assertFalse(r["overlap"]["overlaps"])
        self.assertIn("past_purchases", [i["column"] for i in r["overlap"]["imbalanced"]])
        self.assertFalse(r["wording"]["causal"])
        self.assertIn("not effects", r["wording"]["say"])
        said_random = uplift.evaluate(chosen.customers, treatment="got_offer", outcome="bought", features=FEATURES, randomised=True, boot=20)
        self.assertFalse(said_random["wording"]["causal"])  # saying it was random does not make it so
        responsive = np.array(chosen.facts["responsive"])[r["holdout_rows"]]
        s = np.array(r["holdout_uplift"][r["best"]])
        self.assertTrue(r["weighted"])
        self.assertGreater(responsive[np.argsort(-s)[: len(s) // 10]].mean(), 0.6)  # weighted, the choice is not who was targeted

    def test_a_rule_a_line_cannot_see_is_still_found(self):
        f = self.trial.customers.copy()
        rng = np.random.default_rng(3)
        f["got_offer"] = rng.binomial(1, np.where((f["age"] < 30) | (f["age"] > 62), 0.97, 0.03))  # the youngest and the oldest
        ov = uplift.overlap(f, treatment="got_offer", features=FEATURES)
        self.assertGreater(ov["propensity_auc"], 0.9)
        self.assertFalse(ov["overlaps"])
        self.assertFalse(uplift.wording(True, ov)["causal"])

    def test_one_arm_or_one_outcome_is_refused(self):
        f = self.trial.customers.head(500)
        with self.assertRaises(ValueError):
            uplift.evaluate(f.assign(got_offer=1), treatment="got_offer", outcome="bought", features=FEATURES, randomised=True)
        with self.assertRaises(ValueError):
            uplift.evaluate(f.assign(bought=0), treatment="got_offer", outcome="bought", features=FEATURES, randomised=True)


class KindTest(unittest.TestCase):
    def setUp(self):
        try:
            from dclab_rnd import kinds
        except ImportError as error:
            self.skipTest(str(error))
        self.kinds = kinds

    def test_the_acceptance_holds_without_reading_the_answer(self):
        kind = self.kinds.get("uplift")
        case = kind.case(0)
        run = kind.solve(type(case)(**{**case.__dict__, "facts": {}}))
        decisions = kind.accept({**run, "facts": case.facts})
        self.assertTrue(all(c["ok"] for c in decisions), decisions)
        no_second = {**run, "observational": None, "facts": case.facts}
        self.assertFalse(kind.accept(no_second)[4]["ok"])

    def test_detected_by_its_words_and_run_by_the_engine(self):
        from dclab_rnd.studio.solution import Solution

        self.assertEqual(self.kinds.detect("Find who responds to the discount offer, not who buys anyway (uplift)")["key"], "uplift")
        self.assertNotEqual(self.kinds.detect("Which customers will buy next month?")["key"], "uplift")
        Solution(target="bought", task="binary", prediction_moment="Before the offer", kind="uplift",
                 kind_fields={"treatment_column": "got_offer", "randomised": True})  # 16.15 B: the engine runs it


if __name__ == "__main__":
    unittest.main()
