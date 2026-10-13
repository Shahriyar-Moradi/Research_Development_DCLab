"""Package 16.4: time to event. Censored rows stay censored (dropping them or calling them events is shown to be biased,
on the curve and on the chosen model), training reads no outcome dated after the holdout's first day, the censoring
weights come from the scored rows (the Brier score matches the one computed from the true lifetimes), columns counted
during the follow-up or written when the event happens are found from the data, the estimators agree with
hand-worked examples, and the kind is refused by the engine until the engine runs it."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import survival  # noqa: E402
from dclab_core.replicas import subscription_lifetimes  # noqa: E402

FEATURES = ["plan", "signup_discount", "signup_week_calls", "age"]
ARGS = dict(duration="days_subscribed", event="cancelled", start="signup_date", features=FEATURES, horizons=[90, 180, 365])
ONES = lambda at, before=False: np.ones(len(np.atleast_1d(at)))  # noqa: E731  no censoring: every weight 1


class EstimatorTest(unittest.TestCase):
    def test_kaplan_meier_and_the_c_index_by_hand(self):
        # durations 1,2,3,4,5; the row at 3 is censored: S(1)=4/5, S(2)=3/5, S(4)=3/5*(1/2), S(5)=0
        times, surv = survival.kaplan_meier(np.array([1, 2, 3, 4, 5]), np.array([1, 1, 0, 1, 1]))
        self.assertEqual(list(times), [1, 2, 4, 5])
        np.testing.assert_allclose(surv, [0.8, 0.6, 0.3, 0.0])
        self.assertEqual(float(survival.step(times, surv, [0.5])[0]), 1.0)
        self.assertAlmostEqual(float(survival.step(times, surv, [3])[0]), 0.6)
        d, e = np.array([1, 2, 3, 4]), np.array([1, 1, 0, 1])
        self.assertEqual(survival.c_index(d, e, np.array([4, 3, 2, 1])), 1.0)  # shorter lives, higher risk: every usable pair
        self.assertEqual(survival.c_index(d, e, np.array([1, 2, 3, 4])), 0.0)
        self.assertEqual(survival.c_index(d, e, np.array([4, 3, 0, 1])), 1.0)  # (3, 4): a censored row's shorter time is no pair
        self.assertEqual(survival.c_index(d, e, np.array([1, 1, 1, 1])), 0.5)  # tied risks count a half, never concordant

    def test_cox_with_tied_durations_by_hand(self):
        # no covariate effect: the Breslow baseline hazard is the Nelson-Aalen one, ties counted together
        cox = survival.Cox().fit(np.zeros((4, 1)), np.array([1.0, 1.0, 2.0, 3.0]), np.array([1, 1, 0, 1]))
        self.assertEqual(list(cox.times), [1.0, 3.0])
        np.testing.assert_allclose(cox.cumhaz, [2 / 4, 2 / 4 + 1 / 1])
        # with a covariate: at the fit, the gradient of the Breslow partial likelihood equals the ridge term
        rng = np.random.default_rng(3)
        X, d = rng.normal(size=(40, 2)), rng.integers(1, 8, 40).astype(float)  # many ties
        e = (rng.uniform(size=40) < 0.7).astype(int)
        fit = survival.Cox(ridge=0.1).fit(X, d, e)

        def loglik(b):
            eta = X @ b
            return sum(eta[i] - np.log(np.exp(eta[d >= d[i]]).sum()) for i in range(40) if e[i])
        grad = np.array([(loglik(fit.beta + h) - loglik(fit.beta - h)) / 2e-6 for h in np.eye(2) * 1e-6])
        np.testing.assert_allclose(grad, 0.1 * fit.beta, atol=1e-4)

    def test_cox_recovers_the_planted_effects(self):
        frame = subscription_lifetimes.generate(1).subscribers
        X, = survival._design(frame, [frame], ["signup_discount", "signup_week_calls"])
        cox = survival.Cox().fit(X, frame["days_subscribed"].to_numpy(float), frame["cancelled"].to_numpy(int))
        per_unit = cox.beta / frame[["signup_discount", "signup_week_calls"]].std().to_numpy()  # the design is standardised
        self.assertAlmostEqual(per_unit[0], 0.6, delta=0.15)  # planted: a discount adds 0.6 to the log hazard
        self.assertAlmostEqual(per_unit[1], 0.35, delta=0.1)  # and each signup-week call 0.35

    def test_the_discrete_read_out_inside_an_interval(self):
        h, edges = np.array([[0.5, 0.5]]), np.array([10.0, 20.0])  # constant hazard inside each interval
        np.testing.assert_allclose(survival.read_discrete(h, edges, [5, 10, 15, 20, 30])[0],
                                   [0.5 ** 0.5, 0.5, 0.5 * 0.5 ** 0.5, 0.25, 0.125])

    def test_brier_auc_and_their_integral_by_hand(self):
        d, e = np.array([1.0, 2, 3, 4]), np.array([1, 1, 1, 1])
        self.assertAlmostEqual(survival.brier(d, e, np.array([0.2, 0.4, 0.6, 0.8]), 2.5, ONES), (0.04 + 0.16 + 0.16 + 0.04) / 4)
        self.assertEqual(survival.auc_at(d, e, np.array([4, 3, 2, 1.0]), 2.5, ONES), 1.0)
        S = np.full((4, 5), 0.5)  # every Brier score 0.25: the integral over any span is 0.25
        self.assertAlmostEqual(survival.integrated_brier(d, e, S, np.linspace(1, 3.5, 5), ONES), 0.25)

    def test_the_censoring_weights_recover_the_uncensored_brier_score(self):
        rng = np.random.default_rng(0)
        life, follow = rng.exponential(10, 20000), rng.uniform(0, 25, 20000)
        d, e = np.minimum(life, follow), (life <= follow).astype(int)
        s = np.exp(-6 / 10) * rng.uniform(0.8, 1.2, 20000).clip(0, 1)  # some prediction of survival at t = 6
        true = np.mean(((life > 6).astype(float) - s) ** 2)
        self.assertAlmostEqual(survival.brier(d, e, s, 6.0, survival.censoring_weights(d, e)), true, delta=0.01)
        self.assertGreater(abs(survival.brier(d, e, s, 6.0, ONES) - true), 0.02)  # without the weights it is off

    def test_the_design_is_fitted_on_the_training_rows_only(self):
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "c": ["a", "b", "a"]})
        test = pd.DataFrame({"x": [10.0], "c": ["z"]})
        Xt, Xh = survival._design(train, [train, test], ["x", "c"])
        self.assertAlmostEqual(Xt[:, 0].mean(), 0.0)
        self.assertAlmostEqual(Xh[0, 0], (10 - 2) / 1.0)
        self.assertEqual(list(Xh[0, 1:]), [0.0])  # a level unseen in training is no level


class ReplicaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replica = subscription_lifetimes.generate(0)
        cls.frame = cls.replica.subscribers
        cls.result = survival.evaluate(cls.frame, **ARGS, group="plan")

    def test_dropping_or_counting_censored_rows_is_biased_and_kaplan_meier_is_not(self):
        f, truth = self.frame, self.replica.facts["true_survival"]
        bias = survival.censoring_bias(f["days_subscribed"], f["cancelled"], [90, 180, 365, 5000])
        self.assertGreater(bias["censored_share"], 0.2)
        for h in ("90", "180", "365"):
            row = bias["survival"][h]
            self.assertLess(abs(row["kaplan_meier"] - truth[h]), 0.03, h)
            self.assertLess(row["censored_dropped"], truth[h] - 0.05, h)  # both make subscriptions look shorter
            self.assertLess(row["censored_as_events"], truth[h] - 0.05, h)
        self.assertIsNone(bias["survival"]["5000"]["kaplan_meier"])  # past the longest follow-up nothing is known

    def test_the_chosen_model_fitted_the_wrong_ways_is_biased(self):
        sc = self.result["shortcuts"]
        h = "90"
        observed, honest = sc["observed_on_holdout"][h], sc["mean_survival"]["honest"][h]
        self.assertLess(abs(honest - observed), 0.05)
        for wrong in ("censored_dropped", "censored_as_events"):
            self.assertGreater(abs(sc["mean_survival"][wrong][h] - observed), abs(honest - observed) + 0.05, wrong)

    def test_training_reads_no_outcome_after_the_holdouts_first_day(self):
        r = self.result
        first = (pd.Timestamp(r["split"]["holdout_from"]) - pd.Timestamp(self.frame["signup_date"].min())).days
        start = (pd.to_datetime(self.frame["signup_date"]) - pd.Timestamp(self.frame["signup_date"].min())).dt.days
        changed = self.frame.copy()
        runs_on = (start < first) & (start + changed["days_subscribed"] > first)  # training rows whose outcome lies after the cut
        self.assertGreater(runs_on.sum(), 100)
        changed.loc[runs_on, "days_subscribed"] = (first - start[runs_on] + 1).to_numpy()  # a different future: all leave the next day
        changed.loc[runs_on, "cancelled"] = 1
        again = survival.evaluate(changed, **ARGS, group="plan")
        self.assertEqual(again["holdout"], r["holdout"])
        self.assertEqual(again["validation"], r["validation"])
        self.assertGreater(r["split"]["holdout_from"], r["split"]["train_until"])

    def test_the_holdout_brier_score_matches_the_true_lifetimes(self):
        life = np.array(self.replica.facts["true_lifetime"])[-self.result["split"]["holdout"]:]  # the latest starts, in order
        km = self.result["holdout"]["kaplan_meier"]
        true = np.mean(((life > 90).astype(float) - km["mean_survival"]["90"]) ** 2)
        self.assertAlmostEqual(km["brier"]["90"], true, delta=0.015)

    def test_columns_counted_during_the_follow_up_or_written_at_the_end_are_found(self):
        f = self.frame.copy()
        rng = np.random.default_rng(0)
        f["cancel_reason"] = np.where(f["cancelled"] == 1, rng.choice(["price", "moved"], len(f)), None)
        f["list_price_at_signup"] = 10 + pd.to_datetime(f["signup_date"]).dt.month + 12 * (pd.to_datetime(f["signup_date"]).dt.year - 2024)
        found = survival.duration_check(f, duration="days_subscribed", event="cancelled", exclude=["subscriber_id", "signup_date"])
        self.assertEqual(set(found["past_the_start"]), {"months_billed", "tickets_so_far", "cancel_reason"})
        self.assertIn("follow", found["why"]["tickets_so_far"])
        self.assertNotIn("list_price_at_signup", found["past_the_start"])  # known at signup, only trending with the calendar
        self.assertFalse(survival.censoring_check(pd.Series([1, 1, 1]))["ok"])

    def test_every_model_is_compared_with_kaplan_meier_on_the_latest_starts(self):
        r = self.result
        self.assertNotEqual(r["chosen"], "kaplan_meier")
        self.assertTrue(r["beats_kaplan_meier"][r["chosen"]])
        self.assertGreater(r["holdout"][r["chosen"]]["c_index"], 0.55)
        self.assertEqual(set(r["holdout"]), {"kaplan_meier", "cox", "discrete_logistic", "discrete_boosted"})
        self.assertTrue(r["horizons_reached"])
        self.assertNotIn(365, r["horizons_reached"])  # only horizons the latest starts' follow-up reaches are scored
        self.assertEqual(set(r["curves"]), {"basic", "plus", "pro"})
        self.assertEqual(sum(c["rows"] for c in r["curves"].values()), r["split"]["train"])  # the curves are the training rows'
        self.assertGreater(r["curves"]["pro"]["survival"]["365"], r["curves"]["basic"]["survival"]["365"])

    def test_the_split_refuses_what_it_cannot_order_and_cuts_between_days(self):
        f = self.frame.head(400).copy()
        with self.assertRaises(ValueError):
            survival.evaluate(f.assign(signup_date="2025-01-01"), **ARGS)  # one start day: nothing later to hold out
        with self.assertRaises(ValueError):
            survival.evaluate(f.assign(signup_date=f["signup_date"].where(f.index > 5, "not a date")), **ARGS)
        with self.assertRaises(ValueError):
            survival.evaluate(f.assign(cancelled=0), **ARGS)  # nothing ended: nothing to learn when it happens
        timed = self.frame.copy()
        timed["signup_date"] = timed["signup_date"] + " " + pd.Series(np.random.default_rng(0).integers(0, 24, len(timed))).astype(str).str.zfill(2) + ":30"
        r = survival.evaluate(timed, **ARGS)
        self.assertGreater(r["split"]["holdout_from"], r["split"]["train_until"])


class KindTest(unittest.TestCase):
    def setUp(self):
        try:
            from dclab_rnd import kinds
        except ImportError as error:
            self.skipTest(str(error))
        self.kinds = kinds

    def test_the_kind_is_detected_by_its_words_never_by_a_delivery_or_survival_question(self):
        self.assertEqual(self.kinds.detect("How long until each subscriber cancels?")["key"], "time_to_event")
        self.assertNotEqual(self.kinds.detect("Predict which passengers survived")["key"], "time_to_event")
        self.assertNotEqual(self.kinds.detect("How long until the order is delivered?")["key"], "time_to_event")

    def test_the_acceptance_holds_and_fails_when_the_follow_up_columns_are_kept(self):
        kind = self.kinds.get("time_to_event")
        run = kind.solve(kind.case(0))
        decisions = kind.accept(run)
        self.assertTrue(all(c["ok"] for c in decisions), decisions)
        leaked = {**run, "features": run["features"] + ["months_billed"]}
        self.assertFalse(kind.accept(leaked)[0]["ok"])

    def test_an_export_of_ended_rows_only_is_refused(self):
        kind = self.kinds.get("time_to_event")
        case = kind.case(0)
        ended = case.table[case.table["cancelled"] == 1].reset_index(drop=True)
        run = kind.solve(type(case)(**{**case.__dict__, "table": ended}))
        self.assertIn("refused", run)
        self.assertFalse(any(c["ok"] for c in kind.accept(run)))

    def test_the_engine_runs_the_kind_with_its_duration_as_the_target(self):
        from dclab_rnd.studio.solution import Solution

        fields = {"duration_column": "days_subscribed", "event_column": "cancelled", "start_column": "signup_date"}
        solution = Solution(target="days_subscribed", task="regression", prediction_moment="At signup, before any billing", kind="time_to_event", kind_fields=fields)
        self.assertEqual(solution.problem_kind().key, "time_to_event")  # 16.15: the engine runs its machine
        with self.assertRaises(ValueError) as refused:
            Solution(target="age", task="regression", prediction_moment="At signup, before any billing", kind="time_to_event", kind_fields=fields)
        self.assertIn("is its duration column", str(refused.exception))  # refused for the kind's rule, not another field

if __name__ == "__main__":
    unittest.main()
