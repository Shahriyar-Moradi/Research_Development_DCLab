"""A capacity per day, filled day by day on the holdout (the plan's item 20 and 34.2's daily capacity for targeting).

A team that reviews 5 rides a day cannot review 15 on a busy day and none on a quiet one: given each holdout row's day,
the decision fills each day's own slots with that day's best rows. Only a holdout that holds its days whole (a time
split) is read so; a day partly before the holdout gets its part of the slots. The holdout-sized samples of the
prediction interval are cut to the holdout's rows and into days of its sizes, so the interval and the reading agree.
Targeting accepts a capacity per day the same way.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np

    from dclab_core import decide
    from dclab_core.decide import DecisionSpec

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


def training(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 6, n)
    return (rng.random(n) < p).astype(int), p


def busy_holdout(seed=1):
    """Three days of 100 rows; day 0 holds every high score (a busy fraud day)."""
    rng = np.random.default_rng(seed)
    p = np.r_[rng.uniform(0.6, 0.95, 100), rng.uniform(0.02, 0.4, 200)]
    y = (rng.random(300) < p).astype(int)
    days = np.repeat(["2026-03-01", "2026-03-02", "2026-03-03"], 100)
    return y, p, days


@unittest.skipIf(MISSING, MISSING)
class DailyCapacityTest(unittest.TestCase):
    def setUp(self):
        self.y, self.p = training()
        self.yt, self.pt, self.days = busy_holdout()
        self.spec = DecisionSpec(action="review", value_per_case=100, cost_per_action=1, capacity_per_day=5)

    def run_it(self, **kw):
        return decide.choose_operating_point(self.y, self.p, self.spec, rows_per_day=100, y_test=self.yt, score_test=self.pt,
                                             bootstrap=60, **kw)

    def test_each_day_gets_its_own_slots(self):
        pooled = self.run_it()
        daily = self.run_it(days_test=self.days)
        self.assertEqual(pooled["holdout"]["actions"], 15)  # one pool: the 15 best rows, all from the busy day
        held = daily["holdout"]
        self.assertEqual(held["actions"], 15)
        self.assertEqual(held["days"]["days"], 3)
        self.assertEqual(held["days"]["slots"], 15)
        self.assertEqual(held["days"]["most_in_a_day"], 5)  # never more than a day's slots
        self.assertLess(held["expected_value_per_row"], pooled["holdout"]["expected_value_per_row"])  # the busy day's extra cases are out of reach
        self.assertIn("day by day", daily["applied_as"])
        self.assertIn("one pool", pooled["applied_as"])
        self.assertIn("days of the holdout's sizes", held["prediction_interval_method"])

    def test_a_day_partly_before_the_holdout_gets_its_part_of_the_slots(self):
        out = self.run_it(days_test=self.days, day_rows={"2026-03-01": 200, "2026-03-02": 100, "2026-03-03": 100})
        self.assertEqual(out["holdout"]["days"]["slots"], 12)  # half of the first day's 5, then 5 and 5: 2, 5, 5
        self.assertEqual(out["holdout"]["days"]["partial_days"], 1)
        uneven = decide.choose_operating_point(self.y, self.p, self.spec.model_copy(update={"capacity_per_day": 12.5}), rows_per_day=500,
                                               y_test=np.r_[self.yt, self.yt[:100]], score_test=np.r_[self.pt, self.pt[:100]],
                                               days_test=np.r_[self.days, np.repeat("2026-03-04", 100)], bootstrap=0)
        self.assertEqual(uneven["holdout"]["days"]["slots"], 50)  # 12, 13, 12, 13: a half slot is never lost nor made up

    def test_a_share_cap_beside_a_daily_cap_is_spread_over_the_days(self):
        """The review of 0.115.0: a share of each small day alone rounded every day to no slot (5% of 10 rows is 0)."""
        rng = np.random.default_rng(3)
        p = rng.beta(1, 6, 300)
        both = DecisionSpec(action="review", value_per_case=100, cost_per_action=1, capacity_per_day=5, capacity_share=0.05)
        out = decide.choose_operating_point(self.y, self.p, both, rows_per_day=10, y_test=(rng.random(300) < p).astype(int),
                                            score_test=p, days_test=np.repeat(np.arange(30), 10), bootstrap=0)
        self.assertEqual(out["holdout"]["days"]["slots"], 15)  # 5% of 300 rows, half a slot a day: 0, 1, 0, 1, ...
        self.assertEqual(out["holdout"]["actions"], 15)
        self.assertEqual(out["holdout"]["days"]["most_in_a_day"], 1)
        small = DecisionSpec(action="review", value_per_case=100, cost_per_action=1, capacity_per_day=0.5, capacity_share=0.05)
        sizes = [20] + [10] * 9  # the re-review: each cap spread alone gave its slots on other days, and no day kept one
        q = rng.beta(1, 6, sum(sizes))
        out = decide.choose_operating_point(self.y, self.p, small, rows_per_day=10, y_test=(rng.random(len(q)) < q).astype(int),
                                            score_test=q, days_test=np.repeat(np.arange(10), sizes), bootstrap=0)
        self.assertEqual(out["holdout"]["days"]["slots"], 5)

    def test_the_interval_is_cut_into_days_too(self):
        """A busy day of 280 rows and two of 10: one pool's 15 best rows sit mostly on the busy day, while day by day
        takes 5 there and 5 from each small day's few rows. The samples are the same rows either way (their own stream),
        so the interval must move down with the days."""
        rng = np.random.default_rng(4)
        p = rng.beta(1, 6, 300)
        y = (rng.random(300) < p).astype(int)
        uneven = np.r_[np.repeat("d1", 280), np.repeat("d2", 10), np.repeat("d3", 10)]
        run = lambda **kw: decide.choose_operating_point(self.y, self.p, self.spec, rows_per_day=100, y_test=y, score_test=p,  # noqa: E731
                                                         bootstrap=60, **kw)["holdout"]
        pooled, daily = run()["prediction_interval"], run(days_test=uneven)
        self.assertTrue(all(v is not None for v in pooled + daily["prediction_interval"]))
        self.assertLess(sum(daily["prediction_interval"]), sum(pooled))
        self.assertIn("days of the holdout's sizes", daily["prediction_interval_method"])

    def test_whole_entities_are_cut_to_the_holdout_rows_before_days(self):
        """The review of 0.115.0: entities of 80 rows ran past the 300 holdout rows and piled onto the last day."""
        groups = np.repeat(np.arange(len(self.y) // 80 + 1), 80)[: len(self.y)]
        out = self.run_it(days_test=self.days, groups=groups)["holdout"]
        self.assertIn("(300 rows, about 300 drawn", out["prediction_interval_method"])

    def test_days_are_refused_where_they_would_be_ignored_or_unreadable(self):
        share = DecisionSpec(action="review", value_per_case=100, cost_per_action=1, capacity_share=0.05)
        with self.assertRaises(ValueError):  # a share needs no days: they would be read by nobody
            decide.choose_operating_point(self.y, self.p, share, y_test=self.yt, score_test=self.pt, days_test=self.days, bootstrap=0)
        with self.assertRaises(ValueError):
            self.run_it(days_test=self.days[:-1])
        with self.assertRaises(ValueError):
            self.run_it(days_test=np.r_[self.days[:-1], None])  # a row with no day cannot be placed
        with self.assertRaises(ValueError):
            self.run_it(days_test=self.days, day_rows={"2026-03-01": 0})
        with self.assertRaises(ValueError):  # a day missing from the rows of each day (keyed otherwise) is never read as whole
            self.run_it(days_test=self.days, day_rows={"2026-03-01": 200, "2026-03-02": 100})
        with self.assertRaises(ValueError):
            self.run_it(day_rows={"2026-03-01": 100})  # a day's rows with no days
        with self.assertRaises(ValueError):
            decide.choose_operating_point(self.y, self.p, self.spec, rows_per_day=100, days_test=self.days, bootstrap=0)  # no holdout


def campaign(n, seed, high_first=False):
    rng = np.random.default_rng(seed)
    score = np.sort(rng.random(n))[::-1] if high_first else rng.random(n)
    treated = rng.integers(0, 2, n)
    lift = np.clip(score - 0.5, 0, None) * 0.4
    outcome = (rng.random(n) < 0.05 + treated * lift).astype(int)
    return score, treated, outcome


@unittest.skipIf(MISSING, MISSING)
class DailyTargetingTest(unittest.TestCase):
    def test_targeting_takes_a_capacity_per_day(self):
        s, t, y = campaign(4000, 0)
        st, tt, yt = campaign(600, 1, high_first=True)  # the best-scored customers all arrive on the first day
        days = np.repeat(["d1", "d2", "d3"], 200)
        spec = DecisionSpec(action="send the offer", value_per_case=40, cost_per_action=2, capacity_per_day=20)
        with self.assertRaises(ValueError):
            decide.target_by_uplift(s, t, y, spec, bootstrap=0)  # a daily cap with no rows a day cannot hold
        out = decide.target_by_uplift(s, t, y, spec, rows_per_day=200, score_test=st, t_test=tt, y_test=yt, days_test=days, bootstrap=0)
        self.assertLessEqual(out["share"], 0.1 + 1e-9)  # 20 of 200 a day
        self.assertEqual(out["capacity_share"], 0.1)
        self.assertEqual(out["holdout"]["days"]["most_in_a_day"], 20)
        self.assertLessEqual(out["holdout"]["treated"], 60)
        pooled = decide.target_by_uplift(s, t, y, spec, rows_per_day=200, score_test=st, t_test=tt, y_test=yt, bootstrap=0)
        self.assertNotIn("days", pooled["holdout"])
        again = decide.targeting_what_if(s, t, y, spec, {"cost_per_action": 5.0}, rows_per_day=200)
        self.assertLessEqual(again["after"]["share"], 0.1 + 1e-9)
        with self.assertRaises(ValueError):
            decide.target_by_uplift(s, t, y, DecisionSpec(action="x", value_per_case=40, cost_per_action=2, capacity_share=0.1),
                                    score_test=st, t_test=tt, y_test=yt, days_test=days, bootstrap=0)


if __name__ == "__main__":
    unittest.main()
