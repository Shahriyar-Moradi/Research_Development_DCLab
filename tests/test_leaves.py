"""Package 24.5: the leaves. The shared split, the screen, the classifier that reads it, the links; no label of a later period reaches an earlier score."""

from __future__ import annotations

import unittest
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

from dclab_core import leaves as L


def _rides(n: int = 6000, days: int = 30, seed: int = 0):
    """A small table of rides: honest ones, a few that end where they began (marks), and a ring of accounts that share a phone and order from one biker."""
    rng = np.random.default_rng(seed)
    accounts = pd.DataFrame({"account_id": np.arange(1000, 1400), "phone": [f"ph{i}" for i in range(400)], "city": rng.choice(["n", "c", "s"], 400)})
    accounts.loc[1:5, "phone"] = "ph_ring"                       # accounts 1001..1005 share a phone
    accounts.loc[10:11, "phone"] = "ph_family"                   # an honest pair
    acc = rng.integers(1000, 1400, n)
    biker = rng.integers(50, 90, n)
    day = rng.integers(0, days, n)
    ring = rng.random(n) < 0.04
    acc = np.where(ring, rng.integers(1001, 1006, n), acc)
    biker = np.where(ring, 77, biker)
    day = np.where(ring, rng.integers(days // 2, days, n), day)
    dist = np.exp(rng.normal(np.log(2400), 0.5, n))
    dur = dist / 4.5 / 60 + rng.gamma(4, 1, n)
    fare = 1.2 + 0.00085 * dist + rng.normal(0, 0.2, n)
    mark = (rng.random(n) < 0.02) & ~ring
    dist, dur = np.where(mark, rng.uniform(0, 40, n), dist), np.where(mark, rng.uniform(0.3, 2, n), dur)
    cancelled = (rng.random(n) < 0.03) & ~mark & ~ring
    dist, dur, fare = [np.where(cancelled, 0.0, v) for v in (dist, dur, fare)]
    t = pd.Timestamp("2026-01-01") + pd.to_timedelta(day, unit="D") + pd.to_timedelta(rng.integers(0, 24 * 60, n), unit="m")
    frame = pd.DataFrame({"order_id": 9000 + np.arange(n), "created_at": t.strftime("%Y-%m-%d %H:%M:%S"), "account_id": acc, "biker_id": biker,
                          "fare": fare.round(2), "distance_m": dist.round(1), "duration_min": dur.round(2),
                          "pickup_lat": 12.4 + rng.normal(0, 0.02, n)}).sort_values("created_at").reset_index(drop=True)
    return frame, accounts


class PartitionTest(unittest.TestCase):
    def setUp(self):
        self.frame, _ = _rides()

    def test_by_moment_the_holdout_is_the_latest_days_and_no_day_is_cut(self):
        parts = L.partition(self.frame, moment="created_at", share=0.2)
        day = pd.to_datetime(self.frame["created_at"]).dt.floor("D")
        self.assertLess(day.loc[parts.fit].max(), day.loc[parts.holdout].min())
        self.assertLessEqual(day.loc[parts.validation].max(), day.loc[parts.holdout].min())
        self.assertEqual(set(day.loc[parts.holdout]) & set(day.loc[parts.train]), set())
        self.assertEqual(parts.info["by"], "day")

    def test_the_periods_are_disjoint_and_together_every_row(self):
        parts = L.partition(self.frame, moment="created_at")
        sets = [set(parts.fit), set(parts.validation), set(parts.holdout)]
        self.assertEqual(sum(len(s) for s in sets), len(self.frame))
        self.assertEqual(set().union(*sets), set(self.frame.index))
        self.assertEqual(set(parts.train), sets[0] | sets[1])

    def test_by_group_whole_groups_stay_on_one_side(self):
        parts = L.partition(self.frame, keys=["account_id"], share=0.2)
        ids = {p: set(self.frame.loc[parts.of(p), "account_id"]) for p in ("fit", "validation", "holdout")}
        self.assertEqual(ids["fit"] & ids["holdout"], set())
        self.assertEqual(ids["validation"] & ids["holdout"], set())
        self.assertEqual(ids["fit"] & ids["validation"], set())

    def test_the_same_call_gives_the_same_sets_and_a_seed_changes_a_random_draw(self):
        a = L.partition(self.frame, keys=["account_id"], seed=1)
        b = L.partition(self.frame, keys=["account_id"], seed=1)
        c = L.partition(self.frame, keys=["account_id"], seed=2)
        self.assertEqual(list(a.holdout), list(b.holdout))
        self.assertNotEqual(set(a.holdout), set(c.holdout))
        r1, r2 = L.partition(self.frame, seed=3), L.partition(self.frame, seed=3)
        self.assertEqual(list(r1.holdout), list(r2.holdout))

    def test_a_design_that_names_a_moment_is_never_answered_by_another_split(self):
        numeric = self.frame.assign(created_at=pd.to_datetime(self.frame["created_at"]).astype("int64") // 10**9)
        text = self.frame.assign(created_at=self.frame["biker_id"].astype(str) + " is not a date")
        one_day = self.frame.assign(created_at="2026-01-01 10:00:00")
        for name, frame in (("a number", numeric), ("not dates", text), ("one day", one_day)):
            with self.subTest(name):
                with self.assertRaises(L.PartitionError) as cm:
                    L.partition(frame, moment="created_at", keys=["account_id"])
                self.assertIn("created_at", str(cm.exception))
        with self.assertRaises(L.PartitionError):
            L.partition(self.frame, moment="missing_column", keys=["account_id"])

    def test_rows_with_no_usable_time_are_counted_and_too_many_are_refused(self):
        few = self.frame.copy()
        few.loc[few.index[:60], "created_at"] = "not a time"                       # 1% of the rows
        parts = L.partition(few, moment="created_at")
        self.assertEqual(parts.info["no_period"], 60)
        self.assertEqual(len(parts.fit) + len(parts.validation) + len(parts.holdout), len(few) - 60)
        many = self.frame.copy()
        many.loc[many.index[:400], "created_at"] = "not a time"                    # 6.7%: they belong to no period
        with self.assertRaises(L.PartitionError) as cm:
            L.partition(many, moment="created_at")
        self.assertIn("no usable", str(cm.exception))

    def test_a_table_too_short_to_hold_three_periods_is_refused(self):
        with self.assertRaises(L.PartitionError):
            L.partition(self.frame.iloc[:3], share=0.2)


class ColumnsTest(unittest.TestCase):
    def test_measures_leave_out_identifiers_coordinates_and_what_the_host_excludes(self):
        frame, _ = _rides()
        cols = L.measure_columns(frame, exclude=["fare"])
        self.assertEqual(sorted(cols), ["distance_m", "duration_min"])

    def test_confirmed_labels_are_one_for_a_case_and_zero_for_every_other_row(self):
        frame, _ = _rides(200)
        cases = frame[["order_id"]].iloc[[3, 9]]
        y = L.confirmed_labels(frame, cases, "order_id")
        self.assertEqual(int(y.sum()), 2)
        self.assertEqual(set(y.unique()), {0, 1})
        with self.assertRaises(ValueError):
            L.confirmed_labels(frame, cases, "nope")

    def test_link_columns_are_the_ones_a_few_accounts_share(self):
        _, accounts = _rides()
        self.assertEqual(L.link_columns(accounts, "account_id"), ["phone"], "a city is shared by everyone, an id by nobody")

    def test_the_related_table_is_found_by_its_unique_key(self):
        frame, accounts = _rides()
        self.assertEqual(L.find_table(frame, {"accounts": accounts}), ("accounts", "account_id"))
        self.assertIsNone(L.find_table(frame, {"other": pd.DataFrame({"x": [1, 2]})}))


class CaseDatesTest(unittest.TestCase):
    def test_the_column_that_says_when_a_case_was_confirmed_is_found_by_its_name_and_its_dates(self):
        cases = pd.DataFrame({"order_id": [1, 2, 3], "reason": ["a", "b", "c"], "confirmed_at": ["2026-01-05", "2026-01-09", "2026-01-11"]})
        self.assertEqual(L.confirmation_column(cases, ["order_id"]), "confirmed_at")
        self.assertIsNone(L.confirmation_column(cases.drop(columns="confirmed_at"), ["order_id"]))
        self.assertIsNone(L.confirmation_column(cases.assign(confirmed_at=[1, 2, 3]), ["order_id"]), "a number is not a date")
        both = cases.drop(columns="confirmed_at").assign(opened="2026-01-01", verified_on="2026-01-02")
        self.assertEqual(L.confirmation_column(both, ["order_id"]), "verified_on", "a name that says confirmed or verified wins over a bare date")

    def test_each_row_gets_the_date_of_its_case_and_the_earliest_when_listed_twice(self):
        frame = pd.DataFrame({"order_id": [10, 11, 12, 13]})
        cases = pd.DataFrame({"order_id": [11, 13, 11], "confirmed_at": ["2026-02-02", "2026-02-09", "2026-02-01"]})
        dates = L.case_dates(frame, cases, ["order_id"], "confirmed_at")
        self.assertTrue(dates.iloc[[0, 2]].isna().all())
        self.assertEqual(dates.iloc[1], pd.Timestamp("2026-02-01"))
        self.assertEqual(dates.iloc[3], pd.Timestamp("2026-02-09"))

    def test_a_key_of_two_columns_and_a_float_key_are_joined_as_the_analysis_joins_them(self):
        frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"]})
        cases = pd.DataFrame({"a": [1.0], "b": ["y"], "when": ["2026-03-03"]})
        self.assertEqual(list(L.mark_cases(frame, cases, ["a", "b"])), [0, 1, 0])
        self.assertEqual(L.case_dates(frame, cases, ["a", "b"], "when").dropna().index.tolist(), [1])


class ScreenTest(unittest.TestCase):
    def setUp(self):
        self.frame, self.accounts = _rides()
        self.parts = L.partition(self.frame, moment="created_at")
        self.ctx = L.Context(frame=self.frame, parts=self.parts, time="created_at", entities=["account_id", "biker_id"], exclude=["order_id"], budget=0.02)

    def test_every_row_of_every_period_gets_a_score_and_a_row_with_no_measure_ranks_lowest(self):
        out = L.run_leaf(NS(id="blk1", approach="anomaly"), self.ctx)
        for period in L.PERIODS:
            s = out.scores[period]
            self.assertEqual(set(s.index), set(self.parts.of(period)))
            self.assertFalse(s.isna().any())
        empty = (self.frame[["fare", "distance_m", "duration_min"]] == 0).all(axis=1)
        v = out.scores["validation"]
        ev = empty.loc[v.index]
        self.assertTrue(ev.any())
        self.assertLess(v[ev].max(), v[~ev].min() + 1e-9, "a cancelled ride is not scored as unusual: it ranks below every ride")
        self.assertIn("no measure", out.record["left_out"])

    def test_the_marks_rank_above_the_honest_rides_on_the_rows_it_did_not_fit(self):
        out = L.run_leaf(NS(id="blk1", approach="anomaly"), self.ctx)
        h = out.scores["holdout"]
        short = (self.frame.loc[h.index, "distance_m"] < 45) & (self.frame.loc[h.index, "duration_min"] < 2.5) & (self.frame.loc[h.index, "fare"] > 0)
        self.assertGreater(short.sum(), 0)
        self.assertGreater(h[short].median(), h[~short].median())

    def test_it_records_names_and_counts_never_a_value(self):
        rec = L.run_leaf(NS(id="blk1", approach="anomaly"), self.ctx).record
        flat = str(rec)
        self.assertIn("distance_m", rec["columns"])
        for v in self.frame["order_id"].iloc[:5]:
            self.assertNotIn(str(v), flat)

    def test_an_approach_with_no_runner_is_said(self):
        with self.assertRaises(ValueError) as cm:
            L.run_leaf(NS(id="blk1", approach="forecasting"), self.ctx)
        self.assertIn("no runner", str(cm.exception))


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        self.frame, self.accounts = _rides(8000, seed=1)
        short = (self.frame["distance_m"] < 45) & (self.frame["fare"] > 0)
        self.labels = (short & (np.random.default_rng(2).random(len(self.frame)) < 0.6)).astype(int).rename("confirmed")
        self.parts = L.partition(self.frame, moment="created_at")

    def _ctx(self, labels=None):
        return L.Context(frame=self.frame, parts=self.parts, time="created_at", entities=["account_id", "biker_id"], exclude=["order_id"], budget=0.02,
                         labels=self.labels if labels is None else labels)

    def test_without_confirmed_cases_it_is_refused(self):
        with self.assertRaises(ValueError):
            L.run_leaf(NS(id="blk2", approach="rare_event"), L.Context(frame=self.frame, parts=self.parts, time="created_at", entities=[], labels=None))

    def test_too_few_cases_in_the_fit_rows_are_refused_with_the_count(self):
        y = self.labels * 0
        y.loc[self.parts.fit[:3]] = 1
        with self.assertRaises(ValueError) as cm:
            L.run_leaf(NS(id="blk2", approach="rare_event"), self._ctx(y))
        self.assertIn("3 confirmed", str(cm.exception))

    def test_it_reads_the_score_of_the_leaf_it_depends_on_under_the_same_protocol(self):
        ctx = self._ctx()
        ctx.done["blk1"] = L.run_leaf(NS(id="blk1", approach="anomaly"), ctx)
        out = L.run_leaf(NS(id="blk2", approach="rare_event"), ctx, reads=["blk1"])
        self.assertEqual(out.record["reads"], ["blk1"])
        self.assertIn("read_blk1", out.record["columns"])
        self.assertEqual(set(out.scores["holdout"].index), set(self.parts.holdout))

    def test_a_label_of_the_holdout_never_reaches_any_score(self):
        a = L.run_leaf(NS(id="blk2", approach="rare_event"), self._ctx())
        flipped = self.labels.copy()
        flipped.loc[self.parts.holdout] = 1 - flipped.loc[self.parts.holdout]
        b = L.run_leaf(NS(id="blk2", approach="rare_event"), self._ctx(flipped))
        for period in L.PERIODS:
            self.assertTrue(np.allclose(a.scores[period], b.scores[period]), f"the holdout's labels changed the scores of {period}")

    def test_a_label_of_the_validation_rows_never_trains_the_model_that_scores_them(self):
        from unittest import mock

        only = {"logistic": L._models(0)["logistic"]}  # one model: nothing is chosen on validation, so only training could carry its labels into a score
        with mock.patch.object(L, "_models", lambda seed: only):
            a = L.run_leaf(NS(id="blk2", approach="rare_event"), self._ctx())
            flipped = self.labels.copy()
            flipped.loc[self.parts.validation] = 1 - flipped.loc[self.parts.validation]
            b = L.run_leaf(NS(id="blk2", approach="rare_event"), self._ctx(flipped))
        self.assertTrue(np.allclose(a.scores["fit"], b.scores["fit"]))
        self.assertTrue(np.allclose(a.scores["validation"], b.scores["validation"]), "the model that scores validation never saw validation's labels")
        self.assertFalse(np.allclose(a.scores["holdout"], b.scores["holdout"]), "the model that scores the holdout is refitted on the validation rows too")

    def test_a_model_never_learns_a_case_before_it_was_confirmed(self):
        t = pd.to_datetime(self.frame["created_at"])
        cases = self.labels == 1
        start = {"validation": t.loc[self.parts.validation].min(), "holdout": t.loc[self.parts.holdout].min()}
        confirmed_late = pd.Series(pd.NaT, index=self.frame.index, dtype="datetime64[ns]")
        confirmed_late[cases] = t[cases] + pd.Timedelta(days=9)                   # every case is confirmed nine days after its ride
        ctx = self._ctx()
        ctx.known_at = confirmed_late
        out = L.run_leaf(NS(id="blk2", approach="rare_event"), ctx)
        held_a = int((cases & (confirmed_late >= start["validation"]) & self.frame.index.isin(self.parts.fit)).sum())
        self.assertGreater(held_a, 0, "cases near the edge of the fit rows are confirmed after validation began")
        self.assertTrue(any(f"{held_a} case(s) of the fit rows were held back" in l for l in out.record["limits"]))
        # the labels the model that scores validation sees are the same whether or not the late cases were ever marked
        only_known = self.labels.copy()
        only_known[cases & (confirmed_late >= start["validation"]) & self.frame.index.isin(self.parts.fit)] = 0
        plain = self._ctx(only_known)
        plain.labels = only_known
        from unittest import mock

        only = {"logistic": L._models(0)["logistic"]}
        with mock.patch.object(L, "_models", lambda seed: only):
            a = L.run_leaf(NS(id="blk2", approach="rare_event"), ctx)
            b = L.run_leaf(NS(id="blk2", approach="rare_event"), self._ctx(only_known))
        self.assertTrue(np.allclose(a.scores["fit"], b.scores["fit"], atol=1e-9), "a case held back is, to the model, a ride nobody marked")

    def test_all_the_cases_confirmed_after_the_period_began_leave_nothing_to_learn_from(self):
        t = pd.to_datetime(self.frame["created_at"])
        late = pd.Series(t.max() + pd.Timedelta(days=30), index=self.frame.index)
        ctx = self._ctx()
        ctx.known_at = late.where(self.labels == 1)
        with self.assertRaises(ValueError) as cm:
            L.run_leaf(NS(id="blk2", approach="rare_event"), ctx)
        self.assertIn("known by the time validation starts", str(cm.exception))

    def test_cases_with_no_date_are_said_to_carry_none(self):
        out = L.run_leaf(NS(id="blk2", approach="rare_event"), self._ctx())
        self.assertTrue(any("no date of confirmation" in l for l in out.record["limits"]))

    def test_the_values_of_holdout_rows_never_reach_a_score_made_before_it(self):
        ctx = self._ctx()
        ctx.done["blk1"] = L.run_leaf(NS(id="blk1", approach="anomaly"), ctx)
        clf = L.run_leaf(NS(id="blk2", approach="rare_event"), ctx, reads=["blk1"])
        changed = self.frame.copy()
        changed.loc[self.parts.holdout, ["distance_m", "duration_min", "fare"]] *= 7.0   # every number of the holdout rows is different
        ctx2 = L.Context(frame=changed, parts=self.parts, time="created_at", entities=["account_id", "biker_id"], exclude=["order_id"], budget=0.02, labels=self.labels)
        ctx2.done["blk1"] = L.run_leaf(NS(id="blk1", approach="anomaly"), ctx2)
        clf2 = L.run_leaf(NS(id="blk2", approach="rare_event"), ctx2, reads=["blk1"])
        for leaf_a, leaf_b in ((ctx.done["blk1"], ctx2.done["blk1"]), (clf, clf2)):
            for period in ("fit", "validation"):
                self.assertTrue(np.allclose(leaf_a.scores[period], leaf_b.scores[period]), f"holdout values reached the {period} scores of {leaf_a.id}")
        self.assertFalse(np.allclose(clf.scores["holdout"], clf2.scores["holdout"]), "the holdout's own scores do follow its values")

    def test_the_model_it_chooses_was_judged_on_validation(self):
        out = L.run_leaf(NS(id="blk2", approach="rare_event"), self._ctx())
        self.assertIn(out.record["chosen"], out.record["validation"])
        self.assertEqual(out.record["chosen"], max(out.record["validation"], key=lambda k: (out.record["validation"][k], -list(out.record["validation"]).index(k))))


class LinksTest(unittest.TestCase):
    def setUp(self):
        self.frame, self.accounts = _rides(8000, seed=3)
        self.parts = L.partition(self.frame, moment="created_at")

    def _ctx(self, frame=None, tables=None):
        frame = self.frame if frame is None else frame
        return L.Context(frame=frame, parts=L.partition(frame, moment="created_at"), time="created_at", entities=["account_id", "biker_id"], exclude=["order_id"],
                         tables={"accounts": self.accounts} if tables is None else tables)

    def test_rides_of_linked_accounts_to_one_biker_score_above_the_rest(self):
        out = L.run_leaf(NS(id="blk3", approach="entity_tables"), self._ctx())
        s = out.scores["holdout"]
        ring = self.frame.loc[s.index, "biker_id"].eq(77) & self.frame.loc[s.index, "account_id"].between(1001, 1005)
        self.assertGreater(ring.sum(), 20)
        self.assertGreater(s[ring].median(), s[~ring].quantile(0.99))
        self.assertIn("phone", out.record["columns"])

    def test_a_pair_that_shares_a_phone_honestly_does_not_look_like_a_ring(self):
        out = L.run_leaf(NS(id="blk3", approach="entity_tables"), self._ctx())
        s = out.scores["holdout"]
        family = self.frame.loc[s.index, "account_id"].isin([1010, 1011])
        ring = self.frame.loc[s.index, "account_id"].between(1001, 1005)
        self.assertLess(s[family].quantile(0.9), s[ring].median())

    def test_a_later_ride_never_changes_the_score_of_an_earlier_one(self):
        full = L.run_leaf(NS(id="blk3", approach="entity_tables"), self._ctx()).scores
        cut = pd.to_datetime(self.frame["created_at"]).quantile(0.7)
        early = self.frame[pd.to_datetime(self.frame["created_at"]) < cut]
        head = L.run_leaf(NS(id="blk3", approach="entity_tables"), self._ctx(early)).scores
        a = pd.concat([full[p] for p in ("fit", "validation", "holdout")])
        b = pd.concat([head[p] for p in ("fit", "validation", "holdout")])
        # the score of a ride is the same with or without the rides after its day
        day = pd.to_datetime(self.frame["created_at"]).dt.floor("D")
        last = day.loc[early.index].max()
        settled = early.index[day.loc[early.index] < last]
        self.assertTrue(np.allclose(a.loc[settled].sort_index(), b.loc[settled].sort_index()))

    def test_a_later_ride_of_the_same_day_never_changes_the_score_of_an_earlier_one(self):
        base = self.frame.copy()
        base["t"] = pd.to_datetime(base["created_at"])
        day = base["t"].dt.floor("D").max()
        late = base[base["t"].dt.floor("D") == day].iloc[-1:].copy()
        early = base.drop(index=late.index)
        a = L.run_leaf(NS(id="blk3", approach="entity_tables"), self._ctx(early.drop(columns="t"))).scores
        b = L.run_leaf(NS(id="blk3", approach="entity_tables"), self._ctx(base.drop(columns="t"))).scores
        before = pd.concat([a[p] for p in ("fit", "validation", "holdout")])
        after = pd.concat([b[p] for p in ("fit", "validation", "holdout")])
        same = before.index.intersection(after.index)
        mask = base.loc[same, "t"] < late["t"].iloc[0]
        self.assertTrue(np.allclose(before.loc[same][mask], after.loc[same][mask]), "a ride is scored as of its own moment, not as of the end of its day")

    def test_an_account_opened_later_is_not_part_of_the_group_for_an_earlier_ride(self):
        frame = self.frame.copy()
        accounts = self.accounts.copy()
        accounts["signup_date"] = "2025-12-01"
        late = accounts.index[accounts["phone"] == "ph_ring"][2:]                  # three of the five ring accounts open after the first weeks
        accounts.loc[late, "signup_date"] = "2026-01-25"
        ctx = self._ctx(frame, {"accounts": accounts})
        out = L.run_leaf(NS(id="blk3", approach="entity_tables"), ctx)
        self.assertIn("signup_date", out.record["notes"][0])
        t = pd.to_datetime(frame["created_at"])
        first = accounts.loc[accounts["phone"] == "ph_ring", "account_id"].iloc[0]
        early = frame[(frame["account_id"] == first) & (t < "2026-01-20")].index
        later = frame[(frame["account_id"] == first) & (t >= "2026-01-26")].index
        self.assertGreater(len(early), 0)
        self.assertGreater(len(later), 0)
        scores = pd.concat([out.scores[p] for p in ("fit", "validation", "holdout")])
        static = L.run_leaf(NS(id="blk3", approach="entity_tables"), self._ctx(frame, {"accounts": accounts.drop(columns="signup_date")}))
        static_scores = pd.concat([static.scores[p] for p in ("fit", "validation", "holdout")])
        self.assertLess(float(scores.loc[early].mean()), float(static_scores.loc[early].mean()), "before the others opened, the group was smaller")
        self.assertTrue(np.allclose(scores.loc[later], static_scores.loc[later]), "after they opened, the group is the table's")
        self.assertIn("as the table stands", static.record["notes"][0])

    def test_a_placeholder_shared_by_many_accounts_does_not_hide_the_real_groups(self):
        accounts = self.accounts.copy()
        accounts.loc[100:150, "phone"] = "0000000000"                               # 51 accounts with a placeholder
        self.assertEqual(L.link_columns(accounts, "account_id"), ["phone"])
        out = L.run_leaf(NS(id="blk3", approach="entity_tables"), self._ctx(tables={"accounts": accounts}))
        s = out.scores["holdout"]
        placeholder = self.frame.loc[s.index, "account_id"].isin(accounts.loc[100:150, "account_id"])
        ring = self.frame.loc[s.index, "biker_id"].eq(77) & self.frame.loc[s.index, "account_id"].between(1001, 1005)
        self.assertEqual(float(s[placeholder].max()), 0.0, "a value shared by that many links nobody")
        self.assertGreater(s[ring].median(), 0.0)

    def test_a_moment_that_is_not_dates_blocks_the_links_loudly(self):
        frame = self.frame.assign(created_at=pd.to_datetime(self.frame["created_at"]).astype("int64") // 10**9)
        with self.assertRaises(ValueError) as cm:
            L.run_leaf(NS(id="blk3", approach="entity_tables"), self._ctx(frame))
        self.assertIn("does not read as dates", str(cm.exception))

    def test_without_the_related_table_it_says_what_to_attach(self):
        with self.assertRaises(ValueError) as cm:
            L.run_leaf(NS(id="blk3", approach="entity_tables"), self._ctx(tables={}))
        self.assertIn("table of accounts", str(cm.exception))

    def test_no_label_is_used(self):
        ctx = self._ctx()
        ctx.labels = None
        out = L.run_leaf(NS(id="blk3", approach="entity_tables"), ctx)
        self.assertEqual(out.record["validation"], {})


if __name__ == "__main__":
    unittest.main()
