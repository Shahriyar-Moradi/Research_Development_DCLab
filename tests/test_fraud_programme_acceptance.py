"""Package 24.6: the fraud programme's replica, and the decisions DCLab must reach on it (Part 5 of the plan).

``ReplicaTest`` pins the replica itself: deterministic, every kind of fraud planted, each visible to one leaf only, the labels partial and biased, the honest look-alikes
(a family's shared phone, a short delivery next door, a cancelled ride) there to trip a method that is not careful.

``AcceptanceTest`` is what "DCLab can solve this" means for a problem of several parts, on the seeds 0, 1 and 2 (the replica's scale was calibrated on other seeds, 10 to
15): the first sentence becomes the four parts a careful analyst would draw; the router gives each the right approach; the solution map has one split and no leaf scores
the holdout; no identifier, label key or coordinate reaches a leaf; the composition chosen on validation beats the best leaf chosen on validation, with the interval of
the paired difference above zero; and the holdout is read once. The replica's `truth` (which rides were planted) is used here to say whether the decisions are sound,
never by the programme itself.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core import compose, datacard, leaves, problem_map as pm, programme as pg, solution_map as sm_mod, solutions
    from dclab_core.replicas import fraud_programme as fp
except ImportError as error:  # the agent environment lacks pandas or sklearn
    MISSING = str(error)
else:
    MISSING = ""

WORDS = ("Find the fake rides and fake orders our bikers and customers make. We have a few confirmed cases, nothing else is labelled, "
         "and the team can review about 2% of the rides each day.")
SEEDS = (0, 1, 2)
_RUNS: dict = {}


def run(seed: int, with_accounts: bool = True):
    """The whole path from the first sentence to the composed programme, once per seed and process."""
    key = (seed, with_accounts)
    if key not in _RUNS:
        r = fp.generate(seed=seed)
        frame = r.rides.copy()
        tables = {"rides": frame, **({"accounts": r.accounts} if with_accounts else {})}
        card = datacard.card_of(tables, roles={"rides": "main", "accounts": "related"})
        m = pm.decompose(WORDS)
        routes = solutions.routes(m, card)
        sm = sm_mod.solution_map(m, routes, card)
        labels = leaves.mark_cases(frame, r.cases, ["order_id"])
        column = leaves.confirmation_column(r.cases, ["order_id"])
        known_at = leaves.case_dates(frame, r.cases, ["order_id"], column) if column else None
        got = pg.execute(sm, m, frame, tables={"accounts": r.accounts} if with_accounts else {}, labels=labels, known_at=known_at, entities=["account_id", "biker_id"],
                         time="created_at", exclude=["order_id"], budget=fp.BUDGET_SHARE, seed=seed)
        _RUNS[key] = {"replica": r, "frame": frame, "card": card, "map": m, "routes": routes, "solution": sm, "labels": labels, "known_at": known_at, "run": got}
    return _RUNS[key]


@unittest.skipIf(MISSING, MISSING)
class ReplicaTest(unittest.TestCase):
    def test_it_is_deterministic_and_the_three_kinds_of_fraud_are_planted(self):
        a, b = fp.generate(seed=0), fp.generate(seed=0)
        pd.testing.assert_frame_equal(a.rides, b.rides)
        pd.testing.assert_frame_equal(a.accounts, b.accounts)
        kinds = a.truth["kind"].value_counts().to_dict()
        self.assertEqual(set(kinds), {fp.MARK, fp.RING, fp.BOTH})
        self.assertGreater(kinds[fp.RING], kinds[fp.BOTH], "most ring rides are only ring rides")

    def test_the_labels_are_partial_and_biased_towards_the_visible_kind(self):
        r = fp.generate(seed=0)
        truth = r.truth.set_index("order_id")
        self.assertTrue(set(r.cases["order_id"]) <= set(truth.index), "a confirmed case is a planted one")
        share = truth.groupby("kind")["confirmed"].mean()
        self.assertGreater(share[fp.MARK], share[fp.RING], "a fake mark is confirmed more often than a ring order")
        self.assertLess(r.cases["order_id"].nunique() / len(truth), 0.6, "partial: most frauds were never confirmed")
        self.assertEqual(set(r.rides["order_id"]) & set(r.cases["order_id"]), set(r.cases["order_id"]))

    def test_a_ring_ride_looks_like_an_honest_one_and_only_the_accounts_table_gives_it_away(self):
        r = fp.generate(seed=0)
        ring = r.rides[r.rides["order_id"].isin(r.truth.loc[r.truth["kind"] == fp.RING, "order_id"])]
        honest = r.rides[~r.rides["order_id"].isin(r.truth["order_id"]) & (r.rides["status"] == "DELIVERED")]
        for column in ("distance_m", "duration_min", "fare"):
            self.assertLess(abs(ring[column].median() / honest[column].median() - 1), 0.1, f"a ring ride's {column} is an honest one's")
        sizes = r.accounts.groupby("phone")["phone"].transform("size")
        ring_accounts = set(r.truth.loc[r.truth["kind"].isin([fp.RING, fp.BOTH]), "account_id"])
        shared_ring = sizes[r.accounts["account_id"].isin(ring_accounts)]
        self.assertGreaterEqual(float((shared_ring >= 2).mean()), 0.9, "nearly every ring account shares its phone with another")
        self.assertGreaterEqual(int((sizes == 2).sum() // 2), r.facts["honest_shared_phones"], "families and offices share a phone in pairs: a shared phone alone is not a ring")

    def test_a_fake_mark_is_visible_in_the_ride_itself_and_the_traps_are_there(self):
        r = fp.generate(seed=0)
        marks = r.rides[r.rides["order_id"].isin(r.truth.loc[r.truth["kind"] == fp.MARK, "order_id"])]
        self.assertLess(float(marks["distance_m"].median()), 50)
        self.assertLess(float(marks["duration_min"].median()), 3)
        self.assertGreater(float(marks["fare"].median()), 1.5, "for a normal fare")
        cancelled = r.rides[r.rides["status"] == "CANCELLED"]
        self.assertGreater(len(cancelled), 500)
        self.assertTrue((cancelled["distance_m"] == 0).all(), "a cancelled ride ends where it began, for no fare")
        near = r.rides[(r.rides["status"] == "DELIVERED") & (r.rides["distance_m"] < 200) & ~r.rides["order_id"].isin(r.truth["order_id"])]
        self.assertGreater(len(near), len(marks), "honest deliveries next door outnumber the fake marks")

    def test_the_budget_is_two_percent_and_the_files_are_what_a_person_uploads(self):
        r = fp.generate(seed=0)
        self.assertEqual(fp.BUDGET_SHARE, 0.02)
        self.assertEqual(list(r.accounts.columns), ["account_id", "signup_date", "phone", "device_id", "city"])
        self.assertIn("order_id", r.cases.columns)
        self.assertNotIn("kind", r.rides.columns, "the truth never travels with the rides")


@unittest.skipIf(MISSING, MISSING)
class AcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {seed: run(seed) for seed in SEEDS}

    def each(self):
        for seed, r in self.runs.items():
            with self.subTest(seed=seed):
                yield seed, r

    def test_the_first_sentence_becomes_the_four_parts_and_how_they_combine(self):
        for seed, r in self.each():
            m = r["map"]
            self.assertEqual(m.size, "complex")
            self.assertEqual([s.family for s in m.sub_problems], ["anomaly", "rare_event", "entity_tables", "ranking"])
            self.assertEqual([(c.kind, c.inputs, c.output) for c in m.composition.steps],
                             [("fusion", ["sp1", "sp2", "sp3"], "risk"), ("budget", ["risk", "sp4"], "review_list")])
            self.assertIn("2%", m.composition.budget or "", "the budget the person said, with its percent sign")
            self.assertEqual(pm.validate(m), [])

    def test_the_router_gives_each_part_the_right_approach_and_nothing_is_blocked(self):
        for seed, r in self.each():
            self.assertEqual({k: v.chosen for k, v in r["routes"].items()},
                             {"sp1": "anomaly", "sp2": "rare_event", "sp3": "entity_tables", "sp4": "review_budgeted"})
            self.assertEqual(r["solution"].blockers, [])

    def test_the_solution_has_one_split_by_time_and_entity_and_no_leaf_scores_the_holdout(self):
        for seed, r in self.each():
            sm = r["solution"]
            split = sm.holdout.split
            self.assertEqual((split.strategy, split.moment, sorted(split.keys)), ("time_group", "created_at", ["account_id", "biker_id"]))
            self.assertEqual(sm.holdout.read_by, "composition")
            self.assertTrue(all(b.final == "deferred" and b.split == split for b in sm.blocks))
            self.assertEqual(sm_mod.validate(sm, r["map"]), [])

    def test_every_leaf_runs_and_the_classifier_reads_the_screen(self):
        for seed, r in self.each():
            p = r["run"].programme
            self.assertEqual({l.block: l.status for l in p.leaves}, {"blk1": "built", "blk2": "built", "blk3": "built", "blk4": "composition"})
            by = {l.block: l for l in p.leaves}
            self.assertEqual(by["blk2"].reads, ["blk1"])
            self.assertTrue(all(l.holdout == "predicted" for l in p.leaves if l.status == "built"))
            self.assertIn("no measure", by["blk1"].left_out or "", "the cancelled rides are left out of the screen")

    def test_no_identifier_label_key_or_coordinate_reaches_a_leaf(self):
        for seed, r in self.each():
            p = r["run"].programme
            banned = {"order_id", "account_id", "biker_id", "pickup_lat", "pickup_lon", "drop_lat", "drop_lon", "created_at", "status", "category"}
            for leaf in p.leaves:
                if leaf.status != "built" or leaf.approach == "entity_tables":
                    continue
                self.assertFalse(banned & set(leaf.columns), f"{leaf.block} reads {banned & set(leaf.columns)}")
            by = {l.block: l for l in p.leaves}
            self.assertEqual(set(by["blk1"].columns), {"fare", "distance_m", "duration_min"})
            self.assertEqual(set(by["blk2"].columns), {"fare", "distance_m", "duration_min", "read_blk1"})
            self.assertEqual(set(by["blk3"].columns), {"phone", "device_id", "biker_id"}, "the links read the accounts table's contacts and the people who serve them")

    def test_the_composition_chosen_on_validation_beats_the_best_leaf_with_its_interval_above_zero(self):
        for seed, r in self.each():
            p = r["run"].programme
            h = p.holdout
            self.assertEqual(p.state, "composed")
            self.assertNotIn(p.composition.chosen, ("blk1", "blk2", "blk3"), "a combination of the leaves, not one of them alone")
            self.assertEqual(h.best_leaf, p.composition.best_leaf, "the bar is the best leaf on validation")
            self.assertTrue(h.beats_best_leaf, f"seed {seed}: composed {h.composed.value} against {h.best_leaf} {h.best_leaf_estimate.value}, difference {h.difference}")
            self.assertGreater(h.difference.low, 0)
            self.assertGreater(h.difference.value, 0.1, "a gain that matters, not a hair above zero")
            self.assertGreaterEqual(h.positives, 30, "enough confirmed cases for the interval to mean something")
            self.assertEqual(h.cluster_key, "biker_id")

    def test_it_finds_more_of_the_planted_fraud_not_only_of_what_was_confirmed(self):
        for seed, r in self.each():
            p, run_ = r["run"].programme, r["run"]
            truth = r["frame"]["order_id"].isin(r["replica"].truth["order_id"]).astype(int)
            hold = {b: s.scores["holdout"] for b, s in run_.scores.items()}
            k = p.holdout.budget_rows
            at = lambda s: compose.recall_at(s.to_numpy(float), truth.reindex(s.index).to_numpy(), k)
            composed, best = at(compose.build(p.composition.chosen, hold, p.composition.weights)), at(hold[p.composition.best_leaf])
            self.assertGreater(composed, best + 0.1, f"seed {seed}: the planted fraud found {composed:.2f} against {best:.2f} for the best leaf")
            self.assertGreater(composed, 0.8)

    def test_the_classifier_learns_a_case_only_from_the_day_it_was_confirmed(self):
        for seed, r in self.each():
            self.assertEqual(leaves.confirmation_column(r["replica"].cases, ["order_id"]), "confirmed_at")
            clf = next(l for l in r["run"].programme.leaves if l.block == "blk2")
            held = [l for l in clf.limits if "case(s) of the fit rows were held back" in l]
            self.assertTrue(held, "cases confirmed after validation began were held back from the model that scores it")
            self.assertNotIn("no date of confirmation", " ".join(clf.limits))

    def test_the_links_count_the_accounts_that_had_opened_by_each_ride(self):
        for seed, r in self.each():
            links = next(l for l in r["run"].programme.leaves if l.block == "blk3")
            self.assertIn("signup_date", links.notes[0], "the table says when accounts were opened: the groups are counted as of each ride")
            self.assertIn("no later ride is read, not even one of the same day", links.notes[0])

    def test_the_holdout_is_read_once_and_the_programme_holds(self):
        for seed, r in self.each():
            p = r["run"].programme
            self.assertEqual(p.holdout.reads, 1)
            self.assertEqual(pg.validate(p, r["solution"]), [])
            self.assertEqual(pg.open_issues(p), ["the holdout was already read for this programme: it is read once"])

    def test_cancelled_rides_and_honest_pairs_do_not_fill_the_review_list(self):
        r = self.runs[0]
        p, run_ = r["run"].programme, r["run"]
        hold = {b: s.scores["holdout"] for b, s in run_.scores.items()}
        score = compose.build(p.composition.chosen, hold, p.composition.weights)
        top = score.sort_values(ascending=False, kind="stable").index[:p.holdout.budget_rows]
        frame = r["frame"]
        self.assertEqual(int((frame.loc[top, "status"] == "CANCELLED").sum()), 0, "a cancelled ride is not scored as unusual")
        truth = set(r["replica"].truth["order_id"])
        precision = float(frame.loc[top, "order_id"].isin(truth).mean())
        base = len(truth) / len(frame)
        self.assertGreater(precision, 20 * base, f"the review list is {precision:.2f} planted fraud against {base:.4f} of all rides: look-alikes do not fill it")

    def test_the_run_is_deterministic(self):
        again = run(0)["run"].programme.fingerprint
        _RUNS.pop((0, True))
        self.assertEqual(run(0)["run"].programme.fingerprint, again)

    def test_without_the_table_of_accounts_the_holdout_stays_closed_and_the_rings_are_what_is_missed(self):
        r = run(0, with_accounts=False)
        p = r["run"].programme
        self.assertEqual(p.state, "planned")
        self.assertIsNone(p.holdout)
        self.assertTrue(any(b.startswith("blk3") for b in p.blockers), "the router has no table of events for the rings, so no approach: the blocker names the part")
        dropped = pg.drop_leaf(p, "blk3", "this company has no table of accounts")
        ctx = leaves.Context(frame=r["frame"], parts=leaves.partition(r["frame"], moment="created_at"), time="created_at", entities=["account_id", "biker_id"],
                             labels=r["labels"], exclude=["order_id"], budget=fp.BUDGET_SHARE)
        without = pg.resume(pg.Execution(dropped, r["run"].scores), r["solution"], r["map"], ctx, r["labels"], bootstraps=100).programme
        self.assertEqual(without.state, "composed")
        full = self.runs[0]["run"].programme.holdout
        self.assertLess(without.holdout.composed.value, full.composed.value, "the composition without the rings finds less of the confirmed cases")


if __name__ == "__main__":
    unittest.main()
