"""Package 30.11: the first three modifiers (rare positives, groups, time order), detected from the training rows with
fixed thresholds, applied in the proposal above their threshold with their reason, recorded as decisions, and named as the
part behind the run's split and metric. One cut-off for rare positives everywhere (5%)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import modifiers as M  # noqa: E402


def facts(**kw):
    return M.Facts(**{"task": "binary", **kw})


def by_name(found):
    return {d.name: d for d in found}


class DetectTests(unittest.TestCase):
    def test_rare_positives_under_five_percent_for_a_yes_or_no_problem_only(self):
        self.assertTrue(by_name(M.detect(facts(positive_rate=0.03)))["rare_positives"].applies)
        self.assertFalse(by_name(M.detect(facts(positive_rate=0.07)))["rare_positives"].applies)
        majority = by_name(M.detect(facts(positive_rate=0.97)))["rare_positives"]  # the rare class is the other one: asked
        self.assertEqual((majority.applies, majority.asks_user), (False, True))
        multi = by_name(M.detect(facts(task="multiclass", minority_rate=0.01)))["rare_positives"]
        self.assertFalse(multi.applies)
        self.assertIn("macro F1", multi.reason)

    def test_groups_take_the_coarsest_key_with_enough_values_and_ask_when_several_repeat(self):
        keys = [M.KeyFact(column="store_id", rows_per_key=125, groups=40), M.KeyFact(column="cashier_id", rows_per_key=21, groups=240),
                M.KeyFact(column="region", rows_per_key=1000, groups=5), M.KeyFact(column="bill_id", rows_per_key=1, groups=5000)]
        found = by_name(M.detect(facts(keys=keys)))["groups"]
        self.assertEqual((found.applies, found.value, found.asks_user), (True, "store_id", True))  # region has too few values
        self.assertIn("cashier_id", found.reason)
        self.assertFalse(by_name(M.detect(facts(keys=[M.KeyFact(column="order_id", rows_per_key=1, groups=900)])))["groups"].applies)

    def test_a_key_that_recurs_over_time_is_not_grouped_on_a_forward_split_the_person_is_asked(self):
        recurs = by_name(M.detect(facts(time_column="day", keys=[M.KeyFact(column="shop", rows_per_key=50, groups=60, recurring=1.0)])))["groups"]
        self.assertEqual((recurs.applies, recurs.asks_user), (False, True))
        self.assertIn("recurs over time", recurs.reason)
        new = by_name(M.detect(facts(time_column="day", keys=[M.KeyFact(column="shop", rows_per_key=50, groups=60, recurring=0.2)])))["groups"]
        self.assertTrue(new.applies)

    def test_a_time_column_applies_time_order_and_a_moving_target_makes_it_stronger(self):
        still = by_name(M.detect(facts(time_column="day", rate_change=1.1)))["time_order"]
        moving = by_name(M.detect(facts(time_column="day", rate_change=2.0)))["time_order"]
        self.assertTrue(still.applies and moving.applies)
        self.assertGreater(moving.strength, still.strength)
        self.assertIn("changes 2 times", moving.reason)
        self.assertFalse(by_name(M.detect(facts()))["time_order"].applies)

    def test_one_cut_off_for_rare_positives(self):
        from dclab_core import plan
        from dclab_rnd import kinds

        self.assertEqual((plan.RARE, plan.IMBALANCED), (M.RARE_BELOW, M.RARE_BELOW))
        binary = kinds.get("binary")
        self.assertEqual(binary.default_metric({}, 0.04), "average_precision")
        self.assertEqual(binary.default_metric({}, 0.07), "roc_auc")


def _frame(seed=0, rate=0.03, n=3000, drift=False, new_shops=False):
    """Orders by hour: a shop code that recurs over time (or, ``new_shops``, shops that open as time goes on), a positive rate
    that triples in the last third when ``drift``."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-01-01", periods=n, freq="h")
    p = np.where(np.arange(n) > 2 * n / 3, rate * 3, rate) if drift else np.full(n, rate)
    shops = (np.arange(n) // 25) if new_shops else rng.integers(0, 60, n)
    return pd.DataFrame({"created_at": t.astype(str), "shop_code": [f"S{int(i)}" for i in shops],
                         "amount": rng.normal(50, 10, n).round(2), "is_fraud": (rng.random(n) < p).astype(int)})


class ProposalTests(unittest.TestCase):
    def propose(self, frame):
        from dclab_rnd.studio import ProjectStore, data as sd
        from dclab_rnd.studio.solution import propose

        store = ProjectStore(Path(tempfile.mkdtemp()))
        project = store.create("x", "general", "when the order is placed")
        frame.to_csv(store.data_dir(project["id"]) / "table.csv", index=False)
        sd.attach_data(store, project["id"], "table.csv")
        return store, store.get(project["id"]), propose(frame, store.get(project["id"])["data"]["profile"], "is_fraud", "binary")

    def test_the_proposal_shows_every_modifier_and_applies_those_above_their_threshold(self):
        _, _, prop = self.propose(_frame(drift=True))
        found = {d["name"]: d for d in prop["modifiers"]}
        self.assertTrue(found["rare_positives"]["applies"] and found["time_order"]["applies"], found)
        self.assertEqual(prop["metric"], "average_precision")
        self.assertEqual((found["groups"]["applies"], found["groups"]["asks_user"]), (False, True))  # the shop recurs over time: asked
        self.assertNotIn("group_column", prop)
        self.assertIn("changes", found["time_order"]["reason"])
        _, _, prop = self.propose(_frame(drift=True, new_shops=True))  # shops open as time goes on: grouping and a forward split both hold
        self.assertEqual(prop.get("group_column"), "shop_code")

    def test_a_proposal_never_fails_on_its_modifiers_and_counts_every_row_it_read(self):
        from dclab_rnd.studio import ProjectStore, data as sd
        from dclab_rnd.studio.solution import propose

        n = 900
        frame = pd.DataFrame({"created_at": pd.date_range("2026-01-01", periods=n, freq="h").astype(str), "amount": np.arange(n) % 7,
                              "label": ["a"] * 300 + ["b"] * 300 + ["c"] * 300})  # the classes of the first and last thirds never meet
        store = ProjectStore(Path(tempfile.mkdtemp()))
        project = store.create("x", "general", "when the row is written")
        frame.to_csv(store.data_dir(project["id"]) / "table.csv", index=False)
        sd.attach_data(store, project["id"], "table.csv")
        prop = propose(frame, store.get(project["id"])["data"]["profile"], "label", "multiclass")
        self.assertIn("modifiers", prop)
        from dclab_rnd.studio.solution import modifier_facts

        big = pd.DataFrame({"customer_code": [f"C{i // 4}" for i in range(120_000)], "x": np.zeros(120_000)})
        got = modifier_facts(big, pd.Series(np.zeros(120_000)), "regression", None, ["customer_code"], None)
        self.assertEqual(got.keys[0].rows_per_key, 4.0)  # counted on every row given, never a sample

    def test_a_saved_solution_records_each_modifier_against_the_rule(self):
        from dclab_rnd.ledger import capture
        from dclab_rnd import ledger
        from dclab_core.ledger import check_state

        store, project, prop = self.propose(_frame(drift=True))
        book = ledger.FileLedger(Path(tempfile.mkdtemp()), "w")
        ledger.install(book)
        self.addCleanup(ledger.install, None)
        solution = {"target": "is_fraud", "task": "binary", "metric": None, "group_column": "shop_code", "time_column": "created_at"}
        capture.solution_saved({**prop, "rule": {k: prop.get(k) for k in ("identifiers", "forbidden", "time_candidates", "group_candidates",
                                                                          "modifiers", "modifier_facts")}},
                               project["data"]["profile"], solution, "human", project=project["id"])
        recs = {r.point: r for r in book.records(project=project["id"]) if r.point.startswith("modifier.")}
        self.assertEqual(set(recs), {"modifier.rare_positives", "modifier.groups", "modifier.time_order"})
        self.assertEqual((recs["modifier.groups"].value_used, recs["modifier.time_order"].value_used), (True, True))  # what the saved solution applies
        rule = next(a for a in recs["modifier.groups"].answers if a.by == "rule")
        self.assertIs(rule.value, False)  # the rule asked (the shop recurs over time); the person grouped anyway
        self.assertEqual(book.labels(point="modifier.groups"), [])  # no page shows the rule's answer yet: never an exposed label
        for r in recs.values():
            self.assertEqual(check_state(dict(r.state)), [])
            self.assertTrue(any(a.by == "rule" and a.reason for a in r.answers))


class EngineAttributionTests(unittest.TestCase):
    def test_a_run_built_from_the_proposal_reaches_its_holdout(self):  # the review of 30.11: a grouped forward split must not fail
        from dclab_rnd.kinds import acceptance

        for frame in (_frame(drift=True), _frame(drift=True, new_shops=True)):
            case = acceptance.Case(table=frame, target="is_fraud", task="binary", moment="when the order is placed")
            done = acceptance.engine_run(case, "binary")
            self.assertIn("holdout_metrics", done.evidence("final"))

    def test_the_run_names_the_modifier_behind_its_split(self):
        from dclab_rnd import kinds
        from dclab_rnd.kinds import acceptance

        kind = kinds.get("binary")
        done = acceptance.engine_run(kind.case(0), kind.key)
        recipe = done.evidence("data")["recipe"]
        by = {c["field"]: c for c in recipe["choices"]}
        self.assertEqual(by["split"]["part"], "time_order@1")  # the replica's time column, found by the rows and kept
        self.assertIn("time_order@1", recipe["modifiers"])
        self.assertEqual({d["name"] for d in recipe["detected"]}, {"rare_positives", "groups", "time_order"})


if __name__ == "__main__":
    unittest.main()
