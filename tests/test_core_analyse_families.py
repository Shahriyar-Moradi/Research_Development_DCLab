"""Package 20.5: analysis by family. Each family's replica yields the findings its decisions need (without labels: the
shape of normal, the rows rules flag, each entity's behaviour, what a detector can find among planted shapes; partial
labels; censoring and columns counted during follow-up; series and season; arms; popularity; scales), each naming
decisions the plan and the ledger know; and none reads a held-out row: replacing every value of the held-out rows with
garbage leaves every finding about the training rows unchanged."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd

    from dclab_core.analyse import AFFECTS, analyse
    from dclab_core.ledger import POINTS
    from dclab_core.replicas import (campaign_uplift, customer_behaviours, delivery_fraud, machine_sensors, shop_interactions,
                                     store_demand, subscription_lifetimes)
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""


def cases():
    """(family, tables, main, target, keyword arguments, the time column the split follows, the findings it needs)."""
    fraud = delivery_fraud.generate(seed=0)
    demand, lifetimes, trial = store_demand.generate(seed=0), subscription_lifetimes.generate(seed=0), campaign_uplift.generate(seed=0)
    shop, groups, sensors = shop_interactions.generate(seed=0), customer_behaviours.generate(seed=0), machine_sensors.generate(seed=0)
    f, g, h = demand.facts, lifetimes.facts, trial.facts
    labelled = fraud.rides.assign(is_fraud=fraud.rides["order_id"].isin(fraud.sample["order_id"]).astype(int))
    return [
        ("anomaly", {"rides": fraud.rides}, "rides", None, {"entity": "biker_id"}, None,
         {"F-normal", "F-rules", "F-behaviour-biker_id", "F-detectable"}),
        ("rare_event", {"rides": labelled, "sample": fraud.sample}, "rides", "is_fraud", {"entity": "biker_id", "labels": "sample"}, None,
         {"F-normal", "F-rules", "F-behaviour-biker_id", "F-labels-join"}),
        ("forecasting", {"sales": demand.sales}, "sales", "units", {"entity": f["series"], "time": f["time"]}, f["time"],
         {"F-series", "F-season"}),
        ("time_to_event", {"subs": lifetimes.subscribers}, "subs", g["duration"], {"event": g["event"]}, None,
         {"F-censoring", "F-counted-tickets_so_far"}),
        ("uplift", {"customers": trial.customers}, "customers", h["outcome"], {"treatment": h["treatment"]}, None, {"F-arms"}),
        ("ranking", {"events": shop.interactions}, "events", None, {"entity": shop.facts["item"]}, shop.facts["time"],
         {f"F-popularity-{shop.facts['item']}"}),
        ("segmentation", {"customers": groups.customers}, "customers", None, {}, None, {"F-normal", "F-scales"}),
        ("streams", {"readings": sensors.readings}, "readings", None, {"entity": sensors.facts["unit"], "time": sensors.facts["time"]},
         sensors.facts["time"], {"F-behaviour-" + sensors.facts["unit"], "F-detectable"}),
    ]


def training(frame, time):
    order = frame.sort_values(time).index if time else frame.index
    return list(order[: int(len(order) * 0.75)])


def poisoned(frame, keep):
    """Every value of the rows outside ``keep`` replaced: numbers scaled, flipped and some missing; text and dates moved."""
    out = frame.copy()
    rest = out.index.difference(pd.Index(keep))
    rng = np.random.default_rng(1)
    for c in out.columns:
        if pd.api.types.is_bool_dtype(out[c]):
            out.loc[rest, c] = ~out.loc[rest, c].astype(bool)
        elif pd.api.types.is_integer_dtype(out[c]):  # its type kept: only the values are garbage
            out.loc[rest, c] = -1_000_000 * out.loc[rest, c] + 7
        elif pd.api.types.is_numeric_dtype(out[c]):
            values = -1e6 * out.loc[rest, c].astype(float) + 7
            values[rng.random(len(values)) < 0.3] = np.nan
            out.loc[rest, c] = values
        elif pd.api.types.is_datetime64_any_dtype(out[c]):
            out.loc[rest, c] = out.loc[rest, c] + pd.Timedelta(days=3650)
        else:
            out[c] = out[c].astype(object)
            out.loc[rest, c] = "zz-poison"
    return out


@unittest.skipIf(MISSING, MISSING)
class FamiliesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = cases()

    def run_case(self, case, tables=None):
        family, base, main, target, kw, time, _ = case
        kw = dict(kw)
        labels = kw.pop("labels", None)
        keep = training(base[main], time)
        return analyse(tables or base, target=target, train_index=keep, main=main, labels=labels, family=family,
                       time=kw.pop("time", time), **kw), keep

    def test_each_familys_replica_yields_the_findings_its_decisions_need(self):
        for case in self.cases:
            with self.subTest(family=case[0]):
                out, keep = self.run_case(case)
                ids = {f.id for f in out.findings}
                self.assertTrue(case[6] <= ids, f"{case[0]}: missing {case[6] - ids}")
                self.assertEqual(out.training_rows, len(keep))
                self.assertNotIn("F-no-split", ids)

    def test_every_finding_names_decisions_the_ledger_knows(self):
        self.assertTrue(all(point is None or point in POINTS for point in AFFECTS.values()))
        for case in self.cases:
            out, _ = self.run_case(case)
            for f in out.findings:
                self.assertTrue(f.affects, f.id)
                self.assertTrue(set(f.affects) <= set(AFFECTS), (case[0], f.id, f.affects))

    def test_a_finding_naming_an_unknown_decision_is_refused(self):
        from dclab_core.analyse import _f

        with self.assertRaises(ValueError):
            _f("x", "rules", "s", {}, "i", ["split.strategy", "vibes"])

    def test_what_a_detector_finds_is_measured_and_the_limits_are_said(self):
        out, _ = self.run_case(self.cases[0])  # the fraud replica's rides, no label
        found = next(f for f in out.findings if f.id == "F-detectable")
        self.assertEqual(set(found.evidence["shapes"]), {"far_value", "broken_pair"})
        self.assertIn("model.families", found.affects)
        censoring = next(f for f in self.run_case(self.cases[3])[0].findings if f.id == "F-censoring")
        self.assertGreater(censoring.evidence["censored_share"], 0.2)  # the replica's subscribers still going
        self.assertNotIn("F-counted-subscriber_id", {f.id for f in self.run_case(self.cases[3])[0].findings})  # an id is no measure

    def test_no_finding_about_the_training_rows_reads_a_held_out_row(self):
        for case in self.cases:
            with self.subTest(family=case[0]):
                family, base, main, target, kw, time, _ = case
                clean, keep = self.run_case(case)
                dirty, _ = self.run_case(case, {**base, main: poisoned(base[main], keep)})
                mine = lambda out: {f.id: f.model_dump_json() for f in out.findings if f.scope != "all_rows"}  # noqa: E731
                self.assertEqual(mine(dirty), mine(clean))


@unittest.skipIf(MISSING, MISSING)
class EdgeTest(unittest.TestCase):
    def frame(self, n=3000, seed=0):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=n)
        return pd.DataFrame({"a": a, "b": -a + 0.2 * rng.normal(size=n), "amount": rng.lognormal(3, 1.2, n),
                             "count": rng.poisson(4, n), "who": rng.integers(0, 60, n).astype(str)})

    def shapes(self, frame, **kw):
        out = analyse({"t": frame}, train_index=list(frame.index), family="anomaly", **kw)
        return next(f for f in out.findings if f.id == "F-detectable").evidence["shapes"]

    def test_a_negative_pair_is_broken_against_its_direction_and_a_skewed_far_value_is_far(self):
        shapes = self.shapes(self.frame())
        self.assertIn("a", shapes["broken_pair"]["columns"])
        self.assertGreater(shapes["broken_pair"]["average_precision"], 2 * shapes["broken_pair"]["chance"])  # broken the wrong way it sits at chance
        self.assertGreater(shapes["far_value"]["recall_at_budget"], 0.5)  # beyond the column's range, skewed ones logged

    def test_a_small_table_can_reach_full_recall(self):
        shapes = self.shapes(self.frame(n=300))
        self.assertEqual(shapes["far_value"]["planted"], 10)
        self.assertGreater(shapes["far_value"]["recall_at_budget"], 0.5)  # the budget is the planted count, not 3 rows

    def test_ordinary_data_never_crashes_the_family_analyses(self):
        rng = np.random.default_rng(2)
        n = 400
        days = pd.date_range("2024-01-01", periods=n, freq="D")
        intermittent = pd.DataFrame({"day": days, "shop": "s1", "units": np.r_[np.zeros(n - 20), rng.integers(1, 9, 20)]})
        analyse({"t": intermittent}, target="units", train_index=list(intermittent.index), family="forecasting", entity="shop", time="day")
        missing_entity = self.frame(500).assign(who=np.nan)
        analyse({"t": missing_entity}, train_index=list(missing_entity.index), family="anomaly", entity="who")
        arms = self.frame(800).assign(arm=[f"arm{i % 40}" for i in range(800)], bought=["yes", "no"] * 400)
        out = analyse({"t": arms}, target="bought", train_index=list(arms.index), family="uplift", treatment="arm")
        found = next(f for f in out.findings if f.id == "F-arms")
        self.assertLessEqual(len(found.evidence["arms"]), 11)
        epoch = self.frame(600).assign(ts=1_700_000_000 + np.arange(600) * 3600)
        out = analyse({"t": epoch}, train_index=list(epoch.index), family="anomaly", entity="who", time="ts")
        behaviour = next(f for f in out.findings if f.id == "F-behaviour-who")
        self.assertNotIn("rows_per_entity_day_p99", behaviour.evidence)  # seconds are never read as nanoseconds

    def test_an_event_flag_that_cannot_be_read_is_said(self):
        base = pd.DataFrame({"days": np.arange(1, 301), "x": np.arange(300) % 7})
        text = analyse({"t": base.assign(ev=["ended", "active"] * 150)}, target="days", train_index=list(range(300)),
                       family="time_to_event", event="ev")
        self.assertIn("cannot be read", next(f for f in text.findings if f.id == "F-censoring").statement)
        every = analyse({"t": base.assign(ev=1)}, target="days", train_index=list(range(300)), family="time_to_event", event="ev")
        found = next(f for f in every.findings if f.id == "F-censoring")
        self.assertEqual((found.severity, found.evidence["readable"]), ("warning", False))
        gaps = base.assign(ev=[1.0, 0.0, np.nan] * 100)
        analyse({"t": gaps}, target="days", train_index=list(range(300)), family="time_to_event", event="ev")

    def test_partial_labels_read_the_table_without_them(self):
        frame = self.frame(1000).assign(case=[1.0] * 20 + [np.nan] * 980)
        ids = {f.id for f in analyse({"t": frame}, target="case", train_index=list(frame.index), family="rare_event", entity="who").findings}
        self.assertTrue({"F-normal", "F-rules", "F-behaviour-who", "F-detectable"} <= ids)


if __name__ == "__main__":
    unittest.main()
