"""Package 20.3: each column's purpose with its reason, the grain, units and the refusals. On every replica the
purposes match the replica's facts (what its facts and docstring name: outcome, identifier, entity, time, text,
place, leak); customer_id can never be proposed as an outcome, as an identifier or as a recurring key."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd

    from dclab_core import datacard as dc
    from dclab_core.replicas import (campaign_uplift, city_prices, customer_behaviours, delivery_fraud, delivery_time, machine_sensors,
                                     product_reviews, ride_pricing, shapes_and_notes, shop_customers, shop_interactions, store_demand,
                                     subscription_churn, subscription_lifetimes, ticket_routing)
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""


def replicas():
    """(name, tables, the outcome the problem names, the purposes the replica's facts and docstring say)."""
    r = {m.__name__.rsplit(".", 1)[1]: m.generate(seed=0) for m in (campaign_uplift, city_prices, customer_behaviours, delivery_fraud,
                                                                      delivery_time, machine_sensors, product_reviews, ride_pricing,
                                                                      shop_customers, shop_interactions, store_demand, subscription_churn,
                                                                      subscription_lifetimes, ticket_routing)}
    shapes, notes = shapes_and_notes.shapes(seed=0), shapes_and_notes.notes(seed=0)

    def facts(name, key):
        value = r[name].facts.get(key)
        return value if isinstance(value, list) else [value]

    leak = lambda name, *keys: {c: "leak_suspect" for k in keys for c in facts(name, k)}  # noqa: E731
    return [
        ("campaign_uplift", {"customers": r["campaign_uplift"].customers}, "bought",
         {"bought": "target_candidate", r["campaign_uplift"].facts["identifier"]: "identifier"}),
        ("city_prices", {"flats": r["city_prices"].flats}, "price", {"price": "target_candidate", "lat": "place", "lon": "place"}),
        ("customer_behaviours", {"customers": r["customer_behaviours"].customers}, None, {"customer_id": "identifier"}),
        ("delivery_fraud", {"rides": r["delivery_fraud"].rides, "known_frauds": r["delivery_fraud"].sample}, None,  # its docstring
         {"terminal_ref": "identifier", "order_id": "identifier", "biker_id": "entity", "created_date": "time", "source_lat": "place"}),
        ("delivery_time", {"orders": r["delivery_time"].orders}, "actual_minutes",  # its docstring: the target, the moment
         {"actual_minutes": "target_candidate", "order_id": "identifier", "order_time": "time", r["delivery_time"].facts["entity"]: "entity",
          **leak("delivery_time", "leaky_columns")}),
        ("machine_sensors", {"readings": r["machine_sensors"].readings}, None,
         {r["machine_sensors"].facts["unit"]: "entity", r["machine_sensors"].facts["time"]: "time"}),
        ("product_reviews", {"reviews": r["product_reviews"].reviews}, "negative",
         {"negative": "target_candidate", "review_text": "text", "review_id": "identifier", **leak("product_reviews", "leak")}),
        ("ride_pricing", {"offers": r["ride_pricing"].offers}, "accepted", {"accepted": "target_candidate", **leak("ride_pricing", "leak")}),
        ("shapes", {"table": shapes.table}, "label", {"label": "target_candidate", shapes.facts["source"]: "entity"}),
        ("notes", {"table": notes.table}, "urgent", {"urgent": "target_candidate", notes.facts["source"]: "entity"}),
        ("shop_customers", {"customers": r["shop_customers"].customers}, None,
         {r["shop_customers"].facts["key"]: "identifier", "lifetime_orders": "leak_suspect"}),
        ("shop_interactions", {"items": r["shop_interactions"].items}, None,
         {r["shop_interactions"].facts["item"]: "identifier", **leak("shop_interactions", "leak")}),
        ("store_demand", {"sales": r["store_demand"].sales}, "units",
         {"units": "target_candidate", r["store_demand"].facts["series"]: "entity", r["store_demand"].facts["time"]: "time"}),
        ("subscription_churn", {"subscribers": r["subscription_churn"].subscribers}, "churned",
         {"churned": "target_candidate", "subscriber_id": "identifier", "signup_time": "time", **leak("subscription_churn", "leak")}),
        ("subscription_lifetimes", {"subscribers": r["subscription_lifetimes"].subscribers}, "days_subscribed",
         {"days_subscribed": "target_candidate", "subscriber_id": "identifier", "signup_date": "time", "cancelled": "leak_suspect",
          **leak("subscription_lifetimes", "leaks")}),
        ("ticket_routing", {"tickets": r["ticket_routing"].tickets}, "assigned_team",  # its docstring: ticket_id, created_at
         {"assigned_team": "target_candidate", "ticket_id": "identifier", "created_at": "time", r["ticket_routing"].facts["entity"]: "entity",
          **leak("ticket_routing", "leaky_columns")}),
    ]


# what a table alone cannot show, each with its reason: these purposes come from the problem or a later table, not the card
NOT_FROM_THE_TABLE = {
    ("store_demand", "rain_mm"): "known only on the day, not ahead: a fact about when it is measured, not about the table",
    ("shop_customers", "last_order_date"): "written later than the cutoff; the outcome is in another table (orders)",
}


@unittest.skipIf(MISSING, MISSING)
class ReplicaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = replicas()
        cls.cards = {name: dc.card_of(tables, name=name, target=target) for name, tables, target, _ in cls.cases}

    def test_on_every_replica_the_purposes_match_its_facts(self):
        self.assertEqual(len(self.cases), 16)
        for name, _, _, expected in self.cases:
            got = {c.name: c.purpose for c in self.cards[name].tables[0].columns}
            for column, purpose in expected.items():
                with self.subTest(replica=name, column=column):
                    self.assertEqual(got[column], purpose, next(c.why for c in self.cards[name].tables[0].columns if c.name == column))
        for (name, column), why in NOT_FROM_THE_TABLE.items():
            self.assertNotEqual({c.name: c.purpose for c in self.cards[name].tables[0].columns}[column], "leak_suspect", why)

    def test_every_purpose_says_why_and_the_grain_is_said(self):
        for name, card in self.cards.items():
            for t in card.tables:
                self.assertTrue(t.grain, name)
                for c in t.columns:
                    self.assertTrue(c.purpose and c.why, (name, c.name))
        tickets = self.cards["ticket_routing"].tables[0]
        self.assertEqual((tickets.key, tickets.duplicate_rows > 0), (["ticket_id"], True))
        self.assertIn("repeated whole", tickets.grain)
        self.assertEqual(self.cards["machine_sensors"].tables[0].key, ["time", "machine"])
        self.assertEqual(self.cards["delivery_fraud"].tables[1].joins_on, ["order_id"])  # the label file's key into the rides
        self.assertEqual({c.name: c.unit for c in self.cards["delivery_time"].tables[0].columns}["distance_km"], "km")

    def test_customer_id_is_never_an_outcome(self):
        for name, card in self.cards.items():
            for t in card.tables:
                self.assertNotIn("customer_id", t.target_candidates, name)
        for name, tables, column in (("campaign_uplift", self.cases[0][1], "customer_id"), ("ticket_routing", self.cases[-1][1], "customer_id")):
            asked = dc.card_of(tables, target=column)  # even when the problem names it
            self.assertNotIn(column, asked.tables[0].target_candidates)
            refused = [p for p in asked.problems if p.code == "target_refused"]
            self.assertEqual([p.columns for p in refused], [[column]], name)


@unittest.skipIf(MISSING, MISSING)
class RefusalTest(unittest.TestCase):
    def test_a_constant_a_row_number_and_free_text_are_never_outcomes(self):
        n = 200
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"row": np.arange(1, n + 1), "country": "IT", "note": [f"a long free text about order {i} and what was said" for i in range(n)],
                              "spend": rng.normal(50, 10, n), "is_late": rng.integers(0, 2, n)})
        card = dc.card_of({"t": frame})
        got = {c.name: (c.purpose, c.why) for c in card.tables[0].columns}
        self.assertEqual(got["row"][0], "identifier")
        self.assertEqual(got["country"][0], "constant")
        self.assertEqual(got["note"][0], "text")
        self.assertEqual(card.tables[0].target_candidates, ["is_late"])
        for column in ("row", "country", "note"):
            self.assertIn("target_refused", {p.code for p in dc.card_of({"t": frame}, target=column).problems}, column)
        self.assertEqual(dc.card_of({"t": frame.assign(row=np.arange(n) * 7 + 3)}).tables[0].columns[0].purpose, "identifier")

    def test_a_column_empty_for_one_outcome_or_naming_it_is_suspect(self):
        n = 400
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, n)
        frame = pd.DataFrame({"y": y, "closed_by_team": np.where(y == 1, "team_a", "team_b"), "spend": rng.normal(size=n),
                              "paid_at": np.where(y == 1, "2026-01-02", None)})
        got = {c.name: c.purpose for c in dc.card_of({"t": frame}, target="y").tables[0].columns}
        self.assertEqual(got["paid_at"], "leak_suspect")  # empty exactly when y is 0
        self.assertEqual(got["spend"], "feature")
        plain = frame.rename(columns={"closed_by_team": "team"})
        self.assertEqual({c.name: c.purpose for c in dc.card_of({"t": plain}, target="y").tables[0].columns}["team"], "leak_suspect")

    def test_a_named_outcome_is_judged_as_an_outcome(self):
        n = 400
        rng = np.random.default_rng(2)
        frame = pd.DataFrame({"price": rng.integers(100_000, 900_000, n), "amount": rng.normal(100, 20, n).round(2).astype(str),
                              "category": [f"cat_{i % 30}" for i in range(n)], "destination_city": rng.choice(["Rome", "Milan", "Turin"], n),
                              "final_grade": rng.integers(1, 6, n), "row": np.arange(1, n + 1), "size": rng.normal(size=n)})
        for target in ("price", "amount", "category", "destination_city", "final_grade"):
            with self.subTest(target=target):
                card = dc.card_of({"t": frame}, target=target)
                self.assertIn(target, card.tables[0].target_candidates)
                self.assertNotIn("target_refused", {p.code for p in card.problems})
        self.assertIn("target_refused", {p.code for p in dc.card_of({"t": frame}, target="row").problems})  # a row number never

    def test_customer_id_with_few_values_is_still_a_key(self):
        n = 400
        rng = np.random.default_rng(3)
        for levels in (2, 8):
            frame = pd.DataFrame({"customer_id": [f"C{i % levels:04d}" for i in range(n)], "y": rng.integers(0, 2, n)})
            free = dc.card_of({"t": frame})
            self.assertNotIn("customer_id", free.tables[0].target_candidates, levels)
            named = dc.card_of({"t": frame}, target="customer_id")
            self.assertEqual([p.columns for p in named.problems if p.code == "target_refused"], [["customer_id"]], levels)

    def test_held_out_names_get_their_purpose(self):
        """Realistic names that are not in any replica, with the purpose each should get."""
        n = 300
        rng = np.random.default_rng(4)
        frame = pd.DataFrame({"default_rate": rng.random(n), "class_size": rng.integers(10, 40, n), "positive_reviews": rng.integers(0, 50, n),
                              "y_coord": rng.normal(size=n), "ytd_spend": rng.normal(500, 50, n), "min_balance": rng.normal(10, 2, n),
                              "last_login_at": pd.date_range("2025-01-01", periods=n, freq="D"), "segment": rng.choice(list("abc"), n),
                              "is_late": rng.integers(0, 2, n), "outcome": rng.integers(0, 2, n), "fraud_flag": rng.integers(0, 2, n),
                              "has_coupon": rng.integers(0, 2, n), "delivered_on_time": rng.integers(0, 2, n)})
        got = {c.name: c for c in dc.card_of({"t": frame}).tables[0].columns}
        for column in ("default_rate", "class_size", "positive_reviews", "y_coord", "ytd_spend", "min_balance", "segment"):
            self.assertEqual(got[column].purpose, "feature", column)
        self.assertEqual(got["last_login_at"].purpose, "time")
        for column in ("is_late", "outcome", "fraud_flag", "delivered_on_time"):  # a yes or no, even named written-after
            self.assertEqual(got[column].purpose, "target_candidate", column)
        self.assertIsNone(got["min_balance"].unit)
        with_target = {c.name: c.purpose for c in dc.card_of({"t": frame}, target="is_late").tables[0].columns}
        self.assertEqual((with_target["default_rate"], with_target["class_size"], with_target["outcome"], with_target["fraud_flag"],
                          with_target["has_coupon"]), ("feature", "feature", "leak_suspect", "leak_suspect", "feature"))

    def test_the_data_rules_of_a_leak(self):
        n = 2000
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, n)
        frame = pd.DataFrame({"y": y, "score": np.where(y == 1, rng.normal(10, 1, n), rng.normal(0, 1, n)), "noise": rng.normal(size=n)})
        got = {c.name: c.purpose for c in dc.card_of({"t": frame}, target="y").tables[0].columns}
        self.assertEqual((got["score"], got["noise"]), ("leak_suspect", "feature"))  # separates it alone
        start = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 10_000, n), unit="m")
        minutes = rng.integers(10, 90, n)
        times = pd.DataFrame({"placed": start, "arrived": start + pd.to_timedelta(minutes, unit="m"), "minutes": minutes.astype(float),
                              "basket": rng.integers(1, 9, n)})
        got = {c.name: c.purpose for c in dc.card_of({"t": times}, target="minutes").tables[0].columns}
        self.assertEqual((got["arrived"], got["placed"], got["basket"]), ("leak_suspect", "time", "feature"))  # a time that gives it
        rare = rng.random(20_000) < 0.008
        gaps = pd.DataFrame({"is_fraud_case": rare.astype(int), "promo_amount": np.where(rng.random(20_000) < 0.008, np.nan, rng.normal(size=20_000)),
                             "coupon": np.where(rare | (rng.random(20_000) < 0.3), np.nan, 1.0),  # empty for every case, and for a third of the rest
                             "shop": rng.choice([f"s{i}" for i in range(19)], 20_000)})
        got = {c.name: c.purpose for c in dc.card_of({"t": gaps}, target="is_fraud_case").tables[0].columns}
        self.assertEqual(got["promo_amount"], "feature")  # random gaps on a rare outcome are not a leak
        self.assertEqual(got["coupon"], "feature")  # empty for every case, but most empty rows are not cases
        self.assertEqual(got["shop"], "feature")  # with an outcome nine rows in ten share, every column looks pure

    def test_an_entity_that_shares_its_outcome_stays_an_entity(self):
        n = 600
        rng = np.random.default_rng(6)
        customers = np.arange(n) % 60
        churned = (customers % 3 == 0).astype(int)
        frame = pd.DataFrame({"customer_id": [f"C{c:03d}" for c in customers], "amount": rng.normal(size=n), "churned": churned})
        got = {c.name: c for c in dc.card_of({"t": frame}, target="churned").tables[0].columns}
        self.assertEqual(got["customer_id"].purpose, "entity")
        self.assertIn("split by it", got["customer_id"].why)

    def test_a_long_file_name_never_leaves_a_shared_card(self):
        from dclab_rnd.datasets import anonymous_card

        name = "maria_rossi_private_customer_export_from_crm_2026_with_contacts.csv"
        frame = pd.DataFrame({"customer_id": [f"C{i}" for i in range(50)], "y": [0, 1] * 25})
        card = dc.card_of({name: frame}, target="customer_id").model_dump(mode="json")
        self.assertNotIn("maria", str(anonymous_card(card)))

    def test_a_version_one_card_is_read_as_version_two(self):
        old = {"schema_version": "dclab_core.data_card.v1", "name": "t", "tables": [{"name": "t", "rows": 0}]}
        self.assertEqual(dc.DataCard.read(old).schema_version, "dclab_core.data_card.v3")


if __name__ == "__main__":
    unittest.main()
