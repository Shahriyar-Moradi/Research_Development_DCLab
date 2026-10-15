"""Package 19.11: data cards and synthetic tables. For the fraud replica the card names the planted traps, a table
drawn from the card keeps the roles and the traps, and shares no row with the source; a card holds no row and no
value a person could be found by; a draft and a project keep the card of their table."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd

    from dclab_core import datacard as dc
    from dclab_core.replicas import delivery_fraud as dfr
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""

TRAPS = {  # the replica's traps (delivery_fraud's docstring) and the problem each must show as
    "the pair (account, biker) inflates the labels": "coarse_join_inflates",
    "partial labels, some rows not in the table": "rows_not_in_table",
    "sender_phone means two things": "same_name_other_meaning",
    "the sample's fares disagree with the table's": "values_disagree",
    "the frauds gather on a few bikers": "labels_concentrate_on_entities",
    "cancelled rides have no biker": "empty_by_category",
}


def tables(replica):
    return {"rides": replica.rides, "known_frauds": replica.sample.drop(columns=["reason"])}


@unittest.skipIf(MISSING, MISSING)
class FraudReplicaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replica = dfr.generate(seed=0)
        cls.card = dc.card_of(tables(cls.replica), name="delivery_fraud", synthetic=True)
        cls.drawn = dc.synthesize(cls.card, seed=1)
        cls.again = dc.card_of(cls.drawn, name="drawn")

    def test_the_card_names_every_planted_trap(self):
        found = {p.code for p in self.card.problems}
        for trap, code in TRAPS.items():
            with self.subTest(trap=trap):
                self.assertIn(code, found)
        coarse = next(p for p in self.card.problems if p.code == "coarse_join_inflates")
        self.assertGreaterEqual(coarse.numbers["fanout"], 10)
        pair = next(r for r in self.card.relations if r.kind == "join" and set(r.columns) == {"account_id", "biker_id"})
        self.assertGreaterEqual(pair.fanout, 10)  # the pair (account, biker): each known fraud finds many rides
        key = next(r for r in self.card.relations if r.kind == "key")
        self.assertEqual((key.unmatched, key.concentrated_on), (5, "biker_id"))

    def test_a_drawn_table_keeps_the_roles_and_the_traps(self):
        self.assertEqual({p.code for p in self.again.problems}, {p.code for p in self.card.problems})
        for source, drawn in zip(self.card.tables, self.again.tables):
            self.assertEqual([c.name for c in source.columns], [c.name for c in drawn.columns])
            for a, b in zip(source.columns, drawn.columns):
                with self.subTest(table=source.name, column=a.name):
                    self.assertEqual((a.role, a.dtype, a.nulls), (b.role, b.dtype, b.nulls))
        kept = ("null_when", "one_to_one", "disjoint")
        self.assertEqual({(r.kind, r.table, tuple(r.columns)) for r in self.card.relations if r.kind in kept},
                         {(r.kind, r.table, tuple(r.columns)) for r in self.again.relations if r.kind in kept})
        source, drawn = (next(r for r in c.relations if r.kind == "key") for c in (self.card, self.again))
        self.assertIn(drawn.columns[0], ("order_id", "terminal_ref"))  # both identify a drop-off of an order: either is the key
        self.assertEqual((drawn.unmatched, drawn.concentrated_on), (source.unmatched, source.concentrated_on))

    def test_no_row_of_the_source_and_marked_synthetic(self):
        for name, frame in tables(self.replica).items():
            source = set(map(tuple, frame.astype(str).to_numpy()))
            drawn = self.drawn[name]
            self.assertEqual(sum(tuple(row) in source for row in drawn.astype(str).to_numpy()), 0)
            self.assertTrue(drawn.attrs.get("dclab.synthetic"))
            self.assertEqual(len(drawn), len(frame))
        # not even an identifier: the drawn ids are invented
        for column in ("order_id", "terminal_ref", "biker_phone", "sender_phone", "biker_id", "account_id"):
            ids = lambda f: set(dc._norm(f[column]).dropna())  # noqa: E731 - 3 and 3.0 are one id
            self.assertFalse(ids(self.drawn["rides"]) & ids(self.replica.rides), column)
            self.assertFalse(ids(self.drawn["known_frauds"]) & ids(self.replica.sample), column)

    def test_the_same_seed_draws_the_same_tables(self):
        again = dc.synthesize(self.card, seed=1)
        for name in again:
            pd.testing.assert_frame_equal(again[name], self.drawn[name])
        self.assertFalse(dc.synthesize(self.card, seed=2)["rides"]["order_id"].equals(self.drawn["rides"]["order_id"]))

    def test_the_card_holds_no_value_a_person_could_be_found_by(self):
        text = self.card.model_dump_json()
        rides = self.replica.rides
        for column in ("biker_phone", "sender_phone", "terminal_ref", "order_id", "biker_id", "account_id"):
            for value in rides[column].dropna().astype(str).unique()[:200]:
                if len(value.removesuffix(".0")) >= 5:  # an id of one digit is in any number
                    self.assertNotIn(value.removesuffix(".0"), text, column)
        self.assertNotIn("CANCELLED", text)  # category names are codes unless kept
        self.assertIn("status_b", json.dumps([str(v) for v in dc.synthesize(self.card, seed=3)["rides"]["status"].unique()]))
        rebuilt = dc.DataCard.read(text)  # a frozen, versioned contract
        self.assertEqual(rebuilt, self.card)


@unittest.skipIf(MISSING, MISSING)
class ColumnTest(unittest.TestCase):
    def test_roles_by_rule(self):
        rng = np.random.default_rng(0)
        n = 500
        frame = pd.DataFrame({"customer_id": np.arange(n), "country_code": rng.choice(["IT", "FR", "DE"], n), "amount": rng.normal(100, 20, n),
                              "created": pd.date_range("2025-01-01", periods=n, freq="h"), "note": ["a long free text note number %d about the order" % i for i in range(n)],
                              "rating": rng.integers(1, 6, n)})
        roles = {c.name: c.role for c in dc.card_of({"t": frame}).tables[0].columns}
        self.assertEqual(roles, {"customer_id": "identifier", "country_code": "categorical_code", "amount": "numeric", "created": "datetime",
                                 "note": "free_text", "rating": "categorical_code"})

    def test_a_small_column_keeps_no_quantile_and_a_kept_category_only_when_asked(self):
        small = dc.column_card("salary", pd.Series([51000.0, 72000.0, 64000.0]))
        self.assertIsNone(small.quantiles)
        city = pd.Series(["Rome"] * 30 + ["Milan"] * 20)
        self.assertIsNone(dc.column_card("city", city).categories)
        self.assertEqual(dc.column_card("city", city, keep_categories=True).categories, ["Rome", "Milan"])


@unittest.skipIf(MISSING, MISSING)
class PrivacyTest(unittest.TestCase):
    def test_no_extreme_no_exact_day_no_level_of_one_person(self):
        rng = np.random.default_rng(1)
        salary = pd.Series(rng.normal(60_000, 9_000, 200).round(2))
        card = dc.column_card("salary", salary)
        self.assertFalse(set(card.quantiles) & set(salary.round(2)))  # neither the least nor the most, nor any one salary
        self.assertTrue(all(q % 100 == 0 for q in card.quantiles), card.quantiles)  # rounded to a hundredth of the spread
        paid = pd.Series(np.r_[rng.integers(30, 90, 199) * 1000, 250_000])  # one person earns far more than the rest
        self.assertNotIn(250_000.0, dc.column_card("paid", paid).quantiles)
        births = pd.Series(pd.to_datetime(["1951-03-07"] + ["1988-11-02"] * 30 + ["1990-01-01"] * 30))
        born = dc.column_card("registered", births)
        self.assertNotIn("1951-03-07", (born.start, born.end))  # the oldest person's birthday is not the range
        self.assertIsNone(dc.column_card("registered", births.head(3)).start)  # three people: no range at all
        self.assertIsNone(dc.column_card("dob", births).start)  # named a date of birth: no range of any size (23.7)
        age = pd.Series([30] * 25 + [97])
        self.assertIsNone(dc.column_card("age", age).categories)  # one 97-year-old is a person
        self.assertEqual(dc.column_card("age", pd.Series([30] * 25 + [40] * 6)).categories, ["30", "40"])

    def test_odd_tables_still_have_a_card_and_a_draw(self):
        n = 300
        frame = pd.DataFrame({"customer_id": np.arange(1, n + 1), "store_id": np.arange(n) % 10 + 1, "paid": np.arange(n) % 3 == 0,
                              "at": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"), "amount": np.where(np.arange(n) % 20 == 0, np.inf, np.arange(n) * 1.5),
                              0: np.where(np.arange(n) % 10 == 0, np.nan, np.arange(n) % 4), 1: np.arange(n) % 4, "x" * 190: np.arange(n), "y" * 190: np.arange(n) % 7})
        card = dc.card_of({"t": frame})
        cols = {c.name: c for c in card.tables[0].columns}
        self.assertIsNotNone(cols["amount"].quantiles)  # an inf never costs the card
        drawn = dc.synthesize(card, seed=0, rows={"t": 1000})["t"]
        self.assertEqual(drawn["paid"].dtype, bool)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(drawn["at"]))
        self.assertEqual(drawn["customer_id"].nunique(), 1000)
        self.assertFalse(set(drawn["customer_id"]) & set(frame["customer_id"]))  # invented above the source's largest
        again = {c.name: c for c in dc.card_of({"t": drawn}).tables[0].columns}
        for name in ("paid", "at", "customer_id", "amount"):
            self.assertEqual((again[name].role, again[name].dtype), (cols[name].role, cols[name].dtype), name)

    def test_invented_ids_avoid_the_taken_ones(self):
        rng = np.random.default_rng(0)
        card = dc.column_card("order_id", pd.Series(np.arange(10, 40)))
        taken = {f"{float(x):.12g}" for x in range(10_000, 20_000)}  # inside the range the draw uses (10,000 to 99,999 here)
        self.assertFalse({f"{float(x):.12g}" for x in dc._ids(rng, card, 200, taken)} & taken)


@unittest.skipIf(MISSING, MISSING)
class DraftTest(unittest.TestCase):
    def test_a_drafts_card_takes_its_label_file_when_it_comes_after(self):
        from dclab_rnd.draft import pipeline
        from dclab_rnd.draft.store import DraftStore

        store = DraftStore(Path(tempfile.mkdtemp()))
        draft = store.create("Find the frauds among the rides")["id"]
        replica = dfr.generate(seed=0, n_orders=3000)
        for name, frame in (("rides.csv", replica.rides), ("known_frauds.csv", replica.sample)):
            frame.to_csv(store.data_dir(draft) / name, index=False)
            asset = pipeline.new_asset(store, draft, "upload", name, name)
            pipeline.run(store, draft, asset["id"])
        card = dc.DataCard.read(store.get(draft)["data_card"])
        self.assertEqual([t.name for t in card.tables], ["rides.csv", "labels"])
        self.assertIn("rows_not_in_table", {p.code for p in card.problems})


@unittest.skipIf(MISSING, MISSING)
class ProductTest(unittest.TestCase):
    def test_a_project_and_a_draft_keep_their_tables_card(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        pid = client.post("/api/projects", json={"name": "Card", "goal": "g"}, headers=h).json()["id"]
        frame = dfr.generate(seed=0, n_orders=600).rides
        r = client.put(f"/api/projects/{pid}/data?filename=rides.csv", content=frame.to_csv(index=False).encode(), headers=h)
        self.assertEqual(r.status_code, 200, r.text[:300])
        card = dc.DataCard.read(r.json()["data"]["card"])
        self.assertEqual({c.name: c.role for c in card.tables[0].columns}["terminal_ref"], "identifier")
        self.assertIn("empty_by_category", {p.code for p in card.problems})


if __name__ == "__main__":
    unittest.main()
