"""Package 21.3: files understood together. The two challenge-shaped files, uploaded to separate drafts, are recognised:
the second is a file of known cases, the first the table it labels (key order_request_id), offered with its
coverage and brought into one draft, where the cases join on that key and the key's repeats are reported. A table
that merely shares a column is not taken for a label file."""

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd
    from fastapi.testclient import TestClient

    from dclab_core.replicas import delivery_fraud as dfr
    from dclab_rnd.studio import labels
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""

KEY = "order_request_id"  # as the challenge's files name the order


@unittest.skipIf(MISSING, MISSING)
class TogetherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agentic.server import create_app

        cls.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        cls.h = {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}
        replica = dfr.generate(seed=0, n_orders=3000)
        cls.rides = replica.rides.rename(columns={"order_id": KEY})
        cls.cases = replica.sample.rename(columns={"order_id": KEY})

    def draft(self, problem, name, frame):
        d = self.client.post("/api/drafts", json={"problem": problem}, headers=self.h).json()["id"]
        r = self.client.put(f"/api/drafts/{d}/data", params={"filename": name}, content=frame.to_csv(index=True).encode(), headers=self.h)
        self.assertEqual(r.status_code, 200, r.text[:300])
        return d, self.ready(d)

    def ready(self, d, assets=1):
        deadline = time.time() + 120
        while time.time() < deadline:
            draft = self.client.get(f"/api/drafts/{d}").json()
            done = [a for a in draft["assets"] if a.get("status") in ("ready", "failed")]
            if len(done) >= assets and len(done) == len(draft["assets"]):
                return draft
            time.sleep(0.2)
        self.fail("the pipeline did not finish")

    def test_two_files_uploaded_apart_are_read_together(self):
        orders, _ = self.draft("The orders of the delivery service", "clean.csv", self.rides)  # a name of the draft's own files
        cases_draft, cases = self.draft("Find the fraud", "part2-fraud-sample.csv", self.cases)
        found = cases["analysis"]["cases_table"]
        self.assertIn(KEY, found["keys"])  # a file of known cases, not a table to learn from
        offer = found["tables"][0]
        self.assertEqual((offer["draft_id"], offer["key"]), (orders, KEY))
        self.assertGreaterEqual(offer["coverage"], 0.9)  # most of the cases are orders of that table
        self.assertGreater(offer["key_repeats"], 0)  # an order with two or three drop-offs repeats its key
        r = self.client.post(f"/api/drafts/{cases_draft}/data/from-draft", json={"draft": orders}, headers=self.h)
        self.assertEqual(r.status_code, 200, r.text)
        together = self.ready(cases_draft, assets=2)
        review = together["analysis"]["labels"]
        self.assertEqual(review["key"], [KEY])
        self.assertGreater(review["repeats"]["keys_repeated"], 0)  # found before the join, and how they are resolved
        self.assertEqual((review["repeats"]["key"], review["repeats"]["labelled_keys_repeated"], review["repeats"]["finer_key"]),
                         ([KEY], 0, ["terminal_ref"]))
        self.assertEqual(review["label_rows"], len(self.cases))  # the cases survived the copy: nothing overwrote them
        self.assertEqual(self.client.get(f"/api/drafts/{orders}").json()["assets"][0]["status"], "ready")  # the other draft untouched
        for bad, code in (({"draft": cases_draft}, 422), ({"draft": "dnone"}, 404), ({}, 422), ({"draft": orders}, 409)):  # twice: 409
            self.assertEqual(self.client.post(f"/api/drafts/{cases_draft}/data/from-draft", json=bad, headers=self.h).status_code, code, bad)

    def test_a_table_that_merely_shares_a_column_is_not_a_label_file(self):
        rng = np.random.default_rng(0)
        bikers = self.rides[["biker_id"]].dropna().drop_duplicates().assign(
            joined_on=lambda f: pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 700, len(f)), unit="D"),
            vehicle=lambda f: rng.choice(["bike", "scooter", "car"], len(f)), rating=lambda f: rng.normal(4.5, 0.3, len(f)).round(2))
        orders, _ = self.draft("The orders again", "orders.csv", self.rides)
        _, alone = self.draft("Our bikers and their vehicles", "bikers.csv", bikers)
        self.assertIsNone(alone["analysis"].get("cases_table"))  # a table of bikers, not of cases
        r = self.client.put(f"/api/drafts/{orders}/data", params={"filename": "bikers.csv"}, content=bikers.to_csv(index=False).encode(),
                            headers=self.h)
        self.assertEqual(r.status_code, 200)
        both = self.ready(orders, assets=2)
        self.assertFalse([a for a in both["assets"] if a.get("kind") == "labels"])  # it shares biker_id; it labels nothing
        self.assertNotIn("labels", both["analysis"])

    def test_a_file_sharing_few_of_its_keys_is_not_offered(self):
        d, _ = self.draft("Predict which orders arrive late", "orders.csv", self.rides)
        store = self.client.app.state.services.drafts
        self.assertTrue([o for o in labels.tables_it_labels(store, "elsewhere", self.cases, [KEY]) if o["draft_id"] == d])
        few = self.cases.copy()
        few[KEY] = np.where(np.arange(len(few)) < len(few) // 5, few[KEY], few[KEY] + 10**9)  # four in five keys match nothing
        self.assertFalse([o for o in labels.tables_it_labels(store, "elsewhere", few, [KEY]) if o["draft_id"] == d])

    def test_the_cases_first_and_their_table_later_are_offered_too(self):
        from dclab_rnd.agentic.server import create_app

        shared, token = self.client, self.h
        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))  # a workspace of its own: no table before the cases
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}
        self.addCleanup(setattr, self, "client", shared)
        self.addCleanup(setattr, self, "h", token)
        cases_draft, cases = self.draft("Find the fraud among the rides", "known.csv", self.cases)
        self.assertEqual(cases["analysis"]["cases_table"]["tables"], [])  # nothing to offer yet
        orders, _ = self.draft("Rides of the delivery service", "rides.csv", self.rides)
        deadline, offers = time.time() + 30, []
        while time.time() < deadline and orders not in [o["draft_id"] for o in offers]:  # refreshed just after the table is ready
            offers = self.client.get(f"/api/drafts/{cases_draft}").json()["analysis"]["cases_table"]["tables"]
            time.sleep(0.1)
        self.assertIn(orders, [o["draft_id"] for o in offers])

    def test_a_table_the_cases_would_not_label_is_neither_offered_nor_brought(self):
        small = self.rides[self.rides[KEY].isin(self.cases[KEY])].drop_duplicates(KEY)  # every row a case: the cases are no label file of it
        small_draft, _ = self.draft("A few of the rides", "few.csv", small)
        cases_draft, cases = self.draft("Find the fraud in these rides", "cases.csv", self.cases)
        self.assertNotIn(small_draft, [o["draft_id"] for o in cases["analysis"]["cases_table"]["tables"]])
        r = self.client.post(f"/api/drafts/{cases_draft}/data/from-draft", json={"draft": small_draft}, headers=self.h)
        self.assertEqual(r.status_code, 409, r.text)

    def test_a_labelled_order_with_two_drops_is_found_before_the_join(self):
        multi = self.rides[self.rides[KEY].duplicated(keep=False)].drop_duplicates(KEY).head(5)  # the first drop of five multi-drop orders
        cases = pd.concat([self.cases.iloc[5:], multi.assign(reason="fake_trip")], ignore_index=True)
        review = labels.review(self.rides, cases)
        self.assertEqual(review["key"], ["terminal_ref"])  # the finer key marks only the labelled drop
        self.assertEqual((review["repeats"]["key"], review["repeats"]["labelled_keys_repeated"]), ([KEY], 5))
        self.assertIn("only the labelled rows are marked", review["repeats"]["resolution"])


if __name__ == "__main__":
    unittest.main()
