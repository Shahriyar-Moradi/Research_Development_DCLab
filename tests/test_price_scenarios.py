"""Price scenarios through portable arithmetic, simulation tables and the actual Studio API."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from dclab_core.price_scenarios import (
    PriceScenario,
    assumptions,
    downside,
    fit_study,
    histories,
    price_row,
    study_rows,
)
from dclab_rnd.agent_eval.price_scenarios import generate, spec_for
from dclab_rnd.agentic.server import create_app


class PriceScenariosTest(unittest.TestCase):
    def test_two_randomized_arms_leave_the_untested_price_blank(self):
        offers, events, labels, _ = generate("sales", 19)
        train_ids = offers.loc[offers.split == "train", "offer_id"].iloc[:200].tolist()
        keep = offers.offer_id.isin(train_ids) | (offers.split == "score")
        offers = offers[keep].copy()
        offers.loc[offers.offer_id.isin(train_ids[:100]), "price"] = 100
        offers.loc[offers.offer_id.isin(train_ids[100:]), "price"] = 110
        labels = labels[labels.offer_id.isin(train_ids)].copy()
        labels["outcome"] = labels.offer_id.isin(train_ids[:20] + train_ids[100:115]).astype(int)
        report = fit_study(offers, events, labels, spec_for("sales"))
        self.assertIsNone(report["effects"][0])
        self.assertAlmostEqual(report["effects"][1]["change_pp"], -5)
        self.assertIsNone(report["rates"][1])

    def test_exact_example_and_risk(self):
        spec = PriceScenario(
            price=100, audience=100, target_revenue=2000, probabilities=[None, None, 0.18], independent=True
        )
        out = assumptions(spec)
        self.assertEqual([r["break_even_units"] for r in out["rows"]], [20, 20, 19])
        last = out["rows"][2]
        self.assertAlmostEqual(last["expected_revenue"], 1980)
        self.assertAlmostEqual(last["expected_events"], 18)
        self.assertAlmostEqual(last["downside_probability"], 0.56255241683753)
        self.assertIsNone(out["rows"][0]["expected_events"])
        self.assertEqual(price_row(110, 2000, "purchase", [1, 1], [0.2, 0.8], False)["expected_revenue"], 110)
        churn = price_row(110, 2000, "churn", [1, 1], [0.1, 0.2], False)
        self.assertAlmostEqual(churn["event_rate"], 0.15)
        self.assertAlmostEqual(churn["expected_revenue"], 187)
        self.assertIsNone(churn["downside_probability"])
        self.assertEqual(downside([100], [0], 0), 0)
        self.assertEqual(downside([100], [1], 101), 1)
        for p in (-0.01, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                PriceScenario(price=1, audience=10, target_revenue=0, probabilities=[p, 0, 1])

    def test_event_time_and_stable_order(self):
        offers = pd.DataFrame(
            {"offer_id": [1, 2], "customer_id": ["A", "B"], "offered_at": ["2026-01-10"] * 2}
        )
        events = pd.DataFrame(
            {
                "event_id": list("abcdefgx"),
                "customer_id": ["A"] * 8,
                "occurred_at": ["2026-01-01"] * 6 + ["2026-01-04", "2026-01-10"],
                "available_at": ["2026-01-02"] * 6 + ["2026-01-11", "2026-01-10"],
                "discounted": [1, 0, 0, 0, 0, 1, 1, 1],
                "paid_price": [90, 100, 100, 100, 100, 100, 120, 120],
            }
        )
        out = histories(offers, events)
        self.assertEqual(out.loc[1, "history_count"], 5)
        self.assertAlmostEqual(out.loc[1, "discount_share"], 0.2)
        self.assertEqual(out.loc[1, "last_paid"], 100)
        self.assertTrue(pd.isna(out.loc[2, "discount_share"]))
        duplicated = pd.concat([events, events.iloc[[0]]]).sample(frac=1, random_state=19)
        pd.testing.assert_frame_equal(out, histories(offers, duplicated))

    def test_study_safety_on_generator(self):
        offers, events, labels, _ = generate("sales", 19)
        events["paid_price"] = 100
        spec = spec_for("sales")
        baseline = fit_study(offers, events, labels, spec)
        self.assertEqual(baseline["effect_source"], "randomized estimate")
        self.assertAlmostEqual(baseline["mean_change_from_last_paid"][2], 0.1)
        edited = labels.copy()
        edited.loc[edited.offer_id.isin(offers.loc[offers.split == "score", "offer_id"]), "outcome"] = 99
        offers["post_offer_confirmation"] = 1 - offers.post_offer_confirmation
        self.assertEqual(baseline, fit_study(offers, events, edited, spec))
        fixed = generate("fixed", 19)[:3]
        self.assertNotIn("effects", fit_study(*fixed, spec_for("fixed")))
        late = generate("late", 19)[:3]
        self.assertGreater(fit_study(*late, spec_for("late"))["excluded_training"], 0)
        unknown = spec.model_copy(update={"design": "unknown"})
        self.assertNotIn("effects", fit_study(offers, events, labels, unknown))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            bad = offers.copy()
            bad.loc[bad.index[-1], "customer_id"] = bad.customer_id.iloc[0]
            study_rows(bad, events, labels, 30)

    def test_normal_product_import_prepare_read_and_stale_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(Path(tmp))
            client = TestClient(app)
            headers = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
            project = app.state.projects.create("Price study")
            base = "/api/projects/" + project["id"]
            # A related-table project needs its ordinary main table first.
            main = client.put(
                base + "/data",
                params={"filename": "main.csv"},
                content=pd.DataFrame({"sku": range(40), "units": [1, 2] * 20}).to_csv(index=False).encode(),
                headers=headers,
            )
            self.assertEqual(main.status_code, 200, main.text[:1000])
            offers, events, labels, _ = generate("sales", 19)
            for name, frame in zip(
                ("price_offers", "price_events", "price_outcomes"), (offers, events, labels)
            ):
                response = client.put(
                    base + "/tables/" + name,
                    params={"filename": name + ".csv"},
                    content=frame.to_csv(index=False).encode(),
                    headers=headers,
                )
                self.assertEqual(response.status_code, 200, response.text[:1000])
            body = spec_for("sales").model_dump()
            response = client.post(base + "/price-scenarios/prepare", json=body, headers=headers)
            self.assertEqual(response.status_code, 200, response.text[:1000])
            saved = copy.deepcopy(app.state.projects.get(project["id"]))
            with patch(
                "dclab_rnd.studio.price_scenarios.fit_study", side_effect=AssertionError("unexpected fit")
            ):
                response = client.get(
                    base + "/price-scenarios", params={"target_revenue": 2000}, headers=headers
                )
                self.assertEqual(response.status_code, 200, response.text[:1000])
                self.assertEqual(response.json()["rows"][0]["audience"], 400)
                self.assertEqual(response.json()["source"], "model prediction")
                self.assertEqual(response.json()["version"], saved["price_study"]["version"])
                self.assertEqual(app.state.projects.get(project["id"]), saved)
                scenario = client.post(
                    base + "/price-scenarios/assumptions",
                    headers=headers,
                    json={
                        "price": 100,
                        "audience": 100,
                        "target_revenue": 2000,
                        "probabilities": [0.2, None, 0.18],
                    },
                )
                self.assertAlmostEqual(scenario.json()["rows"][-1]["expected_revenue"], 1980)
            saved["tables"]["price_offers"]["sha256"] = "changed"
            app.state.projects.update(project["id"], tables=saved["tables"])
            self.assertEqual(client.get(base + "/price-scenarios", headers=headers).status_code, 409)
            self.assertEqual(
                client.get("/api/projects/missing/price-scenarios", headers=headers).status_code, 404
            )
            self.assertEqual(
                client.get(base + "/price-scenarios?target_revenue=-2", headers=headers).status_code, 422
            )
