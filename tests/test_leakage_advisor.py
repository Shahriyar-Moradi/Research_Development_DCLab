"""Leakage and Prediction-Moment Advisor tests (Unified Agentic Roadmap 2026-11-19):
- Metadata-only (cell_values=0): zero raw row values sent to any model client.
- Scans all non-identifier columns + heuristic suspects inside stage_leakage.
- Explicitly tells user when a feature is mostly / very likely leakage and recommends comparing BOTH
  clean (without leakage) and leaky (with leakage) on training folds vs. excluding directly.
- Never mutates solution.forbidden automatically; requires explicit confirmation.
- Comparing clean vs. leaky runs strictly on training folds and never reads holdout.
- Changing the prediction moment invalidates prior leakage advice (409 on stale confirmation).
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agentic.server import create_app  # noqa: E402
from dclab_rnd.studio import data as studio_data, engine as studio_engine, leakage_advisor  # noqa: E402
from dclab_rnd.studio.store import ProjectStore  # noqa: E402


def _make_delivery_dataset(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dist = rng.uniform(1.0, 12.0, n)
    items = rng.integers(1, 6, n)
    couriers = rng.integers(5, 30, n)
    base_time = pd.date_range("2026-01-01", periods=n, freq="h")
    actual = np.round(12.0 + 2.4 * dist + 0.8 * items - 0.15 * couriers + rng.normal(0, 2.0, n), 2)
    # Post-delivery leakage feature tightly correlated with target and carrying sentinel values we check never leak to LLM!
    courier_rating = np.round(5.0 - 0.08 * actual + rng.normal(0, 0.02, n), 3)
    secret_city = [f"SENTINEL_CELL_ROW_{i}" for i in range(n)]
    return pd.DataFrame({
        "order_id": [f"ORD-{i:04d}" for i in range(n)],
        "order_time": base_time,
        "distance_km": dist,
        "basket_items": items,
        "couriers_online": couriers,
        "city_note": secret_city,
        "courier_rating": courier_rating,
        "actual_minutes": actual,
    })


class _SpyModelClient:
    def __init__(self, reply_obj):
        self.reply_obj = reply_obj
        self.sent = []

    def complete(self, messages, tools=None, max_tokens=None):
        self.sent.append(messages)
        return {"content": json.dumps(self.reply_obj), "tool_calls": [], "usage": {}}


class LeakageAdvisorTests(unittest.TestCase):
    def _build_project(self, home: Path):
        store = ProjectStore(home / "projects")
        df = _make_delivery_dataset(120)
        p = store.create("Delivery ETA", "logistics and delivery", "Predict actual_minutes at order creation")
        pid = p["id"]
        csv_path = store.data_dir(pid) / "orders.csv"
        df.to_csv(csv_path, index=False)
        studio_data.attach_data(store, pid, "orders.csv")
        p = store.get(pid)
        p["solution"] = {
            "target": "actual_minutes",
            "task": "regression",
            "metric": "mae",
            "identifiers": ["order_id"],
            "time_column": "order_time",
            "prediction_moment": "At order creation before dispatch; exclude courier rating and delivery timestamps.",
            "forbidden": [],  # courier_rating intentionally NOT declared yet so advisor must detect & prompt!
        }
        store.save(p)
        return store, pid, df

    def test_metadata_only_zero_cell_values_and_prompt_recommends_compare_both(self):
        home = Path(tempfile.mkdtemp())
        store, pid, df = self._build_project(home)
        p = store.get(pid)
        profile = p["data"]["profile"]

        spy = _SpyModelClient({
            "columns": [{
                "column": "courier_rating",
                "verdict": "after the moment",
                "reason": "At order creation before dispatch, courier_rating is only recorded after delivery.",
                "records": ["DCLAB-R01"],
            }]
        })
        adv = leakage_advisor.advise(
            columns_meta=profile["columns"],
            prediction_moment=p["solution"]["prediction_moment"],
            target="actual_minutes",
            identifiers=["order_id"],
            time_column="order_time",
            declared_forbidden=[],
            heuristic_candidates=[{"feature": "courier_rating", "reasons": ["extreme_univariate_signal"]}],
            univariate_ranking=[{"feature": "courier_rating", "signal": 0.99}, {"feature": "distance_km", "signal": 0.72}],
            client=spy,
            problem_revision=3,
        )
        # Zero raw cell values ever sent to model client.
        self.assertEqual(adv["cell_values_sent"], 0)
        self.assertEqual(len(spy.sent), 1)
        prompt_dump = json.dumps(spy.sent[0])
        self.assertNotIn("SENTINEL_CELL_ROW_", prompt_dump)
        self.assertNotIn("ORD-0001", prompt_dump)

        # Scans all non-identifier non-target non-time columns.
        rec_cols = {r["column"]: r for r in adv["recommendations"]}
        self.assertEqual(set(rec_cols), {"distance_km", "basket_items", "couriers_online", "city_note", "courier_rating"})
        self.assertEqual(rec_cols["courier_rating"]["action"], "block")
        self.assertEqual(rec_cols["courier_rating"]["likelihood"], "very_likely")
        self.assertTrue(rec_cols["courier_rating"]["provisional"])
        self.assertFalse(rec_cols["courier_rating"]["carries_approval"])
        self.assertIn("mostly / very likely leakage", rec_cols["courier_rating"]["rationale"])

        # Explicit user prompt recommends comparing BOTH clean vs. leaky on training folds.
        prompt = adv["likely_leakage_prompt"]
        self.assertIsNotNone(prompt)
        self.assertEqual(prompt["recommended_choice"], "compare_both")
        self.assertIn("mostly / very likely leakage", prompt["message"])
        opts = {o["id"]: o for o in prompt["options"]}
        self.assertTrue(opts["compare_both"]["recommended"])
        self.assertFalse(opts["exclude_directly"]["recommended"])

    def test_stage_leakage_emits_provisional_advice_compare_both_is_train_only_and_moment_change_invalidates(self):
        home = Path(tempfile.mkdtemp())
        app = create_app(home)
        client = TestClient(app)
        s = app.state.services
        store, pid, _df = self._build_project(home)

        # Run stage_data then stage_leakage (deterministic, no live model).
        studio_engine.execute(store, pid, "data")
        rec = studio_engine.execute(store, pid, "leakage")
        adv = rec["evidence"]["leakage_advisor"]
        self.assertTrue(adv["provisional"])
        self.assertTrue(adv["requires_confirmation"])
        self.assertIn("courier_rating", adv["likely_leakage_columns"])
        # Advisor NEVER auto-mutates solution.forbidden!
        self.assertEqual(store.get(pid)["solution"]["forbidden"], [])
        self.assertEqual(store.get(pid)["holdout_uses"], 0)

        # Choice 1 (Recommended): compare_both via HTTP API -> runs strictly on training folds, holdout untouched!
        headers = {"x-dclab-token": s.csrf}
        r_cmp = client.post(
            f"/api/projects/{pid}/leakage/decide",
            headers=headers,
            json={"choice": "compare_both", "columns": ["courier_rating"]},
        )
        self.assertEqual(r_cmp.status_code, 200, r_cmp.text)
        cmp_body = r_cmp.json()
        self.assertFalse(cmp_body["holdout_touched"])
        self.assertFalse(cmp_body["solution_mutated"])
        comp = cmp_body["comparison"]
        self.assertEqual(comp["scope"], "training_folds_only")
        self.assertFalse(comp["holdout_touched"])
        self.assertEqual(comp["recommended_safe_candidate"], "without_leakage")
        # Leaky shadow feature courier_rating artificially improves MAE (positive oriented gain).
        self.assertGreater(comp["apparent_lift"], 0.0)
        self.assertEqual(store.get(pid)["holdout_uses"], 0)

        # Choice 2: explicit confirmation to exclude_directly updates solution.forbidden cleanly.
        r_ex = client.post(
            f"/api/projects/{pid}/leakage/decide",
            headers=headers,
            json={"choice": "exclude_directly", "columns": ["courier_rating"]},
        )
        self.assertEqual(r_ex.status_code, 200, r_ex.text)
        self.assertTrue(r_ex.json()["solution_mutated"])
        self.assertIn("courier_rating", [f["column"] for f in store.get(pid)["solution"]["forbidden"]])
        # the review: saved as a person saves a solution: logged as a graph move, stages it invalidates cleared, and every
        # later stage still runs (an extra key in forbidden once made them all fail)
        self.assertIn("set_solution", (home / "projects" / pid / "transitions.jsonl").read_text() if (home / "projects" / pid / "transitions.jsonl").exists()
                      else "".join(str(t) for t in store.records(pid)))
        self.assertFalse(store.has_stage(pid, "leakage"))
        studio_engine.execute(store, pid, "data")
        studio_engine.execute(store, pid, "leakage")
        studio_engine.execute(store, pid, "features")
        adv = store.read_stage(pid, "leakage")["evidence"]["leakage_advisor"]
        bad = client.post(f"/api/projects/{pid}/leakage/decide", headers=headers,
                          json={"choice": "exclude_directly", "columns": ["actual_minutes", "no_such_column"]})
        self.assertEqual(bad.status_code, 422, bad.text)  # never the outcome or a column the advisor never reviewed

        # Prediction moment changes -> prior leakage advisor is automatically invalidated (HTTP 409)!
        p_mut = store.get(pid)
        p_mut["solution"]["prediction_moment"] = "After delivery is finished for post-mortem audit."
        store.save(p_mut)
        freshness = leakage_advisor.check_advisor_freshness(adv, current_moment=p_mut["solution"]["prediction_moment"])
        self.assertTrue(freshness["stale"])
        self.assertFalse(freshness["may_confirm"])

        r_stale = client.post(
            f"/api/projects/{pid}/leakage/decide",
            headers=headers,
            json={"choice": "exclude_directly", "columns": ["courier_rating"]},
        )
        self.assertEqual(r_stale.status_code, 409)
        self.assertEqual(r_stale.json()["detail"]["error"], "stale_leakage_advice")


    def test_a_moment_word_or_a_name_alone_is_a_question_never_a_block(self):
        """The review: `couriers_online` shares a stem with the moment's "courier rating" and was reported very likely
        leakage; only the audit's own verdict or an extreme heuristic blocks."""
        home = Path(tempfile.mkdtemp())
        store, pid, _df = self._build_project(home)
        studio_engine.execute(store, pid, "data")
        adv = studio_engine.execute(store, pid, "leakage")["evidence"]["leakage_advisor"]
        actions = {r["column"]: r["action"] for r in adv["recommendations"]}
        self.assertNotEqual(actions.get("couriers_online"), "block")

    def test_a_stage_asks_no_model_when_the_process_installed_no_gateway(self):
        """The review: a layer or baseline run mocks the stage notes and installs nothing; the advisor fell back to a bare
        gateway built from .env, outside every cap. Only an installed gateway is asked, with the project's id."""
        from unittest import mock

        from dclab_rnd import models

        asked = []

        class Gateway:
            def project_private(self, pid):
                return False

            def available(self, purpose):
                return True

            def client(self, purpose, **kw):
                if purpose == "leakage_review":  # the advisor's own request (stage notes have their own rules)
                    asked.append((purpose, kw.get("project_id")))
                return None

        home = Path(tempfile.mkdtemp())
        store, pid, _df = self._build_project(home)
        studio_engine.execute(store, pid, "data")
        with mock.patch.object(models.gateway, "_INSTALLED", None), mock.patch("dclab_rnd.models.gateway.Gateway", return_value=Gateway()):
            studio_engine.execute(store, pid, "leakage")
        self.assertEqual(asked, [])
        with mock.patch.object(models.gateway, "_INSTALLED", Gateway()):
            studio_engine.execute(store, pid, "leakage")
        self.assertEqual(asked, [("leakage_review", pid)])  # installed: asked once, with the project's id for its cap

    def test_stage_leakage_asks_no_model_inside_notes_off(self):
        """2026-11-19: inside an exam's notes_off the advisor called the process's model, outside the exam's cap and
        sealed vault, in the rules arm too. The stage's rules and heuristics still run."""
        from unittest import mock

        from dclab_rnd.studio import agent

        asked = []

        class Gateway:
            def project_private(self, pid):
                return False

            def available(self, purpose):
                return True

            def client(self, purpose, **kw):
                asked.append(purpose)
                return None

        home = Path(tempfile.mkdtemp())
        store, pid, _df = self._build_project(home)
        studio_engine.execute(store, pid, "data")
        with mock.patch("dclab_rnd.models.installed_only", return_value=Gateway()), agent.notes_off():
            rec = studio_engine.execute(store, pid, "leakage")
        self.assertNotIn("leakage_review", asked)
        self.assertIn("leakage_advisor", rec["evidence"])  # the advisor's rules still ran, with no model


if __name__ == "__main__":
    unittest.main()
