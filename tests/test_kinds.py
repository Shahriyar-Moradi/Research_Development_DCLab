"""Package 16.1: the problem-kind interface, its registry, and a toy kind that runs through all five stages with no
edit outside its own definition (here, this test file)."""

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import kinds  # noqa: E402
from dclab_rnd.kinds.binary import Binary  # noqa: E402


class LateFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grace_days: int = Field(default=30, ge=0, le=365)


class LatePayment(Binary):
    """A toy kind: the target is the days an invoice was paid late; the outcome is "later than the grace period"."""

    key = "toy_late_payment"
    title = "Late payment (toy)"
    specificity = 5
    words = r"\b(late payment|paid late)\b"
    questions = ("How many days late count as late?",)
    metrics = (("average_precision", "Average precision"), ("roc_auc", "ROC-AUC"))
    Fields = LateFields

    def target(self, frame: pd.DataFrame, solution: Any):
        days = pd.to_numeric(frame[solution.target], errors="coerce")
        return (days > solution.kind_fields["grace_days"]).astype(int), None

    def default_metric(self, solution, positive_rate, pack=None):
        return "average_precision"

    def extra_recipes(self, spec, X, roles):
        return {"toy_log": {"description": "raw + signed log of the numeric inputs (the toy kind's recipe)", "derive": ("log_numeric",)}}

    def model_families(self, available, rare=False):
        return [f for f in available if f in ("logistic_regression", "extra_trees")]


def invoices(n=900, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"invoice_id": [f"I{i:05d}" for i in range(n)], "amount": rng.gamma(2, 400, n).round(2),
                          "customer_age_days": rng.integers(10, 3000, n), "earlier_late": rng.poisson(0.8, n),
                          "issued": pd.date_range("2025-01-01", periods=n, freq="8h").strftime("%Y-%m-%d %H:%M:%S")})
    risk = 0.002 * frame.amount ** 0.5 + 0.9 * frame.earlier_late - 0.0004 * frame.customer_age_days
    frame["days_late"] = np.clip(rng.normal(10 + 18 * risk, 12), 0, None).round()
    return frame


class RegistryTest(unittest.TestCase):
    def test_the_built_in_kinds_and_their_parts(self):
        keys = [k.key for k in kinds.all_kinds()]
        self.assertTrue({"binary", "multiclass", "regression", "text_tabular", "rare_event"} <= set(keys))
        self.assertLess(keys.index("rare_event"), keys.index("binary"))  # the specific kind is asked first
        for kind in kinds.all_kinds():
            described = kind.describe()
            self.assertIn(kind.base_task, kinds.BASE_TASKS)
            self.assertTrue(described["metrics"] and described["questions"], kind.key)
        self.assertEqual(kinds.for_pack("imbalanced", "binary").key, "rare_event")
        self.assertIsNone(kinds.for_pack("imbalanced", "regression"))
        self.assertIsNone(kinds.for_pack("tabular"))  # the tabular pack spans tasks: the task decides

    def test_resolution_follows_the_declared_kind_then_the_pack_then_text(self):
        base = {"target": "y", "prediction_moment": "Before the outcome is known"}
        self.assertEqual(kinds.resolve({**base, "task": "binary"}).key, "binary")
        self.assertEqual(kinds.resolve({**base, "task": "binary"}, "imbalanced").key, "rare_event")
        self.assertEqual(kinds.resolve({**base, "task": "multiclass"}, "imbalanced").key, "multiclass")
        self.assertEqual(kinds.resolve({**base, "task": "binary", "text_columns": ["note"]}).key, "text_tabular")
        self.assertEqual(kinds.resolve({**base, "task": "binary", "kind": "rare_event"}).key, "rare_event")

    def test_a_packs_kind_applies_only_when_it_fits(self):
        base = {"target": "y", "prediction_moment": "Before the outcome is known", "task": "binary"}
        self.assertEqual(kinds.resolve({**base, "text_columns": []}, "text").key, "binary")  # no text: the service export stays
        self.assertEqual(kinds.resolve({**base, "text_columns": ["note"]}, "text").key, "text_tabular")

    def test_the_proposal_keeps_the_tasks_metric_whatever_the_pack(self):
        from dclab_rnd.studio import data as sd
        from dclab_rnd.studio.solution import propose

        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"x": rng.normal(size=500), "y": (rng.uniform(size=500) < 0.3).astype(int)})
        plain, packed = (propose(frame, sd.profile_table(frame), "y", "binary", pack) for pack in (None, "imbalanced"))
        self.assertEqual((plain["metric"], packed["metric"]), ("roc_auc", "roc_auc"))  # as before 16.1
        self.assertEqual(packed["kind"], "rare_event")

    def test_detection_reads_the_problem_and_the_table(self):
        self.assertEqual(kinds.detect("Find fraudulent rides")["key"], "rare_event")
        self.assertEqual(kinds.detect("Route each ticket", {"text_candidates": ["body"]})["key"], "text_tabular")
        self.assertEqual(kinds.detect("How many minutes will the delivery take?")["key"], "regression")
        self.assertEqual(kinds.detect("Which team should get the ticket?")["key"], "multiclass")
        self.assertEqual(kinds.detect("")["key"], "binary")
        rare = {"labels": {"rows": 10000, "rows_marked": 40}}
        self.assertEqual(kinds.detect("Find the bad orders", rare)["key"], "rare_event")

    def test_a_kind_must_run_on_a_known_task_and_declare_a_metric(self):
        class Odd(Binary):
            key, base_task = "odd", "survival"

        class Silent(Binary):
            key, metrics = "silent", ()
        with self.assertRaises(ValueError):
            kinds.register(Odd())
        with self.assertRaises(ValueError):
            kinds.register(Silent())
        with self.assertRaises(ValueError):
            kinds.register(Binary())  # a second object under a key already taken

    def test_the_solution_checks_its_kind_and_the_kind_checks_its_fields(self):
        from dclab_rnd.studio.solution import Solution

        base = {"target": "y", "prediction_moment": "Before the outcome is known"}
        self.assertEqual(Solution(**base, task="binary").problem_kind().key, "binary")
        with self.assertRaises(ValueError):
            Solution(**base, task="regression", kind="rare_event")  # the kind runs as binary
        with self.assertRaises(ValueError):
            Solution(**base, task="binary", kind="no_such_kind")
        with self.assertRaises(ValueError):
            Solution(**base, task="binary", kind_fields={"anything": 1})  # binary has no fields of its own
        with self.assertRaises(ValueError):
            Solution(**base, task="multiclass", metric="roc_auc")
        with self.assertRaises(ValueError):
            Solution(**base, task="binary", kind_fields={"grace_days": 3})  # fields come with their declared kind


class ToyKindTest(unittest.TestCase):
    def setUp(self):
        kinds.register(LatePayment(), replace=True)
        self.addCleanup(kinds.unregister, "toy_late_payment")

    def test_the_toy_kind_validates_its_fields(self):
        from dclab_rnd.studio.solution import Solution

        base = {"target": "days_late", "task": "binary", "kind": "toy_late_payment", "prediction_moment": "When the invoice is issued"}
        self.assertEqual(Solution(**base).kind_fields, {"grace_days": 30})
        with self.assertRaises(ValueError):
            Solution(**base, kind_fields={"grace_days": -3})
        with self.assertRaises(ValueError):
            Solution(**base, metric="recall_at_budget")  # not one of the toy kind's metrics
        self.assertEqual(kinds.detect("Which invoices will be paid late?")["key"], "toy_late_payment")

    def test_the_toy_kind_runs_through_all_five_stages(self):
        from dclab_rnd.expansion import runner
        from dclab_rnd.studio import ProjectStore, data as sd, engine
        from dclab_rnd.studio.solution import Solution

        store = ProjectStore(Path(tempfile.mkdtemp()))
        project = store.create("Invoices", "finance", "Which invoices will be paid late?")
        invoices().to_csv(store.data_dir(project["id"]) / "invoices.csv", index=False)
        sd.attach_data(store, project["id"], "invoices.csv")
        project = store.get(project["id"])
        project["solution"] = Solution(target="days_late", task="binary", kind="toy_late_payment", kind_fields={"grace_days": 25},
                                       identifiers=["invoice_id"], time_column="issued",
                                       prediction_moment="When the invoice is issued, before any payment.").model_dump()
        project["settings"]["quick"] = True
        store.save(project)
        with runner.fast_profile():
            engine.run_all(store, project["id"])
        records = {stage: store.read_stage(project["id"], stage) for stage in engine.STAGE_KEYS}
        data = records["data"]["evidence"]
        self.assertEqual(data["problem_kind"], "toy_late_payment")
        y = (invoices()["days_late"] > 25).astype(int)
        self.assertAlmostEqual(data["target_summary"]["positive_rate_train"], y.mean(), delta=0.08)  # its own target
        self.assertEqual(records["final"]["primary_metric"], "average_precision")
        self.assertIn("toy_log", {r["recipe"] for r in records["features"]["evidence"]["stage_results"]})
        screened = {r["model"] for r in records["models"]["evidence"]["ranking"]}
        self.assertEqual(screened, {"logistic_regression", "extra_trees"})
        self.assertIn("average_precision", records["final"]["evidence"]["holdout_metrics"])
        from dclab_rnd.studio import export

        with self.assertRaises(export.ExportUnsupported):  # its own target has no notebook code yet: refused, never wrong
            export.notebook(store.get(project["id"]), records)


class ApiTest(unittest.TestCase):
    def test_api_kinds_lists_every_kind_with_its_parts(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        listed = {k["key"]: k for k in client.get("/api/kinds").json()["kinds"]}
        self.assertEqual(listed["rare_event"]["base_task"], "binary")
        self.assertIn("imbalanced", listed["rare_event"]["packs"])
        self.assertIn("budget", listed["rare_event"]["home_fields"])
        self.assertFalse(listed["text_tabular"]["export"]["service"])
        self.assertIn("kinds", client.get("/api/studio").json())
        body = client.get("/api/kinds").json()
        self.assertEqual({r["kind"] for r in body["matrix"]}, set(listed))  # 16.2: the matrix, one row per kind
        self.assertIn("16.14", body["note"])
        packs = {p["key"]: p for p in client.get("/api/packs").json()}
        self.assertIn(packs["imbalanced"]["maturity"], ("Supported", "Partial"))
        self.assertEqual(packs["scenegraph"]["maturity"], "Not built")
        self.assertIn("capability", packs["tabular"])


class BundleTest(unittest.TestCase):
    def test_unstratified_groups_still_hold_both_classes_in_a_classifications_holdout(self):
        from dclab_rnd.expansion import datasets as ds

        rng = np.random.default_rng(3)
        n = 1200
        X = pd.DataFrame({"g": rng.integers(0, 60, n), "x": rng.normal(size=n)})
        y = pd.Series((X["g"] < 3).astype(int))  # positives on three entities only
        spec = ds.DatasetSpec(key="t", name="t", task_type="binary_imbalanced", primary_metric="average_precision", target="y", url="",
                              filename="", sha256="", description="", decision_time_contract="", split_strategy="group", source_rows=n,
                              feature_count=2, group_column="g")
        bundle = ds.prepare_bundle(spec, X, y, stratify=False)
        self.assertEqual(bundle.y_test.nunique(), 2)


if __name__ == "__main__":
    unittest.main()
