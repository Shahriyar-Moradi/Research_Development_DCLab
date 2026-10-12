"""Package 15.2: a label file next to the table (studio/labels.py, data.attach_labels, the attach_labels move,
review_labels, and the target, coverage and limits in the stage records)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd.studio import ProjectStore, data as sd, engine, graph
    from dclab_rnd.studio.solution import Solution
except ImportError as error:  # pydantic is not part of requirements/ci.txt
    STUDIO_MISSING = str(error)
else:
    STUDIO_MISSING = ""


def files(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    customer = rng.integers(0, 60, n)
    table = pd.DataFrame({"order_id": np.arange(5000, 5000 + n), "customer_ref": [f"C{c:03d}" for c in customer],
                          "amount": np.round(rng.gamma(2, 20, n), 2), "items": rng.integers(1, 6, n),
                          "created": pd.date_range("2026-01-01", periods=n, freq="h").astype(str)})
    risky = (table["amount"] > 70) & (rng.random(n) < 0.5)
    cases = table[risky].sample(frac=0.6, random_state=0)  # a sample of the cases: partial labels
    sample = cases[["order_id", "customer_ref"]].assign(amount=cases["amount"] * 1.3, reason="refund abuse")
    sample = pd.concat([sample, pd.DataFrame({"order_id": [1, 2], "customer_ref": ["C001", "C002"], "amount": [1.0, 2.0],
                                              "reason": ["x", "y"]})], ignore_index=True)  # two rows the table does not have
    return table, sample


@unittest.skipIf(STUDIO_MISSING, STUDIO_MISSING)
class LabelFileTest(unittest.TestCase):
    def project(self):
        store = ProjectStore(Path(tempfile.mkdtemp()))
        project = store.create("Orders", "general", "Find refund abuse")
        table, sample = files()
        directory = store.data_dir(project["id"])
        table.to_csv(directory / "orders.csv", index=False)
        sd.attach_data(store, project["id"], "orders.csv")
        sample.to_csv(directory / "known_cases.csv", index=False)
        return store, project["id"], table, sample

    def test_the_key_is_checked_and_the_target_comes_from_the_file(self):
        from dclab_rnd.studio import labels

        store, pid, table, sample = self.project()
        project = sd.attach_labels(store, pid, "known_cases.csv", target="is_case")
        info = project["labels"]
        self.assertEqual(info["key"], ["order_id"])
        self.assertEqual(info["not_found"], 2)
        self.assertIn("amount", info["disagreeing_columns"])  # same name, another meaning in the label file
        refused = {"+".join(c["key"]): c for c in info["candidates"] if c["verdict"] == "refused"}
        self.assertIn("customer_ref", refused)
        self.assertGreater(refused["customer_ref"]["rows_per_label"], 1.5)
        self.assertEqual(project["suggestion"]["labels"], "partial")
        marked = labels.mark(table, sample, ["order_id"])
        self.assertEqual(int(marked.sum()), info["rows_marked"])
        with self.assertRaises(sd.DataError) as refusal:  # the person may not pick an inflating key without saying it is per entity
            sd.attach_labels(store, pid, "known_cases.csv", target="is_case", key=["customer_ref"])
        self.assertIn("rows from", str(refusal.exception))

    def test_the_engine_reads_the_target_and_records_coverage_and_limits(self):
        from dclab_rnd.expansion import runner

        store, pid, _, _ = self.project()
        project = sd.attach_labels(store, pid, "known_cases.csv", target="is_case")
        project["solution"] = Solution(**{**project["suggestion"], "prediction_moment": "When the refund is requested, before review.",
                                          "identifiers": ["order_id"], "group_column": "customer_ref"}).model_dump()
        project["settings"]["quick"] = True
        store.save(project)
        with runner.fast_profile():
            engine.run_all(store, pid)
        final = store.read_stage(pid, "final")
        self.assertEqual(final["labels"]["coverage"], "partial")
        self.assertIn("lower bound", final["evidence"]["precision_note"])
        limits = " ".join(final["evidence"]["limits"])
        self.assertIn("amount", limits)
        self.assertIn("2 label rows are not in the table", limits)
        self.assertNotIn("reason", store.read_stage(pid, "features")["evidence"]["stage_results"][0].get("description", ""))

    def test_a_new_table_clears_the_labels_and_the_move_is_checked(self):
        store, pid, table, _ = self.project()
        project = store.get(pid)
        self.assertTrue(graph.check(project, "attach_labels", "agent").allowed)
        sd.attach_labels(store, pid, "known_cases.csv", target="is_case")
        table.to_csv(store.data_dir(pid) / "orders2.csv", index=False)
        project = sd.attach_data(store, pid, "orders2.csv")
        self.assertIsNone(project["labels"])
        empty = store.create("Empty", "general", "x")
        self.assertFalse(graph.check(empty, "attach_labels", "human").allowed)  # no table yet

    def test_the_intern_reads_the_label_review(self):
        from dclab_rnd.intern.tools import Toolbox

        store, pid, _, _ = self.project()
        self.assertIsNone(Toolbox(store).review_labels(pid)["labels"])
        sd.attach_labels(store, pid, "known_cases.csv", target="is_case")
        review = Toolbox(store).review_labels(pid)
        self.assertEqual((review["key"], review["coverage"]), (["order_id"], "partial"))


if __name__ == "__main__":
    unittest.main()


class BudgetAndSplitTest(unittest.TestCase):
    """Package 15.3: metrics at an alert budget (hand-computed) and the time-and-entity split."""

    def test_budget_metrics_by_hand(self):
        from dclab_rnd.expansion import runner

        y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        m = runner.binary_metrics(y, scores, alert_budget=0.3)  # review the top 3: rows 0, 1, 2 -> two cases of three
        self.assertEqual((m["alerts_at_budget"], m["recall_at_budget"], m["precision_at_budget"]), (3, 2 / 3, 2 / 3))
        self.assertEqual(runner.binary_metrics(y, scores, alert_budget=0.01)["alerts_at_budget"], 1)  # at least one row
        self.assertAlmostEqual(m["ap_lift"], m["average_precision"] / 0.3)
        self.assertNotIn("recall_at_budget", runner.binary_metrics(y, scores))

    @unittest.skipIf(STUDIO_MISSING, STUDIO_MISSING)
    def test_the_time_and_entity_split_never_puts_an_entity_on_both_sides(self):
        from dclab_rnd.expansion import datasets as ds

        rng = np.random.default_rng(1)
        n = 2400
        day = np.sort(rng.integers(0, 200, n))
        who = np.where(day < 160, rng.integers(0, 300, n), rng.integers(300, 380, n))  # new customers arrive late, a few stay
        stay = np.flatnonzero(day >= 160)[::10]  # a tenth of the late rows come from customers seen early
        who[stay] = rng.integers(0, 300, len(stay))
        X = pd.DataFrame({"day": day, "customer": who, "x": rng.normal(size=n)})
        y = pd.Series((rng.random(n) < 0.2).astype(int))
        spec = ds.DatasetSpec(key="t", name="t", task_type="binary_imbalanced", primary_metric="average_precision", target="y", url="",
                              filename="", sha256="", description="", decision_time_contract="", split_strategy="time_group",
                              source_rows=n, feature_count=3, time_column="day", group_column="customer")
        bundle = ds.prepare_bundle(spec, X, y)
        groups = bundle.X["customer"].astype(str).to_numpy()
        self.assertFalse(set(groups[bundle.train_idx]) & set(groups[bundle.test_idx]))
        self.assertGreaterEqual(bundle.X["day"].iloc[bundle.test_idx].min(), bundle.X["day"].iloc[bundle.train_idx].max())
        for fit, valid in bundle.cv_splits:
            train_groups = bundle.X_train["customer"].astype(str).to_numpy()
            self.assertFalse(set(train_groups[fit]) & set(train_groups[valid]))
        self.assertIn("rows removed", bundle.cv_description)
        everywhere = X.assign(customer=rng.integers(0, 20, n))  # every customer in both periods: refused, never silently dropped
        with self.assertRaises(ValueError) as refusal:
            ds.prepare_bundle(spec, everywhere, y)
        self.assertIn("Declare only one", str(refusal.exception))

    @unittest.skipIf(STUDIO_MISSING, STUDIO_MISSING)
    def test_the_budget_metric_needs_a_budget(self):
        with self.assertRaises(ValueError):
            Solution(target="y", task="binary", prediction_moment="When the order is placed.", metric="recall_at_budget")
        Solution(target="y", task="binary", prediction_moment="When the order is placed.", metric="recall_at_budget", alert_budget=0.01)


@unittest.skipIf(STUDIO_MISSING, STUDIO_MISSING)
class StepTwoReviewTest(unittest.TestCase):
    """Fixes from the Step 2 review: the logged move, the export, the coverage answer, the plan's split, the rare rule."""

    def test_attaching_labels_is_a_logged_move_and_the_notebook_joins_them(self):
        from dclab_rnd.studio import export

        store = ProjectStore(Path(tempfile.mkdtemp()))
        project = store.create("Orders", "general", "Find refund abuse")
        table, sample = files()
        table.to_csv(store.data_dir(project["id"]) / "orders.csv", index=False)
        sd.attach_data(store, project["id"], "orders.csv")
        sample.to_csv(store.data_dir(project["id"]) / "known_cases.csv", index=False)
        project = sd.attach_labels(store, project["id"], "known_cases.csv", target="is_case")
        moves = [t for t in store.transitions(project["id"]) if t["move"] == "attach_labels"]
        self.assertEqual([t["status"] for t in moves], ["allowed"])
        project["solution"] = Solution(**{**project["suggestion"], "prediction_moment": "When the refund is requested."}).model_dump()
        meta = {"title": "", "setup_summary": "", "notes": []}
        records = {"features": {**meta, "decision": {"chosen": "raw"}, "evidence": {}}, "models": {**meta, "decision": {"chosen": "isolation_forest"}}}
        source = "".join("".join(c["source"]) for c in export.notebook(project, records)["cells"])
        self.assertIn("labels = pd.read_csv('known_cases.csv'", source)
        self.assertIn("KEY = ['order_id']", source)
        self.assertIn("from dclab_rnd.expansion.detectors import IsolationForestDetector", source)
        with self.assertRaises(export.ExportUnsupported):
            export.notebook(project, {**records, "features": {**meta, "decision": {"chosen": "entity_behaviour"}, "evidence": {}}})

    def test_a_label_coverage_answer_is_complete_only_when_it_says_so(self):
        from dclab_rnd.draft.api import _coverage

        self.assertEqual(_coverage("Every case is labelled"), "complete")
        for answer in ("Not all cases are labelled", "we haven't found all", "A sample of the cases we found", "", None, "all of them, I think"):
            with self.subTest(answer=answer):
                self.assertEqual(_coverage(answer), "partial")

    def test_the_rare_rule_skips_empty_folds_and_keeps_a_best_that_wins_every_fold(self):
        from dclab_rnd.studio.engine import _rare_choice

        def row(model, folds):
            return {"model": model, "fold_metrics": [{"average_precision": v} for v in folds],
                    "metrics": {"average_precision": {"mean": float(np.mean([v for v in folds if v is not None])), "std": 0.1}}}
        empty = [row("xgboost", [0.80, 0.82, None]), row("percentile_rules", [0.05, 0.04, None])]
        self.assertEqual(_rare_choice(empty, "average_precision", True)[0]["model"], "xgboost")
        sweep = [row("extra_trees", [0.60, 0.70, 0.51]), row("percentile_rules", [0.30, 0.50, 0.20])]
        self.assertEqual(_rare_choice(sweep, "average_precision", True)[0]["model"], "extra_trees")  # wins every fold, and far ahead

    def test_the_plan_does_not_ask_for_a_split_that_cannot_test(self):
        from dclab_core.analyse import analyse
        from dclab_core.plan import plan

        rng = np.random.default_rng(4)
        n = 2000
        day = np.sort(rng.integers(0, 300, n))
        frame = pd.DataFrame({"created": pd.Timestamp("2025-01-01") + pd.to_timedelta(day, unit="D"),
                              "customer_id": rng.integers(0, 50, n), "x": rng.normal(size=n)})
        level = rng.random(50) < 0.3
        frame["target"] = ((rng.random(n) < 0.1 + 0.5 * day / 300) | level[frame["customer_id"]]).astype(int)  # drift and recurring customers
        findings = analyse({"t": frame}, target="target", task="binary")
        entity = next(f for f in findings.findings if f.id == "F-entity-customer_id")
        self.assertGreater(entity.evidence["later_rows_seen_before"], 0.9)
        split = plan(None, findings).decision("split.strategy")
        self.assertEqual(split.choice["kind"], "group")
        self.assertIn("almost nothing to test", split.reason)


class LabelDetectionTest(unittest.TestCase):
    @unittest.skipIf(STUDIO_MISSING, STUDIO_MISSING)
    def test_a_sample_of_the_same_table_is_not_a_label_file(self):
        from dclab_rnd.draft import pipeline
        from dclab_rnd.draft.store import DraftStore

        store = DraftStore(Path(tempfile.mkdtemp()))
        draft = store.create("Predict which customers churn")
        table, _ = files(n=1200)
        table = table.assign(churned=(np.random.default_rng(0).random(len(table)) < 0.2).astype(int))
        for name, frame in (("sample.csv", table.sample(300, random_state=0)), ("full.csv", table)):
            frame.to_csv(store.data_dir(draft["id"]) / name, index=False)
            asset = pipeline.new_asset(store, draft["id"], "upload", name, name)
            pipeline.run(store, draft["id"], asset["id"])
        kinds = {a["name"]: a.get("kind") for a in store.get(draft["id"])["assets"]}
        self.assertNotEqual(kinds["sample.csv"], "labels")
