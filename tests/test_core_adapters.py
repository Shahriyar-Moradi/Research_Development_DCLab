"""Package 17.8: dclab_core's outputs mapped onto DCLab main product's contracts (dclab_core.adapters.main_product).

The field lists were read from the main product's code at one commit and recorded in
dclab_core/adapters/main_product_fields.json. These tests hold the mapping to them: every mapped contract has exactly
the recorded fields, in plain JSON; a field the core cannot fill is None with a note, never a guess; values use the
main product's vocabularies (metric names, split strategies, the family portfolio, decision points); nothing from the
holdout crosses; the replicas' briefs and plans map without error and reach the expected proposals; and the adapter
loads nothing outside dclab_core's four dependencies.
"""

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import pydantic  # noqa: F401
except ImportError as error:
    PYDANTIC_MISSING = str(error)
else:
    PYDANTIC_MISSING = ""

RECORD = ROOT / "dclab_core" / "adapters" / "main_product_fields.json"
PLAN_KEYS = ("task.framing", "labels.coverage", "labels.join", "split.strategy", "metric.primary", "metric.secondary",
             "baseline", "features.forbidden", "features.to_confirm", "features.identifiers", "features.direction",
             "data.cleaning", "model.families", "threshold.policy")


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False)) == value


class RecordTest(unittest.TestCase):
    """What was read from the main product, and from where."""

    def setUp(self):
        self.record = json.loads(RECORD.read_text(encoding="utf-8"))

    def test_the_commit_and_the_files_read_are_recorded(self):
        self.assertEqual(self.record["repository"], "Shahriyar-Moradi/DCLab")
        self.assertRegex(self.record["commit"], r"^[0-9a-f]{40}$")
        self.assertGreaterEqual(len(self.record["files_read"]), 10)
        for name, contract in self.record["contracts"].items():
            with self.subTest(contract=name):
                self.assertIn(contract["path"], self.record["files_read"])
                self.assertTrue(contract["fields"])

    def test_every_contract_named_in_the_package_is_recorded(self):
        self.assertTrue({"ProblemSpec", "ProblemProfile", "MetricPlan", "HoldoutPlan", "ValidationPlan", "Objective",
                         "ExperimentPlan", "PlanSplit", "DecisionThreshold", "ExperimentReview",
                         "ExperimentChangeSet"} <= set(self.record["contracts"]))

    def test_the_plan_fields_name_registry_points(self):
        points = self.record["decision_points"]["points"]
        self.assertTrue(set(self.record["plan_fields"]["fields"]) <= set(points))
        recorded = set(self.record["contracts"]["ExperimentPlan"]["fields"])
        for fields in self.record["plan_fields"]["fields"].values():
            self.assertTrue(set(fields) <= recorded)

    @unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
    def test_the_adapter_maps_every_decision_key_onto_a_recorded_point_or_says_why_not(self):
        from dclab_core.adapters import main_product as mp

        points = self.record["decision_points"]["points"]
        self.assertEqual(sorted(mp.DECISIONS), sorted(PLAN_KEYS))
        for key, (point, _reaches, how) in mp.DECISIONS.items():
            with self.subTest(decision=key):
                self.assertTrue(point is None or point in points)
                self.assertGreater(len(how), 20)
        plan_doc = sys.modules["dclab_core.plan"].__doc__
        for key in PLAN_KEYS:
            self.assertIn(f"``{key}``", plan_doc)  # the plan still documents every key mapped here


def _check_output(case: unittest.TestCase, out: dict) -> None:
    from dclab_core.adapters import main_product as mp

    name = out["contract"]
    case.assertEqual(list(out["fields"]), mp.fields_of(name), f"{name}: the fields are not the recorded ones")
    case.assertTrue(_plain(out), f"{name} is not plain JSON")
    case.assertIn(out["status"], ("proposal", "evidence"))
    case.assertEqual(out["main_product"]["commit"], mp.recorded()["commit"])
    case.assertTrue(set(out["notes"]) <= set(out["fields"]), f"{name}: a note for a field that does not exist")
    for field, value in out["fields"].items():
        if value is None:
            case.assertTrue(out["notes"].get(field), f"{name}.{field} is None without a note saying why")


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class ReplicaMappingTest(unittest.TestCase):
    """The three replicas' briefs, findings and plans, mapped."""

    @classmethod
    def setUpClass(cls):
        from test_core_plan import _cases

        from dclab_core.adapters import main_product as mp
        from dclab_core.plan import plan

        cls.mp = mp
        cls.cases = _cases()
        cls.plans = {name: plan(brief, findings) for name, (brief, findings) in cls.cases.items()}
        cls.out = {name: mp.proposals(cls.cases[name][0], cls.cases[name][1], cls.plans[name]) for name in cls.cases}
        cls.record = mp.recorded()

    def fields(self, case, contract):
        return self.out[case][contract]["fields"]

    def test_every_output_has_exactly_the_recorded_fields_in_plain_json_and_explains_every_gap(self):
        for case, outputs in self.out.items():
            for name, out in outputs.items():
                if name == "decision_points":
                    continue
                with self.subTest(case=case, contract=name):
                    _check_output(self, out)

    def test_nested_shapes_are_the_recorded_ones(self):
        nested = self.record["contracts"]["ProblemProfile"]["nested"]
        for case in self.out:
            profile = self.fields(case, "ProblemProfile")
            for item in profile["repeated_entity_candidates"]:
                self.assertEqual(list(item), nested["repeated_entity_candidates[]"])
            for item in profile["time_candidates"]:
                self.assertEqual(list(item), nested["time_candidates[]"])
            if profile["regression_target"] is not None:
                self.assertEqual(list(profile["regression_target"]), nested["regression_target"])
            split = self.fields(case, "ExperimentPlan")["split"]
            if split is not None:
                self.assertEqual(list(split), list(self.record["contracts"]["PlanSplit"]["fields"]))

    def test_values_use_the_main_products_vocabularies(self):
        vocab = self.record["vocabularies"]
        for case, outputs in self.out.items():
            task = self.plans[case].task
            with self.subTest(case=case):
                spec = outputs["ProblemSpec"]["fields"]
                self.assertIn(spec["task_type"], self.record["contracts"]["ProblemSpec"]["task_types"])
                self.assertIn(spec["primary_metric"], vocab["primary_metrics"][task])
                metric = outputs["MetricPlan"]["fields"]
                self.assertIn(metric["primary_metric"], vocab["primary_metrics"][task])
                self.assertTrue(set(metric["secondary_metrics"]) <= set(vocab["secondary_metrics"][task]) | set(vocab["primary_metrics"][task]))
                self.assertIn(outputs["HoldoutPlan"]["fields"]["strategy"], vocab["holdout_strategies"])
                self.assertIn(outputs["ValidationPlan"]["fields"]["strategy"], vocab["validation_strategies"])
                plan = outputs["ExperimentPlan"]["fields"]
                self.assertRegex(plan["primary_metric"], vocab["code_pattern"])
                self.assertIn(plan["split"]["strategy"], vocab["plan_split_strategies"])
                group = "classification" if task in ("binary", "multiclass") else "regression"
                self.assertTrue(set(plan["families"]) <= set(vocab["portfolio_families"][group]))
                self.assertFalse(set(plan["families"]) & set(vocab["dummy_families"]))
                self.assertEqual(outputs["ProblemProfile"]["fields"]["version"], self.record["contracts"]["ProblemProfile"]["version"])

    def test_the_fraud_proposals(self):
        out = self.out["fraud"]
        self.assertEqual(self.fields("fraud", "HoldoutPlan")["strategy"], "group_disjoint")
        self.assertEqual(self.fields("fraud", "HoldoutPlan")["group_column"], "biker_id")
        self.assertEqual(self.fields("fraud", "ValidationPlan")["strategy"], "StratifiedGroupKFold")
        self.assertEqual(self.fields("fraud", "MetricPlan")["primary_metric"], "pr_auc")
        self.assertEqual(self.fields("fraud", "MetricPlan")["secondary_metrics"], ["roc_auc", "brier_score"])
        self.assertIn("recall_at_budget", out["MetricPlan"]["notes"]["secondary_metrics"])
        self.assertIn("pu_bagging", out["ExperimentPlan"]["notes"]["families"])  # a family the main product lacks: said, not dropped silently
        spec = self.fields("fraud", "ProblemSpec")
        self.assertIsNone(spec["target_column"])  # the label lives in known_frauds.csv
        self.assertIn("known_frauds", out["ProblemSpec"]["notes"]["target_column"])
        self.assertEqual(self.fields("fraud", "ExperimentPlan")["target_column"], "is_fraud")
        self.assertEqual(self.fields("fraud", "ProblemProfile")["identifier_columns"], ["order_id", "terminal_ref"])
        self.assertAlmostEqual(self.fields("fraud", "ProblemProfile")["minority_class_fraction"], 0.00677, places=4)
        coverage = next(d for d in out["decision_points"] if d["decision"] == "labels.coverage")
        self.assertIsNone(coverage["decision_point"])
        self.assertIn("partial labels", coverage["note"])

    def test_the_tickets_and_delivery_proposals(self):
        self.assertEqual(self.fields("tickets", "HoldoutPlan")["strategy"], "temporal_future")
        self.assertEqual(self.fields("tickets", "ExperimentPlan")["split"]["time_column"], "created_at")
        self.assertEqual(self.fields("tickets", "ValidationPlan")["strategy"], "TimeSeriesSplit")
        self.assertEqual(self.fields("tickets", "MetricPlan")["primary_metric"], "macro_f1")
        self.assertIsNone(self.fields("tickets", "Objective")["cost_false_positive"])  # costs are binary-only there
        self.assertIn("binary", self.out["tickets"]["Objective"]["notes"]["cost_false_positive"])
        self.assertEqual(self.fields("delivery", "HoldoutPlan")["group_column"], "restaurant_id")
        self.assertEqual(self.fields("delivery", "ValidationPlan")["strategy"], "GroupKFold")
        self.assertEqual(self.fields("delivery", "MetricPlan")["primary_metric"], "mae")
        self.assertIn("quantile_gradient_boosting", self.out["delivery"]["ExperimentPlan"]["notes"]["families"])
        self.assertEqual(self.fields("delivery", "ProblemProfile")["regression_target"]["max"], 60.0)
        forbidden = next(d for d in self.out["delivery"]["decision_points"] if d["decision"] == "features.forbidden")
        self.assertEqual(forbidden["decision_point"], "feature.leakage_suspect")
        self.assertEqual(sorted(forbidden["proposal"]), ["courier_rating", "delivered_at"])

    def test_every_decision_goes_through_a_point_or_says_why_it_cannot(self):
        points = self.record["decision_points"]["points"]
        for case, outputs in self.out.items():
            keys = [d.key for d in self.plans[case].decisions]
            mapped = outputs["decision_points"]
            self.assertEqual([d["decision"] for d in mapped], keys)
            for d in mapped:
                with self.subTest(case=case, decision=d["decision"]):
                    self.assertEqual(d["status"], "proposal")
                    self.assertTrue(d["decision_point"] is None or d["decision_point"] in points)
                    if d["decision_point"] is not None:
                        self.assertEqual(d["cap"], points[d["decision_point"]]["cap"])
                    self.assertTrue(d["note"])
                    self.assertTrue(_plain(d))

    def test_stored_json_maps_the_same_as_the_objects(self):
        brief, findings = self.cases["delivery"]
        plan = self.plans["delivery"]
        again = self.mp.proposals(brief.model_dump_json(), findings.model_dump(mode="json"), plan.model_dump(mode="json"))
        self.assertEqual(again, self.out["delivery"])


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class GapTest(unittest.TestCase):
    """What the core cannot fill is None with a note; nothing is invented."""

    def test_an_empty_brief_gives_an_honest_spec(self):
        from dclab_core.adapters import main_product as mp
        from dclab_core.contracts import ProblemBrief

        out = mp.problem_spec(ProblemBrief())
        _check_output(self, out)
        for field in ("task_type", "business_objective", "target_column", "prediction_unit", "prediction_time_column",
                      "prediction_horizon", "primary_metric"):
            self.assertIsNone(out["fields"][field])
        self.assertEqual(out["fields"]["success_criteria"], {})

    def test_metrics_the_main_product_cannot_score_are_said_not_renamed(self):
        from dclab_core.adapters.main_product import metric_name

        self.assertEqual(metric_name("average_precision", "binary"), ("pr_auc", None))
        self.assertEqual(metric_name("PR-AUC", "binary"), ("pr_auc", None))
        self.assertEqual(metric_name("f1", "multiclass"), ("macro_f1", None))
        self.assertEqual(metric_name("brier", "binary", "secondary"), ("brier_score", None))
        for name, task in (("recall_at_budget", "binary"), ("share_within_tolerance", "regression"),
                           ("top_k_accuracy", "multiclass"), ("macro_f1", "binary"), ("mae", "unknown")):
            with self.subTest(metric=name):
                value, why = metric_name(name, task)
                self.assertIsNone(value)
                self.assertTrue(why)

    def test_stated_costs_of_a_binary_problem_become_its_cost_matrix(self):
        from dclab_core.adapters import main_product as mp
        from dclab_core.contracts import Constraint, Decision, ProblemBrief, SolutionPlan, TaskFact

        text = "A false alarm costs 5 euros and a missed fraud costs 100 euros."
        brief = ProblemBrief(task=TaskFact(value="binary", status="stated", quote="detect fraud", source="text"),
                             constraints=[Constraint(kind="budget", text=text, quote=text)])
        plan = SolutionPlan(task="binary", decisions=[
            Decision(key="metric.primary", choice="average_precision", reason="Positives are rare: rank them."),
            Decision(key="threshold.policy", choice={"kind": "expected_cost", "costs": text},
                     reason="The brief gives the cost of errors.")])
        spec, goal = mp.problem_spec(brief, plan), mp.objective(plan, brief)
        _check_output(self, spec)
        _check_output(self, goal)
        self.assertEqual(spec["fields"]["constraints"]["cost_matrix"], {"false_positive": 5.0, "false_negative": 100.0})
        self.assertEqual(spec["fields"]["constraints"]["primary_metric_reason"], "Positives are rare: rank them.")
        self.assertEqual((goal["fields"]["cost_false_positive"], goal["fields"]["cost_false_negative"]), (5.0, 100.0))
        policy = next(d for d in mp.decision_points(plan) if d["decision"] == "threshold.policy")
        self.assertEqual(policy["proposal"], {"cost_matrix": {"false_positive": 5.0, "false_negative": 100.0}})

    def test_a_split_the_main_product_cannot_hold_is_unsupported_not_forced(self):
        from dclab_core.adapters import main_product as mp
        from dclab_core.contracts import Decision, SolutionPlan

        plan = SolutionPlan(task="binary", decisions=[Decision(
            key="split.strategy", choice={"kind": "time_and_group", "time_column": "day", "group_column": "biker_id"},
            reason="Rows repeat for bikers and the target drifts.")])
        holdout, folds, run = mp.holdout_plan(plan), mp.validation_plan(plan), mp.experiment_plan(plan)
        for out in (holdout, folds, run):
            _check_output(self, out)
        self.assertEqual(holdout["fields"]["strategy"], "unsupported")
        self.assertTrue(holdout["fields"]["stratified"])  # as the main product's plan_holdout answers for a classifier
        self.assertEqual(holdout["fields"]["stratified"], folds["fields"]["stratified"])
        self.assertEqual((holdout["fields"]["group_column"], holdout["fields"]["time_column"]), ("biker_id", "day"))
        self.assertEqual(folds["fields"]["strategy"], "unsupported")
        self.assertIsNone(run["fields"]["split"])
        self.assertIn("person", run["notes"]["split"])

    def test_target_statistics_of_every_row_never_cross(self):
        from dclab_core.adapters import main_product as mp
        from dclab_core.analyse import analyse
        from dclab_core.replicas import delivery_time

        orders = delivery_time.generate(seed=0).orders
        out = mp.problem_profile(analyse({"orders.csv": orders}, target="actual_minutes"))  # no training partition
        _check_output(self, out)
        for field in ("minority_class_fraction", "imbalance_ratio", "regression_target"):
            self.assertIsNone(out["fields"][field])
            self.assertIn("every row", out["notes"][field])
        self.assertIn("all_rows_target_statistics", out["excluded"])
        locked = mp.problem_profile(analyse({"orders.csv": orders}, target="actual_minutes",
                                            train_index=orders.index[:6000]))
        self.assertIsNotNone(locked["fields"]["regression_target"])
        self.assertEqual(locked["excluded"], {})

    def test_before_the_holdout_lock_the_guides_call_reads_no_target(self):
        from test_core_understand import DELIVERY, DELIVERY_FILES

        from dclab_core.analyse import analyse
        from dclab_core.replicas import delivery_time
        from dclab_core.understand import understand

        orders = delivery_time.generate(seed=0).orders
        before = analyse({"orders.csv": orders}, target=None, brief=None)  # what CORE_INTEGRATION.md says to call
        self.assertIsNone(before.target)
        self.assertFalse([f.id for f in before.findings if f.topic in ("target", "leakage", "scope")])
        with_brief = analyse({"orders.csv": orders}, target=None, brief=understand(DELIVERY, DELIVERY_FILES))
        self.assertEqual(with_brief.target, "actual_minutes")  # why the guide says brief=None before the lock

    def test_an_inferred_target_is_flagged_for_a_person(self):
        from dclab_core.adapters import main_product as mp
        from dclab_core.contracts import ProblemBrief, SolutionPlan, TextFact

        brief = ProblemBrief(target_column=TextFact(value="churned", status="inferred", reason="the only yes/no column"))
        out = mp.experiment_plan(SolutionPlan(task="binary"), brief)
        _check_output(self, out)
        self.assertEqual(out["fields"]["target_column"], "churned")
        self.assertIn("inferred", out["notes"]["target_column"])

    def test_constraints_over_the_main_products_size_limit_are_left_to_a_person(self):
        from dclab_core.adapters import main_product as mp
        from dclab_core.contracts import Constraint, ProblemBrief, TextFact

        long = "x" * 3900
        brief = ProblemBrief(constraints=[Constraint(kind="other", text=long, quote=long)] * 5,
                             success_criteria=[TextFact(value=long, status="stated", quote=long, source="text")] * 5)
        out = mp.problem_spec(brief)
        _check_output(self, out)
        limit = mp.recorded()["contracts"]["ProblemSpec"]["json_max_bytes"]
        for field in ("constraints", "success_criteria"):
            self.assertLessEqual(len(json.dumps(out["fields"][field]).encode("utf-8")), limit)
            self.assertIn(str(limit), out["notes"][field])

    def test_a_plan_without_decisions_maps_to_explained_gaps(self):
        from dclab_core.adapters import main_product as mp
        from dclab_core.contracts import SolutionPlan

        for out in mp.proposals(plan=SolutionPlan()).values():
            if isinstance(out, dict):
                _check_output(self, out)


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class AfterTheRunTest(unittest.TestCase):
    """assess() and review(): only out-of-fold, training-row figures cross."""

    def test_a_binary_threshold_crosses_without_its_holdout_figures(self):
        import numpy as np

        from dclab_core.adapters import main_product as mp
        from dclab_core.assess import assess

        rng = np.random.default_rng(0)
        y = (rng.random(4000) < 0.05).astype(int)
        p = np.clip(y * 0.4 + rng.random(4000) * 0.6, 0, 1)
        y_test, p_test = y[:1000][::-1], p[:1000]  # a holdout unlike the training rows
        for kwargs, source, goal in (({"costs": {"false_positive": 1, "false_negative": 30}}, "cost_matrix", "expected_cost"),
                                     ({"budget": 0.02}, None, None), ({}, None, "f1")):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                result = assess("binary", y, p, y_test, p_test, partial_labels=True, **kwargs)
                out = mp.decision_threshold(result)
                _check_output(self, out)
                fields = out["fields"]
                self.assertEqual((fields["source"], fields["goal_metric"]), (source, goal))
                self.assertEqual(fields["selected_on"], "out_of_fold_cv")
                self.assertEqual(fields["value"], result["threshold"]["value"])
                at = fields["oof_at_threshold"]
                self.assertEqual(list(at), mp.recorded()["contracts"]["DecisionThreshold"]["nested"]["oof_at_threshold"])
                self.assertEqual(at["tp"], result["threshold"]["training"]["true_positives"])
                self.assertNotIn("holdout", json.dumps(fields))
                self.assertIn("holdout", out["excluded"])
                self.assertIn("calibration", out["excluded"]["report_only"])
                self.assertIn("lower bound", out["notes"]["value"])  # partial labels: the precision is a lower bound

    def test_the_threshold_record_does_not_move_with_the_holdout(self):
        import numpy as np

        from dclab_core.adapters import main_product as mp
        from dclab_core.assess import assess

        rng = np.random.default_rng(3)
        y = (rng.random(3000) < 0.08).astype(int)
        p = np.clip(y * 0.3 + rng.random(3000) * 0.7, 0, 1)
        for kwargs in ({"budget": 0.05}, {"costs": {"false_positive": 1, "false_negative": 20}}, {}):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                one = assess("binary", y, p, y[:800], p[:800], **kwargs)
                other = assess("binary", y, p, 1 - y[:800], p[:800][::-1], **kwargs)  # a different holdout
                self.assertNotEqual(one["threshold"]["holdout"], other["threshold"]["holdout"])
                a, b = mp.decision_threshold(one), mp.decision_threshold(other)
                self.assertEqual((a["fields"], a["notes"]), (b["fields"], b["notes"]))
                training = one["threshold"]["training"]
                at = a["fields"]["oof_at_threshold"]
                self.assertEqual((at["tp"], at["fp"], at["fn"], at["precision"], at["recall"]),
                                 (training["true_positives"], training["false_positives"], training["false_negatives"],
                                  training["precision"], training["recall"]))

    def test_other_tasks_have_no_threshold_and_say_so(self):
        import numpy as np

        from dclab_core.adapters import main_product as mp
        from dclab_core.assess import assess

        rng = np.random.default_rng(1)
        y = rng.normal(30, 5, 500)
        out = mp.decision_threshold(assess("regression", y, y + rng.normal(0, 1, 500), y, y))
        _check_output(self, out)
        self.assertIsNone(out["fields"]["value"])
        self.assertIn("intervals", out["excluded"]["report_only"])

    def test_a_review_becomes_advice_and_the_changes_the_main_product_has(self):
        import numpy as np

        from dclab_core.adapters import main_product as mp
        from dclab_core.review import review

        rng = np.random.default_rng(2)
        n = 3000
        y = rng.choice(["a", "b", "c", "rare"], n, p=[0.45, 0.35, 0.18, 0.02])
        pred = np.where(y == "rare", "a", y)
        weak = review("multiclass", y, pred)
        advice, changes = mp.experiment_review(weak), mp.experiment_changes(weak)
        _check_output(self, advice)
        _check_output(self, changes)
        vocab = mp.recorded()["vocabularies"]
        self.assertTrue(advice["fields"]["findings"])
        for item in advice["fields"]["findings"]:
            self.assertEqual(list(item), mp.recorded()["contracts"]["ExperimentReview"]["nested"]["findings[]"])
            self.assertRegex(item["check"], vocab["code_pattern"])
            self.assertIn(item["status"], vocab["finding_statuses"])
        self.assertIsNone(advice["fields"]["verdict"])
        self.assertEqual(changes["fields"]["changes"], [{"kind": "class_weighting", "mode": "balanced", "weights": None}])
        self.assertEqual(changes["decision_point"], "improve.next_action")
        self.assertIn(changes["fields"]["changes"][0]["kind"], vocab["change_kinds"])

    def test_experiments_without_a_change_kind_are_named_not_dropped(self):
        import numpy as np

        from dclab_core.adapters import main_product as mp
        from dclab_core.review import review

        rng = np.random.default_rng(0)
        base = rng.gamma(4, 4, 4000)
        y = np.minimum(base + np.where(rng.random(4000) < 0.25, rng.exponential(20, 4000), 0), 60)
        changes = mp.experiment_changes(review("regression", y, base + 4, metric="mae", cap=60.0))
        _check_output(self, changes)
        self.assertIsNone(changes["fields"]["changes"])
        self.assertIn("loss.median", changes["notes"]["changes"])


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class BoundaryTest(unittest.TestCase):
    def test_the_adapter_loads_nothing_outside_the_cores_dependencies(self):
        from test_core_contracts import DEPENDENCY_TREE

        code = ("import sys, json; before = set(sys.modules); import dclab_core.adapters.main_product as mp; "
                "mp.recorded(); std = set(sys.stdlib_module_names); "
                "print(json.dumps(sorted({m.split('.')[0] for m in set(sys.modules) - before} - std)))")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        loaded = {m for m in json.loads(out.stdout.strip()) if not m.startswith(("_sysconfigdata", "_cython_", "cython_runtime"))}
        self.assertNotIn("dclab_rnd", loaded)
        self.assertNotIn("app", loaded)  # the main product's package
        self.assertEqual(sorted(loaded - DEPENDENCY_TREE), [])

    def test_the_adapter_names_no_main_product_module(self):
        source = (ROOT / "dclab_core" / "adapters" / "main_product.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"^\s*(from|import)\s+app\b", source, re.M))

    def test_the_field_lists_ship_in_the_wheel(self):
        text = (ROOT / "dclab_core" / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"dclab_core.adapters" = ["main_product_fields.json"]', text)


if __name__ == "__main__":
    unittest.main()
