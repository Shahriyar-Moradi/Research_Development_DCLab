"""Tests for autonomous ML experimentation operators (SCI-03)."""

from __future__ import annotations

import unittest

from dclab_rnd.studio.experiment_operators import (
    AutonomousExperimentLoop,
    DebugOperator,
    DraftOperator,
    EvaluateOperator,
    ExperimentContext,
    ExperimentResult,
    ImproveOperator,
    ProposedExperiment,
)


class TestExperimentOperators(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ExperimentContext(
            problem_kind="binary_classification",
            target="churn",
            metric="roc_auc",
            higher_is_better=True,
            current_best_score=0.75,
            current_recipe="tabular",
            current_family="lightgbm",
            current_params={},
            feature_names=["age", "tenure", "monthly_fee", "leaked_cancellation_date"],
            leakage_features=["leaked_cancellation_date"],
            rows=1000,
            columns=4,
        )

    def test_draft_operator_excludes_leakage(self) -> None:
        draft_op = DraftOperator()
        proposal = draft_op.propose(self.context)
        self.assertEqual(proposal.operator, "draft")
        self.assertIn(proposal.recipe, ["tabular", "interactions", "binned"])
        if proposal.included_features is not None:
            self.assertNotIn("leaked_cancellation_date", proposal.included_features)

    def test_debug_operator_diagnoses_errors(self) -> None:
        debug_op = DebugOperator()
        failed_result = ExperimentResult(
            proposal=ProposedExperiment(
                operator="draft",
                hypothesis="Test XGBoost",
                recipe="tabular",
                family="xgboost",
                params={"max_iter": 50},
            ),
            status="failed",
            baseline_score=0.75,
            error_message="LinAlgError: Matrix is singular",
        )
        debug_prop = debug_op.diagnose_and_fix(failed_result, self.context)
        self.assertEqual(debug_prop.operator, "debug")
        self.assertIn("alpha", debug_prop.params)
        self.assertIn("shrinkage", debug_prop.hypothesis)

    def test_debug_operator_linear_model(self) -> None:
        debug_op = DebugOperator()
        failed_result = ExperimentResult(
            proposal=ProposedExperiment(
                operator="draft",
                hypothesis="Test Ridge",
                recipe="tabular",
                family="ridge",
                params={},
            ),
            status="failed",
            baseline_score=0.75,
            error_message="Unknown generic error",
        )
        debug_prop = debug_op.diagnose_and_fix(failed_result, self.context)
        self.assertEqual(debug_prop.operator, "debug")
        self.assertIn("alpha", debug_prop.params)
        self.assertNotIn("max_depth", debug_prop.params)

    def test_improve_operator_refinement(self) -> None:
        improve_op = ImproveOperator()
        accepted_result = ExperimentResult(
            proposal=ProposedExperiment(
                operator="draft",
                hypothesis="Test LightGBM",
                recipe="interactions",
                family="lightgbm",
                params={},
            ),
            status="accepted",
            score=0.78,
            baseline_score=0.75,
            lift=0.03,
        )
        imp_prop = improve_op.propose(accepted_result, self.context)
        self.assertEqual(imp_prop.operator, "improve")
        self.assertTrue(len(imp_prop.params) > 0)

    def test_evaluate_operator_rules(self) -> None:
        eval_op = EvaluateOperator(min_gain=0.005, timeout_seconds=1.0)

        # Rule C1: Leakage check
        bad_prop = ProposedExperiment(
            operator="draft",
            hypothesis="Use all features",
            recipe="tabular",
            family="lightgbm",
            included_features=["age", "leaked_cancellation_date"],
        )
        res = eval_op.evaluate(bad_prop, lambda p: (0.95, 0.01), self.context)
        self.assertEqual(res.status, "rejected")
        self.assertFalse(res.safety_verdict)
        self.assertIn("Rule C1 violation", res.reason)

        # Rule C5: Timeout check
        slow_prop = ProposedExperiment(
            operator="draft",
            hypothesis="Slow neural net",
            recipe="tabular",
            family="lightgbm",
            included_features=["age", "tenure", "monthly_fee"],
        )

        def slow_runner(p):
            import time
            time.sleep(1.1)
            return (0.85, 1.1)

        res_slow = eval_op.evaluate(slow_prop, slow_runner, self.context)
        self.assertEqual(res_slow.status, "rejected")
        self.assertFalse(res_slow.safety_verdict)
        self.assertIn("Rule C5 violation", res_slow.reason)

        # Rule C3: Gate check for improvement
        good_prop = ProposedExperiment(
            operator="draft",
            hypothesis="Better params",
            recipe="tabular",
            family="lightgbm",
            included_features=["age", "tenure", "monthly_fee"],
        )
        res_good = eval_op.evaluate(good_prop, lambda p: (0.79, 0.02), self.context)
        self.assertEqual(res_good.status, "accepted")
        self.assertTrue(res_good.safety_verdict)
        self.assertAlmostEqual(res_good.lift, 0.04)

    def test_autonomous_experiment_loop_execution(self) -> None:
        loop = AutonomousExperimentLoop(max_iterations=3, max_total_seconds=5.0, min_gain=0.005)

        # Mock runner that gives varying scores
        def mock_runner(p: ProposedExperiment) -> tuple[float, float]:
            if p.operator == "debug":
                return (0.76, 0.01)
            if p.family == "xgboost":
                raise ValueError("XGBoost missing value NaN error")
            if p.operator == "improve":
                return (0.81, 0.01)
            return (0.78, 0.01)

        final_ctx, history = loop.run(self.context, mock_runner)
        self.assertTrue(len(history) >= 2)
        self.assertGreater(final_ctx.current_best_score, self.context.current_best_score)


if __name__ == "__main__":
    unittest.main()
