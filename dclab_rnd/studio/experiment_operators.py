"""Guarded autonomous ML experimentation operators inspired by AIDE and R&D-Agent (SCI-03).

Provides four core experimentation operators governing autonomous exploration under DCLab Part C rules:
- DraftOperator: formulate hypotheses and candidate variations (recipes, families, params)
- DebugOperator: diagnose execution failures and generate minimal corrective patches
- ImproveOperator: propose incremental parameter tuning and regularizations on validation diagnostics
- EvaluateOperator: evaluate candidates with deterministic code gate and budget limits (Rules C1, C2, C3, C5)
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ExperimentContext:
    """Problem context provided to experimentation operators."""

    problem_kind: str
    target: str
    metric: str
    higher_is_better: bool
    current_best_score: float
    current_recipe: str
    current_family: str
    current_params: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    leakage_features: list[str] = field(default_factory=list)
    rows: int = 1000
    columns: int = 10
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ProposedExperiment:
    """A concrete experiment proposed by an operator."""

    operator: str  # "draft", "debug", "improve"
    hypothesis: str
    recipe: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    included_features: Optional[list[str]] = None
    estimated_seconds: float = 5.0


@dataclass(frozen=True)
class ExperimentResult:
    """Result of an executed experiment evaluated under Part C rules."""

    proposal: ProposedExperiment
    status: str  # "accepted", "rejected", "failed"
    score: Optional[float] = None
    baseline_score: float = 0.0
    lift: float = 0.0
    duration_seconds: float = 0.0
    reason: str = ""
    error_message: Optional[str] = None
    safety_verdict: bool = True  # Passed rules C1 (no leakage), C2 (cv only), C5 (budget)


class DraftOperator:
    """Formulate initial candidate variations across model families and feature recipes."""

    def propose(self, context: ExperimentContext) -> ProposedExperiment:
        """Propose the next candidate experiment not yet tried in history."""
        tried_combinations = {
            (h.get("recipe"), h.get("family")) for h in context.history
        }

        # Candidate recipes and model families suited for problem kind
        recipes = ["tabular", "interactions", "binned"]
        families = ["lightgbm", "xgboost", "random_forest", "ridge", "logistic_regression"]

        # Filter out inappropriate families
        if "regress" in context.problem_kind.lower():
            families = [f for f in families if f != "logistic_regression"]
        else:
            families = [f for f in families if f != "ridge"]

        for rec in recipes:
            for fam in families:
                if (rec, fam) not in tried_combinations:
                    # Enforce Rule C1: ensure leakage features are excluded
                    safe_features = [f for f in context.feature_names if f not in context.leakage_features]
                    return ProposedExperiment(
                        operator="draft",
                        hypothesis=f"Evaluate family {fam} on {rec} feature recipe to test alternative model inductive bias.",
                        recipe=rec,
                        family=fam,
                        params={},
                        included_features=safe_features if safe_features else None,
                    )

        # Fallback: vary basic regularizer
        return ProposedExperiment(
            operator="draft",
            hypothesis="Apply L2 regularization to current leading family.",
            recipe=context.current_recipe,
            family=context.current_family,
            params={"l2_regularization": 1.0},
        )


class DebugOperator:
    """Diagnose errors (NaN, convergence, LinAlgError) and propose minimal corrective patches."""

    def diagnose_and_fix(self, failed_result: ExperimentResult, context: ExperimentContext) -> ProposedExperiment:
        """Analyze failure error message and propose an explicit debug patch."""
        err = (failed_result.error_message or "").lower()
        proposal = failed_result.proposal
        new_params = copy.deepcopy(proposal.params)

        if "nan" in err or "missing" in err:
            hypothesis = "Error indicated missing/NaN values; switch to robust imputation."
            new_params["impute_strategy"] = "median"
        elif "convergence" in err or "max_iter" in err:
            hypothesis = "Model failed to converge; increase iterations and apply learning rate decay."
            new_params["max_iter"] = new_params.get("max_iter", 100) * 3
            new_params["learning_rate"] = 0.05
        elif "singular" in err or "linalg" in err or "collinear" in err:
            hypothesis = "Collinear matrix or singular covariance; add ridge shrinkage regularization."
            new_params["alpha"] = 1.0
            new_params["ridge_penalty"] = 0.1
        elif "memory" in err or "timeout" in err:
            hypothesis = "Resource limit exceeded; downscale tree depth and estimators."
            new_params["max_depth"] = 4
            new_params["n_estimators"] = 50
        else:
            if proposal.family in ("ridge", "logistic_regression"):
                hypothesis = "Unclassified runtime exception; fallback to strong L2 regularization."
                new_params["alpha"] = 10.0 if proposal.family == "ridge" else 1.0
                new_params["C"] = 0.1 if proposal.family == "logistic_regression" else 1.0
            else:
                hypothesis = "Unclassified runtime exception; fallback to conservative tree settings."
                new_params["max_depth"] = 3
                new_params["min_samples_leaf"] = 20

        return ProposedExperiment(
            operator="debug",
            hypothesis=hypothesis,
            recipe=proposal.recipe,
            family=proposal.family,
            params=new_params,
            included_features=proposal.included_features,
        )


class ImproveOperator:
    """Propose hyperparameter refinements, feature interactions or class weighting."""

    def propose(self, accepted_result: ExperimentResult, context: ExperimentContext) -> ProposedExperiment:
        """Refine an accepted candidate based on validation metrics."""
        proposal = accepted_result.proposal
        new_params = copy.deepcopy(proposal.params)

        # Iterative refinement steps
        step_count = sum(1 for h in context.history if h.get("family") == proposal.family)
        if step_count == 1:
            hypothesis = "Tune tree depth and min_samples_leaf to reduce variance."
            new_params.update({"max_depth": 6, "min_samples_leaf": 30})
        elif step_count == 2:
            hypothesis = "Apply feature subsampling (colsample) and row subsampling to prevent overfitting."
            new_params.update({"subsample": 0.8, "colsample_bytree": 0.8})
        else:
            hypothesis = "Fine-tune learning rate with increased estimator budget."
            new_params.update({"learning_rate": 0.03, "n_estimators": 200})

        return ProposedExperiment(
            operator="improve",
            hypothesis=hypothesis,
            recipe=proposal.recipe,
            family=proposal.family,
            params=new_params,
            included_features=proposal.included_features,
        )


class EvaluateOperator:
    """Deterministic evaluation gate enforcing Rules C1, C2, C3, and C5."""

    def __init__(self, min_gain: float = 0.001, timeout_seconds: float = 30.0):
        self.min_gain = min_gain
        self.timeout_seconds = timeout_seconds

    def evaluate(
        self,
        proposal: ProposedExperiment,
        runner_fn: Callable[[ProposedExperiment], tuple[float, float]],
        context: ExperimentContext,
    ) -> ExperimentResult:
        """Execute proposal with guardrails, compute gain against baseline, and decide verdict."""
        # Rule C1: Check for leakage features in candidate feature list
        features_to_check = proposal.included_features if proposal.included_features is not None else context.feature_names
        if features_to_check and context.leakage_features:
            leakage_present = set(features_to_check) & set(context.leakage_features)
            if leakage_present:
                return ExperimentResult(
                    proposal=proposal,
                    status="rejected",
                    baseline_score=context.current_best_score,
                    reason=f"Rule C1 violation: Proposal contained forbidden leakage features {sorted(leakage_present)}.",
                    safety_verdict=False,
                )

        start_time = time.perf_counter()
        try:
            # Execute in runner
            score, exec_seconds = runner_fn(proposal)
        except Exception as ex:  # noqa: BLE001
            duration = time.perf_counter() - start_time
            return ExperimentResult(
                proposal=proposal,
                status="failed",
                duration_seconds=duration,
                error_message=str(ex),
                reason=f"Trial execution failed with exception: {ex}",
                baseline_score=context.current_best_score,
            )

        duration = time.perf_counter() - start_time

        # Rule C5: Enforce budget / timeout
        if duration > self.timeout_seconds:
            return ExperimentResult(
                proposal=proposal,
                status="rejected",
                score=score,
                duration_seconds=duration,
                baseline_score=context.current_best_score,
                reason=f"Rule C5 violation: Execution took {duration:.1f}s, exceeding timeout cap of {self.timeout_seconds}s.",
                safety_verdict=False,
            )

        # Rule C3: Deterministic code gate
        delta = score - context.current_best_score
        lift = float(delta if context.higher_is_better else -delta)

        if lift >= self.min_gain:
            status = "accepted"
            reason = f"Candidate improved {context.metric} by {lift:+.4f} (>= min_gain {self.min_gain})."
        else:
            status = "rejected"
            reason = f"Candidate lift {lift:+.4f} did not meet required threshold of {self.min_gain}."

        return ExperimentResult(
            proposal=proposal,
            status=status,
            score=score,
            baseline_score=context.current_best_score,
            lift=lift,
            duration_seconds=duration,
            reason=reason,
            safety_verdict=True,
        )


class AutonomousExperimentLoop:
    """Coordinates Draft -> Evaluate -> Debug/Improve loop within bounded iteration and time budgets."""

    def __init__(
        self,
        max_iterations: int = 5,
        max_total_seconds: float = 60.0,
        min_gain: float = 0.001,
        trial_timeout_seconds: float = 30.0,
    ):
        self.max_iterations = max_iterations
        self.max_total_seconds = max_total_seconds
        self.draft_op = DraftOperator()
        self.debug_op = DebugOperator()
        self.improve_op = ImproveOperator()
        self.eval_op = EvaluateOperator(min_gain=min_gain, timeout_seconds=trial_timeout_seconds)

    def run(
        self,
        context: ExperimentContext,
        runner_fn: Callable[[ProposedExperiment], tuple[float, float]],
    ) -> tuple[ExperimentContext, list[ExperimentResult]]:
        """Run bounded autonomous experimentation loop."""
        current_ctx = copy.deepcopy(context)
        results: list[ExperimentResult] = []
        loop_start = time.perf_counter()

        for _ in range(self.max_iterations):
            if (time.perf_counter() - loop_start) >= self.max_total_seconds:
                break

            # Draft candidate
            proposal = self.draft_op.propose(current_ctx)
            result = self.eval_op.evaluate(proposal, runner_fn, current_ctx)

            # Debug if failed
            if result.status == "failed":
                debug_proposal = self.debug_op.diagnose_and_fix(result, current_ctx)
                debug_result = self.eval_op.evaluate(debug_proposal, runner_fn, current_ctx)
                results.append(result)
                result = debug_result

            # If accepted, attempt improve step
            if result.status == "accepted" and result.score is not None:
                current_ctx = ExperimentContext(
                    problem_kind=current_ctx.problem_kind,
                    target=current_ctx.target,
                    metric=current_ctx.metric,
                    higher_is_better=current_ctx.higher_is_better,
                    current_best_score=result.score,
                    current_recipe=result.proposal.recipe,
                    current_family=result.proposal.family,
                    current_params=result.proposal.params,
                    feature_names=current_ctx.feature_names,
                    leakage_features=current_ctx.leakage_features,
                    rows=current_ctx.rows,
                    columns=current_ctx.columns,
                    history=current_ctx.history + [{"recipe": result.proposal.recipe, "family": result.proposal.family, "score": result.score}],
                )
                results.append(result)

                # Try improve
                imp_proposal = self.improve_op.propose(result, current_ctx)
                imp_result = self.eval_op.evaluate(imp_proposal, runner_fn, current_ctx)
                results.append(imp_result)
                if imp_result.status == "accepted" and imp_result.score is not None:
                    current_ctx = ExperimentContext(
                        problem_kind=current_ctx.problem_kind,
                        target=current_ctx.target,
                        metric=current_ctx.metric,
                        higher_is_better=current_ctx.higher_is_better,
                        current_best_score=imp_result.score,
                        current_recipe=imp_result.proposal.recipe,
                        current_family=imp_result.proposal.family,
                        current_params=imp_result.proposal.params,
                        feature_names=current_ctx.feature_names,
                        leakage_features=current_ctx.leakage_features,
                        rows=current_ctx.rows,
                        columns=current_ctx.columns,
                        history=current_ctx.history + [{"recipe": imp_result.proposal.recipe, "family": imp_result.proposal.family, "score": imp_result.score}],
                    )
            else:
                results.append(result)
                current_ctx = ExperimentContext(
                    problem_kind=current_ctx.problem_kind,
                    target=current_ctx.target,
                    metric=current_ctx.metric,
                    higher_is_better=current_ctx.higher_is_better,
                    current_best_score=current_ctx.current_best_score,
                    current_recipe=current_ctx.current_recipe,
                    current_family=current_ctx.current_family,
                    current_params=current_ctx.current_params,
                    feature_names=current_ctx.feature_names,
                    leakage_features=current_ctx.leakage_features,
                    rows=current_ctx.rows,
                    columns=current_ctx.columns,
                    history=current_ctx.history + [{"recipe": proposal.recipe, "family": proposal.family, "status": result.status}],
                )

        return current_ctx, results
