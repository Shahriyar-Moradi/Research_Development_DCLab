"""Contrastive leakage benchmarking contract: keeps leakage data and models to compare.

Per the DCLab Scientific ML Reliability standard (SCI-02):
Rather than silently discarding leakage features upon detection, the system retains a contrastive shadow
benchmark model fitted with the suspect/leakage features alongside the honest, leakage-safe model.
This produces an empirical contrast (CV apparent lift vs. holdout/temporal generalization drop),
verifying the hazard, explaining to users why columns were forbidden, and generating preference pairs.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from dclab_core.contracts import Contract, Versioned, _version


class LeakageContrast(Versioned):
    """Contrastive comparison between a clean, leakage-safe pipeline and a shadow model with leakage."""

    schema_version: str = _version("leakage_contrast")
    project: str
    target: str
    leakage_features: list[str]
    decision_time_contract: str = ""
    metric: str
    higher_is_better: bool = True
    clean_cv_metric: float
    leakage_cv_metric: float
    cv_apparent_lift: float
    clean_holdout_metric: Optional[float] = None
    leakage_holdout_metric: Optional[float] = None
    holdout_apparent_lift: Optional[float] = None
    generalization_drop: Optional[float] = None
    per_feature_cv_lift: dict[str, float] = Field(default_factory=dict)
    clean_metrics: dict[str, float] = Field(default_factory=dict)
    leakage_metrics: dict[str, float] = Field(default_factory=dict)
    lesson: str

    def is_generalization_hazard(self, threshold: float = 0.03) -> bool:
        """Whether the leakage features caused an artificial score inflation that dropped on holdout."""
        if self.cv_apparent_lift <= threshold:
            return False
        if self.generalization_drop is not None:
            return self.generalization_drop > (threshold / 2.0)
        return True


def compute_oriented_gain(leakage_val: float, clean_val: float, higher_is_better: bool) -> float:
    """Compute gain of leakage over clean, respecting metric orientation."""
    delta = leakage_val - clean_val
    return float(delta if higher_is_better else -delta)


def build_leakage_contrast(
    project: str,
    target: str,
    leakage_features: list[str],
    metric: str,
    higher_is_better: bool,
    clean_cv_metric: float,
    leakage_cv_metric: float,
    clean_holdout_metric: Optional[float] = None,
    leakage_holdout_metric: Optional[float] = None,
    per_feature_cv_lift: Optional[dict[str, float]] = None,
    clean_metrics: Optional[dict[str, float]] = None,
    leakage_metrics: Optional[dict[str, float]] = None,
    decision_time_contract: str = "",
    custom_lesson: Optional[str] = None,
) -> LeakageContrast:
    """Construct a validated LeakageContrast contract with computed lifts and scientific lessons."""
    cv_lift = compute_oriented_gain(leakage_cv_metric, clean_cv_metric, higher_is_better)
    
    holdout_lift: Optional[float] = None
    gen_drop: Optional[float] = None
    if clean_holdout_metric is not None and leakage_holdout_metric is not None:
        holdout_lift = compute_oriented_gain(leakage_holdout_metric, clean_holdout_metric, higher_is_better)
        gen_drop = float(cv_lift - holdout_lift)

    if custom_lesson:
        lesson = custom_lesson
    else:
        feat_str = ", ".join(f"`{f}`" for f in sorted(leakage_features))
        sign = "+" if cv_lift >= 0 else ""
        lesson_parts = [
            f"Including leakage features ({feat_str}) produced an apparent CV {metric} lift of {sign}{cv_lift:.4f}."
        ]
        if gen_drop is not None:
            lesson_parts.append(
                f"On honest holdout evaluation, generalization dropped by {gen_drop:.4f} "
                f"(holdout lift was {holdout_lift:+.4f}), confirming the CV gain was non-generalizable leakage."
            )
        else:
            lesson_parts.append("The clean pipeline was preserved for deployment and holdout confirmation.")
        lesson = " ".join(lesson_parts)

    return LeakageContrast(
        project=project,
        target=target,
        leakage_features=sorted(leakage_features),
        decision_time_contract=decision_time_contract,
        metric=metric,
        higher_is_better=higher_is_better,
        clean_cv_metric=clean_cv_metric,
        leakage_cv_metric=leakage_cv_metric,
        cv_apparent_lift=cv_lift,
        clean_holdout_metric=clean_holdout_metric,
        leakage_holdout_metric=leakage_holdout_metric,
        holdout_apparent_lift=holdout_lift,
        generalization_drop=gen_drop,
        per_feature_cv_lift=dict(per_feature_cv_lift or {}),
        clean_metrics=dict(clean_metrics or {}),
        leakage_metrics=dict(leakage_metrics or {}),
        lesson=lesson,
    )
