"""Binary classification: yes or no for each row."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import ProblemKind, _get


class Binary(ProblemKind):
    key = "binary"
    title = "Binary classification"
    description = "A yes or no for each row: churn, default, acceptance."
    base_task = "binary"
    task_type = "binary_imbalanced"
    questions = ("What exactly is the outcome, for which rows, and within which time window?",
                 "When is the prediction made, and what is known then?")
    metrics = (("roc_auc", "ROC-AUC (ranking quality, balanced classes)"), ("average_precision", "Average precision (rare positives)"),
               ("recall_at_budget", "Recall at the review budget (the share of cases found when the team reviews its budget)"))
    baselines = ("majority_class",)  # with the random-ranking AP
    checks = ("PIT-001", "PIT-002", "PIT-004", "PIT-005", "PIT-006")
    reliability = ("calibration", "subgroup gaps", "drift by time")
    evidence_track = "tabular-classification"
    models_text = True  # text columns turn a binary kind into the text machine

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        if (pack or {}).get("metric"):
            return pack["metric"]
        return "average_precision" if positive_rate is not None and positive_rate < 0.10 else "roc_auc"

    def target(self, frame: pd.DataFrame, solution: Any) -> tuple[pd.Series, list[str] | None]:
        raw = frame[_get(solution, "target")]
        positive = _get(solution, "positive_label")
        if positive is None:
            values = sorted(raw.dropna().unique(), key=str)
            positive = str(values[-1])
        return (raw.astype(str).str.strip() == str(positive)).astype(int), None

    def positive_rate(self, y: pd.Series) -> float | None:
        return float(y.mean())

    def machine(self, solution: Any, text_columns: tuple[str, ...]) -> str:
        return "text_tabular_binary" if text_columns else self.task_type

    def spec_settings(self, solution: Any, pack: dict[str, Any] | None = None) -> dict[str, Any]:
        budget = _get(solution, "alert_budget")
        return {**({"rare_event": True} if (pack or {}).get("rare_event") else {}), **({"alert_budget": budget} if budget else {})}

    def rare_screen(self, task_type: str, positive_rate: float | None, solution: Any, settings: dict[str, Any]) -> bool:
        return task_type == "binary_imbalanced" and ((positive_rate or 1) < 0.05 or _get(solution, "labels") == "partial"
                                                     or bool(settings.get("rare_event")))

    def summarise_target(self, y: pd.Series, class_labels: list[str] | None, solution: Any) -> tuple[dict[str, Any], str]:
        positives, rate = int(y.sum()), float(y.mean())
        summary = {"positive_rate_train": rate, "positives_train": positives, "negatives_train": int(len(y) - positives),
                   "imbalance_ratio_neg_per_pos": float((len(y) - positives) / max(positives, 1)),
                   "majority_baseline_accuracy_train": float(max(rate, 1 - rate)), "positive_label": _get(solution, "positive_label")}
        return summary, f"a positive rate of {rate:.1%} ({positives} positives)"

    def target_claim(self, summary: dict[str, Any], metric_label: str, spec: Any) -> tuple[str, str, list[str]]:
        return ("risk", f"Predicting the majority class for every row already scores {summary['majority_baseline_accuracy_train']:.1%} accuracy, "
                        f"so accuracy alone cannot judge this model; the solution scores {metric_label}.", ["evidence.target_summary"])

    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import subscription_churn

        from .acceptance import Case

        replica = subscription_churn.generate(seed)
        return Case(replica.subscribers, "churned", "binary", "At renewal time, before the subscriber decides; refunds are paid only "
                    "after a cancellation.", {"positive_label": "yes"}, facts=replica.facts)

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check, kept_out

        final = run.evidence("final")
        auc, chance = run.holdout("roc_auc"), final.get("baselines", {}).get("majority_class", {}).get("roc_auc", 0.5)
        lift = run.evidence("leakage").get("apparent_lift")
        return [kept_out(run, "final_refund_amount", "paid only after a cancellation", roles=("blocked",)),
                kept_out(run, "subscriber_id", "an identifier"),
                check("the holdout is the latest signups", "signup_time" in run.evidence("data").get("holdout_split", ""),
                      run.evidence("data").get("holdout_split", "")),
                check("scored by ROC-AUC (cancellations are not rare)", run.records["final"].get("primary_metric") == "roc_auc",
                      str(run.records["final"].get("primary_metric"))),
                check("the leakage audit measures how much the refund would have inflated the score (by more than 0.1)",
                      lift is not None and lift > 0.1, f"apparent lift {lift}"),
                check("the holdout ROC-AUC is at least 0.65 and 0.1 above chance", auc is not None and auc >= 0.65 and auc >= chance + 0.1,
                      f"{auc} against {chance}")]


KIND = Binary()
