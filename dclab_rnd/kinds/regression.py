"""Regression: a number for each row."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .base import ProblemKind, _get


class Regression(ProblemKind):
    key = "regression"
    title = "Regression"
    description = "A number for each row: a delivery time, a price, a count."
    base_task = "regression"
    task_type = "timeseries_regression"
    words = r"\b(how (much|many|long)|minutes|price|amount|duration|estimate[sd]?|revenue)\b"
    questions = ("Which number should be predicted, in which unit, and for which rows?",
                 "When is the prediction made, and what is known then?")
    metrics = (("mae", "Mean absolute error"),)
    baselines = ("training_mean", "training_median")
    checks = ("PIT-001", "PIT-002", "PIT-004", "PIT-005", "PIT-006")
    reliability = ("errors by size of the target", "drift by time", "capped or censored targets")
    evidence_track = "tabular-regression"
    aliases = ('regression', 'predicting a number', 'delivery time', 'minutes')  # 16.14: the names this kind of problem goes by (read into its evidence record)
    stratify = False

    def target(self, frame: pd.DataFrame, solution: Any) -> tuple[pd.Series, list[str] | None]:
        return pd.to_numeric(frame[_get(solution, "target")], errors="coerce").astype(float), None

    def summarise_target(self, y: pd.Series, class_labels: list[str] | None, solution: Any) -> tuple[dict[str, Any], str]:
        summary = {"mean_train": float(y.mean()), "std_train": float(y.std()), "min_train": float(y.min()),
                   "median_train": float(y.median()), "max_train": float(y.max()),
                   "coefficient_of_variation_train": float(y.std() / y.mean()) if y.mean() else None}
        return summary, f"a target mean of {y.mean():.4g} (std {y.std():.4g}, range {y.min():.4g}–{y.max():.4g})"

    def target_claim(self, summary: dict[str, Any], metric_label: str, spec: Any) -> tuple[str, str, list[str]]:
        return ("fact", f"The target's coefficient of variation is {summary['coefficient_of_variation_train'] or 0:.2f}; "
                        + (f"rows are ordered by `{spec.time_column}` so the holdout is strictly later than training." if spec.time_column
                           else "no time column was declared, so rows are split at random; if these rows are events in time, declare the time column."),
                ["evidence.target_summary", "evidence.holdout_split"])


    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import delivery_time

        from .acceptance import Case

        replica = delivery_time.generate(seed=seed, n=3000)
        return Case(replica.orders, "actual_minutes", "regression", "When the order is placed, before a courier is assigned.", facts=replica.facts)

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check, kept_out

        mae = run.holdout("mae")
        median = run.evidence("final").get("baselines", {}).get("training_median", {}).get("mae")
        at_cap = (run.evidence("final").get("task_assessment") or {}).get("at_cap") or {}
        return [kept_out(run, "delivered_at", "the target can be recomputed from it", roles=("blocked",)),
                kept_out(run, "courier_rating", "given after the delivery", roles=("blocked",)),
                check("scored by mean absolute error", run.records["final"].get("primary_metric") == "mae",
                      str(run.records["final"].get("primary_metric"))),
                check("the holdout MAE is at most 0.9 times the training median's", mae is not None and median is not None and mae <= 0.9 * median,
                      f"{mae} against {median}"),
                check("the cap at 60 minutes is found on the training rows and its errors called censored",
                      at_cap.get("cap") == 60.0 and "censored" in str(at_cap.get("note")), json.dumps(at_cap)[:160])]


KIND = Regression()
