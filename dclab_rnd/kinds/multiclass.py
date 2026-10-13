"""Multiclass classification: one of several classes for each row."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import ProblemKind, _get


class Multiclass(ProblemKind):
    key = "multiclass"
    title = "Multiclass classification"
    description = "One of several classes for each row: a ticket's team, a product's category."
    base_task = "multiclass"
    task_type = "multiclass"
    words = r"\b(which (team|category|class|type|department|queue)|categor(y|ies|ise|ize)|route (it|them|each) to)\b"
    questions = ("Which classes are there, and does every class matter equally or do some matter more?",
                 "When is the prediction made, and what is known then?")
    metrics = (("macro_f1", "Macro-F1 (every class counts equally)"),)
    baselines = ("majority_class",)
    checks = ("PIT-001", "PIT-002", "PIT-004", "PIT-005", "PIT-006")
    reliability = ("per-class recall", "rare classes", "drift by time")
    evidence_track = "tabular-classification"
    aliases = ('multiclass classification', 'ticket routing', 'category classifier', 'uneven classes')  # 16.14: the names this kind of problem goes by (read into its evidence record)

    def target(self, frame: pd.DataFrame, solution: Any) -> tuple[pd.Series, list[str] | None]:
        raw = frame[_get(solution, "target")]
        labels = sorted(raw.dropna().astype(str).unique(), key=str)
        mapping = {label: i for i, label in enumerate(labels)}
        return raw.astype(str).map(mapping).astype(int), labels

    def summarise_target(self, y: pd.Series, class_labels: list[str] | None, solution: Any) -> tuple[dict[str, Any], str]:
        counts = y.value_counts().sort_index()
        share = counts / counts.sum()
        summary = {"classes": int(len(counts)),
                   "class_counts_train": {str(class_labels[i] if class_labels else i): int(v) for i, v in counts.items()},
                   "smallest_class_count": int(counts.min()), "largest_class_count": int(counts.max()),
                   "imbalance_ratio_max_min": float(counts.max() / max(counts.min(), 1)),
                   "majority_baseline_accuracy_train": float(share.max())}
        return summary, f"{len(counts)} classes with {counts.min()}–{counts.max()} training rows each"

    def target_claim(self, summary: dict[str, Any], metric_label: str, spec: Any) -> tuple[str, str, list[str]]:
        return ("risk", f"The smallest class has {summary['smallest_class_count']} training rows (largest/smallest ratio "
                        f"{summary['imbalance_ratio_max_min']:.1f}); macro-F1 weights every class equally, so rare classes decide the score.",
                ["evidence.target_summary"])


    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import ticket_routing

        from .acceptance import Case

        replica = ticket_routing.generate(seed=seed, n=3000)
        return Case(replica.tickets, "assigned_team", "multiclass", "When the ticket is created, before anyone reads it.", facts=replica.facts)

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check, kept_out

        f1 = run.holdout("macro_f1")
        majority = run.evidence("final").get("baselines", {}).get("majority_class", {}).get("macro_f1")
        return [kept_out(run, "resolved_by", "the agent who closed the ticket, nearly a copy of the label", roles=("blocked",)),
                kept_out(run, "first_response_minutes", "known only after the queue answers", roles=("blocked",)),
                check("scored by macro-F1 (two teams hold 0.3 % of tickets each)", run.records["final"].get("primary_metric") == "macro_f1",
                      str(run.records["final"].get("primary_metric"))),
                check("the holdout macro-F1 is 0.1 above the majority class", f1 is not None and majority is not None and f1 >= majority + 0.1,
                      f"{f1} against {majority}"),
                check("no copy of the label is left (holdout macro-F1 below 0.99)", f1 is not None and f1 < 0.99, str(f1)),
                check("the worst classes are named", bool(run.evidence("final").get("holdout_metrics", {}).get("worst_3_classes")), "")]


KIND = Multiclass()
