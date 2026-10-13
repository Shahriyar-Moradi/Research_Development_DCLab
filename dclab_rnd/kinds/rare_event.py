"""Rare events: fraud, abuse, defaults, intrusions (Part 4). A binary kind with rare positives, often partial labels, an
alert budget and entities whose behaviour matters."""

from __future__ import annotations

from typing import Any

from .binary import Binary


class RareEvent(Binary):
    key = "rare_event"
    title = "Rare events and fraud"
    description = "Rare positives: average precision, recall at the review budget, rules, anomaly detectors and PU learning."
    specificity = 20
    packs = ("imbalanced",)
    words = r"\b(fraud|rare|anomal|default(s|ed)? on|chargeback|intrusion|0\.\d+ ?% positive)"
    questions = ("What is the rare event the model should flag, what is scored (a transaction, an account, a claim), and how long "
                 "after it happens is the true label known?",
                 "Were all cases found, or only some (a sample, an investigation)? An unlabelled row may be a case nobody found.",
                 "How many alerts can the team review, for example a day?")
    home_fields = ("target", "prediction_moment", "entity", "labels", "budget", "data_plan")  # 15.6
    moments = ("At the moment of the transaction", "Before approving the case", "In a nightly review")
    metrics = (("average_precision", "Average precision (rare positives)"),
               ("recall_at_budget", "Recall at the review budget (the share of cases found when the team reviews its budget)"),
               ("roc_auc", "ROC-AUC (ranking quality, balanced classes)"))
    baselines = ("majority_class",)  # rules and detectors are screened as families
    checks = ("PIT-001", "PIT-002", "PIT-003", "PIT-004", "PIT-005", "PIT-006")
    reliability = ("precision at the budget as a lower bound when labels are partial", "entities on both sides of the split",
                   "drift by time")
    evidence_track = "anomaly-and-fraud-detection"
    aliases = ('fraud detection', 'imbalanced classification', 'rare positives', 'anomaly detection')  # 16.14: the names this kind of problem goes by (read into its evidence record)
    pack_defaults = {"metric": "average_precision", "rare_event": True}

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return super().default_metric(solution, positive_rate, {**self.pack_defaults, **(pack or {})})

    def spec_settings(self, solution: Any, pack: dict[str, Any] | None = None) -> dict[str, Any]:
        return super().spec_settings(solution, {**self.pack_defaults, **(pack or {})})

    def detect(self, problem: str, profile: dict[str, Any] | None = None) -> str | None:
        found = super().detect(problem, profile)
        labels = (profile or {}).get("labels") or {}
        if labels.get("rows") and labels.get("rows_marked") is not None and labels["rows_marked"] / labels["rows"] < 0.01:
            share = labels["rows_marked"] / labels["rows"]
            return f"the label file marks {labels['rows_marked']} of {labels['rows']} rows ({share:.2%}): rare positives"
        return found


    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import delivery_fraud as dfr

        from .acceptance import Case

        replica = dfr.generate(seed=seed)

        def files(directory):  # the person's two files: the rides, and the sample of found cases (15.2)
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmp:
                paths = dfr.save(replica, Path(tmp))
                (directory / dfr.RIDE_FILE).write_bytes(paths["rides"].read_bytes())
                (directory / dfr.SAMPLE_FILE).write_bytes(paths["sample"].read_bytes())
            return dfr.RIDE_FILE, dfr.SAMPLE_FILE
        return Case(replica.rides, "is_fraud", "binary", "After delivery, in a daily audit of the day's rides; no label is known when a ride "
                    "is scored.", {"group_column": "biker_id", "alert_budget": 0.005}, pack="imbalanced", labels=(replica.sample, "is_fraud"),
                    files=files, quick=False, facts={"budget": 0.005})

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        features, models, final = run.records["features"], run.evidence("models"), run.evidence("final")
        screened = {row["model"] for row in models.get("model_results", [])}
        ap, chance = run.holdout("average_precision"), final.get("baselines", {}).get("majority_class", {}).get("average_precision_random_ranking")
        recall = run.holdout("recall_at_budget")
        key = list((run.labels or {}).get("key") or [])
        return [check("the label file is joined on the order, never on a coarser key", key == ["order_id"], f"joined on {key}"),
                check("labels from a sample are partial: an unlabelled ride is unknown", run.solution.get("labels") == "partial",
                      str(run.solution.get("labels"))),
                check("behaviour features are chosen over the raw columns", (features.get("decision") or {}).get("chosen") in ("entity_behaviour", "engineered"),
                      str((features.get("decision") or {}).get("chosen"))),
                check("folds are grouped by biker", "biker_id" in str(run.evidence("features").get("cv_protocol")), str(run.evidence("features").get("cv_protocol"))),
                check("rules, anomaly detectors and PU bagging are screened", {"isolation_forest", "pu_bagging", "percentile_rules"} <= screened,
                      ", ".join(sorted(screened))),
                check("the holdout average precision is above 0.35 and 20 times chance", ap is not None and chance is not None and ap > 0.35 and ap > 20 * chance,
                      f"{ap} against {chance}"),
                check("recall at the budget beats reviewing at random", recall is not None and recall > 0.005, str(recall)),
                check("precision is reported as a lower bound", "lower bound" in str(final.get("precision_note")), str(final.get("precision_note"))[:120])]


KIND = RareEvent()
