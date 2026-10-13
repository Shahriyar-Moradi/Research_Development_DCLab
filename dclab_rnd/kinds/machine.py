"""Kinds with their own machine, run by the engine (package 16.15).

A kind such as forecasting or time to event does not fit the five tabular stages' code (one table, one row per case,
a locked random or grouped holdout), but it fits their meaning. Its machine (``kind.own_machine()``) is run by the same
engine, under the same graph, gates and records:

- data: the kind's view of the table and where its holdout lies (locked here, by the kind's own rule: the last origin
  of a forecast, the latest starts of a survival table); checks on the training rows only;
- leakage: the columns kept out, each with its reason (unknown at the moment, counted during the follow-up...);
- features: what the models read, built from the training rows' past only;
- models: every candidate scored on validation rows inside the training part, and one chosen; the holdout unread;
- final: the chosen model and the baselines scored once on the holdout, after the holdout gate.

Every stage's evidence carries the fields the record readers share (the narrator, the intern's compact record, the
report), filled by :func:`shape`; the kind's own fields sit beside them. A machine record says ``machine: <kind>``, so a
reader that needs tabular fields knows they are not there. Notebook and service exports refuse these projects.
"""

from __future__ import annotations

from typing import Any

import pandas as pd



def claim(cid: str, kind: str, statement: str, evidence: list[str], limitations: list[str] | None = None) -> dict[str, Any]:
    from ..expansion import runner as rn

    return rn._claim(cid, kind, statement, evidence, limitations)


def bootstrap_mean(values: list[float], seed: int = 0, draws: int = 2000) -> tuple[float | None, float | None]:
    """A 95% percentile interval of a mean over the units given (series, rows), resampled with replacement."""
    import numpy as np

    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (draws, len(v)))].mean(axis=1)
    return round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)


COMMON: dict[str, dict[str, Any]] = {  # what the shared readers index on each stage, with neutral values
    "data": {"train_rows": None, "holdout_rows": None, "holdout_split": "", "cv_protocol": "", "usable_feature_count": 0, "risk_features": {},
             "sampling": {}, "source_rows": None},
    "leakage": {"declared_leakage_features": [], "apparent_lift": None, "apparent_lift_metric": None, "heuristic_review_candidates": [],
                "detector_canary_passed": None},
    "features": {"selected_recipe": "", "best_recipe": "", "selected_recipe_description": "", "best_metric_mean": None, "selected_metric_mean": None,
                 "stage_results": [], "selection_rule": ""},
    "models": {"selected_model": "", "ranking": [], "selection_rule": ""},
    "final": {"model": "", "feature_recipe": "", "selected_model_params": {}, "selected_optimization": "defaults",
              "tuning_decision": {"accepted": False, "gain": 0.0, "required_gain": 0.0, "best_candidate": None},
              "holdout_consumed": True, "holdout_metrics": {}, "holdout_primary_metric_ci": {"low": None, "high": None, "point": None, "method": ""},
              "cv_selected_metric_mean": None, "cv_to_holdout_gap": None, "baselines": {}, "holdout_split": "", "holdout_rows": None,
              "train_rows": None, "production_approved": False, "selection_rule": ""},
}


def shape(stage: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """The kind's evidence with the shared fields filled where it has none."""
    out = {**COMMON[stage], **evidence}
    if stage == "final":
        out["holdout_consumed"], out["production_approved"] = True, False  # never left to a kind
    return out


class Machine:
    """A kind's machine for the five stages. ``run`` returns (evidence, claims, summary, decision) for one stage, reading
    the earlier stages' records it needs (``previous``); the holdout is read in ``final`` only."""

    task_type = "machine"
    primary_metric = ""
    higher_is_better = False

    def __init__(self, kind: Any, solution: Any, frame: pd.DataFrame, project: dict[str, Any]):
        self.kind, self.solution, self.frame, self.project = kind, solution, frame, project
        self.fields = kind.fields(solution.kind_fields)

    # what the record's provenance says
    def holdout_policy(self) -> str:
        return "locked by the kind's rule in the data stage, read once in the final stage"

    def cv_description(self) -> str:
        return ""

    def holdout_description(self) -> str:
        return ""

    def training_rows(self) -> Any:
        """The rows every stage before the final may read (the holdout left out)."""
        raise NotImplementedError

    def run(self, stage: str, eid: str, previous: dict[str, dict[str, Any] | None]) -> tuple[dict, list, str, dict | None]:
        evidence, claims, summary, decision = getattr(self, stage)(eid, previous)
        return shape(stage, evidence), claims, summary, decision

    @staticmethod
    def chosen(record: dict[str, Any] | None) -> Any:
        """The choice a stage record holds (a person's approval may have changed it)."""
        decision = (record or {}).get("decision") or {}
        return decision.get("chosen") or decision.get("selected")
