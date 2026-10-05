"""The solution, and the heuristics that propose one from the data.

The solution is DCLab rule R01 turned into a form: what is predicted, at which moment,
which columns cannot be known at that moment, which columns are keys rather than
features, and how the result is scored. Heuristics only *propose*; the person decides.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dclab_rnd import tools

from .data import detect_task

TASKS = ("binary", "multiclass", "regression")
METRICS = {
    "binary": [("roc_auc", "ROC-AUC (ranking quality, balanced classes)"), ("average_precision", "Average precision (rare positives)")],
    "multiclass": [("macro_f1", "Macro-F1 (every class counts equally)")],
    "regression": [("mae", "Mean absolute error")],
}


class Forbidden(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    reason: str = Field(default="", max_length=600)


class Solution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    task: Literal["binary", "multiclass", "regression"]
    positive_label: str | None = None
    prediction_moment: str = Field(min_length=12, max_length=2000, description="When the prediction is made and what is known then.")
    forbidden: list[Forbidden] = Field(default_factory=list, max_length=200)
    identifiers: list[str] = Field(default_factory=list, max_length=50)
    time_column: str | None = None
    group_column: str | None = None
    text_columns: list[str] = Field(default_factory=list, max_length=10)
    metric: str | None = None
    notes: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def coherent(self):
        names = {f.column for f in self.forbidden}
        if self.target in names or self.target in self.identifiers or self.target in self.text_columns:
            raise ValueError("The target cannot also be forbidden, an identifier or a text column")
        if self.time_column and self.time_column in names:
            raise ValueError("The time column orders the split; it is excluded from the model automatically, not forbidden")
        if self.metric and self.metric not in {m for m, _ in METRICS[self.task]}:
            raise ValueError(f"Metric {self.metric!r} does not fit a {self.task} task")
        if len(names) != len(self.forbidden):
            raise ValueError("A column is listed twice under forbidden")
        return self

    def check_columns(self, columns: list[str]) -> None:
        known = set(columns)
        for name in [self.target, *(f.column for f in self.forbidden), *self.identifiers, *self.text_columns,
                     *( [self.time_column] if self.time_column else []), *([self.group_column] if self.group_column else [])]:
            if name not in known:
                raise ValueError(f"Column {name!r} is not in the data")

    def resolved_metric(self, positive_rate: float | None = None) -> str:
        if self.metric:
            return self.metric
        return default_metric(self.task, positive_rate)


def default_metric(task: str, positive_rate: float | None = None) -> str:
    if task == "binary":
        return "average_precision" if positive_rate is not None and positive_rate < 0.10 else "roc_auc"
    return "macro_f1" if task == "multiclass" else "mae"


def propose(frame: pd.DataFrame, profile: dict[str, Any], target: str, task: str | None = None) -> dict[str, Any]:
    """What the agent suggests for the solution, with the signal behind each suggestion."""
    if target not in frame.columns:
        raise ValueError(f"Target {target!r} is not in the data")
    detected = detect_task(frame, target)
    task = task if task in TASKS else detected["task"]
    columns = {c["name"]: c for c in profile["columns"]}
    y = frame[target]
    X = frame.drop(columns=[target])
    if len(X) > 20000:
        X = X.sample(20000, random_state=42)
        y = y.loc[X.index]
    audit = tools._audit_frame(X, y, task="regression" if task == "regression" else "classification", blind=False)
    identifiers = [name for name, c in columns.items() if name != target and (c["id_like"] or (c["name_id_like"] and c["unique_ratio"] > 0.5))]
    group_candidates = [name for name, c in columns.items() if name != target and c["name_id_like"] and 0.005 < c["unique_ratio"] < 0.5 and c["kind"] != "numeric"]
    forbidden = []
    for row in audit["flagged"]:
        name = row["column"]
        if name in identifiers:
            continue
        proof = [s.split("precedent ")[-1] for s in row["signals"] if "precedent" in s]
        forbidden.append({"column": name, "reason": "; ".join(row["signals"]), "risk_score": row["risk_score"],
                          "proof": proof + ["DCLAB-R04", "DCLAB-R05"]})
    time_candidates = [name for name in profile["time_candidates"] if name != target]
    text_columns = [name for name in profile["text_candidates"] if name != target]
    positive_rate = detected.get("positive_rate")
    return {
        "target": target, "task": task, "detected": detected,
        "positive_label": detected.get("positive_label"),
        "forbidden": forbidden, "identifiers": identifiers, "time_candidates": time_candidates,
        "group_candidates": group_candidates, "text_columns": text_columns,
        "metric": default_metric(task, positive_rate), "metric_options": METRICS[task],
        "prediction_moment_hint": _moment_hint(task, time_candidates, forbidden),
        "audit_note": audit["note"],
        "all_columns": audit["all_columns"],
    }


def _moment_hint(task: str, time_candidates: list[str], forbidden: list[dict[str, Any]]) -> str:
    parts = ["Describe the moment a real prediction is made and what is already known then."]
    if time_candidates:
        parts.append(f"`{time_candidates[0]}` looks like a timestamp: if rows are events in time, the split should follow it.")
    if forbidden:
        names = ", ".join(f"`{f['column']}`" for f in forbidden[:4])
        parts.append(f"The audit flagged {names} as possibly known only after the outcome; confirm or clear each one.")
    return " ".join(parts)
