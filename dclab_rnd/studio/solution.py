"""The solution, and the heuristics that propose one from the data.

The solution is DCLab rule R01 turned into a form: what is predicted, at which moment,
which columns cannot be known at that moment, which columns are keys rather than
features, and how the result is scored. Heuristics only *propose*; the person decides.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dclab_rnd import kinds, tools

from .data import detect_task

TASKS = kinds.BASE_TASKS
METRICS = {task: list(kinds.get(task).metrics) for task in TASKS}  # each task's own kind (16.1); a kind may offer others


class Forbidden(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    reason: str = Field(default="", max_length=600)


class Decision(BaseModel):
    """The action taken on top of the model (package 16.9), with what it is worth and what it costs.

    ``review``: the team reviews the riskiest rows; finding a case is worth ``value_per_case``, a review costs
    ``cost_per_review``. ``price``: an offer at ``price_column`` is accepted or not (the target); selling, an accepted
    offer earns the price less ``unit_cost``; paying (a fee to a courier or a supplier), it earns ``unit_value_column``
    (or ``unit_value_factor`` times the reference price) less the price."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["review", "price"]
    value_per_case: float | None = Field(default=None, gt=0)
    cost_per_review: float | None = Field(default=None, gt=0)
    price_column: str | None = None
    reference_column: str | None = None
    side: Literal["sell", "pay"] = "sell"
    unit_cost: float = Field(default=0.0, ge=0)
    unit_value_column: str | None = None
    unit_value_factor: float | None = Field(default=None, gt=0)
    # the columns that record a price test's arm: they explain the price because the test set it, so they are left out
    # of the check that prices were varied and of the acceptance model's context
    test_arm_columns: list[str] = Field(default_factory=list, max_length=10)
    notes: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def complete(self):
        if self.action == "review" and (self.value_per_case is None or self.cost_per_review is None):
            raise ValueError("A review decision needs what finding a case is worth and what a review costs")
        if self.action == "price":
            if not self.price_column:
                raise ValueError("A price decision needs the column that holds the price offered")
            if self.side == "pay" and not (self.unit_value_column or (self.unit_value_factor and self.reference_column)):
                raise ValueError("Paying a price needs what an accepted offer is worth: a column, or a factor of the reference price")
        return self

    def columns(self) -> list[str]:
        return [c for c in (self.price_column, self.reference_column, self.unit_value_column, *self.test_arm_columns) if c]


class SpatialSplit(BaseModel):
    """Package 16.10: places that cluster are split by area. The engine derives square blocks of ``block_km`` from the
    coordinates and keeps each block on one side of every split."""

    model_config = ConfigDict(extra="forbid")
    lat_column: str
    lon_column: str
    block_km: float = Field(gt=0, le=1000)


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
    # package 15.2: labels from a sample (or a person who says some cases were never found) are partial: an unlabelled
    # row is unknown, not a negative, and every precision is a lower bound
    labels: Literal["complete", "partial"] = "complete"
    label_provenance: str = Field(default="", max_length=600)
    # package 15.3: the share of rows the team can review (asked in words: "about 70 a day" over the rows of a day)
    alert_budget: float | None = Field(default=None, gt=0, lt=1)
    # package 16.1: the problem kind (dclab_rnd.kinds) and its own fields, validated by the kind. None: the kind follows
    # from the task, the project's Domain pack and the text columns (kinds.resolve)
    kind: str | None = Field(default=None, max_length=60)
    kind_fields: dict[str, Any] = Field(default_factory=dict)
    # package 16.9: the decision on top of the model (a review budget or a price), chosen by expected value
    decision: Decision | None = None
    # package 16.10: a split by blocks of places, when nearby rows have alike targets
    spatial: SpatialSplit | None = None

    @model_validator(mode="after")
    def coherent(self):
        names = {f.column for f in self.forbidden}
        if self.target in names or self.target in self.identifiers or self.target in self.text_columns:
            raise ValueError("The target cannot also be forbidden, an identifier or a text column")
        if self.time_column and self.time_column in names:
            raise ValueError("The time column orders the split and is kept out of the model automatically, so it cannot also be "
                             "forbidden: untick it under Forbidden, or choose another time column")
        if self.spatial is not None:
            if self.group_column or self.time_column:
                raise ValueError("A split by blocks of places is one way to split: it cannot be combined with a group or a time column here")
            coords = {self.spatial.lat_column, self.spatial.lon_column}
            if self.target in coords or coords & names:
                raise ValueError("The coordinates that make the blocks cannot be the target or forbidden")
        kind = self.problem_kind()
        if self.kind_fields and not self.kind:
            raise ValueError("A kind's fields need the kind itself: declare `kind` with them")
        if not kind.runs_in_engine:
            raise ValueError(f"The {kind.title.lower()} kind runs as a library (dclab_core) for now: the notebook's engine does not run it "
                             "yet, so a project cannot declare it")
        if kind.base_task != self.task:
            raise ValueError(f"The {kind.title.lower()} kind runs as a {kind.base_task} task, not {self.task}")
        self.kind_fields = kind.fields(self.kind_fields)
        kind.check_solution(self)  # 16.15: what the kind's machine needs (a forecast needs its time column)
        if self.metric and self.metric not in kind.metric_keys():
            raise ValueError(f"Metric {self.metric!r} does not fit a {self.task} task")
        if len(names) != len(self.forbidden):
            raise ValueError("A column is listed twice under forbidden")
        if self.decision is not None and self.task != "binary":
            raise ValueError("A review or price decision reads a yes/no outcome (a case, an accepted offer): the task must be binary")
        if self.decision is not None and self.target in self.decision.columns():
            raise ValueError("The decision's price or value column cannot be the target")
        if self.decision is not None and names & set(self.decision.columns()):
            raise ValueError("A column the decision reads cannot be forbidden: "
                             + ", ".join(sorted(names & set(self.decision.columns())))
                             + " (a price, its reference and the value of an offer are known when the offer is made)")
        if self.metric == "recall_at_budget" and not self.alert_budget:
            raise ValueError("Recall at the review budget needs the budget: say how many rows the team can review")
        return self

    def problem_kind(self, pack: str | None = None):
        """The kind this solution runs as (16.1): the declared one, else the pack's, else the task's."""
        try:
            return kinds.resolve(self, pack)
        except KeyError as error:
            raise ValueError(str(error.args[0])) from None

    def check_columns(self, columns: list[str]) -> None:
        known = set(columns)
        for name in [self.target, *(f.column for f in self.forbidden), *self.identifiers, *self.text_columns,
                     *( [self.time_column] if self.time_column else []), *([self.group_column] if self.group_column else []),
                     *(self.decision.columns() if self.decision else []),
                     *([self.spatial.lat_column, self.spatial.lon_column] if self.spatial else [])]:
            if name not in known:
                raise ValueError(f"Column {name!r} is not in the data")

    def resolved_metric(self, positive_rate: float | None = None) -> str:
        if self.metric:
            return self.metric
        return default_metric(self.task, positive_rate)


def default_metric(task: str, positive_rate: float | None = None) -> str:
    return kinds.get(task).default_metric(None, positive_rate)


def propose(frame: pd.DataFrame, profile: dict[str, Any], target: str, task: str | None = None, pack: str | None = None) -> dict[str, Any]:
    """What the agent suggests for the solution, with the signal behind each suggestion. The problem kind (16.1) follows
    from the task, the Domain pack and the text columns; its metrics are the options offered."""
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
    time_candidates = [name for name in profile["time_candidates"] if name != target]
    forbidden = []
    for row in audit["flagged"]:
        name = row["column"]
        if name in identifiers:
            continue
        # A timestamp is unique on most rows; that makes it the time key, not a leakage candidate. It stays flagged
        # when the audit has any other signal against it (then it should not order the split either).
        if name in time_candidates and all(s.startswith("identifier-like") for s in row["signals"]):
            continue
        # free text is unique on most rows too; that makes it text to model, not an identifier (16.2's review)
        if name in profile.get("text_candidates", []) and all(s.startswith("identifier-like") for s in row["signals"]):
            continue
        proof = [s.split("precedent ")[-1] for s in row["signals"] if "precedent" in s]
        forbidden.append({"column": name, "reason": "; ".join(row["signals"]), "risk_score": row["risk_score"],
                          "proof": proof + ["DCLAB-R04", "DCLAB-R05"]})
    text_columns = [name for name in profile["text_candidates"] if name != target]
    positive_rate = detected.get("positive_rate")
    kind = kinds.resolve({"task": task, "text_columns": text_columns}, pack)
    return {
        "target": target, "task": task, "kind": kind.key, "kind_title": kind.title, "detected": detected,
        "positive_label": detected.get("positive_label"),
        "forbidden": forbidden, "identifiers": identifiers, "time_candidates": time_candidates,
        "group_candidates": group_candidates, "text_columns": text_columns,
        # the metric proposed stays the task's own, as before 16.1 (the wizard saves it): a pack's metric applies when the
        # person leaves the metric empty
        "metric": default_metric(task, positive_rate), "metric_options": METRICS[task],
        "prediction_moment_hint": _moment_hint(task, time_candidates, forbidden),
        "audit_note": audit["note"],
        "all_columns": audit["all_columns"],
        "plan": _core_plan(X, y, target, task, time_candidates[0] if time_candidates else None),
    }


def provisional_training_rows(index: pd.Index, times: Any = None) -> pd.Index:
    """80 % of the rows: what the plan may read before the engine locks its holdout (the rest stays unseen). With
    ``times`` (the rows' time column), the earliest 80 %, so a later, time-based holdout is never read; otherwise a seeded
    random 80 %."""
    import numpy as np

    if times is not None:
        stamps = pd.to_datetime(pd.Series(list(times), index=index), errors="coerce")
        if stamps.notna().mean() > 0.9:
            order = stamps.rank(method="first").fillna(len(index))
            return index[(order <= int(0.8 * len(index))).to_numpy()]
    keep = np.random.default_rng(42).random(len(index)) < 0.8
    return index[keep]


def core_plan(X: pd.DataFrame, y: pd.Series, target: str, task: str, time_column: str | None = None,
              investigator: Any = None, planner: Any = None, moment: str | None = None, brief: Any = None):
    """dclab_core's findings and plan for this table (package 17.4), read on the provisional training rows only (the
    earliest ones when the table has a time column). ``investigator`` and ``planner``: the roles' advisors (18.9), when
    they are on; each proposal is checked by code, and the rules stand when they abstain."""
    from dclab_core.analyse import analyse
    from dclab_core.plan import plan

    frame = X.assign(**{target: y})
    times = frame[time_column] if time_column and time_column in frame.columns else None
    findings = analyse({"table": frame}, target=target, task=task, brief=brief, train_index=provisional_training_rows(frame.index, times),
                       advisor=investigator, moment=moment)
    if planner is not None:
        from dclab_core.plan import plan_with

        return findings, plan_with(planner, brief, findings)["plan"]
    return findings, plan(brief, findings)


def _core_plan(X: pd.DataFrame, y: pd.Series, target: str, task: str, time_column: str | None = None) -> list[dict[str, Any]] | None:
    """The plan, compact (each decision, its choice and why): advice beside the proposal above, which it does not
    change; adopting it is a set_solution move (the intern's adopt_plan). None, logged, when the core cannot read the
    table: the proposal stands without it."""
    try:
        return [{"decision": d.key, "choice": d.choice, "why": d.reason} for d in core_plan(X, y, target, task, time_column)[1].decisions]
    except Exception as exc:  # noqa: BLE001 - a table the core cannot read leaves the proposal as it was
        from ..observe import log

        log.warning("core plan unavailable for target %s: %s", target, type(exc).__name__)
        return None


def _moment_hint(task: str, time_candidates: list[str], forbidden: list[dict[str, Any]]) -> str:
    parts = ["Describe the moment a real prediction is made and what is already known then."]
    if time_candidates:
        parts.append(f"`{time_candidates[0]}` looks like a timestamp: if rows are events in time, the split should follow it.")
    if forbidden:
        names = ", ".join(f"`{f['column']}`" for f in forbidden[:4])
        parts.append(f"The audit flagged {names} as possibly known only after the outcome; confirm or clear each one.")
    return " ".join(parts)
