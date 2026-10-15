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
    target: str | None = None  # 20.7: none only for a kind that reads no outcome (anomaly detection without labels)
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
    labels: Literal["complete", "partial", "none"] = "complete"  # none: nothing is labelled (20.7)
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
        if self.target is None and getattr(kind, "needs_target", True):
            raise ValueError("Name the outcome column: this kind of problem learns from it")
        if self.labels == "none" and getattr(kind, "needs_target", True):
            raise ValueError("A problem that learns from an outcome needs labels: with none, it is anomaly detection")
        if self.target is not None and not getattr(kind, "needs_target", True):
            raise ValueError(f"The {kind.title.lower()} kind reads no outcome column: leave the target empty")
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
        for name in [*([self.target] if self.target else []), *(f.column for f in self.forbidden), *self.identifiers, *self.text_columns,
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


def propose(frame: pd.DataFrame, profile: dict[str, Any], target: str, task: str | None = None, pack: str | None = None,
            time_column: str | None = None) -> dict[str, Any]:
    """What the agent suggests for the solution, with the signal behind each suggestion. The problem kind (16.1) follows
    from the task, the Domain pack and the text columns; its metrics are the options offered. ``time_column``: the
    column the person (or a studied sample) says orders the rows, read first among the time candidates."""
    if target not in frame.columns:
        raise ValueError(f"Target {target!r} is not in the data")
    detected = detect_task(frame, target)  # the target's form: its values and classes, over every row
    task = task if task in TASKS else detected["task"]
    columns = {c["name"]: c for c in profile["columns"]}
    time_candidates = [name for name in profile["time_candidates"] if name != target]
    if time_column and time_column in frame.columns and time_column != target:
        time_candidates = [time_column] + [c for c in time_candidates if c != time_column]
    # What reads the target's relation to the rest (the audit, the plan, the class balance behind the metric) reads the
    # provisional training rows only, the earliest when a column orders the rows in time: the rows a later holdout takes
    # never shape what is proposed (16.15). The rows the engine drops (no target, a target it cannot read) are not read.
    usable = frame[target].notna()
    if task == "regression":
        usable &= pd.to_numeric(frame[target], errors="coerce").notna()
    present = frame.index[usable]
    read, by_time = provisional_reading(present, frame.loc[present, time_candidates[0]] if time_candidates else None)
    y = frame.loc[read, target]
    X = frame.loc[read].drop(columns=[target])
    if detected.get("task") == "binary" and len(y):  # the class balance that decides the metric, on the rows read
        rate = float((y.astype(str) == str(detected["positive_label"])).mean())
        detected = {**detected, "positive_rate": rate,
                    "note": f"Two classes; `{detected['positive_label']}` is treated as the positive class ({rate:.1%} of the rows read)."}
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
    decided = _model_leaks(columns, profile, target, task, identifiers, forbidden, time_candidates, group_candidates)  # 22.9
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
        "rows_read": rows_read_note(len(read), len(present), time_candidates[0] if time_candidates else None, by_time),
        "all_columns": audit["all_columns"],
        "plan": _core_plan(X, y, target, task, time_candidates[0] if time_candidates else None, rows=X.index),
        **({"decided_by_models": decided} if decided else {}),
    }


MODEL_PROOF = "decision-model:"  # a forbidden column a routed decision model added (22.9), never the audit's


def _model_leaks(columns: dict[str, Any], profile: dict[str, Any], target: str, task: str, identifiers: list[str],
                 forbidden: list[dict[str, Any]], time_candidates: list[str], group_candidates: list[str] = ()) -> list[dict[str, Any]]:
    """22.9: the routed leakage model asked about each column the rules would model. It may forbid one the audit did not
    flag (written after the outcome), never free one the audit flagged, and never forbid a column the split may be
    built on (a time, a group, a coordinate): the person may choose it there, and a forbidden one cannot be. The rules
    decide when nothing is routed."""
    from dclab_core.behaviour import coordinate_pairs

    from ..ledger import column_state, use
    from ..ledger.capture import TASKS

    if not use.routed("feature.leakage_suspect"):
        return []
    flagged = {f["column"] for f in forbidden}
    structural = set(time_candidates) | set(group_candidates) | {c for pair in coordinate_pairs(list(columns)) for c in pair[1:]}
    out = []
    for name, column in list(columns.items())[:200]:
        if name == target or name in identifiers:
            continue
        rule = name in flagged

        def check(value: Any, name: str = name, rule: bool = rule) -> list[str]:
            if rule and value is False:
                return ["the audit flagged it: a model never frees a column the rules forbid"]
            if value is True and name in structural:
                return ["the split may be built on it (a time, a group or a coordinate): a model does not forbid it"]
            return []

        state = {**column_state(column, profile.get("rows")), "target": str(target)[:200], "task": TASKS.get(task)}
        choice = use.choose("feature.leakage_suspect", state, ["yes", "no"], rule, "rule:column_audit", subject=name, check=check)
        if choice.by_model and choice.value is True and not rule:
            forbidden.append({"column": name, "reason": f"the decision model {choice.answerer} reads it as written after the outcome",
                              "risk_score": None, "proof": [MODEL_PROOF + str(choice.answerer)]})
        if choice.level:
            out.append({"column": name, "answerer": choice.answerer, "level": choice.level, "model": choice.model_value,
                        "used": choice.used, "changed": choice.changed, "review": choice.review, "refused": choice.refused})
    return out


PROVISIONAL_SHARE = 0.75  # at most 1 - the largest holdout an engine split takes (20 % tabular, 25 % time to event)
MIN_TIMED_ROWS = 30  # fewer rows with a valid time than this: the rows cannot be ordered, a seeded share is read


def provisional_reading(index: pd.Index, times: Any = None) -> tuple[pd.Index, bool]:
    """75 % of the rows: what may be read before the engine locks its holdout (the rest stays unseen), and whether they
    were chosen by time. With ``times`` (the rows' time column), the earliest 75 % of the rows with a valid time and no
    row past the earliest 75 % of its distinct times, so a later holdout (the latest 20 % of a tabular split, the latest
    25 % of starts, a forecast's last dates) is never read; a row without a valid time is never read (the engine drops it,
    or orders it last). Otherwise a seeded random 75 %, which a random holdout drawn later overlaps (``rows_read_note``
    says so). Pass the rows that have a target."""
    import numpy as np

    if times is not None:
        stamps = pd.Series(list(times), index=index)
        stamps = pd.to_numeric(stamps, errors="coerce") if pd.api.types.is_numeric_dtype(stamps) else pd.to_datetime(stamps, errors="coerce")
        valid = stamps.dropna()
        if len(valid) >= MIN_TIMED_ROWS:
            distinct = np.sort(valid.unique())
            cutoff = distinct[max(int(PROVISIONAL_SHARE * len(distinct)) - 1, 0)]
            keep = (valid.rank(method="first") <= int(PROVISIONAL_SHARE * len(valid))) & (valid <= cutoff)
            return valid.index[keep.to_numpy()], True
    keep = np.random.default_rng(42).random(len(index)) < PROVISIONAL_SHARE
    return index[keep], False


def provisional_training_rows(index: pd.Index, times: Any = None) -> pd.Index:
    """The rows of :func:`provisional_reading`."""
    return provisional_reading(index, times)[0]


def rows_read_note(read: int, rows: int, time_column: str | None, by_time: bool) -> str:
    """What the proposal read of the target, in words a person can check."""
    if by_time:
        return (f"The target was read on the earliest {read} of {rows} rows by `{time_column}` (rows without a valid time left out): "
                f"a holdout of later rows by `{time_column}` is never read before the engine locks it.")
    why = (f"`{time_column}` has too few valid times to order the rows" if time_column else "no column orders the rows in time")
    return (f"The target was read on a seeded {read} of {rows} rows: {why}, so a random holdout drawn later shares about three "
            "quarters of its rows with them.")


def core_plan(X: pd.DataFrame, y: pd.Series, target: str, task: str, time_column: str | None = None,
              investigator: Any = None, planner: Any = None, moment: str | None = None, brief: Any = None, rows: Any = None):
    """dclab_core's findings and plan for this table (package 17.4), read on the provisional training rows only (the
    earliest ones when the table has a time column). ``investigator`` and ``planner``: the roles' advisors (18.9), when
    they are on; each proposal is checked by code, and the rules stand when they abstain."""
    from dclab_core.analyse import analyse
    from dclab_core.plan import plan

    frame = X.assign(**{target: y})
    if rows is None:  # ``rows``: the provisional training rows, when the caller has chosen them already
        present = frame.index[frame[target].notna()]
        times = frame.loc[present, time_column] if time_column and time_column in frame.columns else None
        rows = provisional_training_rows(present, times)
    findings = analyse({"table": frame}, target=target, task=task, brief=brief, train_index=pd.Index(rows),
                       advisor=investigator, moment=moment)
    if planner is not None:
        from dclab_core.plan import plan_with

        return findings, plan_with(planner, brief, findings)["plan"]
    return findings, plan(brief, findings)


def _core_plan(X: pd.DataFrame, y: pd.Series, target: str, task: str, time_column: str | None = None, rows: Any = None) -> list[dict[str, Any]] | None:
    """The plan, compact (each decision, its choice and why): advice beside the proposal above, which it does not
    change; adopting it is a set_solution move (the intern's adopt_plan). None, logged, when the core cannot read the
    table: the proposal stands without it."""
    try:
        return [{"decision": d.key, "choice": d.choice, "why": d.reason} for d in core_plan(X, y, target, task, time_column, rows=rows)[1].decisions]
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
