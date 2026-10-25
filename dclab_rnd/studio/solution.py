"""The solution, and the heuristics that propose one from the data.

The solution is DCLab rule R01 turned into a form: what is predicted, at which moment,
which columns cannot be known at that moment, which columns are keys rather than
features, and how the result is scored. Heuristics only *propose*; the person decides.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
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
    """The action taken on top of the model (package 16.9, v2 in 34.4), with what it is worth and what it costs.

    ``review``: the team reviews the riskiest rows; finding a case is worth ``value_per_case``, a review costs
    ``cost_per_review``. ``price``: an offer at ``price_column`` is accepted or not (the target); selling, an accepted
    offer earns the price less ``unit_cost``; paying (a fee to a courier or a supplier), it earns ``unit_value_column``
    (or ``unit_value_factor`` times the reference price) less the price. ``act`` (34.4): any action on a row, stated as
    dclab_core's DecisionSpec (a case's value or a column of values, a cost per action, a false alarm's and a miss's
    cost, a capacity as a share or per day). ``classes`` (34.4): each row's class with the lowest expected cost, from a
    cost matrix (row: the true class, column: the class chosen) over ``classes`` in that order."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["review", "price", "act", "classes"]
    value_per_case: float | None = Field(default=None, gt=0)
    cost_per_review: float | None = Field(default=None, gt=0)
    # act (34.4): what is done to a row, in the person's words, and dclab_core.decide.DecisionSpec's numbers
    what: str = Field(default="", max_length=80)  # DecisionSpec.action's own bound
    cost_per_action: float = Field(default=0.0, ge=0)
    cost_false_positive: float = Field(default=0.0, ge=0)
    cost_false_negative: float = Field(default=0.0, ge=0)
    value_column: str | None = None
    capacity_share: float | None = Field(default=None, gt=0, lt=1)
    capacity_per_day: float | None = Field(default=None, gt=0)
    # classes (34.4): the classes in the cost matrix's order and the matrix itself
    classes: list[str] = Field(default_factory=list, max_length=50)
    cost_matrix: list[list[float]] = Field(default_factory=list, max_length=50)
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
        if self.action == "act":
            self.spec()  # DecisionSpec's own checks: something must be worth deciding
        if self.action == "classes":
            size = len(self.classes)
            if size < 2 or len(set(self.classes)) != size:
                raise ValueError("A cost matrix needs the classes it is written over, each once, in its order")
            if len(self.cost_matrix) != size or any(len(row) != size for row in self.cost_matrix):
                raise ValueError(f"The cost matrix must be {size} by {size}: a row per true class, a column per class chosen")
            if any(c < 0 for row in self.cost_matrix for c in row):
                raise ValueError("A cost matrix holds costs: none may be negative")
        if self.action == "price":
            if not self.price_column:
                raise ValueError("A price decision needs the column that holds the price offered")
            if self.side == "pay" and not (self.unit_value_column or (self.unit_value_factor and self.reference_column)):
                raise ValueError("Paying a price needs what an accepted offer is worth: a column, or a factor of the reference price")
        return self

    def columns(self) -> list[str]:
        return [c for c in (self.price_column, self.reference_column, self.unit_value_column, self.value_column, *self.test_arm_columns) if c]

    def spec(self):
        """The ``act`` decision as dclab_core's DecisionSpec v1 (34.1)."""
        from dclab_core.decide import DecisionSpec

        return DecisionSpec(action=self.what or "act", value_per_case=self.value_per_case or 0.0, cost_per_action=self.cost_per_action,
                            cost_false_positive=self.cost_false_positive, cost_false_negative=self.cost_false_negative,
                            value_column=self.value_column, capacity_share=self.capacity_share,
                            capacity_per_day=self.capacity_per_day, notes=self.notes)


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
    # 8: "at renewal", "at listing" are moments a person says and Home accepts (20.10 v2: at 12 they were a dead end on this page)
    prediction_moment: str = Field(min_length=8, max_length=2000, description="When the prediction is made and what is known then.")
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
    # package 30.12: where each label came from (a noisy source's rows are kept out of training and judging; the column is
    # never a feature), and how many cases a team checks a period (precision at K decides, the threshold is the K-th score
    # of an average period, recall is estimated only from the random audit)
    label_source_column: str | None = None
    confirmed_sources: list[str] = Field(default_factory=list, max_length=20)
    capacity_k: int | None = Field(default=None, ge=1, le=100000)
    capacity_period: Literal["day", "week", "month"] | None = None

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
        if self.task not in (kind.tasks or (kind.base_task,)):
            raise ValueError(f"The {kind.title.lower()} kind runs as {' or '.join(kind.tasks or (kind.base_task,))}, not {self.task}")
        self.kind_fields = kind.fields(self.kind_fields)
        kind.check_solution(self)  # 16.15: what the kind's machine needs (a forecast needs its time column)
        if self.metric and self.metric not in kind.metric_keys():
            raise ValueError(f"Metric {self.metric!r} does not fit a {self.task} task")
        if len(names) != len(self.forbidden):
            raise ValueError("A column is listed twice under forbidden")
        if self.decision is not None and self.decision.action != "classes" and self.task != "binary":
            raise ValueError("A review, price or act decision reads a yes/no outcome (a case, an accepted offer): the task must be binary; "
                             "several classes take a cost matrix (action classes)")
        if self.decision is not None and self.decision.action == "classes" and self.task == "regression":
            raise ValueError("A cost matrix chooses among classes: the task must be binary or multiclass")
        if self.decision is not None and kind.own_machine() is not None:  # the review of 34.4: its final stage draws no decision
            raise ValueError(f"The {kind.title.lower()} kind runs its own machine, which does not draw a decision on top of the model yet: "
                             "leave the decision out")
        if (self.decision is not None and self.decision.action == "classes" and self.task == "binary"
                and sorted(self.decision.classes) != ["0", "1"] and str(self.positive_label) not in self.decision.classes):
            raise ValueError("A binary cost matrix names the classes 0 and 1, or the outcome's own two values with the positive one among them")
        if self.decision is not None and self.target in self.decision.columns():
            raise ValueError("The decision's price or value column cannot be the target")
        if self.decision is not None and names & set(self.decision.columns()):
            raise ValueError("A column the decision reads cannot be forbidden: "
                             + ", ".join(sorted(names & set(self.decision.columns())))
                             + " (a price, its reference and the value of an offer are known when the offer is made)")
        if self.metric == "recall_at_budget" and not self.alert_budget:
            raise ValueError("Recall at the review budget needs the budget: say how many rows the team can review")
        if self.label_source_column and (self.label_source_column == self.target or self.label_source_column in names
                                         or self.label_source_column in self.identifiers or self.label_source_column in self.text_columns):
            raise ValueError("The label-source column says which rows a person confirmed: it is never the target, an identifier, text or "
                             "forbidden (it is kept out of the model on its own)")
        if self.capacity_k is not None:
            if self.task != "binary":
                raise ValueError("A team checking K cases a period screens a yes/no outcome: the task must be binary")
            if not self.time_column or not self.capacity_period:
                raise ValueError("K cases a period needs the period (a day, a week, a month) and the time column that dates each row")
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
                     *(self.decision.columns() if self.decision else []), *([self.label_source_column] if self.label_source_column else []),
                     *([self.spatial.lat_column, self.spatial.lon_column] if self.spatial else [])]:
            if name not in known:
                raise ValueError(f"Column {name!r} is not in the data")

    def resolved_metric(self, positive_rate: float | None = None) -> str:
        if self.metric:
            return self.metric
        return default_metric(self.task, positive_rate)


def label_source_candidates(X: pd.DataFrame) -> list[str]:
    """30.12: the label-source columns of these rows (``dclab_core.modifiers.label_source_columns``)."""
    from dclab_core import modifiers as M

    return M.label_source_columns(X)


def modifier_facts(X: pd.DataFrame, y: pd.Series, task: str, time_column: str | None, group_candidates: list[str],
                   positive_rate: float | None, positive_label: Any = None, label_source: str | None = None,
                   capacity_k: int | None = None, capacity_period: str | None = None) -> Any:
    """30.11: what the modifiers' rules read, from the rows read (``dclab_core.modifiers.facts_of``, 30.13: in the library,
    so a data card reads them too)."""
    from dclab_core import modifiers as M

    return M.facts_of(X, y, task, time_column, group_candidates, positive_rate, positive_label, label_source, capacity_k, capacity_period)


def default_metric(task: str, positive_rate: float | None = None) -> str:
    return kinds.get(task).default_metric(None, positive_rate)


def _family_kind(fp: Any, time_candidates: list[str], task: str, positive_rate: float | None) -> dict[str, Any]:
    """S9: a complete proposal sets the kind and its fields (the proposal's kind is what the Solution saves); an
    incomplete one is only suggested, with the fields the person must give. The task stays the target's own when the
    kind takes it (a text kind answers a class or a number too), else the kind's base task; the metric and its options
    are the kind's own, which its check accepts (the review of S9: a forecast proposed with RMSE was refused). A kind
    that reads no outcome has no target: the outcome named goes to its own field, never among the columns (S9 review)."""
    if not fp.complete:
        return {"kind_suggested": fp.as_dict()}
    from ..kinds.propose import NO_OUTCOME

    k = kinds.get(fp.kind)
    task = task if task in (k.tasks or (k.base_task,)) else k.base_task
    out = {"kind": fp.kind, "kind_title": k.title, "task": task, "kind_fields": fp.fields, "kind_proposed": fp.as_dict(),
           "metric": k.default_metric({"task": task}, positive_rate), "metric_options": list(k.metrics) or METRICS[task]}
    if fp.kind in NO_OUTCOME:
        out["target"] = None
    if fp.time_column:
        out["time_candidates"] = [fp.time_column] + [c for c in time_candidates if c != fp.time_column]
    return out


def card_context(project: dict[str, Any]) -> dict[str, Any]:
    """S9: what a project built from a draft brings to every proposal on it, the same for each caller (the project's
    Solution page, the intern and /mcp, the journeys): its card's family, the person's answers to the kind's questions,
    the related tables attached by name. A project with no draft brings none of them (the plain proposal)."""
    from ..kinds.propose import said_of

    built = project.get("draft") or {}
    return {"family": ((built.get("card") or {}).get("family") or {}).get("value"), "said": said_of(built.get("understanding") or {}),
            "tables": list(project.get("tables") or {})}


def _whole_numbers(series: pd.Series) -> bool:
    """Whether a numeric column holds whole numbers only (an id stored as numbers, read as floats when a cell is empty)."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    return bool(len(values)) and bool(np.all(np.mod(values.to_numpy(dtype=float), 1) == 0))


def propose(frame: pd.DataFrame, profile: dict[str, Any], target: str, task: str | None = None, pack: str | None = None,
            time_column: str | None = None, family: str | None = None, said: dict[str, Any] | None = None,
            tables: Any = None) -> dict[str, Any]:
    """What the agent suggests for the solution, with the signal behind each suggestion. The problem kind (16.1) follows
    from the task, the Domain pack and the text columns; its metrics are the options offered. ``time_column``: the
    column the person (or a studied sample) says orders the rows, read first among the time candidates. ``family``
    (S9): the problem card's family; when it means a kind whose fields the table shows (a forecast's series and dates),
    that kind is proposed with its fields, its assumptions and why; when a field only the person knows is missing, the
    kind is suggested beside the plain one, never applied (``kind_suggested``). ``said`` (S9.3): the person's answers
    to the kind's questions (``kinds.propose.said_of``); ``tables``: the related tables attached, by name, when known."""
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
    X_read, y_read = X, y  # 30.11: the modifiers count on every row read, not the sample (the review)
    if len(X) > 20000:
        X = X.sample(20000, random_state=42)
        y = y.loc[X.index]
    audit = tools._audit_frame(X, y, task="regression" if task == "regression" else "classification", blind=False)
    identifiers = [name for name, c in columns.items() if name != target and (c["id_like"] or (c["name_id_like"] and c["unique_ratio"] > 0.5))]
    group_candidates = [name for name, c in columns.items() if name != target and c["name_id_like"] and 0.005 < c["unique_ratio"] < 0.5
                        and (c["kind"] != "numeric" or _whole_numbers(frame[name]))]  # S8 (live): a biker_id of whole numbers names an entity
    structural = set(time_candidates) | set(profile.get("text_candidates") or []) | {target}
    flagged = {row["column"] for row in audit["flagged"]}
    groups_to_confirm: list[dict[str, str]] = []
    for key in core_keys(frame.loc[X.index], target):  # 2026-10-24: recurring keys the names did not show
        name = key["column"]
        if name in structural:
            continue  # a time or a text column is never a key here, whatever the card reads in its values
        if key["named"]:  # a key's word ends its name (customer_id, user_email): a group, as an id-like name is
            if name not in group_candidates:
                group_candidates.append(name)  # appended: a group the names found stays first, as every reader of [0] expects
            if name not in identifiers and len(identifiers) < IDENTIFIERS_MAX:
                identifiers.append(name)
            continue
        if name in group_candidates:  # already a group by its name: nothing to confirm; flagged, it is a key, never a leak
            if name in flagged and name not in identifiers and len(identifiers) < IDENTIFIERS_MAX:
                identifiers.append(name)
            continue
        if name in flagged and name not in identifiers and len(identifiers) < IDENTIFIERS_MAX:
            identifiers.append(name)  # it predicts well alone because its rows belong together: kept out, never called a leak
        groups_to_confirm.append({"column": name, "why": (
            "its values recur a few times each, as one customer's or one account's rows do"
            + ("; alone it separates the outcome, as a key whose rows share an outcome does" if name in flagged else "")
            + ". If each value is one customer, account or person, split by it (group_column); if it is a property such as "
              "a product or a job, keep it as a feature")})
    forbidden = []
    label_sources = label_source_candidates(X)  # 30.12: read with the modifiers below
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
    from ..kinds.propose import for_family

    by_family = for_family(family, frame, target, said, tables)  # S9: the kind the card understood, with the fields its table shows
    from dclab_core import modifiers as M  # 30.11: the modifiers the rows read show, each applied above its threshold with its reason

    try:  # a modifier never stops a proposal (the review of 30.11): without its facts, the proposal goes on without modifiers
        facts = modifier_facts(X_read, y_read, task, time_candidates[0] if time_candidates else None, group_candidates, positive_rate,
                               detected.get("positive_label"), label_source=label_sources[0] if label_sources else None)
        found = M.detect(facts)
    except Exception:  # noqa: BLE001
        facts, found = None, []
    applied = {d.name: d for d in found if d.applies}
    plan = _core_plan(X, y, target, task, time_candidates[0] if time_candidates else None, rows=X.index)
    source_column = label_sources[0] if "noisy_labels" in applied and label_sources else None
    if source_column:  # 30.12: the label source, applied, is kept out of the model on its own: never forbidden (it would be refused)
        forbidden = [f for f in forbidden if f["column"] != source_column]
    for name in label_sources:  # a label source not applied (its noisy labels only ask) is still never a feature: forbidden, with why
        if name != source_column and name not in {f["column"] for f in forbidden}:
            forbidden.append({"column": name, "reason": "it says how each label was written (a rule, a review): never a feature",
                              "risk_score": 1.0, "proof": ["label source (30.12)"]})
        for decision in plan or []:
            if decision.get("decision") in ("features.forbidden", "features.to_confirm") and isinstance(decision.get("choice"), list):
                decision["choice"] = [c for c in decision["choice"] if not (isinstance(c, dict) and c.get("column") == source_column)]
    honest = (facts.audit_rate if facts is not None and facts.audit_rate is not None else facts.confirmed_rate if facts is not None else None)
    metric_rate = honest if "noisy_labels" in applied and honest is not None else positive_rate  # the rate the rare test read
    return {
        "target": target, "task": task, "kind": kind.key, "kind_title": kind.title, "detected": detected,
        "positive_label": detected.get("positive_label"),
        "forbidden": forbidden, "identifiers": identifiers, "time_candidates": time_candidates,
        "group_candidates": group_candidates, "groups_to_confirm": groups_to_confirm, "text_columns": text_columns,
        # the metric proposed stays the task's own, as before 16.1 (the wizard saves it): a pack's metric applies when the
        # person leaves the metric empty
        "metric": default_metric(task, metric_rate), "metric_options": METRICS[task],
        "prediction_moment_hint": _moment_hint(task, time_candidates, forbidden),
        "audit_note": audit["note"],
        "rows_read": rows_read_note(len(read), len(present), time_candidates[0] if time_candidates else None, by_time),
        "all_columns": audit["all_columns"],
        "plan": plan,
        "modifiers": [d.model_dump() for d in found], "modifier_facts": facts.model_dump() if facts is not None else None,
        **({"group_column": applied["groups"].value} if "groups" in applied else {}),  # shown with its reason on the Solution form
        **(_family_kind(by_family, time_candidates, task, metric_rate) if by_family is not None else {}),
        **({"label_source_column": source_column} if source_column else {}),  # 30.12: shown with its reason, carried by every save
        **({"decided_by_models": decided} if decided else {}),
    }


MODEL_PROOF = "decision-model:"  # a forbidden column a routed decision model added (22.9), never the audit's


KEY_LEVELS = 10  # a key has more values than this: a yes or no, or three sizes, is a property, whatever its name says
IDENTIFIERS_MAX = 50  # what a saved solution holds (Solution.identifiers)


def core_keys(frame: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    """The recurring keys ``dclab_core``'s data card finds that the studio's names miss: a text column the card reads as an
    entity, with more than ``KEY_LEVELS`` values, whose name says it is a key or whose values recur a few times each (10 to
    50 % distinct: a customer's or an account's handful of rows; a category's values recur dozens of times, under 10 %).

    Why (2026-10-24): the studio offered a group only when its name looked like an identifier, so a plain ``account`` or
    ``borrower`` was never offered and, predicting well alone, was proposed as a leak (the proposal found the group in 10
    of 20 challenge check variants and called it a leak in 15; the card in 20 of 20, LRN-008). The card's entity is wider
    than a key, and the review measured the harm of taking all of it: a yes or no named ``PhoneService``, a three-level
    ``ring-number``, a numeric ``word_freq_email``, word categories (``brand``, ``state``: one even became the split's
    group), a date stored as text, then (second review) ``email_domain`` and ``phone_brand`` (a key's word inside the name)
    and, in small tables, ``product_name`` or ``job_title``, whose values recur a few times each exactly as an account's
    do: no rule on the table's shape tells them apart. So: a text entity whose name ends with a key's word is a key
    (``named``); one whose values recur a few times each is only offered, for the intern or the person to confirm
    (the owner left it to Claude's judgement, 2026-10-24: a wrong automatic group drops a feature and changes what the
    holdout measures, silently; an offer costs a choice).
    Read on the rows the proposal reads only."""
    from dclab_core.bands import name_tokens
    from dclab_core.datacard import KEY_TOKENS, card_of

    try:
        card = card_of({"table": frame}, name="table", target=target)
    except Exception:  # noqa: BLE001 - the card is a second opinion: without it the proposal stands as the rules make it
        return []
    out = []
    for c in card.tables[0].columns:
        if c.purpose != "entity" or c.name == target or c.dtype not in ("string", "categorical") or c.distinct <= KEY_LEVELS:
            continue
        tokens = name_tokens(c.name)
        named = bool(tokens) and tokens[-1] in KEY_TOKENS  # the key's word ends the name: customer_id, never email_domain
        if named or c.uniqueness == "from_10_to_50pct":
            out.append({"column": c.name, "named": named})
    return out


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


from dclab_core.modifiers import MIN_TIMED_ROWS, PROVISIONAL_SHARE, provisional_reading  # noqa: E402,F401  (30.13: in the library, a data card reads the same rows)


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
