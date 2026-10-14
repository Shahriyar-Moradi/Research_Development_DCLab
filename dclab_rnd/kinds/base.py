"""The problem-kind interface (package 16.1).

A kind is one module that says everything DCLab must know to solve that kind of problem well: how it is recognised,
what Home asks, which extra solution fields it needs, how the target is read, how rows are split, which metrics and
baselines judge it, which recipes and model families it adds, which checks and reliability questions it brings, what
its export supports and which research track holds its evidence. The engine, ``Solution``, ``prepare_bundle``, the
runner and Home read the kind instead of branching on ``solution.task``.

Every kind runs on one of the engine's numeric machines (``task_type``: ``binary_imbalanced``, ``multiclass``,
``timeseries_regression``, ``text_tabular_binary``): the runner computes folds, fits and metrics for those. A kind
that needs a new machine (forecasting horizons, survival, ranking) brings it in its own package; the interface here
already carries its parts.

Everything here is a default; a kind overrides only what differs. Nothing in this module reads a project or a file.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict

BASE_TASKS = ("binary", "multiclass", "regression")


class NoFields(BaseModel):
    """A kind that needs no field beyond the solution's own."""

    model_config = ConfigDict(extra="forbid")


def _get(solution: Any, key: str) -> Any:
    return solution.get(key) if isinstance(solution, dict) else getattr(solution, key, None)


def default_split(solution: Any) -> str:
    """The solution decides the split, not a setting: a declared time column means the holdout is the latest period
    (DCLAB-R02); otherwise a declared group column keeps each group on one side; otherwise a stratified random split.
    Time and group together: forward in time and no entity on both sides (15.3)."""
    if _get(solution, "spatial"):  # 16.10: blocks of places are the groups
        return "group"
    if _get(solution, "time_column") and _get(solution, "group_column"):
        return "time_group"
    return "time" if _get(solution, "time_column") else "group" if _get(solution, "group_column") else "stratified"


class ProblemKind:
    """The parts of a kind (the table of Part 5). Subclass and override; register with :func:`dclab_rnd.kinds.register`."""

    key: ClassVar[str] = ""
    title: ClassVar[str] = ""
    description: ClassVar[str] = ""
    base_task: ClassVar[str] = "binary"            # Solution.task this kind runs as
    task_type: ClassVar[str] = "binary_imbalanced"  # the runner's machine
    specificity: ClassVar[int] = 0                  # detection order: a more specific kind is asked first
    packs: ClassVar[tuple[str, ...]] = ()           # the Domain packs that mean this kind
    words: ClassVar[str | None] = None              # a pattern of the problem's words that points here
    questions: ClassVar[tuple[str, ...]] = ()       # what Home asks, in the kind's words
    home_fields: ClassVar[tuple[str, ...] | None] = None  # the fields Home works through (draft.plan), None: the generic ones
    moments: ClassVar[tuple[str, ...] | None] = None      # Home's prediction-moment options, None: the generic ones
    metrics: ClassVar[tuple[tuple[str, str], ...]] = ()
    baselines: ClassVar[tuple[str, ...]] = ()       # what the final stage compares against (by name)
    checks: ClassVar[tuple[str, ...]] = ()          # pitfalls of this kind (dclab_rnd.pitfalls ids) for audit and copilot
    reliability: ClassVar[tuple[str, ...]] = ()     # what the final stage must challenge
    evidence_track: ClassVar[str | None] = None     # research/<track>/ whose claims agents may cite
    aliases: ClassVar[tuple[str, ...]] = ()           # the names this kind of problem goes by (16.14: in its evidence record)
    export: ClassVar[dict[str, Any]] = {"notebook": True, "service": True}
    stratify: ClassVar[bool] = True                 # class strata in random and grouped splits
    models_text: ClassVar[bool] = False             # free-text columns become features (a text kind)
    Fields: ClassVar[type[BaseModel]] = NoFields    # the typed kind_fields block of a Solution

    # ------------------------------------------------------------------ detection
    def detect(self, problem: str, profile: dict[str, Any] | None = None) -> str | None:
        """Why this kind fits, in words, or None. The default reads the kind's words in the problem."""
        if self.words:
            found = re.search(self.words, (problem or "").lower())
            if found:
                return f'the problem mentions "{found.group(0).strip()}"'
        return None

    # ------------------------------------------------------------------ solution
    def applies(self, solution: Any) -> bool:
        """Whether this kind fits a solution a pack points here (a text kind needs text columns)."""
        return _get(solution, "task") == self.base_task

    def fields(self, values: dict[str, Any] | None) -> dict[str, Any]:
        """The kind's fields, validated (raises ValueError on a field it does not know)."""
        return self.Fields(**(values or {})).model_dump()

    def metric_keys(self) -> set[str]:
        return {m for m, _ in self.metrics}

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return self.metrics[0][0]

    # ------------------------------------------------------------------ data assembly
    def target(self, frame: pd.DataFrame, solution: Any) -> tuple[pd.Series, list[str] | None]:
        """The target as the machine reads it, and class labels when there are classes."""
        raise NotImplementedError

    def positive_rate(self, y: pd.Series) -> float | None:
        return None

    def text_columns(self, solution: Any, X: pd.DataFrame) -> tuple[tuple[str, ...], dict[str, str]]:
        """(the text columns modelled, the ones excluded with why)."""
        forbidden = {_get(f, "column") for f in (_get(solution, "forbidden") or [])}
        # a forbidden column is never modelled, as text or otherwise (rule 1): it stays blocked
        declared = tuple(c for c in (_get(solution, "text_columns") or []) if c in X and c not in forbidden)
        if self.models_text:
            return declared, {}
        return (), {c: "free text is modeled for binary targets only; excluded here" for c in declared}

    def machine(self, solution: Any, text_columns: tuple[str, ...]) -> str:
        """The runner's machine for this solution (a binary kind with text runs the text machine)."""
        return self.task_type

    def spec_settings(self, solution: Any, pack: dict[str, Any] | None = None) -> dict[str, Any]:
        """Settings this kind adds to the dataset spec (the runner reads them)."""
        return {}

    # ------------------------------------------------------------------ split
    def split(self, solution: Any) -> str:
        return default_split(solution)

    def bundle(self, spec: Any, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> Any:
        """The locked holdout and training folds; the default is the shared one, with this kind's strata."""
        from dclab_rnd.expansion import datasets as ds

        return ds.prepare_bundle(spec, X, y, stratify=self.stratify, **kwargs)

    # ------------------------------------------------------------------ recipes and models
    def extra_recipes(self, spec: Any, X: pd.DataFrame, roles: dict[str, str]) -> dict[str, dict[str, Any]]:
        """Recipes this kind adds to the ladder beside the shared ones (each a runner recipe dict)."""
        return {}

    def model_families(self, available: list[str], rare: bool = False) -> list[str]:
        """The families the screen tries, from the machine's available ones (and the rare-event ones when ``rare``)."""
        from dclab_rnd.expansion import runner as rn

        return list(available) + (list(rn.RARE_FAMILIES) if rare else [])

    def rare_screen(self, task_type: str, positive_rate: float | None, solution: Any, settings: dict[str, Any]) -> bool:
        """Whether rules, detectors and PU bagging join the screen (15.5)."""
        return False

    # ------------------------------------------------------------------ the data stage's words
    def summarise_target(self, y: pd.Series, class_labels: list[str] | None, solution: Any) -> tuple[dict[str, Any], str]:
        """(the target summary of the data stage, a short phrase for its first claim)."""
        raise NotImplementedError

    def target_claim(self, summary: dict[str, Any], metric_label: str, spec: Any) -> tuple[str, str, list[str]]:
        """(claim kind, statement, evidence paths) of the data stage's third claim."""
        raise NotImplementedError

    # ------------------------------------------------------------------ acceptance (16.2)
    def case(self, seed: int = 0) -> Any:
        """The kind's replica as a person brings it (a ``kinds.acceptance.Case``), or None: then the kind is partial."""
        return None

    runs_in_engine = True  # False: the kind runs only as a library (its own ``solve``); the Solution refuses it, the matrix says partial
    needs_target = True  # 20.7: a kind that reads no outcome column (anomaly detection without labels) says False
    # decisions known to fail until the owner acts (weights to download, a purpose to open): still listed as missing, but
    # `make kinds-check` fails only on a decision outside them
    open_decisions: tuple[str, ...] = ()

    def own_machine(self) -> Any:
        """A kind whose problem the five tabular stages' code cannot run returns its machine class
        (``studio.machine.Machine``); the engine then runs it under the same graph, gates and records (16.15)."""
        return None

    def check_solution(self, solution: Any) -> None:
        """Raise ValueError when a solution declaring this kind lacks what its machine needs (called by Solution)."""
        return None

    def accept_engine(self, run: Any) -> list[dict[str, Any]]:
        """For a kind with its own machine: the decisions checked on the engine's records of the same case (16.15)."""
        return []

    def tables_of(self, solution: Any) -> list[str]:
        """The project's related tables the kind's machine reads (16.15 C), by name; none by default."""
        return []

    def table_columns_of(self, solution: Any) -> dict[str, list[str]]:
        """The columns each related table must hold, by table name."""
        return {}

    def columns_of(self, solution: Any) -> list[str]:
        """The columns the kind's own fields name (checked against the table before a machine runs)."""
        return []

    def solve(self, case: Any) -> Any:
        """A kind with its own machine (forecasting, survival, ranking…) solves its case here and returns what ``accept``
        reads; None (the default) means the five-stage engine solves it."""
        return None

    def accept(self, run: Any) -> list[dict[str, Any]]:
        """The decisions a careful data scientist reaches on the replica, each checked on the finished run
        (``kinds.acceptance.check``)."""
        return []

    # ------------------------------------------------------------------ description
    def describe(self) -> dict[str, Any]:
        return {"key": self.key, "title": self.title, "description": self.description, "base_task": self.base_task,
                "task_type": self.task_type, "packs": list(self.packs), "questions": list(self.questions),
                "home_fields": list(self.home_fields) if self.home_fields else None,
                "fields": self.Fields.model_json_schema().get("properties", {}),
                "metrics": [{"key": k, "label": label} for k, label in self.metrics], "baselines": list(self.baselines),
                "checks": list(self.checks), "reliability": list(self.reliability), "evidence_track": self.evidence_track,
                "export": dict(self.export)}
