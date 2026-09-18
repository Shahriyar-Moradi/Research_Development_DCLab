"""Narrow proposal language: agents cannot supply executable code or file paths."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .catalog import DATASETS
from .projects import PROJECTS

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Feature(Strict):
    name: str = Field(pattern=r"^fe_[A-Za-z0-9_]{1,50}$")
    operation: Literal["ratio", "difference", "product", "signed_log", "missing_indicator"]
    inputs: list[str] = Field(min_length=1, max_length=2)
    rationale: str = Field(max_length=1500)
    @model_validator(mode="after")
    def arity(self):
        expected = 1 if self.operation in ("signed_log", "missing_indicator") else 2
        if len(self.inputs) != expected:
            raise ValueError("Wrong number of feature inputs")
        return self

class Parameters(Strict):
    # Null means the executor's documented baseline, not an unconstrained search.
    C: float | None = Field(default=None, ge=0.001, le=100)
    n_estimators: int | None = Field(default=None, ge=30, le=300)
    max_depth: int | None = Field(default=None, ge=2, le=20)
    min_samples_leaf: int | None = Field(default=None, ge=1, le=40)
    learning_rate: float | None = Field(default=None, ge=0.01, le=0.3)

class Experiment(Strict):
    dataset: str
    title: str = Field(min_length=3, max_length=180)
    hypothesis: str = Field(min_length=10, max_length=2000)
    model: Literal["dummy", "logistic_regression", "extra_trees", "random_forest", "hist_gradient_boosting", "lightgbm", "xgboost"]
    parameters: Parameters
    features: list[Feature] = Field(default_factory=list, max_length=6)
    drop_columns: list[str] = Field(default_factory=list, max_length=40)
    stress_columns: list[str] = Field(default_factory=list, max_length=3)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    reference_evidence_id: str | None = None
    expected_learning: str = Field(max_length=2000)
    @model_validator(mode="after")
    def dataset_known(self):
        if self.dataset not in DATASETS:
            raise ValueError("Dataset is not in the allowlist")
        if len({f.name for f in self.features}) != len(self.features):
            raise ValueError("Derived feature names must be unique")
        return self

class Agenda(Strict):
    goal_interpretation: str
    research_questions: list[str]
    sequence: list[str]
    success_criteria: list[str]
    limitations: list[str]

class DataReview(Strict):
    dataset_findings: list[str]
    leakage_risks: list[str]
    availability_assumptions: list[str]
    recommended_tests: list[str]
    external_data_hypotheses: list[str]

class Critique(Strict):
    observation: str
    interpretation: str
    limitations: list[str]
    next_question: str
    evidence_ids: list[str]
    continue_research: bool

class Lesson(Strict):
    claim: str
    evidence_ids: list[str]
    scope: str
    counterevidence: str
    follow_up: str

class Synthesis(Strict):
    summary: str
    lessons: list[Lesson]
    theoretical_principles: list[str]
    workflow_blocks: list[str]
    unanswered_questions: list[str]

DEFAULT_GOAL = "Learn which feature engineering, model families and tuning choices improve robust tabular prediction. Audit leakage and prediction-time availability, compare paired CV evidence, stress missing production features, and capture reusable workflows and negative results."

class RunRequest(Strict):
    project: Literal["general", "hyperack", "telco_churn"] = "general"
    goal: str = Field(default=DEFAULT_GOAL, min_length=20, max_length=6000)
    datasets: list[str] = Field(default_factory=lambda: ["bank_marketing"], min_length=1, max_length=10)
    model: str = Field(default="gpt-5.6-terra", pattern=r"^gpt-[a-zA-Z0-9.\-]+$")
    max_experiments: int = Field(default=4, ge=1, le=50)
    max_rows: int = Field(default=4000, ge=200, le=50000)
    repeats: int = Field(default=2, ge=1, le=3)
    max_minutes: int = Field(default=30, ge=2, le=240)
    @model_validator(mode="after")
    def check_datasets(self):
        if len(set(self.datasets)) != len(self.datasets) or any(d not in DATASETS for d in self.datasets):
            raise ValueError("Use distinct allowlisted datasets")
        if any(DATASETS[d]["project"] != self.project for d in self.datasets):
            raise ValueError("Every dataset must belong to the selected project")
        return self
