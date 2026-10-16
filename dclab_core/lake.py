"""The training lake's record shapes (26.1, 27.1): one store every DCLab model trains from, organised and checked.

Three layers. **Bronze** holds each source as it was captured; **silver** the same rows normalised, deduplicated and
redacted, each with its provenance; **gold** the training-ready sets, one shape per task: ``DecisionSample`` (the
decision plan of 2026-10-13: one answer about one subject at one decision point) and ``TrainingExample`` (one example of
one task family for the ML-solver model). Each gold set carries a ``GoldManifest``: its counts by split, origin and
family, the hash of every input, and what was dropped and why.

Rules every gold set passes (``gold_problems``):

- a group (a dataset, a project, a draft, a replica seed) is in exactly one split, so a model is never scored on the
  table it learned;
- a row of a sealed partition is only ever in the ``sealed`` split, read once to score a frozen model;
- an answer from a teacher model that may not be trained on (Claude's, the owner's decision of 2026-10-16) is only ever
  in ``eval_only``: compared against, never a target;
- no row carries a holdout's values or a cell of a table; inputs are bands, names and words.

The library imports only pydantic; the builders that read workspaces are the studio's (``dclab_rnd/lake``).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional

from pydantic import Field, JsonValue

from .contracts import CONTRACTS, Contract, Versioned, _version

# Where a label comes from. The decision plan's six sources, then what Phase 5 adds.
Origin = Literal[
    "construction",        # known by how the table was made (invented tables, planted columns)
    "replica",             # a replica's declared trap: true by construction
    "factory_rule",        # the rule's answer on computed evidence (the sample factory): learns the rule, mistakes included
    "rule_prod",           # the rule's answer in production: a weak label, never a test truth
    "human_exposed",       # a person's decision after seeing an AI answer: monitors or demotes, never promotes
    "human_blind",         # a person's label without seeing any AI answer: the truth promotion needs
    "measured_outcome",    # what a run measured (a holdout read once, a build that passed)
    "public_catalog",      # a public dataset's own metadata (OpenML, UCI): what each column and task is
    "verified_synthetic",  # an invented problem solved and verified (27.3): traps caught, holdout reproduced
    "research_record",     # an R&D result with its citations (evidence campaigns, research tracks)
    "dev_record",          # how DCLab itself was built and fixed (dev trajectories)
    "teacher_claude",      # Claude's answer: evaluation only, never a training target (owner, 2026-10-16)
]
Split = Literal["train", "dev", "sealed", "eval_only"]
Consent = Literal["owner", "train", "improve", "public", "ours"]  # ours: invented or R&D-made; owner: the owner's own records

# Origins that may be a training target. teacher_claude may not, until the owner's legal check says otherwise.
TRAINABLE: frozenset[str] = frozenset({"construction", "replica", "factory_rule", "rule_prod", "human_exposed", "human_blind",
                                        "measured_outcome", "public_catalog", "verified_synthetic", "research_record"})
# dev_record: DCLab's build history was written in Claude Code sessions, so it is a Claude output too; evaluation only until
# the owner decides (asked 2026-10-16), like every other Claude answer
EVAL_ONLY: frozenset[str] = frozenset({"teacher_claude", "dev_record"})

# The ML-solver model's task families (27.1): what it must learn to do, each with its own examples and measurement.
FAMILIES: tuple[str, ...] = (
    "understand_problem", "ask_question", "read_data", "eda_plan", "leakage_judgment", "split_design", "problem_type",
    "algorithm_choice", "feature_plan", "evaluation_design", "pipeline_code", "read_results", "next_experiment", "report",
    "agent_step", "typed_decision", "evidence_answer", "engineering", "solution_design",
)
Family = Literal[
    "understand_problem", "ask_question", "read_data", "eda_plan", "leakage_judgment", "split_design", "problem_type",
    "algorithm_choice", "feature_plan", "evaluation_design", "pipeline_code", "read_results", "next_experiment", "report",
    "agent_step", "typed_decision", "evidence_answer", "engineering", "solution_design",
]

SALT = "dclab-lake-v1"
SPLITS = (("sealed", 0.15), ("dev", 0.30))  # a group's hash below 0.15: sealed; below 0.30: dev; the rest train
HOLDOUT_MARKERS = ("holdout_rows", "holdout_values", "y_holdout", "X_holdout")  # never a key of a gold row


def split_of(group: str, salt: str = SALT) -> str:
    """A whole group's split, from its hash: the same group always lands in the same split."""
    h = int(hashlib.sha256(f"{salt}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return next((name for name, bound in SPLITS if h < bound), "train")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


class Provenance(Contract):
    """Where a row came from and what it may be used for."""

    source: str = Field(max_length=200, description="The set or record store it was read from.")
    origin: Origin
    record_ids: list[str] = Field(default_factory=list, max_length=20, description="Back-links: decision record, episode, run, package.")
    consent: Consent
    licence: Optional[str] = Field(None, max_length=120, description="A public dataset's licence; None for our own records.")
    group: str = Field(max_length=200, description="What must never be in two splits: a dataset, a project, a draft, a seed.")
    split: Split
    from_sealed: bool = Field(False, description="Read from a source's sealed partition: it may only ever be in the sealed split.")


class DecisionSample(Versioned):
    """One answer about one subject at one decision point (the decision plan's sample record)."""

    schema_version: Literal["dclab_core.decision_sample.v1"] = _version("decision_sample")
    sample_id: str = Field(max_length=80)
    decision_point: str = Field(max_length=80, description="The registry key, e.g. feature.leakage_suspect.")
    subject_kind: Literal["column", "dataset", "problem", "step"]
    subject_ref: str = Field(max_length=200, description="The masked column name, or the dataset or problem reference.")
    question: Optional[str] = Field(None, max_length=400)
    features: dict[str, JsonValue] = Field(default_factory=dict, description="The typed inputs as bands and tokens, never raw values.")
    feature_schema_version: str = Field("v1", max_length=40)
    state_text: str = Field(max_length=16000, description="The same inputs as the text a language model reads (a whole table's column list for a table-level point).")
    options: list[str] = Field(default_factory=list, max_length=64)
    label: JsonValue = Field(description="The answer, from the point's allowlist.")
    label_confidence: float = Field(1.0, ge=0, le=1)
    rule_answer: JsonValue = None
    other_answers: dict[str, JsonValue] = Field(default_factory=dict, description="Other answerers' answers (a model, Jev), never the label.")
    evidence_partition: Literal["metadata", "train"] = "metadata"
    provenance: Provenance


class Verification(Contract):
    kind: Literal["construction", "reproduced", "executed", "cited", "label_match", "person", "none"]
    passed: bool
    detail: Optional[str] = Field(None, max_length=400)


class TrainingExample(Versioned):
    """One example of one task family for the ML-solver model: what it sees, what it should produce, why it is right."""

    schema_version: Literal["dclab_core.training_example.v1"] = _version("training_example")
    example_id: str = Field(max_length=80)
    family: Family
    messages: list[dict[str, str]] = Field(min_length=2, max_length=60,
                                           description="The conversation as a served model gets it (system, user, assistant, tool), the last turn the target.")
    rationale: Optional[str] = Field(None, max_length=4000, description="Why the target is right, when the source says so.")
    verification: Verification
    quality: float = Field(0.5, ge=0, le=1, description="0..1 from the cleaning pipeline (27.5): verified, cited, agreement, length.")
    provenance: Provenance


class GoldManifest(Versioned):
    """A gold set: what it holds, by split, origin and family; every input's hash; what was dropped and why."""

    schema_version: Literal["dclab_core.gold_manifest.v1"] = _version("gold_manifest")
    name: str = Field(max_length=120)
    version: str = Field(max_length=40)
    kind: Literal["decision_sample", "training_example", "eval_record"]
    rows: int = Field(ge=0)
    by_split: dict[str, int] = Field(default_factory=dict)
    by_origin: dict[str, int] = Field(default_factory=dict)
    by_family: dict[str, int] = Field(default_factory=dict, description="Families for training examples; decision points for samples.")
    groups: int = Field(0, ge=0)
    inputs: list[dict[str, str]] = Field(default_factory=list, max_length=500, description="[{path, sha256}] of every file read.")
    dropped: dict[str, int] = Field(default_factory=dict, description="Rows not kept, by reason.")
    built_at: str = Field(max_length=40)
    builder_commit: Optional[str] = Field(None, max_length=60)
    rule: str = Field("A set for training and measuring DCLab's own models; scores on it are evidence about those models on these "
                      "sources, never a claim of production readiness.", max_length=600)


def _keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for k, v in value.items():
            yield str(k)
            yield from _keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from _keys(v)


def gold_problems(rows: Iterable[DecisionSample | TrainingExample]) -> list[str]:
    """Everything that would make a gold set unsafe to train or score on; empty when it is clean."""
    problems: list[str] = []
    splits_of: dict[str, set[str]] = {}
    ids: Counter[str] = Counter()
    for r in rows:
        p = r.provenance
        rid = getattr(r, "sample_id", None) or getattr(r, "example_id", "")
        ids[rid] += 1
        splits_of.setdefault(p.group, set()).add(p.split)
        if p.origin in EVAL_ONLY and p.split != "eval_only":
            problems.append(f"{rid}: a {p.origin} answer outside eval_only (it may not be a training target)")
        if p.split == "eval_only" and p.origin not in EVAL_ONLY:
            problems.append(f"{rid}: eval_only holds only evaluation-only origins, not {p.origin}")
        if p.from_sealed and p.split != "sealed":
            problems.append(f"{rid}: a row of a sealed partition outside the sealed split")
        payload = r.model_dump(mode="json")
        bad = sorted({k for k in _keys(payload) if k in HOLDOUT_MARKERS})
        if bad:
            problems.append(f"{rid}: holdout keys {bad}")
    problems += [f"group {g} is in {sorted(s)}" for g, s in splits_of.items() if len(s - {"eval_only"}) > 1]
    problems += [f"{i} appears {n} times" for i, n in ids.items() if n > 1]
    return problems


def manifest_of(name: str, version: str, kind: str, rows: list[DecisionSample | TrainingExample], inputs: list[dict[str, str]],
                dropped: Mapping[str, int], built_at: str, commit: Optional[str] = None) -> GoldManifest:
    fam = (lambda r: r.decision_point) if kind == "decision_sample" else (lambda r: getattr(r, "family", "eval"))
    return GoldManifest(name=name, version=version, kind=kind, rows=len(rows),  # type: ignore[arg-type]
                        by_split=dict(Counter(r.provenance.split for r in rows)), by_origin=dict(Counter(r.provenance.origin for r in rows)),
                        by_family=dict(Counter(fam(r) for r in rows)), groups=len({r.provenance.group for r in rows}),
                        inputs=list(inputs)[:500], dropped=dict(dropped), built_at=built_at, builder_commit=commit)


CONTRACTS["decision_sample"] = DecisionSample
CONTRACTS["training_example"] = TrainingExample
CONTRACTS["gold_manifest"] = GoldManifest

__all__ = ["Origin", "Split", "Consent", "TRAINABLE", "EVAL_ONLY", "FAMILIES", "SPLITS", "split_of", "fingerprint", "Provenance",
           "DecisionSample", "Verification", "TrainingExample", "GoldManifest", "gold_problems", "manifest_of"]
