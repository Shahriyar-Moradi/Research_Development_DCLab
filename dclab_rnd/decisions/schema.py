"""The decision sample: one row per typed decision the system made, and one label row per thing a person did to it.

A *sample* is written where a decision is computed (the solution proposal, the problem brief, each engine stage): the
decision point, the subject (a column, the table or the problem), the evidence the rule saw as aggregates and names,
the rule's answer and the rule's version, and ``state_text``: the same evidence as one line of text, what a model
would read to answer the same question. A *label* is written where a person acts on it: approving the stage (the
rule's choice stands), choosing another option, saving a solution that changes the proposal, approving a gate.

Never a cell value and never a person's words. Evidence is the numbers the rules already compute on the training rows
(rates, ratios, counts, fold scores) and column names; ``CELL_KEYS`` (previews, row values, dates of a row, class
values, quotes) are dropped from any evidence dict at any depth, and a string value longer than ``TEXT`` is cut. The
text a person typed (a problem description, a reason, a prediction moment) never enters a row: a brief sample keeps
whether a field was quoted and how many words, not the words; a label's reason is one of ``REASONS``. The holdout's
labels are never read: every sample comes from the training partition or from the table's shape (the one holdout
statistic the engine itself reads, a feature-distribution PSI, is dropped too, so ``train`` means train).

Rows are written for every project, opted in or not: they stay in the workspace, like its audit, and leave it only by
the owner's export. The training opt-in (``settings.share_for_training``) governs what leaves, not what is kept.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from datetime import datetime, timezone
from typing import Any

TEXT = 300
CELL_KEYS = frozenset({"examples", "preview", "top_values", "values", "head", "sample_rows", "rows_preview",
                       "start", "end", "time_range", "classes_only_in_last_third", "quote", "statement"})  # the last five: dates, class values and words of a row or a person
SPLIT_KINDS = {"time_and_group": "time_group", "spatial_block": "spatial", "time_group": "time_group", "spatial": "spatial", "stratified": "stratified",
               "random": "random", "group": "group", "time": "time"}
REASONS = ("the rule's choice, approved", "chosen by the person", "stage approved as computed", "cleared by the person", "excluded by the person",
           "first save of the solution", "gate approved")  # the only words a label carries: never what a person typed

# field order of a sample row (files and the decision_sample table share it)
SAMPLE_FIELDS = ("sample_id", "at", "point", "subject_kind", "subject", "key", "project_id", "draft_id", "dataset_ref",
                 "stage", "run_ref", "features", "state_text", "rule_answer", "rule_version", "alternatives", "evidence_partition",
                 "actor", "user_id", "schema_version")
LABEL_FIELDS = ("label_id", "at", "sample_id", "key", "project_id", "label", "source", "who", "user_id", "reason", "move", "schema_version")
SCHEMA_VERSION = 1

LABEL_SOURCES = ("human_exposed", "human_blind", "construction", "replica", "factory_rule", "rule_prod")
SUBJECTS = ("column", "table", "problem")
PARTITIONS = ("metadata", "train")

# Every decision point a sample may carry: its subject and the answers a label may take (None: an open answer, such
# as a family name or a metric, checked against the sample's own alternatives when it has any).
POINTS: dict[str, dict[str, Any]] = {
    # the problem brief (WF-01): one sample per contract field, the rule's status and value
    "contract.task": {"subject": "problem", "answers": ("binary", "multiclass", "regression", "unknown")},
    "contract.target_column": {"subject": "problem", "answers": None},
    "contract.prediction_moment": {"subject": "problem", "answers": None},
    "contract.time_column": {"subject": "problem", "answers": None},
    "contract.label_coverage": {"subject": "problem", "answers": ("complete", "partial", "unknown")},
    # the solution proposal (WF-01/WF-04): dclab_core's findings and plan on the provisional training rows
    "column.forbidden": {"subject": "column", "answers": ("exclude", "confirm", "clear")},
    "column.identifier": {"subject": "column", "answers": ("identifier", "feature")},
    "column.missing_action": {"subject": "column", "answers": ("indicator_and_check", "impute_within_folds")},
    "split.strategy": {"subject": "table", "answers": ("random", "time", "group", "time_group", "spatial", "stratified")},
    "data.cleaning": {"subject": "table", "answers": None},
    "task.framing": {"subject": "problem", "answers": None},
    "labels.coverage": {"subject": "problem", "answers": None},
    "metric.primary": {"subject": "problem", "answers": None},
    "baseline": {"subject": "problem", "answers": None},
    "features.direction": {"subject": "problem", "answers": None},
    "model.families": {"subject": "table", "answers": None},
    "threshold.policy": {"subject": "problem", "answers": None},
    # the engine stages (WF-03 to WF-09)
    "column.role": {"subject": "column", "answers": ("blocked", "identifier", "time", "group", "text", "feature")},
    "column.risk": {"subject": "column", "answers": None},  # a list of risk flags, or "none"
    "column.leakage": {"subject": "column", "answers": ("exclude", "review_flag", "clear")},
    "feature.recipe": {"subject": "table", "answers": None},
    "model.family": {"subject": "table", "answers": None},
    "optimization.config": {"subject": "table", "answers": None},
    "holdout.verdict": {"subject": "table", "answers": ("accept_result", "investigate_distribution_shift", "investigate_gap_without_shift")},
    "calibration.action": {"subject": "table", "answers": ("keep_uncalibrated", "calibrate", "unknown")},
}

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_id(prefix: str) -> str:
    """A sortable id: a 10-character time part (milliseconds, Crockford base 32) and 10 random characters."""
    stamp, chars = int(time.time() * 1000), []
    for _ in range(10):
        stamp, rest = divmod(stamp, 32)
        chars.append(_ALPHABET[rest])
    return prefix + "".join(reversed(chars)) + "".join(secrets.choice(_ALPHABET) for _ in range(10))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dataset_ref(sha256: str | None, filename: str | None = None) -> str | None:
    """A stable, anonymous name for a table: the first 16 hex characters of its content digest (never the file name)."""
    if sha256:
        return str(sha256)[:16]
    return hashlib.sha256(str(filename).encode()).hexdigest()[:16] if filename else None


def scrub(value: Any) -> Any:
    """The value without the keys that carry cells from the table, strings cut to TEXT, floats rounded."""
    if isinstance(value, dict):
        return {str(k): scrub(v) for k, v in value.items() if k not in CELL_KEYS}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value][:64]
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        return value if len(value) <= TEXT else value[:TEXT - 1] + "…"
    if isinstance(value, (int, bool)) or value is None:
        return value
    return scrub(str(value))


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(scrub(value), ensure_ascii=False, default=str))


def split_kind(value: Any) -> str | None:
    """One name for a split strategy, whichever module named it (dclab_core, the engine, the solution)."""
    if isinstance(value, dict):
        value = value.get("kind")
    return SPLIT_KINDS.get(str(value)) if value is not None else None


def key_of(point: str, subject: str | None) -> str:
    """What a label attaches to: the point and its subject (the latest sample with the key in the project)."""
    return f"{point}:{subject or '-'}"


def state_text(point: str, subject: str | None, features: dict[str, Any]) -> str:
    """The evidence as one line a model would read: ``point column=x | a=1 b=0.2 …``; lists and dicts as JSON."""
    parts = []
    for k, v in features.items():
        if isinstance(v, float):
            text = f"{v:.4g}"
        elif isinstance(v, (dict, list)):
            text = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        else:
            text = str(v)
        parts.append(f"{k}={text}")
    head = f"{point} {POINTS.get(point, {}).get('subject', 'problem')}={subject}" if subject else point
    line = f"{head} | " + " ".join(parts)
    return line if len(line) <= 2000 else line[:1999] + "…"


def sample(point: str, subject: str | None, features: dict[str, Any], rule_answer: Any, rule_version: str, *,
           project_id: str | None = None, draft_id: str | None = None, dataset: str | None = None, stage: str | None = None,
           run_ref: str | None = None, alternatives: list[Any] | None = None, partition: str = "train", actor: str = "rule",
           user_id: str | None = None, at: str | None = None) -> dict[str, Any]:
    """One sample row. ``features`` are aggregates and names (scrubbed here); ``rule_answer`` is the rule's choice."""
    if point not in POINTS:
        raise ValueError(f"unknown decision point {point!r}")
    if partition not in PARTITIONS:
        raise ValueError(f"partition {partition!r}: never the holdout")
    feats = jsonable(features)
    return {"sample_id": new_id("ds_"), "at": at or now(), "point": point, "subject_kind": POINTS[point]["subject"],
            "subject": None if subject is None else str(subject)[:TEXT], "key": key_of(point, subject),
            "project_id": project_id, "draft_id": draft_id, "dataset_ref": dataset, "stage": stage, "run_ref": run_ref,
            "features": feats, "state_text": state_text(point, subject, feats), "rule_answer": jsonable(rule_answer),
            "rule_version": str(rule_version)[:120], "alternatives": jsonable(alternatives or []), "evidence_partition": partition,
            "actor": actor, "user_id": user_id, "schema_version": SCHEMA_VERSION}


def label(sample_row: dict[str, Any] | None, key: str, project_id: str | None, value: Any, source: str, who: str, *,
          user_id: str | None = None, reason: str = "", move: str | None = None, at: str | None = None) -> dict[str, Any]:
    """One label row: what a person decided about a sample (by its id when known, always by its key)."""
    if source not in LABEL_SOURCES:
        raise ValueError(f"unknown label source {source!r}")
    if reason and reason not in REASONS:
        raise ValueError("a label's reason is one of REASONS, never a person's words")
    return {"label_id": new_id("dl_"), "at": at or now(), "sample_id": (sample_row or {}).get("sample_id"), "key": key,
            "project_id": project_id, "label": jsonable(value), "source": source, "who": str(who)[:80], "user_id": user_id,
            "reason": reason or "", "move": move, "schema_version": SCHEMA_VERSION}
