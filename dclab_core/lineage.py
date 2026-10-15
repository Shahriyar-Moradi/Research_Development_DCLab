"""Lineage of every processing step (package 19.5): one run record per stage run, pipeline step and agent run.

A :class:`RunRecord` says what a run read (the table's hash and size, how rows were sampled, the solution's hash, the
settings, the choices of earlier runs it used, the seed), what it was told (its parameters), what it made (its metric
and value with an interval when there is one, the option it chose, and its numbers by path), how long it took, and
the code that ran it (the commit, this library's version, the prompts' and tools' hashes). Two runs with the same
:attr:`RunRecord.fingerprint` read the same inputs with the same parameters, so a rerun from a record can be checked:
:func:`compare` says whether the inputs, the code and the number are the same.

A number in a solution report is traced to a record path (``final.evidence.holdout_metrics.roc_auc``, see
``dclab_core.report``); its first part names the stage, and :func:`run_for_path` gives the run that wrote it.

Each host keeps its records where it keeps its stage results; the session record (``dclab_core.experience``) carries a
``run.recorded`` event with the flat parts (:func:`event_payload`), so a run is found from a session and a session
from a run. DCLab's main product maps its ``agent_runs`` and ``ml_run_events`` rows with
``dclab_core.adapters.main_product_lineage``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Literal, Optional, Union

from pydantic import Field, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

Scalar = Union[str, int, float, bool, None]
RunKind = Literal["stage", "pipeline", "agent", "job"]
RunStatus = Literal["completed", "failed", "stopped"]
MAX_NUMBERS = 400   # numbers kept per run: the record's numeric leaves, the first ones in reading order
MAX_TEXT = 300


StepKind = Literal["load", "clean", "impute", "encode", "transform", "fold", "fit", "predict", "metric", "select", "other"]
MAX_STEPS = 600  # a run keeps at most this many; the rest are counted, not kept
SAMPLE_ROWS, SAMPLE_COLS = 5000, 200  # a digest reads at most this much of a matrix, and its shape


class Step(Contract):
    """One action of a run (package 23.8, ``dclab_core.steps``): what it was, with what, on what shape, and a digest of what it made."""

    index: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=120)
    kind: StepKind
    params: dict[str, Scalar] = Field(default_factory=dict, max_length=16)
    rows_in: Optional[int] = Field(default=None, ge=0)
    cols_in: Optional[int] = Field(default=None, ge=0)
    rows_out: Optional[int] = Field(default=None, ge=0)
    cols_out: Optional[int] = Field(default=None, ge=0)
    fold: Optional[int] = Field(default=None, ge=0)
    metric: Optional[str] = Field(default=None, max_length=80)
    value: Optional[float] = None
    seconds: float = Field(default=0.0, ge=0)
    digest: Optional[str] = Field(default=None, max_length=16, description="A hash of what the step produced (``steps.digest_of``); equal for a faithful rerun.")


class RunInputs(Contract):
    """What a run read. A field it did not read is None or empty, never guessed."""

    data_sha256: Optional[str] = Field(default=None, max_length=64, description="The table's SHA-256 as recorded when it was attached.")
    data_rows: Optional[int] = Field(default=None, ge=0)
    data_columns: Optional[int] = Field(default=None, ge=0)
    sampling: dict[str, Scalar] = Field(default_factory=dict, description="How rows were chosen: a rule and its numbers.")
    solution_hash: Optional[str] = Field(default=None, max_length=64, description="The solution's hash (the studio's graph.solution_hash).")
    settings: dict[str, Scalar] = Field(default_factory=dict)
    decisions: dict[str, Scalar] = Field(default_factory=dict, description="The choices of earlier runs this run used, by step.")
    refs: dict[str, str] = Field(default_factory=dict, description="Ids and hashes of what it read that no field above names: "
                                 "a dataset version, a split plan, an agent's context or tool catalog digest.")
    upstream: list[str] = Field(default_factory=list, max_length=50, description="The runs whose outputs this run read.")
    seed: Optional[int] = None


class RunOutputs(Contract):
    metric: Optional[str] = Field(default=None, max_length=80)
    value: Optional[float] = None
    low: Optional[float] = Field(default=None, description="The interval's lower end, when the run measured one.")
    high: Optional[float] = None
    chosen: Optional[str] = Field(default=None, max_length=MAX_TEXT)
    numbers: dict[str, float] = Field(default_factory=dict, description="Every number the run wrote, by its path in the record.")


class RunCode(Contract):
    commit: Optional[str] = Field(default=None, max_length=64, description="The code's commit; None outside a repository.")
    dirty: Optional[bool] = Field(default=None, description="True when tracked files differed from the commit.")
    diff_sha256: Optional[str] = Field(default=None, max_length=64, description="The hash of that difference: two dirty runs "
                                       "ran the same code only when it is equal.")
    core: Optional[str] = Field(default=None, max_length=40, description="dclab_core's version.")
    runner: Optional[str] = Field(default=None, max_length=120, description="What ran it: a module, an agent and its version.")
    prompts: dict[str, str] = Field(default_factory=dict, description="Short hashes of the prompts and tool schemas a model was given.")


class RunRecord(Versioned):
    """One run: what it read, what it was told, what it made, and the code that ran it."""

    schema_version: Literal["dclab_core.run_record.v2"] = _version("run_record", 2)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.run_record.v1"})  # v2 added steps; a stored v1 record stays readable
    run: str = Field(min_length=1, max_length=80)
    kind: RunKind
    name: str = Field(min_length=1, max_length=120, description="The stage, the pipeline step or the agent.")
    workspace: Optional[str] = Field(default=None, max_length=64)
    project: Optional[str] = Field(default=None, max_length=64)
    draft: Optional[str] = Field(default=None, max_length=64)
    job: Optional[str] = Field(default=None, max_length=80)
    parent: Optional[str] = Field(default=None, max_length=80, description="The run this one was part of.")
    status: RunStatus
    started: str
    finished: Optional[str] = None
    seconds: Optional[float] = Field(default=None, ge=0)
    inputs: RunInputs = Field(default_factory=RunInputs)
    params: dict[str, Scalar] = Field(default_factory=dict)
    outputs: RunOutputs = Field(default_factory=RunOutputs)
    code: RunCode = Field(default_factory=RunCode)
    error: Optional[str] = Field(default=None, max_length=600, description="Why a failed run failed (its type and message).")
    fingerprint: str = Field(default="", max_length=64, description="The hash of the inputs and parameters (:func:`fingerprint`).")
    steps: list[Step] = Field(default_factory=list, max_length=MAX_STEPS, description="v2 (23.8): what the run did, one entry per cleaning "
                              "action, transform, fold, fit, prediction and metric, each with a digest of what it produced.")
    steps_dropped: int = Field(default=0, ge=0, description="v2: steps taken beyond MAX_STEPS and not kept.")

    @model_validator(mode="before")
    @classmethod
    def _fingerprinted(cls, data: Any) -> Any:
        """The fingerprint is computed here; one given must match it (a record is never edited after its run)."""
        if not isinstance(data, Mapping):
            return data
        if data.get("schema_version") == "dclab_core.run_record.v1":
            data = {**data, "schema_version": "dclab_core.run_record.v2"}
        raw = data.get("inputs") or {}
        inputs = raw if isinstance(raw, RunInputs) else RunInputs.model_validate(raw)
        expected = fingerprint(inputs, data.get("params") or {})
        if data.get("fingerprint") and data["fingerprint"] != expected:
            raise ValueError("fingerprint does not match the inputs and parameters: a record is never edited after its run")
        return {**data, "fingerprint": expected}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def fingerprint(inputs: RunInputs | Mapping[str, Any], params: Mapping[str, Any]) -> str:
    """The hash of what a run read and what it was told (not when, not by which code): equal for a faithful rerun.
    Fields at their default are left out, so a field added to :class:`RunInputs` later leaves stored records valid."""
    data = (inputs if isinstance(inputs, RunInputs) else RunInputs.model_validate(inputs)).model_dump(exclude_defaults=True)
    data.pop("upstream", None)  # run ids differ between a run and its rerun; what they chose is in ``decisions``
    return hashlib.sha256(_canonical({"inputs": data, "params": dict(params)}).encode()).hexdigest()[:32]


def numbers(record: Any, prefix: str = "", limit: int = MAX_NUMBERS) -> dict[str, float]:
    """The numeric leaves of a record by path (``evidence.holdout_metrics.roc_auc``), in reading order, at most ``limit``;
    booleans and values that are not finite are left out."""
    out: dict[str, float] = {}

    def walk(node: Any, path: str) -> None:
        if len(out) >= limit:
            return
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{path}.{index}" if path else str(index))
        elif isinstance(node, (int, float)) and not isinstance(node, bool) and math.isfinite(node):
            out[path] = float(node)

    walk(record, prefix)
    return out


def flat(mapping: Mapping[str, Any], prefix: str = "", limit: int = 100) -> dict[str, Scalar]:
    """A mapping's scalars by dotted key, nested mappings flattened, lists and long text left out (settings, params)."""
    out: dict[str, Scalar] = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if len(out) >= limit:
            break
        if isinstance(value, Mapping):
            out.update(flat(value, name + ".", limit - len(out)))
        elif isinstance(value, str):
            if len(value) <= MAX_TEXT:
                out[name] = value
        elif value is None or isinstance(value, (bool, int)) or (isinstance(value, float) and math.isfinite(value)):
            out[name] = value
    return out


def run_for_path(runs: Mapping[str, RunRecord], path: str) -> Optional[RunRecord]:
    """The run that wrote the number at a report path: the path's first part names the stage (``final.evidence…``)."""
    head = path.split(".", 1)[0]
    return runs.get(head)


TIMING = re.compile(r"(seconds|elapsed|_ms|duration)", re.I)  # a number that measures the run, not its result


def compare(original: RunRecord, rerun: RunRecord, tolerance: float = 1e-9) -> dict[str, Any]:
    """Whether a rerun read the same inputs with the same parameters, ran the same code, and made the same numbers:
    its value and every number it wrote (timings left out), each within ``tolerance``."""
    a, b = original.inputs.model_dump(), rerun.inputs.model_dump()
    a.pop("upstream"), b.pop("upstream")
    changed_inputs = sorted(k for k in a if a[k] != b[k])
    changed_params = sorted(k for k in set(original.params) | set(rerun.params) if original.params.get(k) != rerun.params.get(k))
    code = sorted(k for k in ("commit", "dirty", "diff_sha256", "core") if getattr(original.code, k) != getattr(rerun.code, k))
    same_value = (original.outputs.value is None and rerun.outputs.value is None) or (
        original.outputs.value is not None and rerun.outputs.value is not None
        and abs(original.outputs.value - rerun.outputs.value) <= tolerance)
    a_numbers = {k: v for k, v in original.outputs.numbers.items() if not TIMING.search(k)}  # fit_seconds.max is a timing too
    b_numbers = {k: v for k, v in rerun.outputs.numbers.items() if not TIMING.search(k)}
    changed_numbers = sorted(k for k in set(a_numbers) | set(b_numbers)
                             if k not in a_numbers or k not in b_numbers or abs(a_numbers[k] - b_numbers[k]) > tolerance)
    from dclab_core import steps as _steps

    walked = _steps.compare(original.steps, rerun.steps) if original.steps else None  # 23.8: where the rerun first differs (none recorded: nothing to compare)
    return {"same_inputs": not changed_inputs and not changed_params, "changed_inputs": changed_inputs, "changed_params": changed_params,
            "same_steps": walked["identical"] if walked else None, "first_step_difference": walked["first_difference"] if walked else None,
            "steps_compared": min(len(original.steps), len(rerun.steps)),
            "same_numbers": not changed_numbers, "changed_numbers": changed_numbers[:50], "numbers_compared": len(a_numbers),
            "same_code": not code, "changed_code": code, "same_value": same_value, "same_choice": original.outputs.chosen == rerun.outputs.chosen,
            "original": original.outputs.value, "rerun": rerun.outputs.value, "metric": original.outputs.metric}


def event_payload(record: RunRecord) -> dict[str, Any]:
    """The flat parts of a run for the session record's ``run.recorded`` event (meta and counts; never a table's values)."""
    inputs = record.inputs.model_dump()
    flat_inputs = {k: v for k, v in inputs.items() if k not in ("sampling", "settings", "decisions", "upstream", "refs")}
    flat_inputs.update({f"refs.{k}": v for k, v in record.inputs.refs.items()})
    flat_inputs.update({f"sampling.{k}": v for k, v in record.inputs.sampling.items()})
    flat_inputs.update({f"decisions.{k}": v for k, v in record.inputs.decisions.items()})
    outputs = {k: v for k, v in record.outputs.model_dump().items() if k != "numbers"}
    code = {k: v for k, v in record.code.model_dump().items() if k != "prompts"}
    return {"run": record.run, "run_kind": record.kind, "name": record.name, "status": record.status, "fingerprint": record.fingerprint,
            "inputs": _short(flat_inputs), "params": _short(record.params), "outputs": _short(outputs), "code": _short(code),
            "upstream": record.inputs.upstream[:50], "parent": record.parent, "seconds": record.seconds,
            "numbers": len(record.outputs.numbers), "steps": len(record.steps)}


def _short(mapping: Mapping[str, Any]) -> dict[str, Scalar]:
    return {str(k)[:MAX_TEXT]: (v[:MAX_TEXT] if isinstance(v, str) else v) for k, v in list(mapping.items())[:200]}


def latest(records: Iterable[RunRecord]) -> dict[str, RunRecord]:
    """The newest completed run of each name (a stage rerun replaces the run its numbers came from)."""
    out: dict[str, RunRecord] = {}
    for r in sorted(records, key=lambda r: r.started):
        if r.status == "completed":
            out[r.name] = r
    return out


CONTRACTS.update({"run_record": RunRecord})
