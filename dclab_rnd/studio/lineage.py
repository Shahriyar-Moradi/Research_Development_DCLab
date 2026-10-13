"""Every stage run's lineage in this studio (package 19.5), on ``dclab_core.lineage``.

The engine gives each stage run an id and writes its :class:`~dclab_core.lineage.RunRecord` into the stage record
(``record["run"]``): the table's hash and size, how rows were sampled, the solution's hash, the project's settings,
the choices of the earlier stages it used and their runs, the seed; the stage's parameters; its metric, value and
interval, its choice and every number it wrote; the code (commit, dclab_core's version, the prompts' hashes). The
session record gets a ``run.recorded`` event, and every event inside the run carries its id.

:func:`trace` answers "where does this number come from?" for a report path (``final.evidence.holdout_metrics.roc_auc``):
its value, the run, its inputs and its code. :func:`replay` reruns a stage from its record in a scratch copy of the
project (never the project itself, never a model) and :func:`dclab_core.lineage.compare` says whether the number is
the same.

    python -m dclab_rnd.experience lineage PROJECT [--path final.evidence.holdout_metrics.roc_auc]
    python -m dclab_rnd.experience replay PROJECT STAGE
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from dclab_core import __version__ as CORE_VERSION
from dclab_core import lineage as contract

ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = ("dclab_rnd", "dclab_core", "general_pipeline")
# the project settings that change what a stage computes (rows, folds, sampling, the critic's rounds, private scope);
# the others (the training opt-in) never enter a run's inputs, so changing them does not make a run different
RUN_SETTINGS = ("max_rows", "quick", "folds", "critic", "private")
# a report's roots that no stage run writes: they are made when the report is (the brief, the findings, the plan)
REPORT_ROOTS = ("title", "brief", "findings", "findings_meta", "plan", "plan_risks", "plan_would_change", "rubric")


def new_run() -> str:
    return "run" + secrets.token_hex(8)


def code(runner: str = "dclab_rnd.studio.engine") -> contract.RunCode:
    """The code this process runs, for ``runner``: its commit, whether tracked code differed from it, dclab_core's
    version, the prompts' and tools' hashes. Read once per process (a server restarts when its code changes)."""
    return _code().model_copy(update={"runner": runner})


@functools.lru_cache(maxsize=1)
def _code() -> contract.RunCode:
    commit = dirty = diff = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
        if commit:
            changes = subprocess.run(["git", "diff", "HEAD", "--", *CODE_PATHS], cwd=ROOT, capture_output=True, timeout=30).stdout
            dirty = bool(changes)
            diff = hashlib.sha256(changes).hexdigest() if changes else None
    except (OSError, subprocess.SubprocessError):
        pass
    if not commit:  # a container has no .git: the image says which commit it was built from (Dockerfile ARG DCLAB_COMMIT)
        commit = (os.environ.get("DCLAB_COMMIT") or "").strip()[:64] or None
    try:
        from .. import prompts

        hashes = prompts.fingerprints()
    except Exception:  # noqa: BLE001 - a record without the prompts' hashes still traces its numbers
        hashes = {}
    return contract.RunCode(commit=commit, dirty=dirty, diff_sha256=diff, core=CORE_VERSION, prompts=hashes)


def _profile() -> str:
    from ..expansion import runner

    return "fast" if runner._PROFILE.get("fast") else "full"


def _short_hash(value: Any) -> Optional[str]:
    if value in (None, {}, [], ""):
        return None
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def inputs_of(project: dict[str, Any], previous: dict[str, Optional[dict[str, Any]]], prepared: Any = None,
              seed: Optional[int] = None) -> contract.RunInputs:
    """What a stage run reads: the table, the sampling, the solution, the settings that change a run, the earlier
    stages' choices and their runs, and by hash everything else ``engine.prepare`` keys its work on (the labels, the
    pack, the feature programs, the plan, the declared categorical columns)."""
    from ..studio.graph import solution_hash

    data = project.get("data") or {}
    earlier = {k: r for k, r in previous.items() if r}
    refs = {"labels": (project.get("labels") or {}).get("sha256") or _short_hash(project.get("labels")),
            "pack": _short_hash(project.get("pack")), "feature_programs": _short_hash(project.get("feature_programs")),
            "plan": _short_hash(project.get("plan")), "categorical": _short_hash(data.get("categorical"))}
    settings = project.get("settings") or {}
    return contract.RunInputs(
        data_sha256=data.get("sha256"), data_rows=data.get("rows"),
        data_columns=len(data["columns"]) if isinstance(data.get("columns"), list) else None,
        sampling=contract.flat(getattr(prepared, "sampling", None) or {}), solution_hash=solution_hash(project.get("solution")),
        # the engine's model sizes (expansion.runner: "fast" shrinks ensembles and draws for tests) change every number
        settings=contract.flat({**{k: settings[k] for k in RUN_SETTINGS if k in settings}, "engine_profile": _profile()}),
        decisions={k: str((r.get("decision") or {}).get("chosen"))[:300] for k, r in earlier.items() if (r.get("decision") or {}).get("chosen") is not None},
        refs={k: v for k, v in refs.items() if v},
        upstream=[str((r.get("run") or {}).get("run")) for r in earlier.values() if (r.get("run") or {}).get("run")], seed=seed)


def _chosen_value(record: dict[str, Any]) -> Optional[float]:
    """The value the stage's choice was made on: the holdout's for the final stage, the chosen option's mean otherwise."""
    metric = record.get("primary_metric")
    evidence = record.get("evidence") or {}
    if record.get("stage") == "final":
        value = (evidence.get("holdout_metrics") or {}).get(metric)
    else:
        decision = record.get("decision") or {}
        value = next((o.get("mean") for o in decision.get("options") or [] if o.get("id") == decision.get("chosen")), None)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def stage_run(run: str, project: dict[str, Any], stage: str, *, previous: dict[str, Optional[dict[str, Any]]], prepared: Any = None,
              record: Optional[dict[str, Any]] = None, params: Optional[dict[str, Any]] = None, started: str, finished: Optional[str],
              seconds: Optional[float], status: str = "completed", error: Optional[str] = None, parent: Optional[str] = None,
              seed: Optional[int] = None) -> contract.RunRecord:
    """The run record of one stage run, from the project as it ran, the earlier stages it read, and what it wrote."""
    inputs = inputs_of(project, previous, prepared, seed)
    outputs = contract.RunOutputs()
    if record is not None:
        ci = (record.get("evidence") or {}).get("holdout_primary_metric_ci") if stage == "final" else None
        chosen = (record.get("decision") or {}).get("chosen")
        outputs = contract.RunOutputs(metric=record.get("primary_metric"), value=_chosen_value(record),
                                      low=(ci or {}).get("low") if isinstance(ci, dict) else None,
                                      high=(ci or {}).get("high") if isinstance(ci, dict) else None,
                                      chosen=str(chosen)[:300] if chosen is not None else None,
                                      numbers=contract.numbers(record.get("evidence") or {}, "evidence"))
    return contract.RunRecord(run=run, kind="stage", name=stage, project=project.get("id"), parent=parent, status=status, started=started,
                              finished=finished, seconds=seconds, inputs=inputs, params=contract.flat(params or {}), outputs=outputs,
                              code=code(), error=(error or None) and error[:600])


def trace(store: Any, project_id: str, path: str) -> dict[str, Any]:
    """Where a report's number comes from: the value at its path (as ``dclab_core.report`` reads it, ``@n`` and ``#n``
    included) and, when a stage wrote it, that stage's run with its inputs and code. A path under the brief, the
    findings or the plan was made when the report was, from the project itself: it is said so, with no run."""
    from dclab_core import report

    from .store import STAGE_KEYS

    head = path.split(".", 1)[0].split("@", 1)[0].split("#", 1)[0]
    if head in STAGE_KEYS:
        records = store.records(project_id)
        if head not in records:
            raise KeyError(f"no stage record {head!r} in this project")
        try:
            value = report.lookup(report.context(records=records), path)
        except KeyError:
            raise KeyError(f"no value at {path!r} in the {head} record") from None
        run = records[head].get("run")
        return {"path": path, "stage": head, "value": _scalar(value), "run": run,
                "note": None if run else "This stage ran before runs were recorded (19.5): rerun it to record its run."}
    if head in REPORT_ROOTS:
        from .solution_report import inputs

        try:
            value = report.lookup(report.context(**{k: v for k, v in inputs(store, project_id, analyse=head.startswith("findings")).items()
                                                    if k in ("brief", "findings", "plan", "records", "rubric", "title", "rows_note")}), path)
        except KeyError:
            raise KeyError(f"no value at {path!r}") from None
        return {"path": path, "stage": None, "value": _scalar(value), "run": None,
                "note": f"Made when the report is written, from the project's {head.split('_')[0]} (not by a stage run)."}
    raise KeyError(f"{path!r} is not a report path: it starts with a stage ({', '.join(STAGE_KEYS)}) or a report part")


def _scalar(value: Any) -> Any:
    return value if isinstance(value, (int, float, str, bool)) or value is None else None


def runs(store: Any, project_id: str) -> dict[str, Any]:
    """Each stage's latest run as recorded, by stage (None for a stage that ran before runs were recorded)."""
    from .store import STAGE_KEYS

    out: dict[str, Any] = {}
    for stage in STAGE_KEYS:
        record = store.read_stage(project_id, stage)
        if record:
            out[stage] = record.get("run")
    return out


def replay(store: Any, project_id: str, stage: str) -> dict[str, Any]:
    """Rerun a stage from its record in a scratch copy of the project and compare what it makes with the record.

    Refused (``replayed`` False, with why) when the stage has no run record, or when anything the run read would be
    different now (the table, the sampling, the solution, the settings that change a run, an earlier stage's choice,
    the labels, the pack, the feature programs, the plan): a replay reproduces a run, it never makes a new one. The
    final stage reads the holdout, so it is replayed only on the very code it ran on: it then reads the same rows with
    the same model and can only repeat the number, never choose anything. The copy lives in a temporary folder that is
    removed after; no model is asked (the run is private) and nothing it does reaches the project, its audit or the
    session record, except one ``run.recorded`` for the replay, with the original run as its parent."""
    from .. import experience
    from ..expansion import datasets as ds
    from ..models.gateway import private_scope
    from . import engine
    from .data import DataError, data_path
    from ..storage.files import JsonRecords, LocalFiles
    from .store import STAGE_KEYS, ProjectStore

    if stage not in STAGE_KEYS:
        raise KeyError(f"unknown stage {stage!r}")
    project = store.get(project_id)
    record = store.read_stage(project_id, stage)
    if not record or not record.get("run"):
        return {"replayed": False, "why": ["This stage has no run record: it ran before runs were recorded (19.5)."]}
    original = contract.RunRecord.model_validate(record["run"])
    earlier = {k: store.read_stage(project_id, k) for k in STAGE_KEYS[:STAGE_KEYS.index(stage)]}
    try:
        now = inputs_of(project, earlier, engine.prepare(store, project), seed=ds.RANDOM_STATE)
    except (DataError, ValueError) as exc:
        return {"replayed": False, "why": [f"The project cannot be read as it was: {exc}"]}
    then, current = original.inputs.model_dump(), now.model_dump()
    then.pop("upstream"), current.pop("upstream")
    changed = sorted(k for k in set(then) | set(current) if then.get(k) != current.get(k))
    if changed:
        return {"replayed": False, "why": [f"What the run read has changed since: {', '.join(changed)}. Rerun the stage instead."]}
    if stage == "final":
        mine = code()
        moved = [k for k in ("commit", "dirty", "diff_sha256") if getattr(original.code, k) != getattr(mine, k)]
        if moved or not original.code.commit:
            return {"replayed": False, "why": ["The final stage reads the holdout: it is replayed only on the code it ran on "
                                               f"({', '.join(moved) or 'no commit recorded'} differ)."]}
    with tempfile.TemporaryDirectory(prefix="dclab-replay-") as tmp:
        scratch = ProjectStore(Path(tmp) / "projects")
        # its files and their records in the scratch folder: never a workspace, a database row or a bucket of their own
        scratch.file_storage = LocalFiles(Path(tmp), records=JsonRecords(Path(tmp)))
        scratch.directory(project_id).mkdir(parents=True)
        copy = {**project, "stages": {k: v for k, v in (project.get("stages") or {}).items() if k in earlier}}
        used = (record.get("evidence") or {}).get("holdout_uses_in_this_project")
        if isinstance(used, int) and used > 0:  # the copy counts the holdout as the run did: a replay repeats a use, adds none
            copy["holdout_uses"] = used - 1
        scratch.save(copy)
        shutil.copytree(data_path(store, project).parent, scratch.data_dir(project_id), dirs_exist_ok=True)
        for k, r in earlier.items():
            if r:
                scratch.write_stage(project_id, k, r)
        reuse = "replay of a run that reused the holdout" if original.params.get("holdout_reuse") else None
        with experience.quiet(), private_scope(True):
            again = engine._execute_stage(scratch, project_id, stage, scratch.get(project_id), reuse, parent=original.run)
    rerun = contract.RunRecord.model_validate(again["run"])
    experience.record("run.recorded", "system", contract.event_payload(rerun), project=project_id, run=rerun.run)
    return {"replayed": True, "run": rerun.run, "original_run": original.run, **contract.compare(original, rerun)}
