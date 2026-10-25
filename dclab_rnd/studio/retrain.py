"""Retrain and champion/challenger in the studio (34.11): newer labelled rows, a challenger fitted the champion's way, and a
comparison on the latest of those rows that neither model trained on.

The newer rows are a file kept beside the project's table (``PUT /api/projects/{id}/newer-rows/{name}``,
:func:`add_newer_rows`), stored through the project's file storage, fingerprinted and audited as a data import, and never
read by the engine: adding them clears no stage and never makes the final stage read the holdout again.

They are read as ``engine.prepare`` read the training rows: yes/no columns as 0 and 1, the labels of unconfirmed sources set
aside, the time column read as the project reads it (dates or numbers; a row without a time is left out), the model's
columns in the training rows' types (a code read as 1.0 in a CSV is the code 1), the outcome coded as the model predicts it.
A newer row equal to a row of the project's own table (its columns and its outcome) is dropped and counted: the window must
be rows neither model has seen, never a re-export of the project's holdout. The window is the latest ``policy.window_share``
of the newer rows after the project's latest time (with no time column, a seeded draw); with a group column, an entity in the
window has none of its rows among the challenger's training rows, and the interval is drawn over whole entities.

The champion is the fit in use: the last promoted one (the training rows and every newer table it was promoted with) or else
the approved model, refitted as the export refits it; the challenger is the same recipe, family and settings fitted on the
champion's rows and the newer rows before the window. ``dclab_core.retrain.compare`` reads both on the window with the
project's own measure. The project's holdout is never read; nothing changes but the project's record of comparisons. A
promotion names the rows the model in use should be fitted on; serving it is the host's step, never a production approval.
Every comparison is kept, its event and its ledger point recorded, and a lesson proposed from it (34.12).
"""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from dclab_core.retrain import ChallengerRecord, RetrainPolicy, compare

from ..expansion import runner as rn
from . import engine
from .data import load_table, safe_name

KEEP = 20  # comparisons kept on the project
PREFIX = "newer_"


class RetrainRefused(ValueError):
    """A retrain that cannot be run, with why."""


def _canonical(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """A key for each row from its model columns and its outcome, the same however a file wrote its numbers."""
    def text(column: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(column, errors="coerce")
        if numeric.notna().sum() == column.notna().sum():
            return numeric.map(lambda v: "" if pd.isna(v) else format(float(v), ".10g"))
        return column.map(lambda v: "" if pd.isna(v) else str(v))

    parts = pd.concat([text(X[c].reset_index(drop=True)) for c in X.columns] + [text(y.reset_index(drop=True))], axis=1)
    joined = parts.astype(str).agg("\x1f".join, axis=1)
    return joined.map(lambda s: hashlib.sha1(s.encode()).hexdigest()).to_numpy()


def _prepared(p: Any, frame: pd.DataFrame) -> dict[str, Any]:
    """The newer rows as ``engine.prepare`` read the training rows: X with the model's columns in their types, y coded, the
    order (a time), the groups and each row's key."""
    solution = p.solution
    missing = [c for c in [solution.target, *p.bundle.X.columns] if c not in frame.columns]
    if missing:
        raise RetrainRefused("The newer rows lack the project's columns: " + ", ".join(map(str, missing[:10])))
    frame = frame.copy()
    for column in frame.columns:
        if column != solution.target and (pd.api.types.is_bool_dtype(frame[column]) or str(frame[column].dtype) == "boolean"):
            frame[column] = pd.to_numeric(frame[column].astype("object").map({True: 1.0, False: 0.0}), errors="coerce")
    try:
        engine._label_sources(frame, solution)  # 30.12: a noisy source's labels set aside, as the training rows' were
    except ValueError as exc:
        raise RetrainRefused(str(exc)) from None
    frame = frame[frame[solution.target].notna()].reset_index(drop=True)
    if frame.empty:
        raise RetrainRefused("The newer rows hold no confirmed outcome")
    y, labels = p.kind.target(frame, solution)
    if p.task == "multiclass" and list(labels or []) != list(p.class_labels or []):
        raise RetrainRefused("The newer rows' classes differ from the model's: a challenger would predict other classes")
    keep = y.notna()
    frame, y = frame[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    order = None
    time = solution.time_column
    if time and time in frame:  # read as the project read it; a row without a time is left out, never sorted to the end
        if p.dated:
            order = pd.to_datetime(frame[time], errors="coerce")
        else:
            order = pd.to_numeric(frame[time], errors="coerce")
        valid = order.notna().to_numpy()
        frame, y, order = frame[valid].reset_index(drop=True), y[valid].reset_index(drop=True), order[valid].reset_index(drop=True)
        frame[time] = order.dt.strftime("%Y-%m-%d %H:%M:%S") if p.dated else order
    X = frame[list(p.bundle.X.columns)].copy()
    codes = set(p.bundle.spec.categorical_columns or [])
    for column in X.columns:  # the training rows' types: a code read as 1.0 is the code 1, a number stays a number
        trained = p.bundle.X[column]
        if column in codes or pd.api.types.is_integer_dtype(trained):
            numeric = pd.to_numeric(X[column], errors="coerce")
            whole = numeric.notna() & (numeric == numeric.round())
            if pd.api.types.is_numeric_dtype(trained) and (whole | X[column].isna()).all():
                X[column] = numeric.map(lambda v: v if pd.isna(v) else int(v)).astype(object if X[column].isna().any() else "int64")
        elif pd.api.types.is_float_dtype(trained):
            X[column] = pd.to_numeric(X[column], errors="coerce")
    group = solution.group_column
    groups = frame[group].astype(str).to_numpy() if group and group in frame else None
    return {"X": X, "y": y, "order": order, "groups": groups, "keys": _canonical(X, y)}


def _project_keys(p: Any) -> set[str]:
    """The keys of every row of the project's table, training and holdout alike (only to drop repeats; no label is read)."""
    return set(_canonical(p.bundle.X, p.bundle.y))


def _latest_time(p: Any) -> Any:
    time = p.solution.time_column
    if not time or time not in p.bundle.X:
        return None
    values = pd.to_datetime(p.bundle.X[time], errors="coerce") if p.dated else pd.to_numeric(p.bundle.X[time], errors="coerce")
    return values.max()


def add_newer_rows(store: Any, project_id: str, body: bytes, filename: str, name: str, actor: str = "human") -> dict[str, Any]:
    """Keep newer labelled rows beside the project's table under ``name``: checked to read as a table, stored through the
    project's file storage, fingerprinted, audited as a data import. The stages and the solution stay as they are."""
    from ..storage.files import FileError, files_for

    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")[:60]
    original = safe_name(filename or f"{name}.csv")
    if not name or not original:
        raise RetrainRefused("Give the newer rows a name and a file name")
    suffix = Path(original).suffix or ".csv"
    file = f"{PREFIX}{name}{suffix}"  # one file per name: two names never share a file, whatever the uploads were called
    folder = store.data_dir(project_id)
    folder.mkdir(parents=True, exist_ok=True)
    staged = folder / f".staged-{file}"
    staged.write_bytes(body)
    try:
        frame = load_table(staged)
    except Exception as exc:  # noqa: BLE001 - whatever the reader says, the bytes are not a table it reads
        staged.unlink(missing_ok=True)
        raise RetrainRefused(f"The newer rows do not read as a table: {str(exc)[:200]}") from None
    path = folder / file
    staged.replace(path)
    try:
        files = files_for(store)
        stored = files.put(files.key_of(path), path)
    except Exception as exc:  # noqa: BLE001 - as attach_table
        path.unlink(missing_ok=True)
        raise RetrainRefused(str(exc) if isinstance(exc, FileError) else f"The newer rows could not be stored ({type(exc).__name__})") from None
    project = store.get(project_id)
    entry = {"filename": file, "sha256": stored["sha256"], "rows": int(len(frame)), "added": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    project["newer_rows"] = {**(project.get("newer_rows") or {}), name: entry}
    store.save(project)
    store.log(project_id, "newer_rows_added", {"name": name, "rows": int(len(frame))})
    from .. import audit

    audit.record_for(store, "data_import", actor, project=project, move="add_newer_rows", status="allowed", source={"kind": "newer_rows", "name": name},
                     filename=file, sha256=stored["sha256"], rows=int(len(frame)), columns=int(frame.shape[1]))
    return {name: entry}


def _load_newer(store: Any, project: dict[str, Any], name: str) -> pd.DataFrame:
    from ..storage.files import FileError, files_for

    known = (project.get("newer_rows") or {}).get(name)
    if known is None:
        raise RetrainRefused(f"The project keeps no newer rows named {name!r}: add them first")
    path = store.data_dir(project["id"]) / known["filename"]
    try:
        files = files_for(store)
        return load_table(files.path(files.key_of(path), expected=known["sha256"]))
    except FileError:
        raise RetrainRefused(f"The newer rows {name!r} changed or are gone since they were added: add them again") from None


def _fingerprint(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _window(n: int, order: Optional[pd.Series], after: Any, share: float) -> tuple[np.ndarray, np.ndarray]:
    """(the rows before the window, the window): the latest ``share`` of the rows later than ``after`` by the time, or a
    seeded draw when there is no time (a file sorted by outcome never makes a one-class window)."""
    k = max(1, int(round(share * n)))
    if order is None:
        drawn = np.random.default_rng(0).permutation(n)
        return np.sort(drawn[k:]), np.sort(drawn[:k])
    ranked = np.argsort(order.to_numpy(), kind="mergesort")
    candidates = ranked if after is None or pd.isna(after) else ranked[(order.to_numpy()[ranked] > after)]
    window = candidates[-k:] if len(candidates) else candidates
    return np.setdiff1d(np.arange(n), window), np.sort(window)


def run(store: Any, project_id: str, newer: str, *, policy: Optional[RetrainPolicy] = None, because: tuple[str, ...] = (),
        actor: str = "human", lesson_store: Any = None, live_report: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Compare a challenger fitted with the newer rows ``newer`` with the champion; returns the comparison's record.
    ``live_report``: the service's ``/monitor`` answer (34.10, a ``LiveReport``), whose alerts say why, and become a lesson."""
    from dclab_core.live import LiveReport

    from ..jobs import checkpoint
    from .data_value import _scorer

    project = store.get(project_id)
    if (project.get("stages", {}).get("final") or {}).get("status") not in ("completed", "approved"):
        raise RetrainRefused("Run the final stage first: there is no champion to compare with")
    policy = policy or RetrainPolicy()
    report = None
    if live_report:
        try:
            report = LiveReport.model_validate(live_report)
        except ValueError as exc:
            raise RetrainRefused(f"The live report is not a monitor's answer: {str(exc)[:200]}") from None
        from dclab_core.retrain import should_retrain

        because = (*because, *should_retrain(policy, live_report=report, new_labelled=10 ** 9)["because"])
    final = store.read_stage(project_id, "final")["evidence"]
    recipe, family, params = final["feature_recipe"], final["model"], final.get("selected_model_params") or None
    p = engine.prepare(store, project)
    if p.task not in ("binary", "multiclass", "regression") or getattr(p, "bundle", None) is None:
        raise RetrainRefused("This kind of problem has no champion and challenger yet")
    if p.solution.spatial is not None:
        raise RetrainRefused("A split by blocks of places derives its block column; newer rows cannot carry it yet")
    known = _project_keys(p)
    preferred = project.get("preferred_fit") or {}
    included = [i for i in preferred.get("includes") or [] if i.get("name") != newer]
    champion_X, champion_y = [p.bundle.X_train], [p.bundle.y_train]
    for item in included:  # the fit in use: the training rows and every newer table it was promoted with
        if ((project.get("newer_rows") or {}).get(item["name"]) or {}).get("sha256") != item.get("sha256"):
            raise RetrainRefused(f"The promoted fit used the newer rows {item['name']!r}, which changed since: compare against the approved model "
                                 "by adding them again under another name")
        rows = _prepared(p, _load_newer(store, project, item["name"]))
        fresh = ~np.isin(rows["keys"], list(known))
        champion_X.append(rows["X"][fresh])
        champion_y.append(rows["y"][fresh])
        known |= set(rows["keys"])
    checkpoint()
    got = _prepared(p, _load_newer(store, project, newer))
    repeat = np.isin(got["keys"], list(known))
    dropped = int(repeat.sum())
    if dropped:  # a re-export of rows a model has seen: never compared on, never trained on twice
        for key in ("X", "y"):
            got[key] = got[key][~repeat].reset_index(drop=True)
        got["order"] = got["order"][~repeat].reset_index(drop=True) if got["order"] is not None else None
        got["groups"] = got["groups"][~repeat] if got["groups"] is not None else None
    n = len(got["y"])
    if n < policy.min_new_labelled:
        raise RetrainRefused(f"{n} newly labelled rows ({dropped} more repeat rows the project holds); a challenger waits for {policy.min_new_labelled}")
    before, window = _window(n, got["order"], _latest_time(p), policy.window_share)
    if not len(window):
        raise RetrainRefused("None of the newer rows is later than the project's own rows: there is no window neither model has seen")
    if got["groups"] is not None:  # an entity in the window has none of its rows among the challenger's
        before = before[~np.isin(got["groups"][before], got["groups"][window])]
    X_champ, y_champ = pd.concat(champion_X, ignore_index=True), pd.concat(champion_y, ignore_index=True)
    X_fit = pd.concat([X_champ, got["X"].iloc[before]], ignore_index=True)
    y_fit = pd.concat([y_champ, got["y"].iloc[before]], ignore_index=True)
    X_window, y_window = got["X"].iloc[window].reset_index(drop=True), got["y"].iloc[window].reset_index(drop=True)
    champion = rn.fit_and_predict(p.bundle, recipe, family, params, X_champ, y_champ, X_window)
    checkpoint()  # Stop works between the fits
    challenger = rn.fit_and_predict(p.bundle, recipe, family, params, X_fit, y_fit, X_window)
    checkpoint()
    task = p.task if p.task in ("binary", "multiclass") else "regression"
    score, fast = _scorer(p.bundle)
    sha = project["newer_rows"][newer]["sha256"]
    against = "the last promoted fit" if included else "the approved model"
    where = ("the latest" if got["order"] is not None else "a seeded draw of") + f" {len(window)} of {n} newer rows of {newer}"
    record: ChallengerRecord = compare(
        task, y_window.to_numpy(dtype=float), champion["predictions"], challenger["predictions"], policy=policy, because=because,
        window=where + (" after the project's latest time" if got["order"] is not None and _latest_time(p) is not None else ""),
        metric=p.bundle.spec.primary_metric, metric_fn=score, higher_is_better=p.bundle.spec.higher_is_better,
        groups=got["groups"][window] if got["groups"] is not None else None, bootstrap=1000 if fast else 200,
        champion_version=_fingerprint(project["data"]["sha256"], *(i["sha256"] for i in included), recipe, family, params),
        challenger_version=_fingerprint(project["data"]["sha256"], *(i["sha256"] for i in included), sha, len(before), recipe, family, params))
    entry = {**record.model_dump(mode="json"), "id": secrets.token_hex(5), "newer": newer, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "against": against, "repeats_dropped": dropped,
             "trained_on": {"champion": "the training rows" + "".join(f" and the newer rows of {i['name']}" for i in included),
                            "challenger": f"the champion's rows and {len(before)} newer rows of {newer}"}}
    project = store.get(project_id)
    project["retrains"] = ([*(project.get("retrains") or []), entry])[-KEEP:]
    if record.promoted:  # the fit the model in use should now be; serving it is the host's step, never a production approval here
        project["preferred_fit"] = {"version": record.challenger_version, "includes": [*included, {"name": newer, "sha256": sha}],
                                    "trained_on": "the training rows and every row of " + ", ".join([*(i["name"] for i in included), newer]),
                                    "since": entry["at"], "production_approved": False}
    store.save(project)
    store.log(project_id, "retrain_compared", {k: entry[k] for k in ("metric", "champion", "challenger", "difference", "interval", "promoted",
                                                                      "window", "against", "repeats_dropped")})
    from ..experience import hooks

    hooks.retrain_compared(project_id, record, actor)
    _capture(project_id, record)
    from . import lessons_from_use

    lessons_from_use.propose(store, project_id, entry, lesson_store)  # 34.12
    if report is not None:
        lessons_from_use.propose_from_live(store, project_id, report.model_dump(mode="json"), lesson_store)
    return entry


def _capture(project_id: str, record: ChallengerRecord) -> None:
    """The ledger point ``model.promote`` (a yes or no): the champion/challenger rule's answer. Its state is what the answer
    could be read from before the comparison (the measure, the window's size), never the difference the rule read."""
    from ..ledger import answer, rows_band
    from ..ledger.capture import _code, _put

    _put("model.promote", ("project", project_id), [answer("rule", "rule:champion_challenger", bool(record.promoted), reason=record.reason[:600])],
         bool(record.promoted), "rule", state={"metric": _code(record.metric), "rows": rows_band(record.window_rows)}, options=("yes", "no"),
         project=project_id)


def view(store: Any, project_id: str) -> dict[str, Any]:
    """The comparisons kept, newest last, and the fit preferred by the last promotion (never a production approval)."""
    project = store.get(project_id)
    return {"retrains": project.get("retrains") or [], "preferred_fit": project.get("preferred_fit")}
