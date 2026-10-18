"""The loop made continuous (package 29.7): a workspace's students kept up with its labels, on a schedule.

The worker's sweep queues one job of kind ``students`` when the workspace has labels it has not learned from (after a
cooldown, so a person labelling thirty disagreements starts one run, not thirty), or once a night. The run:

1. **Sets**: the workspace's labelled decisions per point, rebuilt (``own_models.export``: whole projects to train, dev or
   sealed by their hash; a sealed decision is never trained on).
2. **Train**: a point's student is fitted again only when its homework (its train and dev rows) grew by a margin since
   its last fit (``MARGIN`` of it, at least ``MIN_NEW`` rows); a point without a student gets its first; the others wait,
   with how many rows they still need (``waiting``). Each new card carries its dev score beside the rule's (the dev
   comparison), from ``dclab_core.decision_models``.
3. **Demotion**: a routed model whose weighted accuracy on labels made since it was routed fell below its bar goes back
   to shadow (``ledger.routing.demote``: one audited change by DCLab, the rule answers again). Promotion is never the
   loop's: it stays a person's audited switch, after a sealed pass and a shadow measurement (22.6).
4. **Register**: one line in ``students/register.jsonl`` (``students.record``), so the Students page's history per point
   shows the new card; the run itself is a ``run.recorded`` event and its state is ``students/loop.json``.

The loop keeps which ledger its students learned from (``loop.json``): a folder read on files and on PostgreSQL has two
ledgers, and an empty one is not every label deleted, so another ledger never runs the sweep and the command refuses it
until a person adopts it (``--adopt``). A route naming an own model that is no longer here, or one whose training
decisions shrank, goes back to shadow first (audited); a model learned from deleted decisions that cannot be fitted again
is forgotten. ``DCLAB_STUDENTS_LOOP=0`` turns the loop off. A run that fails is tried again after an hour.

    python -m dclab_rnd.students_loop [--home DIR] [--adopt]   # one run now, without the job table
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

FOLDER = "students"
STATE = "loop.json"
MARGIN = 0.10           # a student is trained again when its homework grew by a tenth since its last fit
MIN_NEW = 5             # ... and by at least this many rows
COOLDOWN_SECONDS = 15 * 60
NIGHTLY_SECONDS = 24 * 3600
RETRY_SECONDS = 3600


def on() -> bool:
    return (os.environ.get("DCLAB_STUDENTS_LOOP") or "1").strip() != "0"


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def state(home: Path) -> dict[str, Any]:
    try:
        return json.loads((Path(home) / FOLDER / STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(home: Path, doc: dict[str, Any]) -> None:
    path = Path(home) / FOLDER / STATE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def ledger_id(ledger: Any) -> str:
    """Which ledger a workspace's students learned from: its kind and workspace (a folder is read by the file ledger and,
    with DCLAB_DATABASE_URL, by PostgreSQL: two ledgers, one folder)."""
    return f"{type(ledger).__name__}:{getattr(ledger, 'workspace', '')}"


class OtherLedger(RuntimeError):
    """The students were learned from another ledger than the one given: nothing is changed until a person adopts it."""


def signature(ledger: Any) -> Optional[str]:
    """What changes when a label is added or deleted, read cheaply (the ledger's ``label_signature``: file metadata, or one
    count query); None when the ledger cannot be read (then no run is due: an unreadable ledger is not a deletion)."""
    if ledger is None:
        return None
    try:
        sign = getattr(ledger, "label_signature", None)
        return str(sign()) if callable(sign) else str(len(list(ledger.labels())))
    except Exception:  # noqa: BLE001
        return None


def due(home: Path, ledger: Any, now: Optional[float] = None) -> bool:
    """New labels since the last run (and the cooldown over), or a night since it; never while switched off, nor within
    an hour of a failed run, nor in a workspace that has never had a label, nor while the ledger cannot be read."""
    if not on():
        return False
    clock = now if now is not None else time.time()
    doc = state(home)
    failed_at = doc.get("failed_clock")
    if isinstance(failed_at, (int, float)) and clock - failed_at < RETRY_SECONDS:
        return False
    if doc.get("ledger") and ledger is not None and doc["ledger"] != ledger_id(ledger):
        return False  # another ledger on this folder (the review of 29.7): its runs are a person's (``--adopt``), never the sweep's
    last = doc.get("clock")
    if isinstance(last, (int, float)) and clock - last < COOLDOWN_SECONDS:
        return False  # before the ledger is read at all: the sweep runs every minute
    if isinstance(last, (int, float)) and clock - last >= NIGHTLY_SECONDS:
        return True
    now_sig = signature(ledger)
    if now_sig is None:
        return False
    if last is None:
        return now_sig not in ("", "0", "0:None")  # a first run only once there is a label
    return now_sig != doc.get("labels")


def failed(home: Path, now: Optional[float] = None) -> None:
    """A run that did not finish: the next one waits ``RETRY_SECONDS``."""
    doc = state(home)
    doc.update(failed_clock=now if now is not None else time.time(), failed_at=_stamp())
    _save(home, doc)


def sweep_due(services: Any, now: Optional[float] = None) -> Optional[dict[str, Any]]:
    """The job, queued when a run is due (one at a time: a queued or running one holds the key); None otherwise."""
    from .jobs import ActiveJob

    home = Path(services.store.home)
    if not due(home, getattr(services, "ledger", None), now):
        return None
    try:
        return services.queue_job("students", "loop", {}, by="agent")
    except ActiveJob:
        return None


def _card(card: Any, homework: int) -> dict[str, Any]:
    return {"answerer": f"{card.name}@{card.version}", "family": card.family, "trained_on": card.trained_on, "homework": homework,
            "dev_accuracy": card.dev.accuracy, "rule_dev_accuracy": card.rule_dev.accuracy, "dev_n": card.dev.n}


def run(home: Path, ledger: Any, routing_store: Any, audit_log: Any, *, measure: bool = True, now: Optional[float] = None,
        adopt: bool = False) -> dict[str, Any]:
    """One run of the loop (the job's work, or the command's); what it rebuilt, retrained, left waiting and demoted. One at
    a time per workspace (the students folder's lock): a command and a job never fit into the same files at once.
    OtherLedger when the students were learned from another ledger than ``ledger`` (a folder read on files and on
    PostgreSQL): an empty ledger is not every label deleted. ``adopt``: a person takes ``ledger`` as the source."""
    from .filelock import held

    with held(Path(home) / FOLDER):
        doc = state(Path(home))
        if doc.get("ledger") and doc["ledger"] != ledger_id(ledger) and not adopt:
            raise OtherLedger(f"the students here learned from {doc['ledger']}, not {ledger_id(ledger)}: run the command with --adopt "
                              "to learn from this one (its labels then replace the other's)")
        return _run(Path(home), ledger, routing_store, audit_log, measure=measure, now=now)


def _retire_route(routing_store: Any, audit_log: Any, point: str, entry: dict[str, Any], reason: str, demoted: list[dict[str, Any]]) -> None:
    """An own model's route moved back to shadow by DCLab (audited), and said in the run's ``demoted``."""
    from .ledger import routing

    routing.switch(routing_store, audit_log, point, entry["answerer"], 0, None, reason, by_system=True)
    demoted.append({"decision_point": point, "answerer": entry["answerer"], "reason": reason})


def _run(home: Path, ledger: Any, routing_store: Any, audit_log: Any, *, measure: bool, now: Optional[float]) -> dict[str, Any]:
    from dclab_core import lineage

    from . import experience, own_models, students
    from .jobs.worker import checkpoint, progress
    from .ledger import routing
    from .studio.lineage import code, new_run

    started, began, clock = _stamp(), time.perf_counter(), now if now is not None else time.time()
    doc = state(home)
    seen = signature(ledger)
    # demotion first: the safety step never waits for a fit that fails (the review of 29.7)
    checkpoint()
    progress(step="demotion", of=4)
    demoted = routing.demote(routing_store, audit_log, list(ledger.records()), list(ledger.labels()))
    checkpoint()
    progress(step="sets", of=4)
    manifest = own_models.export(home, ledger)
    homework = {p: sum(v for k, v in (info.get("by_split") or {}).items() if k != "sealed") for p, info in manifest["points"].items()}
    fitted, skipped_before = dict(doc.get("fitted") or {}), dict(doc.get("skipped") or {})
    def own_route(point: str) -> dict[str, Any]:
        entry = ((routing_store.get() or {}).get("decisions") or {}).get(point) or {}
        return entry if str(entry.get("answerer") or "").startswith(f"{own_models.NAME}@") and int(entry.get("level") or 0) > 0 else {}

    here = {f"{m.name}@{m.version}" for m in own_models.load(home)}
    for point in sorted(((routing_store.get() or {}).get("decisions") or {})):
        entry = own_route(point)
        if entry and entry["answerer"] not in here:  # a route naming a model this workspace no longer has: the rule answers, said
            _retire_route(routing_store, audit_log, point, entry, "its model is not in this workspace any more: back to shadow, the rule answers", demoted)
    cards_dir = own_models.root(home) / "models"
    grown, waiting, retired = [], {}, []
    for point, rows in sorted(homework.items()):
        entry = own_route(point)
        if entry:
            if rows < (fitted.get(point) or 0):  # it learned from decisions since deleted: back to shadow, then fitted on what is left
                _retire_route(routing_store, audit_log, point, entry, "some of its training decisions were deleted: back to shadow to be fitted again", demoted)
            else:
                # a model that serves is never replaced under its route: a person moves it to shadow first
                waiting[point] = {"homework": rows, "frozen": f"it serves at level {entry['level']}: move it to shadow before it is fitted again"}
                continue
        before = skipped_before.get(point, fitted.get(point))  # the last try, fitted or skipped
        if before is None or not (cards_dir / f"{point}.json").is_file() and point not in skipped_before:
            grown.append(point)  # no student yet
            continue
        need = max(MIN_NEW, math.ceil(MARGIN * before))
        if rows < before or rows - before >= need:  # grown by the margin, or shrunk (deleted rows must leave the model)
            grown.append(point)
        else:
            waiting[point] = {"homework": rows, "last_fit_on": before, "rows_needed": before + need - rows}
    for point in sorted(set(fitted) | set(skipped_before)):
        if point in homework:
            continue
        entry = own_route(point)
        if entry:
            _retire_route(routing_store, audit_log, point, entry, "its training decisions were deleted: back to shadow, the rule answers", demoted)
        if own_models.forget(home, point):  # every labelled decision of the point is gone: so is what was learned from them
            retired.append(point)
        fitted.pop(point, None)
        skipped_before.pop(point, None)
    cards: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for i, point in enumerate(grown):
        checkpoint()
        progress(step="train", of=4, point=point, done=i, points=len(grown))
        try:  # one point at a time: a fit that fails costs that point only, and fitted.json is written after each
            got = own_models.fit(home, [point])
        except Exception as exc:  # noqa: BLE001
            got = {"cards": {}, "skipped": {point: f"{type(exc).__name__}: {str(exc)[:200]}"}}
        for name, card in got["cards"].items():
            cards[name] = _card(card, homework[name])
            fitted[name] = homework[name]
            skipped_before.pop(name, None)
        for name, why in got["skipped"].items():
            skipped[name] = str(why)[:300]
            if homework[name] < (fitted.get(name) or 0) and own_models.forget(home, name):  # shrunk and not fitted again: the old model
                retired.append(name)  # learned from deleted decisions, so it goes (the review of 29.7)
                fitted.pop(name, None)
            skipped_before[name] = homework[name]  # tried on this much: tried again when it has grown
    checkpoint()
    progress(step="register", of=4)
    reg = students.read(ledger, routing_store.get(), measure=measure, home=home)
    path = students.record(home, reg)
    doc.update(at=_stamp(), clock=clock, labels=seen, fitted=fitted, skipped=skipped_before, runs=int(doc.get("runs") or 0) + 1,
               ledger=ledger_id(ledger))
    doc.pop("failed_clock", None)
    doc.pop("failed_at", None)
    _save(home, doc)
    seconds = round(time.perf_counter() - began, 3)
    numbers = {"points": float(len(homework)), "retrained": float(len(cards)), "skipped": float(len(skipped)), "waiting": float(len(waiting)),
               "demoted": float(len(demoted)), "retired": float(len(retired))}
    try:  # the run as a record (19.5): what it read and made, counts and names only
        done = lineage.RunRecord(run=new_run(), kind="job", name="students_loop", status="completed", started=started, finished=_stamp(),
                                 seconds=seconds, inputs=lineage.RunInputs(refs={"sets": "own_models/sets/MANIFEST.json"}),
                                 params={"margin": MARGIN, "min_new": MIN_NEW}, code=code("dclab_rnd.students_loop"),
                                 outputs=lineage.RunOutputs(chosen=",".join(sorted(cards))[:300] or None, numbers=numbers))
        experience.record("run.recorded", "system", lineage.event_payload(done))
    except Exception as exc:  # noqa: BLE001 - the record of the run never fails the run
        print(f"students loop: run not recorded: {type(exc).__name__}", file=sys.stderr, flush=True)
    return {"at": doc["at"], "homework": homework, "retrained": sorted(cards), "cards": cards, "skipped": skipped, "waiting": waiting,
            "demoted": demoted, "retired": retired, "register": str(path), "seconds": seconds}


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    from . import audit
    from .experience import file_workspace_id
    from .ledger import open_ledger
    from .models.routing import open_routing
    from .settings import Settings

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.students_loop", description=__doc__.split("\n")[0])
    parser.add_argument("--home", type=Path, default=None, help="the workspace folder (default: DCLAB_AGENT_HOME)")
    parser.add_argument("--no-shadow", action="store_true", help="skip the shadow measurement in the register")
    parser.add_argument("--adopt", action="store_true", help="learn from this ledger even if the students learned from another one")
    args = parser.parse_args(argv)
    home = Path(args.home or Settings().agent_home)
    ledger = open_ledger(home, file_workspace_id(home))  # the workspace's own ledger, on PostgreSQL when the stores are (the review of 29.7)
    try:
        got = run(home, ledger, open_routing(home), audit.open_audit(home), measure=not args.no_shadow, adopt=args.adopt)
    except OtherLedger as exc:
        print(f"not run: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: got[k] for k in ("retrained", "skipped", "waiting", "demoted", "retired", "register", "seconds")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
