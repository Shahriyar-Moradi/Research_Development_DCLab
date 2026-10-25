"""The collector (S2 of ``docs/guides/STUDENTS_100_PLAN.md``; the owner, 2026-10-25): real homework for the students,
gathered by itself. ``python -m dclab_rnd.collector plan|run|status|exam``.

The students trained on invented tables were not better than the rules on real ones; with real homework the column-role
student reached 99.7% against the rule's 93.6%. Real tables are what they need, and OpenML has 1,485 more usable ones
than the lake's 1,200. Each run (a job of kind ``collect``, queued by the worker's sweep once a night when
``DCLAB_COLLECTOR=1``, or by hand):

1. **Guards**: one run at a time (``real_shadow.exclusive``, shared with a hand run of the real-table shadow run); free
   disk of at least ``MIN_FREE_GB`` (the owner's rule: stop under 2 GB); the Mac under 80% busy (load and memory, the
   owner's rule); the learning agent's homework (``dclab_rnd.learning pilot|probe``) not running, so the two never load
   the Mac at once. A run stopped by a guard tries again within the hour, at night only.
2. **New tables**: the next ``tables`` OpenML candidates nobody profiled yet, open licences only, profiled and derived into
   ``COLLECTED`` (apart from the lake's 1,200, so ``real_v1`` never changes). A table whose columns are another known
   table's (OpenML keeps re-uploads under new names: ``depression_2020`` is ``depression``) is set aside, so no copy of
   a training table ever sits in a fresh exam. A network failure leaves no mark (the table is tried again) and three in
   a row end the gathering.
3. **By the lake's split** (``dclab_core.lake.split_of``): a train table is the students' homework (the real-table shadow
   run's train workspace, consent train); a dev table is measured only (its eval workspace, no consent); a sealed table
   is only profiled, for a fresh exam set (``exam``: ``real_v2``, built from the collector's tables alone once enough
   sealed tables are in, then frozen).
4. **The students whisper** on every new train and dev table (``real_shadow.run_one``): their disagreements wait in the
   people's queue (Admin, Decisions).
5. **Claude's silver opinion** on up to ``silver_per_point`` new disagreements per point, within the run's cap and what
   is left of the month's (``DCLAB_COLLECTOR_RUN_EUR``, 0.5 by default, the owner's rule for a live run;
   ``DCLAB_COLLECTOR_MONTHLY_EUR``, 10 by default, the plan's suggestion the owner kept). The run's cap is counted as
   spent before Claude is asked and settled after, so a run killed half way never spends uncounted. Evaluation and
   silver only, kept apart (the owner's rule, 2026-10-17).
6. **Measured** again on the dev tables (``real_shadow.measure``), and a **run record** in ``RUNS``: what was gathered,
   each point's homework (decisions, tables, people's labels, waiting) and what it cost.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from . import real_shadow

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "agent_runs" / "collector"  # its state and its run records (gitignored)
RUNS = HOME / "runs"
COLLECTED = real_shadow.COLLECTED
EXAM = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "real_v2"
MIN_FREE_GB = 3.0
NIGHTLY_SECONDS = 20 * 3600
RETRY_SECONDS = 3600
EXAM_MIN_SEALED_TABLES = 50  # the fresh exam set is built once this many of the collector's tables are sealed
EXAM_THIN = 30  # a point with fewer sealed tables than this has too small an exam in the set: named in the status
EXAM_POINTS = ("column.semantic_role", "column.is_identifier", "task.framing")
RUN_EUR, MONTHLY_EUR = 0.5, 10.0
BUSY = 0.8  # the owner's rule: jobs may use 80% of the Mac
TWIN = 0.8  # the column names' overlap (Jaccard) at which a table is another table's copy
NIGHT = (0, 7)  # local hours a nightly run may start in (DCLAB_COLLECTOR_HOURS="0-7"; "0-24" for any hour)


def setting(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def on() -> bool:
    return (os.environ.get("DCLAB_COLLECTOR") or "0").strip() == "1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def state(home: Path = HOME) -> dict[str, Any]:
    path = Path(home) / "state.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(home: Path, doc: dict[str, Any]) -> None:
    Path(home).mkdir(parents=True, exist_ok=True)
    tmp = Path(home) / "state.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    tmp.replace(Path(home) / "state.json")


def spent_this_month(home: Path = HOME) -> float:
    return float((state(home).get("months") or {}).get(_month(), 0.0))


# ---------------------------------------------------------------------- guards


def free_gb(path: Path = ROOT) -> float:
    return shutil.disk_usage(path).free / 1e9


def learning_busy() -> bool:
    """The learning agent's homework running (its pilot or probe): the collector waits rather than load the Mac twice."""
    try:
        out = subprocess.run(["pgrep", "-f", r"dclab_rnd\.learning (pilot|probe)"], capture_output=True, text=True, timeout=5)
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def busy() -> Optional[str]:
    """The Mac over 80% used (the owner's rule): the load above 80% of its cores, or under 20% of its memory free."""
    try:
        load, cores = os.getloadavg()[0], os.cpu_count() or 1
        if load > BUSY * cores:
            return f"the Mac is busy (load {load:.0f} on {cores} cores)"
    except OSError:
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True, timeout=5)
        free = int(out.stdout.strip())
        if free < 100 * (1 - BUSY):
            return f"memory is short ({free}% free)"
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def guards(home: Path = HOME) -> list[str]:
    """Why a run must not start now (empty when it may)."""
    why = []
    if free_gb() < MIN_FREE_GB:
        why.append(f"free disk {free_gb():.1f} GB, under {MIN_FREE_GB} GB")
    if learning_busy():
        why.append("the learning agent's homework is running")
    loaded = busy()
    if loaded:
        why.append(loaded)
    return why


# ---------------------------------------------------------------------- new tables


def new_candidates(limit: int) -> list[dict[str, Any]]:
    """The next OpenML candidates nobody profiled yet (neither the lake nor an earlier run), most-used shapes first."""
    from .lake import openml

    known = {p.stem for folder in (openml.CACHE / "datasets", COLLECTED / "datasets") if folder.is_dir() for p in folder.glob("*.json")}
    return [d for d in openml.candidates(100_000) if str(d["did"]) not in known][:limit]


def collected_records() -> Iterator[dict[str, Any]]:
    """The collector's profiled tables (an open licence, columns read)."""
    folder = COLLECTED / "datasets"
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not rec.get("skipped") and rec.get("columns"):
            yield rec


def _norm(name: Any) -> str:
    """A table's name without case, punctuation or a year (``depression_2020`` and ``depression``)."""
    import re

    return re.sub(r"[^a-z0-9]+", "", re.sub(r"(19|20)\d\d", "", str(name or "").lower()))


def known_tables() -> list[tuple[int, str, frozenset]]:
    """Every profiled table's id, normalised name and column names: the lake's and the collector's."""
    from .lake import openml

    out = []
    for folder in (openml.CACHE / "datasets", COLLECTED / "datasets"):
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            names = frozenset(str(c.get("name")) for c in rec.get("columns") or [] if c.get("name") is not None)
            if names:
                out.append((int(rec.get("did") or 0), _norm(rec.get("name")), names))
    return out


def twin_of(rec: dict[str, Any], known: list[tuple[int, str, frozenset]]) -> Optional[int]:
    """Another known table this one copies (the review of S2: OpenML re-uploads a table under a new name and id, and the
    split by id would put the copy in the exam and the original in the homework): its columns' names overlap at least
    ``TWIN``, or half of them with the same name. Ids never decide: names do, so a twin is caught whichever came first."""
    names = frozenset(str(c.get("name")) for c in rec.get("columns") or [] if c.get("name") is not None)
    if len(names) < 3:
        return None
    norm = _norm(rec.get("name"))
    for did, other_norm, other in known:
        if did == int(rec["did"]):
            continue
        overlap = len(names & other) / len(names | other)
        if overlap >= TWIN or (norm and norm == other_norm and overlap >= 0.5):
            return did
    return None


def gather(n: int, checkpoint: Callable[[], None] = lambda: None, log=print) -> Counter:
    """Profile and derive the next ``n`` new tables into ``COLLECTED``. A table that cannot be read costs itself only; a
    copy of a known table is set aside; a network failure leaves no mark, so the table is tried again, and three in a
    row end the gathering (one night offline must not use up the candidates). A profiled table whose derived record was
    lost to the network is derived again first."""
    from .lake import openml
    from .real_homework import fetch_one

    counts: Counter = Counter()
    known, in_a_row = known_tables(), 0

    def derive(rec: dict[str, Any]) -> bool:
        """The table's derived record; False when the network failed (its record removed, so it is derived again)."""
        got = fetch_one(rec, COLLECTED / "derived")
        if got.get("failed") and real_shadow.is_retry(got["failed"]):
            (COLLECTED / "derived" / f"{rec['did']}.json").unlink(missing_ok=True)
            counts["network"] += 1
            return False
        counts["failed" if got.get("failed") else f"new_{real_shadow.split_of(rec['did'])}"] += 1
        return True

    waiting = [r for r in collected_records() if not (COLLECTED / "derived" / f"{r['did']}.json").is_file()]
    for rec in waiting[:n]:
        checkpoint()
        in_a_row = 0 if derive(rec) else in_a_row + 1
        if in_a_row >= 3:
            log("gathering stopped: three network failures in a row")
            return counts
    for d in new_candidates(max(0, n - len(waiting))):
        checkpoint()
        did = int(d["did"])
        try:
            rec = openml.profile_dataset(did, cache=COLLECTED)
        except Exception as exc:  # noqa: BLE001 - a table that cannot be read is set aside; the network is tried again
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
            if real_shadow.is_retry(error):
                counts["network"] += 1
                in_a_row += 1
                if in_a_row >= 3:
                    log("gathering stopped: three network failures in a row")
                    break
                time.sleep(2.0)
                continue
            (COLLECTED / "datasets").mkdir(parents=True, exist_ok=True)
            (COLLECTED / "datasets" / f"{did}.json").write_text(json.dumps({"did": did, "skipped": error}), encoding="utf-8")
            counts["failed"] += 1
            continue
        in_a_row = 0
        if rec is None or rec.get("skipped"):
            counts["skipped"] += 1
            continue
        twin = twin_of(rec, known)
        if twin is not None:  # kept with its columns, set aside: neither homework nor exam
            rec["skipped"] = f"near-duplicate of openml:{twin}"
            (COLLECTED / "datasets" / f"{did}.json").write_text(json.dumps(rec), encoding="utf-8")
            counts["twin"] += 1
            continue
        known.append((did, _norm(rec.get("name")), frozenset(str(c.get("name")) for c in rec["columns"] if c.get("name") is not None)))
        in_a_row = 0 if derive(rec) else in_a_row + 1
        if in_a_row >= 3:
            log("gathering stopped: three network failures in a row")
            break
        time.sleep(0.2)
    log(f"gathered: {dict(counts)}")
    return counts


# ---------------------------------------------------------------------- the run


def run(home: Path = HOME, tables: Optional[int] = None, run_eur: Optional[float] = None, monthly_eur: Optional[float] = None,
        silver_per_point: int = 30, checkpoint: Callable[[], None] = lambda: None, log=print) -> dict[str, Any]:
    """One collection run; returns (and keeps) its record. Only one at a time: another run (the collector's or a hand run
    of the real-table shadow run) holding the lock makes this one a skipped run."""
    home = Path(home)
    tables = int(tables if tables is not None else setting("DCLAB_COLLECTOR_TABLES", 100))
    run_eur = float(run_eur if run_eur is not None else setting("DCLAB_COLLECTOR_RUN_EUR", RUN_EUR))
    monthly_eur = float(monthly_eur if monthly_eur is not None else setting("DCLAB_COLLECTOR_MONTHLY_EUR", MONTHLY_EUR))
    record: dict[str, Any] = {"at": _now(), "month": _month(), "settings": {"tables": tables, "run_eur": run_eur, "monthly_eur": monthly_eur,
                                                                          "silver_per_point": silver_per_point}}
    stopped = guards(home)
    if stopped:
        record.update(status="skipped", why=stopped)
        return _keep(home, record)
    try:
        with real_shadow.exclusive():
            return _run(home, record, tables, run_eur, monthly_eur, silver_per_point, checkpoint, log)
    except real_shadow.Busy as exc:
        record.update(status="skipped", why=[str(exc)])
        return _keep(home, record)


def _run(home: Path, record: dict[str, Any], tables: int, run_eur: float, monthly_eur: float, silver_per_point: int,
         checkpoint: Callable[[], None], log) -> dict[str, Any]:
    from .ledger import install, open_ledger

    record["gathered"] = dict(gather(tables, checkpoint, log))
    # the students whisper on every new train and dev table (a sealed table is never run on), each table's decisions in
    # its split's own workspace, never the server's (inside a job, real_shadow.own_books unbinds the services)
    homes = {"train": real_shadow._home(real_shadow.TRAIN_HOME, True), "dev": real_shadow._home(real_shadow.EVAL_HOME, False)}
    books = {k: open_ledger(h) for k, h in homes.items()}
    shadowed: Counter = Counter()
    with real_shadow.own_books():
        for d in real_shadow._tables():
            split = real_shadow.split_of(d["did"])
            if split not in homes or (homes[split] / "done" / f"{d['did']}.json").is_file():
                continue
            checkpoint()
            if free_gb() < MIN_FREE_GB:
                record["stopped"] = f"free disk under {MIN_FREE_GB} GB"
                break
            install(books[split])
            r = real_shadow.run_one(d, homes[split])
            shadowed[("failed_" if r.get("failed") else "done_") + split] += 1
            shadowed["decisions"] += int(r.get("decisions") or 0)
    record["shadowed"] = dict(shadowed)
    # Claude's silver opinion, within the run's cap and what is left of the month's; the cap counted before it is asked
    left = max(0.0, monthly_eur - spent_this_month(home))
    cap = round(min(run_eur, left), 4)
    spent = 0.0
    if cap >= 0.05:
        _spend(home, cap)  # reserved: a run killed or stopped half way keeps its whole cap counted
        spent = cap
        try:
            got = real_shadow.silver(real_shadow.TRAIN_HOME, silver_per_point, cap, log=log, checkpoint=checkpoint)
            spent = float(got.get("spent_eur") or 0.0)
            record["silver"] = got
        except SystemExit as exc:  # no model configured: nothing was asked, nothing spent
            spent = 0.0
            record["silver"] = {"skipped": str(exc)}
        except Exception as exc:  # noqa: BLE001 - Claude's opinion is optional; what it cost is not known, so the cap stays counted
            record["silver"] = {"failed": f"{type(exc).__name__}: {str(exc)[:200]}"}
        finally:
            _spend(home, spent - cap)  # settled (a Stop passes through here too, with the cap still counted)
    else:
        record["silver"] = {"skipped": f"the month's cap is reached ({spent_this_month(home):.2f} of {monthly_eur:.2f} EUR)"}
    record["spent_eur"] = round(spent, 4)
    try:
        measured = real_shadow.measure()
        record["measured"] = {p: {who: {k: m.get(k) for k in ("coverage", "agreement", "accuracy", "truth") if k in m}
                                  for who, m in v.items() if who != "decisions"} | {"decisions": v.get("decisions")} for p, v in measured.items()}
    except Exception as exc:  # noqa: BLE001
        record["measured"] = {"failed": f"{type(exc).__name__}: {str(exc)[:200]}"}
    record["homework"] = homework()
    record["exam"] = exam_status()
    record["status"] = "done"
    return _keep(home, record)


def homework() -> dict[str, Any]:
    """Each point's homework in the train workspace: decisions and tables recorded, how many of them carry a truth (a
    person's label, or a measured or benchmark one; never a model's), and how many disagreements wait for a person. The
    plan's bar is on the labelled ones (300 from 30 tables): decisions without a truth teach a student nothing."""
    from dclab_core.ledger import PRECEDENCE

    from .ledger import labels, open_ledger

    try:
        book = open_ledger(real_shadow.TRAIN_HOME)
        decisions, tables, table_of = Counter(), {}, {}
        for r in book.records():
            decisions[r.point] += 1
            table = r.draft or r.project or r.dataset
            tables.setdefault(r.point, set()).add(table)
            table_of[r.id] = (r.point, table)
        labelled, labelled_tables = Counter(), {}
        for x in {x.record: x for x in book.labels() if x.source in PRECEDENCE}.values():  # one truth per decision
            point, table = table_of.get(x.record, (x.point, None))
            labelled[point] += 1
            labelled_tables.setdefault(point, set()).add(table)
        waiting = dict(labels.queue(book, limit=0).get("points") or {})
    except Exception as exc:  # noqa: BLE001
        return {"failed": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return {p: {"decisions": decisions[p], "tables": len(tables.get(p, ())), "labelled": labelled[p],
                "labelled_tables": len(labelled_tables.get(p, ())), "waiting": waiting.get(p, 0),
                "bar": labelled[p] >= 300 and len(labelled_tables.get(p, ())) >= 30} for p in sorted(decisions)}


def _spend(home: Path, eur: float) -> None:
    """Add to (or, settling a reservation, take from) the month's spend, at once in the state file."""
    if not eur:
        return
    doc = state(home)
    months = doc.setdefault("months", {})
    months[_month()] = round(max(0.0, float(months.get(_month(), 0.0)) + eur), 4)
    _save(home, doc)


def _keep(home: Path, record: dict[str, Any]) -> dict[str, Any]:
    """The run's record, and the clock: a done run waits a night; a skipped one tries again within the hour."""
    doc = state(home)
    record.setdefault("spent_eur", 0.0)
    record["month_spent_eur"] = float((doc.get("months") or {}).get(_month(), 0.0))
    if record.get("status") == "done":
        doc.update(clock=time.time(), at=record["at"], last="done")
        doc.pop("failed_clock", None)
    else:
        doc.update(failed_clock=time.time(), failed_at=record["at"], last=record.get("status"))
    _save(home, doc)
    runs = Path(home) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{record['at'].replace(':', '').replace('-', '')}.json").write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    return record


# ---------------------------------------------------------------------- the fresh exam set


def exam_status() -> dict[str, Any]:
    """The sealed tables a fresh exam would hold, per point: every derived table for the column's role, only those that
    declare a row identifier for the identifier question, those with a task for the task (a failed derivation counts
    for none). The set is for the role student's second exam: ready once its point has enough; a thinner point is named."""
    per: Counter = Counter()
    for rec in collected_records():
        if real_shadow.split_of(rec["did"]) != "sealed":
            continue
        try:
            got = json.loads((COLLECTED / "derived" / f"{rec['did']}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if got.get("failed") or not got.get("columns"):
            continue
        per["column.semantic_role"] += 1
        per["column.is_identifier"] += any(c.get("is_row_identifier") for c in rec.get("columns") or [])
        per["task.framing"] += bool(got.get("task"))
    built = (EXAM / "MANIFEST.json").is_file()
    return {"sealed_tables": per["column.semantic_role"], "per_point": {p: per[p] for p in EXAM_POINTS}, "needed": EXAM_MIN_SEALED_TABLES,
            "thin": [p for p in EXAM_POINTS if per[p] < EXAM_THIN], "built": built,
            "ready": built or per["column.semantic_role"] >= EXAM_MIN_SEALED_TABLES}


def build_exam(force: bool = False) -> dict[str, Any]:
    """``real_v2`` from the collector's tables alone (none of real_v1's, whose sealed partition is spent for roles), once
    enough are sealed; then frozen: a later set would be real_v3. Its sealed fingerprints are what a gate checks."""
    from .real_homework import build

    status = exam_status()
    if status["built"]:
        return {**status, "note": "real_v2 is built and frozen"}
    if not force and status["sealed_tables"] < EXAM_MIN_SEALED_TABLES:
        return {**status, "note": f"waiting for {EXAM_MIN_SEALED_TABLES} sealed tables"}
    summary = build(EXAM, COLLECTED / "derived", datasets=list(collected_records()), version="real_v2", at=_now())
    return {**exam_status(), "sets": {p: {k: v for k, v in s.items() if k in ("decisions", "tables", "splits")} for p, s in summary["sets"].items()}}


# ---------------------------------------------------------------------- the nightly sweep


def night(now: float) -> bool:
    """The local hour inside ``DCLAB_COLLECTOR_HOURS`` (``0-7`` by default: a nightly run never starts in the day)."""
    try:
        start, end = (int(x) for x in (os.environ.get("DCLAB_COLLECTOR_HOURS") or f"{NIGHT[0]}-{NIGHT[1]}").split("-"))
    except ValueError:
        start, end = NIGHT
    return start <= datetime.fromtimestamp(now).hour < end


def due(home: Path = HOME, now: Optional[float] = None) -> bool:
    """Switched on, at night, a night since the last done run, and not within an hour of a skipped or failed one."""
    if not on():
        return False
    clock = now if now is not None else time.time()
    if not night(clock):
        return False
    doc = state(home)
    failed_at = doc.get("failed_clock")
    if isinstance(failed_at, (int, float)) and clock - failed_at < RETRY_SECONDS:
        return False
    last = doc.get("clock")
    return not isinstance(last, (int, float)) or clock - last >= NIGHTLY_SECONDS


def failed(home: Path = HOME, now: Optional[float] = None, reason: str = "failed") -> None:
    """A run that did not finish: tried again within the hour, unless a person stopped it (then it waits a night)."""
    doc = state(home)
    clock = now if now is not None else time.time()
    if reason == "stopped":
        doc.update(clock=clock, at=_now(), last="stopped")
    else:
        doc.update(failed_clock=clock, failed_at=_now(), last=reason)
    _save(home, doc)


def sweep_due(services: Any, now: Optional[float] = None) -> Optional[dict[str, Any]]:
    """The job, queued when a run is due (one at a time: a queued or running one holds the key)."""
    from .jobs import ActiveJob

    if not due(HOME, now):
        return None
    try:
        return services.queue_job("collect", "nightly", {}, by="agent")
    except ActiveJob:
        return None


def plan(home: Path = HOME) -> dict[str, Any]:
    """What a run would do now, without doing it."""
    return {"switched_on": on(), "due": due(home), "guards": guards(home), "new_tables_available": len(new_candidates(100_000)),
            "month_spent_eur": spent_this_month(home), "monthly_cap_eur": setting("DCLAB_COLLECTOR_MONTHLY_EUR", MONTHLY_EUR),
            "run_cap_eur": setting("DCLAB_COLLECTOR_RUN_EUR", RUN_EUR), "tables_per_run": int(setting("DCLAB_COLLECTOR_TABLES", 100)),
            "homework": homework(), "exam": exam_status()}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dclab_rnd.collector", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("plan", "run", "status", "exam"))
    ap.add_argument("--tables", type=int)
    ap.add_argument("--run-eur", type=float)
    ap.add_argument("--silver-per-point", type=int, default=30)
    ap.add_argument("--force", action="store_true", help="exam: build real_v2 now, before enough tables are sealed")
    args = ap.parse_args(argv)
    real_shadow.load_env()  # the model settings live in .env, as the server reads them
    if args.command == "plan":
        out: Any = plan()
    elif args.command == "run":
        out = run(tables=args.tables, run_eur=args.run_eur, silver_per_point=args.silver_per_point)
    elif args.command == "exam":
        out = build_exam(args.force)
    else:
        out = {"state": state(), "last_runs": sorted(p.name for p in RUNS.glob("*.json"))[-5:] if RUNS.is_dir() else []}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
