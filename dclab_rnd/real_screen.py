"""Real tables screened by DCLab's engine (S7 of ``docs/guides/STUDENTS_100_PLAN.md``, "more ability"; the owner,
2026-10-25): ``python -m dclab_rnd.real_screen run|status``.

An expert's main skill is guessing which model family will win on a table before trying them all. Self-play's screens
teach that badly (each generator has its usual winner), so the homework is real: every OpenML table that declares an
outcome (the lake's and the collector's), solved by DCLab's own engine with no model asked (``kinds.acceptance.
engine_run``: the proposal accepted, every stage under its gates), its decisions recorded, among them ``model.first``
with the table at a glance (``ledger.states.table_state``), and every family's training-fold score kept in
``screens.jsonl``: the truth a student is judged by (its pick's regret against the best, in units of the folds' noise).

Each table sits in the lake's split (``dclab_core.lake.split_of``, read through ``dclab_core.datasets.split_of`` as the
store reads it): train and dev tables in their own folders, sealed tables in a folder of their own that no training
reads, for the student's one exam. A table larger than ``MAX_ROWS`` is
screened on a seeded sample of its rows. Resumable (a done marker per table); a run waits while the Mac is over 80% busy
or the disk has under 3 GB free (the owner's rules); one table that cannot be read or solved costs itself only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "agent_runs" / "real_screens"  # gitignored; the archive keeps it (archive.SOURCES)
MAX_ROWS = 20_000
MIN_FREE_GB = 3.0
BUSY = 0.8
SPLITS = ("train", "dev", "sealed")
MOMENT = "When a new row arrives, before its outcome is known."  # a real table states none; this one names no column
PROFILE = "full"  # the product's own model settings (never acceptance's fast test profile)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tables() -> list[dict[str, Any]]:
    """Every real table's derived record that declares an outcome (the lake's and the collector's), each once."""
    from . import real_shadow

    out, seen = [], set()
    for folder in real_shadow.DERIVED_DIRS:
        for path in sorted(folder.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0) if folder.is_dir() else []:
            if path.stem in seen:
                continue
            seen.add(path.stem)
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not d.get("failed") and d.get("target") and d.get("columns"):
                out.append(d)
    return out


def _scrub(text: str) -> str:
    """An error's words without what it quotes (pandas quotes the cell it could not read): never a table's value."""
    import re

    return re.sub(r"'[^']*'|\"[^\"]*\"", "'…'", text)


def room() -> Optional[str]:
    """Why the Mac has no room for a table now, or None: over 80% of its cores or of its memory in use, or its disk short."""
    import subprocess

    load, cores = os.getloadavg()[0], os.cpu_count() or 1
    if load > BUSY * cores:
        return f"the Mac is busy (load {load:.0f} on {cores} cores)"
    try:
        free_memory = int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True, timeout=5).stdout.strip())
        if free_memory < 100 * (1 - BUSY):
            return f"memory is short ({free_memory}% free)"
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < MIN_FREE_GB:
        return f"free disk {free:.1f} GB, under {MIN_FREE_GB} GB"
    return None


def task_of(frame: Any, target: str) -> str:
    from .studio.data import detect_task

    return str(detect_task(frame, target).get("task") or "binary")


def dataset_of(did: Any) -> str:
    """The table's name in every record of its screen: ``openml:<id>`` for a public table, ``craft:<seed>`` for an
    invented one (benchgen v3)."""
    s = str(did)
    return f"craft:{s.removeprefix('craft')}" if s.startswith("craft") else f"openml:{s}"


def split_for(did: Any) -> str:
    """The table's split, the one the store gives its records (``dclab_core.datasets.split_of`` of its dataset: the
    lake's for a public table), so a table measured on here is never trained on there (S8)."""
    from dclab_core.datasets import split_of

    return split_of(f"dataset:{dataset_of(did)}")


def screen_one(derived: dict[str, Any], home: Path = HOME, download: Optional[Callable[[Any], Any]] = None) -> dict[str, Any]:
    """One real table through the engine, its ledger and its screen kept in its split's folder; every decision of the
    run carries the table as its dataset."""
    from . import ledger, real_shadow
    from .kinds import acceptance as A

    did = derived["did"]
    split = split_for(did)
    folder = Path(home) / split
    marker = folder / "done" / f"{did}.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    marker.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        frame = (download or real_shadow._download)(did)
        if len(frame) > MAX_ROWS:
            frame = frame.sample(MAX_ROWS, random_state=0).sort_index().reset_index(drop=True)  # the file's own order kept
        target = derived["target"]
        case = A.Case(table=frame, target=target, task=task_of(frame, target), moment=MOMENT)
        book = ledger.FileLedger(folder / "ledgers" / str(did) / "ledger", f"real-screen-{split}")  # where the store reads a workspace's decisions
        book.home = folder / "ledgers" / str(did)
        before = ledger._INSTALLED.get("ledger")
        ledger.install(book)
        try:
            with ledger.dataset_scope(dataset_of(did)):
                run = A.engine_run(case, fast=False)  # the product's own settings: the test profile's few trees change who wins
        finally:
            ledger.install(before)
        models, features = run.evidence("models"), run.evidence("features")
        metric = (run.records.get("models") or {}).get("primary_metric")
        recipes = [{"recipe": r.get("recipe"), "mean": ((r.get("metrics") or {}).get(metric) or {}).get("mean"),
                    "std": ((r.get("metrics") or {}).get(metric) or {}).get("std"), "features": r.get("feature_count_mean")}
                   for r in features.get("stage_results") or [] if isinstance(r, dict)]  # S7: which recipe first, from the same run
        first = next(iter(book.records(point="model.first")), None)
        record = {"did": did, "split": split, "rows": int(len(frame)), "target": target, "task": case.task, "at": _now(), "profile": PROFILE,
                  "metric": (run.records.get("models") or {}).get("primary_metric"),
                  "state": dict(first.state) if first is not None else None, "first": first.value_used if first is not None else None,
                  "screen": [{k: r.get(k) for k in ("model", "mean", "std", "elapsed_seconds")} for r in models.get("ranking") or [] if isinstance(r, dict)],
                  "selected": models.get("selected_model"), "recipes": recipes, "recipe_selected": features.get("selected_recipe"),
                  "recipe_model": features.get("stage_model"), "seconds": round(time.perf_counter() - started, 1),
                  **({"facts": derived["facts"]} if derived.get("facts") else {})}  # a craft table's planted structure: a fact, never a label
        with (folder / "screens.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        out = {"did": did, "split": split, "families": len(record["screen"]), "seconds": record["seconds"]}
    except Exception as exc:  # noqa: BLE001 - one table that cannot be read or solved costs itself only
        out = {"did": did, "split": split, "failed": f"{type(exc).__name__}: {_scrub(str(exc))[:200]}"}
        if any(k in out["failed"] for k in real_shadow.RETRY):
            return out  # the network: no mark, tried again on the next run
    marker.write_text(json.dumps(out) + "\n", encoding="utf-8")
    return out


def run(home: Path = HOME, limit: Optional[int] = None, wait: bool = True, log=print) -> dict[str, int]:
    """Every table not screened yet (at most ``limit``), waiting for room between tables."""
    counts: Counter = Counter()
    todo = [d for d in tables() if not (Path(home) / split_for(d["did"]) / "done" / f"{d['did']}.json").is_file()]
    for d in todo[:limit] if limit else todo:
        why = room()
        while why and wait:
            log(f"{_now()} waiting: {why}")
            time.sleep(120)
            why = room()
        if why:
            counts["no_room"] += 1
            break
        got = screen_one(d, home)
        counts[("failed_" if got.get("failed") else "done_") + got["split"]] += 1
        log(f"{_now()} {d['did']} {got}")
    return dict(counts)


CRAFT_HOME = ROOT / "agent_runs" / "craft_screens"


def craft(seeds: range, home: Path = CRAFT_HOME, wait: bool = True, log=print) -> dict[str, int]:
    """benchgen v3's craft tables through the same engine (their winners measured, not planted), in their own folders by
    the same split rule, so a craft table never sits in two splits."""
    from dclab_core import benchgen3

    counts: Counter = Counter()
    for seed in seeds:
        frame, target, facts = benchgen3.craft_table(seed)
        why = room()
        while why and wait:
            log(f"{_now()} waiting: {why}")
            time.sleep(120)
            why = room()
        if why:
            counts["no_room"] += 1
            break
        got = screen_one({"did": f"craft{seed}", "target": target, "facts": facts}, home, download=lambda did, f=frame: f)
        counts[("failed_" if got.get("failed") else "done_") + got["split"]] += 1
        log(f"{_now()} craft{seed} {got}")
    return dict(counts)


def status(home: Path = HOME) -> dict[str, Any]:
    out: dict[str, Any] = {"tables_with_an_outcome": len(tables())}
    for split in SPLITS:
        folder = Path(home) / split
        done = list((folder / "done").glob("*.json")) if (folder / "done").is_dir() else []
        failed = sum(1 for p in done if "failed" in p.read_text(encoding="utf-8"))
        screens = sum(1 for _ in (folder / "screens.jsonl").open()) if (folder / "screens.jsonl").is_file() else 0
        out[split] = {"screened": screens, "failed": failed}
    return out


def main(argv: Optional[list[str]] = None) -> int:
    from .real_shadow import load_env

    ap = argparse.ArgumentParser(prog="python -m dclab_rnd.real_screen", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("run", "status", "craft"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seeds", type=int, default=600, help="craft: benchgen v3's craft tables 0 to N-1")
    args = ap.parse_args(argv)
    load_env()
    os.environ.setdefault("DCLAB_NO_LIVE_MODELS", "1")  # the engine asks no model here; nothing can reach one
    out = run(limit=args.limit) if args.command == "run" else craft(range(args.seeds)) if args.command == "craft" else {"real": status(), "craft": status(CRAFT_HOME)}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
