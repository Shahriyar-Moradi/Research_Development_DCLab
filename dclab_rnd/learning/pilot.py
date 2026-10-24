"""The H3 pilot: can the local model do the homework at all? (before any lesson is measured)

``run_pilot(home, model, n)`` gives the intern, served by a local model, the first ``n`` learning variants of a family
with no lesson shown, grades each, and keeps every attempt whole (``solver.intern_attempt``). Its gate was fixed before
the first pilot: the model passes when the final stage completed on at least four of five homeworks. A model that
cannot finish the work would make every later measurement of a lesson a measurement of noise.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dclab_core import variants as V

from .grading import grade_run
from .packs import materialize, public_dir, read_key
from .solver import BUDGET, intern_attempt, keep_grade, summary_of

GATE = {"version": "pilot_gate_v1", "fixed": "2026-10-23, before the first pilot",
        "pass": "the final stage completed on at least four of five homeworks (a share of at least 0.8 for another count)",
        "measures": "whether the model can do the work at all; not whether a lesson helps (that is the probe's)",
        "invalid": "a homework that broke (DCLab's code raised, or the server never answered) makes the run invalid, never a model's fail"}
FINISH_SHARE = 0.8
PAUSE = "PAUSE_PILOT"  # a file in <home>/learning_attempts/: the pilot stops after the homework it is on


def pause_file(home: Path) -> Path:
    return Path(home) / "learning_attempts" / PAUSE


def done_row(home: Path, family: str, seed: int, model: str) -> dict[str, Any] | None:
    """A homework this model already did and the grader graded, read from its kept folder, so a paused pilot goes on
    where it stopped. Matched on the family and the seed (a pack's id is drawn fresh each time, so the solver cannot
    tell variants apart by it). A folder marked invalid, or with no grade (stopped mid-way), does not count."""
    for saved in sorted((Path(home) / "learning_attempts").glob("*/*-pilot/pilot_row.json"), reverse=True):
        folder = saved.parent
        if (folder / "invalid.json").exists() or not (folder / "grade.json").exists():
            continue
        row = json.loads(saved.read_text(encoding="utf-8"))
        if row.get("model") == model and row.get("family", "groups") == family and row.get("seed") == seed:
            return row
    return None


def rebuild_row(folder: Path, home: Path, family: str, seed: int) -> dict[str, Any]:
    """The pilot row of an attempt kept before rows were written beside it (2026-10-23), from its grade, session and
    project; its seed from the run's order. Written as ``pilot_row.json`` so the pilot finds it."""
    from dclab_rnd.studio import ProjectStore

    folder = Path(folder)
    session = json.loads(sorted((folder / "intern").glob("*.json"))[0].read_text(encoding="utf-8"))
    grade = json.loads((folder / "grade.json").read_text(encoding="utf-8"))["grade"]
    finished = False
    if session.get("project_id"):
        try:
            finished = (ProjectStore(folder / "projects").read_stage(session["project_id"], "final") or {}).get("status") == "completed"
        except (KeyError, FileNotFoundError, ValueError):
            finished = False
    seconds = None
    if session.get("created") and session.get("updated"):
        seconds = round((datetime.fromisoformat(session["updated"]) - datetime.fromisoformat(session["created"])).total_seconds(), 1)
    row = {"pack": folder.name.split("-", 1)[1].removesuffix("-pilot"), "seed": seed, "family": family, "model": session.get("model"),
           "broke": broke(session), "finished": finished, "verdict": grade["verdict"], "codes": grade["codes"],
           "decision_score": grade.get("decision_score"), "seconds": seconds, "kept": _relative(str(folder), home),
           "session": summary_of(session), "rebuilt": True}
    (folder / "pilot_row.json").write_text(json.dumps(row, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return row


def broke(session: dict[str, Any]) -> bool:
    """A session that failed before the model answered once (no step, no token back): the server or DCLab failed, the
    model never worked. A model that answered in words and did nothing is a model's fail, not this."""
    used = session.get("used") or {}
    return session.get("status") == "failed" and not (session.get("steps") or []) and not (used.get("output_tokens") or used.get("input_tokens"))


def run_pilot(home: Path, client: Any, model: str, family: str = "groups", n: int = 5, log=print,
              budget: dict[str, Any] | None = None, limit: int | None = None, cap_eur: float | None = None) -> dict[str, Any]:
    """``limit``: at most this many new homeworks in this process, then it pauses (the next process goes on). Measured
    2026-10-24: one process doing homework after homework grew until swap reached 10 GB and the disk fell below 2 GB;
    a fresh process a homework gives its memory back."""
    if client is None:  # with no model the intern runs the standard plan: that would measure the rules, not the model
        raise ValueError("a pilot needs a model client: none serves the intern")
    from .scoreboard import fingerprints

    rows, paused, new = [], False, 0
    with tempfile.TemporaryDirectory() as tmp:
        for seed in V.seeds(family, "learning", n):  # the learning partition only: the first seeds, as the rules' baseline read
            earlier = done_row(home, family, seed, model)
            if earlier is not None:  # done before a pause: kept, graded, not done again
                rows.append({**earlier, "resumed": True})
                log(f"{earlier['pack']}: done before ({earlier['verdict']} {earlier['codes']}), not repeated")
                continue
            variant = materialize(Path(tmp), family, seed)
            spent = sum(float(((r.get("session") or {}).get("eur")) or 0.0) for r in rows)
            if cap_eur is not None and spent >= cap_eur:  # a hosted model: the cap holds for the whole pilot, its earlier homeworks too
                paused = True
                log(f"the cap of EUR {cap_eur:.2f} is spent (EUR {spent:.4f}): stopped before {variant.pack.id}")
                break
            if pause_file(home).exists() or (limit is not None and new >= limit):
                if pause_file(home).exists():
                    pause_file(home).unlink()
                paused = True
                log(f"paused before {variant.pack.id}: run the same command again to go on")
                break
            new += 1
            key = read_key(Path(tmp), family, variant.pack.id)
            started = time.monotonic()
            try:
                out = intern_attempt(public_dir(Path(tmp), family, variant.pack.id), client, key, home=home, label="pilot", budget=budget)
            except Exception as exc:  # noqa: BLE001 - an attempt that breaks is a row with its reason
                rows.append({"pack": variant.pack.id, "seed": seed, "broke": True, "finished": False, "verdict": "broke", "codes": [],
                             "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.monotonic() - started, 1)})
                log(f"{variant.pack.id}: broke ({type(exc).__name__})")
                continue
            grade, observed = grade_run(key, out["run"])
            keep_grade(out["kept"], grade, observed)
            rows.append({"pack": variant.pack.id, "seed": seed, "family": family, "model": model, "broke": broke(out["session"]), "finished": out["finished"],
                         "verdict": grade.verdict, "codes": grade.codes, "decision_score": grade.decision_score, "seconds": out["seconds"],
                         "kept": _relative(out["kept"], home), "knobs": key.knobs, "session": summary_of(out["session"])})
            if out["kept"]:  # the row beside its attempt at once: a pause or a crash later never loses it
                (Path(out["kept"]) / "pilot_row.json").write_text(json.dumps(rows[-1], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
                if rows[-1]["broke"]:  # the server or DCLab failed before the model answered: done again, never the model's fail in training
                    from .probe import mark_invalid

                    mark_invalid(Path(out["kept"]), "the session failed before the model answered (the server or DCLab broke)")
            log(f"{variant.pack.id}: finished {out['finished']}, {grade.verdict} {grade.codes} in {out['seconds']}s "
                f"({rows[-1]['session']['steps']} steps, {rows[-1]['session']['tool_errors']} tool errors)")
    finished = sum(r["finished"] for r in rows)
    seconds = [r["seconds"] for r in rows if isinstance(r.get("seconds"), (int, float))]
    summary = {"homeworks": len(rows), "finished": finished, "passed": sum(r["verdict"] == "pass" for r in rows),
               "mean_seconds": round(sum(seconds) / len(seconds), 1) if seconds else None,
               "codes": dict(Counter(c for r in rows for c in r["codes"]).most_common()),
               "tool_errors": sum((r.get("session") or {}).get("tool_errors", 0) for r in rows),
               "recorded_thoughts": sum(((r.get("session") or {}).get("thoughts") or {}).get("recorded", 0) for r in rows)}
    summary["broke"] = sum(bool(r.get("broke")) for r in rows)
    summary["spent_eur"] = round(sum(float(((r.get("session") or {}).get("eur")) or 0.0) for r in rows), 6)
    ok = bool(rows) and finished / len(rows) >= FINISH_SHARE
    if paused:
        return {"campaign_id": "learning_v1", "kind": "learning_pilot", "status": "paused", "family": family, "model": model, "rows": rows,
                "summary": summary, "verdict": f"paused after {len(rows)} of {n} homeworks: run the same command again to go on"}
    verdict = (f"invalid: {summary['broke']} of {len(rows)} homeworks broke before the model worked (see each row's error)" if summary["broke"] or not rows
               else f"ok: the model finished {finished} of {len(rows)} homeworks" if ok
               else f"fail: the model finished only {finished} of {len(rows)} homeworks (the gate asks {FINISH_SHARE:.0%})")
    return {"campaign_id": "learning_v1", "kind": "learning_pilot", "status": "completed", "family": family, "split": "learning",
            "model": model, "budget": {**BUDGET, **(budget or {})}, "gate": GATE, "fingerprints": fingerprints(), "rows": rows, "summary": summary, "verdict": verdict, "spent_eur": summary["spent_eur"],
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rule": "the first learning variants, no lesson shown, the intern on a local model, graded by code, every attempt kept whole"}


def _relative(path: str | None, home: Path) -> str | None:
    """A kept attempt's folder under the workspace, never an absolute local path in committed evidence."""
    if not path:
        return None
    try:
        return str(Path(path).resolve().relative_to(Path(home).resolve()))
    except ValueError:
        return Path(path).name


__all__ = ["GATE", "FINISH_SHARE", "PAUSE", "pause_file", "done_row", "rebuild_row", "broke", "run_pilot"]
