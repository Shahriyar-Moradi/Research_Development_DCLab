"""The H3 pilot: can the local model do the homework at all? (before any lesson is measured)

``run_pilot(home, model, n)`` gives the intern, served by a local model, the first ``n`` learning variants of a family
with no lesson shown, grades each, and keeps every attempt whole (``solver.intern_attempt``). Its gate was fixed before
the first pilot: the model passes when the final stage completed on at least four of five homeworks. A model that
cannot finish the work would make every later measurement of a lesson a measurement of noise.
"""

from __future__ import annotations

import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dclab_core import variants as V

from .grading import grade_run
from .packs import materialize, public_dir, read_key
from .solver import intern_attempt, keep_grade, summary_of

GATE = {"version": "pilot_gate_v1", "fixed": "2026-10-23, before the first pilot",
        "pass": "the final stage completed on at least four of five homeworks (a share of at least 0.8 for another count)",
        "measures": "whether the model can do the work at all; not whether a lesson helps (that is the probe's)",
        "invalid": "a homework that broke (DCLab's code raised, or the server never answered) makes the run invalid, never a model's fail"}
FINISH_SHARE = 0.8


def broke(session: dict[str, Any]) -> bool:
    """A session that failed before the model answered once (no step, no token back): the server or DCLab failed, the
    model never worked. A model that answered in words and did nothing is a model's fail, not this."""
    used = session.get("used") or {}
    return session.get("status") == "failed" and not (session.get("steps") or []) and not (used.get("output_tokens") or used.get("input_tokens"))


def run_pilot(home: Path, client: Any, model: str, family: str = "groups", n: int = 5, log=print) -> dict[str, Any]:
    if client is None:  # with no model the intern runs the standard plan: that would measure the rules, not the model
        raise ValueError("a pilot needs a model client: none serves the intern")
    from .scoreboard import fingerprints

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in V.seeds(family, "learning", n):  # the learning partition only: the first seeds, as the rules' baseline read
            variant = materialize(Path(tmp), family, seed)
            key = read_key(Path(tmp), family, variant.pack.id)
            started = time.monotonic()
            try:
                out = intern_attempt(public_dir(Path(tmp), family, variant.pack.id), client, key, home=home, label="pilot")
            except Exception as exc:  # noqa: BLE001 - an attempt that breaks is a row with its reason
                rows.append({"pack": variant.pack.id, "seed": seed, "broke": True, "finished": False, "verdict": "broke", "codes": [],
                             "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.monotonic() - started, 1)})
                log(f"{variant.pack.id}: broke ({type(exc).__name__})")
                continue
            grade, observed = grade_run(key, out["run"])
            keep_grade(out["kept"], grade, observed)
            rows.append({"pack": variant.pack.id, "seed": seed, "broke": broke(out["session"]), "finished": out["finished"],
                         "verdict": grade.verdict, "codes": grade.codes, "decision_score": grade.decision_score, "seconds": out["seconds"],
                         "kept": _relative(out["kept"], home), "knobs": key.knobs, "session": summary_of(out["session"])})
            log(f"{variant.pack.id}: finished {out['finished']}, {grade.verdict} {grade.codes} in {out['seconds']}s "
                f"({rows[-1]['session']['steps']} steps, {rows[-1]['session']['tool_errors']} tool errors)")
    finished = sum(r["finished"] for r in rows)
    seconds = [r["seconds"] for r in rows]
    summary = {"homeworks": len(rows), "finished": finished, "passed": sum(r["verdict"] == "pass" for r in rows),
               "mean_seconds": round(sum(seconds) / len(seconds), 1) if seconds else None,
               "codes": dict(Counter(c for r in rows for c in r["codes"]).most_common()),
               "tool_errors": sum((r.get("session") or {}).get("tool_errors", 0) for r in rows),
               "recorded_thoughts": sum(((r.get("session") or {}).get("thoughts") or {}).get("recorded", 0) for r in rows)}
    summary["broke"] = sum(bool(r.get("broke")) for r in rows)
    ok = bool(rows) and finished / len(rows) >= FINISH_SHARE
    verdict = (f"invalid: {summary['broke']} of {len(rows)} homeworks broke before the model worked (see each row's error)" if summary["broke"] or not rows
               else f"ok: the model finished {finished} of {len(rows)} homeworks" if ok
               else f"fail: the model finished only {finished} of {len(rows)} homeworks (the gate asks {FINISH_SHARE:.0%})")
    return {"campaign_id": "learning_v1", "kind": "learning_pilot", "status": "completed", "family": family, "split": "learning",
            "model": model, "gate": GATE, "fingerprints": fingerprints(), "rows": rows, "summary": summary, "verdict": verdict, "spent_eur": 0.0,
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


__all__ = ["GATE", "FINISH_SHARE", "broke", "run_pilot"]
