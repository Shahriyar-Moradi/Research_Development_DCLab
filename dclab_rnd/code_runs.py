"""Code with its execution (32.21): every piece of code the studio runs, kept whole with what it printed, what it raised
and whether it worked, so M4 and M5 learn code from what ran and what failed, not from code alone.

``record(ref, code, stdout, stderr, returncode)`` appends one line to ``<workspace>/code_runs/YYYY/MM/DD.jsonl`` (the
exported pipeline's run, a notebook cell, the intern's code). ``fixes(lines)`` pairs, for each ``ref``, a run that failed
with the next run that passed: both versions of the code, the error, and the passing output.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


def record(ref: str, code: str, stdout: str = "", stderr: str = "", returncode: int = 0, *, kind: str = "pipeline", project: Optional[str] = None,
           home: Optional[Path] = None, extra: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    """One run kept whole; None when no workspace is known. Never a reason for the run to fail."""
    try:
        from .corrections import _home

        home = Path(home) if home else _home()
        if home is None:
            return None
        now = datetime.now(timezone.utc)
        doc = {"at": now.isoformat(timespec="microseconds"), "ref": ref, "kind": kind, "project": project, "code": code, "stdout": stdout,
               "stderr": stderr, "returncode": returncode, "ok": returncode == 0, **(extra or {})}
        folder = home / "code_runs" / f"{now:%Y}" / f"{now:%m}"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / f"{now:%d}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")
        return doc
    except Exception:  # noqa: BLE001
        return None


def fixes(lines: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per ref, a failing run and the next run that passed: one fix each (the code before and after, the error, the output)."""
    out, failing = [], {}
    for run in sorted(lines, key=lambda r: str(r.get("at"))):
        ref = str(run.get("ref"))
        if not run.get("ok"):
            failing.setdefault(ref, run)
        elif ref in failing:
            bad = failing.pop(ref)
            out.append({"ref": ref, "kind": run.get("kind"), "project": run.get("project"), "before": bad.get("code"), "error": bad.get("stderr"),
                        "after": run.get("code"), "output": run.get("stdout"), "failed_at": bad.get("at"), "fixed_at": run.get("at")})
    return out


__all__ = ["record", "fixes"]
