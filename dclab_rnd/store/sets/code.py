"""36.6: verified code for the solver model (m5). A kept ``make code-check`` run (``scripts/code_check.py --keep DIR``) holds,
per engine kind, the exported pipeline, the replica's problem (``problem.json``), what it recorded and what a clean
Python 3.12 reproduced. Each kind whose pipeline reproduced its recorded number becomes one example: the problem as a
person states it (kind, moment of prediction, target, the solution's columns and roles, the table's columns; never a
result) and the code (pipeline.py). A kind is taken from the newest finished run in which it passed, so a partial or a
failed later run never removes it.

The code was written by DCLab's engine from the project's records, not by a model, so its assistant turn carries no
reasoning: the owner's rule (2026-10-28) keeps only a model's own thinking summary, and never adds code-written or
teacher reasoning. The turn says who wrote the code instead. The split is the replica's own group's (the group self-play
solves the same table under), the stricter of the two split rules, so a table sealed for self-play is sealed here too.

    python -m dclab_rnd.store.sets.code agent_runs/code_runs/<run>    # writes sets/m5_solver_llm/verified_code/v1
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dclab_core import lake
from dclab_core.datasets import split_of
from dclab_core.store import ChatExample, Message, Provenance, SetManifest

from ..layout import Layout
from . import write_set

WHO_WROTE = ("written by DCLab's engine from the project's records (the kind's replica run end to end, no model asked), and "
             "re-run to its recorded number in a clean Python; no model thought it, so no reasoning is given")
ASK = ("Write one Python pipeline for this problem. It reads the data kept beside it (the rows as the project prepared them, "
       "the holdout already locked), fits as the solution says, and prints the holdout number.\n\n")
RUNS = Path(__file__).resolve().parents[3] / "agent_runs" / "code_runs"  # where scripts/code_check.py --keep is pointed
ORDER = {"train": 0, "dev": 1, "sealed": 2}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_for(key: str) -> str:
    """The replica's own group (self-play solves case 0 as ``selfplay:<kind>:0``), by the stricter of the two split rules."""
    group = f"selfplay:{key}:0"
    return max(split_of(group), lake.split_of(group), key=ORDER.__getitem__)


def problem_text(problem: dict[str, Any]) -> str:
    """The question, from the replica's own description: what is predicted, when, and from which columns. No result."""
    lines = [f"Kind of problem: {problem.get('kind')} ({problem.get('task')})."]
    if problem.get("moment"):
        lines.append(f"The prediction is made: {problem['moment']}")
    if problem.get("target"):
        lines.append(f"Target: {problem['target']}.")
    if problem.get("columns"):
        lines.append("Columns: " + ", ".join(problem["columns"]) + ".")
    if problem.get("solution"):
        lines.append("The solution as agreed: " + json.dumps(problem["solution"], ensure_ascii=False, sort_keys=True, default=str))
    return "\n".join(lines) + "\n"


def _finished(run: Path) -> Optional[dict[str, Any]]:
    path = run / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _example(run: Path, folder: Path, summary: dict[str, Any]) -> Optional[ChatExample]:
    result_path, code, problem_path = folder / "result.json", folder / "pipeline.py", folder / "problem.json"
    if not (result_path.exists() and code.exists() and problem_path.exists()):
        return None  # no stated problem (a run from before problem.json): never a question made from the README
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("same") is not True:
        return None  # it did not reproduce its number: not verified, not an example
    key = str(result.get("kind") or folder.name)
    at = datetime.fromtimestamp(result_path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    prov = Provenance(workspace="code_check", source=f"code_runs/{run.name}/{folder.name}/pipeline.py", entry=f"{run.name}:{folder.name}",
                      at=at, sha256=_sha(code), consent="train")
    return ChatExample(id=f"m5_solver_llm:verified_code:{key}", model_type="m5_solver_llm", split=split_for(key), provenance=prov,
                       task="verified_pipeline", stage="code",
                       messages=[Message(role="user", content=ASK + problem_text(json.loads(problem_path.read_text(encoding="utf-8")))),
                                 Message(role="assistant", content=code.read_text(encoding="utf-8"), no_reasoning=WHO_WROTE)],
                       check={"verified": True, "kind": key, "metric": result.get("metric"), "recorded": result.get("recorded"),
                              "reproduced": result.get("reproduced"), "python": summary.get("python"), "run": run.name})


def examples(run: Path) -> list[ChatExample]:
    """One run's verified pipelines as m5 examples; a run without its summary (cut short) is refused."""
    run = Path(run)
    summary = _finished(run)
    if summary is None:
        raise ValueError(f"{run} has no summary.json: the code-check run did not finish, so nothing in it is taken as verified")
    return [ex for folder in sorted(p for p in run.iterdir() if p.is_dir()) if (ex := _example(run, folder, summary)) is not None]


def latest(runs: Path) -> list[ChatExample]:
    """Every kind from the newest finished run in which it passed (by the run's own finish stamp, else its summary's
    time), so a later partial or failed run never removes a kind verified before."""
    found = []
    for run in runs.glob("*") if runs.exists() else []:
        summary = _finished(run) if run.is_dir() else None
        if summary is None:
            continue
        stamp = summary.get("finished_at") or datetime.fromtimestamp((run / "summary.json").stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        found += [(stamp, ex) for folder in run.iterdir() if folder.is_dir() and (ex := _example(run, folder, summary)) is not None]
    best: dict[str, tuple[str, ChatExample]] = {}
    for stamp, ex in found:
        if ex.id not in best or stamp > best[ex.id][0]:
            best[ex.id] = (stamp, ex)
    return [ex for _, ex in sorted(best.values(), key=lambda x: x[1].id)]


def _write(layout: Layout, found: list[ChatExample]) -> SetManifest:
    return write_set(layout, "m5_solver_llm", "verified_code", found, sources=[e.provenance.model_dump() for e in found],
                     about="Exported pipelines that reproduced their recorded holdout number in a clean Python 3.12 (36.6): the "
                           "problem as stated, then the code. The code is DCLab's engine's, so no reasoning is given.")


def build(layout: Optional[Layout] = None, run: Optional[Path] = None) -> SetManifest:
    """Write one kept code-check run's verified pipelines as m5's ``verified_code`` set."""
    return _write(layout or Layout(), examples(Path(run)))


def runs_for(layout: Layout) -> Path:
    """Where the kept runs are: DCLAB_CODE_RUNS, else the workspace's agent_runs/code_runs for the real store, else a
    folder inside the layout (a test's store reads no one's real runs)."""
    if os.environ.get("DCLAB_CODE_RUNS"):
        return Path(os.environ["DCLAB_CODE_RUNS"])
    return RUNS if layout.root.resolve() == Layout().root.resolve() else layout.code_runs()


def build_latest(layout: Optional[Layout] = None, runs: Optional[Path] = None) -> dict[str, Any]:
    """The store build's step: every kind from its newest finished run in which it passed; no run yet, no set."""
    layout = layout or Layout()
    found = latest(runs if runs is not None else runs_for(layout))
    if not found:
        return {"sets": {}}
    m = _write(layout, found)
    return {"sets": {"m5_solver_llm/verified_code": m.counts}, "runs": sorted({e.check["run"] for e in found})}


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Name one kept code-check run folder (scripts/code_check.py --keep DIR).", file=sys.stderr)
        return 2
    m = build(run=Path(args[0]))
    print(json.dumps({"set": "m5_solver_llm/verified_code", "counts": m.counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
