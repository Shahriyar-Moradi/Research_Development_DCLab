"""The evaluation dashboard's data (package 25.8): how Claude and the whole system perform, and every answerer's
precision, recall, F1 and accuracy, from what this workspace recorded.

Two sources, each read where it is kept:

* the twin folder (``<workspace>/twin``): the Home sessions' steps (a person's pick is their truth), the twin runs'
  steps and expert steps (a blind seed's planted truth);
* the decision ledger: every answer on every labelled decision (the rule, the roles, the classifiers in shadow, an
  AI-first answer), against the record's effective label.

:func:`board` returns the library's ``EvaluationBoard`` (every answerer by point) and the twin matrix
(``dclab_core.twin.effectiveness``: Claude against the system, step by step, with the mode of ``twin_rule_v1``).
A step appended again (a pick) replaces its earlier line of the same id. Nothing is computed here but the reading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _lines(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def twin_steps(home: Path, expert: bool = True) -> list[Any]:
    """Every twin step of the workspace, the latest line of each id; ``expert=False`` leaves out the expert records'
    steps (their system side repeats the run's own steps)."""
    from dclab_core import twin

    root = Path(home) / "twin"
    files = []
    if root.is_dir():
        files += sorted((root / "sessions").rglob("steps.jsonl")) if (root / "sessions").is_dir() else []
        runs = root / "runs"
        if runs.is_dir():
            files += sorted(runs.glob("*/steps.jsonl")) + (sorted(runs.glob("*/expert_steps.jsonl")) if expert else [])
    latest: dict[str, Any] = {}
    for path in files:
        for row in _lines(path):
            try:
                step = twin.TwinStep.read(row)
            except Exception:  # noqa: BLE001 — a line that does not read is not evidence
                continue
            latest[f"{path.parent}:{step.id}"] = step
    return list(latest.values())


def testers(home: Path) -> list[Path]:
    """25.11: the testers' workspaces the owner collected (``<home>/testers/<id>/``)."""
    root = Path(home) / "testers"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def board(home: Path, ledger: Any = None, with_testers: bool = True) -> dict[str, Any]:
    """The workspace's own records, and (25.11) every collected tester's beside them."""
    from dclab_core import evaluation, twin

    from .ledger import open_ledger

    folders = testers(home) if with_testers else []
    steps = twin_steps(home) + [s for f in folders for s in twin_steps(f)]
    boards = [evaluation.from_twin_steps(steps, "the twin folders")]
    records: list[Any] = []
    labels: list[Any] = []
    books = ([ledger] if ledger is not None else []) + [open_ledger(f) for f in folders if (f / "ledger").is_dir()]
    for book in books:
        try:
            records += list(book.records())
            labels += list(book.labels())
        except Exception:  # noqa: BLE001 - a ledger that cannot be read costs its own scores only
            continue
    if books:
        boards.append(evaluation.from_ledger(records, labels, "the decision ledgers"))
    merged = evaluation.merge(boards, "the twin folders and the decision ledgers" + (f", with {len(folders)} testers" if folders else ""))
    matrix = twin.effectiveness(twin_steps(home, expert=False) + [s for f in folders for s in twin_steps(f, expert=False)],  # each system answer once
                                "the twin folders: Home sessions (a person's pick: a preference between the two) and twin runs (a blind seed's planted truth)")
    return {"board": merged.model_dump(mode="json"), "twin": matrix.model_dump(mode="json"),
            "sources": {"twin_steps": len(steps), "judged_twin_steps": int(sum(r.judged for r in matrix.steps)),
                        "decisions": len(records), "labels": len(labels), "testers": len(folders)}}


def summary(result: dict[str, Any]) -> dict[str, Any]:
    """The four numbers the page leads with."""
    scores = result["board"]["scores"]
    return {"answerers": len({s["answerer"] for s in scores}), "points": len({s["point"] for s in scores}),
            "cases": int(result["board"]["totals"].get("cases", 0)), "judged_twin_steps": result["sources"]["judged_twin_steps"]}


def best(result: dict[str, Any], point: str) -> Optional[dict[str, Any]]:
    """The answerer with the highest F1 (then accuracy) at a point, among those with at least ten cases."""
    rows = [s for s in result["board"]["scores"] if s["point"] == point and s["n"] >= 10]
    rows.sort(key=lambda s: ((s.get("f1") if s.get("f1") is not None else -1), (s.get("accuracy") or 0)), reverse=True)
    return rows[0] if rows else None
