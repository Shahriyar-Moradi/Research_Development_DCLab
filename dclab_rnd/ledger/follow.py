"""The outcome follower (package 37.16, docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): late outcomes linked to the decisions that
caused them.

A list DCLab made (a call list, the customers a model ranked first) is acted on, and weeks later the person brings the
results: for each customer, whether they stayed. :func:`list_result` turns that into one business number, the share of the
listed rows with the outcome against the share among the others, and writes it as an outcome of every decision of the
project (``ledger.outcomes.observe``, observed "business"): a measured label of what followed, never a guess and never a
row. :func:`follow_inbox` runs the follower over ``<workspace>/outcomes_inbox/<project>/`` on a schedule (``make
outcomes-follow``, a nightly task): each folder holds the list (``list.csv``), the results (``results.csv``) and what to read
(``follow.json``: the id column, the outcome column, the value that counts); a folder done moves to ``done/``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional

FLOOR = 5  # a group this small is never measured or shown (the row floor of every number DCLab gives)


def list_result(project_id: str, listed: Iterable[Any], results: Any, *, id_column: str, outcome_column: str, counts: Any = 1,
                higher_is_better: bool = True, by: str = "results file", ledger: Any = None) -> dict[str, Any]:
    """The listed rows' outcome rate against the others', written as a business outcome of the project's decisions."""
    from . import outcomes

    for col in (id_column, outcome_column):
        if col not in results.columns:
            return {"error": f"the results have no column {col!r}", "written": False}
    ids = {str(v) for v in listed}
    on_list = results[id_column].astype(str).isin(ids)
    hit = results[outcome_column].astype(str) == str(counts)
    n_listed, n_others = int(on_list.sum()), int((~on_list).sum())
    if n_listed < FLOOR or n_others < FLOOR:
        return {"error": f"too few rows to measure: {n_listed} listed and {n_others} others (at least {FLOOR} each)", "written": False}
    rate_listed, rate_others = float(hit[on_list].mean()), float(hit[~on_list].mean())
    out = outcomes.observe(project_id, metric="list_outcome_rate", value=round(rate_listed, 4), reference=round(rate_others, 4),
                           higher_is_better=higher_is_better, observed="business", by=by,
                           note=f"{n_listed} listed and {n_others} others in the results", ledger=ledger)
    return {**out, "listed": n_listed, "others": n_others, "rate_listed": round(rate_listed, 4), "rate_others": round(rate_others, 4)}


def follow_inbox(home: Path, ledger: Any = None) -> list[dict[str, Any]]:
    """Every project folder waiting in the inbox, followed; a folder done (or refused, with its reason) moves to done/."""
    import pandas as pd

    inbox = Path(home) / "outcomes_inbox"
    out = []
    for folder in sorted(p for p in inbox.iterdir() if p.is_dir() and p.name != "done") if inbox.is_dir() else []:
        try:
            spec = json.loads((folder / "follow.json").read_text(encoding="utf-8"))
            listed = pd.read_csv(folder / "list.csv")[spec["id_column"]].tolist()
            results = pd.read_csv(folder / "results.csv")
            result = list_result(folder.name, listed, results, id_column=spec["id_column"], outcome_column=spec["outcome_column"],
                                 counts=spec.get("counts", 1), higher_is_better=bool(spec.get("higher_is_better", True)),
                                 by=str(spec.get("by") or "results file"), ledger=ledger)
        except (OSError, ValueError, KeyError) as error:
            result = {"error": f"{type(error).__name__}: {error}"[:300], "written": False}
        (folder / "followed.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        target = inbox / "done" / folder.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(folder), str(target))
        out.append({"project": folder.name, **result})
    return out


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Follow the outcomes waiting in a workspace's outcomes_inbox (37.16)")
    parser.add_argument("--home", default=None, help="the workspace folder (default: the configured one)")
    args = parser.parse_args(argv)
    from ..settings import Settings

    for row in follow_inbox(Path(args.home or Settings().agent_home)):
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
