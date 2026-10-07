"""The owner's inputs (package 14.1): which questions are still open, and whether each contract is a valid solution.

``evaluation/OWNER_ANSWERS.md`` holds one block per question (``### id``, the question, then ``ANSWER:``). A blank answer
is open and stays open: nothing here fills one in. A contract is a file in ``evaluation/contracts/`` with the data file
it is for and a solution; it must load as ``studio.solution.Solution`` and name only columns the data has. The status says,
per source, whether a valid contract file exists and its questions are answered (a re-run under it is a separate step). An answer that starts with
"Recorded from" was written by the assistant from what happened: the status lists it as unconfirmed, never as the owner's answer.

    python -m dclab_rnd.agent_eval.owner_inputs status

A contract file:  {"source": "data/project/telco/....csv", "solution": {"target": "Churn", "task": "binary", "prediction_moment": "...", ...}}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ANSWERS = ROOT / "evaluation" / "OWNER_ANSWERS.md"
RECORDED = "Recorded from"  # an answer the assistant wrote from what happened, for the owner to confirm: it is not the owner's answer yet
CONTRACTS = ROOT / "evaluation" / "contracts"


def answers(path: Path = ANSWERS) -> dict[str, str]:
    """question id -> the text after ANSWER: (empty when open)."""
    text = Path(path).read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for block in re.split(r"^### ", text, flags=re.M)[1:]:
        qid, _, rest = block.partition("\n")
        _, _, given = rest.partition("ANSWER:")
        out[qid.strip()] = given.strip()
    return out


def check_contract(path: Path) -> dict[str, Any]:
    """Load a contract and say whether it is valid: a Solution that fits the columns of its data file."""
    import pandas as pd

    from dclab_rnd.studio.solution import Solution

    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        solution = Solution.model_validate(body["solution"])
        source = ROOT / body["source"]
        columns = list(pd.read_csv(source, nrows=5).columns) if source.suffix == ".csv" else list(pd.read_parquet(source).columns)
        solution.check_columns(columns)
    except Exception as error:  # noqa: BLE001 — a bad contract is reported with its reason, never hidden
        return {"contract": Path(path).name, "valid": False, "why": f"{type(error).__name__}: {str(error)[:300]}"}
    return {"contract": Path(path).name, "valid": True, "source": body["source"], "target": solution.target}


def status(answers_path: Path = ANSWERS, contracts: Path = CONTRACTS) -> dict[str, Any]:
    given = answers(answers_path)
    open_ids = sorted(q for q, a in given.items() if not a)
    found = sorted(Path(contracts).glob("*.json")) if Path(contracts).is_dir() else []
    unconfirmed = sorted(q for q, a in given.items() if a.startswith(RECORDED))
    checked = [check_contract(p) for p in found]
    valid = {c["contract"] for c in checked if c["valid"]}

    def loads(source: str) -> bool:
        return not [q for q in open_ids if q.startswith(source + ".")] and any(p.name in valid for p in found if p.stem.startswith(source))
    return {"questions": len(given), "answered": sorted(q for q, a in given.items() if a and q not in unconfirmed), "unconfirmed": unconfirmed,
            "open": open_ids, "contracts": checked, "contract_loads": {"hyperack": loads("hyperack"), "churn": loads("churn")}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.owner_inputs", description="What the owner has answered, and what is still open (14.1).")
    parser.add_argument("command", choices=("status",))
    parser.parse_args(argv)
    s = status()
    print(f"{len(s['answered'])} of {s['questions']} questions answered by the owner; {len(s['unconfirmed'])} recorded by the assistant, to confirm "
          f"({', '.join(s['unconfirmed']) or 'none'}); {len(s['open'])} open:")
    for q in s["open"]:
        print(f"  open  {q}")
    for c in s["contracts"]:
        print(f"  contract {c['contract']}: " + ("valid, target " + c["target"] if c["valid"] else "INVALID " + c["why"]))
    print("A valid contract file exists and its questions are answered: " + ", ".join(f"{k} {'yes' if v else 'not yet'}" for k, v in s["contract_loads"].items()))
    return 0 if not s["open"] else 1


if __name__ == "__main__":
    sys.exit(main())
