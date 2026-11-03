"""External benchmarks (the external layer of DCLab Bench, B8 of the 2026-11-03 plan): DCLab against others, starting with MLE-bench.

A real run needs three things DCLab does not ship: the MLE-bench package and its grader (``mlebench``, from openai/mle-bench),
the competitions' data (Kaggle, with the owner's key), and compute and a euro budget the owner sets. So this module does what can
be checked now and refuses the rest:

- ``check()``: what is here and what is missing (the grader, the Kaggle key, the data folder, free disk);
- ``plan(tasks)``: what a run on the chosen competitions would need, written before any run;
- ``run(...)``: refused until ``check()`` is clean and a cap is given; it then records each competition's medal and score once in
  ``evidence/bench/external/`` (never over a record), in the external layer.

The competitions are the owner's choice (``tasks.json`` beside this file, empty until chosen). A competition's tasks are never
allowed into training data (the store's rule for the solver model).

    python -m dclab_rnd.bench external check|plan
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[3]
TASKS = Path(__file__).with_name("tasks.json")
OUT = ROOT / "evidence" / "bench" / "external"
DATA = Path(os.environ.get("DCLAB_MLEBENCH_DATA") or (Path.home() / ".cache" / "mle-bench" / "data"))
MIN_FREE_GB = 20


def tasks() -> list[str]:
    try:
        return [str(t) for t in json.loads(TASKS.read_text(encoding="utf-8")).get("competitions", [])]
    except (OSError, ValueError):
        return []


def check() -> dict[str, Any]:
    """What a run needs, and whether each is here: never a key's value, only whether it is set."""
    kaggle = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or (Path.home() / ".kaggle" / "kaggle.json").is_file()
    free = round(shutil.disk_usage(Path.home()).free / 1024 ** 3, 1)
    items = {"mlebench grader installed": importlib.util.find_spec("mlebench") is not None,
             "Kaggle key set": kaggle,
             f"data folder ({DATA})": DATA.is_dir(),
             f"at least {MIN_FREE_GB} GB free (now {free} GB)": free >= MIN_FREE_GB,
             "competitions chosen (tasks.json)": bool(tasks())}
    return {"ready": all(items.values()), "items": items}


def plan(chosen: Optional[list[str]] = None) -> dict[str, Any]:
    chosen = chosen or tasks()
    return {"competitions": chosen, "check": check(),
            "steps": ["install openai/mle-bench (pip install -e) and accept each competition's rules on Kaggle (the owner)",
                      "prepare the data with mlebench prepare -c <competition> into DCLAB_MLEBENCH_DATA",
                      "run DCLab's engine on each competition's training data alone; its submission is graded by mlebench grade",
                      "record the medal, the score and the euro spent per competition once in evidence/bench/external/"],
            "limits": ["a handful of competitions is a sample, not the benchmark's whole suite",
                       "a competition's tasks never enter training data"]}


def run(cap_eur: Optional[float] = None, yes: bool = False) -> Path:
    state = check()
    if not state["ready"]:
        missing = [k for k, ok in state["items"].items() if not ok]
        raise RuntimeError("not ready for an external run: " + "; ".join(missing))
    if not yes or cap_eur is None:
        raise RuntimeError("an external run spends compute and money: pass --yes and --cap-eur (the owner's number)")
    raise NotImplementedError("the grading step is wired once the grader is installed here and a competition is chosen")


__all__ = ["check", "plan", "run", "tasks"]
