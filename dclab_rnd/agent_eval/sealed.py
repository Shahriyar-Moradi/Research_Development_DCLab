"""The one-time exception to "a sealed set is run once" (the owner, 2026-11-11): a new model may take the sealed exams Claude
already took, once per set and per model, so its numbers line up with Claude's on the same people.

Such a run is not an unseen exam: DCLab was tuned after those sets' results were read, so every rerun is named apart
(``<RUN>_<split>_rerun_<model>``), carries ``"unseen": false`` with the reason, and is never a first run's verdict. The
exception is asked for in words (``--model-rerun OWNER-2026-11-11``); Claude, who took the sets, cannot rerun them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

EXCEPTION = "OWNER-2026-11-11"
WHY = ("a rerun of a sealed set a first model already took (the owner's one-time exception, 2026-11-11): DCLab was tuned after "
       "the first run's results were read, so this set is not unseen for it; compare it with the first run, never treat it as a "
       "first sealed verdict")


def slug(model: str) -> str:
    """``qwen/qwen3.8-flash`` → ``qwen3-8-flash``: the model's own name, safe in a file name."""
    name = str(model or "model").split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-") or "model"


def suffix(model: str) -> str:
    return f"rerun_{slug(model)}"


def refusal(token: Optional[str], model: str, done: Iterable[Path]) -> Optional[str]:
    """Why a rerun of a spent sealed set is refused, or None when the exception applies. ``done``: this model's earlier
    scored reruns of the set (an unscored, failed attempt never counts, as a failed first run never spends a set)."""
    if not token:
        return f"the set was already run; a new model may rerun it once with --model-rerun {EXCEPTION}"
    if token != EXCEPTION:
        return f"--model-rerun takes {EXCEPTION} (the owner's exception of 2026-11-11), not {token!r}"
    if "claude" in str(model).lower():  # also anthropic/claude-... served through OpenRouter
        return "the exception is for a new model: Claude already took these sealed sets"
    done = list(done)
    if done:
        return f"{model} already reran this set ({done[0].name}); a rerun is also once"
    return None


def stamp(model: str, first_runs: Iterable[str]) -> dict:
    """What a rerun's result carries: not unseen, the exception, the model and the first runs it lines up with."""
    return {"unseen": False, "rerun": {"exception": EXCEPTION, "model": model, "of": sorted(first_runs), "why": WHY}}


__all__ = ["EXCEPTION", "WHY", "slug", "suffix", "refusal", "stamp"]
