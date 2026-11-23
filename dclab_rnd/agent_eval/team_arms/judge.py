"""The optional judge of a team-arms play (B4, 2026-11-24): a free, non-Claude model reads what the person saw and scores
understanding, plan and report from 0 to 4. Advisory only: the code score and the winner rule never read it.

It is the harness ablation's judge (``prompts/ablation_judge.md``, the rubric the Nemotron judge used in the earlier
exams), unchanged: its ``understanding`` is understanding, its ``planning`` the plan, its ``evidence_discipline`` the
report. The judge is blind to the arm: the words a report or a note carries to say which arm wrote it are taken out of
what it reads. A rules arm shows no plan of its own, so its plan score is what the person saw of one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from . import cases as C
from .play import PLAY, model_refusal

MARKS = (
    (re.compile(r"\s*Written by the agent arm:[^\n]*"), ""),
    (re.compile(r"\(the helper's reading[^)]*\)\s*:?", re.I), ":"),
    (re.compile(r"Risks the helper sees \(advice only[^)]*\)", re.I), "Risks to weigh"),
    (re.compile(r"\bhelper\b", re.I), "assistant"),
    (re.compile(r"\bagent arm\b", re.I), "assistant"),
)


def blind(text: str) -> str:
    for pattern, repl in MARKS:
        text = pattern.sub(repl, text)
    return text.strip()


def judge_refusal(model: Optional[str], arm_models: list[str]) -> Optional[str]:
    """Why ``model`` may not judge, or None: never Claude's, never an arm's model, and free (its price is 0)."""
    why = model_refusal(model)
    if why:
        return why
    def base(name: Any) -> str:
        return str(name).lower().removesuffix(":free")

    if base(model) in {base(m) for m in arm_models}:  # the free variant of an arm's model is still that model
        return "the judge must not be a model it judges"
    from dclab_rnd.models import prices

    price = prices.price_of(str(model))
    if not str(model).endswith(":free") or price is None or (price.input or price.output):
        return f"{model} is not a free judge (only a free model may judge here)"
    return None


def transcript(folder: Path) -> list[dict[str, Any]]:
    """What the person saw, in order: their opening, the messages their channel received, their own replies after the
    question each answered. Team members' words stay in their teams."""
    from .score import Records

    rec = Records(folder)
    person = rec.people.get("person")
    mine = [f for f in rec.play.get("fired") or [] if f.get("person") == person and f.get("text")]
    out = [{"role": "user", "text": rec.opening or ""}]
    for t in rec.person_threads():
        for m in rec.messages.get(t["id"], []):
            author = m.get("author") or {}
            if author.get("kind") == "person":
                continue  # the script's own words are added below, in their order
            out.append({"role": "assistant", "text": blind(str(m.get("text") or ""))})
            if (m.get("handoff") or {}).get("kind") == "clarification" and mine:
                out.append({"role": "user", "text": str(mine.pop(0)["text"])})
    out += [{"role": "user", "text": str(f["text"])} for f in mine]
    return out


def judge_play(folder: Path, case: C.Case, client: Any) -> Optional[dict[str, Any]]:
    """Judge one play once (``judged.json`` beside it), or None when the case has no judge key."""
    if case.judge_key is None:
        return None
    from ..harness_ablation import judge as ablation

    key = case.judge_key.model_dump()
    try:
        found = ablation.judged(client, {"upload": None}, key, transcript(folder))
    except Exception as exc:  # noqa: BLE001 - a judge that fails leaves the play unjudged; the code score stands
        found = {"valid": False, "problems": [f"{type(exc).__name__}: {str(exc)[:300]}"]}
    if found.get("valid"):
        found = {**found, "understanding": found["understanding"], "plan": found["planning"], "report": found["evidence_discipline"]}
    found["model"] = getattr(client, "model", None)
    (Path(folder) / "judged.json").write_text(json.dumps(found, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return found


__all__ = ["blind", "judge_play", "judge_refusal", "transcript"]
