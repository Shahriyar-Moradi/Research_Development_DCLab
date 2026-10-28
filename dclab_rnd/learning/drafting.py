"""Drafting lessons from the learning variants (H2): attempt, grade, reflect, screen, file as drafts, remember the case.

``draft_lessons(home, family, n)`` reads **learning** variants only (a learner never reads check or sealed ones),
runs each as an H0 attempt, reflects on the rules arm's grade (the rules drafter by default; a model through the
``reflector`` purpose only with a cap), screens every draft, and files the bullets that pass as drafts in the family's
playbook. Drafts are not shown to any solver: H3's probe decides which become accepted. Each attempt is also a case.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from dclab_core import variants as V
from dclab_core.challenge_pack import GRADER
from dclab_core.playbook import Refused

from .cases import FileCases
from .packs import public_dir, read_key, read_public
from .playbooks import FilePlaybooks, summary
from .reflect import lessons_for, reflect
from .scoreboard import attempt


def draft_lessons(home: Path, family: str = "groups", n: int = 5, client: Any = None, cap_eur: float = 0.10, log=print) -> dict[str, Any]:
    books, cases = FilePlaybooks(home), FileCases(home)
    rows, spent = [], 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for seed in V.seeds(family, "learning", n):  # the learning partition only
            row = attempt(Path(tmp), family, seed, keep=Path(home), step="draft")  # 31.17: the attempt a lesson is drawn from is kept
            key = read_key(Path(tmp), family, row["pack"])
            pack, _ = read_public(public_dir(Path(tmp), family, row["pack"]))
            book, _ = books.book(family)
            if client is not None and hasattr(client, "spent_eur"):  # this challenge's spend starts at zero, asked or not (a pass asks nothing)
                client.spent_eur = 0.0
            out = reflect(row, key, pack.columns, book, client=client, cap_eur=cap_eur, run=row.get("run"))
            if client is not None:  # each challenge's spend, before the next one starts again at zero
                spent += float(getattr(client, "spent_eur", 0.0) or 0.0)
            filed = []
            for bullet in out["bullets"]:
                try:
                    books.add(family, bullet, reason=f"drawn from a graded attempt ({bullet.evidence[0].code} at {bullet.evidence[0].step})")
                    filed.append(bullet.id)
                except Refused as exc:
                    out["refused"].append({"draft": {"text": bullet.text}, "why": str(exc)})
            cases.add(family, row["pack"], row["arms"]["rules"]["codes"], row["arms"]["rules"]["verdict"], lessons=filed)
            _keep_drafts(home, family, row, out, filed)
            if out.get("step") is not None:  # an attempt that passed or delivered nothing teaches nothing (the reflector's rule)
                answered = {c: b.id for b in out["bullets"] if b.id in filed for c in b.codes}
                keep_lessons(home, lessons_for(family, row["pack"], row.get("run"), row["arms"]["rules"]["codes"], answered, _now(),
                                               producer="rule:dclab", solvable=bool(row.get("solvable", True)), grader=GRADER))
            rows.append({"pack": row["pack"], "verdict": row["arms"]["rules"]["verdict"], "codes": row["arms"]["rules"]["codes"],
                         "step": out["step"], "filed": filed, "refused": [r["why"] for r in out["refused"]], "why": out["why"]})
            log(f"{row['pack']}: {row['arms']['rules']['verdict']} {row['arms']['rules']['codes']} -> filed {len(filed)}, refused {len(out['refused'])}")
    book, refused = books.book(family)
    return {"family": family, "attempts": rows, "playbook": summary(book), "refused_deltas": len(refused), "spent_eur": round(spent, 4)}


def _keep_drafts(home: Path, family: str, row: dict[str, Any], out: dict[str, Any], filed: list[str]) -> None:
    """Every draft of an attempt kept with its fate (filed, or refused with why): the reflector's work is training data
    too, the refused part most of all. ``<workspace>/learning/drafts.jsonl``, only ever appended."""
    import json
    from datetime import datetime, timezone

    path = Path(home) / "learning" / "drafts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"family": family, "pack": row["pack"], "run": row.get("run"), "codes": row["arms"]["rules"]["codes"], "step": out.get("step"),
           "filed": filed, "refused": out.get("refused") or [], "why": out.get("why"), "kept": row.get("kept"),
           "solvable": row.get("solvable"), "grader": GRADER,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def lessons_path(home: Path) -> Path:
    return Path(home) / "learning" / "lessons.jsonl"


def keep_lessons(home: Path, lessons: list[Any]) -> int:
    """31.5: each lesson appended once (by its id) to ``<workspace>/learning/lessons.jsonl``, never rewritten."""
    import json

    path = lessons_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                have.add(json.loads(line).get("id"))
            except ValueError:
                continue
    new = [x for x in lessons if x.id not in have]
    with open(path, "a", encoding="utf-8") as fh:
        for x in new:
            fh.write(x.model_dump_json() + "\n")
    return len(new)


def backfill_lessons(home: Path) -> int:
    """The lessons of drafts written before 31.5, from their notes (codes, the bullets filed), by the same rules. The rules'
    proposal (``rule:gaps_v1``), labelled as such: no model wrote them."""
    import json

    from .playbooks import FilePlaybooks

    path = Path(home) / "learning" / "drafts.jsonl"
    if not path.is_file():
        return 0
    books, made = FilePlaybooks(home), []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if not str(d.get("pack") or "").startswith("pk-") or d.get("step") is None:
            continue  # no pack, or an attempt that passed or delivered nothing: no lesson
        family = str(d.get("family") or "groups")
        book, _ = books.book(family)
        answered = {c: b.id for b in book.bullets if b.id in (d.get("filed") or []) for c in b.codes}
        made += lessons_for(family, d["pack"], d.get("run"), list(d.get("codes") or []), answered, str(d.get("at") or _now()),
                            producer="rule:dclab", solvable=d.get("solvable") is not False,
                            grader=d.get("grader") or "grader_v1")  # drafts before 31.5 were all graded by grader_v1
    return keep_lessons(home, made)


__all__ = ["draft_lessons", "keep_lessons", "backfill_lessons", "lessons_path"]
