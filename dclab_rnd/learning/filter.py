"""The deterministic screen every lesson passes before it is stored (H2; RRSI's brakes, the reports' leak fix).

A lesson is refused when it names any column the family's generator can use (across all its skins, since a skin's
names repeat across the learning, check and sealed variants; whole words, however they are written), carries a digit
(a score, a count, an identifier's value: the rules' lessons have none), cites a step other than where its code's
mistake first shows, answers no grader code or an unknown one, has a scope other than invariant or its family's
domain, or is too short or too long. At most :data:`MOST` drafts of one attempt are kept (an edit cap per candidate).
:func:`door` is the same rule at the store's own door, for a bullet however it was made.
"""

from __future__ import annotations

import re
from typing import Any

from dclab_core.challenge_pack import CODES, GRADER, ChallengeKey, names_in
from dclab_core.playbook import TEXT, Bullet, Evidence, bullet_id

MOST = 3
DIGIT = re.compile(r"\d")


def _family_names(family: str) -> list[str]:
    from dclab_core.variants import family_names

    try:
        return family_names(family)
    except KeyError:
        return []


def door(bullet: Bullet, family: str) -> str:
    """Why a bullet may not enter a family's playbook, whoever made it ("" when it may)."""
    from dclab_core.challenge_pack import ChallengeKey as Key

    if not 10 <= len(bullet.text) <= TEXT:
        return f"a lesson is 10 to {TEXT} characters"
    if DIGIT.search(bullet.text):
        return "carries a digit (a score, a count or an identifier's value)"
    stub = Key(pack="pk-door", family=family, seed=0, partition="learning")
    if names_in(bullet.text, stub, _family_names(family)):
        return "names a column this family's tables can have: a lesson must hold for any table of its kind"
    if not bullet.codes or any(c not in CODES for c in bullet.codes):
        return "answers no grader code, or one that does not exist"
    return ""


def screen(draft: dict[str, Any], key: ChallengeKey, columns: list[str], steps: list[str], pack: str, run: str | None,
           by: str, code_step: dict[str, str] | None = None) -> tuple[Bullet | None, str]:
    """One draft (``text``, ``scope``, ``codes``, ``step``) as a bullet, or None with why it was refused."""
    text = " ".join(str(draft.get("text") or "").split())
    if not 10 <= len(text) <= TEXT:
        return None, f"a lesson is 10 to {TEXT} characters"
    if names_in(text, key, [*columns, *_family_names(key.family)]):
        return None, "names the table's columns, or ones this family's tables can have: a lesson must hold for any table of its kind"
    if DIGIT.search(text):
        return None, "carries a digit (a score, a count or an identifier's value)"
    codes = [c for c in draft.get("codes") or [] if isinstance(c, str)]
    if not codes or any(c not in CODES for c in codes):
        return None, "answers no grader code, or one that does not exist"
    step = str(draft.get("step") or "")
    if step not in steps or (code_step and any(code_step.get(c) != step for c in codes)):
        return None, "cites no step of the graded run, or not the step where its code's mistake first shows"
    raw = str(draft.get("scope") or "")
    scope = "invariant" if raw == "invariant" else f"domain:{key.family}" if raw in ("domain", f"domain:{key.family}") else None
    if scope is None:
        return None, "its scope is neither invariant nor this family's domain"
    evidence = [Evidence(pack=pack, run=run, step=step, code=codes[0], grader=GRADER)]
    return Bullet(id=bullet_id(text, scope), text=text, scope=scope, codes=sorted(set(codes))[:6], evidence=evidence, by=by[:120]), ""


def screen_all(drafts: list[dict[str, Any]], key: ChallengeKey, columns: list[str], steps: list[str], pack: str, run: str | None,
               by: str, code_step: dict[str, str] | None = None) -> tuple[list[Bullet], list[dict[str, Any]]]:
    """Every draft screened; at most :data:`MOST` kept, the rest refused with why."""
    kept: list[Bullet] = []
    refused: list[dict[str, Any]] = []
    for draft in drafts:
        if len(kept) >= MOST:
            refused.append({"draft": draft, "why": f"more than {MOST} lessons from one attempt"})
            continue
        bullet, why = screen(draft, key, columns, steps, pack, run, by, code_step)
        if bullet is None:
            refused.append({"draft": draft, "why": why})
        else:
            kept.append(bullet)
    return kept, refused


__all__ = ["MOST", "door", "screen", "screen_all"]
