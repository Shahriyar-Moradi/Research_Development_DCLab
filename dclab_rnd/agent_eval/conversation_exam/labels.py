"""The owner's labels beside the judge (37.0): ten conversations of a run drawn for the owner, blind to the arm, with the
judge's own questions left blank; agreement between the judge and the owner on what both answered.

A conversation is named by ``ref`` (a hash of its person and arm), so the owner's file never says which arm spoke and
the agreement can still find the judge's answer."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Iterable, Mapping, Optional

SEED = 20261030


def ref(person_id: str, arm: str) -> str:
    return hashlib.sha256(f"{person_id}|{arm}".encode()).hexdigest()[:10]


def todo(conversations: list[Mapping[str, Any]], people: Mapping[str, Mapping[str, Any]], n: int = 10, seed: int = SEED) -> list[dict[str, Any]]:
    """``n`` conversations judged validly, drawn with a fixed seed, each arm in turn; the owner's answers blank."""
    rng = random.Random(seed)
    by_arm: dict[str, list[Mapping[str, Any]]] = {}
    for c in conversations:
        if (c.get("judged") or {}).get("valid"):
            by_arm.setdefault(str(c["arm"]), []).append(c)
    for rows in by_arm.values():
        rng.shuffle(rows)
    drawn: list[Mapping[str, Any]] = []
    arms = sorted(by_arm)
    while len(drawn) < n and any(by_arm.values()):
        for arm in arms:
            if by_arm[arm] and len(drawn) < n:
                drawn.append(by_arm[arm].pop())
    out = []
    for c in drawn:
        p = people[c["id"]]
        out.append({"ref": ref(str(c["id"]), str(c["arm"])),
                    "person": {"truth": p.get("truth"), "level": p.get("level"), "style": p.get("style"),
                               "must_understand": p.get("must_understand"), "trap": p.get("trap"), "trap_check": p.get("trap_check")},
                    "transcript": [{"role": t.get("role"), "text": t.get("text"),
                                    "choices": [{"title": x.get("title"), "what": x.get("what")} for x in t.get("choices") or []]}
                                   for t in c.get("transcript") or []],  # no kind, track or tool: the owner labels blind
                    "owner": {"understood": [None] * len(p.get("must_understand") or []), "options_fit": None, "options_differ": None,
                              "trap_handled": None}})
    rng.shuffle(out)  # the order never tells the arm (the review of 37.0: they ran home, plain, home, plain)
    return out


def _pair_agreement(judged: Mapping[str, Any], owner: Mapping[str, Any]) -> Optional[float]:
    parts = []
    points = [(j, o) for j, o in zip(judged.get("understood") or [], owner.get("understood") or []) if o is not None]
    if points:
        parts.append(sum(j == o for j, o in points) / len(points))
    if judged.get("options_fit") is not None and owner.get("options_fit") is not None:
        parts.append(1.0 if abs(int(judged["options_fit"]) - int(owner["options_fit"])) <= 1 else 0.0)
    for key in ("options_differ", "trap_handled"):
        if judged.get(key) is not None and owner.get(key) is not None:
            parts.append(1.0 if bool(judged[key]) == bool(owner[key]) else 0.0)
    return sum(parts) / len(parts) if parts else None


def agreement(pairs: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> Optional[float]:
    """Mean agreement over the labelled conversations: the share of points marked alike, the options' fit within one, and
    the options' difference and the trap marked alike, each where both answered."""
    values = [v for j, o in pairs if (v := _pair_agreement(j, o)) is not None]
    return round(sum(values) / len(values), 4) if values else None


def agreement_of_run(record: Mapping[str, Any], labelled: list[Mapping[str, Any]]) -> Optional[float]:
    """The run's agreement from the owner's filled file (``todo`` entries with ``owner`` answered)."""
    judged = {ref(str(c["id"]), str(c["arm"])): c.get("judged") or {} for c in record.get("conversations") or []}
    return agreement((judged[x["ref"]], x["owner"]) for x in labelled if x.get("ref") in judged)


def dumps(entries: list[Mapping[str, Any]]) -> str:
    return json.dumps(entries, indent=1, ensure_ascii=False) + "\n"
