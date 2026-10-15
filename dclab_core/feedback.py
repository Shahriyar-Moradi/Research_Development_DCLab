"""Feedback of every kind, tied to the version that was judged (package 23.6).

A person's reaction to something DCLab (or the main product) wrote or decided: a thumb, a score, a correction, an edited
reply, an override. ``check`` validates a feedback request and returns the payload of a ``feedback.given`` event;
``verdict`` reads the feedback on one message into what a training example may use: the person's edit (the label), an
approval, or a rejection. Pure rules: the host stores the events and resolves the version of the message judged.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

KINDS: tuple[str, ...] = ("thumbs", "rating", "correction", "edited_output", "override")
TARGETS: tuple[str, ...] = ("answer", "message", "step", "proposal", "decision", "note", "report")
RATINGS: tuple[str, ...] = ("helpful", "unhelpful", "wrong", "neutral", "edited", "overridden")
MESSAGE_TARGETS: frozenset[str] = frozenset({"answer", "message"})  # the chat messages an edit or a thumb can label
ID = re.compile(r"[0-9a-zA-Z_.:-]{1,80}")
EDIT_MAX = 4000
ROLES_SEEN = frozenset({"owner", "data_scientist", "reviewer", "viewer"})  # the host passes the person's role; stored as meta
READ_ONLY = frozenset({"viewer"})  # a role that may react but not change what is learned: its feedback is recorded and does not count


def kind_of(payload: dict[str, Any]) -> str:
    """The kind of a stored feedback payload; one written before 23.6 has none: a wrong is a correction, the rest thumbs."""
    return payload.get("kind") or ("correction" if payload.get("rating") == "wrong" else "thumbs")


def check(body: dict[str, Any]) -> dict[str, Any]:
    """The ``feedback.given`` payload (without its version) for a request, or ``ValueError`` saying what is wrong.

    ``{"on": target, "ref": its id, "kind"?, "rating"?, "score"?, "reason"?, "edited"?}``; the kind is taken from the
    rating when not given (helpful and unhelpful are thumbs, wrong a correction). A ref that is not an id's shape is
    dropped: ``on`` is meta and travels in an export."""
    on = body.get("on")
    if on not in TARGETS:
        raise ValueError(f"on is one of {', '.join(TARGETS)}")
    ref = str(body.get("ref") or "")[:80]
    ref = ref if ID.fullmatch(ref) else ""
    rating = body.get("rating")
    rating = rating if isinstance(rating, str) else None  # a list or a dict is no rating
    kind = body.get("kind") or {"helpful": "thumbs", "unhelpful": "thumbs", "wrong": "correction"}.get(rating) or ("rating" if body.get("score") is not None else None)
    if kind not in KINDS:
        raise ValueError(f"kind is one of {', '.join(KINDS)}")
    reason = str(body.get("reason") or "").strip()[:300] or None
    out: dict[str, Any] = {"kind": kind, "on": f"{on}:{ref}" if ref else on, "reason": reason}
    if body.get("role") in ROLES_SEEN:
        out["role"] = body["role"]
    if kind == "thumbs":
        if rating not in ("helpful", "unhelpful"):
            raise ValueError("a thumb is helpful or unhelpful")
        out["rating"] = rating
    elif kind == "rating":
        score = body.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError("score is a whole number from 1 to 5")
        out["score"] = score
        out["rating"] = "helpful" if score >= 4 else "unhelpful" if score <= 2 else "neutral"
    elif kind == "correction":
        out["rating"] = "wrong"
    elif kind == "edited_output":
        if on not in MESSAGE_TARGETS or not ref:
            raise ValueError("an edit is of a message: give its ref")
        edited = str(body.get("edited") or "").strip()
        if not edited:
            raise ValueError("edited is the reply as the person would have it")
        out["edited"] = edited[:EDIT_MAX]
        out["rating"] = "edited"
    else:  # override: the person chose another way than the one offered
        if on not in ("decision", "proposal", "step"):
            raise ValueError("an override is of a decision, a proposal or a step")
        out["rating"] = "overridden"
    return {k: v for k, v in out.items() if v is not None}


def verdict(feedback: Iterable[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """What the feedback on one message lets a training example do: ``{"label": "edit", "text": …}`` (the person's own
    reply), ``{"label": "rejected"}`` (a thumb down, a wrong, a low score: the message is not an example), or
    ``{"label": "approved"}``. An edit beats a rejection (the person gave the better reply); a rejection beats an
    approval; None when there is no feedback. The newest edit and the newest thumb of a person win."""
    items = [f for f in feedback if f.get("role") not in READ_ONLY]
    edits = [f for f in items if kind_of(f) == "edited_output" and f.get("edited")]
    if edits:
        return {"label": "edit", "text": str(edits[-1]["edited"])}
    thumbs = [f for f in items if kind_of(f) in ("thumbs", "rating", "correction")]
    if not thumbs:
        return None
    return {"label": "approved" if thumbs[-1].get("rating") == "helpful" else "neutral" if thumbs[-1].get("rating") == "neutral" else "rejected"}


__all__ = ["KINDS", "TARGETS", "RATINGS", "READ_ONLY", "kind_of", "check", "verdict"]
