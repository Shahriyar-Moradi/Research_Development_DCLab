"""The problem map on Home (package 24.2): drawn from the person's words, reviewed, edited by the person, confirmed.

A draft keeps its map under ``problem_map``: the map itself (``dclab_core.problem_map.ProblemMap``), where it came from (the
rules, or the decomposer when it serves), the reviewer's questions, whether the person confirmed it, and the edits made. A small
problem needs no confirmation. A problem of several parts is confirmed by the person, or changed until it is what they mean: each
edit goes through the map validator (``dclab_core.mapping``) and is refused with its reasons when it would make the map unsound; the
accepted ones are a ``graph.move`` in the session record and a ``map.edit`` decision in the ledger, the drawing a ``map.size`` one.

The roles (decomposer, map reviewer, data-need agent) serve only when ``DCLAB_MAP_ROLES`` is on and they passed their sealed
measurement (24.8); until then the rules draw and review, and the roles run only when a measurement asks for them.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

from dclab_core import mapping, problem_map as pm

ACTIONS = ("add", "remove", "merge", "confirm", "redraw")


def switched_on() -> bool:
    return os.environ.get("DCLAB_MAP_ROLES", "").strip().lower() in ("1", "on", "yes", "true")


def roles_passed() -> bool:
    """Whether the three roles passed their sealed measurement (24.8). Not yet: the rules serve."""
    return False


def advisor_for(gateway: Any, *, draft_id: Optional[str] = None, cap_eur: Optional[float] = None, force: bool = False) -> Any:
    """The gateway's advisor for the three roles, or None: off, no model serves one of them, or they did not pass their measurement.
    ``force``: a measurement asks for them."""
    if gateway is None or not all(gateway.available(p) for p in ("decomposer", "map_reviewer", "data_need")):
        return None
    if not force and not (switched_on() and roles_passed()):
        return None
    from ..core_advisor import GatewayAdvisor

    return GatewayAdvisor(gateway, draft_id=draft_id, cap_eur=cap_eur, max_tokens=1600)


# ---------------------------------------------------------------------------------------------------------------- the draft's facts

def words_of(draft: dict[str, Any]) -> str:
    """What the person said of the problem: the first description and their messages."""
    return " ".join([str(draft.get("problem") or "")] + [str(m.get("text") or "") for m in draft.get("messages") or [] if m.get("role") == "user"]).strip()


def _digest(words: str) -> str:
    return hashlib.sha256(words.encode("utf-8")).hexdigest()[:16]


def data_card_of(draft: dict[str, Any]) -> Any:
    """The draft's data card (names, purposes and relations, never rows), or None before there is a table."""
    raw = draft.get("data_card")
    if not raw:
        return None
    try:
        from dclab_core.datacard import DataCard

        return DataCard.read(raw)
    except Exception:  # noqa: BLE001 - a card that does not read is no card: the review goes on without the data
        return None


def stored(draft: dict[str, Any]) -> Optional[dict[str, Any]]:
    return draft.get("problem_map") or None


def current(draft: dict[str, Any]) -> Optional[pm.ProblemMap]:
    held = stored(draft)
    if not held:
        return None
    try:
        return pm.ProblemMap.read(held["map"])
    except Exception:  # noqa: BLE001 - a stored map that no longer reads is drawn again
        return None


def _reading(draft: dict[str, Any]) -> pm.ProblemMap:
    """What the rules read from the person's words now (cheap, no model): the structure a later answer may or may not change."""
    return pm.decompose(words_of(draft), parts=_parts_of(draft))


def stale(draft: dict[str, Any]) -> bool:
    """The person said more since the map was drawn, the rules now read the words differently, and nothing of the person's is in the map (not
    edited, not confirmed). Such a map is drawn again and never holds or lets through a build. Words that change nothing the rules read (an answer
    to a question about the data) leave it as it is: a walk that goes on asking never redraws the map under itself."""
    held, m = stored(draft), current(draft)
    if not held or m is None or m.source == "person" or held.get("confirmed") or held.get("words") == _digest(words_of(draft)):
        return False
    return held.get("rules") is None or held["rules"] != _reading(draft).fingerprint


def _parts_of(draft: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
    """The brief's own parts, when the problem text has them (challenge mode's reading)."""
    from dclab_core.challenge import read_brief

    parts = read_brief(str(draft.get("problem") or ""))["parts"]
    return parts if len(parts) >= 2 else None


def _card_summary(draft: dict[str, Any]) -> dict[str, Any]:
    from .card import card_of

    card = card_of(draft)
    return {"family": card.family.value, "labels": card.labels.value, "unit": card.brief.unit.value, "moment": card.brief.prediction_moment.value}


def propose(draft: dict[str, Any], advisor: Any = None) -> tuple[pm.ProblemMap, str, str]:
    """The map for this draft's words (nothing is stored): the model's when it serves and holds, else the rules'. Needs are pointed at the
    draft's tables when it has them."""
    words = words_of(draft)
    data = data_card_of(draft)
    made, source, reason = mapping.draw(words, advisor=advisor, card=_card_summary(draft), data=data, parts=_parts_of(draft))
    if data is not None:
        made, _, _ = mapping.needs_for(made, data, advisor)
    return made, source, reason


def _review(m: pm.ProblemMap, draft: dict[str, Any], advisor: Any = None) -> mapping.MapReview:
    return mapping.review_with(m, advisor=advisor, words=words_of(draft), data=data_card_of(draft))[0]


# ---------------------------------------------------------------------------------------------------------------- what the API shows

def facts(draft: dict[str, Any]) -> dict[str, Any]:
    """What the coordinator reads: the map's size (None: not drawn), its parts, the reviewer's blocking questions, whether it is confirmed."""
    held, m = stored(draft), current(draft)
    if not held or m is None or stale(draft):  # not drawn, or drawn from fewer words: drawn (again) now
        return {"map_size": None, "map_parts": 0, "map_blockers": [], "map_confirmed": False}
    blockers = [{"code": q.code, "question": q.question} for q in mapping.review_rules(m) if q.blocking]  # the map's own soundness, always current
    return {"map_size": m.size, "map_parts": len(m.parts), "map_blockers": blockers, "map_confirmed": bool(held.get("confirmed")) or m.size == "single"}


def view(draft: dict[str, Any], advisor: Any = None) -> dict[str, Any]:
    """The map as the page shows it: the map with each part's reasons, the review's questions, the issues of the validator, and whether it
    is stored and confirmed. A draft with no stored map shows the rules' proposal (``drawn`` false): nothing is written by looking."""
    held, m = stored(draft), current(draft)
    source, reason = (held.get("source"), held.get("reason")) if held and m is not None else (None, None)
    if m is None:
        m, source, reason = propose(draft, advisor)
        review = _review(m, draft, advisor)
    else:
        review = mapping.review(m, data_card_of(draft), extra=[mapping.MapQuestion.model_validate(q) for q in (held.get("review") or {}).get("questions", []) if q.get("source") == "model"],
                                source=(held.get("review") or {}).get("source", "rules"))
    drawn = bool(held) and current(draft) is not None
    return {"drawn": drawn, "map": json.loads(m.model_dump_json()), "source": source, "reason": reason, "issues": pm.validate(m),
            "review": json.loads(review.model_dump_json()), "confirmed": bool(held and held.get("confirmed")) or m.size == "single",
            "needs_confirmation": m.size == "complex" and not (held and held.get("confirmed")),
            "stale": stale(draft),
            "history": list((held or {}).get("history") or [])[-20:]}


# ---------------------------------------------------------------------------------------------------------------- writing

def _remember(drafts: Any, draft_id: str, m: pm.ProblemMap, review: mapping.MapReview, *, source: str, reason: str, confirmed: bool,
              action: Optional[dict[str, Any]] = None, words: Optional[str] = None, rules: Optional[str] = None) -> dict[str, Any]:
    def keep(d: dict[str, Any]) -> None:
        old = d.get("problem_map") or {}
        history = list(old.get("history") or [])
        if action:
            history.append(action)
        d["problem_map"] = {"map": json.loads(m.model_dump_json()), "source": source, "reason": reason, "words": words or old.get("words") or _digest(words_of(d)),
                            "rules": rules or old.get("rules"), "review": json.loads(review.model_dump_json()), "confirmed": confirmed, "history": history[-50:]}
    return drafts.update(draft_id, keep)


def draw(drafts: Any, draft_id: str, *, advisor: Any = None, redraw: bool = False, who: str = "person") -> dict[str, Any]:
    """Draw the map and keep it (the draft's, from now on). Drawn again when asked (``redraw``: the person's edits and confirmation are dropped, and
    that is a graph move and a decision of theirs) and when it was drawn from fewer words and nothing of the person's is in it (``stale``). A
    person's edits and confirmation are never undone by new words. Returns the view."""
    from .. import experience
    from ..ledger import capture

    draft = drafts.get(draft_id)
    held = current(draft)
    auto = held is not None and stale(draft)
    if held is not None and not redraw and not auto:
        return view(draft, advisor)
    made, source, reason = propose(draft, advisor)
    review = mapping.review(made, data_card_of(draft))  # the rules' questions now; a model's are added below, outside the lock
    words = words_of(draft)
    reading = _reading(draft)
    rules_size = reading.size
    action = {"action": "redraw" if redraw else "draw", "by": who if redraw else ("model" if source == "advisor" else "rule"), "to": made.fingerprint,
              **({"from": held.fingerprint} if held is not None else {}), **({"why": "the person said more"} if auto else {})}
    _remember(drafts, draft_id, made, review, source=source, reason=reason, confirmed=False, words=_digest(words), action=action, rules=reading.fingerprint)
    if redraw and held is not None:
        experience.record("graph.move", "person", {"move": "map:redraw", "allowed": True, "from": held.fingerprint, "to": made.fingerprint, "failed_checks": None,
                                                   "rules": ["problem_map.validate"]}, conversation=draft_id)
        capture.map_edited(draft_id, "redraw", len(held.parts), len(made.parts))
    capture.map_drawn(draft_id, made, rules_size, source == "advisor")
    refresh_review(drafts, draft_id, advisor)
    return view(drafts.get(draft_id), advisor)


def refresh_review(drafts: Any, draft_id: str, advisor: Any = None) -> None:
    """Add the reviewer's model questions to the stored review (outside the draft's lock: a model is slow). Kept only when the map is still the
    one they were asked about."""
    if advisor is None:
        return
    draft = drafts.get(draft_id)
    m = current(draft)
    if m is None:
        return
    review = _review(m, draft, advisor)

    def keep(d: dict[str, Any]) -> None:
        held = current(d)
        if held is not None and held.fingerprint == m.fingerprint and d.get("problem_map"):
            d["problem_map"]["review"] = json.loads(review.model_dump_json())
    drafts.update(draft_id, keep)


EDIT_KEYS = {"remove": ("part",), "merge": ("keep", "other"), "confirm": (), "add": ("title", "family", "question", "unit", "moment", "labels", "quote", "joins", "into")}


def edit(drafts: Any, draft_id: str, action: str, *, who: str = "person", advisor: Any = None, **kw: Any) -> dict[str, Any]:
    """A person's change to the map: ``add`` (title, family, ...), ``remove`` (part), ``merge`` (keep, other), ``confirm`` or ``redraw``. It is read,
    checked and written in one update of the draft (two edits never overwrite each other). The validator refuses an unsound result with its
    reasons (``mapping.MapEditRefused``) and the map stays as it was; any other argument than the action's own is ignored."""
    from .. import experience
    from ..ledger import capture

    if action not in ACTIONS:
        raise mapping.MapEditRefused([f"{action} is not an edit; one of: {', '.join(ACTIONS)}"])
    if action == "redraw":
        return draw(drafts, draft_id, advisor=advisor, redraw=True, who=who)
    kw = {k: v for k, v in kw.items() if k in EDIT_KEYS[action]}
    if current(drafts.get(draft_id)) is None:
        draw(drafts, draft_id, advisor=advisor)
    box: dict[str, Any] = {}

    def apply(d: dict[str, Any]) -> None:
        held = current(d)
        if held is None:
            raise mapping.MapEditRefused(["there is no map to change: draw it first"])
        box["held"] = held
        prior = d.get("problem_map") or {}
        if action == "confirm":
            refusing = [q for q in mapping.review_rules(held) if q.blocking]
            if refusing:
                raise mapping.MapEditRefused([q.question for q in refusing])
            made = held
        else:
            retired = [h[k] for h in prior.get("history") or [] for k in ("part", "other") if isinstance(h.get(k), str)] + [i for h in prior.get("history") or [] for i in h.get("subs") or []]
            if action == "remove":
                made = mapping.remove_part(held, str(kw.get("part") or ""))
            elif action == "merge":
                made = mapping.merge_parts(held, str(kw.get("keep") or ""), str(kw.get("other") or ""))
            else:
                made = mapping.add_part(held, kw.get("title"), kw.get("family"), question=kw.get("question") or "", unit=kw.get("unit"), moment=kw.get("moment"),
                                        labels=kw.get("labels") or "unknown", quote=kw.get("quote"), joins=kw.get("joins"), into=kw.get("into"), avoid=retired)
            data = data_card_of(d)
            if data is not None:
                made = mapping.fill_needs(made, data)
        box["made"] = made
        step = {"action": action, "by": who, "from": held.fingerprint, "to": made.fingerprint}
        if action == "remove":
            step.update(part=kw.get("part"), subs=[s.id for s in held.sub_problems if s.part == kw.get("part")])
        if action == "merge":
            step.update(keep=kw.get("keep"), other=kw.get("other"))
        history = [*(prior.get("history") or []), step][-50:]
        d["problem_map"] = {"map": json.loads(made.model_dump_json()), "source": "person" if action != "confirm" else prior.get("source", "rules"),
                            "reason": prior.get("reason", "") if action == "confirm" else "edited by the person", "words": prior.get("words") or _digest(words_of(d)), "rules": prior.get("rules"),
                            "review": json.loads(mapping.review(made, data_card_of(d)).model_dump_json()), "confirmed": action == "confirm", "history": history}
    try:
        drafts.update(draft_id, apply)
    except mapping.MapEditRefused as refused:
        held = box.get("held")
        if held is not None:
            experience.record("graph.move", "person", {"move": f"map:{action}", "allowed": False, "from": held.fingerprint, "to": held.fingerprint,
                                                       "failed_checks": [i[:80] for i in refused.issues][:20], "rules": ["problem_map.validate"]}, conversation=draft_id)
        raise
    held, made = box["held"], box["made"]
    experience.record("graph.move", "person", {"move": f"map:{action}", "allowed": True, "from": held.fingerprint, "to": made.fingerprint, "failed_checks": None,
                                               "rules": ["problem_map.validate"]}, conversation=draft_id)
    capture.map_edited(draft_id, action, len(held.parts), len(made.parts))
    refresh_review(drafts, draft_id, advisor)
    return view(drafts.get(draft_id), advisor)


def blocker(draft: dict[str, Any]) -> Optional[str]:
    """Why a build waits on the map, or None: the problem is several parts and the person has not confirmed the map. A draft whose words
    make one problem, or whose stored map is confirmed (or single), is not held."""
    held = stored(draft)
    m = current(draft)
    if m is None or stale(draft):  # nothing kept, or kept from fewer words: what the words say now
        m = pm.decompose(words_of(draft), parts=_parts_of(draft))
        confirmed = False
    else:
        confirmed = bool(held.get("confirmed"))
    if m.size == "complex" and not confirmed:
        return (f"The problem has {len(m.parts)} parts that depend on each other ({', '.join(p.title for p in m.parts[:4])}). Confirm the map of the problem "
                "on Home, or change it to the one problem you mean, before building.")
    return None
