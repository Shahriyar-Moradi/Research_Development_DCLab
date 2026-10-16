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

from ..experience import hooks

ACTIONS = ("add", "remove", "merge", "confirm", "redraw")


def switched_on() -> bool:
    return os.environ.get("DCLAB_MAP_ROLES", "").strip().lower() in ("1", "on", "yes", "true")


def roles_passed() -> bool:
    """Whether the three roles passed their sealed measurement (24.8). Not yet: the rules serve."""
    return False


def advisor_for(gateway: Any, *, draft_id: Optional[str] = None, cap_eur: Optional[float] = None, force: bool = False) -> Any:
    """The gateway's advisor for the three roles, or None: off, no model serves one of them, or they did not pass their measurement.
    ``force``: a measurement asks for them."""
    if gateway is None or not all(gateway.available(p) for p in ("decomposer", "map_reviewer", "data_need", "solution_router")):
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


# ---------------------------------------------------------------------------------------------------------------- the solution router (24.3)

def _data_digest(draft: dict[str, Any]) -> str:
    return _digest(json.dumps(draft.get("data_card") or {}, sort_keys=True, default=str))


def compute_routes(draft: dict[str, Any], m: pm.ProblemMap, advisor: Any = None) -> dict[str, Any]:
    """The route of every sub-problem of the map against the draft's data card; a single problem keeps the kind the detection found (never
    replaced). With the advisor serving, it may choose among the candidates the data does not refute."""
    from dclab_core import solutions

    data = data_card_of(draft)
    detected = None
    if m.size == "single":
        from .. import kinds

        detected = kinds.detect(words_of(draft), (draft.get("analysis") or {}).get("profile"))["key"]
    return {s.id: solutions.route_with(s, data, m, detected, advisor)[0] for s in m.sub_problems}


def route_view(r: Any) -> dict[str, Any]:
    """A route as the page shows it: the approach chosen with its title and reason, and every candidate with what blocks it or is not known yet."""
    from dclab_core import solutions

    return {"chosen": r.chosen, "title": solutions.BY_ID[r.chosen].title if r.chosen else None, "reason": r.reason, "source": r.source, "single": r.single,
            "candidates": [{"approach": c.approach, "title": solutions.BY_ID[c.approach].title, "rank": c.rank, "blocking": c.blocking, "unknown": c.unknown,
                            "track": c.evidence_track} for c in r.candidates]}


def store_routes(drafts: Any, draft_id: str, advisor: Any = None) -> dict[str, Any]:
    """Route every sub-problem of the draft's map now and keep the routes with the map and the data card they were made for. A decision is
    recorded for each sub-problem whose approach is new or changed (the data may arrive after the map was confirmed). Returns the routes' view."""
    from dclab_core import solutions
    from ..ledger import capture

    draft = drafts.get(draft_id)
    m = current(draft)
    if m is None:
        return {}
    routes = compute_routes(draft, m, advisor)
    old = ((stored(draft) or {}).get("routes") or {}).get("routes") or {}
    data = data_card_of(draft)
    digest = _data_digest(draft)  # of the data card the routes were made against: one that arrived while a model was thinking is not 'already routed'

    def keep(d: dict[str, Any]) -> None:
        if d.get("problem_map") and (current(d) or m).fingerprint == m.fingerprint:
            d["problem_map"]["routes"] = {"map": m.fingerprint, "data": digest, "routes": {k: json.loads(v.model_dump_json()) for k, v in routes.items()}}
    drafts.update(draft_id, keep)
    for sid, r in routes.items():
        if (old.get(sid) or {}).get("chosen") != r.chosen or (old.get(sid) or {}).get("source") != r.source:
            sub = next(x for x in m.sub_problems if x.id == sid)
            capture.route_chosen(draft_id, sub, r, data, len(m.parts))
            hooks.route_made(draft_id, m.fingerprint, sub, r)
    return {k: route_view(v) for k, v in routes.items()}


def stored_routes(draft: dict[str, Any], m: pm.ProblemMap) -> dict[str, Any]:
    """The routes as objects: the stored ones when they were made for this map and this data card, else the rules' now (looking asks no model)."""
    from dclab_core import solutions

    held = (stored(draft) or {}).get("routes") or {}
    if held.get("map") == m.fingerprint and held.get("data") == _data_digest(draft):
        try:
            return {k: solutions.Route.model_validate(v) for k, v in held["routes"].items()}
        except Exception:  # noqa: BLE001 - a stored route that no longer reads is routed again
            pass
    return compute_routes(draft, m)


def routes_view(draft: dict[str, Any], m: pm.ProblemMap) -> dict[str, Any]:
    """The routes for the page."""
    return {k: route_view(v) for k, v in stored_routes(draft, m).items()}


def solution_view(draft: dict[str, Any], m: Optional[pm.ProblemMap] = None) -> dict[str, Any]:
    """The solution map for the page (24.4): a block per part with its approach, its split and the steps of its workflow, the composition's steps, the
    one holdout design and what stops it (a part with no approach, an issue of the validator). Built from the map and the routes as they are now."""
    from dclab_core import solution_map, solutions

    m = m or current(draft) or propose(draft)[0]
    sm = solution_map.solution_map(m, stored_routes(draft, m), data_card_of(draft))
    parts = {p.id: p for p in m.parts}
    issues = solution_map.validate(sm, m)

    def block(b: Any) -> dict[str, Any]:
        return {"id": b.id, "part": b.part, "part_title": parts[b.part].title if b.part in parts else b.part, "sub_problem": b.sub_problem, "title": b.title, "family": b.family,
                "approach": b.approach, "approach_title": solutions.BY_ID[b.approach].title if b.approach else None, "track": b.track, "final": b.final,
                "split": json.loads(b.split.model_dump_json()),
                "defers": b.defers,
                "workflow": [{"id": s.id, "block": s.block, "name": s.name, "purpose": s.purpose, "gate": s.gate, "kind": s.kind, "deferred": s.id in b.defers} for s in b.workflow.steps]}
    return {"map": m.fingerprint, "fingerprint": sm.fingerprint, "size": m.size, "blocks": [block(b) for b in sm.blocks],
            "steps": [json.loads(s.model_dump_json()) for s in sm.steps], "holdout": json.loads(sm.holdout.model_dump_json()), "blockers": sm.blockers, "notes": sm.notes, "issues": issues}


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
    return {"drawn": drawn, "map": json.loads(m.model_dump_json()), "source": source, "reason": reason, "issues": pm.validate(m), "routes": routes_view(draft, m),
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
        d["problem_map"] = {"map": json.loads(m.model_dump_json()), "first": old.get("first") or json.loads(m.model_dump_json()),  # 24.7: the first map drawn, what the person's edits correct
                            "source": source, "reason": reason, "words": words or old.get("words") or _digest(words_of(d)),
                            "rules": rules or old.get("rules"), "routes": old.get("routes"), "review": json.loads(review.model_dump_json()), "confirmed": confirmed, "history": history[-50:]}
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
    hooks.map_drawn(draft_id, made, source == "advisor")
    refresh_review(drafts, draft_id, advisor)
    if made.size == "single":  # nothing to confirm: its approach is the detected kind, routed now
        store_routes(drafts, draft_id, advisor)
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
        d["problem_map"] = {"map": json.loads(made.model_dump_json()), "first": prior.get("first"), "source": "person" if action != "confirm" else prior.get("source", "rules"),
                            "reason": prior.get("reason", "") if action == "confirm" else "edited by the person", "words": prior.get("words") or _digest(words_of(d)), "rules": prior.get("rules"), "routes": prior.get("routes"),
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
    hooks.map_changed(draft_id, held, made, action, who, part=kw.get("part") or kw.get("keep") or kw.get("title"))
    refresh_review(drafts, draft_id, advisor)
    if action == "confirm" or made.size == "single":  # a confirmed map is routed (again, when the data arrived since)
        store_routes(drafts, draft_id, advisor)
    return view(drafts.get(draft_id), advisor)


def project_plan(draft: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The confirmed plan of a problem with several parts, kept with the project built from the draft (24.5): the map, the routes and the solution map as they were when
    the person confirmed. None for a single problem, a map nobody confirmed, or a draft with no map."""
    held, m = stored(draft), current(draft)
    if not held or m is None or m.size != "complex" or not held.get("confirmed") or stale(draft):
        return None
    from dclab_core import solution_map

    routes = stored_routes(draft, m)
    sm = solution_map.solution_map(m, routes, data_card_of(draft))
    return {"problem_map": json.loads(m.model_dump_json()), "routes": {k: json.loads(v.model_dump_json()) for k, v in routes.items()},
            "solution_map": json.loads(sm.model_dump_json()), "confirmed": True}


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
