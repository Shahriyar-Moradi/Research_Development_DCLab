"""Decision models for data scientists (package 22.8): a technical decision asked while one works.

``ask(point, state)`` answers one of DCLab's technical decisions about a column from its state (its name, its name
tokens and bands: never a value; ``ledger.column_state`` builds one from a profiled column) the same way wherever it is
asked: the API (``POST /api/decide``), the MCP tool and the intern (``decide``, a read tool of the registry), and the
notebook companion (``python -m dclab_rnd.notebook_assist --decide``). The answer carries the rule's answer, every
classifier's answer with its confidence and its sealed verdict, the answer DCLab would act on under the workspace's
routing (22.6), whether they disagree (shown, never hidden), and the evidence records the decision rests on.

On a project, the rule's answer is the project's own: the decision its studio rules recorded for that column
(``recorded``), and what the project did with it. Asked with a state alone, the rule is ``rule:bands-approx``, read
from the bands: an approximation of the studio's rules, which read the whole table; the answer says so
(``rule_source``). ``choose`` records a person's own answer as a decision (the answers shown beside it), and as a label
when it differs from any answer shown (the person saw them: ``user_decision_exposed``); a person's later answer on the
same column and point supersedes their earlier label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

TECHNICAL = ("column.is_identifier", "column.time", "column.group", "column.category_code", "column.semantic_role", "feature.leakage_suspect")
EVIDENCE = {"column.is_identifier": ("DCLAB-R04", "DCLAB-R02"), "column.group": ("DCLAB-R02",), "column.time": ("DCLAB-R02",),
            "column.category_code": ("DCLAB-R11",), "column.semantic_role": ("DCLAB-R11", "DCLAB-R03"),
            "feature.leakage_suspect": ("DCLAB-R04", "DCLAB-R05", "DCLAB-R01")}
OPTIONS = {"column.semantic_role": ["numeric", "categorical_code", "identifier", "datetime", "free_text", "other"]}
ID_WORDS = {"id", "key", "uuid", "guid", "code", "number", "no", "ref"}
TIME_WORDS = {"date", "time", "timestamp", "at", "day", "month", "year", "created", "updated"}
CODE_WORDS = {"code", "type", "category", "class", "level", "zone", "region", "status"}
LEAK_WORDS = {"refund", "final", "outcome", "actual", "after", "post", "settled", "cancellation", "duration", "chargeback", "result"}
UNIQUE = {"all", "over_99pct"}
MANY = {"from_50_to_90pct", "from_90_to_99pct", *UNIQUE}
BANDS_RULE = "rule:bands-approx"


class DecideError(ValueError):
    """A question that cannot be asked as given; the reason says why."""


def options_of(point: str) -> list[str]:
    return list(OPTIONS.get(point) or ["yes", "no"])


def rule(point: str, state: dict[str, Any]) -> Optional[Any]:
    """An approximation of the studio's rule from the bands alone, as an ``Answer`` (``rule:bands-approx``)."""
    from dclab_core.ledger import Answer

    tokens = {str(t).lower() for t in state.get("name_tokens") or []}
    dtype, unique, pattern = state.get("dtype"), state.get("uniqueness"), state.get("value_pattern")
    dated = dtype == "datetime" or pattern == "date_like"
    if point == "column.is_identifier":
        value = bool(not dated and ((unique in UNIQUE and dtype in ("integer", "string") and pattern != "long_text") or (tokens & ID_WORDS and unique in MANY)))
    elif point == "column.time":
        value = bool(dated or (tokens & TIME_WORDS and dtype in ("datetime", "string", "integer")))
    elif point == "column.group":
        value = bool(tokens & ID_WORDS and unique not in UNIQUE and unique not in (None, "none"))
    elif point == "column.category_code":
        value = bool(dtype == "integer" and (tokens & CODE_WORDS or state.get("cardinality") in ("low", "very_low")))
    elif point == "column.semantic_role":
        value = ("identifier" if rule("column.is_identifier", state).value else "datetime" if rule("column.time", state).value
                 else "free_text" if pattern == "long_text" else "categorical_code" if dtype in ("string", "categorical", "boolean")
                 or rule("column.category_code", state).value else "numeric" if dtype in ("integer", "float") else "other")
    elif point == "feature.leakage_suspect":
        value = bool(tokens & LEAK_WORDS)
    else:
        return None
    return Answer(by="rule", answerer=BANDS_RULE, value=value, reason="from the name tokens and bands alone: an approximation of the studio's rules")


def _evidence(point: str) -> list[dict[str, Any]]:
    from .tools import get_record

    out = []
    for rid in EVIDENCE.get(point, ()):
        try:
            record = get_record(rid)
        except Exception:  # noqa: BLE001 - a missing record is left out, never invented
            continue
        if record and not record.get("error"):
            out.append({"id": rid, "title": record.get("title") or str(record.get("text") or "")[:160]})
    return out


def workspace_routing(home: Optional[str] = None) -> tuple[dict[str, Any], str]:
    """The routing of the workspace this runs in, and where it was read: the server's gateway, else the workspace folder
    (``home``, or the settings' folder) when it is one (it holds ``.dclab_workspace``); never a new workspace."""
    try:
        if home is None:
            from .models import installed

            store = getattr(installed(), "routing", None)
            if store is not None:
                return store.get(), "workspace"
        from .settings import Settings

        from .storage.url import database_url

        folder = Path(home) if home is not None else Path(Settings().agent_home)
        if not folder.is_dir():
            return {}, "none: no workspace folder"
        if database_url(required=False):  # on PostgreSQL only a folder already bound to a workspace: never a new one
            if not (folder / ".dclab_workspace").exists():
                return {}, "none: the folder is not a workspace's"
            from .models.routing import open_routing

            return open_routing(folder).get(), "workspace"
        from .models.routing import FileRouting

        return FileRouting(folder / "model_routing.json").get(), "workspace"
    except Exception:  # noqa: BLE001 - no routing to read: the rule answers, and the answer says so
        return {}, "none: unreadable"


def _check(point: str, state: dict[str, Any], options: Optional[list[Any]]) -> tuple[dict[str, Any], list[str]]:
    from dclab_core.ledger import check_state

    if point not in TECHNICAL:
        raise DecideError(f"{point!r} is not a technical decision one asks about a column: {', '.join(TECHNICAL)}")
    if not isinstance(state, dict):
        raise DecideError("the state is an object of bands")
    state = {k: v for k, v in state.items() if v is not None}
    problems = check_state(state)
    if problems:
        raise DecideError("the state holds what is not a band or a name: " + "; ".join(problems[:5]))
    allowed = options_of(point)
    chosen = [str(o) for o in (options or allowed)]
    if any(o not in allowed for o in chosen):
        raise DecideError(f"the options at {point} are among: {', '.join(allowed)}")
    return state, chosen


def ask(point: str, state: dict[str, Any], options: Optional[list[Any]] = None, routing_doc: Optional[dict[str, Any]] = None,
        recorded: Any = None, home: Optional[str] = None) -> dict[str, Any]:
    """One technical decision answered from a column's state: what DCLab would do, with the rule (the project's recorded
    one when ``recorded`` is its decision record), the classifiers and their confidence, the disagreement, the evidence."""
    from dclab_core.answerers import RecordedRule, _key
    from dclab_core.ledger import POINTS

    from .ledger import routing, shadow

    state, options = _check(point, state, options)
    if recorded is not None:
        ruled, source = RecordedRule().of(recorded), "the project's recorded decision"
    else:
        ruled, source = rule(point, state), "bands (an approximation: a project's own rules read its whole table)"
    if routing_doc is None:
        routing_doc, where = workspace_routing(home)
    else:
        where = "workspace"
    models = [m for m in shadow.answerers() if point in m.points] if shadow.banded(state) else []
    verdicts = shadow.gate_results()
    answers = []
    for m in models:
        try:
            got = m.answer(point, state, options)
        except Exception:  # noqa: BLE001 - a classifier that fails answers nothing
            got = None
        if got is not None:
            answers.append({"answerer": got.answerer, "value": got.value, "confidence": got.confidence,
                            "sealed_gate": verdicts.get(got.answerer) or "not judged"})
    value, used, level, _, review = routing.answer(routing_doc, point, state, options, ruled, models)
    if recorded is not None and level < 2:  # the project's own decision is what it did (a person may have overridden the rule)
        value, used = recorded.value_used, recorded.used
    differ = sorted({a["answerer"] for a in answers if ruled is not None and a["value"] is not None and _key(a["value"]) != _key(ruled.value)})
    return {"point": point, "question": POINTS[point].about, "answer": value, "decided_by": used, "level": level, "review": review,
            "rule": {"answerer": ruled.answerer, "value": ruled.value, "reason": ruled.reason} if ruled is not None else None,
            "rule_source": source, "routing": where, "models": answers, "disagree": differ, "evidence": _evidence(point), "options": options,
            "note": ("A model differs from the rule: shown, never hidden; the rule's answer stands unless a model was routed to act."
                     if differ else None)}


def choose(point: str, state: dict[str, Any], value: Any, by: str, asked: Optional[dict[str, Any]] = None, ledger: Any = None) -> dict[str, Any]:
    """A person's own answer: recorded as a decision (the answers they were shown beside it), and as a label when it
    differs from any of them; it supersedes the same person's earlier label on this column and point."""
    from dclab_core.answerers import _key, _option
    from dclab_core.ledger import POINTS, Answer, answer_problem

    from . import ledger as book

    if value is None:
        raise DecideError("say which answer: a value")
    state, options = _check(point, state, None)
    asked = asked or ask(point, state)
    chosen = _option(value) if POINTS[point].answer == "noul" else value
    problem = answer_problem(POINTS[point], chosen)
    if problem:
        raise DecideError(f"{value!r} is not an answer at {point}: {problem}")
    shown = [*([Answer(by="rule", answerer=asked["rule"]["answerer"], value=asked["rule"]["value"])] if asked.get("rule") else []),
             *(Answer(by="classifier", answerer=m["answerer"], value=m["value"], confidence=m["confidence"]) for m in asked["models"])]
    subject = str(state.get("column") or "column")
    record = book.decide(point, ("column", subject), answers=[*shown, Answer(by="person", answerer="person", value=chosen)],  # who: in the label only
                         value_used=chosen, used="person", state=state, options=options, ledger=ledger)
    if record is None:
        raise DecideError("the decision was not kept: no ledger for this workspace, or it refused the record")
    labelled = None
    if any(a.value is not None and _key(a.value) != _key(chosen) for a in shown):
        store = ledger if ledger is not None else book.current_ledger()
        mine = [x for x in (store.labels(point=point) if store is not None else []) if x.by == str(by)[:120] and x.source == "user_decision_exposed"]
        earlier = {r.id for r in (store.records(point=point) if store is not None else []) if r.subject.key == subject}
        previous = sorted((x for x in mine if x.record in earlier), key=lambda x: x.at)
        labelled = book.label(record, chosen, "user_decision_exposed", str(by)[:120], supersedes=previous[-1].id if previous else None, ledger=ledger)
    return {"recorded": record.id, "label": labelled.id if labelled is not None else None}
