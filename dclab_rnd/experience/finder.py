"""Finding sessions in the record (package 19.4): the latest visits as cards, one visit's timeline, and a search.

A search names any of: a person, a project, a draft (or intern session), a kind of event, words in what people and
agents wrote, a code (an error code or a rule), and a time window. It answers with the matching events, newest first,
and the cards of the visits they belong to, so "every session where a build was refused" is
``search(log, kind="draft.refused")`` and "every visit that touched project P" is ``search(log, project=P)``.

A kind is one of the record's kinds (``draft.refused``), a family (``draft``: every ``draft.*``), or one of two groups
the session card counts: ``refused`` (a check said no: a draft, a solution, a decision, a move, a tool call) and
``errors`` (an error raised, a stage that failed).

What a page shows is the event without its cell values (a tool's arguments and results), and without model text
(prompts and replies) unless asked: :func:`view`.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from . import REFUSED_KINDS as ALWAYS_REFUSED, ExperienceLog, cards, contract, refused

REFUSED_KINDS = (*ALWAYS_REFUSED, "graph.move", "tool.call")
GROUPS: dict[str, tuple[str, ...]] = {"refused": REFUSED_KINDS, "errors": ("error.raised", "stage.failed")}
SHOWN = ("meta", "count", "person_text", "agent_text")  # what a page shows; model text on request, cell values never
SEARCHED = ("meta", "person_text", "agent_text")         # where words are looked for
CODE_FIELDS = ("code", "rule", "rules", "failed_checks")
SCAN = 20_000   # the events a search reads at most (newest first); the answer says when it stopped there
MAX_VISITS = 50


def kinds_for(kind: Optional[str]) -> tuple[Optional[frozenset[str]], bool]:
    """The kinds a search reads for ``kind``, and whether only refusals among them (``refused``: the log filters them)."""
    if not kind:
        return None, False
    if kind in contract.KINDS:
        return frozenset({kind}), False
    if kind in GROUPS:
        return frozenset(GROUPS[kind]), kind == "refused"
    family = frozenset(k for k in contract.KINDS if k.split(".")[0] == kind)
    if family:
        return family, False
    raise ValueError(f"unknown kind {kind!r}: a kind of the record, a family ({', '.join(families())}) or refused / errors")


def families() -> list[str]:
    return sorted({k.split(".")[0] for k in contract.KINDS})


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)


def _has_text(event: Any, words: str) -> bool:
    declared = contract.KINDS[event.kind]
    want = words.casefold()
    return any(want in s.casefold() for k, v in event.payload.items() if declared.get(k) in SEARCHED for s in _strings(v))


def _has_code(event: Any, code: str) -> bool:
    want = code.casefold()
    return any(s.casefold() == want for k in CODE_FIELDS for s in _strings(event.payload.get(k)))


def search(log: ExperienceLog, *, person: str | None = None, project: str | None = None, draft: str | None = None,
           kind: str | None = None, text: str | None = None, code: str | None = None, since: str | None = None,
           until: str | None = None, visit: str | None = None, limit: int = 50, scan: int = SCAN) -> dict[str, Any]:
    """The events that match every condition given, newest first (at most ``limit``), and their visits' cards."""
    kinds, refused_only = kinds_for(kind)
    read = log.read(visit=visit, project=project, conversation=draft, user=person, since=since, until=until, kinds=kinds,
                    refused_only=refused_only, limit=scan)
    capped = len(read) >= scan
    found = [e for e in reversed(read) if (not text or _has_text(e, text)) and (not code or _has_code(e, code))]
    visits: list[str] = []
    for e in found:
        if e.ids.visit and e.ids.visit not in visits:
            visits.append(e.ids.visit)
    return {"total": len(found), "events": found[:limit], "sessions": cards(log, visits[:MAX_VISITS]), "visits": len(visits),
            "scanned": len(read), "capped": capped}


def view(event: Any, model_text: bool = False) -> dict[str, Any]:
    """The event as a page shows it: cell values out always, model text out unless asked; what was left out is named."""
    allowed = SHOWN + (("model_text",) if model_text else ())
    shown = contract.redact(event, allowed)
    return {"id": shown.id, "at": shown.at, "seq": shown.seq, "kind": shown.kind, "actor": shown.actor,
            "ids": {k: v for k, v in shown.ids.model_dump().items() if v is not None}, "payload": shown.payload,
            "redacted": shown.redacted}
