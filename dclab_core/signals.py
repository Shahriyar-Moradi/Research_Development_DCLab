"""Signals of quality on a session (package 19.9): failure detectors over its events, and the corrections they show.

Each detector reads one visit's events (``dclab_core.experience``) and returns flags: what went wrong, how bad, the
events that show it, in plain words. They are rules in code, run on every session, so the owner finds the sessions
worth reading and the failures become benchmark cases (19.12). A detector never changes anything.

- ``identifier_target``: an outcome saved that the data said is an identifier (high), or, with no analysis of it,
  that is named like a key (medium; the morning a project predicted ``customer_id``). The analysis's role wins over
  the name: ``country_code`` analysed as categorical is not flagged.
- ``built_with_question_open``: a project built while a blocking question of its conversation had no answer (a
  question about the unit or the action does not block; a question about the outcome is withdrawn or asked again
  once the data arrives).
- ``task_contradicts_words``: a draft's words (its latest edit) that say anomaly, built as anything else; or that say
  rare events or anomaly, built as a plain multiclass or regression. Matched per draft (the conversation).
- ``rephrased``: the problem rewritten, or the same message sent again before any new question: the person was not
  understood.
- ``error``: an error raised or a stage failed.
- ``spend_without_result``: model spend and nothing built, run, exported or answered.
- ``build_refused_again``: a build refused more than once.
- ``question_asked_again``: the agent asked the same question twice (same field, same options): the first answer was
  not taken. Asking again with the columns to choose from is a follow-up, not a repeat.
- ``left_unbuilt``: a draft started and not built in the visit, and where it stopped.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Literal, Optional

from pydantic import Field

from dclab_core.contracts import Contract

KEY_NAME = re.compile(r"(^|[_\s.-])(id|ids|uuid|guid|key|nr|number|ref)$", re.I)  # a code (country_code) is a category
CAMEL_KEY = re.compile(r"[a-z0-9](Id|ID|Uuid|UUID|Key)$")  # customerId: case matters (is_paid is not a key)
RARE_FAMILIES = frozenset({"rare_event", "anomaly"})
PLAIN_TASKS = frozenset({"multiclass", "regression"})
# the questions a build cannot go past (dclab_core.card: blocks=True); Home asks the outcome as "target"
BLOCKING = frozenset({"labels", "family", "outcome", "target", "prediction_moment", "data"})
RESULTS = frozenset({"draft.built", "stage.started", "stage.finished", "export.made"})


class Flag(Contract):
    """One thing a detector found in a session."""

    detector: str
    severity: Literal["high", "medium", "low"]
    message: str = Field(max_length=300)
    events: list[str] = Field(default_factory=list, max_length=20, description="The events that show it.")


def _by_kind(events: list[Any], kind: str) -> list[Any]:
    return [e for e in events if e.kind == kind]


def _conversation(e: Any) -> Optional[str]:
    return getattr(e.ids, "conversation", None)


def identifier_target(events: list[Any]) -> list[Flag]:
    roles: dict[str, str] = {}
    for e in _by_kind(events, "data.analysed"):
        roles.update({str(k): str(v) for k, v in (e.payload.get("roles") or {}).items()})
    out = []
    for e in _by_kind(events, "solution.saved"):
        target = str(e.payload.get("target") or "")
        if not target:
            continue
        if target in roles:  # the analysis read the column: its role, not its name
            if roles[target] == "identifier":
                out.append(Flag(detector="identifier_target", severity="high", events=[e.id],
                                message=f"The outcome saved, {target[:80]}, is an identifier: a model of ids learns nothing."))
        elif KEY_NAME.search(target) or CAMEL_KEY.search(target):
            out.append(Flag(detector="identifier_target", severity="medium", events=[e.id],
                            message=f"The outcome saved, {target[:80]}, is named like a key, and no analysis says what it holds."))
    return out


def built_with_question_open(events: list[Any]) -> list[Flag]:
    out, open_question = [], {}  # conversation -> the blocking question it waits on
    for e in events:
        conversation = _conversation(e)
        if e.kind == "chat.agent" and e.payload.get("question_field"):
            if e.payload["question_field"] in BLOCKING:
                open_question[conversation] = e
            else:
                open_question.pop(conversation, None)  # the agent moved on to a question that does not block
        elif e.kind == "chat.person":
            open_question.pop(conversation, None)
        elif e.kind == "data.analysed" and conversation in open_question and open_question[conversation].payload["question_field"] == "target":
            open_question.pop(conversation)  # withdrawn by a label file, or asked again with the columns (a new question)
        elif e.kind == "draft.built" and conversation in open_question:
            question = open_question.pop(conversation)
            out.append(Flag(detector="built_with_question_open", severity="high", events=[question.id, e.id],
                            message=f"Built while the question about {str(question.payload['question_field'])[:60]} had no answer."))
    return out


def task_contradicts_words(events: list[Any]) -> list[Flag]:
    from dclab_core.card import problem_card
    from dclab_core.understand import understand

    out, words, said, built = [], {}, {}, {}  # conversation -> its latest words and the event that said them; project -> draft
    for e in events:
        conversation = _conversation(e)
        if e.kind == "solution.saved" and conversation is None:  # saved in the project the draft was built into
            conversation = built.get(getattr(e.ids, "project", None))
        if e.kind == "draft.built" and e.payload.get("project"):
            built[e.payload["project"]] = conversation
        elif e.kind == "draft.created" and e.payload.get("problem"):
            words[conversation], said[conversation] = str(e.payload["problem"]), e
        elif e.kind == "draft.edited" and e.payload.get("field") == "problem" and e.payload.get("after"):
            words[conversation], said[conversation] = str(e.payload["after"]), e
        elif e.kind == "solution.saved" and conversation in words:
            card = problem_card(understand(words[conversation]), words=words[conversation])
            families = set(card.in_play or ([card.family.value] if card.family.value else []))
            if not families or not families <= RARE_FAMILIES or e.payload.get("kind") in RARE_FAMILIES:
                continue
            task = e.payload.get("task")
            if task in PLAIN_TASKS or (families == {"anomaly"} and task):  # nothing labelled: no outcome to learn
                out.append(Flag(detector="task_contradicts_words", severity="high", events=[said[conversation].id, e.id],
                                message=f"The words say {' or '.join(sorted(families)).replace('_', ' ')}; the solution saved is {task}."))
    return out


def rephrased(events: list[Any]) -> list[Flag]:
    out = []
    edits = [e for e in _by_kind(events, "draft.edited") if e.payload.get("field") == "problem"]
    if len(edits) >= 2:
        out.append(Flag(detector="rephrased", severity="medium", events=[e.id for e in edits][:20],
                        message=f"The problem was rewritten {len(edits)} times: the first readings missed it."))
    said: dict[Any, dict[str, str]] = {}  # conversation -> what the person said since the agent's last question
    for e in events:
        conversation = _conversation(e)
        if e.kind == "chat.agent" and e.payload.get("question_field"):
            said.pop(conversation, None)  # the same short answer to a new question is an answer, not a resend
        elif e.kind == "chat.person":
            text = " ".join(str(e.payload.get("text") or "").lower().split())
            here = said.setdefault(conversation, {})
            if text and text in here:
                out.append(Flag(detector="rephrased", severity="medium", events=[here[text], e.id], message="The same message was sent again."))
            here.setdefault(text, e.id)
    return out


def error(events: list[Any]) -> list[Flag]:
    bad = [e for e in events if e.kind in ("error.raised", "stage.failed")]
    if not bad:
        return []
    where = sorted({str(e.payload.get("where") or e.payload.get("stage") or "?")[:40] for e in bad})
    return [Flag(detector="error", severity="medium", events=[e.id for e in bad][:20],
                 message=f"{len(bad)} error{'s' if len(bad) != 1 else ''}: {', '.join(where)[:200]}.")]


def spend_without_result(events: list[Any]) -> list[Flag]:
    spent = [e for e in events if e.kind == "model.request" and (e.payload.get("cost") or 0) > 0]
    answered = any(e.kind == "chat.agent" and e.payload.get("kind") == "answer" and e.payload.get("surface") in ("project", "intern") for e in events)
    if spent and not answered and not any(e.kind in RESULTS for e in events):
        total = sum(float(e.payload.get("cost") or 0) for e in spent)
        return [Flag(detector="spend_without_result", severity="low", events=[e.id for e in spent][:20],
                     message=f"{len(spent)} model requests ({total:.4f} in their currency) and nothing built, run or answered.")]
    return []


def build_refused_again(events: list[Any]) -> list[Flag]:
    refused = _by_kind(events, "draft.refused")
    if len(refused) >= 2:
        return [Flag(detector="build_refused_again", severity="medium", events=[e.id for e in refused][:20],
                     message=f"A build was refused {len(refused)} times: the person did not see what was missing.")]
    return []


def question_asked_again(events: list[Any]) -> list[Flag]:
    seen: dict[tuple, Any] = {}  # (conversation, field) -> the question asked
    out = []
    for e in _by_kind(events, "chat.agent"):
        field = e.payload.get("question_field")
        if not field:
            continue
        key = (_conversation(e), field)
        before = seen.get(key)
        if before is not None and (before.payload.get("options") or None) == (e.payload.get("options") or None):
            out.append(Flag(detector="question_asked_again", severity="medium", events=[before.id, e.id],
                            message=f"The question about {str(field)[:60]} was asked again."))
        seen[key] = e  # different options: a follow-up (the outcome asked again with the columns), not a repeat
    return out


def left_unbuilt(events: list[Any]) -> list[Flag]:
    created = _by_kind(events, "draft.created")
    if not created or any(e.kind == "draft.built" for e in events):
        return []
    last = events[-1]
    return [Flag(detector="left_unbuilt", severity="low", events=[created[0].id, last.id],
                 message=f"A draft was started and not built; the visit ended at {last.kind}.")]


DETECTORS = (identifier_target, built_with_question_open, task_contradicts_words, rephrased, error, spend_without_result,
             build_refused_again, question_asked_again, left_unbuilt)


def detect(events: Iterable[Any], only: Optional[Iterable[str]] = None) -> list[Flag]:
    """Every flag the detectors raise on one session's events, in their order."""
    from dclab_core.experience import timeline

    ordered = timeline(events)
    wanted = set(only) if only is not None else None
    return [flag for fn in DETECTORS if wanted is None or fn.__name__ in wanted for flag in fn(ordered)]
