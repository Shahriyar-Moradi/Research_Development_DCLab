"""Home's lead agent and the team agents, connected through the workspace's Lead (2026-11-21).

The way a request travels, in four steps:

1. **Home hands it over** (:func:`ask`). The person's request on Home, in their own words, becomes a top-level task owned by
   the Lead (``team_work.DIRECTORY["lead"]``, a coordinator), asked by Home (the way in, ``DIRECTORY["home"]``) for that
   person. Its answer goes back to the person's own Home thread (``threads.home_team``: one per draft). The kind of help
   the person picked travels as the task's one ``help:<track>`` input, which only the Lead reads.
2. **The Lead coordinates** (``agents._lead``, rules): it checks the task has something the teams may read, hands it to the
   team agents that kind of help needs (``team_work.lead_route``), and joins their reports into one report that says which
   team answered what.
3. **Questions travel up and answers down.** A team's question to the Lead reaches the person in their Home chat
   (:func:`to_home_chat`); the person's answer (:func:`answer`) is recorded as theirs by Home, then by the Lead, and given
   to the team that asked. No agent answers for the person.
4. **The report comes back** to the Home chat as one message with the four lines every reply has (what was found, why,
   who acts next, what changed).

Everything runs on the team agents' tools, guard and outbox (W4, W5), so a restart delivers once, a revoked member's work
is refused, and every move is recorded (W7). The Lead and Home are deterministic in this release, like the team agents.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from dclab_core import team_work as tw

Row = dict[str, Any]
EXPECTED = "one joined report from the teams the request needs, or a precise blocker"
DONE = ("says what each team found, or what blocks the request",)


# what no team does for a person: approve a decision or contact customers (only a person does either)
# what no team does for a person: approve a decision or contact customers (only a person does either). Orders only, at the start
# of a sentence or after "please", "and", "then" or "teams to": "predict which loans are approved" is an ordinary request.
_ORDER = r"(?:^|[.!?;]\s+|\bplease\s+|\band\s+|\bthen\s+|\b(?:teams?|you)\s+(?:to|should|must|can)\s+)"
NOT_THE_TEAMS = re.compile(_ORDER + r"(?:approve|sign off on|sign off)\b|" + _ORDER +
                           r"(?:send|e-?mail|call|text|message|contact)\b[^.?!]{0,30}?\bcustomers?\b", re.I)


class TeamsRefused(ValueError):
    """The request cannot be handed to the teams as asked (no project, no words, a kind of help the Lead does not take)."""


def _work(env: Any) -> Any:
    from .delivery import store_for

    return store_for(env)


def refusal(*, project_id: Optional[str], problem_id: Optional[str], question: str, track: str) -> Optional[str]:
    """Why a request may not go to the teams, or None. The checks code makes before the Lead is asked anything."""
    if not project_id or not problem_id:
        return "the teams work on a project's problem: build or link the project first"
    if not str(question or "").strip():
        return "hand the person's own words to the teams, not an empty request"
    if NOT_THE_TEAMS.search(str(question)):
        return ("the teams never approve a decision or contact customers: only a person does either; the teams can prepare the "
                "evidence for the person's decision instead")
    if not tw.lead_route(track):
        taken = ", ".join(sorted(tw.LEAD_ROUTES))
        return f"the Lead takes {taken} work; a {track or 'missing'} request stays in this conversation"
    return None


def person_of(scope: Any = None) -> tuple[str, Optional[Row]]:
    """Who the teams' work acts for, and the initiator the outbox rechecks: the signed-in person (their authorized scope, or
    the request's principal), else the local owner (no sign-in: no initiator to recheck)."""
    uid = getattr(scope, "principal_id", None)
    role = getattr(scope, "role", None)
    workspace = getattr(scope, "workspace_id", None)
    signed_in = scope is not None and getattr(scope, "auth_mode", "local") == "authenticated"
    if scope is None:
        try:
            from ..accounts.principal import current

            who = current()
        except Exception:  # noqa: BLE001 — no request: the local owner
            who = None
        uid, role, workspace = getattr(who, "user_id", None), getattr(who, "role", None), getattr(who, "workspace_id", None)
        signed_in = bool(uid) and getattr(who, "via", None) not in (None, "anonymous", "local")
    person = str(uid or "owner")  # the local owner, as the team routes name them
    initiator = {"user_id": person, "role": str(role), "workspace_id": str(workspace or "local")} if signed_in and role else None
    return person, initiator


def project_inputs(env: Any, project_id: str) -> tuple[str, ...]:
    """What the teams may read for this project: the inputs it declares for its teams (``team_inputs``), as Business reads."""
    projects = getattr(env, "projects", None)
    if projects is None:
        return ()
    try:
        return tuple(str(i) for i in (projects.get(str(project_id)) or {}).get("team_inputs") or ())
    except KeyError:
        return ()


NO_DATA_YET = ("The teams cannot read this project's data yet: in this release they read only the inputs a project declares for "
               "them; reading its own tables comes with release R3.")  # said to the person as it is (the rules show it whole): no word for a model

_NEW_REQUEST = re.compile(r"^\s*(?:please\s+)?(?:also\s+|now\s+|then\s+)?(?:ask|tell|send (?:it |this )?to|hand (?:it |this )?(?:over )?to|check with)\s+"
                          r"(?:the |our |my )?(?:teams?|lead|workspace(?:'s)? teams?|data science(?: team)?)\b[\s:,.-]*", re.I)  # at the start only (review F2)


def _request_words(text: str) -> str:
    """A request's own words: a leading "please ask the teams:" left out, then without capitals or punctuation."""
    return _plain(_NEW_REQUEST.sub("", str(text or ""), count=1))


def _plain(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def is_answer(text: str, task: Any) -> bool:
    """Whether a message the person sends while the teams' question waits answers it (the live run of 2026-11-21: a new
    question, and the same request sent again, were both recorded as the answer). Not an answer: the request itself again,
    a new request for the teams, or a question of the person's own (it ends with a question mark). Everything else is,
    as before: a person answering in a few words. A rule, not an understanding: a wrong "no" sends the message to Home's
    chat and the question keeps waiting, which the chat says; a wrong "yes" is what happened before."""
    said, asked = _request_words(text), _request_words(getattr(task, "question", ""))
    if not _plain(text):
        return False
    if asked and said == asked:  # the request again (an answer may quote part of it: "seven days before each renewal")
        return False
    lead_in = _NEW_REQUEST.match(str(text))
    if lead_in:  # "ask the teams to/whether/why …" is a new request; "tell the team: failure to renew" answers
        rest = str(text)[lead_in.end():].strip()
        if re.match(r"(?:to|whether|if|about|for|why|how|what|which|can|could|also)\b", rest, re.I) or len(rest.split()) > 8:
            return False
    return not str(text).strip().endswith("?")


_NEGATED = re.compile(r"\b(?:don t|do not|does not|doesn t|not|never|no need to|without)\s+(?:\w+\s+){0,2}$")
QUOTE_WORDS, QUOTE_SHARE = 4, 0.6


def quote_is_request(quote: Any, words: Any) -> bool:
    """Whether a model's quote of the person's request is theirs (the review of 2026-11-22: any two of their words passed):
    one run of their words in one message, at least four words or 60% of that message, and not inside a negation ("do not
    rank the accounts" is no request to rank them). Else the caller hands over their whole latest message."""
    q = _plain(quote)
    if len(q.split()) < 2:
        return False
    for message in words or ():
        said = _plain(message)
        at = re.search(r"(?<!\S)" + re.escape(q) + r"(?!\S)", said)
        if at is None:
            continue
        if len(q.split()) >= QUOTE_WORDS or len(q.split()) >= QUOTE_SHARE * len(said.split()):
            if not _NEGATED.search(said[:at.start()]):
                return True
    return False


def answered_words(out: Mapping[str, Any]) -> str:
    """What Home's chat says after the person's answer went to the Lead (the review of 2026-11-22: it said "given to the
    teams" also when Home's rules skipped it, the question being answered already). Claimed only when a step recorded it."""
    if out.get("refused"):
        return "Your answer could not be passed to the teams: " + str(out["refused"][0])
    tools = {s.get("tool") for s in out.get("steps") or () if s.get("ok", True)}
    if tools & {"team_record_human_response", "team_answer"}:
        return "The Lead has given your answer to the teams; their report will appear here."
    return "The teams' question was already answered, so this message was not passed to them; their report will appear here."


def still_waiting_due(messages: Any, task: Any) -> bool:
    """Whether Home's chat should say the teams' question still waits (said once a question: the review of 2026-11-22 saw
    it after every message): no "still waits" card for this task since its latest question."""
    tid = getattr(task, "id", None)
    for m in reversed(list(messages or ())):
        team = m.get("team") if isinstance(m, Mapping) else None
        if not isinstance(team, Mapping) or team.get("from") != "lead" or team.get("task_id") != tid:
            continue
        if team.get("kind") == "still_waiting":
            return False
        if team.get("kind") == "clarification":
            return True
    return True


def task_id_for(draft_id: str, question: str, person: str = "owner", attempt: int = 1) -> str:
    """The Lead's task for these words from this person's draft; ``attempt`` > 1 when an earlier one with the same words
    has ended (a blocked request asked again after the table was attached is a new task, never the old one silently)."""
    from .tools import _digest, safe_id

    base = safe_id(f"L-{draft_id}-{_digest(draft_id, person, _request_words(question))}")  # live S5: "Can our data" and "can our data" are one request
    return base if attempt <= 1 else f"{base}-{attempt}"


def ask(env: Any, project_id: str, problem_id: str, *, question: str, track: str, person: str, draft_id: str,
        initiator: Optional[Row] = None, inputs: tuple[str, ...] = (), registry: Any = None) -> Row:
    """Home hands the person's request to the Lead: the Lead's task, saved with its delivery (the outbox delivers it).

    The same words from the same draft again are the same task (nothing new is opened). Raises :class:`TeamsRefused` when
    :func:`refusal` refuses it. Returns the task id, the Home thread and Home's turn."""
    from . import agents, threads
    from .tools import TeamTurn

    why = refusal(project_id=project_id, problem_id=problem_id, question=question, track=track)
    if why:
        raise TeamsRefused(why)
    store = threads.threads_for(env)
    thread, made = store.open_thread(str(project_id), str(problem_id), threads.home_team(draft_id, str(person)), label="Home",
                                     by={"kind": "person", "id": str(person)})
    if made:
        threads.record_event("thread.opened", thread, {"kind": "person", "team": "home"}, "open")
    work = _work(env)
    attempt, task_id = 1, task_id_for(draft_id, question, str(person))
    same = _open_same_request(env, str(project_id), draft_id, str(person), question)
    if same is not None:  # live S5: the model quoted the request once with "Please ask the teams:" and once without: one request
        task_id = same.id
    held_task = work.task(str(project_id), task_id)
    while held_task is not None and _ended(held_task):  # ended, or stopped on a blocker, or reported partial (review F3)
        attempt += 1
        task_id = task_id_for(draft_id, question, str(person), attempt)
        held_task = work.task(str(project_id), task_id)
    held = [i for i in inputs if not str(i).startswith(agents.HELP)]
    turn = TeamTurn(env, work, str(project_id), "home", task_id=task_id, initiator=initiator, person=str(person),
                    extra={"mode": "open", "owner": "lead", "question": str(question).strip(), "problem_id": str(problem_id),
                           "thread_id": thread["id"], "inputs": [*held, f"{agents.HELP}{track}"], "expected_output": EXPECTED,
                           "done_criteria": list(DONE)})
    out = agents.run_turn(turn, registry)
    return {"task_id": task_id, "thread_id": thread["id"], "draft_id": draft_id, "refused": out["refused"],
            "opened": not out["skipped"], "already_open": held_task is not None, "turn": out}


def _ended(task: tw.TeamTask) -> bool:
    """A Lead task that asks nobody any more: closed, blocked, or partial (its report came back with a team blocked, and
    the Lead joins only once every team has returned): the same request asked again is a new attempt."""
    return not tw.is_open(task) or task.status in ("blocked", "partial")


def _open_same_request(env: Any, project_id: str, draft_id: str, person: str, question: str) -> Optional[tw.TeamTask]:
    """An open Lead task from this draft and person with these words, compared without a leading "please ask the teams:",
    capitals or punctuation, else None: the same request asked again with or without its lead-in."""
    said = _request_words(question)
    if not said:
        return None
    for t in lead_tasks(env, project_id):
        if _ended(t) or t.requester.member != person or _draft_of_task(env, t) != draft_id:
            continue
        if _request_words(t.question) == said:  # equal words only: a narrower request is its own task (review F3)
            return t
    return None


def answer(env: Any, project_id: str, task_id: str, *, person: str, text: str, initiator: Optional[Row] = None,
           response_id: Optional[str] = None, registry: Any = None) -> Row:
    """The person answered, on Home, the question a team asked through the Lead: Home records the answer as theirs and
    carries it to the Lead (which records it, gives it to the team that asked, and takes the work up again)."""
    from . import agents
    from .tools import TeamTurn, _digest

    response = response_id or f"r-{_digest(task_id, person, text)}"
    turn = TeamTurn(env, _work(env), str(project_id), "home", task_id=task_id, initiator=initiator, person=str(person), text=text,
                    extra={"mode": "answer", "response_id": response})
    return agents.run_turn(turn, registry)


def waiting(env: Any, project_id: str, draft_id: str, person: Optional[str] = None) -> Optional[tw.TeamTask]:
    """The Lead's task from this draft that waits for this person's answer now (their own question, not yet answered), or
    None. ``person`` None: anyone's (the owner's view). A question already answered waits for the teams, not the person: a
    message sent before the worker carried the answer is an ordinary message, never a second answer."""
    work = _work(env)
    rows = work.deliveries(project_id=project_id)
    for row in rows:
        if row["kind"] != "clarification" or row["to_team"] != "home":
            continue
        task = work.task(project_id, row["task_id"])
        if task is None or task.status != "waiting_for_person" or _draft_of_task(env, task) != draft_id:
            continue
        if person is not None and task.requester.member != person:
            continue
        mine = [r for r in rows if r["task_id"] == task.id]  # in delivery order, oldest first
        last_asked = max(i for i, r in enumerate(mine) if r["kind"] == "clarification")
        if not any(r["kind"] == "answer" and r["from_team"] == "home" for r in mine[last_asked + 1:]):
            return task
    return None


def _draft_of_task(env: Any, task: tw.TeamTask) -> Optional[str]:
    from . import threads

    thread = threads.threads_for(env).thread(task.requester.thread_id)
    return threads.draft_of(thread) if thread else None


def _drafts(env: Any) -> Any:
    held = getattr(env, "drafts", None)
    if held is not None and hasattr(held, "update"):
        return held
    from .delivery import _home

    home = _home(env)
    if home is None:
        return None
    from ..storage import open_stores

    return open_stores(home)[1]


def card_visible(env: Any, message: Mapping[str, Any], principal: Any, *, auth_on: bool) -> bool:
    """Whether a reader sees a Home chat card from the Lead: only who may read its Home thread (the person who asked, and the
    owner), as on every team route. A message that is not such a card is always shown."""
    from . import threads

    team = message.get("team") if isinstance(message, Mapping) else None
    if not isinstance(team, Mapping) or team.get("from") != "lead":
        return True
    if not auth_on:
        return True
    try:
        thread = threads.threads_for(env).thread(str(team.get("thread_id") or ""))
        return thread is not None and threads.access(thread, principal, threads._teams_for(env), auth_on=True) is not None
    except Exception:  # noqa: BLE001 — what cannot be checked is not shown
        return False


def card_text(row: Mapping[str, Any]) -> str:
    """The words the Home chat shows for a row from the Lead: a team's question, or the joined report as it travelled."""
    if row.get("kind") == "clarification":
        return f"The teams ask: {row['text']}\nAnswer here and the Lead carries it back to them."
    return _REVISION.sub("", str(row["text"]))  # the UI live check of 2026-11-21: "(L-b52d…@r1)" is for the task page, not the chat


_REVISION = re.compile(r"\s*\([^()\s]*@r\d+\)")


def to_home_chat(env: Any, row: Mapping[str, Any]) -> Optional[Row]:
    """The Lead's question or report, as one message in the person's Home conversation, once per delivered row (a row handed
    over again finds its message there). None when the row is not for a Home conversation or the draft is gone."""
    from . import threads

    if row.get("to_team") != "home" or row.get("kind") not in ("clarification", "reply"):
        return None
    work = _work(env)
    task = work.task(str(row["project_id"]), str(row["task_id"]))
    draft_id = _draft_of_task(env, task) if task is not None else None
    drafts = _drafts(env)
    if not draft_id or drafts is None:
        return None
    message_id = f"team-{row['id']}"
    msg = {"id": message_id, "role": "agent", "kind": "text", "text": card_text(row),
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "team": {"from": "lead", "kind": row["kind"], "task_id": row["task_id"], "item_id": row.get("item_id"),
                    "thread_id": task.requester.thread_id, "project_id": row["project_id"]}}
    added: list[bool] = []
    hold = _turn_open(drafts, draft_id)

    earlier: list[dict[str, Any]] = []

    def put(d: dict[str, Any]) -> None:
        if any(m.get("id") == message_id for m in (d.get("messages") or []) + (d.get(HELD) or [])):
            return
        if not hold and d.get(HELD):  # a card held under a lock no turn's end flushed (review F4): posted first, in order
            have = {m.get("id") for m in d.setdefault("messages", [])}
            for m in d.pop(HELD):
                if m.get("id") not in have:
                    d["messages"].append(m)
                    earlier.append(m)
        d.setdefault(HELD if hold else "messages", []).append(msg)
        added.append(True)

    try:
        drafts.update(draft_id, put)
    except KeyError:  # the draft was deleted: the row stays in the Home thread, where the owner reads it
        return None
    if hold:  # Home is still answering the message that asked: the card waits for that answer (the chat reads in order)
        if added and not _turn_open(drafts, draft_id):  # the turn ended meanwhile: its own flush may have run before this hold
            flush_held(drafts, draft_id)
        return msg if added else None
    for m in earlier:
        _emit(drafts, draft_id, m)
    if added:
        _emit(drafts, draft_id, msg)
    return msg if added else None


HELD = "team_held"  # the Lead's cards that arrived while a Home turn was still answering, posted when it ends


def _turn_open(drafts: Any, draft_id: str) -> bool:
    try:
        from ..draft.api import turn_held

        return bool(hasattr(drafts, "turn") and turn_held(drafts, draft_id))
    except Exception:  # noqa: BLE001 — no lock to ask: post at once, as before
        return False


def _emit(drafts: Any, draft_id: str, msg: Mapping[str, Any]) -> None:
    if hasattr(drafts, "emit"):
        try:
            drafts.emit(draft_id, "chat", dict(msg))
        except Exception:  # noqa: BLE001 — the page reads the saved message on its next load
            pass


def flush_held(drafts: Any, draft_id: str) -> int:
    """2026-11-21: post the Lead's cards held during a Home turn, after that turn's own answer, oldest first. Called when a
    turn ends (``draft.chat.turn``) and by a card held just as the turn ended; the move is one update, so a card is posted
    once. The number posted."""
    moved: list[dict[str, Any]] = []

    def take(d: dict[str, Any]) -> None:
        held = d.pop(HELD, None) or []
        have = {m.get("id") for m in d.setdefault("messages", [])}
        for m in held:
            if m.get("id") not in have:
                d["messages"].append(m)
                moved.append(m)

    try:
        if not (drafts.get(draft_id).get(HELD)):
            return 0
        drafts.update(draft_id, take)
    except KeyError:
        return 0
    for m in moved:
        _emit(drafts, draft_id, m)
    return len(moved)


def lead_tasks(env: Any, project_id: Optional[str] = None) -> list[tw.TeamTask]:
    """Every task the Lead owns (the workspace page's conversations with the Lead), newest first."""
    work = _work(env)
    seen: dict[str, tw.TeamTask] = {}
    for row in work.deliveries(project_id=project_id):
        if row["kind"] == "task" and row["to_team"] == "lead":
            task = work.task(row["project_id"], row["task_id"])
            if task is not None:
                seen[task.id] = task
    return sorted(seen.values(), key=lambda t: t.created_at, reverse=True)


__all__ = ["DONE", "EXPECTED", "HELD", "answered_words", "quote_is_request", "still_waiting_due", "NOT_THE_TEAMS", "NO_DATA_YET", "flush_held", "is_answer", "card_visible", "TeamsRefused", "person_of", "project_inputs", "answer", "ask", "card_text", "lead_tasks", "refusal", "task_id_for", "to_home_chat",
           "waiting"]
