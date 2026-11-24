"""The helper arm's seats (B2 of the team workspace, 2026-11-23): code decides; a model helps at named seats, and its
answer is used only where a code check passes.

A role switched to ``helper`` (``brains.switch``) gets the seats ``dclab_core.team_brain.ROLE_SEATS`` names for it.
Every other role, and every role on the rules (the default), runs exactly as before: no seat is reached, nothing is asked.

**Before the rules act** (``BEFORE_SEATS``): the answer changes what the rules do next, so the turn waits for it, within
``DCLAB_TEAM_HELPER_SEAT_SECONDS`` (20) for the whole turn (:func:`seat_seconds`). The delivery's time limit grows by the
same seconds for a turn where such a seat can run (``delivery.timeout_seconds``), and only there: a turn on the rules
keeps its limit. When the seconds run out, the rules' answer stands and the seat is recorded as ``timeout``.

- ``read_target`` (the requester's helper: Business for its member, the Lead for the person on Home; asked in the turn of
  the team that owns the outcome). Where Data Science would ask what the outcome means (``agents._needs_outcome``), and
  where the keyword reader cannot read the person's answer (``target_contract.read_event`` returns None), the model reads
  the words and names one allowed event, or ``unsure``. Code checks it: an allowed code, the same event as the keyword
  reader's whenever that one reads the words, and a mature label column of the task's table records it
  (``target_contract.agreement``). Only then is it the task's outcome: "customers who quietly leave" read as non-renewal
  on a renewal table goes on without a question. Otherwise the rules ask, as before. The model is never shown which event
  the table records, nor the events in the table's order (it could only echo it). The save is code, not a tool call: it
  makes the guard's access check itself (before asking and before writing), and it is kept in ``team.target.confirmed``
  (``read_by: helper``) and the seat's record, but it is no ``agent.step`` of the turn's run.
- ``route_teams`` (the Lead's). Where the request's words do not name outreach (``team_work.OUTREACH``), the model says
  whether it asks whom to call (``sales``), not (``none``) or ``unsure``; code checks the code is allowed and that Sales
  answers in this release. A Sales task it opened keeps Sales in the Lead's route on later turns (the child task is the
  record), and the seat is asked only before the Lead delegates anything; a run stopped between two delegations reads the
  seat's own used record from the experience log (a log that cannot be read, or lost the event, leaves the rules' route).
- ``word_question`` (Data Science's). When the rules must ask, the model picks one of the questions the rules allow and
  words it plainly; code checks the id, the length, one question mark, no number, no promise, and that every option the
  rules' question offers is still named (``team_brain.check_question``). Otherwise the rules' wording is asked.

**After the rules** (``AFTER_SEATS``): they change nothing the rules do, so they are asked on B1's capped background thread
once the turn has returned a report, and never waited on. ``explain_report``: plain words checked by ``cited.check`` (each
sentence cites the report, every number is the report's, no production-ready, a word limit); one failing sentence drops
it all. ``list_risks``: advice only, each risk short, no number the report does not write, no overclaim, no promise
(``critic_gate`` recomputes a campaign stage's selection rule; a team report has none, so it does not apply here). What
passes is posted in the thread where the report is read, marked as the helper's; the report itself is never changed, and
the risks never block or approve anything.

Every seat that is reached records one ``team.helper.seat`` event and one ledger record (``team.helper_seat``): the seat,
whose helper, the outcome (``used``, or why the rules' answer stood), a code seat's answer as a code, the check's refusal
as a code, and the model request whose exchange keeps the reply and its thinking (the gateway keeps it whole: the team
purposes are never shown a table's cells). The prompts ask only for the answer's fields; never for reasoning.
"""

from __future__ import annotations

import atexit
import contextvars
import json
import os
import re
import sys
import threading
import time
from typing import Any, Mapping, Optional, Sequence

from dclab_core import team_brain as core

ENV = "DCLAB_TEAM_HELPER_SEAT_SECONDS"
DEFAULT_SECONDS = 20.0
PURPOSE = "team_helper"
MAX_TOKENS = 4000  # room for the thinking and a short answer
WORDS = 1500  # the person's words a model reads, at most
EXPLAIN_WORDS = 120
RISKS, RISK_WORDS = 5, 30
HELPER_READ = "helper.request."  # a target read from the request by the helper: its ``response`` starts with this
HELPER_ANSWER = "helper.answer."  # the person's answer, read by the helper where the keyword reader could not (review F7)


def seat_seconds() -> float:
    """The seconds one turn may wait on its before-seats, in all (``DCLAB_TEAM_HELPER_SEAT_SECONDS``; 0: never wait)."""
    try:
        return max(0.0, float(os.environ.get(ENV) or DEFAULT_SECONDS))
    except ValueError:
        return DEFAULT_SECONDS


def _base(name: Any) -> str:
    return str(name or "").split(":")[0]


def _json(reply: Any) -> Optional[dict[str, Any]]:
    text = str((reply or {}).get("content") or "").strip() if isinstance(reply, Mapping) else ""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _code(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    return text if re.fullmatch(r"[a-z0-9_]{1,60}", text) else None


def _system(seat: str, **values: Any) -> str:
    from .. import prompts

    return prompts.text("team_helper") + "\n\n" + prompts.text(f"team_helper_{seat}", **values)


# ---------------------------------------------------------------------------------------------------------- asking
def _client(turn: Any) -> tuple[Any, Optional[str]]:
    """The helper's client, or why there is none (a fallback code)."""
    from . import brains

    try:
        gateway = brains._gateway(turn)
        client = gateway.client(PURPOSE, project_id=turn.project_id) if gateway is not None else None
    except Exception:  # noqa: BLE001 - a store that cannot say whether the project is private: the rules' answer
        return None, "error"
    if client is None:
        return None, "no_live_models" if os.environ.get("DCLAB_NO_LIVE_MODELS") == "1" else "no_model"
    client.request_cells = False  # the team's records, never a table's cells: the exchange is kept whole with its thinking
    return client, None


def _failure(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, RuntimeError) and str(exc).startswith("BudgetExceeded"):
        return "refused"
    return "error"


class Late:
    """A request still out when its seat's seconds ran out. The turn has gone on with the rules' answer; once the request
    ends (the gateway keeps its exchange and thinking either way), its seat is recorded with the request id, so the record
    joins its exchange (the review of B2, F8)."""

    def __init__(self) -> None:
        self.lock, self.done, self.request, self.then = threading.Lock(), False, None, None

    def finish(self, request: Optional[str]) -> None:
        with self.lock:
            self.done, self.request, then = True, request, self.then
        if then is not None:
            then(request)

    def after(self, fn: Any) -> None:
        with self.lock:
            if not self.done:
                self.then = fn
                return
            request = self.request
        fn(request)


_OUT: set[threading.Thread] = set()
_OUT_LOCK = threading.Lock()


def drain(seconds: float = 30.0) -> bool:
    """Wait up to ``seconds`` for the seat requests still out (a test, a process before it exits); True when none is."""
    deadline = time.monotonic() + seconds
    while True:
        with _OUT_LOCK:
            out = list(_OUT)
        if not out:
            return True
        left = deadline - time.monotonic()
        if left <= 0:
            return False
        out[0].join(left)


atexit.register(drain, 70.0)  # a process that ends never cuts off a late seat's exchange, its thinking or its record


def _send(client: Any, messages: list[dict[str, Any]], wait: float) -> tuple[Any, Optional[str], Optional[str], Optional[Late]]:
    """One request (no retry) given at most ``wait`` seconds, on a thread of its own in this context (its run id and span):
    a model that does not answer in time never holds the turn longer; its late exchange is still kept by the gateway.
    Returns the reply, the fallback code (None when it answered), the request id, and on a timeout the :class:`Late`
    request (its id comes when it ends)."""
    box: dict[str, Any] = {}
    late = Late()
    context = contextvars.copy_context()

    def go() -> None:
        try:
            box["reply"] = client.complete(messages, None, MAX_TOKENS, single_attempt=True, timeout_seconds=wait)
        except BaseException as exc:  # noqa: BLE001 - handed back below
            box["error"] = exc
        finally:
            try:
                late.finish(getattr(client, "last_request", None))
            except Exception as exc:  # noqa: BLE001 - a late record that fails never fails anything else
                print(f"teamspace: a late helper seat was not recorded ({type(exc).__name__})", file=sys.stderr, flush=True)
            with _OUT_LOCK:
                _OUT.discard(threading.current_thread())

    # the whole body in the turn's context: the request, and a late seat's record after it (its workspace, run and span)
    thread = threading.Thread(target=lambda: context.run(go), name="dclab-team-helper-seat", daemon=True)
    with _OUT_LOCK:
        _OUT.add(thread)
    try:
        thread.start()
    except BaseException:
        with _OUT_LOCK:
            _OUT.discard(thread)
        raise
    thread.join(wait)
    if thread.is_alive():
        return None, "timeout", None, late
    request = getattr(client, "last_request", None)
    if "error" in box:
        return None, _failure(box["error"]), request, None
    return box.get("reply"), None, request, None


# ---------------------------------------------------------------------------------------------------------- one turn's seats
class Seats:
    """The helper seats of one turn: which roles are on the helper arm, the seconds the before-seats have spent, each
    seat's answer kept for the turn (the rules ask the same question again within a turn: one request, one answer), and
    the records of the before-seats (the turn's summary in ``team.brain.answered``)."""

    def __init__(self, turn: Any, arm: str = core.DEFAULT_ARM, run: Optional[str] = None):
        self.turn, self.run, self.spent = turn, run, 0.0
        self.arms: dict[str, str] = {_base(turn.agent): arm}
        self.memo: dict[tuple[str, ...], Any] = {}
        self.records: list[dict[str, Any]] = []
        self.late: Optional[Late] = None  # the last before-seat's request, when it was still out at its timeout

    # -- who sits
    def arm(self, role: str) -> str:
        role = _base(role)
        if role not in self.arms:
            from . import brains

            self.arms[role] = brains.arm_for(self.turn.env, role)[0]
        return self.arms[role]

    def on(self, role: str, seat: str) -> bool:
        role = _base(role)
        return seat in core.ROLE_SEATS.get(role, ()) and self.arm(role) == "helper"

    def left(self) -> float:
        return seat_seconds() - self.spent

    # -- one before-seat
    def _ask(self, seat: str, user: str, **values: Any) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[str], float]:
        """Ask the helper at a before-seat within what is left of the turn's seconds: the parsed answer (None when there is
        none), the fallback code, the request id and the seconds the turn waited."""
        wait = min(self.left(), _purpose_seconds())
        if wait <= 0:
            return None, "timeout", None, 0.0
        client, why = _client(self.turn)
        if client is None:
            return None, why, None, 0.0
        started = time.monotonic()
        try:
            messages = [{"role": "system", "content": _system(seat, **values)}, {"role": "user", "content": user}]
        except Exception:  # noqa: BLE001 - a prompt that cannot be built: the rules' answer
            return None, "error", None, 0.0
        reply, why, request, self.late = _send(client, messages, wait)
        seconds = time.monotonic() - started
        self.spent += seconds
        if why is not None:
            return None, why, request, seconds
        found = _json(reply)
        return found, (None if found is not None else "invalid_output"), request, seconds

    def _record(self, role: str, seat: str, outcome: str, *, task: Any = None, rule: Any = None, answer: Any = None,
                check: Optional[str] = None, request: Optional[str] = None, seconds: Optional[float] = None,
                report: Optional[str] = None) -> None:
        row = {"role": _base(role), "seat": seat, "outcome": outcome}
        if seat in core.BEFORE_SEATS:
            self.records.append({**row, "request": request})
        late, self.late = self.late, None
        if outcome == "timeout" and late is not None:  # recorded when the request ends, with its id (its exchange keeps the thinking)
            late.after(lambda req: record_seat(self.turn, row["role"], seat, outcome, task=task, rule=rule, answer=answer, check=check,
                                               request=req, seconds=seconds, report=report, run=self.run))
            return
        record_seat(self.turn, row["role"], seat, outcome, task=task, rule=rule, answer=answer, check=check, request=request,
                    seconds=seconds, report=report, run=self.run)

    # -- read_target
    def read_target(self, task: Any, words: Optional[str], *, where: str) -> Optional[str]:
        """The event the helper read in the person's words, when its check passed; else None (the rules' path: ask, or
        "unsure"). ``where``: ``request`` (the brief, before Data Science asks) or ``answer`` (the person's answer)."""
        from dclab_core import target_contract as tc

        from . import fixtures
        from .tools import _digest, sample_of

        role = _base(task.requester.team)
        if not self.on(role, "read_target") or not str(words or "").strip():
            return None
        key = ("read_target", task.id, where, _digest(str(words)))
        if key in self.memo:
            return self.memo[key]
        sample = sample_of(task)
        offered = fixtures.offered(sample)
        labels = fixtures.labels(sample) if sample else []
        rule = tc.read_event(words, offered)
        used: Optional[str] = None
        from .agents import _ambiguous_target

        if not labels:  # nothing declares what the table records: no reading could pass the check, so none is asked
            self._record(role, "read_target", "check_failed", task=task, rule=rule or core.UNSURE, check="no_label")
        elif rule is None and len({d.event for d in labels}) > 1 and _ambiguous_target(str(words)):
            # B4.1 (D-10 of the first dev run: "Customers are leaving us" read as non-renewal; the person meant cancelling): a
            # vague word ("leaving", "churn") on a table that records more than one of the events fits each of them, so only
            # the person can say which they mean. No reading could pass, so none is asked: the rules ask the person
            self._record(role, "read_target", "check_failed", task=task, rule=core.UNSURE, check="words_fit_several")
        else:
            # the review of B2, F1: in DCLab's own order, never the table's (the table's events come first in ``offered``, so
            # that order would tell the model which event the table records, and an echo would pass the label check)
            allowed = "\n".join(f"- {code}: {tc.event_label(code)}" for code in tc.EVENTS if code in offered)
            user = (f"The person's words ({'their request' if where == 'request' else 'their answer to the question what the outcome means'}):\n"
                    f"{str(words)[:WORDS]}\n\nEvents allowed:\n{allowed}")
            found, why, request, seconds = self._ask("read_target", user)
            answer = _code((found or {}).get("event")) if why is None else None
            check = None
            if why is None and answer is None:
                why = "invalid_output"
            elif why is None:
                check = core.check_code(answer, offered)
                if check == core.UNSURE:
                    why, check = "unsure", None
                elif check is None and rule is not None and answer != rule:
                    check = "disagrees_with_rule"  # the keyword reader reads the words clearly, and otherwise
                elif check is None:
                    verdict = tc.agreement(_probe(task, answer), labels)
                    check = None if verdict.ok else f"label_{verdict.kind}"
                if check is not None:
                    why = "check_failed"
            outcome = why or "used"
            used = answer if outcome == "used" else None
            self._record(role, "read_target", outcome, task=task, rule=rule or core.UNSURE, answer=answer, check=check, request=request,
                         seconds=seconds)
        self.memo[key] = used
        return used

    # -- route_teams
    def route(self, task: Any, teams: tuple[str, ...], track: str, view: Mapping[str, Any]) -> tuple[str, ...]:
        """The Lead's route with Sales added where the helper read outreach the keyword rule missed and its check passed;
        else the rules' route. Asked once, before the Lead delegates anything (a later turn reads the Sales task it opened)."""
        from dclab_core import team_work as tw

        if not self.on("lead", "route_teams") or not teams or "sales" in teams or str(track) not in tw.OUTREACH_TRACKS:
            return teams
        key = ("route_teams", task.id, f"b{task.brief_revision}")
        if key not in self.memo:
            if view.get("children"):
                # the Lead already handed work out without Sales: the seat is not asked again, but an earlier answer it used
                # stands (the review of B2, F4: a run stopped between two delegations resumes with Sales still in the route)
                self.memo[key] = _routed_sales(self.turn, task)
            else:
                self.memo[key] = self._route(task, teams, track)
        return teams + ("sales",) if self.memo[key] else teams

    def _route(self, task: Any, teams: tuple[str, ...], track: str) -> bool:
        from .agents import BRAINS

        user = (f"The request:\n{str(task.question)[:WORDS]}\n\nKind of help: {track}.\nTeams already chosen: {', '.join(teams)}.\n"
                "Answers allowed: sales, none, unsure.")
        found, why, request, seconds = self._ask("route_teams", user)
        answer = _code((found or {}).get("team")) if why is None else None
        check = None
        if why is None and answer is None:
            why = "invalid_output"
        elif why is None:
            check = core.check_code(answer, ("sales", "none"))
            if check == core.UNSURE:
                why, check = "unsure", None
            elif check is None and answer == "sales" and "sales" not in BRAINS:
                check = "sales_cannot_answer"
            if check is not None:
                why = "check_failed"
        outcome = why or "used"
        self._record("lead", "route_teams", outcome, task=task, rule="none", answer=answer, check=check, request=request, seconds=seconds)
        return outcome == "used" and answer == "sales"

    # -- word_question
    def question(self, task: Any, qid: str, rules_text: str, options: tuple[str, ...] = ()) -> str:
        """The clarifying question as the helper worded it, when its check passed; else the rules' own words."""
        role = _base(self.turn.agent)
        if not self.on(role, "word_question"):
            return rules_text
        key = ("word_question", task.id, qid, f"b{task.brief_revision}")
        if key in self.memo:
            return self.memo[key]
        user = (f"The person's request:\n{str(task.question)[:WORDS]}\n\nQuestions allowed:\n- {qid}: {rules_text}")
        found, why, request, seconds = self._ask("word_question", user)
        picked = _code((found or {}).get("question_id")) if why is None else None
        text = " ".join(str((found or {}).get("question") or "").split())
        check = None
        if why is None and (picked is None or not text):
            why = "invalid_output"
        elif why is None:
            check = core.check_code(picked, (qid,))
            check = "not_allowed" if check == core.UNSURE else check
            check = check or core.check_question(text, options=options)
            if check is not None:
                why = "check_failed"
        outcome = why or "used"
        self._record(role, "word_question", outcome, task=task, rule=qid, answer=picked, check=check, request=request, seconds=seconds)
        self.memo[key] = text if outcome == "used" else rules_text
        return self.memo[key]

    # -- the turn's summary
    def summary(self) -> Optional[dict[str, Any]]:
        """The before-seats of this turn as the model's part of ``team.brain.answered`` (None when none was reached): the
        helper acted when one of its answers was used; else the first seat's reason is the turn's fallback."""
        mine = self.own()
        if not mine:
            return None
        used = [r for r in mine if r["outcome"] == "used"]
        last = next((r["request"] for r in reversed(mine) if r.get("request")), None)
        return {"purpose": PURPOSE, "tool": None, "request": last, "fallback": None if used else mine[0]["outcome"],
                "used": len(used), "seats": len(mine)}

    def own(self) -> list[dict[str, Any]]:
        """This turn's role's own before-seats (the review of B2, F5: Business's reading, asked in Data Science's turn, is
        Business's, never credited to Data Science's arm)."""
        return [r for r in self.records if r["role"] == _base(self.turn.agent)]

    def acted(self) -> bool:
        return any(r["outcome"] == "used" for r in self.own())


def _routed_sales(turn: Any, task: Any) -> bool:
    """Whether this task's route seat was answered ``sales`` and used on an earlier turn (its ``team.helper.seat`` record)."""
    from .. import experience

    log = experience.current_log()
    if log is None:
        return False
    try:
        rows = log.read(project=turn.project_id, kinds=["team.helper.seat"], limit=2000)
    except Exception:  # noqa: BLE001 - the record cannot be read: the route as the rules give it
        return False
    return any(e.payload.get("task") == task.id and e.payload.get("seat") == "route_teams" and e.payload.get("outcome") == "used"
               and e.payload.get("answer") == "sales" for e in rows)


class _Rules(Seats):
    """A turn with no helper seats (outside a turn, or no seat state): every answer is the rules'."""

    def on(self, role: str, seat: str) -> bool:
        return False


def _purpose_seconds() -> float:
    from ..models import settings as model_settings

    try:
        return float(model_settings.purpose(PURPOSE).timeout)
    except Exception:  # noqa: BLE001
        return 60.0


def _probe(task: Any, event: str) -> Any:
    """A contract with only the event, to ask the label declaration whether it records it (never saved)."""
    from dclab_core import target_contract as tc

    return tc.TargetContract(event=event, words="(the helper's reading)", brief_revision=max(1, int(task.brief_revision)),
                             confirmed_by="helper")


def attach(turn: Any, arm: str, run: Optional[str] = None) -> Seats:
    """The seats of this turn, kept on it (the rules' functions and the tools reach them through :func:`of`)."""
    seats = Seats(turn, arm, run)
    try:
        setattr(turn, "_helper_seats", seats)
    except Exception:  # noqa: BLE001 - a turn that takes no attribute: its seats are the rules'
        pass
    return seats


def of(turn: Any) -> Seats:
    found = getattr(turn, "_helper_seats", None)
    return found if isinstance(found, Seats) else _Rules(turn)


# Where each before-seat is asked: in the turn of which recipient, and whose arm decides (the review of B2, F6). The Lead's
# route in the Lead's own turn; Data Science's question in its own turn; the requester's reading in Data Science's turn.
OWN_TURN_SEATS = {"lead": ("route_teams",), "data_science": ("word_question",)}
REQUESTER_SEATS = {"data_science": ("read_target",)}


def waits_in(env: Any, recipient: Any, requester: Any = None) -> bool:
    """Whether the recipient's turn can wait on a before-seat: its own (``OWN_TURN_SEATS``) on its helper arm, or the
    requester's reading (``REQUESTER_SEATS``) on the requester's helper arm. A routing document that cannot be read: no."""
    from . import brains

    def helper(role: str, seats: tuple[str, ...]) -> bool:
        if not role or not set(seats) & set(core.ROLE_SEATS.get(role, ())):
            return False
        try:
            return brains.arm_for(env, role)[0] == "helper"
        except Exception:  # noqa: BLE001
            return False

    who, by = _base(recipient), _base(requester)
    return helper(who, OWN_TURN_SEATS.get(who, ())) or (by != "" and helper(by, REQUESTER_SEATS.get(who, ())))


# ---------------------------------------------------------------------------------------------------------- read the request
def before_rules(turn: Any) -> None:
    """Before Data Science's rules act on a task whose outcome needs a definition (they would ask the person): the
    requester's helper reads the request's words; when its reading passes the check, it is the task's outcome, saved by
    code with the person's words and who said them, marked as read by the helper (the person can correct it). Never
    raises: any failure leaves the rules to ask."""
    try:
        if _base(turn.agent) != "data_science" or not turn.task_id:
            return
        seats = of(turn)
        task = turn.work.task(turn.project_id, turn.task_id)
        if task is None or task.owner_team != "data_science" or task.target is not None or task.status not in ("open", "accepted"):
            return
        if not seats.on(task.requester.team, "read_target"):
            return
        from .agents import _mine, _needs_outcome
        from .tools import _digest, case_member, contract_for, save_outcome

        if not _needs_outcome(task):
            return
        view = turn.work.task_view(turn.project_id, task.id)
        if view is None or _mine(turn, view, "confirm_target") is not None:
            return  # the question was asked: the person's answer is read where it is recorded
        person = case_member(turn, task)[0]
        if not person:
            return  # nobody to name as having said it: the rules ask
        from .tools import _authority

        # the review of B2, F3: the save is not a tool call, so the guard's check is made here, as at every tool call: before
        # the model is asked (a member who lost access gets nothing asked in their name) and again before the write (the
        # wait may have taken seconds); then the rules' own tools refuse the rest of the turn
        if _authority(turn) is not None:
            return
        event = seats.read_target(task, task.question, where="request")
        if event is None or _authority(turn) is not None:
            return
        target = contract_for(task, event, task.question, person, f"{HELPER_READ}{_digest(task.id, task.question)}")
        save_outcome(turn.env, turn.work, turn.project_id, task, target, by=turn.actor, initiator=turn.initiator, read_by="helper")
    except Exception as exc:  # noqa: BLE001 - the rules ask, as before
        print(f"teamspace: the helper's reading of {getattr(turn, 'task_id', '?')} was not used ({type(exc).__name__})", file=sys.stderr,
              flush=True)


def read_by_helper(target: Any) -> bool:
    """Whether the outcome was read from the person's *request* by the helper: nobody confirmed it."""
    return str(getattr(target, "response", None) or "").startswith(HELPER_READ)


def reading(target: Any) -> Optional[str]:
    """Who read the event from the person's words: ``request`` (the helper, from the request: not confirmed by the person),
    ``answer`` (the helper, from the person's answer: their words, the helper's reading of them), else None (the rules)."""
    said = str(getattr(target, "response", None) or "")
    return "request" if said.startswith(HELPER_READ) else "answer" if said.startswith(HELPER_ANSWER) else None


def outcome_words(target: Any) -> tuple[str, str]:
    """How a report says the outcome and who settled it: the person's own words and "as <person> confirmed", or, where the
    helper read the event, the event's name and who read what (so a report never says a person confirmed what a model read)."""
    from dclab_core import target_contract as tc

    who, label = target.confirmed_by, tc.event_label(target.event)
    how = reading(target)
    if how == "request":
        return label, f"as the helper read {who}'s request (not confirmed by {who}, who can correct it)"
    if how == "answer":
        return label, f"as the helper read {who}'s answer"
    return target.words, f"as {who} confirmed"


# ---------------------------------------------------------------------------------------------------------- after the turn
def _report_text(report: Any) -> str:
    """What the after-seats are shown of a report: its words, limits, assumptions and next owner (never a table)."""
    lines = [f"Status: {report.status}.", f"Answer: {report.answer}"]
    lines += [f"Limit: {x}" for x in report.limits]
    lines += [f"Assumption: {a}" for e in report.evidence for a in e.assumptions]
    nxt = report.next_task
    if nxt.owner or nxt.unblocks:
        lines.append(f"Next: {nxt.owner or 'nobody'}" + (f" ({nxt.unblocks})" if nxt.unblocks else "") + ".")
    if report.simulated:
        lines.append("The results are simulated.")
    return "\n".join(lines)


def done_actions(work: Any, project_id: str, view: Mapping[str, Any]) -> list[str]:
    """The directory actions with a finished record on the report's task and on every task it handed out, and
    ``team_brain.PERSON`` when a person's answer is recorded: what a helper's words may say was done (as the agent arm's)."""
    from dclab_core import team_work as tw

    actions = list(view.get("actions") or ())
    for child in view.get("children") or ():
        actions += list(((work.task_view(project_id, child.id) or {}).get("actions")) or ())
    done = {a.action for a in actions if a.status in tw.DONE_ACTIONS}
    if any(a.source.startswith("human:") and a.responded_by is not None and a.status in tw.DONE_ACTIONS for a in actions):
        done.add(core.PERSON)
    return sorted(done)


def _claims(text: str, done: Optional[Sequence[str]]) -> bool:
    """Whether a sentence claims an action no finished record backs (B4 smoke, 2026-11-24: the after-seats checked numbers
    but never claims). ``done`` None: not checked (a caller with no records)."""
    from .. import cited

    return done is not None and any(core.unrecorded_claims(sentence, done) for sentence in cited.sentences(cited.CITE.sub("", text)))


def check_explanation(text: str, source: str, done: Optional[Sequence[str]] = None) -> tuple[Optional[str], Optional[str]]:
    """The explanation as shown (its citations taken out), or why it is dropped (a code). Every sentence must pass
    ``cited.check`` and claim no action without its finished record (``done``); one that does not drops it all."""
    from .. import cited

    said = " ".join(str(text or "").split())
    if not said:
        return None, "empty"
    out = cited.check(said, {"report": source}, limit_words=EXPLAIN_WORDS)
    if out["dropped"] or not out["kept"]:
        reason = (out["dropped"] or ["no citation"])[0]
        return None, {cited.REASONS[0]: "no_citation", cited.REASONS[1]: "unknown_source", cited.REASONS[2]: "number_not_in_report",
                      cited.REASONS[3]: "overclaims", cited.REASONS[4]: "too_long"}.get(reason, "dropped")
    shown = " ".join(k["text"] for k in out["kept"])
    if _claims(shown, done):
        return None, "claims_unrecorded_action"
    return shown, None


def check_risks(items: Any, source: str, done: Optional[Sequence[str]] = None) -> tuple[Optional[list[str]], Optional[str]]:
    """The risks as shown, or why the list is dropped (a code): a list of 1 to ``RISKS`` short sentences, each without a
    number the report does not write, an overclaim or a promise. One that fails drops the list."""
    from .. import cited

    if not isinstance(items, list) or not items:
        return None, "empty"
    if len(items) > RISKS:
        return None, "too_many"
    known = cited.numbers(source, source=True)
    out = []
    for item in items:
        said = " ".join(str(item or "").split())
        if not said or not isinstance(item, str):
            return None, "empty"
        if cited.words(said) > RISK_WORDS:
            return None, "too_long"
        if not cited.numbers(said) <= known:
            return None, "number_not_in_report"
        if cited.overclaims(said):
            return None, "overclaims"
        if core._PROMISE.search(said):
            return None, "promises_action"
        if _claims(said, done):
            return None, "claims_unrecorded_action"
        out.append(said)
    return out, None


def returned(steps: Any) -> list[tuple[str, str]]:
    """The reports a turn's run returned (task, report), new ones only (a duplicate return changed nothing)."""
    out = []
    for step in steps or ():
        result = getattr(step, "result", None) if not isinstance(step, Mapping) else step.get("result")
        tool = getattr(step, "tool", None) if not isinstance(step, Mapping) else step.get("tool")
        ok = getattr(step, "ok", None) if not isinstance(step, Mapping) else step.get("ok")
        if tool == "team_return_report" and ok and isinstance(result, Mapping) and result.get("report") and not result.get("duplicate"):
            out.append((str(result.get("task_id")), str(result["report"])))
    return out


def after(turn: Any, reports: list[tuple[str, str]], run: Optional[str]) -> list[tuple[str, str, str, Any]]:
    """The after-seats of a turn that returned reports, as (task, report, seat, job) for B1's capped background runner (the
    caller runs each job): one per report and seat the turn's role has on the helper arm. Never asks anything itself."""
    role = _base(turn.agent)
    seats = [s for s in core.AFTER_SEATS if of(turn).on(role, s)]
    if not seats or not reports:
        return []
    return [(tid, rid, seat, lambda tid=tid, rid=rid, seat=seat: _after_seat(turn, role, seat, tid, rid, run))
            for tid, rid in reports for seat in seats]


def _after_seat(turn: Any, role: str, seat: str, task_id: str, report_id: str, run: Optional[str]) -> None:
    """Ask, check, record and (when it passed) post one after-seat for one report. Never raises."""
    try:
        view = turn.work.task_view(turn.project_id, task_id)
        report = next((r for r in (view or {}).get("reports") or () if r.id == report_id), None)
        task = (view or {}).get("task")
        if report is None or task is None:
            return
        source = _report_text(report)
        client, why = _client(turn)
        request, check, shown, late = None, None, None, None
        if client is not None:
            values = {"words": EXPLAIN_WORDS} if seat == "explain_report" else {"count": RISKS}
            messages = [{"role": "system", "content": _system(seat, **values)},
                        {"role": "user", "content": f"The report [report] ({report.id}, by {report.owner}):\n{source}"}]
            reply, why, request, late = _send(client, messages, _purpose_seconds())
            found = _json(reply) if why is None else None
            if why is None and found is None:
                why = "invalid_output"
            elif why is None and seat == "explain_report":
                shown, check = check_explanation(str(found.get("explanation") or ""), source, done_actions(turn.work, turn.project_id, view))
            elif why is None:
                shown, check = check_risks(found.get("risks"), source, done_actions(turn.work, turn.project_id, view))
            if why is None and check is not None:
                why = "check_failed"
        outcome = why or "used"
        if outcome == "timeout" and late is not None:  # recorded with its request id once it ends, as a before-seat is
            late.after(lambda req: record_seat(turn, role, seat, outcome, task=task, rule="none", request=req, report=report.id, run=run))
            return
        record_seat(turn, role, seat, outcome, task=task, rule="none", check=check, request=request, report=report.id, run=run)
        if outcome == "used" and shown:
            _post(turn, task, role, seat, report.id, shown)
    except Exception as exc:  # noqa: BLE001
        print(f"teamspace: the helper's {seat} on {report_id} failed ({type(exc).__name__})", file=sys.stderr, flush=True)


def _post(turn: Any, task: Any, role: str, seat: str, report_id: str, shown: Any) -> None:
    """What passed its check, in the thread where the report is read, marked as the helper's (once per report and seat)."""
    from .threads import threads_for

    if seat == "explain_report":
        text = f"In plain words (the helper's reading, checked against the report; the report itself is unchanged): {shown}"
    else:
        text = "Risks the helper sees (advice only: they block and approve nothing):\n" + "\n".join(f"- {r}" for r in shown)
    thread = task.requester.thread_id
    if not thread:
        return
    try:
        threads_for(turn.env).post(str(thread), {"kind": "agent", "id": role, "team": role}, text,
                                   handoff={"kind": "helper_note", "seat": seat, "task_id": task.id, "item_id": report_id,
                                            "from_team": role, "to_team": _base(task.requester.team)},
                                   key=f"helper:{seat}:{report_id}")
    except (LookupError, ValueError):
        pass


# ---------------------------------------------------------------------------------------------------------- the record
def record_seat(turn: Any, role: str, seat: str, outcome: str, *, task: Any = None, rule: Any = None, answer: Any = None,
                check: Optional[str] = None, request: Optional[str] = None, seconds: Optional[float] = None,
                report: Optional[str] = None, run: Optional[str] = None) -> None:
    """One seat in the experience record (``team.helper.seat``) and the ledger (``team.helper_seat``). Codes, ids and a
    measure. Best effort: a store that cannot be written is reported and the turn stands."""
    try:
        task_id = getattr(task, "id", None) or getattr(turn, "task_id", None)
        record = core.SeatRecord(seat=seat, role=_base(role), agent=_base(turn.agent) or "team", task=task_id, outcome=outcome,
                                 used="model" if outcome == "used" else "rule", answer=_code(answer), check=_code(check),
                                 request=str(request)[:120] if request else None,
                                 seconds=round(float(seconds), 3) if seconds is not None else None)
        payload = {**record.model_dump(mode="json", exclude={"schema_version"}), "problem": getattr(task, "problem_id", None), "report": report}
        try:
            from .. import experience

            experience.record("team.helper.seat", "agent", {k: v for k, v in payload.items() if v is not None}, project=turn.project_id, run=run)
        except Exception as exc:  # noqa: BLE001
            print(f"teamspace: team.helper.seat not recorded ({type(exc).__name__})", file=sys.stderr, flush=True)
        from ..ledger import capture

        capture.team_helper_seat(turn.project_id, str(task_id or "none"), turn.agent, role=record.role, seat=seat, outcome=outcome,
                                 rule=rule, answer_code=record.answer, request=record.request, run=run)
    except Exception as exc:  # noqa: BLE001
        print(f"teamspace: the {seat} seat was not recorded ({type(exc).__name__})", file=sys.stderr, flush=True)


__all__ = ["DEFAULT_SECONDS", "ENV", "HELPER_READ", "Seats", "after", "attach", "before_rules", "check_explanation", "check_risks", "of",
           "read_by_helper", "record_seat", "returned", "seat_seconds", "waits_in"]
