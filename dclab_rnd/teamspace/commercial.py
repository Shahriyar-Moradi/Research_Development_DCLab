"""Sales and Marketing: the two commercial team agents (T3 of the team workspace, 2026-11-23).

Both are deterministic, like every team agent of this release (``agents.py``): code chooses each tool call from the task
as it is stored, through the one loop and :class:`tools.TeamGuard`. What is new is that each asks **its own team**, not
the person who asked:

- **Sales** (``team_work.DIRECTORY["sales"]``) reads the do-not-contact list with its date (``check_exclusions``), then asks
  a member of the workspace's Sales team, in the Sales thread, how many calls the team can make (``ask_capacity``, bound
  to ``human:sales_team``: 0 to 30). With the member's number it drafts the call list within that capacity
  (``draft_contact_list``) and returns a restricted reference to it (``return_restricted_ref``). The report Business reads
  carries counts and the reference only: no account, no customer row, ever leaves the Sales team. A capacity of 0 is a
  blocker ("no calls this week"), and so is a workspace with no Sales team to ask (the capacity is then not on record).
- **Marketing** asks a member of the Marketing team, in the Marketing thread, for the offer's terms (``confirm_terms``,
  bound to ``human:marketing_team``): the amount per offer, the budget, the campaign's start and end dates and who is
  eligible. It checks each (``check_eligibility``, ``check_campaign_dates``, ``check_cost``) and returns the confirmed terms
  (``return_conditions``), or a blocker that names exactly what is missing.

Either answers while the other is offline: each task is its own, and nothing waits across them. A member's answer is
recorded as theirs (``responded_by``) against the task it answers, once per response id; the member needs no approval
right (a constraint is not a decision), but must be in the team when the answer is recorded and when its report is
delivered (:func:`refused_at_delivery`). Their words stay in their own team's thread: the report carries the number or
the terms, never the message (a team's notes never reach another team's thread or a shared report).

The fixtures below are invented (``ACCOUNTS``: how many accounts a sample has that may be called). Nothing here
contacts anyone: ``contact_customers`` stays forbidden to every agent.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from typing import Any, Mapping, Optional

from dclab_core import team_work as tw

Row = dict[str, Any]
Call = tuple[str, dict[str, Any]]
TEAMS = ("sales", "marketing")
CAPACITY_MAX = 30
CAPACITY = "the Sales team confirms how many calls it can make in the week before a renewal"
TERMS = "the Marketing team confirms the offer: its amount, the budget, the campaign's dates and who is eligible"

# The invented accounts of a sample as Sales sees them: the do-not-contact list with the date it was read, and how many
# accounts renewing in the campaign window have contact consent once that list is removed. Counts only, never a row.
ACCOUNTS: dict[str, dict[str, Any]] = {
    "sample:renewal_v1": {"accounts": 1200, "do_not_contact": 3, "as_of": "2026-11-20", "eligible": 102,
                          "window": "renewing in the next 60 days, with contact consent"},
}

# What each agent asks its own team, by directory action, and what the answer gives.
ASKS: dict[str, dict[str, str]] = {
    "sales": {"action": "ask_capacity", "what": "the call capacity (0 to 30)",
              "question": "How many calls can the Sales team make in the week before renewal? Answer with a number from 0 to 30."},
    "marketing": {"action": "confirm_terms", "what": "the offer's terms: amount, budget, dates and who is eligible",
                  "question": ("Which offer may we use? Give the amount per offer (for example EUR 10), the budget, the "
                               "campaign's start and end dates (YYYY-MM-DD) and who is eligible.")},
}
SALES_STEPS = ("draft_contact_list", "return_restricted_ref")
FINAL_STEP = {"sales": "return_restricted_ref", "marketing": "return_conditions"}  # after it, the answer has been used
MARKETING_CHECKS = (("check_eligibility", "eligible"), ("check_campaign_dates", "dates"), ("check_cost", "cost"))


# ---------------------------------------------------------------------------------------------------------- reading answers
def capacity_of(text: str) -> Optional[int]:
    """The one whole number in the member's words, or None: none, or several different ones ("2 of us are off, so 18";
    "Mon 24 to Fri 28 we can do 15"), which is never guessed: the agent asks again for a single number."""
    found = {int(n) for n in re.findall(r"(?<![\d.,-])(\d{1,4})(?![\d.,]\d)", str(text or ""))}
    return found.pop() if len(found) == 1 else None


_AMOUNT = r"(\d{1,3}(?:[,.]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)"  # 1,000 / 1.000 / 1,000.50 / 10.50 / 10
_MONEY = re.compile(rf"(?:€\s?|\beur\s?){_AMOUNT}(?![\d])|{_AMOUNT}\s?(?:€|eur\b|euros?\b)", re.I)
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_ELIGIBLE = re.compile(r"\beligib\w*\s*(?::|=|are|is)?\s*([^.;\n]{3,80})"
                       r"|\bfor\s+((?:accounts|customers|clients|subscribers)\b[^.;\n]{0,72})", re.I)


def _number(value: str) -> float:
    """An amount as written: groups of three after a comma or a point are thousands (EUR 1,000 and EUR 1.000 are 1000);
    a last group of one or two digits is cents (EUR 10.50, EUR 10,50)."""
    tail = re.search(r"[.,](\d{1,2})$", value)
    whole = value[:tail.start()] if tail else value
    return float(re.sub(r"[.,]", "", whole) + ("." + tail.group(1) if tail else ""))


def terms_of(text: str) -> dict[str, Any]:
    """The offer's terms in the member's words: ``amount`` and ``budget`` (euros), ``start`` and ``end`` (ISO dates),
    ``eligible`` (their words, at most 80 characters). A piece not said is None: it is named as missing, never guessed."""
    said = str(text or "")
    amount = budget = None
    for m in _MONEY.finditer(said):
        value = _number(m.group(1) or m.group(2))
        before, after = said[max(0, m.start() - 14):m.start()].lower(), said[m.end():m.end() + 24].lower()
        if "budget" in before or re.match(r"\s*(?:of\s+)?(?:total\s+)?budget", after):
            budget = value if budget is None else budget
        elif re.match(r"\s*(?:credit|offer|discount|voucher|per\b|each|an?\s+(?:account|customer))", after) \
                or re.search(r"(?:credit|offer|discount|voucher)\s*(?:of\s*)?$", before):
            amount = value if amount is None else amount
    dates = []
    for d in _DATE.findall(said):
        try:
            dates.append(date.fromisoformat(d).isoformat())
        except ValueError:
            continue
    found = _ELIGIBLE.search(said)
    eligible = " ".join((found.group(1) or found.group(2) or "").split()).strip(" ,") if found else ""
    return {"amount": amount, "budget": budget, "start": dates[0] if dates else None, "end": dates[1] if len(dates) > 1 else None,
            "eligible": eligible[:80] or None}


def _euros(value: float) -> str:
    return f"EUR {int(value)}" if float(value).is_integer() else f"EUR {value:.2f}"


# ---------------------------------------------------------------------------------------------------------- who may answer
def team_has_members(env: Any, kind: str) -> bool:
    """Whether the workspace has a team of this kind with someone in it: someone the agent can ask."""
    from .threads import _teams_for

    try:
        return any(t.get("kind") == kind and t.get("members") for t in _teams_for(env).teams())
    except Exception:  # noqa: BLE001 — no teams to read: nobody to ask
        return False


def awaiting(view: Mapping[str, Any]) -> Optional[Row]:
    """The question an agent asked its own team on this task and that no member has answered yet, or None: what the
    task waits on (``{"team", "action", "question", "what"}``)."""
    from .tools import own_team_binding

    task = view["task"]
    base = str(task.owner_team).split(":")[0]
    config = tw.DIRECTORY.get(base)
    if config is None or not tw.is_open(task):
        return None
    mine = [a for a in view["actions"] if a.actor.id == task.owner_team]
    for a in mine:
        if not own_team_binding(config.binding(a.action) or ""):
            continue
        ask = ASKS.get(base) or {}
        found = {"team": base, "action": a.action, "question": ask.get("question") or a.action.replace("_", " "),
                 "what": ask.get("what") or a.action.replace("_", " "), "resume": False}
        if a.status == "running":
            return found
        # answered, and cut off before the step the answer feeds: the same answer sent again resumes it (``_answering``)
        if a.status in tw.DONE_ACTIONS and task.status == "accepted" \
                and not any(b.action == FINAL_STEP.get(base) and tw.can_claim(b) for b in mine):
            return {**found, "resume": True}
    return None


def _responder_of(view: Mapping[str, Any], action: str) -> Optional[tw.TeamAction]:
    task = view["task"]
    found = [a for a in view["actions"] if a.actor.id == task.owner_team and a.action == action and a.status in tw.DONE_ACTIONS
             and a.responded_by is not None]
    return found[-1] if found else None


def answer_withdrawn(env: Any, project_id: str, task_id: str) -> Optional[str]:
    """Why a Sales or Marketing task's report no longer stands, or None: it rests on a member's answer, and that member
    is no longer in the team (removed after answering). Read at delivery and when the Lead or Business joins it."""
    from .agents import _work
    from .tools import TeamTurn, holds_role

    work = _work(env)
    view = work.task_view(project_id, task_id)
    if view is None:
        return None
    base = str(view["task"].owner_team).split(":")[0]
    if base not in TEAMS:
        return None
    action = ASKS[base]["action"]
    said = _responder_of(view, action)
    if said is None:
        return None
    turn = TeamTurn(env, work, project_id, base)
    why = holds_role(turn, view["task"], tw.DIRECTORY[base].binding(action) or "", said.responded_by.id)
    return f"the answer it rests on is no longer {said.responded_by.id}'s to give: {why}" if why else None


def refused_at_delivery(env: Any, row: Mapping[str, Any]) -> Optional[str]:
    """Why a Sales or Marketing reply may not be delivered now, or None (:func:`answer_withdrawn`)."""
    base = str(row.get("from_team") or "").split(":")[0]
    if row.get("kind") != "reply" or base not in TEAMS:
        return None
    return answer_withdrawn(env, str(row["project_id"]), str(row["task_id"]))


# ---------------------------------------------------------------------------------------------------------- the tools
def _subject(subject: Optional[str]) -> dict[str, Any]:
    try:
        value = json.loads(subject or "{}")
    except ValueError:
        raise ValueError("a commercial action names what it reads (subject, as JSON)") from None
    if not isinstance(value, dict):
        raise ValueError("a commercial action's subject is a JSON object")
    return value


def _sample(task: tw.TeamTask) -> Optional[str]:
    return next((i for i in task.allowed_inputs if i.startswith("sample:")), None)


def act(ctx: Any, task: tw.TeamTask, action: str, tool: str, subject: Optional[str]) -> Row:
    """A Sales or Marketing directory action bound to a tool of this module (``tools._act`` hands it here), recorded once
    under its retry key like every action: written running, then finished with its terminal status."""
    from . import fixtures
    from .tools import _line, record, safe_id

    status, ref, revision = "succeeded", None, f"brief-{task.brief_revision}"
    sample = _sample(task)
    given = _subject(subject) if tool not in ("check_exclusions",) else {}
    response = safe_id(str(given.get("r") or "none"))
    if tool == "check_exclusions":  # Sales: the do-not-contact list, with the date it was read, removed before any plan
        found = fixtures.sample(sample) if sample else None
        dated = ACCOUNTS.get(sample or "")
        if found is None or found.get("do_not_contact") is None:
            status, ref = "failed", f"no do-not-contact list for {sample or task.id}"
        else:
            when = f" as of {dated['as_of']}" if dated else ""
            ref = f"exclusions:{sample}: {found['do_not_contact']} of {found.get('accounts', '?')} accounts on the do-not-contact list{when}"
    elif tool == "draft_list":  # Sales: the list within the confirmed capacity, never its rows
        capacity = int(given.get("capacity", -1))
        revision = f"capacity-{capacity}@{response}"
        dated = ACCOUNTS.get(sample or "")
        if capacity <= 0:
            status, ref = "failed", "capacity 0: the Sales team can make no calls this week, so no list is drafted"
        elif dated is None:
            status, ref = "failed", f"no eligible accounts are known for {sample or task.id}"
        else:
            listed = min(capacity, int(dated["eligible"]))
            ref = f"shortlist:{listed} of {dated['eligible']} eligible accounts, capped at the capacity of {capacity}"
    elif tool == "share_reference":  # Sales and Marketing: a restricted reference to what stays with the team
        revision = f"share@{response}"
        ref = f"restricted:{ctx.agent}:{safe_id(task.id)}-{response}"[:120]
    elif tool == "offer_rules":
        revision = f"eligible@{response}"
        if given.get("eligible"):
            ref = f"eligible:{str(given['eligible'])[:80]}"
        else:
            status, ref = "failed", "missing: who is eligible for the offer"
    elif tool == "campaign_calendar":
        revision = f"dates@{response}"
        start, end = given.get("start"), given.get("end")
        if not start or not end:
            status, ref = "failed", "missing: the campaign's " + " and ".join(w for w, v in (("start date", start), ("end date", end)) if not v)
        elif end < start:
            status, ref = "failed", f"the campaign's end date {end} is before its start date {start}"
        else:
            ref = f"dates:{start} to {end}"
    elif tool == "offer_cost":
        revision = f"cost@{response}"
        amount, budget = given.get("amount"), given.get("budget")
        if amount is None or budget is None:
            status, ref = "failed", "missing: " + " and ".join(w for w, v in (("the amount per offer", amount), ("the budget", budget)) if v is None)
        elif float(amount) <= 0:
            status, ref = "failed", "the amount per offer is not above zero"
        else:
            funded = math.floor(float(budget) / float(amount) + 1e-9)
            if funded < 1:
                status, ref = "failed", f"the budget {_euros(float(budget))} funds no offer of {_euros(float(amount))}"
            else:
                ref = f"cost:{_euros(float(amount))} an offer; a budget of {_euros(float(budget))} funds {funded} offers"
    else:
        raise ValueError(f"{tool} is not a tool of the commercial agents")
    if status == "succeeded" and tool in ("check_exclusions", "draft_list", "share_reference") and ctx.agent == "sales" \
            and sample in fixtures.SAMPLES:
        status = "simulated"  # a list of an invented sample is labelled as such, so the report says it is simulated
    ended, duplicate = record(ctx, task, action, input_revision=revision[:120], status=status,
                              output_ref=ref if status in tw.DONE_ACTIONS else None,
                              failure_ref=ref if status in tw.FAILED_ACTIONS else None)
    return {"task_id": task.id, "action": _line(ended), "duplicate": duplicate}


# ---------------------------------------------------------------------------------------------------------- the brains
def _mine_any(view: Mapping[str, Any], agent: str, action: str) -> Optional[tw.TeamAction]:
    found = [a for a in view["actions"] if a.actor.id == agent and a.action == action]
    return found[-1] if found else None


def _answering(turn: Any, asked: tw.TeamAction) -> Optional[str]:
    """The response this turn carries for the question asked, or None: an answer turn, and the question not yet answered
    by another response (the same response again is the same answer, so a retried answer resumes where it stopped)."""
    if turn.extra.get("mode") != "answer" or not turn.person:
        return None
    response = str(turn.extra.get("response_id") or "")
    if tw.can_claim(asked) and asked.output_ref != f"response:{response}":
        return None
    return response


def _ask_again(turn: Any, task: tw.TeamTask, team: str) -> Optional[Call]:
    """The question once more, once per turn: a crash between writing it (running) and posting it in the team's thread
    left it unsaid; ``tools._clarify`` takes the same running attempt again and its post is keyed, so nothing doubles."""
    if turn.extra.get("asked_again"):
        return None
    turn.extra["asked_again"] = True
    return ("team_clarify", {"task_id": task.id, "action": ASKS[team]["action"], "question": ASKS[team]["question"]})


def _record_answer(turn: Any, task: tw.TeamTask, action: str, response: str) -> Call:
    return ("team_record_human_response", {"task_id": task.id, "action": action, "response_id": response, "responded_by": turn.person})


def _no_team_blocker(task: tw.TeamTask, team: str, sample: Optional[str], exclusions: Optional[tw.TeamAction] = None) -> Call:
    """No team of this kind to ask: the constraint is not on record, and the blocker names the team that supplies it."""
    if team == "sales":
        from . import fixtures

        found = fixtures.sample(sample or "") or {}
        answer = (f"The calls can be planned from Data Science's ranking: the {found.get('do_not_contact', 'listed')} accounts on the "
                  "do-not-contact list are removed first, and nobody is contacted before the owner approves the list. The list "
                  "waits for the call capacity, which is not on record yet.")
        evidence = [{"source": sample or "sales", "ref": exclusions.output_ref or f"exclusions:{sample}", "input_revision": exclusions.input_revision,
                     "uncertainty": "the fixture's do-not-contact count; no list was drafted"}] \
            if exclusions is not None and exclusions.status in tw.DONE_ACTIONS else []
        return ("team_return_report", {"task_id": task.id, "outcome": "blocked", "revision": 1, "answer": answer, "next_owner": "sales",
                                       "unblocks": CAPACITY, "evidence": evidence,
                                       "limits": ["No contact list was drafted: it needs Data Science's ranking and the Sales team's capacity."]})
    return ("team_return_report", {"task_id": task.id, "outcome": "blocked", "revision": 1, "next_owner": "marketing", "unblocks": TERMS,
                                   "answer": "The offer's terms are not on record: the workspace has no Marketing team to confirm them, "
                                             "so no offer can be proposed yet."})


def sales(turn: Any) -> Optional[Call]:
    """Sales's rules (see the module's docstring)."""
    from . import agents as A

    view = A._view(turn)
    if view is None:
        return None
    task, status = view["task"], view["task"].status
    if task.owner_team != turn.agent:
        return None
    if status == "open":
        return ("team_accept", {"task_id": task.id})
    ask = ASKS["sales"]["action"]
    exclusions = A._mine(turn, view, "check_exclusions", A._brief(task))
    asked = _mine_any(view, turn.agent, ask)
    draft = _mine_any(view, turn.agent, "draft_contact_list")
    shared = _mine_any(view, turn.agent, "return_restricted_ref")
    if status == "accepted":
        if not A._claimed(exclusions):
            return ("team_act", {"task_id": task.id, "action": "check_exclusions"})
        if exclusions.status in tw.FAILED_ACTIONS:
            return ("team_block", {"task_id": task.id, "reason": exclusions.failure_ref or "check_exclusions failed"})
        if asked is None:
            if not team_has_members(turn.env, "sales"):
                return ("team_block", {"task_id": task.id, "reason": "the call capacity is not on record: " + CAPACITY})
            return ("team_clarify", {"task_id": task.id, "action": ask, "question": ASKS["sales"]["question"]})
        response = _answering(turn, asked)
        if not tw.can_claim(asked):  # asked, and no member has answered yet
            return _record_answer(turn, task, ask, response) if response else _ask_again(turn, task, "sales")
        if not A._claimed(draft):
            if response is None:
                return None  # answered, cut off before the list: the same answer again resumes here
            capacity = capacity_of(turn.text or "")
            return ("team_act", {"task_id": task.id, "action": "draft_contact_list",
                                 "subject": json.dumps({"r": response, "capacity": capacity if capacity is not None else -1})})
        if draft.status in tw.FAILED_ACTIONS:
            return ("team_block", {"task_id": task.id, "reason": draft.failure_ref or "the list could not be drafted"})
        if not A._claimed(shared):
            return ("team_act", {"task_id": task.id, "action": "return_restricted_ref",
                                 "subject": json.dumps({"r": draft.input_revision.split("@", 1)[-1]})})
        return None if A._current_report(view) else _sales_report(view, exclusions, asked, draft, shared)
    if status == "blocked" and A._current_report(view) is None:
        if exclusions is not None and exclusions.status in tw.FAILED_ACTIONS:
            why = exclusions.failure_ref or "check_exclusions could not run"
            return ("team_return_report", {"task_id": task.id, "outcome": "blocked", "revision": 1, "answer": f"Blocked: {why}.",
                                           "next_owner": task.requester.team, "unblocks": "give the check exclusions what it reads"})
        if draft is not None and draft.status in tw.FAILED_ACTIONS:
            who = asked.responded_by.id if asked is not None and asked.responded_by else "the Sales team"
            return ("team_return_report", {"task_id": task.id, "outcome": "blocked", "revision": 1, "next_owner": "sales",
                                           "unblocks": "a call capacity above zero from the Sales team",
                                           "answer": f"Blocked: {draft.failure_ref} (confirmed by {who}). Nobody is contacted."})
        return _no_team_blocker(task, "sales", _sample(task), exclusions)
    if status in ("complete", "partial") and A._current_report(view) is None and A._claimed(shared) and A._claimed(draft):
        return _sales_report(view, exclusions, asked, draft, shared)  # moved without its report (an older crash): report again
    return None


def _sales_report(view: Mapping[str, Any], exclusions: Optional[tw.TeamAction], asked: tw.TeamAction, draft: tw.TeamAction,
                  shared: tw.TeamAction) -> Call:
    task = view["task"]
    sample = _sample(task) or ""
    dated = ACCOUNTS.get(sample) or {}
    capacity = int(draft.input_revision.split("@", 1)[0].split("-", 1)[1])
    listed = min(capacity, int(dated.get("eligible") or 0))
    who = asked.responded_by.id if asked.responded_by else "the Sales team"
    removed = (f", after removing the {dated.get('do_not_contact')} accounts on the do-not-contact list of {dated.get('as_of')}"
               if dated else "")
    answer = (f"Sales can make {capacity} calls in the week before renewal (confirmed by {who}). A call list of {listed} of the "
              f"{dated.get('eligible', '?')} eligible accounts is drafted, earliest renewal first{removed}. The list stays with the Sales "
              f"team (restricted reference {shared.output_ref}); nobody is contacted before the owner approves it.")
    evidence = [{"source": sample or "sales", "ref": shared.output_ref or "restricted", "input_revision": draft.input_revision,
                 "uncertainty": "invented eligibility counts; not yet ordered by Data Science's ranking",
                 "assumptions": [f"a call capacity of {capacity}, confirmed by {who}"]}]
    limits = ["Counts and a restricted reference only: no account or customer row leaves the Sales team.",
              "Data Science's risk ranking can reorder the list; its length stays capped at the capacity."]
    return ("team_return_report", {"task_id": task.id, "outcome": "complete", "revision": 1, "answer": answer, "evidence": evidence,
                                   "limits": limits})


def marketing(turn: Any) -> Optional[Call]:
    """Marketing's rules (see the module's docstring)."""
    from . import agents as A

    view = A._view(turn)
    if view is None:
        return None
    task, status = view["task"], view["task"].status
    if task.owner_team != turn.agent:
        return None
    if status == "open":
        return ("team_accept", {"task_id": task.id})
    ask = ASKS["marketing"]["action"]
    asked = _mine_any(view, turn.agent, ask)
    checks = {action: _mine_any(view, turn.agent, action) for action, _ in MARKETING_CHECKS}
    shared = _mine_any(view, turn.agent, "return_conditions")
    failed = [checks[a] for a, _ in MARKETING_CHECKS if checks[a] is not None and checks[a].status in tw.FAILED_ACTIONS]
    if status == "accepted":
        if asked is None:
            if not team_has_members(turn.env, "marketing"):
                return ("team_block", {"task_id": task.id, "reason": "the offer's terms are not on record: " + TERMS})
            return ("team_clarify", {"task_id": task.id, "action": ask, "question": ASKS["marketing"]["question"]})
        response = _answering(turn, asked)
        if not tw.can_claim(asked):
            return _record_answer(turn, task, ask, response) if response else _ask_again(turn, task, "marketing")
        terms = terms_of(turn.text or "") if response else None
        for action, key in MARKETING_CHECKS:
            if not A._claimed(checks[action]):
                if terms is None:
                    return None  # answered, cut off before the checks: the same answer again resumes here
                given = {"r": response, "eligible": terms["eligible"]} if key == "eligible" else \
                    {"r": response, "start": terms["start"], "end": terms["end"]} if key == "dates" else \
                    {"r": response, "amount": terms["amount"], "budget": terms["budget"]}
                return ("team_act", {"task_id": task.id, "action": action, "subject": json.dumps(given)})
        if failed:
            return ("team_block", {"task_id": task.id, "reason": "; ".join(c.failure_ref or c.action for c in failed)})
        if not A._claimed(shared):
            response = checks["check_cost"].input_revision.split("@", 1)[-1]
            return ("team_act", {"task_id": task.id, "action": "return_conditions", "subject": json.dumps({"r": response})})
        return None if A._current_report(view) else _marketing_report(view, asked, checks, shared)
    if status == "blocked" and A._current_report(view) is None:
        if asked is None or not failed:
            return _no_team_blocker(task, "marketing", None)
        who = asked.responded_by.id if asked.responded_by else "the Marketing team"
        missing = "; ".join(c.failure_ref or c.action for c in failed)
        return ("team_return_report", {"task_id": task.id, "outcome": "blocked", "revision": 1, "next_owner": "marketing",
                                       "unblocks": f"the Marketing team gives what is missing ({missing})"[:300],
                                       "answer": f"Blocked: the offer's terms from {who} are not complete: {missing}. No offer is proposed."})
    if status in ("complete", "partial") and A._current_report(view) is None and A._claimed(shared):
        return _marketing_report(view, asked, checks, shared)
    return None


def _after(ref: Optional[str]) -> str:
    return str(ref or "").split(":", 1)[-1]


def _marketing_report(view: Mapping[str, Any], asked: tw.TeamAction, checks: Mapping[str, tw.TeamAction], shared: tw.TeamAction) -> Call:
    task = view["task"]
    who = asked.responded_by.id if asked.responded_by else "the Marketing team"
    eligible, dates, cost = (_after(checks[a].output_ref) for a, _ in MARKETING_CHECKS)
    answer = (f"Marketing confirms the offer (confirmed by {who}): {cost}. Eligible: {eligible}. The campaign runs from {dates}. "
              "No offer is sent and nobody is contacted before the owner approves.")
    # no evidence row: the terms are no input the task reads (its allowed inputs are tables); the checked actions, each with
    # its output and the member who answered, are the record, and the restricted reference names where the terms stay
    return ("team_return_report", {"task_id": task.id, "outcome": "complete", "revision": 1, "answer": answer,
                                   "limits": [f"The terms are the Marketing team's answer ({who}; kept as {shared.output_ref}): they were "
                                              "checked for completeness and arithmetic, not against a campaign system. No offer was tested."]})


BRAINS = {"sales": sales, "marketing": marketing}


# ---------------------------------------------------------------------------------------------------------- answers in a team's thread
def open_asks(env: Any, thread: Mapping[str, Any], team_kind: str) -> list[tw.TeamTask]:
    """The tasks on this thread's problem that its team's agent asked its own team about and that wait on a member's
    answer, oldest first."""
    from .agents import _work

    work = _work(env)
    project_id, problem_id = str(thread["project_id"]), str(thread["problem_id"])
    out, seen = [], set()
    for row in work.deliveries(project_id=project_id):
        if row["kind"] != "task" or row["problem_id"] != problem_id or row["task_id"] in seen:
            continue
        seen.add(row["task_id"])
        view = work.task_view(project_id, row["task_id"])
        if view is None or str(view["task"].owner_team).split(":")[0] != team_kind:
            continue
        if awaiting(view) is not None:
            out.append(view["task"])
    return out


def hint(env: Any, thread: Mapping[str, Any], text: str, *, key: str, handoff: Optional[Row] = None) -> Optional[Row]:
    """The thread agent's own message in its thread (once per ``key``): what it needs to take a member's answer."""
    from .threads import threads_for

    team = str(thread["team"])
    agent = str(thread.get("team_kind") or team)
    message, _ = threads_for(env).post(str(thread["id"]), {"kind": "agent", "id": agent, "team": agent}, text,
                                       handoff=handoff, key=key)
    return message


def answer(env: Any, thread: Mapping[str, Any], task: tw.TeamTask, *, person: str, text: str, initiator: Optional[Row] = None,
           response_id: str, registry: Any = None) -> dict[str, Any]:
    """A member of the team answered its agent's question on ``task`` in their thread: the agent records it as theirs and
    goes on. Words that give nothing it can use (no number for Sales) are not an answer: it says what it needs, and the
    question stays open."""
    from . import agents as A
    from .tools import TeamTurn

    base = str(task.owner_team).split(":")[0]
    if base == "sales":
        capacity = capacity_of(text)
        if capacity is None or not 0 <= capacity <= CAPACITY_MAX:
            hint(env, {**thread, "team_kind": base}, f"I need a number from 0 to {CAPACITY_MAX}: {ASKS['sales']['question']}",
                 key=f"hint:{task.id}:{response_id}", handoff={"kind": "hint", "task_id": task.id, "from_team": base, "to_team": base})
            return {"agent": base, "task_id": task.id, "did": "hint", "skipped": True, "steps": [], "refused": []}
    elif not any(v is not None for v in terms_of(text).values()):
        hint(env, {**thread, "team_kind": base}, f"I could not read any term in that. {ASKS['marketing']['question']}",
             key=f"hint:{task.id}:{response_id}", handoff={"kind": "hint", "task_id": task.id, "from_team": base, "to_team": base})
        return {"agent": base, "task_id": task.id, "did": "hint", "skipped": True, "steps": [], "refused": []}
    turn = TeamTurn(env, A._work(env), str(task.project_id), base, task_id=task.id, initiator=initiator, person=person, text=text,
                    extra={"mode": "answer", "response_id": response_id})
    return {**A.run_turn(turn, registry), "did": "answered"}


__all__ = ["ACCOUNTS", "ASKS", "BRAINS", "CAPACITY", "CAPACITY_MAX", "TEAMS", "TERMS", "act", "answer", "answer_withdrawn", "awaiting", "capacity_of",
           "hint", "marketing", "open_asks", "refused_at_delivery", "sales", "team_has_members", "terms_of"]
