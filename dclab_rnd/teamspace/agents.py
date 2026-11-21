"""The team agents: who each one is, what it may do, and the deterministic agents of the first release (W5, 2026-11-20).

**What every agent knows** (:class:`TeamAgentConfig`): its identity, its responsibility, its action directory (each action
bound to a tool of the ``team`` scope or to a person's recorded answer), the report it owes, what it may read and what it
may never do. Every configuration is built from ``dclab_core.team_work.DIRECTORY`` (one source of truth: no second list
here); this module only adds the runtime: the registered tool each action binds to, and the task moves the agent makes.
:func:`job_description` is the same configuration in words, for people (the W8 screens show it).

**The agents of this release are deterministic**: no language model. Each runs on the one agent loop
(``agents.runtime.run``, traced like every agent: ``harness.AGENTS["teamspace"]``) with :class:`ScriptedChoices`, a
client that asks the agent's rules for its next tool call. The rules read the task as it is stored now, so a turn does
what is left and nothing more: a delivery handed over twice (a crash after the agent acted, before the row was marked
delivered) finds nothing left and runs no step. AI agents come later (R4), behind a switch, on the same tools and guard.

- **Business** opens the case's task for Data Science with the brief (``open_case``); relays a clarification to its
  person and carries the person's answer back (``person_replies``); receives the report and presents it.
- **Data Science** accepts; asks one question when the target is ambiguous (churn: cancellation, or failure to renew);
  then returns a feasibility report built by ``team_work.build_report`` from its action log, or a precise blocker naming
  the missing field and the team that can supply it. For the food delivery question it opens two child tasks (an ETA and
  a HyperAck acceptance specialist), joins both, and returns one report.
- **Model specialist** (``model:eta``, ``model:acceptance``): a fixture, every result ``simulated`` (fixtures E1 and A1).
  A correction (``correct``: the kitchen's added minutes) is a new revision that supersedes the ETA report; the acceptance
  report keeps its dispatch snapshot.

**Delivery**: the outbox's default target (``delivery.default_target``) posts each row into its team thread (W3's
``threads.deliver``, once per row id), then hands it to :func:`deliver`, the recipient agent's turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from dclab_core import team_work as tw

from . import fixtures
from .tools import ACTS, MODEL_TOOLS, TeamTurn, case_member, is_error, safe_id

Row = dict[str, Any]
Call = tuple[str, dict[str, Any]]
AGENT = "teamspace"  # harness.AGENTS: the team agents' runs (policy names teamspace.<agent>)
MAX_STEPS = 40
CLARIFY_TARGET = "Does churn mean cancellation, or failure to renew by the contractual renewal date?"
SYSTEM = ("A deterministic team agent: code chooses each tool call from the task's stored state. No language model is "
          "asked; nothing here is a prompt.")


# ---------------------------------------------------------------------------------------------------------- configurations
def _tool_for(binding: str) -> Optional[str]:
    if binding == "tool:delegate":
        return "team_delegate"
    if binding.startswith("human:"):
        return "team_record_human_response"
    name = binding[5:]
    if name in MODEL_TOOLS:
        return "team_model_call"
    if name in ACTS:
        return "team_act"
    return None  # bound to a tool a later release provides (R2: Sales and Marketing; R3: the studio engine)


@dataclass(frozen=True)
class TeamAgentConfig:
    """One agent's configuration: the directory row (``team_work.AgentConfig``) and the runtime it needs."""

    config: tw.AgentConfig
    tools: Mapping[str, Optional[str]]  # directory action -> the registered tool that performs it (None: not in this release)
    moves: tuple[str, ...]  # the task moves it makes, as registered tools

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def memory_scope(self) -> tuple[str, ...]:
        return tuple(self.config.memory_scope)

    @property
    def forbidden(self) -> tuple[str, ...]:
        return tuple(self.config.forbidden)

    @property
    def policy_tools(self) -> tuple[str, ...]:
        """Every registered tool the agent's loop is offered."""
        return tuple(sorted(set(self.moves) | {t for t in self.tools.values() if t}))


def _make(config: tw.AgentConfig) -> TeamAgentConfig:
    tools = {a.action: _tool_for(a.bound_to) for a in config.action_directory}
    moves = ["team_accept", "team_block", "team_return_report"]
    if any(a.bound_to.startswith("human:") for a in config.action_directory):
        moves += ["team_clarify", "team_answer"]
    return TeamAgentConfig(config=config, tools=tools, moves=tuple(moves))


CONFIGS: dict[str, TeamAgentConfig] = {name: _make(c) for name, c in tw.DIRECTORY.items()}


def config_of(agent: str) -> TeamAgentConfig:
    """The configuration of an agent by its name, a qualified specialist (``model:eta``) included."""
    found = CONFIGS.get(str(agent).split(":")[0])
    if found is None:
        raise KeyError(f"no team agent {agent!r}: the directory has {', '.join(CONFIGS)}")
    return found


_NEVER_WORDS = {"approve": "approve a decision on its owner's behalf (only the decision owner, a person, approves)",
                "train": "train a model (ML Engineering runs an agreed experiment)", "deploy": "deploy a model",
                "contact_customers": "contact customers"}
_REPORT_WORDS = {"feasibility_or_finding": "the feasibility answer or the evaluated finding", "checks_performed": "the checks performed",
                 "next_owner": "who acts next", "question_answered": "the question answered", "contributing_teams": "the teams that contributed",
                 "unresolved_constraints": "unresolved constraints", "remaining_inputs": "the inputs still missing"}
_SCOPE_WORDS = {"person": "the person's own conversation", "team": "its team's records", "workspace": "the workspace",
                "project": "the project", "problem": "the problem", "model_version": "one model version"}


def _lower(text: str) -> str:
    return text[:1].lower() + text[1:]


def _words(name: str) -> str:
    return _REPORT_WORDS.get(name, name.replace("_", " "))


def job_description(name: str) -> dict[str, Any]:
    """The agent's job description for people: who it represents, what it owns, the actions it may take (and through what),
    the report it owes, what it may read and what it may never do. ``text`` is the same in a few sentences."""
    agent = config_of(name)
    c = agent.config
    actions = []
    for a in c.action_directory:
        how = (f"a person's recorded answer ({a.bound_to[6:].replace('_', ' ')})" if a.bound_to.startswith("human:")
               else f"the tool {a.bound_to[5:]}")
        actions.append({"action": a.action, "what": a.description, "bound_to": a.bound_to, "by": how,
                        "tool": agent.tools.get(a.action), "available": agent.tools.get(a.action) is not None})
    never = [_NEVER_WORDS.get(f, f.replace("_", " ")) for f in c.forbidden]
    never += ["start an action outside this directory", "claim an action done without its terminal record"]
    report = [_words(r) for r in c.required_report]
    text = (f"{c.identity}. It is responsible to {_lower(c.responsibility)}. "
            f"It may: {'; '.join(_lower(a.description.rstrip('.')) for a in c.action_directory)}. "
            f"Its report gives {', '.join(report)}. It reads {', '.join(_SCOPE_WORDS.get(s, s) for s in c.memory_scope)}. "
            f"It may never {'; '.join(never)}.")
    return {"name": c.name, "kind": c.kind, "identity": c.identity, "responsibility": c.responsibility, "actions": actions,
            "required_report": list(c.required_report), "report_in_words": report, "reads": list(c.memory_scope),
            "may_never": never, "forbidden": list(c.forbidden), "text": text}


# ---------------------------------------------------------------------------------------------------------- what each agent reads
def _view(turn: TeamTurn, task_id: Optional[str] = None) -> Optional[Row]:
    return turn.work.task_view(turn.project_id, task_id or turn.task_id)


def _mine(turn: TeamTurn, view: Row, action: str, revision: Optional[str] = None) -> Optional[tw.TeamAction]:
    found = [a for a in view["actions"] if a.actor.id == turn.agent and a.action == action
             and (revision is None or a.input_revision == revision)]
    return found[-1] if found else None


def _current_report(view: Row) -> Optional[tw.WorkReport]:
    current = [r for r in view["reports"] if r.status != "superseded"]
    return max(current, key=lambda r: r.revision) if current else None


def _brief(task: tw.TeamTask) -> str:
    return f"brief-{task.brief_revision}"


def _ambiguous_target(question: str) -> bool:
    """Whether the outcome needs a definition: churn (or customers leaving) without saying whether it means a cancellation
    or a failure to renew."""
    said = question.lower()
    vague = re.search(r"\b(churn\w*|leav\w*|left|lose|losing|lost)\b", said)
    defined = re.search(r"fail\w* to renew|non-?renewal|not renew\w*|churn (means|is) |cancell?ation (means|is) ", said)
    return bool(vague) and not defined


def _specialists(task: tw.TeamTask) -> list[str]:
    """The model specialists a dispatch question needs: one for the arrival, one for first-offer acceptance."""
    if not any(i.startswith("dispatch:") for i in task.allowed_inputs):
        return []
    said = task.question.lower()
    kinds = []
    if re.search(r"\b(arriv\w*|eta)\b", said):
        kinds.append("eta")
    if re.search(r"\baccept\w*", said):
        kinds.append("acceptance")
    return kinds


_CHILD_QUESTIONS = {
    "eta": ("Estimate the arrival time of the order at its first dispatch.", "an arrival time from one model version, or a refusal"),
    "acceptance": ("Estimate the probability that a courier accepts the first offer for the order.",
                   "a first-offer acceptance probability from one model version, or a refusal"),
}


# ---------------------------------------------------------------------------------------------------------- Business and Home
def _requester(open_action: str, confirm_action: str, present_action: str) -> Callable[[TeamTurn], Optional[Call]]:
    """The rules of an agent that asks for its person: open the case's task, relay a clarification to its person and carry
    the person's answer back, present the report it receives. Business (in its team thread) and Home (the person's Home
    conversation, which asks the Lead) are the same rules under their own directory actions."""

    def brain(turn: TeamTurn) -> Optional[Call]:
        mode = turn.extra.get("mode")
        if mode == "open":
            if turn.work.task(turn.project_id, turn.task_id) is not None:
                return None
            x = turn.extra
            return ("team_delegate", {"action": open_action, "new_task_id": turn.task_id, "owner_team": x["owner"],
                                      "question": x["question"], "expected_output": x["expected_output"],
                                      "done_criteria": list(x["done_criteria"]), "problem_id": x["problem_id"],
                                      "thread_id": x["thread_id"], "allowed_inputs": list(x.get("inputs") or [])})
        view = _view(turn)
        if view is None:
            return None
        task = view["task"]
        if mode == "answer":
            asked = _mine(turn, view, confirm_action)
            if asked is None or not tw.can_claim(asked):
                return ("team_record_human_response", {"task_id": task.id, "action": confirm_action,
                                                       "response_id": turn.extra["response_id"], "responded_by": turn.person})
            if asked.output_ref != f"response:{turn.extra['response_id']}":
                return None  # the question was already answered: a later message is not a second answer (the review of the Lead)
            key = f"answer:{task.id}:r-{safe_id(turn.extra['response_id'])}"
            if not any(d["key"] == key for d in view["deliveries"]):
                return ("team_answer", {"task_id": task.id, "response_id": turn.extra["response_id"], "text": turn.text or ""})
            return None
        row = turn.row or {}
        if row.get("kind") == "clarification":
            if _mine(turn, view, confirm_action) is None:
                return ("team_clarify", {"task_id": task.id, "action": confirm_action, "question": row.get("text") or ""})
            return None
        if row.get("kind") == "reply":
            report = row.get("item_id")
            if _mine(turn, view, present_action, report) is None and any(r.id == report for r in view["reports"]):
                return ("team_act", {"task_id": task.id, "action": present_action, "subject": report})
        return None

    return brain


_business = _requester("delegate_analysis", "confirm_terms", "present_options")
_home = _requester("open_case", "confirm_with_person", "present_report")


# ---------------------------------------------------------------------------------------------------------- Data Science
def _confirmed_by(turn: TeamTurn, view: Row) -> Optional[tw.TeamAction]:
    """The person's recorded answer on this task, by the agent that asked them (the requester's human-bound action)."""
    task = view["task"]
    found = [a for a in view["actions"] if a.actor.id == task.requester.team and a.source.startswith("human:")
             and a.status in tw.DONE_ACTIONS and a.responded_by is not None]
    return found[-1] if found else None


def _answer_text(view: Row) -> Optional[str]:
    answers = [d for d in view["deliveries"] if d["kind"] == "answer"]
    return answers[-1]["text"] if answers else None


def _data_science(turn: TeamTurn) -> Optional[Call]:
    view = _view(turn)
    if view is None:
        return None
    task, status = view["task"], view["task"].status
    if status == "open":
        return ("team_accept", {"task_id": task.id})
    asked = _mine(turn, view, "confirm_target")
    if status == "waiting_for_person":
        said = _confirmed_by(turn, view)
        if asked is not None and not tw.can_claim(asked) and said is not None and _answer_text(view):
            response = (said.output_ref or "response:x").split(":", 1)[1]
            return ("team_record_human_response", {"task_id": task.id, "action": "confirm_target", "response_id": response,
                                                   "responded_by": said.responded_by.id})
        if asked is not None and tw.can_claim(asked):
            return ("team_accept", {"task_id": task.id})
        return None
    kinds = _specialists(task)
    if kinds and status in ("accepted", "waiting_for_team"):  # the first delegation moves the task to waiting_for_team
        for kind in kinds:
            child = f"{task.id}.{kind}"
            # missing, or its delegation not recorded, or the parent not yet waiting: a crash between two writes resumes here
            if (turn.work.task(turn.project_id, child) is None or _mine(turn, view, "delegate_specialist", child) is None
                    or status == "accepted"):
                question, output = _CHILD_QUESTIONS[kind]
                return ("team_delegate", {"action": "delegate_specialist", "task_id": task.id, "new_task_id": child,
                                          "owner_team": f"model:{kind}", "question": question, "expected_output": output,
                                          "done_criteria": ["names the model version and the dispatch snapshot it used"],
                                          "allowed_inputs": [i for i in task.allowed_inputs if i.startswith("dispatch:")]})
    if status == "accepted":
        if kinds:
            return None
        if _ambiguous_target(task.question) and asked is None:
            return ("team_clarify", {"task_id": task.id, "action": "confirm_target", "question": CLARIFY_TARGET})
        return _feasibility(turn, view)
    if status == "blocked":
        return None if _current_report(view) else _blocker_report(turn, view)
    if status in ("complete", "partial") and not view["children"] and _current_report(view) is None:
        return _feasibility(turn, view)  # completed without its report (an older crash): report again
    if status in ("waiting_for_team", "partial", "complete") and view["children"]:
        return _join(turn, view)
    return None


def _sample(task: tw.TeamTask) -> Optional[str]:
    return next((i for i in task.allowed_inputs if i.startswith("sample:")), None)


def _feasibility(turn: TeamTurn, view: Row) -> Optional[Call]:
    """Inspect the sample; block on what is missing, or check leakage, name the baseline and the uncertainty, and report."""
    task = view["task"]
    if _mine(turn, view, "inspect_schema", _brief(task)) is None:
        return ("team_act", {"task_id": task.id, "action": "inspect_schema"})
    sample = _sample(task) or ""
    check = fixtures.feasibility(sample)
    if not check["ok"]:
        return ("team_block", {"task_id": task.id, "reason": check["why"]})
    for action in ("check_leakage", "define_baseline", "explain_uncertainty"):
        if _mine(turn, view, action, _brief(task)) is None:
            return ("team_act", {"task_id": task.id, "action": action})
    if _current_report(view) is not None:
        return None
    said = _answer_text(view)
    confirmed = _confirmed_by(turn, view)
    assumptions = [f"The outcome is: {said}" + (f" (confirmed by {confirmed.responded_by.id})." if confirmed else ".")] if said else []
    evidence = [{"source": sample, "ref": f"schema:{sample}", "input_revision": _brief(task), "assumptions": assumptions,
                 "uncertainty": "the sample's columns and dates only; no account was scored"}]
    limits = ["Checked on the sample's columns and dates only: no model was trained and no account was scored.",
              f"Excluded as inputs: {', '.join(check['excluded'])}, known only after renewal (it is the outcome)."]
    return ("team_return_report", {"task_id": task.id, "outcome": "complete", "revision": 1, "evidence": evidence, "limits": limits,
                                   "answer": "The sample has renewal dates and mature outcomes. We can score seven days before renewal."})


def _blocker_report(turn: TeamTurn, view: Row) -> Call:
    task = view["task"]
    sample = _sample(task) or "the sample"
    check = fixtures.feasibility(sample)
    from .store import team_label

    supplier = check.get("supplier") or task.requester.team
    answer = (f"Blocked: {check.get('why', 'a needed input is missing')} (missing field: {check.get('missing')}). "
              f"The {team_label(supplier)} team can supply it; work resumes when it is attached to {sample}.")
    return ("team_return_report", {"task_id": task.id, "outcome": "blocked", "revision": 1, "answer": answer,
                                   "next_owner": supplier, "unblocks": f"attach {check.get('missing')} to {sample}"})


def _join(turn: TeamTurn, view: Row, *, attribute: bool = False) -> Optional[Call]:
    """Review each child's reply as it returns; when every child has returned, report once with each result joined once
    (a blocked or failed child keeps the report partial and names who acts next); a corrected child reply supersedes it."""
    task, join = view["task"], view["join"]
    for report_id in join.reports:
        if _mine(turn, view, "review_evidence", report_id) is None:
            return ("team_act", {"task_id": task.id, "action": "review_evidence", "subject": report_id})
    if join.outstanding:
        return None  # still out: preview_report says which
    current = _current_report(view)
    if current is not None and sorted(current.child_reports) == sorted(join.reports):
        return None
    joined = {r.id: r for r in view["child_reports"]}
    order = [c.id for c in view["children"]]
    returned = sorted((joined[rid] for rid in join.reports), key=lambda r: order.index(r.task_id))
    if attribute:  # the Lead says which team answered what
        from .store import team_label

        answer = " ".join(f"{team_label(r.owner)}: {r.answer}" for r in returned)
    else:
        answer = " ".join(r.answer for r in returned)
    evidence = [e.model_dump() for r in returned for e in r.evidence]
    limits = []
    if any(r.simulated for r in returned):
        fixtures_used = sorted({e.ref for r in returned for e in r.evidence})
        limits.append(f"Simulated: the figures come from fixtures ({', '.join(fixtures_used) or 'none'}); they illustrate the "
                      "handoff and are not measured.")
    unresolved = [joined[rid] for rid in join.reports if joined[rid].task_id in join.unresolved]
    call: dict[str, Any] = {"task_id": task.id, "answer": answer, "evidence": evidence, "limits": limits,
                            "revision": 1 if current is None else current.revision + 1,
                            "outcome": "partial" if unresolved else "complete"}
    if unresolved:
        first = unresolved[0]
        call.update(next_owner=first.next_task.owner or first.owner, unblocks=first.next_task.unblocks or "its blocker")
    return ("team_return_report", call)


# ---------------------------------------------------------------------------------------------------------- model specialist
def _model(turn: TeamTurn) -> Optional[Call]:
    view = _view(turn)
    if view is None:
        return None
    task, status = view["task"], view["task"].status
    if status == "open":
        return ("team_accept", {"task_id": task.id})
    kind = turn.agent.split(":", 1)[1] if ":" in turn.agent else ""
    snapshot = next((i for i in task.allowed_inputs if i.startswith("dispatch:")), "")
    first = _mine(turn, view, "score", snapshot or "none")
    if status == "accepted":
        if first is None:
            return ("team_model_call", {"task_id": task.id, "action": "score"})
        if first.status in tw.FAILED_ACTIONS:
            return ("team_block", {"task_id": task.id, "reason": first.failure_ref or "the score was refused"})
        return _estimate_report(task, kind, snapshot, first, None, 1)
    if status == "blocked" and _current_report(view) is None:
        return ("team_return_report", {
            "task_id": task.id, "outcome": "blocked", "revision": 1, "next_owner": "ml_engineering",
            "unblocks": f"register a permitted {kind} model version for {snapshot or 'this task'}",
            "answer": f"No {_LABEL.get(kind, kind)} for this order: the score was denied ({first.failure_ref if first else 'no version'}). "
                      "No estimate is given."})
    if status == "complete" and _current_report(view) is None and first is not None and first.status in tw.DONE_ACTIONS:
        return _estimate_report(task, kind, snapshot, first, None, 1)  # completed without its report: report again
    minutes = turn.extra.get("correction_minutes")
    if status == "complete" and minutes:
        revision = f"{snapshot};kitchen+{int(minutes)}m"
        again = _mine(turn, view, "score", revision)
        if again is None:
            return ("team_model_call", {"task_id": task.id, "action": "score", "correction_minutes": int(minutes)})
        current = _current_report(view)
        if current is not None and all(e.input_revision != revision for e in current.evidence):
            return _estimate_report(task, kind, snapshot, again, (int(minutes), first), current.revision + 1)
    return None


_LABEL = {"eta": "arrival estimate", "acceptance": "first-offer acceptance estimate"}


def _value(action: tw.TeamAction) -> str:
    return (action.output_ref or "").split("=", 1)[-1]


def _estimate_report(task: tw.TeamTask, kind: str, snapshot: str, score: tw.TeamAction,
                     correction: Optional[tuple[int, Optional[tw.TeamAction]]], revision: int) -> Call:
    fixture = (score.output_ref or "").split(":", 1)[-1].split("=", 1)[0]
    at = snapshot.split("@", 1)[-1] if "@" in snapshot else "dispatch"
    if kind == "eta":
        if correction is None:
            answer = f"At {at} the estimated arrival was {_value(score)}."
        else:
            minutes, before = correction
            answer = (f"Revised: the estimated arrival is {_value(score)} (was {_value(before) if before else 'unknown'} at {at}; "
                      f"the kitchen added {minutes} minutes, fixed route).")
        assumptions = ["fixed route"]
    else:
        answer = f"The first-offer acceptance estimate was {_value(score)}."
        assumptions = [f"the courier pool at {at}"]
    evidence = [{"source": snapshot, "ref": f"fixture:{fixture}", "input_revision": score.input_revision,
                 "uncertainty": "a fixture, not a measured estimate", "assumptions": assumptions}]
    return ("team_return_report", {"task_id": task.id, "outcome": "complete", "revision": revision, "answer": answer,
                                   "evidence": evidence, "limits": ["Simulated: a fixture result that illustrates the handoff."]})


# ---------------------------------------------------------------------------------------------------------- the Lead
HELP = "help:"  # the kind of help the person picked, as the one input of a Lead task the Lead reads (never a team's input)


def help_of(task: tw.TeamTask) -> Optional[str]:
    return next((i[len(HELP):] for i in task.allowed_inputs if i.startswith(HELP)), None)


def _lead(turn: TeamTurn) -> Optional[Call]:
    """The Lead (``team_work.DIRECTORY["lead"]``, a coordinator): accept the person's request from Home, check it has what
    the teams need, hand it to the team agents its kind of help needs (``team_work.lead_route``), pass a team's question to
    the person and carry their answer back down, review each team's reply and return one joined report that says which team
    answered what. Its rules read the tasks as stored, like every agent's."""
    view = _view(turn)
    if view is None:
        return None
    task, status = view["task"], view["task"].status
    if status == "open":
        return ("team_accept", {"task_id": task.id})
    row = turn.row or {}
    if row.get("kind") == "clarification" and row.get("task_id") != task.id and status not in tw.CLOSED_TASKS \
            and _mine(turn, view, "confirm_with_requester") is None:
        return ("team_clarify", {"task_id": task.id, "action": "confirm_with_requester", "question": row.get("text") or ""})
    if status == "waiting_for_person":
        return _carry_down(turn, view)
    if status in ("accepted", "waiting_for_team"):
        pre = _mine(turn, view, "check_prerequisites", _brief(task))
        if pre is None:
            return ("team_act", {"task_id": task.id, "action": "check_prerequisites"})
        if pre.status in tw.FAILED_ACTIONS:
            return ("team_block", {"task_id": task.id, "reason": pre.failure_ref or "a prerequisite is missing"})
        inputs = [i for i in task.allowed_inputs if not i.startswith(HELP)]
        for team in tw.lead_route(help_of(task) or ""):
            child = f"{task.id}.{team}"
            # missing, or its delegation not recorded (a crash between the two writes resumes here)
            if turn.work.task(turn.project_id, child) is None or _mine(turn, view, "delegate", child) is None:
                return ("team_delegate", {"action": "delegate", "task_id": task.id, "new_task_id": child, "owner_team": team,
                                          "question": task.question, "expected_output": "a feasibility answer or a precise blocker",
                                          "done_criteria": ["says whether the data can answer the question, or what blocks it"],
                                          "allowed_inputs": inputs})
    if status == "blocked":
        if _current_report(view) is not None:
            return None
        pre = _mine(turn, view, "check_prerequisites", _brief(task))
        why = (pre.failure_ref if pre else None) or "a prerequisite is missing"
        return ("team_return_report", {"task_id": task.id, "outcome": "blocked", "revision": 1, "next_owner": task.requester.team,
                                       "unblocks": "declare what the teams may read for this project (its team inputs)",
                                       "answer": f"Blocked before any team was asked: {why}."})
    if view["children"] and status in ("accepted", "waiting_for_team", "partial", "complete"):
        return _join(turn, view, attribute=True)
    return None


def _missing_child(turn: TeamTurn, task: tw.TeamTask) -> bool:
    return any(turn.work.task(turn.project_id, f"{task.id}.{team}") is None for team in tw.lead_route(help_of(task) or ""))


def _carry_down(turn: TeamTurn, view: Row) -> Optional[Call]:
    """The person answered the Lead's question (carried up from a team): record it as theirs (as Home recorded it), give it
    to each team that asked, then take the task up again."""
    task = view["task"]
    said = _confirmed_by(turn, view)
    mine = _mine(turn, view, "confirm_with_requester")
    if said is None or mine is None:
        return None
    response = (said.output_ref or "response:x").split(":", 1)[1]
    key = f"answer:{task.id}:r-{safe_id(response)}"
    text = next((d["text"] for d in view["deliveries"] if d["kind"] == "answer" and d["key"] == key), None)  # the recorded answer's words
    if not text:
        return None
    if not tw.can_claim(mine):
        return ("team_record_human_response", {"task_id": task.id, "action": "confirm_with_requester", "response_id": response,
                                               "responded_by": said.responded_by.id})
    for child in view["children"]:
        if child.status != "waiting_for_person":
            continue
        cview = turn.work.task_view(turn.project_id, child.id)
        relayed = _mine(turn, cview, "confirm_with_requester")
        if relayed is None or not tw.can_claim(relayed):
            return ("team_record_human_response", {"task_id": child.id, "action": "confirm_with_requester", "response_id": response,
                                                   "responded_by": said.responded_by.id})
        key = f"answer:{child.id}:r-{safe_id(response)}"
        if not any(d["key"] == key for d in cview["deliveries"]):
            return ("team_answer", {"task_id": child.id, "response_id": response, "text": text})
    return ("team_accept", {"task_id": task.id})


BRAINS: dict[str, Callable[[TeamTurn], Optional[Call]]] = {"business": _business, "data_science": _data_science, "model": _model,
                                                           "lead": _lead, "home": _home}


# ---------------------------------------------------------------------------------------------------------- the loop
class ScriptedChoices:
    """The client of a deterministic agent: each reply is the next tool call its rules choose from the stored state (or
    words when nothing is left). No language model is asked, so there is no thinking to keep: the reply carries none,
    and its words are only "Done." or why it stopped. A refused call or a call that would repeat the last one ends the
    turn (the rules never retry blindly)."""

    model = "rules:teamspace"

    def __init__(self, brain: Callable[[TeamTurn], Optional[Call]], turn: TeamTurn, first: Optional[Call]):
        self.brain, self.turn, self.pending, self.last, self.n = brain, turn, first, None, 0

    @staticmethod
    def _say(text: str) -> dict[str, Any]:
        return {"content": text, "tool_calls": [], "finish_reason": "stop", "usage": {"input_tokens": 0, "output_tokens": 0},
                "assistant_message": {"role": "assistant", "content": text}}

    def complete(self, messages: list[dict[str, Any]], tools: Any = None, **_: Any) -> dict[str, Any]:
        last = messages[-1] if messages else {}
        if last.get("role") == "tool" and is_error(last.get("content")):
            return self._say("Stopped: a call was refused.")
        call = self.pending if self.pending is not None else self.brain(self.turn)
        self.pending = None
        if call is None:
            return self._say("Done.")
        if call == self.last:
            return self._say("Stopped: the same call again would make no progress.")
        self.last, self.n = call, self.n + 1
        name, arguments = call
        import json

        cid = f"t{self.n}"
        return {"content": "", "tool_calls": [{"id": cid, "name": name, "arguments": arguments}], "finish_reason": "tool_calls",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "assistant_message": {"role": "assistant", "content": "", "tool_calls": [
                    {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}


def _rules_policy(base: type) -> type:
    """The loop's Policy, marked as chosen by rules (``chooser``): the ledger records the agent's tool choices and its stop
    as rules (``rule:teamspace.<agent>``), never as a model's choices."""
    global _RULES_POLICY
    if _RULES_POLICY is None:
        from dataclasses import dataclass as _dc

        @_dc(frozen=True)
        class RulesPolicy(base):
            chooser: str = "rules"

        _RULES_POLICY = RulesPolicy
    return _RULES_POLICY


_RULES_POLICY: Optional[type] = None


def run_turn(turn: TeamTurn, registry: Any = None) -> dict[str, Any]:
    """One turn of a team agent on the one loop: its rules choose each call, the guard checks it, the run is traced. A turn
    with nothing left to do runs nothing (a repeated delivery). Returns the steps, refusals and how it ended."""
    from ..agents.runtime import Policy, run as run_policy

    base = turn.agent.split(":")[0]
    brain = BRAINS.get(base)
    if brain is None:
        raise KeyError(f"{turn.agent} has no deterministic agent in this release")
    first = brain(turn)
    if first is None:
        return {"agent": turn.agent, "task_id": turn.task_id, "skipped": True, "steps": [], "refused": [], "run": None}
    if registry is None:
        from ..agents import default_registry

        registry = default_registry()
    policy = _rules_policy(Policy)(name=f"{AGENT}.{base}", system=SYSTEM, tools=config_of(turn.agent).policy_tools, max_steps=MAX_STEPS,
                    max_seconds=120.0)
    why = (turn.row or {}).get("kind") or turn.extra.get("mode") or "turn"
    messages = [{"role": "user", "content": f"{why} on {turn.task_id}"}]
    result = run_policy(policy, ScriptedChoices(brain, turn, first), registry, messages, context=turn)
    steps = [{"tool": s.tool, "ok": s.ok, "result": s.result} for s in result.steps]
    return {"agent": turn.agent, "task_id": turn.task_id, "skipped": False, "steps": steps, "stopped": result.stopped,
            "refused": [s["result"]["error"] for s in steps if not s["ok"]], "final": result.final}


# ---------------------------------------------------------------------------------------------------------- entry points
def _work(env: Any) -> Any:
    from .delivery import store_for

    return store_for(env)


def open_case(env: Any, project_id: str, problem_id: str, *, task_id: str, question: str, thread_id: str, member: str,
              initiator: Optional[Row] = None, inputs: tuple[str, ...] = (), owner: str = "data_science",
              expected_output: str = "a feasibility answer or a precise blocker",
              done_criteria: tuple[str, ...] = ("says whether the data can answer the question, or what blocks it",),
              registry: Any = None) -> dict[str, Any]:
    """Business opens the case's task for ``owner`` with the brief, for its person ``member`` in their thread."""
    turn = TeamTurn(env, _work(env), project_id, "business", task_id=task_id, initiator=initiator, person=member,
                    extra={"mode": "open", "owner": owner, "question": question, "problem_id": problem_id, "thread_id": thread_id,
                           "inputs": list(inputs), "expected_output": expected_output, "done_criteria": list(done_criteria)})
    return run_turn(turn, registry)


def person_replies(env: Any, project_id: str, task_id: str, *, person: str, text: str, initiator: Optional[Row] = None,
                   response_id: Optional[str] = None, registry: Any = None) -> dict[str, Any]:
    """The person answered their agent's question in their thread: Business records the response (the person's, never its
    own) and carries it to the task's owner."""
    from .tools import _digest

    response = response_id or f"r-{_digest(task_id, person, text)}"
    turn = TeamTurn(env, _work(env), project_id, "business", task_id=task_id, initiator=initiator, person=person, text=text,
                    extra={"mode": "answer", "response_id": response})
    return run_turn(turn, registry)


def correct(env: Any, project_id: str, task_id: str, *, kitchen_minutes: int, initiator: Optional[Row] = None,
            registry: Any = None) -> dict[str, Any]:
    """A correction after completion (the kitchen adds minutes): the specialist that owns ``task_id`` estimates again on the
    corrected input and supersedes its report; every other task keeps its own snapshot."""
    work = _work(env)
    task = work.task(project_id, task_id)
    if task is None:
        raise KeyError(f"no task {task_id} in project {project_id}")
    turn = TeamTurn(env, work, project_id, task.owner_team, task_id=task_id, initiator=initiator,
                    extra={"mode": "correction", "correction_minutes": int(kitchen_minutes)})
    return run_turn(turn, registry)


def from_thread(env: Any, thread: Mapping[str, Any], *, person: str, text: str, initiator: Optional[Row] = None,
                message_id: str = "", team_kind: Optional[str] = None, ask: Optional[str] = None,
                registry: Any = None) -> Optional[dict[str, Any]]:
    """A person posted in their team thread (the route calls this after saving the message): the team's agent takes it.

    In a Business thread (``team_kind`` the thread's team kind): with ``ask`` (a team agent's name), the message opens a
    case for that team (``open_case``) unless one is already open from this thread, reading the inputs the project declares
    for its teams (``team_inputs``; R3 reads the project's own tables). Without it, a message that answers an open
    clarification asked of this person in this thread is their answer (``person_replies``). Anything else is a message in
    the thread and nothing more. Returns the agent's turn, or None."""
    from .threads import is_home

    if is_home(thread):  # a person's Home conversation with the Lead: a message there answers the Lead's open question, if any
        from . import lead

        work = _work(env)
        waiting = [t for t in lead.lead_tasks(env, str(thread["project_id"])) if t.requester.thread_id == thread["id"]
                   and t.status == "waiting_for_person" and t.requester.member == person]
        if not waiting:
            return None
        return lead.answer(env, str(thread["project_id"]), waiting[0].id, person=person, text=text, initiator=initiator, registry=registry)
    if (team_kind or thread.get("team")) != "business":
        return None
    work = _work(env)
    project_id, problem_id, thread_id = str(thread["project_id"]), str(thread["problem_id"]), str(thread["id"])
    rows = work.deliveries(project_id=project_id)
    asked = [r for r in rows if r["kind"] == "clarification" and r.get("thread_id") == thread_id and r["problem_id"] == problem_id]
    for row in ([] if ask else reversed(asked)):
        task = work.task(project_id, row["task_id"])
        if task is not None and task.status == "waiting_for_person" and case_member(
                TeamTurn(env, work, project_id, "business"), task)[0] == person:
            return person_replies(env, project_id, task.id, person=person, text=text, initiator=initiator, registry=registry)
    if not ask:
        return None
    if ask not in tw.DIRECTORY or tw.DIRECTORY[ask].kind != "team_agent" or ask == "business":
        return {"agent": "business", "refused": [f"{ask} is not a team the Business agent asks"], "steps": [], "skipped": False}
    opened = [work.task(project_id, r["task_id"]) for r in rows if r["kind"] == "task" and r["problem_id"] == problem_id]
    if any(t is not None and t.parent_task_id is None and t.requester.thread_id == thread_id and tw.is_open(t) for t in opened):
        return None
    project = getattr(env, "projects", None)
    inputs = tuple((project.get(project_id) or {}).get("team_inputs") or ()) if project is not None else ()
    return open_case(env, project_id, problem_id, task_id=safe_id(f"T-{message_id or _digest_text(thread_id, text)}"), question=text,
                     thread_id=thread_id, member=person, initiator=initiator, inputs=inputs, owner=ask, registry=registry)


def _digest_text(*parts: str) -> str:
    from .tools import _digest

    return _digest(*parts)


def preview_report(env: Any, project_id: str, task_id: str) -> Optional[tw.WorkReport]:
    """The task's report as it would read now, built from its action log and joined children and never stored: while a
    child is still out it is partial and names which (the W8 task page shows it). None for a task not yet accepted."""
    work = _work(env)
    view = work.task_view(project_id, task_id)
    if view is None or view["task"].status in ("open", "cancelled", "superseded"):
        return None
    task, join = view["task"], view["join"]
    returned = [r for r in view["child_reports"] if join is None or r.id in join.reports]
    waiting = list(join.outstanding) if join is not None else []
    answer = " ".join(r.answer for r in returned if r.status != "superseded") or "No result has returned yet."
    if waiting:
        answer += " Still waiting on: " + ", ".join(waiting) + "."
    owner = view["next_owner"] or task.owner_team
    return tw.build_report(task, view["actions"], view["child_reports"], children=view["children"], answer=answer,
                           next_task=tw.NextStep(owner=owner, unblocks="the outstanding replies" if waiting else "the owner's report"),
                           status="partial" if waiting else None)


# ---------------------------------------------------------------------------------------------------------- delivery
def _focus(work: Any, row: Row) -> Optional[str]:
    """The task the recipient acts on: a child's reply is joined on the recipient's own task (the parent); anything else is
    the row's task."""
    task = work.task(row["project_id"], row["task_id"])
    if task is None:
        return None
    if row.get("kind") in ("reply", "clarification") and task.parent_task_id:
        parent = work.task(row["project_id"], task.parent_task_id)
        if parent is not None and parent.owner_team == row["to_team"]:
            return parent.id
    return task.id


def deliver(env: Any, row: Row) -> Row:
    """The recipient agent's turn on a delivered row (the outbox's default target, ``delivery.default_target``, posts the
    row into its team thread first, then calls this).

    Idempotent on the row id: the agent's rules read the task as stored, so a row handed over again finds its work done
    and runs no step. A recipient without a deterministic agent in this release (Sales, Marketing, ML Engineering...)
    only receives it: the row is its inbox. A refused step blocks the row (the requester sees that it could not be taken,
    never the refusal's words)."""
    from .delivery import PermanentFailure

    receipt: Row = {"row": row["id"], "agent": None, "skipped": True, "steps": []}
    if row["to_team"].split(":")[0] not in BRAINS:
        return receipt
    work = _work(env)
    focus = _focus(work, row)
    if focus is None:
        raise PermanentFailure(f"no task {row['task_id']} in project {row['project_id']}")
    turn = TeamTurn(env, work, row["project_id"], row["to_team"], task_id=focus, initiator=row.get("initiator"), row=dict(row))
    out = run_turn(turn)
    if out["refused"]:
        raise PermanentFailure(f"{row['to_team']} refused a step: {out['refused'][0]}")
    if row["to_team"] == "home":
        from .lead import to_home_chat

        to_home_chat(env, row)  # the Lead's question or report, in the person's Home conversation (once per row)
    return {**receipt, "agent": row["to_team"], "task_id": focus, "skipped": out["skipped"], "steps": [s["tool"] for s in out["steps"]]}


__all__ = ["AGENT", "BRAINS", "CONFIGS", "CLARIFY_TARGET", "ScriptedChoices", "TeamAgentConfig", "config_of", "correct", "deliver",
           "from_thread", "job_description", "open_case", "person_replies", "preview_report", "run_turn"]
