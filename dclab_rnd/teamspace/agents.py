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

from dclab_core import target_contract as _tc
from dclab_core import team_work as tw

from . import fixtures
from .tools import ACTS, MODEL_TOOLS, TeamTurn, case_member, is_error, safe_id

Row = dict[str, Any]
Call = tuple[str, dict[str, Any]]
AGENT = "teamspace"  # harness.AGENTS: the team agents' runs (policy names teamspace.<agent>)
MAX_STEPS = 40
CLARIFY_TARGET = "Does churn mean cancellation, or failure to renew by the contractual renewal date?"
# B2: a reworded CLARIFY_TARGET must still name both events it offers (``team_brain.check_question``), as the keyword reader
# reads them (``target_contract.EVENTS``)
CLARIFY_OPTIONS = tuple(e.names for e in _tc.EVENTS.values())
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


def _claimed(action: Optional[tw.TeamAction]) -> bool:
    """Whether an action has a terminal record: a running one (a crash between its two writes) is done again."""
    return action is not None and tw.can_claim(action)


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


def _needs_outcome(task: tw.TeamTask) -> bool:
    """T1: whether Data Science asks what the outcome means before it judges a table: no outcome is confirmed yet, the
    task reads a table, and its question leaves the outcome vague ("churn") or names one in its own words ("churn means an
    explicit cancellation", "who will cancel"). A brief's sentence is never taken as the confirmed outcome: the person
    confirms it once, and what they confirm is the contract the table is judged against."""
    from dclab_core import target_contract as tc

    if task.target is not None:
        return False
    if _delivery_request(task):  # B4.1 (D-07): an order's arrival or a courier's acceptance has no churn to define
        return False
    if _ambiguous_target(task.question):
        return True
    said = task.question.lower()
    return any(i.startswith("sample:") for i in task.allowed_inputs) and any(re.search(rf"\b(?:{e.names})\b", said)
                                                                            for e in tc.EVENTS.values())


# B4.1 (D-07 of the first dev run: "When's it landing at the customer's door, and will a rider grab it on the first ping?"):
# the arrival and the first-offer acceptance said in everyday words, not only "arrive", "ETA" and "accept"
_ARRIVAL = re.compile(r"\b(?:arriv\w*|eta|landing|land|lands|reach(?:es|ing)?\s+(?:the\s+)?(?:customer|door|address|destination)|"
                      r"get(?:s|ting)?\s+(?:there|to\s+(?:the\s+)?(?:customer|door|address))|delivered|delivery\s+time|"
                      r"at\s+(?:the|their|the\s+customer's|the\s+customer’s)\s+door|what\s+time\s+will|when\s+will\s+(?:it|the\s+order|order))\b", re.I)
_ACCEPTANCE = re.compile(r"\b(?:accept\w*|grab\w*|take\s+(?:it|the\s+(?:order|job|offer))|pick\w*\s+(?:it\s+)?up|"
                         r"first\s+(?:ping|offer)|courier\s+offer)\b", re.I)


def _delivery_request(task: tw.TeamTask) -> bool:
    """B4.1: a request about an order's delivery: it may read a dispatch snapshot and no table sample. Its outcome needs no
    definition (no churn), and the checks a table answers (columns, leakage, a baseline) do not apply to it."""
    return any(i.startswith("dispatch:") for i in task.allowed_inputs) and not any(i.startswith("sample:") for i in task.allowed_inputs)


def _specialists(task: tw.TeamTask) -> list[str]:
    """The model specialists a dispatch question needs: one for the arrival, one for first-offer acceptance."""
    if not any(i.startswith("dispatch:") for i in task.allowed_inputs):
        return []
    said = task.question
    kinds = []
    if _ARRIVAL.search(said):
        kinds.append("eta")
    if _ACCEPTANCE.search(said):
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
            return ("team_delegate", {"action": x.get("action") or open_action, "new_task_id": turn.task_id, "owner_team": x["owner"],
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
            if task.target is None and said is not None and asked.output_ref and asked.responded_by is not None:
                # T1, the review: a crash between recording the answer and saving its outcome leaves clear words unsaved;
                # recording the same response again saves them (the same outcome again changes nothing)
                from .tools import confirmed_outcome

                response = asked.output_ref.split(":", 1)[1]
                if confirmed_outcome(task, _answer_text(view), asked.responded_by.id, response, turn=turn) is not None:
                    return ("team_record_human_response", {"task_id": task.id, "action": "confirm_target", "response_id": response,
                                                           "responded_by": asked.responded_by.id})
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
        if _needs_outcome(task) and asked is None:
            from . import seats  # B2: on the helper arm, the question may be worded plainly where its check passes

            return ("team_clarify", {"task_id": task.id, "action": "confirm_target",
                                     "question": seats.of(turn).question(task, "confirm_target", CLARIFY_TARGET, CLARIFY_OPTIONS)})
        return _feasibility(turn, view)
    if status == "blocked":
        return None if _answered(view) else _blocker_report(turn, view)
    if status in ("complete", "partial") and not view["children"] and not _answered(view):
        return _feasibility(turn, view)  # completed without its report (an older crash): report again
    if status in ("waiting_for_team", "partial", "complete") and view["children"]:
        return _join(turn, view)
    return None


def _sample(task: tw.TeamTask) -> Optional[str]:
    return next((i for i in task.allowed_inputs if i.startswith("sample:")), None)


def _answered(view: Row) -> bool:
    """T1: whether the task's current report answers its current brief revision (a changed outcome raises the revision,
    so the earlier report stands until the new one supersedes it)."""
    current = _current_report(view)
    return current is not None and current.brief_revision >= view["task"].brief_revision


def _next_revision(view: Row) -> int:
    current = _current_report(view)
    return 1 if current is None else current.revision + 1


def _check(turn: TeamTurn, view: Row) -> Row:
    """What the task's sample can answer for the outcome confirmed (``fixtures.feasibility`` with the task's ``target``).
    T1: when the outcome needed a definition, the person answered, and their words did not settle it (both events, none,
    a hedge), the outcome stays open: a blocker naming who confirms it, never a guess."""
    task = view["task"]
    if task.target is None and _claimed(_mine(turn, view, "confirm_target")):
        return {"ok": False, "unsure": True, "missing": "a confirmed outcome", "field": "target", "supplier": task.requester.team,
                "why": "the answer does not settle what the outcome means (it names both events, neither, or one both ways)",
                "matters": "a ranking built on the wrong event answers another question than the one asked"}
    return fixtures.feasibility(_sample(task) or "", task.target)


def _outcome_lines(task: tw.TeamTask, check: Row) -> tuple[list[str], list[str]]:
    """The assumption and the limit that say which outcome was checked, as confirmed (T1): the person's words with who
    confirmed them, and the event with the column that records it, so a Business member reads it in the report."""
    from dclab_core import target_contract as tc

    from . import seats

    t = task.target
    if t is None:
        return [], []
    if seats.read_by_helper(t):  # B2: read from the request by the helper and checked by code; the person did not confirm it
        return ([f"The outcome is: {tc.event_label(t.event)}, read from {t.confirmed_by}'s request by the helper and checked against "
                 f"the table's label; {t.confirmed_by} can correct it."],
                [f"The outcome checked is {tc.event_label(t.event)}, as the helper read {t.confirmed_by}'s request (brief revision "
                 f"{t.brief_revision}, not confirmed by them): {check.get('outcome')} records it, with known outcomes."])
    if seats.reading(t) == "answer":  # B2, review F7: the person's words, the helper's reading of them
        return ([f"The outcome is: {t.words} (said by {t.confirmed_by}; read by the helper as {tc.event_label(t.event)})."],
                [f"The outcome checked is {tc.event_label(t.event)}, as the helper read {t.confirmed_by}'s answer (brief revision "
                 f"{t.brief_revision}): {check.get('outcome')} records it, with known outcomes."])
    return ([f"The outcome is: {t.words} (confirmed by {t.confirmed_by})."],
            [f"The outcome checked is {tc.event_label(t.event)}, as {t.confirmed_by} confirmed (brief revision {t.brief_revision}): "
             f"{check.get('outcome')} records it, with known outcomes."])


def _feasibility(turn: TeamTurn, view: Row) -> Optional[Call]:
    """Inspect the sample; block on what is missing (T1: a record of the outcome the person confirmed included), or check
    leakage, name the baseline and the uncertainty, and report."""
    task = view["task"]
    if not _claimed(_mine(turn, view, "inspect_schema", _brief(task))):  # review F1: running after a crash is run again
        return ("team_act", {"task_id": task.id, "action": "inspect_schema"})
    sample = _sample(task) or ""
    check = _check(turn, view)
    if not check["ok"]:
        return ("team_block", {"task_id": task.id, "reason": check["why"]})
    for action in ("check_leakage", "define_baseline", "explain_uncertainty"):
        if not _claimed(_mine(turn, view, action, _brief(task))):
            return ("team_act", {"task_id": task.id, "action": action})
    if _answered(view):
        return None
    assumptions, checked = _outcome_lines(task, check)
    if task.target is None:
        said = _answer_text(view)
        confirmed = _confirmed_by(turn, view)
        assumptions = [f"The outcome is: {said}" + (f" (confirmed by {confirmed.responded_by.id})." if confirmed else ".")] if said else []
    evidence = [{"source": sample, "ref": f"schema:{sample}", "input_revision": _brief(task), "assumptions": assumptions,
                 "uncertainty": "the sample's columns and dates only; no account was scored"}]
    excluded, outcome = tuple(check["excluded"]), check["outcome"]
    late = [c for c in excluded if c != outcome]
    limits = ["Checked on the sample's columns and dates only: no model was trained and no account was scored.",
              f"Excluded as inputs: {', '.join(excluded)}, known only after renewal (it is the outcome)." if not late else
              f"Excluded as inputs: {', '.join(excluded)}, each known only after renewal ({outcome} is the outcome; "
              f"{', '.join(late)} is written after it, a late feature).", *checked]
    usual = (fixtures.sample(sample) or {}).get("outcome")
    answer = ("The sample has renewal dates and mature outcomes. We can score seven days before renewal." if outcome == usual else
              f"The sample has renewal dates and mature outcomes for the outcome confirmed ({outcome}). We can score seven days before renewal.")
    return ("team_return_report", {"task_id": task.id, "outcome": "complete", "revision": _next_revision(view), "evidence": evidence,
                                   "limits": limits, "answer": answer})


def _blocker_report(turn: TeamTurn, view: Row) -> Call:
    """The blocked report: what is missing, why it matters, who can supply it (and from where), and which task resumes."""
    task = view["task"]
    sample = _sample(task) or "the sample"
    check = _check(turn, view)
    from .store import team_label

    supplier = check.get("supplier") or task.requester.team
    matters = f" Why it matters: {check['matters']}." if check.get("matters") else ""
    if check.get("unsure"):
        answer = (f"Blocked: {check['why']}.{matters} The {team_label(supplier)} team confirms with its person whether the outcome is "
                  f"a cancellation or a failure to renew by the contractual renewal date; work resumes on {task.id} when they do.")
        unblocks = "confirm what the outcome means: a cancellation, or a failure to renew by the contractual renewal date"
    else:
        source = f" ({check['source']})" if check.get("source") else ""
        answer = (f"Blocked: {check.get('why', 'a needed input is missing')} (missing field: {check.get('missing')}).{matters} "
                  f"The {team_label(supplier)} team can supply it{source}; work resumes on {task.id} when it is attached to {sample}.")
        unblocks = f"attach {check.get('missing')} to {sample}"
    return ("team_return_report", {"task_id": task.id, "outcome": "blocked", "revision": _next_revision(view), "answer": answer,
                                   "next_owner": supplier, "unblocks": unblocks})


def _join(turn: TeamTurn, view: Row, *, attribute: bool = False, order: Optional[list[str]] = None) -> Optional[Call]:
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
    order = [o for o in (order or []) if any(c.id == o for c in view["children"])] + [c.id for c in view["children"] if c.id not in (order or [])]
    # T3: a Sales or Marketing report resting on a member no longer in the team is not joined (its reply was refused at
    # delivery too): the parent is partial and names that team, which asks a member again
    from .commercial import answer_withdrawn

    withdrawn = {rid: why for rid in join.reports if (why := answer_withdrawn(turn.env, turn.project_id, joined[rid].task_id))}
    returned = sorted((joined[rid] for rid in join.reports if rid not in withdrawn), key=lambda r: order.index(r.task_id))
    if attribute:  # the Lead says which team answered what
        from .store import team_label

        answer = "\n".join(f"{team_label(r.owner)}: {r.answer}" for r in returned)  # one team a line (three teams read as one block otherwise)
    else:
        answer = " ".join(r.answer for r in returned)
    evidence = [e.model_dump() for r in returned for e in r.evidence]
    limits = []
    if any(r.simulated for r in returned):
        fixtures_used = sorted({e.ref for r in returned for e in r.evidence})
        limits.append(f"Simulated: the figures come from fixtures ({', '.join(fixtures_used) or 'none'}); they illustrate the "
                      "handoff and are not measured.")
    unresolved = [joined[rid] for rid in join.reports if joined[rid].task_id in join.unresolved and rid not in withdrawn]
    if withdrawn:
        from .store import team_label

        gone = [joined[rid] for rid in withdrawn]
        # the standing replies first, then what no longer stands
        answer = "\n".join(([answer] if answer else []) + [f"{team_label(r.owner)}: not joined: {withdrawn[r.id]}." for r in gone])
        limits.append("A reply was withdrawn: the member whose answer it rests on is no longer in that team.")
    call: dict[str, Any] = {"task_id": task.id, "answer": answer, "evidence": evidence, "limits": limits,
                            "revision": 1 if current is None else current.revision + 1,
                            "outcome": "partial" if unresolved or withdrawn else "complete"}
    if withdrawn:
        first = joined[next(iter(withdrawn))]
        call.update(next_owner=first.owner, unblocks=f"a member of the {first.owner} team answers again")
    elif unresolved:
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
        if first is None or not tw.can_claim(first):  # T2: a crash between its two writes leaves it running: the same call
            return ("team_model_call", {"task_id": task.id, "action": "score"})  # finishes that attempt, never a second score
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
    correction = turn.extra.get("correction")  # T2: judged by correct() (tools.judge_correction) before the turn
    if status == "complete" and correction is not None:
        revision, base, minutes = correction["revision"], int(correction["base"]), int(correction["minutes"])
        again = _mine(turn, view, "score", revision)
        if again is None or not tw.can_claim(again):  # a crash between its two writes: the same call finishes that attempt
            args = {"task_id": task.id, "action": "score", "correction_minutes": minutes, "base_revision": base}
            return ("team_model_call", {**args, **({"correction_id": correction["id"]} if correction.get("id") else {})})
        if correction["verdict"] != "apply" or again.status not in tw.DONE_ACTIONS:
            return None  # refused on an older base (a skipped attempt and a reconcile reply), or not scored: no new revision
        current = _current_report(view)
        if current is None or current.revision != base or any(e.input_revision == revision for e in current.evidence):
            return None  # reported already, or another correction moved the report on meanwhile
        shown = {e.input_revision for e in current.evidence}  # "was": the value of the report this one supersedes
        before = next((a for a in view["actions"] if a.action == "score" and a.status in tw.DONE_ACTIONS and a.input_revision in shown), None)
        return _estimate_report(task, kind, snapshot, again, (minutes, before), current.revision + 1)
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
            was = _value(before) if before else "unknown"  # the superseded report's value, not the first score's
            if minutes == 0:  # T2: the kitchen withdrew its delay: the dispatch estimate again, never "added 0 minutes"
                answer = (f"Revised: the estimated arrival is {_value(score)} (was {was} in the previous report; the kitchen's "
                          f"delay was withdrawn, so this is the {at} estimate again, fixed route).")
            else:
                answer = (f"Revised: the estimated arrival is {_value(score)} (was {was} in the previous report; the kitchen "
                          f"added {minutes} minutes to the {at} estimate, fixed route).")
        assumptions = ["fixed route"]
    else:
        answer = f"The first-offer acceptance estimate was {_value(score)}."
        assumptions = [f"the courier pool at {at}"]
    evidence = [{"source": snapshot, "ref": f"fixture:{fixture}", "input_revision": score.input_revision,
                 "uncertainty": "a fixture, not a measured estimate", "assumptions": assumptions}]
    return ("team_return_report", {"task_id": task.id, "outcome": "complete", "revision": revision, "answer": answer,
                                   "evidence": evidence, "limits": ["Simulated: a fixture result that illustrates the handoff."]})


# ---------------------------------------------------------------------------------------------------------- ML Engineering and Sales
FOUND = "\nData Science found: "  # what the Lead adds to ML Engineering's question, after the person's own words
_CONFIRMED = re.compile(r"The outcome is: (.+?)(?: \(confirmed by [^)]*\))?\.?(?:\n|$)")
_WHEN = re.compile(r"\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|fourteen|thirty)\s+days?\s+before\s+(?:each\s+|the\s+)?renewal)\b", re.I)


def _worker(first: str, report: Callable[[TeamTurn, Row, tw.TeamAction], Call],
            blocked: Optional[Callable[[TeamTurn, Row, tw.TeamAction], Call]] = None) -> Callable[[TeamTurn], Optional[Call]]:
    """A team agent that takes a Lead's task, runs one checking action on what it may read, and reports (0.168.0): accept,
    act, then a report built from the action (which may block the task first: ``blocked`` then reports the blocker), or a
    blocker report naming who supplies what is missing."""

    def brain(turn: TeamTurn) -> Optional[Call]:
        view = _view(turn)
        if view is None:
            return None
        task, status = view["task"], view["task"].status
        if status == "open":
            return ("team_accept", {"task_id": task.id})
        done = _mine(turn, view, first, _brief(task))
        if status == "accepted":
            if done is None or not tw.can_claim(done):  # review F1: a crash between its two writes leaves it running: run it again
                return ("team_act", {"task_id": task.id, "action": first})
            if done.status in tw.FAILED_ACTIONS:
                return ("team_block", {"task_id": task.id, "reason": done.failure_ref or f"{first} failed"})
            return None if _answered(view) else report(turn, view, done)
        if status == "blocked" and not _answered(view):
            if blocked is not None and done is not None and done.status in tw.DONE_ACTIONS:
                return blocked(turn, view, done)  # its own check passed: it blocked on what its team must give
            if first == "validate_contract" and not fixtures.feasibility(_sample(task) or "", task.target)["ok"]:  # review F5: its own reason
                return _blocker_report(turn, view)  # the sample itself lacks what is needed (T1: a record of the outcome confirmed)
            why = (done.failure_ref if done else None) or f"{first} could not run"
            return ("team_return_report", {"task_id": task.id, "outcome": "blocked", "revision": _next_revision(view),
                                           "answer": f"Blocked: {why}.", "next_owner": task.requester.team,
                                           "unblocks": f"give the {first.replace('_', ' ')} what it reads"})
        if status in ("complete", "partial") and not _answered(view) and done is not None and done.status in tw.DONE_ACTIONS:
            return report(turn, view, done)  # moved without its report (an older crash): report again
        return None

    return brain


def _contract_report(turn: TeamTurn, view: Row, done: tw.TeamAction) -> Call:
    """ML Engineering: the experiment the request needs, from the contract it checked; nothing is trained (this release binds
    no training tool, and a run needs the owner's approval and a budget)."""
    task = view["task"]
    sample = _sample(task) or ""
    check = fixtures.feasibility(sample, task.target)
    when = _WHEN.search(task.question)
    moment = when.group(1) if when else "at the prediction moment Data Science confirms"
    # review F4: the outcome as the person confirmed it to Data Science, not assumed; the review of 2026-11-22: read only what
    # the Lead added after the person's words ("Data Science found: …"), never a "The outcome is:" the person wrote first.
    # T1: the outcome confirmed travels as a field (``task.target``, carried from Data Science's task); the sentence is
    # read only for a task that carries none
    found = task.question.rpartition(FOUND)[2] if FOUND in task.question else ""
    confirmed = _CONFIRMED.search(found)
    said = task.target.words if task.target is not None else (confirmed.group(1) if confirmed else None)
    excluded = ", ".join(check.get("excluded") or ())
    column = (check.get("outcome") if task.target is not None else excluded) or "the outcome column"
    outcome = (f"the outcome is \"{said.strip().rstrip('.')}\", as confirmed with Data Science (in the sample: {column}, never an input)"
               if said else f"the outcome is the one Data Science confirms (in the sample: {column}, never an input)")
    from . import seats

    if task.target is not None and seats.reading(task.target):  # B2, review F2: never "confirmed" for what the helper read
        label, how = seats.outcome_words(task.target)
        outcome = f"the outcome is {label}, {how} (in the sample: {column}, never an input)"
    answer = (f"The experiment can be set up: score each account {moment}, train on earlier renewals and hold out the latest "
              f"renewals by time; {outcome}. Nothing was trained: a training run needs the owner's approval and a budget.")
    assumptions, checked = _outcome_lines(task, check)
    evidence = [{"source": sample, "ref": done.output_ref or f"contract:{sample}", "input_revision": _brief(task),
                 "uncertainty": "the sample's columns only; no run, no score",
                 "assumptions": [f"the prediction moment: {moment}", *assumptions]}]
    limits = ["Checked against the sample's columns only: no model was trained, evaluated or registered.", *checked]
    if task.target is not None and excluded:
        limits.append(f"Excluded as inputs: {excluded} (known only after renewal).")
    return ("team_return_report", {"task_id": task.id, "outcome": "complete", "revision": _next_revision(view), "answer": answer,
                                   "evidence": evidence, "limits": limits})


_ml_engineering = _worker("validate_contract", _contract_report)
# T3 (2026-11-23): Sales and Marketing are teamspace/commercial.py: each asks its own team's member (Sales its call capacity,
# Marketing the offer's terms) in its own thread, and returns a capped, restricted call list or the confirmed terms, or a
# blocker. With no team of its kind to ask, Sales blocks on the capacity as before (CAPACITY: who supplies it).
from .commercial import CAPACITY, marketing as _marketing, sales as _sales  # noqa: E402


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
        for team in _route(turn, view):
            child = f"{task.id}.{team}"
            question = task.question
            if team == "ml_engineering":  # review F4: after Data Science has answered, with what it found and what the person confirmed
                found = _data_science_answer(turn, view)
                if found is None:
                    continue  # not yet, or Data Science is blocked: the modelling is not agreed, so it is not asked
                question = f"{task.question}{FOUND}{found}"
            delegated = _mine(turn, view, "delegate", child)
            # missing, or its delegation not recorded or not finished (a crash between the two writes resumes here, review F1)
            if turn.work.task(turn.project_id, child) is None or delegated is None or not tw.can_claim(delegated):
                call = {"action": "delegate", "task_id": task.id, "new_task_id": child, "owner_team": team,
                        "question": question, "expected_output": "a feasibility answer or a precise blocker",
                        "done_criteria": ["says whether the data can answer the question, or what blocks it"],
                        "allowed_inputs": inputs}
                science = turn.work.task(turn.project_id, f"{task.id}.data_science") if team == "ml_engineering" else None
                if science is not None and science.target is not None:  # T1: the confirmed outcome travels as a field
                    call["target_from"] = science.id
                return ("team_delegate", call)
    if status == "blocked":
        if _current_report(view) is not None:
            return None
        pre = _mine(turn, view, "check_prerequisites", _brief(task))
        why = (pre.failure_ref if pre else None) or "a prerequisite is missing"
        return ("team_return_report", {"task_id": task.id, "outcome": "blocked", "revision": 1, "next_owner": task.requester.team,
                                       "unblocks": "declare what the teams may read for this project (its team inputs)",
                                       "answer": f"Blocked before any team was asked: {why}."})
    if view["children"] and status in ("accepted", "waiting_for_team", "partial", "complete"):
        return _join(turn, view, attribute=True, order=[f"{task.id}.{t}" for t in _route(turn, view)])  # the route's order
    return None


def _route(turn: TeamTurn, view: Row) -> tuple[str, ...]:
    """The teams the Lead hands this request to (``team_work.lead_route``). B2: Sales stays in it when a Sales task was
    opened for it (the helper read outreach the keyword rule missed, on an earlier turn); on the helper arm, before the
    Lead delegates anything, the helper may add Sales where its check passes (``seats.Seats.route``)."""
    from . import seats

    task = view["task"]
    teams = tw.lead_route(help_of(task) or "", task.question)
    if teams and "sales" not in teams and any(c.id == f"{task.id}.sales" for c in view["children"]):
        teams = teams + ("sales",)
    return seats.of(turn).route(task, teams, help_of(task) or "", view)


def _data_science_answer(turn: TeamTurn, view: Row) -> Optional[str]:
    """Data Science's completed answer on this Lead task, with the outcome the person confirmed to it, or None while it is
    out or when it returned blocked."""
    child = f"{view['task'].id}.data_science"
    done = [r for r in view["child_reports"] if r.task_id == child and r.status == "complete"]
    if not done:
        return None
    r = max(done, key=lambda x: x.revision)
    said = [a for e in r.evidence for a in e.assumptions if a.startswith("The outcome is: ")]
    return r.answer + ("\n" + said[0] if said else "")


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
                                                           "lead": _lead, "home": _home, "ml_engineering": _ml_engineering,
                                                           "sales": _sales, "marketing": _marketing}


def _unsupported(turn: TeamTurn) -> Optional[Call]:
    """T3: the turn of a directory agent with no rules in this release, on a task it was asked: it says so at once (accept,
    block, a blocked report naming who acts next), so the requester sees an answer instead of a task open for ever."""
    from .store import team_label

    view = _view(turn)
    if view is None or view["task"].owner_team != turn.agent:
        return None
    task, status = view["task"], view["task"].status
    if status == "open":
        return ("team_accept", {"task_id": task.id})
    if status == "accepted":
        return ("team_block", {"task_id": task.id, "reason": f"no {team_label(turn.agent)} agent answers in this release"})
    if status == "blocked" and _current_report(view) is None:
        return ("team_return_report", {"task_id": task.id, "outcome": "blocked", "revision": 1, "next_owner": task.requester.team,
                                       "unblocks": "ask a team whose agent can answer (the recipients the thread lists)",
                                       "answer": f"Unsupported: no {team_label(turn.agent)} agent answers in this release, so "
                                                 "nothing was done on this task. Ask a team that can answer."})
    return None


def askable() -> list[dict[str, Any]]:
    """T3: the teams a Business thread can ask, those whose agent really answers in this release (a team agent of the
    directory with its rules), in the directory's order."""
    from .store import team_label

    return [{"name": name, "label": team_label(name), "asks_its_team": name in ("sales", "marketing")}
            for name, c in tw.DIRECTORY.items() if c.kind == "team_agent" and name != "business" and name in BRAINS]


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
    with nothing left to do runs nothing (a repeated delivery). Returns the steps, refusals and how it ended.

    B1 (2026-11-23): the role's brain (``brains.arm_for``: the audited switch, default ``rules``). A role switched to the
    helper or agent arm builds its model's request on the state the rules are about to read, and sends it only after the
    rules' run has ended, on its own thread (``brains.finish``): the rules, and the delivery around them, never wait on a
    model. In B1 the rules always act, and a model that cannot answer falls back to them with its reason. Every turn that
    ran records which brain answered, even when its run raised. The rules arm runs as before. Returns ``brain`` too.

    B2 (2026-11-23): the turn's helper seats (``seats.attach``) are on it before the rules choose anything, so the rules'
    functions and the tools reach the seats a helper-arm role has (none on the rules: the rules run exactly as before).
    Before Data Science's rules would ask what the outcome means, the requester's helper may read the request
    (``seats.before_rules``). ``brain.acted`` is ``helper`` when one of its checked answers was used.

    B3 (2026-11-24): a role on the agent arm no longer answers in shadow. On a delivered row its model runs the turn
    (``agent_arm.act``: plan first, its own choice of calls through the same tools and guard, budgets, one turn per
    delivery). When it cannot act before its first write, the rules run the turn on a run of their own and the fallback is
    recorded; after a write they never run on top. Its result has ``refused`` empty (a call the guard refuses is the guard
    doing its job, not a broken delivery: ``guard_refused`` lists them) and ``agent_turn``."""
    from ..agents.harness import tracer
    from ..agents.runtime import Policy, run as run_policy
    from . import brains, seats

    base = turn.agent.split(":")[0]
    brain = BRAINS.get(base) or (_unsupported if turn.extra.get("unsupported") and base in CONFIGS else None)
    if brain is None:
        raise KeyError(f"{turn.agent} has no deterministic agent in this release")
    from .. import observe

    name = f"{AGENT}.{base}"
    arm, source = brains.arm_for(turn.env, base)
    trace = tracer(name)  # the run's id, known before it starts: a model asked at a seat or beside the rules joins this run
    held = seats.attach(turn, arm, trace.run_id)
    fallback = None
    # one span for the turn: the seats asked before and inside the rules' run, the run, and the requests sent after it
    with observe.span(name=f"team_turn {name}"[:120], **{"dclab.agent": name, "dclab.run": trace.run_id}):
        if arm == "agent" and brain is BRAINS.get(base):
            # B3: the model acts inside the strict layer; before its first write a model that cannot answer leaves the turn to
            # the rules (on their own run), after one the rules never run on top
            from ..agents import default_registry
            from . import agent_arm

            registry = registry if registry is not None else default_registry()

            out = agent_arm.act(turn, registry, base=base, source=source, name=name, trace=trace, brain=brain)
            if not isinstance(out, agent_arm.Fallback):
                return out
            fallback, trace = out, tracer(name)
        seats.before_rules(turn)
        first = brain(turn)
        if first is None:
            if fallback is not None:  # the rules have nothing left either: the turn is recorded all the same
                brains.record(turn, arm, source, ("none", {}), fallback.model, fallback.run)
            return {"agent": turn.agent, "task_id": turn.task_id, "skipped": True, "steps": [], "refused": [], "run": None}
        if registry is None:
            from ..agents import default_registry

            registry = default_registry()
        policy = _rules_policy(Policy)(name=name, system=SYSTEM, tools=config_of(turn.agent).policy_tools, max_steps=MAX_STEPS,
                                       max_seconds=120.0)
        why = (turn.row or {}).get("kind") or turn.extra.get("mode") or "turn"
        messages = [{"role": "user", "content": f"{why} on {turn.task_id}"}]
        if fallback is not None:
            prepared: Any = fallback.model  # recorded at once: why the rules act on the agent arm's turn
        elif arm == "agent":
            prepared = None  # a directory agent without rules of its own (unsupported): nothing to fall back from
        else:
            prepared = brains.brain(arm).prepare(turn, first, policy.tools, registry)  # built on the state the rules read; not sent
        completed, result = False, None
        try:
            result = run_policy(policy, ScriptedChoices(brain, turn, first), registry, messages, context=turn, trace=trace)
            completed = True
        finally:  # recorded whether the run completed or raised; a model beside the rules is asked only after a completed run
            brains.finish(turn, arm, source, first, prepared, trace.run_id, completed, steps=result.steps if result is not None else ())
    steps = [{"tool": s.tool, "ok": s.ok, "result": s.result} for s in result.steps]
    return {"agent": turn.agent, "task_id": turn.task_id, "skipped": False, "steps": steps, "stopped": result.stopped,
            "refused": [s["result"]["error"] for s in steps if not s["ok"]], "final": result.final,
            "brain": {"arm": arm, "source": source, "acted": "helper" if held.acted() else "rules",
                      "asked": isinstance(prepared, brains.Asked) or bool(held.records)}}


# ---------------------------------------------------------------------------------------------------------- entry points
def _work(env: Any) -> Any:
    from .delivery import store_for

    return store_for(env)


def open_case(env: Any, project_id: str, problem_id: str, *, task_id: str, question: str, thread_id: str, member: str,
              initiator: Optional[Row] = None, inputs: tuple[str, ...] = (), owner: str = "data_science",
              expected_output: str = "a feasibility answer or a precise blocker",
              done_criteria: tuple[str, ...] = ("says whether the data can answer the question, or what blocks it",),
              registry: Any = None) -> dict[str, Any]:
    """Business opens the case's task for ``owner`` with the brief, for its person ``member`` in their thread (T3: through
    the directory action that names that team: ``request_sales_input``, ``request_marketing_input``, else
    ``delegate_analysis``)."""
    turn = TeamTurn(env, _work(env), project_id, "business", task_id=task_id, initiator=initiator, person=member,
                    extra={"mode": "open", "owner": owner, "action": OPEN_ACTIONS.get(owner, "delegate_analysis"),
                           "question": question, "problem_id": problem_id, "thread_id": thread_id,
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


def confirm_outcome(env: Any, project_id: str, task_id: str, *, person: str, text: str, initiator: Optional[Row] = None,
                    response_id: Optional[str] = None) -> dict[str, Any]:
    """T1 (0.170.0): the person says (again) what the outcome means, after the task was asked or answered: their words as
    the task's confirmed outcome (``tools.confirmed_outcome``), saved by ``store.confirm_target``.

    - Words that do not settle it (both events, none, a hedge) change nothing: ``{"outcome": "unsure"}``.
    - The same event in other words changes nothing (the work that rested on it stands): ``changed`` False.
    - Another event, or a first one on a task that answered (or blocked) without one, raises the task's brief revision;
      every task that carried the earlier answer (ML Engineering's copy) is revised with it; each works again on its new
      revision and its earlier report stays as it was until the new one supersedes it (``stale`` lists them).

    Only the case's member (the person who asked) confirms it, as the member the call acts for (``initiator``), while
    they may still change work in the project (``delivery.authorize``). No route calls this yet: the person's own route
    (their thread, beside ``from_thread``) is the next package's (T3/T5b) wiring. Returns ``outcome``, ``changed``,
    ``revised`` (task ids), ``stale`` (report ids) and ``refused``."""
    from .tools import _digest, confirmed_outcome, holds_role, save_outcome

    work = _work(env)
    task = work.task(project_id, task_id)
    if task is None:
        raise KeyError(f"no task {task_id} in project {project_id}")
    out: dict[str, Any] = {"task_id": task_id, "outcome": None, "changed": False, "revised": [], "stale": [], "refused": []}
    ctx = TeamTurn(env, work, project_id, task.owner_team, task_id=task_id, initiator=initiator, person=person, text=text)
    who = (initiator or {}).get("user_id")
    why = (f"this call acts for {who}; it cannot confirm an outcome for {person}" if who is not None and str(who) != person
           else holds_role(ctx, task, "human:requester", person))
    if why is None and task.status == "waiting_for_person":  # the review: its question is open; the answer goes there
        why = f"{task_id} is asking its question: answer it in the thread, where it is recorded as the person's answer"
    if why is None and initiator:
        from ..memory.scope import AccessDenied
        from .delivery import authorize

        try:
            authorize(env, {"initiator": initiator, "project_id": project_id})  # a member who lost access changes nothing
        except AccessDenied as exc:
            why = str(exc) or "access denied"
    if why is not None:
        out["refused"].append(why)
        return out
    target = confirmed_outcome(task, text, person, response_id or f"r-{_digest(task_id, person, text)}")
    if target is None:
        out["outcome"] = "unsure"
        return out
    try:
        written = save_outcome(env, work, project_id, task, target, by=tw.Actor(kind="agent", id=task.owner_team, team=task.owner_team),
                               initiator=initiator)
    except ValueError as exc:  # a cancelled, superseded or failed task takes no outcome
        out["refused"].append(str(exc))
        return out
    out.update(outcome=target.event, changed=written.changed, revised=[t.id for t in written.revised], stale=list(written.stale))
    return out


def correct(env: Any, project_id: str, task_id: str, *, kitchen_minutes: int, base_revision: Optional[int] = None,
            correction_id: Optional[str] = None, initiator: Optional[Row] = None, registry: Any = None) -> dict[str, Any]:
    """A correction after completion (the kitchen's minutes; 0 withdraws a delay): the specialist that owns ``task_id``
    estimates again on the corrected input and supersedes its report; every other task keeps its own snapshot.

    T2 (2026-11-23): the correction is applied against the report revision it names (``base_revision``; it may be left
    out only while the first report is current) and judged by ``team_work.accept_correction``
    (``tools.judge_correction``): ``apply`` makes the next revision; ``duplicate`` (already applied, the same
    ``correction_id`` again, or no change) adds nothing; ``stale_base`` (an older report than the current one) writes a
    skipped attempt and a "please reconcile" reply to the requester, never a new revision. One judged ``apply`` whose
    score another correction on the same base overtook is refused the same way. A resend after a crash finishes the
    unfinished attempt: one score, one report. Each judgment is recorded (``team.correction.applied`` or ``.refused``,
    and the ledger point ``team.correction_accept``). Returns the turn with its ``verdict`` (None when refused: no report
    yet, minutes outside 0-240, a base that names no report or none on a revised report, a reused correction id)."""
    from ..ledger import capture
    from . import tools

    work = _work(env)
    task = work.task(project_id, task_id)
    if task is None:
        raise KeyError(f"no task {task_id} in project {project_id}")
    try:
        judged = tools.judge_correction(work, project_id, task, int(kitchen_minutes), base_revision, correction_id)
    except ValueError as exc:
        return {"agent": task.owner_team, "task_id": task_id, "skipped": True, "steps": [], "refused": [str(exc)], "verdict": None}
    out: dict[str, Any] = {"agent": task.owner_team, "task_id": task_id, "skipped": True, "steps": [], "refused": []}
    turn = None
    if judged["verdict"] != "duplicate":
        turn = TeamTurn(env, work, project_id, task.owner_team, task_id=task_id, initiator=initiator,
                        extra={"mode": "correction", "correction": {k: judged[k] for k in ("verdict", "revision", "base", "minutes", "id")}})
        out = run_turn(turn, registry)
    out["verdict"] = judged["verdict"]
    if out["refused"]:
        return out
    view = work.task_view(project_id, task_id)
    made = next((r for r in view["reports"] if any(e.input_revision == judged["revision"] for e in r.evidence)), None)
    scored = next((a for a in view["actions"] if a.action == "score" and a.input_revision == judged["revision"]
                   and a.status in tw.DONE_ACTIONS), None)
    if judged["verdict"] == "apply" and made is None:
        now = max((r.revision for r in view["reports"] if r.status != "superseded"), default=judged["current"])
        if scored is None or now == judged["base"]:
            return out  # not scored (refused or left running): nothing applied, nothing to record yet
        # scored, but another correction on the same base reported first: it supersedes nothing, so it is refused as
        # late (never said to be applied) and the requester is told to reconcile
        judged = {**judged, "verdict": "stale_base", "current": now}
        out["verdict"] = "stale_base"
    if judged["verdict"] == "stale_base" and turn is not None:
        attempt = scored or next((a for a in view["actions"] if a.action == "score" and a.input_revision == judged["revision"]), None)
        if attempt is not None:  # once per attempt (the reply's key): the skipped attempt's reply is already there
            tools._reconcile(turn, task, judged, attempt)
    who = (initiator or {}).get("user_id")
    facts = {"task": task_id, "problem": task.problem_id, "by": str(who) if who else task.owner_team,
             "by_kind": "person" if who else "agent", "verdict": judged["verdict"], "base_revision": judged["base"],
             "request": judged["id"]}
    if judged["verdict"] == "apply":
        _correction_event("team.correction.applied", project_id, {**facts, "revision": made.revision, "report": made.id})
    else:
        _correction_event("team.correction.refused", project_id,
                          {**facts, "revision": judged["current"],
                           "reason": ("older_base" if judged["verdict"] == "stale_base"
                                      else "unchanged" if judged.get("unchanged") else "already_applied")})
    capture.team_correction_accepted(project_id, task_id, judged["verdict"], base_revision=judged["base"],
                                     current_revision=judged["current"])
    return out


def _correction_event(kind: str, project_id: str, payload: Row) -> None:
    """A correction's judgment in the workspace's record (best effort: the correction stands). Ids, revisions and codes only."""
    try:
        from .. import experience

        experience.record(kind, "person" if payload.get("by_kind") == "person" else "agent",
                          {k: v for k, v in payload.items() if v is not None}, project=project_id)
    except Exception:  # noqa: BLE001
        pass


OPEN_ACTIONS = {"sales": "request_sales_input", "marketing": "request_marketing_input"}  # Business's directory action per team


def _which(env: Any, thread: Mapping[str, Any], agent: str, tasks: list[tw.TeamTask], key: str) -> dict[str, Any]:
    """T3: words that could answer several open tasks answer none: the thread's agent asks which one is meant (once per
    message), naming each task, and the person answers with that task's reply button (``task_id``)."""
    from .store import team_label
    from .threads import threads_for

    lines = [f"- {t.id} ({team_label(t.owner_team)}): {str(t.question).strip().splitlines()[0][:140]}" for t in tasks]
    text = ("Your message could answer more than one open task, so I have not taken it as an answer to any. Which task do you "
            "mean? Answer with that task's reply button.\n" + "\n".join(lines))
    try:
        threads_for(env).post(str(thread["id"]), {"kind": "agent", "id": agent, "team": agent}, text,
                              handoff={"kind": "which_task", "task_ids": [t.id for t in tasks], "from_team": agent, "to_team": agent},
                              key=f"which:{key}")
    except LookupError:
        pass
    return {"agent": agent, "did": "which", "which": [t.id for t in tasks], "skipped": True, "steps": [], "refused": []}


def _pick(env: Any, thread: Mapping[str, Any], agent: str, candidates: list[tw.TeamTask], task_id: Optional[str],
          key: str) -> tuple[Optional[tw.TeamTask], Optional[dict[str, Any]]]:
    """The task a person's words answer: the one named (``task_id``: a reply button), else the only candidate; with several
    and none named, the agent asks which (and nothing is answered); a named task that does not wait here is refused."""
    if task_id:
        found = next((t for t in candidates if t.id == task_id), None)
        if found is None:
            return None, {"agent": agent, "did": "refused", "skipped": True, "steps": [],
                          "refused": [f"{task_id} does not wait for your answer in this thread"]}
        return found, None
    if len(candidates) > 1:
        return None, _which(env, thread, agent, candidates, key)
    return (candidates[0], None) if candidates else (None, None)


def _answered_with(work: Any, task: tw.TeamTask, person: str, response: str, agent: Optional[str] = None) -> bool:
    """Whether ``agent`` (default: the task's owner) already recorded the response ``response`` by ``person`` on it."""
    actor = agent or task.owner_team
    view = work.task_view(task.project_id, task.id)
    return view is not None and any(a.actor.id == actor and a.output_ref == f"response:{response}" and a.responded_by is not None
                                    and a.responded_by.id == person for a in view["actions"])


def from_thread(env: Any, thread: Mapping[str, Any], *, person: str, text: str, initiator: Optional[Row] = None,
                message_id: str = "", team_kind: Optional[str] = None, ask: Optional[str] = None,
                task_id: Optional[str] = None, response_id: Optional[str] = None, resend: bool = False,
                registry: Any = None) -> Optional[dict[str, Any]]:
    """A person posted in their team thread (the route calls this after saving the message): the team's agent takes it.

    In a Business thread (``team_kind`` the thread's team kind): with ``ask`` (a team agent's name), the message opens a
    case for that team (``open_case``; T3: each ask its own task, while another case from the thread is open too), reading
    the inputs the project declares for its teams (``team_inputs``). Without it, a message that answers an open
    clarification asked of this person in this thread is their answer (``person_replies``). In a Sales or Marketing thread
    (T3), a member's message answers the question its agent asked the team (``commercial.answer``). In both, ``task_id``
    (the reply button's) names the task answered; words that could answer several tasks answer none and the agent asks
    which; ``response_id`` is the answer's identity, so the same answer sent twice is recorded once (``resend``: the route
    saved no new message, so the words only resume the task that already recorded this response by this person, never
    another task opened since). Anything else is a
    message in the thread and nothing more. Returns the agent's turn (``did``: opened, answered, which, hint, refused),
    or None."""
    from .threads import is_home

    project_id, problem_id, thread_id = str(thread["project_id"]), str(thread["problem_id"]), str(thread["id"])
    key = safe_id(f"{thread_id}:{response_id or message_id or _digest_text(person, text)}")
    if is_home(thread):  # a person's Home conversation with the Lead: a message there answers the Lead's open question, if any
        from . import lead

        waiting = [t for t in lead.lead_tasks(env, project_id) if t.requester.thread_id == thread["id"]
                   and t.status == "waiting_for_person" and t.requester.member == person]
        if task_id:
            waiting = [t for t in waiting if t.id == task_id]
        if not waiting:
            return None
        return lead.answer(env, project_id, waiting[0].id, person=person, text=text, initiator=initiator, registry=registry)
    kind = str(team_kind or thread.get("team") or "")
    work = _work(env)
    if kind in ("sales", "marketing") and not ask:  # T3: a member answers the question its team's agent asked the team
        from . import commercial

        candidates = commercial.open_asks(env, thread, kind)
        if resend:
            candidates = [t for t in candidates if _answered_with(work, t, person, safe_id(response_id or ""))]
            if not candidates:
                return None
        found, said = _pick(env, thread, kind, candidates, task_id, key)
        if found is None:
            return said
        response = safe_id(response_id or f"m-{message_id}" if (response_id or message_id) else f"r-{_digest_text(found.id, person, text)}")
        return commercial.answer(env, thread, found, person=person, text=text, initiator=initiator, response_id=response,
                                 registry=registry)
    if kind != "business":
        return None
    rows = work.deliveries(project_id=project_id)
    if not ask:
        asked = [r for r in rows if r["kind"] == "clarification" and r.get("thread_id") == thread_id and r["problem_id"] == problem_id]
        candidates: list[tw.TeamTask] = []
        for row in reversed(asked):
            task = work.task(project_id, row["task_id"])
            if task is not None and task.status == "waiting_for_person" and all(c.id != task.id for c in candidates) and case_member(
                    TeamTurn(env, work, project_id, "business"), task)[0] == person:
                candidates.append(task)
        if resend:  # only the task that already took this answer from this person resumes (its records are idempotent)
            candidates = [t for t in (work.task(project_id, r["task_id"]) for r in asked) if t is not None
                          and _answered_with(work, t, person, safe_id(response_id or ""), agent="business")][:1]
            if not candidates:
                return None
        found, said = _pick(env, thread, "business", candidates, task_id, key)
        if found is None:
            return said
        out = person_replies(env, project_id, found.id, person=person, text=text, initiator=initiator,
                             response_id=safe_id(response_id) if response_id else None, registry=registry)
        return {**out, "did": "answered"}
    if ask not in tw.DIRECTORY or tw.DIRECTORY[ask].kind != "team_agent" or ask == "business":
        return {"agent": "business", "did": "refused", "refused": [f"{ask} is not a team the Business agent asks"], "steps": [],
                "skipped": False}
    project = getattr(env, "projects", None)
    inputs = tuple((project.get(project_id) or {}).get("team_inputs") or ()) if project is not None else ()
    # a client's response id names the ask only together with its thread and its person: never another person's task
    new = safe_id(f"T-{_digest_text(thread_id, person, response_id)}" if response_id else f"T-{message_id or _digest_text(thread_id, text)}")
    out = open_case(env, project_id, problem_id, task_id=new, question=text, thread_id=thread_id, member=person,
                    initiator=initiator, inputs=inputs, owner=ask, registry=registry)
    return {**out, "did": "opened", "task_id": new}


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
    and runs no step. T3: a task for a directory agent with no rules in this release is answered at once with an
    "unsupported" blocker naming who acts next (``_unsupported``), never left open looking delivered; any other row for it
    is its inbox. A refused step blocks the row (the requester sees that it could not be taken, never the refusal's words)."""
    from .delivery import PermanentFailure

    receipt: Row = {"row": row["id"], "agent": None, "skipped": True, "steps": []}
    base = row["to_team"].split(":")[0]
    unsupported = base not in BRAINS
    if unsupported and not (base in CONFIGS and row.get("kind") == "task"):
        return receipt
    work = _work(env)
    focus = _focus(work, row)
    if focus is None:
        raise PermanentFailure(f"no task {row['task_id']} in project {row['project_id']}")
    turn = TeamTurn(env, work, row["project_id"], row["to_team"], task_id=focus, initiator=row.get("initiator"), row=dict(row),
                    extra={"unsupported": True} if unsupported else {})
    out = run_turn(turn)
    if out["refused"]:
        raise PermanentFailure(f"{row['to_team']} refused a step: {out['refused'][0]}")
    if row["to_team"] == "home":
        from .lead import to_home_chat

        to_home_chat(env, row)  # the Lead's question or report, in the person's Home conversation (once per row)
    return {**receipt, "agent": row["to_team"], "task_id": focus, "skipped": out["skipped"], "steps": [s["tool"] for s in out["steps"]]}


__all__ = ["AGENT", "BRAINS", "CAPACITY", "CONFIGS", "CLARIFY_TARGET", "OPEN_ACTIONS", "ScriptedChoices", "TeamAgentConfig", "askable",
           "config_of", "confirm_outcome", "correct", "deliver", "from_thread", "job_description", "open_case", "person_replies", "preview_report", "run_turn"]
