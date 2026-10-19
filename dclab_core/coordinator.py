"""The coordinator's state machine (package 21.2): from the problem to a built project, one gate at a time.

The coordinator walks a problem through explicit states, in order:

    problem → map → data → links → meet → analysis → workflow → build → built

:func:`situation` reads the facts a host gathers (the problem card, its tables and their cards, the label join, the
meet check, what the roles handed over) and returns the first state that is not satisfied, what blocks it (each
blocker a question with its evidence), and what the coordinator does next. It never skips a state: a project is built
only when every state before ``build`` holds and the build's own checks pass, and the blocker it names is the one that
stopped it. A question the data can settle (is there a file of known cases?) waits for the data and links states
rather than being asked first.

A specialist role hands its work over as a :class:`Handoff` (the role, the state, the columns it names, its proposal).
:func:`gate` checks it against the data cards before anything uses it: a hand-off that names a column the tables do
not hold is refused whole, with the columns and the reason, and the role's rules stand in its place. Deterministic:
no model, no storage; the host records every verdict.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

STATES = ("problem", "map", "data", "links", "meet", "analysis", "workflow", "build", "built")
ACTIONS = ("answer", "draw_map", "confirm_map", "wait_for_data", "bring_table", "analyse", "plan_workflow", "build", "done", "stop")
DATA_QUESTIONS = ("data", "labels")  # problem-card questions a file can settle: carried to the links state


class Blocker(Contract):
    state: str = Field(max_length=20)
    code: str = Field(max_length=60)
    question: str = Field(max_length=600)
    field: Optional[str] = Field(None, max_length=60, description="The problem card's field, when the person is asked.")
    evidence: dict[str, Any] = Field(default_factory=dict)


class Facts(Contract):
    """What the host knows of one draft, gathered from its records (no table values)."""
    blocking_questions: list[dict[str, Any]] = Field(default_factory=list, description="The card's blocking questions: field, text.")
    pending_question: Optional[str] = Field(None, max_length=60, description="The field of the question the person has open.")
    open_field: Optional[str] = Field(None, max_length=60, description=(
        "A field of the host's own plan that changes the solution, never asked, with questions left: asked before the build."))
    files_ready: int = 0
    files_pending: int = 0
    files_failed: int = 0
    cases_only: bool = False
    offers: list[dict[str, Any]] = Field(default_factory=list, description="Tables of other drafts these cases label: draft_id, key, coverage.")
    cases_elsewhere: list[dict[str, Any]] = Field(default_factory=list, description=(
        "Files of known cases in other drafts that label this draft's table: draft_id, key, coverage (the walk continues there)."))
    labels_joined: bool = False
    map_size: Optional[Literal["single", "complex"]] = Field("single", description=(
        "24.2: the problem map's size, or None when it is not drawn yet. A host that draws no map leaves 'single': there is nothing to confirm."))
    map_parts: int = 0
    map_blockers: list[dict[str, Any]] = Field(default_factory=list, description="The map reviewer's blocking questions: code, question.")
    map_confirmed: bool = False
    meet: list[dict[str, Any]] = Field(default_factory=list, description="The meet check's blocking mismatches: code, question.")
    meet_checked: bool = True
    analysed: bool = False
    workflow_problems: list[str] = Field(default_factory=list)
    workflow_ready: bool = False
    build_refusals: list[dict[str, Any]] = Field(default_factory=list, description="The build's checks that refuse: rule, message.")
    built: bool = False


class Situation(Versioned):
    """Where a problem stands: the first state not yet satisfied, what blocks it, and the next action."""
    schema_version: Literal["dclab_core.coordinator_situation.v1"] = _version("coordinator_situation")
    state: str
    passed: list[str] = Field(default_factory=list)
    blockers: list[Blocker] = Field(default_factory=list)
    next: str
    says: str = Field(max_length=1000, description="The sentence the coordinator says: what it does, or which blocker stops it.")


CONTRACTS["coordinator_situation"] = Situation


FAMILY_NOT_RUN = "family not built yet"  # a build refusal's rule: the host runs no machine for this family


def situation(facts: Facts) -> Situation:
    """The first state the facts do not satisfy, in order, with its blockers and the next action."""
    passed: list[str] = []
    # known cases settle the outcome once they join their table (here, or from another draft): that question waits for the links
    data_fields = DATA_QUESTIONS + (("outcome",) if (facts.cases_only and not facts.labels_joined) or facts.cases_elsewhere else ())

    def at(state: str, nxt: str, says: str, blockers: Iterable[Blocker] = ()) -> Situation:
        return Situation(state=state, passed=list(passed), blockers=list(blockers), next=nxt, says=says[:1000])

    not_run = [r for r in facts.build_refusals if r.get("rule") == FAMILY_NOT_RUN]
    if not_run:  # 20.10: a family no machine runs is said once it is known, never after more questions about it
        return at("problem", "stop", f"Not built: {not_run[0].get('message')}",
                  [Blocker(state="problem", code=FAMILY_NOT_RUN, question=str(r.get("message") or "")[:600]) for r in not_run])
    questions = [q for q in facts.blocking_questions if q.get("field") not in data_fields]
    if questions:
        first = questions[0]
        b = [Blocker(state="problem", code="card_question", question=str(q.get("text") or "")[:600], field=q.get("field")) for q in questions]
        if facts.pending_question is not None:
            return at("problem", "answer", f"Waiting for the answer to: {first.get('text')}", b)
        return at("problem", "stop", f"The problem card is not ready: {first.get('text')}", b)
    passed.append("problem")
    if facts.map_size is None:
        return at("map", "draw_map", "Drawing the problem as parts: one problem, or several that depend on each other.")
    if facts.map_size == "complex":
        if facts.map_blockers:
            first = facts.map_blockers[0]
            return at("map", "stop", f"The map of the problem does not hold: {first.get('question')}",
                      [Blocker(state="map", code=str(m.get("code") or "map_unsound")[:60], question=str(m.get("question") or "")[:600]) for m in facts.map_blockers[:10]])
        if not facts.map_confirmed:
            return at("map", "confirm_map", f"The problem has {facts.map_parts} parts that depend on each other: confirm the map, or change it. "
                      "Until the programme of linked projects exists, the build below is for the problem the card names.",
                      [Blocker(state="map", code="map_unconfirmed", question="Confirm the map of the problem, or edit it.")])
    passed.append("map")
    if facts.files_failed:  # a file that could not be read is never built around (the cases alone became a project once)
        return at("data", "stop", f"{facts.files_failed} file(s) could not be read: replace or remove them before going on.",
                  [Blocker(state="data", code="files_failed", question="Replace or remove the file that could not be read.",
                           evidence={"failed": facts.files_failed})])
    if facts.files_pending:
        return at("data", "wait_for_data", f"{facts.files_pending} file(s) are still being read.",
                  [Blocker(state="data", code="files_pending", question="Wait for the files to be read.", evidence={"pending": facts.files_pending})])
    if not facts.files_ready:
        return at("data", "stop", "No table is ready: upload the data, or say there is none.",
                  [Blocker(state="data", code="no_data", question="Upload the table this problem is about.")])
    passed.append("data")
    data_questions = [q for q in facts.blocking_questions if q.get("field") in data_fields]
    if facts.cases_only and not facts.labels_joined:
        if facts.offers:
            best = max(facts.offers, key=lambda o: float(o.get("coverage") or 0))
            return at("links", "bring_table", f"These are known cases; bringing the table they label (key {best.get('key')}, "
                      f"{float(best.get('coverage') or 0):.0%} of the cases found in it).",
                      [Blocker(state="links", code="cases_without_table", question="Bring the table these cases label.", evidence=best)])
        return at("links", "stop", "This draft holds only known cases: add the table they belong to.",
                  [Blocker(state="links", code="cases_without_table", question="Upload the table these cases label.")])
    if data_questions and facts.cases_elsewhere:  # the known cases of this table were uploaded to another draft
        best = max(facts.cases_elsewhere, key=lambda o: float(o.get("coverage") or 0))
        return at("links", "stop", f"The known cases of this table are in draft {best.get('draft_id')} (key {best.get('key')}): "
                  "continue there, where they and this table are read together.",
                  [Blocker(state="links", code="cases_elsewhere", question="Continue in the draft that holds the known cases.", evidence=best)])
    if data_questions:
        first = data_questions[0]
        b = [Blocker(state="links", code="card_question", question=str(q.get("text") or "")[:600], field=q.get("field")) for q in data_questions]
        if facts.pending_question is not None:
            return at("links", "answer", f"Waiting for the answer to: {first.get('text')}", b)
        return at("links", "stop", f"The problem card is not ready: {first.get('text')}", b)
    passed.append("links")
    if facts.meet or not facts.meet_checked:
        b = [Blocker(state="meet", code=str(m.get("code") or "mismatch"), question=str(m.get("question") or "")[:600], field=m.get("field"),
                     evidence=dict(m.get("evidence") or {})) for m in facts.meet] or [
            Blocker(state="meet", code="not_checked", question="The problem could not be checked against the data.")]
        return at("meet", "answer" if facts.pending_question is not None else "stop", f"The problem and the data disagree: {b[0].question}", b)
    passed.append("meet")
    if not facts.analysed:
        return at("analysis", "analyse", "Analysing the data for the decisions (the data investigator's hand-off, checked).")
    passed.append("analysis")
    if facts.workflow_problems or not facts.workflow_ready:
        if facts.workflow_problems:
            return at("workflow", "stop", "The solution workflow does not hold: " + "; ".join(facts.workflow_problems[:3]),
                      [Blocker(state="workflow", code="workflow", question=p[:600]) for p in facts.workflow_problems[:5]])
        return at("workflow", "plan_workflow", "Drawing the solution workflow.")
    passed.append("workflow")
    if facts.built:
        passed.append("build")
        return at("built", "done", "The project is built.")
    if facts.pending_question is not None:  # never a build with a question open (19.9's own failure signal)
        return at("build", "answer", f"A question to the person is open ({facts.pending_question}): its answer comes before the build.",
                  [Blocker(state="build", code="question_open", question="Answer the open question first.", field=facts.pending_question)])
    if facts.open_field:  # a model's turns may end with the plan's own question unasked (live, 2026-10-19: the entity, so no grouped folds)
        return at("build", "ask", f"The question about {facts.open_field} was never asked: it is asked before the build.",
                  [Blocker(state="build", code="plan_open", question=f"Answer the question about {facts.open_field}.", field=facts.open_field)])
    if facts.build_refusals:
        r = facts.build_refusals[0]
        return at("build", "stop", f"Not built: {r.get('message')}",
                  [Blocker(state="build", code=str(r.get("rule") or "refused")[:60], question=str(r.get("message") or "")[:600])
                   for r in facts.build_refusals])
    return at("build", "build", "Every state holds: building the project.")


class Handoff(Contract):
    """What a role hands to the next state: the columns it names and its proposal (column names, never its words)."""
    role: str = Field(max_length=60)
    state: str = Field(max_length=20)
    columns: list[str] = Field(default_factory=list, max_length=500)
    proposal: dict[str, Any] = Field(default_factory=dict)
    malformed: list[str] = Field(default_factory=list, description="What in the answer had not the hand-off's shape.")


class HandoffVerdict(Contract):
    role: str
    state: str
    allowed: bool
    refused_columns: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def columns_of(data_card: Any) -> set[str]:
    """Every column of every table of a data card (a ``DataCard`` or its dict)."""
    tables = getattr(data_card, "tables", None)
    if tables is None and isinstance(data_card, Mapping):
        tables = data_card.get("tables") or []
    out: set[str] = set()
    for t in tables or []:
        cols = getattr(t, "columns", None) if not isinstance(t, Mapping) else t.get("columns")
        for c in cols or []:
            name = getattr(c, "name", None) if not isinstance(c, Mapping) else c.get("name")
            if isinstance(name, str):
                out.add(name)
    return out


def gate(handoff: Handoff, data_card: Any, allowed_states: Iterable[str] = STATES, extra_columns: Iterable[str] = ()) -> HandoffVerdict:
    """A hand-off is used only when every column it names is in the tables (``extra_columns``: those the host makes,
    such as the labels' ``is_case``) and its state is one of the machine's; otherwise it is refused whole."""
    tables = columns_of(data_card)
    known = tables | set(extra_columns)
    refused = sorted({c for c in handoff.columns if c not in known})
    reasons = list(handoff.malformed)
    if handoff.state not in set(allowed_states):
        reasons.append(f"{handoff.role} handed over to an unknown state {handoff.state!r}")
    if refused:
        reasons.append(f"{handoff.role} names column(s) the tables do not hold: {', '.join(refused[:10])}")
    if not tables:  # the host's own columns are no data card: nothing to check against, nothing taken on trust
        reasons.append("no data card to check the columns against")
    return HandoffVerdict(role=handoff.role, state=handoff.state, allowed=not reasons, refused_columns=refused, reasons=reasons)


def investigator_handoff(answer: Mapping[str, Any]) -> Handoff:
    """The data investigator's raw answer as a hand-off to the analysis: every column it names, in after or entities."""
    columns, proposal, malformed = [], {}, []
    if not isinstance(answer, Mapping):
        malformed.append("the answer is not an object")
    for key in ("after", "entities"):
        items = answer.get(key) if isinstance(answer, Mapping) else None
        if items is not None and not isinstance(items, list):
            malformed.append(f"{key} is not a list of {{column, why}}")
            continue
        kept = []
        for item in (items or [])[:50]:
            if not (isinstance(item, Mapping) and isinstance(item.get("column"), str)):
                malformed.append(f"an item of {key} is not a {{column, why}}")
                continue
            columns.append(item["column"])
            kept.append(item["column"])
        proposal[key] = kept
    return Handoff(role="data_investigator", state="analysis", columns=columns, proposal=proposal, malformed=malformed[:10])


__all__ = ["STATES", "ACTIONS", "Blocker", "Facts", "Situation", "Handoff", "HandoffVerdict", "situation", "gate", "columns_of",
           "investigator_handoff"]
