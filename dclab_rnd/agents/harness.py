"""One harness for every agent (package 21.1).

Every agent of DCLab runs through this module, and every step it takes reaches the session record (Part 8):

- a model-driven loop (the Home agent, the intern) runs on ``runtime.run``, which always traces: without a tracer of
  the caller's, it makes one, so no step goes unrecorded; every reply is a step, a reply without a tool call too;
- an agent that is not a loop (the Home agent's script, the intern's standard plan, the team, the research campaign,
  an MCP client's calls) runs inside :func:`session`, whose :meth:`Session.step` records each move the same way
  (``agent.step``, and ``tool.call`` when it calls a tool) and whose end is a ``run.recorded`` like a loop's;
- a role (the problem analyst, the planner, the critic... asked once through ``core_advisor.GatewayAdvisor``) is one
  step of the agent that asked it (:func:`role_step`).

``AGENTS`` lists every entry point, ``ROLES`` the roles, ``MODEL_USES`` the model purposes that are not agents (one
completion for one job: an embedding, a parser, a note). Every purpose of ``models/settings.py`` is in exactly one of
them, and a test walks the package so that a new model call or tool call outside them fails
(``tests/test_agent_harness.py``).
"""

from __future__ import annotations

import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional

from .traces import Tracer


@dataclass(frozen=True)
class Agent:
    """An entry point: where it starts, how it runs (``loop``: ``runtime.run``; ``session``: :func:`session`), and the
    model purposes it sends."""
    entry: str
    runs: str
    purposes: tuple[str, ...] = ()
    note: str = ""


AGENTS: dict[str, Agent] = {
    "home": Agent("dclab_rnd.draft.chat:HomeAgent.model_turn", "loop", ("home_agent",), "the Home agent with a model"),
    "home.project": Agent("dclab_rnd.draft.orchestrator:Orchestrator.model_turn", "loop", ("home_orchestrator",),
                          "the Home orchestrator on a built project's chat: plans the turn and hands it to its specialists"),
    "home.project.rules": Agent("dclab_rnd.draft.orchestrator:Orchestrator.rules_turn", "session", (), "the Home orchestrator's rules, without a model or when it fails"),
    "home.script": Agent("dclab_rnd.draft.chat:HomeAgent.script_turn", "session", (), "the Home agent without a model, or when it fails"),
    "intern": Agent("dclab_rnd.intern.loop:Intern._llm_loop", "loop", ("intern",), "the intern with a model"),
    "case": Agent("dclab_rnd.agents.case_agent:turn", "loop", ("case_agent",), "37.3: the Phase 13 lead agent, one case around the case file; Home's default since 37.11 (DCLAB_CASE_AGENT=0 turns it off)"),
    "workspace_chat": Agent("dclab_rnd.workspace_chat:ask", "loop", ("workspace_chat",), "30.21: a question about the workspace's tables, every number computed"),
    "intern.standard": Agent("dclab_rnd.intern.loop:Intern._plan_run", "session", (), "the intern's standard plan, and its follow-ups"),
    "team": Agent("dclab_rnd.studio.team:run", "session", (), "the team of roles on a project"),
    "campaign": Agent("dclab_rnd.agentic.engine:run_research", "session", ("campaign",), "the research campaign's phases and experiments"),
    "mcp": Agent("dclab_rnd.mcp_server:build_server", "session", (), "the calls of an MCP client (Hugging Face Chat UI, ML Intern)"),
    "orchestra": Agent("dclab_rnd.orchestra.spike:Spike.run", "session", (), "28.1: the orchestration layer's graph on a draft; each node a step, each pause a stopped run"),
    "orchestra.walk": Agent("dclab_rnd.orchestra.walk:Walk.run", "session", (), "28.3: the coordinator's state machine as a graph on a draft; each action a step, each pause a stopped run"),
    "orchestra.solution": Agent("dclab_rnd.orchestra.solution:SolutionTeam.run", "session", (), "28.5: a programme solved by a team on its project: parts routed in parallel, the auditor before the holdout, each node a step"),
    "tutor": Agent("dclab_rnd.store.tutor:teach", "session", (), "32.14: the students' agent: a student fitted and measured for every point the store's sets allow, never promoted"),
    "conductor": Agent("dclab_rnd.store.conductor:conduct", "session", (), "32.2: the store's conductor: the keepers, then the later tiers, each a step"),
    "coordinator": Agent("dclab_rnd.draft.coordinator:Coordinator.run", "session", (), "the lead agent: problem to built project, gate by gate"),
    "learning": Agent("dclab_rnd.learning.scoreboard:attempt", "session", (), "H0: one challenge pack's attempt, each arm solved from the "
                      "public folder and graded against the key, a step with its whole grade"),
    "twin": Agent("dclab_rnd.twin:run", "session", ("ai_first",), "a twin run (25.2, 25.3): the current system and Claude on one problem, two lanes at once, "
                                                       "each a workspace folder; Claude serves every purpose of its lane through the lane's own client"),
}

ROLES: tuple[str, ...] = ("problem_analyst", "conversation_reader", "brief_checker", "solution_planner", "reporter", "critic",
                          "feature_engineer", "data_investigator", "decomposer", "map_reviewer", "data_need", "solution_router",
                          "understanding_critic", "solution_challenger", "debate_judge", "reflector", "task_decomposer")

MODEL_USES: dict[str, str] = {  # purpose -> why it is not an agent: one completion for one job, recorded as model.request
    "embedding": "vectors for evidence search",
    "memory_embedding": "vectors for memory search (35.5, a local model only)",
    "parse_pattern": "how a text file's lines split into columns",
    "synthetic_schema": "the columns of a simulated table",
    "evidence_answer": "an answer from cited evidence records",
    "stage_notes": "the explanation of one stage's record",
    "store_curator": "32.9: where a batch of the store's records goes, in shadow beside the rule curators",
    "reasoning_teacher": "32.10: the reasoning behind one record's answer, checked against the record before it is kept",
    "lesson_proposal": "a lesson proposed from a finished project",
    "project_answer": "an answer about one project",
    "notebook_review": "a notebook's methodology review",
    "notebook_cell": "the next notebook cell for a person's request, checked before it is shown and run only once accepted",
    "leakage_review": "a second reading of leakage suspects",
    "campaign_review": "the review of a finished research campaign",
    "decision_answer": "one decision answered from its state, beside the rule (22.5)",
    "decision_labeler": "one drawn decision labelled from its blind view, kept apart from a person's labels",
    "case_specialist": "37.7: a specialist the lead agent sends, one call with the case: its findings go into the case file",
    "route_check": "2026-11-01: one short reading of which kind of help the person needs, checking the lead agent's start once",
    "exam_plain": "37.0: the conversation exam's Claude-alone arm, one turn of a simulated conversation (an evaluation arm, not a DCLab agent)",
    "exam_person": "37.0: the simulated person of the conversation exam, one reply from a written person's hidden truth",
    "exam_judge": "37.0: the conversation exam's judge, one blind judgment of a whole conversation",
    "home_understand": "Home's understanding conversation: one turn says back what it understood and asks one open question, or writes the options code checks",
    "routing_labeler": "P8: one Home routing decision labelled from the person's words, blind to every answerer's reading, kept apart as silver",
    "solver_answer": "one solver question answered on the benchmark of verified invented problems (27.6), beside their planted truth",
    "metric_question": "a question about a number turned into a tested template and its parameters, validated, beside the rules' reading (30.20)",
    "home_query": "H6: a query plan for a question about the person's own table, from column summaries; code validates and runs it",
    "home_business": "H11: a decision brief reworded in plain words, kept only through the citation check, the code's caveat always added",
    "home_intent": "H2: which kind of help a request needs (a prediction, an analysis, ...), read only where the rules find no clear winner",
    "journey_writer": "S8 B1: invented people for the understanding students' homework, on one replica kind (the truth is the code's, not the writer's)",
    "journey_checker": "S8 B1: whether each written person fits its kind's problem; one that does not is kept apart",
}


def purposes() -> dict[str, str]:
    """Every model purpose by what sends it: an agent's name, ``role``, or ``model_use``."""
    out = {p: name for name, agent in AGENTS.items() for p in agent.purposes}
    out.update({p: "role" for p in ROLES})
    out.update({p: "model_use" for p in MODEL_USES})
    return out


def new_run(agent: str) -> str:
    return f"{agent.split('.')[0]}-" + secrets.token_hex(6)


def tracer(agent: str, run_id: Optional[str] = None, store: Any = None, state: Optional[Callable[[], str]] = None) -> Tracer:
    """A tracer for one run; without a store its rows go to the session record only."""
    return Tracer(store, run_id or new_run(agent), agent, state)


@dataclass
class Session:
    """A run of an agent that is not a loop: each move is a :meth:`step`."""
    agent: str
    trace: Tracer
    steps: int = 0
    failed: int = 0
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    status: Optional[str] = None  # how the run ended when it returned normally but did not complete ("stopped": a declined gate)
    moves: list[tuple[Optional[str], Optional[str]]] = field(default_factory=list)  # D2: (tool, verdict) of every step, for the run's tool counts

    def used(self, tokens: Optional[dict[str, Any]]) -> None:
        """Tokens a model request of this run used (a client outside the gateway's loop reports them here)."""
        for key in self.usage:
            self.usage[key] += int((tokens or {}).get(key) or 0)

    @property
    def run_id(self) -> str:
        return self.trace.run_id

    def step(self, tool: Optional[str], arguments: Optional[dict[str, Any]] = None, result: Any = None, seconds: float = 0.0,
             text: Optional[str] = None, verdict: Optional[str] = None, n: Optional[int] = None,
             model_request: Optional[str] = None) -> dict[str, Any]:
        """One move: a tool it called (``tool``), or what it said without one (``tool`` None, ``text``); ``model_request``
        the request that produced it, when one did."""
        row = self.trace(tool, arguments, result, seconds, verdict=verdict, n=n, text=text, model_request=model_request)
        self.steps += 1
        self.failed += int(row.get("verdict") not in ("ok", None))
        self.moves.append((tool, row.get("verdict")))
        return row

    @contextmanager
    def timed(self, tool: str, arguments: Optional[dict[str, Any]] = None) -> Iterator[dict[str, Any]]:
        """Run a move and record it: ``with s.timed("ask", {...}) as out: out["result"] = ...``; an error is a step too."""
        out: dict[str, Any] = {"result": None}
        self.trace.before()
        clock = time.monotonic()
        try:
            yield out
        except BaseException as exc:
            if not _pause(exc):  # a graph pausing for the person (28.3) is no error: the step is recorded when it resumes
                self.step(tool, arguments, {"error": type(exc).__name__}, time.monotonic() - clock)
            raise
        self.step(tool, arguments, out["result"], time.monotonic() - clock)


def _pause(exc: BaseException) -> bool:
    """LangGraph's interrupt (``GraphBubbleUp`` and its kind), known by name: the harness imports no framework."""
    return any(c.__name__ == "GraphBubbleUp" for c in type(exc).__mro__)


@contextmanager
def session(agent: str, run_id: Optional[str] = None, store: Any = None, state: Optional[Callable[[], str]] = None,
            trace: Optional[Tracer] = None, params: Optional[dict[str, Any]] = None) -> Iterator[Session]:
    """Run ``agent`` (a name in ``AGENTS``) outside a model loop. Its steps are recorded as it takes them; its end is
    one ``run.recorded`` with how it ended (completed, failed, or stopped by a person)."""
    if agent not in AGENTS:
        raise KeyError(f"unknown agent {agent!r}: add it to dclab_rnd.agents.harness.AGENTS")
    from .. import experience

    current = Session(agent, trace or tracer(agent, run_id, store, state))
    began = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    clock, error, interrupted = time.monotonic(), None, False
    conversation = current.run_id if agent.split(".")[0] in ("home", "intern") else None
    from .. import observe

    with observe.span(name=f"agent_run {agent}"[:120], **{"dclab.agent": agent, "dclab.run": current.run_id,  # B4: a session in Langfuse
                                                          "session.id": conversation or current.run_id, "langfuse.trace.name": agent}):  # 23.11: the run is a span; its steps, its end and what it asks are spans or events of one trace
        try:
            with experience.bind(run=current.run_id, conversation=conversation):  # what it asks (a role, a model) joins its run
                yield current
        except BaseException as exc:
            error = f"{type(exc).__name__}: {str(exc)[:500]}"
            interrupted = not isinstance(exc, Exception)
            raise
        finally:
            from ..experience import hooks

            hooks.from_agent_session(agent, current, began, time.monotonic() - clock, error, interrupted, params)


def step(agent: str, run_id: str, tool: Optional[str], arguments: Optional[dict[str, Any]] = None, result: Any = None,
         text: Optional[str] = None) -> None:
    """One move of an agent between runs (the Home agent's script asking its next question): a step of ``run_id``."""
    if agent not in AGENTS:
        raise KeyError(f"unknown agent {agent!r}: add it to dclab_rnd.agents.harness.AGENTS")
    tracer(agent, run_id)(tool, arguments, result, 0.0, text=text)


def role_step(purpose: str, seconds: float, verdict: str, model_request: Optional[str] = None, **ids: Any) -> None:
    """A role answered (or failed to) once: a step of whatever run asked it (the run bound around it). Its reply stays
    in its model request (kept only where the workspace keeps model text): a role's JSON may name the table's columns.
    ``model_request``: the request this call made, when it made one (never an earlier one)."""
    from .. import experience

    if purpose not in ROLES:
        raise KeyError(f"unknown role {purpose!r}: add it to dclab_rnd.agents.harness.ROLES")
    experience.record("agent.step", "agent", {"agent": f"role.{purpose}", "n": None, "state": None, "reply": None, "tool": None,
                                              "verdict": verdict, "seconds": round(float(seconds), 3)}, model_request=model_request, **ids)


__all__ = ["AGENTS", "ROLES", "MODEL_USES", "Agent", "Session", "purposes", "role_step", "session", "step", "tracer"]
