"""The orchestrator (package 28.3): the coordinator's state machine as a LangGraph graph, on Home first.

One node situates the draft (``Coordinator.where``: deterministic, the only thing that chooses the next node), one node per
action does what the coordinator does (``Coordinator.act``, unchanged: enhance, never replace), and the person is reached
through a small adapter whose ``ask`` and ``approve`` pause the graph with a typed interrupt (a question, an approval) and
return the answer when the walk resumes, in this process or another (the checkpoint holds the walk). A build is approved
first (the "build it" of 25.13). Every node advances the team's board (28.2): an experience event per hand-off, the whole
board kept. The coordinator's loop (``Coordinator.run``) stays the path with no framework; ``drive`` walks the graph with
a scripted or written person exactly as the loop does, so the two can be measured on the same journeys.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import time
from datetime import datetime, timezone
import operator
from typing import Annotated, Any, Optional, TypedDict

from ..agents import harness
from . import NothingToResume, thread_id

ACTS = ("answer", "draw_map", "confirm_map", "bring_table", "analyse", "plan_workflow", "build")
_SESSION: contextvars.ContextVar[Any] = contextvars.ContextVar("orchestra_walk_session")
_PAUSES: contextvars.ContextVar[list] = contextvars.ContextVar("orchestra_walk_pauses")  # the pauses a node closed, in order
AGAIN = 5  # at most this many times a node asks again because what is open changed while the walk waited


def _key(*parts: Any) -> str:
    """What a pause asks, as a short key: an answer resumes only the pause it was given to (the review of 28.3)."""
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _closed(item: dict[str, Any]) -> None:
    pauses = _PAUSES.get(None)
    if pauses is not None:
        pauses.append(item)


class WalkState(TypedDict, total=False):
    draft_id: str
    board: dict[str, Any]
    steps: list[dict[str, Any]]
    seen: dict[str, int]
    last: Optional[str]
    next: str
    state: str
    says: str
    situation: dict[str, Any]
    project_id: Optional[str]
    started: str
    runs: Annotated[list[str], operator.add]  # every session of the walk, appended (the review of 28.8: a stale read lost one)


def _answered(draft: dict[str, Any]) -> set[str]:
    """The fields the person answered a question about (a model's doubt about them is not asked again); a question they
    declined is not one: its concern stands, for the person to decide on at the critic's gate."""
    return {str(q.get("field")) for q in draft.get("questions") or [] if q.get("answered") and not q.get("declined")}


IDENTIFIER = __import__("re").compile(r"`([^`]{1,80})`|\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b")


def _grounded(items: list[dict[str, Any]], rules: set[str], card: Any, columns: list[Any], mismatches: list[Any]) -> tuple[list, list]:
    """A model's concerns kept only when grounded in what the critic read (the review of 28.4, ``cited.py``'s rule): every
    identifier it writes is a column or a card field, every number is in the evidence. The rules' own concerns are kept."""
    from dclab_core.understanding_critic import FIELDS

    from ..cited import numbers

    names = {str(c.get("name")) for c in columns} | set(FIELDS) | {"prediction_moment", "data_plan"}
    evidence = json.dumps({"card": card, "columns": columns, "mismatches": mismatches}, default=str)
    stated = numbers(evidence, source=True)
    kept, refused = [], []
    for c in items:
        if json.dumps(c, sort_keys=True) in rules:
            kept.append(c)
            continue
        text = f"{c.get('question') or ''} {c.get('why') or ''}"
        unknown = sorted({a or b for a, b in IDENTIFIER.findall(text)} - names)
        loose = sorted(numbers(text) - stated)
        if unknown or loose:
            refused.append({**c, "refused": f"names {unknown} not in the tables" if unknown else f"numbers {loose} not in the evidence"})
        else:
            kept.append(c)
    return kept, refused


class Asker:
    """The person, as the coordinator calls them, made into the graph's pauses."""

    who = "person"

    @staticmethod
    def _pause(payload: dict[str, Any]) -> Any:
        """Pause with ``payload`` and return the answer given to it; an answer given to another pause (what was open changed
        while the walk waited: the person answered in the chat, the map was redrawn) asks again, never taken for this one."""
        from langgraph.types import interrupt

        payload = {**payload, "key": _key(payload.get("kind"), payload.get("field") or payload.get("gate"), payload.get("text"), payload.get("summary"))}
        for _ in range(AGAIN):
            said = interrupt(payload)
            if isinstance(said, dict) and "key" in said:
                if said.get("key") != payload["key"]:
                    continue  # the next interrupt of this node: the walk pauses on what is open now
                return said.get("answer")
            return said
        return None

    def ask(self, field: Optional[str], question: str) -> Optional[str]:
        said = self._pause({"kind": "question", "field": field, "text": question})
        out = None if said is None or not str(said).strip() else str(said).strip()
        if out is not None:
            _closed({"kind": "question", "field": field, "text": question, "status": "answered"})
        return out

    def approve(self, gate: str, summary: Any) -> bool:
        kind = "confirmation" if gate == "map" else "approval"  # the three pauses: a question, a confirmation, an approval
        answer = self._pause({"kind": kind, "gate": gate, "summary": summary})
        ok = bool(answer is True or (isinstance(answer, str) and answer.strip().lower() in ("yes", "y", "ok", "approve", "approved", "build", "confirm")))
        _closed({"kind": kind, "gate": gate, "status": "answered" if ok else "declined"})
        return ok


class Walk:
    TEAM = "orchestra"

    def __init__(self, coordinator: Any, checkpointer: Any, home: Any = None, max_steps: Optional[int] = None,
                 critic: bool = True, critic_advisor: Any = None, max_eur: Optional[float] = None, node_eur: Optional[float] = None):
        self.c, self.checkpointer, self.home = coordinator, checkpointer, home
        self.critic_on, self.critic_advisor = critic, critic_advisor  # 28.4: the understanding critic (its rules, or a model)
        self.max_steps = max_steps or getattr(coordinator, "max_steps", 40)
        self.max_eur, self.node_eur = max_eur, node_eur  # 28.8: the walk's euro budget across its resumes, and one node's
        self.graph = None
        self.asker = Asker()

    def _cap_node(self) -> None:
        """28.8: a node's budget, set before it acts: the Home agent's model and the critic's may spend at most ``node_eur`` more
        than they had when the node began (the gateway refuses past it and the node falls back to its rules)."""
        if self.node_eur is None:
            return
        client = getattr(getattr(self.c, "agent", None), "client", None)
        if client is not None and hasattr(client, "run_limit_eur"):
            client.run_limit_eur = float(getattr(client, "spent_eur", 0.0) or 0.0) + self.node_eur
        advisor = self.critic_advisor
        if advisor is not None and hasattr(advisor, "cap_eur") and callable(getattr(advisor, "spent_eur", None)):
            advisor.cap_eur = float(advisor.spent_eur()) + self.node_eur

    # ------------------------------------------------------------------------------------------------ the board
    def _board(self, state: WalkState) -> Any:
        from dclab_core.orchestra import Board

        return Board.model_validate(state["board"])

    def _problem(self, draft_id: str) -> dict[str, Optional[str]]:
        u = (self.c.drafts.get(draft_id).get("understanding") or {})
        keep = {"family": u.get("family"), "outcome": u.get("target"), "moment": u.get("prediction_moment"), "unit": u.get("entity"),
                "action": u.get("action"), "labels": u.get("labels")}
        return {k: (" ".join(str(v).split())[:200] if v else None) for k, v in keep.items()}  # the draft keeps the whole words

    def _advance(self, state: WalkState, node: str, what: str, board: Any = None) -> dict[str, Any]:
        """The board after ``node``: the draft as it is now (the review of 28.3: the board after a build says built)."""
        from . import board as B

        now = self.c.where(state["draft_id"])
        draft = self.c.drafts.get(state["draft_id"])
        assets = draft.get("assets") or []
        return B.advance(board if board is not None else self._board(state), node, what, self.TEAM, home=self.home, walk=_walk_id(state), state=str(now.state)[:40],
                         blockers=[str(b.code)[:80] for b in now.blockers][:40], problem=self._problem(state["draft_id"]),
                         open_questions=[str(b.field)[:80] for b in now.blockers if getattr(b, "field", None)][:40],
                         data={"tables": len(assets), "ready": sum(a.get("status") == "ready" for a in assets)},
                         spent_eur=_spent(state["draft_id"], state.get("runs") or [])).model_dump(mode="json")  # 28.8: live, not 0 until the end

    def _close_pauses(self, state: WalkState, node: str) -> Any:
        """The board with the pauses this node closed (answered or declined): each raises its version, each an event."""
        from . import board as B

        board = self._board(state)
        for p in _PAUSES.get(None) or []:
            board = B.interrupt(board, p["kind"], node, self.TEAM, field=p.get("field"), gate=p.get("gate"), text=str(p.get("text") or ""),
                                status=p["status"], home=self.home)
        return board

    # ------------------------------------------------------------------------------------------------ the nodes
    def situate(self, state: WalkState) -> dict[str, Any]:
        now = self.c.where(state["draft_id"])
        nxt = now.next
        if self.critic_on and nxt in ("plan_workflow", "build"):
            if self.c.agent.pending(self.c.drafts.get(state["draft_id"])) is not None:
                nxt = "answer"  # a question the critic sent back is open: its answer comes before the workflow is drawn again
            elif not self._critic_done(state["draft_id"]):
                nxt = "critic"  # 28.4: before a solution is drawn, the understanding is read again
        spent = _spent(state["draft_id"], state.get("runs") or []) if self.max_eur is not None else 0.0
        over_node = next((s for s in state.get("steps") or [] if self.node_eur is not None and float(s.get("cost_eur") or 0.0) > self.node_eur), None)
        if len(state.get("steps") or []) >= self.max_steps and nxt not in ("done", "stop", "wait_for_data"):
            nxt, says = "stop", "The coordinator's step budget was used."
        elif self.max_eur is not None and spent >= self.max_eur and nxt not in ("done", "stop", "wait_for_data"):
            nxt, says = "stop", f"The team's euro budget was used: €{spent:.2f} of €{self.max_eur:.2f}."  # 28.8
        elif over_node is not None and nxt not in ("done", "stop", "wait_for_data"):
            nxt, says = "stop", (f"The {over_node.get('action')} node spent €{float(over_node['cost_eur']):.2f}, over its budget of "
                                 f"€{self.node_eur:.2f}: the team stops for the person to decide.")
        else:
            says = now.says
        from ..ledger import capture

        draft = self.c.drafts.get(state["draft_id"])
        capture.team_node(("draft", state["draft_id"]), "orchestra", now.state, now.next, nxt, blockers=len(now.blockers),  # 29.6: every step a decision
                          open_questions=sum(1 for q in draft.get("questions") or [] if not q.get("answered")),
                          pending=self.c.agent.pending(draft) is not None, critic_passed=self._critic_done(state["draft_id"]) if self.critic_on else None,
                          step=len(state.get("steps") or []))
        return {"next": nxt, "state": now.state, "says": says, "situation": json.loads(now.model_dump_json())}

    def _act(self, action: str):
        def node(state: WalkState) -> dict[str, Any]:
            from dclab_core import coordinator as machine

            draft_id = state["draft_id"]
            now = machine.Situation.model_validate(state["situation"])
            token = _PAUSES.set([])
            try:
                if action == "answer" and self.c.agent.pending(self.c.drafts.get(draft_id)) is None:
                    return {"next": "situate"}  # answered elsewhere while the walk waited: situated again, nothing to ask
                if action == "confirm_map" and self.c.where(draft_id).next != "confirm_map":
                    return {"next": "situate"}  # the map was confirmed (or changed) elsewhere while the walk waited (the second review of 28.8)
                if action == "answer" and (self.c.agent.pending(self.c.drafts.get(draft_id)) or {}).get("source") == "critic":
                    return self._critic_answer(state, now)  # 28.7: a critic's question is advisory: no answer is a decline, never a stop
                if action == "build":
                    approved = self.asker.approve("build", {"state": now.state, "passed": list(now.passed)})
                    _SESSION.get().step("approve", {"gate": "build"}, {"approved": approved})
                    if not approved:
                        steps = [*state.get("steps", []), {"state": now.state, "action": action, "says": now.says, "seconds": 0.0,
                                                           "result": {"stopped": "build not approved"}}]
                        return {"next": "stop", "says": "Waiting for the person's approval to build.", "steps": steps,
                                "board": self._advance(state, "build", "declined", self._close_pauses(state, "build"))}
                    again = self.c.where(draft_id)  # the review of 28.3: the draft may have changed while the approval waited
                    if again.next != "build":
                        return {"next": "situate", "board": self._advance(state, "build", "situated_again", self._close_pauses(state, "build"))}
                    now = again
                clock = time.monotonic()
                self._cap_node()
                before = _spent(draft_id, state.get("runs") or [])
                with _SESSION.get().timed(action, {"state": now.state}) as out:
                    out["result"] = result = self.c.act(draft_id, now, self.asker)  # the coordinator's own action
                seconds = round(time.monotonic() - clock, 3)
                cost = round(max(0.0, _spent(draft_id, state.get("runs") or []) - before), 6)  # 28.8: what this node's model requests cost
                board = self._close_pauses(state, action)
            finally:
                _PAUSES.reset(token)
            steps = [*state.get("steps", []), {"state": now.state, "action": action, "says": now.says, "seconds": seconds, "result": result,
                                               "cost_eur": cost}]
            update: dict[str, Any] = {"steps": steps, "board": self._advance(state, action, action, board),
                                      "project_id": result.get("project_id") or state.get("project_id")}
            if result.get("stopped"):
                return {**update, "next": "stop", "says": result["stopped"]}
            done = json.dumps([now.state, action, json.dumps(result, sort_keys=True, default=str)])
            seen = dict(state.get("seen") or {})
            seen[done] = seen.get(done, 0) + 1
            if done == state.get("last") or seen[done] >= 3:  # the same outcome again, or a loop of questions: no progress
                return {**update, "seen": seen, "last": done, "next": "stop", "says": f"No progress at {now.state}: {now.says}"}
            return {**update, "seen": seen, "last": done, "next": "situate"}
        return node

    def _critic_answer(self, state: WalkState, now: Any) -> dict[str, Any]:
        """A question the critic sent back, put to the person. An answer is taken as the Home agent takes any (``reply``); no
        answer (they do not know, or skip it) declines it: recorded on the question, the concern left standing for the critic's
        gate, where the person decides whether to build on (the measurement of 28.7: the walk stopped on six of fifteen people
        who could not say what one row is)."""
        draft_id = state["draft_id"]
        q = self.c.agent.pending(self.c.drafts.get(draft_id)) or {}
        clock = time.monotonic()
        said = self.asker.ask(q.get("field"), q.get("text") or "")
        if said:
            before = (self.c.drafts.get(draft_id).get("understanding") or {}).get(q.get("field"))
            self.c.agent.reply(draft_id, said)
            after = (self.c.drafts.get(draft_id).get("understanding") or {}).get(q.get("field"))
            _label_send_back(draft_id, str(q.get("field")), before != after)  # 29.6: the send-back was right when the answer changed the card
            result = {"answered": q.get("field"), "source": "critic"}
        else:
            def declined(d: dict[str, Any]) -> None:
                for item in d.get("questions") or []:
                    if item.get("id") == q.get("id"):
                        item["answered"], item["declined"] = "(no answer)", True
            self.c.drafts.update(draft_id, declined)
            result = {"declined": q.get("field"), "source": "critic"}
        _SESSION.get().step("answer", {"state": now.state, "field": q.get("field"), "source": "critic"}, result, time.monotonic() - clock)
        board = self._close_pauses(state, "answer")
        steps = [*state.get("steps", []), {"state": now.state, "action": "answer", "says": now.says, "seconds": round(time.monotonic() - clock, 3), "result": result}]
        return {"steps": steps, "board": self._advance(state, "answer", "declined" if not said else "answer", board), "next": "situate"}

    # ------------------------------------------------------------------------------------------------ 28.4: the critic
    def _critic_done(self, draft_id: str) -> bool:
        own = ((self.c.drafts.get(draft_id).get("coordinator") or {}).get("critic") or {})
        return bool(own.get("passed"))

    def critic(self, state: WalkState) -> dict[str, Any]:
        """The understanding critic reads the card against the data card. Its first new concern is a question the interviewer
        asks next (one a round, as the Home agent asks: one at a time; a field never twice; at most ROUNDS rounds); a model's
        concern is kept only when every name and number in it is in what the critic read. When concerns remain and none can be
        sent back, the person decides (``critic_gate``): build on, or stop with the concerns shown. Every round is kept."""
        from dclab_core import coordinator as machine
        from dclab_core.conversation import summary
        from dclab_core.understanding_critic import ROUNDS, concerns, critique

        from ..draft.card import card_of, meeting

        draft_id = state["draft_id"]
        draft = self.c.drafts.get(draft_id)
        own = dict(((draft.get("coordinator") or {}).get("critic") or {}))
        card = summary(card_of(draft))
        columns = [c for t in ((draft.get("data_card") or {}).get("tables") or []) for c in (t.get("columns") or [])]
        met = meeting(draft, self.c.drafts)
        mismatches = [{"code": m.code, "question": m.question} for m in (met.mismatches if met else []) if m.blocking]
        rounds = int(own.get("rounds") or 0)
        asked = set(own.get("asked") or [])
        self._cap_node()
        before = _spent(draft_id, state.get("runs") or [])
        with _SESSION.get().timed("critic", {"round": rounds + 1}) as out:
            result = critique(self.critic_advisor, card, columns, mismatches)
            rules = {json.dumps(c, sort_keys=True) for c in concerns(card, columns, mismatches)}
            kept, refused = _grounded(result.output, rules, card, columns, mismatches)
            answered = _answered(draft)
            for c in [c for c in kept if json.dumps(c, sort_keys=True) not in rules and c["field"] in answered]:
                kept.remove(c)  # nothing asked twice: a model's doubt about what the person said is kept, not asked again
                refused.append({**c, "refused": "the person answered it already"})
            new = [c for c in kept if c["field"] not in asked]
            sent = None
            if rounds < ROUNDS:
                for c in new:  # the first that can be asked; the rest wait for the next round
                    if self.c.agent.ask(draft_id, c["field"], c["question"], why=c.get("why") or "", source="critic") is not None:
                        sent = c["field"]
                        break
            out["result"] = {"concerns": len(kept), "refused": len(refused), "sent_back": sent, "source": result.source}
        from ..ledger import capture

        for c in kept:  # 29.6: every concern the critic weighed, sent back or not, with why
            src = "rules" if json.dumps(c, sort_keys=True) in rules else "model"
            why = "sent back" if c["field"] == sent else "asked before" if c["field"] in asked else "the rounds are spent" if rounds >= ROUNDS else "one a round"
            capture.critic_weighed(draft_id, c["field"], src, c["field"] == sent, why, rounds)
        for c in refused:
            capture.critic_weighed(draft_id, c["field"], "model", False, str(c.get("refused") or "refused"), rounds)
        waiting = self.c.agent.pending(self.c.drafts.get(draft_id)) is not None
        gate = sent is None and bool(kept) and not waiting  # concerns stand and none can be sent back: the person decides
        history = [*(own.get("history") or []), {"round": rounds + 1, "concerns": kept, "refused": refused, "sent_back": sent,
                                                  "source": result.source, "reason": result.reason}]
        own.update(rounds=rounds + (1 if sent else 0), asked=sorted(asked | ({sent} if sent else set())),
                   passed=not sent and not kept and not waiting, awaiting=[{"field": c["field"], "question": c["question"]} for c in kept] if gate else None,
                   history=history[-12:], last=history[-1])
        self._keep_critic(draft_id, own)
        now = machine.Situation.model_validate(state["situation"])
        steps = [*state.get("steps", []), {"state": now.state, "action": "critic", "says": now.says, "seconds": 0.0,
                                          "result": {"sent_back": [sent] if sent else [], "concerns": len(kept), "refused": len(refused), "source": result.source},
                                          "cost_eur": round(max(0.0, _spent(draft_id, state.get("runs") or []) - before), 6)}]
        what = "send_back" if sent else "concerns_stand" if gate else "critique_passed"
        return {"steps": steps, "next": "critic_gate" if gate else "situate", "board": self._advance(state, "critic", what)}

    def critic_gate(self, state: WalkState) -> dict[str, Any]:
        """The critic's concerns that stand, put to the person (an approval): build on, recorded as overridden, or stop with
        the concerns as the blocker they see. A node of its own: resuming it never asks the critic's model again."""
        from dclab_core import coordinator as machine

        draft_id = state["draft_id"]
        own = dict(((self.c.drafts.get(draft_id).get("coordinator") or {}).get("critic") or {}))
        concerns = list(own.get("awaiting") or [])
        token = _PAUSES.set([])
        try:
            ok = self.asker.approve("understanding", {"concerns": concerns})
            _SESSION.get().step("approve", {"gate": "understanding"}, {"approved": ok, "concerns": len(concerns)})
            board = self._close_pauses(state, "critic")
        finally:
            _PAUSES.reset(token)
        own.update(passed=ok, overridden=ok or None, awaiting=None if ok else concerns)
        self._keep_critic(draft_id, own)
        now = machine.Situation.model_validate(state["situation"])
        steps = [*state.get("steps", []), {"state": now.state, "action": "critic_gate", "says": now.says, "seconds": 0.0,
                                          "result": {"approved": ok, "concerns": [c["field"] for c in concerns]}}]
        update = {"steps": steps, "board": self._advance(state, "critic", "overridden" if ok else "stopped", board)}
        if ok:
            return {**update, "next": "situate"}
        says = "The understanding critic's concerns stand: " + "; ".join(c["question"] for c in concerns[:3])
        return {**update, "next": "stop", "says": says[:600]}

    def _keep_critic(self, draft_id: str, own: dict[str, Any]) -> None:
        def keep(d: dict[str, Any]) -> None:
            d.setdefault("coordinator", {})["critic"] = json.loads(json.dumps(own, default=str))
        self.c.drafts.update(draft_id, keep)

    @staticmethod
    def _route(state: WalkState) -> str:
        nxt = state.get("next")
        return nxt if nxt in (*ACTS, "critic", "critic_gate") else "end"

    @staticmethod
    def _after(state: WalkState) -> str:
        return "situate" if state.get("next") == "situate" else "end"

    def build(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(WalkState)
        g.add_node("situate", self.situate)
        for action in ACTS:
            g.add_node(action, self._act(action))
            g.add_conditional_edges(action, self._after, {"situate": "situate", "end": END})
        g.add_node("critic", self.critic)
        g.add_conditional_edges("critic", lambda st: st.get("next") if st.get("next") in ("situate", "critic_gate") else "end",
                                {"situate": "situate", "critic_gate": "critic_gate", "end": END})
        g.add_node("critic_gate", self.critic_gate)
        g.add_conditional_edges("critic_gate", self._after, {"situate": "situate", "end": END})
        g.add_edge(START, "situate")
        g.add_conditional_edges("situate", self._route, {**{a: a for a in ACTS}, "critic": "critic", "critic_gate": "critic_gate", "end": END})
        self.graph = g.compile(checkpointer=self.checkpointer)
        return self.graph

    # ------------------------------------------------------------------------------------------------ the walk
    def run(self, draft_id: str, resume: Any = None, store: Any = None) -> dict[str, Any]:
        """Start the walk on a draft, or resume it with the person's answer; ``{"status": "paused", "asks": …}`` at a pause,
        else the loop's own result shape (state, next, says, steps, project_id)."""
        from langgraph.types import Command

        from .. import experience
        from . import board as B

        if self.graph is None:
            self.build()
        thread = thread_id("draft", draft_id)
        config = {"configurable": {"thread_id": thread}, "recursion_limit": 4 * self.max_steps + 20}
        with self.c.drafts.turn(draft_id):  # the state read and the walk under one turn: two requests never resume from the same read
            snap = self.graph.get_state(config)
            open_ = [i for t in snap.tasks for i in (getattr(t, "interrupts", None) or ())]
            if resume is not None and not open_:
                raise NothingToResume(f"nothing to resume on {thread}: no pause is open")
            if resume is None and open_:  # started again while a pause waits: the pause stands (the review of 28.8: the walk was thrown away)
                return {"status": "paused", "asks": dict(open_[0].value), "state": snap.values.get("state")}
            if resume is None and snap.values:  # a new walk starts from the draft as it is, on a fresh thread
                self.checkpointer.delete_thread(thread)
            if resume is not None and not (isinstance(resume, dict) and "key" in resume):
                resume = {"key": dict(open_[0].value).get("key"), "answer": resume}  # the answer is to the pause the person was shown
            with experience.bind(conversation=draft_id), \
                    harness.session("orchestra.walk", store=store, params={"draft": draft_id, "resumed": resume is not None}) as current:
                token = _SESSION.set(current)
                try:
                    with experience.bind(parent=current.run_id):  # 28.8: a nested agent's requests (the Home agent's run) count for the walk
                        if resume is None:
                            self.graph.invoke({"draft_id": draft_id, "board": B.new(thread).model_dump(mode="json"), "steps": [], "seen": {}, "last": None,
                                               "started": B.now(), "runs": [current.run_id]}, config)
                        else:
                            self.graph.invoke(Command(resume=resume, update={"runs": [current.run_id]}), config)
                finally:
                    _SESSION.reset(token)
                return self._after_invoke(draft_id, thread, config, current)

    def _after_invoke(self, draft_id: str, thread: str, config: dict[str, Any], current: Any) -> dict[str, Any]:
        """The walk's end of a run, inside its session and its turn: the pause it waits on (opened on the board), or its end
        (the coordinator's own record of where it stopped, and the walk's record with its spend and the models it asked)."""
        from dclab_core.orchestra import Board

        from . import board as B

        snap = self.graph.get_state(config)
        pending = [i for task in snap.tasks for i in (getattr(task, "interrupts", None) or ())]
        values = snap.values
        if pending:
            asks = dict(pending[0].value)
            current.step("interrupt", {"kind": asks.get("kind"), "field": asks.get("field"), "gate": asks.get("gate")}, "paused")
            current.status = "stopped"
            B.interrupt(Board.model_validate(values["board"]), asks.get("kind") or "question", "walk", self.TEAM, field=asks.get("field"),
                        gate=asks.get("gate"), text=str(asks.get("text") or ""), status="open", home=self.home)
            return {"status": "paused", "asks": asks, "state": values.get("state")}
        if values.get("next") != "done":
            current.status = "stopped"
        now = self.c.where(draft_id)
        if values.get("next") == "stop":
            now = now.model_copy(update={"next": "stop", "says": values.get("says") or now.says})
        self.c._remember(draft_id, now)  # the coordinator's own record of where it stopped
        spent, models = _requests(values.get("runs") or [], conversation=draft_id)
        B.record(thread, self.TEAM, _nodes(values.get("steps") or []), Board.model_validate(values["board"]),
                 "completed" if now.next == "done" else "stopped", values.get("started") or B.now(), home=self.home,
                 spent_eur=spent, walk=_walk_id(values), models=models)
        return {"status": "done", "state": now.state, "next": now.next, "says": now.says, "blockers": [b.model_dump() for b in now.blockers],
                "passed": now.passed, "steps": values.get("steps") or [], "board": values.get("board"),
                "project_id": values.get("project_id") or self.c.drafts.get(draft_id).get("project_id")}


def _label_send_back(draft_id: str, field: str, changed: bool) -> None:
    """The measured label of the critic's last send-back of ``field`` on this draft: right when the person's answer changed the
    card's field, not needed when it kept it."""
    from .. import ledger

    book = ledger.current_ledger()
    if book is None:
        return
    mine = [r for r in book.records(draft=draft_id, point="critic.send_back") if r.state.get("field") == field and r.value_used is True]
    if mine:
        ledger.label(mine[-1], bool(changed), "measured", "person_answer", evidence={"kind": "review", "value": 1.0 if changed else 0.0, "ref": f"understanding.{field}"[:200]})


def _walk_id(state: Any) -> Optional[str]:
    """A walk's id: its first session's run (unique, and the same at every resume of the walk)."""
    runs = (state or {}).get("runs") or []
    return str(runs[0]) if runs else None


def _nodes(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The walk node by node: how many times each ran, the seconds it took and what its model requests cost (28.8), in the
    order first run."""
    out: dict[str, dict[str, Any]] = {}
    for s in steps:
        n = out.setdefault(str(s.get("action")), {"node": str(s.get("action")), "steps": 0, "seconds": 0.0, "cost_eur": 0.0})
        n["steps"] += 1
        n["seconds"] = round(n["seconds"] + float(s.get("seconds") or 0.0), 3)
        n["cost_eur"] = round(n["cost_eur"] + float(s.get("cost_eur") or 0.0), 6)
    return list(out.values())


def _requests(runs: list[str], **scope: str) -> tuple[float, Optional[list[str]]]:
    """What the walk's model requests cost and which models answered them: the requests of its sessions (one per start or
    resume) in the experience log; the models None when no log is there to say (unknown: such a walk never trains)."""
    from .. import experience

    log = experience.current_log()
    if log is None:
        return 0.0, None
    mine = set(runs)
    found = [e for e in log.read(**scope) if e.kind == "model.request" and (e.ids.run in mine or e.ids.parent in mine)]  # a nested run's too
    return round(sum(float(e.payload.get("cost") or 0.0) for e in found), 6), sorted({str(e.payload.get("model")) for e in found if e.payload.get("model")})


def _spent(draft_id: str, runs: list[str]) -> float:
    return _requests(runs, conversation=draft_id)[0]


def drive(walk: Walk, draft_id: str, person: Any, max_pauses: int = 60) -> dict[str, Any]:
    """Walk the graph with a person who answers as the coordinator's loop would ask them (``ask``/``answer`` and
    ``approve``): the same journeys can then be measured through the graph and through the loop."""
    out = walk.run(draft_id)
    for _ in range(max_pauses):
        if out.get("status") != "paused":
            return out
        asks = out["asks"]
        if asks.get("kind") == "question":
            ask = getattr(person, "ask", None) or getattr(person, "answer")
            said = ask(asks.get("field"), asks.get("text") or "")
            out = walk.run(draft_id, resume=said if said is not None else "")
        else:
            approve = getattr(person, "approve", None)
            out = walk.run(draft_id, resume=bool(approve(asks.get("gate") or "build", asks.get("summary") or {})) if approve else False)
    return out


__all__ = ["ACTS", "Asker", "Walk", "WalkState", "drive"]
