"""DCLab Code's sets (32.34): the owner's developer app's sessions on this repository (kept by 32.33) as training examples,
from the shaped records only. Each session is read in time order: the agent's tool steps (from its SDK messages), the
permission each step needed and the person's answer, the sealed and holdout guard's questions, the agent's questions and
the answers, plan approvals, the person's grade of each turn, and what became of the work (merged, its copy removed).

- **M6** (:class:`dclab_core.store.Episode`, folder ``dclab_code``): a session as an episode of tool steps. A step's state
  is bands and codes only (its step number, the tool as a code, the tool's family, the permission mode, whether it asked
  and why), its action the tool's name, its outcome ``auto_ok``, ``auto_error``, ``allowed_ok``, ``allowed_error``,
  ``denied``, ``answered``, ``skipped`` or ``plan_<choice>``; the episode's outcome from the record (``merged``, ``stopped``,
  the last turn's result). The steps are the agent's (Claude), so the weight is silver.
- **M7** (:class:`dclab_core.store.JudgedExample`): the person's judgement, gold: a step allowed or denied (folder
  ``dclab_code_steps``: the tool, its command or file path, why it asked), a plan approved or sent back with a note
  (``dclab_code_plans``: the plan whole), a turn's answer graded good or not good with a note (``dclab_code_answers``).
- **M2 and M1** (:class:`dclab_core.store.DecisionExample`): the person's choices at two points, gold:
  ``dclab_code.permission`` (allow, allow_always, deny) and ``dclab_code.plan_approval`` (auto, ask, keep), in M2's
  ``internal`` group and as M1 student sets the tutor fits where enough rows exist.

A tool's output never enters a set (it may hold a table's rows); a step's command or path does, redacted by the keeper. A
session's split is keyed by its SDK session id, the same session as its Claude Code transcript (32.7), so the two never sit
in different splits.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Optional

from dclab_core.datasets import split_of
from dclab_core.store import DecisionExample, Episode, EpisodeStep, JudgedExample, StoreRecord

from ..layout import Layout
from . import allowed, write_set
from .decisions import SILVER

PREFIX = "dclab_code/records/"
INDEX = "dclab_code/sessions.json"
POINT_PERMISSION = "dclab_code.permission"
POINT_PLAN = "dclab_code.plan_approval"
FAMILIES = {"Bash": "shell", "Read": "read", "Write": "write", "Edit": "edit", "MultiEdit": "edit", "NotebookEdit": "edit", "Glob": "search",
            "Grep": "search", "WebFetch": "web", "WebSearch": "web", "Task": "agent", "Agent": "agent", "AskUserQuestion": "question",
            "ExitPlanMode": "plan", "TodoWrite": "tasks", "ToolSearch": "tools"}
MODES = {"default": "ask_first", "acceptEdits": "auto_edits", "plan": "plan_only", "bypassPermissions": "bypass", "dontAsk": "dont_ask", "auto": "auto"}


def _code(value: Any) -> str:
    return re.sub(r"[^a-z0-9_.\-]+", "_", str(value or "none").lower()).strip("_.-")[:80] or "none"


def family(tool: str) -> str:
    if tool.startswith("mcp__dclab__"):
        return "dclab"
    if tool.startswith("mcp__"):
        return "mcp"
    return FAMILIES.get(tool, "other")


def _brief(tool: str, inp: Any) -> dict[str, Any]:
    """What a judge sees of a step: the command, the path, a DCLab tool's arguments; never a file's content."""
    inp = inp if isinstance(inp, dict) else {}
    if tool == "Bash":
        return {"command": str(inp.get("command") or "")[:4000]}
    if family(tool) in ("read", "write", "edit"):
        return {"file_path": str(inp.get("file_path") or inp.get("notebook_path") or "")}
    if family(tool) in ("dclab", "mcp", "search", "web"):
        return {k: v for k, v in inp.items() if isinstance(v, (str, int, float, bool)) and len(str(v)) <= 300}
    return {}


# Decisions the app made itself (a stop, a session ending, a cancelled step): never the person's gold label.
AUTO_TEXTS = ("The person stopped the session.", "The session ended.", "The step was cancelled.")


def _order(r: StoreRecord) -> tuple[str, int]:
    """File order: the entry (a log's later tails come later) and the line within it. The shaper rewrites the lines' own
    timestamps (an id-shaped redaction), so they are never used to order or to join."""
    head, _, n = r.id.rpartition("#")
    return (r.provenance.at, int(n) if n.isdigit() else 0)


class Session:
    """One DCLab Code session read from its record lines in file order, joined by ids (tool use, request), never by time."""

    def __init__(self, sid: str, lines: list[StoreRecord], meta: dict[str, Any]):
        self.sid, self.lines, self.meta = sid, sorted(lines, key=_order), meta
        self.sdk_id = str(meta.get("sdkSessionId") or sid)
        self.model = _code(meta.get("model") or "unknown")
        self.steps: list[dict[str, Any]] = []      # the agent's own tool steps (M6), in order
        self.asked: list[dict[str, Any]] = []      # every step that asked the person, a subagent's included (M7, M2, M1)
        self.plans: list[dict[str, Any]] = []
        self.grades: list[dict[str, Any]] = []
        self.outcome: Optional[str] = None
        self._read()

    @staticmethod
    def _auto(d: dict[str, Any], after_stop: bool) -> bool:
        return d.get("by") == "app" or after_stop or (d.get("message") or d.get("feedback")) in AUTO_TEXTS

    def _read(self) -> None:
        # Pass 1: the joins, by id.
        requests: dict[str, dict[str, Any]] = {}   # toolUseId -> the request line
        decisions: dict[str, dict[str, Any]] = {}  # reqId -> the decision line, with whether the app made it
        after_stop = False
        for r in self.lines:
            b = r.body
            t = b.get("type")
            if t == "stop":
                after_stop = True
            elif t == "user_message":
                after_stop = False
            elif t in ("permission_request", "question_request", "plan_request") and b.get("toolUseId"):
                requests[str(b["toolUseId"])] = {**b, "_record": r}
            elif t in ("permission_decision", "question_answer", "plan_decision") and b.get("reqId"):
                decisions[str(b["reqId"])] = {**b, "_auto": self._auto(b, after_stop), "_record": r}
        # Pass 2: the steps in file order.
        mode, turn, last_text, stopped = "unknown", 0, None, False
        answers_by_turn: list[Optional[str]] = []
        for r in self.lines:
            b = r.body
            m = b.get("message") if isinstance(b.get("message"), dict) else {}
            if b.get("kind") == "sdk":
                if m.get("type") == "system" and m.get("subtype") == "init":
                    mode = MODES.get(str(m.get("permissionMode")), mode)
                    self.model = _code(m.get("model") or self.model)
                elif m.get("type") == "assistant":
                    sub = bool(m.get("parent_tool_use_id"))
                    for block in (m.get("message") or {}).get("content") or []:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "text" and not sub:
                            last_text = block.get("text")
                        elif block.get("type") == "tool_use":
                            self._step(block, mode, turn, r, sub, requests, decisions)
                elif m.get("type") == "user":
                    for block in (m.get("message") or {}).get("content") or []:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            for st in self.steps:
                                if st["id"] == block.get("tool_use_id"):
                                    st["ok"] = not block.get("is_error")
                elif m.get("type") == "result":
                    answers_by_turn.append(last_text)
                    last_text = None  # a turn's grade is about this turn's answer only
                    # A stop ends its turn with an error result: the turn's outcome is the stop.
                    self.outcome = "stopped" if stopped else str(m.get("subtype") or "result")
                    turn += 1
                continue
            t = b.get("type")
            if t == "plan_decision" and not decisions.get(str(b.get("reqId")), {}).get("_auto"):
                if b.get("choice") == "auto":
                    mode = "auto_edits"
                elif b.get("choice") == "ask":
                    mode = "ask_first"
            elif t == "feedback":
                self.grades.append({"turn": b.get("turn"), "value": b.get("value"), "note": b.get("note"), "record": r, "results_before": turn})
            elif t == "outcome":
                o = b.get("outcome")
                self.outcome = "merged" if o == "merged" else ("copy_removed" if b.get("merged") else "copy_removed_unmerged") if o == "copy_removed" else _code(o)
            elif t == "stop":
                self.outcome, stopped = "stopped", True
            elif t == "user_message":
                stopped = False
        for g in self.grades:
            # The turn the app recorded (1-based); an older record without it: the last turn finished before the grade.
            n = g["turn"] if isinstance(g["turn"], int) and g["turn"] >= 1 else g["results_before"]
            g["answer"] = answers_by_turn[n - 1] if 1 <= n <= len(answers_by_turn) else None

    def _step(self, block: dict[str, Any], mode: str, turn: int, r: StoreRecord, sub: bool, requests: dict, decisions: dict) -> None:
        tid = str(block.get("id") or "")
        req = requests.get(tid)
        dec = decisions.get(str(req.get("reqId"))) if req else None
        step = {"id": tid, "n": len(self.steps), "tool": str(block.get("name") or "unknown"), "input": block.get("input"), "mode": mode, "turn": turn,
                "asked": bool(req), "guard": (req or {}).get("guard"), "reason": (req or {}).get("reason"), "decision": None,
                "auto": bool(dec and dec["_auto"]), "ok": None, "at": r.at, "record": (dec or {}).get("_record") or r, "sub": sub}
        if dec:
            t = dec.get("type")
            step["decision"] = dec.get("decision") if t == "permission_decision" else ("answered" if dec.get("answers") else "skipped") \
                if t == "question_answer" else f"plan_{_code(dec.get('choice'))}"
        if req and req.get("type") == "plan_request":
            self.plans.append({"plan": req.get("plan") or "", "choice": (dec or {}).get("choice"), "feedback": (dec or {}).get("feedback"),
                               "record": (dec or {}).get("_record") or r, "mode": mode, "auto": step["auto"]})
        if not sub:
            self.steps.append(step)
        if req and req.get("type") == "permission_request" and step["decision"] in ("allow", "allow_always", "deny") and not step["auto"]:
            self.asked.append(step)

    @staticmethod
    def step_outcome(s: dict[str, Any]) -> str:
        d = s["decision"]
        if s["auto"]:
            return "cancelled"  # the app answered (a stop, a session ending): not the person's choice
        if d == "deny":
            return "denied"
        if d in ("answered", "skipped") or str(d).startswith("plan_"):
            return str(d)
        ran = "ok" if s["ok"] else "error" if s["ok"] is False else "no_result"
        return f"allowed_{ran}" if s["asked"] else f"auto_{ran}"

    def state(self, i: int, s: dict[str, Any]) -> dict[str, Any]:
        return {"step": i, "move": _code(s["tool"]), "kind": family(s["tool"]), "method": s["mode"], "gate_required": bool(s["asked"]),
                "concern_source": s["guard"] or ("sdk" if s["asked"] else "none"), "rounds": s["turn"]}

    def provenance(self) -> tuple[Any, list[Any]]:
        first = self.lines[0].provenance
        more = list({r.provenance.entry: r.provenance for r in self.lines if r.provenance.entry != first.entry}.values())
        return first, more


def sessions(layout: Layout) -> list[Session]:
    from ..service import _current

    index: dict[str, dict[str, Any]] = {}
    lines: dict[str, list[StoreRecord]] = defaultdict(list)
    for r in _current(layout, "agents"):
        src = r.provenance.source
        if src == INDEX:
            for s in r.body.get("sessions") or []:
                if isinstance(s, dict) and s.get("id"):
                    index[str(s["id"])] = s
        elif src.startswith(PREFIX) and src.endswith(".jsonl"):
            lines[src[len(PREFIX):-len(".jsonl")]].append(r)
    out = []
    for sid, rs in sorted(lines.items()):
        out.append(Session(sid, rs, index.get(sid, {})))
    return out


def examples(layout: Layout, choices: Optional[dict] = None) -> dict[tuple[str, str], list[Any]]:
    from . import standing

    choices = choices if choices is not None else standing(layout)
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for s in sessions(layout):
        # The same key the shaper gives the session's Claude Code transcript lines (rule sealed_key "session": the raw id),
        # so the session and its transcript always share a split.
        split = split_of(s.sdk_id)
        first, more = s.provenance()
        ok = {mt: allowed(s.lines[0], mt, choices) for mt in ("m1_students", "m2_workflow_decisions", "m6_stepwise_decisions", "m7_judges")}
        if ok["m6_stepwise_decisions"] and s.steps:
            steps = [EpisodeStep(n=i, state=s.state(i, st), action=st["tool"][:200], outcome=Session.step_outcome(st), at=st["at"]) for i, st in enumerate(s.steps)]
            groups[("m6_stepwise_decisions", "dclab_code")].append(Episode(
                id=f"m6:dclab_code:{s.sid}", model_type="m6_stepwise_decisions", split=split, provenance=first, more=more, weight=SILVER,
                process="dclab_code", steps=steps, outcome=(s.outcome or None) and s.outcome[:200]))
        for i, st in enumerate(s.asked):
            prov = st["record"].provenance
            if ok["m7_judges"]:
                groups[("m7_judges", "dclab_code_steps")].append(JudgedExample(
                    id=f"m7:dclab_code:step:{s.sid}:{i}", model_type="m7_judges", split=split, provenance=prov, judged="step",
                    candidate={"tool": st["tool"], "kind": family(st["tool"]), "input": _brief(st["tool"], st["input"]), "guard": st["guard"],
                               "why_asked": st["reason"], "mode": st["mode"]},
                    candidate_by=f"model:{s.model}", score=0.0 if st["decision"] == "deny" else 1.0, judge="person", evidence="person"))
            state = s.state(st["n"], st)
            for mt, folder, group in (("m2_workflow_decisions", f"internal/{POINT_PERMISSION}", "internal"), ("m1_students", POINT_PERMISSION, POINT_PERMISSION)):
                if ok[mt]:
                    groups[(mt, folder)].append(DecisionExample(
                        id=f"{mt}:dclab_code:permission:{s.sid}:{i}", model_type=mt, split=split, provenance=prov, point=POINT_PERMISSION,
                        group=group, split_group=s.sdk_id, state=state, options=["allow", "allow_always", "deny"], label=st["decision"], label_source="person"))
        for j, p in enumerate(s.plans):
            if not p["choice"] or p["auto"]:
                continue
            prov = p["record"].provenance
            if ok["m7_judges"]:
                groups[("m7_judges", "dclab_code_plans")].append(JudgedExample(
                    id=f"m7:dclab_code:plan:{s.sid}:{j}", model_type="m7_judges", split=split, provenance=prov, judged="plan",
                    candidate={"plan": p["plan"], "feedback": p["feedback"]}, candidate_by=f"model:{s.model}",
                    score=0.0 if p["choice"] == "keep" else 1.0, judge="person", evidence="person"))
            for mt, folder, group in (("m2_workflow_decisions", f"internal/{POINT_PLAN}", "internal"), ("m1_students", POINT_PLAN, POINT_PLAN)):
                if ok[mt]:
                    groups[(mt, folder)].append(DecisionExample(
                        id=f"{mt}:dclab_code:plan:{s.sid}:{j}", model_type=mt, split=split, provenance=prov, point=POINT_PLAN, group=group,
                        split_group=s.sdk_id, state={"method": p["mode"], "step": j}, options=["auto", "ask", "keep"], label=p["choice"], label_source="person"))
        for k, g in enumerate(s.grades):
            if ok["m7_judges"] and g["value"] in ("good", "bad") and g["answer"]:
                groups[("m7_judges", "dclab_code_answers")].append(JudgedExample(
                    id=f"m7:dclab_code:answer:{s.sid}:{k}", model_type="m7_judges", split=split, provenance=g["record"].provenance, judged="answer",
                    candidate={"answer": g["answer"], "note": g["note"]}, candidate_by=f"model:{s.model}",
                    score=1.0 if g["value"] == "good" else 0.0, judge="person", evidence="person"))
    return groups


ABOUTS = {"m6_stepwise_decisions": "DCLab Code sessions as episodes of tool steps: content-free state, the tool, the outcome (silver: the steps are Claude's).",
          "m7_judges": "The person's judgement of DCLab Code's steps, plans and answers (gold).",
          "m2_workflow_decisions": "The person's choices in DCLab Code: allow a step, approve a plan (gold).",
          "m1_students": "A student per DCLab Code decision point: what the person chooses, from the step's bands (gold)."}


def build(layout: Optional[Layout] = None) -> dict[str, Any]:
    layout = layout or Layout()
    report: dict[str, Any] = {"sets": {}}
    for (model_type, folder), exs in sorted(examples(layout).items()):
        sources = {e.provenance.entry: {"entry": e.provenance.entry, "sha256": e.provenance.sha256} for e in exs}
        m = write_set(layout, model_type, folder, exs, sources=sources.values(), about=ABOUTS[model_type])
        report["sets"][f"{model_type}/{folder}"] = m.counts
    return report


__all__ = ["build", "examples", "sessions", "family", "POINT_PERMISSION", "POINT_PLAN"]
