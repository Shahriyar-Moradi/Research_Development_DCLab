"""DCLab Code's sets (32.34): the owner's developer app's sessions on this repository (kept by 32.33) as training examples,
from the shaped records only. Each session is read in time order: the agent's tool steps (from its SDK messages), the
permission each step needed and the person's answer, the sealed and holdout guard's questions, the agent's questions and
the answers, plan approvals, the person's grade of each turn, and what became of the work (merged, its copy removed).

- **M6** (:class:`dclab_core.store.Episode`, folder ``dclab_code``): a session as an episode of tool steps, the person's own
  steps (``person:Bash`` for a command they ran, ``person:undo``, ``person:rewind``, ``person:stop_job`` for a background
  job they stopped) in their place between the agent's (32.39). A step's state
  is bands and codes only (its step number, the tool as a code, the tool's family, the permission mode, whether it asked
  and why), its action the tool's name, its outcome ``auto_ok``, ``auto_error``, ``allowed_ok``, ``allowed_error``,
  ``denied``, ``answered``, ``skipped`` or ``plan_<choice>``; the episode's outcome from the record (``merged``, ``stopped``,
  the last turn's result). A test script's decision or merge keeps its fact with ``_by_script`` (``denied_by_script``,
  ``merged_by_script``): DCLab Code records who acted (``by``), and only a person's act (or a record older than ``by``)
  becomes a label anywhere (32.40). Silver weight while the agent (Claude) made steps in it; an episode of the person's steps alone
  is gold.
- **M7** (:class:`dclab_core.store.JudgedExample`): the person's judgement, gold: a step allowed or denied (folder
  ``dclab_code_steps``: the tool, its command or file path, why it asked), a plan approved or sent back with a note
  (``dclab_code_plans``: the plan whole), a turn's answer graded good or not good with a note (``dclab_code_answers``),
  and the agent's file changes judged by what the person did with them (``dclab_code_changes``: ``committed`` 1,
  ``undone`` 0 by a per-file undo, ``rewound`` 0 by a Rewind to before one of their messages, 32.41).
- **M2 and M1** (:class:`dclab_core.store.DecisionExample`): the person's choices at two points, gold:
  ``dclab_code.permission`` (allow, allow_always, deny) and ``dclab_code.plan_approval`` (auto, ask, keep), in M2's
  ``internal`` group and as M1 student sets the tutor fits where enough rows exist.

A tool's output never enters a set (it may hold a table's rows); a step's command or path does, redacted by the keeper. A
session's split is keyed by its SDK session id, the same session as its Claude Code transcript (32.7), so the two never sit
in different splits.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Optional

from dclab_core.datasets import split_of
from dclab_core.store import DecisionExample, Episode, EpisodeStep, JudgedExample, Message, PreferencePair, StoreRecord

from ..layout import Layout
from .. import provider_outputs
from . import allowed, write_set
from .decisions import SILVER

PREFIX = "dclab_code/records/"
INDEX = "dclab_code/sessions.json"
POINT_PERMISSION = "dclab_code.permission"
POINT_PLAN = "dclab_code.plan_approval"
FAMILIES = {"Bash": "shell", "Read": "read", "Write": "write", "Edit": "edit", "MultiEdit": "edit", "NotebookEdit": "edit", "Glob": "search",
            "Grep": "search", "WebFetch": "web", "WebSearch": "web", "Task": "agent", "Agent": "agent", "AskUserQuestion": "question",
            "ExitPlanMode": "plan", "TodoWrite": "tasks", "ToolSearch": "tools"}
EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
# Code only: a data file (csv, json, parquet...) and a file that often carries a table's rows whole (txt, sql inserts, an
# html report, a notebook's outputs) never enters a set. Markdown stays: the shaper takes tables out of its text.
CODE_FILE = re.compile(r"\.(py|r|scala|js|jsx|ts|tsx|java|go|rs|swift|kt|c|cc|cpp|h|sh|md|toml|yaml|yml|cfg|ini|css)$", re.I)
MAX_CHANGE = 200_000  # a change larger than this (a generated file) is not an example
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
        self.split_key = self.sdk_id  # a fork's is its root source's (sessions() sets it): one conversation, one split
        self.model = _code(meta.get("model") or "unknown")
        self.steps: list[dict[str, Any]] = []      # the agent's own tool steps (M6), in order
        self.asked: list[dict[str, Any]] = []      # every step that asked the person, a subagent's included (M7, M2, M1)
        self.plans: list[dict[str, Any]] = []
        self.grades: list[dict[str, Any]] = []
        self.prompts: dict[int, str] = {}         # turn -> the person's message that started it
        self.edits: dict[str, list[dict[str, Any]]] = defaultdict(list)  # file (relative to the session's folder) -> the agent's changes
        self.undos: dict[str, list[dict[str, Any]]] = defaultdict(list)  # file -> the person's undos of it, in order
        self.kept: list[dict[str, Any]] = []        # each commit or merge of the session's work: its order and the files in it
        self.person_steps: list[dict[str, Any]] = []  # the person's own acts between the agent's steps (commands, undos, rewinds), in order
        self._message_at: dict[str, tuple] = {}     # a person's message uuid -> its place (a rewind goes back to before it)
        self.outcome: Optional[str] = None
        self._read()

    @staticmethod
    def _auto(d: dict[str, Any], after_stop: bool) -> bool:
        """The app answered (a stop, a session ending): a fact about the step. Who decided is apart (``_person``)."""
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
                if b.get("uuid"):
                    self._message_at[str(b["uuid"])] = _order(r)  # a rewind names a message by it
            elif t in ("permission_request", "question_request", "plan_request") and b.get("toolUseId"):
                requests[str(b["toolUseId"])] = {**b, "_record": r}
            elif t in ("permission_decision", "question_answer", "plan_decision") and b.get("reqId"):
                decisions[str(b["reqId"])] = {**b, "_auto": self._auto(b, after_stop), "_person": by_person(b), "_record": r}
        # Pass 2: the steps in file order.
        mode, turn, last_text, stopped, self._thinking = "unknown", 0, None, False, None
        starts: dict[str, tuple] = {}  # a person's command: where it started
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
                        elif block.get("type") == "thinking" and not sub and block.get("thinking"):
                            self._thinking = block.get("thinking")  # the model's recorded thinking before its next step
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
                    last_text, self._thinking = None, None  # a turn's grade and thinking belong to this turn only
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
            elif t == "feedback" and by_person(b):  # a test script's grade is no person's judgement
                self.grades.append({"turn": b.get("turn"), "value": b.get("value"), "note": b.get("note"), "record": r, "results_before": turn})
            elif t == "undo" and isinstance(b.get("file"), str):
                # A script's undo still cuts the file's history (file_verdicts), with no verdict and no person's step.
                self.undos[b["file"]].append({"how": b.get("how"), "record": r, "order": _order(r), "person": by_person(b)})
                if by_person(b):
                    self.person_steps.append({"order": _order(r), "action": "person:undo", "kind": "person_undo", "turn": turn, "mode": mode,
                                              "outcome": _code(b.get("how") or "undone"), "guard": None, "at": r.at})
            elif t == "rewind" and isinstance(b.get("filesChanged"), list):
                # The person put every file back as it was before one of their messages: each change since that message,
                # to the files it names, is rejected (a script's rewind only cuts the history, as its undo does).
                since = self._message_at.get(str(b.get("messageId")))
                for f in b["filesChanged"]:
                    rel = self._rel(str(f))
                    if rel is not None:
                        self.undos[rel].append({"how": "rewind", "record": r, "order": _order(r), "person": by_person(b), "since": since})
                if by_person(b):
                    self.person_steps.append({"order": _order(r), "action": "person:rewind", "kind": "person_rewind", "turn": turn, "mode": mode,
                                              "outcome": "rewound" if b["filesChanged"] else "nothing_to_rewind", "guard": None, "at": r.at})
            elif t == "job_stop" and by_person(b):
                # The person stopped a background job the agent left running (a training run, a test suite): their step.
                self.person_steps.append({"order": _order(r), "action": "person:stop_job", "kind": "person_stop_job", "turn": turn, "mode": mode,
                                          "outcome": "stopped", "guard": None, "at": r.at})
            elif t == "file_opened" and by_person(b):
                # dclab-code A3: the person opened a file the conversation named (or they picked): their step, its path left
                # out of the content-free episode; a sealed set's refusal is said as its outcome.
                self.person_steps.append({"order": _order(r), "action": "person:open_file", "kind": "person_open_file", "turn": turn, "mode": mode,
                                          "outcome": "sealed_refused" if b.get("sealed") else "opened", "guard": "sealed" if b.get("sealed") else None, "at": r.at})
            elif t == "person_command_start" and b.get("id"):
                starts[str(b["id"])] = _order(r)
            elif t == "person_command" and by_person(b):
                code = "timed_out" if b.get("timedOut") else f"exit_{b.get('exit')}" if isinstance(b.get("exit"), int) else "no_exit"
                # placed where it started (the app records the start first), so what the agent did meanwhile comes after
                self.person_steps.append({"order": starts.get(str(b.get("id")), _order(r)), "action": "person:Bash", "kind": "person_shell", "turn": turn, "mode": mode,
                                          "outcome": code, "guard": b.get("guard"), "at": r.at})
            elif t == "outcome":
                o = b.get("outcome")
                if o in ("committed", "merged") and isinstance(b.get("files"), list):  # older records name no files: no verdict from them
                    self.kept.append({"order": _order(r), "files": {str(f) for f in b["files"]}, "person": by_person(b)})
                code = "merged" if o == "merged" else ("copy_removed" if b.get("merged") else "copy_removed_unmerged") if o == "copy_removed" else _code(o)
                self.outcome = code if by_person(b) else f"{code}_by_script"  # a script's merge is no sign the person kept the work
            elif t == "stop":
                self.outcome, stopped = "stopped", True
            elif t == "user_message":
                stopped = False
                if isinstance(b.get("text"), str):
                    self.prompts.setdefault(turn, b["text"])
        for g in self.grades:
            # The turn the app recorded (1-based); an older record without it: the last turn finished before the grade.
            n = g["turn"] if isinstance(g["turn"], int) and g["turn"] >= 1 else g["results_before"]
            g["answer"] = answers_by_turn[n - 1] if 1 <= n <= len(answers_by_turn) else None

    def _step(self, block: dict[str, Any], mode: str, turn: int, r: StoreRecord, sub: bool, requests: dict, decisions: dict) -> None:
        tid = str(block.get("id") or "")
        req = requests.get(tid)
        dec = decisions.get(str(req.get("reqId"))) if req else None
        step = {"id": tid, "n": len(self.steps), "order": _order(r), "decided": _order(dec["_record"]) if dec else None, "tool": str(block.get("name") or "unknown"), "input": block.get("input"), "mode": mode, "turn": turn,
                "asked": bool(req), "guard": (req or {}).get("guard"), "reason": (req or {}).get("reason"), "decision": None,
                "auto": bool(dec and dec["_auto"]), "person": bool(dec and dec["_person"]), "ok": None, "at": r.at, "record": (dec or {}).get("_record") or r, "sub": sub,
                "thinking": None if sub else self._thinking}
        if not sub:
            self._thinking = None
        if dec:
            t = dec.get("type")
            step["decision"] = dec.get("decision") if t == "permission_decision" else ("answered" if dec.get("answers") else "skipped") \
                if t == "question_answer" else f"plan_{_code(dec.get('choice'))}"
        if req and req.get("type") == "plan_request":
            self.plans.append({"plan": req.get("plan") or "", "choice": (dec or {}).get("choice"), "feedback": (dec or {}).get("feedback"),
                               "record": (dec or {}).get("_record") or r, "mode": mode, "auto": step["auto"], "person": step["person"]})
        if not sub:
            self.steps.append(step)
            self._edit(step)
        # A label only where the person decided: never the app's answer, never a test script's.
        if req and req.get("type") == "permission_request" and step["decision"] in ("allow", "allow_always", "deny") and not step["auto"] and step["person"]:
            self.asked.append(step)

    def _edit(self, st: dict[str, Any]) -> None:
        """An agent's change to a code or text file, by the file's path in the session's folder (as git and an undo name it)."""
        inp = st["input"] if isinstance(st["input"], dict) else {}
        path = inp.get("file_path") or inp.get("notebook_path")
        if st["tool"] not in EDIT_TOOLS or not isinstance(path, str) or not CODE_FILE.search(path) or st["decision"] == "deny":
            return
        rel = self._rel(path)
        if rel is None:
            return  # outside the repository: no commit of it can hold this file
        change = {k: inp[k] for k in ("old_string", "new_string", "content", "new_source", "edits", "replace_all") if k in inp}
        if len(json.dumps(change, default=str)) <= MAX_CHANGE:
            self.edits[rel].append({"tool": st["tool"], **change, "_record": st["record"], "_order": st["order"], "_step": st})

    def _rel(self, path: str) -> Optional[str]:
        """A path as git names it: from the repository's root (an own copy's root, else the session's recorded root);
        None outside it. The SDK names a rewound file by its real path (/private/var/... for /var/...): both count."""
        wt = self.meta.get("worktree") if isinstance(self.meta.get("worktree"), dict) else {}
        root = str(wt.get("path") or self.meta.get("repoRoot") or self.meta.get("projectPath") or "").rstrip("/")
        if not root:
            return None
        for base in (root, "/private" + root):
            if path.startswith(base + "/"):
                return path[len(base) + 1:]
        return None

    @staticmethod
    def step_outcome(s: dict[str, Any]) -> str:
        d = s["decision"]
        if s["auto"]:
            return "cancelled"  # the app answered (a stop, a session ending): not the person's choice
        # What happened, as it happened; a test script's decision says so (it is no person's choice).
        mark = "_by_script" if d is not None and not s["person"] else ""
        if d == "deny":
            return "denied" + mark
        if d in ("answered", "skipped") or str(d).startswith("plan_"):
            return str(d) + mark
        ran = "ok" if s["ok"] else "error" if s["ok"] is False else "no_result"
        return (f"allowed_{ran}" + mark) if s["asked"] else f"auto_{ran}"

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
            if isinstance(r.body, dict) and r.body.get("copiedFrom"):
                continue  # dclab-code A4: a fork's copy of its source's history, kept (and counted) once, in the source
            lines[src[len(PREFIX):-len(".jsonl")]].append(r)

    def root(sid: str) -> str:  # the session a chain of forks started from
        seen = set()
        while sid not in seen and isinstance((index.get(sid) or {}).get("forkOf"), dict) and index[sid]["forkOf"].get("session"):
            seen.add(sid)
            sid = str(index[sid]["forkOf"]["session"])
        return sid

    out = []
    for sid, rs in sorted(lines.items()):
        s = Session(sid, rs, index.get(sid, {}))
        origin = root(sid)
        if origin != sid:  # a fork continues its source's conversation: its split is the source's, never another one
            s.split_key = str((index.get(origin) or {}).get("sdkSessionId") or origin)
        out.append(s)
    return out


def examples(layout: Layout, choices: Optional[dict] = None) -> dict[tuple[str, str], list[Any]]:
    from . import standing

    choices = choices if choices is not None else standing(layout)
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for s in sessions(layout):
        # The same key the shaper gives the session's Claude Code transcript lines (rule sealed_key "session": the raw id),
        # so the session and its transcript always share a split.
        split = split_of(s.split_key)
        first, more = s.provenance()
        ok = {mt: allowed(s.lines[0], mt, choices) for mt in ("m1_students", "m2_workflow_decisions", "m5_solver_llm", "m6_stepwise_decisions", "m7_judges")}
        if ok["m6_stepwise_decisions"] and (s.steps or s.person_steps):
            # The agent's steps and the person's own (commands, undos) in file order: what each did, one step after another.
            timeline = sorted([(st["order"], "agent", st) for st in s.steps] + [(p["order"], "person", p) for p in s.person_steps], key=lambda x: x[0])
            steps = []
            for i, (_, who, x) in enumerate(timeline):
                if who == "agent":
                    steps.append(EpisodeStep(n=i, state=s.state(i, x), action=x["tool"][:200], outcome=Session.step_outcome(x), at=x["at"]))
                else:
                    state = {"step": i, "move": _code(x["action"]), "kind": x["kind"], "method": x["mode"], "gate_required": False,
                             "concern_source": x["guard"] or "none", "rounds": x["turn"]}
                    steps.append(EpisodeStep(n=i, state=state, action=x["action"], outcome=x["outcome"], at=x["at"]))
            groups[("m6_stepwise_decisions", "dclab_code")].append(Episode(
                # silver while the agent (Claude) made steps in it; the person's steps alone are gold
                id=f"m6:dclab_code:{s.sid}", model_type="m6_stepwise_decisions", split=split, provenance=first, more=more, weight=SILVER if s.steps else 1.0,
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
                        group=group, split_group=s.split_key, state=state, options=["allow", "allow_always", "deny"], label=st["decision"], label_source="person"))
        for j, p in enumerate(s.plans):
            if not p["choice"] or p["auto"] or not p["person"]:
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
                        split_group=s.split_key, state={"method": p["mode"], "step": j}, options=["auto", "ask", "keep"], label=p["choice"], label_source="person"))
        for k, (chosen, rejected) in enumerate(preferences(s)):
            prompt = [Message(role="user", content=s.prompts.get(chosen["turn"]) or "Go on with the task.")]
            for mt, weight in (("m7_judges", 1.0), ("m5_solver_llm", SILVER)):  # M5: both steps are Claude's, so silver and apart
                if ok[mt]:
                    groups[(mt, "dclab_code_preferences")].append(PreferencePair(
                        id=f"{mt}:dclab_code:preference:{s.sid}:{k}", model_type=mt, split=split, provenance=chosen["record"].provenance,
                        more=[rejected["record"].provenance] if rejected["record"].provenance.entry != chosen["record"].provenance.entry else [],
                        weight=weight, task="dclab_code.step", prompt=prompt, chosen=_as_message(chosen), rejected=_as_message(rejected), decided_by="person"))
        if ok["m7_judges"]:
            for f, changes in sorted(s.edits.items()):
                for n, (verdict, chunk, prov) in enumerate(file_verdicts(f, changes, s.undos.get(f, []), s.kept)):
                    groups[("m7_judges", "dclab_code_changes")].append(JudgedExample(
                        id=f"m7:dclab_code:change:{s.sid}:{n}:{f}"[:200], model_type="m7_judges", split=split, provenance=prov, judged="code",
                        candidate={"file": f, "changes": [{k: v for k, v in c.items() if not k.startswith("_")} for c in chunk], "kept_as": verdict},
                        candidate_by=f"model:{s.model}", score=0.0 if verdict in ("undone", "rewound") else 1.0, judge="person", evidence="person"))
        for k, g in enumerate(s.grades):
            if ok["m7_judges"] and g["value"] in ("good", "bad") and g["answer"]:
                groups[("m7_judges", "dclab_code_answers")].append(JudgedExample(
                    id=f"m7:dclab_code:answer:{s.sid}:{k}", model_type="m7_judges", split=split, provenance=g["record"].provenance, judged="answer",
                    candidate={"answer": g["answer"], "note": g["note"]}, candidate_by=f"model:{s.model}",
                    score=1.0 if g["value"] == "good" else 0.0, judge="person", evidence="person"))
    return groups


def by_person(d: dict[str, Any]) -> bool:
    """Whether a record is the person's own act: its "by" is "person", or it has none (records older than "by").
    Anything else (the app, a test script: ``decidedBy: "script"`` in DCLab Code's config) is never a person's label."""
    return d.get("by") in (None, "person")


def file_verdicts(file: str, changes: list[dict[str, Any]], undos: list[dict[str, Any]], kept: list[dict[str, Any]]) -> list[tuple[str, list, Any]]:
    """A file's history in file order, cut at each undo and at each commit or merge. The changes since the last cut are
    **undone** (0) at an undo, **committed** (1) at a commit or merge that names this file; a commit that does not name it
    leaves them waiting; changes never cut have no verdict. Only a change that ran (its tool result ok) counts. A test
    script's undo or commit cuts the history too, with no verdict: it is no person's judgement. A rewind (32.41) is an
    undo of the changes since one message: those are **rewound** (0); the ones before it stay waiting."""
    ran = [c for c in changes if c["_step"].get("ok") is True]
    events = sorted([(c["_order"], 1, c) for c in ran] + [(u["order"], 0, u) for u in undos] + [(k["order"], 0, k) for k in kept],
                    key=lambda e: (e[0], e[1]))
    out, pending = [], []
    for _, is_change, item in events:
        if is_change:
            pending.append(item)
        elif "how" in item and item.get("since") is not None:  # a rewind: the changes since its message go back
            back = [c for c in pending if c["_order"] >= item["since"]]
            if back and item.get("person", True):
                out.append(("rewound", back, item["record"].provenance))
            pending = [c for c in pending if c["_order"] < item["since"]]
        elif "how" in item:  # an undo (a rewind to a message this session never recorded: everything pending)
            if pending and item.get("person", True):
                out.append(("rewound" if item["how"] == "rewind" else "undone", pending, item["record"].provenance))
            pending = []
        elif file in item["files"]:  # a commit or merge holding this file
            if pending and item.get("person", True):
                out.append(("committed", pending, pending[-1]["_record"].provenance))
            pending = []
    return out


def _as_message(st: dict[str, Any]) -> Message:
    call = {"name": st["tool"], "input": _brief(st["tool"], st["input"])}
    if st.get("thinking"):
        return Message(role="assistant", content="", tool_calls=[call], reasoning=st["thinking"], reasoning_source="recorded")
    return Message(role="assistant", content="", tool_calls=[call], no_reasoning="no thinking text was recorded before this step")


def preferences(s: Session) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Kept-against-refused pairs: two tries at one step. A step of the agent's own the person denied, and the agent's very
    next own step, with the same tool, in the same turn, proposed after the deny, that the person allowed. Not a parallel
    call proposed beside the denied one, not a step after a question or a plan in between (those are steps themselves),
    not a subagent's step, never the app's own deny."""
    pairs = []
    for st in s.steps:
        if st["decision"] != "deny" or st["auto"] or not st["person"] or st["n"] + 1 >= len(s.steps):
            continue
        nxt = s.steps[st["n"] + 1]
        if (nxt["tool"] == st["tool"] and nxt["turn"] == st["turn"] and nxt["asked"] and not nxt["auto"] and nxt["person"]
                and nxt["decision"] in ("allow", "allow_always") and st["decided"] is not None and nxt["order"] > st["decided"]):
            pairs.append((nxt, st))
    return pairs


ABOUTS_M6 = "DCLab Code sessions as episodes: the agent's tool steps and the person's own (commands, undos), content-free state, the outcome (silver while Claude made steps)."
ABOUTS = {"m5_solver_llm": "DCLab Code: the step the person allowed against the one they denied (both Claude's: silver, apart; consent from the owner's decision on provider outputs).",
          "m6_stepwise_decisions": ABOUTS_M6,
          "m7_judges": "The person's judgement of DCLab Code's steps, plans, answers and file changes (undone: 0; kept in committed or merged work: 1) (gold).",
          "m2_workflow_decisions": "The person's choices in DCLab Code: allow a step, approve a plan (gold).",
          "m1_students": "A student per DCLab Code decision point: what the person chooses, from the step's bands (gold)."}


def build(layout: Optional[Layout] = None) -> dict[str, Any]:
    layout = layout or Layout()
    report: dict[str, Any] = {"sets": {}}
    for (model_type, folder), exs in sorted(examples(layout).items()):
        sources = {e.provenance.entry: {"entry": e.provenance.entry, "sha256": e.provenance.sha256} for e in exs}
        # M5's copy holds only Claude's steps: its consent follows the owner's decision on provider outputs (provider_outputs.py).
        consent = provider_outputs.consent() if model_type == "m5_solver_llm" else "train"
        m = write_set(layout, model_type, folder, exs, sources=sources.values(), about=ABOUTS[model_type], consent=consent)
        report["sets"][f"{model_type}/{folder}"] = m.counts
    return report


__all__ = ["build", "examples", "sessions", "family", "POINT_PERMISSION", "POINT_PLAN"]
