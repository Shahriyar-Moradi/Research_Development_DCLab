"""The LLM sets (32.12): M4, the workflow LLM, and M5, the solver LLM, built from the shaped records only.

**M4** (:class:`dclab_core.store.ChatExample`, by stage): every model exchange the gateway kept whole (32.10), its messages
as they were sent and the reply with its recorded thinking as the reasoning (or why it has none), its stage from its purpose
(``STAGES``); and each Home conversation (a draft's turns), its agent's turns without reasoning of their own (the reason
said). **M5** (:class:`dclab_core.store.SolverTrajectory`): each Claude Code session on this repository as a trajectory
(``trajectories/development``: every step's thought, its tool, its arguments and its result, from the transcript's lines),
and each intern session's last version (``trajectories/studio``).

Kept out of every train split, by :func:`contaminated`: an example holding a sealed person's words (every sealed journey and
conversation set of ``agent_eval``) or a benchmark task's name (``benchmarks.txt``: MLE-bench's tasks, one per line, the
owner's to fill). The split is the record's own (its sealed key), a session's by its id.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from dclab_core.datasets import split_of
from dclab_core.store import ChatExample, Message, Provenance, SolverTrajectory, StoreRecord, TrajectoryStep

from ..layout import ROOT, Layout
from . import write_set

STAGES = {"home_agent": "understand_problem", "conversation_reader": "understand_problem", "problem_analyst": "understand_problem",
          "brief_checker": "understand_problem", "understanding_critic": "understand_problem", "data_investigator": "understand_data",
          "workspace_chat": "understand_data", "metric_question": "understand_data", "feature_engineer": "design_solution",
          "solution_planner": "design_solution", "decomposer": "design_solution", "map_reviewer": "design_solution", "data_need": "design_solution",
          "solution_router": "design_solution", "solution_challenger": "design_solution", "intern": "plan", "ai_first": "plan",
          "stage_notes": "report", "reporter": "report", "critic": "report", "evidence_answer": "report", "leakage_review": "report",
          "notebook_review": "code", "solver_answer": "code", "debate_judge": "report"}
BENCHMARKS = Path(__file__).resolve().parent / "benchmarks.txt"
SEALED_GLOBS = ("dclab_rnd/agent_eval/**/*sealed*.json", "dclab_rnd/agent_eval/**/*sealed*.jsonl")
MIN_SEALED = 40  # a sealed sentence this long or longer is looked for in every example


def sealed_texts(root: Optional[Path] = None) -> set[str]:
    """Every sealed person's sentence long enough to recognise (the sealed sets' strings)."""
    root = root or ROOT
    out: set[str] = set()

    def walk(x: Any) -> None:
        if isinstance(x, str) and len(x) >= MIN_SEALED:
            out.add(x.strip().lower())
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for pattern in SEALED_GLOBS:
        for path in root.glob(pattern):
            try:
                text = path.read_text(encoding="utf-8")
                docs = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else [json.loads(text)]
            except (OSError, ValueError):
                continue
            for d in docs:
                walk(d)
    return out


def benchmark_names(path: Optional[Path] = None) -> set[str]:
    path = path or BENCHMARKS  # read at call time, so a test or a caller can point elsewhere
    if not path.exists():
        return set()
    return {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}


class Contamination:
    """Every sealed sentence and benchmark task as one compiled pattern, matched against an example's own texts (its messages,
    its thoughts, its results), never against their JSON (a quote or a line break in a sealed sentence would not match there)."""

    def __init__(self, sealed: set[str], benchmarks: set[str]):
        self.sealed = re.compile("|".join(re.escape(x) for x in sorted(sealed, key=len, reverse=True))) if sealed else None
        self.benchmarks = re.compile("|".join(re.escape(x) for x in sorted(benchmarks, key=len, reverse=True))) if benchmarks else None

    def why(self, texts: Iterable[str]) -> Optional[str]:
        for text in texts:
            low = str(text or "").lower()
            if self.sealed is not None and self.sealed.search(low):
                return "a sealed person's words"
            if self.benchmarks is not None:
                hit = self.benchmarks.search(low)
                if hit:
                    return f"a benchmark task ({hit.group(0)})"
        return None


def texts_of(example: Any) -> list[str]:
    out: list[str] = []
    for m in getattr(example, "messages", None) or []:
        out += [m.content or "", m.reasoning or ""]
    for st in getattr(example, "steps", None) or []:
        out += [str(getattr(st, "thought", "") or ""), json.dumps(getattr(st, "arguments", None) or {}, ensure_ascii=False, default=str),
                str(getattr(st, "result", "") or "")]
    if getattr(example, "problem", None):
        out.append(json.dumps(example.problem, ensure_ascii=False, default=str))
    return out


def contaminated(text: str, sealed: set[str], benchmarks: set[str]) -> Optional[str]:
    return Contamination(sealed, benchmarks).why([text])


def _content(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else message
    if isinstance(content, list):
        return "\n".join(str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") in (None, "text"))
    return str(content or "")


def _role(message: Any) -> str:
    role = str((message or {}).get("role") or "user")
    return role if role in ("system", "user", "assistant", "tool") else "user"


def teacher_reasonings(layout: Layout) -> dict[str, str]:
    """The teacher's checked reasonings by the record they explain (32.10)."""
    from ..store import Store

    store, out = Store(layout), {}
    for e in store.entries("teachers"):
        if e.source.startswith("reasoning/"):
            try:
                doc = json.loads(store.read(e))
                out[str(doc["record"])] = str(doc["reasoning"])
            except Exception:  # noqa: BLE001
                continue
    return out


def from_exchange(record: StoreRecord, taught: Optional[dict[str, str]] = None) -> Optional[ChatExample]:
    """One kept exchange as an M4 example: its messages, and the reply with its thinking as the reasoning."""
    b = record.body
    if not b.get("whole") or not isinstance(b.get("messages"), list):
        return None
    messages = []
    for m in b["messages"]:
        role = _role(m)
        if role == "assistant":
            messages.append(Message(role=role, content=_content(m), no_reasoning="an earlier turn, its thinking was not kept with this exchange"))
        else:
            messages.append(Message(role=role, content=_content(m), name=m.get("name") if isinstance(m.get("name"), str) else None))
    thinking = "\n\n".join(str(t.get("text") or t.get("thinking") or "") for t in b.get("thinking") or [] if isinstance(t, dict)).strip()
    taught = (taught or {}).get(record.id)
    labelled = ({"reasoning": thinking, "reasoning_source": "recorded"} if thinking else
                {"reasoning": taught, "reasoning_source": "teacher"} if taught else {"no_reasoning": "the model did not think, or its thinking was not asked for"})
    reply = Message(role="assistant", content=str(b.get("reply") or ""), tool_calls=[c for c in b.get("tool_calls") or [] if isinstance(c, dict)], **labelled)
    messages.append(reply)
    if len(messages) < 2:
        return None
    purpose = str(b.get("purpose") or "unknown")
    return ChatExample(id=f"m4_workflow_llm:{record.id}", model_type="m4_workflow_llm", split=record.split or split_of(str(b.get("model_request") or record.id)),
                       provenance=record.provenance, task=purpose, stage=STAGES.get(purpose, "plan"), messages=messages,
                       weight=0.5 if (taught and not thinking) else 1.0, check={"outcome": b.get("outcome")})


def from_draft(record: StoreRecord) -> Optional[ChatExample]:
    """A Home conversation (a draft's newest version) as an M4 example of understanding a problem."""
    b = record.body
    turns = [m for m in b.get("messages") or [] if isinstance(m, dict) and (m.get("text") or m.get("content"))]
    if len(turns) < 2:
        return None
    messages = [Message(role="user" if str(m.get("role")) in ("user", "person") else "assistant", content=str(m.get("text") or m.get("content") or ""),
                        **({} if str(m.get("role")) in ("user", "person") else {"no_reasoning": "the Home agent's turn: its thinking is in its model exchange when a model wrote it"}))
                for m in turns]
    group = f"project:{b['project_id']}" if b.get("project_id") else f"draft:{b.get('id')}"
    return ChatExample(id=f"m4_workflow_llm:{record.id}", model_type="m4_workflow_llm", split=split_of(group), provenance=record.provenance,
                       task="home_conversation", stage="understand_problem", messages=messages, check={"built": bool(b.get("project_id"))})


def trajectories(records: Iterable[StoreRecord]) -> Iterable[SolverTrajectory]:
    """Each Claude Code session as one trajectory: its steps in time order, each thought with its tool call and the call's result.
    Streamed transcript by transcript (a transcript's records come together, by its source), so the whole of 1 GB is never held."""
    current, lines_of = None, []

    def done(lines: list[StoreRecord]) -> Iterable[SolverTrajectory]:
        sessions: dict[str, list[StoreRecord]] = defaultdict(list)
        for r in lines:
            sessions[str(r.body["session"])].append(r)
        for session, rows in sorted(sessions.items()):
            yield from _trajectory(session, rows)

    for r in records:
        if r.kind != "agents" or not r.body.get("session") or r.body.get("type") not in ("user", "assistant"):
            continue
        if r.provenance.source != current and lines_of:
            yield from done(lines_of)
            lines_of = []
        current = r.provenance.source
        lines_of.append(r)
    if lines_of:
        yield from done(lines_of)


def _trajectory(session: str, lines: list[StoreRecord]) -> Iterable[SolverTrajectory]:
    lines.sort(key=lambda r: (r.at, r.id))
    results: dict[str, Any] = {}
    problem = None
    for r in lines:
        content = ((r.body.get("message") or {}).get("content"))
        if r.body.get("type") == "user":
            if isinstance(content, str) and problem is None:
                problem = content
            for blk in content if isinstance(content, list) else []:
                if isinstance(blk, dict) and blk.get("type") == "tool_result":
                    results[str(blk.get("tool_use_id"))] = blk.get("content")
                elif isinstance(blk, dict) and blk.get("type") == "text" and problem is None:
                    problem = blk.get("text")
    steps, thought = [], None
    for r in lines:
        content = ((r.body.get("message") or {}).get("content"))
        if r.body.get("type") != "assistant" or not isinstance(content, list):
            continue
        for blk in content:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") == "thinking" and blk.get("thinking"):
                thought = (thought + "\n\n" if thought else "") + str(blk["thinking"])
            elif blk.get("type") == "tool_use":
                steps.append(TrajectoryStep(n=len(steps), thought=thought, thought_source="recorded" if thought else None,
                                            no_thought=None if thought else "no thinking before this call in the transcript",
                                            tool=str(blk.get("name") or "")[:120], arguments=blk.get("input") if isinstance(blk.get("input"), dict) else {},
                                            result=results.get(str(blk.get("id")))))
                thought = None
            elif blk.get("type") == "text" and blk.get("text"):
                steps.append(TrajectoryStep(n=len(steps), thought=thought, thought_source="recorded" if thought else None,
                                            no_thought=None if thought else "no thinking before this answer in the transcript",
                                            tool=None, result=blk.get("text")))
                thought = None
    if not steps:
        return
    first = lines[0]
    prov = first.provenance
    more = list({r.provenance.entry: r.provenance for r in lines[1:] if r.provenance.entry != prov.entry}.values())
    yield SolverTrajectory(id=f"m5_solver_llm:claude_code:{session}", model_type="m5_solver_llm", split=first.split or split_of(session),
                           provenance=prov, more=more, problem_kind="development",
                           problem={"request": problem, "where": first.body.get("where"), "agent": "claude_code"}, steps=steps,
                           metric=None, score=None)  # development: no metric of its own (its fixes are judged in M7)


def intern_trajectories(records: Iterable[StoreRecord]) -> Iterable[SolverTrajectory]:
    """Each intern session's last saved version as a trajectory: its task, every step's tool, arguments and result."""
    newest: dict[str, StoreRecord] = {}
    for r in records:
        if r.type == "intern_session" and r.body.get("id"):
            sid = str(r.body["id"])
            if sid not in newest or (len(r.body.get("steps") or []), r.at) >= (len(newest[sid].body.get("steps") or []), newest[sid].at):
                newest[sid] = r
    for sid, r in sorted(newest.items()):
        steps = []
        for n, st in enumerate(r.body.get("steps") or []):
            if not isinstance(st, dict):
                continue
            thought = st.get("thought") or st.get("plan")
            steps.append(TrajectoryStep(n=n, thought=str(thought) if thought else None, thought_source="recorded" if thought else None,
                                        no_thought=None if thought else "the intern's step has no thought of its own", tool=(st.get("tool") or None) and str(st["tool"])[:120],
                                        arguments=st.get("arguments") if isinstance(st.get("arguments"), dict) else {}, result=st.get("result") or st.get("reply"),
                                        ok=st.get("ok") is not False))
        if steps:
            yield SolverTrajectory(id=f"m5_solver_llm:intern:{sid}", model_type="m5_solver_llm", split=r.split or split_of(f"session:{sid}"),
                                   provenance=r.provenance, problem_kind="studio", problem={"request": r.body.get("task"), "agent": "intern"}, steps=steps)


def build(layout: Optional[Layout] = None) -> dict[str, Any]:
    from .. import shapers

    from . import allowed, standing

    layout = layout or Layout()
    choices = standing(layout)
    taught = teacher_reasonings(layout)
    sealed, benchmarks = sealed_texts(), benchmark_names()
    sets: dict[tuple[str, str], list[Any]] = defaultdict(list)
    out: dict[str, int] = defaultdict(int)

    guard = Contamination(sealed, benchmarks)

    def add(model_type: str, folder: str, example: Any) -> None:
        why = guard.why(texts_of(example))
        if why:
            if example.split != "sealed":
                out[f"kept out of train: {why}"] += 1
                return
        sets[(model_type, folder)].append(example)

    for r in shapers.records(layout, "model_requests"):
        if not allowed(r, "m4_workflow_llm", choices):
            continue
        if r.provenance.source.startswith("model_exchanges/"):
            ex = from_exchange(r, taught)
            if ex is not None:
                add("m4_workflow_llm", ex.stage, ex)
    newest: dict[tuple[str, str], StoreRecord] = {}
    def version(r: StoreRecord) -> int:
        tail = r.provenance.entry.rsplit(":v", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    for r in shapers.records(layout, "conversations"):
        if r.provenance.source.endswith("/draft.json") and allowed(r, "m4_workflow_llm", choices):
            key = (r.provenance.workspace, r.provenance.source)
            if key not in newest or version(r) > version(newest[key]):  # by its version number, never by the id's text
                newest[key] = r
    for r in newest.values():
        ex = from_draft(r)
        if ex is not None:
            add("m4_workflow_llm", "understand_problem", ex)
    for t in trajectories(r for r in shapers.records(layout, "agents") if allowed(r, "m5_solver_llm", choices)):
        add("m5_solver_llm", "trajectories/development", t)
    for t in intern_trajectories(r for r in shapers.records(layout, "conversations") if allowed(r, "m5_solver_llm", choices)):
        add("m5_solver_llm", "trajectories/studio", t)
    report = {"sets": {}, "kept_out": dict(out)}
    abouts = {"m4_workflow_llm": "The workflow LLM's examples of one stage: messages as sent, each assistant turn with its labelled reasoning.",
              "m5_solver_llm": "Whole solving runs: each step's recorded thought, its tool call and the call's result."}
    for (model_type, folder), examples in sorted(sets.items()):
        sources = {e.provenance.entry: {"entry": e.provenance.entry, "sha256": e.provenance.sha256} for e in examples}
        m = write_set(layout, model_type, folder, examples, sources=sources.values(), about=abouts[model_type],
                      decontaminated=["every sealed journey and conversation set", f"{len(benchmarks)} benchmark task names (benchmarks.txt)"])
        report["sets"][f"{model_type}/{folder}"] = m.counts
    return report


__all__ = ["build", "from_exchange", "from_draft", "trajectories", "intern_trajectories", "contaminated", "Contamination", "sealed_texts", "STAGES"]
