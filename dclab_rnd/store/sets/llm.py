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

from dclab_core.datasets import group_of, split_of
from dclab_core.store import REASONING_SOURCES, ChatExample, Message, PreferencePair, Provenance, SolverTrajectory, StoreRecord, TrajectoryStep

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
        out += [m.content or "", m.reasoning or "", json.dumps(m.tool_calls or [], ensure_ascii=False, default=str)]  # a call's arguments are words too
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


def earlier_reasoning(message: dict[str, Any]) -> str:
    """An earlier assistant turn's reasoning as the agent sent it back (D2 / P4): the OpenAI-compatible field the tier names
    (reasoning_content or reasoning, P1), or Anthropic's thinking blocks; "" when the transport returned none."""
    for key in ("reasoning_content", "reasoning"):
        if isinstance(message.get(key), str) and message[key].strip():
            return message[key]
    blocks = message.get("thinking_blocks")
    if isinstance(blocks, list):
        return "\n\n".join(str(b.get("thinking") or "") for b in blocks if isinstance(b, dict) and b.get("type") == "thinking").strip()
    return ""


def earlier_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """An earlier assistant turn's tool calls in the parsed shape ({"id", "name", "arguments": dict}), from the wire shape the
    agent sent back (a JSON string or, for a tier set to objects, a dict); arguments that are not JSON stay under _raw. The
    arguments were one string under a key the shaper does not read as words, so they are redacted here, every string."""
    from .. import shapers

    out = []
    for i, call in enumerate(message.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else None
        name, arguments = (function or call).get("name"), (function or call).get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        arguments = shapers._redact(arguments if isinstance(arguments, dict) else {}, every=True)
        out.append({"id": str(call.get("id") or f"call_{i}")[:120], "name": str(name or ""), "arguments": arguments})
    return out


def all_recorded(example: Any) -> bool:
    """The reasoning-only sets' rule (D2 / P4): every assistant turn of a chat example, or every tool step of a trajectory, carries
    reasoning recorded where it was made; a visible, a rule's or a teacher's reasoning is not that, and an empty one never is."""
    messages = getattr(example, "messages", None)
    if messages is not None:
        turns = [m for m in messages if m.role == "assistant"]
        return bool(turns) and all(m.reasoning_source == "recorded" and bool(m.reasoning) for m in turns)
    steps = [s for s in getattr(example, "steps", None) or [] if s.tool]
    return bool(steps) and all(s.thought_source == "recorded" and bool(s.thought) for s in steps)


def from_exchange(record: StoreRecord, taught: Optional[dict[str, str]] = None) -> Optional[ChatExample]:
    """One kept exchange as an M4 example: its messages (each earlier assistant turn with the reasoning and the tool calls the
    agent sent back, each tool turn with the call it answers), the tools offered, and the reply with its thinking as the reasoning."""
    b = record.body
    if not b.get("whole") or not isinstance(b.get("messages"), list):
        return None
    messages = []
    for m in b["messages"]:
        role = _role(m)
        if role == "assistant":
            earlier = earlier_reasoning(m)
            labelled = ({"reasoning": earlier, "reasoning_source": "recorded"} if earlier else
                        {"no_reasoning": "an earlier turn: the transport returned no reasoning for it, or it was not sent back"})
            messages.append(Message(role=role, content=_content(m), tool_calls=earlier_calls(m), **labelled))
        elif role == "tool":
            messages.append(Message(role=role, content=_content(m), name=m.get("name") if isinstance(m.get("name"), str) else None,
                                    tool_call_id=str(m["tool_call_id"])[:120] if m.get("tool_call_id") else None))
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
                       tools=[t for t in b.get("tools") or [] if isinstance(t, dict)],  # P1 keeps the schemas offered with a whole exchange
                       weight=0.5 if (taught and not thinking) else 1.0, check={"outcome": b.get("outcome"), "model": b.get("model")})


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
        for r in lines:  # a subagent's transcript carries its parent's session id: it is its own run, in its parent's split
            m = re.search(r"/subagents/([^/]+)\.jsonl$", r.provenance.source)
            sessions[f"{r.body['session']}:{m.group(1)}" if m else str(r.body["session"])].append(r)
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
    yield SolverTrajectory(id=f"m5_solver_llm:claude_code:{session}", model_type="m5_solver_llm", split=first.split or split_of(session.split(":", 1)[0]),
                           provenance=prov, more=more, problem_kind="development",
                           problem={"request": problem, "where": first.body.get("where"), "agent": "claude_code"}, steps=steps,
                           metric=None, score=None)  # development: no metric of its own (its fixes are judged in M7)


TWIN_LANE = re.compile(r"^(?:(?P<parent>.+)__)?twin__runs__(?P<run>.+)__(?P<lane>workflow|ai_first)$")
ARM = {"workflow": "system", "ai_first": "ai_first"}
OLLAMA_TAG = re.compile(r"^[a-z0-9][a-z0-9._/-]*:[a-z0-9][a-z0-9._-]*$")  # qwen2.5-coder:1.5b: a model pulled and served on this machine
LOCAL_PREFIXES = ("ollama/", "local/", "local:", "tuned:", "tuned/", "dclab-",
                  "mlx-community/")  # an MLX build runs on this machine only (D2 / P4; a vLLM name like Qwen/... is also a hosted provider's: the endpoint's host is the follow-up)
PROVIDERS = {"groq", "google", "gemini", "mistral", "openrouter", "together", "fireworks", "anthropic", "openai", "deepseek", "cohere",
             "novita", "azure", "bedrock", "vertex", "hf", "huggingface", "xai", "perplexity", "replicate", "nvidia", "cerebras", "sambanova"}
FINAL_TOOLS = ("run_all",)  # run_programme writes no stages/final.json: its score is the programme's, not this one
SLACK = 5.0
OUTCOME_NAMED = "The outcome to predict is the column"  # the phrase of twin/solver_runs.py's task (kept here: the builder imports no runner)


def answerer_of(session: dict[str, Any]) -> str:
    """Who chose the session's moves: ``rules`` (the standard plan), ``local:<model>``, or a provider's model (``claude:``, ``openai:``,
    ``frontier:``). A model counts as local only on its name's evidence (an Ollama tag, a local or tuned prefix); every other name,
    and a model-mode session without one, is a provider's: never clean by default."""
    model = str(session.get("model") or "").strip()
    if session.get("mode") == "standard" and not model:
        return "rules"
    low = model.lower()
    if not model:
        return "frontier:unknown"
    if "claude" in low:
        return f"claude:{model}"
    if low.startswith(("gpt", "o1", "o3", "o4")) or "openai" in low:
        return f"openai:{model}"
    if low.startswith(LOCAL_PREFIXES):
        return f"local:{model}"
    # an Ollama tag (qwen2.5-coder:1.5b): no path, and the part before the colon is not a provider (groq:llama-3 is hosted)
    if OLLAMA_TAG.match(low) and "/" not in low and low.split(":", 1)[0] not in PROVIDERS:
        return f"local:{model}"
    return f"frontier:{model}"


def _when(text: Any) -> Optional[float]:
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def final_windows(steps: Iterable[dict[str, Any]]) -> list[tuple[float, float]]:
    """When the session itself ran the final stage: each ok call to run_all or run_stage(final), from its start (its time less
    its seconds) to its end, a few seconds of slack each side. A score is the session's only if its final completed inside one."""
    out = []
    for st in steps:
        if not isinstance(st, dict) or st.get("ok") is False:
            continue
        args = st.get("arguments") if isinstance(st.get("arguments"), dict) else {}
        if st.get("tool") in FINAL_TOOLS or (st.get("tool") == "run_stage" and args.get("stage") == "final"):
            end = _when(st.get("at"))
            if end is not None:
                out.append((end - float(st.get("elapsed_seconds") or 0) - SLACK, end + SLACK))
    return out


def models_in_results(steps: Iterable[dict[str, Any]]) -> list[str]:
    """The models a step's tool used inside the session (a leakage review or an analyst answered by a model): their text sits in
    the trajectory, so they count for who made it."""
    found = []
    for st in steps:
        result = st.get("result") if isinstance(st, dict) else None
        if isinstance(result, dict):
            model = result.get("model") or result.get("served_by")
            if isinstance(model, str) and model.strip():
                found.append(model.strip())
            elif result.get("mode") == "model":
                found.append("unknown")
    return found


def clean(answerer: str) -> bool:
    return answerer == "rules" or answerer.startswith("local:")


def holdouts(records: Iterable[StoreRecord]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Every version of each project's final stage, by (workspace, project): the holdout's metric and score, the table it ran on,
    when it completed and its record's split (the store keeps every version, a cleared and rerun final included)."""
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        m = re.search(r"(?:^|/)projects/([^/]+)/stages/final\.json$", r.provenance.source)
        project = m.group(1) if m else None
        if project is None and r.provenance.source.startswith("db/stage_record") and r.body.get("stage") == "final":  # PostgreSQL: a row
            project = r.body.get("project_id") or r.body.get("project")
        if not project or r.body.get("status") not in (None, "completed"):
            continue
        run = r.body.get("run") if isinstance(r.body.get("run"), dict) else {}
        outputs, inputs, evidence = run.get("outputs") or {}, run.get("inputs") or {}, r.body.get("evidence") or {}
        value = outputs.get("value")
        if value is None:
            value = (evidence.get("holdout_primary_metric_ci") or {}).get("point")
        if value is None or evidence.get("holdout_consumed") is False:
            continue
        out[(r.provenance.workspace, str(project))].append({
            "metric": outputs.get("metric") or r.body.get("primary_metric"), "score": float(value), "data_sha256": inputs.get("data_sha256"),
            "task_type": r.body.get("task_type"), "holdout_rows": evidence.get("holdout_rows"), "completed_at": _when(r.body.get("completed_at")),
            "split": "sealed" if "sealed" in (r.split, split_of(group_of(project=str(project)))) else r.split,  # its project's split
            "record": f"{r.provenance.workspace}:{r.provenance.source}"})
    return dict(out)


def twin_truths(records: Iterable[StoreRecord]) -> dict[tuple[str, str], dict[str, Any]]:
    """Each blind twin run's judged columns step, by (the run's workspace, run id): the planted truth and each arm's verdict."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in records:
        m = re.search(r"(?:^|/)twin/runs/([^/]+)/steps\.jsonl$", r.provenance.source)
        if m and r.body.get("step") == "solution.columns" and isinstance(r.body.get("verdict"), dict):
            out[(r.provenance.workspace, m.group(1))] = r.body
    return out


def truth_of(workspace: str, truths: dict[tuple[str, str], dict[str, Any]]) -> tuple[str, Optional[str]]:
    """``passed``, ``failed``, ``none`` (not a twin lane: no planted truth to judge), or ``unknown``: a twin lane whose verdict is not
    found (its name cut, its run stopped before the judge, its steps not kept). ``unknown`` is never placed: an unchecked lane is
    not a checked one."""
    m = TWIN_LANE.match(workspace)
    if not m:
        return ("unknown", None) if "twin__runs__" in workspace else ("none", None)
    step = truths.get((m.group("parent") or "dev", m.group("run")))
    fields = ((step or {}).get("verdict") or {}).get(ARM[m.group("lane")])
    if not isinstance(fields, dict) or not fields:
        return "unknown", None
    return ("passed" if all(v is True for v in fields.values()) else "failed"), f"twin:{m.group('run')}:solution.columns"


LEARNING_INVALID = "m5: a learning attempt made on a broken setup (its folder is marked invalid)"


def graded_attempts(records: Iterable[StoreRecord]) -> dict[str, dict[str, Any]]:
    """H3 (2026-10-24): each learning attempt's grade by the challenge grader (``grade.json``: its verdict and codes, which never
    name a column) and its invalid mark (``invalid.json``: made on a broken setup, never a sample of the model's skill), by its
    workspace. A learning attempt's planted truth is its grade, as a twin lane's is its verdict."""
    out: dict[str, dict[str, Any]] = {}
    for r in records:
        if r.provenance.source == "grade.json" and isinstance(r.body.get("grade"), dict):
            g = r.body["grade"]
            out.setdefault(r.provenance.workspace, {})["grade"] = {"verdict": g.get("verdict"), "codes": list(g.get("codes") or []),
                                                                     "grader": g.get("grader"), "decision_score": g.get("decision_score")}
        elif r.provenance.source == "invalid.json":
            out.setdefault(r.provenance.workspace, {})["invalid"] = str(r.body.get("why") or "a broken setup")[:500]
        elif r.provenance.source == "tools.json" and isinstance(r.body.get("tools"), list):  # the schemas the solver was offered
            out.setdefault(r.provenance.workspace, {})["tools"] = [t for t in r.body["tools"] if isinstance(t, dict)]
    return out


def intern_trajectories(records: Iterable[StoreRecord], finals: Optional[dict[tuple[str, str], dict[str, Any]]] = None,
                        truths: Optional[dict[tuple[str, str], dict[str, Any]]] = None,
                        graded: Optional[dict[str, dict[str, Any]]] = None) -> Iterable[SolverTrajectory]:
    """Each intern session's last saved version as a trajectory: its task, every step's tool, arguments and whole result (a session
    from before whole results were kept has its summary, and says so); who chose its moves; its holdout score and the table it ran
    on (its split: one table, one split); its planted truth's verdict where one was judged."""
    finals, truths, graded = finals or {}, truths or {}, graded or {}
    newest: dict[str, StoreRecord] = {}
    for r in records:
        if r.type == "intern_session" and r.body.get("id"):
            sid = str(r.body["id"])
            if sid not in newest or (len(r.body.get("steps") or []), r.at) >= (len(newest[sid].body.get("steps") or []), newest[sid].at):
                newest[sid] = r
    def holdout_of(r: StoreRecord) -> Optional[dict[str, Any]]:
        """The final this session itself ran: one that completed inside one of its own final calls (never a score it only read)."""
        project = str(r.body.get("project_id") or r.body.get("project") or "")
        windows = final_windows(r.body.get("steps") or []) if project else []
        mine = [f for f in finals.get((r.provenance.workspace, project), []) if f.get("completed_at") is not None
                and any(a <= f["completed_at"] <= b for a, b in windows)]
        return max(mine, key=lambda f: f["completed_at"]) if mine else None

    # one table, one split: a table any sealed session ran on, or any sealed final scored, is sealed for every session on it
    sealed_tables = {h["data_sha256"] for r in newest.values() if r.split == "sealed" for h in [holdout_of(r)] if h and h.get("data_sha256")}
    sealed_tables |= {f["data_sha256"] for versions in finals.values() for f in versions if f.get("split") == "sealed" and f.get("data_sha256")}
    for sid, r in sorted(newest.items()):
        steps, cut = [], False
        for n, st in enumerate(r.body.get("steps") or []):
            if not isinstance(st, dict):
                continue
            thought = st.get("thought") or st.get("plan")
            whole = "result" in st
            cut = cut or not whole  # a session from before whole results were kept: its summary quotes the table's cells, so it is not used
            source = st.get("thought_source") if st.get("thought_source") in REASONING_SOURCES else "recorded"  # D2: the step says what filled its slot
            steps.append(TrajectoryStep(n=n, thought=str(thought) if thought else None, thought_source=source if thought else None,
                                        no_thought=None if thought else "the intern's step has no thought of its own", tool=(st.get("tool") or None) and str(st["tool"])[:120],
                                        arguments=st.get("arguments") if isinstance(st.get("arguments"), dict) else {},
                                        result=st.get("result") if whole else None, ok=st.get("ok") is not False))
        if not steps:
            continue
        holdout = holdout_of(r)
        who = answerer_of(r.body)
        helpers = [m for m in models_in_results(r.body.get("steps") or []) if not clean(answerer_of({"mode": "llm", "model": m}))]
        if clean(who) and helpers:  # a hosted model wrote part of its results: not a clean run
            who = f"mixed:{who}+{answerer_of({'mode': 'llm', 'model': helpers[0]})}"
        truth, truth_source = truth_of(r.provenance.workspace, truths)
        learning = graded.get(r.provenance.workspace)
        if not learning and "learning_attempts__" in r.provenance.workspace:  # an attempt whose grade was never written: not judged
            learning = {"ungraded": True}
        if learning:  # a learning attempt: graded by the challenge grader, or marked invalid (a broken setup)
            grade = learning.get("grade") or {}
            truth = "invalid" if learning.get("invalid") else ("passed" if grade.get("verdict") == "pass" else "failed") if grade else "unknown"  # no grade: never placed
            truth_source = f"learning:{grade.get('grader') or 'grader'}"
        table = (holdout or {}).get("data_sha256")
        lane = TWIN_LANE.match(r.provenance.workspace)
        # a run with no holdout of its own (a failed run, P3) still ran on its project's table: any final of the project names it,
        # for the split only (its steps describe that table, so a sealed table's run is sealed); a project with no final at all
        # cannot be tied to a table here, and its split follows its session
        project = str(r.body.get("project_id") or r.body.get("project") or "")
        known = {f.get("data_sha256") for f in finals.get((r.provenance.workspace, project), [])} - {None}
        split = ("sealed" if (r.split == "sealed" or table in sealed_tables or (known & sealed_tables))
                 else (split_of(f"table:{table}") if table else (r.split or split_of(f"session:{sid}"))))
        yield SolverTrajectory(
            id=f"m5_solver_llm:intern:{sid}", model_type="m5_solver_llm", split=split,
            provenance=r.provenance, problem_kind=str((holdout or {}).get("task_type") or "studio"), weight=1.0 if clean(who) else 0.5,
            problem={"request": r.body.get("task"), "agent": "intern", "answerer": who, "data_sha256": table,
                     "status": r.body.get("status"), "ended": r.body.get("ended"),  # how the session ended (a failed run says so)
                     "outcome_named": OUTCOME_NAMED in str(r.body.get("task") or ""),  # the person named the outcome: not the solver's choice
                     "twin": {"run": lane.group("run"), "lane": lane.group("lane")} if lane else None,
                     "learning": {k: v for k, v in (learning or {}).items() if k != "tools"} or None},  # its grade (codes, never a column) or why invalid
            tools=list((learning or {}).get("tools") or []),
            steps=steps, metric=(holdout or {}).get("metric"), score=(holdout or {}).get("score"),
            submission={"holdout": holdout["record"], "holdout_rows": holdout.get("holdout_rows"), "completed_at": holdout.get("completed_at"),
                        "truth": truth, "truth_source": truth_source,
                        "results_cut": cut} if holdout else None)


def keep_sealed(layout: Any, model_type: str, folder: str, version: str, examples: list[Any]) -> list[Any]:
    """Sealed once, sealed always: an example that spans a raw entry an earlier version of the set sealed stays sealed in this
    one (a new version changes the split rule, never what was already held back)."""
    n = int(version.lstrip("v") or 1)
    sealed: set[str] = set()
    for k in range(1, n):
        base = layout.sets(model_type, *folder.split("/"), f"v{k}")
        for name in ("sealed.jsonl", "sealed.pending.jsonl"):
            path = base / name
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        sealed.update(p.get("entry") for p in [row.get("provenance") or {}, *(row.get("more") or [])] if p.get("entry"))
    if not sealed:
        return examples
    return [e if e.split == "sealed" or not ({e.provenance.entry, *(m.entry for m in e.more)} & sealed) else e.model_copy(update={"split": "sealed"})
            for e in examples]


SELFPLAY_WS = re.compile(r"^selfplay__ledgers__(?P<kind>[a-z0-9_]+?)__(?P<seed>\d+)$")
LABEL_RANK = {"benchmark": 0, "measured": 1, "user_decision_exposed": 2, "blind": 3}  # the planted truth first; a model's never


NO_OUTCOME_KINDS = frozenset({"segmentation", "anomaly"})  # self-play kinds that learn no outcome: groups, a review list


def selfplay_problem(kind: str, seed: str, group: str) -> dict[str, Any]:
    """What a self-play trajectory says it was asked: an outcome named, except for a kind that has none to learn."""
    named = kind not in NO_OUTCOME_KINDS
    said = "its outcome and its prediction moment are stated" if named else "it has no outcome to learn; its task and its moment are stated"
    return {"request": f"Solve the invented {kind} problem (seed {seed}): {said}.", "agent": "engine", "answerer": "rules",
            "origin": "verified_synthetic", "outcome_named": named, "group": group}


def selfplay_verified(records: Iterable[StoreRecord]) -> dict[str, bool]:
    """Each self-play solve's group (``selfplay:<kind>:<seed>``) and whether its kind's acceptance decisions all passed."""
    out: dict[str, bool] = {}
    for r in records:
        if r.provenance.workspace == "selfplay" and r.provenance.source == "done.jsonl" and r.body.get("group"):
            out[str(r.body["group"])] = bool(r.body.get("verified"))
    return out


def selfplay_failures(records: Iterable[StoreRecord]) -> dict[str, dict[str, Any]]:
    """D2 / P3: what each failed self-play solve missed (``failed_solves.jsonl``: the checks, the decisions missed, or the error
    that broke it), by its group, for the failed trajectory's submission. The solve itself stays in the record, whole."""
    out: dict[str, dict[str, Any]] = {}
    for r in records:
        if r.provenance.workspace == "selfplay" and r.provenance.source == "failed_solves.jsonl" and r.body.get("group"):
            out[str(r.body["group"])] = {k: r.body.get(k) for k in ("missed", "reason", "checks") if r.body.get(k) is not None}
    return out


def selfplay_trajectories(records: Iterable[StoreRecord], verified: dict[str, bool]) -> Iterable[SolverTrajectory]:
    """27.3's verified solves as trajectories: every decision the engine made on an invented problem, in order (its state as
    bands, never a value; the options; the answer used; the planted or measured label), the holdout's outcome as its score.
    A solve its acceptance decisions did not all pass is not yielded (the caller counts it)."""
    solves: dict[str, dict[str, Any]] = defaultdict(lambda: {"decisions": [], "labels": {}, "outcomes": []})
    for r in records:
        m = SELFPLAY_WS.match(r.provenance.workspace)
        if not m:
            continue
        solve = solves[f"selfplay:{m.group('kind')}:{m.group('seed')}"]
        if r.type == "decision":
            solve["decisions"].append(r)
        elif r.type == "label" and r.body.get("record"):
            old = solve["labels"].get(str(r.body["record"]))
            if old is None or LABEL_RANK.get(str(r.body.get("source")), 9) < LABEL_RANK.get(str(old.get("source")), 9):
                solve["labels"][str(r.body["record"])] = r.body
        elif r.type == "outcome" and r.body.get("observed") == "holdout":
            solve["outcomes"].append(r.body)
    for group, solve in sorted(solves.items()):
        if group not in verified or not solve["decisions"]:  # a solve never tried to the end: nothing to say about it
            continue
        rows = sorted(solve["decisions"], key=lambda r: (r.at, r.id))
        steps = []
        for n, r in enumerate(rows):
            b = r.body
            label = solve["labels"].get(str(b.get("id")))
            reason = next((a.get("reason") for a in b.get("answers") or [] if isinstance(a, dict) and a.get("reason")), None)
            steps.append(TrajectoryStep(n=n, thought=str(reason) if reason else None, thought_source="from_record" if reason else None,
                                        no_thought=None if reason else "the engine's rule decided; it wrote no reason",
                                        tool=f"decide:{str(b.get('point'))[:100]}",
                                        arguments={"subject": b.get("subject"), "state": b.get("state"), "options": b.get("options")},
                                        result={"answer": b.get("value_used"), "by": b.get("used"),
                                                "label": {"value": label.get("value"), "source": label.get("source"), "by": label.get("by")} if label else None},
                                        ok=label is None or label.get("value") == b.get("value_used")))
        outcome = solve["outcomes"][0] if solve["outcomes"] else {}
        _, kind, seed = group.split(":")
        first = rows[0]
        more = list({x.provenance.entry: x.provenance for x in rows[1:] if x.provenance.entry != first.provenance.entry}.values())
        split = "sealed" if any(x.split == "sealed" for x in rows) else (first.split or split_of(group))
        yield SolverTrajectory(
            id=f"m5_solver_llm:selfplay:{kind}:{seed}", model_type="m5_solver_llm", split=split, provenance=first.provenance, more=more,
            problem_kind=kind, steps=steps, metric=outcome.get("metric"), score=outcome.get("value"),
            problem=selfplay_problem(kind, seed, group),
            submission={"verified": bool(verified.get(group)), "interval": outcome.get("interval"), "reference": outcome.get("reference"),
                        "labelled_steps": sum(1 for s in steps if s.result and s.result.get("label")),
                        "steps_against_label": sum(1 for s in steps if not s.ok)})


def selfplay_pairs(t: SolverTrajectory) -> Iterable[PreferencePair]:
    """Step 4: each self-play decision whose answer the planted or measured truth contradicts, as a pair: the label chosen over
    the engine's answer, on the decision as it was posed (its state as bands and its options). From failed solves too: the label
    is true either way. Never a model's answer as the chosen one."""
    for s in t.steps:
        label = (s.result or {}).get("label") if isinstance(s.result, dict) else None
        if s.ok or not label or label.get("source") not in ("benchmark", "measured"):
            continue
        point = str(s.tool or "").removeprefix("decide:")
        ask = Message(role="user", content=json.dumps({"problem": t.problem_kind, "decide": point, **(s.arguments or {})}, ensure_ascii=False, default=str))
        yield PreferencePair(id=f"{t.id}:pair:{s.n}", model_type="m5_solver_llm", split=t.split, provenance=t.provenance, more=t.more,
                             task=f"selfplay:{point}"[:120], prompt=[ask],
                             chosen=Message(role="assistant", content=json.dumps(label.get("value"), default=str),
                                            no_reasoning=f"the {label['source']} truth ({label.get('by') or 'planted'})"),
                             rejected=Message(role="assistant", content=json.dumps(s.result.get("answer"), default=str),
                                              no_reasoning="the engine's rule answer"),
                             decided_by=label["source"])


def newest_per_table(placed: list[tuple[str, SolverTrajectory]]) -> Iterable[tuple[str, SolverTrajectory, bool]]:
    """One solve per table and solver: the newest (by when its final completed). A rerun reflects the solver as it is now,
    and a table solved ten times the same way would weigh ten times; the older solves are counted, not placed."""
    def key(t: SolverTrajectory) -> tuple[str, str]:
        return str(t.problem.get("data_sha256")), str(t.problem.get("answerer"))

    def when(t: SolverTrajectory) -> float:
        return float((t.submission or {}).get("completed_at") or 0.0)

    best: dict[tuple[str, str], SolverTrajectory] = {}
    for _folder, t in placed:
        if key(t) not in best or (when(t), t.id) > (when(best[key(t)]), best[key(t)].id):
            best[key(t)] = t
    for folder, t in placed:
        yield folder, t, best[key(t)] is t


FAILED = "trajectories/failed"  # D2 / P3: runs that went wrong, kept whole with why; never mixed into studio
REASONING = "reasoning"  # D2 / P4: M4's reasoning-only set (every assistant turn's reasoning recorded)
TRAJECTORIES_REASONING = "trajectories/reasoning"  # D2 / P4: M5's (every tool step's thought recorded), from studio only


def failure_of(t: SolverTrajectory) -> str:
    """Why a run is not a clean, verified solve ("" when it is): never scored on its holdout (the session's own ending named
    when it did not complete), failed its planted truth, or its truth was not judged."""
    if (t.problem.get("learning") or {}).get("invalid"):
        return LEARNING_INVALID
    if t.score is None or not t.submission:
        status = str(t.problem.get("status") or "")
        return "m5: not scored on its holdout" + (f" (the session ended {status})" if status and status != "completed" else "")
    if t.submission.get("truth") == "failed":
        return "m5: failed its planted truth"
    if t.submission.get("truth") == "unknown":
        return "m5: its planted truth was not judged"
    return ""


def as_failed(t: SolverTrajectory, why: str, **facts: Any) -> SolverTrajectory:
    """The trajectory marked as a failed run: its submission keeps what it had and says why (``verified`` false), so "what went
    wrong" is read beside the steps that led there."""
    return t.model_copy(update={"submission": {**(t.submission or {}), "verified": False, "why": why, **facts}})


def place(t: SolverTrajectory) -> tuple[Optional[str], str]:
    """Where an intern trajectory goes: the clean set, the frontier's silver set, or (D2 / P3) the failed set with why (a
    frontier model's failed run goes to silver with the same why: never to a training folder). Nowhere only when the table
    it ran on is not known, since its split could not follow the table; the reason comes back with None."""
    silver = not clean(str(t.problem.get("answerer")))
    why = failure_of(t)
    if why:  # a failed run often has no holdout, so no table: its split followed its session instead
        return ("trajectories/silver_frontier" if silver else FAILED), why
    if not t.problem.get("data_sha256"):
        return None, "m5: the table it ran on is not known"
    return ("trajectories/silver_frontier" if silver else "trajectories/studio"), ""


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
                add("m4_workflow_llm", ex.stage, ex)  # the reasoning-only set is derived from the stage sets below, after their sealed rows are carried
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
    verified = selfplay_verified(shapers.records(layout, "other"))
    failures = selfplay_failures(shapers.records(layout, "other"))
    for t in selfplay_trajectories((r for r in shapers.records(layout, "decisions") if r.provenance.workspace.startswith("selfplay__")
                                    and allowed(r, "m5_solver_llm", choices)), verified):
        if t.submission and t.submission.get("verified"):
            add("m5_solver_llm", "trajectories/selfplay", t)
        else:  # D2 / P3: a solve that did not pass its checks is kept whole, with what it missed, beside the verified ones
            facts = failures.get(t.id.removeprefix("m5_solver_llm:"), {})
            add("m5_solver_llm", FAILED, as_failed(t, "m5: a self-play solve that did not pass its checks", **facts))
        for pair in selfplay_pairs(t):
            add("m5_solver_llm", "preferences/selfplay", pair)
    if verified:
        out["m5: a self-play solve that did not pass its checks (kept in trajectories/failed)"] += sum(1 for v in verified.values() if not v)
    finals = holdouts(shapers.records(layout, "workflows"))
    truths = twin_truths(r for r in shapers.records(layout, "conversations") if r.provenance.source.endswith("steps.jsonl"))
    graded = graded_attempts(shapers.records(layout, "other"))
    placed = []
    for t in intern_trajectories((r for r in shapers.records(layout, "conversations") if allowed(r, "m5_solver_llm", choices)), finals, truths,
                                 graded):
        folder, why = place(t)
        if folder is None:
            out[why] += 1
            continue
        if why:  # D2 / P3: a run that went wrong, kept whole with why; it never competes with a verified solve of its table
            add("m5_solver_llm", folder, as_failed(t, why))
            out[f"{why} (kept in {folder})"] += 1
            continue
        placed.append((folder, t))
    for folder, t, newest in newest_per_table(placed):
        if newest:
            add("m5_solver_llm", folder, t)  # the reasoning-only set is derived from studio below, after its sealed rows are carried
        else:
            out["m5: superseded by a newer solve of its table by the same solver"] += 1
    report = {"sets": {}, "kept_out": dict(out)}
    abouts = {"m4_workflow_llm": "The workflow LLM's examples of one stage: messages as sent, each assistant turn with its labelled reasoning.",
              "m5_solver_llm": "Whole solving runs: each step's recorded thought, its tool call and the call's result."}
    folder_abouts = {
        "trajectories/studio": "Whole solving runs whose moves the rules or a local model chose, each scored once on its holdout (its metric "
                               "and score), on the table named by its SHA-256 (one table, one split), and passed its planted truth where one "
                               "was judged (submission.truth: passed or none). The clean set for training.",
        "trajectories/silver_frontier": "Whole solving runs whose moves a provider's model (Claude, GPT) chose, scored and checked as the clean "
                                        "set, at weight 0.5. The owner's rule (2026-10-21): evaluation by default; NOT FOR TRAINING until the "
                                        "owner records that the provider's terms were checked. Its manifest's consent is improve, not train: every export must check it.",
        "trajectories/selfplay": "27.3's verified solves of invented problems: every decision the engine made in order (state as bands, the "
                                 "options, the answer, the planted or measured label), the holdout outcome as the score. Clean (the rules); "
                                 "the replicas' column names are kept (the lake also masks them): a model may learn their shapes.",
        "preferences/selfplay": "Kept against refused: each self-play decision the planted (benchmark) or measured truth contradicts, the "
                                "label chosen over the engine's answer, on the decision as posed (state as bands, options). Failed solves included.",
        "trajectories/development": "Claude Code sessions on this repository, each by its own session id: how DCLab is built, not an ML solve.",
        FAILED: "D2 / P3: runs that went wrong, kept as carefully as the verified ones: intern sessions never scored on their holdout "
                "(the session's own ending named), ones that failed their planted truth or were not judged, and self-play solves "
                "that did not pass their checks (what they missed). Each submission says why (verified: false). What went wrong is "
                "training data too; never mixed into studio, and a frontier model's failed run is in silver_frontier, not here. A run "
                "with no holdout is sealed when any final of its project ran on a sealed table; a project with no final at all cannot "
                "be tied to a table, so its split follows its session.",
        REASONING: "D2 / P4: the reasoning-only set of the workflow LLM: exchanges whose every assistant turn carries reasoning recorded "
                   "where it was made (the local model's reasoning field sent back on every turn, P1), answered by a clean model only "
                   "(a provider's model's exchanges are evaluation and silver until the owner's D11). Teaches why; the stage sets teach "
                   "tool use on every exchange. Never an empty, a visible, a rule's or a teacher's reasoning.",
        TRAJECTORIES_REASONING: "D2 / P4: the reasoning-only set of the solver: studio's clean, verified runs whose every tool step carries the "
                                "model's recorded thought (never the standard plan's lines, which are from_record). Teaches why; studio "
                                "teaches the moves on every verified run."}
    # v2 of the stage sets: each earlier assistant turn with its reasoning and tool calls, each tool turn with its call, the tools
    # offered (P4); studio v3: each step's thought with its source (P2), the failed runs apart (P3); development v2: each session by
    # its own id (v1's sealed rows were written once and are carried by keep_sealed)
    versions = {**{stage: "v2" for stage in set(STAGES.values()) | {"plan", "understand_problem"}},
                "trajectories/studio": "v3", "trajectories/development": "v2"}
    written: dict[tuple[str, str], list[Any]] = {}

    def write(model_type: str, folder: str, examples: list[Any]) -> None:
        examples = keep_sealed(layout, model_type, folder, versions.get(folder, "v1"), examples)
        written[(model_type, folder)] = examples
        sources = {e.provenance.entry: {"entry": e.provenance.entry, "sha256": e.provenance.sha256} for e in examples}
        m = write_set(layout, model_type, folder, examples, version=versions.get(folder, "v1"), sources=sources.values(),
                      consent="improve" if folder == "trajectories/silver_frontier" else "train",  # not for training until the terms are checked
                      about=folder_abouts.get(folder, abouts[model_type]),
                      decontaminated=["every sealed journey and conversation set", f"{len(benchmarks)} benchmark task names (benchmarks.txt)"])
        report["sets"][f"{model_type}/{folder}"] = m.counts

    for (model_type, folder), examples in sorted(sets.items()):
        write(model_type, folder, examples)
    # D2 / P4: the reasoning-only sets, derived from their parents after the parents' sealed rows are carried, so "sealed once,
    # sealed always" holds across folders (studio v1 split by session, v3 by table: a row sealed there is sealed here too); every
    # row passed the contamination guard on its way into the parent. M4's takes a clean answerer's exchanges only (a provider's
    # model's are evaluation and silver until the owner's D11; an exchange naming no model is not clean either).
    write("m4_workflow_llm", REASONING, [e for (mt, f), kept in written.items() if mt == "m4_workflow_llm" for e in kept
                                        if all_recorded(e) and clean(answerer_of({"mode": "llm", "model": e.check.get("model")}))])
    write("m5_solver_llm", TRAJECTORIES_REASONING, [e for e in written.get(("m5_solver_llm", "trajectories/studio"), []) if all_recorded(e)])
    return report


__all__ = ["build", "from_exchange", "from_draft", "trajectories", "intern_trajectories", "contaminated", "Contamination", "sealed_texts", "STAGES"]
