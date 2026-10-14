"""How Claude designs the solution and its workflow, and what it then did with the design (R&D only, package A6.7).

The AI-first lane of a twin run is not only run but watched. Before the intern starts, Claude is asked for a solution
design: its understanding of the problem, the data checks it plans, its leakage hypotheses, the split, the features,
the families, the evaluation, and the workflow that will produce the solution, as steps (each on a WF-01…WF-10 step,
with the tool it calls, why it comes there, the evidence it needs and the rule it will choose by) and routes back.
The design is handed to the agent with its task, and the agent is asked to name the design step before each tool call.

While the agent works, a ``Monitor`` sits on the agent's client (``ClaudeClient.observer``): before each request it
reads the steps that finished since the last one and writes them to the timeline, one row per tool call, with what
the agent said before it, the design step it named, the step of the design it matches, and whether it was in order,
out of order or not in the design. At each checkpoint (a stage approved: data, leakage and models by default) Claude
is asked to revise the design from that stage's evidence, listing every change with the number that made it; the
revision is handed to the agent (``steer``, on by default) as a message in its conversation. When the run ends the
last steps are read from the session, Claude writes a final revision with a retrospective, and the lane's
``design/`` folder gets everything:

| File | What it holds |
|---|---|
| ``design_vNN.json``, ``design_vNN.md`` | each version of the design, as Claude wrote it (parsed) and as a page |
| ``designs.jsonl`` | one row per design request: checkpoint, version, parsed or not, problems found, the raw answer |
| ``timeline.jsonl`` | the monitoring log: one row per tool call and one per design event, in order |
| ``adherence.json`` | the plan against the act: planned steps done in order, out of order, skipped; actions not in the design |
| ``intents.jsonl`` | the design's decisions on the decision points of A6.5 (``point:subject`` keys), per version |
| ``sft_candidates.jsonl`` | each parsed design as a chat sample with its outcome; Claude's words, ``verified: false`` |
| ``DESIGN.md`` | the story: the first design, each revision and its changes, the timeline, the adherence |

The design request (purpose ``twin_designer``) carries the table's description through the trace's scrub (no cell
examples, whatever the run's ``capture_cells``), the workflow's steps and the tool names. A checkpoint carries that stage's compact record
(the one the agent itself reads with ``get_results``), so no revision before the end sees a holdout score (the data
stage's drift note, which the agent reads too, compares feature distributions with the holdout); the final revision
sees the final stage's record, as the agent's report does, after nothing can be tuned any more. A design
request that fails or is refused by the run's caps is a timeline row, never a failure of the run. Claude's hidden
reasoning is not returned by the OpenAI-compatible endpoint; what is kept is the reasoning Claude writes down: the
design's ``why``, ``evidence`` and ``decision_rule`` per step, each change's ``because`` and ``evidence``, and what
the agent says before each call.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

DESIGN_KEYS = ("understanding", "data_plan", "leakage_hypotheses", "identifiers", "split", "features", "models", "evaluation", "workflow", "routes",
               "risks", "open_questions", "confidence", "changes")
CHECKPOINTS = ("data", "leakage", "models")
CHECKPOINT_STAGES = ("data", "leakage", "features", "models")  # never final: the end has its own revision, after the run
TASKS = ("binary", "multiclass", "regression")
SPLITS = ("stratified", "time", "group", "random")
ACTIONS = {"exclude": "exclude", "review": "review_flag", "keep": "clear"}  # the design's word → the leakage stage's answer (A6.5)
STEP_REF = re.compile(r"\bS(\d{1,2})\b")
WF_REF = re.compile(r"\bWF-(\d\d)\b")
WF_IDS = tuple(f"WF-{i:02d}" for i in range(1, 11))
READ_ONLY = ("describe_data", "get_results", "get_graph", "check_move", "ask_project", "list_samples", "review_labels")


LIST_KEYS = ("leakage_hypotheses", "identifiers", "workflow", "routes", "risks", "open_questions", "changes")
BASELINES = ("dummy", "majority", "baseline")  # the rule screens a baseline beside the families; a design need not name it
# what the designer never sees, beyond the trace's cell keys: the target's values (its positive label, its classes, its range)
TARGET_VALUE_KEYS = frozenset({"target_summary", "positive_label", "class_counts", "class_counts_train", "classes", "min_train", "max_train",
                               "median_train", "label_values"})


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _list(value: Any) -> list[Any]:
    """A field the design should give as a list, whatever Claude wrote: a string or an object is one item."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def clean(value: Any) -> Any:
    """What the designer may see of a tool result or a record: the trace's scrub, then no value of the target."""
    from ..agents.traces import scrub

    def drop(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: drop(x) for k, x in v.items() if k not in TARGET_VALUE_KEYS}
        if isinstance(v, list):
            return [drop(x) for x in v]
        return v

    return drop(scrub(value))


# ---------------------------------------------------------------------------------------------- the design itself


def parse(text: str | None) -> tuple[dict[str, Any] | None, list[str]]:
    """The design in an answer: the first JSON object in it (fenced or not). Returns the object and what is wrong
    with it; a missing key is a problem, not a refusal, so a partial design is still kept and compared."""
    if not text or "{" not in text:
        return None, ["no JSON object in the answer"]
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", body, re.S)
    if fence:
        body = fence.group(1)
    else:
        body = body[body.index("{"):body.rindex("}") + 1] if "}" in body else body
    try:
        design = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, [f"not valid JSON: {exc.msg} at {exc.pos}"]
    if not isinstance(design, dict):
        return None, ["the JSON is not an object"]
    return design, problems(design)


def problems(design: dict[str, Any], tools: list[str] | None = None) -> list[str]:
    out = [f"missing: {k}" for k in DESIGN_KEYS if k not in design]
    out += [f"{k} is not a list" for k in LIST_KEYS if k in design and design[k] is not None and not isinstance(design[k], list)]
    out += [f"{k} is not an object" for k in ("understanding", "data_plan", "split", "features", "models", "evaluation")
            if k in design and design[k] is not None and not isinstance(design[k], dict)]
    u = design.get("understanding") or {}
    if isinstance(u, dict) and u.get("task") not in (None, *TASKS):
        out.append(f"understanding.task is {u.get('task')!r}")
    s = design.get("split") or {}
    if isinstance(s, dict) and s.get("strategy") not in (None, *SPLITS):
        out.append(f"split.strategy is {s.get('strategy')!r}")
    ids = set()
    for i, step in enumerate(steps(design)):
        sid = step.get("id")
        if not sid:
            out.append(f"workflow[{i}] has no id")
        elif sid in ids:
            out.append(f"workflow step {sid} appears twice")
        ids.add(sid)
        if step.get("after") is not None and not isinstance(step.get("after"), list):
            out.append(f"workflow step {sid}: after is not a list")
        if step.get("wf") and str(step["wf"]) not in WF_IDS:
            out.append(f"workflow step {sid}: wf {step['wf']!r} is not WF-01…WF-10")
        if tools is not None and step.get("tool") and step["tool"] not in tools:
            out.append(f"workflow step {sid}: tool {step['tool']!r} does not exist")
    for r in _list(design.get("routes")):
        if isinstance(r, dict) and (r.get("from") not in ids or r.get("to") not in ids):
            out.append(f"route {r.get('from')}→{r.get('to')} names a step that is not in the workflow")
    return out


def steps(design: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [s for s in _list((design or {}).get("workflow")) if isinstance(s, dict)]


def intents(design: dict[str, Any] | None) -> dict[str, Any]:
    """The design's decisions keyed like the decision samples of A6.5 (``point:subject``), so the design can be read
    beside the rule's answer and what the agent finally did."""
    if not design:
        return {}
    out: dict[str, Any] = {}
    u = design.get("understanding") if isinstance(design.get("understanding"), dict) else {}
    for point, value in (("contract.task", u.get("task")), ("contract.target_column", u.get("target"))):
        if value:
            out[f"{point}:-"] = value
    if u.get("prediction_moment"):  # the rule records whether the moment is stated, not the sentence: so does the intent
        out["contract.prediction_moment:-"] = "stated"
    split = design.get("split") if isinstance(design.get("split"), dict) else {}
    if split.get("strategy"):
        out["split.strategy:-"] = split["strategy"]
    ev = design.get("evaluation") if isinstance(design.get("evaluation"), dict) else {}
    if ev.get("metric"):
        out["metric.primary:-"] = ev["metric"]
    models = design.get("models") if isinstance(design.get("models"), dict) else {}
    if models.get("families"):
        out["model.families:-"] = [str(f) for f in _list(models["families"])]
    for h in _list(design.get("leakage_hypotheses")):
        if isinstance(h, dict) and h.get("column"):
            out[f"column.leakage:{h['column']}"] = ACTIONS.get(str(h.get("action") or "review").lower(), "review_flag")
            if str(h.get("action") or "").lower() == "exclude":
                out[f"column.forbidden:{h['column']}"] = "exclude"
    for c in _list(design.get("identifiers")):
        if isinstance(c, str):
            out[f"column.identifier:{c}"] = "identifier"
    return out


def markdown(design: dict[str, Any] | None, version: int, checkpoint: str, raw: str | None = None) -> str:
    if not design:
        return f"# Design v{version:02d} ({checkpoint})\n\nThe answer held no design that could be read.\n\n```\n{(raw or '')[:4000]}\n```\n"
    u = design.get("understanding") or {}
    lines = [f"# Design v{version:02d} ({checkpoint})", "", f"Confidence: {design.get('confidence', '—')}", "", "## Understanding", ""]
    if isinstance(u, dict):
        lines += [f"- **{k}**: {v}" for k, v in u.items()]
    for key, title in (("data_plan", "Data plan"), ("split", "Split"), ("features", "Features"), ("models", "Models"), ("evaluation", "Evaluation")):
        block = design.get(key)
        if isinstance(block, dict):
            lines += ["", f"## {title}", ""] + [f"- **{k}**: {json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v}" for k, v in block.items()]
    hyps = [h for h in _list(design.get("leakage_hypotheses")) if isinstance(h, dict)]
    if hyps:
        lines += ["", "## Leakage hypotheses", "", "| column | action | why | test |", "|---|---|---|---|"]
        lines += [f"| {h.get('column')} | {h.get('action')} | {h.get('why', '')} | {h.get('test', '')} |" for h in hyps]
    if design.get("identifiers"):
        lines += ["", f"Identifiers: {', '.join(map(str, _list(design['identifiers'])))}"]
    if steps(design):
        lines += ["", "## Workflow", "", "| id | wf | tool | stage | does | why | evidence | decision rule | after |", "|---|---|---|---|---|---|---|---|---|"]
        for s in steps(design):
            lines.append(f"| {s.get('id')} | {s.get('wf', '')} | {s.get('tool', '')} | {s.get('stage') or ''} | {s.get('does', '')} | {s.get('why', '')} | "
                         f"{s.get('evidence', '')} | {s.get('decision_rule', '')} | {', '.join(map(str, _list(s.get('after'))))} |")
    routes = [r for r in _list(design.get("routes")) if isinstance(r, dict)]
    if routes:
        lines += ["", "Routes: " + "; ".join(f"{r.get('from')} → {r.get('to')} when {r.get('when', '')}" for r in routes)]
    for key, title in (("risks", "Risks"), ("open_questions", "Open questions")):
        if design.get(key):
            lines += ["", f"## {title}", ""] + [f"- {x}" for x in _list(design[key])]
    changes = changes_of(design)
    if changes:
        lines += ["", "## Changes from the previous version", "", "| field | from | to | because | evidence |", "|---|---|---|---|---|"]
        lines += [f"| {c.get('field')} | {_short(c.get('from'))} | {_short(c.get('to'))} | {c.get('because', '')} | {c.get('evidence', '')} |" for c in changes]
    retro = design.get("retrospective")
    if isinstance(retro, dict):
        lines += ["", "## Retrospective", ""]
        for k, v in retro.items():
            lines += [f"**{k.replace('_', ' ')}**", ""] + [f"- {x}" for x in (v if isinstance(v, list) else [v])] + [""]
    return "\n".join(lines) + "\n"


def changes_of(design: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [c for c in _list((design or {}).get("changes")) if isinstance(c, dict)]


def _short(v: Any, n: int = 60) -> str:
    text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return text if len(text) <= n else text[: n - 1] + "…"


# ---------------------------------------------------------------------------------------------- plan against act


def match(design: dict[str, Any] | None, done: set[str], tool: str, stage: str | None) -> str | None:
    """The first step of the design not yet done that this call carries out (same tool, and the same stage when the
    step names one)."""
    for s in steps(design):
        if s.get("id") in done or s.get("tool") != tool:
            continue
        if s.get("stage") and stage and s["stage"] != stage:
            continue
        return s.get("id")
    return None


def replay(design: dict[str, Any] | None, actions: list[dict[str, Any]]) -> tuple[set[str], int]:
    """The design's steps the successful ``actions`` carried out, and the index of the latest: what the monitor
    starts from when a revision changes the workflow."""
    plan = steps(design)
    order = {s.get("id"): i for i, s in enumerate(plan)}
    done: set[str] = set()
    last = -1
    for a in actions:
        if a.get("ok") is False:
            continue
        sid = match(design, done, a.get("tool"), a.get("stage"))
        if sid is not None:
            done.add(sid)
            last = max(last, order[sid])
    return done, last


def adherence(design: dict[str, Any] | None, actions: list[dict[str, Any]]) -> dict[str, Any]:
    """The planned workflow against the calls the agent made. ``actions`` are timeline rows (tool, stage, ok); a call
    that failed or that the validator refused carries out no step and is counted apart. A matched step is in order when
    it comes after every step matched before it (a skipped step makes nothing after it a reorder)."""
    plan = steps(design)
    order = {s.get("id"): i for i, s in enumerate(plan)}
    done: set[str] = set()
    matched, in_order, out_of_order, unplanned, last = [], [], [], [], -1
    failed = [a.get("tool") + (f":{a['stage']}" if a.get("stage") else "") for a in actions if a.get("ok") is False]
    for a in actions:
        if a.get("ok") is False:
            continue
        sid = match(design, done, a.get("tool"), a.get("stage"))
        if sid is None:
            unplanned.append(a.get("tool") + (f":{a['stage']}" if a.get("stage") else ""))
            continue
        done.add(sid)
        matched.append(sid)
        (in_order if order[sid] > last else out_of_order).append(sid)
        last = max(last, order[sid])
    skipped = [s.get("id") for s in plan if s.get("id") not in done]
    planned = len(plan)
    return {"planned_steps": planned, "done": len(matched), "in_order": in_order, "out_of_order": out_of_order, "skipped": skipped,
            "unplanned": unplanned, "unplanned_writes": [u for u in unplanned if u.split(":")[0] not in READ_ONLY], "failed_or_refused": failed,
            "share_done": round(len(matched) / planned, 3) if planned else None,
            "share_in_order": round(len(in_order) / len(matched), 3) if matched else None}


# ---------------------------------------------------------------------------------------------- the designer and the monitor


class Designer:
    """Asks Claude for the design and its revisions, and keeps every version in the lane's ``design/`` folder."""

    PURPOSE = "twin_designer"
    RESERVE = 3  # a checkpoint revision never takes one of the run's last requests: they are the agent's

    def __init__(self, lane: Any, gateway: Any, *, run_id: str = "", model: str = "", checkpoints: tuple[str, ...] = CHECKPOINTS,
                 steer: bool = True, max_tokens: int = 3200):
        self.lane, self.gateway, self.run_id, self.model = lane, gateway, run_id, model
        self.checkpoints, self.steer, self.max_tokens = tuple(checkpoints), steer, max_tokens
        self.home = Path(lane.home) / "design"
        self.home.mkdir(parents=True, exist_ok=True)
        self.versions: list[dict[str, Any]] = []  # {version, checkpoint, design, problems, request, raw}
        self.timeline_n = 0
        self._lock = threading.Lock()

    # -- what the requests carry

    def tool_names(self) -> list[str]:
        from ..intern.loop import SESSION_TOOLS
        from ..intern.tools import Toolbox

        return Toolbox(self.lane.projects).names() + [t for t in SESSION_TOOLS]

    def context(self) -> dict[str, Any]:
        from ..intern.tools import Toolbox
        from ..studio import graph

        project = self.lane.projects.get(self.lane.project_id)
        described = Toolbox(self.lane.projects).describe_data(self.lane.project_id)
        return {"project": {"id": self.lane.project_id, "name": project.get("name"), "goal": project.get("goal"), "industry": project.get("industry")},
                "data": clean(described),  # never an example value or the target's values, whatever the run's capture setting
                "workflow_steps": [{"id": n["id"], "name": n["name"], "stage": n["stage"], "exit": n["exit"]} for n in graph.NODES],
                "gates": graph.GATES, "tools": self.tool_names()}

    def current(self) -> dict[str, Any] | None:
        parsed = [v for v in self.versions if v["design"]]
        return parsed[-1] if parsed else None

    # -- the requests

    def _ask(self, checkpoint: str, user: str) -> dict[str, Any] | None:
        from .. import prompts

        messages = [{"role": "system", "content": prompts.text("twin_design")}, {"role": "user", "content": user}]
        client = self.gateway.client(self.PURPOSE, project_id=self.lane.project_id)
        started = time.monotonic()
        try:
            out = client.complete(messages, None, self.max_tokens)
        except Exception as exc:  # noqa: BLE001 — a design request that fails is kept as an event; the run goes on
            self.event("design_failed", checkpoint=checkpoint, outcome=f"{type(exc).__name__}: {str(exc)[:300]}")
            return None
        raw = str(out.get("content") or "")
        version = len(self.versions) + 1
        try:  # an answer that cannot be read is a version with problems, never an error in the agent's run
            design, found = parse(raw)
            if design is not None:
                found = problems(design, self.tool_names())
                markdown(design, version, checkpoint)  # every reader of a version must be able to read it
                intents(design)
        except Exception as exc:  # noqa: BLE001
            design, found = None, [f"the design could not be read: {type(exc).__name__}: {str(exc)[:200]}"]
        entry = {"version": version, "checkpoint": checkpoint, "at": now(), "design": design, "problems": found, "request": user, "raw": raw,
                 "seconds": round(time.monotonic() - started, 3), "usage": out.get("usage")}
        self.versions.append(entry)
        try:
            self._write_version(entry)
        except Exception as exc:  # noqa: BLE001 — a file that cannot be written loses the page, not the version
            self.event("design_failed", checkpoint=checkpoint, version=version, outcome=f"files: {type(exc).__name__}: {str(exc)[:200]}")
        self.event("design", checkpoint=checkpoint, version=version, parsed=design is not None, problems=found[:10],
                   steps=len(steps(design)), changes=len(changes_of(design)), confidence=(design or {}).get("confidence"))
        return entry

    def initial(self) -> dict[str, Any] | None:
        user = ("DESIGN REQUEST: initial\n\nDesign the solution and its workflow for this project before anything runs. The tools listed are the "
                "ones the agent can call; the workflow steps are the ones the validator checks.\n\n" + json.dumps(self.context(), ensure_ascii=False, default=str))
        return self._ask("initial", user)

    def revise(self, checkpoint: str, evidence: Any, since: list[dict[str, Any]], final: bool = False) -> dict[str, Any] | None:
        shared = getattr(self.gateway, "shared", None)
        if not final and isinstance(shared, dict) and shared.get("max_requests") is not None and shared["max_requests"] - shared["requests"] <= self.RESERVE:
            self.event("design_skipped", checkpoint=checkpoint, outcome=f"the last {self.RESERVE} requests of the run's cap are kept for the agent")
            return None
        evidence = clean(evidence)
        since = [{k: v for k, v in row.items() if k != "said"} for row in since]  # the agent's words stay in the timeline, not in the designer's request
        cur = self.current()
        head = "DESIGN REQUEST: final (the run has finished; add the retrospective)" if final else f"DESIGN REQUEST: checkpoint after the {checkpoint} stage"
        user = (f"{head}\n\nYour current design (v{cur['version'] if cur else 0}):\n" + json.dumps(cur["design"] if cur else None, ensure_ascii=False)
                + "\n\nWhat the agent did since then (one row per tool call):\n" + json.dumps(since, ensure_ascii=False, default=str)
                + "\n\nThe evidence:\n" + json.dumps(evidence, ensure_ascii=False, default=str)
                + "\n\nWrite the whole design again, revised from this evidence, with every change in \"changes\" (an empty list if nothing changed).")
        return self._ask("final" if final else checkpoint, user)

    # -- the files

    def _write_version(self, entry: dict[str, Any]) -> None:
        v = entry["version"]
        if entry["design"] is not None:
            (self.home / f"design_v{v:02d}.json").write_text(json.dumps(entry["design"], indent=2, ensure_ascii=False), encoding="utf-8")
        (self.home / f"design_v{v:02d}.md").write_text(markdown(entry["design"], v, entry["checkpoint"], entry["raw"]), encoding="utf-8")
        row = {k: entry[k] for k in ("version", "checkpoint", "at", "problems", "seconds", "usage")}
        row.update(parsed=entry["design"] is not None, raw=entry["raw"])
        self._append("designs.jsonl", row)
        for key, value in intents(entry["design"]).items():
            self._append("intents.jsonl", {"version": v, "checkpoint": entry["checkpoint"], "key": key, "point": key.split(":", 1)[0],
                                           "subject": key.split(":", 1)[1], "value": value, "source": "agent_design"})

    def _append(self, name: str, row: dict[str, Any]) -> None:
        with self._lock, (self.home / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def event(self, kind: str, **fields: Any) -> None:
        with self._lock:
            self.timeline_n += 1
            n = self.timeline_n
        self._append("timeline.jsonl", {"n": n, "at": now(), "kind": kind, **fields})

    def follow_text(self) -> str:
        """What the agent is told with its task: the design, as a page, and to name the design step before each call."""
        from .. import prompts

        cur = self.current()
        if not cur:
            return ""
        return prompts.text("twin_follow_design", version=f"v{cur['version']:02d}", design=markdown(cur["design"], cur["version"], cur["checkpoint"]))


class Monitor:
    """On the agent's client: reads the finished steps before each request, writes the timeline, revises at the
    checkpoints. ``before(messages)`` is called by ``ClaudeClient.complete``; ``finish(session)`` by the lane."""

    def __init__(self, designer: Designer):
        self.d = designer
        self.seen = 0  # messages already read
        self.actions: list[dict[str, Any]] = []
        self.done: set[str] = set()
        self.last = -1  # the design index of the latest step matched
        self.since: list[dict[str, Any]] = []  # the rows since the last design version
        self.revised: set[str] = set()
        self._busy = False

    def _calls(self, messages: list[dict[str, Any]]) -> list[tuple[dict[str, Any], list[dict[str, Any]], int]]:
        """Each assistant reply after ``seen`` with its tool results, once its results are all there, and the index
        after it (``seen`` moves past a reply only once the reply has been read)."""
        out, i = [], self.seen
        while i < len(messages):
            m = messages[i]
            if m.get("role") != "assistant":
                i += 1
                continue
            j = i + 1
            results = []
            while j < len(messages) and messages[j].get("role") == "tool":
                results.append(messages[j])
                j += 1
            calls = m.get("tool_calls") or []
            if calls and len(results) < len(calls) and j >= len(messages):
                break  # the results are not all in yet
            out.append((m, results, j))
            i = j
        return out

    def _row(self, said: str | None, call: dict[str, Any], result: dict[str, Any] | None) -> dict[str, Any]:
        fn = call.get("function") or {}
        name = fn.get("name") or call.get("name")
        try:
            args = json.loads(fn.get("arguments") or "{}") if "function" in call else call.get("arguments") or {}
        except (ValueError, TypeError):
            args = {"_raw": str(fn.get("arguments"))[:400]}
        content: Any = None
        if result is not None:
            try:
                content = json.loads(result.get("content") or "null")
            except (ValueError, TypeError):
                content = result.get("content")
        from ..agents.traces import verdict_of

        verdict = verdict_of(content) if result is not None else "error"  # ok, error, or the validator's blocked / needs_approval
        ok = verdict == "ok"
        stage = args.get("stage") if isinstance(args, dict) else None
        cur = self.d.current()
        sid = match(cur["design"] if cur else None, self.done, name, stage) if ok else None
        order = None
        if sid:  # in order when it comes after every step matched so far, as ``adherence`` counts it (a skipped step is not a reorder)
            index = [s.get("id") for s in steps(cur["design"])].index(sid)
            order = "in_order" if index > self.last else "out_of_order"
            self.last = max(self.last, index)
            self.done.add(sid)
        elif ok:
            order = "unplanned"
        from ..agents.traces import summary

        return {"kind": "tool_call", "said": (said or "")[:1500], "named_steps": sorted(set(STEP_REF.findall(said or "")), key=int),
                "named_wf": sorted({m.group(0) for m in WF_REF.finditer(said or "")}), "design_version": cur["version"] if cur else None, "tool": name,
                "stage": stage, "arguments": args if len(json.dumps(args, default=str)) <= 1500 else {"_truncated": True},
                "ok": ok, "verdict": verdict, "result": summary(clean(content), 400) if content is not None else None, "planned_step": sid, "plan": order,
                "_stage_approved": name == "approve_stage" and ok}

    def before(self, messages: list[dict[str, Any]]) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            self._read(messages, steer=True)
        finally:
            self._busy = False

    def _read(self, messages: list[dict[str, Any]], steer: bool) -> None:
        for assistant, results, end in self._calls(messages):
            by_id = {r.get("tool_call_id"): r for r in results}
            for k, call in enumerate(assistant.get("tool_calls") or []):
                try:
                    result = by_id.get(call.get("id")) if len(by_id) == len(results) else (results[k] if k < len(results) else None)
                    row = self._row(assistant.get("content") if k == 0 else None, call, result)
                    approved = row.pop("_stage_approved")
                    self.d.event(**row)
                    self.actions.append({"tool": row["tool"], "stage": row["stage"], "ok": row["ok"]})
                    self.since.append({k2: row[k2] for k2 in ("said", "tool", "stage", "ok", "verdict", "result", "planned_step", "plan")})
                    if approved and row["stage"] in self.d.checkpoints and row["stage"] not in self.revised:
                        self.revised.add(row["stage"])
                        self._checkpoint(row["stage"], messages if steer else None)
                except Exception as exc:  # noqa: BLE001 — one call the monitor cannot read never hides the others
                    self.d.event("monitor_error", outcome=f"{type(exc).__name__}: {str(exc)[:300]}")
            self.seen = max(self.seen, end)

    def _checkpoint(self, stage: str, messages: list[dict[str, Any]] | None) -> None:
        from ..intern.tools import compact_record

        record = self.d.lane.projects.records(self.d.lane.project_id).get(stage)
        before = self.d.current()
        evidence = compact_record(record)
        if stage == "data" and isinstance(evidence, dict):  # its claims spell out the target (a regression's range): the numbers stay, the sentences go
            evidence = {k: v for k, v in evidence.items() if k != "claims"}
        entry = self.d.revise(stage, evidence, self.since)
        if entry is None:  # skipped or failed: the rows wait for the next revision
            return
        self.since = []
        if entry["design"]:  # the new workflow may number its steps differently: start from what the calls already did there
            self.done, self.last = replay(entry["design"], self.actions)
        if entry and entry["design"] and messages is not None and self.d.steer and entry is not before:
            changes = changes_of(entry["design"])
            text = (f"Design revised to v{entry['version']:02d} after the {stage} stage: {len(changes)} change(s).\n"
                    + "".join(f"- {c.get('field')}: {_short(c.get('from'))} → {_short(c.get('to'))} because {c.get('because', '')}\n" for c in changes if isinstance(c, dict))
                    + "\n" + self.d.follow_text())
            messages.append({"role": "user", "content": text})
            self.seen = max(self.seen, len(messages))
            self.d.event("design_handed_to_agent", version=entry["version"], checkpoint=stage, changes=len(changes))

    def finish(self, session: dict[str, Any] | None, outcome: dict[str, Any]) -> dict[str, Any]:
        """After the run: the last steps from the session, the final revision with its retrospective, the adherence,
        the SFT candidates and DESIGN.md. Returns the design summary (also ``design/summary.json``)."""
        if session and session.get("messages"):
            self._read(session["messages"], steer=False)
        first = self.d.versions[0] if self.d.versions else None
        final_rec = self.d.lane.projects.records(self.d.lane.project_id).get("final") if self.d.lane.project_id else None
        from ..intern.tools import compact_record

        evidence = {"final_stage": compact_record(final_rec), "session_status": (session or {}).get("status"), "report": ((session or {}).get("final") or "")[:3000]}
        self.d.revise("final", evidence, self.since, final=True)
        self.since = []
        plan_v1 = adherence(first["design"] if first and first["design"] else None, self.actions)
        cur = self.d.current()
        # against the last version written before the end: the final one is written after Claude saw everything
        revised = [v for v in self.d.versions if v["design"] and v["checkpoint"] != "final"]
        plan_last = adherence(revised[-1]["design"] if revised else None, self.actions)
        result = {"versions": [{"version": v["version"], "checkpoint": v["checkpoint"], "parsed": v["design"] is not None, "problems": len(v["problems"]),
                                "steps": len(steps(v["design"])), "changes": len(changes_of(v["design"])),
                                "confidence": (v["design"] or {}).get("confidence")} for v in self.d.versions],
                  "adherence_to_first": plan_v1, "adherence_to_last_revision": plan_last,
                  "last_revision": revised[-1]["version"] if revised else None, "actions": len(self.actions),
                  "design_seconds": round(sum(float(v.get("seconds") or 0.0) for v in self.d.versions), 3),
                  "intents_first": intents(first["design"]) if first else {}, "intents_last": intents(cur["design"]) if cur else {},
                  "first_workflow": [{k: s.get(k) for k in ("id", "wf", "tool", "stage", "does")} for s in steps(first["design"] if first else None)],
                  "changes": [{"version": v["version"], "checkpoint": v["checkpoint"], **c} for v in self.d.versions for c in changes_of(v["design"])],
                  "retrospective": (cur["design"] or {}).get("retrospective") if cur else None, "outcome": outcome}
        (self.d.home / "adherence.json").write_text(json.dumps({"to_first": plan_v1, "to_last_revision": plan_last}, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.d.home / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        self._sft(outcome, plan_v1)
        (self.d.home / "DESIGN.md").write_text(self.story(result), encoding="utf-8")
        return result

    def _sft(self, outcome: dict[str, Any], plan: dict[str, Any]) -> None:
        from .. import prompts

        dataset = None
        try:
            from ..decisions.schema import dataset_ref

            dataset = dataset_ref(((self.d.lane.projects.get(self.d.lane.project_id).get("data")) or {}).get("sha256"))
        except Exception:  # noqa: BLE001
            pass
        pid = str(self.d.lane.project_id or "")
        for v in self.d.versions:
            if not v["design"]:
                continue
            self.d._append("sft_candidates.jsonl", {
                "messages": [{"role": "system", "content": prompts.text("twin_design")}, {"role": "user", "content": v["request"].replace(pid, "<project>") if pid else v["request"]},
                             {"role": "assistant", "content": json.dumps(v["design"], ensure_ascii=False)}],
                "metadata": {"stage": "solution_design" if v["checkpoint"] == "initial" else "design_revision", "checkpoint": v["checkpoint"], "version": v["version"],
                             "source": "twin:claude", "model": self.d.model, "run_id": self.d.run_id, "dataset_ref": dataset, "verified": False,
                             "problems": v["problems"], "outcome": outcome, "adherence_share_done": plan.get("share_done")}})

    def story(self, s: dict[str, Any]) -> str:
        lines = ["# How Claude designed the solution", "", f"Run {self.d.run_id} · model {self.d.model} · checkpoints: {', '.join(self.d.checkpoints)} · "
                 f"steer: {'on' if self.d.steer else 'off'}", "", "| version | checkpoint | parsed | steps | changes | confidence | problems |", "|---|---|---|---|---|---|---|"]
        lines += [f"| v{v['version']:02d} | {v['checkpoint']} | {'yes' if v['parsed'] else '**no**'} | {v['steps']} | {v['changes']} | {v['confidence'] or '—'} | {v['problems']} |"
                  for v in s["versions"]]
        a, b = s["adherence_to_first"], s["adherence_to_last_revision"]
        lines += ["", "## Plan against act", "",
                  f"Against the first design: {a['done']} of {a['planned_steps']} planned steps done, {len(a['in_order'])} in order, {len(a['out_of_order'])} out of order; "
                  f"skipped {a['skipped'] or '—'}; not in the design: {a['unplanned'] or '—'}.", "",
                  f"Against the last revision before the end (v{s['last_revision'] or 0:02d}): {b['done']} of {b['planned_steps']} done; skipped {b['skipped'] or '—'}; "
                  f"not in the design: {b['unplanned'] or '—'}.", "",
                  f"Calls that failed or that the validator refused (no step counted): {a.get('failed_or_refused') or '—'}.", ""]
        if s["changes"]:
            lines += ["## What changed, and why", "", "| version | after | field | from | to | because | evidence |", "|---|---|---|---|---|---|---|"]
            lines += [f"| v{c['version']:02d} | {c['checkpoint']} | {c.get('field')} | {_short(c.get('from'))} | {_short(c.get('to'))} | {c.get('because', '')} | {c.get('evidence', '')} |"
                      for c in s["changes"]]
            lines.append("")
        lines += ["## Timeline", "", "| n | kind | design | step | plan | tool | stage | verdict | said |", "|---|---|---|---|---|---|---|---|---|"]
        path = self.d.home / "timeline.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
            r = json.loads(line)
            if r["kind"] == "tool_call":
                lines.append(f"| {r['n']} | call | v{r['design_version'] or 0:02d} | {r.get('planned_step') or '—'} | {r.get('plan') or '—'} | {r['tool']} | {r.get('stage') or ''} | "
                             f"{r.get('verdict') or ('ok' if r.get('ok') else 'error')} | {_short((r.get('said') or '').replace('|', '/').replace(chr(10), ' '), 120)} |")
            else:
                lines.append(f"| {r['n']} | {r['kind']} | {('v%02d' % r['version']) if r.get('version') else ''} | | | | {r.get('checkpoint') or ''} | {r.get('outcome') or ''} | |")
        if s.get("retrospective"):
            lines += ["", "## Retrospective", "", "```json", json.dumps(s["retrospective"], indent=2, ensure_ascii=False), "```"]
        lines += ["", "Each version is in `design_vNN.md` and `design_vNN.json`; Claude's raw answers in `designs.jsonl`."]
        return "\n".join(lines) + "\n"
