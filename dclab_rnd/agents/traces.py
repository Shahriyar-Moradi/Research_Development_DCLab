"""A trace for every agent run, and its replay (package A2.3).

One row per step: the run, the step number, which model reply asked for it, a summary of the state the agent saw,
the tool and its arguments, the verdict, a summary of the result, the tokens of that reply and the seconds the
tool took. Rows are rows of ``agent_step`` when ``DCLAB_DATABASE_URL`` is set, otherwise lines of a JSON file in
the workspace.

Never a cell value: arguments are what the model sent (the tools take names, ids and sentences, never rows), and a
result is summarised after the keys that carry values from the table (``CELL_KEYS``: column examples, previews) are
removed. Arguments longer than ``ARGUMENT_CHARS`` are cut, and a run with cut arguments cannot be replayed.

``replay(run_id)`` feeds the stored model replies back through ``run()`` with a scripted model and the recorded
results, and checks that the runtime makes the same moves: the same tools, in the same order, with the same verdicts.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from .registry import Registry, Tool
from .runtime import Policy, Step, run

FIELDS = ("run_id", "agent", "n", "reply", "at", "state", "tool", "arguments", "verdict", "result",
          "input_tokens", "output_tokens", "seconds")
ARGUMENT_CHARS = 8000  # names and sentences; far above any real call, so a cut row is rare
RESULT_CHARS = 360
CELL_KEYS = frozenset({"examples", "preview", "top_values", "values", "head", "sample_rows", "rows_preview"})
VERDICTS = ("ok", "error", "blocked", "needs_approval")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def verdict_of(result: Any) -> str:
    """What happened to one call: ok, error (a bad call or a failed tool), or the validator's blocked/needs_approval."""
    if isinstance(result, dict):
        if result.get("status") in ("blocked", "needs_approval"):
            return result["status"]
        if "error" in result:
            return "error"
    return "ok"


def scrub(value: Any) -> Any:
    """The value without the keys that carry cells from the user's table (at any depth)."""
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k not in CELL_KEYS}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def summary(result: Any, chars: int = RESULT_CHARS) -> str:
    if isinstance(result, dict) and "error" in result:
        return ("error: " + str(result["error"]))[:chars]
    text = json.dumps(scrub(result), ensure_ascii=False, default=str)
    return text if len(text) <= chars else text[:chars] + "…"


def arguments_of(arguments: dict[str, Any] | None) -> dict[str, Any]:
    arguments = dict(arguments or {})
    text = json.dumps(arguments, ensure_ascii=False, default=str)
    return arguments if len(text) <= ARGUMENT_CHARS else {"_truncated": text[:ARGUMENT_CHARS] + "…"}


class TraceStore(Protocol):
    def record(self, row: dict[str, Any]) -> None: ...
    def steps(self, run_id: str) -> list[dict[str, Any]]: ...
    def delete(self, run_id: str) -> None: ...


class FileTraces:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, row: dict[str, Any]) -> None:
        line = json.dumps({k: row.get(k) for k in FIELDS}, ensure_ascii=False, default=str)
        with self._lock:
            torn = self.path.is_file() and self.path.stat().st_size and not self.path.read_bytes().endswith(b"\n")
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(("\n" if torn else "") + line + "\n")  # a line torn by a crash never swallows the next row

    def _rows(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn line
        return rows

    def steps(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = [r for r in self._rows() if isinstance(r, dict) and r.get("run_id") == run_id]
        return sorted(rows, key=lambda r: r.get("n") or 0)

    def delete(self, run_id: str) -> None:
        """Forget a run's rows (its session was deleted). The file is replaced whole, never rewritten in place."""
        with self._lock:
            keep = [r for r in self._rows() if isinstance(r, dict) and r.get("run_id") != run_id]
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in keep), encoding="utf-8")
            os.replace(temporary, self.path)


class PgTraces:
    def __init__(self, workspace_id: str, url: str | None = None):
        from ..storage import db

        self.workspace_id, self.engine = workspace_id, db.engine(url)

    def record(self, row: dict[str, Any]) -> None:
        import sqlalchemy as sa

        from ..storage.models import agent_step

        with self.engine.begin() as c:
            c.execute(sa.insert(agent_step).values(workspace_id=self.workspace_id, **{k: row.get(k) for k in FIELDS}))

    def steps(self, run_id: str) -> list[dict[str, Any]]:
        import sqlalchemy as sa

        from ..storage.models import agent_step as t

        q = sa.select(*[t.c[k] for k in FIELDS]).where(t.c.workspace_id == self.workspace_id, t.c.run_id == run_id).order_by(t.c.n)
        with self.engine.connect() as c:
            return [dict(r._mapping) for r in c.execute(q)]

    def delete(self, run_id: str) -> None:
        import sqlalchemy as sa

        from ..storage.models import agent_step as t

        with self.engine.begin() as c:
            c.execute(sa.delete(t).where(t.c.workspace_id == self.workspace_id, t.c.run_id == run_id))


def open_traces(home: Path) -> TraceStore:
    from ..storage import db

    if db.database_url(required=False):
        return PgTraces(db.workspace(str(Path(home).resolve()), Path(home).name))
    return FileTraces(Path(home) / "agent_steps.jsonl")


class Tracer:
    """Writes the rows of one run. ``state`` returns a short summary of what the agent sees (for a project, the graph
    state string); it is read before each row. A trace that cannot be written never stops the run.

    A run can span several ``run()`` calls (an intern session with follow-ups): ``Tracer.resume`` continues the step
    and reply numbers where the stored rows end.
    """

    def __init__(self, store: TraceStore | None, run_id: str, agent: str, state: Callable[[], str] | None = None,
                 start: int = 0, reply_base: int = 0):
        self.store, self.run_id, self.agent, self.state = store, run_id, agent, state
        self.n, self.reply_base, self.last_reply = start, reply_base, reply_base - 1
        self._seen: str | None = None  # the state before the current move (``before``)

    @classmethod
    def resume(cls, store: TraceStore | None, run_id: str, agent: str, state: Callable[[], str] | None = None) -> "Tracer":
        rows = []
        if store is not None:
            try:
                rows = store.steps(run_id)
            except Exception:  # noqa: BLE001 — start a fresh numbering rather than fail the run
                rows = []
        replies = [int(r["reply"]) for r in rows if r.get("reply") is not None]
        return cls(store, run_id, agent, state, start=max((int(r["n"]) for r in rows), default=0),
                   reply_base=max(replies, default=-1) + 1)

    def _state(self) -> str:
        try:
            return str((self.state() if self.state else "") or "")[:200]
        except Exception:  # noqa: BLE001 — a state summary is a convenience, never a reason to fail
            return ""

    def before(self) -> None:
        """Note the state the agent sees now, before its move runs; the next row records it."""
        self._seen = self._state()

    def next_call(self) -> None:
        """Start the numbering for the next ``run()`` call: its reply 0 follows the last reply written."""
        self.reply_base = self.last_reply + 1

    def __call__(self, tool: str, arguments: dict[str, Any] | None, result: Any, seconds: float, reply: int | None = None,
                 tokens: dict[str, int] | None = None, verdict: str | None = None, n: int | None = None) -> dict[str, Any]:
        """One row. ``n`` is the caller's own step number when it keeps one (the intern session's), so the two never drift."""
        self.n = n if n is not None else self.n + 1
        if reply is not None:
            reply = self.reply_base + reply
            self.last_reply = max(self.last_reply, reply)
        state = self._seen if self._seen is not None else self._state()
        self._seen = None
        row = {"run_id": self.run_id, "agent": self.agent, "n": self.n, "reply": reply, "at": now(), "state": state,
               "tool": tool, "arguments": arguments_of(arguments), "verdict": verdict or verdict_of(result), "result": summary(result),
               "input_tokens": int((tokens or {}).get("input_tokens") or 0), "output_tokens": int((tokens or {}).get("output_tokens") or 0),
               "seconds": round(float(seconds), 3)}
        if self.store is not None:
            try:
                self.store.record(row)
            except Exception:  # noqa: BLE001 — losing a trace row must not lose the user's work
                pass
        return row


def moves(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(r["tool"], r["verdict"]) for r in rows]


class _Script:
    """A scripted model that sends the stored replies again, in order, then answers without a tool call."""

    def __init__(self, replies: list[list[dict[str, Any]]]):
        self.replies, self.sent = list(replies), -1

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **_: Any) -> dict[str, Any]:
        self.sent += 1
        calls = self.replies.pop(0) if self.replies else []
        tool_calls = [{"id": f"r{self.sent}.{i}", "name": c["tool"], "arguments": c["arguments"]} for i, c in enumerate(calls)]
        return {"content": None if calls else "(replayed)", "tool_calls": tool_calls, "usage": {"input_tokens": 0, "output_tokens": 0},
                "assistant_message": {"role": "assistant", "content": None, "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in tool_calls]}}


def replay(rows: list[dict[str, Any]], policy: Policy, registry: Registry) -> dict[str, Any]:
    """Run the stored replies through the runtime again, each tool answering what it answered then.

    The runtime's own decisions run for real: whether the tool is in the policy, the schema check, the budget and the
    end of the run. The tools answer from the trace (their recorded verdict and summary), so a replay changes no
    project and needs no model. Returns the recorded and the replayed moves and whether they are the same.
    """
    if any(r.get("reply") is None for r in rows):
        raise ValueError("this trace has rows without a reply number (not written by run()); it cannot be replayed")
    if any("_truncated" in (r.get("arguments") or {}) for r in rows):
        raise ValueError("this trace has arguments that were cut when stored; it cannot be replayed exactly")
    numbers = sorted({int(r["reply"]) for r in rows})
    by_reply = {k: [r for r in rows if int(r["reply"]) == k] for k in numbers}
    script = _Script([[{"tool": r["tool"], "arguments": r["arguments"]} for r in by_reply[k]] for k in numbers])
    seen: dict[int, int] = {}  # reply -> steps the replay has made in it so far

    def recorded(name: str) -> Callable[..., Any]:
        def answer(**_: Any) -> Any:
            reply = script.sent
            calls = by_reply.get(numbers[reply], []) if reply < len(numbers) else []
            index = seen.get(reply, 0)
            row = calls[index] if index < len(calls) else None
            if row is None or (stub.get(row["tool"]) is None or stub.get(row["tool"]).name) != name:  # called by an alias
                return {"error": "replay: the runtime made a move the trace does not have"}
            if row["verdict"] in ("blocked", "needs_approval"):
                return {"error": row["result"], "status": row["verdict"]}
            if row["verdict"] == "error":
                return {"error": row["result"]}
            return {"replayed": row["result"]}
        return answer

    stub = Registry()
    for name in policy.tools:
        tool = registry.get(name)
        stub.register(Tool(tool.name, tool.description, tool.parameters, recorded(tool.name), aliases=tool.aliases))

    def count(step: Step) -> None:  # keyed by the script's reply count: a step's own reply number restarts with each run()
        seen[script.sent] = seen.get(script.sent, 0) + 1

    # A run that stopped (a question to the person, a terminal tool) and was continued later is replayed the same way:
    # run() again on the remaining replies, with the steps already used counted against the budget.
    messages: list[dict[str, Any]] = [{"role": "system", "content": policy.system}]
    steps: list[Step] = []
    while True:
        before = len(script.replies)
        out = run(policy, script, stub, messages, on_step=count, steps_used=len(steps))
        steps += out.steps
        if not script.replies or len(script.replies) == before or out.stopped in ("steps", "time"):
            break
    again = [(s.tool, verdict_of(s.result)) for s in steps]
    return {"recorded": moves(rows), "replayed": again, "same": moves(rows) == again}
