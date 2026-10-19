"""Ask the workspace (package 30.21): a chat over the workspace's own tables, every number from a typed computation run on them.

``ask(services, question, person=...)`` runs one agent (the harness's ``workspace_chat``, a model loop on ``agents.runtime.run``, always
traced) with three read tools of the scope "workspace": ``list_tables`` (each project's table: its name, columns and rows), ``describe_table``
(each column's kind, missing share and distinct count; never a value) and ``compute`` (``dclab_core.workspace_chat.compute``: a count, sum,
mean, median, minimum, maximum, share or distinct count, grouped and filtered; a personal column never a group). Every result is masked
before the model reads it. The answer's numbers must be numbers a computation returned, or it is refused and the person told why. Every
exchange (the question, each tool call and result, the answer or the refusal, the cost) is kept whole in ``<workspace>/workspace_chat/`` and
written to the audit trail. A private project's table is never offered. With no model serving, the person is told so: nothing guesses.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .agents.registry import Registry, Tool

PURPOSE = "workspace_chat"
SYSTEM = (Path(__file__).resolve().parent / "prompts" / "workspace_chat.md")


class Session:
    """What one question may read: the workspace's tables (never a private project's), and what its tools returned."""

    def __init__(self, services: Any):
        self.services, self.results, self.calls, self.shown = services, [], [], 0
        self._tables: Optional[dict[str, dict[str, Any]]] = None

    def tables(self) -> dict[str, dict[str, Any]]:
        if self._tables is None:
            from .studio import data as sd

            out = {}
            for p in self.services.projects.list():
                project = self.services.projects.get(p["id"]) if isinstance(p, dict) else self.services.projects.get(p)
                if (project.get("settings") or {}).get("private") or not (project.get("data") or {}).get("filename"):
                    continue
                try:
                    frame = sd.load_table(sd.data_path(self.services.projects, project))
                except Exception:  # noqa: BLE001 - a table that does not read is not offered
                    continue
                name = f"{project['id']}:{str(project['data']['filename']).rsplit('.', 1)[0]}"
                out[name] = {"frame": frame, "project": project["id"], "personal": _personal(frame)}
            self._tables = out
        return self._tables


PERSONAL_NAME = re.compile(r"(^|_)((first|last|full|given|family|sur|user)_?name|name|e_?mail|phone|mobile|address|street|postcode|zip|"
                           r"ssn|passport|iban|card|birth|dob|ip)($|_)", re.I)
SHOWN_PER_QUESTION = 100  # group values one question may show the model, over all its computations (the purpose's cell budget)


def _personal(frame: Any) -> list[str]:
    """Columns whose values read as a person's (an e-mail, a phone, a card) or whose name says so (a name, an address, a birth date):
    never computed on, filtered on or grouped by."""
    from dclab_core import redact

    out = []
    for col in frame.columns:
        sample = frame[col].dropna().astype(str).head(50)
        if PERSONAL_NAME.search(str(col)) or (len(sample) and sum(redact.redact_text(v)[0] != v for v in sample) >= max(1, len(sample) // 5)):
            out.append(str(col))
    return out


def list_tables(session: Session, /, **_: Any) -> dict[str, Any]:
    """The workspace's tables: each one's name, its columns and its number of rows (never a row)."""
    return {"tables": [{"table": name, "columns": [str(c) for c in t["frame"].columns], "rows": int(len(t["frame"])),
                        "personal_columns": t["personal"]} for name, t in session.tables().items()]}


def describe_table(session: Session, /, table: str = "", **_: Any) -> dict[str, Any]:
    """One table's columns: kind (number, text, date), share missing and distinct count; never a value."""
    import pandas as pd

    t = session.tables().get(table)
    if t is None:
        return {"error": f"No table {table!r}. Call list_tables."}
    f = t["frame"]
    cols = []
    for c in f.columns:
        s = f[c]
        kind = "number" if pd.api.types.is_numeric_dtype(s) else "date" if pd.api.types.is_datetime64_any_dtype(s) else "text"
        cols.append({"column": str(c), "kind": kind, "missing": round(float(s.isna().mean()), 4), "distinct": int(s.nunique()), "personal": str(c) in t["personal"]})
    return {"table": table, "rows": int(len(f)), "columns": cols}


def compute(session: Session, /, **args: Any) -> dict[str, Any]:
    """A computation on one table: op (count, sum, mean, median, min, max, share, distinct), column, group_by (up to two), where (column ->
    value). Its result is the only source of a number in the answer."""
    from dclab_core import workspace_chat as wc

    try:
        c = wc.Computation(**{k: v for k, v in args.items() if k in ("table", "op", "column", "group_by", "where")})
    except Exception as exc:  # noqa: BLE001 - a malformed computation is told back to the model
        return {"error": f"not a computation: {exc}"}
    t = session.tables().get(c.table)
    if t is None:
        return {"error": f"No table {c.table!r}. Call list_tables."}
    try:
        result = wc.compute(t["frame"], c, t["personal"])
    except ValueError as exc:
        return {"error": str(exc)}
    groups = len(result["result"]) if isinstance(result["result"], list) else 1
    if session.shown + groups > SHOWN_PER_QUESTION:
        return {"error": f"this question has been shown {session.shown} values; {groups} more would pass {SHOWN_PER_QUESTION}: group less or filter"}
    session.shown += groups
    shown = wc.masked(result)  # a group label that reads as personal is masked before the model reads it
    session.results.append(result)
    return shown


def register(registry: Registry) -> None:
    S = {"type": "string"}
    registry.register(Tool("list_tables", "The workspace's tables: names, columns and rows (never a row).", {"type": "object", "properties": {}}, list_tables,
                           scope="workspace", takes_context=True))
    registry.register(Tool("describe_table", "One table's columns: kind, share missing, distinct count; never a value.",
                           {"type": "object", "properties": {"table": S}, "required": ["table"]}, describe_table, scope="workspace", takes_context=True))
    registry.register(Tool("compute", "Compute on one table: op is count, sum, mean, median, min, max, share or distinct; column (not for count); "
                           "group_by up to two columns; where: column to value (equality). Every number you give must come from a result of this tool.",
                           {"type": "object", "properties": {"table": S, "op": {**S, "enum": ["count", "sum", "mean", "median", "min", "max", "share", "distinct"]},
                                                              "column": S, "group_by": {"type": "array", "items": S},
                                                              "where": {"type": "object", "additionalProperties": True}}, "required": ["table", "op"]},
                           compute, scope="workspace", takes_context=True))


def ask(services: Any, question: str, *, person: str = "person", gateway: Any = None, max_steps: int = 8, cap_eur: Optional[float] = None) -> dict[str, Any]:
    """One question answered from the workspace's tables, or refused with why; the whole exchange kept and audited."""
    from dclab_core import workspace_chat as wc

    from .agents import default_registry
    from .agents.runtime import Policy, run as run_policy
    from .models import installed

    gateway = gateway or getattr(services, "gateway", None) or installed()
    session = Session(services)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    client = gateway.client(PURPOSE) if gateway is not None else None
    record: dict[str, Any] = {"at": started, "person": person, "question": question, "tables": list(session.tables())}
    if client is None:
        record.update(status="no_model", answer=None, why="No model serves the workspace chat here: nothing answers by guessing.")
        return _keep(services, record)
    if cap_eur is not None:
        client.run_limit_eur = cap_eur
    registry = default_registry()
    policy = Policy(PURPOSE, SYSTEM.read_text(encoding="utf-8"), tuple(registry.names("workspace")), max_steps=max_steps, must_act=True)
    from dclab_core import redact

    asked = redact.redact_text(question)[0]  # a person's e-mail or phone in the question never reaches the model; the record keeps the words
    messages = [{"role": "system", "content": policy.system}, {"role": "user", "content": asked}]
    try:
        out = run_policy(policy, client, registry, messages, context=session, contexts={"workspace": session})
        answer, stopped = out.final or "", out.stopped
        record["steps"] = [{"n": s.n, "tool": s.tool, "arguments": s.arguments, "result": s.result, "ok": s.ok, "seconds": round(s.seconds, 3)} for s in out.steps]
        record["usage"] = out.usage
    except Exception as exc:  # noqa: BLE001 - a failed run is kept and said, never a half answer
        record.update(status="failed", answer=None, why=f"{type(exc).__name__}: {exc}")
        return _keep(services, record)
    bad = wc.ungrounded(answer, session.results)
    record["computations"] = session.results
    record["spent_eur"] = round(float(getattr(client, "spent_eur", 0.0) or 0.0), 6)
    if not session.results or bad:
        record.update(status="refused", answer=None, said=answer, ungrounded=bad,
                      why=("No computation was run, so no number can be given." if not session.results else
                           f"The answer gave {', '.join(str(b) for b in bad)}, which no computation returned: it is not shown."))
    else:
        record.update(status="answered", answer=answer, stopped=stopped)
    return _keep(services, record)


def _keep(services: Any, record: dict[str, Any]) -> dict[str, Any]:
    """The exchange kept whole (never cut) under the workspace, one line a question, one file a day, and written to the audit trail
    whatever its status (answered, refused, failed, no model)."""
    from . import audit

    try:
        audit.record_for(services.projects, "workspace_chat", "agent", record.get("person") or "person", move="ask_workspace",
                         status=record["status"], args={"question_chars": len(record.get("question") or ""),
                                                        "computations": len(record.get("computations") or []), "spent_eur": record.get("spent_eur", 0.0)},
                         message=record.get("why") or "answered")
    except Exception:  # noqa: BLE001 - the audit is written where it can be; the exchange is kept either way
        pass
    home = Path(getattr(getattr(services, "store", None), "home", "") or ".")
    folder = home / "workspace_chat"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    return record


__all__ = ["ask", "register", "Session", "list_tables", "describe_table", "compute", "PURPOSE"]
