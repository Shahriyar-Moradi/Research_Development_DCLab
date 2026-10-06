"""Trajectory export (package A6.1): finished runs as training records for the policy model, only from projects whose
owner opted in.

A record is one decision: the state before it, what could be chosen there, what was chosen, the verdict, the records
it cited and what followed. Two sources give them:

- ``move``: an entry of the project's transition log (``studio/graph.py``): every move a person or an agent proposed,
  with the validator's verdict, the rules and records it cited, the moves the graph allowed at that state (written
  with the entry since this package; older entries say ``null``) and the outcome.
- ``step``: a row of an agent's trace (``agents/traces.py``): every tool the intern or the Home agent called on the
  project, with the tools its scope offered, the arguments, the verdict, the record ids in the result and whether a
  model or the standard plan chose it.

Nothing that could hold a value is kept. A tool's arguments are kept only where its schema says the field is not free
text: an enum, a boolean, a number, or a field that names a column or an id (``target``, ``identifiers``,
``project_id``…). Every other string is dropped, the positive label (a value of the target) and the prediction moment
(a person's words) among them. No result text, message, reason or outcome summary is exported: only the outcome's kind.

A name-type string is kept only when it is one of the project's own names (its columns, the option ids of its stage
decisions, stage and gate names, sample keys, its ids) or an evidence record id: a model can put a cell value in a field
named ``identifiers`` or ``choice``, and the field's name alone proves nothing. Cited ids are kept only when they are
evidence records or the project's own claims.

A project is exported only when ``settings.share_for_training`` is true: off by default, set by its owner. The export
goes to its own folder, never into the v3 corpus (``out_v3``), which is built by its own command.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import cited
from .store import STAGE_KEYS

OPT_IN = "share_for_training"
V3 = Path(__file__).resolve().parents[2] / "research" / "llm-fine-tuning" / "experiments" / "sft" / "out_v3"
VERSION = 1
# string fields that name a column, a sample, an option or a record: structure, never a cell or a person's words
NAMES = {"project_id", "key", "record_id", "choice", "target", "time_column", "group_column", "identifiers", "text_columns",
         "column", "columns", "stage", "gate", "move"}
SCOPE_OF = {"intern": ("project", "session"), "home": ("draft",)}  # what each agent's policy offers it


def opted_in(project: dict[str, Any]) -> bool:
    return bool((project.get("settings") or {}).get(OPT_IN))


def _schemas(scope: str) -> dict[str, dict[str, Any]]:
    from ..agents import default_registry

    out = {}
    for schema in default_registry().schemas(scope):
        fn = schema.get("function", schema)
        out[fn["name"]] = fn.get("parameters") or {}
    return out


def vocabulary(project: dict[str, Any], stage_records: dict[str, Any], run_ids: list[str]) -> set[str]:
    """Every name a kept string may be: the project's columns and ids, its stage decisions' option ids, stage, gate and
    move names, sample keys and evidence record ids. Nothing a model or a person typed is in it unless it is one of these."""
    from .. import tools
    from . import data as studio_data, graph

    names = set((project.get("data") or {}).get("columns") or []) | set(STAGE_KEYS) | set(graph.GATES) | {project["id"], *run_ids}
    names |= {m for m in ("create_project", "attach_data", "propose_solution", "set_solution", "set_settings", "run_stage", "approve_stage",
                          "approve_gate", "capture")}
    if (project.get("draft") or {}).get("id"):
        names.add(project["draft"]["id"])
    for record in (stage_records or {}).values():
        decision = (record or {}).get("decision") or {}
        names |= {str(o.get("id")) for o in decision.get("options") or [] if o.get("id") is not None}
        names |= {str(decision[k]) for k in ("selected", "chosen") if decision.get(k)}
        names |= {c["claim_id"] for c in (record or {}).get("claims") or [] if c.get("claim_id")}
    try:
        names |= {s["key"] for s in studio_data.sample_catalog()}
    except Exception:  # noqa: BLE001 — without the catalog, a sample key is simply not kept
        pass
    names |= {r["record_id"] for r in tools._index().records}
    return names


def keep(value: Any, schema: dict[str, Any], name: str = "", vocab: set[str] | frozenset[str] = frozenset()) -> Any:
    """``value`` reduced to what its schema says is not free text, or None when nothing is left. A name-type string is
    kept only when it is in ``vocab``."""
    if value is None or not isinstance(schema, dict):
        return None
    if "enum" in schema:
        return value if value in schema["enum"] else None
    kind = schema.get("type")
    if isinstance(kind, list):  # ["string", "null"] and the like
        kind = next((k for k in kind if k != "null"), None)
    if kind == "boolean":
        return value if isinstance(value, bool) else None
    if kind in ("integer", "number"):
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    if kind == "string":
        return value if name in NAMES and isinstance(value, str) and value in vocab else None
    if kind == "array" and isinstance(value, list):
        items = [keep(v, schema.get("items") or {}, name, vocab) for v in value]
        items = [v for v in items if v is not None]
        return items or None
    if kind == "object" and isinstance(value, dict):
        props = schema.get("properties") or {}
        out = {k: kept for k, v in value.items() if k in props and (kept := keep(v, props[k], k, vocab)) is not None}
        return out or None
    return None


def arguments(tool: str, args: Any, schemas: dict[str, dict[str, Any]], vocab: set[str] | frozenset[str] = frozenset()) -> dict[str, Any]:
    if tool not in schemas or not isinstance(args, dict):
        return {}
    return keep(args, {"type": "object", **schemas[tool]}, "", vocab) or {}


def _outcome(entry: dict[str, Any]) -> str | None:
    """The kind of what happened, never its text (a stage summary or an error message can hold anything)."""
    text = entry.get("outcome")
    if not text:
        return None
    return "failed" if str(text).startswith("failed") else "done"


def _ids(text: Any, vocab: set[str] | frozenset[str]) -> list[str]:
    """Record ids in a result, only those that are evidence records or the project's own claims (a table's own codes can
    look like an id)."""
    return sorted({i for i in cited.IDS.findall(str(text or "")) if i in vocab})


def moves(project: dict[str, Any], log: list[dict[str, Any]], vocab: set[str] | frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    out = []
    for i, e in enumerate(log):
        after = log[i + 1] if i + 1 < len(log) else None
        args = {k: v for k, v in (e.get("args") or {}).items() if k in NAMES and isinstance(v, str) and v in vocab}  # stage, gate, choice: never a reason
        out.append({"kind": "move", "project_id": project["id"], "n": i + 1, "actor": e.get("actor"), "state": e.get("state"),
                    "node": e.get("from"), "allowed": e.get("allowed"), "move": e.get("move"), "arguments": args,
                    "verdict": e.get("status"), "failed_checks": [c for c in e.get("failed_checks") or [] if isinstance(c, str)],
                    "cited": sorted({i for i in (e.get("rules") or []) + (e.get("evidence") or []) if isinstance(i, str) and i in vocab}),
                    "next": {"outcome": _outcome(e), "to": e.get("to") if e.get("status") == "allowed" else e.get("from"),
                             "state": after.get("state") if after else None}})
    return out


def steps(project: dict[str, Any], run_id: str, rows: list[dict[str, Any]], schemas_by_scope: dict[str, dict[str, Any]],
          vocab: set[str] | frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(rows):
        scopes = SCOPE_OF.get(r.get("agent"), ("project",))
        schemas = {}
        for scope in scopes:
            schemas.update(schemas_by_scope.setdefault(scope, _schemas(scope)))
        after = rows[i + 1] if i + 1 < len(rows) else None
        args = r.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        out.append({"kind": "step", "project_id": project["id"], "run": run_id, "agent": r.get("agent"), "n": r.get("n"),
                    "chosen_by": "plan" if r.get("reply") is None else "model", "state": r.get("state"), "allowed": sorted(schemas),
                    "tool": r.get("tool"), "arguments": arguments(r.get("tool"), args, schemas, vocab), "verdict": r.get("verdict"),
                    "cited": _ids(r.get("result"), vocab), "next": {"state": after.get("state") if after else None}})
    return out


def records(project: dict[str, Any], projects: Any, sessions: Any, traces: Any) -> list[dict[str, Any]]:
    """Every record of one project: its transition log, then each agent run on it (the intern's sessions, the Home draft)."""
    runs = [s["id"] for s in sessions.list() if s.get("project_id") == project["id"]]
    if (project.get("draft") or {}).get("id"):
        runs.append(project["draft"]["id"])
    vocab = vocabulary(project, projects.records(project["id"]), runs)
    out = moves(project, projects.transitions(project["id"], 10**6), vocab)
    cache: dict[str, dict[str, Any]] = {}
    for run_id in runs:
        out += steps(project, run_id, traces.steps(run_id), cache, vocab)
    return out


def _inside(path: Path, folder: Path) -> bool:
    """Whether ``path`` is ``folder`` or under it, whatever the letter case (macOS and Windows file systems ignore it)."""
    a, b = str(path.resolve()).casefold(), str(folder.resolve()).casefold().rstrip(os.sep)
    return a == b or a.startswith(b + os.sep)


def export(workspace: Path, out: Path) -> dict[str, Any]:
    """Write ``trajectories.jsonl`` and ``manifest.json`` into ``out`` for the workspace's opted-in projects."""
    from ..agents.traces import open_traces
    from ..storage import open_stores

    if _inside(out, V3):
        raise SystemExit(f"refusing to write into the v3 corpus ({V3}): it is built by its own command; choose another folder")
    projects, _, sessions = open_stores(Path(workspace))
    traces = open_traces(Path(workspace))
    rows: list[dict[str, Any]] = []
    shared, skipped = [], 0
    for project in projects.list():
        if not opted_in(project):
            skipped += 1
            continue
        mine = records(project, projects, sessions, traces)
        rows += mine
        shared.append({"project_id": project["id"], "records": len(mine)})
    out.mkdir(parents=True, exist_ok=True)
    (out / "trajectories.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    manifest = {"version": VERSION, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "projects": len(shared),
                "not_opted_in": skipped, "records": len(rows), "by_kind": dict(Counter(r["kind"] for r in rows)), "items": shared,
                "kept": "state strings, the moves or tools offered, the move or tool chosen, arguments that are enums, numbers, booleans "
                        "or names of columns and ids, the verdict, cited record ids and what followed",
                "dropped": "every other string argument (the positive label, the prediction moment, reasons, questions, goals), tool "
                           "results, validator messages and outcome summaries"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest
