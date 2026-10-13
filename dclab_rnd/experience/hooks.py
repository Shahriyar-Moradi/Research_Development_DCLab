"""What the existing writers already record, emitted into the experience log as well (package 19.3).

The audit trail, the drafts' events, the projects' activity, the agents' traces and the model gateway each keep their
own store for their own readers; each also calls one function here, which maps its record onto the vocabulary of
``dclab_core.experience`` and records it with the ids it knows. A record the vocabulary has no kind for is skipped;
a mapping that fails is reported and skipped: a writer never fails because of its experience event.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

from . import model_text_allowed, record


def _quiet(fn: Callable[..., Any]) -> Callable[..., Any]:
    def run(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the writer's own record stands; its experience event is skipped
            print(f"experience: {fn.__name__} skipped ({type(exc).__name__})", file=sys.stderr, flush=True)
            return None
    run.__name__ = fn.__name__
    return run


def _actor(actor: Any) -> str:
    return "agent" if actor == "agent" else "system" if actor == "system" else "person"


@_quiet
def from_audit(row: dict[str, Any]) -> None:
    """An audit row (its arguments under ``detail.args``, its words under ``detail.message``): a validated move, a
    saved solution, a project's data import, an approval, a policy or a routing change, a lesson review, a removed note.
    A draft's own upload and the copies a build makes are recorded once, by the draft's events, not again here."""
    kind, detail = row.get("kind"), row.get("detail") or {}
    args = detail.get("args") if isinstance(detail.get("args"), dict) else {}
    ids = {"project": row.get("project_id"), "conversation": row.get("draft_id"), "user": row.get("user_id")}
    actor = _actor(row.get("actor"))
    if kind == "move":
        record("graph.move", actor, {"move": row.get("move"), "allowed": row.get("status") == "allowed",
                                     "from": detail.get("from"), "to": detail.get("to"),
                                     "failed_checks": [str(c)[:80] for c in detail.get("failed_checks") or []][:20] or None,
                                     "rules": [str(r)[:40] for r in detail.get("rules") or []][:20] or None}, **ids)
    elif kind == "solution":
        if detail.get("built"):  # the draft's solution copied into its project at build: saved once, in the draft
            return
        record("solution.saved", actor, {"target": detail.get("target"), "task": detail.get("task"),
                                         "forbidden": [str(c)[:120] for c in detail.get("forbidden") or []][:200] or None}, **ids)
    elif kind == "data_import":
        source = detail.get("source") if isinstance(detail.get("source"), dict) else {}
        if row.get("draft_id") or source.get("kind") == "draft":  # the draft's upload is its own event; the build's copy is no upload
            return
        record("data.uploaded", actor, {"name": str(detail.get("filename") or "")[:200] or None, "rows": detail.get("rows"),
                                        "columns": detail.get("columns"), "source": str(source.get("kind") or "")[:40] or None}, **ids)
    elif kind == "approval":
        record("decision.accepted", actor, {"point": f"gate:{str(args.get('gate') or '')[:60]}", "by": actor}, **ids)
    elif kind == "policy":
        record("decision.accepted", actor, {"point": f"policy:{str(args.get('policy') or '')[:60]}", "choice": "on" if args.get("on") else "off",
                                            "by": actor}, **ids)
    elif kind == "routing":
        record("decision.accepted", actor, {"point": f"routing:{str(args.get('purpose') or '')[:60]}",
                                            "choice": {k: str(args.get(k))[:80] for k in ("setting", "from", "to") if args.get(k) is not None} or None,
                                            "by": actor}, **ids)
    elif kind == "lesson_review":
        accepted = str(args.get("action") or "").startswith("accept")
        if accepted:
            record("decision.accepted", actor, {"point": "lesson", "choice": str(args.get("lesson") or "")[:60] or None, "by": actor}, **ids)
        else:
            record("decision.refused", actor, {"point": "lesson", "choice": str(args.get("lesson") or "")[:60] or None, "rule": "person",
                                               "reason": str(detail.get("message") or "")[:2000] or None}, **ids)
    elif kind == "memory_removed":
        record("decision.overridden", actor, {"point": "memory", "proposed": str(args.get("note") or "")[:60] or None,
                                              "reason": str(detail.get("message") or "")[:2000] or None}, **ids)


@_quiet
def from_draft_event(draft_id: str, kind: str, data: dict[str, Any] | None) -> None:
    """A draft's event (Home): the chat, the data pipeline, the analysis, the build."""
    data = data or {}
    ids = {"conversation": draft_id}
    if kind == "chat":
        message = data
        text = message.get("text")
        if message.get("role") == "user":
            record("chat.person", "person", {"text": text, "surface": "home"}, **ids)
        elif message.get("role") == "agent":
            question = message.get("question") or {}
            record("chat.agent", "agent", {"text": text, "kind": message.get("kind"), "question_field": question.get("field"),
                                           "options": [str(o)[:120] for o in question.get("options") or []][:12] or None,
                                           "surface": "home"}, **ids)
    elif kind == "pipeline":
        step, asset = data.get("step"), data.get("asset")
        if step == "queued" and isinstance(asset, dict):
            record("data.uploaded", "person", {"name": str(asset.get("name") or "")[:200] or None, "bytes": asset.get("bytes"),
                                               "source": asset.get("kind")}, **ids)
        elif step == "failed":
            record("error.raised", "system", {"where": "data pipeline", "code": str(data.get("code") or "pipeline_failed")[:80],
                                              "message": str(data.get("text") or data.get("error") or "")[:2000] or None}, **ids)
        elif step == "cleaned":
            log = data.get("log") or []
            record("data.cleaned", "system", {"steps": len(log), "rows_before": (log[0] or {}).get("rows_before") if log else None,
                                              "rows_after": (log[-1] or {}).get("rows_after") if log else None}, **ids)
    elif kind == "analysis":
        analysis = data.get("analysis") or {}
        profile = analysis.get("profile") or {}
        columns = profile.get("columns") or []
        roles = {str(c.get("name"))[:120]: _role(c) for c in columns[:200]}
        record("data.analysed", "system", {"rows": profile.get("rows"), "columns": profile.get("column_count") or len(columns),
                                           "column_names": [str(c.get("name"))[:120] for c in columns][:200] or None,
                                           "roles": roles or None}, **ids)
        cases = analysis.get("cases_table")
        if cases:
            record("data.cases_found", "system", {"keys": cases.get("keys"), "case_kinds": sorted(cases.get("case_kinds") or {}),
                                                  "tables_matched": len(cases.get("tables") or [])}, **ids)
    elif kind == "status" and data.get("built"):
        record("draft.built", "person", {"project": data.get("project_id")}, project=data.get("project_id"), **ids)


def _role(column: dict[str, Any]) -> str:
    for flag, role in (("id_like", "identifier"), ("time_like", "time"), ("text_like", "text"), ("target_name_like", "outcome-like")):
        if column.get(flag):
            return role
    return str(column.get("kind") or "column")


@_quiet
def from_activity(project_id: str, kind: str, payload: Any) -> None:
    """A project's activity: its stages, its data and labels, a question asked of its agent."""
    p = payload if isinstance(payload, dict) else {}
    ids = {"project": project_id}
    if kind == "stage_started":
        record("stage.started", "system", {"stage": p.get("stage")}, **ids)
    elif kind == "stage_completed":
        record("stage.finished", "system", {"stage": p.get("stage"), "seconds": p.get("elapsed_seconds"), "metric": p.get("metric"),
                                            "value": p.get("value"), "chosen": p.get("chosen")}, **ids)
    elif kind == "stage_failed":
        record("stage.failed", "system", {"stage": p.get("stage"), "error": str(p.get("error") or "")[:2000] or None}, **ids)
    elif kind == "stage_approved":
        record("stage.approved", "person", {"stage": p.get("stage"), "choice": p.get("choice")}, **ids)
    elif kind == "labels_attached":
        record("data.labels_joined", "system", {"key": p.get("key"), "rows_marked": p.get("rows_marked"),
                                                "label_rows": p.get("label_rows"), "not_found": p.get("not_found")}, **ids)
    elif kind == "question":
        record("chat.person", "person", {"text": p.get("question"), "surface": "project"}, **ids)
    elif kind == "answer":
        record("chat.agent", "agent", {"text": p.get("answer"), "kind": "answer", "surface": "project"}, **ids)


@_quiet
def from_trace(row: dict[str, Any]) -> None:
    """One agent step (the Home agent, the intern, the campaign): the step, and its tool call when it made one."""
    ids = {"run": row.get("run_id")}
    if row.get("agent") in ("home", "intern"):
        ids["conversation"] = row.get("run_id")
    from . import last_model_request

    record("agent.step", "agent", {"agent": row.get("agent"), "n": row.get("n"), "state": str(row.get("state") or "")[:300] or None,
                                   "reply": text if isinstance(text := row.get("reply_text"), str) and text else None, "tool": row.get("tool"),
                                   "verdict": row.get("verdict"), "seconds": row.get("seconds")}, model_request=last_model_request(), **ids)
    if row.get("tool"):
        arguments = row.get("arguments")
        names = sorted(arguments)[:50] if isinstance(arguments, dict) else None
        record("tool.call", "agent", {"tool": row.get("tool"), "argument_names": names, "arguments": arguments if isinstance(arguments, dict) else None,
                                      "verdict": row.get("verdict"), "result": row.get("result")}, **ids)


@_quiet
def from_model_request(entry: dict[str, Any], messages: list[dict[str, Any]] | None = None, reply: str | None = None) -> None:
    """One model request through the gateway: what it cost, and its prompt and reply when the workspace keeps them."""
    from ..models import settings as model_settings

    try:  # a purpose shown cell values (the parser's sample lines, the intern's tool results) keeps no prompt or reply
        shown_cells = int(getattr(model_settings.purpose(str(entry.get("purpose"))), "cell_values", 0) or 0) > 0
    except Exception:  # noqa: BLE001 - an unknown purpose is treated as one shown cells
        shown_cells = True
    keep = model_text_allowed() and not shown_cells
    prompt = "\n\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages or [] if isinstance(m.get("content"), str)) if keep else None
    record("model.request", "model", {"purpose": entry.get("purpose"), "tier": entry.get("tier"), "model": entry.get("model"),
                                      "input_tokens": entry.get("input_tokens"), "output_tokens": entry.get("output_tokens"),
                                      "cost": entry.get("cost_eur"), "currency": "EUR" if entry.get("cost_eur") is not None else None,
                                      "seconds": entry.get("seconds"), "outcome": entry.get("outcome"),
                                      "prompt": prompt[:20000] if prompt else None, "reply": (reply or None) if keep else None},
           project=entry.get("project_id"), conversation=entry.get("draft_id"), model_request=entry.get("id"), run=entry.get("run"))


@_quiet
def from_mcp_call(tool: str, arguments: dict[str, Any], result: Any) -> None:
    """A tool an MCP client (Hugging Face Chat UI, ML Intern) called through /mcp."""
    from ..agents.traces import arguments_of, summary, verdict_of

    record("tool.call", "agent", {"tool": tool, "scope": "mcp", "argument_names": sorted(arguments)[:50] or None,
                                  "arguments": arguments_of(arguments) or None, "verdict": verdict_of(result), "result": summary(result)},
           project=arguments.get("project_id") if isinstance(arguments.get("project_id"), str) else None)


@_quiet
def settings_changed(changes: dict[str, Any], **ids: Any) -> None:
    """A person changed how a draft or a project runs (split, folds, quick, rows, budget): a decision, with its values."""
    choice = {str(k)[:40]: v if isinstance(v, (bool, int, float)) or v is None else str(v)[:80] for k, v in changes.items()
              if not isinstance(v, (dict, list))}
    if choice:
        record("decision.accepted", "person", {"point": "settings", "choice": choice, "by": "person"}, **ids)
