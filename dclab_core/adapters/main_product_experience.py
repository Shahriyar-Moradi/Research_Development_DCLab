"""The main product's own records, read as experience events (package 19.1).

DCLab's main product already records what people and agents do, in append-only tables: ``agent_runs`` and
``agent_events`` (each agent run or assistant thread and its events), ``llm_invocations`` (one row per gateway call),
``project_decision_records`` (the decision ledger) and ``ml_run_events`` (the pipeline's execution boundaries). Its
rules forbid a parallel system for an existing concern, so it adds no table for experience: these functions read its
rows (as plain dicts, column names as recorded in ``main_product_fields.json`` at one commit) and return
:class:`dclab_core.experience.ExperienceEvent`, so the timeline, the session card, redaction and every later builder
run on its data unchanged.

A row type with no matching kind is not invented into one: :data:`AGENT_EVENTS_NOT_MAPPED` and
:data:`DECISIONS_NOT_MAPPED` say why each is left out, and :data:`IDS_MISSING` names the ids the main product does
not store today (a visit and a request id), which its side of package 19.2 adds.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any, Optional

from dclab_core import errors
from dclab_core.adapters.main_product import recorded
from dclab_core.experience import ExperienceEvent, make
from dclab_core.steps import MAX_STEPS, Step

# agent_events.type -> kind
AGENT_EVENT_KINDS = {
    "user_message": "chat.person",
    "assistant_message": "chat.agent",
    "clarification_requested": "chat.agent",
    "llm_call_finished": "model.request",
    "tool_call_finished": "tool.call",
    "tool_call_denied": "tool.call",
    "step_rejected": "decision.refused",
    "proposal_created": "decision.proposed",
    "proposal_auto_applied": "decision.accepted",
    "run_failed": "error.raised",
}
AGENT_EVENTS_NOT_MAPPED = {
    "run_started": "the run's own row (agent_runs) says when it started",
    "context_built": "a digest of the context, no event of the person's or the agent's",
    "budget_reserved": "spend accounting; the cost is on the model request",
    "budget_exhausted": "spend accounting; the run then fails or finishes",
    "budget_settled": "spend accounting; the cost is on the model request",
    "llm_call_started": "the call is one event, when it finishes (llm_call_finished)",
    "tool_call_requested": "the call is one event, when it finishes or is denied",
    "step_validated": "a step that passed: the accepted proposal or tool call is the event",
    "run_finished": "the run's own row says when it finished",
    "thread_closed": "the thread's own row says it closed",
    "replay_checked": "a replay of an earlier run, not something a person or agent did",
}
# project_decision_records.decision_type -> kind
DECISION_KINDS = {
    "problem_spec_locked": "solution.saved",
    "split_plan_created": "decision.accepted",
    "winner_locked": "decision.accepted",
    "champion_promoted": "decision.accepted",
    "experiment_accepted": "decision.accepted",
    "decision_point_resolved": "decision.accepted",
    "operating_point_chosen": "decision.accepted",
    "proposal_accepted": "decision.accepted",
    "experiment_rejected": "decision.refused",
    "proposal_rejected": "decision.refused",
    "proposal_reverted": "decision.overridden",
    "ref_moved": "graph.move",
}
DECISIONS_NOT_MAPPED = {"ref_initialized": "the first pointer of a project, set by the system when it is created"}
ACTORS = {"human": "person", "rule": "system", "agent": "agent"}
# ml_run_events: the main product's own rules (workflow_execution_service._stage_status and apply_live_stage_from_event):
# a missing status is "completed"; events named cv_fold_* and candidate_* are steps inside a stage, not its boundaries;
# the "terminal" stage (or a pipeline_terminal event) closes the run
RUN_STATUS = {"queued": "queued", "running": "running", "completed": "completed", "failed": "failed", "skipped": "skipped",
              "started": "running", "fail": "failed", "error": "failed", "skip": "skipped", "not_applicable": "skipped"}
INTRA_STAGE_PREFIXES = ("cv_fold_", "candidate_")
TERMINAL_STAGE = "terminal"
# fields of its own that map onto a contract here but are not in the field lists read at the pinned commit (23.5, 23.7): a test fails
# when the lists gain them, so the mapping is written then, not guessed now
FIELDS_MISSING = {
    "dataset_columns.sensitivity_class": "expected to hold a column's sensitivity class (23.7: dclab_core.sensitivity.classify gives restricted, pii, "
                                         "sensitive or none); no dataset_* table is in the lists read, so its names and values are unverified",
}

# contract kinds the main product's tables cannot fill today, and why (never invented)
KINDS_MISSING = {
    "consent.changed": "no consent switch on its workspace rows (23.1): the host passes the level it allows when it maps them (CONSENT_MISSING)",
    "model.checked": "the verdict of the code that checks a model's reply is not a row of its own there (23.10): the host records it where it checks",
    "pipeline.step_started": "its fold and candidate rows are steps of a run, read by steps_from_run_events() (23.8), not events",
    "pipeline.step_finished": "its fold and candidate rows are steps of a run, read by steps_from_run_events() (23.8), not events",
    "outcome.observed": "an outcome is a row of its own there (23.5, adapters.main_product_outcomes), not an event of its agent_events",
    "visit.started": "no visit is recorded (19.2 adds the auth session as the visit)",
    "auth.sign_in": "auth_sessions and service_tokens rows are not in the field lists read at this commit (23.3; refreshed by 23.12)",
    "auth.sign_in_failed": "no failed sign-in row in the main product: a counter and a log line (23.3)",
    "auth.sign_out": "auth_sessions rows are not read yet (23.3)", "auth.session_expired": "auth_sessions rows are not read yet (23.3)",
    "auth.token_created": "service_tokens rows are not read yet (23.3)", "auth.token_revoked": "service_tokens rows are not read yet (23.3)",
    "auth.workspace_switched": "no row of its own in the main product (23.3)",
    "auth.role_changed": "membership rows are not in the field lists read at this commit (23.3)",
    "ui.page_viewed": "the web app's own events (23.2): its sender posts them through its API under the same kinds; this adapter reads tables",
    "ui.action": "the web app's own events (23.2)", "ui.upload_started": "the web app's own events (23.2)",
    "ui.upload_finished": "the web app's own events (23.2)", "ui.error_shown": "the web app's own events (23.2)",
    "visit.ended": "no visit is recorded (19.2)",
    "draft.created": "the main product has no draft before a project",
    "draft.edited": "no draft", "draft.built": "no draft", "draft.refused": "no draft",
    "data.uploaded": "dataset versions are not read by this adapter yet",
    "data.cleaned": "dataset versions are not read by this adapter yet",
    "data.analysed": "dataset investigations are proposals (agent_proposals), not read yet",
    "data.labels_joined": "no label-file join in the main product",
    "data.cases_found": "no list-of-cases recognition in the main product",
    "solution.proposed": "ExperimentPlanProposal rows (agent_proposals) are not read yet",
    "solution.refused": "a refused problem spec is a validator's step_rejected, read as decision.refused",
    "stage.approved": "approvals are decision records of other types",
    "problem.mapped": "the main product states one ProblemSpec per problem and has no map of parts (24.1): the host keeps the map beside the specs (adapters.main_product_problem_map.problem_specs)",
    "problem.part_changed": "no map of parts, so no edit of one (24.1, 24.2)",
    "solution.routed": "an approach is a proposal in a sub-problem's place in the map, not a row of its own there (24.3)",
    "solution.composed": "nothing there relates experiments or combines their outputs: the composition is the programme document's (24.5)",
    "programme.built": "a programme is a document the host keeps; each leaf is an ExperimentPlan proposal (adapters.main_product_problem_map.programme_experiments, 24.5)",
    "agent.step": "its agent steps are the agent_events read as their own kinds",
    "feedback.given": "no feedback table: a person's accept, reject and revert of a proposal map through feedback_events() (23.6)",
    "export.made": "downloads are not recorded",
    "job.started": "ml_jobs are not read yet", "job.stopped": "ml_jobs are not read yet",
}
TABLES_NOT_READ = {
    "agent_proposals": "proposals and their status; their decisions are read from project_decision_records",
    "execution_requests": "commands with the surface they came from; no request id is stored on them yet (19.2)",
    "data_access_events": "data access by purpose; read when 19.6's consent and retention work needs it",
    "ml_jobs": "the queue; its runs are read through ml_run_events",
}
IDS_MISSING = {"parent": "no parent id on its rows: an agent run's parent is its conversation (``conversation_of``) and a run record's parent travels in ``RunRecord.parent``, not in the event's ids",
               "visit": "no visit id on its rows: auth_sessions is per sign-in and not stamped on events (19.2)",
               "request": "X-Request-Id is bound per request but not stored (19.2); with_trace() stamps it where the host knows it (23.11)",
               "trace": "its agent_events, llm_invocations, ml_run_events and project_decision_records carry no trace_id yet (A.9 lists the column as its own change); with_trace() derives one from a request id (23.11)",
               "span": "no span_id on its rows (23.11); the same change as the trace_id",
               "parent_span": "no parent span id on its rows (23.11); the same change as the trace_id"}
CONSENT_MISSING = ("no consent level on its rows (23.1): the host passes the level its workspace settings allow when it maps them "
                   "(``events(..., consent=)``); a row mapped without one is ``operate`` and never leaves")


def _at(value: Any) -> str:
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if moment.tzinfo is None:  # its columns are timestamptz; a naive value read from them is UTC
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.isoformat(timespec="microseconds")


def _id(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _text(payload: Mapping[str, Any]) -> Optional[str]:
    """The message of a user_message or assistant_message (its payload key, read at the recorded commit)."""
    return payload["message"] if isinstance(payload.get("message"), str) else None


def conversation_of(run: Mapping[str, Any]) -> Optional[str]:
    """A thread is the conversation; each model turn is a ``lead`` run under it (parent_run_id)."""
    return _id(run.get("parent_run_id") if run.get("kind") == "lead" and run.get("parent_run_id") else run.get("id"))


def check_columns(table: str, row: Mapping[str, Any]) -> None:
    """A row must have only the columns recorded for its table at the read commit: a renamed column fails loudly."""
    known = set(recorded()["experience_tables"][table])
    unknown = set(row) - known
    if unknown:
        raise ValueError(f"{table}: columns not recorded at the read commit: {sorted(unknown)}; read the main product again")


def from_agent_event(row: Mapping[str, Any], run: Mapping[str, Any]) -> Optional[ExperienceEvent]:
    """One ``agent_events`` row (with its ``agent_runs`` row) as an event, or None for a type with no kind."""
    check_columns("agent_events", row)
    check_columns("agent_runs", run)
    kind = AGENT_EVENT_KINDS.get(row["type"])
    if kind is None:
        return None
    payload = dict(row.get("payload") or {})
    ids = {"workspace": _id(row["workspace_id"]), "user": _id(run.get("created_by_user_id")), "conversation": conversation_of(run),
           "project": _id(run.get("project_id")), "run": _id(run["id"]), "model_request": _id(row.get("llm_invocation_id"))}
    actor = {"chat.person": "person", "model.request": "model"}.get(kind, "agent")
    body: dict[str, Any] = {}
    if kind == "chat.person":
        body = {"text": _text(payload), "surface": run.get("kind")}
    elif kind == "chat.agent":
        body = {"text": _text(payload), "kind": "question" if row["type"] == "clarification_requested" else "text", "surface": run.get("kind")}
    elif kind == "model.request":
        body = {"purpose": run.get("purpose"), "model": run.get("model")}
    elif kind == "tool.call":  # the harness writes tool, ok and code; a denial carries its code
        verdict = "denied" if row["type"] == "tool_call_denied" else "error" if payload.get("ok") is False else "ok"
        body = {"tool": payload.get("tool"), "verdict": verdict, "code": payload.get("code")}
    elif kind in ("decision.proposed", "decision.accepted"):
        body = {"point": run.get("decision_point_key"), "by": run.get("agent_key")}
    elif kind == "decision.refused":  # step_rejected: one reason code, or the validator's list
        reasons = payload.get("reasons") or ([payload["reason"]] if payload.get("reason") else [])
        body = {"point": run.get("decision_point_key"), "rule": "validator", "choice": [str(r)[:80] for r in reasons][:20] or None}
    elif kind == "error.raised":
        code = payload.get("error_code") or run.get("error_code")
        body = {"where": run.get("agent_key"), "code": code, **(errors.fingerprint(str(code), []) if code else {})}  # 23.4: by type alone
    return make(kind, actor, ids, body, id=f"mp-ae-{row['id']}", at=_at(row["created_at"]), seq=int(row["seq"]),
                source=f"agent_events:{row['type']}")


def from_llm_invocation(row: Mapping[str, Any], runs: Mapping[Any, Mapping[str, Any]] | None = None) -> ExperienceEvent:
    """One ``llm_invocations`` row: the model request with its tokens, cost (in its own currency: its gateway writes USD)
    and outcome; no prompt is stored there. With ``runs``, the request joins its agent run's conversation."""
    check_columns("llm_invocations", row)
    cost = row.get("cost_micros")
    run = (runs or {}).get(row.get("agent_run_id")) if row.get("agent_run_id") is not None else None
    ids = {"workspace": _id(row["workspace_id"]), "project": _id(row.get("project_id")), "run": _id(row.get("agent_run_id") or row.get("workflow_run_id")),
           "job": _id(row.get("experiment_id")), "conversation": conversation_of(run) if run else _id(row.get("agent_run_id")),
           "model_request": _id(row["id"])}
    body = {"purpose": row.get("purpose"), "model": row.get("provider_resolved_model") or row.get("model"), "input_tokens": row.get("input_tokens"),
            "output_tokens": row.get("output_tokens"), "cost": round(cost / 1e6, 6) if cost is not None else None,
            "currency": (row.get("currency") or None) if cost is not None else None,
            "seconds": round(row["latency_ms"] / 1000, 3) if row.get("latency_ms") is not None else None,
            "outcome": row.get("refusal_code") or row.get("status")}
    return make("model.request", "model", ids, body, id=f"mp-llm-{row['id']}", at=_at(row.get("completed_at") or row["started_at"]),
                source="llm_invocations")


def from_decision_record(row: Mapping[str, Any]) -> Optional[ExperienceEvent]:
    """One ``project_decision_records`` row as a decision event, or None for a type with no kind."""
    check_columns("project_decision_records", row)
    kind = DECISION_KINDS.get(row["decision_type"])
    if kind is None:
        return None
    actor = ACTORS.get(row["actor_kind"], "system")
    ids = {"workspace": _id(row["workspace_id"]), "user": _id(row.get("actor_user_id")), "project": _id(row["project_id"]),
           "run": _id(row.get("actor_agent_run_id"))}
    point = row["decision_type"]
    reason = None if row.get("rationale_untrusted") else row.get("rationale")  # an untrusted rationale stays out
    body: dict[str, Any] = {
        "solution.saved": {"changed": point},
        "decision.accepted": {"point": point, "by": actor},
        "decision.refused": {"point": point, "rule": row.get("actor_rule") or actor},
        "decision.overridden": {"point": point, "reason": reason},
        "graph.move": {"move": point, "allowed": True},
    }[kind]
    return make(kind, actor, ids, body, id=f"mp-pdr-{row['id']}", at=_at(row["event_at"]), source=f"project_decision_records:{point}")


FEEDBACK_DECISIONS = {"proposal_accepted": ("thumbs", "helpful"), "proposal_rejected": ("thumbs", "unhelpful"), "proposal_reverted": ("override", "overridden")}


def with_trace(events: Iterable[ExperienceEvent], request_of: Any) -> list[ExperienceEvent]:
    """23.11: events with the trace of the request they belong to. ``request_of(event)`` gives the request id the host knows for an
    event (its ``request_id`` column, a log line's id) or None; the event gets that request id and the trace id
    ``tracing.trace_id_for`` derives from it: the id this studio's edge puts in the ``traceparent`` it answers a request that came
    WITHOUT one. A request that came with a ``traceparent`` is named from the trace it was given (the request id is the trace's first
    16 characters), so its trace is the caller's, not a hash of the id: the host that kept the header gives that trace itself. An
    event with no request is returned as it is."""
    from dclab_core.tracing import trace_id_for

    out = []
    for e in events:
        request = request_of(e)
        if not request:
            out.append(e)
            continue
        out.append(e.model_copy(update={"ids": e.ids.model_copy(update={"request": str(request)[:80], "trace": trace_id_for(str(request))})}))
    return out


def feedback_events(decisions: Iterable[Mapping[str, Any]], consent: str = "operate") -> list[ExperienceEvent]:
    """23.6: a person's accept, reject and revert of an agent's proposal (``project_decision_records``) as ``feedback.given``,
    beside the decision event the same row already is. Only rows a person made (``actor_kind`` user); the rationale is
    the person's reason unless it is untrusted; the target is the proposal's subject digest."""
    out = []
    for row in decisions:
        check_columns("project_decision_records", row)
        if row["decision_type"] not in FEEDBACK_DECISIONS or ACTORS.get(row["actor_kind"]) != "person":
            continue
        kind, rating = FEEDBACK_DECISIONS[row["decision_type"]]
        ref = re.sub(r"[^0-9a-zA-Z_.:-]", "", str(row.get("subject_digest") or row["id"]))[:80]
        reason = None if row.get("rationale_untrusted") else (row.get("rationale") or None)
        ids = {"workspace": _id(row["workspace_id"]), "user": _id(row.get("actor_user_id")), "project": _id(row["project_id"])}
        out.append(make("feedback.given", "person", ids, {"kind": kind, "on": f"proposal:{ref}" if ref else "proposal", "rating": rating, "reason": reason},
                        id=f"mp-fb-{row['id']}", at=_at(row["event_at"]), source=f"project_decision_records:{row['decision_type']}", consent=consent))
    return out


def from_run_event(row: Mapping[str, Any]) -> Optional[ExperienceEvent]:
    """One ``ml_run_events`` row as a stage boundary, by the main product's own rules; a step inside a stage (a fold, a
    candidate), a queued or skipped stage, and the terminal row are not boundaries of a stage (None)."""
    check_columns("ml_run_events", row)
    name, stage = str(row.get("event_type") or ""), str(row.get("stage") or "stage")
    status = RUN_STATUS.get(str(row.get("status") or "completed").lower(), "completed")
    if stage == TERMINAL_STAGE or name == "pipeline_terminal":
        kind, stage = ("stage.failed" if status == "failed" else "stage.finished" if status == "completed" else None), TERMINAL_STAGE
    elif name.startswith(INTRA_STAGE_PREFIXES):
        return None
    elif status == "running" or name.endswith("_started") or name == "operation_started":
        kind = "stage.started"
    elif status == "failed" or name.endswith("_failed"):
        kind = "stage.failed"
    elif status == "completed" or name == "operation_completed" or name.endswith("_completed"):
        kind = "stage.finished"
    else:
        return None
    if kind is None:
        return None
    ids = {"workspace": _id(row["workspace_id"]), "run": _id(row["workflow_run_id"]), "job": _id(row["experiment_id"])}
    data = dict(row.get("payload") or {})
    body: dict[str, Any] = {"stage": stage}
    if kind == "stage.finished" and row.get("duration_ms") is not None:
        body["seconds"] = round(row["duration_ms"] / 1000, 3)
    if kind == "stage.failed":
        body["code"] = data.get("failure_code") or None
        if body["code"]:
            body.update(errors.fingerprint(str(body["code"]), []))  # 23.4: the main product's rows group by their code
    return make(kind, "system", ids, body, id=f"mp-run-{row['id']}", at=_at(row["timestamp"]), seq=int(row["sequence"]),
                source=f"ml_run_events:{name}"[:120])


def steps_from_run_events(rows: Iterable[Mapping[str, Any]], *, run_id: Optional[str] = None) -> list[Step]:
    """23.8: the ``ml_run_events`` rows that are steps inside a stage (``cv_fold_*``, ``candidate_*``: the ones ``from_run_event``
    leaves out) as ``Step`` entries in the order of their ``sequence``: a fold is a ``fold`` step, a candidate scored a ``fit``,
    with the duration, the fold and the metric and value the payload names when they are numbers. The main product keeps no
    digest of what a step produced, so none is claimed; ``run_id`` keeps one run's rows when the rows of several are given."""
    mine = [r for r in rows if str(r.get("event_type") or "").startswith(INTRA_STAGE_PREFIXES)
            and (run_id is None or str(r.get("workflow_run_id")) == str(run_id))]
    out = []
    for index, row in enumerate(sorted(mine, key=lambda r: int(r["sequence"]))[:MAX_STEPS]):
        check_columns("ml_run_events", row)
        data = dict(row.get("payload") or {})
        name = str(row["event_type"])
        fold = data.get("fold", data.get("fold_index"))
        value = data.get("value", data.get("score"))
        out.append(Step(index=index, name=name[:120], kind="fold" if name.startswith("cv_fold") else "fit", params={"stage": str(row.get("stage") or "")[:40]},
                        fold=int(fold) if isinstance(fold, int) and not isinstance(fold, bool) and fold >= 0 else None,
                        metric=str(data["metric"])[:80] if isinstance(data.get("metric"), str) else None,
                        value=float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value else None,
                        seconds=round(float(row["duration_ms"]) / 1000, 4) if isinstance(row.get("duration_ms"), (int, float)) else 0.0))
    return out


def from_run_record(record: Any, run_row: Optional[Mapping[str, Any]] = None) -> ExperienceEvent:
    """A run's lineage (``dclab_core.lineage.RunRecord``, from ``adapters.main_product_lineage``) as ``run.recorded``."""
    from dclab_core.lineage import event_payload

    ids = {"workspace": record.workspace or str((run_row or {}).get("workspace_id") or "unknown"), "project": record.project, "run": record.run}
    return make("run.recorded", "system", ids, event_payload(record), id=f"mp-run-{record.run}"[:80], at=record.finished or record.started,
                source=f"lineage:{record.kind}")


def events(agent_events: Iterable[Mapping[str, Any]] = (), runs: Mapping[Any, Mapping[str, Any]] | None = None,
           invocations: Iterable[Mapping[str, Any]] = (), decisions: Iterable[Mapping[str, Any]] = (),
           run_events: Iterable[Mapping[str, Any]] = (), experiments: Iterable[Mapping[str, Any]] = (),
           stage_runs: Iterable[Mapping[str, Any]] = (), consent: str = "operate") -> list[ExperienceEvent]:
    """Every row read as an event, the unmapped ones left out. ``runs`` maps an agent run's id to its row; each finished
    agent run, experiment and stage run is also a ``run.recorded`` (19.5). ``consent``: the level the host's workspace
    allows now (``dclab_core.consent.level_of`` on its settings), stamped on every event (23.1; ``CONSENT_MISSING``)."""
    from dclab_core.consent import LEVELS

    if consent not in LEVELS:
        raise ValueError(f"consent is one of {', '.join(LEVELS)}")
    from dclab_core.adapters import main_product_lineage as lineage
    runs = runs or {}
    invocations = list(invocations)
    given = {str(r["id"]) for r in invocations}
    out = [from_llm_invocation(r, runs) for r in invocations]
    # an llm_call_finished is the same request as its invocation row: read once, from the row when it is given
    out += [e for r in agent_events if not (r["type"] == "llm_call_finished" and str(r.get("llm_invocation_id")) in given)
            for e in [from_agent_event(r, runs[r["run_id"]])] if e]
    out += [e for r in decisions for e in [from_decision_record(r)] if e]
    out += [e for r in run_events for e in [from_run_event(r)] if e]
    records = [lineage.from_agent_run(r) for r in (runs or {}).values()]
    records += [lineage.from_experiment(r) for r in experiments] + [lineage.from_stage_run(r) for r in stage_runs]
    out += [from_run_record(r) for r in records if r is not None]
    return [e if e.consent == consent else e.model_copy(update={"consent": consent}) for e in out]
