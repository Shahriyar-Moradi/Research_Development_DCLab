"""Bounded agentic choice loop for Lead (Packages A1 and A2, Agentic_Plan_and_Context_Plan_2026-11-20).

At each decision point, host code builds an explicit menu from registered tools, workflow state, approvals, and
remaining budget, omitting any semantic choice that has no admitted implementation. The model picks one entry with
a brief reason and evidence references. Host code checks staleness, permissions, approvals, budget, and idempotency,
binds arguments to the authenticated case, executes at most one committed effect through the shared agent runtime,
persists the typed observation, and either stops or offers a fresh choice based on the new evidence.

Behind a workspace switch (``lead_choice.json`` / ``DCLAB_LEAD_BOUNDED_CHOICE``), initially off.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from .registry import Registry
from .runtime import Policy, run as run_policy

SWITCH_ENV = "DCLAB_LEAD_BOUNDED_CHOICE"
ADVISORY_ENV = "DCLAB_ADVISORY_GATED_ROLES"
SETTINGS_FILE = "lead_choice.json"
MAX_DECISIONS = 10
MAX_REVISIONS = 4
MAX_EXPERIMENTS = 4
OBSERVATION_STATUSES = ("succeeded", "blocked", "waiting_for_person", "stopped", "failed")
STOP_WORDS = re.compile(r"\b(?:stop|pause|hold on|do not run|don't run|cancel|wait here)\b", re.I)
COMPARE_TOOL = "training_compare"
REVISE_TOOL = "plan_revise"
# A replay returns the stored result without running again. Only for choices whose result is all they do: a question, options
# or a note replayed would reach the person as nothing (their words are written by the tool into this turn's state).
REPLAYABLE = frozenset({"compare", "consult", "inspect", "read_table", "query"})
NO_TO_PLAN = re.compile(r"^\W*(?:no|nope|not (?:yet|now|that)|don'?t|do not|keep the (?:old|current|first) plan)\b", re.I)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def settings(home: Path | None = None) -> dict[str, bool]:
    """Workspace switches for bounded Lead choice and advisory gated roles (both initially off)."""
    out = {"enabled": False, "advisory_gated_roles": False}
    if home is not None:
        path = Path(home) / SETTINGS_FILE
        if path.is_file():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(stored, dict):
                    if isinstance(stored.get("enabled"), bool):
                        out["enabled"] = stored["enabled"]
                    if isinstance(stored.get("advisory_gated_roles"), bool):
                        out["advisory_gated_roles"] = stored["advisory_gated_roles"]
            except (OSError, ValueError):
                pass
    env = os.environ.get(SWITCH_ENV, "").strip().lower()
    if env in ("1", "true", "on", "yes"):
        out["enabled"] = True
    elif env in ("0", "false", "off", "no"):
        out["enabled"] = False
    adv = os.environ.get(ADVISORY_ENV, "").strip().lower()
    if adv in ("1", "true", "on", "yes"):
        out["advisory_gated_roles"] = True
    elif adv in ("0", "false", "off", "no"):
        out["advisory_gated_roles"] = False
    return out


def is_enabled(home: Path | None = None) -> bool:
    return settings(home)["enabled"]


def advisory_roles_enabled(home: Path | None = None) -> bool:
    return settings(home)["advisory_gated_roles"]


def set_settings(home: Path, *, enabled: Optional[bool] = None, advisory_gated_roles: Optional[bool] = None) -> dict[str, bool]:
    """Persist workspace switches atomically without touching orchestra/team.json."""
    current = settings(home)
    if enabled is not None:
        current["enabled"] = bool(enabled)
    if advisory_gated_roles is not None:
        current["advisory_gated_roles"] = bool(advisory_gated_roles)
    path = Path(home) / SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{SETTINGS_FILE}.{secrets.token_hex(4)}.part")
    tmp.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return current


@dataclass(frozen=True)
class MenuOption:
    id: str  # ask, inspect, consult, compare, revise, stop
    tool: Optional[str]  # registered tool name in scope "case", "training_compare", "plan_revise", or None for stop
    summary: str
    requires_approval: bool = False
    defaults: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Menu:
    revision: str
    options: tuple[MenuOption, ...]
    evidence_refs: tuple[str, ...]
    remaining_eur: float
    approvals_pending: bool
    stop_requested: bool
    problem_revision: Optional[int] = None
    problem_summary: Optional[dict[str, Any]] = None

    def by_id(self, choice_id: str) -> Optional[MenuOption]:
        return next((o for o in self.options if o.id == choice_id), None)

    def as_prompt_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "menu_revision": self.revision,
            "remaining_eur": round(self.remaining_eur, 4),
            "approvals_pending": self.approvals_pending,
            "stop_requested": self.stop_requested,
            "evidence_refs": list(self.evidence_refs),
            "options": [{"id": o.id, "tool": o.tool, "summary": o.summary, "defaults": o.defaults} for o in self.options],
        }
        if self.problem_revision is not None:
            out["problem_revision"] = self.problem_revision
        if isinstance(self.problem_summary, Mapping):
            out["problem_context"] = {
                "project_id": self.problem_summary.get("project_id"),
                "problem_id": self.problem_summary.get("problem_id"),
                "problem_revision": self.problem_summary.get("problem_revision"),
                "resume_url": self.problem_summary.get("resume_url"),
                "text": self.problem_summary.get("text", ""),
                "active_facts": [
                    {"field": f.get("field"), "text": f.get("text"), "revision": f.get("revision"), "correction": bool(f.get("correction"))}
                    for f in (self.problem_summary.get("active_facts") or [])
                ],
                "stale_drafts": [
                    {"entry_id": d.get("entry_id"), "field": d.get("field"), "stale_reason": d.get("stale_reason"),
                     "changed_sources": d.get("changed_sources") or []}
                    for d in (self.problem_summary.get("pending_drafts") or []) if d.get("stale")
                ],
                "personal_preferences": [p.get("text") for p in (self.problem_summary.get("personal_preferences") or [])],
            }
        return out


@dataclass
class Observation:
    decision_id: str
    choice: str
    tool: Optional[str]
    status: str  # succeeded, blocked, waiting_for_person, stopped, failed
    reason: str
    evidence_refs: list[str]
    available_options: list[str]
    menu_revision: str
    result: dict[str, Any] = field(default_factory=dict)
    blocked_reason: Optional[str] = None
    committed_effects: int = 0
    idempotency_key: Optional[str] = None
    replayed: bool = False
    arguments_digest: Optional[str] = None  # the choice's host-bound arguments, hashed: a replay needs the same choice and arguments
    at: str = field(default_factory=_now)

    def to_record(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "choice": self.choice,
            "tool": self.tool,
            "status": self.status,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
            "available_options": list(self.available_options),
            "menu_revision": self.menu_revision,
            "result": self.result,
            "blocked_reason": self.blocked_reason,
            "committed_effects": self.committed_effects,
            "idempotency_key": self.idempotency_key,
            "replayed": self.replayed,
            "arguments_digest": self.arguments_digest,
            "at": self.at,
        }


def _fact_fingerprint(state: Any) -> str:
    """Fingerprint of user facts, case, table, problem memory, and plan (excluding observation count)."""
    case = state.case
    table_sig = (state.table.get("table"), [c.get("name") for c in (state.table.get("columns") or [])]) if isinstance(state.table, dict) else None
    payload = {
        "case_id": getattr(case, "case_id", getattr(case, "id", "")),
        "case_version": getattr(case, "version", 0),
        "table": table_sig,
        "constraints": [f.value for f in (getattr(case, "constraints", None) or [])],
        "unknowns": [u.field for u in (getattr(case, "unknowns", None) or [])],
        "findings": len(getattr(case, "findings", None) or []),
        "problem_revision": getattr(state, "problem_revision", None),
        "plan": str(getattr(state, "plan", "") or ""),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def state_revision(state: Any) -> str:
    """Stable revision fingerprint of the current case, table, problem memory, and recorded observations."""
    case = state.case
    table_sig = (state.table.get("table"), [c.get("name") for c in (state.table.get("columns") or [])]) if isinstance(state.table, dict) else None
    payload = {
        "case_id": getattr(case, "case_id", getattr(case, "id", "")),
        "case_version": getattr(case, "version", 0),
        "table": table_sig,
        "constraints": [f.value for f in (getattr(case, "constraints", None) or [])],
        "unknowns": [u.field for u in (getattr(case, "unknowns", None) or [])],
        "findings": len(getattr(case, "findings", None) or []),
        "decisions": len(getattr(state, "decisions", None) or []),
        "revisions": int(getattr(state, "decision_revisions", 0)),
        "experiments": int(getattr(state, "decision_experiments", 0)),
        "approvals_pending": bool(getattr(state, "approvals_pending", False)),
        "problem_revision": getattr(state, "problem_revision", None),
        "plan": str(getattr(state, "plan", "") or ""),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _evidence_refs(state: Any) -> tuple[str, ...]:
    refs: list[str] = []
    case = state.case
    if getattr(case, "goal", None):
        refs.append("case:goal")
    if getattr(case, "decision", None):
        refs.append("case:decision")
    if getattr(case, "data", None):
        refs.append("case:data")
    for idx, _ in enumerate(getattr(case, "constraints", None) or []):
        refs.append(f"case:constraint:{idx}")
    if isinstance(state.table, dict) and state.table.get("table"):
        refs.append(f"table:{state.table['table']}")
    for idx, _ in enumerate(getattr(case, "findings", None) or []):
        refs.append(f"case:finding:{idx}")
    for d in (getattr(state, "decisions", None) or [])[-5:]:
        if d.get("decision_id"):
            refs.append(f"decision:{d['decision_id']}")
    if getattr(state, "problem_revision", None):
        refs.append(f"problem:rev:{state.problem_revision}")
    return tuple(refs)


def _outcome_candidate(state: Any) -> Optional[str]:
    if not isinstance(state.table, dict):
        return None
    columns = [str(c.get("name")) for c in (state.table.get("columns") or []) if c.get("name")]
    if not columns:
        return None
    text = " ".join(str(t.get("content") or "") for t in (getattr(state, "turns", None) or []) if t.get("role") == "user")
    if getattr(state.case, "goal", None):
        text = f"{state.case.goal.value} {text}"
    for col in columns:
        if re.search(rf"(?<![\w]){re.escape(col)}(?![\w])", text):
            return col
    for col in reversed(columns):
        if any(tok in col.lower() for tok in ("target", "label", "minutes", "churn", "fraud", "team", "outcome")):
            return col
    return columns[-1]


def _has_tool(registry: Registry, name: str) -> bool:
    try:
        return registry.get(name) is not None
    except KeyError:
        return False


def _sync_problem_context(state: Any) -> Optional[dict[str, Any]]:
    """Read the live versioned resumable summary for a bound problem and revalidate pending Advisor recommendations."""
    scope = getattr(state, "authorized_scope", None)
    home = getattr(state, "workspace_home", None)
    if scope is None or home is None or not (getattr(scope, "project_id", None) and getattr(scope, "problem_id", None)):
        return getattr(state, "problem_summary", None)
    try:
        from ..core_advisor import check_recommendation_freshness
        from ..memory import summary as mem_summary

        doc = mem_summary.read_summary(Path(home), scope, auto_rebuild=True, include_personal_overlay=True)
        doc["resume_url"] = f"/projects/{scope.project_id}?problem={scope.problem_id}"
        state.problem_summary = doc
        state.problem_revision = int(doc.get("problem_revision") or 0)
        pending = list(getattr(state, "pending_recommendations", None) or [])
        if pending:
            state.pending_recommendations = [check_recommendation_freshness(r, doc) for r in pending]
        return doc
    except Exception:  # noqa: BLE001 - never break non-memory sessions if summary is unavailable
        return getattr(state, "problem_summary", None)


def build_menu(state: Any, registry: Registry, *, allow_a2: bool = True, max_revisions: int = MAX_REVISIONS,
               max_experiments: int = MAX_EXPERIMENTS) -> Menu:
    """Construct the host-authorized choice menu from registered tools, workflow state, approvals, and budget."""
    prob_summary = _sync_problem_context(state)
    prob_rev = int(prob_summary["problem_revision"]) if isinstance(prob_summary, Mapping) and prob_summary.get("problem_revision") is not None else getattr(state, "problem_revision", None)
    remaining = max(0.0, float(getattr(state, "cap_eur", 0.0)) - float(getattr(state, "spent_eur", 0.0)))
    approvals_pending = bool(getattr(state, "approvals_pending", False))
    last_user = next((str(t.get("content") or "") for t in reversed(getattr(state, "turns", None) or []) if t.get("role") == "user"), "")
    stop_requested = bool(STOP_WORDS.search(last_user))
    options: list[MenuOption] = []

    if _has_tool(registry, "case_ask") and not getattr(state, "waiting_on_budget", False) and not approvals_pending:
        options.append(MenuOption(id="ask", tool="case_ask",
                                  summary="Ask the person one clarifying question (for example when prediction time is unknown)."))

    if stop_requested:  # the person said stop: nothing runs; the Lead may only stop, or ask what they want next
        options.append(MenuOption(id="stop", tool=None, summary="Pause or stop cleanly without running any tool or changing state."))
        return Menu(revision=state_revision(state), options=tuple(options), evidence_refs=_evidence_refs(state),
                    remaining_eur=remaining, approvals_pending=approvals_pending, stop_requested=True,
                    problem_revision=(int(prob_rev) if isinstance(prob_rev, int) else None),
                    problem_summary=prob_summary if isinstance(prob_summary, dict) else None)

    if state.table is not None and not approvals_pending:
        outcome = _outcome_candidate(state)
        if outcome and _has_tool(registry, "case_check_leakage"):
            options.append(MenuOption(id="inspect", tool="case_check_leakage",
                                      summary="Run read-only leakage and column availability inspection on the attached table.",
                                      defaults={"outcome": outcome}))
        elif _has_tool(registry, "case_read_table"):
            options.append(MenuOption(id="inspect", tool="case_read_table",
                                      summary="Inspect table columns and metadata (read-only, no cell values)."))

    # Preserve 100% of existing registered Lead capabilities so enabling the bounded choice loop
    # strictly increases AI and DCLab capability and never narrows available tools.
    if not approvals_pending:
        native_always = (
            ("note", "case_note", "Record stated facts, constraints, or assumptions into the case file."),
            ("add_unknown", "case_add_unknown", "Record an open unknown and which options wait for it."),
            ("resolve", "case_resolve", "Mark an open unknown resolved."),
            ("confirm_assumption", "case_confirm_assumption", "Record user confirmation or rejection of an assumption."),
            ("propose", "case_propose", "Present two or three contrasting options for the person to choose from."),
            ("start", "case_start", "Start the work or track the person chose."),
            ("request_build", "case_request_build", "Request building the DCLab Studio project once the person approved."),
        )
        for opt_id, tool_name, desc in native_always:
            if _has_tool(registry, tool_name):
                options.append(MenuOption(id=opt_id, tool=tool_name, summary=desc))

        if state.table is not None:
            native_table = (
                ("read_table", "case_read_table", "Read table columns, kinds, notices, and row count (no cell values)."),
                ("query", "case_query", "Run a read-only aggregate query on the attached table."),
                ("answer", "case_answer", "Answer a data question with a verified card from the attached table."),
                ("dashboard", "case_dashboard", "Build a multi-tile dashboard card from the attached table."),
                ("preview_step", "case_preview_step", "Preview a preprocessing step on the attached table."),
            )
            for opt_id, tool_name, desc in native_table:
                if _has_tool(registry, tool_name) and not any(o.tool == tool_name for o in options):
                    options.append(MenuOption(id=opt_id, tool=tool_name, summary=desc))
        elif _has_tool(registry, "case_design_solution_without_data"):
            options.append(MenuOption(
                id="blueprint", tool="case_design_solution_without_data",
                summary="Design a 3-layer Business + Data Science + ML blueprint when no table has been uploaded yet.",
            ))

    if allow_a2 and not approvals_pending:
        from . import specialists as _spec

        consulted = list(getattr(state, "consulted", None) or [])
        if (_has_tool(registry, "case_consult") and getattr(state, "specialist_client", None) is not None
                and remaining > 0.0 and len(consulted) < _spec.MAX_CONSULTS):
            allowed_specs = [
                s for s in _spec.SPECIALISTS
                if _spec.allowed(
                    s, state.case, state.table, consulted,
                    virtual_columns=getattr(state, "virtual_columns", None),
                    has_blueprint=bool(getattr(state, "blueprint", None)),
                ) is None
            ]
            if allowed_specs:
                options.append(MenuOption(
                    id="consult", tool="case_consult",
                    summary="Consult a specialist Advisor for evidence, uncertainty, and a suggested next check.",
                    defaults={"allowed_specialists": allowed_specs, "specialist": allowed_specs[0]},
                ))

        outcome = _outcome_candidate(state)
        if (getattr(state, "frame", None) is not None and outcome
                and int(getattr(state, "decision_experiments", 0)) < max_experiments
                and remaining >= 0.0):
            options.append(MenuOption(id="compare", tool=COMPARE_TOOL,
                                      summary=("Compare two feature sets on the provisional training rows (the earliest rows by time when "
                                               "the table has a time column), with time-ordered or grouped folds. Identifiers, personal "
                                               "columns and columns flagged as written after the outcome are refused."),
                                      defaults={"outcome": outcome}))

        has_evidence = bool(getattr(state, "decisions", None) or getattr(state.case, "findings", None) or getattr(state, "computed", None))
        if has_evidence and int(getattr(state, "decision_revisions", 0)) < max_revisions:
            options.append(MenuOption(id="revise", tool=REVISE_TOOL,
                                      summary=("Propose a revised working plan in response to new evidence. It is shown to the person and "
                                               "used only after they say yes.")))

    options.append(MenuOption(id="stop", tool=None, summary="Pause or stop cleanly without running any tool or changing state."))
    return Menu(revision=state_revision(state), options=tuple(options), evidence_refs=_evidence_refs(state),
                remaining_eur=remaining, approvals_pending=approvals_pending, stop_requested=stop_requested,
                problem_revision=(int(prob_rev) if isinstance(prob_rev, int) else None),
                problem_summary=prob_summary if isinstance(prob_summary, dict) else None)


class _BoundOnceClient:
    """Feeds one host-validated tool call into runtime.run so every execution passes Registry, CaseGuard, and Tracer."""

    def __init__(self, tool_name: str, arguments: dict[str, Any]):
        self.tool_name = tool_name
        self.arguments = dict(arguments)
        self.used = False

    def complete(self, messages: list[dict[str, Any]], tools: Any = None, max_tokens: Any = None) -> dict[str, Any]:
        if self.used:
            return {"content": "", "tool_calls": [], "usage": {}, "assistant_message": {"role": "assistant", "content": ""}}
        self.used = True
        call_id = "bc_" + secrets.token_hex(4)
        call = {"id": call_id, "name": self.tool_name, "arguments": self.arguments}
        assistant_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": self.tool_name, "arguments": json.dumps(self.arguments, default=str)}}],
        }
        return {"content": "", "tool_calls": [call], "usage": {}, "assistant_message": assistant_msg}


def _args_digest(choice: str, arguments: Mapping[str, Any]) -> str:
    body = json.dumps({"choice": choice, "arguments": arguments}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:20]


def _idempotency_key(case_id: str, facts: str, choice: str, arguments: Mapping[str, Any]) -> str:
    """The host's key of one choice: the case, the facts it was chosen on (never the menu's revision, which every observation
    moves), the choice and its bound arguments. The same check on the same facts is one effect."""
    body = json.dumps({"case_id": case_id, "facts": facts, "choice": choice, "arguments": arguments}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:20]


def _as_int(value: Any) -> Optional[int]:
    """An integer revision, or None for anything else (a model may write "latest", 2.5 or true)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        return int(value)
    return None


# ------------------------------------------------------------------------------------------------ the training comparison

REFUSED_PURPOSES = ("identifier", "leak_suspect")


def _epoch_seconds(values: pd.Series) -> pd.Series:
    """A time column as seconds since 1970, time zones dropped (a zone-aware column cannot be subtracted from a naive one)."""
    stamps = pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None)
    return (stamps - pd.Timestamp("1970-01-01")).dt.total_seconds().astype(float)


def _leak_columns(frame: pd.DataFrame, outcome: str, rows: pd.Index) -> set[str]:
    """The columns dclab_core's leakage check flags on the rows read: the later column of a recomputed target, else the
    column (or the columns) a finding names."""
    from dclab_core.analyse import analyse

    found = analyse({"table": frame}, target=outcome, train_index=rows)
    out: set[str] = set()
    for f in found.findings:
        if f.topic != "leakage":
            continue
        ev = f.evidence or {}
        if ev.get("later"):
            out.add(str(ev["later"]))
        elif ev.get("column"):
            out.add(str(ev["column"]))
        else:
            out.update(str(c) for c in (ev.get("columns") or []))
    return out


def _training_compare(state: Any, outcome: str, features_a: list[str], features_b: list[str]) -> dict[str, Any]:
    """Two feature sets compared on the provisional training rows (dclab_core.modifiers.provisional_reading: the earliest 75 %
    by the time column when the table has one, so a later holdout is never read; else a seeded 75 %, which a random holdout
    drawn at the build overlaps, and the result says so). Folds follow the time, and keep an entity's rows together when the
    data card says the rows of one entity share their outcome. Identifiers, personal columns and every column the data card
    or the leakage check flags as written after the outcome are refused. Never raises: a failure is an error result."""
    try:
        return _training_compare_unchecked(state, outcome, features_a, features_b)
    except Exception as exc:  # noqa: BLE001 - a table the comparison cannot read is a failed observation, never a crash
        return {"error": f"the comparison could not run: {type(exc).__name__}"}


def _training_compare_unchecked(state: Any, outcome: str, features_a: list[str], features_b: list[str]) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

    from dclab_core.datacard import card_of
    from dclab_core.modifiers import provisional_reading

    frame = getattr(state, "frame", None)
    if frame is None or not isinstance(frame, pd.DataFrame) or outcome not in frame.columns:
        return {"error": f"outcome {outcome!r} is not available in the attached table"}
    frame = frame.rename(columns=str)
    zoned = [c for c in frame.columns if isinstance(frame[c].dtype, pd.DatetimeTZDtype)]
    if zoned:  # times in UTC without their zone: the data card subtracts one time column from another
        frame = frame.assign(**{c: frame[c].dt.tz_convert(None) for c in zoned})
    present = frame.index[frame[outcome].notna()]
    if len(present) < 20:
        return {"error": "fewer than 20 labelled rows available for training-fold comparison"}
    card = card_of({"table": frame.loc[present]}, target=outcome)
    purposes = {c.name: (c.purpose, c.why) for c in card.tables[0].columns}
    time_col = next((c for c, (p, _) in purposes.items() if p == "time" and c != outcome), None)
    rows, by_time = provisional_reading(present, frame.loc[present, time_col] if time_col else None)
    rows = pd.Index(rows)
    if len(rows) < 16:
        return {"error": "fewer than 16 provisional training rows to compare on"}
    leaks = _leak_columns(frame, outcome, rows)
    personal = {str(c) for c in (getattr(state, "personal", None) or set())}
    refused: list[dict[str, str]] = []
    for col in dict.fromkeys([*features_a, *features_b]):
        why = None
        if col == outcome:
            why = "it is the outcome"
        elif col not in frame.columns:
            why = "the table has no such column"
        elif col in personal:
            why = "a personal column"
        elif purposes.get(col, (None, ""))[0] in REFUSED_PURPOSES:
            why = f"{purposes[col][0].replace('_', ' ')}: {purposes[col][1]}"
        elif col in leaks:
            why = "the leakage check flags it as written after the outcome"
        if why:
            refused.append({"column": col, "why": why[:200]})
    if refused:
        return {"error": "these columns may not be compared: " + ", ".join(r["column"] for r in refused), "refused": refused}
    cols_a, cols_b = list(dict.fromkeys(features_a)), list(dict.fromkeys(features_b))
    if not cols_a or not cols_b:
        return {"error": "both feature sets must name at least one allowed non-target column"}
    group_col = next((c for c, (p, why) in purposes.items() if p == "entity" and "split by it" in str(why)), None)

    train = frame.loc[rows]
    if by_time and time_col:
        train = train.assign(_order=_epoch_seconds(train[time_col])).sort_values("_order", kind="stable").drop(columns="_order")
    train = train.reset_index(drop=True)
    y_raw = train[outcome]
    is_numeric = pd.api.types.is_numeric_dtype(y_raw) and not pd.api.types.is_bool_dtype(y_raw) and y_raw.nunique(dropna=True) > 10
    task = "regression" if is_numeric else "classification"
    if task == "classification" and y_raw.astype(str).nunique() < 2:
        return {"error": "the outcome has one value in the rows read: there is nothing to compare"}
    groups = train[group_col].astype(str).to_numpy() if group_col else None

    if by_time and time_col:
        folds_kind = "time_ordered" + ("_grouped" if groups is not None else "")
        splits = list(TimeSeriesSplit(n_splits=3).split(train))
        if groups is not None:  # an entity's rows never on both sides of a fold
            splits = [(fi[~np.isin(groups[fi], groups[vi])], vi) for fi, vi in splits]
    elif groups is not None:
        n_groups = len(set(groups))
        if n_groups < 2:
            return {"error": f"{group_col} has one entity in the rows read: grouped folds cannot be drawn"}
        folds_kind = "grouped"
        splits = list(GroupKFold(n_splits=min(3, n_groups)).split(train, groups=groups))
    elif task == "classification":
        counts = y_raw.astype(str).value_counts()
        if int(counts.min()) >= 2:
            folds_kind = "stratified"
            splits = list(StratifiedKFold(n_splits=min(3, int(counts.min())), shuffle=True, random_state=0).split(train, y_raw.astype(str)))
        else:
            folds_kind = "shuffled"
            splits = list(KFold(n_splits=3, shuffle=True, random_state=0).split(train))
    else:
        folds_kind = "shuffled"
        splits = list(KFold(n_splits=3, shuffle=True, random_state=0).split(train))

    def _encode_fold(fit_df: pd.DataFrame, val_df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        fit_parts, val_parts = {}, {}
        for col in cols:
            if pd.api.types.is_datetime64_any_dtype(fit_df[col]):
                f, v = _epoch_seconds(fit_df[col]), _epoch_seconds(val_df[col])
                med = float(f.median()) if f.notna().any() else 0.0
                fit_parts[col], val_parts[col] = f.fillna(med), v.fillna(med)
            elif pd.api.types.is_numeric_dtype(fit_df[col]) or pd.api.types.is_bool_dtype(fit_df[col]):
                f = pd.to_numeric(fit_df[col], errors="coerce").astype(float)
                v = pd.to_numeric(val_df[col], errors="coerce").astype(float)
                med = float(f.median()) if f.notna().any() else 0.0
                fit_parts[col], val_parts[col] = f.fillna(med), v.fillna(med)
            else:  # fold-local frequency encoding, learned on the fit rows only
                freq = fit_df[col].astype(str).value_counts(normalize=True).to_dict()
                fit_parts[col] = fit_df[col].astype(str).map(freq).fillna(0.0).astype(float)
                val_parts[col] = val_df[col].astype(str).map(freq).fillna(0.0).astype(float)
        return pd.DataFrame(fit_parts), pd.DataFrame(val_parts)

    def _cv_score(cols: list[str]) -> Optional[float]:
        y = y_raw.astype(float).to_numpy() if task == "regression" else y_raw.astype(str).to_numpy()
        scores = []
        for fi, vi in splits:
            if len(fi) < 8 or len(vi) == 0 or (task == "classification" and len(set(y[fi])) < 2):
                continue  # a fold with too few rows, or one class to learn, is skipped rather than fitted
            xf, xv = _encode_fold(train.iloc[fi], train.iloc[vi], cols)
            if task == "regression":
                m = HistGradientBoostingRegressor(max_iter=20, random_state=0).fit(xf, y[fi])
                scores.append(float(np.mean(np.abs(y[vi] - m.predict(xv)))))
            else:
                m = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(xf, y[fi])
                scores.append(float(np.mean(m.predict(xv) == y[vi])))
        return round(float(np.mean(scores)), 4) if scores else None

    score_a, score_b = _cv_score(cols_a), _cv_score(cols_b)
    if score_a is None or score_b is None:
        return {"error": "no fold had enough rows (and both classes) to fit on"}
    return {
        "scope": "provisional_training_rows",
        "rows_read": "earliest_by_time" if by_time else "seeded_share",
        "later_rows_unseen": bool(by_time),  # true only of a holdout of the latest rows by this time column (the build's when it splits by it)
        "note": (f"Read the earliest rows by {time_col}: a holdout of the latest rows by {time_col} is never read; a random "
                 "holdout drawn at the build would share rows with these." if by_time else
                 "No column orders the rows in time, so a seeded share was read: a random holdout drawn at the build "
                 "shares about three quarters of its rows with these."),
        "time_column": time_col if by_time else None,
        "group_column": group_col,
        "folds": folds_kind,
        "train_rows": int(len(train)),
        "task": task,
        "metric": "mae" if task == "regression" else "accuracy",
        "set_a": {"features": cols_a, "score": score_a},
        "set_b": {"features": cols_b, "score": score_b},
        "delta_b_minus_a": round(score_b - score_a, 4),
    }


# ------------------------------------------------------------------------------------------------ the plan revision

def _revise_plan(state: Any, arguments: Mapping[str, Any], reason: str) -> dict[str, Any]:
    """A revised working plan, proposed: kept pending in the observation and used only after the person says yes
    (``_resolve_pending_revision``); ``state.plan`` is never changed here."""
    excluded = [str(c) for c in (arguments.get("exclude_columns") or []) if c]
    next_steps = [str(s) for s in (arguments.get("steps") or []) if s]
    if int(getattr(state, "decision_revisions", 0)) >= MAX_REVISIONS:
        return {"error": "revision limit reached"}
    if not next_steps and not excluded:
        return {"error": "a revision names its steps or the columns it would leave out"}
    lines = [f"{i + 1}. {step}" for i, step in enumerate(next_steps)]
    if excluded:
        lines.append("Leave out: " + ", ".join(excluded))
    state.decision_revisions = int(getattr(state, "decision_revisions", 0)) + 1
    return {
        "proposed": True,
        "pending": True,
        "applied": False,
        "revision": state.decision_revisions,
        "provisional_exclusions": excluded,
        "plan": "\n".join(lines),
        "requires_approval": True,
    }


def _pending_revision(state: Any) -> Optional[dict[str, Any]]:
    """The last revision proposed and not yet answered by the person, or None."""
    for rec in reversed(list(getattr(state, "decisions", None) or [])):
        if rec.get("choice") != "revise" or rec.get("status") not in ("waiting_for_person", "succeeded", "stopped"):
            continue
        result = rec.get("result") or {}
        return rec if rec.get("status") == "waiting_for_person" and result.get("pending") else None
    return None


def _resolve_pending_revision(state: Any, words: str) -> Optional[Observation]:
    """The person's answer to a proposed revision: a yes applies its plan, a no drops it, anything else leaves it pending."""
    pending = _pending_revision(state)
    if pending is None:
        return None
    from . import case_agent

    words = str(words or "")
    yes = bool(getattr(state, "go_ahead", False)) or case_agent.says_yes(words)
    no = bool(NO_TO_PLAN.search(words)) or case_agent.declines(words)
    if yes == no:
        return None
    result = dict(pending.get("result") or {})
    if yes:
        state.plan = str(result.get("plan") or "")
    obs = Observation(
        decision_id="dec_" + secrets.token_hex(5), choice="revise", tool=REVISE_TOOL, status="succeeded" if yes else "stopped",
        reason="the person approved the revised plan" if yes else "the person declined the revised plan",
        evidence_refs=[f"decision:{pending.get('decision_id')}"], available_options=[], menu_revision=state_revision(state),
        result={"pending": False, "applied": yes, "declined": not yes, "plan": result.get("plan"), "revision": result.get("revision"),
                "answers": pending.get("decision_id")},
        committed_effects=1 if yes else 0)
    _persist_observation(state, obs)
    return obs


# ------------------------------------------------------------------------------------------------ the loop's own tools
# Registered once, in a registry of the bounded loop's own: registering them in default_registry() would offer them to every
# reader of the case scope and change the case tools' fingerprint (test_case_agent pins the case scope to TOOL_NAMES, the
# prompt gate hashes it). They run through runtime.run like every other tool, so each call is guarded and traced.

def _compare_tool(state: Any, outcome: str, features_a: Optional[list[str]] = None, features_b: Optional[list[str]] = None) -> dict[str, Any]:
    if int(getattr(state, "decision_experiments", 0)) >= MAX_EXPERIMENTS:
        return {"error": "experiment cap reached"}
    res = _training_compare(state, str(outcome or ""), [str(c) for c in features_a or [] if c], [str(c) for c in features_b or [] if c])
    if "error" not in res:
        state.decision_experiments = int(getattr(state, "decision_experiments", 0)) + 1
        state.computed.append(json.dumps(res, default=str))  # what code computed: its numbers may be said
    return res


def _revise_tool(state: Any, exclude_columns: Optional[list[str]] = None, steps: Optional[list[str]] = None) -> dict[str, Any]:
    return _revise_plan(state, {"exclude_columns": exclude_columns or [], "steps": steps or []}, "")


class _LeadGuard:
    """The bounded loop's own writes: a case in hand, no rewrite under way, no approval pending, no stop asked for."""

    def __call__(self, tool: Any, context: Any, arguments: dict[str, Any], proceed: Any) -> Any:
        if getattr(context, "case", None) is None:
            return {"error": "the Lead's tools need a case"}
        if getattr(context, "rewriting", False):
            return {"error": "no tool runs while the message is being written again"}
        if getattr(context, "approvals_pending", False):
            return {"error": "waiting for the person's approval: nothing else runs"}
        return proceed()


_OWN: dict[str, Registry] = {}


def _own_registry() -> Registry:
    if "registry" not in _OWN:
        from .registry import Tool

        reg = Registry()
        A = {"type": "array", "items": {"type": "string"}}
        reg.register(Tool(name=COMPARE_TOOL, description="Compare two feature sets on the provisional training rows.",
                          parameters={"type": "object", "properties": {"outcome": {"type": "string"}, "features_a": A, "features_b": A},
                                      "required": ["outcome", "features_a", "features_b"]},
                          handler=_compare_tool, effect="write", scope="case", takes_context=True, move="lead.compare"))
        reg.register(Tool(name=REVISE_TOOL, description="Propose a revised working plan; used only after the person says yes.",
                          parameters={"type": "object", "properties": {"exclude_columns": A, "steps": A}},
                          handler=_revise_tool, effect="write", scope="case", takes_context=True, move="lead.revise"))
        reg.guards["case"] = _LeadGuard()
        _OWN["registry"] = reg
    return _OWN["registry"]


# ------------------------------------------------------------------------------------------------ the record

def _persist_observation(state: Any, obs: Observation) -> None:
    """Persist decision + observation into state, checkpoint, experience log, and Problem Base before the next choice. The
    experience events are dclab_core's decision kinds with names and codes only; the model's own words are never written
    into the shared problem record."""
    from .. import experience
    from . import case_session

    rec = obs.to_record()
    state.decisions.append(rec)
    if obs.idempotency_key and obs.committed_effects > 0:
        state.decision_effects[obs.idempotency_key] = rec
    draft = getattr(state, "draft_id", None)
    try:
        experience.record("decision.proposed", "agent", {"point": "lead.choice", "by": "agent:lead",
                                                         "choice": f"{obs.choice}:{obs.status}"[:120],
                                                         "why": f"Lead chose {obs.choice}; {obs.status}; tool {obs.tool or 'none'}"},
                          conversation=draft)
        if obs.status == "blocked":
            known = obs.blocked_reason if obs.blocked_reason and re.fullmatch(r"[a-z_]{1,60}", obs.blocked_reason) else "tool_refused"
            experience.record("decision.refused", "system", {"point": "lead.choice", "choice": obs.choice[:120], "rule": known,
                                                             "reason": str(obs.blocked_reason or "")[:200] or None}, conversation=draft)
    except Exception:  # noqa: BLE001 - the record failing must not break the turn
        pass
    # A4/A6: material decisions recorded idempotently into the bound Problem Base, as a fixed template (never the model's words).
    scope = getattr(state, "authorized_scope", None)
    home = getattr(state, "workspace_home", None)
    if (scope is not None and home is not None and getattr(scope, "project_id", None) and getattr(scope, "problem_id", None)
            and obs.choice in ("inspect", "compare", "revise", "consult") and obs.status in ("succeeded", "blocked")):
        try:
            from ..memory import bases as mem_bases

            status_map = {"succeeded": "succeeded", "blocked": "blocked", "stopped": "stopped", "failed": "failed"}
            summary_text = f"Lead {obs.choice}: {obs.status} (tool {obs.tool or 'none'})"
            mem_bases.record_material_event(
                Path(home), scope, target="problem",
                source_type="lead_decision", source_id=obs.decision_id,
                source_revision=str(getattr(state, "problem_revision", 0) or 0),
                text=summary_text, outcome_status=status_map.get(obs.status, "proposed"),
                who="agent:lead",
                depends_on_revision=int(_as_int(getattr(state, "problem_revision", 0)) or 0),
            )
        except Exception:  # noqa: BLE001 - memory write must not fail the turn
            pass
    if getattr(state, "checkpoint", None):
        try:
            case_session.save(state, state.checkpoint)
        except Exception:  # noqa: BLE001 - a checkpoint that cannot be written loses this step only
            pass


def dispatch_choice(state: Any, menu: Menu, selection: Mapping[str, Any], registry: Registry) -> Observation:
    """Validate a model's menu choice against current revision, permissions, approvals, budget, and idempotency;
    execute at most one committed effect via runtime.run and return a typed Observation."""
    decision_id = "dec_" + secrets.token_hex(5)
    available = [o.id for o in menu.options]
    if not isinstance(selection, Mapping) or not isinstance(selection.get("choice"), str):
        obs = Observation(decision_id=decision_id, choice="invalid", tool=None, status="failed",
                          reason="selection missing choice id", evidence_refs=[], available_options=available,
                          menu_revision=menu.revision, blocked_reason="invalid_selection")
        _persist_observation(state, obs)
        return obs

    choice = selection["choice"].strip()
    reason = str(selection.get("reason") or "").strip()[:300]
    refs = [str(r) for r in (selection.get("evidence_refs") or []) if r] or list(menu.evidence_refs[:4])
    args = dict(selection.get("arguments") or {}) if isinstance(selection.get("arguments"), Mapping) else {}
    supplied_rev = selection.get("menu_revision")

    def blocked(why: str, code: str, tool: Optional[str] = None, result: Optional[dict[str, Any]] = None) -> Observation:
        obs = Observation(decision_id=decision_id, choice=choice, tool=tool, status="blocked", reason=reason or why,
                          evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                          result=result or {}, blocked_reason=code)
        _persist_observation(state, obs)
        return obs

    if menu.stop_requested and choice not in ("stop", "ask"):
        return blocked("the person asked to stop: only stopping or asking is allowed", "stop_requested")
    # A6: revalidate live problem revision before executing any action so a pending action on superseded facts cannot run unchecked.
    live_summary = _sync_problem_context(state)
    live_prob_rev = _as_int(getattr(state, "problem_revision", None))
    raw_dep = selection.get("depends_on_problem_revision", args.get("depends_on_problem_revision"))
    dep_prob_rev = _as_int(raw_dep)
    if choice != "stop" and live_prob_rev is not None:
        menu_rev = _as_int(menu.problem_revision)
        stale_dep = (
            (raw_dep is not None and dep_prob_rev is None)  # a revision that is not a number cannot be checked: stale
            or (dep_prob_rev is not None and dep_prob_rev < live_prob_rev)
            or (menu_rev is not None and menu_rev < live_prob_rev)
        )
        if stale_dep:
            since = dep_prob_rev if dep_prob_rev is not None else (menu_rev or 0)
            changed = [
                {"field": f.get("field"), "text": f.get("text"), "revision": f.get("revision"), "correction": bool(f.get("correction"))}
                for f in ((live_summary or {}).get("active_facts") or [])
                if (_as_int(f.get("revision")) or 0) > since
            ]
            return blocked("problem facts changed since this action was proposed", "stale_problem_fact",
                           result={"stale": True, "old_problem_revision": dep_prob_rev if dep_prob_rev is not None else menu_rev,
                                   "current_problem_revision": live_prob_rev, "changed_sources": changed})
    current_rev = state_revision(state)
    if menu.revision != current_rev or (supplied_rev is not None and str(supplied_rev) != current_rev):
        return blocked("stale decision menu", "stale_menu_revision")

    option = menu.by_id(choice)
    if option is None:
        return blocked(f"choice {choice!r} is not in the allowed menu", "forbidden_choice")

    if option.requires_approval or (menu.approvals_pending and choice != "stop"):
        return blocked("waiting for explicit approval", "waiting_for_approval", tool=option.tool)

    # Bind arguments strictly to the authenticated case in host code.
    bound_args: dict[str, Any] = {}
    if choice == "ask":
        q = str(args.get("question") or "").strip()
        if not q:
            obs = Observation(decision_id=decision_id, choice=choice, tool=option.tool, status="failed",
                              reason=reason or "ask requires a non-empty question", evidence_refs=refs,
                              available_options=available, menu_revision=menu.revision, blocked_reason="invalid_arguments")
            _persist_observation(state, obs)
            return obs
        bound_args = {"question": q}
    elif choice == "inspect":
        if option.tool == "case_check_leakage":
            outcome = str(args.get("outcome") or option.defaults.get("outcome") or "").strip()
            bound_args = {"outcome": outcome}
        else:
            bound_args = {}
    elif choice == "consult":
        allowed_specs = list(option.defaults.get("allowed_specialists") or [])
        specialist = str(args.get("specialist") or option.defaults.get("specialist") or "leakage_auditor").strip()
        if allowed_specs and specialist not in allowed_specs:
            return blocked(f"specialist {specialist!r} is not allowed in the current state", "specialist_not_allowed", tool=option.tool)
        question = str(args.get("question") or reason or "Review available columns against the prediction moment.").strip()
        bound_args = {"specialist": specialist, "question": question}
    elif choice == "compare":
        outcome = str(args.get("outcome") or option.defaults.get("outcome") or "").strip()
        features_a = [str(c) for c in (args.get("features_a") or []) if c]
        features_b = [str(c) for c in (args.get("features_b") or []) if c]
        bound_args = {"outcome": outcome, "features_a": features_a, "features_b": features_b}
    elif choice == "revise":
        bound_args = {"exclude_columns": [str(c) for c in (args.get("exclude_columns") or []) if c],
                      "steps": [str(s) for s in (args.get("steps") or []) if s]}
    elif option.tool and option.tool.startswith("case_"):
        # Full existing Lead capabilities pass host-validated arguments directly through runtime.run + CaseGuard.
        bound_args = dict(args)

    case_id = str(getattr(state.case, "case_id", getattr(state.case, "id", "")))
    digest = _args_digest(choice, bound_args)
    idem_key = _idempotency_key(case_id, _fact_fingerprint(state), choice, bound_args)
    model_key = str(selection.get("idempotency_key") or "").strip()[:120] or None
    effects = getattr(state, "decision_effects", None) or {}
    existing = None
    if choice in REPLAYABLE:
        for key in (idem_key, model_key):
            found = effects.get(key) if key else None
            if found is not None and found.get("choice") == choice and found.get("arguments_digest") == digest:
                existing = found  # a replay only for the same choice with the same arguments (a key alone never replays another)
                break
    if existing is not None:
        return Observation(
            decision_id=existing["decision_id"], choice=existing["choice"], tool=existing.get("tool"),
            status=existing["status"], reason=existing.get("reason", ""), evidence_refs=list(existing.get("evidence_refs") or []),
            available_options=available, menu_revision=menu.revision, result=dict(existing.get("result") or {}),
            blocked_reason=existing.get("blocked_reason"), committed_effects=0, idempotency_key=idem_key, replayed=True,
            arguments_digest=digest,
        )

    def record(obs: Observation) -> Observation:
        obs.arguments_digest = digest
        _persist_observation(state, obs)
        if model_key and obs.committed_effects > 0 and choice in REPLAYABLE:
            state.decision_effects[model_key] = obs.to_record()
        return obs

    if choice == "stop":
        msg = str(args.get("message") or "").strip()
        return record(Observation(decision_id=decision_id, choice="stop", tool=None, status="stopped",
                                  reason=reason or "stopped cleanly", evidence_refs=refs, available_options=available,
                                  menu_revision=menu.revision, result={"stopped": True, "message": msg},
                                  committed_effects=0, idempotency_key=idem_key))

    # Dispatch every tool through runtime.run so a registry, its scope's guard and the tracer own every call: the case tools
    # through the shared registry, the loop's compare and revise through its own.
    own = choice in ("compare", "revise")
    run_registry = _own_registry() if own else registry
    if option.tool is None or not _has_tool(run_registry, option.tool):
        return blocked("tool not registered", "tool_unavailable", tool=option.tool)

    single = Policy(name="case", system="", tools=(option.tool,), max_steps=1, max_replies=1, stop_after=(option.tool,))
    run_res = run_policy(single, _BoundOnceClient(option.tool, bound_args), run_registry,
                         [{"role": "user", "content": f"[DCLab bounded choice] {choice}"}], context=state, trace=state.trace)
    if not run_res.steps:
        return record(Observation(decision_id=decision_id, choice=choice, tool=option.tool, status="failed",
                                  reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                                  blocked_reason="no_step_executed", committed_effects=0, idempotency_key=idem_key))

    step = run_res.steps[-1]
    res = step.result if isinstance(step.result, dict) else {"value": step.result}
    if choice == "consult" and isinstance(res.get("error"), str) and "gave no finding" in res["error"]:
        advisory = {
            "specialist": bound_args["specialist"],
            "provisional": True,
            "carries_approval": False,
            "insufficient_evidence": True,
            "findings": [],
            "problems": list(res.get("problems") or []),
            "evidence_refs": refs,
        }
        return record(Observation(decision_id=decision_id, choice="consult", tool=option.tool, status="succeeded",
                                  reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                                  result=advisory, committed_effects=1, idempotency_key=idem_key))
    if not step.ok or "error" in res:
        error = str(res.get("error") or "tool_error")
        if choice == "compare":
            code = "experiment_cap" if "cap reached" in error else "compare_refused" if res.get("refused") else "compare_error"
            status = "blocked" if code in ("experiment_cap", "compare_refused") else "failed"
        elif choice == "revise":
            code, status = ("revision_limit", "blocked") if "limit" in error else ("revision_invalid", "failed")
        else:
            status = "blocked" if any(w in error.lower() for w in ("cannot", "forbidden", "refused", "not allowed", "budget", "already")) else "failed"
            code = error[:200]
        return record(Observation(decision_id=decision_id, choice=choice, tool=option.tool, status=status,
                                  reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                                  result=res, blocked_reason=code, committed_effects=0, idempotency_key=idem_key))

    if choice == "ask":
        obs = Observation(decision_id=decision_id, choice="ask", tool=option.tool, status="waiting_for_person",
                          reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                          result={"question": bound_args["question"], **res}, committed_effects=1, idempotency_key=idem_key)
    elif choice == "propose":
        obs = Observation(decision_id=decision_id, choice="propose", tool=option.tool, status="waiting_for_person",
                          reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                          result=res, committed_effects=1, idempotency_key=idem_key)
    elif choice == "revise":
        # A proposal waits for the person's yes; the question is code's, the plan's lines are the agent's (checked when shown).
        state.said.append("I would change the working plan to this:\n" + str(res.get("plan") or "") + "\nShall I use it?")
        state.asking = True
        obs = Observation(decision_id=decision_id, choice="revise", tool=option.tool, status="waiting_for_person",
                          reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                          result=res, committed_effects=1, idempotency_key=idem_key)
    elif choice == "consult":
        findings = list(res.get("findings") or [])
        advisory = {
            "specialist": bound_args["specialist"],
            "provisional": True,
            "carries_approval": False,
            "insufficient_evidence": not findings,
            "findings": findings,
            "evidence_refs": refs,
        }
        obs = Observation(decision_id=decision_id, choice="consult", tool=option.tool, status="succeeded",
                          reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                          result=advisory, committed_effects=1, idempotency_key=idem_key)
    else:
        obs = Observation(decision_id=decision_id, choice=choice, tool=option.tool, status="succeeded",
                          reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                          result=res, committed_effects=1, idempotency_key=idem_key)
    return record(obs)


def _call_parts(call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    fn = call.get("function") if isinstance(call.get("function"), Mapping) else {}
    name = str(call.get("name") or fn.get("name") or "")
    args = call.get("arguments", fn.get("arguments")) or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    return name, (args if isinstance(args, dict) else {})


def _selection_of_call(call: Mapping[str, Any], menu: Menu, content: str = "") -> dict[str, Any]:
    """One tool call of a reply as a menu selection."""
    name, args = _call_parts(call)
    if name in ("choose_action", "decide"):
        return dict(args)
    by_tool = {o.tool: o.id for o in menu.options if o.tool}
    if name in by_tool:
        return {"choice": by_tool[name], "arguments": args, "reason": (content or f"selected {by_tool[name]}")[:300],
                "menu_revision": menu.revision}
    if name in ("stop", "case_stop"):
        return {"choice": "stop", "arguments": args, "reason": (content or str(args.get("reason") or "stop"))[:300],
                "menu_revision": menu.revision}
    return {"choice": name, "arguments": args, "reason": content[:300], "menu_revision": menu.revision}


def _text_selection(text: str) -> Optional[dict[str, Any]]:
    """A choice written as JSON in the reply's text."""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict) and isinstance(obj.get("choice"), str):
                return obj
        except ValueError:
            pass
    return None


def _parse_model_selection(reply: Mapping[str, Any], menu: Menu) -> Optional[dict[str, Any]]:
    """The reply's first choice: a tool call, or a JSON choice in its text (None when it holds neither)."""
    tool_calls = [c for c in (reply.get("tool_calls") or []) if isinstance(c, Mapping)]
    content = str(reply.get("content") or "").strip()
    if tool_calls:
        return _selection_of_call(tool_calls[0], menu, content)
    return _text_selection(content) if content else None


def _choice_tools(menu: Menu, registry: Optional[Registry] = None) -> list[dict[str, Any]]:
    ids = [o.id for o in menu.options]
    tools: list[dict[str, Any]] = [{
        "type": "function",
        "function": {
            "name": "choose_action",
            "description": "Select one permitted next action from the current menu with a brief evidence-tied reason.",
            "parameters": {
                "type": "object",
                "required": ["choice", "reason"],
                "properties": {
                    "choice": {"type": "string", "enum": ids},
                    "reason": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "arguments": {"type": "object"},
                    "menu_revision": {"type": "string"},
                },
            },
        },
    }]
    if registry is not None:
        seen = {"choose_action"}
        for opt in menu.options:
            if opt.tool and opt.tool not in seen and _has_tool(registry, opt.tool):
                t = registry.get(opt.tool)
                if t is not None:
                    seen.add(opt.tool)
                    tools.append({
                        "type": "function",
                        "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                    })
    return tools


def _spent(client: Any) -> float:
    try:
        return float(getattr(client, "spent_eur", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


TURN_FAILED = ("Something went wrong on my side before I could finish this step, so I stopped here. Tell me how you would like "
               "to go on.")
OPTIONS_SHOWN = "Here are the options I see. Pick one, or tell me what to change."


def _summarize_observations(observations: list[Observation], fallback_text: str = "") -> str:
    """Code's words for a turn in which the agent wrote none: what ran and what came back (checked like any message)."""
    if not observations:
        return fallback_text or "No action was taken."
    last = observations[-1]
    if last.status == "stopped":
        return str(last.result.get("message") or "") or fallback_text or "I stopped here without running anything else."
    if last.status == "blocked":
        return f"I could not run that step ({last.blocked_reason or 'not permitted'}), so nothing was changed."
    if last.status == "failed":
        return fallback_text or f"Could not complete the requested step ({last.blocked_reason or 'invalid choice'})."
    if last.choice == "propose" and last.status == "waiting_for_person":
        return OPTIONS_SHOWN
    parts = []
    for obs in observations:
        if obs.choice == "inspect":
            suspects = obs.result.get("suspects") or []
            if suspects:
                cols = ", ".join(str(s.get("column") or s.get("statement") or "") for s in suspects[:6])
                parts.append(f"The leakage check flagged columns that may be written after the outcome: {cols}.")
            else:
                parts.append("I checked the table's columns for leakage.")
        elif obs.choice == "consult":
            if obs.result.get("insufficient_evidence"):
                parts.append("The specialist found too little evidence to say more (provisional).")
            else:
                parts.append(f"The specialist ({obs.result.get('specialist')}) came back with provisional findings.")
        elif obs.choice == "compare" and "set_a" in obs.result:
            parts.append(f"On the training rows ({obs.result.get('metric')}): set A {obs.result['set_a']['score']}, "
                         f"set B {obs.result['set_b']['score']}.")
        elif obs.choice == "revise" and obs.result.get("applied"):
            parts.append("I am using the revised plan you approved.")
        elif obs.choice == "revise" and obs.result.get("declined"):
            parts.append("I kept the plan as it was.")
    return " ".join(parts) or fallback_text or "I finished this step."


def run_bounded_turn(state: Any, client: Any, registry: Registry, context_messages: list[dict[str, Any]], *,
                     max_decisions: int = MAX_DECISIONS) -> dict[str, Any]:
    """Run one bounded Lead turn: offer permitted menu, validate & dispatch choice, persist observation, repeat until
    waiting_for_person, stopped, blocked, budget limit, unchanged repeat, or max_decisions. Never raises: a failure inside
    the turn is a failed observation and a safe reply (its caller catches nothing but RuntimeError)."""
    try:
        return _bounded_turn(state, client, registry, context_messages, max_decisions=max_decisions)
    except Exception as exc:  # noqa: BLE001 - the person gets a safe reply, the failure is recorded
        obs = Observation(decision_id="dec_" + secrets.token_hex(5), choice="stop", tool=None, status="failed",
                          reason=f"turn error: {type(exc).__name__}", evidence_refs=[], available_options=[], menu_revision="",
                          blocked_reason="turn_error")
        try:
            _persist_observation(state, obs)
        except Exception:  # noqa: BLE001
            pass
        try:
            state.turns.append({"role": "assistant", "content": TURN_FAILED})
        except Exception:  # noqa: BLE001
            pass
        return {"text": TURN_FAILED, "options": [], "route": getattr(getattr(state, "case", None), "route", None), "asking": False,
                "cards": [], "observations": [obs.to_record()], "decisions": list(getattr(state, "decisions", None) or [])}


def _bounded_turn(state: Any, client: Any, registry: Registry, context_messages: list[dict[str, Any]], *,
                  max_decisions: int) -> dict[str, Any]:
    from . import case_agent

    observations: list[Observation] = []
    ran_actions: set[tuple[str, str, str]] = set()
    messages = list(context_messages)
    fallback_text = ""
    agent_words: list[dict[str, Any]] = []  # the agent's own words this turn, as replies (spoken() drops working notes)
    queue: list[Mapping[str, Any]] = []  # a reply's later tool calls, each run on the menu as it is when its turn comes
    words = next((str(t.get("content") or "") for t in reversed(getattr(state, "turns", None) or []) if t.get("role") == "user"), "")
    max_tokens = case_agent.THINK_FIRST_TOKENS if case_agent.think_first() else case_agent.POLICY.max_tokens

    answered = _resolve_pending_revision(state, words)  # a proposed revision answered by the person's latest words
    if answered is not None:
        observations.append(answered)

    def ended(obs: Observation) -> None:
        _persist_observation(state, obs)
        observations.append(obs)

    for _ in range(max_decisions):
        menu = build_menu(state, registry)
        if queue:
            selection = _selection_of_call(queue.pop(0), menu)
            selection["menu_revision"] = menu.revision  # checked again on this menu: the choice, the facts, the approvals
        else:
            if client is None:
                ended(Observation(decision_id="dec_" + secrets.token_hex(5), choice="stop", tool=None, status="failed",
                                  reason="model unavailable", evidence_refs=list(menu.evidence_refs),
                                  available_options=[o.id for o in menu.options], menu_revision=menu.revision,
                                  blocked_reason="model_unavailable"))
                fallback_text = "The model is unavailable; no action was executed."
                break
            before_spend = _spent(client) + _spent(getattr(state, "specialist_client", None))
            menu_msg = {"role": "user", "content": "[DCLab decision menu]\n" + json.dumps(menu.as_prompt_dict(), ensure_ascii=False)}
            try:
                reply = client.complete([*messages, menu_msg], _choice_tools(menu, registry), max_tokens=max_tokens)
            except Exception as exc:  # noqa: BLE001 - explicit typed failure on transport/budget exception
                ended(Observation(decision_id="dec_" + secrets.token_hex(5), choice="stop", tool=None, status="failed",
                                  reason=f"model error: {type(exc).__name__}", evidence_refs=list(menu.evidence_refs),
                                  available_options=[o.id for o in menu.options], menu_revision=menu.revision,
                                  blocked_reason="model_unavailable"))
                fallback_text = f"Paused safely because the model could not answer ({type(exc).__name__})."
                break
            state.spent_eur += max(0.0, _spent(client) + _spent(getattr(state, "specialist_client", None)) - before_spend)
            reply = reply if isinstance(reply, Mapping) else {}
            content = str(reply.get("content") or "").strip()
            calls = [c for c in (reply.get("tool_calls") or []) if isinstance(c, Mapping)]
            if calls:
                selection = _selection_of_call(calls[0], menu, content)
                queue = list(calls[1:])
                if content:  # the agent's words beside its calls are for the person, unless they are only a working note
                    names = []
                    for c in calls:
                        sel = _selection_of_call(c, menu)
                        opt = menu.by_id(str(sel.get("choice") or ""))
                        names.append({"name": (opt.tool if opt and opt.tool else _call_parts(c)[0])})
                    agent_words.append({"role": "assistant", "content": content, "tool_calls": names})
            elif content and _text_selection(content) is not None:
                selection = _text_selection(content)
            elif content:  # an answer in words, the native way to reply: said (after the checks) and the turn stops
                agent_words.append({"role": "assistant", "content": content})
                ended(Observation(decision_id="dec_" + secrets.token_hex(5), choice="say", tool=None, status="stopped",
                                  reason="the agent answered in words", evidence_refs=list(menu.evidence_refs),
                                  available_options=[o.id for o in menu.options], menu_revision=menu.revision,
                                  result={"said": True}, committed_effects=0))
                break
            else:
                ended(Observation(decision_id="dec_" + secrets.token_hex(5), choice="invalid", tool=None, status="failed",
                                  reason="model reply was not a valid menu choice", evidence_refs=list(menu.evidence_refs),
                                  available_options=[o.id for o in menu.options], menu_revision=menu.revision,
                                  blocked_reason="invalid_selection"))
                fallback_text = "Could not validate the proposed next action; please clarify what you want to check or run next."
                break

        sig_args = json.dumps(selection.get("arguments") or {}, sort_keys=True, default=str)
        repeat_sig = (str(selection.get("choice") or ""), sig_args, _fact_fingerprint(state))
        if repeat_sig in ran_actions:
            ended(Observation(decision_id="dec_" + secrets.token_hex(5), choice="stop", tool=None, status="stopped",
                              reason="stopped repeated identical choice with unchanged evidence",
                              evidence_refs=list(menu.evidence_refs), available_options=[o.id for o in menu.options],
                              menu_revision=menu.revision, result={"stopped": True, "message": ""},
                              blocked_reason="repeated_unchanged_choice"))
            break
        ran_actions.add(repeat_sig)

        before_dispatch_spend = _spent(getattr(state, "specialist_client", None))
        obs = dispatch_choice(state, menu, selection, registry)
        state.spent_eur += max(0.0, _spent(getattr(state, "specialist_client", None)) - before_dispatch_spend)
        observations.append(obs)
        if obs.choice == "stop" and str(obs.result.get("message") or "").strip():
            agent_words.append({"role": "assistant", "content": str(obs.result["message"]).strip()})
        messages.append({"role": "assistant", "content": json.dumps({"choice": obs.choice, "reason": obs.reason})})
        messages.append({"role": "user", "content": "[DCLab observation]\n" + json.dumps(obs.to_record(), default=str)})

        if obs.status in ("waiting_for_person", "stopped", "blocked", "failed"):
            break
        if state.spent_eur >= state.cap_eur:
            state.waiting_on_budget = True
            ended(Observation(decision_id="dec_" + secrets.token_hex(5), choice="stop", tool=None, status="stopped",
                              reason="run budget cap reached", evidence_refs=list(menu.evidence_refs),
                              available_options=[o.id for o in menu.options], menu_revision=state_revision(state),
                              result={"stopped": True, "message": "Stopped at the run budget cap."},
                              blocked_reason="budget_exhausted"))
            break

    texts = case_agent.spoken(agent_words) if agent_words else []
    if not texts and not state.said:
        texts = [_summarize_observations(observations, fallback_text)]
    plan_before = state.plan
    text = case_agent.checked_message(state, texts, words)  # every word shown passes the native turn's checks
    if _pending_revision(state) is not None:
        state.plan = plan_before  # a plan shown for the person's yes is not the working plan until they give it
    state.turns.append({"role": "assistant", "content": text + ("\n(Options: " + "; ".join(o.title for o in state.shown) + ")" if state.shown else "")})
    if getattr(state, "checkpoint", None):
        from . import case_session

        try:
            case_session.save(state, state.checkpoint)
        except Exception:  # noqa: BLE001
            pass
    return {
        "text": text,
        "options": list(state.shown),
        "route": state.case.route,
        "asking": state.asking,
        "cards": list(state.cards),
        "observations": [o.to_record() for o in observations],
        "decisions": list(state.decisions),
    }


__all__ = [
    "ADVISORY_ENV", "MAX_DECISIONS", "MAX_EXPERIMENTS", "MAX_REVISIONS", "Menu", "MenuOption", "OBSERVATION_STATUSES",
    "Observation", "SETTINGS_FILE", "SWITCH_ENV", "advisory_roles_enabled", "build_menu", "dispatch_choice", "is_enabled",
    "run_bounded_turn", "set_settings", "settings", "state_revision",
]
