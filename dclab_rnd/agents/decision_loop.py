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
MAX_DECISIONS = 4
MAX_REVISIONS = 2
MAX_EXPERIMENTS = 2
OBSERVATION_STATUSES = ("succeeded", "blocked", "waiting_for_person", "stopped", "failed")
STOP_WORDS = re.compile(r"\b(?:stop|pause|hold on|do not run|don't run|cancel|wait here)\b", re.I)


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
        "unknowns": [u.field for u in (getattr(case, "unknowns", None) or []) if not u.resolved],
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
        "unknowns": [u.field for u in (getattr(case, "unknowns", None) or []) if not u.resolved],
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

    if state.table is not None and not approvals_pending:
        outcome = _outcome_candidate(state)
        if outcome and _has_tool(registry, "case_check_leakage"):
            options.append(MenuOption(id="inspect", tool="case_check_leakage",
                                      summary="Run read-only leakage and column availability inspection on the attached table.",
                                      defaults={"outcome": outcome}))
        elif _has_tool(registry, "case_read_table"):
            options.append(MenuOption(id="inspect", tool="case_read_table",
                                      summary="Inspect table columns and metadata (read-only, no cell values)."))

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
            options.append(MenuOption(id="compare", tool="training_compare",
                                      summary="Run a budgeted training-fold comparison between two allowed feature sets (never reads holdout).",
                                      defaults={"outcome": outcome}))

        has_evidence = bool(getattr(state, "decisions", None) or getattr(state.case, "findings", None) or getattr(state, "computed", None))
        if has_evidence and int(getattr(state, "decision_revisions", 0)) < max_revisions:
            options.append(MenuOption(id="revise", tool="plan_revise",
                                      summary="Revise the working investigation plan in response to new evidence (provisional, never auto-approves)."))

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


def _idempotency_key(case_id: str, revision: str, choice: str, arguments: Mapping[str, Any]) -> str:
    body = json.dumps({"case_id": case_id, "revision": revision, "choice": choice, "arguments": arguments}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:20]


def _training_compare(state: Any, outcome: str, features_a: list[str], features_b: list[str]) -> dict[str, Any]:
    """Host-owned training-only validation comparison: fold-local preprocessing, never touches any holdout split."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, StratifiedKFold

    frame = getattr(state, "frame", None)
    if frame is None or not isinstance(frame, pd.DataFrame) or outcome not in frame.columns:
        return {"error": f"outcome {outcome!r} is not available in the attached table"}
    clean = frame.dropna(subset=[outcome]).copy()
    if len(clean) < 20:
        return {"error": "fewer than 20 labelled rows available for training-fold comparison"}
    # Reserve the final 20% strictly untouched (holdout boundary) so comparison runs on training rows only.
    train_cut = max(16, int(len(clean) * 0.8))
    train = clean.iloc[:train_cut].reset_index(drop=True)
    y_raw = train[outcome]
    is_numeric = pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique(dropna=True) > 10
    task = "regression" if is_numeric else "classification"
    allowed_cols = [str(c) for c in train.columns if str(c) != outcome and str(c) not in (getattr(state, "personal", set()) or set())]
    cols_a = [c for c in features_a if c in allowed_cols]
    cols_b = [c for c in features_b if c in allowed_cols]
    if not cols_a or not cols_b:
        return {"error": "both feature sets must name at least one allowed non-target column"}

    def _encode_fold(fit_df: pd.DataFrame, val_df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        fit_parts, val_parts = {}, {}
        for col in cols:
            if pd.api.types.is_datetime64_any_dtype(fit_df[col]):
                fit_parts[col] = (fit_df[col] - pd.Timestamp("1970-01-01")).dt.total_seconds().astype(float)
                val_parts[col] = (val_df[col] - pd.Timestamp("1970-01-01")).dt.total_seconds().astype(float)
            elif pd.api.types.is_numeric_dtype(fit_df[col]) or pd.api.types.is_bool_dtype(fit_df[col]):
                med = float(pd.to_numeric(fit_df[col], errors="coerce").median()) if fit_df[col].notna().any() else 0.0
                fit_parts[col] = pd.to_numeric(fit_df[col], errors="coerce").fillna(med).astype(float)
                val_parts[col] = pd.to_numeric(val_df[col], errors="coerce").fillna(med).astype(float)
            else:
                # Fold-local frequency encoding learned strictly on fit_df only.
                freq = fit_df[col].astype(str).value_counts(normalize=True).to_dict()
                fit_parts[col] = fit_df[col].astype(str).map(freq).fillna(0.0).astype(float)
                val_parts[col] = val_df[col].astype(str).map(freq).fillna(0.0).astype(float)
        return pd.DataFrame(fit_parts), pd.DataFrame(val_parts)

    def _cv_score(cols: list[str]) -> float:
        if task == "regression":
            y = y_raw.astype(float).to_numpy()
            splitter = KFold(n_splits=3, shuffle=False)
            scores = []
            for fi, vi in splitter.split(train):
                xf, xv = _encode_fold(train.iloc[fi], train.iloc[vi], cols)
                m = HistGradientBoostingRegressor(max_iter=20, random_state=0).fit(xf, y[fi])
                scores.append(float(np.mean(np.abs(y[vi] - m.predict(xv)))))
            return round(float(np.mean(scores)), 4)
        y = y_raw.astype(str).to_numpy()
        n_splits = min(3, max(2, int(pd.Series(y).value_counts().min())))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        scores = []
        for fi, vi in splitter.split(train, y):
            xf, xv = _encode_fold(train.iloc[fi], train.iloc[vi], cols)
            m = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(xf, y[fi])
            scores.append(float(np.mean(m.predict(xv) == y[vi])))
        return round(float(np.mean(scores)), 4)

    metric = "mae" if task == "regression" else "accuracy"
    score_a, score_b = _cv_score(cols_a), _cv_score(cols_b)
    return {
        "scope": "training_folds_only",
        "holdout_touched": False,
        "train_rows": int(len(train)),
        "task": task,
        "metric": metric,
        "set_a": {"features": cols_a, "score": score_a},
        "set_b": {"features": cols_b, "score": score_b},
        "delta_b_minus_a": round(score_b - score_a, 4),
    }


def _revise_plan(state: Any, arguments: Mapping[str, Any], reason: str) -> dict[str, Any]:
    """Revise the provisional working plan using observed evidence; never auto-approves solution mutations."""
    from ..tasks import catalog as task_catalog

    excluded = [str(c) for c in (arguments.get("exclude_columns") or []) if c]
    next_steps = [str(s) for s in (arguments.get("steps") or []) if s]
    current_plan = {"revisions": int(getattr(state, "decision_revisions", 0)), "steps": [s for s in str(getattr(state, "plan", "") or "").splitlines() if s.strip()]}
    if current_plan["revisions"] >= MAX_REVISIONS:
        return {"error": "revision limit reached"}
    # Reuse tasks.catalog.replan_from_evidence when a structured catalog plan is active; otherwise update provisional plan directly.
    revised_text_lines = []
    if next_steps:
        revised_text_lines.extend(f"{i + 1}. {step}" for i, step in enumerate(next_steps))
    else:
        revised_text_lines.append(f"1. Revised plan: {reason}")
    if excluded:
        revised_text_lines.append("Provisional exclusions (pending approval): " + ", ".join(excluded))
    catalog_plan = getattr(state, "catalog_plan", None)
    catalog_out = None
    if isinstance(catalog_plan, Mapping) and getattr(state, "table", None):
        cols = [str(c.get("name")) for c in (state.table.get("columns") or []) if c.get("name")]
        catalog_out = task_catalog.replan_from_evidence(
            brief=str(getattr(getattr(state.case, "goal", None), "value", "") or reason),
            tables={str(state.table.get("table") or "data"): cols},
            known_facts={"excluded_suspects": excluded},
            current_plan=catalog_plan,
            stage_evidence=[d.get("result") or {} for d in (getattr(state, "decisions", None) or [])[-3:]],
            project_id=str(getattr(state, "draft_id", None) or getattr(state.case, "case_id", "")),
            gateway=getattr(state, "models", None),
            max_revisions=MAX_REVISIONS,
        )
    state.plan = "\n".join(revised_text_lines)
    state.decision_revisions = int(getattr(state, "decision_revisions", 0)) + 1
    return {
        "revised": True,
        "revision": state.decision_revisions,
        "provisional_exclusions": excluded,
        "plan": state.plan,
        "requires_approval": True,
        "catalog_replan": bool(catalog_out and catalog_out.get("revised")),
    }


def _persist_observation(state: Any, obs: Observation) -> None:
    """Persist decision + observation into state, checkpoint, experience log, and tracer before the next choice."""
    from .. import experience
    from . import case_session

    rec = obs.to_record()
    state.decisions.append(rec)
    if obs.idempotency_key and obs.committed_effects > 0:
        state.decision_effects[obs.idempotency_key] = rec
    try:
        experience.record("agent.decision", "agent", rec, draft=getattr(state, "draft_id", None))
    except Exception:  # noqa: BLE001 - tracing failure must not break the turn
        pass
    # A4/A6: record material decisions/outcomes idempotently into the bound Problem Base.
    scope = getattr(state, "authorized_scope", None)
    home = getattr(state, "workspace_home", None)
    if (scope is not None and home is not None and getattr(scope, "project_id", None) and getattr(scope, "problem_id", None)
            and obs.choice in ("inspect", "compare", "revise", "consult") and obs.status in ("succeeded", "blocked")):
        try:
            from ..memory import bases as mem_bases

            status_map = {"succeeded": "succeeded", "blocked": "blocked", "stopped": "stopped", "failed": "failed"}
            summary_text = f"Lead {obs.choice} ({obs.status}): {obs.reason or obs.blocked_reason or 'completed'}"[:280]
            mem_bases.record_material_event(
                Path(home), scope, target="problem",
                source_type="lead_decision", source_id=obs.decision_id,
                source_revision=str(getattr(state, "problem_revision", 0) or 0),
                text=summary_text, outcome_status=status_map.get(obs.status, "proposed"),
                who="agent:lead",
                depends_on_revision=int(getattr(state, "problem_revision", 0) or 0),
            )
        except Exception:  # noqa: BLE001 - memory write must not fail the turn
            pass
    if getattr(state, "checkpoint", None):
        case_session.save(state, state.checkpoint)


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
    # A6: revalidate live problem revision before executing any action so a pending action on superseded facts cannot run unchecked.
    live_summary = _sync_problem_context(state)
    live_prob_rev = getattr(state, "problem_revision", None)
    dep_prob_rev = selection.get("depends_on_problem_revision", args.get("depends_on_problem_revision"))
    if choice != "stop" and isinstance(live_prob_rev, int):
        stale_dep = (
            (dep_prob_rev is not None and int(dep_prob_rev) < live_prob_rev)
            or (menu.problem_revision is not None and int(menu.problem_revision) < live_prob_rev)
        )
        if stale_dep:
            changed = [
                {"field": f.get("field"), "text": f.get("text"), "revision": f.get("revision"), "correction": bool(f.get("correction"))}
                for f in ((live_summary or {}).get("active_facts") or [])
                if int(f.get("revision") or 0) > int(dep_prob_rev if dep_prob_rev is not None else menu.problem_revision or 0)
            ]
            obs = Observation(
                decision_id=decision_id, choice=choice, tool=None, status="blocked",
                reason=reason or "problem facts changed since this action was proposed",
                evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                result={"stale": True, "old_problem_revision": dep_prob_rev if dep_prob_rev is not None else menu.problem_revision,
                        "current_problem_revision": live_prob_rev, "changed_sources": changed},
                blocked_reason="stale_problem_fact",
            )
            _persist_observation(state, obs)
            return obs
    current_rev = state_revision(state)
    if menu.revision != current_rev or (supplied_rev is not None and str(supplied_rev) != current_rev):
        obs = Observation(decision_id=decision_id, choice=choice, tool=None, status="blocked",
                          reason=reason or "stale decision menu", evidence_refs=refs, available_options=available,
                          menu_revision=menu.revision, blocked_reason="stale_menu_revision")
        _persist_observation(state, obs)
        return obs

    option = menu.by_id(choice)
    if option is None:
        obs = Observation(decision_id=decision_id, choice=choice, tool=None, status="blocked",
                          reason=reason or f"choice {choice!r} is not in the allowed menu", evidence_refs=refs,
                          available_options=available, menu_revision=menu.revision, blocked_reason="forbidden_choice")
        _persist_observation(state, obs)
        return obs

    if option.requires_approval or (menu.approvals_pending and choice != "stop"):
        obs = Observation(decision_id=decision_id, choice=choice, tool=option.tool, status="blocked",
                          reason=reason or "waiting for explicit approval", evidence_refs=refs,
                          available_options=available, menu_revision=menu.revision, blocked_reason="waiting_for_approval")
        _persist_observation(state, obs)
        return obs

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
            obs = Observation(decision_id=decision_id, choice=choice, tool=option.tool, status="blocked",
                              reason=reason or f"specialist {specialist!r} is not allowed in the current state",
                              evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                              blocked_reason="specialist_not_allowed")
            _persist_observation(state, obs)
            return obs
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

    case_id = str(getattr(state.case, "case_id", getattr(state.case, "id", "")))
    idem_key = str(selection.get("idempotency_key") or _idempotency_key(case_id, menu.revision, choice, bound_args))
    existing = (getattr(state, "decision_effects", None) or {}).get(idem_key)
    if existing is not None:
        return Observation(
            decision_id=existing["decision_id"], choice=existing["choice"], tool=existing.get("tool"),
            status=existing["status"], reason=existing.get("reason", ""), evidence_refs=list(existing.get("evidence_refs") or []),
            available_options=available, menu_revision=menu.revision, result=dict(existing.get("result") or {}),
            blocked_reason=existing.get("blocked_reason"), committed_effects=0, idempotency_key=idem_key, replayed=True,
        )

    if choice == "stop":
        msg = str(args.get("message") or reason or "Stopped without running any tool.").strip()
        obs = Observation(decision_id=decision_id, choice="stop", tool=None, status="stopped",
                          reason=reason or "stopped cleanly", evidence_refs=refs, available_options=available,
                          menu_revision=menu.revision, result={"stopped": True, "message": msg},
                          committed_effects=0, idempotency_key=idem_key)
        _persist_observation(state, obs)
        return obs

    if choice == "compare":
        if int(getattr(state, "decision_experiments", 0)) >= MAX_EXPERIMENTS:
            obs = Observation(decision_id=decision_id, choice=choice, tool=option.tool, status="blocked",
                              reason=reason or "experiment cap reached", evidence_refs=refs, available_options=available,
                              menu_revision=menu.revision, blocked_reason="experiment_cap")
            _persist_observation(state, obs)
            return obs
        res = _training_compare(state, bound_args["outcome"], bound_args["features_a"], bound_args["features_b"])
        if "error" in res:
            obs = Observation(decision_id=decision_id, choice=choice, tool=option.tool, status="failed",
                              reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                              result=res, blocked_reason="compare_error", committed_effects=0, idempotency_key=idem_key)
        else:
            state.decision_experiments = int(getattr(state, "decision_experiments", 0)) + 1
            state.computed.append(json.dumps(res, default=str))
            obs = Observation(decision_id=decision_id, choice=choice, tool=option.tool, status="succeeded",
                              reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                              result=res, committed_effects=1, idempotency_key=idem_key)
        _persist_observation(state, obs)
        return obs

    if choice == "revise":
        res = _revise_plan(state, bound_args, reason)
        if "error" in res:
            obs = Observation(decision_id=decision_id, choice=choice, tool=option.tool, status="blocked",
                              reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                              result=res, blocked_reason="revision_limit", committed_effects=0, idempotency_key=idem_key)
        else:
            obs = Observation(decision_id=decision_id, choice=choice, tool=option.tool, status="succeeded",
                              reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                              result=res, committed_effects=1, idempotency_key=idem_key)
        _persist_observation(state, obs)
        return obs

    # Dispatch registered case tools exclusively through runtime.run so Registry, CaseGuard, and Tracer own every call.
    if option.tool is None or not _has_tool(registry, option.tool):
        obs = Observation(decision_id=decision_id, choice=choice, tool=option.tool, status="blocked",
                          reason="tool not registered", evidence_refs=refs, available_options=available,
                          menu_revision=menu.revision, blocked_reason="tool_unavailable")
        _persist_observation(state, obs)
        return obs

    single = Policy(name="case", system="", tools=(option.tool,), max_steps=1, max_replies=1, stop_after=(option.tool,))
    run_res = run_policy(single, _BoundOnceClient(option.tool, bound_args), registry, [{"role": "user", "content": reason or choice}],
                         context=state, trace=state.trace)
    if not run_res.steps:
        obs = Observation(decision_id=decision_id, choice=choice, tool=option.tool, status="failed",
                          reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                          blocked_reason="no_step_executed", committed_effects=0, idempotency_key=idem_key)
        _persist_observation(state, obs)
        return obs

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
        obs = Observation(decision_id=decision_id, choice="consult", tool=option.tool, status="succeeded",
                          reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                          result=advisory, committed_effects=1, idempotency_key=idem_key)
        _persist_observation(state, obs)
        return obs
    if not step.ok or "error" in res:
        status = "blocked" if any(w in str(res.get("error") or "").lower() for w in ("cannot", "forbidden", "refused", "not allowed", "budget", "already")) else "failed"
        obs = Observation(decision_id=decision_id, choice=choice, tool=option.tool, status=status,
                          reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                          result=res, blocked_reason=str(res.get("error") or "tool_error")[:200], committed_effects=0,
                          idempotency_key=idem_key)
        _persist_observation(state, obs)
        return obs

    if choice == "ask":
        obs = Observation(decision_id=decision_id, choice="ask", tool=option.tool, status="waiting_for_person",
                          reason=reason, evidence_refs=refs, available_options=available, menu_revision=menu.revision,
                          result={"question": bound_args["question"], **res}, committed_effects=1, idempotency_key=idem_key)
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
    _persist_observation(state, obs)
    return obs


def _parse_model_selection(reply: Mapping[str, Any], menu: Menu) -> Optional[dict[str, Any]]:
    """Accept either a structured JSON choice or a standard tool call from a scripted or live model."""
    tool_calls = reply.get("tool_calls") or []
    if tool_calls:
        first = tool_calls[0]
        name = str(first.get("name") or "")
        args = first.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        if name in ("choose_action", "decide"):
            return args
        by_tool = {o.tool: o.id for o in menu.options if o.tool}
        if name in by_tool:
            return {"choice": by_tool[name], "arguments": args, "reason": str(reply.get("content") or f"selected {by_tool[name]}").strip()[:300],
                    "menu_revision": menu.revision}
        if name in ("stop", "case_stop"):
            return {"choice": "stop", "arguments": args, "reason": str(reply.get("content") or args.get("reason") or "stop").strip()[:300],
                    "menu_revision": menu.revision}
        return {"choice": name, "arguments": args, "reason": str(reply.get("content") or "").strip()[:300], "menu_revision": menu.revision}
    text = str(reply.get("content") or "").strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict) and isinstance(obj.get("choice"), str):
                return obj
        except ValueError:
            pass
    return None


def _choice_tools(menu: Menu) -> list[dict[str, Any]]:
    ids = [o.id for o in menu.options]
    return [{
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


def _spent(client: Any) -> float:
    try:
        return float(getattr(client, "spent_eur", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _summarize_observations(state: Any, observations: list[Observation], fallback_text: str = "") -> str:
    if state.said:
        return "\n\n".join(state.said)
    if not observations:
        return fallback_text or "No action was taken."
    last = observations[-1]
    if last.status == "stopped":
        return str(last.result.get("message") or last.reason or "Stopped without running any tool.")
    if last.status == "blocked":
        return f"Action blocked ({last.blocked_reason or 'not permitted'}): {last.reason or 'no change was made.'}"
    if last.status == "failed":
        return fallback_text or f"Could not complete the requested step ({last.blocked_reason or 'invalid choice'})."
    parts = []
    for obs in observations:
        if obs.choice == "inspect":
            suspects = obs.result.get("suspects") or []
            if suspects:
                cols = ", ".join(str(s.get("column")) for s in suspects[:6])
                parts.append(f"Inspection flagged possible leakage columns: {cols}.")
            else:
                parts.append("Completed read-only table inspection.")
        elif obs.choice == "consult":
            if obs.result.get("insufficient_evidence"):
                parts.append("Advisor reported insufficient evidence (provisional).")
            else:
                parts.append(f"Advisor ({obs.result.get('specialist')}) returned {len(obs.result.get('findings') or [])} provisional finding(s).")
        elif obs.choice == "compare":
            parts.append(f"Training-fold comparison ({obs.result.get('metric')}): set A {obs.result['set_a']['score']} vs set B {obs.result['set_b']['score']}.")
        elif obs.choice == "revise":
            parts.append(f"Revised provisional plan (revision {obs.result.get('revision')}): {obs.result.get('plan')}")
    return " ".join(parts) or fallback_text or "Completed bounded investigation step."


def run_bounded_turn(state: Any, client: Any, registry: Registry, context_messages: list[dict[str, Any]], *,
                     max_decisions: int = MAX_DECISIONS) -> dict[str, Any]:
    """Run one bounded Lead turn: offer permitted menu, validate & dispatch choice, persist observation, repeat until
    waiting_for_person, stopped, blocked, budget limit, unchanged repeat, or max_decisions."""
    observations: list[Observation] = []
    ran_actions: set[tuple[str, str, str]] = set()
    messages = list(context_messages)
    fallback_text = ""

    for _ in range(max_decisions):
        menu = build_menu(state, registry)
        if client is None:
            obs = Observation(decision_id="dec_" + secrets.token_hex(5), choice="stop", tool=None, status="failed",
                              reason="model unavailable", evidence_refs=list(menu.evidence_refs),
                              available_options=[o.id for o in menu.options], menu_revision=menu.revision,
                              blocked_reason="model_unavailable")
            _persist_observation(state, obs)
            observations.append(obs)
            fallback_text = "The model is unavailable; no action was executed."
            break

        before_spend = _spent(client) + _spent(getattr(state, "specialist_client", None))
        menu_msg = {"role": "user", "content": "[DCLab decision menu]\n" + json.dumps(menu.as_prompt_dict(), ensure_ascii=False)}
        try:
            reply = client.complete([*messages, menu_msg], _choice_tools(menu), max_tokens=1200)
        except Exception as exc:  # noqa: BLE001 - explicit typed failure on transport/budget exception
            obs = Observation(decision_id="dec_" + secrets.token_hex(5), choice="stop", tool=None, status="failed",
                              reason=f"model error: {type(exc).__name__}", evidence_refs=list(menu.evidence_refs),
                              available_options=[o.id for o in menu.options], menu_revision=menu.revision,
                              blocked_reason="model_unavailable")
            _persist_observation(state, obs)
            observations.append(obs)
            fallback_text = f"Paused safely because the model could not answer ({type(exc).__name__})."
            break
        state.spent_eur += max(0.0, _spent(client) + _spent(getattr(state, "specialist_client", None)) - before_spend)

        selection = _parse_model_selection(reply or {}, menu)
        if selection is None:
            # Bounded safe fallback on invalid reply: record traceable failure and surface a safe blocker without running tools.
            obs = Observation(decision_id="dec_" + secrets.token_hex(5), choice="invalid", tool=None, status="failed",
                              reason="model reply was not a valid menu choice", evidence_refs=list(menu.evidence_refs),
                              available_options=[o.id for o in menu.options], menu_revision=menu.revision,
                              blocked_reason="invalid_selection")
            _persist_observation(state, obs)
            observations.append(obs)
            fallback_text = "Could not validate the proposed next action; please clarify what you want to check or run next."
            break

        sig_args = json.dumps(selection.get("arguments") or {}, sort_keys=True, default=str)
        repeat_sig = (str(selection.get("choice") or ""), sig_args, _fact_fingerprint(state))
        if repeat_sig in ran_actions:
            obs = Observation(decision_id="dec_" + secrets.token_hex(5), choice="stop", tool=None, status="stopped",
                              reason="stopped repeated identical choice with unchanged evidence",
                              evidence_refs=list(menu.evidence_refs), available_options=[o.id for o in menu.options],
                              menu_revision=menu.revision, result={"stopped": True, "message": "Stopped repeated unchanged choice."},
                              blocked_reason="repeated_unchanged_choice")
            _persist_observation(state, obs)
            observations.append(obs)
            break
        ran_actions.add(repeat_sig)

        before_dispatch_spend = _spent(getattr(state, "specialist_client", None))
        obs = dispatch_choice(state, menu, selection, registry)
        state.spent_eur += max(0.0, _spent(getattr(state, "specialist_client", None)) - before_dispatch_spend)
        observations.append(obs)
        messages.append({"role": "assistant", "content": json.dumps({"choice": obs.choice, "reason": obs.reason})})
        messages.append({"role": "user", "content": "[DCLab observation]\n" + json.dumps(obs.to_record(), default=str)})

        if obs.status in ("waiting_for_person", "stopped", "blocked", "failed"):
            break
        if state.spent_eur >= state.cap_eur:
            state.waiting_on_budget = True
            budget_obs = Observation(decision_id="dec_" + secrets.token_hex(5), choice="stop", tool=None, status="stopped",
                                     reason="run budget cap reached", evidence_refs=list(menu.evidence_refs),
                                     available_options=[o.id for o in menu.options], menu_revision=state_revision(state),
                                     result={"stopped": True, "message": "Stopped at the run budget cap."},
                                     blocked_reason="budget_exhausted")
            _persist_observation(state, budget_obs)
            observations.append(budget_obs)
            break

    text = _summarize_observations(state, observations, fallback_text)
    state.turns.append({"role": "assistant", "content": text})
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
