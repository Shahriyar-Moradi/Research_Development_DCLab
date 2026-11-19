"""Leakage and Prediction-Moment Advisor for stage_leakage (Unified Agentic Roadmap 2026-11-19).

Reviews all non-identifier columns plus statistical heuristic suspects using ONLY column names, types,
summary statistics, heuristic findings, and the active problem's prediction moment (``cell_values=0``;
never raw table rows).

Design guarantees:
- Emits visible provisional recommendations (``block``, ``keep``, ``ask``) with rationale, uncertainty,
  evidence references, and pinned ``prediction_moment`` dependency hash / problem revision.
- Whenever a column is identified as very likely (mostly) leakage, explicitly tells the user and asks
  whether they want to:
    1. (Recommended) Build and compare BOTH without the suspected leakage column and with it on
       training folds (clean safe candidate vs. leaky shadow contrast; holdout untouched), or
    2. Exclude the suspected column directly.
- Never mutates ``solution.forbidden`` automatically; explicit user confirmation is required and is
  refused automatically if the active prediction moment changed (stale advice).
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

from dclab_core.leakage_contrast import build_leakage_contrast

from . import leakage_review

POST_EVENT_HINT = re.compile(
    r"(?:delivered_at|delivery_time|courier_rating|refund_settled|chargeback|resolution_time|"
    r"closed_at|cancelled_at|cancel_reason|actual_arrival|post_outcome|after_|settlement_date|"
    r"churn_date|default_date|final_status|outcome_flag|label_date|payment_settled)",
    re.I,
)


class StaleLeakageAdviceError(ValueError):
    """Raised when confirming a leakage recommendation whose prediction moment changed."""

    def __init__(self, message: str, *, old_moment: str, current_moment: str):
        self.old_moment = old_moment
        self.current_moment = current_moment
        super().__init__(message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def moment_fingerprint(moment: Optional[str]) -> str:
    norm = re.sub(r"\s+", " ", str(moment or "")).strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]


def build_likely_leakage_prompt(columns: list[str], moment: str) -> Optional[dict[str, Any]]:
    """Build the required owner UX prompt whenever features are identified as very likely / mostly leakage."""
    cols = [str(c) for c in columns if c]
    if not cols:
        return None
    joined = ", ".join(cols)
    plural = len(cols) > 1
    subject = f"These features ({joined}) are" if plural else f"Feature '{cols[0]}' is"
    moment_note = f" at the prediction moment ({moment})" if moment else ""
    message = (
        f"{subject} mostly / very likely leakage{moment_note}. "
        f"Would you like DCLab to build and compare BOTH models on training folds (clean without {joined} vs. "
        f"shadow with {joined}) so you can inspect the exact leakage lift while keeping the clean model safe, "
        f"or exclude {'them' if plural else 'it'} directly?"
    )
    return {
        "columns": cols,
        "message": message,
        "recommended_choice": "compare_both",
        "options": [
            {
                "id": "compare_both",
                "label": "Compare both models: without leakage vs. with leakage (Recommended)",
                "recommended": True,
                "description": (
                    f"Run a training-fold side-by-side comparison of the clean model (excluding {joined}) against a "
                    f"shadow model (including {joined}) to measure the exact apparent leakage lift while keeping the "
                    f"clean pipeline as the safe candidate and leaving holdout untouched."
                ),
            },
            {
                "id": "exclude_directly",
                "label": f"Exclude {joined} directly",
                "recommended": False,
                "description": (
                    f"Add {joined} directly to forbidden features without running the leaky shadow comparison."
                ),
            },
        ],
    }


def _metadata_profile(columns: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Ensure strictly zero cell values enter the reviewer: only name, kind, missing_rate, unique."""
    clean_cols = []
    for c in columns:
        if not isinstance(c, Mapping) or not c.get("name"):
            continue
        clean_cols.append({
            "name": str(c["name"]),
            "kind": str(c.get("kind") or "unknown"),
            "missing_rate": round(float(c.get("missing_rate") or 0.0), 4),
            "unique": int(c["unique"]) if c.get("unique") is not None else None,
        })
    return {"columns": clean_cols}


def advise(
    *,
    columns_meta: Iterable[Mapping[str, Any]],
    prediction_moment: Optional[str],
    target: Optional[str],
    identifiers: Iterable[str] = (),
    time_column: Optional[str] = None,
    declared_forbidden: Iterable[str] = (),
    heuristic_candidates: Iterable[Mapping[str, Any]] = (),
    univariate_ranking: Iterable[Mapping[str, Any]] = (),
    client: Any = None,
    problem_revision: Optional[int] = None,
) -> dict[str, Any]:
    """Scan all non-identifier columns and heuristic suspects against the active prediction moment (cell_values=0)."""
    profile = _metadata_profile(columns_meta)
    id_set = {str(c) for c in identifiers if c}
    declared_set = {str(c) for c in declared_forbidden if c}
    moment_str = str(prediction_moment or "").strip()

    heuristic_by_col = {
        str(f.get("feature")): dict(f)
        for f in heuristic_candidates
        if isinstance(f, Mapping) and f.get("feature")
    }
    signal_by_col = {
        str(r.get("feature")): float(r.get("signal") or 0.0)
        for r in univariate_ranking
        if isinstance(r, Mapping) and r.get("feature")
    }

    flagged_input = [
        {"column": col, "reason": ", ".join(heuristic_by_col[col].get("reasons") or ["declared forbidden"]),
         "proof": ["DCLAB-R05"]}
        for col in sorted(declared_set) if col in heuristic_by_col
    ]
    rev_out = leakage_review.review(
        profile, moment_str, flagged_input, target=target, identifiers=sorted(id_set),
        client=client, time_column=time_column,
    )
    review_by_col = {str(i["column"]): i for i in rev_out.get("items") or [] if isinstance(i, Mapping) and i.get("column")}
    disagreements_by_col = {str(d["column"]): d for d in rev_out.get("disagreements") or [] if isinstance(d, Mapping) and d.get("column")}

    recommendations: list[dict[str, Any]] = []
    likely_leakage_cols: list[str] = []
    ask_cols: list[str] = []

    skip = id_set | ({str(target)} if target else set()) | ({str(time_column)} if time_column else set())
    scanned_names = [c["name"] for c in profile["columns"] if c["name"] not in skip]
    for col in heuristic_by_col:
        if col not in skip and col not in scanned_names:
            scanned_names.append(col)

    for col in scanned_names:
        heur = heuristic_by_col.get(col)
        heur_reasons = list((heur or {}).get("reasons") or [])
        non_decl_reasons = [r for r in heur_reasons if r != "declared_post_outcome_or_contested"]
        rev_item = review_by_col.get(col)
        signal = signal_by_col.get(col)
        already_declared = col in declared_set

        evidence_refs: list[str] = []
        if problem_revision is not None:
            evidence_refs.append(f"problem:rev:{problem_revision}")
        if moment_str:
            evidence_refs.append("DCLAB-R01")

        # Determine provisional verdict: block / ask / keep.
        name_post_event = bool(POST_EVENT_HINT.search(col))
        extreme_sig = any("extreme" in r or "near_duplicate" in r or "duplicate_of_target" in r for r in non_decl_reasons)
        strong_sig = any("strong" in r or "name" in r for r in non_decl_reasons)

        if already_declared:
            action = "block"
            likelihood = "very_likely"
            confidence = "high"
            rationale = f"Already declared forbidden under the active solution; excluded from clean training features."
            uncertainty = "Low (confirmed by current solution contract)."
            evidence_refs.append("solution:forbidden")
        # block only on the audit's own verdict (apply True) or an extreme or duplicate heuristic; the moment rule and the model
        # are shown, never ticked (leakage_review's apply False), and a name alone is a question (the review, 2026-11-19:
        # `couriers_online` blocked for sharing a stem with the moment; a count of past events blocked by its name)
        elif (rev_item is not None and rev_item.get("verdict") == "after the moment" and rev_item.get("apply") is True) or extreme_sig:
            action = "block"
            likelihood = "very_likely"
            confidence = "high"
            reasons_parts = []
            if rev_item is not None:
                reasons_parts.append(str(rev_item.get("reason") or "recorded after the prediction moment"))
                evidence_refs.extend(str(r) for r in (rev_item.get("records") or []))
            if non_decl_reasons:
                reasons_parts.append(f"statistical heuristic flags: {', '.join(non_decl_reasons)}")
                evidence_refs.extend(f"heuristic:{r}" for r in non_decl_reasons)
            if name_post_event and not reasons_parts:
                reasons_parts.append(f"column name '{col}' indicates post-event or post-outcome information relative to the prediction moment")
                evidence_refs.append("heuristic:post_event_name")
            rationale = (
                f"Feature '{col}' is mostly / very likely leakage: {'; '.join(reasons_parts)}."
            )
            uncertainty = "Provisional advice; confirm availability at the prediction moment or compare clean vs. leaky on training folds."
            likely_leakage_cols.append(col)
        elif strong_sig or name_post_event or (col in disagreements_by_col) or (rev_item is not None and rev_item.get("verdict") == "after the moment"):
            action = "ask"
            likelihood = "possible"
            confidence = "medium"
            rationale = (
                f"Feature '{col}' shows heuristic warning ({', '.join(non_decl_reasons) or 'timing ambiguity'})"
                + (f" with univariate signal {signal:.3f}" if signal is not None else "")
                + "; verify whether it is recorded strictly before the prediction moment."
            )
            uncertainty = "Statistical signal alone cannot distinguish genuine predictive drivers from post-moment leakage."
            evidence_refs.extend(f"heuristic:{r}" for r in non_decl_reasons)
            ask_cols.append(col)
        else:
            action = "keep"
            likelihood = "unlikely"
            confidence = "medium" if moment_str else "low"
            rationale = (
                f"No timing contradiction or statistical leakage alert found for '{col}'"
                + (f" against prediction moment '{moment_str[:120]}'." if moment_str else " (write the prediction moment for stronger checks).")
            )
            uncertainty = "Confirm domain timing if upstream pipelines backfill this field."

        col_prompt = build_likely_leakage_prompt([col], moment_str) if (action == "block" and not already_declared) else None
        recommendations.append({
            "column": col,
            "action": action,
            "likelihood": likelihood,
            "confidence": confidence,
            "already_declared": already_declared,
            "provisional": True,
            "carries_approval": False,
            "rationale": rationale,
            "uncertainty": uncertainty,
            "signal": round(signal, 4) if signal is not None else None,
            "heuristic_reasons": non_decl_reasons,
            "evidence_refs": sorted(set(evidence_refs)),
            "user_prompt": col_prompt,
        })

    top_prompt = build_likely_leakage_prompt(likely_leakage_cols, moment_str)
    return {
        "schema_version": "dclab.leakage_advisor.v1",
        "provisional": True,
        "carries_approval": False,
        "requires_confirmation": True,
        "cell_values_sent": 0,
        "mode": rev_out.get("mode", "rules"),
        "scanned_columns": len(scanned_names),
        "likely_leakage_columns": likely_leakage_cols,
        "ask_columns": ask_cols,
        "recommendations": recommendations,
        "likely_leakage_prompt": top_prompt,
        "disagreements": list(rev_out.get("disagreements") or []),
        "depends_on": {
            "prediction_moment": moment_str,
            "prediction_moment_hash": moment_fingerprint(moment_str),
            "problem_revision": problem_revision,
        },
        "stale": False,
        "created_at": _now(),
    }


def check_advisor_freshness(
    advisory: Mapping[str, Any],
    *,
    current_moment: Optional[str],
    current_problem_revision: Optional[int] = None,
) -> dict[str, Any]:
    """Invalidate leakage advisor output automatically whenever the active prediction moment changes."""
    out = dict(advisory)
    deps = dict(out.get("depends_on") or {})
    old_moment = str(deps.get("prediction_moment") or "")
    old_hash = str(deps.get("prediction_moment_hash") or moment_fingerprint(old_moment))
    new_moment = str(current_moment or "").strip()
    new_hash = moment_fingerprint(new_moment)
    old_rev = deps.get("problem_revision")

    changed_sources: list[dict[str, Any]] = []
    if old_hash != new_hash:
        changed_sources.append({
            "field": "prediction_moment",
            "old_value": old_moment,
            "new_value": new_moment,
            "reason": "active prediction moment changed; leakage recommendations must be re-run",
        })
    if old_rev is not None and current_problem_revision is not None and int(current_problem_revision) > int(old_rev):
        changed_sources.append({
            "field": "problem_revision",
            "from_revision": int(old_rev),
            "to_revision": int(current_problem_revision),
            "reason": "problem base updated since leakage advisor ran",
        })

    stale = bool(changed_sources)
    out["stale"] = stale
    out["may_confirm"] = not stale
    out["changed_sources"] = changed_sources
    if stale:
        out["stale_reason"] = "Prediction moment or problem facts changed since this leakage review ran."
    return out


def compare_with_and_without_suspects(p: Any, suspect_columns: Iterable[str]) -> dict[str, Any]:
    """Run the recommended side-by-side training-fold comparison: clean (without suspects) vs. leaky shadow (with suspects).

    Strictly uses training folds (``bundle.X_train``, ``bundle.cv_splits``); holdout is never read.
    """
    from dataclasses import replace

    from dclab_rnd.expansion import runner as rn

    bundle, spec, task = p.bundle, p.bundle.spec, p.bundle.task_type
    metric, higher = spec.primary_metric, spec.higher_is_better
    suspects = [str(c) for c in suspect_columns if str(c) in bundle.X_train.columns]
    if not suspects:
        return {"error": "no valid suspect columns available for training-fold comparison", "holdout_touched": False}

    audit_model = rn._fallback_family("lightgbm") if task != "text_tabular_binary" else "logistic_regression"
    recipe = rn.LEAKAGE_RECIPE[spec.key]

    # Build a temporary clean bundle excluding suspects (plus already declared blocked features) WITHOUT touching holdout.
    orig_blocked = dict(spec.blocked_features)
    extra_blocked = {**orig_blocked, **{c: "suspected leakage under side-by-side training-CV comparison" for c in suspects}}
    clean_spec = replace(spec, blocked_features=extra_blocked)
    clean_bundle = replace(bundle, spec=clean_spec)

    clean_cv = rn.cross_validate(clean_bundle, recipe, audit_model, config_id="advisor_clean_without_suspects")
    leaky_cv = rn.cross_validate(clean_bundle, recipe, audit_model, include_blocked=suspects, config_id="advisor_leaky_with_suspects")
    clean_mean = float(rn._mean(clean_cv, metric))
    leaky_mean = float(rn._mean(leaky_cv, metric))
    lift = float(rn.oriented_gain(leaky_mean, clean_mean, higher))

    per_lifts: dict[str, float] = {}
    if 1 < len(suspects) <= 6:
        for col in suspects:
            single = rn.cross_validate(clean_bundle, recipe, audit_model, include_blocked=[col], config_id=f"advisor_leaky_{col}")
            per_lifts[col] = float(rn.oriented_gain(rn._mean(single, metric), clean_mean, higher))

    contrast = build_leakage_contrast(
        project=spec.key,
        target=spec.target,
        leakage_features=suspects,
        metric=metric,
        higher_is_better=higher,
        clean_cv_metric=clean_mean,
        leakage_cv_metric=leaky_mean,
        per_feature_cv_lift=per_lifts,
        decision_time_contract=spec.decision_time_contract,
    )
    def _public_cv(cv_doc: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in cv_doc.items() if not str(k).startswith("_")}

    return {
        "scope": "training_folds_only",
        "holdout_touched": False,
        "recommended_safe_candidate": "without_leakage",
        "suspect_columns": suspects,
        "model": audit_model,
        "recipe": recipe,
        "metric": metric,
        "higher_is_better": higher,
        "without_leakage_cv": _public_cv(clean_cv),
        "with_leakage_cv": _public_cv(leaky_cv),
        "clean_cv_mean": clean_mean,
        "leaky_cv_mean": leaky_mean,
        "apparent_lift": lift,
        "per_column_lift": per_lifts,
        "leakage_contrast": contrast.model_dump(),
    }


def apply_leakage_decision(
    store: Any,
    project_id: str,
    advisory: Mapping[str, Any],
    *,
    choice: str,
    columns: Iterable[str],
    current_moment: Optional[str] = None,
    current_problem_revision: Optional[int] = None,
    prepared: Any = None,
    by: str = "person:owner",
    confirm_exclude_after_compare: bool = False,
) -> dict[str, Any]:
    """Apply an explicit user confirmation to a leakage recommendation after verifying freshness.

    Choices:
    - ``compare_both`` (Recommended): runs training-fold clean vs. leaky comparison; only updates ``solution.forbidden``
      when ``confirm_exclude_after_compare=True``.
    - ``exclude_directly``: explicitly adds confirmed suspect columns to ``solution.forbidden`` and logs a studio note.
    - ``keep_confirmed``: records user confirmation that the column is genuinely available at the prediction moment.
    """
    if choice not in ("compare_both", "exclude_directly", "keep_confirmed"):
        raise ValueError(f"unknown choice {choice!r}; expected 'compare_both', 'exclude_directly', or 'keep_confirmed'")

    project = store.get(project_id)
    sol = dict(project.get("solution") or {})
    live_moment = current_moment if current_moment is not None else sol.get("prediction_moment")
    checked = check_advisor_freshness(advisory, current_moment=live_moment, current_problem_revision=current_problem_revision)
    if checked["stale"]:
        old_m = str((advisory.get("depends_on") or {}).get("prediction_moment") or "")
        raise StaleLeakageAdviceError(
            "Leakage advice is stale because the active prediction moment changed; re-run stage_leakage before confirming.",
            old_moment=old_m,
            current_moment=str(live_moment or ""),
        )

    cols = [str(c) for c in columns if c]
    if not cols:
        raise ValueError("at least one column must be named")

    out: dict[str, Any] = {
        "choice": choice,
        "columns": cols,
        "confirmed_by": by,
        "solution_mutated": False,
        "holdout_touched": False,
    }

    if choice == "compare_both":
        if prepared is None:
            from . import engine as studio_engine

            prepared = studio_engine.prepare(store, project)
        out["comparison"] = compare_with_and_without_suspects(prepared, cols)
        if not confirm_exclude_after_compare:
            out["next_step"] = (
                "Review the clean vs. leaky training-CV contrast above, then confirm exclusion to add these "
                "columns to solution.forbidden (the clean model stays safe and holdout remains untouched)."
            )
            return out

    if choice in ("exclude_directly", "compare_both"):
        before = dict(sol)
        forbidden = list(sol.get("forbidden") or [])
        existing_cols = {str(f.get("column")) for f in forbidden if isinstance(f, Mapping)}
        rec_map = {str(r["column"]): r for r in (advisory.get("recommendations") or []) if isinstance(r, Mapping) and r.get("column")}
        added = []
        unknown = [c for c in cols if c not in rec_map]
        if unknown:  # only a column the advisor reviewed: never the outcome, a missing column or one it never saw
            raise ValueError(f"not reviewed by the leakage advisor: {', '.join(unknown[:5])}")
        for col in cols:
            if col not in existing_cols:
                reason = str((rec_map.get(col) or {}).get("rationale") or "Confirmed leakage exclusion by user.")[:240]
                # the Solution's own shape only (a column and its reason): an extra key made every later stage fail
                forbidden.append({"column": col, "reason": f"{reason} (confirmed by {by})"[:300]})
                added.append(col)
        out["added_forbidden"] = added
        if added:
            # proposed, not saved here: the caller saves it through the person's own path (the solution's checks, the graph's
            # validator and log, the audit, the ledger, and the stages it invalidates cleared), never beside it (the review)
            out["proposed_solution"] = {**sol, "forbidden": forbidden}
        _ = before
        return out

    # choice == "keep_confirmed"
    from . import memory as studio_memory

    studio_memory.add(
        project, "decision",
        f"Confirmed available at prediction moment: {', '.join(cols)}",
        f"Prediction moment: {live_moment or 'not written'}",
        by, "confirm_leakage_keep", "leakage",
    )
    store.save(project)
    out["confirmed_keep"] = cols
    return out


def advise_stage(p: Any, findings: list[dict[str, Any]], ranking: list[dict[str, Any]], declared: list[str], *,
                 client: Any = None, problem_revision: Optional[int] = None) -> dict[str, Any]:
    """Run the Leakage & Prediction-Moment Advisor inside stage_leakage with zero raw cell values."""
    X_tr = p.bundle.X_train
    prof_cols = [
        {"name": str(c), "kind": ("numeric" if str(X_tr[c].dtype).startswith(("int", "float", "bool")) else "categorical"),
         "missing_rate": float(X_tr[c].isna().mean()), "unique": int(X_tr[c].nunique(dropna=True))}
        for c in X_tr.columns
    ]
    sol = getattr(p, "solution", None)
    sol_moment = sol.get("prediction_moment") if isinstance(sol, Mapping) else getattr(sol, "prediction_moment", None)
    sol_time = sol.get("time_column") if isinstance(sol, Mapping) else getattr(sol, "time_column", None)
    moment = sol_moment or p.bundle.spec.decision_time_contract
    identifiers = [c for c, role in p.roles.items() if role == "identifier"]
    time_col = sol_time or p.bundle.spec.time_column
    return advise(
        columns_meta=prof_cols,
        prediction_moment=moment,
        target=p.bundle.spec.target,
        identifiers=identifiers,
        time_column=time_col,
        declared_forbidden=declared,
        heuristic_candidates=findings,
        univariate_ranking=ranking,
        client=client,
        problem_revision=problem_revision,
    )


__all__ = [
    "StaleLeakageAdviceError",
    "advise",
    "advise_stage",
    "apply_leakage_decision",
    "build_likely_leakage_prompt",
    "check_advisor_freshness",
    "compare_with_and_without_suspects",
    "moment_fingerprint",
]
