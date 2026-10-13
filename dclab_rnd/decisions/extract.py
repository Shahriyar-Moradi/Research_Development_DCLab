"""Sample rows from the places a decision is computed.

* ``from_brief``: the problem brief (WF-01) — one row per contract field, with the rule's status and value.
* ``from_core``: dclab_core's findings and plan for a table (the solution proposal, WF-01/WF-04) — one row per column
  for the forbidden, identifier and missing-value judgments, one row per plan decision.
* ``from_stage``: an engine stage record (WF-03 to WF-09) — the column roles and risks, the leakage judgment per
  column, the recipe, the family and the optimization choice, each with the options the rule ranked.

Every row carries the evidence the rule saw, as the aggregates the rules already compute; the holdout is never read.
"""

from __future__ import annotations

from typing import Any

from . import schema
from .schema import sample

COLUMN_FEATURES = ("kind", "dtype", "missing_rate", "unique", "unique_ratio", "id_like", "name_id_like", "time_like", "text_like",
                   "target_name_like", "low_cardinality")


def _core_version() -> str:
    try:
        from dclab_core import __version__

        return f"dclab_core@{__version__}"
    except Exception:  # noqa: BLE001
        return "dclab_core"


def _studio_version(stage: str) -> str:
    from dclab_rnd import __version__

    return f"studio.{stage}@{__version__}"


def _profile_features(profile: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Per-column aggregates from a project profile (``studio.data.profile``): name → the keys worth a model's eye."""
    out = {}
    for c in (profile or {}).get("columns") or []:
        if c.get("name") is not None:
            out[str(c["name"])] = {k: c[k] for k in COLUMN_FEATURES if k in c}
    return out


# ------------------------------------------------------------------------------------------------ the brief

BRIEF_POINTS = {"task": "contract.task", "target_column": "contract.target_column", "prediction_moment": "contract.prediction_moment",
                "time_column": "contract.time_column", "label_coverage": "contract.label_coverage"}


def from_brief(brief: Any, *, project_id: str | None = None, draft_id: str | None = None, columns: list[str] | None = None,
               actor: str = "rule", user_id: str | None = None) -> list[dict[str, Any]]:
    """One row per contract field: the rule's status (stated, inferred, unknown), value, reason and quote, with the
    other fields' values and the column names as context. ``brief`` is a ``ProblemBrief`` or its dict."""
    doc = brief.model_dump(mode="json") if hasattr(brief, "model_dump") else dict(brief or {})
    names = {str(c) for c in (columns or [])}
    # context: the other fields' status and, for the fields whose value is a column or a code, the value; never the words
    context = {f: {"status": (doc.get(f) or {}).get("status", "unknown"), **({"value": (doc.get(f) or {}).get("value")}
                                                                             if f in ("task", "target_column", "time_column", "label_coverage") else {})}
               for f in ("goal", "action", "unit", "task", "target_column", "prediction_moment", "time_column", "label_coverage")}
    rows = []
    for field, point in BRIEF_POINTS.items():
        fact = doc.get(field) or {}
        quote = str(fact.get("quote") or "")
        value = fact.get("value")
        features = {"status": fact.get("status", "unknown"), "source": fact.get("source"), "quoted": bool(quote), "quote_words": len(quote.split()),
                    "value_is_column": isinstance(value, str) and value in names,
                    "context": {k: v for k, v in context.items() if k != field}, "columns": sorted(names)[:64],
                    "open_questions": [q.get("field") for q in doc.get("open_questions") or [] if isinstance(q, dict)][:12]}
        if field == "prediction_moment":  # the moment is the person's sentence: keep what it names, not what it says
            answer = ("stated" if fact.get("status") == "stated" else fact.get("status", "unknown"))
            features["columns_named"] = sorted(c for c in names if c and c in str(value or ""))[:16]
        else:
            answer = value if fact.get("status") != "unknown" else "unknown"
        rows.append(sample(point, None, features, answer, _core_version(), project_id=project_id, draft_id=draft_id,
                           partition="metadata", actor=actor, user_id=user_id))
    return rows


# ------------------------------------------------------------------------------------------------ findings and plan

def _findings_by_column(findings: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Findings whose id names a column (``F-leak-<col>``, ``F-missing-<col>``, …), by kind and column."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {"leak": {}, "name": {}, "confirm": {}, "missing": {}, "entity": {}}
    for f in findings:
        fid = str(f.get("id", ""))
        for kind in out:
            prefix = f"F-{kind}-"
            if fid.startswith(prefix) and not fid.startswith("F-leak-sum-") and not fid.startswith("F-leak-recompute-"):
                out[kind].setdefault(fid[len(prefix):], []).append(f)
    return out


def from_core(findings: Any, plan: Any, *, columns: list[str], profile: dict[str, Any] | None = None, target: str | None = None,
              project_id: str | None = None, draft_id: str | None = None, dataset: str | None = None, run_ref: str | None = None,
              actor: str = "rule", user_id: str | None = None) -> list[dict[str, Any]]:
    """Rows from dclab_core's ``DataFindings`` and ``SolutionPlan`` (objects or dicts) for a table whose column names
    are ``columns`` (``profile``: the project profile, for per-column aggregates)."""
    fdoc = findings.model_dump(mode="json") if hasattr(findings, "model_dump") else dict(findings or {})
    pdoc = plan.model_dump(mode="json") if hasattr(plan, "model_dump") else dict(plan or {})
    flist = [f for f in fdoc.get("findings") or [] if isinstance(f, dict)]
    decisions = {d["key"]: d for d in pdoc.get("decisions") or [] if isinstance(d, dict) and d.get("key")}
    by_col = _findings_by_column(flist)
    profiles = _profile_features(profile)
    version = _core_version()
    common = dict(project_id=project_id, draft_id=draft_id, dataset=dataset, run_ref=run_ref, actor=actor, user_id=user_id)
    rows: list[dict[str, Any]] = []

    forbidden = {str(f.get("column")): f for f in (decisions.get("features.forbidden") or {}).get("choice") or [] if isinstance(f, dict)}
    identifiers = {str(c) for c in (decisions.get("features.identifiers") or {}).get("choice") or [] if not isinstance(c, dict)}
    identifiers |= {str(c.get("column")) for c in (decisions.get("features.identifiers") or {}).get("choice") or [] if isinstance(c, dict)}
    task = fdoc.get("task")
    for column in columns:
        column = str(column)
        if column == target:
            continue
        base = {"target": target, "task": task, "training_rows": fdoc.get("training_rows"), **profiles.get(column, {})}
        leak, name, confirm = by_col["leak"].get(column), by_col["name"].get(column), by_col["confirm"].get(column)
        if leak:
            answer, cited = "exclude", leak
        elif name or confirm:
            answer, cited = "confirm", (name or []) + (confirm or [])
        else:
            answer, cited = "clear", []
        features = {**base, "findings": [{"id": f["id"], "severity": f.get("severity"), **(f.get("evidence") or {})} for f in cited],
                    "in_plan_forbidden": column in forbidden}
        rows.append(sample("column.forbidden", column, features, answer, version, partition="train", **common))
        ent = by_col["entity"].get(column) or []
        features = {**base, "entity": [f.get("evidence") or {} for f in ent]}
        rows.append(sample("column.identifier", column, features, "identifier" if column in identifiers else "feature", version,
                           partition="train", **common))
        missing = by_col["missing"].get(column)
        if missing:
            worst = max(missing, key=lambda f: {"info": 0, "warning": 1, "critical": 2}.get(f.get("severity"), 0))
            answer = "indicator_and_check" if worst.get("severity") in ("warning", "critical") else "impute_within_folds"
            features = {**base, **(worst.get("evidence") or {}), "severity": worst.get("severity")}
            rows.append(sample("column.missing_action", column, features, answer, version, partition="train", **common))

    cited_evidence = {f["id"]: f.get("evidence") or {} for f in flist if f.get("id")}
    for key, d in decisions.items():
        if key in ("features.forbidden", "features.identifiers", "features.to_confirm") or key not in schema.POINTS:
            continue
        choice = d.get("choice")
        answer = schema.split_kind(choice) if key == "split.strategy" else choice
        features = {"target": target, "task": task, "training_rows": fdoc.get("training_rows"), "columns": len(columns),
                    "cites": {c: cited_evidence.get(c, {}) for c in d.get("cites") or []},
                    "choice_detail": choice if key == "split.strategy" else None}
        rows.append(sample(key, None, {k: v for k, v in features.items() if v is not None}, answer, version, partition="train",
                           alternatives=[a.get("choice") for a in d.get("alternatives") or [] if isinstance(a, dict)], **common))
    return rows


# ------------------------------------------------------------------------------------------------ engine stages

def _stage_common(record: dict[str, Any], project: dict[str, Any]) -> dict[str, Any]:
    data = project.get("data") or {}
    return dict(project_id=project.get("id"), dataset=schema.dataset_ref(data.get("sha256"), data.get("filename")),
                stage=record.get("stage"), run_ref=f"{record.get('experiment_id')}@{record.get('started_at')}")


def _options(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The options the stage ranked, as the record's decision lists them (id, mean, std, features, seconds)."""
    out = []
    for o in (record.get("decision") or {}).get("options") or []:
        if isinstance(o, dict) and o.get("id") is not None:
            out.append({"id": o["id"], "mean": o.get("mean"), "std": o.get("std"), "adjusted": o.get("adjusted"),
                        "features": o.get("features"), "seconds": o.get("elapsed_seconds")})
    return out


def from_stage(record: dict[str, Any], project: dict[str, Any], *, actor: str = "rule", user_id: str | None = None) -> list[dict[str, Any]]:
    """Rows from one completed stage record (``engine._execute_stage``'s record, as stored)."""
    stage, evidence = record.get("stage"), record.get("evidence") or {}
    common = {**_stage_common(record, project), "actor": actor, "user_id": user_id}
    version = _studio_version(str(stage))
    context = {k: v for k, v in (("task_type", record.get("task_type")), ("primary_metric", record.get("primary_metric")),
                                 ("train_rows", evidence.get("train_rows"))) if v is not None}
    rows: list[dict[str, Any]] = []
    if stage == "data":
        for pr in evidence.get("feature_profiles") or []:
            column = str(pr.get("feature"))
            features = {**context, **{k: pr.get(k) for k in ("dtype", "missing_rate", "unique_count", "unique_ratio", "top_value_frequency")}}
            rows.append(sample("column.role", column, features, pr.get("role"), version, **common))
            risks = pr.get("production_risks") or []
            rows.append(sample("column.risk", column, features, sorted(risks) if risks else "none", version, **common))
        if evidence.get("holdout_split"):
            features = {**context, "holdout_split": evidence.get("holdout_split"), "cv_protocol": evidence.get("cv_protocol"),
                        "group_stats": evidence.get("group_stats"), "has_time_range": bool(evidence.get("time_range")),
                        "spatial": {k: v for k, v in (evidence.get("spatial") or {}).items() if k in ("block_km", "train_blocks", "holdout_blocks")} or None}
            kind = str(evidence.get("holdout_split"))
            answer = ("spatial" if evidence.get("spatial") else "time_group" if "group" in kind and "time" in kind else "time" if "time" in kind
                      else "group" if "group" in kind else "stratified" if "stratif" in kind else "random")
            rows.append(sample("split.strategy", None, {k: v for k, v in features.items() if v is not None}, answer, version, **common))
    elif stage == "leakage":
        declared = {str(c) for c in evidence.get("declared_leakage_features") or []}
        flagged = {str(f.get("feature")): f for f in evidence.get("heuristic_review_candidates") or [] if isinstance(f, dict)}
        signals = {str(s.get("feature")): s for s in evidence.get("top_univariate_signals") or [] if isinstance(s, dict)}
        rules = evidence.get("heuristic_rules") or {}
        per_column = (evidence.get("safe_vs_unsafe_training_cv") or {}).get("per_column") or {}
        for column in sorted(declared | set(flagged) | set(signals)):
            found = flagged.get(column) or {}
            features = {**context, "reasons": [r for r in found.get("reasons") or [] if r != "declared_post_outcome_or_contested"],
                        "signal": (signals.get(column) or {}).get("signal", found.get("signal")), "unique_ratio": found.get("unique_ratio"),
                        "strong_signal": rules.get("strong_univariate_signal"), "extreme_signal": rules.get("extreme_univariate_signal"),
                        "apparent_lift_alone": (per_column.get(column) or {}).get("apparent_lift"), "declared": column in declared}
            answer = "exclude" if column in declared else "review_flag" if column in flagged else "clear"
            rows.append(sample("column.leakage", column, {k: v for k, v in features.items() if v is not None}, answer, version, **common))
    elif stage == "features":
        options = _options(record)
        features = {**context, "options": options, "rule": evidence.get("selection_rule")}
        rows.append(sample("feature.recipe", None, features, evidence.get("selected_recipe"), version,
                           alternatives=[o["id"] for o in options], **common))
    elif stage == "models":
        options = _options(record) or [{"id": o.get("model"), "mean": o.get("mean"), "std": o.get("std")} for o in evidence.get("ranking") or []]
        features = {**context, "recipe": evidence.get("feature_recipe"), "options": options, "rule": evidence.get("selection_rule"),
                    "rare_event": bool(evidence.get("rare_event_choice"))}
        rows.append(sample("model.family", None, features, evidence.get("selected_model"), version,
                           alternatives=[o["id"] for o in options], **common))
    elif stage == "final":
        decision = evidence.get("tuning_decision") or {}
        metric = record.get("primary_metric")
        configs = []
        for c in evidence.get("configuration_results") or []:
            stats = ((c.get("metrics") or {}).get(metric) or {}) if isinstance(c, dict) else {}
            configs.append({"id": c.get("config_id"), "mean": stats.get("mean"), "std": stats.get("std")})
        features = {**context, "recipe": evidence.get("feature_recipe"), "model": evidence.get("model"), "options": configs,
                    "gain": decision.get("gain"), "required_gain": decision.get("required_gain"), "rule": evidence.get("selection_rule")}
        rows.append(sample("optimization.config", None, {k: v for k, v in features.items() if v is not None}, evidence.get("selected_optimization"),
                           version, alternatives=[c["id"] for c in configs], **common))
    return rows


def from_records(records: dict[str, dict[str, Any]], project: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows from every completed stage of a project (``store.records``), in stage order."""
    rows = []
    for stage in ("data", "leakage", "features", "models", "final"):
        record = records.get(stage)
        if record and record.get("status") == "completed":
            rows.extend(from_stage(record, project))
    return rows
