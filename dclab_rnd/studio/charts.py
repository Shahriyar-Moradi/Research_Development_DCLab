"""A project's charts, as numbers a page draws (the owner, 2026-10-27: "show plot" got bars drawn in text; "plot
accuracy" was refused although the record kept it).

``charts(records)`` reads the finished stages' records and returns chart-ready series: the feature ladder and the model
screen (cross-validation means with their fold spread, the chosen one marked), the holdout score with its interval
beside the cross-validation mean and chance, the leakage audit at a glance, and for a binary task the review budgets,
the gains and precision-recall curves and calibration. Every number is copied or derived from a record: nothing is
fitted, scored or read again, and no row of a table is kept, only counts, rates and scores. A chart whose numbers the
records do not hold is left out and named in ``missing``.
"""

from __future__ import annotations

from typing import Any, Optional

from dclab_core import model_evaluation

# The primary metrics real projects use where lower is better (forecasting's MASE, survival's integrated Brier score);
# the records do not say it, so it is kept here beside the others.
LOWER_IS_BETTER = {"mae", "rmse", "log_loss", "brier", "mape", "mase", "smape", "integrated_brier"}
DECLARED = "declared_post_outcome_or_contested"  # the leakage stage lists declared columns among its candidates too
CURVE_POINTS = 15  # with 3 decimals, a binary project's charts fit the intern's tool-result limit


def _r(x: Any, digits: int = 3) -> Optional[float]:
    return round(float(x), digits) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _thin(points: list, limit: int = CURVE_POINTS) -> list[list[float]]:
    if len(points) <= limit:
        return [[_r(a), _r(b)] for a, b in points]
    step = (len(points) - 1) / (limit - 1)
    return [[_r(points[round(i * step)][0]), _r(points[round(i * step)][1])] for i in range(limit)]


def _bars(rows: list[dict[str, Any]], title: str, metric: str, rule: str | None) -> dict[str, Any]:
    rule = rule if not rule or len(rule) <= 150 else rule[:149].rstrip() + "…"  # the stage record keeps it whole
    return {"type": "bars", "title": title, "metric": metric, "better": "lower" if metric in LOWER_IS_BETTER else "higher",
            "rows": rows, "rule": rule}


def _leakage(ev: dict[str, Any]) -> dict[str, Any]:
    kept_out = [{"column": c, "why": why} for c, why in (ev.get("identifier_exclusions") or {}).items()]
    kept_out += [{"column": c, "why": "declared written after the prediction moment"} for c in ev.get("declared_leakage_features") or []]
    out_names = {k["column"] for k in kept_out}
    suspects = [{"column": f.get("feature"), "why": "; ".join(f.get("reasons") or [])} for f in ev.get("heuristic_review_candidates") or []
                if f.get("feature") not in out_names and DECLARED not in (f.get("reasons") or [])]  # already kept out: not a suspect
    lift = _r(ev.get("apparent_lift"))
    passed = ev.get("detector_canary_passed")
    if suspects:
        verdict = "leak suspects to review"
    elif passed is False:
        verdict = "the detector's own check failed"
    elif lift is not None and lift >= 0.005:  # the columns kept out would raise the score: the leak the audit caught
        verdict = "leak caught and kept out" if kept_out else "the unsafe columns lift the score"
    elif passed is None and lift is None:  # a kind whose leakage stage runs its own checks, without the detector
        verdict = "no detector run: the kind's own checks only"
    else:
        verdict = "no leak found"
    return {"type": "facts", "title": "Leakage audit", "verdict": verdict, "kept_out": kept_out, "suspects": suspects,
            "unsafe_lift": lift, "unsafe_lift_metric": ev.get("apparent_lift_metric"), "detector_check": passed}


def charts(records: dict[str, dict[str, Any]], project_id: str | None = None) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    missing: list[str] = []
    metric = next((r.get("primary_metric") for r in records.values() if r.get("primary_metric")), None)
    leakage = records.get("leakage")
    if leakage:
        out.append({"id": "leakage", **_leakage(leakage.get("evidence") or {})})
    features = records.get("features")
    if features:
        ev = features["evidence"]
        rows = [{"label": r["recipe"], "value": _r(r["metrics"][metric]["mean"]), "spread": _r(r["metrics"][metric].get("std")),
                 "note": f"{int(r['feature_count_mean'])} features" if r.get("feature_count_mean") is not None else None,
                 "chosen": r["recipe"] == ev.get("selected_recipe")} for r in ev.get("stage_results") or [] if metric in (r.get("metrics") or {})]
        if rows:
            out.append({"id": "features", **_bars(rows, f"Feature recipes, cross-validation {metric}", metric, ev.get("selection_rule"))})
    models = records.get("models")
    if models:
        ev = models["evidence"]
        rows = [{"label": r["model"], "value": _r(r["mean"]), "spread": _r(r.get("std")), "chosen": r["model"] == ev.get("selected_model")}
                for r in ev.get("ranking") or []]
        if rows:
            out.append({"id": "models", **_bars(rows, f"Model screen, cross-validation {metric}", metric, ev.get("selection_rule"))})
    final = records.get("final")
    if not final:
        missing.append("holdout charts: the final stage has not run")
    else:
        ev = final["evidence"]
        ci = ev.get("holdout_primary_metric_ci") or {}
        hm = ev.get("holdout_metrics") or {}
        base = (ev.get("baselines") or {}).get("majority_class") or {}
        marks = [{"label": "cross-validation mean", "value": _r(ev.get("cv_selected_metric_mean"))}]
        chance = base.get(metric) if metric != "average_precision" else base.get("average_precision_random_ranking")
        if isinstance(chance, (int, float)):
            marks.append({"label": "chance" if metric in ("roc_auc", "average_precision") else "baseline", "value": _r(chance)})
        out.append({"id": "holdout", "type": "interval", "title": f"Holdout {metric}, read {ev.get('holdout_uses_in_this_project') or 1} time(s)",
                    "metric": metric, "point": _r(hm.get(metric, ci.get("point"))), "low": _r(ci.get("low")), "high": _r(ci.get("high")),
                    "method": ci.get("method"), "marks": marks, "model": ev.get("model"), "recipe": ev.get("feature_recipe")})
        try:
            evaluation = model_evaluation.from_record(ev)
        except (TypeError, ValueError, KeyError):  # a kind whose final record has another shape: the generic charts only
            evaluation = None
        if evaluation is not None and evaluation.kind == "binary":
            m = evaluation.metrics
            out.append({"id": "measures", "type": "facts", "title": f"Holdout measures at the cutoff {evaluation.cutoff}",
                        "rows": [{"label": k.replace("_", " "), "value": _r(m.get(k))} for k in
                                 ("accuracy", "precision", "recall", "f1", "average_precision", "brier") if m.get(k) is not None],
                        "baseline_accuracy": _r(base.get("accuracy")), "positive_rate": _r(m.get("positive_rate"))})
            budgets = model_evaluation.review_budgets(evaluation)
            if budgets:
                out.append({"id": "budgets", "type": "budgets", "title": "Review budgets: the highest scores checked first",
                            "positives": budgets[0]["positives"], "source": budgets[0]["source"],
                            "rows": [{k: b[k] for k in ("share", "flagged", "caught", "recall", "precision", "lift")} for b in budgets]})
            else:
                missing.append("review budgets: this record kept no scores or gains curve")
            curves = evaluation.curves
            rate = m.get("positive_rate")
            if curves.get("gains"):
                out.append({"id": "gains", "type": "lines", "title": "Cases caught against the share of rows checked", "x": "share of rows checked",
                            "y": "share of cases caught", "series": [{"name": "model", "points": _thin(curves["gains"])},
                                                                       {"name": "random pick", "points": [[0.0, 0.0], [1.0, 1.0]], "dashed": True}]})
            if curves.get("pr"):
                pr = [{"name": "model", "points": _thin(curves["pr"])}]
                if rate:
                    pr.append({"name": "random pick", "points": [[0.0, _r(rate)], [1.0, _r(rate)]], "dashed": True})
                out.append({"id": "pr", "type": "lines", "title": "Precision against recall", "x": "recall", "y": "precision", "series": pr})
            else:
                missing.append("gains and precision-recall curves: this record kept no scores")
            cal = curves.get("calibration") or []
            if cal:
                out.append({"id": "calibration", "type": "calibration", "title": "Calibration: predicted against observed",
                            "bins": [{k: (_r(b.get(k)) if k != "count" else b.get("count")) for k in ("lower", "upper", "count", "predicted", "observed")} for b in cal]})
    return {"project_id": project_id, "metric": metric, "source": "the project's stage records: nothing refitted, scored or read again",
            "charts": out, "missing": missing}


__all__ = ["charts"]
